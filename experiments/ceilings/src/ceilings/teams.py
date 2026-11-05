"""Arm design 3: agent teams that work the way deployed agent systems do (prereg/DESIGN3_PLAN.md).

Every agent runs an agentic loop. In each turn it thinks (the adapter's reasoning stage at the
arm's depth) and then takes one action, a JSON object `{"tool": ..., "args": ...}`; the harness
carries the action out and shows the result at the start of the agent's next turn. The arms
differ only in topology:

- S: one agent alone (the reference);
- L: a lead that hands parts of the work to 2-3 workers (Claude Code's subagent pattern); a
  worker starts with the task and its brief, and the lead sees only its report;
- W: 4 peers sharing a notes board, answering by majority vote;
- W0: W without the board (4 independent attempts and a vote);
- R: an author and a reviewer (code review);
- S_ref: S raised until it spends what a team spends (ceilings.team_match), through deeper
  reasoning per turn and then several independent attempts and a vote, as design 2's K and
  A2_single_ref were (the samples of one agent are its compute, as in every arm's baseline).

Every agent's first prompt holds every fact of the task: a brief is extra instruction, never a
substitute for the facts. The team shares one token budget (ledger total tokens, checked before
each turn), and no run ends without an answer: an agent whose turns or budget run out is asked
for its answer (or report, or review) at once, outside the budget, and the run records it as
forced.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from ceilings.agents import Policy, with_user
from ceilings.ledger import Ledger
from ceilings.orchestrator import TeamOutcome, aggregate
from ceilings.records import TaskInstance

TEAM_ARMS = ("S", "L", "W", "R", "W0", "S_ref")
NAMES = ["A", "B", "C", "D"]

TOOLS = {
    "note": ('{"tool": "note", "args": {"text": "..."}}',
             "write down working or partial results for yourself; you will see them in your "
             "later turns"),
    "submit": ('{"tool": "submit", "args": {}}',
               "you are ready to answer; you will be asked for your final answer next"),
    "delegate": ('{"tool": "delegate", "args": {"briefs": ["...", "..."]}}',
                 "start one worker per brief; each worker sees the task and its brief, works on "
                 "its own, and sends you a report"),
    "report": ('{"tool": "report", "args": {"text": "..."}}',
               "send the lead your report; this ends your work"),
    "append_note": ('{"tool": "append_note", "args": {"text": "..."}}',
                    "add a note to the team's shared notes, which every teammate sees from the "
                    "next round"),
    "request_review": ('{"tool": "request_review", "args": {"solution": "..."}}',
                       "send the reviewer your worked solution (once)"),
    "review": ('{"tool": "review", "args": {"text": "..."}}',
               "send the author your review; this ends your work"),
}
TEXT_ARG = {"note": "text", "report": "text", "append_note": "text", "review": "text",
            "request_review": "solution"}


@dataclass(frozen=True)
class TeamConfig:
    arm_id: str
    budget: int = 0  # team cap on ledger total tokens, checked before each turn; 0 means no cap
    max_turns: int = 6  # per agent
    depth: str = "medium"  # every turn's reasoning stage
    # Every turn's action reply. 1,200 after the pilot: at 600, MiMo's long notes were cut off
    # mid-JSON, rejected and retried, and one solo run grew to 101,619 tokens.
    reply_tokens: int = 1200
    report_tokens: int = 300  # briefs, reports, notes, solutions and reviews
    min_workers: int = 2  # L: reports needed before the lead may submit
    max_workers: int = 3  # L: workers over the whole run
    peers: int = 4  # W and W0
    samples: int = 1  # S and S_ref: independent attempts by the one agent, majority vote

    def to_dict(self) -> dict:
        return asdict(self)


def team_config(arm_id: str, budget: int = 0, **kw) -> TeamConfig:
    if arm_id not in TEAM_ARMS:
        raise ValueError(f"unknown team arm {arm_id!r}; expected one of {TEAM_ARMS}")
    cfg = TeamConfig(arm_id=arm_id, budget=budget, **kw)
    if cfg.samples != 1 and arm_id not in ("S", "S_ref"):
        raise ValueError("only S and S_ref take several attempts")
    return cfg


def ref_config(strength: str, budget: int = 0, **kw) -> TeamConfig:
    """S_ref at a strength from team_match.LADDER: 'max' (one attempt) or 'max:3' (three)."""
    depth, _, n = strength.partition(":")
    return team_config("S_ref", budget, depth=depth, samples=int(n or 1), **kw)


@dataclass
class TeamAgent:
    """One agent's conversation and state. `public` is the structured mirror of the facts that
    only SimPolicy reads (never the answer)."""

    name: str
    role: str
    tools: tuple[str, ...]
    messages: list[dict]
    public: dict
    answer_space: list[str]
    scenario: str
    turns: int = 0
    pending: str = ""  # the result of the last action, shown at the start of the next turn
    inbox: list[str] = field(default_factory=list)  # reports, notes and reviews received
    notes: list[str] = field(default_factory=list)  # what this agent wrote down
    seen_notes: int = 0  # W: shared notes already shown to this agent
    depth: str = "medium"
    last_reasoning: str = ""  # the reasoning stage of the agent's latest turn
    done: bool = False


def parse_action(text: str, tools: tuple[str, ...]) -> tuple[dict | None, str | None]:
    """The first JSON object in a reply, checked against the agent's tools."""
    t = (text or "").strip()
    start = t.find("{")
    if start < 0:
        return None, "no JSON object"
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[start:])
    except json.JSONDecodeError as e:
        return None, f"not valid JSON ({e.msg})"
    if not isinstance(obj, dict) or obj.get("tool") not in tools:
        return None, f"the tool must be one of {', '.join(tools)}"
    args = obj.get("args", {})
    if not isinstance(args, dict):
        return None, "args must be an object"
    tool = obj["tool"]
    if tool in TEXT_ARG:
        value = args.get(TEXT_ARG[tool])
        if not isinstance(value, str) or not value.strip():
            return None, f"{tool} needs a non-empty \"{TEXT_ARG[tool]}\""
        return {"tool": tool, "args": {TEXT_ARG[tool]: value.strip()}}, None
    if tool == "delegate":
        briefs = args.get("briefs")
        if (not isinstance(briefs, list) or not briefs
                or not all(isinstance(b, str) and b.strip() for b in briefs)):
            return None, "delegate needs \"briefs\", a list of non-empty strings"
        return {"tool": tool, "args": {"briefs": [b.strip() for b in briefs]}}, None
    return {"tool": tool, "args": {}}, None


class _Team:
    def __init__(self, inst: TaskInstance, cfg: TeamConfig, policy: Policy, seed: int) -> None:
        self.inst, self.cfg, self.policy = inst, cfg, policy
        self.rng = np.random.default_rng(seed)
        self.ledger = Ledger()
        self.transcript: list[dict] = []
        self.request_ids: list[str] = []
        self.forced: dict[str, int] = {"answer": 0, "report": 0, "review": 0}
        self.invalid = 0
        self.refused = 0

    # --- accounting -------------------------------------------------------------------------

    def seed(self) -> int:
        return int(self.rng.integers(2**31))

    def charge(self, agent: str, usage, log: dict | None) -> None:
        self.ledger.add_call(agent, usage.input_tokens, usage.output_tokens,
                             usage.reasoning_tokens, usage.dollars)
        if log and log.get("request_id"):
            self.request_ids.append(log["request_id"])

    def budget_left(self) -> int | None:
        return None if not self.cfg.budget else self.cfg.budget - self.ledger.total_tokens

    def out_of_budget(self) -> bool:
        left = self.budget_left()
        return left is not None and left <= 0

    def cap(self, text: str) -> tuple[str, bool]:
        """`text` cut to whole words within the report cap."""
        count = self.policy.count_tokens
        if count(text) <= self.cfg.report_tokens:
            return text, False
        words = text.split()
        while words and count(" ".join(words)) > self.cfg.report_tokens:
            words = words[: max(0, len(words) - max(1, len(words) // 20))]
        return " ".join(words), True

    def deliver(self, kind: str, sender: str, receivers: list[str], text: str,
                forced: bool = False) -> str:
        text, truncated = self.cap(text)
        tokens = self.policy.count_tokens(text) if text else 0
        self.ledger.add_message(tokens)
        self.transcript.append({"step": "deliver", "kind": kind, "from": sender,
                                "to": receivers, "text": text, "tokens": tokens,
                                "cap": self.cfg.report_tokens, "truncated": truncated,
                                "forced": forced})
        return text

    # --- agents -----------------------------------------------------------------------------

    def start(self, name: str, role: str, tools: tuple[str, ...], role_text: str) -> TeamAgent:
        cfg = self.cfg
        lines = "\n".join(f"- {TOOLS[t][0]}: {TOOLS[t][1]}" for t in tools)
        ends = "your report" if role == "worker" else "your review" if role == "reviewer" \
            else "your final answer"
        rules = (f"You work in turns. In each turn, reply with exactly one action, a JSON object, "
                 f"and nothing else. Your actions:\n{lines}\nYou have at most {cfg.max_turns} "
                 f"turns. Keep each action short: a reply that is cut off is not valid JSON and "
                 f"is rejected. Text you send to others (briefs, reports, notes, solutions, "
                 f"reviews) is cut to {cfg.report_tokens} tokens.")
        if cfg.budget:
            rules += (f" The team shares a budget of {cfg.budget} tokens: everything any "
                      "teammate reads and writes counts against it, and each turn shows what is "
                      "left.")
        rules += f" If your turns or the budget run out, you will be asked for {ends} at once."
        user = (f"Task: {self.inst.scenario}\n\n{self.inst.shared_evidence}\n\n{role_text}\n\n"
                f"{rules}")
        messages = [{"role": "system",
                     "content": f"You are agent {name}, one member of a small team working on "
                                "a task. Use only the information you are shown in this "
                                "conversation."},
                    {"role": "user", "content": user}]
        agent = TeamAgent(name, role, tools, messages, self.inst.payload.get("public", {}),
                          self.inst.answer_space, self.inst.scenario, depth=cfg.depth)
        self.transcript.append({"step": "start", "agent": name, "role": role,
                                "tools": list(tools), "prompt": [dict(m) for m in messages]})
        return agent

    def turn(self, agent: TeamAgent, extra: str = "") -> dict | None:
        """One turn: show the last result, then get one action. None when the agent's turns or
        the team's budget are used up (the caller then forces its answer, report or review)."""
        if agent.turns >= self.cfg.max_turns or self.out_of_budget():
            return None
        if agent.turns:
            status = f"Turns left: {self.cfg.max_turns - agent.turns}."
            if self.cfg.budget:
                status += f" Team budget left: {self.budget_left()} tokens."
            agent.messages = with_user(
                agent.messages, "\n\n".join(p for p in (agent.pending, extra, status) if p))
        agent.pending = ""
        left = self.budget_left()
        calls, retried = [], False
        act = self.policy.act(agent, self.cfg.depth, self.cfg.reply_tokens, self.seed())
        self.charge(agent.name, act.usage, act.log)
        calls.append(act.log)
        action, error = parse_action(act.value, agent.tools)
        agent.messages = agent.messages + [{"role": "assistant", "content": act.value or ""}]
        if action is None:
            retried = True
            agent.messages = with_user(agent.messages, f"Rejected: {error}. Reply with exactly "
                                                         "one action, a JSON object.")
            act = self.policy.act(agent, self.cfg.depth, self.cfg.reply_tokens, self.seed())
            self.charge(agent.name, act.usage, act.log)
            calls.append(act.log)
            action, error = parse_action(act.value, agent.tools)
            agent.messages = agent.messages + [{"role": "assistant", "content": act.value or ""}]
            if action is None:
                self.invalid += 1
                agent.pending = f"Rejected again: {error}. That turn is used up."
        agent.turns += 1
        agent.last_reasoning = (act.log or {}).get("reasoning") or ""
        self.transcript.append({"step": "turn", "agent": agent.name, "role": agent.role,
                                "turn": agent.turns, "budget_left": left, "action": action,
                                "error": error if action is None else None,
                                "retried": retried, "calls": calls})
        if action and action["tool"] == "note":
            agent.notes.append(action["args"]["text"])
            agent.pending = "Noted."
        return action or {"tool": None, "args": {}}

    def refuse(self, agent: TeamAgent, tool: str, reason: str) -> None:
        self.refused += 1
        agent.pending = f"Refused: {reason}"
        self.transcript.append({"step": "refused", "agent": agent.name, "tool": tool,
                                "reason": reason})

    @staticmethod
    def flush(agent: TeamAgent) -> None:
        """Before an agent is asked for an answer, report or review outside a turn: keep the
        reasoning of its latest turn with the action it led to (only the action stays in the
        conversation between turns, so without this an agent that submits right after working
        the chain out would answer without its own working), and show it its last action's result
        (e.g. the workers' reports)."""
        if agent.last_reasoning:
            i = max(i for i, m in enumerate(agent.messages) if m["role"] == "assistant")
            msg = agent.messages[i]
            agent.messages = agent.messages[:i] + [
                {"role": "assistant", "content": f"{agent.last_reasoning}\n\n{msg['content']}"}
            ] + agent.messages[i + 1:]
            agent.last_reasoning = ""
        if agent.pending:
            agent.messages = with_user(agent.messages, agent.pending)
            agent.pending = ""

    def write(self, agent: TeamAgent, kind: str) -> str:
        """A forced report or review, when the agent's turns or the budget ran out first."""
        self.flush(agent)
        self.forced[kind] += 1
        instruction = (f"Your turns or the team's budget ran out. Write your {kind} now, in at "
                       f"most {self.cfg.report_tokens} tokens, with nothing else.")
        act = self.policy.write(agent, instruction, self.seed())
        self.charge(agent.name, act.usage, act.log)
        self.transcript.append({"step": "forced", "agent": agent.name, "kind": kind,
                                "call": act.log})
        return act.value or ""

    def answer(self, agent: TeamAgent, forced: bool):
        self.flush(agent)
        if forced:
            self.forced["answer"] += 1
        res = self.policy.final_answer(agent, forced, self.seed())
        self.charge(agent.name, res.usage, res.extra.get("log"))
        self.transcript.append({"step": "answer", "agent": agent.name, "answer": res.final_answer,
                                "probabilities": res.probabilities, "refusal": res.refusal,
                                "valid": res.valid, "forced": forced,
                                "call": res.extra.get("log")})
        agent.done = True
        return res

    # --- topologies -------------------------------------------------------------------------

    def solo(self) -> list:
        """One agent; with `samples` > 1 it makes that many independent attempts (each in a
        fresh conversation) and the team answer is their majority vote."""
        results = []
        for _ in range(self.cfg.samples):
            a = self.start("A", "solo", ("note", "submit"),
                           "You are working on this task alone.")
            while True:
                act = self.turn(a)
                if act is None or act["tool"] == "submit":
                    results.append(self.answer(a, forced=act is None))
                    break
        return results

    def lead(self) -> list:
        cfg = self.cfg
        tools = ("note", "delegate", "submit")
        lead = self.start(
            "A", "lead", tools,
            f"You lead a team. Use delegate to hand parts of the work to workers, one brief per "
            f"worker (up to {cfg.max_workers} workers in all). Until at least {cfg.min_workers} "
            f"workers have reported, delegate is your only action; after that you may also note "
            "and submit. Workers see the task and the brief you give them, not your working.")
        # Delegate first (decided 2026-11-04): in the pilot, leads free to choose mostly worked
        # alone, so L did not test splitting. Any other action is rejected until enough reports.
        lead.tools = ("delegate",)
        reports = 0
        used = 0
        while True:
            act = self.turn(lead)
            if act is None:
                return [self.answer(lead, forced=True)]
            if act["tool"] == "submit":
                if reports >= cfg.min_workers:
                    return [self.answer(lead, forced=False)]
                self.refuse(lead, "submit", f"you need reports from at least {cfg.min_workers} "
                                            f"workers first; you have {reports}. Start workers "
                                            f"with {TOOLS['delegate'][0]}, one brief per worker.")
            elif act["tool"] == "delegate":
                room = cfg.max_workers - used
                if room <= 0:
                    self.refuse(lead, "delegate", "all workers have been used.")
                    continue
                briefs = act["args"]["briefs"][:room]
                texts = self.run_workers(lead, NAMES[1 + used: 1 + used + len(briefs)], briefs)
                used += len(briefs)
                reports += len(texts)
                if reports >= cfg.min_workers:
                    lead.tools = tools
                dropped = len(act["args"]["briefs"]) - len(briefs)
                lead.pending = "Reports:\n" + "\n".join(
                    f"From worker {n}: {t}" for n, t in texts) + (
                    f"\n({dropped} brief(s) dropped: only {cfg.max_workers} workers in all.)"
                    if dropped else "")
                lead.inbox += [t for _, t in texts]

    def run_workers(self, lead: TeamAgent, names: list[str],
                    briefs: list[str]) -> list[tuple[str, str]]:
        """Workers run side by side: one turn each per round, so none of them can use up the
        budget before the others start."""
        workers = []
        for name, brief in zip(names, briefs, strict=True):
            brief = self.deliver("brief", lead.name, [name], brief)
            workers.append(self.start(
                name, "worker", ("note", "report"),
                f"You are a worker. The lead gave you this brief:\n{brief}\n\nWork on it and "
                f"send the lead a report. The lead sees only your report, not your working."))
        out: dict[str, str] = {}
        active = list(workers)
        while active:
            for w in list(active):
                act = self.turn(w)
                if act is None:
                    text, forced = self.write(w, "report"), True
                elif act["tool"] == "report":
                    text, forced = act["args"]["text"], False
                else:
                    continue
                out[w.name] = self.deliver("report", w.name, [lead.name], text, forced)
                w.done = True
                active.remove(w)
        return [(w.name, out[w.name]) for w in workers]

    def swarm(self, shared: bool) -> list:
        cfg = self.cfg
        tools = ("note", "append_note", "submit") if shared else ("note", "submit")
        text = (f"You are one of {cfg.peers} teammates working on this task at the same time. "
                + ("The team's shared notes are shown at the start of each turn. " if shared else
                   "You cannot see each other's work. ")
                + "The team's answer is the majority of the teammates' final answers.")
        peers = [self.start(n, "peer", tools, text) for n in NAMES[: cfg.peers]]
        board: list[tuple[str, str]] = []
        results = {}
        rnd = 0
        while any(not p.done for p in peers):
            rnd += 1
            visible = len(board)  # a round sees the notes written before it started
            order = [peers[i] for i in self.rng.permutation(len(peers))]
            self.transcript.append({"step": "round", "round": rnd,
                                    "order": [p.name for p in order], "notes_visible": visible})
            for p in order:
                if p.done:
                    continue
                extra = ""
                if shared and visible > p.seen_notes:
                    new = board[p.seen_notes: visible]
                    extra = "New shared notes:\n" + "\n".join(f"From {a}: {t}" for a, t in new)
                    p.inbox += [t for _, t in new]
                    p.seen_notes = visible
                act = self.turn(p, extra)
                if act is None:
                    results[p.name] = self.answer(p, forced=True)
                elif act["tool"] == "submit":
                    results[p.name] = self.answer(p, forced=False)
                elif act["tool"] == "append_note":
                    board.append((p.name, self.deliver("note", p.name, ["team"],
                                                       act["args"]["text"])))
                    p.pending = "Added to the shared notes."
        return [results[p.name] for p in peers]

    def review(self) -> list:
        author = self.start(
            "A", "author", ("note", "request_review", "submit"),
            "You solve the task. Send your worked solution to the reviewer once, read the "
            "review, revise if needed, then submit. You can submit only after the review.")
        reviewed = False
        while True:
            act = self.turn(author)
            if act is None:
                return [self.answer(author, forced=True)]
            if act["tool"] == "submit":
                if reviewed:
                    return [self.answer(author, forced=False)]
                self.refuse(author, "submit", "send your solution for review first.")
            elif act["tool"] == "request_review":
                if reviewed:
                    self.refuse(author, "request_review", "the review has already been done.")
                    continue
                solution = self.deliver("solution", "A", ["B"], act["args"]["solution"])
                reviewer = self.start(
                    "B", "reviewer", ("note", "review"),
                    f"You are a reviewer. An author solved this task and sent this worked "
                    f"solution:\n{solution}\n\nCheck it against the facts and send the author "
                    "your review.")
                while True:
                    r = self.turn(reviewer)
                    if r is None:
                        text, forced = self.write(reviewer, "review"), True
                        break
                    if r["tool"] == "review":
                        text, forced = r["args"]["text"], False
                        break
                text = self.deliver("review", "B", ["A"], text, forced)
                reviewer.done = True
                author.pending = f"Review from the reviewer:\n{text}"
                author.inbox.append(text)
                reviewed = True

    def run(self) -> TeamOutcome:
        arm = self.cfg.arm_id
        results = {"S": self.solo, "S_ref": self.solo, "L": self.lead, "W": lambda: self.swarm(True),
                   "W0": lambda: self.swarm(False), "R": self.review}[arm]()
        final, mean = aggregate(results, self.inst.answer_space, self.rng)
        refusal = next((r.refusal for r in results if r.refusal), None)
        turns = {s["agent"]: s["turn"] for s in self.transcript if s["step"] == "turn"}
        summary = {
            "budget": self.cfg.budget, "spent": self.ledger.total_tokens,
            "depth": self.cfg.depth, "samples": self.cfg.samples,
            "generated": sum(a.output_tokens for a in self.ledger.agents.values()),
            "turns": turns, "agents": sorted(self.ledger.agents),
            "workers": sum(s["kind"] == "brief" for s in self.transcript
                           if s["step"] == "deliver"),
            "forced_answers": self.forced["answer"], "forced_reports": self.forced["report"],
            "forced_reviews": self.forced["review"], "invalid_actions": self.invalid,
            "refused_actions": self.refused,
        }
        return TeamOutcome(final, mean, [r.final_answer for r in results], refusal,
                           all(r.valid for r in results), self.ledger, self.transcript,
                           self.request_ids, team=summary)


def run_team_arm(inst: TaskInstance, cfg: TeamConfig, policy: Policy, seed: int) -> TeamOutcome:
    if inst.family_id != "reason":
        raise ValueError("design-3 team arms run on the reason family only")
    return _Team(inst, cfg, policy, seed).run()


__all__ = ["TEAM_ARMS", "TeamConfig", "TeamAgent", "team_config", "ref_config", "parse_action",
           "run_team_arm"]
