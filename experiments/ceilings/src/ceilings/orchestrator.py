"""TeamOrchestrator: one fixed protocol, varied by one resource per arm (contracts/interfaces.md).

Protocol: (1) build each agent's view; (2) tool rounds through the ToolSimulator, results go to
the shared record; (3) message rounds through the Channel (helpers send to the answerer, or relay
partial work down the chain D -> C -> B -> A); (4) answering agents give `samples` answers each,
aggregated by majority vote (ties broken by mean probability) and by mean probability.

Every model call is logged in the transcript with the agent, the step, the prompt and the raw
reply, so `ceilings check` and human reviewers can verify what each agent saw and did.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from ceilings.agents import Policy, View
from ceilings.arms import ArmConfig, extra_text
from ceilings.channel import Channel
from ceilings.ledger import Ledger
from ceilings.records import TaskInstance
from ceilings.tasks import evidence_for_arm_e
from ceilings.tasks.comm import without_ready_message
from ceilings.tools import ToolSimulator

AGENT_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]


@dataclass
class TeamOutcome:
    final_answer: str | None
    probabilities: dict[str, float]
    sample_answers: list[str | None]
    refusal: dict | None
    valid: bool
    ledger: Ledger
    transcript: list[dict] = field(default_factory=list)
    request_ids: list[str] = field(default_factory=list)


def _view(instance: TaskInstance, name: str, cfg: ArmConfig, answerer: bool) -> View:
    private_text = instance.private_evidence.get(name, "")
    if cfg.comm_sender == "encode" and instance.family_id.startswith("comm_"):
        private_text = without_ready_message(private_text)
    extra = ""
    if answerer and name == cfg.answerers[0]:
        evidence = (
            evidence_for_arm_e(instance, cfg.extra_amount)
            if cfg.extra_kind in ("restated", "same_kind")
            else ""
        )
        extra = extra_text(cfg.extra_kind, cfg.extra_amount, cfg.filler_tokens, evidence)
    return View(
        name=name,
        family_id=instance.family_id,
        scenario=instance.scenario,
        shared_text=instance.shared_evidence,
        private_text=private_text,
        public=instance.payload.get("public", {}),
        private=instance.payload.get("private", {}).get(name, {}),
        answer_space=instance.answer_space,
        extra_text=extra,
    )


def run_team(instance: TaskInstance, cfg: ArmConfig, policy: Policy, seed: int) -> TeamOutcome:
    # Team-level harness options (ceilings.options) override the arm's defaults when set.
    team_opts = {k: v for k, v in policy.options.non_default().items()
                 if k in ("probe_policy", "comm_sender")}
    if team_opts:
        cfg = replace(cfg, **team_opts)
    rng = np.random.default_rng(seed)
    ledger = Ledger()
    transcript: list[dict] = []
    request_ids: list[str] = []
    names = AGENT_NAMES[: cfg.n_agents]
    views = {n: _view(instance, n, cfg, n in cfg.answerers) for n in names}

    def call_seed() -> int:
        return int(rng.integers(2**31))

    def charge(agent: str, usage) -> None:
        ledger.add_call(agent, usage.input_tokens, usage.output_tokens, usage.reasoning_tokens,
                        usage.dollars)

    # (2) Tool rounds. With the default "rotation" policy the harness probes attribute t mod k, so
    # every model receives the same evidence and the test isolates whether it can use it; with
    # "agent" the probing agent (round-robin over the team) chooses. Results go to everyone.
    if cfg.tool_rounds and instance.tool is not None:
        tool = ToolSimulator(
            instance.tool,
            instance.payload["hidden"]["value"],
            instance.payload["public"]["k"],
            np.random.default_rng(seed + 1),
            ledger,
        )
        for t in range(cfg.tool_rounds):
            prober = names[t % len(names)]
            if cfg.probe_policy == "agent":
                action = policy.choose_probe(views[prober], cfg.depth, call_seed())
                charge(prober, action.usage)
                query, log = action.value, action.log
            else:
                query, log = t % instance.payload["public"]["k"], None
            result = tool.probe(query)
            for v in views.values():
                v.tool_results.append((query, result))
            transcript.append({"step": "probe", "agent": prober, "query": query,
                               "result": result, "policy": cfg.probe_policy, "call": log})

    # (3) Message rounds.
    # A bit budget of 0 means nothing can cross, so no message calls are made (and none charged).
    can_send = cfg.message_budget > 0 or (cfg.relay and cfg.message_mode == "tokens")
    if can_send and len(names) > 1:
        channel = Channel(cfg.message_mode, cfg.message_budget, ledger, policy.count_tokens)
        chain = list(reversed(names))  # D -> C -> B -> A when relaying; B -> A with two agents
        senders = chain[:-1] if cfg.relay else [n for n in names if n not in cfg.answerers]
        for i, sender in enumerate(senders):
            receiver = chain[i + 1] if cfg.relay else cfg.answerers[0]
            calls: list[dict] = []

            def compose(retry, sender=sender, receiver=receiver, calls=calls):
                action = policy.compose(views[sender], receiver, cfg.depth, cfg.message_mode,
                                        cfg.message_budget, retry, call_seed())
                charge(sender, action.usage)
                calls.append(action.log)
                return action.value

            delivery = channel.send(sender, receiver, compose)
            if delivery.text:
                views[receiver].inbox.append((sender, delivery.text))
            transcript.append({
                "step": "message", "from": sender, "to": receiver, "text": delivery.text,
                "bits": delivery.bits, "tokens": delivery.tokens, "retried": delivery.retried,
                "truncated": delivery.truncated, "budget": cfg.message_budget,
                "mode": cfg.message_mode, "calls": calls,
            })

    # (4) Answers.
    results = []
    for name in cfg.answerers:
        for _ in range(cfg.samples):
            res = policy.answer(views[name], cfg.depth, call_seed())
            charge(name, res.usage)
            request_ids.append(res.request_id)
            results.append(res)
            transcript.append({"step": "answer", "agent": name, "answer": res.final_answer,
                               "probabilities": res.probabilities, "refusal": res.refusal,
                               "valid": res.valid, "call": res.extra.get("log")})

    space = instance.answer_space
    mean = {a: float(np.mean([r.probabilities.get(a, 0.0) for r in results])) for a in space}
    total = sum(mean.values()) or 1.0
    mean = {a: v / total for a, v in mean.items()}
    votes = Counter(r.final_answer for r in results if r.final_answer is not None)
    if votes:
        top = max(votes.values())
        tied = [a for a, c in votes.items() if c == top]
        final = max(tied, key=lambda a: mean[a])
    else:
        final = None
    refusal = next((r.refusal for r in results if r.refusal), None)
    return TeamOutcome(final, mean, [r.final_answer for r in results], refusal,
                       all(r.valid for r in results), ledger, transcript, request_ids)
