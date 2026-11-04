"""Arm design 3: agent teams (prereg/DESIGN3_PLAN.md)."""

import json

import pytest

from ceilings import arms, tasks
from ceilings.adapters.base import AdapterResult, ChatAdapter, ChatReply, Usage, approx_tokens
from ceilings.agents import Action, LLMPolicy, Policy, make_policy
from ceilings.checks import run_checks
from ceilings.runner import Cell, cell_run_id, load_runs, run_grid
from ceilings.tasks import reason
from ceilings.teams import TEAM_ARMS, parse_action, run_team_arm, team_config


def _inst(d=9, i=0):
    return tasks.generate_many("reason", i + 1, "design3", d=d, layout="sections")[i]


class Scripted(Policy):
    """Plays a fixed list of replies per role (the last one repeats) and records every final
    answer prompt."""

    model_id = "scripted-team"

    def __init__(self, script):
        self.script = script
        self.calls: dict[str, int] = {}
        self.final_prompts: dict[str, list] = {}

    def act(self, agent, depth, reply_tokens, seed):
        replies = self.script[agent.role]
        i = self.calls.get(agent.name, 0)
        self.calls[agent.name] = i + 1
        reply = replies[min(i, len(replies) - 1)]
        text = reply if isinstance(reply, str) else json.dumps(reply)
        usage = Usage(approx_tokens("\n".join(m["content"] for m in agent.messages)), 50)
        return Action(text, usage, {"kind": "turn", "prompt": list(agent.messages), "value": text})

    def write(self, agent, instruction, seed):
        return Action("forced text", Usage(10, 5), {"kind": "write", "value": "forced text"})

    def final_answer(self, agent, forced, seed):
        self.final_prompts[agent.name] = list(agent.messages)
        probs = {a: 0.25 for a in agent.answer_space}
        return AdapterResult("A", probs, None, Usage(10, 5), f"s{seed}",
                             extra={"log": {"kind": "answer", "prompt": list(agent.messages)}})


SUBMIT = {"tool": "submit", "args": {}}


def test_sections_layout_lists_the_chain_in_parts_and_keeps_old_instances_unchanged():
    inst = _inst(d=12)
    lines = inst.shared_evidence.splitlines()
    heads = [i for i, ln in enumerate(lines) if ln.startswith("Rota, part")]
    assert len(heads) == reason.SECTIONS and heads[0] == 0
    assert inst.difficulty == {"d": 12, "layout": "sections"}
    pub = inst.payload["public"]
    start_line = f"{pub['start'][0]} works shift {'ABCD'[pub['start'][1]]}."
    assert start_line in lines[heads[0]: heads[1]]  # the starting fact is in part 1
    parts = pub["sections"]
    assert parts[0]["from"] == pub["start"][0] and parts[-1]["to"] == pub["target"]
    assert all(a["to"] == b["from"] for a, b in zip(parts, parts[1:], strict=False))
    assert parts[-1]["to_shift"] == inst.answer
    assert (pub["start"][1] + sum(p["offset"] for p in parts)) % 4 == "ABCD".index(inst.answer)
    assert set(inst.theory.I_held_by_agent) == set("ABCD")
    old = reason.generate(5, d=3)
    assert old.difficulty == {"d": 3} and "sections" not in old.payload["public"]
    assert old.instance_id != reason.generate(5, d=3, layout="sections").instance_id


def test_design3_split_never_repeats_a_main_or_calibration_seed():
    seeds = {i.seed for i in tasks.generate_many("reason", 20, "design3", d=9, layout="sections")}
    other = {i.seed for s in ("main", "calibration")
             for i in tasks.generate_many("reason", 20, s, d=9)}
    assert not seeds & other


@pytest.mark.parametrize("text,ok", [
    ('{"tool": "note", "args": {"text": "Ada is B"}}', True),
    ('```json\n{"tool": "submit", "args": {}}\n```', True),
    ('Thinking... {"tool": "delegate", "args": {"briefs": ["part 1", "part 2"]}}', True),
    ('{"tool": "delegate", "args": {"briefs": []}}', False),
    ('{"tool": "note", "args": {}}', False),
    ('{"tool": "report", "args": {"text": "x"}}', False),  # not one of this agent's tools
    ("I think the answer is B.", False),
])
def test_actions_are_one_json_object_naming_one_of_the_agents_tools(text, ok):
    action, error = parse_action(text, ("note", "delegate", "submit"))
    assert (action is not None) == ok and (error is None) == ok


@pytest.mark.parametrize("arm", TEAM_ARMS)
def test_every_agent_starts_with_every_fact_and_every_run_ends_with_answers(arm):
    inst = _inst()
    out = run_team_arm(inst, team_config(arm), make_policy("sim-bounded"), 3)
    starts = [s for s in out.transcript if s["step"] == "start"]
    assert len(starts) >= (2 if arm != "S" else 1)
    for s in starts:
        assert inst.shared_evidence in s["prompt"][-1]["content"]
    answers = [s for s in out.transcript if s["step"] == "answer"]
    assert len(answers) == (4 if arm in ("W", "W0") else 1)
    assert out.final_answer == inst.answer  # the simulated agents can solve d=9 in their turns


def test_the_lead_cannot_submit_before_two_workers_report():
    script = {"lead": [SUBMIT, {"tool": "delegate", "args": {"briefs": ["part 1", "part 2"]}},
                       SUBMIT],
              "worker": [{"tool": "report", "args": {"text": "part done"}}]}
    out = run_team_arm(_inst(), team_config("L"), Scripted(script), 1)
    refused = [s for s in out.transcript if s["step"] == "refused"]
    assert [s["tool"] for s in refused] == ["submit"]
    reports = [s for s in out.transcript if s["step"] == "deliver" and s["kind"] == "report"]
    assert len(reports) == 2 and out.team["forced_answers"] == 0
    assert out.team["workers"] == 2


def test_extra_briefs_beyond_three_workers_are_dropped():
    script = {"lead": [{"tool": "delegate", "args": {"briefs": ["1", "2", "3", "4"]}}, SUBMIT],
              "worker": [{"tool": "report", "args": {"text": "done"}}]}
    out = run_team_arm(_inst(), team_config("L"), Scripted(script), 1)
    assert out.team["workers"] == 3 and set(out.team["agents"]) == set("ABCD")


def test_a_lead_out_of_turns_still_reads_the_reports_before_its_forced_answer():
    script = {"lead": [{"tool": "delegate", "args": {"briefs": ["part 1", "part 2"]}}],
              "worker": [{"tool": "report", "args": {"text": "Faye works shift C"}}]}
    policy = Scripted(script)
    out = run_team_arm(_inst(), team_config("L", max_turns=1), policy, 1)
    assert out.team["forced_answers"] == 1
    assert any("Faye works shift C" in m["content"] for m in policy.final_prompts["A"])


def test_the_budget_stops_turns_and_forces_answers_but_never_skips_one():
    script = {"peer": [{"tool": "append_note", "args": {"text": "still working"}}]}
    out = run_team_arm(_inst(), team_config("W", budget=4000), Scripted(script), 1)
    turns = [s for s in out.transcript if s["step"] == "turn"]
    assert all(s["budget_left"] > 0 for s in turns)
    assert out.team["forced_answers"] == 4 and out.final_answer is not None
    assert len(turns) < 4 * team_config("W").max_turns  # the budget, not the turn cap, ended it


def test_shared_notes_arrive_the_round_after_they_are_written_and_never_in_w0():
    script = {"peer": [{"tool": "append_note", "args": {"text": "NOTE-X"}}, SUBMIT]}
    out = run_team_arm(_inst(), team_config("W"), Scripted(script), 1)
    first_round = [s for s in out.transcript if s["step"] == "turn" and s["turn"] == 1]
    assert not any("NOTE-X" in c["prompt"][-1]["content"] for s in first_round
                   for c in s["calls"])
    second = [s for s in out.transcript if s["step"] == "turn" and s["turn"] == 2]
    assert all("New shared notes" in s["calls"][0]["prompt"][-1]["content"] for s in second)
    script0 = {"peer": [{"tool": "append_note", "args": {"text": "NOTE-X"}}, SUBMIT]}
    out0 = run_team_arm(_inst(), team_config("W0"), Scripted(script0), 1)
    starts = [s for s in out0.transcript if s["step"] == "start"]
    assert all("append_note" not in s["tools"] for s in starts)
    assert not any(s["step"] == "deliver" for s in out0.transcript)
    # append_note is not a W0 tool: every peer's first reply is rejected and retried
    assert all(s["retried"] for s in out0.transcript if s["step"] == "turn" and s["turn"] == 1)


def test_an_invalid_action_gets_one_retry_then_the_turn_is_used_up():
    script = {"solo": ["no json here", "still none", SUBMIT]}
    out = run_team_arm(_inst(), team_config("S"), Scripted(script), 1)
    turns = [s for s in out.transcript if s["step"] == "turn"]
    assert turns[0]["retried"] and turns[0]["action"] is None and len(turns[0]["calls"]) == 2
    assert turns[1]["action"]["tool"] == "submit" and out.team["invalid_actions"] == 1


def test_the_author_submits_only_after_the_review():
    script = {"author": [SUBMIT, {"tool": "request_review", "args": {"solution": "Kemi is D"}},
                         SUBMIT],
              "reviewer": [{"tool": "review", "args": {"text": "Check part 2 again"}}]}
    policy = Scripted(script)
    out = run_team_arm(_inst(), team_config("R"), policy, 1)
    assert [s["tool"] for s in out.transcript if s["step"] == "refused"] == ["submit"]
    reviewer = next(s for s in out.transcript if s["step"] == "start" and s["agent"] == "B")
    assert "Kemi is D" in reviewer["prompt"][-1]["content"]
    assert any("Check part 2 again" in m["content"] for m in policy.final_prompts["A"])


def test_text_sent_between_agents_is_cut_to_the_cap():
    script = {"lead": [{"tool": "delegate", "args": {"briefs": ["a " * 900, "b"]}}, SUBMIT],
              "worker": [{"tool": "report", "args": {"text": "word " * 900}}]}
    out = run_team_arm(_inst(), team_config("L"), Scripted(script), 1)
    sent = [s for s in out.transcript if s["step"] == "deliver"]
    assert all(s["tokens"] <= s["cap"] for s in sent) and any(s["truncated"] for s in sent)


class JsonChat(ChatAdapter):
    """A chat model that always submits, recording each free-text reply cap."""

    model_name = "json-chat"
    model_id = "scripted/json-chat"

    def __init__(self):
        self.caps = []

    def _chat(self, messages, schema, max_tokens, seed):
        n = approx_tokens(" ".join(m["content"] for m in messages))
        if schema is not None:
            body = json.dumps({"answer": "A", "probabilities": {o: 0.25 for o in "ABCD"}})
            return ChatReply(body, n, 10, "r", self.model_name, "stop")
        if messages[-1]["content"].endswith("next turn."):  # the reasoning stage
            return ChatReply("thinking about the chain", n, 20, "r", self.model_name, "stop")
        self.caps.append(max_tokens)
        return ChatReply('{"tool": "submit", "args": {}}', n, 10, "r", self.model_name, "stop")


def test_llm_turns_reason_first_and_reply_within_the_turn_cap():
    chat = JsonChat()
    out = run_team_arm(_inst(), team_config("S"), LLMPolicy(chat), 1)
    turn = next(s for s in out.transcript if s["step"] == "turn")
    assert turn["action"]["tool"] == "submit"
    assert turn["calls"][0]["reasoning"] == "thinking about the chain"
    assert chat.caps == [team_config("S").reply_tokens]
    assert out.final_answer == "A"


def test_team_runs_record_their_config_in_the_run_id_and_pass_the_checks(tmp_path):
    insts = tasks.generate_many("reason", 3, "design3", d=9, layout="sections")
    tasks.save_instances(insts, tmp_path)
    policy = make_policy("sim-bounded")
    a, b = (cell_run_id(Cell(insts[0], team_config("S", budget=x)), policy) for x in (0, 5000))
    assert a != b
    cells = [Cell(i, team_config(arm, budget=20000)) for arm in TEAM_ARMS for i in insts]
    run_grid(cells, policy, tmp_path)
    records = load_runs(tmp_path)
    assert len(records) == 15 and {r["arm_design"] for r in records} == {arms.ARM_DESIGN}
    assert all(r["team"]["budget"] == 20000 for r in records)
    res = run_checks(tmp_path, records)
    assert res["passed"], {k: v for k, v in res["integrity"].items() if v["status"] != "PASS"}
    # a transcript whose worker never saw part of the task fails the facts check
    ref = next(r["transcript_ref"] for r in records if r["arm_id"] == "L")
    t = json.loads(open(ref).read())
    start = next(s for s in t["steps"] if s["step"] == "start" and s["agent"] == "B")
    start["prompt"][-1]["content"] = start["prompt"][-1]["content"].split("Rota, part 2")[0]
    open(ref, "w").write(json.dumps(t))
    res = run_checks(tmp_path, records)
    assert res["integrity"]["facts_in_every_prompt"]["status"] == "FAIL"
