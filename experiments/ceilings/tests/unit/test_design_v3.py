"""Arm design 3: agent teams (prereg/DESIGN3_PLAN.md)."""

import json

import pytest

from ceilings import arms, tasks
from ceilings.adapters.base import (
    TURN_ACTION_PROMPT,
    TURN_REASON_PROMPT,
    AdapterResult,
    ChatAdapter,
    ChatReply,
    Usage,
    approx_tokens,
)
from ceilings.agents import Action, LLMPolicy, Policy, make_policy
from ceilings.checks import run_checks
from ceilings.records import content_hash
from ceilings.runner import Cell, cell_run_id, load_runs, run_grid
from ceilings.tasks import reason
from ceilings.teams import (
    TEAM_ARMS,
    action_in_reasoning,
    parse_action,
    run_team_arm,
    team_config,
)


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
    ('{"tool": "delegate", "briefs": ["part 1", "part 2"]}', True),  # args beside "tool"
    ('{"tool": "note", "text": "Ada is B"}', True),
    ('{"tool": "delegate", "briefs": []}', False),
])
def test_actions_are_one_json_object_naming_one_of_the_agents_tools(text, ok):
    action, error = parse_action(text, ("note", "delegate", "submit"))
    assert (action is not None) == ok and (error is None) == ok


@pytest.mark.parametrize("arm", TEAM_ARMS)
def test_every_agent_starts_with_every_fact_and_every_run_ends_with_answers(arm):
    inst = _inst()
    out = run_team_arm(inst, team_config(arm), make_policy("sim-bounded"), 3)
    starts = [s for s in out.transcript if s["step"] == "start"]
    assert len(starts) >= (1 if arm in ("S", "S_ref") else 2)
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
    # until two workers have reported, delegate is the lead's only action: an early submit is
    # rejected as an invalid action and retried
    first = next(s for s in out.transcript if s["step"] == "turn" and s["agent"] == "A")
    assert first["retried"] and first["action"]["tool"] == "delegate"
    reports = [s for s in out.transcript if s["step"] == "deliver" and s["kind"] == "report"]
    assert len(reports) == 2 and out.team["forced_answers"] == 0
    assert out.team["workers"] == 2


def test_a_lead_that_starts_one_worker_must_start_another_before_it_can_submit():
    script = {"lead": [{"tool": "delegate", "args": {"briefs": ["part 1"]}}, SUBMIT,
                       {"tool": "delegate", "args": {"briefs": ["part 2"]}}, SUBMIT],
              "worker": [{"tool": "report", "args": {"text": "part done"}}]}
    out = run_team_arm(_inst(), team_config("L"), Scripted(script), 1)
    tools = [s["action"]["tool"] for s in out.transcript
             if s["step"] == "turn" and s["agent"] == "A"]
    assert tools == ["delegate", "delegate", "submit"] and out.team["workers"] == 2


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
        self.reply_prompts = []

    def _chat(self, messages, schema, max_tokens, seed):
        n = approx_tokens(" ".join(m["content"] for m in messages))
        if schema is not None:
            body = json.dumps({"answer": "A", "probabilities": {o: 0.25 for o in "ABCD"}})
            return ChatReply(body, n, 10, "r", self.model_name, "stop")
        if "write out only your reasoning" in messages[-1]["content"]:  # the reasoning stage
            return ChatReply("thinking about the chain", n, 20, "r", self.model_name, "stop")
        self.caps.append(max_tokens)
        self.reply_prompts.append(messages[-1]["content"])
        return ChatReply('{"tool": "submit", "args": {}}', n, 10, "r", self.model_name, "stop")


def test_llm_turns_reason_first_and_reply_within_the_turn_cap():
    chat = JsonChat()
    out = run_team_arm(_inst(), team_config("S"), LLMPolicy(chat), 1)
    turn = next(s for s in out.transcript if s["step"] == "turn")
    assert turn["action"]["tool"] == "submit"
    assert turn["calls"][0]["reasoning"] == "thinking about the chain"
    assert chat.caps == [team_config("S").reply_tokens]
    assert out.final_answer == "A"
    # the final answer is made with the reasoning of the turn that submitted, not without it
    answer = next(s for s in out.transcript if s["step"] == "answer")
    assert any("thinking about the chain" in m["content"] for m in answer["call"]["prompt"])


def test_a_turn_asks_for_its_action_not_for_the_final_answer():
    chat = JsonChat()
    out = run_team_arm(_inst(), team_config("S"), LLMPolicy(chat), 1)
    turn = next(s for s in out.transcript if s["step"] == "turn")
    reasoning_prompt, reply_prompt = (c[-1]["content"] for c in turn["calls"][0]["sent"])
    assert reasoning_prompt.endswith(TURN_REASON_PROMPT) and "final answer" not in TURN_REASON_PROMPT
    assert chat.reply_prompts == [TURN_ACTION_PROMPT]


class ReasonsTheAction(Scripted):
    """Writes the delegate in its reasoning, then replies with the step after it (seen in the
    November 2026 matching runs on GLM and Mistral)."""

    def act(self, agent, depth, reply_tokens, seed):
        a = super().act(agent, depth, reply_tokens, seed)
        if agent.role == "lead" and self.calls[agent.name] == 1:
            a.log["reasoning"] = ('Plan: two workers. {"tool": "note", "args": {"text": "x"}} '
                                  'no, delegate: {"tool": "delegate", "args": {"briefs": '
                                  '["part 1", "parts 2 and 3"]}}')
        return a


def test_an_action_written_only_in_the_reasoning_is_the_turns_action():
    script = {"lead": [SUBMIT, SUBMIT], "worker": [{"tool": "report", "args": {"text": "B"}}]}
    out = run_team_arm(_inst(), team_config("L"), ReasonsTheAction(script), 1)
    first = next(s for s in out.transcript if s["step"] == "turn" and s["agent"] == "A")
    assert first["action"]["tool"] == "delegate" and first["from_reasoning"]
    assert not first["retried"] and len(first["calls"]) == 1
    assert out.team["workers"] == 2 and out.team["actions_from_reasoning"] == 1
    assert out.team["invalid_actions"] == 0
    assert action_in_reasoning("no action here", ("delegate",)) is None
    assert action_in_reasoning('{"tool": "submit", "args": {}}', ("delegate",)) is None


def test_team_runs_record_their_config_in_the_run_id_and_pass_the_checks(tmp_path):
    insts = tasks.generate_many("reason", 3, "design3", d=9, layout="sections")
    tasks.save_instances(insts, tmp_path)
    policy = make_policy("sim-bounded")
    a, b = (cell_run_id(Cell(insts[0], team_config("S", budget=x)), policy) for x in (0, 5000))
    assert a != b
    cells = [Cell(i, team_config(arm, budget=20000)) for arm in TEAM_ARMS for i in insts]
    run_grid(cells, policy, tmp_path)
    records = load_runs(tmp_path)
    assert len(records) == 3 * len(TEAM_ARMS)
    assert {r["arm_design"] for r in records} == {arms.ARM_DESIGN}
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


def test_matching_words_in_an_agents_own_reply_are_not_a_reasoning_leak():
    from collections import defaultdict

    from ceilings.checks import _check_team_transcript

    inst = _inst()
    words = '{"tool": "submit", "args": {}} after working the whole chain through'
    user = {"role": "user", "content": f"Task: {inst.scenario}\n\n{inst.shared_evidence}"}

    def call(reasoning, prompt):
        return {"reasoning": reasoning, "prompt": prompt}

    def transcript(b_prompt):
        steps = [{"step": "start", "agent": a, "role": "peer", "prompt": [user]} for a in "AB"]
        steps += [{"step": "turn", "agent": "A", "turn": 1, "budget_left": None,
                   "action": SUBMIT, "calls": [call(words, [user])]},
                  {"step": "turn", "agent": "B", "turn": 1, "budget_left": None,
                   "action": SUBMIT, "calls": [call("", b_prompt)]}]
        return {"run_id": "r" * 64, "instance_id": inst.instance_id, "model_id": "m",
                "family": "reason", "steps": steps,
                "config": team_config("W", peers=2).to_dict()}

    problems, rates = defaultdict(list), defaultdict(lambda: defaultdict(list))
    own = [user, {"role": "assistant", "content": words}]  # B wrote the same words itself
    _check_team_transcript(transcript(own), inst, problems, rates)
    assert not problems["reasoning_hidden"]
    leaked = [{"role": "user", "content": f"{user['content']}\n\n{words}"}]
    _check_team_transcript(transcript(leaked), inst, problems, rates)
    assert problems["reasoning_hidden"]


def test_reasoning_that_only_copies_what_the_agent_was_shown_is_not_a_leak():
    from collections import defaultdict

    from ceilings.checks import _check_team_transcript

    inst = _inst()
    note = "From D: Part 1: Tara=D. Finn is 2 after Tara: D->A->B (Finn=B)."
    seen = [{"role": "user", "content": f"Task: {inst.scenario}\n\nNew shared notes:\n{note}"}]
    steps = [{"step": "start", "agent": a, "role": "peer", "prompt": seen} for a in "AB"]
    steps += [{"step": "turn", "agent": a, "turn": 1, "budget_left": None, "action": SUBMIT,
               "calls": [{"reasoning": note, "prompt": seen}]} for a in "AB"]
    t = {"run_id": "r" * 64, "instance_id": inst.instance_id, "model_id": "m",
         "family": "reason", "steps": steps, "config": team_config("W", peers=2).to_dict()}
    problems, rates = defaultdict(list), defaultdict(lambda: defaultdict(list))
    _check_team_transcript(t, inst, problems, rates)
    assert not problems["reasoning_hidden"]


def test_instances_saved_for_the_same_setting_are_merged_not_replaced(tmp_path):
    a = tasks.generate_many("reason", 3, "design3", 0, d=6, layout="sections")
    b = tasks.generate_many("reason", 2, "design3", 10_000_000, d=6, layout="sections")
    tasks.save_instances(a, tmp_path)
    tasks.save_instances(b, tmp_path)
    tasks.save_instances(a, tmp_path)  # saving again adds nothing
    from ceilings.checks import load_instances_index

    assert set(load_instances_index(tmp_path)) == {i.instance_id for i in a + b}


def test_s_ref_is_one_agent_making_several_attempts_at_a_chosen_depth():
    from ceilings.teams import ref_config

    cfg = ref_config("max:3", 5000, max_turns=4)
    assert (cfg.arm_id, cfg.depth, cfg.samples, cfg.budget) == ("S_ref", "max", 3, 5000)
    assert ref_config("high").samples == 1
    with pytest.raises(ValueError, match="only S and S_ref"):
        team_config("W", samples=2)
    out = run_team_arm(_inst(), cfg, make_policy("sim-bounded"), 2)
    starts = [s for s in out.transcript if s["step"] == "start"]
    assert len(starts) == 3 and {s["agent"] for s in starts} == {"A"}
    assert len(out.sample_answers) == 3 and out.team["samples"] == 3


def test_a_half_rung_makes_the_whole_number_below_or_above_by_a_coin_fixed_per_instance():
    from ceilings.analysis.teams import strength_of
    from ceilings.teams import attempts, ref_config

    cfg = ref_config("max:3.5")
    assert cfg.samples == 3.5 and ref_config("max:3").samples == 3
    insts = tasks.generate_many("reason", 40, "design3", d=9, layout="sections")
    made = [attempts(cfg, i.instance_id) for i in insts]
    assert set(made) == {3, 4} and 10 <= made.count(4) <= 30
    assert made == [attempts(cfg, i.instance_id) for i in insts]  # the same on every rerun
    out = run_team_arm(insts[0], cfg, make_policy("sim-bounded"), 2)
    assert len(out.sample_answers) == made[0] == out.team["attempts"]
    assert strength_of(out.team) == "max:3.5" and strength_of({"depth": "max", "samples": 3}) \
        == "max:3"
    with pytest.raises(ValueError, match="one decimal"):
        ref_config("max:3.25")


def test_a_tenth_rung_adds_an_attempt_on_that_share_of_instances_and_keeps_the_halves():
    from ceilings.teams import attempts, ref_config

    insts = tasks.generate_many("reason", 200, "design3", d=9, layout="sections")
    seven = [attempts(ref_config("max:1.7"), i.instance_id) for i in insts]
    assert set(seven) == {1, 2} and 120 <= seven.count(2) <= 160
    # a half adds one exactly when the instance's hash is odd, as before tenths were allowed
    for i in insts:
        odd = int(content_hash([i.instance_id, "attempts"])[:8], 16) % 2
        assert attempts(ref_config("max:3.5"), i.instance_id) == 3 + odd
        assert attempts(ref_config("max:1.7"), i.instance_id) >= 1 + odd  # nested shares


def test_a_half_rung_lies_only_between_neighbouring_attempt_counts():
    from ceilings.team_match import half_between

    assert half_between("max:3", "max:4") == "max:3.5"
    assert half_between("max", "max:2") == "max:1.5"
    assert half_between("max:8", "max:10") is None and half_between("xhigh", "max") is None


def test_the_tenth_rung_interpolates_the_cost_between_its_neighbours():
    from ceilings.team_match import interpolated

    assert interpolated("max", "max:2", 5412, 11013, 9080) == "max:1.7"
    assert interpolated("max:3", "max:4", 1000, 2000, 1500) is None  # that is the half
    assert interpolated("max:3", "max:4", 1000, 2000, 1010) == "max:3.1"
    assert interpolated("max:8", "max:10", 1000, 2000, 1500) is None


def test_team_match_finds_each_teams_rung_and_team_run_adds_the_references(tmp_path):
    from ceilings import cli
    from ceilings.team_match import LADDER, load_table

    root = str(tmp_path)
    args = ["--root", root, "team-match", "--d", "9", "--n", "4", "--budget", "0",
            "--arms", "L,R"]
    code = cli.main(args)
    table = load_table(root, "sim-bounded")
    assert set(table["arms"]) == {"L", "R"} and code in (0, 2)
    for e in table["arms"].values():
        assert e["strength"] in LADDER or e["strength"] in table["ladder_tokens"]
        assert e["matched"] == (abs(e["gap"]) <= table["tolerance"])
    # rungs are measured in ladder order; deeper reasoning can finish in fewer turns, so cost
    # need not rise at every rung, and each team gets the nearest rung by measured cost
    whole = [s for s in table["ladder_tokens"] if s in LADDER]
    assert whole == LADDER[: len(whole)]
    for e in table["arms"].values():
        gaps = [abs(t / e["target_tokens"] - 1) for t in table["ladder_tokens"].values()]
        assert abs(e["gap"]) == pytest.approx(min(gaps))
    assert cli.main(["--root", root, "team-run", "--d", "9", "--n", "3", "--budget", "0",
                     "--arms", "L,R", "--with-refs"]) == 0
    main = {i.instance_id for i in tasks.generate_many("reason", 3, "design3", d=9,
                                                        layout="sections")}
    refs = [r for r in load_runs(root) if r["arm_id"] == "S_ref" and r["instance_id"] in main]
    picked = {table["arms"][a]["strength"] for a in ("L", "R")}
    assert {f"{r['team']['depth']}:{r['team']['samples']}" for r in refs} == {
        s if ":" in s else f"{s}:1" for s in picked}
    # a table matched under other settings is refused
    assert cli.main(["--root", root, "team-run", "--d", "9", "--n", "3", "--budget", "100",
                     "--arms", "L", "--with-refs"]) == 2


def test_claims_are_graded_against_the_chain():
    from ceilings.analysis.teams import chain_shifts, claims

    inst = _inst(d=9)
    truth = chain_shifts(inst.payload["public"])
    target = inst.payload["public"]["target"]
    assert truth[target] == inst.answer and len(truth) == 10
    wrong = "ABCD"[("ABCD".index(inst.answer) + 1) % 4]
    text = f"{target} works shift {wrong}. Nobody=A. {target}={inst.answer}"
    assert claims(text, set(truth)) == [(target, wrong), (target, inst.answer)]


def test_team_analysis_pairs_each_team_with_its_matched_reference(tmp_path):
    from ceilings import cli

    root = str(tmp_path)
    cli.main(["--root", root, "team-match", "--d", "9", "--n", "4", "--budget", "0",
              "--arms", "L,R,W"])
    cli.main(["--root", root, "team-run", "--d", "9", "--n", "6", "--budget", "0",
              "--arms", "L,R,W", "--with-refs"])
    out = tmp_path / "reports"
    assert cli.main(["--root", root, "team-analyze", "--out", str(out)]) == 0
    res = json.loads((out / "team_verdicts.json").read_text())
    table = json.loads(next((tmp_path / "team_match").glob("*.json")).read_text())
    for t in res["tests"]:
        assert t["ref"] == table["arms"][t["arm"]]["strength"]
        assert t["n"] == 6  # only the main instances: the matching ones chose the strengths
        assert ("p_tost_holm" in t) == t["matched"]  # unmatched teams stay out of the verdicts
        assert t["realized_cost_gap"] > -1
    for h in ("H5a", "H5b"):
        assert res[h]["verdict"] in ("supported", "contradicted", "inconclusive",
                                     "not evaluable")
    assert {c["arm"] for c in res["H5c"]["cells"]} <= {"L", "R", "W"}
