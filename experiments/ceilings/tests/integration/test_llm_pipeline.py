"""The LLM path end to end with a scripted text-solving model: prompts carry the information,
the checks pass, the diagonal appears, concurrency changes nothing, and the checks catch
deliberate violations."""

import json

from ceilings import arms, checks, experiment, tasks
from ceilings.agents import LLMPolicy
from ceilings.analysis.gains import gain_table, runs_frame
from ceilings.numbers import entries, show_line
from ceilings.orchestrator import run_team
from ceilings.runner import Cell, load_runs, run_grid
from tests.scripted import ScriptedChat

DIFF = {"info": {"k": 3, "p": 0.0}, "reason": {"d": 6}, "comm_ip": {"n": 4},
        "comm_parity": {"n": 4}}


def _policy():
    return LLMPolicy(ScriptedChat())


def test_prompts_carry_the_information_under_the_matching_arm():
    policy = _policy()
    for fam, arm, strength in [("info", "E", 3), ("reason", "K", "high"),
                               ("comm_ip", "M", 4), ("comm_parity", "M", 1)]:
        insts = tasks.generate_many(fam, 15, "calibration", **DIFF[fam])
        full = [run_team(i, arms.build(arm, fam, strength), policy, 1).final_answer == i.answer
                for i in insts]
        assert all(full), (fam, sum(full))


def test_scripted_suite_passes_checks_and_shows_diagonal(tmp_path):
    policy = _policy()
    for fam in ["info", "reason", "comm_ip"]:
        prep = experiment.prepare(fam, policy, tmp_path, n_main=40, n_pilot=12,
                                  difficulty=DIFF[fam], levels={"medium": 1.0}, concurrency=4)
        run_grid(experiment.main_cells(prep["main"], prep["table"], deltas=("medium",)),
                 policy, tmp_path, concurrency=4)
    records = load_runs(tmp_path)
    res = checks.run_checks(tmp_path, records)
    assert res["passed"], {k: v for k, v in res["integrity"].items() if v["status"] != "PASS"}
    assert res["n_transcripts"] == len(records)
    helps = {r["family"]: r["status"] for r in res["expectations"]["matching_arm_helps"]}
    assert helps == {"info": "met", "reason": "met", "comm_ip": "met"}
    assert res["quality"]["answer_parse_failed"]["status"] == "PASS"

    table = gain_table(runs_frame(records), n_boot=500)
    row = table[(table.family == "info") & (table.arm == "E")].iloc[0].to_dict()
    assert show_line(row).startswith("gain model=scripted/scripted family=info arm=E")
    entry = entries(table, "data", [row])[0]
    assert entry["cwd"] == "../experiments/ceilings" and "ceilings --root data show" in \
        entry["command"]


def test_concurrency_gives_identical_records(tmp_path):
    insts = tasks.generate_many("info", 8, "main", **DIFF["info"])
    cells = [Cell(i, arms.build(a, "info", s), None if a == "baseline" else "medium")
             for i in insts for a, s in [("baseline", None), ("E", 2), ("P2", None)]]
    a = run_grid(cells, _policy(), tmp_path / "a", concurrency=1)
    b = run_grid(cells, _policy(), tmp_path / "b", concurrency=6)
    key = lambda r: r.run_id  # noqa: E731
    assert [(r.run_id, r.final_answer, r.ledger.total_tokens) for r in sorted(a, key=key)] == \
        [(r.run_id, r.final_answer, r.ledger.total_tokens) for r in sorted(b, key=key)]


def _one_transcript(tmp_path, fam, arm, strength):
    inst = tasks.generate_many(fam, 1, "main", **DIFF[fam])[0]
    tasks.save_instances([inst], tmp_path)
    rec = run_grid([Cell(inst, arms.build(arm, fam, strength), "medium")], _policy(), tmp_path)[0]
    return inst, rec, json.loads(open(rec.transcript_ref).read())


def test_checks_catch_a_leaked_answer(tmp_path):
    inst, rec, t = _one_transcript(tmp_path, "info", "P2", None)
    call = next(s["call"] for s in t["steps"] if s["step"] == "answer")
    call["prompt"][-1]["content"] += f"\nHint: it is {inst.answer}. {inst.answer}!"
    open(rec.transcript_ref, "w").write(json.dumps(t))
    res = checks.run_checks(tmp_path, load_runs(tmp_path))
    assert res["integrity"]["no_answer_leak"]["status"] == "FAIL"


def test_checks_catch_private_evidence_crossing_agents(tmp_path):
    inst, rec, t = _one_transcript(tmp_path, "comm_ip", "M", 4)
    call = next(s["call"] for s in t["steps"] if s["step"] == "answer")
    call["prompt"][-1]["content"] += "\n" + inst.private_evidence["B"]
    open(rec.transcript_ref, "w").write(json.dumps(t))
    res = checks.run_checks(tmp_path, load_runs(tmp_path))
    assert res["integrity"]["private_isolation"]["status"] == "FAIL"


def test_checks_catch_an_arm_changing_two_resources(tmp_path):
    inst, rec, t = _one_transcript(tmp_path, "reason", "K", "high")
    t["config"]["message_budget"] = 50
    open(rec.transcript_ref, "w").write(json.dumps(t))
    res = checks.run_checks(tmp_path, load_runs(tmp_path))
    assert res["integrity"]["arm_changes_one_resource"]["status"] == "FAIL"


def test_rotation_probes_every_attribute_without_model_calls():
    inst = tasks.generate_many("info", 1, "main", k=3, p=0.0)[0]
    policy = _policy()
    out = run_team(inst, arms.build("E", "info", 6), policy, 3)
    probes = [s for s in out.transcript if s["step"] == "probe"]
    assert [p["query"] for p in probes] == [0, 1, 2, 0, 1, 2]
    assert all(p["call"] is None and p["policy"] == "rotation" for p in probes)
    assert out.final_answer == inst.answer


def test_unmatched_prepare_marks_every_cell(tmp_path):
    prep = experiment.prepare("reason", _policy(), tmp_path, n_main=4, n_pilot=4,
                              difficulty={"d": 3}, levels={"medium": 1.0}, match=False)
    table = prep["table"]
    assert table["unmatched_defaults"] and table["targets"]["medium"] > 0
    assert all(not e["matched"] for v in table["arms"].values() for e in v.values())
    assert table["arms"]["K"]["medium"]["strength"] == "high"


def test_no_message_calls_when_nothing_can_cross():
    inst = tasks.generate_many("comm_ip", 1, "main", n=4)[0]
    out = run_team(inst, arms.build("A2", "comm_ip", "medium"), _policy(), 2)
    assert not [s for s in out.transcript if s["step"] == "message"]


def test_transcripts_record_every_turn_sent():
    inst = tasks.generate_many("reason", 1, "main", d=3)[0]
    out = run_team(inst, arms.build("K", "reason", "high"), _policy(), 2)
    call = next(s["call"] for s in out.transcript if s["step"] == "answer")
    assert len(call["sent"]) == 2  # the reasoning turn, then the final-answer turn
    assert call["sent"][0][-1]["content"].endswith("in the next turn.")
    final = call["sent"][1][-1]["content"]
    assert "in the required format." in final and "Reply with only a JSON object" in final


def test_request_ids_cover_message_and_probe_calls():
    inst = tasks.generate_many("comm_ip", 1, "main", n=4)[0]
    out = run_team(inst, arms.build("M", "comm_ip", 4), _policy(), 2)
    assert len(out.request_ids) == 5  # one message call, four answer calls
