"""Harness options for the ablations: parsing, what each one changes, and how runs record them."""

import json

import pytest

from ceilings import arms, tasks
from ceilings.adapters.base import answer_schema
from ceilings.agents import LLMPolicy, make_policy
from ceilings.options import HarnessOptions
from ceilings.orchestrator import run_team
from ceilings.runner import Cell, load_runs, run_grid, run_id_for
from tests.scripted import ScriptedChat


def _policy(text=None):
    return LLMPolicy(ScriptedChat(), HarnessOptions.parse(text))


def test_parse_and_non_default():
    opts = HarnessOptions.parse("probe=agent,sender=encode,schema=answer_first,shuffle=off,"
                                "temp=0,reason=0:64:128:256:512")
    assert opts.non_default() == {
        "probe_policy": "agent", "comm_sender": "encode", "answer_order": "answer_first",
        "shuffle_options": False, "temperature": 0.0, "reasoning_tokens": (0, 64, 128, 256, 512)}
    assert HarnessOptions.parse(None).non_default() == {}
    assert HarnessOptions.parse("shuffle=on").non_default() == {}
    for bad in ("probe=random", "colour=red", "reason=1:2"):
        with pytest.raises(ValueError):
            HarnessOptions.parse(bad)


def test_default_run_ids_are_unchanged_and_ablations_get_their_own():
    base = run_id_for("i" * 64, "E", "medium", "m", 0, None)
    assert run_id_for("i" * 64, "E", "medium", "m", 0, None, {}) == base
    assert run_id_for("i" * 64, "E", "medium", "m", 0, None, {"temperature": 0.0}) != base


def test_answer_first_schema_and_no_shuffle():
    space = ["A", "B", "C", "D"]
    s = answer_schema(space, 5, answer_first=True)
    assert list(s["properties"]) == ["answer", "probabilities"] == s["required"]
    adapter = ScriptedChat()
    adapter.configure(HarnessOptions.parse("shuffle=off"))
    assert all(adapter.schema_for(space, seed)["properties"]["answer"]["enum"] == space
               for seed in range(10))


def test_sender_encode_removes_the_ready_message_only():
    inst = tasks.generate_many("comm_ip", 1, "main", n=4)[0]
    cfg = arms.build("M", "comm_ip", 4)
    default = run_team(inst, cfg, _policy(), 1)
    encode = run_team(inst, cfg, _policy("sender=encode"), 1)

    def b_prompt(out):
        msg = next(s for s in out.transcript if s["step"] == "message")
        return msg["calls"][0]["prompt"][-1]["content"]

    assert "already written in the message format" in b_prompt(default)
    assert "already written in the message format" not in b_prompt(encode)
    assert "Role requires" in b_prompt(encode)
    assert encode.final_answer == inst.answer  # the scripted model encodes its own list


def test_probe_agent_lets_the_model_choose():
    inst = tasks.generate_many("info", 1, "main", k=3, p=0.0)[0]
    out = run_team(inst, arms.build("E", "info", 3), _policy("probe=agent"), 3)
    probes = [s for s in out.transcript if s["step"] == "probe"]
    assert all(p["policy"] == "agent" and p["call"] is not None for p in probes)
    assert sorted(p["query"] for p in probes) == [0, 1, 2]


def test_temperature_and_reasoning_caps_reach_the_adapter(monkeypatch):
    monkeypatch.setenv("CEILINGS_OPENAI_BASE_URL", "https://h/v1")
    policy = make_policy("openai/m", HarnessOptions.parse("temp=0,reason=0:32:64:128:256"))
    assert policy.adapter.temperature == 0.0 and policy.adapter.reasoning_budget("high") == 64
    rec = policy.harness_record()
    assert rec["temperature"] == 0.0
    assert rec["resolved"]["reasoning_tokens"] == [0, 32, 64, 128, 256]


def test_run_records_store_the_harness(tmp_path):
    inst = tasks.generate_many("comm_parity", 2, "main", n=4)
    tasks.save_instances(inst, tmp_path)
    cells = [Cell(i, arms.build("M", "comm_parity", 4), "medium") for i in inst]
    run_grid(cells, _policy(), tmp_path)
    run_grid(cells, _policy("schema=answer_first"), tmp_path)
    runs = load_runs(tmp_path)
    assert len(runs) == 4 and len({r["run_id"] for r in runs}) == 4
    harness = [r["harness"] for r in runs]
    assert harness[0] == {"resolved": harness[0]["resolved"]}
    assert harness[2]["answer_order"] == "answer_first"
    assert json.load(open(runs[2]["transcript_ref"]))["harness"] == harness[2]


def test_comm_depth_floors_reasoning_in_comm_families_only():
    comm = tasks.generate_many("comm_parity", 1, "main", n=2)[0]
    info = tasks.generate_many("info", 1, "main", k=2, p=0.0)[0]
    pol = _policy("comm_depth=medium")
    out = run_team(comm, arms.build("baseline", "comm_parity"), pol, 1)
    call = next(s["call"] for s in out.transcript if s["step"] == "answer")
    assert len(call["sent"]) == 2  # a reasoning turn, then the answer turn
    out = run_team(comm, arms.build("K", "comm_parity", "high"), pol, 1)
    call = next(s["call"] for s in out.transcript if s["step"] == "answer")
    assert len(call["sent"]) == 2  # K above the floor keeps its own depth
    out = run_team(info, arms.build("baseline", "info"), pol, 1)
    call = next(s["call"] for s in out.transcript if s["step"] == "answer")
    assert len(call["sent"]) == 1  # other families unchanged
    assert HarnessOptions.parse("comm_depth=medium").non_default() == {"comm_depth": "medium"}
    with pytest.raises(ValueError):
        HarnessOptions.parse("comm_depth=deep")
