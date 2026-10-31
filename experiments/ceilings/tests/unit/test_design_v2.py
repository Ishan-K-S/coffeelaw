"""Arm design 2: fixes made after the October 2026 paid run (README, "Design changes after the
paid run")."""

import json

import pytest

from ceilings import arms, calibration, tasks
from ceilings.adapters.base import ANSWER_TOKENS_PER_OPTION, ChatAdapter
from ceilings.agents import LLMPolicy, make_policy
from ceilings.channel import Channel
from ceilings.ledger import Ledger
from ceilings.orchestrator import run_team
from ceilings.runner import check_design, run_id_for
from tests.scripted import ScriptedChat


def _channel(budget=100):
    lines = ["Cleo works shift C.", "Zola works 1 shift(s) after Cleo."]
    return Channel("quote", budget, Ledger(), lambda t: len(t.split()), quotable=lines)


def test_quote_channel_passes_only_verbatim_task_lines_in_task_order():
    ch = _channel()
    sent = "So Ugo works shift D.\nZola  works 1 shift(s) after Cleo.\nCleo works shift C."
    d = ch.send("B", "A", lambda retry: sent)
    assert d.text == "Cleo works shift C.\nZola works 1 shift(s) after Cleo."
    assert ch.send("B", "A", lambda retry: "The answer is D.").text == ""


def test_quote_channel_cuts_whole_lines_at_the_budget():
    d = _channel(budget=5).send("B", "A", lambda retry: "Cleo works shift C.\n"
                                "Zola works 1 shift(s) after Cleo.")
    assert d.text == "Cleo works shift C." and d.truncated


def test_m_on_reason_cannot_carry_a_worked_answer():
    inst = tasks.generate_many("reason", 1, "main", d=3)[0]
    out = run_team(inst, arms.build("M", "reason", 100), make_policy("sim-bounded"), 3)
    msg = next(s for s in out.transcript if s["step"] == "message")
    assert msg["mode"] == "quote"
    assert all(line in inst.shared_evidence.splitlines() for line in msg["text"].splitlines())


@pytest.mark.parametrize("family,diff", [("comm_ip", {"n": 4}), ("comm_parity", {"n": 4})])
def test_a2_on_comm_runs_its_helpers_but_never_relays_b(family, diff):
    inst = tasks.generate_many(family, 1, "main", **diff)[0]
    out = run_team(inst, arms.build("A2", family, "medium"), make_policy("sim-bounded"), 5)
    hops = [(s["from"], s["to"]) for s in out.transcript if s["step"] == "message"]
    assert hops == [("D", "C"), ("C", "A")]
    assert set(out.ledger.to_record().per_agent) == {"A", "C", "D"}


def test_a2_agents_reason_on_reason():
    inst = tasks.generate_many("reason", 1, "main", d=3)[0]
    out = run_team(inst, arms.build("A2", "reason", "medium"), LLMPolicy(ScriptedChat()), 2)
    calls = [s["calls"][-1] for s in out.transcript if s["step"] == "message"]
    calls += [s["call"] for s in out.transcript if s["step"] == "answer"]
    assert len(calls) == 4 and all(len(c["sent"]) == 2 for c in calls)  # reasoning, then reply


def test_single_agent_reference_is_matched_to_a2s_realized_cost():
    insts = tasks.generate_many("reason", 6, "calibration", d=3)
    table = calibration.match_arms("reason", insts, make_policy("sim-bounded"),
                                   levels={"medium": 1.0})
    a2 = table["arms"]["A2"]["medium"]
    ref = table["arms"]["A2_single_ref"]["medium"]
    assert ref["target"] == round(a2["added"])
    assert a2["strength"] in ("medium", "high")  # never "low": the agents reason


def test_instances_never_repeat_a_prompt():
    insts = tasks.generate_many("info", 90, "main", k=2, p=0.0)
    assert len({tasks.prompt_key(i) for i in insts}) == 90
    with pytest.raises(RuntimeError, match="too few distinct prompts"):
        tasks.generate_many("info", 97, "main", k=2, p=0.0)  # 4 hosts x 24 row orders = 96


def test_answer_cap_grows_with_the_number_of_options():
    sent = []

    class Capture(ScriptedChat):
        def _chat(self, messages, schema, max_tokens, seed):
            if schema is not None:
                sent.append(max_tokens)
            return super()._chat(messages, schema, max_tokens, seed)

    for k in (2, 4):
        inst = tasks.generate_many("info", 1, "main", k=k, p=0.0)[0]
        run_team(inst, arms.build("baseline", "info"), LLMPolicy(Capture()), 1)
    assert sent[0] == ChatAdapter.answer_max_tokens  # 4 options
    assert sent[-1] == 16 * ANSWER_TOKENS_PER_OPTION  # 16 options


def test_design_is_part_of_new_run_ids_and_runs_of_two_designs_never_mix(tmp_path):
    base = run_id_for("i" * 64, "K", "medium", "m", 0, None)
    assert run_id_for("i" * 64, "K", "medium", "m", 0, None, None, 1) == base
    assert run_id_for("i" * 64, "K", "medium", "m", 0, None, None, 2) != base
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps({"run_id": base}) + "\n")  # an old record: design 1
    with pytest.raises(RuntimeError, match="arm design"):
        check_design(path)
    path.write_text(json.dumps({"run_id": base, "arm_design": arms.ARM_DESIGN}) + "\n")
    check_design(path)


def test_k_candidates_rise_in_sample_count_then_depth():
    # matching's binary search needs cost to rise along the list, and cost tracks sample count
    order = arms.DEPTH_ORDER
    for fam in ("reason", "comm_ip"):
        cfgs = [arms.build("K", fam, s) for s in arms.candidates("K", fam)]
        keys = [(c.samples, order.index(c.depth)) for c in cfgs]
        assert keys == sorted(keys), fam
