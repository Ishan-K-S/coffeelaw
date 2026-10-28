"""Design changes made after the free-tier pilot (PILOT_RESULTS.md, "Recommendations")."""

import json
from collections import Counter

import pytest

from ceilings import arms, calibration, tasks, theory
from ceilings.agents import LLMPolicy, make_policy
from ceilings.orchestrator import run_team
from tests.scripted import ScriptedChat


def test_host_table_order_is_shuffled_per_instance():
    firsts = set()
    for seed in range(20):
        inst = tasks.info.generate(seed, k=3, p=0.0)
        rows = [line.split(":")[0] for line in inst.shared_evidence.splitlines()
                if line.startswith("host-")]
        assert sorted(rows) == inst.answer_space  # every host once
        firsts.add(rows[0])
    assert len(firsts) > 3  # the first row is no longer always host-00


def test_known_attributes_and_chance():
    inst = tasks.info.generate(3, k=3, p=0.0, known=1)
    assert "Already known: the failing host's rack is" in inst.shared_evidence
    assert inst.theory.I_held_team == 1.0 and tasks.chance(inst) == pytest.approx(0.25)
    assert tasks.chance(tasks.info.generate(3, k=3, p=0.0)) == pytest.approx(0.125)
    with pytest.raises(ValueError):
        tasks.info.generate(3, k=3, known=3)


@pytest.mark.parametrize("family,diff", [("info", {"k": 3, "p": 0.0}), ("reason", {"d": 3}),
                                         ("comm_parity", {"n": 4})])
def test_true_labels_are_balanced(family, diff):
    insts = tasks.generate_many(family, 24, "main", **diff)
    counts = Counter(i.answer for i in insts)
    k = len(insts[0].answer_space)
    assert set(counts) == set(insts[0].answer_space)
    assert max(counts.values()) - min(counts.values()) <= (1 if 24 % k else 0)
    assert len({i.instance_id for i in insts}) == 24
    again = tasks.generate_many(family, 24, "main", **diff)
    assert [i.instance_id for i in again] == [i.instance_id for i in insts]  # deterministic


def test_strong_fano_bound():
    # k=3 (8 hosts), exact probes: 3, 2 and 1 bits left after T = 0, 1, 2 probes.
    assert theory.fano_floor_strong(3, 0, 8) == pytest.approx(0.875, abs=1e-6)
    assert theory.fano_floor_strong(3, 1, 8) == pytest.approx(0.373, abs=2e-3)
    assert theory.fano_floor_strong(3, 2, 8) == pytest.approx(0.144, abs=2e-3)
    assert theory.fano_floor_strong(3, 3, 8) == 0.0
    for b in (0.0, 0.5, 1.0, 1.5):
        assert theory.fano_floor_strong(3, b, 8) >= theory.fano_floor(3, b, 8) - 1e-9


def test_confirm_gate_is_strict_and_screen_is_not():
    # 30 instances: a baseline of 13/30 at chance 0.25 passes the screen but cannot confirm.
    assert calibration.gate_passes("screen", 13, 30, 30, 0.25)
    assert not calibration.gate_passes("confirm", 13, 30, 30, 0.25)
    # 300 instances at chance with a near-perfect fix confirm.
    assert calibration.gate_passes("confirm", 75, 296, 300, 0.25)
    # a fix whose lower bound is under 0.90 does not
    assert not calibration.gate_passes("confirm", 75, 275, 300, 0.25)
    with pytest.raises(ValueError):
        calibration.gate_passes("lenient", 1, 1, 2, 0.5)


def test_k_reasons_on_only_some_samples():
    inst = tasks.generate_many("reason", 1, "main", d=3)[0]
    out = run_team(inst, arms.build("K", "reason", "medium:1"), LLMPolicy(ScriptedChat()), 2)
    turns = [len(s["call"]["sent"]) for s in out.transcript if s["step"] == "answer"]
    assert turns == [2, 1, 1, 1]  # one sample reasons, three answer directly


def test_ties_are_broken_by_the_run_seed():
    inst = tasks.generate_many("comm_parity", 1, "main", n=2)[0]
    finals = {run_team(inst, arms.build("baseline", "comm_parity"), make_policy("sim-bounded"),
                       seed).final_answer for seed in range(40)}
    assert finals <= set(inst.answer_space)


def test_run_records_p_true_and_refuses_a_mismatched_table(tmp_path, capsys):
    from ceilings.cli import main
    from ceilings.runner import load_runs

    assert main(["--root", str(tmp_path), "prepare", "--model", "sim-bounded", "--family",
                 "comm_parity", "--n-main", "4", "--n-pilot", "4", "--delta", "medium",
                 "--no-match", "--gate", "screen"]) in (0, 2)
    assert main(["--root", str(tmp_path), "run", "--model", "sim-bounded", "--family",
                 "comm_parity", "--delta", "medium", "--arms", "baseline"]) == 0
    rec = load_runs(tmp_path)[0]
    assert rec["p_true"] is not None and 0 <= rec["p_true"] <= 1
    code = main(["--root", str(tmp_path), "run", "--model", "sim-bounded", "--family",
                 "comm_parity", "--delta", "medium", "--arms", "baseline", "--harness",
                 "temp=0"])
    assert code == 2 and "run prepare with the same --harness" in capsys.readouterr().err
    table = json.loads(next((tmp_path / "arms" / "comm_parity").glob("*.json")).read_text())
    assert table["harness"] == {}


def test_two_stage_gate_stage_one():
    # Nemotron's paid info gate: baseline 29/300 at chance 0.125 fails confirm, passes stage 1.
    assert not calibration.gate_passes("confirm", 29, 300, 300, 0.125)
    assert calibration.gate_passes("two_stage", 29, 300, 300, 0.125)
    assert not calibration.gate_passes("two_stage", 274, 274, 300, 0.25)  # fix too weak
    assert not calibration.gate_passes("two_stage", 120, 300, 300, 0.5)  # 0.40: too far


def _gate_files(root, base_correct, n_gate, chance):
    (root / "calibration" / "runs").mkdir(parents=True)
    (root / "calibration" / "comm_ip__m.json").write_text(json.dumps({
        "family_id": "comm_ip", "model_id": "m", "n": n_gate, "baseline_acc": 0.5,
        "chance_acc": chance, "matched_full_acc": 1.0, "passed": True, "gate": "two_stage"}))
    with open(root / "calibration" / "runs" / "comm_ip__m.jsonl", "w") as f:
        for i in range(n_gate):
            for arm in ("baseline", "M"):
                f.write(json.dumps({"instance_id": f"p{i}", "config": {"arm_id": arm},
                                    "correct": arm == "M" or i < base_correct}) + "\n")


def test_two_stage_gate_stage_two_pools_gate_and_main_baselines(tmp_path):
    import pandas as pd

    from ceilings.analysis.report import load_gates

    _gate_files(tmp_path, 147, 300, 0.5)
    # gate runs alone (n = 300, 0.49 at chance 0.5) cannot show equivalence
    detail: dict = {}
    assert load_gates(tmp_path, None, detail) == {("m", "comm_ip"): False}
    assert detail[("m", "comm_ip")]["pooled_n"] == 300
    # pooled with 850 main-grid baseline runs at chance it can; other arms and sweeps ignored
    rows = [{"model": "m", "family": "comm_ip", "arm": "baseline", "is_sweep": False,
             "instance_id": f"x{i}", "correct": i % 2 == 0} for i in range(850)]
    rows += [{"model": "m", "family": "comm_ip", "arm": "K", "is_sweep": False,
              "instance_id": f"x{i}", "correct": True} for i in range(850)]
    detail = {}
    assert load_gates(tmp_path, pd.DataFrame(rows), detail) == {("m", "comm_ip"): True}
    assert detail[("m", "comm_ip")]["pooled_n"] == 1150
    # a baseline well above chance in the main grid fails stage 2
    for r in rows[:850]:
        r["correct"] = True
    assert load_gates(tmp_path, pd.DataFrame(rows)) == {("m", "comm_ip"): False}
