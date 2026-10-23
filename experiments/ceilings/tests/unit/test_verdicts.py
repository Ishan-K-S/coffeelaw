"""Verdict rules on synthetic run tables: gates, sample size, falsification conditions."""

import numpy as np
import pandas as pd

from ceilings.analysis.report import NOT_EVALUABLE, assign_verdicts, threshold_verdict

ARMS = ["baseline", "E", "K", "M", "A1", "P1", "P2"]


def _frame(model, family, acc_by_arm, n, seed=0):
    """Rows for one model/family: each arm's correctness drawn at the given accuracy; arms not
    listed copy the baseline outcomes exactly (zero difference)."""
    rng = np.random.default_rng(seed)
    base = rng.random(n) < acc_by_arm.get("baseline", 0.5)
    rows = []
    for arm in ARMS:
        if arm in acc_by_arm and arm != "baseline":
            corr = rng.random(n) < acc_by_arm[arm]
        else:
            corr = base
        for i in range(n):
            rows.append({"instance_id": f"{family}-{i}", "family": family, "arm": arm,
                         "delta": "none" if arm == "baseline" else "medium", "model": model,
                         "correct": bool(corr[i]), "valid": True, "tokens": 100,
                         "is_sweep": False, "replicate": 0})
    return rows


def _df(n, families=("info", "reason", "comm_parity", "comm_ip"), models=("m1", "m2"),
        matching_acc=0.95):
    match = {"info": "E", "reason": "K", "comm_parity": "M", "comm_ip": "M"}
    rows = []
    for m in models:
        for j, fam in enumerate(families):
            rows += _frame(m, fam, {"baseline": 0.3, match[fam]: matching_acc}, n, seed=j)
    return pd.DataFrame(rows)


def _verdicts(df, **kw):
    preds, _ = assign_verdicts(df, **kw)
    return {p.prediction_id: p.verdict for p in preds}


def test_clean_diagonal_is_supported_with_two_models():
    v = _verdicts(_df(900), min_models=2)
    assert v["H1"] == v["H2"] == v["H3"] == "supported", v


def test_one_model_cannot_decide_under_the_preregistered_rule():
    v = _verdicts(_df(900, models=("m1",)), min_models=2)
    assert v["H1"] == "inconclusive"


def test_matching_arm_that_fails_is_never_a_contradiction():
    """Regression: H3 was 'contradicted' when M simply did not help (all-zero differences)."""
    v = _verdicts(_df(8, matching_acc=0.3), min_models=1)
    assert v["H3"] != "contradicted" and v["H1"] != "contradicted", v


def test_failed_gate_makes_a_model_not_evaluable():
    gates = {("m1", "comm_ip"): False, ("m2", "comm_ip"): False}
    preds, _ = assign_verdicts(_df(900), gates=gates, min_models=2)
    h3 = next(p for p in preds if p.prediction_id == "H3")
    assert h3.verdict == NOT_EVALUABLE
    assert "calibration gate failed for comm_ip" in h3.detail["reasons"]["m1"]


def test_too_few_instances_is_not_evaluable():
    assert _verdicts(_df(50), min_n=850, min_models=1)["H1"] == NOT_EVALUABLE


def test_k_helping_on_info_contradicts_h1():
    rows = []
    for m in ("m1", "m2"):
        rows += _frame(m, "info", {"baseline": 0.2, "E": 0.95, "K": 0.6}, 900, seed=1)
    v = _verdicts(pd.DataFrame(rows), min_models=2)
    assert v["H1"] == "contradicted"


def _sweep(fam, acc_by_L, n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for L, acc in acc_by_L.items():
        corr = rng.random(n) < acc
        rows += [{"instance_id": f"{fam}-{i}", "family": fam, "model": "m", "is_sweep": True,
                  "L": float(L), "correct": bool(c)} for i, c in enumerate(corr)]
    return rows


def test_flat_sweep_is_not_contradiction():
    """Regression: a flat line at n=4 was read as 'steadily rising'."""
    rows = _sweep("comm_ip", {0: 0.75, 1: 0.75, 2: 0.75, 4: 0.75}, n=4)
    rows += _sweep("comm_parity", {0: 0.75, 1: 0.75, 2: 0.75, 4: 0.75}, n=4)
    verdict, _ = threshold_verdict(pd.DataFrame(rows), min_models=1)
    assert verdict != "contradicted"


def test_threshold_and_parity_jump_supported():
    rows = _sweep("comm_ip", {0: 0.5, 2: 0.5, 4: 0.5, 6: 0.55, 8: 0.97, 10: 0.98}, n=80)
    rows += _sweep("comm_parity", {0: 0.5, 1: 0.97, 2: 0.97, 4: 0.98}, n=80)
    verdict, detail = threshold_verdict(pd.DataFrame(rows), min_models=1)
    assert verdict == "supported", detail


def test_steady_rise_contradicts_h7():
    rows = _sweep("comm_ip", {L: 0.6 + 0.04 * L for L in range(0, 10, 2)}, n=200)
    rows += _sweep("comm_parity", {0: 0.5, 1: 0.97, 2: 0.97, 4: 0.98}, n=80)
    verdict, detail = threshold_verdict(pd.DataFrame(rows), min_models=1)
    assert verdict == "contradicted", detail


def test_h7_needs_two_models_to_agree():
    rows = _sweep("comm_ip", {0: 0.5, 2: 0.5, 4: 0.5, 6: 0.55, 8: 0.97, 10: 0.98}, n=80)
    rows += _sweep("comm_parity", {0: 0.5, 1: 0.97, 2: 0.97, 4: 0.98}, n=80)
    one = pd.DataFrame(rows)
    assert threshold_verdict(one)[0] == "inconclusive"  # pre-registered: 2 models must agree
    two = pd.concat([one, one.assign(model="m2")])
    assert threshold_verdict(two)[0] == "supported"


def test_h6_judges_each_model_and_needs_two_to_agree():
    from ceilings.analysis.report import fano_verdict

    def cells(model, errors, n=60):
        return [{"model": model, "family": "info", "is_sweep": True, "T": 2, "flip_rate": 0.0,
                 "correct": i >= errors} for i in range(n)]

    floors = {(2, 0.0): 0.5}
    below = pd.DataFrame(cells("m1", 5))  # 8% error against a 50% floor
    assert fano_verdict(below, floors)[0] == "inconclusive"
    assert fano_verdict(pd.concat([below, pd.DataFrame(cells("m2", 4))]), floors)[0] == \
        "contradicted"
    at_floor = pd.DataFrame(cells("m1", 30) + cells("m2", 31))
    assert fano_verdict(at_floor, floors)[0] == "supported"
