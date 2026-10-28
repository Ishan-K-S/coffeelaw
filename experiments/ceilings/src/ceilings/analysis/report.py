"""Verdicts for the pre-registered predictions (FR-020) and report files.

Rules (spec, "Pre-registered Predictions and Falsification Conditions"):

- Each (model, family) is *evaluable* only if its calibration gate passed, every arm the hypothesis
  needs is present, and each test has at least `min_n` paired instances. Otherwise that model's
  verdict is "not evaluable", with the reason recorded.
- Support needs every pre-registered test of the hypothesis to succeed after Holm correction
  (one Holm family per model for the support tests).
- Contradiction needs the hypothesis's own falsification condition, also Holm-adjusted (a second
  Holm family per model for the falsification tests). A matching arm that fails to help is never a
  contradiction; it leaves the hypothesis inconclusive.
- Across models: a verdict needs at least `min_models` evaluable models agreeing (pre-registered:
  2 of 3). Otherwise "inconclusive", or "not evaluable" if no model was evaluable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ceilings.analysis import changepoint
from ceilings.analysis.equivalence import holm, superiority, tost
from ceilings.analysis.gains import gain_table, paired_diffs
from ceilings.records import Prediction

ALPHA = 0.05
NOT_EVALUABLE = "not evaluable"
FAMILY_ARM = {"info": "E", "reason": "K", "comm_parity": "M", "comm_ip": "M"}
HYPOTHESES = {
    # id: (families, matching arm, arms that must not help)
    "H1": (["info"], "E", ["K", "M", "A1", "P1", "P2"]),
    "H2": (["reason"], "K", ["E", "M", "A1", "P1", "P2"]),
    "H3": (["comm_parity", "comm_ip"], "M", ["K", "E", "A1", "P1", "P2"]),
}
STATEMENTS = {
    "H1": ("On information-bound tasks, E helps; K, M, A1, P1 and P2 do not", "Thm 1, Cor 1"),
    "H2": (
        "On reasoning-bound tasks, K helps; E (restated evidence), M, A1, P1, P2 do not",
        "Thm 2, Cor 3",
    ),
    "H3": ("On communication-bound tasks, M helps; K, same-kind E, A1, P1, P2 do not", "Thm 5"),
    "H4": ("A1 (more agents, same totals) gives no gain on any family", "Cor 1, Thm 3"),
    "H5": ("A2 gain equals the gain from the same compute given to one agent", "Thm 3"),
    "H6": ("Information-bound error never falls significantly below the Fano floor", "Cor 2"),
    "H7": ("Inner-product splits stay flat then rise; parity splits jump at 1 bit", "Thm 5"),
}


def _pair(df, model, fam, arm, delta, reference="baseline", ref_delta="none"):
    p = paired_diffs(df, model, fam, arm, delta, reference, ref_delta)
    return p["arm"].to_numpy(), p["ref"].to_numpy()


def _unique_tests(df: pd.DataFrame, model: str, delta: str, margin: float) -> dict:
    """Each (family, arm, reference) comparison computed once; hypotheses refer to them by key."""
    out = {}
    fams = set(df[(df["model"] == model) & ~df["is_sweep"]]["family"])
    arms_needed = {"E", "K", "M", "A1", "P1", "P2"}
    for fam in fams:
        for arm in arms_needed:
            a, r = _pair(df, model, fam, arm, delta)
            if len(a) >= 2:
                out[(fam, arm, "baseline")] = {
                    "family": fam, "arm": arm, "reference": "baseline", "n": len(a),
                    "gain": float((a - r).mean()), "arm_acc": float(a.mean()),
                    "p_sup": superiority(a, r), "p_equiv": tost(a, r, margin)}
        a, r = _pair(df, model, fam, "A2", delta, "A2_single_ref", delta)
        if len(a) >= 2:
            out[(fam, "A2", "A2_single_ref")] = {
                "family": fam, "arm": "A2", "reference": "A2_single_ref", "n": len(a),
                "gain": float((a - r).mean()), "arm_acc": float(a.mean()),
                "p_sup": superiority(a, r), "p_equiv": tost(a, r, margin)}
    return out


def _plan(hid: str, fams_present: set) -> tuple[list, list]:
    """(support tests, falsification tests) as lists of (key, which p, family)."""
    support, falsify = [], []
    if hid in HYPOTHESES:
        fams, match, nots = HYPOTHESES[hid]
        for fam in fams:
            support.append(((fam, match, "baseline"), "p_sup"))
            support += [((fam, arm, "baseline"), "p_equiv") for arm in nots]
        if hid == "H1":
            falsify.append(((("info", "K", "baseline")), "p_sup"))
        elif hid == "H2":
            falsify.append(((("reason", "E", "baseline")), "p_sup"))
        else:
            falsify.append(((("comm_ip", "K", "baseline")), "p_sup"))
    elif hid == "H4":
        for fam in sorted(fams_present):
            support.append(((fam, "A1", "baseline"), "p_equiv"))
            falsify.append(((fam, "A1", "baseline"), "p_sup"))
    elif hid == "H5":
        for fam in sorted(fams_present):
            support.append(((fam, "A2", "A2_single_ref"), "p_equiv"))
            falsify.append(((fam, "A2", "A2_single_ref"), "p_sup"))
    return support, falsify


def _model_verdicts(df, model, delta, margin, gates, min_n) -> tuple[dict, list]:
    tests = _unique_tests(df, model, delta, margin)
    fams_present = {k[0] for k in tests}
    plans = {h: _plan(h, fams_present) for h in ["H1", "H2", "H3", "H4", "H5"]}
    # One Holm family for support tests, one for falsification tests (each comparison once).
    sup_keys = sorted({(k, w) for s, _ in plans.values() for k, w in s if k in tests})
    fal_keys = sorted({(k, w) for _, f in plans.values() for k, w in f if k in tests})
    sup_adj = dict(zip(sup_keys, holm([tests[k][w] for k, w in sup_keys]), strict=True))
    fal_adj = dict(zip(fal_keys, holm([tests[k][w] for k, w in fal_keys]), strict=True))
    verdicts, rows = {}, []
    for hid, (support, falsify) in plans.items():
        reasons = []
        if not support:
            reasons.append("no families with data")
        for key, _ in support + falsify:
            fam = key[0]
            if gates.get((model, fam)) is False:
                reasons.append(f"calibration gate failed for {fam}")
            if key not in tests:
                reasons.append(f"missing arm {key[1]} vs {key[2]} on {fam}")
            elif tests[key]["n"] < min_n:
                reasons.append(f"n={tests[key]['n']} < {min_n} for {key[1]} on {fam}")
        if reasons:
            verdicts[hid] = (NOT_EVALUABLE, sorted(set(reasons)))
            continue
        contradicted = [k for k, w in falsify
                        if fal_adj[(k, w)] < ALPHA and tests[k]["gain"] > margin]
        if hid == "H2" and contradicted:
            # Falsified only if restated evidence helps as much as K and P1 does not explain it.
            e, k, p1 = (tests[("reason", a, "baseline")] for a in ("E", "K", "P1"))
            if not (e["gain"] >= k["gain"] and p1["p_sup"] >= ALPHA):
                contradicted = []
        if contradicted:
            verdicts[hid] = ("contradicted", [f"{k[1]} vs {k[2]} on {k[0]}" for k in contradicted])
        elif all(sup_adj[(k, w)] < ALPHA for k, w in support):
            verdicts[hid] = ("supported", [])
        else:
            failed = [f"{k[1]} on {k[0]} ({w})" for k, w in support if sup_adj[(k, w)] >= ALPHA]
            verdicts[hid] = ("inconclusive", failed)
    for (k, w), p in list(sup_adj.items()) + list(fal_adj.items()):
        rows.append({"model": model, **tests[k], "test": w,
                     "p_adj": p, "holm_family": "support" if (k, w) in sup_adj else "falsify"})
    return verdicts, rows


def _combine(per_model: dict[str, str], min_models: int) -> str:
    evaluable = [v for v in per_model.values() if v != NOT_EVALUABLE]
    if not evaluable:
        return NOT_EVALUABLE
    for v in ("contradicted", "supported"):
        if evaluable.count(v) >= min_models:
            return v
    return "inconclusive"


def fano_verdict(df: pd.DataFrame, floors: dict, gates: dict | None = None,
                 min_power_p: float = ALPHA, min_models: int = 2) -> tuple[str, list]:
    """H6: in each (T, p) cell with a positive floor, is the error significantly below it?

    A cell counts only if it could reject at all: with zero errors its p-value must be below
    `min_power_p`. Each model is judged on its own cells (Holm within the model): contradicted if
    any powered cell is significantly below the floor, supported if none is, inconclusive without
    a powered cell, not evaluable if its info gate failed. Models are then combined like H1-H5
    (`min_models` evaluable models must agree).
    """
    cells = []
    if "T" not in df.columns:
        return "inconclusive", cells
    sweep = df[df["is_sweep"] & (df["family"] == "info")]
    for (model, T, p), g in sweep.groupby(["model", "T", "flip_rate"]):
        floor = floors.get((int(T), float(p)), 0.0)
        errors, n = int((~g["correct"]).sum()), len(g)
        best_p = (1 - floor) ** n if floor > 0 else 1.0
        cells.append({"model": model, "T": int(T), "p": float(p), "n": n, "error": errors / n,
                      "floor": floor, "powered": floor > 0 and best_p < min_power_p,
                      "p_below": float(stats.binomtest(errors, n, floor, alternative="less")
                                       .pvalue) if floor > 0 else None})
    per_model = {}
    for model in sorted({c["model"] for c in cells}):
        tested = [c for c in cells if c["model"] == model and c["powered"]]
        for c, a in zip(tested, holm([c["p_below"] for c in tested]), strict=True):
            c["p_adj"] = a
        if gates and gates.get((model, "info")) is False:
            per_model[model] = NOT_EVALUABLE
        elif any(c["p_adj"] < ALPHA for c in tested):
            per_model[model] = "contradicted"
        else:
            per_model[model] = "supported" if tested else "inconclusive"
    for c in cells:
        c["model_verdict"] = per_model[c["model"]]
    if not per_model:
        return "inconclusive", cells
    return _combine(per_model, min_models), cells


def threshold_verdict(df: pd.DataFrame, gates: dict | None = None,
                      min_models: int = 2) -> tuple[str, dict]:
    """H7 from the bit-mode message sweeps.

    Supported: for inner product, the threshold model beats the linear one on held-out instances
    and L=0 sits near chance; for parity, accuracy at L=1 reaches 90% of its ceiling and that
    ceiling is clearly above chance. Contradicted only on positive evidence of a steady rise for
    inner product (the pre-registered falsifier): accuracy significantly above chance already at
    the smallest positive budget, and a significant positive trend in L. Anything else is
    inconclusive. "Supported" also requires the smallest positive budget not to be above chance,
    so the two outcomes cannot both hold.
    """
    detail: dict = {}
    if "L" not in df.columns:
        return "inconclusive", detail
    sweep = df[df["is_sweep"] & df["family"].isin(["comm_ip", "comm_parity"])]
    outcome = {}
    for (model, fam), g in sweep.groupby(["model", "family"]):
        if gates and gates.get((model, fam)) is False:
            detail[f"{model}/{fam}"] = {"verdict": NOT_EVALUABLE, "reason": "gate failed"}
            outcome[(model, fam)] = NOT_EVALUABLE
            continue
        L = g["L"].to_numpy(dtype=float)
        y = g["correct"].to_numpy(dtype=float)
        acc = {float(k): float(v) for k, v in g.groupby("L")["correct"].mean().items()}
        n_by_L = g.groupby("L").size()
        if fam == "comm_ip":
            res = changepoint.compare(L, y, groups=g["instance_id"].to_numpy())
            slope_p = changepoint.slope_pvalue(L, y)
            positive = sorted(x for x in acc if x > 0)
            first = g[g["L"] == positive[0]]["correct"] if positive else g.iloc[0:0]["correct"]
            above_chance_1 = len(first) > 0 and stats.binomtest(
                int(first.sum()), len(first), 0.5, alternative="greater").pvalue < ALPHA
            near_chance_0 = abs(acc.get(0.0, 0.5) - 0.5) <= 0.15
            if res["threshold_wins"] and near_chance_0 and not above_chance_1:
                v = "supported"
            elif slope_p < ALPHA and above_chance_1:
                v = "contradicted"
            else:
                v = "inconclusive"
            detail[f"{model}/comm_ip"] = {**res, "slope_p": slope_p, "accuracy_by_L": acc,
                                          "n_by_L": n_by_L.to_dict(), "verdict": v}
        else:
            ceiling = acc[max(acc)]
            jump = acc.get(1.0, 0.0)
            v = ("supported" if ceiling >= 0.7 and jump >= 0.9 * ceiling else "inconclusive")
            detail[f"{model}/comm_parity"] = {"accuracy_by_L": acc, "acc_L1": jump,
                                              "ceiling": ceiling, "verdict": v}
        outcome[(model, fam)] = v
    models = {m for m, _ in outcome}
    per_model = {}
    for m in models:
        ip, par = outcome.get((m, "comm_ip")), outcome.get((m, "comm_parity"))
        if NOT_EVALUABLE in (ip, par) or ip is None or par is None:
            per_model[m] = NOT_EVALUABLE
        elif ip == "contradicted":
            per_model[m] = "contradicted"
        elif ip == "supported" and par == "supported":
            per_model[m] = "supported"
        else:
            per_model[m] = "inconclusive"
    detail["per_model"] = per_model
    if not per_model:
        return "inconclusive", detail
    return _combine(per_model, min_models), detail


def assign_verdicts(
    df: pd.DataFrame,
    delta: str = "medium",
    margin: float = 0.05,
    floors: dict | None = None,
    gates: dict | None = None,
    min_n: int = 0,
    min_models: int = 2,
) -> tuple[list[Prediction], list[dict]]:
    """`gates` maps (model, family) to the calibration gate result; `min_n` is the
    pre-registered paired instances per cell; `min_models` is how many evaluable models must
    agree (pre-registered: 2)."""
    gates = gates or {}
    all_rows: list[dict] = []
    per_h: dict[str, dict[str, str]] = defaultdict(dict)
    reasons: dict[str, dict[str, list]] = defaultdict(dict)
    for model in sorted(set(df[~df["is_sweep"]]["model"])):
        verdicts, rows = _model_verdicts(df, model, delta, margin, gates, min_n)
        all_rows += rows
        for hid, (v, why) in verdicts.items():
            per_h[hid][model] = v
            reasons[hid][model] = why
    preds = []
    for hid in ["H1", "H2", "H3", "H4", "H5"]:
        statement, source = STATEMENTS[hid]
        verdict = _combine(per_h[hid], min_models) if per_h[hid] else NOT_EVALUABLE
        preds.append(Prediction(
            prediction_id=hid, statement=statement, paper_source=source,
            test="exact McNemar superiority + Newcombe TOST, Holm-adjusted",
            verdict=verdict,
            detail={"per_model": per_h[hid], "reasons": reasons[hid], "min_n": min_n,
                    "min_models": min_models}))
    v6, cells = fano_verdict(df, floors or {}, gates, min_models=min_models)
    per6 = {c["model"]: c["model_verdict"] for c in cells}
    why6 = (["calibration gate failed for info for every model with sweep runs"]
            if v6 == NOT_EVALUABLE else
            ["no cell with a positive floor and enough runs to reject"]
            if v6 == "inconclusive" and not any(c.get("powered") for c in cells) else
            [f"fewer than {min_models} evaluable models agree: {per6}"]
            if v6 == "inconclusive" else [])
    preds.append(Prediction(prediction_id="H6", statement=STATEMENTS["H6"][0],
                            paper_source=STATEMENTS["H6"][1],
                            test="binomial floor check on powered cells, Holm",
                            verdict=v6, detail={"cells": cells, "per_model": per6,
                                                "reasons": {"all": why6},
                                                "min_models": min_models}))
    v7, detail = threshold_verdict(df, gates, min_models=min_models)
    detail["reasons"] = {f"{m}/{f}": ["calibration gate failed"]
                         for (m, f), ok in gates.items()
                         if f in ("comm_ip", "comm_parity") and ok is False}
    preds.append(Prediction(prediction_id="H7", statement=STATEMENTS["H7"][0],
                            paper_source=STATEMENTS["H7"][1],
                            test="change-point vs linear (instance-grouped CV) + slope test",
                            verdict=v7, detail=detail))
    return preds, all_rows


def load_gates(root: Path | str, df: pd.DataFrame | None = None,
               detail: dict | None = None) -> dict:
    """(model, family) -> calibration gate result, from `<root>/calibration/*.json`; a family
    whose difficulty search (`<root>/difficulty/*.json`) found no passing setting, and was
    therefore never prepared, counts as a failed gate.

    A "two_stage" gate passes only if stage 2 does too: the baseline equivalent to chance
    (`calibration.baseline_equivalent`) on the gate's baseline runs pooled with the main grid's
    baseline runs in `df`. Without `df` it is not yet decided and counts as failed. `detail`, if
    given, is filled with each gate's numbers."""
    from ceilings.calibration import baseline_equivalent

    gates = {}
    for path in Path(root).glob("difficulty/*.json"):
        if path.name.endswith(".settings.json"):
            continue
        d = json.loads(path.read_text())
        if d.get("chosen") is None:
            gates[(d["model_id"], d["family"])] = False
            if detail is not None:
                detail[(d["model_id"], d["family"])] = {"gate": "search", "passed": False}
    for path in Path(root).glob("calibration/*.json"):
        d = json.loads(path.read_text())
        key = (d["model_id"], d["family_id"])
        passed = bool(d["passed"])
        info = {"gate": d.get("gate", "screen"), "stage1_passed": passed, "n_gate": d["n"],
                "baseline_acc": d["baseline_acc"], "chance": d["chance_acc"],
                "full_acc": d["matched_full_acc"]}
        if d.get("gate") == "two_stage":
            correct, n = _pooled_baseline(root, path.stem, df, key)
            stage2 = n > 0 and baseline_equivalent(correct, n, d["chance_acc"])
            info.update(pooled_correct=correct, pooled_n=n, stage2_passed=bool(stage2))
            passed = passed and bool(stage2)
        gates[key] = passed
        if detail is not None:
            detail[key] = {**info, "passed": passed}
    return gates


def _pooled_baseline(root, stem: str, df: pd.DataFrame | None, key) -> tuple[int, int]:
    """Correct count and n over the gate's baseline runs and the main grid's (one per instance)."""
    runs = Path(root) / "calibration" / "runs" / f"{stem}.jsonl"
    seen: dict[str, bool] = {}
    if runs.exists():
        for line in runs.read_text().splitlines():
            r = json.loads(line)
            if r["config"]["arm_id"] == "baseline":
                seen.setdefault("gate:" + r["instance_id"], bool(r["correct"]))
    if df is not None and not df.empty:
        main = df[(df.model == key[0]) & (df.family == key[1]) & (df.arm == "baseline")
                  & ~df.is_sweep]
        for iid, c in main.groupby("instance_id").correct.first().items():
            seen.setdefault("main:" + iid, bool(c))
    return sum(seen.values()), len(seen)


def write_reports(
    df: pd.DataFrame,
    out: Path | str,
    delta: str = "medium",
    floors: dict | None = None,
    n_boot: int = 10_000,
    gates: dict | None = None,
    min_n: int = 0,
    min_models: int = 2,
) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = gain_table(df, n_boot=n_boot)
    table.to_csv(out / "gain_table.csv", index=False)
    preds, tests = assign_verdicts(df, delta=delta, floors=floors, gates=gates, min_n=min_n,
                                   min_models=min_models)
    (out / "predictions.json").write_text(
        json.dumps([p.model_dump() for p in preds], indent=2, default=_json_default)
    )
    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)
    return {"gain_table": table, "predictions": preds, "tests": tests}


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    return str(o)
