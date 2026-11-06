"""H5a-H5c for the arm design 3 teams (prereg/DESIGN3_PLAN.md).

Each team arm is paired, instance by instance, with its own S_ref: the one agent matched to that
team's spend (ceilings.team_match). Per model, with Holm correction across its team arms:

- H5a (no team beats one agent that spends as much): contradicted if a team's exact one-sided
  McNemar superiority test rejects; supported if every team is shown to be no more than the
  margin better than its S_ref (the upper end of the paired 90% Newcombe interval below +margin,
  a one-sided test at 0.05); otherwise inconclusive.
- H5b (each team within +/- margin of its S_ref): supported if every team's TOST rejects;
  contradicted if a team's 95% interval lies wholly outside +/- margin; otherwise inconclusive.

The matching instances (the table's `instances`) are left out of every test, since they chose the
S_ref strengths. A team whose S_ref is unmatched (more than the tolerance away in cost) is tested and reported, but
left out of the verdicts. A hypothesis needs MIN_MODELS agreeing evaluable models.

H5c (descriptive): how often the team's answer agrees with a wrong claim about the target that a
teammate sent (in a report, note, solution or review), next to the team's deficit to its S_ref.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

from ceilings.analysis.equivalence import holm, newcombe_paired_ci, superiority, tost

ALPHA = 0.05
MARGIN = 0.05
MIN_MODELS = 2
TEAM_ORDER = ("L", "W", "R", "W0")
SHIFTS = "ABCD"
CLAIM_RE = re.compile(r"\b([A-Z][a-z]+)(?:'s shift)?\s*(?:works(?: on)?|is on|is in|is at|is|=|:|->)"
                      r"\s*(?:shift\s*)?([ABCD])\b")


def strength_of(summary: dict) -> str:
    """A run's S_ref strength, as team_match.LADDER names it."""
    n = summary.get("samples", 1)
    return summary["depth"] if n == 1 else f"{summary['depth']}:{n:g}"


def not_better_p(arm, ref, margin: float = MARGIN) -> float:
    """One-sided p-value that the arm is no more than `margin` better than the reference: the
    smallest alpha at which the upper end of the paired (1 - 2 alpha) interval is below margin."""
    def below(alpha: float) -> bool:
        return newcombe_paired_ci(arm, ref, 1 - 2 * alpha)[2] < margin

    if not below(0.5 - 1e-9):
        return 1.0
    lo, hi = 1e-12, 0.5 - 1e-9
    if below(lo):
        return lo
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if below(mid) else (mid, hi)
    return float(hi)


def _combine(per_model: dict[str, str]) -> str:
    evaluable = [v for v in per_model.values() if v != "not evaluable"]
    for v in ("contradicted", "supported"):
        if evaluable.count(v) >= MIN_MODELS:
            return v
    return "not evaluable" if not evaluable else "inconclusive"


def team_tests(records: list[dict], tables: dict[str, dict]) -> dict:
    """Paired tests of every team arm against its S_ref, per model, and the H5a/H5b verdicts."""
    rows, h5a, h5b = [], {}, {}
    for model, table in sorted(tables.items()):
        # the matching instances chose the S_ref strengths, so they never enter the tests
        held = set(table.get("instances", []))
        recs = [r for r in records if r["model_id"] == model and r.get("team")
                and r["instance_id"] not in held]
        ref = {(strength_of(r["team"]), r["instance_id"]): float(r["correct"])
               for r in recs if r["arm_id"] == "S_ref"}
        tokens = {(r["arm_id"] if r["arm_id"] != "S_ref" else strength_of(r["team"]),
                   r["instance_id"]): r["ledger"]["total_tokens"] for r in recs}
        model_rows = []
        for arm in TEAM_ORDER:
            if arm not in table["arms"]:
                continue
            m = table["arms"][arm]
            team = {r["instance_id"]: float(r["correct"]) for r in recs if r["arm_id"] == arm}
            ids = sorted(i for i in team if (m["strength"], i) in ref)
            if len(ids) < 2:
                continue
            a = np.array([team[i] for i in ids])
            b = np.array([ref[(m["strength"], i)] for i in ids])
            d, lo, hi = newcombe_paired_ci(a, b, 0.95)
            model_rows.append({
                "model": model, "arm": arm, "ref": m["strength"], "n": len(ids),
                "team_acc": float(a.mean()), "ref_acc": float(b.mean()), "diff": d,
                "ci95": [lo, hi], "matched": m["matched"], "cost_gap": m["gap"],
                # the same gap on these instances (reported; verdicts use the matching gap)
                "realized_cost_gap": float(np.mean([tokens[(m["strength"], i)] for i in ids])
                                           / np.mean([tokens[(arm, i)] for i in ids]) - 1),
                "p_superiority": superiority(a, b), "p_not_better": not_better_p(a, b),
                "p_tost": tost(a, b, MARGIN)})
        tested = [r for r in model_rows if r["matched"]]
        for key in ("p_superiority", "p_not_better", "p_tost"):
            for r, p in zip(tested, holm([r[key] for r in tested]), strict=True):
                r[f"{key}_holm"] = p
        if not tested:
            h5a[model] = h5b[model] = "not evaluable"
        else:
            if any(r["p_superiority_holm"] < ALPHA for r in tested):
                h5a[model] = "contradicted"
            elif all(r["p_not_better_holm"] < ALPHA for r in tested):
                h5a[model] = "supported"
            else:
                h5a[model] = "inconclusive"
            if all(r["p_tost_holm"] < ALPHA for r in tested):
                h5b[model] = "supported"
            elif any(r["ci95"][0] > MARGIN or r["ci95"][1] < -MARGIN for r in tested):
                h5b[model] = "contradicted"
            else:
                h5b[model] = "inconclusive"
        rows += model_rows
    return {"tests": rows,
            "H5a": {"verdict": _combine(h5a), "per_model": h5a},
            "H5b": {"verdict": _combine(h5b), "per_model": h5b},
            "alpha": ALPHA, "margin": MARGIN, "min_models": MIN_MODELS}


def chain_shifts(public: dict) -> dict[str, str]:
    """Every person's true shift, by walking the chain from its start."""
    person, shift = public["start"][0], int(public["start"][1])
    child_of = {parent: (child, int(off)) for child, parent, off in public["edges"]}
    out = {person: SHIFTS[shift]}
    while person in child_of:
        person, off = child_of[person]
        shift = (shift + off) % 4
        out[person] = SHIFTS[shift]
    return out


def claims(text: str, people: set[str]) -> list[tuple[str, str]]:
    """(person, shift) statements in free text, e.g. "Faye works shift C", "Juno=D"."""
    return [(p, s) for p, s in CLAIM_RE.findall(text or "") if p in people]


def mechanism(records: list[dict], index: dict, tests: list[dict],
              held: dict[str, set] | None = None) -> list[dict]:
    """H5c: per model and team arm, how teammates' claims about the target fared. `held` maps a
    model to its matching instances, which are left out here as in the tests."""
    deficit = {(t["model"], t["arm"]): t["ref_acc"] - t["team_acc"] for t in tests}
    held = held or {}
    acc: dict[tuple[str, str], dict] = {}
    for r in records:
        if r["arm_id"] not in TEAM_ORDER or not r.get("transcript_ref"):
            continue
        if r["instance_id"] in held.get(r["model_id"], ()):
            continue
        inst = index.get(r["instance_id"])
        ref = Path(r["transcript_ref"])
        if inst is None or not ref.exists():
            continue
        truth = chain_shifts(inst.payload["public"])
        target = inst.payload["public"]["target"]
        c = acc.setdefault((r["model_id"], r["arm_id"]),
                           {"texts": 0, "with_claims": 0, "claims": 0, "claims_right": 0,
                            "wrong_target": 0, "accepted": 0})
        for s in json.loads(ref.read_text())["steps"]:
            if s["step"] != "deliver" or s["kind"] == "brief":
                continue
            found = claims(s["text"], set(truth))
            c["texts"] += 1
            c["with_claims"] += bool(found)
            c["claims"] += len(found)
            c["claims_right"] += sum(truth[p] == v for p, v in found)
            wrong = [v for p, v in found if p == target and v != truth[p]]
            if wrong:
                c["wrong_target"] += 1
                c["accepted"] += r["final_answer"] in wrong
    rows = []
    for (model, arm), c in sorted(acc.items()):
        rows.append({
            "model": model, "arm": arm, "texts": c["texts"],
            "coverage": c["with_claims"] / c["texts"] if c["texts"] else None,
            "claim_accuracy": c["claims_right"] / c["claims"] if c["claims"] else None,
            "wrong_target_texts": c["wrong_target"],
            "accepted_share": c["accepted"] / c["wrong_target"] if c["wrong_target"] else None,
            "deficit": deficit.get((model, arm))})
    return rows


def spearman(rows: list[dict]) -> dict | None:
    """Rank correlation of the deficit with the accepted share across model x arm cells."""
    pts = [(r["accepted_share"], r["deficit"]) for r in rows
           if r["accepted_share"] is not None and r["deficit"] is not None]
    if len(pts) < 4:
        return None
    rho, p = stats.spearmanr(*zip(*pts, strict=True))
    return {"rho": float(rho), "p": float(p), "cells": len(pts)}


__all__ = ["team_tests", "mechanism", "spearman", "chain_shifts", "claims", "strength_of",
           "not_better_p"]
