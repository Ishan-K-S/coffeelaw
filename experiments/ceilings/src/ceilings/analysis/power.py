"""Sample size per cell from pilot runs (FR-017, research R10).

For each comparison the verdicts use (the same plan as `report._plan`), the pilot gives the paired
discordance rate pi_d = P(arm != reference) and the mean difference d. The required number of
paired instances is then:

- equivalence (TOST at margin m), planned under a true difference of 0:
  n = ((z_{1-a} + z_{1-b/2}) * sigma_d / m)^2 with sigma_d^2 = pi_d,
  which gives the planning figure of about 850 at pi_d = 0.24 (research R10);
- superiority (one-sided exact McNemar), planned at the pilot's own gain d > 0 (Connor 1987):
  n = (z_{1-a} sqrt(pi_d) + z_{1-b} sqrt(pi_d - d^2))^2 / d^2.

Holm needs the smallest p-value in a family below a/k, so `a` is planned as a/k per Holm family
(Bonferroni; conservative). With a pilot this small, pi_d and d are themselves noisy, so the
result is a planning figure to pre-register, not a guarantee.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ceilings.analysis.gains import paired_diffs
from ceilings.analysis.report import _plan, _unique_tests

PI_D_FLOOR = 0.02  # a pilot with no discordant pairs still leaves room for some


def n_equivalence(pi_d: float, margin: float, alpha: float, power: float) -> float:
    sd = math.sqrt(max(pi_d, PI_D_FLOOR))
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - (1 - power) / 2)
    return math.ceil((z * sd / margin) ** 2)


def n_superiority(pi_d: float, gain: float, alpha: float, power: float) -> float:
    if gain <= 0:
        return math.inf
    pi_d = max(pi_d, gain, PI_D_FLOOR)
    za, zb = stats.norm.ppf(1 - alpha), stats.norm.ppf(power)
    return math.ceil((za * math.sqrt(pi_d) + zb * math.sqrt(max(pi_d - gain**2, 0.0))) ** 2
                     / gain**2)


def sample_sizes(df: pd.DataFrame, delta: str = "medium", margin: float = 0.05,
                 alpha: float = 0.05, power: float = 0.8, gates: dict | None = None) -> dict:
    """Per model: one row per planned comparison, and the per-cell n the equivalence tests need
    (max over families whose gate passed). Superiority rows are reported but not maximized over,
    since a matching arm with a small pilot gain would otherwise demand an unbounded n."""
    gates = gates or {}
    out = {"delta": delta, "margin": margin, "alpha": alpha, "power": power, "models": {}}
    for model in sorted(df["model"].unique()):
        tests = _unique_tests(df, model, delta, margin)
        fams = {k[0] for k in tests}
        plans = [_plan(h, fams) for h in ("H1", "H2", "H3", "H4", "H5")]
        support = sorted({(k, w) for s, _ in plans for k, w in s if k in tests})
        falsify = sorted({(k, w) for _, f in plans for k, w in f if k in tests})
        rows = []
        for family_name, keys in (("support", support), ("falsify", falsify)):
            a = alpha / max(1, len(keys))
            for key, which in keys:
                fam, arm, ref = key
                ref_delta = delta if ref == "A2_single_ref" else "none"
                p = paired_diffs(df, model, fam, arm, delta, ref, ref_delta)
                d = (p["arm"] - p["ref"]).to_numpy()
                pi_d = float(np.mean(np.abs(d) > 0)) if len(d) else float("nan")
                gain = float(d.mean()) if len(d) else float("nan")
                n = (n_equivalence(pi_d, margin, a, power) if which == "p_equiv"
                     else n_superiority(pi_d, gain, a, power))
                rows.append({"family": fam, "arm": arm, "reference": ref, "test": which,
                             "holm_family": family_name, "alpha_planned": a,
                             "pilot_n": len(d), "pi_d": pi_d, "gain": gain, "n_required": n,
                             "gate_passed": gates.get((model, fam)) is not False})
        equiv = [r["n_required"] for r in rows if r["test"] == "p_equiv" and r["gate_passed"]]
        out["models"][model] = {"per_cell": max(equiv) if equiv else None, "rows": rows}
    cells = [m["per_cell"] for m in out["models"].values() if m["per_cell"] is not None]
    out["per_cell"] = max(cells) if cells else None
    return out
