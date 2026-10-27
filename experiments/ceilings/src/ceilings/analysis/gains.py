"""Gains over baseline with paired cluster-bootstrap intervals, and realized-cost accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd

B_RESAMPLES = 10_000


def runs_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        sweep = r.get("sweep") or {}
        rows.append(
            {
                "run_id": r["run_id"],
                "instance_id": r["instance_id"],
                "family": r["family_id"],
                "arm": r["arm_id"],
                "delta": r.get("delta_level") or "none",
                "model": r["model_id"],
                "replicate": r["replicate"],
                "correct": bool(r["correct"]),
                "valid": r.get("valid", True),
                "refused": r.get("refusal") is not None,
                "tokens": r["ledger"]["total_tokens"],
                "dollars": r["ledger"].get("dollars") or 0.0,
                "p_true": r.get("p_true"),
                "probabilities": r["probabilities"],
                "final": r["final_answer"],
                "T": sweep.get("T"),
                "flip_rate": sweep.get("flip_rate"),
                "L": sweep.get("L_bits"),
                "is_sweep": bool(sweep),
            }
        )
    return pd.DataFrame(rows)


def paired_diffs(
    df: pd.DataFrame,
    model: str,
    family: str,
    arm: str,
    delta: str,
    reference: str = "baseline",
    ref_delta: str = "none",
) -> pd.DataFrame:
    """Per-instance arm and reference correctness (replicates averaged) and their differences."""
    main = df[~df["is_sweep"] & df["valid"] & (df["model"] == model) & (df["family"] == family)]
    a = main[(main["arm"] == arm) & (main["delta"] == delta)]
    b = main[(main["arm"] == reference) & (main["delta"] == ref_delta)]
    a = a.groupby("instance_id")[["correct", "tokens"]].mean()
    b = b.groupby("instance_id")[["correct", "tokens"]].mean()
    joined = a.join(b, lsuffix="_arm", rsuffix="_ref", how="inner")
    return pd.DataFrame(
        {
            "arm": joined["correct_arm"],
            "ref": joined["correct_ref"],
            "d_correct": joined["correct_arm"] - joined["correct_ref"],
            "d_tokens": joined["tokens_arm"] - joined["tokens_ref"],
        }
    )


def bootstrap_ci(
    values: np.ndarray, seed: int = 0, n_boot: int = B_RESAMPLES, level: float = 0.95
) -> tuple[float, float]:
    """Percentile CI for the mean, resampling instances (each instance is one cluster)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def gain_table(df: pd.DataFrame, n_boot: int = B_RESAMPLES) -> pd.DataFrame:
    """One row per model x family x arm x delta: accuracy, gain vs baseline with CI, added cost."""
    rows = []
    main = df[~df["is_sweep"] & df["valid"]]
    for (model, family), g in main.groupby(["model", "family"]):
        base = g[g["arm"] == "baseline"]
        base_acc = base["correct"].mean()
        for (arm, delta), _ in g[g["arm"] != "baseline"].groupby(["arm", "delta"]):
            pdiff = paired_diffs(df, model, family, arm, delta)
            if len(pdiff) < 2:
                continue
            lo, hi = bootstrap_ci(pdiff["d_correct"].to_numpy(), n_boot=n_boot)
            rows.append(
                {
                    "model": model,
                    "family": family,
                    "arm": arm,
                    "delta": delta,
                    "n": len(pdiff),
                    "baseline_acc": base_acc,
                    "arm_acc": base_acc + pdiff["d_correct"].mean(),
                    "gain": pdiff["d_correct"].mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "added_tokens": pdiff["d_tokens"].mean(),
                    # secondary outcome: mean probability on the true answer (not affected by
                    # how an uninformed team breaks ties)
                    "baseline_p_true": base["p_true"].astype(float).mean(),
                    "arm_p_true": g[(g["arm"] == arm) & (g["delta"] == delta)]["p_true"]
                    .astype(float).mean(),
                }
            )
    return pd.DataFrame(rows)
