"""Integrity checks over run records and transcripts (SC-004, FR-011, FR-012)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ceilings.analysis.gains import paired_diffs, runs_frame

MATCH_TOLERANCE = 0.05
MAX_REFUSAL = 0.05


def audit(
    records: list[dict], prereg_hash: str, match_tables: dict[str, dict] | None = None
) -> dict:
    """Return {"passed": bool, "checks": {...}}; every failure is listed with its run or cell."""
    checks: dict[str, list] = {
        "prereg_hash": [],
        "ledger_totals": [],
        "bit_mode": [],
        "delta_matching": [],
        "refusals": [],
        "invalid_runs": [],
    }
    for r in records:
        if r["prereg_hash"] != prereg_hash:
            checks["prereg_hash"].append(r["run_id"])
        led = r["ledger"]
        parts = sum(a["input_tokens"] + a["output_tokens"] for a in led["per_agent"].values())
        parts += led["tool_calls"]["charged_tokens"]
        if max(1, parts) != led["total_tokens"]:
            checks["ledger_totals"].append(r["run_id"])
        if not r.get("valid", True):
            checks["invalid_runs"].append(r["run_id"])
        ref = r.get("transcript_ref")
        if ref and Path(ref).exists():
            t = json.loads(Path(ref).read_text())
            cfg = t["config"]
            if cfg["message_mode"] == "bits":
                pat = re.compile(rf"^[01]{{0,{cfg['message_budget']}}}$")
                for step in t["steps"]:
                    if step["step"] == "message" and not pat.match(step["text"] or ""):
                        checks["bit_mode"].append(r["run_id"])
    df = runs_frame(records)
    if not df.empty:
        main = df[~df["is_sweep"]]
        for (model, family, arm, delta), g in main.groupby(["model", "family", "arm", "delta"]):
            if g["refused"].mean() > MAX_REFUSAL:
                checks["refusals"].append(f"{model}/{family}/{arm}/{delta}")
            table = (match_tables or {}).get(f"{model}/{family}")
            if arm == "baseline" or table is None:
                continue
            # An arm's own target (A2_single_ref is matched to A2's realized cost), else the level's.
            target = table["arms"].get(arm, {}).get(delta, {}).get("target",
                                                                    table["targets"][delta])
            added = paired_diffs(df, model, family, arm, delta)["d_tokens"].mean()
            if abs(added - target) > MATCH_TOLERANCE * target:
                checks["delta_matching"].append(
                    f"{model}/{family}/{arm}/{delta}: added {added:.0f} vs target {target}"
                )
    return {
        "passed": all(not v for v in checks.values()),
        "checks": checks,
        "n_records": len(records),
    }


def refusal_table(records: list[dict]) -> pd.DataFrame:
    df = runs_frame(records)
    return df.groupby(["model", "family", "arm"])["refused"].mean().reset_index()
