"""`ceilings check`: did the runs go as designed, and did they give the results we needed?

Three tiers:

- integrity (FAIL on any violation; the run is not valid evidence): no answer leakage into
  prompts, private evidence isolated between agents, message budgets respected, each arm changes
  exactly one resource, step counts match the arm configuration, token ledgers add up;
- quality (WARN above a threshold): unparseable answers, invalid probes, invalid bit messages,
  reasoning truncated by the token cap, refusals, model substitution;
- expectations (reported): baseline at chance, and the matching intervention of each family
  helps (the diagonal). With small models or small n these may be "not met" or "inconclusive";
  that is a result to report, not a bug.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ceilings import arms
from ceilings.analysis.gains import bootstrap_ci, paired_diffs, runs_frame
from ceilings.records import TaskInstance

QUALITY_THRESHOLDS = {
    "answer_parse_failed": 0.05,
    "answer_not_argmax": 0.10,
    "probabilities_not_summing_to_1": 0.20,
    "answer_parse_text_fallback": 0.20,
    "probe_invalid": 0.10,
    "message_rejected": 0.20,
    "reasoning_truncated": 0.25,
    "refused": 0.05,
    "invalid_model": 0.0,
}
MATCHING_ARM = {"info": "E", "reason": "K", "comm_parity": "M", "comm_ip": "M"}
ALLOWED_CHANGES = {
    "E": {"tool_rounds", "extra_kind", "extra_amount"},
    "K": {"depth", "reason_samples"},
    "M": {"message_budget", "relay"},
    "A1": {"n_agents", "samples", "answerers"},
    "A2": {"n_agents", "samples", "depth", "relay", "message_mode", "message_budget"},
    "A2_single_ref": {"n_agents", "samples", "depth"},
    "P1": {"extra_kind"},
    "P2": {"extra_kind"},
    "baseline": set(),
}
# Harness-wide settings (ceilings.options), not resources an arm may change.
IGNORED_FIELDS = {"arm_id", "strength", "filler_tokens", "probe_policy", "comm_sender"}


def _baseline_for(t: dict, fam: str) -> dict:
    """The baseline config this run is compared with: the default baseline after the run's own
    harness options (e.g. a comm_depth reasoning floor applies to the baseline too)."""
    from types import SimpleNamespace

    from ceilings.options import HarnessOptions
    from ceilings.orchestrator import effective_config

    given = {k: v for k, v in (t.get("harness") or {}).items() if k != "resolved"}
    if "reasoning_tokens" in given and given["reasoning_tokens"] is not None:
        given["reasoning_tokens"] = tuple(given["reasoning_tokens"])
    policy = SimpleNamespace(options=HarnessOptions(**given))
    base = asdict(effective_config(arms.baseline(fam), policy, fam))
    base["answerers"] = list(base["answerers"])
    return base


def load_instances_index(root: Path | str) -> dict[str, TaskInstance]:
    index = {}
    for path in Path(root).glob("instances/**/*.jsonl"):
        for line in path.read_text().splitlines():
            inst = TaskInstance.model_validate(json.loads(line))
            index[inst.instance_id] = inst
    return index


def _prompt_text(call: dict | None) -> str:
    if not call or not call.get("prompt"):
        return ""
    return "\n".join(m["content"] for m in call["prompt"] if m["role"] != "system")


def _calls(step: dict) -> list[dict]:
    if step["step"] == "message":
        return step.get("calls", [])
    return [step["call"]] if step.get("call") else []


def _check_transcript(t: dict, inst: TaskInstance | None, problems: dict, rates: dict) -> None:
    rid, fam, cfg = t["run_id"], t["family"], t["config"]
    steps = t["steps"]
    key = (t["model_id"], fam, cfg["arm_id"])

    # Each arm changes exactly one resource relative to baseline.
    base = _baseline_for(t, fam)
    changed = {k for k in base if base[k] != cfg.get(k)} - IGNORED_FIELDS
    if t.get("sweep") is None and not changed <= ALLOWED_CHANGES.get(cfg["arm_id"], set()):
        problems["arm_changes_one_resource"].append(f"{rid[:12]} {cfg['arm_id']}: {sorted(changed)}")

    # Step counts match the configuration.
    n_probe = sum(s["step"] == "probe" for s in steps)
    n_answer = sum(s["step"] == "answer" for s in steps)
    if fam == "info" and n_probe != cfg["tool_rounds"]:
        problems["step_counts"].append(f"{rid[:12]}: {n_probe} probes, config {cfg['tool_rounds']}")
    if n_answer != cfg["samples"] * len(cfg["answerers"]):
        problems["step_counts"].append(f"{rid[:12]}: {n_answer} answers")

    for s in steps:
        if s["step"] == "probe":
            rates["probe_invalid"][key].append(s["result"] == "ERR")
        if s["step"] == "message":
            if s["mode"] == "bits":
                ok = re.fullmatch(rf"[01]{{0,{s['budget']}}}", s["text"] or "") is not None
                if not ok:
                    problems["message_budget"].append(f"{rid[:12]}: {s['text']!r}")
                rates["message_rejected"][key].append(bool(s["retried"]))
            elif s["tokens"] > s["budget"]:
                problems["message_budget"].append(f"{rid[:12]}: {s['tokens']} > {s['budget']}")
        if s["step"] == "answer":
            call = s.get("call") or {}
            rates["answer_parse_failed"][key].append(s["answer"] is None and not s["refusal"])
            rates["answer_parse_text_fallback"][key].append(call.get("parse") == "text")
            rates["answer_not_argmax"][key].append(bool(call.get("answer_not_argmax")))
            raw_sum = call.get("raw_prob_sum")
            rates["probabilities_not_summing_to_1"][key].append(
                raw_sum is not None and abs(raw_sum - 1) > 0.05)
            rates["refused"][key].append(bool(s["refusal"]))
            rates["invalid_model"][key].append(not s.get("valid", True))
        for call in _calls(s):
            if call.get("reasoning"):
                rates["reasoning_truncated"][key].append(call.get("reasoning_stop") == "length")

    if inst is None:
        problems["instance_available"].append(f"{rid[:12]}: instance {t['instance_id'][:12]}")
        return
    # No leakage: before any probe result or message arrives, the true answer must not be singled
    # out. Every option should appear in the prompt as often as the most-mentioned other option
    # (the host table and the "host-00 to host-07" range; the even/odd question); a prompt that
    # names the answer more often than every other option leaks it.
    if fam in ("info", "comm_parity", "comm_ip"):
        for s in steps:
            for call in _calls(s):
                text = _prompt_text(call)
                if not text or "Probe result:" in text or "Messages you received" in text:
                    continue
                counts = {a: len(re.findall(rf"(?<![\w-]){re.escape(a)}(?![\w-])", text))
                          for a in inst.answer_space}
                others = max(c for a, c in counts.items() if a != inst.answer)
                if counts[inst.answer] > others:
                    problems["no_answer_leak"].append(f"{rid[:12]}: {counts}")
    # Private evidence stays private: an agent's prompt never contains another agent's
    # private lines (messages are summarized by the sender, never copied by the harness).
    for s in steps:
        agent = s.get("agent") or s.get("from")
        for call in _calls(s):
            text = _prompt_text(call)
            if not text:
                continue
            for other, priv in inst.private_evidence.items():
                if other == agent or not priv:
                    continue
                leaked = [ln for ln in priv.splitlines() if ln and ln in text]
                if leaked:
                    problems["private_isolation"].append(f"{rid[:12]} agent {agent} saw {other}")


def _rate_table(rates: dict) -> dict:
    out = {}
    for name, by_key in rates.items():
        rows = []
        for (model, fam, arm), vals in sorted(by_key.items()):
            if vals:
                rows.append({"model": model, "family": fam, "arm": arm, "n": len(vals),
                             "rate": float(np.mean(vals))})
        worst = max((r["rate"] for r in rows), default=0.0)
        status = "PASS" if worst <= QUALITY_THRESHOLDS[name] else "WARN"
        out[name] = {"status": status, "threshold": QUALITY_THRESHOLDS[name],
                     "worst": worst, "rows": [r for r in rows if r["rate"] > 0]}
    return out


def _expectations(records: list[dict], delta: str, index: dict | None = None) -> dict:
    df = runs_frame(records)
    out = {"baseline_at_chance": [], "matching_arm_helps": []}
    if df.empty:
        return out
    main = df[~df["is_sweep"]]
    for (model, fam), g in main.groupby(["model", "family"]):
        base = g[g["arm"] == "baseline"]
        if base.empty:
            continue
        from ceilings.tasks import chance as instance_chance

        known = [instance_chance(index[i]) for i in base["instance_id"] if index and i in index]
        chance = float(np.mean(known)) if known else 1 / len(base["probabilities"].iloc[0])
        acc = float(base["correct"].mean())
        se = float(np.sqrt(chance * (1 - chance) / len(base)))
        if 1.96 * se > 0.10:
            status = "too few instances"
        else:
            status = "met" if abs(acc - chance) <= 0.05 + 1.96 * se else "not met"
        out["baseline_at_chance"].append({
            "model": model, "family": fam, "n": len(base), "baseline_acc": acc, "chance": chance,
            "status": status})
        arm = MATCHING_ARM.get(fam)
        d = paired_diffs(df, model, fam, arm, delta)["d_correct"].to_numpy() if arm else []
        if len(d) >= 2:
            lo, hi = bootstrap_ci(d, n_boot=2000)
            status = "met" if lo > 0 else ("not met" if hi <= 0 else "inconclusive")
            out["matching_arm_helps"].append({"model": model, "family": fam, "arm": arm,
                                              "n": len(d), "gain": float(d.mean()),
                                              "ci": [lo, hi], "status": status})
    return out


def run_checks(root: Path | str, records: list[dict], delta: str = "medium") -> dict:
    root = Path(root)
    index = load_instances_index(root)
    problems: dict[str, list] = {k: [] for k in ("no_answer_leak", "private_isolation",
                                                 "message_budget", "arm_changes_one_resource",
                                                 "step_counts", "ledger_totals",
                                                 "instance_available")}
    rates: dict = {k: defaultdict(list) for k in QUALITY_THRESHOLDS}
    n_transcripts = 0
    for r in records:
        led = r["ledger"]
        parts = sum(a["input_tokens"] + a["output_tokens"] for a in led["per_agent"].values())
        if max(1, parts + led["tool_calls"]["charged_tokens"]) != led["total_tokens"]:
            problems["ledger_totals"].append(r["run_id"][:12])
        ref = r.get("transcript_ref")
        if ref and Path(ref).exists():
            t = json.loads(Path(ref).read_text())
            _check_transcript(t, index.get(r["instance_id"]), problems, rates)
            n_transcripts += 1
    integrity = {k: {"status": "PASS" if not v else "FAIL", "n_violations": len(v),
                     "examples": v[:5]} for k, v in problems.items()}
    quality = _rate_table(rates)
    expectations = _expectations(records, delta, index)
    passed = all(v["status"] == "PASS" for v in integrity.values())
    return {"passed": passed, "n_records": len(records), "n_transcripts": n_transcripts,
            "integrity": integrity, "quality": quality, "expectations": expectations}


def summary_lines(res: dict) -> list[str]:
    lines = [f"records {res['n_records']}, transcripts checked {res['n_transcripts']}"]
    for name, v in res["integrity"].items():
        lines.append(f"integrity  {v['status']:4s} {name} ({v['n_violations']} violations)")
    for name, v in res["quality"].items():
        lines.append(f"quality    {v['status']:4s} {name} (worst {v['worst']:.0%}, "
                     f"threshold {v['threshold']:.0%})")
    for row in res["expectations"]["baseline_at_chance"]:
        lines.append(f"expect     baseline at chance {row['model']}/{row['family']}: "
                     f"{row['baseline_acc']:.2f} vs {row['chance']:.2f} -> {row['status']}")
    for row in res["expectations"]["matching_arm_helps"]:
        lines.append(f"expect     {row['arm']} helps {row['model']}/{row['family']}: gain "
                     f"{row['gain']:+.2f} CI [{row['ci'][0]:+.2f}, {row['ci'][1]:+.2f}] -> "
                     f"{row['status']}")
    return lines
