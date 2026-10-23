"""Check one model end to end through the harness before spending a real budget on it.

For each family, one instance at the default difficulty runs under baseline and under the family's
own fix at full strength. Prints, per family: correctness, how each answer was parsed, whether the
served model matched, hidden reasoning, truncation, tokens, requests and wall time. Transcripts go
to `<root>/doctor/<model>/`.

    python scripts/doctor.py --model openai/<name> [--root data] [--families info,reason]

Uses the same environment as `ceilings` (CEILINGS_OPENAI_*), so the provider profile and the rate
limit apply.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ceilings import arms, experiment, tasks
from ceilings.adapters import safe_model_dir
from ceilings.agents import make_policy
from ceilings.orchestrator import run_team


def calls(outcome) -> list[dict]:
    out = []
    for step in outcome.transcript:
        logs = step.get("calls") or ([step["call"]] if step.get("call") else [])
        out += [c for c in logs if c]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--root", default="data")
    p.add_argument("--families", default="info,reason,comm_parity,comm_ip")
    a = p.parse_args()
    policy = make_policy(a.model)
    adapter = getattr(policy, "adapter", None)
    out_dir = Path(a.root) / "doctor" / safe_model_dir(a.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model": a.model, "families": {}}
    for fam in a.families.split(","):
        diff = experiment.DEFAULT_DIFFICULTY[fam]
        inst = tasks.generate_many(fam, 1, "calibration", 9_000_000, **diff)[0]
        arm_id, strength = arms.full_strength(fam, diff)
        for label, cfg in (("baseline", arms.build("baseline", fam)),
                           (f"{arm_id}={strength}", arms.build(arm_id, fam, strength))):
            before = adapter.requests if adapter is not None else 0
            t0 = time.time()
            try:
                out = run_team(inst, cfg, policy, seed=1)
            except Exception as e:  # noqa: BLE001 - report and keep going
                print(f"{fam:12s} {label:14s} ERROR {type(e).__name__}: {str(e)[:160]}",
                      flush=True)
                summary["families"].setdefault(fam, {})[label] = {"error": str(e)[:300]}
                continue
            secs = time.time() - t0
            cs = calls(out)
            row = {
                "correct": out.final_answer == inst.answer,
                "answer": out.final_answer, "truth": inst.answer, "valid": out.valid,
                "parse": [c.get("parse") for c in cs if c.get("kind") == "answer"],
                "answer_not_argmax": sum(bool(c.get("answer_not_argmax")) for c in cs),
                "hidden_reasoning": sum(bool(c.get("hidden_reasoning")) for c in cs),
                "truncated": sum(c.get("stop") == "length" or c.get("reasoning_stop") == "length"
                                 for c in cs),
                "messages": [s["text"] for s in out.transcript if s["step"] == "message"],
                "tokens": out.ledger.total_tokens,
                "requests": (adapter.requests - before) if adapter is not None else None,
                "secs": round(secs, 1),
            }
            summary["families"].setdefault(fam, {})[label] = row
            (out_dir / f"{fam}__{label.replace('=', '-')}.json").write_text(
                json.dumps({"instance": inst.model_dump(), "config": cfg.to_dict(),
                            "steps": out.transcript}, default=str, indent=1))
            print(f"{fam:12s} {label:14s} correct={row['correct']!s:5s} valid={row['valid']!s:5s} "
                  f"parse={row['parse']} not_argmax={row['answer_not_argmax']} "
                  f"hidden={row['hidden_reasoning']} "
                  f"trunc={row['truncated']} msgs={row['messages']} tok={row['tokens']} "
                  f"req={row['requests']} {row['secs']}s", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
