"""Pilot variant: the communication families with a reasoning floor on every call.

In the smaller pilot every model failed both comm gates at n=2 because agent A, answering with
no reasoning, could not combine B's (correct) message with its own list. This variant gives every
call in the comm families at least `COMM_DEPTH` reasoning (`--harness comm_depth=...`), so the
gate tests communication, then for each family that passes: prepare (gate + cost matching at
medium), the main grid at n=30, the H7 bit sweep at n=20, and `ceilings check`.

Run with ROOT=data/pilot_comm_reason (its own root: the option changes every run ID).

    ROOT=data/pilot_comm_reason python scripts/pilot_comm_reason.py --model openai/<name>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pilot_small as p  # noqa: E402  (ROOT comes from the environment)

COMM = ["comm_parity", "comm_ip"]
COMM_DEPTH = os.environ.get("COMM_DEPTH", "medium")
RPM = float(os.environ.get("CEILINGS_OPENAI_RPM", "6"))


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--model", required=True)
    m = a.parse_args().model
    harness = f"comm_depth={COMM_DEPTH}"
    done = p.STATE / "done" / p.safe(m)
    done.parent.mkdir(parents=True, exist_ok=True)
    if done.exists():
        p.log("already done; idle")
        while True:
            time.sleep(3600)
    reports = Path("reports/pilot_comm_reason") / p.safe(m)
    p.ceilings(["calibrate", "--model", m, "--family", ",".join(COMM), "--n", "30",
                "--concurrency", "6", "--max-requests", "8000", "--harness", harness], rpm=RPM)
    fams = [f for f in COMM if p.chosen(m, f) is not None]
    p.log(f"comm families passing with {harness}: {fams}")
    if fams:
        p.ceilings(["prepare", "--model", m, "--family", ",".join(fams), "--n-main", "30",
                    "--n-pilot", "30", "--n-match", "6", "--delta", "medium",
                    "--concurrency", "6", "--max-requests", "12000", "--harness", harness],
                   rpm=RPM)
        run_fams = [f for f in fams if p.gate_passed(m, f)]
        if run_fams:
            p.ceilings(["run", "--model", m, "--family", ",".join(run_fams), "--delta",
                        "medium", "--concurrency", "6", "--max-requests", "12000",
                        "--harness", harness], rpm=RPM)
        ip = p.chosen(m, "comm_ip") or p.chosen(m, "comm_parity")
        n_bits = int(ip["n"])
        Ls = sorted({0, 1, 2, max(1, n_bits // 2), n_bits})
        p.ceilings(["sweep", "--model", m, "--family", "comm", "--n-bits", str(n_bits),
                    "--L", ",".join(map(str, Ls)), "--n", "20", "--concurrency", "6",
                    "--max-requests", "4000", "--harness", harness], rpm=RPM)
        p.ceilings(["check", "--model", m, "--out", str(reports)])
    p.log("comm-with-reasoning variant done")
    done.touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
