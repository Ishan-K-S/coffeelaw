"""The smaller free-tier pilot for one model, end to end, inside its own container.

Every model runs this; the containers coordinate only through files under the run root:

1. calibrate (difficulty search, resumable), then mark the model calibrated;
2. wait until every pilot model is calibrated, then pick the models to continue with: the ones
   with the most families whose search found a passing setting, at most one per developer, ties
   broken by the order of PILOT_MODELS (fastest first); every container computes the same choice;
3. selected models only: prepare (gate + real cost matching at medium) for the passing families,
   the main grid at n=30, `ceilings check`;
4. the first SWEEP_MODELS selected models: the H6 and H7 sweeps at n=20, sized to the model's own
   difficulty, then `ceilings check`;
5. the first selected model: the ablations (one harness option each, baseline plus the family's
   matching arm, n=20), each in its own root, then `ceilings check`.

A crashed step (exit 1) is retried after a pause; every step resumes from its cache or run IDs.
Exit code 2 (a failed gate) and 4 (request cap) are results, not errors.

    python scripts/pilot_small.py --model openai/<name>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PILOT_MODELS = os.environ.get("PILOT_MODELS", ",".join([
    "openai/nvidia/nemotron-3-ultra-550b-a55b",
    "openai/nvidia/nemotron-3.5-super-vl-120b-a12b",
    "openai/stepfun-ai/step-5-preview",
    "openai/z-ai/glm-5.3-flash",
    "openai/nvidia/nemotron-3.5-lightning-30b-a3b",
    "openai/meta/llama-3.2-11b-vision-instruct",
])).split(",")
FAMILIES = ["info", "reason", "comm_parity", "comm_ip"]
MATCHING_ARM = {"info": "E", "reason": "K", "comm_parity": "M", "comm_ip": "M"}
N_SELECT, SWEEP_MODELS = 3, 2
ABLATIONS = {  # name: (--harness options, families it can affect)
    "probe_agent": ("probe=agent", ["info"]),
    "sender_encode": ("sender=encode", ["comm_parity", "comm_ip"]),
    "answer_first": ("schema=answer_first", FAMILIES),
    "shuffle_off": ("shuffle=off", FAMILIES),
    "temp0": ("temp=0", FAMILIES),
}
ROOT = Path(os.environ.get("ROOT", "data/pilot"))
STATE = ROOT / "state"


def log(msg: str) -> None:
    print(f"=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def ceilings(args: list[str], root: Path = ROOT, rpm: float | None = None) -> int:
    env = dict(os.environ)
    if rpm is not None:
        env["CEILINGS_OPENAI_RPM"] = str(rpm)
    for attempt in range(1, 7):
        log(f"attempt {attempt}: ceilings --root {root} {' '.join(args)}")
        code = subprocess.call(["ceilings", "--root", str(root), *args], env=env)
        log(f"exit {code}")
        if code in (0, 2, 3, 4):
            return code
        time.sleep(300)
    return 1


def safe(model: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def chosen(model: str, family: str) -> dict | None:
    path = ROOT / "difficulty" / f"{family}__{safe(model)}.json"
    return json.loads(path.read_text()).get("chosen") if path.exists() else None


def passing(model: str) -> list[str]:
    return [f for f in FAMILIES if chosen(model, f) is not None]


def gate_passed(model: str, family: str) -> bool:
    path = ROOT / "calibration" / f"{family}__{safe(model)}.json"
    return path.exists() and bool(json.loads(path.read_text())["passed"])


def selection() -> list[str]:
    ranked = sorted(PILOT_MODELS, key=lambda m: (-len(passing(m)), PILOT_MODELS.index(m)))
    picked, developers = [], set()
    for m in ranked:
        dev = m.split("/")[-2] if m.count("/") >= 1 else m
        if passing(m) and dev not in developers:
            picked.append(m)
            developers.add(dev)
        if len(picked) == N_SELECT:
            break
    return picked


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=PILOT_MODELS)
    a = p.parse_args()
    m = a.model
    reports = Path("reports/pilot") / safe(m)
    (STATE / "calibrated").mkdir(parents=True, exist_ok=True)

    ceilings(["calibrate", "--model", m, "--n", "30", "--concurrency", "6",
              "--max-requests", "7000"], rpm=6)
    (STATE / "calibrated" / safe(m)).touch()
    while not all((STATE / "calibrated" / safe(x)).exists() for x in PILOT_MODELS):
        time.sleep(120)
    picked = selection()
    (STATE / "selection.json").write_text(json.dumps(
        {"selected": picked, "passing": {x: passing(x) for x in PILOT_MODELS}}, indent=1))
    log(f"selected: {picked}")
    if m not in picked:
        log("not selected; done")
        return 0
    rpm = 36 / len(picked)
    fams = passing(m)
    ceilings(["prepare", "--model", m, "--family", ",".join(fams), "--n-main", "30",
              "--n-pilot", "30", "--n-match", "6", "--delta", "medium", "--concurrency", "6",
              "--max-requests", "12000"], rpm=rpm)
    run_fams = [f for f in fams if gate_passed(m, f)]
    log(f"gate passed after prepare: {run_fams}")
    if run_fams:
        ceilings(["run", "--model", m, "--family", ",".join(run_fams), "--delta", "medium",
                  "--concurrency", "6", "--max-requests", "15000"], rpm=rpm)
    ceilings(["check", "--model", m, "--out", str(reports / "main")])

    if m in picked[:SWEEP_MODELS]:
        info, ip = chosen(m, "info"), chosen(m, "comm_ip") or chosen(m, "comm_parity")
        if info is not None:
            ceilings(["sweep", "--model", m, "--family", "info", "--k", str(info["k"]),
                      "--T", "0,1,2,4", "--flip-rate", "0.0,0.1", "--n", "20",
                      "--concurrency", "6", "--max-requests", "3000"], rpm=rpm)
        if ip is not None:
            n_bits = int(ip["n"])
            Ls = sorted({0, 1, 2, max(1, n_bits // 2), n_bits})
            ceilings(["sweep", "--model", m, "--family", "comm", "--n-bits", str(n_bits),
                      "--L", ",".join(map(str, Ls)), "--n", "20", "--concurrency", "6",
                      "--max-requests", "3000"], rpm=rpm)
        ceilings(["check", "--model", m, "--out", str(reports / "sweeps")])

    if m == picked[0]:
        for name, (opts, fams_affected) in ABLATIONS.items():
            fs = [f for f in fams_affected if f in run_fams]
            if not fs:
                log(f"ablation {name}: no evaluable family; skipped")
                continue
            root = ROOT.parent / f"{ROOT.name}_ablations" / name
            for sub in ("instances", "arms", "calibration", "difficulty"):
                if (ROOT / sub).exists() and not (root / sub).exists():
                    shutil.copytree(ROOT / sub, root / sub)
            for f in fs:
                ceilings(["run", "--model", m, "--family", f, "--delta", "medium",
                          "--arms", f"baseline,{MATCHING_ARM[f]}", "--limit", "20",
                          "--harness", opts, "--concurrency", "6", "--max-requests", "3000"],
                         root=root, rpm=rpm)
            ceilings(["check", "--model", m, "--out", str(reports / f"ablation_{name}")],
                     root=root)
    log("pilot steps for this model done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
