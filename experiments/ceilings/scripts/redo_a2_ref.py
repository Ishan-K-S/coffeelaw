"""Re-match and rerun the A2_single_ref arm after the pooled-budget fix (commit 8f47629).

Backs up the run records and matching cache, drops only the A2_single_ref entries for one
model and family, then the caller reruns `prepare` (everything else cached) and
`run --arms A2_single_ref`.

    python scripts/redo_a2_ref.py --root data/pilot --model openai/<m> --family info
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from ceilings.adapters import safe_model_dir

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/pilot")
p.add_argument("--model", required=True)
p.add_argument("--family", required=True)
a = p.parse_args()
root, tag = Path(a.root), safe_model_dir(a.model)
backup = root / "backup" / time.strftime("%Y%m%dT%H%M%S")
backup.mkdir(parents=True, exist_ok=True)
runs = root / "runs" / tag / a.family / "runs.jsonl"
cache = root / "matching" / f"{a.family}__{tag}.json"
for f in (runs, cache, root / "arms" / a.family / f"{tag}.json"):
    if f.exists():
        shutil.copy2(f, backup / f"{f.parent.name}__{f.name}")
lines = runs.read_text().splitlines()
keep = [ln for ln in lines if json.loads(ln)["arm_id"] != "A2_single_ref"]
runs.write_text("".join(ln + "\n" for ln in keep))
data = json.loads(cache.read_text())
for scope in data.values():
    for k in [k for k in scope if k != "baseline" and json.loads(k)["arm_id"] == "A2_single_ref"]:
        del scope[k]
cache.write_text(json.dumps(data, indent=2, default=str))
print(f"{a.model} {a.family}: dropped {len(lines) - len(keep)} A2_single_ref runs; backup {backup}")
