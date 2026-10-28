"""Calibration gate (FR-007), per-model difficulty search, and cost matching (FR-009, SC-004)."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

from ceilings import arms, tasks
from ceilings.agents import Policy
from ceilings.orchestrator import effective_config, run_team
from ceilings.records import CalibrationResult, TaskInstance, content_hash
from ceilings.runner import pmap

LEAK_CONFIDENCE = 0.9
MATCH_TOLERANCE = 0.05
GATE_FULL = 0.90

# Difficulty grids for the search, ordered from easiest to hardest. "quick" is a short grid for
# throttled local test runs.
DIFFICULTY_GRID = {
    "info": [{"k": 2, "p": 0.0}, {"k": 3, "p": 0.0}, {"k": 3, "p": 0.1}, {"k": 4, "p": 0.1}],
    "reason": [{"d": d} for d in (2, 3, 4, 6, 8, 12, 16)],
    "comm_parity": [{"n": n} for n in (2, 3, 4, 6, 8)],
    "comm_ip": [{"n": n} for n in (2, 3, 4, 6, 8)],
}
QUICK_GRID = {
    "info": [{"k": 2, "p": 0.0}, {"k": 3, "p": 0.0}],
    "reason": [{"d": d} for d in (3, 6, 10)],
    "comm_parity": [{"n": n} for n in (2, 4)],
    "comm_ip": [{"n": n} for n in (2, 4)],
}


def _seed(*parts) -> int:
    return int(content_hash(list(parts))[:8], 16)


def _run(inst: TaskInstance, cfg: arms.ArmConfig, policy: Policy):
    return run_team(inst, cfg, policy, _seed(inst.instance_id, cfg.to_dict(), "pilot"))


def _runs(instances, cfg, policy, concurrency):
    return pmap(lambda i: _run(i, cfg, policy), instances, concurrency)


def _log_row(inst, cfg, o, policy) -> dict:
    return {
        "instance_id": inst.instance_id, "family": inst.family_id,
        "difficulty": dict(inst.difficulty), "answer": inst.answer,
        "model_id": policy.model_id, "harness": policy.harness_record(),
        "config": effective_config(cfg, policy, inst.family_id).to_dict(),
        "final_answer": o.final_answer,
        "correct": o.final_answer == inst.answer, "valid": o.valid,
        "probabilities": o.probabilities, "refusal": o.refusal,
        "tokens": o.ledger.total_tokens, "steps": o.transcript}


class _Logged:
    """A gate run read back from the log: the fields the gate uses."""

    def __init__(self, row: dict):
        self.final_answer = row["final_answer"]
        self.probabilities = row.get("probabilities") or {}
        self.refusal = row.get("refusal")
        self.tokens = row["tokens"]


def _logged_runs(path: Path, instances, cfg, policy, concurrency) -> list:
    """Every gate run and its transcript, appended to `path` as each one finishes, so a gate
    result can be audited and an interrupted gate resumes with only the missing runs."""
    def key(inst):
        conf = effective_config(cfg, policy, inst.family_id).to_dict()
        return inst.instance_id, json.dumps(json.loads(json.dumps(conf, default=str)),
                                            sort_keys=True)

    done: dict = {}
    if path.exists():
        text = path.read_text()
        if text and not text.endswith("\n"):
            with open(path, "a") as f:
                f.write("\n")  # end the line cut off by the interruption before appending
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by the interruption
            done[(row["instance_id"], json.dumps(row["config"], sort_keys=True))] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def run_and_log(inst):
        o = _run(inst, cfg, policy)
        row = json.loads(json.dumps(_log_row(inst, cfg, o, policy), default=str))
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            done[key(inst)] = row

    pmap(run_and_log, [i for i in instances if key(i) not in done], concurrency)
    return [_Logged(done[key(i)]) for i in instances]


def _tokens(o) -> int:
    return o.tokens if isinstance(o, _Logged) else o.ledger.total_tokens


GATE_RULES = ("screen", "confirm", "two_stage")


def baseline_equivalent(base_correct: int, n: int, chance: float) -> bool:
    """Baseline equivalent to chance within +-0.05: two one-sided exact binomial tests at 0.05."""
    from scipy import stats

    lo_ok = stats.binomtest(base_correct, n, max(0.0, chance - 0.05), "greater").pvalue < 0.05
    hi_ok = stats.binomtest(base_correct, n, min(1.0, chance + 0.05), "less").pvalue < 0.05
    return bool(lo_ok and hi_ok)


def gate_passes(rule: str, base_correct: int, full_correct: int, n: int, chance: float) -> bool:
    """The calibration gate (FR-007).

    - "screen" (difficulty search): baseline within 5 points of chance, the tolerance applied to
      the estimate's 95% interval; full fix >= 90%. Loose at small n: at n = 30 any baseline from
      about 0.05 to 0.45 passes at chance 0.25, which is why it only screens.
    - "confirm": baseline equivalent to chance within +-0.05 (`baseline_equivalent`), and the
      full fix's one-sided 95% exact lower bound >= 0.90. Underpowered at gate sizes: with the
      baseline exactly at chance and n = 300 it passes 5% of the time at chance 0.5, 21% at 0.25
      and 65% at 0.125.
    - "two_stage" (prepare, on fresh instances): stage 1 here, the full fix's lower bound >= 0.90
      and the baseline's estimate within 0.05 of chance; stage 2, `baseline_equivalent` on the
      gate's baseline runs pooled with the main grid's, is applied in the analysis
      (`analysis.report.load_gates`).
    """
    from scipy import stats

    if rule == "screen":
        se = float(np.sqrt(chance * (1 - chance) / n))
        return abs(base_correct / n - chance) <= 0.05 + 1.96 * se and full_correct / n >= GATE_FULL
    if rule not in GATE_RULES:
        raise ValueError(f"gate rule must be one of {GATE_RULES}")
    full_lower = (stats.beta.ppf(0.05, full_correct, n - full_correct + 1)
                  if full_correct > 0 else 0.0)
    if rule == "two_stage":
        return bool(abs(base_correct / n - chance) <= 0.05 + 1e-9 and full_lower >= GATE_FULL)
    return bool(baseline_equivalent(base_correct, n, chance) and full_lower >= GATE_FULL)


def calibrate(family: str, instances: list[TaskInstance], policy: Policy,
              concurrency: int = 1, log_path: Path | None = None,
              rule: str = "screen") -> CalibrationResult:
    """Run baseline and the family's own fix at full strength and apply the gate (`rule`, see
    `gate_passes`). With `log_path`, every run and its transcript is appended there."""
    chance = float(np.mean([tasks.chance(i) for i in instances]))
    base_cfg = arms.build("baseline", family)
    arm_id, strength = arms.full_strength(family, dict(instances[0].difficulty))
    full_cfg = arms.build(arm_id, family, strength)
    if log_path is not None:
        base = _logged_runs(log_path, instances, base_cfg, policy, concurrency)
        full = _logged_runs(log_path, instances, full_cfg, policy, concurrency)
    else:
        base = _runs(instances, base_cfg, policy, concurrency)
        full = _runs(instances, full_cfg, policy, concurrency)
    base_acc = float(np.mean([o.final_answer == i.answer for o, i in zip(base, instances,
                                                                       strict=True)]))
    full_acc = float(np.mean([o.final_answer == i.answer for o, i in zip(full, instances,
                                                                       strict=True)]))
    leaks = (
        [o.final_answer == i.answer and o.probabilities.get(i.answer, 0) >= LEAK_CONFIDENCE
         for o, i in zip(base, instances, strict=True)]
        if family == "info" else []
    )
    refusals = [o.refusal is not None for o in base + full]
    return CalibrationResult(
        family_id=family,
        model_id=policy.model_id,
        n=len(instances),
        baseline_acc=base_acc,
        chance_acc=chance,
        matched_full_acc=full_acc,
        leak_rate=float(np.mean(leaks)) if leaks else 0.0,
        refusal_rate=float(np.mean(refusals)),
        passed=gate_passes(rule, round(base_acc * len(instances)),
                           round(full_acc * len(instances)), len(instances), chance),
        baseline_mean_tokens=float(np.mean([_tokens(o) for o in base])),
    )


def search_difficulty(family: str, policy: Policy, n: int = 30, concurrency: int = 1,
                      grid: list[dict] | str | None = None, seed0: int = 3_000_000,
                      stop_early: bool = True, cache_path: Path | None = None) -> dict:
    """Try each difficulty on fresh calibration instances and choose one per family rule.

    - info and comm: the hardest setting whose gate passes (more room for the sweeps);
    - reason: the easiest setting whose gate passes (baseline at chance, and the full fix still
      solves it).
    Returns {"chosen": difficulty or None, "results": [...]}; `chosen` is None when no setting
    passes, which is itself a finding: the model cannot exhibit that ceiling cleanly. With
    `stop_early`, settings that cannot change the choice are not run (`results` lists only the
    settings tried). With `cache_path`, each finished setting is saved as it completes and reused
    on a rerun with the same n (instances and seeds are deterministic), so a long search that is
    interrupted resumes where it stopped.
    """
    if grid == "quick":
        grid = QUICK_GRID[family]
    grid = grid or DIFFICULTY_GRID[family]
    results = []
    cache = {}
    if cache_path is not None and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text())
    for diff in grid:
        key = json.dumps({"difficulty": diff, "n": n, "seed0": seed0,
                          "harness": policy.options.non_default()}, sort_keys=True, default=list)
        if key in cache:
            cal = CalibrationResult(**cache[key])
        else:
            insts = tasks.generate_many(family, n, "calibration", seed0, **diff)
            log = (None if cache_path is None else Path(cache_path).parent / "runs"
                   / Path(cache_path).name.replace(".settings.json", "")
                   / f"{tasks.difficulty_tag(diff)}_n{n}.jsonl")
            if log is not None and log.exists():
                log.unlink()  # a partial log from an interrupted setting
            cal = calibrate(family, insts, policy, concurrency, log_path=log)
            if cache_path is not None:
                cache[key] = cal.model_dump()
                save_json(cache, Path(cache_path))
        results.append({"difficulty": diff, **cal.model_dump()})
        # The grid runs from easy to hard, so once the full fix misses 90% no harder setting
        # can pass; reason takes the easiest passing setting, so its first pass ends the search.
        if stop_early and (cal.matched_full_acc < GATE_FULL
                           or (family == "reason" and cal.passed)):
            break
    passing = [r for r in results if r["passed"]]
    if not passing:
        chosen = None
    elif family == "reason":
        chosen = passing[0]["difficulty"]
    else:
        chosen = passing[-1]["difficulty"]
    return {"family": family, "model_id": policy.model_id, "n": n, "chosen": chosen,
            "results": results}


def _mean_added(instances, cfg, policy, base_cost, concurrency) -> float:
    outs = _runs(instances, cfg, policy, concurrency)
    return float(np.mean([o.ledger.total_tokens - base_cost[i.instance_id]
                          for o, i in zip(outs, instances, strict=True)]))


def largest_fitting(strengths: list, measure, limit: float) -> tuple[object, float]:
    """The largest strength whose measured cost fits under `limit` (the first one if none fits),
    and its cost. Cost rises with strength, so a binary search needs about log2(len) runs of the
    matching instances instead of one per strength."""
    cache: dict[int, float] = {}

    def cost(i: int) -> float:
        if i not in cache:
            cache[i] = measure(strengths[i])
        return cache[i]

    lo, hi = 0, len(strengths) - 1
    if cost(lo) > limit:
        return strengths[lo], cost(lo)
    if cost(hi) <= limit:
        return strengths[hi], cost(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if cost(mid) <= limit else (lo, mid)
    return strengths[lo], cost(lo)


class _Measurements:
    """Disk cache of matching measurements (mean added cost per arm config on these instances), so
    an interrupted `prepare` resumes instead of re-running every measurement."""

    def __init__(self, path: Path | None, instances, policy) -> None:
        self.path = Path(path) if path is not None else None
        self.scope = content_hash([[i.instance_id for i in instances], policy.model_id,
                                   policy.options.non_default()])
        self.data = {}
        if self.path is not None and self.path.exists():
            self.data = json.loads(self.path.read_text()).get(self.scope, {})

    def get(self, key: str, compute):
        if key not in self.data:
            self.data[key] = compute()
            if self.path is not None:
                save_json({self.scope: self.data}, self.path)
        return self.data[key]


def refine_filler(filler: int, added: float, base_natural: float, target: float,
                  samples: int) -> int:
    """Next filler size by proportional correction. Filler is read once per answer call, so a
    token of filler costs about `samples` tokens; when the measured slope is not plausibly
    positive (noisy arms such as A2 can cost less with more filler), that nominal slope is used,
    and the result is capped at 4x the target. An unguarded slope near zero once asked for a
    ~250-million-token filler and exhausted memory."""
    slope = (added - base_natural) / max(1, filler)
    if slope < 0.25 * max(1, samples):
        slope = float(max(1, samples))
    return int(min(4 * target, max(0, filler + (target - added) / slope)))


def match_arms(
    family: str,
    instances: list[TaskInstance],
    policy: Policy,
    levels: dict[str, float] | None = None,
    concurrency: int = 1,
    refinements: int = 3,
    cache_path: Path | None = None,
) -> dict:
    """Choose each arm's strength and filler so its mean added cost is within 5% of the target.
    With `cache_path`, every measurement is saved as it completes and reused on a rerun."""
    levels = levels or arms.DELTA_MULTIPLIERS
    memo = _Measurements(cache_path, instances, policy)

    def baseline_costs():
        outs = _runs(instances, arms.build("baseline", family), policy, concurrency)
        return {i.instance_id: o.ledger.total_tokens for i, o in zip(instances, outs,
                                                                     strict=True)}

    base_cost = memo.get("baseline", baseline_costs)

    def cost_of(cfg) -> float:
        key = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
        return memo.get(key, lambda: _mean_added(instances, cfg, policy, base_cost, concurrency))

    base_mean = float(np.mean(list(base_cost.values())))
    n_bits = instances[0].payload.get("public", {}).get("n")
    table: dict = {
        "family": family,
        "model_id": policy.model_id,
        "difficulty": dict(instances[0].difficulty),
        "n_instances": len(instances),
        "baseline_mean_tokens": base_mean,
        "targets": {lvl: round(m * base_mean) for lvl, m in levels.items()},
        "arms": {},
    }
    for arm_id in [a for a in arms.ARMS if a != "baseline"]:
        table["arms"][arm_id] = {}
        for lvl, target in table["targets"].items():
            strengths = arms.candidates(arm_id, family, n_bits)
            chosen, natural = largest_fitting(
                strengths, lambda s, arm_id=arm_id: cost_of(arms.build(arm_id, family, s)),
                target * (1 + MATCH_TOLERANCE))
            cfg = arms.build(arm_id, family, chosen)
            if arm_id in ("P1", "P2"):
                filler = max(0, int(target / max(1, cfg.samples)))
                base_natural = 0.0
            else:
                filler = max(0, int((target - natural) / max(1, cfg.samples)))
                base_natural = natural
            added = natural
            for _ in range(refinements):  # proportional correction of the filler size
                cfg = replace(arms.build(arm_id, family, chosen), filler_tokens=filler)
                added = cost_of(cfg)
                if abs(added - target) <= MATCH_TOLERANCE * target or filler == 0:
                    break
                filler = refine_filler(filler, added, base_natural, target, cfg.samples)
            table["arms"][arm_id][lvl] = {
                "strength": chosen,
                "filler_tokens": filler,
                "natural_added": natural,
                "added": added,
                "target": target,
                "matched": abs(added - target) <= MATCH_TOLERANCE * target,
            }
    return table


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def default_strength(arm_id: str, family: str, difficulty: dict) -> object:
    """Unmatched default strength per arm (used by `prepare --no-match` for quick test runs)."""
    if arm_id == "E":
        if family == "info":
            return arms.full_strength(family, difficulty)[1]
        return 8
    if arm_id == "K":
        return "high"
    if arm_id == "A2_single_ref":
        return arms.pooled_depth("medium", 4)
    if arm_id == "M":
        if family == "comm_parity":
            return 1
        return int(difficulty["n"]) if family.startswith("comm_") else 100
    if arm_id == "A2":
        return "medium"
    return None


def unmatched_table(family: str, policy: Policy, difficulty: dict, baseline_mean: float,
                    levels: dict[str, float] | None = None) -> dict:
    """A table with default strengths and no filler. Every cell is marked unmatched, so the audit
    reports the unequal costs instead of hiding them; use it only for smoke tests."""
    levels = levels or arms.DELTA_MULTIPLIERS
    targets = {lvl: round(m * baseline_mean) for lvl, m in levels.items()}
    return {
        "family": family, "model_id": policy.model_id, "difficulty": dict(difficulty),
        "n_instances": 0, "baseline_mean_tokens": baseline_mean, "targets": targets,
        "unmatched_defaults": True,
        # P1/P2 are sized to the target (filler read on every answer call) so they remain real
        # placebos; every other arm gets no filler.
        "arms": {a: {lvl: {"strength": default_strength(a, family, difficulty),
                           "filler_tokens": (int(t / arms.baseline(family).samples)
                                             if a in ("P1", "P2") else 0),
                           "natural_added": None, "added": None,
                           "target": t, "matched": False}
                     for lvl, t in targets.items()}
                 for a in arms.ARMS if a != "baseline"},
    }


def config_from_table(table: dict, arm_id: str, level: str | None) -> arms.ArmConfig:
    family = table["family"]
    if arm_id == "baseline":
        return arms.build("baseline", family)
    entry = table["arms"][arm_id][level]
    return arms.build(arm_id, family, entry["strength"], entry["filler_tokens"])
