"""Calibration gate (FR-007), per-model difficulty search, and cost matching (FR-009, SC-004)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ceilings import arms, tasks
from ceilings.agents import Policy
from ceilings.orchestrator import run_team
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


def _log_runs(path: Path, instances, cfg, outcomes, policy) -> None:
    """Append each gate run's answer and full transcript, so a gate result can be audited."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for inst, o in zip(instances, outcomes, strict=True):
            f.write(json.dumps({
                "instance_id": inst.instance_id, "family": inst.family_id,
                "difficulty": dict(inst.difficulty), "answer": inst.answer,
                "model_id": policy.model_id, "harness": policy.harness_record(),
                "config": cfg.to_dict(), "final_answer": o.final_answer,
                "correct": o.final_answer == inst.answer, "valid": o.valid,
                "probabilities": o.probabilities, "tokens": o.ledger.total_tokens,
                "steps": o.transcript}, default=str) + "\n")


def calibrate(family: str, instances: list[TaskInstance], policy: Policy,
              concurrency: int = 1, log_path: Path | None = None) -> CalibrationResult:
    """Baseline must sit within 5 points of chance (allowing for sampling error); the family's own
    full fix must reach 90%. With `log_path`, every run and its transcript is appended there."""
    chance = 1 / len(instances[0].answer_space)
    base_cfg = arms.build("baseline", family)
    base = _runs(instances, base_cfg, policy, concurrency)
    arm_id, strength = arms.full_strength(family, dict(instances[0].difficulty))
    full_cfg = arms.build(arm_id, family, strength)
    full = _runs(instances, full_cfg, policy, concurrency)
    if log_path is not None:
        _log_runs(log_path, instances, base_cfg, base, policy)
        _log_runs(log_path, instances, full_cfg, full, policy)
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
    # FR-007's 5-point tolerance is applied to the estimate's 95% interval, not the point
    # estimate: with n=200 and chance=0.25 a point-estimate rule fails ~11% of truly-at-chance
    # families on sampling noise alone.
    se = float(np.sqrt(chance * (1 - chance) / len(instances)))
    return CalibrationResult(
        family_id=family,
        model_id=policy.model_id,
        n=len(instances),
        baseline_acc=base_acc,
        chance_acc=chance,
        matched_full_acc=full_acc,
        leak_rate=float(np.mean(leaks)) if leaks else 0.0,
        refusal_rate=float(np.mean(refusals)),
        passed=abs(base_acc - chance) <= 0.05 + 1.96 * se and full_acc >= GATE_FULL,
        baseline_mean_tokens=float(np.mean([o.ledger.total_tokens for o in base])),
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
            if arm_id == "A2_single_ref":
                strengths = [table["arms"]["A2"][lvl]["strength"]]
            else:
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
    if arm_id == "M":
        if family == "comm_parity":
            return 1
        return int(difficulty["n"]) if family.startswith("comm_") else 100
    if arm_id in ("A2", "A2_single_ref"):
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
