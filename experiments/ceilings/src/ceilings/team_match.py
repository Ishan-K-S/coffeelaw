"""Match a one-agent reference (S_ref) to what each team spends (arm design 3,
prereg/DESIGN3_PLAN.md).

Given the same budget, a lone agent stops after a turn or two while a team spends 2-15x as much
(November 2026 pilot), so comparing them at equal caps would compare unequal compute. H5 compares
each team instead with S_ref: the same one agent raised up LADDER (deeper reasoning per turn, then
several independent attempts and a vote) until its mean total tokens on the matching instances
are within MATCH_TOLERANCE of the team's. A team that even the top rung cannot reach is recorded
as unmatched, with the gap.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from ceilings.adapters import safe_model_dir
from ceilings.agents import Policy
from ceilings.records import TaskInstance
from ceilings.runner import Cell, cell_run_id, load_runs, run_grid
from ceilings.teams import TeamConfig, ref_config

LADDER = ["medium", "high", "xhigh", "max", "max:2", "max:3", "max:4", "max:6", "max:8", "max:12",
          "max:16"]
MATCH_TOLERANCE = 0.10


def table_path(root: Path | str, model_id: str) -> Path:
    return Path(root) / "team_match" / f"{safe_model_dir(model_id)}.json"


def mean_tokens(root: Path | str, policy: Policy, insts: list[TaskInstance],
                cfg: TeamConfig) -> float | None:
    """Mean total tokens of this config's runs on these instances, found by run ID."""
    ids = {cell_run_id(Cell(i, cfg), policy) for i in insts}
    vals = [r["ledger"]["total_tokens"] for r in load_runs(root, policy.model_id)
            if r["run_id"] in ids]
    return statistics.mean(vals) if vals else None


def match(policy: Policy, insts: list[TaskInstance], teams: dict[str, TeamConfig],
          root: Path | str, max_cost: float | None = None, concurrency: int = 1,
          progress=None) -> dict:
    """Run the teams (skipping runs already on disk), then climb LADDER until a rung spends more
    than every team; pick for each team the rung whose cost is nearest its own."""
    spent = 0.0

    def run(cells) -> None:
        nonlocal spent
        left = None if max_cost is None else max(0.0, max_cost - spent)
        spent += sum(r.ledger.dollars or 0.0 for r in
                     run_grid(cells, policy, root, left, concurrency=concurrency,
                              progress=progress))

    first = next(iter(teams.values()))
    run([Cell(i, cfg) for cfg in teams.values() for i in insts])
    targets = {arm: mean_tokens(root, policy, insts, cfg) for arm, cfg in teams.items()}
    measured: dict[str, float] = {}
    for strength in LADDER:
        cfg = ref_config(strength, first.budget, max_turns=first.max_turns)
        run([Cell(i, cfg) for i in insts])
        measured[strength] = mean_tokens(root, policy, insts, cfg)
        if measured[strength] > (1 + MATCH_TOLERANCE) * max(targets.values()):
            break
    arms = {}
    for arm, target in targets.items():
        best = min(measured, key=lambda s, t=target: abs(measured[s] - t) / t)
        gap = measured[best] / target - 1
        arms[arm] = {"target_tokens": target, "strength": best, "ref_tokens": measured[best],
                     "gap": gap, "matched": abs(gap) <= MATCH_TOLERANCE}
    table = {"model_id": policy.model_id, "instances": [i.instance_id for i in insts],
             "difficulty": dict(insts[0].difficulty), "budget": first.budget,
             "max_turns": first.max_turns, "tolerance": MATCH_TOLERANCE,
             "ladder_tokens": measured, "arms": arms, "dollars": round(spent, 6)}
    path = table_path(root, policy.model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2))
    return table


def load_table(root: Path | str, model_id: str) -> dict:
    path = table_path(root, model_id)
    if not path.exists():
        raise FileNotFoundError(f"no S_ref matching table at {path}; run `ceilings team-match`")
    return json.loads(path.read_text())


__all__ = ["LADDER", "MATCH_TOLERANCE", "match", "load_table", "table_path", "mean_tokens"]
