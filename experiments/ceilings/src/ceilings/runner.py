"""Grid execution with content-hash run IDs, resume, transcripts, concurrency and a spending cap."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from ceilings import prereg
from ceilings.adapters import safe_model_dir
from ceilings.agents import Policy
from ceilings.arms import ArmConfig
from ceilings.orchestrator import run_team
from ceilings.records import Refusal, RunRecord, Sweep, TaskInstance, content_hash, now


class BudgetExceeded(RuntimeError):
    """Accumulated spend passed --max-cost (CLI exit code 4)."""


@dataclass
class Cell:
    instance: TaskInstance
    cfg: ArmConfig
    delta_level: str | None = None
    replicate: int = 0
    sweep: dict | None = None


def run_id_for(
    instance_id: str,
    arm_id: str,
    delta_level: str | None,
    model_id: str,
    replicate: int,
    sweep: dict | None = None,
    harness: dict | None = None,
) -> str:
    """Content hash of the run's identity. Non-default harness options are part of it (default
    runs keep the IDs they had before options existed)."""
    parts = [instance_id, arm_id, delta_level, model_id, replicate, sweep]
    return content_hash(parts + [harness] if harness else parts)


def runs_path(root: Path, model_id: str, family: str) -> Path:
    return Path(root) / "runs" / safe_model_dir(model_id) / family / "runs.jsonl"


def existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["run_id"] for line in path.read_text().splitlines() if line}


def run_cell(
    cell: Cell, policy: Policy, prereg_hash: str, transcript_dir: Path | None
) -> RunRecord:
    inst, cfg = cell.instance, cell.cfg
    rid = run_id_for(
        inst.instance_id, cfg.arm_id, cell.delta_level, policy.model_id, cell.replicate, cell.sweep,
        policy.options.non_default(),
    )
    started = now()
    outcome = run_team(inst, cfg, policy, seed=int(rid[:8], 16))
    ref = ""
    if transcript_dir is not None:
        transcript_dir.mkdir(parents=True, exist_ok=True)
        ref = str(transcript_dir / f"{rid}.json")
        Path(ref).write_text(json.dumps({
            "run_id": rid, "instance_id": inst.instance_id, "family": inst.family_id,
            "answer": inst.answer, "model_id": policy.model_id, "delta_level": cell.delta_level,
            "sweep": cell.sweep, "config": cfg.to_dict(), "harness": policy.harness_record(),
            "steps": outcome.transcript,
        }, default=str))
    return RunRecord(
        run_id=rid,
        instance_id=inst.instance_id,
        family_id=inst.family_id,
        arm_id=cfg.arm_id,
        delta_level=cell.delta_level,
        model_id=policy.model_id,
        replicate=cell.replicate,
        sweep=Sweep(**cell.sweep) if cell.sweep else None,
        harness=policy.harness_record(),
        prereg_hash=prereg_hash,
        final_answer=outcome.final_answer,
        probabilities=outcome.probabilities,
        sample_answers=outcome.sample_answers,
        correct=outcome.final_answer == inst.answer,
        valid=outcome.valid,
        refusal=Refusal(**outcome.refusal) if outcome.refusal else None,
        ledger=outcome.ledger.to_record(),
        transcript_ref=ref,
        provider_request_ids=outcome.request_ids,
        started_at=started,
        finished_at=now(),
    )


def run_grid(
    cells: Iterable[Cell],
    policy: Policy,
    root: Path | str = "data",
    max_cost: float | None = None,
    require_frozen: bool = False,
    save_transcripts: bool = True,
    concurrency: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> list[RunRecord]:
    """Run every cell not already on disk; append records as JSONL; return the new records.

    With `concurrency > 1`, cells run in worker threads (model calls are I/O-bound); records are
    still written one at a time, and the spending cap is checked after every completed run.
    """
    root = Path(root)
    prereg_hash = prereg.check_hash() if require_frozen else prereg.current_hash()
    done: dict[Path, set[str]] = {}
    todo: list[tuple[Cell, Path]] = []
    for cell in cells:
        path = runs_path(root, policy.model_id, cell.instance.family_id)
        if path not in done:
            done[path] = existing_run_ids(path)
        rid = run_id_for(cell.instance.instance_id, cell.cfg.arm_id, cell.delta_level,
                         policy.model_id, cell.replicate, cell.sweep, policy.options.non_default())
        if rid not in done[path]:
            done[path].add(rid)
            todo.append((cell, path))

    new: list[RunRecord] = []
    spent = 0.0
    lock = threading.Lock()

    def finish(record: RunRecord, path: Path) -> None:
        nonlocal spent
        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(record.model_dump_json() + "\n")
            new.append(record)
            spent += record.ledger.dollars or 0.0
            if progress:
                progress(len(new), len(todo))

    def over_budget() -> bool:
        return max_cost is not None and spent > max_cost

    if concurrency <= 1:
        for cell, path in todo:
            tdir = path.parent / "transcripts" if save_transcripts else None
            finish(run_cell(cell, policy, prereg_hash, tdir), path)
            if over_budget():
                raise BudgetExceeded(f"spent ${spent:.2f} > --max-cost ${max_cost:.2f}")
        return new

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {}
        queue = iter(todo)

        def submit_next() -> bool:
            item = next(queue, None)
            if item is None:
                return False
            cell, path = item
            tdir = path.parent / "transcripts" if save_transcripts else None
            pending[pool.submit(run_cell, cell, policy, prereg_hash, tdir)] = path
            return True

        for _ in range(concurrency):
            if not submit_next():
                break
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                path = pending.pop(fut)
                finish(fut.result(), path)
            if over_budget():
                for fut in pending:
                    fut.cancel()
                raise BudgetExceeded(f"spent ${spent:.2f} > --max-cost ${max_cost:.2f}")
            for _ in finished:
                submit_next()
    return new


def load_runs(root: Path | str = "data", model_id: str | None = None) -> list[dict]:
    root = Path(root) / "runs"
    if not root.exists():
        return []
    pattern = safe_model_dir(model_id) if model_id else "*"
    files = sorted(root.glob(f"{pattern}/*/runs.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line]


def pmap(fn: Callable, items: list, concurrency: int = 1) -> list:
    """Order-preserving parallel map used by calibration and matching."""
    if concurrency <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(fn, items))
