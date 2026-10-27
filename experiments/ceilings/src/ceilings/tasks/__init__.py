"""Task-family registry and instance storage."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ceilings.records import TaskInstance
from ceilings.tasks import comm, info, reason

BINDING = {
    "info": "information",
    "reason": "reasoning",
    "comm_parity": "communication",
    "comm_ip": "communication",
}

GENERATORS: dict[str, Callable[..., TaskInstance]] = {
    "info": lambda seed, split="main", **d: info.generate(seed, split=split, **d),
    "reason": lambda seed, split="main", **d: reason.generate(seed, split=split, **d),
    "comm_parity": lambda seed, split="main", **d: comm.generate(
        seed, variant="parity", split=split, **d
    ),
    "comm_ip": lambda seed, split="main", **d: comm.generate(seed, variant="ip", split=split, **d),
}


def evidence_for_arm_e(instance: TaskInstance, amount: int) -> str:
    """Text injected by arm E (the information family uses extra probes instead)."""
    if instance.family_id == "reason":
        return reason.restated_evidence(instance, amount)
    if instance.family_id.startswith("comm_"):
        return comm.same_kind_private_evidence(instance, amount)
    return ""


TRIES = 4096


def generate_many(
    family: str, count: int, split: str, seed0: int = 0, balance: bool = True, **difficulty
) -> list[TaskInstance]:
    """`count` instances with deterministic seeds. With `balance`, true answers cycle through
    every option in a seeded order, so each label is the truth equally often (the pilot's 30
    instances had host-00 as the truth twice, and floor accuracy tracked that); instance i takes
    the first seed in its own block whose answer is its target label."""
    offset = {"calibration": 1_000_000, "main": 0, "heldout_blind": 2_000_000}[split]
    gen = GENERATORS[family]
    if not balance:
        return [gen(seed0 + offset + i, split=split, **difficulty) for i in range(count)]
    space = gen(seed0 + offset, split=split, **difficulty).answer_space
    targets = [space[int(j)] for j in
               np.random.default_rng([seed0, offset, len(space)]).permutation(len(space))]
    out = []
    for i in range(count):
        start = seed0 + offset + i * TRIES
        for seed in range(start, start + TRIES):
            inst = gen(seed, split=split, **difficulty)
            if inst.answer == targets[i % len(targets)]:
                out.append(inst)
                break
        else:
            raise RuntimeError(f"no {family} instance with answer {targets[i % len(targets)]}")
    return out


def chance(instance: TaskInstance) -> float:
    """Expected accuracy with no information beyond what the instance states up front."""
    known = instance.payload.get("public", {}).get("known", 0) if instance.family_id == "info" \
        else 0
    return min(1.0, 2**known / len(instance.answer_space))


def difficulty_tag(difficulty: dict) -> str:
    """Folder name for one difficulty setting, e.g. {"k": 3, "p": 0.1} -> "k3_p0.1"."""
    return "_".join(f"{k}{v}" for k, v in sorted(difficulty.items()) if k != "variant") or "default"


def instance_path(root: Path, family: str, version: str, split: str,
                  difficulty: dict | None = None) -> Path:
    return (Path(root) / "instances" / family / version / difficulty_tag(difficulty or {})
            / f"{split}.jsonl")


def save_instances(instances: list[TaskInstance], root: Path) -> Path:
    first = instances[0]
    path = instance_path(root, first.family_id, first.family_version, first.split,
                         dict(first.difficulty))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i.model_dump_json() + "\n" for i in instances))
    return path


def load_instances(root: Path, family: str, version: str, split: str,
                   difficulty: dict | None = None) -> list[TaskInstance]:
    path = instance_path(root, family, version, split, difficulty)
    return [TaskInstance.model_validate(json.loads(line)) for line in path.read_text().splitlines()]


def family_version(family: str) -> str:
    return {"info": info.VERSION, "reason": reason.VERSION}.get(family, comm.VERSION)
