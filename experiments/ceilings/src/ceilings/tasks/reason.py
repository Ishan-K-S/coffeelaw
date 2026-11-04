"""Reasoning-bound family (FR-002): every fact is in context; the answer needs d chained steps.

Formal mapping: shift(P_0) = s_0 is given; shift(P_i) = shift(P_{i-1}) + a_i (mod 4) for
i = 1..d, listed in shuffled order. Theta = shift(P_d). I(Theta; D_0) = H(Theta) = 2 bits, so any
failure is a reasoning gap (Theorem 2), never missing information.
"""

from __future__ import annotations

import numpy as np

from ceilings.records import TaskInstance, TheoryQuantities, content_hash

FAMILY_ID = "reason"
VERSION = "1"
SHIFTS = ["A", "B", "C", "D"]
NAMES = [
    "Ada",
    "Bram",
    "Cleo",
    "Dev",
    "Esme",
    "Finn",
    "Gita",
    "Hugo",
    "Ines",
    "Jory",
    "Kemi",
    "Lars",
    "Mira",
    "Nils",
    "Opal",
    "Pia",
    "Quin",
    "Rosa",
    "Sven",
    "Tara",
    "Ugo",
    "Vera",
    "Wes",
    "Xena",
    "Yuri",
    "Zola",
    "Arlo",
    "Bea",
    "Cyrus",
    "Dina",
    "Ezra",
    "Faye",
    "Gus",
    "Hana",
    "Ivo",
    "Juno",
]
FORMAL_MAPPING = "Chain of d modular additions over Z_4 with shuffled premises; Theta = value at the chain's end."


def _line(child: str, parent: str, offset: int) -> str:
    step = "the same shift as" if offset == 0 else f"{offset} shift(s) after"
    return f"{child} works {step} {parent}."


SECTIONS = 3  # arm design 3: the chain is cut into this many consecutive parts


def generate(seed: int, d: int = 12, split: str = "main", layout: str = "shuffled") -> TaskInstance:
    """`layout="shuffled"` lists every premise in one shuffled list (designs 1 and 2).
    `layout="sections"` (arm design 3) cuts the chain into SECTIONS consecutive parts, listed in
    chain order with headers, each part's own lines shuffled (the starting fact is in part 1), so
    a team can divide the work by part the way coding agents divide a repo by module."""
    if d + 1 > len(NAMES):
        raise ValueError(f"d must be at most {len(NAMES) - 1}")
    if layout not in ("shuffled", "sections"):
        raise ValueError("layout must be 'shuffled' or 'sections'")
    if layout == "sections" and d < SECTIONS:
        raise ValueError(f"the sections layout needs d >= {SECTIONS}")
    rng = np.random.default_rng(seed)
    people = [NAMES[i] for i in rng.permutation(len(NAMES))[: d + 1]]
    start = int(rng.integers(4))
    offsets = [int(rng.integers(4)) for _ in range(d)]
    edges = [(people[i + 1], people[i], offsets[i]) for i in range(d)]
    value = (start + sum(offsets)) % 4
    target = people[-1]
    scenario = (
        f"Rota planning. Shifts cycle A, B, C, D, then back to A. Which shift does {target} work?"
    )
    start_line = f"{people[0]} works shift {SHIFTS[start]}."
    sections = None
    if layout == "shuffled":
        order = rng.permutation(d)
        shuffled = [edges[i] for i in order]
        lines = [start_line] + [_line(*e) for e in shuffled]
    else:
        shuffled, lines, sections = [], [], []
        for k, idx in enumerate(np.array_split(np.arange(d), SECTIONS)):
            part = [edges[i] for i in idx]
            text = ([start_line] if k == 0 else []) + [_line(*e) for e in part]
            perm = rng.permutation(len(text))
            lines.append(f"Rota, part {k + 1} of {SECTIONS}:")
            lines += [text[i] for i in perm]
            shuffled += part
            # ground truth for grading what agents report about each part (H5c)
            sections.append({"from": people[int(idx[0])], "to": people[int(idx[-1]) + 1],
                             "offset": sum(offsets[int(i)] for i in idx) % 4,
                             "to_shift": SHIFTS[(start + sum(offsets[: int(idx[-1]) + 1])) % 4]})
    shared = "\n".join(lines)
    body = {"family": FAMILY_ID, "version": VERSION, "seed": seed, "d": d, "split": split}
    difficulty: dict = {"d": d}
    if layout != "shuffled":  # shuffled instances keep the IDs they always had
        body["layout"] = difficulty["layout"] = layout
    return TaskInstance(
        instance_id=content_hash(body),
        family_id=FAMILY_ID,
        family_version=VERSION,
        seed=seed,
        difficulty=difficulty,
        answer=SHIFTS[value],
        answer_space=list(SHIFTS),
        scenario=scenario,
        shared_evidence=shared,
        private_evidence={"A": "", "B": ""},
        tool=None,
        # every agent sees every premise; design-3 teams have up to four agents
        theory=TheoryQuantities(H_theta=2.0, I_held_team=2.0, I_held_by_agent=dict.fromkeys(
            "AB" if layout == "shuffled" else "ABCD", 2.0)),
        split=split,
        payload={
            "public": {
                "start": [people[0], start],
                "edges": [list(e) for e in shuffled],
                "target": target,
                "d": d,
                **({"sections": sections} if sections else {}),
            },
            "private": {"A": {}, "B": {}},
            "hidden": {"value": value},
        },
    )


def solve(
    public: dict, budget: int, progress: tuple[str, int, int] | None = None
) -> tuple[int | None, tuple[str, int, int]]:
    """Follow the chain for at most `budget` steps, optionally resuming from `progress`.

    Returns (answer index or None, progress=(person, shift, steps_done)).
    """
    child_of = {parent: (child, off) for child, parent, off in public["edges"]}
    person, shift = public["start"][0], public["start"][1]
    done = 0
    if progress is not None:
        person, shift, done = progress
    steps = 0
    while person != public["target"] and steps < budget:
        child, off = child_of[person]
        person, shift = child, (shift + off) % 4
        steps += 1
    done += steps
    return (shift if person == public["target"] else None), (person, shift, done)


def restated_evidence(instance: TaskInstance, amount: int) -> str:
    """Arm E on this family: paraphrases of premises already given (no new information)."""
    edges = instance.payload["public"]["edges"]
    out = []
    for i in range(amount):
        child, parent, off = edges[i % len(edges)]
        out.append(f"(Restating) {parent}'s shift plus {off} gives {child}'s shift.")
    return "\n".join(out)
