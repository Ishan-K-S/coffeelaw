"""Communication-bound family (FR-003): the evidence is split so neither agent alone knows anything.

Formal mapping: agent A holds x, agent B holds y, both uniform on {0,1}^n.
  parity: Theta = (sum x + sum y) mod 2  (a one-bit summary from B suffices)
  ip:     Theta = <x, y> mod 2           (about n/2 bits must cross; Theorem 5)

The message format is fixed and stated to both agents, so the experiment measures how much has to
cross the channel, not whether two agents can invent a code: for parity B sends the parity of its
list (one bit); for inner product B sends its list, one bit per skill in order (the first L if the
budget is L < n). Agent B's private notes include its message already written in that format: it
is a function of B's own list, so it adds no information (data processing), but it keeps B's
encoding work out of the communication test.
"""

from __future__ import annotations

import numpy as np

from ceilings import theory
from ceilings.records import TaskInstance, TheoryQuantities, content_hash

VERSION = "4"
READY_PREFIX = "\nYour message to agent A, already written in the message format: "
SKILLS = [
    "Python",
    "SQL",
    "Kubernetes",
    "Terraform",
    "Go",
    "Rust",
    "React",
    "GraphQL",
    "Spark",
    "Kafka",
    "Airflow",
    "dbt",
    "Docker",
    "Linux",
    "Bash",
    "Java",
    "Scala",
    "C++",
    "TypeScript",
    "Node",
    "Postgres",
    "Redis",
    "AWS",
    "GCP",
    "Azure",
    "Snowflake",
    "Pandas",
    "PyTorch",
    "Jax",
    "Ray",
    "gRPC",
    "Nginx",
]
EXTRA = [
    "Mentoring",
    "Public speaking",
    "On-call",
    "Hiring",
    "Writing",
    "Design review",
    "Budgeting",
]
FORMAL_MAPPING = __doc__


def family_id(variant: str) -> str:
    return f"comm_{variant}"


def _yes(b: int) -> str:
    return "yes" if b else "no"


def generate(seed: int, n: int = 8, variant: str = "ip", split: str = "main") -> TaskInstance:
    if variant not in ("parity", "ip"):
        raise ValueError("variant must be 'parity' or 'ip'")
    if n > len(SKILLS):
        raise ValueError(f"n must be at most {len(SKILLS)}")
    # The variant is part of the seed so the two families never share private lists.
    rng = np.random.default_rng([seed, 0 if variant == "parity" else 1])
    x = [int(b) for b in rng.integers(0, 2, n)]
    y = [int(b) for b in rng.integers(0, 2, n)]
    f = theory.inner_product if variant == "ip" else theory.parity
    value = f(x, y)
    skills = SKILLS[:n]
    # Alternate the order the two options are named in, so a bias toward the last-named option
    # cannot line up with the truth.
    options = "even or odd" if rng.integers(2) == 0 else "odd or even"
    if variant == "ip":
        question = (
            f"Is the number of skills that the role requires AND the candidate has {options}?"
        )
        protocol = (
            "Message format: agent B may send agent A one character per skill, in list order. "
            "Character i is 1 if the role requires the i-th skill on the list and 0 if it does "
            "not. If the message is shorter than the list, the requirements for the remaining "
            "skills are unknown."
        )
    else:
        question = f"Across both lists, is the total number of 'yes' answers {options}?"
        protocol = (
            "Message format: agent B may send agent A a single character: 1 if the number of "
            "skills the role requires is odd, 0 if it is even."
        )
    scenario = f"Hiring decision for one candidate and one role. {question}"
    shared = (
        f"Agent A holds the candidate's skills; agent B holds the role's requirements. "
        f"Both lists use the same {n} skill names, in the same order.\n{protocol}"
    )
    priv_a = "\n".join(f"Candidate has {s}: {_yes(b)}" for s, b in zip(skills, x, strict=True))
    priv_b = "\n".join(f"Role requires {s}: {_yes(b)}" for s, b in zip(skills, y, strict=True))
    ready = str(sum(y) % 2) if variant == "parity" else "".join(str(b) for b in y)
    priv_b += f"{READY_PREFIX}{ready}"
    h = theory.theta_entropy_comm(n, variant)
    alone = theory.agent_alone_information(n, variant)
    if variant == "ip":
        bounds = [min(h, theory.ip_bound(n, L)) for L in range(n + 1)]
    else:
        bounds = [0.0] + [1.0] * n
    fid = family_id(variant)
    body = {"family": fid, "version": VERSION, "seed": seed, "n": n, "split": split}
    return TaskInstance(
        instance_id=content_hash(body),
        family_id=fid,
        family_version=VERSION,
        seed=seed,
        difficulty={"n": n, "variant": variant},
        answer="odd" if value else "even",
        answer_space=["even", "odd"],
        scenario=scenario,
        shared_evidence=shared,
        private_evidence={"A": priv_a, "B": priv_b},
        tool=None,
        theory=TheoryQuantities(
            H_theta=h,
            I_held_by_agent={"A": alone, "B": alone},
            I_held_team=h,
            comm_bound_by_L=bounds,
        ),
        split=split,
        payload={
            "public": {"n": n, "variant": variant, "protocol": protocol},
            "private": {"A": {"bits": x}, "B": {"bits": y}},
            "hidden": {"value": value},
        },
    )


def without_ready_message(private_text: str) -> str:
    """B's notes with the ready-made message removed (the `sender=encode` ablation): B must then
    encode its own list in the stated format."""
    return private_text.split(READY_PREFIX, 1)[0]


def same_kind_private_evidence(instance: TaskInstance, amount: int) -> str:
    """Arm E on this family: more private attributes of the same format that the question ignores."""
    rng = np.random.default_rng(instance.seed + 1)
    return "\n".join(
        f"Candidate has {EXTRA[i % len(EXTRA)]} (not part of the question): "
        f"{_yes(int(rng.integers(2)))}"
        for i in range(amount)
    )
