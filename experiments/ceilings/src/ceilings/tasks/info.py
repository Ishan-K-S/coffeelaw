"""Information-bound family (FR-001): a fresh hidden host reachable only through a noisy probe.

Formal mapping: Theta is uniform on {0..2^k-1}. Host i has k two-valued attributes; attribute q
takes its second value exactly when bit q of i is 1. The probe reports one attribute of the failing
host through a binary symmetric channel with flip rate p, so each call carries at most 1 - h(p) bits
(Theorem 1). Agents only look values up in the host table; no binary arithmetic is needed.

Version 4 (after the free-tier pilot): the host table is listed in a seeded random row order per
instance, because uninformed models default to the first row (host-00 in 24-30 of 30 runs), which
made floor accuracy track how often host-00 was the truth. `known` attributes of the failing host
can be stated up front (the partial-information setting): the team then holds `known` bits, the
answer keeps k - known bits of uncertainty, and chance accuracy is 2^known / 2^k.
"""

from __future__ import annotations

import numpy as np

from ceilings import theory
from ceilings.records import TaskInstance, TheoryQuantities, ToolSpec, content_hash

FAMILY_ID = "info"
VERSION = "4"
T_MAX = 16
ATTRIBUTES = [
    ("rack", "A", "B"),
    ("row", "front", "back"),
    ("zone", "east", "west"),
    ("power feed", "1", "2"),
    ("switch", "sw-1", "sw-2"),
    ("tier", "gold", "silver"),
]
FORMAL_MAPPING = __doc__


def host_name(i: int) -> str:
    return f"host-{i:02d}"


def attribute_value(host: int, q: int) -> str:
    name, v0, v1 = ATTRIBUTES[q]
    return v1 if (host >> q) & 1 else v0


def probe_text(attributes: list, query: object, result: str) -> str:
    """How a probe result reads to an agent."""
    if result not in ("0", "1") or not isinstance(query, int):
        return f"Probe result: invalid request {query!r} (no such attribute); nothing learned."
    name, v0, v1 = attributes[query]
    return f"Probe result: the failing host's {name} is {v1 if result == '1' else v0}."


def generate(seed: int, k: int = 3, p: float = 0.1, split: str = "main",
             known: int = 0) -> TaskInstance:
    if not 1 <= k <= len(ATTRIBUTES):
        raise ValueError(f"k must be between 1 and {len(ATTRIBUTES)}")
    if not 0 <= known < k:
        raise ValueError("known must be between 0 and k - 1")
    rng = np.random.default_rng(seed)
    hidden = int(rng.integers(2**k))
    order = [int(i) for i in rng.permutation(2**k)]
    hosts = [host_name(i) for i in range(2**k)]
    attrs = [list(a) for a in ATTRIBUTES[:k]]
    names = ", ".join(a[0] for a in attrs)
    scenario = (
        f"Incident triage. Exactly one of {2**k} hosts ({hosts[0]} to {hosts[-1]}) is failing. "
        "Which host is it?"
    )
    table = "\n".join(
        f"{host_name(i)}: " + ", ".join(f"{a[0]} {attribute_value(i, q)}" for q, a in
                                        enumerate(attrs))
        for i in order
    )
    known_text = "".join(
        f"\nAlready known: the failing host's {attrs[q][0]} is {attribute_value(hidden, q)}."
        for q in range(known))
    reliability = (
        "Probe reports are always correct." if p == 0 else
        f"Each probe report is wrong with probability {p}, independently of the others; a wrong "
        "report shows the attribute's other value."
    )
    shared = (
        f"Host table (each host differs in at least one attribute):\n{table}\n{known_text}\n"
        f"The team may receive probe reports. A probe report states one attribute ({names}) of "
        f"the failing host. {reliability} Apart from probe reports"
        + (" and what is already known" if known else "")
        + ", nothing you know says which host is failing."
    )
    c = theory.bsc_capacity(p)
    budgets = [theory.budget(0.0, [c] * T) for T in range(T_MAX + 1)]
    left = float(k - known)  # bits of uncertainty the probes must resolve
    floors = [theory.fano_floor_strong(left, b, 2 ** (k - known)) for b in budgets]
    tool = ToolSpec(
        kind="bit_probe",
        answer_alphabet=["0", "1", "ERR"],
        flip_rate=p,
        capacity_bits=c,
        charge_tokens=len(probe_text(attrs, 0, "0")) // 4,
    )
    body = {"family": FAMILY_ID, "version": VERSION, "seed": seed, "k": k, "p": p, "split": split}
    if known:
        body["known"] = known
    return TaskInstance(
        instance_id=content_hash(body),
        family_id=FAMILY_ID,
        family_version=VERSION,
        seed=seed,
        difficulty={"k": k, "p": p, **({"known": known} if known else {})},
        answer=host_name(hidden),
        answer_space=hosts,
        scenario=scenario,
        shared_evidence=shared,
        private_evidence={"A": "", "B": ""},
        tool=tool,
        theory=TheoryQuantities(
            H_theta=float(k),
            I_held_by_agent={"A": float(known), "B": float(known)},
            I_held_team=float(known),
            budget_by_T=budgets,
            fano_floor_by_T=floors,
        ),
        split=split,
        payload={
            "public": {"k": k, "p": p, "attributes": attrs, "known": known, "order": order},
            "private": {"A": {}, "B": {}},
            "hidden": {"value": hidden},
        },
    )
