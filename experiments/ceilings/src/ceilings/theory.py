"""Exact information-theoretic quantities used by the generators, tests and analysis.

All logarithms are base 2. Reference values come from litreview-coffee-law-check.py.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping

Joint = Mapping[tuple[Hashable, Hashable], float]


def entropy(probs: Iterable[float]) -> float:
    """Shannon entropy of a distribution given as probabilities."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


def binary_entropy(p: float) -> float:
    """h(p) for a coin that lands 1 with probability p."""
    return entropy([p, 1 - p])


def mutual_information(joint: Joint) -> float:
    """I(Theta; Obs) from a joint distribution {(theta, obs): prob}."""
    p_t: dict[Hashable, float] = defaultdict(float)
    p_o: dict[Hashable, float] = defaultdict(float)
    for (t, o), p in joint.items():
        p_t[t] += p
        p_o[o] += p
    return sum(p * math.log2(p / (p_t[t] * p_o[o])) for (t, o), p in joint.items() if p > 0)


def map_error(joint: Joint) -> float:
    """Error of the maximum-a-posteriori guess of Theta from Obs."""
    by_obs: dict[Hashable, dict[Hashable, float]] = defaultdict(lambda: defaultdict(float))
    for (t, o), p in joint.items():
        by_obs[o][t] += p
    return 1 - sum(max(d.values()) for d in by_obs.values())


def bsc_capacity(p: float) -> float:
    """Capacity of a binary symmetric channel that flips with probability p."""
    return 1 - binary_entropy(p)


def budget(i0: float, capacities: Iterable[float]) -> float:
    """B_T = I(Theta; D_0) + sum of per-round capacities (Theorem 1)."""
    return i0 + sum(capacities)


def fano_floor(h_theta: float, bud: float, m: int) -> float:
    """Theorem 1 error floor: (H - min(H, B) - 1) / log2(M - 1), clipped at 0. Needs M >= 3."""
    if m < 3:
        raise ValueError("Fano's inequality in this form needs at least 3 possible answers")
    return max(0.0, (h_theta - min(h_theta, bud) - 1) / math.log2(m - 1))


def fano_floor_strong(h_theta: float, bud: float, m: int) -> float:
    """The full Fano bound: the smallest error Pe with h(Pe) + Pe log2(M - 1) >= H - min(H, B),
    for Pe in [0, 1 - 1/M]. Tighter than `fano_floor` (which drops h(Pe) <= 1) and valid for
    M >= 2; the pilot's H6 cells at T >= 2 could not reject at all under the weak form."""
    need = h_theta - min(h_theta, bud)
    if need <= 0 or m < 2:
        return 0.0

    def lhs(pe: float) -> float:
        h = 0.0 if pe in (0.0, 1.0) else -pe * math.log2(pe) - (1 - pe) * math.log2(1 - pe)
        return h + (pe * math.log2(m - 1) if m > 2 else 0.0)

    lo, hi = 0.0, 1 - 1 / m
    if lhs(hi) < need:
        return hi
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if lhs(mid) < need else (lo, mid)
    return hi


def cor2_min_rounds(k: int, c: float, eps: float) -> float:
    """Corollary 2: tool rounds needed for error <= eps when Theta is uniform on 2^k values."""
    return ((1 - eps) * k - 1) / c


def ip_bound(n: int, L: int) -> float:
    """Theorem 5(b): bits about <x,y> mod 2 agent 1 can hold after an L-bit protocol."""
    return math.log2(1 + 2 ** (L + 1 - n / 2))


def _vectors(n: int) -> list[tuple[int, ...]]:
    return list(itertools.product((0, 1), repeat=n))


def inner_product(x: Iterable[int], y: Iterable[int]) -> int:
    return sum(a * b for a, b in zip(x, y, strict=True)) % 2


def parity(x: Iterable[int], y: Iterable[int]) -> int:
    return (sum(x) + sum(y)) % 2


def best_one_bit_message_ip(n: int) -> float:
    """Exhaustive optimum of I(Theta; x, m(y)) over all one-bit messages m (n <= 4)."""
    if n > 4:
        raise ValueError("exhaustive search is only feasible for n <= 4")
    vecs = _vectors(n)
    w = 1 / len(vecs) ** 2
    best = 0.0
    for mask in range(2 ** len(vecs)):
        joint: dict = defaultdict(float)
        for x in vecs:
            for yi, y in enumerate(vecs):
                joint[(inner_product(x, y), (x, (mask >> yi) & 1))] += w
        best = max(best, mutual_information(joint))
    return best


def ip_prefix_information(n: int, L: int) -> float:
    """I(Theta; x, first L bits of y) for the inner-product split."""
    vecs = _vectors(n)
    w = 1 / len(vecs) ** 2
    joint: dict = defaultdict(float)
    for x in vecs:
        for y in vecs:
            joint[(inner_product(x, y), (x, y[:L]))] += w
    return mutual_information(joint)


def agent_alone_information(n: int, variant: str) -> float:
    """I(Theta; x) for one agent of a split task (by symmetry the same for y)."""
    f = inner_product if variant == "ip" else parity
    vecs = _vectors(n) if n <= 10 else None
    if vecs is not None:
        w = 1 / len(vecs) ** 2
        joint: dict = defaultdict(float)
        for x in vecs:
            for y in vecs:
                joint[(f(x, y), x)] += w
        return mutual_information(joint)
    if variant == "parity":
        return 0.0
    # Closed form for IP: Theta = 0 surely when x = 0, otherwise uniform.
    p1 = (1 - 2.0**-n) / 2
    return binary_entropy(p1) - (1 - 2.0**-n)


def theta_entropy_comm(n: int, variant: str) -> float:
    if variant == "parity":
        return 1.0
    return binary_entropy((1 - 2.0**-n) / 2)


# --- Noisy bit-probe channel used by the information family -----------------------------------


def probe_joint(k: int, p: float, queries: list[int]) -> dict:
    """Joint of (Theta, answers) for a uniform k-bit Theta probed at fixed bit indices."""
    joint: dict = defaultdict(float)
    for theta in range(2**k):
        for flips in itertools.product((0, 1), repeat=len(queries)):
            prob = 1 / 2**k
            answers = []
            for q, f in zip(queries, flips, strict=True):
                prob *= p if f else 1 - p
                answers.append(((theta >> q) & 1) ^ f)
            joint[(theta, tuple(answers))] += prob
    return joint


def probe_information(k: int, p: float, T: int) -> float:
    """Information from T probes asked round-robin over the k bits (exact)."""
    queries = [t % k for t in range(T)]
    if T == 0:
        return 0.0
    return mutual_information(probe_joint(k, p, queries))


def probe_map_error(k: int, p: float, T: int) -> float:
    """MAP error after T round-robin probes (exact)."""
    if T == 0:
        return 1 - 1 / 2**k
    return map_error(probe_joint(k, p, [t % k for t in range(T)]))
