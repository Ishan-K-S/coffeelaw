"""Arm definitions (FR-010) and cost matching (FR-009, research R4).

Every non-baseline arm changes exactly one resource relative to baseline, then is topped up with
neutral filler text read by the answerer so its added cost hits the target delta. Filler on its own
is arm P2, so any effect of the top-up is measured directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

DEPTH_ORDER = ["low", "medium", "high", "xhigh", "max"]
ARMS = ["baseline", "E", "K", "M", "A1", "A2", "A2_single_ref", "P1", "P2"]
DELTA_MULTIPLIERS = {"small": 0.5, "medium": 1.0, "large": 2.0}
FILLER_LINE = "-- (no content) --"
IRRELEVANT_LINES = [
    "The cafeteria rotates its soup menu every Tuesday.",
    "Parking level 3 is closed for repainting next month.",
    "The quarterly newsletter will include a crossword.",
    "Office plants are watered on Mondays and Thursdays.",
    "A new coffee machine was installed on the fourth floor.",
]


@dataclass(frozen=True)
class ArmConfig:
    arm_id: str = "baseline"
    n_agents: int = 2
    depth: str = "low"
    samples: int = 4
    answerers: tuple[str, ...] = ("A",)
    tool_rounds: int = 0
    probe_policy: str = "rotation"  # "rotation": attribute t mod k; "agent": the model chooses
    comm_sender: str = "ready"  # "ready": B's notes include its message; "encode": B writes it
    message_mode: str = "tokens"
    message_budget: int = 0
    relay: bool = False
    extra_kind: str | None = None
    extra_amount: int = 0
    filler_tokens: int = 0
    strength: object = None
    # How many answer samples reason at `depth`; the rest answer at the family's baseline depth.
    # None: all of them. Lets K add compute in steps smaller than "every sample reasons".
    reason_samples: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["answerers"] = list(self.answerers)
        return d


# Baseline reasoning depth per family. The communication families reason at "medium" in every arm:
# in the pilot no model could combine B's (correct) message with its own list without a reasoning
# stage, so the comm gate tested arithmetic, not communication; with one, only M helped (+0.60).
COMM_BASE_DEPTH = "medium"


def base_depth(family: str) -> str:
    return COMM_BASE_DEPTH if family.startswith("comm_") else "low"


def baseline(family: str) -> ArmConfig:
    mode = "bits" if family.startswith("comm_") else "tokens"
    return ArmConfig(message_mode=mode, depth=base_depth(family))


def parse_compute(strength: object) -> tuple[str, int | None]:
    """K's strength: "high" (every sample reasons) or "high:2" (two of the samples do)."""
    depth, _, n = str(strength).partition(":")
    return depth, (int(n) if n else None)


def _above(family: str, depths: list[str]) -> list[str]:
    floor = DEPTH_ORDER.index(base_depth(family))
    return [d for d in depths if DEPTH_ORDER.index(d.partition(":")[0]) > floor]


def pooled_depth(depth: str, n_agents: int) -> str:
    """The shallowest depth whose reasoning-token cap covers `n_agents` agents at `depth`, so the
    single-agent reference gets A2's pooled reasoning budget in the experiment's currency
    (tokens). The earlier step-budget mapping gave one agent a 1024-token stage against four
    A2 agents with none at "low"."""
    from ceilings.adapters.base import reasoning_tokens

    pooled = n_agents * reasoning_tokens(depth)
    return next((d for d in DEPTH_ORDER if reasoning_tokens(d) >= pooled), DEPTH_ORDER[-1])


def build(arm_id: str, family: str, strength: object = None, filler_tokens: int = 0) -> ArmConfig:
    """Arm config for one family. `strength` is the arm's own knob (see `candidates`)."""
    base = baseline(family)
    if arm_id == "baseline":
        cfg = base
    elif arm_id == "E":
        if family == "info":
            cfg = replace(base, tool_rounds=int(strength))
        else:
            kind = "restated" if family == "reason" else "same_kind"
            cfg = replace(base, extra_kind=kind, extra_amount=int(strength))
    elif arm_id == "K":
        depth, n = parse_compute(strength)
        cfg = replace(base, depth=depth, reason_samples=n)
    elif arm_id == "M":
        cfg = replace(base, message_budget=int(strength), relay=family == "reason")
    elif arm_id == "A1":
        # Same totals: the baseline's 4 answer calls spread over 4 agents; nothing else changes.
        cfg = replace(base, n_agents=4, samples=1, answerers=("A", "B", "C", "D"))
    elif arm_id == "A2":
        # More agents plus compute; agents relay partial work (tokens channel, generous budget).
        cfg = replace(
            base,
            n_agents=4,
            samples=1,
            depth=str(strength),
            relay=True,
            message_mode="tokens" if not family.startswith("comm_") else "bits",
            message_budget=200 if not family.startswith("comm_") else 0,
        )
    elif arm_id == "A2_single_ref":
        # One agent given A2's added budget as its own reasoning, chosen by cost matching (the
        # pilot topped it up with filler instead, so H5 compared A2 with an agent that got no
        # extra compute).
        cfg = replace(base, n_agents=1, samples=1, depth=str(strength))
    elif arm_id == "P1":
        cfg = replace(base, extra_kind="irrelevant")
    elif arm_id == "P2":
        cfg = replace(base, extra_kind="padding")
    else:
        raise ValueError(f"unknown arm {arm_id}")
    return replace(cfg, arm_id=arm_id, filler_tokens=filler_tokens, strength=strength)


def candidates(arm_id: str, family: str, n_bits: int | None = None) -> list[object]:
    """Strengths in increasing cost order; matching picks the largest that fits the target."""
    if arm_id == "E":
        return list(range(1, 17)) if family == "info" else [2, 4, 8, 16, 32, 64]
    if arm_id == "K":
        # Reasoning on 1, 2 or all 4 samples, then deeper: K could not reach a medium delta in
        # the pilot because a reasoning stage on every sample cost 1.8-2x the target.
        return _above(family, ["medium:1", "medium:2", "high:1", "medium", "high:2", "xhigh:1",
                               "high", "xhigh:2", "xhigh", "max"])
    if arm_id == "M":
        if family == "comm_parity":
            return [1]
        return [n_bits] if family.startswith("comm_") else [50, 100, 200]
    if arm_id == "A2":
        return [base_depth(family)] + _above(family, ["medium", "high"])
    if arm_id == "A2_single_ref":
        return [base_depth(family)] + _above(family, ["medium", "high", "xhigh", "max"])
    return [None]


def full_strength(family: str, difficulty: dict) -> tuple[str, object]:
    """The family's own intervention at full strength (calibration gate, FR-007).

    info: every attribute probed once when the probe is exact, five times when noisy (a
    Bayes-optimal reader then reaches about 97% at p=0.1, k=3, clear of the 90% gate);
    reason: depth `xhigh`; comm: the whole private list may be sent.
    """
    if family == "info":
        k, p = int(difficulty["k"]), float(difficulty["p"])
        return "E", k if p == 0 else 5 * k
    if family == "reason":
        return "K", "xhigh"
    # Parity needs one bit (the stated message format is a single character); inner product
    # needs the whole list.
    return "M", 1 if family == "comm_parity" else int(difficulty["n"])


def extra_text(kind: str | None, amount: int, filler_tokens: int, evidence: str) -> str:
    """Context injected into the answerer's view for E (text), P1, P2 and cost top-up."""
    parts = []
    if kind in ("restated", "same_kind"):
        parts.append(evidence)
    elif kind == "irrelevant":
        parts.append(
            "\n".join(
                IRRELEVANT_LINES[i % len(IRRELEVANT_LINES)]
                for i in range(max(1, filler_tokens // 10))
            )
        )
        filler_tokens = 0
    elif kind == "padding":
        parts.append("\n".join(FILLER_LINE for _ in range(max(1, filler_tokens // 5))))
        filler_tokens = 0
    if filler_tokens > 0:
        parts.append("\n".join(FILLER_LINE for _ in range(max(1, filler_tokens // 5))))
    return "\n".join(p for p in parts if p)
