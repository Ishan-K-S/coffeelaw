"""Arm definitions (FR-010) and cost matching (FR-009, research R4).

Every non-baseline arm changes exactly one resource relative to baseline, then is topped up with
neutral filler text read by the answerer so its added cost hits the target delta. Filler on its own
is arm P2, so any effect of the top-up is measured directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

DEPTH_ORDER = ["low", "medium", "high", "xhigh", "max"]
# Arm design version, part of every new run ID and run record. Version 2 (after the October 2026
# paid run) fixes K, M on reason and A2; see README "Design changes after the paid run". Version 3
# adds the agent-team arms S, L, W, W0 and R (ceilings.teams, prereg/DESIGN3_PLAN.md) and changes
# no earlier arm. Runs of different versions never share a run root.
ARM_DESIGN = 3
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
    message_mode: str = "tokens"  # "tokens", "bits", or "quote" (only verbatim task lines cross)
    message_budget: int = 0
    relay: bool = False
    # Agents left out of the relay chain (A2 on comm: B, whose notes would carry M's information).
    relay_skip: tuple[str, ...] = ()
    extra_kind: str | None = None
    extra_amount: int = 0
    filler_tokens: int = 0
    strength: object = None
    # How many answer samples reason at `depth`; the rest answer at the family's baseline depth.
    # None: all of them. Design 1 used it for K; design 2 leaves it None (see `build`).
    reason_samples: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["answerers"] = list(self.answerers)
        d["relay_skip"] = list(self.relay_skip)
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
    """K's strength: "high" (all 4 samples reason) or "high:2" (the team answers with 2 samples,
    both reasoning). Design 1 kept 4 samples and reasoned on 2 of them; the 2 or 3 samples that
    did not reason out-voted the one that did (reason family, paid run: the reasoning sample was
    right 90-100% of the time, the team's vote 40-57%)."""
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
        cfg = replace(base, depth=depth, samples=n or base.samples)
    elif arm_id == "M":
        # On reason both agents hold every fact, so a free-text message lets B work the chain and
        # send the result: compute, not communication (design 1: B stated the answer in 407-782
        # of 850 messages and M beat K on every model). Only verbatim task lines cross instead.
        cfg = replace(base, message_budget=int(strength), relay=family == "reason",
                      message_mode="quote" if family == "reason" else base.message_mode)
    elif arm_id == "A1":
        # Same totals: the baseline's 4 answer calls spread over 4 agents; nothing else changes.
        cfg = replace(base, n_agents=4, samples=1, answerers=("A", "B", "C", "D"))
    elif arm_id == "A2":
        # More agents plus compute; every agent reasons and relays partial work down the chain
        # (tokens channel, generous budget). On comm, B is left out of the chain: its notes hold
        # the role's list, so any message from B would be M's resource. Design 1 used a 0-bit
        # channel there, so B, C and D never made a call.
        cfg = replace(
            base,
            n_agents=4,
            samples=1,
            depth=str(strength),
            relay=True,
            message_mode="tokens",
            message_budget=200,
            relay_skip=("B",) if family.startswith("comm_") else (),
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
        # 1, 2 or all 4 reasoning samples, each at rising depth: K could not reach a medium delta
        # in the pilot because a reasoning stage on every sample cost 1.8-2x the target. Cost
        # grows mostly with the number of samples (paid run: about 520-960 tokens per reasoning
        # sample at any depth), so the list is ordered by count first; matching's binary search
        # needs cost to rise along it.
        return _above(family, ["medium:1", "high:1", "xhigh:1", "medium:2", "high:2", "xhigh:2",
                               "medium", "high", "xhigh", "max"])
    if arm_id == "M":
        if family == "comm_parity":
            return [1]
        return [n_bits] if family.startswith("comm_") else [50, 100, 200]
    if arm_id == "A2":
        # Agents always reason. Design 1 started at "low", which cost matching chose on 8 of 9
        # reason models: four agents with no reasoning stage, the budget spent on filler.
        return [d for d in ["medium", "high"]
                if DEPTH_ORDER.index(d) >= DEPTH_ORDER.index(base_depth(family))]
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
