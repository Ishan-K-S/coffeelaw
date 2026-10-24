"""Harness options: design choices the ablations vary, held fixed within one run root.

The defaults are the main design. Each ablation changes one option, so a result that holds only
under the default choice shows up as a difference between roots:

| key       | option             | default       | alternative                                    |
|-----------|--------------------|---------------|------------------------------------------------|
| `probe`   | `probe_policy`     | `rotation`    | `agent`: the model picks each attribute        |
| `sender`  | `comm_sender`      | `ready`       | `encode`: B's ready-made message is removed    |
| `schema`  | `answer_order`     | `probs_first` | `answer_first`: answer, then probabilities     |
| `shuffle` | `shuffle_options`  | `on`          | `off`: options in their given order            |
| `temp`    | `temperature`      | adapter's     | e.g. `0`                                       |
| `reason`  | `reasoning_tokens` | adapter's     | five caps low..max, e.g. `0:128:512:1024:2048` |
| `comm_depth` | `comm_depth`    | none          | every call in the comm families reasons at least this deep (e.g. `medium`), so the gate tests communication rather than mental arithmetic |

Delta levels (small/medium/large) need no option: `--delta` already selects them.

`--harness probe=agent,temp=0` on the CLI. Non-default options are part of every run ID, and the
full resolved set is stored in every run record (`harness`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

CHOICES = {
    "probe_policy": ("rotation", "agent"),
    "comm_sender": ("ready", "encode"),
    "answer_order": ("probs_first", "answer_first"),
    "comm_depth": (None, "low", "medium", "high", "xhigh", "max"),
}
KEYS = {"probe": "probe_policy", "sender": "comm_sender", "schema": "answer_order",
        "shuffle": "shuffle_options", "temp": "temperature", "reason": "reasoning_tokens",
        "comm_depth": "comm_depth"}


@dataclass(frozen=True)
class HarnessOptions:
    probe_policy: str = "rotation"
    comm_sender: str = "ready"
    answer_order: str = "probs_first"
    shuffle_options: bool = True
    temperature: float | None = None  # None: the adapter's setting (CEILINGS_TEMPERATURE)
    reasoning_tokens: tuple[int, ...] | None = None  # None: the adapter's caps
    comm_depth: str | None = None  # floor on reasoning depth for every comm-family call

    def __post_init__(self) -> None:
        for name, allowed in CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.reasoning_tokens is not None and len(self.reasoning_tokens) != 5:
            raise ValueError("reasoning_tokens needs five caps (low, medium, high, xhigh, max)")

    @classmethod
    def parse(cls, text: str | None) -> HarnessOptions:
        """`probe=agent,schema=answer_first,shuffle=off,temp=0,reason=0:128:512:1024:2048`."""
        opts = cls()
        for part in filter(None, (p.strip() for p in (text or "").split(","))):
            key, _, value = part.partition("=")
            name = KEYS.get(key.strip(), key.strip())
            if name not in {f.name for f in fields(cls)}:
                raise ValueError(f"unknown harness option {key!r}; use {', '.join(KEYS)}")
            value = value.strip()
            if name == "shuffle_options":
                parsed: object = value.lower() in ("on", "true", "1", "yes")
            elif name == "temperature":
                parsed = float(value)
            elif name == "reasoning_tokens":
                parsed = tuple(int(v) for v in value.split(":"))
            else:
                parsed = value
            opts = replace(opts, **{name: parsed})
        return opts

    def non_default(self) -> dict:
        default = HarnessOptions()
        return {k: v for k, v in asdict(self).items() if v != getattr(default, k)}


DEFAULT = HarnessOptions()
