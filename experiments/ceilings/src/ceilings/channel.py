"""The only path between agents (research R8). Budgets are enforced in tokens or in bits.

`quote` mode is a tokens channel that carries only lines copied word for word from the task (M on
the reason family): the sender can pass facts on but not the result of working them through."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ceilings.ledger import Ledger


@dataclass
class Delivery:
    sender: str
    receiver: str
    text: str
    tokens: int
    bits: int | None
    truncated: bool = False
    retried: bool = False


@dataclass
class Channel:
    """Message channel with a `tokens:B` or `bits:L` budget per direction."""

    mode: str
    budget: int
    ledger: Ledger
    count_tokens: Callable[[str], int]
    quotable: list[str] = field(default_factory=list)  # the task's lines, for `quote` mode
    deliveries: list[Delivery] = field(default_factory=list)
    _used: dict[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("tokens", "bits", "quote"):
            raise ValueError("mode must be 'tokens', 'bits' or 'quote'")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        self._pattern = re.compile(rf"^[01]{{0,{self.budget}}}$") if self.mode == "bits" else None

    def valid_bits(self, text: str) -> bool:
        return bool(self._pattern and self._pattern.match(text))

    def send(self, sender: str, receiver: str, compose: Callable[[str | None], str]) -> Delivery:
        """Deliver `compose(None)`.

        Bit mode: a string of 0s and 1s longer than the budget is cut to its first L bits (only L
        bits cross the channel); anything else malformed gets one retry, then '' is delivered.
        Quote mode: only lines that match a task line exactly cross, once each and in the task's
        own order (so the order cannot carry worked-out structure either).

        `compose` receives the rejection reason on retry so an agent can correct itself.
        """
        text = compose(None)
        retried = False
        truncated = False
        if self.mode == "bits":
            if re.fullmatch(r"[01]+", text or "") and len(text) > self.budget:
                text, truncated = text[: self.budget], True  # only the first L bits cross
            if not self.valid_bits(text):
                retried = True
                text = compose(f"Rejected: that reply did not follow the message format. Use at most "
                               f"{self.budget} characters, each 1 or 0, and no other text.")
                if re.fullmatch(r"[01]+", text or "") and len(text) > self.budget:
                    text, truncated = text[: self.budget], True
                if not self.valid_bits(text):
                    text = ""
            delivery = Delivery(
                sender,
                receiver,
                text,
                self.count_tokens(text) if text else 0,
                bits=len(text),
                truncated=truncated,
                retried=retried,
            )
            self.ledger.add_message(delivery.tokens, bits=len(text))
        else:
            if self.mode == "quote":
                text = self.quoted(text)
            key = (sender, receiver)
            left = self.budget - self._used.get(key, 0)
            tokens = self.count_tokens(text)
            truncated = False
            if tokens > left:
                # Quote mode cuts whole lines, so part of a fact is never delivered.
                sep = "\n" if self.mode == "quote" else " "
                parts = text.splitlines() if self.mode == "quote" else text.split()
                while parts and self.count_tokens(sep.join(parts)) > left:
                    parts.pop()
                text = sep.join(parts)
                tokens = self.count_tokens(text) if text else 0
                truncated = True
            self._used[key] = self._used.get(key, 0) + tokens
            delivery = Delivery(sender, receiver, text, tokens, bits=None, truncated=truncated)
            self.ledger.add_message(tokens)
        self.deliveries.append(delivery)
        return delivery

    def quoted(self, text: str) -> str:
        """The task lines that `text` quotes exactly (whitespace-normalised, and ignoring a list
        marker or quotes around a line, which models add), in task order."""
        def norm(line: str) -> str:
            return " ".join(line.split())

        def unwrap(line: str) -> str:
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
            return line.strip("\"'`“”")

        sent = {norm(unwrap(line)) for line in (text or "").splitlines()}
        return "\n".join(line for line in self.quotable if norm(line) and norm(line) in sent)
