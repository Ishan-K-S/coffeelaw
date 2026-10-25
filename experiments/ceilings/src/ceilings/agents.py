"""Agent policies: what one agent does with the slice of the record it can see.

Two implementations share one interface:

- `SimPolicy` ("sim-bounded"): a simulated agent with an explicit step budget per depth level.
  It is not a language model. It exists so the whole pipeline can be exercised for free, and so
  the analysis can be checked against a system whose ceilings are known exactly. It reads only
  the structured mirror of the text its agent is shown (`View.public` / `View.private` / the
  record), never the hidden answer.
- `LLMPolicy`: renders the same view as chat messages and calls a `ModelAdapter`.

Every call returns a log entry (prompt and raw reply) that the orchestrator writes to the run's
transcript, so each run can be audited after the fact.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ceilings.adapters.base import (
    ANSWER_RULES,
    AdapterResult,
    ModelAdapter,
    Usage,
    approx_tokens,
    normalize_probabilities,
)
from ceilings.options import DEFAULT as DEFAULT_OPTIONS
from ceilings.options import HarnessOptions
from ceilings.records import content_hash
from ceilings.tasks.info import probe_text

STEP_BUDGET = {"low": 4, "medium": 8, "high": 16, "xhigh": 32, "max": 64}
CONFIDENT = 0.98
PROGRESS_RE = re.compile(r"PROGRESS (\S+) (\d) (\d+)")


@dataclass
class View:
    """Everything one agent can see at one moment."""

    name: str
    family_id: str
    scenario: str
    shared_text: str
    private_text: str
    public: dict
    private: dict
    answer_space: list[str]
    tool_results: list[tuple[object, str]] = field(default_factory=list)
    inbox: list[tuple[str, str]] = field(default_factory=list)  # (sender, text)
    extra_text: str = ""

    def render(self) -> str:
        parts = [f"Task: {self.scenario}", self.shared_text]
        if self.private_text:
            parts.append(f"Only you (agent {self.name}) can see this:\n{self.private_text}")
        if self.tool_results:
            attrs = self.public.get("attributes", [])
            parts.append("\n".join(probe_text(attrs, q, v) for q, v in self.tool_results))
        if self.inbox:
            parts.append(
                "Messages you received:\n"
                + "\n".join(f"From agent {s}: {t}" for s, t in self.inbox)
            )
        if self.extra_text:
            parts.append(self.extra_text)
        return "\n\n".join(p for p in parts if p)


@dataclass
class Action:
    """What a probe or message call produced, its cost, and its log entry."""

    value: object
    usage: Usage
    log: dict


class Policy:
    model_id: str
    options: HarnessOptions = DEFAULT_OPTIONS

    def harness_record(self) -> dict:
        """The harness options in effect, resolved (stored in every run record)."""
        return self.options.non_default() | {"resolved": {
            "probe_policy": self.options.probe_policy, "comm_sender": self.options.comm_sender,
            "answer_order": self.options.answer_order,
            "shuffle_options": self.options.shuffle_options,
            "comm_depth": self.options.comm_depth}}

    def choose_probe(self, view: View, depth: str, seed: int) -> Action:
        raise NotImplementedError

    def compose(self, view: View, receiver: str, depth: str, mode: str, budget: int,
                retry: str | None, seed: int) -> Action:
        raise NotImplementedError

    def answer(self, view: View, depth: str, seed: int) -> AdapterResult:
        """`AdapterResult.extra` carries the log entry under `log`."""
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return approx_tokens(text)


# --- Simulated bounded agent -------------------------------------------------------------------


def _confident(answer: str, space: list[str]) -> dict[str, float]:
    rest = (1 - CONFIDENT) / (len(space) - 1)
    return {a: (CONFIDENT if a == answer else rest) for a in space}


def _uniform(space: list[str]) -> dict[str, float]:
    return {a: 1 / len(space) for a in space}


class SimPolicy(Policy):
    """Bounded agent: solves exactly what its step budget and its view allow, otherwise guesses."""

    model_id = "sim-bounded"

    def _usage(self, view: View, budget_steps: int) -> Usage:
        return Usage(input_tokens=approx_tokens(view.render()),
                     output_tokens=10 * budget_steps + 20)

    def choose_probe(self, view, depth, seed):
        k = view.public["k"]
        counts = [0] * k
        for q, _ in view.tool_results:
            if isinstance(q, int) and 0 <= q < k:
                counts[q] += 1
        q = counts.index(min(counts))
        return Action(q, Usage(approx_tokens(view.render()), 5), {"kind": "probe", "value": q})

    def compose(self, view, receiver, depth, mode, budget, retry, seed):
        steps = STEP_BUDGET[depth]
        usage = self._usage(view, steps)
        fam = view.family_id
        text = ""
        if fam.startswith("comm_"):
            bits = view.private.get("bits")
            if bits is not None and budget > 0:
                text = (str(sum(bits) % 2) if fam == "comm_parity"
                        else "".join(str(b) for b in bits[:budget]))
        elif fam == "reason":
            progress = self._latest_progress(view)
            _, (person, shift, done) = _reason_solve(view.public, steps, progress)
            text = f"PROGRESS {person} {shift} {done}"
        return Action(text, usage, {"kind": "message", "to": receiver, "value": text})

    @staticmethod
    def _latest_progress(view: View) -> tuple[str, int, int] | None:
        best = None
        for _, msg in view.inbox:
            m = PROGRESS_RE.search(msg)
            if m and (best is None or int(m.group(3)) > best[2]):
                best = (m.group(1), int(m.group(2)), int(m.group(3)))
        return best

    def answer(self, view, depth, seed):
        steps = STEP_BUDGET[depth]
        usage = self._usage(view, steps)
        space = view.answer_space
        fam = view.family_id
        probs = _uniform(space)
        if fam == "info":
            probs = _info_posterior(view, steps)
        elif fam == "reason":
            value, _ = _reason_solve(view.public, steps, self._latest_progress(view))
            if value is not None:
                probs = _confident(space[value], space)
        elif fam.startswith("comm_"):
            probs = _comm_posterior(view, steps)
        best = max(probs, key=probs.get)
        # An uninformed agent guesses the same way whenever it sees the same text, as a model
        # given the same prompt tends to; this keeps paired comparisons efficient.
        guess = space[
            int(content_hash([view.scenario, view.shared_text, view.private_text])[:8], 16)
            % len(space)
        ]
        answer = best if probs[best] > 1 / len(space) + 1e-9 else guess
        probs = normalize_probabilities(probs, space)
        return AdapterResult(answer, probs, None, usage, f"sim-{seed}",
                             extra={"log": {"kind": "answer", "value": answer}})


def _reason_solve(public, steps, progress):
    from ceilings.tasks.reason import solve

    return solve(public, steps, progress)


def _info_posterior(view: View, steps: int) -> dict[str, float]:
    k, p = view.public["k"], view.public["p"]
    results = [(q, v) for q, v in view.tool_results if v in ("0", "1")]
    usable = results[: steps * 4]  # Bayes updates cost 1 step per 4 results
    weights = []
    for theta in range(2**k):
        w = 1.0
        for q, v in usable:
            w *= (1 - p) if ((theta >> q) & 1) == int(v) else p
        weights.append(w)
    total = sum(weights)
    return {view.answer_space[i]: w / total for i, w in enumerate(weights)}


def _comm_posterior(view: View, steps: int) -> dict[str, float]:
    space = view.answer_space
    own = view.private.get("bits")
    n = view.public["n"]
    if own is None or steps < math.ceil(n / 4):  # one step per 4 bits combined
        return _uniform(space)
    msg = ""
    for _, m in view.inbox:
        if re.fullmatch(r"[01]*", m):
            msg = m
    if view.family_id == "comm_parity":
        if not msg:
            return _uniform(space)
        value = (sum(own) + int(msg[0])) % 2
        return _confident(space[value], space)
    L = min(len(msg), n)
    partial = sum(a * int(b) for a, b in zip(own[:L], msg[:L], strict=True)) % 2
    if all(b == 0 for b in own[L:]):
        return _confident(space[partial], space)
    return _uniform(space)


# --- Language-model agent ----------------------------------------------------------------------

SYSTEM = (
    "You are agent {name}, one member of a small team working on a task. Use only the "
    "information you are shown in this conversation."
)


def parse_probe(text: str, attributes: list) -> object:
    """Map a reply to an attribute index: the reply is an attribute name (or its index), or names
    exactly one attribute. Anything else (prose naming several, a reply cut off mid-sentence) is
    passed through and the tool returns ERR, so the harness never chooses for the model."""
    low = (text or "").strip().lower().strip(" .\"'`*")
    names = [name.lower() for name, _, _ in attributes]
    if low in names:
        return names.index(low)
    m = re.fullmatch(r"\s*(\d+)\s*", low)
    if m and int(m.group(1)) < len(attributes):
        return int(m.group(1))
    named = [i for i, n in enumerate(names) if re.search(rf"\b{re.escape(n)}\b", low)]
    if len(named) == 1:
        return named[0]
    return (text or "").strip()[:40]


def clean_bits(text: str) -> str:
    """Strip code fences, quotes and whitespace around a bit string; keep everything else intact
    so the channel can still reject a malformed message."""
    t = (text or "").strip().strip("`").strip().strip("\"'").strip()
    return re.sub(r"\s+", "", t) if re.fullmatch(r"[01\s]*", t) else t


class LLMPolicy(Policy):
    def __init__(self, adapter: ModelAdapter, options: HarnessOptions = DEFAULT_OPTIONS) -> None:
        self.adapter = adapter
        self.model_id = adapter.model_id
        self.options = options
        adapter.configure(options)

    def harness_record(self) -> dict:
        from ceilings.adapters.base import DEPTHS

        rec = super().harness_record()
        rec["resolved"]["temperature"] = getattr(self.adapter, "temperature", None)
        rec["resolved"]["reasoning_tokens"] = [self.adapter.reasoning_budget(d) for d in DEPTHS]
        if getattr(self.adapter, "profile", None):
            rec["resolved"]["provider_profile"] = self.adapter.profile
        return rec

    def _messages(self, view: View, instruction: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM.format(name=view.name)},
            {"role": "user", "content": f"{view.render()}\n\n{instruction}"},
        ]

    @staticmethod
    def _log(kind: str, messages: list[dict], res: AdapterResult, value) -> dict:
        return {"kind": kind, "prompt": messages, "raw": res.text, "request_id": res.request_id,
                "reasoning": res.extra.get("reasoning", ""),
                "reasoning_stop": res.extra.get("reasoning_stop"), "parse": res.extra.get("parse"),
                "stop": res.extra.get("stop"), "value": value,
                "sent": res.extra.get("sent"),
                "raw_prob_sum": res.extra.get("raw_prob_sum"),
                "answer_not_argmax": res.extra.get("answer_not_argmax"),
                "hidden_reasoning": res.extra.get("hidden_reasoning"),
                "schema_mode": res.extra.get("schema_mode"),
                "usage": [res.usage.input_tokens, res.usage.output_tokens,
                          res.usage.reasoning_tokens]}

    def choose_probe(self, view, depth, seed):
        attrs = view.public["attributes"]
        names = ", ".join(a[0] for a in attrs)
        msgs = self._messages(
            view, f"Choose one attribute to probe next: {names}. Reply with only the attribute "
                  "name.")
        res = self.adapter.complete(msgs, "low", None, seed, max_tokens=50)
        q = parse_probe(res.text, attrs)
        return Action(q, res.usage, self._log("probe", msgs, res, q))

    def compose(self, view, receiver, depth, mode, budget, retry, seed):
        if mode == "bits":
            n = view.public.get("n")
            if view.family_id == "comm_ip" and n and budget < n:
                length = (f"You may send at most {budget} characters, so send only the first "
                          f"{budget} characters of the full message.")
            else:
                length = f"You may send at most {budget} characters."
            rule = (
                f"Send agent {receiver} your message now, in the message format described above. "
                f"{length} Your entire reply must be the message characters themselves, with no "
                "other words."
            )
        else:
            rule = (
                f"Write a short message for agent {receiver} that helps answer the question, for "
                f"example a partial result you worked out. Use at most {budget} tokens. Reply with "
                "only the message."
            )
        if retry:
            rule = f"{retry} {rule}"
        msgs = self._messages(view, rule)
        res = self.adapter.complete(msgs, depth, None, seed, max_tokens=4000)
        text = clean_bits(res.text) if mode == "bits" else (res.text or "").strip()
        return Action(text, res.usage, self._log("message", msgs, res, text))

    def answer(self, view, depth, seed):
        msgs = self._messages(view, f"Answer the question. {ANSWER_RULES}")
        res = self.adapter.complete(msgs, depth, view.answer_space, seed)
        res.extra["log"] = self._log("answer", msgs, res, res.final_answer)
        return res

    def count_tokens(self, text: str) -> int:
        return self.adapter.count_tokens(text)


def make_policy(model_id: str, options: HarnessOptions = DEFAULT_OPTIONS) -> Policy:
    if model_id == "sim-bounded":
        policy = SimPolicy()
        policy.options = options
        return policy
    from ceilings.adapters import make_adapter

    return LLMPolicy(make_adapter(model_id), options)


__all__ = ["View", "Action", "Policy", "SimPolicy", "LLMPolicy", "make_policy", "STEP_BUDGET"]
