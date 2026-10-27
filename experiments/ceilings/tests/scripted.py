"""A scripted chat model that solves tasks from the rendered prompt text alone.

It never sees structured payloads, so if it can answer, the natural-language prompts carry the
information the theory says they carry. It mimics bounded reasoning: the reasoning family is solved
only when the adapter ran a reasoning stage (depth above `low`).
"""

from __future__ import annotations

import json
import re
from collections import Counter

from ceilings.adapters.base import REASON_PROMPT, ChatAdapter, ChatReply, approx_tokens


def _user_text(messages):
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


def solve_info(text: str) -> str | None:
    table = {}
    for line in text.splitlines():
        m = re.match(r"(host-\d\d): (.+)$", line)
        if m:
            table[m.group(1)] = dict(kv.rsplit(" ", 1) for kv in m.group(2).split(", "))
    votes: dict[str, Counter] = {}
    for name, value in re.findall(r"the failing host's (.+?) is (\S+?)\.", text):
        votes.setdefault(name, Counter())[value] += 1
    if not votes:
        return None
    best = {k: c.most_common(1)[0][0] for k, c in votes.items()}
    hits = [h for h, attrs in table.items() if all(attrs.get(k) == v for k, v in best.items())]
    return hits[0] if len(hits) == 1 else None


def solve_reason(text: str) -> str | None:
    shifts = "ABCD"
    m = re.search(r"Which shift does (\w+) work\?", text)
    start = re.search(r"^(\w+) works shift ([ABCD])\.$", text, re.M)
    if not (m and start):
        return None
    child_of = {}
    for child, rel, parent in re.findall(
            r"^(\w+) works (the same shift as|\d shift\(s\) after) (\w+)\.$", text, re.M):
        child_of[parent] = (child, 0 if rel.startswith("the same") else int(rel[0]))
    person, shift = start.group(1), shifts.index(start.group(2))
    while person != m.group(1) and person in child_of:
        person, off = child_of[person]
        shift = (shift + off) % 4
    return shifts[shift] if person == m.group(1) else None


def solve_comm(text: str) -> str | None:
    own = [1 if v == "yes" else 0
           for v in re.findall(r"^Candidate has [^:(]+: (yes|no)$", text, re.M)]
    msg = re.findall(r"From agent B: ([01]+)", text)
    if not own or not msg:
        return None
    bits = [int(b) for b in msg[-1]]
    if "total number of 'yes'" in text:
        return "odd" if (sum(own) + bits[0]) % 2 else "even"
    L = len(bits)
    if any(own[L:]):
        return None
    return "odd" if sum(a * b for a, b in zip(own[:L], bits, strict=True)) % 2 else "even"


class ScriptedChat(ChatAdapter):
    def __init__(self, name: str = "scripted") -> None:
        self.model_name = name
        self.model_id = f"scripted/{name}"
        self.calls = 0

    def _reply(self, text: str, messages, out: int = 10) -> ChatReply:
        self.calls += 1
        return ChatReply(text, approx_tokens(_user_text(messages)), out, f"s{self.calls}",
                         self.model_name, "stop")

    def _chat(self, messages, schema, max_tokens, seed):
        text = _user_text(messages)
        last = messages[-1]["content"]
        if last.endswith(REASON_PROMPT):
            return self._reply("Working through it step by step.", messages, max_tokens)
        if schema is not None:
            options = schema["properties"]["answer"]["enum"]
            reasoned = any(m["role"] == "assistant" for m in messages)
            if "Host table" in text:
                answer = solve_info(text)
            elif "Rota planning" in text:
                answer = solve_reason(text) if reasoned else None
            else:
                answer = solve_comm(text)
            if answer is None:
                answer = options[seed % len(options)]
                probs = {o: 1 / len(options) for o in options}
            else:
                rest = 0.02 / max(1, len(options) - 1)
                probs = {o: (0.98 if o == answer else rest) for o in options}
            return self._reply(json.dumps({"answer": answer, "probabilities": probs}), messages)
        if "Choose one attribute" in last:
            names = re.search(r"probe next: (.+?)\. Reply", last).group(1).split(", ")
            asked = Counter(re.findall(r"the failing host's (.+?) is", text))
            return self._reply(min(names, key=lambda n: asked[n]), messages, 2)
        # The message instruction is in the last user turn, or one turn earlier when the agent
        # had a reasoning stage first (the comm families reason by design).
        m = re.search(r"You may send at most (\d+) characters", last) or \
            re.search(r"You may send at most (\d+) characters", text)
        if m:
            own = [("1" if v == "yes" else "0")
                   for v in re.findall(r"^Role requires [^:]+: (yes|no)$", text, re.M)]
            if "total number of 'yes'" in text:  # parity: one bit says everything
                return self._reply(str(own.count("1") % 2), messages, 2)
            return self._reply("".join(own)[: int(m.group(1))], messages, 4)
        return self._reply("", messages, 1)
