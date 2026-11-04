"""Stub adapter for plumbing tests: no network, deterministic per seed."""

from __future__ import annotations

import re

import numpy as np

from ceilings.adapters.base import AdapterResult, ModelAdapter, Usage, approx_tokens

ORACLE_MARKER = re.compile(r"__oracle_answer__=(\S+)")


class StubOracleAdapter(ModelAdapter):
    """Right (p=0.99) when a test fixture injects `__oracle_answer__=<a>`, otherwise a guess."""

    model_id = "stub-oracle"

    def complete(self, messages, depth, answer_space, seed, max_tokens=16000, text_tokens=None):
        text = "\n".join(str(m["content"]) for m in messages)
        rng = np.random.default_rng(seed)
        usage = Usage(input_tokens=approx_tokens(text), output_tokens=20)
        if answer_space is None:
            return AdapterResult(None, {}, None, usage, f"stub-{seed}", text="")
        m = ORACLE_MARKER.search(text)
        if m and m.group(1) in answer_space:
            known = m.group(1)
            rest = (1 - 0.99) / max(1, len(answer_space) - 1)
            probs = {a: (0.99 if a == known else rest) for a in answer_space}
            answer = known
        else:
            probs = {a: 1 / len(answer_space) for a in answer_space}
            answer = answer_space[int(rng.integers(len(answer_space)))]
        return AdapterResult(answer, probs, None, usage, f"stub-{seed}")
