"""Claude adapter (research R3, R5, R6, R7).

- Model `claude-opus-5-5`; reasoning depth only through `output_config.effort`, always set
  explicitly (this model's default is `medium`). No thinking budget and no sampling parameters:
  the model rejects both.
- Final answers through structured output (`output_config.format`).
- Refusal fallbacks are deliberately NOT enabled: a fallback would silently swap models inside
  one model-family cell. A refusal is recorded and scored as wrong.
"""

from __future__ import annotations

import json
import time

from ceilings.adapters.base import (
    AdapterResult,
    ModelAdapter,
    Usage,
    normalize_probabilities,
    probability_diagnostics,
)

MODEL_ID = "claude-opus-5-5"
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00}


class ClaudeAdapter(ModelAdapter):
    model_id = MODEL_ID
    supports_batch = True

    def __init__(self, client=None, model_id: str = MODEL_ID) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model_id = model_id

    @staticmethod
    def _dollars(input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * PRICE_PER_MTOK["input"] + output_tokens * PRICE_PER_MTOK["output"]
        ) / 1e6

    def _params(self, messages, depth, answer_space, max_tokens, seed=None) -> dict:
        system = [m["content"] for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]
        output_config: dict = {"effort": depth}
        if answer_space is not None:
            output_config["format"] = {"type": "json_schema",
                                       "schema": self.schema_for(answer_space, seed)}
        params = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": convo,
            "output_config": output_config,
        }
        if system:
            params["system"] = "\n\n".join(system)
        return params

    def _parse(self, response, answer_space) -> AdapterResult:
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning_tokens=None,  # thinking is billed inside output_tokens and not split out
            dollars=self._dollars(response.usage.input_tokens, response.usage.output_tokens),
        )
        request_id = getattr(response, "_request_id", None) or getattr(response, "id", "")
        valid = getattr(response, "model", self.model_id) == self.model_id
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            refusal = {
                "category": getattr(details, "category", None),
                "explanation": getattr(details, "explanation", None),
            }
            probs = normalize_probabilities(None, answer_space) if answer_space else {}
            return AdapterResult(None, probs, refusal, usage, request_id, valid=valid)
        text = "".join(b.text for b in response.content if b.type == "text")
        extra = {"raw": text, "stop": response.stop_reason}
        if answer_space is None:
            return AdapterResult(None, {}, None, usage, request_id, text=text, valid=valid,
                                 extra=extra)
        try:
            data = json.loads(text)
            answer = data["answer"] if data["answer"] in answer_space else None
            probs = normalize_probabilities(data.get("probabilities"), answer_space)
            extra["parse"] = "json" if answer else "failed"
            extra.update(probability_diagnostics(text, answer, probs, answer_space))
        except (json.JSONDecodeError, KeyError, TypeError):
            answer, probs = None, normalize_probabilities(None, answer_space)
            extra["parse"] = "failed"
        return AdapterResult(answer, probs, None, usage, request_id, text=text, valid=valid,
                             extra=extra)

    def complete(self, messages, depth, answer_space, seed, max_tokens=16000):
        params = self._params(messages, depth, answer_space, max_tokens, seed)
        if max_tokens > 16000:
            with self.client.messages.stream(**params) as stream:
                response = stream.get_final_message()
        else:
            response = self.client.messages.create(**params)
        result = self._parse(response, answer_space)
        if answer_space is not None and result.final_answer is None and result.refusal is None:
            retry = self._parse(self.client.messages.create(**params), answer_space)  # one retry
            retry.usage.input_tokens += result.usage.input_tokens
            retry.usage.output_tokens += result.usage.output_tokens
            retry.usage.dollars += result.usage.dollars
            result = retry
        return result

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        resp = self.client.messages.count_tokens(
            model=self.model_id, messages=[{"role": "user", "content": text}]
        )
        return resp.input_tokens

    def run_batch(
        self,
        requests: list[tuple[str, list[dict], str, list[str] | None]],
        max_tokens: int = 16000,
        poll_seconds: float = 30.0,
    ) -> dict[str, AdapterResult]:
        """Message Batches (50% cost). requests: (custom_id, messages, depth, answer_space)."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        spaces = {cid: space for cid, _, _, space in requests}
        batch = self.client.messages.batches.create(
            requests=[
                Request(
                    custom_id=cid,
                    params=MessageCreateParamsNonStreaming(
                        **self._params(msgs, depth, space, max_tokens)
                    ),
                )
                for cid, msgs, depth, space in requests
            ]
        )
        while self.client.messages.batches.retrieve(batch.id).processing_status != "ended":
            time.sleep(poll_seconds)
        out: dict[str, AdapterResult] = {}
        for entry in self.client.messages.batches.results(batch.id):  # any order: key by id
            if entry.result.type == "succeeded":
                res = self._parse(entry.result.message, spaces[entry.custom_id])
                res.usage.dollars /= 2
                out[entry.custom_id] = res
        return out
