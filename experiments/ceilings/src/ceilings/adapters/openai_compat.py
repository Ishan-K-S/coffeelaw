"""Adapter for any server exposing an OpenAI-style `/chat/completions` endpoint.

This covers hosted providers (including NVIDIA build, https://integrate.api.nvidia.com/v1) and
self-hosted servers (vLLM, llama.cpp server, LM Studio, and so on). Model IDs look like
`openai/<model-name>`. Configuration comes from the environment:

- `CEILINGS_OPENAI_BASE_URL`: base URL including the version path, e.g. https://host/v1
- `CEILINGS_OPENAI_API_KEY`: bearer token (omit for unauthenticated local servers)
- `CEILINGS_OPENAI_PRICE_IN` / `CEILINGS_OPENAI_PRICE_OUT`: optional USD per million tokens, so
  `--max-cost` can stop a run
- `CEILINGS_OPENAI_RPM`: requests per minute for this process (default 0: no limit). Free hosted
  tiers count requests, not dollars, so this is the throttle that matters there.
- `CEILINGS_OPENAI_RETRIES`: attempts per request on 404/5xx/network errors (default 8), with
  exponential backoff capped at 60 s, or the server's `Retry-After` when it sends one
- `CEILINGS_OPENAI_TIMEOUT`: seconds to wait for one reply before retrying (default 600); a
  flaky endpoint that sometimes hangs is cheaper with a short timeout (NVIDIA build: 120)
- `CEILINGS_OPENAI_RETRIES_429`: separate allowance for rate-limit replies (default 80, about an
  hour of backoff), since those clear once the load drops
- `CEILINGS_OPENAI_SCHEMA`: how to ask for structured output: `json_schema` (default; falls back
  to `nvext` and then to text on a 400), `nvext` (NVIDIA's `nvext.guided_json`), or `none`
- `CEILINGS_OPENAI_EXTRA_BODY`: optional JSON merged into every request body, e.g.
  `{"chat_template_kwargs": {"thinking": false}}` to switch off a model's built-in reasoning
- `CEILINGS_OPENAI_PROFILES`: optional JSON file of per-model settings (see
  `configs/nvidia_build.json`): `extra_body`, `answer_max_tokens`, `text_max_tokens`, `schema`,
  and `served_aliases`, names the provider is allowed to report for this model. An alias is a
  recorded decision, not a silent substitution: the profile is stored in every run record.
- `CEILINGS_TEMPERATURE`: sampling temperature, always sent explicitly (default 0.7)

Reasoning models may put hidden reasoning in a `<think>...</think>` block or a separate
`reasoning_content` field. It is removed before the answer is parsed and kept in the transcript;
its tokens are already part of `completion_tokens`, so the ledger charges for it.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

from ceilings.adapters.base import ChatAdapter, ChatReply, RequestCapExceeded

SCHEMA_MODES = ("json_schema", "nvext", "none")
THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.S)


def split_think(text: str) -> tuple[str, str]:
    """(answer text, hidden reasoning): strips `<think>` blocks, including an unclosed one, and a
    stray `</think>` left when the server already removed the opening tag."""
    text = text or ""
    hidden = "\n".join(m.group(1).strip() for m in THINK_RE.finditer(text))
    visible = THINK_RE.sub("", text)
    if "</think>" in visible:
        before, visible = visible.rsplit("</think>", 1)
        hidden = "\n".join(p for p in (hidden, before.strip()) if p)
    return visible.strip(), hidden


class RateLimiter:
    """Spaces request starts at least 60/rpm seconds apart across all threads of a process."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            time.sleep(start - now)


class OpenAICompatAdapter(ChatAdapter):
    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None, retries: int | None = None) -> None:
        self.model_name = model_name
        self.model_id = f"openai/{model_name}"
        base = base_url or os.environ.get("CEILINGS_OPENAI_BASE_URL")
        if not base:
            raise ValueError("set CEILINGS_OPENAI_BASE_URL (for example https://host/v1)")
        self.base_url = base.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("CEILINGS_OPENAI_API_KEY")
        self.timeout = timeout or float(os.environ.get("CEILINGS_OPENAI_TIMEOUT", "600") or 600)
        self.retries = retries or int(os.environ.get("CEILINGS_OPENAI_RETRIES", "8") or 8)
        self.retries_429 = int(os.environ.get("CEILINGS_OPENAI_RETRIES_429", "80") or 80)
        self.price_in = float(os.environ.get("CEILINGS_OPENAI_PRICE_IN", "0") or 0)
        self.price_out = float(os.environ.get("CEILINGS_OPENAI_PRICE_OUT", "0") or 0)
        self.profile = load_profile(model_name)
        self.schema_mode = (os.environ.get("CEILINGS_OPENAI_SCHEMA") or self.profile.get("schema")
                            or "json_schema")
        if self.schema_mode not in SCHEMA_MODES:
            raise ValueError(f"CEILINGS_OPENAI_SCHEMA must be one of {SCHEMA_MODES}")
        self.extra_body = {**self.profile.get("extra_body", {}),
                           **json.loads(os.environ.get("CEILINGS_OPENAI_EXTRA_BODY") or "{}")}
        self.answer_max_tokens = int(self.profile.get("answer_max_tokens",
                                                      ChatAdapter.answer_max_tokens))
        self.text_max_tokens = int(self.profile.get("text_max_tokens",
                                                    ChatAdapter.text_max_tokens))
        self.served_aliases = tuple(self.profile.get("served_aliases", ()))
        self.temperature = float(os.environ.get("CEILINGS_TEMPERATURE", "0.7"))
        self.limiter = RateLimiter(float(os.environ.get("CEILINGS_OPENAI_RPM", "0") or 0))
        self.max_requests: int | None = None
        self.requests = 0
        self._count_lock = threading.Lock()

    def served_ok(self, served: str) -> bool:
        return served in self.served_aliases or super().served_ok(served)

    def _count(self) -> None:
        with self._count_lock:
            if self.max_requests is not None and self.requests >= self.max_requests:
                raise RequestCapExceeded(f"request cap of {self.max_requests} reached")
            self.requests += 1

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        structured = "response_format" in body or "nvext" in body
        last: Exception | None = None
        attempt = limited = 0
        # Rate limiting (429) clears once the load drops, so it has its own, longer allowance
        # (CEILINGS_OPENAI_RETRIES_429): 20 straight 429s ended a calibration run on GLM.
        while attempt < self.retries and limited < self.retries_429:
            self._count()
            self.limiter.wait()
            req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code in (400, 422) and structured:
                    raise _SchemaUnsupported() from e
                # 404 is retried too: NVIDIA's gateway briefly returns it for a model that is
                # being redeployed (seen on muse-glimmer-30b, served again minutes later).
                if e.code == 429:
                    last, limited = e, limited + 1
                    with self._count_lock:
                        self.requests -= 1  # a rate-limited reply did no work; not capped
                    time.sleep(_backoff(min(limited, 6), e.headers.get("Retry-After")
                                        if e.headers else None))
                    continue
                if e.code in (404, 408, 409) or e.code >= 500:
                    last, attempt = e, attempt + 1
                    time.sleep(_backoff(attempt, e.headers.get("Retry-After") if e.headers
                                        else None))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last, attempt = e, attempt + 1
                time.sleep(_backoff(attempt, None))
        raise RuntimeError(f"request to {self.base_url} failed after {attempt} errors and "
                           f"{limited} rate-limit replies: {last}")

    def _body(self, messages, schema, max_tokens, seed, mode) -> dict:
        body = {"model": self.model_name, "messages": messages, "max_tokens": max_tokens,
                "seed": int(seed) % (2**31), "temperature": self.temperature, **self.extra_body}
        if schema is not None and mode == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "schema": schema,
                                                       "strict": True}}
        elif schema is not None and mode == "nvext":
            body["nvext"] = {**body.get("nvext", {}), "guided_json": schema}
        return body

    def _chat(self, messages, schema, max_tokens, seed):
        while True:
            try:
                d = self._post(self._body(messages, schema, max_tokens, seed, self.schema_mode))
                break
            except _SchemaUnsupported:
                # Fall back one step at a time: json_schema -> nvext -> plain text.
                self.schema_mode = SCHEMA_MODES[SCHEMA_MODES.index(self.schema_mode) + 1]
        choice = (d.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = d.get("usage") or {}
        content = message.get("content") or ""
        text, hidden = split_think(content)
        field_reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        details = usage.get("completion_tokens_details") or {}
        return ChatReply(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            request_id=d.get("id", ""),
            model_served=d.get("model", ""),
            stop=choice.get("finish_reason") or "",
            hidden_reasoning="\n".join(p for p in (field_reasoning, hidden) if p),
            cost=float(usage["cost"]) if usage.get("cost") is not None else None,
            provider=str(d.get("provider") or ""),
            full_text=content.replace("<think>", "").replace("</think>", "").strip(),
            reasoning_tokens=details.get("reasoning_tokens"),
        )

    def complete(self, messages, depth, answer_space, seed, max_tokens=16000):
        res = super().complete(messages, depth, answer_space, seed, max_tokens)
        if res.extra.get("reported_cost") is None:  # no provider-reported cost: use the prices
            res.usage.dollars = (res.usage.input_tokens * self.price_in
                                 + res.usage.output_tokens * self.price_out) / 1e6
        res.extra["schema_mode"] = self.schema_mode
        return res


def load_profile(model_name: str) -> dict:
    """This model's entry in the `CEILINGS_OPENAI_PROFILES` file (empty if none)."""
    path = os.environ.get("CEILINGS_OPENAI_PROFILES")
    if not path:
        return {}
    with open(path) as f:
        return dict(json.load(f).get("models", {}).get(model_name, {}))


def _backoff(attempt: int, retry_after: str | None) -> float:
    try:
        if retry_after is not None:
            return min(120.0, float(retry_after))
    except ValueError:
        pass
    return min(60.0, 2.0**attempt) * (0.5 + random.random())


class _SchemaUnsupported(Exception):
    pass
