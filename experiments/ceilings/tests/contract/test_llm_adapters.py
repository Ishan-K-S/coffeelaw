"""Ollama / OpenAI-compatible adapters, the reasoning stage, and reply parsing. No network."""

import io
import json
import urllib.error

import pytest

from ceilings.adapters import make_adapter, safe_model_dir
from ceilings.adapters.base import REASON_PROMPT, REASONING_TOKENS, parse_answer
from ceilings.adapters.ollama import OllamaAdapter
from ceilings.adapters.openai_compat import OpenAICompatAdapter
from ceilings.agents import clean_bits, parse_probe

SPACE = ["even", "odd"]


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(replies, seen):
    def urlopen(req, timeout=None):
        seen.append((req.full_url, json.loads(req.data), dict(req.headers)))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return _Resp(json.dumps(r).encode())
    return urlopen


def _ollama_reply(content, model="qwen2.5:1.5b", stop="stop", pin=20, pout=5):
    return {"model": model, "message": {"content": content}, "prompt_eval_count": pin,
            "eval_count": pout, "done_reason": stop, "created_at": "t"}


def test_model_ids():
    assert make_adapter("ollama/qwen2.5:1.5b").model_id == "ollama/qwen2.5:1.5b"
    assert safe_model_dir("ollama/qwen2.5:1.5b") == "ollama_qwen2.5_1.5b"
    with pytest.raises(ValueError):
        make_adapter("mystery-model")


def test_ollama_direct_answer_low_depth(monkeypatch):
    seen = []
    good = json.dumps({"answer": "odd", "probabilities": {"even": 0.2, "odd": 0.8}})
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([_ollama_reply(good)], seen))
    res = OllamaAdapter("qwen2.5:1.5b", host="http://x:1").complete(
        [{"role": "user", "content": "q"}], "low", SPACE, 7)
    url, body, _ = seen[0]
    assert url == "http://x:1/api/chat" and len(seen) == 1  # no reasoning stage at low
    assert sorted(body["format"]["properties"]["answer"]["enum"]) == SPACE
    assert body["options"]["temperature"] == 0.7
    assert body["options"]["seed"] == 8 and body["stream"] is False
    assert res.final_answer == "odd" and res.extra["parse"] == "json" and res.valid
    assert (res.usage.input_tokens, res.usage.output_tokens) == (20, 5)


def test_reasoning_stage_budget_and_accounting(monkeypatch):
    seen = []
    good = json.dumps({"answer": "even", "probabilities": {"even": 0.9, "odd": 0.1}})
    replies = [_ollama_reply("thinking...", stop="length", pout=1024), _ollama_reply(good)]
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(replies, seen))
    res = OllamaAdapter("qwen2.5:1.5b", host="http://x:1").complete(
        [{"role": "user", "content": "q"}], "high", SPACE, 0)
    first, second = seen[0][1], seen[1][1]
    assert first["options"]["num_predict"] == REASONING_TOKENS["high"]
    assert "format" not in first and first["messages"][-1]["content"].endswith(REASON_PROMPT)
    assert second["messages"][-2] == {"role": "assistant", "content": "thinking..."}
    assert res.usage.output_tokens == 1029 and res.usage.input_tokens == 40
    assert res.extra["reasoning_stop"] == "length" and res.final_answer == "even"


def test_model_substitution_marks_invalid(monkeypatch):
    good = json.dumps({"answer": "odd", "probabilities": {"even": 0.5, "odd": 0.5}})
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen([_ollama_reply(good, model="llama3.2:3b")], []))
    res = OllamaAdapter("qwen2.5:1.5b", host="http://x:1").complete(
        [{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert not res.valid


def test_openai_compat_schema_auth_price_and_fallback(monkeypatch):
    monkeypatch.setenv("CEILINGS_OPENAI_BASE_URL", "https://h/v1")
    monkeypatch.setenv("CEILINGS_OPENAI_API_KEY", "k")
    monkeypatch.setenv("CEILINGS_OPENAI_PRICE_IN", "1")
    monkeypatch.setenv("CEILINGS_OPENAI_PRICE_OUT", "2")
    seen = []

    def err():
        return urllib.error.HTTPError("https://h/v1/chat/completions", 400, "bad", {}, None)

    ok = _openai_reply("I think odd.")
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([err(), err(), ok], seen))
    adapter = OpenAICompatAdapter("m")
    res = adapter.complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert seen[0][1]["response_format"]["type"] == "json_schema"
    assert "guided_json" in seen[1][1]["nvext"] and "response_format" not in seen[1][1]
    assert "response_format" not in seen[2][1] and "nvext" not in seen[2][1]
    assert adapter.schema_mode == "none" and res.extra["schema_mode"] == "none"
    assert seen[2][2]["Authorization"] == "Bearer k"
    assert res.final_answer == "odd" and res.extra["parse"] == "text"
    assert res.usage.dollars == pytest.approx((1000 * 1 + 500 * 2) / 1e6)


def _openai_reply(content, model="m", reasoning_content=None, reasoning_tokens=None):
    message = {"content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {"id": "r1", "model": model, "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": usage}


@pytest.fixture
def openai_env(monkeypatch):
    monkeypatch.setenv("CEILINGS_OPENAI_BASE_URL", "https://h/v1")
    for name in ("CEILINGS_OPENAI_RPM", "CEILINGS_OPENAI_SCHEMA", "CEILINGS_OPENAI_EXTRA_BODY",
                 "CEILINGS_OPENAI_PRICE_IN", "CEILINGS_OPENAI_PRICE_OUT"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("text,visible,hidden", [
    ("<think>count rows</think>{\"answer\": \"odd\"}", '{"answer": "odd"}', "count rows"),
    ("plan</think>odd", "odd", "plan"),  # server dropped the opening tag
    ("<think>never closed", "", "never closed"),
    ("plain", "plain", ""),
])
def test_split_think(text, visible, hidden):
    from ceilings.adapters.openai_compat import split_think

    assert split_think(text) == (visible, hidden)


def test_openai_compat_strips_hidden_reasoning(monkeypatch, openai_env):
    good = json.dumps({"probabilities": {"even": 0.1, "odd": 0.9}, "answer": "odd"})
    reply = _openai_reply(f"<think>bits sum to 3</think>{good}", reasoning_content="field",
                          reasoning_tokens=40)
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([reply], []))
    res = OpenAICompatAdapter("m").complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert res.final_answer == "odd" and res.extra["parse"] == "json"
    assert res.extra["hidden_reasoning"] == ["field\nbits sum to 3"]
    assert res.usage.reasoning_tokens == 40 and res.usage.output_tokens == 500


def test_openai_compat_nvext_mode_and_extra_body(monkeypatch, openai_env):
    monkeypatch.setenv("CEILINGS_OPENAI_SCHEMA", "nvext")
    monkeypatch.setenv("CEILINGS_OPENAI_EXTRA_BODY",
                       '{"chat_template_kwargs": {"thinking": false}}')
    seen = []
    good = json.dumps({"probabilities": {"even": 0.6, "odd": 0.4}, "answer": "even"})
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([_openai_reply(good)], seen))
    OpenAICompatAdapter("m").complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    body = seen[0][1]
    assert "response_format" not in body and body["nvext"]["guided_json"]["type"] == "object"
    assert body["chat_template_kwargs"] == {"thinking": False}


def test_openai_compat_retry_after_and_request_cap(monkeypatch, openai_env):
    from ceilings.adapters.base import RequestCapExceeded

    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    limited = urllib.error.HTTPError("https://h/v1/chat/completions", 429, "slow down",
                                     {"Retry-After": "7"}, None)
    gone = urllib.error.HTTPError("https://h/v1/chat/completions", 404, "not found", {}, None)
    good = json.dumps({"probabilities": {"even": 0.6, "odd": 0.4}, "answer": "even"})
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen([limited, gone, _openai_reply(good)], []))
    adapter = OpenAICompatAdapter("m")
    adapter.max_requests = 2
    res = adapter.complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    # the 429 is not counted against the cap; the 404 and the success are
    assert res.final_answer == "even" and sleeps[0] == 7.0 and adapter.requests == 2
    with pytest.raises(RequestCapExceeded):
        adapter.complete([{"role": "user", "content": "q"}], "low", SPACE, 0)


def test_rate_limiter_spaces_requests(monkeypatch):
    from ceilings.adapters.openai_compat import RateLimiter

    clock, sleeps = [100.0], []
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    limiter = RateLimiter(rpm=30)  # one start every 2 s
    for _ in range(3):
        limiter.wait()
    assert sleeps == [pytest.approx(2.0), pytest.approx(4.0)]
    assert RateLimiter(rpm=0).interval == 0


def test_max_requests_flag_reaches_adapter(monkeypatch, openai_env):
    from ceilings.cli import _policy, build_parser

    a = build_parser().parse_args(["run", "--model", "openai/m", "--max-requests", "5"])
    assert _policy(a).adapter.max_requests == 5


def test_openai_compat_needs_base_url(monkeypatch):
    monkeypatch.delenv("CEILINGS_OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatAdapter("m")


@pytest.mark.parametrize("text,expected,how", [
    ('{"answer": "B", "probabilities": {"A": 0.1, "B": 0.9}}', "B", "json"),
    ("The answer is host-03.", "host-03", "text"),
    ("I am not sure.", None, "failed"),
])
def test_parse_answer(text, expected, how):
    space = ["A", "B"] if expected in ("A", "B") else ["host-03", "host-13"]
    answer, probs, parsed = parse_answer(text, space)
    assert answer == expected and parsed == how and abs(sum(probs.values()) - 1) < 1e-9


@pytest.mark.parametrize("text,expected,how", [
    # A server that ignored the schema and discussed every option (seen on NVIDIA build).
    ("Option 1: host-1, probability 0\nOption 3: host-3, probability 1", None, "failed"),
    ("host-1 is ruled out. Final answer: **host-3**", "host-3", "text"),
    ('Sure.\n```json\n{"probabilities": {"host-1": 0.1, "host-3": 0.9}, "answer": "host-3"}\n```',
     "host-3", "json_embedded"),
])
def test_parse_answer_ignored_schema(text, expected, how):
    answer, _, parsed = parse_answer(text, ["host-1", "host-3"])
    assert (answer, parsed) == (expected, how)


def test_json_format_note_follows_schema_order():
    from ceilings.adapters.base import answer_schema, json_format_note

    note = json_format_note(answer_schema(["even", "odd"], None, answer_first=True))
    assert note.index('"answer"') < note.index('"probabilities"') and '"even": <number>' in note


def test_parse_answer_does_not_confuse_prefixes():
    assert parse_answer("host-13", ["host-1", "host-13"])[0] == "host-13"


def test_parse_probe_and_clean_bits():
    attrs = [["rack", "A", "B"], ["row", "front", "back"], ["power feed", "1", "2"]]
    assert parse_probe("Rack", attrs) == 0
    assert parse_probe("I'd probe the power feed.", attrs) == 2
    assert parse_probe("1", attrs) == 1
    assert parse_probe("colour", attrs) == "colour"
    # prose naming several attributes, or a reply cut off mid-sentence, is not a choice
    assert parse_probe("The rack, row and power feed results identify host-03.", attrs) != 0
    assert parse_probe("Row.", attrs) == 1 and parse_probe("**power feed**", attrs) == 2
    assert clean_bits("```\n0 1 1\n```") == "011"
    assert clean_bits("yes") == "yes"


def test_reasoning_budget_override(monkeypatch):
    from ceilings.adapters.base import reasoning_tokens

    assert reasoning_tokens("high") == REASONING_TOKENS["high"]
    monkeypatch.setenv("CEILINGS_REASONING_TOKENS", "0,32,64,128,256")
    assert reasoning_tokens("low") == 0 and reasoning_tokens("high") == 64


def test_ollama_duty_cycle_threads_and_keep_alive(monkeypatch):
    monkeypatch.setenv("CEILINGS_OLLAMA_DUTY", "0.25")
    monkeypatch.setenv("CEILINGS_OLLAMA_NUM_THREAD", "2")
    monkeypatch.setenv("CEILINGS_OLLAMA_KEEP_ALIVE", "30s")
    seen, sleeps = [], []
    clock = iter([100.0, 102.0])  # the call takes 2 s
    monkeypatch.setattr("time.monotonic", lambda: next(clock))
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    good = json.dumps({"answer": "odd", "probabilities": {"even": 0.4, "odd": 0.6}})
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([_ollama_reply(good)], seen))
    OllamaAdapter("qwen2.5:1.5b", host="http://x:1").complete(
        [{"role": "user", "content": "q"}], "low", SPACE, 0)
    body = seen[0][1]
    assert body["options"]["num_thread"] == 2 and body["keep_alive"] == "30s"
    assert sleeps == [pytest.approx(6.0)]  # busy 25% of the time: sleep 3x the call


def test_ollama_rejects_bad_duty(monkeypatch):
    monkeypatch.setenv("CEILINGS_OLLAMA_DUTY", "0")
    with pytest.raises(ValueError):
        OllamaAdapter("m")


def test_truncated_reasoning_is_flagged_to_the_model(monkeypatch):
    seen = []
    good = json.dumps({"probabilities": {"even": 0.9, "odd": 0.1}, "answer": "even"})
    replies = [_ollama_reply("partial", stop="length"), _ollama_reply(good)]
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(replies, seen))
    OllamaAdapter("qwen2.5:1.5b", host="http://x:1").complete(
        [{"role": "user", "content": "q"}], "medium", SPACE, 0)
    assert seen[1][1]["messages"][-1]["content"].startswith("(Your reasoning was cut off")


def test_probability_diagnostics_and_argmax_fallback():
    from ceilings.adapters.base import probability_diagnostics

    text = json.dumps({"probabilities": {"even": 0.9, "odd": 0.9}, "answer": "odd"})
    answer, probs, how = parse_answer(text, SPACE)
    d = probability_diagnostics(text, answer, probs, SPACE)
    assert d["raw_prob_sum"] == pytest.approx(1.8) and not d["answer_not_argmax"]
    text = json.dumps({"probabilities": {"even": 0.8, "odd": 0.2}, "answer": "odd"})
    answer, probs, _ = parse_answer(text, SPACE)
    assert probability_diagnostics(text, answer, probs, SPACE)["answer_not_argmax"]
    answer, _, how = parse_answer(json.dumps({"probabilities": {"even": 0.3, "odd": 0.7}}), SPACE)
    assert (answer, how) == ("odd", "json_argmax")


def test_rate_limit_replies_have_their_own_allowance(monkeypatch, openai_env):
    monkeypatch.setenv("CEILINGS_OPENAI_RETRIES", "2")
    monkeypatch.setenv("CEILINGS_OPENAI_RETRIES_429", "5")
    monkeypatch.setattr("time.sleep", lambda s: None)

    def limited():
        return urllib.error.HTTPError("https://h/v1/chat/completions", 429, "slow", {}, None)

    good = json.dumps({"probabilities": {"even": 0.6, "odd": 0.4}, "answer": "even"})
    replies = [limited() for _ in range(4)] + [_openai_reply(good)]
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(replies, []))
    res = OpenAICompatAdapter("m").complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert res.final_answer == "even"  # 4 rate-limit replies, more than RETRIES=2
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen([limited() for _ in range(5)], []))
    with pytest.raises(RuntimeError, match="5 rate-limit replies"):
        OpenAICompatAdapter("m").complete([{"role": "user", "content": "q"}], "low", SPACE, 0)


def test_reasoning_stage_keeps_text_before_a_closing_think_tag(monkeypatch, openai_env):
    seen = []
    stage = _openai_reply("Shift of Gus is D, so Dina is D.</think>A: 0.0\nD: 1.0")
    good = json.dumps({"probabilities": {"even": 0.1, "odd": 0.9}, "answer": "odd"})
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen([stage, _openai_reply(good)], seen))
    res = OpenAICompatAdapter("m").complete([{"role": "user", "content": "q"}], "medium",
                                            SPACE, 0)
    passed_on = seen[1][1]["messages"][-2]
    assert passed_on["role"] == "assistant" and passed_on["content"].startswith("Shift of Gus")
    assert "D: 1.0" in passed_on["content"] and res.extra["reasoning"].startswith("Shift")


def test_truncated_reply_and_top_ties_are_parse_failures(monkeypatch, openai_env):
    cut = {"id": "r", "model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 400},
           "choices": [{"message": {"content": '{"probabilities": {"odd": 0.0000000000000'},
                        "finish_reason": "length"}]}
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([cut], []))
    res = OpenAICompatAdapter("m").complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert res.final_answer is None and res.extra["parse"] == "failed"
    tie = json.dumps({"probabilities": {"even": 0.5, "odd": 0.5}})
    assert parse_answer(tie, SPACE)[0] is None and parse_answer(tie, SPACE)[2] == "failed"


def test_timeout_comes_from_the_environment(monkeypatch, openai_env):
    monkeypatch.setenv("CEILINGS_OPENAI_TIMEOUT", "120")
    assert OpenAICompatAdapter("m").timeout == 120.0
    monkeypatch.delenv("CEILINGS_OPENAI_TIMEOUT")
    assert OpenAICompatAdapter("m").timeout == 600.0


def test_reported_cost_and_provider_are_used(monkeypatch, openai_env):
    monkeypatch.setenv("CEILINGS_OPENAI_PRICE_IN", "100")  # ignored when the provider reports cost
    good = json.dumps({"probabilities": {"even": 0.6, "odd": 0.4}, "answer": "even"})
    reply = _openai_reply(good)
    reply["usage"]["cost"] = 0.00042
    reply["provider"] = "DeepInfra"
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([reply], []))
    res = OpenAICompatAdapter("m").complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert res.usage.dollars == pytest.approx(0.00042)
    assert res.extra["served_by"] == ["DeepInfra"] and res.extra["reported_cost"] == 0.00042


def test_pinned_schema_retries_a_400_and_never_falls_back(monkeypatch, openai_env):
    monkeypatch.setenv("CEILINGS_OPENAI_SCHEMA", "json_schema")
    monkeypatch.setattr("time.sleep", lambda s: None)
    bad = urllib.error.HTTPError("https://h/v1/chat/completions", 400, "bad", {},
                                 io.BytesIO(b'{"error": "provider said no"}'))
    good = json.dumps({"probabilities": {"even": 0.6, "odd": 0.4}, "answer": "even"})
    seen = []
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen([bad, _openai_reply(good)], seen))
    adapter = OpenAICompatAdapter("m")
    res = adapter.complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert res.final_answer == "even" and adapter.schema_mode == "json_schema"
    assert all(s[1]["response_format"]["type"] == "json_schema" for s in seen)
    # a 400 that keeps coming ends the call with the provider's reason
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(
        urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"provider said no"))))
    with pytest.raises(RuntimeError, match="provider said no"):
        adapter.complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert adapter.schema_mode == "json_schema"


def test_concurrent_schema_fallbacks_move_one_step(monkeypatch, openai_env):
    from ceilings.adapters import openai_compat

    adapter = OpenAICompatAdapter("m")
    good = _openai_reply(json.dumps({"probabilities": {"even": 0.6, "odd": 0.4},
                                     "answer": "even"}))
    # Three calls all fail on json_schema before any of them handles the failure (the race
    # that crashed Claude Haiku's prepare with an IndexError).
    failed_on = []

    def post(body):
        if "response_format" in body and len(failed_on) < 3:
            failed_on.append(1)
            raise openai_compat._SchemaUnsupported()
        return good

    monkeypatch.setattr(adapter, "_post", post)
    for _ in range(3):
        adapter._chat([{"role": "user", "content": "q"}], {"type": "object"}, 10, 0)
    assert adapter.schema_mode == "nvext"


def test_a_400_reporting_a_rate_limit_is_waited_out(monkeypatch, openai_env):
    monkeypatch.setenv("CEILINGS_OPENAI_SCHEMA", "json_schema")
    monkeypatch.setattr("time.sleep", lambda s: None)
    raw = (b'{"error":{"code":400,"metadata":{"raw":"Grammar compilation rate limit exceeded. '
           b'50 compilations per minute."}}}')
    replies = [urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(raw)) for _ in range(15)]
    good = json.dumps({"probabilities": {"even": 0.6, "odd": 0.4}, "answer": "even"})
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen(replies + [_openai_reply(good)], []))
    adapter = OpenAICompatAdapter("m")  # retries = 8 < 15 rate-limited replies
    res = adapter.complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert res.final_answer == "even" and adapter.requests == 1  # limited replies not capped
