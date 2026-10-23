"""Difficulty search stops once no further setting can change its choice; provider profiles."""

import json

import pytest

from ceilings import calibration
from ceilings.records import CalibrationResult


def _fake_calibrate(full_by_setting, base_ok=lambda d: True):
    tried = []

    def fake(family, insts, policy, concurrency=1):
        d = dict(insts[0].difficulty)
        key = d.get("d", d.get("n", d.get("k")))
        tried.append(key)
        full = full_by_setting[key]
        return CalibrationResult(family_id=family, model_id="m", n=len(insts), baseline_acc=0.5,
                                 chance_acc=0.5, matched_full_acc=full, leak_rate=0.0,
                                 refusal_rate=0.0, passed=full >= 0.9 and base_ok(key))
    return fake, tried


class _P:
    model_id = "m"


def test_comm_search_stops_when_the_full_fix_fails(monkeypatch):
    fake, tried = _fake_calibrate({2: 1.0, 3: 0.95, 4: 0.8, 6: 0.99, 8: 0.99})
    monkeypatch.setattr(calibration, "calibrate", fake)
    res = calibration.search_difficulty("comm_ip", _P(), n=2)
    assert tried == [2, 3, 4] and res["chosen"] == {"n": 3}
    tried.clear()
    full = calibration.search_difficulty("comm_ip", _P(), n=2, stop_early=False)
    assert tried == [2, 3, 4, 6, 8] and full["chosen"] == {"n": 8}


def test_reason_search_stops_at_the_first_pass(monkeypatch):
    fake, tried = _fake_calibrate({d: 1.0 for d in (2, 3, 4, 6, 8, 12, 16)},
                                  base_ok=lambda d: d >= 4)  # easy settings: baseline solves
    monkeypatch.setattr(calibration, "calibrate", fake)
    res = calibration.search_difficulty("reason", _P(), n=2)
    assert tried == [2, 3, 4] and res["chosen"] == {"d": 4}


def test_provider_profile_sets_body_caps_and_alias(monkeypatch, tmp_path):
    from ceilings.adapters.openai_compat import OpenAICompatAdapter

    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"models": {"vendor/m": {
        "extra_body": {"chat_template_kwargs": {"thinking": False}},
        "answer_max_tokens": 1500, "served_aliases": ["private/vendor/m-base"]}}}))
    monkeypatch.setenv("CEILINGS_OPENAI_BASE_URL", "https://h/v1")
    monkeypatch.setenv("CEILINGS_OPENAI_PROFILES", str(path))
    monkeypatch.setenv("CEILINGS_OPENAI_EXTRA_BODY", '{"top_p": 1}')
    a = OpenAICompatAdapter("vendor/m")
    assert a.extra_body == {"chat_template_kwargs": {"thinking": False}, "top_p": 1}
    assert a.answer_max_tokens == 1500 and a.text_max_tokens == 300
    assert a.served_ok("private/vendor/m-base") and a.served_ok("vendor/m")
    assert not a.served_ok("other/model")
    assert OpenAICompatAdapter("vendor/unlisted").profile == {}


def test_shipped_nvidia_profiles_parse():
    from pathlib import Path

    cfg = json.loads((Path(__file__).parents[2] / "configs" / "nvidia_build.json").read_text())
    assert len(cfg["models"]) >= 10
    for name, prof in cfg["models"].items():
        assert set(prof) <= {"extra_body", "answer_max_tokens", "text_max_tokens", "schema",
                             "served_aliases", "notes"}, name


@pytest.fixture(autouse=True)
def _no_env_leak(monkeypatch):
    for name in ("CEILINGS_OPENAI_SCHEMA", "CEILINGS_OPENAI_RPM"):
        monkeypatch.delenv(name, raising=False)


def test_largest_fitting_uses_few_measurements():
    seen = []

    def measure(s):
        seen.append(s)
        return 60.0 * s

    assert calibration.largest_fitting(list(range(1, 17)), measure, 500) == (8, 480.0)
    assert len(seen) <= 6
    assert calibration.largest_fitting([1, 2], lambda s: 999.0, 10)[0] == 1
    assert calibration.largest_fitting([1, 2, 3], lambda s: 1.0, 10)[0] == 3
