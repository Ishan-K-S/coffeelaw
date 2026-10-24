"""Difficulty search stops once no further setting can change its choice; provider profiles."""

import json

import pytest

from ceilings import calibration
from ceilings.records import CalibrationResult


def _fake_calibrate(full_by_setting, base_ok=lambda d: True):
    tried = []

    def fake(family, insts, policy, concurrency=1, log_path=None):
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
    from ceilings.options import DEFAULT as options


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


def test_search_resumes_from_its_cache(monkeypatch, tmp_path):
    fake, tried = _fake_calibrate({2: 1.0, 3: 0.95, 4: 0.8, 6: 0.99, 8: 0.99})
    monkeypatch.setattr(calibration, "calibrate", fake)
    cache = tmp_path / "settings.json"
    first = calibration.search_difficulty("comm_ip", _P(), n=2, cache_path=cache)
    tried.clear()
    again = calibration.search_difficulty("comm_ip", _P(), n=2, cache_path=cache)
    assert tried == [] and again["chosen"] == first["chosen"] == {"n": 3}
    calibration.search_difficulty("comm_ip", _P(), n=3, cache_path=cache)
    assert tried == [2, 3, 4]  # a different n is a different search


def test_calibrate_logs_every_gate_run(tmp_path):
    from ceilings import tasks
    from ceilings.agents import make_policy

    insts = tasks.generate_many("comm_parity", 3, "calibration", 0, n=2)
    log = tmp_path / "gate.jsonl"
    cal = calibration.calibrate("comm_parity", insts, make_policy("sim-bounded"), log_path=log)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 6 and {r["config"]["arm_id"] for r in rows} == {"baseline", "M"}
    assert sum(r["correct"] for r in rows if r["config"]["arm_id"] == "M") / 3 == \
        cal.matched_full_acc
    assert all(r["steps"] for r in rows)


def test_prepare_resumes_gate_and_matching(tmp_path, monkeypatch):
    from ceilings import experiment
    from ceilings.agents import make_policy
    from ceilings.analysis.report import load_gates

    policy = make_policy("sim-bounded")
    kw = dict(n_main=4, n_pilot=4, difficulty={"n": 2}, levels={"medium": 1.0}, n_match=3)
    first = experiment.prepare("comm_parity", policy, tmp_path, **kw)
    assert (tmp_path / "matching" / "comm_parity__sim-bounded.json").exists()
    assert not list((tmp_path / "arms").glob("*/*.measurements.json"))
    calls = []
    monkeypatch.setattr(calibration, "_runs", lambda *a, **k: calls.append(1) or [])
    again = experiment.prepare("comm_parity", policy, tmp_path, **kw)
    assert calls == [] and again["table"] == first["table"]
    assert load_gates(tmp_path)[("sim-bounded", "comm_parity")] == first["calibration"].passed


def test_failed_search_counts_as_failed_gate(tmp_path):
    from ceilings.analysis.report import load_gates

    (tmp_path / "difficulty").mkdir()
    (tmp_path / "difficulty" / "info__m.json").write_text(json.dumps(
        {"family": "info", "model_id": "m", "chosen": None, "results": []}))
    (tmp_path / "difficulty" / "info__m.settings.json").write_text("{}")
    assert load_gates(tmp_path) == {("m", "info"): False}


def test_prepare_skips_matching_when_the_gate_fails(tmp_path, monkeypatch):
    from ceilings import experiment
    from ceilings.agents import make_policy

    failing = CalibrationResult(family_id="reason", model_id="sim-bounded", n=4, baseline_acc=0.9,
                                chance_acc=0.25, matched_full_acc=1.0, leak_rate=0.0,
                                refusal_rate=0.0, passed=False, baseline_mean_tokens=100.0)
    monkeypatch.setattr(calibration, "calibrate", lambda *a, **k: failing)
    monkeypatch.setattr(calibration, "match_arms",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("matched")))
    res = experiment.prepare("reason", make_policy("sim-bounded"), tmp_path, n_main=2, n_pilot=2,
                             difficulty={"d": 2}, levels={"medium": 1.0})
    assert res["table"]["gate_failed"] and not res["calibration"].passed
