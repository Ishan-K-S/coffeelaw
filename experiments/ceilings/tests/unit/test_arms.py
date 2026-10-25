"""Each arm differs from baseline in exactly one resource (FR-010)."""

from dataclasses import asdict

import pytest

from ceilings import arms

RESOURCE_FIELDS = {
    "E": {"tool_rounds", "extra_kind", "extra_amount"},
    "K": {"depth"},
    "M": {"message_budget", "relay"},
    "A1": {"n_agents", "samples", "answerers"},
    "P1": {"extra_kind"},
    "P2": {"extra_kind"},
}
IGNORED = {"arm_id", "strength", "filler_tokens"}


@pytest.mark.parametrize(
    "family,strengths",
    [
        ("info", {"E": 3, "K": "high", "M": 100}),
        ("reason", {"E": 4, "K": "high", "M": 100}),
        ("comm_ip", {"E": 4, "K": "high", "M": 8}),
    ],
)
def test_single_resource_changes(family, strengths):
    base = asdict(arms.baseline(family))
    for arm_id, allowed in RESOURCE_FIELDS.items():
        cfg = asdict(arms.build(arm_id, family, strengths.get(arm_id)))
        changed = {k for k in base if base[k] != cfg[k]} - IGNORED
        assert changed and changed <= allowed, (arm_id, changed)


def test_a1_keeps_total_answer_calls():
    base, a1 = arms.baseline("reason"), arms.build("A1", "reason")
    assert base.samples * len(base.answerers) == a1.samples * len(a1.answerers)
    assert a1.message_budget == base.message_budget == 0


def test_a2_single_ref_has_the_pooled_budget():
    from ceilings.adapters.base import reasoning_tokens

    for depth in ("low", "medium", "high"):
        a2 = arms.build("A2", "reason", depth)
        ref = arms.build("A2_single_ref", "reason", depth)
        assert ref.n_agents == 1
        assert reasoning_tokens(ref.depth) >= a2.n_agents * reasoning_tokens(a2.depth)
    assert arms.build("A2_single_ref", "reason", "low").depth == "low"  # 4 x 0 tokens
    assert arms.build("A2_single_ref", "reason", "medium").depth == "high"  # 4 x 256 <= 1024


def test_irrelevant_text_has_no_answer_tokens():
    text = arms.extra_text("irrelevant", 0, 200, "")
    for answer in ["A", "B", "C", "D", "host-", "even", "odd"]:
        assert f" {answer} " not in f" {text} "
