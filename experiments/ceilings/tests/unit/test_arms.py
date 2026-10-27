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


def test_a2_single_ref_is_one_agent_reasoning_at_its_matched_depth():
    from ceilings.adapters.base import reasoning_tokens

    ref = arms.build("A2_single_ref", "reason", "xhigh")
    assert ref.n_agents == 1 and ref.samples == 1 and ref.depth == "xhigh"
    # candidates are reasoning depths in cost order, matched on their own (not A2's strength)
    assert arms.candidates("A2_single_ref", "reason") == ["low", "medium", "high", "xhigh", "max"]
    assert arms.candidates("A2_single_ref", "comm_ip") == ["medium", "high", "xhigh", "max"]
    # the unmatched default pools four medium agents' budgets
    assert reasoning_tokens(arms.pooled_depth("medium", 4)) >= 4 * reasoning_tokens("medium")


def test_k_can_reason_on_some_samples_and_comm_reasons_by_design():
    k = arms.build("K", "reason", "medium:2")
    assert (k.depth, k.reason_samples, k.samples) == ("medium", 2, 4)
    assert arms.build("K", "reason", "high").reason_samples is None
    assert arms.candidates("K", "reason")[0] == "medium:1"
    assert arms.baseline("comm_parity").depth == "medium" and arms.baseline("info").depth == "low"
    assert all(not c.startswith("medium") for c in arms.candidates("K", "comm_ip"))


def test_irrelevant_text_has_no_answer_tokens():
    text = arms.extra_text("irrelevant", 0, 200, "")
    for answer in ["A", "B", "C", "D", "host-", "even", "odd"]:
        assert f" {answer} " not in f" {text} "
