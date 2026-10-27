"""Record types (data-model.md) and content-hash IDs.

RunRecord mirrors contracts/run-record.schema.json; tests/contract/test_schema.py keeps them in step.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ArmId = Literal["baseline", "E", "K", "M", "A1", "A2", "A2_single_ref", "P1", "P2", "E+K", "M+K"]
DeltaLevel = Literal["small", "medium", "large"]
Split = Literal["calibration", "main", "heldout_blind"]


def content_hash(obj: Any) -> str:
    """SHA-256 of canonical (sorted-key, compact) JSON."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def now() -> datetime:
    return datetime.now(timezone.utc)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolSpec(_Frozen):
    kind: Literal["bit_probe", "oracle"]
    answer_alphabet: list[str]
    flip_rate: float = Field(ge=0, le=0.5)
    capacity_bits: float = Field(ge=0)
    charge_tokens: int = Field(ge=0)


class TheoryQuantities(_Frozen):
    H_theta: float
    I_held_by_agent: dict[str, float]
    I_held_team: float
    budget_by_T: list[float] = []
    fano_floor_by_T: list[float] = []
    comm_bound_by_L: list[float] = []


class TaskInstance(_Frozen):
    instance_id: str
    family_id: str
    family_version: str
    seed: int
    difficulty: dict[str, float | int | str]
    answer: str
    answer_space: list[str]
    scenario: str
    shared_evidence: str
    private_evidence: dict[str, str]
    tool: ToolSpec | None = None
    theory: TheoryQuantities
    split: Split
    excluded: bool = False
    exclusion_reason: str | None = None
    payload: dict[str, Any] = {}

    @model_validator(mode="after")
    def _answer_in_space(self) -> TaskInstance:
        if self.answer not in self.answer_space:
            raise ValueError("answer must be one of answer_space")
        return self


class TaskFamily(_Frozen):
    family_id: str
    version: str
    binding: list[Literal["information", "reasoning", "communication"]] | None
    scenario: str
    formal_mapping: str
    difficulty: dict[str, float | int | str]
    state: Literal["draft", "calibrating", "ready", "rejected"] = "draft"


class CalibrationResult(_Frozen):
    family_id: str
    model_id: str
    n: int
    baseline_acc: float
    chance_acc: float
    matched_full_acc: float
    leak_rate: float
    refusal_rate: float
    passed: bool
    baseline_mean_tokens: float | None = None


class Arm(_Frozen):
    arm_id: ArmId
    delta_level: DeltaLevel | None
    target_delta_tokens: int | None
    config: dict[str, Any]
    matched: bool | None = None


class AgentUsage(_Frozen):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    calls: int = Field(ge=0)


class ToolCalls(_Frozen):
    count: int = Field(ge=0)
    charged_tokens: int = Field(ge=0)


class Messages(_Frozen):
    count: int = Field(ge=0)
    tokens: int = Field(ge=0)
    bits: int | None = Field(default=None, ge=0)


class ResourceLedger(_Frozen):
    per_agent: dict[str, AgentUsage]
    tool_calls: ToolCalls
    messages: Messages
    total_tokens: int = Field(gt=0)
    added_vs_baseline: float | None = None
    dollars: float | None = Field(default=None, ge=0)


class Refusal(_Frozen):
    category: str | None
    explanation: str | None


class Sweep(_Frozen):
    T: int | None = Field(default=None, ge=0)
    flip_rate: float | None = Field(default=None, ge=0, le=0.5)
    L_bits: int | None = Field(default=None, ge=0)


class RunRecord(_Frozen):
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    instance_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    family_id: str
    arm_id: ArmId
    delta_level: DeltaLevel | None = None
    model_id: str
    replicate: int = Field(ge=0)
    sweep: Sweep | None = None
    harness: dict | None = None  # harness options in effect (ceilings.options), resolved
    prereg_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_answer: str | None
    probabilities: dict[str, float]
    sample_answers: list[str | None] = []
    correct: bool
    p_true: float | None = None  # the team's probability on the true answer
    valid: bool = True
    refusal: Refusal | None
    ledger: ResourceLedger
    transcript_ref: str
    provider_request_ids: list[str]
    started_at: datetime
    finished_at: datetime

    @field_validator("probabilities")
    @classmethod
    def _sums_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        if v and not math.isclose(sum(v.values()), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to 1 +/- 1e-6")
        if any(p < 0 or p > 1 for p in v.values()):
            raise ValueError("probabilities must lie in [0, 1]")
        return v


class Prediction(_Frozen):
    prediction_id: str
    statement: str
    paper_source: str
    test: str
    verdict: Literal["supported", "contradicted", "inconclusive", "not evaluable"]
    estimate: float | None = None
    ci: tuple[float, float] | None = None
    p_adj: float | None = None
    detail: dict[str, Any] = {}


class DiagnosticCall(_Frozen):
    masked_task_id: str
    analyst_label: Literal["information", "reasoning", "communication", "none_single"]
    evidence_seen: str
    truth_label: str | None = None
    annotator_labels: list[str] = []
