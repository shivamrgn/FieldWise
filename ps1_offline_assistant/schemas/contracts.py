"""
Strict Inter-Pipeline JSON Contracts for FieldWise PS1 Offline Edge-AI Assistant.

Defines Pydantic v2 schemas for:
  - Pipeline A Output: VisionFinding
  - Pipeline B Input:  AdvisoryRequest (including grounded DiseasePestCard)
  - Pipeline B Output: AdvisoryResponse

Enforces strict type safety (rejects booleans/strings in numeric fields,
rejects NaN/Inf, rejects unknown extra fields, enforces bounded ranges)
and safety invariants (e.g., no chemical guidance on uncertain/low-confidence
detections; mandatory KVK escalation on SEVERE_ESCALATION).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VisionCategory(str, Enum):
    DISEASE = "disease"
    PEST = "pest"
    HEALTHY = "healthy"
    UNKNOWN = "unknown"


class SeverityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNDETERMINED = "undetermined"


class SupportedLanguage(str, Enum):
    EN = "en"
    HI = "hi"
    GU = "gu"
    MR = "mr"
    TE = "te"
    TA = "ta"


class AdvisoryStatus(str, Enum):
    SUCCESS = "SUCCESS"
    UNCERTAIN_IMAGE = "UNCERTAIN_IMAGE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    SEVERE_ESCALATION = "SEVERE_ESCALATION"


class ConditionType(str, Enum):
    DISEASE = "disease"
    PEST = "pest"


# ---------------------------------------------------------------------------
# Shared Strict Validation Helpers
# ---------------------------------------------------------------------------


def _validate_non_empty_str(value: Any) -> str:
    """Requires a genuine string that is non-empty after stripping whitespace."""
    if not isinstance(value, str):
        raise ValueError(f"Expected str, got {type(value).__name__}.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("String value must not be empty or whitespace-only.")
    return cleaned


def _validate_strict_bool(value: Any) -> bool:
    """Requires a genuine bool (rejects ints 0/1 and strings)."""
    if not isinstance(value, bool):
        raise ValueError(f"Expected strict bool, got {type(value).__name__}.")
    return value


def _validate_finite_float(value: Any) -> float:
    """Requires a finite int/float (rejects bool, str, NaN, Inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Expected finite numeric value (int/float), got {type(value).__name__}."
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"Numeric value must be finite (NaN/Inf rejected), got {value!r}.")
    return fval


def _validate_str_list(value: Any) -> List[str]:
    """Requires a list of non-empty strings (rejects bare strings or dicts)."""
    if not isinstance(value, list):
        raise ValueError(f"Expected list of strings, got {type(value).__name__}.")
    return [_validate_non_empty_str(item) for item in value]


StrictNonEmptyStr = Annotated[str, BeforeValidator(_validate_non_empty_str)]
StrictBool = Annotated[bool, BeforeValidator(_validate_strict_bool)]
StrictFiniteFloat = Annotated[float, BeforeValidator(_validate_finite_float)]
StrictStrList = Annotated[List[str], BeforeValidator(_validate_str_list)]


class _StrictContractModel(BaseModel):
    """Base contract model enforcing strict schema boundaries."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Grounded Knowledge Base Card Models
# ---------------------------------------------------------------------------


class SeverityThresholds(_StrictContractModel):
    """Agronomic severity brackets for a crop disease or pest."""

    low: StrictNonEmptyStr
    moderate: StrictNonEmptyStr
    severe: StrictNonEmptyStr


class ApprovedChemicalOption(_StrictContractModel):
    """CIB&RC-aligned chemical option paired with a mandatory safety warning."""

    active_ingredient: StrictNonEmptyStr
    target_stage: StrictNonEmptyStr
    cibrc_aligned_note: StrictNonEmptyStr
    safety_warning: StrictNonEmptyStr


class DiseasePestCard(_StrictContractModel):
    """Grounded ICAR/KVK-aligned treatment card stored in agri_kb.json."""

    id: StrictNonEmptyStr
    crop: StrictNonEmptyStr
    condition: StrictNonEmptyStr
    type: ConditionType
    aliases: StrictStrList = Field(default_factory=list)
    symptoms: StrictStrList
    severity_thresholds: SeverityThresholds
    cultural_organic_control: StrictStrList
    approved_chemical_options: List[ApprovedChemicalOption]
    region_notes: Dict[StrictNonEmptyStr, StrictNonEmptyStr]
    kvk_escalation_trigger: StrictNonEmptyStr
    source_reference: StrictNonEmptyStr

    @field_validator("symptoms", "cultural_organic_control")
    @classmethod
    def _require_non_empty_list(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Treatment card list field must contain at least one entry.")
        return value

    @field_validator("region_notes")
    @classmethod
    def _require_region_notes(cls, value: Dict[str, str]) -> Dict[str, str]:
        if not value:
            raise ValueError("region_notes must contain at least one regional note.")
        return value


# ---------------------------------------------------------------------------
# Inter-Pipeline Contracts (Pipeline A <-> Pipeline B)
# ---------------------------------------------------------------------------


class VisionFinding(_StrictContractModel):
    """Strict output schema produced by Pipeline A (On-Device Vision)."""

    crop: StrictNonEmptyStr
    condition_name: StrictNonEmptyStr
    category: VisionCategory
    confidence: StrictFiniteFloat = Field(..., ge=0.0, le=1.0)
    severity_level: SeverityLevel
    infected_area_pct: StrictFiniteFloat = Field(..., ge=0.0, le=100.0)
    image_quality_ok: StrictBool
    blur_score: StrictFiniteFloat = Field(..., ge=0.0)
    inference_latency_ms: StrictFiniteFloat = Field(..., ge=0.0)
    runtime_backend: StrictNonEmptyStr


class AdvisoryRequest(_StrictContractModel):
    """Strict input schema passed into Pipeline B (On-Device Llama Advisory)."""

    vision_finding: VisionFinding
    state_region: StrictNonEmptyStr
    language: SupportedLanguage
    kb_entry: Optional[DiseasePestCard] = None


class AdvisoryResponse(_StrictContractModel):
    """Strict output schema produced by Pipeline B (Multilingual Advisory)."""

    status: AdvisoryStatus
    explanation: StrictNonEmptyStr
    immediate_actions: StrictStrList = Field(default_factory=list)
    organic_cultural_steps: StrictStrList = Field(default_factory=list)
    chemical_guidance_if_approved: StrictStrList = Field(default_factory=list)
    preventive_measures: StrictStrList = Field(default_factory=list)
    escalate_to_kvk: StrictBool
    disclaimer: StrictNonEmptyStr
    language: SupportedLanguage
    ttft_ms: StrictFiniteFloat = Field(..., ge=0.0)
    tokens_per_sec: StrictFiniteFloat = Field(..., ge=0.0)

    @model_validator(mode="after")
    def _enforce_safety_invariants(self) -> "AdvisoryResponse":
        if self.status == AdvisoryStatus.SEVERE_ESCALATION and not self.escalate_to_kvk:
            raise ValueError(
                "SEVERE_ESCALATION status strictly requires escalate_to_kvk=True."
            )
        if (
            self.status in (AdvisoryStatus.UNCERTAIN_IMAGE, AdvisoryStatus.LOW_CONFIDENCE)
            and self.chemical_guidance_if_approved
        ):
            raise ValueError(
                f"Chemical guidance must be withheld (empty) when status is {self.status.value}."
            )
        return self
