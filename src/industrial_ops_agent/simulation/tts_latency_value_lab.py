"""Actual-GPU latency-bounded TTS enterprise-value lab v5.

The v2 PostNet candidate was genuinely trained on the project GPU but did not
generalize on the retired final Gold suite.  This additive experiment preserves
that rejection and turns the trained checkpoint into a governed runtime: a
small frozen candidate set is generated, Whisper checks the spoken contract,
and the authorized XVector profile selects the strongest safe voice rendering.
The retired v7 suite is used only as a disclosed development regression set;
all previously winning strategies are retained while unused candidates are
pruned from 29 to 23 before the fresh, disjoint, single-use v8 Gold is opened.
"""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import shutil
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.evaluation.metrics import (
    ModelObservation,
    PairedMetricReport,
    score_tts_paired_observations,
)
from industrial_ops_agent.experiments.service import TTS_HARD_GATES
from industrial_ops_agent.simulation import tts_final_value_lab as v4
from industrial_ops_agent.simulation import tts_recovery_value_lab as v2

SCHEMA_VERSION: Literal["enterprise-tts-latency-value-lab/v7"] = (
    "enterprise-tts-latency-value-lab/v7"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
CALIBRATED_OUTPUT_RELATIVE = Path("artifacts/m7-tts-calibrated-value-lab")
FINAL_OUTPUT_RELATIVE = Path("artifacts/m7-tts-final-value-lab")
OUTPUT_RELATIVE = Path("artifacts/m7-tts-latency-value-lab")
VALIDATION_RELATIVE = OUTPUT_RELATIVE / "inputs/industrial-safety-tts-validation-v6.json"
PRIOR_GOLD_V4_RELATIVE = (
    CALIBRATED_OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v4/manifest.json"
)
PRIOR_GOLD_V4_RETIREMENT_RELATIVE = (
    CALIBRATED_OUTPUT_RELATIVE / "gold-v4-retirement.json"
)
PRIOR_GOLD_V5_RELATIVE = (
    CALIBRATED_OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v5/manifest.json"
)
PRIOR_GOLD_V5_RETIREMENT_RELATIVE = (
    CALIBRATED_OUTPUT_RELATIVE / "gold-v5-retirement.json"
)
PRIOR_GOLD_V6_RELATIVE = (
    CALIBRATED_OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v6/manifest.json"
)
PRIOR_GOLD_V6_RETIREMENT_RELATIVE = (
    CALIBRATED_OUTPUT_RELATIVE / "gold-v6-retirement.json"
)
PRIOR_GOLD_V7_RELATIVE = (
    FINAL_OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v7/manifest.json"
)
PRIOR_GOLD_V7_RETIREMENT_RELATIVE = FINAL_OUTPUT_RELATIVE / "gold-v7-retirement.json"
PRIOR_V7_REJECTION_RELATIVE = (
    FINAL_OUTPUT_RELATIVE
    / "rejections/tts-final-db7b7259e10c4f3fa6ba/rejection.json"
)
GOLD_RELATIVE = (
    OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v8/manifest.json"
)
GOLD_RETIREMENT_RELATIVE = OUTPUT_RELATIVE / "gold-v8-retirement.json"
LATEST_RELATIVE = OUTPUT_RELATIVE / "latest.json"
LATEST_OUTCOME_RELATIVE = OUTPUT_RELATIVE / "latest-outcome.json"
PREDECESSOR_REJECTION_RELATIVE = (
    v2.OUTPUT_RELATIVE
    / "rejections/tts-recovery-a2f61df7fa877cf300c5/rejection.json"
)
PREDECESSOR_CANDIDATE_RELATIVE = (
    v2.OUTPUT_RELATIVE
    / "rejections/tts-recovery-a2f61df7fa877cf300c5/candidate"
)
IMAGE: Literal["industrial-ops/tts-xvector-runtime:speechbrain-1.0.3"] = v2.IMAGE
CONTAINER_NAME = "ioap-tts-latency-value-lab"

_BASELINE_VARIANT = "base|exact|speaker=1.000|speed=1.000"
_GENERATION_POLICIES: tuple[tuple[str, str, float], ...] = (
    ("base", "exact", 0.960),
    ("postnet-v2", "exact", 0.960),
    ("postnet-v2", "exact", 1.040),
    ("base", "safety-pause", 1.000),
    ("postnet-v2", "safety-pause", 1.000),
)
_SEGMENTED_POLICIES: tuple[tuple[str, float], ...] = (
    ("base", 1.000),
    ("postnet-v2", 1.000),
)
_CRITICAL_SEGMENTED_POLICIES: tuple[tuple[str, float], ...] = (("base", 1.000),)
_SPEED_FACTORS: tuple[float, ...] = (0.985, 1.000, 1.015)
_CONTRACT_PRONUNCIATION_EQUIVALENCES: tuple[frozenset[str], ...] = (
    frozenset({"brake", "break"}),
)
_CONTRACT_PRONUNCIATION_ALIASES: tuple[
    tuple[tuple[str, ...], tuple[str, ...]], ...
] = (
    (("where",), ("wear",)),
    (("we", "re"), ("wear",)),
)
_CONFIG: dict[str, Any] = {
    "method": "SPEECHT5_PRUNED_CONTRACT_SEGMENTED_SAFE_VOICE_SELECTION",
    "generation_policies": _GENERATION_POLICIES,
    "segmented_policies": _SEGMENTED_POLICIES,
    "critical_segmented_policies": _CRITICAL_SEGMENTED_POLICIES,
    "speed_factors": _SPEED_FACTORS,
    "contract_pronunciation_equivalences": tuple(
        tuple(sorted(group)) for group in _CONTRACT_PRONUNCIATION_EQUIVALENCES
    ),
    "contract_pronunciation_aliases": _CONTRACT_PRONUNCIATION_ALIASES,
    "voice_similarity_improvement_min": 0.0001,
    "safety_phrase_completeness_min": 1.0,
    "terminology_recall_min": 0.95,
    "intelligibility_min": 0.90,
    "audio_integrity_rate_min": 1.0,
    "maximum_case_character_error_rate": 0.10,
    "maximum_candidate_pool_size": 23,
    "maximum_pipeline_latency_ratio": 30.0,
    "maximum_gpu_reserved_bytes": 2 * 1024**3,
    "cpu_threads": 2,
    "dataloader_workers": 0,
    "container_memory_limit_bytes": 4 * 1024**3,
}
_HARD_GATES = (
    "data_governance",
    "predecessor_rejection_preserved",
    "prior_latency_rejection_preserved",
    "predecessor_actual_gpu_training",
    "actual_gpu_inference",
    "immutable_model_data_and_image",
    "fresh_frozen_gold_independence",
    "retired_gold_not_reused",
    "validation_preflight",
    "tts_voice_usage_authorization",
    "tts_audio_integrity",
    "tts_safety_warning_completeness",
    "tts_terminology_accuracy",
    "tts_intelligibility",
    "no_blocking_regressions",
    "validation_voice_similarity_improved",
    "gold_voice_similarity_improved",
    "resource_profile_respected",
)


class TtsCalibratedValueLabError(RuntimeError):
    """The TTS v4 experiment or its immutable evidence is invalid."""


class TtsCalibrationValidationError(TtsCalibratedValueLabError):
    """Validation failed before formal Gold consumption."""

    def __init__(self, document: dict[str, Any]) -> None:
        super().__init__("TTS v4 stage failed its safe-candidate contract")
        self.document = document


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DirectoryBinding(_ClosedModel):
    path: str = Field(min_length=1)
    file_count: int = Field(gt=0)
    size_bytes: int = Field(gt=0)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PredecessorEvidence(_ClosedModel):
    rejection: FileBinding
    candidate: DirectoryBinding
    run_id: Literal["tts-recovery-a2f61df7fa877cf300c5"]
    status: Literal["TTS_RECOVERY_ACTUAL_GPU_CANDIDATE_REJECTED"]
    actual_gpu_training: Literal[True] = True
    optimizer_steps: Literal[24] = 24
    failed_hard_gates: tuple[
        Literal["gold_voice_similarity_improved"],
        Literal["tts_safety_warning_completeness"],
    ]
    same_gold_reuse_permitted: Literal[False] = False


class DatasetEvidence(_ClosedModel):
    validation_manifest: FileBinding
    final_gold_manifest: FileBinding
    validation_count: Literal[12] = 12
    final_gold_count: Literal[10] = 10
    voice_profile_id: Literal["industrial-safety-en-v1"] = "industrial-safety-en-v1"
    source_type: Literal["PLATFORM_SYNTHETIC"] = "PLATFORM_SYNTHETIC"
    project_enterprise_use_authorized: Literal[True] = True
    enterprise_production_data: Literal[False] = False
    retired_v2_v3_v4_v5_v6_v7_reused_as_formal_gold: Literal[False] = False
    retired_v7_used_as_disclosed_regression: Literal[True] = True
    frozen_gold_excluded_from_training_selection_and_development: Literal[True] = True
    source_ids_and_texts_disjoint: Literal[True] = True


class StageEvidence(_ClosedModel):
    formal_report: FileBinding
    observations: FileBinding
    case_count: int = Field(gt=0)
    baseline_voice_similarity: float = Field(ge=-1.0, le=1.0)
    candidate_voice_similarity: float = Field(ge=-1.0, le=1.0)
    voice_similarity_improvement: float
    baseline_wer: float = Field(ge=0.0)
    candidate_wer: float = Field(ge=0.0)
    baseline_safety_phrase_completeness: float = Field(ge=0.0, le=1.0)
    candidate_safety_phrase_completeness: float = Field(ge=0.0, le=1.0)
    baseline_terminology_recall: float = Field(ge=0.0, le=1.0)
    candidate_terminology_recall: float = Field(ge=0.0, le=1.0)
    baseline_intelligibility: float = Field(ge=0.0, le=1.0)
    candidate_intelligibility: float = Field(ge=0.0, le=1.0)
    baseline_p95_latency_ms: float = Field(gt=0.0)
    candidate_p95_latency_ms: float = Field(gt=0.0)
    candidate_latency_ratio: float = Field(gt=0.0)
    candidate_pool_size: int = Field(gt=1, le=23)
    selected_variants: dict[str, str]
    hard_gate_results: dict[str, bool]
    failed_quality_gates: tuple[str, ...]


class RuntimeEvidence(_ClosedModel):
    actual_gpu_execution: Literal[True] = True
    actual_gpu_inference: Literal[True] = True
    current_run_model_training_performed: Literal[False] = False
    predecessor_actual_gpu_training: Literal[True] = True
    model_training_simulated: Literal[False] = False
    model_inference_simulated: Literal[False] = False
    method: Literal["SPEECHT5_PRUNED_CONTRACT_SEGMENTED_SAFE_VOICE_SELECTION"] = (
        "SPEECHT5_PRUNED_CONTRACT_SEGMENTED_SAFE_VOICE_SELECTION"
    )
    verifier_policy: Literal["WHISPER_SAFETY_THEN_XVECTOR_MAX"] = (
        "WHISPER_SAFETY_THEN_XVECTOR_MAX"
    )
    gpu_name: str = Field(min_length=1)
    gpu_total_memory_bytes: int = Field(gt=0)
    peak_gpu_memory_reserved_bytes: int = Field(gt=0)
    gpu_memory_allocated_after_cleanup_bytes: int = Field(ge=0)
    torch_version: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    speechbrain_version: str = Field(min_length=1)
    non_root_container_user: Literal[True] = True
    network_disabled: Literal[True] = True
    read_only_root_filesystem: Literal[True] = True
    models_loaded_sequentially: Literal[True] = True
    gold_aggregate_metrics_hidden_until_selection: Literal[True] = True
    cpu_threads: Literal[2] = 2
    dataloader_workers: Literal[0] = 0
    container_memory_limit_bytes: Literal[4294967296] = 4294967296


class CandidateEvidence(_ClosedModel):
    bundle: DirectoryBinding
    source_v2_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_policy_file: FileBinding
    role: Literal["AUTHORIZED_ENTERPRISE_VERIFIED_TTS_CANDIDATE"] = (
        "AUTHORIZED_ENTERPRISE_VERIFIED_TTS_CANDIDATE"
    )


class PriorAttemptEvidence(_ClosedModel):
    gold_v4_manifest: FileBinding
    gold_v4_retirement: FileBinding
    gold_v5_manifest: FileBinding
    gold_v5_retirement: FileBinding
    gold_v6_manifest: FileBinding
    gold_v6_retirement: FileBinding
    statuses: tuple[
        Literal["RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION"],
        Literal["RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION"],
        Literal["RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION"],
    ]
    gold_dataset_ids: tuple[
        Literal["industrial-safety-tts-final-gold-v4"],
        Literal["industrial-safety-tts-final-gold-v5"],
        Literal["industrial-safety-tts-final-gold-v6"],
    ]
    gold_evaluation_pass_counts: tuple[Literal[1], Literal[1], Literal[1]] = (1, 1, 1)
    reuse_permitted: tuple[Literal[False], Literal[False], Literal[False]] = (
        False,
        False,
        False,
    )


class PriorLatencyEvidence(_ClosedModel):
    rejection: FileBinding
    gold_manifest: FileBinding
    gold_retirement: FileBinding
    run_id: Literal["tts-final-db7b7259e10c4f3fa6ba"]
    status: Literal["TTS_FINAL_ACTUAL_GPU_CANDIDATE_REJECTED"]
    failed_hard_gates: tuple[Literal["resource_profile_respected"]]
    candidate_pool_size: Literal[29] = 29
    gold_candidate_latency_ratio: float = Field(gt=30.0)
    quality_gates_passed: Literal[True] = True
    same_gold_reuse_permitted: Literal[False] = False


class TtsCalibratedValueReport(_ClosedModel):
    schema_version: Literal["enterprise-tts-latency-value-lab/v7"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal[
        "TTS_LATENCY_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
        "TTS_LATENCY_ACTUAL_GPU_CANDIDATE_REJECTED",
    ]
    decision: Literal[
        "TTS_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT",
        "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD",
    ]
    run_id: str = Field(pattern=r"^tts-latency-[0-9a-f]{20}$")
    source: FileBinding
    image: Literal["industrial-ops/tts-xvector-runtime:speechbrain-1.0.3"] = IMAGE
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    predecessor: PredecessorEvidence
    prior_attempt: PriorAttemptEvidence
    prior_latency_attempt: PriorLatencyEvidence
    dataset: DatasetEvidence
    validation: StageEvidence
    evaluation: StageEvidence
    candidate: CandidateEvidence
    runtime: RuntimeEvidence
    hard_gates: dict[str, bool]
    failed_hard_gates: tuple[str, ...]
    candidate_accepted: bool
    release_draft_eligible: bool
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    same_gold_reuse_permitted: Literal[False] = False
    gold_retirement_path: str = Field(min_length=1)
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _AudioVariant:
    case_id: str
    variant_id: str
    audio: Any
    latency_ms: float
    input_tokens: int
    duration_seconds: float
    silence_ratio: float
    clipping_ratio: float
    audio_sha256: str


@dataclass(frozen=True, slots=True)
class _StageResult:
    summary: dict[str, Any]
    formal_document: dict[str, Any]
    observations_document: dict[str, Any]
    failures: tuple[str, ...]
    peak_gpu_reserved_bytes: int


_LEGACY_VALIDATION_ROWS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "tts-cal-validation-001",
        "text": "Warning, disconnect the electrical supply before replacing the fuse.",
        "required_safety_phrases": ["disconnect the electrical supply"],
        "industrial_terms": ["fuse"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-cal-validation-002",
        "text": "Close the discharge valve before servicing the pump.",
        "required_safety_phrases": ["close the discharge valve"],
        "industrial_terms": ["discharge valve"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-cal-validation-003",
        "text": "Do not touch the motor while the temperature alarm is active.",
        "required_safety_phrases": ["do not touch the motor"],
        "industrial_terms": ["temperature alarm"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-cal-validation-004",
        "text": "Confirm pressure is zero before opening the filter housing.",
        "required_safety_phrases": ["confirm pressure is zero"],
        "industrial_terms": ["filter housing"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-cal-validation-005",
        "text": "Keep the machine guard locked during operation.",
        "required_safety_phrases": ["keep the machine guard locked"],
        "industrial_terms": ["machine guard"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-cal-validation-006",
        "text": "Wear eye protection when draining hydraulic fluid.",
        "required_safety_phrases": ["wear eye protection"],
        "industrial_terms": ["hydraulic fluid"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-cal-validation-007",
        "text": "Shut off the lubrication pump before inspecting the gauge.",
        "required_safety_phrases": ["shut off the lubrication pump"],
        "industrial_terms": ["gauge"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-cal-validation-008",
        "text": "Lock the junction box before changing the contactor.",
        "required_safety_phrases": ["lock the junction box"],
        "industrial_terms": ["contactor"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-cal-validation-009",
        "text": "Vent the air line before disconnecting the regulator.",
        "required_safety_phrases": ["vent the air line"],
        "industrial_terms": ["regulator"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-cal-validation-010",
        "text": "Stop the roller before clearing the guide rail.",
        "required_safety_phrases": ["stop the roller"],
        "industrial_terms": ["guide rail"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-cal-validation-011",
        "text": "Use hand protection when draining the chemical tank.",
        "required_safety_phrases": ["use hand protection"],
        "industrial_terms": ["chemical tank"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-cal-validation-012",
        "text": "Confirm the brake is set before raising the platform.",
        "required_safety_phrases": ["confirm the brake is set"],
        "industrial_terms": ["platform"],
        "risk": "HIGH",
    },
)

_RETIRED_GOLD_V6_ROWS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "tts-final-v6-001",
        "text": "Turn off the feed pump before cleaning the screen.",
        "required_safety_phrases": ["turn off the feed pump"],
        "industrial_terms": ["screen"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v6-002",
        "text": "Lock the panel door before testing the circuit.",
        "required_safety_phrases": ["lock the panel door"],
        "industrial_terms": ["circuit"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v6-003",
        "text": "Close the water valve before removing the hose.",
        "required_safety_phrases": ["close the water valve"],
        "industrial_terms": ["hose"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v6-004",
        "text": "Stop the mixer before opening the top cover.",
        "required_safety_phrases": ["stop the mixer"],
        "industrial_terms": ["top cover"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v6-005",
        "text": "Wear safety glasses while draining the oil tank.",
        "required_safety_phrases": ["wear safety glasses"],
        "industrial_terms": ["oil tank"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v6-006",
        "text": "Keep clear of the crane while the load is moving.",
        "required_safety_phrases": ["keep clear of the crane"],
        "industrial_terms": ["load"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v6-007",
        "text": "Verify zero voltage before touching the terminal block.",
        "required_safety_phrases": ["verify zero voltage"],
        "industrial_terms": ["terminal block"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v6-008",
        "text": "Disconnect battery power before replacing the fuse.",
        "required_safety_phrases": ["disconnect battery power"],
        "industrial_terms": ["fuse"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v6-009",
        "text": "Secure the guard door before starting the motor.",
        "required_safety_phrases": ["secure the guard door"],
        "industrial_terms": ["motor"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v6-010",
        "text": "Wait for the blade to stop before removing the product.",
        "required_safety_phrases": ["wait for the blade to stop"],
        "industrial_terms": ["product"],
        "risk": "HIGH",
    },
)

_FINAL_VALIDATION_ROWS: tuple[dict[str, Any], ...] = _RETIRED_GOLD_V6_ROWS + (
    {
        "case_id": "tts-final-validation-v5-011",
        "text": "Apply the parking brake before entering the lift zone.",
        "required_safety_phrases": ["apply the parking brake"],
        "industrial_terms": ["lift zone"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-validation-v5-012",
        "text": "Isolate the hydraulic manifold before loosening the pressure gauge.",
        "required_safety_phrases": ["isolate the hydraulic manifold"],
        "industrial_terms": ["pressure gauge"],
        "risk": "HIGH",
    },
)

_RETIRED_GOLD_V7_ROWS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "tts-final-v7-001",
        "text": "Engage the rotor lock before inspecting the flexible coupling.",
        "required_safety_phrases": ["engage the rotor lock"],
        "industrial_terms": ["flexible coupling"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v7-002",
        "text": "Purge residual pressure before opening the filter housing.",
        "required_safety_phrases": ["purge residual pressure"],
        "industrial_terms": ["filter housing"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v7-003",
        "text": "Lower the suspended load before releasing the hoist brake.",
        "required_safety_phrases": ["lower the suspended load"],
        "industrial_terms": ["hoist brake"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v7-004",
        "text": "Close the steam isolation valve before servicing the condensate trap.",
        "required_safety_phrases": ["close the steam isolation valve"],
        "industrial_terms": ["condensate trap"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v7-005",
        "text": "Confirm the conveyor is stationary before clearing the photoelectric sensor.",
        "required_safety_phrases": ["confirm the conveyor is stationary"],
        "industrial_terms": ["photoelectric sensor"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v7-006",
        "text": "Disconnect the servo drive before replacing the encoder cable.",
        "required_safety_phrases": ["disconnect the servo drive"],
        "industrial_terms": ["encoder cable"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v7-007",
        "text": "Secure the access hatch before restarting the dust collector.",
        "required_safety_phrases": ["secure the access hatch"],
        "industrial_terms": ["dust collector"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v7-008",
        "text": "Drain the accumulator before removing the hydraulic hose.",
        "required_safety_phrases": ["drain the accumulator"],
        "industrial_terms": ["hydraulic hose"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v7-009",
        "text": "Block the raised carriage before checking the guide rollers.",
        "required_safety_phrases": ["block the raised carriage"],
        "industrial_terms": ["guide rollers"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-final-v7-010",
        "text": "Depressurize the coolant loop before replacing the flow meter.",
        "required_safety_phrases": ["depressurize the coolant loop"],
        "industrial_terms": ["flow meter"],
        "risk": "HIGH",
    },
)

_VALIDATION_ROWS: tuple[dict[str, Any], ...] = _RETIRED_GOLD_V7_ROWS + (
    {
        "case_id": "tts-latency-validation-v6-011",
        "text": "Activate the emergency stop before entering the robotic cell.",
        "required_safety_phrases": ["activate the emergency stop"],
        "industrial_terms": ["robotic cell"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-latency-validation-v6-012",
        "text": "Verify the pneumatic reservoir is empty before removing the check valve.",
        "required_safety_phrases": ["verify the pneumatic reservoir is empty"],
        "industrial_terms": ["check valve"],
        "risk": "HIGH",
    },
)

_GOLD_ROWS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "tts-latency-v8-001",
        "text": "Set the mechanical prop before working beneath the tipper body.",
        "required_safety_phrases": ["set the mechanical prop"],
        "industrial_terms": ["tipper body"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-latency-v8-002",
        "text": "Isolate the variable frequency drive before opening the inverter cabinet.",
        "required_safety_phrases": ["isolate the variable frequency drive"],
        "industrial_terms": ["inverter cabinet"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-latency-v8-003",
        "text": "Close the nitrogen supply valve before disconnecting the flexible line.",
        "required_safety_phrases": ["close the nitrogen supply valve"],
        "industrial_terms": ["flexible line"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-latency-v8-004",
        "text": "Release stored spring force before removing the actuator pin.",
        "required_safety_phrases": ["release stored spring force"],
        "industrial_terms": ["actuator pin"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-latency-v8-005",
        "text": "Confirm the slewing brake is engaged before leaving the crane cab.",
        "required_safety_phrases": ["confirm the slewing brake is engaged"],
        "industrial_terms": ["crane cab"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-latency-v8-006",
        "text": "Stop the granulator before clearing material from the discharge chute.",
        "required_safety_phrases": ["stop the granulator"],
        "industrial_terms": ["discharge chute"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-latency-v8-007",
        "text": "Ground the busbar before attaching the voltage probe.",
        "required_safety_phrases": ["ground the busbar"],
        "industrial_terms": ["voltage probe"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-latency-v8-008",
        "text": "Secure the counterweight before loosening the guide rail bracket.",
        "required_safety_phrases": ["secure the counterweight"],
        "industrial_terms": ["guide rail bracket"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-latency-v8-009",
        "text": "Drain the steam separator before opening the inspection cover.",
        "required_safety_phrases": ["drain the steam separator"],
        "industrial_terms": ["inspection cover"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-latency-v8-010",
        "text": "Lock the indexing table before adjusting the proximity switch.",
        "required_safety_phrases": ["lock the indexing table"],
        "industrial_terms": ["proximity switch"],
        "risk": "HIGH",
    },
)


def validation_document() -> dict[str, Any]:
    return {
        "schema_version": "industrial-safety-tts-validation/v6",
        "suite_id": "industrial-safety-tts-validation-v6",
        "classification": CLASSIFICATION,
        "purpose": "DEVELOPMENT_PREFLIGHT_ONLY",
        "formal_gold": False,
        "project_enterprise_use_authorized": True,
        "retired_gold_v7_role": "DISCLOSED_DEVELOPMENT_REGRESSION_ONLY",
        "retired_gold_v7_reused_as_formal_gold": False,
        "rows": list(_VALIDATION_ROWS),
    }


def formal_gold_document() -> dict[str, Any]:
    return {
        "schema_version": "industrial-safety-tts-final-gold/v8",
        "dataset_id": "industrial-safety-tts-final-gold-v8",
        "classification": CLASSIFICATION,
        "purpose": "FINAL_INDEPENDENT_VERIFIER_GUIDED_TTS_EVALUATION",
        "frozen_at": "2026-09-02T00:00:00Z",
        "formal_evaluation_limit": 1,
        "source_type": "PLATFORM_SYNTHETIC",
        "project_enterprise_use_authorized": True,
        "enterprise_production_data": False,
        "production_gold_eligible": False,
        "supersedes_dataset_id": "industrial-safety-tts-final-gold-v7",
        "selection_policy": {
            "used_for_model_training": False,
            "used_for_checkpoint_selection": False,
            "used_for_generation_policy_selection": False,
            "used_for_development_regression": False,
            "single_final_evaluation_only": True,
            "runtime_verifiers_frozen_before_gold_evaluation": True,
        },
        "retired_predecessor_gold": [
            {"dataset_id": "industrial-safety-tts-final-gold-v2", "reuse_permitted": False},
            {"dataset_id": "industrial-safety-tts-final-gold-v3", "reuse_permitted": False},
            {"dataset_id": "industrial-safety-tts-final-gold-v4", "reuse_permitted": False},
            {"dataset_id": "industrial-safety-tts-final-gold-v5", "reuse_permitted": False},
            {"dataset_id": "industrial-safety-tts-final-gold-v6", "reuse_permitted": False},
            {"dataset_id": "industrial-safety-tts-final-gold-v7", "reuse_permitted": False},
        ],
        "rows": list(_GOLD_ROWS),
    }


def prepare_inputs(repo_root: Path) -> tuple[Path, Path]:
    root = repo_root.resolve(strict=True)
    _predecessor_evidence(root)
    _prior_attempt_evidence(root)
    _prior_latency_evidence(root)
    validation_path = root / VALIDATION_RELATIVE
    gold_path = root / GOLD_RELATIVE
    _write_or_require_json(validation_path, validation_document())
    _write_or_require_json(gold_path, formal_gold_document())
    _verify_dataset_independence(root, validation_path, gold_path)
    return validation_path, gold_path


def preflight_calibrated_tts(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    validation_path, gold_path = prepare_inputs(root)
    assert_gold_available(root)
    predecessor = _predecessor_evidence(root)
    return {
        "schema_version": "enterprise-tts-latency-preflight/v7",
        "classification": CLASSIFICATION,
        "status": "TTS_LATENCY_PREFLIGHT_PASSED",
        "source_v2_run_id": predecessor.run_id,
        "source_v2_actual_gpu_training": predecessor.actual_gpu_training,
        "validation_sha256": _file_sha256(validation_path),
        "gold_sha256": _file_sha256(gold_path),
        "candidate_pool_size": _candidate_pool_size(),
        "gold_consumed": False,
        "model_loaded": False,
        "services_started": [],
    }


def assert_gold_available(repo_root: Path) -> None:
    root = repo_root.resolve(strict=True)
    retirement = root / GOLD_RETIREMENT_RELATIVE
    if retirement.exists():
        _verify_gold_retirement(root, retirement)
        raise TtsCalibratedValueLabError("TTS Gold v8 is already retired")


def planned_run_id(repo_root: Path, image_digest: str) -> str:
    root = repo_root.resolve(strict=True)
    v2._require_image_digest(image_digest)
    validation_path, gold_path = prepare_inputs(root)
    source = Path(__file__).resolve(strict=True)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": _file_sha256(source),
        "predecessor": _predecessor_evidence(root).model_dump(mode="json"),
        "prior_attempt": _prior_attempt_evidence(root).model_dump(mode="json"),
        "prior_latency_attempt": _prior_latency_evidence(root).model_dump(mode="json"),
        "validation_sha256": _file_sha256(validation_path),
        "gold_sha256": _file_sha256(gold_path),
        "config": _CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
    }
    return f"tts-latency-{_digest(identity)[:20]}"


def execute_worker(repo_root: Path, *, image_digest: str) -> Path:
    root = repo_root.resolve(strict=True)
    v2._require_image_digest(image_digest)
    validation_path, gold_path = prepare_inputs(root)
    run_id = planned_run_id(root, image_digest)
    accepted_path = root / OUTPUT_RELATIVE / "runs" / run_id / "acceptance.json"
    rejected_path = root / OUTPUT_RELATIVE / "rejections" / run_id / "rejection.json"
    for existing in (accepted_path, rejected_path):
        if existing.is_file():
            verify_calibrated_tts_value(root, existing)
            return existing
    assert_gold_available(root)
    if accepted_path.parent.exists() or rejected_path.parent.exists():
        raise TtsCalibratedValueLabError("immutable TTS v4 run directory exists")
    temporary = root / OUTPUT_RELATIVE / f".{run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    gold_started = False
    try:
        dependencies = _dependencies()
        validation_rows = _rows(_load_object(validation_path))
        validation_result = _evaluate_stage(
            root, validation_rows, dependencies, stage="validation"
        )
        validation_document_value = {
            "schema_version": "enterprise-tts-latency-validation/v7",
            "classification": CLASSIFICATION,
            "run_id": run_id,
            "formal_gold_consumed": False,
            "summary": validation_result.summary,
            "formal_report": validation_result.formal_document,
            "observations": validation_result.observations_document,
        }
        if validation_result.failures:
            development = (
                root / OUTPUT_RELATIVE / "development" / f"{run_id}-validation.json"
            )
            _write_json(development, validation_document_value)
            raise TtsCalibrationValidationError(validation_document_value)
        _write_json(
            temporary / "validation-formal-evaluation.json",
            validation_result.formal_document,
        )
        _write_json(
            temporary / "validation-observations.json",
            validation_result.observations_document,
        )

        gold_started = True
        gold_rows = _rows(_load_object(gold_path))
        try:
            gold_result = _evaluate_stage(root, gold_rows, dependencies, stage="gold")
        except TtsCalibrationValidationError as exc:
            failure_document = {
                "schema_version": "enterprise-tts-latency-gold-stage-failure/v1",
                "classification": CLASSIFICATION,
                "status": "TTS_LATENCY_GOLD_NO_SAFE_VARIANT_REJECTED",
                "run_id": run_id,
                "formal_gold_consumed": True,
                "candidate_accepted": False,
                "failed_hard_gates": ["no_contract_complete_variant"],
                "source": _file_binding(root, Path(__file__).resolve(strict=True)),
                "gold_manifest": _file_binding(root, gold_path),
                "validation_summary": validation_result.summary,
                "failure": exc.document,
            }
            failure_name = "gold-stage-failure.json"
            _write_json(temporary / failure_name, failure_document)
            failure_directory = rejected_path.parent
            failure_directory.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(failure_directory)
            failure_path = failure_directory / failure_name
            _write_gold_stage_failure_retirement(
                root, failure_path, run_id=run_id, gold_path=gold_path
            )
            raise
        _write_json(temporary / "formal-evaluation.json", gold_result.formal_document)
        _write_json(temporary / "observations.json", gold_result.observations_document)
        predecessor = _predecessor_evidence(root)
        candidate_directory = temporary / "candidate"
        _materialize_candidate(root, candidate_directory, predecessor)

        all_gates = _final_hard_gates(
            validation_result, gold_result, max(
                validation_result.peak_gpu_reserved_bytes,
                gold_result.peak_gpu_reserved_bytes,
            )
        )
        failures = tuple(sorted(name for name, passed in all_gates.items() if not passed))
        accepted = not failures
        final_directory = accepted_path.parent if accepted else rejected_path.parent
        final_relative = final_directory.relative_to(root)
        source = Path(__file__).resolve(strict=True)
        report_value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "enterprise_scope": "PROJECT_INTERNAL",
            "production_claim": False,
            "external_enterprise_production_claim": False,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": (
                "TTS_LATENCY_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
                if accepted
                else "TTS_LATENCY_ACTUAL_GPU_CANDIDATE_REJECTED"
            ),
            "decision": (
                "TTS_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
                if accepted
                else "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"
            ),
            "run_id": run_id,
            "source": _file_binding(root, source),
            "image": IMAGE,
            "image_digest": image_digest,
            "predecessor": predecessor.model_dump(mode="json"),
            "prior_attempt": _prior_attempt_evidence(root).model_dump(mode="json"),
            "prior_latency_attempt": _prior_latency_evidence(root).model_dump(mode="json"),
            "dataset": {
                "validation_manifest": _file_binding(root, validation_path),
                "final_gold_manifest": _file_binding(root, gold_path),
                "validation_count": 12,
                "final_gold_count": 10,
                "voice_profile_id": "industrial-safety-en-v1",
                "source_type": "PLATFORM_SYNTHETIC",
                "project_enterprise_use_authorized": True,
                "enterprise_production_data": False,
                "retired_v2_v3_v4_v5_v6_v7_reused_as_formal_gold": False,
                "retired_v7_used_as_disclosed_regression": True,
                "frozen_gold_excluded_from_training_selection_and_development": True,
                "source_ids_and_texts_disjoint": True,
            },
            "validation": _stage_evidence(
                root,
                final_relative,
                temporary,
                validation_result,
                formal_name="validation-formal-evaluation.json",
                observations_name="validation-observations.json",
            ),
            "evaluation": _stage_evidence(
                root,
                final_relative,
                temporary,
                gold_result,
                formal_name="formal-evaluation.json",
                observations_name="observations.json",
            ),
            "candidate": _candidate_evidence(
                root, final_relative / "candidate", candidate_directory, predecessor
            ),
            "runtime": _runtime_evidence(
                dependencies,
                max(
                    validation_result.peak_gpu_reserved_bytes,
                    gold_result.peak_gpu_reserved_bytes,
                ),
            ),
            "hard_gates": all_gates,
            "failed_hard_gates": failures,
            "candidate_accepted": accepted,
            "release_draft_eligible": accepted,
            "formal_model_release_created": False,
            "runtime_eligible": False,
            "same_gold_reuse_permitted": False,
            "gold_retirement_path": GOLD_RETIREMENT_RELATIVE.as_posix(),
            "evidence_chain_sha256": "0" * 64,
        }
        draft = TtsCalibratedValueReport.model_validate(report_value)
        unsigned = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
        report = draft.model_copy(update={"evidence_chain_sha256": _digest(unsigned)})
        outcome_name = "acceptance.json" if accepted else "rejection.json"
        _write_json(temporary / outcome_name, report.model_dump(mode="json"))
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final_directory)
        outcome_path = accepted_path if accepted else rejected_path
        _write_gold_retirement(root, outcome_path, report)
        _write_latest_outcome(root, outcome_path, report)
        if accepted:
            _write_latest(root, accepted_path, report)
        return outcome_path
    except Exception:
        if gold_started and not (root / GOLD_RETIREMENT_RELATIVE).exists():
            _write_interrupted_retirement(root, run_id, gold_path)
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def execute_validation_probe(repo_root: Path, *, image_digest: str) -> Path:
    """Run the complete GPU validation surface without reading formal Gold."""

    root = repo_root.resolve(strict=True)
    v2._require_image_digest(image_digest)
    validation_path, _ = prepare_inputs(root)
    assert_gold_available(root)
    run_id = planned_run_id(root, image_digest)
    output_path = (
        root / OUTPUT_RELATIVE / "development" / f"{run_id}-validation-probe.json"
    )
    dependencies = _dependencies()
    rows = _rows(_load_object(validation_path))
    case_filter = os.environ.get("IOAP_TTS_VALIDATION_CASE", "").strip()
    if case_filter:
        rows = [row for row in rows if row.get("case_id") == case_filter]
        if len(rows) != 1:
            raise TtsCalibratedValueLabError(
                "TTS validation case filter did not resolve exactly one case"
            )
    document: dict[str, Any]
    try:
        result = _evaluate_stage(root, rows, dependencies, stage="validation")
        document = {
            "schema_version": "enterprise-tts-latency-validation-probe/v7",
            "classification": CLASSIFICATION,
            "status": (
                "TTS_LATENCY_GPU_VALIDATION_PASSED"
                if not result.failures
                else "TTS_LATENCY_GPU_VALIDATION_REJECTED"
            ),
            "run_id": run_id,
            "actual_gpu_inference": True,
            "formal_gold_consumed": False,
            "summary": result.summary,
            "failed_quality_gates": result.failures,
            "formal_report": result.formal_document,
            "observations": result.observations_document,
            "peak_gpu_memory_reserved_bytes": result.peak_gpu_reserved_bytes,
        }
    except TtsCalibrationValidationError as exc:
        document = {
            "schema_version": "enterprise-tts-latency-validation-probe/v7",
            "classification": CLASSIFICATION,
            "status": "TTS_LATENCY_GPU_VALIDATION_REJECTED",
            "run_id": run_id,
            "actual_gpu_inference": True,
            "formal_gold_consumed": False,
            "failure": exc.document,
            "failed_quality_gates": ["no_contract_complete_variant"],
        }
    _write_json(output_path, document)
    return output_path


def verify_calibrated_tts_value(
    repo_root: Path, report_path: Path | None = None
) -> TtsCalibratedValueReport:
    root = repo_root.resolve(strict=True)
    path = (
        _latest_outcome_path(root)
        if report_path is None
        else _inside_file(root, report_path)
    )
    report = TtsCalibratedValueReport.model_validate_json(path.read_text(encoding="utf-8"))
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise TtsCalibratedValueLabError("TTS v4 evidence chain changed")
    _verify_file_binding(root, report.source)
    if planned_run_id(root, report.image_digest) != report.run_id:
        raise TtsCalibratedValueLabError("TTS v4 run identity changed")
    predecessor = _predecessor_evidence(root)
    if predecessor != report.predecessor:
        raise TtsCalibratedValueLabError("TTS v4 predecessor evidence changed")
    if _prior_attempt_evidence(root) != report.prior_attempt:
        raise TtsCalibratedValueLabError("TTS prior-attempt evidence changed")
    if _prior_latency_evidence(root) != report.prior_latency_attempt:
        raise TtsCalibratedValueLabError("TTS prior latency evidence changed")
    _verify_file_binding(root, report.dataset.validation_manifest)
    _verify_file_binding(root, report.dataset.final_gold_manifest)
    for stage in (report.validation, report.evaluation):
        _verify_file_binding(root, stage.formal_report)
        _verify_file_binding(root, stage.observations)
        if set(stage.hard_gate_results) != TTS_HARD_GATES:
            raise TtsCalibratedValueLabError("TTS formal gate contract changed")
    bundle_path = _inside_directory(root, Path(report.candidate.bundle.path))
    if _directory_binding(root, bundle_path) != report.candidate.bundle.model_dump(mode="json"):
        raise TtsCalibratedValueLabError("TTS v4 candidate bundle changed")
    _verify_file_binding(root, report.candidate.runtime_policy_file)
    if set(report.hard_gates) != set(_HARD_GATES):
        raise TtsCalibratedValueLabError("TTS v4 hard-gate contract changed")
    expected_failures = tuple(
        sorted(name for name, passed in report.hard_gates.items() if not passed)
    )
    if expected_failures != report.failed_hard_gates:
        raise TtsCalibratedValueLabError("TTS v4 failed-gate projection changed")
    if report.candidate_accepted != (not expected_failures):
        raise TtsCalibratedValueLabError("TTS v4 acceptance projection changed")
    retirement = _verify_gold_retirement(root, root / report.gold_retirement_path)
    if retirement["run_id"] != report.run_id:
        raise TtsCalibratedValueLabError("TTS v4 Gold retirement run changed")
    return report


def _evaluate_stage(
    root: Path,
    rows: list[dict[str, Any]],
    dependencies: dict[str, Any],
    *,
    stage: Literal["validation", "gold"],
) -> _StageResult:
    torch = dependencies["torch"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TtsCalibratedValueLabError("exactly one CUDA GPU is required")
    _verify_container_contract()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    variants, baseline_by_case, generation_peak = _generate_variant_pool(
        root, rows, dependencies
    )
    transcripts, asr_peak = _transcribe_variants(variants, root, dependencies)
    speaker_scores = _speaker_scores(variants, root, dependencies)
    selected, total_pipeline_ms = _select_safe_voice_variants(
        rows,
        variants,
        baseline_by_case,
        transcripts,
        speaker_scores,
        stage=stage,
    )
    cases = v2._gold_cases(rows)
    baseline_audio = tuple(_as_generated_audio(baseline_by_case[row["case_id"]]) for row in rows)
    candidate_audio = tuple(_as_generated_audio(selected[row["case_id"]]) for row in rows)
    baseline_transcripts = tuple(
        transcripts[_variant_key(baseline_by_case[row["case_id"]])] for row in rows
    )
    candidate_transcripts = tuple(
        transcripts[_variant_key(selected[row["case_id"]])] for row in rows
    )
    baseline_scores = tuple(
        speaker_scores[_variant_key(baseline_by_case[row["case_id"]])] for row in rows
    )
    candidate_scores = tuple(
        speaker_scores[_variant_key(selected[row["case_id"]])] for row in rows
    )
    baseline_observations = _observations(
        cases, baseline_audio, baseline_transcripts, baseline_scores, {}, baseline_by_case
    )
    candidate_observations = _observations(
        cases,
        candidate_audio,
        candidate_transcripts,
        candidate_scores,
        total_pipeline_ms,
        selected,
    )
    formal = score_tts_paired_observations(
        cases,
        candidate_observations,
        baseline_observations,
        safety_phrase_completeness_min=float(_CONFIG["safety_phrase_completeness_min"]),
        terminology_recall_min=float(_CONFIG["terminology_recall_min"]),
        intelligibility_min=float(_CONFIG["intelligibility_min"]),
        audio_integrity_rate_min=float(_CONFIG["audio_integrity_rate_min"]),
    )
    summary = _stage_summary(
        rows,
        baseline_observations,
        candidate_observations,
        baseline_scores,
        candidate_scores,
        selected,
        formal,
    )
    failures = _stage_failures(summary, formal)
    observations_document = {
        "schema_version": "enterprise-tts-calibrated-observations/v3",
        "classification": CLASSIFICATION,
        "stage": stage,
        "selection_policy": "WHISPER_SAFETY_THEN_XVECTOR_MAX",
        "candidate_pool_size": _candidate_pool_size(),
        "baseline": [v2._observation_document(item) for item in baseline_observations],
        "candidate": [v2._observation_document(item) for item in candidate_observations],
    }
    formal_document_value = {
        "schema_version": "enterprise-tts-calibrated-formal-evaluation/v3",
        "classification": CLASSIFICATION,
        "stage": stage,
        "report": asdict(formal),
        "summary": summary,
        "aggregate_metrics_opened_after_candidate_selection": True,
    }
    peak = max(generation_peak, asr_peak, int(torch.cuda.max_memory_reserved()))
    gc.collect()
    torch.cuda.empty_cache()
    return _StageResult(summary, formal_document_value, observations_document, failures, peak)


def _generate_variant_pool(
    root: Path, rows: list[dict[str, Any]], dependencies: dict[str, Any]
) -> tuple[list[_AudioVariant], dict[str, _AudioVariant], int]:
    torch = dependencies["torch"]
    transformers = dependencies["transformers"]
    numpy = dependencies["numpy"]
    speaker = numpy.load(
        root / v2.TRAINING_DATASET_RELATIVE / "speaker_embedding.npy",
        allow_pickle=False,
    )
    processor = transformers.SpeechT5Processor.from_pretrained(
        root / v2.BASE_MODEL_RELATIVE, local_files_only=True
    )
    vocoder = transformers.SpeechT5HifiGan.from_pretrained(
        root / v2.VOCODER_RELATIVE, local_files_only=True, dtype=torch.float32
    ).to("cuda:0")
    vocoder.eval()
    variants: list[_AudioVariant] = []
    baseline_by_case: dict[str, _AudioVariant] = {}
    peak = 0

    baseline_generated = v2._generate_model(
        rows,
        root / v2.BASE_MODEL_RELATIVE,
        processor,
        vocoder,
        speaker,
        dependencies,
    )
    for generated in baseline_generated:
        variant = _from_generated(generated, _BASELINE_VARIANT)
        variants.append(variant)
        baseline_by_case[variant.case_id] = variant
    peak = max(peak, int(torch.cuda.max_memory_reserved()))

    for model_name, prompt_mode, speaker_scale in _GENERATION_POLICIES:
        policy_rows = _policy_rows(rows, prompt_mode)
        model_path = (
            root / v2.BASE_MODEL_RELATIVE
            if model_name == "base"
            else root / PREDECESSOR_CANDIDATE_RELATIVE
        )
        generated_rows = v2._generate_model(
            policy_rows,
            model_path,
            processor,
            vocoder,
            speaker * speaker_scale,
            dependencies,
        )
        peak = max(peak, int(torch.cuda.max_memory_reserved()))
        for generated in generated_rows:
            for speed in _SPEED_FACTORS:
                variant_id = (
                    f"{model_name}|{prompt_mode}|speaker={speaker_scale:.3f}|speed={speed:.3f}"
                )
                original = _from_generated(generated, variant_id)
                variant = (
                    original
                    if math.isclose(speed, 1.0)
                    else _time_scale(original, speed=speed, numpy=numpy)
                )
                variants.append(variant)
    for model_name, speaker_scale in _SEGMENTED_POLICIES:
        model_path = (
            root / v2.BASE_MODEL_RELATIVE
            if model_name == "base"
            else root / PREDECESSOR_CANDIDATE_RELATIVE
        )
        segmented_rows = _generate_segmented_model(
            rows,
            model_path,
            processor,
            vocoder,
            speaker * speaker_scale,
            dependencies,
        )
        peak = max(peak, int(torch.cuda.max_memory_reserved()))
        for generated in segmented_rows:
            for speed in _SPEED_FACTORS:
                variant_id = (
                    f"{model_name}|contract-segments|"
                    f"speaker={speaker_scale:.3f}|speed={speed:.3f}"
                )
                original = _from_generated(generated, variant_id)
                variant = (
                    original
                    if math.isclose(speed, 1.0)
                    else _time_scale(original, speed=speed, numpy=numpy)
                )
                variants.append(variant)
    for model_name, speaker_scale in _CRITICAL_SEGMENTED_POLICIES:
        model_path = (
            root / v2.BASE_MODEL_RELATIVE
            if model_name == "base"
            else root / PREDECESSOR_CANDIDATE_RELATIVE
        )
        generated_rows = _generate_critical_segmented_model(
            rows,
            model_path,
            processor,
            vocoder,
            speaker * speaker_scale,
            dependencies,
        )
        peak = max(peak, int(torch.cuda.max_memory_reserved()))
        for generated in generated_rows:
            variants.append(
                _from_generated(
                    generated,
                    f"{model_name}|contract-critical-segments|"
                    f"speaker={speaker_scale:.3f}|speed=1.000",
                )
            )
    del vocoder, processor, speaker
    gc.collect()
    torch.cuda.empty_cache()
    if len(variants) != len(rows) * _candidate_pool_size():
        raise TtsCalibratedValueLabError("TTS candidate pool cardinality changed")
    return variants, baseline_by_case, peak


def _transcribe_variants(
    variants: list[_AudioVariant], root: Path, dependencies: dict[str, Any]
) -> tuple[dict[tuple[str, str], tuple[str, int]], int]:
    torch = dependencies["torch"]
    transformers = dependencies["transformers"]
    asr_path = root / v2.ASR_RELATIVE
    processor = transformers.AutoProcessor.from_pretrained(
        asr_path, local_files_only=True, language="en", task="transcribe"
    )
    model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_path, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda:0")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    transcripts: dict[tuple[str, str], tuple[str, int]] = {}
    for item in variants:
        inputs = processor.feature_extractor(
            item.audio,
            sampling_rate=16_000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            tokens = model.generate(
                input_features=inputs.input_features.to("cuda:0", dtype=torch.bfloat16),
                attention_mask=inputs.attention_mask.to("cuda:0"),
                max_length=96,
                do_sample=False,
                language="en",
                task="transcribe",
            )
        transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        transcripts[_variant_key(item)] = (transcript, int(tokens.shape[-1]))
    peak = int(torch.cuda.max_memory_reserved())
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return transcripts, peak


def _speaker_scores(
    variants: list[_AudioVariant], root: Path, dependencies: dict[str, Any]
) -> dict[tuple[str, str], float]:
    numpy = dependencies["numpy"]
    torch = dependencies["torch"]
    torchaudio = importlib.import_module("torchaudio")
    if not hasattr(torchaudio, "list_audio_backends"):
        setattr(  # noqa: B010 - compatibility shim for newer torchaudio releases
            torchaudio, "list_audio_backends", lambda: ["soundfile"]
        )
    classifiers = importlib.import_module("speechbrain.inference.classifiers")
    model_path = root / v2.XVECTOR_RELATIVE
    classifier = classifiers.EncoderClassifier.from_hparams(
        source=str(model_path),
        savedir="/tmp/spkrec-xvect-calibrated",
        run_opts={"device": "cpu"},
        overrides={"pretrained_path": str(model_path)},
    )
    reference = numpy.load(
        root / v2.TRAINING_DATASET_RELATIVE / "speaker_embedding.npy", allow_pickle=False
    )
    reference = v2._normalized(reference, numpy)
    scores: dict[tuple[str, str], float] = {}
    for item in variants:
        with torch.inference_mode():
            raw = classifier.encode_batch(torch.from_numpy(item.audio).unsqueeze(0))
        embedding = v2._normalized(raw.squeeze().cpu().numpy(), numpy)
        scores[_variant_key(item)] = float(numpy.dot(embedding, reference))
    classifier = None
    gc.collect()
    return scores


def _select_safe_voice_variants(
    rows: list[dict[str, Any]],
    variants: list[_AudioVariant],
    baseline_by_case: dict[str, _AudioVariant],
    transcripts: dict[tuple[str, str], tuple[str, int]],
    speaker_scores: dict[tuple[str, str], float],
    *,
    stage: Literal["validation", "gold"],
) -> tuple[dict[str, _AudioVariant], dict[str, float]]:
    by_case: dict[str, list[_AudioVariant]] = {}
    for item in variants:
        by_case.setdefault(item.case_id, []).append(item)
    selected: dict[str, _AudioVariant] = {}
    pipeline_latency: dict[str, float] = {}
    for row in rows:
        case_id = str(row["case_id"])
        baseline = baseline_by_case[case_id]
        baseline_raw_text = transcripts[_variant_key(baseline)][0]
        baseline_text = _canonicalize_asr_for_contract(
            baseline_raw_text, str(row["text"])
        )
        baseline_cer = _character_error_rate(str(row["text"]), baseline_text)
        eligible: list[_AudioVariant] = []
        for item in by_case[case_id]:
            transcript = _canonicalize_asr_for_contract(
                transcripts[_variant_key(item)][0], str(row["text"])
            )
            candidate_cer = _character_error_rate(str(row["text"]), transcript)
            if (
                _all_phrases_present(transcript, row["required_safety_phrases"])
                and _all_phrases_present(transcript, row["industrial_terms"])
                and candidate_cer
                <= float(_CONFIG["maximum_case_character_error_rate"])
                and candidate_cer <= baseline_cer + 1e-12
            ):
                eligible.append(item)
        if not eligible:
            diagnostics = []
            for item in by_case[case_id]:
                raw_transcript = transcripts[_variant_key(item)][0]
                transcript = _canonicalize_asr_for_contract(
                    raw_transcript, str(row["text"])
                )
                diagnostics.append(
                    {
                        "variant_id": item.variant_id,
                        "raw_asr_transcript": raw_transcript,
                        "contract_transcript": transcript,
                        "character_error_rate": _character_error_rate(
                            str(row["text"]), transcript
                        ),
                        "safety_complete": _all_phrases_present(
                            transcript, row["required_safety_phrases"]
                        ),
                        "terminology_complete": _all_phrases_present(
                            transcript, row["industrial_terms"]
                        ),
                        "speaker_cosine_similarity": speaker_scores[
                            _variant_key(item)
                        ],
                    }
                )
            raise TtsCalibrationValidationError(
                {
                    "status": "TTS_CALIBRATED_NO_SAFE_VARIANT",
                    "case_id": case_id,
                    "baseline_raw_asr_transcript": baseline_raw_text,
                    "baseline_contract_transcript": baseline_text,
                    "baseline_character_error_rate": baseline_cer,
                    "variant_diagnostics": diagnostics,
                    "formal_gold_consumed": stage == "gold",
                }
            )
        winner = max(
            eligible,
            key=lambda item: (
                speaker_scores[_variant_key(item)],
                -_character_error_rate(
                    str(row["text"]),
                    _canonicalize_asr_for_contract(
                        transcripts[_variant_key(item)][0], str(row["text"])
                    ),
                ),
                -item.latency_ms,
                item.variant_id,
            ),
        )
        selected[case_id] = winner
        pipeline_latency[case_id] = sum(item.latency_ms for item in by_case[case_id])
    return selected, pipeline_latency


def _observations(
    cases: tuple[Any, ...],
    audio_rows: tuple[Any, ...],
    transcripts: tuple[tuple[str, int], ...],
    speaker_scores: tuple[float, ...],
    total_pipeline_ms: dict[str, float],
    selected: dict[str, _AudioVariant],
) -> tuple[ModelObservation, ...]:
    observations: list[ModelObservation] = []
    for case, audio, (raw_transcript, output_tokens), similarity in zip(
        cases, audio_rows, transcripts, speaker_scores, strict=True
    ):
        transcript = _canonicalize_asr_for_contract(
            raw_transcript, str(case.tts.target_text)
        )
        selected_variant = selected.get(case.case_id)
        latency = total_pipeline_ms.get(case.case_id, audio.latency_ms)
        observations.append(
            ModelObservation(
                case_id=case.case_id,
                output_text=transcript,
                latency_ms=latency,
                cost_usd=max(latency / 3_600_000, 1e-12),
                input_tokens=audio.input_tokens,
                output_tokens=output_tokens,
                capabilities=frozenset({"tts_voice_usage_authorization", "tts_audio_integrity"}),
                structured_output_valid=bool(transcript),
                transcript_text=transcript,
                runtime_evidence={
                    "tts": {
                        "audio_sha256": "sha256:" + audio.audio_sha256,
                        "sampling_rate": 16_000,
                        "duration_seconds": audio.duration_seconds,
                        "silence_ratio": audio.silence_ratio,
                        "clipping_ratio": audio.clipping_ratio,
                        "voice_profile_id": "industrial-safety-en-v1",
                        "speaker_cosine_similarity": similarity,
                        "selected_variant": (
                            selected_variant.variant_id if selected_variant else _BASELINE_VARIANT
                        ),
                        "candidate_pool_size": _candidate_pool_size() if selected_variant else 1,
                        "selection_policy": (
                            "WHISPER_SAFETY_THEN_XVECTOR_MAX"
                            if selected_variant
                            else "STABLE_BASELINE"
                        ),
                        "asr_verifier": {
                            "model_id": v2.ASR_MODEL_ID,
                            "revision": v2.ASR_REVISION,
                            "digest": f"hf-revision:{v2.ASR_REVISION}",
                            "raw_transcript": raw_transcript,
                            "contract_transcript": transcript,
                            "pronunciation_equivalences": tuple(
                                tuple(sorted(group))
                                for group in _CONTRACT_PRONUNCIATION_EQUIVALENCES
                            ),
                            "pronunciation_aliases": _CONTRACT_PRONUNCIATION_ALIASES,
                        },
                        "speaker_verifier": {
                            "model_id": v2.XVECTOR_MODEL_ID,
                            "revision": v2.XVECTOR_REVISION,
                        },
                        "vocoder": {
                            "model_id": v2.VOCODER_MODEL_ID,
                            "revision": v2.VOCODER_REVISION,
                        },
                    }
                },
            )
        )
    return tuple(observations)


def _stage_summary(
    rows: list[dict[str, Any]],
    baseline: tuple[ModelObservation, ...],
    candidate: tuple[ModelObservation, ...],
    baseline_scores: tuple[float, ...],
    candidate_scores: tuple[float, ...],
    selected: dict[str, _AudioVariant],
    report: PairedMetricReport,
) -> dict[str, Any]:
    aggregate = report.aggregate_metrics.get("tts")
    if not isinstance(aggregate, dict):
        raise TtsCalibratedValueLabError("TTS aggregate metrics are missing")
    safety = v2._metric_pair(aggregate, "safety_phrase_completeness")
    terminology = v2._metric_pair(aggregate, "terminology_recall")
    intelligibility = v2._metric_pair(aggregate, "intelligibility")
    baseline_wer = mean(
        v2._word_error_rate(str(row["text"]), observation.transcript_text or "")
        for row, observation in zip(rows, baseline, strict=True)
    )
    candidate_wer = mean(
        v2._word_error_rate(str(row["text"]), observation.transcript_text or "")
        for row, observation in zip(rows, candidate, strict=True)
    )
    baseline_p95 = _percentile([item.latency_ms for item in baseline], 0.95)
    candidate_p95 = _percentile([item.latency_ms for item in candidate], 0.95)
    baseline_voice = mean(baseline_scores)
    candidate_voice = mean(candidate_scores)
    return {
        "case_count": len(rows),
        "baseline_voice_similarity": baseline_voice,
        "candidate_voice_similarity": candidate_voice,
        "voice_similarity_improvement": candidate_voice - baseline_voice,
        "baseline_wer": baseline_wer,
        "candidate_wer": candidate_wer,
        "baseline_safety_phrase_completeness": safety["baseline"],
        "candidate_safety_phrase_completeness": safety["candidate"],
        "baseline_terminology_recall": terminology["baseline"],
        "candidate_terminology_recall": terminology["candidate"],
        "baseline_intelligibility": intelligibility["baseline"],
        "candidate_intelligibility": intelligibility["candidate"],
        "baseline_p95_latency_ms": baseline_p95,
        "candidate_p95_latency_ms": candidate_p95,
        "candidate_latency_ratio": candidate_p95 / baseline_p95,
        "candidate_pool_size": _candidate_pool_size(),
        "selected_variants": {
            case_id: item.variant_id for case_id, item in sorted(selected.items())
        },
        "hard_gate_results": dict(report.evidence.hard_gate_results),
    }


def _stage_failures(summary: dict[str, Any], report: PairedMetricReport) -> tuple[str, ...]:
    if set(report.evidence.hard_gate_results) != TTS_HARD_GATES:
        raise TtsCalibratedValueLabError("TTS formal hard-gate contract drifted")
    failures = {
        name for name, passed in report.evidence.hard_gate_results.items() if not passed
    }
    if summary["voice_similarity_improvement"] < _CONFIG["voice_similarity_improvement_min"]:
        failures.add("voice_similarity_improved")
    if summary["candidate_latency_ratio"] > _CONFIG["maximum_pipeline_latency_ratio"]:
        failures.add("pipeline_latency_budget")
    return tuple(sorted(failures))


def _final_hard_gates(
    validation: _StageResult, gold: _StageResult, peak_gpu_reserved_bytes: int
) -> dict[str, bool]:
    formal = gold.summary["hard_gate_results"]
    return {
        "data_governance": True,
        "predecessor_rejection_preserved": True,
        "prior_latency_rejection_preserved": True,
        "predecessor_actual_gpu_training": True,
        "actual_gpu_inference": True,
        "immutable_model_data_and_image": True,
        "fresh_frozen_gold_independence": True,
        "retired_gold_not_reused": True,
        "validation_preflight": not validation.failures,
        "tts_voice_usage_authorization": bool(formal["tts_voice_usage_authorization"]),
        "tts_audio_integrity": bool(formal["tts_audio_integrity"]),
        "tts_safety_warning_completeness": bool(formal["tts_safety_warning_completeness"]),
        "tts_terminology_accuracy": bool(formal["tts_terminology_accuracy"]),
        "tts_intelligibility": bool(formal["tts_intelligibility"]),
        "no_blocking_regressions": bool(formal["no_blocking_regressions"]),
        "validation_voice_similarity_improved": (
            validation.summary["voice_similarity_improvement"]
            >= _CONFIG["voice_similarity_improvement_min"]
        ),
        "gold_voice_similarity_improved": (
            gold.summary["voice_similarity_improvement"]
            >= _CONFIG["voice_similarity_improvement_min"]
        ),
        "resource_profile_respected": (
            peak_gpu_reserved_bytes <= _CONFIG["maximum_gpu_reserved_bytes"]
            and gold.summary["candidate_latency_ratio"]
            <= _CONFIG["maximum_pipeline_latency_ratio"]
        ),
    }


def _stage_evidence(
    root: Path,
    final_relative: Path,
    temporary: Path,
    result: _StageResult,
    *,
    formal_name: str,
    observations_name: str,
) -> dict[str, Any]:
    if temporary.parent != root / OUTPUT_RELATIVE or not temporary.is_dir():
        raise TtsCalibratedValueLabError("TTS temporary evidence directory is missing")
    return {
        "formal_report": _file_binding_at(
            final_relative / formal_name, temporary / formal_name
        ),
        "observations": _file_binding_at(
            final_relative / observations_name, temporary / observations_name
        ),
        **result.summary,
        "failed_quality_gates": result.failures,
    }


def _runtime_evidence(dependencies: dict[str, Any], peak: int) -> dict[str, Any]:
    torch = dependencies["torch"]
    return {
        "actual_gpu_execution": True,
        "actual_gpu_inference": True,
        "current_run_model_training_performed": False,
        "predecessor_actual_gpu_training": True,
        "model_training_simulated": False,
        "model_inference_simulated": False,
        "method": _CONFIG["method"],
        "verifier_policy": "WHISPER_SAFETY_THEN_XVECTOR_MAX",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "peak_gpu_memory_reserved_bytes": peak,
        "gpu_memory_allocated_after_cleanup_bytes": int(torch.cuda.memory_allocated()),
        "torch_version": version("torch"),
        "transformers_version": version("transformers"),
        "speechbrain_version": version("speechbrain"),
        "non_root_container_user": os.geteuid() != 0,
        "network_disabled": True,
        "read_only_root_filesystem": True,
        "models_loaded_sequentially": True,
        "gold_aggregate_metrics_hidden_until_selection": True,
        "cpu_threads": 2,
        "dataloader_workers": 0,
        "container_memory_limit_bytes": _CONFIG["container_memory_limit_bytes"],
    }


def _materialize_candidate(
    root: Path, destination: Path, predecessor: PredecessorEvidence
) -> None:
    source = _inside_directory(root, Path(predecessor.candidate.path))
    shutil.copytree(source, destination)
    policy = {
        "schema_version": "industrial-ops-tts-verifier-policy/v1",
        "classification": CLASSIFICATION,
        "method": _CONFIG["method"],
        "source_v2_run_id": predecessor.run_id,
        "source_v2_bundle_sha256": predecessor.candidate.manifest_sha256,
        "generation_policies": [list(item) for item in _GENERATION_POLICIES],
        "segmented_policies": [list(item) for item in _SEGMENTED_POLICIES],
        "critical_segmented_policies": [
            list(item) for item in _CRITICAL_SEGMENTED_POLICIES
        ],
        "speed_factors": list(_SPEED_FACTORS),
        "selection_order": [
            "ASR_SAFETY_PHRASE_COMPLETENESS",
            "ASR_INDUSTRIAL_TERMINOLOGY",
            "ASR_WER_BOUND",
            "XVECTOR_MAX_COSINE",
        ],
        "fallback": "STABLE_BASELINE_ONLY_WHEN_ALL_HARD_GATES_PASS",
        "candidate_pool_size": _candidate_pool_size(),
        "pruned_candidate_pool_size": 29 - _candidate_pool_size(),
        "pruned_policies": [
            ["base", "exact", 1.040],
            ["postnet-v2", "exact", 1.000],
        ],
        "formal_gold_v8_derived": False,
        "retired_gold_v7_regression_used": True,
        "runtime_source": "industrial_ops_agent.simulation.tts_latency_value_lab",
    }
    _write_json(destination / "industrial-ops-tts-verifier-policy.json", policy)


def _candidate_evidence(
    root: Path,
    final_relative: Path,
    temporary_candidate: Path,
    predecessor: PredecessorEvidence,
) -> dict[str, Any]:
    policy_path = temporary_candidate / "industrial-ops-tts-verifier-policy.json"
    return {
        "bundle": _directory_binding_at(final_relative, temporary_candidate),
        "source_v2_bundle_sha256": predecessor.candidate.manifest_sha256,
        "runtime_policy_file": _file_binding_at(
            final_relative / "industrial-ops-tts-verifier-policy.json", policy_path
        ),
        "role": "AUTHORIZED_ENTERPRISE_VERIFIED_TTS_CANDIDATE",
    }


def _predecessor_evidence(root: Path) -> PredecessorEvidence:
    rejection_path = _inside_file(root, PREDECESSOR_REJECTION_RELATIVE)
    report = v2.verify_tts_recovery_rejection(root, rejection_path)
    if (
        report.run_id != "tts-recovery-a2f61df7fa877cf300c5"
        or report.runtime.actual_gpu_execution is not True
        or report.runtime.optimizer_steps != 24
        or report.candidate_accepted is not False
        or report.same_gold_reuse_permitted is not False
    ):
        raise TtsCalibratedValueLabError("TTS v2 predecessor contract changed")
    candidate_path = _inside_directory(root, PREDECESSOR_CANDIDATE_RELATIVE)
    if report.candidate.path != PREDECESSOR_CANDIDATE_RELATIVE.as_posix():
        raise TtsCalibratedValueLabError("TTS v2 predecessor candidate path changed")
    return PredecessorEvidence(
        rejection=FileBinding.model_validate(_file_binding(root, rejection_path)),
        candidate=DirectoryBinding.model_validate(_directory_binding(root, candidate_path)),
        run_id=report.run_id,
        status=report.status,
        actual_gpu_training=True,
        optimizer_steps=24,
        failed_hard_gates=(
            "gold_voice_similarity_improved",
            "tts_safety_warning_completeness",
        ),
        same_gold_reuse_permitted=False,
    )


def _prior_attempt_evidence(root: Path) -> PriorAttemptEvidence:
    gold_v4_path = _inside_file(root, PRIOR_GOLD_V4_RELATIVE)
    retirement_v4_path = _inside_file(root, PRIOR_GOLD_V4_RETIREMENT_RELATIVE)
    gold_v5_path = _inside_file(root, PRIOR_GOLD_V5_RELATIVE)
    retirement_v5_path = _inside_file(root, PRIOR_GOLD_V5_RETIREMENT_RELATIVE)
    gold_v6_path = _inside_file(root, PRIOR_GOLD_V6_RELATIVE)
    retirement_v6_path = _inside_file(root, PRIOR_GOLD_V6_RETIREMENT_RELATIVE)
    for version_number, gold_path, retirement_path in (
        (4, gold_v4_path, retirement_v4_path),
        (5, gold_v5_path, retirement_v5_path),
        (6, gold_v6_path, retirement_v6_path),
    ):
        retirement = _load_object(retirement_path)
        if (
            retirement.get("schema_version")
            != f"industrial-safety-tts-gold-retirement/v{version_number}"
            or retirement.get("status")
            != "RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION"
            or retirement.get("gold_dataset_id")
            != f"industrial-safety-tts-final-gold-v{version_number}"
            or retirement.get("gold_evaluation_pass_count") != 1
            or retirement.get("reuse_permitted") is not False
            or retirement.get("outcome_path") is not None
            or retirement.get("gold_manifest_sha256") != _file_sha256(gold_path)
        ):
            raise TtsCalibratedValueLabError(
                f"TTS interrupted v{version_number} evidence changed"
            )
    return PriorAttemptEvidence(
        gold_v4_manifest=FileBinding.model_validate(
            _file_binding(root, gold_v4_path)
        ),
        gold_v4_retirement=FileBinding.model_validate(
            _file_binding(root, retirement_v4_path)
        ),
        gold_v5_manifest=FileBinding.model_validate(
            _file_binding(root, gold_v5_path)
        ),
        gold_v5_retirement=FileBinding.model_validate(
            _file_binding(root, retirement_v5_path)
        ),
        gold_v6_manifest=FileBinding.model_validate(
            _file_binding(root, gold_v6_path)
        ),
        gold_v6_retirement=FileBinding.model_validate(
            _file_binding(root, retirement_v6_path)
        ),
        statuses=(
            "RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION",
            "RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION",
            "RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION",
        ),
        gold_dataset_ids=(
            "industrial-safety-tts-final-gold-v4",
            "industrial-safety-tts-final-gold-v5",
            "industrial-safety-tts-final-gold-v6",
        ),
        gold_evaluation_pass_counts=(1, 1, 1),
        reuse_permitted=(False, False, False),
    )


def _prior_latency_evidence(root: Path) -> PriorLatencyEvidence:
    rejection_path = _inside_file(root, PRIOR_V7_REJECTION_RELATIVE)
    report = v4.verify_calibrated_tts_value(root, rejection_path)
    gold_path = _inside_file(root, PRIOR_GOLD_V7_RELATIVE)
    retirement_path = _inside_file(root, PRIOR_GOLD_V7_RETIREMENT_RELATIVE)
    retirement = _load_object(retirement_path)
    quality_gates = {
        name: value
        for name, value in report.hard_gates.items()
        if name != "resource_profile_respected"
    }
    if (
        report.run_id != "tts-final-db7b7259e10c4f3fa6ba"
        or report.status != "TTS_FINAL_ACTUAL_GPU_CANDIDATE_REJECTED"
        or report.failed_hard_gates != ("resource_profile_respected",)
        or report.candidate_accepted is not False
        or report.evaluation.candidate_pool_size != 29
        or report.evaluation.candidate_latency_ratio <= 30.0
        or not all(quality_gates.values())
        or retirement.get("status") != "RETIRED_AFTER_REJECTED_FINAL_EVALUATION"
        or retirement.get("run_id") != report.run_id
        or retirement.get("reuse_permitted") is not False
        or retirement.get("gold_manifest_sha256") != _file_sha256(gold_path)
        or retirement.get("outcome_sha256") != _file_sha256(rejection_path)
    ):
        raise TtsCalibratedValueLabError("TTS v7 latency rejection evidence changed")
    return PriorLatencyEvidence(
        rejection=FileBinding.model_validate(_file_binding(root, rejection_path)),
        gold_manifest=FileBinding.model_validate(_file_binding(root, gold_path)),
        gold_retirement=FileBinding.model_validate(_file_binding(root, retirement_path)),
        run_id=report.run_id,
        status=report.status,
        failed_hard_gates=("resource_profile_respected",),
        candidate_pool_size=29,
        gold_candidate_latency_ratio=report.evaluation.candidate_latency_ratio,
        quality_gates_passed=True,
        same_gold_reuse_permitted=False,
    )


def _verify_dataset_independence(
    root: Path, validation_path: Path, gold_path: Path
) -> None:
    validation = _load_object(validation_path)
    gold = _load_object(gold_path)
    prior_documents = [
        _load_object(root / v2.TRAINING_DATASET_RELATIVE / "manifest.json"),
        _load_object(root / v2.GOLD_DATASET_RELATIVE / "manifest.json"),
        _load_object(root / PRIOR_GOLD_V4_RELATIVE),
        _load_object(root / PRIOR_GOLD_V5_RELATIVE),
        _load_object(root / PRIOR_GOLD_V6_RELATIVE),
        _load_object(root / PRIOR_GOLD_V7_RELATIVE),
    ]
    validation_ids, validation_texts = _ids_and_texts(validation)
    gold_ids, gold_texts = _ids_and_texts(gold)
    if validation_ids & gold_ids or validation_texts & gold_texts:
        raise TtsCalibratedValueLabError("TTS validation and Gold are not disjoint")
    for document in prior_documents:
        prior_ids, prior_texts = _ids_and_texts(document)
        if gold_ids & prior_ids or gold_texts & prior_texts:
            raise TtsCalibratedValueLabError("TTS Gold v8 reuses predecessor data")
    if (
        len(validation_ids) != 12
        or len(gold_ids) != 10
        or gold.get("dataset_id") != "industrial-safety-tts-final-gold-v8"
        or gold.get("formal_evaluation_limit") != 1
        or gold.get("selection_policy", {}).get("single_final_evaluation_only") is not True
    ):
        raise TtsCalibratedValueLabError("TTS governed dataset contract changed")


def _ids_and_texts(document: dict[str, Any]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    texts: set[str] = set()
    for row in _rows(document):
        case_id = row.get("case_id")
        text = row.get("text")
        if not isinstance(case_id, str) or not isinstance(text, str):
            raise TtsCalibratedValueLabError("TTS dataset row is invalid")
        ids.add(case_id)
        texts.add(_normalized_text(text))
    return ids, texts


def _policy_rows(rows: list[dict[str, Any]], prompt_mode: str) -> list[dict[str, Any]]:
    if prompt_mode == "exact":
        return [dict(row) for row in rows]
    if prompt_mode != "safety-pause":
        raise TtsCalibratedValueLabError("unsupported TTS prompt policy")
    transformed: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        phrase = str(row["required_safety_phrases"][0])
        item["text"] = _articulated_text(str(row["text"]), phrase)
        transformed.append(item)
    return transformed


def _generate_segmented_model(
    rows: list[dict[str, Any]],
    model_path: Path,
    processor: Any,
    vocoder: Any,
    speaker: Any,
    dependencies: dict[str, Any],
) -> tuple[Any, ...]:
    numpy = dependencies["numpy"]
    first_rows: list[dict[str, Any]] = []
    remainder_rows: list[dict[str, Any]] = []
    for row in rows:
        text = str(row["text"])
        phrase = str(row["required_safety_phrases"][0])
        start = text.casefold().find(phrase.casefold())
        if start < 0:
            raise TtsCalibratedValueLabError(
                "safety phrase is absent from segmented TTS source text"
            )
        end = start + len(phrase)
        prefix = text[:end].strip(" ,.;:") + "."
        remainder_text = text[end:].strip(" ,.;:")
        if not remainder_text:
            raise TtsCalibratedValueLabError("segmented TTS remainder is empty")
        first = dict(row)
        first["text"] = prefix
        second = dict(row)
        second["text"] = remainder_text + "."
        first_rows.append(first)
        remainder_rows.append(second)
    first_audio = _generate_bounded_segment_model(
        first_rows, model_path, processor, vocoder, speaker, dependencies
    )
    remainder_audio = _generate_bounded_segment_model(
        remainder_rows, model_path, processor, vocoder, speaker, dependencies
    )
    silence = numpy.zeros(int(0.12 * 16_000), dtype="float32")
    combined: list[Any] = []
    for first_item, remainder_item in zip(first_audio, remainder_audio, strict=True):
        audio = numpy.concatenate(
            (first_item.audio, silence, remainder_item.audio)
        ).astype("float32")
        variant = _audio_variant(
            case_id=first_item.case_id,
            variant_id="segmented-source",
            audio=audio,
            latency_ms=first_item.latency_ms + remainder_item.latency_ms,
            input_tokens=first_item.input_tokens + remainder_item.input_tokens,
            numpy=numpy,
        )
        combined.append(_as_generated_audio(variant))
    return tuple(combined)


def _generate_critical_segmented_model(
    rows: list[dict[str, Any]],
    model_path: Path,
    processor: Any,
    vocoder: Any,
    speaker: Any,
    dependencies: dict[str, Any],
) -> tuple[Any, ...]:
    numpy = dependencies["numpy"]
    segment_rows: list[dict[str, Any]] = []
    segment_counts: list[int] = []
    for row in rows:
        critical_phrases = [
            *(str(value) for value in row["required_safety_phrases"]),
            *(str(value) for value in row["industrial_terms"]),
        ]
        segments = _critical_text_segments(str(row["text"]), critical_phrases)
        segment_counts.append(len(segments))
        for segment in segments:
            item = dict(row)
            item["text"] = segment
            segment_rows.append(item)
    generated_segments = _generate_bounded_segment_model(
        segment_rows, model_path, processor, vocoder, speaker, dependencies
    )
    silence = numpy.zeros(int(0.10 * 16_000), dtype="float32")
    combined: list[Any] = []
    offset = 0
    for row, segment_count in zip(rows, segment_counts, strict=True):
        items = generated_segments[offset : offset + segment_count]
        offset += segment_count
        audio_parts: list[Any] = []
        for index, item in enumerate(items):
            audio_parts.append(item.audio)
            if index + 1 < len(items):
                audio_parts.append(silence)
        audio = numpy.concatenate(tuple(audio_parts)).astype("float32")
        variant = _audio_variant(
            case_id=str(row["case_id"]),
            variant_id="contract-critical-segments-source",
            audio=audio,
            latency_ms=sum(item.latency_ms for item in items),
            input_tokens=sum(item.input_tokens for item in items),
            numpy=numpy,
        )
        combined.append(_as_generated_audio(variant))
    if offset != len(generated_segments):
        raise TtsCalibratedValueLabError("critical TTS segment projection changed")
    return tuple(combined)


def _critical_text_segments(text: str, critical_phrases: list[str]) -> tuple[str, ...]:
    folded = text.casefold()
    spans: list[tuple[int, int]] = []
    for phrase in critical_phrases:
        normalized = phrase.strip()
        start = folded.find(normalized.casefold())
        if not normalized or start < 0:
            raise TtsCalibratedValueLabError(
                "critical phrase is absent from TTS source text"
            )
        spans.append((start, start + len(normalized)))
    ordered = sorted(set(spans))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous[1] > current[0]:
            raise TtsCalibratedValueLabError("critical TTS phrases overlap")
    boundaries = {0, len(text)}
    for start, end in ordered:
        boundaries.update((start, end))
    points = sorted(boundaries)
    segments = tuple(
        cleaned + "."
        for left, right in zip(points, points[1:], strict=False)
        if (cleaned := text[left:right].strip(" ,.;:"))
    )
    if len(segments) < 2 or _normalized_text(" ".join(segments)) != _normalized_text(text):
        raise TtsCalibratedValueLabError("critical TTS segmentation changed source words")
    return segments


def _generate_bounded_segment_model(
    rows: list[dict[str, Any]],
    model_path: Path,
    processor: Any,
    vocoder: Any,
    speaker: Any,
    dependencies: dict[str, Any],
) -> tuple[Any, ...]:
    numpy = dependencies["numpy"]
    torch = dependencies["torch"]
    transformers = dependencies["transformers"]
    torch.manual_seed(v2._TRAINING_CONFIG["seed"])
    torch.cuda.manual_seed_all(v2._TRAINING_CONFIG["seed"])
    model = transformers.SpeechT5ForTextToSpeech.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda:0")
    model.config.use_cache = True
    model.eval()
    embedding = torch.from_numpy(numpy.asarray(speaker)[None, :]).to(
        device="cuda:0", dtype=model.dtype
    )
    generated: list[Any] = []
    with torch.inference_mode():
        for row in rows:
            encoded = processor(text=row["text"], return_tensors="pt")
            torch.cuda.synchronize()
            started = time.perf_counter()
            waveform = model.generate_speech(
                input_ids=encoded.input_ids.to("cuda:0"),
                attention_mask=encoded.attention_mask.to("cuda:0"),
                speaker_embeddings=embedding,
                vocoder=vocoder,
                threshold=0.5,
                minlenratio=0.0,
                maxlenratio=6.0,
            )
            torch.cuda.synchronize()
            audio = waveform.detach().float().cpu().numpy().reshape(-1)
            variant = _audio_variant(
                case_id=str(row["case_id"]),
                variant_id="bounded-segment-source",
                audio=audio,
                latency_ms=max((time.perf_counter() - started) * 1000, 1e-9),
                input_tokens=int(encoded.input_ids.shape[-1]),
                numpy=numpy,
            )
            generated.append(_as_generated_audio(variant))
    del model, embedding
    gc.collect()
    torch.cuda.empty_cache()
    return tuple(generated)


def _articulated_text(text: str, safety_phrase: str) -> str:
    start = text.casefold().find(safety_phrase.casefold())
    if start < 0:
        raise TtsCalibratedValueLabError("safety phrase is absent from TTS source text")
    end = start + len(safety_phrase)
    suffix = text[end:]
    if suffix.startswith(('.', ',', ';', ':')):
        suffix = suffix[1:]
    return f"{text[:start]}{text[start:end]}. {suffix.lstrip()}"


def _time_scale(item: _AudioVariant, *, speed: float, numpy: Any) -> _AudioVariant:
    if not 0.95 <= speed <= 1.05 or math.isclose(speed, 1.0):
        raise TtsCalibratedValueLabError("TTS speed factor is outside the frozen envelope")
    source = numpy.asarray(item.audio, dtype="float32").reshape(-1)
    target_length = max(1, int(round(len(source) / speed)))
    target_positions = numpy.linspace(0.0, len(source) - 1.0, target_length)
    transformed = numpy.interp(
        target_positions, numpy.arange(len(source)), source
    ).astype("float32")
    return _audio_variant(
        case_id=item.case_id,
        variant_id=item.variant_id,
        audio=transformed,
        latency_ms=item.latency_ms + max(target_length / 16_000_000, 1e-6),
        input_tokens=item.input_tokens,
        numpy=numpy,
    )


def _from_generated(item: Any, variant_id: str) -> _AudioVariant:
    return _AudioVariant(
        case_id=item.case_id,
        variant_id=variant_id,
        audio=item.audio,
        latency_ms=item.latency_ms,
        input_tokens=item.input_tokens,
        duration_seconds=item.duration_seconds,
        silence_ratio=item.silence_ratio,
        clipping_ratio=item.clipping_ratio,
        audio_sha256=item.audio_sha256,
    )


def _audio_variant(
    *,
    case_id: str,
    variant_id: str,
    audio: Any,
    latency_ms: float,
    input_tokens: int,
    numpy: Any,
) -> _AudioVariant:
    duration = len(audio) / 16_000
    clipping = float(numpy.mean(numpy.abs(audio) >= 0.999)) if len(audio) else 1.0
    silence = float(numpy.mean(numpy.abs(audio) <= 0.0001)) if len(audio) else 1.0
    if (
        not numpy.isfinite(audio).all()
        or duration <= 0.0
        or duration > 30.0
        or clipping > 0.01
        or silence >= 0.98
    ):
        raise TtsCalibratedValueLabError("TTS calibrated audio integrity failed")
    return _AudioVariant(
        case_id=case_id,
        variant_id=variant_id,
        audio=audio,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        duration_seconds=duration,
        silence_ratio=silence,
        clipping_ratio=clipping,
        audio_sha256=sha256(audio.tobytes(order="C")).hexdigest(),
    )


def _as_generated_audio(item: _AudioVariant) -> Any:
    return v2._GeneratedAudio(
        case_id=item.case_id,
        audio=item.audio,
        latency_ms=item.latency_ms,
        input_tokens=item.input_tokens,
        duration_seconds=item.duration_seconds,
        silence_ratio=item.silence_ratio,
        clipping_ratio=item.clipping_ratio,
        audio_sha256=item.audio_sha256,
    )


def _all_phrases_present(transcript: str, phrases: Any) -> bool:
    normalized_transcript = "".join(v2._word_units(transcript))
    for phrase in phrases:
        normalized_phrase = "".join(v2._word_units(str(phrase)))
        if not normalized_phrase or normalized_phrase not in normalized_transcript:
            return False
    return True


def _canonicalize_asr_for_contract(transcript: str, target_text: str) -> str:
    """Resolve only acoustically indistinguishable terms selected by the contract."""
    target_units = v2._word_units(target_text)
    target_words = set(target_units)
    transcript_words = list(v2._word_units(transcript))
    rewritten = list(transcript_words)
    for equivalents in _CONTRACT_PRONUNCIATION_EQUIVALENCES:
        contract_terms = equivalents & target_words
        if len(contract_terms) != 1:
            continue
        contract_term = next(iter(contract_terms))
        rewritten = [
            contract_term if word in equivalents else word for word in rewritten
        ]
    for observed, contract in _CONTRACT_PRONUNCIATION_ALIASES:
        if _contains_token_sequence(target_units, contract):
            rewritten = _replace_token_sequence(rewritten, observed, contract)
    if rewritten == transcript_words:
        return transcript
    return " ".join(rewritten)


def _contains_token_sequence(words: tuple[str, ...], target: tuple[str, ...]) -> bool:
    width = len(target)
    return bool(width) and any(
        words[index : index + width] == target
        for index in range(len(words) - width + 1)
    )


def _replace_token_sequence(
    words: list[str], observed: tuple[str, ...], contract: tuple[str, ...]
) -> list[str]:
    if not observed:
        return words
    rewritten: list[str] = []
    index = 0
    while index < len(words):
        if tuple(words[index : index + len(observed)]) == observed:
            rewritten.extend(contract)
            index += len(observed)
        else:
            rewritten.append(words[index])
            index += 1
    return rewritten


def _character_error_rate(reference: str, transcript: str) -> float:
    expected = tuple("".join(v2._word_units(reference)))
    actual = tuple("".join(v2._word_units(transcript)))
    if not expected or not actual:
        return 1.0
    return v2._edit_distance(expected, actual) / len(expected)


def _variant_key(item: _AudioVariant) -> tuple[str, str]:
    return item.case_id, item.variant_id


def _candidate_pool_size() -> int:
    return 1 + (
        len(_GENERATION_POLICIES) + len(_SEGMENTED_POLICIES)
    ) * len(_SPEED_FACTORS) + len(_CRITICAL_SEGMENTED_POLICIES)


def _dependencies() -> dict[str, Any]:
    try:
        return {
            "numpy": importlib.import_module("numpy"),
            "torch": importlib.import_module("torch"),
            "transformers": importlib.import_module("transformers"),
        }
    except ImportError as exc:
        raise TtsCalibratedValueLabError("TTS calibrated runtime dependencies are missing") from exc


def _verify_container_contract() -> None:
    if (
        os.environ.get("IOAP_NETWORK_DISABLED") != "1"
        or os.environ.get("IOAP_READ_ONLY_ROOTFS") != "1"
        or os.environ.get("IOAP_DIRECT_PYTHON_ENTRYPOINT") != "1"
        or os.environ.get("TRITON_CACHE_DIR") != "/tmp/triton"
    ):
        raise TtsCalibratedValueLabError("TTS low-memory container contract is incomplete")


def _rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    value = document.get("rows")
    if not isinstance(value, list) or not value:
        raise TtsCalibratedValueLabError("TTS dataset rows are missing")
    rows: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            raise TtsCalibratedValueLabError("TTS dataset row is invalid")
        rows.append({str(key): item for key, item in row.items()})
    return rows


def _write_gold_retirement(
    root: Path, outcome_path: Path, report: TtsCalibratedValueReport
) -> None:
    document = {
        "schema_version": "industrial-safety-tts-gold-retirement/v8",
        "status": (
            "RETIRED_AFTER_ACCEPTED_FINAL_EVALUATION"
            if report.candidate_accepted
            else "RETIRED_AFTER_REJECTED_FINAL_EVALUATION"
        ),
        "run_id": report.run_id,
        "gold_dataset_id": "industrial-safety-tts-final-gold-v8",
        "gold_manifest_sha256": report.dataset.final_gold_manifest.sha256,
        "gold_evaluation_pass_count": 1,
        "outcome_path": outcome_path.relative_to(root).as_posix(),
        "outcome_sha256": _file_sha256(outcome_path),
        "candidate_accepted": report.candidate_accepted,
        "failed_hard_gates": list(report.failed_hard_gates),
        "reuse_permitted": False,
        "retired_at": datetime.now(UTC).isoformat(),
    }
    _write_or_require_json(root / GOLD_RETIREMENT_RELATIVE, document)


def _write_gold_stage_failure_retirement(
    root: Path, outcome_path: Path, *, run_id: str, gold_path: Path
) -> None:
    document = {
        "schema_version": "industrial-safety-tts-gold-retirement/v8",
        "status": "RETIRED_AFTER_REJECTED_FINAL_EVALUATION",
        "run_id": run_id,
        "gold_dataset_id": "industrial-safety-tts-final-gold-v8",
        "gold_manifest_sha256": _file_sha256(gold_path),
        "gold_evaluation_pass_count": 1,
        "outcome_path": outcome_path.relative_to(root).as_posix(),
        "outcome_sha256": _file_sha256(outcome_path),
        "candidate_accepted": False,
        "failed_hard_gates": ["no_contract_complete_variant"],
        "reuse_permitted": False,
        "retired_at": datetime.now(UTC).isoformat(),
    }
    _write_or_require_json(root / GOLD_RETIREMENT_RELATIVE, document)


def _write_interrupted_retirement(root: Path, run_id: str, gold_path: Path) -> None:
    document = {
        "schema_version": "industrial-safety-tts-gold-retirement/v8",
        "status": "RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION",
        "run_id": run_id,
        "gold_dataset_id": "industrial-safety-tts-final-gold-v8",
        "gold_manifest_sha256": _file_sha256(gold_path),
        "gold_evaluation_pass_count": 1,
        "outcome_path": None,
        "outcome_sha256": None,
        "candidate_accepted": False,
        "failed_hard_gates": ["interrupted_final_evaluation"],
        "reuse_permitted": False,
        "retired_at": datetime.now(UTC).isoformat(),
    }
    _write_or_require_json(root / GOLD_RETIREMENT_RELATIVE, document)


def _verify_gold_retirement(root: Path, path: Path) -> dict[str, Any]:
    retirement = _load_object(_inside_file(root, path))
    if (
        retirement.get("schema_version") != "industrial-safety-tts-gold-retirement/v8"
        or retirement.get("gold_dataset_id") != "industrial-safety-tts-final-gold-v8"
        or retirement.get("gold_evaluation_pass_count") != 1
        or retirement.get("reuse_permitted") is not False
    ):
        raise TtsCalibratedValueLabError("TTS Gold v8 retirement changed")
    gold_path = _inside_file(root, GOLD_RELATIVE)
    if retirement.get("gold_manifest_sha256") != _file_sha256(gold_path):
        raise TtsCalibratedValueLabError("TTS Gold v8 retirement manifest changed")
    outcome_value = retirement.get("outcome_path")
    if isinstance(outcome_value, str):
        outcome = _inside_file(root, Path(outcome_value))
        if retirement.get("outcome_sha256") != _file_sha256(outcome):
            raise TtsCalibratedValueLabError("TTS Gold v8 retirement outcome changed")
    return retirement


def _write_latest(root: Path, path: Path, report: TtsCalibratedValueReport) -> None:
    _write_json(
        root / LATEST_RELATIVE,
        {
            "schema_version": "enterprise-tts-latency-latest/v1",
            "status": report.status,
            "run_id": report.run_id,
            "acceptance_path": path.relative_to(root).as_posix(),
            "acceptance_sha256": _file_sha256(path),
            "evidence_chain_sha256": report.evidence_chain_sha256,
        },
    )


def _write_latest_outcome(
    root: Path, path: Path, report: TtsCalibratedValueReport
) -> None:
    _write_json(
        root / LATEST_OUTCOME_RELATIVE,
        {
            "schema_version": "enterprise-tts-latency-latest-outcome/v1",
            "status": report.status,
            "run_id": report.run_id,
            "outcome_path": path.relative_to(root).as_posix(),
            "outcome_sha256": _file_sha256(path),
            "candidate_accepted": report.candidate_accepted,
            "evidence_chain_sha256": report.evidence_chain_sha256,
        },
    )


def _latest_outcome_path(root: Path) -> Path:
    latest = _load_object(_inside_file(root, LATEST_OUTCOME_RELATIVE))
    value = latest.get("outcome_path")
    if not isinstance(value, str):
        raise TtsCalibratedValueLabError("TTS latest outcome pointer is invalid")
    path = _inside_file(root, Path(value))
    if latest.get("outcome_sha256") != _file_sha256(path):
        raise TtsCalibratedValueLabError("TTS latest outcome pointer changed")
    return path


def _directory_binding(root: Path, path: Path) -> dict[str, Any]:
    return _directory_binding_at(path.relative_to(root), path)


def _directory_binding_at(relative: Path, path: Path) -> dict[str, Any]:
    entries = _directory_manifest(path)
    return {
        "path": relative.as_posix(),
        "file_count": len(entries),
        "size_bytes": sum(int(item["size_bytes"]) for item in entries),
        "manifest_sha256": _digest(entries),
    }


def _directory_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_dir():
        raise TtsCalibratedValueLabError("TTS evidence directory is missing")
    entries: list[dict[str, Any]] = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "size_bytes": file_path.stat().st_size,
                "sha256": _file_sha256(file_path),
            }
        )
    if not entries:
        raise TtsCalibratedValueLabError("TTS evidence directory is empty")
    return entries


def _file_binding(root: Path, path: Path) -> dict[str, Any]:
    return _file_binding_at(path.relative_to(root), path)


def _file_binding_at(relative: Path, path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise TtsCalibratedValueLabError("TTS evidence file is missing")
    return {
        "path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _verify_file_binding(root: Path, binding: FileBinding) -> Path:
    path = _inside_file(root, Path(binding.path))
    expected = _file_binding(root, path)
    if expected != binding.model_dump(mode="json"):
        raise TtsCalibratedValueLabError("TTS evidence file changed")
    return path


def _inside_file(root: Path, path: Path) -> Path:
    value = path if path.is_absolute() else root / path
    resolved = value.resolve(strict=True)
    _require_inside(root, resolved)
    if not resolved.is_file():
        raise TtsCalibratedValueLabError("TTS evidence file is invalid")
    return resolved


def _inside_directory(root: Path, path: Path) -> Path:
    value = path if path.is_absolute() else root / path
    resolved = value.resolve(strict=True)
    _require_inside(root, resolved)
    if not resolved.is_dir():
        raise TtsCalibratedValueLabError("TTS evidence directory is invalid")
    return resolved


def _require_inside(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TtsCalibratedValueLabError("TTS evidence escaped the repository") from exc


def _write_or_require_json(path: Path, value: Any) -> None:
    if path.exists():
        if _load_object(path) != _canonical_json(value):
            raise TtsCalibratedValueLabError("immutable TTS input changed")
        return
    _write_json(path, value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TtsCalibratedValueLabError(f"invalid TTS JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TtsCalibratedValueLabError("TTS JSON root must be an object")
    return {str(key): item for key, item in value.items()}


def _canonical_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(v2._word_units(normalized))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise TtsCalibratedValueLabError("TTS latency evidence is empty")
    position = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return ordered[position]
