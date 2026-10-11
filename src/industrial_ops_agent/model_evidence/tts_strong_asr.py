"""Read-only verification of retained TTS experiment evidence; no training or GPU execution."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.model_evidence import tts_files
from industrial_ops_agent.model_evidence import tts_final as v4
from industrial_ops_agent.model_evidence import tts_latency as v5
from industrial_ops_agent.model_evidence import tts_recovery as v2
from industrial_ops_agent.model_evidence.report_files import file_sha256
from industrial_ops_agent.model_evidence.retired_sources import retired_source_binding
from industrial_ops_agent.model_evidence.tts_contracts import (
    TTS_HARD_GATES,
    DirectoryBinding,
    FileBinding,
    PredecessorEvidence,
    PriorLatencyEvidence,
)

_LEGACY_SOURCE_PATH = "src/industrial_ops_agent/simulation/tts_strong_asr_value_lab.py"

SCHEMA_VERSION: Literal["enterprise-tts-strong-asr-value-lab/v8"] = (
    "enterprise-tts-strong-asr-value-lab/v8"
)


CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"


CALIBRATED_OUTPUT_RELATIVE = Path("artifacts/m7-tts-calibrated-value-lab")


FINAL_OUTPUT_RELATIVE = Path("artifacts/m7-tts-final-value-lab")


LATENCY_OUTPUT_RELATIVE = Path("artifacts/m7-tts-latency-value-lab")


OUTPUT_RELATIVE = Path("artifacts/m7-tts-strong-asr-value-lab")


VALIDATION_RELATIVE = OUTPUT_RELATIVE / "inputs/industrial-safety-tts-validation-v7.json"


PRIOR_GOLD_V4_RELATIVE = (
    CALIBRATED_OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v4/manifest.json"
)


PRIOR_GOLD_V4_RETIREMENT_RELATIVE = CALIBRATED_OUTPUT_RELATIVE / "gold-v4-retirement.json"


PRIOR_GOLD_V5_RELATIVE = (
    CALIBRATED_OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v5/manifest.json"
)


PRIOR_GOLD_V5_RETIREMENT_RELATIVE = CALIBRATED_OUTPUT_RELATIVE / "gold-v5-retirement.json"


PRIOR_GOLD_V6_RELATIVE = (
    CALIBRATED_OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v6/manifest.json"
)


PRIOR_GOLD_V6_RETIREMENT_RELATIVE = CALIBRATED_OUTPUT_RELATIVE / "gold-v6-retirement.json"


PRIOR_GOLD_V7_RELATIVE = (
    FINAL_OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v7/manifest.json"
)


PRIOR_GOLD_V7_RETIREMENT_RELATIVE = FINAL_OUTPUT_RELATIVE / "gold-v7-retirement.json"


PRIOR_V7_REJECTION_RELATIVE = (
    FINAL_OUTPUT_RELATIVE / "rejections/tts-final-db7b7259e10c4f3fa6ba/rejection.json"
)


PRIOR_GOLD_V8_RELATIVE = (
    LATENCY_OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v8/manifest.json"
)


PRIOR_GOLD_V8_RETIREMENT_RELATIVE = LATENCY_OUTPUT_RELATIVE / "gold-v8-retirement.json"


PRIOR_V8_FAILURE_RELATIVE = (
    LATENCY_OUTPUT_RELATIVE / "rejections/tts-latency-f1c6f42b8bc3f3fa5f34/gold-stage-failure.json"
)


PRIOR_V8_VALIDATION_RELATIVE = (
    LATENCY_OUTPUT_RELATIVE / "development/tts-latency-f1c6f42b8bc3f3fa5f34-validation-probe.json"
)


GOLD_RELATIVE = OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v9/manifest.json"


LATEST_OUTCOME_RELATIVE = OUTPUT_RELATIVE / "latest-outcome.json"


ASR_MODEL_ID: Literal["openai/whisper-small.en"] = "openai/whisper-small.en"


ASR_REVISION: Literal["e8727524f962ee844a7319d92be39ac1bd25655a"] = (
    "e8727524f962ee844a7319d92be39ac1bd25655a"
)


ASR_RELATIVE = Path("artifacts/m7-tts-asr-value-lab/inputs/whisper-small-en-e8727524")


PREDECESSOR_REJECTION_RELATIVE = (
    v2.OUTPUT_RELATIVE / "rejections/tts-recovery-a2f61df7fa877cf300c5/rejection.json"
)


PREDECESSOR_CANDIDATE_RELATIVE = (
    v2.OUTPUT_RELATIVE / "rejections/tts-recovery-a2f61df7fa877cf300c5/candidate"
)


IMAGE: Literal["industrial-ops/tts-xvector-runtime:speechbrain-1.0.3"] = v2.IMAGE


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


_CONTRACT_PRONUNCIATION_EQUIVALENCES: tuple[frozenset[str], ...] = (frozenset({"brake", "break"}),)


_CONTRACT_PRONUNCIATION_ALIASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("where",), ("wear",)),
    (("we", "re"), ("wear",)),
)


_CONFIG: dict[str, Any] = {
    "method": "SPEECHT5_PRUNED_CONTRACT_SEGMENTED_WHISPER_SMALL_SELECTION",
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
    "prior_asr_failure_preserved",
    "predecessor_actual_gpu_training",
    "actual_gpu_inference",
    "immutable_model_data_and_image",
    "strong_asr_snapshot_immutable",
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
    """The TTS strong-ASR experiment or its immutable evidence is invalid."""


_files = tts_files.TtsEvidenceFiles(TtsCalibratedValueLabError, v2._word_units)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AsrModelEvidence(_ClosedModel):
    bundle: DirectoryBinding
    model_id: Literal["openai/whisper-small.en"] = ASR_MODEL_ID
    revision: Literal["e8727524f962ee844a7319d92be39ac1bd25655a"] = ASR_REVISION
    model_format: Literal["SAFETENSORS"] = "SAFETENSORS"
    local_files_only: Literal[True] = True


class DatasetEvidence(_ClosedModel):
    validation_manifest: FileBinding
    final_gold_manifest: FileBinding
    validation_count: Literal[12] = 12
    final_gold_count: Literal[10] = 10
    voice_profile_id: Literal["industrial-safety-en-v1"] = "industrial-safety-en-v1"
    source_type: Literal["PLATFORM_SYNTHETIC"] = "PLATFORM_SYNTHETIC"
    project_enterprise_use_authorized: Literal[True] = True
    enterprise_production_data: Literal[False] = False
    retired_v2_v3_v4_v5_v6_v7_v8_reused_as_formal_gold: Literal[False] = False
    retired_v8_used_as_disclosed_regression: Literal[True] = True
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
    method: Literal["SPEECHT5_PRUNED_CONTRACT_SEGMENTED_WHISPER_SMALL_SELECTION"] = (
        "SPEECHT5_PRUNED_CONTRACT_SEGMENTED_WHISPER_SMALL_SELECTION"
    )
    verifier_policy: Literal["WHISPER_SMALL_SAFETY_THEN_XVECTOR_MAX"] = (
        "WHISPER_SMALL_SAFETY_THEN_XVECTOR_MAX"
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


class PriorAsrEvidence(_ClosedModel):
    gold_stage_failure: FileBinding
    validation_probe: FileBinding
    gold_manifest: FileBinding
    gold_retirement: FileBinding
    run_id: Literal["tts-latency-f1c6f42b8bc3f3fa5f34"]
    status: Literal["TTS_LATENCY_GOLD_NO_SAFE_VARIANT_REJECTED"]
    failed_hard_gates: tuple[Literal["no_contract_complete_variant"]]
    candidate_pool_size: Literal[23] = 23
    weak_asr_model_id: Literal["openai/whisper-tiny"] = "openai/whisper-tiny"
    weak_asr_revision: Literal["c4f62d5fce5d73978a1dbfcac01926312acd1a51"] = (
        "c4f62d5fce5d73978a1dbfcac01926312acd1a51"
    )
    failure_case_id: Literal["tts-latency-v8-001"] = "tts-latency-v8-001"
    validation_quality_gates_passed: Literal[True] = True
    same_gold_reuse_permitted: Literal[False] = False


class TtsCalibratedValueReport(_ClosedModel):
    schema_version: Literal["enterprise-tts-strong-asr-value-lab/v8"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal[
        "TTS_STRONG_ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
        "TTS_STRONG_ASR_ACTUAL_GPU_CANDIDATE_REJECTED",
    ]
    decision: Literal[
        "TTS_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT",
        "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD",
    ]
    run_id: str = Field(pattern=r"^tts-strong-asr-[0-9a-f]{20}$")
    source: FileBinding
    image: Literal["industrial-ops/tts-xvector-runtime:speechbrain-1.0.3"] = IMAGE
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    asr_verifier: AsrModelEvidence
    predecessor: PredecessorEvidence
    prior_attempt: PriorAttemptEvidence
    prior_latency_attempt: PriorLatencyEvidence
    prior_asr_attempt: PriorAsrEvidence
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


_STRONG_ASR_EXTRA_ROWS: tuple[dict[str, Any], ...] = (
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


_RETIRED_GOLD_V8_ROWS: tuple[dict[str, Any], ...] = (
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


_VALIDATION_ROWS: tuple[dict[str, Any], ...] = _RETIRED_GOLD_V8_ROWS + _STRONG_ASR_EXTRA_ROWS


_GOLD_ROWS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "tts-strong-asr-v9-001",
        "text": "Disconnect the main power before opening the electrical panel.",
        "required_safety_phrases": ["disconnect the main power"],
        "industrial_terms": ["electrical panel"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-strong-asr-v9-002",
        "text": "Close the fuel valve before replacing the supply hose.",
        "required_safety_phrases": ["close the fuel valve"],
        "industrial_terms": ["supply hose"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-strong-asr-v9-003",
        "text": "Install the wheel chock before releasing the parking brake.",
        "required_safety_phrases": ["install the wheel chock"],
        "industrial_terms": ["parking brake"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-strong-asr-v9-004",
        "text": "Vent the air pressure before removing the cylinder cap.",
        "required_safety_phrases": ["vent the air pressure"],
        "industrial_terms": ["cylinder cap"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-strong-asr-v9-005",
        "text": "Apply the safety lock before entering the conveyor area.",
        "required_safety_phrases": ["apply the safety lock"],
        "industrial_terms": ["conveyor area"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-strong-asr-v9-006",
        "text": "Lower the lifting platform before inspecting the drive chain.",
        "required_safety_phrases": ["lower the lifting platform"],
        "industrial_terms": ["drive chain"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-strong-asr-v9-007",
        "text": "Wear insulated gloves before testing the high voltage cable.",
        "required_safety_phrases": ["wear insulated gloves"],
        "industrial_terms": ["high voltage cable"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-strong-asr-v9-008",
        "text": "Stop the cooling fan before removing the protective guard.",
        "required_safety_phrases": ["stop the cooling fan"],
        "industrial_terms": ["protective guard"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-strong-asr-v9-009",
        "text": "Secure the service door before restarting the compressor.",
        "required_safety_phrases": ["secure the service door"],
        "industrial_terms": ["compressor"],
        "risk": "HIGH",
    },
    {
        "case_id": "tts-strong-asr-v9-010",
        "text": "Drain the hydraulic tank before replacing the return filter.",
        "required_safety_phrases": ["drain the hydraulic tank"],
        "industrial_terms": ["return filter"],
        "risk": "HIGH",
    },
)


def validation_document() -> dict[str, Any]:
    return {
        "schema_version": "industrial-safety-tts-validation/v7",
        "suite_id": "industrial-safety-tts-validation-v7",
        "classification": CLASSIFICATION,
        "purpose": "DEVELOPMENT_PREFLIGHT_ONLY",
        "formal_gold": False,
        "project_enterprise_use_authorized": True,
        "retired_gold_v8_role": "DISCLOSED_DEVELOPMENT_REGRESSION_ONLY",
        "retired_gold_v8_reused_as_formal_gold": False,
        "inherited_validation_role": "DISCLOSED_DEVELOPMENT_REGRESSION_ONLY",
        "rows": list(_VALIDATION_ROWS),
    }


def formal_gold_document() -> dict[str, Any]:
    return {
        "schema_version": "industrial-safety-tts-final-gold/v9",
        "dataset_id": "industrial-safety-tts-final-gold-v9",
        "classification": CLASSIFICATION,
        "purpose": "FINAL_INDEPENDENT_VERIFIER_GUIDED_TTS_EVALUATION",
        "frozen_at": "2026-08-25T13:00:00Z",
        "formal_evaluation_limit": 1,
        "source_type": "PLATFORM_SYNTHETIC",
        "project_enterprise_use_authorized": True,
        "enterprise_production_data": False,
        "production_gold_eligible": False,
        "supersedes_dataset_id": "industrial-safety-tts-final-gold-v8",
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
            {"dataset_id": "industrial-safety-tts-final-gold-v8", "reuse_permitted": False},
        ],
        "rows": list(_GOLD_ROWS),
    }


def planned_run_id(repo_root: Path, image_digest: str) -> str:
    root = repo_root.resolve(strict=True)
    v2._FILES.require_image_digest(image_digest)
    validation_path, gold_path = read_inputs(root)
    source_binding = _retired_source(root)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_binding["sha256"],
        "predecessor": _predecessor_evidence(root).model_dump(mode="json"),
        "prior_attempt": _prior_attempt_evidence(root).model_dump(mode="json"),
        "prior_latency_attempt": _prior_latency_evidence(root).model_dump(mode="json"),
        "prior_asr_attempt": _prior_asr_evidence(root).model_dump(mode="json"),
        "asr_verifier": _asr_model_evidence(root).model_dump(mode="json"),
        "validation_sha256": file_sha256(validation_path),
        "gold_sha256": file_sha256(gold_path),
        "config": _CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
    }
    return f"tts-strong-asr-{tts_files.digest(identity)[:20]}"


def verify_calibrated_tts_value(
    repo_root: Path, report_path: Path | None = None
) -> TtsCalibratedValueReport:
    root = repo_root.resolve(strict=True)
    path = (
        _files.latest_outcome_path(root, LATEST_OUTCOME_RELATIVE)
        if report_path is None
        else _files.inside_file(root, report_path)
    )
    report = TtsCalibratedValueReport.model_validate_json(path.read_text(encoding="utf-8"))
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != tts_files.digest(unsigned):
        raise TtsCalibratedValueLabError("TTS v4 evidence chain changed")
    if report.source.model_dump(mode="json") != _retired_source(root):
        raise TtsCalibratedValueLabError("TTS historical source binding changed")
    if planned_run_id(root, report.image_digest) != report.run_id:
        raise TtsCalibratedValueLabError("TTS v4 run identity changed")
    predecessor = _predecessor_evidence(root)
    if predecessor != report.predecessor:
        raise TtsCalibratedValueLabError("TTS v4 predecessor evidence changed")
    if _prior_attempt_evidence(root) != report.prior_attempt:
        raise TtsCalibratedValueLabError("TTS prior-attempt evidence changed")
    if _prior_latency_evidence(root) != report.prior_latency_attempt:
        raise TtsCalibratedValueLabError("TTS prior latency evidence changed")
    if _prior_asr_evidence(root) != report.prior_asr_attempt:
        raise TtsCalibratedValueLabError("TTS prior ASR evidence changed")
    if _asr_model_evidence(root) != report.asr_verifier:
        raise TtsCalibratedValueLabError("TTS strong ASR snapshot changed")
    _files.verify_file_binding(root, report.dataset.validation_manifest)
    _files.verify_file_binding(root, report.dataset.final_gold_manifest)
    for stage in (report.validation, report.evaluation):
        _files.verify_file_binding(root, stage.formal_report)
        _files.verify_file_binding(root, stage.observations)
        if set(stage.hard_gate_results) != TTS_HARD_GATES:
            raise TtsCalibratedValueLabError("TTS formal gate contract changed")
    bundle_path = _files.inside_directory(root, Path(report.candidate.bundle.path))
    if _files.directory_binding(root, bundle_path) != report.candidate.bundle.model_dump(
        mode="json"
    ):
        raise TtsCalibratedValueLabError("TTS v4 candidate bundle changed")
    _files.verify_file_binding(root, report.candidate.runtime_policy_file)
    if set(report.hard_gates) != set(_HARD_GATES):
        raise TtsCalibratedValueLabError("TTS v4 hard-gate contract changed")
    expected_failures = tuple(
        sorted(name for name, passed in report.hard_gates.items() if not passed)
    )
    if expected_failures != report.failed_hard_gates:
        raise TtsCalibratedValueLabError("TTS v4 failed-gate projection changed")
    if report.candidate_accepted != (not expected_failures):
        raise TtsCalibratedValueLabError("TTS v4 acceptance projection changed")
    retirement = _files.verify_gold_retirement(
        root, root / report.gold_retirement_path, gold_relative=GOLD_RELATIVE, version=9
    )
    if retirement["run_id"] != report.run_id:
        raise TtsCalibratedValueLabError("TTS v4 Gold retirement run changed")
    return report


def _predecessor_evidence(root: Path) -> PredecessorEvidence:
    return _files.predecessor_evidence(
        root,
        rejection_relative=PREDECESSOR_REJECTION_RELATIVE,
        candidate_relative=PREDECESSOR_CANDIDATE_RELATIVE,
        load_rejection=v2.verify_tts_recovery_rejection,
    )


def _prior_attempt_evidence(root: Path) -> PriorAttemptEvidence:
    return _files.prior_attempt_evidence(
        root,
        manifests=(
            (4, PRIOR_GOLD_V4_RELATIVE, PRIOR_GOLD_V4_RETIREMENT_RELATIVE),
            (5, PRIOR_GOLD_V5_RELATIVE, PRIOR_GOLD_V5_RETIREMENT_RELATIVE),
            (6, PRIOR_GOLD_V6_RELATIVE, PRIOR_GOLD_V6_RETIREMENT_RELATIVE),
        ),
        evidence_model=PriorAttemptEvidence,
    )


def _prior_latency_evidence(root: Path) -> PriorLatencyEvidence:
    return _files.prior_latency_evidence(
        root,
        rejection_relative=PRIOR_V7_REJECTION_RELATIVE,
        gold_relative=PRIOR_GOLD_V7_RELATIVE,
        retirement_relative=PRIOR_GOLD_V7_RETIREMENT_RELATIVE,
        load_final_report=v4.verify_calibrated_tts_value,
    )


def _prior_asr_evidence(root: Path) -> PriorAsrEvidence:
    failure_path = _files.inside_file(root, PRIOR_V8_FAILURE_RELATIVE)
    validation_path = _files.inside_file(root, PRIOR_V8_VALIDATION_RELATIVE)
    gold_path = _files.inside_file(root, PRIOR_GOLD_V8_RELATIVE)
    retirement_path = _files.inside_file(root, PRIOR_GOLD_V8_RETIREMENT_RELATIVE)
    failure = _files.load_object(failure_path)
    validation = _files.load_object(validation_path)
    retirement = _files.load_object(retirement_path)
    failure_detail = failure.get("failure")
    validation_summary = validation.get("summary")
    serialized_validation = json.dumps(validation, sort_keys=True)
    if (
        failure.get("schema_version") != "enterprise-tts-latency-gold-stage-failure/v1"
        or failure.get("status") != "TTS_LATENCY_GOLD_NO_SAFE_VARIANT_REJECTED"
        or failure.get("run_id") != "tts-latency-f1c6f42b8bc3f3fa5f34"
        or failure.get("failed_hard_gates") != ["no_contract_complete_variant"]
        or failure.get("candidate_accepted") is not False
        or not isinstance(failure_detail, dict)
        or failure_detail.get("case_id") != "tts-latency-v8-001"
        or failure_detail.get("formal_gold_consumed") is not True
        or validation.get("status") != "TTS_LATENCY_GPU_VALIDATION_PASSED"
        or validation.get("run_id") != "tts-latency-f1c6f42b8bc3f3fa5f34"
        or validation.get("formal_gold_consumed") is not False
        or validation.get("failed_quality_gates") != []
        or not isinstance(validation_summary, dict)
        or validation_summary.get("candidate_pool_size") != 23
        or v2.ASR_MODEL_ID not in serialized_validation
        or v2.ASR_REVISION not in serialized_validation
        or retirement.get("schema_version") != "industrial-safety-tts-gold-retirement/v8"
        or retirement.get("status") != "RETIRED_AFTER_REJECTED_FINAL_EVALUATION"
        or retirement.get("run_id") != "tts-latency-f1c6f42b8bc3f3fa5f34"
        or retirement.get("failed_hard_gates") != ["no_contract_complete_variant"]
        or retirement.get("reuse_permitted") is not False
        or retirement.get("gold_manifest_sha256") != file_sha256(gold_path)
        or retirement.get("outcome_sha256") != file_sha256(failure_path)
    ):
        raise TtsCalibratedValueLabError("TTS v8 ASR failure evidence changed")
    source_value = failure.get("source")
    if source_value != v5._retired_source(root):
        raise TtsCalibratedValueLabError("TTS v8 source evidence changed")
    return PriorAsrEvidence(
        gold_stage_failure=FileBinding.model_validate(_files.file_binding(root, failure_path)),
        validation_probe=FileBinding.model_validate(_files.file_binding(root, validation_path)),
        gold_manifest=FileBinding.model_validate(_files.file_binding(root, gold_path)),
        gold_retirement=FileBinding.model_validate(_files.file_binding(root, retirement_path)),
        run_id="tts-latency-f1c6f42b8bc3f3fa5f34",
        status="TTS_LATENCY_GOLD_NO_SAFE_VARIANT_REJECTED",
        failed_hard_gates=("no_contract_complete_variant",),
        candidate_pool_size=23,
        weak_asr_model_id="openai/whisper-tiny",
        weak_asr_revision="c4f62d5fce5d73978a1dbfcac01926312acd1a51",
        failure_case_id="tts-latency-v8-001",
        validation_quality_gates_passed=True,
        same_gold_reuse_permitted=False,
    )


def _asr_model_evidence(root: Path) -> AsrModelEvidence:
    path = _files.inside_directory(root, ASR_RELATIVE)
    required = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    if any(not (path / name).is_file() for name in required):
        raise TtsCalibratedValueLabError("Whisper Small snapshot is incomplete")
    config = _files.load_object(path / "config.json")
    if config.get("model_type") != "whisper":
        raise TtsCalibratedValueLabError("Whisper Small model contract changed")
    return AsrModelEvidence(
        bundle=DirectoryBinding.model_validate(_files.directory_binding(root, path)),
        model_id=ASR_MODEL_ID,
        revision=ASR_REVISION,
        model_format="SAFETENSORS",
        local_files_only=True,
    )


def _verify_dataset_independence(root: Path, validation_path: Path, gold_path: Path) -> None:
    validation = _files.load_object(validation_path)
    gold = _files.load_object(gold_path)
    prior_documents = [
        _files.load_object(root / v2.TRAINING_DATASET_RELATIVE / "manifest.json"),
        _files.load_object(root / v2.GOLD_DATASET_RELATIVE / "manifest.json"),
        _files.load_object(root / PRIOR_GOLD_V4_RELATIVE),
        _files.load_object(root / PRIOR_GOLD_V5_RELATIVE),
        _files.load_object(root / PRIOR_GOLD_V6_RELATIVE),
        _files.load_object(root / PRIOR_GOLD_V7_RELATIVE),
        _files.load_object(root / PRIOR_GOLD_V8_RELATIVE),
    ]
    validation_ids, validation_texts = _files.ids_and_texts(validation)
    gold_ids, gold_texts = _files.ids_and_texts(gold)
    if validation_ids & gold_ids or validation_texts & gold_texts:
        raise TtsCalibratedValueLabError("TTS validation and Gold are not disjoint")
    for document in prior_documents:
        prior_ids, prior_texts = _files.ids_and_texts(document)
        if gold_ids & prior_ids or gold_texts & prior_texts:
            raise TtsCalibratedValueLabError("TTS Gold v9 reuses predecessor data")
    if (
        len(validation_ids) != 12
        or len(gold_ids) != 10
        or gold.get("dataset_id") != "industrial-safety-tts-final-gold-v9"
        or gold.get("formal_evaluation_limit") != 1
        or gold.get("selection_policy", {}).get("single_final_evaluation_only") is not True
    ):
        raise TtsCalibratedValueLabError("TTS governed dataset contract changed")


def _retired_source(root: Path) -> dict[str, Any]:
    return retired_source_binding(root, _LEGACY_SOURCE_PATH, error_type=TtsCalibratedValueLabError)


def read_inputs(repo_root: Path) -> tuple[Path, Path]:
    root = repo_root.resolve(strict=True)
    _predecessor_evidence(root)
    _prior_attempt_evidence(root)
    _prior_latency_evidence(root)
    _prior_asr_evidence(root)
    _asr_model_evidence(root)
    validation_path = _files.inside_file(root, VALIDATION_RELATIVE)
    gold_path = _files.inside_file(root, GOLD_RELATIVE)
    _files.require_json(validation_path, validation_document())
    _files.require_json(gold_path, formal_gold_document())
    _verify_dataset_independence(root, validation_path, gold_path)
    return validation_path, gold_path
