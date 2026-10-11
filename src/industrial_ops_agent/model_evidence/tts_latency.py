"""Read-only verification of retained TTS experiment evidence; no training or GPU execution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.model_evidence import tts_files
from industrial_ops_agent.model_evidence import tts_final as v4
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

_LEGACY_SOURCE_PATH = "src/industrial_ops_agent/simulation/tts_latency_value_lab.py"

SCHEMA_VERSION: Literal["enterprise-tts-latency-value-lab/v7"] = (
    "enterprise-tts-latency-value-lab/v7"
)


CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"


CALIBRATED_OUTPUT_RELATIVE = Path("artifacts/m7-tts-calibrated-value-lab")


FINAL_OUTPUT_RELATIVE = Path("artifacts/m7-tts-final-value-lab")


OUTPUT_RELATIVE = Path("artifacts/m7-tts-latency-value-lab")


VALIDATION_RELATIVE = OUTPUT_RELATIVE / "inputs/industrial-safety-tts-validation-v6.json"


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


GOLD_RELATIVE = OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v8/manifest.json"


LATEST_OUTCOME_RELATIVE = OUTPUT_RELATIVE / "latest-outcome.json"


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


_files = tts_files.TtsEvidenceFiles(TtsCalibratedValueLabError, v2._word_units)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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
    verifier_policy: Literal["WHISPER_SAFETY_THEN_XVECTOR_MAX"] = "WHISPER_SAFETY_THEN_XVECTOR_MAX"
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
        "validation_sha256": file_sha256(validation_path),
        "gold_sha256": file_sha256(gold_path),
        "config": _CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
    }
    return f"tts-latency-{tts_files.digest(identity)[:20]}"


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
        root, root / report.gold_retirement_path, gold_relative=GOLD_RELATIVE, version=8
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
    ]
    validation_ids, validation_texts = _files.ids_and_texts(validation)
    gold_ids, gold_texts = _files.ids_and_texts(gold)
    if validation_ids & gold_ids or validation_texts & gold_texts:
        raise TtsCalibratedValueLabError("TTS validation and Gold are not disjoint")
    for document in prior_documents:
        prior_ids, prior_texts = _files.ids_and_texts(document)
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


def _retired_source(root: Path) -> dict[str, Any]:
    return retired_source_binding(root, _LEGACY_SOURCE_PATH, error_type=TtsCalibratedValueLabError)


def read_inputs(repo_root: Path) -> tuple[Path, Path]:
    root = repo_root.resolve(strict=True)
    _predecessor_evidence(root)
    _prior_attempt_evidence(root)
    _prior_latency_evidence(root)
    validation_path = _files.inside_file(root, VALIDATION_RELATIVE)
    gold_path = _files.inside_file(root, GOLD_RELATIVE)
    _files.require_json(validation_path, validation_document())
    _files.require_json(gold_path, formal_gold_document())
    _verify_dataset_independence(root, validation_path, gold_path)
    return validation_path, gold_path
