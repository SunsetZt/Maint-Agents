"""Read-only verification of retained TTS experiment evidence; no training or GPU execution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.model_evidence import tts_files
from industrial_ops_agent.model_evidence import tts_recovery as v2
from industrial_ops_agent.model_evidence.report_files import file_sha256
from industrial_ops_agent.model_evidence.retired_sources import retired_source_binding
from industrial_ops_agent.model_evidence.tts_contracts import (
    TTS_HARD_GATES,
    DirectoryBinding,
    FileBinding,
    PredecessorEvidence,
)

_LEGACY_SOURCE_PATH = "src/industrial_ops_agent/simulation/tts_calibrated_value_lab.py"

SCHEMA_VERSION: Literal["enterprise-tts-calibrated-value-lab/v5"] = (
    "enterprise-tts-calibrated-value-lab/v5"
)


CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"


OUTPUT_RELATIVE = Path("artifacts/m7-tts-calibrated-value-lab")


VALIDATION_RELATIVE = OUTPUT_RELATIVE / "inputs/industrial-safety-tts-validation-v4.json"


PRIOR_GOLD_V4_RELATIVE = (
    OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v4/manifest.json"
)


PRIOR_GOLD_V4_RETIREMENT_RELATIVE = OUTPUT_RELATIVE / "gold-v4-retirement.json"


PRIOR_GOLD_V5_RELATIVE = (
    OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v5/manifest.json"
)


PRIOR_GOLD_V5_RETIREMENT_RELATIVE = OUTPUT_RELATIVE / "gold-v5-retirement.json"


GOLD_RELATIVE = OUTPUT_RELATIVE / "inputs/industrial-safety-tts-final-gold-v6/manifest.json"


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
    ("base", "exact", 1.040),
    ("postnet-v2", "exact", 0.960),
    ("postnet-v2", "exact", 1.000),
    ("postnet-v2", "exact", 1.040),
    ("base", "safety-pause", 1.000),
    ("postnet-v2", "safety-pause", 1.000),
)


_SEGMENTED_POLICIES: tuple[tuple[str, float], ...] = (
    ("base", 1.000),
    ("postnet-v2", 1.000),
)


_WORD_SEGMENTED_POLICIES: tuple[tuple[str, float], ...] = (("base", 1.000),)


_SPEED_FACTORS: tuple[float, ...] = (0.985, 1.000, 1.015)


_CONTRACT_PRONUNCIATION_EQUIVALENCES: tuple[frozenset[str], ...] = (frozenset({"brake", "break"}),)


_CONFIG: dict[str, Any] = {
    "method": "SPEECHT5_VERIFIER_GUIDED_SAFE_VOICE_SELECTION",
    "generation_policies": _GENERATION_POLICIES,
    "segmented_policies": _SEGMENTED_POLICIES,
    "word_segmented_policies": _WORD_SEGMENTED_POLICIES,
    "speed_factors": _SPEED_FACTORS,
    "contract_pronunciation_equivalences": tuple(
        tuple(sorted(group)) for group in _CONTRACT_PRONUNCIATION_EQUIVALENCES
    ),
    "voice_similarity_improvement_min": 0.0001,
    "safety_phrase_completeness_min": 1.0,
    "terminology_recall_min": 0.95,
    "intelligibility_min": 0.90,
    "audio_integrity_rate_min": 1.0,
    "maximum_case_character_error_rate": 0.10,
    "maximum_candidate_pool_size": 29,
    "maximum_pipeline_latency_ratio": 30.0,
    "maximum_gpu_reserved_bytes": 2 * 1024**3,
    "cpu_threads": 2,
    "dataloader_workers": 0,
    "container_memory_limit_bytes": 4 * 1024**3,
}


_HARD_GATES = (
    "data_governance",
    "predecessor_rejection_preserved",
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
    """The TTS v3 experiment or its immutable evidence is invalid."""


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
    retired_v2_v3_v4_v5_gold_reused: Literal[False] = False
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
    candidate_pool_size: int = Field(gt=1, le=29)
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
    method: Literal["SPEECHT5_VERIFIER_GUIDED_SAFE_VOICE_SELECTION"] = (
        "SPEECHT5_VERIFIER_GUIDED_SAFE_VOICE_SELECTION"
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
    statuses: tuple[
        Literal["RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION"],
        Literal["RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION"],
    ]
    gold_dataset_ids: tuple[
        Literal["industrial-safety-tts-final-gold-v4"],
        Literal["industrial-safety-tts-final-gold-v5"],
    ]
    gold_evaluation_pass_counts: tuple[Literal[1], Literal[1]] = (1, 1)
    reuse_permitted: tuple[Literal[False], Literal[False]] = (False, False)


class TtsCalibratedValueReport(_ClosedModel):
    schema_version: Literal["enterprise-tts-calibrated-value-lab/v5"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal[
        "TTS_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
        "TTS_CALIBRATED_ACTUAL_GPU_CANDIDATE_REJECTED",
    ]
    decision: Literal[
        "TTS_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT",
        "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD",
    ]
    run_id: str = Field(pattern=r"^tts-calibrated-[0-9a-f]{20}$")
    source: FileBinding
    image: Literal["industrial-ops/tts-xvector-runtime:speechbrain-1.0.3"] = IMAGE
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    predecessor: PredecessorEvidence
    prior_attempt: PriorAttemptEvidence
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


_VALIDATION_ROWS: tuple[dict[str, Any], ...] = (
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


_GOLD_ROWS: tuple[dict[str, Any], ...] = (
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


def validation_document() -> dict[str, Any]:
    return {
        "schema_version": "industrial-safety-tts-validation/v4",
        "suite_id": "industrial-safety-tts-validation-v4",
        "classification": CLASSIFICATION,
        "purpose": "DEVELOPMENT_PREFLIGHT_ONLY",
        "formal_gold": False,
        "project_enterprise_use_authorized": True,
        "rows": list(_VALIDATION_ROWS),
    }


def formal_gold_document() -> dict[str, Any]:
    return {
        "schema_version": "industrial-safety-tts-final-gold/v6",
        "dataset_id": "industrial-safety-tts-final-gold-v6",
        "classification": CLASSIFICATION,
        "purpose": "FINAL_INDEPENDENT_VERIFIER_GUIDED_TTS_EVALUATION",
        "frozen_at": "2026-09-02T00:00:00Z",
        "formal_evaluation_limit": 1,
        "source_type": "PLATFORM_SYNTHETIC",
        "project_enterprise_use_authorized": True,
        "enterprise_production_data": False,
        "production_gold_eligible": False,
        "supersedes_dataset_id": "industrial-safety-tts-final-gold-v5",
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
        "validation_sha256": file_sha256(validation_path),
        "gold_sha256": file_sha256(gold_path),
        "config": _CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
    }
    return f"tts-calibrated-{tts_files.digest(identity)[:20]}"


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
        raise TtsCalibratedValueLabError("TTS v3 evidence chain changed")
    if report.source.model_dump(mode="json") != _retired_source(root):
        raise TtsCalibratedValueLabError("TTS historical source binding changed")
    if planned_run_id(root, report.image_digest) != report.run_id:
        raise TtsCalibratedValueLabError("TTS v3 run identity changed")
    predecessor = _predecessor_evidence(root)
    if predecessor != report.predecessor:
        raise TtsCalibratedValueLabError("TTS v3 predecessor evidence changed")
    if _prior_attempt_evidence(root) != report.prior_attempt:
        raise TtsCalibratedValueLabError("TTS prior-attempt evidence changed")
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
        raise TtsCalibratedValueLabError("TTS v3 candidate bundle changed")
    _files.verify_file_binding(root, report.candidate.runtime_policy_file)
    if set(report.hard_gates) != set(_HARD_GATES):
        raise TtsCalibratedValueLabError("TTS v3 hard-gate contract changed")
    expected_failures = tuple(
        sorted(name for name, passed in report.hard_gates.items() if not passed)
    )
    if expected_failures != report.failed_hard_gates:
        raise TtsCalibratedValueLabError("TTS v3 failed-gate projection changed")
    if report.candidate_accepted != (not expected_failures):
        raise TtsCalibratedValueLabError("TTS v3 acceptance projection changed")
    retirement = _files.verify_gold_retirement(
        root, root / report.gold_retirement_path, gold_relative=GOLD_RELATIVE, version=6
    )
    if retirement["run_id"] != report.run_id:
        raise TtsCalibratedValueLabError("TTS v3 Gold retirement run changed")
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
        ),
        evidence_model=PriorAttemptEvidence,
    )


def _verify_dataset_independence(root: Path, validation_path: Path, gold_path: Path) -> None:
    validation = _files.load_object(validation_path)
    gold = _files.load_object(gold_path)
    prior_documents = [
        _files.load_object(root / v2.TRAINING_DATASET_RELATIVE / "manifest.json"),
        _files.load_object(root / v2.GOLD_DATASET_RELATIVE / "manifest.json"),
        _files.load_object(root / PRIOR_GOLD_V4_RELATIVE),
        _files.load_object(root / PRIOR_GOLD_V5_RELATIVE),
    ]
    validation_ids, validation_texts = _files.ids_and_texts(validation)
    gold_ids, gold_texts = _files.ids_and_texts(gold)
    if validation_ids & gold_ids or validation_texts & gold_texts:
        raise TtsCalibratedValueLabError("TTS validation and Gold are not disjoint")
    for document in prior_documents:
        prior_ids, prior_texts = _files.ids_and_texts(document)
        if gold_ids & prior_ids or gold_texts & prior_texts:
            raise TtsCalibratedValueLabError("TTS Gold v6 reuses predecessor data")
    if (
        len(validation_ids) != 12
        or len(gold_ids) != 10
        or gold.get("dataset_id") != "industrial-safety-tts-final-gold-v6"
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
    validation_path = _files.inside_file(root, VALIDATION_RELATIVE)
    gold_path = _files.inside_file(root, GOLD_RELATIVE)
    _files.require_json(validation_path, validation_document())
    _files.require_json(gold_path, formal_gold_document())
    _verify_dataset_independence(root, validation_path, gold_path)
    return validation_path, gold_path
