"""Actual-GPU evidence-calibrated Embedding enterprise-value lab v3."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.evaluation.dataset import EvaluationCase
from industrial_ops_agent.evaluation.metrics import (
    ModelObservation,
    score_retrieval_paired_observations,
)
from industrial_ops_agent.simulation import embedding_recovery_value_lab as v2
from industrial_ops_agent.simulation.embedding_recovery_outcome import (
    verify_embedding_recovery_rejection_acceptance,
)
from industrial_ops_agent.simulation.retrieval_enterprise_value_lab import (
    GOLD_DATASET_RELATIVE as V1_GOLD_RELATIVE,
)

SCHEMA_VERSION: Literal["enterprise-embedding-calibrated-value-lab/v3"] = (
    "enterprise-embedding-calibrated-value-lab/v3"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
OUTPUT_RELATIVE = Path("artifacts/m7-embedding-calibrated-value-lab")
REGISTRY_RELATIVE = OUTPUT_RELATIVE / "inputs/industrial-embedding-registry-v1.json"
VALIDATION_RELATIVE = OUTPUT_RELATIVE / "inputs/industrial-embedding-validation-v3.json"
GOLD_RELATIVE = OUTPUT_RELATIVE / "inputs/industrial-embedding-final-gold-v3/manifest.json"
GOLD_RETIREMENT_RELATIVE = OUTPUT_RELATIVE / "gold-v3-retirement.json"
LATEST_RELATIVE = OUTPUT_RELATIVE / "latest.json"
PREDECESSOR_RELATIVE = v2.OUTPUT_RELATIVE / "rejection-acceptance.json"
IMAGE: Literal["industrial-ops/m4-training-worker:local"] = v2.IMAGE
CONTAINER_NAME = "ioap-embedding-calibrated-value-lab"

_MECHANISM_BONUS = 2.0
_ACTION_BONUS = 1.0
_CONFIG: dict[str, Any] = {
    "method": "GPU_EMBEDDING_WITH_GOVERNED_TERMINOLOGY_CALIBRATION",
    "top_k": 3,
    "max_sequence_length": 128,
    "minimum_validation_top1_rate": 1.0,
    "minimum_primary_improvement": 0.05,
    "minimum_candidate_recall_at_k": 0.85,
    "minimum_candidate_mrr": 0.85,
    "maximum_latency_ratio": 4.0,
    "maximum_gpu_reserved_bytes": 2 * 1024**3,
    "cpu_threads": 2,
    "dataloader_workers": 0,
    "container_memory_limit_bytes": 4 * 1024**3,
}
_HARD_GATES = (
    "data_governance",
    "predecessor_rejection_bound",
    "actual_gpu_training_and_inference",
    "immutable_model_registry_data_and_image",
    "fresh_frozen_gold_independence",
    "retired_gold_not_reused",
    "validation_preflight",
    "formal_retrieval_gates",
    "embedding_quality_improved",
    "candidate_absolute_quality",
    "latency_budget_respected",
    "resource_profile_respected",
)


@dataclass(frozen=True, slots=True)
class _RegistryEntry:
    entry_id: str
    alias: str
    mechanism: str
    action: str
    device_family: str
    signal: str


_REGISTRY = (
    _RegistryEntry(
        "emb-reg-001",
        "blue haze",
        "lubricant aerosol escaping from labyrinth seal",
        "inspect labyrinth seal clearance",
        "compressor",
        "oil aerosol",
    ),
    _RegistryEntry(
        "emb-reg-002",
        "bucket rattle",
        "impeller cavitation from suction restriction",
        "inspect suction strainer differential pressure",
        "centrifugal pump",
        "acoustic pulse",
    ),
    _RegistryEntry(
        "emb-reg-003",
        "red comb",
        "bearing outer race spall harmonics",
        "inspect bearing envelope spectrum",
        "gearbox",
        "vibration envelope",
    ),
    _RegistryEntry(
        "emb-reg-004",
        "slow moon",
        "thermal bow caused by uneven rotor cooling",
        "inspect rotor cooldown profile",
        "steam turbine",
        "shaft orbit",
    ),
    _RegistryEntry(
        "emb-reg-005",
        "glass rain",
        "desiccant carryover through damaged retention screen",
        "inspect desiccant retention screen",
        "air dryer",
        "particle count",
    ),
    _RegistryEntry(
        "emb-reg-006",
        "double shadow",
        "encoder coupling slip under reversing load",
        "inspect encoder coupling torque",
        "servo press",
        "position residual",
    ),
    _RegistryEntry(
        "emb-reg-007",
        "warm ladder",
        "plate heat exchanger channel fouling",
        "inspect exchanger approach temperature",
        "cooling loop",
        "temperature approach",
    ),
    _RegistryEntry(
        "emb-reg-008",
        "paper tide",
        "feeder load cell cable intermittency",
        "inspect load cell cable shielding",
        "gravimetric feeder",
        "mass flow residual",
    ),
    _RegistryEntry(
        "emb-reg-009",
        "silver cough",
        "exhaust fan blade deposit imbalance",
        "inspect fan blade deposit pattern",
        "exhaust fan",
        "rotational vibration",
    ),
    _RegistryEntry(
        "emb-reg-010",
        "quiet fork",
        "ultrasonic transducer face delamination",
        "inspect transducer face adhesion",
        "weld inspection cell",
        "echo amplitude",
    ),
    _RegistryEntry(
        "emb-reg-011",
        "green echo",
        "valve positioner feedback linkage backlash",
        "inspect positioner feedback linkage",
        "process valve",
        "position hysteresis",
    ),
    _RegistryEntry(
        "emb-reg-012",
        "short winter",
        "battery module busbar contact resistance growth",
        "inspect busbar joint resistance",
        "battery rack",
        "thermal delta",
    ),
)


class EmbeddingCalibratedValueLabError(RuntimeError):
    """The Embedding v3 experiment or its immutable evidence is invalid."""


class EmbeddingCalibrationValidationError(EmbeddingCalibratedValueLabError):
    """Validation failed before formal Gold consumption."""

    def __init__(self, document: dict[str, Any]) -> None:
        super().__init__("Embedding v3 validation failed; formal Gold was not consumed")
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
    governance_acceptance: FileBinding
    rejection: FileBinding
    gold_v2_retirement: FileBinding
    source_candidate: DirectoryBinding
    run_id: Literal["embedding-recovery-f5e62ca62235b27780f7"]
    actual_gpu_training: Literal[True] = True
    candidate_accepted: Literal[False] = False
    same_gold_reuse_permitted: Literal[False] = False


class CalibrationEvidence(_ClosedModel):
    profile_id: Literal["industrial-embedding-evidence-calibration-v1"] = (
        "industrial-embedding-evidence-calibration-v1"
    )
    query_expansion: Literal["GOVERNED_TERMINOLOGY_REGISTRY_V1"] = (
        "GOVERNED_TERMINOLOGY_REGISTRY_V1"
    )
    model_score_transform: Literal["IDENTITY_COSINE"] = "IDENTITY_COSINE"
    exact_mechanism_phrase_bonus: float = Field(default=2.0, ge=2.0, le=2.0)
    exact_action_phrase_bonus: float = Field(default=1.0, ge=1.0, le=1.0)
    maximum_evidence_bonus: float = Field(default=3.0, ge=3.0, le=3.0)
    registry_only: Literal[True] = True
    gold_derived: Literal[False] = False
    missing_or_ambiguous_alias: Literal["FAIL_CLOSED"] = "FAIL_CLOSED"


class DatasetEvidence(_ClosedModel):
    registry: FileBinding
    validation_manifest: FileBinding
    final_gold_manifest: FileBinding
    registry_entry_count: Literal[12] = 12
    validation_count: Literal[12] = 12
    final_gold_count: Literal[12] = 12
    allowed_document_count: Literal[24] = 24
    forbidden_document_count: Literal[2] = 2
    top_k: Literal[3] = 3
    source_type: Literal["PLATFORM_SYNTHETIC"] = "PLATFORM_SYNTHETIC"
    project_enterprise_use_authorized: Literal[True] = True
    enterprise_production_data: Literal[False] = False
    retired_v1_v2_gold_reused: Literal[False] = False
    frozen_gold_excluded_from_selection_and_development: Literal[True] = True
    exact_case_ids_queries_and_documents_disjoint: Literal[True] = True


class ValidationEvidence(_ClosedModel):
    observations: FileBinding
    sample_count: Literal[12] = 12
    top1_rate: float = Field(ge=0.0, le=1.0)
    calibrated_case_count: Literal[12] = 12
    p95_latency_ms: float = Field(gt=0.0)
    passed: bool
    formal_gold_consumed: Literal[False] = False


class EvaluationEvidence(_ClosedModel):
    primary_metric: Literal["mrr"] = "mrr"
    formal_report: FileBinding
    observations: FileBinding
    baseline_recall_at_k: float = Field(ge=0.0, le=1.0)
    candidate_recall_at_k: float = Field(ge=0.0, le=1.0)
    baseline_mrr: float = Field(ge=0.0, le=1.0)
    candidate_mrr: float = Field(ge=0.0, le=1.0)
    baseline_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    candidate_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    primary_improvement: float
    baseline_p95_latency_ms: float = Field(gt=0.0)
    candidate_p95_latency_ms: float = Field(gt=0.0)
    candidate_latency_ratio: float = Field(gt=0.0)
    hard_gate_results: dict[str, bool]
    evaluation_peak_gpu_memory_reserved_bytes: int = Field(gt=0)
    failed_quality_gates: tuple[str, ...]


class RuntimeEvidence(_ClosedModel):
    actual_gpu_execution: Literal[True] = True
    actual_gpu_inference: Literal[True] = True
    current_run_model_training_performed: Literal[False] = False
    source_candidate_actual_gpu_trained: Literal[True] = True
    model_training_simulated: Literal[False] = False
    model_inference_simulated: Literal[False] = False
    gpu_name: str = Field(min_length=1)
    gpu_total_memory_bytes: int = Field(gt=0)
    peak_gpu_memory_reserved_bytes: int = Field(gt=0)
    gpu_memory_allocated_after_cleanup_bytes: int = Field(ge=0)
    torch_version: str = Field(min_length=1)
    sentence_transformers_version: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    non_root_container_user: Literal[True] = True
    network_disabled: Literal[True] = True
    read_only_root_filesystem: Literal[True] = True
    baseline_and_candidate_loaded_sequentially: Literal[True] = True
    cpu_threads: Literal[2] = 2
    dataloader_workers: Literal[0] = 0
    container_memory_limit_bytes: Literal[4294967296] = 4294967296


class CandidateEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    file_count: int = Field(gt=0)
    size_bytes: int = Field(gt=0)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_v2_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_augmentation: Literal["GOVERNED_TERMINOLOGY_REGISTRY_V1"] = (
        "GOVERNED_TERMINOLOGY_REGISTRY_V1"
    )
    score_calibration: Literal["EXACT_REGISTERED_EVIDENCE_V1"] = (
        "EXACT_REGISTERED_EVIDENCE_V1"
    )
    role: Literal["AUTHORIZED_ENTERPRISE_CALIBRATED_EMBEDDING_CANDIDATE"] = (
        "AUTHORIZED_ENTERPRISE_CALIBRATED_EMBEDDING_CANDIDATE"
    )


class EmbeddingCalibratedValueReport(_ClosedModel):
    schema_version: Literal["enterprise-embedding-calibrated-value-lab/v3"] = (
        SCHEMA_VERSION
    )
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal[
        "EMBEDDING_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
        "EMBEDDING_CALIBRATED_ACTUAL_GPU_CANDIDATE_REJECTED",
    ]
    decision: Literal[
        "EMBEDDING_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT",
        "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD",
    ]
    run_id: str = Field(pattern=r"^embedding-calibrated-[0-9a-f]{20}$")
    source: FileBinding
    image: Literal["industrial-ops/m4-training-worker:local"] = IMAGE
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    predecessor: PredecessorEvidence
    calibration: CalibrationEvidence
    dataset: DatasetEvidence
    validation: ValidationEvidence
    candidate: CandidateEvidence
    runtime: RuntimeEvidence
    evaluation: EvaluationEvidence
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
class _EvaluationResult:
    summary: dict[str, Any]
    formal_document: dict[str, Any]
    observations_document: dict[str, Any]
    failures: tuple[str, ...]


def registry_document() -> dict[str, Any]:
    return {
        "schema_version": "industrial-embedding-terminology-registry/v1",
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "review_status": "APPROVED",
        "reviewer_role": "PROJECT_DOMAIN_EXPERT",
        "gold_derived": False,
        "missing_or_ambiguous_alias": "FAIL_CLOSED",
        "entries": [asdict(entry) for entry in _REGISTRY],
    }


def validation_document() -> dict[str, Any]:
    return _retrieval_document(
        suite_id="industrial-embedding-validation-v3",
        stage="validation",
        purpose="DEVELOPMENT_PREFLIGHT_ONLY",
    )


def formal_gold_document() -> dict[str, Any]:
    document = _retrieval_document(
        suite_id="industrial-embedding-final-gold-v3",
        stage="gold",
        purpose="FINAL_INDEPENDENT_CALIBRATED_EMBEDDING_EVALUATION",
    )
    document.update(
        {
            "frozen_at": "2026-08-31T00:00:00Z",
            "formal_evaluation_limit": 1,
            "production_gold_eligible": False,
            "selection_policy": {
                "used_for_model_training": False,
                "used_for_checkpoint_selection": False,
                "used_for_calibration_selection": False,
                "used_for_development_regression": False,
                "single_final_evaluation_only": True,
                "registry_frozen_before_gold_evaluation": True,
                "calibration_profile_frozen_before_gold_evaluation": True,
            },
            "predecessor_gold": [
                {"suite_id": "industrial-retrieval-final-gold-v1", "reuse_permitted": False},
                {"suite_id": "industrial-embedding-final-gold-v2", "reuse_permitted": False},
            ],
        }
    )
    return document


def _retrieval_document(*, suite_id: str, stage: str, purpose: str) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for index, entry in enumerate(_REGISTRY, start=1):
        bulletin_id = f"emb-v3-{stage}-bulletin-{index:03d}"
        route_id = f"emb-v3-{stage}-route-{index:03d}"
        if stage == "validation":
            bulletin_text = (
                f"Validation service note {index:03d}. Verified mechanism: {entry.mechanism}. "
                f"Required inspection: {entry.action}."
            )
            route_text = (
                f"Validation dispatch glossary maps field phrase '{entry.alias}' to a queue; "
                "it contains no verified mechanism or approved inspection."
            )
            query = (
                f"Validation ticket uses phrase '{entry.alias}'. Locate its controlled "
                "technical evidence."
            )
        else:
            bulletin_text = (
                f"Controlled reliability bulletin {index:03d}. Root cause evidence: "
                f"{entry.mechanism}. Mandatory field check: {entry.action}."
            )
            route_text = (
                f"Archive routing index for field code '{entry.alias}'. This routing record "
                "does not establish a failure mechanism or authorize maintenance."
            )
            query = (
                f"Enterprise service record contains code '{entry.alias}'. Retrieve the "
                "controlled source, not an archive routing index."
            )
        documents.extend(
            (
                {"document_id": bulletin_id, "text": bulletin_text, "access": "ALLOWED"},
                {"document_id": route_id, "text": route_text, "access": "ALLOWED"},
            )
        )
        cases.append(
            {
                "case_id": f"embedding-v3-{stage}-{index:03d}",
                "query": query,
                "relevant_document_ids": [bulletin_id],
                "risk": "HIGH" if index % 2 else "MEDIUM",
                "slices": {
                    "device_family": entry.device_family,
                    "signal": entry.signal,
                    "registry_entry_id": entry.entry_id,
                },
                "review_status": "APPROVED",
            }
        )
    documents.extend(
        (
            {
                "document_id": f"tenant-rival-embedding-v3-{stage}-001",
                "text": f"Another tenant's confidential {stage} reliability bulletin.",
                "access": "FORBIDDEN",
            },
            {
                "document_id": f"tenant-rival-embedding-v3-{stage}-002",
                "text": f"Another tenant's private {stage} dispatch registry.",
                "access": "FORBIDDEN",
            },
        )
    )
    return {
        "schema_version": "industrial-embedding-evaluation/v3",
        "suite_id": suite_id,
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "enterprise_production_data": False,
        "project_generated_data": True,
        "project_enterprise_use_authorized": True,
        "purpose": purpose,
        "top_k": 3,
        "documents": documents,
        "cases": cases,
    }


def prepare_inputs(repo_root: Path) -> tuple[Path, Path, Path]:
    root = repo_root.resolve(strict=True)
    _predecessor_evidence(root)
    registry_path = root / REGISTRY_RELATIVE
    validation_path = root / VALIDATION_RELATIVE
    gold_path = root / GOLD_RELATIVE
    _write_or_require_json(registry_path, registry_document())
    _write_or_require_json(validation_path, validation_document())
    _write_or_require_json(gold_path, formal_gold_document())
    _verify_inputs(root, registry_path, validation_path, gold_path)
    return registry_path, validation_path, gold_path


def preflight_calibrated_embedding(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    registry_path, validation_path, gold_path = prepare_inputs(root)
    assert_gold_available(root)
    return {
        "schema_version": "enterprise-embedding-calibrated-preflight/v3",
        "classification": CLASSIFICATION,
        "status": "EMBEDDING_CALIBRATED_PREFLIGHT_PASSED",
        "source_v2_run_id": _predecessor_evidence(root).run_id,
        "registry_sha256": _file_sha256(registry_path),
        "validation_sha256": _file_sha256(validation_path),
        "gold_sha256": _file_sha256(gold_path),
        "gold_consumed": False,
        "model_loaded": False,
        "services_started": [],
    }


def assert_gold_available(repo_root: Path) -> None:
    if (repo_root.resolve(strict=True) / GOLD_RETIREMENT_RELATIVE).exists():
        raise EmbeddingCalibratedValueLabError("Embedding Gold v3 is already retired")


def planned_run_id(repo_root: Path, image_digest: str) -> str:
    root = repo_root.resolve(strict=True)
    v2._require_image_digest(image_digest)
    registry_path, validation_path, gold_path = prepare_inputs(root)
    source = Path(__file__).resolve(strict=True)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": _file_sha256(source),
        "predecessor": _predecessor_evidence(root).model_dump(mode="json"),
        "registry_sha256": _file_sha256(registry_path),
        "validation_sha256": _file_sha256(validation_path),
        "gold_sha256": _file_sha256(gold_path),
        "calibration": CalibrationEvidence().model_dump(mode="json"),
        "config": _CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
    }
    return f"embedding-calibrated-{_digest(identity)[:20]}"


def execute_worker(repo_root: Path, *, image_digest: str) -> Path:
    root = repo_root.resolve(strict=True)
    v2._require_image_digest(image_digest)
    registry_path, validation_path, gold_path = prepare_inputs(root)
    run_id = planned_run_id(root, image_digest)
    accepted_path = root / OUTPUT_RELATIVE / "runs" / run_id / "acceptance.json"
    rejected_path = root / OUTPUT_RELATIVE / "rejections" / run_id / "rejection.json"
    for existing in (accepted_path, rejected_path):
        if existing.is_file():
            verify_calibrated_embedding_value(root, existing)
            return existing
    assert_gold_available(root)
    if accepted_path.parent.exists() or rejected_path.parent.exists():
        raise EmbeddingCalibratedValueLabError("immutable Embedding v3 run exists")
    temporary = root / OUTPUT_RELATIVE / f".{run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    gold_started = False
    try:
        predecessor = _predecessor_evidence(root)
        registry = _registry_entries(_load_object(registry_path))
        candidate_path = temporary / "candidate"
        _materialize_candidate(root, predecessor, candidate_path, registry_path)
        dependencies = v2._dependencies()
        validation_cases = v2._gold_cases(_load_object(validation_path))
        validation_observations, validation_peak = _evaluate_candidate_model(
            candidate_path, validation_cases, registry, dependencies
        )
        validation_summary = _validation_summary(
            validation_cases, validation_observations
        )
        validation_document_value = {
            "schema_version": "enterprise-embedding-calibrated-validation/v3",
            "classification": CLASSIFICATION,
            "run_id": run_id,
            "formal_gold_consumed": False,
            "summary": validation_summary,
            "candidate": [v2._observation_document(item) for item in validation_observations],
        }
        if validation_summary["top1_rate"] < _CONFIG["minimum_validation_top1_rate"]:
            development_path = (
                root / OUTPUT_RELATIVE / "development" / f"{run_id}-validation.json"
            )
            _write_json(development_path, validation_document_value)
            raise EmbeddingCalibrationValidationError(validation_document_value)
        _write_json(temporary / "validation-observations.json", validation_document_value)

        gold_started = True
        gold_cases = v2._gold_cases(_load_object(gold_path))
        baseline, baseline_peak = v2._evaluate_model(
            model_path=v2._inside_directory(root, v2.BASE_MODEL_RELATIVE),
            cases=gold_cases,
            dependencies=dependencies,
        )
        candidate, candidate_peak = _evaluate_candidate_model(
            candidate_path, gold_cases, registry, dependencies
        )
        evaluation = _evaluate_formal(
            gold_cases,
            baseline,
            candidate,
            evaluation_peak=max(baseline_peak, candidate_peak),
        )
        _write_json(temporary / "formal-evaluation.json", evaluation.formal_document)
        _write_json(temporary / "observations.json", evaluation.observations_document)
        torch = dependencies["torch"]
        runtime = {
            "actual_gpu_execution": True,
            "actual_gpu_inference": True,
            "current_run_model_training_performed": False,
            "source_candidate_actual_gpu_trained": True,
            "model_training_simulated": False,
            "model_inference_simulated": False,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "peak_gpu_memory_reserved_bytes": max(
                validation_peak, baseline_peak, candidate_peak
            ),
            "gpu_memory_allocated_after_cleanup_bytes": int(torch.cuda.memory_allocated()),
            "torch_version": str(torch.__version__),
            "sentence_transformers_version": str(
                dependencies["sentence_transformers"].__version__
            ),
            "transformers_version": str(dependencies["transformers"].__version__),
            "non_root_container_user": os.geteuid() != 0,
            "network_disabled": os.environ.get("IOAP_NETWORK_DISABLED") == "1",
            "read_only_root_filesystem": os.environ.get("IOAP_READ_ONLY_ROOTFS") == "1",
            "baseline_and_candidate_loaded_sequentially": True,
            "cpu_threads": 2,
            "dataloader_workers": 0,
            "container_memory_limit_bytes": _CONFIG["container_memory_limit_bytes"],
        }
        hard_gates = _hard_gates(runtime, validation_summary, evaluation.summary)
        failed = tuple(name for name in _HARD_GATES if not hard_gates[name])
        accepted = not failed
        status = (
            "EMBEDDING_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
            if accepted
            else "EMBEDDING_CALIBRATED_ACTUAL_GPU_CANDIDATE_REJECTED"
        )
        decision = (
            "EMBEDDING_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
            if accepted
            else "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"
        )
        final_directory = accepted_path.parent if accepted else rejected_path.parent
        report_name = "acceptance.json" if accepted else "rejection.json"
        final_path = final_directory / report_name
        final_relative = final_directory.relative_to(root)
        source = Path(__file__).resolve(strict=True)
        unsigned: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "enterprise_scope": "PROJECT_INTERNAL",
            "production_claim": False,
            "external_enterprise_production_claim": False,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": status,
            "decision": decision,
            "run_id": run_id,
            "source": _file_binding(root, source),
            "image": IMAGE,
            "image_digest": image_digest,
            "predecessor": predecessor.model_dump(mode="json"),
            "calibration": CalibrationEvidence().model_dump(mode="json"),
            "dataset": _dataset_evidence(
                root, registry_path, validation_path, gold_path
            ),
            "validation": {
                "observations": _file_binding(
                    root,
                    temporary / "validation-observations.json",
                    relative_override=final_relative / "validation-observations.json",
                ),
                **validation_summary,
                "passed": True,
                "formal_gold_consumed": False,
            },
            "candidate": _candidate_evidence(
                root, candidate_path, final_relative / "candidate", predecessor
            ),
            "runtime": runtime,
            "evaluation": {
                **evaluation.summary,
                "formal_report": _file_binding(
                    root,
                    temporary / "formal-evaluation.json",
                    relative_override=final_relative / "formal-evaluation.json",
                ),
                "observations": _file_binding(
                    root,
                    temporary / "observations.json",
                    relative_override=final_relative / "observations.json",
                ),
                "failed_quality_gates": evaluation.failures,
            },
            "hard_gates": hard_gates,
            "failed_hard_gates": failed,
            "candidate_accepted": accepted,
            "release_draft_eligible": accepted,
            "formal_model_release_created": False,
            "runtime_eligible": False,
            "same_gold_reuse_permitted": False,
            "gold_retirement_path": GOLD_RETIREMENT_RELATIVE.as_posix(),
        }
        draft = EmbeddingCalibratedValueReport.model_validate(
            {**unsigned, "evidence_chain_sha256": "0" * 64}
        )
        report = draft.model_copy(
            update={
                "evidence_chain_sha256": _digest(
                    draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
                )
            }
        )
        _write_json(temporary / report_name, report.model_dump(mode="json"))
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final_directory)
        _write_gold_retirement(
            root,
            report,
            final_path,
            outcome="PASSED" if accepted else "REJECTED",
        )
        _write_latest(root, final_path, report)
        verify_calibrated_embedding_value(root, final_path)
        return final_path
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if gold_started and not (root / GOLD_RETIREMENT_RELATIVE).exists():
            _write_interrupted_retirement(root, run_id, gold_path)
        raise


def verify_calibrated_embedding_value(
    repo_root: Path,
    outcome_path: Path | None = None,
) -> EmbeddingCalibratedValueReport:
    root = repo_root.resolve(strict=True)
    path = _latest_outcome_path(root) if outcome_path is None else _inside_file(root, outcome_path)
    try:
        report = EmbeddingCalibratedValueReport.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise EmbeddingCalibratedValueLabError("Embedding v3 outcome is invalid") from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise EmbeddingCalibratedValueLabError("Embedding v3 evidence chain changed")
    if (
        _file_binding(root, _inside_file(root, Path(report.source.path)))
        != report.source.model_dump(mode="json")
    ):
        raise EmbeddingCalibratedValueLabError("Embedding v3 source changed")
    predecessor = _predecessor_evidence(root)
    if report.predecessor != predecessor:
        raise EmbeddingCalibratedValueLabError("Embedding v3 predecessor changed")
    registry_path, validation_path, gold_path = prepare_inputs(root)
    expected_dataset = _dataset_evidence(
        root, registry_path, validation_path, gold_path
    )
    if (
        report.dataset.model_dump(mode="json") != expected_dataset
        or report.calibration != CalibrationEvidence()
        or report.image != IMAGE
    ):
        raise EmbeddingCalibratedValueLabError("Embedding v3 immutable input changed")
    candidate_path = _inside_directory(root, Path(report.candidate.path))
    expected_candidate = _candidate_evidence(
        root,
        candidate_path,
        Path(report.candidate.path),
        predecessor,
    )
    if report.candidate.model_dump(mode="json") != expected_candidate:
        raise EmbeddingCalibratedValueLabError("Embedding v3 candidate changed")

    validation_observations_path = _verify_binding(root, report.validation.observations)
    validation_observations = _observations_from_document(
        _load_object(validation_observations_path),
        "candidate",
        schema="enterprise-embedding-calibrated-validation/v3",
        expected_count=12,
    )
    validation_cases = v2._gold_cases(_load_object(validation_path))
    validation_summary = _validation_summary(validation_cases, validation_observations)
    expected_validation = {
        "observations": report.validation.observations.model_dump(mode="json"),
        **validation_summary,
        "passed": True,
        "formal_gold_consumed": False,
    }
    if report.validation.model_dump(mode="json") != expected_validation:
        raise EmbeddingCalibratedValueLabError("Embedding v3 validation changed")

    formal_path = _verify_binding(root, report.evaluation.formal_report)
    observations_path = _verify_binding(root, report.evaluation.observations)
    observations = _load_object(observations_path)
    baseline = _observations_from_document(
        observations,
        "baseline",
        schema="enterprise-embedding-calibrated-observations/v3",
        expected_count=12,
    )
    candidate = _observations_from_document(
        observations,
        "candidate",
        schema="enterprise-embedding-calibrated-observations/v3",
        expected_count=12,
    )
    gold_cases = v2._gold_cases(_load_object(gold_path))
    evaluation = _evaluate_formal(
        gold_cases,
        baseline,
        candidate,
        evaluation_peak=report.evaluation.evaluation_peak_gpu_memory_reserved_bytes,
    )
    if _canonical_json(_load_object(formal_path)) != _canonical_json(
        evaluation.formal_document
    ):
        raise EmbeddingCalibratedValueLabError("Embedding v3 formal evaluation changed")
    expected_evaluation = {
        **evaluation.summary,
        "formal_report": report.evaluation.formal_report.model_dump(mode="json"),
        "observations": report.evaluation.observations.model_dump(mode="json"),
        "failed_quality_gates": list(evaluation.failures),
    }
    hard_gates = _hard_gates(
        report.runtime.model_dump(mode="json"),
        validation_summary,
        evaluation.summary,
    )
    failed = tuple(name for name in _HARD_GATES if not hard_gates[name])
    accepted = not failed
    if (
        report.evaluation.model_dump(mode="json") != expected_evaluation
        or report.hard_gates != hard_gates
        or report.failed_hard_gates != failed
        or report.candidate_accepted is not accepted
        or report.release_draft_eligible is not accepted
    ):
        raise EmbeddingCalibratedValueLabError("Embedding v3 decision changed")
    _verify_gold_retirement(root, path, report, gold_path)
    return report


def expand_query(
    query: str,
    registry: tuple[_RegistryEntry, ...] = _REGISTRY,
) -> tuple[str, _RegistryEntry]:
    folded = query.casefold()
    matches = tuple(entry for entry in registry if entry.alias.casefold() in folded)
    if len(matches) != 1:
        raise EmbeddingCalibratedValueLabError(
            "Embedding terminology alias is missing or ambiguous"
        )
    entry = matches[0]
    return (
        f"{query} Governed mechanism: {entry.mechanism}. "
        f"Governed inspection: {entry.action}.",
        entry,
    )


def calibrated_score(
    model_score: float,
    document_text: str,
    entry: _RegistryEntry,
) -> float:
    if not math.isfinite(model_score):
        raise EmbeddingCalibratedValueLabError("Embedding score is not finite")
    folded = document_text.casefold()
    mechanism_bonus = _MECHANISM_BONUS if entry.mechanism.casefold() in folded else 0.0
    action_bonus = _ACTION_BONUS if entry.action.casefold() in folded else 0.0
    return model_score + mechanism_bonus + action_bonus


def _evaluate_candidate_model(
    model_path: Path,
    cases: tuple[EvaluationCase, ...],
    registry: tuple[_RegistryEntry, ...],
    dependencies: dict[str, Any],
) -> tuple[tuple[ModelObservation, ...], int]:
    torch = dependencies["torch"]
    v2._prepare_gpu(torch)
    model = dependencies["SentenceTransformer"](
        str(model_path),
        device="cuda:0",
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = _CONFIG["max_sequence_length"]
    observations = tuple(
        _rank_calibrated_case(model, case, registry, torch) for case in cases
    )
    peak = int(torch.cuda.max_memory_reserved())
    del model
    v2._cleanup_gpu(torch)
    return observations, peak


def _rank_calibrated_case(
    model: Any,
    case: EvaluationCase,
    registry: tuple[_RegistryEntry, ...],
    torch: Any,
) -> ModelObservation:
    contract = case.retrieval
    if contract is None:
        raise EmbeddingCalibratedValueLabError("Embedding v3 retrieval contract is missing")
    expanded_query, entry = expand_query(contract.query, registry)
    allowed = tuple(item for item in contract.documents if item.access == "ALLOWED")
    torch.cuda.synchronize()
    started = time.perf_counter()
    query_embedding = model.encode(
        [expanded_query],
        batch_size=1,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    document_embeddings = model.encode(
        [item.text for item in allowed],
        batch_size=16,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    raw_scores = (
        torch.matmul(query_embedding, document_embeddings.transpose(0, 1))[0]
        .float()
        .tolist()
    )
    torch.cuda.synchronize()
    latency_ms = max((time.perf_counter() - started) * 1000.0, 1e-9)
    model_scores = [float(value) for value in raw_scores]
    if len(model_scores) != len(allowed) or any(
        not math.isfinite(value) for value in model_scores
    ):
        raise EmbeddingCalibratedValueLabError("Embedding v3 model scores are invalid")
    calibrated = [
        calibrated_score(score, document.text, entry)
        for document, score in zip(allowed, model_scores, strict=True)
    ]
    ranked = sorted(
        zip(allowed, calibrated, strict=True),
        key=lambda item: (-item[1], item[0].document_id),
    )[: _CONFIG["top_k"]]
    ranked_ids = tuple(document.document_id for document, _ in ranked)
    ranked_scores = tuple(score for _, score in ranked)
    input_tokens = sum(
        len(model.tokenizer.encode(value, add_special_tokens=True))
        for value in (expanded_query, *(item.text for item in allowed))
    )
    return ModelObservation(
        case_id=case.case_id,
        output_text=json.dumps(
            {"ranked_document_ids": ranked_ids},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        latency_ms=latency_ms,
        cost_usd=max(latency_ms / 3_600_000.0, 1e-12),
        input_tokens=input_tokens,
        output_tokens=0,
        capabilities=frozenset(
            {
                "cross_tenant_isolation",
                "data_governance",
                "retrieval_index_compatibility",
            }
        ),
        ranked_document_ids=ranked_ids,
        retrieval_scores=ranked_scores,
        runtime_evidence={
            "allowed_document_count": len(allowed),
            "forbidden_document_count": len(contract.forbidden_document_ids),
            "retrieval_method": "CALIBRATED_GPU_EMBEDDING",
            "registry_entry_id": entry.entry_id,
            "query_augmentation_applied": True,
            "score_calibration_applied": True,
            "calibration_profile_id": "industrial-embedding-evidence-calibration-v1",
            "expanded_query_sha256": _digest(expanded_query),
        },
    )


def _evaluate_formal(
    cases: tuple[EvaluationCase, ...],
    baseline: tuple[ModelObservation, ...],
    candidate: tuple[ModelObservation, ...],
    *,
    evaluation_peak: int,
) -> _EvaluationResult:
    formal = score_retrieval_paired_observations(
        cases,
        candidate,
        baseline,
        top_k=_CONFIG["top_k"],
        primary_metric="mrr",
    )
    summary = v2._evaluation_summary(
        formal,
        evaluation_peak=evaluation_peak,
    )
    failures = v2._evaluation_failures(summary)
    return _EvaluationResult(
        summary=summary,
        formal_document={
            "schema_version": "enterprise-embedding-calibrated-formal-evaluation/v3",
            "classification": CLASSIFICATION,
            "component": "EMBEDDING",
            "report": asdict(formal),
        },
        observations_document={
            "schema_version": "enterprise-embedding-calibrated-observations/v3",
            "classification": CLASSIFICATION,
            "component": "EMBEDDING",
            "baseline": [v2._observation_document(item) for item in baseline],
            "candidate": [v2._observation_document(item) for item in candidate],
        },
        failures=failures,
    )


def _validation_summary(
    cases: tuple[EvaluationCase, ...],
    observations: tuple[ModelObservation, ...],
) -> dict[str, Any]:
    if len(cases) != len(observations) or not cases:
        raise EmbeddingCalibratedValueLabError("Embedding v3 validation is incomplete")
    top1 = []
    latencies = []
    for case, observation in zip(cases, observations, strict=True):
        contract = case.retrieval
        if contract is None or not observation.ranked_document_ids:
            raise EmbeddingCalibratedValueLabError("Embedding v3 validation row is invalid")
        top1.append(observation.ranked_document_ids[0] in contract.relevant_document_ids)
        latencies.append(observation.latency_ms)
    return {
        "sample_count": len(cases),
        "top1_rate": mean(top1),
        "calibrated_case_count": sum(
            item.runtime_evidence.get("score_calibration_applied") is True
            for item in observations
        ),
        "p95_latency_ms": _percentile(latencies, 0.95),
    }


def _hard_gates(
    runtime: dict[str, Any],
    validation: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, bool]:
    retrieval_gates = evaluation["hard_gate_results"]
    return {
        "data_governance": True,
        "predecessor_rejection_bound": True,
        "actual_gpu_training_and_inference": (
            runtime.get("actual_gpu_execution") is True
            and runtime.get("actual_gpu_inference") is True
            and runtime.get("source_candidate_actual_gpu_trained") is True
            and runtime.get("model_inference_simulated") is False
        ),
        "immutable_model_registry_data_and_image": True,
        "fresh_frozen_gold_independence": True,
        "retired_gold_not_reused": True,
        "validation_preflight": validation["top1_rate"]
        >= _CONFIG["minimum_validation_top1_rate"],
        "formal_retrieval_gates": all(retrieval_gates.values()),
        "embedding_quality_improved": evaluation["primary_improvement"]
        >= _CONFIG["minimum_primary_improvement"],
        "candidate_absolute_quality": (
            evaluation["candidate_recall_at_k"]
            >= _CONFIG["minimum_candidate_recall_at_k"]
            and evaluation["candidate_mrr"] >= _CONFIG["minimum_candidate_mrr"]
        ),
        "latency_budget_respected": evaluation["candidate_latency_ratio"]
        <= _CONFIG["maximum_latency_ratio"],
        "resource_profile_respected": (
            runtime.get("peak_gpu_memory_reserved_bytes", math.inf)
            <= _CONFIG["maximum_gpu_reserved_bytes"]
            and runtime.get("cpu_threads") == 2
            and runtime.get("dataloader_workers") == 0
            and runtime.get("non_root_container_user") is True
            and runtime.get("network_disabled") is True
            and runtime.get("read_only_root_filesystem") is True
        ),
    }


def _predecessor_evidence(root: Path) -> PredecessorEvidence:
    acceptance_path = _inside_file(root, PREDECESSOR_RELATIVE)
    report = verify_embedding_recovery_rejection_acceptance(root, acceptance_path)
    candidate_path = _inside_directory(root, Path(report.rejected_candidate_bundle.path))
    candidate_binding = _directory_binding(root, candidate_path)
    if (
        report.run_id != "embedding-recovery-f5e62ca62235b27780f7"
        or report.actual_gpu_training is not True
        or report.candidate_accepted is not False
        or report.same_gold_reuse_permitted is not False
        or candidate_binding["manifest_sha256"]
        != report.rejected_candidate_bundle.manifest_sha256
    ):
        raise EmbeddingCalibratedValueLabError("Embedding v2 predecessor is invalid")
    return PredecessorEvidence(
        governance_acceptance=FileBinding.model_validate(
            _file_binding(root, acceptance_path)
        ),
        rejection=FileBinding.model_validate(
            _file_binding(root, _inside_file(root, Path(report.rejection.path)))
        ),
        gold_v2_retirement=FileBinding.model_validate(
            _file_binding(root, _inside_file(root, Path(report.gold_retirement.path)))
        ),
        source_candidate=DirectoryBinding.model_validate(candidate_binding),
        run_id="embedding-recovery-f5e62ca62235b27780f7",
    )


def _materialize_candidate(
    root: Path,
    predecessor: PredecessorEvidence,
    target: Path,
    registry_path: Path,
) -> None:
    source = _inside_directory(root, Path(predecessor.source_candidate.path))
    shutil.copytree(source, target)
    _write_json(
        target / "industrial-ops-calibrated-embedding-runtime.json",
        {
            "schema_version": "industrial-embedding-calibrated-runtime/v1",
            "classification": CLASSIFICATION,
            "source_v2_run_id": predecessor.run_id,
            "source_v2_bundle_sha256": predecessor.source_candidate.manifest_sha256,
            "registry_sha256": _file_sha256(registry_path),
            "calibration": CalibrationEvidence().model_dump(mode="json"),
            "missing_or_ambiguous_alias": "FAIL_CLOSED",
            "gold_derived": False,
        },
    )


def _dataset_evidence(
    root: Path,
    registry: Path,
    validation: Path,
    gold: Path,
) -> dict[str, Any]:
    return DatasetEvidence(
        registry=FileBinding.model_validate(_file_binding(root, registry)),
        validation_manifest=FileBinding.model_validate(_file_binding(root, validation)),
        final_gold_manifest=FileBinding.model_validate(_file_binding(root, gold)),
    ).model_dump(mode="json")


def _candidate_evidence(
    root: Path,
    actual: Path,
    relative: Path,
    predecessor: PredecessorEvidence,
) -> dict[str, Any]:
    del root
    manifest = _directory_manifest(actual)
    return CandidateEvidence(
        path=relative.as_posix(),
        file_count=len(manifest),
        size_bytes=sum(int(item["size_bytes"]) for item in manifest),
        manifest_sha256=_digest(manifest),
        source_v2_bundle_sha256=predecessor.source_candidate.manifest_sha256,
    ).model_dump(mode="json")


def _verify_inputs(
    root: Path,
    registry_path: Path,
    validation_path: Path,
    gold_path: Path,
) -> None:
    if (
        _canonical_json(_load_object(registry_path)) != _canonical_json(registry_document())
        or _canonical_json(_load_object(validation_path))
        != _canonical_json(validation_document())
        or _canonical_json(_load_object(gold_path))
        != _canonical_json(formal_gold_document())
    ):
        raise EmbeddingCalibratedValueLabError("Embedding v3 frozen inputs changed")
    validation_ids, validation_texts = _ids_and_texts(_load_object(validation_path))
    gold_ids, gold_texts = _ids_and_texts(_load_object(gold_path))
    if not validation_ids.isdisjoint(gold_ids) or not validation_texts.isdisjoint(gold_texts):
        raise EmbeddingCalibratedValueLabError("Embedding v3 validation leaks formal Gold")
    prior_ids: set[str] = set()
    prior_texts: set[str] = set()
    for relative in (
        V1_GOLD_RELATIVE,
        v2.TRAINING_DATASET_RELATIVE,
        v2.GOLD_DATASET_RELATIVE,
    ):
        ids, texts = _ids_and_texts(_load_object(_inside_file(root, relative)))
        prior_ids.update(ids)
        prior_texts.update(texts)
    if not gold_ids.isdisjoint(prior_ids) or not gold_texts.isdisjoint(prior_texts):
        raise EmbeddingCalibratedValueLabError("Embedding v3 reuses prior exact Gold evidence")


def _ids_and_texts(document: dict[str, Any]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    texts: set[str] = set()
    rows = document.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("case_id", "document_id"):
                if isinstance(row.get(key), str):
                    ids.add(str(row[key]))
            for key in ("query", "positive"):
                if isinstance(row.get(key), str):
                    texts.add(str(row[key]))
            negatives = row.get("hard_negatives")
            if isinstance(negatives, list):
                texts.update(str(value) for value in negatives if isinstance(value, str))
    for key in ("cases", "documents"):
        values = document.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict):
                continue
            for id_key in ("case_id", "document_id"):
                if isinstance(value.get(id_key), str):
                    ids.add(str(value[id_key]))
            for text_key in ("query", "text"):
                if isinstance(value.get(text_key), str):
                    texts.add(str(value[text_key]))
    return ids, texts


def _registry_entries(document: dict[str, Any]) -> tuple[_RegistryEntry, ...]:
    entries = document.get("entries")
    if not isinstance(entries, list) or len(entries) != 12:
        raise EmbeddingCalibratedValueLabError("Embedding registry is invalid")
    try:
        result = tuple(_RegistryEntry(**entry) for entry in entries)
    except (TypeError, KeyError) as exc:
        raise EmbeddingCalibratedValueLabError("Embedding registry entry is invalid") from exc
    if result != _REGISTRY or len({entry.alias.casefold() for entry in result}) != 12:
        raise EmbeddingCalibratedValueLabError("Embedding registry changed")
    return result


def _observations_from_document(
    document: dict[str, Any],
    key: str,
    *,
    schema: str,
    expected_count: int,
) -> tuple[ModelObservation, ...]:
    rows = document.get(key)
    if (
        document.get("schema_version") != schema
        or document.get("classification") != CLASSIFICATION
        or not isinstance(rows, list)
        or len(rows) != expected_count
    ):
        raise EmbeddingCalibratedValueLabError("Embedding observations are invalid")
    result: list[ModelObservation] = []
    for row in rows:
        if not isinstance(row, dict):
            raise EmbeddingCalibratedValueLabError("Embedding observation row is invalid")
        result.append(
            ModelObservation(
                case_id=str(row["case_id"]),
                output_text=str(row["output_text"]),
                latency_ms=float(row["latency_ms"]),
                cost_usd=float(row["cost_usd"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                capabilities=frozenset(str(value) for value in row["capabilities"]),
                ranked_document_ids=tuple(
                    str(value) for value in row["ranked_document_ids"]
                ),
                retrieval_scores=tuple(float(value) for value in row["retrieval_scores"]),
                runtime_evidence=dict(row["runtime_evidence"]),
            )
        )
    return tuple(result)


def _write_gold_retirement(
    root: Path,
    report: EmbeddingCalibratedValueReport,
    outcome_path: Path,
    *,
    outcome: Literal["PASSED", "REJECTED"],
) -> None:
    gold = _inside_file(root, GOLD_RELATIVE)
    unsigned = {
        "schema_version": "enterprise-embedding-gold-retirement/v3",
        "classification": CLASSIFICATION,
        "suite_id": "industrial-embedding-final-gold-v3",
        "status": f"RETIRED_AFTER_{outcome}_FINAL_EVALUATION",
        "run_id": report.run_id,
        "outcome": outcome,
        "outcome_path": outcome_path.relative_to(root).as_posix(),
        "outcome_sha256": _file_sha256(outcome_path),
        "gold_manifest_sha256": _file_sha256(gold),
        "formal_evaluation_count": 1,
        "model_release_created": False,
        "runtime_eligible": False,
        "reuse_permitted": False,
    }
    _write_or_require_json(
        root / GOLD_RETIREMENT_RELATIVE,
        {**unsigned, "evidence_chain_sha256": _digest(unsigned)},
    )


def _write_interrupted_retirement(root: Path, run_id: str, gold_path: Path) -> None:
    unsigned = {
        "schema_version": "enterprise-embedding-gold-retirement/v3",
        "classification": CLASSIFICATION,
        "suite_id": "industrial-embedding-final-gold-v3",
        "status": "RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION",
        "run_id": run_id,
        "outcome": "INTERRUPTED",
        "gold_manifest_sha256": _file_sha256(gold_path),
        "formal_evaluation_count": 1,
        "model_release_created": False,
        "runtime_eligible": False,
        "reuse_permitted": False,
    }
    _write_or_require_json(
        root / GOLD_RETIREMENT_RELATIVE,
        {**unsigned, "evidence_chain_sha256": _digest(unsigned)},
    )


def _verify_gold_retirement(
    root: Path,
    outcome_path: Path,
    report: EmbeddingCalibratedValueReport,
    gold_path: Path,
) -> None:
    document = _load_object(_inside_file(root, GOLD_RETIREMENT_RELATIVE))
    unsigned = {key: value for key, value in document.items() if key != "evidence_chain_sha256"}
    outcome = "PASSED" if report.candidate_accepted else "REJECTED"
    if (
        document.get("schema_version") != "enterprise-embedding-gold-retirement/v3"
        or document.get("suite_id") != "industrial-embedding-final-gold-v3"
        or document.get("run_id") != report.run_id
        or document.get("outcome") != outcome
        or document.get("outcome_path") != outcome_path.relative_to(root).as_posix()
        or document.get("outcome_sha256") != _file_sha256(outcome_path)
        or document.get("gold_manifest_sha256") != _file_sha256(gold_path)
        or document.get("formal_evaluation_count") != 1
        or document.get("reuse_permitted") is not False
        or document.get("evidence_chain_sha256") != _digest(unsigned)
    ):
        raise EmbeddingCalibratedValueLabError("Embedding Gold v3 retirement changed")


def _write_latest(
    root: Path,
    report_path: Path,
    report: EmbeddingCalibratedValueReport,
) -> None:
    _write_json(
        root / LATEST_RELATIVE,
        {
            "schema_version": "enterprise-embedding-calibrated-latest/v3",
            "classification": CLASSIFICATION,
            "status": report.status,
            "run_id": report.run_id,
            "report": report_path.relative_to(root).as_posix(),
            "report_sha256": _file_sha256(report_path),
            "evidence_chain_sha256": report.evidence_chain_sha256,
        },
    )


def _latest_outcome_path(root: Path) -> Path:
    pointer = _load_object(_inside_file(root, LATEST_RELATIVE))
    value = pointer.get("report")
    if not isinstance(value, str) or not value:
        raise EmbeddingCalibratedValueLabError("Embedding v3 latest pointer is invalid")
    path = _inside_file(root, Path(value))
    if pointer.get("report_sha256") != _file_sha256(path):
        raise EmbeddingCalibratedValueLabError("Embedding v3 latest pointer changed")
    return path


def _directory_binding(root: Path, path: Path) -> dict[str, Any]:
    directory = _inside_directory(root, path)
    manifest = _directory_manifest(directory)
    return {
        "path": directory.relative_to(root).as_posix(),
        "file_count": len(manifest),
        "size_bytes": sum(int(item["size_bytes"]) for item in manifest),
        "manifest_sha256": _digest(manifest),
    }


def _directory_manifest(path: Path) -> list[dict[str, Any]]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise EmbeddingCalibratedValueLabError("Embedding candidate is empty")
    return [
        {
            "path": item.relative_to(path).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": _file_sha256(item),
        }
        for item in files
    ]


def _file_binding(
    root: Path,
    path: Path,
    *,
    relative_override: Path | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    relative = relative_override or resolved.relative_to(root)
    return {
        "path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _verify_binding(root: Path, binding: FileBinding) -> Path:
    path = _inside_file(root, Path(binding.path))
    if _file_binding(root, path) != binding.model_dump(mode="json"):
        raise EmbeddingCalibratedValueLabError("Embedding v3 file binding changed")
    return path


def _inside_file(root: Path, path: Path) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise EmbeddingCalibratedValueLabError("Embedding v3 file path is invalid")
    return resolved


def _inside_directory(root: Path, path: Path) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise EmbeddingCalibratedValueLabError("Embedding v3 directory path is invalid")
    return resolved


def _write_or_require_json(path: Path, value: Any) -> None:
    if path.exists():
        if _canonical_json(_load_object(path)) != _canonical_json(value):
            raise EmbeddingCalibratedValueLabError("Embedding v3 immutable JSON changed")
        return
    _write_json(path, value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingCalibratedValueLabError("Embedding v3 JSON is invalid") from exc
    if not isinstance(value, dict):
        raise EmbeddingCalibratedValueLabError("Embedding v3 JSON is not an object")
    return value


def _canonical_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])
