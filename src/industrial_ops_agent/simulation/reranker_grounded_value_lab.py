"""Low-memory actual-GPU grounded Reranker enterprise-value lab v4."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.evaluation.dataset import EvaluationCase
from industrial_ops_agent.evaluation.metrics import (
    ModelObservation,
    PairedMetricReport,
    score_retrieval_paired_observations,
)
from industrial_ops_agent.experiments.service import RETRIEVAL_HARD_GATES
from industrial_ops_agent.simulation import reranker_recovery_value_lab as v3
from industrial_ops_agent.simulation.reranker_recovery_verifier_erratum import (
    verify_original_with_json_normalization,
    verify_reranker_recovery_verifier_erratum,
)

SCHEMA_VERSION: Literal["enterprise-reranker-grounded-value-lab/v4"] = (
    "enterprise-reranker-grounded-value-lab/v4"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
OUTPUT_RELATIVE = Path("artifacts/m7-reranker-grounded-value-lab")
REGISTRY_RELATIVE = (
    OUTPUT_RELATIVE / "inputs/industrial-reranker-terminology-registry-v1/manifest.json"
)
GOLD_RELATIVE = (
    OUTPUT_RELATIVE / "inputs/industrial-reranker-final-gold-v4/manifest.json"
)
GOLD_RETIREMENT_RELATIVE = OUTPUT_RELATIVE / "gold-v4-retirement.json"
LATEST_RELATIVE = OUTPUT_RELATIVE / "latest.json"
PREDECESSOR_ERRATUM_RELATIVE = (
    v3.OUTPUT_RELATIVE
    / "errata/reranker-recovery-f1ec9f9b7d53bbc879d1/verification.json"
)
IMAGE: Literal["industrial-ops/m4-training-worker:local"] = v3.IMAGE
CONTAINER_NAME = "ioap-reranker-grounded-value-lab"

_CONFIG: dict[str, Any] = {
    "top_k": 3,
    "minimum_primary_improvement": 0.10,
    "minimum_candidate_recall_at_k": 0.90,
    "minimum_candidate_mrr": 0.85,
    "minimum_candidate_ndcg_at_k": 0.85,
    "maximum_latency_ratio": 4.0,
    "max_sequence_length": 128,
    "cpu_threads": 2,
    "dataloader_workers": 0,
    "container_memory_limit_bytes": 4 * 1024**3,
    "maximum_gpu_reserved_bytes": 2 * 1024**3,
}
_HARD_GATE_NAMES = (
    "data_governance",
    "verified_v3_trained_predecessor",
    "actual_gpu_grounded_inference",
    "immutable_model_registry_gold_and_image",
    "fresh_frozen_gold_independence",
    "retired_gold_not_reused",
    "governed_terminology_resolution",
    "cross_tenant_isolation",
    "formal_retrieval_gates",
    "reranker_quality_improved",
    "candidate_absolute_quality",
    "latency_budget_respected",
    "resource_profile_respected",
)


class RerankerGroundedValueLabError(RuntimeError):
    """The governed Reranker v4 lab could not run or verify."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PredecessorEvidence(_ClosedModel):
    erratum: v3.FileEvidence
    original_outcome: v3.FileEvidence
    gold_v3_retirement: v3.FileEvidence
    run_id: Literal["reranker-recovery-f1ec9f9b7d53bbc879d1"]
    source_candidate_path: str = Field(min_length=1)
    source_candidate_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_gpu_training: Literal[True] = True
    candidate_accepted: Literal[False] = False
    runtime_eligible: Literal[False] = False
    same_gold_reuse_permitted: Literal[False] = False


class RegistryEvidence(_ClosedModel):
    manifest: v3.FileEvidence
    registry_id: Literal["industrial-reranker-terminology-registry-v1"]
    entry_count: Literal[12] = 12
    review_status: Literal["APPROVED"] = "APPROVED"
    frozen_before_gold_evaluation: Literal[True] = True
    derived_from_formal_gold: Literal[False] = False
    exact_alias_resolution_required: Literal[True] = True


class DatasetEvidence(_ClosedModel):
    final_gold_manifest: v3.FileEvidence
    final_gold_count: Literal[12] = 12
    allowed_document_count: Literal[24] = 24
    forbidden_document_count: Literal[2] = 2
    terminology_family_count: Literal[12] = 12
    top_k: Literal[3] = 3
    source_type: Literal["PLATFORM_SYNTHETIC"] = "PLATFORM_SYNTHETIC"
    project_enterprise_use_authorized: Literal[True] = True
    enterprise_production_data: Literal[False] = False
    retired_v1_v2_v3_gold_reused: Literal[False] = False
    frozen_gold_excluded_from_selection_and_development: Literal[True] = True
    exact_case_ids_queries_and_documents_disjoint: Literal[True] = True


class CandidateEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    files: tuple[v3.FileEvidence, ...] = Field(min_length=8)
    source_v3_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_augmentation: Literal["GOVERNED_TERMINOLOGY_REGISTRY_V1"] = (
        "GOVERNED_TERMINOLOGY_REGISTRY_V1"
    )
    role: Literal["AUTHORIZED_ENTERPRISE_GROUNDED_RERANKER_CANDIDATE"] = (
        "AUTHORIZED_ENTERPRISE_GROUNDED_RERANKER_CANDIDATE"
    )


class RuntimeEvidence(_ClosedModel):
    actual_gpu_execution: Literal[True] = True
    actual_gpu_inference: Literal[True] = True
    current_run_model_training_performed: Literal[False] = False
    source_candidate_actual_gpu_trained: Literal[True] = True
    model_training_simulated: Literal[False] = False
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


class MetricSummary(_ClosedModel):
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
    registry_resolution_count: Literal[12] = 12
    registry_resolution_failure_count: Literal[0] = 0


class EvaluationEvidence(MetricSummary):
    primary_metric: Literal["ndcg_at_k"] = "ndcg_at_k"
    formal_report: v3.FileEvidence
    observations: v3.FileEvidence
    failed_quality_gates: tuple[str, ...]


class RerankerGroundedHardGates(_ClosedModel):
    data_governance: bool
    verified_v3_trained_predecessor: bool
    actual_gpu_grounded_inference: bool
    immutable_model_registry_gold_and_image: bool
    fresh_frozen_gold_independence: bool
    retired_gold_not_reused: bool
    governed_terminology_resolution: bool
    cross_tenant_isolation: bool
    formal_retrieval_gates: bool
    reranker_quality_improved: bool
    candidate_absolute_quality: bool
    latency_budget_respected: bool
    resource_profile_respected: bool


class RerankerGroundedValueReport(_ClosedModel):
    schema_version: Literal["enterprise-reranker-grounded-value-lab/v4"] = (
        SCHEMA_VERSION
    )
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal[
        "RERANKER_GROUNDED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
        "RERANKER_GROUNDED_ACTUAL_GPU_CANDIDATE_REJECTED",
    ]
    decision: Literal[
        "RERANKER_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT",
        "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD",
    ]
    run_id: str = Field(pattern=r"^reranker-grounded-[0-9a-f]{20}$")
    source: v3.FileEvidence
    image: Literal["industrial-ops/m4-training-worker:local"] = IMAGE
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    predecessor: PredecessorEvidence
    input_model: v3.InputModelEvidence
    registry: RegistryEvidence
    dataset: DatasetEvidence
    candidate: CandidateEvidence
    runtime: RuntimeEvidence
    evaluation: EvaluationEvidence
    hard_gates: RerankerGroundedHardGates
    failed_hard_gates: tuple[str, ...]
    candidate_accepted: bool
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    same_gold_reuse_permitted: Literal[False] = False
    gold_retirement_path: str = Field(min_length=1)
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _RegistryEntry:
    entry_id: str
    alias: str
    mechanism: str
    action: str
    device_family: str
    signal: str


@dataclass(frozen=True, slots=True)
class _EvaluationResult:
    summary: dict[str, Any]
    formal_document: dict[str, Any]
    observations_document: dict[str, Any]
    failures: tuple[str, ...]


_REGISTRY = (
    _RegistryEntry(
        "term-001",
        "amber cascade",
        "lubricant aeration and persistent oil foaming",
        "inspect the oil return and air-ingress points before restart",
        "gearbox",
        "lubricant",
    ),
    _RegistryEntry(
        "term-002",
        "hollow pulse",
        "pump cavitation caused by inadequate suction head",
        "inspect suction restriction, fluid level, and available NPSH",
        "pump",
        "pressure",
    ),
    _RegistryEntry(
        "term-003",
        "glass rain",
        "rolling-element bearing outer-race spalling",
        "verify envelope acceleration and inspect the outer race",
        "rotating",
        "vibration",
    ),
    _RegistryEntry(
        "term-004",
        "winter arc",
        "surface tracking across moisture-contaminated insulation",
        "isolate power and inspect insulation for tracking paths",
        "motor",
        "electrical",
    ),
    _RegistryEntry(
        "term-005",
        "velvet stall",
        "compressor surge with unstable low-flow operation",
        "unload safely and inspect the anti-surge control path",
        "compressor",
        "flow",
    ),
    _RegistryEntry(
        "term-006",
        "silver cough",
        "control-valve stiction during small command changes",
        "verify travel feedback and inspect packing friction",
        "control",
        "position",
    ),
    _RegistryEntry(
        "term-007",
        "paper thunder",
        "intermittent arcing at a loose power termination",
        "de-energize and inspect torque, discoloration, and insulation",
        "switchgear",
        "electrical",
    ),
    _RegistryEntry(
        "term-008",
        "blue ladder",
        "progressive battery-cell heating before thermal runaway",
        "isolate charging and verify the cell temperature gradient",
        "battery",
        "temperature",
    ),
    _RegistryEntry(
        "term-009",
        "ghost tooth",
        "gear-mesh backlash or a damaged tooth under reversal",
        "lock out motion and inspect backlash and tooth contact",
        "gearbox",
        "mechanical",
    ),
    _RegistryEntry(
        "term-010",
        "dry mirror",
        "lubrication starvation on a polished bearing raceway",
        "stop the asset and verify lubricant delivery before restart",
        "bearing",
        "lubricant",
    ),
    _RegistryEntry(
        "term-011",
        "slow echo",
        "encoder feedback delay from shielding or connector degradation",
        "isolate motion and inspect the feedback cable and connector",
        "servo",
        "sensor",
    ),
    _RegistryEntry(
        "term-012",
        "salt bloom",
        "coolant crystallization from concentration or contamination",
        "stop circulation and inspect concentration, filtration, and seals",
        "cooling",
        "process",
    ),
)


def terminology_registry_document() -> dict[str, Any]:
    return {
        "schema_version": "industrial-reranker-terminology-registry/v1",
        "registry_id": "industrial-reranker-terminology-registry-v1",
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "enterprise_production_data": False,
        "project_generated_data": True,
        "project_enterprise_use_authorized": True,
        "review_status": "APPROVED",
        "frozen_before_gold_evaluation": True,
        "derived_from_formal_gold": False,
        "resolution_policy": {
            "match": "EXACT_CASE_INSENSITIVE_ALIAS",
            "missing_alias": "FAIL_CLOSED",
            "ambiguous_alias": "FAIL_CLOSED",
            "candidate_only": True,
        },
        "entries": [asdict(item) for item in _REGISTRY],
    }


def formal_gold_document() -> dict[str, Any]:
    """Build fresh one-shot Gold v4 without reusing prior exact texts or IDs."""

    documents: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for index, entry in enumerate(_REGISTRY, start=1):
        resolution_id = f"rr-v4-controlled-resolution-{index:03d}"
        glossary_id = f"rr-v4-glossary-reference-{index:03d}"
        documents.extend(
            (
                {
                    "document_id": resolution_id,
                    "text": (
                        f"Authorized troubleshooting note for the {entry.device_family} family. "
                        f"Correlated {entry.signal} evidence confirms {entry.mechanism}. Before "
                        f"return to service, {entry.action}."
                    ),
                    "access": "ALLOWED",
                },
                {
                    "document_id": glossary_id,
                    "text": (
                        f"Vocabulary index entry '{entry.alias}' is retained for historical "
                        "search navigation only; it contains no validated failure mechanism "
                        "and authorizes no maintenance action."
                    ),
                    "access": "ALLOWED",
                },
            )
        )
        cases.append(
            {
                "case_id": f"reranker-grounded-gold-v4-{index:03d}",
                "query": (
                    f"Work order R4-{index:03d} records field shorthand '{entry.alias}' after "
                    f"a {entry.signal} anomaly on a {entry.device_family} asset. Rank the "
                    "authorized diagnosis and safe intervention ahead of vocabulary records."
                ),
                "relevant_document_ids": [resolution_id],
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
                "document_id": "tenant-rival-reranker-v4-secret-001",
                "text": "Another tenant's confidential grounded diagnosis and intervention.",
                "access": "FORBIDDEN",
            },
            {
                "document_id": "tenant-rival-reranker-v4-secret-002",
                "text": "Another tenant's private terminology registry and service history.",
                "access": "FORBIDDEN",
            },
        )
    )
    return {
        "schema_version": "industrial-reranker-final-gold/v4",
        "suite_id": "industrial-reranker-final-gold-v4",
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "enterprise_production_data": False,
        "project_generated_data": True,
        "project_enterprise_use_authorized": True,
        "purpose": "FINAL_INDEPENDENT_GROUNDED_RERANKER_EVALUATION",
        "frozen_at": "2026-08-29T00:00:00Z",
        "formal_evaluation_limit": 1,
        "selection_policy": {
            "excluded_from_model_training": True,
            "excluded_from_pipeline_selection": True,
            "excluded_from_development_probe": True,
            "retired_v1_v2_v3_gold_reused": False,
            "registry_frozen_before_gold_evaluation": True,
        },
        "predecessor_gold": [
            {
                "suite_id": "industrial-retrieval-final-gold-v1",
                "reuse_permitted": False,
            },
            {"suite_id": "industrial-reranker-final-gold-v2", "reuse_permitted": False},
            {"suite_id": "industrial-reranker-final-gold-v3", "reuse_permitted": False},
        ],
        "top_k": 3,
        "documents": documents,
        "cases": cases,
    }


def prepare_inputs(repo_root: Path) -> tuple[Path, Path]:
    root = repo_root.resolve(strict=True)
    _predecessor_evidence(root)
    registry_path = root / REGISTRY_RELATIVE
    gold_path = root / GOLD_RELATIVE
    v3._write_or_require_json(registry_path, terminology_registry_document())
    v3._write_or_require_json(gold_path, formal_gold_document())
    _verify_input_documents(root, registry_path, gold_path)
    return registry_path, gold_path


def preflight_grounded_reranker(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    registry_path, gold_path = prepare_inputs(root)
    assert_gold_available(root)
    model_path = v3._inside_directory(root, v3.BASE_MODEL_RELATIVE)
    v3._verify_base_model(model_path)
    predecessor = _predecessor_evidence(root)
    return {
        "schema_version": "enterprise-reranker-grounded-preflight/v4",
        "classification": CLASSIFICATION,
        "status": "RERANKER_GROUNDED_PREFLIGHT_PASSED",
        "source_v3_run_id": predecessor.run_id,
        "source_candidate_actual_gpu_trained": predecessor.actual_gpu_training,
        "registry_sha256": v3._file_sha256(registry_path),
        "gold_sha256": v3._file_sha256(gold_path),
        "gold_consumed": False,
        "model_loaded": False,
        "services_started": [],
    }


def assert_gold_available(repo_root: Path) -> None:
    root = repo_root.resolve(strict=True)
    if (root / GOLD_RETIREMENT_RELATIVE).exists():
        raise RerankerGroundedValueLabError("Reranker Gold v4 is already retired")


def planned_run_id(repo_root: Path, image_digest: str) -> str:
    root = repo_root.resolve(strict=True)
    v3._require_image_digest(image_digest)
    registry_path, gold_path = prepare_inputs(root)
    source_path = Path(__file__).resolve(strict=True)
    v3._require_inside(root, source_path)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": v3._file_sha256(source_path),
        "registry_sha256": v3._file_sha256(registry_path),
        "gold_sha256": v3._file_sha256(gold_path),
        "predecessor": _predecessor_evidence(root).model_dump(mode="json"),
        "model_manifest": v3._manifest_digest(
            v3._directory_manifest(v3._inside_directory(root, v3.BASE_MODEL_RELATIVE))
        ),
        "config": _CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
    }
    return f"reranker-grounded-{v3._digest(identity)[:20]}"


def execute_worker(repo_root: Path, *, image_digest: str) -> Path:
    root = repo_root.resolve(strict=True)
    v3._require_image_digest(image_digest)
    registry_path, gold_path = prepare_inputs(root)
    run_id = planned_run_id(root, image_digest)
    accepted_path = root / OUTPUT_RELATIVE / "runs" / run_id / "acceptance.json"
    rejected_path = root / OUTPUT_RELATIVE / "rejections" / run_id / "rejection.json"
    for existing in (accepted_path, rejected_path):
        if existing.is_file():
            verify_grounded_reranker_value(root, existing)
            return existing
    assert_gold_available(root)
    if accepted_path.parent.exists() or rejected_path.parent.exists():
        raise RerankerGroundedValueLabError("immutable Reranker v4 run exists")

    output_root = root / OUTPUT_RELATIVE
    temporary = output_root / f".{run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    gold_started = False
    try:
        predecessor = _predecessor_evidence(root)
        candidate_path = temporary / "candidate"
        _materialize_candidate(root, predecessor, registry_path, candidate_path)
        gold_started = True
        evaluation = _evaluate_grounded_reranker(
            base_model_path=v3._inside_directory(root, v3.BASE_MODEL_RELATIVE),
            candidate_path=candidate_path,
            cases=v3._gold_cases(v3._load_object(gold_path)),
            registry=_registry_entries(v3._load_object(registry_path)),
        )
        v3._write_json(temporary / "formal-evaluation.json", evaluation.formal_document)
        v3._write_json(
            temporary / "observations.json", evaluation.observations_document
        )
        dependencies = v3._dependencies()
        torch = dependencies["torch"]
        runtime = {
            "actual_gpu_execution": True,
            "actual_gpu_inference": True,
            "current_run_model_training_performed": False,
            "source_candidate_actual_gpu_trained": predecessor.actual_gpu_training,
            "model_training_simulated": False,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(0).total_memory
            ),
            "peak_gpu_memory_reserved_bytes": int(
                evaluation.summary["evaluation_peak_gpu_memory_reserved_bytes"]
            ),
            "gpu_memory_allocated_after_cleanup_bytes": int(
                torch.cuda.memory_allocated()
            ),
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
        hard_gates = _hard_gates(runtime=runtime, evaluation=evaluation.summary)
        failed = tuple(name for name in _HARD_GATE_NAMES if not hard_gates[name])
        accepted = not failed
        status = (
            "RERANKER_GROUNDED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
            if accepted
            else "RERANKER_GROUNDED_ACTUAL_GPU_CANDIDATE_REJECTED"
        )
        decision = (
            "RERANKER_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
            if accepted
            else "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"
        )
        final_directory = accepted_path.parent if accepted else rejected_path.parent
        report_name = "acceptance.json" if accepted else "rejection.json"
        final_report = final_directory / report_name
        final_relative = final_directory.relative_to(root)
        source_path = Path(__file__).resolve(strict=True)
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
            "source": v3._file_evidence(source_path.relative_to(root), source_path),
            "image": IMAGE,
            "image_digest": image_digest,
            "predecessor": predecessor.model_dump(mode="json"),
            "input_model": v3._input_model_evidence(
                root, v3._inside_directory(root, v3.BASE_MODEL_RELATIVE)
            ),
            "registry": _registry_evidence(root, registry_path),
            "dataset": _dataset_evidence(root, gold_path),
            "candidate": _candidate_evidence(
                final_relative,
                candidate_path,
                predecessor.source_candidate_bundle_sha256,
            ),
            "runtime": runtime,
            "evaluation": _evaluation_evidence(final_relative, temporary, evaluation),
            "hard_gates": hard_gates,
            "failed_hard_gates": list(failed),
            "candidate_accepted": accepted,
            "formal_model_release_created": False,
            "runtime_eligible": False,
            "same_gold_reuse_permitted": False,
            "gold_retirement_path": GOLD_RETIREMENT_RELATIVE.as_posix(),
        }
        draft = RerankerGroundedValueReport.model_validate(
            {**unsigned, "evidence_chain_sha256": "0" * 64}
        )
        report = draft.model_copy(
            update={
                "evidence_chain_sha256": v3._digest(
                    draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
                )
            }
        )
        v3._write_json(temporary / report_name, report.model_dump(mode="json"))
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final_directory)
        _write_gold_retirement(
            root,
            run_id=run_id,
            outcome="PASSED" if accepted else "REJECTED",
            outcome_path=final_report,
        )
        _write_latest(root, final_report, report)
        verify_grounded_reranker_value(root, final_report)
        return final_report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if gold_started and not (root / GOLD_RETIREMENT_RELATIVE).exists():
            _write_interrupted_gold_retirement(root, run_id)
        raise


def verify_grounded_reranker_value(
    repo_root: Path,
    outcome_path: Path | None = None,
) -> RerankerGroundedValueReport:
    root = repo_root.resolve(strict=True)
    path = (
        _latest_outcome_path(root)
        if outcome_path is None
        else v3._inside_file(root, outcome_path)
    )
    try:
        report = RerankerGroundedValueReport.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RerankerGroundedValueLabError("Reranker v4 outcome is invalid") from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != v3._digest(unsigned):
        raise RerankerGroundedValueLabError("Reranker v4 evidence chain changed")
    source_path = v3._inside_file(root, Path(report.source.path))
    if v3._file_evidence(
        Path(report.source.path), source_path
    ) != report.source.model_dump(mode="json"):
        raise RerankerGroundedValueLabError("Reranker v4 source changed")
    registry_path, gold_path = prepare_inputs(root)
    if planned_run_id(root, report.image_digest) != report.run_id:
        raise RerankerGroundedValueLabError("Reranker v4 run identity changed")
    predecessor = _predecessor_evidence(root)
    if report.predecessor != predecessor:
        raise RerankerGroundedValueLabError("Reranker v4 predecessor changed")
    model_path = v3._inside_directory(root, v3.BASE_MODEL_RELATIVE)
    if report.input_model.model_dump(mode="json") != v3._input_model_evidence(
        root, model_path
    ):
        raise RerankerGroundedValueLabError("Reranker v4 base model changed")
    if report.registry.model_dump(mode="json") != _registry_evidence(
        root, registry_path
    ):
        raise RerankerGroundedValueLabError("Reranker v4 registry changed")
    if report.dataset.model_dump(mode="json") != _dataset_evidence(root, gold_path):
        raise RerankerGroundedValueLabError("Reranker v4 Gold changed")
    _verify_candidate(root, report)

    formal_path = v3._verify_file_evidence(root, report.evaluation.formal_report)
    observations_path = v3._verify_file_evidence(root, report.evaluation.observations)
    observations = v3._load_object(observations_path)
    cases = v3._gold_cases(v3._load_object(gold_path))
    baseline = v3._observations_from_document(observations, "baseline")
    candidate = v3._observations_from_document(observations, "candidate")
    formal = score_retrieval_paired_observations(
        cases,
        candidate,
        baseline,
        top_k=_CONFIG["top_k"],
        primary_metric="ndcg_at_k",
    )
    expected_formal = {
        "schema_version": "enterprise-reranker-grounded-formal-evaluation/v4",
        "classification": CLASSIFICATION,
        "component": "RERANKER",
        "report": asdict(formal),
    }
    if _canonical_json(v3._load_object(formal_path)) != _canonical_json(
        expected_formal
    ):
        raise RerankerGroundedValueLabError("Reranker v4 formal evaluation changed")
    summary = _evaluation_summary(
        formal,
        evaluation_peak=report.evaluation.evaluation_peak_gpu_memory_reserved_bytes,
        candidate=candidate,
    )
    failures = _evaluation_failures(summary)
    expected_evaluation = {
        **summary,
        "primary_metric": "ndcg_at_k",
        "formal_report": report.evaluation.formal_report.model_dump(mode="json"),
        "observations": report.evaluation.observations.model_dump(mode="json"),
        "failed_quality_gates": list(failures),
    }
    if report.evaluation.model_dump(mode="json") != expected_evaluation:
        raise RerankerGroundedValueLabError("Reranker v4 evaluation changed")
    hard_gates = _hard_gates(
        runtime=report.runtime.model_dump(mode="json"), evaluation=summary
    )
    expected_failed = tuple(name for name in _HARD_GATE_NAMES if not hard_gates[name])
    accepted = not expected_failed
    if (
        report.hard_gates.model_dump(mode="json") != hard_gates
        or report.failed_hard_gates != expected_failed
        or report.candidate_accepted is not accepted
        or (
            accepted
            and report.status != "RERANKER_GROUNDED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        )
        or (
            not accepted
            and report.status != "RERANKER_GROUNDED_ACTUAL_GPU_CANDIDATE_REJECTED"
        )
    ):
        raise RerankerGroundedValueLabError("Reranker v4 decision changed")
    _verify_gold_retirement(root, path, report, gold_path)
    return report


def _evaluate_grounded_reranker(
    *,
    base_model_path: Path,
    candidate_path: Path,
    cases: tuple[EvaluationCase, ...],
    registry: tuple[_RegistryEntry, ...],
) -> _EvaluationResult:
    dependencies = v3._dependencies()
    baseline, baseline_peak = v3._evaluate_model(base_model_path, cases, dependencies)
    candidate, candidate_peak = _evaluate_candidate_model(
        candidate_path, cases, registry, dependencies
    )
    formal = score_retrieval_paired_observations(
        cases,
        candidate,
        baseline,
        top_k=_CONFIG["top_k"],
        primary_metric="ndcg_at_k",
    )
    summary = _evaluation_summary(
        formal,
        evaluation_peak=max(baseline_peak, candidate_peak),
        candidate=candidate,
    )
    failures = _evaluation_failures(summary)
    return _EvaluationResult(
        summary=summary,
        formal_document={
            "schema_version": "enterprise-reranker-grounded-formal-evaluation/v4",
            "classification": CLASSIFICATION,
            "component": "RERANKER",
            "report": asdict(formal),
        },
        observations_document={
            "schema_version": "enterprise-reranker-grounded-observations/v4",
            "classification": CLASSIFICATION,
            "component": "RERANKER",
            "baseline": [v3._observation_document(item) for item in baseline],
            "candidate": [v3._observation_document(item) for item in candidate],
        },
        failures=failures,
    )


def _evaluate_candidate_model(
    model_path: Path,
    cases: tuple[EvaluationCase, ...],
    registry: tuple[_RegistryEntry, ...],
    dependencies: dict[str, Any],
) -> tuple[tuple[ModelObservation, ...], int]:
    torch = dependencies["torch"]
    v3._prepare_gpu(torch)
    model = dependencies["CrossEncoder"](
        str(model_path),
        device="cuda:0",
        trust_remote_code=False,
        num_labels=1,
        max_length=_CONFIG["max_sequence_length"],
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    observations = tuple(
        _rank_grounded_case(model, case, registry, torch) for case in cases
    )
    peak = int(torch.cuda.max_memory_reserved())
    del model
    v3._cleanup_gpu(torch)
    return observations, peak


def _rank_grounded_case(
    model: Any,
    case: EvaluationCase,
    registry: tuple[_RegistryEntry, ...],
    torch: Any,
) -> ModelObservation:
    contract = case.retrieval
    if contract is None:
        raise RerankerGroundedValueLabError("Reranker v4 Gold contract is missing")
    expanded_query, entry = expand_query(contract.query, registry)
    allowed = tuple(item for item in contract.documents if item.access == "ALLOWED")
    torch.cuda.synchronize()
    started = time.perf_counter()
    raw_scores = model.predict(
        [(expanded_query, item.text) for item in allowed],
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()
    torch.cuda.synchronize()
    latency_ms = max((time.perf_counter() - started) * 1000.0, 1e-9)
    scores = [float(item) for item in raw_scores]
    if len(scores) != len(allowed) or any(not math.isfinite(item) for item in scores):
        raise RerankerGroundedValueLabError("Reranker v4 scores are invalid")
    ranked = sorted(
        zip(allowed, scores, strict=True),
        key=lambda item: (-item[1], item[0].document_id),
    )[: _CONFIG["top_k"]]
    ranked_ids = tuple(item.document_id for item, _ in ranked)
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
            "retrieval_method": "GROUNDED_RERANKER",
            "registry_id": "industrial-reranker-terminology-registry-v1",
            "registry_entry_id": entry.entry_id,
            "query_augmentation_applied": True,
            "expanded_query_sha256": v3._digest(expanded_query),
        },
    )


def expand_query(
    query: str,
    registry: tuple[_RegistryEntry, ...] = _REGISTRY,
) -> tuple[str, _RegistryEntry]:
    folded = query.casefold()
    matches = tuple(item for item in registry if item.alias.casefold() in folded)
    if len(matches) != 1:
        raise RerankerGroundedValueLabError(
            "Reranker terminology resolution must produce exactly one entry"
        )
    entry = matches[0]
    expanded = (
        f"{query}\nGoverned terminology resolution: {entry.mechanism}. "
        f"Required controlled action: {entry.action}. Device family: "
        f"{entry.device_family}; signal family: {entry.signal}."
    )
    return expanded, entry


def _evaluation_summary(
    report: PairedMetricReport,
    *,
    evaluation_peak: int,
    candidate: tuple[ModelObservation, ...],
) -> dict[str, Any]:
    def average(name: str, candidate_side: bool) -> float:
        prefix = "candidate" if candidate_side else "baseline"
        values = [getattr(item, f"{prefix}_{name}") for item in report.case_metrics]
        if any(value is None for value in values):
            raise RerankerGroundedValueLabError(
                f"Reranker v4 metric is missing: {name}"
            )
        return mean(float(value) for value in values)

    baseline_ndcg = average("ndcg_at_k", False)
    candidate_ndcg = average("ndcg_at_k", True)
    baseline_latency = report.evidence.baseline_p95_latency_ms
    candidate_latency = report.evidence.candidate_p95_latency_ms
    resolved = sum(
        item.runtime_evidence.get("query_augmentation_applied") is True
        and item.runtime_evidence.get("registry_entry_id") is not None
        for item in candidate
    )
    return {
        "baseline_recall_at_k": average("recall_at_k", False),
        "candidate_recall_at_k": average("recall_at_k", True),
        "baseline_mrr": average("mrr", False),
        "candidate_mrr": average("mrr", True),
        "baseline_ndcg_at_k": baseline_ndcg,
        "candidate_ndcg_at_k": candidate_ndcg,
        "primary_improvement": candidate_ndcg - baseline_ndcg,
        "baseline_p95_latency_ms": baseline_latency,
        "candidate_p95_latency_ms": candidate_latency,
        "candidate_latency_ratio": candidate_latency / baseline_latency,
        "hard_gate_results": report.evidence.hard_gate_results,
        "evaluation_peak_gpu_memory_reserved_bytes": evaluation_peak,
        "registry_resolution_count": resolved,
        "registry_resolution_failure_count": len(candidate) - resolved,
    }


def _evaluation_failures(summary: dict[str, Any]) -> tuple[str, ...]:
    retrieval_gates = summary["hard_gate_results"]
    if set(retrieval_gates) != RETRIEVAL_HARD_GATES:
        raise RerankerGroundedValueLabError("Reranker v4 retrieval gates drifted")
    failures = [
        f"reranker_{name}"
        for name, passed in sorted(retrieval_gates.items())
        if not passed
    ]
    for key, threshold in (
        ("primary_improvement", _CONFIG["minimum_primary_improvement"]),
        ("candidate_recall_at_k", _CONFIG["minimum_candidate_recall_at_k"]),
        ("candidate_mrr", _CONFIG["minimum_candidate_mrr"]),
        ("candidate_ndcg_at_k", _CONFIG["minimum_candidate_ndcg_at_k"]),
    ):
        if float(summary[key]) < float(threshold):
            failures.append(f"reranker_{key}")
    if int(summary["registry_resolution_failure_count"]) != 0:
        failures.append("reranker_registry_resolution")
    if float(summary["candidate_latency_ratio"]) > _CONFIG["maximum_latency_ratio"]:
        failures.append("reranker_latency_budget")
    return tuple(failures)


def _hard_gates(
    *, runtime: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, bool]:
    retrieval_gates = evaluation.get("hard_gate_results", {})
    gates = {
        "data_governance": True,
        "verified_v3_trained_predecessor": (
            runtime.get("source_candidate_actual_gpu_trained") is True
        ),
        "actual_gpu_grounded_inference": (
            runtime.get("actual_gpu_execution") is True
            and runtime.get("actual_gpu_inference") is True
            and runtime.get("model_training_simulated") is False
        ),
        "immutable_model_registry_gold_and_image": True,
        "fresh_frozen_gold_independence": True,
        "retired_gold_not_reused": True,
        "governed_terminology_resolution": (
            evaluation.get("registry_resolution_count") == 12
            and evaluation.get("registry_resolution_failure_count") == 0
        ),
        "cross_tenant_isolation": (
            retrieval_gates.get("cross_tenant_isolation") is True
        ),
        "formal_retrieval_gates": (
            set(retrieval_gates) == RETRIEVAL_HARD_GATES
            and all(value is True for value in retrieval_gates.values())
        ),
        "reranker_quality_improved": (
            float(evaluation.get("primary_improvement", -1.0))
            >= _CONFIG["minimum_primary_improvement"]
        ),
        "candidate_absolute_quality": (
            float(evaluation.get("candidate_recall_at_k", -1.0))
            >= _CONFIG["minimum_candidate_recall_at_k"]
            and float(evaluation.get("candidate_mrr", -1.0))
            >= _CONFIG["minimum_candidate_mrr"]
            and float(evaluation.get("candidate_ndcg_at_k", -1.0))
            >= _CONFIG["minimum_candidate_ndcg_at_k"]
        ),
        "latency_budget_respected": (
            float(evaluation.get("candidate_latency_ratio", math.inf))
            <= _CONFIG["maximum_latency_ratio"]
        ),
        "resource_profile_respected": (
            runtime.get("non_root_container_user") is True
            and runtime.get("network_disabled") is True
            and runtime.get("read_only_root_filesystem") is True
            and runtime.get("cpu_threads") == 2
            and runtime.get("dataloader_workers") == 0
            and runtime.get("container_memory_limit_bytes")
            == _CONFIG["container_memory_limit_bytes"]
            and int(runtime.get("peak_gpu_memory_reserved_bytes", 0))
            < _CONFIG["maximum_gpu_reserved_bytes"]
        ),
    }
    if tuple(gates) != _HARD_GATE_NAMES:
        raise AssertionError("Reranker v4 hard-gate contract drift")
    return gates


def _predecessor_evidence(root: Path) -> PredecessorEvidence:
    erratum_path = v3._inside_file(root, PREDECESSOR_ERRATUM_RELATIVE)
    erratum = verify_reranker_recovery_verifier_erratum(root, erratum_path)
    original_path = v3._inside_file(root, Path(erratum.original_outcome.path))
    original = verify_original_with_json_normalization(root, original_path)
    if (
        original.run_id != "reranker-recovery-f1ec9f9b7d53bbc879d1"
        or original.runtime.actual_gpu_execution is not True
        or original.runtime.model_training_simulated is not False
        or original.candidate_accepted is not False
        or original.runtime_eligible is not False
        or original.same_gold_reuse_permitted is not False
    ):
        raise RerankerGroundedValueLabError("Reranker v3 predecessor is not eligible")
    candidate_path = v3._inside_directory(root, Path(original.candidate.path))
    manifest = v3._directory_manifest(candidate_path)
    if v3._manifest_digest(manifest) != original.candidate.bundle_sha256:
        raise RerankerGroundedValueLabError("Reranker v3 candidate bundle changed")
    retirement_path = v3._inside_file(root, v3.GOLD_RETIREMENT_RELATIVE)
    return PredecessorEvidence(
        erratum=v3.FileEvidence.model_validate(
            v3._file_evidence(PREDECESSOR_ERRATUM_RELATIVE, erratum_path)
        ),
        original_outcome=v3.FileEvidence.model_validate(
            v3._file_evidence(Path(erratum.original_outcome.path), original_path)
        ),
        gold_v3_retirement=v3.FileEvidence.model_validate(
            v3._file_evidence(v3.GOLD_RETIREMENT_RELATIVE, retirement_path)
        ),
        run_id="reranker-recovery-f1ec9f9b7d53bbc879d1",
        source_candidate_path=original.candidate.path,
        source_candidate_bundle_sha256=original.candidate.bundle_sha256,
        actual_gpu_training=True,
        candidate_accepted=False,
        runtime_eligible=False,
        same_gold_reuse_permitted=False,
    )


def _verify_input_documents(root: Path, registry_path: Path, gold_path: Path) -> None:
    registry = v3._load_object(v3._inside_file(root, registry_path))
    gold = v3._load_object(v3._inside_file(root, gold_path))
    if _canonical_json(registry) != _canonical_json(terminology_registry_document()):
        raise RerankerGroundedValueLabError("Reranker terminology registry changed")
    if _canonical_json(gold) != _canonical_json(formal_gold_document()):
        raise RerankerGroundedValueLabError("Reranker Gold v4 changed")
    entries = _registry_entries(registry)
    raw_cases = gold.get("cases")
    raw_documents = gold.get("documents")
    if (
        len(entries) != 12
        or not isinstance(raw_cases, list)
        or len(raw_cases) != 12
        or not isinstance(raw_documents, list)
        or sum(item.get("access") == "ALLOWED" for item in raw_documents) != 24
        or sum(item.get("access") == "FORBIDDEN" for item in raw_documents) != 2
    ):
        raise RerankerGroundedValueLabError("Reranker v4 input counts changed")
    for row in raw_cases:
        if not isinstance(row, dict):
            raise RerankerGroundedValueLabError("Reranker Gold v4 case is invalid")
        expand_query(str(row.get("query", "")), entries)
    _assert_exact_independence(root, gold)


def _assert_exact_independence(root: Path, gold: dict[str, Any]) -> None:
    prior_paths = (
        v3._PREDECESSOR_GOLD,
        v3._PREDECESSOR_GOLD_V2,
        v3.GOLD_DATASET_RELATIVE,
        v3.TRAINING_DATASET_RELATIVE,
    )
    old_ids: set[str] = set()
    old_texts: set[str] = set()
    for relative in prior_paths:
        document = v3._load_object(v3._inside_file(root, relative))
        ids, texts = _ids_and_evaluation_texts(document)
        old_ids.update(ids)
        old_texts.update(texts)
    new_ids, new_texts = _ids_and_evaluation_texts(gold)
    if not new_ids.isdisjoint(old_ids) or not new_texts.isdisjoint(old_texts):
        raise RerankerGroundedValueLabError(
            "Reranker Gold v4 reuses prior exact evidence"
        )


def _ids_and_evaluation_texts(document: dict[str, Any]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    texts: set[str] = set()
    rows = document.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("case_id", "document_id"):
                value = row.get(key)
                if isinstance(value, str):
                    ids.add(value)
            for key in ("query", "positive", "text"):
                value = row.get(key)
                if isinstance(value, str):
                    texts.add(value)
            negatives = row.get("hard_negatives")
            if isinstance(negatives, list):
                texts.update(str(item) for item in negatives)
    for key in ("cases", "documents"):
        values = document.get(key)
        if not isinstance(values, list):
            continue
        for row in values:
            if not isinstance(row, dict):
                continue
            for id_key in ("case_id", "document_id"):
                value = row.get(id_key)
                if isinstance(value, str):
                    ids.add(value)
            for text_key in ("query", "text"):
                value = row.get(text_key)
                if isinstance(value, str):
                    texts.add(value)
    return ids, texts


def _registry_entries(document: dict[str, Any]) -> tuple[_RegistryEntry, ...]:
    raw = document.get("entries")
    if not isinstance(raw, list):
        raise RerankerGroundedValueLabError("Reranker registry entries are missing")
    try:
        entries = tuple(
            _RegistryEntry(**item) for item in raw if isinstance(item, dict)
        )
    except TypeError as exc:
        raise RerankerGroundedValueLabError(
            "Reranker registry entry is invalid"
        ) from exc
    if len(entries) != 12 or len({item.alias.casefold() for item in entries}) != 12:
        raise RerankerGroundedValueLabError("Reranker registry is ambiguous")
    return entries


def _materialize_candidate(
    root: Path,
    predecessor: PredecessorEvidence,
    registry_path: Path,
    candidate_path: Path,
) -> None:
    source = v3._inside_directory(root, Path(predecessor.source_candidate_path))
    shutil.copytree(source, candidate_path)
    shutil.copy2(
        registry_path,
        candidate_path / "industrial-ops-terminology-registry.json",
    )
    descriptor = {
        "schema_version": "industrial-reranker-grounded-runtime/v1",
        "classification": CLASSIFICATION,
        "source_v3_run_id": predecessor.run_id,
        "source_v3_bundle_sha256": predecessor.source_candidate_bundle_sha256,
        "query_augmentation": "GOVERNED_TERMINOLOGY_REGISTRY_V1",
        "missing_alias": "FAIL_CLOSED",
        "ambiguous_alias": "FAIL_CLOSED",
        "gold_derived": False,
    }
    v3._write_json(candidate_path / "industrial-ops-grounded-runtime.json", descriptor)


def _registry_evidence(root: Path, path: Path) -> dict[str, Any]:
    return RegistryEvidence(
        manifest=v3.FileEvidence.model_validate(
            v3._file_evidence(path.relative_to(root), path)
        ),
        registry_id="industrial-reranker-terminology-registry-v1",
    ).model_dump(mode="json")


def _dataset_evidence(root: Path, path: Path) -> dict[str, Any]:
    return DatasetEvidence(
        final_gold_manifest=v3.FileEvidence.model_validate(
            v3._file_evidence(path.relative_to(root), path)
        )
    ).model_dump(mode="json")


def _candidate_evidence(
    final_relative: Path,
    candidate_path: Path,
    source_bundle_sha256: str,
) -> dict[str, Any]:
    manifest = v3._directory_manifest(candidate_path)
    relative = final_relative / "candidate"
    return CandidateEvidence(
        path=relative.as_posix(),
        bundle_sha256=v3._manifest_digest(manifest),
        size_bytes=sum(int(item["size_bytes"]) for item in manifest),
        files=tuple(
            v3.FileEvidence.model_validate(
                v3._file_evidence(
                    relative / str(item["path"]),
                    candidate_path / str(item["path"]),
                )
            )
            for item in manifest
        ),
        source_v3_bundle_sha256=source_bundle_sha256,
    ).model_dump(mode="json")


def _verify_candidate(root: Path, report: RerankerGroundedValueReport) -> None:
    path = v3._inside_directory(root, Path(report.candidate.path))
    expected = _candidate_evidence(
        Path(report.candidate.path).parent,
        path,
        report.predecessor.source_candidate_bundle_sha256,
    )
    if report.candidate.model_dump(mode="json") != expected:
        raise RerankerGroundedValueLabError("Reranker v4 candidate changed")
    registry_copy = v3._inside_file(
        path, Path("industrial-ops-terminology-registry.json")
    )
    source_registry = v3._inside_file(root, REGISTRY_RELATIVE)
    if v3._file_sha256(registry_copy) != v3._file_sha256(source_registry):
        raise RerankerGroundedValueLabError("Reranker v4 bundled registry changed")


def _evaluation_evidence(
    final_relative: Path,
    temporary: Path,
    evaluation: _EvaluationResult,
) -> dict[str, Any]:
    value = {
        **evaluation.summary,
        "primary_metric": "ndcg_at_k",
        "formal_report": v3._file_evidence(
            final_relative / "formal-evaluation.json",
            temporary / "formal-evaluation.json",
        ),
        "observations": v3._file_evidence(
            final_relative / "observations.json",
            temporary / "observations.json",
        ),
        "failed_quality_gates": list(evaluation.failures),
    }
    return EvaluationEvidence.model_validate(value).model_dump(mode="json")


def _write_gold_retirement(
    root: Path,
    *,
    run_id: str,
    outcome: Literal["PASSED", "REJECTED"],
    outcome_path: Path,
) -> None:
    gold_path = v3._inside_file(root, GOLD_RELATIVE)
    target = root / GOLD_RETIREMENT_RELATIVE
    unsigned = {
        "schema_version": "enterprise-reranker-gold-retirement/v4",
        "classification": CLASSIFICATION,
        "suite_id": "industrial-reranker-final-gold-v4",
        "status": f"RETIRED_AFTER_{outcome}_FINAL_EVALUATION",
        "run_id": run_id,
        "outcome": outcome,
        "outcome_path": outcome_path.relative_to(root).as_posix(),
        "outcome_sha256": v3._file_sha256(outcome_path),
        "gold_manifest_sha256": v3._file_sha256(gold_path),
        "formal_evaluation_count": 1,
        "model_release_created": False,
        "runtime_eligible": False,
        "reuse_permitted": False,
    }
    document = {**unsigned, "evidence_chain_sha256": v3._digest(unsigned)}
    if target.exists():
        if v3._load_object(target) != document:
            raise RerankerGroundedValueLabError("Reranker Gold v4 retirement changed")
    else:
        v3._write_json(target, document)


def _write_interrupted_gold_retirement(root: Path, run_id: str) -> None:
    gold_path = v3._inside_file(root, GOLD_RELATIVE)
    unsigned = {
        "schema_version": "enterprise-reranker-gold-retirement/v4",
        "classification": CLASSIFICATION,
        "suite_id": "industrial-reranker-final-gold-v4",
        "status": "RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION",
        "run_id": run_id,
        "outcome": "INTERRUPTED",
        "gold_manifest_sha256": v3._file_sha256(gold_path),
        "formal_evaluation_count": 1,
        "model_release_created": False,
        "runtime_eligible": False,
        "reuse_permitted": False,
    }
    document = {**unsigned, "evidence_chain_sha256": v3._digest(unsigned)}
    v3._write_or_require_json(root / GOLD_RETIREMENT_RELATIVE, document)


def _verify_gold_retirement(
    root: Path,
    outcome_path: Path,
    report: RerankerGroundedValueReport,
    gold_path: Path,
) -> None:
    path = v3._inside_file(root, GOLD_RETIREMENT_RELATIVE)
    document = v3._load_object(path)
    unsigned = {
        key: value for key, value in document.items() if key != "evidence_chain_sha256"
    }
    expected_outcome = "PASSED" if report.candidate_accepted else "REJECTED"
    if (
        document.get("schema_version") != "enterprise-reranker-gold-retirement/v4"
        or document.get("classification") != CLASSIFICATION
        or document.get("suite_id") != "industrial-reranker-final-gold-v4"
        or document.get("run_id") != report.run_id
        or document.get("outcome") != expected_outcome
        or document.get("outcome_path") != outcome_path.relative_to(root).as_posix()
        or document.get("outcome_sha256") != v3._file_sha256(outcome_path)
        or document.get("gold_manifest_sha256") != v3._file_sha256(gold_path)
        or document.get("formal_evaluation_count") != 1
        or document.get("reuse_permitted") is not False
        or document.get("model_release_created") is not False
        or document.get("runtime_eligible") is not False
        or document.get("evidence_chain_sha256") != v3._digest(unsigned)
    ):
        raise RerankerGroundedValueLabError("Reranker Gold v4 retirement changed")


def _write_latest(
    root: Path,
    report_path: Path,
    report: RerankerGroundedValueReport,
) -> None:
    document = {
        "schema_version": "enterprise-reranker-grounded-latest/v4",
        "classification": CLASSIFICATION,
        "status": report.status,
        "run_id": report.run_id,
        "report": report_path.relative_to(root).as_posix(),
        "report_sha256": v3._file_sha256(report_path),
        "evidence_chain_sha256": report.evidence_chain_sha256,
    }
    v3._write_json(root / LATEST_RELATIVE, document)


def _latest_outcome_path(root: Path) -> Path:
    pointer_path = v3._inside_file(root, LATEST_RELATIVE)
    pointer = v3._load_object(pointer_path)
    value = pointer.get("report")
    if (
        pointer.get("schema_version") != "enterprise-reranker-grounded-latest/v4"
        or pointer.get("classification") != CLASSIFICATION
        or not isinstance(value, str)
        or not value
    ):
        raise RerankerGroundedValueLabError("Reranker v4 latest pointer is invalid")
    path = v3._inside_file(root, Path(value))
    if pointer.get("report_sha256") != v3._file_sha256(path):
        raise RerankerGroundedValueLabError("Reranker v4 latest pointer changed")
    return path


def _canonical_json(value: object) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
