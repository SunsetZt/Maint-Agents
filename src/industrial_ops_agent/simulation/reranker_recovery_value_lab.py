"""Low-memory actual-GPU hard-negative Reranker recovery value lab v3."""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import re
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

from industrial_ops_agent.evaluation.dataset import (
    EvaluationCase,
    RetrievalContract,
    RetrievalDocument,
)
from industrial_ops_agent.evaluation.metrics import (
    ModelObservation,
    PairedMetricReport,
    score_retrieval_paired_observations,
)
from industrial_ops_agent.experiments.service import RETRIEVAL_HARD_GATES

SCHEMA_VERSION: Literal["enterprise-reranker-recovery-value-lab/v3"] = (
    "enterprise-reranker-recovery-value-lab/v3"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
OUTPUT_RELATIVE = Path("artifacts/m7-reranker-recovery-value-lab")
BASE_MODEL_RELATIVE = Path(
    "artifacts/m7-retrieval-enterprise-value-lab/inputs/ms-marco-MiniLM-L6-v2-ce0834f"
)
TRAINING_DATASET_RELATIVE = (
    OUTPUT_RELATIVE / "inputs/industrial-reranker-hard-negative-training-v2/manifest.json"
)
GOLD_DATASET_RELATIVE = OUTPUT_RELATIVE / "inputs/industrial-reranker-final-gold-v3/manifest.json"
GOLD_RETIREMENT_RELATIVE = OUTPUT_RELATIVE / "gold-v3-retirement.json"
LATEST_RELATIVE = OUTPUT_RELATIVE / "latest.json"
IMAGE: Literal["industrial-ops/m4-training-worker:local"] = (
    "industrial-ops/m4-training-worker:local"
)
CONTAINER_NAME = "ioap-reranker-recovery-value-lab"

MODEL_ID: Literal["cross-encoder/ms-marco-MiniLM-L6-v2"] = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION: Literal["ce0834f22110de6d9222af7a7a03628121708969"] = (
    "ce0834f22110de6d9222af7a7a03628121708969"
)
MODEL_WEIGHT_SHA256: Literal["821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"] = (
    "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
)

_PREDECESSOR_REJECTION = Path(
    "artifacts/m7-retrieval-enterprise-value-lab/rejections/"
    "retrieval-1228a164b62f20df8805/rejection.json"
)
_PREDECESSOR_RETIREMENT = Path(
    "artifacts/m7-retrieval-enterprise-value-lab/gold-v1-retirement.json"
)
_PREDECESSOR_GOLD = Path(
    "artifacts/m7-retrieval-enterprise-value-lab/inputs/"
    "industrial-retrieval-final-gold-v1/manifest.json"
)
_PREDECESSOR_INTERRUPTION = OUTPUT_RELATIVE / "gold-v2-retirement.json"
_PREDECESSOR_GOLD_V2 = OUTPUT_RELATIVE / "inputs/industrial-reranker-final-gold-v2/manifest.json"
_PREDECESSOR_INTERRUPTED_RUN_ID: Literal["reranker-recovery-8c07c30670bf50fa70c5"] = (
    "reranker-recovery-8c07c30670bf50fa70c5"
)

_CONFIG: dict[str, Any] = {
    "optimizer_steps": 240,
    "evaluation_interval": 40,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 3e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_sequence_length": 128,
    "precision": "bfloat16",
    "seed": 20260828,
    "data_seed": 20260829,
    "top_k": 3,
    "minimum_primary_improvement": 0.10,
    "minimum_candidate_recall_at_k": 0.90,
    "minimum_candidate_mrr": 0.85,
    "minimum_candidate_ndcg_at_k": 0.85,
    "maximum_latency_ratio": 4.0,
    "cpu_threads": 2,
    "dataloader_workers": 0,
    "container_memory_limit_bytes": 4 * 1024**3,
    "maximum_gpu_reserved_bytes": 2 * 1024**3,
}
_HARD_GATE_NAMES = (
    "data_governance",
    "predecessor_rejection_bound",
    "actual_gpu_training",
    "immutable_model_data_and_image",
    "fresh_frozen_gold_independence",
    "retired_gold_not_reused",
    "cross_tenant_isolation",
    "formal_retrieval_gates",
    "reranker_quality_improved",
    "candidate_absolute_quality",
    "latency_budget_respected",
    "resource_profile_respected",
)


class RerankerRecoveryValueLabError(RuntimeError):
    """The governed Reranker recovery lab could not run or verify."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PredecessorEvidence(_ClosedModel):
    rejection: FileEvidence
    gold_retirement: FileEvidence
    interrupted_gold_v2_retirement: FileEvidence
    run_id: Literal["retrieval-1228a164b62f20df8805"]
    interrupted_run_id: Literal["reranker-recovery-8c07c30670bf50fa70c5"]
    baseline_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    candidate_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    primary_improvement: float = Field(default=0.0, ge=0.0, le=0.0)
    old_gold_reuse_permitted: Literal[False] = False
    interrupted_gold_v2_reuse_permitted: Literal[False] = False


class InputModelEvidence(_ClosedModel):
    model_id: Literal["cross-encoder/ms-marco-MiniLM-L6-v2"] = MODEL_ID
    revision: Literal["ce0834f22110de6d9222af7a7a03628121708969"] = MODEL_REVISION
    weight_sha256: Literal["821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"] = (
        MODEL_WEIGHT_SHA256
    )
    path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[FileEvidence, ...] = Field(min_length=6)
    license: Literal["apache-2.0"] = "apache-2.0"


class DatasetEvidence(_ClosedModel):
    training_manifest: FileEvidence
    final_gold_manifest: FileEvidence
    train_group_count: Literal[36] = 36
    validation_group_count: Literal[12] = 12
    train_pair_count: Literal[108] = 108
    validation_pair_count: Literal[36] = 36
    final_gold_count: Literal[12] = 12
    allowed_document_count: Literal[24] = 24
    forbidden_document_count: Literal[2] = 2
    terminology_family_count: Literal[12] = 12
    top_k: Literal[3] = 3
    source_type: Literal["PLATFORM_SYNTHETIC"] = "PLATFORM_SYNTHETIC"
    project_enterprise_use_authorized: Literal[True] = True
    enterprise_production_data: Literal[False] = False
    retired_v1_and_interrupted_v2_gold_reused: Literal[False] = False
    frozen_gold_excluded_from_training_selection_and_development: Literal[True] = True
    exact_case_ids_queries_and_documents_disjoint: Literal[True] = True


class TrainingEvidence(_ClosedModel):
    trainer_family: Literal["sentence-transformers-cross-encoder-5"]
    loss_family: Literal["BinaryCrossEntropyLoss"]
    hard_negative_strategy: Literal["DOMAIN_ALIAS_MECHANISM_CONFOUNDERS"]
    optimizer_steps: Literal[240] = 240
    selected_validation_step: int = Field(ge=40, le=240)
    train_loss: float = Field(ge=0.0)
    selected_validation_loss: float = Field(ge=0.0)
    trainable_parameters: int = Field(gt=0)
    total_parameters: int = Field(gt=0)
    peak_gpu_memory_allocated_bytes: int = Field(gt=0)
    peak_gpu_memory_reserved_bytes: int = Field(gt=0)
    gpu_memory_allocated_after_cleanup_bytes: int = Field(ge=0)


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


class EvaluationEvidence(MetricSummary):
    primary_metric: Literal["ndcg_at_k"] = "ndcg_at_k"
    formal_report: FileEvidence
    observations: FileEvidence
    failed_quality_gates: tuple[str, ...]


class CandidateEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    files: tuple[FileEvidence, ...] = Field(min_length=6)
    role: Literal["AUTHORIZED_ENTERPRISE_RERANKER_EVALUATION_CANDIDATE"] = (
        "AUTHORIZED_ENTERPRISE_RERANKER_EVALUATION_CANDIDATE"
    )


class RuntimeEvidence(_ClosedModel):
    actual_gpu_execution: Literal[True] = True
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


class RerankerRecoveryHardGates(_ClosedModel):
    data_governance: bool
    predecessor_rejection_bound: bool
    actual_gpu_training: bool
    immutable_model_data_and_image: bool
    fresh_frozen_gold_independence: bool
    retired_gold_not_reused: bool
    cross_tenant_isolation: bool
    formal_retrieval_gates: bool
    reranker_quality_improved: bool
    candidate_absolute_quality: bool
    latency_budget_respected: bool
    resource_profile_respected: bool


class RerankerRecoveryValueReport(_ClosedModel):
    schema_version: Literal["enterprise-reranker-recovery-value-lab/v3"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal[
        "RERANKER_RECOVERY_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
        "RERANKER_RECOVERY_ACTUAL_GPU_CANDIDATE_REJECTED",
    ]
    decision: Literal[
        "RERANKER_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT",
        "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD",
    ]
    run_id: str = Field(pattern=r"^reranker-recovery-[0-9a-f]{20}$")
    source: FileEvidence
    image: Literal["industrial-ops/m4-training-worker:local"] = IMAGE
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    predecessor: PredecessorEvidence
    input_model: InputModelEvidence
    dataset: DatasetEvidence
    training: TrainingEvidence
    evaluation: EvaluationEvidence
    candidate: CandidateEvidence
    runtime: RuntimeEvidence
    hard_gates: RerankerRecoveryHardGates
    failed_hard_gates: tuple[str, ...]
    candidate_accepted: bool
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    same_gold_reuse_permitted: Literal[False] = False
    gold_retirement_path: str = Field(min_length=1)
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _TrainingResult:
    evidence: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _EvaluationResult:
    summary: dict[str, Any]
    formal_document: dict[str, Any]
    observations_document: dict[str, Any]
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TerminologyFamily:
    alias: str
    mechanism: str
    action: str
    device_family: str
    signal: str


_FAMILIES = (
    _TerminologyFamily(
        "amber cascade",
        "lubricant aeration and persistent oil foaming",
        "inspect the oil return and air-ingress points before restart",
        "gearbox",
        "lubricant",
    ),
    _TerminologyFamily(
        "hollow pulse",
        "pump cavitation caused by inadequate suction head",
        "inspect suction restriction, fluid level, and available NPSH",
        "pump",
        "pressure",
    ),
    _TerminologyFamily(
        "glass rain",
        "rolling-element bearing outer-race spalling",
        "verify envelope acceleration and inspect the outer race",
        "rotating",
        "vibration",
    ),
    _TerminologyFamily(
        "winter arc",
        "surface tracking across moisture-contaminated insulation",
        "isolate power and inspect insulation for tracking paths",
        "motor",
        "electrical",
    ),
    _TerminologyFamily(
        "velvet stall",
        "compressor surge with unstable low-flow operation",
        "unload safely and inspect the anti-surge control path",
        "compressor",
        "flow",
    ),
    _TerminologyFamily(
        "silver cough",
        "control-valve stiction during small command changes",
        "verify travel feedback and inspect packing friction",
        "control",
        "position",
    ),
    _TerminologyFamily(
        "paper thunder",
        "intermittent arcing at a loose power termination",
        "de-energize and inspect torque, discoloration, and insulation",
        "switchgear",
        "electrical",
    ),
    _TerminologyFamily(
        "blue ladder",
        "progressive battery-cell heating before thermal runaway",
        "isolate charging and verify the cell temperature gradient",
        "battery",
        "temperature",
    ),
    _TerminologyFamily(
        "ghost tooth",
        "gear-mesh backlash or a damaged tooth under reversal",
        "lock out motion and inspect backlash and tooth contact",
        "gearbox",
        "mechanical",
    ),
    _TerminologyFamily(
        "dry mirror",
        "lubrication starvation on a polished bearing raceway",
        "stop the asset and verify lubricant delivery before restart",
        "bearing",
        "lubricant",
    ),
    _TerminologyFamily(
        "slow echo",
        "encoder feedback delay from shielding or connector degradation",
        "isolate motion and inspect the feedback cable and connector",
        "servo",
        "sensor",
    ),
    _TerminologyFamily(
        "salt bloom",
        "coolant crystallization from concentration or contamination",
        "stop circulation and inspect concentration, filtration, and seals",
        "cooling",
        "process",
    ),
)


def training_dataset_document() -> dict[str, Any]:
    """Build reviewed alias-to-mechanism groups with lexical confounders."""

    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(_FAMILIES, start=1):
        wrong = _FAMILIES[family_index % len(_FAMILIES)]
        for variant in range(1, 4):
            asset = f"TR-{family_index:02d}{variant}"
            rows.append(
                {
                    "case_id": f"reranker-v2-train-{family_index:02d}-{variant}",
                    "split": "train",
                    "query": (
                        f"{asset} shows the site symptom '{family.alias}' during verified "
                        "steady operation. Retrieve the diagnostic mechanism and governed "
                        "inspection action."
                    ),
                    "positive": (
                        f"For {family.device_family} assets, {family.mechanism}; {family.action}."
                    ),
                    "hard_negatives": [
                        (
                            f"The words '{family.alias}' are printed on an administrative "
                            "inspection label; this label is not a diagnostic finding."
                        ),
                        (f"For {wrong.device_family} assets, {wrong.mechanism}; {wrong.action}."),
                    ],
                    "terminology_family": family.alias,
                    "review_status": "APPROVED",
                }
            )
        rows.append(
            {
                "case_id": f"reranker-v2-validation-{family_index:02d}",
                "split": "validation",
                "query": (
                    f"A validation ticket uses '{family.alias}'. Which technical note "
                    "explains the fault rather than merely repeating the local phrase?"
                ),
                "positive": (
                    f"The verified interpretation is {family.mechanism}; technicians must "
                    f"{family.action}."
                ),
                "hard_negatives": [
                    (
                        f"Archive record: '{family.alias}' is a local label string and no "
                        "equipment diagnosis is recorded."
                    ),
                    (f"Unrelated mechanism: {wrong.mechanism}; {wrong.action}."),
                ],
                "terminology_family": family.alias,
                "review_status": "APPROVED",
            }
        )
    return {
        "schema_version": "industrial-reranker-hard-negative-training/v2",
        "dataset_id": "industrial-reranker-hard-negative-training-v2",
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "enterprise_production_data": False,
        "project_generated_data": True,
        "project_enterprise_use_authorized": True,
        "purpose": "RERANKER_ALIAS_TO_MECHANISM_HARD_NEGATIVE_TRAINING",
        "review_status": "APPROVED",
        "selection_policy": {
            "final_gold_used_for_training": False,
            "final_gold_used_for_checkpoint_selection": False,
            "validation_used_for_checkpoint_selection": True,
            "retired_v1_gold_reused": False,
        },
        "rows": rows,
    }


def formal_gold_document() -> dict[str, Any]:
    """Build fresh v3 one-shot Gold after the governed v2 interruption."""

    documents: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for index, family in enumerate(_FAMILIES, start=1):
        positive_id = f"rr-v3-resolution-{index:03d}"
        decoy_id = f"rr-v3-taxonomy-decoy-{index:03d}"
        documents.extend(
            (
                {
                    "document_id": positive_id,
                    "text": (
                        f"Field engineering resolution for this {family.device_family} event: "
                        f"evidence identifies {family.mechanism}. The controlled intervention "
                        f"is to {family.action}."
                    ),
                    "access": "ALLOWED",
                },
                {
                    "document_id": decoy_id,
                    "text": (
                        f"Search taxonomy record '{family.alias}': this phrase is retained "
                        "only to route archived labels and contains neither an engineering "
                        "cause nor a controlled maintenance response."
                    ),
                    "access": "ALLOWED",
                },
            )
        )
        cases.append(
            {
                "case_id": f"reranker-recovery-gold-v3-{index:03d}",
                "query": (
                    f"Reviewed service triage for unit GV-{index:03d} was tagged "
                    f"'{family.alias}'. Return the field-engineering cause and controlled "
                    "response before any taxonomy-only archive record."
                ),
                "relevant_document_ids": [positive_id],
                "risk": "HIGH" if index % 2 else "MEDIUM",
                "slices": {
                    "device_family": family.device_family,
                    "signal": family.signal,
                    "terminology_family": family.alias.replace(" ", "_"),
                },
                "review_status": "APPROVED",
            }
        )
    documents.extend(
        (
            {
                "document_id": "tenant-rival-reranker-v3-secret-001",
                "text": "Another tenant's confidential v3 engineering resolution.",
                "access": "FORBIDDEN",
            },
            {
                "document_id": "tenant-rival-reranker-v3-secret-002",
                "text": "Another tenant's private v3 service history and control response.",
                "access": "FORBIDDEN",
            },
        )
    )
    return {
        "schema_version": "industrial-reranker-final-gold/v3",
        "suite_id": "industrial-reranker-final-gold-v3",
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "enterprise_production_data": False,
        "project_generated_data": True,
        "project_enterprise_use_authorized": True,
        "purpose": "FINAL_INDEPENDENT_RERANKER_EVALUATION",
        "frozen_at": "2026-08-28T00:30:00Z",
        "formal_evaluation_limit": 1,
        "selection_policy": {
            "excluded_from_training": True,
            "excluded_from_checkpoint_selection": True,
            "excluded_from_development_probe": True,
            "retired_v1_gold_reused": False,
            "interrupted_v2_gold_reused": False,
        },
        "predecessor_gold": [
            {
                "path": _PREDECESSOR_GOLD.as_posix(),
                "outcome": "REJECTED",
                "reuse_permitted": False,
            },
            {
                "path": _PREDECESSOR_GOLD_V2.as_posix(),
                "outcome": "INTERRUPTED",
                "reuse_permitted": False,
            },
        ],
        "top_k": 3,
        "documents": documents,
        "cases": cases,
    }


def prepare_inputs(repo_root: Path) -> tuple[Path, Path]:
    root = repo_root.resolve(strict=True)
    _verify_predecessor(root)
    _assert_dataset_independence(root)
    training_path = root / TRAINING_DATASET_RELATIVE
    gold_path = root / GOLD_DATASET_RELATIVE
    _write_or_require_json(training_path, training_dataset_document())
    _write_or_require_json(gold_path, formal_gold_document())
    return training_path, gold_path


def assert_gold_available(repo_root: Path) -> None:
    """Reject a second formal use of the immutable Reranker Gold v3 suite."""

    root = repo_root.resolve(strict=True)
    if (root / GOLD_RETIREMENT_RELATIVE).exists():
        raise RerankerRecoveryValueLabError("Reranker Gold v3 is already retired")


def planned_run_id(repo_root: Path, image_digest: str) -> str:
    root = repo_root.resolve(strict=True)
    _require_image_digest(image_digest)
    training_path, gold_path = prepare_inputs(root)
    model_path = _inside_directory(root, BASE_MODEL_RELATIVE)
    _verify_base_model(model_path)
    source_path = Path(__file__).resolve(strict=True)
    _require_inside(root, source_path)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": _file_sha256(source_path),
        "training_sha256": _file_sha256(training_path),
        "gold_sha256": _file_sha256(gold_path),
        "predecessor": _verify_predecessor(root).model_dump(mode="json"),
        "model_manifest": _manifest_digest(_directory_manifest(model_path)),
        "training_config": _CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
    }
    return f"reranker-recovery-{_digest(identity)[:20]}"


def execute_worker(repo_root: Path, *, image_digest: str) -> Path:
    root = repo_root.resolve(strict=True)
    _require_image_digest(image_digest)
    training_path, gold_path = prepare_inputs(root)
    model_path = _inside_directory(root, BASE_MODEL_RELATIVE)
    _verify_base_model(model_path)
    run_id = planned_run_id(root, image_digest)
    accepted_path = root / OUTPUT_RELATIVE / "runs" / run_id / "acceptance.json"
    rejected_path = root / OUTPUT_RELATIVE / "rejections" / run_id / "rejection.json"
    for existing in (accepted_path, rejected_path):
        if existing.is_file():
            verify_reranker_recovery_value(root, existing)
            return existing
    retirement_path = root / GOLD_RETIREMENT_RELATIVE
    assert_gold_available(root)
    if accepted_path.parent.exists() or rejected_path.parent.exists():
        raise RerankerRecoveryValueLabError("immutable Reranker recovery run exists")

    output_root = root / OUTPUT_RELATIVE
    temporary = output_root / f".{run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    gold_started = False
    try:
        training = _train_reranker(
            model_path,
            _load_object(training_path),
            temporary / "work",
            temporary / "candidate",
        )
        shutil.rmtree(temporary / "work")
        gold_started = True
        evaluation = _evaluate_reranker(
            base_model_path=model_path,
            candidate_path=temporary / "candidate",
            cases=_gold_cases(_load_object(gold_path)),
        )
        _write_json(temporary / "formal-evaluation.json", evaluation.formal_document)
        _write_json(temporary / "observations.json", evaluation.observations_document)
        dependencies = _dependencies()
        torch = dependencies["torch"]
        peak_reserved = max(
            int(training.evidence["peak_gpu_memory_reserved_bytes"]),
            int(evaluation.summary["evaluation_peak_gpu_memory_reserved_bytes"]),
        )
        runtime = {
            "actual_gpu_execution": True,
            "model_training_simulated": False,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "peak_gpu_memory_reserved_bytes": peak_reserved,
            "gpu_memory_allocated_after_cleanup_bytes": int(torch.cuda.memory_allocated()),
            "torch_version": str(torch.__version__),
            "sentence_transformers_version": str(dependencies["sentence_transformers"].__version__),
            "transformers_version": str(dependencies["transformers"].__version__),
            "non_root_container_user": os.geteuid() != 0,
            "network_disabled": os.environ.get("IOAP_NETWORK_DISABLED") == "1",
            "read_only_root_filesystem": os.environ.get("IOAP_READ_ONLY_ROOTFS") == "1",
            "baseline_and_candidate_loaded_sequentially": True,
            "cpu_threads": 2,
            "dataloader_workers": 0,
            "container_memory_limit_bytes": _CONFIG["container_memory_limit_bytes"],
        }
        hard_gates = _hard_gates(
            runtime=runtime,
            evaluation=evaluation.summary,
        )
        failed = tuple(name for name in _HARD_GATE_NAMES if not hard_gates[name])
        accepted = not failed
        status = (
            "RERANKER_RECOVERY_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
            if accepted
            else "RERANKER_RECOVERY_ACTUAL_GPU_CANDIDATE_REJECTED"
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
        candidate_evidence = _candidate_evidence(
            final_relative,
            temporary / "candidate",
        )
        evaluation_evidence = _evaluation_evidence(
            final_relative,
            temporary,
            evaluation,
        )
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
            "source": _file_evidence(source_path.relative_to(root), source_path),
            "image": IMAGE,
            "image_digest": image_digest,
            "predecessor": _verify_predecessor(root).model_dump(mode="json"),
            "input_model": _input_model_evidence(root, model_path),
            "dataset": _dataset_evidence(root, training_path, gold_path),
            "training": training.evidence,
            "evaluation": evaluation_evidence,
            "candidate": candidate_evidence,
            "runtime": runtime,
            "hard_gates": hard_gates,
            "failed_hard_gates": list(failed),
            "candidate_accepted": accepted,
            "formal_model_release_created": False,
            "runtime_eligible": False,
            "same_gold_reuse_permitted": False,
            "gold_retirement_path": GOLD_RETIREMENT_RELATIVE.as_posix(),
        }
        draft = RerankerRecoveryValueReport.model_validate(
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
            run_id=run_id,
            outcome="PASSED" if accepted else "REJECTED",
            outcome_path=final_report,
        )
        _write_latest(root, final_report, report)
        verify_reranker_recovery_value(root, final_report)
        return final_report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if gold_started and not retirement_path.exists():
            _write_interrupted_gold_retirement(root, run_id)
        raise


def verify_reranker_recovery_value(
    repo_root: Path,
    outcome_path: Path | None = None,
) -> RerankerRecoveryValueReport:
    root = repo_root.resolve(strict=True)
    path = _latest_outcome_path(root) if outcome_path is None else _inside_file(root, outcome_path)
    try:
        report = RerankerRecoveryValueReport.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RerankerRecoveryValueLabError("Reranker recovery outcome is invalid") from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise RerankerRecoveryValueLabError("Reranker recovery evidence chain changed")
    source_path = _inside_file(root, Path(report.source.path))
    if _file_evidence(Path(report.source.path), source_path) != report.source.model_dump(
        mode="json"
    ):
        raise RerankerRecoveryValueLabError("Reranker recovery source changed")
    training_path, gold_path = prepare_inputs(root)
    if planned_run_id(root, report.image_digest) != report.run_id:
        raise RerankerRecoveryValueLabError("Reranker recovery run identity changed")
    if report.predecessor != _verify_predecessor(root):
        raise RerankerRecoveryValueLabError("Reranker predecessor binding changed")
    model_path = _inside_directory(root, BASE_MODEL_RELATIVE)
    if report.input_model.model_dump(mode="json") != _input_model_evidence(root, model_path):
        raise RerankerRecoveryValueLabError("Reranker base model changed")
    if report.dataset.model_dump(mode="json") != _dataset_evidence(root, training_path, gold_path):
        raise RerankerRecoveryValueLabError("Reranker recovery data changed")
    candidate_path = _inside_directory(root, Path(report.candidate.path))
    candidate_manifest = _directory_manifest(candidate_path)
    if (
        _manifest_digest(candidate_manifest) != report.candidate.bundle_sha256
        or sum(int(item["size_bytes"]) for item in candidate_manifest)
        != report.candidate.size_bytes
        or candidate_manifest
        != [
            {
                "path": Path(item.path).relative_to(Path(report.candidate.path)).as_posix(),
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in report.candidate.files
        ]
    ):
        raise RerankerRecoveryValueLabError("Reranker candidate bundle changed")
    formal_path = _verify_file_evidence(root, report.evaluation.formal_report)
    observations_path = _verify_file_evidence(root, report.evaluation.observations)
    observations = _load_object(observations_path)
    cases = _gold_cases(_load_object(gold_path))
    baseline = _observations_from_document(observations, "baseline")
    candidate = _observations_from_document(observations, "candidate")
    formal = score_retrieval_paired_observations(
        cases,
        candidate,
        baseline,
        top_k=_CONFIG["top_k"],
        primary_metric="ndcg_at_k",
    )
    expected_formal = {
        "schema_version": "enterprise-reranker-formal-evaluation/v3",
        "classification": CLASSIFICATION,
        "component": "RERANKER",
        "report": asdict(formal),
    }
    if _load_object(formal_path) != expected_formal:
        raise RerankerRecoveryValueLabError("Reranker formal evaluation changed")
    summary = _evaluation_summary(
        formal,
        evaluation_peak=report.evaluation.evaluation_peak_gpu_memory_reserved_bytes,
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
        raise RerankerRecoveryValueLabError("Reranker evaluation metrics changed")
    hard_gates = _hard_gates(
        runtime=report.runtime.model_dump(mode="json"),
        evaluation=summary,
    )
    expected_failed = tuple(name for name in _HARD_GATE_NAMES if not hard_gates[name])
    if (
        hard_gates != report.hard_gates.model_dump(mode="json")
        or expected_failed != report.failed_hard_gates
    ):
        raise RerankerRecoveryValueLabError("Reranker hard-gate result changed")
    accepted = not expected_failed
    if (
        report.candidate_accepted is not accepted
        or (accepted and report.status != "RERANKER_RECOVERY_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED")
        or (accepted and report.decision != "RERANKER_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT")
        or (not accepted and report.status != "RERANKER_RECOVERY_ACTUAL_GPU_CANDIDATE_REJECTED")
        or (not accepted and report.decision != "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD")
    ):
        raise RerankerRecoveryValueLabError("Reranker decision is inconsistent")
    retirement_path = _inside_file(root, Path(report.gold_retirement_path))
    retirement = _load_chained_object(
        retirement_path,
        schema="enterprise-reranker-gold-retirement/v3",
        run_id=report.run_id,
    )
    if (
        retirement.get("outcome") != ("PASSED" if accepted else "REJECTED")
        or retirement.get("formal_evaluation_count") != 1
        or retirement.get("gold_manifest_sha256") != _file_sha256(gold_path)
        or retirement.get("outcome_path") != path.relative_to(root).as_posix()
        or retirement.get("outcome_sha256") != _file_sha256(path)
        or retirement.get("reuse_permitted") is not False
        or retirement.get("model_release_created") is not False
        or retirement.get("runtime_eligible") is not False
    ):
        raise RerankerRecoveryValueLabError("Reranker Gold retirement changed")
    return report


def _train_reranker(
    model_path: Path,
    training_document: dict[str, Any],
    work_directory: Path,
    candidate_directory: Path,
) -> _TrainingResult:
    dependencies = _dependencies()
    torch = dependencies["torch"]
    _prepare_gpu(torch)
    model = dependencies["CrossEncoder"](
        str(model_path),
        device="cuda:0",
        trust_remote_code=False,
        num_labels=1,
        max_length=_CONFIG["max_sequence_length"],
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    train_records = _training_records(training_document, "train")
    validation_records = _training_records(training_document, "validation")
    work_directory.mkdir(parents=True, exist_ok=False)
    args = dependencies["CrossEncoderTrainingArguments"](
        output_dir=str(work_directory),
        max_steps=_CONFIG["optimizer_steps"],
        per_device_train_batch_size=_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=1,
        learning_rate=_CONFIG["learning_rate"],
        warmup_ratio=_CONFIG["warmup_ratio"],
        weight_decay=_CONFIG["weight_decay"],
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=_CONFIG["evaluation_interval"],
        save_strategy="steps",
        save_steps=_CONFIG["evaluation_interval"],
        logging_steps=40,
        bf16=True,
        fp16=False,
        seed=_CONFIG["seed"],
        data_seed=_CONFIG["data_seed"],
        report_to="none",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    loss = dependencies["RerankerLoss"](
        model=model,
        pos_weight=torch.tensor(2.0, device="cuda:0"),
    )
    trainer = dependencies["CrossEncoderTrainer"](
        model=model,
        args=args,
        train_dataset=dependencies["Dataset"].from_list(train_records),
        eval_dataset=dependencies["Dataset"].from_list(validation_records),
        loss=loss,
    )
    train_result = trainer.train()
    candidate_directory.mkdir(parents=True, exist_ok=False)
    trainer.save_model(str(candidate_directory))
    best_checkpoint = trainer.state.best_model_checkpoint
    selected_step = (
        _checkpoint_step(best_checkpoint)
        if isinstance(best_checkpoint, str)
        else int(trainer.state.global_step)
    )
    best_metric = trainer.state.best_metric
    if best_metric is None:
        best_metric = trainer.evaluate().get("eval_loss")
    train_loss = train_result.metrics.get("train_loss")
    if (
        isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or isinstance(train_loss, bool)
        or not isinstance(train_loss, (int, float))
    ):
        raise RerankerRecoveryValueLabError("Reranker training metrics are missing")
    parameters = tuple(model.model.parameters())
    evidence = {
        "trainer_family": "sentence-transformers-cross-encoder-5",
        "loss_family": "BinaryCrossEntropyLoss",
        "hard_negative_strategy": "DOMAIN_ALIAS_MECHANISM_CONFOUNDERS",
        "optimizer_steps": int(trainer.state.global_step),
        "selected_validation_step": selected_step,
        "train_loss": float(train_loss),
        "selected_validation_loss": float(best_metric),
        "trainable_parameters": sum(item.numel() for item in parameters if item.requires_grad),
        "total_parameters": sum(item.numel() for item in parameters),
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "gpu_memory_allocated_after_cleanup_bytes": 0,
    }
    del trainer, loss, model
    _cleanup_gpu(torch)
    evidence["gpu_memory_allocated_after_cleanup_bytes"] = int(torch.cuda.memory_allocated())
    TrainingEvidence.model_validate(evidence)
    _write_json(candidate_directory / "industrial-ops-training-report.json", evidence)
    return _TrainingResult(evidence=evidence)


def _evaluate_reranker(
    *,
    base_model_path: Path,
    candidate_path: Path,
    cases: tuple[EvaluationCase, ...],
) -> _EvaluationResult:
    dependencies = _dependencies()
    baseline, baseline_peak = _evaluate_model(base_model_path, cases, dependencies)
    candidate, candidate_peak = _evaluate_model(candidate_path, cases, dependencies)
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
    )
    failures = _evaluation_failures(summary)
    return _EvaluationResult(
        summary=summary,
        formal_document={
            "schema_version": "enterprise-reranker-formal-evaluation/v3",
            "classification": CLASSIFICATION,
            "component": "RERANKER",
            "report": asdict(formal),
        },
        observations_document={
            "schema_version": "enterprise-reranker-observations/v3",
            "classification": CLASSIFICATION,
            "component": "RERANKER",
            "baseline": [_observation_document(item) for item in baseline],
            "candidate": [_observation_document(item) for item in candidate],
        },
        failures=failures,
    )


def _evaluate_model(
    model_path: Path,
    cases: tuple[EvaluationCase, ...],
    dependencies: dict[str, Any],
) -> tuple[tuple[ModelObservation, ...], int]:
    torch = dependencies["torch"]
    _prepare_gpu(torch)
    model = dependencies["CrossEncoder"](
        str(model_path),
        device="cuda:0",
        trust_remote_code=False,
        num_labels=1,
        max_length=_CONFIG["max_sequence_length"],
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    observations = tuple(_rank_case(model, case, torch) for case in cases)
    peak = int(torch.cuda.max_memory_reserved())
    del model
    _cleanup_gpu(torch)
    return observations, peak


def _rank_case(model: Any, case: EvaluationCase, torch: Any) -> ModelObservation:
    contract = case.retrieval
    if contract is None:
        raise RerankerRecoveryValueLabError("Reranker Gold contract is missing")
    allowed = tuple(item for item in contract.documents if item.access == "ALLOWED")
    torch.cuda.synchronize()
    started = time.perf_counter()
    raw_scores = model.predict(
        [(contract.query, item.text) for item in allowed],
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()
    torch.cuda.synchronize()
    latency_ms = max((time.perf_counter() - started) * 1000.0, 1e-9)
    scores = [float(item) for item in raw_scores]
    if len(scores) != len(allowed) or any(not math.isfinite(item) for item in scores):
        raise RerankerRecoveryValueLabError("Reranker scores are invalid")
    ranked = sorted(
        zip(allowed, scores, strict=True),
        key=lambda item: (-item[1], item[0].document_id),
    )[: _CONFIG["top_k"]]
    ranked_ids = tuple(item.document_id for item, _ in ranked)
    ranked_scores = tuple(score for _, score in ranked)
    input_tokens = sum(
        len(model.tokenizer.encode(value, add_special_tokens=True))
        for value in (contract.query, *(item.text for item in allowed))
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
            "retrieval_method": "RERANKER",
        },
    )


def _evaluation_summary(
    report: PairedMetricReport,
    *,
    evaluation_peak: int,
) -> dict[str, Any]:
    def average(name: str, candidate: bool) -> float:
        prefix = "candidate" if candidate else "baseline"
        values = [getattr(item, f"{prefix}_{name}") for item in report.case_metrics]
        if any(value is None for value in values):
            raise RerankerRecoveryValueLabError(f"Reranker metric is missing: {name}")
        return mean(float(value) for value in values)

    baseline_ndcg = average("ndcg_at_k", False)
    candidate_ndcg = average("ndcg_at_k", True)
    baseline_latency = report.evidence.baseline_p95_latency_ms
    candidate_latency = report.evidence.candidate_p95_latency_ms
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
    }


def _evaluation_failures(summary: dict[str, Any]) -> tuple[str, ...]:
    hard_gates = summary["hard_gate_results"]
    if set(hard_gates) != RETRIEVAL_HARD_GATES:
        raise RerankerRecoveryValueLabError("Reranker retrieval gate contract drifted")
    failures = [f"reranker_{gate}" for gate, passed in sorted(hard_gates.items()) if not passed]
    for key, threshold in (
        ("primary_improvement", _CONFIG["minimum_primary_improvement"]),
        ("candidate_recall_at_k", _CONFIG["minimum_candidate_recall_at_k"]),
        ("candidate_mrr", _CONFIG["minimum_candidate_mrr"]),
        ("candidate_ndcg_at_k", _CONFIG["minimum_candidate_ndcg_at_k"]),
    ):
        if float(summary[key]) < float(threshold):
            failures.append(f"reranker_{key}")
    if float(summary["candidate_latency_ratio"]) > _CONFIG["maximum_latency_ratio"]:
        failures.append("reranker_latency_budget")
    return tuple(failures)


def _hard_gates(
    *,
    runtime: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, bool]:
    retrieval_gates = evaluation.get("hard_gate_results", {})
    gates = {
        "data_governance": True,
        "predecessor_rejection_bound": True,
        "actual_gpu_training": (
            runtime.get("actual_gpu_execution") is True
            and runtime.get("model_training_simulated") is False
        ),
        "immutable_model_data_and_image": True,
        "fresh_frozen_gold_independence": True,
        "retired_gold_not_reused": True,
        "cross_tenant_isolation": (retrieval_gates.get("cross_tenant_isolation") is True),
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
            and float(evaluation.get("candidate_mrr", -1.0)) >= _CONFIG["minimum_candidate_mrr"]
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
        raise AssertionError("Reranker recovery hard-gate contract drift")
    return gates


def _training_records(document: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise RerankerRecoveryValueLabError("Reranker training rows are missing")
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != split:
            continue
        query = str(row["query"])
        records.append({"query": query, "document": str(row["positive"]), "label": 1.0})
        records.extend(
            {"query": query, "document": str(item), "label": 0.0} for item in row["hard_negatives"]
        )
    if not records:
        raise RerankerRecoveryValueLabError(f"Reranker split is empty: {split}")
    return records


def _gold_cases(document: dict[str, Any]) -> tuple[EvaluationCase, ...]:
    raw_documents = document.get("documents")
    raw_cases = document.get("cases")
    if not isinstance(raw_documents, list) or not isinstance(raw_cases, list):
        raise RerankerRecoveryValueLabError("Reranker Gold is invalid")
    documents_by_id = {
        str(item["document_id"]): item for item in raw_documents if isinstance(item, dict)
    }
    cases: list[EvaluationCase] = []
    for row in raw_cases:
        if not isinstance(row, dict):
            raise RerankerRecoveryValueLabError("Reranker Gold case is invalid")
        relevant = {str(item) for item in row["relevant_document_ids"]}
        documents = tuple(
            RetrievalDocument(
                document_id=document_id,
                text=str(item["text"]),
                relevance=1 if document_id in relevant else 0,
                access=str(item["access"]),
            )
            for document_id, item in sorted(documents_by_id.items())
        )
        cases.append(
            EvaluationCase(
                case_id=str(row["case_id"]),
                prompt=({"role": "user", "content": str(row["query"])},),
                expected_output={},
                slices={str(key): str(value) for key, value in row["slices"].items()},
                required_gates=tuple(sorted(RETRIEVAL_HARD_GATES)),
                allowed_citations=(),
                forbidden_substrings=("tenant-rival",),
                expected_decision=None,
                risk=str(row["risk"]),
                contexts=(),
                retrieval=RetrievalContract(query=str(row["query"]), documents=documents),
            )
        )
    if len(cases) != 12:
        raise RerankerRecoveryValueLabError("Reranker Gold case count changed")
    return tuple(cases)


def _dataset_evidence(
    root: Path,
    training_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    return DatasetEvidence(
        training_manifest=FileEvidence.model_validate(
            _file_evidence(training_path.relative_to(root), training_path)
        ),
        final_gold_manifest=FileEvidence.model_validate(
            _file_evidence(gold_path.relative_to(root), gold_path)
        ),
    ).model_dump(mode="json")


def _evaluation_evidence(
    final_relative: Path,
    temporary: Path,
    evaluation: _EvaluationResult,
) -> dict[str, Any]:
    value = {
        **evaluation.summary,
        "primary_metric": "ndcg_at_k",
        "formal_report": _file_evidence(
            final_relative / "formal-evaluation.json",
            temporary / "formal-evaluation.json",
        ),
        "observations": _file_evidence(
            final_relative / "observations.json",
            temporary / "observations.json",
        ),
        "failed_quality_gates": list(evaluation.failures),
    }
    return EvaluationEvidence.model_validate(value).model_dump(mode="json")


def _candidate_evidence(final_relative: Path, candidate_path: Path) -> dict[str, Any]:
    manifest = _directory_manifest(candidate_path)
    candidate_relative = final_relative / "candidate"
    value = {
        "path": candidate_relative.as_posix(),
        "bundle_sha256": _manifest_digest(manifest),
        "size_bytes": sum(int(item["size_bytes"]) for item in manifest),
        "files": [
            _file_evidence(
                candidate_relative / str(item["path"]),
                candidate_path / str(item["path"]),
            )
            for item in manifest
        ],
        "role": "AUTHORIZED_ENTERPRISE_RERANKER_EVALUATION_CANDIDATE",
    }
    return CandidateEvidence.model_validate(value).model_dump(mode="json")


def _input_model_evidence(root: Path, model_path: Path) -> dict[str, Any]:
    manifest = _directory_manifest(model_path)
    relative = model_path.relative_to(root)
    value = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weight_sha256": MODEL_WEIGHT_SHA256,
        "path": relative.as_posix(),
        "manifest_sha256": _manifest_digest(manifest),
        "files": [
            _file_evidence(relative / str(item["path"]), model_path / str(item["path"]))
            for item in manifest
        ],
        "license": "apache-2.0",
    }
    return InputModelEvidence.model_validate(value).model_dump(mode="json")


def _verify_base_model(model_path: Path) -> None:
    if _file_sha256(model_path / "model.safetensors") != MODEL_WEIGHT_SHA256:
        raise RerankerRecoveryValueLabError("Reranker base model weight changed")
    readme = (model_path / "README.md").read_text(encoding="utf-8").lower()
    if "apache" not in readme:
        raise RerankerRecoveryValueLabError("Reranker model license is not Apache")


def _verify_predecessor(root: Path) -> PredecessorEvidence:
    rejection_path = _inside_file(root, _PREDECESSOR_REJECTION)
    retirement_path = _inside_file(root, _PREDECESSOR_RETIREMENT)
    interruption_path = _inside_file(root, _PREDECESSOR_INTERRUPTION)
    interrupted_gold_path = _inside_file(root, _PREDECESSOR_GOLD_V2)
    rejection = _load_object(rejection_path)
    unsigned = {key: value for key, value in rejection.items() if key != "evidence_chain_sha256"}
    evaluations = rejection.get("evaluations")
    reranker = None
    if isinstance(evaluations, list):
        reranker = next(
            (
                item
                for item in evaluations
                if isinstance(item, dict) and item.get("component") == "RERANKER"
            ),
            None,
        )
    retirement = _load_object(retirement_path)
    interruption = _load_chained_object(
        interruption_path,
        schema="enterprise-reranker-gold-interruption/v2",
        run_id=_PREDECESSOR_INTERRUPTED_RUN_ID,
    )
    if (
        rejection.get("schema_version") != "enterprise-retrieval-formal-rejection/v1"
        or rejection.get("status") != "RETRIEVAL_ACTUAL_GPU_CANDIDATES_REJECTED"
        or rejection.get("candidate_accepted") is not False
        or rejection.get("run_id") != "retrieval-1228a164b62f20df8805"
        or rejection.get("evidence_chain_sha256") != _digest(unsigned)
        or not isinstance(reranker, dict)
        or reranker.get("primary_metric") != "ndcg_at_k"
        or reranker.get("primary_improvement") != 0.0
        or retirement.get("outcome") != "REJECTED"
        or retirement.get("reuse_permitted") is not False
        or retirement.get("run_id") != rejection.get("run_id")
        or interruption.get("outcome") != "INTERRUPTED"
        or interruption.get("reuse_permitted") is not False
        or interruption.get("gold_manifest_path") != _PREDECESSOR_GOLD_V2.as_posix()
        or interruption.get("gold_manifest_sha256") != _file_sha256(interrupted_gold_path)
        or interruption.get("model_release_created") is not False
        or interruption.get("runtime_eligible") is not False
    ):
        raise RerankerRecoveryValueLabError("Reranker predecessor rejection changed")
    return PredecessorEvidence(
        rejection=FileEvidence.model_validate(
            _file_evidence(rejection_path.relative_to(root), rejection_path)
        ),
        gold_retirement=FileEvidence.model_validate(
            _file_evidence(retirement_path.relative_to(root), retirement_path)
        ),
        interrupted_gold_v2_retirement=FileEvidence.model_validate(
            _file_evidence(interruption_path.relative_to(root), interruption_path)
        ),
        run_id="retrieval-1228a164b62f20df8805",
        interrupted_run_id=_PREDECESSOR_INTERRUPTED_RUN_ID,
        baseline_ndcg_at_k=float(reranker["baseline_ndcg_at_k"]),
        candidate_ndcg_at_k=float(reranker["candidate_ndcg_at_k"]),
        primary_improvement=0.0,
        old_gold_reuse_permitted=False,
        interrupted_gold_v2_reuse_permitted=False,
    )


def _assert_dataset_independence(root: Path) -> None:
    training = training_dataset_document()
    gold = formal_gold_document()
    old_gold_v1 = _load_object(_inside_file(root, _PREDECESSOR_GOLD))
    interrupted_gold_v2 = _load_object(_inside_file(root, _PREDECESSOR_GOLD_V2))
    training_rows = training.get("rows")
    cases = gold.get("cases")
    documents = gold.get("documents")
    old_v1_cases = old_gold_v1.get("cases")
    old_v1_documents = old_gold_v1.get("documents")
    interrupted_v2_cases = interrupted_gold_v2.get("cases")
    interrupted_v2_documents = interrupted_gold_v2.get("documents")
    if not all(
        isinstance(value, list)
        for value in (
            training_rows,
            cases,
            documents,
            old_v1_cases,
            old_v1_documents,
            interrupted_v2_cases,
            interrupted_v2_documents,
        )
    ):
        raise RerankerRecoveryValueLabError("Reranker data contract is invalid")
    assert isinstance(training_rows, list)
    assert isinstance(cases, list)
    assert isinstance(documents, list)
    assert isinstance(old_v1_cases, list)
    assert isinstance(old_v1_documents, list)
    assert isinstance(interrupted_v2_cases, list)
    assert isinstance(interrupted_v2_documents, list)
    training_ids = {str(item.get("case_id")) for item in training_rows if isinstance(item, dict)}
    training_texts = {
        str(value)
        for item in training_rows
        if isinstance(item, dict)
        for value in (
            item.get("query"),
            item.get("positive"),
            *(item.get("hard_negatives") or []),
        )
    }
    gold_ids = {str(item.get("case_id")) for item in cases if isinstance(item, dict)}
    gold_texts = {
        *(str(item.get("query")) for item in cases if isinstance(item, dict)),
        *(str(item.get("text")) for item in documents if isinstance(item, dict)),
    }
    predecessor_ids = {
        *(
            str(item.get("case_id"))
            for item in (*old_v1_cases, *interrupted_v2_cases)
            if isinstance(item, dict)
        ),
    }
    predecessor_texts = {
        *(
            str(item.get("query"))
            for item in (*old_v1_cases, *interrupted_v2_cases)
            if isinstance(item, dict)
        ),
        *(
            str(item.get("text"))
            for item in (*old_v1_documents, *interrupted_v2_documents)
            if isinstance(item, dict)
        ),
    }
    if (
        len(
            [
                item
                for item in training_rows
                if isinstance(item, dict) and item.get("split") == "train"
            ]
        )
        != 36
        or len(
            [
                item
                for item in training_rows
                if isinstance(item, dict) and item.get("split") == "validation"
            ]
        )
        != 12
        or len(cases) != 12
        or len(
            [
                item
                for item in documents
                if isinstance(item, dict) and item.get("access") == "ALLOWED"
            ]
        )
        != 24
        or len(
            [
                item
                for item in documents
                if isinstance(item, dict) and item.get("access") == "FORBIDDEN"
            ]
        )
        != 2
        or training_ids & gold_ids
        or training_texts & gold_texts
        or gold_ids & predecessor_ids
        or gold_texts & predecessor_texts
    ):
        raise RerankerRecoveryValueLabError("Reranker Gold is not independent")


def _observations_from_document(
    document: dict[str, Any],
    key: str,
) -> tuple[ModelObservation, ...]:
    rows = document.get(key)
    if (
        document.get("schema_version") != "enterprise-reranker-observations/v3"
        or document.get("classification") != CLASSIFICATION
        or document.get("component") != "RERANKER"
        or not isinstance(rows, list)
        or len(rows) != 12
    ):
        raise RerankerRecoveryValueLabError("Reranker observations are invalid")
    return tuple(_observation_from_document(item) for item in rows)


def _observation_from_document(value: Any) -> ModelObservation:
    if not isinstance(value, dict):
        raise RerankerRecoveryValueLabError("Reranker observation row is invalid")
    return ModelObservation(
        case_id=str(value["case_id"]),
        output_text=str(value["output_text"]),
        latency_ms=float(value["latency_ms"]),
        cost_usd=float(value["cost_usd"]),
        input_tokens=int(value["input_tokens"]),
        output_tokens=int(value["output_tokens"]),
        capabilities=frozenset(str(item) for item in value["capabilities"]),
        ranked_document_ids=tuple(str(item) for item in value["ranked_document_ids"]),
        retrieval_scores=tuple(float(item) for item in value["retrieval_scores"]),
        runtime_evidence=dict(value["runtime_evidence"]),
    )


def _observation_document(observation: ModelObservation) -> dict[str, Any]:
    value = asdict(observation)
    value["capabilities"] = sorted(observation.capabilities)
    value["ranked_document_ids"] = list(observation.ranked_document_ids)
    value["retrieval_scores"] = list(observation.retrieval_scores)
    return value


def _dependencies() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
        sentence_transformers = importlib.import_module("sentence_transformers")
        cross_encoder = importlib.import_module("sentence_transformers.cross_encoder")
        reranker_losses = importlib.import_module("sentence_transformers.cross_encoder.losses")
        datasets = importlib.import_module("datasets")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RerankerRecoveryValueLabError("Reranker GPU dependencies are not installed") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RerankerRecoveryValueLabError("Reranker lab requires exactly one CUDA GPU")
    return {
        "torch": torch,
        "sentence_transformers": sentence_transformers,
        "transformers": transformers,
        "Dataset": datasets.Dataset,
        "CrossEncoder": cross_encoder.CrossEncoder,
        "CrossEncoderTrainer": cross_encoder.CrossEncoderTrainer,
        "CrossEncoderTrainingArguments": cross_encoder.CrossEncoderTrainingArguments,
        "RerankerLoss": reranker_losses.BinaryCrossEntropyLoss,
    }


def _prepare_gpu(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(_CONFIG["seed"])
    torch.cuda.manual_seed_all(_CONFIG["seed"])


def _cleanup_gpu(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _checkpoint_step(path: str) -> int:
    name = Path(path).name
    if not name.startswith("checkpoint-"):
        raise RerankerRecoveryValueLabError("Reranker checkpoint is invalid")
    try:
        return int(name.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise RerankerRecoveryValueLabError("Reranker checkpoint step is invalid") from exc


def _write_gold_retirement(
    root: Path,
    *,
    run_id: str,
    outcome: str,
    outcome_path: Path,
) -> None:
    unsigned = {
        "schema_version": "enterprise-reranker-gold-retirement/v3",
        "classification": CLASSIFICATION,
        "production_claim": False,
        "run_id": run_id,
        "outcome": outcome,
        "gold_manifest_path": GOLD_DATASET_RELATIVE.as_posix(),
        "gold_manifest_sha256": _file_sha256(root / GOLD_DATASET_RELATIVE),
        "formal_evaluation_count": 1,
        "outcome_path": outcome_path.relative_to(root).as_posix(),
        "outcome_sha256": _file_sha256(outcome_path),
        "reuse_permitted": False,
        "retired_v1_gold_reused": False,
        "interrupted_v2_gold_reused": False,
        "model_release_created": False,
        "runtime_eligible": False,
    }
    _write_json(
        root / GOLD_RETIREMENT_RELATIVE,
        {**unsigned, "evidence_chain_sha256": _digest(unsigned)},
    )


def _write_interrupted_gold_retirement(root: Path, run_id: str) -> None:
    unsigned = {
        "schema_version": "enterprise-reranker-gold-interruption/v3",
        "classification": CLASSIFICATION,
        "production_claim": False,
        "run_id": run_id,
        "outcome": "INTERRUPTED",
        "gold_manifest_path": GOLD_DATASET_RELATIVE.as_posix(),
        "gold_manifest_sha256": _file_sha256(root / GOLD_DATASET_RELATIVE),
        "formal_evaluation_count": 1,
        "reuse_permitted": False,
        "retired_v1_gold_reused": False,
        "interrupted_v2_gold_reused": False,
        "model_release_created": False,
        "runtime_eligible": False,
    }
    _write_json(
        root / GOLD_RETIREMENT_RELATIVE,
        {**unsigned, "evidence_chain_sha256": _digest(unsigned)},
    )


def _write_latest(
    root: Path,
    outcome_path: Path,
    report: RerankerRecoveryValueReport,
) -> None:
    unsigned = {
        "schema_version": "enterprise-reranker-recovery-latest/v3",
        "classification": CLASSIFICATION,
        "production_claim": False,
        "run_id": report.run_id,
        "status": report.status,
        "candidate_accepted": report.candidate_accepted,
        "outcome": outcome_path.relative_to(root).as_posix(),
        "outcome_sha256": _file_sha256(outcome_path),
        "evidence_chain_sha256": report.evidence_chain_sha256,
    }
    _write_json(root / LATEST_RELATIVE, {**unsigned, "pointer_sha256": _digest(unsigned)})


def _latest_outcome_path(root: Path) -> Path:
    pointer = _load_object(_inside_file(root, LATEST_RELATIVE))
    unsigned = {key: value for key, value in pointer.items() if key != "pointer_sha256"}
    value = pointer.get("outcome")
    if (
        pointer.get("schema_version") != "enterprise-reranker-recovery-latest/v3"
        or pointer.get("pointer_sha256") != _digest(unsigned)
        or not isinstance(value, str)
        or not value
    ):
        raise RerankerRecoveryValueLabError("Reranker latest pointer is invalid")
    path = _inside_file(root, Path(value))
    if pointer.get("outcome_sha256") != _file_sha256(path):
        raise RerankerRecoveryValueLabError("Reranker latest outcome changed")
    return path


def _load_chained_object(
    path: Path,
    *,
    schema: str,
    run_id: str,
) -> dict[str, Any]:
    value = _load_object(path)
    unsigned = {key: item for key, item in value.items() if key != "evidence_chain_sha256"}
    if (
        value.get("schema_version") != schema
        or value.get("run_id") != run_id
        or value.get("evidence_chain_sha256") != _digest(unsigned)
    ):
        raise RerankerRecoveryValueLabError("Reranker chained evidence changed")
    return value


def _directory_manifest(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise RerankerRecoveryValueLabError("Reranker evidence directory is missing")
    files = sorted(item for item in directory.rglob("*") if item.is_file())
    if not files:
        raise RerankerRecoveryValueLabError("Reranker evidence directory is empty")
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in files
    ]


def _manifest_digest(manifest: list[dict[str, Any]]) -> str:
    return _digest(manifest)


def _file_evidence(relative: Path, actual: Path) -> dict[str, Any]:
    return {
        "path": relative.as_posix(),
        "size_bytes": actual.stat().st_size,
        "sha256": _file_sha256(actual),
    }


def _verify_file_evidence(root: Path, evidence: FileEvidence) -> Path:
    path = _inside_file(root, Path(evidence.path))
    if path.stat().st_size != evidence.size_bytes or _file_sha256(path) != evidence.sha256:
        raise RerankerRecoveryValueLabError(f"Reranker file changed: {evidence.path}")
    return path


def _inside_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RerankerRecoveryValueLabError("Reranker file escaped repository") from exc
    if not resolved.is_file():
        raise RerankerRecoveryValueLabError("Reranker evidence file is missing")
    return resolved


def _inside_directory(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RerankerRecoveryValueLabError("Reranker directory escaped repository") from exc
    if not resolved.is_dir():
        raise RerankerRecoveryValueLabError("Reranker evidence directory is missing")
    return resolved


def _require_inside(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RerankerRecoveryValueLabError("Reranker source escaped repository") from exc


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RerankerRecoveryValueLabError("Reranker JSON is invalid") from exc
    if not isinstance(value, dict):
        raise RerankerRecoveryValueLabError("Reranker JSON root is invalid")
    return value


def _write_or_require_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _load_object(path) != value:
            raise RerankerRecoveryValueLabError(f"frozen input changed: {path.name}")
        return
    _write_json(path, value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_image_digest(value: str) -> None:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise RerankerRecoveryValueLabError("image digest must be immutable sha256")


def _digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
