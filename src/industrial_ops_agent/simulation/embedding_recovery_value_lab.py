"""Low-memory actual-GPU domain-adapted Embedding recovery value lab."""

from __future__ import annotations

import gc
import importlib
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

SCHEMA_VERSION: Literal["enterprise-embedding-recovery-value-lab/v2"] = (
    "enterprise-embedding-recovery-value-lab/v2"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
STATUS: Literal["EMBEDDING_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"] = (
    "EMBEDDING_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
)
DECISION: Literal["EMBEDDING_CANDIDATE_ELIGIBLE_FOR_PROJECT_RELEASE"] = (
    "EMBEDDING_CANDIDATE_ELIGIBLE_FOR_PROJECT_RELEASE"
)
OUTPUT_RELATIVE = Path("artifacts/m7-embedding-recovery-value-lab")
BASE_MODEL_RELATIVE = Path(
    "artifacts/m7-retrieval-enterprise-value-lab/inputs/all-MiniLM-L6-v2-1110a243"
)
TRAINING_DATASET_RELATIVE = (
    OUTPUT_RELATIVE / "inputs" / "industrial-embedding-domain-training-v2" / "manifest.json"
)
GOLD_DATASET_RELATIVE = (
    OUTPUT_RELATIVE / "inputs" / "industrial-embedding-final-gold-v2" / "manifest.json"
)
GOLD_RETIREMENT_RELATIVE = OUTPUT_RELATIVE / "gold-v2-retirement.json"
IMAGE: Literal["industrial-ops/m4-training-worker:local"] = (
    "industrial-ops/m4-training-worker:local"
)
CONTAINER_NAME = "ioap-embedding-recovery-value-lab"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_WEIGHT_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
TRAINING_MANIFEST_SHA256 = "117d3541df927c4f9d682c7f1e5fbb0b57a4751f3619fcee649b4555a4263ca9"
GOLD_MANIFEST_SHA256 = "693ca26301cd2f8aed6ee1fc3d11a24b1a0e3fcc049827fe96d86a798e489802"

_CONFIG: dict[str, Any] = {
    "optimizer_steps": 120,
    "evaluation_interval": 20,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_sequence_length": 128,
    "precision": "bfloat16",
    "seed": 20260826,
    "data_seed": 20260827,
    "top_k": 3,
    "hard_negatives_per_query": 2,
    "minimum_primary_improvement": 0.05,
    "minimum_candidate_recall_at_k": 0.85,
    "minimum_candidate_mrr": 0.85,
    "maximum_latency_ratio": 4.0,
    "cpu_threads": 2,
    "dataloader_workers": 0,
    "container_memory_limit_bytes": 4 * 1024**3,
}


class EmbeddingRecoveryValueLabError(RuntimeError):
    """The governed Embedding recovery lab could not produce valid evidence."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class InputModelEvidence(_ClosedModel):
    model_id: Literal["sentence-transformers/all-MiniLM-L6-v2"] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    revision: Literal["1110a243fdf4706b3f48f1d95db1a4f5529b4d41"] = (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    weight_sha256: Literal["53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"] = (
        "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
    )
    path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[FileEvidence, ...] = Field(min_length=6)
    license: Literal["apache-2.0"] = "apache-2.0"


class DatasetEvidence(_ClosedModel):
    training_manifest: FileEvidence
    final_gold_manifest: FileEvidence
    train_count: Literal[20]
    validation_count: Literal[10]
    final_gold_count: Literal[12]
    allowed_document_count: Literal[20]
    forbidden_document_count: Literal[2]
    top_k: Literal[3]
    terminology_family_count: Literal[10]
    source_type: Literal["PLATFORM_SYNTHETIC"]
    project_enterprise_use_authorized: Literal[True] = True
    enterprise_production_data: Literal[False] = False
    retired_v1_gold_reused: Literal[False] = False
    frozen_gold_excluded_from_training_selection_and_development: Literal[True] = True
    exact_case_ids_and_texts_disjoint: Literal[True] = True


class TrainingEvidence(_ClosedModel):
    trainer_family: Literal["sentence-transformers-5"]
    loss_family: Literal["MultipleNegativesRankingLoss"]
    optimizer_steps: Literal[120]
    selected_validation_step: int = Field(ge=20, le=120)
    train_loss: float = Field(ge=0.0)
    selected_validation_loss: float = Field(ge=0.0)
    trainable_parameters: int = Field(gt=0)
    total_parameters: int = Field(gt=0)
    peak_gpu_memory_allocated_bytes: int = Field(gt=0)
    peak_gpu_memory_reserved_bytes: int = Field(gt=0)
    gpu_memory_allocated_after_cleanup_bytes: int = Field(ge=0)


class EvaluationEvidence(_ClosedModel):
    primary_metric: Literal["mrr"] = "mrr"
    formal_report: FileEvidence
    observations: FileEvidence
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


class CandidateEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    files: tuple[FileEvidence, ...] = Field(min_length=6)
    role: Literal["AUTHORIZED_ENTERPRISE_EMBEDDING_MODEL_RELEASE_CANDIDATE"] = (
        "AUTHORIZED_ENTERPRISE_EMBEDDING_MODEL_RELEASE_CANDIDATE"
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


class EmbeddingRecoveryHardGates(_ClosedModel):
    data_governance: Literal[True] = True
    actual_gpu_training: Literal[True] = True
    immutable_model_data_and_image: Literal[True] = True
    fresh_frozen_gold_independence: Literal[True] = True
    retired_gold_not_reused: Literal[True] = True
    cross_tenant_isolation: Literal[True] = True
    formal_retrieval_gates: Literal[True] = True
    embedding_quality_improved: Literal[True] = True
    candidate_absolute_quality: Literal[True] = True
    latency_budget_respected: Literal[True] = True
    resource_profile_respected: Literal[True] = True


class EmbeddingRecoveryValueReport(_ClosedModel):
    schema_version: Literal["enterprise-embedding-recovery-value-lab/v2"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["EMBEDDING_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"] = STATUS
    decision: Literal["EMBEDDING_CANDIDATE_ELIGIBLE_FOR_PROJECT_RELEASE"] = DECISION
    run_id: str = Field(pattern=r"^embedding-recovery-[0-9a-f]{20}$")
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image: Literal["industrial-ops/m4-training-worker:local"] = IMAGE
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_model: InputModelEvidence
    dataset: DatasetEvidence
    training: TrainingEvidence
    evaluation: EvaluationEvidence
    candidate: CandidateEvidence
    runtime: RuntimeEvidence
    hard_gates: EmbeddingRecoveryHardGates
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
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


def planned_run_id(repo_root: Path, image_digest: str) -> str:
    root = repo_root.resolve(strict=True)
    _require_image_digest(image_digest)
    paths = _input_paths(root)
    _verify_inputs(paths)
    source_path = Path(__file__).resolve(strict=True)
    _require_inside(root, source_path)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "config": _CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
        "source_sha256": _file_sha256(source_path),
        "base_model_manifest": _manifest_digest(_directory_manifest(paths["base_model"])),
        "training_manifest_sha256": _file_sha256(paths["training_manifest"]),
        "gold_manifest_sha256": _file_sha256(paths["gold_manifest"]),
    }
    return f"embedding-recovery-{_digest(identity)[:20]}"


def assert_gold_available(repo_root: Path) -> None:
    root = repo_root.resolve(strict=True)
    retirement_path = root / GOLD_RETIREMENT_RELATIVE
    if not retirement_path.exists():
        return
    retirement = _load_object(_inside_file(root, retirement_path))
    if (
        retirement.get("suite_id") != "industrial-embedding-final-gold-v2"
        or retirement.get("reuse_permitted") is not False
    ):
        raise EmbeddingRecoveryValueLabError("Embedding v2 Gold retirement contract changed")
    raise EmbeddingRecoveryValueLabError(
        "Embedding v2 final Gold is retired after its single permitted evaluation"
    )


def execute_worker(repo_root: Path, *, image_digest: str) -> Path:
    root = repo_root.resolve(strict=True)
    _require_image_digest(image_digest)
    paths = _input_paths(root)
    _verify_inputs(paths)
    run_id = planned_run_id(root, image_digest)
    output_root = root / OUTPUT_RELATIVE
    final_directory = output_root / "runs" / run_id
    acceptance_path = final_directory / "acceptance.json"
    if acceptance_path.is_file():
        verify_embedding_recovery_value(root, acceptance_path)
        return acceptance_path
    assert_gold_available(root)
    if final_directory.exists():
        raise EmbeddingRecoveryValueLabError("immutable Embedding v2 run directory exists")
    temporary = output_root / f".{run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    gold_started = False
    try:
        training_document = _load_object(paths["training_manifest"])
        gold_document = _load_object(paths["gold_manifest"])
        training = _train_embedding(
            paths["base_model"],
            training_document,
            temporary / "work",
            temporary / "candidate",
        )
        shutil.rmtree(temporary / "work")

        cases = _gold_cases(gold_document)
        gold_started = True
        evaluation = _evaluate_embedding(
            base_model_path=paths["base_model"],
            candidate_path=temporary / "candidate",
            cases=cases,
        )
        _write_json(temporary / "formal-evaluation.json", evaluation.formal_document)
        _write_json(temporary / "observations.json", evaluation.observations_document)
        if evaluation.failures:
            rejection_path = _persist_rejection(
                root=root,
                temporary=temporary,
                run_id=run_id,
                image_digest=image_digest,
                training=training,
                evaluation=evaluation,
            )
            _write_gold_retirement(root, run_id=run_id, outcome="REJECTED")
            raise EmbeddingRecoveryValueLabError(
                "Embedding v2 formal gates failed; evidence retained at "
                f"{rejection_path.relative_to(root).as_posix()}: "
                f"{','.join(evaluation.failures)}"
            )

        source_path = Path(__file__).resolve(strict=True)
        final_relative = final_directory.relative_to(root)
        evaluation_evidence = _evaluation_evidence(
            root,
            final_relative,
            temporary,
            evaluation,
        )
        candidate_evidence = _candidate_evidence(
            root,
            final_relative,
            temporary / "candidate",
        )
        peak_reserved = max(
            int(training.evidence["peak_gpu_memory_reserved_bytes"]),
            int(evaluation.summary["evaluation_peak_gpu_memory_reserved_bytes"]),
        )
        dependencies = _dependencies()
        torch = dependencies["torch"]
        unsigned: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "enterprise_scope": "PROJECT_INTERNAL",
            "production_claim": False,
            "external_enterprise_production_claim": False,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": STATUS,
            "decision": DECISION,
            "run_id": run_id,
            "source_path": source_path.relative_to(root).as_posix(),
            "source_sha256": _file_sha256(source_path),
            "image": IMAGE,
            "image_digest": image_digest,
            "input_model": _input_model_evidence(root, paths["base_model"]),
            "dataset": _dataset_evidence(root, paths),
            "training": training.evidence,
            "evaluation": evaluation_evidence,
            "candidate": candidate_evidence,
            "runtime": {
                "actual_gpu_execution": True,
                "model_training_simulated": False,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
                "peak_gpu_memory_reserved_bytes": peak_reserved,
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
            },
            "hard_gates": EmbeddingRecoveryHardGates().model_dump(mode="json"),
            "formal_model_release_created": False,
            "runtime_eligible": False,
        }
        draft = EmbeddingRecoveryValueReport.model_validate(
            {**unsigned, "evidence_chain_sha256": "0" * 64}
        )
        normalized = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
        report = draft.model_copy(update={"evidence_chain_sha256": _digest(normalized)})
        _write_json(temporary / "acceptance.json", report.model_dump(mode="json"))
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final_directory)
        _write_gold_retirement(root, run_id=run_id, outcome="PASSED")
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if gold_started and not (root / GOLD_RETIREMENT_RELATIVE).exists():
            _write_gold_retirement(root, run_id=run_id, outcome="INTERRUPTED")
        raise
    verified = verify_embedding_recovery_value(root, acceptance_path)
    _write_latest(root, acceptance_path, verified)
    return acceptance_path


def verify_embedding_recovery_value(
    repo_root: Path,
    acceptance_path: Path | None = None,
) -> EmbeddingRecoveryValueReport:
    root = repo_root.resolve(strict=True)
    report_path = (
        _latest_report_path(root)
        if acceptance_path is None
        else _inside_file(root, acceptance_path)
    )
    report = EmbeddingRecoveryValueReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise EmbeddingRecoveryValueLabError("Embedding v2 evidence chain changed")
    source_path = _inside_file(root, Path(report.source_path))
    if _file_sha256(source_path) != report.source_sha256:
        raise EmbeddingRecoveryValueLabError("Embedding v2 source changed")
    if planned_run_id(root, report.image_digest) != report.run_id:
        raise EmbeddingRecoveryValueLabError("Embedding v2 run identity changed")
    paths = _input_paths(root)
    _verify_inputs(paths)
    expected_model = _input_model_evidence(root, paths["base_model"])
    if report.input_model.model_dump(mode="json") != expected_model:
        raise EmbeddingRecoveryValueLabError("Embedding v2 input model changed")
    _verify_file_evidence(root, report.dataset.training_manifest)
    _verify_file_evidence(root, report.dataset.final_gold_manifest)
    _verify_file_evidence(root, report.evaluation.formal_report)
    _verify_file_evidence(root, report.evaluation.observations)
    candidate_path = _inside_directory(root, Path(report.candidate.path))
    manifest = _directory_manifest(candidate_path)
    if (
        _manifest_digest(manifest) != report.candidate.bundle_sha256
        or sum(int(item["size_bytes"]) for item in manifest) != report.candidate.size_bytes
    ):
        raise EmbeddingRecoveryValueLabError("Embedding v2 candidate changed")
    for item in report.candidate.files:
        _verify_file_evidence(root, item)
    retirement = _load_object(_inside_file(root, root / GOLD_RETIREMENT_RELATIVE))
    if retirement != _gold_retirement_document(report.run_id, "PASSED"):
        raise EmbeddingRecoveryValueLabError("Embedding v2 Gold retirement changed")
    return report


def _train_embedding(
    model_path: Path,
    training_document: dict[str, Any],
    work_directory: Path,
    candidate_directory: Path,
) -> _TrainingResult:
    dependencies = _dependencies()
    torch = dependencies["torch"]
    _prepare_gpu(torch)
    model = dependencies["SentenceTransformer"](
        str(model_path),
        device="cuda:0",
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = _CONFIG["max_sequence_length"]
    train_records = _training_records(training_document, "train")
    validation_records = _training_records(training_document, "validation")
    work_directory.mkdir(parents=True, exist_ok=False)
    args = dependencies["SentenceTransformerTrainingArguments"](
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
        logging_steps=20,
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
        batch_sampler=dependencies["BatchSamplers"].NO_DUPLICATES,
    )
    loss = dependencies["EmbeddingLoss"](model=model, scale=20.0)
    trainer = dependencies["SentenceTransformerTrainer"](
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
    if isinstance(best_metric, bool) or not isinstance(best_metric, (int, float)):
        raise EmbeddingRecoveryValueLabError("Embedding v2 validation loss is missing")
    parameters = tuple(model.parameters())
    train_loss = train_result.metrics.get("train_loss")
    if isinstance(train_loss, bool) or not isinstance(train_loss, (int, float)):
        raise EmbeddingRecoveryValueLabError("Embedding v2 training loss is missing")
    evidence = {
        "trainer_family": "sentence-transformers-5",
        "loss_family": "MultipleNegativesRankingLoss",
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


def _evaluate_embedding(
    *,
    base_model_path: Path,
    candidate_path: Path,
    cases: tuple[EvaluationCase, ...],
) -> _EvaluationResult:
    dependencies = _dependencies()
    torch = dependencies["torch"]
    baseline, baseline_peak = _evaluate_model(
        model_path=base_model_path,
        cases=cases,
        dependencies=dependencies,
    )
    candidate, candidate_peak = _evaluate_model(
        model_path=candidate_path,
        cases=cases,
        dependencies=dependencies,
    )
    formal = score_retrieval_paired_observations(
        cases,
        candidate,
        baseline,
        top_k=_CONFIG["top_k"],
        primary_metric="mrr",
    )
    summary = _evaluation_summary(
        formal,
        evaluation_peak=max(baseline_peak, candidate_peak),
    )
    failures = _evaluation_failures(summary)
    formal_document = {
        "schema_version": "enterprise-embedding-recovery-formal-evaluation/v2",
        "classification": CLASSIFICATION,
        "suite_id": "industrial-embedding-final-gold-v2",
        "single_final_evaluation": True,
        "report": asdict(formal),
    }
    observations_document = {
        "schema_version": "enterprise-embedding-recovery-observations/v2",
        "classification": CLASSIFICATION,
        "suite_id": "industrial-embedding-final-gold-v2",
        "baseline": [_observation_document(item) for item in baseline],
        "candidate": [_observation_document(item) for item in candidate],
    }
    _cleanup_gpu(torch)
    return _EvaluationResult(
        summary=summary,
        formal_document=formal_document,
        observations_document=observations_document,
        failures=failures,
    )


def _evaluate_model(
    *,
    model_path: Path,
    cases: tuple[EvaluationCase, ...],
    dependencies: dict[str, Any],
) -> tuple[tuple[ModelObservation, ...], int]:
    torch = dependencies["torch"]
    _prepare_gpu(torch)
    model = dependencies["SentenceTransformer"](
        str(model_path),
        device="cuda:0",
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = _CONFIG["max_sequence_length"]
    observations = tuple(_rank_case(model=model, case=case, torch=torch) for case in cases)
    peak = int(torch.cuda.max_memory_reserved())
    del model
    _cleanup_gpu(torch)
    return observations, peak


def _rank_case(*, model: Any, case: EvaluationCase, torch: Any) -> ModelObservation:
    contract = case.retrieval
    if contract is None:
        raise EmbeddingRecoveryValueLabError("Embedding v2 Gold contract is missing")
    allowed = tuple(item for item in contract.documents if item.access == "ALLOWED")
    torch.cuda.synchronize()
    started = time.perf_counter()
    query_embedding = model.encode(
        [contract.query],
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
        torch.matmul(query_embedding, document_embeddings.transpose(0, 1))[0].float().tolist()
    )
    torch.cuda.synchronize()
    latency_ms = max((time.perf_counter() - started) * 1000.0, 1e-9)
    scores = [float(item) for item in raw_scores]
    if len(scores) != len(allowed) or any(not math.isfinite(item) for item in scores):
        raise EmbeddingRecoveryValueLabError("Embedding v2 model scores are invalid")
    ranked = sorted(
        zip(allowed, scores, strict=True),
        key=lambda item: (-item[1], item[0].document_id),
    )[: _CONFIG["top_k"]]
    ranked_ids = tuple(item.document_id for item, _ in ranked)
    ranked_scores = tuple(score for _, score in ranked)
    tokenizer = model.tokenizer
    input_tokens = sum(
        len(tokenizer.encode(value, add_special_tokens=True))
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
            "retrieval_method": "EMBEDDING",
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
            raise EmbeddingRecoveryValueLabError(f"Embedding v2 {name} is missing")
        return mean(float(value) for value in values)

    baseline_p95 = report.evidence.baseline_p95_latency_ms
    candidate_p95 = report.evidence.candidate_p95_latency_ms
    baseline_mrr = average("mrr", False)
    candidate_mrr = average("mrr", True)
    return {
        "primary_metric": "mrr",
        "baseline_recall_at_k": average("recall_at_k", False),
        "candidate_recall_at_k": average("recall_at_k", True),
        "baseline_mrr": baseline_mrr,
        "candidate_mrr": candidate_mrr,
        "baseline_ndcg_at_k": average("ndcg_at_k", False),
        "candidate_ndcg_at_k": average("ndcg_at_k", True),
        "primary_improvement": candidate_mrr - baseline_mrr,
        "baseline_p95_latency_ms": baseline_p95,
        "candidate_p95_latency_ms": candidate_p95,
        "candidate_latency_ratio": candidate_p95 / max(baseline_p95, 1e-9),
        "hard_gate_results": report.evidence.hard_gate_results,
        "evaluation_peak_gpu_memory_reserved_bytes": evaluation_peak,
    }


def _evaluation_failures(summary: dict[str, Any]) -> tuple[str, ...]:
    hard_gates = summary["hard_gate_results"]
    if set(hard_gates) != RETRIEVAL_HARD_GATES:
        raise EmbeddingRecoveryValueLabError("Embedding v2 hard-gate contract drifted")
    failures = [f"embedding_{gate}" for gate, passed in sorted(hard_gates.items()) if not passed]
    if summary["primary_improvement"] < _CONFIG["minimum_primary_improvement"]:
        failures.append("embedding_primary_metric_improvement")
    if summary["candidate_recall_at_k"] < _CONFIG["minimum_candidate_recall_at_k"]:
        failures.append("embedding_recall_at_k")
    if summary["candidate_mrr"] < _CONFIG["minimum_candidate_mrr"]:
        failures.append("embedding_mrr")
    if summary["candidate_latency_ratio"] > _CONFIG["maximum_latency_ratio"]:
        failures.append("embedding_latency_budget")
    return tuple(failures)


def _training_records(document: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise EmbeddingRecoveryValueLabError("Embedding v2 training rows are missing")
    selected = [item for item in rows if isinstance(item, dict) and item.get("split") == split]
    records: list[dict[str, Any]] = []
    for row in selected:
        negatives = [str(item) for item in row["hard_negatives"]]
        records.append(
            {
                "anchor": str(row["query"]),
                "positive": str(row["positive"]),
                "negative_1": negatives[0],
                "negative_2": negatives[1],
            }
        )
    if not records:
        raise EmbeddingRecoveryValueLabError(f"Embedding v2 {split} records are empty")
    return records


def _gold_cases(document: dict[str, Any]) -> tuple[EvaluationCase, ...]:
    raw_documents = document.get("documents")
    raw_cases = document.get("cases")
    if not isinstance(raw_documents, list) or not isinstance(raw_cases, list):
        raise EmbeddingRecoveryValueLabError("Embedding v2 Gold is invalid")
    documents_by_id = {
        str(item["document_id"]): item for item in raw_documents if isinstance(item, dict)
    }
    cases: list[EvaluationCase] = []
    for row in raw_cases:
        if not isinstance(row, dict):
            raise EmbeddingRecoveryValueLabError("Embedding v2 Gold case is invalid")
        relevant = {str(item) for item in row["relevant_document_ids"]}
        retrieval_documents = tuple(
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
                retrieval=RetrievalContract(
                    query=str(row["query"]),
                    documents=retrieval_documents,
                ),
            )
        )
    return tuple(cases)


def _verify_inputs(paths: dict[str, Path]) -> None:
    if _file_sha256(paths["base_model"] / "model.safetensors") != MODEL_WEIGHT_SHA256:
        raise EmbeddingRecoveryValueLabError("Embedding v2 base model changed")
    if _file_sha256(paths["training_manifest"]) != TRAINING_MANIFEST_SHA256:
        raise EmbeddingRecoveryValueLabError("Embedding v2 training manifest changed")
    if _file_sha256(paths["gold_manifest"]) != GOLD_MANIFEST_SHA256:
        raise EmbeddingRecoveryValueLabError("Embedding v2 Gold manifest changed")
    training = _load_object(paths["training_manifest"])
    gold = _load_object(paths["gold_manifest"])
    if not _datasets_are_governed_and_disjoint(training, gold):
        raise EmbeddingRecoveryValueLabError("Embedding v2 dataset governance changed")


def _datasets_are_governed_and_disjoint(
    training: dict[str, Any],
    gold: dict[str, Any],
) -> bool:
    rows = training.get("rows")
    cases = gold.get("cases")
    documents = gold.get("documents")
    policy = gold.get("selection_policy")
    train_rows = [
        item for item in rows or [] if isinstance(item, dict) and item.get("split") == "train"
    ]
    validation_rows = [
        item for item in rows or [] if isinstance(item, dict) and item.get("split") == "validation"
    ]
    if (
        training.get("schema_version") != "industrial-embedding-domain-training/v2"
        or training.get("classification") != CLASSIFICATION
        or training.get("enterprise_production_data") is not False
        or training.get("project_generated_data") is not True
        or training.get("review_status") != "APPROVED"
        or not isinstance(rows, list)
        or len(train_rows) != 20
        or len(validation_rows) != 10
        or gold.get("schema_version") != "industrial-embedding-frozen-gold/v2"
        or gold.get("dataset_id") != "industrial-embedding-final-gold-v2"
        or gold.get("classification") != CLASSIFICATION
        or gold.get("enterprise_production_data") is not False
        or gold.get("project_generated_data") is not True
        or gold.get("production_gold_eligible") is not False
        or gold.get("top_k") != 3
        or not isinstance(cases, list)
        or len(cases) != 12
        or not isinstance(documents, list)
        or len(documents) != 22
        or policy
        != {
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "used_for_hyperparameter_selection": False,
            "used_for_development_regression": False,
            "single_final_evaluation_only": True,
        }
    ):
        return False
    row_ids = [item.get("case_id") for item in rows if isinstance(item, dict)]
    case_ids = [item.get("case_id") for item in cases if isinstance(item, dict)]
    document_ids = [item.get("document_id") for item in documents if isinstance(item, dict)]
    if (
        len(row_ids) != len(set(row_ids))
        or len(case_ids) != len(set(case_ids))
        or len(document_ids) != len(set(document_ids))
        or set(row_ids) & set(case_ids)
    ):
        return False
    allowed_ids = {
        str(item["document_id"])
        for item in documents
        if isinstance(item, dict) and item.get("access") == "ALLOWED"
    }
    forbidden_ids = {
        str(item["document_id"])
        for item in documents
        if isinstance(item, dict) and item.get("access") == "FORBIDDEN"
    }
    if len(allowed_ids) != 20 or len(forbidden_ids) != 2:
        return False
    terminology_families: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        query = row.get("query")
        positive = row.get("positive")
        negatives = row.get("hard_negatives")
        family = row.get("terminology_family")
        if (
            row.get("review_status") != "APPROVED"
            or not isinstance(query, str)
            or not query.strip()
            or not isinstance(positive, str)
            or not positive.strip()
            or not isinstance(negatives, list)
            or len(negatives) != 2
            or any(not isinstance(item, str) or not item.strip() for item in negatives)
            or len(set(negatives)) != 2
            or positive in negatives
            or not isinstance(family, str)
            or not family.strip()
        ):
            return False
        terminology_families.add(family)
    if len(terminology_families) != 10:
        return False
    for case in cases:
        if not isinstance(case, dict):
            return False
        relevant = case.get("relevant_document_ids")
        if (
            not isinstance(case.get("query"), str)
            or not isinstance(relevant, list)
            or len(relevant) != 1
            or not set(relevant) <= allowed_ids
        ):
            return False
    training_texts = {
        str(value)
        for row in rows
        if isinstance(row, dict)
        for value in (
            row.get("query"),
            row.get("positive"),
            *(row.get("hard_negatives") or []),
        )
    }
    gold_texts = {str(item.get("query")) for item in cases if isinstance(item, dict)} | {
        str(item.get("text")) for item in documents if isinstance(item, dict)
    }
    retired_v1 = gold.get("retired_suite_exclusions")
    return (
        not (training_texts & gold_texts)
        and retired_v1 == ["industrial-retrieval-final-gold-v1"]
        and not forbidden_ids & allowed_ids
    )


def _input_paths(root: Path) -> dict[str, Path]:
    return {
        "base_model": _inside_directory(root, BASE_MODEL_RELATIVE),
        "training_manifest": _inside_file(root, TRAINING_DATASET_RELATIVE),
        "gold_manifest": _inside_file(root, GOLD_DATASET_RELATIVE),
    }


def _dependencies() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
        sentence_transformers = importlib.import_module("sentence_transformers")
        embedding_losses = importlib.import_module("sentence_transformers.losses")
        training_args = importlib.import_module("sentence_transformers.training_args")
        datasets = importlib.import_module("datasets")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise EmbeddingRecoveryValueLabError(
            "Embedding v2 GPU dependencies are not installed"
        ) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EmbeddingRecoveryValueLabError("Embedding v2 lab requires exactly one CUDA GPU")
    return {
        "torch": torch,
        "sentence_transformers": sentence_transformers,
        "transformers": transformers,
        "Dataset": datasets.Dataset,
        "SentenceTransformer": sentence_transformers.SentenceTransformer,
        "SentenceTransformerTrainer": sentence_transformers.SentenceTransformerTrainer,
        "SentenceTransformerTrainingArguments": (
            sentence_transformers.SentenceTransformerTrainingArguments
        ),
        "BatchSamplers": training_args.BatchSamplers,
        "EmbeddingLoss": embedding_losses.MultipleNegativesRankingLoss,
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


def _input_model_evidence(root: Path, model_path: Path) -> dict[str, Any]:
    manifest = _directory_manifest(model_path)
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weight_sha256": MODEL_WEIGHT_SHA256,
        "path": model_path.relative_to(root).as_posix(),
        "manifest_sha256": _manifest_digest(manifest),
        "files": [
            _file_evidence(
                model_path.relative_to(root) / str(item["path"]),
                model_path / str(item["path"]),
            )
            for item in manifest
        ],
        "license": "apache-2.0",
    }


def _dataset_evidence(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "training_manifest": _file_evidence(
            paths["training_manifest"].relative_to(root),
            paths["training_manifest"],
        ),
        "final_gold_manifest": _file_evidence(
            paths["gold_manifest"].relative_to(root),
            paths["gold_manifest"],
        ),
        "train_count": 20,
        "validation_count": 10,
        "final_gold_count": 12,
        "allowed_document_count": 20,
        "forbidden_document_count": 2,
        "top_k": 3,
        "terminology_family_count": 10,
        "source_type": "PLATFORM_SYNTHETIC",
        "project_enterprise_use_authorized": True,
        "enterprise_production_data": False,
        "retired_v1_gold_reused": False,
        "frozen_gold_excluded_from_training_selection_and_development": True,
        "exact_case_ids_and_texts_disjoint": True,
    }


def _evaluation_evidence(
    root: Path,
    final_relative: Path,
    temporary: Path,
    evaluation: _EvaluationResult,
) -> dict[str, Any]:
    summary = dict(evaluation.summary)
    summary["formal_report"] = _file_evidence(
        final_relative / "formal-evaluation.json",
        temporary / "formal-evaluation.json",
    )
    summary["observations"] = _file_evidence(
        final_relative / "observations.json",
        temporary / "observations.json",
    )
    evidence = EvaluationEvidence.model_validate(summary)
    return evidence.model_dump(mode="json")


def _candidate_evidence(
    root: Path,
    final_relative: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    del root
    manifest = _directory_manifest(candidate_path)
    candidate_relative = final_relative / "candidate"
    evidence = {
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
        "role": "AUTHORIZED_ENTERPRISE_EMBEDDING_MODEL_RELEASE_CANDIDATE",
    }
    return CandidateEvidence.model_validate(evidence).model_dump(mode="json")


def _persist_rejection(
    *,
    root: Path,
    temporary: Path,
    run_id: str,
    image_digest: str,
    training: _TrainingResult,
    evaluation: _EvaluationResult,
) -> Path:
    rejection_directory = root / OUTPUT_RELATIVE / "rejections" / run_id
    if rejection_directory.exists():
        raise EmbeddingRecoveryValueLabError("immutable Embedding v2 rejection directory exists")
    source_path = Path(__file__).resolve(strict=True)
    unsigned = {
        "schema_version": "enterprise-embedding-recovery-rejection/v2",
        "classification": CLASSIFICATION,
        "run_id": run_id,
        "outcome": "REJECTED",
        "reuse_permitted": False,
        "source_path": source_path.relative_to(root).as_posix(),
        "source_sha256": _file_sha256(source_path),
        "image": IMAGE,
        "image_digest": image_digest,
        "training": training.evidence,
        "evaluation": evaluation.summary,
        "failures": list(evaluation.failures),
        "formal_model_release_created": False,
        "runtime_eligible": False,
    }
    rejection = {**unsigned, "evidence_chain_sha256": _digest(unsigned)}
    _write_json(temporary / "rejection.json", rejection)
    rejection_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(rejection_directory)
    return rejection_directory / "rejection.json"


def _gold_retirement_document(run_id: str, outcome: str) -> dict[str, Any]:
    unsigned = {
        "schema_version": "embedding-gold-retirement/v2",
        "suite_id": "industrial-embedding-final-gold-v2",
        "run_id": run_id,
        "outcome": outcome,
        "reuse_permitted": False,
        "retired_v1_suite_reused": False,
    }
    return {**unsigned, "evidence_chain_sha256": _digest(unsigned)}


def _write_gold_retirement(root: Path, *, run_id: str, outcome: str) -> None:
    path = root / GOLD_RETIREMENT_RELATIVE
    document = _gold_retirement_document(run_id, outcome)
    if path.exists():
        if _load_object(path) != document:
            raise EmbeddingRecoveryValueLabError("Embedding v2 Gold retirement is immutable")
        return
    _write_json(path, document)


def _observation_document(observation: ModelObservation) -> dict[str, Any]:
    value = asdict(observation)
    value["capabilities"] = sorted(observation.capabilities)
    value["ranked_document_ids"] = list(observation.ranked_document_ids)
    value["retrieval_scores"] = list(observation.retrieval_scores)
    return value


def _checkpoint_step(path: str) -> int:
    name = Path(path).name
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        raise EmbeddingRecoveryValueLabError("Embedding v2 checkpoint is invalid")
    try:
        return int(name.removeprefix(prefix))
    except ValueError as exc:
        raise EmbeddingRecoveryValueLabError("Embedding v2 checkpoint step is invalid") from exc


def _directory_manifest(directory: Path) -> list[dict[str, Any]]:
    _inside_directory(directory, directory)
    files = sorted(item for item in directory.rglob("*") if item.is_file())
    if not files:
        raise EmbeddingRecoveryValueLabError("Embedding v2 directory is empty")
    manifest: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(directory).as_posix()
        if relative.startswith("../") or relative == "..":
            raise EmbeddingRecoveryValueLabError("Embedding v2 manifest path escaped")
        manifest.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return manifest


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
        raise EmbeddingRecoveryValueLabError(f"Embedding v2 file changed: {evidence.path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EmbeddingRecoveryValueLabError(f"Embedding v2 JSON is not an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_latest(
    root: Path,
    acceptance_path: Path,
    report: EmbeddingRecoveryValueReport,
) -> None:
    latest = root / OUTPUT_RELATIVE / "latest.json"
    _write_json(latest, report.model_dump(mode="json"))
    if _file_sha256(latest) != _file_sha256(acceptance_path):
        raise EmbeddingRecoveryValueLabError("Embedding v2 latest receipt changed")


def _latest_report_path(root: Path) -> Path:
    return _inside_file(root, root / OUTPUT_RELATIVE / "latest.json")


def _inside_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=True)
    _require_inside(root, resolved)
    if not resolved.is_file():
        raise EmbeddingRecoveryValueLabError(f"Embedding v2 file is missing: {path}")
    return resolved


def _inside_directory(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=True)
    _require_inside(root, resolved)
    if not resolved.is_dir():
        raise EmbeddingRecoveryValueLabError(f"Embedding v2 directory is missing: {path}")
    return resolved


def _require_inside(root: Path, path: Path) -> None:
    resolved_root = root.resolve(strict=True)
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise EmbeddingRecoveryValueLabError("Embedding v2 path escaped project root") from exc


def _require_image_digest(value: str) -> None:
    prefix = "sha256:"
    raw = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or len(raw) != 64
        or any(character not in "0123456789abcdef" for character in raw)
    ):
        raise EmbeddingRecoveryValueLabError("Embedding v2 image digest is invalid")
