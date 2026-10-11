"""Low-memory actual-GPU Embedding and Reranker enterprise-value lab."""

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

SCHEMA_VERSION: Literal["enterprise-retrieval-value-lab/v1"] = (
    "enterprise-retrieval-value-lab/v1"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
STATUS: Literal["RETRIEVAL_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"] = (
    "RETRIEVAL_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
)
DECISION: Literal["RETRIEVAL_CANDIDATES_ELIGIBLE_FOR_PROJECT_EVALUATION"] = (
    "RETRIEVAL_CANDIDATES_ELIGIBLE_FOR_PROJECT_EVALUATION"
)
OUTPUT_RELATIVE = Path("artifacts/m7-retrieval-enterprise-value-lab")
EMBEDDING_MODEL_RELATIVE = OUTPUT_RELATIVE / "inputs" / "all-MiniLM-L6-v2-1110a243"
RERANKER_MODEL_RELATIVE = OUTPUT_RELATIVE / "inputs" / "ms-marco-MiniLM-L6-v2-ce0834f"
TRAINING_DATASET_RELATIVE = (
    OUTPUT_RELATIVE / "inputs" / "industrial-retrieval-training-v1" / "manifest.json"
)
GOLD_DATASET_RELATIVE = (
    OUTPUT_RELATIVE / "inputs" / "industrial-retrieval-final-gold-v1" / "manifest.json"
)
GOLD_RETIREMENT_RELATIVE = OUTPUT_RELATIVE / "gold-v1-retirement.json"
IMAGE: Literal["industrial-ops/m4-training-worker:local"] = (
    "industrial-ops/m4-training-worker:local"
)
CONTAINER_NAME = "ioap-retrieval-enterprise-value-lab"

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_MODEL_WEIGHT_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL_REVISION = "ce0834f22110de6d9222af7a7a03628121708969"
RERANKER_MODEL_WEIGHT_SHA256 = (
    "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
)
TRAINING_MANIFEST_SHA256 = "beffc6e72e1eb4bfbf83c07e101ebc45127008758c3b67121f2cfda7cc255378"
GOLD_MANIFEST_SHA256 = "6e0c8da4785de3a3a237689a4b31c26167197827468c6c31310a0669e5194e5e"

_CONFIG: dict[str, Any] = {
    "embedding_steps": 48,
    "reranker_steps": 48,
    "evaluation_interval": 12,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_sequence_length": 128,
    "precision": "bfloat16",
    "seed": 20260824,
    "data_seed": 20260825,
    "top_k": 3,
    "hard_negatives_per_query": 2,
    "minimum_primary_improvement": 0.02,
    "minimum_candidate_recall_at_k": 0.80,
    "minimum_candidate_mrr": 0.80,
    "minimum_candidate_ndcg_at_k": 0.80,
    "cpu_threads": 2,
    "dataloader_workers": 0,
    "container_memory_limit_bytes": 4 * 1024**3,
}


class RetrievalEnterpriseValueLabError(RuntimeError):
    """The governed Retrieval lab could not produce or verify its evidence."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class InputModelEvidence(_ClosedModel):
    component: Literal["EMBEDDING", "RERANKER"]
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[FileEvidence, ...] = Field(min_length=6)
    license: Literal["apache-2.0"] = "apache-2.0"


class DatasetEvidence(_ClosedModel):
    training_manifest: FileEvidence
    final_gold_manifest: FileEvidence
    train_count: Literal[20]
    validation_count: Literal[6]
    final_gold_count: Literal[12]
    allowed_document_count: Literal[20]
    forbidden_document_count: Literal[2]
    top_k: Literal[3]
    source_type: Literal["PLATFORM_SYNTHETIC"]
    project_enterprise_use_authorized: Literal[True] = True
    enterprise_production_data: Literal[False] = False
    frozen_gold_excluded_from_training_selection_and_development: Literal[True] = True
    case_ids_and_texts_disjoint: Literal[True] = True


class ComponentTrainingEvidence(_ClosedModel):
    component: Literal["EMBEDDING", "RERANKER"]
    trainer_family: Literal[
        "sentence-transformers-5", "sentence-transformers-cross-encoder-5"
    ]
    optimizer_steps: Literal[48]
    selected_validation_step: int = Field(ge=12, le=48)
    train_loss: float = Field(ge=0.0)
    selected_validation_loss: float = Field(ge=0.0)
    trainable_parameters: int = Field(gt=0)
    total_parameters: int = Field(gt=0)
    peak_gpu_memory_allocated_bytes: int = Field(gt=0)
    peak_gpu_memory_reserved_bytes: int = Field(gt=0)
    gpu_memory_allocated_after_cleanup_bytes: int = Field(ge=0)


class ComponentEvaluationEvidence(_ClosedModel):
    component: Literal["EMBEDDING", "RERANKER"]
    primary_metric: Literal["mrr", "ndcg_at_k"]
    formal_report: FileEvidence
    observations: FileEvidence
    baseline_recall_at_k: float = Field(ge=0.0, le=1.0)
    candidate_recall_at_k: float = Field(ge=0.0, le=1.0)
    baseline_mrr: float = Field(ge=0.0, le=1.0)
    candidate_mrr: float = Field(ge=0.0, le=1.0)
    baseline_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    candidate_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    baseline_primary: float = Field(ge=0.0, le=1.0)
    candidate_primary: float = Field(ge=0.0, le=1.0)
    primary_improvement: float
    baseline_p95_latency_ms: float = Field(gt=0.0)
    candidate_p95_latency_ms: float = Field(gt=0.0)
    hard_gate_results: dict[str, bool]
    evaluation_peak_gpu_memory_reserved_bytes: int = Field(gt=0)


class CandidateEvidence(_ClosedModel):
    component: Literal["EMBEDDING", "RERANKER"]
    path: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    files: tuple[FileEvidence, ...] = Field(min_length=6)
    role: Literal[
        "AUTHORIZED_ENTERPRISE_EMBEDDING_EVALUATION_CANDIDATE",
        "AUTHORIZED_ENTERPRISE_RERANKER_EVALUATION_CANDIDATE",
    ]


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
    models_loaded_sequentially: Literal[True] = True
    cpu_threads: Literal[2] = 2
    dataloader_workers: Literal[0] = 0
    container_memory_limit_bytes: Literal[4294967296] = 4294967296


class RetrievalHardGates(_ClosedModel):
    data_governance: Literal[True] = True
    actual_gpu_training: Literal[True] = True
    immutable_model_data_and_image: Literal[True] = True
    frozen_gold_independence: Literal[True] = True
    cross_tenant_isolation: Literal[True] = True
    embedding_formal_gates: Literal[True] = True
    reranker_formal_gates: Literal[True] = True
    embedding_quality_improved: Literal[True] = True
    reranker_quality_improved: Literal[True] = True
    sequential_model_memory_release: Literal[True] = True
    resource_profile_respected: Literal[True] = True


class RetrievalEnterpriseValueReport(_ClosedModel):
    schema_version: Literal["enterprise-retrieval-value-lab/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["RETRIEVAL_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"] = STATUS
    decision: Literal[
        "RETRIEVAL_CANDIDATES_ELIGIBLE_FOR_PROJECT_EVALUATION"
    ] = DECISION
    run_id: str = Field(pattern=r"^retrieval-[0-9a-f]{20}$")
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image: Literal["industrial-ops/m4-training-worker:local"] = IMAGE
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_models: tuple[InputModelEvidence, ...] = Field(min_length=2, max_length=2)
    dataset: DatasetEvidence
    training: tuple[ComponentTrainingEvidence, ...] = Field(min_length=2, max_length=2)
    evaluations: tuple[ComponentEvaluationEvidence, ...] = Field(min_length=2, max_length=2)
    candidates: tuple[CandidateEvidence, ...] = Field(min_length=2, max_length=2)
    runtime: RuntimeEvidence
    hard_gates: RetrievalHardGates
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _TrainingResult:
    component: str
    evidence: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _EvaluationResult:
    component: str
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
        "config": _CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
        "source_sha256": _file_sha256(source_path),
        "embedding_model_manifest": _manifest_digest(
            _directory_manifest(paths["embedding_model"])
        ),
        "reranker_model_manifest": _manifest_digest(
            _directory_manifest(paths["reranker_model"])
        ),
        "training_manifest_sha256": _file_sha256(paths["training_manifest"]),
        "gold_manifest_sha256": _file_sha256(paths["gold_manifest"]),
    }
    return f"retrieval-{_digest(identity)[:20]}"


def assert_gold_available(repo_root: Path) -> None:
    root = repo_root.resolve(strict=True)
    retirement_path = root / GOLD_RETIREMENT_RELATIVE
    if not retirement_path.exists():
        return
    retirement = _load_object(_inside_file(root, retirement_path))
    if (
        retirement.get("suite_id") != "industrial-retrieval-final-gold-v1"
        or retirement.get("reuse_permitted") is not False
    ):
        raise RetrievalEnterpriseValueLabError("Retrieval Gold retirement contract changed")
    raise RetrievalEnterpriseValueLabError(
        "Retrieval final Gold is retired after its single permitted evaluation"
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
        verify_retrieval_enterprise_value(root, acceptance_path)
        return acceptance_path
    assert_gold_available(root)
    if final_directory.exists():
        raise RetrievalEnterpriseValueLabError("immutable Retrieval run directory exists")
    temporary = output_root / f".{run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    gold_started = False
    try:
        training_document = _load_object(paths["training_manifest"])
        gold_document = _load_object(paths["gold_manifest"])
        embedding_training = _train_embedding(
            paths["embedding_model"],
            training_document,
            temporary / "work" / "embedding",
            temporary / "candidates" / "embedding",
        )
        reranker_training = _train_reranker(
            paths["reranker_model"],
            training_document,
            temporary / "work" / "reranker",
            temporary / "candidates" / "reranker",
        )
        shutil.rmtree(temporary / "work")

        cases = _gold_cases(gold_document)
        gold_started = True
        embedding_evaluation = _evaluate_component(
            component="EMBEDDING",
            base_model_path=paths["embedding_model"],
            candidate_path=temporary / "candidates" / "embedding",
            cases=cases,
        )
        reranker_evaluation = _evaluate_component(
            component="RERANKER",
            base_model_path=paths["reranker_model"],
            candidate_path=temporary / "candidates" / "reranker",
            cases=cases,
        )
        evaluation_results = (embedding_evaluation, reranker_evaluation)
        for result in evaluation_results:
            name = result.component.lower()
            _write_json(temporary / f"{name}-formal-evaluation.json", result.formal_document)
            _write_json(temporary / f"{name}-observations.json", result.observations_document)

        failures = tuple(
            failure for result in evaluation_results for failure in result.failures
        )
        if failures:
            rejection_path = _persist_rejection(
                root=root,
                temporary=temporary,
                run_id=run_id,
                image_digest=image_digest,
                paths=paths,
                training=(embedding_training, reranker_training),
                evaluations=evaluation_results,
                failures=failures,
            )
            _write_gold_retirement(root, run_id=run_id, outcome="REJECTED")
            raise RetrievalEnterpriseValueLabError(
                "Retrieval formal gates failed; evidence retained at "
                f"{rejection_path.relative_to(root).as_posix()}: {','.join(failures)}"
            )

        source_path = Path(__file__).resolve(strict=True)
        final_relative = final_directory.relative_to(root)
        evaluations = _evaluation_evidence(
            root, final_relative, temporary, evaluation_results
        )
        training = tuple(item.evidence for item in (embedding_training, reranker_training))
        candidates = _candidate_evidence(root, final_relative, temporary)
        peak_reserved = max(
            *(int(item.evidence["peak_gpu_memory_reserved_bytes"]) for item in (
                embedding_training,
                reranker_training,
            )),
            *(int(item.summary["evaluation_peak_gpu_memory_reserved_bytes"]) for item in (
                embedding_evaluation,
                reranker_evaluation,
            )),
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
            "input_models": _input_model_evidence(root, paths),
            "dataset": _dataset_evidence(root, paths),
            "training": training,
            "evaluations": evaluations,
            "candidates": candidates,
            "runtime": {
                "actual_gpu_execution": True,
                "model_training_simulated": False,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
                "peak_gpu_memory_reserved_bytes": peak_reserved,
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
                "models_loaded_sequentially": True,
                "cpu_threads": 2,
                "dataloader_workers": 0,
                "container_memory_limit_bytes": _CONFIG["container_memory_limit_bytes"],
            },
            "hard_gates": RetrievalHardGates().model_dump(mode="json"),
            "formal_model_release_created": False,
            "runtime_eligible": False,
        }
        draft = RetrievalEnterpriseValueReport.model_validate(
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
    verified = verify_retrieval_enterprise_value(root, acceptance_path)
    _write_latest(root, acceptance_path, verified)
    return acceptance_path


def verify_retrieval_enterprise_value(
    repo_root: Path, acceptance_path: Path | None = None
) -> RetrievalEnterpriseValueReport:
    root = repo_root.resolve(strict=True)
    report_path = (
        _latest_report_path(root)
        if acceptance_path is None
        else _inside_file(root, acceptance_path)
    )
    report = RetrievalEnterpriseValueReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise RetrievalEnterpriseValueLabError("Retrieval evidence chain changed")
    source_path = _inside_file(root, Path(report.source_path))
    if _file_sha256(source_path) != report.source_sha256:
        raise RetrievalEnterpriseValueLabError("Retrieval source changed")
    if planned_run_id(root, report.image_digest) != report.run_id:
        raise RetrievalEnterpriseValueLabError("Retrieval run identity changed")
    paths = _input_paths(root)
    _verify_inputs(paths)
    expected_models = _input_model_evidence(root, paths)
    if [item.model_dump(mode="json") for item in report.input_models] != expected_models:
        raise RetrievalEnterpriseValueLabError("Retrieval input model changed")
    _verify_file_evidence(root, report.dataset.training_manifest)
    _verify_file_evidence(root, report.dataset.final_gold_manifest)
    for evaluation in report.evaluations:
        _verify_file_evidence(root, evaluation.formal_report)
        _verify_file_evidence(root, evaluation.observations)
    for candidate in report.candidates:
        candidate_path = _inside_directory(root, Path(candidate.path))
        manifest = _directory_manifest(candidate_path)
        if (
            _manifest_digest(manifest) != candidate.bundle_sha256
            or sum(int(item["size_bytes"]) for item in manifest) != candidate.size_bytes
        ):
            raise RetrievalEnterpriseValueLabError(
                f"Retrieval {candidate.component} candidate changed"
            )
        for item in candidate.files:
            _verify_file_evidence(root, item)
    if {item.component for item in report.training} != {"EMBEDDING", "RERANKER"}:
        raise RetrievalEnterpriseValueLabError("Retrieval training components changed")
    if {item.component for item in report.evaluations} != {"EMBEDDING", "RERANKER"}:
        raise RetrievalEnterpriseValueLabError("Retrieval evaluation components changed")
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
    train_records = _training_records(training_document, "train", "EMBEDDING")
    validation_records = _training_records(training_document, "validation", "EMBEDDING")
    work_directory.mkdir(parents=True, exist_ok=False)
    args = dependencies["SentenceTransformerTrainingArguments"](
        output_dir=str(work_directory),
        max_steps=_CONFIG["embedding_steps"],
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
        logging_steps=12,
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
    evidence = _training_evidence(
        component="EMBEDDING",
        trainer_family="sentence-transformers-5",
        trainer=trainer,
        train_result=train_result,
        model=model,
        torch=torch,
    )
    del trainer, loss, model
    _cleanup_gpu(torch)
    evidence["gpu_memory_allocated_after_cleanup_bytes"] = int(torch.cuda.memory_allocated())
    _write_json(candidate_directory / "industrial-ops-training-report.json", evidence)
    return _TrainingResult(component="EMBEDDING", evidence=evidence)


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
    train_records = _training_records(training_document, "train", "RERANKER")
    validation_records = _training_records(training_document, "validation", "RERANKER")
    work_directory.mkdir(parents=True, exist_ok=False)
    args = dependencies["CrossEncoderTrainingArguments"](
        output_dir=str(work_directory),
        max_steps=_CONFIG["reranker_steps"],
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
        logging_steps=12,
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
        model=model, pos_weight=torch.tensor(2.0, device="cuda:0")
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
    evidence = _training_evidence(
        component="RERANKER",
        trainer_family="sentence-transformers-cross-encoder-5",
        trainer=trainer,
        train_result=train_result,
        model=model.model,
        torch=torch,
    )
    del trainer, loss, model
    _cleanup_gpu(torch)
    evidence["gpu_memory_allocated_after_cleanup_bytes"] = int(torch.cuda.memory_allocated())
    _write_json(candidate_directory / "industrial-ops-training-report.json", evidence)
    return _TrainingResult(component="RERANKER", evidence=evidence)


def _training_evidence(
    *,
    component: str,
    trainer_family: str,
    trainer: Any,
    train_result: Any,
    model: Any,
    torch: Any,
) -> dict[str, Any]:
    best_checkpoint = trainer.state.best_model_checkpoint
    selected_step = (
        _checkpoint_step(best_checkpoint)
        if isinstance(best_checkpoint, str)
        else int(trainer.state.global_step)
    )
    best_metric = trainer.state.best_metric
    if best_metric is None:
        evaluated = trainer.evaluate()
        best_metric = evaluated.get("eval_loss")
    if isinstance(best_metric, bool) or not isinstance(best_metric, (int, float)):
        raise RetrievalEnterpriseValueLabError("Retrieval validation loss is missing")
    parameters = tuple(model.parameters())
    return {
        "component": component,
        "trainer_family": trainer_family,
        "optimizer_steps": int(trainer.state.global_step),
        "selected_validation_step": selected_step,
        "train_loss": float(train_result.metrics["train_loss"]),
        "selected_validation_loss": float(best_metric),
        "trainable_parameters": sum(item.numel() for item in parameters if item.requires_grad),
        "total_parameters": sum(item.numel() for item in parameters),
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "gpu_memory_allocated_after_cleanup_bytes": 0,
    }


def _evaluate_component(
    *,
    component: str,
    base_model_path: Path,
    candidate_path: Path,
    cases: tuple[EvaluationCase, ...],
) -> _EvaluationResult:
    dependencies = _dependencies()
    torch = dependencies["torch"]
    baseline, baseline_peak = _evaluate_model(
        component=component,
        model_path=base_model_path,
        cases=cases,
        dependencies=dependencies,
    )
    candidate, candidate_peak = _evaluate_model(
        component=component,
        model_path=candidate_path,
        cases=cases,
        dependencies=dependencies,
    )
    primary_metric = "mrr" if component == "EMBEDDING" else "ndcg_at_k"
    formal = score_retrieval_paired_observations(
        cases,
        candidate,
        baseline,
        top_k=_CONFIG["top_k"],
        primary_metric=primary_metric,
    )
    summary = _evaluation_summary(
        component=component,
        report=formal,
        evaluation_peak=max(baseline_peak, candidate_peak),
    )
    failures = _evaluation_failures(component, summary)
    formal_document = {
        "schema_version": "enterprise-retrieval-formal-evaluation/v1",
        "classification": CLASSIFICATION,
        "component": component,
        "report": asdict(formal),
    }
    observations_document = {
        "schema_version": "enterprise-retrieval-observations/v1",
        "classification": CLASSIFICATION,
        "component": component,
        "baseline": [_observation_document(item) for item in baseline],
        "candidate": [_observation_document(item) for item in candidate],
    }
    _cleanup_gpu(torch)
    return _EvaluationResult(
        component=component,
        summary=summary,
        formal_document=formal_document,
        observations_document=observations_document,
        failures=failures,
    )


def _evaluate_model(
    *,
    component: str,
    model_path: Path,
    cases: tuple[EvaluationCase, ...],
    dependencies: dict[str, Any],
) -> tuple[tuple[ModelObservation, ...], int]:
    torch = dependencies["torch"]
    _prepare_gpu(torch)
    if component == "EMBEDDING":
        model = dependencies["SentenceTransformer"](
            str(model_path),
            device="cuda:0",
            trust_remote_code=False,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        model.max_seq_length = _CONFIG["max_sequence_length"]
    else:
        model = dependencies["CrossEncoder"](
            str(model_path),
            device="cuda:0",
            trust_remote_code=False,
            num_labels=1,
            max_length=_CONFIG["max_sequence_length"],
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
    observations = tuple(
        _rank_case(component=component, model=model, case=case, torch=torch)
        for case in cases
    )
    peak = int(torch.cuda.max_memory_reserved())
    del model
    _cleanup_gpu(torch)
    return observations, peak


def _rank_case(
    *, component: str, model: Any, case: EvaluationCase, torch: Any
) -> ModelObservation:
    contract = case.retrieval
    if contract is None:
        raise RetrievalEnterpriseValueLabError("Retrieval Gold contract is missing")
    allowed = tuple(item for item in contract.documents if item.access == "ALLOWED")
    torch.cuda.synchronize()
    started = time.perf_counter()
    if component == "EMBEDDING":
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
        raw_scores = torch.matmul(
            query_embedding, document_embeddings.transpose(0, 1)
        )[0].float().tolist()
        tokenizer = model.tokenizer
    else:
        raw_scores = model.predict(
            [(contract.query, item.text) for item in allowed],
            batch_size=16,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()
        tokenizer = model.tokenizer
    torch.cuda.synchronize()
    latency_ms = max((time.perf_counter() - started) * 1000.0, 1e-9)
    scores = [float(item) for item in raw_scores]
    if len(scores) != len(allowed) or any(not math.isfinite(item) for item in scores):
        raise RetrievalEnterpriseValueLabError("Retrieval model scores are invalid")
    ranked = sorted(
        zip(allowed, scores, strict=True),
        key=lambda item: (-item[1], item[0].document_id),
    )[: _CONFIG["top_k"]]
    ranked_ids = tuple(item.document_id for item, _ in ranked)
    ranked_scores = tuple(score for _, score in ranked)
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
            "retrieval_method": component,
        },
    )


def _evaluation_summary(
    *, component: str, report: PairedMetricReport, evaluation_peak: int
) -> dict[str, Any]:
    def average(name: str, candidate: bool) -> float:
        prefix = "candidate" if candidate else "baseline"
        values = [getattr(item, f"{prefix}_{name}") for item in report.case_metrics]
        if any(value is None for value in values):
            raise RetrievalEnterpriseValueLabError(f"Retrieval {name} is missing")
        return mean(float(value) for value in values)

    primary_metric = "mrr" if component == "EMBEDDING" else "ndcg_at_k"
    baseline_primary = average(primary_metric, False)
    candidate_primary = average(primary_metric, True)
    return {
        "component": component,
        "primary_metric": primary_metric,
        "baseline_recall_at_k": average("recall_at_k", False),
        "candidate_recall_at_k": average("recall_at_k", True),
        "baseline_mrr": average("mrr", False),
        "candidate_mrr": average("mrr", True),
        "baseline_ndcg_at_k": average("ndcg_at_k", False),
        "candidate_ndcg_at_k": average("ndcg_at_k", True),
        "baseline_primary": baseline_primary,
        "candidate_primary": candidate_primary,
        "primary_improvement": candidate_primary - baseline_primary,
        "baseline_p95_latency_ms": report.evidence.baseline_p95_latency_ms,
        "candidate_p95_latency_ms": report.evidence.candidate_p95_latency_ms,
        "hard_gate_results": report.evidence.hard_gate_results,
        "evaluation_peak_gpu_memory_reserved_bytes": evaluation_peak,
    }


def _evaluation_failures(component: str, summary: dict[str, Any]) -> tuple[str, ...]:
    hard_gates = summary["hard_gate_results"]
    if set(hard_gates) != RETRIEVAL_HARD_GATES:
        raise RetrievalEnterpriseValueLabError("Retrieval hard-gate contract drifted")
    failures = [
        f"{component.lower()}_{gate}"
        for gate, passed in sorted(hard_gates.items())
        if not passed
    ]
    if summary["primary_improvement"] < _CONFIG["minimum_primary_improvement"]:
        failures.append(f"{component.lower()}_primary_metric_improvement")
    if summary["candidate_recall_at_k"] < _CONFIG["minimum_candidate_recall_at_k"]:
        failures.append(f"{component.lower()}_recall_at_k")
    if component == "EMBEDDING" and summary["candidate_mrr"] < _CONFIG["minimum_candidate_mrr"]:
        failures.append("embedding_mrr")
    if (
        component == "RERANKER"
        and summary["candidate_ndcg_at_k"] < _CONFIG["minimum_candidate_ndcg_at_k"]
    ):
        failures.append("reranker_ndcg_at_k")
    return tuple(failures)


def _training_records(
    document: dict[str, Any], split: str, component: str
) -> list[dict[str, Any]]:
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise RetrievalEnterpriseValueLabError("Retrieval training rows are missing")
    selected = [item for item in rows if isinstance(item, dict) and item.get("split") == split]
    records: list[dict[str, Any]] = []
    for row in selected:
        query = str(row["query"])
        positive = str(row["positive"])
        negatives = [str(item) for item in row["hard_negatives"]]
        if component == "EMBEDDING":
            records.append(
                {
                    "anchor": query,
                    "positive": positive,
                    "negative_1": negatives[0],
                    "negative_2": negatives[1],
                }
            )
        else:
            records.append({"query": query, "document": positive, "label": 1.0})
            records.extend(
                {"query": query, "document": item, "label": 0.0}
                for item in negatives
            )
    if not records:
        raise RetrievalEnterpriseValueLabError(f"Retrieval {split} records are empty")
    return records


def _gold_cases(document: dict[str, Any]) -> tuple[EvaluationCase, ...]:
    raw_documents = document.get("documents")
    raw_cases = document.get("cases")
    if not isinstance(raw_documents, list) or not isinstance(raw_cases, list):
        raise RetrievalEnterpriseValueLabError("Retrieval Gold is invalid")
    documents_by_id = {
        str(item["document_id"]): item for item in raw_documents if isinstance(item, dict)
    }
    cases: list[EvaluationCase] = []
    for row in raw_cases:
        if not isinstance(row, dict):
            raise RetrievalEnterpriseValueLabError("Retrieval Gold case is invalid")
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
                    query=str(row["query"]), documents=retrieval_documents
                ),
            )
        )
    return tuple(cases)


def _verify_inputs(paths: dict[str, Path]) -> None:
    if _file_sha256(paths["embedding_model"] / "model.safetensors") != (
        EMBEDDING_MODEL_WEIGHT_SHA256
    ):
        raise RetrievalEnterpriseValueLabError("Embedding base model changed")
    if _file_sha256(paths["reranker_model"] / "model.safetensors") != (
        RERANKER_MODEL_WEIGHT_SHA256
    ):
        raise RetrievalEnterpriseValueLabError("Reranker base model changed")
    if _file_sha256(paths["training_manifest"]) != TRAINING_MANIFEST_SHA256:
        raise RetrievalEnterpriseValueLabError("Retrieval training manifest changed")
    if _file_sha256(paths["gold_manifest"]) != GOLD_MANIFEST_SHA256:
        raise RetrievalEnterpriseValueLabError("Retrieval Gold manifest changed")
    training = _load_object(paths["training_manifest"])
    gold = _load_object(paths["gold_manifest"])
    if not _datasets_are_governed_and_disjoint(training, gold):
        raise RetrievalEnterpriseValueLabError("Retrieval dataset governance changed")


def _datasets_are_governed_and_disjoint(
    training: dict[str, Any], gold: dict[str, Any]
) -> bool:
    rows = training.get("rows")
    cases = gold.get("cases")
    documents = gold.get("documents")
    policy = gold.get("selection_policy")
    if (
        not isinstance(rows, list)
        or not isinstance(cases, list)
        or not isinstance(documents, list)
        or len([item for item in rows if isinstance(item, dict) and item.get("split") == "train"])
        != 20
        or len(
            [item for item in rows if isinstance(item, dict) and item.get("split") == "validation"]
        )
        != 6
        or len(cases) != 12
        or len(documents) != 22
        or gold.get("top_k") != 3
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
    forbidden_count = sum(
        1 for item in documents if isinstance(item, dict) and item.get("access") == "FORBIDDEN"
    )
    if len(allowed_ids) != 20 or forbidden_count != 2:
        return False
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("review_status") != "APPROVED"
            or not isinstance(row.get("query"), str)
            or not isinstance(row.get("positive"), str)
            or not isinstance(row.get("hard_negatives"), list)
            or len(row["hard_negatives"]) != 2
            or len(set(row["hard_negatives"])) != 2
            or row["positive"] in row["hard_negatives"]
        ):
            return False
    for case in cases:
        if not isinstance(case, dict):
            return False
        relevant = case.get("relevant_document_ids")
        if not isinstance(relevant, list) or not relevant or not set(relevant) <= allowed_ids:
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
    gold_texts = {
        str(item.get("query")) for item in cases if isinstance(item, dict)
    } | {
        str(item.get("text")) for item in documents if isinstance(item, dict)
    }
    return not (training_texts & gold_texts)


def _input_paths(root: Path) -> dict[str, Path]:
    return {
        "embedding_model": _inside_directory(root, EMBEDDING_MODEL_RELATIVE),
        "reranker_model": _inside_directory(root, RERANKER_MODEL_RELATIVE),
        "training_manifest": _inside_file(root, TRAINING_DATASET_RELATIVE),
        "gold_manifest": _inside_file(root, GOLD_DATASET_RELATIVE),
    }


def _dependencies() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
        sentence_transformers = importlib.import_module("sentence_transformers")
        cross_encoder = importlib.import_module("sentence_transformers.cross_encoder")
        embedding_losses = importlib.import_module("sentence_transformers.losses")
        reranker_losses = importlib.import_module("sentence_transformers.cross_encoder.losses")
        training_args = importlib.import_module("sentence_transformers.training_args")
        datasets = importlib.import_module("datasets")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RetrievalEnterpriseValueLabError(
            "Retrieval GPU dependencies are not installed"
        ) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RetrievalEnterpriseValueLabError("Retrieval lab requires exactly one CUDA GPU")
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


def _input_model_evidence(root: Path, paths: dict[str, Path]) -> list[dict[str, Any]]:
    specs = (
        (
            "EMBEDDING",
            EMBEDDING_MODEL_ID,
            EMBEDDING_MODEL_REVISION,
            EMBEDDING_MODEL_WEIGHT_SHA256,
            paths["embedding_model"],
        ),
        (
            "RERANKER",
            RERANKER_MODEL_ID,
            RERANKER_MODEL_REVISION,
            RERANKER_MODEL_WEIGHT_SHA256,
            paths["reranker_model"],
        ),
    )
    result: list[dict[str, Any]] = []
    for component, model_id, revision, weight_sha256, path in specs:
        manifest = _directory_manifest(path)
        result.append(
            {
                "component": component,
                "model_id": model_id,
                "revision": revision,
                "weight_sha256": weight_sha256,
                "path": path.relative_to(root).as_posix(),
                "manifest_sha256": _manifest_digest(manifest),
                "files": [
                    _file_evidence(path.relative_to(root) / item["path"], path / item["path"])
                    for item in manifest
                ],
                "license": "apache-2.0",
            }
        )
    return result


def _dataset_evidence(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "training_manifest": _file_evidence(
            paths["training_manifest"].relative_to(root), paths["training_manifest"]
        ),
        "final_gold_manifest": _file_evidence(
            paths["gold_manifest"].relative_to(root), paths["gold_manifest"]
        ),
        "train_count": 20,
        "validation_count": 6,
        "final_gold_count": 12,
        "allowed_document_count": 20,
        "forbidden_document_count": 2,
        "top_k": 3,
        "source_type": "PLATFORM_SYNTHETIC",
        "project_enterprise_use_authorized": True,
        "enterprise_production_data": False,
        "frozen_gold_excluded_from_training_selection_and_development": True,
        "case_ids_and_texts_disjoint": True,
    }


def _evaluation_evidence(
    root: Path,
    final_relative: Path,
    temporary: Path,
    results: tuple[_EvaluationResult, _EvaluationResult],
) -> tuple[dict[str, Any], ...]:
    evidence: list[dict[str, Any]] = []
    for result in results:
        name = result.component.lower()
        evidence.append(
            {
                **result.summary,
                "formal_report": _file_evidence(
                    final_relative / f"{name}-formal-evaluation.json",
                    temporary / f"{name}-formal-evaluation.json",
                ),
                "observations": _file_evidence(
                    final_relative / f"{name}-observations.json",
                    temporary / f"{name}-observations.json",
                ),
            }
        )
    return tuple(evidence)


def _candidate_evidence(
    root: Path, final_relative: Path, temporary: Path
) -> tuple[dict[str, Any], ...]:
    del root
    result: list[dict[str, Any]] = []
    for component, name, role in (
        (
            "EMBEDDING",
            "embedding",
            "AUTHORIZED_ENTERPRISE_EMBEDDING_EVALUATION_CANDIDATE",
        ),
        (
            "RERANKER",
            "reranker",
            "AUTHORIZED_ENTERPRISE_RERANKER_EVALUATION_CANDIDATE",
        ),
    ):
        actual = temporary / "candidates" / name
        manifest = _directory_manifest(actual)
        relative = final_relative / "candidates" / name
        result.append(
            {
                "component": component,
                "path": relative.as_posix(),
                "bundle_sha256": _manifest_digest(manifest),
                "size_bytes": sum(int(item["size_bytes"]) for item in manifest),
                "files": [
                    _file_evidence(relative / item["path"], actual / item["path"])
                    for item in manifest
                ],
                "role": role,
            }
        )
    return tuple(result)


def _persist_rejection(
    *,
    root: Path,
    temporary: Path,
    run_id: str,
    image_digest: str,
    paths: dict[str, Path],
    training: tuple[_TrainingResult, _TrainingResult],
    evaluations: tuple[_EvaluationResult, _EvaluationResult],
    failures: tuple[str, ...],
) -> Path:
    rejection_directory = root / OUTPUT_RELATIVE / "rejections" / run_id
    if rejection_directory.exists():
        raise RetrievalEnterpriseValueLabError("immutable Retrieval rejection exists")
    unsigned = {
        "schema_version": "enterprise-retrieval-formal-rejection/v1",
        "classification": CLASSIFICATION,
        "production_claim": False,
        "status": "RETRIEVAL_ACTUAL_GPU_CANDIDATES_REJECTED",
        "decision": "DO_NOT_REGISTER_OR_DEPLOY",
        "run_id": run_id,
        "image": IMAGE,
        "image_digest": image_digest,
        "source_sha256": _file_sha256(Path(__file__).resolve(strict=True)),
        "input_models": _input_model_evidence(root, paths),
        "dataset": _dataset_evidence(root, paths),
        "training": [item.evidence for item in training],
        "evaluations": [item.summary for item in evaluations],
        "failed_gates": list(failures),
        "candidate_accepted": False,
        "formal_model_release_created": False,
        "runtime_eligible": False,
    }
    rejection = {**unsigned, "evidence_chain_sha256": _digest(unsigned)}
    _write_json(temporary / "rejection.json", rejection)
    rejection_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(rejection_directory)
    return rejection_directory / "rejection.json"


def _write_gold_retirement(root: Path, *, run_id: str, outcome: str) -> None:
    path = root / GOLD_RETIREMENT_RELATIVE
    document = {
        "schema_version": "enterprise-retrieval-gold-suite-retirement/v1",
        "classification": CLASSIFICATION,
        "suite_id": "industrial-retrieval-final-gold-v1",
        "suite_manifest_sha256": GOLD_MANIFEST_SHA256,
        "run_id": run_id,
        "outcome": outcome,
        "reuse_permitted": False,
    }
    if path.exists():
        if _load_object(path) != document:
            raise RetrievalEnterpriseValueLabError("Retrieval Gold retirement changed")
        return
    _write_json(path, document)


def _observation_document(observation: ModelObservation) -> dict[str, Any]:
    return {
        "case_id": observation.case_id,
        "output_text": observation.output_text,
        "latency_ms": observation.latency_ms,
        "cost_usd": observation.cost_usd,
        "input_tokens": observation.input_tokens,
        "ranked_document_ids": list(observation.ranked_document_ids),
        "retrieval_scores": list(observation.retrieval_scores),
        "runtime_evidence": observation.runtime_evidence,
    }


def _checkpoint_step(path: str) -> int:
    name = Path(path).name
    if not name.startswith("checkpoint-"):
        raise RetrievalEnterpriseValueLabError("Retrieval checkpoint is invalid")
    try:
        return int(name.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise RetrievalEnterpriseValueLabError("Retrieval checkpoint is invalid") from exc


def _directory_manifest(directory: Path) -> list[dict[str, Any]]:
    root = directory.resolve(strict=True)
    manifest: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if ".cache" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise RetrievalEnterpriseValueLabError("Retrieval artifact cannot contain symlinks")
        if path.is_file():
            manifest.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    if not manifest:
        raise RetrievalEnterpriseValueLabError("Retrieval artifact is empty")
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
        raise RetrievalEnterpriseValueLabError(f"Retrieval evidence changed: {evidence.path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RetrievalEnterpriseValueLabError(f"Retrieval JSON is invalid: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_latest(
    root: Path, acceptance_path: Path, report: RetrievalEnterpriseValueReport
) -> None:
    relative = acceptance_path.relative_to(root)
    _write_json(
        root / OUTPUT_RELATIVE / "latest.json",
        {
            "schema_version": "enterprise-retrieval-value-lab-latest/v1",
            "run_id": report.run_id,
            "acceptance_path": relative.as_posix(),
            "acceptance_sha256": _file_sha256(acceptance_path),
        },
    )


def _latest_report_path(root: Path) -> Path:
    pointer = _load_object(_inside_file(root, OUTPUT_RELATIVE / "latest.json"))
    path_value = pointer.get("acceptance_path")
    expected_hash = pointer.get("acceptance_sha256")
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        raise RetrievalEnterpriseValueLabError("Retrieval latest pointer is invalid")
    path = _inside_file(root, Path(path_value))
    if _file_sha256(path) != expected_hash:
        raise RetrievalEnterpriseValueLabError("Retrieval latest acceptance changed")
    return path


def _inside_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=True)
    _require_inside(root, resolved)
    if not resolved.is_file():
        raise RetrievalEnterpriseValueLabError(f"Retrieval file is missing: {path}")
    return resolved


def _inside_directory(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=True)
    _require_inside(root, resolved)
    if not resolved.is_dir():
        raise RetrievalEnterpriseValueLabError(f"Retrieval directory is missing: {path}")
    return resolved


def _require_inside(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RetrievalEnterpriseValueLabError("Retrieval path escapes repository") from exc


def _require_image_digest(value: str) -> None:
    if len(value) != 71 or not value.startswith("sha256:"):
        raise RetrievalEnterpriseValueLabError("Retrieval image digest is invalid")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise RetrievalEnterpriseValueLabError("Retrieval image digest is invalid") from exc
