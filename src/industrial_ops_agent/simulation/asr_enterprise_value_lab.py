"""Low-memory actual-GPU ASR LoRA enterprise-value experiment."""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import re
import time
import unicodedata
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.evaluation.dataset import (
    AsrEvaluationContract,
    EvaluationCase,
)
from industrial_ops_agent.evaluation.metrics import (
    ModelObservation,
    PairedMetricReport,
    score_asr_paired_observations,
)
from industrial_ops_agent.experiments.service import ASR_HARD_GATES

SCHEMA_VERSION: Literal["enterprise-asr-value-lab/v1"] = "enterprise-asr-value-lab/v1"
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
STATUS: Literal["ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"] = (
    "ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
)
DECISION: Literal["ASR_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"] = (
    "ASR_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"
)
OUTPUT_RELATIVE = Path("artifacts/m7-asr-enterprise-value-lab")
MODEL_INPUT_RELATIVE = OUTPUT_RELATIVE / "inputs" / "whisper-tiny-c4f62d5f"
DATASET_INPUT_RELATIVE = (
    OUTPUT_RELATIVE
    / "inputs"
    / "librispeech-asr-dummy"
    / "clean"
    / "librispeech_asr_dummy-validation.parquet"
)
IMAGE = "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
CONTAINER_NAME = "ioap-asr-enterprise-value-lab"
MODEL_ID = "openai/whisper-tiny"
MODEL_REVISION = "c4f62d5fce5d73978a1dbfcac01926312acd1a51"
MODEL_LICENSE = "Apache-2.0"
MODEL_WEIGHT_SHA256 = "9607f98a2b22d9e229ae43c52ecea79dcede9e0c5cfae67e8da6eda86d8aac1d"
DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"
DATASET_REVISION = "ac8a12cae595fb1266b476d8177b782184fe7311"
DATASET_FILE_SHA256 = "ca1b00b86bcdb7f150b96145b82295820c2ae5bc1410c9165cd05b984f7dc563"

_TRAIN_CHAPTER = 128104
_VALIDATION_CHAPTER = 135031
_GOLD_CHAPTER = 141231
_TRAINING_CONFIG: dict[str, Any] = {
    "method": "ASR_LORA",
    "max_steps": 48,
    "evaluation_interval": 12,
    "train_rows": 15,
    "validation_rows": 6,
    "gold_rows": 12,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 5e-4,
    "precision": "bfloat16",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_scope": "whisper_encoder_self_attention_qv",
    "sampling_rate": 16_000,
    "noise_profile": "deterministic_industrial_snr_3_db_v1",
    "noise_snr_db": 3.0,
    "noise_training_ratio": "2_of_3_steps",
    "max_generation_length": 64,
    "wer_max": 0.55,
    "cer_max": 0.40,
    "noise_wer_max": 0.80,
    "minimum_wer_improvement": 0.01,
    "minimum_noise_wer_improvement": 0.05,
    "seed": 20260824,
    "data_seed": 20260825,
    "cpu_threads": 2,
    "dataloader_workers": 0,
    "container_memory_limit_bytes": 4 * 1024**3,
}


class AsrEnterpriseValueLabError(RuntimeError):
    """The governed ASR lab could not produce or verify its evidence."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AsrDatasetEvidence(_ClosedModel):
    source_file: FileEvidence
    dataset_id: Literal["hf-internal-testing/librispeech_asr_dummy"]
    revision: Literal["ac8a12cae595fb1266b476d8177b782184fe7311"]
    source_audio_license: Literal["LIBRISPEECH_PUBLIC_DOMAIN"]
    repository_code_license: Literal["Apache-2.0"]
    source_row_count: Literal[73]
    train_count: Literal[15]
    validation_count: Literal[6]
    gold_count: Literal[12]
    train_chapter_id: Literal[128104]
    validation_chapter_id: Literal[135031]
    gold_chapter_id: Literal[141231]
    split_sha256s: dict[str, str]
    manifest: FileEvidence
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project_enterprise_use_authorized: Literal[True] = True
    frozen_gold_excluded_from_training_and_selection: Literal[True] = True
    source_ids_disjoint_across_splits: Literal[True] = True
    deterministic_noise_transform: Literal[True] = True


class AsrModelEvidence(_ClosedModel):
    model_id: Literal["openai/whisper-tiny"]
    revision: Literal["c4f62d5fce5d73978a1dbfcac01926312acd1a51"]
    license: Literal["Apache-2.0"]
    snapshot_path: str = Field(min_length=1)
    image: Literal[
        "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
    ]
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    weight_sha256: Literal[
        "9607f98a2b22d9e229ae43c52ecea79dcede9e0c5cfae67e8da6eda86d8aac1d"
    ]
    snapshot_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_files: tuple[FileEvidence, ...] = Field(min_length=10)


class AsrAdapterEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    files: tuple[FileEvidence, ...] = Field(min_length=2)


class AsrRuntimeEvidence(_ClosedModel):
    actual_gpu_execution: Literal[True] = True
    model_training_simulated: Literal[False] = False
    method: Literal["ASR_LORA"]
    trainer_family: Literal["manual-transformers-peft-whisper"]
    gpu_name: str = Field(min_length=1)
    gpu_total_memory_bytes: int = Field(gt=0)
    peak_gpu_memory_allocated_bytes: int = Field(gt=0)
    peak_gpu_memory_reserved_bytes: int = Field(gt=0)
    gpu_memory_allocated_after_cleanup_bytes: int = Field(ge=0)
    optimizer_steps: Literal[48]
    selected_validation_step: int = Field(ge=12, le=48)
    train_loss_first: float
    train_loss_last: float
    selected_validation_objective: float = Field(ge=0.0)
    total_parameters: int = Field(gt=0)
    trainable_parameters: int = Field(gt=0)
    torch_version: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    peft_version: str = Field(min_length=1)
    pyarrow_version: str = Field(min_length=1)
    non_root_container_user: Literal[True] = True
    network_disabled: Literal[True] = True
    cpu_threads: Literal[2] = 2
    dataloader_workers: Literal[0] = 0
    container_memory_limit_bytes: Literal[4294967296] = 4294967296


class AsrEvaluationEvidence(_ClosedModel):
    formal_report: FileEvidence
    observations: FileEvidence
    frozen_gold_case_count: Literal[12]
    clean_case_count: int = Field(ge=1)
    noisy_case_count: int = Field(ge=1)
    blocking_case_count: int = Field(ge=1)
    baseline_wer: float = Field(ge=0.0)
    candidate_wer: float = Field(ge=0.0)
    wer_improvement: float = Field(gt=0.0)
    baseline_cer: float = Field(ge=0.0)
    candidate_cer: float = Field(ge=0.0)
    cer_improvement: float
    baseline_noise_wer: float = Field(ge=0.0)
    candidate_noise_wer: float = Field(ge=0.0)
    noise_wer_improvement: float = Field(gt=0.0)
    validation_baseline_wer: float = Field(ge=0.0)
    validation_candidate_wer: float = Field(ge=0.0)
    validation_history: tuple[dict[str, Any], ...] = Field(min_length=4, max_length=4)
    wer_max: float = Field(ge=0.55, le=0.55)
    cer_max: float = Field(ge=0.4, le=0.4)
    noise_wer_max: float = Field(ge=0.8, le=0.8)


class AsrHardGates(_ClosedModel):
    data_governance: Literal[True] = True
    actual_gpu_training: Literal[True] = True
    immutable_model_data_and_image: Literal[True] = True
    frozen_gold_independence: Literal[True] = True
    asr_media_integrity: Literal[True] = True
    asr_transcript_quality: Literal[True] = True
    asr_noise_robustness: Literal[True] = True
    no_blocking_regressions: Literal[True] = True
    overall_wer_improved: Literal[True] = True
    noise_wer_improved: Literal[True] = True
    resource_profile_respected: Literal[True] = True


class AsrEnterpriseValueReport(_ClosedModel):
    schema_version: Literal["enterprise-asr-value-lab/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"] = STATUS
    decision: Literal["ASR_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"] = DECISION
    run_id: str = Field(pattern=r"^asr-[0-9a-f]{20}$")
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset: AsrDatasetEvidence
    base_model: AsrModelEvidence
    adapter: AsrAdapterEvidence
    runtime: AsrRuntimeEvidence
    evaluation: AsrEvaluationEvidence
    hard_gates: AsrHardGates
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _AudioRow:
    source_id: str
    transcript: str
    audio: Any
    source_media: bytes
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class _ValidationResult:
    average_wer: float
    clean_wer: float
    noisy_wer: float
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _TrainingResult:
    runtime: dict[str, Any]
    dataset_document: dict[str, Any]
    observations_document: dict[str, Any]
    formal_report_document: dict[str, Any]
    evaluation: dict[str, Any]


def planned_run_id(
    repo_root: Path,
    image_digest: str,
    *,
    model_path: Path | None = None,
    dataset_path: Path | None = None,
) -> str:
    root = repo_root.resolve(strict=True)
    _require_image_digest(image_digest)
    resolved_model = _inside_directory(root, model_path or MODEL_INPUT_RELATIVE)
    resolved_dataset = _inside_file(root, dataset_path or DATASET_INPUT_RELATIVE)
    _verify_inputs(resolved_model, resolved_dataset)
    source_path = Path(__file__).resolve(strict=True)
    _require_inside(root, source_path)
    identity = {
        "training_config": _TRAINING_CONFIG,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": _manifest_digest(_directory_manifest(resolved_model)),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_file_sha256": _file_sha256(resolved_dataset),
        "image": IMAGE,
        "image_digest": image_digest,
        "source_sha256": _file_sha256(source_path),
    }
    return f"asr-{_digest(identity)[:20]}"


def execute_worker(
    repo_root: Path,
    *,
    image_digest: str,
    model_path: Path | None = None,
    dataset_path: Path | None = None,
) -> Path:
    """Run one actual ASR LoRA job inside the fixed, network-disabled GPU image."""

    root = repo_root.resolve(strict=True)
    _require_image_digest(image_digest)
    resolved_model = _inside_directory(root, model_path or MODEL_INPUT_RELATIVE)
    resolved_dataset = _inside_file(root, dataset_path or DATASET_INPUT_RELATIVE)
    _verify_inputs(resolved_model, resolved_dataset)
    run_id = planned_run_id(
        root,
        image_digest,
        model_path=resolved_model,
        dataset_path=resolved_dataset,
    )
    output_root = root / OUTPUT_RELATIVE
    final_directory = output_root / "runs" / run_id
    acceptance_path = final_directory / "acceptance.json"
    if acceptance_path.is_file():
        verify_asr_enterprise_value(root, acceptance_path)
        return acceptance_path
    if final_directory.exists():
        raise AsrEnterpriseValueLabError("immutable ASR run directory already exists")

    temporary = output_root / f".{run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    training = _train_actual_asr(
        model_path=resolved_model,
        dataset_path=resolved_dataset,
        output_directory=temporary / "adapter",
    )
    dataset_manifest_path = temporary / "dataset-manifest.json"
    observations_path = temporary / "observations.json"
    formal_report_path = temporary / "formal-evaluation.json"
    _write_json(dataset_manifest_path, training.dataset_document)
    _write_json(observations_path, training.observations_document)
    _write_json(formal_report_path, training.formal_report_document)

    source_path = Path(__file__).resolve(strict=True)
    model_files = _directory_manifest(resolved_model)
    adapter_files = _directory_manifest(temporary / "adapter")
    final_relative = final_directory.relative_to(root)
    dataset_relative = resolved_dataset.relative_to(root)
    model_relative = resolved_model.relative_to(root)
    evaluation = training.evaluation
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
        "dataset": {
            "source_file": _file_evidence(dataset_relative, resolved_dataset),
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "source_audio_license": "LIBRISPEECH_PUBLIC_DOMAIN",
            "repository_code_license": "Apache-2.0",
            "source_row_count": 73,
            "train_count": 15,
            "validation_count": 6,
            "gold_count": 12,
            "train_chapter_id": _TRAIN_CHAPTER,
            "validation_chapter_id": _VALIDATION_CHAPTER,
            "gold_chapter_id": _GOLD_CHAPTER,
            "split_sha256s": training.dataset_document["split_sha256s"],
            "manifest": _file_evidence(
                final_relative / "dataset-manifest.json", dataset_manifest_path
            ),
            "manifest_sha256": _digest(training.dataset_document),
            "project_enterprise_use_authorized": True,
            "frozen_gold_excluded_from_training_and_selection": True,
            "source_ids_disjoint_across_splits": True,
            "deterministic_noise_transform": True,
        },
        "base_model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "snapshot_path": model_relative.as_posix(),
            "image": IMAGE,
            "image_digest": image_digest,
            "weight_sha256": MODEL_WEIGHT_SHA256,
            "snapshot_manifest_sha256": _manifest_digest(model_files),
            "snapshot_files": model_files,
        },
        "adapter": {
            "path": (final_relative / "adapter").as_posix(),
            "bundle_sha256": _manifest_digest(adapter_files),
            "size_bytes": sum(item["size_bytes"] for item in adapter_files),
            "files": adapter_files,
        },
        "runtime": training.runtime,
        "evaluation": {
            "formal_report": _file_evidence(
                final_relative / "formal-evaluation.json", formal_report_path
            ),
            "observations": _file_evidence(
                final_relative / "observations.json", observations_path
            ),
            **evaluation,
        },
        "hard_gates": AsrHardGates().model_dump(mode="json"),
        "formal_model_release_created": False,
        "runtime_eligible": False,
    }
    draft = AsrEnterpriseValueReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    normalized = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    report = draft.model_copy(update={"evidence_chain_sha256": _digest(normalized)})
    _write_json(temporary / "acceptance.json", report.model_dump(mode="json"))
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(final_directory)
    verified = verify_asr_enterprise_value(root, acceptance_path)
    _write_latest(root, acceptance_path, verified)
    return acceptance_path


def verify_asr_enterprise_value(
    repo_root: Path,
    acceptance_path: Path | None = None,
) -> AsrEnterpriseValueReport:
    """Verify the immutable ASR receipt without importing PyTorch or PyArrow."""

    root = repo_root.resolve(strict=True)
    report_path = (
        _latest_report_path(root)
        if acceptance_path is None
        else _inside_file(root, acceptance_path)
    )
    report = AsrEnterpriseValueReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise AsrEnterpriseValueLabError("ASR evidence chain changed")
    source_path = _inside_file(root, Path(report.source_path))
    if _file_sha256(source_path) != report.source_sha256:
        raise AsrEnterpriseValueLabError("ASR trainer source changed")
    dataset_path = _inside_file(root, Path(report.dataset.source_file.path))
    model_path = _inside_directory(root, Path(report.base_model.snapshot_path))
    if planned_run_id(
        root,
        report.base_model.image_digest,
        model_path=model_path,
        dataset_path=dataset_path,
    ) != report.run_id:
        raise AsrEnterpriseValueLabError("ASR run identity changed")
    if _file_sha256(dataset_path) != report.dataset.source_file.sha256:
        raise AsrEnterpriseValueLabError("ASR source dataset changed")
    model_files = _directory_manifest(model_path)
    if (
        model_files
        != [item.model_dump(mode="json") for item in report.base_model.snapshot_files]
        or _manifest_digest(model_files) != report.base_model.snapshot_manifest_sha256
    ):
        raise AsrEnterpriseValueLabError("ASR base model snapshot changed")
    manifest_path = _verify_file_evidence(root, report.dataset.manifest)
    manifest = _load_object(manifest_path)
    if (
        _digest(manifest) != report.dataset.manifest_sha256
        or manifest.get("split_sha256s") != report.dataset.split_sha256s
        or not _dataset_manifest_is_disjoint(manifest)
    ):
        raise AsrEnterpriseValueLabError("ASR dataset manifest changed")
    _verify_file_evidence(root, report.evaluation.formal_report)
    _verify_file_evidence(root, report.evaluation.observations)
    adapter_path = _inside_directory(root, Path(report.adapter.path))
    adapter_files = _directory_manifest(adapter_path)
    if (
        adapter_files != [item.model_dump(mode="json") for item in report.adapter.files]
        or _manifest_digest(adapter_files) != report.adapter.bundle_sha256
        or sum(item["size_bytes"] for item in adapter_files) != report.adapter.size_bytes
    ):
        raise AsrEnterpriseValueLabError("ASR adapter bundle changed")
    return report


def _train_actual_asr(
    *,
    model_path: Path,
    dataset_path: Path,
    output_directory: Path,
) -> _TrainingResult:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    dependencies = _asr_dependencies()
    numpy = dependencies["numpy"]
    peft = dependencies["peft"]
    soundfile = dependencies["soundfile"]
    torch = dependencies["torch"]
    transformers = dependencies["transformers"]
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AsrEnterpriseValueLabError("exactly one CUDA GPU is required")
    transformers.set_seed(_TRAINING_CONFIG["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()

    rows_by_chapter, source_row_count = _load_audio_rows(dataset_path, dependencies)
    train_rows = rows_by_chapter[_TRAIN_CHAPTER][:_TRAINING_CONFIG["train_rows"]]
    validation_rows = rows_by_chapter[_VALIDATION_CHAPTER][
        : _TRAINING_CONFIG["validation_rows"]
    ]
    gold_rows = rows_by_chapter[_GOLD_CHAPTER][:_TRAINING_CONFIG["gold_rows"]]
    _verify_split_rows(train_rows, validation_rows, gold_rows, source_row_count)

    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        language="en",
        task="transcribe",
        trust_remote_code=False,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise AsrEnterpriseValueLabError("Whisper processor has no tokenizer")
    base_model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = True
    target_modules = [
        f"model.encoder.layers.{layer}.self_attn.{projection}"
        for layer in range(4)
        for projection in ("q_proj", "v_proj")
    ]
    model = peft.get_peft_model(
        base_model,
        peft.LoraConfig(
            r=_TRAINING_CONFIG["lora_rank"],
            lora_alpha=_TRAINING_CONFIG["lora_alpha"],
            lora_dropout=_TRAINING_CONFIG["lora_dropout"],
            bias="none",
            target_modules=target_modules,
        ),
    )
    baseline_validation = _evaluate_validation(
        processor=processor,
        model=model,
        rows=validation_rows,
        dependencies=dependencies,
        adapter_disabled=True,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=_TRAINING_CONFIG["learning_rate"],
        weight_decay=0.0,
    )
    losses: list[float] = []
    validation_history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_step = 0
    best_objective = math.inf
    model.config.use_cache = False
    for step in range(1, _TRAINING_CONFIG["max_steps"] + 1):
        model.train()
        features, attention_mask, labels = _training_inputs(
            processor=processor,
            row=train_rows[(step - 1) % len(train_rows)],
            step=step,
            dependencies=dependencies,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_features=features,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = output.loss
        if not torch.isfinite(loss):
            raise AsrEnterpriseValueLabError("ASR training loss is not finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            1.0,
        )
        optimizer.step()
        losses.append(float(loss.detach().float().item()))
        if step % _TRAINING_CONFIG["evaluation_interval"] != 0:
            continue
        model.config.use_cache = True
        candidate_validation = _evaluate_validation(
            processor=processor,
            model=model,
            rows=validation_rows,
            dependencies=dependencies,
            adapter_disabled=False,
        )
        model.config.use_cache = False
        objective, blocking_regressions = _validation_objective(
            baseline_validation,
            candidate_validation,
        )
        validation_history.append(
            {
                "step": step,
                "training_loss": losses[-1],
                "objective": objective,
                "candidate_average_wer": candidate_validation.average_wer,
                "candidate_clean_wer": candidate_validation.clean_wer,
                "candidate_noisy_wer": candidate_validation.noisy_wer,
                "blocking_regressions": blocking_regressions,
            }
        )
        if objective < best_objective:
            best_objective = objective
            best_step = step
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in peft.get_peft_model_state_dict(model).items()
            }
    if best_state is None or best_step == 0:
        raise AsrEnterpriseValueLabError("ASR validation did not select a checkpoint")
    peft.set_peft_model_state_dict(model, best_state)
    model.config.use_cache = True
    selected_validation = _evaluate_validation(
        processor=processor,
        model=model,
        rows=validation_rows,
        dependencies=dependencies,
        adapter_disabled=False,
    )
    if (
        selected_validation.average_wer >= baseline_validation.average_wer
        or selected_validation.noisy_wer >= baseline_validation.noisy_wer
    ):
        raise AsrEnterpriseValueLabError("ASR validation candidate did not improve")

    gold_cases, gold_manifest = _build_gold_cases(
        gold_rows,
        numpy=numpy,
        soundfile=soundfile,
    )
    baseline_observations = _evaluate_gold(
        processor=processor,
        model=model,
        cases=gold_cases,
        dependencies=dependencies,
        adapter_disabled=True,
    )
    candidate_observations = _evaluate_gold(
        processor=processor,
        model=model,
        cases=gold_cases,
        dependencies=dependencies,
        adapter_disabled=False,
    )
    formal_report = score_asr_paired_observations(
        gold_cases,
        candidate_observations,
        baseline_observations,
        wer_max=_TRAINING_CONFIG["wer_max"],
        cer_max=_TRAINING_CONFIG["cer_max"],
        noise_wer_max=_TRAINING_CONFIG["noise_wer_max"],
    )
    evaluation = _formal_evaluation_summary(
        formal_report,
        gold_cases=gold_cases,
        baseline_validation=baseline_validation,
        selected_validation=selected_validation,
        validation_history=validation_history,
    )
    _require_formal_gates(formal_report, evaluation)

    output_directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(output_directory), safe_serialization=True)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    device = torch.cuda.get_device_properties(0)
    runtime: dict[str, Any] = {
        "actual_gpu_execution": True,
        "model_training_simulated": False,
        "method": "ASR_LORA",
        "trainer_family": "manual-transformers-peft-whisper",
        "gpu_name": device.name,
        "gpu_total_memory_bytes": int(device.total_memory),
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "gpu_memory_allocated_after_cleanup_bytes": 0,
        "optimizer_steps": _TRAINING_CONFIG["max_steps"],
        "selected_validation_step": best_step,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "selected_validation_objective": best_objective,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "torch_version": version("torch"),
        "transformers_version": version("transformers"),
        "peft_version": version("peft"),
        "pyarrow_version": version("pyarrow"),
        "non_root_container_user": os.geteuid() != 0,
        "network_disabled": True,
        "cpu_threads": 2,
        "dataloader_workers": 0,
        "container_memory_limit_bytes": _TRAINING_CONFIG[
            "container_memory_limit_bytes"
        ],
    }
    dataset_document = _dataset_document(
        dataset_path=dataset_path,
        source_row_count=source_row_count,
        train_rows=train_rows,
        validation_rows=validation_rows,
        gold_rows=gold_rows,
        gold_manifest=gold_manifest,
    )
    observations_document = {
        "schema_version": "enterprise-asr-observations/v1",
        "classification": CLASSIFICATION,
        "baseline": [_observation_document(item) for item in baseline_observations],
        "candidate": [_observation_document(item) for item in candidate_observations],
    }
    formal_report_document = {
        "schema_version": "enterprise-asr-formal-evaluation/v1",
        "classification": CLASSIFICATION,
        "report": asdict(formal_report),
    }

    del features
    del attention_mask
    del labels
    del output
    del loss
    del optimizer
    del best_state
    del model
    del base_model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    runtime["gpu_memory_allocated_after_cleanup_bytes"] = int(
        torch.cuda.memory_allocated()
    )
    return _TrainingResult(
        runtime=runtime,
        dataset_document=dataset_document,
        observations_document=observations_document,
        formal_report_document=formal_report_document,
        evaluation=evaluation,
    )


def _asr_dependencies() -> dict[str, Any]:
    try:
        return {
            "numpy": importlib.import_module("numpy"),
            "peft": importlib.import_module("peft"),
            "pyarrow": importlib.import_module("pyarrow"),
            "parquet": importlib.import_module("pyarrow.parquet"),
            "soundfile": importlib.import_module("soundfile"),
            "torch": importlib.import_module("torch"),
            "transformers": importlib.import_module("transformers"),
        }
    except ImportError as exc:  # pragma: no cover - fixed GPU image only
        raise AsrEnterpriseValueLabError("ASR GPU dependencies are missing") from exc


def _load_audio_rows(
    dataset_path: Path,
    dependencies: dict[str, Any],
) -> tuple[dict[int, list[_AudioRow]], int]:
    table = dependencies["parquet"].read_table(
        dataset_path,
        columns=["audio", "text", "chapter_id", "id"],
    )
    rows_by_chapter: dict[int, list[_AudioRow]] = {
        _TRAIN_CHAPTER: [],
        _VALIDATION_CHAPTER: [],
        _GOLD_CHAPTER: [],
    }
    soundfile = dependencies["soundfile"]
    for item in table.to_pylist():
        chapter_id = int(item["chapter_id"])
        if chapter_id not in rows_by_chapter:
            raise AsrEnterpriseValueLabError("unexpected LibriSpeech chapter")
        audio_value = item.get("audio")
        if not isinstance(audio_value, dict) or not isinstance(audio_value.get("bytes"), bytes):
            raise AsrEnterpriseValueLabError("ASR source audio payload is invalid")
        source_media = audio_value["bytes"]
        try:
            audio, sample_rate = soundfile.read(
                BytesIO(source_media),
                dtype="float32",
                always_2d=False,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise AsrEnterpriseValueLabError("ASR source audio decode failed") from exc
        if int(sample_rate) != _TRAINING_CONFIG["sampling_rate"]:
            raise AsrEnterpriseValueLabError("ASR source sampling rate changed")
        if getattr(audio, "ndim", 1) == 2:
            audio = audio.mean(axis=1)
        source_id = str(item["id"])
        transcript = str(item["text"]).strip()
        if not source_id or not transcript or len(audio) == 0:
            raise AsrEnterpriseValueLabError("ASR source row is invalid")
        rows_by_chapter[chapter_id].append(
            _AudioRow(
                source_id=source_id,
                transcript=transcript,
                audio=audio,
                source_media=source_media,
                duration_seconds=len(audio) / float(_TRAINING_CONFIG["sampling_rate"]),
            )
        )
    for rows in rows_by_chapter.values():
        rows.sort(key=lambda row: row.source_id)
    expected_counts = {_TRAIN_CHAPTER: 15, _VALIDATION_CHAPTER: 25, _GOLD_CHAPTER: 33}
    observed_counts = {chapter: len(rows) for chapter, rows in rows_by_chapter.items()}
    if table.num_rows != 73 or observed_counts != expected_counts:
        raise AsrEnterpriseValueLabError("ASR source dataset shape changed")
    return rows_by_chapter, int(table.num_rows)


def _verify_split_rows(
    train_rows: list[_AudioRow],
    validation_rows: list[_AudioRow],
    gold_rows: list[_AudioRow],
    source_row_count: int,
) -> None:
    if (
        source_row_count != 73
        or len(train_rows) != 15
        or len(validation_rows) != 6
        or len(gold_rows) != 12
    ):
        raise AsrEnterpriseValueLabError("ASR governed split count changed")
    split_ids = [
        {row.source_id for row in train_rows},
        {row.source_id for row in validation_rows},
        {row.source_id for row in gold_rows},
    ]
    if (
        len(set.union(*split_ids)) != sum(len(values) for values in split_ids)
        or any(not values for values in split_ids)
    ):
        raise AsrEnterpriseValueLabError("ASR governed splits overlap")


def _training_inputs(
    *,
    processor: Any,
    row: _AudioRow,
    step: int,
    dependencies: dict[str, Any],
) -> tuple[Any, Any, Any]:
    torch = dependencies["torch"]
    audio = row.audio
    if step % 3 != 0:
        audio = _industrial_noise(
            audio,
            identity=f"train:{row.source_id}:epoch:{step // 15}",
            numpy=dependencies["numpy"],
        )
    inputs = processor.feature_extractor(
        audio,
        sampling_rate=_TRAINING_CONFIG["sampling_rate"],
        return_attention_mask=True,
        return_tensors="pt",
    )
    labels = processor.tokenizer(row.transcript, return_tensors="pt").input_ids
    bos_token_id = processor.tokenizer.bos_token_id
    if (
        bos_token_id is not None
        and labels.shape[1] > 0
        and int(labels[0, 0]) == bos_token_id
    ):
        labels = labels[:, 1:]
    return (
        inputs.input_features.to(device="cuda:0", dtype=torch.bfloat16),
        inputs.attention_mask.to(device="cuda:0"),
        labels.to(device="cuda:0", dtype=torch.long),
    )


def _industrial_noise(audio: Any, *, identity: str, numpy: Any) -> Any:
    seed = int.from_bytes(sha256(identity.encode()).digest()[:8], "big")
    generator = numpy.random.default_rng(seed)
    signal_rms = max(float(numpy.sqrt(numpy.mean(numpy.square(audio)))), 1e-6)
    white = generator.normal(0.0, 1.0, len(audio)).astype(numpy.float32)
    time_axis = numpy.arange(len(audio), dtype=numpy.float32) / float(
        _TRAINING_CONFIG["sampling_rate"]
    )
    machinery = (
        numpy.sin(2.0 * numpy.pi * 60.0 * time_axis)
        + 0.5 * numpy.sin(2.0 * numpy.pi * 180.0 * time_axis)
    ).astype(numpy.float32)
    disturbance = 0.75 * white + 0.25 * machinery
    disturbance /= max(float(numpy.sqrt(numpy.mean(numpy.square(disturbance)))), 1e-6)
    noise_amplitude = 10.0 ** (-float(_TRAINING_CONFIG["noise_snr_db"]) / 20.0)
    return numpy.clip(
        audio + signal_rms * noise_amplitude * disturbance,
        -1.0,
        1.0,
    ).astype(numpy.float32)


def _transcribe(
    *,
    processor: Any,
    model: Any,
    audio: Any,
    dependencies: dict[str, Any],
) -> tuple[str, float, int, int]:
    torch = dependencies["torch"]
    inputs = processor.feature_extractor(
        audio,
        sampling_rate=_TRAINING_CONFIG["sampling_rate"],
        return_attention_mask=True,
        return_tensors="pt",
    )
    features = inputs.input_features.to(device="cuda:0", dtype=torch.bfloat16)
    attention_mask = inputs.attention_mask.to(device="cuda:0")
    input_frames = int(features.shape[-1])
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_features=features,
            attention_mask=attention_mask,
            max_length=_TRAINING_CONFIG["max_generation_length"],
            do_sample=False,
            language="en",
            task="transcribe",
        )
    torch.cuda.synchronize()
    latency_ms = max((time.perf_counter() - started) * 1000.0, 1e-9)
    transcript = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return transcript, latency_ms, input_frames, int(generated.shape[-1])


def _evaluate_validation(
    *,
    processor: Any,
    model: Any,
    rows: list[_AudioRow],
    dependencies: dict[str, Any],
    adapter_disabled: bool,
) -> _ValidationResult:
    model.eval()
    observed: list[dict[str, Any]] = []
    context = model.disable_adapter() if adapter_disabled else nullcontext()
    with context:
        for row in rows:
            for condition, audio in (
                ("CLEAN", row.audio),
                (
                    "INDUSTRIAL_SNR_3_DB",
                    _industrial_noise(
                        row.audio,
                        identity=row.source_id,
                        numpy=dependencies["numpy"],
                    ),
                ),
            ):
                transcript, _, _, _ = _transcribe(
                    processor=processor,
                    model=model,
                    audio=audio,
                    dependencies=dependencies,
                )
                observed.append(
                    {
                        "source_id": row.source_id,
                        "word_count": len(_transcript_units(row.transcript)[0]),
                        "condition": condition,
                        "wer": _word_error_rate(row.transcript, transcript),
                    }
                )
    clean = [float(item["wer"]) for item in observed if item["condition"] == "CLEAN"]
    noisy = [
        float(item["wer"])
        for item in observed
        if item["condition"] != "CLEAN"
    ]
    return _ValidationResult(
        average_wer=sum(clean + noisy) / len(clean + noisy),
        clean_wer=sum(clean) / len(clean),
        noisy_wer=sum(noisy) / len(noisy),
        rows=tuple(observed),
    )


def _validation_objective(
    baseline: _ValidationResult,
    candidate: _ValidationResult,
) -> tuple[float, int]:
    baseline_by_key = {
        (str(row["source_id"]), str(row["condition"])): float(row["wer"])
        for row in baseline.rows
    }
    blocking_regressions = 0
    blocking_penalty = 0.0
    for row in candidate.rows:
        if row["condition"] == "CLEAN" or int(row["word_count"]) < 20:
            continue
        baseline_wer = baseline_by_key[(str(row["source_id"]), str(row["condition"]))]
        regression = max(float(row["wer"]) - baseline_wer, 0.0)
        if regression > 0.0:
            blocking_regressions += 1
            blocking_penalty += regression
    clean_penalty = max(candidate.clean_wer - baseline.clean_wer, 0.0)
    objective = candidate.average_wer + 4.0 * blocking_penalty + clean_penalty
    return objective, blocking_regressions


def _build_gold_cases(
    rows: list[_AudioRow],
    *,
    numpy: Any,
    soundfile: Any,
) -> tuple[tuple[EvaluationCase, ...], list[dict[str, Any]]]:
    cases: list[EvaluationCase] = []
    manifest: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        word_count = len(_transcript_units(row.transcript)[0])
        condition = _gold_noise_condition(index=index, word_count=word_count)
        risk = "HIGH" if word_count >= 20 else "LOW"
        if condition == "CLEAN":
            media_content = row.source_media
            media_mime_type = "audio/flac"
        else:
            noisy = _industrial_noise(
                row.audio,
                identity=f"gold:{row.source_id}",
                numpy=numpy,
            )
            target = BytesIO()
            soundfile.write(
                target,
                noisy,
                _TRAINING_CONFIG["sampling_rate"],
                format="WAV",
                subtype="PCM_16",
            )
            media_content = target.getvalue()
            media_mime_type = "audio/wav"
        case_id = f"asr-gold-{row.source_id}-{condition.lower()}"
        media_id = f"asr-media-{row.source_id}-{condition.lower()}"
        case = EvaluationCase(
            case_id=case_id,
            prompt=(
                {
                    "role": "user",
                    "content": "Transcribe the governed industrial field recording.",
                },
            ),
            expected_output={"transcript": row.transcript},
            slices={
                "component": "ASR",
                "split": "gold",
                "noise_condition": condition,
                "risk": risk,
                "length_band": "LONG" if word_count >= 20 else "SHORT",
            },
            required_gates=tuple(sorted(ASR_HARD_GATES)),
            allowed_citations=(),
            forbidden_substrings=(),
            expected_decision=None,
            risk=risk,
            contexts=(),
            asr=AsrEvaluationContract(
                transcript=row.transcript,
                language="en",
                noise_condition=condition,
            ),
            media_id=media_id,
            media_content=media_content,
            media_mime_type=media_mime_type,
        )
        cases.append(case)
        manifest.append(
            {
                "case_id": case_id,
                "source_id": row.source_id,
                "transcript_sha256": sha256(row.transcript.encode()).hexdigest(),
                "word_count": word_count,
                "noise_condition": condition,
                "risk": risk,
                "media_id": media_id,
                "media_mime_type": media_mime_type,
                "media_size_bytes": len(media_content),
                "media_sha256": sha256(media_content).hexdigest(),
                "duration_seconds": row.duration_seconds,
            }
        )
    return tuple(cases), manifest


def _gold_noise_condition(*, index: int, word_count: int) -> str:
    if word_count >= 20:
        return "MEDIUM"
    return "LOW" if index % 2 == 1 else "CLEAN"


def _evaluate_gold(
    *,
    processor: Any,
    model: Any,
    cases: tuple[EvaluationCase, ...],
    dependencies: dict[str, Any],
    adapter_disabled: bool,
) -> tuple[ModelObservation, ...]:
    model.eval()
    observations: list[ModelObservation] = []
    context = model.disable_adapter() if adapter_disabled else nullcontext()
    with context:
        for case in cases:
            if case.asr is None or case.media_content is None:
                raise AsrEnterpriseValueLabError("ASR Gold media binding is missing")
            try:
                audio, sample_rate = dependencies["soundfile"].read(
                    BytesIO(case.media_content),
                    dtype="float32",
                    always_2d=False,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise AsrEnterpriseValueLabError("ASR Gold audio decode failed") from exc
            if int(sample_rate) != _TRAINING_CONFIG["sampling_rate"]:
                raise AsrEnterpriseValueLabError("ASR Gold sampling rate changed")
            if getattr(audio, "ndim", 1) == 2:
                audio = audio.mean(axis=1)
            transcript, latency_ms, input_frames, output_tokens = _transcribe(
                processor=processor,
                model=model,
                audio=audio,
                dependencies=dependencies,
            )
            observations.append(
                ModelObservation(
                    case_id=case.case_id,
                    output_text=transcript,
                    latency_ms=latency_ms,
                    cost_usd=max(latency_ms / 3_600_000.0 * 0.35, 1e-12),
                    input_tokens=input_frames,
                    output_tokens=output_tokens,
                    capabilities=frozenset({"asr_media_integrity"}),
                    structured_output_valid=bool(transcript),
                    transcript_text=transcript,
                    runtime_evidence={
                        "media_id": case.media_id,
                        "media_mime_type": case.media_mime_type,
                        "media_size_bytes": len(case.media_content),
                        "media_sha256": sha256(case.media_content).hexdigest(),
                        "sampling_rate": _TRAINING_CONFIG["sampling_rate"],
                        "duration_seconds": len(audio)
                        / float(_TRAINING_CONFIG["sampling_rate"]),
                        "language": case.asr.language,
                        "noise_condition": case.asr.noise_condition,
                        "adapter_enabled": not adapter_disabled,
                    },
                )
            )
    return tuple(observations)


def _formal_evaluation_summary(
    report: PairedMetricReport,
    *,
    gold_cases: tuple[EvaluationCase, ...],
    baseline_validation: _ValidationResult,
    selected_validation: _ValidationResult,
    validation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    asr = report.aggregate_metrics.get("asr")
    if not isinstance(asr, dict):
        raise AsrEnterpriseValueLabError("ASR aggregate metrics are missing")
    wer = _metric_pair(asr, "wer")
    cer = _metric_pair(asr, "cer")
    noise_wer = _metric_pair(asr, "noise_wer")
    clean_count = sum(
        case.asr is not None and case.asr.noise_condition == "CLEAN"
        for case in gold_cases
    )
    noisy_count = len(gold_cases) - clean_count
    blocking_count = sum(
        case.risk == "HIGH"
        or (case.asr is not None and case.asr.noise_condition in {"MEDIUM", "HIGH"})
        for case in gold_cases
    )
    return {
        "frozen_gold_case_count": len(gold_cases),
        "clean_case_count": clean_count,
        "noisy_case_count": noisy_count,
        "blocking_case_count": blocking_count,
        "baseline_wer": wer["baseline"],
        "candidate_wer": wer["candidate"],
        "wer_improvement": wer["baseline"] - wer["candidate"],
        "baseline_cer": cer["baseline"],
        "candidate_cer": cer["candidate"],
        "cer_improvement": cer["baseline"] - cer["candidate"],
        "baseline_noise_wer": noise_wer["baseline"],
        "candidate_noise_wer": noise_wer["candidate"],
        "noise_wer_improvement": noise_wer["baseline"] - noise_wer["candidate"],
        "validation_baseline_wer": baseline_validation.average_wer,
        "validation_candidate_wer": selected_validation.average_wer,
        "validation_history": validation_history,
        "wer_max": _TRAINING_CONFIG["wer_max"],
        "cer_max": _TRAINING_CONFIG["cer_max"],
        "noise_wer_max": _TRAINING_CONFIG["noise_wer_max"],
    }


def _metric_pair(document: dict[str, Any], name: str) -> dict[str, float]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise AsrEnterpriseValueLabError(f"ASR {name} metric is missing")
    candidate = value.get("candidate")
    baseline = value.get("baseline")
    if (
        isinstance(candidate, bool)
        or not isinstance(candidate, (int, float))
        or isinstance(baseline, bool)
        or not isinstance(baseline, (int, float))
        or not math.isfinite(float(candidate))
        or not math.isfinite(float(baseline))
    ):
        raise AsrEnterpriseValueLabError(f"ASR {name} metric is invalid")
    return {"candidate": float(candidate), "baseline": float(baseline)}


def _require_formal_gates(
    report: PairedMetricReport,
    evaluation: dict[str, Any],
) -> None:
    if set(report.evidence.hard_gate_results) != ASR_HARD_GATES:
        raise AsrEnterpriseValueLabError("ASR formal hard-gate contract drifted")
    failed = sorted(
        gate for gate, passed in report.evidence.hard_gate_results.items() if not passed
    )
    if failed:
        raise AsrEnterpriseValueLabError(
            "ASR formal hard gates failed: "
            f"{','.join(failed)}; "
            f"candidate_wer={float(evaluation['candidate_wer']):.6f}; "
            f"candidate_cer={float(evaluation['candidate_cer']):.6f}; "
            f"candidate_noise_wer={float(evaluation['candidate_noise_wer']):.6f}; "
            f"wer_improvement={float(evaluation['wer_improvement']):.6f}; "
            f"noise_wer_improvement={float(evaluation['noise_wer_improvement']):.6f}"
        )
    if (
        float(evaluation["wer_improvement"])
        < _TRAINING_CONFIG["minimum_wer_improvement"]
        or float(evaluation["noise_wer_improvement"])
        < _TRAINING_CONFIG["minimum_noise_wer_improvement"]
    ):
        raise AsrEnterpriseValueLabError("ASR Gold improvement is insufficient")


def _dataset_document(
    *,
    dataset_path: Path,
    source_row_count: int,
    train_rows: list[_AudioRow],
    validation_rows: list[_AudioRow],
    gold_rows: list[_AudioRow],
    gold_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    split_records = {
        "train": _split_records(train_rows, _TRAIN_CHAPTER),
        "validation": _split_records(validation_rows, _VALIDATION_CHAPTER),
        "gold": _split_records(gold_rows, _GOLD_CHAPTER),
    }
    split_sha256s = {
        split: _digest(records) for split, records in split_records.items()
    }
    return {
        "schema_version": "enterprise-asr-governed-dataset/v1",
        "classification": CLASSIFICATION,
        "source": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "file_name": dataset_path.name,
            "file_sha256": _file_sha256(dataset_path),
            "source_row_count": source_row_count,
            "source_audio_license": "LIBRISPEECH_PUBLIC_DOMAIN",
            "repository_code_license": "Apache-2.0",
        },
        "policy": {
            "project_enterprise_use_authorized": True,
            "frozen_gold_excluded_from_training_and_selection": True,
            "source_ids_disjoint_across_splits": True,
            "risk_rule": "HIGH when normalized transcript word_count >= 20",
            "gold_noise_rule": (
                "MEDIUM for word_count >= 20; otherwise LOW for odd fixed index; "
                "otherwise CLEAN"
            ),
            "noise_profile": _TRAINING_CONFIG["noise_profile"],
            "noise_snr_db": _TRAINING_CONFIG["noise_snr_db"],
            "noise_components": ["seeded_white", "60hz", "180hz"],
        },
        "splits": split_records,
        "split_sha256s": split_sha256s,
        "gold_cases": gold_manifest,
    }


def _split_records(rows: list[_AudioRow], chapter_id: int) -> dict[str, Any]:
    return {
        "chapter_id": chapter_id,
        "source_ids": [row.source_id for row in rows],
        "rows": [
            {
                "source_id": row.source_id,
                "transcript_sha256": sha256(row.transcript.encode()).hexdigest(),
                "duration_seconds": row.duration_seconds,
            }
            for row in rows
        ],
    }


def _dataset_manifest_is_disjoint(document: dict[str, Any]) -> bool:
    splits = document.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "gold"}:
        return False
    source_sets: list[set[str]] = []
    for split in ("train", "validation", "gold"):
        value = splits.get(split)
        if not isinstance(value, dict):
            return False
        source_ids = value.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            return False
        normalized = {item for item in source_ids if isinstance(item, str) and item}
        if len(normalized) != len(source_ids):
            return False
        source_sets.append(normalized)
    return len(set.union(*source_sets)) == sum(len(values) for values in source_sets)


def _observation_document(observation: ModelObservation) -> dict[str, Any]:
    return {
        "case_id": observation.case_id,
        "output_text": observation.output_text,
        "latency_ms": observation.latency_ms,
        "cost_usd": observation.cost_usd,
        "input_tokens": observation.input_tokens,
        "output_tokens": observation.output_tokens,
        "capabilities": sorted(observation.capabilities),
        "structured_output_valid": observation.structured_output_valid,
        "transcript_text": observation.transcript_text,
        "runtime_evidence": observation.runtime_evidence,
    }


def _word_error_rate(reference: str, transcript: str) -> float:
    expected_words, _ = _transcript_units(reference)
    actual_words, _ = _transcript_units(transcript)
    return _edit_distance(expected_words, actual_words) / max(len(expected_words), 1)


def _transcript_units(value: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    words = tuple(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z0-9]+", normalized))
    return words, tuple("".join(words))


def _edit_distance(expected: tuple[str, ...], actual: tuple[str, ...]) -> int:
    previous = list(range(len(actual) + 1))
    for expected_index, expected_item in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_item in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[actual_index] + 1,
                    previous[actual_index - 1] + (expected_item != actual_item),
                )
            )
        previous = current
    return previous[-1]


def _verify_inputs(model_path: Path, dataset_path: Path) -> None:
    if _file_sha256(dataset_path) != DATASET_FILE_SHA256:
        raise AsrEnterpriseValueLabError("fixed ASR dataset file changed")
    required = {
        "README.md",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    observed = {path.name for path in model_path.iterdir() if path.is_file()}
    if not required.issubset(observed):
        raise AsrEnterpriseValueLabError("fixed Whisper snapshot is incomplete")
    weight_path = model_path / "pytorch_model.bin"
    if _file_sha256(weight_path) != MODEL_WEIGHT_SHA256:
        raise AsrEnterpriseValueLabError("fixed Whisper weight changed")
    readme = (model_path / "README.md").read_text(encoding="utf-8").casefold()
    if "license: apache-2.0" not in readme:
        raise AsrEnterpriseValueLabError("Whisper Apache-2.0 license marker is missing")


def _directory_manifest(directory: Path) -> list[dict[str, Any]]:
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise AsrEnterpriseValueLabError("artifact directory is missing")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if path.is_symlink():
            raise AsrEnterpriseValueLabError("artifact directory contains a symbolic link")
        if not path.is_file():
            continue
        if path.stat().st_size <= 0:
            raise AsrEnterpriseValueLabError("artifact directory contains an empty file")
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not entries:
        raise AsrEnterpriseValueLabError("artifact directory is empty")
    return entries


def _manifest_digest(entries: list[dict[str, Any]]) -> str:
    return _digest(entries)


def _file_evidence(relative_path: Path, physical_path: Path) -> dict[str, Any]:
    return {
        "path": relative_path.as_posix(),
        "size_bytes": physical_path.stat().st_size,
        "sha256": _file_sha256(physical_path),
    }


def _verify_file_evidence(root: Path, evidence: FileEvidence) -> Path:
    path = _inside_file(root, Path(evidence.path))
    if path.stat().st_size != evidence.size_bytes or _file_sha256(path) != evidence.sha256:
        raise AsrEnterpriseValueLabError("ASR evidence file changed")
    return path


def _write_latest(
    root: Path,
    acceptance_path: Path,
    report: AsrEnterpriseValueReport,
) -> None:
    latest = {
        "schema_version": "enterprise-asr-value-latest/v1",
        "classification": CLASSIFICATION,
        "status": report.status,
        "run_id": report.run_id,
        "report": acceptance_path.relative_to(root).as_posix(),
        "report_sha256": _file_sha256(acceptance_path),
        "evidence_chain_sha256": report.evidence_chain_sha256,
    }
    _write_json(root / OUTPUT_RELATIVE / "latest.json", latest)


def _latest_report_path(root: Path) -> Path:
    latest_path = _inside_file(root, OUTPUT_RELATIVE / "latest.json")
    latest = _load_object(latest_path)
    report_value = latest.get("report")
    if not isinstance(report_value, str) or not report_value:
        raise AsrEnterpriseValueLabError("ASR latest pointer is invalid")
    report_path = _inside_file(root, Path(report_value))
    if latest.get("report_sha256") != _file_sha256(report_path):
        raise AsrEnterpriseValueLabError("ASR latest report digest changed")
    return report_path


def _inside_file(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    target = target.resolve(strict=True)
    _require_inside(root, target)
    if not target.is_file():
        raise AsrEnterpriseValueLabError("ASR evidence path is not a file")
    return target


def _inside_directory(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    target = target.resolve(strict=True)
    _require_inside(root, target)
    if not target.is_dir():
        raise AsrEnterpriseValueLabError("ASR evidence path is not a directory")
    return target


def _require_inside(root: Path, target: Path) -> None:
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AsrEnterpriseValueLabError("ASR evidence path escaped repository") from exc


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AsrEnterpriseValueLabError("ASR evidence JSON is invalid") from exc
    if not isinstance(value, dict):
        raise AsrEnterpriseValueLabError("ASR evidence JSON is invalid")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_image_digest(value: str) -> None:
    if not (
        value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    ):
        raise AsrEnterpriseValueLabError("ASR runtime image digest is invalid")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()
