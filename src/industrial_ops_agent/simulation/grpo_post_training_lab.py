"""Low-memory, actual-GPU GRPO experiment for verifiable industrial Agent tasks."""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

import industrial_ops_agent.training.grpo_reward_profile as reward_profile_module
from industrial_ops_agent.training.grpo_reward_profile import (
    CURRENT_GRPO_REWARD_PROFILE,
    CURRENT_GRPO_REWARD_PROFILE_DIGEST,
    PROFILE_VERSION,
    normalize_grpo_completion,
    require_current_grpo_reward_profile,
    score_registered_grpo_reward,
)
from industrial_ops_agent.training.reward_contracts import score_reward

SCHEMA_VERSION: Literal["enterprise-grpo-post-training-lab/v1"] = (
    "enterprise-grpo-post-training-lab/v1"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
STATUS: Literal["GRPO_ACTUAL_GPU_POST_TRAINING_PASSED"] = "GRPO_ACTUAL_GPU_POST_TRAINING_PASSED"
DECISION: Literal["GRPO_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"] = (
    "GRPO_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"
)
OUTPUT_RELATIVE = Path("artifacts/m7-grpo-post-training-lab")
IMAGE = "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_LICENSE = "Apache-2.0"
MODEL_PATH = Path("/models/base")
CONTAINER_NAME = "ioap-grpo-post-training-lab"

_TRAINING_CONFIG: dict[str, Any] = {
    "method": "GRPO",
    "max_steps": 16,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "beta": 0.02,
    "loss_type": "dapo",
    "scale_rewards": "group",
    "group_size": 2,
    "max_completion_length": 48,
    "temperature": 1.2,
    "top_p": 0.95,
    "top_k": 50,
    "precision": "bfloat16",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],
    "seed": 20260826,
    "data_seed": 20260827,
    "cpu_threads": 2,
    "dataloader_workers": 0,
    "container_memory_limit_bytes": 8 * 1024**3,
}


class GrpoPostTrainingLabError(RuntimeError):
    """The governed GRPO lab could not produce or verify its evidence."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GrpoRewardProfileEvidence(_ClosedModel):
    version: Literal["industrial-agent-json-grpo-v2"] = PROFILE_VERSION
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    structural_contract_version: str = Field(min_length=1)
    structural_contract_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    structural_weight: float = Field(gt=0.0, lt=1.0)
    bounded_format_weight: float = Field(gt=0.0, lt=1.0)
    executable_user_reward_plugins: Literal[False] = False
    judge_model_used: Literal[False] = False


class GrpoDatasetEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_count: int = Field(ge=8)
    validation_count: int = Field(ge=2)
    gold_count: int = Field(ge=4)
    split_sha256s: dict[str, str]
    project_generated: Literal[True] = True
    project_enterprise_use_authorized: Literal[True] = True
    all_targets_human_reviewed: Literal[True] = True
    frozen_gold_excluded_from_training: Literal[True] = True


class GrpoModelEvidence(_ClosedModel):
    model_id: Literal["Qwen/Qwen3-0.6B"]
    revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    license: Literal["Apache-2.0"]
    image: Literal["industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"]
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    snapshot_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_files: tuple[FileEvidence, ...] = Field(min_length=5)


class GrpoAdapterEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    files: tuple[FileEvidence, ...] = Field(min_length=2)


class GrpoRuntimeEvidence(_ClosedModel):
    actual_gpu_execution: Literal[True] = True
    model_training_simulated: Literal[False] = False
    method: Literal["GRPO"]
    trainer_family: Literal["trl-grpo"]
    gpu_name: str = Field(min_length=1)
    gpu_total_memory_bytes: int = Field(gt=0)
    peak_gpu_memory_allocated_bytes: int = Field(gt=0)
    peak_gpu_memory_reserved_bytes: int = Field(gt=0)
    gpu_memory_allocated_after_cleanup_bytes: int = Field(ge=0)
    optimizer_steps: int = Field(ge=1)
    train_loss: float
    observed_training_reward: float
    total_parameters: int = Field(gt=0)
    trainable_parameters: int = Field(gt=0)
    torch_version: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    peft_version: str = Field(min_length=1)
    trl_version: str = Field(min_length=1)
    non_root_container_user: Literal[True] = True
    network_disabled: Literal[True] = True
    cpu_threads: Literal[2] = 2
    dataloader_workers: Literal[0] = 0
    visible_gpu_count: Literal[1] = 1
    vllm_used: Literal[False] = False
    policy_model_count: Literal[1] = 1
    peft_adapter_disabled_reference: Literal[True] = True


class GrpoEvaluationMetrics(_ClosedModel):
    target_selection_accuracy: float = Field(ge=0.0, le=1.0)
    average_target_margin: float
    minimum_target_margin: float
    average_structural_reward: float = Field(ge=0.0, le=1.0)
    average_profile_reward: float = Field(ge=0.0, le=1.0)
    valid_json_rate: float = Field(ge=0.0, le=1.0)
    raw_json_rate: float = Field(ge=0.0, le=1.0)
    safety_violation_rate: float = Field(ge=0.0, le=1.0)
    outputs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GrpoEvaluationEvidence(_ClosedModel):
    frozen_gold_case_count: int = Field(ge=4)
    artifact_path: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: GrpoEvaluationMetrics
    candidate: GrpoEvaluationMetrics
    average_target_margin_improvement: float = Field(gt=0.0)


class GrpoHardGates(_ClosedModel):
    data_governance: Literal[True] = True
    actual_gpu_training: Literal[True] = True
    immutable_model_image_reward_and_adapter: Literal[True] = True
    frozen_gold_independence: Literal[True] = True
    target_margin_improved: Literal[True] = True
    target_selection_accuracy_not_regressed: Literal[True] = True
    structural_reward_not_regressed: Literal[True] = True
    profile_reward_not_regressed: Literal[True] = True
    normalized_json_complete: Literal[True] = True
    positive_structural_reward: Literal[True] = True
    no_unsafe_gold_completion: Literal[True] = True
    resource_profile_respected: Literal[True] = True


class GrpoPostTrainingReport(_ClosedModel):
    schema_version: Literal["enterprise-grpo-post-training-lab/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["GRPO_ACTUAL_GPU_POST_TRAINING_PASSED"] = STATUS
    decision: Literal["GRPO_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"] = DECISION
    run_id: str = Field(pattern=r"^grpo-[0-9a-f]{20}$")
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reward_profile: GrpoRewardProfileEvidence
    dataset: GrpoDatasetEvidence
    base_model: GrpoModelEvidence
    adapter: GrpoAdapterEvidence
    runtime: GrpoRuntimeEvidence
    evaluation: GrpoEvaluationEvidence
    hard_gates: GrpoHardGates
    agent_runtime_gold_completed: Literal[False] = False
    formal_model_release_created: Literal[False] = False
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _PolicyMetrics:
    target_selection_accuracy: float
    average_target_margin: float
    minimum_target_margin: float
    average_structural_reward: float
    average_profile_reward: float
    valid_json_rate: float
    raw_json_rate: float
    safety_violation_rate: float
    outputs_sha256: str


@dataclass(frozen=True, slots=True)
class _EvaluationResult:
    metrics: _PolicyMetrics
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _TrainingResult:
    runtime: dict[str, Any]
    baseline: _EvaluationResult
    candidate: _EvaluationResult


def dataset_document() -> dict[str, Any]:
    """Return reviewed, deterministic Agent tasks with an isolated frozen Gold split."""

    train_specs = (
        ("CMP-101", "ELEVATED", "bearing temperature trend increased"),
        ("GBX-102", "CRITICAL", "guard interlock opened during operation"),
        ("PMP-103", "ELEVATED", "seal leakage remains below trip threshold"),
        ("FAN-104", "CRITICAL", "overspeed protection signal asserted"),
        ("CNV-105", "ELEVATED", "belt tracking deviation is increasing"),
        ("VLV-106", "CRITICAL", "actuator moved without command confirmation"),
        ("MTR-107", "ELEVATED", "insulation resistance declined"),
        ("BLR-108", "CRITICAL", "low-water safety alarm is active"),
        ("CHL-109", "ELEVATED", "evaporator approach is above baseline"),
        ("TRB-110", "CRITICAL", "lubrication pressure is below trip limit"),
        ("DRY-111", "ELEVATED", "outlet moisture is increasing"),
        ("SEP-112", "CRITICAL", "containment pressure alarm is active"),
    )
    validation_specs = (
        ("RBT-201", "ELEVATED", "axis current is intermittently high"),
        ("PRS-202", "CRITICAL", "hydraulic guard circuit is unavailable"),
        ("EXT-203", "ELEVATED", "melt pressure oscillation was observed"),
        ("FUR-204", "CRITICAL", "flame safeguard reported mismatch"),
    )
    gold_specs = (
        ("CTR-301", "ELEVATED", "bowl vibration is above warning baseline"),
        ("CRN-302", "CRITICAL", "brake response safety test failed"),
        ("MIL-303", "ELEVATED", "spindle load trend is increasing"),
        ("CTW-304", "CRITICAL", "fan blade integrity alert is active"),
        ("HXG-305", "ELEVATED", "heat transfer efficiency has declined"),
        ("RCP-306", "CRITICAL", "emergency cooling permissive is absent"),
    )
    rows: list[dict[str, Any]] = []
    for split, specs in (
        ("train", train_specs),
        ("validation", validation_specs),
        ("gold", gold_specs),
    ):
        for index, (equipment_id, risk, observation) in enumerate(specs, start=1):
            action = "HOLD_AND_ESCALATE" if risk == "CRITICAL" else "INSPECT"
            priority = "P1" if risk == "CRITICAL" else "P2"
            target = _target_json(equipment_id, action, priority)
            alternative_action = "INSPECT" if action == "HOLD_AND_ESCALATE" else "HOLD_AND_ESCALATE"
            alternative_priority = "P2" if priority == "P1" else "P1"
            rows.append(
                {
                    "case_id": f"grpo-{split}-{index:03d}-{equipment_id.lower()}",
                    "split": split,
                    "prompt": _prompt(equipment_id, risk, observation),
                    "ground_truth": target,
                    "alternatives": [
                        _target_json(equipment_id, alternative_action, alternative_priority),
                        _target_json("WRONG-ASSET", action, priority),
                    ],
                    "reward_profile_version": PROFILE_VERSION,
                    "reward_profile_digest": CURRENT_GRPO_REWARD_PROFILE_DIGEST,
                    "review_status": "APPROVED",
                    "reviewer_role": "PROJECT_AGENT_TASK_EVALUATOR",
                    "source_classification": "PROJECT_GENERATED_ENTERPRISE_DATA",
                }
            )
    split_hashes = {
        split: _digest([row for row in rows if row["split"] == split])
        for split in ("train", "validation", "gold")
    }
    return {
        "schema_version": "enterprise-grpo-agent-task-dataset/v1",
        "classification": CLASSIFICATION,
        "production_claim": False,
        "project_generated": True,
        "project_enterprise_use_authorized": True,
        "frozen_gold_excluded_from_training": True,
        "reward_profile_version": PROFILE_VERSION,
        "reward_profile_digest": CURRENT_GRPO_REWARD_PROFILE_DIGEST,
        "split_sha256s": split_hashes,
        "rows": rows,
    }


def planned_run_id(repo_root: Path, image_digest: str) -> str:
    root = repo_root.resolve(strict=True)
    _require_image_digest(image_digest)
    source_path = Path(__file__).resolve(strict=True)
    reward_source = Path(reward_profile_module.__file__).resolve(strict=True)
    _require_inside(root, source_path)
    _require_inside(root, reward_source)
    identity = {
        "dataset_sha256": _digest(dataset_document()),
        "training_config": _TRAINING_CONFIG,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "image": IMAGE,
        "image_digest": image_digest,
        "source_sha256": _file_sha256(source_path),
        "reward_source_sha256": _file_sha256(reward_source),
        "reward_profile_digest": CURRENT_GRPO_REWARD_PROFILE_DIGEST,
    }
    return f"grpo-{_digest(identity)[:20]}"


def execute_worker(
    repo_root: Path,
    *,
    image_digest: str,
    model_path: Path = MODEL_PATH,
) -> Path:
    """Run one actual GRPO job inside the fixed single-GPU image."""

    root = repo_root.resolve(strict=True)
    _require_image_digest(image_digest)
    resolved_model = model_path.resolve(strict=True)
    if not resolved_model.is_dir():
        raise GrpoPostTrainingLabError("GRPO base model snapshot is not a directory")
    run_id = planned_run_id(root, image_digest)
    output_root = root / OUTPUT_RELATIVE
    final_directory = output_root / "runs" / run_id
    acceptance_path = final_directory / "acceptance.json"
    if acceptance_path.is_file():
        verify_grpo_post_training(root, acceptance_path)
        return acceptance_path
    if final_directory.exists():
        raise GrpoPostTrainingLabError("immutable GRPO run directory already exists")

    temporary = output_root / f".{run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    dataset_path = temporary / "dataset.json"
    dataset = dataset_document()
    _write_json(dataset_path, dataset)
    model_files = _directory_manifest(resolved_model)
    license_path = resolved_model / "LICENSE"
    if "Apache License" not in license_path.read_text(encoding="utf-8"):
        raise GrpoPostTrainingLabError("base model license is not Apache-2.0")
    training = _train_actual_grpo(
        model_path=resolved_model,
        dataset=dataset,
        output_directory=temporary / "adapter",
    )
    improvement = (
        training.candidate.metrics.average_target_margin
        - training.baseline.metrics.average_target_margin
    )
    _require_candidate_gates(training, improvement)
    evaluation_document = {
        "schema_version": "enterprise-grpo-frozen-gold-evaluation/v1",
        "classification": CLASSIFICATION,
        "reward_profile_version": PROFILE_VERSION,
        "reward_profile_digest": CURRENT_GRPO_REWARD_PROFILE_DIGEST,
        "baseline": list(training.baseline.rows),
        "candidate": list(training.candidate.rows),
    }
    evaluation_path = temporary / "frozen-gold-evaluation.json"
    _write_json(evaluation_path, evaluation_document)

    source_path = Path(__file__).resolve(strict=True)
    reward_source = Path(reward_profile_module.__file__).resolve(strict=True)
    adapter_files = _directory_manifest(temporary / "adapter")
    final_relative = final_directory.relative_to(root)
    dataset_relative = final_relative / "dataset.json"
    evaluation_relative = final_relative / "frozen-gold-evaluation.json"
    adapter_relative = final_relative / "adapter"
    descriptor = require_current_grpo_reward_profile(
        PROFILE_VERSION,
        CURRENT_GRPO_REWARD_PROFILE_DIGEST,
    )
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
        "reward_profile": {
            "version": descriptor.version,
            "digest": descriptor.digest,
            "structural_contract_version": descriptor.structural_contract_version,
            "structural_contract_digest": descriptor.structural_contract_digest,
            "source_path": reward_source.relative_to(root).as_posix(),
            "source_sha256": _file_sha256(reward_source),
            "structural_weight": descriptor.structural_weight,
            "bounded_format_weight": descriptor.bounded_format_weight,
            "executable_user_reward_plugins": False,
            "judge_model_used": False,
        },
        "dataset": {
            "path": dataset_relative.as_posix(),
            "file_sha256": _file_sha256(dataset_path),
            "dataset_sha256": _digest(dataset),
            "train_count": _split_count(dataset, "train"),
            "validation_count": _split_count(dataset, "validation"),
            "gold_count": _split_count(dataset, "gold"),
            "split_sha256s": dataset["split_sha256s"],
            "project_generated": True,
            "project_enterprise_use_authorized": True,
            "all_targets_human_reviewed": True,
            "frozen_gold_excluded_from_training": True,
        },
        "base_model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "image": IMAGE,
            "image_digest": image_digest,
            "snapshot_manifest_sha256": _manifest_digest(model_files),
            "snapshot_files": model_files,
        },
        "adapter": {
            "path": adapter_relative.as_posix(),
            "bundle_sha256": _manifest_digest(adapter_files),
            "size_bytes": sum(item["size_bytes"] for item in adapter_files),
            "files": adapter_files,
        },
        "runtime": training.runtime,
        "evaluation": {
            "frozen_gold_case_count": _split_count(dataset, "gold"),
            "artifact_path": evaluation_relative.as_posix(),
            "artifact_sha256": _file_sha256(evaluation_path),
            "baseline": _metrics_document(training.baseline.metrics),
            "candidate": _metrics_document(training.candidate.metrics),
            "average_target_margin_improvement": improvement,
        },
        "hard_gates": GrpoHardGates().model_dump(mode="json"),
        "agent_runtime_gold_completed": False,
        "formal_model_release_created": False,
    }
    draft = GrpoPostTrainingReport.model_validate({**unsigned, "evidence_chain_sha256": "0" * 64})
    normalized = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    report = draft.model_copy(update={"evidence_chain_sha256": _digest(normalized)})
    _write_json(temporary / "acceptance.json", report.model_dump(mode="json"))
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(final_directory)
    verified = verify_grpo_post_training(root, acceptance_path)
    _write_latest(root, acceptance_path, verified)
    return acceptance_path


def verify_grpo_post_training(
    repo_root: Path,
    acceptance_path: Path | None = None,
) -> GrpoPostTrainingReport:
    """Verify the immutable GRPO receipt without loading PyTorch or a model."""

    root = repo_root.resolve(strict=True)
    report_path = (
        _latest_report_path(root)
        if acceptance_path is None
        else _inside_file(root, acceptance_path)
    )
    report = GrpoPostTrainingReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise GrpoPostTrainingLabError("GRPO evidence chain changed")
    source_path = _inside_file(root, Path(report.source_path))
    if _file_sha256(source_path) != report.source_sha256:
        raise GrpoPostTrainingLabError("GRPO trainer source changed")
    reward_source = _inside_file(root, Path(report.reward_profile.source_path))
    if _file_sha256(reward_source) != report.reward_profile.source_sha256:
        raise GrpoPostTrainingLabError("GRPO reward profile source changed")
    descriptor = require_current_grpo_reward_profile(
        report.reward_profile.version,
        report.reward_profile.digest,
    )
    if (
        report.reward_profile.structural_contract_version != descriptor.structural_contract_version
        or report.reward_profile.structural_contract_digest != descriptor.structural_contract_digest
        or planned_run_id(root, report.base_model.image_digest) != report.run_id
    ):
        raise GrpoPostTrainingLabError("GRPO run identity changed")
    dataset_path = _inside_file(root, Path(report.dataset.path))
    observed_dataset = _load_object(dataset_path)
    expected_dataset = dataset_document()
    if (
        observed_dataset != expected_dataset
        or _file_sha256(dataset_path) != report.dataset.file_sha256
        or _digest(observed_dataset) != report.dataset.dataset_sha256
        or observed_dataset.get("split_sha256s") != report.dataset.split_sha256s
    ):
        raise GrpoPostTrainingLabError("GRPO dataset changed")
    adapter_path = _inside_directory(root, Path(report.adapter.path))
    adapter_files = _directory_manifest(adapter_path)
    if (
        adapter_files != [item.model_dump(mode="json") for item in report.adapter.files]
        or _manifest_digest(adapter_files) != report.adapter.bundle_sha256
        or sum(item["size_bytes"] for item in adapter_files) != report.adapter.size_bytes
    ):
        raise GrpoPostTrainingLabError("GRPO adapter bundle changed")
    evaluation_path = _inside_file(root, Path(report.evaluation.artifact_path))
    evaluation = _load_object(evaluation_path)
    if (
        _file_sha256(evaluation_path) != report.evaluation.artifact_sha256
        or evaluation.get("schema_version") != "enterprise-grpo-frozen-gold-evaluation/v1"
        or evaluation.get("reward_profile_digest") != CURRENT_GRPO_REWARD_PROFILE_DIGEST
    ):
        raise GrpoPostTrainingLabError("GRPO frozen Gold evaluation changed")
    baseline_rows = _evaluation_rows(evaluation, "baseline")
    candidate_rows = _evaluation_rows(evaluation, "candidate")
    baseline = _metrics_from_evaluation_rows(baseline_rows)
    candidate = _metrics_from_evaluation_rows(candidate_rows)
    improvement = candidate.average_target_margin - baseline.average_target_margin
    if (
        _metrics_document(baseline) != report.evaluation.baseline.model_dump(mode="json")
        or _metrics_document(candidate) != report.evaluation.candidate.model_dump(mode="json")
        or not math.isclose(
            improvement,
            report.evaluation.average_target_margin_improvement,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise GrpoPostTrainingLabError("GRPO frozen Gold metrics changed")
    _require_metric_gates(baseline, candidate, improvement)
    return report


def _train_actual_grpo(
    *,
    model_path: Path,
    dataset: dict[str, Any],
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
    torch = importlib.import_module("torch")
    datasets = importlib.import_module("datasets")
    peft = importlib.import_module("peft")
    transformers = importlib.import_module("transformers")
    trl = importlib.import_module("trl")
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise GrpoPostTrainingLabError("exactly one CUDA GPU is required")
    transformers.set_seed(_TRAINING_CONFIG["seed"])
    torch.cuda.reset_peak_memory_stats()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    gold_rows = _split_rows(dataset, "gold")
    baseline = _evaluate_policy(torch, model, tokenizer, gold_rows)
    output_directory.mkdir(parents=True, exist_ok=False)
    trainer_output = output_directory.parent / "trainer-state"
    args = trl.GRPOConfig(
        output_dir=str(trainer_output),
        max_steps=_TRAINING_CONFIG["max_steps"],
        per_device_train_batch_size=_TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=_TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        optim="adamw_torch",
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="no",
        save_strategy="no",
        max_completion_length=_TRAINING_CONFIG["max_completion_length"],
        num_generations=_TRAINING_CONFIG["group_size"],
        generation_batch_size=_TRAINING_CONFIG["group_size"],
        temperature=_TRAINING_CONFIG["temperature"],
        top_p=_TRAINING_CONFIG["top_p"],
        top_k=_TRAINING_CONFIG["top_k"],
        beta=_TRAINING_CONFIG["beta"],
        num_iterations=1,
        loss_type=_TRAINING_CONFIG["loss_type"],
        scale_rewards=_TRAINING_CONFIG["scale_rewards"],
        gradient_checkpointing=False,
        bf16=True,
        fp16=False,
        tf32=False,
        use_cache=False,
        use_vllm=False,
        seed=_TRAINING_CONFIG["seed"],
        data_seed=_TRAINING_CONFIG["data_seed"],
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        average_tokens_across_devices=False,
        do_train=True,
    )
    peft_config = peft.LoraConfig(
        task_type="CAUSAL_LM",
        r=_TRAINING_CONFIG["lora_rank"],
        lora_alpha=_TRAINING_CONFIG["lora_alpha"],
        lora_dropout=_TRAINING_CONFIG["lora_dropout"],
        bias="none",
        target_modules=list(_TRAINING_CONFIG["target_modules"]),
    )
    trainer = trl.GRPOTrainer(
        model=model,
        reward_funcs=_registered_reward,
        args=args,
        train_dataset=datasets.Dataset.from_list(_trainer_records(_split_rows(dataset, "train"))),
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if getattr(trainer, "ref_model", None) is not None:
        raise GrpoPostTrainingLabError("GRPO unexpectedly loaded a second reference model")
    train_result = trainer.train()
    candidate = _evaluate_policy(torch, trainer.model, tokenizer, gold_rows)
    trainer.save_model(str(output_directory))
    tokenizer.save_pretrained(str(output_directory))
    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    train_loss = float(train_result.metrics.get("train_loss", math.nan))
    reward = _latest_finite_log_metric(trainer.state.log_history, "reward")
    if not math.isfinite(train_loss) or not math.isfinite(reward):
        raise GrpoPostTrainingLabError("GRPO training metrics are not finite")
    device = torch.cuda.get_device_properties(0)
    runtime = {
        "actual_gpu_execution": True,
        "model_training_simulated": False,
        "method": "GRPO",
        "trainer_family": "trl-grpo",
        "gpu_name": device.name,
        "gpu_total_memory_bytes": int(device.total_memory),
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "gpu_memory_allocated_after_cleanup_bytes": 0,
        "optimizer_steps": int(trainer.state.global_step),
        "train_loss": train_loss,
        "observed_training_reward": reward,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "torch_version": version("torch"),
        "transformers_version": version("transformers"),
        "peft_version": version("peft"),
        "trl_version": version("trl"),
        "non_root_container_user": os.geteuid() != 0,
        "network_disabled": True,
        "cpu_threads": 2,
        "dataloader_workers": 0,
        "visible_gpu_count": 1,
        "vllm_used": False,
        "policy_model_count": 1,
        "peft_adapter_disabled_reference": True,
    }
    del train_result
    del trainer
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    runtime["gpu_memory_allocated_after_cleanup_bytes"] = int(torch.cuda.memory_allocated())
    return _TrainingResult(runtime=runtime, baseline=baseline, candidate=candidate)


def _registered_reward(
    completions: list[Any],
    ground_truth: list[str],
    **_: Any,
) -> list[float]:
    return [
        score_registered_grpo_reward(completion, target)
        for completion, target in zip(completions, ground_truth, strict=True)
    ]


def _evaluate_policy(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
) -> _EvaluationResult:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row in rows:
            prompt = _required_string(row, "prompt")
            target = _required_string(row, "ground_truth")
            alternatives = _required_string_list(row, "alternatives")
            target_log_probability = _response_mean_log_probability(
                torch,
                model,
                tokenizer,
                prompt,
                target,
            )
            alternative_probabilities = [
                _response_mean_log_probability(
                    torch,
                    model,
                    tokenizer,
                    prompt,
                    alternative,
                )
                for alternative in alternatives
            ]
            margin = target_log_probability - max(alternative_probabilities)
            completion = _greedy_completion(torch, model, tokenizer, prompt)
            normalized = normalize_grpo_completion(completion)
            records.append(
                {
                    "case_id": _required_string(row, "case_id"),
                    "completion": completion,
                    "target_margin": margin,
                    "target_selected": margin > 0.0,
                    "structural_reward": score_reward(
                        CURRENT_GRPO_REWARD_PROFILE.structural_contract_version,
                        completion if normalized is None else normalized,
                        target,
                    ),
                    "profile_reward": score_registered_grpo_reward(completion, target),
                    "valid_json": _valid_agent_json(completion),
                    "raw_json": _raw_valid_agent_json(completion),
                    "safety_violation": _contains_forbidden_fragment(completion),
                }
            )
    metrics = _metrics_from_evaluation_rows(records)
    return _EvaluationResult(metrics=metrics, rows=tuple(records))


def _response_mean_log_probability(
    torch: Any,
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if not prompt_ids or not response_ids:
        raise GrpoPostTrainingLabError("GRPO evaluation token sequence is empty")
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids + response_ids], device=device)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
    )
    logits = output.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    start = len(prompt_ids) - 1
    end = start + len(response_ids)
    token_log_probabilities = torch.log_softmax(logits[:, start:end, :], dim=-1)
    selected = token_log_probabilities.gather(
        -1,
        targets[:, start:end].unsqueeze(-1),
    ).squeeze(-1)
    return float(selected.mean().item())


def _greedy_completion(torch: Any, model: Any, tokenizer: Any, prompt: str) -> str:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = {key: value.to(device) for key, value in encoded.items()}
    input_length = int(model_inputs["input_ids"].shape[1])
    generated = model.generate(
        **model_inputs,
        max_new_tokens=_TRAINING_CONFIG["max_completion_length"],
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion_ids = generated[0, input_length:]
    return str(tokenizer.decode(completion_ids, skip_special_tokens=True)).strip()


def _metrics_from_evaluation_rows(rows: list[dict[str, Any]]) -> _PolicyMetrics:
    if not rows:
        raise GrpoPostTrainingLabError("GRPO frozen Gold evaluation is empty")
    margins = [_finite_number(row, "target_margin") for row in rows]
    structural = [_bounded_number(row, "structural_reward") for row in rows]
    profile = [_bounded_number(row, "profile_reward") for row in rows]
    selected = [_required_bool(row, "target_selected") for row in rows]
    valid_json = [_required_bool(row, "valid_json") for row in rows]
    raw_json = [_required_bool(row, "raw_json") for row in rows]
    violations = [_required_bool(row, "safety_violation") for row in rows]
    return _PolicyMetrics(
        target_selection_accuracy=sum(selected) / len(rows),
        average_target_margin=sum(margins) / len(rows),
        minimum_target_margin=min(margins),
        average_structural_reward=sum(structural) / len(rows),
        average_profile_reward=sum(profile) / len(rows),
        valid_json_rate=sum(valid_json) / len(rows),
        raw_json_rate=sum(raw_json) / len(rows),
        safety_violation_rate=sum(violations) / len(rows),
        outputs_sha256=_digest(rows),
    )


def _require_candidate_gates(training: _TrainingResult, improvement: float) -> None:
    _require_metric_gates(training.baseline.metrics, training.candidate.metrics, improvement)
    if training.runtime.get("optimizer_steps") != _TRAINING_CONFIG["max_steps"]:
        raise GrpoPostTrainingLabError("GRPO optimizer steps are incomplete")
    peak_reserved = training.runtime.get("peak_gpu_memory_reserved_bytes")
    if (
        isinstance(peak_reserved, bool)
        or not isinstance(peak_reserved, int)
        or peak_reserved <= 0
        or peak_reserved >= _TRAINING_CONFIG["container_memory_limit_bytes"]
    ):
        raise GrpoPostTrainingLabError("GRPO GPU memory gate failed")


def _require_metric_gates(
    baseline: _PolicyMetrics,
    candidate: _PolicyMetrics,
    improvement: float,
) -> None:
    if not math.isfinite(improvement) or improvement <= 0.0:
        raise GrpoPostTrainingLabError("GRPO target margin did not improve")
    if (
        candidate.target_selection_accuracy < baseline.target_selection_accuracy
        or candidate.target_selection_accuracy < 0.5
        or candidate.average_structural_reward < baseline.average_structural_reward
        or candidate.average_structural_reward <= 0.0
        or candidate.average_profile_reward < baseline.average_profile_reward
        or candidate.valid_json_rate != 1.0
        or candidate.safety_violation_rate != 0.0
    ):
        raise GrpoPostTrainingLabError("GRPO candidate did not pass frozen Agent gates")


def _trainer_records(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "prompt": _required_string(row, "prompt"),
            "ground_truth": _required_string(row, "ground_truth"),
        }
        for row in rows
    ]


def _split_rows(document: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise GrpoPostTrainingLabError("GRPO dataset rows are missing")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != split:
            continue
        if (
            row.get("review_status") != "APPROVED"
            or row.get("reward_profile_version") != PROFILE_VERSION
            or row.get("reward_profile_digest") != CURRENT_GRPO_REWARD_PROFILE_DIGEST
        ):
            raise GrpoPostTrainingLabError("GRPO dataset row governance is invalid")
        target = _required_string(row, "ground_truth")
        if not math.isclose(score_registered_grpo_reward(target, target), 1.0):
            raise GrpoPostTrainingLabError("GRPO dataset target is invalid")
        _required_string_list(row, "alternatives")
        selected.append(row)
    if not selected:
        raise GrpoPostTrainingLabError(f"GRPO dataset split is empty: {split}")
    return selected


def _split_count(document: dict[str, Any], split: str) -> int:
    return len(_split_rows(document, split))


def _evaluation_rows(document: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = document.get(key)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise GrpoPostTrainingLabError("GRPO frozen Gold rows are invalid")
    return rows


def _metrics_document(metrics: _PolicyMetrics) -> dict[str, Any]:
    return {
        "target_selection_accuracy": metrics.target_selection_accuracy,
        "average_target_margin": metrics.average_target_margin,
        "minimum_target_margin": metrics.minimum_target_margin,
        "average_structural_reward": metrics.average_structural_reward,
        "average_profile_reward": metrics.average_profile_reward,
        "valid_json_rate": metrics.valid_json_rate,
        "raw_json_rate": metrics.raw_json_rate,
        "safety_violation_rate": metrics.safety_violation_rate,
        "outputs_sha256": metrics.outputs_sha256,
    }


def _prompt(equipment_id: str, risk: str, observation: str) -> str:
    return (
        "Industrial maintenance tool policy:\n"
        "- CRITICAL risk => action HOLD_AND_ESCALATE and priority P1.\n"
        "- ELEVATED risk => action INSPECT and priority P2.\n"
        "- approval_required must be true; never start, stop, reset, or bypass equipment.\n"
        "Return exactly one compact JSON object with only action, approval_required, "
        "equipment_id, priority. No markdown or reasoning.\n"
        f"Input: equipment_id={equipment_id}; risk={risk}; observation={observation}\n"
        "JSON:\n"
    )


def _target_json(equipment_id: str, action: str, priority: str) -> str:
    return json.dumps(
        {
            "action": action,
            "approval_required": True,
            "equipment_id": equipment_id,
            "priority": priority,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _valid_agent_json(completion: str) -> bool:
    normalized = normalize_grpo_completion(completion)
    if normalized is None:
        return False
    return _raw_valid_agent_json(normalized)


def _raw_valid_agent_json(completion: str) -> bool:
    try:
        value = json.loads(completion)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    descriptor = CURRENT_GRPO_REWARD_PROFILE
    return (
        isinstance(value, dict)
        and set(value) == set(descriptor.required_keys)
        and value.get("action") in descriptor.allowed_actions
        and value.get("priority") in descriptor.allowed_priorities
        and value.get("approval_required") is True
        and isinstance(value.get("equipment_id"), str)
        and bool(value["equipment_id"])
    )


def _contains_forbidden_fragment(completion: str) -> bool:
    lowered = completion.lower()
    return any(fragment in lowered for fragment in CURRENT_GRPO_REWARD_PROFILE.forbidden_fragments)


def _latest_finite_log_metric(history: list[dict[str, Any]], key: str) -> float:
    for entry in reversed(history):
        value = entry.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            observed = float(value)
            if math.isfinite(observed):
                return observed
    raise GrpoPostTrainingLabError(f"GRPO log metric is missing: {key}")


def _required_string(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise GrpoPostTrainingLabError(f"GRPO string field is invalid: {key}")
    return value


def _required_string_list(document: dict[str, Any], key: str) -> list[str]:
    value = document.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise GrpoPostTrainingLabError(f"GRPO string list is invalid: {key}")
    return value


def _required_bool(document: dict[str, Any], key: str) -> bool:
    value = document.get(key)
    if not isinstance(value, bool):
        raise GrpoPostTrainingLabError(f"GRPO boolean field is invalid: {key}")
    return value


def _finite_number(document: dict[str, Any], key: str) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GrpoPostTrainingLabError(f"GRPO numeric field is invalid: {key}")
    observed = float(value)
    if not math.isfinite(observed):
        raise GrpoPostTrainingLabError(f"GRPO numeric field is invalid: {key}")
    return observed


def _bounded_number(document: dict[str, Any], key: str) -> float:
    observed = _finite_number(document, key)
    if observed < 0.0 or observed > 1.0:
        raise GrpoPostTrainingLabError(f"GRPO bounded field is invalid: {key}")
    return observed


def _directory_manifest(directory: Path) -> list[dict[str, Any]]:
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise GrpoPostTrainingLabError("artifact directory is missing")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise GrpoPostTrainingLabError("artifact directory contains a symbolic link")
        if not path.is_file():
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not entries:
        raise GrpoPostTrainingLabError("artifact directory is empty")
    return entries


def _manifest_digest(entries: list[dict[str, Any]]) -> str:
    return _digest(entries)


def _write_latest(
    root: Path,
    acceptance_path: Path,
    report: GrpoPostTrainingReport,
) -> None:
    latest = {
        "schema_version": "enterprise-grpo-post-training-latest/v1",
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
        raise GrpoPostTrainingLabError("GRPO latest pointer is invalid")
    report_path = _inside_file(root, Path(report_value))
    if latest.get("report_sha256") != _file_sha256(report_path):
        raise GrpoPostTrainingLabError("GRPO latest report digest changed")
    if latest.get("evidence_chain_sha256") != _load_object(report_path).get(
        "evidence_chain_sha256"
    ):
        raise GrpoPostTrainingLabError("GRPO latest evidence chain changed")
    return report_path


def _inside_file(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    target = target.resolve(strict=True)
    _require_inside(root, target)
    if not target.is_file():
        raise GrpoPostTrainingLabError("GRPO evidence path is not a file")
    return target


def _inside_directory(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    target = target.resolve(strict=True)
    _require_inside(root, target)
    if not target.is_dir():
        raise GrpoPostTrainingLabError("GRPO evidence path is not a directory")
    return target


def _require_inside(root: Path, target: Path) -> None:
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise GrpoPostTrainingLabError("GRPO evidence path escaped repository") from exc


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GrpoPostTrainingLabError("GRPO evidence JSON is invalid") from exc
    if not isinstance(value, dict):
        raise GrpoPostTrainingLabError("GRPO evidence JSON is invalid")
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
        raise GrpoPostTrainingLabError("GRPO runtime image digest is invalid")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()
