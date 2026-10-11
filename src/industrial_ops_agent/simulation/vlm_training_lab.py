"""Real single-GPU VLM LoRA/QLoRA training on the governed M7 simulation snapshot."""

from __future__ import annotations

import gc
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.data_pipeline.contracts import (
    production_release_blocker_for_dataset_contract,
)
from industrial_ops_agent.data_pipeline.storage import FilesystemDatasetStore
from industrial_ops_agent.evaluation.backend import (
    EvaluationRuntimeConfig,
    VlmComponentEvaluationBackend,
)
from industrial_ops_agent.evaluation.dataset import (
    FrozenEvaluationDatasetLoader,
    NormalizedRegion,
)
from industrial_ops_agent.evaluation.metrics import (
    ModelObservation,
    PairedMetricReport,
    VlmObservationFinding,
    score_vlm_paired_observations,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    DatasetSnapshotRecord,
    EvaluationSuiteRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.simulation.multimodal_platform_lab import (
    TENANT_ID,
    verify_simulated_multimodal_platform_lab,
)
from industrial_ops_agent.simulation.multimodal_snapshot import SIMULATION_TASK_TYPE
from industrial_ops_agent.training.artifacts import store_training_bundle
from industrial_ops_agent.training.backend import HuggingFaceTrainingBackend, TrainingOutcome
from industrial_ops_agent.training.config import CompiledVlmConfig, compile_training_config
from industrial_ops_agent.training.dataset import GovernedSftDatasetLoader
from industrial_ops_agent.training.model_contract import (
    expected_model_digest,
    tokenizer_contract_digests,
)

SCHEMA_VERSION = "m7-simulated-vlm-training-lab/v1"
CLASSIFICATION = "SIMULATED_NON_PRODUCTION"
STATUS = "SIMULATED_VLM_TRAINING_EVALUATED"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
COMPARISON_GROUP_ID = "m7-qwen2-vl-simulated-lora-qlora-v1"
PROFILES = ("LORA", "QLORA")
REGION_IOU_MIN = 0.10
HALLUCINATION_RATE_MAX = 0.0


class SimulatedVlmTrainingLabError(RuntimeError):
    """The local VLM experiment is incomplete, inconsistent, or unsafe."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileEvidence(_ClosedModel):
    path: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrainingEvidence(_ClosedModel):
    profile: Literal["LORA", "QLORA"]
    experiment_id: str
    load_in_4bit: bool
    optimizer_steps: int = Field(gt=0)
    train_loss: float
    validation_loss: float
    trainable_parameters: int = Field(gt=0)
    total_parameters: int = Field(gt=0)
    quantized_modules: int = Field(ge=0)
    runtime_seconds: float = Field(gt=0)
    peak_gpu_memory_allocated_bytes: int = Field(gt=0)
    adapter_bundle_object_key: str
    adapter_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_bundle_size_bytes: int = Field(gt=0)
    compiled_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluationEvidence(_ClosedModel):
    profile: Literal["LORA", "QLORA"]
    baseline_accuracy: float = Field(ge=0, le=1)
    candidate_accuracy: float = Field(ge=0, le=1)
    accuracy_improvement: float
    baseline_mean_region_iou: float = Field(ge=0, le=1)
    candidate_mean_region_iou: float = Field(ge=0, le=1)
    baseline_hallucination_rate: float = Field(ge=0, le=1)
    candidate_hallucination_rate: float = Field(ge=0, le=1)
    hard_gates: dict[str, bool]
    decision: Literal["SIMULATION_CANDIDATE_ELIGIBLE", "SIMULATION_NO_GAIN", "REJECTED"]


class SimulatedVlmTrainingReport(_ClosedModel):
    schema_version: Literal["m7-simulated-vlm-training-lab/v1"] = (
        "m7-simulated-vlm-training-lab/v1"
    )
    classification: Literal["SIMULATED_NON_PRODUCTION"] = "SIMULATED_NON_PRODUCTION"
    production_claim: Literal[False] = False
    enterprise_production_data: Literal[False] = False
    production_release_eligible: Literal[False] = False
    formal_release_created: Literal[False] = False
    status: Literal["SIMULATED_VLM_TRAINING_EVALUATED"] = (
        "SIMULATED_VLM_TRAINING_EVALUATED"
    )
    decision: Literal[
        "SIMULATION_CANDIDATE_ELIGIBLE",
        "SIMULATION_NO_CANDIDATE_GAIN",
    ]
    selected_profile: Literal["LORA", "QLORA"] | None
    generated_at: str
    model_id: Literal["Qwen/Qwen2-VL-2B-Instruct"] = "Qwen/Qwen2-VL-2B-Instruct"
    model_revision: Literal["895c3a49bc3fa70a340399125c650a463535e71c"] = (
        "895c3a49bc3fa70a340399125c650a463535e71c"
    )
    model_license: Literal["Apache-2.0"] = "Apache-2.0"
    comparison_group_id: str
    source_platform_acceptance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_snapshot_id: str
    training_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_snapshot_id: str
    evaluation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_reference_suite_id: str
    train_cases: int = Field(gt=0)
    validation_cases: int = Field(gt=0)
    evaluation_cases: int = Field(gt=0)
    gpu_name: str
    gpu_total_memory_bytes: int = Field(gt=0)
    runtime_versions: dict[str, str]
    tracking_backend: Literal["LOCAL_IMMUTABLE_JSON"] = "LOCAL_IMMUTABLE_JSON"
    mlflow_service_called: Literal[False] = False
    trainings: tuple[TrainingEvidence, TrainingEvidence]
    evaluations: tuple[EvaluationEvidence, EvaluationEvidence]
    evidence_files: tuple[FileEvidence, ...]
    release_gate: Literal["DENIED_SIMULATION_CONTRACT"] = "DENIED_SIMULATION_CONTRACT"
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_simulated_vlm_training_lab(
    repo_root: Path,
    *,
    platform_dir: Path,
    output_dir: Path,
    max_steps: int = 12,
) -> tuple[SimulatedVlmTrainingReport, bool]:
    root = repo_root.resolve(strict=True)
    platform = _inside(root, platform_dir)
    output = output_dir if output_dir.is_absolute() else root / output_dir
    output = output.resolve()
    if output.exists():
        return verify_simulated_vlm_training_lab(output), False
    if not 1 <= max_steps <= 200:
        raise SimulatedVlmTrainingLabError("vlm_training_max_steps_is_out_of_range")
    verify_simulated_multimodal_platform_lab(platform)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        report = _build_into(platform=platform, output=temporary, max_steps=max_steps)
        _write_new(temporary / "acceptance.json", _json_bytes(report.model_dump(mode="json")))
        verify_simulated_vlm_training_lab(temporary)
        temporary.rename(output)
    except Exception:
        _safe_remove_temporary(temporary, output.parent)
        raise
    return verify_simulated_vlm_training_lab(output), True


def rescore_simulated_vlm_training_lab(
    source_dir: Path,
    *,
    output_dir: Path,
) -> tuple[SimulatedVlmTrainingReport, bool]:
    """Recompute metric evidence from frozen observations without retraining."""

    source = source_dir.resolve(strict=True)
    source_report = verify_simulated_vlm_training_lab(source)
    output = output_dir.resolve()
    if output.exists():
        return verify_simulated_vlm_training_lab(output), False
    if output == source or output.is_relative_to(source):
        raise SimulatedVlmTrainingLabError("vlm_rescore_output_path_is_invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        evaluation_dir = temporary / "evaluation"
        shutil.copy2(
            temporary / "acceptance.json",
            evaluation_dir / "source-training-acceptance.json",
        )
        for profile in PROFILES:
            paired_path = evaluation_dir / f"{profile.lower()}-paired-metrics.json"
            shutil.copy2(
                paired_path,
                evaluation_dir / f"original-{profile.lower()}-paired-metrics.json",
            )
        (temporary / "acceptance.json").unlink()
        database = Database(f"sqlite+pysqlite:///{temporary / 'catalog.sqlite3'}")
        store = FilesystemDatasetStore(temporary / "object-store")
        context = TenantContext(TENANT_ID, "m7-vlm-rescore-worker")
        try:
            suite = _load_suite(
                database,
                context,
                source_report.simulation_reference_suite_id,
            )
            evaluation = FrozenEvaluationDatasetLoader(database, store).load(context, suite)
        finally:
            database.dispose()
        baseline = _load_observations(evaluation_dir / "baseline.json")
        rescored: list[EvaluationEvidence] = []
        for profile in PROFILES:
            candidate = _load_observations(evaluation_dir / f"{profile.lower()}.json")
            paired = score_vlm_paired_observations(
                evaluation.cases,
                candidate,
                baseline,
                region_iou_min=REGION_IOU_MIN,
                hallucination_rate_max=HALLUCINATION_RATE_MAX,
            )
            rescored.append(_evaluation_evidence(profile, paired))
            _replace_file(
                evaluation_dir / f"{profile.lower()}-paired-metrics.json",
                _json_bytes(_paired_report(paired)),
            )
        eligible = [
            item for item in rescored if item.decision == "SIMULATION_CANDIDATE_ELIGIBLE"
        ]
        selected = max(
            eligible,
            key=lambda item: (item.candidate_accuracy, item.candidate_mean_region_iou),
            default=None,
        )
        report = source_report.model_copy(
            update={
                "decision": (
                    "SIMULATION_CANDIDATE_ELIGIBLE"
                    if selected is not None
                    else "SIMULATION_NO_CANDIDATE_GAIN"
                ),
                "selected_profile": selected.profile if selected is not None else None,
                "generated_at": datetime.now(UTC).isoformat(),
                "evaluations": tuple(rescored),
                "evidence_files": tuple(_evidence_files(temporary)),
                "evidence_chain_sha256": "0" * 64,
            }
        )
        report = report.model_copy(update={"evidence_chain_sha256": _report_hash(report)})
        _write_new(temporary / "acceptance.json", _json_bytes(report.model_dump(mode="json")))
        verify_simulated_vlm_training_lab(temporary)
        temporary.rename(output)
    except Exception:
        _safe_remove_temporary(temporary, output.parent)
        raise
    return verify_simulated_vlm_training_lab(output), True


def verify_simulated_vlm_training_lab(output_dir: Path) -> SimulatedVlmTrainingReport:
    output = output_dir.resolve(strict=True)
    try:
        report = SimulatedVlmTrainingReport.model_validate_json(
            (output / "acceptance.json").read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise SimulatedVlmTrainingLabError("vlm_training_acceptance_is_invalid") from exc
    if report.evidence_chain_sha256 != _report_hash(report):
        raise SimulatedVlmTrainingLabError("vlm_training_acceptance_digest_mismatch")
    for evidence in report.evidence_files:
        path = (output / evidence.path).resolve(strict=True)
        if not path.is_relative_to(output) or not path.is_file():
            raise SimulatedVlmTrainingLabError("vlm_training_evidence_path_is_invalid")
        content = path.read_bytes()
        if len(content) != evidence.size_bytes or sha256(content).hexdigest() != evidence.sha256:
            raise SimulatedVlmTrainingLabError("vlm_training_evidence_integrity_failed")
    source_acceptance = output / "source-platform-acceptance.json"
    if (
        sha256(source_acceptance.read_bytes()).hexdigest()
        != report.source_platform_acceptance_sha256
    ):
        raise SimulatedVlmTrainingLabError("vlm_training_source_acceptance_changed")
    database = Database(f"sqlite+pysqlite:///{output / 'catalog.sqlite3'}")
    store = FilesystemDatasetStore(output / "object-store")
    context = TenantContext(TENANT_ID, "m7-vlm-training-verifier")
    try:
        with database.transaction(context) as session:
            suite = session.get(EvaluationSuiteRecord, report.simulation_reference_suite_id)
            training_snapshot = session.get(
                DatasetSnapshotRecord,
                report.training_snapshot_id,
            )
            experiments = {
                profile.profile: session.get(TrainingExperimentRecord, profile.experiment_id)
                for profile in report.trainings
            }
            artifacts = {
                profile.profile: session.get(
                    TrainingArtifactRecord,
                    _artifact_id(profile.profile, profile.adapter_bundle_sha256),
                )
                for profile in report.trainings
            }
        if suite is None or training_snapshot is None:
            raise SimulatedVlmTrainingLabError("vlm_training_catalog_binding_is_missing")
        if (
            suite.tier != "SIMULATION_REFERENCE"
            or suite.source_snapshot_id != report.evaluation_snapshot_id
            or suite.manifest_hash != report.evaluation_manifest_sha256
            or production_release_blocker_for_dataset_contract(training_snapshot.contract_version)
            != "simulated_multimodal_snapshot_is_not_production_releasable"
        ):
            raise SimulatedVlmTrainingLabError("vlm_training_dataset_binding_changed")
        evaluation = FrozenEvaluationDatasetLoader(database, store).load(context, suite)
        if len(evaluation.cases) != report.evaluation_cases:
            raise SimulatedVlmTrainingLabError("vlm_training_evaluation_case_count_changed")
        for profile in report.trainings:
            experiment = experiments[profile.profile]
            artifact = artifacts[profile.profile]
            if experiment is None or artifact is None:
                raise SimulatedVlmTrainingLabError("vlm_training_catalog_artifact_is_missing")
            training = GovernedSftDatasetLoader(database, store).load(context, experiment)
            content = store.get_bytes(artifact.object_key)
            if (
                experiment.status != "COMPLETED"
                or experiment.method != "VLM"
                or experiment.task_type != SIMULATION_TASK_TYPE
                or experiment.dataset_snapshot_id != report.training_snapshot_id
                or len(training.train) != report.train_cases
                or len(training.validation) != report.validation_cases
                or artifact.content_hash != profile.adapter_bundle_sha256
                or len(content) != artifact.size_bytes
                or sha256(content).hexdigest() != artifact.content_hash
            ):
                raise SimulatedVlmTrainingLabError("vlm_training_catalog_artifact_changed")
        if report.production_release_eligible or report.formal_release_created:
            raise SimulatedVlmTrainingLabError("vlm_training_production_boundary_changed")
    finally:
        database.dispose()
    return report


def preflight_simulated_vlm_training_lab() -> dict[str, Any]:
    try:
        import bitsandbytes
        import peft
        import torch
        import transformers
        import trl
        from transformers import AutoConfig
    except ImportError as exc:
        raise SimulatedVlmTrainingLabError("vlm_training_dependencies_are_not_installed") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SimulatedVlmTrainingLabError("exactly_one_cuda_gpu_is_required")
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    return {
        "status": "M7_VLM_TRAINING_PREFLIGHT_PASSED",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_type": str(getattr(config, "model_type", "")),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "trl": trl.__version__,
        },
    }


def _build_into(*, platform: Path, output: Path, max_steps: int) -> SimulatedVlmTrainingReport:
    platform_report = verify_simulated_multimodal_platform_lab(platform)
    shutil.copy2(platform / "catalog.sqlite3", output / "catalog.sqlite3")
    shutil.copytree(platform / "object-store", output / "object-store")
    source_acceptance = (platform / "acceptance.json").read_bytes()
    _write_new(output / "source-platform-acceptance.json", source_acceptance)
    database = Database(f"sqlite+pysqlite:///{output / 'catalog.sqlite3'}")
    store = FilesystemDatasetStore(output / "object-store")
    context = TenantContext(TENANT_ID, "m7-vlm-training-worker")
    try:
        suite = _load_suite(database, context, platform_report.simulation_reference_suite_id)
        evaluation = FrozenEvaluationDatasetLoader(database, store).load(context, suite)
        tokenizer_digest, chat_template_digest = _model_contract()
        git_commit = _git_commit()
        runtime_versions = _runtime_versions()
        runtime_digest = "sha256:" + sha256(_json_text(runtime_versions).encode()).hexdigest()
        experiments = {
            profile: _experiment(
                profile=profile,
                snapshot_id=platform_report.training_snapshot.snapshot_id,
                manifest_hash=platform_report.training_snapshot.manifest_hash,
                tokenizer_digest=tokenizer_digest,
                chat_template_digest=chat_template_digest,
                git_commit=git_commit,
                runtime_digest=runtime_digest,
                max_steps=max_steps,
            )
            for profile in PROFILES
        }
        baseline = VlmComponentEvaluationBackend().evaluate(
            experiment=experiments["LORA"],
            cases=evaluation.cases,
            config=_evaluation_runtime(),
            adapter_directory=None,
        )
        _write_new(output / "evaluation/baseline.json", _json_bytes(_observations(baseline)))
        _release_gpu()

        training_evidence: list[TrainingEvidence] = []
        evaluation_evidence: list[EvaluationEvidence] = []
        for profile in PROFILES:
            experiment = experiments[profile]
            _persist_experiment(database, context, experiment)
            dataset = GovernedSftDatasetLoader(database, store).load(context, experiment)
            compiled = compile_training_config(experiment)
            if not isinstance(compiled, CompiledVlmConfig):
                raise SimulatedVlmTrainingLabError("vlm_training_config_did_not_compile")
            adapter_directory = output / "adapters" / profile.lower()
            outcome = HuggingFaceTrainingBackend().train(
                experiment=experiment,
                config=compiled,
                dataset=dataset,
                output_directory=adapter_directory,
                resume_from_checkpoint=None,
            )
            stored = store_training_bundle(
                store,
                tenant_id=TENANT_ID,
                experiment_id=experiment.experiment_id,
                directory=adapter_directory,
                kind="vlm_adapter_bundle",
                name="vlm-adapter",
                serialization="safetensors",
                metadata={
                    "classification": CLASSIFICATION,
                    "profile": profile,
                    "base_model_id": MODEL_ID,
                    "base_model_revision": MODEL_REVISION,
                    "dataset_manifest_hash": dataset.manifest_hash,
                    "compiled_config_digest": compiled.digest,
                    "production_release_eligible": False,
                },
            )
            evidence = _training_evidence(profile, experiment, compiled, outcome, stored.descriptor)
            _complete_experiment(
                database, context, experiment, evidence, outcome, stored.descriptor
            )
            training_evidence.append(evidence)
            _release_gpu()
            candidate = VlmComponentEvaluationBackend().evaluate(
                experiment=experiment,
                cases=evaluation.cases,
                config=_evaluation_runtime(),
                adapter_directory=adapter_directory,
            )
            _write_new(
                output / f"evaluation/{profile.lower()}.json",
                _json_bytes(_observations(candidate)),
            )
            paired = score_vlm_paired_observations(
                evaluation.cases,
                candidate,
                baseline,
                region_iou_min=REGION_IOU_MIN,
                hallucination_rate_max=HALLUCINATION_RATE_MAX,
            )
            evaluation_evidence.append(_evaluation_evidence(profile, paired))
            _write_new(
                output / f"evaluation/{profile.lower()}-paired-metrics.json",
                _json_bytes(_paired_report(paired)),
            )
            _release_gpu()
    finally:
        database.dispose()

    eligible = [
        item for item in evaluation_evidence if item.decision == "SIMULATION_CANDIDATE_ELIGIBLE"
    ]
    selected = max(
        eligible,
        key=lambda item: (item.candidate_accuracy, item.candidate_mean_region_iou),
        default=None,
    )
    evidence_files = _evidence_files(output)
    import torch

    report = SimulatedVlmTrainingReport(
        decision=(
            "SIMULATION_CANDIDATE_ELIGIBLE"
            if selected is not None
            else "SIMULATION_NO_CANDIDATE_GAIN"
        ),
        selected_profile=selected.profile if selected is not None else None,
        generated_at=datetime.now(UTC).isoformat(),
        comparison_group_id=COMPARISON_GROUP_ID,
        source_platform_acceptance_sha256=sha256(source_acceptance).hexdigest(),
        training_snapshot_id=platform_report.training_snapshot.snapshot_id,
        training_manifest_sha256=platform_report.training_snapshot.manifest_hash,
        evaluation_snapshot_id=platform_report.evaluation_snapshot.snapshot_id,
        evaluation_manifest_sha256=platform_report.evaluation_snapshot.manifest_hash,
        simulation_reference_suite_id=platform_report.simulation_reference_suite_id,
        train_cases=platform_report.training_records,
        validation_cases=platform_report.validation_records,
        evaluation_cases=platform_report.simulation_reference_cases,
        gpu_name=torch.cuda.get_device_name(0),
        gpu_total_memory_bytes=int(torch.cuda.get_device_properties(0).total_memory),
        runtime_versions=runtime_versions,
        trainings=tuple(training_evidence),
        evaluations=tuple(evaluation_evidence),
        evidence_files=tuple(evidence_files),
        evidence_chain_sha256="0" * 64,
    )
    return report.model_copy(update={"evidence_chain_sha256": _report_hash(report)})


def _model_contract() -> tuple[str, str]:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise SimulatedVlmTrainingLabError("vlm_training_dependencies_are_not_installed") from exc
    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise SimulatedVlmTrainingLabError("vlm_processor_has_no_tokenizer")
    return tokenizer_contract_digests(tokenizer)


def _experiment(
    *,
    profile: str,
    snapshot_id: str,
    manifest_hash: str,
    tokenizer_digest: str,
    chat_template_digest: str,
    git_commit: str,
    runtime_digest: str,
    max_steps: int,
) -> TrainingExperimentRecord:
    now = datetime.now(UTC)
    config: dict[str, Any] = {
        "base_model_revision": MODEL_REVISION,
        "vision_contract_version": "industrial-vision-instruction-v1",
        "max_image_pixels": 307_200,
        "load_in_4bit": profile == "QLORA",
        "max_steps": max_steps,
        "effective_batch_size": 2,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "evaluation_interval": max_steps,
        "save_steps": max_steps,
        "logging_steps": 1,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "lr_scheduler_type": "constant",
        "max_sequence_length": 512,
        "gradient_checkpointing": True,
        "precision": "bfloat16",
        "data_seed": 20260823,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_bias": "none",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "gpu_hourly_cost_usd": 0.50,
        "simulation_only": True,
        "comparison_profile": profile,
    }
    experiment_id = f"experiment-m7-vlm-{profile.lower()}-{MODEL_REVISION[:12]}"
    return TrainingExperimentRecord(
        experiment_id=experiment_id,
        tenant_id=TENANT_ID,
        idempotency_key=experiment_id,
        comparison_group_id=COMPARISON_GROUP_ID,
        method="VLM",
        task_type=SIMULATION_TASK_TYPE,
        status="RUNNING",
        dataset_snapshot_id=snapshot_id,
        dataset_manifest_hash=manifest_hash,
        base_model_id=MODEL_ID,
        base_model_digest=expected_model_digest(MODEL_REVISION),
        tokenizer_digest=tokenizer_digest,
        chat_template_digest=chat_template_digest,
        git_commit=git_commit,
        container_digest=runtime_digest,
        training_config=config,
        config_hash="sha256:" + sha256(_json_text(config).encode()).hexdigest(),
        distributed_profile={"strategy": "single_gpu", "world_size": 1, "node_count": 1},
        random_seeds=[20260823],
        hardware_topology={"accelerator": "NVIDIA_GPU", "count": 1, "profile": "LOCAL_WSL"},
        mlflow_experiment_name="m7-simulated-vlm-lora-qlora",
        mlflow_run_id=None,
        metrics={},
        cost_summary={},
        license_status="SIMULATION_ONLY",
        created_by_subject_id="simulation-model-engineer",
        completed_at=None,
        failure_reason=None,
        version=1,
        created_at=now,
        updated_at=now,
    )


def _persist_experiment(
    database: Database,
    context: TenantContext,
    experiment: TrainingExperimentRecord,
) -> None:
    with database.transaction(context) as session:
        session.add(experiment)
        session.flush()


def _complete_experiment(
    database: Database,
    context: TenantContext,
    experiment: TrainingExperimentRecord,
    evidence: TrainingEvidence,
    outcome: TrainingOutcome,
    descriptor: Any,
) -> None:
    with database.transaction(context) as session:
        stored = session.get(TrainingExperimentRecord, experiment.experiment_id)
        if stored is None or stored.status != "RUNNING":
            raise SimulatedVlmTrainingLabError("vlm_training_experiment_state_changed")
        stored.status = "COMPLETED"
        stored.metrics = outcome.metrics
        stored.cost_summary = outcome.cost_summary
        stored.completed_at = datetime.now(UTC)
        stored.updated_at = stored.completed_at
        stored.version += 1
        session.add(
            TrainingArtifactRecord(
                artifact_id=_artifact_id(evidence.profile, descriptor.content_hash),
                tenant_id=TENANT_ID,
                experiment_id=experiment.experiment_id,
                kind=descriptor.kind,
                object_key=descriptor.object_key,
                content_hash=descriptor.content_hash,
                size_bytes=descriptor.size_bytes,
                metadata_json=dict(descriptor.metadata),
                created_at=stored.completed_at,
                updated_at=stored.completed_at,
            )
        )


def _load_suite(
    database: Database,
    context: TenantContext,
    suite_id: str,
) -> EvaluationSuiteRecord:
    with database.transaction(context) as session:
        suite = session.get(EvaluationSuiteRecord, suite_id)
    if suite is None or suite.tier != "SIMULATION_REFERENCE":
        raise SimulatedVlmTrainingLabError("simulation_reference_suite_is_missing")
    return suite


def _training_evidence(
    profile: str,
    experiment: TrainingExperimentRecord,
    compiled: CompiledVlmConfig,
    outcome: TrainingOutcome,
    descriptor: Any,
) -> TrainingEvidence:
    metrics = outcome.metrics
    optimizer_steps = int(metrics.get("optimizer_steps", 0))
    quantized_modules = int(metrics.get("quantized_modules", 0))
    if optimizer_steps != compiled.max_steps:
        raise SimulatedVlmTrainingLabError("vlm_training_optimizer_steps_are_incomplete")
    if profile == "QLORA" and quantized_modules <= 0:
        raise SimulatedVlmTrainingLabError("qlora_nf4_modules_are_missing")
    if profile == "LORA" and quantized_modules != 0:
        raise SimulatedVlmTrainingLabError("lora_base_was_unexpectedly_quantized")
    return TrainingEvidence(
        profile=profile,
        experiment_id=experiment.experiment_id,
        load_in_4bit=profile == "QLORA",
        optimizer_steps=optimizer_steps,
        train_loss=float(metrics.get("train_loss", metrics.get("loss", 0.0))),
        validation_loss=float(metrics.get("eval_loss", 0.0)),
        trainable_parameters=int(metrics.get("trainable_parameters", 0)),
        total_parameters=int(metrics.get("total_parameters", 0)),
        quantized_modules=quantized_modules,
        runtime_seconds=float(metrics.get("train_runtime_seconds", 0.0)),
        peak_gpu_memory_allocated_bytes=int(metrics.get("peak_gpu_memory_allocated_bytes", 0)),
        adapter_bundle_object_key=descriptor.object_key,
        adapter_bundle_sha256=descriptor.content_hash,
        adapter_bundle_size_bytes=descriptor.size_bytes,
        compiled_config_sha256=compiled.digest,
    )


def _evaluation_evidence(profile: str, paired: PairedMetricReport) -> EvaluationEvidence:
    aggregate = paired.aggregate_metrics.get("vlm")
    if not isinstance(aggregate, dict):
        raise SimulatedVlmTrainingLabError("vlm_evaluation_aggregate_is_missing")
    accuracy = aggregate["diagnostic_accuracy"]
    region = aggregate["mean_region_iou"]
    hallucination = aggregate["hallucination_rate"]
    gates = dict(paired.evidence.hard_gate_results)
    baseline_accuracy = float(accuracy["baseline"])
    candidate_accuracy = float(accuracy["candidate"])
    improvement = candidate_accuracy - baseline_accuracy
    if all(gates.values()) and improvement > 0:
        decision = "SIMULATION_CANDIDATE_ELIGIBLE"
    elif all(gates.values()):
        decision = "SIMULATION_NO_GAIN"
    else:
        decision = "REJECTED"
    return EvaluationEvidence(
        profile=profile,
        baseline_accuracy=baseline_accuracy,
        candidate_accuracy=candidate_accuracy,
        accuracy_improvement=improvement,
        baseline_mean_region_iou=float(region["baseline"]),
        candidate_mean_region_iou=float(region["candidate"]),
        baseline_hallucination_rate=float(hallucination["baseline"]),
        candidate_hallucination_rate=float(hallucination["candidate"]),
        hard_gates=gates,
        decision=decision,
    )


def _evaluation_runtime() -> EvaluationRuntimeConfig:
    return EvaluationRuntimeConfig(
        max_new_tokens=128,
        precision="bfloat16",
        gpu_hourly_cost_usd=0.50,
        max_image_pixels=307_200,
    )


def _observations(observations: tuple[ModelObservation, ...]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": item.case_id,
            "output_text": item.output_text,
            "latency_ms": item.latency_ms,
            "cost_usd": item.cost_usd,
            "input_tokens": item.input_tokens,
            "output_tokens": item.output_tokens,
            "capabilities": sorted(item.capabilities),
            "structured_output_valid": item.structured_output_valid,
            "visual_findings": [
                {
                    "label": finding.label,
                    "region": {
                        "x": finding.region.x,
                        "y": finding.region.y,
                        "width": finding.region.width,
                        "height": finding.region.height,
                    },
                }
                for finding in item.visual_findings
            ],
            "runtime_evidence": item.runtime_evidence,
        }
        for item in observations
    ]


def _load_observations(path: Path) -> tuple[ModelObservation, ...]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulatedVlmTrainingLabError("vlm_observation_evidence_is_invalid") from exc
    if not isinstance(payload, list) or not payload:
        raise SimulatedVlmTrainingLabError("vlm_observation_evidence_is_invalid")
    observations: list[ModelObservation] = []
    try:
        for item in payload:
            if not isinstance(item, dict):
                raise TypeError
            findings = tuple(
                VlmObservationFinding(
                    label=str(finding["label"]),
                    region=NormalizedRegion(
                        x=float(finding["region"]["x"]),
                        y=float(finding["region"]["y"]),
                        width=float(finding["region"]["width"]),
                        height=float(finding["region"]["height"]),
                    ),
                )
                for finding in item["visual_findings"]
            )
            observations.append(
                ModelObservation(
                    case_id=str(item["case_id"]),
                    output_text=str(item["output_text"]),
                    latency_ms=float(item["latency_ms"]),
                    cost_usd=float(item["cost_usd"]),
                    input_tokens=int(item["input_tokens"]),
                    output_tokens=int(item["output_tokens"]),
                    capabilities=frozenset(str(value) for value in item["capabilities"]),
                    visual_findings=findings,
                    structured_output_valid=bool(item["structured_output_valid"]),
                    runtime_evidence=dict(item["runtime_evidence"]),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise SimulatedVlmTrainingLabError("vlm_observation_evidence_is_invalid") from exc
    return tuple(observations)


def _paired_report(report: PairedMetricReport) -> dict[str, Any]:
    return {
        "aggregate_metrics": report.aggregate_metrics,
        "gate_coverage": report.gate_coverage,
        "hard_gate_results": report.evidence.hard_gate_results,
        "candidate_outcomes": report.evidence.candidate_outcomes,
        "baseline_outcomes": report.evidence.baseline_outcomes,
        "candidate_p95_latency_ms": report.evidence.candidate_p95_latency_ms,
        "baseline_p95_latency_ms": report.evidence.baseline_p95_latency_ms,
        "candidate_cost_per_case": report.evidence.candidate_cost_per_case,
        "baseline_cost_per_case": report.evidence.baseline_cost_per_case,
    }


def _runtime_versions() -> dict[str, str]:
    return {
        name: version(name)
        for name in (
            "torch",
            "transformers",
            "peft",
            "bitsandbytes",
            "accelerate",
            "datasets",
            "trl",
            "safetensors",
        )
    }


def _git_commit() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SimulatedVlmTrainingLabError("git_commit_could_not_be_resolved") from exc
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise SimulatedVlmTrainingLabError("git_commit_is_invalid")
    return value


def _evidence_files(output: Path) -> list[FileEvidence]:
    roots = [output / "source-platform-acceptance.json", output / "adapters", output / "evaluation"]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
        else:
            raise SimulatedVlmTrainingLabError("vlm_training_evidence_is_missing")
    evidence: list[FileEvidence] = []
    for path in sorted(files):
        if path.is_symlink():
            raise SimulatedVlmTrainingLabError("vlm_training_evidence_contains_symlink")
        content = path.read_bytes()
        evidence.append(
            FileEvidence(
                path=path.relative_to(output).as_posix(),
                size_bytes=len(content),
                sha256=sha256(content).hexdigest(),
            )
        )
    return evidence


def _artifact_id(profile: str, digest: str) -> str:
    return f"artifact-m7-vlm-{profile.lower()}-{digest[:24]}"


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.exists():
        raise SimulatedVlmTrainingLabError("vlm_training_path_is_outside_repo")
    return resolved


def _report_hash(report: SimulatedVlmTrainingReport) -> str:
    document = report.model_dump(mode="json")
    document.pop("evidence_chain_sha256", None)
    return sha256(_json_text(document).encode()).hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(content)


def _replace_file(path: Path, content: bytes) -> None:
    if not path.is_file() or path.is_symlink():
        raise SimulatedVlmTrainingLabError("vlm_rescore_evidence_path_is_invalid")
    path.unlink()
    _write_new(path, content)


def _safe_remove_temporary(temporary: Path, parent: Path) -> None:
    resolved = temporary.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith("."):
        raise SimulatedVlmTrainingLabError("temporary_vlm_training_cleanup_target_is_unsafe")
    shutil.rmtree(resolved, ignore_errors=True)
