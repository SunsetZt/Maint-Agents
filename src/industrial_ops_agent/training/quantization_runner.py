"""Governed execution from a completed PEFT experiment to a quantized candidate artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.evaluation.artifacts import extract_adapter_bundle, verify_artifact_bytes
from industrial_ops_agent.experiments.service import ExperimentRegistryService
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.training.artifacts import store_training_bundle
from industrial_ops_agent.training.dataset import GovernedSftDatasetLoader
from industrial_ops_agent.training.image_acceptance import (
    gpu_acceptance_artifact_metadata,
    verify_gpu_acceptance_report,
    write_gpu_acceptance_report,
)
from industrial_ops_agent.training.quantization import (
    QuantizationBackend,
    compile_quantization_config,
)
from industrial_ops_agent.training.runner import RuntimeFingerprint


class QuantizationExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class QuantizationRunResult:
    experiment_id: str
    status: str
    dry_run: bool
    quantization_profile_id: str
    source_experiment_id: str
    source_artifact_id: str
    calibration_manifest_hash: str
    artifact_object_key: str | None
    metrics: dict[str, float]


class QuantizationRunner:
    def __init__(
        self,
        database: Database,
        store: DatasetStore,
        registry: ExperimentRegistryService,
        backend: QuantizationBackend,
    ) -> None:
        self._database = database
        self._store = store
        self._registry = registry
        self._backend = backend
        self._dataset_loader = GovernedSftDatasetLoader(database, store)

    def run(
        self,
        identity: IdentityContext,
        experiment_id: str,
        *,
        runtime: RuntimeFingerprint,
        dry_run: bool,
        request_id: str,
    ) -> QuantizationRunResult:
        experiment, source, source_artifact = self._load(identity, experiment_id)
        if experiment.method != "QUANTIZATION":
            raise QuantizationExecutionError("experiment_is_not_a_quantization_run")
        if dry_run:
            if experiment.status not in {"PLANNED", "RUNNING"}:
                raise QuantizationExecutionError(
                    "dry_run_requires_planned_or_running_quantization_experiment"
                )
        elif experiment.status != "RUNNING":
            raise QuantizationExecutionError("quantization_requires_running_experiment")
        try:
            config = compile_quantization_config(experiment.training_config)
            _verify_runtime(experiment, config.algorithm, runtime, dry_run=dry_run)
            _verify_source_binding(experiment, source, source_artifact, config.source_artifact_hash)
            source_content = self._store.get_bytes(source_artifact.object_key)
            verify_artifact_bytes(
                source_content,
                content_hash=source_artifact.content_hash,
                size_bytes=source_artifact.size_bytes,
            )
            dataset = self._dataset_loader.load(identity.tenant_context, experiment)
            if config.calibration_sample_count > len(dataset.train):
                raise QuantizationExecutionError(
                    "calibration_snapshot_has_insufficient_samples"
                )
            if dry_run:
                return QuantizationRunResult(
                    experiment_id=experiment.experiment_id,
                    status="VALIDATED",
                    dry_run=True,
                    quantization_profile_id=config.quantization_profile_id,
                    source_experiment_id=source.experiment_id,
                    source_artifact_id=source_artifact.artifact_id,
                    calibration_manifest_hash=dataset.manifest_hash,
                    artifact_object_key=None,
                    metrics={},
                )
            with TemporaryDirectory(prefix="industrial-ops-quantization-") as temporary:
                root = Path(temporary)
                source_directory = root / "source-adapter"
                extract_adapter_bundle(source_content, source_directory)
                outcome = self._backend.quantize(
                    experiment=experiment,
                    config=config,
                    dataset=dataset,
                    source_adapter_directory=source_directory,
                    output_directory=root / "quantized-model",
                )
                if runtime.gpu_acceptance_report is not None:
                    write_gpu_acceptance_report(
                        outcome.output_directory / "industrial-ops-gpu-image-acceptance.json",
                        runtime.gpu_acceptance_report,
                    )
                stored = store_training_bundle(
                    self._store,
                    tenant_id=identity.tenant_id,
                    experiment_id=experiment.experiment_id,
                    directory=outcome.output_directory,
                    kind="quantized_model_bundle",
                    name="quantized-model",
                    serialization=(
                        "gguf" if config.algorithm == "GGUF" else "compressed-tensors"
                    ),
                    metadata={
                        "quantization_profile_id": config.quantization_profile_id,
                        "algorithm": config.algorithm,
                        "scheme": config.scheme,
                        "target_runtime": config.target_runtime,
                        "target_hardware_profile": config.target_hardware_profile,
                        "tool_version": config.tool_version,
                        "source_experiment_id": source.experiment_id,
                        "source_artifact_id": source_artifact.artifact_id,
                        "source_artifact_hash": source_artifact.content_hash,
                        "calibration_snapshot_id": dataset.snapshot_id,
                        "calibration_manifest_hash": dataset.manifest_hash,
                        "calibration_sample_count": config.calibration_sample_count,
                        "max_sequence_length": config.max_sequence_length,
                        "incompatibilities": list(config.incompatibilities),
                        "approval_inheritance": False,
                        "quantization_manifest_sha256": outcome.manifest[
                            "manifest_sha256"
                        ],
                        "runtime": outcome.runtime_metadata,
                        "gpu_image_acceptance": (
                            gpu_acceptance_artifact_metadata(runtime.gpu_acceptance_report)
                            if runtime.gpu_acceptance_report is not None
                            else {"required": False, "reason": "CPU_EDGE_PROFILE"}
                        ),
                    },
                )
            completed = self._registry.complete(
                identity,
                experiment.experiment_id,
                expected_version=experiment.version,
                metrics=outcome.metrics,
                cost_summary=outcome.cost_summary,
                artifacts=[stored.descriptor],
                request_id=f"{request_id}:completed",
            )
            return QuantizationRunResult(
                experiment_id=experiment.experiment_id,
                status=completed.status,
                dry_run=False,
                quantization_profile_id=config.quantization_profile_id,
                source_experiment_id=source.experiment_id,
                source_artifact_id=source_artifact.artifact_id,
                calibration_manifest_hash=dataset.manifest_hash,
                artifact_object_key=stored.descriptor.object_key,
                metrics=outcome.metrics,
            )
        except Exception as exc:
            if not dry_run:
                self._close_failed(identity, experiment_id, exc, request_id)
            if isinstance(exc, QuantizationExecutionError):
                raise
            raise QuantizationExecutionError(_safe_reason(exc)) from exc

    def _load(
        self, identity: IdentityContext, experiment_id: str
    ) -> tuple[TrainingExperimentRecord, TrainingExperimentRecord, TrainingArtifactRecord]:
        with self._database.transaction(identity.tenant_context) as session:
            experiment = session.scalar(
                select(TrainingExperimentRecord).where(
                    TrainingExperimentRecord.tenant_id == identity.tenant_id,
                    TrainingExperimentRecord.experiment_id == experiment_id,
                )
            )
            if experiment is None:
                raise QuantizationExecutionError("quantization_experiment_not_visible")
            source_id = experiment.training_config.get("source_experiment_id")
            artifact_id = experiment.training_config.get("source_artifact_id")
            source = session.scalar(
                select(TrainingExperimentRecord).where(
                    TrainingExperimentRecord.tenant_id == identity.tenant_id,
                    TrainingExperimentRecord.experiment_id == source_id,
                )
            )
            artifact = session.scalar(
                select(TrainingArtifactRecord).where(
                    TrainingArtifactRecord.tenant_id == identity.tenant_id,
                    TrainingArtifactRecord.experiment_id == source_id,
                    TrainingArtifactRecord.artifact_id == artifact_id,
                    TrainingArtifactRecord.kind == "adapter_bundle",
                )
            )
            if source is None or artifact is None:
                raise QuantizationExecutionError("quantization_source_not_visible")
            return experiment, source, artifact

    def _close_failed(
        self,
        identity: IdentityContext,
        experiment_id: str,
        exc: Exception,
        request_id: str,
    ) -> None:
        try:
            experiment, _, _ = self._load(identity, experiment_id)
            if experiment.status in {"RUNNING", "COMPLETION_FAILED"}:
                self._registry.fail(
                    identity,
                    experiment_id,
                    expected_version=experiment.version,
                    reason_code=_safe_reason(exc),
                    request_id=f"{request_id}:failed",
                )
        except Exception as close_error:
            raise QuantizationExecutionError(
                f"{_safe_reason(exc)}:failure_state_could_not_be_closed"
            ) from close_error


def _verify_source_binding(
    experiment: TrainingExperimentRecord,
    source: TrainingExperimentRecord,
    artifact: TrainingArtifactRecord,
    expected_artifact_hash: str,
) -> None:
    if source.status != "COMPLETED" or source.method not in {"LORA", "QLORA", "DPO", "GRPO"}:
        raise QuantizationExecutionError("quantization_source_is_not_completed_peft")
    if artifact.content_hash != expected_artifact_hash:
        raise QuantizationExecutionError("quantization_source_artifact_hash_changed")
    if (
        source.comparison_group_id != experiment.comparison_group_id
        or source.dataset_snapshot_id != experiment.dataset_snapshot_id
        or source.dataset_manifest_hash != experiment.dataset_manifest_hash
        or source.base_model_digest != experiment.base_model_digest
        or source.tokenizer_digest != experiment.tokenizer_digest
        or source.chat_template_digest != experiment.chat_template_digest
        or source.random_seeds != experiment.random_seeds
    ):
        raise QuantizationExecutionError("quantization_source_contract_changed")


def _verify_runtime(
    experiment: TrainingExperimentRecord,
    algorithm: str,
    runtime: RuntimeFingerprint,
    *,
    dry_run: bool,
) -> None:
    if runtime.git_commit != experiment.git_commit:
        raise QuantizationExecutionError("runtime_git_commit_mismatch")
    if runtime.container_digest != experiment.container_digest:
        raise QuantizationExecutionError("runtime_container_digest_mismatch")
    if not dry_run and algorithm != "GGUF":
        verify_gpu_acceptance_report(
            runtime.gpu_acceptance_report,
            expected_image_digest=runtime.container_digest,
            expected_git_commit=runtime.git_commit,
        )


def _safe_reason(exc: Exception) -> str:
    if isinstance(exc, (ValueError, RuntimeError)):
        value = str(exc).strip().lower().replace(" ", "_")
        safe = "".join(
            character
            for character in value
            if character in "abcdefghijklmnopqrstuvwxyz0123456789_:-"
        )
        if safe:
            return safe[:255]
    return "quantization_worker_internal_error"
