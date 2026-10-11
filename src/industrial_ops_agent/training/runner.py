"""One governed execution path from a registered run to an immutable model artifact."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.experiments.service import ExperimentRegistryService
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    TrainingExperimentRecord,
)
from industrial_ops_agent.predictive_maintenance.rul_training_dataset import (
    GovernedRulDatasetLoader,
    RulTrainingDatasetBundle,
)
from industrial_ops_agent.predictive_maintenance.training_dataset import (
    GovernedTelemetryDatasetLoader,
    TelemetryTrainingDatasetBundle,
)
from industrial_ops_agent.training.artifacts import store_training_bundle
from industrial_ops_agent.training.backend import TrainingBackend
from industrial_ops_agent.training.checkpoints import (
    GovernedCheckpointError,
    resolve_governed_resume_checkpoint,
)
from industrial_ops_agent.training.config import (
    CompiledPeftConfig,
    CompiledTrainingConfig,
    CompiledTtsConfig,
    compile_training_config,
)
from industrial_ops_agent.training.dataset import (
    GovernedSftDatasetLoader,
    TrainingDatasetBundle,
)
from industrial_ops_agent.training.distributed import process_rank
from industrial_ops_agent.training.image_acceptance import (
    gpu_acceptance_artifact_metadata,
    verify_gpu_acceptance_report,
    write_gpu_acceptance_report,
)


class TrainingExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeFingerprint:
    git_commit: str
    container_digest: str
    gpu_acceptance_report: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    experiment_id: str
    status: str
    dry_run: bool
    config_digest: str
    manifest_hash: str
    train_rows: int
    validation_rows: int
    artifact_object_key: str | None
    metrics: dict[str, float]


class TrainingRunner:
    def __init__(
        self,
        database: Database,
        store: DatasetStore,
        registry: ExperimentRegistryService,
        backend: TrainingBackend,
    ) -> None:
        self._database = database
        self._store = store
        self._registry = registry
        self._backend = backend
        self._dataset_loader = GovernedSftDatasetLoader(database, store)
        self._telemetry_dataset_loader = GovernedTelemetryDatasetLoader(database, store)
        self._rul_dataset_loader = GovernedRulDatasetLoader(database, store)

    def run(
        self,
        identity: IdentityContext,
        experiment_id: str,
        *,
        runtime: RuntimeFingerprint,
        dry_run: bool,
        request_id: str,
    ) -> TrainingRunResult:
        experiment = self._load_experiment(identity, experiment_id)
        if dry_run:
            if experiment.status not in {"PLANNED", "RUNNING"}:
                raise TrainingExecutionError("dry_run_requires_planned_or_running_experiment")
        elif experiment.status != "RUNNING":
            raise TrainingExecutionError("training_requires_running_experiment")

        try:
            compiled = compile_training_config(experiment)
            _verify_runtime(
                experiment,
                runtime,
                require_gpu_acceptance=not dry_run,
                expected_gpu_count=_compiled_gpu_count(compiled),
            )
            dataset: (
                TrainingDatasetBundle | TelemetryTrainingDatasetBundle | RulTrainingDatasetBundle
            )
            if experiment.method == "TIMESERIES_TRANSFORMER":
                dataset = self._telemetry_dataset_loader.load(identity.tenant_context, experiment)
            elif experiment.method == "RUL_TRANSFORMER":
                dataset = self._rul_dataset_loader.load(identity.tenant_context, experiment)
            else:
                dataset = self._dataset_loader.load(identity.tenant_context, experiment)
            if dry_run:
                return TrainingRunResult(
                    experiment_id=experiment_id,
                    status="VALIDATED",
                    dry_run=True,
                    config_digest=compiled.digest,
                    manifest_hash=dataset.manifest_hash,
                    train_rows=len(dataset.train),
                    validation_rows=len(dataset.validation),
                    artifact_object_key=None,
                    metrics={},
                )
            return self._execute(
                identity,
                experiment,
                compiled,
                dataset,
                runtime,
                request_id=request_id,
            )
        except Exception as exc:
            if not dry_run and process_rank() == 0:
                reason_code = _safe_reason_code(exc)
                try:
                    current = self._load_experiment(identity, experiment_id)
                    self._registry.fail(
                        identity,
                        experiment_id,
                        expected_version=current.version,
                        reason_code=reason_code,
                        request_id=f"{request_id}:failed",
                    )
                except Exception as close_error:
                    raise TrainingExecutionError(
                        f"{reason_code}:failure_state_could_not_be_closed"
                    ) from close_error
            if isinstance(exc, TrainingExecutionError):
                raise
            raise TrainingExecutionError(_safe_reason_code(exc)) from exc

    def _execute(
        self,
        identity: IdentityContext,
        experiment: TrainingExperimentRecord,
        compiled: CompiledTrainingConfig,
        dataset: (
            TrainingDatasetBundle | TelemetryTrainingDatasetBundle | RulTrainingDatasetBundle
        ),
        runtime: RuntimeFingerprint,
        *,
        request_id: str,
    ) -> TrainingRunResult:
        with _training_output_directory(compiled) as output_directory:
            resume_checkpoint = self._prepare_resume_checkpoint(
                identity,
                experiment,
                compiled,
                output_directory.parent,
            )
            outcome = self._backend.train(
                experiment=experiment,
                config=compiled,
                dataset=dataset,
                output_directory=output_directory,
                resume_from_checkpoint=resume_checkpoint,
            )
            _distributed_barrier(compiled)
            if process_rank() != 0:
                return TrainingRunResult(
                    experiment_id=experiment.experiment_id,
                    status="DISTRIBUTED_WORKER_COMPLETE",
                    dry_run=False,
                    config_digest=compiled.digest,
                    manifest_hash=dataset.manifest_hash,
                    train_rows=len(dataset.train),
                    validation_rows=len(dataset.validation),
                    artifact_object_key=None,
                    metrics=outcome.metrics,
                )
            _verify_training_output(compiled, outcome.output_directory)
            assert runtime.gpu_acceptance_report is not None
            write_gpu_acceptance_report(
                outcome.output_directory / "industrial-ops-gpu-image-acceptance.json",
                runtime.gpu_acceptance_report,
            )
            artifact_profile = _artifact_profile(experiment.method)
            stored = store_training_bundle(
                self._store,
                tenant_id=identity.tenant_id,
                experiment_id=experiment.experiment_id,
                directory=outcome.output_directory,
                kind=artifact_profile[0],
                name=artifact_profile[1],
                serialization=artifact_profile[2],
                metadata={
                    "method": experiment.method,
                    "base_model_id": experiment.base_model_id,
                    "base_model_digest": experiment.base_model_digest,
                    "base_model_revision": compiled.base_model_revision,
                    "dataset_manifest_hash": dataset.manifest_hash,
                    "compiled_config_digest": compiled.digest,
                    "runtime": outcome.runtime_metadata,
                    "gpu_image_acceptance": gpu_acceptance_artifact_metadata(
                        runtime.gpu_acceptance_report or {}
                    ),
                    **_resume_artifact_metadata(compiled),
                    **_method_artifact_metadata(compiled),
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
        return TrainingRunResult(
            experiment_id=experiment.experiment_id,
            status=completed.status,
            dry_run=False,
            config_digest=compiled.digest,
            manifest_hash=dataset.manifest_hash,
            train_rows=len(dataset.train),
            validation_rows=len(dataset.validation),
            artifact_object_key=stored.descriptor.object_key,
            metrics=outcome.metrics,
        )

    def _prepare_resume_checkpoint(
        self,
        identity: IdentityContext,
        experiment: TrainingExperimentRecord,
        compiled: CompiledTrainingConfig,
        workspace: Path,
    ) -> Path | None:
        if not isinstance(compiled, CompiledPeftConfig):
            return None
        try:
            return resolve_governed_resume_checkpoint(
                self._database,
                self._store,
                identity.tenant_context,
                experiment,
                compiled,
                workspace,
            )
        except GovernedCheckpointError as exc:
            raise TrainingExecutionError(str(exc)) from exc

    def _load_experiment(
        self, identity: IdentityContext, experiment_id: str
    ) -> TrainingExperimentRecord:
        with self._database.transaction(identity.tenant_context) as session:
            experiment = session.scalar(
                select(TrainingExperimentRecord).where(
                    TrainingExperimentRecord.tenant_id == identity.tenant_id,
                    TrainingExperimentRecord.experiment_id == experiment_id,
                )
            )
            if experiment is None:
                raise TrainingExecutionError("training_experiment_not_visible")
            return experiment


def _verify_runtime(
    experiment: TrainingExperimentRecord,
    runtime: RuntimeFingerprint,
    *,
    require_gpu_acceptance: bool,
    expected_gpu_count: int,
) -> None:
    if not runtime.git_commit or runtime.git_commit != experiment.git_commit:
        raise TrainingExecutionError("runtime_git_commit_mismatch")
    if not runtime.container_digest or runtime.container_digest != experiment.container_digest:
        raise TrainingExecutionError("runtime_container_digest_mismatch")
    if require_gpu_acceptance:
        verify_gpu_acceptance_report(
            runtime.gpu_acceptance_report,
            expected_image_digest=runtime.container_digest,
            expected_git_commit=runtime.git_commit,
            expected_visible_gpu_count=expected_gpu_count,
        )


def _compiled_gpu_count(config: CompiledTrainingConfig) -> int:
    distributed = getattr(config, "distributed", None)
    return int(getattr(distributed, "world_size", 1))


@contextmanager
def _training_output_directory(config: CompiledTrainingConfig) -> Iterator[Path]:
    if _compiled_gpu_count(config) > 1:
        shared = os.getenv("IOAP_DISTRIBUTED_OUTPUT_DIRECTORY")
        if not shared:
            raise TrainingExecutionError("distributed_output_directory_is_required")
        path = Path(shared)
        path.parent.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with TemporaryDirectory(prefix="industrial-ops-training-") as directory:
        yield Path(directory) / "model-output"


def _distributed_barrier(config: CompiledTrainingConfig) -> None:
    if _compiled_gpu_count(config) == 1:
        return
    torch = importlib.import_module("torch")
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        raise TrainingExecutionError("distributed_process_group_is_not_initialized")
    distributed.barrier()


def _safe_reason_code(exc: Exception) -> str:
    if isinstance(exc, (ValueError, RuntimeError)):
        value = str(exc).strip().lower().replace(" ", "_")
        safe = "".join(
            character
            for character in value
            if character in "abcdefghijklmnopqrstuvwxyz0123456789_:-"
        )
        if safe:
            return safe[:255]
    return "training_worker_internal_error"


def _artifact_profile(method: str) -> tuple[str, str, str]:
    if method in {"LORA", "QLORA", "DPO", "GRPO"}:
        return "adapter_bundle", "adapter", "safetensors"
    if method == "PPO":
        return "ppo_policy_adapter_bundle", "ppo-policy", "safetensors"
    if method == "EMBEDDING":
        return "embedding_model_bundle", "embedding-model", "safetensors"
    if method == "RERANKER":
        return "reranker_model_bundle", "reranker-model", "safetensors"
    if method == "VLM":
        return "vlm_adapter_bundle", "vlm-adapter", "safetensors"
    if method == "ASR":
        return "asr_adapter_bundle", "asr-adapter", "safetensors"
    if method == "TTS":
        return "tts_model_bundle", "tts-model", "safetensors"
    if method == "TIMESERIES_TRANSFORMER":
        return "timeseries_transformer_bundle", "timeseries-transformer", "safetensors"
    if method == "RUL_TRANSFORMER":
        return "rul_transformer_bundle", "rul-transformer", "safetensors"
    raise TrainingExecutionError("training_method_has_no_artifact_profile")


def _verify_training_output(config: CompiledTrainingConfig, output_directory: Path) -> None:
    if not isinstance(config, CompiledTtsConfig):
        return
    required = {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "industrial-ops-tts-voice-profile.json",
    }
    if not all((output_directory / name).is_file() for name in required) or not (
        (output_directory / "model.safetensors").is_file()
        or (output_directory / "model.safetensors.index.json").is_file()
    ):
        raise TrainingExecutionError("tts_training_output_is_incomplete")
    try:
        profile = json.loads(
            (output_directory / "industrial-ops-tts-voice-profile.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingExecutionError("tts_voice_profile_artifact_is_invalid") from exc
    expected = {
        "schema_version": "industrial-ops-tts-voice-profile/v1",
        "voice_profile_id": config.voice_profile_id,
        "language": config.language,
        "sampling_rate": config.sampling_rate,
        "speaker_embedding_dimension": config.speaker_embedding_dimension,
        "speech_contract_version": config.speech_contract_version,
        "base_model_revision": config.base_model_revision,
    }
    if not isinstance(profile, dict) or any(
        profile.get(key) != value for key, value in expected.items()
    ):
        raise TrainingExecutionError("tts_voice_profile_artifact_binding_changed")
    if "speaker_embedding" in profile or "media_content" in profile:
        raise TrainingExecutionError("tts_voice_profile_artifact_contains_sensitive_data")


def _method_artifact_metadata(config: CompiledTrainingConfig) -> dict[str, object]:
    if not isinstance(config, CompiledTtsConfig):
        return {}
    return {
        "voice_profile": {
            "voice_profile_id": config.voice_profile_id,
            "language": config.language,
            "sampling_rate": config.sampling_rate,
            "speaker_embedding_dimension": config.speaker_embedding_dimension,
            "speech_contract_version": config.speech_contract_version,
        }
    }


def _resume_artifact_metadata(config: CompiledTrainingConfig) -> dict[str, object]:
    if not isinstance(config, CompiledPeftConfig):
        return {}
    descriptor = config.resume_from_checkpoint
    if descriptor is None:
        return {}
    return {
        "resume_lineage": {
            "source_experiment_id": descriptor.source_experiment_id,
            "source_artifact_id": descriptor.artifact_id,
            "source_content_hash": descriptor.content_hash,
            "checkpoint_path": descriptor.checkpoint_path,
        }
    }
