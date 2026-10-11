"""Governed M7 VLM checkpoint continuation with targeted curriculum evidence."""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from industrial_ops_agent.data_pipeline.storage import FilesystemDatasetStore
from industrial_ops_agent.evaluation.backend import (
    EvaluationRuntimeConfig,
    VlmComponentEvaluationBackend,
    _vlm_output_contract_with_evidence,
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
    EvaluationSuiteRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.simulation.multimodal_platform_lab import TENANT_ID
from industrial_ops_agent.simulation.multimodal_snapshot import SIMULATION_TASK_TYPE
from industrial_ops_agent.simulation.vlm_training_lab import (
    MODEL_ID,
    MODEL_REVISION,
    SimulatedVlmTrainingReport,
    preflight_simulated_vlm_training_lab,
    verify_simulated_vlm_training_lab,
)
from industrial_ops_agent.training.artifacts import store_training_bundle
from industrial_ops_agent.training.backend import (
    HuggingFaceTrainingBackend,
    TrainingOutcome,
)
from industrial_ops_agent.training.checkpoints import (
    GovernedCheckpointError,
    resolve_governed_resume_checkpoint,
)
from industrial_ops_agent.training.config import (
    CompiledVlmConfig,
    compile_training_config,
)
from industrial_ops_agent.training.dataset import (
    GovernedSftDatasetLoader,
    TrainingDatasetBundle,
)

CLASSIFICATION = "SIMULATED_NON_PRODUCTION"
PARENT_PROFILE = "LORA"
REGION_IOU_MIN = 0.10
HALLUCINATION_RATE_MAX = 0.0
CURRICULUM_SCHEMA = "industrial-vlm-hard-negative-curriculum-v1"


class SimulatedVlmContinuationLabError(RuntimeError):
    """The continuation experiment or its immutable evidence is invalid."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileEvidence(_ClosedModel):
    path: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CurriculumEvidence(_ClosedModel):
    schema_version: Literal["industrial-vlm-hard-negative-curriculum-v1"] = (
        "industrial-vlm-hard-negative-curriculum-v1"
    )
    repeat_factor: int = Field(ge=2, le=8)
    base_train_rows: int = Field(gt=0)
    effective_train_rows: int = Field(gt=0)
    hard_negative_candidate_ids: tuple[str, ...]
    region_grounding_candidate_ids: tuple[str, ...]
    wide_region_candidate_ids: tuple[str, ...]
    target_candidate_ids: tuple[str, ...]
    selection_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class ContinuationTrainingEvidence(_ClosedModel):
    parent_experiment_id: str
    parent_artifact_id: str
    parent_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_checkpoint_path: str
    parent_checkpoint_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    child_experiment_id: str
    child_artifact_id: str
    child_artifact_object_key: str
    child_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    child_artifact_size_bytes: int = Field(gt=0)
    resumed_from_checkpoint: Literal[True] = True
    parent_optimizer_steps: int = Field(gt=0)
    child_optimizer_steps: int = Field(gt=0)
    added_optimizer_steps: int = Field(gt=0)
    train_loss: float
    validation_loss: float
    trainable_parameters: int = Field(gt=0)
    total_parameters: int = Field(gt=0)
    optimizer_learning_rate: float = Field(gt=0)
    runtime_seconds: float = Field(gt=0)
    peak_gpu_memory_allocated_bytes: int = Field(gt=0)
    compiled_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ContinuationEvaluationEvidence(_ClosedModel):
    parent_accuracy: float = Field(ge=0, le=1)
    child_accuracy: float = Field(ge=0, le=1)
    accuracy_delta: float
    parent_mean_region_iou: float = Field(ge=0, le=1)
    child_mean_region_iou: float = Field(ge=0, le=1)
    region_iou_delta: float
    parent_hallucination_rate: float = Field(ge=0, le=1)
    child_hallucination_rate: float = Field(ge=0, le=1)
    hard_gates: dict[str, bool]
    decision: Literal[
        "SIMULATION_CONTINUATION_ELIGIBLE",
        "SIMULATION_CONTINUATION_NO_GAIN",
        "SIMULATION_CONTINUATION_REJECTED",
    ]


class SimulatedVlmContinuationReport(_ClosedModel):
    schema_version: Literal["m7-vlm-checkpoint-continuation-lab/v1"] = (
        "m7-vlm-checkpoint-continuation-lab/v1"
    )
    classification: Literal["SIMULATED_NON_PRODUCTION"] = "SIMULATED_NON_PRODUCTION"
    production_claim: Literal[False] = False
    enterprise_production_data: Literal[False] = False
    production_release_eligible: Literal[False] = False
    formal_release_created: Literal[False] = False
    status: Literal["SIMULATED_VLM_CHECKPOINT_CONTINUATION_EVALUATED"] = (
        "SIMULATED_VLM_CHECKPOINT_CONTINUATION_EVALUATED"
    )
    decision: Literal[
        "SIMULATION_CONTINUATION_ELIGIBLE",
        "SIMULATION_CONTINUATION_NO_GAIN",
        "SIMULATION_CONTINUATION_REJECTED",
    ]
    generated_at: str
    model_id: Literal["Qwen/Qwen2-VL-2B-Instruct"] = "Qwen/Qwen2-VL-2B-Instruct"
    model_revision: Literal["895c3a49bc3fa70a340399125c650a463535e71c"] = (
        "895c3a49bc3fa70a340399125c650a463535e71c"
    )
    model_license: Literal["Apache-2.0"] = "Apache-2.0"
    source_training_acceptance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_rescore_acceptance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_snapshot_id: str
    training_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_snapshot_id: str
    evaluation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_reference_suite_id: str
    evaluation_cases: int = Field(gt=0)
    gpu_name: str
    gpu_total_memory_bytes: int = Field(gt=0)
    runtime_versions: dict[str, str]
    curriculum: CurriculumEvidence
    training: ContinuationTrainingEvidence
    evaluation: ContinuationEvaluationEvidence
    evidence_files: tuple[FileEvidence, ...]
    release_gate: Literal["DENIED_SIMULATION_CONTRACT"] = "DENIED_SIMULATION_CONTRACT"
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_simulated_vlm_continuation_lab(
    repo_root: Path,
    *,
    source_training_dir: Path,
    source_rescore_dir: Path,
    output_dir: Path,
    additional_steps: int = 20,
    repeat_factor: int = 3,
) -> tuple[SimulatedVlmContinuationReport, bool]:
    root = repo_root.resolve(strict=True)
    source_training = _inside_existing(root, source_training_dir)
    source_rescore = _inside_existing(root, source_rescore_dir)
    output = _new_output_path(root, output_dir)
    if output.exists():
        return verify_simulated_vlm_continuation_lab(output), False
    if not 1 <= additional_steps <= 120:
        raise SimulatedVlmContinuationLabError("vlm_continuation_additional_steps_are_out_of_range")
    if not 2 <= repeat_factor <= 8:
        raise SimulatedVlmContinuationLabError("vlm_continuation_repeat_factor_is_out_of_range")
    source_report = verify_simulated_vlm_training_lab(source_training)
    rescore_report = verify_simulated_vlm_training_lab(source_rescore)
    _verify_source_pair(source_report, rescore_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        report = _build_into(
            source_training=source_training,
            source_rescore=source_rescore,
            source_report=source_report,
            rescore_report=rescore_report,
            output=temporary,
            additional_steps=additional_steps,
            repeat_factor=repeat_factor,
        )
        _write_new(
            temporary / "acceptance.json",
            _json_bytes(report.model_dump(mode="json")),
        )
        verify_simulated_vlm_continuation_lab(temporary)
        temporary.rename(output)
    except Exception:
        _safe_remove_temporary(temporary, output.parent)
        raise
    return verify_simulated_vlm_continuation_lab(output), True


def rescore_simulated_vlm_continuation_lab(
    source_dir: Path,
    *,
    output_dir: Path,
) -> tuple[SimulatedVlmContinuationReport, bool]:
    """Reparse frozen model outputs and recompute evidence without retraining."""

    source = source_dir.resolve(strict=True)
    source_report = verify_simulated_vlm_continuation_lab(source)
    output = output_dir.resolve()
    if output.exists():
        return verify_simulated_vlm_continuation_lab(output), False
    if output == source or output.is_relative_to(source):
        raise SimulatedVlmContinuationLabError("vlm_continuation_rescore_output_path_is_invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        evaluation_dir = temporary / "evaluation"
        shutil.copy2(
            temporary / "acceptance.json",
            evaluation_dir / "source-continuation-acceptance.json",
        )
        shutil.copy2(
            evaluation_dir / "child-lora-continuation.json",
            evaluation_dir / "original-child-lora-continuation.json",
        )
        shutil.copy2(
            evaluation_dir / "parent-lora.json",
            evaluation_dir / "original-parent-lora.json",
        )
        shutil.copy2(
            evaluation_dir / "child-vs-parent-paired-metrics.json",
            evaluation_dir / "original-child-vs-parent-paired-metrics.json",
        )
        (temporary / "acceptance.json").unlink()

        child_observations = _normalize_vlm_observations(
            _load_observations(evaluation_dir / "child-lora-continuation.json")
        )
        parent_observations = _normalize_vlm_observations(
            _load_observations(evaluation_dir / "parent-lora.json")
        )
        _replace_file(
            evaluation_dir / "child-lora-continuation.json",
            _json_bytes(_observations(child_observations)),
        )
        _replace_file(
            evaluation_dir / "parent-lora.json",
            _json_bytes(_observations(parent_observations)),
        )

        database = Database(f"sqlite+pysqlite:///{temporary / 'catalog.sqlite3'}")
        store = FilesystemDatasetStore(temporary / "object-store")
        context = TenantContext(TENANT_ID, "m7-vlm-continuation-rescore-worker")
        try:
            with database.transaction(context) as session:
                suite = session.get(
                    EvaluationSuiteRecord,
                    source_report.simulation_reference_suite_id,
                )
            if suite is None:
                raise SimulatedVlmContinuationLabError("vlm_continuation_rescore_suite_is_missing")
            evaluation = FrozenEvaluationDatasetLoader(database, store).load(
                context,
                suite,
            )
        finally:
            database.dispose()
        paired = score_vlm_paired_observations(
            evaluation.cases,
            child_observations,
            parent_observations,
            region_iou_min=REGION_IOU_MIN,
            hallucination_rate_max=HALLUCINATION_RATE_MAX,
        )
        evaluation_evidence = _evaluation_evidence(paired)
        _replace_file(
            evaluation_dir / "child-vs-parent-paired-metrics.json",
            _json_bytes(_paired_report(paired)),
        )
        report = source_report.model_copy(
            update={
                "decision": evaluation_evidence.decision,
                "generated_at": datetime.now(UTC).isoformat(),
                "evaluation": evaluation_evidence,
                "evidence_files": tuple(_evidence_files(temporary)),
                "evidence_chain_sha256": "0" * 64,
            }
        )
        report = report.model_copy(update={"evidence_chain_sha256": _report_hash(report)})
        _write_new(
            temporary / "acceptance.json",
            _json_bytes(report.model_dump(mode="json")),
        )
        verify_simulated_vlm_continuation_lab(temporary)
        temporary.rename(output)
    except Exception:
        _safe_remove_temporary(temporary, output.parent)
        raise
    return verify_simulated_vlm_continuation_lab(output), True


def verify_simulated_vlm_continuation_lab(
    output_dir: Path,
) -> SimulatedVlmContinuationReport:
    output = output_dir.resolve(strict=True)
    try:
        report = SimulatedVlmContinuationReport.model_validate_json(
            (output / "acceptance.json").read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise SimulatedVlmContinuationLabError("vlm_continuation_acceptance_is_invalid") from exc
    if report.evidence_chain_sha256 != _report_hash(report):
        raise SimulatedVlmContinuationLabError("vlm_continuation_acceptance_digest_mismatch")
    for evidence in report.evidence_files:
        path = (output / evidence.path).resolve(strict=True)
        if not path.is_relative_to(output) or not path.is_file() or path.is_symlink():
            raise SimulatedVlmContinuationLabError("vlm_continuation_evidence_path_is_invalid")
        content = path.read_bytes()
        if len(content) != evidence.size_bytes or sha256(content).hexdigest() != evidence.sha256:
            raise SimulatedVlmContinuationLabError("vlm_continuation_evidence_integrity_failed")
    if (
        sha256((output / "source-training-acceptance.json").read_bytes()).hexdigest()
        != report.source_training_acceptance_sha256
        or sha256((output / "source-rescore-acceptance.json").read_bytes()).hexdigest()
        != report.source_rescore_acceptance_sha256
    ):
        raise SimulatedVlmContinuationLabError("vlm_continuation_source_acceptance_changed")

    database = Database(f"sqlite+pysqlite:///{output / 'catalog.sqlite3'}")
    store = FilesystemDatasetStore(output / "object-store")
    context = TenantContext(TENANT_ID, "m7-vlm-continuation-verifier")
    try:
        with database.transaction(context) as session:
            parent = session.get(
                TrainingExperimentRecord,
                report.training.parent_experiment_id,
            )
            child = session.get(
                TrainingExperimentRecord,
                report.training.child_experiment_id,
            )
            artifact = session.get(
                TrainingArtifactRecord,
                report.training.child_artifact_id,
            )
            suite = session.get(
                EvaluationSuiteRecord,
                report.simulation_reference_suite_id,
            )
        if parent is None or child is None or artifact is None or suite is None:
            raise SimulatedVlmContinuationLabError("vlm_continuation_catalog_binding_is_missing")
        compiled = compile_training_config(child)
        if not isinstance(compiled, CompiledVlmConfig):
            raise SimulatedVlmContinuationLabError("vlm_continuation_config_did_not_compile")
        dataset = GovernedSftDatasetLoader(database, store).load(context, child)
        content = store.get_bytes(artifact.object_key)
        metadata = artifact.metadata_json
        expected_lineage = {
            "source_experiment_id": report.training.parent_experiment_id,
            "source_artifact_id": report.training.parent_artifact_id,
            "source_content_hash": report.training.parent_artifact_sha256,
            "checkpoint_path": report.training.parent_checkpoint_path,
        }
        if (
            parent.status != "COMPLETED"
            or parent.method != "VLM"
            or child.status != "COMPLETED"
            or child.method != "VLM"
            or child.dataset_snapshot_id != parent.dataset_snapshot_id
            or child.dataset_manifest_hash != parent.dataset_manifest_hash
            or compiled.digest != report.training.compiled_config_sha256
            or compiled.resume_from_checkpoint is None
            or compiled.curriculum is None
            or not compiled.resume_ignore_data_skip
            or tuple(compiled.curriculum.target_candidate_ids)
            != report.curriculum.target_candidate_ids
            or len(dataset.train) != report.curriculum.base_train_rows
            or artifact.experiment_id != child.experiment_id
            or artifact.kind != "vlm_adapter_bundle"
            or artifact.content_hash != report.training.child_artifact_sha256
            or len(content) != artifact.size_bytes
            or sha256(content).hexdigest() != artifact.content_hash
            or metadata.get("resume_lineage") != expected_lineage
            or metadata.get("vlm_curriculum") != report.curriculum.model_dump(mode="json")
        ):
            raise SimulatedVlmContinuationLabError("vlm_continuation_catalog_or_lineage_changed")
        runtime_report = _load_json_object(
            output / "adapters/lora-continuation/industrial-ops-training-report.json"
        )
        runtime = runtime_report.get("runtime")
        if (
            not isinstance(runtime, dict)
            or runtime.get("resumed_from_checkpoint") is not True
            or runtime.get("resume_source_experiment_id") != report.training.parent_experiment_id
            or runtime.get("resume_source_artifact_id") != report.training.parent_artifact_id
            or runtime.get("resume_checkpoint_path") != report.training.parent_checkpoint_path
            or runtime.get("trainer_train_rows") != report.curriculum.effective_train_rows
            or runtime.get("resume_optimizer_learning_rate_override") is not True
            or runtime.get("optimizer_learning_rates") != [report.training.optimizer_learning_rate]
        ):
            raise SimulatedVlmContinuationLabError("vlm_continuation_runtime_lineage_changed")
        final_checkpoint = (
            output
            / "adapters/lora-continuation"
            / f"checkpoint-{report.training.child_optimizer_steps}"
            / "trainer_state.json"
        )
        if _trainer_global_step(final_checkpoint) != report.training.child_optimizer_steps:
            raise SimulatedVlmContinuationLabError(
                "vlm_continuation_final_checkpoint_is_incomplete"
            )
        evaluation = FrozenEvaluationDatasetLoader(database, store).load(
            context,
            suite,
        )
        child_observations = _load_observations(output / "evaluation/child-lora-continuation.json")
        parent_observations = _load_observations(output / "evaluation/parent-lora.json")
        if (
            len(evaluation.cases) != report.evaluation_cases
            or len(child_observations) != report.evaluation_cases
            or len(parent_observations) != report.evaluation_cases
        ):
            raise SimulatedVlmContinuationLabError("vlm_continuation_evaluation_count_changed")
        expected_evaluation = _evaluation_evidence(
            score_vlm_paired_observations(
                evaluation.cases,
                child_observations,
                parent_observations,
                region_iou_min=REGION_IOU_MIN,
                hallucination_rate_max=HALLUCINATION_RATE_MAX,
            )
        )
        if report.evaluation != expected_evaluation:
            raise SimulatedVlmContinuationLabError("vlm_continuation_evaluation_evidence_changed")
        if (
            report.production_claim
            or report.enterprise_production_data
            or report.production_release_eligible
            or report.formal_release_created
        ):
            raise SimulatedVlmContinuationLabError("vlm_continuation_production_boundary_changed")
    finally:
        database.dispose()
    return report


def preflight_simulated_vlm_continuation_lab(
    source_training_dir: Path,
    source_rescore_dir: Path,
) -> dict[str, Any]:
    source_report = verify_simulated_vlm_training_lab(source_training_dir.resolve(strict=True))
    rescore_report = verify_simulated_vlm_training_lab(source_rescore_dir.resolve(strict=True))
    _verify_source_pair(source_report, rescore_report)
    parent = _parent_training(source_report)
    return {
        **preflight_simulated_vlm_training_lab(),
        "continuation_status": "M7_VLM_CHECKPOINT_CONTINUATION_PREFLIGHT_PASSED",
        "parent_experiment_id": parent.experiment_id,
        "parent_artifact_sha256": parent.adapter_bundle_sha256,
        "parent_optimizer_steps": parent.optimizer_steps,
    }


def _build_into(
    *,
    source_training: Path,
    source_rescore: Path,
    source_report: SimulatedVlmTrainingReport,
    rescore_report: SimulatedVlmTrainingReport,
    output: Path,
    additional_steps: int,
    repeat_factor: int,
) -> SimulatedVlmContinuationReport:
    shutil.copy2(source_training / "catalog.sqlite3", output / "catalog.sqlite3")
    shutil.copytree(
        source_training / "object-store",
        output / "object-store",
        copy_function=os.link,
    )
    source_training_acceptance = (source_training / "acceptance.json").read_bytes()
    source_rescore_acceptance = (source_rescore / "acceptance.json").read_bytes()
    _write_new(
        output / "source-training-acceptance.json",
        source_training_acceptance,
    )
    _write_new(
        output / "source-rescore-acceptance.json",
        source_rescore_acceptance,
    )

    database = Database(f"sqlite+pysqlite:///{output / 'catalog.sqlite3'}")
    store = FilesystemDatasetStore(output / "object-store")
    context = TenantContext(TENANT_ID, "m7-vlm-continuation-worker")
    parent_evidence = _parent_training(source_report)
    try:
        with database.transaction(context) as session:
            parent = session.get(
                TrainingExperimentRecord,
                parent_evidence.experiment_id,
            )
            parent_artifact = session.scalar(
                select(TrainingArtifactRecord).where(
                    TrainingArtifactRecord.tenant_id == TENANT_ID,
                    TrainingArtifactRecord.experiment_id == parent_evidence.experiment_id,
                    TrainingArtifactRecord.kind == "vlm_adapter_bundle",
                )
            )
            suite = session.get(
                EvaluationSuiteRecord,
                source_report.simulation_reference_suite_id,
            )
        if parent is None or parent_artifact is None or suite is None:
            raise SimulatedVlmContinuationLabError("vlm_continuation_parent_binding_is_missing")
        if parent_artifact.content_hash != parent_evidence.adapter_bundle_sha256:
            raise SimulatedVlmContinuationLabError("vlm_continuation_parent_artifact_changed")
        parent_dataset = GovernedSftDatasetLoader(database, store).load(
            context,
            parent,
        )
        curriculum = _build_curriculum(parent_dataset, repeat_factor)
        child = _child_experiment(
            parent,
            parent_artifact,
            parent_optimizer_steps=parent_evidence.optimizer_steps,
            additional_steps=additional_steps,
            curriculum=curriculum,
        )
        _persist_experiment(database, context, child)
        compiled = compile_training_config(child)
        if not isinstance(compiled, CompiledVlmConfig):
            raise SimulatedVlmContinuationLabError("vlm_continuation_config_did_not_compile")
        child_dataset = GovernedSftDatasetLoader(database, store).load(
            context,
            child,
        )
        with tempfile.TemporaryDirectory(
            prefix="m7-vlm-resume-",
            dir=output.parent,
        ) as workspace_value:
            workspace = Path(workspace_value)
            try:
                checkpoint = resolve_governed_resume_checkpoint(
                    database,
                    store,
                    context,
                    child,
                    compiled,
                    workspace,
                )
            except GovernedCheckpointError as exc:
                raise SimulatedVlmContinuationLabError(str(exc)) from exc
            if checkpoint is None:
                raise SimulatedVlmContinuationLabError(
                    "vlm_continuation_checkpoint_was_not_resolved"
                )
            parent_step = _trainer_global_step(checkpoint / "trainer_state.json")
            if parent_step != parent_evidence.optimizer_steps:
                raise SimulatedVlmContinuationLabError("vlm_continuation_parent_step_changed")
            parent_checkpoint_manifest = _directory_manifest_digest(checkpoint)
            _write_new(
                output / "checkpoint-parent-receipt.json",
                _json_bytes(
                    {
                        "schema_version": "m7-vlm-parent-checkpoint-receipt/v1",
                        "source_experiment_id": parent.experiment_id,
                        "source_artifact_id": parent_artifact.artifact_id,
                        "source_artifact_sha256": parent_artifact.content_hash,
                        "checkpoint_path": checkpoint.name,
                        "checkpoint_manifest_sha256": parent_checkpoint_manifest,
                        "optimizer_steps": parent_step,
                    }
                ),
            )
            adapter_directory = output / "adapters/lora-continuation"
            outcome = HuggingFaceTrainingBackend().train(
                experiment=child,
                config=compiled,
                dataset=child_dataset,
                output_directory=adapter_directory,
                resume_from_checkpoint=checkpoint,
            )

        stored = store_training_bundle(
            store,
            tenant_id=TENANT_ID,
            experiment_id=child.experiment_id,
            directory=adapter_directory,
            kind="vlm_adapter_bundle",
            name="vlm-adapter-continuation",
            serialization="safetensors",
            metadata={
                "classification": CLASSIFICATION,
                "profile": "LORA_CONTINUATION",
                "base_model_id": MODEL_ID,
                "base_model_revision": MODEL_REVISION,
                "dataset_manifest_hash": child_dataset.manifest_hash,
                "compiled_config_digest": compiled.digest,
                "production_release_eligible": False,
                "resume_lineage": {
                    "source_experiment_id": parent.experiment_id,
                    "source_artifact_id": parent_artifact.artifact_id,
                    "source_content_hash": parent_artifact.content_hash,
                    "checkpoint_path": f"checkpoint-{parent_step}",
                },
                "vlm_curriculum": curriculum.model_dump(mode="json"),
            },
        )
        training = _complete_experiment(
            database,
            context,
            child,
            parent_artifact,
            parent_checkpoint_manifest,
            parent_step,
            compiled,
            outcome,
            stored.descriptor,
        )
        _release_gpu()

        evaluation = FrozenEvaluationDatasetLoader(database, store).load(
            context,
            suite,
        )
        parent_observations = _load_observations(source_rescore / "evaluation/lora.json")
        _write_new(
            output / "evaluation/parent-lora.json",
            _json_bytes(_observations(parent_observations)),
        )
        child_observations = VlmComponentEvaluationBackend().evaluate(
            experiment=child,
            cases=evaluation.cases,
            config=_evaluation_runtime(),
            adapter_directory=adapter_directory,
        )
        _write_new(
            output / "evaluation/child-lora-continuation.json",
            _json_bytes(_observations(child_observations)),
        )
        paired = score_vlm_paired_observations(
            evaluation.cases,
            child_observations,
            parent_observations,
            region_iou_min=REGION_IOU_MIN,
            hallucination_rate_max=HALLUCINATION_RATE_MAX,
        )
        evaluation_evidence = _evaluation_evidence(paired)
        _write_new(
            output / "evaluation/child-vs-parent-paired-metrics.json",
            _json_bytes(_paired_report(paired)),
        )
        _release_gpu()
    finally:
        database.dispose()

    evidence_files = tuple(_evidence_files(output))
    import torch

    report = SimulatedVlmContinuationReport(
        decision=evaluation_evidence.decision,
        generated_at=datetime.now(UTC).isoformat(),
        source_training_acceptance_sha256=sha256(source_training_acceptance).hexdigest(),
        source_rescore_acceptance_sha256=sha256(source_rescore_acceptance).hexdigest(),
        training_snapshot_id=source_report.training_snapshot_id,
        training_manifest_sha256=source_report.training_manifest_sha256,
        evaluation_snapshot_id=rescore_report.evaluation_snapshot_id,
        evaluation_manifest_sha256=rescore_report.evaluation_manifest_sha256,
        simulation_reference_suite_id=rescore_report.simulation_reference_suite_id,
        evaluation_cases=rescore_report.evaluation_cases,
        gpu_name=torch.cuda.get_device_name(0),
        gpu_total_memory_bytes=int(torch.cuda.get_device_properties(0).total_memory),
        runtime_versions=_runtime_versions(),
        curriculum=curriculum,
        training=training,
        evaluation=evaluation_evidence,
        evidence_files=evidence_files,
        evidence_chain_sha256="0" * 64,
    )
    return report.model_copy(update={"evidence_chain_sha256": _report_hash(report)})


def _build_curriculum(
    dataset: TrainingDatasetBundle,
    repeat_factor: int,
) -> CurriculumEvidence:
    hard_negative_ids: list[str] = []
    grounding_ids: list[str] = []
    wide_region_ids: list[str] = []
    for sample in dataset.train:
        if sample.multimodal_answer is None:
            raise SimulatedVlmContinuationLabError("vlm_continuation_training_answer_is_missing")
        finding = _single_finding(sample.multimodal_answer)
        region = finding["region"]
        if sample.candidate_id.endswith("valve_actuator_stiction"):
            hard_negative_ids.append(sample.candidate_id)
        if (
            float(region["x"]) < 0.20
            or float(region["y"]) < 0.18
            or float(region["x"]) + float(region["width"]) > 0.82
            or float(region["y"]) + float(region["height"]) > 0.82
        ):
            grounding_ids.append(sample.candidate_id)
        if float(region["width"]) >= 2.0 * float(region["height"]):
            wide_region_ids.append(sample.candidate_id)
    selected = set(hard_negative_ids + grounding_ids + wide_region_ids)
    targets = tuple(
        sample.candidate_id for sample in dataset.train if sample.candidate_id in selected
    )
    if not hard_negative_ids or not grounding_ids or not wide_region_ids or not targets:
        raise SimulatedVlmContinuationLabError("vlm_continuation_curriculum_selection_is_empty")
    contract = {
        "schema_version": CURRICULUM_SCHEMA,
        "target_candidate_ids": list(targets),
        "repeat_factor": repeat_factor,
    }
    effective_rows = len(dataset.train) + len(targets) * (repeat_factor - 1)
    return CurriculumEvidence(
        repeat_factor=repeat_factor,
        base_train_rows=len(dataset.train),
        effective_train_rows=effective_rows,
        hard_negative_candidate_ids=tuple(hard_negative_ids),
        region_grounding_candidate_ids=tuple(grounding_ids),
        wide_region_candidate_ids=tuple(wide_region_ids),
        target_candidate_ids=targets,
        selection_digest=sha256(_json_text(contract).encode()).hexdigest(),
    )


def _single_finding(answer: str) -> dict[str, Any]:
    try:
        document = json.loads(answer)
        findings = document["findings"]
        finding = findings[0]
        region = finding["region"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise SimulatedVlmContinuationLabError(
            "vlm_continuation_answer_contract_is_invalid"
        ) from exc
    if (
        not isinstance(document, dict)
        or not isinstance(findings, list)
        or len(findings) != 1
        or not isinstance(finding, dict)
        or not isinstance(region, dict)
        or set(region) != {"x", "y", "width", "height"}
    ):
        raise SimulatedVlmContinuationLabError("vlm_continuation_answer_contract_is_invalid")
    return finding


def _child_experiment(
    parent: TrainingExperimentRecord,
    parent_artifact: TrainingArtifactRecord,
    *,
    parent_optimizer_steps: int,
    additional_steps: int,
    curriculum: CurriculumEvidence,
) -> TrainingExperimentRecord:
    now = datetime.now(UTC)
    total_steps = parent_optimizer_steps + additional_steps
    config = dict(parent.training_config)
    config.update(
        {
            "max_steps": total_steps,
            "evaluation_interval": total_steps,
            "save_steps": total_steps,
            "learning_rate": 5e-5,
            "comparison_profile": "LORA_CONTINUATION",
            "resume_ignore_data_skip": True,
            "resume_optimizer_learning_rate_override": True,
            "resume_from_checkpoint": {
                "source_experiment_id": parent.experiment_id,
                "artifact_id": parent_artifact.artifact_id,
                "content_hash": parent_artifact.content_hash,
                "checkpoint_path": f"checkpoint-{parent_optimizer_steps}",
            },
            "vlm_curriculum": {
                "schema_version": curriculum.schema_version,
                "target_candidate_ids": list(curriculum.target_candidate_ids),
                "repeat_factor": curriculum.repeat_factor,
                "selection_digest": curriculum.selection_digest,
            },
        }
    )
    experiment_id = f"experiment-m7-vlm-lora-continuation-{total_steps}-{MODEL_REVISION[:12]}"
    return TrainingExperimentRecord(
        experiment_id=experiment_id,
        tenant_id=parent.tenant_id,
        idempotency_key=experiment_id,
        comparison_group_id="m7-qwen2-vl-simulated-checkpoint-continuation-v1",
        method="VLM",
        task_type=SIMULATION_TASK_TYPE,
        status="RUNNING",
        dataset_snapshot_id=parent.dataset_snapshot_id,
        dataset_manifest_hash=parent.dataset_manifest_hash,
        base_model_id=parent.base_model_id,
        base_model_digest=parent.base_model_digest,
        tokenizer_digest=parent.tokenizer_digest,
        chat_template_digest=parent.chat_template_digest,
        git_commit=_git_commit(),
        container_digest=_runtime_digest(),
        training_config=config,
        config_hash="sha256:" + sha256(_json_text(config).encode()).hexdigest(),
        distributed_profile=dict(parent.distributed_profile),
        random_seeds=list(parent.random_seeds),
        hardware_topology=dict(parent.hardware_topology),
        mlflow_experiment_name="m7-simulated-vlm-checkpoint-continuation",
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
    parent_artifact: TrainingArtifactRecord,
    parent_checkpoint_manifest: str,
    parent_step: int,
    compiled: CompiledVlmConfig,
    outcome: TrainingOutcome,
    descriptor: Any,
) -> ContinuationTrainingEvidence:
    optimizer_steps = int(outcome.metrics.get("optimizer_steps", 0))
    if optimizer_steps != compiled.max_steps or optimizer_steps <= parent_step:
        raise SimulatedVlmContinuationLabError("vlm_continuation_optimizer_steps_are_incomplete")
    if outcome.runtime_metadata.get("resumed_from_checkpoint") is not True:
        raise SimulatedVlmContinuationLabError("vlm_continuation_runtime_did_not_resume")
    optimizer_learning_rates = outcome.runtime_metadata.get("optimizer_learning_rates")
    if (
        optimizer_learning_rates != [compiled.learning_rate]
        or not compiled.resume_optimizer_learning_rate_override
    ):
        raise SimulatedVlmContinuationLabError(
            "vlm_continuation_optimizer_learning_rate_was_not_overridden"
        )
    artifact_id = f"artifact-m7-vlm-lora-continuation-{descriptor.content_hash[:24]}"
    now = datetime.now(UTC)
    with database.transaction(context) as session:
        stored = session.get(TrainingExperimentRecord, experiment.experiment_id)
        if stored is None or stored.status != "RUNNING":
            raise SimulatedVlmContinuationLabError("vlm_continuation_experiment_state_changed")
        stored.status = "COMPLETED"
        stored.metrics = outcome.metrics
        stored.cost_summary = outcome.cost_summary
        stored.completed_at = now
        stored.updated_at = now
        stored.version += 1
        session.add(
            TrainingArtifactRecord(
                artifact_id=artifact_id,
                tenant_id=TENANT_ID,
                experiment_id=experiment.experiment_id,
                kind=descriptor.kind,
                object_key=descriptor.object_key,
                content_hash=descriptor.content_hash,
                size_bytes=descriptor.size_bytes,
                metadata_json=dict(descriptor.metadata),
                created_at=now,
                updated_at=now,
            )
        )
    metrics = outcome.metrics
    return ContinuationTrainingEvidence(
        parent_experiment_id=parent_artifact.experiment_id,
        parent_artifact_id=parent_artifact.artifact_id,
        parent_artifact_sha256=parent_artifact.content_hash,
        parent_checkpoint_path=f"checkpoint-{parent_step}",
        parent_checkpoint_manifest_sha256=parent_checkpoint_manifest,
        child_experiment_id=experiment.experiment_id,
        child_artifact_id=artifact_id,
        child_artifact_object_key=descriptor.object_key,
        child_artifact_sha256=descriptor.content_hash,
        child_artifact_size_bytes=descriptor.size_bytes,
        parent_optimizer_steps=parent_step,
        child_optimizer_steps=optimizer_steps,
        added_optimizer_steps=optimizer_steps - parent_step,
        train_loss=float(metrics.get("train_loss", metrics.get("loss", 0.0))),
        validation_loss=float(metrics.get("eval_loss", 0.0)),
        trainable_parameters=int(metrics.get("trainable_parameters", 0)),
        total_parameters=int(metrics.get("total_parameters", 0)),
        optimizer_learning_rate=compiled.learning_rate,
        runtime_seconds=float(metrics.get("train_runtime_seconds", 0.0)),
        peak_gpu_memory_allocated_bytes=int(metrics.get("peak_gpu_memory_allocated_bytes", 0)),
        compiled_config_sha256=compiled.digest,
    )


def _evaluation_evidence(
    paired: PairedMetricReport,
) -> ContinuationEvaluationEvidence:
    aggregate = paired.aggregate_metrics.get("vlm")
    if not isinstance(aggregate, dict):
        raise SimulatedVlmContinuationLabError("vlm_continuation_evaluation_aggregate_is_missing")
    accuracy = aggregate["diagnostic_accuracy"]
    region = aggregate["mean_region_iou"]
    hallucination = aggregate["hallucination_rate"]
    parent_accuracy = float(accuracy["baseline"])
    child_accuracy = float(accuracy["candidate"])
    parent_region = float(region["baseline"])
    child_region = float(region["candidate"])
    gates = dict(paired.evidence.hard_gate_results)
    decision: Literal[
        "SIMULATION_CONTINUATION_ELIGIBLE",
        "SIMULATION_CONTINUATION_NO_GAIN",
        "SIMULATION_CONTINUATION_REJECTED",
    ]
    if all(gates.values()) and (child_accuracy > parent_accuracy or child_region > parent_region):
        decision = "SIMULATION_CONTINUATION_ELIGIBLE"
    elif all(gates.values()):
        decision = "SIMULATION_CONTINUATION_NO_GAIN"
    else:
        decision = "SIMULATION_CONTINUATION_REJECTED"
    return ContinuationEvaluationEvidence(
        parent_accuracy=parent_accuracy,
        child_accuracy=child_accuracy,
        accuracy_delta=child_accuracy - parent_accuracy,
        parent_mean_region_iou=parent_region,
        child_mean_region_iou=child_region,
        region_iou_delta=child_region - parent_region,
        parent_hallucination_rate=float(hallucination["baseline"]),
        child_hallucination_rate=float(hallucination["candidate"]),
        hard_gates=gates,
        decision=decision,
    )


def _parent_training(report: SimulatedVlmTrainingReport) -> Any:
    for training in report.trainings:
        if training.profile == PARENT_PROFILE:
            return training
    raise SimulatedVlmContinuationLabError("vlm_continuation_parent_profile_is_missing")


def _verify_source_pair(
    source: SimulatedVlmTrainingReport,
    rescore: SimulatedVlmTrainingReport,
) -> None:
    if (
        source.training_snapshot_id != rescore.training_snapshot_id
        or source.training_manifest_sha256 != rescore.training_manifest_sha256
        or source.evaluation_snapshot_id != rescore.evaluation_snapshot_id
        or source.evaluation_manifest_sha256 != rescore.evaluation_manifest_sha256
        or tuple(
            (item.profile, item.experiment_id, item.adapter_bundle_sha256)
            for item in source.trainings
        )
        != tuple(
            (item.profile, item.experiment_id, item.adapter_bundle_sha256)
            for item in rescore.trainings
        )
    ):
        raise SimulatedVlmContinuationLabError("vlm_continuation_source_reports_do_not_match")


def _evaluation_runtime() -> EvaluationRuntimeConfig:
    return EvaluationRuntimeConfig(
        max_new_tokens=128,
        precision="bfloat16",
        gpu_hourly_cost_usd=0.50,
        max_image_pixels=307_200,
    )


def _observations(
    observations: tuple[ModelObservation, ...],
) -> list[dict[str, Any]]:
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
        raise SimulatedVlmContinuationLabError(
            "vlm_continuation_observation_evidence_is_invalid"
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise SimulatedVlmContinuationLabError("vlm_continuation_observation_evidence_is_invalid")
    observations: list[ModelObservation] = []
    try:
        for item in payload:
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
        raise SimulatedVlmContinuationLabError(
            "vlm_continuation_observation_evidence_is_invalid"
        ) from exc
    return tuple(observations)


def _normalize_vlm_observations(
    observations: tuple[ModelObservation, ...],
) -> tuple[ModelObservation, ...]:
    normalized: list[ModelObservation] = []
    for observation in observations:
        findings, valid, repair = _vlm_output_contract_with_evidence(observation.output_text)
        normalized.append(
            ModelObservation(
                case_id=observation.case_id,
                output_text=observation.output_text,
                latency_ms=observation.latency_ms,
                cost_usd=observation.cost_usd,
                input_tokens=observation.input_tokens,
                output_tokens=observation.output_tokens,
                capabilities=observation.capabilities,
                visual_findings=findings,
                structured_output_valid=valid,
                runtime_evidence={
                    **observation.runtime_evidence,
                    "structured_output_repair": repair,
                },
            )
        )
    return tuple(normalized)


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


def _runtime_digest() -> str:
    return "sha256:" + sha256(_json_text(_runtime_versions()).encode()).hexdigest()


def _git_commit() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SimulatedVlmContinuationLabError(
            "vlm_continuation_git_commit_could_not_be_resolved"
        ) from exc
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise SimulatedVlmContinuationLabError("vlm_continuation_git_commit_is_invalid")
    return value


def _directory_manifest_digest(directory: Path) -> str:
    manifest = [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    if not manifest:
        raise SimulatedVlmContinuationLabError("vlm_continuation_checkpoint_is_empty")
    return sha256(_json_text(manifest).encode()).hexdigest()


def _trainer_global_step(path: Path) -> int:
    document = _load_json_object(path)
    value = document.get("global_step")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SimulatedVlmContinuationLabError("vlm_continuation_trainer_state_is_invalid")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulatedVlmContinuationLabError("vlm_continuation_json_evidence_is_invalid") from exc
    if not isinstance(value, dict):
        raise SimulatedVlmContinuationLabError("vlm_continuation_json_evidence_is_invalid")
    return value


def _evidence_files(output: Path) -> list[FileEvidence]:
    roots = [
        output / "source-training-acceptance.json",
        output / "source-rescore-acceptance.json",
        output / "checkpoint-parent-receipt.json",
        output / "adapters",
        output / "evaluation",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
        else:
            raise SimulatedVlmContinuationLabError("vlm_continuation_evidence_is_missing")
    evidence: list[FileEvidence] = []
    for path in sorted(files):
        if path.is_symlink():
            raise SimulatedVlmContinuationLabError("vlm_continuation_evidence_contains_symlink")
        content = path.read_bytes()
        evidence.append(
            FileEvidence(
                path=path.relative_to(output).as_posix(),
                size_bytes=len(content),
                sha256=sha256(content).hexdigest(),
            )
        )
    return evidence


def _inside_existing(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.exists():
        raise SimulatedVlmContinuationLabError("vlm_continuation_source_path_is_outside_repo")
    return resolved


def _new_output_path(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise SimulatedVlmContinuationLabError("vlm_continuation_output_path_is_outside_repo")
    return resolved


def _report_hash(report: SimulatedVlmContinuationReport) -> str:
    document = report.model_dump(mode="json")
    document.pop("evidence_chain_sha256", None)
    return sha256(_json_text(document).encode()).hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(content)


def _replace_file(path: Path, content: bytes) -> None:
    if not path.is_file() or path.is_symlink():
        raise SimulatedVlmContinuationLabError("vlm_continuation_rescore_evidence_path_is_invalid")
    path.unlink()
    _write_new(path, content)


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def _safe_remove_temporary(temporary: Path, parent: Path) -> None:
    resolved = temporary.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith("."):
        raise SimulatedVlmContinuationLabError(
            "temporary_vlm_continuation_cleanup_target_is_unsafe"
        )
    shutil.rmtree(resolved, ignore_errors=True)
