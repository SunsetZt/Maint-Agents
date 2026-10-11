"""Governed checkpoint resolution shared by workers and controlled laboratories."""

from __future__ import annotations

import fcntl
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import select

from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.training.artifacts import extract_training_bundle
from industrial_ops_agent.training.config import CompiledPeftConfig


class GovernedCheckpointError(RuntimeError):
    """A checkpoint failed visibility, integrity, lineage, or completeness checks."""


def resolve_governed_resume_checkpoint(
    database: Database,
    store: DatasetStore,
    context: TenantContext,
    experiment: TrainingExperimentRecord,
    compiled: CompiledPeftConfig,
    workspace: Path,
) -> Path | None:
    """Resolve one immutable parent checkpoint after all governed bindings pass."""

    descriptor = compiled.resume_from_checkpoint
    if descriptor is None:
        return None
    with database.transaction(context) as session:
        source = session.scalar(
            select(TrainingExperimentRecord).where(
                TrainingExperimentRecord.tenant_id == context.tenant_id,
                TrainingExperimentRecord.experiment_id == descriptor.source_experiment_id,
            )
        )
        artifact = session.scalar(
            select(TrainingArtifactRecord).where(
                TrainingArtifactRecord.tenant_id == context.tenant_id,
                TrainingArtifactRecord.experiment_id == descriptor.source_experiment_id,
                TrainingArtifactRecord.artifact_id == descriptor.artifact_id,
            )
        )
        if source is None or artifact is None:
            raise GovernedCheckpointError("resume_checkpoint_artifact_is_not_visible")
        _verify_resume_source(
            experiment,
            source,
            artifact,
            descriptor.content_hash,
        )
        object_key = artifact.object_key
        size_bytes = artifact.size_bytes

    workspace.mkdir(parents=True, exist_ok=True)
    resume_root = workspace / "resume-input"
    ready_path = workspace / "resume-input.ready.json"
    lock_path = workspace / ".resume-input.lock"
    expected_receipt = {
        "artifact_id": descriptor.artifact_id,
        "content_hash": descriptor.content_hash,
        "checkpoint_path": descriptor.checkpoint_path,
    }
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if ready_path.is_file():
            try:
                receipt = json.loads(ready_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise GovernedCheckpointError("resume_checkpoint_receipt_is_invalid") from exc
            if receipt != expected_receipt:
                raise GovernedCheckpointError("resume_checkpoint_receipt_mismatch")
        elif resume_root.is_dir():
            checkpoint = resume_root / descriptor.checkpoint_path
            _verify_checkpoint_directory(checkpoint)
            _write_resume_receipt(ready_path, expected_receipt)
        else:
            content = store.get_bytes(object_key)
            if len(content) != size_bytes or sha256(content).hexdigest() != descriptor.content_hash:
                raise GovernedCheckpointError("resume_checkpoint_artifact_integrity_failed")
            with TemporaryDirectory(prefix="resume-unpack-", dir=workspace) as temporary:
                unpacked = Path(temporary) / "bundle"
                extract_training_bundle(content, unpacked)
                checkpoint = unpacked / descriptor.checkpoint_path
                _verify_checkpoint_directory(checkpoint)
                unpacked.rename(resume_root)
            _write_resume_receipt(ready_path, expected_receipt)
        checkpoint = resume_root / descriptor.checkpoint_path
        _verify_checkpoint_directory(checkpoint)
        return checkpoint


def _verify_resume_source(
    target: TrainingExperimentRecord,
    source: TrainingExperimentRecord,
    artifact: TrainingArtifactRecord,
    expected_content_hash: str,
) -> None:
    if source.status != "COMPLETED" or source.method != target.method:
        raise GovernedCheckpointError("resume_source_must_be_a_completed_matching_method")
    expected_kind = {
        "LORA": "adapter_bundle",
        "QLORA": "adapter_bundle",
        "VLM": "vlm_adapter_bundle",
    }.get(target.method)
    if expected_kind is None:
        raise GovernedCheckpointError("resume_method_is_not_supported")
    if artifact.kind != expected_kind or artifact.content_hash != expected_content_hash:
        raise GovernedCheckpointError("resume_checkpoint_artifact_binding_changed")
    if (
        source.dataset_snapshot_id != target.dataset_snapshot_id
        or source.dataset_manifest_hash != target.dataset_manifest_hash
        or source.base_model_id != target.base_model_id
        or source.base_model_digest != target.base_model_digest
        or source.tokenizer_digest != target.tokenizer_digest
        or source.chat_template_digest != target.chat_template_digest
        or source.random_seeds != target.random_seeds
    ):
        raise GovernedCheckpointError("resume_checkpoint_training_contract_changed")


def _verify_checkpoint_directory(checkpoint: Path) -> None:
    if not checkpoint.is_dir() or not (checkpoint / "trainer_state.json").is_file():
        raise GovernedCheckpointError("resume_checkpoint_directory_is_incomplete")
    if not any(path.is_file() for path in checkpoint.rglob("*")):
        raise GovernedCheckpointError("resume_checkpoint_directory_is_empty")


def _write_resume_receipt(path: Path, receipt: dict[str, str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)
