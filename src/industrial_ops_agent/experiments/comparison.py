"""Dataset comparison contracts shared by experiment, evaluation and release gates."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.data_pipeline.contracts import SYNTHETIC_AUGMENTATION_CONTRACT
from industrial_ops_agent.persistence.models import (
    DatasetSnapshotRecord,
    TrainingExperimentRecord,
)

MODEL_METHOD_COMPARISON = "MODEL_METHOD"
SYNTHETIC_DATASET_ABLATION = "SYNTHETIC_DATASET_ABLATION"
QUANTIZATION_ABLATION = "QUANTIZATION_ABLATION"


class SyntheticAblationContractError(ValueError):
    """A declared base/derived snapshot pair is not a controlled VLM ablation."""


class QuantizationAblationContractError(ValueError):
    """A quantized candidate is not bound to the evaluated source model artifact."""


def quantization_ablation_context(
    session: Session,
    tenant_id: str,
    candidate: TrainingExperimentRecord,
    baseline: TrainingExperimentRecord,
) -> dict[str, Any] | None:
    """Return the immutable source/calibration binding for a quantization comparison."""

    if candidate.method != "QUANTIZATION":
        return None
    config = candidate.training_config
    if config.get("source_experiment_id") != baseline.experiment_id:
        raise QuantizationAblationContractError("quantization_source_experiment_mismatch")
    source_artifact_id = config.get("source_artifact_id")
    source_artifact_hash = str(config.get("source_artifact_hash", "")).removeprefix(
        "sha256:"
    )
    if not isinstance(source_artifact_id, str) or not isinstance(source_artifact_hash, str):
        raise QuantizationAblationContractError("quantization_source_artifact_binding_missing")

    # Import locally so this comparison module stays lightweight for dataset-only callers.
    from industrial_ops_agent.persistence.models import TrainingArtifactRecord

    source_artifact = session.scalar(
        select(TrainingArtifactRecord).where(
            TrainingArtifactRecord.tenant_id == tenant_id,
            TrainingArtifactRecord.experiment_id == baseline.experiment_id,
            TrainingArtifactRecord.artifact_id == source_artifact_id,
        )
    )
    if source_artifact is None or source_artifact.content_hash != source_artifact_hash:
        raise QuantizationAblationContractError("quantization_source_artifact_binding_changed")
    if source_artifact.kind != "adapter_bundle":
        raise QuantizationAblationContractError("quantization_source_must_be_peft_adapter")

    return {
        "source_experiment_id": baseline.experiment_id,
        "source_artifact_id": source_artifact.artifact_id,
        "source_artifact_hash": source_artifact.content_hash,
        "calibration_snapshot_id": candidate.dataset_snapshot_id,
        "calibration_manifest_hash": candidate.dataset_manifest_hash,
        "quantization_profile_id": config.get("quantization_profile_id"),
        "algorithm": config.get("algorithm"),
        "scheme": config.get("scheme"),
        "target_runtime": config.get("target_runtime"),
        "target_hardware_profile": config.get("target_hardware_profile"),
        "approval_inheritance": False,
    }


def snapshots_form_synthetic_ablation_pair(
    session: Session,
    tenant_id: str,
    left_snapshot_id: str,
    right_snapshot_id: str,
) -> bool:
    """Return whether either snapshot is a valid direct synthetic derivative of the other."""

    snapshots = {
        item.snapshot_id: item
        for item in session.scalars(
            select(DatasetSnapshotRecord).where(
                DatasetSnapshotRecord.tenant_id == tenant_id,
                DatasetSnapshotRecord.snapshot_id.in_([left_snapshot_id, right_snapshot_id]),
            )
        )
    }
    if len(snapshots) != 2:
        return False
    left = snapshots[left_snapshot_id]
    right = snapshots[right_snapshot_id]
    if left.base_snapshot_id == right.snapshot_id:
        _snapshot_ablation_context(left, right)
        return True
    if right.base_snapshot_id == left.snapshot_id:
        _snapshot_ablation_context(right, left)
        return True
    return False


def synthetic_ablation_context(
    session: Session,
    tenant_id: str,
    candidate: TrainingExperimentRecord,
    baseline: TrainingExperimentRecord,
) -> dict[str, Any] | None:
    """Build immutable evidence when candidate is derived from the real-data baseline."""

    if candidate.method != "VLM" or baseline.method != "VLM":
        return None
    snapshots = {
        item.snapshot_id: item
        for item in session.scalars(
            select(DatasetSnapshotRecord).where(
                DatasetSnapshotRecord.tenant_id == tenant_id,
                DatasetSnapshotRecord.snapshot_id.in_(
                    [candidate.dataset_snapshot_id, baseline.dataset_snapshot_id]
                ),
            )
        )
    }
    if len(snapshots) != 2:
        return None
    candidate_snapshot = snapshots[candidate.dataset_snapshot_id]
    baseline_snapshot = snapshots[baseline.dataset_snapshot_id]
    if candidate_snapshot.base_snapshot_id != baseline_snapshot.snapshot_id:
        return None
    context = _snapshot_ablation_context(candidate_snapshot, baseline_snapshot)
    if (
        candidate.dataset_manifest_hash != candidate_snapshot.manifest_hash
        or baseline.dataset_manifest_hash != baseline_snapshot.manifest_hash
    ):
        raise SyntheticAblationContractError("snapshot_manifest_binding_changed")
    return context


def _snapshot_ablation_context(
    candidate: DatasetSnapshotRecord,
    baseline: DatasetSnapshotRecord,
) -> dict[str, Any]:
    origins = candidate.sample_origin_counts
    synthetic_splits = candidate.synthetic_split_counts
    if (
        candidate.augmentation_contract_version != SYNTHETIC_AUGMENTATION_CONTRACT
        or candidate.lineage_status != "CONFIRMED"
        or not candidate.training_eligible
        or baseline.base_snapshot_id is not None
        or baseline.augmentation_contract_version is not None
        or not isinstance(origins, dict)
        or set(origins) != {"real", "synthetic"}
        or not isinstance(synthetic_splits, dict)
        or set(synthetic_splits) != {"train", "validation", "test"}
    ):
        raise SyntheticAblationContractError("snapshot_derivation_contract_is_invalid")
    values = [*origins.values(), *synthetic_splits.values()]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise SyntheticAblationContractError("snapshot_origin_counts_are_invalid")
    synthetic_count = origins["synthetic"]
    if (
        synthetic_count <= 0
        or origins["real"] != baseline.row_count
        or candidate.row_count != baseline.row_count + synthetic_count
        or synthetic_splits
        != {"train": synthetic_count, "validation": 0, "test": 0}
        or candidate.split_counts.get("train")
        != baseline.split_counts.get("train", 0) + synthetic_count
        or candidate.split_counts.get("validation") != baseline.split_counts.get("validation")
        or candidate.split_counts.get("test") != baseline.split_counts.get("test")
        or candidate.source_work_order_ids != baseline.source_work_order_ids
    ):
        raise SyntheticAblationContractError("snapshot_split_or_origin_binding_is_invalid")
    baseline_origins = baseline.sample_origin_counts
    if baseline_origins not in (None, {"real": baseline.row_count, "synthetic": 0}):
        raise SyntheticAblationContractError("baseline_snapshot_must_contain_real_samples_only")
    if not candidate.manifest_hash or not baseline.manifest_hash:
        raise SyntheticAblationContractError("snapshot_manifest_is_not_published")
    return {
        "augmentation_contract_version": candidate.augmentation_contract_version,
        "baseline_snapshot_id": baseline.snapshot_id,
        "baseline_manifest_hash": baseline.manifest_hash,
        "candidate_snapshot_id": candidate.snapshot_id,
        "candidate_manifest_hash": candidate.manifest_hash,
        "sample_origin_counts": dict(origins),
        "synthetic_split_counts": dict(synthetic_splits),
    }
