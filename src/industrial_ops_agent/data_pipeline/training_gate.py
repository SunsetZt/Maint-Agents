"""Shared fail-closed eligibility gate for governed training snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from industrial_ops_agent.data_pipeline.quality import quality_report_artifact_is_registered
from industrial_ops_agent.persistence.models import (
    DataLineageRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
)

DatasetTrainingBlockerCode = Literal[
    "snapshot_not_candidate",
    "lineage_not_confirmed",
    "manifest_not_published",
    "quality_report_missing",
    "dataset_empty",
    "training_eligibility_not_granted",
]


def dataset_training_blockers(
    snapshot: DatasetSnapshotRecord,
    lineage: DataLineageRunRecord | None,
    quality_artifacts: Sequence[DatasetArtifactRecord],
) -> tuple[DatasetTrainingBlockerCode, ...]:
    """Return stable reasons why a snapshot cannot be consumed for training."""

    blockers: list[DatasetTrainingBlockerCode] = []
    if snapshot.status != "CANDIDATE":
        blockers.append("snapshot_not_candidate")
    if snapshot.lineage_status != "CONFIRMED" or lineage is None or lineage.status != "CONFIRMED":
        blockers.append("lineage_not_confirmed")
    if snapshot.manifest_key is None or snapshot.manifest_hash is None:
        blockers.append("manifest_not_published")
    if not quality_report_artifact_is_registered(snapshot, quality_artifacts):
        blockers.append("quality_report_missing")
    if snapshot.row_count <= 0:
        blockers.append("dataset_empty")
    if not snapshot.training_eligible and not blockers:
        blockers.append("training_eligibility_not_granted")
    return tuple(blockers)
