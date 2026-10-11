"""Integrity and semantic validation for published dataset quality reports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

from industrial_ops_agent.persistence.models import (
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
)


class DatasetQualityReportError(ValueError):
    """A quality report is missing required integrity or semantic evidence."""


@dataclass(frozen=True, slots=True)
class VerifiedDatasetQualityReport:
    contract_version: str
    pandera_validation: str
    row_count: int
    split_counts: dict[str, int]
    group_leakage_count: int
    exclusions: tuple[dict[str, str], ...]
    content_hash: str


def quality_report_artifact_is_registered(
    snapshot: DatasetSnapshotRecord,
    artifacts: Sequence[DatasetArtifactRecord],
) -> bool:
    """Return whether exactly one catalog record binds the report to the snapshot."""

    if snapshot.quality_report_key is None or len(artifacts) != 1:
        return False
    artifact = artifacts[0]
    expected_schema_digest = sha256(snapshot.contract_version.encode()).hexdigest()
    return (
        artifact.tenant_id == snapshot.tenant_id
        and artifact.snapshot_id == snapshot.snapshot_id
        and artifact.kind == "quality_report"
        and artifact.split is None
        and artifact.object_key == snapshot.quality_report_key
        and _is_sha256(artifact.content_hash)
        and artifact.size_bytes > 0
        and artifact.row_count == snapshot.row_count
        and artifact.schema_hash.removeprefix("sha256:") == expected_schema_digest
    )


def verify_dataset_quality_report(
    snapshot: DatasetSnapshotRecord,
    artifact: DatasetArtifactRecord,
    payload: bytes,
) -> VerifiedDatasetQualityReport:
    """Cross-check one report against its snapshot and immutable artifact catalog."""

    actual_digest = sha256(payload).hexdigest()
    if (
        not quality_report_artifact_is_registered(snapshot, (artifact,))
        or artifact.content_hash.removeprefix("sha256:") != actual_digest
        or artifact.size_bytes != len(payload)
    ):
        raise DatasetQualityReportError("dataset_quality_report_integrity_failed")
    try:
        document = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DatasetQualityReportError("dataset_quality_report_integrity_failed") from exc
    if not isinstance(document, dict):
        raise DatasetQualityReportError("dataset_quality_report_integrity_failed")
    split_counts = _integer_mapping(document.get("split_counts"))
    exclusions = _exclusions(document.get("exclusions"))
    row_count = _integer(document.get("row_count"))
    group_leakage_count = _integer(document.get("group_leakage_count"))
    if (
        document.get("contract_version") != snapshot.contract_version
        or document.get("pandera_validation") != "PASSED"
        or row_count != snapshot.row_count
        or split_counts != snapshot.split_counts
        or group_leakage_count != 0
    ):
        raise DatasetQualityReportError("dataset_quality_report_integrity_failed")
    return VerifiedDatasetQualityReport(
        contract_version=snapshot.contract_version,
        pandera_validation="PASSED",
        row_count=row_count,
        split_counts=split_counts,
        group_leakage_count=group_leakage_count,
        exclusions=exclusions,
        content_hash=f"sha256:{actual_digest}",
    )


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetQualityReportError("dataset_quality_report_integrity_failed")
    return cast(int, value)


def _integer_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DatasetQualityReportError("dataset_quality_report_integrity_failed")
    return {key: _integer(item) for key, item in value.items()}


def _exclusions(value: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list):
        raise DatasetQualityReportError("dataset_quality_report_integrity_failed")
    exclusions: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or not all(
            isinstance(key, str) and isinstance(field, str) for key, field in item.items()
        ):
            raise DatasetQualityReportError("dataset_quality_report_integrity_failed")
        exclusions.append(dict(item))
    return tuple(exclusions)


def _is_sha256(value: str) -> bool:
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdefABCDEF" for character in digest)
