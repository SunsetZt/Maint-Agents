"""Shared immutable snapshot records for telemetry classification and RUL datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

from industrial_ops_agent.data_pipeline.lineage import LineageEmitter
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CurationRunRecord,
    DataLineageRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext


@dataclass(frozen=True, slots=True)
class SnapshotArtifact:
    kind: str
    split: str | None
    object_key: str
    content: bytes
    row_count: int
    schema_hash: str

    @property
    def content_hash(self) -> str:
        return digest_bytes(self.content)

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "split": self.split,
            "object_key": self.object_key,
            "content_hash": self.content_hash,
            "size_bytes": len(self.content),
            "row_count": self.row_count,
            "schema_hash": self.schema_hash,
        }


def record_published(
    database: Database,
    lineage: LineageEmitter,
    context: TenantContext,
    *,
    contract_version: str,
    code_version: str,
    run_id: str,
    snapshot_id: str,
    window_start: datetime,
    window_end: datetime,
    engine: str,
    config_hash: str,
    input_manifest_hash: str,
    exclusions: list[dict[str, str]],
    manifest_key: str,
    manifest_hash: str,
    quality_key: str,
    artifacts: list[SnapshotArtifact],
    split_counts: dict[str, int],
    row_count: int,
    training_eligible: bool,
    output_dataset: str,
    now: datetime,
) -> None:
    with database.transaction(context) as session:
        session.add(
            CurationRunRecord(
                run_id=run_id,
                tenant_id=context.tenant_id,
                window_start=window_start,
                window_end=window_end,
                engine=engine,
                contract_version=contract_version,
                code_version=code_version,
                config_hash=config_hash,
                input_manifest_hash=input_manifest_hash,
                input_count=row_count,
                exclusion_report=exclusions,
                status="COMPLETED",
                failure_reason=None,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            DatasetSnapshotRecord(
                snapshot_id=snapshot_id,
                tenant_id=context.tenant_id,
                run_id=run_id,
                status="CANDIDATE",
                contract_version=contract_version,
                input_manifest_hash=input_manifest_hash,
                manifest_key=manifest_key,
                manifest_hash=manifest_hash,
                quality_report_key=quality_key,
                row_count=row_count,
                split_counts=split_counts,
                source_work_order_ids=[],
                lineage_status="CONFIRMED",
                training_eligible=training_eligible,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        for artifact in artifacts:
            session.add(
                DatasetArtifactRecord(
                    artifact_id=f"artifact-{uuid4().hex}",
                    tenant_id=context.tenant_id,
                    snapshot_id=snapshot_id,
                    kind=artifact.kind,
                    split=artifact.split,
                    object_key=artifact.object_key,
                    content_hash=artifact.content_hash,
                    size_bytes=len(artifact.content),
                    row_count=artifact.row_count,
                    schema_hash=artifact.schema_hash,
                    created_at=now,
                    updated_at=now,
                )
            )
        session.add(
            DataLineageRunRecord(
                lineage_id=f"lineage-{uuid4().hex}",
                tenant_id=context.tenant_id,
                run_id=run_id,
                snapshot_id=snapshot_id,
                openlineage_run_id=run_id,
                job_namespace=lineage.namespace,
                job_name=lineage.job_name,
                status="CONFIRMED",
                source_work_order_ids=[],
                output_dataset=output_dataset,
                failure_reason=None,
                created_at=now,
                updated_at=now,
            )
        )


def record_failed(
    database: Database,
    context: TenantContext,
    *,
    contract_version: str,
    code_version: str,
    run_id: str,
    window_start: datetime,
    window_end: datetime,
    engine: str,
    input_manifest_hash: str,
    exclusions: list[dict[str, str]],
    reason: str,
    now: datetime,
) -> None:
    with database.transaction(context) as session:
        session.add(
            CurationRunRecord(
                run_id=run_id,
                tenant_id=context.tenant_id,
                window_start=window_start,
                window_end=window_end,
                engine=engine,
                contract_version=contract_version,
                code_version=code_version,
                config_hash=digest_text(f"{engine}:{window_start}:{window_end}"),
                input_manifest_hash=input_manifest_hash,
                input_count=0,
                exclusion_report=exclusions,
                status="FAILED",
                failure_reason=reason,
                created_at=now,
                updated_at=now,
            )
        )


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_bytes(value: Any) -> bytes:
    return json_text(value).encode()


def digest_json(value: Any) -> str:
    return digest_bytes(json_bytes(value))


def digest_text(value: str) -> str:
    return "sha256:" + sha256(value.encode()).hexdigest()


def digest_bytes(value: bytes) -> str:
    return "sha256:" + sha256(value).hexdigest()
