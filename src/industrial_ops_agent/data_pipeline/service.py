"""Atomic governed dataset snapshot builder with lineage as a publish gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID, uuid4

import pyarrow as arrow
import pyarrow.parquet as parquet
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.data_governance.service import (
    GovernanceGateDenied,
    authoritative_candidate_content,
    eligibility_gate_reason,
)
from industrial_ops_agent.data_pipeline.contracts import (
    CONTRACT_VERSION,
    DATASET_COLUMNS,
    SPLITS,
    prepare_rows,
)
from industrial_ops_agent.data_pipeline.lineage import (
    LineageEmitter,
    event_now,
)
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.domain.json import legacy_canonical_digest as _digest_json
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AnnotationRevisionRecord,
    AnnotationTaskRecord,
    CurationRunRecord,
    DataLineageRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
    DlpProcessingResultRecord,
    FeedbackCandidateRecord,
    IncidentRecord,
    MediaObjectRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    run_id: str
    snapshot_id: str | None
    status: str
    training_eligible: bool
    row_count: int
    split_counts: dict[str, int]
    input_manifest_hash: str
    manifest_key: str | None


@dataclass(frozen=True, slots=True)
class _Artifact:
    kind: str
    split: str | None
    object_key: str
    content: bytes
    content_hash: str
    row_count: int
    schema_hash: str
    source_media_id: str | None = None
    candidate_ids: tuple[str, ...] = ()
    mime_type: str | None = None

    def manifest_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "kind": self.kind,
            "split": self.split,
            "object_key": self.object_key,
            "content_hash": self.content_hash,
            "size_bytes": len(self.content),
            "row_count": self.row_count,
            "schema_hash": self.schema_hash,
        }
        if self.source_media_id is not None:
            entry.update(
                {
                    "source_media_id": self.source_media_id,
                    "candidate_ids": list(self.candidate_ids),
                    "mime_type": self.mime_type,
                }
            )
        return entry


class DatasetPipelineError(RuntimeError):
    pass


class DatasetPipelineService:
    def __init__(
        self,
        database: Database,
        store: DatasetStore,
        lineage: LineageEmitter,
        *,
        code_version: str,
        media_store: DatasetStore | None = None,
    ) -> None:
        self._database = database
        self._store = store
        self._lineage = lineage
        self._code_version = code_version
        self._media_store = media_store

    def build_snapshot(
        self,
        context: TenantContext,
        *,
        window_start: datetime,
        window_end: datetime,
        engine: str,
        reserved_run_id: str | None = None,
    ) -> DatasetBuildResult:
        if window_start >= window_end:
            raise ValueError("window_start must be before window_end")
        if engine not in {"local", "spark"}:
            raise ValueError("engine must be local or spark")
        now = datetime.now(UTC)
        # OpenLineage run identifiers are UUIDs. Use the same identifier in
        # persistence, manifests and Marquez so operators can correlate them
        # without a lossy translation layer.
        run_id = str(uuid4()) if reserved_run_id is None else str(UUID(reserved_run_id))
        snapshot_id = f"dataset-{uuid4().hex}"
        rows, exclusions = self._freeze_inputs(
            context,
            window_start=window_start,
            window_end=window_end,
            now=now,
        )
        if not rows:
            self._record_no_data_run(
                context,
                run_id=run_id,
                window_start=window_start,
                window_end=window_end,
                engine=engine,
                exclusions=exclusions,
                reason="no_eligible_inputs",
                now=now,
            )
            return DatasetBuildResult(
                run_id=run_id,
                snapshot_id=None,
                status="NO_DATA",
                training_eligible=False,
                row_count=0,
                split_counts={split: 0 for split in SPLITS},
                input_manifest_hash="sha256:unavailable",
                manifest_key=None,
            )

        prepared = prepare_rows(rows)
        published_prefix = f"tenant/{context.tenant_id}/datasets/snapshots/{snapshot_id}/published"
        media_artifacts = self._freeze_media_artifacts(
            context,
            prepared,
            published_prefix=published_prefix,
        )
        input_manifest_hash = (
            _digest_json(prepared)
            if not media_artifacts
            else _digest_json(
                {
                    "rows": prepared,
                    "media": [artifact.manifest_entry() for artifact in media_artifacts],
                }
            )
        )
        config_hash = _digest_json(
            {
                "contract_version": CONTRACT_VERSION,
                "engine": engine,
                "window_start": _iso(window_start),
                "window_end": _iso(window_end),
            }
        )
        split_counts = {split: sum(row["split"] == split for row in prepared) for split in SPLITS}
        work_order_ids = tuple(sorted({str(row["work_order_id"]) for row in prepared}))
        staging_prefix = f"tenant/{context.tenant_id}/datasets/staging/{run_id}"
        output_dataset = f"{published_prefix}/manifest.json"

        failure_stage = "LINEAGE_START"
        try:
            self._lineage.emit(
                event_now(
                    "START",
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    tenant_id=context.tenant_id,
                    source_work_order_ids=work_order_ids,
                    output_dataset=output_dataset,
                    input_manifest_hash=input_manifest_hash,
                )
            )
            failure_stage = "BUILD_AND_STAGE"
            payloads = _parquet_payloads(prepared, engine=engine)
            schema_hash = _digest_text("\0".join(DATASET_COLUMNS))
            for split, content in payloads.items():
                self._store.put_bytes(f"{staging_prefix}/{split}.parquet", content)
            quality_report = {
                "contract_version": CONTRACT_VERSION,
                "pandera_validation": "PASSED",
                "row_count": len(prepared),
                "split_counts": split_counts,
                "group_leakage_count": _group_leakage_count(prepared),
                "exclusions": exclusions,
            }
            if quality_report["group_leakage_count"]:
                raise DatasetPipelineError("near-duplicate group leakage detected")
            quality_bytes = _json_bytes(quality_report)
            self._store.put_bytes(f"{staging_prefix}/quality-report.json", quality_bytes)
            failure_stage = "REVALIDATE_INPUTS"
            self._revalidate_inputs(context, prepared, now=datetime.now(UTC))
            self._revalidate_media_catalog(context, media_artifacts)
            failure_stage = "LINEAGE_COMPLETE"
            self._lineage.emit(
                event_now(
                    "COMPLETE",
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    tenant_id=context.tenant_id,
                    source_work_order_ids=work_order_ids,
                    output_dataset=output_dataset,
                    input_manifest_hash=input_manifest_hash,
                )
            )
        except Exception as exc:
            lineage_pending = _is_lineage_failure(exc)
            _log_build_failure(context, run_id, failure_stage, exc, lineage_pending)
            reason = _safe_reason(exc)
            self._record_pending_or_failed(
                context,
                run_id=run_id,
                snapshot_id=snapshot_id,
                window_start=window_start,
                window_end=window_end,
                engine=engine,
                config_hash=config_hash,
                input_manifest_hash=input_manifest_hash,
                input_count=len(prepared),
                exclusions=exclusions,
                split_counts=split_counts,
                work_order_ids=work_order_ids,
                output_dataset=output_dataset,
                reason=reason,
                lineage_pending=lineage_pending,
                now=now,
            )
            if lineage_pending:
                return DatasetBuildResult(
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    status="LINEAGE_PENDING",
                    training_eligible=False,
                    row_count=len(prepared),
                    split_counts=split_counts,
                    input_manifest_hash=input_manifest_hash,
                    manifest_key=None,
                )
            raise DatasetPipelineError(reason) from exc

        artifacts: list[_Artifact] = []
        failure_stage = "PUBLISH_OBJECTS"
        try:
            for split in SPLITS:
                key = f"{published_prefix}/{split}.parquet"
                content = payloads[split]
                self._store.put_bytes(key, content)
                artifacts.append(
                    _Artifact(
                        kind="parquet",
                        split=split,
                        object_key=key,
                        content=content,
                        content_hash=_digest_bytes(content),
                        row_count=split_counts[split],
                        schema_hash=schema_hash,
                    )
                )
            quality_key = f"{published_prefix}/quality-report.json"
            self._store.put_bytes(quality_key, quality_bytes)
            artifacts.append(
                _Artifact(
                    kind="quality_report",
                    split=None,
                    object_key=quality_key,
                    content=quality_bytes,
                    content_hash=_digest_bytes(quality_bytes),
                    row_count=len(prepared),
                    schema_hash=_digest_text(CONTRACT_VERSION),
                )
            )
            for artifact in media_artifacts:
                self._store.put_bytes(artifact.object_key, artifact.content)
                artifacts.append(artifact)
            manifest = {
                "snapshot_id": snapshot_id,
                "run_id": run_id,
                "tenant_id": context.tenant_id,
                "contract_version": CONTRACT_VERSION,
                "code_version": self._code_version,
                "input_manifest_hash": input_manifest_hash,
                "window_start": _iso(window_start),
                "window_end": _iso(window_end),
                "row_count": len(prepared),
                "split_counts": split_counts,
                "source_work_order_ids": list(work_order_ids),
                "lineage_status": "CONFIRMED",
                "artifacts": [artifact.manifest_entry() for artifact in artifacts],
            }
            manifest_bytes = _json_bytes(manifest)
            manifest_key = f"{published_prefix}/manifest.json"
            self._store.put_bytes(manifest_key, manifest_bytes)
            failure_stage = "PUBLISH_RECORDS"
            self._record_published(
                context,
                run_id=run_id,
                snapshot_id=snapshot_id,
                window_start=window_start,
                window_end=window_end,
                engine=engine,
                config_hash=config_hash,
                input_manifest_hash=input_manifest_hash,
                exclusions=exclusions,
                manifest_key=manifest_key,
                manifest_hash=_digest_bytes(manifest_bytes),
                quality_key=quality_key,
                artifacts=artifacts,
                split_counts=split_counts,
                work_order_ids=work_order_ids,
                output_dataset=output_dataset,
                now=now,
            )
        except Exception as exc:
            _log_build_failure(context, run_id, failure_stage, exc, False)
            self._record_failed_run(
                context,
                run_id=run_id,
                window_start=window_start,
                window_end=window_end,
                engine=engine,
                exclusions=exclusions,
                reason=_safe_reason(exc),
                now=now,
                config_hash=config_hash,
                input_manifest_hash=input_manifest_hash,
                input_count=len(prepared),
            )
            raise DatasetPipelineError(_safe_reason(exc)) from exc

        return DatasetBuildResult(
            run_id=run_id,
            snapshot_id=snapshot_id,
            status="CANDIDATE",
            training_eligible=True,
            row_count=len(prepared),
            split_counts=split_counts,
            input_manifest_hash=input_manifest_hash,
            manifest_key=manifest_key,
        )

    def _freeze_media_artifacts(
        self,
        context: TenantContext,
        rows: list[dict[str, Any]],
        *,
        published_prefix: str,
    ) -> list[_Artifact]:
        references: dict[str, dict[str, Any]] = {}
        for row in rows:
            labels = json.loads(str(row["labels_json"]))
            contracts = (
                [
                    labels.get("_multimodal_training"),
                    labels.get("_vlm_evaluation"),
                    labels.get("_asr_evaluation"),
                ]
                if isinstance(labels, dict)
                else []
            )
            active_contracts = [contract for contract in contracts if contract is not None]
            if len(active_contracts) > 1:
                raise DatasetPipelineError("multimodal_contract_binding_is_ambiguous")
            contract = active_contracts[0] if active_contracts else None
            if contract is None:
                continue
            if (
                not isinstance(contract, dict)
                or contract.get("review_status") != "APPROVED"
                or contract.get("modality") not in {"image", "audio"}
                or not isinstance(contract.get("media_id"), str)
                or not contract["media_id"].strip()
            ):
                raise DatasetPipelineError("multimodal_training_contract_is_invalid")
            media_id = contract["media_id"].strip()
            existing = references.setdefault(
                media_id,
                {
                    "candidate_ids": [],
                    "incident_ids": set(),
                    "splits": set(),
                    "modality": contract["modality"],
                },
            )
            if existing["modality"] != contract["modality"]:
                raise DatasetPipelineError("media_modality_binding_is_ambiguous")
            existing["candidate_ids"].append(str(row["candidate_id"]))
            existing["incident_ids"].add(str(row["incident_id"]))
            existing["splits"].add(str(row["split"]))

        if not references:
            return []
        if self._media_store is None:
            raise DatasetPipelineError("media_store_is_required_for_multimodal_snapshot")

        artifacts: list[_Artifact] = []
        with self._database.transaction(context) as session:
            for media_id, binding in sorted(references.items()):
                if len(binding["splits"]) != 1:
                    raise DatasetPipelineError("same_media_cannot_cross_dataset_splits")
                media = session.scalar(
                    select(MediaObjectRecord).where(
                        MediaObjectRecord.tenant_id == context.tenant_id,
                        MediaObjectRecord.media_id == media_id,
                    )
                )
                incidents = list(
                    session.scalars(
                        select(IncidentRecord).where(
                            IncidentRecord.tenant_id == context.tenant_id,
                            IncidentRecord.incident_id.in_(binding["incident_ids"]),
                        )
                    )
                )
                if (
                    media is None
                    or media.scan_state != "CLEAN"
                    or not media.clean_key
                    or len(incidents) != len(binding["incident_ids"])
                    or any(incident.source_draft_id != media.draft_id for incident in incidents)
                ):
                    raise DatasetPipelineError("multimodal_source_is_not_clean_or_bound")
                mime_type = str(media.detected_mime or "")
                expected_prefix = "image/" if binding["modality"] == "image" else "audio/"
                if not mime_type.startswith(expected_prefix):
                    raise DatasetPipelineError("multimodal_source_mime_does_not_match_modality")
                content = self._media_store.get_bytes(media.clean_key)
                if (
                    len(content) != media.size_bytes
                    or sha256(content).hexdigest() != media.content_hash.removeprefix("sha256:")
                ):
                    raise DatasetPipelineError("multimodal_source_integrity_failed")
                extension = {
                    "image/png": "png",
                    "image/jpeg": "jpg",
                    "audio/wav": "wav",
                    "audio/flac": "flac",
                }.get(mime_type)
                if extension is None:
                    raise DatasetPipelineError("multimodal_source_mime_is_not_trainable")
                split = next(iter(binding["splits"]))
                media_key = sha256(media.media_id.encode()).hexdigest()[:16]
                content_digest = media.content_hash.removeprefix("sha256:")
                key = (
                    f"{published_prefix}/media/{split}/"
                    f"{media_key}-{content_digest}.{extension}"
                )
                artifacts.append(
                    _Artifact(
                        kind=f"media_{binding['modality']}",
                        split=split,
                        object_key=key,
                        content=content,
                        content_hash=_digest_bytes(content),
                        row_count=len(binding["candidate_ids"]),
                        schema_hash=_digest_text("industrial-governed-media-v1"),
                        source_media_id=media.media_id,
                        candidate_ids=tuple(sorted(binding["candidate_ids"])),
                        mime_type=mime_type,
                    )
                )
        return artifacts

    def _revalidate_media_catalog(
        self,
        context: TenantContext,
        artifacts: list[_Artifact],
    ) -> None:
        if not artifacts:
            return
        with self._database.transaction(context) as session:
            for artifact in artifacts:
                media = session.scalar(
                    select(MediaObjectRecord).where(
                        MediaObjectRecord.tenant_id == context.tenant_id,
                        MediaObjectRecord.media_id == artifact.source_media_id,
                    )
                )
                if (
                    media is None
                    or media.scan_state != "CLEAN"
                    or media.content_hash.removeprefix("sha256:")
                    != artifact.content_hash.removeprefix("sha256:")
                    or media.size_bytes != len(artifact.content)
                    or media.detected_mime != artifact.mime_type
                ):
                    raise DatasetPipelineError("multimodal_source_changed_before_publish")

    def _freeze_inputs(
        self,
        context: TenantContext,
        *,
        window_start: datetime,
        window_end: datetime,
        now: datetime,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        rows: list[dict[str, Any]] = []
        exclusions: list[dict[str, str]] = []
        with self._database.transaction(context) as session:
            candidates = list(
                session.scalars(
                    select(FeedbackCandidateRecord)
                    .where(
                        FeedbackCandidateRecord.tenant_id == context.tenant_id,
                        FeedbackCandidateRecord.created_at >= window_start,
                        FeedbackCandidateRecord.created_at < window_end,
                    )
                    .order_by(FeedbackCandidateRecord.candidate_id)
                )
            )
            for candidate in candidates:
                reason = eligibility_gate_reason(candidate, now)
                if reason is None and candidate.status != "READY_FOR_CURATION":
                    reason = "annotation_not_approved"
                governed_source_hash = candidate.source_content_hash
                if candidate.lineage_origin.get("consumer_name"):
                    try:
                        _, governed_source_hash = authoritative_candidate_content(
                            session,
                            candidate,
                        )
                    except GovernanceGateDenied:
                        reason = "authoritative_source_missing_or_changed"
                dlp = session.scalar(
                    select(DlpProcessingResultRecord)
                    .where(
                        DlpProcessingResultRecord.tenant_id == context.tenant_id,
                        DlpProcessingResultRecord.candidate_id == candidate.candidate_id,
                        DlpProcessingResultRecord.status == "PASSED",
                    )
                    .order_by(DlpProcessingResultRecord.created_at.desc())
                )
                if reason is None and (
                    dlp is None
                    or dlp.residual_entity_types
                    or dlp.source_content_hash != governed_source_hash
                ):
                    reason = "dlp_not_current_or_approved"
                task = session.scalar(
                    select(AnnotationTaskRecord)
                    .where(
                        AnnotationTaskRecord.tenant_id == context.tenant_id,
                        AnnotationTaskRecord.candidate_id == candidate.candidate_id,
                        AnnotationTaskRecord.status == "APPROVED",
                        AnnotationTaskRecord.consistency_status == "CONSENSUS",
                    )
                    .order_by(AnnotationTaskRecord.updated_at.desc())
                )
                if reason is None and (task is None or task.dlp_result_id != dlp.dlp_result_id):
                    reason = "annotation_not_current_or_approved"
                revision = None
                if task is not None:
                    revision = session.scalar(
                        select(AnnotationRevisionRecord)
                        .where(
                            AnnotationRevisionRecord.tenant_id == context.tenant_id,
                            AnnotationRevisionRecord.task_id == task.task_id,
                            AnnotationRevisionRecord.task_payload_hash == task.payload_hash,
                        )
                        .order_by(AnnotationRevisionRecord.submitted_at.desc())
                    )
                if reason is None and revision is None:
                    reason = "approved_annotation_revision_missing"
                if reason is not None:
                    exclusions.append({"candidate_id": candidate.candidate_id, "reason": reason})
                    continue
                incident = session.scalar(
                    select(IncidentRecord).where(
                        IncidentRecord.tenant_id == context.tenant_id,
                        IncidentRecord.incident_id == candidate.incident_id,
                    )
                )
                device_id = str(candidate.lineage_origin.get("asset_id") or "")
                if not device_id and incident is not None:
                    device_id = incident.asset_id
                near_duplicate_group = str(
                    candidate.lineage_origin.get("near_duplicate_group") or dlp.content_hash
                )
                rows.append(
                    {
                        "tenant_id": context.tenant_id,
                        "candidate_id": candidate.candidate_id,
                        "candidate_version": candidate.version,
                        "work_order_id": candidate.work_order_id,
                        "incident_id": candidate.incident_id,
                        "device_id": device_id,
                        "source_event_id": candidate.source_event_id,
                        "source_content_hash": governed_source_hash,
                        "redacted_content_json": _json_text(dlp.redacted_content),
                        "labels_json": _json_text(revision.labels),
                        "lineage_json": _json_text(candidate.lineage_origin),
                        "occurred_at": _iso(candidate.created_at),
                        "near_duplicate_group": near_duplicate_group,
                    }
                )
        return rows, exclusions

    def _revalidate_inputs(
        self,
        context: TenantContext,
        rows: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> None:
        """Reject source drift between the frozen manifest and final publication."""

        with self._database.transaction(context) as session:
            for row in rows:
                candidate = session.scalar(
                    select(FeedbackCandidateRecord).where(
                        FeedbackCandidateRecord.tenant_id == context.tenant_id,
                        FeedbackCandidateRecord.candidate_id == row["candidate_id"],
                    )
                )
                governed_source_hash = None
                if candidate is not None:
                    governed_source_hash = candidate.source_content_hash
                    if candidate.lineage_origin.get("consumer_name"):
                        try:
                            _, governed_source_hash = authoritative_candidate_content(
                                session,
                                candidate,
                            )
                        except GovernanceGateDenied:
                            governed_source_hash = None
                if (
                    candidate is None
                    or candidate.version != row["candidate_version"]
                    or candidate.status != "READY_FOR_CURATION"
                    or governed_source_hash != row["source_content_hash"]
                    or eligibility_gate_reason(candidate, now) is not None
                ):
                    raise DatasetPipelineError(
                        f"governed input changed before publish: {row['candidate_id']}"
                    )

    def _record_published(
        self,
        context: TenantContext,
        *,
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
        artifacts: list[_Artifact],
        split_counts: dict[str, int],
        work_order_ids: tuple[str, ...],
        output_dataset: str,
        now: datetime,
    ) -> None:
        with self._database.transaction(context) as session:
            _add_run(
                session,
                context,
                run_id=run_id,
                window_start=window_start,
                window_end=window_end,
                engine=engine,
                code_version=self._code_version,
                config_hash=config_hash,
                input_manifest_hash=input_manifest_hash,
                input_count=sum(split_counts.values()),
                exclusions=exclusions,
                status="COMPLETED",
                failure_reason=None,
                now=now,
            )
            session.flush()
            session.add(
                DatasetSnapshotRecord(
                    snapshot_id=snapshot_id,
                    tenant_id=context.tenant_id,
                    run_id=run_id,
                    status="CANDIDATE",
                    contract_version=CONTRACT_VERSION,
                    input_manifest_hash=input_manifest_hash,
                    manifest_key=manifest_key,
                    manifest_hash=manifest_hash,
                    quality_report_key=quality_key,
                    row_count=sum(split_counts.values()),
                    split_counts=split_counts,
                    source_work_order_ids=list(work_order_ids),
                    lineage_status="CONFIRMED",
                    training_eligible=True,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            for artifact in artifacts:
                session.add(
                    DatasetArtifactRecord(
                        artifact_id="artifact-"
                        + sha256(f"{snapshot_id}\0{artifact.object_key}".encode()).hexdigest()[:32],
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
            _add_lineage(
                session,
                context,
                run_id=run_id,
                snapshot_id=snapshot_id,
                namespace=self._lineage.namespace,
                job_name=self._lineage.job_name,
                status="CONFIRMED",
                work_order_ids=work_order_ids,
                output_dataset=output_dataset,
                failure_reason=None,
                now=now,
            )

    def _record_pending_or_failed(
        self,
        context: TenantContext,
        *,
        run_id: str,
        snapshot_id: str,
        window_start: datetime,
        window_end: datetime,
        engine: str,
        config_hash: str,
        input_manifest_hash: str,
        input_count: int,
        exclusions: list[dict[str, str]],
        split_counts: dict[str, int],
        work_order_ids: tuple[str, ...],
        output_dataset: str,
        reason: str,
        lineage_pending: bool,
        now: datetime,
    ) -> None:
        if not lineage_pending:
            self._record_failed_run(
                context,
                run_id=run_id,
                window_start=window_start,
                window_end=window_end,
                engine=engine,
                exclusions=exclusions,
                reason=reason,
                now=now,
                config_hash=config_hash,
                input_manifest_hash=input_manifest_hash,
                input_count=input_count,
            )
            return
        with self._database.transaction(context) as session:
            _add_run(
                session,
                context,
                run_id=run_id,
                window_start=window_start,
                window_end=window_end,
                engine=engine,
                code_version=self._code_version,
                config_hash=config_hash,
                input_manifest_hash=input_manifest_hash,
                input_count=input_count,
                exclusions=exclusions,
                status="LINEAGE_PENDING",
                failure_reason=reason,
                now=now,
            )
            session.flush()
            session.add(
                DatasetSnapshotRecord(
                    snapshot_id=snapshot_id,
                    tenant_id=context.tenant_id,
                    run_id=run_id,
                    status="LINEAGE_PENDING",
                    contract_version=CONTRACT_VERSION,
                    input_manifest_hash=input_manifest_hash,
                    manifest_key=None,
                    manifest_hash=None,
                    quality_report_key=None,
                    row_count=input_count,
                    split_counts=split_counts,
                    source_work_order_ids=list(work_order_ids),
                    lineage_status="PENDING",
                    training_eligible=False,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            _add_lineage(
                session,
                context,
                run_id=run_id,
                snapshot_id=snapshot_id,
                namespace=self._lineage.namespace,
                job_name=self._lineage.job_name,
                status="PENDING",
                work_order_ids=work_order_ids,
                output_dataset=output_dataset,
                failure_reason=reason,
                now=now,
            )

    def _record_failed_run(
        self,
        context: TenantContext,
        *,
        run_id: str,
        window_start: datetime,
        window_end: datetime,
        engine: str,
        exclusions: list[dict[str, str]],
        reason: str,
        now: datetime,
        config_hash: str = "sha256:unavailable",
        input_manifest_hash: str = "sha256:unavailable",
        input_count: int = 0,
    ) -> None:
        with self._database.transaction(context) as session:
            _add_run(
                session,
                context,
                run_id=run_id,
                window_start=window_start,
                window_end=window_end,
                engine=engine,
                code_version=self._code_version,
                config_hash=config_hash,
                input_manifest_hash=input_manifest_hash,
                input_count=input_count,
                exclusions=exclusions,
                status="FAILED",
                failure_reason=reason,
                now=now,
            )

    def _record_no_data_run(
        self,
        context: TenantContext,
        *,
        run_id: str,
        window_start: datetime,
        window_end: datetime,
        engine: str,
        exclusions: list[dict[str, str]],
        reason: str,
        now: datetime,
    ) -> None:
        """Persist an expected empty-window outcome without creating a snapshot."""

        with self._database.transaction(context) as session:
            _add_run(
                session,
                context,
                run_id=run_id,
                window_start=window_start,
                window_end=window_end,
                engine=engine,
                code_version=self._code_version,
                config_hash="sha256:unavailable",
                input_manifest_hash="sha256:unavailable",
                input_count=0,
                exclusions=exclusions,
                status="NO_DATA",
                failure_reason=reason,
                now=now,
            )


def _add_run(
    session: Session,
    context: TenantContext,
    *,
    run_id: str,
    window_start: datetime,
    window_end: datetime,
    engine: str,
    code_version: str,
    config_hash: str,
    input_manifest_hash: str,
    input_count: int,
    exclusions: list[dict[str, str]],
    status: str,
    failure_reason: str | None,
    now: datetime,
) -> None:
    session.add(
        CurationRunRecord(
            run_id=run_id,
            tenant_id=context.tenant_id,
            window_start=window_start,
            window_end=window_end,
            engine=engine,
            contract_version=CONTRACT_VERSION,
            code_version=code_version,
            config_hash=config_hash,
            input_manifest_hash=input_manifest_hash,
            input_count=input_count,
            exclusion_report=exclusions,
            status=status,
            failure_reason=failure_reason,
            created_at=now,
            updated_at=now,
        )
    )


def _add_lineage(
    session: Session,
    context: TenantContext,
    *,
    run_id: str,
    snapshot_id: str,
    namespace: str,
    job_name: str,
    status: str,
    work_order_ids: tuple[str, ...],
    output_dataset: str,
    failure_reason: str | None,
    now: datetime,
) -> None:
    session.add(
        DataLineageRunRecord(
            lineage_id=f"lineage-{uuid4().hex}",
            tenant_id=context.tenant_id,
            run_id=run_id,
            snapshot_id=snapshot_id,
            openlineage_run_id=run_id,
            job_namespace=namespace,
            job_name=job_name,
            status=status,
            source_work_order_ids=list(work_order_ids),
            output_dataset=output_dataset,
            failure_reason=failure_reason,
            created_at=now,
            updated_at=now,
        )
    )


def _parquet_payloads(rows: list[dict[str, Any]], *, engine: str) -> dict[str, bytes]:
    if engine == "spark":
        with TemporaryDirectory(prefix="industrial-ops-spark-") as directory:
            from industrial_ops_agent.data_pipeline.spark_curate import build_spark_snapshot

            output = build_spark_snapshot(rows, Path(directory) / "snapshot")
            tables: dict[str, arrow.Table] = {}
            for split in SPLITS:
                files = sorted((output / f"split={split}").glob("*.parquet"))
                tables[split] = (
                    arrow.concat_tables([parquet.read_table(path) for path in files])
                    if files
                    else _table([])
                )
            return {split: _write_table(tables[split]) for split in SPLITS}
    return {
        split: _write_table(_table([row for row in rows if row["split"] == split]))
        for split in SPLITS
    }


def _table(rows: list[dict[str, Any]]) -> arrow.Table:
    fields = [
        arrow.field(name, arrow.int64() if name == "candidate_version" else arrow.string())
        for name in DATASET_COLUMNS
    ]
    return arrow.Table.from_pylist(rows, schema=arrow.schema(fields))


def _write_table(table: arrow.Table) -> bytes:
    target = BytesIO()
    parquet.write_table(table, target, compression="zstd")
    return target.getvalue()


def _group_leakage_count(rows: list[dict[str, Any]]) -> int:
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(str(row["group_key"]), set()).add(str(row["split"]))
    return sum(len(splits) > 1 for splits in groups.values())


def _is_lineage_failure(exc: Exception) -> bool:
    from industrial_ops_agent.data_pipeline.lineage import LineageUnavailable

    return isinstance(exc, LineageUnavailable)


def _log_build_failure(
    context: TenantContext, run_id: str, stage: str, exc: Exception, lineage_pending: bool
) -> None:
    # Reasons remain API-compatible; do not add raw exception text or payloads to logs.
    structlog.get_logger(__name__).warning(
        "dataset_pipeline_failed",
        tenant_id=context.tenant_id,
        run_id=run_id,
        failure_stage=stage,
        error_type=type(exc).__name__,
        lineage_pending=lineage_pending,
    )


def _safe_reason(exc: Exception) -> str:
    value = str(exc).strip() or type(exc).__name__
    return value[:255]


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: Any) -> bytes:
    return _json_text(value).encode()


def _digest_text(value: str) -> str:
    return _digest_bytes(value.encode())


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + sha256(value).hexdigest()


def _iso(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat()
