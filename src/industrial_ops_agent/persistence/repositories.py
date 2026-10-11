"""Tenant-scoped repositories keep an explicit filter in front of PostgreSQL RLS."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from industrial_ops_agent.persistence.models import (
    AgentCheckpointRecord,
    AgentEventRecord,
    AgentRunRecord,
    AssetRecord,
    AssetSiteLinkRecord,
    DiagnosisRunRecord,
    DraftTimelineEventRecord,
    EvidenceBundleRecord,
    IdempotencyRecord,
    IncidentDraftRecord,
    IncidentRecord,
    IndexReleaseRecord,
    MediaObjectRecord,
    RecognitionCorrectionRecord,
    RecognitionResultRecord,
    RecognitionRunRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext


class AssetRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def get(self, asset_id: str) -> AssetRecord | None:
        statement = select(AssetRecord).where(
            AssetRecord.tenant_id == self._context.tenant_id,
            AssetRecord.asset_id == asset_id,
        )
        return self._session.scalar(statement)

    def list_allowed(
        self,
        asset_ids: frozenset[str],
        site_ids: frozenset[str] = frozenset(),
    ) -> list[AssetRecord]:
        scope_conditions = []
        if asset_ids:
            scope_conditions.append(AssetRecord.asset_id.in_(asset_ids))
        if site_ids:
            site_assets = select(AssetSiteLinkRecord.asset_id).where(
                AssetSiteLinkRecord.tenant_id == self._context.tenant_id,
                AssetSiteLinkRecord.site_id.in_(site_ids),
            )
            scope_conditions.append(AssetRecord.asset_id.in_(site_assets))
        if not scope_conditions:
            return []
        statement = (
            select(AssetRecord)
            .where(
                AssetRecord.tenant_id == self._context.tenant_id,
                or_(*scope_conditions),
            )
            .order_by(AssetRecord.asset_id)
        )
        return list(self._session.scalars(statement))


class IncidentDraftRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def get(self, draft_id: str) -> IncidentDraftRecord | None:
        statement = select(IncidentDraftRecord).where(
            IncidentDraftRecord.tenant_id == self._context.tenant_id,
            IncidentDraftRecord.draft_id == draft_id,
        )
        return self._session.scalar(statement)

    def add(self, record: IncidentDraftRecord) -> None:
        self._session.add(record)

    def update_description(
        self,
        *,
        draft_id: str,
        expected_version: int,
        description: str,
        updated_at: datetime,
    ) -> bool:
        statement = (
            update(IncidentDraftRecord)
            .where(
                IncidentDraftRecord.tenant_id == self._context.tenant_id,
                IncidentDraftRecord.draft_id == draft_id,
                IncidentDraftRecord.version == expected_version,
                IncidentDraftRecord.status == "DRAFT",
            )
            .values(
                description=description,
                version=expected_version + 1,
                updated_at=updated_at,
            )
            .execution_options(synchronize_session=False)
        )
        result = self._session.connection().execute(statement)
        return result.rowcount == 1

    def submit(
        self,
        *,
        draft_id: str,
        expected_version: int,
        updated_at: datetime,
    ) -> bool:
        statement = (
            update(IncidentDraftRecord)
            .where(
                IncidentDraftRecord.tenant_id == self._context.tenant_id,
                IncidentDraftRecord.draft_id == draft_id,
                IncidentDraftRecord.version == expected_version,
                IncidentDraftRecord.status == "DRAFT",
            )
            .values(
                status="SUBMITTED",
                version=expected_version + 1,
                updated_at=updated_at,
            )
            .execution_options(synchronize_session=False)
        )
        result = self._session.connection().execute(statement)
        return result.rowcount == 1


class IdempotencyRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def get(self, *, subject_id: str, key: str) -> IdempotencyRecord | None:
        statement = select(IdempotencyRecord).where(
            IdempotencyRecord.tenant_id == self._context.tenant_id,
            IdempotencyRecord.subject_id == subject_id,
            IdempotencyRecord.key == key,
        )
        return self._session.scalar(statement)

    def add(self, record: IdempotencyRecord) -> None:
        self._session.add(record)


class DraftTimelineRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def add(self, record: DraftTimelineEventRecord) -> None:
        self._session.add(record)

    def list(self, draft_id: str) -> list[DraftTimelineEventRecord]:
        statement = (
            select(DraftTimelineEventRecord)
            .where(
                DraftTimelineEventRecord.tenant_id == self._context.tenant_id,
                DraftTimelineEventRecord.draft_id == draft_id,
            )
            .order_by(DraftTimelineEventRecord.sequence)
        )
        return list(self._session.scalars(statement))

    def next_sequence(self, draft_id: str) -> int:
        statement = select(func.coalesce(func.max(DraftTimelineEventRecord.sequence), 0) + 1).where(
            DraftTimelineEventRecord.tenant_id == self._context.tenant_id,
            DraftTimelineEventRecord.draft_id == draft_id,
        )
        return int(self._session.scalar(statement) or 1)


class MediaObjectRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def get(self, media_id: str) -> MediaObjectRecord | None:
        statement = select(MediaObjectRecord).where(
            MediaObjectRecord.tenant_id == self._context.tenant_id,
            MediaObjectRecord.media_id == media_id,
        )
        return self._session.scalar(statement)

    def add(self, record: MediaObjectRecord) -> None:
        self._session.add(record)

    def list_for_draft(self, draft_id: str) -> list[MediaObjectRecord]:
        statement = (
            select(MediaObjectRecord)
            .where(
                MediaObjectRecord.tenant_id == self._context.tenant_id,
                MediaObjectRecord.draft_id == draft_id,
            )
            .order_by(MediaObjectRecord.created_at, MediaObjectRecord.media_id)
        )
        return list(self._session.scalars(statement))

    def complete_scan(
        self,
        *,
        media_id: str,
        expected_version: int,
        scan_state: str,
        clean_key: str | None,
        content_credential_status: str,
        content_credential_report: dict[str, Any] | None,
        content_credential_digest: str | None,
        content_credential_inspected_at: datetime | None,
        updated_at: datetime,
    ) -> bool:
        statement = (
            update(MediaObjectRecord)
            .where(
                MediaObjectRecord.tenant_id == self._context.tenant_id,
                MediaObjectRecord.media_id == media_id,
                MediaObjectRecord.scan_state == "PENDING",
                MediaObjectRecord.scan_version == expected_version,
            )
            .values(
                scan_state=scan_state,
                scan_version=expected_version + 1,
                clean_key=clean_key,
                content_credential_status=content_credential_status,
                content_credential_report=content_credential_report,
                content_credential_digest=content_credential_digest,
                content_credential_inspected_at=content_credential_inspected_at,
                updated_at=updated_at,
            )
        )
        result = self._session.connection().execute(statement)
        return result.rowcount == 1


class RecognitionRunRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def get(self, recognition_run_id: str) -> RecognitionRunRecord | None:
        statement = select(RecognitionRunRecord).where(
            RecognitionRunRecord.tenant_id == self._context.tenant_id,
            RecognitionRunRecord.recognition_run_id == recognition_run_id,
        )
        return self._session.scalar(statement)

    def add(self, record: RecognitionRunRecord) -> None:
        self._session.add(record)

    def for_field_upload(
        self,
        *,
        work_order_id: str,
        field_evidence_upload_id: str,
    ) -> RecognitionRunRecord | None:
        statement = (
            select(RecognitionRunRecord)
            .where(
                RecognitionRunRecord.tenant_id == self._context.tenant_id,
                RecognitionRunRecord.context_kind == "FIELD_EVIDENCE",
                RecognitionRunRecord.work_order_id == work_order_id,
                RecognitionRunRecord.field_evidence_upload_id
                == field_evidence_upload_id,
            )
            .order_by(
                RecognitionRunRecord.created_at.desc(),
                RecognitionRunRecord.recognition_run_id.desc(),
            )
            .limit(1)
        )
        return self._session.scalar(statement)

    def transition(
        self,
        *,
        recognition_run_id: str,
        expected_version: int,
        expected_status: str,
        status: str,
        updated_at: datetime,
        evidence_bundle_id: str | None = None,
        failure_reason: str | None = None,
    ) -> bool:
        statement = (
            update(RecognitionRunRecord)
            .where(
                RecognitionRunRecord.tenant_id == self._context.tenant_id,
                RecognitionRunRecord.recognition_run_id == recognition_run_id,
                RecognitionRunRecord.version == expected_version,
                RecognitionRunRecord.status == expected_status,
            )
            .values(
                status=status,
                evidence_bundle_id=evidence_bundle_id,
                failure_reason=failure_reason,
                version=expected_version + 1,
                updated_at=updated_at,
            )
            .execution_options(synchronize_session=False)
        )
        result = self._session.connection().execute(statement)
        return result.rowcount == 1


class EvidenceBundleRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def get(self, bundle_id: str) -> EvidenceBundleRecord | None:
        statement = select(EvidenceBundleRecord).where(
            EvidenceBundleRecord.tenant_id == self._context.tenant_id,
            EvidenceBundleRecord.bundle_id == bundle_id,
        )
        return self._session.scalar(statement)

    def latest_for_draft(self, draft_id: str) -> EvidenceBundleRecord | None:
        statement = (
            select(EvidenceBundleRecord)
            .where(
                EvidenceBundleRecord.tenant_id == self._context.tenant_id,
                EvidenceBundleRecord.draft_id == draft_id,
                EvidenceBundleRecord.context_kind == "INCIDENT_DRAFT",
            )
            .order_by(EvidenceBundleRecord.created_at.desc(), EvidenceBundleRecord.bundle_id.desc())
            .limit(1)
        )
        return self._session.scalar(statement)

    def for_field_upload(
        self,
        *,
        work_order_id: str,
        field_evidence_upload_id: str,
    ) -> EvidenceBundleRecord | None:
        statement = (
            select(EvidenceBundleRecord)
            .where(
                EvidenceBundleRecord.tenant_id == self._context.tenant_id,
                EvidenceBundleRecord.context_kind == "FIELD_EVIDENCE",
                EvidenceBundleRecord.work_order_id == work_order_id,
                EvidenceBundleRecord.field_evidence_upload_id
                == field_evidence_upload_id,
            )
            .order_by(
                EvidenceBundleRecord.created_at.desc(),
                EvidenceBundleRecord.bundle_id.desc(),
            )
            .limit(1)
        )
        return self._session.scalar(statement)

    def add(self, record: EvidenceBundleRecord) -> None:
        self._session.add(record)

    def add_result(self, record: RecognitionResultRecord) -> None:
        self._session.add(record)

    def list_results(self, bundle_id: str) -> list[RecognitionResultRecord]:
        statement = (
            select(RecognitionResultRecord)
            .where(
                RecognitionResultRecord.tenant_id == self._context.tenant_id,
                RecognitionResultRecord.bundle_id == bundle_id,
            )
            .order_by(RecognitionResultRecord.result_type, RecognitionResultRecord.ordinal)
        )
        return list(self._session.scalars(statement))

    def add_correction(self, record: RecognitionCorrectionRecord) -> None:
        self._session.add(record)

    def list_corrections(self, bundle_id: str) -> list[RecognitionCorrectionRecord]:
        statement = (
            select(RecognitionCorrectionRecord)
            .where(
                RecognitionCorrectionRecord.tenant_id == self._context.tenant_id,
                RecognitionCorrectionRecord.bundle_id == bundle_id,
            )
            .order_by(
                RecognitionCorrectionRecord.evidence_version,
                RecognitionCorrectionRecord.created_at,
                RecognitionCorrectionRecord.correction_id,
            )
        )
        return list(self._session.scalars(statement))

    def confirm(
        self,
        *,
        bundle_id: str,
        expected_version: int,
        updated_at: datetime,
        automation_eligible: bool,
        automation_blockers: tuple[str, ...],
    ) -> bool:
        statement = (
            update(EvidenceBundleRecord)
            .where(
                EvidenceBundleRecord.tenant_id == self._context.tenant_id,
                EvidenceBundleRecord.bundle_id == bundle_id,
                EvidenceBundleRecord.version == expected_version,
                EvidenceBundleRecord.status == "READY_FOR_CONFIRMATION",
            )
            .values(
                status="CONFIRMED",
                version=expected_version + 1,
                updated_at=updated_at,
                automation_eligible=automation_eligible,
                automation_blockers_json=list(automation_blockers),
            )
            .execution_options(synchronize_session=False)
        )
        result = self._session.connection().execute(statement)
        return result.rowcount == 1


class IncidentRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def get(self, incident_id: str) -> IncidentRecord | None:
        statement = select(IncidentRecord).where(
            IncidentRecord.tenant_id == self._context.tenant_id,
            IncidentRecord.incident_id == incident_id,
        )
        return self._session.scalar(statement)

    def get_by_draft(self, draft_id: str) -> IncidentRecord | None:
        statement = select(IncidentRecord).where(
            IncidentRecord.tenant_id == self._context.tenant_id,
            IncidentRecord.source_draft_id == draft_id,
        )
        return self._session.scalar(statement)

    def add(self, record: IncidentRecord) -> None:
        self._session.add(record)

    def update_status(
        self,
        *,
        incident_id: str,
        expected_version: int,
        current_status: str,
        next_status: str,
        updated_at: datetime,
    ) -> bool:
        statement = (
            update(IncidentRecord)
            .where(
                IncidentRecord.tenant_id == self._context.tenant_id,
                IncidentRecord.incident_id == incident_id,
                IncidentRecord.version == expected_version,
                IncidentRecord.status == current_status,
            )
            .values(
                status=next_status,
                version=expected_version + 1,
                updated_at=updated_at,
            )
            .execution_options(synchronize_session=False)
        )
        result = self._session.connection().execute(statement)
        return result.rowcount == 1


class IndexReleaseRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def active(self) -> IndexReleaseRecord | None:
        statement = select(IndexReleaseRecord).where(
            IndexReleaseRecord.tenant_id == self._context.tenant_id,
            IndexReleaseRecord.status == "PUBLISHED",
            IndexReleaseRecord.is_active.is_(True),
        )
        return self._session.scalar(statement)


class DiagnosisRunRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def add(self, record: DiagnosisRunRecord) -> None:
        self._session.add(record)

    def get(self, diagnosis_run_id: str) -> DiagnosisRunRecord | None:
        statement = select(DiagnosisRunRecord).where(
            DiagnosisRunRecord.tenant_id == self._context.tenant_id,
            DiagnosisRunRecord.diagnosis_run_id == diagnosis_run_id,
        )
        return self._session.scalar(statement)

    def get_by_agent_run(self, agent_run_id: str) -> DiagnosisRunRecord | None:
        statement = select(DiagnosisRunRecord).where(
            DiagnosisRunRecord.tenant_id == self._context.tenant_id,
            DiagnosisRunRecord.agent_run_id == agent_run_id,
        )
        return self._session.scalar(statement)

    def list_for_incident(self, incident_id: str) -> list[DiagnosisRunRecord]:
        statement = (
            select(DiagnosisRunRecord)
            .where(
                DiagnosisRunRecord.tenant_id == self._context.tenant_id,
                DiagnosisRunRecord.incident_id == incident_id,
            )
            .order_by(DiagnosisRunRecord.created_at, DiagnosisRunRecord.diagnosis_run_id)
        )
        return list(self._session.scalars(statement))


class AgentRunRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def add(self, record: AgentRunRecord) -> None:
        self._session.add(record)

    def get(self, agent_run_id: str) -> AgentRunRecord | None:
        statement = select(AgentRunRecord).where(
            AgentRunRecord.tenant_id == self._context.tenant_id,
            AgentRunRecord.agent_run_id == agent_run_id,
        )
        return self._session.scalar(statement)


class AgentEventRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def add(self, record: AgentEventRecord) -> None:
        self._session.add(record)

    def list_after(self, agent_run_id: str, sequence: int) -> list[AgentEventRecord]:
        statement = (
            select(AgentEventRecord)
            .where(
                AgentEventRecord.tenant_id == self._context.tenant_id,
                AgentEventRecord.agent_run_id == agent_run_id,
                AgentEventRecord.sequence > sequence,
            )
            .order_by(AgentEventRecord.sequence)
        )
        return list(self._session.scalars(statement))


class AgentCheckpointRepository:
    def __init__(self, session: Session, context: TenantContext) -> None:
        self._session = session
        self._context = context

    def add(self, record: AgentCheckpointRecord) -> None:
        self._session.add(record)

    def latest(self, agent_run_id: str) -> AgentCheckpointRecord | None:
        statement = (
            select(AgentCheckpointRecord)
            .where(
                AgentCheckpointRecord.tenant_id == self._context.tenant_id,
                AgentCheckpointRecord.agent_run_id == agent_run_id,
            )
            .order_by(AgentCheckpointRecord.sequence.desc())
            .limit(1)
        )
        return self._session.scalar(statement)
