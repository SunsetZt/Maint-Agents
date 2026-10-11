"""Governed field-media upload bindings and safe scan-state projections."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.maintenance_planning.review_isolation import work_order_read_transaction
from industrial_ops_agent.media.service import MediaService, TenantObjectStore
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetSiteLinkRecord,
    EvidenceBundleRecord,
    IncidentRecord,
    MediaObjectRecord,
    RecognitionRunRecord,
    WorkOrderFieldEvidenceUploadRecord,
    WorkOrderRecord,
)
from industrial_ops_agent.workorders.service import WorkOrderConflict, WorkOrderNotVisible

FIELD_EVIDENCE_EXECUTABLE_STATUSES = frozenset({"ACCEPTED", "IN_PROGRESS", "ON_HOLD"})


class FieldEvidenceValidationError(Exception):
    """A client-supplied content binding cannot be verified."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class FieldEvidenceUploadView:
    evidence_upload_id: str
    work_order_id: str
    media_id: str
    uploaded_by_subject_id: str
    client_operation_id: str
    content_hash: str
    declared_mime: str
    detected_mime: str | None
    size_bytes: int
    work_order_version: int
    scan_state: str
    scan_version: int
    content_credential_status: str
    ready_to_attach: bool
    legal_actions: list[str]
    recognition_run_id: str | None
    recognition_status: str | None
    recognition_bundle_id: str | None
    recognition_bundle_status: str | None
    recognition_bundle_version: int | None
    recognition_failure_summary: str | None
    occurred_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class FieldEvidenceUploadResult:
    upload: FieldEvidenceUploadView
    created: bool


@dataclass(frozen=True, slots=True)
class _FieldEvidenceContext:
    work: WorkOrderRecord
    incident: IncidentRecord
    site_id: str | None


class FieldEvidenceService:
    """Authorize WorkOrder-bound media before entering the shared media pipeline."""

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        object_store: TenantObjectStore,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._object_store = object_store

    async def upload(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_work_order_version: int,
        client_operation_id: str,
        expected_content_hash: str,
        declared_mime: str,
        content: bytes,
        request_id: str,
    ) -> FieldEvidenceUploadResult:
        normalized_mime = declared_mime.split(";", 1)[0].strip().casefold()
        actual_hash = sha256(content).hexdigest()
        if expected_content_hash.casefold() != f"sha256:{actual_hash}":
            raise FieldEvidenceValidationError("field_evidence_content_hash_mismatch")

        def prepare() -> FieldEvidenceUploadResult | str:
            with work_order_read_transaction(
                self._database, identity, work_order_id, request_id=request_id
            ) as session:
                context = self._load_context(
                    session,
                    identity,
                    work_order_id,
                    request_id=request_id,
                    for_update=True,
                )
                existing = self._find_operation(
                    session,
                    identity,
                    work_order_id,
                    client_operation_id,
                )
                if existing is not None:
                    self._require_same_request(
                        existing,
                        context,
                        identity,
                        expected_work_order_version,
                        actual_hash,
                        normalized_mime,
                        len(content),
                    )
                    bound_media = self._load_bound_media(session, identity, existing)
                    return FieldEvidenceUploadResult(
                        upload=_view(
                            existing,
                            bound_media,
                            acting_subject_id=identity.subject_id,
                            current_work_order_version=context.work.version,
                        ),
                        created=False,
                    )
                self._require_version(context.work, expected_work_order_version)
                source_draft_id = context.incident.source_draft_id
            return source_draft_id

        prepared = await asyncio.to_thread(prepare)
        if isinstance(prepared, FieldEvidenceUploadResult):
            return prepared
        source_draft_id = prepared

        media = await MediaService(
            self._database,
            self._authorizer,
            self._object_store,
        ).upload_authorized(
            identity,
            source_draft_id,
            declared_mime=normalized_mime,
            content=content,
            request_id=request_id,
        )

        def finalize() -> FieldEvidenceUploadResult:
            with work_order_read_transaction(
                self._database, identity, work_order_id, request_id=request_id
            ) as session:
                context = self._load_context(
                    session,
                    identity,
                    work_order_id,
                    request_id=request_id,
                    for_update=True,
                )
                existing = self._find_operation(
                    session,
                    identity,
                    work_order_id,
                    client_operation_id,
                )
                if existing is not None:
                    self._require_same_request(
                        existing,
                        context,
                        identity,
                        expected_work_order_version,
                        actual_hash,
                        normalized_mime,
                        len(content),
                    )
                    bound_media = self._load_bound_media(session, identity, existing)
                    return FieldEvidenceUploadResult(
                        upload=_view(
                            existing,
                            bound_media,
                            acting_subject_id=identity.subject_id,
                            current_work_order_version=context.work.version,
                        ),
                        created=False,
                    )
                self._require_version(context.work, expected_work_order_version)
                if media.draft_id != context.incident.source_draft_id:
                    raise WorkOrderConflict(
                        "field_evidence_business_binding_changed",
                        context.work.version,
                    )
                now = datetime.now(UTC)
                record = WorkOrderFieldEvidenceUploadRecord(
                    evidence_upload_id=f"field-evidence-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    work_order_id=work_order_id,
                    incident_id=context.incident.incident_id,
                    source_draft_id=context.incident.source_draft_id,
                    media_id=media.media_id,
                    uploaded_by_subject_id=identity.subject_id,
                    client_operation_id=client_operation_id,
                    content_hash=media.content_hash,
                    declared_mime=media.declared_mime,
                    size_bytes=media.size_bytes,
                    work_order_version=context.work.version,
                    occurred_at=media.created_at,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
                session.flush()
                persisted_media = self._load_bound_media(session, identity, record)
                return FieldEvidenceUploadResult(
                    upload=_view(
                        record,
                        persisted_media,
                        acting_subject_id=identity.subject_id,
                        current_work_order_version=context.work.version,
                    ),
                    created=True,
                )

        return await asyncio.to_thread(finalize)

    def list(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> list[FieldEvidenceUploadView]:
        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            context = self._load_context(
                session,
                identity,
                work_order_id,
                request_id=request_id,
            )
            latest_run_id = (
                select(RecognitionRunRecord.recognition_run_id)
                .where(
                    RecognitionRunRecord.tenant_id == identity.tenant_id,
                    RecognitionRunRecord.context_kind == "FIELD_EVIDENCE",
                    RecognitionRunRecord.work_order_id == work_order_id,
                    RecognitionRunRecord.field_evidence_upload_id
                    == WorkOrderFieldEvidenceUploadRecord.evidence_upload_id,
                )
                .order_by(
                    RecognitionRunRecord.created_at.desc(),
                    RecognitionRunRecord.recognition_run_id.desc(),
                )
                .limit(1)
                .correlate(WorkOrderFieldEvidenceUploadRecord)
                .scalar_subquery()
            )
            rows = session.execute(
                select(
                    WorkOrderFieldEvidenceUploadRecord,
                    MediaObjectRecord,
                    RecognitionRunRecord,
                    EvidenceBundleRecord,
                )
                .join(
                    MediaObjectRecord,
                    MediaObjectRecord.media_id == WorkOrderFieldEvidenceUploadRecord.media_id,
                )
                .outerjoin(
                    RecognitionRunRecord,
                    RecognitionRunRecord.recognition_run_id == latest_run_id,
                )
                .outerjoin(
                    EvidenceBundleRecord,
                    and_(
                        EvidenceBundleRecord.tenant_id == identity.tenant_id,
                        EvidenceBundleRecord.context_kind == "FIELD_EVIDENCE",
                        EvidenceBundleRecord.bundle_id == RecognitionRunRecord.evidence_bundle_id,
                    ),
                )
                .where(
                    WorkOrderFieldEvidenceUploadRecord.tenant_id == identity.tenant_id,
                    MediaObjectRecord.tenant_id == identity.tenant_id,
                    WorkOrderFieldEvidenceUploadRecord.work_order_id == work_order_id,
                )
                .order_by(
                    WorkOrderFieldEvidenceUploadRecord.occurred_at.asc(),
                    WorkOrderFieldEvidenceUploadRecord.evidence_upload_id.asc(),
                )
            ).tuples()
            return [
                _view(
                    binding,
                    media,
                    run,
                    bundle,
                    acting_subject_id=identity.subject_id,
                    current_work_order_version=context.work.version,
                )
                for binding, media, run, bundle in rows
            ]

    def _load_context(
        self,
        session: Session,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
        for_update: bool = False,
    ) -> _FieldEvidenceContext:
        statement = (
            select(WorkOrderRecord, IncidentRecord, AssetSiteLinkRecord.site_id)
            .join(IncidentRecord, IncidentRecord.incident_id == WorkOrderRecord.incident_id)
            .outerjoin(
                AssetSiteLinkRecord,
                AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
            )
            .where(
                WorkOrderRecord.tenant_id == identity.tenant_id,
                IncidentRecord.tenant_id == identity.tenant_id,
                WorkOrderRecord.work_order_id == work_order_id,
            )
        )
        if for_update:
            statement = statement.with_for_update(of=WorkOrderRecord)
        row = session.execute(statement).tuples().one_or_none()
        if row is None:
            raise WorkOrderNotVisible
        work, incident, site_id = row
        if incident.asset_id in identity.asset_ids:
            resource = ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id=work_order_id,
                asset_id=incident.asset_id,
            )
        elif site_id is not None and site_id in identity.site_ids:
            resource = ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id=work_order_id,
                site_id=site_id,
            )
        else:
            resource = ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id=work_order_id,
                asset_id=incident.asset_id,
            )
        self._authorizer.require(
            identity,
            Action.EXECUTE_WORK_ORDER,
            resource,
            request_id=request_id,
        )
        if work.assigned_subject_id != identity.subject_id:
            raise WorkOrderConflict("assignee_mismatch", work.version)
        if work.status not in FIELD_EVIDENCE_EXECUTABLE_STATUSES:
            raise WorkOrderConflict("field_evidence_state_invalid", work.version)
        return _FieldEvidenceContext(work=work, incident=incident, site_id=site_id)

    @staticmethod
    def _require_version(work: WorkOrderRecord, expected_version: int) -> None:
        if work.version != expected_version:
            raise WorkOrderConflict("version_conflict", work.version)

    @staticmethod
    def _find_operation(
        session: Session,
        identity: IdentityContext,
        work_order_id: str,
        client_operation_id: str,
    ) -> WorkOrderFieldEvidenceUploadRecord | None:
        return session.scalar(
            select(WorkOrderFieldEvidenceUploadRecord).where(
                WorkOrderFieldEvidenceUploadRecord.tenant_id == identity.tenant_id,
                WorkOrderFieldEvidenceUploadRecord.work_order_id == work_order_id,
                WorkOrderFieldEvidenceUploadRecord.uploaded_by_subject_id == identity.subject_id,
                WorkOrderFieldEvidenceUploadRecord.client_operation_id == client_operation_id,
            )
        )

    @staticmethod
    def _require_same_request(
        existing: WorkOrderFieldEvidenceUploadRecord,
        context: _FieldEvidenceContext,
        identity: IdentityContext,
        expected_version: int,
        content_hash: str,
        declared_mime: str,
        size_bytes: int,
    ) -> None:
        if (
            existing.incident_id != context.incident.incident_id
            or existing.source_draft_id != context.incident.source_draft_id
            or existing.uploaded_by_subject_id != identity.subject_id
            or existing.work_order_version != expected_version
            or existing.content_hash != content_hash
            or existing.declared_mime != declared_mime
            or existing.size_bytes != size_bytes
        ):
            raise WorkOrderConflict(
                "field_evidence_idempotency_mismatch",
                context.work.version,
            )

    @staticmethod
    def _load_bound_media(
        session: Session,
        identity: IdentityContext,
        binding: WorkOrderFieldEvidenceUploadRecord,
    ) -> MediaObjectRecord:
        media = session.scalar(
            select(MediaObjectRecord).where(
                MediaObjectRecord.tenant_id == identity.tenant_id,
                MediaObjectRecord.media_id == binding.media_id,
            )
        )
        if media is None:
            raise WorkOrderConflict("field_evidence_integrity_failed")
        return media


def _view(
    binding: WorkOrderFieldEvidenceUploadRecord,
    media: MediaObjectRecord,
    run: RecognitionRunRecord | None = None,
    bundle: EvidenceBundleRecord | None = None,
    *,
    acting_subject_id: str,
    current_work_order_version: int,
) -> FieldEvidenceUploadView:
    binding_matches_media = (
        media.draft_id == binding.source_draft_id
        and media.content_hash == binding.content_hash
        and media.declared_mime == binding.declared_mime
        and media.size_bytes == binding.size_bytes
    )
    ready = binding_matches_media and media.scan_state == "CLEAN"
    ready_to_attach = ready and binding.uploaded_by_subject_id == acting_subject_id
    has_current_run = (
        run is not None
        and run.requested_by_subject_id == acting_subject_id
        and run.work_order_version == current_work_order_version
    )
    legal_actions: list[str] = []
    if ready_to_attach:
        legal_actions.append("ATTACH")
    detected_mime = (media.detected_mime or media.declared_mime).casefold()
    if (
        ready
        and detected_mime in {"image/jpeg", "image/png", "video/mp4", "video/webm"}
        and not has_current_run
    ):
        legal_actions.append("RECOGNIZE")
    if has_current_run and bundle is not None and bundle.status == "READY_FOR_CONFIRMATION":
        legal_actions.append("CONFIRM_RECOGNITION")
    if has_current_run and bundle is not None and bundle.status == "CONFIRMED":
        legal_actions.append("CREATE_AI_OBSERVATION")
    return FieldEvidenceUploadView(
        evidence_upload_id=binding.evidence_upload_id,
        work_order_id=binding.work_order_id,
        media_id=binding.media_id,
        uploaded_by_subject_id=binding.uploaded_by_subject_id,
        client_operation_id=binding.client_operation_id,
        content_hash=f"sha256:{binding.content_hash}",
        declared_mime=binding.declared_mime,
        detected_mime=media.detected_mime,
        size_bytes=binding.size_bytes,
        work_order_version=binding.work_order_version,
        scan_state=media.scan_state,
        scan_version=media.scan_version,
        content_credential_status=media.content_credential_status,
        ready_to_attach=ready_to_attach,
        legal_actions=legal_actions,
        recognition_run_id=run.recognition_run_id if run is not None else None,
        recognition_status=run.status if run is not None else None,
        recognition_bundle_id=bundle.bundle_id if bundle is not None else None,
        recognition_bundle_status=bundle.status if bundle is not None else None,
        recognition_bundle_version=bundle.version if bundle is not None else None,
        recognition_failure_summary=run.failure_reason if run is not None else None,
        occurred_at=_utc(binding.occurred_at),
        updated_at=_utc(media.updated_at),
    )
