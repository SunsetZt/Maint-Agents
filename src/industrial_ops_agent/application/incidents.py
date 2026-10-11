"""Transactional draft, evidence-confirmation, and incident lifecycle services."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.domain.incidents import (
    DraftStateConflict,
    DraftTimelineEvent,
    DraftVersionConflict,
    Incident,
    IncidentDraft,
    IncidentStateConflict,
    IncidentSubmissionBlocked,
    IncidentVersionConflict,
    create_request_hash,
)
from industrial_ops_agent.maintenance_planning.review_isolation import (
    draft_read_transaction,
    incident_read_transaction,
    lock_subject,
    record_draft_disclosure,
    work_order_read_transaction,
)
from industrial_ops_agent.media.models import ScanState
from industrial_ops_agent.media.service import MediaService, TenantObjectStore
from industrial_ops_agent.model_gateway.service import ModelGatewayError
from industrial_ops_agent.multimodal.models import (
    AsrEntityCandidate,
    AsrSegment,
    BoundingBox,
    EntityValidationStatus,
    EvidenceBundle,
    EvidenceSchemaError,
    EvidenceSecurityFinding,
    EvidenceStatus,
    EvidenceVersionConflict,
    ExtractedEntity,
    FindingDisposition,
    FindingReview,
    HumanCorrection,
    OcrBlock,
    OcrDisposition,
    OcrReview,
    QrCodeCandidate,
    QrDisposition,
    QrPayloadKind,
    QrReview,
    TranscriptDisposition,
    TranscriptReview,
    VideoEvent,
    VideoEventReview,
    VideoKeyframe,
    VisualFinding,
)
from industrial_ops_agent.multimodal.processor import MultimodalProcessor, ProcessingRequest
from industrial_ops_agent.multimodal.providers import MultimodalProviderUnavailable
from industrial_ops_agent.object_store.keys import generated_object_key
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetSiteLinkRecord,
    DraftTimelineEventRecord,
    EvidenceBundleRecord,
    IdempotencyRecord,
    IncidentDraftRecord,
    IncidentRecord,
    MediaObjectRecord,
    RecognitionCorrectionRecord,
    RecognitionResultRecord,
    RecognitionRunRecord,
    WorkOrderFieldEvidenceUploadRecord,
    WorkOrderRecord,
)
from industrial_ops_agent.persistence.repositories import (
    AssetRepository,
    DiagnosisRunRepository,
    DraftTimelineRepository,
    EvidenceBundleRepository,
    IdempotencyRepository,
    IncidentDraftRepository,
    IncidentRepository,
    MediaObjectRepository,
    RecognitionRunRepository,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.workorders.service import WorkOrderConflict, WorkOrderNotVisible

RECOGNITION_CONTEXT_INCIDENT_DRAFT = "INCIDENT_DRAFT"
RECOGNITION_CONTEXT_FIELD_EVIDENCE = "FIELD_EVIDENCE"
FIELD_IMAGE_RECOGNITION_MIME_TYPES = frozenset({"image/jpeg", "image/png"})
FIELD_VIDEO_RECOGNITION_MIME_TYPES = frozenset({"video/mp4", "video/webm"})
FIELD_RECOGNITION_MIME_TYPES = (
    FIELD_IMAGE_RECOGNITION_MIME_TYPES | FIELD_VIDEO_RECOGNITION_MIME_TYPES
)
FIELD_RECOGNITION_WORK_ORDER_STATUSES = frozenset({"ACCEPTED", "IN_PROGRESS", "ON_HOLD"})


class IdempotencyConflict(Exception):
    """An idempotency key was reused for a non-equivalent request."""


@dataclass(frozen=True, slots=True)
class DraftCreationResult:
    draft: IncidentDraft
    created: bool


@dataclass(frozen=True, slots=True)
class RecognitionRun:
    recognition_run_id: str
    draft_id: str
    media_id: str
    context_kind: str
    work_order_id: str | None
    field_evidence_upload_id: str | None
    work_order_version: int | None
    requested_by_subject_id: str | None
    workflow_id: str
    status: str
    processor_profile: str
    evidence_bundle_id: str | None
    failure_reason: str | None
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RecognitionRunCreationResult:
    run: RecognitionRun
    created: bool


@dataclass(frozen=True, slots=True)
class _FieldRecognitionContext:
    work: WorkOrderRecord
    incident: IncidentRecord
    upload: WorkOrderFieldEvidenceUploadRecord
    media: MediaObjectRecord
    asset_model: str


@dataclass(frozen=True, slots=True)
class IncidentSubmissionResult:
    incident: Incident
    created: bool


class IncidentDraftService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def create(
        self,
        identity: IdentityContext,
        *,
        asset_id: str,
        description: str,
        idempotency_key: str,
        request_id: str,
    ) -> DraftCreationResult:
        _require(
            self._authorizer,
            identity,
            Action.CREATE_INCIDENT_DRAFT,
            asset_id=asset_id,
            request_id=request_id,
        )
        request_hash = create_request_hash(asset_id=asset_id, description=description)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            if AssetRepository(session, identity.tenant_context).get(asset_id) is None:
                raise ResourceNotVisible
            idempotency = IdempotencyRepository(session, identity.tenant_context)
            existing = idempotency.get(
                subject_id=identity.subject_id,
                key=idempotency_key,
            )
            drafts = IncidentDraftRepository(session, identity.tenant_context)
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise IdempotencyConflict
                draft_record = drafts.get(existing.result_ref)
                if draft_record is None:
                    raise RuntimeError("idempotency record references a missing draft")
                record_draft_disclosure(
                    session, identity, draft_record.draft_id, request_id=request_id
                )
                return DraftCreationResult(_to_draft(draft_record), created=False)

            draft, event = IncidentDraft.create(
                draft_id=f"draft-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                asset_id=asset_id,
                author_subject_id=identity.subject_id,
                description=description,
                occurred_at=now,
            )
            drafts.add(_draft_record(draft))
            # Persist the parent before adding records that reference it. These
            # models intentionally have no ORM relationships, so relying on the
            # unit-of-work sorter can emit the timeline row first on PostgreSQL.
            session.flush()
            DraftTimelineRepository(session, identity.tenant_context).add(
                _event_record(draft, event)
            )
            idempotency.add(
                IdempotencyRecord(
                    record_id=f"idem-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=idempotency_key,
                    request_hash=request_hash,
                    result_ref=draft.draft_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            record_draft_disclosure(session, identity, draft.draft_id, request_id=request_id)
            return DraftCreationResult(draft, created=True)

    def get(
        self,
        identity: IdentityContext,
        draft_id: str,
        *,
        request_id: str,
    ) -> IncidentDraft:
        with draft_read_transaction(
            self._database,
            identity,
            draft_id,
            request_id=request_id,
        ) as session:
            record = _visible_draft(
                session,
                self._authorizer,
                identity,
                draft_id,
                action=Action.READ_ASSET,
                request_id=request_id,
            )
            return _to_draft(record)

    def update_description(
        self,
        identity: IdentityContext,
        draft_id: str,
        *,
        description: str,
        expected_version: int,
        request_id: str,
    ) -> IncidentDraft:
        now = datetime.now(UTC)
        with draft_read_transaction(
            self._database,
            identity,
            draft_id,
            request_id=request_id,
        ) as session:
            record = _visible_draft(
                session,
                self._authorizer,
                identity,
                draft_id,
                action=Action.UPDATE_INCIDENT_DRAFT,
                request_id=request_id,
            )
            current = _to_draft(record)
            updated, event = current.update_description(
                description=description,
                expected_version=expected_version,
                occurred_at=now,
            )
            drafts = IncidentDraftRepository(session, identity.tenant_context)
            changed = drafts.update_description(
                draft_id=draft_id,
                expected_version=expected_version,
                description=description,
                updated_at=now,
            )
            if not changed:
                session.expire_all()
                winner = drafts.get(draft_id)
                raise DraftVersionConflict(
                    winner.version if winner is not None else current.version
                )
            timeline = DraftTimelineRepository(session, identity.tenant_context)
            timeline.add(
                _event_record(
                    updated,
                    event,
                    sequence=timeline.next_sequence(updated.draft_id),
                )
            )
            return updated

    def timeline(
        self,
        identity: IdentityContext,
        draft_id: str,
        *,
        request_id: str,
    ) -> list[DraftTimelineEvent]:
        with draft_read_transaction(
            self._database,
            identity,
            draft_id,
            request_id=request_id,
        ) as session:
            _visible_draft(
                session,
                self._authorizer,
                identity,
                draft_id,
                action=Action.READ_ASSET,
                request_id=request_id,
            )
            records = DraftTimelineRepository(session, identity.tenant_context).list(draft_id)
            return [
                DraftTimelineEvent(
                    sequence=record.sequence,
                    event_type=record.event_type,
                    occurred_at=_as_utc(record.occurred_at),
                    payload_metadata=record.payload_metadata,
                )
                for record in records
            ]


class RecognitionService:
    """Own the worker-safe OCR/VLM persistence and human confirmation use cases."""

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
    ) -> None:
        self._database = database
        self._authorizer = authorizer

    def enqueue(
        self,
        identity: IdentityContext,
        draft_id: str,
        *,
        media_id: str,
        processor_profile: str,
        idempotency_key: str,
        request_id: str,
    ) -> RecognitionRunCreationResult:
        now = datetime.now(UTC)
        request_hash = _canonical_hash(
            {
                "draft_id": draft_id,
                "media_id": media_id,
                "processor_profile": processor_profile,
            }
        )
        with draft_read_transaction(
            self._database,
            identity,
            draft_id,
            request_id=request_id,
        ) as session:
            draft = _visible_draft(
                session,
                self._authorizer,
                identity,
                draft_id,
                action=Action.RUN_RECOGNITION,
                request_id=request_id,
            )
            if draft.status != "DRAFT":
                raise DraftStateConflict("recognition is only allowed for a DRAFT")
            media = MediaObjectRepository(session, identity.tenant_context).get(media_id)
            if media is None or media.draft_id != draft_id:
                raise ResourceNotVisible
            if media.scan_state != ScanState.CLEAN.value:
                raise IncidentSubmissionBlocked("media_not_clean")

            idempotency = IdempotencyRepository(session, identity.tenant_context)
            existing = idempotency.get(subject_id=identity.subject_id, key=idempotency_key)
            runs = RecognitionRunRepository(session, identity.tenant_context)
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise IdempotencyConflict
                existing_run = runs.get(existing.result_ref)
                if existing_run is None:
                    raise RuntimeError("idempotency record references a missing recognition run")
                return RecognitionRunCreationResult(_to_recognition_run(existing_run), False)

            run_id = f"recognition-{uuid4().hex}"
            workflow_id = f"media-recognition-{identity.tenant_id}-{run_id}"
            record = RecognitionRunRecord(
                recognition_run_id=run_id,
                tenant_id=identity.tenant_id,
                draft_id=draft_id,
                media_id=media_id,
                context_kind=RECOGNITION_CONTEXT_INCIDENT_DRAFT,
                work_order_id=None,
                field_evidence_upload_id=None,
                work_order_version=None,
                requested_by_subject_id=identity.subject_id,
                workflow_id=workflow_id,
                status="QUEUED",
                processor_profile=processor_profile,
                evidence_bundle_id=None,
                failure_reason=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            runs.add(record)
            idempotency.add(
                IdempotencyRecord(
                    record_id=f"idem-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=idempotency_key,
                    request_hash=request_hash,
                    result_ref=run_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            _append_draft_event(
                session,
                identity.tenant_context,
                draft,
                event_type="recognition.queued",
                occurred_at=now,
                metadata={"recognition_run_id": run_id, "media_id": media_id},
            )
            return RecognitionRunCreationResult(_to_recognition_run(record), True)

    def get_run(
        self,
        identity: IdentityContext,
        draft_id: str,
        recognition_run_id: str,
        *,
        request_id: str,
    ) -> RecognitionRun:
        """Read the current status of one draft-bound recognition run."""
        with draft_read_transaction(
            self._database,
            identity,
            draft_id,
            request_id=request_id,
        ) as session:
            _visible_draft(
                session,
                self._authorizer,
                identity,
                draft_id,
                action=Action.RUN_RECOGNITION,
                request_id=request_id,
            )
            run = RecognitionRunRepository(session, identity.tenant_context).get(recognition_run_id)
            if (
                run is None
                or run.draft_id != draft_id
                or run.context_kind != RECOGNITION_CONTEXT_INCIDENT_DRAFT
            ):
                raise ResourceNotVisible
            return _to_recognition_run(run)

    def enqueue_field(
        self,
        identity: IdentityContext,
        work_order_id: str,
        field_evidence_upload_id: str,
        *,
        expected_work_order_version: int,
        idempotency_key: str,
        request_id: str,
        temporal_analysis: bool = False,
    ) -> RecognitionRunCreationResult:
        now = datetime.now(UTC)
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            context = _field_recognition_context(
                session,
                self._authorizer,
                identity,
                work_order_id,
                field_evidence_upload_id,
                request_id=request_id,
                expected_work_order_version=expected_work_order_version,
                for_update=True,
            )
            detected_mime = (context.media.detected_mime or context.media.declared_mime).casefold()
            processor_profile = _field_processor_profile(
                detected_mime,
                temporal_analysis=temporal_analysis,
                current_work_order_version=context.work.version,
            )
            request_hash = _canonical_hash(
                {
                    "context_kind": RECOGNITION_CONTEXT_FIELD_EVIDENCE,
                    "work_order_id": work_order_id,
                    "field_evidence_upload_id": field_evidence_upload_id,
                    "work_order_version": expected_work_order_version,
                    "processor_profile": processor_profile,
                }
            )
            idempotency = IdempotencyRepository(session, identity.tenant_context)
            existing_idempotency = idempotency.get(
                subject_id=identity.subject_id,
                key=idempotency_key,
            )
            runs = RecognitionRunRepository(session, identity.tenant_context)
            existing_run = runs.for_field_upload(
                work_order_id=work_order_id,
                field_evidence_upload_id=field_evidence_upload_id,
            )
            if existing_idempotency is not None:
                if existing_idempotency.request_hash != request_hash:
                    raise IdempotencyConflict
                replay = runs.get(existing_idempotency.result_ref)
                if replay is None or replay.recognition_run_id != getattr(
                    existing_run, "recognition_run_id", None
                ):
                    raise RuntimeError("idempotency record references a missing recognition run")
                return RecognitionRunCreationResult(_to_recognition_run(replay), False)
            if (
                existing_run is not None
                and existing_run.requested_by_subject_id == identity.subject_id
                and existing_run.work_order_version == expected_work_order_version
            ):
                raise WorkOrderConflict(
                    "field_recognition_already_started",
                    context.work.version,
                )

            run_id = f"recognition-{uuid4().hex}"
            workflow_id = f"field-recognition-{identity.tenant_id}-{run_id}"
            record = RecognitionRunRecord(
                recognition_run_id=run_id,
                tenant_id=identity.tenant_id,
                draft_id=context.incident.source_draft_id,
                media_id=context.media.media_id,
                context_kind=RECOGNITION_CONTEXT_FIELD_EVIDENCE,
                work_order_id=work_order_id,
                field_evidence_upload_id=field_evidence_upload_id,
                work_order_version=expected_work_order_version,
                requested_by_subject_id=identity.subject_id,
                workflow_id=workflow_id,
                status="QUEUED",
                processor_profile=processor_profile,
                evidence_bundle_id=None,
                failure_reason=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            runs.add(record)
            idempotency.add(
                IdempotencyRecord(
                    record_id=f"idem-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=idempotency_key,
                    request_hash=request_hash,
                    result_ref=run_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            return RecognitionRunCreationResult(_to_recognition_run(record), True)

    async def process(
        self,
        identity: IdentityContext,
        recognition_run_id: str,
        *,
        object_store: TenantObjectStore,
        processor: MultimodalProcessor,
        request_id: str,
    ) -> EvidenceBundle:
        started_at = datetime.now(UTC)
        try:
            with self._database.transaction(identity.tenant_context) as session:
                runs = RecognitionRunRepository(session, identity.tenant_context)
                run = runs.get(recognition_run_id)
                if run is None:
                    raise ResourceNotVisible
                if run.context_kind == RECOGNITION_CONTEXT_FIELD_EVIDENCE:
                    if (
                        run.work_order_id is None
                        or run.field_evidence_upload_id is None
                        or run.work_order_version is None
                        or run.requested_by_subject_id != identity.subject_id
                    ):
                        raise DraftStateConflict("field recognition binding is incomplete")
                    field_context = _field_recognition_context(
                        session,
                        self._authorizer,
                        identity,
                        run.work_order_id,
                        run.field_evidence_upload_id,
                        request_id=request_id,
                        expected_work_order_version=run.work_order_version,
                        for_update=True,
                    )
                    if (
                        field_context.incident.source_draft_id != run.draft_id
                        or field_context.media.media_id != run.media_id
                    ):
                        raise DraftStateConflict("field recognition binding changed")
                    draft_id = field_context.incident.source_draft_id
                    asset_id = field_context.incident.asset_id
                    asset_model = field_context.asset_model
                    media = field_context.media
                    _validate_field_processor_profile(
                        (media.detected_mime or media.declared_mime).casefold(),
                        run.processor_profile,
                    )
                else:
                    draft = _visible_draft(
                        session,
                        self._authorizer,
                        identity,
                        run.draft_id,
                        action=Action.RUN_RECOGNITION,
                        request_id=request_id,
                    )
                    media = MediaObjectRepository(session, identity.tenant_context).get(
                        run.media_id
                    )
                    asset = AssetRepository(session, identity.tenant_context).get(draft.asset_id)
                    if media is None or media.draft_id != draft.draft_id or asset is None:
                        raise ResourceNotVisible
                    if draft.status != "DRAFT":
                        raise DraftStateConflict("recognition is only allowed for a DRAFT")
                    if media.scan_state != ScanState.CLEAN.value:
                        raise IncidentSubmissionBlocked("media_not_clean")
                    draft_id = draft.draft_id
                    asset_id = draft.asset_id
                    asset_model = asset.model_code or "unknown-model"
                changed = runs.transition(
                    recognition_run_id=recognition_run_id,
                    expected_version=run.version,
                    expected_status="QUEUED",
                    status="RUNNING",
                    updated_at=started_at,
                )
                if not changed:
                    raise DraftStateConflict("recognition run is no longer queued")
                media_id = media.media_id
                content_hash = media.content_hash
                mime_type = media.detected_mime or media.declared_mime
                processor_profile = run.processor_profile

            content = await MediaService(
                self._database,
                self._authorizer,
                object_store,
            ).read_clean(identity, media_id, request_id=request_id)
            bundle = await processor.process(
                ProcessingRequest(
                    bundle_id=f"evidence-{uuid4().hex}",
                    recognition_run_id=recognition_run_id,
                    tenant_id=identity.tenant_id,
                    draft_id=draft_id,
                    asset_id=asset_id,
                    asset_model=asset_model,
                    media_id=media_id,
                    content_hash=content_hash,
                    mime_type=mime_type,
                    content=content,
                    occurred_at=datetime.now(UTC),
                    subject_id=identity.subject_id,
                    trace_id=request_id,
                    processor_profile=processor_profile,
                )
            )
            await self._store_video_keyframes(
                identity,
                bundle,
                object_store=object_store,
                request_id=request_id,
            )
            self._complete_run(
                identity,
                recognition_run_id,
                bundle,
                request_id=request_id,
            )
            return bundle
        except Exception as exc:
            self._fail_run(identity, recognition_run_id, _recognition_failure_code(exc))
            raise

    def get_evidence(
        self,
        identity: IdentityContext,
        draft_id: str,
        *,
        request_id: str,
    ) -> EvidenceBundle:
        with draft_read_transaction(
            self._database,
            identity,
            draft_id,
            request_id=request_id,
        ) as session:
            _visible_draft(
                session,
                self._authorizer,
                identity,
                draft_id,
                action=Action.READ_ASSET,
                request_id=request_id,
            )
            evidence = EvidenceBundleRepository(session, identity.tenant_context)
            record = evidence.latest_for_draft(draft_id)
            if record is None:
                raise ResourceNotVisible
            return _to_evidence_bundle(
                record,
                evidence.list_results(record.bundle_id),
                evidence.list_corrections(record.bundle_id),
            )

    def get_field_evidence(
        self,
        identity: IdentityContext,
        work_order_id: str,
        field_evidence_upload_id: str,
        *,
        request_id: str,
    ) -> EvidenceBundle:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            _field_recognition_context(
                session,
                self._authorizer,
                identity,
                work_order_id,
                field_evidence_upload_id,
                request_id=request_id,
            )
            runs = RecognitionRunRepository(session, identity.tenant_context)
            run = runs.for_field_upload(
                work_order_id=work_order_id,
                field_evidence_upload_id=field_evidence_upload_id,
            )
            if run is None or run.evidence_bundle_id is None:
                raise ResourceNotVisible
            evidence = EvidenceBundleRepository(session, identity.tenant_context)
            record = evidence.get(run.evidence_bundle_id)
            if (
                record is None
                or record.context_kind != RECOGNITION_CONTEXT_FIELD_EVIDENCE
                or record.work_order_id != work_order_id
                or record.field_evidence_upload_id != field_evidence_upload_id
            ):
                raise ResourceNotVisible
            return _to_evidence_bundle(
                record,
                evidence.list_results(record.bundle_id),
                evidence.list_corrections(record.bundle_id),
            )

    async def read_video_keyframe(
        self,
        identity: IdentityContext,
        draft_id: str,
        bundle_id: str,
        frame_id: str,
        *,
        object_store: TenantObjectStore,
        request_id: str,
    ) -> bytes:
        def load_binding() -> str:
            with draft_read_transaction(
                self._database,
                identity,
                draft_id,
                request_id=request_id,
            ) as session:
                _visible_draft(
                    session,
                    self._authorizer,
                    identity,
                    draft_id,
                    action=Action.READ_ASSET,
                    request_id=request_id,
                )
                repository = EvidenceBundleRepository(session, identity.tenant_context)
                bundle = repository.get(bundle_id)
                if (
                    bundle is None
                    or bundle.context_kind != RECOGNITION_CONTEXT_INCIDENT_DRAFT
                    or bundle.draft_id != draft_id
                    or bundle.source_type != "video"
                ):
                    raise ResourceNotVisible
                expected_hash: str | None = None
                for result in repository.list_results(bundle_id):
                    if (
                        result.result_type == "VIDEO_KEYFRAME"
                        and result.payload.get("frame_id") == frame_id
                    ):
                        value = result.payload.get("image_sha256")
                        expected_hash = str(value) if value is not None else None
                        break
                if expected_hash is None or len(expected_hash) != 64:
                    raise ResourceNotVisible
            return expected_hash

        expected_hash = await asyncio.to_thread(load_binding)
        return await self._read_verified_video_keyframe_artifact(
            identity,
            bundle_id,
            frame_id,
            expected_hash,
            object_store=object_store,
            request_id=request_id,
        )

    async def read_field_video_keyframe(
        self,
        identity: IdentityContext,
        work_order_id: str,
        field_evidence_upload_id: str,
        frame_id: str,
        *,
        object_store: TenantObjectStore,
        request_id: str,
    ) -> bytes:
        def load_binding() -> tuple[str, str]:
            with work_order_read_transaction(
                self._database,
                identity,
                work_order_id,
                request_id=request_id,
            ) as session:
                _field_recognition_context(
                    session,
                    self._authorizer,
                    identity,
                    work_order_id,
                    field_evidence_upload_id,
                    request_id=request_id,
                )
                runs = RecognitionRunRepository(session, identity.tenant_context)
                run = runs.for_field_upload(
                    work_order_id=work_order_id,
                    field_evidence_upload_id=field_evidence_upload_id,
                )
                if run is None or run.evidence_bundle_id is None:
                    raise ResourceNotVisible
                bundle_id = run.evidence_bundle_id
                repository = EvidenceBundleRepository(session, identity.tenant_context)
                bundle = repository.get(bundle_id)
                if (
                    bundle is None
                    or bundle.context_kind != RECOGNITION_CONTEXT_FIELD_EVIDENCE
                    or bundle.work_order_id != work_order_id
                    or bundle.field_evidence_upload_id != field_evidence_upload_id
                    or bundle.source_type != "video"
                ):
                    raise ResourceNotVisible
                expected_hash: str | None = None
                for result in repository.list_results(bundle_id):
                    if (
                        result.result_type == "VIDEO_KEYFRAME"
                        and result.payload.get("frame_id") == frame_id
                    ):
                        value = result.payload.get("image_sha256")
                        expected_hash = str(value) if value is not None else None
                        break
                if expected_hash is None or len(expected_hash) != 64:
                    raise ResourceNotVisible
            return bundle_id, expected_hash

        bundle_id, expected_hash = await asyncio.to_thread(load_binding)
        return await self._read_verified_video_keyframe_artifact(
            identity,
            bundle_id,
            frame_id,
            expected_hash,
            object_store=object_store,
            request_id=request_id,
        )

    async def _read_verified_video_keyframe_artifact(
        self,
        identity: IdentityContext,
        bundle_id: str,
        frame_id: str,
        expected_hash: str,
        *,
        object_store: TenantObjectStore,
        request_id: str,
    ) -> bytes:
        try:
            key = generated_object_key(
                identity.tenant_context,
                bundle_id,
                f"{frame_id}.jpg",
            )
        except ValueError as exc:
            raise ResourceNotVisible from exc
        content = await object_store.get(identity.tenant_context, key)
        if (
            content is None
            or len(content) > 2 * 1024 * 1024
            or not content.startswith(b"\xff\xd8\xff")
            or sha256(content).hexdigest() != expected_hash
        ):
            self._authorizer.record_guard_decision(
                identity,
                action="video_keyframe.read",
                decision="deny",
                reason_code="derived_artifact_integrity_failed",
                request_id=request_id,
                resource_id=frame_id,
            )
            raise ResourceNotVisible
        self._authorizer.record_guard_decision(
            identity,
            action="video_keyframe.read",
            decision="allow",
            reason_code="derived_artifact_verified",
            request_id=request_id,
            resource_id=frame_id,
        )
        return content

    def confirm(
        self,
        identity: IdentityContext,
        draft_id: str,
        *,
        bundle_id: str,
        corrections: dict[str, str],
        finding_dispositions: dict[str, FindingDisposition],
        expected_version: int,
        request_id: str,
        transcript_decisions: dict[str, tuple[TranscriptDisposition, str | None]] | None = None,
        ocr_block_decisions: dict[str, tuple[OcrDisposition, str | None]] | None = None,
        video_event_dispositions: dict[str, FindingDisposition] | None = None,
        qr_code_decisions: dict[str, tuple[QrDisposition, str | None]] | None = None,
    ) -> EvidenceBundle:
        now = datetime.now(UTC)
        with draft_read_transaction(
            self._database,
            identity,
            draft_id,
            request_id=request_id,
        ) as session:
            draft = _visible_draft(
                session,
                self._authorizer,
                identity,
                draft_id,
                action=Action.CONFIRM_RECOGNITION,
                request_id=request_id,
            )
            if draft.status != "DRAFT":
                raise DraftStateConflict("evidence can only be confirmed for a DRAFT")
            repository = EvidenceBundleRepository(session, identity.tenant_context)
            record = repository.get(bundle_id)
            if (
                record is None
                or record.context_kind != RECOGNITION_CONTEXT_INCIDENT_DRAFT
                or record.draft_id != draft_id
            ):
                raise ResourceNotVisible
            current = _to_evidence_bundle(
                record,
                repository.list_results(bundle_id),
                repository.list_corrections(bundle_id),
            )
            confirmed = current.confirm(
                reviewer_subject_id=identity.subject_id,
                corrections=corrections,
                finding_dispositions=finding_dispositions,
                transcript_decisions=transcript_decisions,
                ocr_block_decisions=ocr_block_decisions,
                video_event_dispositions=video_event_dispositions,
                qr_code_decisions=qr_code_decisions,
                expected_version=expected_version,
                occurred_at=now,
            )
            changed = repository.confirm(
                bundle_id=bundle_id,
                expected_version=expected_version,
                updated_at=now,
                automation_eligible=confirmed.automation_eligible,
                automation_blockers=confirmed.automation_blockers,
            )
            if not changed:
                session.expire_all()
                winner = repository.get(bundle_id)
                raise EvidenceVersionConflict(
                    winner.version if winner is not None else current.version
                )
            for correction in confirmed.human_corrections:
                repository.add_correction(
                    _entity_correction_record(identity.tenant_id, bundle_id, confirmed, correction)
                )
            for review in confirmed.finding_reviews:
                repository.add_correction(
                    _finding_review_record(identity.tenant_id, bundle_id, confirmed, review)
                )
            for ocr_review in confirmed.ocr_reviews:
                repository.add_correction(
                    _ocr_review_record(identity.tenant_id, bundle_id, confirmed, ocr_review)
                )
            for transcript_review in confirmed.transcript_reviews:
                repository.add_correction(
                    _transcript_review_record(
                        identity.tenant_id, bundle_id, confirmed, transcript_review
                    )
                )
            for video_event_review in confirmed.video_event_reviews:
                repository.add_correction(
                    _video_event_review_record(
                        identity.tenant_id,
                        bundle_id,
                        confirmed,
                        video_event_review,
                    )
                )
            for qr_review in confirmed.qr_reviews:
                repository.add_correction(
                    _qr_review_record(identity.tenant_id, bundle_id, confirmed, qr_review)
                )
            _append_draft_event(
                session,
                identity.tenant_context,
                draft,
                event_type="recognition.confirmed",
                occurred_at=now,
                metadata={
                    "bundle_id": bundle_id,
                    "correction_count": len(confirmed.human_corrections),
                    "finding_review_count": len(confirmed.finding_reviews),
                    "transcript_review_count": len(confirmed.transcript_reviews),
                    "qr_review_count": len(confirmed.qr_reviews),
                },
            )
            return confirmed

    def confirm_field(
        self,
        identity: IdentityContext,
        work_order_id: str,
        field_evidence_upload_id: str,
        *,
        corrections: dict[str, str],
        finding_dispositions: dict[str, FindingDisposition],
        ocr_block_decisions: dict[str, tuple[OcrDisposition, str | None]],
        transcript_decisions: dict[str, tuple[TranscriptDisposition, str | None]],
        video_event_dispositions: dict[str, FindingDisposition],
        expected_version: int,
        request_id: str,
        qr_code_decisions: dict[str, tuple[QrDisposition, str | None]] | None = None,
    ) -> EvidenceBundle:
        now = datetime.now(UTC)
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            field_context = _field_recognition_context(
                session,
                self._authorizer,
                identity,
                work_order_id,
                field_evidence_upload_id,
                request_id=request_id,
                for_update=True,
            )
            runs = RecognitionRunRepository(session, identity.tenant_context)
            run = runs.for_field_upload(
                work_order_id=work_order_id,
                field_evidence_upload_id=field_evidence_upload_id,
            )
            if run is None or run.evidence_bundle_id is None:
                raise ResourceNotVisible
            if run.requested_by_subject_id != identity.subject_id:
                raise WorkOrderConflict(
                    "recognition_state_changed",
                    field_context.work.version,
                )
            repository = EvidenceBundleRepository(session, identity.tenant_context)
            record = repository.get(run.evidence_bundle_id)
            if (
                record is None
                or record.context_kind != RECOGNITION_CONTEXT_FIELD_EVIDENCE
                or record.work_order_id != work_order_id
                or record.field_evidence_upload_id != field_evidence_upload_id
            ):
                raise ResourceNotVisible
            current = _to_evidence_bundle(
                record,
                repository.list_results(record.bundle_id),
                repository.list_corrections(record.bundle_id),
            )
            confirmed = current.confirm(
                reviewer_subject_id=identity.subject_id,
                corrections=corrections,
                finding_dispositions=finding_dispositions,
                transcript_decisions=transcript_decisions,
                ocr_block_decisions=ocr_block_decisions,
                video_event_dispositions=video_event_dispositions,
                qr_code_decisions=qr_code_decisions,
                expected_version=expected_version,
                occurred_at=now,
            )
            changed = repository.confirm(
                bundle_id=record.bundle_id,
                expected_version=expected_version,
                updated_at=now,
                automation_eligible=confirmed.automation_eligible,
                automation_blockers=confirmed.automation_blockers,
            )
            if not changed:
                session.expire_all()
                winner = repository.get(record.bundle_id)
                raise EvidenceVersionConflict(
                    winner.version if winner is not None else current.version
                )
            for correction in confirmed.human_corrections:
                repository.add_correction(
                    _entity_correction_record(
                        identity.tenant_id,
                        record.bundle_id,
                        confirmed,
                        correction,
                    )
                )
            for ocr_review in confirmed.ocr_reviews:
                repository.add_correction(
                    _ocr_review_record(
                        identity.tenant_id,
                        record.bundle_id,
                        confirmed,
                        ocr_review,
                    )
                )
            for review in confirmed.finding_reviews:
                repository.add_correction(
                    _finding_review_record(
                        identity.tenant_id,
                        record.bundle_id,
                        confirmed,
                        review,
                    )
                )
            for transcript_review in confirmed.transcript_reviews:
                repository.add_correction(
                    _transcript_review_record(
                        identity.tenant_id,
                        record.bundle_id,
                        confirmed,
                        transcript_review,
                    )
                )
            for video_event_review in confirmed.video_event_reviews:
                repository.add_correction(
                    _video_event_review_record(
                        identity.tenant_id,
                        record.bundle_id,
                        confirmed,
                        video_event_review,
                    )
                )
            for qr_review in confirmed.qr_reviews:
                repository.add_correction(
                    _qr_review_record(
                        identity.tenant_id,
                        record.bundle_id,
                        confirmed,
                        qr_review,
                    )
                )
            return confirmed

    def _complete_run(
        self,
        identity: IdentityContext,
        recognition_run_id: str,
        bundle: EvidenceBundle,
        *,
        request_id: str,
    ) -> None:
        with self._database.transaction(identity.tenant_context) as session:
            runs = RecognitionRunRepository(session, identity.tenant_context)
            run = runs.get(recognition_run_id)
            media = MediaObjectRepository(session, identity.tenant_context).get(bundle.media_id)
            if (
                run is None
                or run.status != "RUNNING"
                or media is None
                or media.scan_state != ScanState.CLEAN.value
                or media.content_hash != bundle.source_sha256
            ):
                raise DraftStateConflict("recognition preconditions changed before persistence")
            draft: IncidentDraftRecord | None = None
            if run.context_kind == RECOGNITION_CONTEXT_FIELD_EVIDENCE:
                if (
                    run.work_order_id is None
                    or run.field_evidence_upload_id is None
                    or run.work_order_version is None
                ):
                    raise DraftStateConflict("field recognition binding is incomplete")
                field_context = _field_recognition_context(
                    session,
                    self._authorizer,
                    identity,
                    run.work_order_id,
                    run.field_evidence_upload_id,
                    request_id=request_id,
                    expected_work_order_version=run.work_order_version,
                    for_update=True,
                )
                if (
                    field_context.incident.source_draft_id != bundle.draft_id
                    or field_context.media.media_id != bundle.media_id
                    or run.draft_id != bundle.draft_id
                    or run.media_id != bundle.media_id
                ):
                    raise DraftStateConflict(
                        "field recognition preconditions changed before persistence"
                    )
            else:
                draft = IncidentDraftRepository(session, identity.tenant_context).get(
                    bundle.draft_id
                )
                if draft is None or draft.status != "DRAFT":
                    raise DraftStateConflict("recognition preconditions changed before persistence")
            evidence = EvidenceBundleRepository(session, identity.tenant_context)
            evidence.add(_evidence_record(bundle, run))
            for result in _evidence_result_records(bundle):
                evidence.add_result(result)
            changed = runs.transition(
                recognition_run_id=recognition_run_id,
                expected_version=run.version,
                expected_status="RUNNING",
                status="SUCCEEDED",
                evidence_bundle_id=bundle.bundle_id,
                updated_at=bundle.updated_at,
            )
            if not changed:
                raise DraftStateConflict("recognition completion lost a concurrency race")
            if draft is not None:
                _append_draft_event(
                    session,
                    identity.tenant_context,
                    draft,
                    event_type="recognition.completed",
                    occurred_at=bundle.updated_at,
                    metadata={
                        "recognition_run_id": recognition_run_id,
                        "bundle_id": bundle.bundle_id,
                    },
                )

    async def _store_video_keyframes(
        self,
        identity: IdentityContext,
        bundle: EvidenceBundle,
        *,
        object_store: TenantObjectStore,
        request_id: str,
    ) -> None:
        if bundle.source_type != "video":
            return
        for frame in bundle.video_keyframes:
            if frame.image_jpeg is None:
                raise RuntimeError("new video evidence is missing a keyframe artifact")
            key = generated_object_key(
                identity.tenant_context,
                bundle.bundle_id,
                f"{frame.frame_id}.jpg",
            )
            await object_store.put(identity.tenant_context, key, frame.image_jpeg)
            stored = await object_store.get(identity.tenant_context, key)
            if stored is None or sha256(stored).hexdigest() != frame.image_sha256:
                self._authorizer.record_guard_decision(
                    identity,
                    action="video_keyframe.persist",
                    decision="deny",
                    reason_code="derived_artifact_write_verification_failed",
                    request_id=request_id,
                    resource_id=frame.frame_id,
                )
                raise RuntimeError("video keyframe artifact failed write verification")
            self._authorizer.record_guard_decision(
                identity,
                action="video_keyframe.persist",
                decision="allow",
                reason_code="derived_artifact_write_verified",
                request_id=request_id,
                resource_id=frame.frame_id,
            )

    def _fail_run(
        self,
        identity: IdentityContext,
        recognition_run_id: str,
        reason_code: str,
    ) -> None:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            repository = RecognitionRunRepository(session, identity.tenant_context)
            run = repository.get(recognition_run_id)
            if run is None or run.status not in {"QUEUED", "RUNNING"}:
                return
            repository.transition(
                recognition_run_id=recognition_run_id,
                expected_version=run.version,
                expected_status=run.status,
                status="FAILED",
                failure_reason=reason_code,
                updated_at=now,
            )
            draft = IncidentDraftRepository(session, identity.tenant_context).get(run.draft_id)
            if draft is not None and run.context_kind == RECOGNITION_CONTEXT_INCIDENT_DRAFT:
                _append_draft_event(
                    session,
                    identity.tenant_context,
                    draft,
                    event_type="recognition.failed",
                    occurred_at=now,
                    metadata={
                        "recognition_run_id": recognition_run_id,
                        "reason_code": reason_code,
                    },
                )


class IncidentLifecycleService:
    """Submit a confirmed draft into exactly one formal Incident."""

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def submit(
        self,
        identity: IdentityContext,
        draft_id: str,
        *,
        evidence_bundle_id: str,
        expected_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> IncidentSubmissionResult:
        now = datetime.now(UTC)
        request_hash = _canonical_hash(
            {
                "draft_id": draft_id,
                "evidence_bundle_id": evidence_bundle_id,
                "expected_version": expected_version,
            }
        )
        with draft_read_transaction(
            self._database,
            identity,
            draft_id,
            request_id=request_id,
        ) as session:
            draft_record = _visible_draft(
                session,
                self._authorizer,
                identity,
                draft_id,
                action=Action.SUBMIT_INCIDENT,
                request_id=request_id,
            )
            incidents = IncidentRepository(session, identity.tenant_context)
            idempotency = IdempotencyRepository(session, identity.tenant_context)
            existing_key = idempotency.get(
                subject_id=identity.subject_id,
                key=idempotency_key,
            )
            if existing_key is not None:
                if existing_key.request_hash != request_hash:
                    raise IdempotencyConflict
                existing = incidents.get(existing_key.result_ref)
                if existing is None:
                    raise RuntimeError("idempotency record references a missing incident")
                return IncidentSubmissionResult(_to_incident(existing), False)
            existing = incidents.get_by_draft(draft_id)
            if existing is not None:
                if existing.evidence_bundle_id != evidence_bundle_id:
                    raise IdempotencyConflict
                return IncidentSubmissionResult(_to_incident(existing), False)

            evidence = EvidenceBundleRepository(session, identity.tenant_context).get(
                evidence_bundle_id
            )
            if evidence is None or evidence.draft_id != draft_id:
                raise ResourceNotVisible
            media = MediaObjectRepository(session, identity.tenant_context).list_for_draft(draft_id)
            current = _to_draft(draft_record)
            submitted, incident, event = current.submit(
                incident_id=f"incident-{uuid4().hex}",
                evidence_bundle_id=evidence_bundle_id,
                expected_version=expected_version,
                evidence_confirmed=evidence.status == EvidenceStatus.CONFIRMED.value,
                all_media_clean=bool(media)
                and all(item.scan_state == ScanState.CLEAN.value for item in media),
                device_authorized=True,
                occurred_at=now,
            )
            changed = IncidentDraftRepository(session, identity.tenant_context).submit(
                draft_id=draft_id,
                expected_version=expected_version,
                updated_at=now,
            )
            if not changed:
                session.expire_all()
                winner = IncidentDraftRepository(session, identity.tenant_context).get(draft_id)
                raise DraftVersionConflict(
                    winner.version if winner is not None else current.version
                )
            incidents.add(_incident_record(incident))
            timeline = DraftTimelineRepository(session, identity.tenant_context)
            timeline.add(
                _event_record(
                    submitted,
                    event,
                    sequence=timeline.next_sequence(draft_id),
                )
            )
            idempotency.add(
                IdempotencyRecord(
                    record_id=f"idem-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=idempotency_key,
                    request_hash=request_hash,
                    result_ref=incident.incident_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            return IncidentSubmissionResult(incident, True)

    def get(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        request_id: str,
    ) -> Incident:
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            record = IncidentRepository(session, identity.tenant_context).get(incident_id)
            if record is None:
                raise ResourceNotVisible
            _require(
                self._authorizer,
                identity,
                Action.READ_ASSET,
                asset_id=record.asset_id,
                resource_id=incident_id,
                request_id=request_id,
            )
            diagnosis_run_ids = tuple(
                run.diagnosis_run_id
                for run in DiagnosisRunRepository(
                    session, identity.tenant_context
                ).list_for_incident(incident_id)
            )
            return _to_incident(record, diagnosis_run_ids=diagnosis_run_ids)

    def triage(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> Incident:
        now = datetime.now(UTC)
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            repository = IncidentRepository(session, identity.tenant_context)
            record = repository.get(incident_id)
            if record is None:
                raise ResourceNotVisible
            _require(
                self._authorizer,
                identity,
                Action.TRIAGE_INCIDENT,
                asset_id=record.asset_id,
                resource_id=incident_id,
                request_id=request_id,
            )
            current = _to_incident(record)
            triaged = current.triage(expected_version=expected_version, occurred_at=now)
            changed = repository.update_status(
                incident_id=incident_id,
                expected_version=expected_version,
                current_status="SUBMITTED",
                next_status="TRIAGED",
                updated_at=now,
            )
            if not changed:
                session.expire_all()
                winner = repository.get(incident_id)
                if winner is not None and winner.version != expected_version:
                    raise IncidentVersionConflict(winner.version)
                raise IncidentStateConflict("Incident can no longer be triaged")
            return triaged


def _to_draft(record: IncidentDraftRecord) -> IncidentDraft:
    return IncidentDraft(
        draft_id=record.draft_id,
        tenant_id=record.tenant_id,
        asset_id=record.asset_id,
        author_subject_id=record.author_subject_id,
        description=record.description,
        version=record.version,
        status=record.status,
        created_at=_as_utc(record.created_at),
        updated_at=_as_utc(record.updated_at),
    )


def _draft_record(draft: IncidentDraft) -> IncidentDraftRecord:
    return IncidentDraftRecord(
        draft_id=draft.draft_id,
        tenant_id=draft.tenant_id,
        asset_id=draft.asset_id,
        author_subject_id=draft.author_subject_id,
        description=draft.description,
        version=draft.version,
        status=draft.status,
        created_at=draft.created_at,
        updated_at=draft.updated_at,
    )


def _event_record(
    draft: IncidentDraft,
    event: DraftTimelineEvent,
    *,
    sequence: int | None = None,
) -> DraftTimelineEventRecord:
    return DraftTimelineEventRecord(
        event_id=f"event-{uuid4().hex}",
        tenant_id=draft.tenant_id,
        draft_id=draft.draft_id,
        sequence=sequence if sequence is not None else event.sequence,
        event_type=event.event_type,
        occurred_at=event.occurred_at,
        payload_metadata=event.payload_metadata,
        created_at=event.occurred_at,
        updated_at=event.occurred_at,
    )


def _visible_draft(
    session: Session,
    authorizer: Authorizer,
    identity: IdentityContext,
    draft_id: str,
    *,
    action: Action,
    request_id: str,
) -> IncidentDraftRecord:
    record = IncidentDraftRepository(session, identity.tenant_context).get(draft_id)
    if record is None:
        raise ResourceNotVisible
    _require(
        authorizer,
        identity,
        action,
        asset_id=record.asset_id,
        resource_id=draft_id,
        request_id=request_id,
    )
    if AssetRepository(session, identity.tenant_context).get(record.asset_id) is None:
        raise ResourceNotVisible
    return record


def _require(
    authorizer: Authorizer,
    identity: IdentityContext,
    action: Action,
    *,
    asset_id: str,
    request_id: str,
    resource_id: str | None = None,
) -> None:
    try:
        authorizer.require(
            identity,
            action,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id=resource_id,
                asset_id=asset_id,
            ),
            request_id=request_id,
        )
    except AuthorizationDenied as exc:
        raise ResourceNotVisible from exc


def _append_draft_event(
    session: Session,
    context: TenantContext,
    draft: IncidentDraftRecord,
    *,
    event_type: str,
    occurred_at: datetime,
    metadata: dict[str, object],
) -> None:
    timeline = DraftTimelineRepository(session, context)
    timeline.add(
        DraftTimelineEventRecord(
            event_id=f"event-{uuid4().hex}",
            tenant_id=draft.tenant_id,
            draft_id=draft.draft_id,
            sequence=timeline.next_sequence(draft.draft_id),
            event_type=event_type,
            occurred_at=occurred_at,
            payload_metadata=metadata,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
    )


def _field_recognition_context(
    session: Session,
    authorizer: Authorizer,
    identity: IdentityContext,
    work_order_id: str,
    field_evidence_upload_id: str,
    *,
    request_id: str,
    expected_work_order_version: int | None = None,
    for_update: bool = False,
) -> _FieldRecognitionContext:
    work_statement = select(WorkOrderRecord).where(
        WorkOrderRecord.tenant_id == identity.tenant_id,
        WorkOrderRecord.work_order_id == work_order_id,
    )
    if for_update:
        work_statement = work_statement.with_for_update(of=WorkOrderRecord)
    work = session.scalar(work_statement)
    if work is None:
        raise WorkOrderNotVisible
    incident = session.scalar(
        select(IncidentRecord).where(
            IncidentRecord.tenant_id == identity.tenant_id,
            IncidentRecord.incident_id == work.incident_id,
        )
    )
    if incident is None:
        raise WorkOrderNotVisible
    site_id = session.scalar(
        select(AssetSiteLinkRecord.site_id).where(
            AssetSiteLinkRecord.tenant_id == identity.tenant_id,
            AssetSiteLinkRecord.asset_id == incident.asset_id,
        )
    )
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
    authorizer.require(
        identity,
        Action.EXECUTE_WORK_ORDER,
        resource,
        request_id=request_id,
    )
    if work.assigned_subject_id != identity.subject_id:
        raise WorkOrderConflict("assignee_mismatch", work.version)
    if work.status not in FIELD_RECOGNITION_WORK_ORDER_STATUSES:
        raise WorkOrderConflict("field_recognition_state_invalid", work.version)
    if expected_work_order_version is not None and work.version != expected_work_order_version:
        raise WorkOrderConflict("version_conflict", work.version)

    upload = session.scalar(
        select(WorkOrderFieldEvidenceUploadRecord).where(
            WorkOrderFieldEvidenceUploadRecord.tenant_id == identity.tenant_id,
            WorkOrderFieldEvidenceUploadRecord.evidence_upload_id == field_evidence_upload_id,
            WorkOrderFieldEvidenceUploadRecord.work_order_id == work_order_id,
            WorkOrderFieldEvidenceUploadRecord.incident_id == incident.incident_id,
            WorkOrderFieldEvidenceUploadRecord.source_draft_id == incident.source_draft_id,
        )
    )
    if upload is None:
        raise WorkOrderNotVisible
    media = MediaObjectRepository(session, identity.tenant_context).get(upload.media_id)
    asset = AssetRepository(session, identity.tenant_context).get(incident.asset_id)
    if media is None or asset is None:
        raise WorkOrderNotVisible
    binding_matches = (
        media.draft_id == upload.source_draft_id
        and media.content_hash == upload.content_hash
        and media.declared_mime == upload.declared_mime
        and media.size_bytes == upload.size_bytes
    )
    if not binding_matches:
        raise WorkOrderConflict("field_recognition_binding_invalid", work.version)
    if media.scan_state != ScanState.CLEAN.value:
        raise WorkOrderConflict("field_recognition_media_not_clean", work.version)
    detected_mime = (media.detected_mime or media.declared_mime).casefold()
    if detected_mime not in FIELD_RECOGNITION_MIME_TYPES:
        raise WorkOrderConflict("field_recognition_image_required", work.version)
    return _FieldRecognitionContext(
        work=work,
        incident=incident,
        upload=upload,
        media=media,
        asset_model=asset.model_code or "unknown-model",
    )


def _field_processor_profile(
    detected_mime: str,
    *,
    temporal_analysis: bool,
    current_work_order_version: int,
) -> str:
    if detected_mime in FIELD_IMAGE_RECOGNITION_MIME_TYPES:
        if temporal_analysis:
            raise WorkOrderConflict(
                "field_recognition_temporal_requires_video",
                current_work_order_version,
            )
        return "local-core-v1"
    if detected_mime in FIELD_VIDEO_RECOGNITION_MIME_TYPES:
        return "video-temporal-v1" if temporal_analysis else "local-video-v1"
    raise WorkOrderConflict(
        "field_recognition_image_required",
        current_work_order_version,
    )


def _validate_field_processor_profile(detected_mime: str, processor_profile: str) -> None:
    allowed_profiles = (
        {"local-core-v1"}
        if detected_mime in FIELD_IMAGE_RECOGNITION_MIME_TYPES
        else {"local-video-v1", "video-temporal-v1"}
        if detected_mime in FIELD_VIDEO_RECOGNITION_MIME_TYPES
        else set()
    )
    if processor_profile not in allowed_profiles:
        raise DraftStateConflict("field recognition processor profile is invalid")


def _canonical_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(canonical.encode()).hexdigest()


def _to_recognition_run(record: RecognitionRunRecord) -> RecognitionRun:
    return RecognitionRun(
        recognition_run_id=record.recognition_run_id,
        draft_id=record.draft_id,
        media_id=record.media_id,
        context_kind=record.context_kind,
        work_order_id=record.work_order_id,
        field_evidence_upload_id=record.field_evidence_upload_id,
        work_order_version=record.work_order_version,
        requested_by_subject_id=record.requested_by_subject_id,
        workflow_id=record.workflow_id,
        status=record.status,
        processor_profile=record.processor_profile,
        evidence_bundle_id=record.evidence_bundle_id,
        failure_reason=record.failure_reason,
        version=record.version,
        created_at=_as_utc(record.created_at),
        updated_at=_as_utc(record.updated_at),
    )


def _evidence_record(
    bundle: EvidenceBundle,
    run: RecognitionRunRecord,
) -> EvidenceBundleRecord:
    return EvidenceBundleRecord(
        bundle_id=bundle.bundle_id,
        tenant_id=bundle.tenant_id,
        draft_id=bundle.draft_id,
        asset_id=bundle.asset_id,
        media_id=bundle.media_id,
        context_kind=run.context_kind,
        work_order_id=run.work_order_id,
        field_evidence_upload_id=run.field_evidence_upload_id,
        source_sha256=bundle.source_sha256,
        source_type=bundle.source_type,
        processor_versions=bundle.processor_versions,
        security_policy_version=bundle.security_policy_version,
        security_findings_json=[item.as_dict() for item in bundle.security_findings],
        automation_eligible=bundle.automation_eligible,
        automation_blockers_json=list(bundle.automation_blockers),
        status=bundle.status.value,
        version=bundle.version,
        created_at=bundle.created_at,
        updated_at=bundle.updated_at,
    )


def _bbox_payload(box: BoundingBox) -> dict[str, float]:
    return {"x": box.x, "y": box.y, "width": box.width, "height": box.height}


def _evidence_result_records(bundle: EvidenceBundle) -> tuple[RecognitionResultRecord, ...]:
    records: list[RecognitionResultRecord] = []
    grouped: tuple[tuple[str, tuple[object, ...]], ...] = (
        ("OCR_BLOCK", bundle.ocr_blocks),
        ("ASR_SEGMENT", bundle.asr_segments),
        ("ENTITY", bundle.extracted_entities),
        ("VISUAL_FINDING", bundle.visual_findings),
        ("VIDEO_KEYFRAME", bundle.video_keyframes),
        ("VIDEO_EVENT", bundle.video_events),
        ("QR_CODE", bundle.qr_codes),
    )
    for result_type, values in grouped:
        for ordinal, value in enumerate(values, start=1):
            payload: dict[str, object]
            if isinstance(value, OcrBlock):
                payload = {
                    "block_id": value.block_id,
                    "page_number": value.page_number,
                    "text": value.text,
                    "bbox": _bbox_payload(value.bbox),
                    "confidence": value.confidence,
                    "source_frame_id": value.source_frame_id,
                }
            elif isinstance(value, AsrSegment):
                payload = {
                    "segment_id": value.segment_id,
                    "start_ms": value.start_ms,
                    "end_ms": value.end_ms,
                    "text": value.text,
                    "language": value.language,
                    "confidence": value.confidence,
                    "model_release_id": value.model_release_id,
                    "source_audio_track_id": value.source_audio_track_id,
                    "hotword_profile_id": value.hotword_profile_id,
                    "entity_candidates": [
                        {
                            "entity_id": item.entity_id,
                            "source_segment_id": item.source_segment_id,
                            "entity_type": item.entity_type,
                            "value": item.value,
                            "normalized_value": item.normalized_value,
                            "confidence": item.confidence,
                            "start_offset": item.start_offset,
                            "end_offset": item.end_offset,
                            "requires_confirmation": item.requires_confirmation,
                        }
                        for item in value.entity_candidates
                    ],
                }
            elif isinstance(value, ExtractedEntity):
                payload = {
                    "entity_id": value.entity_id,
                    "entity_type": value.entity_type,
                    "value": value.value,
                    "normalized_value": value.normalized_value,
                    "confidence": value.confidence,
                    "validation_status": value.validation_status.value,
                    "source_block_id": value.source_block_id,
                    "validation_reason": value.validation_reason,
                }
            elif isinstance(value, VisualFinding):
                payload = {
                    "finding_id": value.finding_id,
                    "label": value.label,
                    "bbox": _bbox_payload(value.bbox),
                    "confidence": value.confidence,
                    "evidence_level": value.evidence_level,
                    "model_release_id": value.model_release_id,
                    "description": value.description,
                    "source_frame_id": value.source_frame_id,
                }
            elif isinstance(value, VideoKeyframe):
                payload = {
                    "frame_id": value.frame_id,
                    "timestamp_ms": value.timestamp_ms,
                    "image_sha256": value.image_sha256,
                    "sampling_reason": value.sampling_reason,
                }
            elif isinstance(value, VideoEvent):
                payload = {
                    "event_id": value.event_id,
                    "event_type": value.event_type,
                    "start_ms": value.start_ms,
                    "end_ms": value.end_ms,
                    "keyframe_ids": list(value.keyframe_ids),
                    "finding_ids": list(value.finding_ids),
                    "model_release_id": value.model_release_id,
                    "confidence": value.confidence,
                    "description": value.description,
                }
            elif isinstance(value, QrCodeCandidate):
                payload = {
                    "candidate_id": value.candidate_id,
                    "text": value.text,
                    "payload_kind": value.payload_kind.value,
                    "bbox": _bbox_payload(value.bbox),
                    "source_frame_id": value.source_frame_id,
                    "security_findings": [item.as_dict() for item in value.security_findings],
                }
            else:  # pragma: no cover - the closed tuple above controls this branch
                raise TypeError("unsupported recognition result")
            records.append(
                RecognitionResultRecord(
                    result_id=f"result-{uuid4().hex}",
                    tenant_id=bundle.tenant_id,
                    bundle_id=bundle.bundle_id,
                    result_type=result_type,
                    ordinal=ordinal,
                    payload=payload,
                    created_at=bundle.created_at,
                    updated_at=bundle.updated_at,
                )
            )
    return tuple(records)


def _to_evidence_bundle(
    record: EvidenceBundleRecord,
    results: list[RecognitionResultRecord],
    corrections: list[RecognitionCorrectionRecord],
) -> EvidenceBundle:
    ocr_blocks: list[OcrBlock] = []
    asr_segments: list[AsrSegment] = []
    entities: list[ExtractedEntity] = []
    findings: list[VisualFinding] = []
    keyframes: list[VideoKeyframe] = []
    video_events: list[VideoEvent] = []
    qr_codes: list[QrCodeCandidate] = []
    for result in results:
        payload = result.payload
        if result.result_type == "OCR_BLOCK":
            ocr_blocks.append(
                OcrBlock(
                    block_id=str(payload["block_id"]),
                    page_number=int(payload["page_number"]),
                    text=str(payload["text"]),
                    bbox=_bbox_from_payload(payload["bbox"]),
                    confidence=float(payload["confidence"]),
                    source_frame_id=_optional_string(payload.get("source_frame_id")),
                )
            )
        elif result.result_type == "ASR_SEGMENT":
            asr_segments.append(
                AsrSegment(
                    segment_id=str(payload["segment_id"]),
                    start_ms=int(payload["start_ms"]),
                    end_ms=int(payload["end_ms"]),
                    text=str(payload["text"]),
                    language=str(payload["language"]),
                    confidence=float(payload["confidence"]),
                    model_release_id=str(payload["model_release_id"]),
                    source_audio_track_id=_optional_string(payload.get("source_audio_track_id")),
                    hotword_profile_id=_optional_string(payload.get("hotword_profile_id")),
                    entity_candidates=_asr_entity_candidates_from_payload(
                        str(payload["segment_id"]),
                        payload.get("entity_candidates"),
                    ),
                )
            )
        elif result.result_type == "ENTITY":
            reason = payload.get("validation_reason")
            entities.append(
                ExtractedEntity(
                    entity_id=str(payload["entity_id"]),
                    entity_type=str(payload["entity_type"]),
                    value=str(payload["value"]),
                    normalized_value=str(payload["normalized_value"]),
                    confidence=float(payload["confidence"]),
                    validation_status=EntityValidationStatus(str(payload["validation_status"])),
                    source_block_id=str(payload["source_block_id"]),
                    validation_reason=str(reason) if reason is not None else None,
                )
            )
        elif result.result_type == "VISUAL_FINDING":
            findings.append(
                VisualFinding(
                    finding_id=str(payload["finding_id"]),
                    label=str(payload["label"]),
                    bbox=_bbox_from_payload(payload["bbox"]),
                    confidence=float(payload["confidence"]),
                    evidence_level=str(payload["evidence_level"]),
                    model_release_id=str(payload["model_release_id"]),
                    description=str(payload["description"]),
                    source_frame_id=_optional_string(payload.get("source_frame_id")),
                )
            )
        elif result.result_type == "VIDEO_KEYFRAME":
            keyframes.append(
                VideoKeyframe(
                    frame_id=str(payload["frame_id"]),
                    timestamp_ms=int(payload["timestamp_ms"]),
                    image_sha256=str(payload["image_sha256"]),
                    sampling_reason=str(payload["sampling_reason"]),
                )
            )
        elif result.result_type == "VIDEO_EVENT":
            keyframe_ids = payload["keyframe_ids"]
            finding_ids = payload.get("finding_ids", [])
            if not isinstance(keyframe_ids, list) or not isinstance(finding_ids, list):
                raise RuntimeError("stored video event references are invalid")
            video_events.append(
                VideoEvent(
                    event_id=str(payload["event_id"]),
                    event_type=str(payload["event_type"]),
                    start_ms=int(payload["start_ms"]),
                    end_ms=int(payload["end_ms"]),
                    keyframe_ids=tuple(str(item) for item in keyframe_ids),
                    finding_ids=tuple(str(item) for item in finding_ids),
                    model_release_id=_optional_string(payload.get("model_release_id")),
                    confidence=float(payload["confidence"]),
                    description=str(payload["description"]),
                )
            )
        elif result.result_type == "QR_CODE":
            qr_codes.append(
                QrCodeCandidate(
                    candidate_id=str(payload["candidate_id"]),
                    text=str(payload["text"]),
                    payload_kind=QrPayloadKind(str(payload["payload_kind"])),
                    bbox=_bbox_from_payload(payload["bbox"]),
                    source_frame_id=_optional_string(payload.get("source_frame_id")),
                    security_findings=_security_findings_from_payload(
                        payload.get("security_findings", [])
                    ),
                )
            )
    human_corrections = tuple(
        HumanCorrection(
            correction_id=item.correction_id,
            entity_id=item.target_id,
            original_value=item.original_value or "",
            corrected_value=item.corrected_value or "",
            reviewer_subject_id=item.reviewer_subject_id,
            occurred_at=_as_utc(item.occurred_at),
        )
        for item in corrections
        if item.target_type == "ENTITY"
    )
    ocr_reviews = tuple(
        OcrReview(
            block_id=item.target_id,
            disposition=OcrDisposition(item.disposition or ""),
            original_text=item.original_value or "",
            corrected_text=item.corrected_value,
            reviewer_subject_id=item.reviewer_subject_id,
            occurred_at=_as_utc(item.occurred_at),
        )
        for item in corrections
        if item.target_type == "OCR_BLOCK"
    )
    finding_reviews = tuple(
        FindingReview(
            finding_id=item.target_id,
            disposition=FindingDisposition(item.disposition or ""),
            reviewer_subject_id=item.reviewer_subject_id,
            occurred_at=_as_utc(item.occurred_at),
        )
        for item in corrections
        if item.target_type == "FINDING"
    )
    transcript_reviews = tuple(
        TranscriptReview(
            segment_id=item.target_id,
            disposition=TranscriptDisposition(item.disposition or ""),
            original_text=item.original_value or "",
            corrected_text=item.corrected_value,
            reviewer_subject_id=item.reviewer_subject_id,
            occurred_at=_as_utc(item.occurred_at),
        )
        for item in corrections
        if item.target_type == "ASR_SEGMENT"
    )
    video_event_reviews = tuple(
        VideoEventReview(
            event_id=item.target_id,
            disposition=FindingDisposition(item.disposition or ""),
            reviewer_subject_id=item.reviewer_subject_id,
            occurred_at=_as_utc(item.occurred_at),
        )
        for item in corrections
        if item.target_type == "VIDEO_EVENT"
    )
    qr_reviews = tuple(
        QrReview(
            candidate_id=item.target_id,
            disposition=QrDisposition(item.disposition or ""),
            original_text=item.original_value or "",
            corrected_text=item.corrected_value,
            reviewer_subject_id=item.reviewer_subject_id,
            occurred_at=_as_utc(item.occurred_at),
        )
        for item in corrections
        if item.target_type == "QR_CODE"
    )
    return EvidenceBundle(
        bundle_id=record.bundle_id,
        tenant_id=record.tenant_id,
        draft_id=record.draft_id,
        asset_id=record.asset_id,
        media_id=record.media_id,
        source_sha256=record.source_sha256,
        source_type=record.source_type,
        ocr_blocks=tuple(ocr_blocks),
        asr_segments=tuple(asr_segments),
        visual_findings=tuple(findings),
        video_keyframes=tuple(keyframes),
        video_events=tuple(video_events),
        extracted_entities=tuple(entities),
        human_corrections=human_corrections,
        ocr_reviews=ocr_reviews,
        finding_reviews=finding_reviews,
        transcript_reviews=transcript_reviews,
        video_event_reviews=video_event_reviews,
        qr_codes=tuple(qr_codes),
        qr_reviews=qr_reviews,
        processor_versions=record.processor_versions,
        status=EvidenceStatus(record.status),
        version=record.version,
        created_at=_as_utc(record.created_at),
        updated_at=_as_utc(record.updated_at),
        security_policy_version=record.security_policy_version,
        security_findings=_security_findings_from_payload(record.security_findings_json),
        automation_eligible=record.automation_eligible,
        automation_blockers=tuple(record.automation_blockers_json),
    )


def _bbox_from_payload(value: object) -> BoundingBox:
    if not isinstance(value, dict):
        raise RuntimeError("stored recognition bounding box is invalid")
    return BoundingBox(
        x=float(value["x"]),
        y=float(value["y"]),
        width=float(value["width"]),
        height=float(value["height"]),
    )


def _security_findings_from_payload(
    value: object,
) -> tuple[EvidenceSecurityFinding, ...]:
    if not isinstance(value, list):
        raise RuntimeError("stored multimodal security findings are invalid")
    findings: list[EvidenceSecurityFinding] = []
    for payload in value:
        if not isinstance(payload, dict):
            raise RuntimeError("stored multimodal security finding is invalid")
        try:
            findings.append(
                EvidenceSecurityFinding(
                    source_type=str(payload["source_type"]),
                    source_id=str(payload["source_id"]),
                    policy_version=str(payload["policy_version"]),
                    pattern_id=str(payload["pattern_id"]),
                    category=str(payload["category"]),
                    severity=str(payload["severity"]),
                    content_hash=str(payload["content_hash"]),
                )
            )
        except (KeyError, EvidenceSchemaError) as exc:
            raise RuntimeError("stored multimodal security finding is invalid") from exc
    return tuple(findings)


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _asr_entity_candidates_from_payload(
    segment_id: str,
    value: object,
) -> tuple[AsrEntityCandidate, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RuntimeError("stored ASR entity candidates are invalid")
    rows: list[AsrEntityCandidate] = []
    for payload in value:
        if not isinstance(payload, dict) or not isinstance(
            payload.get("requires_confirmation"), bool
        ):
            raise RuntimeError("stored ASR entity candidate is invalid")
        rows.append(
            AsrEntityCandidate(
                entity_id=str(payload["entity_id"]),
                source_segment_id=str(payload.get("source_segment_id", segment_id)),
                entity_type=str(payload["entity_type"]),
                value=str(payload["value"]),
                normalized_value=str(payload["normalized_value"]),
                confidence=float(payload["confidence"]),
                start_offset=int(payload["start_offset"]),
                end_offset=int(payload["end_offset"]),
                requires_confirmation=payload["requires_confirmation"],
            )
        )
    return tuple(rows)


def _entity_correction_record(
    tenant_id: str,
    bundle_id: str,
    bundle: EvidenceBundle,
    correction: HumanCorrection,
) -> RecognitionCorrectionRecord:
    return RecognitionCorrectionRecord(
        correction_id=correction.correction_id,
        tenant_id=tenant_id,
        bundle_id=bundle_id,
        target_type="ENTITY",
        target_id=correction.entity_id,
        original_value=correction.original_value,
        corrected_value=correction.corrected_value,
        disposition=None,
        reviewer_subject_id=correction.reviewer_subject_id,
        occurred_at=correction.occurred_at,
        evidence_version=bundle.version,
        created_at=correction.occurred_at,
        updated_at=correction.occurred_at,
    )


def _finding_review_record(
    tenant_id: str,
    bundle_id: str,
    bundle: EvidenceBundle,
    review: FindingReview,
) -> RecognitionCorrectionRecord:
    key = f"{bundle_id}\0{review.finding_id}\0{bundle.version}\0{review.disposition.value}"
    return RecognitionCorrectionRecord(
        correction_id=f"review-{sha256(key.encode()).hexdigest()}",
        tenant_id=tenant_id,
        bundle_id=bundle_id,
        target_type="FINDING",
        target_id=review.finding_id,
        original_value=None,
        corrected_value=None,
        disposition=review.disposition.value,
        reviewer_subject_id=review.reviewer_subject_id,
        occurred_at=review.occurred_at,
        evidence_version=bundle.version,
        created_at=review.occurred_at,
        updated_at=review.occurred_at,
    )


def _ocr_review_record(
    tenant_id: str,
    bundle_id: str,
    bundle: EvidenceBundle,
    review: OcrReview,
) -> RecognitionCorrectionRecord:
    key = f"{bundle_id}\0{review.block_id}\0{bundle.version}\0{review.disposition.value}"
    return RecognitionCorrectionRecord(
        correction_id=f"ocr-review-{sha256(key.encode()).hexdigest()}",
        tenant_id=tenant_id,
        bundle_id=bundle_id,
        target_type="OCR_BLOCK",
        target_id=review.block_id,
        original_value=review.original_text,
        corrected_value=review.corrected_text,
        disposition=review.disposition.value,
        reviewer_subject_id=review.reviewer_subject_id,
        occurred_at=review.occurred_at,
        evidence_version=bundle.version,
        created_at=review.occurred_at,
        updated_at=review.occurred_at,
    )


def _qr_review_record(
    tenant_id: str,
    bundle_id: str,
    bundle: EvidenceBundle,
    review: QrReview,
) -> RecognitionCorrectionRecord:
    key = f"{bundle_id}\0{review.candidate_id}\0{bundle.version}\0{review.disposition.value}"
    return RecognitionCorrectionRecord(
        correction_id=f"qr-review-{sha256(key.encode()).hexdigest()}",
        tenant_id=tenant_id,
        bundle_id=bundle_id,
        target_type="QR_CODE",
        target_id=review.candidate_id,
        original_value=review.original_text,
        corrected_value=review.corrected_text,
        disposition=review.disposition.value,
        reviewer_subject_id=review.reviewer_subject_id,
        occurred_at=review.occurred_at,
        evidence_version=bundle.version,
        created_at=review.occurred_at,
        updated_at=review.occurred_at,
    )


def _transcript_review_record(
    tenant_id: str,
    bundle_id: str,
    bundle: EvidenceBundle,
    review: TranscriptReview,
) -> RecognitionCorrectionRecord:
    key = f"{bundle_id}\0{review.segment_id}\0{bundle.version}\0{review.disposition.value}"
    return RecognitionCorrectionRecord(
        correction_id=f"transcript-review-{sha256(key.encode()).hexdigest()}",
        tenant_id=tenant_id,
        bundle_id=bundle_id,
        target_type="ASR_SEGMENT",
        target_id=review.segment_id,
        original_value=review.original_text,
        corrected_value=review.corrected_text,
        disposition=review.disposition.value,
        reviewer_subject_id=review.reviewer_subject_id,
        occurred_at=review.occurred_at,
        evidence_version=bundle.version,
        created_at=review.occurred_at,
        updated_at=review.occurred_at,
    )


def _video_event_review_record(
    tenant_id: str,
    bundle_id: str,
    bundle: EvidenceBundle,
    review: VideoEventReview,
) -> RecognitionCorrectionRecord:
    key = f"{bundle_id}\0{review.event_id}\0{bundle.version}\0{review.disposition.value}"
    return RecognitionCorrectionRecord(
        correction_id=f"video-event-review-{sha256(key.encode()).hexdigest()}",
        tenant_id=tenant_id,
        bundle_id=bundle_id,
        target_type="VIDEO_EVENT",
        target_id=review.event_id,
        original_value=None,
        corrected_value=None,
        disposition=review.disposition.value,
        reviewer_subject_id=review.reviewer_subject_id,
        occurred_at=review.occurred_at,
        evidence_version=bundle.version,
        created_at=review.occurred_at,
        updated_at=review.occurred_at,
    )


def _incident_record(incident: Incident) -> IncidentRecord:
    return IncidentRecord(
        incident_id=incident.incident_id,
        tenant_id=incident.tenant_id,
        asset_id=incident.asset_id,
        source_draft_id=incident.source_draft_id,
        evidence_bundle_id=incident.evidence_bundle_id,
        reporter_subject_id=incident.reporter_subject_id,
        description=incident.description,
        status=incident.status,
        version=incident.version,
        created_at=incident.created_at,
        updated_at=incident.updated_at,
    )


def _to_incident(
    record: IncidentRecord,
    *,
    diagnosis_run_ids: tuple[str, ...] = (),
) -> Incident:
    return Incident(
        incident_id=record.incident_id,
        tenant_id=record.tenant_id,
        asset_id=record.asset_id,
        source_draft_id=record.source_draft_id,
        evidence_bundle_id=record.evidence_bundle_id,
        reporter_subject_id=record.reporter_subject_id,
        description=record.description,
        status=record.status,
        version=record.version,
        diagnosis_run_ids=diagnosis_run_ids,
        created_at=_as_utc(record.created_at),
        updated_at=_as_utc(record.updated_at),
    )


def _recognition_failure_code(exc: Exception) -> str:
    if isinstance(exc, IncidentSubmissionBlocked):
        return exc.reason_code
    if isinstance(exc, DraftStateConflict):
        return "recognition_state_changed"
    if isinstance(exc, (WorkOrderConflict, WorkOrderNotVisible, AuthorizationDenied)):
        return "recognition_state_changed"
    if isinstance(exc, ModelGatewayError):
        return exc.reason
    if isinstance(exc, MultimodalProviderUnavailable):
        return "multimodal_provider_unavailable"
    if isinstance(exc, EvidenceSchemaError):
        return "evidence_schema_invalid"
    return "processor_failed"
