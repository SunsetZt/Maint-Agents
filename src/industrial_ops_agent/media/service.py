"""Authorized media quarantine, status, and clean-read application service."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from hashlib import sha256
from typing import Protocol
from uuid import uuid4

from sqlalchemy.orm import Session

from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.maintenance_planning.review_isolation import (
    draft_read_transaction,
    lock_subject,
    record_draft_disclosure,
)
from industrial_ops_agent.media.content_credentials import ContentCredentialStatus
from industrial_ops_agent.media.models import MediaObject, ScanState
from industrial_ops_agent.media.validation import MediaValidationFailure, inspect_media
from industrial_ops_agent.object_store.keys import (
    ObjectZone,
    TenantObjectKey,
    quarantine_object_key,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    DraftTimelineEventRecord,
    IncidentDraftRecord,
    MediaObjectRecord,
)
from industrial_ops_agent.persistence.repositories import (
    DraftTimelineRepository,
    IncidentDraftRepository,
    MediaObjectRepository,
)
from industrial_ops_agent.persistence.tenant import TenantContext

MAX_MEDIA_BYTES = 10 * 1024 * 1024


class TenantObjectStore(Protocol):
    async def put(
        self,
        context: TenantContext,
        key: TenantObjectKey,
        content: bytes,
    ) -> None: ...

    async def get(
        self,
        context: TenantContext,
        key: TenantObjectKey,
    ) -> bytes | None: ...

    async def delete(
        self,
        context: TenantContext,
        key: TenantObjectKey,
    ) -> None: ...


class MediaNotReadable(Exception):
    """Media has not passed every clean-read guard."""


class MediaUploadRejected(Exception):
    def __init__(self, media: MediaObject, reason_code: str) -> None:
        super().__init__("media upload rejected")
        self.media = media
        self.reason_code = reason_code


class MediaService:
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
        draft_id: str,
        *,
        declared_mime: str,
        content: bytes,
        request_id: str,
    ) -> MediaObject:
        await asyncio.to_thread(
            self._get_visible_draft,
            identity,
            draft_id,
            action=Action.UPLOAD_MEDIA,
            request_id=request_id,
        )
        return await self._upload_authorized(
            identity,
            draft_id,
            declared_mime=declared_mime,
            content=content,
            request_id=request_id,
        )

    async def upload_authorized(
        self,
        identity: IdentityContext,
        draft_id: str,
        *,
        declared_mime: str,
        content: bytes,
        request_id: str,
    ) -> MediaObject:
        """Upload after a production caller has authorized its business resource.

        This entry point deliberately skips IncidentDraft authorization because field
        engineers are authorized against their assigned WorkOrder instead. Callers
        must establish the draft from that authorized server-side business chain.
        """
        return await self._upload_authorized(
            identity,
            draft_id,
            declared_mime=declared_mime,
            content=content,
            request_id=request_id,
        )

    async def _upload_authorized(
        self,
        identity: IdentityContext,
        draft_id: str,
        *,
        declared_mime: str,
        content: bytes,
        request_id: str,
    ) -> MediaObject:
        media_id = f"media-{uuid4().hex}"
        key = quarantine_object_key(identity.tenant_context, draft_id, media_id)
        await self._object_store.put(identity.tenant_context, key, content)
        occurred_at = datetime.now(UTC)
        normalized_mime = declared_mime.split(";", 1)[0].strip().casefold()
        try:
            inspection = inspect_media(
                content,
                declared_mime=normalized_mime,
                max_bytes=MAX_MEDIA_BYTES,
            )
        except MediaValidationFailure as exc:
            media = MediaObject.create(
                media_id=media_id,
                tenant_id=identity.tenant_id,
                draft_id=draft_id,
                quarantine_key=key.value,
                content_hash=sha256(content).hexdigest(),
                declared_mime=normalized_mime,
                detected_mime=exc.detected_mime,
                size_bytes=len(content),
                occurred_at=occurred_at,
                state=ScanState.REJECTED,
            )
            await asyncio.to_thread(
                self._persist_new_authorized,
                identity,
                media,
                event_type="media.upload_rejected",
                reason_code=exc.reason_code,
            )
            raise MediaUploadRejected(media, exc.reason_code) from exc

        media = MediaObject.create(
            media_id=media_id,
            tenant_id=identity.tenant_id,
            draft_id=draft_id,
            quarantine_key=key.value,
            content_hash=inspection.content_hash,
            declared_mime=normalized_mime,
            detected_mime=inspection.detected_mime,
            size_bytes=inspection.size_bytes,
            occurred_at=occurred_at,
        )
        await asyncio.to_thread(
            self._persist_new_authorized,
            identity,
            media,
            event_type="media.uploaded",
            reason_code=None,
        )
        return media

    def get_status(
        self,
        identity: IdentityContext,
        media_id: str,
        *,
        request_id: str,
    ) -> MediaObject:
        return self._status(identity, media_id, request_id=request_id, human_read=True)

    def _status(
        self, identity: IdentityContext, media_id: str, *, request_id: str, human_read: bool
    ) -> MediaObject:
        with self._database.transaction(identity.tenant_context) as session:
            if human_read:
                lock_subject(session, identity)
            record = self._visible_media(
                session,
                identity,
                media_id,
                request_id=request_id,
            )
            if human_read:
                record_draft_disclosure(session, identity, record.draft_id, request_id=request_id)
            return media_from_record(record)

    async def read_clean(
        self,
        identity: IdentityContext,
        media_id: str,
        *,
        request_id: str,
    ) -> bytes:
        # Internal recognition consumption is not evidence that the requester read the source.
        media = await asyncio.to_thread(
            self._status, identity, media_id, request_id=request_id, human_read=False
        )
        expected_key = f"tenant/{identity.tenant_id}/clean/{media.draft_id}/{media.media_id}"
        if media.scan_state is not ScanState.CLEAN or media.clean_key != expected_key:
            self._audit_read_guard(
                identity,
                media_id,
                request_id=request_id,
                decision="deny",
                reason_code="media_not_clean",
            )
            raise MediaNotReadable
        key = TenantObjectKey(
            tenant_id=identity.tenant_id,
            zone=ObjectZone.CLEAN,
            value=expected_key,
        )
        content = await self._object_store.get(identity.tenant_context, key)
        if content is None or sha256(content).hexdigest() != media.content_hash:
            self._audit_read_guard(
                identity,
                media_id,
                request_id=request_id,
                decision="deny",
                reason_code="clean_object_integrity_failed",
            )
            raise MediaNotReadable
        self._audit_read_guard(
            identity,
            media_id,
            request_id=request_id,
            decision="allow",
            reason_code="clean_object_verified",
        )
        return content

    def _audit_read_guard(
        self,
        identity: IdentityContext,
        media_id: str,
        *,
        request_id: str,
        decision: str,
        reason_code: str,
    ) -> None:
        self._authorizer.record_guard_decision(
            identity,
            action="media.read_clean",
            decision=decision,
            reason_code=reason_code,
            request_id=request_id,
            resource_id=media_id,
        )

    def _persist_new_authorized(
        self,
        identity: IdentityContext,
        media: MediaObject,
        *,
        event_type: str,
        reason_code: str | None,
    ) -> None:
        with self._database.transaction(identity.tenant_context) as session:
            MediaObjectRepository(session, identity.tenant_context).add(media_record(media))
            _append_media_event(
                session,
                identity.tenant_context,
                media,
                event_type=event_type,
                occurred_at=media.created_at,
                reason_code=reason_code,
            )

    def _get_visible_draft(
        self,
        identity: IdentityContext,
        draft_id: str,
        *,
        action: Action,
        request_id: str,
    ) -> IncidentDraftRecord:
        with draft_read_transaction(
            self._database, identity, draft_id, request_id=request_id
        ) as session:
            return self._visible_draft(
                session,
                identity,
                draft_id,
                action=action,
                request_id=request_id,
            )

    def _visible_draft(
        self,
        session: Session,
        identity: IdentityContext,
        draft_id: str,
        *,
        action: Action,
        request_id: str,
    ) -> IncidentDraftRecord:
        record = IncidentDraftRepository(session, identity.tenant_context).get(draft_id)
        if record is None:
            raise ResourceNotVisible
        try:
            self._authorizer.require(
                identity,
                action,
                ResourceContext(
                    tenant_id=identity.tenant_id,
                    resource_id=draft_id,
                    asset_id=record.asset_id,
                ),
                request_id=request_id,
            )
        except AuthorizationDenied as exc:
            raise ResourceNotVisible from exc
        return record

    def _visible_media(
        self,
        session: Session,
        identity: IdentityContext,
        media_id: str,
        *,
        request_id: str,
    ) -> MediaObjectRecord:
        media = MediaObjectRepository(session, identity.tenant_context).get(media_id)
        if media is None:
            raise ResourceNotVisible
        self._visible_draft(
            session,
            identity,
            media.draft_id,
            action=Action.READ_ASSET,
            request_id=request_id,
        )
        return media


def media_record(media: MediaObject) -> MediaObjectRecord:
    return MediaObjectRecord(
        media_id=media.media_id,
        tenant_id=media.tenant_id,
        draft_id=media.draft_id,
        quarantine_key=media.quarantine_key,
        clean_key=media.clean_key,
        content_hash=media.content_hash,
        declared_mime=media.declared_mime,
        detected_mime=media.detected_mime,
        size_bytes=media.size_bytes,
        scan_state=media.scan_state.value,
        scan_version=media.scan_version,
        content_credential_status=media.content_credential_status.value,
        content_credential_report=media.content_credential_report,
        content_credential_digest=media.content_credential_digest,
        content_credential_inspected_at=media.content_credential_inspected_at,
        created_at=media.created_at,
        updated_at=media.updated_at,
    )


def media_from_record(record: MediaObjectRecord) -> MediaObject:
    return MediaObject(
        media_id=record.media_id,
        tenant_id=record.tenant_id,
        draft_id=record.draft_id,
        quarantine_key=record.quarantine_key,
        clean_key=record.clean_key,
        content_hash=record.content_hash,
        declared_mime=record.declared_mime,
        detected_mime=record.detected_mime,
        size_bytes=record.size_bytes,
        scan_state=ScanState(record.scan_state),
        scan_version=record.scan_version,
        content_credential_status=ContentCredentialStatus(record.content_credential_status),
        content_credential_report=record.content_credential_report,
        content_credential_digest=record.content_credential_digest,
        content_credential_inspected_at=(
            _as_utc(record.content_credential_inspected_at)
            if record.content_credential_inspected_at is not None
            else None
        ),
        created_at=_as_utc(record.created_at),
        updated_at=_as_utc(record.updated_at),
    )


def _append_media_event(
    session: Session,
    context: TenantContext,
    media: MediaObject,
    *,
    event_type: str,
    occurred_at: datetime,
    reason_code: str | None,
) -> None:
    timeline = DraftTimelineRepository(session, context)
    metadata: dict[str, object] = {
        "media_id": media.media_id,
        "scan_state": media.scan_state.value,
        "content_credential_status": media.content_credential_status.value,
    }
    if reason_code is not None:
        metadata["reason_code"] = reason_code
    timeline.add(
        DraftTimelineEventRecord(
            event_id=f"event-{uuid4().hex}",
            tenant_id=media.tenant_id,
            draft_id=media.draft_id,
            sequence=timeline.next_sequence(media.draft_id),
            event_type=event_type,
            occurred_at=occurred_at,
            payload_metadata=metadata,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
    )


def append_scan_event(
    session: Session,
    context: TenantContext,
    media: MediaObject,
    *,
    occurred_at: datetime,
) -> None:
    _append_media_event(
        session,
        context,
        media,
        event_type=f"media.scan_{media.scan_state.value.casefold()}",
        occurred_at=occurred_at,
        reason_code=None,
    )
