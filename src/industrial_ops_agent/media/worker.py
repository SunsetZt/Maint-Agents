"""Fail-closed media scan worker and ClamAV INSTREAM adapter."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Protocol

from industrial_ops_agent.media.content_credentials import (
    C2paContentCredentialInspector,
    ContentCredentialInspector,
    ContentCredentialReport,
)
from industrial_ops_agent.media.models import (
    MediaObject,
    MediaStateConflict,
    ScanState,
)
from industrial_ops_agent.media.service import (
    MAX_MEDIA_BYTES,
    TenantObjectStore,
    append_scan_event,
    media_from_record,
)
from industrial_ops_agent.media.validation import MediaValidationFailure, inspect_media
from industrial_ops_agent.object_store.keys import (
    ObjectZone,
    TenantObjectKey,
    clean_object_key,
    quarantine_object_key,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.errors import InfrastructureUnavailable
from industrial_ops_agent.persistence.repositories import MediaObjectRepository
from industrial_ops_agent.persistence.tenant import TenantContext


class ScanVerdict(StrEnum):
    CLEAN = "CLEAN"
    INFECTED = "INFECTED"


class MalwareScanner(Protocol):
    async def scan(self, content: bytes) -> ScanVerdict: ...


class MediaScanWorker:
    def __init__(
        self,
        database: Database,
        object_store: TenantObjectStore,
        scanner: MalwareScanner,
        credential_inspector: ContentCredentialInspector | None = None,
    ) -> None:
        self._database = database
        self._object_store = object_store
        self._scanner = scanner
        self._credential_inspector = (
            credential_inspector or C2paContentCredentialInspector()
        )

    async def process(self, context: TenantContext, media_id: str) -> MediaObject:
        current = self._pending(context, media_id)
        expected_quarantine = quarantine_object_key(
            context,
            current.draft_id,
            current.media_id,
        )
        if current.quarantine_key != expected_quarantine.value:
            return self._finish(context, current, ScanState.REJECTED, clean_key=None)
        content = await self._object_store.get(context, expected_quarantine)
        if content is None or not self._content_matches(current, content):
            return self._finish(context, current, ScanState.REJECTED, clean_key=None)
        try:
            inspection = inspect_media(
                content,
                declared_mime=current.declared_mime,
                max_bytes=MAX_MEDIA_BYTES,
            )
        except MediaValidationFailure:
            return self._finish(context, current, ScanState.REJECTED, clean_key=None)
        if (
            inspection.content_hash != current.content_hash
            or inspection.detected_mime != current.detected_mime
        ):
            return self._finish(context, current, ScanState.REJECTED, clean_key=None)

        verdict = await self._scanner.scan(content)
        if verdict is ScanVerdict.INFECTED:
            return self._finish(context, current, ScanState.INFECTED, clean_key=None)

        assert inspection.detected_mime is not None
        credential = await asyncio.to_thread(
            self._credential_inspector.inspect,
            content,
            inspection.detected_mime,
        )

        clean_key = clean_object_key(context, current.draft_id, current.media_id)
        await self._object_store.put(context, clean_key, content)
        promoted = await self._object_store.get(context, clean_key)
        if promoted is None or sha256(promoted).hexdigest() != current.content_hash:
            return self._finish(context, current, ScanState.REJECTED, clean_key=None)
        return self._finish(
            context,
            current,
            ScanState.CLEAN,
            clean_key=clean_key.value,
            content_credential=credential,
        )

    def _pending(self, context: TenantContext, media_id: str) -> MediaObject:
        with self._database.transaction(context) as session:
            record = MediaObjectRepository(session, context).get(media_id)
            if record is None:
                raise MediaStateConflict("media is not visible to the worker tenant")
            current = media_from_record(record)
            if current.scan_state is not ScanState.PENDING:
                raise MediaStateConflict("media scan result is already terminal")
            return current

    def _finish(
        self,
        context: TenantContext,
        current: MediaObject,
        state: ScanState,
        *,
        clean_key: str | None,
        content_credential: ContentCredentialReport | None = None,
    ) -> MediaObject:
        occurred_at = datetime.now(UTC)
        completed = current.complete_scan(
            state=state,
            clean_key=clean_key,
            occurred_at=occurred_at,
            content_credential=content_credential,
        )
        with self._database.transaction(context) as session:
            repository = MediaObjectRepository(session, context)
            changed = repository.complete_scan(
                media_id=current.media_id,
                expected_version=current.scan_version,
                scan_state=state.value,
                clean_key=clean_key,
                content_credential_status=completed.content_credential_status.value,
                content_credential_report=completed.content_credential_report,
                content_credential_digest=completed.content_credential_digest,
                content_credential_inspected_at=completed.content_credential_inspected_at,
                updated_at=occurred_at,
            )
            if not changed:
                raise MediaStateConflict("media scan result lost a concurrency race")
            append_scan_event(
                session,
                context,
                completed,
                occurred_at=occurred_at,
            )
        return completed

    @staticmethod
    def _content_matches(media: MediaObject, content: bytes) -> bool:
        return (
            len(content) == media.size_bytes
            and sha256(content).hexdigest() == media.content_hash
        )


class ClamAvScanner:
    """Minimal bounded ClamAV INSTREAM client used by the scan worker."""

    def __init__(
        self,
        host: str = "clamav",
        port: int = 3310,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout_seconds = timeout_seconds

    async def scan(self, content: bytes) -> ScanVerdict:
        try:
            return await asyncio.wait_for(
                self._scan_stream(content),
                timeout=self._timeout_seconds,
            )
        except (
            OSError,
            TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ) as exc:
            raise InfrastructureUnavailable("clamav") from exc

    async def _scan_stream(self, content: bytes) -> ScanVerdict:
        reader, writer = await asyncio.open_connection(self._host, self._port)
        try:
            writer.write(b"zINSTREAM\0")
            for offset in range(0, len(content), 8192):
                chunk = content[offset : offset + 8192]
                writer.write(len(chunk).to_bytes(4, "big"))
                writer.write(chunk)
            writer.write((0).to_bytes(4, "big"))
            await writer.drain()
            response = await reader.readuntil(b"\0")
        finally:
            writer.close()
            await writer.wait_closed()
        if response.endswith(b"OK\0"):
            return ScanVerdict.CLEAN
        if b"FOUND" in response:
            return ScanVerdict.INFECTED
        raise InfrastructureUnavailable("clamav")


def clean_key_from_value(context: TenantContext, value: str) -> TenantObjectKey:
    """Rebuild a clean key only after its canonical value has been checked."""

    prefix = f"tenant/{context.tenant_id}/clean/"
    if not value.startswith(prefix):
        raise MediaStateConflict("stored clean key is outside the worker tenant")
    return TenantObjectKey(context.tenant_id, ObjectZone.CLEAN, value)
