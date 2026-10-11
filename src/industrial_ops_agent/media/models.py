"""Immutable media scan state shared by the API and scanner worker."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Any

from industrial_ops_agent.media.content_credentials import (
    ContentCredentialReport,
    ContentCredentialStatus,
)


class ScanState(StrEnum):
    PENDING = "PENDING"
    CLEAN = "CLEAN"
    INFECTED = "INFECTED"
    REJECTED = "REJECTED"


class MediaStateConflict(Exception):
    """A terminal or stale media state cannot be overwritten."""


@dataclass(frozen=True, slots=True)
class MediaObject:
    media_id: str
    tenant_id: str
    draft_id: str
    quarantine_key: str
    clean_key: str | None
    content_hash: str
    declared_mime: str
    detected_mime: str | None
    size_bytes: int
    scan_state: ScanState
    scan_version: int
    content_credential_status: ContentCredentialStatus
    content_credential_report: dict[str, Any] | None
    content_credential_digest: str | None
    content_credential_inspected_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls,
        *,
        media_id: str,
        tenant_id: str,
        draft_id: str,
        quarantine_key: str,
        content_hash: str,
        declared_mime: str,
        detected_mime: str | None,
        size_bytes: int,
        occurred_at: datetime,
        state: ScanState = ScanState.PENDING,
    ) -> MediaObject:
        expected = f"tenant/{tenant_id}/quarantine/{draft_id}/{media_id}"
        if quarantine_key != expected:
            raise ValueError("media quarantine key is not canonical")
        if len(content_hash) != 64:
            raise ValueError("media content hash is invalid")
        if size_bytes < 0:
            raise ValueError("media size cannot be negative")
        if state not in {ScanState.PENDING, ScanState.REJECTED}:
            raise ValueError("initial media scan state is invalid")
        return cls(
            media_id=media_id,
            tenant_id=tenant_id,
            draft_id=draft_id,
            quarantine_key=quarantine_key,
            clean_key=None,
            content_hash=content_hash,
            declared_mime=declared_mime,
            detected_mime=detected_mime,
            size_bytes=size_bytes,
            scan_state=state,
            scan_version=1,
            content_credential_status=(
                ContentCredentialStatus.PENDING_INSPECTION
                if state is ScanState.PENDING
                else ContentCredentialStatus.NOT_INSPECTED
            ),
            content_credential_report=None,
            content_credential_digest=None,
            content_credential_inspected_at=None,
            created_at=occurred_at,
            updated_at=occurred_at,
        )

    def complete_scan(
        self,
        *,
        state: ScanState,
        clean_key: str | None,
        occurred_at: datetime,
        content_credential: ContentCredentialReport | None = None,
    ) -> MediaObject:
        if self.scan_state is not ScanState.PENDING:
            raise MediaStateConflict("media scan result is already terminal")
        if state not in {ScanState.CLEAN, ScanState.INFECTED, ScanState.REJECTED}:
            raise ValueError("terminal media scan state is invalid")
        expected_clean = (
            f"tenant/{self.tenant_id}/clean/{self.draft_id}/{self.media_id}"
        )
        if state is ScanState.CLEAN and clean_key != expected_clean:
            raise ValueError("clean media requires its canonical clean key")
        if state is not ScanState.CLEAN and clean_key is not None:
            raise ValueError("non-clean media cannot have a clean key")
        if state is not ScanState.CLEAN and content_credential is not None:
            raise ValueError("non-clean media cannot publish credential inspection")
        credential_document = (
            content_credential.document() if content_credential is not None else None
        )
        return replace(
            self,
            clean_key=clean_key,
            scan_state=state,
            scan_version=self.scan_version + 1,
            content_credential_status=(
                content_credential.status
                if content_credential is not None
                else ContentCredentialStatus.NOT_INSPECTED
            ),
            content_credential_report=credential_document,
            content_credential_digest=(
                str(credential_document["report_digest"])
                if credential_document is not None
                else None
            ),
            content_credential_inspected_at=(
                occurred_at if content_credential is not None else None
            ),
            updated_at=occurred_at,
        )
