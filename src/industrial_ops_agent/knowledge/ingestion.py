"""Governed text knowledge ingestion, review, chunking, and release construction."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import isfinite
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.domain.json import legacy_canonical_hex_digest as _canonical_digest
from industrial_ops_agent.knowledge.embedding import deterministic_embedding, tokenize
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CitationAnchorRecord,
    IdempotencyRecord,
    IndexReleaseRecord,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    KnowledgeDocumentVersionRecord,
    KnowledgeIndexEvaluationRecord,
)
from industrial_ops_agent.persistence.tenant import validate_boundary_identifier

_MAX_SOURCE_CHARACTERS = 500_000
_MAX_CHUNK_CHARACTERS = 1_200
_MIN_BREAK_CHARACTERS = 500
_MAX_CHUNKS_PER_DOCUMENT = 500
_RELEASE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


class KnowledgeIngestionError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ExtractedKnowledgeStructure:
    """A layout element grounded in the source document."""

    kind: str
    page_number: int
    text: str
    bounding_box: dict[str, float] | None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeVersionView:
    document_id: str
    document_version_id: str
    version: int
    title: str
    source_uri: str
    classification: str
    source_checksum: str
    content_checksum: str
    parser_version: str
    extraction_metadata: dict[str, Any] | None
    status: str
    state_version: int
    acl_subject_ids: tuple[str, ...]
    acl_roles: tuple[str, ...]
    device_families: tuple[str, ...]
    device_models: tuple[str, ...]
    valid_from: datetime
    valid_to: datetime | None
    created_by_subject_id: str | None
    reviewed_by_subject_id: str | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class KnowledgeReleaseView:
    release_id: str
    name: str
    version: int
    status: str
    is_active: bool
    activation_version: int
    rollback_eligible: bool
    content_checksum: str
    chunk_count: int
    created_by_subject_id: str | None
    evaluation_id: str | None
    evaluation_status: str | None
    evaluation_metrics: dict[str, Any]
    evaluation_gate_results: dict[str, bool]
    evaluation_failure_codes: tuple[str, ...]
    evaluated_at: datetime | None
    published_by: str | None
    published_at: datetime | None
    created_at: datetime
    updated_at: datetime


class KnowledgeIngestionService:
    PARSER_VERSION = "governed-text-parser-v1"
    CHUNKER_VERSION = "bounded-paragraph-chunker-v1"

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def create_document(
        self,
        identity: IdentityContext,
        *,
        title: str,
        source_uri: str,
        content: str,
        classification: str,
        acl_subject_ids: tuple[str, ...],
        acl_roles: tuple[str, ...],
        device_families: tuple[str, ...],
        device_models: tuple[str, ...],
        valid_from: datetime,
        valid_to: datetime | None,
        idempotency_key: str,
        request_id: str,
    ) -> KnowledgeVersionView:
        return self._create_version(
            identity,
            document_id=None,
            title=title,
            source_uri=source_uri,
            content=content,
            classification=classification,
            acl_subject_ids=acl_subject_ids,
            acl_roles=acl_roles,
            device_families=device_families,
            device_models=device_models,
            valid_from=valid_from,
            valid_to=valid_to,
            idempotency_key=idempotency_key,
            request_id=request_id,
            parser_version=self.PARSER_VERSION,
            source_checksum_override=None,
            extraction_metadata=None,
        )

    def create_extracted_document(
        self,
        identity: IdentityContext,
        *,
        title: str,
        source_uri: str,
        pages: tuple[tuple[int, str, str], ...],
        structures: tuple[ExtractedKnowledgeStructure, ...],
        parser_version: str,
        source_checksum: str,
        classification: str,
        acl_subject_ids: tuple[str, ...],
        acl_roles: tuple[str, ...],
        device_families: tuple[str, ...],
        device_models: tuple[str, ...],
        valid_from: datetime,
        valid_to: datetime | None,
        idempotency_key: str,
        request_id: str,
    ) -> KnowledgeVersionView:
        content, extraction_metadata = _combine_pages(pages, structures)
        if not re.fullmatch(r"[0-9a-f]{64}", source_checksum):
            raise KnowledgeIngestionError("knowledge_source_checksum_invalid")
        if not parser_version or len(parser_version) > 128:
            raise KnowledgeIngestionError("knowledge_parser_version_invalid")
        return self._create_version(
            identity,
            document_id=None,
            title=title,
            source_uri=source_uri,
            content=content,
            classification=classification,
            acl_subject_ids=acl_subject_ids,
            acl_roles=acl_roles,
            device_families=device_families,
            device_models=device_models,
            valid_from=valid_from,
            valid_to=valid_to,
            idempotency_key=idempotency_key,
            request_id=request_id,
            parser_version=parser_version,
            source_checksum_override=source_checksum,
            extraction_metadata=extraction_metadata,
        )

    def create_version(
        self,
        identity: IdentityContext,
        document_id: str,
        *,
        content: str,
        acl_subject_ids: tuple[str, ...],
        acl_roles: tuple[str, ...],
        device_families: tuple[str, ...],
        device_models: tuple[str, ...],
        valid_from: datetime,
        valid_to: datetime | None,
        idempotency_key: str,
        request_id: str,
    ) -> KnowledgeVersionView:
        return self._create_version(
            identity,
            document_id=document_id,
            title=None,
            source_uri=None,
            content=content,
            classification=None,
            acl_subject_ids=acl_subject_ids,
            acl_roles=acl_roles,
            device_families=device_families,
            device_models=device_models,
            valid_from=valid_from,
            valid_to=valid_to,
            idempotency_key=idempotency_key,
            request_id=request_id,
            parser_version=self.PARSER_VERSION,
            source_checksum_override=None,
            extraction_metadata=None,
        )

    def review_version(
        self,
        identity: IdentityContext,
        document_version_id: str,
        *,
        expected_state_version: int,
        request_id: str,
    ) -> KnowledgeVersionView:
        self._require_manage(identity, document_version_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            version = session.scalar(
                select(KnowledgeDocumentVersionRecord).where(
                    KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id,
                    KnowledgeDocumentVersionRecord.document_version_id == document_version_id,
                )
            )
            if version is None:
                raise KnowledgeIngestionError("knowledge_version_not_visible")
            if (
                version.status == "REVIEWED"
                and version.reviewed_by_subject_id == identity.subject_id
            ):
                return _version_view(session, version)
            if version.status != "DRAFT" or version.state_version != expected_state_version:
                raise KnowledgeIngestionError("knowledge_version_state_changed")
            if version.created_by_subject_id == identity.subject_id:
                raise KnowledgeIngestionError("knowledge_review_separation_required")
            if not version.extracted_text or version.extraction_checksum != _digest(
                version.extracted_text
            ):
                raise KnowledgeIngestionError("knowledge_extraction_integrity_failed")
            version.status = "REVIEWED"
            version.reviewed_by_subject_id = identity.subject_id
            version.reviewed_at = now
            version.state_version += 1
            version.updated_at = now
            session.flush()
            return _version_view(session, version)

    def build_release(
        self,
        identity: IdentityContext,
        *,
        name: str,
        document_version_ids: tuple[str, ...],
        idempotency_key: str,
        request_id: str,
    ) -> KnowledgeReleaseView:
        self._require_manage(identity, name, request_id)
        _validate_idempotency_key(idempotency_key)
        if not _RELEASE_NAME.fullmatch(name):
            raise KnowledgeIngestionError("knowledge_release_name_invalid")
        selected_ids = tuple(sorted(set(document_version_ids)))
        if not selected_ids or len(selected_ids) != len(document_version_ids):
            raise KnowledgeIngestionError("knowledge_release_versions_invalid")
        if len(selected_ids) > 1_000:
            raise KnowledgeIngestionError("knowledge_release_versions_limit_exceeded")
        request_hash = _canonical_digest({"name": name, "version_ids": selected_ids})
        storage_key = f"knowledge-release:{idempotency_key}"
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            replay = _idempotency_replay(
                session,
                identity,
                storage_key=storage_key,
                request_hash=request_hash,
            )
            if replay is not None:
                release = session.scalar(
                    select(IndexReleaseRecord).where(
                        IndexReleaseRecord.tenant_id == identity.tenant_id,
                        IndexReleaseRecord.release_id == replay,
                    )
                )
                if release is None:
                    raise KnowledgeIngestionError("knowledge_idempotency_state_invalid")
                return _release_view(session, release)

            versions = list(
                session.scalars(
                    select(KnowledgeDocumentVersionRecord).where(
                        KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id,
                        KnowledgeDocumentVersionRecord.document_version_id.in_(selected_ids),
                    )
                )
            )
            if len(versions) != len(selected_ids):
                raise KnowledgeIngestionError("knowledge_version_not_visible")
            if len({item.document_id for item in versions}) != len(versions):
                raise KnowledgeIngestionError("knowledge_release_document_version_conflict")
            if any(
                item.status not in {"REVIEWED", "PUBLISHED"}
                or not item.extracted_text
                or item.extraction_checksum != _digest(item.extracted_text)
                for item in versions
            ):
                raise KnowledgeIngestionError("knowledge_release_version_not_reviewed")
            release_version = (
                int(
                    session.scalar(
                        select(func.coalesce(func.max(IndexReleaseRecord.version), 0)).where(
                            IndexReleaseRecord.tenant_id == identity.tenant_id,
                            IndexReleaseRecord.name == name,
                        )
                    )
                    or 0
                )
                + 1
            )
            release_id = f"index-release-{uuid4().hex}"
            content_checksum = _canonical_digest(
                {
                    "chunker_version": self.CHUNKER_VERSION,
                    "documents": sorted(
                        (item.document_version_id, item.content_checksum) for item in versions
                    ),
                }
            )
            release = IndexReleaseRecord(
                tenant_id=identity.tenant_id,
                release_id=release_id,
                name=name,
                version=release_version,
                status="EVALUATION_PENDING",
                is_active=False,
                content_checksum=content_checksum,
                created_by_subject_id=identity.subject_id,
                published_by=None,
                published_at=None,
                created_at=now,
                updated_at=now,
            )
            session.add(release)
            session.flush()
            for version in sorted(versions, key=lambda item: item.document_version_id):
                for ordinal, chunk in enumerate(_chunk_text(version.extracted_text or ""), start=1):
                    checksum = _digest(chunk.text)
                    identity_digest = _digest(
                        f"{release_id}\0{version.document_version_id}\0{ordinal}\0{checksum}"
                    )[:24]
                    chunk_id = f"knowledge-chunk-{identity_digest}"
                    session.add(
                        KnowledgeChunkRecord(
                            tenant_id=identity.tenant_id,
                            chunk_id=chunk_id,
                            document_version_id=version.document_version_id,
                            release_id=release_id,
                            ordinal=ordinal,
                            content=chunk.text,
                            content_checksum=checksum,
                            embedding=deterministic_embedding(chunk.text),
                            token_count=max(1, len(tokenize(chunk.text))),
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    session.flush()
                    anchor_kind, page_number, bounding_box = _citation_location(
                        version.extraction_metadata,
                        chunk.start_offset,
                        chunk.end_offset,
                    )
                    session.add(
                        CitationAnchorRecord(
                            tenant_id=identity.tenant_id,
                            citation_id=f"citation-{identity_digest}",
                            chunk_id=chunk_id,
                            document_version_id=version.document_version_id,
                            anchor_kind=anchor_kind,
                            page_number=page_number,
                            bounding_box=bounding_box,
                            start_offset=chunk.start_offset,
                            end_offset=chunk.end_offset,
                            excerpt=chunk.text,
                            excerpt_checksum=checksum,
                            created_at=now,
                            updated_at=now,
                        )
                    )
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=release_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _release_view(session, release)

    def get_version(
        self,
        identity: IdentityContext,
        document_version_id: str,
        *,
        request_id: str,
    ) -> KnowledgeVersionView:
        self._require_manage(identity, document_version_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            version = session.scalar(
                select(KnowledgeDocumentVersionRecord).where(
                    KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id,
                    KnowledgeDocumentVersionRecord.document_version_id == document_version_id,
                )
            )
            if version is None:
                raise KnowledgeIngestionError("knowledge_version_not_visible")
            return _version_view(session, version)

    def list_versions(
        self,
        identity: IdentityContext,
        *,
        status: str | None,
        limit: int,
        offset: int,
        request_id: str,
    ) -> tuple[tuple[KnowledgeVersionView, ...], int]:
        self._require_manage(identity, "knowledge-document-versions", request_id)
        if status is not None and status not in {
            "DRAFT",
            "REVIEWED",
            "PUBLISHED",
            "DELETION_PENDING",
            "REVOCATION_PENDING",
            "EXPIRY_PENDING",
            "DELETED",
            "REVOKED",
            "EXPIRED",
        }:
            raise KnowledgeIngestionError("knowledge_version_status_invalid")
        if not 1 <= limit <= 100 or offset < 0:
            raise KnowledgeIngestionError("knowledge_page_invalid")
        with self._database.transaction(identity.tenant_context) as session:
            filters = [KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id]
            if status is not None:
                filters.append(KnowledgeDocumentVersionRecord.status == status)
            total = int(
                session.scalar(
                    select(func.count()).select_from(KnowledgeDocumentVersionRecord).where(*filters)
                )
                or 0
            )
            versions = tuple(
                session.scalars(
                    select(KnowledgeDocumentVersionRecord)
                    .where(*filters)
                    .order_by(
                        KnowledgeDocumentVersionRecord.updated_at.desc(),
                        KnowledgeDocumentVersionRecord.document_version_id,
                    )
                    .offset(offset)
                    .limit(limit)
                )
            )
            return tuple(_version_view(session, item) for item in versions), total

    def get_release(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        request_id: str,
    ) -> KnowledgeReleaseView:
        self._require_manage(identity, release_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            release = session.scalar(
                select(IndexReleaseRecord).where(
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.release_id == release_id,
                )
            )
            if release is None:
                raise KnowledgeIngestionError("knowledge_release_not_visible")
            return _release_view(session, release)

    def list_releases(
        self,
        identity: IdentityContext,
        *,
        status: str | None,
        limit: int,
        offset: int,
        request_id: str,
    ) -> tuple[tuple[KnowledgeReleaseView, ...], int]:
        self._require_manage(identity, "knowledge-index-releases", request_id)
        if status is not None and status not in {
            "EVALUATION_PENDING",
            "EVALUATING",
            "CANDIDATE",
            "REJECTED",
            "PUBLISHED",
            "REVOKED",
        }:
            raise KnowledgeIngestionError("knowledge_release_status_invalid")
        if not 1 <= limit <= 100 or offset < 0:
            raise KnowledgeIngestionError("knowledge_page_invalid")
        with self._database.transaction(identity.tenant_context) as session:
            filters = [IndexReleaseRecord.tenant_id == identity.tenant_id]
            if status is not None:
                filters.append(IndexReleaseRecord.status == status)
            total = int(
                session.scalar(select(func.count()).select_from(IndexReleaseRecord).where(*filters))
                or 0
            )
            releases = tuple(
                session.scalars(
                    select(IndexReleaseRecord)
                    .where(*filters)
                    .order_by(
                        IndexReleaseRecord.created_at.desc(),
                        IndexReleaseRecord.release_id,
                    )
                    .offset(offset)
                    .limit(limit)
                )
            )
            return tuple(_release_view(session, item) for item in releases), total

    def _create_version(
        self,
        identity: IdentityContext,
        *,
        document_id: str | None,
        title: str | None,
        source_uri: str | None,
        content: str,
        classification: str | None,
        acl_subject_ids: tuple[str, ...],
        acl_roles: tuple[str, ...],
        device_families: tuple[str, ...],
        device_models: tuple[str, ...],
        valid_from: datetime,
        valid_to: datetime | None,
        idempotency_key: str,
        request_id: str,
        parser_version: str,
        source_checksum_override: str | None,
        extraction_metadata: dict[str, Any] | None,
    ) -> KnowledgeVersionView:
        self._require_manage(identity, document_id or "new-document", request_id)
        _validate_idempotency_key(idempotency_key)
        normalized = _normalize_text(content)
        _validate_scope(
            classification=classification,
            acl_subject_ids=acl_subject_ids,
            acl_roles=acl_roles,
            device_families=device_families,
            device_models=device_models,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        if document_id is None:
            if title is None or not title.strip() or len(title.strip()) > 255:
                raise KnowledgeIngestionError("knowledge_title_invalid")
            _validate_source_uri(source_uri)
        source_checksum = source_checksum_override or _digest(content)
        content_checksum = _digest(normalized)
        request_hash = _canonical_digest(
            {
                "document_id": document_id,
                "title": title,
                "source_uri": source_uri,
                "classification": classification,
                "source_checksum": source_checksum,
                "content_checksum": content_checksum,
                "parser_version": parser_version,
                "extraction_metadata": extraction_metadata,
                "acl_subject_ids": sorted(acl_subject_ids),
                "acl_roles": sorted(acl_roles),
                "device_families": sorted(device_families),
                "device_models": sorted(device_models),
                "valid_from": valid_from.isoformat(),
                "valid_to": valid_to.isoformat() if valid_to else None,
            }
        )
        storage_key = f"knowledge-version:{idempotency_key}"
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            replay = _idempotency_replay(
                session,
                identity,
                storage_key=storage_key,
                request_hash=request_hash,
            )
            if replay is not None:
                existing = session.scalar(
                    select(KnowledgeDocumentVersionRecord).where(
                        KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id,
                        KnowledgeDocumentVersionRecord.document_version_id == replay,
                    )
                )
                if existing is None:
                    raise KnowledgeIngestionError("knowledge_idempotency_state_invalid")
                return _version_view(session, existing)

            if document_id is None:
                document = KnowledgeDocumentRecord(
                    tenant_id=identity.tenant_id,
                    document_id=f"knowledge-document-{uuid4().hex}",
                    title=(title or "").strip(),
                    source_uri=source_uri or "",
                    classification=classification or "internal",
                    source_checksum=source_checksum,
                    current_version=1,
                    created_at=now,
                    updated_at=now,
                )
                session.add(document)
                session.flush()
                version_number = 1
            else:
                existing_document = session.scalar(
                    select(KnowledgeDocumentRecord)
                    .where(
                        KnowledgeDocumentRecord.tenant_id == identity.tenant_id,
                        KnowledgeDocumentRecord.document_id == document_id,
                    )
                    .with_for_update()
                )
                if existing_document is None:
                    raise KnowledgeIngestionError("knowledge_document_not_visible")
                _validate_scope(
                    classification=existing_document.classification,
                    acl_subject_ids=acl_subject_ids,
                    acl_roles=acl_roles,
                    device_families=device_families,
                    device_models=device_models,
                    valid_from=valid_from,
                    valid_to=valid_to,
                )
                document = existing_document
                version_number = document.current_version + 1
                document.current_version = version_number
                document.source_checksum = source_checksum
                document.updated_at = now
            version = KnowledgeDocumentVersionRecord(
                tenant_id=identity.tenant_id,
                document_version_id=f"knowledge-version-{uuid4().hex}",
                document_id=document.document_id,
                version=version_number,
                parser_version=parser_version,
                chunker_version=self.CHUNKER_VERSION,
                acl_subject_ids=list(sorted(set(acl_subject_ids))),
                acl_roles=list(sorted(set(acl_roles))),
                device_families=list(sorted(set(device_families))),
                device_models=list(sorted(set(device_models))),
                valid_from=valid_from,
                valid_to=valid_to,
                source_checksum=source_checksum,
                content_checksum=content_checksum,
                extracted_text=normalized,
                extraction_checksum=content_checksum,
                extraction_metadata=extraction_metadata,
                created_by_subject_id=identity.subject_id,
                reviewed_by_subject_id=None,
                reviewed_at=None,
                state_version=1,
                status="DRAFT",
                created_at=now,
                updated_at=now,
            )
            session.add(version)
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=version.document_version_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _version_view(session, version)

    def _require_manage(
        self,
        identity: IdentityContext,
        resource_id: str,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            Action.PUBLISH_KNOWLEDGE,
            ResourceContext(identity.tenant_id, resource_id),
            request_id=request_id,
        )


@dataclass(frozen=True, slots=True)
class _TextChunk:
    text: str
    start_offset: int
    end_offset: int


def _chunk_text(content: str) -> tuple[_TextChunk, ...]:
    chunks: list[_TextChunk] = []
    cursor = 0
    while cursor < len(content):
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if cursor >= len(content):
            break
        end = min(len(content), cursor + _MAX_CHUNK_CHARACTERS)
        if end < len(content):
            break_at = max(
                content.rfind("\n\n", cursor + _MIN_BREAK_CHARACTERS, end),
                content.rfind("。", cursor + _MIN_BREAK_CHARACTERS, end),
                content.rfind(". ", cursor + _MIN_BREAK_CHARACTERS, end),
            )
            if break_at > cursor:
                end = break_at + (1 if content[break_at] != "." else 2)
        raw = content[cursor:end]
        text = raw.rstrip()
        actual_end = cursor + len(text)
        if text:
            chunks.append(_TextChunk(text, cursor, actual_end))
        cursor = max(end, actual_end)
        if len(chunks) > _MAX_CHUNKS_PER_DOCUMENT:
            raise KnowledgeIngestionError("knowledge_chunk_limit_exceeded")
    if not chunks:
        raise KnowledgeIngestionError("knowledge_content_invalid")
    return tuple(chunks)


def _normalize_text(content: str) -> str:
    if not content or len(content) > _MAX_SOURCE_CHARACTERS or "\x00" in content:
        raise KnowledgeIngestionError("knowledge_content_invalid")
    normalized = _normalize_fragment(content)
    if len(normalized) < 20:
        raise KnowledgeIngestionError("knowledge_content_invalid")
    return normalized


def _normalize_fragment(content: str) -> str:
    normalized = unicodedata.normalize("NFKC", content).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\t", "    ")
    return "".join(
        character
        for character in normalized
        if character == "\n" or unicodedata.category(character) not in {"Cc", "Cf"}
    ).strip()


def _combine_pages(
    pages: tuple[tuple[int, str, str], ...],
    structures: tuple[ExtractedKnowledgeStructure, ...] = (),
) -> tuple[str, dict[str, Any]]:
    if not pages or len(pages) > 500 or len(structures) > 10_000:
        raise KnowledgeIngestionError("knowledge_extraction_pages_invalid")
    parts: list[str] = []
    metadata: list[dict[str, Any]] = []
    cursor = 0
    expected_page = 1
    for page_number, raw_text, method in pages:
        if page_number != expected_page or method not in {
            "docling",
            "native",
            "paddleocr",
            "text",
        }:
            raise KnowledgeIngestionError("knowledge_extraction_pages_invalid")
        page_text = _normalize_fragment(raw_text)
        if page_text:
            if parts:
                parts.append("\n\n")
                cursor += 2
            start_offset = cursor
            parts.append(page_text)
            cursor += len(page_text)
            metadata.append(
                {
                    "page_number": page_number,
                    "start_offset": start_offset,
                    "end_offset": cursor,
                    "method": method,
                }
            )
        expected_page += 1
    content = _normalize_text("".join(parts))
    structure_metadata = _ground_structures(content, metadata, structures)
    return content, {
        "schema": "knowledge-extraction-v2",
        "pages": metadata,
        "structures": structure_metadata,
    }


def _ground_structures(
    content: str,
    pages: list[dict[str, Any]],
    structures: tuple[ExtractedKnowledgeStructure, ...],
) -> list[dict[str, Any]]:
    page_ranges = {
        int(page["page_number"]): (int(page["start_offset"]), int(page["end_offset"]))
        for page in pages
    }
    cursors = {page_number: start for page_number, (start, _) in page_ranges.items()}
    grounded: list[dict[str, Any]] = []
    for structure in structures:
        if structure.kind not in {"section", "table", "figure"}:
            raise KnowledgeIngestionError("knowledge_extraction_structure_invalid")
        page_range = page_ranges.get(structure.page_number)
        text = _normalize_fragment(structure.text)
        bounding_box = _validate_bounding_box(structure.bounding_box)
        if page_range is None or not text or len(text) > _MAX_SOURCE_CHARACTERS:
            raise KnowledgeIngestionError("knowledge_extraction_structure_invalid")
        page_start, page_end = page_range
        located_offset = content.find(text, cursors[structure.page_number], page_end)
        if located_offset < 0:
            located_offset = content.find(text, page_start, page_end)
        start_offset: int | None = located_offset if located_offset >= 0 else None
        end_offset: int | None = None
        if start_offset is not None:
            end_offset = start_offset + len(text)
            cursors[structure.page_number] = end_offset
        grounded.append(
            {
                "kind": structure.kind,
                "page_number": structure.page_number,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "bounding_box": bounding_box,
                "text_checksum": _digest(text),
                "excerpt": text[:240],
                **(
                    {"analysis": _validate_figure_analysis(structure.metadata)}
                    if structure.kind == "figure"
                    else {}
                ),
            }
        )
    return grounded


def _validate_bounding_box(
    bounding_box: dict[str, float] | None,
) -> dict[str, float] | None:
    if bounding_box is None:
        return None
    if set(bounding_box) != {"x", "y", "width", "height"}:
        raise KnowledgeIngestionError("knowledge_extraction_structure_invalid")
    normalized = {key: float(value) for key, value in bounding_box.items()}
    if (
        any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized.values())
        or normalized["width"] <= 0.0
        or normalized["height"] <= 0.0
        or normalized["x"] + normalized["width"] > 1.001
        or normalized["y"] + normalized["height"] > 1.001
    ):
        raise KnowledgeIngestionError("knowledge_extraction_structure_invalid")
    return normalized


def _page_number(metadata: dict[str, Any] | None, offset: int) -> int | None:
    if metadata is None:
        return None
    pages = metadata.get("pages")
    if not isinstance(pages, list):
        return None
    for page in pages:
        if (
            isinstance(page, dict)
            and isinstance(page.get("page_number"), int)
            and isinstance(page.get("start_offset"), int)
            and isinstance(page.get("end_offset"), int)
            and page["start_offset"] <= offset < page["end_offset"]
        ):
            return int(page["page_number"])
    return None


def _citation_location(
    metadata: dict[str, Any] | None,
    start_offset: int,
    end_offset: int,
) -> tuple[str, int | None, dict[str, float] | None]:
    page_number = _page_number(metadata, start_offset)
    if metadata is None:
        return "text_offset", page_number, None
    structures = metadata.get("structures")
    if not isinstance(structures, list):
        return "text_offset", page_number, None
    for preferred_kind in ("table", "figure", "section"):
        for structure in structures:
            if not isinstance(structure, dict) or structure.get("kind") != preferred_kind:
                continue
            structure_start = structure.get("start_offset")
            structure_end = structure.get("end_offset")
            if (
                not isinstance(structure_start, int)
                or not isinstance(structure_end, int)
                or structure_end <= start_offset
                or structure_start >= end_offset
            ):
                continue
            structure_page = structure.get("page_number")
            structure_box = structure.get("bounding_box")
            return (
                preferred_kind,
                structure_page if isinstance(structure_page, int) else page_number,
                structure_box if isinstance(structure_box, dict) else None,
            )
    return "text_offset", page_number, None


def _validate_figure_analysis(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        raise KnowledgeIngestionError("knowledge_extraction_structure_invalid")
    allowed = {
        "figure_id",
        "analysis_status",
        "source_image_sha256",
        "processor_version",
        "model_release_ids",
        "finding_count",
        "failure_reason",
    }
    if not set(metadata).issubset(allowed):
        raise KnowledgeIngestionError("knowledge_extraction_structure_invalid")
    figure_id = metadata.get("figure_id")
    status = metadata.get("analysis_status")
    digest = metadata.get("source_image_sha256")
    finding_count = metadata.get("finding_count")
    if (
        not isinstance(figure_id, str)
        or not 1 <= len(figure_id) <= 128
        or status not in {"ANALYZED", "NO_FINDINGS", "UNSUPPORTED", "UNAVAILABLE"}
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or isinstance(finding_count, bool)
        or not isinstance(finding_count, int)
        or not 0 <= finding_count <= 100
    ):
        raise KnowledgeIngestionError("knowledge_extraction_structure_invalid")
    processor_version = metadata.get("processor_version")
    if processor_version is not None and (
        not isinstance(processor_version, str) or not 1 <= len(processor_version) <= 128
    ):
        raise KnowledgeIngestionError("knowledge_extraction_structure_invalid")
    release_ids = metadata.get("model_release_ids", [])
    if (
        not isinstance(release_ids, list)
        or len(release_ids) > 16
        or any(not isinstance(value, str) or not 1 <= len(value) <= 128 for value in release_ids)
    ):
        raise KnowledgeIngestionError("knowledge_extraction_structure_invalid")
    failure_reason = metadata.get("failure_reason")
    if failure_reason is not None and (
        not isinstance(failure_reason, str) or not 1 <= len(failure_reason) <= 128
    ):
        raise KnowledgeIngestionError("knowledge_extraction_structure_invalid")
    return {
        "figure_id": figure_id,
        "analysis_status": status,
        "source_image_sha256": digest,
        "processor_version": processor_version,
        "model_release_ids": release_ids,
        "finding_count": finding_count,
        "failure_reason": failure_reason,
    }


def _validate_scope(
    *,
    classification: str | None,
    acl_subject_ids: tuple[str, ...],
    acl_roles: tuple[str, ...],
    device_families: tuple[str, ...],
    device_models: tuple[str, ...],
    valid_from: datetime,
    valid_to: datetime | None,
) -> None:
    if classification is not None and classification not in {"internal", "restricted"}:
        raise KnowledgeIngestionError("knowledge_classification_invalid")
    if classification == "restricted" and not acl_roles:
        raise KnowledgeIngestionError("restricted_knowledge_role_acl_required")
    try:
        for subject_id in acl_subject_ids:
            validate_boundary_identifier(subject_id, field="acl_subject_id")
        for role in acl_roles:
            Role(role)
    except ValueError as exc:
        raise KnowledgeIngestionError("knowledge_acl_invalid") from exc
    if (
        len(acl_subject_ids) > 100
        or len(acl_roles) > 100
        or len(device_families) > 100
        or len(device_models) > 100
        or len(set(acl_subject_ids)) != len(acl_subject_ids)
        or len(set(acl_roles)) != len(acl_roles)
        or len(set(device_families)) != len(device_families)
        or len(set(device_models)) != len(device_models)
        or any(not value or len(value) > 128 for value in (*device_families, *device_models))
    ):
        raise KnowledgeIngestionError("knowledge_scope_invalid")
    if valid_from.tzinfo is None or valid_from.utcoffset() is None:
        raise KnowledgeIngestionError("knowledge_validity_invalid")
    if valid_to is not None and (
        valid_to.tzinfo is None or valid_to.utcoffset() is None or valid_to <= valid_from
    ):
        raise KnowledgeIngestionError("knowledge_validity_invalid")


def _validate_idempotency_key(value: str) -> None:
    if not value or len(value) > 200 or any(character.isspace() for character in value):
        raise KnowledgeIngestionError("knowledge_idempotency_key_invalid")


def _validate_source_uri(source_uri: str | None) -> None:
    parsed = urlparse(source_uri or "")
    if (
        parsed.scheme not in {"https", "s3", "minio"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or len(source_uri or "") > 1_024
    ):
        raise KnowledgeIngestionError("knowledge_source_uri_invalid")


def _idempotency_replay(
    session: Session,
    identity: IdentityContext,
    *,
    storage_key: str,
    request_hash: str,
) -> str | None:
    record = session.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.tenant_id == identity.tenant_id,
            IdempotencyRecord.subject_id == identity.subject_id,
            IdempotencyRecord.key == storage_key,
        )
    )
    if record is None:
        return None
    if record.request_hash != request_hash:
        raise KnowledgeIngestionError("knowledge_idempotency_conflict")
    return str(record.result_ref)


def _version_view(
    session: Session, version: KnowledgeDocumentVersionRecord
) -> KnowledgeVersionView:
    document = session.scalar(
        select(KnowledgeDocumentRecord).where(
            KnowledgeDocumentRecord.tenant_id == version.tenant_id,
            KnowledgeDocumentRecord.document_id == version.document_id,
        )
    )
    if document is None:
        raise KnowledgeIngestionError("knowledge_document_state_invalid")
    return KnowledgeVersionView(
        document_id=document.document_id,
        document_version_id=version.document_version_id,
        version=version.version,
        title=document.title,
        source_uri=document.source_uri,
        classification=document.classification,
        source_checksum=version.source_checksum,
        content_checksum=version.content_checksum,
        parser_version=version.parser_version,
        extraction_metadata=version.extraction_metadata,
        status=version.status,
        state_version=version.state_version,
        acl_subject_ids=tuple(version.acl_subject_ids),
        acl_roles=tuple(version.acl_roles),
        device_families=tuple(version.device_families),
        device_models=tuple(version.device_models),
        valid_from=_utc(version.valid_from),
        valid_to=_utc(version.valid_to) if version.valid_to else None,
        created_by_subject_id=version.created_by_subject_id,
        reviewed_by_subject_id=version.reviewed_by_subject_id,
        reviewed_at=_utc(version.reviewed_at) if version.reviewed_at else None,
        created_at=_utc(version.created_at),
        updated_at=_utc(version.updated_at),
    )


def _release_view(session: Session, release: IndexReleaseRecord) -> KnowledgeReleaseView:
    chunk_count = int(
        session.scalar(
            select(func.count())
            .select_from(KnowledgeChunkRecord)
            .where(
                KnowledgeChunkRecord.tenant_id == release.tenant_id,
                KnowledgeChunkRecord.release_id == release.release_id,
            )
        )
        or 0
    )
    evaluation = session.scalar(
        select(KnowledgeIndexEvaluationRecord)
        .where(
            KnowledgeIndexEvaluationRecord.tenant_id == release.tenant_id,
            KnowledgeIndexEvaluationRecord.release_id == release.release_id,
        )
        .order_by(
            KnowledgeIndexEvaluationRecord.created_at.desc(),
            KnowledgeIndexEvaluationRecord.evaluation_id.desc(),
        )
        .limit(1)
    )
    active_version = session.scalar(
        select(IndexReleaseRecord.version).where(
            IndexReleaseRecord.tenant_id == release.tenant_id,
            IndexReleaseRecord.name == release.name,
            IndexReleaseRecord.status == "PUBLISHED",
            IndexReleaseRecord.is_active.is_(True),
        )
    )
    return KnowledgeReleaseView(
        release_id=release.release_id,
        name=release.name,
        version=release.version,
        status=release.status,
        is_active=release.is_active,
        activation_version=release.activation_version,
        rollback_eligible=(
            release.status == "PUBLISHED"
            and not release.is_active
            and active_version is not None
            and release.version < active_version
            and evaluation is not None
            and evaluation.status == "PASSED"
        ),
        content_checksum=release.content_checksum,
        chunk_count=chunk_count,
        created_by_subject_id=release.created_by_subject_id,
        evaluation_id=evaluation.evaluation_id if evaluation else None,
        evaluation_status=evaluation.status if evaluation else None,
        evaluation_metrics=dict(evaluation.metrics_json) if evaluation else {},
        evaluation_gate_results=dict(evaluation.gate_results_json) if evaluation else {},
        evaluation_failure_codes=tuple(evaluation.failure_codes) if evaluation else (),
        evaluated_at=(
            _utc(evaluation.completed_at) if evaluation and evaluation.completed_at else None
        ),
        published_by=release.published_by,
        published_at=_utc(release.published_at) if release.published_at else None,
        created_at=_utc(release.created_at),
        updated_at=_utc(release.updated_at),
    )


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()
