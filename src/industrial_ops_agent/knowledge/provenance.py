"""Authorized, integrity-aware provenance for one diagnosis citation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

from sqlalchemy import select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.knowledge.models import CitationView
from industrial_ops_agent.knowledge.service import CitationService, KnowledgeNotVisible
from industrial_ops_agent.media.service import TenantObjectStore
from industrial_ops_agent.object_store.keys import (
    ObjectZone,
    TenantObjectKey,
    tenant_object_key_from_value,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CitationAnchorRecord,
    IndexReleaseRecord,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    KnowledgeDocumentVersionRecord,
    KnowledgeIngestionAttemptRecord,
    KnowledgeIngestionJobRecord,
)


class CitationSourceNotAvailable(RuntimeError):
    """The citation is visible, but it has no readable managed source object."""


@dataclass(frozen=True, slots=True)
class CitationIngestionAttemptView:
    attempt_id: str
    attempt_number: int
    workflow_id: str
    status: str
    failure_reason: str | None
    parser_version: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CitationSourceProvenanceView:
    origin: str
    source_uri: str
    source_filename: str | None
    declared_mime: str | None
    detected_mime: str | None
    size_bytes: int | None
    source_checksum: str
    ingestion_id: str | None
    ingestion_workflow_id: str | None
    ingestion_status: str | None
    ingestion_attempt_count: int
    ingestion_created_by_subject_id: str | None
    attempts: tuple[CitationIngestionAttemptView, ...]
    download_path: str | None


@dataclass(frozen=True, slots=True)
class CitationDocumentProvenanceView:
    document_id: str
    document_version_id: str
    document_version: int
    title: str
    classification: str
    version_status: str
    parser_version: str
    chunker_version: str
    content_checksum: str
    extraction_checksum: str | None
    extraction_metadata: dict[str, object] | None
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
class CitationChunkProvenanceView:
    chunk_id: str
    ordinal: int
    token_count: int
    content_checksum: str
    embedding_dimensions: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class CitationAnchorProvenanceView:
    citation_id: str
    anchor_kind: str
    page_number: int | None
    bounding_box: dict[str, float] | None
    start_offset: int
    end_offset: int
    excerpt_checksum: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class CitationIndexReleaseProvenanceView:
    release_id: str
    name: str
    version: int
    status: str
    is_active: bool
    content_checksum: str
    created_by_subject_id: str | None
    published_by_subject_id: str | None
    published_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CitationIntegrityView:
    status: str
    source_checksum_consistent: bool
    extraction_checksum_valid: bool
    chunk_checksum_valid: bool
    citation_checksum_valid: bool
    document_links_valid: bool
    release_published: bool
    chain_digest: str


@dataclass(frozen=True, slots=True)
class CitationProvenanceView:
    source: CitationSourceProvenanceView
    document: CitationDocumentProvenanceView
    chunk: CitationChunkProvenanceView
    anchor: CitationAnchorProvenanceView
    index_release: CitationIndexReleaseProvenanceView
    integrity: CitationIntegrityView


@dataclass(frozen=True, slots=True)
class AuthorizedCitationView:
    citation: CitationView
    provenance: CitationProvenanceView


@dataclass(frozen=True, slots=True)
class CitationSourceFile:
    content: bytes
    filename: str
    media_type: str
    source_checksum: str


class CitationProvenanceService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def get(
        self,
        identity: IdentityContext,
        citation_id: str,
        *,
        request_id: str,
    ) -> AuthorizedCitationView:
        citation = CitationService(self._database, self._authorizer).get(
            identity,
            citation_id,
            request_id=request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            row = session.execute(
                select(
                    CitationAnchorRecord,
                    KnowledgeChunkRecord,
                    KnowledgeDocumentVersionRecord,
                    KnowledgeDocumentRecord,
                    IndexReleaseRecord,
                )
                .join(
                    KnowledgeChunkRecord,
                    KnowledgeChunkRecord.chunk_id == CitationAnchorRecord.chunk_id,
                )
                .join(
                    KnowledgeDocumentVersionRecord,
                    KnowledgeDocumentVersionRecord.document_version_id
                    == CitationAnchorRecord.document_version_id,
                )
                .join(
                    KnowledgeDocumentRecord,
                    KnowledgeDocumentRecord.document_id
                    == KnowledgeDocumentVersionRecord.document_id,
                )
                .join(
                    IndexReleaseRecord,
                    IndexReleaseRecord.release_id == KnowledgeChunkRecord.release_id,
                )
                .where(
                    CitationAnchorRecord.tenant_id == identity.tenant_id,
                    CitationAnchorRecord.citation_id == citation_id,
                    KnowledgeChunkRecord.tenant_id == identity.tenant_id,
                    KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id,
                    KnowledgeDocumentRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                )
            ).one_or_none()
            if row is None:
                raise KnowledgeNotVisible
            anchor, chunk, version, document, release = row
            ingestion = session.scalar(
                select(KnowledgeIngestionJobRecord)
                .where(
                    KnowledgeIngestionJobRecord.tenant_id == identity.tenant_id,
                    KnowledgeIngestionJobRecord.document_version_id
                    == version.document_version_id,
                )
                .order_by(
                    KnowledgeIngestionJobRecord.created_at.desc(),
                    KnowledgeIngestionJobRecord.ingestion_id.desc(),
                )
                .limit(1)
            )
            attempts = (
                list(
                    session.scalars(
                        select(KnowledgeIngestionAttemptRecord)
                        .where(
                            KnowledgeIngestionAttemptRecord.tenant_id
                            == identity.tenant_id,
                            KnowledgeIngestionAttemptRecord.ingestion_id
                            == ingestion.ingestion_id,
                        )
                        .order_by(KnowledgeIngestionAttemptRecord.attempt_number)
                    )
                )
                if ingestion is not None
                else []
            )
            provenance = _provenance(
                citation,
                anchor,
                chunk,
                version,
                document,
                release,
                ingestion,
                attempts,
            )
        return AuthorizedCitationView(citation=citation, provenance=provenance)

    async def read_managed_source(
        self,
        identity: IdentityContext,
        citation_id: str,
        object_store: TenantObjectStore,
        *,
        request_id: str,
    ) -> CitationSourceFile:
        def resolve_source() -> tuple[TenantObjectKey, str, str, str]:
            trace = self.get(identity, citation_id, request_id=request_id)
            source = trace.provenance.source
            if source.origin != "MANAGED_FILE" or source.ingestion_id is None:
                self._record_source_guard(identity, citation_id, request_id, "deny", "not_managed")
                raise CitationSourceNotAvailable("citation_source_not_managed")
            with self._database.transaction(identity.tenant_context) as session:
                ingestion = session.scalar(
                    select(KnowledgeIngestionJobRecord).where(
                        KnowledgeIngestionJobRecord.tenant_id == identity.tenant_id,
                        KnowledgeIngestionJobRecord.ingestion_id == source.ingestion_id,
                        KnowledgeIngestionJobRecord.document_version_id
                        == trace.citation.document_version_id,
                    )
                )
                if ingestion is None or ingestion.clean_key is None:
                    self._record_source_guard(
                        identity, citation_id, request_id, "deny", "clean_source_missing"
                    )
                    raise CitationSourceNotAvailable("citation_source_not_available")
                clean_key_value = ingestion.clean_key
                filename = ingestion.source_filename
                media_type = ingestion.detected_mime or ingestion.declared_mime
            try:
                clean_key = tenant_object_key_from_value(
                    identity.tenant_context,
                    clean_key_value,
                )
            except Exception as exc:
                self._record_source_guard(
                    identity, citation_id, request_id, "deny", "object_key_invalid"
                )
                raise CitationSourceNotAvailable("citation_source_not_available") from exc
            if clean_key.zone is not ObjectZone.CLEAN:
                self._record_source_guard(
                    identity, citation_id, request_id, "deny", "object_not_clean"
                )
                raise CitationSourceNotAvailable("citation_source_not_available")
            return clean_key, filename, media_type, source.source_checksum

        clean_key, filename, media_type, source_checksum = await asyncio.to_thread(resolve_source)
        content = await object_store.get(identity.tenant_context, clean_key)
        if content is None or _digest_bytes(content) != source_checksum:
            await asyncio.to_thread(
                self._record_source_guard,
                identity, citation_id, request_id, "deny", "source_integrity_failed"
            )
            raise CitationSourceNotAvailable("citation_source_integrity_failed")
        await asyncio.to_thread(
            self._record_source_guard, identity, citation_id, request_id, "allow", "source_verified"
        )
        return CitationSourceFile(
            content=content,
            filename=filename,
            media_type=media_type,
            source_checksum=source_checksum,
        )

    def _record_source_guard(
        self,
        identity: IdentityContext,
        citation_id: str,
        request_id: str,
        decision: str,
        reason_code: str,
    ) -> None:
        self._authorizer.record_guard_decision(
            identity,
            action="citation.source.download",
            decision=decision,
            reason_code=reason_code,
            request_id=request_id,
            resource_id=citation_id,
        )


def _provenance(
    citation: CitationView,
    anchor: CitationAnchorRecord,
    chunk: KnowledgeChunkRecord,
    version: KnowledgeDocumentVersionRecord,
    document: KnowledgeDocumentRecord,
    release: IndexReleaseRecord,
    ingestion: KnowledgeIngestionJobRecord | None,
    attempts: list[KnowledgeIngestionAttemptRecord],
) -> CitationProvenanceView:
    source_consistent = bool(
        _is_sha256(version.source_checksum)
        and (ingestion is None or ingestion.source_checksum == version.source_checksum)
    )
    extraction_valid = bool(
        version.extracted_text is not None
        and version.extraction_checksum is not None
        and _digest_text(version.extracted_text) == version.extraction_checksum
        and version.extraction_checksum == version.content_checksum
    )
    chunk_valid = _digest_text(chunk.content) == chunk.content_checksum
    citation_valid = bool(
        _digest_text(anchor.excerpt) == anchor.excerpt_checksum
        and anchor.excerpt_checksum == chunk.content_checksum
        and anchor.excerpt == chunk.content
    )
    links_valid = bool(
        anchor.chunk_id == chunk.chunk_id
        and anchor.document_version_id == chunk.document_version_id
        and version.document_version_id == chunk.document_version_id
        and version.document_id == document.document_id
    )
    release_published = release.status == "PUBLISHED" and version.status == "PUBLISHED"
    checks = (
        source_consistent,
        extraction_valid,
        chunk_valid,
        citation_valid,
        links_valid,
        release_published,
    )
    chain_digest = _chain_digest(
        document.document_id,
        version.document_version_id,
        version.source_checksum,
        version.content_checksum,
        chunk.chunk_id,
        chunk.content_checksum,
        anchor.citation_id,
        anchor.excerpt_checksum,
        release.release_id,
        release.content_checksum,
    )
    attempt_views = tuple(
        CitationIngestionAttemptView(
            attempt_id=item.attempt_id,
            attempt_number=item.attempt_number,
            workflow_id=item.workflow_id,
            status=item.status,
            failure_reason=item.failure_reason,
            parser_version=item.parser_version,
            started_at=item.started_at,
            completed_at=item.completed_at,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
        for item in attempts
    )
    source = CitationSourceProvenanceView(
        origin="MANAGED_FILE" if ingestion is not None else "DIRECT_ENTRY",
        source_uri=document.source_uri,
        source_filename=ingestion.source_filename if ingestion is not None else None,
        declared_mime=ingestion.declared_mime if ingestion is not None else None,
        detected_mime=ingestion.detected_mime if ingestion is not None else None,
        size_bytes=ingestion.size_bytes if ingestion is not None else None,
        source_checksum=version.source_checksum,
        ingestion_id=ingestion.ingestion_id if ingestion is not None else None,
        ingestion_workflow_id=ingestion.workflow_id if ingestion is not None else None,
        ingestion_status=ingestion.status if ingestion is not None else None,
        ingestion_attempt_count=ingestion.attempt_count if ingestion is not None else 0,
        ingestion_created_by_subject_id=(
            ingestion.created_by_subject_id if ingestion is not None else None
        ),
        attempts=attempt_views,
        download_path=(
            f"/api/v1/citations/{citation.citation_id}/source"
            if ingestion is not None and ingestion.clean_key is not None
            else None
        ),
    )
    return CitationProvenanceView(
        source=source,
        document=CitationDocumentProvenanceView(
            document_id=document.document_id,
            document_version_id=version.document_version_id,
            document_version=version.version,
            title=document.title,
            classification=document.classification,
            version_status=version.status,
            parser_version=version.parser_version,
            chunker_version=version.chunker_version,
            content_checksum=version.content_checksum,
            extraction_checksum=version.extraction_checksum,
            extraction_metadata=version.extraction_metadata,
            acl_subject_ids=tuple(version.acl_subject_ids),
            acl_roles=tuple(version.acl_roles),
            device_families=tuple(version.device_families),
            device_models=tuple(version.device_models),
            valid_from=version.valid_from,
            valid_to=version.valid_to,
            created_by_subject_id=version.created_by_subject_id,
            reviewed_by_subject_id=version.reviewed_by_subject_id,
            reviewed_at=version.reviewed_at,
            created_at=version.created_at,
            updated_at=version.updated_at,
        ),
        chunk=CitationChunkProvenanceView(
            chunk_id=chunk.chunk_id,
            ordinal=chunk.ordinal,
            token_count=chunk.token_count,
            content_checksum=chunk.content_checksum,
            embedding_dimensions=len(chunk.embedding),
            created_at=chunk.created_at,
        ),
        anchor=CitationAnchorProvenanceView(
            citation_id=anchor.citation_id,
            anchor_kind=anchor.anchor_kind,
            page_number=anchor.page_number,
            bounding_box=anchor.bounding_box,
            start_offset=anchor.start_offset,
            end_offset=anchor.end_offset,
            excerpt_checksum=anchor.excerpt_checksum,
            created_at=anchor.created_at,
        ),
        index_release=CitationIndexReleaseProvenanceView(
            release_id=release.release_id,
            name=release.name,
            version=release.version,
            status=release.status,
            is_active=release.is_active,
            content_checksum=release.content_checksum,
            created_by_subject_id=release.created_by_subject_id,
            published_by_subject_id=release.published_by,
            published_at=release.published_at,
            created_at=release.created_at,
            updated_at=release.updated_at,
        ),
        integrity=CitationIntegrityView(
            status="VERIFIED" if all(checks) else "BROKEN",
            source_checksum_consistent=source_consistent,
            extraction_checksum_valid=extraction_valid,
            chunk_checksum_valid=chunk_valid,
            citation_checksum_valid=citation_valid,
            document_links_valid=links_valid,
            release_published=release_published,
            chain_digest=chain_digest,
        ),
    )


def _chain_digest(*values: str) -> str:
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _digest_text(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "AuthorizedCitationView",
    "CitationProvenanceService",
    "CitationProvenanceView",
    "CitationSourceFile",
    "CitationSourceNotAvailable",
]
