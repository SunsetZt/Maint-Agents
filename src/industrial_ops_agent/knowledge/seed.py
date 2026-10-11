"""Synthetic, idempotent M2 manuals for local development and acceptance tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256

from sqlalchemy.orm import Session

from industrial_ops_agent.knowledge.embedding import deterministic_embedding
from industrial_ops_agent.persistence.models import (
    CitationAnchorRecord,
    IndexReleaseRecord,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    KnowledgeDocumentVersionRecord,
)

ACTIVE_RELEASE_ID = "index-release-m2-active"
CANDIDATE_RELEASE_ID = "index-release-m2-candidate"


def install_synthetic_knowledge(session: Session, *, tenant_id: str, now: datetime) -> None:
    """Install authorized and deliberately inapplicable records without customer data."""

    if session.get(IndexReleaseRecord, ACTIVE_RELEASE_ID) is not None:
        return
    active_content = (
        "PUMP-X100 alarm E-42 indicates inlet restriction; inspect and clean the inlet filter."
    )
    session.add_all(
        [
            IndexReleaseRecord(
                tenant_id=tenant_id,
                release_id=ACTIVE_RELEASE_ID,
                name="m2-synthetic-manuals",
                version=1,
                status="PUBLISHED",
                is_active=True,
                content_checksum=_digest("active-release-v1"),
                published_by="system-seed",
                published_at=now,
            ),
            IndexReleaseRecord(
                tenant_id=tenant_id,
                release_id=CANDIDATE_RELEASE_ID,
                name="m2-synthetic-manuals",
                version=2,
                status="CANDIDATE",
                is_active=False,
                content_checksum=_digest("candidate-release-v2"),
                published_by=None,
                published_at=None,
            ),
        ]
    )
    session.flush()
    _add_document(
        session,
        tenant_id=tenant_id,
        key="eligible",
        release_id=ACTIVE_RELEASE_ID,
        title="PUMP-X100 Service Manual",
        content=active_content,
        classification="internal",
        acl_roles=["field_engineer", "after_sales_engineer", "domain_expert"],
        device_models=["PUMP-X100"],
        valid_from=now - timedelta(days=365),
        valid_to=None,
        page_number=42,
    )
    _add_document(
        session,
        tenant_id=tenant_id,
        key="wrong-model",
        release_id=ACTIVE_RELEASE_ID,
        title="PUMP-Z900 Service Manual",
        content="Alarm E-42 E-42 E-42 requires replacing the Z900 controller immediately.",
        classification="internal",
        acl_roles=["field_engineer"],
        device_models=["PUMP-Z900"],
        valid_from=now - timedelta(days=365),
        valid_to=None,
        page_number=7,
    )
    _add_document(
        session,
        tenant_id=tenant_id,
        key="restricted",
        release_id=ACTIVE_RELEASE_ID,
        title="Restricted expert bulletin",
        content="Alarm E-42 expert-only experimental teardown instructions.",
        classification="restricted",
        acl_roles=["domain_expert"],
        device_models=["PUMP-X100"],
        valid_from=now - timedelta(days=30),
        valid_to=None,
        page_number=3,
    )
    _add_document(
        session,
        tenant_id=tenant_id,
        key="expired",
        release_id=ACTIVE_RELEASE_ID,
        title="Expired PUMP-X100 bulletin",
        content="Alarm E-42 obsolete procedure: bypass the inlet safety switch.",
        classification="internal",
        acl_roles=["field_engineer"],
        device_models=["PUMP-X100"],
        valid_from=now - timedelta(days=365),
        valid_to=now - timedelta(days=1),
        page_number=2,
    )
    _add_document(
        session,
        tenant_id=tenant_id,
        key="candidate",
        release_id=CANDIDATE_RELEASE_ID,
        title="Unpublished candidate manual",
        content="Alarm E-42 candidate-only content must not enter active diagnosis.",
        classification="internal",
        acl_roles=["field_engineer"],
        device_models=["PUMP-X100"],
        valid_from=now - timedelta(days=1),
        valid_to=None,
        page_number=1,
    )


def _add_document(
    session: Session,
    *,
    tenant_id: str,
    key: str,
    release_id: str,
    title: str,
    content: str,
    classification: str,
    acl_roles: list[str],
    device_models: list[str],
    valid_from: datetime,
    valid_to: datetime | None,
    page_number: int,
) -> None:
    document_id = f"knowledge-document-{key}"
    version_id = f"knowledge-version-{key}-v1"
    chunk_id = f"knowledge-chunk-{key}-1"
    citation_id = f"citation-{key}-1"
    checksum = _digest(content)
    session.add(
        KnowledgeDocumentRecord(
            tenant_id=tenant_id,
            document_id=document_id,
            title=title,
            source_uri=f"synthetic://m2/{key}.md",
            classification=classification,
            source_checksum=checksum,
            current_version=1,
        )
    )
    session.flush()
    session.add(
        KnowledgeDocumentVersionRecord(
            tenant_id=tenant_id,
            document_version_id=version_id,
            document_id=document_id,
            version=1,
            parser_version="markdown-parser-v1",
            chunker_version="semantic-chunker-v1",
            acl_subject_ids=[],
            acl_roles=acl_roles,
            device_families=["pump"],
            device_models=device_models,
            valid_from=valid_from,
            valid_to=valid_to,
            source_checksum=checksum,
            content_checksum=checksum,
            status="PUBLISHED",
        )
    )
    session.flush()
    session.add(
        KnowledgeChunkRecord(
            tenant_id=tenant_id,
            chunk_id=chunk_id,
            document_version_id=version_id,
            release_id=release_id,
            ordinal=1,
            content=content,
            content_checksum=checksum,
            embedding=deterministic_embedding(content),
            token_count=len(content.split()),
        )
    )
    # These seed models intentionally omit ORM relationships; explicit flushes keep
    # the database foreign-key order deterministic across SQLAlchemy versions.
    session.flush()
    session.add(
        CitationAnchorRecord(
            tenant_id=tenant_id,
            citation_id=citation_id,
            chunk_id=chunk_id,
            document_version_id=version_id,
            anchor_kind="page",
            page_number=page_number,
            bounding_box=None,
            start_offset=0,
            end_offset=len(content),
            excerpt=content,
            excerpt_checksum=checksum,
        )
    )


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()
