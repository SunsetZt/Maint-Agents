"""Tenant-safe PostgreSQL FTS + pgvector retrieval with a SQLite test fallback."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import and_, exists, false, func, or_, select, text
from sqlalchemy.engine import RowMapping
from sqlalchemy.orm import Session

from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.knowledge.embedding import (
    cosine_similarity,
    deterministic_embedding,
    tokenize,
)
from industrial_ops_agent.knowledge.models import (
    RetrievalGuardTrace,
    RetrievalQuery,
    RetrievalResult,
    RetrievedEvidence,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CitationAnchorRecord,
    IndexReleaseRecord,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    KnowledgeDocumentVersionRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext


class KnowledgeReleaseUnavailable(Exception):
    """A run referenced an unpublished or missing knowledge release."""


class RetrievalAuthorizationInvariantViolation(RuntimeError):
    """A candidate escaped the mandatory SQL authorization boundary."""

    def __init__(self, reason: str, *, candidate_count: int = 1) -> None:
        self.reason = reason
        self.candidate_count = max(1, candidate_count)
        super().__init__(reason)


_POLICY_DIMENSIONS = (
    "tenant",
    "acl_subject",
    "acl_role",
    "classification",
    "device_family",
    "device_model",
    "validity_window",
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    chunk_id: str
    citation_id: str
    document_id: str
    document_version_id: str
    document_version: int
    title: str
    content: str
    content_checksum: str
    source_checksum: str
    page_number: int | None
    embedding: list[float]
    acl_subject_ids: frozenset[str]
    acl_roles: frozenset[str]
    device_families: frozenset[str]
    device_models: frozenset[str]
    valid_from: datetime
    valid_to: datetime | None
    classification: str


class HybridRetriever:
    """Apply authorization and applicability filters before both recall paths."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def search(self, query: RetrievalQuery) -> list[RetrievedEvidence]:
        return list(self.search_with_trace(query).evidence)

    def search_with_trace(self, query: RetrievalQuery) -> RetrievalResult:
        return self._search(
            query,
            release_status="PUBLISHED",
            version_statuses=("PUBLISHED",),
        )

    def evaluate_candidate(self, query: RetrievalQuery) -> list[RetrievedEvidence]:
        """Run the production retrieval path against one non-public evaluation release."""

        return list(self.evaluate_candidate_with_trace(query).evidence)

    def evaluate_candidate_with_trace(self, query: RetrievalQuery) -> RetrievalResult:
        """Evaluate an unpublished release through the identical guarded recall path."""

        return self._search(
            query,
            release_status="EVALUATING",
            version_statuses=("REVIEWED", "PUBLISHED"),
        )

    def _search(
        self,
        query: RetrievalQuery,
        *,
        release_status: str,
        version_statuses: tuple[str, ...],
    ) -> RetrievalResult:
        context = TenantContext(tenant_id=query.tenant_id, subject_id=query.subject_id)
        with self._database.transaction(context) as session:
            release = session.scalar(
                select(IndexReleaseRecord).where(
                    IndexReleaseRecord.tenant_id == query.tenant_id,
                    IndexReleaseRecord.release_id == query.release_id,
                    IndexReleaseRecord.status == release_status,
                )
            )
            if release is None:
                raise KnowledgeReleaseUnavailable(query.release_id)
            if session.bind is not None and session.bind.dialect.name == "postgresql":
                return self._postgres_search(session, query, version_statuses)
            candidates = self._sqlite_prefilter(session, query, version_statuses)
            evidence, sparse_count, vector_count, fused_count = _rank(candidates, query)
            return RetrievalResult(
                evidence=tuple(evidence),
                guard=_guard(
                    query,
                    backend="sqlite",
                    sparse_count=sparse_count,
                    vector_count=vector_count,
                    fused_count=fused_count,
                    returned_count=len(evidence),
                ),
            )

    def _sqlite_prefilter(
        self,
        session: Session,
        query: RetrievalQuery,
        version_statuses: tuple[str, ...],
    ) -> list[_Candidate]:
        subject_acl = (
            func.json_each(KnowledgeDocumentVersionRecord.acl_subject_ids)
            .table_valued("key", "value")
            .alias("subject_acl")
        )
        role_acl = (
            func.json_each(KnowledgeDocumentVersionRecord.acl_roles)
            .table_valued("key", "value")
            .alias("role_acl")
        )
        device_models = (
            func.json_each(KnowledgeDocumentVersionRecord.device_models)
            .table_valued("key", "value")
            .alias("device_models")
        )
        device_families = (
            func.json_each(KnowledgeDocumentVersionRecord.device_families)
            .table_valued("key", "value")
            .alias("device_families")
        )
        subject_allowed = exists(
            select(1).select_from(subject_acl).where(subject_acl.c.value == query.subject_id)
        )
        role_allowed = (
            exists(select(1).select_from(role_acl).where(role_acl.c.value.in_(sorted(query.roles))))
            if query.roles
            else false()
        )
        model_allowed = exists(
            select(1).select_from(device_models).where(device_models.c.value == query.device_model)
        )
        family_allowed = (
            exists(
                select(1)
                .select_from(device_families)
                .where(device_families.c.value == query.device_family)
            )
            if query.device_family is not None
            else false()
        )
        statement = (
            select(
                KnowledgeChunkRecord,
                KnowledgeDocumentVersionRecord,
                KnowledgeDocumentRecord,
                CitationAnchorRecord,
            )
            .join(
                KnowledgeDocumentVersionRecord,
                KnowledgeDocumentVersionRecord.document_version_id
                == KnowledgeChunkRecord.document_version_id,
            )
            .join(
                KnowledgeDocumentRecord,
                KnowledgeDocumentRecord.document_id == KnowledgeDocumentVersionRecord.document_id,
            )
            .join(
                CitationAnchorRecord,
                and_(
                    CitationAnchorRecord.chunk_id == KnowledgeChunkRecord.chunk_id,
                    CitationAnchorRecord.document_version_id
                    == KnowledgeChunkRecord.document_version_id,
                ),
            )
            .where(
                KnowledgeChunkRecord.tenant_id == query.tenant_id,
                KnowledgeChunkRecord.release_id == query.release_id,
                KnowledgeDocumentVersionRecord.tenant_id == query.tenant_id,
                KnowledgeDocumentVersionRecord.status.in_(version_statuses),
                KnowledgeDocumentRecord.tenant_id == query.tenant_id,
                CitationAnchorRecord.tenant_id == query.tenant_id,
                KnowledgeDocumentVersionRecord.valid_from <= query.as_of,
                or_(
                    KnowledgeDocumentVersionRecord.valid_to.is_(None),
                    KnowledgeDocumentVersionRecord.valid_to > query.as_of,
                ),
                or_(
                    and_(
                        func.json_array_length(KnowledgeDocumentVersionRecord.acl_subject_ids) == 0,
                        func.json_array_length(KnowledgeDocumentVersionRecord.acl_roles) == 0,
                    ),
                    subject_allowed,
                    role_allowed,
                ),
                or_(
                    func.json_array_length(KnowledgeDocumentVersionRecord.device_models) == 0,
                    model_allowed,
                ),
                or_(
                    func.json_array_length(KnowledgeDocumentVersionRecord.device_families) == 0,
                    family_allowed,
                ),
                or_(KnowledgeDocumentRecord.classification != "restricted", role_allowed),
            )
        )
        rows = session.execute(statement).all()
        candidates = [_candidate(*row) for row in rows]
        _verify_prefiltered_candidates(candidates, query)
        return candidates

    def _postgres_search(
        self,
        session: Session,
        query: RetrievalQuery,
        version_statuses: tuple[str, ...],
    ) -> RetrievalResult:
        params = {
            "tenant_id": query.tenant_id,
            "subject_id": query.subject_id,
            "roles": sorted(query.roles),
            "release_id": query.release_id,
            "device_family": query.device_family,
            "device_model": query.device_model,
            "as_of": query.as_of,
            "query": query.query,
            "embedding": json.dumps(deterministic_embedding(query.query)),
            "candidate_limit": max(query.limit * 4, 12),
            "version_statuses": list(version_statuses),
        }
        filtered = """
            FROM knowledge_chunks c
            JOIN knowledge_document_versions v ON v.document_version_id = c.document_version_id
            JOIN knowledge_documents d ON d.document_id = v.document_id
            JOIN citation_anchors a ON a.chunk_id = c.chunk_id
                                     AND a.document_version_id = c.document_version_id
            WHERE c.tenant_id = :tenant_id
              AND v.tenant_id = :tenant_id
              AND d.tenant_id = :tenant_id
              AND a.tenant_id = :tenant_id
              AND c.release_id = :release_id
              AND v.status = ANY(CAST(:version_statuses AS text[]))
              AND v.valid_from <= :as_of
              AND (v.valid_to IS NULL OR v.valid_to > :as_of)
              AND (
                    jsonb_array_length(CAST(v.acl_subject_ids AS jsonb)) = 0
                    AND jsonb_array_length(CAST(v.acl_roles AS jsonb)) = 0
                    OR CAST(v.acl_subject_ids AS jsonb) ? :subject_id
                    OR CAST(v.acl_roles AS jsonb) ?| CAST(:roles AS text[])
              )
              AND (
                    jsonb_array_length(CAST(v.device_models AS jsonb)) = 0
                    OR CAST(v.device_models AS jsonb) ? :device_model
              )
              AND (
                    jsonb_array_length(CAST(v.device_families AS jsonb)) = 0
                    OR CAST(:device_family AS text) IS NOT NULL
                       AND CAST(v.device_families AS jsonb) ? CAST(:device_family AS text)
              )
              AND (
                    d.classification <> 'restricted'
                    OR CAST(v.acl_roles AS jsonb) ?| CAST(:roles AS text[])
              )
        """
        projection = """
            SELECT c.chunk_id, a.citation_id, d.document_id, v.document_version_id,
                   v.version AS document_version, d.title, c.content, c.content_checksum,
                   v.source_checksum, a.page_number, c.embedding,
                   v.acl_subject_ids, v.acl_roles, v.device_families, v.device_models,
                   v.valid_from, v.valid_to, d.classification
        """
        sparse_sql = text(
            projection
            + filtered
            + """
              AND to_tsvector('simple', c.content) @@ plainto_tsquery('simple', :query)
            ORDER BY ts_rank_cd(
                to_tsvector('simple', c.content), plainto_tsquery('simple', :query)
            ) DESC,
                     c.chunk_id
            LIMIT :candidate_limit
            """
        )
        vector_sql = text(
            projection
            + filtered
            + """
            ORDER BY c.embedding <=> CAST(:embedding AS vector), c.chunk_id
            LIMIT :candidate_limit
            """
        )
        sparse = [_mapping_candidate(row) for row in session.execute(sparse_sql, params).mappings()]
        vector = [_mapping_candidate(row) for row in session.execute(vector_sql, params).mappings()]
        _verify_prefiltered_candidates([*sparse, *vector], query)
        evidence = _rrf(sparse, vector, query.device_model, query.limit)
        return RetrievalResult(
            evidence=tuple(evidence),
            guard=_guard(
                query,
                backend="postgresql",
                sparse_count=len(sparse),
                vector_count=len(vector),
                fused_count=len({item.chunk_id for item in [*sparse, *vector]}),
                returned_count=len(evidence),
            ),
        )


def _candidate(
    chunk: KnowledgeChunkRecord,
    version: KnowledgeDocumentVersionRecord,
    document: KnowledgeDocumentRecord,
    anchor: CitationAnchorRecord,
) -> _Candidate:
    return _Candidate(
        chunk_id=chunk.chunk_id,
        citation_id=anchor.citation_id,
        document_id=document.document_id,
        document_version_id=version.document_version_id,
        document_version=version.version,
        title=document.title,
        content=chunk.content,
        content_checksum=chunk.content_checksum,
        source_checksum=version.source_checksum,
        page_number=anchor.page_number,
        embedding=list(chunk.embedding),
        acl_subject_ids=frozenset(version.acl_subject_ids),
        acl_roles=frozenset(version.acl_roles),
        device_families=frozenset(version.device_families),
        device_models=frozenset(version.device_models),
        valid_from=version.valid_from,
        valid_to=version.valid_to,
        classification=document.classification,
    )


def _mapping_candidate(row: RowMapping) -> _Candidate:
    return _Candidate(
        chunk_id=str(row["chunk_id"]),
        citation_id=str(row["citation_id"]),
        document_id=str(row["document_id"]),
        document_version_id=str(row["document_version_id"]),
        document_version=int(row["document_version"]),
        title=str(row["title"]),
        content=str(row["content"]),
        content_checksum=str(row["content_checksum"]),
        source_checksum=str(row["source_checksum"]),
        page_number=int(row["page_number"]) if row["page_number"] is not None else None,
        embedding=_embedding_values(row["embedding"]),
        acl_subject_ids=_string_set(row["acl_subject_ids"]),
        acl_roles=_string_set(row["acl_roles"]),
        device_families=_string_set(row["device_families"]),
        device_models=_string_set(row["device_models"]),
        valid_from=_datetime_value(row["valid_from"]),
        valid_to=(_datetime_value(row["valid_to"]) if row["valid_to"] is not None else None),
        classification=str(row["classification"]),
    )


def _embedding_values(value: object) -> list[float]:
    """Normalize pgvector values returned through typed or textual SQL paths."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("knowledge chunk embedding is not valid JSON") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError("knowledge chunk embedding has an unsupported database type")
    return [float(item) for item in value]


def _rank(
    candidates: Iterable[_Candidate],
    query: RetrievalQuery,
) -> tuple[list[RetrievedEvidence], int, int, int]:
    query_tokens = set(tokenize(query.query))
    embedding = deterministic_embedding(query.query)
    items = list(candidates)
    sparse = sorted(
        items,
        key=lambda item: (-len(query_tokens.intersection(tokenize(item.content))), item.chunk_id),
    )
    sparse = [item for item in sparse if query_tokens.intersection(tokenize(item.content))]
    vector = sorted(
        items,
        key=lambda item: (-cosine_similarity(embedding, item.embedding), item.chunk_id),
    )
    evidence = _rrf(sparse, vector, query.device_model, query.limit)
    return evidence, len(sparse), len(vector), len({item.chunk_id for item in [*sparse, *vector]})


def _rrf(
    sparse: list[_Candidate],
    vector: list[_Candidate],
    device_model: str,
    limit: int,
) -> list[RetrievedEvidence]:
    sparse_rank = {item.chunk_id: index for index, item in enumerate(sparse, start=1)}
    vector_rank = {item.chunk_id: index for index, item in enumerate(vector, start=1)}
    candidates = {item.chunk_id: item for item in [*sparse, *vector]}
    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            -(
                (1 / (60 + sparse_rank[item.chunk_id]) if item.chunk_id in sparse_rank else 0)
                + (1 / (60 + vector_rank[item.chunk_id]) if item.chunk_id in vector_rank else 0)
            ),
            item.chunk_id,
        ),
    )
    results: list[RetrievedEvidence] = []
    for item in ordered[:limit]:
        sparse_position = sparse_rank.get(item.chunk_id)
        vector_position = vector_rank.get(item.chunk_id)
        score = (1 / (60 + sparse_position) if sparse_position else 0) + (
            1 / (60 + vector_position) if vector_position else 0
        )
        results.append(
            RetrievedEvidence(
                chunk_id=item.chunk_id,
                citation_id=item.citation_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                document_version=item.document_version,
                title=item.title,
                content=item.content,
                content_checksum=item.content_checksum,
                source_checksum=item.source_checksum,
                page_number=item.page_number,
                device_model=device_model,
                sparse_rank=sparse_position,
                vector_rank=vector_position,
                fused_score=score,
            )
        )
    return results


def _verify_prefiltered_candidates(
    candidates: Iterable[_Candidate],
    query: RetrievalQuery,
) -> None:
    unauthorized = {
        item.chunk_id for item in candidates if not _candidate_is_authorized(item, query)
    }
    if unauthorized:
        raise RetrievalAuthorizationInvariantViolation(
            "sql authorization prefilter leaked candidates",
            candidate_count=len(unauthorized),
        )


def _candidate_is_authorized(candidate: _Candidate, query: RetrievalQuery) -> bool:
    valid_from = _as_utc(candidate.valid_from)
    valid_to = _as_utc(candidate.valid_to) if candidate.valid_to is not None else None
    if valid_from > query.as_of or valid_to is not None and valid_to <= query.as_of:
        return False
    subject_allowed = query.subject_id in candidate.acl_subject_ids
    role_allowed = bool(query.roles.intersection(candidate.acl_roles))
    if (
        (candidate.acl_subject_ids or candidate.acl_roles)
        and not subject_allowed
        and not role_allowed
    ):
        return False
    if candidate.device_models and query.device_model not in candidate.device_models:
        return False
    if candidate.device_families and query.device_family not in candidate.device_families:
        return False
    return candidate.classification != "restricted" or role_allowed


def _guard(
    query: RetrievalQuery,
    *,
    backend: str,
    sparse_count: int,
    vector_count: int,
    fused_count: int,
    returned_count: int,
) -> RetrievalGuardTrace:
    return RetrievalGuardTrace(
        release_id=query.release_id,
        backend=backend,
        filter_stage="SQL_PRE_RECALL",
        policy_dimensions=_POLICY_DIMENSIONS,
        sparse_candidate_count=sparse_count,
        vector_candidate_count=vector_count,
        fused_candidate_count=fused_count,
        returned_count=returned_count,
        unauthorized_candidate_count=0,
    )


def _string_set(value: object) -> frozenset[str]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise RetrievalAuthorizationInvariantViolation("candidate policy metadata is invalid")
    return frozenset(parsed)


def _datetime_value(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise RetrievalAuthorizationInvariantViolation(
                "candidate validity metadata is invalid"
            ) from exc
    raise RetrievalAuthorizationInvariantViolation("candidate validity metadata is invalid")


__all__ = [
    "HybridRetriever",
    "KnowledgeReleaseUnavailable",
    "RetrievalAuthorizationInvariantViolation",
]
