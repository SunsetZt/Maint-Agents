"""Governed OpenSearch profile lifecycle, A/B gates, and authorized search."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.knowledge.embedding import deterministic_embedding
from industrial_ops_agent.knowledge.models import RetrievalQuery, RetrievedEvidence
from industrial_ops_agent.knowledge.retrieval import HybridRetriever
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetRecord,
    CitationAnchorRecord,
    IdempotencyRecord,
    IndexReleaseRecord,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    KnowledgeDocumentVersionRecord,
    KnowledgeSearchEvaluationRecord,
    KnowledgeSearchProfileRecord,
    KnowledgeSearchProjectionRecord,
    KnowledgeSearchRebuildJobRecord,
)
from industrial_ops_agent.search_profiles.embedding import (
    EmbeddingInput,
    EmbeddingOutcome,
    EmbeddingUnavailable,
    KnowledgeEmbedder,
)
from industrial_ops_agent.search_profiles.reranker import (
    KnowledgeReranker,
    RerankCandidate,
    RerankerUnavailable,
)
from industrial_ops_agent.search_profiles.store import SearchDocument, SearchHit, SearchStore

POLICY_VERSION = "opensearch-shadow-ab-gates-v1"
SCHEMA_VERSION = "industrial-knowledge-search-v2"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.:-]{1,127}$")
SearchRebuildProgress = Callable[[str, int, int, int], None]


class KnowledgeSearchNotVisible(Exception):
    pass


class KnowledgeSearchConflict(RuntimeError):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class SearchEvaluationView:
    evaluation_id: str
    status: str
    policy_version: str
    cases: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    gate_results: dict[str, bool]
    failure_codes: tuple[str, ...]
    requested_by_subject_id: str
    completed_at: datetime | None
    version: int


@dataclass(frozen=True, slots=True)
class SearchProfileView:
    search_profile_id: str
    source_index_release_id: str
    name: str
    version: int
    backend: str
    status: str
    is_active_shadow: bool
    index_name: str
    schema_version: str
    manifest_hash: str | None
    embedding_model_release_id: str | None
    embedding_manifest_hash: str | None
    embedding_component_model_id: str | None
    embedding_artifact_content_hash: str | None
    embedding_dimension: int | None
    document_count: int
    metrics: dict[str, Any]
    gate_results: dict[str, bool]
    failure_codes: tuple[str, ...]
    created_by_subject_id: str
    evaluated_by_subject_id: str | None
    activated_by_subject_id: str | None
    evaluated_at: datetime | None
    activated_at: datetime | None
    state_version: int
    created_at: datetime
    updated_at: datetime
    evaluation: SearchEvaluationView | None


@dataclass(frozen=True, slots=True)
class SearchResultView:
    profile_id: str
    source_index_release_id: str
    backend: str
    guard_stage: str
    embedding: EmbeddingTrace
    rerank: RerankTrace
    evidence: tuple[RetrievedEvidence, ...]


@dataclass(frozen=True, slots=True)
class RerankTrace:
    status: str
    candidate_count: int
    returned_count: int
    model_release_id: str | None = None
    manifest_hash: str | None = None
    component_model_id: str | None = None
    artifact_content_hash: str | None = None
    latency_ms: float | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingTrace:
    status: str
    dimension: int | None
    model_release_id: str | None = None
    manifest_hash: str | None = None
    component_model_id: str | None = None
    artifact_content_hash: str | None = None
    latency_ms: float | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _EmbeddingBinding:
    model_release_id: str | None
    manifest_hash: str | None
    component_model_id: str | None
    artifact_content_hash: str | None
    dimension: int | None


class KnowledgeSearchProfileService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        store: SearchStore,
        *,
        index_prefix: str = "ioap-knowledge",
        reranker: KnowledgeReranker | None = None,
        embedder: KnowledgeEmbedder | None = None,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._store = store
        self._index_prefix = index_prefix
        self._baseline = HybridRetriever(database)
        self._reranker = reranker
        self._embedder = embedder

    def list(self, identity: IdentityContext, *, request_id: str) -> tuple[SearchProfileView, ...]:
        self._require(identity, Action.READ_KNOWLEDGE_SEARCH, "knowledge-search", request_id)
        with self._database.transaction(identity.tenant_context) as session:
            records = tuple(
                session.scalars(
                    select(KnowledgeSearchProfileRecord)
                    .where(KnowledgeSearchProfileRecord.tenant_id == identity.tenant_id)
                    .order_by(
                        KnowledgeSearchProfileRecord.updated_at.desc(),
                        KnowledgeSearchProfileRecord.search_profile_id,
                    )
                )
            )
            return tuple(_profile_view(session, identity.tenant_id, item) for item in records)

    def get(
        self, identity: IdentityContext, search_profile_id: str, *, request_id: str
    ) -> SearchProfileView:
        self._require(identity, Action.READ_KNOWLEDGE_SEARCH, search_profile_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            return _profile_view(
                session,
                identity.tenant_id,
                _profile_or_hidden(session, identity.tenant_id, search_profile_id),
            )

    def create(
        self,
        identity: IdentityContext,
        *,
        source_index_release_id: str,
        name: str,
        idempotency_key: str,
        request_id: str,
    ) -> SearchProfileView:
        self._require(identity, Action.MANAGE_KNOWLEDGE_SEARCH, source_index_release_id, request_id)
        normalized_name = name.strip().lower()
        if not _NAME.fullmatch(normalized_name) or not 8 <= len(idempotency_key) <= 255:
            raise KnowledgeSearchConflict("search_profile_input_invalid")
        fingerprint = _digest(
            {"source_index_release_id": source_index_release_id, "name": normalized_name}
        )
        storage_key = f"knowledge-search:{idempotency_key}"
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            replay = session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.tenant_id == identity.tenant_id,
                    IdempotencyRecord.subject_id == identity.subject_id,
                    IdempotencyRecord.key == storage_key,
                )
            )
            if replay is not None:
                if replay.request_hash != fingerprint:
                    raise KnowledgeSearchConflict("search_profile_idempotency_conflict")
                return _profile_view(
                    session,
                    identity.tenant_id,
                    _profile_or_hidden(session, identity.tenant_id, str(replay.result_ref)),
                )
            source = session.scalar(
                select(IndexReleaseRecord).where(
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.release_id == source_index_release_id,
                    IndexReleaseRecord.status == "PUBLISHED",
                )
            )
            if source is None:
                raise KnowledgeSearchNotVisible
            next_version = (
                int(
                    session.scalar(
                        select(
                            func.coalesce(func.max(KnowledgeSearchProfileRecord.version), 0)
                        ).where(
                            KnowledgeSearchProfileRecord.tenant_id == identity.tenant_id,
                            KnowledgeSearchProfileRecord.name == normalized_name,
                        )
                    )
                    or 0
                )
                + 1
            )
            profile_id = f"search-profile-{uuid4().hex}"
            tenant_digest = sha256(identity.tenant_id.encode()).hexdigest()[:10]
            index_name = f"{self._index_prefix}-{tenant_digest}-{uuid4().hex[:16]}"
            record = KnowledgeSearchProfileRecord(
                search_profile_id=profile_id,
                tenant_id=identity.tenant_id,
                source_index_release_id=source_index_release_id,
                name=normalized_name,
                version=next_version,
                backend="OPENSEARCH",
                status="DRAFT",
                is_active_shadow=False,
                index_name=index_name,
                schema_version=SCHEMA_VERSION,
                manifest_hash=None,
                embedding_model_release_id=None,
                embedding_manifest_hash=None,
                embedding_component_model_id=None,
                embedding_artifact_content_hash=None,
                embedding_dimension=None,
                document_count=0,
                metrics_json={},
                gate_results_json={},
                failure_codes=[],
                created_by_subject_id=identity.subject_id,
                evaluated_by_subject_id=None,
                activated_by_subject_id=None,
                evaluated_at=None,
                activated_at=None,
                state_version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=fingerprint,
                    result_ref=profile_id,
                    expires_at=now + timedelta(days=30),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _profile_view(session, identity.tenant_id, record)

    def sync(
        self,
        identity: IdentityContext,
        search_profile_id: str,
        *,
        expected_version: int,
        request_id: str,
        progress: SearchRebuildProgress | None = None,
        rebuild_job_id: str | None = None,
    ) -> SearchProfileView:
        self._require(identity, Action.MANAGE_KNOWLEDGE_SEARCH, search_profile_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            profile = _profile_or_hidden(session, identity.tenant_id, search_profile_id, lock=True)
            _require_state(profile, expected_version, ("DRAFT", "SYNCED"))
            _require_rebuild_owner(session, profile, rebuild_job_id)
            documents = _projection_documents(session, profile)
            if not documents:
                raise KnowledgeSearchConflict("search_profile_source_empty")
            index_name = profile.index_name
            source_release_id = profile.source_index_release_id
            existing_embedding_release_id = profile.embedding_model_release_id
        if progress is not None:
            progress("LOADING_SOURCE", 0, len(documents), 10)
        embedding_outcome = None
        if self._embedder is not None:
            documents, embedding_outcome = self._embed_projection_documents(
                identity,
                source_release_id=source_release_id,
                documents=documents,
                progress=progress,
            )
        elif existing_embedding_release_id is not None:
            raise EmbeddingUnavailable(
                "embedding_runtime_not_configured",
                release_id=existing_embedding_release_id,
            )
        elif progress is not None:
            progress("PREPARING_INDEX", len(documents), len(documents), 70)
        manifest_hash = _digest(
            {
                "schema_version": SCHEMA_VERSION,
                "embedding": (
                    {
                        "model_release_id": embedding_outcome.model_release_id,
                        "manifest_hash": embedding_outcome.manifest_hash,
                        "component_model_id": embedding_outcome.component_model_id,
                        "artifact_content_hash": embedding_outcome.artifact_content_hash,
                        "dimension": embedding_outcome.dimension,
                    }
                    if embedding_outcome is not None
                    else {"profile": "legacy-deterministic-embedding-v1", "dimension": 16}
                ),
                "documents": [
                    {
                        "chunk_id": item.chunk_id,
                        "citation_id": item.citation_id,
                        "content_checksum": item.content_checksum,
                        "source_checksum": item.source_checksum,
                    }
                    for item in documents
                ],
            }
        )
        if progress is not None:
            progress("INDEXING", len(documents), len(documents), 80)
        self._store.replace_release(index_name, documents)
        if progress is not None:
            progress("VERIFYING", len(documents), len(documents), 90)
        external_count = self._store.count(index_name)
        if external_count != len(documents):
            raise KnowledgeSearchConflict("search_profile_projection_count_mismatch")
        now = datetime.now(UTC)
        if progress is not None:
            progress("PERSISTING", len(documents), len(documents), 95)
        with self._database.transaction(identity.tenant_context) as session:
            profile = _profile_or_hidden(session, identity.tenant_id, search_profile_id, lock=True)
            _require_state(profile, expected_version, ("DRAFT", "SYNCED"))
            _require_rebuild_owner(session, profile, rebuild_job_id)
            session.execute(
                delete(KnowledgeSearchProjectionRecord).where(
                    KnowledgeSearchProjectionRecord.tenant_id == identity.tenant_id,
                    KnowledgeSearchProjectionRecord.search_profile_id == search_profile_id,
                )
            )
            session.add_all(
                [
                    KnowledgeSearchProjectionRecord(
                        projection_id=f"search-projection-{uuid4().hex}",
                        tenant_id=identity.tenant_id,
                        search_profile_id=search_profile_id,
                        citation_id=item.citation_id,
                        chunk_id=item.chunk_id,
                        document_id=item.document_id,
                        document_version_id=item.document_version_id,
                        content_checksum=item.content_checksum,
                        source_checksum=item.source_checksum,
                        created_at=now,
                        updated_at=now,
                    )
                    for item in documents
                ]
            )
            profile.status = "SYNCED"
            profile.manifest_hash = manifest_hash
            if embedding_outcome is not None:
                profile.embedding_model_release_id = embedding_outcome.model_release_id
                profile.embedding_manifest_hash = embedding_outcome.manifest_hash
                profile.embedding_component_model_id = embedding_outcome.component_model_id
                profile.embedding_artifact_content_hash = embedding_outcome.artifact_content_hash
                profile.embedding_dimension = embedding_outcome.dimension
            profile.document_count = len(documents)
            profile.state_version += 1
            profile.updated_at = now
            session.flush()
            return _profile_view(session, identity.tenant_id, profile)

    def _embed_projection_documents(
        self,
        identity: IdentityContext,
        *,
        source_release_id: str,
        documents: tuple[SearchDocument, ...],
        progress: SearchRebuildProgress | None = None,
    ) -> tuple[tuple[SearchDocument, ...], EmbeddingOutcome]:
        if self._embedder is None:
            raise EmbeddingUnavailable("embedding_runtime_not_configured")
        outcomes: list[EmbeddingOutcome] = []
        vectors: dict[str, tuple[float, ...]] = {}
        for offset in range(0, len(documents), 64):
            batch = documents[offset : offset + 64]
            outcome = self._embedder.embed(
                identity.tenant_context,
                index_release_id=source_release_id,
                inputs=tuple(
                    EmbeddingInput(
                        item_id=item.chunk_id,
                        text=f"{item.title}\n{item.content}",
                    )
                    for item in batch
                ),
            )
            if outcomes and not _same_embedding_binding(outcomes[0], outcome):
                raise EmbeddingUnavailable(
                    "embedding_release_changed_during_rebuild",
                    release_id=outcome.model_release_id,
                )
            outcomes.append(outcome)
            vectors.update({item.item_id: item.vector for item in outcome.items})
            if progress is not None:
                completed = min(offset + len(batch), len(documents))
                percent = 10 + round(60 * completed / len(documents))
                progress("EMBEDDING", completed, len(documents), percent)
        first = outcomes[0]
        if set(vectors) != {item.chunk_id for item in documents}:
            raise EmbeddingUnavailable(
                "embedding_rebuild_result_incomplete",
                release_id=first.model_release_id,
            )
        combined = EmbeddingOutcome(
            model_release_id=first.model_release_id,
            manifest_hash=first.manifest_hash,
            component_model_id=first.component_model_id,
            artifact_content_hash=first.artifact_content_hash,
            dimension=first.dimension,
            latency_ms=sum(item.latency_ms for item in outcomes),
            items=tuple(item for outcome in outcomes for item in outcome.items),
        )
        return (
            tuple(replace(item, embedding=vectors[item.chunk_id]) for item in documents),
            combined,
        )

    def evaluate(
        self,
        identity: IdentityContext,
        search_profile_id: str,
        *,
        cases: tuple[dict[str, Any], ...],
        expected_version: int,
        request_id: str,
    ) -> SearchProfileView:
        self._require(identity, Action.EVALUATE_KNOWLEDGE_SEARCH, search_profile_id, request_id)
        normalized_cases = tuple(_validate_case(item) for item in cases)
        with self._database.transaction(identity.tenant_context) as session:
            profile = _profile_or_hidden(session, identity.tenant_id, search_profile_id, lock=True)
            _require_state(profile, expected_version, ("SYNCED",))
            if profile.created_by_subject_id == identity.subject_id:
                raise KnowledgeSearchConflict("search_profile_evaluation_separation_required")
            projection_count = int(
                session.scalar(
                    select(func.count(KnowledgeSearchProjectionRecord.projection_id)).where(
                        KnowledgeSearchProjectionRecord.tenant_id == identity.tenant_id,
                        KnowledgeSearchProjectionRecord.search_profile_id == search_profile_id,
                    )
                )
                or 0
            )
            source_release_id = profile.source_index_release_id
            index_name = profile.index_name
            expected_document_count = profile.document_count
            embedding_binding = _embedding_binding(profile)
        baseline_hits = 0
        candidate_hits = 0
        candidate_unauthorized = 0
        candidate_result_count = 0
        embedding_fallback_count = 0
        roles = tuple(sorted(role.value for role in identity.roles))
        for case in normalized_cases:
            query = RetrievalQuery(
                tenant_id=identity.tenant_id,
                subject_id=identity.subject_id,
                roles=frozenset(roles),
                release_id=source_release_id,
                device_family=case["device_family"],
                device_model=case["device_model"],
                query=case["query"],
                as_of=datetime.now(UTC),
                limit=case["limit"],
            )
            baseline = self._baseline.search(query)
            query_embedding, embedding_trace = self._query_embedding(
                identity,
                source_release_id=source_release_id,
                query_text=case["query"],
                binding=embedding_binding,
            )
            if embedding_trace.status == "FALLBACK_SPARSE":
                embedding_fallback_count += 1
            candidate = self._store.search(
                index_name,
                tenant_id=identity.tenant_id,
                search_profile_id=search_profile_id,
                subject_id=identity.subject_id,
                roles=roles,
                device_family=case["device_family"],
                device_model=case["device_model"],
                query_text=case["query"],
                query_embedding=query_embedding,
                as_of=query.as_of,
                limit=case["limit"],
            )
            candidate_unauthorized += self._count_unauthorized(
                identity,
                query,
                candidate,
                expected_profile_id=search_profile_id,
            )
            expected = set(case["expected_citation_ids"])
            baseline_hits += len(expected.intersection(item.citation_id for item in baseline))
            candidate_hits += len(
                expected.intersection(item.document.citation_id for item in candidate)
            )
            candidate_result_count += len(candidate)
        expected_total = sum(len(item["expected_citation_ids"]) for item in normalized_cases)
        baseline_recall = baseline_hits / expected_total if expected_total else 0.0
        candidate_recall = candidate_hits / expected_total if expected_total else 0.0
        external_count = self._store.count(index_name)
        gates = {
            "minimum_gold_cases": len(normalized_cases) >= 3,
            "projection_consistent": projection_count == expected_document_count == external_count,
            "authorization_pre_recall": candidate_unauthorized == 0,
            "candidate_recall_floor": candidate_recall >= 0.8,
            "baseline_non_regression": candidate_recall >= baseline_recall,
            "independent_evaluator": True,
            "embedding_runtime_bound": embedding_fallback_count == 0,
        }
        failures = sorted(name for name, passed in gates.items() if not passed)
        metrics: dict[str, Any] = {
            "gold_case_count": len(normalized_cases),
            "expected_citation_count": expected_total,
            "baseline_expected_recall_at_k": baseline_recall,
            "candidate_expected_recall_at_k": candidate_recall,
            "recall_delta": candidate_recall - baseline_recall,
            "candidate_result_count": candidate_result_count,
            "candidate_unauthorized_count": candidate_unauthorized,
            "embedding_fallback_case_count": embedding_fallback_count,
            "sql_projection_count": projection_count,
            "external_projection_count": external_count,
        }
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            profile = _profile_or_hidden(session, identity.tenant_id, search_profile_id, lock=True)
            _require_state(profile, expected_version, ("SYNCED",))
            session.add(
                KnowledgeSearchEvaluationRecord(
                    evaluation_id=f"search-evaluation-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    search_profile_id=search_profile_id,
                    status="FAILED" if failures else "PASSED",
                    policy_version=POLICY_VERSION,
                    cases_json=list(normalized_cases),
                    metrics_json=metrics,
                    gate_results_json=gates,
                    failure_codes=failures,
                    requested_by_subject_id=identity.subject_id,
                    completed_at=now,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            profile.status = "REJECTED" if failures else "EVALUATED"
            profile.metrics_json = metrics
            profile.gate_results_json = gates
            profile.failure_codes = failures
            profile.evaluated_by_subject_id = identity.subject_id
            profile.evaluated_at = now
            profile.state_version += 1
            profile.updated_at = now
            session.flush()
            return _profile_view(session, identity.tenant_id, profile)

    def activate_shadow(
        self,
        identity: IdentityContext,
        search_profile_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> SearchProfileView:
        self._require(identity, Action.ACTIVATE_KNOWLEDGE_SEARCH, search_profile_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            profile = _profile_or_hidden(session, identity.tenant_id, search_profile_id, lock=True)
            if (
                profile.state_version != expected_version
                or profile.status != "EVALUATED"
                or not profile.gate_results_json
                or not all(profile.gate_results_json.values())
            ):
                raise KnowledgeSearchConflict(
                    "search_profile_activation_gate_failed", profile.state_version
                )
            session.execute(
                update(KnowledgeSearchProfileRecord)
                .where(
                    KnowledgeSearchProfileRecord.tenant_id == identity.tenant_id,
                    KnowledgeSearchProfileRecord.name == profile.name,
                    KnowledgeSearchProfileRecord.is_active_shadow.is_(True),
                )
                .values(
                    is_active_shadow=False,
                    status="RETIRED",
                    state_version=KnowledgeSearchProfileRecord.state_version + 1,
                    updated_at=now,
                )
            )
            profile.status = "ACTIVE_SHADOW"
            profile.is_active_shadow = True
            profile.activated_by_subject_id = identity.subject_id
            profile.activated_at = now
            profile.state_version += 1
            profile.updated_at = now
            session.flush()
            return _profile_view(session, identity.tenant_id, profile)

    def query(
        self,
        identity: IdentityContext,
        *,
        profile_name: str,
        query_text: str,
        device_family: str | None,
        device_model: str,
        limit: int,
        request_id: str,
    ) -> SearchResultView:
        self._require(identity, Action.READ_KNOWLEDGE_SEARCH, "active-shadow-search", request_id)
        normalized_name = profile_name.strip().lower()
        if not _NAME.fullmatch(normalized_name) or not query_text.strip() or not 1 <= limit <= 20:
            raise KnowledgeSearchConflict("search_query_input_invalid")
        with self._database.transaction(identity.tenant_context) as session:
            profile = session.scalar(
                select(KnowledgeSearchProfileRecord).where(
                    KnowledgeSearchProfileRecord.tenant_id == identity.tenant_id,
                    KnowledgeSearchProfileRecord.name == normalized_name,
                    KnowledgeSearchProfileRecord.status == "ACTIVE_SHADOW",
                    KnowledgeSearchProfileRecord.is_active_shadow.is_(True),
                )
            )
            if profile is None:
                raise KnowledgeSearchNotVisible
            if not _can_query_arbitrary_model(identity):
                visible_models = frozenset(
                    model
                    for model in session.scalars(
                        select(AssetRecord.model_code).where(
                            AssetRecord.tenant_id == identity.tenant_id,
                            AssetRecord.asset_id.in_(identity.asset_ids),
                        )
                    )
                    if model is not None
                )
                if device_model not in visible_models:
                    raise KnowledgeSearchNotVisible
            profile_id = profile.search_profile_id
            index_name = profile.index_name
            source_release_id = profile.source_index_release_id
            embedding_binding = _embedding_binding(profile)
        now = datetime.now(UTC)
        roles = tuple(sorted(role.value for role in identity.roles))
        retrieval_query = RetrievalQuery(
            tenant_id=identity.tenant_id,
            subject_id=identity.subject_id,
            roles=frozenset(roles),
            release_id=source_release_id,
            device_family=device_family,
            device_model=device_model,
            query=query_text.strip(),
            as_of=now,
            limit=limit,
        )
        query_embedding, embedding_trace = self._query_embedding(
            identity,
            source_release_id=source_release_id,
            query_text=query_text.strip(),
            binding=embedding_binding,
        )
        candidate_limit = min(max(limit * 4, 12), 50) if self._reranker is not None else limit
        hits = self._store.search(
            index_name,
            tenant_id=identity.tenant_id,
            search_profile_id=profile_id,
            subject_id=identity.subject_id,
            roles=roles,
            device_family=device_family,
            device_model=device_model,
            query_text=query_text.strip(),
            query_embedding=query_embedding,
            as_of=now,
            limit=candidate_limit,
        )
        if self._count_unauthorized(
            identity,
            retrieval_query,
            hits,
            expected_profile_id=profile_id,
        ):
            raise KnowledgeSearchConflict("search_authorization_invariant_violation")
        selected_hits = hits[:limit]
        reranker_scores: dict[str, float] = {}
        rerank = RerankTrace(
            status="NOT_CONFIGURED",
            candidate_count=len(hits),
            returned_count=len(selected_hits),
        )
        if self._reranker is not None and hits:
            try:
                outcome = self._reranker.rerank(
                    identity.tenant_context,
                    index_release_id=source_release_id,
                    query=query_text.strip(),
                    candidates=tuple(
                        RerankCandidate(
                            chunk_id=item.document.chunk_id,
                            title=item.document.title,
                            content=item.document.content,
                            recall_score=item.fused_score,
                        )
                        for item in hits
                    ),
                    limit=min(limit, len(hits)),
                )
                hits_by_id = {item.document.chunk_id: item for item in hits}
                returned_ids = [item.chunk_id for item in outcome.items]
                if (
                    not returned_ids
                    or len(returned_ids) != len(set(returned_ids))
                    or not set(returned_ids).issubset(hits_by_id)
                    or len(returned_ids) > limit
                ):
                    raise RerankerUnavailable(
                        "reranker_result_contract_invalid",
                        release_id=outcome.model_release_id,
                    )
                selected_hits = tuple(hits_by_id[item_id] for item_id in returned_ids)
                reranker_scores = {item.chunk_id: item.score for item in outcome.items}
                rerank = RerankTrace(
                    status="APPLIED",
                    candidate_count=len(hits),
                    returned_count=len(selected_hits),
                    model_release_id=outcome.model_release_id,
                    manifest_hash=outcome.manifest_hash,
                    component_model_id=outcome.component_model_id,
                    artifact_content_hash=outcome.artifact_content_hash,
                    latency_ms=outcome.latency_ms,
                )
            except RerankerUnavailable as exc:
                selected_hits = hits[:limit]
                rerank = RerankTrace(
                    status="FALLBACK",
                    candidate_count=len(hits),
                    returned_count=len(selected_hits),
                    model_release_id=exc.release_id,
                    fallback_reason=exc.reason,
                )
        return SearchResultView(
            profile_id=profile_id,
            source_index_release_id=source_release_id,
            backend="opensearch",
            guard_stage="OPENSEARCH_PRE_RECALL_SQL_DEFENSIVE_RECHECK",
            embedding=embedding_trace,
            rerank=rerank,
            evidence=tuple(
                _evidence(
                    item,
                    device_model,
                    reranker_score=reranker_scores.get(item.document.chunk_id),
                    final_rank=rank if rerank.status == "APPLIED" else None,
                )
                for rank, item in enumerate(selected_hits, start=1)
            ),
        )

    def _query_embedding(
        self,
        identity: IdentityContext,
        *,
        source_release_id: str,
        query_text: str,
        binding: _EmbeddingBinding,
    ) -> tuple[tuple[float, ...] | None, EmbeddingTrace]:
        if binding.model_release_id is None:
            vector = tuple(deterministic_embedding(query_text))
            return vector, EmbeddingTrace(status="LEGACY", dimension=len(vector))
        if (
            binding.manifest_hash is None
            or binding.component_model_id is None
            or binding.artifact_content_hash is None
            or binding.dimension is None
        ):
            return None, EmbeddingTrace(
                status="FALLBACK_SPARSE",
                dimension=binding.dimension,
                model_release_id=binding.model_release_id,
                fallback_reason="embedding_profile_binding_incomplete",
            )
        if self._embedder is None:
            return None, EmbeddingTrace(
                status="FALLBACK_SPARSE",
                dimension=binding.dimension,
                model_release_id=binding.model_release_id,
                manifest_hash=binding.manifest_hash,
                component_model_id=binding.component_model_id,
                artifact_content_hash=binding.artifact_content_hash,
                fallback_reason="embedding_runtime_not_configured",
            )
        try:
            outcome = self._embedder.embed(
                identity.tenant_context,
                index_release_id=source_release_id,
                inputs=(EmbeddingInput(item_id="query", text=query_text),),
            )
            if (
                outcome.model_release_id != binding.model_release_id
                or outcome.manifest_hash != binding.manifest_hash
                or outcome.component_model_id != binding.component_model_id
                or outcome.artifact_content_hash != binding.artifact_content_hash
                or outcome.dimension != binding.dimension
                or len(outcome.items) != 1
                or outcome.items[0].item_id != "query"
            ):
                raise EmbeddingUnavailable(
                    "embedding_query_profile_binding_mismatch",
                    release_id=outcome.model_release_id,
                )
        except EmbeddingUnavailable as exc:
            return None, EmbeddingTrace(
                status="FALLBACK_SPARSE",
                dimension=binding.dimension,
                model_release_id=exc.release_id or binding.model_release_id,
                manifest_hash=binding.manifest_hash,
                component_model_id=binding.component_model_id,
                artifact_content_hash=binding.artifact_content_hash,
                fallback_reason=exc.reason,
            )
        return outcome.items[0].vector, EmbeddingTrace(
            status="APPLIED",
            dimension=outcome.dimension,
            model_release_id=outcome.model_release_id,
            manifest_hash=outcome.manifest_hash,
            component_model_id=outcome.component_model_id,
            artifact_content_hash=outcome.artifact_content_hash,
            latency_ms=outcome.latency_ms,
        )

    def deactivate_shadow(
        self,
        identity: IdentityContext,
        search_profile_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> SearchProfileView:
        """Remove a candidate from the query surface; PostgreSQL needs no recovery action."""

        self._require(identity, Action.ACTIVATE_KNOWLEDGE_SEARCH, search_profile_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            profile = _profile_or_hidden(session, identity.tenant_id, search_profile_id, lock=True)
            _require_state(profile, expected_version, ("ACTIVE_SHADOW",))
            profile.status = "RETIRED"
            profile.is_active_shadow = False
            profile.state_version += 1
            profile.updated_at = now
            session.flush()
            return _profile_view(session, identity.tenant_id, profile)

    def _count_unauthorized(
        self,
        identity: IdentityContext,
        query: RetrievalQuery,
        hits: tuple[SearchHit, ...],
        *,
        expected_profile_id: str,
    ) -> int:
        if not hits:
            return 0
        with self._database.transaction(identity.tenant_context) as session:
            authorized_signatures = _authorized_document_signatures(
                session, identity.tenant_id, query, tuple(item.document.chunk_id for item in hits)
            )
        return sum(
            1
            for hit in hits
            if hit.document.tenant_id != identity.tenant_id
            or hit.document.search_profile_id != expected_profile_id
            or hit.document.source_index_release_id != query.release_id
            or _document_signature(hit.document) not in authorized_signatures
        )

    def _require(
        self, identity: IdentityContext, action: Action, resource_id: str, request_id: str
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(identity.tenant_id, resource_id),
            request_id=request_id,
        )


def purge_search_source(
    session: Session, *, tenant_id: str, document_version_id: str
) -> tuple[int, tuple[str, ...]]:
    projections = tuple(
        session.scalars(
            select(KnowledgeSearchProjectionRecord).where(
                KnowledgeSearchProjectionRecord.tenant_id == tenant_id,
                KnowledgeSearchProjectionRecord.document_version_id == document_version_id,
            )
        )
    )
    profile_ids = tuple(sorted({item.search_profile_id for item in projections}))
    if projections:
        session.execute(
            delete(KnowledgeSearchProjectionRecord).where(
                KnowledgeSearchProjectionRecord.tenant_id == tenant_id,
                KnowledgeSearchProjectionRecord.document_version_id == document_version_id,
            )
        )
    if profile_ids:
        session.execute(
            update(KnowledgeSearchProfileRecord)
            .where(
                KnowledgeSearchProfileRecord.tenant_id == tenant_id,
                KnowledgeSearchProfileRecord.search_profile_id.in_(profile_ids),
            )
            .values(
                status="REVOKED",
                is_active_shadow=False,
                failure_codes=["source_document_deleted"],
                state_version=KnowledgeSearchProfileRecord.state_version + 1,
                updated_at=datetime.now(UTC),
            )
        )
    return len(projections), profile_ids


def _projection_documents(
    session: Session, profile: KnowledgeSearchProfileRecord
) -> tuple[SearchDocument, ...]:
    rows = session.execute(
        select(
            KnowledgeChunkRecord,
            CitationAnchorRecord,
            KnowledgeDocumentVersionRecord,
            KnowledgeDocumentRecord,
        )
        .join(
            CitationAnchorRecord,
            and_(
                CitationAnchorRecord.chunk_id == KnowledgeChunkRecord.chunk_id,
                CitationAnchorRecord.document_version_id
                == KnowledgeChunkRecord.document_version_id,
            ),
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
        .where(
            KnowledgeChunkRecord.tenant_id == profile.tenant_id,
            KnowledgeChunkRecord.release_id == profile.source_index_release_id,
            CitationAnchorRecord.tenant_id == profile.tenant_id,
            KnowledgeDocumentVersionRecord.tenant_id == profile.tenant_id,
            KnowledgeDocumentVersionRecord.status == "PUBLISHED",
            KnowledgeDocumentRecord.tenant_id == profile.tenant_id,
        )
        .order_by(KnowledgeChunkRecord.chunk_id, CitationAnchorRecord.citation_id)
    ).all()
    by_chunk: dict[str, SearchDocument] = {}
    for chunk, anchor, version, document in rows:
        by_chunk.setdefault(
            chunk.chunk_id,
            SearchDocument(
                chunk_id=chunk.chunk_id,
                citation_id=anchor.citation_id,
                tenant_id=profile.tenant_id,
                search_profile_id=profile.search_profile_id,
                source_index_release_id=profile.source_index_release_id,
                document_id=document.document_id,
                document_version_id=version.document_version_id,
                document_version=version.version,
                title=document.title,
                content=chunk.content,
                content_checksum=chunk.content_checksum,
                source_checksum=version.source_checksum,
                page_number=anchor.page_number,
                embedding=tuple(float(value) for value in chunk.embedding),
                acl_subject_ids=tuple(version.acl_subject_ids),
                acl_roles=tuple(version.acl_roles),
                device_families=tuple(version.device_families),
                device_models=tuple(version.device_models),
                valid_from=version.valid_from,
                valid_to=version.valid_to,
                classification=document.classification,
            ),
        )
    return tuple(by_chunk[key] for key in sorted(by_chunk))


def _authorized_document_signatures(
    session: Session,
    tenant_id: str,
    query: RetrievalQuery,
    chunk_ids: tuple[str, ...],
) -> set[tuple[str, str, str, str, str, str]]:
    roles = sorted(query.roles)
    records = session.execute(
        select(
            KnowledgeChunkRecord,
            CitationAnchorRecord,
            KnowledgeDocumentVersionRecord,
            KnowledgeDocumentRecord,
        )
        .join(
            CitationAnchorRecord,
            and_(
                CitationAnchorRecord.chunk_id == KnowledgeChunkRecord.chunk_id,
                CitationAnchorRecord.document_version_id
                == KnowledgeChunkRecord.document_version_id,
            ),
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
        .where(
            KnowledgeChunkRecord.tenant_id == tenant_id,
            KnowledgeChunkRecord.release_id == query.release_id,
            KnowledgeChunkRecord.chunk_id.in_(chunk_ids),
            CitationAnchorRecord.tenant_id == tenant_id,
            KnowledgeDocumentVersionRecord.tenant_id == tenant_id,
            KnowledgeDocumentVersionRecord.status == "PUBLISHED",
            KnowledgeDocumentRecord.tenant_id == tenant_id,
        )
    ).all()
    authorized: set[tuple[str, str, str, str, str, str]] = set()
    for chunk, anchor, version, document in records:
        valid_from = _as_utc(version.valid_from)
        valid_to = _as_utc(version.valid_to) if version.valid_to is not None else None
        role_allowed = bool(set(roles).intersection(version.acl_roles))
        if (
            valid_from <= query.as_of
            and (valid_to is None or valid_to > query.as_of)
            and (
                not version.acl_subject_ids
                and not version.acl_roles
                or query.subject_id in version.acl_subject_ids
                or role_allowed
            )
            and (not version.device_models or query.device_model in version.device_models)
            and (not version.device_families or query.device_family in version.device_families)
            and (document.classification != "restricted" or role_allowed)
        ):
            authorized.add(
                (
                    chunk.chunk_id,
                    anchor.citation_id,
                    document.document_id,
                    version.document_version_id,
                    chunk.content_checksum,
                    version.source_checksum,
                )
            )
    return authorized


def _document_signature(
    document: SearchDocument,
) -> tuple[str, str, str, str, str, str]:
    return (
        document.chunk_id,
        document.citation_id,
        document.document_id,
        document.document_version_id,
        document.content_checksum,
        document.source_checksum,
    )


def _evidence(
    hit: SearchHit,
    device_model: str,
    *,
    reranker_score: float | None = None,
    final_rank: int | None = None,
) -> RetrievedEvidence:
    item = hit.document
    return RetrievedEvidence(
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
        sparse_rank=hit.sparse_rank,
        vector_rank=hit.vector_rank,
        fused_score=hit.fused_score,
        reranker_score=reranker_score,
        final_rank=final_rank,
    )


def _validate_case(value: dict[str, Any]) -> dict[str, Any]:
    query = value.get("query")
    model = value.get("device_model")
    family = value.get("device_family")
    citations = value.get("expected_citation_ids")
    limit = value.get("limit", 6)
    if (
        not isinstance(query, str)
        or not query.strip()
        or not isinstance(model, str)
        or not model.strip()
        or family is not None
        and not isinstance(family, str)
        or not isinstance(citations, list)
        or not citations
        or not all(isinstance(item, str) and item for item in citations)
        or len(citations) != len(set(citations))
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= 20
    ):
        raise KnowledgeSearchConflict("search_evaluation_case_invalid")
    return {
        "query": query.strip(),
        "device_family": family.strip() if isinstance(family, str) else None,
        "device_model": model.strip(),
        "expected_citation_ids": sorted(citations),
        "limit": limit,
    }


def _profile_or_hidden(
    session: Session, tenant_id: str, profile_id: str, *, lock: bool = False
) -> KnowledgeSearchProfileRecord:
    statement = select(KnowledgeSearchProfileRecord).where(
        KnowledgeSearchProfileRecord.tenant_id == tenant_id,
        KnowledgeSearchProfileRecord.search_profile_id == profile_id,
    )
    if lock:
        statement = statement.with_for_update()
    record = session.scalar(statement)
    if record is None:
        raise KnowledgeSearchNotVisible
    return record


def _require_state(
    profile: KnowledgeSearchProfileRecord,
    expected_version: int,
    statuses: tuple[str, ...],
) -> None:
    if profile.state_version != expected_version or profile.status not in statuses:
        raise KnowledgeSearchConflict("search_profile_state_changed", profile.state_version)


def _require_rebuild_owner(
    session: Session,
    profile: KnowledgeSearchProfileRecord,
    rebuild_job_id: str | None,
) -> None:
    active = session.scalar(
        select(KnowledgeSearchRebuildJobRecord).where(
            KnowledgeSearchRebuildJobRecord.tenant_id == profile.tenant_id,
            KnowledgeSearchRebuildJobRecord.search_profile_id == profile.search_profile_id,
            KnowledgeSearchRebuildJobRecord.status.in_(("QUEUED", "RUNNING")),
        )
    )
    if active is None:
        if rebuild_job_id is not None:
            raise KnowledgeSearchConflict("search_profile_rebuild_not_active")
        return
    if rebuild_job_id != active.rebuild_job_id:
        raise KnowledgeSearchConflict("search_profile_rebuild_already_active", active.version)


def _profile_view(
    session: Session, tenant_id: str, record: KnowledgeSearchProfileRecord
) -> SearchProfileView:
    evaluation = session.scalar(
        select(KnowledgeSearchEvaluationRecord)
        .where(
            KnowledgeSearchEvaluationRecord.tenant_id == tenant_id,
            KnowledgeSearchEvaluationRecord.search_profile_id == record.search_profile_id,
        )
        .order_by(KnowledgeSearchEvaluationRecord.created_at.desc())
        .limit(1)
    )
    return SearchProfileView(
        search_profile_id=record.search_profile_id,
        source_index_release_id=record.source_index_release_id,
        name=record.name,
        version=record.version,
        backend=record.backend,
        status=record.status,
        is_active_shadow=record.is_active_shadow,
        index_name=record.index_name,
        schema_version=record.schema_version,
        manifest_hash=record.manifest_hash,
        embedding_model_release_id=record.embedding_model_release_id,
        embedding_manifest_hash=record.embedding_manifest_hash,
        embedding_component_model_id=record.embedding_component_model_id,
        embedding_artifact_content_hash=record.embedding_artifact_content_hash,
        embedding_dimension=record.embedding_dimension,
        document_count=record.document_count,
        metrics=dict(record.metrics_json),
        gate_results=dict(record.gate_results_json),
        failure_codes=tuple(record.failure_codes),
        created_by_subject_id=record.created_by_subject_id,
        evaluated_by_subject_id=record.evaluated_by_subject_id,
        activated_by_subject_id=record.activated_by_subject_id,
        evaluated_at=record.evaluated_at,
        activated_at=record.activated_at,
        state_version=record.state_version,
        created_at=record.created_at,
        updated_at=record.updated_at,
        evaluation=(
            SearchEvaluationView(
                evaluation_id=evaluation.evaluation_id,
                status=evaluation.status,
                policy_version=evaluation.policy_version,
                cases=tuple(evaluation.cases_json),
                metrics=dict(evaluation.metrics_json),
                gate_results=dict(evaluation.gate_results_json),
                failure_codes=tuple(evaluation.failure_codes),
                requested_by_subject_id=evaluation.requested_by_subject_id,
                completed_at=evaluation.completed_at,
                version=evaluation.version,
            )
            if evaluation is not None
            else None
        ),
    )


def _can_query_arbitrary_model(identity: IdentityContext) -> bool:
    return bool(
        identity.roles.intersection(
            {Role.DOMAIN_EXPERT, Role.MODEL_EVALUATOR, Role.TENANT_ADMIN, Role.SECURITY_AUDITOR}
        )
    )


def _embedding_binding(record: KnowledgeSearchProfileRecord) -> _EmbeddingBinding:
    return _EmbeddingBinding(
        model_release_id=record.embedding_model_release_id,
        manifest_hash=record.embedding_manifest_hash,
        component_model_id=record.embedding_component_model_id,
        artifact_content_hash=record.embedding_artifact_content_hash,
        dimension=record.embedding_dimension,
    )


def _same_embedding_binding(left: EmbeddingOutcome, right: EmbeddingOutcome) -> bool:
    return (
        left.model_release_id == right.model_release_id
        and left.manifest_hash == right.manifest_hash
        and left.component_model_id == right.component_model_id
        and left.artifact_content_hash == right.artifact_content_hash
        and left.dimension == right.dimension
    )


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "KnowledgeSearchConflict",
    "KnowledgeSearchNotVisible",
    "KnowledgeSearchProfileService",
    "SearchProfileView",
    "SearchResultView",
    "purge_search_source",
]
