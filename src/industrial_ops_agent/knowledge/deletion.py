"""KB-003 fail-closed knowledge deletion, revocation, and expiry propagation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Literal, Protocol
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.domain.json import legacy_canonical_hex_digest as _digest_json
from industrial_ops_agent.graph_rag.service import purge_graph_source
from industrial_ops_agent.graph_rag.store import GraphStore
from industrial_ops_agent.media.service import TenantObjectStore
from industrial_ops_agent.object_store.keys import tenant_object_key_from_value
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CitationAnchorRecord,
    DiagnosisRunRecord,
    FeedbackCandidateRecord,
    IdempotencyRecord,
    IndexReleaseRecord,
    KnowledgeChunkRecord,
    KnowledgeDeletionRequestRecord,
    KnowledgeDocumentVersionRecord,
    KnowledgeGraphEdgeRecord,
    KnowledgeGraphNodeRecord,
    KnowledgeIngestionJobRecord,
    KnowledgeSearchProjectionRecord,
    ModelReleaseRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.search_profiles.service import purge_search_source
from industrial_ops_agent.search_profiles.store import SearchStore

DeletionTrigger = Literal["LEGAL_DELETION", "ACCESS_REVOKED", "RETENTION_EXPIRED"]
TERMINAL_DELETION_STATUSES = frozenset({"COMPLETED", "PARTIAL", "FAILED"})
_REQUESTABLE_VERSION_STATUSES = frozenset({"DRAFT", "REVIEWED", "PUBLISHED"})
_PENDING_VERSION_STATUS: dict[str, str] = {
    "LEGAL_DELETION": "DELETION_PENDING",
    "ACCESS_REVOKED": "REVOCATION_PENDING",
    "RETENTION_EXPIRED": "EXPIRY_PENDING",
}
_TERMINAL_VERSION_STATUS: dict[str, str] = {
    "LEGAL_DELETION": "DELETED",
    "ACCESS_REVOKED": "REVOKED",
    "RETENTION_EXPIRED": "EXPIRED",
}


class KnowledgeDeletionError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class KnowledgeCacheInvalidator(Protocol):
    async def invalidate_namespace(
        self,
        context: TenantContext,
        namespace: str,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class KnowledgeDeletionView:
    deletion_id: str
    workflow_id: str
    document_id: str
    document_version_id: str
    trigger: str
    reason: str
    status: str
    target_source_checksum: str
    target_content_checksum: str
    scope: dict[str, Any]
    impact: dict[str, Any]
    verification: dict[str, Any]
    failure_reason: str | None
    requested_by_subject_id: str
    requested_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    attempt_count: int
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class KnowledgeDeletionActivityInput:
    deletion_id: str
    tenant_id: str
    workflow_id: str


class KnowledgeDeletionDispatcher(Protocol):
    async def dispatch(self, command: KnowledgeDeletionActivityInput) -> None: ...


class KnowledgeDeletionDispatchUnavailable(RuntimeError):
    """Temporal could not durably accept a deletion command."""


class KnowledgeDeletionService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def request(
        self,
        identity: IdentityContext,
        document_version_id: str,
        *,
        trigger: DeletionTrigger,
        reason: str,
        idempotency_key: str,
        request_id: str,
    ) -> KnowledgeDeletionView:
        self._require(identity, Action.REQUEST_KNOWLEDGE_DELETION, document_version_id, request_id)
        if trigger not in _PENDING_VERSION_STATUS:
            raise KnowledgeDeletionError("knowledge_deletion_trigger_invalid")
        normalized_reason = reason.strip()
        if not 8 <= len(normalized_reason) <= 1_024:
            raise KnowledgeDeletionError("knowledge_deletion_reason_invalid")
        if not 1 <= len(idempotency_key) <= 200:
            raise KnowledgeDeletionError("knowledge_deletion_idempotency_key_invalid")
        now = datetime.now(UTC)
        request_hash = _digest_json(
            {
                "document_version_id": document_version_id,
                "trigger": trigger,
                "reason": normalized_reason,
            }
        )
        storage_key = f"knowledge-deletion:{idempotency_key}"
        with self._database.transaction(identity.tenant_context) as session:
            replay = session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.tenant_id == identity.tenant_id,
                    IdempotencyRecord.subject_id == identity.subject_id,
                    IdempotencyRecord.key == storage_key,
                )
            )
            if replay is not None:
                if replay.request_hash != request_hash:
                    raise KnowledgeDeletionError("knowledge_deletion_idempotency_conflict")
                record = _deletion_or_hidden(session, identity.tenant_id, replay.result_ref)
                return _view(record)

            version = session.scalar(
                select(KnowledgeDocumentVersionRecord)
                .where(
                    KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id,
                    KnowledgeDocumentVersionRecord.document_version_id == document_version_id,
                )
                .with_for_update()
            )
            if version is None:
                raise KnowledgeDeletionError("knowledge_deletion_target_not_visible")
            if version.status not in _REQUESTABLE_VERSION_STATUSES:
                raise KnowledgeDeletionError("knowledge_deletion_target_inactive")
            if trigger == "RETENTION_EXPIRED" and (
                version.valid_to is None or _as_utc(version.valid_to) > now
            ):
                raise KnowledgeDeletionError("knowledge_deletion_target_not_expired")

            scope, impact = _discover_impact(session, identity.tenant_id, version)
            deletion_id = f"knowledge-deletion-{uuid4().hex}"
            workflow_id = f"knowledge-deletion-workflow-{deletion_id}"
            record = KnowledgeDeletionRequestRecord(
                tenant_id=identity.tenant_id,
                deletion_id=deletion_id,
                workflow_id=workflow_id,
                document_id=version.document_id,
                document_version_id=version.document_version_id,
                trigger=trigger,
                reason=normalized_reason,
                status="QUEUED",
                target_source_checksum=version.source_checksum,
                target_content_checksum=version.content_checksum,
                scope_json=scope,
                impact_json=impact,
                verification_json={
                    "postgresql": "PENDING",
                    "object_store": "PENDING",
                    "full_text_index": "PENDING",
                    "vector_index": "PENDING",
                    "tenant_cache": "PENDING",
                    "training_candidates": "PENDING",
                    "graph_index": (
                        "PENDING" if impact.get("graph_release_ids") else "NOT_CONFIGURED"
                    ),
                    "external_search": (
                        "PENDING" if impact.get("search_profile_ids") else "NOT_CONFIGURED"
                    ),
                    "data_lake": "NO_DIRECT_LINEAGE",
                    "label_platform": "NO_DIRECT_LINEAGE",
                    "published_model_review": "PENDING",
                },
                failure_reason=None,
                requested_by_subject_id=identity.subject_id,
                requested_at=now,
                started_at=None,
                completed_at=None,
                attempt_count=0,
                version=1,
                created_at=now,
                updated_at=now,
            )
            version.status = _PENDING_VERSION_STATUS[trigger]
            version.state_version += 1
            version.updated_at = now
            session.add(record)
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=deletion_id,
                    expires_at=now + timedelta(days=30),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _view(record)

    def request_expired(
        self,
        identity: IdentityContext,
        *,
        reason: str,
        request_id: str,
        limit: int = 100,
    ) -> tuple[KnowledgeDeletionView, ...]:
        self._require(
            identity,
            Action.REQUEST_KNOWLEDGE_DELETION,
            "expired-knowledge-sweep",
            request_id,
        )
        if not 1 <= limit <= 500:
            raise KnowledgeDeletionError("knowledge_deletion_page_invalid")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            ids = tuple(
                session.scalars(
                    select(KnowledgeDocumentVersionRecord.document_version_id)
                    .where(
                        KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id,
                        KnowledgeDocumentVersionRecord.status.in_(_REQUESTABLE_VERSION_STATUSES),
                        KnowledgeDocumentVersionRecord.valid_to.is_not(None),
                        KnowledgeDocumentVersionRecord.valid_to <= now,
                    )
                    .order_by(KnowledgeDocumentVersionRecord.valid_to)
                    .limit(limit)
                )
            )
        results: list[KnowledgeDeletionView] = []
        for document_version_id in ids:
            idempotency_key = "expiry-" + sha256(document_version_id.encode()).hexdigest()
            results.append(
                self.request(
                    identity,
                    document_version_id,
                    trigger="RETENTION_EXPIRED",
                    reason=reason,
                    idempotency_key=idempotency_key,
                    request_id=request_id,
                )
            )
        return tuple(results)

    def get(
        self,
        identity: IdentityContext,
        deletion_id: str,
        *,
        request_id: str,
    ) -> KnowledgeDeletionView:
        self._require(identity, Action.READ_KNOWLEDGE_DELETION, deletion_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            return _view(_deletion_or_hidden(session, identity.tenant_id, deletion_id))

    def list(
        self,
        identity: IdentityContext,
        *,
        status: str | None,
        limit: int,
        offset: int,
        request_id: str,
    ) -> tuple[tuple[KnowledgeDeletionView, ...], int]:
        self._require(
            identity,
            Action.READ_KNOWLEDGE_DELETION,
            "knowledge-deletions",
            request_id,
        )
        allowed = {"QUEUED", "RUNNING", *TERMINAL_DELETION_STATUSES}
        if status is not None and status not in allowed:
            raise KnowledgeDeletionError("knowledge_deletion_status_invalid")
        if not 1 <= limit <= 100 or offset < 0:
            raise KnowledgeDeletionError("knowledge_deletion_page_invalid")
        with self._database.transaction(identity.tenant_context) as session:
            filters = [KnowledgeDeletionRequestRecord.tenant_id == identity.tenant_id]
            if status is not None:
                filters.append(KnowledgeDeletionRequestRecord.status == status)
            total = int(
                session.scalar(
                    select(func.count()).select_from(KnowledgeDeletionRequestRecord).where(*filters)
                )
                or 0
            )
            records = tuple(
                session.scalars(
                    select(KnowledgeDeletionRequestRecord)
                    .where(*filters)
                    .order_by(
                        KnowledgeDeletionRequestRecord.requested_at.desc(),
                        KnowledgeDeletionRequestRecord.deletion_id,
                    )
                    .offset(offset)
                    .limit(limit)
                )
            )
            return tuple(_view(record) for record in records), total

    def retry(
        self,
        identity: IdentityContext,
        deletion_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> KnowledgeDeletionView:
        self._require(identity, Action.REQUEST_KNOWLEDGE_DELETION, deletion_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(KnowledgeDeletionRequestRecord)
                .where(
                    KnowledgeDeletionRequestRecord.tenant_id == identity.tenant_id,
                    KnowledgeDeletionRequestRecord.deletion_id == deletion_id,
                )
                .with_for_update()
            )
            if record is None:
                raise KnowledgeDeletionError("knowledge_deletion_not_visible")
            if record.version != expected_version or record.status not in {"PARTIAL", "FAILED"}:
                raise KnowledgeDeletionError("knowledge_deletion_state_changed")
            next_attempt = record.attempt_count + 1
            record.workflow_id = f"knowledge-deletion-workflow-{deletion_id}-attempt-{next_attempt}"
            record.status = "QUEUED"
            record.failure_reason = None
            record.completed_at = None
            record.version += 1
            record.updated_at = now
            session.flush()
            return _view(record)

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(identity.tenant_id, resource_id),
            request_id=request_id,
        )


class KnowledgeDeletionActivity:
    """Idempotently purge serving derivatives and retain only minimal audit hashes."""

    def __init__(
        self,
        database: Database,
        object_store: TenantObjectStore,
        cache: KnowledgeCacheInvalidator,
        graph_store: GraphStore | None = None,
        search_store: SearchStore | None = None,
    ) -> None:
        self._database = database
        self._object_store = object_store
        self._cache = cache
        self._graph_store = graph_store
        self._search_store = search_store

    async def execute(
        self,
        command: KnowledgeDeletionActivityInput,
    ) -> KnowledgeDeletionView:
        context = TenantContext(command.tenant_id, "knowledge-deletion-worker")
        with self._database.transaction(context) as session:
            record = _deletion_or_hidden(session, command.tenant_id, command.deletion_id)
            if record.status == "COMPLETED":
                return _view(record)
            if record.status not in {"QUEUED", "RUNNING", "PARTIAL", "FAILED"}:
                raise KnowledgeDeletionError("knowledge_deletion_state_invalid")
            now = datetime.now(UTC)
            record.status = "RUNNING"
            record.started_at = record.started_at or now
            record.attempt_count += 1
            record.version += 1
            record.updated_at = now
            jobs = tuple(
                session.scalars(
                    select(KnowledgeIngestionJobRecord).where(
                        KnowledgeIngestionJobRecord.tenant_id == command.tenant_id,
                        KnowledgeIngestionJobRecord.document_version_id
                        == record.document_version_id,
                    )
                )
            )
            object_values = tuple(
                sorted(
                    {
                        value
                        for job in jobs
                        for value in (job.quarantine_key, job.clean_key)
                        if value
                    }
                )
            )

        deleted_objects = 0
        external_failures: list[str] = []
        for value in object_values:
            try:
                await self._object_store.delete(
                    context,
                    tenant_object_key_from_value(context, value),
                )
                deleted_objects += 1
            except Exception:  # dependency details are intentionally not persisted
                external_failures.append("object_store_delete_failed")
        try:
            invalidated_cache_keys = await self._cache.invalidate_namespace(context, "knowledge")
        except Exception:
            invalidated_cache_keys = 0
            external_failures.append("tenant_cache_invalidation_failed")
        graph_external_status = "NOT_CONFIGURED"
        if self._graph_store is not None:
            try:
                await asyncio.to_thread(
                    self._graph_store.remove_document_version,
                    command.tenant_id,
                    record.document_version_id,
                )
                graph_external_status = "VERIFIED"
            except Exception:
                graph_external_status = "FAILED"
                external_failures.append("graph_store_deletion_failed")
        search_external_status = "NOT_CONFIGURED"
        if self._search_store is not None:
            try:
                await asyncio.to_thread(
                    self._search_store.remove_document_version,
                    command.tenant_id,
                    record.document_version_id,
                )
                search_external_status = "VERIFIED"
            except Exception:
                search_external_status = "FAILED"
                external_failures.append("search_store_deletion_failed")

        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            record = _deletion_or_hidden(session, command.tenant_id, command.deletion_id)
            version = session.scalar(
                select(KnowledgeDocumentVersionRecord).where(
                    KnowledgeDocumentVersionRecord.tenant_id == command.tenant_id,
                    KnowledgeDocumentVersionRecord.document_version_id
                    == record.document_version_id,
                )
            )
            if version is None:
                raise KnowledgeDeletionError("knowledge_deletion_target_not_visible")

            chunks = tuple(
                session.scalars(
                    select(KnowledgeChunkRecord).where(
                        KnowledgeChunkRecord.tenant_id == command.tenant_id,
                        KnowledgeChunkRecord.document_version_id == record.document_version_id,
                    )
                )
            )
            chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
            release_ids = sorted({chunk.release_id for chunk in chunks})
            graph_node_count, graph_edge_count, graph_release_ids = purge_graph_source(
                session,
                tenant_id=command.tenant_id,
                document_version_id=record.document_version_id,
            )
            search_projection_count, search_profile_ids = purge_search_source(
                session,
                tenant_id=command.tenant_id,
                document_version_id=record.document_version_id,
            )
            revoked_release_ids, replacement_release_ids = _replace_affected_releases(
                session,
                command.tenant_id,
                record.document_version_id,
                release_ids,
                now,
            )
            if chunk_ids:
                session.execute(
                    delete(CitationAnchorRecord).where(
                        CitationAnchorRecord.tenant_id == command.tenant_id,
                        CitationAnchorRecord.chunk_id.in_(chunk_ids),
                    )
                )
                session.execute(
                    delete(KnowledgeChunkRecord).where(
                        KnowledgeChunkRecord.tenant_id == command.tenant_id,
                        KnowledgeChunkRecord.chunk_id.in_(chunk_ids),
                    )
                )

            candidates = tuple(
                session.scalars(
                    select(FeedbackCandidateRecord).where(
                        FeedbackCandidateRecord.tenant_id == command.tenant_id
                    )
                )
            )
            revoked_candidate_ids: list[str] = []
            for candidate in candidates:
                if _lineage_references_version(
                    candidate.lineage_origin,
                    record.document_version_id,
                ):
                    candidate.status = "REVOKED"
                    candidate.eligibility_status = "INELIGIBLE"
                    candidate.allow_training = False
                    candidate.version += 1
                    candidate.updated_at = now
                    revoked_candidate_ids.append(candidate.candidate_id)

            version.extracted_text = None
            version.extraction_checksum = None
            version.extraction_metadata = None
            version.status = _TERMINAL_VERSION_STATUS[record.trigger]
            version.state_version += 1
            version.updated_at = now
            for job in session.scalars(
                select(KnowledgeIngestionJobRecord).where(
                    KnowledgeIngestionJobRecord.tenant_id == command.tenant_id,
                    KnowledgeIngestionJobRecord.document_version_id == record.document_version_id,
                )
            ):
                job.status = "PURGED"
                job.failure_reason = f"purged_by:{record.deletion_id}"
                job.version += 1
                job.updated_at = now

            discovered_scope, discovered_impact = _discover_impact(
                session,
                command.tenant_id,
                version,
                known_release_ids=release_ids,
            )
            scope = _merge_scope(record.scope_json, discovered_scope)
            impact = _merge_impact(record.impact_json, discovered_impact)
            impact["revoked_training_candidate_ids"] = sorted(
                set(impact.get("revoked_training_candidate_ids", [])) | set(revoked_candidate_ids)
            )
            impact["deleted_source_object_count"] = deleted_objects
            impact["invalidated_cache_key_count"] = invalidated_cache_keys
            impact["revoked_index_release_ids"] = revoked_release_ids
            impact["replacement_index_release_ids"] = replacement_release_ids
            impact["revoked_graph_release_ids"] = list(graph_release_ids)
            impact["deleted_graph_node_count"] = graph_node_count
            impact["deleted_graph_edge_count"] = graph_edge_count
            impact["revoked_search_profile_ids"] = list(search_profile_ids)
            impact["deleted_search_projection_count"] = search_projection_count
            published_model_ids = list(impact.get("model_release_ids", []))

            session.flush()
            remaining_chunks = int(
                session.scalar(
                    select(func.count())
                    .select_from(KnowledgeChunkRecord)
                    .where(
                        KnowledgeChunkRecord.tenant_id == command.tenant_id,
                        KnowledgeChunkRecord.document_version_id == record.document_version_id,
                    )
                )
                or 0
            )
            remaining_anchors = int(
                session.scalar(
                    select(func.count())
                    .select_from(CitationAnchorRecord)
                    .where(
                        CitationAnchorRecord.tenant_id == command.tenant_id,
                        CitationAnchorRecord.document_version_id == record.document_version_id,
                    )
                )
                or 0
            )
            remaining_search_projections = int(
                session.scalar(
                    select(func.count())
                    .select_from(KnowledgeSearchProjectionRecord)
                    .where(
                        KnowledgeSearchProjectionRecord.tenant_id == command.tenant_id,
                        KnowledgeSearchProjectionRecord.document_version_id
                        == record.document_version_id,
                    )
                )
                or 0
            )
            verification = {
                "postgresql": "VERIFIED" if version.extracted_text is None else "FAILED",
                "object_store": "VERIFIED" if not external_failures else "FAILED",
                "full_text_index": "VERIFIED" if remaining_chunks == 0 else "FAILED",
                "vector_index": "VERIFIED" if remaining_chunks == 0 else "FAILED",
                "citation_anchors": "VERIFIED" if remaining_anchors == 0 else "FAILED",
                "tenant_cache": "VERIFIED"
                if "tenant_cache_invalidation_failed" not in external_failures
                else "FAILED",
                "training_candidates": "VERIFIED",
                "graph_index": (graph_external_status if graph_release_ids else "NOT_CONFIGURED"),
                "external_search": (
                    search_external_status if search_profile_ids else "NOT_CONFIGURED"
                ),
                "data_lake": "NO_DIRECT_LINEAGE",
                "label_platform": "NO_DIRECT_LINEAGE",
                "published_model_review": "REQUIRED" if published_model_ids else "NOT_REQUIRED",
            }
            failures = sorted(set(external_failures))
            if remaining_chunks or remaining_anchors or version.extracted_text is not None:
                failures.append("database_derivative_verification_failed")
            if graph_release_ids and graph_external_status != "VERIFIED":
                failures.append("graph_derivative_verification_failed")
            if remaining_search_projections:
                failures.append("search_projection_verification_failed")
            if search_profile_ids and search_external_status != "VERIFIED":
                failures.append("search_derivative_verification_failed")
            if published_model_ids:
                failures.append("published_model_review_required")
            record.scope_json = scope
            record.impact_json = impact
            record.verification_json = verification
            record.failure_reason = ",".join(failures) or None
            record.status = "PARTIAL" if failures else "COMPLETED"
            record.completed_at = now
            record.version += 1
            record.updated_at = now
            session.flush()
            return _view(record)


def _discover_impact(
    session: Session,
    tenant_id: str,
    version: KnowledgeDocumentVersionRecord,
    *,
    known_release_ids: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    chunks = tuple(
        session.scalars(
            select(KnowledgeChunkRecord).where(
                KnowledgeChunkRecord.tenant_id == tenant_id,
                KnowledgeChunkRecord.document_version_id == version.document_version_id,
            )
        )
    )
    release_ids = sorted(set(known_release_ids or ()) | {chunk.release_id for chunk in chunks})
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    citation_count = 0
    if chunk_ids:
        citation_count = int(
            session.scalar(
                select(func.count())
                .select_from(CitationAnchorRecord)
                .where(
                    CitationAnchorRecord.tenant_id == tenant_id,
                    CitationAnchorRecord.chunk_id.in_(chunk_ids),
                )
            )
            or 0
        )
    jobs = tuple(
        session.scalars(
            select(KnowledgeIngestionJobRecord).where(
                KnowledgeIngestionJobRecord.tenant_id == tenant_id,
                KnowledgeIngestionJobRecord.document_version_id == version.document_version_id,
            )
        )
    )
    object_count = len(
        {value for job in jobs for value in (job.quarantine_key, job.clean_key) if value}
    )
    diagnosis_ids = [
        run.diagnosis_run_id
        for run in session.scalars(
            select(DiagnosisRunRecord).where(DiagnosisRunRecord.tenant_id == tenant_id)
        )
        if str(run.manifest.get("index_release_id", "")) in release_ids
    ]
    model_release_ids = [
        release.release_id
        for release in session.scalars(
            select(ModelReleaseRecord).where(ModelReleaseRecord.tenant_id == tenant_id)
        )
        if _manifest_index_release_id(release.manifest_json) in release_ids
    ]
    candidate_ids = [
        candidate.candidate_id
        for candidate in session.scalars(
            select(FeedbackCandidateRecord).where(FeedbackCandidateRecord.tenant_id == tenant_id)
        )
        if _lineage_references_version(candidate.lineage_origin, version.document_version_id)
    ]
    graph_nodes = tuple(
        session.scalars(
            select(KnowledgeGraphNodeRecord).where(
                KnowledgeGraphNodeRecord.tenant_id == tenant_id,
                KnowledgeGraphNodeRecord.document_version_id == version.document_version_id,
            )
        )
    )
    graph_node_ids = tuple(item.graph_node_id for item in graph_nodes)
    graph_edge_scope = KnowledgeGraphEdgeRecord.document_version_id == version.document_version_id
    if graph_node_ids:
        graph_edge_scope = (
            graph_edge_scope
            | KnowledgeGraphEdgeRecord.source_node_id.in_(graph_node_ids)
            | KnowledgeGraphEdgeRecord.target_node_id.in_(graph_node_ids)
        )
    graph_edges = tuple(
        session.scalars(
            select(KnowledgeGraphEdgeRecord).where(
                KnowledgeGraphEdgeRecord.tenant_id == tenant_id,
                graph_edge_scope,
            )
        )
    )
    graph_release_ids = sorted(
        {item.graph_release_id for item in graph_nodes}
        | {item.graph_release_id for item in graph_edges}
    )
    search_projections = tuple(
        session.scalars(
            select(KnowledgeSearchProjectionRecord).where(
                KnowledgeSearchProjectionRecord.tenant_id == tenant_id,
                KnowledgeSearchProjectionRecord.document_version_id == version.document_version_id,
            )
        )
    )
    search_profile_ids = sorted({item.search_profile_id for item in search_projections})
    scope = {
        "source_object_count": object_count,
        "knowledge_chunk_count": len(chunks),
        "citation_anchor_count": citation_count,
        "index_release_count": len(release_ids),
        "training_candidate_count": len(candidate_ids),
        "graph_node_count": len(graph_nodes),
        "graph_edge_count": len(graph_edges),
        "search_projection_count": len(search_projections),
    }
    impact = {
        "index_release_ids": release_ids,
        "diagnosis_run_ids": sorted(diagnosis_ids),
        "model_release_ids": sorted(model_release_ids),
        "training_candidate_ids": sorted(candidate_ids),
        "graph_release_ids": graph_release_ids,
        "search_profile_ids": search_profile_ids,
        "model_disposition": "REVIEW_REQUIRED" if model_release_ids else "NOT_AFFECTED",
        "historical_audit_policy": "HASHES_AND_IDS_RETAINED_WITHOUT_SOURCE_BODY",
    }
    return scope, impact


def _replace_affected_releases(
    session: Session,
    tenant_id: str,
    document_version_id: str,
    release_ids: list[str],
    now: datetime,
) -> tuple[list[str], list[str]]:
    """Revoke tainted immutable releases and atomically clone safe active content."""

    revoked_ids: list[str] = []
    replacement_ids: list[str] = []
    for release_id in release_ids:
        release = session.scalar(
            select(IndexReleaseRecord)
            .where(
                IndexReleaseRecord.tenant_id == tenant_id,
                IndexReleaseRecord.release_id == release_id,
            )
            .with_for_update()
        )
        if release is None:
            continue
        was_active = release.status == "PUBLISHED" and release.is_active
        release.status = "REVOKED"
        release.is_active = False
        release.updated_at = now
        revoked_ids.append(release.release_id)
        session.flush()
        if not was_active:
            continue

        remaining_chunks = tuple(
            session.scalars(
                select(KnowledgeChunkRecord)
                .where(
                    KnowledgeChunkRecord.tenant_id == tenant_id,
                    KnowledgeChunkRecord.release_id == release_id,
                    KnowledgeChunkRecord.document_version_id != document_version_id,
                )
                .order_by(
                    KnowledgeChunkRecord.document_version_id,
                    KnowledgeChunkRecord.ordinal,
                )
            )
        )
        if not remaining_chunks:
            continue
        remaining_version_ids = sorted({chunk.document_version_id for chunk in remaining_chunks})
        versions = tuple(
            session.scalars(
                select(KnowledgeDocumentVersionRecord).where(
                    KnowledgeDocumentVersionRecord.tenant_id == tenant_id,
                    KnowledgeDocumentVersionRecord.document_version_id.in_(remaining_version_ids),
                    KnowledgeDocumentVersionRecord.status == "PUBLISHED",
                )
            )
        )
        if len(versions) != len(remaining_version_ids):
            continue
        next_version = (
            int(
                session.scalar(
                    select(func.coalesce(func.max(IndexReleaseRecord.version), 0)).where(
                        IndexReleaseRecord.tenant_id == tenant_id,
                        IndexReleaseRecord.name == release.name,
                    )
                )
                or 0
            )
            + 1
        )
        anchors_by_chunk = {
            chunk.chunk_id: tuple(
                session.scalars(
                    select(CitationAnchorRecord).where(
                        CitationAnchorRecord.tenant_id == tenant_id,
                        CitationAnchorRecord.chunk_id == chunk.chunk_id,
                    )
                )
            )
            for chunk in remaining_chunks
        }
        replacement_id = f"index-release-{uuid4().hex}"
        replacement = IndexReleaseRecord(
            tenant_id=tenant_id,
            release_id=replacement_id,
            name=release.name,
            version=next_version,
            status="PUBLISHED",
            is_active=True,
            content_checksum=_retained_content_checksum(
                versions, remaining_chunks, anchors_by_chunk
            ),
            published_by="knowledge-deletion-worker",
            published_at=now,
            created_at=now,
            updated_at=now,
        )
        session.add(replacement)
        session.flush()
        for chunk in remaining_chunks:
            chunk_identity = sha256(
                (
                    f"{replacement_id}\0{chunk.document_version_id}\0"
                    f"{chunk.ordinal}\0{chunk.content_checksum}"
                ).encode()
            ).hexdigest()[:24]
            new_chunk_id = f"knowledge-chunk-{chunk_identity}"
            session.add(
                KnowledgeChunkRecord(
                    tenant_id=tenant_id,
                    chunk_id=new_chunk_id,
                    document_version_id=chunk.document_version_id,
                    release_id=replacement_id,
                    ordinal=chunk.ordinal,
                    content=chunk.content,
                    content_checksum=chunk.content_checksum,
                    embedding=list(chunk.embedding),
                    token_count=chunk.token_count,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            for anchor in anchors_by_chunk[chunk.chunk_id]:
                anchor_identity = sha256(
                    f"{new_chunk_id}\0{anchor.citation_id}".encode()
                ).hexdigest()[:24]
                session.add(
                    CitationAnchorRecord(
                        tenant_id=tenant_id,
                        citation_id=f"citation-{anchor_identity}",
                        chunk_id=new_chunk_id,
                        document_version_id=anchor.document_version_id,
                        anchor_kind=anchor.anchor_kind,
                        page_number=anchor.page_number,
                        bounding_box=anchor.bounding_box,
                        start_offset=anchor.start_offset,
                        end_offset=anchor.end_offset,
                        excerpt=anchor.excerpt,
                        excerpt_checksum=anchor.excerpt_checksum,
                        created_at=now,
                        updated_at=now,
                    )
                )
        replacement_ids.append(replacement_id)
    return sorted(revoked_ids), sorted(replacement_ids)


def _retained_content_checksum(
    versions: tuple[KnowledgeDocumentVersionRecord, ...],
    chunks: tuple[KnowledgeChunkRecord, ...],
    anchors_by_chunk: dict[str, tuple[CitationAnchorRecord, ...]],
) -> str:
    """Fingerprint copied content, without claiming a historical build algorithm.

    Document chunker metadata is not the algorithm that built the old release.
    Keep this schema distinct from ingestion; never rewrite existing checksums.
    New IDs and timestamps are intentionally excluded.
    """
    return _digest_json(
        {
            "schema": "knowledge-retained-content-v1",
            "documents": sorted(
                (
                    version.document_version_id,
                    version.content_checksum,
                    version.chunker_version,
                )
                for version in versions
            ),
            "document_fields": [
                "document_version_id", "content_checksum", "recorded_document_chunker_version"
            ],
            "chunks": sorted(
                _digest_json(
                    {
                        "document_version_id": chunk.document_version_id,
                        "ordinal": chunk.ordinal,
                        "content_checksum": chunk.content_checksum,
                        "content_sha256": sha256(chunk.content.encode()).hexdigest(),
                        "embedding": list(chunk.embedding),
                        "token_count": chunk.token_count,
                        "anchors": sorted(
                            _digest_json(
                                {
                                    "document_version_id": anchor.document_version_id,
                                    "anchor_kind": anchor.anchor_kind,
                                    "page_number": anchor.page_number,
                                    "bounding_box": anchor.bounding_box,
                                    "start_offset": anchor.start_offset,
                                    "end_offset": anchor.end_offset,
                                    "excerpt": anchor.excerpt,
                                    "excerpt_checksum": anchor.excerpt_checksum,
                                }
                            )
                            for anchor in anchors_by_chunk[chunk.chunk_id]
                        ),
                    }
                )
                for chunk in chunks
            ),
        }
    )


def _manifest_index_release_id(manifest: dict[str, Any]) -> str:
    retrieval = manifest.get("retrieval")
    if isinstance(retrieval, dict):
        return str(retrieval.get("index_release_id", ""))
    return str(manifest.get("index_release_id", ""))


def _lineage_references_version(lineage: dict[str, Any], document_version_id: str) -> bool:
    if lineage.get("document_version_id") == document_version_id:
        return True
    values = lineage.get("knowledge_document_version_ids")
    return isinstance(values, list) and document_version_id in values


def _merge_scope(original: dict[str, Any], discovered: dict[str, Any]) -> dict[str, Any]:
    merged = dict(original)
    for key, value in discovered.items():
        if isinstance(value, int):
            merged[key] = max(int(merged.get(key, 0)), value)
        else:
            merged[key] = value
    return merged


def _merge_impact(original: dict[str, Any], discovered: dict[str, Any]) -> dict[str, Any]:
    merged = dict(original)
    for key, value in discovered.items():
        if key.endswith("_ids") and isinstance(value, list):
            merged[key] = sorted(set(merged.get(key, [])) | set(value))
        elif key not in merged or value not in {"NOT_AFFECTED", "NOT_REQUIRED"}:
            merged[key] = value
    return merged


def _deletion_or_hidden(
    session: Session,
    tenant_id: str,
    deletion_id: str,
) -> KnowledgeDeletionRequestRecord:
    record = session.scalar(
        select(KnowledgeDeletionRequestRecord).where(
            KnowledgeDeletionRequestRecord.tenant_id == tenant_id,
            KnowledgeDeletionRequestRecord.deletion_id == deletion_id,
        )
    )
    if record is None:
        raise KnowledgeDeletionError("knowledge_deletion_not_visible")
    return record


def _view(record: KnowledgeDeletionRequestRecord) -> KnowledgeDeletionView:
    return KnowledgeDeletionView(
        deletion_id=record.deletion_id,
        workflow_id=record.workflow_id,
        document_id=record.document_id,
        document_version_id=record.document_version_id,
        trigger=record.trigger,
        reason=record.reason,
        status=record.status,
        target_source_checksum=record.target_source_checksum,
        target_content_checksum=record.target_content_checksum,
        scope=dict(record.scope_json),
        impact=dict(record.impact_json),
        verification=dict(record.verification_json),
        failure_reason=record.failure_reason,
        requested_by_subject_id=record.requested_by_subject_id,
        requested_at=record.requested_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        attempt_count=record.attempt_count,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
