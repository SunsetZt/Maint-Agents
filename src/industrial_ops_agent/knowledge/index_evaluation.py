"""KB-002 candidate index evaluation and publication evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import isfinite
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import func, select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.knowledge.models import RetrievalQuery
from industrial_ops_agent.knowledge.retrieval import (
    HybridRetriever,
    KnowledgeReleaseUnavailable,
    RetrievalAuthorizationInvariantViolation,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CitationAnchorRecord,
    IdempotencyRecord,
    IndexReleaseRecord,
    KnowledgeChunkRecord,
    KnowledgeDeletionRequestRecord,
    KnowledgeDocumentRecord,
    KnowledgeDocumentVersionRecord,
    KnowledgeIndexEvaluationRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext

POLICY_VERSION = "knowledge-index-gates-v2"
EVALUATOR_VERSION = "hybrid-retrieval-evaluator-v2"
TERMINAL_EVALUATION_STATUSES = frozenset({"PASSED", "FAILED"})


class KnowledgeIndexEvaluationError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class KnowledgeIndexEvaluationView:
    evaluation_id: str
    release_id: str
    workflow_id: str
    status: str
    policy_version: str
    evaluator_version: str
    metrics: dict[str, Any]
    gate_results: dict[str, bool]
    failure_codes: tuple[str, ...]
    requested_by_subject_id: str
    started_at: datetime | None
    completed_at: datetime | None
    attempt_count: int
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class KnowledgeIndexEvaluationActivityInput:
    evaluation_id: str
    tenant_id: str
    workflow_id: str


class KnowledgeIndexEvaluationDispatcher(Protocol):
    async def dispatch(self, command: KnowledgeIndexEvaluationActivityInput) -> None: ...


class KnowledgeIndexEvaluationDispatchUnavailable(RuntimeError):
    """Temporal could not durably accept an index evaluation command."""


class KnowledgeIndexEvaluationService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def start(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> KnowledgeIndexEvaluationView:
        self._authorizer.require(
            identity,
            Action.EVALUATE_KNOWLEDGE_INDEX,
            ResourceContext(identity.tenant_id, release_id),
            request_id=request_id,
        )
        if not 1 <= len(idempotency_key) <= 200:
            raise KnowledgeIndexEvaluationError("knowledge_index_evaluation_idempotency_invalid")
        request_hash = sha256(f"{release_id}\0{POLICY_VERSION}".encode()).hexdigest()
        storage_key = f"knowledge-index-evaluation:{idempotency_key}"
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
                if replay.request_hash != request_hash:
                    raise KnowledgeIndexEvaluationError(
                        "knowledge_index_evaluation_idempotency_conflict"
                    )
                record = _evaluation_or_hidden(session, identity.tenant_id, str(replay.result_ref))
                return _view(record)

            release = session.scalar(
                select(IndexReleaseRecord)
                .where(
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.release_id == release_id,
                )
                .with_for_update()
            )
            if release is None:
                raise KnowledgeIndexEvaluationError("knowledge_index_release_not_visible")
            if release.status != "EVALUATION_PENDING":
                raise KnowledgeIndexEvaluationError("knowledge_index_evaluation_state_invalid")
            evaluation_id = f"knowledge-index-evaluation-{uuid4().hex}"
            workflow_id = f"knowledge-index-evaluation-workflow-{evaluation_id}"
            record = KnowledgeIndexEvaluationRecord(
                tenant_id=identity.tenant_id,
                evaluation_id=evaluation_id,
                release_id=release_id,
                workflow_id=workflow_id,
                status="QUEUED",
                policy_version=POLICY_VERSION,
                evaluator_version=EVALUATOR_VERSION,
                metrics_json={},
                gate_results_json={},
                failure_codes=[],
                requested_by_subject_id=identity.subject_id,
                started_at=None,
                completed_at=None,
                attempt_count=0,
                version=1,
                created_at=now,
                updated_at=now,
            )
            release.status = "EVALUATING"
            release.updated_at = now
            session.add(record)
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=evaluation_id,
                    expires_at=now + timedelta(days=30),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _view(record)

    def get(
        self,
        identity: IdentityContext,
        evaluation_id: str,
        *,
        request_id: str,
    ) -> KnowledgeIndexEvaluationView:
        self._authorizer.require(
            identity,
            Action.EVALUATE_KNOWLEDGE_INDEX,
            ResourceContext(identity.tenant_id, evaluation_id),
            request_id=request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            return _view(_evaluation_or_hidden(session, identity.tenant_id, evaluation_id))


class KnowledgeIndexEvaluationActivity:
    def __init__(self, database: Database) -> None:
        self._database = database
        self._retriever = HybridRetriever(database)

    async def execute(
        self, command: KnowledgeIndexEvaluationActivityInput
    ) -> KnowledgeIndexEvaluationView:
        context = TenantContext(command.tenant_id, "knowledge-index-evaluator")
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            record = _evaluation_or_hidden(session, command.tenant_id, command.evaluation_id)
            if record.workflow_id != command.workflow_id:
                raise KnowledgeIndexEvaluationError("knowledge_index_evaluation_workflow_stale")
            if record.status in TERMINAL_EVALUATION_STATUSES:
                return _view(record)
            release = session.scalar(
                select(IndexReleaseRecord).where(
                    IndexReleaseRecord.tenant_id == command.tenant_id,
                    IndexReleaseRecord.release_id == record.release_id,
                )
            )
            if release is None or release.status != "EVALUATING":
                raise KnowledgeIndexEvaluationError("knowledge_index_evaluation_release_invalid")
            record.status = "RUNNING"
            record.started_at = record.started_at or now
            record.attempt_count += 1
            record.version += 1
            record.updated_at = now

        try:
            metrics, gates = self._evaluate(command.tenant_id, record.release_id)
        except RetrievalAuthorizationInvariantViolation as exc:
            metrics = {"unauthorized_candidate_count": exc.candidate_count}
            gates = {"pre_recall_authorization": False}
        failure_codes = sorted(name for name, passed in gates.items() if not passed)
        completed_at = datetime.now(UTC)
        with self._database.transaction(context) as session:
            record = _evaluation_or_hidden(session, command.tenant_id, command.evaluation_id)
            release = session.scalar(
                select(IndexReleaseRecord)
                .where(
                    IndexReleaseRecord.tenant_id == command.tenant_id,
                    IndexReleaseRecord.release_id == record.release_id,
                )
                .with_for_update()
            )
            if release is None or release.status != "EVALUATING":
                raise KnowledgeIndexEvaluationError("knowledge_index_evaluation_release_invalid")
            record.metrics_json = metrics
            record.gate_results_json = gates
            record.failure_codes = failure_codes
            record.status = "FAILED" if failure_codes else "PASSED"
            record.completed_at = completed_at
            record.version += 1
            record.updated_at = completed_at
            release.status = "REJECTED" if failure_codes else "CANDIDATE"
            release.updated_at = completed_at
            session.flush()
            return _view(record)

    def _evaluate(self, tenant_id: str, release_id: str) -> tuple[dict[str, Any], dict[str, bool]]:
        now = datetime.now(UTC)
        context = TenantContext(tenant_id, "knowledge-index-evaluator")
        with self._database.transaction(context) as session:
            rows = list(
                session.execute(
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
                        KnowledgeDocumentRecord.document_id
                        == KnowledgeDocumentVersionRecord.document_id,
                    )
                    .outerjoin(
                        CitationAnchorRecord,
                        CitationAnchorRecord.chunk_id == KnowledgeChunkRecord.chunk_id,
                    )
                    .where(
                        KnowledgeChunkRecord.tenant_id == tenant_id,
                        KnowledgeChunkRecord.release_id == release_id,
                        KnowledgeDocumentVersionRecord.tenant_id == tenant_id,
                        KnowledgeDocumentRecord.tenant_id == tenant_id,
                    )
                    .order_by(KnowledgeChunkRecord.chunk_id)
                )
            )
            version_ids = {row[1].document_version_id for row in rows}
            pending_deletions = (
                int(
                    session.scalar(
                        select(func.count())
                        .select_from(KnowledgeDeletionRequestRecord)
                        .where(
                            KnowledgeDeletionRequestRecord.tenant_id == tenant_id,
                            KnowledgeDeletionRequestRecord.document_version_id.in_(version_ids),
                            KnowledgeDeletionRequestRecord.status.in_(
                                {"QUEUED", "RUNNING", "PARTIAL"}
                            ),
                        )
                    )
                    or 0
                )
                if version_ids
                else 0
            )

        chunks = [row[0] for row in rows]
        anchors = [row[3] for row in rows]
        versions = {row[1].document_version_id: row[1] for row in rows}
        documents = {row[2].document_id: row[2] for row in rows}
        structure_ok = bool(chunks) and len({chunk.chunk_id for chunk in chunks}) == len(chunks)
        citation_ok = bool(chunks) and all(
            anchor is not None
            and anchor.document_version_id == chunk.document_version_id
            and anchor.excerpt == chunk.content
            and anchor.excerpt_checksum == chunk.content_checksum
            for chunk, anchor in zip(chunks, anchors, strict=True)
        )
        embedding_ok = bool(chunks) and all(
            len(chunk.embedding) == 16 and all(isfinite(float(value)) for value in chunk.embedding)
            for chunk in chunks
        )
        validity_ok = bool(versions) and all(
            version.status in {"REVIEWED", "PUBLISHED"}
            and _utc(version.valid_from) <= now
            and (version.valid_to is None or _utc(version.valid_to) > now)
            for version in versions.values()
        )

        first_by_document: dict[str, tuple[Any, Any, Any]] = {}
        for chunk, version, document, _anchor in rows:
            first_by_document.setdefault(document.document_id, (chunk, version, document))
        recall_hits = 0
        acl_leaks = 0
        acl_cases = 0
        device_leaks = 0
        device_cases = 0
        unauthorized_candidates = 0
        for document_id, (chunk, version, _document) in first_by_document.items():
            roles = frozenset(version.acl_roles or {"domain_expert"})
            subject = (
                version.acl_subject_ids[0] if version.acl_subject_ids else "evaluation-authorized"
            )
            family = version.device_families[0] if version.device_families else None
            model = version.device_models[0] if version.device_models else "evaluation-device"
            retrieval = self._retriever.evaluate_candidate_with_trace(
                RetrievalQuery(
                    tenant_id,
                    subject,
                    roles,
                    release_id,
                    family,
                    model,
                    chunk.content,
                    now,
                    3,
                )
            )
            results = retrieval.evidence
            unauthorized_candidates += retrieval.guard.unauthorized_candidate_count
            if any(item.document_id == document_id for item in results):
                recall_hits += 1
            if version.acl_subject_ids or version.acl_roles:
                acl_cases += 1
                denied_retrieval = self._retriever.evaluate_candidate_with_trace(
                    RetrievalQuery(
                        tenant_id,
                        "evaluation-unauthorized",
                        frozenset({"evaluation_unauthorized_role"}),
                        release_id,
                        family,
                        model,
                        chunk.content,
                        now,
                        10,
                    )
                )
                denied = denied_retrieval.evidence
                unauthorized_candidates += denied_retrieval.guard.unauthorized_candidate_count
                acl_leaks += sum(item.document_id == document_id for item in denied)
            if version.device_models or version.device_families:
                device_cases += 1
                denied_retrieval = self._retriever.evaluate_candidate_with_trace(
                    RetrievalQuery(
                        tenant_id,
                        subject,
                        roles,
                        release_id,
                        "evaluation-wrong-family",
                        "evaluation-wrong-model",
                        chunk.content,
                        now,
                        10,
                    )
                )
                denied = denied_retrieval.evidence
                unauthorized_candidates += denied_retrieval.guard.unauthorized_candidate_count
                device_leaks += sum(item.document_id == document_id for item in denied)

        tenant_isolation = False
        foreign_tenant_id = (
            "evaluation-other-tenant"
            if tenant_id == "evaluation-foreign-tenant"
            else "evaluation-foreign-tenant"
        )
        try:
            self._retriever.evaluate_candidate(
                RetrievalQuery(
                    foreign_tenant_id,
                    "evaluation-foreign",
                    frozenset({"domain_expert"}),
                    release_id,
                    None,
                    "evaluation-device",
                    "tenant isolation probe",
                    now,
                    1,
                )
            )
        except KnowledgeReleaseUnavailable:
            tenant_isolation = True

        document_count = len(documents)
        recall = recall_hits / document_count if document_count else 0.0
        gates = {
            "structure_integrity": structure_ok,
            "citation_integrity": citation_ok,
            "embedding_integrity": embedding_ok,
            "retrieval_recall": document_count > 0 and recall >= 1.0,
            "acl_isolation": acl_leaks == 0,
            "device_scope_isolation": device_leaks == 0,
            "validity_window": validity_ok,
            "tenant_isolation": tenant_isolation,
            "pre_recall_authorization": unauthorized_candidates == 0,
            "deletion_propagation": pending_deletions == 0,
        }
        metrics: dict[str, Any] = {
            "document_count": document_count,
            "chunk_count": len(chunks),
            "citation_count": sum(anchor is not None for anchor in anchors),
            "retrieval_case_count": document_count,
            "recall_at_3": round(recall, 6),
            "acl_probe_count": acl_cases,
            "unauthorized_result_count": acl_leaks,
            "device_scope_probe_count": device_cases,
            "device_scope_leak_count": device_leaks,
            "unauthorized_candidate_count": unauthorized_candidates,
            "pending_deletion_count": pending_deletions,
        }
        return metrics, gates


def _evaluation_or_hidden(session: Any, tenant_id: str, evaluation_id: str) -> Any:
    record = session.scalar(
        select(KnowledgeIndexEvaluationRecord).where(
            KnowledgeIndexEvaluationRecord.tenant_id == tenant_id,
            KnowledgeIndexEvaluationRecord.evaluation_id == evaluation_id,
        )
    )
    if record is None:
        raise KnowledgeIndexEvaluationError("knowledge_index_evaluation_not_visible")
    return record


def _view(record: KnowledgeIndexEvaluationRecord) -> KnowledgeIndexEvaluationView:
    return KnowledgeIndexEvaluationView(
        evaluation_id=record.evaluation_id,
        release_id=record.release_id,
        workflow_id=record.workflow_id,
        status=record.status,
        policy_version=record.policy_version,
        evaluator_version=record.evaluator_version,
        metrics=dict(record.metrics_json),
        gate_results=dict(record.gate_results_json),
        failure_codes=tuple(record.failure_codes),
        requested_by_subject_id=record.requested_by_subject_id,
        started_at=_utc(record.started_at) if record.started_at else None,
        completed_at=_utc(record.completed_at) if record.completed_at else None,
        attempt_count=record.attempt_count,
        version=record.version,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
    )


__all__ = [
    "KnowledgeIndexEvaluationActivity",
    "KnowledgeIndexEvaluationActivityInput",
    "KnowledgeIndexEvaluationDispatcher",
    "KnowledgeIndexEvaluationDispatchUnavailable",
    "KnowledgeIndexEvaluationError",
    "KnowledgeIndexEvaluationService",
    "KnowledgeIndexEvaluationView",
]
