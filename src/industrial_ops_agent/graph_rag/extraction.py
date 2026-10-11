"""Durable, source-bound GraphRAG extraction job lifecycle."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.graph_rag.service import (
    NODE_TYPES,
    RELATION_TYPES,
    KnowledgeGraphConflict,
    apply_reviewed_candidate,
)
from industrial_ops_agent.graph_rag.validation import has_key_cycle as _has_key_cycle
from industrial_ops_agent.model_gateway.context_manifest import (
    ContextReference,
    ContextTruncation,
)
from industrial_ops_agent.model_gateway.service import (
    GatewayRequest,
    GatewayResponse,
    ModelGatewayError,
    ProductionModelResolver,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CitationAnchorRecord,
    IdempotencyRecord,
    IndexReleaseRecord,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    KnowledgeDocumentVersionRecord,
    KnowledgeGraphEdgeRecord,
    KnowledgeGraphExtractionJobRecord,
    KnowledgeGraphNodeRecord,
    KnowledgeGraphReleaseRecord,
    PromptBundleRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.prompting import GRAPH_EXTRACTION_PROMPT_BUNDLE_ID
from industrial_ops_agent.prompting.registry import PromptBundleNotDeployed, default_prompt_registry

MAX_EXTRACTION_CITATIONS = 32
MAX_EXTRACTION_CONTEXT_CHARACTERS = 9_000
MAX_EXTRACTION_EXCERPT_CHARACTERS = 1_800
MAX_EXTRACTION_NODES = 64
MAX_EXTRACTION_EDGES = 128
MAX_EXTRACTION_BUNDLE_BYTES = 128 * 1024
MAX_EXTRACTION_METADATA_BYTES = 8_192
GRAPH_EXTRACTION_PROFILE = "industrial-graphrag-causal-v1"
GRAPH_EXTRACTION_SCHEMA_VERSION = "industrial_graphrag_candidate_bundle_v1"
TERMINAL_EXTRACTION_STATUSES = frozenset({"ACCEPTED", "REJECTED"})
_NODE_KEY = re.compile(r"^[a-z0-9][a-z0-9_.:-]{1,127}$")
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "command",
        "code",
        "endpoint",
        "executable",
        "script",
        "shell",
        "tool",
        "tool_call",
        "uri",
        "url",
    }
)
_FORBIDDEN_TEXT = re.compile(
    r"(?:https?://|file://|javascript:|\bcurl\b|\bwget\b|\bexecute\b|"
    r"\brun\s+(?:this\s+)?command\b)",
    re.IGNORECASE,
)


class GraphExtractionNotVisible(Exception):
    pass


class GraphExtractionConflict(RuntimeError):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


class GraphExtractionDispatchUnavailable(RuntimeError):
    """Temporal could not durably accept an extraction command."""


@dataclass(frozen=True, slots=True)
class GraphExtractionRuntimeBinding:
    extraction_profile: str
    model_alias: str
    model_release_id: str
    model_manifest_hash: str
    prompt_bundle_id: str
    prompt_bundle_hash: str
    response_schema_version: str


class GraphExtractionBindingResolver(Protocol):
    def resolve(self, identity: IdentityContext) -> GraphExtractionRuntimeBinding: ...


@dataclass(frozen=True, slots=True)
class GraphExtractionActivityInput:
    extraction_job_id: str
    workflow_id: str
    identity: IdentityContext
    request_id: str


class GraphExtractionDispatcher(Protocol):
    async def dispatch(self, command: GraphExtractionActivityInput) -> None: ...


class GraphExtractionModelGateway(Protocol):
    async def complete(
        self,
        context: TenantContext,
        request: GatewayRequest,
    ) -> GatewayResponse: ...


class ProductionGraphExtractionBindingResolver:
    """Resolve one image-bound Prompt and exact production model for a new job."""

    def __init__(
        self,
        database: Database,
        model_resolver: ProductionModelResolver,
        *,
        model_alias: str,
    ) -> None:
        self._database = database
        self._model_resolver = model_resolver
        self._model_alias = model_alias

    def resolve(self, identity: IdentityContext) -> GraphExtractionRuntimeBinding:
        try:
            definition = default_prompt_registry().get(GRAPH_EXTRACTION_PROMPT_BUNDLE_ID)
        except PromptBundleNotDeployed as exc:
            raise GraphExtractionConflict("graph_extraction_prompt_not_deployed") from exc
        if (
            definition.output_schema_name != GRAPH_EXTRACTION_SCHEMA_VERSION
            or self._model_alias not in definition.compatible_model_aliases
        ):
            raise GraphExtractionConflict("graph_extraction_prompt_binding_invalid")
        try:
            model = self._model_resolver.resolve(
                identity.tenant_context,
                self._model_alias,
            )
        except ModelGatewayError as exc:
            raise GraphExtractionConflict("graph_extraction_model_unavailable") from exc
        with self._database.transaction(identity.tenant_context) as session:
            governed = session.scalar(
                select(PromptBundleRecord).where(
                    PromptBundleRecord.tenant_id == identity.tenant_id,
                    PromptBundleRecord.prompt_bundle_id == definition.prompt_bundle_id,
                )
            )
            if governed is None or governed.status != "APPROVED":
                raise GraphExtractionConflict("graph_extraction_prompt_not_approved")
            if (
                governed.content_hash != definition.content_hash
                or governed.output_schema_name != definition.output_schema_name
                or self._model_alias not in governed.compatible_model_aliases_json
            ):
                raise GraphExtractionConflict("graph_extraction_prompt_hash_mismatch")
        return GraphExtractionRuntimeBinding(
            extraction_profile=GRAPH_EXTRACTION_PROFILE,
            model_alias=model.alias,
            model_release_id=model.release_id,
            model_manifest_hash=model.manifest_hash,
            prompt_bundle_id=definition.prompt_bundle_id,
            prompt_bundle_hash=definition.content_hash,
            response_schema_version=definition.output_schema_name,
        )


class GraphExtractionActivity:
    """Run one source-bound inference and persist only a review candidate."""

    def __init__(
        self,
        database: Database,
        model_gateway: GraphExtractionModelGateway,
        *,
        timeout_seconds: float,
    ) -> None:
        self._database = database
        self._model_gateway = model_gateway
        self._timeout_seconds = timeout_seconds

    async def execute(self, command: GraphExtractionActivityInput) -> GraphExtractionView:
        current = self._begin(command)
        if current.status != "RUNNING":
            return current
        try:
            source = self._load_source(command, current)
            request = _gateway_request(current, source, self._timeout_seconds)
            response = await self._model_gateway.complete(
                command.identity.tenant_context,
                request,
            )
            if (
                response.inference_request_id != current.inference_request_id
                or response.resolved_release_id != current.model_release_id
                or response.manifest_hash != current.model_manifest_hash
            ):
                raise _ExtractionFailure(
                    "MODEL_BINDING_FAILED",
                    "graph_extraction_model_binding_changed",
                )
            candidate = _validated_candidate(
                response.content,
                selected_citation_ids=frozenset(current.citation_ids),
                existing_node_keys=source.existing_node_keys,
                existing_causes=source.existing_causes,
            )
            return self._finish(command, current, candidate, response)
        except _ExtractionFailure as exc:
            return self._fail(command, stage=exc.stage, failure_code=exc.failure_code)
        except (GraphExtractionConflict, GraphExtractionNotVisible):
            return self._fail(
                command,
                stage="SOURCE_VALIDATION_FAILED",
                failure_code="graph_extraction_source_binding_changed",
            )
        except ModelGatewayError as exc:
            failure_code = (
                "graph_extraction_inference_retryable"
                if exc.reason
                in {
                    "model_quota_unavailable",
                    "model_request_rate_exceeded",
                    "model_daily_token_quota_exceeded",
                    "production_model_unavailable",
                }
                else "graph_extraction_inference_rejected"
            )
            return self._fail(
                command,
                stage="MODEL_INFERENCE_FAILED",
                failure_code=failure_code,
            )
        except Exception:
            return self._fail(
                command,
                stage="DEPENDENCY_FAILED",
                failure_code="graph_extraction_dependency_unavailable",
            )

    def _begin(self, command: GraphExtractionActivityInput) -> GraphExtractionView:
        now = datetime.now(UTC)
        with self._database.transaction(command.identity.tenant_context) as session:
            record = _job_or_hidden(
                session,
                command.identity.tenant_id,
                command.extraction_job_id,
                lock=True,
            )
            if record.workflow_id != command.workflow_id:
                raise GraphExtractionConflict("graph_extraction_workflow_binding_changed")
            if record.status in {
                "REVIEW_PENDING",
                "ACCEPTED",
                "REJECTED",
                "FAILED",
            }:
                return _view(record)
            if record.status == "QUEUED":
                record.status = "RUNNING"
                record.stage = "SOURCE_VALIDATION"
                record.attempt_count += 1
                record.started_at = now
                record.completed_at = None
                record.failure_code = None
                record.version += 1
                session.flush()
            elif record.status != "RUNNING":
                raise GraphExtractionConflict(
                    "graph_extraction_activity_state_invalid",
                    record.version,
                )
            return _view(record)

    def _load_source(
        self,
        command: GraphExtractionActivityInput,
        current: GraphExtractionView,
    ) -> _ExtractionSource:
        with self._database.transaction(command.identity.tenant_context) as session:
            record = _job_or_hidden(
                session,
                command.identity.tenant_id,
                command.extraction_job_id,
            )
            if record.status != "RUNNING" or record.workflow_id != command.workflow_id:
                raise GraphExtractionConflict("graph_extraction_activity_state_changed")
            return _validated_source(session, command.identity, record, current)

    def _finish(
        self,
        command: GraphExtractionActivityInput,
        current: GraphExtractionView,
        candidate: dict[str, Any],
        response: GatewayResponse,
    ) -> GraphExtractionView:
        now = datetime.now(UTC)
        with self._database.transaction(command.identity.tenant_context) as session:
            record = _job_or_hidden(
                session,
                command.identity.tenant_id,
                command.extraction_job_id,
                lock=True,
            )
            if record.status == "REVIEW_PENDING":
                return _view(record)
            if record.status != "RUNNING" or record.workflow_id != command.workflow_id:
                raise GraphExtractionConflict(
                    "graph_extraction_activity_state_changed",
                    record.version,
                )
            _validated_source(session, command.identity, record, current)
            record.candidate_bundle_json = candidate
            record.result_hash = _digest(candidate)
            record.context_hash = response.context_hash
            record.status = "REVIEW_PENDING"
            record.stage = "REVIEW_PENDING"
            record.failure_code = None
            record.completed_at = now
            record.version += 1
            session.flush()
            return _view(record)

    def _fail(
        self,
        command: GraphExtractionActivityInput,
        *,
        stage: str,
        failure_code: str,
    ) -> GraphExtractionView:
        now = datetime.now(UTC)
        with self._database.transaction(command.identity.tenant_context) as session:
            record = _job_or_hidden(
                session,
                command.identity.tenant_id,
                command.extraction_job_id,
                lock=True,
            )
            if record.status in {
                "REVIEW_PENDING",
                "ACCEPTED",
                "REJECTED",
                "FAILED",
            }:
                return _view(record)
            record.status = "FAILED"
            record.stage = stage[:64]
            record.failure_code = failure_code[:128]
            record.candidate_bundle_json = None
            record.result_hash = None
            record.context_hash = None
            record.completed_at = now
            record.version += 1
            session.flush()
            return _view(record)


@dataclass(frozen=True, slots=True)
class GraphExtractionView:
    extraction_job_id: str
    graph_release_id: str
    source_index_release_id: str
    expected_graph_version: int
    citation_ids: tuple[str, ...]
    citation_bindings: tuple[dict[str, Any], ...]
    source_binding_hash: str
    extraction_profile: str
    workflow_id: str
    inference_request_id: str
    model_alias: str
    model_release_id: str
    model_manifest_hash: str
    prompt_bundle_id: str
    prompt_bundle_hash: str
    response_schema_version: str
    status: str
    stage: str
    attempt_count: int
    candidate_bundle: dict[str, Any] | None
    result_hash: str | None
    context_hash: str | None
    failure_code: str | None
    requested_by_subject_id: str
    reviewed_by_subject_id: str | None
    review_decision: str | None
    review_reason: str | None
    started_at: datetime | None
    completed_at: datetime | None
    reviewed_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    dispatch_required: bool = False

    @property
    def legal_actions(self) -> tuple[str, ...]:
        if self.status == "FAILED":
            return ("RETRY",)
        if self.status == "REVIEW_PENDING":
            return ("REVIEW_ACCEPT", "REVIEW_REJECT")
        return ()


@dataclass(frozen=True, slots=True)
class _ExtractionSource:
    citations: tuple[dict[str, str], ...]
    context_references: tuple[ContextReference, ...]
    truncations: tuple[ContextTruncation, ...]
    existing_node_keys: frozenset[str]
    existing_causes: tuple[tuple[str, str], ...]


class _ExtractionFailure(RuntimeError):
    def __init__(self, stage: str, failure_code: str) -> None:
        self.stage = stage
        self.failure_code = failure_code
        super().__init__(failure_code)


class GraphExtractionService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def request(
        self,
        identity: IdentityContext,
        graph_release_id: str,
        *,
        citation_ids: tuple[str, ...],
        expected_graph_version: int,
        idempotency_key: str,
        binding: GraphExtractionRuntimeBinding,
        request_id: str,
    ) -> GraphExtractionView:
        self._require(
            identity,
            Action.MANAGE_KNOWLEDGE_GRAPH,
            graph_release_id,
            request_id,
        )
        normalized_citations = _normalize_citation_ids(citation_ids)
        if not 8 <= len(idempotency_key) <= 255:
            raise GraphExtractionConflict("graph_extraction_idempotency_invalid")
        binding_document = _binding_document(binding)
        request_hash = _digest(
            {
                "graph_release_id": graph_release_id,
                "expected_graph_version": expected_graph_version,
                "citation_ids": normalized_citations,
                "binding": binding_document,
            }
        )
        storage_key = f"graph-extraction:{idempotency_key}"
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
                    raise GraphExtractionConflict("graph_extraction_idempotency_conflict")
                return _view(
                    _job_or_hidden(session, identity.tenant_id, str(replay.result_ref))
                )

            release = session.scalar(
                select(KnowledgeGraphReleaseRecord)
                .where(
                    KnowledgeGraphReleaseRecord.tenant_id == identity.tenant_id,
                    KnowledgeGraphReleaseRecord.graph_release_id == graph_release_id,
                )
                .with_for_update()
            )
            if release is None:
                raise GraphExtractionNotVisible
            if (
                release.status != "DRAFT"
                or release.state_version != expected_graph_version
            ):
                raise GraphExtractionConflict(
                    "graph_extraction_graph_state_changed", release.state_version
                )
            source_release = session.scalar(
                select(IndexReleaseRecord).where(
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.release_id == release.source_index_release_id,
                    IndexReleaseRecord.status == "PUBLISHED",
                )
            )
            if source_release is None:
                raise GraphExtractionConflict("graph_extraction_source_release_unpublished")
            citation_bindings = tuple(
                _citation_binding(
                    session,
                    identity,
                    release.source_index_release_id,
                    citation_id,
                    now,
                )
                for citation_id in normalized_citations
            )
            source_binding_hash = _digest(
                {
                    "source_index_release_id": release.source_index_release_id,
                    "source_release_checksum": source_release.content_checksum,
                    "citations": citation_bindings,
                }
            )
            extraction_job_id = f"graph-extraction-{uuid4().hex}"
            workflow_id = f"graph-extraction-workflow-{uuid4().hex}"
            inference_request_id = f"inference-{extraction_job_id}"
            record = KnowledgeGraphExtractionJobRecord(
                extraction_job_id=extraction_job_id,
                tenant_id=identity.tenant_id,
                graph_release_id=graph_release_id,
                source_index_release_id=release.source_index_release_id,
                expected_graph_version=expected_graph_version,
                citation_ids_json=list(normalized_citations),
                citation_bindings_json=[dict(item) for item in citation_bindings],
                source_binding_hash=source_binding_hash,
                extraction_profile=str(binding.extraction_profile),
                workflow_id=workflow_id,
                inference_request_id=inference_request_id,
                model_alias=str(binding.model_alias),
                model_release_id=str(binding.model_release_id),
                model_manifest_hash=str(binding.model_manifest_hash),
                prompt_bundle_id=str(binding.prompt_bundle_id),
                prompt_bundle_hash=str(binding.prompt_bundle_hash),
                response_schema_version=str(binding.response_schema_version),
                status="QUEUED",
                stage="QUEUED",
                attempt_count=0,
                candidate_bundle_json=None,
                result_hash=None,
                context_hash=None,
                failure_code=None,
                requested_by_subject_id=identity.subject_id,
                reviewed_by_subject_id=None,
                review_decision=None,
                review_reason=None,
                started_at=None,
                completed_at=None,
                reviewed_at=None,
                version=1,
            )
            session.add(record)
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=extraction_job_id,
                    expires_at=now + timedelta(days=30),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _view(record, dispatch_required=True)

    def list(
        self,
        identity: IdentityContext,
        graph_release_id: str,
        *,
        request_id: str,
    ) -> tuple[GraphExtractionView, ...]:
        self._require(
            identity,
            Action.READ_KNOWLEDGE_GRAPH,
            graph_release_id,
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            release = session.scalar(
                select(KnowledgeGraphReleaseRecord.graph_release_id).where(
                    KnowledgeGraphReleaseRecord.tenant_id == identity.tenant_id,
                    KnowledgeGraphReleaseRecord.graph_release_id == graph_release_id,
                )
            )
            if release is None:
                raise GraphExtractionNotVisible
            records = session.scalars(
                select(KnowledgeGraphExtractionJobRecord)
                .where(
                    KnowledgeGraphExtractionJobRecord.tenant_id == identity.tenant_id,
                    KnowledgeGraphExtractionJobRecord.graph_release_id == graph_release_id,
                )
                .order_by(
                    KnowledgeGraphExtractionJobRecord.created_at.desc(),
                    KnowledgeGraphExtractionJobRecord.extraction_job_id,
                )
            )
            return tuple(_view(item) for item in records)

    def get(
        self,
        identity: IdentityContext,
        extraction_job_id: str,
        *,
        request_id: str,
    ) -> GraphExtractionView:
        self._require(
            identity,
            Action.READ_KNOWLEDGE_GRAPH,
            extraction_job_id,
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            return _view(_job_or_hidden(session, identity.tenant_id, extraction_job_id))

    def retry(
        self,
        identity: IdentityContext,
        extraction_job_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> GraphExtractionView:
        self._require(
            identity,
            Action.MANAGE_KNOWLEDGE_GRAPH,
            extraction_job_id,
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(KnowledgeGraphExtractionJobRecord)
                .where(
                    KnowledgeGraphExtractionJobRecord.tenant_id == identity.tenant_id,
                    KnowledgeGraphExtractionJobRecord.extraction_job_id == extraction_job_id,
                )
                .with_for_update()
            )
            if record is None:
                raise GraphExtractionNotVisible
            if record.version != expected_version or record.status != "FAILED":
                raise GraphExtractionConflict(
                    "graph_extraction_retry_state_invalid", record.version
                )
            if record.failure_code not in {
                "graph_extraction_dispatch_unavailable",
                "graph_extraction_dependency_unavailable",
                "graph_extraction_inference_retryable",
            }:
                raise GraphExtractionConflict(
                    "graph_extraction_failure_not_retryable", record.version
                )
            record.status = "QUEUED"
            record.stage = "QUEUED"
            record.failure_code = None
            record.completed_at = None
            record.version += 1
            session.flush()
            return _view(record, dispatch_required=True)

    def mark_dispatch_failed(
        self,
        identity: IdentityContext,
        extraction_job_id: str,
        *,
        workflow_id: str,
    ) -> GraphExtractionView:
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(KnowledgeGraphExtractionJobRecord)
                .where(
                    KnowledgeGraphExtractionJobRecord.tenant_id == identity.tenant_id,
                    KnowledgeGraphExtractionJobRecord.extraction_job_id == extraction_job_id,
                )
                .with_for_update()
            )
            if record is None:
                raise GraphExtractionNotVisible
            if record.workflow_id != workflow_id:
                raise GraphExtractionConflict("graph_extraction_workflow_binding_changed")
            if record.status == "FAILED":
                return _view(record)
            if record.status != "QUEUED":
                raise GraphExtractionConflict(
                    "graph_extraction_dispatch_state_invalid", record.version
                )
            record.status = "FAILED"
            record.stage = "DISPATCH_FAILED"
            record.failure_code = "graph_extraction_dispatch_unavailable"
            record.completed_at = datetime.now(UTC)
            record.version += 1
            session.flush()
            return _view(record)

    def review(
        self,
        identity: IdentityContext,
        extraction_job_id: str,
        *,
        decision: str,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> GraphExtractionView:
        normalized_decision = decision.strip().upper()
        normalized_reason = reason.strip()
        if normalized_decision not in {"ACCEPT", "REJECT"} or not 8 <= len(
            normalized_reason
        ) <= 1_000:
            raise GraphExtractionConflict("graph_extraction_review_input_invalid")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = _job_or_hidden(
                session,
                identity.tenant_id,
                extraction_job_id,
                lock=True,
            )
            if record.requested_by_subject_id == identity.subject_id:
                raise GraphExtractionConflict(
                    "graph_extraction_review_separation_required",
                    record.version,
                )
            self._require(
                identity,
                Action.EVALUATE_KNOWLEDGE_GRAPH,
                extraction_job_id,
                request_id,
            )
            if record.version != expected_version:
                raise GraphExtractionConflict(
                    "graph_extraction_review_state_changed",
                    record.version,
                )
            if record.status in TERMINAL_EXTRACTION_STATUSES:
                if record.review_decision == normalized_decision:
                    return _view(record)
                raise GraphExtractionConflict(
                    "graph_extraction_review_already_completed",
                    record.version,
                )
            if record.status != "REVIEW_PENDING":
                raise GraphExtractionConflict(
                    "graph_extraction_review_state_changed",
                    record.version,
                )
            release = session.scalar(
                select(KnowledgeGraphReleaseRecord)
                .where(
                    KnowledgeGraphReleaseRecord.tenant_id == identity.tenant_id,
                    KnowledgeGraphReleaseRecord.graph_release_id == record.graph_release_id,
                )
                .with_for_update()
            )
            if release is None:
                raise GraphExtractionNotVisible
            current = _view(record)
            try:
                source = _validated_source(session, identity, record, current)
            except (GraphExtractionConflict, GraphExtractionNotVisible) as exc:
                raise GraphExtractionConflict(
                    "graph_extraction_source_binding_changed",
                    record.version,
                ) from exc
            candidate_value = record.candidate_bundle_json
            if (
                candidate_value is None
                or record.result_hash is None
                or _digest(candidate_value) != record.result_hash
            ):
                raise GraphExtractionConflict(
                    "graph_extraction_candidate_binding_changed",
                    record.version,
                )
            try:
                candidate = _validated_candidate(
                    candidate_value,
                    selected_citation_ids=frozenset(record.citation_ids_json),
                    existing_node_keys=source.existing_node_keys,
                    existing_causes=source.existing_causes,
                )
            except _ExtractionFailure as exc:
                raise GraphExtractionConflict(
                    "graph_extraction_candidate_conflict",
                    record.version,
                ) from exc
            if _digest(candidate) != record.result_hash:
                raise GraphExtractionConflict(
                    "graph_extraction_candidate_binding_changed",
                    record.version,
                )
            if normalized_decision == "ACCEPT":
                try:
                    apply_reviewed_candidate(session, identity, release, candidate)
                except KnowledgeGraphConflict as exc:
                    raise GraphExtractionConflict(exc.reason, record.version) from exc
                record.status = "ACCEPTED"
                record.stage = "ACCEPTED_DRAFT"
            else:
                record.status = "REJECTED"
                record.stage = "REJECTED"
            record.reviewed_by_subject_id = identity.subject_id
            record.review_decision = normalized_decision
            record.review_reason = normalized_reason
            record.reviewed_at = now
            record.completed_at = now
            record.version += 1
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


def _normalize_citation_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(sorted(item.strip() for item in values if item.strip()))
    if (
        not normalized
        or len(normalized) > MAX_EXTRACTION_CITATIONS
        or len(normalized) != len(values)
        or len(set(normalized)) != len(normalized)
        or any(len(item) > 128 for item in normalized)
    ):
        raise GraphExtractionConflict("graph_extraction_citations_invalid")
    return normalized


def _binding_document(binding: GraphExtractionRuntimeBinding) -> dict[str, str]:
    document = {
        "extraction_profile": str(binding.extraction_profile),
        "model_alias": str(binding.model_alias),
        "model_release_id": str(binding.model_release_id),
        "model_manifest_hash": str(binding.model_manifest_hash),
        "prompt_bundle_id": str(binding.prompt_bundle_id),
        "prompt_bundle_hash": str(binding.prompt_bundle_hash),
        "response_schema_version": str(binding.response_schema_version),
    }
    if any(not item or len(item) > 255 for item in document.values()):
        raise GraphExtractionConflict("graph_extraction_runtime_binding_invalid")
    return document


def _citation_binding(
    session: Session,
    identity: IdentityContext,
    source_release_id: str,
    citation_id: str,
    now: datetime,
) -> dict[str, Any]:
    row = session.execute(
        select(
            CitationAnchorRecord,
            KnowledgeChunkRecord,
            KnowledgeDocumentVersionRecord,
            KnowledgeDocumentRecord,
        )
        .join(KnowledgeChunkRecord, KnowledgeChunkRecord.chunk_id == CitationAnchorRecord.chunk_id)
        .join(
            KnowledgeDocumentVersionRecord,
            KnowledgeDocumentVersionRecord.document_version_id
            == CitationAnchorRecord.document_version_id,
        )
        .join(
            KnowledgeDocumentRecord,
            KnowledgeDocumentRecord.document_id == KnowledgeDocumentVersionRecord.document_id,
        )
        .where(
            CitationAnchorRecord.tenant_id == identity.tenant_id,
            CitationAnchorRecord.citation_id == citation_id,
            KnowledgeChunkRecord.tenant_id == identity.tenant_id,
            KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id,
            KnowledgeDocumentRecord.tenant_id == identity.tenant_id,
        )
    ).one_or_none()
    if row is None:
        raise GraphExtractionNotVisible
    anchor, chunk, version, document = row
    if chunk.release_id != source_release_id:
        raise GraphExtractionConflict("graph_extraction_citation_release_mismatch")
    if version.status != "PUBLISHED":
        raise GraphExtractionConflict("graph_extraction_citation_unpublished")
    roles = {role.value for role in identity.roles}
    acl_allowed = (
        not version.acl_subject_ids
        and not version.acl_roles
        or identity.subject_id in version.acl_subject_ids
        or bool(roles.intersection(version.acl_roles))
    )
    valid_from = _as_utc(version.valid_from)
    valid_to = _as_utc(version.valid_to) if version.valid_to is not None else None
    if (
        not acl_allowed
        or valid_from > now
        or valid_to is not None
        and valid_to <= now
        or document.classification == "restricted"
        and not roles.intersection(version.acl_roles)
    ):
        raise GraphExtractionNotVisible
    return {
        "citation_id": anchor.citation_id,
        "chunk_id": chunk.chunk_id,
        "document_version_id": version.document_version_id,
        "document_id": document.document_id,
        "excerpt_checksum": anchor.excerpt_checksum,
        "chunk_content_checksum": chunk.content_checksum,
        "document_content_checksum": version.content_checksum,
        "source_checksum": version.source_checksum,
    }


def _validated_source(
    session: Session,
    identity: IdentityContext,
    record: KnowledgeGraphExtractionJobRecord,
    current: GraphExtractionView,
) -> _ExtractionSource:
    now = datetime.now(UTC)
    release = session.scalar(
        select(KnowledgeGraphReleaseRecord).where(
            KnowledgeGraphReleaseRecord.tenant_id == identity.tenant_id,
            KnowledgeGraphReleaseRecord.graph_release_id == record.graph_release_id,
        )
    )
    source_release = session.scalar(
        select(IndexReleaseRecord).where(
            IndexReleaseRecord.tenant_id == identity.tenant_id,
            IndexReleaseRecord.release_id == record.source_index_release_id,
            IndexReleaseRecord.status == "PUBLISHED",
        )
    )
    if (
        release is None
        or source_release is None
        or release.status != "DRAFT"
        or release.state_version != record.expected_graph_version
        or release.source_index_release_id != record.source_index_release_id
        or tuple(record.citation_ids_json) != current.citation_ids
    ):
        raise GraphExtractionConflict("graph_extraction_source_binding_changed")
    bindings = tuple(
        _citation_binding(
            session,
            identity,
            record.source_index_release_id,
            citation_id,
            now,
        )
        for citation_id in record.citation_ids_json
    )
    current_hash = _digest(
        {
            "source_index_release_id": record.source_index_release_id,
            "source_release_checksum": source_release.content_checksum,
            "citations": bindings,
        }
    )
    if (
        current_hash != record.source_binding_hash
        or [dict(item) for item in bindings] != record.citation_bindings_json
    ):
        raise GraphExtractionConflict("graph_extraction_source_binding_changed")

    excerpts = tuple(
        _citation_excerpt(
            session,
            identity.tenant_id,
            record.source_index_release_id,
            binding,
        )
        for binding in bindings
    )
    citations, truncations = _bounded_citations(bindings, excerpts)
    nodes = tuple(
        session.scalars(
            select(KnowledgeGraphNodeRecord).where(
                KnowledgeGraphNodeRecord.tenant_id == identity.tenant_id,
                KnowledgeGraphNodeRecord.graph_release_id == record.graph_release_id,
            )
        )
    )
    node_keys = {item.graph_node_id: item.node_key for item in nodes}
    causes = tuple(
        sorted(
            (node_keys[item.source_node_id], node_keys[item.target_node_id])
            for item in session.scalars(
                select(KnowledgeGraphEdgeRecord).where(
                    KnowledgeGraphEdgeRecord.tenant_id == identity.tenant_id,
                    KnowledgeGraphEdgeRecord.graph_release_id == record.graph_release_id,
                    KnowledgeGraphEdgeRecord.relation_type == "CAUSES",
                )
            )
            if item.source_node_id in node_keys and item.target_node_id in node_keys
        )
    )
    return _ExtractionSource(
        citations=citations,
        context_references=tuple(
            ContextReference(
                reference_id=str(binding["citation_id"]),
                content_hash=_digest(binding),
            )
            for binding in bindings
        ),
        truncations=truncations,
        existing_node_keys=frozenset(node_keys.values()),
        existing_causes=causes,
    )


def _citation_excerpt(
    session: Session,
    tenant_id: str,
    source_release_id: str,
    binding: dict[str, Any],
) -> str:
    row = session.execute(
        select(CitationAnchorRecord, KnowledgeChunkRecord)
        .join(KnowledgeChunkRecord, KnowledgeChunkRecord.chunk_id == CitationAnchorRecord.chunk_id)
        .where(
            CitationAnchorRecord.tenant_id == tenant_id,
            CitationAnchorRecord.citation_id == binding["citation_id"],
            CitationAnchorRecord.chunk_id == binding["chunk_id"],
            KnowledgeChunkRecord.tenant_id == tenant_id,
            KnowledgeChunkRecord.release_id == source_release_id,
            KnowledgeChunkRecord.document_version_id == binding["document_version_id"],
        )
    ).one_or_none()
    if row is None:
        raise GraphExtractionConflict("graph_extraction_source_binding_changed")
    anchor, chunk = row
    if (
        anchor.excerpt_checksum != binding["excerpt_checksum"]
        or chunk.content_checksum != binding["chunk_content_checksum"]
        or not anchor.excerpt.strip()
    ):
        raise GraphExtractionConflict("graph_extraction_source_binding_changed")
    return anchor.excerpt.strip()


def _bounded_citations(
    bindings: tuple[dict[str, Any], ...],
    excerpts: tuple[str, ...],
) -> tuple[tuple[dict[str, str], ...], tuple[ContextTruncation, ...]]:
    remaining = MAX_EXTRACTION_CONTEXT_CHARACTERS
    citations: list[dict[str, str]] = []
    truncations: list[ContextTruncation] = []
    for index, (binding, excerpt) in enumerate(zip(bindings, excerpts, strict=True)):
        remaining_items = len(bindings) - index
        allowance = min(
            MAX_EXTRACTION_EXCERPT_CHARACTERS,
            max(1, remaining // remaining_items),
        )
        included = excerpt[:allowance]
        remaining -= len(included)
        citation_id = str(binding["citation_id"])
        citations.append(
            {
                "citation_id": citation_id,
                "document_version_id": str(binding["document_version_id"]),
                "chunk_id": str(binding["chunk_id"]),
                "excerpt": included,
            }
        )
        if len(included) < len(excerpt):
            truncations.append(
                ContextTruncation(
                    source_type="EVIDENCE",
                    source_id=citation_id,
                    reason="CHARACTER_LIMIT",
                    original_units=len(excerpt),
                    included_units=len(included),
                )
            )
    return tuple(citations), tuple(truncations)


def _gateway_request(
    job: GraphExtractionView,
    source: _ExtractionSource,
    timeout_seconds: float,
) -> GatewayRequest:
    if (
        job.extraction_profile != GRAPH_EXTRACTION_PROFILE
        or job.response_schema_version != GRAPH_EXTRACTION_SCHEMA_VERSION
    ):
        raise _ExtractionFailure(
            "MODEL_BINDING_FAILED",
            "graph_extraction_runtime_binding_changed",
        )
    try:
        prompt = default_prompt_registry().get(job.prompt_bundle_id)
    except PromptBundleNotDeployed as exc:
        raise _ExtractionFailure(
            "PROMPT_BINDING_FAILED",
            "graph_extraction_prompt_not_deployed",
        ) from exc
    if (
        prompt.prompt_bundle_id != GRAPH_EXTRACTION_PROMPT_BUNDLE_ID
        or prompt.content_hash != job.prompt_bundle_hash
        or prompt.output_schema_name != job.response_schema_version
        or job.model_alias not in prompt.compatible_model_aliases
    ):
        raise _ExtractionFailure(
            "PROMPT_BINDING_FAILED",
            "graph_extraction_prompt_binding_changed",
        )
    existing_keys = sorted(source.existing_node_keys)[:256]
    user_context = json.dumps(
        {
            "authorized_citations": list(source.citations),
            "existing_graph_node_keys": existing_keys,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return GatewayRequest(
        inference_request_id=job.inference_request_id,
        model_alias=job.model_alias,
        required_release_id=job.model_release_id,
        subject_id=job.requested_by_subject_id,
        trace_id=f"graph-extraction:{job.extraction_job_id}",
        request_class="GRAPH_CAUSAL_EXTRACTION",
        data_classification="CONFIDENTIAL",
        messages=(
            {"role": "system", "content": prompt.render_system()},
            {"role": "user", "content": user_context},
        ),
        response_schema_name=job.response_schema_version,
        response_schema=_candidate_response_schema(job.citation_ids),
        deadline=datetime.now(UTC) + timedelta(seconds=timeout_seconds),
        max_output_tokens=4_096,
        temperature=0.0,
        prompt_bundle_id=job.prompt_bundle_id,
        prompt_bundle_hash=job.prompt_bundle_hash,
        index_release_id=job.source_index_release_id,
        context_evidence=source.context_references,
        context_token_budget=4_096,
        context_truncated_items=source.truncations,
    )


def _candidate_response_schema(citation_ids: tuple[str, ...]) -> dict[str, Any]:
    scalar_schema: dict[str, Any] = {
        "type": ["string", "number", "integer", "boolean", "null"]
    }
    metadata_schema: dict[str, Any] = {
        "type": "object",
        "maxProperties": 64,
        "additionalProperties": scalar_schema,
    }
    node_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["node_key", "node_type", "display_name", "citation_id", "metadata"],
        "properties": {
            "node_key": {"type": "string", "pattern": _NODE_KEY.pattern},
            "node_type": {"type": "string", "enum": sorted(NODE_TYPES)},
            "display_name": {"type": "string", "minLength": 2, "maxLength": 255},
            "citation_id": {"type": "string", "enum": list(citation_ids)},
            "metadata": metadata_schema,
        },
    }
    edge_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source_node_key",
            "target_node_key",
            "relation_type",
            "confidence",
            "citation_id",
            "metadata",
        ],
        "properties": {
            "source_node_key": {"type": "string", "pattern": _NODE_KEY.pattern},
            "target_node_key": {"type": "string", "pattern": _NODE_KEY.pattern},
            "relation_type": {"type": "string", "enum": sorted(RELATION_TYPES)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "citation_id": {"type": "string", "enum": list(citation_ids)},
            "metadata": metadata_schema,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["nodes", "edges"],
        "properties": {
            "nodes": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_EXTRACTION_NODES,
                "items": node_schema,
            },
            "edges": {
                "type": "array",
                "maxItems": MAX_EXTRACTION_EDGES,
                "items": edge_schema,
            },
        },
    }


def _validated_candidate(
    value: object,
    *,
    selected_citation_ids: frozenset[str],
    existing_node_keys: frozenset[str],
    existing_causes: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"nodes", "edges"}:
        raise _candidate_failure()
    nodes_value = value.get("nodes")
    edges_value = value.get("edges")
    if (
        not isinstance(nodes_value, list)
        or not 1 <= len(nodes_value) <= MAX_EXTRACTION_NODES
        or not isinstance(edges_value, list)
        or len(edges_value) > MAX_EXTRACTION_EDGES
    ):
        raise _candidate_failure()

    nodes: list[dict[str, Any]] = []
    candidate_keys: set[str] = set()
    node_fields = {"node_key", "node_type", "display_name", "citation_id", "metadata"}
    for item in nodes_value:
        if not isinstance(item, dict) or set(item) != node_fields:
            raise _candidate_failure()
        key = item.get("node_key")
        node_type = item.get("node_type")
        display_name = item.get("display_name")
        citation_id = item.get("citation_id")
        metadata = item.get("metadata")
        if (
            not isinstance(key, str)
            or not _NODE_KEY.fullmatch(key)
            or key in candidate_keys
            or key in existing_node_keys
            or node_type not in NODE_TYPES
            or not isinstance(display_name, str)
            or not 2 <= len(display_name.strip()) <= 255
            or _FORBIDDEN_TEXT.search(display_name) is not None
            or citation_id not in selected_citation_ids
            or not isinstance(metadata, dict)
        ):
            raise _candidate_failure()
        normalized_metadata = _validated_candidate_metadata(metadata)
        candidate_keys.add(key)
        nodes.append(
            {
                "node_key": key,
                "node_type": node_type,
                "display_name": display_name.strip(),
                "citation_id": citation_id,
                "metadata": normalized_metadata,
            }
        )

    all_keys = candidate_keys | set(existing_node_keys)
    edges: list[dict[str, Any]] = []
    edge_identities: set[tuple[str, str, str]] = set()
    causes = list(existing_causes)
    edge_fields = {
        "source_node_key",
        "target_node_key",
        "relation_type",
        "confidence",
        "citation_id",
        "metadata",
    }
    for item in edges_value:
        if not isinstance(item, dict) or set(item) != edge_fields:
            raise _candidate_failure()
        source_key = item.get("source_node_key")
        target_key = item.get("target_node_key")
        relation_type = item.get("relation_type")
        confidence = item.get("confidence")
        citation_id = item.get("citation_id")
        metadata = item.get("metadata")
        identity = (str(source_key), str(target_key), str(relation_type))
        if (
            not isinstance(source_key, str)
            or source_key not in all_keys
            or not isinstance(target_key, str)
            or target_key not in all_keys
            or source_key == target_key
            or relation_type not in RELATION_TYPES
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or citation_id not in selected_citation_ids
            or not isinstance(metadata, dict)
            or identity in edge_identities
        ):
            raise _candidate_failure()
        normalized_metadata = _validated_candidate_metadata(metadata)
        edge_identities.add(identity)
        if relation_type == "CAUSES":
            causes.append((source_key, target_key))
        edges.append(
            {
                "source_node_key": source_key,
                "target_node_key": target_key,
                "relation_type": relation_type,
                "confidence": float(confidence),
                "citation_id": citation_id,
                "metadata": normalized_metadata,
            }
        )
    if _has_key_cycle(all_keys, causes):
        raise _candidate_failure()
    candidate = {"nodes": nodes, "edges": edges}
    try:
        encoded = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise _candidate_failure() from exc
    if len(encoded) > MAX_EXTRACTION_BUNDLE_BYTES:
        raise _candidate_failure()
    return candidate


def _validated_candidate_metadata(value: dict[str, Any]) -> dict[str, Any]:
    if len(value) > 64:
        raise _candidate_failure()
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or key.strip().lower() in _FORBIDDEN_METADATA_KEYS
            or not isinstance(item, (str, int, float, bool))
            and item is not None
            or isinstance(item, float)
            and not math.isfinite(item)
            or isinstance(item, str)
            and _FORBIDDEN_TEXT.search(item) is not None
        ):
            raise _candidate_failure()
        normalized[key] = item
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise _candidate_failure() from exc
    if len(encoded) > MAX_EXTRACTION_METADATA_BYTES:
        raise _candidate_failure()
    return normalized



def _candidate_failure() -> _ExtractionFailure:
    return _ExtractionFailure(
        "CANDIDATE_VALIDATION_FAILED",
        "graph_extraction_candidate_invalid",
    )


def _job_or_hidden(
    session: Session,
    tenant_id: str,
    extraction_job_id: str,
    *,
    lock: bool = False,
) -> KnowledgeGraphExtractionJobRecord:
    statement = select(KnowledgeGraphExtractionJobRecord).where(
            KnowledgeGraphExtractionJobRecord.tenant_id == tenant_id,
            KnowledgeGraphExtractionJobRecord.extraction_job_id == extraction_job_id,
        )
    if lock:
        statement = statement.with_for_update()
    record = session.scalar(statement)
    if record is None:
        raise GraphExtractionNotVisible
    return record


def _view(
    record: KnowledgeGraphExtractionJobRecord,
    *,
    dispatch_required: bool = False,
) -> GraphExtractionView:
    return GraphExtractionView(
        extraction_job_id=record.extraction_job_id,
        graph_release_id=record.graph_release_id,
        source_index_release_id=record.source_index_release_id,
        expected_graph_version=record.expected_graph_version,
        citation_ids=tuple(record.citation_ids_json),
        citation_bindings=tuple(dict(item) for item in record.citation_bindings_json),
        source_binding_hash=record.source_binding_hash,
        extraction_profile=record.extraction_profile,
        workflow_id=record.workflow_id,
        inference_request_id=record.inference_request_id,
        model_alias=record.model_alias,
        model_release_id=record.model_release_id,
        model_manifest_hash=record.model_manifest_hash,
        prompt_bundle_id=record.prompt_bundle_id,
        prompt_bundle_hash=record.prompt_bundle_hash,
        response_schema_version=record.response_schema_version,
        status=record.status,
        stage=record.stage,
        attempt_count=record.attempt_count,
        candidate_bundle=(
            dict(record.candidate_bundle_json)
            if record.candidate_bundle_json is not None
            else None
        ),
        result_hash=record.result_hash,
        context_hash=record.context_hash,
        failure_code=record.failure_code,
        requested_by_subject_id=record.requested_by_subject_id,
        reviewed_by_subject_id=record.reviewed_by_subject_id,
        review_decision=record.review_decision,
        review_reason=record.review_reason,
        started_at=_optional_utc(record.started_at),
        completed_at=_optional_utc(record.completed_at),
        reviewed_at=_optional_utc(record.reviewed_at),
        version=record.version,
        created_at=_as_utc(record.created_at),
        updated_at=_as_utc(record.updated_at),
        dispatch_required=dispatch_required,
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{sha256(encoded.encode()).hexdigest()}"



def _optional_utc(value: datetime | None) -> datetime | None:
    return _as_utc(value) if value is not None else None
