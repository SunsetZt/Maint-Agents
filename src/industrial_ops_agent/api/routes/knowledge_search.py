"""Governed OpenSearch profile, A/B evaluation, and shadow-query API."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_knowledge_search_rebuild_dispatcher,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.search_profiles.embedding import (
    EmbeddingUnavailable,
    KnowledgeEmbedder,
)
from industrial_ops_agent.search_profiles.rebuild import (
    SearchProfileRebuildActivityInput,
    SearchProfileRebuildDispatcher,
    SearchProfileRebuildDispatchUnavailable,
    SearchProfileRebuildService,
    SearchProfileRebuildView,
)
from industrial_ops_agent.search_profiles.reranker import KnowledgeReranker
from industrial_ops_agent.search_profiles.service import (
    KnowledgeSearchConflict,
    KnowledgeSearchNotVisible,
    KnowledgeSearchProfileService,
    SearchProfileView,
    SearchResultView,
)
from industrial_ops_agent.search_profiles.store import SearchStore, SearchStoreUnavailable

router = APIRouter(prefix="/knowledge-search-profiles", tags=["knowledge-search"])


class CreateSearchProfileBody(BaseModel):
    source_index_release_id: str = Field(min_length=3, max_length=128)
    name: str = Field(min_length=2, max_length=128)


class SearchEvaluationCaseBody(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    device_family: str | None = Field(default=None, max_length=128)
    device_model: str = Field(min_length=1, max_length=128)
    expected_citation_ids: list[str] = Field(min_length=1, max_length=50)
    limit: int = Field(default=6, ge=1, le=20)


class EvaluateSearchProfileBody(BaseModel):
    cases: list[SearchEvaluationCaseBody] = Field(min_length=1, max_length=100)


class SearchQueryBody(BaseModel):
    profile_name: str = Field(min_length=2, max_length=128)
    query_text: str = Field(min_length=1, max_length=2000)
    device_family: str | None = Field(default=None, max_length=128)
    device_model: str = Field(min_length=1, max_length=128)
    limit: int = Field(default=6, ge=1, le=20)


class SearchEvaluationResponse(BaseModel):
    evaluation_id: str
    status: str
    policy_version: str
    cases: list[dict[str, Any]]
    metrics: dict[str, Any]
    gate_results: dict[str, bool]
    failure_codes: list[str]
    requested_by_subject_id: str
    completed_at: datetime | None
    version: int


class SearchProfileResponse(BaseModel):
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
    failure_codes: list[str]
    created_by_subject_id: str
    evaluated_by_subject_id: str | None
    activated_by_subject_id: str | None
    evaluated_at: datetime | None
    activated_at: datetime | None
    state_version: int
    created_at: datetime
    updated_at: datetime
    evaluation: SearchEvaluationResponse | None
    legal_actions: list[
        Literal["SYNC", "EVALUATE", "ACTIVATE_SHADOW", "DEACTIVATE_SHADOW", "QUERY"]
    ]


class SearchProfileEnvelope(BaseModel):
    data: SearchProfileResponse
    meta: dict[str, str]


class SearchProfileListEnvelope(BaseModel):
    data: list[SearchProfileResponse]
    meta: dict[str, str | int]


class SearchProfileRebuildResponse(BaseModel):
    rebuild_job_id: str
    search_profile_id: str
    workflow_id: str
    expected_profile_version: int
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]
    stage: str
    progress_percent: int
    completed_items: int
    total_items: int
    attempt_count: int
    requested_by_subject_id: str
    failure_code: str | None
    started_at: datetime | None
    completed_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    legal_actions: list[Literal["RETRY"]]


class SearchProfileRebuildEnvelope(BaseModel):
    data: SearchProfileRebuildResponse
    meta: dict[str, str]


class SearchProfileRebuildListEnvelope(BaseModel):
    data: list[SearchProfileRebuildResponse]
    meta: dict[str, str | int]


class SearchEvidenceResponse(BaseModel):
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
    device_model: str
    sparse_rank: int | None
    vector_rank: int | None
    fused_score: float
    reranker_score: float | None
    final_rank: int | None


class RerankTraceResponse(BaseModel):
    status: Literal["NOT_CONFIGURED", "APPLIED", "FALLBACK"]
    candidate_count: int
    returned_count: int
    model_release_id: str | None
    manifest_hash: str | None
    component_model_id: str | None
    artifact_content_hash: str | None
    latency_ms: float | None
    fallback_reason: str | None


class EmbeddingTraceResponse(BaseModel):
    status: Literal["LEGACY", "APPLIED", "FALLBACK_SPARSE"]
    dimension: int | None
    model_release_id: str | None
    manifest_hash: str | None
    component_model_id: str | None
    artifact_content_hash: str | None
    latency_ms: float | None
    fallback_reason: str | None


class SearchResultResponse(BaseModel):
    profile_id: str
    source_index_release_id: str
    backend: str
    guard_stage: str
    embedding: EmbeddingTraceResponse
    rerank: RerankTraceResponse
    evidence: list[SearchEvidenceResponse]


class SearchResultEnvelope(BaseModel):
    data: SearchResultResponse
    meta: dict[str, str | int]


@router.get("", response_model=SearchProfileListEnvelope)
async def list_profiles(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> SearchProfileListEnvelope:
    try:
        items = await asyncio.to_thread(
            _service(request, database, authorizer).list,
            identity,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "knowledge_search_forbidden", "authorization") from exc
    return SearchProfileListEnvelope(
        data=[_profile_response(item) for item in items],
        meta={"request_id": request.state.request_id, "total": len(items)},
    )


@router.post("", response_model=SearchProfileEnvelope, status_code=201)
async def create_profile(
    body: CreateSearchProfileBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=255)],
) -> SearchProfileEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer).create,
            identity,
            **body.model_dump(),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeSearchNotVisible, KnowledgeSearchConflict) as exc:
        raise _command_error(exc) from exc
    return _envelope(item, request.state.request_id)


@router.get("/rebuilds", response_model=SearchProfileRebuildListEnvelope)
async def list_rebuilds(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    search_profile_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> SearchProfileRebuildListEnvelope:
    try:
        items = await asyncio.to_thread(
            SearchProfileRebuildService(database, authorizer).list,
            identity,
            search_profile_id=search_profile_id,
            limit=limit,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeSearchNotVisible, KnowledgeSearchConflict) as exc:
        raise _command_error(exc) from exc
    return SearchProfileRebuildListEnvelope(
        data=[_rebuild_response(item) for item in items],
        meta={"request_id": request.state.request_id, "total": len(items)},
    )


@router.post(
    "/{search_profile_id}/rebuilds",
    response_model=SearchProfileRebuildEnvelope,
    status_code=202,
)
async def request_rebuild(
    search_profile_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[
        SearchProfileRebuildDispatcher,
        Depends(get_knowledge_search_rebuild_dispatcher),
    ],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=255)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> SearchProfileRebuildEnvelope:
    service = SearchProfileRebuildService(database, authorizer)
    try:
        item = await asyncio.to_thread(
            service.request,
            identity,
            search_profile_id,
            expected_profile_version=_version(if_match),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
        if item.status == "QUEUED":
            try:
                await dispatcher.dispatch(
                    _rebuild_command(item, identity, request.state.request_id)
                )
            except SearchProfileRebuildDispatchUnavailable:
                await asyncio.to_thread(
                    service.mark_dispatch_failed,
                    identity,
                    item.rebuild_job_id,
                    workflow_id=item.workflow_id,
                )
                raise
    except (
        AuthorizationDenied,
        KnowledgeSearchNotVisible,
        KnowledgeSearchConflict,
        SearchProfileRebuildDispatchUnavailable,
    ) as exc:
        raise _command_error(exc) from exc
    return _rebuild_envelope(item, request.state.request_id)


@router.get("/rebuilds/{rebuild_job_id}", response_model=SearchProfileRebuildEnvelope)
async def get_rebuild(
    rebuild_job_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> SearchProfileRebuildEnvelope:
    try:
        item = await asyncio.to_thread(
            SearchProfileRebuildService(database, authorizer).get,
            identity,
            rebuild_job_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeSearchNotVisible) as exc:
        raise _command_error(exc) from exc
    return _rebuild_envelope(item, request.state.request_id)


@router.post(
    "/rebuilds/{rebuild_job_id}/retry",
    response_model=SearchProfileRebuildEnvelope,
    status_code=202,
)
async def retry_rebuild(
    rebuild_job_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[
        SearchProfileRebuildDispatcher,
        Depends(get_knowledge_search_rebuild_dispatcher),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> SearchProfileRebuildEnvelope:
    service = SearchProfileRebuildService(database, authorizer)
    try:
        item = await asyncio.to_thread(
            service.retry,
            identity,
            rebuild_job_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
        try:
            await dispatcher.dispatch(_rebuild_command(item, identity, request.state.request_id))
        except SearchProfileRebuildDispatchUnavailable:
            await asyncio.to_thread(
                service.mark_dispatch_failed,
                identity,
                item.rebuild_job_id,
                workflow_id=item.workflow_id,
            )
            raise
    except (
        AuthorizationDenied,
        KnowledgeSearchNotVisible,
        KnowledgeSearchConflict,
        SearchProfileRebuildDispatchUnavailable,
    ) as exc:
        raise _command_error(exc) from exc
    return _rebuild_envelope(item, request.state.request_id)


@router.get("/{search_profile_id}", response_model=SearchProfileEnvelope)
async def get_profile(
    search_profile_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> SearchProfileEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer).get,
            identity,
            search_profile_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeSearchNotVisible) as exc:
        raise _command_error(exc) from exc
    return _envelope(item, request.state.request_id)


@router.post("/{search_profile_id}/sync", response_model=SearchProfileEnvelope)
async def sync_profile(
    search_profile_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> SearchProfileEnvelope:
    return await _mutate(
        request,
        _service(request, database, authorizer).sync,
        identity,
        search_profile_id,
        _version(if_match),
    )


@router.post("/{search_profile_id}/evaluations", response_model=SearchProfileEnvelope)
async def evaluate_profile(
    search_profile_id: str,
    body: EvaluateSearchProfileBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> SearchProfileEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer).evaluate,
            identity,
            search_profile_id,
            cases=tuple(item.model_dump() for item in body.cases),
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        KnowledgeSearchNotVisible,
        KnowledgeSearchConflict,
        SearchStoreUnavailable,
        EmbeddingUnavailable,
    ) as exc:
        raise _command_error(exc) from exc
    return _envelope(item, request.state.request_id)


@router.post("/{search_profile_id}/activate-shadow", response_model=SearchProfileEnvelope)
async def activate_shadow(
    search_profile_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> SearchProfileEnvelope:
    return await _mutate(
        request,
        _service(request, database, authorizer).activate_shadow,
        identity,
        search_profile_id,
        _version(if_match),
    )


@router.post("/{search_profile_id}/deactivate-shadow", response_model=SearchProfileEnvelope)
async def deactivate_shadow(
    search_profile_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> SearchProfileEnvelope:
    return await _mutate(
        request,
        _service(request, database, authorizer).deactivate_shadow,
        identity,
        search_profile_id,
        _version(if_match),
    )


@router.post("/query", response_model=SearchResultEnvelope)
async def query_profile(
    body: SearchQueryBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> SearchResultEnvelope:
    try:
        result = await asyncio.to_thread(
            _service(request, database, authorizer).query,
            identity,
            **body.model_dump(),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        KnowledgeSearchNotVisible,
        KnowledgeSearchConflict,
        SearchStoreUnavailable,
        EmbeddingUnavailable,
    ) as exc:
        raise _command_error(exc) from exc
    return _result_envelope(result, request.state.request_id)


async def _mutate(
    request: Request,
    command: Any,
    identity: IdentityContext,
    profile_id: str,
    expected_version: int,
) -> SearchProfileEnvelope:
    try:
        item = await asyncio.to_thread(
            command,
            identity,
            profile_id,
            expected_version=expected_version,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        KnowledgeSearchNotVisible,
        KnowledgeSearchConflict,
        SearchStoreUnavailable,
        EmbeddingUnavailable,
    ) as exc:
        raise _command_error(exc) from exc
    return _envelope(item, request.state.request_id)


def _service(
    request: Request, database: Database, authorizer: Authorizer
) -> KnowledgeSearchProfileService:
    store: SearchStore | None = getattr(request.app.state, "knowledge_search_store", None)
    if store is None:
        raise _error(503, "knowledge_search_unavailable", "dependency", retryable=True)
    prefix = str(getattr(request.app.state.settings, "opensearch_index_prefix", "ioap-knowledge"))
    reranker: KnowledgeReranker | None = getattr(request.app.state, "knowledge_reranker", None)
    embedder: KnowledgeEmbedder | None = getattr(request.app.state, "knowledge_embedder", None)
    return KnowledgeSearchProfileService(
        database,
        authorizer,
        store,
        index_prefix=prefix,
        reranker=reranker,
        embedder=embedder,
    )


def _profile_response(item: SearchProfileView) -> SearchProfileResponse:
    legal: list[Literal["SYNC", "EVALUATE", "ACTIVATE_SHADOW", "DEACTIVATE_SHADOW", "QUERY"]] = []
    if item.status in {"DRAFT", "SYNCED"}:
        legal.append("SYNC")
    if item.status == "SYNCED":
        legal.append("EVALUATE")
    if item.status == "EVALUATED":
        legal.append("ACTIVATE_SHADOW")
    if item.status == "ACTIVE_SHADOW":
        legal.append("DEACTIVATE_SHADOW")
        legal.append("QUERY")
    evaluation = (
        SearchEvaluationResponse(
            evaluation_id=item.evaluation.evaluation_id,
            status=item.evaluation.status,
            policy_version=item.evaluation.policy_version,
            cases=list(item.evaluation.cases),
            metrics=item.evaluation.metrics,
            gate_results=item.evaluation.gate_results,
            failure_codes=list(item.evaluation.failure_codes),
            requested_by_subject_id=item.evaluation.requested_by_subject_id,
            completed_at=item.evaluation.completed_at,
            version=item.evaluation.version,
        )
        if item.evaluation is not None
        else None
    )
    return SearchProfileResponse(
        search_profile_id=item.search_profile_id,
        source_index_release_id=item.source_index_release_id,
        name=item.name,
        version=item.version,
        backend=item.backend,
        status=item.status,
        is_active_shadow=item.is_active_shadow,
        index_name=item.index_name,
        schema_version=item.schema_version,
        manifest_hash=item.manifest_hash,
        embedding_model_release_id=item.embedding_model_release_id,
        embedding_manifest_hash=item.embedding_manifest_hash,
        embedding_component_model_id=item.embedding_component_model_id,
        embedding_artifact_content_hash=item.embedding_artifact_content_hash,
        embedding_dimension=item.embedding_dimension,
        document_count=item.document_count,
        metrics=item.metrics,
        gate_results=item.gate_results,
        failure_codes=list(item.failure_codes),
        created_by_subject_id=item.created_by_subject_id,
        evaluated_by_subject_id=item.evaluated_by_subject_id,
        activated_by_subject_id=item.activated_by_subject_id,
        evaluated_at=item.evaluated_at,
        activated_at=item.activated_at,
        state_version=item.state_version,
        created_at=item.created_at,
        updated_at=item.updated_at,
        evaluation=evaluation,
        legal_actions=legal,
    )


def _result_envelope(item: SearchResultView, request_id: str) -> SearchResultEnvelope:
    return SearchResultEnvelope(
        data=SearchResultResponse(
            profile_id=item.profile_id,
            source_index_release_id=item.source_index_release_id,
            backend=item.backend,
            guard_stage=item.guard_stage,
            embedding=EmbeddingTraceResponse(**asdict(item.embedding)),
            rerank=RerankTraceResponse(**asdict(item.rerank)),
            evidence=[SearchEvidenceResponse(**asdict(evidence)) for evidence in item.evidence],
        ),
        meta={"request_id": request_id, "total": len(item.evidence)},
    )


def _envelope(item: SearchProfileView, request_id: str) -> SearchProfileEnvelope:
    return SearchProfileEnvelope(data=_profile_response(item), meta={"request_id": request_id})


def _rebuild_command(
    item: SearchProfileRebuildView,
    identity: IdentityContext,
    request_id: str,
) -> SearchProfileRebuildActivityInput:
    return SearchProfileRebuildActivityInput(
        rebuild_job_id=item.rebuild_job_id,
        workflow_id=item.workflow_id,
        identity=identity,
        request_id=request_id,
    )


def _rebuild_response(item: SearchProfileRebuildView) -> SearchProfileRebuildResponse:
    return SearchProfileRebuildResponse(
        rebuild_job_id=item.rebuild_job_id,
        search_profile_id=item.search_profile_id,
        workflow_id=item.workflow_id,
        expected_profile_version=item.expected_profile_version,
        status=cast(Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"], item.status),
        stage=item.stage,
        progress_percent=item.progress_percent,
        completed_items=item.completed_items,
        total_items=item.total_items,
        attempt_count=item.attempt_count,
        requested_by_subject_id=item.requested_by_subject_id,
        failure_code=item.failure_code,
        started_at=item.started_at,
        completed_at=item.completed_at,
        version=item.version,
        created_at=item.created_at,
        updated_at=item.updated_at,
        legal_actions=["RETRY"] if item.status == "FAILED" else [],
    )


def _rebuild_envelope(
    item: SearchProfileRebuildView,
    request_id: str,
) -> SearchProfileRebuildEnvelope:
    return SearchProfileRebuildEnvelope(
        data=_rebuild_response(item),
        meta={"request_id": request_id},
    )


def _version(value: str) -> int:
    normalized = value.strip().strip("W/").strip('"')
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise _error(400, "invalid_if_match", "validation") from exc
    if parsed < 1:
        raise _error(400, "invalid_if_match", "validation")
    return parsed


def _command_error(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return _error(403, "knowledge_search_forbidden", "authorization")
    if isinstance(exc, KnowledgeSearchNotVisible):
        return _error(404, "knowledge_search_not_found", "not_found")
    if isinstance(exc, SearchStoreUnavailable):
        return _error(503, "knowledge_search_dependency_unavailable", "dependency", retryable=True)
    if isinstance(exc, EmbeddingUnavailable):
        return _error(
            503,
            exc.reason,
            "dependency",
            retryable=True,
            details={"release_id": exc.release_id} if exc.release_id else None,
        )
    if isinstance(exc, SearchProfileRebuildDispatchUnavailable):
        return _error(
            503,
            "knowledge_search_rebuild_workflow_unavailable",
            "dependency",
            retryable=True,
        )
    if isinstance(exc, KnowledgeSearchConflict):
        return _error(
            409,
            exc.reason,
            "conflict",
            details={"current_version": exc.current_version} if exc.current_version else None,
        )
    return _error(500, "knowledge_search_failed", "internal")


def _error(
    status: int,
    code: str,
    category: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> AppError:
    return AppError(
        status_code=status,
        code=code,
        category=category,
        message=code.replace("_", " ").capitalize(),
        retryable=retryable,
        details=details,
    )
