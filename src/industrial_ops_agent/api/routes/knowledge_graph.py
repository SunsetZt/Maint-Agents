"""Governed GraphRAG causal evidence API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_graph_extraction_binding_resolver,
    get_graph_extraction_dispatcher,
    get_identity,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.graph_rag.extraction import (
    GraphExtractionActivityInput,
    GraphExtractionBindingResolver,
    GraphExtractionConflict,
    GraphExtractionDispatcher,
    GraphExtractionDispatchUnavailable,
    GraphExtractionNotVisible,
    GraphExtractionService,
    GraphExtractionView,
)
from industrial_ops_agent.graph_rag.service import (
    GraphEdgeView,
    GraphEvaluationView,
    GraphNodeView,
    GraphPathView,
    GraphReleaseView,
    KnowledgeGraphConflict,
    KnowledgeGraphNotVisible,
    KnowledgeGraphService,
)
from industrial_ops_agent.graph_rag.store import GraphStore, GraphStoreUnavailable
from industrial_ops_agent.persistence.database import Database

router = APIRouter(prefix="/knowledge-graphs", tags=["knowledge-graph"])

NodeType = Literal["COMPONENT", "FAILURE_MODE", "SYMPTOM", "CAUSE", "ACTION", "MATERIAL"]
RelationType = Literal[
    "DEPENDS_ON",
    "CAUSES",
    "MANIFESTS_AS",
    "MITIGATED_BY",
    "COMPATIBLE_WITH",
    "INCOMPATIBLE_WITH",
    "PART_OF",
]


class CreateGraphReleaseBody(BaseModel):
    source_index_release_id: str = Field(min_length=3, max_length=128)
    name: str = Field(min_length=2, max_length=128)


class CreateGraphNodeBody(BaseModel):
    node_key: str = Field(min_length=2, max_length=128)
    node_type: NodeType
    display_name: str = Field(min_length=2, max_length=255)
    citation_id: str = Field(min_length=3, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateGraphEdgeBody(BaseModel):
    source_node_id: str = Field(min_length=3, max_length=128)
    target_node_id: str = Field(min_length=3, max_length=128)
    relation_type: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    citation_id: str = Field(min_length=3, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEvaluationCaseBody(BaseModel):
    start_node_key: str = Field(min_length=2, max_length=128)
    target_node_type: NodeType
    relation_types: list[RelationType] = Field(min_length=1, max_length=7)
    max_hops: int = Field(ge=1, le=4)
    expected_target_node_key: str = Field(min_length=2, max_length=128)


class EvaluateGraphBody(BaseModel):
    cases: list[GraphEvaluationCaseBody] = Field(min_length=1, max_length=100)


class GraphQueryBody(BaseModel):
    graph_name: str = Field(min_length=2, max_length=128)
    start_node_key: str = Field(min_length=2, max_length=128)
    target_node_type: NodeType | None = None
    relation_types: list[RelationType] = Field(min_length=1, max_length=7)
    max_hops: int = Field(default=3, ge=1, le=4)
    limit: int = Field(default=10, ge=1, le=20)


class GraphNodeResponse(BaseModel):
    graph_node_id: str
    node_key: str
    node_type: str
    display_name: str
    citation_id: str
    document_version_id: str
    classification: str
    device_families: list[str]
    device_models: list[str]
    metadata: dict[str, Any]


class GraphEdgeResponse(BaseModel):
    graph_edge_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    confidence: float
    citation_id: str
    document_version_id: str
    metadata: dict[str, Any]


class GraphEvaluationResponse(BaseModel):
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


class GraphReleaseResponse(BaseModel):
    graph_release_id: str
    source_index_release_id: str
    name: str
    version: int
    status: str
    is_active: bool
    schema_version: str
    manifest_hash: str | None
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
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
    evaluation: GraphEvaluationResponse | None
    legal_actions: list[
        Literal[
            "ADD_NODE",
            "ADD_EDGE",
            "REQUEST_EXTRACTION",
            "SYNC",
            "EVALUATE",
            "ACTIVATE",
        ]
    ]


class GraphReleaseEnvelope(BaseModel):
    data: GraphReleaseResponse
    meta: dict[str, str]


class GraphReleaseListEnvelope(BaseModel):
    data: list[GraphReleaseResponse]
    meta: dict[str, str | int]


class GraphPathResponse(BaseModel):
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]


class GraphPathEnvelope(BaseModel):
    data: list[GraphPathResponse]
    meta: dict[str, str | int]


class CreateGraphExtractionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_ids: list[str] = Field(min_length=1, max_length=32)


class ReviewGraphExtractionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["ACCEPT", "REJECT"]
    reason: str = Field(min_length=8, max_length=1_000)


GraphExtractionStatus = Literal[
    "QUEUED",
    "RUNNING",
    "REVIEW_PENDING",
    "ACCEPTED",
    "REJECTED",
    "FAILED",
]
GraphExtractionStage = Literal[
    "QUEUED",
    "SOURCE_VALIDATION",
    "MODEL_BINDING_FAILED",
    "PROMPT_BINDING_FAILED",
    "MODEL_INFERENCE_FAILED",
    "CANDIDATE_VALIDATION_FAILED",
    "SOURCE_VALIDATION_FAILED",
    "DEPENDENCY_FAILED",
    "DISPATCH_FAILED",
    "REVIEW_PENDING",
    "ACCEPTED_DRAFT",
    "REJECTED",
]
GraphExtractionFailureCode = Literal[
    "graph_extraction_candidate_invalid",
    "graph_extraction_dependency_unavailable",
    "graph_extraction_dispatch_unavailable",
    "graph_extraction_inference_rejected",
    "graph_extraction_inference_retryable",
    "graph_extraction_model_binding_changed",
    "graph_extraction_prompt_binding_changed",
    "graph_extraction_prompt_not_deployed",
    "graph_extraction_runtime_binding_changed",
    "graph_extraction_source_binding_changed",
]


class GraphExtractionCitationBindingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    chunk_id: str
    document_version_id: str
    document_id: str
    excerpt_checksum: str
    chunk_content_checksum: str
    document_content_checksum: str
    source_checksum: str


class GraphExtractionCandidateNodeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_key: str = Field(min_length=2, max_length=128)
    node_type: NodeType
    display_name: str = Field(min_length=2, max_length=255)
    citation_id: str = Field(min_length=3, max_length=128)
    metadata: dict[str, Any]


class GraphExtractionCandidateEdgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_node_key: str = Field(min_length=2, max_length=128)
    target_node_key: str = Field(min_length=2, max_length=128)
    relation_type: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    citation_id: str = Field(min_length=3, max_length=128)
    metadata: dict[str, Any]


class GraphExtractionCandidateBundleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[GraphExtractionCandidateNodeResponse] = Field(max_length=64)
    edges: list[GraphExtractionCandidateEdgeResponse] = Field(max_length=128)


class GraphExtractionResponse(BaseModel):
    extraction_job_id: str
    graph_release_id: str
    source_index_release_id: str
    expected_graph_version: int
    citation_ids: list[str]
    citation_bindings: list[GraphExtractionCitationBindingResponse]
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
    status: GraphExtractionStatus
    stage: GraphExtractionStage
    attempt_count: int
    candidate_bundle: GraphExtractionCandidateBundleResponse | None
    result_hash: str | None
    context_hash: str | None
    failure_code: GraphExtractionFailureCode | None
    requested_by_subject_id: str
    reviewed_by_subject_id: str | None
    review_decision: Literal["ACCEPT", "REJECT"] | None
    review_reason: str | None
    started_at: datetime | None
    completed_at: datetime | None
    reviewed_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    legal_actions: list[Literal["RETRY", "REVIEW_ACCEPT", "REVIEW_REJECT"]]


class GraphExtractionEnvelope(BaseModel):
    data: GraphExtractionResponse
    meta: dict[str, str]


class GraphExtractionListEnvelope(BaseModel):
    data: list[GraphExtractionResponse]
    meta: dict[str, str | int]


@router.get("", response_model=GraphReleaseListEnvelope)
async def list_graph_releases(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> GraphReleaseListEnvelope:
    try:
        items = await asyncio.to_thread(
            _service(request, database, authorizer).list,
            identity,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "knowledge_graph_forbidden", "authorization") from exc
    return GraphReleaseListEnvelope(
        data=[
            _release_response(item, extraction_available=_extraction_available(request))
            for item in items
        ],
        meta={"request_id": request.state.request_id, "total": len(items)},
    )


@router.post("", response_model=GraphReleaseEnvelope, status_code=201)
async def create_graph_release(
    body: CreateGraphReleaseBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=255)],
) -> GraphReleaseEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer).create,
            identity,
            **body.model_dump(),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeGraphNotVisible, KnowledgeGraphConflict) as exc:
        raise _command_error(exc) from exc
    return _envelope(
        item,
        request.state.request_id,
        extraction_available=_extraction_available(request),
    )


@router.get("/{graph_release_id}", response_model=GraphReleaseEnvelope)
async def get_graph_release(
    graph_release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> GraphReleaseEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer).get,
            identity,
            graph_release_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeGraphNotVisible) as exc:
        raise _command_error(exc) from exc
    return _envelope(
        item,
        request.state.request_id,
        extraction_available=_extraction_available(request),
    )


@router.post(
    "/{graph_release_id}/extractions",
    response_model=GraphExtractionEnvelope,
    status_code=202,
)
async def request_graph_extraction(
    graph_release_id: str,
    body: CreateGraphExtractionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    binding_resolver: Annotated[
        GraphExtractionBindingResolver,
        Depends(get_graph_extraction_binding_resolver),
    ],
    dispatcher: Annotated[
        GraphExtractionDispatcher,
        Depends(get_graph_extraction_dispatcher),
    ],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=255)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> GraphExtractionEnvelope:
    service = GraphExtractionService(database, authorizer)
    try:
        binding = await asyncio.to_thread(binding_resolver.resolve, identity)
        item = await asyncio.to_thread(
            service.request,
            identity,
            graph_release_id,
            citation_ids=tuple(body.citation_ids),
            expected_graph_version=_version(if_match),
            idempotency_key=idempotency_key,
            binding=binding,
            request_id=request.state.request_id,
        )
        if item.dispatch_required:
            await _dispatch_extraction(
                service,
                dispatcher,
                identity,
                item,
                request.state.request_id,
            )
    except (
        AuthorizationDenied,
        GraphExtractionNotVisible,
        GraphExtractionConflict,
        GraphExtractionDispatchUnavailable,
    ) as exc:
        raise _extraction_error(exc) from exc
    return _extraction_envelope(item, request.state.request_id, identity)


@router.get(
    "/{graph_release_id}/extractions",
    response_model=GraphExtractionListEnvelope,
)
async def list_graph_extractions(
    graph_release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> GraphExtractionListEnvelope:
    try:
        items = await asyncio.to_thread(
            GraphExtractionService(database, authorizer).list,
            identity,
            graph_release_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, GraphExtractionNotVisible) as exc:
        raise _extraction_error(exc) from exc
    return GraphExtractionListEnvelope(
        data=[_extraction_response(item, identity) for item in items],
        meta={"request_id": request.state.request_id, "total": len(items)},
    )


@router.get(
    "/extractions/{extraction_job_id}",
    response_model=GraphExtractionEnvelope,
)
async def get_graph_extraction(
    extraction_job_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> GraphExtractionEnvelope:
    try:
        item = await asyncio.to_thread(
            GraphExtractionService(database, authorizer).get,
            identity,
            extraction_job_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, GraphExtractionNotVisible) as exc:
        raise _extraction_error(exc) from exc
    return _extraction_envelope(item, request.state.request_id, identity)


@router.post(
    "/extractions/{extraction_job_id}/retry",
    response_model=GraphExtractionEnvelope,
    status_code=202,
)
async def retry_graph_extraction(
    extraction_job_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[
        GraphExtractionDispatcher,
        Depends(get_graph_extraction_dispatcher),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> GraphExtractionEnvelope:
    service = GraphExtractionService(database, authorizer)
    try:
        item = await asyncio.to_thread(
            service.retry,
            identity,
            extraction_job_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
        await _dispatch_extraction(service, dispatcher, identity, item, request.state.request_id)
    except (
        AuthorizationDenied,
        GraphExtractionNotVisible,
        GraphExtractionConflict,
        GraphExtractionDispatchUnavailable,
    ) as exc:
        raise _extraction_error(exc) from exc
    return _extraction_envelope(item, request.state.request_id, identity)


@router.post(
    "/extractions/{extraction_job_id}/review",
    response_model=GraphExtractionEnvelope,
)
async def review_graph_extraction(
    extraction_job_id: str,
    body: ReviewGraphExtractionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> GraphExtractionEnvelope:
    try:
        item = await asyncio.to_thread(
            GraphExtractionService(database, authorizer).review,
            identity,
            extraction_job_id,
            decision=body.decision,
            reason=body.reason,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        GraphExtractionNotVisible,
        GraphExtractionConflict,
    ) as exc:
        raise _extraction_error(exc) from exc
    return _extraction_envelope(item, request.state.request_id, identity)


@router.post("/{graph_release_id}/nodes", response_model=GraphReleaseEnvelope)
async def add_graph_node(
    graph_release_id: str,
    body: CreateGraphNodeBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> GraphReleaseEnvelope:
    return await _mutate(
        request,
        _service(request, database, authorizer).add_node,
        identity,
        graph_release_id,
        body.model_dump(),
        if_match,
    )


@router.post("/{graph_release_id}/edges", response_model=GraphReleaseEnvelope)
async def add_graph_edge(
    graph_release_id: str,
    body: CreateGraphEdgeBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> GraphReleaseEnvelope:
    return await _mutate(
        request,
        _service(request, database, authorizer).add_edge,
        identity,
        graph_release_id,
        body.model_dump(),
        if_match,
    )


@router.post("/{graph_release_id}/sync", response_model=GraphReleaseEnvelope)
async def sync_graph_release(
    graph_release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> GraphReleaseEnvelope:
    return await _mutate(
        request,
        _service(request, database, authorizer).sync,
        identity,
        graph_release_id,
        {},
        if_match,
    )


@router.post("/{graph_release_id}/evaluations", response_model=GraphReleaseEnvelope)
async def evaluate_graph_release(
    graph_release_id: str,
    body: EvaluateGraphBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> GraphReleaseEnvelope:
    return await _mutate(
        request,
        _service(request, database, authorizer).evaluate,
        identity,
        graph_release_id,
        {"cases": tuple(item.model_dump() for item in body.cases)},
        if_match,
    )


@router.post("/{graph_release_id}/activate", response_model=GraphReleaseEnvelope)
async def activate_graph_release(
    graph_release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> GraphReleaseEnvelope:
    return await _mutate(
        request,
        _service(request, database, authorizer).activate,
        identity,
        graph_release_id,
        {},
        if_match,
    )


@router.post("/query", response_model=GraphPathEnvelope)
async def query_active_graph(
    body: GraphQueryBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> GraphPathEnvelope:
    try:
        paths = await asyncio.to_thread(
            _service(request, database, authorizer).query,
            identity,
            **body.model_dump(),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        KnowledgeGraphNotVisible,
        KnowledgeGraphConflict,
        GraphStoreUnavailable,
    ) as exc:
        raise _command_error(exc) from exc
    return GraphPathEnvelope(
        data=[_path_response(path) for path in paths],
        meta={"request_id": request.state.request_id, "total": len(paths)},
    )


async def _mutate(
    request: Request,
    command: Any,
    identity: IdentityContext,
    graph_release_id: str,
    values: dict[str, Any],
    if_match: str,
) -> GraphReleaseEnvelope:
    try:
        item = await asyncio.to_thread(
            command,
            identity,
            graph_release_id,
            **values,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        KnowledgeGraphNotVisible,
        KnowledgeGraphConflict,
        GraphStoreUnavailable,
    ) as exc:
        raise _command_error(exc) from exc
    return _envelope(
        item,
        request.state.request_id,
        extraction_available=_extraction_available(request),
    )


def _service(request: Request, database: Database, authorizer: Authorizer) -> KnowledgeGraphService:
    store: GraphStore | None = getattr(request.app.state, "knowledge_graph_store", None)
    if store is None:
        raise _error(503, "knowledge_graph_store_unavailable", "dependency", retryable=True)
    return KnowledgeGraphService(database, authorizer, store)


def _release_response(
    item: GraphReleaseView,
    *,
    extraction_available: bool = False,
) -> GraphReleaseResponse:
    actions: list[
        Literal[
            "ADD_NODE",
            "ADD_EDGE",
            "REQUEST_EXTRACTION",
            "SYNC",
            "EVALUATE",
            "ACTIVATE",
        ]
    ] = []
    if item.status == "DRAFT":
        actions.extend(["ADD_NODE", "ADD_EDGE"])
        if extraction_available:
            actions.append("REQUEST_EXTRACTION")
        if len(item.nodes) >= 2 and item.edges:
            actions.append("SYNC")
    elif item.status == "SYNCED":
        actions.append("EVALUATE")
    elif item.status == "EVALUATED":
        actions.append("ACTIVATE")
    return GraphReleaseResponse(
        **{
            "graph_release_id": item.graph_release_id,
            "source_index_release_id": item.source_index_release_id,
            "name": item.name,
            "version": item.version,
            "status": item.status,
            "is_active": item.is_active,
            "schema_version": item.schema_version,
            "manifest_hash": item.manifest_hash,
            "metrics": item.metrics,
            "gate_results": item.gate_results,
            "failure_codes": list(item.failure_codes),
            "created_by_subject_id": item.created_by_subject_id,
            "evaluated_by_subject_id": item.evaluated_by_subject_id,
            "activated_by_subject_id": item.activated_by_subject_id,
            "evaluated_at": item.evaluated_at,
            "activated_at": item.activated_at,
            "state_version": item.state_version,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "nodes": [_node_response(node) for node in item.nodes],
            "edges": [_edge_response(edge) for edge in item.edges],
            "evaluation": _evaluation_response(item.evaluation) if item.evaluation else None,
            "legal_actions": actions,
        }
    )


def _node_response(item: GraphNodeView) -> GraphNodeResponse:
    return GraphNodeResponse(
        graph_node_id=item.graph_node_id,
        node_key=item.node_key,
        node_type=item.node_type,
        display_name=item.display_name,
        citation_id=item.citation_id,
        document_version_id=item.document_version_id,
        classification=item.classification,
        device_families=list(item.device_families),
        device_models=list(item.device_models),
        metadata=item.metadata,
    )


def _edge_response(item: GraphEdgeView) -> GraphEdgeResponse:
    return GraphEdgeResponse(
        graph_edge_id=item.graph_edge_id,
        source_node_id=item.source_node_id,
        target_node_id=item.target_node_id,
        relation_type=item.relation_type,
        confidence=item.confidence,
        citation_id=item.citation_id,
        document_version_id=item.document_version_id,
        metadata=item.metadata,
    )


def _evaluation_response(item: GraphEvaluationView) -> GraphEvaluationResponse:
    return GraphEvaluationResponse(
        evaluation_id=item.evaluation_id,
        status=item.status,
        policy_version=item.policy_version,
        cases=list(item.cases),
        metrics=item.metrics,
        gate_results=item.gate_results,
        failure_codes=list(item.failure_codes),
        requested_by_subject_id=item.requested_by_subject_id,
        completed_at=item.completed_at,
        version=item.version,
    )


def _path_response(item: GraphPathView) -> GraphPathResponse:
    return GraphPathResponse(
        nodes=[_node_response(node) for node in item.nodes],
        edges=[_edge_response(edge) for edge in item.edges],
    )


async def _dispatch_extraction(
    service: GraphExtractionService,
    dispatcher: GraphExtractionDispatcher,
    identity: IdentityContext,
    item: GraphExtractionView,
    request_id: str,
) -> None:
    try:
        await dispatcher.dispatch(
            GraphExtractionActivityInput(
                extraction_job_id=item.extraction_job_id,
                workflow_id=item.workflow_id,
                identity=identity,
                request_id=request_id,
            )
        )
    except Exception as exc:
        await asyncio.to_thread(
            service.mark_dispatch_failed,
            identity,
            item.extraction_job_id,
            workflow_id=item.workflow_id,
        )
        raise GraphExtractionDispatchUnavailable from exc


def _extraction_response(
    item: GraphExtractionView,
    identity: IdentityContext,
) -> GraphExtractionResponse:
    legal_actions = list(item.legal_actions)
    if item.status == "REVIEW_PENDING" and item.requested_by_subject_id == identity.subject_id:
        legal_actions = []
    return GraphExtractionResponse(
        extraction_job_id=item.extraction_job_id,
        graph_release_id=item.graph_release_id,
        source_index_release_id=item.source_index_release_id,
        expected_graph_version=item.expected_graph_version,
        citation_ids=list(item.citation_ids),
        citation_bindings=[dict(binding) for binding in item.citation_bindings],
        source_binding_hash=item.source_binding_hash,
        extraction_profile=item.extraction_profile,
        workflow_id=item.workflow_id,
        inference_request_id=item.inference_request_id,
        model_alias=item.model_alias,
        model_release_id=item.model_release_id,
        model_manifest_hash=item.model_manifest_hash,
        prompt_bundle_id=item.prompt_bundle_id,
        prompt_bundle_hash=item.prompt_bundle_hash,
        response_schema_version=item.response_schema_version,
        status=item.status,
        stage=item.stage,
        attempt_count=item.attempt_count,
        candidate_bundle=item.candidate_bundle,
        result_hash=item.result_hash,
        context_hash=item.context_hash,
        failure_code=item.failure_code,
        requested_by_subject_id=item.requested_by_subject_id,
        reviewed_by_subject_id=item.reviewed_by_subject_id,
        review_decision=item.review_decision,
        review_reason=item.review_reason,
        started_at=item.started_at,
        completed_at=item.completed_at,
        reviewed_at=item.reviewed_at,
        version=item.version,
        created_at=item.created_at,
        updated_at=item.updated_at,
        legal_actions=legal_actions,
    )


def _extraction_envelope(
    item: GraphExtractionView,
    request_id: str,
    identity: IdentityContext,
) -> GraphExtractionEnvelope:
    return GraphExtractionEnvelope(
        data=_extraction_response(item, identity),
        meta={"request_id": request_id},
    )


def _envelope(
    item: GraphReleaseView,
    request_id: str,
    *,
    extraction_available: bool = False,
) -> GraphReleaseEnvelope:
    return GraphReleaseEnvelope(
        data=_release_response(item, extraction_available=extraction_available),
        meta={"request_id": request_id},
    )


def _extraction_available(request: Request) -> bool:
    return bool(
        getattr(request.app.state, "graph_extraction_dispatcher", None) is not None
        and getattr(request.app.state, "graph_extraction_binding_resolver", None) is not None
    )


def _version(value: str) -> int:
    try:
        return int(value.strip().strip('"'))
    except ValueError as exc:
        raise _error(400, "if_match_invalid", "validation") from exc


def _command_error(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return _error(403, "knowledge_graph_forbidden", "authorization")
    if isinstance(exc, KnowledgeGraphNotVisible):
        return _error(404, "knowledge_graph_not_found", "not_found")
    if isinstance(exc, GraphStoreUnavailable):
        return _error(503, "knowledge_graph_store_unavailable", "dependency", retryable=True)
    if isinstance(exc, KnowledgeGraphConflict):
        return _error(409, exc.reason, "conflict")
    return _error(500, "knowledge_graph_command_failed", "internal")


def _extraction_error(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return _error(403, "graph_extraction_forbidden", "authorization")
    if isinstance(exc, GraphExtractionNotVisible):
        return _error(404, "graph_extraction_not_found", "not_found")
    if isinstance(exc, GraphExtractionDispatchUnavailable):
        return _error(
            503,
            "graph_extraction_workflow_unavailable",
            "dependency",
            retryable=True,
        )
    if isinstance(exc, GraphExtractionConflict):
        return _error(409, exc.reason, "conflict")
    return _error(500, "graph_extraction_command_failed", "internal")


def _error(status: int, code: str, category: str, *, retryable: bool = False) -> AppError:
    return AppError(
        status_code=status,
        code=code,
        category=category,
        message=code.replace("_", " ").capitalize(),
        retryable=retryable,
    )
