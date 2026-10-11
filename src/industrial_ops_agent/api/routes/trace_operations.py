"""Prompt-free Agent and AI business trace APIs."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path, Query, Request
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.trace_operations.service import (
    TraceDetail,
    TraceNode,
    TraceNotFound,
    TraceOperationsService,
    TracePage,
    TraceSummary,
)

router = APIRouter(tags=["trace-operations"])

TraceHealthValue = Literal["HEALTHY", "RUNNING", "ATTENTION", "FAILED"]
TraceEvidenceValue = Literal["COMPLETE", "PARTIAL"]


class TraceSummaryResponse(BaseModel):
    agent_run_id: str
    diagnosis_run_id: str
    incident_id: str
    workflow_id: str
    agent_status: str
    diagnosis_status: str
    stop_reason: str | None
    health: TraceHealthValue
    evidence_status: TraceEvidenceValue
    started_at: datetime
    finished_at: datetime | None
    duration_ms: float | None
    latest_activity_at: datetime
    event_count: int
    tool_call_count: int
    inference_count: int
    failed_node_count: int
    prompt_tokens: int
    completion_tokens: int
    trace_ids: list[str]
    model_release_ids: list[str]
    request_classes: list[str]
    work_order_ids: list[str]


class TraceNodeResponse(BaseModel):
    node_id: str
    parent_node_id: str | None
    category: Literal[
        "WORKFLOW",
        "AGENT",
        "AGENT_EVENT",
        "RETRIEVAL",
        "TOOL",
        "MODEL",
        "BUSINESS",
    ]
    name: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    duration_ms: float | None
    attributes: dict[str, Any]


class TraceListDataResponse(BaseModel):
    items: list[TraceSummaryResponse]
    total: int
    limit: int
    offset: int
    projection_version: str


class TraceDetailResponse(BaseModel):
    summary: TraceSummaryResponse
    nodes: list[TraceNodeResponse]
    missing_correlations: list[str]
    privacy_notice: str
    projection_version: str


class TraceListEnvelope(BaseModel):
    data: TraceListDataResponse
    meta: dict[str, str]


class TraceDetailEnvelope(BaseModel):
    data: TraceDetailResponse
    meta: dict[str, str]


@router.get("/operations/traces", response_model=TraceListEnvelope)
async def list_operation_traces(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TraceListEnvelope:
    try:
        page = await asyncio.to_thread(
            TraceOperationsService(database, authorizer).list_traces,
            identity,
            request_id=request.state.request_id,
            status=status,
            limit=limit,
            offset=offset,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "trace_operations_forbidden") from exc
    return TraceListEnvelope(
        data=_page(page),
        meta={"request_id": request.state.request_id},
    )


@router.get("/operations/traces/{agent_run_id}", response_model=TraceDetailEnvelope)
async def get_operation_trace(
    agent_run_id: Annotated[
        str,
        Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ],
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> TraceDetailEnvelope:
    try:
        detail = await asyncio.to_thread(
            TraceOperationsService(database, authorizer).get_trace,
            identity,
            agent_run_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "trace_operations_forbidden") from exc
    except TraceNotFound as exc:
        raise _error(404, "trace_not_found") from exc
    return TraceDetailEnvelope(
        data=_detail(detail),
        meta={"request_id": request.state.request_id},
    )


def _page(page: TracePage) -> TraceListDataResponse:
    return TraceListDataResponse(
        items=[_summary(item) for item in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
        projection_version=TraceOperationsService.PROJECTION_VERSION,
    )


def _detail(detail: TraceDetail) -> TraceDetailResponse:
    return TraceDetailResponse(
        summary=_summary(detail.summary),
        nodes=[_node(item) for item in detail.nodes],
        missing_correlations=list(detail.missing_correlations),
        privacy_notice=detail.privacy_notice,
        projection_version=TraceOperationsService.PROJECTION_VERSION,
    )


def _summary(item: TraceSummary) -> TraceSummaryResponse:
    return TraceSummaryResponse(
        agent_run_id=item.agent_run_id,
        diagnosis_run_id=item.diagnosis_run_id,
        incident_id=item.incident_id,
        workflow_id=item.workflow_id,
        agent_status=item.agent_status,
        diagnosis_status=item.diagnosis_status,
        stop_reason=item.stop_reason,
        health=item.health,
        evidence_status=item.evidence_status,
        started_at=item.started_at,
        finished_at=item.finished_at,
        duration_ms=item.duration_ms,
        latest_activity_at=item.latest_activity_at,
        event_count=item.event_count,
        tool_call_count=item.tool_call_count,
        inference_count=item.inference_count,
        failed_node_count=item.failed_node_count,
        prompt_tokens=item.prompt_tokens,
        completion_tokens=item.completion_tokens,
        trace_ids=list(item.trace_ids),
        model_release_ids=list(item.model_release_ids),
        request_classes=list(item.request_classes),
        work_order_ids=list(item.work_order_ids),
    )


def _node(item: TraceNode) -> TraceNodeResponse:
    return TraceNodeResponse(
        node_id=item.node_id,
        parent_node_id=item.parent_node_id,
        category=item.category,
        name=item.name,
        status=item.status,
        started_at=item.started_at,
        finished_at=item.finished_at,
        duration_ms=item.duration_ms,
        attributes=item.attributes,
    )


def _error(status_code: int, code: str) -> AppError:
    return AppError(
        status_code=status_code,
        code=code,
        category="authorization" if status_code == 403 else "not_found",
        message="Trace operations request was rejected",
    )
