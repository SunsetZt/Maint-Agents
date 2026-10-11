"""Controlled supplier A2A collaboration API with human acceptance gates."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_dlp_processor,
    get_identity,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.collaboration.a2a import SupplierA2AError, SupplierAgentClient
from industrial_ops_agent.collaboration.service import (
    CollaborationConflict,
    CollaborationEventView,
    CollaborationNotVisible,
    SupplierCollaborationService,
    SupplierCollaborationView,
)
from industrial_ops_agent.data_governance.dlp import PresidioDlpProcessor
from industrial_ops_agent.persistence.database import Database

router = APIRouter(prefix="/supplier-collaborations", tags=["supplier-agent-collaboration"])


class CreateSupplierCollaborationBody(BaseModel):
    incident_id: str = Field(min_length=3, max_length=128)
    diagnosis_run_id: str = Field(min_length=3, max_length=128)
    question: str = Field(min_length=8, max_length=2_000)
    sharing_reason: str = Field(min_length=8, max_length=1_024)


class ReviewSupplierCollaborationBody(BaseModel):
    decision: Literal["ACCEPTED", "REJECTED"]
    reason: str = Field(min_length=8, max_length=2_000)


class SupplierCollaborationEventResponse(BaseModel):
    sequence: int
    event_type: str
    actor_subject_id: str
    reason_code: str
    metadata: dict[str, Any]
    occurred_at: datetime


class SupplierCollaborationResponse(BaseModel):
    collaboration_id: str
    incident_id: str
    diagnosis_run_id: str
    asset_id: str
    agent_name: str
    agent_version: str
    skill_id: str
    status: str
    sharing_reason: str
    request_payload: dict[str, Any]
    request_payload_digest: str
    dlp_policy_version: str
    dlp_finding_count: int
    remote_task_id: str | None
    response_payload: dict[str, Any] | None
    response_digest: str | None
    guardrail_policy_version: str | None
    requested_by_subject_id: str
    reviewed_by_subject_id: str | None
    review_decision: str | None
    review_reason: str | None
    reviewed_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    events: list[SupplierCollaborationEventResponse]
    legal_actions: list[Literal["DISPATCH", "REFRESH", "CANCEL", "REVIEW"]]


class SupplierCollaborationEnvelope(BaseModel):
    data: SupplierCollaborationResponse
    meta: dict[str, str]


class SupplierCollaborationListEnvelope(BaseModel):
    data: list[SupplierCollaborationResponse]
    legal_actions: list[Literal["CREATE"]]
    meta: dict[str, str | int]


@router.get("", response_model=SupplierCollaborationListEnvelope)
async def list_supplier_collaborations(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[str | None, Query(min_length=3, max_length=32)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> SupplierCollaborationListEnvelope:
    try:
        service = _service(request, database, authorizer, None)
        results = await asyncio.to_thread(
            service.list,
            identity,
            status=status,
            limit=limit,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "supplier_collaboration_forbidden", "authorization") from exc
    return SupplierCollaborationListEnvelope(
        data=[_response(item) for item in results],
        legal_actions=(
            ["CREATE"]
            if getattr(request.app.state, "supplier_agent_client", None) is not None
            and authorizer.decide(
                identity,
                Action.CREATE_AGENT_COLLABORATION,
                ResourceContext(identity.tenant_id, "supplier-collaborations"),
            ).allowed
            else []
        ),
        meta={"request_id": request.state.request_id, "total": len(results)},
    )


@router.post("", response_model=SupplierCollaborationEnvelope, status_code=201)
async def create_supplier_collaboration(
    body: CreateSupplierCollaborationBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dlp: Annotated[PresidioDlpProcessor, Depends(get_dlp_processor)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=255),
    ],
) -> SupplierCollaborationEnvelope:
    service = _service(request, database, authorizer, dlp)
    try:
        item = await asyncio.to_thread(
            service.create,
            identity,
            **body.model_dump(),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "supplier_collaboration_forbidden", "authorization") from exc
    except CollaborationNotVisible as exc:
        raise _error(404, "supplier_collaboration_source_not_found", "not_found") from exc
    except CollaborationConflict as exc:
        raise _conflict(exc) from exc
    return _envelope(item, request.state.request_id)


@router.get("/{collaboration_id}", response_model=SupplierCollaborationEnvelope)
async def get_supplier_collaboration(
    collaboration_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> SupplierCollaborationEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer, None).get,
            identity,
            collaboration_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "supplier_collaboration_forbidden", "authorization") from exc
    except CollaborationNotVisible as exc:
        raise _error(404, "supplier_collaboration_not_found", "not_found") from exc
    return _envelope(item, request.state.request_id)


@router.post("/{collaboration_id}/dispatch", response_model=SupplierCollaborationEnvelope)
async def dispatch_supplier_collaboration(
    collaboration_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> SupplierCollaborationEnvelope:
    service = _service(request, database, authorizer, None)
    try:
        item = await service.dispatch(
            identity,
            collaboration_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        CollaborationNotVisible,
        CollaborationConflict,
        SupplierA2AError,
    ) as exc:
        raise _command_error(exc) from exc
    return _envelope(item, request.state.request_id)


@router.post("/{collaboration_id}/refresh", response_model=SupplierCollaborationEnvelope)
async def refresh_supplier_collaboration(
    collaboration_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> SupplierCollaborationEnvelope:
    service = _service(request, database, authorizer, None)
    try:
        item = await service.refresh(
            identity,
            collaboration_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        CollaborationNotVisible,
        CollaborationConflict,
        SupplierA2AError,
    ) as exc:
        raise _command_error(exc) from exc
    return _envelope(item, request.state.request_id)


@router.post("/{collaboration_id}/cancel", response_model=SupplierCollaborationEnvelope)
async def cancel_supplier_collaboration(
    collaboration_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> SupplierCollaborationEnvelope:
    service = _service(request, database, authorizer, None)
    try:
        item = await service.cancel(
            identity,
            collaboration_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        CollaborationNotVisible,
        CollaborationConflict,
        SupplierA2AError,
    ) as exc:
        raise _command_error(exc) from exc
    return _envelope(item, request.state.request_id)


@router.post("/{collaboration_id}/reviews", response_model=SupplierCollaborationEnvelope)
async def review_supplier_collaboration(
    collaboration_id: str,
    body: ReviewSupplierCollaborationBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> SupplierCollaborationEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer, None).review,
            identity,
            collaboration_id,
            decision=body.decision,
            reason=body.reason,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, CollaborationNotVisible, CollaborationConflict) as exc:
        raise _command_error(exc) from exc
    return _envelope(item, request.state.request_id)


def _service(
    request: Request,
    database: Database,
    authorizer: Authorizer,
    dlp: PresidioDlpProcessor | None,
) -> SupplierCollaborationService:
    client: SupplierAgentClient | None = getattr(request.app.state, "supplier_agent_client", None)
    return SupplierCollaborationService(database, authorizer, dlp, client)


def _response(item: SupplierCollaborationView) -> SupplierCollaborationResponse:
    payload = {
        field: getattr(item, field)
        for field in SupplierCollaborationResponse.model_fields
        if field not in {"events", "legal_actions"}
    }
    payload["events"] = [_event(value) for value in item.events]
    payload["legal_actions"] = list(item.legal_actions)
    return SupplierCollaborationResponse.model_validate(payload)


def _event(item: CollaborationEventView) -> SupplierCollaborationEventResponse:
    return SupplierCollaborationEventResponse.model_validate(
        {field: getattr(item, field) for field in SupplierCollaborationEventResponse.model_fields}
    )


def _envelope(item: SupplierCollaborationView, request_id: str) -> SupplierCollaborationEnvelope:
    return SupplierCollaborationEnvelope(data=_response(item), meta={"request_id": request_id})


def _version(value: str) -> int:
    try:
        parsed = int(value.strip().strip('"'))
    except ValueError as exc:
        raise _error(400, "invalid_version_precondition", "validation") from exc
    if parsed <= 0:
        raise _error(400, "invalid_version_precondition", "validation")
    return parsed


def _command_error(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return _error(403, "supplier_collaboration_forbidden", "authorization")
    if isinstance(exc, CollaborationNotVisible):
        return _error(404, "supplier_collaboration_not_found", "not_found")
    if isinstance(exc, CollaborationConflict):
        return _conflict(exc)
    if isinstance(exc, SupplierA2AError):
        category = "security" if exc.reason.endswith(("blocked", "rejected")) else "dependency"
        retryable = any(
            marker in exc.reason for marker in ("unavailable", "timeout", "temporarily")
        )
        return AppError(
            503,
            exc.reason,
            category,
            "Supplier Agent collaboration failed safely",
            retryable=retryable,
        )
    return _error(500, "supplier_collaboration_failed", "internal")


def _conflict(exc: CollaborationConflict) -> AppError:
    if exc.reason == "supplier_a2a_not_configured":
        return AppError(
            503,
            exc.reason,
            "dependency",
            "Supplier Agent collaboration is not configured",
        )
    details = {"current_version": exc.current_version} if exc.current_version else None
    return AppError(
        409, exc.reason, "conflict", "Supplier collaboration state conflict", details=details
    )


def _error(status_code: int, code: str, category: str) -> AppError:
    return AppError(status_code, code, category, "Supplier collaboration request failed")
