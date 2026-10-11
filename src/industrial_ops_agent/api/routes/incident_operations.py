"""INC-001 incident intake queue and manual control API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.incident_operations.service import (
    IncidentControlView,
    IncidentNotVisible,
    IncidentOperationConflict,
    IncidentOperationsService,
    IncidentOperationsView,
    IncidentQueueItem,
    IncidentResolutionView,
)
from industrial_ops_agent.persistence.database import Database

router = APIRouter(prefix="/incidents", tags=["incident-operations"])


class IncidentOperationsResponse(BaseModel):
    incident_id: str
    asset_id: str
    description: str
    status: str
    severity: str | None
    category: str | None
    responsible_queue: str | None
    information_due_at: datetime | None
    escalation_responsible_subject_id: str | None
    version: int
    created_at: datetime
    updated_at: datetime
    legal_actions: list[str]


class IncidentQueueItemResponse(IncidentOperationsResponse):
    asset_display_name: str | None
    model_code: str | None
    site_id: str | None
    site_name: str | None


class IncidentQueueEnvelope(BaseModel):
    data: list[IncidentQueueItemResponse]
    meta: dict[str, str | int]


class IncidentOperationsEnvelope(BaseModel):
    data: IncidentOperationsResponse
    meta: dict[str, str | int]


class IncidentControlResponse(BaseModel):
    control_id: str
    incident_version: int
    command_type: str
    previous_status: str
    target_status: str
    reason: str
    actor_subject_id: str
    responsible_subject_id: str | None
    recovery_condition: str | None
    related_incident_id: str | None
    details: dict[str, Any]
    occurred_at: datetime


class IncidentControlsEnvelope(BaseModel):
    data: list[IncidentControlResponse]
    meta: dict[str, str]


class IncidentResolutionResponse(BaseModel):
    resolution_id: str
    incident_id: str
    mode: Literal["REMOTE", "WORK_ORDER"]
    diagnosis_run_id: str | None
    work_order_id: str | None
    completion_id: str | None
    verification_id: str | None
    summary: str
    evidence_ids: list[str]
    resolved_by_subject_id: str
    resolved_at: datetime


class IncidentResolutionEnvelope(BaseModel):
    data: IncidentResolutionResponse | None
    meta: dict[str, str]


class ClassifyIncidentRequest(BaseModel):
    severity: Literal["P1", "P2", "P3", "P4"]
    category: str = Field(min_length=1, max_length=64)
    responsible_queue: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=2_000)


class RequestInformationRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2_000)
    information_due_at: datetime


class ReasonRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2_000)


class EscalateIncidentRequest(ReasonRequest):
    responsible_subject_id: str = Field(min_length=1, max_length=128)
    recovery_condition: str = Field(min_length=1, max_length=2_000)


class ResumeIncidentRequest(ReasonRequest):
    return_status: Literal["TRIAGED", "DIAGNOSING"]


class MergeDuplicateRequest(ReasonRequest):
    canonical_incident_id: str = Field(min_length=1, max_length=128)


class ResolveRemoteIncidentRequest(BaseModel):
    summary: str = Field(min_length=1, max_length=4_000)
    evidence_ids: list[str] = Field(min_length=1, max_length=64)


class CloseIncidentRequest(ReasonRequest):
    confirmation_type: Literal["CUSTOMER", "EXPERT"]
    customer_update_id: str | None = Field(default=None, min_length=1, max_length=128)


@router.get("", response_model=IncidentQueueEnvelope, responses=STANDARD_ERROR_RESPONSES)
def list_incident_queue(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[str | None, Query(max_length=32)] = None,
    severity: Annotated[str | None, Query(max_length=16)] = None,
    responsible_queue: Annotated[str | None, Query(max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> IncidentQueueEnvelope:
    items, total = IncidentOperationsService(database, authorizer).list_queue(
        identity,
        status=status,
        severity=severity,
        responsible_queue=responsible_queue,
        limit=limit,
        offset=offset,
        request_id=request.state.request_id,
    )
    return IncidentQueueEnvelope(
        data=[_queue_response(item) for item in items],
        meta={"request_id": request.state.request_id, "total": total},
    )


@router.get(
    "/{incident_id}/operations",
    response_model=IncidentOperationsEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_incident_operations(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> IncidentOperationsEnvelope:
    return _run_command(
        request,
        lambda: IncidentOperationsService(database, authorizer).get(
            identity,
            incident_id,
            request_id=request.state.request_id,
        ),
    )


@router.get(
    "/{incident_id}/resolution",
    response_model=IncidentResolutionEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_incident_resolution(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> IncidentResolutionEnvelope:
    try:
        resolution = IncidentOperationsService(database, authorizer).get_resolution(
            identity,
            incident_id,
            request_id=request.state.request_id,
        )
    except IncidentNotVisible as exc:
        raise _not_visible() from exc
    return IncidentResolutionEnvelope(
        data=_resolution_response(resolution) if resolution is not None else None,
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/{incident_id}/resolution/remote",
    response_model=IncidentOperationsEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def resolve_incident_remote(
    incident_id: str,
    payload: ResolveRemoteIncidentRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> IncidentOperationsEnvelope:
    return _run_command(
        request,
        lambda: IncidentOperationsService(database, authorizer).resolve_remote(
            identity,
            incident_id,
            summary=payload.summary,
            evidence_ids=payload.evidence_ids,
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        ),
    )


@router.post(
    "/{incident_id}/closure",
    response_model=IncidentOperationsEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def close_incident(
    incident_id: str,
    payload: CloseIncidentRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> IncidentOperationsEnvelope:
    return _run_command(
        request,
        lambda: IncidentOperationsService(database, authorizer).close(
            identity,
            incident_id,
            confirmation_type=payload.confirmation_type,
            customer_update_id=payload.customer_update_id,
            reason=payload.reason,
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        ),
    )


@router.get(
    "/{incident_id}/controls",
    response_model=IncidentControlsEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def list_incident_controls(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> IncidentControlsEnvelope:
    try:
        controls = IncidentOperationsService(database, authorizer).list_controls(
            identity, incident_id, request_id=request.state.request_id
        )
    except IncidentNotVisible as exc:
        raise _not_visible() from exc
    return IncidentControlsEnvelope(
        data=[_control_response(item) for item in controls],
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/{incident_id}/classification",
    response_model=IncidentOperationsEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def classify_incident(
    incident_id: str,
    payload: ClassifyIncidentRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> IncidentOperationsEnvelope:
    return _run_command(
        request,
        lambda: IncidentOperationsService(database, authorizer).classify(
            identity,
            incident_id,
            severity=payload.severity,
            category=payload.category,
            responsible_queue=payload.responsible_queue,
            reason=payload.reason,
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        ),
    )


@router.post(
    "/{incident_id}/request-information",
    response_model=IncidentOperationsEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def request_incident_information(
    incident_id: str,
    payload: RequestInformationRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> IncidentOperationsEnvelope:
    return _run_command(
        request,
        lambda: IncidentOperationsService(database, authorizer).request_information(
            identity,
            incident_id,
            reason=payload.reason,
            information_due_at=payload.information_due_at,
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        ),
    )


@router.post(
    "/{incident_id}/information-received",
    response_model=IncidentOperationsEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def mark_incident_information_received(
    incident_id: str,
    payload: ReasonRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> IncidentOperationsEnvelope:
    return _run_command(
        request,
        lambda: IncidentOperationsService(database, authorizer).mark_information_received(
            identity,
            incident_id,
            reason=payload.reason,
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        ),
    )


@router.post(
    "/{incident_id}/escalation",
    response_model=IncidentOperationsEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def escalate_incident(
    incident_id: str,
    payload: EscalateIncidentRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> IncidentOperationsEnvelope:
    return _run_command(
        request,
        lambda: IncidentOperationsService(database, authorizer).escalate(
            identity,
            incident_id,
            reason=payload.reason,
            responsible_subject_id=payload.responsible_subject_id,
            recovery_condition=payload.recovery_condition,
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        ),
    )


@router.post(
    "/{incident_id}/resume",
    response_model=IncidentOperationsEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def resume_incident(
    incident_id: str,
    payload: ResumeIncidentRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> IncidentOperationsEnvelope:
    return _run_command(
        request,
        lambda: IncidentOperationsService(database, authorizer).resume(
            identity,
            incident_id,
            return_status=payload.return_status,
            reason=payload.reason,
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        ),
    )


@router.post(
    "/{incident_id}/duplicate-merge",
    response_model=IncidentOperationsEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def merge_duplicate_incident(
    incident_id: str,
    payload: MergeDuplicateRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> IncidentOperationsEnvelope:
    return _run_command(
        request,
        lambda: IncidentOperationsService(database, authorizer).merge_duplicate(
            identity,
            incident_id,
            canonical_incident_id=payload.canonical_incident_id,
            reason=payload.reason,
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        ),
    )


def _run_command(request: Request, command: Any) -> IncidentOperationsEnvelope:
    try:
        incident = command()
    except IncidentNotVisible as exc:
        raise _not_visible() from exc
    except IncidentOperationConflict as exc:
        details = (
            {"current_version": exc.current_version} if exc.current_version is not None else None
        )
        raise AppError(
            status_code=409,
            code="incident_operation_conflict",
            category="conflict",
            message=exc.reason,
            details=details,
        ) from exc
    return IncidentOperationsEnvelope(
        data=_operations_response(incident),
        meta={"request_id": request.state.request_id, "version": incident.version},
    )


def _parse_version(value: str) -> int:
    try:
        version = int(value.strip().strip('"'))
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="invalid_version_precondition",
            category="validation",
            message="If-Match must contain a positive incident version",
        ) from exc
    if version < 1:
        raise AppError(
            status_code=400,
            code="invalid_version_precondition",
            category="validation",
            message="If-Match must contain a positive incident version",
        )
    return version


def _operations_response(item: IncidentOperationsView) -> IncidentOperationsResponse:
    return IncidentOperationsResponse(
        incident_id=item.incident_id,
        asset_id=item.asset_id,
        description=item.description,
        status=item.status,
        severity=item.severity,
        category=item.category,
        responsible_queue=item.responsible_queue,
        information_due_at=item.information_due_at,
        escalation_responsible_subject_id=item.escalation_responsible_subject_id,
        version=item.version,
        created_at=item.created_at,
        updated_at=item.updated_at,
        legal_actions=list(item.legal_actions),
    )


def _queue_response(item: IncidentQueueItem) -> IncidentQueueItemResponse:
    incident = item.incident
    return IncidentQueueItemResponse(
        **_operations_response(incident).model_dump(),
        asset_display_name=item.asset_display_name,
        model_code=item.model_code,
        site_id=item.site_id,
        site_name=item.site_name,
    )


def _control_response(item: IncidentControlView) -> IncidentControlResponse:
    return IncidentControlResponse(
        control_id=item.control_id,
        incident_version=item.incident_version,
        command_type=item.command_type,
        previous_status=item.previous_status,
        target_status=item.target_status,
        reason=item.reason,
        actor_subject_id=item.actor_subject_id,
        responsible_subject_id=item.responsible_subject_id,
        recovery_condition=item.recovery_condition,
        related_incident_id=item.related_incident_id,
        details=item.details,
        occurred_at=item.occurred_at,
    )


def _resolution_response(item: IncidentResolutionView) -> IncidentResolutionResponse:
    return IncidentResolutionResponse(
        resolution_id=item.resolution_id,
        incident_id=item.incident_id,
        mode=item.mode,
        diagnosis_run_id=item.diagnosis_run_id,
        work_order_id=item.work_order_id,
        completion_id=item.completion_id,
        verification_id=item.verification_id,
        summary=item.summary,
        evidence_ids=list(item.evidence_ids),
        resolved_by_subject_id=item.resolved_by_subject_id,
        resolved_at=item.resolved_at,
    )


def _not_visible() -> AppError:
    return AppError(
        status_code=404,
        code="resource_not_found_or_not_visible",
        category="not_found",
        message="Resource not found or not visible",
    )
