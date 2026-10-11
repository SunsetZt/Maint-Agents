"""External side-effect reconciliation queries and guarded commands."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_identity, get_tool_gateway
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.persistence.models import (
    ExecutionReconciliationEventRecord,
    ExecutionReconciliationRecord,
)
from industrial_ops_agent.tools.gateway import (
    ExecutionReconciliationConflict,
    ExecutionReconciliationNotVisible,
    ToolGateway,
)

router = APIRouter(prefix="/execution-reconciliations", tags=["execution-reconciliations"])


class EscalationBody(BaseModel):
    reason: str = Field(min_length=10, max_length=255)


class ExecutionReconciliationResponse(BaseModel):
    reconciliation_id: str
    proposal_id: str
    approval_id: str
    incident_id: str
    initial_execution_attempt_id: str
    resolved_execution_attempt_id: str | None
    operation_id: str
    tool_id: str
    status: str
    outcome: str
    reason: str | None
    external_reference_id: str | None
    version: int
    last_checked_at: datetime | None
    resolved_at: datetime | None
    created_at: datetime
    legal_actions: list[str]


class ExecutionReconciliationEventResponse(BaseModel):
    event_id: str
    sequence: int
    event_type: str
    actor_subject_id: str
    reason: str | None
    outcome: str
    created_at: datetime


class ExecutionReconciliationListEnvelope(BaseModel):
    data: list[ExecutionReconciliationResponse]
    meta: dict[str, str | int]


class ExecutionReconciliationEnvelope(BaseModel):
    data: ExecutionReconciliationResponse
    meta: dict[str, str]


class ExecutionReconciliationDetail(BaseModel):
    reconciliation: ExecutionReconciliationResponse
    events: list[ExecutionReconciliationEventResponse]


class ExecutionReconciliationDetailEnvelope(BaseModel):
    data: ExecutionReconciliationDetail
    meta: dict[str, str]


@router.get("", response_model=ExecutionReconciliationListEnvelope)
def list_execution_reconciliations(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[
        Literal["PENDING", "RESOLVED", "ESCALATED"] | None,
        Query(),
    ] = None,
) -> ExecutionReconciliationListEnvelope:
    try:
        records = gateway.list_execution_reconciliations(
            identity,
            status=status,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    return ExecutionReconciliationListEnvelope(
        data=[_reconciliation(item, identity, authorizer) for item in records],
        meta={"request_id": request.state.request_id, "total": len(records)},
    )


@router.get("/{reconciliation_id}", response_model=ExecutionReconciliationDetailEnvelope)
def get_execution_reconciliation(
    reconciliation_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ExecutionReconciliationDetailEnvelope:
    try:
        record, events = gateway.get_execution_reconciliation(
            identity,
            reconciliation_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ExecutionReconciliationNotVisible as exc:
        raise _not_found() from exc
    return ExecutionReconciliationDetailEnvelope(
        data=ExecutionReconciliationDetail(
            reconciliation=_reconciliation(record, identity, authorizer),
            events=[_event(item) for item in events],
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post("/{reconciliation_id}/check", response_model=ExecutionReconciliationEnvelope)
def check_execution_reconciliation(
    reconciliation_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ExecutionReconciliationEnvelope:
    try:
        record = gateway.reconcile_execution(
            identity,
            reconciliation_id=reconciliation_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ExecutionReconciliationNotVisible as exc:
        raise _not_found() from exc
    except ExecutionReconciliationConflict as exc:
        raise _conflict() from exc
    return ExecutionReconciliationEnvelope(
        data=_reconciliation(record, identity, authorizer),
        meta={"request_id": request.state.request_id},
    )


@router.post("/{reconciliation_id}/escalate", response_model=ExecutionReconciliationEnvelope)
def escalate_execution_reconciliation(
    reconciliation_id: str,
    body: EscalationBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ExecutionReconciliationEnvelope:
    try:
        record = gateway.escalate_execution_reconciliation(
            identity,
            reconciliation_id=reconciliation_id,
            expected_version=_version(if_match),
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ExecutionReconciliationNotVisible as exc:
        raise _not_found() from exc
    except ExecutionReconciliationConflict as exc:
        raise _conflict() from exc
    return ExecutionReconciliationEnvelope(
        data=_reconciliation(record, identity, authorizer),
        meta={"request_id": request.state.request_id},
    )


def _reconciliation(
    record: ExecutionReconciliationRecord,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> ExecutionReconciliationResponse:
    can_reconcile = (
        record.status == "PENDING"
        and authorizer.decide(
            identity,
            Action.RECONCILE_EXECUTION,
            ResourceContext(identity.tenant_id, record.reconciliation_id),
        ).allowed
    )
    return ExecutionReconciliationResponse(
        reconciliation_id=record.reconciliation_id,
        proposal_id=record.proposal_id,
        approval_id=record.approval_id,
        incident_id=record.incident_id,
        initial_execution_attempt_id=record.initial_execution_attempt_id,
        resolved_execution_attempt_id=record.resolved_execution_attempt_id,
        operation_id=record.operation_id,
        tool_id=record.tool_id,
        status=record.status,
        outcome=record.outcome,
        reason=record.reason,
        external_reference_id=record.external_reference_id,
        version=record.version,
        last_checked_at=_optional_utc(record.last_checked_at),
        resolved_at=_optional_utc(record.resolved_at),
        created_at=_utc(record.created_at),
        legal_actions=["CHECK", "ESCALATE"] if can_reconcile else [],
    )


def _event(record: ExecutionReconciliationEventRecord) -> ExecutionReconciliationEventResponse:
    return ExecutionReconciliationEventResponse(
        event_id=record.event_id,
        sequence=record.sequence,
        event_type=record.event_type,
        actor_subject_id=record.actor_subject_id,
        reason=record.reason,
        outcome=record.outcome,
        created_at=_utc(record.created_at),
    )


def _version(value: str) -> int:
    try:
        return int(value.strip('W/"'))
    except ValueError as exc:
        raise AppError(
            400, "invalid_if_match", "validation", "If-Match must be an integer"
        ) from exc


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _optional_utc(value: datetime | None) -> datetime | None:
    return _utc(value) if value is not None else None


def _forbidden() -> AppError:
    return AppError(403, "execution_reconciliation_forbidden", "authorization", "Forbidden")


def _not_found() -> AppError:
    return AppError(
        404,
        "execution_reconciliation_not_found",
        "not_found",
        "Execution reconciliation not found or not visible",
    )


def _conflict() -> AppError:
    return AppError(
        409,
        "execution_reconciliation_conflict",
        "conflict",
        "Execution reconciliation state or version changed",
    )
