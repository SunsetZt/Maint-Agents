"""Read-only security audit and Guardrail operations workspace API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import SecurityAuditEventRecord
from industrial_ops_agent.security.service import SecurityAuditService

router = APIRouter(tags=["security"])


class SecurityAuditEventResponse(BaseModel):
    event_id: str
    occurred_at: datetime
    subject_hash: str
    tenant_hash: str
    action: str
    decision: str
    reason_code: str
    request_id: str
    resource_hash: str | None


class SecurityAuditListMeta(BaseModel):
    request_id: str
    next_cursor: str | None


class SecurityAuditListEnvelope(BaseModel):
    data: list[SecurityAuditEventResponse]
    meta: SecurityAuditListMeta


class SecurityAuditSummaryResponse(BaseModel):
    window_start: datetime
    window_end: datetime
    total_events: int
    allowed_events: int
    denied_events: int
    guardrail_blocks: int
    action_counts: dict[str, int]


class SecurityAuditSummaryEnvelope(BaseModel):
    data: SecurityAuditSummaryResponse
    meta: dict[str, str]


@router.get("/security/audit-events", response_model=SecurityAuditListEnvelope)
async def list_security_audit_events(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: str | None = None,
    decision: Annotated[str | None, Query(pattern="^(allow|deny)$")] = None,
    action: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    emergency_grant_id: Annotated[
        str | None,
        Header(alias="X-Emergency-Grant-ID", min_length=1, max_length=128),
    ] = None,
) -> SecurityAuditListEnvelope:
    del emergency_grant_id
    try:
        page = SecurityAuditService(database, authorizer).list_events(
            identity,
            request_id=request.state.request_id,
            limit=limit,
            cursor=cursor,
            decision=decision,
            action=action,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "security_audit_forbidden") from exc
    except ValueError as exc:
        raise _error(422, "security_audit_query_invalid") from exc
    return SecurityAuditListEnvelope(
        data=[_event(record) for record in page.events],
        meta=SecurityAuditListMeta(
            request_id=request.state.request_id,
            next_cursor=page.next_cursor,
        ),
    )


@router.get("/security/summary", response_model=SecurityAuditSummaryEnvelope)
async def get_security_audit_summary(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    window_hours: Annotated[int, Query(ge=1, le=24 * 31)] = 24,
    emergency_grant_id: Annotated[
        str | None,
        Header(alias="X-Emergency-Grant-ID", min_length=1, max_length=128),
    ] = None,
) -> SecurityAuditSummaryEnvelope:
    del emergency_grant_id
    try:
        summary = SecurityAuditService(database, authorizer).summarize(
            identity,
            request_id=request.state.request_id,
            window_hours=window_hours,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "security_audit_forbidden") from exc
    except ValueError as exc:
        raise _error(422, "security_audit_query_invalid") from exc
    return SecurityAuditSummaryEnvelope(
        data=SecurityAuditSummaryResponse(
            window_start=summary.window_start,
            window_end=summary.window_end,
            total_events=summary.total_events,
            allowed_events=summary.allowed_events,
            denied_events=summary.denied_events,
            guardrail_blocks=summary.guardrail_blocks,
            action_counts=summary.action_counts,
        ),
        meta={"request_id": request.state.request_id},
    )


def _event(record: SecurityAuditEventRecord) -> SecurityAuditEventResponse:
    return SecurityAuditEventResponse(
        event_id=record.event_id,
        occurred_at=record.created_at,
        subject_hash=record.subject_hash,
        tenant_hash=record.tenant_hash,
        action=record.action,
        decision=record.decision,
        reason_code=record.reason,
        request_id=record.request_id,
        resource_hash=record.resource_hash,
    )


def _error(status_code: int, code: str) -> AppError:
    return AppError(
        status_code=status_code,
        code=code,
        category="authorization" if status_code == 403 else "validation",
        message="Security audit request was rejected",
    )
