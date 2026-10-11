"""IAM-004 break-glass request, approval, revocation, and audit API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.security.emergency_access import (
    EmergencyAccessConflict,
    EmergencyAccessGrantAggregate,
    EmergencyAccessNotVisible,
    EmergencyAccessService,
)

router = APIRouter(tags=["emergency-access"])


class EmergencyAccessRequestBody(BaseModel):
    incident_number: str = Field(min_length=3, max_length=128)
    action: Literal["security_audit.read"]
    resource_id: Literal["security-audit"]
    justification: str = Field(min_length=10, max_length=1000)
    ttl_minutes: int = Field(ge=5, le=60)


class EmergencyAccessDecisionBody(BaseModel):
    decision: Literal["APPROVE", "REJECT"]
    reason: str = Field(min_length=3, max_length=500)


class EmergencyAccessRevokeBody(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class EmergencyAccessDecisionResponse(BaseModel):
    decision_id: str
    decision: str
    actor_subject_id: str
    reason: str
    occurred_at: datetime


class EmergencyAccessUsageResponse(BaseModel):
    usage_id: str
    subject_id: str
    action: str
    resource_id: str
    request_id: str
    occurred_at: datetime


class EmergencyAccessGrantResponse(BaseModel):
    grant_id: str
    incident_number: str
    requester_subject_id: str
    action: str
    resource_id: str
    justification: str
    status: str
    requested_expires_at: datetime
    approved_at: datetime | None
    approver_subject_id: str | None
    revoked_at: datetime | None
    revoked_by_subject_id: str | None
    version: int
    usage_count: int
    usage_history: list[EmergencyAccessUsageResponse]
    decisions: list[EmergencyAccessDecisionResponse]
    legal_actions: list[str]


class EmergencyAccessGrantEnvelope(BaseModel):
    data: EmergencyAccessGrantResponse
    meta: dict[str, str]


class EmergencyAccessGrantListEnvelope(BaseModel):
    data: list[EmergencyAccessGrantResponse]
    meta: dict[str, str | int]


@router.get(
    "/security/emergency-access-grants",
    response_model=EmergencyAccessGrantListEnvelope,
)
async def list_emergency_access_grants(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[str | None, Query()] = None,
) -> EmergencyAccessGrantListEnvelope:
    try:
        grants = EmergencyAccessService(database, authorizer).list_grants(
            identity,
            request_id=request.state.request_id,
            status=status,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "emergency_access_forbidden", str(exc)) from exc
    except EmergencyAccessConflict as exc:
        raise _conflict_error(exc) from exc
    return EmergencyAccessGrantListEnvelope(
        data=[_response(item, identity) for item in grants],
        meta={"request_id": request.state.request_id, "total": len(grants)},
    )


@router.post(
    "/security/emergency-access-grants",
    response_model=EmergencyAccessGrantEnvelope,
    status_code=201,
)
async def request_emergency_access_grant(
    body: EmergencyAccessRequestBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> EmergencyAccessGrantEnvelope:
    try:
        grant = EmergencyAccessService(database, authorizer).request_grant(
            identity,
            incident_number=body.incident_number,
            action=Action(body.action),
            resource_id=body.resource_id,
            justification=body.justification,
            ttl_minutes=body.ttl_minutes,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "emergency_access_forbidden", str(exc)) from exc
    except EmergencyAccessConflict as exc:
        raise _conflict_error(exc) from exc
    return EmergencyAccessGrantEnvelope(
        data=_response(grant, identity),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/security/emergency-access-grants/{grant_id}/decisions",
    response_model=EmergencyAccessGrantEnvelope,
)
async def decide_emergency_access_grant(
    grant_id: str,
    body: EmergencyAccessDecisionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> EmergencyAccessGrantEnvelope:
    try:
        grant = EmergencyAccessService(database, authorizer).decide_grant(
            identity,
            grant_id,
            expected_version=_version(if_match),
            decision=body.decision,
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "emergency_access_forbidden", str(exc)) from exc
    except EmergencyAccessNotVisible as exc:
        raise _error(404, "resource_not_found_or_not_visible", "not_visible") from exc
    except EmergencyAccessConflict as exc:
        raise _conflict_error(exc) from exc
    return EmergencyAccessGrantEnvelope(
        data=_response(grant, identity),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/security/emergency-access-grants/{grant_id}/revoke",
    response_model=EmergencyAccessGrantEnvelope,
)
async def revoke_emergency_access_grant(
    grant_id: str,
    body: EmergencyAccessRevokeBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> EmergencyAccessGrantEnvelope:
    try:
        grant = EmergencyAccessService(database, authorizer).revoke_grant(
            identity,
            grant_id,
            expected_version=_version(if_match),
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "emergency_access_forbidden", str(exc)) from exc
    except EmergencyAccessNotVisible as exc:
        raise _error(404, "resource_not_found_or_not_visible", "not_visible") from exc
    except EmergencyAccessConflict as exc:
        raise _conflict_error(exc) from exc
    return EmergencyAccessGrantEnvelope(
        data=_response(grant, identity),
        meta={"request_id": request.state.request_id},
    )


def _response(
    aggregate: EmergencyAccessGrantAggregate,
    identity: IdentityContext,
) -> EmergencyAccessGrantResponse:
    grant = aggregate.grant
    legal_actions: list[str] = []
    if (
        grant.status == "PENDING"
        and Role.SECURITY_AUDITOR in identity.roles
        and grant.requester_subject_id != identity.subject_id
    ):
        legal_actions.append("DECIDE")
    if grant.status == "APPROVED" and Role.SECURITY_AUDITOR in identity.roles:
        legal_actions.append("REVOKE")
    if grant.status == "APPROVED" and grant.requester_subject_id == identity.subject_id:
        legal_actions.append("USE_SECURITY_AUDIT")
    return EmergencyAccessGrantResponse(
        grant_id=grant.grant_id,
        incident_number=grant.incident_number,
        requester_subject_id=grant.requester_subject_id,
        action=grant.action,
        resource_id=grant.resource_id,
        justification=grant.justification,
        status=grant.status,
        requested_expires_at=_utc(grant.requested_expires_at),
        approved_at=_optional_utc(grant.approved_at),
        approver_subject_id=grant.approver_subject_id,
        revoked_at=_optional_utc(grant.revoked_at),
        revoked_by_subject_id=grant.revoked_by_subject_id,
        version=grant.version,
        usage_count=aggregate.usage_count,
        usage_history=[
            EmergencyAccessUsageResponse(
                usage_id=item.usage_id,
                subject_id=item.subject_id,
                action=item.action,
                resource_id=item.resource_id,
                request_id=item.request_id,
                occurred_at=_utc(item.occurred_at),
            )
            for item in aggregate.usages
        ],
        decisions=[
            EmergencyAccessDecisionResponse(
                decision_id=item.decision_id,
                decision=item.decision,
                actor_subject_id=item.actor_subject_id,
                reason=item.reason,
                occurred_at=_utc(item.occurred_at),
            )
            for item in aggregate.decisions
        ],
        legal_actions=legal_actions,
    )


def _version(value: str) -> int:
    normalized = value.strip()
    if normalized.startswith('W/"') and normalized.endswith('"'):
        normalized = normalized[3:-1]
    elif normalized.startswith('"') and normalized.endswith('"'):
        normalized = normalized[1:-1]
    try:
        version = int(normalized)
    except ValueError as exc:
        raise EmergencyAccessConflict("if_match_invalid") from exc
    if version < 1:
        raise EmergencyAccessConflict("if_match_invalid")
    return version


def _conflict_error(exc: EmergencyAccessConflict) -> AppError:
    status = 409 if exc.reason in {
        "version_conflict",
        "emergency_grant_not_pending",
        "emergency_grant_not_active",
        "emergency_self_approval_denied",
    } else 422
    return _error(status, exc.reason, exc.reason)


def _error(status_code: int, code: str, reason: str) -> AppError:
    del reason
    return AppError(
        status_code=status_code,
        code=code,
        category="authorization" if status_code in {403, 404} else "conflict",
        message="Emergency access request was rejected",
    )


def _optional_utc(value: datetime | None) -> datetime | None:
    return _utc(value) if value is not None else None
