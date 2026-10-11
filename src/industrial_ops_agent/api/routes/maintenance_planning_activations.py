"""Governed activation and rollback API for evaluated maintenance-planning councils."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.api.versions import numeric_precondition as _version
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.maintenance_planning import (
    MaintenancePlanningActivationConflict,
    MaintenancePlanningActivationNotVisible,
    MaintenancePlanningActivationService,
    MaintenancePlanningActivationView,
)
from industrial_ops_agent.persistence.database import Database

router = APIRouter(
    prefix="/maintenance-planning-activations",
    tags=["maintenance-planning-activation"],
)


class CreateMaintenancePlanningActivationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(min_length=3, max_length=128)
    target_environment: Literal["PROJECT_STAGING", "PRODUCTION"]
    reason: str = Field(min_length=8, max_length=1_000)


class DecideMaintenancePlanningActivationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["APPROVE", "REJECT"]
    reason: str = Field(min_length=8, max_length=1_000)


class RollbackMaintenancePlanningActivationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=8, max_length=1_000)


class MaintenancePlanningActivationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activation_id: str
    evaluation_id: str
    suite_id: str
    target_environment: Literal["PROJECT_STAGING", "PRODUCTION"]
    status: Literal["PENDING_APPROVAL", "ACTIVE", "REJECTED", "ROLLED_BACK"]
    evaluation_decision: Literal["PROJECT_CANDIDATE_ELIGIBLE", "PRODUCTION_CANDIDATE_ELIGIBLE"]
    evaluation_report_hash: str
    evaluation_policy_hash: str
    suite_manifest_hash: str
    activation_policy_hash: str
    request_reason: str
    requested_by_subject_id: str
    decided_by_subject_id: str | None
    decision: Literal["APPROVE", "REJECT"] | None
    decision_reason: str | None
    activated_at: datetime | None
    rolled_back_by_subject_id: str | None
    rollback_reason: str | None
    rolled_back_at: datetime | None
    version: int
    legal_actions: list[Literal["APPROVE", "REJECT", "ROLLBACK"]]
    runtime_activation_authorized: bool
    created_at: datetime
    updated_at: datetime


class MaintenancePlanningActivationEnvelope(BaseModel):
    data: MaintenancePlanningActivationResponse
    meta: dict[str, str | bool]


class MaintenancePlanningActivationListEnvelope(BaseModel):
    data: list[MaintenancePlanningActivationResponse]
    meta: dict[str, str | int]


@router.get("", response_model=MaintenancePlanningActivationListEnvelope)
async def list_maintenance_planning_activations(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    target_environment: Annotated[
        Literal["PROJECT_STAGING", "PRODUCTION"] | None,
        Query(),
    ] = None,
) -> MaintenancePlanningActivationListEnvelope:
    try:
        items = await asyncio.to_thread(
            MaintenancePlanningActivationService(database, authorizer).list,
            identity,
            request_id=request.state.request_id,
            target_environment=target_environment,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningActivationNotVisible,
        MaintenancePlanningActivationConflict,
    ) as exc:
        raise _error(exc) from exc
    return MaintenancePlanningActivationListEnvelope(
        data=[_response(item) for item in items],
        meta={"request_id": request.state.request_id, "total": len(items)},
    )


@router.post(
    "",
    response_model=MaintenancePlanningActivationEnvelope,
    status_code=201,
)
async def create_maintenance_planning_activation(
    body: CreateMaintenancePlanningActivationBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=255),
    ],
) -> MaintenancePlanningActivationEnvelope:
    try:
        item, created = await asyncio.to_thread(
            MaintenancePlanningActivationService(database, authorizer).request,
            identity,
            evaluation_id=body.evaluation_id,
            target_environment=body.target_environment,
            reason=body.reason,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningActivationNotVisible,
        MaintenancePlanningActivationConflict,
    ) as exc:
        raise _error(exc) from exc
    return MaintenancePlanningActivationEnvelope(
        data=_response(item),
        meta={"request_id": request.state.request_id, "created": created},
    )


@router.get(
    "/{activation_id}",
    response_model=MaintenancePlanningActivationEnvelope,
)
async def get_maintenance_planning_activation(
    activation_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> MaintenancePlanningActivationEnvelope:
    try:
        item = await asyncio.to_thread(
            MaintenancePlanningActivationService(database, authorizer).get,
            identity,
            activation_id,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningActivationNotVisible,
        MaintenancePlanningActivationConflict,
    ) as exc:
        raise _error(exc) from exc
    return _envelope(item, request.state.request_id)


@router.post(
    "/{activation_id}/decision",
    response_model=MaintenancePlanningActivationEnvelope,
)
async def decide_maintenance_planning_activation(
    activation_id: str,
    body: DecideMaintenancePlanningActivationBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> MaintenancePlanningActivationEnvelope:
    try:
        item = await asyncio.to_thread(
            MaintenancePlanningActivationService(database, authorizer).decide,
            identity,
            activation_id,
            decision=body.decision,
            reason=body.reason,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningActivationNotVisible,
        MaintenancePlanningActivationConflict,
    ) as exc:
        raise _error(exc) from exc
    return _envelope(item, request.state.request_id)


@router.post(
    "/{activation_id}/rollback",
    response_model=MaintenancePlanningActivationEnvelope,
)
async def rollback_maintenance_planning_activation(
    activation_id: str,
    body: RollbackMaintenancePlanningActivationBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> MaintenancePlanningActivationEnvelope:
    try:
        item = await asyncio.to_thread(
            MaintenancePlanningActivationService(database, authorizer).rollback,
            identity,
            activation_id,
            reason=body.reason,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningActivationNotVisible,
        MaintenancePlanningActivationConflict,
    ) as exc:
        raise _error(exc) from exc
    return _envelope(item, request.state.request_id)


def _response(item: MaintenancePlanningActivationView) -> MaintenancePlanningActivationResponse:
    return MaintenancePlanningActivationResponse(
        activation_id=item.activation_id,
        evaluation_id=item.evaluation_id,
        suite_id=item.suite_id,
        target_environment=item.target_environment,
        status=item.status,
        evaluation_decision=item.evaluation_decision,
        evaluation_report_hash=item.evaluation_report_hash,
        evaluation_policy_hash=item.evaluation_policy_hash,
        suite_manifest_hash=item.suite_manifest_hash,
        activation_policy_hash=item.activation_policy_hash,
        request_reason=item.request_reason,
        requested_by_subject_id=item.requested_by_subject_id,
        decided_by_subject_id=item.decided_by_subject_id,
        decision=item.decision,
        decision_reason=item.decision_reason,
        activated_at=item.activated_at,
        rolled_back_by_subject_id=item.rolled_back_by_subject_id,
        rollback_reason=item.rollback_reason,
        rolled_back_at=item.rolled_back_at,
        version=item.version,
        legal_actions=list(item.legal_actions),
        runtime_activation_authorized=item.status == "ACTIVE",
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def _envelope(
    item: MaintenancePlanningActivationView,
    request_id: str,
) -> MaintenancePlanningActivationEnvelope:
    return MaintenancePlanningActivationEnvelope(
        data=_response(item),
        meta={"request_id": request_id},
    )



def _error(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(
            403,
            "maintenance_planning_activation_forbidden",
            "authorization",
            "Forbidden",
        )
    if isinstance(exc, MaintenancePlanningActivationNotVisible):
        return AppError(
            404,
            "maintenance_planning_activation_not_found",
            "visibility",
            "Not found",
        )
    if isinstance(exc, MaintenancePlanningActivationConflict):
        details = (
            {"current_version": exc.current_version} if exc.current_version is not None else None
        )
        return AppError(
            409,
            exc.reason,
            "conflict",
            "Maintenance planning activation conflicts with current evidence",
            details=details,
        )
    return AppError(
        500,
        "maintenance_planning_activation_failed",
        "internal",
        "Request failed",
    )
