"""Review-only multi-agent maintenance planning council API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_maintenance_planning_binding_resolver,
    get_maintenance_planning_dispatcher,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.api.versions import numeric_precondition as _version
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.maintenance_planning import (
    MaintenancePlanningActivityInput,
    MaintenancePlanningBindingResolver,
    MaintenancePlanningConflict,
    MaintenancePlanningDispatcher,
    MaintenancePlanningDispatchUnavailable,
    MaintenancePlanningNotVisible,
    MaintenancePlanningService,
    MaintenancePlanningView,
)
from industrial_ops_agent.persistence.database import Database

router = APIRouter(
    prefix="/maintenance-planning-councils",
    tags=["maintenance-planning"],
)


class CreateMaintenancePlanningBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(min_length=3, max_length=128)
    diagnosis_run_id: str = Field(min_length=3, max_length=128)


class ReviewMaintenancePlanningBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["ACCEPT", "REJECT"]
    reason: str = Field(min_length=8, max_length=1_000)


class MaintenancePlanningContributionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contribution_id: str
    attempt_number: int
    agent_role: Literal["SAFETY", "PARTS", "DISPATCH", "COORDINATOR"]
    inference_request_id: str
    input_digest: str
    output: dict[str, Any]
    output_digest: str
    model_release_id: str
    model_manifest_hash: str
    context_hash: str
    completed_at: datetime


class MaintenancePlanningResponse(BaseModel):
    council_id: str
    incident_id: str
    diagnosis_run_id: str
    asset_id: str
    diagnosis_version: int
    diagnosis_report_digest: str
    diagnosis_manifest_digest: str
    workflow_id: str
    model_alias: str
    model_release_id: str
    model_manifest_hash: str
    prompt_bundle_id: str
    prompt_bundle_hash: str
    response_schema_version: str
    activation_id: str | None
    activation_policy_hash: str | None
    activation_target_environment: Literal["PROJECT_STAGING", "PRODUCTION"] | None
    status: Literal[
        "QUEUED",
        "RUNNING",
        "REVIEW_PENDING",
        "ACCEPTED",
        "REJECTED",
        "FAILED",
    ]
    stage: str
    attempt_count: int
    plan: dict[str, Any] | None
    result_digest: str | None
    coordinator_context_hash: str | None
    failure_code: str | None
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
    contributions: list[MaintenancePlanningContributionResponse]
    legal_actions: list[Literal["RETRY", "REVIEW_ACCEPT", "REVIEW_REJECT"]]
    advisory_only: Literal[True] = True
    business_side_effects_created: Literal[False] = False


class MaintenancePlanningEnvelope(BaseModel):
    data: MaintenancePlanningResponse
    meta: dict[str, str]


class MaintenancePlanningListEnvelope(BaseModel):
    data: list[MaintenancePlanningResponse]
    meta: dict[str, str | int]


@router.get("", response_model=MaintenancePlanningListEnvelope)
async def list_maintenance_planning_councils(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    incident_id: Annotated[str | None, Query(min_length=3, max_length=128)] = None,
) -> MaintenancePlanningListEnvelope:
    try:
        items = await asyncio.to_thread(
            MaintenancePlanningService(database, authorizer).list,
            identity,
            request_id=request.state.request_id,
            incident_id=incident_id,
        )
    except (AuthorizationDenied, MaintenancePlanningNotVisible) as exc:
        raise _error(exc) from exc
    return MaintenancePlanningListEnvelope(
        data=[_response(item, identity, authorizer) for item in items],
        meta={"request_id": request.state.request_id, "total": len(items)},
    )


@router.post("", response_model=MaintenancePlanningEnvelope, status_code=202)
async def create_maintenance_planning_council(
    body: CreateMaintenancePlanningBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    resolver: Annotated[
        MaintenancePlanningBindingResolver,
        Depends(get_maintenance_planning_binding_resolver),
    ],
    dispatcher: Annotated[
        MaintenancePlanningDispatcher,
        Depends(get_maintenance_planning_dispatcher),
    ],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=255),
    ],
) -> MaintenancePlanningEnvelope:
    service = MaintenancePlanningService(database, authorizer)
    try:
        binding = await asyncio.to_thread(resolver.resolve, identity)
        item = await asyncio.to_thread(
            service.request,
            identity,
            body.incident_id,
            diagnosis_run_id=body.diagnosis_run_id,
            idempotency_key=idempotency_key,
            binding=binding,
            request_id=request.state.request_id,
        )
        if item.dispatch_required:
            await _dispatch(service, dispatcher, identity, item, request.state.request_id)
    except (
        AuthorizationDenied,
        MaintenancePlanningNotVisible,
        MaintenancePlanningConflict,
        MaintenancePlanningDispatchUnavailable,
    ) as exc:
        raise _error(exc) from exc
    return _envelope(item, request.state.request_id, identity, authorizer)


@router.get("/{council_id}", response_model=MaintenancePlanningEnvelope)
async def get_maintenance_planning_council(
    council_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> MaintenancePlanningEnvelope:
    try:
        item = await asyncio.to_thread(
            MaintenancePlanningService(database, authorizer).get,
            identity,
            council_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, MaintenancePlanningNotVisible) as exc:
        raise _error(exc) from exc
    return _envelope(item, request.state.request_id, identity, authorizer)


@router.post(
    "/{council_id}/retry",
    response_model=MaintenancePlanningEnvelope,
    status_code=202,
)
async def retry_maintenance_planning_council(
    council_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[
        MaintenancePlanningDispatcher,
        Depends(get_maintenance_planning_dispatcher),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> MaintenancePlanningEnvelope:
    service = MaintenancePlanningService(database, authorizer)
    try:
        item = await asyncio.to_thread(
            service.retry,
            identity,
            council_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
        await _dispatch(service, dispatcher, identity, item, request.state.request_id)
    except (
        AuthorizationDenied,
        MaintenancePlanningNotVisible,
        MaintenancePlanningConflict,
        MaintenancePlanningDispatchUnavailable,
    ) as exc:
        raise _error(exc) from exc
    return _envelope(item, request.state.request_id, identity, authorizer)


@router.post("/{council_id}/review", response_model=MaintenancePlanningEnvelope)
async def review_maintenance_planning_council(
    council_id: str,
    body: ReviewMaintenancePlanningBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> MaintenancePlanningEnvelope:
    try:
        item = await asyncio.to_thread(
            MaintenancePlanningService(database, authorizer).review,
            identity,
            council_id,
            decision=body.decision,
            reason=body.reason,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningNotVisible,
        MaintenancePlanningConflict,
    ) as exc:
        raise _error(exc) from exc
    return _envelope(item, request.state.request_id, identity, authorizer)


async def _dispatch(
    service: MaintenancePlanningService,
    dispatcher: MaintenancePlanningDispatcher,
    identity: IdentityContext,
    item: MaintenancePlanningView,
    request_id: str,
) -> None:
    try:
        await dispatcher.dispatch(
            MaintenancePlanningActivityInput(
                council_id=item.council_id,
                workflow_id=item.workflow_id,
                identity=identity,
                request_id=request_id,
            )
        )
    except MaintenancePlanningDispatchUnavailable:
        await asyncio.to_thread(
            service.mark_dispatch_failed,
            identity,
            item.council_id,
            workflow_id=item.workflow_id,
        )
        raise


def _response(
    item: MaintenancePlanningView,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> MaintenancePlanningResponse:
    resource = ResourceContext(
        tenant_id=identity.tenant_id,
        resource_id=item.council_id,
        asset_id=item.asset_id,
    )
    actions: list[str] = []
    if (
        "RETRY" in item.legal_actions
        and authorizer.decide(identity, Action.REQUEST_MAINTENANCE_COUNCIL, resource).allowed
    ):
        actions.append("RETRY")
    if (
        "REVIEW_ACCEPT" in item.legal_actions
        and item.requested_by_subject_id != identity.subject_id
        and authorizer.decide(identity, Action.REVIEW_MAINTENANCE_COUNCIL, resource).allowed
    ):
        actions.extend(("REVIEW_ACCEPT", "REVIEW_REJECT"))
    return MaintenancePlanningResponse(
        council_id=item.council_id,
        incident_id=item.incident_id,
        diagnosis_run_id=item.diagnosis_run_id,
        asset_id=item.asset_id,
        diagnosis_version=item.diagnosis_version,
        diagnosis_report_digest=item.diagnosis_report_digest,
        diagnosis_manifest_digest=item.diagnosis_manifest_digest,
        workflow_id=item.workflow_id,
        model_alias=item.model_alias,
        model_release_id=item.model_release_id,
        model_manifest_hash=item.model_manifest_hash,
        prompt_bundle_id=item.prompt_bundle_id,
        prompt_bundle_hash=item.prompt_bundle_hash,
        response_schema_version=item.response_schema_version,
        activation_id=item.activation_id,
        activation_policy_hash=item.activation_policy_hash,
        activation_target_environment=item.activation_target_environment,
        status=item.status,
        stage=item.stage,
        attempt_count=item.attempt_count,
        plan=item.plan,
        result_digest=item.result_digest,
        coordinator_context_hash=item.coordinator_context_hash,
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
        contributions=[
            MaintenancePlanningContributionResponse(
                contribution_id=value.contribution_id,
                attempt_number=value.attempt_number,
                agent_role=value.agent_role,
                inference_request_id=value.inference_request_id,
                input_digest=value.input_digest,
                output=value.output,
                output_digest=value.output_digest,
                model_release_id=value.model_release_id,
                model_manifest_hash=value.model_manifest_hash,
                context_hash=value.context_hash,
                completed_at=value.completed_at,
            )
            for value in item.contributions
        ],
        legal_actions=actions,
    )


def _envelope(
    item: MaintenancePlanningView,
    request_id: str,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> MaintenancePlanningEnvelope:
    return MaintenancePlanningEnvelope(
        data=_response(item, identity, authorizer),
        meta={"request_id": request_id},
    )



def _error(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(403, "maintenance_planning_forbidden", "authorization", "Forbidden")
    if isinstance(exc, MaintenancePlanningNotVisible):
        return AppError(404, "maintenance_planning_not_found", "visibility", "Not found")
    if isinstance(exc, MaintenancePlanningDispatchUnavailable):
        return AppError(
            503,
            "maintenance_planning_workflow_unavailable",
            "dependency",
            "Maintenance planning workflow unavailable",
            retryable=True,
        )
    if isinstance(exc, MaintenancePlanningConflict):
        details = (
            {"current_version": exc.current_version} if exc.current_version is not None else None
        )
        return AppError(
            409,
            exc.reason,
            "conflict",
            "Maintenance planning command conflicts with current state",
            retryable=exc.reason
            in {
                "maintenance_planning_model_unavailable",
            },
            details=details,
        )
    return AppError(500, "maintenance_planning_failed", "internal", "Request failed")
