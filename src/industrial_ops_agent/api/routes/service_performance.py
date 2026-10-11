"""Read-only after-sales business performance API."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.service_performance.service import (
    ServicePerformanceBaseline,
    ServicePerformanceBaselineConflict,
    ServicePerformanceBaselineInvalid,
    ServicePerformanceBaselineMetrics,
    ServicePerformanceBaselineNotFound,
    ServicePerformanceBaselineSamples,
    ServicePerformanceBaselineTargets,
    ServicePerformanceDataQuality,
    ServicePerformanceQueryInvalid,
    ServicePerformanceService,
    ServicePerformanceSnapshotV3,
    ServicePerformanceTargetComparison,
)

router = APIRouter(prefix="/operations", tags=["service-performance"])


class _AttributeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ServicePerformanceSummaryResponse(_AttributeResponse):
    closed_work_order_count: int
    first_time_fix_eligible_count: int
    first_time_fix_count: int
    first_time_fix_rate: float | None
    reworked_work_order_count: int
    rework_rate: float | None
    average_repair_rounds: float | None
    mttr_sample_count: int
    mean_mttr_minutes: float | None
    p50_mttr_minutes: float | None
    p90_mttr_minutes: float | None
    resolution_sample_count: int
    remote_resolution_count: int
    work_order_resolution_count: int
    remote_resolution_rate: float | None
    closed_resolution_count: int
    pending_confirmation_count: int
    confirmation_sample_count: int
    mean_confirmation_minutes: float | None
    p50_confirmation_minutes: float | None
    p90_confirmation_minutes: float | None


class ServicePerformanceSliceResponse(_AttributeResponse):
    dimension: Literal["SITE", "INCIDENT_CATEGORY"]
    key: str
    label: str
    closed_work_order_count: int
    first_time_fix_eligible_count: int
    first_time_fix_count: int
    first_time_fix_rate: float | None
    reworked_work_order_count: int
    mean_mttr_minutes: float | None
    resolution_sample_count: int
    remote_resolution_count: int
    remote_resolution_rate: float | None
    pending_confirmation_count: int


class ServicePerformanceDataQualityResponse(_AttributeResponse):
    status: Literal["COMPLETE", "PARTIAL", "NO_DATA"]
    missing_completion_count: int
    missing_passed_verification_count: int
    invalid_timeline_count: int
    missing_close_control_count: int
    invalid_close_timeline_count: int
    exclusions: list[str]


class ServicePerformanceProjectionResponse(_AttributeResponse):
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    window_basis: str
    resolution_window_basis: str
    mttr_definition: str
    first_time_fix_definition: str
    resolution_definition: str
    summary: ServicePerformanceSummaryResponse
    slices: list[ServicePerformanceSliceResponse]
    data_quality: ServicePerformanceDataQualityResponse


class ServicePerformanceResponse(ServicePerformanceProjectionResponse):
    metric_contract_version: str
    baseline_status: Literal["NOT_CONFIGURED"]


class ServicePerformanceEnvelope(BaseModel):
    data: ServicePerformanceResponse
    meta: dict[str, str]


class ServicePerformanceBaselineTargetsBody(_AttributeResponse):
    mttr_reduction_rate: float = Field(ge=0, le=1)
    first_time_fix_lift: float = Field(ge=0, le=1)
    remote_resolution_lift: float = Field(ge=0, le=1)


class ServicePerformanceBaselineCreateBody(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    reference_window_start: datetime
    reference_window_end: datetime
    minimum_work_order_samples: int = Field(ge=1, le=1_000_000)
    minimum_resolution_samples: int = Field(ge=1, le=1_000_000)
    targets: ServicePerformanceBaselineTargetsBody


class ServicePerformanceBaselineActivationBody(BaseModel):
    reason: str = Field(min_length=5, max_length=2_000)


class ServicePerformanceBaselineResponse(_AttributeResponse):
    baseline_id: str
    name: str
    scope_type: Literal["TENANT"]
    reference_window_start: datetime
    reference_window_end: datetime
    metric_contract_version: str
    metrics: ServicePerformanceBaselineMetrics
    samples: ServicePerformanceBaselineSamples
    minimum_work_order_samples: int
    minimum_resolution_samples: int
    targets: ServicePerformanceBaselineTargets
    data_quality: ServicePerformanceDataQuality
    content_digest: str
    status: Literal["DRAFT", "ACTIVE", "RETIRED"]
    version: int
    created_by_subject_id: str
    activated_by_subject_id: str | None
    activation_reason: str | None
    activated_at: datetime | None
    retired_at: datetime | None
    created_at: datetime
    updated_at: datetime
    legal_actions: list[str] = Field(default_factory=list)


class ServicePerformanceBaselineEnvelope(BaseModel):
    data: ServicePerformanceBaselineResponse
    meta: dict[str, str]


class ServicePerformanceBaselineListEnvelope(BaseModel):
    data: list[ServicePerformanceBaselineResponse]
    meta: dict[str, str]
    legal_actions: list[str]


class ServicePerformanceV3Response(ServicePerformanceProjectionResponse):
    metric_contract_version: Literal["service-performance/v3"]
    baseline_status: Literal["ACTIVE", "NOT_CONFIGURED", "NOT_EFFECTIVE"]
    baseline: ServicePerformanceBaselineResponse | None
    target_comparisons: list[ServicePerformanceTargetComparison]


class ServicePerformanceV3Envelope(BaseModel):
    data: ServicePerformanceV3Response
    meta: dict[str, str]


@router.get(
    "/service-performance/baselines",
    response_model=ServicePerformanceBaselineListEnvelope,
)
async def list_service_performance_baselines(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ServicePerformanceBaselineListEnvelope:
    try:
        baselines = ServicePerformanceService(database, authorizer).list_baselines(
            identity,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(403, "service_performance_baseline_forbidden", "authorization",
                       "Service performance baseline access was denied") from exc
    return ServicePerformanceBaselineListEnvelope(
        data=[_baseline_response(item, identity, authorizer) for item in baselines],
        meta={"request_id": request.state.request_id},
        legal_actions=_baseline_collection_actions(identity, authorizer),
    )


@router.post(
    "/service-performance/baselines",
    response_model=ServicePerformanceBaselineEnvelope,
    status_code=201,
)
async def create_service_performance_baseline(
    body: ServicePerformanceBaselineCreateBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ServicePerformanceBaselineEnvelope:
    try:
        baseline = ServicePerformanceService(database, authorizer).capture_baseline(
            identity,
            name=body.name,
            reference_window_start=body.reference_window_start,
            reference_window_end=body.reference_window_end,
            minimum_work_order_samples=body.minimum_work_order_samples,
            minimum_resolution_samples=body.minimum_resolution_samples,
            targets=ServicePerformanceBaselineTargets(
                mttr_reduction_rate=body.targets.mttr_reduction_rate,
                first_time_fix_lift=body.targets.first_time_fix_lift,
                remote_resolution_lift=body.targets.remote_resolution_lift,
            ),
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(403, "service_performance_baseline_forbidden", "authorization",
                       "Service performance baseline management was denied") from exc
    except ServicePerformanceQueryInvalid as exc:
        raise AppError(422, exc.reason, "validation",
                       "Service performance reference window is invalid") from exc
    except ServicePerformanceBaselineInvalid as exc:
        raise AppError(422, exc.reason, "validation",
                       "Service performance baseline evidence was rejected") from exc
    return ServicePerformanceBaselineEnvelope(
        data=_baseline_response(baseline, identity, authorizer),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/service-performance/baselines/{baseline_id}/activation",
    response_model=ServicePerformanceBaselineEnvelope,
)
async def activate_service_performance_baseline(
    baseline_id: str,
    body: ServicePerformanceBaselineActivationBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ServicePerformanceBaselineEnvelope:
    try:
        baseline = ServicePerformanceService(database, authorizer).activate_baseline(
            identity,
            baseline_id,
            expected_version=_version(if_match),
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(403, "service_performance_baseline_forbidden", "authorization",
                       "Service performance baseline management was denied") from exc
    except ServicePerformanceBaselineNotFound as exc:
        raise AppError(404, "service_performance_baseline_not_found", "not_found",
                       "Service performance baseline was not found") from exc
    except ServicePerformanceBaselineInvalid as exc:
        raise AppError(422, exc.reason, "validation",
                       "Service performance baseline activation was rejected") from exc
    except ServicePerformanceBaselineConflict as exc:
        raise AppError(
            409,
            exc.reason,
            "conflict",
            "Service performance baseline state rejected this action",
            details=({"current_version": exc.current_version}
                     if exc.current_version is not None else None),
        ) from exc
    return ServicePerformanceBaselineEnvelope(
        data=_baseline_response(baseline, identity, authorizer),
        meta={"request_id": request.state.request_id})


@router.get(
    "/service-performance",
    response_model=ServicePerformanceEnvelope | ServicePerformanceV3Envelope,
)
async def get_service_performance(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    window_start: Annotated[datetime, Query()],
    window_end: Annotated[datetime, Query()],
    contract_version: Annotated[Literal["v2", "v3"], Query()] = "v2",
) -> ServicePerformanceEnvelope | ServicePerformanceV3Envelope:
    try:
        service = ServicePerformanceService(database, authorizer)
        if contract_version == "v3":
            v3_snapshot = service.snapshot_v3(
                identity,
                window_start=window_start,
                window_end=window_end,
                request_id=request.state.request_id,
            )
            return ServicePerformanceV3Envelope(
                data=_v3_response(v3_snapshot, identity, authorizer),
                meta={"request_id": request.state.request_id},
            )
        snapshot = service.snapshot(
            identity,
            window_start=window_start,
            window_end=window_end,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(403, "service_performance_forbidden", "authorization",
                       "Service performance access was denied") from exc
    except ServicePerformanceQueryInvalid as exc:
        raise AppError(422, exc.reason, "validation",
                       "Service performance window is invalid") from exc
    return ServicePerformanceEnvelope(
        data=ServicePerformanceResponse.model_validate(snapshot),
        meta={"request_id": request.state.request_id},
    )


def _v3_response(
    snapshot: ServicePerformanceSnapshotV3,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> ServicePerformanceV3Response:
    current = ServicePerformanceResponse.model_validate(snapshot.current)
    return ServicePerformanceV3Response(
        **current.model_dump(exclude={"metric_contract_version", "baseline_status"}),
        metric_contract_version=snapshot.metric_contract_version,
        baseline_status=snapshot.baseline_status,
        baseline=(
            _baseline_response(snapshot.baseline, identity, authorizer)
            if snapshot.baseline is not None
            else None
        ),
        target_comparisons=list(snapshot.target_comparisons),
    )


def _baseline_response(
    baseline: ServicePerformanceBaseline,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> ServicePerformanceBaselineResponse:
    resource = ResourceContext(identity.tenant_id, baseline.baseline_id)
    legal_actions = (
        ["ACTIVATE"]
        if authorizer.decide(
            identity,
            Action.MANAGE_SERVICE_PERFORMANCE_BASELINE,
            resource,
        ).allowed
        and baseline.status == "DRAFT"
        and baseline.created_by_subject_id != identity.subject_id
        else []
    )
    response = ServicePerformanceBaselineResponse.model_validate(baseline)
    return response.model_copy(update={"legal_actions": legal_actions})


def _baseline_collection_actions(
    identity: IdentityContext,
    authorizer: Authorizer,
) -> list[str]:
    resource = ResourceContext(identity.tenant_id, "service-performance-baselines")
    return ["CREATE"] if authorizer.decide(
        identity,
        Action.MANAGE_SERVICE_PERFORMANCE_BASELINE,
        resource,
    ).allowed else []


def _version(value: str) -> int:
    match = re.fullmatch(r'"?([1-9][0-9]*)"?', value.strip())
    if match is None:
        raise AppError(
            400,
            "invalid_version_precondition",
            "validation",
            "If-Match is invalid",
        )
    return int(match.group(1))
