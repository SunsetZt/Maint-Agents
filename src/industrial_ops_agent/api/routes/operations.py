"""Read-only production operations BFF for SLOs, alerts, and runbooks."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_gpu_operations_service,
    get_identity,
    get_operations_service,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.gpu_operations.service import (
    GpuOperationsService,
    GpuOperationsSnapshot,
)
from industrial_ops_agent.operations.service import OperationsService, OperationsSnapshot

router = APIRouter(tags=["operations"])


class SloObservationResponse(BaseModel):
    slo_id: str
    name: str
    owner: str
    objective: float
    comparison: Literal[">=", "<="]
    unit: str
    window: str
    status: Literal["HEALTHY", "BREACHED", "NO_DATA"]
    current_value: float | None
    error_budget_remaining_fraction: float | None
    runbook_id: str


class ActiveAlertResponse(BaseModel):
    name: str
    severity: Literal["critical", "warning", "info"]
    state: Literal["firing", "pending"]
    active_at: datetime
    summary: str
    runbook_id: str
    owner: str
    security_hard_gate: bool


class RunbookResponse(BaseModel):
    runbook_id: str
    title: str
    owner: str
    user_impact: str
    confirmation_queries: list[str]
    safety_containment: list[str]
    recovery: list[str]
    rollback: list[str]
    consistency_checks: list[str]
    escalation: str
    postmortem: str


class OperationsOverviewResponse(BaseModel):
    policy_version: str
    generated_at: datetime
    data_source_status: Literal["AVAILABLE", "DEGRADED", "UNAVAILABLE"]
    high_risk_release_allowed: bool
    release_gate_reasons: list[str]
    slos: list[SloObservationResponse]
    alerts: list[ActiveAlertResponse]
    runbooks: list[RunbookResponse]


class OperationsOverviewEnvelope(BaseModel):
    data: OperationsOverviewResponse
    meta: dict[str, str]


class ClusterGpuPostureResponse(BaseModel):
    health: Literal["HEALTHY", "DEGRADED", "CRITICAL", "NO_DATA"]
    reasons: list[str]
    discovered_gpu_count: float | None
    allocatable_gpu_count: float | None
    requested_gpu_count: float | None
    capacity_headroom_gpu: float | None
    average_utilization: float | None
    memory_utilization: float | None
    max_temperature_celsius: float | None
    power_usage_watts: float | None
    xid_errors_15m: float | None
    ecc_dbe_errors_15m: float | None
    dcgm_targets_up_ratio: float | None


class RuntimeGpuPostureResponse(BaseModel):
    deployment_id: str
    release_id: str
    provider: str
    namespace: str
    service_name: str
    deployment_status: str
    desired_stage: str
    current_stage: str
    desired_traffic_percent: float
    observed_traffic_percent: float
    runtime_engine: str | None
    gpu_model: str | None
    gpu_per_replica: float | None
    running_replicas: float | None
    request_rate_per_second: float | None
    error_rate: float | None
    p95_latency_ms: float | None
    gpu_utilization: float | None
    gpu_memory_utilization: float | None
    queue_age_seconds: float | None
    xid_errors_15m: float | None
    latest_observation_at: datetime | None
    latest_observation_decision: str | None
    advisory: Literal[
        "HOLD",
        "SCALE_OUT_RECOMMENDED",
        "SCALE_IN_CANDIDATE",
        "INVESTIGATE",
        "NO_DATA",
    ]
    advisory_reasons: list[str]


class GpuOperationsResponse(BaseModel):
    policy_version: str
    generated_at: datetime
    data_source_status: Literal["AVAILABLE", "DEGRADED", "UNAVAILABLE"]
    cluster: ClusterGpuPostureResponse
    runtimes: list[RuntimeGpuPostureResponse]
    thresholds: dict[str, float]


class GpuOperationsEnvelope(BaseModel):
    data: GpuOperationsResponse
    meta: dict[str, str]


@router.get("/operations/overview", response_model=OperationsOverviewEnvelope)
async def get_operations_overview(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[OperationsService, Depends(get_operations_service)],
) -> OperationsOverviewEnvelope:
    try:
        authorizer.require(
            identity,
            Action.READ_OPERATIONS,
            ResourceContext(tenant_id=identity.tenant_id, resource_id="operations-overview"),
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(
            status_code=403,
            code="operations_overview_forbidden",
            category="authorization",
            message="Operations overview access was denied",
        ) from exc
    snapshot = await asyncio.to_thread(service.snapshot)
    return OperationsOverviewEnvelope(
        data=_response(snapshot),
        meta={"request_id": request.state.request_id},
    )


@router.get("/operations/gpu", response_model=GpuOperationsEnvelope)
async def get_gpu_operations(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[GpuOperationsService, Depends(get_gpu_operations_service)],
) -> GpuOperationsEnvelope:
    try:
        authorizer.require(
            identity,
            Action.READ_OPERATIONS,
            ResourceContext(tenant_id=identity.tenant_id, resource_id="gpu-operations"),
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(
            status_code=403,
            code="gpu_operations_forbidden",
            category="authorization",
            message="GPU operations access was denied",
        ) from exc
    snapshot = await asyncio.to_thread(
        service.snapshot,
        identity.tenant_id,
        identity.subject_id,
    )
    return GpuOperationsEnvelope(
        data=_gpu_response(snapshot),
        meta={"request_id": request.state.request_id},
    )


def _response(snapshot: OperationsSnapshot) -> OperationsOverviewResponse:
    return OperationsOverviewResponse(
        policy_version=snapshot.policy_version,
        generated_at=snapshot.generated_at,
        data_source_status=snapshot.data_source_status,
        high_risk_release_allowed=snapshot.high_risk_release_allowed,
        release_gate_reasons=list(snapshot.release_gate_reasons),
        slos=[
            SloObservationResponse(
                **{field: getattr(item, field) for field in SloObservationResponse.model_fields}
            )
            for item in snapshot.slos
        ],
        alerts=[
            ActiveAlertResponse(
                **{field: getattr(item, field) for field in ActiveAlertResponse.model_fields}
            )
            for item in snapshot.alerts
        ],
        runbooks=[
            RunbookResponse(
                runbook_id=item.runbook_id,
                title=item.title,
                owner=item.owner,
                user_impact=item.user_impact,
                confirmation_queries=list(item.confirmation_queries),
                safety_containment=list(item.safety_containment),
                recovery=list(item.recovery),
                rollback=list(item.rollback),
                consistency_checks=list(item.consistency_checks),
                escalation=item.escalation,
                postmortem=item.postmortem,
            )
            for item in snapshot.runbooks
        ],
    )


def _gpu_response(snapshot: GpuOperationsSnapshot) -> GpuOperationsResponse:
    cluster = ClusterGpuPostureResponse(
        **{
            field: _response_value(getattr(snapshot.cluster, field))
            for field in ClusterGpuPostureResponse.model_fields
        }
    )
    runtimes = [
        RuntimeGpuPostureResponse(
            **{
                field: _response_value(getattr(item, field))
                for field in RuntimeGpuPostureResponse.model_fields
            }
        )
        for item in snapshot.runtimes
    ]
    return GpuOperationsResponse(
        policy_version=snapshot.policy_version,
        generated_at=snapshot.generated_at,
        data_source_status=snapshot.data_source_status,
        cluster=cluster,
        runtimes=runtimes,
        thresholds=snapshot.thresholds,
    )


def _response_value(value: object) -> object:
    return list(value) if isinstance(value, tuple) else value
