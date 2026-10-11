"""Governed desired-state deployment and progressive delivery API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_optional_assurance_service,
    get_optional_operations_service,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.assurance.service import AssuranceService
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.deployment.service import (
    AutoscalingMode,
    DeploymentAggregate,
    DeploymentConflict,
    DeploymentNotVisible,
    DeploymentPlan,
    ModelDeploymentService,
    RoutingMode,
)
from industrial_ops_agent.operations.service import OperationsService
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import ModelReleaseObservationRecord

router = APIRouter(tags=["model-deployments"])


class DeploymentPlanBody(BaseModel):
    namespace: str = Field(min_length=1, max_length=63)
    gateway_name: str = Field(min_length=1, max_length=63)
    hostname: str = Field(min_length=3, max_length=253)
    route_name: str = Field(min_length=1, max_length=63)
    stable_service_name: str = Field(min_length=1, max_length=63)
    service_account_name: str = Field(min_length=1, max_length=63)
    serving_runtime_name: str = Field(min_length=1, max_length=63)
    artifact_uri_prefix: str = Field(min_length=6, max_length=1024)
    routing_mode: RoutingMode = "STANDARD"
    endpoint_picker_service_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=63,
    )
    endpoint_picker_service_port: int | None = Field(default=None, ge=1, le=65535)
    autoscaling_mode: AutoscalingMode = "FIXED"
    autoscaling_min_replicas: int | None = Field(default=None, ge=2, le=8)
    autoscaling_max_replicas: int | None = Field(default=None, ge=2, le=32)
    autoscaling_target_running_requests: int | None = Field(default=None, ge=1, le=32)


class RollbackBody(BaseModel):
    reason_code: str = Field(min_length=1, max_length=128)


class ReleaseObservationResponse(BaseModel):
    observation_id: str
    stage: str
    sequence: int
    window_start: datetime
    window_end: datetime
    request_count: int
    critical_case_count: int
    metrics: dict[str, float]
    thresholds: dict[str, Any]
    source_refs: dict[str, str]
    evidence_hash: str
    decision: str
    failure_reasons: list[str]
    collected_by_subject_id: str
    created_at: datetime


class ModelDeploymentResponse(BaseModel):
    deployment_id: str
    release_id: str
    release_status: str
    release_version: int
    manifest_hash: str
    provider: str
    namespace: str
    service_name: str
    route_name: str
    stable_service_name: str
    status: str
    desired_stage: str
    current_stage: str
    desired_traffic_percent: float
    observed_traffic_percent: float
    desired_spec_hash: str
    applied_spec_hash: str | None
    provider_revision: str | None
    endpoint_url: str | None
    routing_mode: RoutingMode
    inference_pool_name: str | None
    endpoint_picker_service_name: str | None
    endpoint_picker_service_port: int | None
    endpoint_picker_failure_mode: str | None
    autoscaling_mode: AutoscalingMode
    autoscaling_policy_version: str | None
    autoscaling_min_replicas: int | None
    autoscaling_max_replicas: int | None
    autoscaling_metric_backend: str | None
    autoscaling_metric_name: str | None
    autoscaling_metric_scope: str | None
    autoscaling_target_running_requests: int | None
    autoscaling_scale_down_stabilization_seconds: int | None
    retry_count: int
    failure_reason: str | None
    requested_by_subject_id: str
    reconciled_by_subject_id: str | None
    version: int
    created_at: datetime
    updated_at: datetime
    observations: list[ReleaseObservationResponse]
    legal_actions: list[str]


class ModelDeploymentEnvelope(BaseModel):
    data: ModelDeploymentResponse
    meta: dict[str, str]


class ModelDeploymentListEnvelope(BaseModel):
    data: list[ModelDeploymentResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


@router.get("/model-deployments", response_model=ModelDeploymentListEnvelope)
async def list_model_deployments(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ModelDeploymentListEnvelope:
    try:
        aggregates = ModelDeploymentService(database, authorizer).list(
            identity, request_id=request.state.request_id
        )
    except (AuthorizationDenied, DeploymentNotVisible, DeploymentConflict) as exc:
        raise _translate(exc) from exc
    data = [_response(item, identity, authorizer) for item in aggregates]
    return ModelDeploymentListEnvelope(
        data=data,
        meta={"request_id": request.state.request_id, "total": len(data)},
        legal_actions=(
            ["REQUEST_MODEL_DEPLOYMENT"]
            if _allowed(
                identity,
                authorizer,
                Action.REQUEST_MODEL_DEPLOYMENT,
                "model-deployments",
            )
            else []
        ),
    )


@router.get(
    "/model-releases/{release_id}/deployment",
    response_model=ModelDeploymentEnvelope,
)
async def get_model_deployment(
    release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ModelDeploymentEnvelope:
    try:
        aggregate = ModelDeploymentService(database, authorizer).get(
            identity, release_id, request_id=request.state.request_id
        )
    except (AuthorizationDenied, DeploymentNotVisible, DeploymentConflict) as exc:
        raise _translate(exc) from exc
    return _envelope(aggregate, identity, authorizer, request.state.request_id)


@router.post(
    "/model-releases/{release_id}/deployments",
    response_model=ModelDeploymentEnvelope,
    status_code=202,
)
async def request_model_deployment(
    release_id: str,
    body: DeploymentPlanBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ModelDeploymentEnvelope:
    try:
        aggregate = ModelDeploymentService(database, authorizer).request_shadow(
            identity,
            release_id,
            DeploymentPlan(**body.model_dump()),
            expected_release_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, DeploymentNotVisible, DeploymentConflict) as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc
    return _envelope(aggregate, identity, authorizer, request.state.request_id)


@router.post(
    "/model-releases/{release_id}/promotions",
    response_model=ModelDeploymentEnvelope,
    status_code=202,
)
async def promote_model_release(
    release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    operations: Annotated[
        OperationsService | None,
        Depends(get_optional_operations_service),
    ],
    assurance: Annotated[
        AssuranceService | None,
        Depends(get_optional_assurance_service),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ModelDeploymentEnvelope:
    if _allowed(identity, authorizer, Action.PROMOTE_MODEL_RELEASE, release_id):
        await _require_release_gate(operations)
        current = await asyncio.to_thread(
            ModelDeploymentService(database, authorizer).get,
            identity,
            release_id,
            request_id=request.state.request_id,
        )
        if current.deployment.current_stage == "CANARY_25":
            await _require_production_acceptance(
                assurance,
                tenant_id=identity.tenant_id,
                release_id=release_id,
            )
    try:
        aggregate = ModelDeploymentService(database, authorizer).promote(
            identity,
            release_id,
            expected_deployment_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, DeploymentNotVisible, DeploymentConflict) as exc:
        raise _translate(exc) from exc
    return _envelope(aggregate, identity, authorizer, request.state.request_id)


@router.post(
    "/model-releases/{release_id}/rollbacks",
    response_model=ModelDeploymentEnvelope,
    status_code=202,
)
async def rollback_model_release(
    release_id: str,
    body: RollbackBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ModelDeploymentEnvelope:
    try:
        aggregate = ModelDeploymentService(database, authorizer).request_rollback(
            identity,
            release_id,
            expected_deployment_version=_version(if_match),
            reason_code=body.reason_code,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, DeploymentNotVisible, DeploymentConflict) as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc
    return _envelope(aggregate, identity, authorizer, request.state.request_id)


def _envelope(
    aggregate: DeploymentAggregate,
    identity: IdentityContext,
    authorizer: Authorizer,
    request_id: str,
) -> ModelDeploymentEnvelope:
    return ModelDeploymentEnvelope(
        data=_response(aggregate, identity, authorizer),
        meta={"request_id": request_id},
    )


def _response(
    aggregate: DeploymentAggregate,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> ModelDeploymentResponse:
    release = aggregate.release
    deployment = aggregate.deployment
    latest = aggregate.observations[-1] if aggregate.observations else None
    legal_actions: list[str] = []
    ready_for_promotion = (
        deployment.status == "READY"
        and deployment.current_stage == deployment.desired_stage
        and deployment.current_stage in {"SHADOW", "CANARY_5", "CANARY_25"}
        and latest is not None
        and latest.stage == deployment.current_stage
        and latest.decision == "PASS"
    )
    if ready_for_promotion and _allowed(
        identity, authorizer, Action.PROMOTE_MODEL_RELEASE, release.release_id
    ):
        legal_actions.append("PROMOTE_MODEL_RELEASE")
    if deployment.current_stage not in {"NONE", "ROLLED_BACK"} and _allowed(
        identity, authorizer, Action.ROLLBACK_MODEL_RELEASE, release.release_id
    ):
        legal_actions.append("ROLLBACK_MODEL_RELEASE")
    routing = deployment.desired_spec_json.get("routing")
    pool_routing = (
        routing
        if isinstance(routing, dict) and routing.get("mode") == "INFERENCE_POOL"
        else None
    )
    endpoint_picker = (
        pool_routing.get("endpoint_picker_ref") if pool_routing is not None else None
    )
    if not isinstance(endpoint_picker, dict):
        endpoint_picker = None
    autoscaling = deployment.desired_spec_json.get("autoscaling")
    governed_autoscaling = (
        autoscaling
        if isinstance(autoscaling, dict) and autoscaling.get("mode") == "KEDA_VLLM"
        else None
    )
    autoscaling_metric = (
        governed_autoscaling.get("metric") if governed_autoscaling is not None else None
    )
    if not isinstance(autoscaling_metric, dict):
        autoscaling_metric = None
    autoscaling_behavior = (
        governed_autoscaling.get("behavior") if governed_autoscaling is not None else None
    )
    if not isinstance(autoscaling_behavior, dict):
        autoscaling_behavior = None
    return ModelDeploymentResponse(
        deployment_id=deployment.deployment_id,
        release_id=release.release_id,
        release_status=release.status,
        release_version=release.version,
        manifest_hash=release.manifest_hash,
        provider=deployment.provider,
        namespace=deployment.namespace,
        service_name=deployment.service_name,
        route_name=deployment.route_name,
        stable_service_name=deployment.stable_service_name,
        status=deployment.status,
        desired_stage=deployment.desired_stage,
        current_stage=deployment.current_stage,
        desired_traffic_percent=deployment.desired_traffic_percent,
        observed_traffic_percent=deployment.observed_traffic_percent,
        desired_spec_hash=deployment.desired_spec_hash,
        applied_spec_hash=deployment.applied_spec_hash,
        provider_revision=deployment.provider_revision,
        endpoint_url=deployment.endpoint_url,
        routing_mode="INFERENCE_POOL" if pool_routing is not None else "STANDARD",
        inference_pool_name=(
            str(pool_routing["inference_pool_name"])
            if pool_routing is not None
            and isinstance(pool_routing.get("inference_pool_name"), str)
            else None
        ),
        endpoint_picker_service_name=(
            str(endpoint_picker["name"])
            if endpoint_picker is not None and isinstance(endpoint_picker.get("name"), str)
            else None
        ),
        endpoint_picker_service_port=(
            int(endpoint_picker["port"])
            if endpoint_picker is not None
            and isinstance(endpoint_picker.get("port"), int)
            else None
        ),
        endpoint_picker_failure_mode=(
            str(endpoint_picker["failure_mode"])
            if endpoint_picker is not None
            and isinstance(endpoint_picker.get("failure_mode"), str)
            else None
        ),
        autoscaling_mode="KEDA_VLLM" if governed_autoscaling is not None else "FIXED",
        autoscaling_policy_version=(
            str(governed_autoscaling["policy_version"])
            if governed_autoscaling is not None
            and isinstance(governed_autoscaling.get("policy_version"), str)
            else None
        ),
        autoscaling_min_replicas=(
            int(governed_autoscaling["min_replicas"])
            if governed_autoscaling is not None
            and isinstance(governed_autoscaling.get("min_replicas"), int)
            else None
        ),
        autoscaling_max_replicas=(
            int(governed_autoscaling["max_replicas"])
            if governed_autoscaling is not None
            and isinstance(governed_autoscaling.get("max_replicas"), int)
            else None
        ),
        autoscaling_metric_backend=(
            str(autoscaling_metric["backend"])
            if autoscaling_metric is not None
            and isinstance(autoscaling_metric.get("backend"), str)
            else None
        ),
        autoscaling_metric_name=(
            str(autoscaling_metric["name"])
            if autoscaling_metric is not None
            and isinstance(autoscaling_metric.get("name"), str)
            else None
        ),
        autoscaling_metric_scope=(
            f"namespace={deployment.namespace},inferenceservice={deployment.service_name}"
            if governed_autoscaling is not None
            else None
        ),
        autoscaling_target_running_requests=(
            int(autoscaling_metric["target_value"])
            if autoscaling_metric is not None
            and isinstance(autoscaling_metric.get("target_value"), int)
            else None
        ),
        autoscaling_scale_down_stabilization_seconds=(
            int(autoscaling_behavior["scale_down_stabilization_seconds"])
            if autoscaling_behavior is not None
            and isinstance(
                autoscaling_behavior.get("scale_down_stabilization_seconds"), int
            )
            else None
        ),
        retry_count=deployment.retry_count,
        failure_reason=deployment.failure_reason,
        requested_by_subject_id=deployment.requested_by_subject_id,
        reconciled_by_subject_id=deployment.reconciled_by_subject_id,
        version=deployment.version,
        created_at=deployment.created_at,
        updated_at=deployment.updated_at,
        observations=[_observation_response(item) for item in aggregate.observations],
        legal_actions=legal_actions,
    )


def _observation_response(record: ModelReleaseObservationRecord) -> ReleaseObservationResponse:
    return ReleaseObservationResponse(
        observation_id=record.observation_id,
        stage=record.stage,
        sequence=record.sequence,
        window_start=record.window_start,
        window_end=record.window_end,
        request_count=record.request_count,
        critical_case_count=record.critical_case_count,
        metrics=record.metrics_json,
        thresholds=record.thresholds_json,
        source_refs=record.source_refs_json,
        evidence_hash=record.evidence_hash,
        decision=record.decision,
        failure_reasons=record.failure_reasons,
        collected_by_subject_id=record.collected_by_subject_id,
        created_at=record.created_at,
    )


def _allowed(
    identity: IdentityContext,
    authorizer: Authorizer,
    action: Action,
    resource_id: str,
) -> bool:
    return authorizer.decide(
        identity,
        action,
        ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
    ).allowed


def _version(value: str) -> int:
    try:
        version = int(value.strip().strip('"'))
    except ValueError as exc:
        raise AppError(
            400, "invalid_version_precondition", "validation", "If-Match is invalid"
        ) from exc
    if version < 1:
        raise AppError(400, "invalid_version_precondition", "validation", "If-Match is invalid")
    return version


async def _require_release_gate(operations: OperationsService | None) -> None:
    if operations is None:
        return
    snapshot = await asyncio.to_thread(operations.snapshot)
    if snapshot.high_risk_release_allowed:
        return
    raise AppError(
        status_code=409,
        code="production_release_gate_closed",
        category="conflict",
        message="Production SLO, alert, or security evidence paused this release action",
        details={"reasons": list(snapshot.release_gate_reasons)},
    )


async def _require_production_acceptance(
    assurance: AssuranceService | None,
    *,
    tenant_id: str,
    release_id: str,
) -> None:
    if assurance is None:
        return
    reasons = await asyncio.to_thread(
        assurance.production_release_reasons,
        tenant_id,
        release_id,
    )
    if not reasons:
        return
    raise AppError(
        status_code=409,
        code="production_acceptance_gate_closed",
        category="conflict",
        message="Production acceptance evidence or sign-offs paused final promotion",
        details={"reasons": list(reasons)},
    )


def _translate(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(403, "authorization_denied", "authorization", "Action is not allowed")
    if isinstance(exc, DeploymentNotVisible):
        return AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        )
    if isinstance(exc, DeploymentConflict):
        return AppError(
            409,
            exc.reason,
            "model_deployment_governance",
            "Model deployment governance rejected the operation",
            details={"current_version": exc.current_version},
        )
    raise TypeError("unsupported model deployment error")


def _invalid(exc: ValueError) -> AppError:
    return AppError(400, "invalid_model_deployment_request", "validation", str(exc))
