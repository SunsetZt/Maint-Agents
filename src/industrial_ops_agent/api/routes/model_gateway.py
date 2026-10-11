"""Read-safe model serving governance and quota administration API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_model_resolver,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.model_gateway.context_manifest import context_manifest_is_valid
from industrial_ops_agent.model_gateway.governance import (
    ModelGatewayGovernanceConflict,
    ModelGatewayGovernanceService,
    ModelRouteView,
    ModelRuntimeComponentView,
)
from industrial_ops_agent.model_gateway.service import EnvironmentAwareModelResolver
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import ModelInferenceRecord

router = APIRouter(tags=["model-gateway"])


class ModelQuotaBody(BaseModel):
    requests_per_minute: int = Field(ge=1, le=100_000)
    tokens_per_day: int = Field(ge=1, le=10_000_000_000)
    max_output_tokens: int = Field(ge=1, le=32_768)
    allowed_request_classes: list[str] = Field(min_length=1, max_length=6)
    enabled: bool


class ModelQuotaResponse(ModelQuotaBody):
    version: int


class ModelRouteResponse(BaseModel):
    alias: str
    active_release_id: str
    deployment_id: str
    manifest_hash: str
    endpoint_url: str
    runtime_profile: str
    target_environment: str
    status: str
    version: int
    quota: ModelQuotaResponse | None


class ContextReferenceResponse(BaseModel):
    id: str
    content_hash: str


class ContextTruncationResponse(BaseModel):
    source_type: Literal["QUERY", "EVIDENCE", "MEMORY", "TOOL_RESULT"]
    source_id: str
    reason: Literal["COUNT_LIMIT", "CHARACTER_LIMIT", "TOKEN_BUDGET"]
    original_units: int
    included_units: int


class ContextManifestResponse(BaseModel):
    schema_version: Literal["context-manifest/v1"]
    trace_id: str
    prompt_bundle_id: str | None
    prompt_bundle_hash: str | None
    model_release_id: str
    retrieval_index_id: str | None
    evidence: list[ContextReferenceResponse]
    memories: list[ContextReferenceResponse]
    tool_results: list[ContextReferenceResponse]
    token_budget: int
    truncated_items: list[ContextTruncationResponse]
    rendered_context_hash: str
    context_hash: str


class ModelInferenceResponse(BaseModel):
    inference_request_id: str
    model_alias: str
    resolved_release_id: str
    manifest_hash: str
    subject_id: str
    trace_id: str
    agent_run_id: str | None
    diagnosis_run_id: str | None
    prompt_bundle_id: str | None
    index_release_id: str | None
    context_hash: str | None
    context_manifest: ContextManifestResponse | None
    request_class: str
    data_classification: str
    status: str
    deadline: datetime
    max_output_tokens: int
    prompt_hash: str
    response_hash: str | None
    usage_prompt_tokens: int
    usage_completion_tokens: int
    finish_reason: str | None
    latency_breakdown: dict[str, float]
    safety_decision: str
    guardrail_policy_version: str
    guardrail_findings: list[dict[str, str]]
    failure_reason: str | None
    created_at: datetime
    completed_at: datetime | None


class ModelRouteListEnvelope(BaseModel):
    data: list[ModelRouteResponse]
    meta: dict[str, str | int]


class ModelRouteEnvelope(BaseModel):
    data: ModelRouteResponse
    meta: dict[str, str]


class ModelInferenceListEnvelope(BaseModel):
    data: list[ModelInferenceResponse]
    meta: dict[str, str | int]


class ModelRuntimeComponentResponse(BaseModel):
    binding_status: Literal["READY", "MODEL_UNAVAILABLE"]
    reason: str | None
    alias: str | None
    release_id: str | None
    deployment_id: str | None
    provider: str | None
    processor_version: str | None
    model_version: str | None
    accelerator: str | None
    endpoint_status: Literal["READY", "MODEL_UNAVAILABLE"] | None


class ModelExecutionResponse(BaseModel):
    component: Literal["vlm", "diagnosis"]
    request_class: str
    target_environment: Literal["STAGING", "PRODUCTION"]
    model_alias: str
    release_id: str
    deployment_id: str
    inference_request_id: str
    processor_version: str
    manifest_hash: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    guardrail_decision: str
    guardrail_policy_version: str
    request_id: str


class ModelRuntimeStatusResponse(BaseModel):
    status: Literal["READY", "MODEL_UNAVAILABLE"]
    target_environment: Literal["STAGING", "PRODUCTION"]
    gateway_status: Literal["READY", "MODEL_UNAVAILABLE"]
    gpu_serving_required: bool
    request_id: str
    components: dict[str, ModelRuntimeComponentResponse]


class ModelRuntimeStatusEnvelope(BaseModel):
    data: ModelRuntimeStatusResponse
    meta: dict[str, str]


@router.get("/model-gateway/runtime-status", response_model=ModelRuntimeStatusEnvelope)
async def get_model_runtime_status(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    resolver: Annotated[
        EnvironmentAwareModelResolver | None, Depends(get_model_resolver)
    ],
) -> ModelRuntimeStatusEnvelope:
    try:
        view = ModelGatewayGovernanceService(database, authorizer).runtime_status(
            identity,
            resolver,
            required_environment=request.app.state.settings.model_gateway_required_environment,
            alias_name=request.app.state.settings.model_gateway_alias,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "model_gateway_forbidden") from exc
    return ModelRuntimeStatusEnvelope(
        data=ModelRuntimeStatusResponse(
            status=view.status,
            target_environment=view.target_environment,
            gateway_status=view.gateway_status,
            gpu_serving_required=view.gpu_serving_required,
            request_id=request.state.request_id,
            components={
                name: _runtime_component(component)
                for name, component in view.components.items()
            },
        ),
        meta={"request_id": request.state.request_id},
    )


@router.get("/model-gateway/routes", response_model=ModelRouteListEnvelope)
async def list_model_routes(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ModelRouteListEnvelope:
    try:
        routes = ModelGatewayGovernanceService(database, authorizer).list_routes(
            identity, request_id=request.state.request_id
        )
    except AuthorizationDenied as exc:
        raise _error(403, "model_gateway_forbidden") from exc
    return ModelRouteListEnvelope(
        data=[_route(item) for item in routes],
        meta={"request_id": request.state.request_id, "total": len(routes)},
    )


@router.get("/model-gateway/inferences", response_model=ModelInferenceListEnvelope)
async def list_model_inferences(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ModelInferenceListEnvelope:
    try:
        records = ModelGatewayGovernanceService(database, authorizer).list_inferences(
            identity, request_id=request.state.request_id, limit=limit
        )
    except AuthorizationDenied as exc:
        raise _error(403, "model_gateway_forbidden") from exc
    return ModelInferenceListEnvelope(
        data=[_inference(item) for item in records],
        meta={"request_id": request.state.request_id, "total": len(records)},
    )


@router.put("/model-gateway/routes/{alias_name}/quota", response_model=ModelRouteEnvelope)
async def update_model_quota(
    alias_name: str,
    body: ModelQuotaBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ModelRouteEnvelope:
    try:
        route = ModelGatewayGovernanceService(database, authorizer).update_quota(
            identity,
            alias_name,
            **body.model_dump(),
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "model_gateway_forbidden") from exc
    except ModelGatewayGovernanceConflict as exc:
        raise AppError(
            status_code=409,
            code=exc.reason,
            category="conflict",
            message="Model Gateway quota update conflicted with current state",
            details={"current_version": exc.current_version},
        ) from exc
    except ValueError as exc:
        raise _error(422, "model_quota_invalid") from exc
    return ModelRouteEnvelope(data=_route(route), meta={"request_id": request.state.request_id})


def _route(view: ModelRouteView) -> ModelRouteResponse:
    alias = view.alias
    quota = view.quota
    return ModelRouteResponse(
        alias=alias.alias,
        active_release_id=alias.active_release_id,
        deployment_id=alias.deployment_id,
        manifest_hash=alias.manifest_hash,
        endpoint_url=alias.endpoint_url,
        runtime_profile=alias.runtime_profile,
        target_environment=view.target_environment,
        status=alias.status,
        version=alias.version,
        quota=(
            ModelQuotaResponse(
                requests_per_minute=quota.requests_per_minute,
                tokens_per_day=quota.tokens_per_day,
                max_output_tokens=quota.max_output_tokens,
                allowed_request_classes=quota.allowed_request_classes,
                enabled=quota.enabled,
                version=quota.version,
            )
            if quota is not None
            else None
        ),
    )


def _runtime_component(view: ModelRuntimeComponentView) -> ModelRuntimeComponentResponse:
    return ModelRuntimeComponentResponse(
        binding_status=view.binding_status,
        reason=view.reason,
        alias=view.alias,
        release_id=view.release_id,
        deployment_id=view.deployment_id,
        provider=view.provider,
        processor_version=view.processor_version,
        model_version=view.model_version,
        accelerator=view.accelerator,
        endpoint_status=view.endpoint_status,
    )


def _inference(record: ModelInferenceRecord) -> ModelInferenceResponse:
    context_manifest = (
        ContextManifestResponse.model_validate(record.context_manifest_json)
        if record.context_hash is not None
        and record.context_manifest_json.get("context_hash") == record.context_hash
        and record.context_manifest_json.get("trace_id") == record.trace_id
        and context_manifest_is_valid(record.context_manifest_json)
        else None
    )
    return ModelInferenceResponse(
        **{
            field: (
                record.latency_breakdown_json
                if field == "latency_breakdown"
                else record.guardrail_findings_json
                if field == "guardrail_findings"
                else context_manifest
                if field == "context_manifest"
                else getattr(record, field)
            )
            for field in ModelInferenceResponse.model_fields
        }
    )


def _error(status_code: int, code: str) -> AppError:
    return AppError(
        status_code=status_code,
        code=code,
        category="authorization" if status_code == 403 else "validation",
        message="Model Gateway request was rejected",
    )


def _version(value: str) -> int:
    try:
        version = int(value.strip().strip('"'))
    except ValueError as exc:
        raise _error(422, "model_quota_version_invalid") from exc
    if version < 1:
        raise _error(422, "model_quota_version_invalid")
    return version
