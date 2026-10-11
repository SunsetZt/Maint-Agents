"""AIReleaseManifest registration, validation and separation-of-duties approval API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_tool_gateway,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ModelReleaseApprovalRecord,
    ModelReleaseTransitionRecord,
)
from industrial_ops_agent.releases.service import (
    ModelReleaseService,
    ReleaseAggregate,
    ReleaseConflict,
    ReleaseManifestPlan,
    ReleaseNotVisible,
)
from industrial_ops_agent.tools.gateway import ToolGateway

router = APIRouter(tags=["model-releases"])


class ReleaseManifestBody(BaseModel):
    evaluation_admission: Literal["STANDARD", "STAGING_DIAGNOSIS_SMOKE_14"] = "STANDARD"
    prompt_evaluation_only: bool = False
    local_staging_adapter: bool = False
    evaluation_id: str = Field(min_length=1, max_length=128)
    component_evaluation_ids: dict[str, str] = Field(min_length=4, max_length=7)
    target_environment: Literal["STAGING", "PRODUCTION"]
    quantization_profile_id: str = Field(min_length=1, max_length=255)
    runtime_profile_id: str = Field(min_length=1, max_length=255)
    runtime_image_repository: str = Field(min_length=3, max_length=512)
    runtime_image_digest: str = Field(min_length=1, max_length=128)
    prompt_bundle_id: str = Field(min_length=1, max_length=255)
    agent_graph_id: str = Field(min_length=1, max_length=255)
    tool_versions: dict[str, str] = Field(min_length=1)
    mcp_server_versions: dict[str, str] = Field(default_factory=dict)
    index_release_id: str = Field(min_length=1, max_length=128)
    embedding_model_id: str = Field(min_length=1, max_length=255)
    reranker_model_id: str = Field(min_length=1, max_length=255)
    multimodal_model_ids: dict[str, str] = Field(min_length=4, max_length=4)
    inference_config: dict[str, Any]
    hardware_profile: dict[str, Any]
    supply_chain_evidence_id: str | None = Field(default=None, min_length=1, max_length=128)
    rollback_release_id: str | None = Field(default=None, max_length=128)
    timeseries_model_id: str | None = Field(default=None, max_length=255)
    timeseries_supply_chain_evidence_id: str | None = Field(default=None, max_length=128)
    rul_model_id: str | None = Field(default=None, max_length=255)
    rul_supply_chain_evidence_id: str | None = Field(default=None, max_length=128)
    reranker_supply_chain_evidence_id: str | None = Field(default=None, max_length=128)
    embedding_supply_chain_evidence_id: str | None = Field(default=None, max_length=128)
    tts_supply_chain_evidence_id: str | None = Field(default=None, max_length=128)
    vlm_supply_chain_evidence_id: str | None = Field(default=None, max_length=128)
    asr_supply_chain_evidence_id: str | None = Field(default=None, max_length=128)


class SupplyChainSuccessorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vlm_supply_chain_evidence_id: str | None = Field(default=None, min_length=1, max_length=128)
    asr_supply_chain_evidence_id: str | None = Field(default=None, min_length=1, max_length=128)


class ReleaseApprovalDecisionBody(BaseModel):
    approval_version: int = Field(ge=1)
    decision: Literal["APPROVED", "REJECTED"]
    reason: str = Field(min_length=1, max_length=1024)


class ReleaseTransitionResponse(BaseModel):
    transition_id: str
    sequence: int
    from_status: str
    to_status: str
    actor_subject_id: str
    reason_code: str
    evidence: dict[str, Any]
    evidence_hash: str
    occurred_at: datetime


class ReleaseApprovalResponse(BaseModel):
    approval_id: str
    manifest_hash: str
    status: str
    requested_by_subject_id: str
    decided_by_subject_id: str | None
    decision_reason: str | None
    expires_at: datetime
    decided_at: datetime | None
    version: int


class ModelReleaseResponse(BaseModel):
    release_id: str
    evaluation_id: str
    candidate_experiment_id: str
    status: str
    target_environment: str
    manifest: dict[str, Any]
    manifest_hash: str
    rollback_release_id: str | None
    traffic_percent: float
    failure_reason: str | None
    created_by_subject_id: str
    version: int
    created_at: datetime
    approval: ReleaseApprovalResponse | None
    transitions: list[ReleaseTransitionResponse]
    legal_actions: list[str]


class ModelReleaseEnvelope(BaseModel):
    data: ModelReleaseResponse
    meta: dict[str, str]


class ModelReleaseListEnvelope(BaseModel):
    data: list[ModelReleaseResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]
    mcp_server_versions: dict[str, str]


@router.post("/model-releases", response_model=ModelReleaseEnvelope, status_code=201)
async def create_model_release(
    body: ReleaseManifestBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)],
) -> ModelReleaseEnvelope:
    try:
        if body.mcp_server_versions != gateway.mcp_server_versions():
            raise ValueError("mcp_server_versions must match the active governed MCP deployment")
        aggregate = ModelReleaseService(database, authorizer).create(
            identity,
            ReleaseManifestPlan(**body.model_dump()),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, ReleaseNotVisible, ReleaseConflict) as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc
    return ModelReleaseEnvelope(
        data=_response(aggregate, identity, authorizer),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/model-releases/{release_id}/supply-chain-successor",
    response_model=ModelReleaseEnvelope,
    status_code=201,
)
def create_supply_chain_successor(
    release_id: str,
    body: SupplyChainSuccessorBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)],
) -> ModelReleaseEnvelope:
    try:
        aggregate = ModelReleaseService(database, authorizer).create_supply_chain_successor(
            identity,
            release_id,
            **body.model_dump(),
            expected_version=_version(if_match),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, ReleaseNotVisible, ReleaseConflict) as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc
    return ModelReleaseEnvelope(
        data=_response(aggregate, identity, authorizer),
        meta={"request_id": request.state.request_id},
    )


@router.get("/model-releases", response_model=ModelReleaseListEnvelope)
async def list_model_releases(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> ModelReleaseListEnvelope:
    try:
        aggregates = ModelReleaseService(database, authorizer).list(
            identity, request_id=request.state.request_id
        )
    except (AuthorizationDenied, ReleaseNotVisible, ReleaseConflict) as exc:
        raise _translate(exc) from exc
    data = [_response(item, identity, authorizer) for item in aggregates]
    return ModelReleaseListEnvelope(
        data=data,
        meta={"request_id": request.state.request_id, "total": len(data)},
        legal_actions=(
            ["CREATE_MODEL_RELEASE"]
            if _allowed(identity, authorizer, Action.CREATE_MODEL_RELEASE, "model-releases")
            else []
        ),
        mcp_server_versions=gateway.mcp_server_versions(),
    )


@router.get("/model-releases/{release_id}", response_model=ModelReleaseEnvelope)
async def get_model_release(
    release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ModelReleaseEnvelope:
    try:
        aggregate = ModelReleaseService(database, authorizer).get(
            identity, release_id, request_id=request.state.request_id
        )
    except (AuthorizationDenied, ReleaseNotVisible, ReleaseConflict) as exc:
        raise _translate(exc) from exc
    return ModelReleaseEnvelope(
        data=_response(aggregate, identity, authorizer),
        meta={"request_id": request.state.request_id},
    )


@router.post("/model-releases/{release_id}/validate", response_model=ModelReleaseEnvelope)
async def validate_model_release(
    release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ModelReleaseEnvelope:
    try:
        aggregate = ModelReleaseService(database, authorizer).validate(
            identity,
            release_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, ReleaseNotVisible, ReleaseConflict) as exc:
        raise _translate(exc) from exc
    return ModelReleaseEnvelope(
        data=_response(aggregate, identity, authorizer),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/model-releases/{release_id}/submit-approval",
    response_model=ModelReleaseEnvelope,
)
async def submit_model_release_approval(
    release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ModelReleaseEnvelope:
    try:
        aggregate = ModelReleaseService(database, authorizer).submit_for_approval(
            identity,
            release_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, ReleaseNotVisible, ReleaseConflict) as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc
    return ModelReleaseEnvelope(
        data=_response(aggregate, identity, authorizer),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/model-releases/{release_id}/approval-decisions",
    response_model=ModelReleaseEnvelope,
)
async def decide_model_release_approval(
    release_id: str,
    body: ReleaseApprovalDecisionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ModelReleaseEnvelope:
    try:
        aggregate = ModelReleaseService(database, authorizer).decide_approval(
            identity,
            release_id,
            expected_version=_version(if_match),
            expected_approval_version=body.approval_version,
            decision=body.decision,
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, ReleaseNotVisible, ReleaseConflict) as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc
    return ModelReleaseEnvelope(
        data=_response(aggregate, identity, authorizer),
        meta={"request_id": request.state.request_id},
    )


def _response(
    aggregate: ReleaseAggregate,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> ModelReleaseResponse:
    release = aggregate.release
    approval = aggregate.approval
    legal_actions: list[str] = []
    if (
        release.status == "DRAFT"
        and not release.manifest_json.get("prompt_evaluation_only")
        and _allowed(identity, authorizer, Action.VALIDATE_MODEL_RELEASE, release.release_id)
    ):
        legal_actions.append("VALIDATE_MODEL_RELEASE")
    if release.status == "CANDIDATE" and _allowed(
        identity,
        authorizer,
        Action.SUBMIT_MODEL_RELEASE_APPROVAL,
        release.release_id,
    ):
        legal_actions.append("SUBMIT_MODEL_RELEASE_APPROVAL")
    if (
        release.status == "APPROVAL_PENDING"
        and approval is not None
        and approval.status == "PENDING"
        and approval.requested_by_subject_id != identity.subject_id
        and _allowed(
            identity,
            authorizer,
            Action.DECIDE_MODEL_RELEASE_APPROVAL,
            release.release_id,
        )
    ):
        legal_actions.append("DECIDE_MODEL_RELEASE_APPROVAL")
    return ModelReleaseResponse(
        release_id=release.release_id,
        evaluation_id=release.evaluation_id,
        candidate_experiment_id=release.candidate_experiment_id,
        status=release.status,
        target_environment=release.target_environment,
        manifest=release.manifest_json,
        manifest_hash=release.manifest_hash,
        rollback_release_id=release.rollback_release_id,
        traffic_percent=release.traffic_percent,
        failure_reason=release.failure_reason,
        created_by_subject_id=release.created_by_subject_id,
        version=release.version,
        created_at=release.created_at,
        approval=_approval_response(approval) if approval is not None else None,
        transitions=[_transition_response(item) for item in aggregate.transitions],
        legal_actions=legal_actions,
    )


def _approval_response(record: ModelReleaseApprovalRecord) -> ReleaseApprovalResponse:
    return ReleaseApprovalResponse(
        approval_id=record.approval_id,
        manifest_hash=record.manifest_hash,
        status=record.status,
        requested_by_subject_id=record.requested_by_subject_id,
        decided_by_subject_id=record.decided_by_subject_id,
        decision_reason=record.decision_reason,
        expires_at=record.expires_at,
        decided_at=record.decided_at,
        version=record.version,
    )


def _transition_response(record: ModelReleaseTransitionRecord) -> ReleaseTransitionResponse:
    return ReleaseTransitionResponse(
        transition_id=record.transition_id,
        sequence=record.sequence,
        from_status=record.from_status,
        to_status=record.to_status,
        actor_subject_id=record.actor_subject_id,
        reason_code=record.reason_code,
        evidence=record.evidence_json,
        evidence_hash=record.evidence_hash,
        occurred_at=record.occurred_at,
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


def _translate(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(403, "authorization_denied", "authorization", "Action is not allowed")
    if isinstance(exc, ReleaseNotVisible):
        return AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        )
    if isinstance(exc, ReleaseConflict):
        return AppError(
            409,
            exc.reason,
            "release_governance",
            "Model release governance rejected the operation",
            details={"current_version": exc.current_version},
        )
    raise TypeError("unsupported model release error")


def _invalid(exc: ValueError) -> AppError:
    return AppError(400, "invalid_model_release_request", "validation", str(exc))
