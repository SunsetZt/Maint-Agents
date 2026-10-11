"""IAM-005 tenant membership, access and retention administration API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.tenant_administration.service import (
    HUMAN_ASSIGNABLE_ROLES,
    TenantAdministrationConflict,
    TenantAdministrationNotVisible,
    TenantAdministrationService,
    TenantAdministrationView,
    TenantAssetOption,
    TenantMemberView,
    TenantOverview,
    TenantRetentionPolicyView,
    TenantSiteOption,
)

router = APIRouter(prefix="/tenant", tags=["tenant-administration"])


class TenantOverviewResponse(BaseModel):
    tenant_id: str
    display_name: str | None
    status: str
    version: int
    member_count: int
    active_member_count: int
    asset_count: int
    site_count: int


class RetentionPolicyResponse(BaseModel):
    policy_id: str | None
    configured: bool
    incident_days: int | None
    media_days: int | None
    knowledge_days: int | None
    training_data_days: int | None
    audit_days: int | None
    legal_hold: bool
    updated_by_subject_id: str | None
    version: int
    updated_at: datetime | None


class TenantOverviewEnvelope(BaseModel):
    data: TenantOverviewResponse
    retention_policy: RetentionPolicyResponse
    assignable_roles: list[str]
    assets: list[TenantAssetOptionResponse]
    sites: list[TenantSiteOptionResponse]
    meta: dict[str, str]


class TenantAssetOptionResponse(BaseModel):
    asset_id: str
    display_name: str | None
    model_code: str | None


class TenantSiteOptionResponse(BaseModel):
    site_id: str
    site_name: str


class TenantMemberResponse(BaseModel):
    subject_id: str
    oidc_issuer: str
    oidc_subject: str
    display_name: str | None
    email: str | None
    status: str
    roles: list[str]
    asset_ids: list[str]
    site_ids: list[str]
    version: int
    created_at: datetime
    updated_at: datetime


class TenantMemberEnvelope(BaseModel):
    data: TenantMemberResponse
    meta: dict[str, str | int | bool]


class TenantMembersEnvelope(BaseModel):
    data: list[TenantMemberResponse]
    meta: dict[str, str | int]


class TenantAdministrationResponse(BaseModel):
    administration_id: str
    command_type: str
    target_type: str
    target_id: str
    actor_subject_id: str
    reason: str
    previous_version: int
    target_version: int
    before: dict[str, Any]
    after: dict[str, Any]
    occurred_at: datetime


class TenantAdministrationsEnvelope(BaseModel):
    data: list[TenantAdministrationResponse]
    meta: dict[str, str | int]


class MemberAccessBody(BaseModel):
    display_name: str | None = Field(default=None, max_length=255)
    email: str | None = Field(default=None, max_length=320)
    roles: list[str] = Field(min_length=1, max_length=16)
    asset_ids: list[str] = Field(default_factory=list, max_length=1_000)
    site_ids: list[str] = Field(default_factory=list, max_length=1_000)
    reason: str = Field(min_length=2, max_length=2_000)


class CreateTenantMemberBody(MemberAccessBody):
    subject_id: str = Field(min_length=1, max_length=128)
    oidc_issuer: str = Field(min_length=1, max_length=512)
    oidc_subject: str = Field(min_length=1, max_length=255)


class UpdateTenantMemberBody(MemberAccessBody):
    status: Literal["ACTIVE", "SUSPENDED"]


class UpdateRetentionPolicyBody(BaseModel):
    incident_days: int = Field(ge=1, le=36_500)
    media_days: int = Field(ge=1, le=36_500)
    knowledge_days: int = Field(ge=1, le=36_500)
    training_data_days: int = Field(ge=1, le=36_500)
    audit_days: int = Field(ge=1, le=36_500)
    legal_hold: bool = False
    reason: str = Field(min_length=2, max_length=2_000)


@router.get("", response_model=TenantOverviewEnvelope, responses=STANDARD_ERROR_RESPONSES)
async def get_tenant_overview(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> TenantOverviewEnvelope:
    try:
        service = TenantAdministrationService(database, authorizer)
        tenant, policy = service.overview(identity, request_id=request.state.request_id)
        assets, sites = service.scope_catalog(identity, request_id=request.state.request_id)
    except TenantAdministrationNotVisible as exc:
        raise _not_visible() from exc
    return TenantOverviewEnvelope(
        data=_tenant_response(tenant),
        retention_policy=_retention_response(policy),
        assignable_roles=[role.value for role in HUMAN_ASSIGNABLE_ROLES],
        assets=[_asset_option_response(item) for item in assets],
        sites=[_site_option_response(item) for item in sites],
        meta={"request_id": request.state.request_id},
    )


@router.get("/members", response_model=TenantMembersEnvelope, responses=STANDARD_ERROR_RESPONSES)
async def list_tenant_members(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[str | None, Query(max_length=24)] = None,
    role: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TenantMembersEnvelope:
    members, total = TenantAdministrationService(database, authorizer).list_members(
        identity,
        status=status,
        role=role,
        limit=limit,
        offset=offset,
        request_id=request.state.request_id,
    )
    return TenantMembersEnvelope(
        data=[_member_response(member) for member in members],
        meta={
            "request_id": request.state.request_id,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
    )


@router.post(
    "/members",
    response_model=TenantMemberEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
async def create_tenant_member(
    body: CreateTenantMemberBody,
    response: Response,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)],
) -> TenantMemberEnvelope:
    try:
        member, created = TenantAdministrationService(database, authorizer).create_member(
            identity,
            subject_id=body.subject_id,
            oidc_issuer=body.oidc_issuer,
            oidc_subject=body.oidc_subject,
            display_name=body.display_name,
            email=body.email,
            roles=tuple(body.roles),
            asset_ids=tuple(body.asset_ids),
            site_ids=tuple(body.site_ids),
            reason=body.reason,
            client_operation_id=idempotency_key,
            request_id=request.state.request_id,
        )
    except TenantAdministrationNotVisible as exc:
        raise _not_visible() from exc
    except TenantAdministrationConflict as exc:
        raise _conflict(exc) from exc
    response.status_code = 201 if created else 200
    return TenantMemberEnvelope(
        data=_member_response(member),
        meta={
            "request_id": request.state.request_id,
            "version": member.version,
            "created": created,
        },
    )


@router.put(
    "/members/{subject_id}",
    response_model=TenantMemberEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
async def update_tenant_member(
    subject_id: str,
    body: UpdateTenantMemberBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)],
) -> TenantMemberEnvelope:
    try:
        member = TenantAdministrationService(database, authorizer).update_member(
            identity,
            subject_id,
            display_name=body.display_name,
            email=body.email,
            status=body.status,
            roles=tuple(body.roles),
            asset_ids=tuple(body.asset_ids),
            site_ids=tuple(body.site_ids),
            reason=body.reason,
            expected_version=_parse_version(if_match, allow_zero=False),
            client_operation_id=idempotency_key,
            request_id=request.state.request_id,
        )
    except TenantAdministrationNotVisible as exc:
        raise _not_visible() from exc
    except TenantAdministrationConflict as exc:
        raise _conflict(exc) from exc
    return TenantMemberEnvelope(
        data=_member_response(member),
        meta={"request_id": request.state.request_id, "version": member.version},
    )


@router.put(
    "/retention-policy",
    response_model=RetentionPolicyResponse,
    responses=STANDARD_ERROR_RESPONSES,
)
async def update_tenant_retention_policy(
    body: UpdateRetentionPolicyBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)],
) -> RetentionPolicyResponse:
    try:
        policy = TenantAdministrationService(database, authorizer).update_retention_policy(
            identity,
            incident_days=body.incident_days,
            media_days=body.media_days,
            knowledge_days=body.knowledge_days,
            training_data_days=body.training_data_days,
            audit_days=body.audit_days,
            legal_hold=body.legal_hold,
            reason=body.reason,
            expected_version=_parse_version(if_match, allow_zero=True),
            client_operation_id=idempotency_key,
            request_id=request.state.request_id,
        )
    except TenantAdministrationNotVisible as exc:
        raise _not_visible() from exc
    except TenantAdministrationConflict as exc:
        raise _conflict(exc) from exc
    return _retention_response(policy)


@router.get(
    "/administration-records",
    response_model=TenantAdministrationsEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
async def list_tenant_administration_records(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TenantAdministrationsEnvelope:
    records, total = TenantAdministrationService(database, authorizer).list_administration_records(
        identity,
        limit=limit,
        offset=offset,
        request_id=request.state.request_id,
    )
    return TenantAdministrationsEnvelope(
        data=[_administration_response(record) for record in records],
        meta={
            "request_id": request.state.request_id,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
    )


def _tenant_response(item: TenantOverview) -> TenantOverviewResponse:
    return TenantOverviewResponse(
        tenant_id=item.tenant_id,
        display_name=item.display_name,
        status=item.status,
        version=item.version,
        member_count=item.member_count,
        active_member_count=item.active_member_count,
        asset_count=item.asset_count,
        site_count=item.site_count,
    )


def _asset_option_response(item: TenantAssetOption) -> TenantAssetOptionResponse:
    return TenantAssetOptionResponse(
        asset_id=item.asset_id,
        display_name=item.display_name,
        model_code=item.model_code,
    )


def _site_option_response(item: TenantSiteOption) -> TenantSiteOptionResponse:
    return TenantSiteOptionResponse(site_id=item.site_id, site_name=item.site_name)


def _member_response(item: TenantMemberView) -> TenantMemberResponse:
    return TenantMemberResponse(
        subject_id=item.subject_id,
        oidc_issuer=item.oidc_issuer,
        oidc_subject=item.oidc_subject,
        display_name=item.display_name,
        email=item.email,
        status=item.status,
        roles=list(item.roles),
        asset_ids=list(item.asset_ids),
        site_ids=list(item.site_ids),
        version=item.version,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def _retention_response(item: TenantRetentionPolicyView) -> RetentionPolicyResponse:
    return RetentionPolicyResponse(
        policy_id=item.policy_id,
        configured=item.configured,
        incident_days=item.incident_days,
        media_days=item.media_days,
        knowledge_days=item.knowledge_days,
        training_data_days=item.training_data_days,
        audit_days=item.audit_days,
        legal_hold=item.legal_hold,
        updated_by_subject_id=item.updated_by_subject_id,
        version=item.version,
        updated_at=item.updated_at,
    )


def _administration_response(
    item: TenantAdministrationView,
) -> TenantAdministrationResponse:
    return TenantAdministrationResponse(
        administration_id=item.administration_id,
        command_type=item.command_type,
        target_type=item.target_type,
        target_id=item.target_id,
        actor_subject_id=item.actor_subject_id,
        reason=item.reason,
        previous_version=item.previous_version,
        target_version=item.target_version,
        before=item.before,
        after=item.after,
        occurred_at=item.occurred_at,
    )


def _parse_version(value: str, *, allow_zero: bool) -> int:
    try:
        version = int(value.strip().strip('"'))
    except ValueError as exc:
        raise _invalid_version(allow_zero) from exc
    if version < (0 if allow_zero else 1):
        raise _invalid_version(allow_zero)
    return version


def _invalid_version(allow_zero: bool) -> AppError:
    return AppError(
        status_code=400,
        code="invalid_version_precondition",
        category="validation",
        message=(
            "If-Match must contain a non-negative policy version"
            if allow_zero
            else "If-Match must contain a positive member version"
        ),
    )


def _conflict(exc: TenantAdministrationConflict) -> AppError:
    return AppError(
        status_code=409,
        code="tenant_administration_conflict",
        category="conflict",
        message=exc.reason,
        details=(
            {"current_version": exc.current_version} if exc.current_version is not None else None
        ),
    )


def _not_visible() -> AppError:
    return AppError(
        status_code=404,
        code="resource_not_found_or_not_visible",
        category="not_found",
        message="Resource not found or not visible",
    )
