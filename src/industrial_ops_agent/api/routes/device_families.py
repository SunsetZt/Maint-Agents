"""M7 governed device-family onboarding API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.api.versions import numeric_precondition as _version
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.device_families.service import (
    DeviceFamilyConflict,
    DeviceFamilyCreate,
    DeviceFamilyNotVisible,
    DeviceFamilyProfileService,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import DeviceFamilyProfileRecord

router = APIRouter(prefix="/device-families", tags=["device-families"])


class CreateDeviceFamilyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_code: str = Field(min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_-]+$")
    display_name: str = Field(min_length=2, max_length=255)
    risk_class: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    model_codes: list[str] = Field(min_length=1, max_length=100)
    source_system: str = Field(min_length=2, max_length=128)
    source_record_id: str = Field(min_length=1, max_length=255)
    source_as_of: datetime
    knowledge_release_id: str = Field(min_length=3, max_length=128)
    evaluation_suite_id: str = Field(min_length=3, max_length=128)
    evaluation_id: str = Field(min_length=3, max_length=128)
    minimum_gold_samples: int = Field(default=30, ge=1, le=100_000)


class ReviewDeviceFamilyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["ACCEPT", "REJECT"]
    reason: str = Field(min_length=8, max_length=2_000)


class DeviceFamilyResponse(BaseModel):
    profile_id: str
    family_code: str
    family_version: int
    display_name: str
    risk_class: str
    model_codes: list[str]
    source_system: str
    source_record_id: str
    source_as_of: datetime
    knowledge_release_id: str
    evaluation_suite_id: str
    evaluation_id: str
    minimum_gold_samples: int
    readiness: dict[str, Any]
    evidence_digest: str
    configuration_digest: str
    status: Literal["BLOCKED", "READY", "ACTIVE", "REJECTED", "RETIRED"]
    requested_by_subject_id: str
    reviewed_by_subject_id: str | None
    review_decision: Literal["ACCEPT", "REJECT"] | None
    review_reason: str | None
    reviewed_at: datetime | None
    activated_at: datetime | None
    retired_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    legal_actions: list[Literal["REVALIDATE", "ACCEPT", "REJECT"]]


class DeviceFamilyEnvelope(BaseModel):
    data: DeviceFamilyResponse
    meta: dict[str, str | bool]


class DeviceFamilyListEnvelope(BaseModel):
    data: list[DeviceFamilyResponse]
    meta: dict[str, str | int]


@router.get(
    "",
    response_model=DeviceFamilyListEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
async def list_device_families(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[str | None, Query(max_length=32)] = None,
) -> DeviceFamilyListEnvelope:
    try:
        records = await asyncio.to_thread(
            DeviceFamilyProfileService(database, authorizer).list,
            identity,
            status=status,
            request_id=request.state.request_id,
        )
    except DeviceFamilyNotVisible as exc:
        raise _error(exc) from exc
    return DeviceFamilyListEnvelope(
        data=[_response(record, identity, authorizer) for record in records],
        meta={"request_id": request.state.request_id, "total": len(records)},
    )


@router.post(
    "",
    response_model=DeviceFamilyEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
async def create_device_family(
    body: CreateDeviceFamilyBody,
    response: Response,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=255),
    ],
) -> DeviceFamilyEnvelope:
    command = DeviceFamilyCreate(
        family_code=body.family_code,
        display_name=body.display_name,
        risk_class=body.risk_class,
        model_codes=tuple(body.model_codes),
        source_system=body.source_system,
        source_record_id=body.source_record_id,
        source_as_of=body.source_as_of,
        knowledge_release_id=body.knowledge_release_id,
        evaluation_suite_id=body.evaluation_suite_id,
        evaluation_id=body.evaluation_id,
        minimum_gold_samples=body.minimum_gold_samples,
    )
    try:
        record, created = await asyncio.to_thread(
            DeviceFamilyProfileService(database, authorizer).create,
            identity,
            command,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (DeviceFamilyNotVisible, DeviceFamilyConflict, ValueError) as exc:
        raise _error(exc) from exc
    response.status_code = 201 if created else 200
    return _envelope(record, request.state.request_id, identity, authorizer, created=created)


@router.get(
    "/{profile_id}",
    response_model=DeviceFamilyEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
async def get_device_family(
    profile_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> DeviceFamilyEnvelope:
    try:
        record = await asyncio.to_thread(
            DeviceFamilyProfileService(database, authorizer).get,
            identity,
            profile_id,
            request_id=request.state.request_id,
        )
    except DeviceFamilyNotVisible as exc:
        raise _error(exc) from exc
    return _envelope(record, request.state.request_id, identity, authorizer)


@router.post(
    "/{profile_id}/revalidate",
    response_model=DeviceFamilyEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
async def revalidate_device_family(
    profile_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> DeviceFamilyEnvelope:
    try:
        record = await asyncio.to_thread(
            DeviceFamilyProfileService(database, authorizer).revalidate,
            identity,
            profile_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (DeviceFamilyNotVisible, DeviceFamilyConflict) as exc:
        raise _error(exc) from exc
    return _envelope(record, request.state.request_id, identity, authorizer)


@router.post(
    "/{profile_id}/review",
    response_model=DeviceFamilyEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
async def review_device_family(
    profile_id: str,
    body: ReviewDeviceFamilyBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> DeviceFamilyEnvelope:
    try:
        record = await asyncio.to_thread(
            DeviceFamilyProfileService(database, authorizer).review,
            identity,
            profile_id,
            decision=body.decision,
            reason=body.reason,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (DeviceFamilyNotVisible, DeviceFamilyConflict, ValueError) as exc:
        raise _error(exc) from exc
    return _envelope(record, request.state.request_id, identity, authorizer)


def _response(
    record: DeviceFamilyProfileRecord,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> DeviceFamilyResponse:
    resource = ResourceContext(tenant_id=identity.tenant_id, resource_id=record.profile_id)
    actions: list[Literal["REVALIDATE", "ACCEPT", "REJECT"]] = []
    if record.status in {"BLOCKED", "READY"}:
        if authorizer.decide(identity, Action.PROPOSE_DEVICE_FAMILY, resource).allowed:
            actions.append("REVALIDATE")
        can_review = authorizer.decide(
            identity,
            Action.REVIEW_DEVICE_FAMILY,
            resource,
        ).allowed
        if can_review and record.requested_by_subject_id != identity.subject_id:
            if record.status == "READY":
                actions.append("ACCEPT")
            actions.append("REJECT")
    return DeviceFamilyResponse(
        profile_id=record.profile_id,
        family_code=record.family_code,
        family_version=record.family_version,
        display_name=record.display_name,
        risk_class=record.risk_class,
        model_codes=list(record.model_codes_json),
        source_system=record.source_system,
        source_record_id=record.source_record_id,
        source_as_of=record.source_as_of,
        knowledge_release_id=record.knowledge_release_id,
        evaluation_suite_id=record.evaluation_suite_id,
        evaluation_id=record.evaluation_id,
        minimum_gold_samples=record.minimum_gold_samples,
        readiness=dict(record.readiness_json),
        evidence_digest=record.evidence_digest,
        configuration_digest=record.configuration_digest,
        status=record.status,
        requested_by_subject_id=record.requested_by_subject_id,
        reviewed_by_subject_id=record.reviewed_by_subject_id,
        review_decision=record.review_decision,
        review_reason=record.review_reason,
        reviewed_at=record.reviewed_at,
        activated_at=record.activated_at,
        retired_at=record.retired_at,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
        legal_actions=actions,
    )


def _envelope(
    record: DeviceFamilyProfileRecord,
    request_id: str,
    identity: IdentityContext,
    authorizer: Authorizer,
    *,
    created: bool | None = None,
) -> DeviceFamilyEnvelope:
    meta: dict[str, str | bool] = {"request_id": request_id}
    if created is not None:
        meta["created"] = created
    return DeviceFamilyEnvelope(data=_response(record, identity, authorizer), meta=meta)



def _error(exc: Exception) -> AppError:
    if isinstance(exc, DeviceFamilyNotVisible):
        return AppError(404, "device_family_not_found", "visibility", "Not found")
    if isinstance(exc, DeviceFamilyConflict):
        details = (
            {"current_version": exc.current_version} if exc.current_version is not None else None
        )
        return AppError(
            409,
            exc.reason,
            "conflict",
            "Device family request conflicts with current authoritative state",
            details=details,
        )
    if isinstance(exc, ValueError):
        return AppError(400, "invalid_device_family_request", "validation", str(exc))
    return AppError(500, "device_family_error", "internal", "Device family operation failed")
