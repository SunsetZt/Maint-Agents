"""M1 idempotent incident-draft and read-only timeline API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
)
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.application.incidents import (
    IdempotencyConflict,
    IncidentDraftService,
)
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.domain.incidents import (
    DraftTimelineEvent,
    DraftVersionConflict,
    IncidentDraft,
)
from industrial_ops_agent.persistence.database import Database

router = APIRouter(prefix="/incident-drafts", tags=["incident-drafts"])


class CreateDraftRequest(BaseModel):
    asset_id: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=10_000)


class UpdateDraftRequest(BaseModel):
    description: str = Field(min_length=1, max_length=10_000)


class IncidentDraftResponse(BaseModel):
    draft_id: str
    asset_id: str
    author_subject_id: str
    description: str
    version: int
    status: str
    created_at: datetime
    updated_at: datetime


class DraftEnvelope(BaseModel):
    data: IncidentDraftResponse
    meta: dict[str, str | int]


class TimelineEventResponse(BaseModel):
    sequence: int
    event_type: str
    occurred_at: datetime
    payload_metadata: dict[str, Any]


class TimelineEnvelope(BaseModel):
    data: list[TimelineEventResponse]
    meta: dict[str, str]


@router.post(
    "",
    response_model=DraftEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
def create_draft(
    payload: CreateDraftRequest,
    response: Response,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> DraftEnvelope:
    service = IncidentDraftService(database, authorizer)
    try:
        result = service.create(
            identity,
            asset_id=payload.asset_id,
            description=payload.description,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except IdempotencyConflict as exc:
        raise AppError(
            status_code=409,
            code="idempotency_conflict",
            category="conflict",
            message="Idempotency key was already used for another request",
        ) from exc
    response.status_code = 201 if result.created else 200
    return _draft_envelope(result.draft, request.state.request_id)


@router.get(
    "/{draft_id}",
    response_model=DraftEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_draft(
    draft_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> DraftEnvelope:
    service = IncidentDraftService(database, authorizer)
    try:
        draft = service.get(identity, draft_id, request_id=request.state.request_id)
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    return _draft_envelope(draft, request.state.request_id)


@router.patch(
    "/{draft_id}",
    response_model=DraftEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def update_draft(
    draft_id: str,
    payload: UpdateDraftRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> DraftEnvelope:
    expected_version = _parse_version(if_match)
    service = IncidentDraftService(database, authorizer)
    try:
        draft = service.update_description(
            identity,
            draft_id,
            description=payload.description,
            expected_version=expected_version,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except DraftVersionConflict as exc:
        raise AppError(
            status_code=409,
            code="version_conflict",
            category="conflict",
            message="Incident draft version conflict",
            details={"current_version": exc.current_version},
        ) from exc
    return _draft_envelope(draft, request.state.request_id)


@router.get(
    "/{draft_id}/timeline",
    response_model=TimelineEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_timeline(
    draft_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> TimelineEnvelope:
    service = IncidentDraftService(database, authorizer)
    try:
        timeline = service.timeline(
            identity,
            draft_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    return TimelineEnvelope(
        data=[_timeline_response(event) for event in timeline],
        meta={"request_id": request.state.request_id},
    )


def _parse_version(value: str) -> int:
    normalized = value.strip()
    if normalized.startswith('"') and normalized.endswith('"'):
        normalized = normalized[1:-1]
    try:
        version = int(normalized)
    except ValueError as exc:
        raise _invalid_version() from exc
    if version < 1:
        raise _invalid_version()
    return version


def _invalid_version() -> AppError:
    return AppError(
        status_code=400,
        code="invalid_version_precondition",
        category="validation",
        message="If-Match must contain a positive draft version",
    )


def _not_visible() -> AppError:
    return AppError(
        status_code=404,
        code="resource_not_found_or_not_visible",
        category="not_found",
        message="Resource not found or not visible",
    )


def _draft_envelope(draft: IncidentDraft, request_id: str) -> DraftEnvelope:
    return DraftEnvelope(
        data=IncidentDraftResponse(
            draft_id=draft.draft_id,
            asset_id=draft.asset_id,
            author_subject_id=draft.author_subject_id,
            description=draft.description,
            version=draft.version,
            status=draft.status,
            created_at=draft.created_at,
            updated_at=draft.updated_at,
        ),
        meta={"request_id": request_id, "version": draft.version},
    )


def _timeline_response(event: DraftTimelineEvent) -> TimelineEventResponse:
    return TimelineEventResponse(
        sequence=event.sequence,
        event_type=event.event_type,
        occurred_at=event.occurred_at,
        payload_metadata=event.payload_metadata,
    )
