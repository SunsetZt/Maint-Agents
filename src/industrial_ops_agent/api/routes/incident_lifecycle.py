"""M2 explicit incident submission and formal incident query API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.application.incidents import IdempotencyConflict, IncidentLifecycleService
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.domain.incidents import (
    DraftStateConflict,
    DraftVersionConflict,
    Incident,
    IncidentStateConflict,
    IncidentSubmissionBlocked,
    IncidentVersionConflict,
)
from industrial_ops_agent.persistence.database import Database

router = APIRouter(tags=["incidents"])


class SubmitIncidentRequest(BaseModel):
    evidence_bundle_id: str = Field(min_length=1, max_length=128)


class IncidentResponse(BaseModel):
    incident_id: str
    asset_id: str
    source_draft_id: str
    evidence_bundle_id: str
    reporter_subject_id: str
    description: str
    status: str
    version: int
    diagnosis_run_ids: list[str]
    created_at: datetime
    updated_at: datetime


class IncidentEnvelope(BaseModel):
    data: IncidentResponse
    meta: dict[str, str | bool | int]


@router.post(
    "/incident-drafts/{draft_id}/submit",
    response_model=IncidentEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
def submit_incident(
    draft_id: str,
    payload: SubmitIncidentRequest,
    response: Response,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> IncidentEnvelope:
    try:
        result = IncidentLifecycleService(database, authorizer).submit(
            identity,
            draft_id,
            evidence_bundle_id=payload.evidence_bundle_id,
            expected_version=_parse_version(if_match),
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
            message="Idempotency key or submitted evidence conflicts",
        ) from exc
    except DraftVersionConflict as exc:
        raise AppError(
            status_code=409,
            code="version_conflict",
            category="conflict",
            message="Incident draft version conflict",
            details={"current_version": exc.current_version},
        ) from exc
    except DraftStateConflict as exc:
        raise AppError(
            status_code=409,
            code="draft_state_conflict",
            category="conflict",
            message=str(exc),
        ) from exc
    except IncidentSubmissionBlocked as exc:
        raise AppError(
            status_code=422,
            code=exc.reason_code,
            category="validation",
            message="Incident submission preconditions are incomplete",
        ) from exc
    response.status_code = 201 if result.created else 200
    return _envelope(result.incident, request.state.request_id, result.created)


@router.get(
    "/incidents/{incident_id}",
    response_model=IncidentEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_incident(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> IncidentEnvelope:
    try:
        incident = IncidentLifecycleService(database, authorizer).get(
            identity,
            incident_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    return _envelope(incident, request.state.request_id, False)


@router.post(
    "/incidents/{incident_id}/triage",
    response_model=IncidentEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def triage_incident(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> IncidentEnvelope:
    try:
        incident = IncidentLifecycleService(database, authorizer).triage(
            identity,
            incident_id,
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except IncidentVersionConflict as exc:
        raise AppError(
            status_code=409,
            code="version_conflict",
            category="conflict",
            message="Incident version conflict",
            details={"current_version": exc.current_version},
        ) from exc
    except IncidentStateConflict as exc:
        raise AppError(
            status_code=409,
            code="incident_state_conflict",
            category="conflict",
            message=str(exc),
        ) from exc
    return _envelope(incident, request.state.request_id, False)


def _envelope(incident: Incident, request_id: str, created: bool) -> IncidentEnvelope:
    return IncidentEnvelope(
        data=IncidentResponse(
            incident_id=incident.incident_id,
            asset_id=incident.asset_id,
            source_draft_id=incident.source_draft_id,
            evidence_bundle_id=incident.evidence_bundle_id,
            reporter_subject_id=incident.reporter_subject_id,
            description=incident.description,
            status=incident.status,
            version=incident.version,
            diagnosis_run_ids=list(incident.diagnosis_run_ids),
            created_at=incident.created_at,
            updated_at=incident.updated_at,
        ),
        meta={"request_id": request_id, "created": created, "version": incident.version},
    )


def _parse_version(value: str) -> int:
    normalized = value.strip().strip('"')
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
