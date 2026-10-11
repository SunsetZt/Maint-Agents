"""Governed WorkOrder remote expert collaboration HTTP surface."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_realtime_session_service,
)
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.api.routes.realtime import (
    IceServerResponse,
    RealtimeSessionGrantEnvelope,
    RealtimeSessionGrantResponse,
)
from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.multimodal.realtime import (
    RealtimeSessionError,
    RealtimeSessionGrant,
    RealtimeSessionService,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.workorders.expert_collaboration import (
    ExpertCandidate,
    ExpertCollaboration,
    ExpertCollaborationConflict,
    ExpertCollaborationNotVisible,
    ExpertCollaborationOverview,
    ExpertCollaborationService,
    ExpertRecommendation,
)

router = APIRouter(tags=["expert-collaboration"])


class CreateExpertCollaborationRequest(BaseModel):
    work_order_version: int = Field(ge=1)
    invited_subject_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=10, max_length=2_000)


class EndExpertCollaborationRequest(BaseModel):
    reason: str = Field(min_length=4, max_length=2_000)


class SubmitExpertRecommendationRequest(BaseModel):
    recommendation_type: Literal[
        "ADVICE", "REQUEST_MORE_EVIDENCE", "STOP_AND_ESCALATE"
    ]
    summary: str = Field(min_length=10, max_length=2_000)
    basis: str = Field(min_length=10, max_length=4_000)
    recommended_checks: list[str] = Field(min_length=1, max_length=10)
    safety_notice: str | None = Field(default=None, min_length=4, max_length=2_000)
    evidence_entry_ids: list[str] = Field(default_factory=list, max_length=50)


class DecideExpertRecommendationRequest(BaseModel):
    decision: Literal["ACCEPT", "REJECT"]
    reason: str = Field(min_length=4, max_length=2_000)


class ExpertCandidateResponse(BaseModel):
    subject_id: str
    display_name: str


class ExpertCollaborationEventResponse(BaseModel):
    sequence: int
    event_type: str
    actor_subject_id: str
    from_status: str | None
    to_status: str
    reason_code: str
    occurred_at: datetime


class ExpertRecommendationResponse(BaseModel):
    recommendation_id: str
    collaboration_id: str
    work_order_id: str
    work_order_version: int
    incident_id: str
    asset_id: str
    expert_subject_id: str
    recommendation_type: str
    summary: str
    basis: str
    recommended_checks: list[str]
    safety_notice: str | None
    evidence_entry_ids: list[str]
    status: str
    reviewed_by_subject_id: str | None
    review_reason: str | None
    reviewed_at: datetime | None
    field_entry_id: str | None
    version: int
    legal_actions: list[str]


class ExpertCollaborationResponse(BaseModel):
    collaboration_id: str
    work_order_id: str
    work_order_version: int
    incident_id: str
    asset_id: str
    requested_by_subject_id: str
    invited_subject_id: str
    reason: str
    status: str
    accepted_at: datetime | None
    ended_by_subject_id: str | None
    ended_reason: str | None
    ended_at: datetime | None
    expires_at: datetime
    version: int
    recommendation: ExpertRecommendationResponse | None
    legal_actions: list[str]
    events: list[ExpertCollaborationEventResponse]


class ExpertCollaborationOverviewResponse(BaseModel):
    work_order_id: str
    work_order_version: int
    collaboration: ExpertCollaborationResponse | None
    candidates: list[ExpertCandidateResponse]
    legal_actions: list[str]


class ExpertCollaborationEnvelope(BaseModel):
    data: ExpertCollaborationResponse
    meta: dict[str, str]


class ExpertCollaborationListEnvelope(BaseModel):
    data: list[ExpertCollaborationResponse]
    meta: dict[str, str]


class ExpertCollaborationOverviewEnvelope(BaseModel):
    data: ExpertCollaborationOverviewResponse
    meta: dict[str, str]


@router.get(
    "/field/work-orders/{work_order_id}/expert-collaboration",
    response_model=ExpertCollaborationOverviewEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
    tags=["field-service"],
)
def get_field_expert_collaboration(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ExpertCollaborationOverviewEnvelope:
    try:
        overview = ExpertCollaborationService(database, authorizer).field_overview(
            identity, work_order_id, request_id=request.state.request_id
        )
    except ExpertCollaborationNotVisible as exc:
        raise _not_visible() from exc
    except ExpertCollaborationConflict as exc:
        raise _conflict(exc) from exc
    return ExpertCollaborationOverviewEnvelope(
        data=_overview_response(overview),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/field/work-orders/{work_order_id}/expert-collaborations",
    response_model=ExpertCollaborationEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
    tags=["field-service"],
)
def create_expert_collaboration(
    work_order_id: str,
    body: CreateExpertCollaborationRequest,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)],
) -> ExpertCollaborationEnvelope:
    try:
        collaboration, created = ExpertCollaborationService(database, authorizer).create(
            identity,
            work_order_id,
            work_order_version=body.work_order_version,
            invited_subject_id=body.invited_subject_id,
            reason=body.reason,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except ExpertCollaborationNotVisible as exc:
        raise _not_visible() from exc
    except ExpertCollaborationConflict as exc:
        raise _conflict(exc) from exc
    if not created:
        response.status_code = 200
    return ExpertCollaborationEnvelope(
        data=_collaboration_response(collaboration),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/expert-collaborations",
    response_model=ExpertCollaborationListEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def list_expert_collaborations(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ExpertCollaborationListEnvelope:
    try:
        records = ExpertCollaborationService(database, authorizer).list_for_expert(
            identity, request_id=request.state.request_id
        )
    except ExpertCollaborationNotVisible as exc:
        raise _not_visible() from exc
    return ExpertCollaborationListEnvelope(
        data=[_collaboration_response(item) for item in records],
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/expert-collaborations/{collaboration_id}",
    response_model=ExpertCollaborationEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_expert_collaboration(
    collaboration_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ExpertCollaborationEnvelope:
    try:
        record = ExpertCollaborationService(database, authorizer).get(
            identity, collaboration_id, request_id=request.state.request_id
        )
    except ExpertCollaborationNotVisible as exc:
        raise _not_visible() from exc
    return ExpertCollaborationEnvelope(
        data=_collaboration_response(record),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/expert-collaborations/{collaboration_id}/accept",
    response_model=ExpertCollaborationEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def accept_expert_collaboration(
    collaboration_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ExpertCollaborationEnvelope:
    try:
        record = ExpertCollaborationService(database, authorizer).accept(
            identity,
            collaboration_id,
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        )
    except ExpertCollaborationNotVisible as exc:
        raise _not_visible() from exc
    except ExpertCollaborationConflict as exc:
        raise _conflict(exc) from exc
    return ExpertCollaborationEnvelope(
        data=_collaboration_response(record),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/expert-collaborations/{collaboration_id}/end",
    response_model=ExpertCollaborationEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def end_expert_collaboration(
    collaboration_id: str,
    body: EndExpertCollaborationRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ExpertCollaborationEnvelope:
    try:
        record = ExpertCollaborationService(database, authorizer).end(
            identity,
            collaboration_id,
            expected_version=_parse_version(if_match),
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except ExpertCollaborationNotVisible as exc:
        raise _not_visible() from exc
    except ExpertCollaborationConflict as exc:
        raise _conflict(exc) from exc
    return ExpertCollaborationEnvelope(
        data=_collaboration_response(record),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/expert-collaborations/{collaboration_id}/recommendations",
    response_model=ExpertCollaborationEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
def submit_expert_recommendation(
    collaboration_id: str,
    body: SubmitExpertRecommendationRequest,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8)],
) -> ExpertCollaborationEnvelope:
    try:
        record, created = ExpertCollaborationService(
            database, authorizer
        ).submit_recommendation(
            identity,
            collaboration_id,
            recommendation_type=body.recommendation_type,
            summary=body.summary,
            basis=body.basis,
            recommended_checks=tuple(body.recommended_checks),
            safety_notice=body.safety_notice,
            evidence_entry_ids=tuple(body.evidence_entry_ids),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except ExpertCollaborationNotVisible as exc:
        raise _not_visible() from exc
    except ExpertCollaborationConflict as exc:
        raise _conflict(exc) from exc
    if not created:
        response.status_code = 200
    return ExpertCollaborationEnvelope(
        data=_collaboration_response(record),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/expert-collaborations/{collaboration_id}/recommendation/decision",
    response_model=ExpertCollaborationEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def decide_expert_recommendation(
    collaboration_id: str,
    body: DecideExpertRecommendationRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ExpertCollaborationEnvelope:
    try:
        record = ExpertCollaborationService(database, authorizer).decide_recommendation(
            identity,
            collaboration_id,
            expected_version=_parse_version(if_match),
            decision=body.decision,
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except ExpertCollaborationNotVisible as exc:
        raise _not_visible() from exc
    except ExpertCollaborationConflict as exc:
        raise _conflict(exc) from exc
    return ExpertCollaborationEnvelope(
        data=_collaboration_response(record),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/expert-collaborations/{collaboration_id}/realtime-sessions",
    response_model=RealtimeSessionGrantEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
def create_expert_collaboration_realtime_session(
    collaboration_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[RealtimeSessionService, Depends(get_realtime_session_service)],
) -> RealtimeSessionGrantEnvelope:
    try:
        grant = service.create_expert_collaboration(
            identity, collaboration_id, request_id=request.state.request_id
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except RealtimeSessionError as exc:
        raise AppError(
            409,
            exc.reason,
            "state",
            "Expert collaboration media session state conflict",
        ) from exc
    return RealtimeSessionGrantEnvelope(
        data=_grant_response(grant),
        meta={"request_id": request.state.request_id},
    )


def _candidate_response(candidate: ExpertCandidate) -> ExpertCandidateResponse:
    return ExpertCandidateResponse(
        subject_id=candidate.subject_id,
        display_name=candidate.display_name,
    )


def _overview_response(
    overview: ExpertCollaborationOverview,
) -> ExpertCollaborationOverviewResponse:
    return ExpertCollaborationOverviewResponse(
        work_order_id=overview.work_order_id,
        work_order_version=overview.work_order_version,
        collaboration=(
            _collaboration_response(overview.collaboration)
            if overview.collaboration is not None
            else None
        ),
        candidates=[_candidate_response(item) for item in overview.candidates],
        legal_actions=list(overview.legal_actions),
    )


def _collaboration_response(
    collaboration: ExpertCollaboration,
) -> ExpertCollaborationResponse:
    return ExpertCollaborationResponse(
        collaboration_id=collaboration.collaboration_id,
        work_order_id=collaboration.work_order_id,
        work_order_version=collaboration.work_order_version,
        incident_id=collaboration.incident_id,
        asset_id=collaboration.asset_id,
        requested_by_subject_id=collaboration.requested_by_subject_id,
        invited_subject_id=collaboration.invited_subject_id,
        reason=collaboration.reason,
        status=collaboration.status,
        accepted_at=collaboration.accepted_at,
        ended_by_subject_id=collaboration.ended_by_subject_id,
        ended_reason=collaboration.ended_reason,
        ended_at=collaboration.ended_at,
        expires_at=collaboration.expires_at,
        version=collaboration.version,
        recommendation=(
            _recommendation_response(collaboration.recommendation)
            if collaboration.recommendation is not None
            else None
        ),
        legal_actions=list(collaboration.legal_actions),
        events=[
            ExpertCollaborationEventResponse(
                sequence=item.sequence,
                event_type=item.event_type,
                actor_subject_id=item.actor_subject_id,
                from_status=item.from_status,
                to_status=item.to_status,
                reason_code=item.reason_code,
                occurred_at=item.occurred_at,
            )
            for item in collaboration.events
        ],
    )


def _recommendation_response(
    recommendation: ExpertRecommendation,
) -> ExpertRecommendationResponse:
    return ExpertRecommendationResponse(
        recommendation_id=recommendation.recommendation_id,
        collaboration_id=recommendation.collaboration_id,
        work_order_id=recommendation.work_order_id,
        work_order_version=recommendation.work_order_version,
        incident_id=recommendation.incident_id,
        asset_id=recommendation.asset_id,
        expert_subject_id=recommendation.expert_subject_id,
        recommendation_type=recommendation.recommendation_type,
        summary=recommendation.summary,
        basis=recommendation.basis,
        recommended_checks=list(recommendation.recommended_checks),
        safety_notice=recommendation.safety_notice,
        evidence_entry_ids=list(recommendation.evidence_entry_ids),
        status=recommendation.status,
        reviewed_by_subject_id=recommendation.reviewed_by_subject_id,
        review_reason=recommendation.review_reason,
        reviewed_at=recommendation.reviewed_at,
        field_entry_id=recommendation.field_entry_id,
        version=recommendation.version,
        legal_actions=list(recommendation.legal_actions),
    )


def _grant_response(grant: RealtimeSessionGrant) -> RealtimeSessionGrantResponse:
    session = grant.session
    return RealtimeSessionGrantResponse(
        session_id=session.session_id,
        incident_id=session.incident_id,
        diagnosis_run_id=session.diagnosis_run_id,
        work_order_id=session.work_order_id,
        work_order_version=session.work_order_version,
        expert_collaboration_id=session.expert_collaboration_id,
        participant_subject_id=session.participant_subject_id,
        status=session.status,
        media_kind=session.media_kind,
        token_expires_at=session.token_expires_at,
        lease_expires_at=session.lease_expires_at,
        max_expires_at=session.max_expires_at,
        reconnect_count=session.reconnect_count,
        version=session.version,
        access_token=grant.access_token,
        signaling_url=grant.signaling_url,
        ice_servers=[
            IceServerResponse(
                urls=list(item.urls),
                username=item.username,
                credential=item.credential,
            )
            for item in grant.ice_servers
        ],
        heartbeat_interval_seconds=grant.heartbeat_interval_seconds,
    )


def _not_visible() -> AppError:
    return AppError(
        404,
        "expert_collaboration_not_found",
        "not_found",
        "Expert collaboration not found",
    )


def _conflict(exc: ExpertCollaborationConflict) -> AppError:
    return AppError(
        409,
        exc.reason,
        "state",
        "Expert collaboration state conflict",
        details=(
            {"current_version": exc.current_version} if exc.current_version is not None else None
        ),
    )


def _parse_version(value: str) -> int:
    normalized = value.strip().removeprefix("W/").strip('"')
    try:
        version = int(normalized)
    except ValueError as exc:
        raise AppError(
            400,
            "invalid_version_precondition",
            "validation",
            "If-Match is invalid",
        ) from exc
    if version < 1:
        raise AppError(
            400,
            "invalid_version_precondition",
            "validation",
            "If-Match is invalid",
        )
    return version
