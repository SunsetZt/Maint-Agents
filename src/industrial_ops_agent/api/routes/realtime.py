"""WebRTC session control API; SDP and media stay on the media service."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from industrial_ops_agent.api.dependencies import (
    get_database,
    get_identity,
    get_realtime_session_service,
    get_realtime_transcript_service,
    get_speech_gateway,
)
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.model_gateway.service import ModelGatewayError
from industrial_ops_agent.multimodal.realtime import (
    IceServer,
    RealtimeMediaAdmission,
    RealtimeSession,
    RealtimeSessionError,
    RealtimeSessionGrant,
    RealtimeSessionService,
    authorize_media_capability,
)
from industrial_ops_agent.multimodal.realtime_transcripts import (
    RealtimeTranscriptSegment,
    RealtimeTranscriptService,
)
from industrial_ops_agent.multimodal.speech import SpeechSynthesisGateway
from industrial_ops_agent.persistence.database import Database

router = APIRouter(tags=["realtime-media"])


class CreateRealtimeSessionRequest(BaseModel):
    diagnosis_run_id: str | None = Field(default=None, max_length=128)
    media_kind: Literal["audio"] = "audio"


class CreateFieldRealtimeSessionRequest(BaseModel):
    diagnosis_run_id: str = Field(min_length=1, max_length=128)
    work_order_version: int = Field(ge=1)
    media_kind: Literal["audio"] = "audio"


class ConsumeRealtimeSessionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    access_token: str = Field(min_length=32, max_length=512)


class IceServerResponse(BaseModel):
    urls: list[str]
    username: str | None = None
    credential: str | None = None


class RealtimeSessionResponse(BaseModel):
    session_id: str
    incident_id: str
    diagnosis_run_id: str | None
    work_order_id: str | None
    work_order_version: int | None
    expert_collaboration_id: str | None = None
    participant_subject_id: str
    status: str
    media_kind: str
    token_expires_at: datetime
    lease_expires_at: datetime
    max_expires_at: datetime
    reconnect_count: int
    version: int


class RealtimeSessionGrantResponse(RealtimeSessionResponse):
    access_token: str
    signaling_url: str
    ice_servers: list[IceServerResponse]
    heartbeat_interval_seconds: int


class RealtimeSessionEnvelope(BaseModel):
    data: RealtimeSessionResponse
    meta: dict[str, str]


class RealtimeSessionGrantEnvelope(BaseModel):
    data: RealtimeSessionGrantResponse
    meta: dict[str, str]


class RealtimeMediaAdmissionResponse(BaseModel):
    session: RealtimeSessionResponse
    media_access_token: str
    expert_collaboration_id: str | None = None
    participant_subject_id: str


class RealtimeMediaAdmissionEnvelope(BaseModel):
    data: RealtimeMediaAdmissionResponse
    meta: dict[str, str]


class TranscriptEntityResponse(BaseModel):
    entity_id: str
    entity_type: str
    value: str
    confidence: float
    requires_confirmation: bool


class RealtimeTranscriptResponse(BaseModel):
    segment_id: str
    session_id: str
    sequence: int
    start_ms: int
    end_ms: int
    text: str
    confirmed_text: str | None
    confidence: float
    language: str
    entities: list[TranscriptEntityResponse]
    resolved_release_id: str
    manifest_hash: str
    status: str
    agent_event_id: str | None
    agent_event_sequence: int | None
    security_policy_version: str | None
    security_findings: list[dict[str, str]]
    version: int


class RealtimeTranscriptEnvelope(BaseModel):
    data: RealtimeTranscriptResponse
    meta: dict[str, str]


class ConfirmRealtimeTranscriptRequest(BaseModel):
    decision: Literal["ACCEPTED", "REJECTED"]
    corrected_text: str | None = Field(default=None, max_length=4_000)


@router.post(
    "/incidents/{incident_id}/realtime-sessions",
    response_model=RealtimeSessionGrantEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
def create_realtime_session(
    incident_id: str,
    body: CreateRealtimeSessionRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[RealtimeSessionService, Depends(get_realtime_session_service)],
) -> RealtimeSessionGrantEnvelope:
    try:
        grant = service.create(
            identity,
            incident_id,
            diagnosis_run_id=body.diagnosis_run_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except RealtimeSessionError as exc:
        raise _session_error(exc) from exc
    return RealtimeSessionGrantEnvelope(
        data=_grant_response(grant),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/field/work-orders/{work_order_id}/realtime-sessions",
    response_model=RealtimeSessionGrantEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
    tags=["field-service"],
)
def create_field_realtime_session(
    work_order_id: str,
    body: CreateFieldRealtimeSessionRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[RealtimeSessionService, Depends(get_realtime_session_service)],
    database: Annotated[Database, Depends(get_database)],
) -> RealtimeSessionGrantEnvelope:
    from industrial_ops_agent.persistence.models import WorkOrderRecord

    with database.transaction(identity.tenant_context) as db_session:
        work = db_session.get(WorkOrderRecord, work_order_id)
        if work is None or work.tenant_id != identity.tenant_id:
            raise _not_visible()
        incident_id = work.incident_id
    try:
        grant = service.create(
            identity,
            incident_id,
            diagnosis_run_id=body.diagnosis_run_id,
            work_order_id=work_order_id,
            work_order_version=body.work_order_version,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except RealtimeSessionError as exc:
        raise _session_error(exc) from exc
    return RealtimeSessionGrantEnvelope(
        data=_grant_response(grant),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/realtime-sessions/consume",
    response_model=RealtimeMediaAdmissionEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def consume_realtime_session(
    body: ConsumeRealtimeSessionRequest,
    request: Request,
    service: Annotated[RealtimeSessionService, Depends(get_realtime_session_service)],
) -> RealtimeMediaAdmissionEnvelope:
    """One-time admission endpoint used by the configured media service."""

    try:
        admission = service.consume(body.session_id, body.access_token)
    except RealtimeSessionError as exc:
        raise AppError(
            status_code=401,
            code="realtime_session_admission_denied",
            category="authentication",
            message="Realtime media admission denied",
        ) from exc
    return RealtimeMediaAdmissionEnvelope(
        data=_admission_response(admission),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/realtime-sessions/{session_id}/transcriptions",
    response_model=RealtimeTranscriptEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
async def transcribe_realtime_segment(
    session_id: str,
    request: Request,
    service: Annotated[RealtimeTranscriptService, Depends(get_realtime_transcript_service)],
    authorization: Annotated[str, Header(alias="Authorization", min_length=40)],
    segment_id: Annotated[str, Header(alias="Segment-Id", min_length=1, max_length=128)],
    sequence: Annotated[int, Header(alias="Segment-Sequence", ge=1)],
    start_ms: Annotated[int, Header(alias="Segment-Start-Ms", ge=0)],
    end_ms: Annotated[int, Header(alias="Segment-End-Ms", ge=1)],
    content_type: Annotated[str, Header(alias="Content-Type")],
) -> RealtimeTranscriptEnvelope:
    if not authorization.startswith("Bearer ") or content_type.split(";", 1)[0] != "audio/wav":
        raise AppError(
            status_code=401,
            code="realtime_media_capability_invalid",
            category="authentication",
            message="Realtime media capability denied",
        )
    try:
        result = await service.transcribe(
            authorization.removeprefix("Bearer ").strip(),
            session_id,
            segment_id=segment_id,
            sequence=sequence,
            start_ms=start_ms,
            end_ms=end_ms,
            audio_wav=await request.body(),
            request_id=request.state.request_id,
        )
    except RealtimeSessionError as exc:
        raise AppError(
            status_code=401,
            code="realtime_media_capability_invalid",
            category="authentication",
            message="Realtime media capability denied",
        ) from exc
    except ModelGatewayError as exc:
        raise AppError(
            status_code=503,
            code=exc.reason,
            category="dependency",
            message="Realtime transcription unavailable",
            retryable=True,
        ) from exc
    return RealtimeTranscriptEnvelope(
        data=_transcript_response(result),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/realtime-sessions/{session_id}/speech-syntheses/{synthesis_id}/audio",
    responses={**STANDARD_ERROR_RESPONSES, 200: {"content": {"audio/wav": {}}}},
)
async def realtime_speech_audio(
    session_id: str,
    synthesis_id: str,
    database: Annotated[Database, Depends(get_database)],
    gateway: Annotated[SpeechSynthesisGateway, Depends(get_speech_gateway)],
    authorization: Annotated[str, Header(alias="Authorization", min_length=40)],
) -> Response:
    """Serve pre-authorized TTS only to its live, diagnosis-bound media session."""

    if not authorization.startswith("Bearer "):
        raise _media_capability_denied()
    try:
        capability = await run_in_threadpool(
            authorize_media_capability,
            database,
            authorization.removeprefix("Bearer ").strip(),
            session_id,
        )
        result = await run_in_threadpool(
            gateway.describe, capability.tenant_context, synthesis_id
        )
        if (
            capability.session.diagnosis_run_id is None
            or result.diagnosis_run_id != capability.session.diagnosis_run_id
        ):
            raise RealtimeSessionError("realtime_speech_diagnosis_mismatch")
        content = await gateway.read_audio(capability.tenant_context, result)
        await run_in_threadpool(
            authorize_media_capability,
            database,
            authorization.removeprefix("Bearer ").strip(),
            session_id,
        )
    except (RealtimeSessionError, ModelGatewayError) as exc:
        raise _media_capability_denied() from exc
    return Response(
        content=content,
        media_type=result.media_type,
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": 'inline; filename="realtime-diagnosis-speech.wav"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/realtime-transcripts/{segment_id}/confirmation",
    response_model=RealtimeTranscriptEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def confirm_realtime_transcript(
    segment_id: str,
    body: ConfirmRealtimeTranscriptRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[RealtimeTranscriptService, Depends(get_realtime_transcript_service)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> RealtimeTranscriptEnvelope:
    try:
        result = service.confirm(
            identity,
            segment_id,
            decision=body.decision,
            corrected_text=body.corrected_text,
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except RealtimeSessionError as exc:
        raise _session_error(exc) from exc
    except ValueError as exc:
        raise AppError(422, "realtime_transcript_invalid", "validation", str(exc)) from exc
    return RealtimeTranscriptEnvelope(
        data=_transcript_response(result),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/realtime-sessions/{session_id}/reconnect",
    response_model=RealtimeSessionGrantEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def reconnect_realtime_session(
    session_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[RealtimeSessionService, Depends(get_realtime_session_service)],
) -> RealtimeSessionGrantEnvelope:
    try:
        grant = service.reconnect(
            identity,
            session_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except RealtimeSessionError as exc:
        raise _session_error(exc) from exc
    return RealtimeSessionGrantEnvelope(
        data=_grant_response(grant),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/realtime-sessions/{session_id}/heartbeat",
    response_model=RealtimeSessionEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def heartbeat_realtime_session(
    session_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[RealtimeSessionService, Depends(get_realtime_session_service)],
) -> RealtimeSessionEnvelope:
    try:
        session = service.heartbeat(
            identity,
            session_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except RealtimeSessionError as exc:
        raise _session_error(exc) from exc
    return RealtimeSessionEnvelope(
        data=_session_response(session),
        meta={"request_id": request.state.request_id},
    )


@router.delete(
    "/realtime-sessions/{session_id}",
    response_model=RealtimeSessionEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def end_realtime_session(
    session_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[RealtimeSessionService, Depends(get_realtime_session_service)],
) -> RealtimeSessionEnvelope:
    try:
        session = service.end(
            identity,
            session_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    return RealtimeSessionEnvelope(
        data=_session_response(session),
        meta={"request_id": request.state.request_id},
    )


def _grant_response(grant: RealtimeSessionGrant) -> RealtimeSessionGrantResponse:
    session = grant.session
    return RealtimeSessionGrantResponse(
        **_session_response(session).model_dump(),
        access_token=grant.access_token,
        signaling_url=grant.signaling_url,
        ice_servers=[_ice_response(item) for item in grant.ice_servers],
        heartbeat_interval_seconds=grant.heartbeat_interval_seconds,
    )


def _admission_response(
    admission: RealtimeMediaAdmission,
) -> RealtimeMediaAdmissionResponse:
    return RealtimeMediaAdmissionResponse(
        session=_session_response(admission.session),
        media_access_token=admission.media_access_token,
        expert_collaboration_id=admission.session.expert_collaboration_id,
        participant_subject_id=admission.session.participant_subject_id,
    )


def _transcript_response(
    transcript: RealtimeTranscriptSegment,
) -> RealtimeTranscriptResponse:
    return RealtimeTranscriptResponse(
        segment_id=transcript.segment_id,
        session_id=transcript.session_id,
        sequence=transcript.sequence,
        start_ms=transcript.start_ms,
        end_ms=transcript.end_ms,
        text=transcript.text,
        confirmed_text=transcript.confirmed_text,
        confidence=transcript.confidence,
        language=transcript.language,
        entities=[TranscriptEntityResponse.model_validate(item) for item in transcript.entities],
        resolved_release_id=transcript.resolved_release_id,
        manifest_hash=transcript.manifest_hash,
        status=transcript.status,
        agent_event_id=transcript.agent_event_id,
        agent_event_sequence=transcript.agent_event_sequence,
        security_policy_version=transcript.security_policy_version,
        security_findings=list(transcript.security_findings),
        version=transcript.version,
    )


def _session_response(session: RealtimeSession) -> RealtimeSessionResponse:
    return RealtimeSessionResponse(
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
    )


def _ice_response(server: IceServer) -> IceServerResponse:
    return IceServerResponse(
        urls=list(server.urls),
        username=server.username,
        credential=server.credential,
    )


def _not_visible() -> AppError:
    return AppError(
        status_code=404,
        code="realtime_session_not_found",
        category="not_found",
        message="Realtime session not found",
    )


def _session_error(exc: RealtimeSessionError) -> AppError:
    if exc.reason == "identity_expiry_too_close":
        return AppError(
            status_code=401,
            code=exc.reason,
            category="authentication",
            message="Sign in again before starting realtime media",
        )
    if "expired" in exc.reason:
        return AppError(
            status_code=410,
            code=exc.reason,
            category="state",
            message="Realtime media session expired",
        )
    return AppError(
        status_code=409,
        code=exc.reason,
        category="state",
        message="Realtime media session state conflict",
    )


def _media_capability_denied() -> AppError:
    return AppError(
        status_code=401,
        code="realtime_media_capability_invalid",
        category="authentication",
        message="Realtime media capability denied",
    )


def _parse_version(value: str) -> int:
    normalized = value.strip().removeprefix('W/').strip('"')
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
