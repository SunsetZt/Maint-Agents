"""TTS endpoints bound to completed diagnosis reports and safety confirmation."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_speech_gateway,
)
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.application.speech import (
    DiagnosisSpeechService,
    SpeechSynthesisDenied,
)
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.model_gateway.service import ModelGatewayError
from industrial_ops_agent.multimodal.speech import (
    SpeechSynthesisGateway,
    SpeechSynthesisResult,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.workorders.service import WorkOrderConflict, WorkOrderNotVisible
from industrial_ops_agent.workorders.voice_guidance import FieldVoiceGuidanceService

router = APIRouter(tags=["speech"])


class CreateDiagnosisSpeechRequest(BaseModel):
    safety_acknowledged: bool
    voice: Literal["default"] = "default"


class CreateFieldDiagnosisSpeechRequest(CreateDiagnosisSpeechRequest):
    diagnosis_run_id: str
    work_order_version: int


class SpeechSynthesisResponse(BaseModel):
    synthesis_id: str
    diagnosis_run_id: str
    resolved_release_id: str
    manifest_hash: str
    media_type: str
    size_bytes: int
    audio_sha256: str
    voice: str
    safety_warning_digest: str
    audio_url: str
    replayed: bool


class SpeechSynthesisEnvelope(BaseModel):
    data: SpeechSynthesisResponse
    meta: dict[str, str]


class FieldSpeechSynthesisResponse(BaseModel):
    synthesis_id: str
    diagnosis_run_id: str
    resolved_release_id: str
    manifest_hash: str
    media_type: str
    size_bytes: int
    audio_sha256: str
    voice: str
    safety_warning_digest: str
    replayed: bool


class FieldSpeechSynthesisEnvelope(BaseModel):
    data: FieldSpeechSynthesisResponse
    meta: dict[str, str]


@router.post(
    "/diagnosis-runs/{diagnosis_run_id}/speech",
    response_model=SpeechSynthesisEnvelope,
    status_code=201,
    responses={
        **STANDARD_ERROR_RESPONSES,
        200: {"model": SpeechSynthesisEnvelope, "description": "Idempotent replay"},
    },
)
async def synthesize_diagnosis_speech(
    diagnosis_run_id: str,
    body: CreateDiagnosisSpeechRequest,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    gateway: Annotated[SpeechSynthesisGateway, Depends(get_speech_gateway)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> SpeechSynthesisEnvelope:
    service = _service(request, database, authorizer, gateway)
    try:
        result = await service.synthesize(
            identity,
            diagnosis_run_id,
            safety_acknowledged=body.safety_acknowledged,
            voice=body.voice,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except SpeechSynthesisDenied as exc:
        raise AppError(
            status_code=422,
            code=exc.reason,
            category="validation",
            message="Diagnosis speech safety gate rejected the request",
        ) from exc
    except ModelGatewayError as exc:
        raise _gateway_error(exc) from exc
    if result.replayed:
        response.status_code = 200
    return SpeechSynthesisEnvelope(
        data=_response(result),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/field/work-orders/{work_order_id}/voice-guidance/speech",
    response_model=FieldSpeechSynthesisEnvelope,
    status_code=201,
    responses={
        **STANDARD_ERROR_RESPONSES,
        200: {"model": FieldSpeechSynthesisEnvelope, "description": "Idempotent replay"},
    },
    tags=["field-service"],
)
async def synthesize_field_voice_guidance(
    work_order_id: str,
    body: CreateFieldDiagnosisSpeechRequest,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    gateway: Annotated[SpeechSynthesisGateway, Depends(get_speech_gateway)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> FieldSpeechSynthesisEnvelope:
    try:
        await asyncio.to_thread(
            FieldVoiceGuidanceService(database, authorizer).require_exact,
            identity,
            work_order_id,
            expected_work_order_version=body.work_order_version,
            diagnosis_run_id=body.diagnosis_run_id,
            require_completed=True,
            request_id=request.state.request_id,
        )
        result = await _service(request, database, authorizer, gateway).synthesize(
            identity,
            body.diagnosis_run_id,
            safety_acknowledged=body.safety_acknowledged,
            voice=body.voice,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (WorkOrderNotVisible, ResourceNotVisible) as exc:
        raise _not_visible() from exc
    except WorkOrderConflict as exc:
        raise AppError(
            status_code=409,
            code=exc.reason,
            category="conflict",
            message="Field voice guidance binding changed",
            details={"current_version": exc.current_version},
        ) from exc
    except SpeechSynthesisDenied as exc:
        raise AppError(
            status_code=422,
            code=exc.reason,
            category="validation",
            message="Diagnosis speech safety gate rejected the request",
        ) from exc
    except ModelGatewayError as exc:
        raise _gateway_error(exc) from exc
    if result.replayed:
        response.status_code = 200
    return FieldSpeechSynthesisEnvelope(
        data=_field_response(result),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/speech-syntheses/{synthesis_id}/audio",
    responses={**STANDARD_ERROR_RESPONSES, 200: {"content": {"audio/wav": {}}}},
)
async def diagnosis_speech_audio(
    synthesis_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    gateway: Annotated[SpeechSynthesisGateway, Depends(get_speech_gateway)],
) -> Response:
    try:
        result, content = await _service(request, database, authorizer, gateway).audio(
            identity,
            synthesis_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except (SpeechSynthesisDenied, ModelGatewayError) as exc:
        raise _not_visible() from exc
    return Response(
        content=content,
        media_type=result.media_type,
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": 'inline; filename="diagnosis-speech.wav"',
            "X-Content-Type-Options": "nosniff",
        },
    )


def _service(
    request: Request,
    database: Database,
    authorizer: Authorizer,
    gateway: SpeechSynthesisGateway,
) -> DiagnosisSpeechService:
    return DiagnosisSpeechService(
        database,
        authorizer,
        gateway,
        model_alias=request.app.state.settings.tts_model_alias,
        timeout_seconds=request.app.state.settings.tts_timeout_seconds,
    )


def _response(result: SpeechSynthesisResult) -> SpeechSynthesisResponse:
    return SpeechSynthesisResponse(
        synthesis_id=result.synthesis_id,
        diagnosis_run_id=result.diagnosis_run_id,
        resolved_release_id=result.resolved_release_id,
        manifest_hash=result.manifest_hash,
        media_type=result.media_type,
        size_bytes=result.size_bytes,
        audio_sha256=result.audio_sha256,
        voice=result.voice,
        safety_warning_digest=result.safety_warning_digest,
        audio_url=f"/api/v1/speech-syntheses/{result.synthesis_id}/audio",
        replayed=result.replayed,
    )


def _field_response(result: SpeechSynthesisResult) -> FieldSpeechSynthesisResponse:
    """FIELD speech is playable only through a live realtime media capability."""

    return FieldSpeechSynthesisResponse(
        synthesis_id=result.synthesis_id,
        diagnosis_run_id=result.diagnosis_run_id,
        resolved_release_id=result.resolved_release_id,
        manifest_hash=result.manifest_hash,
        media_type=result.media_type,
        size_bytes=result.size_bytes,
        audio_sha256=result.audio_sha256,
        voice=result.voice,
        safety_warning_digest=result.safety_warning_digest,
        replayed=result.replayed,
    )


def _gateway_error(exc: ModelGatewayError) -> AppError:
    if exc.reason in {
        "model_request_rate_exceeded",
        "model_daily_token_quota_exceeded",
    }:
        return AppError(
            status_code=429,
            code=exc.reason,
            category="rate_limit",
            message="Speech synthesis quota exceeded",
            retryable=True,
        )
    if exc.reason in {
        "production_model_unavailable",
        "production_model_route_inconsistent",
        "tts_endpoint_unavailable",
        "tts_endpoint_failed",
        "tts_unexpected_failure",
    }:
        return AppError(
            status_code=503,
            code=exc.reason,
            category="dependency",
            message="Speech synthesis service unavailable",
            retryable=True,
        )
    return AppError(
        status_code=409,
        code=exc.reason,
        category="conflict",
        message="Speech synthesis request could not be completed",
    )


def _not_visible() -> AppError:
    return AppError(
        status_code=404,
        code="resource_not_found_or_not_visible",
        category="not_found",
        message="Resource not found or not visible",
    )
