"""M2 asynchronous recognition, evidence query, and human confirmation API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_object_store,
    get_recognition_dispatcher,
)
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.api.routes.model_gateway import ModelExecutionResponse
from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.application.incidents import (
    IdempotencyConflict,
    RecognitionRun,
    RecognitionService,
)
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.domain.incidents import DraftStateConflict, IncidentSubmissionBlocked
from industrial_ops_agent.media.service import TenantObjectStore
from industrial_ops_agent.model_gateway.evidence import ModelExecutionEvidence
from industrial_ops_agent.multimodal.models import (
    BoundingBox,
    EvidenceBundle,
    EvidenceConfirmationRequired,
    EvidenceVersionConflict,
    FindingDisposition,
    OcrDisposition,
    QrDisposition,
    TranscriptDisposition,
)
from industrial_ops_agent.orchestration.activities import (
    RecognitionActivityInput,
    RecognitionDispatcher,
    RecognitionDispatchUnavailable,
)
from industrial_ops_agent.persistence.database import Database

router = APIRouter(prefix="/incident-drafts", tags=["recognition"])


class StartRecognitionRequest(BaseModel):
    media_id: str = Field(min_length=1, max_length=128)
    processor_profile: Literal["local-core-v1", "local-video-v1", "video-temporal-v1"] = (
        "local-core-v1"
    )


class RecognitionRunResponse(BaseModel):
    recognition_run_id: str
    draft_id: str
    media_id: str
    workflow_id: str
    status: str
    processor_profile: str
    evidence_bundle_id: str | None
    failure_reason: str | None
    version: int
    created_at: datetime
    updated_at: datetime


class RecognitionRunEnvelope(BaseModel):
    data: RecognitionRunResponse
    meta: dict[str, str | bool]


class BoundingBoxResponse(BaseModel):
    x: float
    y: float
    width: float
    height: float


class OcrBlockResponse(BaseModel):
    block_id: str
    page_number: int
    text: str
    bbox: BoundingBoxResponse
    confidence: float
    source_frame_id: str | None
    disposition: str | None
    corrected_text: str | None


class AsrEntityCandidateResponse(BaseModel):
    entity_id: str
    source_segment_id: str
    entity_type: str
    value: str
    normalized_value: str
    confidence: float
    start_offset: int
    end_offset: int
    requires_confirmation: bool


class AsrSegmentResponse(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    text: str
    language: str
    confidence: float
    model_release_id: str
    source_audio_track_id: str | None
    hotword_profile_id: str | None
    entity_candidates: list[AsrEntityCandidateResponse]
    disposition: str | None
    corrected_text: str | None


class ExtractedEntityResponse(BaseModel):
    entity_id: str
    entity_type: str
    original_value: str
    normalized_value: str
    confidence: float
    validation_status: str
    source_block_id: str
    validation_reason: str | None
    corrected_value: str | None


class VisualFindingResponse(BaseModel):
    finding_id: str
    label: str
    bbox: BoundingBoxResponse
    confidence: float
    evidence_level: str
    model_release_id: str
    description: str
    source_frame_id: str | None
    disposition: str | None


class VideoKeyframeResponse(BaseModel):
    frame_id: str
    timestamp_ms: int
    image_sha256: str
    sampling_reason: str
    image_url: str


class VideoEventResponse(BaseModel):
    event_id: str
    event_type: str
    start_ms: int
    end_ms: int
    keyframe_ids: list[str]
    finding_ids: list[str]
    model_release_id: str | None
    confidence: float
    description: str
    disposition: str | None


class EvidenceSecurityFindingResponse(BaseModel):
    source_type: str
    source_id: str
    policy_version: str
    pattern_id: str
    category: str
    severity: str
    content_hash: str


class QrCodeResponse(BaseModel):
    candidate_id: str
    text: str
    payload_kind: str
    bbox: BoundingBoxResponse
    source_frame_id: str | None
    security_findings: list[EvidenceSecurityFindingResponse]
    disposition: str | None
    corrected_text: str | None


class EvidenceResponse(BaseModel):
    bundle_id: str
    draft_id: str
    asset_id: str
    media_id: str
    source_sha256: str
    source_type: str
    ocr_blocks: list[OcrBlockResponse]
    asr_segments: list[AsrSegmentResponse]
    extracted_entities: list[ExtractedEntityResponse]
    visual_findings: list[VisualFindingResponse]
    video_keyframes: list[VideoKeyframeResponse]
    video_events: list[VideoEventResponse]
    qr_codes: list[QrCodeResponse] = Field(default_factory=list)
    processor_versions: dict[str, str]
    model_executions: list[ModelExecutionResponse] = Field(default_factory=list)
    security_policy_version: str | None = None
    security_findings: list[EvidenceSecurityFindingResponse] = Field(default_factory=list)
    automation_eligible: bool | None = None
    automation_blockers: list[str] = Field(default_factory=list)
    status: str
    version: int
    created_at: datetime
    updated_at: datetime


class EvidenceEnvelope(BaseModel):
    data: EvidenceResponse
    meta: dict[str, str | int]


class TranscriptDecisionRequest(BaseModel):
    disposition: TranscriptDisposition
    corrected_text: str | None = Field(default=None, max_length=4_000)


class OcrDecisionRequest(BaseModel):
    disposition: OcrDisposition
    corrected_text: str | None = Field(default=None, max_length=4_000)


class QrCodeDecisionRequest(BaseModel):
    disposition: QrDisposition
    corrected_text: str | None = Field(default=None, max_length=4_096)


class RecognitionConfirmationRequest(BaseModel):
    bundle_id: str = Field(min_length=1, max_length=128)
    corrections: dict[str, str] = Field(default_factory=dict)
    finding_dispositions: dict[str, FindingDisposition] = Field(default_factory=dict)
    transcript_decisions: dict[str, TranscriptDecisionRequest] = Field(default_factory=dict)
    ocr_block_decisions: dict[str, OcrDecisionRequest] = Field(default_factory=dict)
    video_event_dispositions: dict[str, FindingDisposition] = Field(default_factory=dict)
    qr_code_decisions: dict[str, QrCodeDecisionRequest] = Field(default_factory=dict)


@router.post(
    "/{draft_id}/recognition-runs",
    response_model=RecognitionRunEnvelope,
    status_code=202,
    responses=STANDARD_ERROR_RESPONSES,
)
async def start_recognition(
    draft_id: str,
    payload: StartRecognitionRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[RecognitionDispatcher, Depends(get_recognition_dispatcher)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> RecognitionRunEnvelope:
    service = RecognitionService(database, authorizer)
    try:
        result = service.enqueue(
            identity,
            draft_id,
            media_id=payload.media_id,
            processor_profile=payload.processor_profile,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
        if result.run.status == "QUEUED":
            await dispatcher.dispatch(
                RecognitionActivityInput(
                    recognition_run_id=result.run.recognition_run_id,
                    identity=identity,
                    request_id=request.state.request_id,
                    workflow_id=result.run.workflow_id,
                )
            )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except IdempotencyConflict as exc:
        raise _conflict("idempotency_conflict", "Idempotency key was reused") from exc
    except DraftStateConflict as exc:
        raise _conflict("draft_state_conflict", str(exc)) from exc
    except IncidentSubmissionBlocked as exc:
        raise _blocked(exc.reason_code) from exc
    except RecognitionDispatchUnavailable as exc:
        raise AppError(
            status_code=503,
            code="recognition_workflow_unavailable",
            category="dependency",
            message="Recognition workflow unavailable",
            retryable=True,
        ) from exc
    return _run_envelope(result.run, result.created, request.state.request_id)


@router.get(
    "/{draft_id}/recognition-runs/{recognition_run_id}",
    response_model=RecognitionRunEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_recognition_run(
    draft_id: str,
    recognition_run_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> RecognitionRunEnvelope:
    try:
        run = RecognitionService(database, authorizer).get_run(
            identity,
            draft_id,
            recognition_run_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    return _run_envelope(run, False, request.state.request_id)


@router.get(
    "/{draft_id}/evidence",
    response_model=EvidenceEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_evidence(
    draft_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> EvidenceEnvelope:
    try:
        bundle = RecognitionService(database, authorizer).get_evidence(
            identity,
            draft_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    return _evidence_envelope(bundle, request.state.request_id)


@router.get(
    "/{draft_id}/evidence/{bundle_id}/video-keyframes/{frame_id}/image",
    responses={**STANDARD_ERROR_RESPONSES, 200: {"content": {"image/jpeg": {}}}},
)
async def get_video_keyframe_image(
    draft_id: str,
    bundle_id: str,
    frame_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
) -> Response:
    try:
        content = await RecognitionService(database, authorizer).read_video_keyframe(
            identity,
            draft_id,
            bundle_id,
            frame_id,
            object_store=object_store,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": 'inline; filename="video-keyframe.jpg"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/{draft_id}/recognition-confirmations",
    response_model=EvidenceEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def confirm_recognition(
    draft_id: str,
    payload: RecognitionConfirmationRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> EvidenceEnvelope:
    try:
        bundle = RecognitionService(database, authorizer).confirm(
            identity,
            draft_id,
            bundle_id=payload.bundle_id,
            corrections=payload.corrections,
            finding_dispositions={
                finding_id: disposition
                for finding_id, disposition in payload.finding_dispositions.items()
            },
            transcript_decisions={
                segment_id: (decision.disposition, decision.corrected_text)
                for segment_id, decision in payload.transcript_decisions.items()
            },
            ocr_block_decisions={
                block_id: (decision.disposition, decision.corrected_text)
                for block_id, decision in payload.ocr_block_decisions.items()
            }
            if payload.ocr_block_decisions
            else None,
            video_event_dispositions=payload.video_event_dispositions or None,
            qr_code_decisions={
                candidate_id: (decision.disposition, decision.corrected_text)
                for candidate_id, decision in payload.qr_code_decisions.items()
            },
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except DraftStateConflict as exc:
        raise _conflict("draft_state_conflict", str(exc)) from exc
    except EvidenceVersionConflict as exc:
        raise AppError(
            status_code=409,
            code="evidence_version_conflict",
            category="conflict",
            message="Evidence version conflict",
            details={"current_version": exc.current_version},
        ) from exc
    except EvidenceConfirmationRequired as exc:
        raise _blocked(exc.reason_code) from exc
    return _evidence_envelope(bundle, request.state.request_id)


def _run_envelope(run: RecognitionRun, created: bool, request_id: str) -> RecognitionRunEnvelope:
    return RecognitionRunEnvelope(
        data=RecognitionRunResponse(
            recognition_run_id=run.recognition_run_id,
            draft_id=run.draft_id,
            media_id=run.media_id,
            workflow_id=run.workflow_id,
            status=run.status,
            processor_profile=run.processor_profile,
            evidence_bundle_id=run.evidence_bundle_id,
            failure_reason=run.failure_reason,
            version=run.version,
            created_at=run.created_at,
            updated_at=run.updated_at,
        ),
        meta={"request_id": request_id, "created": created},
    )


def _evidence_envelope(
    bundle: EvidenceBundle,
    request_id: str,
    *,
    video_keyframe_base_path: str | None = None,
) -> EvidenceEnvelope:
    corrections = {item.entity_id: item.corrected_value for item in bundle.human_corrections}
    ocr_reviews = {item.block_id: item for item in bundle.ocr_reviews}
    reviews = {item.finding_id: item.disposition.value for item in bundle.finding_reviews}
    transcript_reviews = {item.segment_id: item for item in bundle.transcript_reviews}
    video_event_reviews = {
        item.event_id: item.disposition.value for item in bundle.video_event_reviews
    }
    qr_reviews = {item.candidate_id: item for item in bundle.qr_reviews}
    keyframe_base_path = video_keyframe_base_path or (
        f"/api/v1/incident-drafts/{bundle.draft_id}/evidence/{bundle.bundle_id}/video-keyframes"
    )
    model_execution = ModelExecutionEvidence.from_processor_versions(
        bundle.processor_versions,
        prefix="vlm",
        component="vlm",
    )
    return EvidenceEnvelope(
        data=EvidenceResponse(
            bundle_id=bundle.bundle_id,
            draft_id=bundle.draft_id,
            asset_id=bundle.asset_id,
            media_id=bundle.media_id,
            source_sha256=bundle.source_sha256,
            source_type=bundle.source_type,
            ocr_blocks=[
                OcrBlockResponse(
                    block_id=item.block_id,
                    page_number=item.page_number,
                    text=item.text,
                    bbox=BoundingBoxResponse(**_box(item.bbox)),
                    confidence=item.confidence,
                    source_frame_id=item.source_frame_id,
                    disposition=(
                        ocr_reviews[item.block_id].disposition.value
                        if item.block_id in ocr_reviews
                        else None
                    ),
                    corrected_text=(
                        ocr_reviews[item.block_id].corrected_text
                        if item.block_id in ocr_reviews
                        else None
                    ),
                )
                for item in bundle.ocr_blocks
            ],
            asr_segments=[
                AsrSegmentResponse(
                    segment_id=item.segment_id,
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                    text=item.text,
                    language=item.language,
                    confidence=item.confidence,
                    model_release_id=item.model_release_id,
                    source_audio_track_id=item.source_audio_track_id,
                    hotword_profile_id=item.hotword_profile_id,
                    entity_candidates=[
                        AsrEntityCandidateResponse(
                            entity_id=candidate.entity_id,
                            source_segment_id=candidate.source_segment_id,
                            entity_type=candidate.entity_type,
                            value=candidate.value,
                            normalized_value=candidate.normalized_value,
                            confidence=candidate.confidence,
                            start_offset=candidate.start_offset,
                            end_offset=candidate.end_offset,
                            requires_confirmation=candidate.requires_confirmation,
                        )
                        for candidate in item.entity_candidates
                    ],
                    disposition=(
                        transcript_reviews[item.segment_id].disposition.value
                        if item.segment_id in transcript_reviews
                        else None
                    ),
                    corrected_text=(
                        transcript_reviews[item.segment_id].corrected_text
                        if item.segment_id in transcript_reviews
                        else None
                    ),
                )
                for item in bundle.asr_segments
            ],
            extracted_entities=[
                ExtractedEntityResponse(
                    entity_id=item.entity_id,
                    entity_type=item.entity_type,
                    original_value=item.value,
                    normalized_value=item.normalized_value,
                    confidence=item.confidence,
                    validation_status=item.validation_status.value,
                    source_block_id=item.source_block_id,
                    validation_reason=item.validation_reason,
                    corrected_value=corrections.get(item.entity_id),
                )
                for item in bundle.extracted_entities
            ],
            visual_findings=[
                VisualFindingResponse(
                    finding_id=item.finding_id,
                    label=item.label,
                    bbox=BoundingBoxResponse(**_box(item.bbox)),
                    confidence=item.confidence,
                    evidence_level=item.evidence_level,
                    model_release_id=item.model_release_id,
                    description=item.description,
                    source_frame_id=item.source_frame_id,
                    disposition=reviews.get(item.finding_id),
                )
                for item in bundle.visual_findings
            ],
            video_keyframes=[
                VideoKeyframeResponse(
                    frame_id=item.frame_id,
                    timestamp_ms=item.timestamp_ms,
                    image_sha256=item.image_sha256,
                    sampling_reason=item.sampling_reason,
                    image_url=f"{keyframe_base_path}/{item.frame_id}/image",
                )
                for item in bundle.video_keyframes
            ],
            video_events=[
                VideoEventResponse(
                    event_id=item.event_id,
                    event_type=item.event_type,
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                    keyframe_ids=list(item.keyframe_ids),
                    finding_ids=list(item.finding_ids),
                    model_release_id=item.model_release_id,
                    confidence=item.confidence,
                    description=item.description,
                    disposition=video_event_reviews.get(item.event_id),
                )
                for item in bundle.video_events
            ],
            qr_codes=[
                QrCodeResponse(
                    candidate_id=item.candidate_id,
                    text=item.text,
                    payload_kind=item.payload_kind.value,
                    bbox=BoundingBoxResponse(**_box(item.bbox)),
                    source_frame_id=item.source_frame_id,
                    security_findings=[
                        EvidenceSecurityFindingResponse(**finding.as_dict())
                        for finding in item.security_findings
                    ],
                    disposition=(
                        qr_reviews[item.candidate_id].disposition.value
                        if item.candidate_id in qr_reviews
                        else None
                    ),
                    corrected_text=(
                        qr_reviews[item.candidate_id].corrected_text
                        if item.candidate_id in qr_reviews
                        else None
                    ),
                )
                for item in bundle.qr_codes
            ],
            processor_versions=bundle.processor_versions,
            model_executions=(
                [ModelExecutionResponse.model_validate(model_execution.as_dict())]
                if model_execution is not None
                else []
            ),
            security_policy_version=bundle.security_policy_version,
            security_findings=[
                EvidenceSecurityFindingResponse(**item.as_dict())
                for item in bundle.security_findings
            ],
            automation_eligible=bundle.automation_eligible,
            automation_blockers=list(bundle.automation_blockers),
            status=bundle.status.value,
            version=bundle.version,
            created_at=bundle.created_at,
            updated_at=bundle.updated_at,
        ),
        meta={"request_id": request_id, "version": bundle.version},
    )


def _box(value: BoundingBox) -> dict[str, float]:
    return {
        "x": value.x,
        "y": value.y,
        "width": value.width,
        "height": value.height,
    }


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
        message="If-Match must contain a positive evidence version",
    )


def _not_visible() -> AppError:
    return AppError(
        status_code=404,
        code="resource_not_found_or_not_visible",
        category="not_found",
        message="Resource not found or not visible",
    )


def _conflict(code: str, message: str) -> AppError:
    return AppError(status_code=409, code=code, category="conflict", message=message)


def _blocked(reason_code: str) -> AppError:
    return AppError(
        status_code=422,
        code=reason_code,
        category="validation",
        message="Recognition or submission preconditions are incomplete",
    )
