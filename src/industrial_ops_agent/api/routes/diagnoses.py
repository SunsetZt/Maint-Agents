"""Explicit diagnosis start/query and citation re-authorization endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from industrial_ops_agent.agent.models import DiagnosisRun
from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_diagnosis_dispatcher,
    get_identity,
    get_model_resolver,
    get_object_store,
)
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.api.routes.model_gateway import ModelExecutionResponse
from industrial_ops_agent.application.agent_runs import (
    AgentRunCancellationDispatchFailed,
    AgentRunControlService,
    AgentRunControlView,
    AgentRunPreconditionFailed,
    AgentRunResumeDispatchFailed,
    AgentRunVersionConflict,
)
from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.application.diagnoses import (
    DiagnosisDispatcher,
    DiagnosisDispatchFailed,
    DiagnosisPreconditionFailed,
    DiagnosisService,
    DiagnosisVersionConflict,
)
from industrial_ops_agent.application.expert_diagnoses import (
    ExpertDiagnosisError,
    ExpertDiagnosisInterventionView,
    ExpertDiagnosisRevisionInput,
    ExpertDiagnosisService,
)
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.knowledge.models import CitationView
from industrial_ops_agent.knowledge.provenance import (
    CitationProvenanceService,
    CitationProvenanceView,
    CitationSourceNotAvailable,
)
from industrial_ops_agent.knowledge.service import KnowledgeNotVisible
from industrial_ops_agent.maintenance_planning.review_isolation import record_disclosure
from industrial_ops_agent.media.service import TenantObjectStore
from industrial_ops_agent.model_gateway.service import ProductionModelResolver
from industrial_ops_agent.persistence.database import Database

router = APIRouter(tags=["diagnoses"])


class ExpertDiagnosisRevisionResponse(BaseModel):
    revision_id: str
    ordinal: int
    outcome: str
    report: dict[str, Any]
    report_digest: str
    source_ai_report_digest: str
    revision_reason: str
    authored_by_subject_id: str
    created_at: datetime


class ExpertDiagnosisInterventionResponse(BaseModel):
    intervention_id: str
    diagnosis_run_id: str
    status: str
    source_run_version: int
    ai_report_snapshot: dict[str, Any] | None
    ai_report_digest: str | None
    takeover_reason: str
    requested_by_subject_id: str
    completed_at: datetime | None
    version: int
    revisions: list[ExpertDiagnosisRevisionResponse]
    created_at: datetime
    updated_at: datetime


class DiagnosisRunResponse(BaseModel):
    diagnosis_run_id: str
    incident_id: str
    evidence_bundle_id: str
    agent_run_id: str
    agent_run_version: int
    workflow_id: str
    stream_url: str
    status: str
    manifest: dict[str, Any]
    model_execution: ModelExecutionResponse | None
    report: dict[str, Any] | None
    ai_report: dict[str, Any] | None
    expert_interventions: list[ExpertDiagnosisInterventionResponse]
    stop_reason: str | None
    version: int
    created_at: datetime
    updated_at: datetime
    legal_actions: list[str]


class DiagnosisEnvelope(BaseModel):
    data: DiagnosisRunResponse
    meta: dict[str, str | bool | int]


class ReanalyzeDiagnosisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript_segment_id: str | None = Field(default=None, min_length=1, max_length=128)
    work_order_id: str | None = Field(default=None, min_length=1, max_length=128)
    work_order_version: int | None = Field(default=None, ge=1)
    field_entry_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
    )

    @model_validator(mode="after")
    def accept_exactly_one_reanalysis_source(self) -> ReanalyzeDiagnosisRequest:
        transcript_mode = self.transcript_segment_id is not None
        field_values = (
            self.work_order_id,
            self.work_order_version,
            self.field_entry_ids,
        )
        field_mode = all(value is not None for value in field_values)
        if (
            transcript_mode == field_mode
            or any(value is not None for value in field_values) != field_mode
        ):
            raise ValueError("reanalysis_source_must_be_exactly_one_mode")
        if self.field_entry_ids is not None and len(set(self.field_entry_ids)) != len(
            self.field_entry_ids
        ):
            raise ValueError("field_entry_ids_must_be_unique")
        return self


class CancelAgentRunRequest(BaseModel):
    reason: str = Field(min_length=8, max_length=1024)


class ResumeAgentRunRequest(BaseModel):
    clarification: str = Field(min_length=2, max_length=4000)


class AgentRunControlResponse(BaseModel):
    agent_run_id: str
    diagnosis_run_id: str
    status: str
    version: int
    last_sequence: int
    stream_url: str


class AgentRunControlEnvelope(BaseModel):
    data: AgentRunControlResponse
    meta: dict[str, str | bool | int]


class ExpertDiagnosisTakeoverRequest(BaseModel):
    reason: str = Field(min_length=8, max_length=1024)


class ExpertDiagnosisRevisionRequest(BaseModel):
    outcome: Literal["COMPLETED", "ESCALATED"]
    conclusion: str | None = Field(default=None, max_length=4000)
    citation_ids: list[str] = Field(default_factory=list, max_length=50)
    contradictions: list[str] = Field(default_factory=list, max_length=50)
    missing_information: list[str] = Field(default_factory=list, max_length=50)
    next_checks: list[str] = Field(default_factory=list, max_length=50)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=8, max_length=1024)


class CitationIngestionAttemptResponse(BaseModel):
    attempt_id: str
    attempt_number: int
    workflow_id: str
    status: str
    failure_reason: str | None
    parser_version: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class CitationSourceProvenanceResponse(BaseModel):
    origin: str
    source_uri: str
    source_filename: str | None
    declared_mime: str | None
    detected_mime: str | None
    size_bytes: int | None
    source_checksum: str
    ingestion_id: str | None
    ingestion_workflow_id: str | None
    ingestion_status: str | None
    ingestion_attempt_count: int
    ingestion_created_by_subject_id: str | None
    attempts: list[CitationIngestionAttemptResponse]
    download_path: str | None


class CitationDocumentProvenanceResponse(BaseModel):
    document_id: str
    document_version_id: str
    document_version: int
    title: str
    classification: str
    version_status: str
    parser_version: str
    chunker_version: str
    content_checksum: str
    extraction_checksum: str | None
    extraction_metadata: dict[str, Any] | None
    acl_subject_ids: list[str]
    acl_roles: list[str]
    device_families: list[str]
    device_models: list[str]
    valid_from: datetime
    valid_to: datetime | None
    created_by_subject_id: str | None
    reviewed_by_subject_id: str | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class CitationChunkProvenanceResponse(BaseModel):
    chunk_id: str
    ordinal: int
    token_count: int
    content_checksum: str
    embedding_dimensions: int
    created_at: datetime


class CitationAnchorProvenanceResponse(BaseModel):
    citation_id: str
    anchor_kind: str
    page_number: int | None
    bounding_box: dict[str, float] | None
    start_offset: int
    end_offset: int
    excerpt_checksum: str
    created_at: datetime


class CitationIndexReleaseProvenanceResponse(BaseModel):
    release_id: str
    name: str
    version: int
    status: str
    is_active: bool
    content_checksum: str
    created_by_subject_id: str | None
    published_by_subject_id: str | None
    published_at: datetime | None
    created_at: datetime
    updated_at: datetime


class CitationIntegrityResponse(BaseModel):
    status: str
    source_checksum_consistent: bool
    extraction_checksum_valid: bool
    chunk_checksum_valid: bool
    citation_checksum_valid: bool
    document_links_valid: bool
    release_published: bool
    chain_digest: str


class CitationProvenanceResponse(BaseModel):
    source: CitationSourceProvenanceResponse
    document: CitationDocumentProvenanceResponse
    chunk: CitationChunkProvenanceResponse
    anchor: CitationAnchorProvenanceResponse
    index_release: CitationIndexReleaseProvenanceResponse
    integrity: CitationIntegrityResponse


class CitationResponse(BaseModel):
    citation_id: str
    document_id: str
    document_version_id: str
    document_version: int
    chunk_id: str
    title: str
    source_uri: str
    source_checksum: str
    content_checksum: str
    content_excerpt: str
    page_number: int | None
    bounding_box: dict[str, float] | None
    start_offset: int
    end_offset: int
    provenance: CitationProvenanceResponse


class CitationEnvelope(BaseModel):
    data: CitationResponse
    meta: dict[str, str]


@router.post(
    "/incidents/{incident_id}/diagnoses",
    response_model=DiagnosisEnvelope,
    status_code=202,
    responses=STANDARD_ERROR_RESPONSES,
)
async def start_diagnosis(
    incident_id: str,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[DiagnosisDispatcher, Depends(get_diagnosis_dispatcher)],
    model_resolver: Annotated[ProductionModelResolver | None, Depends(get_model_resolver)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> DiagnosisEnvelope:
    try:
        result = await DiagnosisService(
            database,
            authorizer,
            dispatcher,
            model_resolver=model_resolver,
            model_alias=request.app.state.settings.model_gateway_alias,
        ).start(
            identity,
            incident_id,
            expected_version=_parse_version(if_match),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except DiagnosisVersionConflict as exc:
        raise AppError(
            status_code=409,
            code="version_conflict",
            category="conflict",
            message="Incident version conflict",
            details={"current_version": exc.current_version},
        ) from exc
    except DiagnosisPreconditionFailed as exc:
        status_code = 409 if exc.reason_code == "idempotency_conflict" else 422
        raise AppError(
            status_code=status_code,
            code=exc.reason_code,
            category="conflict" if status_code == 409 else "validation",
            message="Diagnosis preconditions are incomplete",
        ) from exc
    except DiagnosisDispatchFailed as exc:
        raise AppError(
            status_code=503,
            code="diagnosis_workflow_unavailable",
            category="dependency",
            message="Diagnosis workflow unavailable",
            retryable=True,
        ) from exc
    response.status_code = 202 if result.created else 200
    return await asyncio.to_thread(
        _diagnosis_envelope,
        result.run,
        request.state.request_id,
        result.created,
        identity,
        database,
        authorizer,
        dispatcher,
    )


@router.post(
    "/diagnosis-runs/{diagnosis_run_id}/reanalyses",
    response_model=DiagnosisEnvelope,
    status_code=202,
    responses=STANDARD_ERROR_RESPONSES,
)
async def reanalyze_diagnosis_with_confirmed_transcript(
    diagnosis_run_id: str,
    body: ReanalyzeDiagnosisRequest,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[DiagnosisDispatcher, Depends(get_diagnosis_dispatcher)],
    model_resolver: Annotated[ProductionModelResolver | None, Depends(get_model_resolver)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> DiagnosisEnvelope:
    try:
        service = DiagnosisService(
            database,
            authorizer,
            dispatcher,
            model_resolver=model_resolver,
            model_alias=request.app.state.settings.model_gateway_alias,
        )
        if body.transcript_segment_id is not None:
            result = await service.reanalyze_with_confirmed_transcript(
                identity,
                diagnosis_run_id,
                transcript_segment_id=body.transcript_segment_id,
                expected_version=_parse_version(if_match),
                idempotency_key=idempotency_key,
                request_id=request.state.request_id,
            )
        else:
            if (
                body.work_order_id is None
                or body.work_order_version is None
                or body.field_entry_ids is None
            ):
                raise DiagnosisPreconditionFailed("field_observations_invalid")
            result = await service.reanalyze_with_field_observations(
                identity,
                diagnosis_run_id,
                work_order_id=body.work_order_id,
                work_order_version=body.work_order_version,
                field_entry_ids=tuple(body.field_entry_ids),
                expected_version=_parse_version(if_match),
                idempotency_key=idempotency_key,
                request_id=request.state.request_id,
            )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except DiagnosisVersionConflict as exc:
        raise AppError(
            status_code=409,
            code="version_conflict",
            category="conflict",
            message="Source diagnosis version conflict",
            details={"current_version": exc.current_version},
        ) from exc
    except DiagnosisPreconditionFailed as exc:
        status_code = 409 if exc.reason_code == "idempotency_conflict" else 422
        raise AppError(
            status_code=status_code,
            code=exc.reason_code,
            category="conflict" if status_code == 409 else "validation",
            message="Diagnosis reanalysis preconditions are incomplete",
        ) from exc
    except DiagnosisDispatchFailed as exc:
        raise AppError(
            status_code=503,
            code="diagnosis_workflow_unavailable",
            category="dependency",
            message="Diagnosis workflow unavailable",
            retryable=True,
        ) from exc
    response.status_code = 202 if result.created else 200
    return await asyncio.to_thread(
        _diagnosis_envelope,
        result.run,
        request.state.request_id,
        result.created,
        identity,
        database,
        authorizer,
        dispatcher,
    )


@router.post(
    "/agent-runs/{agent_run_id}/cancel",
    response_model=AgentRunControlEnvelope,
    status_code=202,
    responses=STANDARD_ERROR_RESPONSES,
)
async def cancel_agent_run(
    agent_run_id: str,
    body: CancelAgentRunRequest,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[DiagnosisDispatcher, Depends(get_diagnosis_dispatcher)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> AgentRunControlEnvelope:
    try:
        result = await AgentRunControlService(database, authorizer, dispatcher).cancel(
            identity,
            agent_run_id,
            reason=body.reason,
            expected_version=_parse_version(if_match),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except AgentRunVersionConflict as exc:
        raise AppError(
            409,
            "agent_run_version_conflict",
            "conflict",
            "Agent run version conflict",
            details={"current_version": exc.current_version},
        ) from exc
    except AgentRunPreconditionFailed as exc:
        raise _agent_run_precondition_error(exc) from exc
    except AgentRunCancellationDispatchFailed as exc:
        raise AppError(
            503,
            "agent_cancellation_dispatch_unavailable",
            "dependency",
            "Agent cancellation is persisted and awaiting workflow delivery",
            retryable=True,
        ) from exc
    response.status_code = 202 if result.created else 200
    return await asyncio.to_thread(
        _agent_control_envelope, result, request.state.request_id, identity, database
    )


@router.post(
    "/agent-runs/{agent_run_id}/resume",
    response_model=DiagnosisEnvelope,
    status_code=202,
    responses=STANDARD_ERROR_RESPONSES,
)
async def resume_agent_run(
    agent_run_id: str,
    body: ResumeAgentRunRequest,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[DiagnosisDispatcher, Depends(get_diagnosis_dispatcher)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> DiagnosisEnvelope:
    try:
        result = await AgentRunControlService(database, authorizer, dispatcher).resume(
            identity,
            agent_run_id,
            clarification=body.clarification,
            expected_version=_parse_version(if_match),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except AgentRunVersionConflict as exc:
        raise AppError(
            409,
            "agent_run_version_conflict",
            "conflict",
            "Agent run version conflict",
            details={"current_version": exc.current_version},
        ) from exc
    except AgentRunPreconditionFailed as exc:
        raise _agent_run_precondition_error(exc) from exc
    except AgentRunResumeDispatchFailed as exc:
        raise AppError(
            503,
            "agent_resume_workflow_unavailable",
            "dependency",
            "Agent resume workflow unavailable",
            retryable=True,
        ) from exc
    response.status_code = 202 if result.created else 200
    return await asyncio.to_thread(
        _diagnosis_envelope,
        result.run,
        request.state.request_id,
        result.created,
        identity,
        database,
        authorizer,
        dispatcher,
    )


@router.get(
    "/diagnosis-runs/{diagnosis_run_id}",
    response_model=DiagnosisEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_diagnosis(
    diagnosis_run_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[DiagnosisDispatcher, Depends(get_diagnosis_dispatcher)],
) -> DiagnosisEnvelope:
    try:
        run = DiagnosisService(database, authorizer, dispatcher).get(
            identity,
            diagnosis_run_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    return _diagnosis_envelope(
        run,
        request.state.request_id,
        False,
        identity,
        database,
        authorizer,
        dispatcher,
    )


@router.post(
    "/diagnosis-runs/{diagnosis_run_id}/expert-interventions",
    response_model=DiagnosisEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
def take_over_diagnosis(
    diagnosis_run_id: str,
    body: ExpertDiagnosisTakeoverRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[DiagnosisDispatcher, Depends(get_diagnosis_dispatcher)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> DiagnosisEnvelope:
    try:
        ExpertDiagnosisService(database, authorizer).take_over(
            identity,
            diagnosis_run_id,
            reason=body.reason,
            expected_version=_parse_version(if_match),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
        run = DiagnosisService(database, authorizer, dispatcher).get(
            identity,
            diagnosis_run_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except ExpertDiagnosisError as exc:
        raise _expert_error(exc) from exc
    return _diagnosis_envelope(
        run,
        request.state.request_id,
        False,
        identity,
        database,
        authorizer,
        dispatcher,
    )


@router.post(
    "/diagnosis-runs/{diagnosis_run_id}/expert-interventions/{intervention_id}/revisions",
    response_model=DiagnosisEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
def submit_expert_diagnosis_revision(
    diagnosis_run_id: str,
    intervention_id: str,
    body: ExpertDiagnosisRevisionRequest,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[DiagnosisDispatcher, Depends(get_diagnosis_dispatcher)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> DiagnosisEnvelope:
    try:
        ExpertDiagnosisService(database, authorizer).submit_revision(
            identity,
            diagnosis_run_id,
            intervention_id,
            ExpertDiagnosisRevisionInput(
                outcome=body.outcome,
                conclusion=body.conclusion,
                citation_ids=tuple(body.citation_ids),
                contradictions=tuple(body.contradictions),
                missing_information=tuple(body.missing_information),
                next_checks=tuple(body.next_checks),
                confidence=body.confidence,
                reason=body.reason,
            ),
            expected_version=_parse_version(if_match),
            request_id=request.state.request_id,
        )
        run = DiagnosisService(database, authorizer, dispatcher).get(
            identity,
            diagnosis_run_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except ExpertDiagnosisError as exc:
        raise _expert_error(exc) from exc
    return _diagnosis_envelope(
        run,
        request.state.request_id,
        False,
        identity,
        database,
        authorizer,
        dispatcher,
    )


@router.get(
    "/citations/{citation_id}",
    response_model=CitationEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_citation(
    citation_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> CitationEnvelope:
    try:
        authorized = CitationProvenanceService(database, authorizer).get(
            identity,
            citation_id,
            request_id=request.state.request_id,
        )
    except KnowledgeNotVisible as exc:
        raise _not_visible() from exc
    return CitationEnvelope(
        data=_citation(authorized.citation, authorized.provenance),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/citations/{citation_id}/source",
    response_class=Response,
    response_model=None,
    responses={
        200: {
            "description": "Integrity-verified managed knowledge source",
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        },
        **STANDARD_ERROR_RESPONSES,
    },
)
async def download_citation_source(
    citation_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
) -> Response:
    try:
        source = await CitationProvenanceService(database, authorizer).read_managed_source(
            identity,
            citation_id,
            object_store,
            request_id=request.state.request_id,
        )
    except KnowledgeNotVisible as exc:
        raise _not_visible() from exc
    except CitationSourceNotAvailable as exc:
        if str(exc) == "citation_source_integrity_failed":
            raise AppError(
                503,
                "citation_source_integrity_failed",
                "dependency",
                "Citation source integrity verification failed",
                retryable=False,
            ) from exc
        raise _not_visible() from exc
    encoded_filename = quote(source.filename, safe="")
    return Response(
        content=source.content,
        media_type=source.media_type,
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}",
            "ETag": f'"{source.source_checksum}"',
            "X-Content-Type-Options": "nosniff",
            "X-Request-ID": request.state.request_id,
        },
    )


def _diagnosis_envelope(
    run: DiagnosisRun,
    request_id: str,
    created: bool,
    identity: IdentityContext,
    database: Database,
    authorizer: Authorizer,
    dispatcher: DiagnosisDispatcher,
) -> DiagnosisEnvelope:
    expert_service = ExpertDiagnosisService(database, authorizer)
    interventions = expert_service.list_for_run(
        identity,
        run.diagnosis_run_id,
        request_id=request_id,
    )
    effective_report = expert_service.effective_report(
        identity,
        run.diagnosis_run_id,
        request_id=request_id,
    )
    control_actions, agent_run_version = AgentRunControlService(
        database,
        authorizer,
        dispatcher,
    ).legal_actions(identity, run.agent_run_id)
    execution_payload = run.manifest.get("model_execution")
    model_execution = (
        ModelExecutionResponse.model_validate(execution_payload)
        if isinstance(execution_payload, dict)
        else None
    )
    # All callers, including command replays, must guard the complete response.
    # Existing report services have already authorized their reads; no DB lock
    # is held across workflow dispatch or response construction.
    _record_response_disclosure(identity, database, run.diagnosis_run_id, request_id)
    return DiagnosisEnvelope(
        data=DiagnosisRunResponse(
            diagnosis_run_id=run.diagnosis_run_id,
            incident_id=run.incident_id,
            evidence_bundle_id=run.evidence_bundle_id,
            agent_run_id=run.agent_run_id,
            agent_run_version=agent_run_version,
            workflow_id=run.workflow_id,
            stream_url=f"/api/v1/agent-runs/{run.agent_run_id}/events",
            status=run.status,
            manifest=run.manifest,
            model_execution=model_execution,
            report=effective_report,
            ai_report=run.report,
            expert_interventions=[_expert_intervention(item) for item in interventions],
            stop_reason=run.stop_reason,
            version=run.version,
            created_at=run.created_at,
            updated_at=run.updated_at,
            legal_actions=list(expert_service.legal_actions(identity, run.diagnosis_run_id))
            + control_actions,
        ),
        meta={"request_id": request_id, "created": created, "version": run.version},
    )


def _agent_control_envelope(
    result: AgentRunControlView,
    request_id: str,
    identity: IdentityContext,
    database: Database,
) -> AgentRunControlEnvelope:
    _record_response_disclosure(identity, database, result.diagnosis_run_id, request_id)
    return AgentRunControlEnvelope(
        data=AgentRunControlResponse(
            agent_run_id=result.agent_run_id,
            diagnosis_run_id=result.diagnosis_run_id,
            status=result.status,
            version=result.version,
            last_sequence=result.last_sequence,
            stream_url=f"/api/v1/agent-runs/{result.agent_run_id}/ag-ui/events",
        ),
        meta={
            "request_id": request_id,
            "created": result.created,
            "version": result.version,
        },
    )


def _record_response_disclosure(
    identity: IdentityContext, database: Database, diagnosis_run_id: str, request_id: str
) -> None:
    # Both response builders are reached only after the original business authorization.
    with database.transaction(identity.tenant_context) as session:
        record_disclosure(
            session,
            identity,
            (diagnosis_run_id,),
            resource_kind="diagnosis",
            resource_id=diagnosis_run_id,
            request_id=request_id,
        )


def _agent_run_precondition_error(exc: AgentRunPreconditionFailed) -> AppError:
    status_code = (
        409
        if exc.reason_code
        in {
            "idempotency_conflict",
            "agent_cancel_raced_with_terminal",
        }
        else 422
    )
    return AppError(
        status_code,
        exc.reason_code,
        "conflict" if status_code == 409 else "validation",
        "Agent run control preconditions are incomplete",
    )


def _expert_intervention(
    value: ExpertDiagnosisInterventionView,
) -> ExpertDiagnosisInterventionResponse:
    return ExpertDiagnosisInterventionResponse(
        intervention_id=value.intervention_id,
        diagnosis_run_id=value.diagnosis_run_id,
        status=value.status,
        source_run_version=value.source_run_version,
        ai_report_snapshot=value.ai_report_snapshot,
        ai_report_digest=value.ai_report_digest,
        takeover_reason=value.takeover_reason,
        requested_by_subject_id=value.requested_by_subject_id,
        completed_at=value.completed_at,
        version=value.version,
        revisions=[
            ExpertDiagnosisRevisionResponse(
                revision_id=item.revision_id,
                ordinal=item.ordinal,
                outcome=item.outcome,
                report=item.report,
                report_digest=item.report_digest,
                source_ai_report_digest=item.source_ai_report_digest,
                revision_reason=item.revision_reason,
                authored_by_subject_id=item.authored_by_subject_id,
                created_at=item.created_at,
            )
            for item in value.revisions
        ],
        created_at=value.created_at,
        updated_at=value.updated_at,
    )


def _citation(
    citation: CitationView,
    provenance: CitationProvenanceView,
) -> CitationResponse:
    return CitationResponse(
        **{
            field: getattr(citation, field)
            for field in CitationResponse.model_fields
            if field != "provenance"
        },
        provenance=CitationProvenanceResponse.model_validate(provenance, from_attributes=True),
    )


def _parse_version(value: str) -> int:
    try:
        version = int(value.strip().strip('"'))
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="invalid_version_precondition",
            category="validation",
            message="If-Match must contain a positive resource version",
        ) from exc
    if version < 1:
        raise AppError(
            status_code=400,
            code="invalid_version_precondition",
            category="validation",
            message="If-Match must contain a positive resource version",
        )
    return version


def _not_visible() -> AppError:
    return AppError(
        status_code=404,
        code="resource_not_found_or_not_visible",
        category="not_found",
        message="Resource not found or not visible",
    )


def _expert_error(exc: ExpertDiagnosisError) -> AppError:
    conflict_reasons = {
        "expert_diagnosis_version_conflict",
        "expert_intervention_version_conflict",
        "expert_diagnosis_idempotency_conflict",
        "expert_diagnosis_takeover_state_invalid",
        "expert_diagnosis_takeover_already_open",
        "expert_diagnosis_revision_state_invalid",
        "expert_diagnosis_ai_draft_not_ready",
    }
    return AppError(
        status_code=409 if exc.reason in conflict_reasons else 422,
        code=exc.reason,
        category="conflict" if exc.reason in conflict_reasons else "validation",
        message="Expert diagnosis intervention rejected the operation",
        details=(
            {"current_version": exc.current_version} if exc.current_version is not None else None
        ),
    )
