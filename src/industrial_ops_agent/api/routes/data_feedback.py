"""Server-authorized feedback governance commands, catalog and lineage reads."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Annotated, Any, Literal, Never, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_dataset_pipeline,
    get_dataset_store,
    get_dlp_processor,
    get_identity,
    get_label_studio_adapter,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.data_governance.dlp import PresidioDlpProcessor
from industrial_ops_agent.data_governance.service import (
    CandidateGovernanceView,
    DataGovernanceService,
    DlpRunView,
    GovernanceBlockerCode,
    GovernanceConflict,
    GovernanceGateDenied,
    GovernanceNotVisible,
    candidate_governance_blockers,
    diagnosis_feedback_count,
)
from industrial_ops_agent.data_pipeline.contracts import CONTRACT_VERSION
from industrial_ops_agent.data_pipeline.quality import (
    DatasetQualityReportError,
    verify_dataset_quality_report,
)
from industrial_ops_agent.data_pipeline.service import (
    DatasetPipelineError,
    DatasetPipelineService,
)
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.data_pipeline.training_gate import (
    DatasetTrainingBlockerCode,
    dataset_training_blockers,
)
from industrial_ops_agent.labeling.label_studio import (
    LabelStudioAdapter,
    LabelStudioUnavailable,
)
from industrial_ops_agent.labeling.service import AnnotationTaskView, LabelingService
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AnnotationReviewRecord,
    AnnotationRevisionRecord,
    AnnotationTaskRecord,
    CurationRunRecord,
    DataEligibilityDecisionRecord,
    DataLineageRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
    DlpProcessingResultRecord,
    EventInboxRecord,
    EventOutboxRecord,
    FeedbackCandidateRecord,
    IdempotencyRecord,
)

router = APIRouter(tags=["data-feedback"])

CandidateStatus = Literal[
    "PENDING_GOVERNANCE",
    "ELIGIBLE",
    "ELIGIBILITY_DENIED",
    "DLP_APPROVED",
    "DLP_REVIEW_REQUIRED",
    "ANNOTATION_PENDING",
    "ANNOTATION_REVIEW_REQUIRED",
    "ANNOTATION_CONFLICT",
    "READY_FOR_CURATION",
]
EligibilityStatus = Literal["UNDECIDED", "ELIGIBLE", "INELIGIBLE"]
CurationRunStatus = Literal["COMPLETED", "NO_DATA", "FAILED", "LINEAGE_PENDING"]


class EligibilityDecisionBody(BaseModel):
    purpose: str = Field(min_length=1, max_length=128)
    allow_training: bool
    consent_basis: str | None = Field(default=None, max_length=255)
    license_status: Literal["APPROVED", "REJECTED", "UNKNOWN"]
    retention_until: datetime | None
    policy_version: str = Field(min_length=1, max_length=128)


class DlpRunBody(BaseModel):
    policy_version: str = Field(min_length=1, max_length=128)


class AnnotationTaskBody(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    schema_version: str = Field(min_length=1, max_length=128)
    risk_level: Literal["STANDARD", "HIGH"]


class CandidateGovernanceResponse(BaseModel):
    candidate_id: str
    status: str
    eligibility_status: str
    allow_training: bool | None
    consent_basis: str | None
    license_status: str
    retention_until: datetime | None
    governance_blockers: list[GovernanceBlockerCode]
    version: int
    legal_actions: list[str]


class DlpRunResponse(BaseModel):
    dlp_result_id: str
    candidate_id: str
    status: str
    policy_version: str
    source_content_hash: str
    content_hash: str
    redacted_content: dict[str, str]
    findings: list[dict[str, Any]]
    residual_entity_types: list[str]
    candidate_version: int


class AnnotationTaskResponse(BaseModel):
    task_id: str
    candidate_id: str
    status: str
    risk_level: str
    required_reviews: int
    completed_reviews: int
    external_task_id: str
    external_version: str
    version: int
    candidate_version: int
    requires_arbitration: bool
    legal_actions: list[str]


class CandidateEnvelope(BaseModel):
    data: CandidateGovernanceResponse
    meta: dict[str, str]


class DlpEnvelope(BaseModel):
    data: DlpRunResponse
    meta: dict[str, str]


class AnnotationTaskEnvelope(BaseModel):
    data: AnnotationTaskResponse
    meta: dict[str, str]


class TimelineStageResponse(BaseModel):
    stage: str
    status: str
    resource_id: str
    occurred_at: datetime | None = None
    facts: dict[str, str | int | bool | None] = Field(default_factory=dict)


class FeedbackCandidateSummaryResponse(BaseModel):
    candidate_id: str
    work_order_id: str
    incident_id: str
    source_event_id: str
    source_content_hash: str
    data_classification: str
    status: str
    eligibility_status: str
    allow_training: bool | None
    license_status: str
    retention_until: datetime | None
    governance_blockers: list[GovernanceBlockerCode]
    diagnosis_feedback_count: int
    version: int
    created_at: datetime
    legal_actions: list[str]


class EligibilityDecisionResponse(BaseModel):
    decision_id: str
    allow_training: bool
    purpose: str
    consent_basis: str | None
    license_status: str
    retention_until: datetime | None
    policy_version: str
    candidate_version: int
    decided_by_subject_id: str
    created_at: datetime


class DlpProcessingHistoryResponse(BaseModel):
    dlp_result_id: str
    status: str
    policy_version: str
    source_content_hash: str
    content_hash: str
    finding_counts: dict[str, int]
    residual_entity_types: list[str]
    candidate_version: int
    processed_by_subject_id: str
    created_at: datetime


class AnnotationRevisionAuditResponse(BaseModel):
    revision_id: str
    external_annotation_id: str
    external_version: str
    reviewer_subject_id: str
    labels_digest: str
    task_payload_hash: str
    submitted_at: datetime


class AnnotationReviewAuditResponse(BaseModel):
    review_id: str
    revision_id: str
    reviewer_subject_id: str
    outcome: str
    consistency_key: str
    reviewed_at: datetime


class AnnotationTaskHistoryResponse(BaseModel):
    task_id: str
    dlp_result_id: str
    project_id: str
    external_task_id: str
    external_version: str
    payload_hash: str
    schema_version: str
    risk_level: str
    required_reviews: int
    status: str
    consistency_status: str
    created_by_subject_id: str
    version: int
    created_at: datetime
    revisions: list[AnnotationRevisionAuditResponse]
    reviews: list[AnnotationReviewAuditResponse]


class FeedbackCandidateDetailResponse(FeedbackCandidateSummaryResponse):
    completion_id: str
    verification_id: str
    eligibility_decisions: list[EligibilityDecisionResponse]
    dlp_runs: list[DlpProcessingHistoryResponse]
    annotation_tasks: list[AnnotationTaskHistoryResponse]
    timeline: list[TimelineStageResponse]


class FeedbackCandidateListEnvelope(BaseModel):
    data: list[FeedbackCandidateSummaryResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


class FeedbackCandidateDetailEnvelope(BaseModel):
    data: FeedbackCandidateDetailResponse
    meta: dict[str, str]


class CurationRunBody(BaseModel):
    window_start: datetime
    window_end: datetime
    engine: Literal["local", "spark"]


class CurationExclusionResponse(BaseModel):
    candidate_id: str
    reason: str


class CurationRunResponse(BaseModel):
    run_id: str
    status: CurationRunStatus
    engine: str
    window_start: datetime
    window_end: datetime
    contract_version: str
    code_version: str
    config_hash: str
    input_manifest_hash: str
    input_count: int
    exclusion_report: list[CurationExclusionResponse]
    failure_reason: str | None
    snapshot_id: str | None
    created_at: datetime


class CurationRunEnvelope(BaseModel):
    data: CurationRunResponse
    meta: dict[str, str]


class CurationRunListEnvelope(BaseModel):
    data: list[CurationRunResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


class DatasetArtifactResponse(BaseModel):
    artifact_id: str
    kind: str
    split: str | None
    content_hash: str
    size_bytes: int
    row_count: int
    schema_hash: str


class DatasetSnapshotResponse(BaseModel):
    snapshot_id: str
    run_id: str
    status: str
    contract_version: str
    input_manifest_hash: str
    manifest_hash: str | None
    row_count: int
    split_counts: dict[str, int]
    source_work_order_ids: list[str]
    lineage_status: str
    lineage_id: str | None
    training_eligible: bool
    training_blockers: list[DatasetTrainingBlockerCode]
    base_snapshot_id: str | None
    augmentation_contract_version: str | None
    sample_origin_counts: dict[str, int] | None
    synthetic_split_counts: dict[str, int] | None
    created_at: datetime
    artifacts: list[DatasetArtifactResponse] = Field(default_factory=list)
    legal_actions: list[str]


class DatasetSnapshotListEnvelope(BaseModel):
    data: list[DatasetSnapshotResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


class DatasetSnapshotEnvelope(BaseModel):
    data: DatasetSnapshotResponse
    meta: dict[str, str]


class DatasetQualityReportResponse(BaseModel):
    contract_version: str
    pandera_validation: Literal["PASSED"]
    row_count: int
    split_counts: dict[str, int]
    group_leakage_count: int
    exclusions: list[dict[str, str]]
    content_hash: str


class DatasetQualityReportEnvelope(BaseModel):
    data: DatasetQualityReportResponse
    meta: dict[str, str]


class DataLineageSourceChainResponse(BaseModel):
    work_order_id: str
    candidate_id: str | None
    stages: list[TimelineStageResponse]


class DataLineageResponse(BaseModel):
    lineage_id: str
    openlineage_run_id: str
    run_id: str
    snapshot_id: str
    job_namespace: str
    job_name: str
    status: str
    source_work_order_ids: list[str]
    source_chains: list[DataLineageSourceChainResponse]
    pipeline_stages: list[TimelineStageResponse]
    failure_reason: str | None
    created_at: datetime


class DataLineageEnvelope(BaseModel):
    data: DataLineageResponse
    meta: dict[str, str]


@router.get(
    "/data-feedback/candidates",
    response_model=FeedbackCandidateListEnvelope,
)
def list_feedback_candidates(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[CandidateStatus | None, Query()] = None,
    eligibility_status: Annotated[EligibilityStatus | None, Query()] = None,
    work_order_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128),
    ] = None,
    governance_blocker: Annotated[GovernanceBlockerCode | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> FeedbackCandidateListEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_DATA_FEEDBACK,
        "feedback-candidates",
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        filters = [FeedbackCandidateRecord.tenant_id == identity.tenant_id]
        if status is not None:
            filters.append(FeedbackCandidateRecord.status == status)
        if eligibility_status is not None:
            filters.append(FeedbackCandidateRecord.eligibility_status == eligibility_status)
        if work_order_id is not None:
            filters.append(FeedbackCandidateRecord.work_order_id == work_order_id)
        if governance_blocker is not None:
            filters.append(_governance_blocker_filter(governance_blocker, datetime.now(UTC)))
        total = (
            session.scalar(
                select(func.count()).select_from(FeedbackCandidateRecord).where(*filters)
            )
            or 0
        )
        candidates = list(
            session.scalars(
                select(FeedbackCandidateRecord)
                .where(*filters)
                .order_by(
                    FeedbackCandidateRecord.created_at.desc(),
                    FeedbackCandidateRecord.candidate_id,
                )
                .offset(offset)
                .limit(limit)
            )
        )
        data = [_candidate_summary(item, identity, authorizer) for item in candidates]
    return FeedbackCandidateListEnvelope(
        data=data,
        meta={
            "request_id": request.state.request_id,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        legal_actions=(
            ["START_CURATION_RUN"]
            if authorizer.decide(
                identity,
                Action.START_CURATION_RUN,
                ResourceContext(tenant_id=identity.tenant_id, resource_id="curation-runs"),
            ).allowed
            else []
        ),
    )


@router.get(
    "/data-feedback/candidates/{candidate_id}",
    response_model=FeedbackCandidateDetailEnvelope,
)
def get_feedback_candidate(
    candidate_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> FeedbackCandidateDetailEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_DATA_FEEDBACK,
        candidate_id,
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        candidate = _candidate_or_404(session, identity.tenant_id, candidate_id)
        summary = _candidate_summary(candidate, identity, authorizer)
        eligibility_decisions = _eligibility_decision_history(session, candidate)
        dlp_runs = _dlp_processing_history(session, candidate)
        annotation_tasks = _annotation_task_history(session, candidate)
        timeline = _candidate_timeline(session, candidate)
    return FeedbackCandidateDetailEnvelope(
        data=FeedbackCandidateDetailResponse(
            **summary.model_dump(),
            completion_id=candidate.completion_id,
            verification_id=candidate.verification_id,
            eligibility_decisions=eligibility_decisions,
            dlp_runs=dlp_runs,
            annotation_tasks=annotation_tasks,
            timeline=timeline,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/curation-runs",
    response_model=CurationRunEnvelope,
    status_code=201,
)
def start_curation_run(
    body: CurationRunBody,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    pipeline: Annotated[DatasetPipelineService, Depends(get_dataset_pipeline)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> CurationRunEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.START_CURATION_RUN,
        "curation-runs",
        request.state.request_id,
    )
    try:
        data, created = _build_curation_once(
            database, identity, pipeline, body, idempotency_key
        )
    except ValueError as exc:
        raise AppError(400, "invalid_curation_window", "validation", str(exc)) from exc
    except DatasetPipelineError as exc:
        raise AppError(409, "curation_gate_rejected", "governance_gate", str(exc)) from exc
    if isinstance(data, AppError):
        raise data
    if not created:
        response.status_code = 200
    return CurationRunEnvelope(data=data, meta={"request_id": request.state.request_id})


def _build_curation_once(
    database: Database,
    identity: IdentityContext,
    pipeline: DatasetPipelineService,
    body: CurationRunBody,
    idempotency_key: str,
) -> tuple[CurationRunResponse | AppError, bool]:
    if body.window_start >= body.window_end:
        raise ValueError("window_start must be before window_end")
    request_hash = _digest_json(body.model_dump(mode="json"))
    storage_key = f"curation:{idempotency_key}"
    lookup = select(IdempotencyRecord).where(
        IdempotencyRecord.tenant_id == identity.tenant_id,
        IdempotencyRecord.subject_id == identity.subject_id,
        IdempotencyRecord.key == storage_key,
    )
    run_id = str(uuid4())
    now = datetime.now(UTC)
    # Commit the unique claim before any object/lineage writes. Do not hold a
    # database connection or write lock across the pipeline's own transactions.
    try:
        with database.transaction(identity.tenant_context) as session:
            existing = session.scalar(lookup)
            if existing is not None:
                return _replay_curation(session, existing, request_hash), False
            session.add(
                IdempotencyRecord(
                    record_id=f"idempotency-curation-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=run_id,
                    expires_at=now + timedelta(days=30),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
    except IntegrityError:
        # Only a competing claim for this exact key is a replay. Other database
        # failures keep their original exception instead of being called success.
        with database.transaction(identity.tenant_context) as session:
            existing = session.scalar(lookup)
            if existing is None:
                raise
            return _replay_curation(session, existing, request_hash), False

    result = pipeline.build_snapshot(
        identity.tenant_context,
        window_start=body.window_start,
        window_end=body.window_end,
        engine=body.engine,
        reserved_run_id=run_id,
    )
    if result.run_id != run_id:
        raise RuntimeError("Curation result diverged from its reserved run")
    with database.transaction(identity.tenant_context) as session:
        run = _curation_run_or_404(session, identity.tenant_id, run_id)
        return _curation_run_response(session, run), True


def _replay_curation(
    session: Session, existing: IdempotencyRecord, request_hash: str
) -> CurationRunResponse | AppError:
    # AppError is frozen; carry it out of the generator-based transaction before
    # raising, so contextlib does not try to mutate its __traceback__.
    if existing.request_hash != request_hash:
        return AppError(
            409, "idempotency_conflict", "conflict",
            "Idempotency key was reused for another curation window",
        )
    run = session.scalar(
        select(CurationRunRecord).where(
            CurationRunRecord.tenant_id == existing.tenant_id,
            CurationRunRecord.run_id == existing.result_ref,
        )
    )
    if run is None:
        # The owner may still be building, or may have stopped after external
        # writes. Never take over blindly and publish a second snapshot.
        return AppError(
            409, "curation_in_progress_or_unresolved", "conflict",
            "Curation is still running or requires reconciliation; retry the same operation",
            retryable=True,
            details={"run_id": existing.result_ref},
        )
    if run.status == "FAILED":
        return AppError(
            409, "curation_gate_rejected", "governance_gate",
            run.failure_reason or "Curation failed",
            details={"run_id": run.run_id},
        )
    return _curation_run_response(session, run)


@router.get("/curation-runs/{run_id}", response_model=CurationRunEnvelope)
def get_curation_run(
    run_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> CurationRunEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_CURATION_RUN,
        run_id,
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        run = _curation_run_or_404(session, identity.tenant_id, run_id)
        data = _curation_run_response(session, run)
    return CurationRunEnvelope(data=data, meta={"request_id": request.state.request_id})


@router.get("/curation-runs", response_model=CurationRunListEnvelope)
def list_curation_runs(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    run_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ] = None,
    status: Annotated[
        CurationRunStatus | None,
        Query(),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CurationRunListEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_CURATION_RUN,
        "curation-runs",
        request.state.request_id,
    )
    filters = [CurationRunRecord.tenant_id == identity.tenant_id]
    if run_id is not None:
        filters.append(CurationRunRecord.run_id == run_id)
    if status is not None:
        filters.append(CurationRunRecord.status == status)
    with database.transaction(identity.tenant_context) as session:
        total = int(
            session.scalar(select(func.count()).select_from(CurationRunRecord).where(*filters)) or 0
        )
        runs = list(
            session.scalars(
                select(CurationRunRecord)
                .where(*filters)
                .order_by(CurationRunRecord.created_at.desc(), CurationRunRecord.run_id)
                .offset(offset)
                .limit(limit)
            )
        )
        data = [_curation_run_response(session, run) for run in runs]
    start_allowed = authorizer.decide(
        identity,
        Action.START_CURATION_RUN,
        ResourceContext(tenant_id=identity.tenant_id, resource_id="curation-runs"),
    ).allowed
    return CurationRunListEnvelope(
        data=data,
        meta={
            "request_id": request.state.request_id,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        legal_actions=["START_CURATION_RUN"] if start_allowed else [],
    )


@router.get("/dataset-snapshots", response_model=DatasetSnapshotListEnvelope)
def list_dataset_snapshots(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    snapshot_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ] = None,
    run_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ] = None,
    status: Annotated[
        Literal["CANDIDATE", "LINEAGE_PENDING"] | None,
        Query(),
    ] = None,
    lineage_status: Annotated[
        Literal["CONFIRMED", "PENDING"] | None,
        Query(),
    ] = None,
    training_eligible: Annotated[bool | None, Query()] = None,
    training_blocker: Annotated[DatasetTrainingBlockerCode | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetSnapshotListEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_DATASET_SNAPSHOT,
        "dataset-snapshots",
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        filters = [DatasetSnapshotRecord.tenant_id == identity.tenant_id]
        if snapshot_id is not None:
            filters.append(DatasetSnapshotRecord.snapshot_id == snapshot_id)
        if run_id is not None:
            filters.append(DatasetSnapshotRecord.run_id == run_id)
        if status is not None:
            filters.append(DatasetSnapshotRecord.status == status)
        if lineage_status is not None:
            filters.append(DatasetSnapshotRecord.lineage_status == lineage_status)
        if training_eligible is not None:
            filters.append(_dataset_training_eligible_filter(training_eligible))
        if training_blocker is not None:
            filters.append(_dataset_training_blocker_filter(training_blocker))
        total = int(
            session.scalar(select(func.count()).select_from(DatasetSnapshotRecord).where(*filters))
            or 0
        )
        snapshots = list(
            session.scalars(
                select(DatasetSnapshotRecord)
                .where(*filters)
                .order_by(
                    DatasetSnapshotRecord.created_at.desc(),
                    DatasetSnapshotRecord.snapshot_id,
                )
                .offset(offset)
                .limit(limit)
            )
        )
        data = [
            _dataset_snapshot_response(
                session,
                item,
                identity=identity,
                authorizer=authorizer,
                include_artifacts=False,
            )
            for item in snapshots
        ]
    return DatasetSnapshotListEnvelope(
        data=data,
        meta={
            "request_id": request.state.request_id,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        legal_actions=[],
    )


@router.get(
    "/dataset-snapshots/{snapshot_id}",
    response_model=DatasetSnapshotEnvelope,
)
def get_dataset_snapshot(
    snapshot_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> DatasetSnapshotEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_DATASET_SNAPSHOT,
        snapshot_id,
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        snapshot = _dataset_snapshot_or_404(session, identity.tenant_id, snapshot_id)
        data = _dataset_snapshot_response(
            session,
            snapshot,
            identity=identity,
            authorizer=authorizer,
            include_artifacts=True,
        )
    return DatasetSnapshotEnvelope(data=data, meta={"request_id": request.state.request_id})


@router.get(
    "/dataset-snapshots/{snapshot_id}/quality-report",
    response_model=DatasetQualityReportEnvelope,
)
def get_dataset_quality_report(
    snapshot_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    store: Annotated[DatasetStore, Depends(get_dataset_store)],
) -> DatasetQualityReportEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_DATASET_SNAPSHOT,
        snapshot_id,
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        snapshot = _dataset_snapshot_or_404(session, identity.tenant_id, snapshot_id)
        quality_artifacts = list(
            session.scalars(
                select(DatasetArtifactRecord).where(
                    DatasetArtifactRecord.tenant_id == identity.tenant_id,
                    DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
                    DatasetArtifactRecord.kind == "quality_report",
                )
            )
        )
        quality_key = snapshot.quality_report_key
    if quality_key is None or len(quality_artifacts) != 1:
        raise AppError(
            409,
            "dataset_quality_report_not_available",
            "governance_gate",
            "Dataset quality report is not available",
        )
    try:
        payload = store.get_bytes(quality_key)
    except FileNotFoundError as exc:
        raise AppError(
            503,
            "dataset_quality_report_unavailable",
            "dependency",
            "Dataset quality report is unavailable",
            retryable=True,
        ) from exc
    try:
        report = verify_dataset_quality_report(snapshot, quality_artifacts[0], payload)
    except DatasetQualityReportError as exc:
        raise AppError(
            503,
            "dataset_quality_report_integrity_failed",
            "dependency",
            "Dataset quality report integrity check failed",
        ) from exc
    return DatasetQualityReportEnvelope(
        data=DatasetQualityReportResponse(
            contract_version=report.contract_version,
            pandera_validation="PASSED",
            row_count=report.row_count,
            split_counts=report.split_counts,
            group_leakage_count=report.group_leakage_count,
            exclusions=list(report.exclusions),
            content_hash=report.content_hash,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.get("/dataset-snapshots/{snapshot_id}/manifest")
def download_dataset_manifest(
    snapshot_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    store: Annotated[DatasetStore, Depends(get_dataset_store)],
) -> Response:
    _authorize(
        authorizer,
        identity,
        Action.DOWNLOAD_DATASET_MANIFEST,
        snapshot_id,
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        snapshot = _dataset_snapshot_or_404(session, identity.tenant_id, snapshot_id)
        lineage = session.scalar(
            select(DataLineageRunRecord).where(
                DataLineageRunRecord.tenant_id == snapshot.tenant_id,
                DataLineageRunRecord.snapshot_id == snapshot.snapshot_id,
            )
        )
        quality_artifacts = list(
            session.scalars(
                select(DatasetArtifactRecord).where(
                    DatasetArtifactRecord.tenant_id == snapshot.tenant_id,
                    DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
                    DatasetArtifactRecord.kind == "quality_report",
                )
            )
        )
        training_blockers = list(dataset_training_blockers(snapshot, lineage, quality_artifacts))
        manifest_key = snapshot.manifest_key
        expected_manifest_hash = snapshot.manifest_hash
    if training_blockers:
        raise AppError(
            409,
            "dataset_manifest_not_releasable",
            "governance_gate",
            "Dataset manifest is not releasable",
            details={"training_blockers": training_blockers},
        )
    assert manifest_key is not None
    assert expected_manifest_hash is not None
    try:
        payload = store.get_bytes(manifest_key)
    except FileNotFoundError as exc:
        raise AppError(
            503,
            "dataset_manifest_unavailable",
            "dependency",
            "Dataset manifest unavailable",
            retryable=True,
        ) from exc
    actual_manifest_hash = f"sha256:{sha256(payload).hexdigest()}"
    if actual_manifest_hash != expected_manifest_hash:
        raise AppError(
            503,
            "dataset_manifest_integrity_failed",
            "dependency",
            "Dataset manifest integrity check failed",
        )
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AppError(
            503,
            "dataset_manifest_integrity_failed",
            "dependency",
            "Dataset manifest integrity check failed",
        ) from exc
    if (
        not isinstance(parsed, dict)
        or parsed.get("snapshot_id") != snapshot_id
        or parsed.get("tenant_id") != identity.tenant_id
    ):
        raise AppError(
            503,
            "dataset_manifest_integrity_failed",
            "dependency",
            "Dataset manifest integrity check failed",
        )
    return Response(
        content=payload,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{snapshot_id}-manifest.json"',
            "Cache-Control": "private, no-store",
            "ETag": f'"{expected_manifest_hash}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/data-lineage/{lineage_run_id}", response_model=DataLineageEnvelope)
def get_data_lineage(
    lineage_run_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> DataLineageEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_DATA_LINEAGE,
        lineage_run_id,
        request.state.request_id,
    )
    data: DataLineageResponse | None = None
    with database.transaction(identity.tenant_context) as session:
        lineage = session.scalar(
            select(DataLineageRunRecord).where(
                DataLineageRunRecord.tenant_id == identity.tenant_id,
                (
                    (DataLineageRunRecord.lineage_id == lineage_run_id)
                    | (DataLineageRunRecord.openlineage_run_id == lineage_run_id)
                ),
            )
        )
        if lineage is not None:
            data = DataLineageResponse(
                lineage_id=lineage.lineage_id,
                openlineage_run_id=lineage.openlineage_run_id,
                run_id=lineage.run_id,
                snapshot_id=lineage.snapshot_id,
                job_namespace=lineage.job_namespace,
                job_name=lineage.job_name,
                status=lineage.status,
                source_work_order_ids=lineage.source_work_order_ids,
                source_chains=_data_lineage_source_chains(session, lineage),
                pipeline_stages=_data_lineage_pipeline_stages(session, lineage),
                failure_reason=lineage.failure_reason,
                created_at=lineage.created_at,
            )
    if data is None:
        _not_visible()
    return DataLineageEnvelope(data=data, meta={"request_id": request.state.request_id})


@router.post(
    "/data-feedback/candidates/{candidate_id}/eligibility-decisions",
    response_model=CandidateEnvelope,
)
def decide_eligibility(
    candidate_id: str,
    body: EligibilityDecisionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dlp_processor: Annotated[PresidioDlpProcessor, Depends(get_dlp_processor)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> CandidateEnvelope:
    try:
        view = DataGovernanceService(database, authorizer, dlp_processor).decide_eligibility(
            identity,
            candidate_id,
            expected_version=_version(if_match),
            purpose=body.purpose,
            allow_training=body.allow_training,
            consent_basis=body.consent_basis,
            license_status=body.license_status,
            retention_until=body.retention_until,
            policy_version=body.policy_version,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        GovernanceNotVisible,
        GovernanceConflict,
        GovernanceGateDenied,
    ) as exc:
        raise _translate(exc) from exc
    return CandidateEnvelope(
        data=_candidate_response(view),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/data-feedback/candidates/{candidate_id}/dlp-runs",
    response_model=DlpEnvelope,
)
def run_dlp(
    candidate_id: str,
    body: DlpRunBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dlp_processor: Annotated[PresidioDlpProcessor, Depends(get_dlp_processor)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> DlpEnvelope:
    try:
        view = DataGovernanceService(database, authorizer, dlp_processor).run_dlp(
            identity,
            candidate_id,
            expected_version=_version(if_match),
            policy_version=body.policy_version,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        GovernanceNotVisible,
        GovernanceConflict,
        GovernanceGateDenied,
    ) as exc:
        raise _translate(exc) from exc
    return DlpEnvelope(
        data=_dlp_response(view),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/data-feedback/candidates/{candidate_id}/annotation-tasks",
    response_model=AnnotationTaskEnvelope,
)
def create_annotation_task(
    candidate_id: str,
    body: AnnotationTaskBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    adapter: Annotated[LabelStudioAdapter, Depends(get_label_studio_adapter)],
    dlp_processor: Annotated[PresidioDlpProcessor, Depends(get_dlp_processor)],
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> AnnotationTaskEnvelope:
    try:
        view = LabelingService(database, authorizer, adapter, dlp_processor).create_task(
            identity,
            candidate_id,
            expected_version=_version(if_match),
            project_id=body.project_id,
            schema_version=body.schema_version,
            risk_level=body.risk_level,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except LabelStudioUnavailable as exc:
        raise _label_studio_unavailable() from exc
    except ValueError as exc:
        raise AppError(
            400,
            "label_studio_project_invalid",
            "validation",
            "Label Studio project is invalid",
        ) from exc
    except (
        AuthorizationDenied,
        GovernanceNotVisible,
        GovernanceConflict,
        GovernanceGateDenied,
    ) as exc:
        raise _translate(exc) from exc
    return _task_envelope(view, request)


@router.post(
    "/annotation-tasks/{task_id}/sync",
    response_model=AnnotationTaskEnvelope,
)
def sync_annotation_task(
    task_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    adapter: Annotated[LabelStudioAdapter, Depends(get_label_studio_adapter)],
    dlp_processor: Annotated[PresidioDlpProcessor, Depends(get_dlp_processor)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> AnnotationTaskEnvelope:
    try:
        view = LabelingService(database, authorizer, adapter, dlp_processor).sync_task(
            identity,
            task_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except LabelStudioUnavailable as exc:
        raise _label_studio_unavailable() from exc
    except (
        AuthorizationDenied,
        GovernanceNotVisible,
        GovernanceConflict,
        GovernanceGateDenied,
    ) as exc:
        raise _translate(exc) from exc
    return _task_envelope(view, request)


def _authorize(
    authorizer: Authorizer,
    identity: IdentityContext,
    action: Action,
    resource_id: str,
    request_id: str,
) -> None:
    try:
        authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )
    except AuthorizationDenied as exc:
        raise _translate(exc) from exc


def _candidate_or_404(
    session: Session,
    tenant_id: str,
    candidate_id: str,
) -> FeedbackCandidateRecord:
    candidate = session.scalar(
        select(FeedbackCandidateRecord).where(
            FeedbackCandidateRecord.tenant_id == tenant_id,
            FeedbackCandidateRecord.candidate_id == candidate_id,
        )
    )
    if candidate is None:
        _not_visible()
    return candidate


def _curation_run_or_404(
    session: Session,
    tenant_id: str,
    run_id: str,
) -> CurationRunRecord:
    run = session.scalar(
        select(CurationRunRecord).where(
            CurationRunRecord.tenant_id == tenant_id,
            CurationRunRecord.run_id == run_id,
        )
    )
    if run is None:
        _not_visible()
    return run


def _dataset_snapshot_or_404(
    session: Session,
    tenant_id: str,
    snapshot_id: str,
) -> DatasetSnapshotRecord:
    snapshot = session.scalar(
        select(DatasetSnapshotRecord).where(
            DatasetSnapshotRecord.tenant_id == tenant_id,
            DatasetSnapshotRecord.snapshot_id == snapshot_id,
        )
    )
    if snapshot is None:
        _not_visible()
    return snapshot


def _not_visible() -> Never:
    raise AppError(
        404,
        "resource_not_found_or_not_visible",
        "not_found",
        "Resource not found or not visible",
    )


def _governance_blocker_filter(
    blocker: GovernanceBlockerCode,
    now: datetime,
) -> ColumnElement[bool]:
    candidate = FeedbackCandidateRecord
    authorized = and_(
        candidate.allow_training.is_(True),
        candidate.eligibility_status == "ELIGIBLE",
    )
    consent_present = and_(
        candidate.consent_basis.is_not(None),
        candidate.consent_basis != "",
    )
    license_approved = candidate.license_status == "APPROVED"
    retention_present = candidate.retention_until.is_not(None)
    eligibility_passed = and_(
        authorized,
        consent_present,
        license_approved,
        retention_present,
        candidate.retention_until > now,
    )
    if blocker == "training_not_authorized":
        return or_(
            candidate.allow_training.is_not(True),
            candidate.eligibility_status != "ELIGIBLE",
        )
    if blocker == "consent_basis_missing":
        return and_(
            authorized,
            or_(candidate.consent_basis.is_(None), candidate.consent_basis == ""),
        )
    if blocker == "license_not_approved":
        return and_(authorized, consent_present, candidate.license_status != "APPROVED")
    if blocker == "retention_missing":
        return and_(
            authorized, consent_present, license_approved, candidate.retention_until.is_(None)
        )
    if blocker == "retention_expired":
        return and_(
            authorized,
            consent_present,
            license_approved,
            retention_present,
            candidate.retention_until <= now,
        )
    stage_status = {
        "dlp_not_run": "ELIGIBLE",
        "dlp_review_required": "DLP_REVIEW_REQUIRED",
        "annotation_not_started": "DLP_APPROVED",
        "annotation_review_pending": "ANNOTATION_PENDING",
        "annotation_review_required": "ANNOTATION_REVIEW_REQUIRED",
        "annotation_conflict": "ANNOTATION_CONFLICT",
    }.get(blocker)
    if stage_status is not None:
        return and_(eligibility_passed, candidate.status == stage_status)
    return and_(
        eligibility_passed,
        candidate.status.not_in(
            (
                "ELIGIBLE",
                "DLP_REVIEW_REQUIRED",
                "DLP_APPROVED",
                "ANNOTATION_PENDING",
                "ANNOTATION_REVIEW_REQUIRED",
                "ANNOTATION_CONFLICT",
                "READY_FOR_CURATION",
            )
        ),
    )


def _candidate_summary(
    candidate: FeedbackCandidateRecord,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> FeedbackCandidateSummaryResponse:
    actions: list[str] = []
    resource = ResourceContext(
        tenant_id=identity.tenant_id,
        resource_id=candidate.candidate_id,
    )
    candidates = (
        (
            "DECIDE_ELIGIBILITY",
            Action.DECIDE_DATA_ELIGIBILITY,
            candidate.status in {"PENDING_GOVERNANCE", "ELIGIBLE", "ELIGIBILITY_DENIED"},
        ),
        ("RUN_DLP", Action.RUN_DATA_DLP, candidate.status == "ELIGIBLE"),
        (
            "CREATE_ANNOTATION_TASK",
            Action.CREATE_ANNOTATION_TASK,
            candidate.status == "DLP_APPROVED",
        ),
        (
            "SYNC_ANNOTATION",
            Action.SYNC_ANNOTATION_TASK,
            candidate.status
            in {"ANNOTATION_PENDING", "ANNOTATION_REVIEW_REQUIRED", "ANNOTATION_CONFLICT"},
        ),
    )
    for label, action, state_allows in candidates:
        if state_allows and authorizer.decide(identity, action, resource).allowed:
            actions.append(label)
    return FeedbackCandidateSummaryResponse(
        candidate_id=candidate.candidate_id,
        work_order_id=candidate.work_order_id,
        incident_id=candidate.incident_id,
        source_event_id=candidate.source_event_id,
        source_content_hash=candidate.source_content_hash,
        data_classification=candidate.data_classification,
        status=candidate.status,
        eligibility_status=candidate.eligibility_status,
        allow_training=candidate.allow_training,
        license_status=candidate.license_status,
        retention_until=candidate.retention_until,
        governance_blockers=candidate_governance_blockers(candidate, datetime.now(UTC)),
        diagnosis_feedback_count=diagnosis_feedback_count(candidate),
        version=candidate.version,
        created_at=candidate.created_at,
        legal_actions=actions,
    )


def _candidate_timeline(
    session: Session,
    candidate: FeedbackCandidateRecord,
    *,
    include_dataset_stages: bool = True,
) -> list[TimelineStageResponse]:
    stages = [
        TimelineStageResponse(
            stage="WORK_ORDER",
            status="CLOSED",
            resource_id=candidate.work_order_id,
            facts={
                "incident_id": candidate.incident_id,
                "completion_id": candidate.completion_id,
                "verification_id": candidate.verification_id,
            },
        )
    ]
    outbox = session.scalar(
        select(EventOutboxRecord).where(
            EventOutboxRecord.tenant_id == candidate.tenant_id,
            EventOutboxRecord.event_id == candidate.source_event_id,
        )
    )
    if outbox is not None:
        stages.append(
            TimelineStageResponse(
                stage="OUTBOX_EVENT",
                status="PUBLISHED" if outbox.published_at else "COMMITTED",
                resource_id=outbox.event_id,
                occurred_at=outbox.occurred_at,
                facts={
                    "aggregate_version": outbox.aggregate_version,
                    "event_version": outbox.event_version,
                    "trace_context_profile": (
                        outbox.trace_context_profile or "LEGACY_CORRELATION_ONLY"
                    ),
                    "source_trace_id": outbox.trace_id,
                },
            )
        )
    inbox = session.scalar(
        select(EventInboxRecord).where(
            EventInboxRecord.tenant_id == candidate.tenant_id,
            EventInboxRecord.event_id == candidate.source_event_id,
        )
    )
    if inbox is not None:
        stages.append(
            TimelineStageResponse(
                stage="INBOX_EVENT",
                status=inbox.status,
                resource_id=inbox.inbox_id,
                occurred_at=inbox.received_at,
                facts={
                    "aggregate_version": inbox.aggregate_version,
                    "trace_context_propagation": (
                        inbox.trace_context_propagation or "LEGACY_ROOT"
                    ),
                    "parent_trace_id": inbox.parent_trace_id,
                },
            )
        )
    stages.append(
        TimelineStageResponse(
            stage="FEEDBACK_CANDIDATE",
            status=candidate.status,
            resource_id=candidate.candidate_id,
            occurred_at=candidate.created_at,
            facts={
                "version": candidate.version,
                "source_content_hash": candidate.source_content_hash,
            },
        )
    )
    eligibility = session.scalar(
        select(DataEligibilityDecisionRecord)
        .where(
            DataEligibilityDecisionRecord.tenant_id == candidate.tenant_id,
            DataEligibilityDecisionRecord.candidate_id == candidate.candidate_id,
        )
        .order_by(DataEligibilityDecisionRecord.created_at.desc())
    )
    if eligibility is not None:
        stages.append(
            TimelineStageResponse(
                stage="ELIGIBILITY",
                status="ELIGIBLE" if eligibility.allow_training else "INELIGIBLE",
                resource_id=eligibility.decision_id,
                occurred_at=eligibility.created_at,
                facts={
                    "purpose": eligibility.purpose,
                    "policy_version": eligibility.policy_version,
                    "license_status": eligibility.license_status,
                },
            )
        )
    dlp = session.scalar(
        select(DlpProcessingResultRecord)
        .where(
            DlpProcessingResultRecord.tenant_id == candidate.tenant_id,
            DlpProcessingResultRecord.candidate_id == candidate.candidate_id,
        )
        .order_by(DlpProcessingResultRecord.created_at.desc())
    )
    if dlp is not None:
        stages.append(
            TimelineStageResponse(
                stage="DLP",
                status=dlp.status,
                resource_id=dlp.dlp_result_id,
                occurred_at=dlp.created_at,
                facts={
                    "policy_version": dlp.policy_version,
                    "content_hash": dlp.content_hash,
                    "residual_entity_count": len(dlp.residual_entity_types),
                },
            )
        )
    annotation = session.scalar(
        select(AnnotationTaskRecord)
        .where(
            AnnotationTaskRecord.tenant_id == candidate.tenant_id,
            AnnotationTaskRecord.candidate_id == candidate.candidate_id,
        )
        .order_by(AnnotationTaskRecord.created_at.desc())
    )
    if annotation is not None:
        stages.append(
            TimelineStageResponse(
                stage="ANNOTATION",
                status=annotation.status,
                resource_id=annotation.task_id,
                occurred_at=annotation.created_at,
                facts={
                    "schema_version": annotation.schema_version,
                    "required_reviews": annotation.required_reviews,
                    "consistency_status": annotation.consistency_status,
                    "version": annotation.version,
                },
            )
        )
    if not include_dataset_stages:
        return stages
    snapshots = list(
        session.scalars(
            select(DatasetSnapshotRecord).where(
                DatasetSnapshotRecord.tenant_id == candidate.tenant_id
            )
        )
    )
    for snapshot in snapshots:
        if candidate.work_order_id not in snapshot.source_work_order_ids:
            continue
        run = session.scalar(
            select(CurationRunRecord).where(
                CurationRunRecord.tenant_id == candidate.tenant_id,
                CurationRunRecord.run_id == snapshot.run_id,
            )
        )
        if run is not None:
            stages.append(
                TimelineStageResponse(
                    stage="CURATION_RUN",
                    status=run.status,
                    resource_id=run.run_id,
                    occurred_at=run.created_at,
                    facts={"engine": run.engine, "input_count": run.input_count},
                )
            )
        stages.append(
            TimelineStageResponse(
                stage="DATASET_SNAPSHOT",
                status=snapshot.status,
                resource_id=snapshot.snapshot_id,
                occurred_at=snapshot.created_at,
                facts={
                    "row_count": snapshot.row_count,
                    "lineage_status": snapshot.lineage_status,
                    "training_eligible": snapshot.training_eligible,
                },
            )
        )
    return stages


def _eligibility_decision_history(
    session: Session,
    candidate: FeedbackCandidateRecord,
) -> list[EligibilityDecisionResponse]:
    decisions = list(
        session.scalars(
            select(DataEligibilityDecisionRecord)
            .where(
                DataEligibilityDecisionRecord.tenant_id == candidate.tenant_id,
                DataEligibilityDecisionRecord.candidate_id == candidate.candidate_id,
            )
            .order_by(
                DataEligibilityDecisionRecord.created_at,
                DataEligibilityDecisionRecord.decision_id,
            )
        )
    )
    return [
        EligibilityDecisionResponse(
            decision_id=decision.decision_id,
            allow_training=decision.allow_training,
            purpose=decision.purpose,
            consent_basis=decision.consent_basis,
            license_status=decision.license_status,
            retention_until=decision.retention_until,
            policy_version=decision.policy_version,
            candidate_version=decision.candidate_version,
            decided_by_subject_id=decision.decided_by_subject_id,
            created_at=decision.created_at,
        )
        for decision in decisions
    ]


def _dlp_processing_history(
    session: Session,
    candidate: FeedbackCandidateRecord,
) -> list[DlpProcessingHistoryResponse]:
    runs = list(
        session.scalars(
            select(DlpProcessingResultRecord)
            .where(
                DlpProcessingResultRecord.tenant_id == candidate.tenant_id,
                DlpProcessingResultRecord.candidate_id == candidate.candidate_id,
            )
            .order_by(
                DlpProcessingResultRecord.created_at,
                DlpProcessingResultRecord.dlp_result_id,
            )
        )
    )
    return [
        DlpProcessingHistoryResponse(
            dlp_result_id=run.dlp_result_id,
            status=run.status,
            policy_version=run.policy_version,
            source_content_hash=run.source_content_hash,
            content_hash=run.content_hash,
            finding_counts=_dlp_finding_counts(run.findings),
            residual_entity_types=run.residual_entity_types,
            candidate_version=run.candidate_version,
            processed_by_subject_id=run.processed_by_subject_id,
            created_at=run.created_at,
        )
        for run in runs
    ]


def _dlp_finding_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        entity_type = str(finding.get("entity_type") or "UNKNOWN")
        counts[entity_type] = counts.get(entity_type, 0) + 1
    return dict(sorted(counts.items()))


def _annotation_task_history(
    session: Session,
    candidate: FeedbackCandidateRecord,
) -> list[AnnotationTaskHistoryResponse]:
    tasks = list(
        session.scalars(
            select(AnnotationTaskRecord)
            .where(
                AnnotationTaskRecord.tenant_id == candidate.tenant_id,
                AnnotationTaskRecord.candidate_id == candidate.candidate_id,
            )
            .order_by(AnnotationTaskRecord.created_at, AnnotationTaskRecord.task_id)
        )
    )
    task_ids = [task.task_id for task in tasks]
    if not task_ids:
        return []
    revisions = list(
        session.scalars(
            select(AnnotationRevisionRecord)
            .where(
                AnnotationRevisionRecord.tenant_id == candidate.tenant_id,
                AnnotationRevisionRecord.task_id.in_(task_ids),
            )
            .order_by(
                AnnotationRevisionRecord.submitted_at,
                AnnotationRevisionRecord.revision_id,
            )
        )
    )
    reviews = list(
        session.scalars(
            select(AnnotationReviewRecord)
            .where(
                AnnotationReviewRecord.tenant_id == candidate.tenant_id,
                AnnotationReviewRecord.task_id.in_(task_ids),
            )
            .order_by(
                AnnotationReviewRecord.reviewed_at,
                AnnotationReviewRecord.review_id,
            )
        )
    )
    revisions_by_task: dict[str, list[AnnotationRevisionAuditResponse]] = {
        task_id: [] for task_id in task_ids
    }
    for revision in revisions:
        revisions_by_task[revision.task_id].append(
            AnnotationRevisionAuditResponse(
                revision_id=revision.revision_id,
                external_annotation_id=revision.external_annotation_id,
                external_version=revision.external_version,
                reviewer_subject_id=revision.reviewer_subject_id,
                labels_digest=revision.labels_digest,
                task_payload_hash=revision.task_payload_hash,
                submitted_at=revision.submitted_at,
            )
        )
    reviews_by_task: dict[str, list[AnnotationReviewAuditResponse]] = {
        task_id: [] for task_id in task_ids
    }
    for review in reviews:
        reviews_by_task[review.task_id].append(
            AnnotationReviewAuditResponse(
                review_id=review.review_id,
                revision_id=review.revision_id,
                reviewer_subject_id=review.reviewer_subject_id,
                outcome=review.outcome,
                consistency_key=review.consistency_key,
                reviewed_at=review.reviewed_at,
            )
        )
    return [
        AnnotationTaskHistoryResponse(
            task_id=task.task_id,
            dlp_result_id=task.dlp_result_id,
            project_id=task.project_id,
            external_task_id=task.external_task_id,
            external_version=task.external_version,
            payload_hash=task.payload_hash,
            schema_version=task.schema_version,
            risk_level=task.risk_level,
            required_reviews=task.required_reviews,
            status=task.status,
            consistency_status=task.consistency_status,
            created_by_subject_id=task.created_by_subject_id,
            version=task.version,
            created_at=task.created_at,
            revisions=revisions_by_task[task.task_id],
            reviews=reviews_by_task[task.task_id],
        )
        for task in tasks
    ]


def _data_lineage_source_chains(
    session: Session,
    lineage: DataLineageRunRecord,
) -> list[DataLineageSourceChainResponse]:
    work_order_ids = list(dict.fromkeys(lineage.source_work_order_ids))
    if not work_order_ids:
        return []
    candidates = list(
        session.scalars(
            select(FeedbackCandidateRecord)
            .where(
                FeedbackCandidateRecord.tenant_id == lineage.tenant_id,
                FeedbackCandidateRecord.work_order_id.in_(work_order_ids),
            )
            .order_by(FeedbackCandidateRecord.created_at.desc())
        )
    )
    candidates_by_work_order: dict[str, FeedbackCandidateRecord] = {}
    for candidate in candidates:
        candidates_by_work_order.setdefault(candidate.work_order_id, candidate)

    chains: list[DataLineageSourceChainResponse] = []
    for work_order_id in work_order_ids:
        source_candidate = candidates_by_work_order.get(work_order_id)
        if source_candidate is None:
            chains.append(
                DataLineageSourceChainResponse(
                    work_order_id=work_order_id,
                    candidate_id=None,
                    stages=[
                        TimelineStageResponse(
                            stage="WORK_ORDER",
                            status="CHAIN_INCOMPLETE",
                            resource_id=work_order_id,
                        )
                    ],
                )
            )
            continue
        chains.append(
            DataLineageSourceChainResponse(
                work_order_id=work_order_id,
                candidate_id=source_candidate.candidate_id,
                stages=_complete_source_lineage_stages(
                    source_candidate,
                    _candidate_timeline(
                        session,
                        source_candidate,
                        include_dataset_stages=False,
                    ),
                ),
            )
        )
    return chains


def _complete_source_lineage_stages(
    candidate: FeedbackCandidateRecord,
    recorded_stages: list[TimelineStageResponse],
) -> list[TimelineStageResponse]:
    by_stage = {stage.stage: stage for stage in recorded_stages}
    expected_resources = {
        "WORK_ORDER": candidate.work_order_id,
        "OUTBOX_EVENT": candidate.source_event_id,
        "INBOX_EVENT": candidate.source_event_id,
        "FEEDBACK_CANDIDATE": candidate.candidate_id,
        "ELIGIBILITY": candidate.candidate_id,
        "DLP": candidate.candidate_id,
        "ANNOTATION": candidate.candidate_id,
    }
    return [
        by_stage.get(stage)
        or TimelineStageResponse(
            stage=stage,
            status="NOT_RECORDED",
            resource_id=resource_id,
            facts={"reason": "governance_record_not_available"},
        )
        for stage, resource_id in expected_resources.items()
    ]


def _data_lineage_pipeline_stages(
    session: Session,
    lineage: DataLineageRunRecord,
) -> list[TimelineStageResponse]:
    stages: list[TimelineStageResponse] = []
    run = session.scalar(
        select(CurationRunRecord).where(
            CurationRunRecord.tenant_id == lineage.tenant_id,
            CurationRunRecord.run_id == lineage.run_id,
        )
    )
    if run is not None:
        stages.append(
            TimelineStageResponse(
                stage="CURATION_RUN",
                status=run.status,
                resource_id=run.run_id,
                occurred_at=run.created_at,
                facts={
                    "engine": run.engine,
                    "input_count": run.input_count,
                    "contract_version": run.contract_version,
                    "input_manifest_hash": run.input_manifest_hash,
                },
            )
        )
    snapshot = session.scalar(
        select(DatasetSnapshotRecord).where(
            DatasetSnapshotRecord.tenant_id == lineage.tenant_id,
            DatasetSnapshotRecord.snapshot_id == lineage.snapshot_id,
        )
    )
    if snapshot is not None:
        quality_artifacts = list(
            session.scalars(
                select(DatasetArtifactRecord).where(
                    DatasetArtifactRecord.tenant_id == snapshot.tenant_id,
                    DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
                    DatasetArtifactRecord.kind == "quality_report",
                )
            )
        )
        training_blockers = dataset_training_blockers(
            snapshot,
            lineage,
            quality_artifacts,
        )
        stages.append(
            TimelineStageResponse(
                stage="DATASET_SNAPSHOT",
                status=snapshot.status,
                resource_id=snapshot.snapshot_id,
                occurred_at=snapshot.created_at,
                facts={
                    "row_count": snapshot.row_count,
                    "lineage_status": snapshot.lineage_status,
                    "training_eligible": not training_blockers,
                    "training_blocker_count": len(training_blockers),
                    "manifest_hash": snapshot.manifest_hash,
                },
            )
        )
    stages.append(
        TimelineStageResponse(
            stage="OPENLINEAGE",
            status=lineage.status,
            resource_id=lineage.lineage_id,
            occurred_at=lineage.created_at,
            facts={
                "openlineage_run_id": lineage.openlineage_run_id,
                "job_namespace": lineage.job_namespace,
                "job_name": lineage.job_name,
                "source_work_order_count": len(lineage.source_work_order_ids),
            },
        )
    )
    return stages


def _curation_run_response(
    session: Session,
    run: CurationRunRecord,
) -> CurationRunResponse:
    snapshot = session.scalar(
        select(DatasetSnapshotRecord).where(
            DatasetSnapshotRecord.tenant_id == run.tenant_id,
            DatasetSnapshotRecord.run_id == run.run_id,
        )
    )
    return CurationRunResponse(
        run_id=run.run_id,
        status=cast(CurationRunStatus, run.status),
        engine=run.engine,
        window_start=run.window_start,
        window_end=run.window_end,
        contract_version=run.contract_version,
        code_version=run.code_version,
        config_hash=run.config_hash,
        input_manifest_hash=run.input_manifest_hash,
        input_count=run.input_count,
        exclusion_report=run.exclusion_report,
        failure_reason=run.failure_reason,
        snapshot_id=snapshot.snapshot_id if snapshot else None,
        created_at=run.created_at,
    )


def _dataset_snapshot_response(
    session: Session,
    snapshot: DatasetSnapshotRecord,
    *,
    identity: IdentityContext,
    authorizer: Authorizer,
    include_artifacts: bool,
) -> DatasetSnapshotResponse:
    lineage = session.scalar(
        select(DataLineageRunRecord).where(
            DataLineageRunRecord.tenant_id == snapshot.tenant_id,
            DataLineageRunRecord.snapshot_id == snapshot.snapshot_id,
        )
    )
    artifacts: list[DatasetArtifactRecord] = []
    if include_artifacts:
        artifacts = list(
            session.scalars(
                select(DatasetArtifactRecord)
                .where(
                    DatasetArtifactRecord.tenant_id == snapshot.tenant_id,
                    DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
                )
                .order_by(DatasetArtifactRecord.kind, DatasetArtifactRecord.split)
            )
        )
        quality_artifacts = [item for item in artifacts if item.kind == "quality_report"]
    else:
        quality_artifacts = list(
            session.scalars(
                select(DatasetArtifactRecord).where(
                    DatasetArtifactRecord.tenant_id == snapshot.tenant_id,
                    DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
                    DatasetArtifactRecord.kind == "quality_report",
                )
            )
        )
    training_blockers = list(dataset_training_blockers(snapshot, lineage, quality_artifacts))
    legal_actions: list[str] = []
    if not training_blockers:
        resource = ResourceContext(
            tenant_id=identity.tenant_id,
            resource_id=snapshot.snapshot_id,
        )
        if authorizer.decide(identity, Action.DOWNLOAD_DATASET_MANIFEST, resource).allowed:
            legal_actions.append("DOWNLOAD_MANIFEST")
        if authorizer.decide(identity, Action.CREATE_TRAINING_EXPERIMENT, resource).allowed:
            legal_actions.append("CREATE_EXPERIMENT")
    return DatasetSnapshotResponse(
        snapshot_id=snapshot.snapshot_id,
        run_id=snapshot.run_id,
        status=snapshot.status,
        contract_version=snapshot.contract_version,
        input_manifest_hash=snapshot.input_manifest_hash,
        manifest_hash=snapshot.manifest_hash,
        row_count=snapshot.row_count,
        split_counts=snapshot.split_counts,
        source_work_order_ids=snapshot.source_work_order_ids,
        lineage_status=snapshot.lineage_status,
        lineage_id=lineage.lineage_id if lineage else None,
        training_eligible=not training_blockers,
        training_blockers=training_blockers,
        base_snapshot_id=snapshot.base_snapshot_id,
        augmentation_contract_version=snapshot.augmentation_contract_version,
        sample_origin_counts=snapshot.sample_origin_counts,
        synthetic_split_counts=snapshot.synthetic_split_counts,
        created_at=snapshot.created_at,
        artifacts=[
            DatasetArtifactResponse(
                artifact_id=item.artifact_id,
                kind=item.kind,
                split=item.split,
                content_hash=item.content_hash,
                size_bytes=item.size_bytes,
                row_count=item.row_count,
                schema_hash=item.schema_hash,
            )
            for item in artifacts
        ],
        legal_actions=legal_actions,
    )


def _dataset_training_blocker_filter(
    blocker: DatasetTrainingBlockerCode,
) -> ColumnElement[bool]:
    snapshot = DatasetSnapshotRecord
    confirmed_lineage_exists = _confirmed_lineage_exists_filter()
    quality_report_registered = _quality_report_registered_filter()
    if blocker == "snapshot_not_candidate":
        return snapshot.status != "CANDIDATE"
    if blocker == "lineage_not_confirmed":
        return or_(snapshot.lineage_status != "CONFIRMED", ~confirmed_lineage_exists)
    if blocker == "manifest_not_published":
        return or_(snapshot.manifest_key.is_(None), snapshot.manifest_hash.is_(None))
    if blocker == "quality_report_missing":
        return ~quality_report_registered
    if blocker == "dataset_empty":
        return snapshot.row_count <= 0
    return and_(
        snapshot.training_eligible.is_(False),
        snapshot.status == "CANDIDATE",
        snapshot.lineage_status == "CONFIRMED",
        confirmed_lineage_exists,
        snapshot.manifest_key.is_not(None),
        snapshot.manifest_hash.is_not(None),
        quality_report_registered,
        snapshot.row_count > 0,
    )


def _dataset_training_eligible_filter(eligible: bool) -> ColumnElement[bool]:
    snapshot = DatasetSnapshotRecord
    gate_passed = and_(
        snapshot.training_eligible.is_(True),
        snapshot.status == "CANDIDATE",
        snapshot.lineage_status == "CONFIRMED",
        _confirmed_lineage_exists_filter(),
        snapshot.manifest_key.is_not(None),
        snapshot.manifest_hash.is_not(None),
        _quality_report_registered_filter(),
        snapshot.row_count > 0,
    )
    return gate_passed if eligible else ~gate_passed


def _confirmed_lineage_exists_filter() -> ColumnElement[bool]:
    snapshot = DatasetSnapshotRecord
    return (
        select(DataLineageRunRecord.lineage_id)
        .where(
            DataLineageRunRecord.tenant_id == snapshot.tenant_id,
            DataLineageRunRecord.snapshot_id == snapshot.snapshot_id,
            DataLineageRunRecord.status == "CONFIRMED",
        )
        .exists()
    )


def _quality_report_registered_filter() -> ColumnElement[bool]:
    snapshot = DatasetSnapshotRecord
    quality_schema_digest = sha256(CONTRACT_VERSION.encode()).hexdigest()
    quality_artifact_count = (
        select(func.count())
        .select_from(DatasetArtifactRecord)
        .where(
            DatasetArtifactRecord.tenant_id == snapshot.tenant_id,
            DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
            DatasetArtifactRecord.kind == "quality_report",
        )
        .scalar_subquery()
    )
    registered_quality_exists = (
        select(DatasetArtifactRecord.artifact_id)
        .where(
            DatasetArtifactRecord.tenant_id == snapshot.tenant_id,
            DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
            DatasetArtifactRecord.kind == "quality_report",
            DatasetArtifactRecord.split.is_(None),
            DatasetArtifactRecord.object_key == snapshot.quality_report_key,
            DatasetArtifactRecord.size_bytes > 0,
            DatasetArtifactRecord.row_count == snapshot.row_count,
            DatasetArtifactRecord.schema_hash.in_(
                (quality_schema_digest, f"sha256:{quality_schema_digest}")
            ),
            or_(
                func.length(DatasetArtifactRecord.content_hash) == 64,
                and_(
                    func.length(DatasetArtifactRecord.content_hash) == 71,
                    DatasetArtifactRecord.content_hash.like("sha256:%"),
                ),
            ),
        )
        .exists()
    )
    return and_(
        snapshot.quality_report_key.is_not(None),
        quality_artifact_count == 1,
        registered_quality_exists,
    )


def _digest_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(payload.encode()).hexdigest()


def _candidate_response(view: CandidateGovernanceView) -> CandidateGovernanceResponse:
    legal_actions = ["DECIDE_ELIGIBILITY"]
    if view.status == "ELIGIBLE":
        legal_actions.append("RUN_DLP")
    return CandidateGovernanceResponse(
        candidate_id=view.candidate_id,
        status=view.status,
        eligibility_status=view.eligibility_status,
        allow_training=view.allow_training,
        consent_basis=view.consent_basis,
        license_status=view.license_status,
        retention_until=view.retention_until,
        governance_blockers=list(view.governance_blockers),
        version=view.version,
        legal_actions=legal_actions,
    )


def _dlp_response(view: DlpRunView) -> DlpRunResponse:
    return DlpRunResponse(
        dlp_result_id=view.dlp_result_id,
        candidate_id=view.candidate_id,
        status=view.status,
        policy_version=view.policy_version,
        source_content_hash=view.source_content_hash,
        content_hash=view.content_hash,
        redacted_content=view.redacted_content,
        findings=view.findings,
        residual_entity_types=view.residual_entity_types,
        candidate_version=view.candidate_version,
    )


def _task_envelope(view: AnnotationTaskView, request: Request) -> AnnotationTaskEnvelope:
    legal_actions = ["SYNC"] if view.status in {"PENDING_REVIEW", "REVIEW_REQUIRED"} else []
    return AnnotationTaskEnvelope(
        data=AnnotationTaskResponse(
            task_id=view.task_id,
            candidate_id=view.candidate_id,
            status=view.status,
            risk_level=view.risk_level,
            required_reviews=view.required_reviews,
            completed_reviews=view.completed_reviews,
            external_task_id=view.external_task_id,
            external_version=view.external_version,
            version=view.version,
            candidate_version=view.candidate_version,
            requires_arbitration=view.requires_arbitration,
            legal_actions=legal_actions,
        ),
        meta={"request_id": request.state.request_id},
    )


def _version(value: str) -> int:
    try:
        version = int(value.strip().strip('"'))
    except ValueError as exc:
        raise AppError(
            400,
            "invalid_version_precondition",
            "validation",
            "If-Match is invalid",
        ) from exc
    if version < 1:
        raise AppError(400, "invalid_version_precondition", "validation", "If-Match is invalid")
    return version


def _translate(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(403, "authorization_denied", "authorization", "Action is not allowed")
    if isinstance(exc, GovernanceNotVisible):
        return AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        )
    if isinstance(exc, GovernanceConflict):
        return AppError(
            409,
            exc.reason,
            "conflict",
            "Governance state or version changed",
            details={"current_version": exc.current_version},
        )
    if isinstance(exc, GovernanceGateDenied):
        return AppError(
            409,
            exc.reason,
            "governance_gate",
            "Data governance gate rejected the operation",
        )
    raise TypeError("unsupported governance error")


def _label_studio_unavailable() -> AppError:
    return AppError(
        503,
        "label_studio_unavailable",
        "dependency",
        "Label Studio unavailable",
        retryable=True,
    )
