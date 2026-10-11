"""Governed experiment registry, MLflow state transitions and independent evaluation API."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_dataset_store,
    get_experiment_tracker,
    get_identity,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.experiments.mlflow import ExperimentTracker, TrackingUnavailable
from industrial_ops_agent.experiments.service import (
    ArtifactInput,
    EvaluationGovernanceService,
    EvaluationJobPlan,
    EvaluationJobService,
    ExperimentConflict,
    ExperimentGateDenied,
    ExperimentNotVisible,
    ExperimentPlan,
    ExperimentRegistryService,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    EvaluationArtifactRecord,
    EvaluationPolicyRecord,
    EvaluationSuiteRecord,
    ModelEvaluationJobRecord,
    ModelEvaluationRunRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)

router = APIRouter(tags=["training-evaluation"])

ExperimentMethod = Literal[
    "BASELINE",
    "LORA",
    "QLORA",
    "DPO",
    "GRPO",
    "PPO",
    "EMBEDDING",
    "RERANKER",
    "VLM",
    "ASR",
    "TTS",
    "QUANTIZATION",
    "TIMESERIES_TRANSFORMER",
    "TIMESERIES_RULE_BASELINE",
    "RUL_TRANSFORMER",
    "RUL_EMPIRICAL_BASELINE",
]
ExperimentStatus = Literal[
    "PLANNED",
    "TRACKING_PENDING",
    "TRACKING_FAILED",
    "RUNNING",
    "COMPLETION_PENDING",
    "COMPLETION_FAILED",
    "COMPLETED",
    "FAILED",
    "CLOSED_NO_GAIN",
]
EvaluationJobStatus = Literal["PLANNED", "RUNNING", "COMPLETED", "FAILED"]
EvaluationDecision = Literal["CANDIDATE", "REJECTED", "SMOKE_PASSED", "NO_GAIN"]


class ExperimentPlanBody(BaseModel):
    comparison_group_id: str = Field(min_length=1, max_length=128)
    method: ExperimentMethod
    task_type: str = Field(min_length=1, max_length=64)
    dataset_snapshot_id: str = Field(min_length=1, max_length=128)
    base_model_id: str = Field(min_length=1, max_length=255)
    base_model_digest: str = Field(min_length=1, max_length=128)
    tokenizer_digest: str = Field(min_length=1, max_length=128)
    chat_template_digest: str = Field(min_length=1, max_length=128)
    git_commit: str = Field(min_length=1, max_length=128)
    container_digest: str = Field(min_length=1, max_length=255)
    training_config: dict[str, Any]
    distributed_profile: dict[str, Any]
    random_seeds: list[int] = Field(min_length=1)
    hardware_topology: dict[str, Any]
    mlflow_experiment_name: str = Field(min_length=1, max_length=255)
    license_status: Literal["APPROVED", "REJECTED", "UNKNOWN"]


class TrainingArtifactBody(BaseModel):
    kind: str = Field(min_length=1, max_length=32)
    object_key: str = Field(min_length=1, max_length=1024)
    content_hash: str = Field(min_length=1, max_length=128)
    size_bytes: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentCompletionBody(BaseModel):
    metrics: dict[str, float] = Field(min_length=1)
    cost_summary: dict[str, Any]
    artifacts: list[TrainingArtifactBody] = Field(min_length=1)


class TrainingArtifactResponse(BaseModel):
    artifact_id: str
    kind: str
    object_key: str
    content_hash: str
    size_bytes: int
    metadata: dict[str, Any]


class TrainingExperimentResponse(BaseModel):
    experiment_id: str
    comparison_group_id: str
    method: str
    task_type: str
    status: str
    dataset_snapshot_id: str
    dataset_manifest_hash: str
    base_model_id: str
    base_model_digest: str
    tokenizer_digest: str
    chat_template_digest: str
    git_commit: str
    container_digest: str
    training_config: dict[str, Any]
    config_hash: str
    distributed_profile: dict[str, Any]
    random_seeds: list[int]
    hardware_topology: dict[str, Any]
    mlflow_experiment_name: str
    mlflow_run_id: str | None
    metrics: dict[str, float]
    cost_summary: dict[str, Any]
    license_status: str
    created_by_subject_id: str
    completed_at: datetime | None
    failure_reason: str | None
    version: int
    created_at: datetime
    artifacts: list[TrainingArtifactResponse] = Field(default_factory=list)
    legal_actions: list[str] = Field(default_factory=list)


class ExperimentEnvelope(BaseModel):
    data: TrainingExperimentResponse
    meta: dict[str, str]


class ExperimentListEnvelope(BaseModel):
    data: list[TrainingExperimentResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


class EvaluationSuiteBody(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    version: str = Field(min_length=1, max_length=64)
    tier: Literal["SMOKE", "GOLD", "SIMULATION_REFERENCE"]
    source_snapshot_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(min_length=1, max_length=128)
    sample_count: int = Field(gt=0)
    slice_counts: dict[str, int] = Field(min_length=1)


class EvaluationSuiteResponse(BaseModel):
    suite_id: str
    name: str
    version: str
    tier: str
    source_snapshot_id: str
    purpose: str
    status: str
    manifest_hash: str
    sample_count: int
    slice_counts: dict[str, int]
    created_by_subject_id: str
    created_at: datetime


class EvaluationSuiteEnvelope(BaseModel):
    data: EvaluationSuiteResponse
    meta: dict[str, str]


class EvaluationSuiteListEnvelope(BaseModel):
    data: list[EvaluationSuiteResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


class EvaluationPolicyBody(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    version: str = Field(min_length=1, max_length=64)
    primary_metric: str = Field(min_length=1, max_length=128)
    hard_gates: dict[str, bool] = Field(min_length=1)
    thresholds: dict[str, float] = Field(min_length=1)


class EvaluationPolicyResponse(BaseModel):
    policy_id: str
    name: str
    version: str
    status: str
    primary_metric: str
    hard_gates: dict[str, Any]
    thresholds: dict[str, float]
    policy_hash: str
    created_by_subject_id: str
    created_at: datetime


class EvaluationPolicyEnvelope(BaseModel):
    data: EvaluationPolicyResponse
    meta: dict[str, str]


class EvaluationPolicyListEnvelope(BaseModel):
    data: list[EvaluationPolicyResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


class EvaluationJobBody(BaseModel):
    candidate_experiment_id: str = Field(min_length=1, max_length=128)
    baseline_experiment_id: str = Field(min_length=1, max_length=128)
    suite_id: str = Field(min_length=1, max_length=128)
    policy_id: str = Field(min_length=1, max_length=128)
    target_profile: Literal[
        "MODEL_COMPONENT",
        "EDGE_MODEL_COMPONENT",
        "AGENT_RUNTIME",
        "RETRIEVAL_COMPONENT",
        "VLM_COMPONENT",
        "ASR_COMPONENT",
        "TTS_COMPONENT",
        "TIMESERIES_COMPONENT",
        "RUL_COMPONENT",
        "PPO_RESEARCH_SAFETY",
    ]
    runner_git_commit: str = Field(min_length=1, max_length=128)
    container_digest: str = Field(min_length=1, max_length=255)
    runner_config: dict[str, Any]


class EvaluationJobResponse(BaseModel):
    job_id: str
    candidate_experiment_id: str
    baseline_experiment_id: str
    suite_id: str
    policy_id: str
    status: str
    target_profile: str
    runner_git_commit: str
    container_digest: str
    runner_config: dict[str, Any]
    config_hash: str
    result_evaluation_id: str | None
    created_by_subject_id: str
    started_at: datetime | None
    completed_at: datetime | None
    failure_reason: str | None
    version: int
    created_at: datetime


class EvaluationJobEnvelope(BaseModel):
    data: EvaluationJobResponse
    meta: dict[str, str]


class EvaluationJobListEnvelope(BaseModel):
    data: list[EvaluationJobResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


class ModelEvaluationResponse(BaseModel):
    evaluation_id: str
    candidate_experiment_id: str
    baseline_experiment_id: str
    suite_id: str
    policy_id: str
    status: str
    decision: str
    primary_metric: str
    candidate_score: float
    baseline_score: float
    quality_delta: float
    ci_low: float
    ci_high: float
    latency_improvement: float
    cost_improvement: float
    hard_gate_results: dict[str, bool]
    slice_metrics: dict[str, Any]
    comparison_kind: str
    comparison_context: dict[str, Any]
    report_hash: str
    triggered_by_subject_id: str
    completed_at: datetime
    failure_reason: str | None


class ModelEvaluationEnvelope(BaseModel):
    data: ModelEvaluationResponse
    meta: dict[str, str]


class ModelEvaluationListEnvelope(BaseModel):
    data: list[ModelEvaluationResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


@router.post("/experiments", response_model=ExperimentEnvelope, status_code=201)
def create_experiment(
    body: ExperimentPlanBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    tracker: Annotated[ExperimentTracker, Depends(get_experiment_tracker)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)],
) -> ExperimentEnvelope:
    try:
        record = ExperimentRegistryService(database, authorizer, tracker).create(
            identity,
            ExperimentPlan(**body.model_dump()),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        ExperimentNotVisible,
        ExperimentConflict,
        ExperimentGateDenied,
    ) as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc
    with database.transaction(identity.tenant_context) as session:
        data = _experiment_response(session, record, identity, authorizer)
    return ExperimentEnvelope(data=data, meta={"request_id": request.state.request_id})


@router.get("/experiments", response_model=ExperimentListEnvelope)
def list_experiments(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    experiment_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ] = None,
    dataset_snapshot_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ] = None,
    method: Annotated[ExperimentMethod | None, Query()] = None,
    status: Annotated[ExperimentStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ExperimentListEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_TRAINING_EXPERIMENT,
        "training-experiments",
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        filters = [TrainingExperimentRecord.tenant_id == identity.tenant_id]
        if experiment_id is not None:
            filters.append(TrainingExperimentRecord.experiment_id == experiment_id)
        if dataset_snapshot_id is not None:
            filters.append(TrainingExperimentRecord.dataset_snapshot_id == dataset_snapshot_id)
        if method is not None:
            filters.append(TrainingExperimentRecord.method == method)
        if status is not None:
            filters.append(TrainingExperimentRecord.status == status)
        total = int(
            session.scalar(
                select(func.count()).select_from(TrainingExperimentRecord).where(*filters)
            )
            or 0
        )
        records = list(
            session.scalars(
                select(TrainingExperimentRecord)
                .where(*filters)
                .order_by(
                    TrainingExperimentRecord.created_at.desc(),
                    TrainingExperimentRecord.experiment_id,
                )
                .offset(offset)
                .limit(limit)
            )
        )
        data = [
            _experiment_response(session, record, identity, authorizer, include_artifacts=False)
            for record in records
        ]
    return ExperimentListEnvelope(
        data=data,
        meta={
            "request_id": request.state.request_id,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        legal_actions=_collection_actions(identity, authorizer),
    )


@router.get("/experiments/{experiment_id}", response_model=ExperimentEnvelope)
def get_experiment(
    experiment_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ExperimentEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_TRAINING_EXPERIMENT,
        experiment_id,
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        record = _experiment_or_404(session, identity.tenant_id, experiment_id)
        data = _experiment_response(session, record, identity, authorizer)
    return ExperimentEnvelope(data=data, meta={"request_id": request.state.request_id})


@router.post("/experiments/{experiment_id}/start", response_model=ExperimentEnvelope)
def start_experiment(
    experiment_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    tracker: Annotated[ExperimentTracker, Depends(get_experiment_tracker)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ExperimentEnvelope:
    try:
        record = ExperimentRegistryService(database, authorizer, tracker).start(
            identity,
            experiment_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except TrackingUnavailable as exc:
        raise AppError(503, "mlflow_unavailable", "dependency", "MLflow unavailable", True) from exc
    except (
        AuthorizationDenied,
        ExperimentNotVisible,
        ExperimentConflict,
        ExperimentGateDenied,
    ) as exc:
        raise _translate(exc) from exc
    with database.transaction(identity.tenant_context) as session:
        data = _experiment_response(session, record, identity, authorizer)
    return ExperimentEnvelope(data=data, meta={"request_id": request.state.request_id})


@router.post("/experiments/{experiment_id}/results", response_model=ExperimentEnvelope)
def complete_experiment(
    experiment_id: str,
    body: ExperimentCompletionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    tracker: Annotated[ExperimentTracker, Depends(get_experiment_tracker)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ExperimentEnvelope:
    try:
        record = ExperimentRegistryService(database, authorizer, tracker).complete(
            identity,
            experiment_id,
            expected_version=_version(if_match),
            metrics=body.metrics,
            cost_summary=body.cost_summary,
            artifacts=[ArtifactInput(**item.model_dump()) for item in body.artifacts],
            request_id=request.state.request_id,
        )
    except TrackingUnavailable as exc:
        raise AppError(503, "mlflow_unavailable", "dependency", "MLflow unavailable", True) from exc
    except (
        AuthorizationDenied,
        ExperimentNotVisible,
        ExperimentConflict,
        ExperimentGateDenied,
    ) as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc
    with database.transaction(identity.tenant_context) as session:
        data = _experiment_response(session, record, identity, authorizer)
    return ExperimentEnvelope(data=data, meta={"request_id": request.state.request_id})


@router.post("/evaluation-suites", response_model=EvaluationSuiteEnvelope, status_code=201)
def register_evaluation_suite(
    body: EvaluationSuiteBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> EvaluationSuiteEnvelope:
    try:
        record = EvaluationGovernanceService(database, authorizer).register_suite(
            identity,
            **body.model_dump(),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        ExperimentNotVisible,
        ExperimentConflict,
        ExperimentGateDenied,
    ) as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc
    return EvaluationSuiteEnvelope(
        data=_suite_response(record), meta={"request_id": request.state.request_id}
    )


@router.get("/evaluation-suites", response_model=EvaluationSuiteListEnvelope)
def list_evaluation_suites(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> EvaluationSuiteListEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_EVALUATION_SUITE,
        "evaluation-suites",
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        records = list(
            session.scalars(
                select(EvaluationSuiteRecord)
                .where(EvaluationSuiteRecord.tenant_id == identity.tenant_id)
                .order_by(EvaluationSuiteRecord.created_at.desc())
            )
        )
        data = [_suite_response(record) for record in records]
    return EvaluationSuiteListEnvelope(
        data=data,
        meta={"request_id": request.state.request_id, "total": len(data)},
        legal_actions=_legal_collection_action(
            identity, authorizer, Action.MANAGE_EVALUATION_SUITE, "REGISTER_EVALUATION_SUITE"
        ),
    )


@router.post("/evaluation-policies", response_model=EvaluationPolicyEnvelope, status_code=201)
def register_evaluation_policy(
    body: EvaluationPolicyBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> EvaluationPolicyEnvelope:
    try:
        record = EvaluationGovernanceService(database, authorizer).register_policy(
            identity,
            **body.model_dump(),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        ExperimentNotVisible,
        ExperimentConflict,
        ExperimentGateDenied,
    ) as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc
    return EvaluationPolicyEnvelope(
        data=_policy_response(record), meta={"request_id": request.state.request_id}
    )


@router.get("/evaluation-policies", response_model=EvaluationPolicyListEnvelope)
def list_evaluation_policies(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> EvaluationPolicyListEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_EVALUATION_POLICY,
        "evaluation-policies",
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        records = list(
            session.scalars(
                select(EvaluationPolicyRecord)
                .where(EvaluationPolicyRecord.tenant_id == identity.tenant_id)
                .order_by(EvaluationPolicyRecord.created_at.desc())
            )
        )
        data = [_policy_response(record) for record in records]
    return EvaluationPolicyListEnvelope(
        data=data,
        meta={"request_id": request.state.request_id, "total": len(data)},
        legal_actions=_legal_collection_action(
            identity, authorizer, Action.MANAGE_EVALUATION_POLICY, "REGISTER_EVALUATION_POLICY"
        ),
    )


@router.post("/model-evaluation-jobs", response_model=EvaluationJobEnvelope, status_code=201)
def create_model_evaluation_job(
    body: EvaluationJobBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)],
) -> EvaluationJobEnvelope:
    try:
        record = EvaluationJobService(database, authorizer).create(
            identity,
            EvaluationJobPlan(**body.model_dump()),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        ExperimentNotVisible,
        ExperimentConflict,
        ExperimentGateDenied,
    ) as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc
    return EvaluationJobEnvelope(
        data=_evaluation_job_response(record), meta={"request_id": request.state.request_id}
    )


@router.get("/model-evaluation-jobs", response_model=EvaluationJobListEnvelope)
def list_model_evaluation_jobs(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    job_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ] = None,
    candidate_experiment_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ] = None,
    baseline_experiment_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ] = None,
    status: Annotated[EvaluationJobStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> EvaluationJobListEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_MODEL_EVALUATION,
        "model-evaluation-jobs",
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        filters = [ModelEvaluationJobRecord.tenant_id == identity.tenant_id]
        if job_id is not None:
            filters.append(ModelEvaluationJobRecord.job_id == job_id)
        if candidate_experiment_id is not None:
            filters.append(
                ModelEvaluationJobRecord.candidate_experiment_id == candidate_experiment_id
            )
        if baseline_experiment_id is not None:
            filters.append(
                ModelEvaluationJobRecord.baseline_experiment_id == baseline_experiment_id
            )
        if status is not None:
            filters.append(ModelEvaluationJobRecord.status == status)
        total = int(
            session.scalar(
                select(func.count()).select_from(ModelEvaluationJobRecord).where(*filters)
            )
            or 0
        )
        records = list(
            session.scalars(
                select(ModelEvaluationJobRecord)
                .where(*filters)
                .order_by(
                    ModelEvaluationJobRecord.created_at.desc(),
                    ModelEvaluationJobRecord.job_id,
                )
                .offset(offset)
                .limit(limit)
            )
        )
    return EvaluationJobListEnvelope(
        data=[_evaluation_job_response(record) for record in records],
        meta={
            "request_id": request.state.request_id,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        legal_actions=_legal_collection_action(
            identity, authorizer, Action.RUN_MODEL_EVALUATION, "CREATE_EVALUATION_JOB"
        ),
    )


@router.get("/model-evaluations", response_model=ModelEvaluationListEnvelope)
def list_model_evaluations(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    evaluation_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ] = None,
    candidate_experiment_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ] = None,
    baseline_experiment_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    ] = None,
    decision: Annotated[EvaluationDecision | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ModelEvaluationListEnvelope:
    _authorize(
        authorizer,
        identity,
        Action.READ_MODEL_EVALUATION,
        "model-evaluations",
        request.state.request_id,
    )
    with database.transaction(identity.tenant_context) as session:
        filters = [ModelEvaluationRunRecord.tenant_id == identity.tenant_id]
        if evaluation_id is not None:
            filters.append(ModelEvaluationRunRecord.evaluation_id == evaluation_id)
        if candidate_experiment_id is not None:
            filters.append(
                ModelEvaluationRunRecord.candidate_experiment_id == candidate_experiment_id
            )
        if baseline_experiment_id is not None:
            filters.append(
                ModelEvaluationRunRecord.baseline_experiment_id == baseline_experiment_id
            )
        if decision is not None:
            filters.append(ModelEvaluationRunRecord.decision == decision)
        total = int(
            session.scalar(
                select(func.count()).select_from(ModelEvaluationRunRecord).where(*filters)
            )
            or 0
        )
        records = list(
            session.scalars(
                select(ModelEvaluationRunRecord)
                .where(*filters)
                .order_by(
                    ModelEvaluationRunRecord.completed_at.desc(),
                    ModelEvaluationRunRecord.evaluation_id,
                )
                .offset(offset)
                .limit(limit)
            )
        )
        data = [_evaluation_response(record) for record in records]
    return ModelEvaluationListEnvelope(
        data=data,
        meta={
            "request_id": request.state.request_id,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        legal_actions=[],
    )


@router.get("/model-evaluations/{evaluation_id}", response_model=ModelEvaluationEnvelope)
def get_model_evaluation(
    evaluation_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ModelEvaluationEnvelope:
    _authorize(
        authorizer, identity, Action.READ_MODEL_EVALUATION, evaluation_id, request.state.request_id
    )
    with database.transaction(identity.tenant_context) as session:
        record = session.scalar(
            select(ModelEvaluationRunRecord).where(
                ModelEvaluationRunRecord.tenant_id == identity.tenant_id,
                ModelEvaluationRunRecord.evaluation_id == evaluation_id,
            )
        )
        if record is None:
            raise _translate(ExperimentNotVisible())
        data = _evaluation_response(record)
    return ModelEvaluationEnvelope(data=data, meta={"request_id": request.state.request_id})


@router.get(
    "/model-evaluations/{evaluation_id}/evidence",
    response_class=Response,
    responses={200: {"content": {"application/json": {}}}},
)
def get_model_evaluation_evidence(
    evaluation_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    store: Annotated[DatasetStore, Depends(get_dataset_store)],
) -> Response:
    """Reauthorize and integrity-check immutable case evidence before download."""

    _authorize(
        authorizer, identity, Action.READ_MODEL_EVALUATION, evaluation_id, request.state.request_id
    )
    with database.transaction(identity.tenant_context) as session:
        evaluation = session.scalar(
            select(ModelEvaluationRunRecord.evaluation_id).where(
                ModelEvaluationRunRecord.tenant_id == identity.tenant_id,
                ModelEvaluationRunRecord.evaluation_id == evaluation_id,
            )
        )
        if evaluation is None:
            raise _translate(ExperimentNotVisible())
        artifact = session.scalar(
            select(EvaluationArtifactRecord).where(
                EvaluationArtifactRecord.tenant_id == identity.tenant_id,
                EvaluationArtifactRecord.evaluation_id == evaluation_id,
                EvaluationArtifactRecord.kind == "case_evidence",
            )
        )
        if artifact is None:
            raise AppError(
                404,
                "evaluation_evidence_not_found",
                "not_found",
                "Evaluation evidence is not available",
            )
        object_key = artifact.object_key
        content_hash = artifact.content_hash
        size_bytes = artifact.size_bytes

    try:
        content = store.get_bytes(object_key)
    except (FileNotFoundError, OSError) as exc:
        raise AppError(
            503,
            "evaluation_evidence_unavailable",
            "dependency",
            "Evaluation evidence is unavailable",
            True,
        ) from exc
    if len(content) != size_bytes or sha256(content).hexdigest() != content_hash:
        raise AppError(
            409,
            "evaluation_evidence_integrity_failed",
            "integrity",
            "Evaluation evidence failed integrity verification",
        )
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "ETag": f'"{content_hash}"',
            "Content-Disposition": (
                f'attachment; filename="evaluation-{evaluation_id}-evidence.json"'
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


def _experiment_response(
    session: Session,
    record: TrainingExperimentRecord,
    identity: IdentityContext,
    authorizer: Authorizer,
    *,
    include_artifacts: bool = True,
) -> TrainingExperimentResponse:
    artifacts: list[TrainingArtifactResponse] = []
    if include_artifacts:
        rows = session.scalars(
            select(TrainingArtifactRecord).where(
                TrainingArtifactRecord.tenant_id == record.tenant_id,
                TrainingArtifactRecord.experiment_id == record.experiment_id,
            )
        )
        artifacts = [
            TrainingArtifactResponse(
                artifact_id=item.artifact_id,
                kind=item.kind,
                object_key=item.object_key,
                content_hash=item.content_hash,
                size_bytes=item.size_bytes,
                metadata=item.metadata_json,
            )
            for item in rows
        ]
    legal_actions: list[str] = []
    if (
        record.method not in {"TIMESERIES_RULE_BASELINE", "RUL_EMPIRICAL_BASELINE"}
        and record.status in {"PLANNED", "TRACKING_FAILED"}
        and _allowed(identity, authorizer, Action.START_TRAINING_EXPERIMENT, record.experiment_id)
    ):
        legal_actions.append("START_EXPERIMENT")
    if record.status in {"RUNNING", "COMPLETION_FAILED"} and _allowed(
        identity, authorizer, Action.COMPLETE_TRAINING_EXPERIMENT, record.experiment_id
    ):
        legal_actions.append("COMPLETE_EXPERIMENT")
    return TrainingExperimentResponse(
        experiment_id=record.experiment_id,
        comparison_group_id=record.comparison_group_id,
        method=record.method,
        task_type=record.task_type,
        status=record.status,
        dataset_snapshot_id=record.dataset_snapshot_id,
        dataset_manifest_hash=record.dataset_manifest_hash,
        base_model_id=record.base_model_id,
        base_model_digest=record.base_model_digest,
        tokenizer_digest=record.tokenizer_digest,
        chat_template_digest=record.chat_template_digest,
        git_commit=record.git_commit,
        container_digest=record.container_digest,
        training_config=record.training_config,
        config_hash=record.config_hash,
        distributed_profile=record.distributed_profile,
        random_seeds=record.random_seeds,
        hardware_topology=record.hardware_topology,
        mlflow_experiment_name=record.mlflow_experiment_name,
        mlflow_run_id=record.mlflow_run_id,
        metrics=record.metrics,
        cost_summary=record.cost_summary,
        license_status=record.license_status,
        created_by_subject_id=record.created_by_subject_id,
        completed_at=record.completed_at,
        failure_reason=record.failure_reason,
        version=record.version,
        created_at=record.created_at,
        artifacts=artifacts,
        legal_actions=legal_actions,
    )


def _suite_response(record: EvaluationSuiteRecord) -> EvaluationSuiteResponse:
    return EvaluationSuiteResponse(
        suite_id=record.suite_id,
        name=record.name,
        version=record.version,
        tier=record.tier,
        source_snapshot_id=record.source_snapshot_id,
        purpose=record.purpose,
        status=record.status,
        manifest_hash=record.manifest_hash,
        sample_count=record.sample_count,
        slice_counts=record.slice_counts,
        created_by_subject_id=record.created_by_subject_id,
        created_at=record.created_at,
    )


def _policy_response(record: EvaluationPolicyRecord) -> EvaluationPolicyResponse:
    return EvaluationPolicyResponse(
        policy_id=record.policy_id,
        name=record.name,
        version=record.version,
        status=record.status,
        primary_metric=record.primary_metric,
        hard_gates=record.hard_gates,
        thresholds=record.thresholds,
        policy_hash=record.policy_hash,
        created_by_subject_id=record.created_by_subject_id,
        created_at=record.created_at,
    )


def _evaluation_response(record: ModelEvaluationRunRecord) -> ModelEvaluationResponse:
    return ModelEvaluationResponse(
        evaluation_id=record.evaluation_id,
        candidate_experiment_id=record.candidate_experiment_id,
        baseline_experiment_id=record.baseline_experiment_id,
        suite_id=record.suite_id,
        policy_id=record.policy_id,
        status=record.status,
        decision=record.decision,
        primary_metric=record.primary_metric,
        candidate_score=record.candidate_score,
        baseline_score=record.baseline_score,
        quality_delta=record.quality_delta,
        ci_low=record.ci_low,
        ci_high=record.ci_high,
        latency_improvement=record.latency_improvement,
        cost_improvement=record.cost_improvement,
        hard_gate_results=record.hard_gate_results,
        slice_metrics=record.slice_metrics,
        comparison_kind=record.comparison_kind,
        comparison_context=record.comparison_context,
        report_hash=record.report_hash,
        triggered_by_subject_id=record.triggered_by_subject_id,
        completed_at=record.completed_at,
        failure_reason=record.failure_reason,
    )


def _evaluation_job_response(record: ModelEvaluationJobRecord) -> EvaluationJobResponse:
    return EvaluationJobResponse(
        job_id=record.job_id,
        candidate_experiment_id=record.candidate_experiment_id,
        baseline_experiment_id=record.baseline_experiment_id,
        suite_id=record.suite_id,
        policy_id=record.policy_id,
        status=record.status,
        target_profile=record.target_profile,
        runner_git_commit=record.runner_git_commit,
        container_digest=record.container_digest,
        runner_config=record.runner_config,
        config_hash=record.config_hash,
        result_evaluation_id=record.result_evaluation_id,
        created_by_subject_id=record.created_by_subject_id,
        started_at=record.started_at,
        completed_at=record.completed_at,
        failure_reason=record.failure_reason,
        version=record.version,
        created_at=record.created_at,
    )


def _experiment_or_404(
    session: Session, tenant_id: str, experiment_id: str
) -> TrainingExperimentRecord:
    record = session.scalar(
        select(TrainingExperimentRecord).where(
            TrainingExperimentRecord.tenant_id == tenant_id,
            TrainingExperimentRecord.experiment_id == experiment_id,
        )
    )
    if record is None:
        raise _translate(ExperimentNotVisible())
    return record


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


def _allowed(
    identity: IdentityContext, authorizer: Authorizer, action: Action, resource_id: str
) -> bool:
    return authorizer.decide(
        identity,
        action,
        ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
    ).allowed


def _collection_actions(identity: IdentityContext, authorizer: Authorizer) -> list[str]:
    return _legal_collection_action(
        identity, authorizer, Action.CREATE_TRAINING_EXPERIMENT, "CREATE_EXPERIMENT"
    )


def _legal_collection_action(
    identity: IdentityContext,
    authorizer: Authorizer,
    action: Action,
    label: str,
) -> list[str]:
    return [label] if _allowed(identity, authorizer, action, "collection") else []


def _version(value: str) -> int:
    try:
        version = int(value.strip().strip('"'))
    except ValueError as exc:
        raise AppError(
            400, "invalid_version_precondition", "validation", "If-Match is invalid"
        ) from exc
    if version < 1:
        raise AppError(400, "invalid_version_precondition", "validation", "If-Match is invalid")
    return version


def _translate(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(403, "authorization_denied", "authorization", "Action is not allowed")
    if isinstance(exc, ExperimentNotVisible):
        return AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        )
    if isinstance(exc, ExperimentConflict):
        return AppError(
            409,
            exc.reason,
            "conflict",
            "Experiment or evaluation state changed",
            details={"current_version": exc.current_version},
        )
    if isinstance(exc, ExperimentGateDenied):
        return AppError(
            409,
            exc.reason,
            "governance_gate",
            "Training or evaluation governance gate rejected the operation",
        )
    raise TypeError("unsupported experiment error")


def _invalid(exc: ValueError) -> AppError:
    return AppError(400, "invalid_experiment_request", "validation", str(exc))
