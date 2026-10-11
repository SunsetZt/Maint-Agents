"""Formal API for governed telemetry, alert review, and maintenance referral."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field, StrictInt

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_identity,
    get_predictive_maintenance_service,
    get_rul_dataset_pipeline,
    get_telemetry_dataset_pipeline,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import ROLE_ACTIONS, Action, Authorizer, ResourceContext
from industrial_ops_agent.predictive_maintenance.dataset_pipeline import (
    TelemetryDatasetPipelineError,
    TelemetryDatasetPipelineService,
)
from industrial_ops_agent.predictive_maintenance.rul_dataset_pipeline import (
    RulDatasetPipelineError,
    RulDatasetPipelineService,
)
from industrial_ops_agent.predictive_maintenance.service import (
    AlertCandidateView,
    AlertDecision,
    DetectionResult,
    MaintenanceReferralView,
    OutcomeType,
    PredictiveMaintenanceConflict,
    PredictiveMaintenanceNotFound,
    PredictiveMaintenanceOverview,
    PredictiveMaintenanceService,
    PredictiveOutcomeInput,
    PredictiveOutcomeView,
    RulForecastCalibrationView,
    RulForecastDecision,
    RulForecastView,
    RulReleaseCalibrationView,
    TelemetryEventInput,
    TelemetryEventView,
    TelemetryIngestResult,
    TelemetryWindowView,
)

router = APIRouter(tags=["predictive-maintenance"])


class TelemetryEventBody(BaseModel):
    event_id: str = Field(min_length=1, max_length=128)
    asset_id: str = Field(min_length=1, max_length=128)
    schema_version: str = Field(min_length=1, max_length=64)
    sequence: StrictInt = Field(ge=0)
    source_system: str = Field(min_length=1, max_length=128)
    source_ref: str = Field(min_length=1, max_length=255)
    event_time: datetime
    measurements: dict[str, float] = Field(min_length=1, max_length=16)
    quality_flags: list[str] = Field(default_factory=list, max_length=8)


class BuildTelemetryWindowBody(BaseModel):
    asset_id: str = Field(min_length=1, max_length=128)
    schema_version: str = Field(min_length=1, max_length=64)
    window_start: datetime
    window_end: datetime
    include_too_late: bool = False


class BuildTelemetryDatasetBody(BaseModel):
    window_start: datetime
    window_end: datetime
    engine: str = Field(default="local", pattern=r"^(local|spark)$")


class AlertDecisionBody(BaseModel):
    decision: AlertDecision
    reason_code: str = Field(min_length=1, max_length=128)
    notes: str | None = Field(default=None, max_length=2_000)


class MaintenanceReferralBody(BaseModel):
    incident_draft_id: str = Field(min_length=1, max_length=128)


class PredictiveOutcomeBody(BaseModel):
    asset_id: str = Field(min_length=1, max_length=128)
    alert_candidate_id: str | None = Field(default=None, min_length=1, max_length=128)
    outcome_type: OutcomeType
    observed_at: datetime
    failure_observed_at: datetime | None = None
    work_order_id: str | None = Field(default=None, min_length=1, max_length=128)
    avoided_downtime_minutes: StrictInt = Field(default=0, ge=0, le=10_000_000)
    evidence_ref: str = Field(min_length=1, max_length=255)
    evidence_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class RulForecastDecisionBody(BaseModel):
    decision: RulForecastDecision
    reason: str = Field(min_length=3, max_length=2_000)


class TelemetryEventResponse(BaseModel):
    event_id: str
    asset_id: str
    schema_version: str
    sequence: int
    source_system: str
    source_ref: str
    event_time: datetime
    received_at: datetime
    measurements: dict[str, float]
    quality_flags: list[str]
    ingestion_status: str
    version: int


class TelemetryIngestResponse(BaseModel):
    event: TelemetryEventResponse
    duplicate: bool


class TelemetryWindowResponse(BaseModel):
    window_id: str
    asset_id: str
    schema_version: str
    window_start: datetime
    window_end: datetime
    feature_snapshot_id: str
    source_event_ids: list[str]
    features: dict[str, Any]
    missing_signals: list[str]
    supporting_signal_refs: list[str]
    sample_count: int
    quality_status: str
    includes_too_late: bool
    supersedes_window_id: str | None
    version: int


class MaintenanceReferralResponse(BaseModel):
    referral_id: str
    alert_candidate_id: str
    incident_draft_id: str
    linked_by_subject_id: str
    linked_at: datetime


class PredictiveAlertCandidateResponse(BaseModel):
    alert_candidate_id: str
    asset_id: str
    window_id: str
    schema_version: str
    window_start: datetime
    window_end: datetime
    feature_snapshot_id: str
    model_release_id: str
    detector_kind: str
    anomaly_score: float
    threshold: float
    threshold_policy_id: str
    supporting_signal_refs: list[str]
    explanation_codes: list[str]
    status: str
    detected_at: datetime
    reviewed_at: datetime | None
    reviewed_by_subject_id: str | None
    decision_reason_code: str | None
    decision_notes: str | None
    referral: MaintenanceReferralResponse | None
    version: int


class DetectionResponse(BaseModel):
    window_id: str
    model_release_id: str
    detector_kind: str
    anomaly_score: float
    threshold: float
    threshold_policy_id: str
    explanation_codes: list[str]
    fallback_reason: str | None
    candidate: PredictiveAlertCandidateResponse | None


class PredictiveOutcomeResponse(BaseModel):
    outcome_id: str
    asset_id: str
    alert_candidate_id: str | None
    outcome_type: str
    observed_at: datetime
    failure_observed_at: datetime | None
    work_order_id: str | None
    avoided_downtime_minutes: int
    evidence_ref: str
    recorded_by_subject_id: str
    version: int


class RulForecastResponse(BaseModel):
    rul_forecast_id: str
    alert_candidate_id: str
    asset_id: str
    window_id: str
    feature_snapshot_id: str
    source_model_release_id: str
    forecast_model_version: str
    estimate_minutes: float
    lower_bound_minutes: float
    upper_bound_minutes: float
    recommended_inspection_at: datetime
    historical_failure_count: int
    source_outcome_ids: list[str]
    cohort_digest: str
    history_cutoff_at: datetime
    source_alert_version: int
    status: Literal["PENDING_REVIEW", "ACCEPTED", "REJECTED"]
    created_by_subject_id: str
    reviewed_by_subject_id: str | None
    review_reason: str | None
    reviewed_at: datetime | None
    version: int


class RulForecastCalibrationResponse(BaseModel):
    calibration_id: str
    rul_forecast_id: str
    outcome_id: str
    alert_candidate_id: str
    asset_id: str
    source_model_release_id: str
    forecast_model_version: str
    forecast_review_status: str
    actual_minutes: float
    absolute_error_minutes: float
    interval_covered: bool
    calibration_status: str
    retraining_recommended: bool
    decision_reasons: list[str]
    evidence_digest: str
    evaluated_at: datetime


class RulReleaseCalibrationResponse(BaseModel):
    source_model_release_id: str
    forecast_model_version: str
    labeled_outcome_count: int
    reference_outcome_count: int
    recent_outcome_count: int
    median_absolute_error_minutes: float | None
    p90_absolute_error_minutes: float | None
    interval_coverage: float | None
    reference_median_absolute_error_minutes: float | None
    recent_median_absolute_error_minutes: float | None
    recent_interval_coverage: float | None
    status: str
    retraining_recommended: bool
    reason_codes: list[str]
    latest_evaluated_at: datetime | None
    evidence_digest: str


class PredictiveMetricsResponse(BaseModel):
    alert_candidate_count: int
    pending_confirmation_count: int
    confirmed_count: int
    dismissed_count: int
    referred_count: int
    work_order_conversion_count: int
    work_order_conversion_rate: float | None
    labeled_alert_count: int
    false_positive_rate: float | None
    false_positives_per_device_day: dict[str, int]
    observed_failure_count: int
    missed_failure_count: int
    miss_rate: float | None
    median_lead_time_minutes: float | None
    avoided_downtime_minutes: int
    degraded_window_count: int
    window_count: int
    missing_signal_window_rate: float | None
    drift_status: str
    label_sufficiency: str
    rul_claim_allowed: bool


class PredictiveMaintenanceOverviewResponse(BaseModel):
    telemetry_contract_version: str
    threshold_policy_id: str
    events: list[TelemetryEventResponse]
    windows: list[TelemetryWindowResponse]
    alert_candidates: list[PredictiveAlertCandidateResponse]
    outcomes: list[PredictiveOutcomeResponse]
    rul_forecasts: list[RulForecastResponse]
    rul_calibrations: list[RulForecastCalibrationResponse]
    rul_release_calibrations: list[RulReleaseCalibrationResponse]
    metrics: PredictiveMetricsResponse


class OverviewEnvelope(BaseModel):
    data: PredictiveMaintenanceOverviewResponse
    legal_actions: list[str]
    meta: dict[str, str]


class TelemetryEventEnvelope(BaseModel):
    data: TelemetryIngestResponse
    meta: dict[str, str]


class TelemetryWindowEnvelope(BaseModel):
    data: TelemetryWindowResponse
    meta: dict[str, str]


class DetectionEnvelope(BaseModel):
    data: DetectionResponse
    meta: dict[str, str]


class AlertCandidateEnvelope(BaseModel):
    data: PredictiveAlertCandidateResponse
    meta: dict[str, str]


class PredictiveOutcomeEnvelope(BaseModel):
    data: PredictiveOutcomeResponse
    meta: dict[str, str]


class RulForecastEnvelope(BaseModel):
    data: RulForecastResponse
    meta: dict[str, str]


class TelemetryDatasetSnapshotResponse(BaseModel):
    run_id: str
    snapshot_id: str
    status: str
    training_eligible: bool
    row_count: int
    split_counts: dict[str, int]
    input_manifest_hash: str
    manifest_key: str | None
    blocker_codes: list[str]


class TelemetryDatasetSnapshotEnvelope(BaseModel):
    data: TelemetryDatasetSnapshotResponse
    meta: dict[str, str]


@router.get("/predictive-maintenance/overview", response_model=OverviewEnvelope)
async def get_predictive_maintenance_overview(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[
        PredictiveMaintenanceService,
        Depends(get_predictive_maintenance_service),
    ],
) -> OverviewEnvelope:
    try:
        overview = await asyncio.to_thread(
            service.overview,
            identity,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _translate(exc) from exc
    return OverviewEnvelope(
        data=_overview_response(overview),
        legal_actions=_legal_actions(identity),
        meta={"request_id": request.state.request_id},
    )


@router.post("/telemetry/events", response_model=TelemetryEventEnvelope, status_code=201)
async def ingest_telemetry_event(
    body: TelemetryEventBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[
        PredictiveMaintenanceService,
        Depends(get_predictive_maintenance_service),
    ],
) -> TelemetryEventEnvelope:
    try:
        result = await asyncio.to_thread(
            service.ingest_event,
            identity,
            TelemetryEventInput(
                event_id=body.event_id,
                asset_id=body.asset_id,
                schema_version=body.schema_version,
                sequence=body.sequence,
                source_system=body.source_system,
                source_ref=body.source_ref,
                event_time=body.event_time,
                measurements=body.measurements,
                quality_flags=tuple(body.quality_flags),
            ),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        PredictiveMaintenanceConflict,
        PredictiveMaintenanceNotFound,
        ValueError,
    ) as exc:
        raise _translate(exc) from exc
    return TelemetryEventEnvelope(
        data=_ingest_response(result),
        meta={"request_id": request.state.request_id},
    )


@router.post("/telemetry/windows", response_model=TelemetryWindowEnvelope, status_code=201)
async def build_telemetry_window(
    body: BuildTelemetryWindowBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[
        PredictiveMaintenanceService,
        Depends(get_predictive_maintenance_service),
    ],
) -> TelemetryWindowEnvelope:
    try:
        window = await asyncio.to_thread(
            service.build_window,
            identity,
            asset_id=body.asset_id,
            schema_version=body.schema_version,
            window_start=body.window_start,
            window_end=body.window_end,
            include_too_late=body.include_too_late,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        PredictiveMaintenanceConflict,
        PredictiveMaintenanceNotFound,
        ValueError,
    ) as exc:
        raise _translate(exc) from exc
    return TelemetryWindowEnvelope(
        data=_window_response(window),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/predictive-maintenance/dataset-snapshots",
    response_model=TelemetryDatasetSnapshotEnvelope,
    status_code=201,
)
async def build_telemetry_dataset_snapshot(
    body: BuildTelemetryDatasetBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    pipeline: Annotated[
        TelemetryDatasetPipelineService,
        Depends(get_telemetry_dataset_pipeline),
    ],
) -> TelemetryDatasetSnapshotEnvelope:
    try:
        authorizer.require(
            identity,
            Action.BUILD_TELEMETRY_DATASET,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="predictive-maintenance-dataset-snapshots",
            ),
            request_id=request.state.request_id,
        )
        result = await asyncio.to_thread(
            pipeline.build_snapshot,
            identity.tenant_context,
            window_start=body.window_start,
            window_end=body.window_end,
            engine=body.engine,
        )
    except AuthorizationDenied as exc:
        raise _translate(exc) from exc
    except TelemetryDatasetPipelineError as exc:
        raise AppError(409, "telemetry_dataset_gate_rejected", "governance_gate", str(exc)) from exc
    except ValueError as exc:
        raise AppError(400, "telemetry_dataset_request_invalid", "validation", str(exc)) from exc
    return TelemetryDatasetSnapshotEnvelope(
        data=TelemetryDatasetSnapshotResponse(
            run_id=result.run_id,
            snapshot_id=result.snapshot_id,
            status=result.status,
            training_eligible=result.training_eligible,
            row_count=result.row_count,
            split_counts=result.split_counts,
            input_manifest_hash=result.input_manifest_hash,
            manifest_key=result.manifest_key,
            blocker_codes=list(result.blocker_codes),
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/predictive-maintenance/rul-dataset-snapshots",
    response_model=TelemetryDatasetSnapshotEnvelope,
    status_code=201,
)
async def build_rul_dataset_snapshot(
    body: BuildTelemetryDatasetBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    pipeline: Annotated[
        RulDatasetPipelineService,
        Depends(get_rul_dataset_pipeline),
    ],
) -> TelemetryDatasetSnapshotEnvelope:
    try:
        authorizer.require(
            identity,
            Action.BUILD_RUL_DATASET,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="predictive-maintenance-rul-dataset-snapshots",
            ),
            request_id=request.state.request_id,
        )
        result = await asyncio.to_thread(
            pipeline.build_snapshot,
            identity.tenant_context,
            window_start=body.window_start,
            window_end=body.window_end,
            engine=body.engine,
        )
    except AuthorizationDenied as exc:
        raise _translate(exc) from exc
    except RulDatasetPipelineError as exc:
        raise AppError(409, "rul_dataset_gate_rejected", "governance_gate", str(exc)) from exc
    except ValueError as exc:
        raise AppError(400, "rul_dataset_request_invalid", "validation", str(exc)) from exc
    return TelemetryDatasetSnapshotEnvelope(
        data=TelemetryDatasetSnapshotResponse(
            run_id=result.run_id,
            snapshot_id=result.snapshot_id,
            status=result.status,
            training_eligible=result.training_eligible,
            row_count=result.row_count,
            split_counts=result.split_counts,
            input_manifest_hash=result.input_manifest_hash,
            manifest_key=result.manifest_key,
            blocker_codes=list(result.blocker_codes),
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post("/telemetry/windows/{window_id}/detect", response_model=DetectionEnvelope)
async def detect_telemetry_window(
    window_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[
        PredictiveMaintenanceService,
        Depends(get_predictive_maintenance_service),
    ],
) -> DetectionEnvelope:
    try:
        result = await asyncio.to_thread(
            service.detect_window,
            identity,
            window_id,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        PredictiveMaintenanceConflict,
        PredictiveMaintenanceNotFound,
        ValueError,
    ) as exc:
        raise _translate(exc) from exc
    return DetectionEnvelope(
        data=_detection_response(result),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/alert-candidates/{alert_candidate_id}/decisions",
    response_model=AlertCandidateEnvelope,
)
async def decide_alert_candidate(
    alert_candidate_id: str,
    body: AlertDecisionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[
        PredictiveMaintenanceService,
        Depends(get_predictive_maintenance_service),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> AlertCandidateEnvelope:
    try:
        candidate = await asyncio.to_thread(
            service.decide_candidate,
            identity,
            alert_candidate_id,
            decision=body.decision,
            reason_code=body.reason_code,
            notes=body.notes,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        PredictiveMaintenanceConflict,
        PredictiveMaintenanceNotFound,
        ValueError,
    ) as exc:
        raise _translate(exc) from exc
    return AlertCandidateEnvelope(
        data=_candidate_response(candidate),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/alert-candidates/{alert_candidate_id}/referrals",
    response_model=AlertCandidateEnvelope,
)
async def refer_alert_candidate(
    alert_candidate_id: str,
    body: MaintenanceReferralBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[
        PredictiveMaintenanceService,
        Depends(get_predictive_maintenance_service),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> AlertCandidateEnvelope:
    try:
        candidate = await asyncio.to_thread(
            service.link_incident_draft,
            identity,
            alert_candidate_id,
            incident_draft_id=body.incident_draft_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        PredictiveMaintenanceConflict,
        PredictiveMaintenanceNotFound,
        ValueError,
    ) as exc:
        raise _translate(exc) from exc
    return AlertCandidateEnvelope(
        data=_candidate_response(candidate),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/predictive-maintenance/outcomes",
    response_model=PredictiveOutcomeEnvelope,
    status_code=201,
)
async def record_predictive_outcome(
    body: PredictiveOutcomeBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[
        PredictiveMaintenanceService,
        Depends(get_predictive_maintenance_service),
    ],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> PredictiveOutcomeEnvelope:
    try:
        outcome = await asyncio.to_thread(
            service.record_outcome,
            identity,
            PredictiveOutcomeInput(
                asset_id=body.asset_id,
                alert_candidate_id=body.alert_candidate_id,
                outcome_type=body.outcome_type,
                observed_at=body.observed_at,
                failure_observed_at=body.failure_observed_at,
                work_order_id=body.work_order_id,
                avoided_downtime_minutes=body.avoided_downtime_minutes,
                evidence_ref=body.evidence_ref,
                evidence_digest=body.evidence_digest,
            ),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        PredictiveMaintenanceConflict,
        PredictiveMaintenanceNotFound,
        ValueError,
    ) as exc:
        raise _translate(exc) from exc
    return PredictiveOutcomeEnvelope(
        data=_outcome_response(outcome),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/alert-candidates/{alert_candidate_id}/rul-forecasts",
    response_model=RulForecastEnvelope,
    status_code=201,
)
async def create_rul_forecast(
    alert_candidate_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[
        PredictiveMaintenanceService,
        Depends(get_predictive_maintenance_service),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> RulForecastEnvelope:
    try:
        forecast = await asyncio.to_thread(
            service.create_rul_forecast,
            identity,
            alert_candidate_id,
            expected_alert_version=_version(if_match),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        PredictiveMaintenanceConflict,
        PredictiveMaintenanceNotFound,
        ValueError,
    ) as exc:
        raise _translate(exc) from exc
    return RulForecastEnvelope(
        data=_rul_forecast_response(forecast),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/rul-forecast-candidates/{rul_forecast_id}/decisions",
    response_model=RulForecastEnvelope,
)
async def review_rul_forecast(
    rul_forecast_id: str,
    body: RulForecastDecisionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[
        PredictiveMaintenanceService,
        Depends(get_predictive_maintenance_service),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> RulForecastEnvelope:
    try:
        forecast = await asyncio.to_thread(
            service.review_rul_forecast,
            identity,
            rul_forecast_id,
            decision=body.decision,
            reason=body.reason,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        PredictiveMaintenanceConflict,
        PredictiveMaintenanceNotFound,
        ValueError,
    ) as exc:
        raise _translate(exc) from exc
    return RulForecastEnvelope(
        data=_rul_forecast_response(forecast),
        meta={"request_id": request.state.request_id},
    )


def _overview_response(
    value: PredictiveMaintenanceOverview,
) -> PredictiveMaintenanceOverviewResponse:
    return PredictiveMaintenanceOverviewResponse(
        telemetry_contract_version=value.telemetry_contract_version,
        threshold_policy_id=value.threshold_policy_id,
        events=[_event_response(item) for item in value.events],
        windows=[_window_response(item) for item in value.windows],
        alert_candidates=[_candidate_response(item) for item in value.alert_candidates],
        outcomes=[_outcome_response(item) for item in value.outcomes],
        rul_forecasts=[_rul_forecast_response(item) for item in value.rul_forecasts],
        rul_calibrations=[_rul_calibration_response(item) for item in value.rul_calibrations],
        rul_release_calibrations=[
            _rul_release_calibration_response(item) for item in value.rul_release_calibrations
        ],
        metrics=PredictiveMetricsResponse(**asdict(value.metrics)),
    )


def _ingest_response(value: TelemetryIngestResult) -> TelemetryIngestResponse:
    return TelemetryIngestResponse(event=_event_response(value.event), duplicate=value.duplicate)


def _event_response(value: TelemetryEventView) -> TelemetryEventResponse:
    return TelemetryEventResponse(
        event_id=value.event_id,
        asset_id=value.asset_id,
        schema_version=value.schema_version,
        sequence=value.sequence,
        source_system=value.source_system,
        source_ref=value.source_ref,
        event_time=value.event_time,
        received_at=value.received_at,
        measurements=value.measurements,
        quality_flags=list(value.quality_flags),
        ingestion_status=value.ingestion_status,
        version=value.version,
    )


def _window_response(value: TelemetryWindowView) -> TelemetryWindowResponse:
    return TelemetryWindowResponse(
        window_id=value.window_id,
        asset_id=value.asset_id,
        schema_version=value.schema_version,
        window_start=value.window_start,
        window_end=value.window_end,
        feature_snapshot_id=value.feature_snapshot_id,
        source_event_ids=list(value.source_event_ids),
        features=value.features,
        missing_signals=list(value.missing_signals),
        supporting_signal_refs=list(value.supporting_signal_refs),
        sample_count=value.sample_count,
        quality_status=value.quality_status,
        includes_too_late=value.includes_too_late,
        supersedes_window_id=value.supersedes_window_id,
        version=value.version,
    )


def _candidate_response(value: AlertCandidateView) -> PredictiveAlertCandidateResponse:
    return PredictiveAlertCandidateResponse(
        alert_candidate_id=value.alert_candidate_id,
        asset_id=value.asset_id,
        window_id=value.window_id,
        schema_version=value.schema_version,
        window_start=value.window_start,
        window_end=value.window_end,
        feature_snapshot_id=value.feature_snapshot_id,
        model_release_id=value.model_release_id,
        detector_kind=value.detector_kind,
        anomaly_score=value.anomaly_score,
        threshold=value.threshold,
        threshold_policy_id=value.threshold_policy_id,
        supporting_signal_refs=list(value.supporting_signal_refs),
        explanation_codes=list(value.explanation_codes),
        status=value.status,
        detected_at=value.detected_at,
        reviewed_at=value.reviewed_at,
        reviewed_by_subject_id=value.reviewed_by_subject_id,
        decision_reason_code=value.decision_reason_code,
        decision_notes=value.decision_notes,
        referral=_referral_response(value.referral) if value.referral else None,
        version=value.version,
    )


def _referral_response(value: MaintenanceReferralView) -> MaintenanceReferralResponse:
    return MaintenanceReferralResponse(
        referral_id=value.referral_id,
        alert_candidate_id=value.alert_candidate_id,
        incident_draft_id=value.incident_draft_id,
        linked_by_subject_id=value.linked_by_subject_id,
        linked_at=value.linked_at,
    )


def _detection_response(value: DetectionResult) -> DetectionResponse:
    return DetectionResponse(
        window_id=value.window_id,
        model_release_id=value.model_release_id,
        detector_kind=value.detector_kind,
        anomaly_score=value.anomaly_score,
        threshold=value.threshold,
        threshold_policy_id=value.threshold_policy_id,
        explanation_codes=list(value.explanation_codes),
        fallback_reason=value.fallback_reason,
        candidate=_candidate_response(value.candidate) if value.candidate else None,
    )


def _outcome_response(value: PredictiveOutcomeView) -> PredictiveOutcomeResponse:
    return PredictiveOutcomeResponse(
        outcome_id=value.outcome_id,
        asset_id=value.asset_id,
        alert_candidate_id=value.alert_candidate_id,
        outcome_type=value.outcome_type,
        observed_at=value.observed_at,
        failure_observed_at=value.failure_observed_at,
        work_order_id=value.work_order_id,
        avoided_downtime_minutes=value.avoided_downtime_minutes,
        evidence_ref=value.evidence_ref,
        recorded_by_subject_id=value.recorded_by_subject_id,
        version=value.version,
    )


def _rul_forecast_response(value: RulForecastView) -> RulForecastResponse:
    return RulForecastResponse(
        rul_forecast_id=value.rul_forecast_id,
        alert_candidate_id=value.alert_candidate_id,
        asset_id=value.asset_id,
        window_id=value.window_id,
        feature_snapshot_id=value.feature_snapshot_id,
        source_model_release_id=value.source_model_release_id,
        forecast_model_version=value.forecast_model_version,
        estimate_minutes=value.estimate_minutes,
        lower_bound_minutes=value.lower_bound_minutes,
        upper_bound_minutes=value.upper_bound_minutes,
        recommended_inspection_at=value.recommended_inspection_at,
        historical_failure_count=value.historical_failure_count,
        source_outcome_ids=list(value.source_outcome_ids),
        cohort_digest=value.cohort_digest,
        history_cutoff_at=value.history_cutoff_at,
        source_alert_version=value.source_alert_version,
        status=cast(Literal["PENDING_REVIEW", "ACCEPTED", "REJECTED"], value.status),
        created_by_subject_id=value.created_by_subject_id,
        reviewed_by_subject_id=value.reviewed_by_subject_id,
        review_reason=value.review_reason,
        reviewed_at=value.reviewed_at,
        version=value.version,
    )


def _rul_calibration_response(
    value: RulForecastCalibrationView,
) -> RulForecastCalibrationResponse:
    return RulForecastCalibrationResponse(
        calibration_id=value.calibration_id,
        rul_forecast_id=value.rul_forecast_id,
        outcome_id=value.outcome_id,
        alert_candidate_id=value.alert_candidate_id,
        asset_id=value.asset_id,
        source_model_release_id=value.source_model_release_id,
        forecast_model_version=value.forecast_model_version,
        forecast_review_status=value.forecast_review_status,
        actual_minutes=value.actual_minutes,
        absolute_error_minutes=value.absolute_error_minutes,
        interval_covered=value.interval_covered,
        calibration_status=value.calibration_status,
        retraining_recommended=value.retraining_recommended,
        decision_reasons=list(value.decision_reasons),
        evidence_digest=value.evidence_digest,
        evaluated_at=value.evaluated_at,
    )


def _rul_release_calibration_response(
    value: RulReleaseCalibrationView,
) -> RulReleaseCalibrationResponse:
    return RulReleaseCalibrationResponse(
        source_model_release_id=value.source_model_release_id,
        forecast_model_version=value.forecast_model_version,
        labeled_outcome_count=value.labeled_outcome_count,
        reference_outcome_count=value.reference_outcome_count,
        recent_outcome_count=value.recent_outcome_count,
        median_absolute_error_minutes=value.median_absolute_error_minutes,
        p90_absolute_error_minutes=value.p90_absolute_error_minutes,
        interval_coverage=value.interval_coverage,
        reference_median_absolute_error_minutes=(value.reference_median_absolute_error_minutes),
        recent_median_absolute_error_minutes=value.recent_median_absolute_error_minutes,
        recent_interval_coverage=value.recent_interval_coverage,
        status=value.status,
        retraining_recommended=value.retraining_recommended,
        reason_codes=list(value.reason_codes),
        latest_evaluated_at=value.latest_evaluated_at,
        evidence_digest=value.evidence_digest,
    )


def _version(value: str) -> int:
    try:
        version = int(value.strip('"'))
    except ValueError as exc:
        raise ValueError("If-Match must contain an integer version") from exc
    if version < 1:
        raise ValueError("If-Match must contain a positive version")
    return version


def _legal_actions(identity: IdentityContext) -> list[str]:
    allowed = frozenset().union(*(ROLE_ACTIONS.get(role, frozenset()) for role in identity.roles))
    relevant = {
        Action.INGEST_TELEMETRY,
        Action.BUILD_TELEMETRY_WINDOW,
        Action.RUN_ANOMALY_DETECTION,
        Action.DECIDE_ALERT_CANDIDATE,
        Action.REFER_ALERT_CANDIDATE,
        Action.RECORD_PREDICTIVE_OUTCOME,
        Action.BUILD_TELEMETRY_DATASET,
        Action.BUILD_RUL_DATASET,
        Action.GENERATE_RUL_FORECAST,
        Action.REVIEW_RUL_FORECAST,
    }
    return sorted(action.value for action in allowed & relevant)


def _translate(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(
            403, "predictive_maintenance_forbidden", "authorization", "Action is not allowed"
        )
    if isinstance(exc, PredictiveMaintenanceNotFound):
        return AppError(404, str(exc), "not_found", "Resource not found or not visible")
    if isinstance(exc, PredictiveMaintenanceConflict):
        return AppError(409, str(exc), "conflict", "Predictive-maintenance state conflict")
    return AppError(400, "predictive_maintenance_request_invalid", "validation", str(exc))
