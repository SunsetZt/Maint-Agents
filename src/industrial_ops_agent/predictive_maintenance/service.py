"""Event-time telemetry processing and human-governed maintenance alerting."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from statistics import median
from time import perf_counter
from typing import Any
from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _stored_utc
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AlertCandidateDecisionRecord,
    AlertCandidateRecord,
    AssetRecord,
    IncidentDraftRecord,
    IncidentRecord,
    ModelAliasRecord,
    ModelReleaseRecord,
    PredictiveMaintenanceReferralRecord,
    PredictiveOutcomeRecord,
    RulForecastCalibrationRecord,
    RulForecastCandidateRecord,
    TelemetryEventRecord,
    TelemetryStreamStateRecord,
    TelemetryWindowRecord,
    WorkOrderRecord,
)
from industrial_ops_agent.predictive_maintenance.dataset_contract import SIGNAL_ORDER
from industrial_ops_agent.predictive_maintenance.inference import (
    TimeseriesDetectionInput,
    TimeseriesDetector,
    TimeseriesInferenceUnavailable,
)
from industrial_ops_agent.predictive_maintenance.rul_inference import (
    RulForecaster,
    RulForecastInput,
    RulInferenceUnavailable,
)
from industrial_ops_agent.runtime_metrics import PredictiveMaintenanceOnlineMetrics

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ALLOWED_LATENESS = timedelta(minutes=10)
_MAX_FUTURE_SKEW = timedelta(minutes=5)
_MAX_WINDOW = timedelta(hours=24)
_MIN_SAMPLES = 3
_RULE_MODEL_RELEASE = "rules-baseline-rotating/v1"
_THRESHOLD_POLICY = "rotating-equipment-threshold/v1"
_ALERT_THRESHOLD = 0.65
_RUL_FORECAST_MODEL = "empirical-lead-time-baseline/v1"
_MIN_RUL_FAILURES = 30
_MIN_RUL_LABELED_ALERTS = 100
_MIN_RUL_CALIBRATIONS = 30
_RUL_RECENT_CALIBRATIONS = 10
_RUL_MAE_RELATIVE_REGRESSION = 0.25
_RUL_MAE_ABSOLUTE_REGRESSION_MINUTES = 30.0
_RUL_INTERVAL_COVERAGE_FLOOR = 0.70
_TENANT_WIDE_ROLES = frozenset({Role.TENANT_ADMIN, Role.PLATFORM_OPERATOR})


class PredictiveMaintenanceConflict(RuntimeError):
    pass


class PredictiveMaintenanceNotFound(RuntimeError):
    pass


class AlertDecision(StrEnum):
    CONFIRMED = "CONFIRMED"
    DISMISSED = "DISMISSED"


class OutcomeType(StrEnum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    MISSED_FAILURE = "MISSED_FAILURE"


class RulForecastDecision(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class TelemetrySchema:
    schema_version: str
    required_signals: frozenset[str]
    ranges: dict[str, tuple[float, float]]


ROTATING_EQUIPMENT_V1 = TelemetrySchema(
    schema_version="rotating-equipment.telemetry.v1",
    required_signals=frozenset({"vibration_rms_mm_s", "bearing_temperature_c"}),
    ranges={
        "vibration_rms_mm_s": (0.0, 200.0),
        "bearing_temperature_c": (-50.0, 250.0),
        "motor_current_a": (0.0, 10_000.0),
        "rpm": (0.0, 100_000.0),
    },
)
SCHEMAS = {ROTATING_EQUIPMENT_V1.schema_version: ROTATING_EQUIPMENT_V1}


@dataclass(frozen=True, slots=True)
class TelemetryEventInput:
    event_id: str
    asset_id: str
    schema_version: str
    sequence: int
    source_system: str
    source_ref: str
    event_time: datetime
    measurements: dict[str, float]
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TelemetryEventView:
    event_id: str
    asset_id: str
    schema_version: str
    sequence: int
    source_system: str
    source_ref: str
    event_time: datetime
    received_at: datetime
    measurements: dict[str, float]
    quality_flags: tuple[str, ...]
    ingestion_status: str
    version: int


@dataclass(frozen=True, slots=True)
class TelemetryIngestResult:
    event: TelemetryEventView
    duplicate: bool


@dataclass(frozen=True, slots=True)
class TelemetryWindowView:
    window_id: str
    asset_id: str
    schema_version: str
    window_start: datetime
    window_end: datetime
    feature_snapshot_id: str
    source_event_ids: tuple[str, ...]
    features: dict[str, Any]
    missing_signals: tuple[str, ...]
    supporting_signal_refs: tuple[str, ...]
    sample_count: int
    quality_status: str
    includes_too_late: bool
    supersedes_window_id: str | None
    version: int


@dataclass(frozen=True, slots=True)
class MaintenanceReferralView:
    referral_id: str
    alert_candidate_id: str
    incident_draft_id: str
    linked_by_subject_id: str
    linked_at: datetime


@dataclass(frozen=True, slots=True)
class AlertCandidateView:
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
    supporting_signal_refs: tuple[str, ...]
    explanation_codes: tuple[str, ...]
    status: str
    detected_at: datetime
    reviewed_at: datetime | None
    reviewed_by_subject_id: str | None
    decision_reason_code: str | None
    decision_notes: str | None
    referral: MaintenanceReferralView | None
    version: int


@dataclass(frozen=True, slots=True)
class DetectionResult:
    window_id: str
    model_release_id: str
    detector_kind: str
    anomaly_score: float
    threshold: float
    threshold_policy_id: str
    explanation_codes: tuple[str, ...]
    fallback_reason: str | None
    candidate: AlertCandidateView | None


@dataclass(frozen=True, slots=True)
class PredictiveOutcomeInput:
    asset_id: str
    outcome_type: OutcomeType
    observed_at: datetime
    evidence_ref: str
    evidence_digest: str
    alert_candidate_id: str | None = None
    failure_observed_at: datetime | None = None
    work_order_id: str | None = None
    avoided_downtime_minutes: int = 0


@dataclass(frozen=True, slots=True)
class PredictiveOutcomeView:
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


@dataclass(frozen=True, slots=True)
class RulForecastView:
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
    source_outcome_ids: tuple[str, ...]
    cohort_digest: str
    history_cutoff_at: datetime
    source_alert_version: int
    status: str
    created_by_subject_id: str
    reviewed_by_subject_id: str | None
    review_reason: str | None
    reviewed_at: datetime | None
    version: int


@dataclass(frozen=True, slots=True)
class RulForecastCalibrationView:
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
    decision_reasons: tuple[str, ...]
    evidence_digest: str
    evaluated_at: datetime


@dataclass(frozen=True, slots=True)
class RulReleaseCalibrationView:
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
    reason_codes: tuple[str, ...]
    latest_evaluated_at: datetime | None
    evidence_digest: str


@dataclass(frozen=True, slots=True)
class PredictiveMetrics:
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


@dataclass(frozen=True, slots=True)
class PredictiveMaintenanceOverview:
    telemetry_contract_version: str
    threshold_policy_id: str
    events: tuple[TelemetryEventView, ...]
    windows: tuple[TelemetryWindowView, ...]
    alert_candidates: tuple[AlertCandidateView, ...]
    outcomes: tuple[PredictiveOutcomeView, ...]
    rul_forecasts: tuple[RulForecastView, ...]
    rul_calibrations: tuple[RulForecastCalibrationView, ...]
    rul_release_calibrations: tuple[RulReleaseCalibrationView, ...]
    metrics: PredictiveMetrics


class PredictiveMaintenanceService:
    """Own telemetry derivation while delegating maintenance execution to Incident/WorkOrder."""

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        timeseries_detector: TimeseriesDetector | None = None,
        online_metrics: PredictiveMaintenanceOnlineMetrics | None = None,
        rul_forecaster: RulForecaster | None = None,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._timeseries_detector = timeseries_detector
        self._online_metrics = online_metrics
        self._rul_forecaster = rul_forecaster

    def ingest_event(
        self,
        identity: IdentityContext,
        event: TelemetryEventInput,
        *,
        request_id: str,
        received_at: datetime | None = None,
    ) -> TelemetryIngestResult:
        schema, event_time, measurements, quality_flags = _validate_event(event)
        self._authorizer.require(
            identity,
            Action.INGEST_TELEMETRY,
            ResourceContext(identity.tenant_id, event.event_id, event.asset_id),
            request_id=request_id,
        )
        now = _as_utc(received_at or datetime.now(UTC))
        if event_time > now + _MAX_FUTURE_SKEW:
            raise ValueError("telemetry event time exceeds the allowed future skew")
        digest = _digest(
            {
                "event_id": event.event_id,
                "asset_id": event.asset_id,
                "schema_version": schema.schema_version,
                "sequence": event.sequence,
                "source_system": event.source_system,
                "source_ref": event.source_ref,
                "event_time": event_time.isoformat(),
                "measurements": measurements,
                "quality_flags": quality_flags,
            }
        )
        with self._database.transaction(identity.tenant_context) as session:
            _asset_or_hidden(session, identity.tenant_id, event.asset_id)
            existing = session.scalar(
                select(TelemetryEventRecord).where(
                    TelemetryEventRecord.tenant_id == identity.tenant_id,
                    TelemetryEventRecord.event_id == event.event_id,
                )
            )
            if existing is not None:
                if existing.payload_digest != digest:
                    raise PredictiveMaintenanceConflict("telemetry_event_id_payload_conflict")
                return TelemetryIngestResult(_event_view(existing), duplicate=True)

            sequence_owner = session.scalar(
                select(TelemetryEventRecord).where(
                    TelemetryEventRecord.tenant_id == identity.tenant_id,
                    TelemetryEventRecord.asset_id == event.asset_id,
                    TelemetryEventRecord.schema_version == schema.schema_version,
                    TelemetryEventRecord.sequence == event.sequence,
                )
            )
            if sequence_owner is not None:
                raise PredictiveMaintenanceConflict("telemetry_sequence_payload_conflict")

            stream = session.scalar(
                select(TelemetryStreamStateRecord)
                .where(
                    TelemetryStreamStateRecord.tenant_id == identity.tenant_id,
                    TelemetryStreamStateRecord.asset_id == event.asset_id,
                    TelemetryStreamStateRecord.schema_version == schema.schema_version,
                )
                .with_for_update()
            )
            status = _classify_event(stream, event_time, event.sequence)
            record = TelemetryEventRecord(
                event_id=event.event_id,
                tenant_id=identity.tenant_id,
                asset_id=event.asset_id,
                schema_version=schema.schema_version,
                sequence=event.sequence,
                source_system=event.source_system,
                source_ref=event.source_ref,
                event_time=event_time,
                received_at=now,
                measurements_json=measurements,
                quality_flags_json=list(quality_flags),
                payload_digest=digest,
                ingestion_status=status,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            if stream is None:
                session.add(
                    TelemetryStreamStateRecord(
                        stream_id=f"stream-{uuid4().hex}",
                        tenant_id=identity.tenant_id,
                        asset_id=event.asset_id,
                        schema_version=schema.schema_version,
                        max_event_time=event_time,
                        max_sequence=event.sequence,
                        watermark_at=event_time - _ALLOWED_LATENESS,
                        version=1,
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                stream.max_event_time = max(_stored_utc(stream.max_event_time), event_time)
                stream.max_sequence = max(stream.max_sequence, event.sequence)
                stream.watermark_at = stream.max_event_time - _ALLOWED_LATENESS
                stream.version += 1
                stream.updated_at = now
            session.flush()
            return TelemetryIngestResult(_event_view(record), duplicate=False)

    def build_window(
        self,
        identity: IdentityContext,
        *,
        asset_id: str,
        schema_version: str,
        window_start: datetime,
        window_end: datetime,
        include_too_late: bool,
        request_id: str,
    ) -> TelemetryWindowView:
        schema = _schema(schema_version)
        start = _as_utc(window_start)
        end = _as_utc(window_end)
        if start >= end or end - start > _MAX_WINDOW:
            raise ValueError("telemetry window must be positive and no longer than 24 hours")
        if end > datetime.now(UTC) + _MAX_FUTURE_SKEW:
            raise ValueError("telemetry window end exceeds the allowed future skew")
        self._authorizer.require(
            identity,
            Action.BUILD_TELEMETRY_WINDOW,
            ResourceContext(identity.tenant_id, asset_id, asset_id),
            request_id=request_id,
        )
        now = datetime.now(UTC)
        statuses = ["ACCEPTED", "OUT_OF_ORDER_ACCEPTED", "LATE_ACCEPTED"]
        if include_too_late:
            statuses.append("TOO_LATE")
        with self._database.transaction(identity.tenant_context) as session:
            _asset_or_hidden(session, identity.tenant_id, asset_id)
            events = list(
                session.scalars(
                    select(TelemetryEventRecord)
                    .where(
                        TelemetryEventRecord.tenant_id == identity.tenant_id,
                        TelemetryEventRecord.asset_id == asset_id,
                        TelemetryEventRecord.schema_version == schema_version,
                        TelemetryEventRecord.event_time >= start,
                        TelemetryEventRecord.event_time < end,
                        TelemetryEventRecord.ingestion_status.in_(statuses),
                    )
                    .order_by(
                        TelemetryEventRecord.event_time,
                        TelemetryEventRecord.sequence,
                        TelemetryEventRecord.event_id,
                    )
                )
            )
            source_identity = [
                {"event_id": item.event_id, "digest": item.payload_digest} for item in events
            ]
            source_digest = _digest(source_identity)
            feature_snapshot_id = "feature-" + _digest(
                {
                    "asset_id": asset_id,
                    "schema_version": schema_version,
                    "window_start": start.isoformat(),
                    "window_end": end.isoformat(),
                    "source_digest": source_digest,
                    "include_too_late": include_too_late,
                },
                prefix=False,
            )
            existing = session.scalar(
                select(TelemetryWindowRecord).where(
                    TelemetryWindowRecord.tenant_id == identity.tenant_id,
                    TelemetryWindowRecord.feature_snapshot_id == feature_snapshot_id,
                )
            )
            if existing is not None:
                return _window_view(existing)

            features = _window_features(events, schema)
            present = frozenset(features)
            missing = tuple(sorted(schema.required_signals - present))
            if len(events) < _MIN_SAMPLES:
                quality = "INSUFFICIENT"
            elif missing:
                quality = "DEGRADED"
            else:
                quality = "READY"
            previous = session.scalar(
                select(TelemetryWindowRecord)
                .where(
                    TelemetryWindowRecord.tenant_id == identity.tenant_id,
                    TelemetryWindowRecord.asset_id == asset_id,
                    TelemetryWindowRecord.schema_version == schema_version,
                    TelemetryWindowRecord.window_start == start,
                    TelemetryWindowRecord.window_end == end,
                )
                .order_by(TelemetryWindowRecord.created_at.desc())
            )
            refs = tuple(
                dict.fromkeys(
                    f"telemetry://{item.source_system}/{item.source_ref}" for item in events
                )
            )[:128]
            record = TelemetryWindowRecord(
                window_id=f"window-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                asset_id=asset_id,
                schema_version=schema_version,
                window_start=start,
                window_end=end,
                feature_snapshot_id=feature_snapshot_id,
                source_digest=source_digest,
                source_event_ids_json=[item.event_id for item in events],
                features_json=features,
                missing_signals_json=list(missing),
                supporting_signal_refs_json=list(refs),
                sample_count=len(events),
                quality_status=quality,
                includes_too_late=include_too_late,
                supersedes_window_id=previous.window_id if previous else None,
                created_by_subject_id=identity.subject_id,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            return _window_view(record)

    def detect_window(
        self,
        identity: IdentityContext,
        window_id: str,
        *,
        request_id: str,
    ) -> DetectionResult:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            window = _window_or_hidden(session, identity.tenant_id, window_id)
            self._authorizer.require(
                identity,
                Action.RUN_ANOMALY_DETECTION,
                ResourceContext(identity.tenant_id, window_id, window.asset_id),
                request_id=request_id,
            )
            if window.quality_status == "INSUFFICIENT":
                raise PredictiveMaintenanceConflict("telemetry_window_insufficient")
            model_release_id = _RULE_MODEL_RELEASE
            detector_kind = "RULE_TREND"
            threshold = _ALERT_THRESHOLD
            threshold_policy_id = _THRESHOLD_POLICY
            fallback_reason: str | None = None
            detection = None
            if self._timeseries_detector is not None:
                events = list(
                    session.scalars(
                        select(TelemetryEventRecord).where(
                            TelemetryEventRecord.tenant_id == identity.tenant_id,
                            TelemetryEventRecord.event_id.in_(window.source_event_ids_json),
                        )
                    )
                )
                events.sort(key=lambda item: (_stored_utc(item.event_time), item.sequence))
                if len(events) != len(window.source_event_ids_json):
                    raise PredictiveMaintenanceConflict("telemetry_window_source_event_missing")
                sequence, mask = _inference_sequence(events)
                inference_started = perf_counter()
                try:
                    detection = self._timeseries_detector.detect(
                        identity.tenant_context,
                        TimeseriesDetectionInput(
                            feature_snapshot_id=window.feature_snapshot_id,
                            sequence=sequence,
                            mask=mask,
                        ),
                    )
                except TimeseriesInferenceUnavailable as exc:
                    fallback_reason = exc.reason
                    if self._online_metrics is not None:
                        self._online_metrics.observe_detection(
                            release_id=exc.release_id or "unresolved",
                            status="fallback",
                            seconds=perf_counter() - inference_started,
                            fallback_reason=exc.reason,
                        )
                else:
                    if self._online_metrics is not None:
                        self._online_metrics.observe_detection(
                            release_id=detection.model_release_id,
                            status="success",
                            seconds=perf_counter() - inference_started,
                        )
            if detection is None:
                score, explanations = _rule_score(dict(window.features_json))
                if fallback_reason is not None:
                    explanations = (
                        *explanations,
                        "TIMESERIES_RUNTIME_UNAVAILABLE_RULE_FALLBACK",
                    )
            else:
                model_release_id = detection.model_release_id
                detector_kind = detection.detector_kind
                threshold = detection.threshold
                threshold_policy_id = detection.threshold_policy_id
                score = detection.anomaly_score
                explanations = detection.explanation_codes
            if score < threshold:
                return DetectionResult(
                    window_id=window.window_id,
                    model_release_id=model_release_id,
                    detector_kind=detector_kind,
                    anomaly_score=score,
                    threshold=threshold,
                    threshold_policy_id=threshold_policy_id,
                    explanation_codes=explanations,
                    fallback_reason=fallback_reason,
                    candidate=None,
                )
            existing = session.scalar(
                select(AlertCandidateRecord).where(
                    AlertCandidateRecord.tenant_id == identity.tenant_id,
                    AlertCandidateRecord.window_id == window.window_id,
                    AlertCandidateRecord.threshold_policy_id == threshold_policy_id,
                    AlertCandidateRecord.model_release_id == model_release_id,
                )
            )
            if existing is None:
                existing = AlertCandidateRecord(
                    alert_candidate_id=f"alert-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    asset_id=window.asset_id,
                    window_id=window.window_id,
                    schema_version=window.schema_version,
                    window_start=window.window_start,
                    window_end=window.window_end,
                    feature_snapshot_id=window.feature_snapshot_id,
                    model_release_id=model_release_id,
                    detector_kind=detector_kind,
                    anomaly_score=score,
                    threshold=threshold,
                    threshold_policy_id=threshold_policy_id,
                    supporting_signal_refs_json=list(window.supporting_signal_refs_json),
                    explanation_codes_json=list(explanations),
                    status="PENDING_CONFIRMATION",
                    detected_at=now,
                    reviewed_at=None,
                    reviewed_by_subject_id=None,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
                session.add(existing)
                session.flush()
            candidate = _candidate_view(session, identity.tenant_id, existing)
            return DetectionResult(
                window_id=window.window_id,
                model_release_id=model_release_id,
                detector_kind=detector_kind,
                anomaly_score=score,
                threshold=threshold,
                threshold_policy_id=threshold_policy_id,
                explanation_codes=explanations,
                fallback_reason=fallback_reason,
                candidate=candidate,
            )

    def decide_candidate(
        self,
        identity: IdentityContext,
        alert_candidate_id: str,
        *,
        decision: AlertDecision,
        reason_code: str,
        notes: str | None,
        expected_version: int,
        request_id: str,
    ) -> AlertCandidateView:
        _validate_reference(reason_code, "alert decision reason code")
        normalized_notes = notes.strip() if notes else None
        if normalized_notes is not None and len(normalized_notes) > 2_000:
            raise ValueError("alert decision notes exceed the supported size")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            candidate = _candidate_or_hidden(
                session, identity.tenant_id, alert_candidate_id, for_update=True
            )
            self._authorizer.require(
                identity,
                Action.DECIDE_ALERT_CANDIDATE,
                ResourceContext(identity.tenant_id, alert_candidate_id, candidate.asset_id),
                request_id=request_id,
            )
            if candidate.version != expected_version:
                raise PredictiveMaintenanceConflict("alert_candidate_version_conflict")
            if candidate.status != "PENDING_CONFIRMATION":
                raise PredictiveMaintenanceConflict("alert_candidate_already_decided")
            next_version = candidate.version + 1
            candidate.status = decision.value
            candidate.reviewed_at = now
            candidate.reviewed_by_subject_id = identity.subject_id
            candidate.version = next_version
            candidate.updated_at = now
            session.add(
                AlertCandidateDecisionRecord(
                    decision_id=f"decision-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    alert_candidate_id=candidate.alert_candidate_id,
                    decision=decision.value,
                    reason_code=reason_code,
                    notes=normalized_notes,
                    decided_by_subject_id=identity.subject_id,
                    decided_at=now,
                    candidate_version=next_version,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _candidate_view(session, identity.tenant_id, candidate)

    def link_incident_draft(
        self,
        identity: IdentityContext,
        alert_candidate_id: str,
        *,
        incident_draft_id: str,
        expected_version: int,
        request_id: str,
    ) -> AlertCandidateView:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            candidate = _candidate_or_hidden(
                session, identity.tenant_id, alert_candidate_id, for_update=True
            )
            self._authorizer.require(
                identity,
                Action.REFER_ALERT_CANDIDATE,
                ResourceContext(identity.tenant_id, alert_candidate_id, candidate.asset_id),
                request_id=request_id,
            )
            if candidate.version != expected_version:
                raise PredictiveMaintenanceConflict("alert_candidate_version_conflict")
            if candidate.status != "CONFIRMED":
                raise PredictiveMaintenanceConflict("only_confirmed_alert_can_be_referred")
            draft = session.scalar(
                select(IncidentDraftRecord).where(
                    IncidentDraftRecord.tenant_id == identity.tenant_id,
                    IncidentDraftRecord.draft_id == incident_draft_id,
                )
            )
            if draft is None:
                raise PredictiveMaintenanceNotFound("incident_draft_not_found")
            if draft.asset_id != candidate.asset_id:
                raise PredictiveMaintenanceConflict("incident_draft_asset_mismatch")
            if draft.status not in {"DRAFT", "SUBMITTED"}:
                raise PredictiveMaintenanceConflict("incident_draft_not_active")
            existing = session.scalar(
                select(PredictiveMaintenanceReferralRecord).where(
                    PredictiveMaintenanceReferralRecord.tenant_id == identity.tenant_id,
                    or_(
                        PredictiveMaintenanceReferralRecord.alert_candidate_id
                        == alert_candidate_id,
                        PredictiveMaintenanceReferralRecord.incident_draft_id == incident_draft_id,
                    ),
                )
            )
            if existing is not None:
                raise PredictiveMaintenanceConflict("predictive_referral_already_exists")
            session.add(
                PredictiveMaintenanceReferralRecord(
                    referral_id=f"referral-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    alert_candidate_id=alert_candidate_id,
                    incident_draft_id=incident_draft_id,
                    linked_by_subject_id=identity.subject_id,
                    linked_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            candidate.status = "REFERRED"
            candidate.version += 1
            candidate.updated_at = now
            session.flush()
            return _candidate_view(session, identity.tenant_id, candidate)

    def record_outcome(
        self,
        identity: IdentityContext,
        outcome: PredictiveOutcomeInput,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> PredictiveOutcomeView:
        _validate_outcome(outcome, idempotency_key)
        self._authorizer.require(
            identity,
            Action.RECORD_PREDICTIVE_OUTCOME,
            ResourceContext(identity.tenant_id, outcome.alert_candidate_id, outcome.asset_id),
            request_id=request_id,
        )
        observed_at = _as_utc(outcome.observed_at)
        failure_at = _as_utc(outcome.failure_observed_at) if outcome.failure_observed_at else None
        identity_payload = {
            "asset_id": outcome.asset_id,
            "alert_candidate_id": outcome.alert_candidate_id,
            "outcome_type": outcome.outcome_type.value,
            "observed_at": observed_at.isoformat(),
            "failure_observed_at": failure_at.isoformat() if failure_at else None,
            "work_order_id": outcome.work_order_id,
            "avoided_downtime_minutes": outcome.avoided_downtime_minutes,
            "evidence_ref": outcome.evidence_ref,
            "evidence_digest": outcome.evidence_digest,
        }
        request_hash = _digest(identity_payload)
        now = datetime.now(UTC)
        metric_release_id: str | None = None
        rul_metric: tuple[str, float, bool] | None = None
        rul_calibration: (
            tuple[
                RulForecastCandidateRecord,
                AlertCandidateRecord,
                datetime,
                float,
                float,
                bool,
            ]
            | None
        ) = None
        with self._database.transaction(identity.tenant_context) as session:
            _asset_or_hidden(session, identity.tenant_id, outcome.asset_id)
            existing = session.scalar(
                select(PredictiveOutcomeRecord).where(
                    PredictiveOutcomeRecord.tenant_id == identity.tenant_id,
                    PredictiveOutcomeRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise PredictiveMaintenanceConflict("predictive_outcome_idempotency_conflict")
                return _outcome_view(existing)
            if outcome.alert_candidate_id:
                candidate = _candidate_or_hidden(
                    session, identity.tenant_id, outcome.alert_candidate_id
                )
                metric_release_id = candidate.model_release_id
                if candidate.asset_id != outcome.asset_id:
                    raise PredictiveMaintenanceConflict("predictive_outcome_asset_mismatch")
                if candidate.status == "PENDING_CONFIRMATION":
                    raise PredictiveMaintenanceConflict("predictive_outcome_requires_review")
                previous_label = session.scalar(
                    select(PredictiveOutcomeRecord).where(
                        PredictiveOutcomeRecord.tenant_id == identity.tenant_id,
                        PredictiveOutcomeRecord.alert_candidate_id == outcome.alert_candidate_id,
                    )
                )
                if previous_label is not None:
                    raise PredictiveMaintenanceConflict("predictive_alert_already_labeled")
                if outcome.outcome_type == OutcomeType.TRUE_POSITIVE and failure_at is not None:
                    rul_forecast = session.scalar(
                        select(RulForecastCandidateRecord).where(
                            RulForecastCandidateRecord.tenant_id == identity.tenant_id,
                            RulForecastCandidateRecord.alert_candidate_id
                            == outcome.alert_candidate_id,
                        )
                    )
                    if rul_forecast is not None:
                        actual_minutes = (
                            failure_at - _stored_utc(candidate.detected_at)
                        ).total_seconds() / 60.0
                        if actual_minutes < 0:
                            raise PredictiveMaintenanceConflict(
                                "predictive_outcome_failure_precedes_alert"
                            )
                        absolute_error = abs(actual_minutes - rul_forecast.estimate_minutes)
                        interval_covered = (
                            rul_forecast.lower_bound_minutes
                            <= actual_minutes
                            <= rul_forecast.upper_bound_minutes
                        )
                        rul_calibration = (
                            rul_forecast,
                            candidate,
                            failure_at,
                            actual_minutes,
                            absolute_error,
                            interval_covered,
                        )
                        if rul_forecast.forecast_model_version != _RUL_FORECAST_MODEL:
                            rul_metric = (
                                rul_forecast.source_model_release_id,
                                absolute_error,
                                interval_covered,
                            )
            elif outcome.outcome_type == OutcomeType.MISSED_FAILURE:
                metric_release_id = _active_timeseries_release_id(
                    session,
                    identity.tenant_id,
                )
            if outcome.work_order_id:
                work_order = session.scalar(
                    select(WorkOrderRecord).where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.work_order_id == outcome.work_order_id,
                    )
                )
                if work_order is None:
                    raise PredictiveMaintenanceNotFound("work_order_not_found")
            outcome_id = f"outcome-{uuid4().hex}"
            record = PredictiveOutcomeRecord(
                outcome_id=outcome_id,
                tenant_id=identity.tenant_id,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                asset_id=outcome.asset_id,
                alert_candidate_id=outcome.alert_candidate_id,
                outcome_type=outcome.outcome_type.value,
                observed_at=observed_at,
                failure_observed_at=failure_at,
                work_order_id=outcome.work_order_id,
                avoided_downtime_minutes=outcome.avoided_downtime_minutes,
                evidence_ref=outcome.evidence_ref,
                evidence_digest=outcome.evidence_digest,
                recorded_by_subject_id=identity.subject_id,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            if rul_calibration is not None:
                (
                    forecast,
                    candidate,
                    calibration_failure_at,
                    actual_minutes,
                    absolute_error,
                    interval_covered,
                ) = rul_calibration
                existing_calibrations = list(
                    session.scalars(
                        select(RulForecastCalibrationRecord).where(
                            RulForecastCalibrationRecord.tenant_id == identity.tenant_id,
                            RulForecastCalibrationRecord.source_model_release_id
                            == forecast.source_model_release_id,
                            RulForecastCalibrationRecord.forecast_model_version
                            == forecast.forecast_model_version,
                        )
                    )
                )
                decision_status, decision_reasons = _rul_calibration_decision(
                    [
                        (
                            _stored_utc(item.evaluated_at),
                            item.absolute_error_minutes,
                            item.interval_covered,
                        )
                        for item in existing_calibrations
                    ]
                    + [(calibration_failure_at, absolute_error, interval_covered)]
                )
                calibration_id = f"rul-calibration-{uuid4().hex}"
                calibration_digest = _digest(
                    {
                        "calibration_id": calibration_id,
                        "rul_forecast_id": forecast.rul_forecast_id,
                        "outcome_id": outcome_id,
                        "alert_candidate_id": candidate.alert_candidate_id,
                        "asset_id": candidate.asset_id,
                        "source_model_release_id": forecast.source_model_release_id,
                        "forecast_model_version": forecast.forecast_model_version,
                        "forecast_review_status": forecast.status,
                        "forecast": {
                            "p10_minutes": forecast.lower_bound_minutes,
                            "p50_minutes": forecast.estimate_minutes,
                            "p90_minutes": forecast.upper_bound_minutes,
                            "cohort_digest": forecast.cohort_digest,
                        },
                        "outcome": {
                            "failure_observed_at": calibration_failure_at.isoformat(),
                            "actual_minutes": round(actual_minutes, 6),
                            "evidence_digest": outcome.evidence_digest,
                        },
                        "absolute_error_minutes": round(absolute_error, 6),
                        "interval_covered": interval_covered,
                        "calibration_status": decision_status,
                        "decision_reasons": list(decision_reasons),
                    }
                )
                session.add(
                    RulForecastCalibrationRecord(
                        calibration_id=calibration_id,
                        tenant_id=identity.tenant_id,
                        rul_forecast_id=forecast.rul_forecast_id,
                        outcome_id=outcome_id,
                        alert_candidate_id=candidate.alert_candidate_id,
                        asset_id=candidate.asset_id,
                        source_model_release_id=forecast.source_model_release_id,
                        forecast_model_version=forecast.forecast_model_version,
                        forecast_review_status=forecast.status,
                        actual_minutes=round(actual_minutes, 6),
                        absolute_error_minutes=round(absolute_error, 6),
                        interval_covered=interval_covered,
                        calibration_status=decision_status,
                        retraining_recommended=(decision_status == "RETRAINING_RECOMMENDED"),
                        decision_reasons_json=list(decision_reasons),
                        evidence_digest=calibration_digest,
                        evaluated_at=calibration_failure_at,
                        created_at=now,
                        updated_at=now,
                    )
                )
            session.flush()
            result = _outcome_view(record)
        if self._online_metrics is not None and metric_release_id is not None:
            self._online_metrics.observe_outcome(
                release_id=metric_release_id,
                outcome=outcome.outcome_type.value,
            )
        if self._online_metrics is not None and rul_metric is not None:
            self._online_metrics.observe_rul_outcome(
                release_id=rul_metric[0],
                absolute_error_minutes=rul_metric[1],
                interval_covered=rul_metric[2],
            )
        return result

    def create_rul_forecast(
        self,
        identity: IdentityContext,
        alert_candidate_id: str,
        *,
        expected_alert_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> RulForecastView:
        _validate_reference(idempotency_key, "RUL forecast idempotency key")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            candidate = _candidate_or_hidden(
                session, identity.tenant_id, alert_candidate_id, for_update=True
            )
            self._authorizer.require(
                identity,
                Action.GENERATE_RUL_FORECAST,
                ResourceContext(identity.tenant_id, alert_candidate_id, candidate.asset_id),
                request_id=request_id,
            )
            existing = session.scalar(
                select(RulForecastCandidateRecord).where(
                    RulForecastCandidateRecord.tenant_id == identity.tenant_id,
                    RulForecastCandidateRecord.alert_candidate_id == alert_candidate_id,
                )
            )
            if existing is not None:
                if (
                    existing.idempotency_key != idempotency_key
                    or existing.created_by_subject_id != identity.subject_id
                ):
                    raise PredictiveMaintenanceConflict("rul_forecast_already_exists")
                return _rul_forecast_view(existing)
            if candidate.version != expected_alert_version:
                raise PredictiveMaintenanceConflict("alert_candidate_version_conflict")
            if candidate.status not in {"CONFIRMED", "REFERRED"}:
                raise PredictiveMaintenanceConflict("rul_forecast_requires_confirmed_alert")

            outcomes = list(
                session.scalars(
                    select(PredictiveOutcomeRecord).where(
                        PredictiveOutcomeRecord.tenant_id == identity.tenant_id
                    )
                )
            )
            labeled = sum(
                item.outcome_type in {"TRUE_POSITIVE", "FALSE_POSITIVE"} for item in outcomes
            )
            observed_failures = sum(
                item.outcome_type in {"TRUE_POSITIVE", "MISSED_FAILURE"} for item in outcomes
            )
            if labeled < _MIN_RUL_LABELED_ALERTS or observed_failures < _MIN_RUL_FAILURES:
                raise PredictiveMaintenanceConflict("rul_research_gate_insufficient")

            true_positive_ids = {
                item.alert_candidate_id
                for item in outcomes
                if item.outcome_type == "TRUE_POSITIVE"
                and item.alert_candidate_id is not None
                and item.failure_observed_at is not None
                and item.alert_candidate_id != alert_candidate_id
            }
            source_candidates = (
                {
                    item.alert_candidate_id: item
                    for item in session.scalars(
                        select(AlertCandidateRecord).where(
                            AlertCandidateRecord.tenant_id == identity.tenant_id,
                            AlertCandidateRecord.alert_candidate_id.in_(true_positive_ids),
                            AlertCandidateRecord.schema_version == candidate.schema_version,
                        )
                    )
                }
                if true_positive_ids
                else {}
            )
            cohort: list[tuple[str, float]] = []
            for outcome in outcomes:
                source = source_candidates.get(outcome.alert_candidate_id or "")
                if source is None or outcome.failure_observed_at is None:
                    continue
                lead = (
                    _stored_utc(outcome.failure_observed_at) - _stored_utc(source.detected_at)
                ).total_seconds() / 60.0
                if lead >= 0 and _stored_utc(outcome.failure_observed_at) <= now:
                    cohort.append((outcome.outcome_id, lead))
            if len(cohort) < _MIN_RUL_FAILURES:
                raise PredictiveMaintenanceConflict("rul_lead_time_cohort_insufficient")
            cohort.sort(key=lambda item: item[0])
            lead_times = sorted(item[1] for item in cohort)
            lower = round(_percentile(lead_times, 0.1), 2)
            estimate = round(_percentile(lead_times, 0.5), 2)
            upper = round(_percentile(lead_times, 0.9), 2)
            source_ids = tuple(item[0] for item in cohort)
            source_model_release_id = candidate.model_release_id
            forecast_model_version = _RUL_FORECAST_MODEL
            forecast_binding: dict[str, Any] = {
                "kind": "EMPIRICAL_BASELINE",
                "model_version": _RUL_FORECAST_MODEL,
            }
            if self._rul_forecaster is not None:
                window = _window_or_hidden(
                    session,
                    identity.tenant_id,
                    candidate.window_id,
                )
                events = list(
                    session.scalars(
                        select(TelemetryEventRecord).where(
                            TelemetryEventRecord.tenant_id == identity.tenant_id,
                            TelemetryEventRecord.event_id.in_(window.source_event_ids_json),
                        )
                    )
                )
                events.sort(key=lambda item: (_stored_utc(item.event_time), item.sequence))
                if len(events) != len(window.source_event_ids_json):
                    raise PredictiveMaintenanceConflict("telemetry_window_source_event_missing")
                sequence, mask = _inference_sequence(events)
                inference_started = perf_counter()
                try:
                    model_forecast = self._rul_forecaster.forecast(
                        identity.tenant_context,
                        RulForecastInput(
                            feature_snapshot_id=candidate.feature_snapshot_id,
                            sequence=sequence,
                            mask=mask,
                        ),
                    )
                except RulInferenceUnavailable as exc:
                    if self._online_metrics is not None:
                        self._online_metrics.observe_rul_forecast(
                            release_id=exc.release_id or "unresolved",
                            status="fallback",
                            seconds=perf_counter() - inference_started,
                            fallback_reason=exc.reason,
                        )
                    forecast_binding["fallback_reason"] = exc.reason
                else:
                    lower = round(model_forecast.p10_minutes, 2)
                    estimate = round(model_forecast.p50_minutes, 2)
                    upper = round(model_forecast.p90_minutes, 2)
                    source_model_release_id = model_forecast.model_release_id
                    forecast_model_version = model_forecast.component_model_id
                    forecast_binding = {
                        "kind": "RUL_TRANSFORMER",
                        "release_id": model_forecast.model_release_id,
                        "component_model_id": model_forecast.component_model_id,
                        "explanation_codes": list(model_forecast.explanation_codes),
                    }
                    if self._online_metrics is not None:
                        self._online_metrics.observe_rul_forecast(
                            release_id=model_forecast.model_release_id,
                            status="success",
                            seconds=perf_counter() - inference_started,
                        )
            cohort_digest = _digest(
                {
                    "schema_version": candidate.schema_version,
                    "cutoff_at": now.isoformat(),
                    "source_alert_model_release_id": candidate.model_release_id,
                    "forecast_binding": forecast_binding,
                    "source_outcomes": [
                        {"outcome_id": outcome_id, "lead_minutes": lead}
                        for outcome_id, lead in cohort
                    ],
                }
            )
            record = RulForecastCandidateRecord(
                rul_forecast_id=f"rul-forecast-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                alert_candidate_id=alert_candidate_id,
                asset_id=candidate.asset_id,
                window_id=candidate.window_id,
                feature_snapshot_id=candidate.feature_snapshot_id,
                source_model_release_id=source_model_release_id,
                forecast_model_version=forecast_model_version,
                estimate_minutes=estimate,
                lower_bound_minutes=lower,
                upper_bound_minutes=upper,
                recommended_inspection_at=_stored_utc(candidate.detected_at)
                + timedelta(minutes=lower),
                historical_failure_count=len(cohort),
                source_outcome_ids_json=list(source_ids),
                cohort_digest=cohort_digest,
                history_cutoff_at=now,
                source_alert_version=candidate.version,
                idempotency_key=idempotency_key,
                status="PENDING_REVIEW",
                created_by_subject_id=identity.subject_id,
                reviewed_by_subject_id=None,
                review_reason=None,
                reviewed_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            return _rul_forecast_view(record)

    def review_rul_forecast(
        self,
        identity: IdentityContext,
        rul_forecast_id: str,
        *,
        decision: RulForecastDecision,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> RulForecastView:
        normalized_reason = " ".join(reason.split())
        if not 3 <= len(normalized_reason) <= 2_000:
            raise ValueError("RUL forecast review reason is invalid")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = _rul_forecast_or_hidden(
                session, identity.tenant_id, rul_forecast_id, for_update=True
            )
            self._authorizer.require(
                identity,
                Action.REVIEW_RUL_FORECAST,
                ResourceContext(identity.tenant_id, rul_forecast_id, record.asset_id),
                request_id=request_id,
            )
            if record.version != expected_version:
                raise PredictiveMaintenanceConflict("rul_forecast_version_conflict")
            if record.status != "PENDING_REVIEW":
                raise PredictiveMaintenanceConflict("rul_forecast_already_reviewed")
            if record.created_by_subject_id == identity.subject_id:
                raise PredictiveMaintenanceConflict("rul_forecast_self_review_forbidden")
            record.status = decision.value
            record.reviewed_by_subject_id = identity.subject_id
            record.review_reason = normalized_reason
            record.reviewed_at = now
            record.version += 1
            record.updated_at = now
            session.flush()
            return _rul_forecast_view(record)

    def overview(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> PredictiveMaintenanceOverview:
        self._authorizer.require(
            identity,
            Action.READ_PREDICTIVE_MAINTENANCE,
            ResourceContext(identity.tenant_id),
            request_id=request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            asset_ids = None if identity.roles & _TENANT_WIDE_ROLES else identity.asset_ids
            events_query = select(TelemetryEventRecord).where(
                TelemetryEventRecord.tenant_id == identity.tenant_id
            )
            windows_query = select(TelemetryWindowRecord).where(
                TelemetryWindowRecord.tenant_id == identity.tenant_id
            )
            candidates_query = select(AlertCandidateRecord).where(
                AlertCandidateRecord.tenant_id == identity.tenant_id
            )
            outcomes_query = select(PredictiveOutcomeRecord).where(
                PredictiveOutcomeRecord.tenant_id == identity.tenant_id
            )
            rul_query = select(RulForecastCandidateRecord).where(
                RulForecastCandidateRecord.tenant_id == identity.tenant_id
            )
            rul_calibration_query = select(RulForecastCalibrationRecord).where(
                RulForecastCalibrationRecord.tenant_id == identity.tenant_id
            )
            if asset_ids is not None:
                events_query = events_query.where(TelemetryEventRecord.asset_id.in_(asset_ids))
                windows_query = windows_query.where(TelemetryWindowRecord.asset_id.in_(asset_ids))
                candidates_query = candidates_query.where(
                    AlertCandidateRecord.asset_id.in_(asset_ids)
                )
                outcomes_query = outcomes_query.where(
                    PredictiveOutcomeRecord.asset_id.in_(asset_ids)
                )
                rul_query = rul_query.where(RulForecastCandidateRecord.asset_id.in_(asset_ids))
                rul_calibration_query = rul_calibration_query.where(
                    RulForecastCalibrationRecord.asset_id.in_(asset_ids)
                )
            events = list(
                session.scalars(
                    events_query.order_by(TelemetryEventRecord.event_time.desc()).limit(100)
                )
            )
            windows = list(
                session.scalars(windows_query.order_by(TelemetryWindowRecord.window_end.desc()))
            )
            candidates = list(
                session.scalars(candidates_query.order_by(AlertCandidateRecord.detected_at.desc()))
            )
            outcomes = list(
                session.scalars(outcomes_query.order_by(PredictiveOutcomeRecord.observed_at.desc()))
            )
            rul_forecasts = list(
                session.scalars(rul_query.order_by(RulForecastCandidateRecord.created_at.desc()))
            )
            rul_calibrations = list(
                session.scalars(
                    rul_calibration_query.order_by(
                        RulForecastCalibrationRecord.evaluated_at.desc(),
                        RulForecastCalibrationRecord.calibration_id,
                    )
                )
            )
            return PredictiveMaintenanceOverview(
                telemetry_contract_version=ROTATING_EQUIPMENT_V1.schema_version,
                threshold_policy_id=_THRESHOLD_POLICY,
                events=tuple(_event_view(item) for item in events),
                windows=tuple(_window_view(item) for item in windows[:100]),
                alert_candidates=tuple(
                    _candidate_view(session, identity.tenant_id, item) for item in candidates[:100]
                ),
                outcomes=tuple(_outcome_view(item) for item in outcomes[:100]),
                rul_forecasts=tuple(_rul_forecast_view(item) for item in rul_forecasts[:100]),
                rul_calibrations=tuple(
                    _rul_calibration_view(item) for item in rul_calibrations[:100]
                ),
                rul_release_calibrations=_rul_release_calibrations(rul_calibrations),
                metrics=_metrics(session, identity.tenant_id, candidates, windows, outcomes),
            )


def _validate_event(
    event: TelemetryEventInput,
) -> tuple[TelemetrySchema, datetime, dict[str, float], tuple[str, ...]]:
    for reference_value, field in (
        (event.event_id, "telemetry event id"),
        (event.asset_id, "telemetry asset id"),
        (event.source_system, "telemetry source system"),
        (event.source_ref, "telemetry source reference"),
    ):
        _validate_reference(reference_value, field)
    if event.sequence < 0 or event.sequence > 9_223_372_036_854_775_807:
        raise ValueError("telemetry sequence is outside the supported range")
    schema = _schema(event.schema_version)
    event_time = _as_utc(event.event_time)
    if not event.measurements or len(event.measurements) > len(schema.ranges):
        raise ValueError("telemetry measurements are empty or oversized")
    normalized: dict[str, float] = {}
    for name, value in event.measurements.items():
        if name not in schema.ranges:
            raise ValueError("telemetry measurement is not in the server-owned schema")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("telemetry measurement must be numeric")
        number = float(value)
        low, high = schema.ranges[name]
        if not math.isfinite(number) or number < low or number > high:
            raise ValueError("telemetry measurement is outside the physical safety range")
        normalized[name] = number
    allowed_flags = {"GOOD", "ESTIMATED", "SENSOR_FAULT", "CALIBRATION_DUE", "MISSING"}
    flags = tuple(sorted(set(event.quality_flags)))
    if len(flags) > 8 or any(item not in allowed_flags for item in flags):
        raise ValueError("telemetry quality flag is not supported")
    return schema, event_time, normalized, flags


def _validate_outcome(outcome: PredictiveOutcomeInput, idempotency_key: str) -> None:
    _validate_reference(idempotency_key, "predictive outcome idempotency key")
    _validate_reference(outcome.asset_id, "predictive outcome asset id")
    _validate_reference(outcome.evidence_ref, "predictive outcome evidence reference")
    if not _DIGEST.fullmatch(outcome.evidence_digest):
        raise ValueError("predictive outcome evidence digest is invalid")
    if outcome.alert_candidate_id:
        _validate_reference(outcome.alert_candidate_id, "predictive outcome alert candidate id")
    if outcome.work_order_id:
        _validate_reference(outcome.work_order_id, "predictive outcome work order id")
    if not 0 <= outcome.avoided_downtime_minutes <= 10_000_000:
        raise ValueError("avoided downtime is outside the supported range")
    if (
        outcome.outcome_type in {OutcomeType.TRUE_POSITIVE, OutcomeType.FALSE_POSITIVE}
        and not outcome.alert_candidate_id
    ):
        raise ValueError("labeled alert outcome requires an alert candidate")
    if outcome.outcome_type == OutcomeType.MISSED_FAILURE and outcome.alert_candidate_id:
        raise ValueError("missed failure cannot reference an alert candidate")
    if (
        outcome.outcome_type in {OutcomeType.TRUE_POSITIVE, OutcomeType.MISSED_FAILURE}
        and outcome.failure_observed_at is None
    ):
        raise ValueError("failure outcome requires failure_observed_at")
    if outcome.outcome_type == OutcomeType.FALSE_POSITIVE and outcome.failure_observed_at:
        raise ValueError("false-positive outcome cannot carry failure_observed_at")


def _classify_event(
    stream: TelemetryStreamStateRecord | None,
    event_time: datetime,
    sequence: int,
) -> str:
    if stream is None:
        return "ACCEPTED"
    watermark = _stored_utc(stream.watermark_at)
    maximum = _stored_utc(stream.max_event_time)
    if event_time < watermark:
        return "TOO_LATE"
    if event_time < maximum:
        return "OUT_OF_ORDER_ACCEPTED"
    if sequence < stream.max_sequence:
        return "LATE_ACCEPTED"
    return "ACCEPTED"


def _window_features(
    events: list[TelemetryEventRecord],
    schema: TelemetrySchema,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for signal in schema.ranges:
        points = [
            (_stored_utc(item.event_time), float(item.measurements_json[signal]))
            for item in events
            if signal in item.measurements_json
            and "SENSOR_FAULT" not in item.quality_flags_json
            and "MISSING" not in item.quality_flags_json
        ]
        if not points:
            continue
        values = [item[1] for item in points]
        average = sum(values) / len(values)
        variance = sum((item - average) ** 2 for item in values) / len(values)
        duration_minutes = max(
            (points[-1][0] - points[0][0]).total_seconds() / 60.0,
            1.0,
        )
        result[signal] = {
            "count": len(values),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "mean": round(average, 6),
            "last": round(values[-1], 6),
            "stddev": round(math.sqrt(variance), 6),
            "slope_per_minute": round((values[-1] - values[0]) / duration_minutes, 6),
        }
    return result


def _inference_sequence(
    events: list[TelemetryEventRecord],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[int, ...], ...]]:
    sequence: list[tuple[float, ...]] = []
    mask: list[tuple[int, ...]] = []
    for event in events:
        invalid = set(str(item) for item in event.quality_flags_json) & {
            "SENSOR_FAULT",
            "MISSING",
        }
        values: list[float] = []
        present: list[int] = []
        for signal in SIGNAL_ORDER:
            available = not invalid and signal in event.measurements_json
            values.append(float(event.measurements_json[signal]) if available else 0.0)
            present.append(1 if available else 0)
        sequence.append(tuple(values))
        mask.append(tuple(present))
    return tuple(sequence), tuple(mask)


def _rule_score(features: dict[str, Any]) -> tuple[float, tuple[str, ...]]:
    scores: list[tuple[float, str]] = []
    vibration = features.get("vibration_rms_mm_s")
    if isinstance(vibration, dict):
        mean_value = float(vibration.get("mean", 0.0))
        peak = float(vibration.get("max", 0.0))
        slope = float(vibration.get("slope_per_minute", 0.0))
        scores.extend(
            (
                (_normalized(mean_value, 2.8, 7.1), "VIBRATION_MEAN_HIGH"),
                (_normalized(peak, 4.5, 11.2), "VIBRATION_PEAK_HIGH"),
                (_normalized(slope, 0.02, 0.3), "VIBRATION_RISING"),
            )
        )
    temperature = features.get("bearing_temperature_c")
    if isinstance(temperature, dict):
        mean_value = float(temperature.get("mean", 0.0))
        peak = float(temperature.get("max", 0.0))
        slope = float(temperature.get("slope_per_minute", 0.0))
        scores.extend(
            (
                (_normalized(mean_value, 65.0, 95.0), "BEARING_TEMPERATURE_HIGH"),
                (_normalized(peak, 75.0, 110.0), "BEARING_TEMPERATURE_PEAK"),
                (_normalized(slope, 0.05, 1.0), "BEARING_TEMPERATURE_RISING"),
            )
        )
    rpm = features.get("rpm")
    if isinstance(rpm, dict) and float(rpm.get("mean", 0.0)) > 0:
        instability = (float(rpm.get("max", 0.0)) - float(rpm.get("min", 0.0))) / float(rpm["mean"])
        scores.append((_normalized(instability, 0.08, 0.35), "RPM_UNSTABLE"))
    if not scores:
        return 0.0, ("NO_SUPPORTED_SIGNAL",)
    ordered = sorted(scores, key=lambda item: item[0], reverse=True)
    score = round(max(item[0] for item in ordered), 6)
    explanations = tuple(item[1] for item in ordered if item[0] >= 0.5)
    return score, explanations or ("WITHIN_RULE_BASELINE",)


def _normalized(value: float, baseline: float, alarm: float) -> float:
    if value <= baseline:
        return 0.0
    return min(1.0, (value - baseline) / (alarm - baseline))


def _metrics(
    session: Session,
    tenant_id: str,
    candidates: list[AlertCandidateRecord],
    windows: list[TelemetryWindowRecord],
    outcomes: list[PredictiveOutcomeRecord],
) -> PredictiveMetrics:
    candidate_ids = {item.alert_candidate_id for item in candidates}
    referrals = (
        list(
            session.scalars(
                select(PredictiveMaintenanceReferralRecord).where(
                    PredictiveMaintenanceReferralRecord.tenant_id == tenant_id,
                    PredictiveMaintenanceReferralRecord.alert_candidate_id.in_(candidate_ids),
                )
            )
        )
        if candidate_ids
        else []
    )
    work_order_count = 0
    for referral in referrals:
        incident = session.scalar(
            select(IncidentRecord).where(
                IncidentRecord.tenant_id == tenant_id,
                IncidentRecord.source_draft_id == referral.incident_draft_id,
            )
        )
        if (
            incident is not None
            and session.scalar(
                select(WorkOrderRecord.work_order_id).where(
                    WorkOrderRecord.tenant_id == tenant_id,
                    WorkOrderRecord.incident_id == incident.incident_id,
                )
            )
            is not None
        ):
            work_order_count += 1
    true_positive = [item for item in outcomes if item.outcome_type == "TRUE_POSITIVE"]
    false_positive = [item for item in outcomes if item.outcome_type == "FALSE_POSITIVE"]
    missed = [item for item in outcomes if item.outcome_type == "MISSED_FAILURE"]
    labeled = len(true_positive) + len(false_positive)
    observed_failures = len(true_positive) + len(missed)
    candidate_map = {item.alert_candidate_id: item for item in candidates}
    lead_times = []
    for outcome in true_positive:
        candidate = candidate_map.get(outcome.alert_candidate_id or "")
        if candidate is not None and outcome.failure_observed_at is not None:
            lead = (
                _stored_utc(outcome.failure_observed_at) - _stored_utc(candidate.detected_at)
            ).total_seconds() / 60.0
            if lead >= 0:
                lead_times.append(lead)
    count = len(candidates)
    window_count = len(windows)
    degraded = sum(item.quality_status != "READY" for item in windows)
    daily_false_positives: dict[str, int] = {}
    for item in false_positive:
        key = f"{item.asset_id}|{_stored_utc(item.observed_at).date().isoformat()}"
        daily_false_positives[key] = daily_false_positives.get(key, 0) + 1
    return PredictiveMetrics(
        alert_candidate_count=count,
        pending_confirmation_count=sum(
            item.status == "PENDING_CONFIRMATION" for item in candidates
        ),
        confirmed_count=sum(item.status in {"CONFIRMED", "REFERRED"} for item in candidates),
        dismissed_count=sum(item.status == "DISMISSED" for item in candidates),
        referred_count=len(referrals),
        work_order_conversion_count=work_order_count,
        work_order_conversion_rate=round(work_order_count / count, 6) if count else None,
        labeled_alert_count=labeled,
        false_positive_rate=round(len(false_positive) / labeled, 6) if labeled else None,
        false_positives_per_device_day=daily_false_positives,
        observed_failure_count=observed_failures,
        missed_failure_count=len(missed),
        miss_rate=round(len(missed) / observed_failures, 6) if observed_failures else None,
        median_lead_time_minutes=round(median(lead_times), 2) if lead_times else None,
        avoided_downtime_minutes=sum(item.avoided_downtime_minutes for item in outcomes),
        degraded_window_count=degraded,
        window_count=window_count,
        missing_signal_window_rate=round(degraded / window_count, 6) if window_count else None,
        drift_status=_drift_status(windows),
        label_sufficiency="SUFFICIENT"
        if observed_failures >= 30 and labeled >= 100
        else "INSUFFICIENT",
        rul_claim_allowed=observed_failures >= 30 and labeled >= 100,
    )


def _drift_status(windows: list[TelemetryWindowRecord]) -> str:
    values = [
        float(item.features_json["vibration_rms_mm_s"]["mean"])
        for item in windows
        if isinstance(item.features_json.get("vibration_rms_mm_s"), dict)
    ]
    if len(values) < 20:
        return "NO_BASELINE"
    recent = values[:10]
    baseline = values[10:]
    baseline_mean = sum(baseline) / len(baseline)
    if baseline_mean <= 0:
        return "NO_BASELINE"
    relative_shift = abs(sum(recent) / len(recent) - baseline_mean) / baseline_mean
    return "REVIEW_REQUIRED" if relative_shift > 0.3 else "STABLE"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile values are empty")
    position = (len(values) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)
    weight = position - lower_index
    return values[lower_index] * (1 - weight) + values[upper_index] * weight


def _rul_calibration_decision(
    samples: list[tuple[datetime, float, bool]],
) -> tuple[str, tuple[str, ...]]:
    ordered = sorted(samples, key=lambda item: item[0])
    if len(ordered) < _MIN_RUL_CALIBRATIONS:
        return "INSUFFICIENT_EVIDENCE", ("MINIMUM_30_LABELED_OUTCOMES_NOT_MET",)
    reference = ordered[:-_RUL_RECENT_CALIBRATIONS]
    recent = ordered[-_RUL_RECENT_CALIBRATIONS:]
    reference_median = median(item[1] for item in reference)
    recent_median = median(item[1] for item in recent)
    recent_coverage = sum(item[2] for item in recent) / len(recent)
    reasons: list[str] = []
    regression_threshold = max(
        reference_median * (1 + _RUL_MAE_RELATIVE_REGRESSION),
        reference_median + _RUL_MAE_ABSOLUTE_REGRESSION_MINUTES,
    )
    if recent_median > regression_threshold:
        reasons.append("RECENT_MEDIAN_MAE_REGRESSION")
    if recent_coverage < _RUL_INTERVAL_COVERAGE_FLOOR:
        reasons.append("RECENT_INTERVAL_COVERAGE_BELOW_FLOOR")
    if reasons:
        return "RETRAINING_RECOMMENDED", tuple(reasons)
    return "STABLE", ("ONLINE_CALIBRATION_WITHIN_GATES",)


def _rul_release_calibrations(
    records: list[RulForecastCalibrationRecord],
) -> tuple[RulReleaseCalibrationView, ...]:
    grouped: dict[tuple[str, str], list[RulForecastCalibrationRecord]] = {}
    for record in records:
        grouped.setdefault(
            (record.source_model_release_id, record.forecast_model_version), []
        ).append(record)

    views: list[RulReleaseCalibrationView] = []
    for (release_id, model_version), group in grouped.items():
        ordered = sorted(
            group,
            key=lambda item: (_stored_utc(item.evaluated_at), item.calibration_id),
        )
        samples = [
            (
                _stored_utc(item.evaluated_at),
                item.absolute_error_minutes,
                item.interval_covered,
            )
            for item in ordered
        ]
        status, reasons = _rul_calibration_decision(samples)
        errors = sorted(item.absolute_error_minutes for item in ordered)
        coverage = sum(item.interval_covered for item in ordered) / len(ordered)
        reference_count = max(0, len(ordered) - _RUL_RECENT_CALIBRATIONS)
        recent_count = min(len(ordered), _RUL_RECENT_CALIBRATIONS)
        reference_errors = [item.absolute_error_minutes for item in ordered[:reference_count]]
        recent = ordered[reference_count:]
        recent_errors = [item.absolute_error_minutes for item in recent]
        recent_coverage = (
            sum(item.interval_covered for item in recent) / len(recent) if recent else None
        )
        latest_at = _stored_utc(ordered[-1].evaluated_at)
        evidence_digest = _digest(
            {
                "kind": "RUL_RELEASE_CALIBRATION_SUMMARY",
                "source_model_release_id": release_id,
                "forecast_model_version": model_version,
                "thresholds": {
                    "minimum_labeled_outcomes": _MIN_RUL_CALIBRATIONS,
                    "recent_outcomes": _RUL_RECENT_CALIBRATIONS,
                    "mae_relative_regression": _RUL_MAE_RELATIVE_REGRESSION,
                    "mae_absolute_regression_minutes": (_RUL_MAE_ABSOLUTE_REGRESSION_MINUTES),
                    "interval_coverage_floor": _RUL_INTERVAL_COVERAGE_FLOOR,
                },
                "status": status,
                "reason_codes": list(reasons),
                "calibrations": [
                    {
                        "calibration_id": item.calibration_id,
                        "evidence_digest": item.evidence_digest,
                    }
                    for item in ordered
                ],
            }
        )
        views.append(
            RulReleaseCalibrationView(
                source_model_release_id=release_id,
                forecast_model_version=model_version,
                labeled_outcome_count=len(ordered),
                reference_outcome_count=reference_count,
                recent_outcome_count=recent_count,
                median_absolute_error_minutes=round(median(errors), 2),
                p90_absolute_error_minutes=round(_percentile(errors, 0.9), 2),
                interval_coverage=round(coverage, 6),
                reference_median_absolute_error_minutes=(
                    round(median(reference_errors), 2) if reference_errors else None
                ),
                recent_median_absolute_error_minutes=(
                    round(median(recent_errors), 2) if recent_errors else None
                ),
                recent_interval_coverage=(
                    round(recent_coverage, 6) if recent_coverage is not None else None
                ),
                status=status,
                retraining_recommended=status == "RETRAINING_RECOMMENDED",
                reason_codes=reasons,
                latest_evaluated_at=latest_at,
                evidence_digest=evidence_digest,
            )
        )
    views.sort(
        key=lambda item: item.latest_evaluated_at or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )
    return tuple(views)


def _candidate_view(
    session: Session,
    tenant_id: str,
    record: AlertCandidateRecord,
) -> AlertCandidateView:
    decision = session.scalar(
        select(AlertCandidateDecisionRecord).where(
            AlertCandidateDecisionRecord.tenant_id == tenant_id,
            AlertCandidateDecisionRecord.alert_candidate_id == record.alert_candidate_id,
        )
    )
    referral = session.scalar(
        select(PredictiveMaintenanceReferralRecord).where(
            PredictiveMaintenanceReferralRecord.tenant_id == tenant_id,
            PredictiveMaintenanceReferralRecord.alert_candidate_id == record.alert_candidate_id,
        )
    )
    return AlertCandidateView(
        alert_candidate_id=record.alert_candidate_id,
        asset_id=record.asset_id,
        window_id=record.window_id,
        schema_version=record.schema_version,
        window_start=_stored_utc(record.window_start),
        window_end=_stored_utc(record.window_end),
        feature_snapshot_id=record.feature_snapshot_id,
        model_release_id=record.model_release_id,
        detector_kind=record.detector_kind,
        anomaly_score=record.anomaly_score,
        threshold=record.threshold,
        threshold_policy_id=record.threshold_policy_id,
        supporting_signal_refs=tuple(record.supporting_signal_refs_json),
        explanation_codes=tuple(record.explanation_codes_json),
        status=record.status,
        detected_at=_stored_utc(record.detected_at),
        reviewed_at=_stored_utc(record.reviewed_at) if record.reviewed_at else None,
        reviewed_by_subject_id=record.reviewed_by_subject_id,
        decision_reason_code=decision.reason_code if decision else None,
        decision_notes=decision.notes if decision else None,
        referral=_referral_view(referral) if referral else None,
        version=record.version,
    )


def _event_view(record: TelemetryEventRecord) -> TelemetryEventView:
    return TelemetryEventView(
        event_id=record.event_id,
        asset_id=record.asset_id,
        schema_version=record.schema_version,
        sequence=record.sequence,
        source_system=record.source_system,
        source_ref=record.source_ref,
        event_time=_stored_utc(record.event_time),
        received_at=_stored_utc(record.received_at),
        measurements={key: float(value) for key, value in record.measurements_json.items()},
        quality_flags=tuple(record.quality_flags_json),
        ingestion_status=record.ingestion_status,
        version=record.version,
    )


def _window_view(record: TelemetryWindowRecord) -> TelemetryWindowView:
    return TelemetryWindowView(
        window_id=record.window_id,
        asset_id=record.asset_id,
        schema_version=record.schema_version,
        window_start=_stored_utc(record.window_start),
        window_end=_stored_utc(record.window_end),
        feature_snapshot_id=record.feature_snapshot_id,
        source_event_ids=tuple(record.source_event_ids_json),
        features=dict(record.features_json),
        missing_signals=tuple(record.missing_signals_json),
        supporting_signal_refs=tuple(record.supporting_signal_refs_json),
        sample_count=record.sample_count,
        quality_status=record.quality_status,
        includes_too_late=record.includes_too_late,
        supersedes_window_id=record.supersedes_window_id,
        version=record.version,
    )


def _referral_view(record: PredictiveMaintenanceReferralRecord) -> MaintenanceReferralView:
    return MaintenanceReferralView(
        referral_id=record.referral_id,
        alert_candidate_id=record.alert_candidate_id,
        incident_draft_id=record.incident_draft_id,
        linked_by_subject_id=record.linked_by_subject_id,
        linked_at=_stored_utc(record.linked_at),
    )


def _outcome_view(record: PredictiveOutcomeRecord) -> PredictiveOutcomeView:
    return PredictiveOutcomeView(
        outcome_id=record.outcome_id,
        asset_id=record.asset_id,
        alert_candidate_id=record.alert_candidate_id,
        outcome_type=record.outcome_type,
        observed_at=_stored_utc(record.observed_at),
        failure_observed_at=(
            _stored_utc(record.failure_observed_at) if record.failure_observed_at else None
        ),
        work_order_id=record.work_order_id,
        avoided_downtime_minutes=record.avoided_downtime_minutes,
        evidence_ref=record.evidence_ref,
        recorded_by_subject_id=record.recorded_by_subject_id,
        version=record.version,
    )


def _rul_forecast_view(record: RulForecastCandidateRecord) -> RulForecastView:
    return RulForecastView(
        rul_forecast_id=record.rul_forecast_id,
        alert_candidate_id=record.alert_candidate_id,
        asset_id=record.asset_id,
        window_id=record.window_id,
        feature_snapshot_id=record.feature_snapshot_id,
        source_model_release_id=record.source_model_release_id,
        forecast_model_version=record.forecast_model_version,
        estimate_minutes=record.estimate_minutes,
        lower_bound_minutes=record.lower_bound_minutes,
        upper_bound_minutes=record.upper_bound_minutes,
        recommended_inspection_at=_stored_utc(record.recommended_inspection_at),
        historical_failure_count=record.historical_failure_count,
        source_outcome_ids=tuple(record.source_outcome_ids_json),
        cohort_digest=record.cohort_digest,
        history_cutoff_at=_stored_utc(record.history_cutoff_at),
        source_alert_version=record.source_alert_version,
        status=record.status,
        created_by_subject_id=record.created_by_subject_id,
        reviewed_by_subject_id=record.reviewed_by_subject_id,
        review_reason=record.review_reason,
        reviewed_at=_stored_utc(record.reviewed_at) if record.reviewed_at else None,
        version=record.version,
    )


def _rul_calibration_view(
    record: RulForecastCalibrationRecord,
) -> RulForecastCalibrationView:
    return RulForecastCalibrationView(
        calibration_id=record.calibration_id,
        rul_forecast_id=record.rul_forecast_id,
        outcome_id=record.outcome_id,
        alert_candidate_id=record.alert_candidate_id,
        asset_id=record.asset_id,
        source_model_release_id=record.source_model_release_id,
        forecast_model_version=record.forecast_model_version,
        forecast_review_status=record.forecast_review_status,
        actual_minutes=record.actual_minutes,
        absolute_error_minutes=record.absolute_error_minutes,
        interval_covered=record.interval_covered,
        calibration_status=record.calibration_status,
        retraining_recommended=record.retraining_recommended,
        decision_reasons=tuple(record.decision_reasons_json),
        evidence_digest=record.evidence_digest,
        evaluated_at=_stored_utc(record.evaluated_at),
    )


def _asset_or_hidden(session: Session, tenant_id: str, asset_id: str) -> AssetRecord:
    record = session.scalar(
        select(AssetRecord).where(
            AssetRecord.tenant_id == tenant_id,
            AssetRecord.asset_id == asset_id,
        )
    )
    if record is None:
        raise PredictiveMaintenanceNotFound("asset_not_found")
    return record


def _active_timeseries_release_id(session: Session, tenant_id: str) -> str | None:
    alias = session.scalar(
        select(ModelAliasRecord).where(
            ModelAliasRecord.tenant_id == tenant_id,
            ModelAliasRecord.alias == "industrial-diagnosis",
            ModelAliasRecord.status == "ACTIVE",
        )
    )
    if alias is None:
        return None
    release = session.scalar(
        select(ModelReleaseRecord).where(
            ModelReleaseRecord.tenant_id == tenant_id,
            ModelReleaseRecord.release_id == alias.active_release_id,
            ModelReleaseRecord.status == "PRODUCTION",
        )
    )
    if release is None:
        return None
    components = release.manifest_json.get("specialized_components")
    return (
        release.release_id
        if isinstance(components, dict) and isinstance(components.get("timeseries"), dict)
        else None
    )


def _window_or_hidden(session: Session, tenant_id: str, window_id: str) -> TelemetryWindowRecord:
    record = session.scalar(
        select(TelemetryWindowRecord).where(
            TelemetryWindowRecord.tenant_id == tenant_id,
            TelemetryWindowRecord.window_id == window_id,
        )
    )
    if record is None:
        raise PredictiveMaintenanceNotFound("telemetry_window_not_found")
    return record


def _candidate_or_hidden(
    session: Session,
    tenant_id: str,
    alert_candidate_id: str,
    *,
    for_update: bool = False,
) -> AlertCandidateRecord:
    query = select(AlertCandidateRecord).where(
        AlertCandidateRecord.tenant_id == tenant_id,
        AlertCandidateRecord.alert_candidate_id == alert_candidate_id,
    )
    if for_update:
        query = query.with_for_update()
    record = session.scalar(query)
    if record is None:
        raise PredictiveMaintenanceNotFound("alert_candidate_not_found")
    return record


def _rul_forecast_or_hidden(
    session: Session,
    tenant_id: str,
    rul_forecast_id: str,
    *,
    for_update: bool = False,
) -> RulForecastCandidateRecord:
    query = select(RulForecastCandidateRecord).where(
        RulForecastCandidateRecord.tenant_id == tenant_id,
        RulForecastCandidateRecord.rul_forecast_id == rul_forecast_id,
    )
    if for_update:
        query = query.with_for_update()
    record = session.scalar(query)
    if record is None:
        raise PredictiveMaintenanceNotFound("rul_forecast_not_found")
    return record


def _schema(schema_version: str) -> TelemetrySchema:
    schema = SCHEMAS.get(schema_version)
    if schema is None:
        raise ValueError("telemetry schema version is not supported")
    return schema


def _validate_reference(value: str, field: str) -> None:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} is invalid")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("predictive-maintenance timestamp must include timezone")
    return value.astimezone(UTC)


def _digest(value: Any, *, prefix: bool = True) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    result = sha256(encoded).hexdigest()
    return f"sha256:{result}" if prefix else result
