"""Business KPI projection from authoritative incident and work-order facts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import floor
from typing import Literal, TypeVar, cast
from uuid import uuid4

from sqlalchemy import and_, func, select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetRecord,
    AssetSiteLinkRecord,
    IncidentControlRecord,
    IncidentRecord,
    IncidentResolutionRecord,
    ServicePerformanceBaselineRecord,
    WorkOrderCompletionRecord,
    WorkOrderRecord,
    WorkOrderReworkRecord,
    WorkOrderVerificationRecord,
)


class ServicePerformanceQueryInvalid(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ServicePerformanceBaselineInvalid(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ServicePerformanceBaselineConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.current_version = current_version


class ServicePerformanceBaselineNotFound(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ServicePerformanceSummary:
    closed_work_order_count: int
    first_time_fix_eligible_count: int
    first_time_fix_count: int
    first_time_fix_rate: float | None
    reworked_work_order_count: int
    rework_rate: float | None
    average_repair_rounds: float | None
    mttr_sample_count: int
    mean_mttr_minutes: float | None
    p50_mttr_minutes: float | None
    p90_mttr_minutes: float | None
    resolution_sample_count: int
    remote_resolution_count: int
    work_order_resolution_count: int
    remote_resolution_rate: float | None
    closed_resolution_count: int
    pending_confirmation_count: int
    confirmation_sample_count: int
    mean_confirmation_minutes: float | None
    p50_confirmation_minutes: float | None
    p90_confirmation_minutes: float | None


@dataclass(frozen=True, slots=True)
class ServicePerformanceSlice:
    dimension: Literal["SITE", "INCIDENT_CATEGORY"]
    key: str
    label: str
    closed_work_order_count: int
    first_time_fix_eligible_count: int
    first_time_fix_count: int
    first_time_fix_rate: float | None
    reworked_work_order_count: int
    mean_mttr_minutes: float | None
    resolution_sample_count: int
    remote_resolution_count: int
    remote_resolution_rate: float | None
    pending_confirmation_count: int


@dataclass(frozen=True, slots=True)
class ServicePerformanceDataQuality:
    status: Literal["COMPLETE", "PARTIAL", "NO_DATA"]
    missing_completion_count: int
    missing_passed_verification_count: int
    invalid_timeline_count: int
    missing_close_control_count: int
    invalid_close_timeline_count: int
    exclusions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ServicePerformanceSnapshot:
    metric_contract_version: str
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    window_basis: str
    resolution_window_basis: str
    mttr_definition: str
    first_time_fix_definition: str
    resolution_definition: str
    baseline_status: Literal["NOT_CONFIGURED"]
    summary: ServicePerformanceSummary
    slices: tuple[ServicePerformanceSlice, ...]
    data_quality: ServicePerformanceDataQuality


@dataclass(frozen=True, slots=True)
class ServicePerformanceBaselineTargets:
    mttr_reduction_rate: float
    first_time_fix_lift: float
    remote_resolution_lift: float


@dataclass(frozen=True, slots=True)
class ServicePerformanceBaselineMetrics:
    mean_mttr_minutes: float
    first_time_fix_rate: float
    remote_resolution_rate: float


@dataclass(frozen=True, slots=True)
class ServicePerformanceBaselineSamples:
    closed_work_order_count: int
    first_time_fix_sample_count: int
    mttr_sample_count: int
    resolution_sample_count: int


@dataclass(frozen=True, slots=True)
class ServicePerformanceBaseline:
    baseline_id: str
    name: str
    scope_type: Literal["TENANT"]
    reference_window_start: datetime
    reference_window_end: datetime
    metric_contract_version: str
    metrics: ServicePerformanceBaselineMetrics
    samples: ServicePerformanceBaselineSamples
    minimum_work_order_samples: int
    minimum_resolution_samples: int
    targets: ServicePerformanceBaselineTargets
    data_quality: ServicePerformanceDataQuality
    content_digest: str
    status: Literal["DRAFT", "ACTIVE", "RETIRED"]
    version: int
    created_by_subject_id: str
    activated_by_subject_id: str | None
    activation_reason: str | None
    activated_at: datetime | None
    retired_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ServicePerformanceTargetComparison:
    metric: Literal["MEAN_MTTR", "FIRST_TIME_FIX_RATE", "REMOTE_RESOLUTION_RATE"]
    baseline_value: float | None
    current_value: float | None
    improvement: float | None
    target: float | None
    status: Literal["MET", "NOT_MET", "INSUFFICIENT_DATA"]
    reason: str | None


@dataclass(frozen=True, slots=True)
class ServicePerformanceSnapshotV3:
    metric_contract_version: Literal["service-performance/v3"]
    current: ServicePerformanceSnapshot
    baseline_status: Literal["ACTIVE", "NOT_CONFIGURED", "NOT_EFFECTIVE"]
    baseline: ServicePerformanceBaseline | None
    target_comparisons: tuple[ServicePerformanceTargetComparison, ...]


@dataclass(frozen=True, slots=True)
class _PerformanceFact:
    work_order_id: str
    incident_started_at: datetime
    restored_at: datetime | None
    completion_count: int
    rework_count: int
    site_id: str | None
    site_name: str | None
    category: str | None


@dataclass(frozen=True, slots=True)
class _ResolutionFact:
    incident_id: str
    mode: Literal["REMOTE", "WORK_ORDER"]
    resolved_at: datetime
    closed_at: datetime | None
    incident_status: str
    site_id: str | None
    site_name: str | None
    category: str | None


_FactT = TypeVar("_FactT", _PerformanceFact, _ResolutionFact)
_ComparisonMetric = Literal["MEAN_MTTR", "FIRST_TIME_FIX_RATE", "REMOTE_RESOLUTION_RATE"]


class ServicePerformanceService:
    """Calculate explainable business metrics without copying business facts."""

    CONTRACT_VERSION = "service-performance/v2"
    MAX_WINDOW = timedelta(days=366)

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def snapshot(
        self,
        identity: IdentityContext,
        *,
        window_start: datetime,
        window_end: datetime,
        request_id: str,
    ) -> ServicePerformanceSnapshot:
        start, end = _utc(window_start), _utc(window_end)
        if end <= start:
            raise ServicePerformanceQueryInvalid("service_performance_window_invalid")
        if end - start > self.MAX_WINDOW:
            raise ServicePerformanceQueryInvalid("service_performance_window_too_large")
        self._authorizer.require(
            identity,
            Action.READ_OPERATIONS,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="service-performance",
            ),
            request_id=request_id,
        )

        with self._database.transaction(identity.tenant_context) as session:
            rows = (
                session.execute(
                    select(
                        WorkOrderRecord.work_order_id,
                        IncidentRecord.created_at,
                        IncidentRecord.category,
                        AssetSiteLinkRecord.site_id,
                        AssetSiteLinkRecord.site_name,
                    )
                    .join(
                        IncidentRecord,
                        IncidentRecord.incident_id == WorkOrderRecord.incident_id,
                    )
                    .join(AssetRecord, AssetRecord.asset_id == IncidentRecord.asset_id)
                    .outerjoin(
                        AssetSiteLinkRecord,
                        and_(
                            AssetSiteLinkRecord.tenant_id == identity.tenant_id,
                            AssetSiteLinkRecord.asset_id == AssetRecord.asset_id,
                        ),
                    )
                    .where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.tenant_id == identity.tenant_id,
                        AssetRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.status == "CLOSED",
                        WorkOrderRecord.updated_at >= start,
                        WorkOrderRecord.updated_at < end,
                    )
                    .order_by(WorkOrderRecord.work_order_id)
                )
                .tuples()
                .all()
            )
            work_order_ids = [row[0] for row in rows]
            completion_counts: dict[str, int] = {}
            rework_counts: dict[str, int] = {}
            restored_at: dict[str, datetime] = {}
            if work_order_ids:
                completion_counts = {
                    work_order_id: int(count)
                    for work_order_id, count in session.execute(
                        select(
                            WorkOrderCompletionRecord.work_order_id,
                            func.count(WorkOrderCompletionRecord.completion_id),
                        )
                        .where(
                            WorkOrderCompletionRecord.tenant_id == identity.tenant_id,
                            WorkOrderCompletionRecord.work_order_id.in_(work_order_ids),
                        )
                        .group_by(WorkOrderCompletionRecord.work_order_id)
                    ).tuples()
                }
                rework_counts = {
                    work_order_id: int(count)
                    for work_order_id, count in session.execute(
                        select(
                            WorkOrderReworkRecord.work_order_id,
                            func.count(WorkOrderReworkRecord.rework_id),
                        )
                        .where(
                            WorkOrderReworkRecord.tenant_id == identity.tenant_id,
                            WorkOrderReworkRecord.work_order_id.in_(work_order_ids),
                        )
                        .group_by(WorkOrderReworkRecord.work_order_id)
                    ).tuples()
                }
                restored_at = {
                    work_order_id: _utc(verified_at)
                    for work_order_id, verified_at in session.execute(
                        select(
                            WorkOrderVerificationRecord.work_order_id,
                            func.min(WorkOrderVerificationRecord.verified_at),
                        )
                        .where(
                            WorkOrderVerificationRecord.tenant_id == identity.tenant_id,
                            WorkOrderVerificationRecord.work_order_id.in_(work_order_ids),
                            WorkOrderVerificationRecord.passed.is_(True),
                        )
                        .group_by(WorkOrderVerificationRecord.work_order_id)
                    ).tuples()
                    if verified_at is not None
                }

            resolution_rows = (
                session.execute(
                    select(
                        IncidentResolutionRecord.incident_id,
                        IncidentResolutionRecord.mode,
                        IncidentResolutionRecord.resolved_at,
                        IncidentRecord.status,
                        IncidentRecord.category,
                        AssetSiteLinkRecord.site_id,
                        AssetSiteLinkRecord.site_name,
                    )
                    .join(
                        IncidentRecord,
                        IncidentRecord.incident_id == IncidentResolutionRecord.incident_id,
                    )
                    .join(AssetRecord, AssetRecord.asset_id == IncidentRecord.asset_id)
                    .outerjoin(
                        AssetSiteLinkRecord,
                        and_(
                            AssetSiteLinkRecord.tenant_id == identity.tenant_id,
                            AssetSiteLinkRecord.asset_id == AssetRecord.asset_id,
                        ),
                    )
                    .where(
                        IncidentResolutionRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.tenant_id == identity.tenant_id,
                        AssetRecord.tenant_id == identity.tenant_id,
                        IncidentResolutionRecord.resolved_at >= start,
                        IncidentResolutionRecord.resolved_at < end,
                    )
                    .order_by(IncidentResolutionRecord.incident_id)
                )
                .tuples()
                .all()
            )
            resolution_incident_ids = [row[0] for row in resolution_rows]
            closed_at: dict[str, datetime] = {}
            if resolution_incident_ids:
                closed_at = {
                    incident_id: _utc(occurred_at)
                    for incident_id, occurred_at in session.execute(
                        select(
                            IncidentControlRecord.incident_id,
                            func.min(IncidentControlRecord.occurred_at),
                        )
                        .where(
                            IncidentControlRecord.tenant_id == identity.tenant_id,
                            IncidentControlRecord.incident_id.in_(resolution_incident_ids),
                            IncidentControlRecord.command_type == "CLOSE",
                        )
                        .group_by(IncidentControlRecord.incident_id)
                    ).tuples()
                    if occurred_at is not None
                }

        facts = [
            _PerformanceFact(
                work_order_id=work_order_id,
                incident_started_at=_utc(incident_started_at),
                restored_at=restored_at.get(work_order_id),
                completion_count=completion_counts.get(work_order_id, 0),
                rework_count=rework_counts.get(work_order_id, 0),
                site_id=site_id,
                site_name=site_name,
                category=category,
            )
            for work_order_id, incident_started_at, category, site_id, site_name in rows
        ]
        resolution_facts = [
            _ResolutionFact(
                incident_id=incident_id,
                mode=cast(Literal["REMOTE", "WORK_ORDER"], mode),
                resolved_at=_utc(resolved_at),
                closed_at=closed_at.get(incident_id),
                incident_status=incident_status,
                site_id=site_id,
                site_name=site_name,
                category=category,
            )
            for (
                incident_id,
                mode,
                resolved_at,
                incident_status,
                category,
                site_id,
                site_name,
            ) in resolution_rows
        ]
        return _snapshot(facts, resolution_facts=resolution_facts, start=start, end=end)

    def snapshot_v3(
        self,
        identity: IdentityContext,
        *,
        window_start: datetime,
        window_end: datetime,
        request_id: str,
    ) -> ServicePerformanceSnapshotV3:
        current = self.snapshot(
            identity,
            window_start=window_start,
            window_end=window_end,
            request_id=request_id,
        )
        effective_at = current.window_start
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(ServicePerformanceBaselineRecord)
                .where(
                    ServicePerformanceBaselineRecord.tenant_id == identity.tenant_id,
                    ServicePerformanceBaselineRecord.status.in_(("ACTIVE", "RETIRED")),
                    ServicePerformanceBaselineRecord.activated_at.is_not(None),
                    ServicePerformanceBaselineRecord.activated_at <= effective_at,
                    (
                        ServicePerformanceBaselineRecord.retired_at.is_(None)
                        | (ServicePerformanceBaselineRecord.retired_at > effective_at)
                    ),
                )
                .order_by(ServicePerformanceBaselineRecord.activated_at.desc())
            )
            if record is None:
                approved_count = session.scalar(
                    select(func.count(ServicePerformanceBaselineRecord.baseline_id)).where(
                        ServicePerformanceBaselineRecord.tenant_id == identity.tenant_id,
                        ServicePerformanceBaselineRecord.status.in_(("ACTIVE", "RETIRED")),
                    )
                )
                baseline_status: Literal[
                    "ACTIVE", "NOT_CONFIGURED", "NOT_EFFECTIVE"
                ] = "NOT_EFFECTIVE" if approved_count else "NOT_CONFIGURED"
                return ServicePerformanceSnapshotV3(
                    metric_contract_version="service-performance/v3",
                    current=current,
                    baseline_status=baseline_status,
                    baseline=None,
                    target_comparisons=_target_comparisons(
                        current,
                        None,
                        missing_reason=(
                            "baseline_not_effective"
                            if baseline_status == "NOT_EFFECTIVE"
                            else "baseline_not_configured"
                        ),
                    ),
                )
            baseline = _baseline(record)
        return ServicePerformanceSnapshotV3(
            metric_contract_version="service-performance/v3",
            current=current,
            baseline_status="ACTIVE",
            baseline=baseline,
            target_comparisons=_target_comparisons(current, baseline),
        )

    def list_baselines(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> tuple[ServicePerformanceBaseline, ...]:
        self._authorizer.require(
            identity,
            Action.READ_OPERATIONS,
            ResourceContext(identity.tenant_id, "service-performance-baselines"),
            request_id=request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            records = session.scalars(
                select(ServicePerformanceBaselineRecord)
                .where(ServicePerformanceBaselineRecord.tenant_id == identity.tenant_id)
                .order_by(
                    ServicePerformanceBaselineRecord.created_at.desc(),
                    ServicePerformanceBaselineRecord.baseline_id.desc(),
                )
            ).all()
            return tuple(_baseline(record) for record in records)

    def capture_baseline(
        self,
        identity: IdentityContext,
        *,
        name: str,
        reference_window_start: datetime,
        reference_window_end: datetime,
        minimum_work_order_samples: int,
        minimum_resolution_samples: int,
        targets: ServicePerformanceBaselineTargets,
        request_id: str,
    ) -> ServicePerformanceBaseline:
        self._authorizer.require(
            identity,
            Action.MANAGE_SERVICE_PERFORMANCE_BASELINE,
            ResourceContext(identity.tenant_id, "service-performance-baselines"),
            request_id=request_id,
        )
        normalized_name = name.strip()
        if not normalized_name:
            raise ServicePerformanceBaselineInvalid(
                "service_performance_baseline_name_invalid"
            )
        if minimum_work_order_samples <= 0 or minimum_resolution_samples <= 0:
            raise ServicePerformanceBaselineInvalid(
                "service_performance_baseline_minimum_samples_invalid"
            )
        _validate_targets(targets)
        snapshot = self.snapshot(
            identity,
            window_start=reference_window_start,
            window_end=reference_window_end,
            request_id=request_id,
        )
        if snapshot.data_quality.status != "COMPLETE":
            raise ServicePerformanceBaselineInvalid(
                "service_performance_baseline_evidence_incomplete"
            )
        summary = snapshot.summary
        if (
            summary.first_time_fix_eligible_count < minimum_work_order_samples
            or summary.mttr_sample_count < minimum_work_order_samples
            or summary.resolution_sample_count < minimum_resolution_samples
        ):
            raise ServicePerformanceBaselineInvalid(
                "service_performance_baseline_samples_insufficient"
            )
        if (
            summary.mean_mttr_minutes is None
            or summary.first_time_fix_rate is None
            or summary.remote_resolution_rate is None
        ):
            raise ServicePerformanceBaselineInvalid(
                "service_performance_baseline_evidence_incomplete"
            )

        created_at = datetime.now(UTC)
        quality = _quality_payload(snapshot.data_quality)
        digest = _baseline_digest(
            tenant_id=identity.tenant_id,
            name=normalized_name,
            snapshot=snapshot,
            minimum_work_order_samples=minimum_work_order_samples,
            minimum_resolution_samples=minimum_resolution_samples,
            targets=targets,
        )
        with self._database.transaction(identity.tenant_context) as session:
            record = ServicePerformanceBaselineRecord(
                baseline_id=f"service-performance-baseline-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                name=normalized_name,
                scope_type="TENANT",
                reference_window_start=snapshot.window_start,
                reference_window_end=snapshot.window_end,
                metric_contract_version=snapshot.metric_contract_version,
                mean_mttr_minutes=summary.mean_mttr_minutes,
                first_time_fix_rate=summary.first_time_fix_rate,
                remote_resolution_rate=summary.remote_resolution_rate,
                closed_work_order_count=summary.closed_work_order_count,
                first_time_fix_sample_count=summary.first_time_fix_eligible_count,
                mttr_sample_count=summary.mttr_sample_count,
                resolution_sample_count=summary.resolution_sample_count,
                minimum_work_order_samples=minimum_work_order_samples,
                minimum_resolution_samples=minimum_resolution_samples,
                target_mttr_reduction_rate=targets.mttr_reduction_rate,
                target_first_time_fix_lift=targets.first_time_fix_lift,
                target_remote_resolution_lift=targets.remote_resolution_lift,
                data_quality_status=snapshot.data_quality.status,
                data_quality_json=quality,
                content_digest=digest,
                status="DRAFT",
                version=1,
                created_by_subject_id=identity.subject_id,
                activated_by_subject_id=None,
                activation_reason=None,
                activated_at=None,
                retired_at=None,
                created_at=created_at,
                updated_at=created_at,
            )
            session.add(record)
            session.flush()
            return _baseline(record)

    def activate_baseline(
        self,
        identity: IdentityContext,
        baseline_id: str,
        *,
        expected_version: int,
        reason: str,
        request_id: str,
    ) -> ServicePerformanceBaseline:
        self._authorizer.require(
            identity,
            Action.MANAGE_SERVICE_PERFORMANCE_BASELINE,
            ResourceContext(identity.tenant_id, baseline_id),
            request_id=request_id,
        )
        normalized_reason = reason.strip()
        if len(normalized_reason) < 5:
            raise ServicePerformanceBaselineInvalid(
                "service_performance_baseline_activation_reason_invalid"
            )
        activated_at = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(ServicePerformanceBaselineRecord)
                .where(
                    ServicePerformanceBaselineRecord.tenant_id == identity.tenant_id,
                    ServicePerformanceBaselineRecord.baseline_id == baseline_id,
                )
                .with_for_update()
            )
            if record is None:
                raise ServicePerformanceBaselineNotFound
            if record.version != expected_version:
                raise ServicePerformanceBaselineConflict(
                    "service_performance_baseline_version_conflict",
                    record.version,
                )
            if record.status != "DRAFT":
                raise ServicePerformanceBaselineConflict(
                    "service_performance_baseline_not_draft",
                    record.version,
                )
            if record.created_by_subject_id == identity.subject_id:
                raise ServicePerformanceBaselineConflict(
                    "service_performance_baseline_separation_of_duties_required",
                    record.version,
                )

            active = session.scalar(
                select(ServicePerformanceBaselineRecord)
                .where(
                    ServicePerformanceBaselineRecord.tenant_id == identity.tenant_id,
                    ServicePerformanceBaselineRecord.status == "ACTIVE",
                )
                .with_for_update()
            )
            if active is not None:
                active.status = "RETIRED"
                active.retired_at = activated_at
                active.version += 1
                active.updated_at = activated_at
                session.flush()

            record.status = "ACTIVE"
            record.activated_by_subject_id = identity.subject_id
            record.activation_reason = normalized_reason
            record.activated_at = activated_at
            record.version += 1
            record.updated_at = activated_at
            session.flush()
            return _baseline(record)


def _target_comparisons(
    current: ServicePerformanceSnapshot,
    baseline: ServicePerformanceBaseline | None,
    *,
    missing_reason: str | None = None,
) -> tuple[ServicePerformanceTargetComparison, ...]:
    summary = current.summary
    specs = (
        ("MEAN_MTTR", "mean_mttr_minutes", "mttr_reduction_rate",
         summary.mttr_sample_count, "minimum_work_order_samples", True),
        ("FIRST_TIME_FIX_RATE", "first_time_fix_rate", "first_time_fix_lift",
         summary.first_time_fix_eligible_count, "minimum_work_order_samples", False),
        ("REMOTE_RESOLUTION_RATE", "remote_resolution_rate", "remote_resolution_lift",
         summary.resolution_sample_count, "minimum_resolution_samples", False),
    )
    comparisons = []
    for metric, value_field, target_field, samples, minimum_field, relative in specs:
        current_value = cast(float | None, getattr(summary, value_field))
        baseline_value = (
            cast(float, getattr(baseline.metrics, value_field)) if baseline else None
        )
        target = cast(float, getattr(baseline.targets, target_field)) if baseline else None
        ready = bool(
            baseline
            and samples >= getattr(baseline, minimum_field)
            and current_value is not None
            and (not relative or cast(float, baseline_value) > 0)
        )
        reason = missing_reason
        if (
            reason is None
            and relative
            and baseline_value is not None
            and baseline_value <= 0
        ):
            reason = "baseline_mttr_non_positive"
        if reason is None and (current.data_quality.status != "COMPLETE" or not ready):
            reason = reason or "current_samples_insufficient"
        improvement = None
        if reason is None:
            assert (
                baseline_value is not None
                and current_value is not None
                and target is not None
            )
            improvement = (
                (baseline_value - current_value) / baseline_value
                if relative
                else current_value - baseline_value
            )
        comparisons.append(
            ServicePerformanceTargetComparison(
                metric=cast(_ComparisonMetric, metric),
                baseline_value=baseline_value,
                current_value=current_value,
                improvement=improvement,
                target=target,
                status=(
                    "INSUFFICIENT_DATA"
                    if reason
                    else "MET" if improvement >= target else "NOT_MET"
                ),
                reason=reason,
            )
        )
    return tuple(comparisons)


def _validate_targets(targets: ServicePerformanceBaselineTargets) -> None:
    if any(value < 0 or value > 1 for value in (
        targets.mttr_reduction_rate, targets.first_time_fix_lift,
        targets.remote_resolution_lift,
    )):
        raise ServicePerformanceBaselineInvalid(
            "service_performance_baseline_targets_invalid"
        )


def _quality_payload(quality: ServicePerformanceDataQuality) -> dict[str, object]:
    return {
        "status": quality.status,
        "missing_completion_count": quality.missing_completion_count,
        "missing_passed_verification_count": quality.missing_passed_verification_count,
        "invalid_timeline_count": quality.invalid_timeline_count,
        "missing_close_control_count": quality.missing_close_control_count,
        "invalid_close_timeline_count": quality.invalid_close_timeline_count,
        "exclusions": list(quality.exclusions),
    }


def _baseline_digest(
    *,
    tenant_id: str,
    name: str,
    snapshot: ServicePerformanceSnapshot,
    minimum_work_order_samples: int,
    minimum_resolution_samples: int,
    targets: ServicePerformanceBaselineTargets,
) -> str:
    summary = snapshot.summary
    payload = {
        "tenant_id": tenant_id,
        "name": name,
        "scope_type": "TENANT",
        "reference_window_start": snapshot.window_start.isoformat(),
        "reference_window_end": snapshot.window_end.isoformat(),
        "metric_contract_version": snapshot.metric_contract_version,
        "metrics": {
            "mean_mttr_minutes": summary.mean_mttr_minutes,
            "first_time_fix_rate": summary.first_time_fix_rate,
            "remote_resolution_rate": summary.remote_resolution_rate,
        },
        "samples": {
            "closed_work_order_count": summary.closed_work_order_count,
            "first_time_fix_sample_count": summary.first_time_fix_eligible_count,
            "mttr_sample_count": summary.mttr_sample_count,
            "resolution_sample_count": summary.resolution_sample_count,
        },
        "minimum_work_order_samples": minimum_work_order_samples,
        "minimum_resolution_samples": minimum_resolution_samples,
        "targets": {
            "mttr_reduction_rate": targets.mttr_reduction_rate,
            "first_time_fix_lift": targets.first_time_fix_lift,
            "remote_resolution_lift": targets.remote_resolution_lift,
        },
        "data_quality": _quality_payload(snapshot.data_quality),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"


def _baseline(record: ServicePerformanceBaselineRecord) -> ServicePerformanceBaseline:
    quality = record.data_quality_json
    return ServicePerformanceBaseline(
        baseline_id=record.baseline_id,
        name=record.name,
        scope_type=cast(Literal["TENANT"], record.scope_type),
        reference_window_start=_utc(record.reference_window_start),
        reference_window_end=_utc(record.reference_window_end),
        metric_contract_version=record.metric_contract_version,
        metrics=ServicePerformanceBaselineMetrics(
            mean_mttr_minutes=record.mean_mttr_minutes,
            first_time_fix_rate=record.first_time_fix_rate,
            remote_resolution_rate=record.remote_resolution_rate,
        ),
        samples=ServicePerformanceBaselineSamples(
            closed_work_order_count=record.closed_work_order_count,
            first_time_fix_sample_count=record.first_time_fix_sample_count,
            mttr_sample_count=record.mttr_sample_count,
            resolution_sample_count=record.resolution_sample_count,
        ),
        minimum_work_order_samples=record.minimum_work_order_samples,
        minimum_resolution_samples=record.minimum_resolution_samples,
        targets=ServicePerformanceBaselineTargets(
            mttr_reduction_rate=record.target_mttr_reduction_rate,
            first_time_fix_lift=record.target_first_time_fix_lift,
            remote_resolution_lift=record.target_remote_resolution_lift,
        ),
        data_quality=ServicePerformanceDataQuality(
            status=cast(
                Literal["COMPLETE", "PARTIAL", "NO_DATA"],
                record.data_quality_status,
            ),
            missing_completion_count=int(quality.get("missing_completion_count", 0)),
            missing_passed_verification_count=int(
                quality.get("missing_passed_verification_count", 0)
            ),
            invalid_timeline_count=int(quality.get("invalid_timeline_count", 0)),
            missing_close_control_count=int(
                quality.get("missing_close_control_count", 0)
            ),
            invalid_close_timeline_count=int(
                quality.get("invalid_close_timeline_count", 0)
            ),
            exclusions=tuple(str(item) for item in quality.get("exclusions", [])),
        ),
        content_digest=record.content_digest,
        status=cast(Literal["DRAFT", "ACTIVE", "RETIRED"], record.status),
        version=record.version,
        created_by_subject_id=record.created_by_subject_id,
        activated_by_subject_id=record.activated_by_subject_id,
        activation_reason=record.activation_reason,
        activated_at=_utc(record.activated_at) if record.activated_at is not None else None,
        retired_at=_utc(record.retired_at) if record.retired_at is not None else None,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
    )


def _snapshot(
    facts: list[_PerformanceFact],
    *,
    resolution_facts: list[_ResolutionFact] | None = None,
    start: datetime,
    end: datetime,
) -> ServicePerformanceSnapshot:
    resolutions = resolution_facts or []
    missing_completion_count = sum(item.completion_count == 0 for item in facts)
    missing_verification_count = sum(item.restored_at is None for item in facts)
    invalid_timeline_count = sum(
        item.restored_at is not None and item.restored_at < item.incident_started_at
        for item in facts
    )
    missing_close_control_count = sum(
        item.incident_status == "CLOSED" and item.closed_at is None for item in resolutions
    )
    invalid_close_timeline_count = sum(
        item.closed_at is not None and item.closed_at < item.resolved_at for item in resolutions
    )
    exclusions = tuple(
        reason
        for count, reason in (
            (missing_completion_count, "closed_work_order_missing_completion"),
            (missing_verification_count, "closed_work_order_missing_passed_verification"),
            (invalid_timeline_count, "verification_precedes_incident"),
            (missing_close_control_count, "closed_incident_missing_close_control"),
            (invalid_close_timeline_count, "close_precedes_resolution"),
        )
        if count
    )
    status: Literal["COMPLETE", "PARTIAL", "NO_DATA"] = (
        "NO_DATA" if not facts and not resolutions else "PARTIAL" if exclusions else "COMPLETE"
    )
    slices: list[ServicePerformanceSlice] = []
    for dimension, key_factory in (
        ("SITE", _site_key),
        ("INCIDENT_CATEGORY", _category_key),
    ):
        work_groups = _group(facts, key_factory)
        resolution_groups = _group(resolutions, key_factory)
        for key, label in sorted(set(work_groups) | set(resolution_groups)):
            slices.append(
                _slice(
                    dimension,
                    key,
                    label,
                    work_groups.get((key, label), []),
                    resolution_groups.get((key, label), []),
                )
            )
    return ServicePerformanceSnapshot(
        metric_contract_version=ServicePerformanceService.CONTRACT_VERSION,
        generated_at=datetime.now(UTC),
        window_start=start,
        window_end=end,
        window_basis="work_order_closed_at",
        resolution_window_basis="incident_resolved_at",
        mttr_definition="incident_created_at_to_first_passed_verification",
        first_time_fix_definition="closed_work_order_without_rework_round",
        resolution_definition="registered_remote_or_work_order_resolution",
        baseline_status="NOT_CONFIGURED",
        summary=_summary(facts, resolutions),
        slices=tuple(slices),
        data_quality=ServicePerformanceDataQuality(
            status=status,
            missing_completion_count=missing_completion_count,
            missing_passed_verification_count=missing_verification_count,
            invalid_timeline_count=invalid_timeline_count,
            missing_close_control_count=missing_close_control_count,
            invalid_close_timeline_count=invalid_close_timeline_count,
            exclusions=exclusions,
        ),
    )


def _summary(
    facts: list[_PerformanceFact],
    resolutions: list[_ResolutionFact] | None = None,
) -> ServicePerformanceSummary:
    resolution_facts = resolutions or []
    eligible = [item for item in facts if item.completion_count > 0]
    first_fix_count = sum(item.rework_count == 0 for item in eligible)
    reworked_count = sum(item.rework_count > 0 for item in eligible)
    mttr = _mttr_samples(facts)
    remote_count = sum(item.mode == "REMOTE" for item in resolution_facts)
    work_order_count = sum(item.mode == "WORK_ORDER" for item in resolution_facts)
    confirmation = _confirmation_samples(resolution_facts)
    closed_count = sum(
        item.closed_at is not None and item.closed_at >= item.resolved_at
        for item in resolution_facts
    )
    pending_count = sum(
        item.incident_status == "RESOLVED" and item.closed_at is None
        for item in resolution_facts
    )
    return ServicePerformanceSummary(
        closed_work_order_count=len(facts),
        first_time_fix_eligible_count=len(eligible),
        first_time_fix_count=first_fix_count,
        first_time_fix_rate=_ratio(first_fix_count, len(eligible)),
        reworked_work_order_count=reworked_count,
        rework_rate=_ratio(reworked_count, len(eligible)),
        average_repair_rounds=(
            sum(item.completion_count for item in eligible) / len(eligible) if eligible else None
        ),
        mttr_sample_count=len(mttr),
        mean_mttr_minutes=sum(mttr) / len(mttr) if mttr else None,
        p50_mttr_minutes=_percentile(mttr, 0.5),
        p90_mttr_minutes=_percentile(mttr, 0.9),
        resolution_sample_count=len(resolution_facts),
        remote_resolution_count=remote_count,
        work_order_resolution_count=work_order_count,
        remote_resolution_rate=_ratio(remote_count, len(resolution_facts)),
        closed_resolution_count=closed_count,
        pending_confirmation_count=pending_count,
        confirmation_sample_count=len(confirmation),
        mean_confirmation_minutes=(
            sum(confirmation) / len(confirmation) if confirmation else None
        ),
        p50_confirmation_minutes=_percentile(confirmation, 0.5),
        p90_confirmation_minutes=_percentile(confirmation, 0.9),
    )


def _slice(
    dimension: Literal["SITE", "INCIDENT_CATEGORY"],
    key: str,
    label: str,
    facts: list[_PerformanceFact],
    resolutions: list[_ResolutionFact],
) -> ServicePerformanceSlice:
    summary = _summary(facts, resolutions)
    return ServicePerformanceSlice(
        dimension=dimension,
        key=key,
        label=label,
        closed_work_order_count=summary.closed_work_order_count,
        first_time_fix_eligible_count=summary.first_time_fix_eligible_count,
        first_time_fix_count=summary.first_time_fix_count,
        first_time_fix_rate=summary.first_time_fix_rate,
        reworked_work_order_count=summary.reworked_work_order_count,
        mean_mttr_minutes=summary.mean_mttr_minutes,
        resolution_sample_count=summary.resolution_sample_count,
        remote_resolution_count=summary.remote_resolution_count,
        remote_resolution_rate=summary.remote_resolution_rate,
        pending_confirmation_count=summary.pending_confirmation_count,
    )


def _group(
    facts: list[_FactT],
    key_factory: Callable[[_FactT], tuple[str, str]],
) -> dict[tuple[str, str], list[_FactT]]:
    groups: dict[tuple[str, str], list[_FactT]] = {}
    for item in facts:
        key = key_factory(item)
        groups.setdefault(key, []).append(item)
    return groups


def _site_key(item: _PerformanceFact | _ResolutionFact) -> tuple[str, str]:
    return item.site_id or "UNASSIGNED", item.site_name or "未分配站点"


def _category_key(item: _PerformanceFact | _ResolutionFact) -> tuple[str, str]:
    return item.category or "UNCLASSIFIED", item.category or "未分类故障"


def _mttr_samples(facts: list[_PerformanceFact]) -> list[float]:
    return sorted(
        (item.restored_at - item.incident_started_at).total_seconds() / 60
        for item in facts
        if item.restored_at is not None and item.restored_at >= item.incident_started_at
    )


def _confirmation_samples(facts: list[_ResolutionFact]) -> list[float]:
    return sorted(
        (item.closed_at - item.resolved_at).total_seconds() / 60
        for item in facts
        if item.closed_at is not None and item.closed_at >= item.resolved_at
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * percentile
    lower = floor(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction
