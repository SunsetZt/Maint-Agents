"""Tenant-safe GPU inference health, capacity, and scaling recommendations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import select

from industrial_ops_agent.operations.prometheus import PrometheusReader, PrometheusUnavailable
from industrial_ops_agent.operations.service import DataSourceStatus
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ModelDeploymentRecord,
    ModelReleaseObservationRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext

_PROMETHEUS_LABEL = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class GpuHealth(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    NO_DATA = "NO_DATA"


class ScalingAdvisory(StrEnum):
    HOLD = "HOLD"
    SCALE_OUT_RECOMMENDED = "SCALE_OUT_RECOMMENDED"
    SCALE_IN_CANDIDATE = "SCALE_IN_CANDIDATE"
    INVESTIGATE = "INVESTIGATE"
    NO_DATA = "NO_DATA"


@dataclass(frozen=True, slots=True)
class ClusterGpuPosture:
    health: GpuHealth
    reasons: tuple[str, ...]
    discovered_gpu_count: float | None
    allocatable_gpu_count: float | None
    requested_gpu_count: float | None
    capacity_headroom_gpu: float | None
    average_utilization: float | None
    memory_utilization: float | None
    max_temperature_celsius: float | None
    power_usage_watts: float | None
    xid_errors_15m: float | None
    ecc_dbe_errors_15m: float | None
    dcgm_targets_up_ratio: float | None


@dataclass(frozen=True, slots=True)
class RuntimeGpuPosture:
    deployment_id: str
    release_id: str
    provider: str
    namespace: str
    service_name: str
    deployment_status: str
    desired_stage: str
    current_stage: str
    desired_traffic_percent: float
    observed_traffic_percent: float
    runtime_engine: str | None
    gpu_model: str | None
    gpu_per_replica: float | None
    running_replicas: float | None
    request_rate_per_second: float | None
    error_rate: float | None
    p95_latency_ms: float | None
    gpu_utilization: float | None
    gpu_memory_utilization: float | None
    queue_age_seconds: float | None
    xid_errors_15m: float | None
    latest_observation_at: datetime | None
    latest_observation_decision: str | None
    advisory: ScalingAdvisory
    advisory_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GpuOperationsSnapshot:
    policy_version: str
    generated_at: datetime
    data_source_status: DataSourceStatus
    cluster: ClusterGpuPosture
    runtimes: tuple[RuntimeGpuPosture, ...]
    thresholds: dict[str, float]


@dataclass(slots=True)
class _QueryEvidence:
    successes: int = 0
    missing: int = 0
    failures: int = 0

    def query(self, prometheus: PrometheusReader, query: str) -> float | None:
        try:
            value = prometheus.query_scalar(query)
        except PrometheusUnavailable:
            self.failures += 1
            return None
        if value is None:
            self.missing += 1
            return None
        if not math.isfinite(value):
            self.failures += 1
            return None
        self.successes += 1
        return value

    @property
    def status(self) -> DataSourceStatus:
        if self.successes == 0 and self.failures > 0:
            return DataSourceStatus.UNAVAILABLE
        if self.failures or self.missing:
            return DataSourceStatus.DEGRADED
        return DataSourceStatus.AVAILABLE


CLUSTER_QUERIES: dict[str, str] = {
    "discovered_gpu_count": "count(DCGM_FI_DEV_GPU_UTIL)",
    "allocatable_gpu_count": ('sum(kube_node_status_allocatable{resource="nvidia_com_gpu"})'),
    "requested_gpu_count": ('sum(kube_pod_container_resource_requests{resource="nvidia_com_gpu"})'),
    "average_utilization": "avg(DCGM_FI_DEV_GPU_UTIL) / 100",
    "memory_utilization": ("sum(DCGM_FI_DEV_FB_USED) / clamp_min(sum(DCGM_FI_DEV_FB_TOTAL), 1)"),
    "max_temperature_celsius": "max(DCGM_FI_DEV_GPU_TEMP)",
    "power_usage_watts": "sum(DCGM_FI_DEV_POWER_USAGE)",
    "xid_errors_15m": "sum(increase(DCGM_FI_DEV_XID_ERRORS[15m]))",
    "ecc_dbe_errors_15m": "sum(increase(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL[15m]))",
    "dcgm_targets_up_ratio": 'avg(up{job=~".*dcgm.*"})',
}

THRESHOLDS: dict[str, float] = {
    "gpu_temperature_warning_celsius": 85.0,
    "gpu_temperature_critical_celsius": 90.0,
    "gpu_memory_pressure_ratio": 0.90,
    "gpu_high_utilization_ratio": 0.85,
    "gpu_low_utilization_ratio": 0.25,
    "queue_scale_out_seconds": 10.0,
    "queue_scale_in_seconds": 1.0,
    "error_rate_warning_ratio": 0.01,
}


class GpuOperationsService:
    """Combine immutable rollout records with server-owned Prometheus/DCGM queries."""

    POLICY_VERSION = "gpu-operations/v1"

    def __init__(self, database: Database, prometheus: PrometheusReader) -> None:
        self._database = database
        self._prometheus = prometheus

    def snapshot(
        self,
        tenant_id: str,
        subject_id: str,
        *,
        now: datetime | None = None,
    ) -> GpuOperationsSnapshot:
        generated_at = (now or datetime.now(UTC)).astimezone(UTC)
        deployments, latest_observations = self._load_deployments(
            TenantContext(tenant_id=tenant_id, subject_id=subject_id)
        )
        evidence = _QueryEvidence()
        cluster_values = {
            name: evidence.query(self._prometheus, query) for name, query in CLUSTER_QUERIES.items()
        }
        cluster = _cluster_posture(cluster_values)
        runtimes = tuple(
            self._runtime_posture(
                deployment,
                latest_observations.get(deployment.release_id),
                evidence,
            )
            for deployment in deployments
        )
        return GpuOperationsSnapshot(
            policy_version=self.POLICY_VERSION,
            generated_at=generated_at,
            data_source_status=evidence.status,
            cluster=cluster,
            runtimes=runtimes,
            thresholds=dict(THRESHOLDS),
        )

    def _load_deployments(
        self,
        context: TenantContext,
    ) -> tuple[
        tuple[ModelDeploymentRecord, ...],
        dict[str, ModelReleaseObservationRecord],
    ]:
        with self._database.transaction(context) as session:
            deployments = tuple(
                session.scalars(
                    select(ModelDeploymentRecord)
                    .where(ModelDeploymentRecord.tenant_id == context.tenant_id)
                    .order_by(ModelDeploymentRecord.updated_at.desc())
                )
            )
            if not deployments:
                return (), {}
            release_ids = [item.release_id for item in deployments]
            observations = tuple(
                session.scalars(
                    select(ModelReleaseObservationRecord)
                    .where(
                        ModelReleaseObservationRecord.tenant_id == context.tenant_id,
                        ModelReleaseObservationRecord.release_id.in_(release_ids),
                    )
                    .order_by(
                        ModelReleaseObservationRecord.created_at.desc(),
                        ModelReleaseObservationRecord.sequence.desc(),
                    )
                )
            )
        latest: dict[str, ModelReleaseObservationRecord] = {}
        for observation in observations:
            latest.setdefault(observation.release_id, observation)
        return deployments, latest

    def _runtime_posture(
        self,
        deployment: ModelDeploymentRecord,
        observation: ModelReleaseObservationRecord | None,
        evidence: _QueryEvidence,
    ) -> RuntimeGpuPosture:
        desired = deployment.desired_spec_json
        hardware = _mapping(desired.get("hardware_profile"))
        inference = _mapping(desired.get("inference_config"))
        live = self._runtime_metrics(deployment, evidence)
        advisory, reasons = _scaling_advisory(live)
        return RuntimeGpuPosture(
            deployment_id=deployment.deployment_id,
            release_id=deployment.release_id,
            provider=deployment.provider,
            namespace=deployment.namespace,
            service_name=deployment.service_name,
            deployment_status=deployment.status,
            desired_stage=deployment.desired_stage,
            current_stage=deployment.current_stage,
            desired_traffic_percent=deployment.desired_traffic_percent,
            observed_traffic_percent=deployment.observed_traffic_percent,
            runtime_engine=_optional_string(inference.get("engine")),
            gpu_model=_optional_string(hardware.get("gpu_model")),
            gpu_per_replica=_optional_number(hardware.get("gpu_count")),
            running_replicas=live["running_replicas"],
            request_rate_per_second=live["request_rate_per_second"],
            error_rate=live["error_rate"],
            p95_latency_ms=live["p95_latency_ms"],
            gpu_utilization=live["gpu_utilization"],
            gpu_memory_utilization=live["gpu_memory_utilization"],
            queue_age_seconds=live["queue_age_seconds"],
            xid_errors_15m=live["xid_errors_15m"],
            latest_observation_at=(observation.window_end if observation is not None else None),
            latest_observation_decision=(observation.decision if observation is not None else None),
            advisory=advisory,
            advisory_reasons=reasons,
        )

    def _runtime_metrics(
        self,
        deployment: ModelDeploymentRecord,
        evidence: _QueryEvidence,
    ) -> dict[str, float | None]:
        if not _PROMETHEUS_LABEL.fullmatch(deployment.release_id):
            evidence.failures += 1
            return {name: None for name in _RUNTIME_METRIC_NAMES}
        if not _PROMETHEUS_LABEL.fullmatch(deployment.namespace) or not _PROMETHEUS_LABEL.fullmatch(
            deployment.service_name
        ):
            evidence.failures += 1
            return {name: None for name in _RUNTIME_METRIC_NAMES}
        release = deployment.release_id
        namespace = deployment.namespace
        service = deployment.service_name
        queries = {
            "running_replicas": (
                "count(kube_pod_status_phase{"
                f'namespace="{namespace}",pod=~"{service}.*",phase="Running"}})'
            ),
            "request_rate_per_second": (
                f'sum(rate(ioap_inference_requests_total{{release_id="{release}"}}[5m]))'
            ),
            "error_rate": (
                "sum(rate(ioap_inference_requests_total{"
                f'release_id="{release}",status="error"}}[5m])) '
                "/ clamp_min(sum(rate(ioap_inference_requests_total{"
                f'release_id="{release}"}}[5m])), 0.000001)'
            ),
            "p95_latency_ms": (
                "histogram_quantile(0.95, sum(rate(ioap_inference_latency_seconds_bucket{"
                f'release_id="{release}"}}[5m])) by (le)) * 1000'
            ),
            "gpu_utilization": (f'avg(DCGM_FI_DEV_GPU_UTIL{{release_id="{release}"}}) / 100'),
            "gpu_memory_utilization": (
                f'max(DCGM_FI_DEV_FB_USED{{release_id="{release}"}} '
                f'/ clamp_min(DCGM_FI_DEV_FB_TOTAL{{release_id="{release}"}}, 1))'
            ),
            "queue_age_seconds": (
                f'max(ioap_inference_queue_oldest_age_seconds{{release_id="{release}"}})'
            ),
            "xid_errors_15m": (
                f'sum(increase(DCGM_FI_DEV_XID_ERRORS{{release_id="{release}"}}[15m]))'
            ),
        }
        return {name: evidence.query(self._prometheus, query) for name, query in queries.items()}


_RUNTIME_METRIC_NAMES = (
    "running_replicas",
    "request_rate_per_second",
    "error_rate",
    "p95_latency_ms",
    "gpu_utilization",
    "gpu_memory_utilization",
    "queue_age_seconds",
    "xid_errors_15m",
)


def _cluster_posture(values: dict[str, float | None]) -> ClusterGpuPosture:
    reasons: list[str] = []
    allocatable = values["allocatable_gpu_count"]
    requested = values["requested_gpu_count"]
    headroom = (
        allocatable - requested if allocatable is not None and requested is not None else None
    )
    if all(value is None for value in values.values()):
        health = GpuHealth.NO_DATA
        reasons.append("gpu_metrics_unavailable")
    elif _positive(values["xid_errors_15m"]):
        health = GpuHealth.CRITICAL
        reasons.append("gpu_xid_errors_detected")
    elif _positive(values["ecc_dbe_errors_15m"]):
        health = GpuHealth.CRITICAL
        reasons.append("gpu_ecc_double_bit_errors_detected")
    elif _at_least(
        values["max_temperature_celsius"], THRESHOLDS["gpu_temperature_critical_celsius"]
    ):
        health = GpuHealth.CRITICAL
        reasons.append("gpu_temperature_critical")
    else:
        if _at_least(
            values["max_temperature_celsius"], THRESHOLDS["gpu_temperature_warning_celsius"]
        ):
            reasons.append("gpu_temperature_warning")
        if _at_least(values["memory_utilization"], THRESHOLDS["gpu_memory_pressure_ratio"]):
            reasons.append("gpu_memory_pressure")
        if headroom is not None and headroom <= 0:
            reasons.append("gpu_capacity_exhausted")
        if values["dcgm_targets_up_ratio"] is not None and values["dcgm_targets_up_ratio"] < 1:
            reasons.append("dcgm_target_unavailable")
        if any(value is None for value in values.values()):
            reasons.append("gpu_metrics_partial")
        health = GpuHealth.DEGRADED if reasons else GpuHealth.HEALTHY
    return ClusterGpuPosture(
        health=health,
        reasons=tuple(reasons),
        capacity_headroom_gpu=headroom,
        **values,
    )


def _scaling_advisory(
    metrics: dict[str, float | None],
) -> tuple[ScalingAdvisory, tuple[str, ...]]:
    if all(value is None for value in metrics.values()):
        return ScalingAdvisory.NO_DATA, ("runtime_metrics_unavailable",)
    if _positive(metrics["xid_errors_15m"]):
        return ScalingAdvisory.INVESTIGATE, ("gpu_xid_errors_detected",)
    if _at_least(metrics["error_rate"], THRESHOLDS["error_rate_warning_ratio"]):
        return ScalingAdvisory.INVESTIGATE, ("inference_error_rate_high",)
    scale_out: list[str] = []
    if _at_least(metrics["queue_age_seconds"], THRESHOLDS["queue_scale_out_seconds"]):
        scale_out.append("inference_queue_age_high")
    if _at_least(metrics["gpu_memory_utilization"], THRESHOLDS["gpu_memory_pressure_ratio"]):
        scale_out.append("gpu_memory_pressure")
    if _at_least(metrics["gpu_utilization"], THRESHOLDS["gpu_high_utilization_ratio"]):
        scale_out.append("gpu_utilization_high")
    if scale_out:
        return ScalingAdvisory.SCALE_OUT_RECOMMENDED, tuple(scale_out)
    replicas = metrics["running_replicas"]
    utilization = metrics["gpu_utilization"]
    queue_age = metrics["queue_age_seconds"]
    if (
        replicas is not None
        and replicas > 1
        and utilization is not None
        and utilization < THRESHOLDS["gpu_low_utilization_ratio"]
        and queue_age is not None
        and queue_age < THRESHOLDS["queue_scale_in_seconds"]
    ):
        return ScalingAdvisory.SCALE_IN_CANDIDATE, (
            "gpu_utilization_low",
            "inference_queue_empty",
        )
    if any(value is None for value in metrics.values()):
        return ScalingAdvisory.NO_DATA, ("runtime_metrics_partial",)
    return ScalingAdvisory.HOLD, ("runtime_within_capacity_policy",)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive(value: float | None) -> bool:
    return value is not None and value > 0


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold
