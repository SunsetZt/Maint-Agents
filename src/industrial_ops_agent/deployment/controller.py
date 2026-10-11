"""One-writer reconciliation loop for deployment state and online evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.deployment.prometheus import (
    PrometheusObservationCollector,
    PrometheusUnavailable,
)
from industrial_ops_agent.deployment.service import (
    DeploymentConflict,
    DeploymentProvider,
    ModelDeploymentService,
    OnlineDeploymentStage,
)
from industrial_ops_agent.domain import as_utc as _as_utc


@dataclass(frozen=True, slots=True)
class ControllerReport:
    reconciled: int
    observed: int
    rolled_back: int
    conflicts: int
    dependency_failures: int


@dataclass(slots=True)
class DeploymentController:
    service: ModelDeploymentService
    provider: DeploymentProvider
    collector: PrometheusObservationCollector
    minimum_collection_interval_seconds: int = 60

    def run_once(self, identity: IdentityContext) -> ControllerReport:
        reconciled = 0
        observed = 0
        rolled_back = 0
        conflicts = 0
        dependency_failures = 0

        pending = self.service.list_pending(
            identity,
            request_id=_request_id("list-pending"),
        )
        for deployment in pending:
            try:
                result = self.service.reconcile(
                    identity,
                    deployment.deployment_id,
                    self.provider,
                    expected_version=deployment.version,
                    request_id=_request_id("reconcile"),
                )
                if result.deployment.status == "READY":
                    reconciled += 1
            except DeploymentConflict:
                conflicts += 1

        now = datetime.now(UTC)
        observable = self.service.list_observable(
            identity,
            request_id=_request_id("list-observable"),
        )
        for aggregate in observable:
            deployment = aggregate.deployment
            stage_observations = [
                item for item in aggregate.observations if item.stage == deployment.current_stage
            ]
            latest = stage_observations[-1] if stage_observations else None
            if latest is not None and latest.decision in {"PASS", "ROLLBACK"}:
                continue
            window_start = _as_utc(deployment.updated_at)
            if latest is not None:
                elapsed = (now - _as_utc(latest.window_end)).total_seconds()
                if elapsed < self.minimum_collection_interval_seconds:
                    continue
            if (now - window_start).total_seconds() < self.minimum_collection_interval_seconds:
                continue
            try:
                evidence = self.collector.collect(
                    release_id=deployment.release_id,
                    stage=cast(OnlineDeploymentStage, deployment.current_stage),
                    window_start=window_start,
                    window_end=now,
                    include_timeseries=_manifest_has_timeseries(aggregate.release.manifest_json),
                    include_rul=_manifest_has_rul(aggregate.release.manifest_json),
                )
                result = self.service.record_observation(
                    identity,
                    deployment.deployment_id,
                    evidence,
                    expected_deployment_version=deployment.version,
                    request_id=_request_id("observe"),
                )
                observed += 1
                if result.deployment.desired_stage == "ROLLED_BACK":
                    rollback = self.service.reconcile(
                        identity,
                        result.deployment.deployment_id,
                        self.provider,
                        expected_version=result.deployment.version,
                        request_id=_request_id("automatic-rollback"),
                    )
                    if rollback.deployment.current_stage == "ROLLED_BACK":
                        rolled_back += 1
            except DeploymentConflict:
                conflicts += 1
            except PrometheusUnavailable:
                dependency_failures += 1

        return ControllerReport(
            reconciled=reconciled,
            observed=observed,
            rolled_back=rolled_back,
            conflicts=conflicts,
            dependency_failures=dependency_failures,
        )


def _request_id(operation: str) -> str:
    return f"deployment-controller:{operation}:{uuid4().hex}"


def _manifest_has_timeseries(manifest: dict[str, object]) -> bool:
    components = manifest.get("specialized_components")
    return isinstance(components, dict) and isinstance(components.get("timeseries"), dict)


def _manifest_has_rul(manifest: dict[str, object]) -> bool:
    components = manifest.get("specialized_components")
    return isinstance(components, dict) and isinstance(components.get("rul"), dict)
