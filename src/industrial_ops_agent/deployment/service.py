"""Durable desired-state rollout and evidence-based promotion rules."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.domain.json import strict_document_digest as _digest_json
from industrial_ops_agent.model_gateway.service import MODEL_ALIAS_BY_ENVIRONMENT
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ModelAliasRecord,
    ModelDeploymentRecord,
    ModelGatewayQuotaRecord,
    ModelReleaseApprovalRecord,
    ModelReleaseObservationRecord,
    ModelReleaseRecord,
    ModelReleaseTransitionRecord,
)
from industrial_ops_agent.supply_chain.service import require_media_deployment_supply_chain

DeploymentStage = Literal["SHADOW", "CANARY_5", "CANARY_25", "PRODUCTION", "ROLLED_BACK"]
OnlineDeploymentStage = Literal["SHADOW", "CANARY_5", "CANARY_25", "PRODUCTION"]
ObservationDecision = Literal["PASS", "INSUFFICIENT", "ROLLBACK"]
RoutingMode = Literal["STANDARD", "INFERENCE_POOL"]
AutoscalingMode = Literal["FIXED", "KEDA_VLLM"]

KEDA_AUTOSCALING_POLICY_VERSION = "kserve-gpu-autoscaling/v1"
KEDA_PROMETHEUS_SERVER_ADDRESS = "http://prometheus-prometheus.industrial-observability.svc:9090"
KEDA_VLLM_METRIC_NAME = "vllm:num_requests_running"
KEDA_TARGET_TYPE = "Value"
KEDA_SCALE_UP_STABILIZATION_SECONDS = 0
KEDA_SCALE_DOWN_STABILIZATION_SECONDS = 600
KEDA_SCALE_UP_MAX_PODS_PER_MINUTE = 2
KEDA_SCALE_DOWN_MAX_PODS_PER_TWO_MINUTES = 1
PRODUCTION_MODEL_ALIAS = MODEL_ALIAS_BY_ENVIRONMENT["PRODUCTION"]
STAGING_MODEL_ALIAS = MODEL_ALIAS_BY_ENVIRONMENT["STAGING"]

STAGE_TRAFFIC: dict[str, float] = {
    "SHADOW": 0.0,
    "CANARY_5": 5.0,
    "CANARY_25": 25.0,
    "PRODUCTION": 100.0,
    "ROLLED_BACK": 0.0,
}
NEXT_STAGE = {
    "SHADOW": "CANARY_5",
    "CANARY_5": "CANARY_25",
    "CANARY_25": "PRODUCTION",
}
STAGE_RELEASE_STATUS = {
    "SHADOW": "SHADOW",
    "CANARY_5": "CANARY",
    "CANARY_25": "CANARY",
    "PRODUCTION": "PRODUCTION",
    "ROLLED_BACK": "ROLLED_BACK",
}
STAGE_POLICY: dict[str, dict[str, float | int]] = {
    "SHADOW": {
        "min_window_seconds": 1800,
        "min_request_count": 500,
        "min_critical_case_count": 50,
    },
    "CANARY_5": {
        "min_window_seconds": 3600,
        "min_request_count": 1000,
        "min_critical_case_count": 100,
    },
    "CANARY_25": {
        "min_window_seconds": 7200,
        "min_request_count": 5000,
        "min_critical_case_count": 250,
    },
    "PRODUCTION": {
        "min_window_seconds": 900,
        "min_request_count": 1000,
        "min_critical_case_count": 50,
    },
}
METRIC_THRESHOLDS: dict[str, float] = {
    "cross_tenant_leak_count_max": 0.0,
    "unauthorized_side_effect_count_max": 0.0,
    "high_risk_miss_rate_max": 0.005,
    "task_success_delta_min": -0.02,
    "error_rate_max": 0.01,
    "p95_latency_ms_max": 5000.0,
    "human_edit_rate_delta_max": 0.05,
    "wrong_part_rate_max": 0.005,
    "gpu_xid_error_count_max": 0.0,
    "gpu_memory_utilization_max": 0.95,
    "queue_age_seconds_max": 30.0,
}
REQUIRED_METRICS = frozenset(
    {
        "cross_tenant_leak_count",
        "unauthorized_side_effect_count",
        "high_risk_miss_rate",
        "task_success_delta",
        "error_rate",
        "p95_latency_ms",
        "human_edit_rate_delta",
        "wrong_part_rate",
        "gpu_xid_error_count",
        "gpu_memory_utilization",
        "queue_age_seconds",
    }
)
TIMESERIES_REQUIRED_METRICS = frozenset(
    {
        "timeseries_request_count",
        "timeseries_error_rate",
        "timeseries_p95_latency_ms",
        "timeseries_detection_count",
        "timeseries_fallback_rate",
        "timeseries_labeled_outcome_count",
        "timeseries_false_positive_rate",
        "timeseries_miss_rate",
    }
)
TIMESERIES_METRIC_THRESHOLDS: dict[str, float] = {
    "timeseries_error_rate_max": 0.02,
    "timeseries_p95_latency_ms_max": 500.0,
    "timeseries_fallback_rate_max": 0.05,
    "timeseries_false_positive_rate_max": 0.20,
    "timeseries_miss_rate_max": 0.20,
}
TIMESERIES_STAGE_POLICY: dict[str, dict[str, int]] = {
    "SHADOW": {
        "min_timeseries_request_count": 100,
        "min_timeseries_detection_count": 0,
        "min_timeseries_labeled_outcome_count": 0,
    },
    "CANARY_5": {
        "min_timeseries_request_count": 200,
        "min_timeseries_detection_count": 0,
        "min_timeseries_labeled_outcome_count": 0,
    },
    "CANARY_25": {
        "min_timeseries_request_count": 500,
        "min_timeseries_detection_count": 0,
        "min_timeseries_labeled_outcome_count": 0,
    },
    "PRODUCTION": {
        "min_timeseries_request_count": 500,
        "min_timeseries_detection_count": 100,
        "min_timeseries_labeled_outcome_count": 30,
    },
}
RUL_REQUIRED_METRICS = frozenset(
    {
        "rul_runtime_request_count",
        "rul_runtime_error_rate",
        "rul_runtime_p95_latency_ms",
        "rul_business_forecast_count",
        "rul_fallback_rate",
        "rul_labeled_outcome_count",
        "rul_median_absolute_error_minutes",
        "rul_interval_coverage",
    }
)
RUL_METRIC_THRESHOLDS: dict[str, float] = {
    "rul_runtime_error_rate_max": 0.02,
    "rul_runtime_p95_latency_ms_max": 500.0,
    "rul_fallback_rate_max": 0.05,
    "rul_median_absolute_error_minutes_max": 1440.0,
    "rul_interval_coverage_min": 0.70,
}
RUL_STAGE_POLICY: dict[str, dict[str, int]] = {
    "SHADOW": {
        "min_rul_runtime_request_count": 100,
        "min_rul_business_forecast_count": 0,
        "min_rul_labeled_outcome_count": 0,
    },
    "CANARY_5": {
        "min_rul_runtime_request_count": 200,
        "min_rul_business_forecast_count": 0,
        "min_rul_labeled_outcome_count": 0,
    },
    "CANARY_25": {
        "min_rul_runtime_request_count": 500,
        "min_rul_business_forecast_count": 0,
        "min_rul_labeled_outcome_count": 0,
    },
    "PRODUCTION": {
        "min_rul_runtime_request_count": 500,
        "min_rul_business_forecast_count": 100,
        "min_rul_labeled_outcome_count": 30,
    },
}
DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")


class DeploymentNotVisible(Exception):
    pass


class DeploymentConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


class DeploymentProviderUnavailable(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    namespace: str
    gateway_name: str
    hostname: str
    route_name: str
    stable_service_name: str
    service_account_name: str
    serving_runtime_name: str
    artifact_uri_prefix: str
    routing_mode: RoutingMode = "STANDARD"
    endpoint_picker_service_name: str | None = None
    endpoint_picker_service_port: int | None = None
    autoscaling_mode: AutoscalingMode = "FIXED"
    autoscaling_min_replicas: int | None = None
    autoscaling_max_replicas: int | None = None
    autoscaling_target_running_requests: int | None = None


@dataclass(frozen=True, slots=True)
class ObservationInput:
    stage: OnlineDeploymentStage
    window_start: datetime
    window_end: datetime
    request_count: int
    critical_case_count: int
    metrics: dict[str, float]
    source_refs: dict[str, str]


@dataclass(frozen=True, slots=True)
class ProviderTarget:
    deployment_id: str
    release_id: str
    manifest_hash: str
    namespace: str
    service_name: str
    route_name: str
    stable_service_name: str
    desired_stage: str
    desired_traffic_percent: float
    desired_spec: dict[str, Any]
    desired_spec_hash: str


@dataclass(frozen=True, slots=True)
class ProviderResult:
    ready: bool
    observed_stage: str
    observed_traffic_percent: float
    applied_spec_hash: str | None
    provider_revision: str | None
    endpoint_url: str | None
    reason_code: str | None = None


class DeploymentProvider(Protocol):
    def reconcile(self, target: ProviderTarget) -> ProviderResult: ...


@dataclass(frozen=True, slots=True)
class DeploymentAggregate:
    release: ModelReleaseRecord
    deployment: ModelDeploymentRecord
    observations: tuple[ModelReleaseObservationRecord, ...]


class ModelDeploymentService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def request_shadow(
        self,
        identity: IdentityContext,
        release_id: str,
        plan: DeploymentPlan,
        *,
        expected_release_version: int,
        request_id: str,
    ) -> DeploymentAggregate:
        self._require(identity, Action.REQUEST_MODEL_DEPLOYMENT, release_id, request_id)
        _validate_deployment_plan(plan)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, release_id)
            _require_release_version(release, expected_release_version)
            _require_manifest_integrity(release)
            approval = _approval_or_conflict(session, identity.tenant_id, release_id)
            if release.status != "APPROVAL_PENDING" or approval.status != "APPROVED":
                raise DeploymentConflict("release_approval_required", release.version)
            existing = _deployment_for_release(session, identity.tenant_id, release_id)
            if existing is not None:
                raise DeploymentConflict("model_deployment_already_requested", existing.version)
            _require_media_admission(session, release)
            service_name = f"ioap-{release.manifest_hash[:24]}"
            desired_spec = _build_desired_spec(release, plan, service_name)
            deployment = ModelDeploymentRecord(
                deployment_id=f"model-deployment-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                release_id=release_id,
                provider="KSERVE_GATEWAY_API",
                namespace=plan.namespace,
                service_name=service_name,
                route_name=plan.route_name,
                stable_service_name=plan.stable_service_name,
                status="PENDING",
                desired_stage="SHADOW",
                current_stage="NONE",
                desired_traffic_percent=0.0,
                observed_traffic_percent=0.0,
                desired_spec_json=desired_spec,
                desired_spec_hash=_digest_json(desired_spec),
                applied_spec_hash=None,
                provider_revision=None,
                endpoint_url=None,
                retry_count=0,
                failure_reason=None,
                requested_by_subject_id=identity.subject_id,
                reconciled_by_subject_id=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(deployment)
            session.flush()
            return _aggregate(session, identity.tenant_id, release, deployment)

    def ensure_shadow(
        self,
        identity: IdentityContext,
        release_id: str,
        plan: DeploymentPlan,
        *,
        expected_release_version: int,
        request_id: str,
    ) -> DeploymentAggregate:
        """Create Shadow once and safely replay an identical deployment plan."""

        self._require(identity, Action.REQUEST_MODEL_DEPLOYMENT, release_id, request_id)
        _validate_deployment_plan(plan)
        try:
            existing = self.get(
                identity,
                release_id,
                request_id=f"{request_id}:existing",
            )
        except DeploymentNotVisible:
            try:
                return self.request_shadow(
                    identity,
                    release_id,
                    plan,
                    expected_release_version=expected_release_version,
                    request_id=request_id,
                )
            except DeploymentConflict as exc:
                if exc.reason != "model_deployment_already_requested":
                    raise
                return self.ensure_shadow(
                    identity,
                    release_id,
                    plan,
                    expected_release_version=expected_release_version,
                    request_id=f"{request_id}:concurrent-replay",
                )

        expected_spec = _build_desired_spec(
            existing.release,
            plan,
            existing.deployment.service_name,
        )
        persisted_spec = dict(existing.deployment.desired_spec_json)
        persisted_spec.pop("rollback_reason", None)
        if _digest_json(persisted_spec) != _digest_json(expected_spec):
            raise DeploymentConflict(
                "model_deployment_plan_conflict",
                existing.deployment.version,
            )
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, release_id)
            deployment = _deployment_or_hidden(session, identity.tenant_id, release_id)
            if deployment.version != existing.deployment.version:
                raise DeploymentConflict("model_deployment_plan_conflict", deployment.version)
            _require_media_admission(session, release, deployment, allow_exact_recovery=True)
            return _aggregate(session, identity.tenant_id, release, deployment)

    def get(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        request_id: str,
    ) -> DeploymentAggregate:
        self._require(identity, Action.READ_MODEL_DEPLOYMENT, release_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, release_id)
            deployment = _deployment_for_release(session, identity.tenant_id, release_id)
            if deployment is None:
                raise DeploymentNotVisible
            return _aggregate(session, identity.tenant_id, release, deployment)

    def list(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> Sequence[DeploymentAggregate]:
        self._require(
            identity,
            Action.READ_MODEL_DEPLOYMENT,
            "model-deployments",
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            deployments = list(
                session.scalars(
                    select(ModelDeploymentRecord)
                    .where(ModelDeploymentRecord.tenant_id == identity.tenant_id)
                    .order_by(ModelDeploymentRecord.created_at.desc())
                )
            )
            return [
                _aggregate(
                    session,
                    identity.tenant_id,
                    _release_or_hidden(session, identity.tenant_id, deployment.release_id),
                    deployment,
                )
                for deployment in deployments
            ]

    def list_pending(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> Sequence[ModelDeploymentRecord]:
        self._require(
            identity,
            Action.RECONCILE_MODEL_DEPLOYMENT,
            "model-deployments",
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            return list(
                session.scalars(
                    select(ModelDeploymentRecord)
                    .where(
                        ModelDeploymentRecord.tenant_id == identity.tenant_id,
                        ModelDeploymentRecord.status.in_({"PENDING", "APPLYING", "FAILED"}),
                    )
                    .order_by(ModelDeploymentRecord.updated_at)
                )
            )

    def list_observable(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> Sequence[DeploymentAggregate]:
        """Return ready stages whose telemetry may be collected by the controller."""

        self._require(
            identity,
            Action.RECORD_RELEASE_OBSERVATION,
            "model-deployments",
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            deployments = list(
                session.scalars(
                    select(ModelDeploymentRecord)
                    .where(
                        ModelDeploymentRecord.tenant_id == identity.tenant_id,
                        ModelDeploymentRecord.status == "READY",
                        ModelDeploymentRecord.current_stage.in_(tuple(STAGE_POLICY)),
                    )
                    .order_by(ModelDeploymentRecord.updated_at)
                )
            )
            return [
                _aggregate(
                    session,
                    identity.tenant_id,
                    _release_or_hidden(session, identity.tenant_id, deployment.release_id),
                    deployment,
                )
                for deployment in deployments
            ]

    def promote(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        expected_deployment_version: int,
        request_id: str,
    ) -> DeploymentAggregate:
        self._require(identity, Action.PROMOTE_MODEL_RELEASE, release_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, release_id)
            deployment = _deployment_or_hidden(session, identity.tenant_id, release_id)
            _require_deployment_version(deployment, expected_deployment_version)
            _require_manifest_integrity(release)
            if deployment.status != "READY" or deployment.current_stage != deployment.desired_stage:
                raise DeploymentConflict("deployment_not_ready_for_promotion", deployment.version)
            _require_media_admission(session, release)
            next_stage = NEXT_STAGE.get(deployment.current_stage)
            if next_stage is None:
                raise DeploymentConflict("deployment_has_no_next_stage", deployment.version)
            observation = _latest_observation(
                session, identity.tenant_id, release_id, deployment.current_stage
            )
            if observation is None or observation.decision != "PASS":
                raise DeploymentConflict("passing_stage_observation_required", deployment.version)
            deployment.desired_stage = next_stage
            deployment.desired_traffic_percent = STAGE_TRAFFIC[next_stage]
            deployment.status = "PENDING"
            deployment.failure_reason = None
            deployment.requested_by_subject_id = identity.subject_id
            deployment.version += 1
            session.flush()
            return _aggregate(session, identity.tenant_id, release, deployment)

    def request_rollback(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        expected_deployment_version: int,
        reason_code: str,
        request_id: str,
    ) -> DeploymentAggregate:
        self._require(identity, Action.ROLLBACK_MODEL_RELEASE, release_id, request_id)
        normalized = _validate_reason_code(reason_code)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, release_id)
            deployment = _deployment_or_hidden(session, identity.tenant_id, release_id)
            _require_deployment_version(deployment, expected_deployment_version)
            if deployment.current_stage in {"NONE", "ROLLED_BACK"}:
                raise DeploymentConflict("deployment_not_rollback_ready", deployment.version)
            _require_rollback_media_target(session, release)
            _set_rollback_desired(deployment, normalized, identity.subject_id)
            session.flush()
            return _aggregate(session, identity.tenant_id, release, deployment)

    def reconcile(
        self,
        identity: IdentityContext,
        deployment_id: str,
        provider: DeploymentProvider,
        *,
        expected_version: int,
        request_id: str,
    ) -> DeploymentAggregate:
        self._require(identity, Action.RECONCILE_MODEL_DEPLOYMENT, deployment_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            deployment = _deployment_by_id_or_hidden(session, identity.tenant_id, deployment_id)
            _require_deployment_version(deployment, expected_version)
            if deployment.status not in {"PENDING", "APPLYING", "FAILED"}:
                raise DeploymentConflict("deployment_not_reconcilable", deployment.version)
            release = _release_or_hidden(session, identity.tenant_id, deployment.release_id)
            _require_manifest_integrity(release)
            admission = (
                _require_rollback_media_target(session, release, deployment)
                if deployment.desired_stage == "ROLLED_BACK"
                else _require_media_admission(
                    session, release, deployment, allow_exact_recovery=True
                )
            )
            recovery_endpoint = deployment.endpoint_url
            target = _provider_target(release, deployment)
            deployment.status = "APPLYING"
            deployment.failure_reason = None
            deployment.version += 1
            applying_version = deployment.version

        try:
            result = provider.reconcile(target)
        except DeploymentProviderUnavailable as exc:
            return self._record_provider_failure(
                identity,
                deployment_id,
                applying_version,
                exc.reason,
            )

        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            deployment = _deployment_by_id_or_hidden(session, identity.tenant_id, deployment_id)
            if deployment.version != applying_version or deployment.status != "APPLYING":
                raise DeploymentConflict("deployment_changed_during_reconcile", deployment.version)
            release = _release_or_hidden(session, identity.tenant_id, deployment.release_id)
            try:
                _require_manifest_integrity(release)
                if deployment.desired_stage == "ROLLED_BACK":
                    _require_rollback_media_target(session, release, deployment)
                else:
                    _require_media_admission(
                        session, release, deployment, allow_exact_recovery=True
                    )
                if (
                    admission == "LEGACY_EVIDENCE_REQUIRED"
                    and result.endpoint_url != recovery_endpoint
                ):
                    raise DeploymentConflict("legacy_endpoint_changed")
            except DeploymentConflict as exc:
                deployment.status = "FAILED"
                deployment.failure_reason = exc.reason
                deployment.retry_count += 1
                deployment.version += 1
                session.flush()
                return _aggregate(session, identity.tenant_id, release, deployment)
            deployment.provider_revision = result.provider_revision
            deployment.endpoint_url = result.endpoint_url
            deployment.reconciled_by_subject_id = identity.subject_id
            if not result.ready:
                deployment.failure_reason = result.reason_code
                deployment.version += 1
                session.flush()
                return _aggregate(session, identity.tenant_id, release, deployment)
            if (
                result.observed_stage != deployment.desired_stage
                or result.observed_traffic_percent != deployment.desired_traffic_percent
                or result.applied_spec_hash != deployment.desired_spec_hash
                or not result.endpoint_url
                or not result.endpoint_url.startswith(("http://", "https://"))
            ):
                deployment.status = "FAILED"
                deployment.failure_reason = "provider_observed_state_mismatch"
                deployment.retry_count += 1
                deployment.version += 1
                session.flush()
                return _aggregate(session, identity.tenant_id, release, deployment)

            previous_release_status = release.status
            deployment.status = "READY"
            deployment.current_stage = deployment.desired_stage
            deployment.observed_traffic_percent = result.observed_traffic_percent
            deployment.applied_spec_hash = result.applied_spec_hash
            deployment.failure_reason = None
            deployment.version += 1
            release.status = STAGE_RELEASE_STATUS[deployment.current_stage]
            release.traffic_percent = result.observed_traffic_percent
            release.failure_reason = (
                deployment.desired_spec_json.get("rollback_reason")
                if deployment.current_stage == "ROLLED_BACK"
                else None
            )
            release.version += 1
            _append_transition(
                session,
                release,
                from_status=previous_release_status,
                to_status=release.status,
                actor_subject_id=identity.subject_id,
                reason_code=_applied_reason(deployment.current_stage),
                evidence={
                    "deployment_id": deployment.deployment_id,
                    "desired_spec_hash": deployment.desired_spec_hash,
                    "provider_revision": result.provider_revision,
                    "observed_traffic_percent": result.observed_traffic_percent,
                    "stage": deployment.current_stage,
                },
                occurred_at=now,
            )
            _sync_environment_alias(
                session,
                release,
                deployment,
                previous_release_status=previous_release_status,
                actor_subject_id=identity.subject_id,
                occurred_at=now,
            )
            session.flush()
            return _aggregate(session, identity.tenant_id, release, deployment)

    def record_observation(
        self,
        identity: IdentityContext,
        deployment_id: str,
        observation: ObservationInput,
        *,
        expected_deployment_version: int,
        request_id: str,
    ) -> DeploymentAggregate:
        self._require(
            identity,
            Action.RECORD_RELEASE_OBSERVATION,
            deployment_id,
            request_id,
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            deployment = _deployment_by_id_or_hidden(session, identity.tenant_id, deployment_id)
            _require_deployment_version(deployment, expected_deployment_version)
            release = _release_or_hidden(session, identity.tenant_id, deployment.release_id)
            timeseries_required = _release_has_timeseries(release)
            rul_required = _release_has_rul(release)
            _validate_observation(
                observation,
                timeseries_required=timeseries_required,
                rul_required=rul_required,
            )
            if deployment.status != "READY" or deployment.current_stage != observation.stage:
                raise DeploymentConflict("observation_stage_not_active", deployment.version)
            decision, failure_reasons, thresholds = _evaluate_observation(
                observation,
                timeseries_required=timeseries_required,
                rul_required=rul_required,
            )
            sequence = _next_observation_sequence(
                session, identity.tenant_id, release.release_id, observation.stage
            )
            evidence = {
                "critical_case_count": observation.critical_case_count,
                "deployment_id": deployment.deployment_id,
                "desired_spec_hash": deployment.desired_spec_hash,
                "manifest_hash": release.manifest_hash,
                "metrics": observation.metrics,
                "provider_revision": deployment.provider_revision,
                "request_count": observation.request_count,
                "source_refs": observation.source_refs,
                "stage": observation.stage,
                "thresholds": thresholds,
                "window_end": _as_utc(observation.window_end).isoformat(),
                "window_start": _as_utc(observation.window_start).isoformat(),
            }
            session.add(
                ModelReleaseObservationRecord(
                    observation_id=f"release-observation-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    release_id=release.release_id,
                    deployment_id=deployment.deployment_id,
                    stage=observation.stage,
                    sequence=sequence,
                    window_start=_as_utc(observation.window_start),
                    window_end=_as_utc(observation.window_end),
                    request_count=observation.request_count,
                    critical_case_count=observation.critical_case_count,
                    metrics_json=observation.metrics,
                    thresholds_json=thresholds,
                    source_refs_json=observation.source_refs,
                    evidence_hash=_digest_json(evidence),
                    decision=decision,
                    failure_reasons=failure_reasons,
                    collected_by_subject_id=identity.subject_id,
                    created_at=now,
                    updated_at=now,
                )
            )
            if decision == "ROLLBACK":
                _set_rollback_desired(
                    deployment,
                    "automatic_observation_gate_failed",
                    identity.subject_id,
                )
            session.flush()
            return _aggregate(session, identity.tenant_id, release, deployment)

    def _record_provider_failure(
        self,
        identity: IdentityContext,
        deployment_id: str,
        applying_version: int,
        reason: str,
    ) -> DeploymentAggregate:
        safe_reason = _validate_reason_code(reason)
        with self._database.transaction(identity.tenant_context) as session:
            deployment = _deployment_by_id_or_hidden(session, identity.tenant_id, deployment_id)
            if deployment.version != applying_version or deployment.status != "APPLYING":
                raise DeploymentConflict("deployment_changed_during_reconcile", deployment.version)
            release = _release_or_hidden(session, identity.tenant_id, deployment.release_id)
            deployment.status = "FAILED"
            deployment.failure_reason = safe_reason
            deployment.retry_count += 1
            deployment.reconciled_by_subject_id = identity.subject_id
            deployment.version += 1
            session.flush()
            return _aggregate(session, identity.tenant_id, release, deployment)

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )


def _build_desired_spec(
    release: ModelReleaseRecord,
    plan: DeploymentPlan,
    service_name: str,
) -> dict[str, Any]:
    _validate_deployment_plan(plan)
    manifest = release.manifest_json
    adapter_uri = (
        f"{plan.artifact_uri_prefix.rstrip('/')}/{manifest['adapter']['object_key'].lstrip('/')}"
    )
    quantization = manifest.get("quantization")
    engine = manifest["runtime"]["inference_config"].get("engine")
    if plan.routing_mode == "INFERENCE_POOL" and engine != "vllm":
        raise DeploymentConflict("inference_pool_requires_vllm")
    hardware_profile = manifest["runtime"]["hardware_profile"]
    if plan.autoscaling_mode == "KEDA_VLLM":
        if engine != "vllm":
            raise DeploymentConflict("keda_autoscaling_requires_vllm")
        gpu_count = hardware_profile.get("gpu_count")
        if not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count <= 0:
            raise DeploymentConflict("keda_autoscaling_requires_gpu")
    edge_runtime = engine == "llama.cpp"
    model_uri = None
    model_content_hash = None
    model_file = None
    model_mount_subpath = None
    if edge_runtime:
        if not isinstance(quantization, dict):
            raise DeploymentConflict("llama_cpp_release_requires_GGUF_artifact")
        runtime_metadata = quantization.get("metadata", {}).get("runtime", {})
        model_file = runtime_metadata.get("model_file")
        if not isinstance(model_file, str):
            raise DeploymentConflict("llama_cpp_release_model_file_is_missing")
        storage_object = (
            manifest.get("model_storage_subpath")
            if manifest.get("schema_version") == "project-authorized-local-staging-release/v1"
            else quantization["object_key"]
        )
        if not isinstance(storage_object, str) or not storage_object:
            raise DeploymentConflict("llama_cpp_release_storage_path_is_missing")
        if manifest.get("schema_version") == "project-authorized-local-staging-release/v1":
            model_mount_subpath = storage_object
        model_uri = f"{plan.artifact_uri_prefix.rstrip('/')}/{storage_object.lstrip('/')}"
        model_content_hash = quantization["content_hash"]
    local_runtime_image_reference = None
    local_runtime_git_commit = None
    if manifest.get("schema_version") == "project-authorized-local-staging-release/v1":
        local_runtime_image_reference = manifest["runtime"].get("image_reference")
        local_runtime_git_commit = manifest["runtime"].get("git_commit")
        if (
            not isinstance(local_runtime_image_reference, str)
            or not local_runtime_image_reference
            or not isinstance(local_runtime_git_commit, str)
            or len(local_runtime_git_commit) != 40
        ):
            raise DeploymentConflict("local_runtime_image_binding_is_missing")
    desired_spec: dict[str, Any] = {
        "schema_version": "ioap-kserve-rollout/v2",
        "release_id": release.release_id,
        "manifest_hash": release.manifest_hash,
        "service_account_name": plan.service_account_name,
        "serving_runtime_name": f"{service_name}-runtime",
        "serving_runtime_template": plan.serving_runtime_name,
        "runtime_image_repository": manifest["runtime"]["image_repository"],
        "runtime_image_digest": manifest["runtime"]["image_digest"],
        "base_model_id": manifest["base_model"]["id"],
        "base_model_digest": manifest["base_model"]["digest"],
        "adapter_uri": adapter_uri,
        "adapter_content_hash": manifest["adapter"]["content_hash"],
        "model_uri": model_uri,
        "model_content_hash": model_content_hash,
        "model_file": model_file,
        "inference_config": manifest["runtime"]["inference_config"],
        "hardware_profile": hardware_profile,
        "route": {
            "namespace": plan.namespace,
            "gateway_name": plan.gateway_name,
            "hostname": plan.hostname,
            "route_name": plan.route_name,
            "stable_service_name": plan.stable_service_name,
        },
        "routing": {"mode": "STANDARD"},
    }
    if model_mount_subpath is not None:
        desired_spec["model_mount_subpath"] = model_mount_subpath
    if local_runtime_image_reference is not None:
        desired_spec["runtime_image_reference"] = local_runtime_image_reference
        desired_spec["runtime_git_commit"] = local_runtime_git_commit
    if plan.autoscaling_mode == "KEDA_VLLM":
        desired_spec["schema_version"] = "ioap-kserve-rollout/v3"
        desired_spec["autoscaling"] = {
            "mode": "KEDA_VLLM",
            "policy_version": KEDA_AUTOSCALING_POLICY_VERSION,
            "min_replicas": plan.autoscaling_min_replicas,
            "max_replicas": plan.autoscaling_max_replicas,
            "metric": {
                "backend": "prometheus",
                "name": KEDA_VLLM_METRIC_NAME,
                "server_address": KEDA_PROMETHEUS_SERVER_ADDRESS,
                "query": autoscaling_metric_query(plan.namespace, service_name),
                "target_type": KEDA_TARGET_TYPE,
                "target_value": plan.autoscaling_target_running_requests,
            },
            "behavior": {
                "scale_up_stabilization_seconds": (KEDA_SCALE_UP_STABILIZATION_SECONDS),
                "scale_down_stabilization_seconds": (KEDA_SCALE_DOWN_STABILIZATION_SECONDS),
                "scale_up_max_pods_per_minute": KEDA_SCALE_UP_MAX_PODS_PER_MINUTE,
                "scale_down_max_pods_per_two_minutes": (KEDA_SCALE_DOWN_MAX_PODS_PER_TWO_MINUTES),
            },
        }
    if plan.routing_mode == "INFERENCE_POOL":
        pool_name = f"{service_name}-pool"
        desired_spec["routing"] = {
            "mode": "INFERENCE_POOL",
            "inference_pool_name": pool_name,
            "target_port": 8080,
            "app_protocol": "http",
            "selector": {
                "matchLabels": {
                    "industrial-ops.ai/inference-pool": pool_name,
                    "industrial-ops.ai/release-id": release.release_id[:63],
                }
            },
            "endpoint_picker_ref": {
                "name": plan.endpoint_picker_service_name,
                "port": plan.endpoint_picker_service_port,
                "failure_mode": "FailClose",
            },
        }
    specialized = manifest.get("specialized_components")
    embedding = specialized.get("embedding") if isinstance(specialized, dict) else None
    if isinstance(embedding, dict) and isinstance(embedding.get("runtime"), dict):
        runtime = embedding["runtime"]
        artifact = embedding["artifact"]
        desired_spec["embedding"] = {
            "service_name": f"{service_name}-embedding",
            "stable_service_name": _component_service_name(plan.stable_service_name, "embedding"),
            "serving_runtime_name": f"{service_name}-embedding-runtime",
            "runtime_image_repository": runtime["image_repository"],
            "runtime_image_digest": runtime["image_digest"],
            "artifact_uri": (
                f"{plan.artifact_uri_prefix.rstrip('/')}/{str(artifact['object_key']).lstrip('/')}"
            ),
            "artifact_content_hash": artifact["content_hash"],
            "component_model_id": embedding["runtime_model_id"],
            "inference_config": runtime["inference_config"],
            "hardware_profile": runtime["hardware_profile"],
            "route_path": "/v1/embedding",
        }
    reranker = specialized.get("reranker") if isinstance(specialized, dict) else None
    if isinstance(reranker, dict) and isinstance(reranker.get("runtime"), dict):
        runtime = reranker["runtime"]
        artifact = reranker["artifact"]
        desired_spec["reranker"] = {
            "service_name": f"{service_name}-reranker",
            "stable_service_name": _component_service_name(plan.stable_service_name, "reranker"),
            "serving_runtime_name": f"{service_name}-reranker-runtime",
            "runtime_image_repository": runtime["image_repository"],
            "runtime_image_digest": runtime["image_digest"],
            "artifact_uri": (
                f"{plan.artifact_uri_prefix.rstrip('/')}/{str(artifact['object_key']).lstrip('/')}"
            ),
            "artifact_content_hash": artifact["content_hash"],
            "component_model_id": reranker["runtime_model_id"],
            "inference_config": runtime["inference_config"],
            "hardware_profile": runtime["hardware_profile"],
            "route_path": "/v1/retrieval",
        }
    tts = specialized.get("tts") if isinstance(specialized, dict) else None
    if isinstance(tts, dict) and isinstance(tts.get("runtime"), dict):
        runtime = tts["runtime"]
        artifact = tts["artifact"]
        desired_spec["tts"] = {
            "service_name": f"{service_name}-tts",
            "stable_service_name": _component_service_name(plan.stable_service_name, "tts"),
            "serving_runtime_name": f"{service_name}-tts-runtime",
            "runtime_image_repository": runtime["image_repository"],
            "runtime_image_digest": runtime["image_digest"],
            "artifact_uri": (
                f"{plan.artifact_uri_prefix.rstrip('/')}/{str(artifact['object_key']).lstrip('/')}"
            ),
            "artifact_content_hash": artifact["content_hash"],
            "component_model_id": tts["runtime_model_id"],
            "inference_config": runtime["inference_config"],
            "hardware_profile": runtime["hardware_profile"],
            "route_path": "/v1/audio/speech",
        }
    timeseries = specialized.get("timeseries") if isinstance(specialized, dict) else None
    if isinstance(timeseries, dict):
        runtime = timeseries["runtime"]
        artifact = timeseries["artifact"]
        desired_spec["timeseries"] = {
            "service_name": f"{service_name}-timeseries",
            "stable_service_name": _component_service_name(plan.stable_service_name, "timeseries"),
            "serving_runtime_name": f"{service_name}-timeseries-runtime",
            "runtime_image_repository": runtime["image_repository"],
            "runtime_image_digest": runtime["image_digest"],
            "artifact_uri": (
                f"{plan.artifact_uri_prefix.rstrip('/')}/{str(artifact['object_key']).lstrip('/')}"
            ),
            "artifact_content_hash": artifact["content_hash"],
            "component_model_id": timeseries["runtime_model_id"],
            "inference_config": runtime["inference_config"],
            "hardware_profile": runtime["hardware_profile"],
            "route_path": "/v1/timeseries",
        }
    rul = specialized.get("rul") if isinstance(specialized, dict) else None
    if isinstance(rul, dict) and isinstance(rul.get("runtime"), dict):
        runtime = rul["runtime"]
        artifact = rul["artifact"]
        desired_spec["rul"] = {
            "service_name": f"{service_name}-rul",
            "stable_service_name": _component_service_name(plan.stable_service_name, "rul"),
            "serving_runtime_name": f"{service_name}-rul-runtime",
            "runtime_image_repository": runtime["image_repository"],
            "runtime_image_digest": runtime["image_digest"],
            "artifact_uri": (
                f"{plan.artifact_uri_prefix.rstrip('/')}/{str(artifact['object_key']).lstrip('/')}"
            ),
            "artifact_content_hash": artifact["content_hash"],
            "component_model_id": rul["runtime_model_id"],
            "inference_config": runtime["inference_config"],
            "hardware_profile": runtime["hardware_profile"],
            "route_path": "/v1/rul",
        }
    return desired_spec


def _component_service_name(base: str, component: str) -> str:
    suffix = f"-{component}"
    return f"{base[: 63 - len(suffix)].rstrip('-')}{suffix}"


def autoscaling_metric_query(namespace: str, service_name: str) -> str:
    """Build the closed service-scoped vLLM metric query."""

    if not DNS_LABEL.fullmatch(namespace) or not DNS_LABEL.fullmatch(service_name):
        raise ValueError("autoscaling metric scope must use Kubernetes DNS labels")
    return (
        f'sum({KEDA_VLLM_METRIC_NAME}{{namespace="{namespace}",'
        f'serving_kserve_io_inferenceservice="{service_name}"}})'
    )


def validate_deployment_plan(plan: DeploymentPlan) -> None:
    """Validate a deployment plan before it is persisted by another workflow."""

    _validate_deployment_plan(plan)


def _validate_deployment_plan(plan: DeploymentPlan) -> None:
    for field, value in (
        ("namespace", plan.namespace),
        ("gateway_name", plan.gateway_name),
        ("route_name", plan.route_name),
        ("stable_service_name", plan.stable_service_name),
        ("service_account_name", plan.service_account_name),
        ("serving_runtime_name", plan.serving_runtime_name),
    ):
        if not DNS_LABEL.fullmatch(value):
            raise ValueError(f"{field} must be a Kubernetes DNS label")
    if not plan.artifact_uri_prefix.startswith(("s3://", "pvc://")):
        raise ValueError("artifact_uri_prefix must be an s3 or pvc URI")
    if plan.artifact_uri_prefix.startswith("pvc://"):
        pvc_name = plan.artifact_uri_prefix.removeprefix("pvc://").split("/", 1)[0]
        if not DNS_LABEL.fullmatch(pvc_name):
            raise ValueError("artifact_uri_prefix pvc claim must be a DNS label")
    if (
        len(plan.hostname) > 253
        or "." not in plan.hostname
        or any(not DNS_LABEL.fullmatch(label) for label in plan.hostname.split("."))
    ):
        raise ValueError("hostname must be a DNS name")
    if plan.routing_mode not in {"STANDARD", "INFERENCE_POOL"}:
        raise ValueError("routing_mode is invalid")
    endpoint_name = plan.endpoint_picker_service_name
    endpoint_port = plan.endpoint_picker_service_port
    if plan.routing_mode == "STANDARD":
        if endpoint_name is not None or endpoint_port is not None:
            raise ValueError("endpoint picker fields are only valid for inference pool routing")
    else:
        if endpoint_name is None or endpoint_port is None:
            raise ValueError("endpoint picker service and port are required")
        if not DNS_LABEL.fullmatch(endpoint_name):
            raise ValueError("endpoint_picker_service_name must be a Kubernetes DNS label")
        if not 1 <= endpoint_port <= 65535:
            raise ValueError("endpoint_picker_service_port must be between 1 and 65535")

    autoscaling_values = (
        plan.autoscaling_min_replicas,
        plan.autoscaling_max_replicas,
        plan.autoscaling_target_running_requests,
    )
    if plan.autoscaling_mode == "FIXED":
        if any(value is not None for value in autoscaling_values):
            raise ValueError("autoscaling values are only valid for KEDA_VLLM mode")
        return
    if plan.autoscaling_mode != "KEDA_VLLM":
        raise ValueError("autoscaling_mode is invalid")
    if any(value is None for value in autoscaling_values):
        raise ValueError("all KEDA_VLLM autoscaling values are required")
    minimum, maximum, target = autoscaling_values
    if not isinstance(minimum, int) or isinstance(minimum, bool) or not 2 <= minimum <= 8:
        raise ValueError("autoscaling_min_replicas must be between 2 and 8")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 2 <= maximum <= 32:
        raise ValueError("autoscaling_max_replicas must be between 2 and 32")
    if not isinstance(target, int) or isinstance(target, bool) or not 1 <= target <= 32:
        raise ValueError("autoscaling_target_running_requests must be between 1 and 32")
    if minimum > maximum:
        raise ValueError("autoscaling_min_replicas cannot exceed autoscaling_max_replicas")


def _validate_observation(
    observation: ObservationInput,
    *,
    timeseries_required: bool = False,
    rul_required: bool = False,
) -> None:
    if observation.stage not in STAGE_POLICY:
        raise ValueError("observation stage is invalid")
    start = _as_utc(observation.window_start)
    end = _as_utc(observation.window_end)
    if end <= start or end > datetime.now(UTC):
        raise ValueError("observation window is invalid")
    if observation.request_count < 0 or observation.critical_case_count < 0:
        raise ValueError("observation counts cannot be negative")
    required_metrics = (
        REQUIRED_METRICS
        | (TIMESERIES_REQUIRED_METRICS if timeseries_required else frozenset())
        | (RUL_REQUIRED_METRICS if rul_required else frozenset())
    )
    if set(observation.metrics) != required_metrics:
        raise ValueError("observation metrics are incomplete")
    unsigned_metrics = (
        value for key, value in observation.metrics.items() if key != "task_success_delta"
    )
    if any(not math.isfinite(value) or value < 0 for value in unsigned_metrics):
        raise ValueError("observation metrics must be finite and non-negative")
    if not math.isfinite(observation.metrics["task_success_delta"]):
        raise ValueError("task_success_delta must be finite")
    if timeseries_required:
        rate_metrics = (
            "timeseries_error_rate",
            "timeseries_fallback_rate",
            "timeseries_false_positive_rate",
            "timeseries_miss_rate",
        )
        if any(observation.metrics[key] > 1 for key in rate_metrics):
            raise ValueError("time-series observation rates cannot exceed one")
        count_metrics = (
            "timeseries_request_count",
            "timeseries_detection_count",
            "timeseries_labeled_outcome_count",
        )
        if any(not float(observation.metrics[key]).is_integer() for key in count_metrics):
            raise ValueError("time-series observation counts must be whole numbers")
    if rul_required:
        rul_rate_metrics = (
            "rul_runtime_error_rate",
            "rul_fallback_rate",
            "rul_interval_coverage",
        )
        if any(observation.metrics[key] > 1 for key in rul_rate_metrics):
            raise ValueError("RUL observation rates cannot exceed one")
        rul_count_metrics = (
            "rul_runtime_request_count",
            "rul_business_forecast_count",
            "rul_labeled_outcome_count",
        )
        if any(not float(observation.metrics[key]).is_integer() for key in rul_count_metrics):
            raise ValueError("RUL observation counts must be whole numbers")
    required_sources = {
        "collector_version",
        "prometheus_endpoint_id",
        "prometheus_query_bundle_hash",
        "trace_query_hash",
    }
    if set(observation.source_refs) != required_sources or any(
        not value for value in observation.source_refs.values()
    ):
        raise ValueError("observation source references are incomplete")
    for key in ("prometheus_query_bundle_hash", "trace_query_hash"):
        _validate_sha256(observation.source_refs[key], key)


def _evaluate_observation(
    observation: ObservationInput,
    *,
    timeseries_required: bool = False,
    rul_required: bool = False,
) -> tuple[ObservationDecision, list[str], dict[str, Any]]:
    stage = STAGE_POLICY[observation.stage]
    thresholds: dict[str, Any] = {
        **stage,
        **METRIC_THRESHOLDS,
        "policy_version": "m5-online-v1",
    }
    timeseries_stage = TIMESERIES_STAGE_POLICY[observation.stage]
    rul_stage = RUL_STAGE_POLICY[observation.stage]
    if timeseries_required:
        thresholds.update(timeseries_stage)
        thresholds.update(TIMESERIES_METRIC_THRESHOLDS)
        thresholds["policy_version"] = "m7-timeseries-online-v1"
    if rul_required:
        thresholds.update(rul_stage)
        thresholds.update(RUL_METRIC_THRESHOLDS)
        thresholds["policy_version"] = (
            "m7-predictive-components-online-v1" if timeseries_required else "m7-rul-online-v1"
        )
    metrics = observation.metrics
    catastrophic = []
    if metrics["cross_tenant_leak_count"] > 0:
        catastrophic.append("cross_tenant_leak_detected")
    if metrics["unauthorized_side_effect_count"] > 0:
        catastrophic.append("unauthorized_side_effect_detected")
    if metrics["gpu_xid_error_count"] > 0:
        catastrophic.append("gpu_xid_error_detected")
    if catastrophic:
        return "ROLLBACK", catastrophic, thresholds

    window_seconds = (
        _as_utc(observation.window_end) - _as_utc(observation.window_start)
    ).total_seconds()
    insufficient = []
    if window_seconds < float(stage["min_window_seconds"]):
        insufficient.append("observation_window_too_short")
    if observation.request_count < int(stage["min_request_count"]):
        insufficient.append("request_count_too_low")
    if observation.critical_case_count < int(stage["min_critical_case_count"]):
        insufficient.append("critical_case_count_too_low")
    if timeseries_required and metrics["timeseries_request_count"] < int(
        timeseries_stage["min_timeseries_request_count"]
    ):
        insufficient.append("timeseries_request_count_too_low")
    if timeseries_required and metrics["timeseries_detection_count"] < int(
        timeseries_stage["min_timeseries_detection_count"]
    ):
        insufficient.append("timeseries_detection_count_too_low")
    if timeseries_required and metrics["timeseries_labeled_outcome_count"] < int(
        timeseries_stage["min_timeseries_labeled_outcome_count"]
    ):
        insufficient.append("timeseries_labeled_outcome_count_too_low")
    if rul_required and metrics["rul_runtime_request_count"] < int(
        rul_stage["min_rul_runtime_request_count"]
    ):
        insufficient.append("rul_runtime_request_count_too_low")
    if rul_required and metrics["rul_business_forecast_count"] < int(
        rul_stage["min_rul_business_forecast_count"]
    ):
        insufficient.append("rul_business_forecast_count_too_low")
    if rul_required and metrics["rul_labeled_outcome_count"] < int(
        rul_stage["min_rul_labeled_outcome_count"]
    ):
        insufficient.append("rul_labeled_outcome_count_too_low")
    if insufficient:
        return "INSUFFICIENT", insufficient, thresholds

    failures = []
    comparisons = (
        ("high_risk_miss_rate", "high_risk_miss_rate_max", "high_risk_miss_rate_exceeded"),
        ("error_rate", "error_rate_max", "error_rate_exceeded"),
        ("p95_latency_ms", "p95_latency_ms_max", "p95_latency_exceeded"),
        (
            "human_edit_rate_delta",
            "human_edit_rate_delta_max",
            "human_edit_rate_regressed",
        ),
        ("wrong_part_rate", "wrong_part_rate_max", "wrong_part_rate_exceeded"),
        (
            "gpu_memory_utilization",
            "gpu_memory_utilization_max",
            "gpu_memory_pressure_exceeded",
        ),
        ("queue_age_seconds", "queue_age_seconds_max", "queue_age_exceeded"),
    )
    for metric, threshold, reason in comparisons:
        if metrics[metric] > METRIC_THRESHOLDS[threshold]:
            failures.append(reason)
    if metrics["task_success_delta"] < METRIC_THRESHOLDS["task_success_delta_min"]:
        failures.append("task_success_regressed")
    if timeseries_required:
        timeseries_comparisons = [
            (
                "timeseries_error_rate",
                "timeseries_error_rate_max",
                "timeseries_runtime_error_rate_exceeded",
            ),
            (
                "timeseries_p95_latency_ms",
                "timeseries_p95_latency_ms_max",
                "timeseries_runtime_latency_exceeded",
            ),
        ]
        if observation.stage == "PRODUCTION":
            timeseries_comparisons.extend(
                [
                    (
                        "timeseries_fallback_rate",
                        "timeseries_fallback_rate_max",
                        "timeseries_rule_fallback_rate_exceeded",
                    ),
                    (
                        "timeseries_false_positive_rate",
                        "timeseries_false_positive_rate_max",
                        "timeseries_false_positive_rate_exceeded",
                    ),
                    (
                        "timeseries_miss_rate",
                        "timeseries_miss_rate_max",
                        "timeseries_miss_rate_exceeded",
                    ),
                ]
            )
        for metric, threshold, reason in timeseries_comparisons:
            if metrics[metric] > TIMESERIES_METRIC_THRESHOLDS[threshold]:
                failures.append(reason)
    if rul_required:
        rul_comparisons = [
            (
                "rul_runtime_error_rate",
                "rul_runtime_error_rate_max",
                "rul_runtime_error_rate_exceeded",
            ),
            (
                "rul_runtime_p95_latency_ms",
                "rul_runtime_p95_latency_ms_max",
                "rul_runtime_latency_exceeded",
            ),
        ]
        if observation.stage == "PRODUCTION":
            rul_comparisons.extend(
                [
                    (
                        "rul_fallback_rate",
                        "rul_fallback_rate_max",
                        "rul_empirical_fallback_rate_exceeded",
                    ),
                    (
                        "rul_median_absolute_error_minutes",
                        "rul_median_absolute_error_minutes_max",
                        "rul_median_absolute_error_exceeded",
                    ),
                ]
            )
        for metric, threshold, reason in rul_comparisons:
            if metrics[metric] > RUL_METRIC_THRESHOLDS[threshold]:
                failures.append(reason)
        if (
            observation.stage == "PRODUCTION"
            and metrics["rul_interval_coverage"]
            < RUL_METRIC_THRESHOLDS["rul_interval_coverage_min"]
        ):
            failures.append("rul_interval_coverage_below_minimum")
    return ("ROLLBACK", failures, thresholds) if failures else ("PASS", [], thresholds)


def _release_has_timeseries(release: ModelReleaseRecord) -> bool:
    components = release.manifest_json.get("specialized_components")
    return isinstance(components, dict) and isinstance(components.get("timeseries"), dict)


def _release_has_rul(release: ModelReleaseRecord) -> bool:
    components = release.manifest_json.get("specialized_components")
    return isinstance(components, dict) and isinstance(components.get("rul"), dict)


def _set_rollback_desired(
    deployment: ModelDeploymentRecord,
    reason_code: str,
    subject_id: str,
) -> None:
    desired = dict(deployment.desired_spec_json)
    desired["rollback_reason"] = reason_code
    deployment.desired_spec_json = desired
    deployment.desired_spec_hash = _digest_json(desired)
    deployment.desired_stage = "ROLLED_BACK"
    deployment.desired_traffic_percent = 0.0
    deployment.status = "PENDING"
    deployment.failure_reason = reason_code
    deployment.requested_by_subject_id = subject_id
    deployment.version += 1


def _provider_target(
    release: ModelReleaseRecord,
    deployment: ModelDeploymentRecord,
) -> ProviderTarget:
    return ProviderTarget(
        deployment_id=deployment.deployment_id,
        release_id=release.release_id,
        manifest_hash=release.manifest_hash,
        namespace=deployment.namespace,
        service_name=deployment.service_name,
        route_name=deployment.route_name,
        stable_service_name=deployment.stable_service_name,
        desired_stage=deployment.desired_stage,
        desired_traffic_percent=deployment.desired_traffic_percent,
        desired_spec=deployment.desired_spec_json,
        desired_spec_hash=deployment.desired_spec_hash,
    )


def _applied_reason(stage: str) -> str:
    return {
        "SHADOW": "shadow_route_verified",
        "CANARY_5": "canary_5_route_verified",
        "CANARY_25": "canary_25_route_verified",
        "PRODUCTION": "production_route_verified",
        "ROLLED_BACK": "rollback_route_verified",
    }[stage]


def _release_or_hidden(session: Session, tenant_id: str, release_id: str) -> ModelReleaseRecord:
    release = session.scalar(
        select(ModelReleaseRecord).where(
            ModelReleaseRecord.tenant_id == tenant_id,
            ModelReleaseRecord.release_id == release_id,
        )
    )
    if release is None:
        raise DeploymentNotVisible
    return release


def _deployment_for_release(
    session: Session,
    tenant_id: str,
    release_id: str,
) -> ModelDeploymentRecord | None:
    return session.scalar(
        select(ModelDeploymentRecord).where(
            ModelDeploymentRecord.tenant_id == tenant_id,
            ModelDeploymentRecord.release_id == release_id,
        )
    )


def _deployment_or_hidden(
    session: Session,
    tenant_id: str,
    release_id: str,
) -> ModelDeploymentRecord:
    deployment = _deployment_for_release(session, tenant_id, release_id)
    if deployment is None:
        raise DeploymentNotVisible
    return deployment


def _deployment_by_id_or_hidden(
    session: Session,
    tenant_id: str,
    deployment_id: str,
) -> ModelDeploymentRecord:
    deployment = session.scalar(
        select(ModelDeploymentRecord).where(
            ModelDeploymentRecord.tenant_id == tenant_id,
            ModelDeploymentRecord.deployment_id == deployment_id,
        )
    )
    if deployment is None:
        raise DeploymentNotVisible
    return deployment


def _approval_or_conflict(
    session: Session,
    tenant_id: str,
    release_id: str,
) -> ModelReleaseApprovalRecord:
    approval = session.scalar(
        select(ModelReleaseApprovalRecord).where(
            ModelReleaseApprovalRecord.tenant_id == tenant_id,
            ModelReleaseApprovalRecord.release_id == release_id,
        )
    )
    if approval is None:
        raise DeploymentConflict("release_approval_required")
    return approval


def _latest_observation(
    session: Session,
    tenant_id: str,
    release_id: str,
    stage: str,
) -> ModelReleaseObservationRecord | None:
    return session.scalar(
        select(ModelReleaseObservationRecord)
        .where(
            ModelReleaseObservationRecord.tenant_id == tenant_id,
            ModelReleaseObservationRecord.release_id == release_id,
            ModelReleaseObservationRecord.stage == stage,
        )
        .order_by(ModelReleaseObservationRecord.sequence.desc())
        .limit(1)
    )


def _next_observation_sequence(
    session: Session,
    tenant_id: str,
    release_id: str,
    stage: str,
) -> int:
    sequence = session.scalar(
        select(func.max(ModelReleaseObservationRecord.sequence)).where(
            ModelReleaseObservationRecord.tenant_id == tenant_id,
            ModelReleaseObservationRecord.release_id == release_id,
            ModelReleaseObservationRecord.stage == stage,
        )
    )
    return int(sequence or 0) + 1


def _aggregate(
    session: Session,
    tenant_id: str,
    release: ModelReleaseRecord,
    deployment: ModelDeploymentRecord,
) -> DeploymentAggregate:
    observations = tuple(
        session.scalars(
            select(ModelReleaseObservationRecord)
            .where(
                ModelReleaseObservationRecord.tenant_id == tenant_id,
                ModelReleaseObservationRecord.release_id == release.release_id,
            )
            .order_by(
                ModelReleaseObservationRecord.created_at,
                ModelReleaseObservationRecord.sequence,
            )
        )
    )
    return DeploymentAggregate(release=release, deployment=deployment, observations=observations)


def _append_transition(
    session: Session,
    release: ModelReleaseRecord,
    *,
    from_status: str,
    to_status: str,
    actor_subject_id: str,
    reason_code: str,
    evidence: dict[str, Any],
    occurred_at: datetime,
) -> None:
    sequence = (
        int(
            session.scalar(
                select(func.max(ModelReleaseTransitionRecord.sequence)).where(
                    ModelReleaseTransitionRecord.tenant_id == release.tenant_id,
                    ModelReleaseTransitionRecord.release_id == release.release_id,
                )
            )
            or 0
        )
        + 1
    )
    session.add(
        ModelReleaseTransitionRecord(
            transition_id=f"release-transition-{uuid4().hex}",
            tenant_id=release.tenant_id,
            release_id=release.release_id,
            sequence=sequence,
            from_status=from_status,
            to_status=to_status,
            actor_subject_id=actor_subject_id,
            reason_code=reason_code,
            evidence_json=evidence,
            evidence_hash=_digest_json(evidence),
            occurred_at=occurred_at,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
    )


def _sync_environment_alias(
    session: Session,
    release: ModelReleaseRecord,
    deployment: ModelDeploymentRecord,
    *,
    previous_release_status: str,
    actor_subject_id: str,
    occurred_at: datetime,
) -> None:
    """Switch only the release environment's alias after a verified route change."""

    admission = (
        _require_rollback_media_target(session, release, deployment)
        if deployment.current_stage == "ROLLED_BACK"
        else _require_media_admission(session, release, deployment, allow_exact_recovery=True)
    )
    if admission == "LEGACY_EVIDENCE_REQUIRED":
        return
    model_alias = MODEL_ALIAS_BY_ENVIRONMENT.get(release.target_environment)
    if model_alias is None:
        return
    deployment_manifest_hash = deployment.desired_spec_json.get(
        "manifest_hash",
        deployment.desired_spec_json.get("release_manifest_hash"),
    )
    if deployment_manifest_hash != release.manifest_hash:
        raise DeploymentConflict("model_manifest_inconsistent", deployment.version)
    active_release = release
    if deployment.current_stage == "ROLLED_BACK":
        if previous_release_status != "PRODUCTION" or release.rollback_release_id is None:
            return
        active_release = _release_or_hidden(session, release.tenant_id, release.rollback_release_id)
        _require_manifest_integrity(active_release)
        _require_rollback_media_target(session, release)
        if active_release.target_environment != release.target_environment:
            raise DeploymentConflict("model_alias_environment_mismatch", deployment.version)
        if active_release.status != "PRODUCTION":
            restored_from = active_release.status
            active_release.status = "PRODUCTION"
            active_release.traffic_percent = 100.0
            active_release.failure_reason = None
            active_release.version += 1
            _append_transition(
                session,
                active_release,
                from_status=restored_from,
                to_status="PRODUCTION",
                actor_subject_id=actor_subject_id,
                reason_code="rollback_restored_release",
                evidence={
                    "rolled_back_release_id": release.release_id,
                    "deployment_id": deployment.deployment_id,
                },
                occurred_at=occurred_at,
            )
    elif deployment.current_stage != "PRODUCTION":
        return

    alias = session.scalar(
        select(ModelAliasRecord).where(
            ModelAliasRecord.tenant_id == release.tenant_id,
            ModelAliasRecord.alias == model_alias,
        )
    )
    if alias is not None and alias.active_release_id != active_release.release_id:
        previous = _release_or_hidden(session, release.tenant_id, alias.active_release_id)
        if previous.target_environment != release.target_environment:
            raise DeploymentConflict("model_alias_environment_mismatch", deployment.version)
        if previous.status == "PRODUCTION":
            previous.status = "RETIRED"
            previous.traffic_percent = 0.0
            previous.failure_reason = None
            previous.version += 1
            _append_transition(
                session,
                previous,
                from_status="PRODUCTION",
                to_status="RETIRED",
                actor_subject_id=actor_subject_id,
                reason_code="superseded_by_production_release",
                evidence={
                    "replacement_release_id": active_release.release_id,
                    "deployment_id": deployment.deployment_id,
                },
                occurred_at=occurred_at,
            )

    endpoint = deployment.endpoint_url
    if endpoint is None:  # guarded by reconcile validation; keeps this helper total for typing.
        raise DeploymentConflict("model_endpoint_unavailable", deployment.version)
    runtime_profile = str(active_release.manifest_json["runtime"]["profile_id"])
    if alias is None:
        alias = ModelAliasRecord(
            alias_id=f"model-alias-{uuid4().hex}",
            tenant_id=release.tenant_id,
            alias=model_alias,
            active_release_id=active_release.release_id,
            deployment_id=deployment.deployment_id,
            manifest_hash=active_release.manifest_hash,
            endpoint_url=endpoint,
            runtime_profile=runtime_profile,
            status="ACTIVE",
            updated_by_subject_id=actor_subject_id,
            version=1,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
        session.add(alias)
    else:
        alias.active_release_id = active_release.release_id
        alias.deployment_id = deployment.deployment_id
        alias.manifest_hash = active_release.manifest_hash
        alias.endpoint_url = endpoint
        alias.runtime_profile = runtime_profile
        alias.status = "ACTIVE"
        alias.updated_by_subject_id = actor_subject_id
        alias.version += 1

    quota = session.scalar(
        select(ModelGatewayQuotaRecord).where(
            ModelGatewayQuotaRecord.tenant_id == release.tenant_id,
            ModelGatewayQuotaRecord.model_alias == model_alias,
        )
    )
    if quota is None:
        session.add(
            ModelGatewayQuotaRecord(
                quota_id=f"model-quota-{uuid4().hex}",
                tenant_id=release.tenant_id,
                model_alias=model_alias,
                requests_per_minute=60,
                tokens_per_day=100_000,
                max_output_tokens=2048,
                allowed_request_classes=[
                    "ASR",
                    "DIAGNOSIS",
                    "GRAPH_CAUSAL_EXTRACTION",
                    "MAINTENANCE_PLANNING",
                    "TTS",
                    "VLM",
                ],
                enabled=True,
                created_by_subject_id=actor_subject_id,
                version=1,
                created_at=occurred_at,
                updated_at=occurred_at,
            )
        )


def _require_manifest_integrity(release: ModelReleaseRecord) -> None:
    if _digest_json(release.manifest_json) != release.manifest_hash:
        raise DeploymentConflict("release_manifest_integrity_failed", release.version)


def _require_release_version(release: ModelReleaseRecord, expected: int) -> None:
    if release.version != expected:
        raise DeploymentConflict("release_version_mismatch", release.version)


def _require_deployment_version(deployment: ModelDeploymentRecord, expected: int) -> None:
    if deployment.version != expected:
        raise DeploymentConflict("deployment_version_mismatch", deployment.version)


def _validate_reason_code(value: str) -> str:
    normalized = value.strip().lower()
    if (
        not normalized
        or len(normalized) > 255
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_:-" for character in normalized
        )
    ):
        raise ValueError("deployment reason code is invalid")
    return normalized


def _validate_sha256(value: str, field: str) -> None:
    if (
        not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field} must be a sha256 digest")


def _require_media_admission(
    session: Session,
    release: ModelReleaseRecord,
    deployment: ModelDeploymentRecord | None = None,
    *,
    allow_exact_recovery: bool = False,
) -> str:
    try:
        # Import after package initialization: releases and deployment expose eager facades.
        from industrial_ops_agent.releases.staging_smoke import (
            require_publishable_staging_admission,
        )

        require_publishable_staging_admission(
            session, release, deployment_provider="KSERVE_GATEWAY_API"
        )
        return require_media_deployment_supply_chain(
            session, release, deployment, allow_exact_recovery=allow_exact_recovery
        )
    except ValueError as exc:
        raise DeploymentConflict(str(exc), release.version) from exc


def _require_rollback_media_target(
    session: Session,
    release: ModelReleaseRecord,
    deployment: ModelDeploymentRecord | None = None,
) -> str:
    if release.rollback_release_id is None:
        # A route withdrawal without a target cannot create serving authority.
        return "VERIFIED"
    target = _release_or_hidden(session, release.tenant_id, release.rollback_release_id)
    _require_manifest_integrity(target)
    if target.target_environment != release.target_environment or release.manifest_json.get(
        "rollback"
    ) != {"release_id": target.release_id, "manifest_hash": target.manifest_hash}:
        raise DeploymentConflict("rollback_target_binding_changed", release.version)
    return _require_media_admission(
        session, target, deployment, allow_exact_recovery=deployment is not None
    )
