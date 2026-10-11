from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from industrial_ops_agent.staging_acceptance.contracts import (
    Digest,
    GitCommit,
    SafeIdentity,
    UnverifiedScenario,
)

AssertionIdentity = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$",
    ),
]
EvidenceReference = Annotated[
    str,
    Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,254}$",
    ),
]
ObservationCode = Annotated[
    str,
    Field(
        min_length=1,
        max_length=96,
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$",
    ),
]
T = TypeVar("T")

_DNS_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
_SAFE_CONTEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,127}$")
_SAFE_RUNNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+@:-]{0,127}$")

REQUIRED_ASSERTIONS: Final[Mapping[UnverifiedScenario, tuple[str, ...]]] = (
    MappingProxyType(
        {
            UnverifiedScenario.KSERVE_METRIC_LABEL_QUERY: (
                "kserve_metric_label_query.query_returns_series",
                "kserve_metric_label_query.labels_match_contract",
            ),
            UnverifiedScenario.KSERVE_LOW_LOAD: (
                "kserve_low_load.min_replica_floor",
                "kserve_low_load.latency_slo",
            ),
            UnverifiedScenario.KSERVE_HIGH_LOAD: (
                "kserve_high_load.scaling_triggered",
                "kserve_high_load.capacity_reached",
                "kserve_high_load.latency_slo",
            ),
            UnverifiedScenario.KSERVE_METRIC_OUTAGE: (
                "kserve_metric_outage.fallback_activated",
                "kserve_metric_outage.min_replica_floor",
                "kserve_metric_outage.alert_fired",
            ),
            UnverifiedScenario.KSERVE_GPU_PENDING: (
                "kserve_gpu_pending.pending_detected",
                "kserve_gpu_pending.capacity_alert_fired",
                "kserve_gpu_pending.request_protected",
            ),
            UnverifiedScenario.KSERVE_OSCILLATION: (
                "kserve_oscillation.scaling_stabilized",
                "kserve_oscillation.oscillation_bounded",
            ),
            UnverifiedScenario.KSERVE_ROUTING_ORTHOGONALITY: (
                "kserve_routing_orthogonality.stable_route_unchanged",
                "kserve_routing_orthogonality.candidate_route_isolated",
            ),
            UnverifiedScenario.KSERVE_HIGH_LOAD_ROLLBACK: (
                "kserve_high_load_rollback.rollback_completed",
                "kserve_high_load_rollback.stable_capacity_restored",
                "kserve_high_load_rollback.recovery_slo",
            ),
            UnverifiedScenario.CORE_HTTP_AGENT_LOAD: (
                "core_http_agent_load.error_rate_slo",
                "core_http_agent_load.latency_slo",
                "core_http_agent_load.queue_bounded",
            ),
            UnverifiedScenario.CORE_MEDIA_LOAD: (
                "core_media_load.upload_slo",
                "core_media_load.pipeline_success_rate",
            ),
            UnverifiedScenario.CORE_TEMPORAL_BACKLOG: (
                "core_temporal_backlog.backlog_observed",
                "core_temporal_backlog.workers_scaled",
                "core_temporal_backlog.backlog_recovered",
            ),
            UnverifiedScenario.CORE_KAFKA_LAG: (
                "core_kafka_lag.lag_observed",
                "core_kafka_lag.consumers_scaled",
                "core_kafka_lag.lag_recovered",
            ),
            UnverifiedScenario.CORE_METRIC_FALLBACK: (
                "core_metric_fallback.fallback_activated",
                "core_metric_fallback.min_replica_floor",
                "core_metric_fallback.alert_fired",
            ),
            UnverifiedScenario.CORE_QUOTA_HEADROOM: (
                "core_quota_headroom.quota_headroom_sufficient",
                "core_quota_headroom.burst_absorbed",
            ),
            UnverifiedScenario.CORE_SCALE_DOWN_RELIABILITY: (
                "core_scale_down_reliability.cooldown_honored",
                "core_scale_down_reliability.inflight_drained",
                "core_scale_down_reliability.min_replica_floor",
            ),
        }
    )
)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DynamicAssertionEvidence(_ClosedModel):
    assertion_id: AssertionIdentity
    passed: bool
    observation_code: ObservationCode
    observation_digest: Digest


class DynamicScenarioResult(_ClosedModel):
    scenario: UnverifiedScenario
    status: Literal["PASSED", "FAILED", "BLOCKED"]
    started_at: datetime
    completed_at: datetime
    evidence_ref: EvidenceReference
    artifact_digest: Digest
    assertions: tuple[DynamicAssertionEvidence, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_execution(self) -> DynamicScenarioResult:
        if not _is_utc(self.started_at) or not _is_utc(self.completed_at):
            raise ValueError("dynamic execution window must use UTC timestamps")
        if not self.started_at < self.completed_at:
            raise ValueError("dynamic execution window must have positive duration")
        if self.completed_at - self.started_at > timedelta(hours=72):
            raise ValueError("dynamic execution window exceeds 72 hours")
        _unique(
            self.assertions,
            lambda item: item.assertion_id,
            "duplicate dynamic assertion",
        )
        allowed = set(REQUIRED_ASSERTIONS[self.scenario])
        if any(item.assertion_id not in allowed for item in self.assertions):
            raise ValueError("unknown dynamic assertion for scenario")
        return self


class DynamicAcceptanceResults(_ClosedModel):
    schema_version: Literal[1]
    environment: Literal["STAGING"]
    environment_id: SafeIdentity
    kube_context: SafeIdentity
    git_commit: GitCommit
    static_plan_sha256: Digest
    runner_version: SafeIdentity
    scenarios: tuple[DynamicScenarioResult, ...] = Field(min_length=1, max_length=15)

    @field_validator("environment_id")
    @classmethod
    def validate_environment_id(cls, value: str) -> str:
        return _validated(value, _DNS_NAME, "invalid environment identity")

    @field_validator("kube_context")
    @classmethod
    def validate_context(cls, value: str) -> str:
        return _validated(value, _SAFE_CONTEXT, "invalid Kubernetes context")

    @field_validator("runner_version")
    @classmethod
    def validate_runner_version(cls, value: str) -> str:
        return _validated(value, _SAFE_RUNNER, "invalid Staging runner version")

    @model_validator(mode="after")
    def validate_scenarios(self) -> DynamicAcceptanceResults:
        _unique(
            self.scenarios,
            lambda item: item.scenario,
            "duplicate dynamic scenario",
        )
        return self


class DynamicScenarioManifestEntry(_ClosedModel):
    scenario: UnverifiedScenario
    status: Literal["PASSED", "FAILED", "BLOCKED", "MISSING"]
    reason: Literal[
        "verified",
        "dynamic_scenario_missing",
        "dynamic_scenario_failed",
        "dynamic_scenario_blocked",
        "dynamic_assertion_contract_mismatch",
        "dynamic_assertion_failed",
        "static_preflight_not_healthy",
        "dynamic_binding_mismatch",
        "dynamic_execution_window_invalid",
    ]
    started_at: datetime | None
    completed_at: datetime | None
    evidence_ref: EvidenceReference | None
    artifact_digest: Digest | None
    assertions: tuple[DynamicAssertionEvidence, ...] = Field(max_length=4)


class StagingDynamicAcceptanceManifest(_ClosedModel):
    schema_version: Literal[1]
    manifest_id: Digest
    aggregator_version: Literal["staging-dynamic-acceptance-aggregator/v1"]
    generated_at: datetime
    status: Literal[
        "BLOCKED", "STAGING_ACCEPTANCE_COMPLETE_PRODUCTION_REVIEW_REQUIRED"
    ]
    environment: Literal["STAGING"]
    environment_id: SafeIdentity
    kube_context: SafeIdentity
    git_commit: GitCommit
    static_plan_sha256: Digest
    runner_version: SafeIdentity
    static_preflight_report_id: Digest
    static_preflight_artifact_digest: Digest
    dynamic_results_sha256: Digest
    summary: dict[str, int]
    blocker_codes: tuple[str, ...]
    scenarios: tuple[DynamicScenarioManifestEntry, ...] = Field(
        min_length=15, max_length=15
    )
    limitation: Literal[
        "Dynamic Staging evidence is complete only for production review; "
        "production sign-offs remain required."
    ]

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return normalized_utc(value)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) != {"passed", "failed", "blocked", "missing"}:
            raise ValueError("invalid dynamic manifest summary")
        if any(item < 0 for item in value.values()):
            raise ValueError("invalid dynamic manifest summary")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> StagingDynamicAcceptanceManifest:
        _unique(
            self.scenarios,
            lambda item: item.scenario,
            "duplicate manifest scenario",
        )
        if {item.scenario for item in self.scenarios} != set(UnverifiedScenario):
            raise ValueError("dynamic manifest must contain every scenario once")
        expected_summary = {
            "passed": sum(item.status == "PASSED" for item in self.scenarios),
            "failed": sum(item.status == "FAILED" for item in self.scenarios),
            "blocked": sum(item.status == "BLOCKED" for item in self.scenarios),
            "missing": sum(item.status == "MISSING" for item in self.scenarios),
        }
        if self.summary != expected_summary:
            raise ValueError("dynamic manifest summary does not match scenarios")
        complete = all(item.status == "PASSED" for item in self.scenarios)
        if self.status == "BLOCKED" and complete and not self.blocker_codes:
            raise ValueError("blocked dynamic manifest requires a blocker")
        if self.status != "BLOCKED" and (not complete or self.blocker_codes):
            raise ValueError("complete dynamic manifest cannot contain blockers")
        return self


def _is_utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == timedelta(0)


def _validated(value: str, pattern: re.Pattern[str], message: str) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(message)
    return value


def _unique(items: tuple[T, ...], identity: Callable[[T], Hashable], message: str) -> None:
    identities = [identity(item) for item in items]
    if len(identities) != len(set(identities)):
        raise ValueError(message)


def normalized_utc(value: datetime) -> datetime:
    if not _is_utc(value):
        raise ValueError("timestamp must use UTC")
    return value.astimezone(UTC)
