from __future__ import annotations

import re
from collections.abc import Callable, Hashable
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DnsName = Annotated[str, Field(min_length=1, max_length=253)]
SafeIdentity = Annotated[str, Field(min_length=1, max_length=128)]
GitCommit = Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
JsonScalar = bool | int | str | None
T = TypeVar("T")

_DNS_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
_SAFE_CONTEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,127}$")
_SAFE_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+@:-]{0,127}$")
_KUBERNETES_VERSION = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")


class GitOpsComponent(StrEnum):
    KEDA = "KEDA"
    PROMETHEUS = "PROMETHEUS"
    KSERVE_CRD = "KSERVE_CRD"
    KSERVE_RESOURCES = "KSERVE_RESOURCES"
    OBSERVABILITY = "OBSERVABILITY"
    RELEASE_CONTROLLER = "RELEASE_CONTROLLER"
    CORE_APPLICATION = "CORE_APPLICATION"


class UnverifiedScenario(StrEnum):
    KSERVE_METRIC_LABEL_QUERY = "KSERVE_METRIC_LABEL_QUERY"
    KSERVE_LOW_LOAD = "KSERVE_LOW_LOAD"
    KSERVE_HIGH_LOAD = "KSERVE_HIGH_LOAD"
    KSERVE_METRIC_OUTAGE = "KSERVE_METRIC_OUTAGE"
    KSERVE_GPU_PENDING = "KSERVE_GPU_PENDING"
    KSERVE_OSCILLATION = "KSERVE_OSCILLATION"
    KSERVE_ROUTING_ORTHOGONALITY = "KSERVE_ROUTING_ORTHOGONALITY"
    KSERVE_HIGH_LOAD_ROLLBACK = "KSERVE_HIGH_LOAD_ROLLBACK"
    CORE_HTTP_AGENT_LOAD = "CORE_HTTP_AGENT_LOAD"
    CORE_MEDIA_LOAD = "CORE_MEDIA_LOAD"
    CORE_TEMPORAL_BACKLOG = "CORE_TEMPORAL_BACKLOG"
    CORE_KAFKA_LAG = "CORE_KAFKA_LAG"
    CORE_METRIC_FALLBACK = "CORE_METRIC_FALLBACK"
    CORE_QUOTA_HEADROOM = "CORE_QUOTA_HEADROOM"
    CORE_SCALE_DOWN_RELIABILITY = "CORE_SCALE_DOWN_RELIABILITY"


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NamedResourceExpectation(_ClosedModel):
    namespace: DnsName
    name: DnsName

    @field_validator("namespace", "name")
    @classmethod
    def validate_resource_identity(cls, value: str) -> str:
        return _validated(value, _DNS_NAME, "invalid Kubernetes resource identity")


class GitOpsApplicationExpectation(NamedResourceExpectation):
    component: GitOpsComponent
    expected_revision: SafeIdentity

    @field_validator("expected_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        return _validated(value, _SAFE_REVISION, "invalid GitOps revision")


class DeploymentExpectation(NamedResourceExpectation):
    min_ready_replicas: int = Field(ge=1, le=64)


class HpaExpectation(NamedResourceExpectation):
    min_replicas: int = Field(ge=1, le=64)
    max_replicas: int = Field(ge=1, le=128)

    @model_validator(mode="after")
    def validate_bounds(self) -> HpaExpectation:
        if self.min_replicas > self.max_replicas:
            raise ValueError("minimum replicas cannot exceed maximum replicas")
        return self


class ScaledObjectExpectation(HpaExpectation):
    pass


class StagingAcceptancePlan(_ClosedModel):
    schema_version: Literal[1]
    environment: Literal["STAGING"]
    environment_id: SafeIdentity
    kube_context: SafeIdentity
    git_commit: GitCommit
    kubernetes_server_version: SafeIdentity
    gitops_applications: tuple[GitOpsApplicationExpectation, ...] = Field(
        min_length=7, max_length=7
    )
    deployments: tuple[DeploymentExpectation, ...] = Field(min_length=1, max_length=12)
    hpas: tuple[HpaExpectation, ...] = Field(min_length=1, max_length=8)
    scaled_objects: tuple[ScaledObjectExpectation, ...] = Field(min_length=1, max_length=8)
    inference_services: tuple[NamedResourceExpectation, ...] = Field(
        min_length=1, max_length=4
    )
    pod_monitors: tuple[NamedResourceExpectation, ...] = Field(min_length=1, max_length=4)

    @field_validator("environment_id")
    @classmethod
    def validate_environment_id(cls, value: str) -> str:
        return _validated(value, _DNS_NAME, "invalid environment identity")

    @field_validator("kube_context")
    @classmethod
    def validate_context(cls, value: str) -> str:
        return _validated(value, _SAFE_CONTEXT, "invalid Kubernetes context")

    @field_validator("kubernetes_server_version")
    @classmethod
    def validate_kubernetes_version(cls, value: str) -> str:
        return _validated(value, _KUBERNETES_VERSION, "invalid Kubernetes version")

    @model_validator(mode="after")
    def validate_resource_sets(self) -> StagingAcceptancePlan:
        components = {item.component for item in self.gitops_applications}
        if components != set(GitOpsComponent):
            raise ValueError("gitops_applications must contain every approved component once")
        _unique(self.gitops_applications, lambda item: (item.namespace, item.name))
        _unique(self.deployments, lambda item: (item.namespace, item.name))
        _unique(self.hpas, lambda item: (item.namespace, item.name))
        _unique(self.scaled_objects, lambda item: (item.namespace, item.name))
        _unique(self.inference_services, lambda item: (item.namespace, item.name))
        _unique(self.pod_monitors, lambda item: (item.namespace, item.name))
        return self


class CommandEvidence(_ClosedModel):
    argv: tuple[str, ...]
    exit_code: int | None
    stdout_sha256: Digest
    stderr_sha256: Digest


class CheckEvidence(_ClosedModel):
    check_id: str
    status: Literal["PASS", "BLOCKED"]
    reason: str
    command: CommandEvidence | None
    facts: dict[str, JsonScalar]


class UnverifiedEvidence(_ClosedModel):
    scenario: UnverifiedScenario
    status: Literal["UNVERIFIED"] = "UNVERIFIED"
    reason: Literal["requires_independent_staging_execution"] = (
        "requires_independent_staging_execution"
    )


class StagingAcceptanceReport(_ClosedModel):
    schema_version: Literal[1]
    report_id: Digest
    collector_version: Literal["staging-acceptance-collector/v1"]
    generated_at: datetime
    status: Literal["BLOCKED", "PREFLIGHT_PASSED_ACCEPTANCE_INCOMPLETE"]
    environment: Literal["STAGING"]
    environment_id: SafeIdentity
    kube_context: SafeIdentity
    git_commit: GitCommit
    plan_sha256: Digest
    summary: dict[str, int]
    checks: tuple[CheckEvidence, ...]
    unverified_scenarios: tuple[UnverifiedEvidence, ...]
    limitation: Literal[
        "Static preflight only; dynamic Staging scenarios and production sign-offs remain required."
    ]

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) != {"passed", "blocked"} or any(item < 0 for item in value.values()):
            raise ValueError("invalid report summary")
        return value


def _validated(value: str, pattern: re.Pattern[str], message: str) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(message)
    return value


def _unique(items: tuple[T, ...], identity: Callable[[T], Hashable]) -> None:
    identities = [identity(item) for item in items]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate resource identity")
