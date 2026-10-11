from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from industrial_ops_agent.staging_acceptance.contracts import (
    Digest,
    GitCommit,
    SafeIdentity,
    UnverifiedScenario,
)
from industrial_ops_agent.staging_acceptance.dynamic_contracts import (
    REQUIRED_ASSERTIONS,
    AssertionIdentity,
    ObservationCode,
)

AuthorizationReference = Annotated[
    str,
    Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,254}$",
    ),
]
RelativeEvidencePath = Annotated[str, Field(min_length=1, max_length=255)]

_DNS_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
_SAFE_CONTEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,127}$")
_SAFE_RUNNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+@:-]{0,127}$")
_MAX_AUTHORIZATION_WINDOW = timedelta(days=7)


class ScenarioRisk(StrEnum):
    OBSERVE_ONLY = "OBSERVE_ONLY"
    ACTIVE_LOAD = "ACTIVE_LOAD"
    FAULT_INJECTION = "FAULT_INJECTION"
    ROLLBACK = "ROLLBACK"


SCENARIO_RISKS: Final[Mapping[UnverifiedScenario, ScenarioRisk]] = MappingProxyType(
    {
        UnverifiedScenario.KSERVE_METRIC_LABEL_QUERY: ScenarioRisk.OBSERVE_ONLY,
        UnverifiedScenario.KSERVE_LOW_LOAD: ScenarioRisk.ACTIVE_LOAD,
        UnverifiedScenario.KSERVE_HIGH_LOAD: ScenarioRisk.ACTIVE_LOAD,
        UnverifiedScenario.KSERVE_METRIC_OUTAGE: ScenarioRisk.FAULT_INJECTION,
        UnverifiedScenario.KSERVE_GPU_PENDING: ScenarioRisk.FAULT_INJECTION,
        UnverifiedScenario.KSERVE_OSCILLATION: ScenarioRisk.ACTIVE_LOAD,
        UnverifiedScenario.KSERVE_ROUTING_ORTHOGONALITY: ScenarioRisk.ACTIVE_LOAD,
        UnverifiedScenario.KSERVE_HIGH_LOAD_ROLLBACK: ScenarioRisk.ROLLBACK,
        UnverifiedScenario.CORE_HTTP_AGENT_LOAD: ScenarioRisk.ACTIVE_LOAD,
        UnverifiedScenario.CORE_MEDIA_LOAD: ScenarioRisk.ACTIVE_LOAD,
        UnverifiedScenario.CORE_TEMPORAL_BACKLOG: ScenarioRisk.ACTIVE_LOAD,
        UnverifiedScenario.CORE_KAFKA_LAG: ScenarioRisk.ACTIVE_LOAD,
        UnverifiedScenario.CORE_METRIC_FALLBACK: ScenarioRisk.FAULT_INJECTION,
        UnverifiedScenario.CORE_QUOTA_HEADROOM: ScenarioRisk.OBSERVE_ONLY,
        UnverifiedScenario.CORE_SCALE_DOWN_RELIABILITY: ScenarioRisk.ACTIVE_LOAD,
    }
)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StagingDriverBinding(_ClosedModel):
    driver_id: SafeIdentity
    executable_path: str = Field(min_length=1, max_length=4096)
    sha256: Digest

    @field_validator("executable_path")
    @classmethod
    def validate_executable_path(cls, value: str) -> str:
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("driver executable path contains unsafe characters")
        if not Path(value).is_absolute():
            raise ValueError("driver executable path must be absolute")
        return value


class StagingChangeAuthorization(_ClosedModel):
    reference: AuthorizationReference
    approved_by: SafeIdentity
    approved_at: datetime
    not_before: datetime
    expires_at: datetime
    allowed_risk_classes: tuple[ScenarioRisk, ...] = Field(min_length=4, max_length=4)

    @field_validator("approved_at", "not_before", "expires_at")
    @classmethod
    def validate_utc(cls, value: datetime) -> datetime:
        if not _is_utc(value):
            raise ValueError("authorization timestamps must use UTC")
        return value

    @model_validator(mode="after")
    def validate_window_and_risks(self) -> StagingChangeAuthorization:
        if not self.approved_at <= self.not_before < self.expires_at:
            raise ValueError("authorization window is invalid")
        if self.expires_at - self.not_before > _MAX_AUTHORIZATION_WINDOW:
            raise ValueError("authorization window cannot exceed seven days")
        if len(set(self.allowed_risk_classes)) != len(self.allowed_risk_classes):
            raise ValueError("authorization risk classes must be unique")
        if set(self.allowed_risk_classes) != set(ScenarioRisk):
            raise ValueError("authorization must cover every fixed risk class")
        return self


class StagingScenarioPolicy(_ClosedModel):
    scenario: UnverifiedScenario
    risk_class: ScenarioRisk
    timeout_seconds: int = Field(ge=30, le=7200)

    @model_validator(mode="after")
    def validate_risk(self) -> StagingScenarioPolicy:
        if self.risk_class != SCENARIO_RISKS[self.scenario]:
            raise ValueError("scenario risk classification does not match platform policy")
        return self


class StagingDynamicExecutionPlan(_ClosedModel):
    schema_version: Literal[1]
    environment: Literal["STAGING"]
    environment_id: SafeIdentity
    kube_context: SafeIdentity
    git_commit: GitCommit
    static_plan_sha256: Digest
    run_id: SafeIdentity
    runner_version: SafeIdentity
    driver: StagingDriverBinding
    authorization: StagingChangeAuthorization
    scenarios: tuple[StagingScenarioPolicy, ...] = Field(min_length=15, max_length=15)

    @field_validator("environment_id")
    @classmethod
    def validate_environment_id(cls, value: str) -> str:
        return _validated(value, _DNS_NAME, "invalid environment identity")

    @field_validator("kube_context")
    @classmethod
    def validate_context(cls, value: str) -> str:
        return _validated(value, _SAFE_CONTEXT, "invalid Kubernetes context")

    @field_validator("run_id", "runner_version")
    @classmethod
    def validate_runner_identity(cls, value: str) -> str:
        return _validated(value, _SAFE_RUNNER, "invalid Runner identity")

    @model_validator(mode="after")
    def validate_scenario_set(self) -> StagingDynamicExecutionPlan:
        actual = tuple(item.scenario for item in self.scenarios)
        expected = tuple(UnverifiedScenario)
        if actual != expected:
            raise ValueError("execution plan must contain every approved scenario in order")
        return self


class ScenarioDriverRequest(_ClosedModel):
    schema_version: Literal[1]
    run_id: SafeIdentity
    environment: Literal["STAGING"]
    environment_id: SafeIdentity
    kube_context: SafeIdentity
    git_commit: GitCommit
    static_plan_sha256: Digest
    authorization_reference: AuthorizationReference
    scenario: UnverifiedScenario
    risk_class: ScenarioRisk
    required_assertions: tuple[AssertionIdentity, ...] = Field(
        min_length=1, max_length=4
    )
    evidence_directory: str = Field(min_length=1, max_length=4096)

    @field_validator("evidence_directory")
    @classmethod
    def validate_evidence_directory(cls, value: str) -> str:
        if "\x00" in value or not Path(value).is_absolute():
            raise ValueError("driver evidence directory must be an absolute safe path")
        return value

    @model_validator(mode="after")
    def validate_closed_scenario(self) -> ScenarioDriverRequest:
        if self.risk_class != SCENARIO_RISKS[self.scenario]:
            raise ValueError("driver request risk classification drifted")
        if self.required_assertions != REQUIRED_ASSERTIONS[self.scenario]:
            raise ValueError("driver request assertion contract drifted")
        return self


class DriverAssertionReceipt(_ClosedModel):
    assertion_id: AssertionIdentity
    passed: bool
    observation_code: ObservationCode
    observation_file: RelativeEvidencePath

    @field_validator("observation_file")
    @classmethod
    def validate_observation_file(cls, value: str) -> str:
        return _relative_file(value)


class ScenarioDriverReceipt(_ClosedModel):
    schema_version: Literal[1]
    scenario: UnverifiedScenario
    status: Literal["PASSED", "FAILED", "BLOCKED"]
    evidence_file: RelativeEvidencePath
    assertions: tuple[DriverAssertionReceipt, ...] = Field(min_length=1, max_length=4)

    @field_validator("evidence_file")
    @classmethod
    def validate_evidence_file(cls, value: str) -> str:
        return _relative_file(value)

    @model_validator(mode="after")
    def validate_assertions(self) -> ScenarioDriverReceipt:
        identities = tuple(item.assertion_id for item in self.assertions)
        if len(set(identities)) != len(identities):
            raise ValueError("driver assertion receipts must be unique")
        if set(identities) != set(REQUIRED_ASSERTIONS[self.scenario]):
            raise ValueError("driver receipt must contain every fixed assertion")
        return self


class ScenarioCleanupReceipt(_ClosedModel):
    schema_version: Literal[1]
    scenario: UnverifiedScenario
    cleaned: bool
    evidence_file: RelativeEvidencePath

    @field_validator("evidence_file")
    @classmethod
    def validate_evidence_file(cls, value: str) -> str:
        return _relative_file(value)


def _relative_file(value: str) -> str:
    if "\x00" in value or "\n" in value or "\r" in value or "\\" in value:
        raise ValueError("evidence file path contains unsafe characters")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("evidence file path must be a safe relative path")
    return value


def _validated(value: str, pattern: re.Pattern[str], message: str) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(message)
    return value


def _is_utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == timedelta(0)
