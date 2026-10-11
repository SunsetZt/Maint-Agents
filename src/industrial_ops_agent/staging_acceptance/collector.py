from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal, Protocol, cast

from industrial_ops_agent.staging_acceptance.contracts import (
    CheckEvidence,
    CommandEvidence,
    DeploymentExpectation,
    GitOpsApplicationExpectation,
    HpaExpectation,
    NamedResourceExpectation,
    ScaledObjectExpectation,
    StagingAcceptancePlan,
    StagingAcceptanceReport,
    UnverifiedEvidence,
    UnverifiedScenario,
)

FailureKind = Literal["NOT_FOUND", "TIMEOUT", "OUTPUT_LIMIT", "OS_ERROR"]
MAX_COMMAND_OUTPUT_BYTES = 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class RawCommandResult:
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    failure: FailureKind | None = None


class CommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> RawCommandResult: ...


class SubprocessCommandRunner:
    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> RawCommandResult:
        try:
            completed = subprocess.run(  # noqa: S603
                list(argv),
                check=False,
                capture_output=True,
                shell=False,
                timeout=timeout_seconds,
            )
        except FileNotFoundError:
            return RawCommandResult(None, b"", b"", "NOT_FOUND")
        except subprocess.TimeoutExpired as exc:
            return RawCommandResult(
                None,
                _as_bytes(exc.stdout),
                _as_bytes(exc.stderr),
                "TIMEOUT",
            )
        except OSError:
            return RawCommandResult(None, b"", b"", "OS_ERROR")
        if len(completed.stdout) > max_output_bytes or len(completed.stderr) > max_output_bytes:
            return RawCommandResult(None, completed.stdout, completed.stderr, "OUTPUT_LIMIT")
        return RawCommandResult(completed.returncode, completed.stdout, completed.stderr)


class ProjectionFailure(ValueError):
    pass


class StagingAcceptanceCollector:
    def __init__(
        self,
        *,
        runner: CommandRunner | None = None,
        kubectl_path: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._runner = runner or SubprocessCommandRunner()
        self._kubectl_path = kubectl_path
        self._clock = clock or (lambda: datetime.now(UTC))

    def collect(self, plan: StagingAcceptancePlan) -> StagingAcceptanceReport:
        checks: list[CheckEvidence] = []
        kubectl = self._kubectl_path or shutil.which("kubectl")
        if kubectl is None:
            checks.append(_blocked("kubectl.available", "kubectl_not_found"))
            return self._report(plan, checks)

        context_argv = (kubectl, "config", "current-context")
        context_result = self._runner.run(
            context_argv,
            timeout_seconds=COMMAND_TIMEOUT_SECONDS,
            max_output_bytes=MAX_COMMAND_OUTPUT_BYTES,
        )
        context_error = _transport_reason(context_result)
        if context_error is not None:
            checks.append(
                _blocked(
                    "kubernetes.context",
                    context_error,
                    _command_evidence(context_argv, context_result),
                )
            )
            return self._report(plan, checks)
        current_context = context_result.stdout.decode("utf-8", errors="replace").strip()
        if current_context != plan.kube_context:
            checks.append(
                _blocked(
                    "kubernetes.context",
                    "kube_context_mismatch",
                    _command_evidence(context_argv, context_result),
                    {"matches_approved_context": False},
                )
            )
            return self._report(plan, checks)
        checks.append(
            _passed(
                "kubernetes.context",
                _command_evidence(context_argv, context_result),
                {"matches_approved_context": True},
            )
        )

        checks.append(
            self._json_check(
                "kubernetes.version",
                (kubectl, "--context", plan.kube_context, "version", "-o", "json"),
                lambda payload: _project_version(payload, plan.kubernetes_server_version),
            )
        )
        checks.extend(
            (
                self._json_check(
                    "crd.scaledobjects",
                    _get_argv(kubectl, plan, "crd", "scaledobjects.keda.sh"),
                    lambda payload: _project_condition_resource(
                        payload, "scaledobjects.keda.sh", "Established", "crd_not_established"
                    ),
                ),
                self._json_check(
                    "crd.inferenceservices",
                    _get_argv(
                        kubectl, plan, "crd", "inferenceservices.serving.kserve.io"
                    ),
                    lambda payload: _project_condition_resource(
                        payload,
                        "inferenceservices.serving.kserve.io",
                        "Established",
                        "crd_not_established",
                    ),
                ),
                self._json_check(
                    "apiservice.external-metrics",
                    _get_argv(
                        kubectl, plan, "apiservice", "v1beta1.external.metrics.k8s.io"
                    ),
                    lambda payload: _project_condition_resource(
                        payload,
                        "v1beta1.external.metrics.k8s.io",
                        "Available",
                        "external_metrics_api_unavailable",
                    ),
                ),
            )
        )
        for application in plan.gitops_applications:
            checks.append(
                self._json_check(
                    f"gitops.{application.component.value.lower()}",
                    _get_argv(
                        kubectl,
                        plan,
                        "applications.argoproj.io",
                        application.name,
                        application.namespace,
                    ),
                    partial(_project_application, expected=application),
                )
            )
        for deployment in plan.deployments:
            checks.append(
                self._json_check(
                    f"deployment.{deployment.namespace}.{deployment.name}",
                    _get_argv(
                        kubectl, plan, "deployment", deployment.name, deployment.namespace
                    ),
                    partial(_project_deployment, expected=deployment),
                )
            )
        for hpa in plan.hpas:
            checks.append(
                self._json_check(
                    f"hpa.{hpa.namespace}.{hpa.name}",
                    _get_argv(kubectl, plan, "hpa", hpa.name, hpa.namespace),
                    partial(_project_hpa, expected=hpa),
                )
            )
        for scaled_object in plan.scaled_objects:
            checks.append(
                self._json_check(
                    f"scaledobject.{scaled_object.namespace}.{scaled_object.name}",
                    _get_argv(
                        kubectl,
                        plan,
                        "scaledobject.keda.sh",
                        scaled_object.name,
                        scaled_object.namespace,
                    ),
                    partial(_project_scaled_object, expected=scaled_object),
                )
            )
        for inference_service in plan.inference_services:
            checks.append(
                self._json_check(
                    f"inferenceservice.{inference_service.namespace}.{inference_service.name}",
                    _get_argv(
                        kubectl,
                        plan,
                        "inferenceservice.serving.kserve.io",
                        inference_service.name,
                        inference_service.namespace,
                    ),
                    partial(_project_inference_service, expected=inference_service),
                )
            )
        for pod_monitor in plan.pod_monitors:
            checks.append(
                self._json_check(
                    f"podmonitor.{pod_monitor.namespace}.{pod_monitor.name}",
                    _get_argv(
                        kubectl,
                        plan,
                        "podmonitor.monitoring.coreos.com",
                        pod_monitor.name,
                        pod_monitor.namespace,
                    ),
                    partial(_project_pod_monitor, expected=pod_monitor),
                )
            )
        return self._report(plan, checks)

    def _json_check(
        self,
        check_id: str,
        argv: tuple[str, ...],
        projector: Callable[[Mapping[str, Any]], dict[str, bool | int | str | None]],
    ) -> CheckEvidence:
        result = self._runner.run(
            argv,
            timeout_seconds=COMMAND_TIMEOUT_SECONDS,
            max_output_bytes=MAX_COMMAND_OUTPUT_BYTES,
        )
        command = _command_evidence(argv, result)
        reason = _transport_reason(result)
        if reason is not None:
            return _blocked(check_id, reason, command)
        try:
            raw = json.loads(result.stdout)
            if not isinstance(raw, dict):
                raise ValueError
            facts = projector(cast(Mapping[str, Any], raw))
        except ProjectionFailure as exc:
            return _blocked(check_id, str(exc), command)
        except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
            return _blocked(check_id, "invalid_json", command)
        return _passed(check_id, command, facts)

    def _report(
        self,
        plan: StagingAcceptancePlan,
        checks: Sequence[CheckEvidence],
    ) -> StagingAcceptanceReport:
        generated_at = self._clock().astimezone(UTC)
        blocked = sum(item.status == "BLOCKED" for item in checks)
        plan_payload = plan.model_dump(mode="json")
        plan_sha256 = _digest(_canonical(plan_payload))
        unverified = tuple(UnverifiedEvidence(scenario=item) for item in UnverifiedScenario)
        identity_payload = {
            "collector_version": "staging-acceptance-collector/v1",
            "generated_at": generated_at.isoformat(),
            "plan_sha256": plan_sha256,
            "checks": [item.model_dump(mode="json") for item in checks],
            "unverified_scenarios": [item.model_dump(mode="json") for item in unverified],
        }
        return StagingAcceptanceReport(
            schema_version=1,
            report_id=_digest(_canonical(identity_payload)),
            collector_version="staging-acceptance-collector/v1",
            generated_at=generated_at,
            status=(
                "BLOCKED" if blocked else "PREFLIGHT_PASSED_ACCEPTANCE_INCOMPLETE"
            ),
            environment=plan.environment,
            environment_id=plan.environment_id,
            kube_context=plan.kube_context,
            git_commit=plan.git_commit,
            plan_sha256=plan_sha256,
            summary={"passed": len(checks) - blocked, "blocked": blocked},
            checks=tuple(checks),
            unverified_scenarios=unverified,
            limitation=(
                "Static preflight only; dynamic Staging scenarios and production sign-offs "
                "remain required."
            ),
        )


def _project_version(
    payload: Mapping[str, Any], expected: str
) -> dict[str, bool | int | str | None]:
    observed = _mapping(payload.get("serverVersion")).get("gitVersion")
    if observed != expected:
        raise ProjectionFailure("kubernetes_version_mismatch")
    return {"server_version": cast(str, observed)}


def _project_condition_resource(
    payload: Mapping[str, Any],
    expected_name: str,
    condition: str,
    failure: str,
) -> dict[str, bool | int | str | None]:
    _identity(payload, expected_name)
    ready = _conditions(payload).get(condition) == "True"
    if not ready:
        raise ProjectionFailure(failure)
    return {"name": expected_name, condition.lower(): True}


def _project_application(
    payload: Mapping[str, Any], expected: GitOpsApplicationExpectation
) -> dict[str, bool | int | str | None]:
    _identity(payload, expected.name, expected.namespace)
    spec_source = _mapping(_mapping(payload.get("spec")).get("source"))
    status = _mapping(payload.get("status"))
    sync = _mapping(status.get("sync"))
    health = _mapping(status.get("health"))
    configured = spec_source.get("targetRevision")
    if (
        configured != expected.expected_revision
        or sync.get("status") != "Synced"
        or health.get("status") != "Healthy"
    ):
        raise ProjectionFailure("gitops_application_not_healthy")
    return {
        "component": expected.component.value,
        "configured_revision": cast(str, configured),
        "observed_revision": _safe_optional_string(sync.get("revision")),
        "synced": True,
        "healthy": True,
    }


def _project_deployment(
    payload: Mapping[str, Any], expected: DeploymentExpectation
) -> dict[str, bool | int | str | None]:
    metadata = _identity(payload, expected.name, expected.namespace)
    spec = _mapping(payload.get("spec"))
    status = _mapping(payload.get("status"))
    generation = _integer(metadata.get("generation"))
    observed = _integer(status.get("observedGeneration"))
    desired = _integer(spec.get("replicas"))
    ready = _integer(status.get("readyReplicas"))
    available = _integer(status.get("availableReplicas"))
    if (
        generation is None
        or observed != generation
        or desired is None
        or desired < expected.min_ready_replicas
        or ready is None
        or ready < expected.min_ready_replicas
        or available is None
        or available < expected.min_ready_replicas
    ):
        raise ProjectionFailure("deployment_not_ready")
    return {"desired_replicas": desired, "ready_replicas": ready, "available_replicas": available}


def _project_hpa(
    payload: Mapping[str, Any], expected: HpaExpectation
) -> dict[str, bool | int | str | None]:
    _identity(payload, expected.name, expected.namespace)
    spec = _mapping(payload.get("spec"))
    status = _mapping(payload.get("status"))
    minimum = _integer(spec.get("minReplicas"))
    maximum = _integer(spec.get("maxReplicas"))
    current = _integer(status.get("currentReplicas"))
    desired = _integer(status.get("desiredReplicas"))
    able = _conditions(payload).get("AbleToScale") == "True"
    if (
        minimum != expected.min_replicas
        or maximum != expected.max_replicas
        or current is None
        or desired is None
        or not able
    ):
        raise ProjectionFailure("hpa_not_ready")
    return {
        "min_replicas": minimum,
        "max_replicas": maximum,
        "current_replicas": current,
        "desired_replicas": desired,
        "able_to_scale": True,
    }


def _project_scaled_object(
    payload: Mapping[str, Any], expected: ScaledObjectExpectation
) -> dict[str, bool | int | str | None]:
    _identity(payload, expected.name, expected.namespace)
    spec = _mapping(payload.get("spec"))
    conditions = _conditions(payload)
    minimum = _integer(spec.get("minReplicaCount"))
    maximum = _integer(spec.get("maxReplicaCount"))
    ready = conditions.get("Ready") == "True"
    active = conditions.get("Active") == "True"
    fallback = conditions.get("Fallback") == "True"
    if (
        minimum != expected.min_replicas
        or maximum != expected.max_replicas
        or not ready
        or not active
        or fallback
    ):
        raise ProjectionFailure("scaled_object_not_ready")
    return {
        "min_replicas": minimum,
        "max_replicas": maximum,
        "ready": ready,
        "active": active,
        "fallback": fallback,
    }


def _project_inference_service(
    payload: Mapping[str, Any], expected: NamedResourceExpectation
) -> dict[str, bool | int | str | None]:
    metadata = _identity(payload, expected.name, expected.namespace)
    status = _mapping(payload.get("status"))
    generation = _integer(metadata.get("generation"))
    observed = _integer(status.get("observedGeneration"))
    revision = _safe_optional_string(status.get("latestReadyRevision"))
    if (
        generation is None
        or observed != generation
        or _conditions(payload).get("Ready") != "True"
        or revision is None
    ):
        raise ProjectionFailure("inference_service_not_ready")
    return {"ready": True, "latest_ready_revision": revision}


def _project_pod_monitor(
    payload: Mapping[str, Any], expected: NamedResourceExpectation
) -> dict[str, bool | int | str | None]:
    _identity(payload, expected.name, expected.namespace)
    spec = _mapping(payload.get("spec"))
    selector = _mapping(spec.get("selector"))
    expressions = selector.get("matchExpressions")
    endpoints = spec.get("podMetricsEndpoints")
    keys = (
        sorted(
            item["key"]
            for item in expressions
            if isinstance(item, dict) and isinstance(item.get("key"), str)
        )
        if isinstance(expressions, list)
        else []
    )
    valid_endpoint = isinstance(endpoints, list) and any(
        isinstance(item, dict) and item.get("port") == "http" and item.get("path") == "/metrics"
        for item in endpoints
    )
    if "serving.kserve.io/inferenceservice" not in keys or not valid_endpoint:
        raise ProjectionFailure("pod_monitor_not_ready")
    return {
        "endpoint_count": len(cast(list[object], endpoints)),
        "has_inference_service_selector": True,
    }


def _identity(
    payload: Mapping[str, Any], expected_name: str, expected_namespace: str | None = None
) -> Mapping[str, Any]:
    metadata = _mapping(payload.get("metadata"))
    if metadata.get("name") != expected_name or (
        expected_namespace is not None and metadata.get("namespace") != expected_namespace
    ):
        raise ProjectionFailure("resource_identity_mismatch")
    return metadata


def _conditions(payload: Mapping[str, Any]) -> dict[str, str]:
    raw = _mapping(payload.get("status")).get("conditions")
    if not isinstance(raw, list):
        return {}
    return {
        item["type"]: item["status"]
        for item in raw
        if isinstance(item, dict)
        and isinstance(item.get("type"), str)
        and isinstance(item.get("status"), str)
    }


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, dict) else {}


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _safe_optional_string(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 253
        or any(ch.isspace() for ch in value)
    ):
        return None
    return value


def _get_argv(
    kubectl: str,
    plan: StagingAcceptancePlan,
    kind: str,
    name: str,
    namespace: str | None = None,
) -> tuple[str, ...]:
    argv = [kubectl, "--context", plan.kube_context, "get", kind, name]
    if namespace is not None:
        argv.extend(("-n", namespace))
    argv.extend(("-o", "json"))
    return tuple(argv)


def _transport_reason(result: RawCommandResult) -> str | None:
    reasons = {
        "NOT_FOUND": "kubectl_not_found",
        "TIMEOUT": "command_timeout",
        "OUTPUT_LIMIT": "command_output_limit",
        "OS_ERROR": "command_os_error",
    }
    if result.failure is not None:
        return reasons[result.failure]
    if result.exit_code != 0:
        return "command_failed"
    return None


def _command_evidence(argv: Sequence[str], result: RawCommandResult) -> CommandEvidence:
    return CommandEvidence(
        argv=tuple(argv),
        exit_code=result.exit_code,
        stdout_sha256=_digest(result.stdout),
        stderr_sha256=_digest(result.stderr),
    )


def _passed(
    check_id: str,
    command: CommandEvidence,
    facts: dict[str, bool | int | str | None],
) -> CheckEvidence:
    return CheckEvidence(
        check_id=check_id,
        status="PASS",
        reason="ready",
        command=command,
        facts=facts,
    )


def _blocked(
    check_id: str,
    reason: str,
    command: CommandEvidence | None = None,
    facts: dict[str, bool | int | str | None] | None = None,
) -> CheckEvidence:
    return CheckEvidence(
        check_id=check_id,
        status="BLOCKED",
        reason=reason,
        command=command,
        facts=facts or {},
    )


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _as_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value.encode() if isinstance(value, str) else value
