from __future__ import annotations

import hashlib
import json
import os
import selectors
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from industrial_ops_agent.staging_acceptance.contracts import StagingAcceptanceReport
from industrial_ops_agent.staging_acceptance.dynamic_contracts import (
    REQUIRED_ASSERTIONS,
    DynamicAcceptanceResults,
    DynamicAssertionEvidence,
    DynamicScenarioResult,
    normalized_utc,
)
from industrial_ops_agent.staging_acceptance.dynamic_manifest import (
    _healthy_static_preflight,
)
from industrial_ops_agent.staging_acceptance.runner_contracts import (
    DriverAssertionReceipt,
    ScenarioCleanupReceipt,
    ScenarioDriverReceipt,
    ScenarioDriverRequest,
    StagingDynamicExecutionPlan,
    StagingScenarioPolicy,
)

DriverFailure = Literal["NOT_FOUND", "TIMEOUT", "OUTPUT_LIMIT", "OS_ERROR"]
MAX_DRIVER_OUTPUT_BYTES = 1024 * 1024
MAX_EVIDENCE_FILE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class RawDriverResult:
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    failure: DriverFailure | None = None


class DriverProcessRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> RawDriverResult: ...


class SubprocessDriverRunner:
    def run(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> RawDriverResult:
        try:
            process = subprocess.Popen(  # noqa: S603
                list(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except FileNotFoundError:
            return RawDriverResult(None, b"", b"", "NOT_FOUND")
        except OSError:
            return RawDriverResult(None, b"", b"", "OS_ERROR")

        stdin = process.stdin
        stdout = process.stdout
        stderr = process.stderr
        assert stdin is not None and stdout is not None and stderr is not None
        streams = {stdout: bytearray(), stderr: bytearray()}
        selector = selectors.DefaultSelector()
        failure: DriverFailure | None = None
        deadline = time.monotonic() + timeout_seconds
        try:
            try:
                stdin.write(input_bytes)
                stdin.flush()
            except BrokenPipeError:
                pass
            finally:
                stdin.close()

            selector.register(stdout, selectors.EVENT_READ)
            selector.register(stderr, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure = "TIMEOUT"
                    break
                events = selector.select(remaining)
                if not events:
                    failure = "TIMEOUT"
                    break
                for key, _mask in events:
                    stream = key.fileobj
                    if stream not in streams:
                        failure = "OS_ERROR"
                        break
                    try:
                        chunk = os.read(stream.fileno(), 64 * 1024)
                    except OSError:
                        failure = "OS_ERROR"
                        break
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    buffer = streams[stream]
                    available = max(0, max_output_bytes - len(buffer))
                    buffer.extend(chunk[:available])
                    if len(chunk) > available:
                        failure = "OUTPUT_LIMIT"
                        break
                if failure is not None:
                    break
            if failure is None and process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure = "TIMEOUT"
                else:
                    try:
                        process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        failure = "TIMEOUT"
        finally:
            selector.close()
            if process.poll() is None and failure is not None:
                process.kill()
            process.wait()
            stdin.close()
            stdout.close()
            stderr.close()

        stdout_bytes = bytes(streams[stdout])
        stderr_bytes = bytes(streams[stderr])
        if failure is not None:
            return RawDriverResult(None, stdout_bytes, stderr_bytes, failure)
        return RawDriverResult(process.returncode, stdout_bytes, stderr_bytes)


class RunnerInputError(ValueError):
    pass


class _ScenarioFailure(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _MaterializedReceipt:
    receipt: ScenarioDriverReceipt
    evidence_ref: str
    artifact_digest: str
    assertions: tuple[DynamicAssertionEvidence, ...]


class StagingDynamicRunner:
    def __init__(
        self,
        *,
        process_runner: DriverProcessRunner | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._process_runner = process_runner or SubprocessDriverRunner()
        self._clock = clock or (lambda: datetime.now(UTC))

    def run(
        self,
        preflight: StagingAcceptanceReport,
        plan: StagingDynamicExecutionPlan,
        *,
        confirm_context: str,
        evidence_root: Path,
    ) -> DynamicAcceptanceResults:
        started = normalized_utc(self._clock())
        self._validate_before_execution(
            preflight,
            plan,
            confirm_context=confirm_context,
            now=started,
            evidence_root=evidence_root,
        )
        _create_evidence_root(evidence_root)

        scenarios: list[DynamicScenarioResult] = []
        for policy in plan.scenarios:
            scenario_started = normalized_utc(self._clock())
            scenario_dir = evidence_root / policy.scenario.value.lower()
            scenario_dir.mkdir(mode=0o700)
            if not _authorization_active(plan, scenario_started):
                scenarios.append(
                    self._blocked_without_driver(
                        policy,
                        scenario_dir,
                        scenario_started,
                        "execution_authorization_expired",
                    )
                )
                break
            try:
                _verify_driver(plan)
            except RunnerInputError as exc:
                scenarios.append(
                    self._blocked_without_driver(
                        policy,
                        scenario_dir,
                        scenario_started,
                        str(exc),
                    )
                )
                break

            request = _driver_request(plan, policy, scenario_dir)
            request_bytes = _canonical_json(request.model_dump(mode="json"))
            execute = self._process_runner.run(
                (plan.driver.executable_path, "execute"),
                input_bytes=request_bytes,
                timeout_seconds=float(policy.timeout_seconds),
                max_output_bytes=MAX_DRIVER_OUTPUT_BYTES,
            )
            execute_failure: _ScenarioFailure | None = None
            materialized: _MaterializedReceipt | None = None
            try:
                materialized = _materialize_execute_receipt(
                    execute,
                    policy=policy,
                    scenario_dir=scenario_dir,
                    evidence_root=evidence_root,
                )
            except _ScenarioFailure as exc:
                execute_failure = exc

            cleanup: RawDriverResult
            cleanup_digest: str | None = None
            cleanup_ok = False
            try:
                _verify_driver(plan)
            except RunnerInputError:
                cleanup = RawDriverResult(None, b"", b"", "OS_ERROR")
            else:
                cleanup = self._process_runner.run(
                    (plan.driver.executable_path, "cleanup"),
                    input_bytes=request_bytes,
                    timeout_seconds=float(policy.timeout_seconds),
                    max_output_bytes=MAX_DRIVER_OUTPUT_BYTES,
                )
                try:
                    cleanup_digest = _validate_cleanup_receipt(
                        cleanup,
                        policy=policy,
                        scenario_dir=scenario_dir,
                    )
                    cleanup_ok = True
                except _ScenarioFailure:
                    cleanup_ok = False

            scenario_completed = _completed_at(
                scenario_started,
                normalized_utc(self._clock()),
            )
            if not cleanup_ok:
                scenarios.append(
                    _blocked_result(
                        policy,
                        scenario_dir,
                        evidence_root,
                        scenario_started,
                        scenario_completed,
                        reason="driver_cleanup_failed",
                        execute=execute,
                        cleanup=cleanup,
                        cleanup_evidence_digest=cleanup_digest,
                    )
                )
                break
            if execute_failure is not None or materialized is None:
                scenarios.append(
                    _blocked_result(
                        policy,
                        scenario_dir,
                        evidence_root,
                        scenario_started,
                        scenario_completed,
                        reason=(
                            execute_failure.reason
                            if execute_failure is not None
                            else "driver_receipt_invalid"
                        ),
                        execute=execute,
                        cleanup=cleanup,
                        cleanup_evidence_digest=cleanup_digest,
                    )
                )
                continue
            scenarios.append(
                DynamicScenarioResult(
                    scenario=policy.scenario,
                    status=_effective_status(materialized),
                    started_at=scenario_started,
                    completed_at=scenario_completed,
                    evidence_ref=materialized.evidence_ref,
                    artifact_digest=materialized.artifact_digest,
                    assertions=materialized.assertions,
                )
            )

        return DynamicAcceptanceResults(
            schema_version=1,
            environment="STAGING",
            environment_id=plan.environment_id,
            kube_context=plan.kube_context,
            git_commit=plan.git_commit,
            static_plan_sha256=plan.static_plan_sha256,
            runner_version=plan.runner_version,
            scenarios=tuple(scenarios),
        )

    def _validate_before_execution(
        self,
        preflight: StagingAcceptanceReport,
        plan: StagingDynamicExecutionPlan,
        *,
        confirm_context: str,
        now: datetime,
        evidence_root: Path,
    ) -> None:
        if not _healthy_static_preflight(preflight):
            raise RunnerInputError("static_preflight_not_healthy")
        if preflight.generated_at > now:
            raise RunnerInputError("static_preflight_from_future")
        if not (
            preflight.environment == plan.environment
            and preflight.environment_id == plan.environment_id
            and preflight.kube_context == plan.kube_context
            and preflight.git_commit == plan.git_commit
            and preflight.plan_sha256 == plan.static_plan_sha256
        ):
            raise RunnerInputError("execution_plan_binding_mismatch")
        if confirm_context != plan.kube_context:
            raise RunnerInputError("confirmed_context_mismatch")
        if not _authorization_active(plan, now):
            raise RunnerInputError("execution_authorization_inactive")
        if evidence_root.exists() or evidence_root.is_symlink():
            raise FileExistsError("evidence_directory_exists")
        _verify_driver(plan)

    def _blocked_without_driver(
        self,
        policy: StagingScenarioPolicy,
        scenario_dir: Path,
        started_at: datetime,
        reason: str,
    ) -> DynamicScenarioResult:
        completed_at = _completed_at(started_at, normalized_utc(self._clock()))
        return _blocked_result(
            policy,
            scenario_dir,
            scenario_dir.parent,
            started_at,
            completed_at,
            reason=reason,
            execute=None,
            cleanup=None,
            cleanup_evidence_digest=None,
        )


def _driver_request(
    plan: StagingDynamicExecutionPlan,
    policy: StagingScenarioPolicy,
    scenario_dir: Path,
) -> ScenarioDriverRequest:
    return ScenarioDriverRequest(
        schema_version=1,
        run_id=plan.run_id,
        environment="STAGING",
        environment_id=plan.environment_id,
        kube_context=plan.kube_context,
        git_commit=plan.git_commit,
        static_plan_sha256=plan.static_plan_sha256,
        authorization_reference=plan.authorization.reference,
        scenario=policy.scenario,
        risk_class=policy.risk_class,
        required_assertions=REQUIRED_ASSERTIONS[policy.scenario],
        evidence_directory=str(scenario_dir.resolve()),
    )


def _materialize_execute_receipt(
    raw: RawDriverResult,
    *,
    policy: StagingScenarioPolicy,
    scenario_dir: Path,
    evidence_root: Path,
) -> _MaterializedReceipt:
    _require_successful_process(raw, phase="driver")
    try:
        payload = _load_closed_json(raw.stdout)
        receipt = ScenarioDriverReceipt.model_validate(payload)
    except (UnicodeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        reason = (
            "driver_evidence_invalid"
            if "evidence file path" in str(exc)
            else "driver_receipt_invalid"
        )
        raise _ScenarioFailure(reason) from exc
    if receipt.scenario != policy.scenario:
        raise _ScenarioFailure("driver_receipt_invalid")
    try:
        artifact = _read_scenario_file(scenario_dir, receipt.evidence_file)
        assertions = tuple(
            _materialize_assertion(scenario_dir, item) for item in receipt.assertions
        )
    except (OSError, ValueError) as exc:
        raise _ScenarioFailure("driver_evidence_invalid") from exc
    evidence_path = scenario_dir / receipt.evidence_file
    evidence_ref = evidence_path.relative_to(evidence_root).as_posix()
    return _MaterializedReceipt(
        receipt=receipt,
        evidence_ref=evidence_ref,
        artifact_digest=_digest(artifact),
        assertions=assertions,
    )


def _materialize_assertion(
    scenario_dir: Path,
    receipt: DriverAssertionReceipt,
) -> DynamicAssertionEvidence:
    observation = _read_scenario_file(scenario_dir, receipt.observation_file)
    return DynamicAssertionEvidence(
        assertion_id=receipt.assertion_id,
        passed=receipt.passed,
        observation_code=receipt.observation_code,
        observation_digest=_digest(observation),
    )


def _validate_cleanup_receipt(
    raw: RawDriverResult,
    *,
    policy: StagingScenarioPolicy,
    scenario_dir: Path,
) -> str:
    _require_successful_process(raw, phase="cleanup")
    try:
        payload = _load_closed_json(raw.stdout)
        receipt = ScenarioCleanupReceipt.model_validate(payload)
        if receipt.scenario != policy.scenario or not receipt.cleaned:
            raise ValueError("cleanup was not confirmed")
        evidence = _read_scenario_file(scenario_dir, receipt.evidence_file)
    except (UnicodeError, OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        raise _ScenarioFailure("driver_cleanup_failed") from exc
    return _digest(evidence)


def _require_successful_process(raw: RawDriverResult, *, phase: str) -> None:
    if raw.failure == "TIMEOUT":
        raise _ScenarioFailure(f"{phase}_timeout")
    if raw.failure == "OUTPUT_LIMIT":
        raise _ScenarioFailure(f"{phase}_output_limit")
    if raw.failure is not None:
        raise _ScenarioFailure(f"{phase}_start_failed")
    if raw.exit_code != 0:
        raise _ScenarioFailure(f"{phase}_exit_nonzero")


def _blocked_result(
    policy: StagingScenarioPolicy,
    scenario_dir: Path,
    evidence_root: Path,
    started_at: datetime,
    completed_at: datetime,
    *,
    reason: str,
    execute: RawDriverResult | None,
    cleanup: RawDriverResult | None,
    cleanup_evidence_digest: str | None,
) -> DynamicScenarioResult:
    record = {
        "schema_version": 1,
        "scenario": policy.scenario.value,
        "reason": reason,
        "execute": _raw_summary(execute),
        "cleanup": _raw_summary(cleanup),
        "cleanup_evidence_digest": cleanup_evidence_digest,
    }
    record_path = scenario_dir / "runner-execution.json"
    record_bytes = _write_json_exclusive(record_path, record)
    record_digest = _digest(record_bytes)
    return DynamicScenarioResult(
        scenario=policy.scenario,
        status="BLOCKED",
        started_at=started_at,
        completed_at=completed_at,
        evidence_ref=record_path.relative_to(evidence_root).as_posix(),
        artifact_digest=record_digest,
        assertions=tuple(
            DynamicAssertionEvidence(
                assertion_id=assertion_id,
                passed=False,
                observation_code=reason,
                observation_digest=record_digest,
            )
            for assertion_id in REQUIRED_ASSERTIONS[policy.scenario]
        ),
    )


def _raw_summary(raw: RawDriverResult | None) -> Mapping[str, Any] | None:
    if raw is None:
        return None
    return {
        "exit_code": raw.exit_code,
        "failure": raw.failure,
        "stdout_sha256": _digest(raw.stdout),
        "stderr_sha256": _digest(raw.stderr),
    }


def _authorization_active(plan: StagingDynamicExecutionPlan, now: datetime) -> bool:
    authorization = plan.authorization
    return authorization.not_before <= now < authorization.expires_at


def _effective_status(
    materialized: _MaterializedReceipt,
) -> Literal["PASSED", "FAILED", "BLOCKED"]:
    if materialized.receipt.status == "PASSED" and not all(
        assertion.passed for assertion in materialized.assertions
    ):
        return "BLOCKED"
    return materialized.receipt.status


def _verify_driver(plan: StagingDynamicExecutionPlan) -> None:
    path = Path(plan.driver.executable_path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RunnerInputError("driver_invalid") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o111 == 0
    ):
        raise RunnerInputError("driver_invalid")
    try:
        raw = _read_regular_file(path, max_bytes=MAX_EVIDENCE_FILE_BYTES)
    except (OSError, ValueError) as exc:
        raise RunnerInputError("driver_invalid") from exc
    if _digest(raw) != plan.driver.sha256:
        raise RunnerInputError("driver_digest_mismatch")


def _create_evidence_root(path: Path) -> None:
    parent = path.parent
    metadata = parent.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or parent.is_symlink():
        raise OSError("evidence parent is not a safe directory")
    os.mkdir(path, mode=0o700)
    path.chmod(0o700)


def _read_scenario_file(scenario_dir: Path, relative: str) -> bytes:
    base = scenario_dir.resolve(strict=True)
    current = scenario_dir
    parts = Path(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        metadata = current.lstat()
        if current.is_symlink():
            raise ValueError("evidence path cannot contain symlinks")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("evidence parent component is not a directory")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(base):
        raise ValueError("evidence path escapes scenario directory")
    return _read_regular_file(current, max_bytes=MAX_EVIDENCE_FILE_BYTES)


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb", closefd=True) as stream:
        metadata = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > max_bytes
        ):
            raise ValueError("evidence file size is invalid")
        raw = stream.read(max_bytes + 1)
    if not raw or len(raw) > max_bytes:
        raise ValueError("evidence file size is invalid")
    return raw


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> bytes:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return encoded


def _load_closed_json(raw: bytes) -> Mapping[str, Any]:
    if not raw or len(raw) > MAX_DRIVER_OUTPUT_BYTES:
        raise ValueError("driver output size is invalid")
    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_closed_object)
    if not isinstance(payload, dict):
        raise ValueError("driver output must be a JSON object")
    return payload


def _closed_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _completed_at(started_at: datetime, observed: datetime) -> datetime:
    minimum = started_at + timedelta(microseconds=1)
    return observed if observed >= minimum else minimum


def _as_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="replace")
