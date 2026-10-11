"""Fail-closed Kubernetes Lease leader election for the release controller."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from industrial_ops_agent.deployment.kserve import KubernetesApiError

_LEASE_API_VERSION = "coordination.k8s.io/v1"
_LEASE_KIND = "Lease"
_LEASE_PLURAL = "leases"


class LeaseResourceApi(Protocol):
    def get(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
    ) -> dict[str, Any]: ...

    def create(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        document: dict[str, Any],
    ) -> dict[str, Any]: ...

    def replace(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
        document: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class LeaseElectionConfig:
    namespace: str
    lease_name: str
    holder_identity: str
    lease_duration_seconds: int = 15
    renew_deadline_seconds: int = 10
    retry_period_seconds: int = 2

    def __post_init__(self) -> None:
        if not self.namespace.strip() or not self.lease_name.strip():
            raise ValueError("Lease namespace and name are required")
        if not self.holder_identity.strip():
            raise ValueError("Lease holder identity is required")
        timings = (
            self.retry_period_seconds,
            self.renew_deadline_seconds,
            self.lease_duration_seconds,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in timings):
            raise ValueError("Lease timing values must be integers")
        if not (
            0
            < self.retry_period_seconds
            < self.renew_deadline_seconds
            < self.lease_duration_seconds
        ):
            raise ValueError("Lease timing must satisfy 0 < retry < renew deadline < duration")


@dataclass(frozen=True, slots=True)
class _ObservedLease:
    resource_version: str
    holder_identity: str
    lease_duration_seconds: int
    acquire_time: datetime
    renew_time: datetime
    transitions: int


class KubernetesLeaseElector:
    """Maintain one local leadership decision backed by a Kubernetes Lease."""

    def __init__(
        self,
        api: LeaseResourceApi,
        config: LeaseElectionConfig,
        *,
        utc_now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._api = api
        self._config = config
        self._utc_now = utc_now or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._state_lock = threading.Lock()
        self._is_leader = False
        self._last_successful_renewal: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_leader(self) -> bool:
        with self._state_lock:
            self._expire_local_leadership_locked()
            return self._is_leader

    def try_acquire_or_renew(self) -> bool:
        now = _utc(self._utc_now())
        try:
            current = self._get_lease()
        except KubernetesApiError as exc:
            if exc.reason_code != "kubernetes_api_not_found":
                return self._leadership_after_dependency_failure()
            return self._try_create(now)

        try:
            observed = _parse_lease(current, self._config, now)
        except ValueError:
            self._lose_leadership()
            return False

        if observed.holder_identity == self._config.holder_identity:
            document = self._lease_document(
                now=now,
                acquire_time=observed.acquire_time,
                transitions=observed.transitions,
                resource_version=observed.resource_version,
            )
            return self._try_replace(
                document,
                expected_renew_time=now,
                retain_until_deadline=True,
            )

        if observed.renew_time + timedelta(seconds=observed.lease_duration_seconds) <= now:
            document = self._lease_document(
                now=now,
                acquire_time=now,
                transitions=observed.transitions + 1,
                resource_version=observed.resource_version,
            )
            self._lose_leadership()
            return self._try_replace(
                document,
                expected_renew_time=now,
                retain_until_deadline=False,
            )

        self._lose_leadership()
        return False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.try_acquire_or_renew()
        self._thread = threading.Thread(
            target=self._renew_loop,
            name="release-controller-lease-renewal",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._thread = None
        self._lose_leadership()

    def _renew_loop(self) -> None:
        while not self._stop_event.wait(self._config.retry_period_seconds):
            self.try_acquire_or_renew()

    def _get_lease(self) -> dict[str, Any]:
        return self._api.get(
            api_version=_LEASE_API_VERSION,
            plural=_LEASE_PLURAL,
            namespace=self._config.namespace,
            name=self._config.lease_name,
        )

    def _try_create(self, now: datetime) -> bool:
        document = self._lease_document(now=now, acquire_time=now, transitions=0)
        try:
            response = self._api.create(
                api_version=_LEASE_API_VERSION,
                plural=_LEASE_PLURAL,
                namespace=self._config.namespace,
                document=document,
            )
        except KubernetesApiError:
            self._lose_leadership()
            return False
        return self._confirm_mutation_response(response, now)

    def _try_replace(
        self,
        document: dict[str, Any],
        *,
        expected_renew_time: datetime,
        retain_until_deadline: bool,
    ) -> bool:
        previous_resource_version = document.get("metadata", {}).get("resourceVersion")
        if not isinstance(previous_resource_version, str) or not previous_resource_version:
            self._lose_leadership()
            return False
        try:
            response = self._api.replace(
                api_version=_LEASE_API_VERSION,
                plural=_LEASE_PLURAL,
                namespace=self._config.namespace,
                name=self._config.lease_name,
                document=document,
            )
        except KubernetesApiError as exc:
            if retain_until_deadline and exc.reason_code not in {
                "kubernetes_api_conflict",
                "kubernetes_api_not_found",
                "kubernetes_api_rejected_resource",
            }:
                return self._leadership_after_dependency_failure()
            self._lose_leadership()
            return False
        return self._confirm_mutation_response(
            response,
            expected_renew_time,
            previous_resource_version=previous_resource_version,
        )

    def _confirm_mutation_response(
        self,
        response: dict[str, Any],
        expected_renew_time: datetime,
        *,
        previous_resource_version: str | None = None,
    ) -> bool:
        expected_renew_time = _utc(expected_renew_time)
        try:
            observed = _parse_lease(response, self._config, expected_renew_time)
        except ValueError:
            self._lose_leadership()
            return False
        if (
            observed.holder_identity != self._config.holder_identity
            or observed.lease_duration_seconds != self._config.lease_duration_seconds
            or observed.renew_time != expected_renew_time
            or observed.resource_version == previous_resource_version
        ):
            self._lose_leadership()
            return False
        self._mark_renewed()
        return True

    def _lease_document(
        self,
        *,
        now: datetime,
        acquire_time: datetime,
        transitions: int,
        resource_version: str | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "name": self._config.lease_name,
            "namespace": self._config.namespace,
        }
        if resource_version is not None:
            metadata["resourceVersion"] = resource_version
        return {
            "apiVersion": _LEASE_API_VERSION,
            "kind": _LEASE_KIND,
            "metadata": metadata,
            "spec": {
                "holderIdentity": self._config.holder_identity,
                "leaseDurationSeconds": self._config.lease_duration_seconds,
                "acquireTime": _rfc3339(acquire_time),
                "renewTime": _rfc3339(now),
                "leaseTransitions": transitions,
            },
        }

    def _mark_renewed(self) -> None:
        with self._state_lock:
            self._is_leader = True
            self._last_successful_renewal = self._monotonic()

    def _leadership_after_dependency_failure(self) -> bool:
        with self._state_lock:
            self._expire_local_leadership_locked()
            return self._is_leader

    def _lose_leadership(self) -> None:
        with self._state_lock:
            self._is_leader = False
            self._last_successful_renewal = None

    def _expire_local_leadership_locked(self) -> None:
        if not self._is_leader or self._last_successful_renewal is None:
            return
        elapsed = self._monotonic() - self._last_successful_renewal
        if elapsed >= self._config.renew_deadline_seconds:
            self._is_leader = False
            self._last_successful_renewal = None


def _parse_lease(
    document: dict[str, Any],
    config: LeaseElectionConfig,
    now: datetime,
) -> _ObservedLease:
    if document.get("apiVersion") != _LEASE_API_VERSION or document.get("kind") != _LEASE_KIND:
        raise ValueError("unexpected Lease type")
    metadata = document.get("metadata")
    spec = document.get("spec")
    if not isinstance(metadata, dict) or not isinstance(spec, dict):
        raise ValueError("Lease metadata and spec are required")
    if metadata.get("name") != config.lease_name or metadata.get("namespace") != config.namespace:
        raise ValueError("Lease coordinates do not match")
    resource_version = metadata.get("resourceVersion")
    holder = spec.get("holderIdentity")
    duration = spec.get("leaseDurationSeconds")
    transitions = spec.get("leaseTransitions")
    if not isinstance(resource_version, str) or not resource_version:
        raise ValueError("Lease resourceVersion is required")
    if not isinstance(holder, str) or not holder:
        raise ValueError("Lease holderIdentity is required")
    if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
        raise ValueError("Lease duration is invalid")
    if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions < 0:
        raise ValueError("Lease transitions are invalid")
    acquire_time = _parse_rfc3339(spec.get("acquireTime"))
    renew_time = _parse_rfc3339(spec.get("renewTime"))
    if acquire_time > now or renew_time > now or renew_time < acquire_time:
        raise ValueError("Lease timestamps are invalid")
    return _ObservedLease(
        resource_version=resource_version,
        holder_identity=holder,
        lease_duration_seconds=duration,
        acquire_time=acquire_time,
        renew_time=renew_time,
        transitions=transitions,
    )


def _parse_rfc3339(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("Lease timestamp is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Lease timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("Lease timestamp timezone is required")
    return parsed.astimezone(UTC)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("UTC clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _rfc3339(value: datetime) -> str:
    return _utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")
