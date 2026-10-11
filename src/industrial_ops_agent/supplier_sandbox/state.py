"""Thread-safe control and idempotency state for the local supplier A2A peer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from threading import RLock
from typing import Final, Literal

SupplierFaultMode = Literal[
    "UNAVAILABLE",
    "OUTCOME_UNKNOWN",
    "MALICIOUS_ARTIFACT",
    "MALFORMED_ARTIFACT",
    "FAILED",
    "REJECTED",
    "WORKING",
]

TRANSPORT_FAULTS: Final[frozenset[str]] = frozenset(
    {"UNAVAILABLE", "OUTCOME_UNKNOWN"}
)
AGENT_FAULTS: Final[frozenset[str]] = frozenset(
    {
        "MALICIOUS_ARTIFACT",
        "MALFORMED_ARTIFACT",
        "FAILED",
        "REJECTED",
        "WORKING",
    }
)
SUPPORTED_FAULTS: Final[frozenset[str]] = TRANSPORT_FAULTS | AGENT_FAULTS


class SupplierSandboxIdempotencyConflict(ValueError):
    """A collaboration ID was reused with a different request payload."""


@dataclass(frozen=True, slots=True)
class SupplierTaskBinding:
    collaboration_id: str
    request_digest: str
    remote_task_id: str


@dataclass(frozen=True, slots=True)
class _Fault:
    mode: str
    remaining: int


class SupplierSandboxState:
    """Process-local provider ledger with no platform business-side mutations."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._bindings: dict[str, SupplierTaskBinding] = {}
        self._canceled_task_ids: set[str] = set()
        self._fault: _Fault | None = None
        self._counters: Counter[str] = Counter()

    def existing_task(self, collaboration_id: str, request_digest: str) -> str | None:
        with self._lock:
            binding = self._bindings.get(collaboration_id)
            if binding is None:
                return None
            if binding.request_digest != request_digest:
                raise SupplierSandboxIdempotencyConflict(collaboration_id)
            self._counters["idempotent_replays"] += 1
            return binding.remote_task_id

    def bind_task(
        self,
        collaboration_id: str,
        request_digest: str,
        remote_task_id: str,
    ) -> None:
        binding = SupplierTaskBinding(
            collaboration_id=collaboration_id,
            request_digest=request_digest,
            remote_task_id=remote_task_id,
        )
        with self._lock:
            existing = self._bindings.get(collaboration_id)
            if existing is not None and existing != binding:
                raise SupplierSandboxIdempotencyConflict(collaboration_id)
            self._bindings[collaboration_id] = binding
            self._counters["unique_tasks"] = len(self._bindings)

    def configure_fault(self, mode: SupplierFaultMode, remaining: int) -> None:
        normalized = mode.strip().upper()
        if normalized not in SUPPORTED_FAULTS or not 1 <= remaining <= 10:
            raise ValueError("supplier sandbox fault configuration is invalid")
        with self._lock:
            self._fault = _Fault(normalized, remaining)

    def clear_fault(self) -> None:
        with self._lock:
            self._fault = None

    def consume_transport_fault(self) -> str | None:
        return self._consume_fault(TRANSPORT_FAULTS)

    def consume_agent_fault(self) -> str | None:
        return self._consume_fault(AGENT_FAULTS)

    def record_request(self) -> None:
        with self._lock:
            self._counters["agent_executions"] += 1

    def record_completed_advice(self) -> None:
        with self._lock:
            self._counters["completed_advice"] += 1

    def record_cancellation(self, task_id: str) -> None:
        with self._lock:
            self._canceled_task_ids.add(task_id)
            self._counters["cancellations"] += 1

    def is_canceled(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._canceled_task_ids

    def task_id_for(self, collaboration_id: str) -> str | None:
        with self._lock:
            binding = self._bindings.get(collaboration_id)
            return None if binding is None else binding.remote_task_id

    def reset(self) -> None:
        with self._lock:
            self._bindings.clear()
            self._canceled_task_ids.clear()
            self._fault = None
            self._counters.clear()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "classification": "SIMULATED_NON_PRODUCTION",
                "status": "READY",
                "record_counts": dict(sorted(self._counters.items())),
                "active_fault": (
                    None
                    if self._fault is None
                    else {"mode": self._fault.mode, "remaining": self._fault.remaining}
                ),
                "business_side_effects": {
                    "tool_executions": 0,
                    "work_order_mutations": 0,
                    "equipment_controls": 0,
                },
            }

    def _consume_fault(self, applicable: frozenset[str]) -> str | None:
        with self._lock:
            fault = self._fault
            if fault is None or fault.mode not in applicable:
                return None
            if fault.remaining == 1:
                self._fault = None
            else:
                self._fault = _Fault(fault.mode, fault.remaining - 1)
            self._counters[f"fault_{fault.mode.lower()}"] += 1
            return fault.mode
