"""Thread-safe, process-local state for the enterprise integration sandbox."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from threading import RLock
from typing import Final

SUPPORTED_FAULTS: Final[frozenset[str]] = frozenset({"UNAVAILABLE", "REJECT", "OUTCOME_UNKNOWN", "STALE"})
WRITE_FAULTS: Final[frozenset[str]] = frozenset({"UNAVAILABLE", "REJECT", "OUTCOME_UNKNOWN"})


class IdempotencyConflict(ValueError):
    """The same provider operation ID was reused for a different payload."""


@dataclass(frozen=True, slots=True)
class SandboxOperation:
    """A minimal provider-side result retained for reconciliation and callbacks."""

    provider: str
    operation_id: str
    payload_digest: str
    response: dict[str, object]
    sequence: int


@dataclass(frozen=True, slots=True)
class WriteDecision:
    """Atomic outcome of idempotency lookup, fault consumption, and optional commit."""

    operation: SandboxOperation | None
    replayed: bool
    fault: str | None


@dataclass(frozen=True, slots=True)
class _Fault:
    mode: str
    remaining: int


class SandboxState:
    """In-memory external-system ledger; it never writes platform business state."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._operations: dict[tuple[str, str], SandboxOperation] = {}
        self._faults: dict[str, _Fault] = {}
        self._next_sequence = 1
        self._receipt_events: dict[tuple[str, str, str], str] = {}
        self._next_receipt_sequence = 1

    def decide_write(
        self,
        *,
        provider: str,
        operation_id: str,
        payload: Mapping[str, object],
        response: Mapping[str, object],
    ) -> WriteDecision:
        """Replay, reject, or atomically retain a write and its consumed fault."""

        digest = _payload_digest(payload)
        key = (provider, operation_id)
        with self._lock:
            existing = self._operations.get(key)
            if existing is not None:
                if existing.payload_digest != digest:
                    raise IdempotencyConflict(operation_id)
                return WriteDecision(_copy_operation(existing), True, None)

            fault_provider = "wms" if provider.startswith("wms-") else provider
            fault = self._consume_fault_locked(fault_provider, WRITE_FAULTS)
            if fault in {"UNAVAILABLE", "REJECT"}:
                return WriteDecision(None, False, fault)

            operation = SandboxOperation(
                provider=provider,
                operation_id=operation_id,
                payload_digest=digest,
                response=deepcopy(dict(response)),
                sequence=self._next_sequence,
            )
            self._next_sequence += 1
            self._operations[key] = operation
            return WriteDecision(_copy_operation(operation), False, fault)

    def get(self, provider: str, operation_id: str) -> SandboxOperation | None:
        with self._lock:
            operation = self._operations.get((provider, operation_id))
            return None if operation is None else _copy_operation(operation)

    def consume_fault(self, provider: str, applicable: frozenset[str]) -> str | None:
        """Consume one fault only when its mode applies to the requested operation."""
        with self._lock:
            return self._consume_fault_locked(provider, applicable)

    def configure_fault(self, provider: str, mode: str, remaining: int) -> None:
        normalized = mode.strip().upper()
        if normalized not in SUPPORTED_FAULTS:
            raise ValueError("unsupported sandbox fault mode")
        if not provider or not 1 <= remaining <= 10:
            raise ValueError("sandbox fault count must be between 1 and 10")
        with self._lock:
            self._faults[provider] = _Fault(normalized, remaining)

    def clear_fault(self, provider: str) -> None:
        with self._lock:
            self._faults.pop(provider, None)

    def snapshot(self) -> dict[str, object]:
        """Return only non-sensitive counts and fault budgets for the control plane."""

        with self._lock:
            counts = Counter(item.provider for item in self._operations.values())
            return {
                "classification": "SIMULATED_NON_PRODUCTION",
                "record_counts": dict(sorted(counts.items())),
                "faults": {
                    provider: {"mode": fault.mode, "remaining": fault.remaining}
                    for provider, fault in sorted(self._faults.items())
                },
                "receipt_count": len(self._receipt_events),
            }

    def receipt_event_id(self, provider: str, operation_id: str, outcome: str) -> str:
        """Return one stable synthetic event ID for a retried completion."""

        key = (provider, operation_id, outcome)
        with self._lock:
            existing = self._receipt_events.get(key)
            if existing is not None:
                return existing
            digest = sha256("\0".join(key).encode("utf-8")).hexdigest()[:12]
            event_id = f"sandbox-{provider}-event-{self._next_receipt_sequence}-{digest}"
            self._next_receipt_sequence += 1
            self._receipt_events[key] = event_id
            return event_id

    def reset(self) -> None:
        with self._lock:
            self._operations.clear()
            self._faults.clear()
            self._receipt_events.clear()
            self._next_sequence = 1
            self._next_receipt_sequence = 1

    def _consume_fault_locked(self, provider: str,
                              applicable: frozenset[str] = SUPPORTED_FAULTS) -> str | None:
        fault = self._faults.get(provider)
        if fault is None or fault.mode not in applicable:
            return None
        if fault.remaining == 1:
            del self._faults[provider]
        else:
            self._faults[provider] = _Fault(fault.mode, fault.remaining - 1)
        return fault.mode


def _payload_digest(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(dict(payload), ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(canonical).hexdigest()


def _copy_operation(operation: SandboxOperation) -> SandboxOperation:
    return replace(operation, response=deepcopy(operation.response))
