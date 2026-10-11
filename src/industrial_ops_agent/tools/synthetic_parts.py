"""Idempotent synthetic WMS adapter used by the M2 business loop."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from industrial_ops_agent.tools.authority import attest_tool_fact, authoritative_source
from industrial_ops_agent.tools.contracts import (
    PartIssueOutcomeUnknown,
    PartIssueResult,
    PartMovementResult,
    ReservationOutcomeUnknown,
    ReservationResult,
)


class SyntheticPartsAdapter:
    def __init__(self) -> None:
        self._reservations: dict[str, ReservationResult] = {}
        self._issues: dict[str, PartIssueResult] = {}
        self._movements: dict[str, PartMovementResult] = {}
        self._unknown_after_commit_once = False
        self._issue_unknown_after_commit_once = False
        self._available_quantity = 8

    @property
    def reservation_count(self) -> int:
        return len(self._reservations)

    @property
    def issue_count(self) -> int:
        return len(self._issues)

    @property
    def movement_count(self) -> int:
        return len(self._movements)

    def simulate_unknown_after_commit_once(self) -> None:
        self._unknown_after_commit_once = True

    def simulate_issue_unknown_after_commit_once(self) -> None:
        self._issue_unknown_after_commit_once = True

    def check_availability(self, *, part_number: str, quantity: int) -> dict[str, object]:
        return attest_tool_fact(
            "parts.availability",
            {
                "part_number": part_number,
                "requested_quantity": quantity,
                "available_quantity": self._available_quantity,
                "available": quantity <= self._available_quantity,
                "source": authoritative_source("parts.availability", "synthetic"),
                "source_record_id": f"stock-{part_number}",
                "as_of": datetime.now(UTC),
            },
            profile="synthetic",
        )

    def reserve(self, *, operation_id: str, part_number: str, quantity: int) -> ReservationResult:
        existing = self._reservations.get(operation_id)
        if existing is not None:
            return existing
        result = ReservationResult(
            reservation_id=f"reservation-{uuid4().hex}",
            operation_id=operation_id,
            part_number=part_number,
            quantity=quantity,
            source=authoritative_source("parts.availability", "synthetic"),
            source_record_id=f"wms-{uuid4().hex}",
            as_of=datetime.now(UTC),
        )
        self._reservations[operation_id] = result
        if self._unknown_after_commit_once:
            self._unknown_after_commit_once = False
            raise ReservationOutcomeUnknown(operation_id)
        return result

    def reconcile(self, operation_id: str) -> ReservationResult | None:
        return self._reservations.get(operation_id)

    def issue(
        self,
        *,
        operation_id: str,
        reservation_id: str,
        part_number: str,
        quantity: int,
    ) -> PartIssueResult:
        existing = self._issues.get(operation_id)
        if existing is not None:
            return existing
        result = PartIssueResult(
            issue_id=f"part-issue-{uuid4().hex}",
            operation_id=operation_id,
            reservation_id=reservation_id,
            part_number=part_number,
            quantity=quantity,
            status="ISSUED",
            source=authoritative_source("parts.availability", "synthetic"),
            source_record_id=f"wms-issue-{uuid4().hex}",
            as_of=datetime.now(UTC),
        )
        self._issues[operation_id] = result
        if self._issue_unknown_after_commit_once:
            self._issue_unknown_after_commit_once = False
            raise PartIssueOutcomeUnknown(operation_id)
        return result

    def reconcile_issue(self, operation_id: str) -> PartIssueResult | None:
        return self._issues.get(operation_id)

    def apply_movement(
        self,
        *,
        operation_id: str,
        movement_kind: str,
        part_issue_id: str,
        reservation_id: str,
        part_number: str,
        quantity: int,
    ) -> PartMovementResult:
        existing = self._movements.get(operation_id)
        if existing is not None:
            return existing
        result = PartMovementResult(
            movement_id=f"wms-movement-{uuid4().hex}",
            operation_id=operation_id,
            movement_kind=movement_kind,
            part_issue_id=part_issue_id,
            reservation_id=reservation_id,
            part_number=part_number,
            quantity=quantity,
            status="APPLIED",
            source=authoritative_source("parts.availability", "synthetic"),
            source_record_id=f"wms-movement-record-{uuid4().hex}",
            as_of=datetime.now(UTC),
        )
        self._movements[operation_id] = result
        return result

    def reconcile_movement(
        self,
        operation_id: str,
        movement_kind: str,
    ) -> PartMovementResult | None:
        result = self._movements.get(operation_id)
        return result if result is None or result.movement_kind == movement_kind else None
