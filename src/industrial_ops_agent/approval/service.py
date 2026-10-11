"""Deterministic approval invariant checks, independent of external side effects."""

from __future__ import annotations

from datetime import datetime

from industrial_ops_agent.approval.models import (
    ApprovalConflict,
    SeparationOfDutiesViolation,
)


class ApprovalBindingGuard:
    """Enforce SoD and immutable approval bindings before gateway execution."""

    @staticmethod
    def require_decidable(
        *,
        initiator_subject_id: str,
        decider_subject_id: str,
        current_status: str,
        current_version: int,
        expected_version: int,
        expires_at: datetime,
        now: datetime,
        decision: str,
    ) -> None:
        if initiator_subject_id == decider_subject_id:
            raise SeparationOfDutiesViolation
        if (
            current_version != expected_version
            or current_status != "PENDING"
            or expires_at <= now
            or decision not in {"APPROVED", "REJECTED"}
        ):
            raise ApprovalConflict

    @staticmethod
    def execution_rejection(
        *,
        status: str,
        current_version: int,
        expected_version: int | None,
        expires_at: datetime,
        now: datetime,
        approved_parameters_hash: str,
        supplied_parameters_hash: str,
    ) -> str | None:
        if status == "PENDING":
            return "PENDING_APPROVAL"
        if supplied_parameters_hash != approved_parameters_hash:
            return "APPROVAL_BINDING_MISMATCH"
        if expected_version is not None and current_version != expected_version:
            return "APPROVAL_VERSION_MISMATCH"
        if status != "APPROVED" or expires_at <= now:
            return "APPROVAL_INVALID"
        return None
