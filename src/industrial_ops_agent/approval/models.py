"""Approval domain errors and immutable command results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ApprovalNotVisible(Exception):
    pass


class ApprovalConflict(Exception):
    pass


class PurchaseNotRequired(Exception):
    pass


class PurchaseDependencyUnavailable(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class RefundNotEligible(Exception):
    pass


class RefundAlreadyRequested(Exception):
    pass


class SeparationOfDutiesViolation(Exception):
    pass


@dataclass(frozen=True, slots=True)
class PartsProposal:
    proposal_id: str
    approval_id: str
    parameters: dict[str, Any]
    operation_id: str


@dataclass(frozen=True, slots=True)
class PartIssueProposal:
    proposal_id: str
    approval_id: str
    parameters: dict[str, Any]
    operation_id: str
    part_issue_id: str


@dataclass(frozen=True, slots=True)
class CustomerNotificationProposal:
    proposal_id: str
    approval_id: str
    parameters: dict[str, Any]
    operation_id: str


@dataclass(frozen=True, slots=True)
class PurchaseRequestProposal:
    proposal_id: str
    approval_id: str
    parameters: dict[str, Any]
    operation_id: str


@dataclass(frozen=True, slots=True)
class RefundRequestProposal:
    proposal_id: str
    approval_id: str
    parameters: dict[str, Any]
    operation_id: str


@dataclass(frozen=True, slots=True)
class WorkOrderAssignmentProposal:
    proposal_id: str
    approval_id: str
    parameters: dict[str, Any]
    operation_id: str


@dataclass(frozen=True, slots=True)
class WorkOrderClosureProposal:
    proposal_id: str
    approval_id: str
    parameters: dict[str, Any]
    operation_id: str


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    status: str
    reservation_id: str | None = None
    work_order_id: str | None = None
    reason: str | None = None
    notification_id: str | None = None
    notification_delivery_id: str | None = None
    purchase_request_id: str | None = None
    refund_request_id: str | None = None
    assignment_delivery_id: str | None = None
    part_issue_id: str | None = None
    part_movement_id: str | None = None
    execution_attempt_id: str | None = None
    reconciliation_id: str | None = None
