"""Tool governance, T2 proposal/approval, and idempotent execution boundary."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from threading import Lock
from time import monotonic
from typing import Any, cast
from uuid import uuid4

from jsonschema.exceptions import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from industrial_ops_agent.application.assets import AssetQueryService
from industrial_ops_agent.approval.models import (
    ApprovalConflict,
    ApprovalNotVisible,
    CustomerNotificationProposal,
    ExecutionResult,
    PartIssueProposal,
    PartsProposal,
    PurchaseDependencyUnavailable,
    PurchaseNotRequired,
    PurchaseRequestProposal,
    RefundAlreadyRequested,
    RefundNotEligible,
    RefundRequestProposal,
    WorkOrderAssignmentProposal,
    WorkOrderClosureProposal,
)
from industrial_ops_agent.approval.service import ApprovalBindingGuard
from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.maintenance_planning.review_isolation import (
    ReviewIsolationConflict,
    incident_read_transaction,
    lock_subject,
    record_draft_disclosure,
    record_incident_disclosure,
    work_order_read_transaction,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ActionProposalRecord,
    ApprovalDecisionRecord,
    ApprovalRequestRecord,
    AssetComponentRecord,
    AssetSiteLinkRecord,
    CustomerNotificationDeliveryRecord,
    CustomerNotificationReceiptRecord,
    CustomerServiceUpdateRecord,
    DiagnosisRunRecord,
    EquipmentControlHandoffRecord,
    EvidenceBundleRecord,
    ExecutionAttemptRecord,
    ExecutionReconciliationEventRecord,
    ExecutionReconciliationRecord,
    FsmAssignmentDeliveryRecord,
    FsmAssignmentReceiptRecord,
    IdempotencyRecord,
    IncidentControlRecord,
    IncidentRecord,
    PartIssueRecord,
    PartMaterialMovementRecord,
    PartReservationRecord,
    PurchaseFulfillmentReceiptRecord,
    PurchaseFulfillmentRecord,
    PurchaseRequestReceiptRecord,
    PurchaseRequestRecord,
    RefundRequestReceiptRecord,
    RefundRequestRecord,
    ServiceQuotationDecisionRecord,
    ServiceQuotationRecord,
    ToolCallRecord,
    WorkOrderAssignmentRecord,
    WorkOrderEventRecord,
    WorkOrderFieldEntryRecord,
    WorkOrderRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.tools.authority import validate_attested_tool_fact
from industrial_ops_agent.tools.contracts import (
    EnterpriseToolError,
    FsmAssignmentAdapter,
    FsmAssignmentCandidate,
    FsmAssignmentDeliveryResult,
    FsmAssignmentReceipt,
    NotificationContact,
    NotificationDeliveryAdapter,
    NotificationDeliveryResult,
    NotificationReceipt,
    PartIssueOutcomeUnknown,
    PartIssueResult,
    PartMovementOutcomeUnknown,
    PartMovementResult,
    PartsAdapter,
    ProcurementDeliveryAdapter,
    ProcurementDeliveryResult,
    ProcurementFulfillmentReceipt,
    ProcurementProfile,
    ProcurementReceipt,
    ReadonlyToolAdapter,
    RefundDeliveryAdapter,
    RefundDeliveryResult,
    RefundProfile,
    RefundReceipt,
    ReservationOutcomeUnknown,
    ReservationResult,
    ServiceQuotationCandidate,
    ServiceQuotationCatalogAdapter,
    ToolRateLimiter,
    ToolTransportBinding,
    ToolTransportBindingProvider,
)
from industrial_ops_agent.tools.quotations import (
    service_quotation_candidate_binding,
    validate_service_quotation_candidate,
)
from industrial_ops_agent.tools.registry import ToolDefinition, ToolRegistry
from industrial_ops_agent.tools.synthetic_readonly import SyntheticReadonlyAdapter
from industrial_ops_agent.workorders.assignment import is_eligible_field_engineer
from industrial_ops_agent.workorders.service import WorkOrderConflict, WorkOrderService

T2_APPROVAL_TTL = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class ToolInvocationResult:
    status: str
    tool_call_id: str
    data: dict[str, Any] | None = None
    handoff: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PartMaterialMovementProposal:
    proposal_id: str
    approval_id: str
    parameters: dict[str, Any]
    operation_id: str
    movement_id: str


class ToolRateLimitExceeded(RuntimeError):
    pass


class PartsReservationProposalNotEligible(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class PartsReservationProposalDependencyUnavailable(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class PartsReservationProposalIdempotencyConflict(Exception):
    pass


class PartIssueProposalConflict(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class PartIssueProposalIdempotencyConflict(Exception):
    pass


class PartMovementProposalConflict(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class PartMovementProposalIdempotencyConflict(Exception):
    pass


class EquipmentControlHandoffNotVisible(Exception):
    pass


class EquipmentControlHandoffConflict(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ExecutionReconciliationNotVisible(Exception):
    pass


class ExecutionReconciliationConflict(Exception):
    pass


class CustomerNotificationAuthorityUnavailable(Exception):
    def __init__(self, reason: str = "contact_authority_unavailable") -> None:
        self.reason = reason
        super().__init__(reason)


class CustomerNotificationContactConflict(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class CustomerNotificationReceiptRejected(Exception):
    def __init__(self, reason: str = "notification_receipt_rejected") -> None:
        self.reason = reason
        super().__init__(reason)


class PurchaseDeliveryProfileUnavailable(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class PurchaseDeliveryProfileConflict(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class PurchaseRequestReceiptRejected(Exception):
    def __init__(self, reason: str = "procurement_receipt_rejected") -> None:
        self.reason = reason
        super().__init__(reason)


class ProcurementFulfillmentReceiptRejected(Exception):
    def __init__(self, reason: str = "procurement_fulfillment_receipt_rejected") -> None:
        self.reason = reason
        super().__init__(reason)


class RefundDeliveryProfileUnavailable(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class RefundDeliveryProfileConflict(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class RefundRequestReceiptRejected(Exception):
    def __init__(self, reason: str = "refund_receipt_rejected") -> None:
        self.reason = reason
        super().__init__(reason)


class FsmAssignmentProfileUnavailable(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class FsmAssignmentProfileConflict(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class FsmAssignmentReceiptRejected(Exception):
    def __init__(self, reason: str = "fsm_assignment_receipt_rejected") -> None:
        self.reason = reason
        super().__init__(reason)


class ServiceQuotationUnavailable(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ServiceQuotationConflict(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class RepairWorkOrderUnavailable(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class RepairWorkOrderConflict(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class RepairWorkOrderProposalIdempotencyConflict(Exception):
    pass


@dataclass(frozen=True, slots=True)
class CustomerNotificationOption:
    channel: str
    contact_point_id: str | None
    masked_destination: str
    source_system: str
    source_record_id: str | None
    source_version: str | None
    as_of: str | None
    consent_basis_digest: str | None
    expires_at: str | None


@dataclass(frozen=True, slots=True)
class CustomerNotificationOptions:
    incident_id: str
    options: tuple[CustomerNotificationOption, ...]
    external_unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class CustomerNotificationReceiptResult:
    delivery_id: str
    status: str
    state_changed: bool
    reconciliation_id: str | None


@dataclass(frozen=True, slots=True)
class PurchaseRequestOption:
    delivery_target: str
    provider: str | None
    profile_id: str | None
    profile_version: str | None
    display_name: str
    as_of: str | None
    expires_at: str | None


@dataclass(frozen=True, slots=True)
class PurchaseRequestOptions:
    incident_id: str
    options: tuple[PurchaseRequestOption, ...]
    erp_unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class PurchaseRequestReceiptResult:
    purchase_request_id: str
    status: str
    state_changed: bool
    reconciliation_id: str | None


@dataclass(frozen=True, slots=True)
class ProcurementFulfillmentReceiptResult:
    purchase_request_id: str
    fulfillment_id: str
    status: str
    received_quantity: int
    state_changed: bool


@dataclass(frozen=True, slots=True)
class ProcurementReservationReadiness:
    status: str
    reason: str
    next_action: str
    source: str | None = None
    source_record_id: str | None = None
    as_of: datetime | None = None
    available_quantity: int | None = None


@dataclass(frozen=True, slots=True)
class PurchaseRequestProjection:
    request: PurchaseRequestRecord
    reconciliation_id: str | None
    fulfillment: PurchaseFulfillmentRecord | None
    reservation_readiness: ProcurementReservationReadiness | None


@dataclass(frozen=True, slots=True)
class RefundRequestOption:
    delivery_target: str
    provider: str | None
    profile_id: str | None
    profile_version: str | None
    display_name: str
    as_of: str | None
    expires_at: str | None


@dataclass(frozen=True, slots=True)
class RefundRequestOptions:
    work_order_id: str
    options: tuple[RefundRequestOption, ...]
    finance_unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class RefundRequestReceiptResult:
    refund_request_id: str
    status: str
    payment_status: str
    state_changed: bool
    reconciliation_id: str | None


@dataclass(frozen=True, slots=True)
class ServiceQuotationOption:
    provider: str
    candidate_id: str
    source_version: str
    display_name: str
    currency: str
    line_items: tuple[dict[str, Any], ...]
    subtotal: str
    discount: str
    tax: str
    total: str
    as_of: str
    expires_at: str


@dataclass(frozen=True, slots=True)
class ServiceQuotationOptions:
    incident_id: str
    diagnosis_run_id: str
    diagnosis_version: int
    entitlement_decision: str
    options: tuple[ServiceQuotationOption, ...]
    catalog_unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class ServiceQuotationProposal:
    proposal_id: str
    approval_id: str
    parameters: dict[str, Any]
    operation_id: str


@dataclass(frozen=True, slots=True)
class ServiceQuotationExecutionResult(ExecutionResult):
    service_quotation_id: str | None = None


@dataclass(frozen=True, slots=True)
class RepairWorkOrderOption:
    authorization_type: str
    fulfillment_mode: str
    quotation_id: str | None
    quotation_version: int | None
    quotation_state_version: int | None
    currency: str | None
    total: str | None


@dataclass(frozen=True, slots=True)
class RepairWorkOrderOptions:
    incident_id: str
    incident_version: int
    asset_id: str
    diagnosis_run_id: str | None
    diagnosis_version: int | None
    entitlement_decision: str | None
    options: tuple[RepairWorkOrderOption, ...]
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class RepairWorkOrderProposal:
    proposal_id: str
    approval_id: str
    parameters: dict[str, Any]
    operation_id: str


@dataclass(frozen=True, slots=True)
class WorkOrderAssignmentOption:
    assignment_target: str
    provider: str | None
    candidate_profile_id: str | None
    profile_version: str | None
    assignee_subject_id: str | None
    display_name: str
    matched_skill_codes: tuple[str, ...]
    service_window_start: str | None
    service_window_end: str | None
    travel_minutes: int | None
    remaining_work_minutes: int | None
    eligibility_code: str | None
    source_record_id: str | None
    as_of: str | None
    expires_at: str | None


@dataclass(frozen=True, slots=True)
class WorkOrderAssignmentOptions:
    work_order_id: str
    options: tuple[WorkOrderAssignmentOption, ...]
    fsm_unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class FsmAssignmentReceiptResult:
    delivery_id: str
    status: str
    state_changed: bool
    reconciliation_id: str | None


_EQUIPMENT_CONTROL_TOOL_IDS = {
    "STOP": "equipment.control.stop",
    "REMOTE_START": "equipment.control.start",
    "PLC_PARAMETER_CHANGE": "equipment.control.plc_parameter_change",
    "INTERLOCK_OVERRIDE": "equipment.control.interlock_override",
}
_EQUIPMENT_CONTROL_TARGET = "certified-equipment-control-system"


class ProcessToolRateLimiter:
    """Bounded Compose-Lite limiter; distributed deployments replace it with Redis."""

    def __init__(self) -> None:
        self._calls: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def require(self, tenant_id: str, tool_id: str, limit: int) -> None:
        now = monotonic()
        key = (tenant_id, tool_id)
        with self._lock:
            calls = self._calls[key]
            while calls and calls[0] <= now - 60:
                calls.popleft()
            if len(calls) >= limit:
                raise ToolRateLimitExceeded("tool_rate_limit_exceeded")
            calls.append(now)


class ToolGateway:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        registry: ToolRegistry,
        parts: PartsAdapter,
        readonly: ReadonlyToolAdapter | None = None,
        rate_limiter: ToolRateLimiter | None = None,
        notifications: NotificationDeliveryAdapter | None = None,
        procurement: ProcurementDeliveryAdapter | None = None,
        refunds: RefundDeliveryAdapter | None = None,
        fsm_assignments: FsmAssignmentAdapter | None = None,
        quotations: ServiceQuotationCatalogAdapter | None = None,
    ) -> None:
        self._database, self._authorizer, self._registry, self._parts = (
            database,
            authorizer,
            registry,
            parts,
        )
        self._readonly = readonly or SyntheticReadonlyAdapter()
        self._rate_limiter = rate_limiter or ProcessToolRateLimiter()
        self._notifications = notifications
        self._procurement = procurement
        self._refunds = refunds
        self._fsm_assignments = fsm_assignments
        self._quotations = quotations

    def list_repair_work_order_options(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        request_id: str,
    ) -> RepairWorkOrderOptions:
        """Project server-owned repair eligibility without accepting business facts."""

        context = self._repair_work_order_context(
            identity,
            incident_id=incident_id,
            request_id=request_id,
            action=Action.READ_REPAIR_WORK_ORDER_AUTHORIZATION,
        )
        option = context.get("option")
        return RepairWorkOrderOptions(
            incident_id=incident_id,
            incident_version=int(context["incident_version"]),
            asset_id=str(context["asset_id"]),
            diagnosis_run_id=cast(str | None, context.get("diagnosis_run_id")),
            diagnosis_version=cast(int | None, context.get("diagnosis_version")),
            entitlement_decision=cast(str | None, context.get("entitlement_decision")),
            options=(cast(RepairWorkOrderOption, option),) if option is not None else (),
            unavailable_reason=cast(str | None, context.get("unavailable_reason")),
        )

    def propose_repair_work_order(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        incident_version: int,
        authorization_type: str,
        idempotency_key: str,
        request_id: str,
    ) -> RepairWorkOrderProposal:
        """Freeze a no-parts repair command behind the shared T2 boundary."""

        request_hash = _digest(
            {
                "incident_id": incident_id,
                "incident_version": incident_version,
                "authorization_type": authorization_type,
            }
        )
        storage_key = (
            "repair-work-order-proposal:"
            + sha256(idempotency_key.encode()).hexdigest()
        )
        replay = self._repair_work_order_proposal_replay(
            identity,
            storage_key=storage_key,
            request_hash=request_hash,
            request_id=request_id,
        )
        if replay is not None:
            return replay

        context = self._repair_work_order_context(
            identity,
            incident_id=incident_id,
            request_id=request_id,
            action=Action.PROPOSE_REPAIR_WORK_ORDER,
        )
        unavailable_reason = context.get("unavailable_reason")
        if unavailable_reason is not None:
            if unavailable_reason == "service_entitlement_facts_unavailable":
                raise RepairWorkOrderUnavailable(str(unavailable_reason))
            raise RepairWorkOrderConflict(str(unavailable_reason))
        if int(context["incident_version"]) != incident_version:
            raise RepairWorkOrderConflict("incident_version_changed")
        option = cast(RepairWorkOrderOption | None, context.get("option"))
        if option is None or option.authorization_type != authorization_type:
            raise RepairWorkOrderConflict("repair_authorization_option_not_available")

        now = datetime.now(UTC)
        proposal_id = f"proposal-{uuid4().hex}"
        approval_id = f"approval-{uuid4().hex}"
        operation_id = f"operation-{uuid4().hex}"
        parameters: dict[str, Any] = {
            "incident_id": incident_id,
            "asset_id": context["asset_id"],
            "incident_version": incident_version,
            "operation_id": operation_id,
            "authorization_type": authorization_type,
            "fulfillment_mode": "NO_INITIAL_PARTS",
            "diagnosis": {
                "diagnosis_run_id": context["diagnosis_run_id"],
                "version": context["diagnosis_version"],
            },
            "entitlement": context["entitlement"],
            "quotation": context.get("quotation"),
        }
        definition = self._registry.get("work_order.create", "1.0.0")
        definition.validate(parameters)
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            existing = session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.tenant_id == identity.tenant_id,
                    IdempotencyRecord.subject_id == identity.subject_id,
                    IdempotencyRecord.key == storage_key,
                )
            )
            if existing is not None:
                return self._repair_work_order_proposal_from_record(
                    session,
                    identity,
                    existing,
                    request_hash=request_hash,
                    request_id=request_id,
                )
            current_incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            current_diagnosis = session.scalar(
                select(DiagnosisRunRecord).where(
                    DiagnosisRunRecord.tenant_id == identity.tenant_id,
                    DiagnosisRunRecord.diagnosis_run_id == context["diagnosis_run_id"],
                )
            )
            existing_work_order = session.scalar(
                select(WorkOrderRecord.work_order_id).where(
                    WorkOrderRecord.tenant_id == identity.tenant_id,
                    WorkOrderRecord.incident_id == incident_id,
                )
            )
            if (
                current_incident is None
                or current_incident.status != "DIAGNOSED"
                or current_incident.version != incident_version
                or current_diagnosis is None
                or current_diagnosis.status != "COMPLETED"
                or current_diagnosis.version != context["diagnosis_version"]
                or existing_work_order is not None
                or not self._repair_quotation_binding_matches(
                    session,
                    identity.tenant_id,
                    current_incident,
                    cast(dict[str, Any] | None, context.get("quotation")),
                )
            ):
                raise RepairWorkOrderConflict("repair_authorization_context_changed")
            session.add(
                ActionProposalRecord(
                    proposal_id=proposal_id,
                    tenant_id=identity.tenant_id,
                    incident_id=incident_id,
                    initiator_subject_id=identity.subject_id,
                    tool_id=definition.tool_id,
                    tool_version=definition.version,
                    risk_tier=definition.risk_tier,
                    parameters=parameters,
                    parameters_hash=_digest(parameters),
                    resource_version=incident_version,
                    operation_id=operation_id,
                    status="PENDING_APPROVAL",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ApprovalRequestRecord(
                    approval_id=approval_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    policy_version="m2-t2-v1",
                    status="PENDING",
                    expires_at=now + T2_APPROVAL_TTL,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                IdempotencyRecord(
                    record_id=f"idem-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=proposal_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
        return RepairWorkOrderProposal(
            proposal_id,
            approval_id,
            parameters,
            operation_id,
        )

    def _repair_work_order_proposal_replay(
        self,
        identity: IdentityContext,
        *,
        storage_key: str,
        request_hash: str,
        request_id: str,
    ) -> RepairWorkOrderProposal | None:
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.tenant_id == identity.tenant_id,
                    IdempotencyRecord.subject_id == identity.subject_id,
                    IdempotencyRecord.key == storage_key,
                )
            )
            if record is None:
                return None
            return self._repair_work_order_proposal_from_record(
                session,
                identity,
                record,
                request_hash=request_hash,
                request_id=request_id,
            )

    @staticmethod
    def _repair_work_order_proposal_from_record(
        session: Session,
        identity: IdentityContext,
        record: IdempotencyRecord,
        *,
        request_hash: str,
        request_id: str,
    ) -> RepairWorkOrderProposal:
        if record.request_hash != request_hash:
            raise RepairWorkOrderProposalIdempotencyConflict
        proposal = session.scalar(
            select(ActionProposalRecord).where(
                ActionProposalRecord.tenant_id == identity.tenant_id,
                ActionProposalRecord.proposal_id == record.result_ref,
                ActionProposalRecord.tool_id == "work_order.create",
            )
        )
        approval = session.scalar(
            select(ApprovalRequestRecord).where(
                ApprovalRequestRecord.tenant_id == identity.tenant_id,
                ApprovalRequestRecord.proposal_id == record.result_ref,
            )
        )
        if proposal is None or approval is None:
            raise RuntimeError("repair proposal idempotency record references missing facts")
        record_incident_disclosure(
            session, identity, (proposal.incident_id,),
            resource_kind="approval", resource_id=approval.approval_id, request_id=request_id,
        )
        return RepairWorkOrderProposal(
            proposal.proposal_id,
            approval.approval_id,
            dict(proposal.parameters),
            proposal.operation_id,
        )

    def _repair_work_order_context(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        request_id: str,
        action: Action,
    ) -> dict[str, Any]:
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident is None:
                raise ApprovalNotVisible
            self._authorizer.require(
                identity,
                action,
                ResourceContext(identity.tenant_id, incident_id, incident.asset_id),
                request_id=request_id,
            )
            base: dict[str, Any] = {
                "incident_id": incident_id,
                "incident_version": incident.version,
                "asset_id": incident.asset_id,
                "diagnosis_run_id": None,
                "diagnosis_version": None,
                "entitlement_decision": None,
                "entitlement": None,
                "quotation": None,
                "option": None,
                "unavailable_reason": None,
            }
            if incident.status != "DIAGNOSED":
                base["unavailable_reason"] = "incident_not_diagnosed"
                return base
            if session.scalar(
                select(WorkOrderRecord.work_order_id).where(
                    WorkOrderRecord.tenant_id == identity.tenant_id,
                    WorkOrderRecord.incident_id == incident_id,
                )
            ) is not None:
                base["unavailable_reason"] = "work_order_already_exists"
                return base
            diagnosis = session.scalar(
                select(DiagnosisRunRecord)
                .where(
                    DiagnosisRunRecord.tenant_id == identity.tenant_id,
                    DiagnosisRunRecord.incident_id == incident_id,
                )
                .order_by(
                    DiagnosisRunRecord.created_at.desc(),
                    DiagnosisRunRecord.diagnosis_run_id.desc(),
                )
            )
            if diagnosis is None or diagnosis.status != "COMPLETED":
                base["unavailable_reason"] = "terminal_diagnosis_required"
                return base
            base["diagnosis_run_id"] = diagnosis.diagnosis_run_id
            base["diagnosis_version"] = diagnosis.version
            asset_id = incident.asset_id

        from industrial_ops_agent.service_entitlements.service import (
            ServiceEntitlementService,
        )

        entitlement = ServiceEntitlementService(
            AssetQueryService(self._database, self._authorizer),
            self,
        ).lookup(identity, asset_id, request_id=request_id)
        base["entitlement_decision"] = entitlement.decision
        if entitlement.decision == "FACTS_UNAVAILABLE":
            base["unavailable_reason"] = "service_entitlement_facts_unavailable"
            return base
        if entitlement.decision not in {"COVERED", "NOT_COVERED"}:
            base["unavailable_reason"] = "service_entitlement_review_required"
            return base
        base["entitlement"] = _service_entitlement_binding(entitlement)
        if entitlement.decision == "COVERED":
            base["option"] = RepairWorkOrderOption(
                "COVERED_SERVICE",
                "NO_INITIAL_PARTS",
                None,
                None,
                None,
                None,
                None,
            )
            return base

        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            current_incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            quotation = session.scalar(
                select(ServiceQuotationRecord)
                .where(
                    ServiceQuotationRecord.tenant_id == identity.tenant_id,
                    ServiceQuotationRecord.incident_id == incident_id,
                )
                .order_by(ServiceQuotationRecord.quotation_version.desc())
            )
            if current_incident is None or quotation is None:
                base["unavailable_reason"] = "accepted_quotation_required"
                return base
            decision = session.scalar(
                select(ServiceQuotationDecisionRecord).where(
                    ServiceQuotationDecisionRecord.tenant_id == identity.tenant_id,
                    ServiceQuotationDecisionRecord.quotation_id == quotation.quotation_id,
                )
            )
            if (
                quotation.status != "ACCEPTED"
                or quotation.asset_id != asset_id
                or quotation.diagnosis_run_id != base["diagnosis_run_id"]
                or quotation.diagnosis_version != base["diagnosis_version"]
                or decision is None
                or decision.decision != "ACCEPTED"
                or decision.quotation_version != quotation.quotation_version
                or decision.reporter_subject_id != current_incident.reporter_subject_id
            ):
                base["unavailable_reason"] = "accepted_quotation_not_current"
                return base
            binding = _repair_quotation_binding(quotation, decision)
            base["quotation"] = binding
            base["option"] = RepairWorkOrderOption(
                "ACCEPTED_QUOTATION",
                "NO_INITIAL_PARTS",
                quotation.quotation_id,
                quotation.quotation_version,
                quotation.state_version,
                quotation.currency,
                quotation.total,
            )
            return base

    @staticmethod
    def _repair_quotation_binding_matches(
        session: Session,
        tenant_id: str,
        incident: IncidentRecord,
        approved: dict[str, Any] | None,
    ) -> bool:
        if approved is None:
            return True
        quotation = session.scalar(
            select(ServiceQuotationRecord).where(
                ServiceQuotationRecord.tenant_id == tenant_id,
                ServiceQuotationRecord.quotation_id == approved.get("quotation_id"),
                ServiceQuotationRecord.incident_id == incident.incident_id,
            )
        )
        if quotation is None:
            return False
        decision = session.scalar(
            select(ServiceQuotationDecisionRecord).where(
                ServiceQuotationDecisionRecord.tenant_id == tenant_id,
                ServiceQuotationDecisionRecord.quotation_id == quotation.quotation_id,
            )
        )
        return (
            decision is not None
            and decision.reporter_subject_id == incident.reporter_subject_id
            and _repair_quotation_binding(quotation, decision) == approved
        )

    def list_service_quotation_options(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        request_id: str,
    ) -> ServiceQuotationOptions:
        """Return CPQ candidates only after diagnosis and entitlement checks."""

        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident is None:
                raise ApprovalNotVisible
            self._authorizer.require(
                identity,
                Action.READ_SERVICE_QUOTATION,
                ResourceContext(identity.tenant_id, incident_id, incident.asset_id),
                request_id=request_id,
            )
            diagnosis = session.scalar(
                select(DiagnosisRunRecord)
                .where(
                    DiagnosisRunRecord.tenant_id == identity.tenant_id,
                    DiagnosisRunRecord.incident_id == incident_id,
                    DiagnosisRunRecord.status.in_(("COMPLETED", "NEEDS_INFORMATION")),
                )
                .order_by(
                    DiagnosisRunRecord.created_at.desc(),
                    DiagnosisRunRecord.diagnosis_run_id.desc(),
                )
            )
            if diagnosis is None:
                raise ApprovalConflict
            asset_id = incident.asset_id
            diagnosis_id = diagnosis.diagnosis_run_id
            diagnosis_version = diagnosis.version

        from industrial_ops_agent.service_entitlements.service import (
            ServiceEntitlementService,
        )

        entitlement = ServiceEntitlementService(
            AssetQueryService(self._database, self._authorizer),
            self,
        ).lookup(identity, asset_id, request_id=request_id)
        if entitlement.decision not in {"COVERED", "NOT_COVERED"}:
            return ServiceQuotationOptions(
                incident_id,
                diagnosis_id,
                diagnosis_version,
                entitlement.decision,
                (),
                "service_entitlement_not_reliable",
            )
        if self._quotations is None:
            return ServiceQuotationOptions(
                incident_id,
                diagnosis_id,
                diagnosis_version,
                entitlement.decision,
                (),
                "enterprise_service_quotation_catalog_disabled",
            )
        now = datetime.now(UTC)
        try:
            raw_candidates = self._quotations.list_candidates(
                tenant_id=identity.tenant_id,
                incident_ref=incident_id,
                asset_ref=asset_id,
                diagnosis_ref=diagnosis_id,
            )
            candidates = tuple(
                validate_service_quotation_candidate(
                    candidate,
                    provider_id=self._quotations.provider_id,
                    incident_ref=incident_id,
                    asset_ref=asset_id,
                    diagnosis_ref=diagnosis_id,
                    now=now,
                )
                for candidate in raw_candidates
            )
            if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
                raise EnterpriseToolError("service_quotation_candidates_invalid")
        except EnterpriseToolError as exc:
            return ServiceQuotationOptions(
                incident_id,
                diagnosis_id,
                diagnosis_version,
                entitlement.decision,
                (),
                exc.reason,
            )
        except (OSError, TimeoutError):
            return ServiceQuotationOptions(
                incident_id,
                diagnosis_id,
                diagnosis_version,
                entitlement.decision,
                (),
                "service_quotation_catalog_unavailable",
            )
        return ServiceQuotationOptions(
            incident_id,
            diagnosis_id,
            diagnosis_version,
            entitlement.decision,
            tuple(self._service_quotation_option(candidate) for candidate in candidates),
            None,
        )

    @staticmethod
    def _service_quotation_option(
        candidate: ServiceQuotationCandidate,
    ) -> ServiceQuotationOption:
        return ServiceQuotationOption(
            provider=candidate.provider,
            candidate_id=candidate.candidate_id,
            source_version=candidate.source_version,
            display_name=candidate.display_name,
            currency=candidate.currency,
            line_items=tuple(
                {
                    "code": item.code,
                    "description": item.description,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "line_total": item.line_total,
                }
                for item in candidate.line_items
            ),
            subtotal=candidate.subtotal,
            discount=candidate.discount,
            tax=candidate.tax,
            total=candidate.total,
            as_of=candidate.as_of.isoformat(),
            expires_at=candidate.expires_at.isoformat(),
        )

    def propose_service_quotation(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        candidate_id: str,
        request_id: str,
    ) -> ServiceQuotationProposal:
        """Freeze one current CPQ candidate behind the shared T2 approval boundary."""

        if self._quotations is None:
            raise ServiceQuotationUnavailable(
                "enterprise_service_quotation_catalog_disabled"
            )
        context = self._service_quotation_context(
            identity,
            incident_id=incident_id,
            request_id=request_id,
            action=Action.PROPOSE_SERVICE_QUOTATION,
        )
        now = datetime.now(UTC)
        candidate = self._resolve_service_quotation_candidate(
            identity,
            candidate_id=candidate_id,
            incident_id=incident_id,
            asset_id=str(context["asset_id"]),
            diagnosis_id=str(context["diagnosis_run_id"]),
            now=now,
        )
        parameters: dict[str, Any] = {
            "incident_id": incident_id,
            "asset_id": context["asset_id"],
            "incident_version": context["incident_version"],
            "diagnosis": {
                "diagnosis_run_id": context["diagnosis_run_id"],
                "version": context["diagnosis_version"],
            },
            "candidate": service_quotation_candidate_binding(candidate),
            "entitlement": context["entitlement"],
        }
        definition = self._registry.get("service.quote.publish", "1.0.0")
        definition.validate(parameters)
        proposal_id = f"proposal-{uuid4().hex}"
        approval_id = f"approval-{uuid4().hex}"
        operation_id = f"operation-{uuid4().hex}"
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            current_incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            current_diagnosis = session.scalar(
                select(DiagnosisRunRecord).where(
                    DiagnosisRunRecord.tenant_id == identity.tenant_id,
                    DiagnosisRunRecord.diagnosis_run_id
                    == context["diagnosis_run_id"],
                )
            )
            if (
                current_incident is None
                or current_incident.version != context["incident_version"]
                or current_diagnosis is None
                or current_diagnosis.version != context["diagnosis_version"]
                or current_diagnosis.status not in {"COMPLETED", "NEEDS_INFORMATION"}
            ):
                raise ServiceQuotationConflict("service_quotation_context_changed")
            session.add(
                ActionProposalRecord(
                    proposal_id=proposal_id,
                    tenant_id=identity.tenant_id,
                    incident_id=incident_id,
                    initiator_subject_id=identity.subject_id,
                    tool_id=definition.tool_id,
                    tool_version=definition.version,
                    risk_tier=definition.risk_tier,
                    parameters=parameters,
                    parameters_hash=_digest(parameters),
                    resource_version=int(context["incident_version"]),
                    operation_id=operation_id,
                    status="PENDING_APPROVAL",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ApprovalRequestRecord(
                    approval_id=approval_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    policy_version="m2-t2-v1",
                    status="PENDING",
                    expires_at=min(candidate.expires_at, now + T2_APPROVAL_TTL),
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
        return ServiceQuotationProposal(
            proposal_id,
            approval_id,
            parameters,
            operation_id,
        )

    def list_service_quotations(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        request_id: str,
    ) -> list[ServiceQuotationRecord]:
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident is None:
                raise ApprovalNotVisible
            self._authorizer.require(
                identity,
                Action.READ_SERVICE_QUOTATION,
                ResourceContext(identity.tenant_id, incident_id, incident.asset_id),
                request_id=request_id,
            )
            return list(
                session.scalars(
                    select(ServiceQuotationRecord)
                    .where(
                        ServiceQuotationRecord.tenant_id == identity.tenant_id,
                        ServiceQuotationRecord.incident_id == incident_id,
                    )
                    .order_by(ServiceQuotationRecord.quotation_version.desc())
                )
            )

    def _service_quotation_context(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        request_id: str,
        action: Action,
    ) -> dict[str, Any]:
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident is None:
                raise ApprovalNotVisible
            self._authorizer.require(
                identity,
                action,
                ResourceContext(identity.tenant_id, incident_id, incident.asset_id),
                request_id=request_id,
            )
            if incident.status in {"CLOSED", "CANCELLED"}:
                raise ServiceQuotationConflict("incident_not_quotable")
            diagnosis = session.scalar(
                select(DiagnosisRunRecord)
                .where(
                    DiagnosisRunRecord.tenant_id == identity.tenant_id,
                    DiagnosisRunRecord.incident_id == incident_id,
                    DiagnosisRunRecord.status.in_(("COMPLETED", "NEEDS_INFORMATION")),
                )
                .order_by(
                    DiagnosisRunRecord.created_at.desc(),
                    DiagnosisRunRecord.diagnosis_run_id.desc(),
                )
            )
            if diagnosis is None:
                raise ServiceQuotationConflict("terminal_diagnosis_required")
            asset_id = incident.asset_id
            incident_version = incident.version
            diagnosis_id = diagnosis.diagnosis_run_id
            diagnosis_version = diagnosis.version

        from industrial_ops_agent.service_entitlements.service import (
            ServiceEntitlementService,
        )

        entitlement = ServiceEntitlementService(
            AssetQueryService(self._database, self._authorizer),
            self,
        ).lookup(identity, asset_id, request_id=request_id)
        if entitlement.decision == "FACTS_UNAVAILABLE":
            raise ServiceQuotationUnavailable("service_entitlement_facts_unavailable")
        if entitlement.decision != "COVERED" and entitlement.decision != "NOT_COVERED":
            raise ServiceQuotationConflict("service_entitlement_review_required")
        return {
            "incident_id": incident_id,
            "asset_id": asset_id,
            "incident_version": incident_version,
            "diagnosis_run_id": diagnosis_id,
            "diagnosis_version": diagnosis_version,
            "entitlement": _service_entitlement_binding(entitlement),
        }

    def _resolve_service_quotation_candidate(
        self,
        identity: IdentityContext,
        *,
        candidate_id: str,
        incident_id: str,
        asset_id: str,
        diagnosis_id: str,
        now: datetime,
    ) -> ServiceQuotationCandidate:
        adapter = self._quotations
        if adapter is None:
            raise ServiceQuotationUnavailable(
                "enterprise_service_quotation_catalog_disabled"
            )
        try:
            raw = adapter.resolve_candidate(
                tenant_id=identity.tenant_id,
                candidate_id=candidate_id,
                incident_ref=incident_id,
                asset_ref=asset_id,
                diagnosis_ref=diagnosis_id,
            )
        except EnterpriseToolError as exc:
            raise ServiceQuotationUnavailable(exc.reason) from exc
        except (OSError, TimeoutError) as exc:
            raise ServiceQuotationUnavailable(
                "service_quotation_catalog_unavailable"
            ) from exc
        if raw is None:
            raise ServiceQuotationConflict("service_quotation_candidate_not_available")
        try:
            return validate_service_quotation_candidate(
                raw,
                provider_id=adapter.provider_id,
                incident_ref=incident_id,
                asset_ref=asset_id,
                diagnosis_ref=diagnosis_id,
                now=now,
            )
        except EnterpriseToolError as exc:
            raise ServiceQuotationConflict(exc.reason) from exc

    def _approval_is_service_quotation(
        self,
        identity: IdentityContext,
        approval_id: str,
    ) -> bool:
        with self._database.transaction(identity.tenant_context) as session:
            return (
                session.scalar(
                    select(ActionProposalRecord.tool_id)
                    .join(
                        ApprovalRequestRecord,
                        ApprovalRequestRecord.proposal_id
                        == ActionProposalRecord.proposal_id,
                    )
                    .where(
                        ApprovalRequestRecord.tenant_id == identity.tenant_id,
                        ApprovalRequestRecord.approval_id == approval_id,
                    )
                )
                == "service.quote.publish"
            )

    def _approval_is_repair_work_order(
        self,
        identity: IdentityContext,
        approval_id: str,
    ) -> bool:
        with self._database.transaction(identity.tenant_context) as session:
            return (
                session.scalar(
                    select(ActionProposalRecord.tool_id)
                    .join(
                        ApprovalRequestRecord,
                        ApprovalRequestRecord.proposal_id
                        == ActionProposalRecord.proposal_id,
                    )
                    .where(
                        ApprovalRequestRecord.tenant_id == identity.tenant_id,
                        ApprovalRequestRecord.approval_id == approval_id,
                    )
                )
                == "work_order.create"
            )

    def _execute_repair_work_order(
        self,
        identity: IdentityContext,
        *,
        approval_id: str,
        parameters: dict[str, Any],
        request_id: str,
        expected_approval_version: int | None,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            approval, proposal = self._load(session, identity.tenant_id, approval_id)
            rejection = ApprovalBindingGuard.execution_rejection(
                status=approval.status,
                current_version=approval.version,
                expected_version=expected_approval_version,
                expires_at=_utc(approval.expires_at),
                now=now,
                approved_parameters_hash=proposal.parameters_hash,
                supplied_parameters_hash=_digest(parameters),
            )
            if rejection is not None:
                return ExecutionResult(rejection)
            self._authorizer.require(
                identity,
                Action.EXECUTE_ACTION,
                ResourceContext(identity.tenant_id, approval_id),
                request_id=request_id,
            )
            definition = self._registry.get(proposal.tool_id, proposal.tool_version)
            definition.validate(parameters)
            if definition.risk_tier != "T2" or not definition.approval_required:
                return ExecutionResult("RISK_CONTRACT_CHANGED")
            existing = session.scalar(
                select(WorkOrderRecord).where(
                    WorkOrderRecord.tenant_id == identity.tenant_id,
                    WorkOrderRecord.proposal_id == proposal.proposal_id,
                )
            )
            if existing is not None:
                return ExecutionResult(
                    "SUCCEEDED",
                    existing.reservation_id,
                    existing.work_order_id,
                )
            incident_id = proposal.incident_id
            operation_id = proposal.operation_id
            proposal_id = proposal.proposal_id

        context = self._repair_work_order_context(
            identity,
            incident_id=incident_id,
            request_id=request_id,
            action=Action.EXECUTE_ACTION,
        )
        approved_diagnosis = parameters.get("diagnosis")
        approved_entitlement = parameters.get("entitlement")
        approved_quotation = parameters.get("quotation")
        if (
            context.get("unavailable_reason") is not None
            or context.get("option") is None
            or context["asset_id"] != parameters.get("asset_id")
            or context["incident_version"] != parameters.get("incident_version")
            or operation_id != parameters.get("operation_id")
            or parameters.get("fulfillment_mode") != "NO_INITIAL_PARTS"
            or not isinstance(approved_diagnosis, dict)
            or approved_diagnosis
            != {
                "diagnosis_run_id": context["diagnosis_run_id"],
                "version": context["diagnosis_version"],
            }
            or not isinstance(approved_entitlement, dict)
            or _stable_entitlement_binding(approved_entitlement)
            != _stable_entitlement_binding(cast(dict[str, Any], context["entitlement"]))
            or approved_quotation != context.get("quotation")
            or cast(RepairWorkOrderOption, context["option"]).authorization_type
            != parameters.get("authorization_type")
        ):
            return ExecutionResult("AUTHORITATIVE_FACT_CHANGED")

        with self._database.transaction(identity.tenant_context) as session:
            approval, proposal = self._load(session, identity.tenant_id, approval_id)
            rejection = ApprovalBindingGuard.execution_rejection(
                status=approval.status,
                current_version=approval.version,
                expected_version=expected_approval_version,
                expires_at=_utc(approval.expires_at),
                now=now,
                approved_parameters_hash=proposal.parameters_hash,
                supplied_parameters_hash=_digest(parameters),
            )
            if rejection is not None:
                return ExecutionResult(rejection)
            existing = session.scalar(
                select(WorkOrderRecord).where(
                    WorkOrderRecord.tenant_id == identity.tenant_id,
                    WorkOrderRecord.proposal_id == proposal_id,
                )
            )
            if existing is not None:
                return ExecutionResult(
                    "SUCCEEDED",
                    existing.reservation_id,
                    existing.work_order_id,
                )
            incident = session.scalar(
                select(IncidentRecord)
                .where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
                .with_for_update()
            )
            diagnosis = session.scalar(
                select(DiagnosisRunRecord).where(
                    DiagnosisRunRecord.tenant_id == identity.tenant_id,
                    DiagnosisRunRecord.diagnosis_run_id == context["diagnosis_run_id"],
                )
            )
            other_work_order = session.scalar(
                select(WorkOrderRecord.work_order_id).where(
                    WorkOrderRecord.tenant_id == identity.tenant_id,
                    WorkOrderRecord.incident_id == incident_id,
                )
            )
            if (
                incident is None
                or incident.status != "DIAGNOSED"
                or incident.version != context["incident_version"]
                or diagnosis is None
                or diagnosis.status != "COMPLETED"
                or diagnosis.version != context["diagnosis_version"]
                or other_work_order is not None
                or not self._repair_quotation_binding_matches(
                    session,
                    identity.tenant_id,
                    incident,
                    cast(dict[str, Any] | None, approved_quotation),
                )
            ):
                return ExecutionResult("AUTHORITATIVE_FACT_CHANGED")

            work_order_id = f"work-order-{uuid4().hex}"
            attempt_id = f"attempt-{uuid4().hex}"
            session.add(
                ExecutionAttemptRecord(
                    attempt_id=attempt_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    approval_id=approval_id,
                    operation_id=operation_id,
                    attempt_number=1,
                    status="SUCCEEDED",
                    reason=None,
                    result={
                        "work_order_id": work_order_id,
                        "creation_mode": "SERVICE_AUTHORIZATION",
                        "idempotency_key_digest": (
                            sha256(idempotency_key.encode()).hexdigest()
                            if idempotency_key is not None
                            else None
                        ),
                    },
                    started_at=now,
                    finished_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            quotation_id = (
                str(approved_quotation["quotation_id"])
                if isinstance(approved_quotation, dict)
                else None
            )
            session.add(
                WorkOrderRecord(
                    work_order_id=work_order_id,
                    tenant_id=identity.tenant_id,
                    incident_id=incident_id,
                    proposal_id=proposal_id,
                    reservation_id=None,
                    creation_mode="SERVICE_AUTHORIZATION",
                    authorization_type=str(parameters["authorization_type"]),
                    diagnosis_run_id=str(context["diagnosis_run_id"]),
                    diagnosis_version=int(context["diagnosis_version"]),
                    service_quotation_id=quotation_id,
                    status="READY",
                    assigned_subject_id=None,
                    pending_assignment_operation_id=None,
                    part_issue_required=False,
                    part_accounting_required=False,
                    priority="NORMAL",
                    sla_due_at=now + timedelta(hours=8),
                    service_window_start=None,
                    service_window_end=None,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            # The event mapper has no ORM relationship to the new work order.
            # Flush the parent explicitly so PostgreSQL cannot insert the child first.
            session.flush()
            session.add(
                WorkOrderEventRecord(
                    event_id=f"work-event-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    work_order_id=work_order_id,
                    sequence=1,
                    event_type="work_order.ready",
                    actor_subject_id=identity.subject_id,
                    payload={
                        "creation_mode": "SERVICE_AUTHORIZATION",
                        "authorization_type": parameters["authorization_type"],
                        "diagnosis_run_id": context["diagnosis_run_id"],
                        "service_quotation_id": quotation_id,
                    },
                    occurred_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            previous_status = incident.status
            incident.status = "WORK_ORDER_CREATED"
            incident.version += 1
            incident.updated_at = now
            session.add(
                IncidentControlRecord(
                    control_id=f"incident-control-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    incident_id=incident_id,
                    incident_version=incident.version,
                    command_type="CREATE_WORK_ORDER",
                    previous_status=previous_status,
                    target_status=incident.status,
                    reason="Approved service authorization created work order",
                    actor_subject_id=identity.subject_id,
                    responsible_subject_id=None,
                    recovery_condition=None,
                    related_incident_id=None,
                    details_json={
                        "work_order_id": work_order_id,
                        "proposal_id": proposal_id,
                        "approval_id": approval_id,
                        "operation_id": operation_id,
                        "creation_mode": "SERVICE_AUTHORIZATION",
                        "authorization_type": parameters["authorization_type"],
                        "diagnosis_run_id": context["diagnosis_run_id"],
                        "service_quotation_id": quotation_id,
                    },
                    occurred_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            proposal.status = "EXECUTED"
            return ExecutionResult(
                "SUCCEEDED",
                None,
                work_order_id,
                execution_attempt_id=attempt_id,
            )

    def _execute_service_quotation(
        self,
        identity: IdentityContext,
        *,
        approval_id: str,
        parameters: dict[str, Any],
        request_id: str,
        expected_approval_version: int | None,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            approval, proposal = self._load(session, identity.tenant_id, approval_id)
            rejection = ApprovalBindingGuard.execution_rejection(
                status=approval.status,
                current_version=approval.version,
                expected_version=expected_approval_version,
                expires_at=_utc(approval.expires_at),
                now=now,
                approved_parameters_hash=proposal.parameters_hash,
                supplied_parameters_hash=_digest(parameters),
            )
            if rejection is not None:
                return ExecutionResult(rejection)
            self._authorizer.require(
                identity,
                Action.EXECUTE_ACTION,
                ResourceContext(identity.tenant_id, approval_id),
                request_id=request_id,
            )
            definition = self._registry.get(proposal.tool_id, proposal.tool_version)
            definition.validate(parameters)
            if definition.risk_tier != "T2" or not definition.approval_required:
                return ExecutionResult("RISK_CONTRACT_CHANGED")
            existing = session.scalar(
                select(ServiceQuotationRecord).where(
                    ServiceQuotationRecord.tenant_id == identity.tenant_id,
                    ServiceQuotationRecord.operation_id == proposal.operation_id,
                )
            )
            if existing is not None:
                return ServiceQuotationExecutionResult(
                    status="SUCCEEDED",
                    service_quotation_id=existing.quotation_id,
                )
            incident_id = proposal.incident_id
            operation_id = proposal.operation_id
            proposal_id = proposal.proposal_id

        context = self._service_quotation_context(
            identity,
            incident_id=incident_id,
            request_id=request_id,
            action=Action.PROPOSE_SERVICE_QUOTATION,
        )
        approved_diagnosis = parameters.get("diagnosis")
        approved_entitlement = parameters.get("entitlement")
        approved_candidate = parameters.get("candidate")
        if (
            context["asset_id"] != parameters.get("asset_id")
            or context["incident_version"] != parameters.get("incident_version")
            or not isinstance(approved_diagnosis, dict)
            or approved_diagnosis
            != {
                "diagnosis_run_id": context["diagnosis_run_id"],
                "version": context["diagnosis_version"],
            }
            or not isinstance(approved_entitlement, dict)
            or _stable_entitlement_binding(approved_entitlement)
            != _stable_entitlement_binding(context["entitlement"])
            or not isinstance(approved_candidate, dict)
        ):
            return ExecutionResult("AUTHORITATIVE_FACT_CHANGED")
        candidate = self._resolve_service_quotation_candidate(
            identity,
            candidate_id=str(approved_candidate.get("candidate_id", "")),
            incident_id=incident_id,
            asset_id=str(context["asset_id"]),
            diagnosis_id=str(context["diagnosis_run_id"]),
            now=now,
        )
        if service_quotation_candidate_binding(candidate) != approved_candidate:
            return ExecutionResult("AUTHORITATIVE_FACT_CHANGED")

        with self._database.transaction(identity.tenant_context) as session:
            approval, proposal = self._load(session, identity.tenant_id, approval_id)
            rejection = ApprovalBindingGuard.execution_rejection(
                status=approval.status,
                current_version=approval.version,
                expected_version=expected_approval_version,
                expires_at=_utc(approval.expires_at),
                now=now,
                approved_parameters_hash=proposal.parameters_hash,
                supplied_parameters_hash=_digest(parameters),
            )
            if rejection is not None:
                return ExecutionResult(rejection)
            existing = session.scalar(
                select(ServiceQuotationRecord).where(
                    ServiceQuotationRecord.tenant_id == identity.tenant_id,
                    ServiceQuotationRecord.operation_id == operation_id,
                )
            )
            if existing is not None:
                return ServiceQuotationExecutionResult(
                    status="SUCCEEDED",
                    service_quotation_id=existing.quotation_id,
                )
            incident = session.scalar(
                select(IncidentRecord)
                .where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
                .with_for_update()
            )
            diagnosis = session.scalar(
                select(DiagnosisRunRecord).where(
                    DiagnosisRunRecord.tenant_id == identity.tenant_id,
                    DiagnosisRunRecord.diagnosis_run_id
                    == context["diagnosis_run_id"],
                )
            )
            if (
                incident is None
                or incident.version != context["incident_version"]
                or diagnosis is None
                or diagnosis.version != context["diagnosis_version"]
                or diagnosis.status not in {"COMPLETED", "NEEDS_INFORMATION"}
            ):
                return ExecutionResult("AUTHORITATIVE_FACT_CHANGED")
            previous = session.scalar(
                select(ServiceQuotationRecord)
                .where(
                    ServiceQuotationRecord.tenant_id == identity.tenant_id,
                    ServiceQuotationRecord.incident_id == incident_id,
                )
                .order_by(ServiceQuotationRecord.quotation_version.desc())
                .with_for_update()
            )
            latest_version = session.scalar(
                select(func.max(ServiceQuotationRecord.quotation_version)).where(
                    ServiceQuotationRecord.tenant_id == identity.tenant_id,
                    ServiceQuotationRecord.incident_id == incident_id,
                )
            )
            quotation_version = int(latest_version or 0) + 1
            quotation_id = f"service-quotation-{uuid4().hex}"
            if previous is not None and previous.status == "PUBLISHED":
                previous.status = "SUPERSEDED"
                previous.state_version += 1
                previous.updated_at = now
            session.add(
                ServiceQuotationRecord(
                    quotation_id=quotation_id,
                    tenant_id=identity.tenant_id,
                    incident_id=incident_id,
                    asset_id=str(context["asset_id"]),
                    diagnosis_run_id=str(context["diagnosis_run_id"]),
                    diagnosis_version=int(context["diagnosis_version"]),
                    quotation_version=quotation_version,
                    supersedes_quotation_id=(
                        previous.quotation_id if previous is not None else None
                    ),
                    proposal_id=proposal_id,
                    approval_id=approval_id,
                    operation_id=operation_id,
                    provider=candidate.provider,
                    candidate_id=candidate.candidate_id,
                    source_version=candidate.source_version,
                    source_as_of=candidate.as_of,
                    line_items_json=list(approved_candidate["line_items"]),
                    currency=candidate.currency,
                    subtotal=candidate.subtotal,
                    discount=candidate.discount,
                    tax=candidate.tax,
                    total=candidate.total,
                    entitlement_json=approved_entitlement,
                    status="PUBLISHED",
                    published_at=now,
                    expires_at=candidate.expires_at,
                    state_version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ExecutionAttemptRecord(
                    attempt_id=f"attempt-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    approval_id=approval_id,
                    operation_id=operation_id,
                    attempt_number=1,
                    status="SUCCEEDED",
                    reason=None,
                    result={
                        "service_quotation_id": quotation_id,
                        "quotation_version": quotation_version,
                        "idempotency_key_digest": (
                            sha256(idempotency_key.encode()).hexdigest()
                            if idempotency_key is not None
                            else None
                        ),
                    },
                    started_at=now,
                    finished_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            proposal.status = "EXECUTED"
            return ServiceQuotationExecutionResult(
                status="SUCCEEDED",
                service_quotation_id=quotation_id,
            )

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Expose the active immutable registry for read-only governance views."""

        return self._registry.definitions()

    def transport_bindings(self) -> tuple[ToolTransportBinding, ...]:
        """Expose non-secret adapter bindings for governance and release manifests."""

        if not isinstance(self._readonly, ToolTransportBindingProvider):
            return ()
        bindings: list[ToolTransportBinding] = []
        for definition in self._registry.definitions():
            if definition.risk_tier not in {"T0", "T1"}:
                continue
            try:
                bindings.append(self._readonly.transport_binding(definition.tool_id))
            except EnterpriseToolError:
                continue
        return tuple(bindings)

    def mcp_server_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        for binding in self.transport_bindings():
            if (
                binding.transport == "MCP_STREAMABLE_HTTP"
                and binding.server_id is not None
                and binding.server_version is not None
            ):
                existing = versions.setdefault(binding.server_id, binding.server_version)
                if existing != binding.server_version:
                    raise RuntimeError("MCP server has conflicting configured versions")
        return dict(sorted(versions.items()))

    def create_equipment_control_handoff(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        incident_version: int,
        requested_operation: str,
        reason: str,
        evidence_ids: list[str],
        client_operation_id: str,
        request_id: str,
        agent_run_id: str | None = None,
    ) -> EquipmentControlHandoffRecord:
        """Persist an evidence-only T3 handoff without exposing an execution path."""

        requested_tool_id = _EQUIPMENT_CONTROL_TOOL_IDS.get(requested_operation)
        if requested_tool_id is None:
            raise EquipmentControlHandoffConflict("unsupported_operation")
        normalized_evidence_ids = sorted(set(evidence_ids))
        if len(normalized_evidence_ids) != len(evidence_ids):
            raise EquipmentControlHandoffConflict("duplicate_evidence")
        request_payload: dict[str, Any] = {
            "incident_id": incident_id,
            "incident_version": incident_version,
            "requested_operation": requested_operation,
            "reason": reason,
            "evidence_ids": normalized_evidence_ids,
        }
        request_digest = _digest(request_payload)
        now = datetime.now(UTC)
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            row = (
                session.execute(
                    select(IncidentRecord, AssetSiteLinkRecord.site_id)
                    .outerjoin(
                        AssetSiteLinkRecord,
                        AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                    )
                    .where(
                        IncidentRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.incident_id == incident_id,
                    )
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                raise EquipmentControlHandoffNotVisible
            incident, site_id = row
            resource = _scoped_resource(
                identity,
                resource_id=incident_id,
                asset_id=incident.asset_id,
                site_id=site_id,
            )
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                resource,
                request_id=request_id,
            )
            existing = session.scalar(
                select(EquipmentControlHandoffRecord).where(
                    EquipmentControlHandoffRecord.tenant_id == identity.tenant_id,
                    EquipmentControlHandoffRecord.client_operation_id == client_operation_id,
                )
            )
            if existing is not None:
                if existing.request_digest != request_digest:
                    raise EquipmentControlHandoffConflict("idempotency_key_reused")
                _record_handoff_sources(session, identity, (existing,), request_id)
                return existing
            if incident.version != incident_version:
                raise EquipmentControlHandoffConflict("incident_version_changed")
            if incident.status in {"CLOSED", "CANCELLED"}:
                raise EquipmentControlHandoffConflict("incident_not_active")
            bundles = list(
                session.scalars(
                    select(EvidenceBundleRecord).where(
                        EvidenceBundleRecord.tenant_id == identity.tenant_id,
                        EvidenceBundleRecord.bundle_id.in_(normalized_evidence_ids),
                        EvidenceBundleRecord.asset_id == incident.asset_id,
                        EvidenceBundleRecord.status == "CONFIRMED",
                    )
                )
            )
            if len(bundles) != len(normalized_evidence_ids):
                raise EquipmentControlHandoffConflict("evidence_not_confirmed_or_visible")
            parameters = {
                "incident_id": incident_id,
                "asset_id": incident.asset_id,
                "requested_operation": requested_operation,
                "reason": reason,
                "evidence_ids": normalized_evidence_ids,
            }
            definition = self._registry.get(requested_tool_id, "1.1.0")
            definition.validate(parameters)
            if definition.risk_tier != "T3":
                raise EquipmentControlHandoffConflict("risk_contract_changed")
            tool_call_id = f"tool-call-{uuid4().hex}"
            handoff_id = f"equipment-handoff-{uuid4().hex}"
            output_metadata = {
                "type": "EXTERNAL_SAFETY_HANDOFF",
                "handoff_id": handoff_id,
                "target": _EQUIPMENT_CONTROL_TARGET,
                "requested_tool_id": requested_tool_id,
                "reason": "external_safety_handoff_required",
            }
            self._authorizer.record_guard_decision(
                identity,
                action="tool.t3.execute",
                decision="deny",
                reason_code="external_safety_handoff_required",
                request_id=request_id,
                resource_id=requested_tool_id,
            )
            session.add(
                ToolCallRecord(
                    tool_call_id=tool_call_id,
                    tenant_id=identity.tenant_id,
                    agent_run_id=agent_run_id,
                    tool_id=requested_tool_id,
                    tool_version=definition.version,
                    risk_tier="T3",
                    status="EXTERNAL_HANDOFF",
                    input_digest=_digest(parameters),
                    output_metadata=output_metadata,
                    created_at=now,
                    updated_at=now,
                )
            )
            handoff = EquipmentControlHandoffRecord(
                handoff_id=handoff_id,
                tenant_id=identity.tenant_id,
                client_operation_id=client_operation_id,
                request_digest=request_digest,
                incident_id=incident_id,
                asset_id=incident.asset_id,
                tool_call_id=tool_call_id,
                requested_tool_id=requested_tool_id,
                requested_operation=requested_operation,
                reason=reason,
                evidence_ids=normalized_evidence_ids,
                target_system=_EQUIPMENT_CONTROL_TARGET,
                status="EXTERNAL_REVIEW_REQUIRED",
                requested_by_subject_id=identity.subject_id,
                incident_version=incident_version,
                created_at=now,
                updated_at=now,
            )
            session.add(handoff)
            _record_handoff_sources(session, identity, (handoff,), request_id)
            return handoff

    def list_equipment_control_handoffs(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        request_id: str,
    ) -> list[EquipmentControlHandoffRecord]:
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            row = (
                session.execute(
                    select(IncidentRecord, AssetSiteLinkRecord.site_id)
                    .outerjoin(
                        AssetSiteLinkRecord,
                        AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                    )
                    .where(
                        IncidentRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.incident_id == incident_id,
                    )
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                raise EquipmentControlHandoffNotVisible
            incident, site_id = row
            self._authorizer.require(
                identity,
                Action.READ_EQUIPMENT_CONTROL_HANDOFF,
                _scoped_resource(
                    identity,
                    resource_id=incident_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                ),
                request_id=request_id,
            )
            records = list(
                session.scalars(
                    select(EquipmentControlHandoffRecord)
                    .where(
                        EquipmentControlHandoffRecord.tenant_id == identity.tenant_id,
                        EquipmentControlHandoffRecord.incident_id == incident_id,
                    )
                    .order_by(
                        EquipmentControlHandoffRecord.created_at.desc(),
                        EquipmentControlHandoffRecord.handoff_id,
                    )
                )
            )
            _record_handoff_sources(session, identity, records, request_id)
            return records

    def invoke(
        self,
        identity: IdentityContext,
        *,
        tool_id: str,
        version: str,
        parameters: dict[str, Any],
        request_id: str,
        agent_run_id: str | None = None,
    ) -> ToolInvocationResult:
        """Invoke a registered T0/T1 tool or convert every T3 call to a handoff."""

        try:
            definition = self._registry.get(tool_id, version)
        except ValueError:
            self._authorizer.record_guard_decision(
                identity,
                action="tool.registry.resolve",
                decision="deny",
                reason_code="tool_not_registered",
                request_id=request_id,
                resource_id=tool_id,
            )
            raise
        try:
            definition.validate(parameters)
        except ValidationError:
            self._authorizer.record_guard_decision(
                identity,
                action="tool.schema.validate",
                decision="deny",
                reason_code="tool_parameters_invalid",
                request_id=request_id,
                resource_id=tool_id,
            )
            raise
        if definition.risk_tier != "T3":
            self._rate_limiter.require(
                identity.tenant_id,
                tool_id,
                definition.rate_limit_per_minute,
            )
        tool_call_id = f"tool-call-{uuid4().hex}"
        now = datetime.now(UTC)
        if definition.risk_tier == "T2":
            metadata = {
                "approval_required": True,
                "reason": "T2 tools require a bound proposal and a separate human approval",
            }
            self._authorizer.record_guard_decision(
                identity,
                action="tool.t2.execute",
                decision="deny",
                reason_code="bound_approval_required",
                request_id=request_id,
                resource_id=tool_id,
            )
            self._record_tool_call(
                identity,
                tool_call_id=tool_call_id,
                agent_run_id=agent_run_id,
                tool_id=tool_id,
                version=version,
                risk_tier="T2",
                status="APPROVAL_REQUIRED",
                parameters=parameters,
                output_metadata=metadata,
                now=now,
            )
            return ToolInvocationResult("APPROVAL_REQUIRED", tool_call_id, data=metadata)
        if definition.risk_tier == "T3":
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                self._readonly_resource_context(identity, tool_id, parameters),
                request_id=request_id,
            )
            handoff = {
                "type": "EXTERNAL_SAFETY_HANDOFF",
                "target": _EQUIPMENT_CONTROL_TARGET,
                "requested_tool_id": tool_id,
                "reason": "T3 equipment control is never executed by the Agent platform",
            }
            self._authorizer.record_guard_decision(
                identity,
                action="tool.t3.execute",
                decision="deny",
                reason_code="external_safety_handoff_required",
                request_id=request_id,
                resource_id=tool_id,
            )
            self._record_tool_call(
                identity,
                tool_call_id=tool_call_id,
                agent_run_id=agent_run_id,
                tool_id=tool_id,
                version=version,
                risk_tier="T3",
                status="EXTERNAL_HANDOFF",
                parameters=parameters,
                output_metadata=handoff,
                now=now,
            )
            return ToolInvocationResult("EXTERNAL_HANDOFF", tool_call_id, handoff=handoff)

        self._authorizer.require(
            identity,
            Action(definition.required_action),
            self._readonly_resource_context(identity, tool_id, parameters),
            request_id=request_id,
        )
        try:
            data = self._readonly.invoke(tool_id, parameters)
            validate_attested_tool_fact(tool_id, data)
        except EnterpriseToolError as exc:
            self._record_tool_call(
                identity,
                tool_call_id=tool_call_id,
                agent_run_id=agent_run_id,
                tool_id=tool_id,
                version=version,
                risk_tier=definition.risk_tier,
                status="FAILED",
                parameters=parameters,
                output_metadata={
                    "reason": exc.reason,
                    **self._transport_metadata(tool_id),
                },
                now=now,
            )
            raise
        self._record_tool_call(
            identity,
            tool_call_id=tool_call_id,
            agent_run_id=agent_run_id,
            tool_id=tool_id,
            version=version,
            risk_tier=definition.risk_tier,
            status="SUCCEEDED",
            parameters=parameters,
            output_metadata={
                **{key: data[key] for key in ("source", "source_record_id", "as_of")},
                "authority": data["authority"],
                **self._transport_metadata(tool_id),
            },
            now=now,
        )
        return ToolInvocationResult("SUCCEEDED", tool_call_id, data=data)

    def _transport_metadata(self, tool_id: str) -> dict[str, Any]:
        if not isinstance(self._readonly, ToolTransportBindingProvider):
            return {"transport": "IN_PROCESS_ADAPTER"}
        try:
            binding = self._readonly.transport_binding(tool_id)
        except EnterpriseToolError:
            return {"transport": "UNKNOWN"}
        return {
            "transport": binding.transport,
            "remote_tool_name": binding.remote_tool_name,
            "server_id": binding.server_id,
            "server_version": binding.server_version,
            "required_scopes": list(binding.required_scopes),
            "input_schema_digest": binding.input_schema_digest,
        }

    def _readonly_resource_context(
        self,
        identity: IdentityContext,
        tool_id: str,
        parameters: dict[str, Any],
    ) -> ResourceContext:
        asset_id = str(parameters["asset_id"])
        if asset_id in identity.asset_ids:
            return ResourceContext(
                identity.tenant_id,
                resource_id=tool_id,
                asset_id=asset_id,
            )
        if identity.site_ids:
            with self._database.transaction(identity.tenant_context) as session:
                site_id = session.scalar(
                    select(AssetSiteLinkRecord.site_id).where(
                        AssetSiteLinkRecord.tenant_id == identity.tenant_id,
                        AssetSiteLinkRecord.asset_id == asset_id,
                    )
                )
            if site_id in identity.site_ids:
                return ResourceContext(
                    identity.tenant_id,
                    resource_id=tool_id,
                    site_id=site_id,
                )
        return ResourceContext(
            identity.tenant_id,
            resource_id=tool_id,
            asset_id=asset_id,
        )

    def propose_parts_reservation(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        incident_version: int,
        part_number: str,
        quantity: int,
        request_id: str,
        idempotency_key: str | None = None,
    ) -> PartsProposal:
        now = datetime.now(UTC)
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident is None:
                raise ApprovalNotVisible
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                ResourceContext(identity.tenant_id, incident_id, incident.asset_id),
                request_id=request_id,
            )
            if incident.version != incident_version or incident.status != "DIAGNOSED":
                raise ApprovalConflict
            parameters = {
                "incident_id": incident_id,
                "asset_id": incident.asset_id,
                "part_number": part_number,
                "quantity": quantity,
            }
            if idempotency_key is not None:
                installed_component_id = session.scalar(
                    select(AssetComponentRecord.component_id).where(
                        AssetComponentRecord.tenant_id == identity.tenant_id,
                        AssetComponentRecord.asset_id == incident.asset_id,
                        AssetComponentRecord.part_number == part_number,
                        AssetComponentRecord.status == "INSTALLED",
                    )
                )
                if installed_component_id is None:
                    raise PartsReservationProposalNotEligible(
                        "installed_component_mismatch"
                    )
                try:
                    availability = self._parts.check_availability(
                        part_number=part_number,
                        quantity=quantity,
                    )
                    snapshot = _purchase_inventory_snapshot(
                        availability,
                        expected_part_number=part_number,
                        requested_quantity=quantity,
                        now=now,
                    )
                except EnterpriseToolError as exc:
                    raise PartsReservationProposalDependencyUnavailable(
                        exc.reason
                    ) from exc
                except (KeyError, TypeError, ValueError) as exc:
                    raise PartsReservationProposalDependencyUnavailable(
                        "inventory_fact_invalid"
                    ) from exc
                if not bool(availability["available"]):
                    raise PartsReservationProposalNotEligible("inventory_insufficient")
                parameters.update(
                    inventory_snapshot=snapshot,
                    compatibility_evidence="INSTALLED_COMPONENT_MATCH",
                )
            definition = self._registry.get("parts.reserve", "1.0.0")
            definition.validate(parameters)
            if idempotency_key is not None:
                request_hash = _digest(
                    {
                        "incident_id": incident_id,
                        "incident_version": incident_version,
                        "part_number": part_number,
                        "quantity": quantity,
                    }
                )
                storage_key = (
                    "parts-reservation-proposal:"
                    + sha256(idempotency_key.encode()).hexdigest()
                )
                existing = session.scalar(
                    select(IdempotencyRecord).where(
                        IdempotencyRecord.tenant_id == identity.tenant_id,
                        IdempotencyRecord.subject_id == identity.subject_id,
                        IdempotencyRecord.key == storage_key,
                    )
                )
                if existing is not None:
                    if existing.request_hash != request_hash:
                        raise PartsReservationProposalIdempotencyConflict
                    stored_proposal = session.scalar(
                        select(ActionProposalRecord).where(
                            ActionProposalRecord.tenant_id == identity.tenant_id,
                            ActionProposalRecord.proposal_id == existing.result_ref,
                        )
                    )
                    stored_approval = session.scalar(
                        select(ApprovalRequestRecord).where(
                            ApprovalRequestRecord.tenant_id == identity.tenant_id,
                            ApprovalRequestRecord.proposal_id == existing.result_ref,
                        )
                    )
                    if stored_proposal is None or stored_approval is None:
                        raise RuntimeError(
                            "parts proposal idempotency record references missing facts"
                        )
                    return PartsProposal(
                        stored_proposal.proposal_id,
                        stored_approval.approval_id,
                        dict(stored_proposal.parameters),
                        stored_proposal.operation_id,
                    )
            proposal_id, approval_id, operation_id = (
                f"proposal-{uuid4().hex}",
                f"approval-{uuid4().hex}",
                f"operation-{uuid4().hex}",
            )
            digest = _digest(parameters)
            session.add(
                ActionProposalRecord(
                    proposal_id=proposal_id,
                    tenant_id=identity.tenant_id,
                    incident_id=incident_id,
                    initiator_subject_id=identity.subject_id,
                    tool_id=definition.tool_id,
                    tool_version=definition.version,
                    risk_tier=definition.risk_tier,
                    parameters=parameters,
                    parameters_hash=digest,
                    resource_version=incident_version,
                    operation_id=operation_id,
                    status="PENDING_APPROVAL",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ApprovalRequestRecord(
                    approval_id=approval_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    policy_version="m2-t2-v1",
                    status="PENDING",
                    expires_at=now + T2_APPROVAL_TTL,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            if idempotency_key is not None:
                session.add(
                    IdempotencyRecord(
                        record_id=f"idem-{uuid4().hex}",
                        tenant_id=identity.tenant_id,
                        subject_id=identity.subject_id,
                        key=storage_key,
                        request_hash=request_hash,
                        result_ref=proposal_id,
                        expires_at=now + timedelta(hours=24),
                        created_at=now,
                        updated_at=now,
                    )
                )
            return PartsProposal(proposal_id, approval_id, parameters, operation_id)

    def propose_part_issue(
        self,
        identity: IdentityContext,
        *,
        work_order_id: str,
        work_order_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> PartIssueProposal:
        """Create one server-bound WMS issue proposal without calling the WMS."""

        now = datetime.now(UTC)
        request_hash = _digest(
            {
                "work_order_id": work_order_id,
                "work_order_version": work_order_version,
            }
        )
        storage_key = "part-issue-proposal:" + sha256(idempotency_key.encode()).hexdigest()
        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            row = (
                session.execute(
                    select(WorkOrderRecord, IncidentRecord, AssetSiteLinkRecord.site_id)
                    .join(
                        IncidentRecord,
                        IncidentRecord.incident_id == WorkOrderRecord.incident_id,
                    )
                    .outerjoin(
                        AssetSiteLinkRecord,
                        AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                    )
                    .where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.work_order_id == work_order_id,
                    )
                    .with_for_update()
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                raise ApprovalNotVisible
            work, incident, site_id = row
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                _scoped_resource(
                    identity,
                    resource_id=work_order_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                ),
                request_id=request_id,
            )
            if work.assigned_subject_id != identity.subject_id:
                raise PartIssueProposalConflict("assignee_mismatch")
            if work.creation_mode == "SERVICE_AUTHORIZATION":
                raise PartIssueProposalConflict("parts_not_required")

            existing_idempotency = session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.tenant_id == identity.tenant_id,
                    IdempotencyRecord.subject_id == identity.subject_id,
                    IdempotencyRecord.key == storage_key,
                )
            )
            if existing_idempotency is not None:
                if existing_idempotency.request_hash != request_hash:
                    raise PartIssueProposalIdempotencyConflict
                stored = (
                    session.execute(
                        select(
                            ActionProposalRecord,
                            ApprovalRequestRecord,
                            PartIssueRecord,
                        )
                        .join(
                            ApprovalRequestRecord,
                            ApprovalRequestRecord.proposal_id
                            == ActionProposalRecord.proposal_id,
                        )
                        .join(
                            PartIssueRecord,
                            PartIssueRecord.proposal_id == ActionProposalRecord.proposal_id,
                        )
                        .where(
                            ActionProposalRecord.tenant_id == identity.tenant_id,
                            ActionProposalRecord.proposal_id
                            == existing_idempotency.result_ref,
                        )
                    )
                    .tuples()
                    .one_or_none()
                )
                if stored is None:
                    raise RuntimeError(
                        "part issue idempotency record references missing facts"
                    )
                stored_proposal, stored_approval, stored_issue = stored
                return PartIssueProposal(
                    proposal_id=stored_proposal.proposal_id,
                    approval_id=stored_approval.approval_id,
                    parameters=dict(stored_proposal.parameters),
                    operation_id=stored_proposal.operation_id,
                    part_issue_id=stored_issue.part_issue_id,
                )

            if work.version != work_order_version:
                raise PartIssueProposalConflict("work_order_version_conflict")
            if work.status not in {"ACCEPTED", "IN_PROGRESS", "ON_HOLD"}:
                raise PartIssueProposalConflict("work_order_state_invalid")
            if not work.part_issue_required:
                raise PartIssueProposalConflict("part_issue_not_required")
            reservation = session.scalar(
                select(PartReservationRecord)
                .where(
                    PartReservationRecord.tenant_id == identity.tenant_id,
                    PartReservationRecord.reservation_id == work.reservation_id,
                )
                .with_for_update()
            )
            if (
                reservation is None
                or reservation.incident_id != work.incident_id
                or reservation.status != "RESERVED"
            ):
                raise PartIssueProposalConflict("part_allocation_unavailable")
            existing_issue = session.scalar(
                select(PartIssueRecord).where(
                    PartIssueRecord.tenant_id == identity.tenant_id,
                    PartIssueRecord.work_order_id == work_order_id,
                )
            )
            if existing_issue is not None:
                raise PartIssueProposalConflict("part_issue_already_requested")
            parameters = {
                "incident_id": work.incident_id,
                "asset_id": incident.asset_id,
                "work_order_id": work_order_id,
                "reservation_id": reservation.reservation_id,
                "part_number": reservation.part_number,
                "quantity": reservation.quantity,
            }
            definition = self._registry.get("parts.issue", "1.0.0")
            definition.validate(parameters)
            proposal_id = f"proposal-{uuid4().hex}"
            approval_id = f"approval-{uuid4().hex}"
            operation_id = f"operation-{uuid4().hex}"
            part_issue_id = f"part-issue-{uuid4().hex}"
            session.add(
                ActionProposalRecord(
                    proposal_id=proposal_id,
                    tenant_id=identity.tenant_id,
                    incident_id=work.incident_id,
                    initiator_subject_id=identity.subject_id,
                    tool_id=definition.tool_id,
                    tool_version=definition.version,
                    risk_tier=definition.risk_tier,
                    parameters=parameters,
                    parameters_hash=_digest(parameters),
                    resource_version=work_order_version,
                    operation_id=operation_id,
                    status="PENDING_APPROVAL",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ApprovalRequestRecord(
                    approval_id=approval_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    policy_version="part-003-t2-v1",
                    status="PENDING",
                    expires_at=now + T2_APPROVAL_TTL,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                PartIssueRecord(
                    part_issue_id=part_issue_id,
                    tenant_id=identity.tenant_id,
                    operation_id=operation_id,
                    proposal_id=proposal_id,
                    approval_id=approval_id,
                    work_order_id=work_order_id,
                    incident_id=work.incident_id,
                    reservation_id=reservation.reservation_id,
                    part_number=reservation.part_number,
                    quantity=reservation.quantity,
                    status="PENDING_APPROVAL",
                    external_issue_id=None,
                    source=None,
                    source_record_id=None,
                    as_of=None,
                    reconciliation_id=None,
                    reason=None,
                    submitted_at=None,
                    issued_at=None,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                IdempotencyRecord(
                    record_id=f"idem-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=proposal_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            return PartIssueProposal(
                proposal_id=proposal_id,
                approval_id=approval_id,
                parameters=parameters,
                operation_id=operation_id,
                part_issue_id=part_issue_id,
            )

    def propose_part_consumption(
        self,
        identity: IdentityContext,
        *,
        work_order_id: str,
        work_order_version: int,
        field_entry_id: str,
        idempotency_key: str,
        request_id: str,
    ) -> PartMaterialMovementProposal:
        return self._propose_part_movement(
            identity,
            work_order_id=work_order_id,
            work_order_version=work_order_version,
            movement_kind="CONSUME",
            field_entry_id=field_entry_id,
            quantity=None,
            idempotency_key=idempotency_key,
            request_id=request_id,
        )

    def propose_part_return(
        self,
        identity: IdentityContext,
        *,
        work_order_id: str,
        work_order_version: int,
        quantity: int,
        idempotency_key: str,
        request_id: str,
    ) -> PartMaterialMovementProposal:
        return self._propose_part_movement(
            identity,
            work_order_id=work_order_id,
            work_order_version=work_order_version,
            movement_kind="RETURN",
            field_entry_id=None,
            quantity=quantity,
            idempotency_key=idempotency_key,
            request_id=request_id,
        )

    def _propose_part_movement(
        self,
        identity: IdentityContext,
        *,
        work_order_id: str,
        work_order_version: int,
        movement_kind: str,
        field_entry_id: str | None,
        quantity: int | None,
        idempotency_key: str,
        request_id: str,
    ) -> PartMaterialMovementProposal:
        """Create a server-bound T2 material movement without calling the WMS."""

        now = datetime.now(UTC)
        request_hash = _digest(
            {
                "work_order_id": work_order_id,
                "work_order_version": work_order_version,
                "movement_kind": movement_kind,
                "field_entry_id": field_entry_id,
                "quantity": quantity,
            }
        )
        storage_key = "part-movement-proposal:" + sha256(
            idempotency_key.encode()
        ).hexdigest()
        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            row = (
                session.execute(
                    select(WorkOrderRecord, IncidentRecord, AssetSiteLinkRecord.site_id)
                    .join(
                        IncidentRecord,
                        IncidentRecord.incident_id == WorkOrderRecord.incident_id,
                    )
                    .outerjoin(
                        AssetSiteLinkRecord,
                        AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                    )
                    .where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.work_order_id == work_order_id,
                    )
                    .with_for_update()
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                raise ApprovalNotVisible
            work, incident, site_id = row
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                _scoped_resource(
                    identity,
                    resource_id=work_order_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                ),
                request_id=request_id,
            )
            if work.assigned_subject_id != identity.subject_id:
                raise PartMovementProposalConflict("assignee_mismatch")
            if work.creation_mode == "SERVICE_AUTHORIZATION":
                raise PartMovementProposalConflict("parts_not_required")

            existing_idempotency = session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.tenant_id == identity.tenant_id,
                    IdempotencyRecord.subject_id == identity.subject_id,
                    IdempotencyRecord.key == storage_key,
                )
            )
            if existing_idempotency is not None:
                if existing_idempotency.request_hash != request_hash:
                    raise PartMovementProposalIdempotencyConflict
                stored = (
                    session.execute(
                        select(
                            ActionProposalRecord,
                            ApprovalRequestRecord,
                            PartMaterialMovementRecord,
                        )
                        .join(
                            ApprovalRequestRecord,
                            ApprovalRequestRecord.proposal_id
                            == ActionProposalRecord.proposal_id,
                        )
                        .join(
                            PartMaterialMovementRecord,
                            PartMaterialMovementRecord.proposal_id
                            == ActionProposalRecord.proposal_id,
                        )
                        .where(
                            ActionProposalRecord.tenant_id == identity.tenant_id,
                            ActionProposalRecord.proposal_id
                            == existing_idempotency.result_ref,
                        )
                    )
                    .tuples()
                    .one_or_none()
                )
                if stored is None:
                    raise RuntimeError(
                        "part movement idempotency record references missing facts"
                    )
                stored_proposal, stored_approval, stored_movement = stored
                return PartMaterialMovementProposal(
                    proposal_id=stored_proposal.proposal_id,
                    approval_id=stored_approval.approval_id,
                    parameters=dict(stored_proposal.parameters),
                    operation_id=stored_proposal.operation_id,
                    movement_id=stored_movement.movement_id,
                )

            if work.version != work_order_version:
                raise PartMovementProposalConflict("work_order_version_conflict")
            if work.status not in {"ACCEPTED", "IN_PROGRESS", "ON_HOLD"}:
                raise PartMovementProposalConflict("work_order_state_invalid")
            if not work.part_accounting_required:
                raise PartMovementProposalConflict("part_accounting_not_required")
            reservation = session.scalar(
                select(PartReservationRecord)
                .where(
                    PartReservationRecord.tenant_id == identity.tenant_id,
                    PartReservationRecord.reservation_id == work.reservation_id,
                )
                .with_for_update()
            )
            issue = session.scalar(
                select(PartIssueRecord)
                .where(
                    PartIssueRecord.tenant_id == identity.tenant_id,
                    PartIssueRecord.work_order_id == work_order_id,
                )
                .with_for_update()
            )
            if (
                reservation is None
                or issue is None
                or reservation.incident_id != work.incident_id
                or reservation.status != "ISSUED"
                or issue.status != "ISSUED"
                or issue.reservation_id != reservation.reservation_id
                or issue.quantity != reservation.quantity
                or issue.part_number != reservation.part_number
            ):
                raise PartMovementProposalConflict("part_issue_not_issued")

            accounting = WorkOrderService.part_accounting_in_session(session, work)
            if movement_kind == "CONSUME":
                entry = session.scalar(
                    select(WorkOrderFieldEntryRecord).where(
                        WorkOrderFieldEntryRecord.tenant_id == identity.tenant_id,
                        WorkOrderFieldEntryRecord.work_order_id == work_order_id,
                        WorkOrderFieldEntryRecord.entry_id == field_entry_id,
                        WorkOrderFieldEntryRecord.entry_type == "PART",
                    )
                )
                if entry is None:
                    raise PartMovementProposalConflict("part_field_entry_not_eligible")
                if (
                    entry.payload.get("part_reservation_id") != reservation.reservation_id
                    or entry.payload.get("part_number") != reservation.part_number
                ):
                    raise PartMovementProposalConflict("part_field_entry_not_eligible")
                bound_quantity = WorkOrderService._part_entry_quantity(entry, work.version)
                if not any(item.entry_id == entry.entry_id for item in accounting.eligible_entries):
                    raise PartMovementProposalConflict("part_field_entry_already_accounted")
                parameters = {
                    "incident_id": work.incident_id,
                    "asset_id": incident.asset_id,
                    "work_order_id": work_order_id,
                    "reservation_id": reservation.reservation_id,
                    "part_issue_id": issue.part_issue_id,
                    "part_number": reservation.part_number,
                    "field_entry_id": entry.entry_id,
                    "movement_kind": "CONSUME",
                    "quantity": bound_quantity,
                }
                tool_id = "parts.consume"
            else:
                if (
                    not isinstance(quantity, int)
                    or isinstance(quantity, bool)
                    or quantity <= 0
                    or quantity > accounting.returnable_quantity
                ):
                    raise PartMovementProposalConflict(
                        "part_return_exceeds_available_quantity"
                    )
                parameters = {
                    "incident_id": work.incident_id,
                    "asset_id": incident.asset_id,
                    "work_order_id": work_order_id,
                    "reservation_id": reservation.reservation_id,
                    "part_issue_id": issue.part_issue_id,
                    "part_number": reservation.part_number,
                    "movement_kind": "RETURN",
                    "quantity": quantity,
                }
                tool_id = "parts.return"

            definition = self._registry.get(tool_id, "1.0.0")
            definition.validate(parameters)
            proposal_id = f"proposal-{uuid4().hex}"
            approval_id = f"approval-{uuid4().hex}"
            operation_id = f"operation-{uuid4().hex}"
            movement_id = f"part-movement-{uuid4().hex}"
            session.add(
                ActionProposalRecord(
                    proposal_id=proposal_id,
                    tenant_id=identity.tenant_id,
                    incident_id=work.incident_id,
                    initiator_subject_id=identity.subject_id,
                    tool_id=definition.tool_id,
                    tool_version=definition.version,
                    risk_tier=definition.risk_tier,
                    parameters=parameters,
                    parameters_hash=_digest(parameters),
                    resource_version=work_order_version,
                    operation_id=operation_id,
                    status="PENDING_APPROVAL",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ApprovalRequestRecord(
                    approval_id=approval_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    policy_version="part-004-t2-v1",
                    status="PENDING",
                    expires_at=now + T2_APPROVAL_TTL,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                PartMaterialMovementRecord(
                    movement_id=movement_id,
                    tenant_id=identity.tenant_id,
                    operation_id=operation_id,
                    proposal_id=proposal_id,
                    approval_id=approval_id,
                    work_order_id=work_order_id,
                    incident_id=work.incident_id,
                    part_issue_id=issue.part_issue_id,
                    reservation_id=reservation.reservation_id,
                    field_entry_id=field_entry_id,
                    movement_kind=movement_kind,
                    part_number=reservation.part_number,
                    quantity=int(parameters["quantity"]),
                    status="PENDING_APPROVAL",
                    external_movement_id=None,
                    source=None,
                    source_record_id=None,
                    as_of=None,
                    reconciliation_id=None,
                    reason=None,
                    submitted_at=None,
                    applied_at=None,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                IdempotencyRecord(
                    record_id=f"idem-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=proposal_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            return PartMaterialMovementProposal(
                proposal_id=proposal_id,
                approval_id=approval_id,
                parameters=parameters,
                operation_id=operation_id,
                movement_id=movement_id,
            )

    def list_purchase_request_options(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        request_id: str,
    ) -> PurchaseRequestOptions:
        """Return internal purchase plus a currently valid, safe ERP profile projection."""

        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident is None:
                raise ApprovalNotVisible
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                ResourceContext(identity.tenant_id, incident_id, incident.asset_id),
                request_id=request_id,
            )

        internal = PurchaseRequestOption(
            delivery_target="INTERNAL",
            provider=None,
            profile_id=None,
            profile_version=None,
            display_name="平台内部采购申请",
            as_of=None,
            expires_at=None,
        )
        try:
            profile = self._resolve_procurement_profile(
                tenant_id=identity.tenant_id,
                now=datetime.now(UTC),
            )
        except PurchaseDeliveryProfileUnavailable as exc:
            return PurchaseRequestOptions(incident_id, (internal,), exc.reason)
        enterprise = PurchaseRequestOption(
            delivery_target="ENTERPRISE_ERP",
            provider=profile.provider,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            display_name=profile.display_name,
            as_of=profile.as_of.isoformat(),
            expires_at=profile.expires_at.isoformat(),
        )
        return PurchaseRequestOptions(incident_id, (internal, enterprise), None)

    def list_refund_request_options(
        self,
        identity: IdentityContext,
        *,
        work_order_id: str,
        request_id: str,
    ) -> RefundRequestOptions:
        """Return the internal target plus one currently authorized finance profile."""

        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            row = (
                session.execute(
                    select(WorkOrderRecord, IncidentRecord, AssetSiteLinkRecord.site_id)
                    .join(IncidentRecord, IncidentRecord.incident_id == WorkOrderRecord.incident_id)
                    .outerjoin(
                        AssetSiteLinkRecord,
                        AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                    )
                    .where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.work_order_id == work_order_id,
                    )
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                raise ApprovalNotVisible
            work, incident, site_id = row
            self._authorizer.require(
                identity,
                Action.PROPOSE_REFUND_REQUEST,
                _scoped_resource(
                    identity,
                    resource_id=work_order_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                ),
                request_id=request_id,
            )
            confirmation = _latest_customer_result_confirmation(
                session,
                tenant_id=identity.tenant_id,
                work_order_id=work_order_id,
            )
            if work.status not in {"COMPLETED", "VERIFIED", "CLOSED"} or not _refund_eligible(
                confirmation
            ):
                raise RefundNotEligible
            customer_subject_id = incident.reporter_subject_id

        internal = RefundRequestOption(
            delivery_target="INTERNAL",
            provider=None,
            profile_id=None,
            profile_version=None,
            display_name="平台内部退款申请",
            as_of=None,
            expires_at=None,
        )
        try:
            profile = self._resolve_refund_profile(
                tenant_id=identity.tenant_id,
                customer_subject_id=customer_subject_id,
                now=datetime.now(UTC),
            )
        except RefundDeliveryProfileUnavailable as exc:
            return RefundRequestOptions(work_order_id, (internal,), exc.reason)
        enterprise = RefundRequestOption(
            delivery_target="ENTERPRISE_FINANCE",
            provider=profile.provider,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            display_name=profile.display_name,
            as_of=profile.as_of.isoformat(),
            expires_at=profile.expires_at.isoformat(),
        )
        return RefundRequestOptions(work_order_id, (internal, enterprise), None)

    def list_work_order_assignment_options(
        self,
        identity: IdentityContext,
        *,
        work_order_id: str,
        request_id: str,
    ) -> WorkOrderAssignmentOptions:
        """Return the local path and currently valid server-authorized FSM candidates."""

        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            row = (
                session.execute(
                    select(WorkOrderRecord, IncidentRecord, AssetSiteLinkRecord.site_id)
                    .join(IncidentRecord, IncidentRecord.incident_id == WorkOrderRecord.incident_id)
                    .outerjoin(
                        AssetSiteLinkRecord,
                        AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                    )
                    .where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.work_order_id == work_order_id,
                    )
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                raise ApprovalNotVisible
            work, incident, site_id = row
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                _scoped_resource(
                    identity,
                    resource_id=work_order_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                ),
                request_id=request_id,
            )
            if work.status != "READY" or work.pending_assignment_operation_id is not None:
                raise ApprovalConflict
            service_window_start = work.service_window_start
            service_window_end = work.service_window_end
            asset_id = incident.asset_id

        local = WorkOrderAssignmentOption(
            assignment_target="LOCAL_DIRECTORY",
            provider=None,
            candidate_profile_id=None,
            profile_version=None,
            assignee_subject_id=None,
            display_name="平台本地人员目录",
            matched_skill_codes=(),
            service_window_start=None,
            service_window_end=None,
            travel_minutes=None,
            remaining_work_minutes=None,
            eligibility_code=None,
            source_record_id=None,
            as_of=None,
            expires_at=None,
        )
        if site_id is None:
            return WorkOrderAssignmentOptions(
                work_order_id,
                (local,),
                "fsm_assignment_site_unavailable",
            )
        try:
            candidates = self._resolve_fsm_assignment_candidates(
                tenant_id=identity.tenant_id,
                work_order_id=work_order_id,
                asset_id=asset_id,
                site_id=site_id,
                service_window_start=service_window_start,
                service_window_end=service_window_end,
                now=datetime.now(UTC),
            )
        except FsmAssignmentProfileUnavailable as exc:
            return WorkOrderAssignmentOptions(work_order_id, (local,), exc.reason)

        eligible: list[FsmAssignmentCandidate] = []
        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            current = session.scalar(
                select(WorkOrderRecord).where(
                    WorkOrderRecord.tenant_id == identity.tenant_id,
                    WorkOrderRecord.work_order_id == work_order_id,
                    WorkOrderRecord.status == "READY",
                )
            )
            if current is None:
                raise ApprovalConflict
            for candidate in candidates:
                if is_eligible_field_engineer(
                    session,
                    tenant_id=identity.tenant_id,
                    subject_id=candidate.assignee_subject_id,
                    asset_id=asset_id,
                    site_id=site_id,
                ):
                    eligible.append(candidate)
        options = tuple(
            WorkOrderAssignmentOption(
                assignment_target="ENTERPRISE_FSM",
                provider=candidate.provider,
                candidate_profile_id=candidate.candidate_profile_id,
                profile_version=candidate.profile_version,
                assignee_subject_id=candidate.assignee_subject_id,
                display_name=candidate.display_name,
                matched_skill_codes=candidate.matched_skill_codes,
                service_window_start=candidate.service_window_start.isoformat(),
                service_window_end=candidate.service_window_end.isoformat(),
                travel_minutes=candidate.travel_minutes,
                remaining_work_minutes=candidate.remaining_work_minutes,
                eligibility_code=candidate.eligibility_code,
                source_record_id=candidate.source_record_id,
                as_of=candidate.as_of.isoformat(),
                expires_at=candidate.expires_at.isoformat(),
            )
            for candidate in eligible
        )
        return WorkOrderAssignmentOptions(work_order_id, (local, *options), None)

    def list_customer_notification_options(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        request_id: str,
    ) -> CustomerNotificationOptions:
        """Return only safe, currently eligible contact projections."""

        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident is None:
                raise ApprovalNotVisible
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                ResourceContext(identity.tenant_id, incident_id, incident.asset_id),
                request_id=request_id,
            )
            recipient_subject_id = incident.reporter_subject_id

        options = [_portal_notification_option()]
        if self._notifications is None:
            return CustomerNotificationOptions(
                incident_id,
                tuple(options),
                "external_notification_adapter_disabled",
            )
        try:
            contacts = self._notifications.resolve_contacts(
                tenant_id=identity.tenant_id,
                recipient_subject_id=recipient_subject_id,
            )
        except EnterpriseToolError as exc:
            return CustomerNotificationOptions(incident_id, tuple(options), exc.reason)
        seen: set[tuple[str, str]] = set()
        now = datetime.now(UTC)
        for contact in contacts:
            channel = {
                "EMAIL": "PORTAL_AND_EMAIL",
                "SMS": "PORTAL_AND_SMS",
            }.get(contact.channel)
            if channel is None:
                continue
            try:
                binding = _notification_contact_binding(contact, channel=channel, now=now)
            except CustomerNotificationContactConflict:
                continue
            identity_key = (channel, binding["contact_point_id"])
            if identity_key in seen:
                continue
            seen.add(identity_key)
            options.append(
                CustomerNotificationOption(
                    channel=channel,
                    contact_point_id=binding["contact_point_id"],
                    masked_destination=binding["masked_destination"],
                    source_system=binding["source_system"],
                    source_record_id=binding["source_record_id"],
                    source_version=binding["source_version"],
                    as_of=binding["as_of"],
                    consent_basis_digest=binding["consent_basis_digest"],
                    expires_at=binding["expires_at"],
                )
            )
        return CustomerNotificationOptions(incident_id, tuple(options), None)

    def process_customer_notification_receipt(
        self,
        *,
        provider: str,
        raw_body: bytes,
        signature: str,
    ) -> CustomerNotificationReceiptResult:
        """Verify and monotonically apply one provider receipt without browser identity."""

        adapter = self._notifications
        if adapter is None or provider != adapter.provider_id:
            raise CustomerNotificationReceiptRejected
        try:
            raw_receipt = adapter.verify_receipt(raw_body, signature)
            receipt = _validated_notification_receipt(raw_receipt)
            context = TenantContext(receipt.tenant_id, f"provider-{provider}")
        except (EnterpriseToolError, ValueError, TypeError, KeyError) as exc:
            raise CustomerNotificationReceiptRejected from exc
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            duplicate = session.scalar(
                select(CustomerNotificationReceiptRecord).where(
                    CustomerNotificationReceiptRecord.tenant_id == receipt.tenant_id,
                    CustomerNotificationReceiptRecord.provider == provider,
                    CustomerNotificationReceiptRecord.provider_event_id
                    == receipt.provider_event_id,
                )
            )
            if duplicate is not None:
                if (
                    duplicate.operation_id != receipt.operation_id
                    or duplicate.provider_message_id != receipt.provider_message_id
                    or duplicate.status != receipt.status
                ):
                    raise CustomerNotificationReceiptRejected
                delivery = session.scalar(
                    select(CustomerNotificationDeliveryRecord).where(
                        CustomerNotificationDeliveryRecord.tenant_id == receipt.tenant_id,
                        CustomerNotificationDeliveryRecord.delivery_id == duplicate.delivery_id,
                    )
                )
                if delivery is None:
                    raise CustomerNotificationReceiptRejected
                reconciliation = session.scalar(
                    select(ExecutionReconciliationRecord).where(
                        ExecutionReconciliationRecord.tenant_id == receipt.tenant_id,
                        ExecutionReconciliationRecord.operation_id == receipt.operation_id,
                    )
                )
                return CustomerNotificationReceiptResult(
                    delivery.delivery_id,
                    delivery.status,
                    duplicate.processing_result == "APPLIED",
                    reconciliation.reconciliation_id if reconciliation else None,
                )
            delivery = session.scalar(
                select(CustomerNotificationDeliveryRecord)
                .where(
                    CustomerNotificationDeliveryRecord.tenant_id == receipt.tenant_id,
                    CustomerNotificationDeliveryRecord.operation_id == receipt.operation_id,
                    CustomerNotificationDeliveryRecord.provider == provider,
                )
                .with_for_update()
            )
            if (
                delivery is None
                or delivery.provider_message_id is None
                or delivery.provider_message_id != receipt.provider_message_id
            ):
                raise CustomerNotificationReceiptRejected
            state_changed = False
            processing_result = "IGNORED_NONTERMINAL"
            terminal = {"DELIVERED", "FAILED", "UNSUBSCRIBED"}
            if delivery.status not in terminal and receipt.status in terminal:
                delivery.status = receipt.status
                delivery.reason = receipt.reason
                delivery.terminal_at = receipt.occurred_at
                delivery.last_receipt_at = receipt.occurred_at
                delivery.version += 1
                delivery.updated_at = now
                state_changed = True
                processing_result = "APPLIED"
            elif delivery.status in terminal:
                processing_result = "IGNORED_TERMINAL"
            session.add(
                CustomerNotificationReceiptRecord(
                    receipt_id=f"notification-receipt-{uuid4().hex}",
                    tenant_id=receipt.tenant_id,
                    delivery_id=delivery.delivery_id,
                    provider=provider,
                    provider_event_id=receipt.provider_event_id,
                    operation_id=receipt.operation_id,
                    provider_message_id=receipt.provider_message_id,
                    status=receipt.status,
                    reason=receipt.reason,
                    occurred_at=receipt.occurred_at,
                    signature_digest=receipt.signature_digest,
                    processing_result=processing_result,
                    created_at=now,
                    updated_at=now,
                )
            )
            reconciliation = session.scalar(
                select(ExecutionReconciliationRecord)
                .where(
                    ExecutionReconciliationRecord.tenant_id == receipt.tenant_id,
                    ExecutionReconciliationRecord.operation_id == receipt.operation_id,
                )
                .with_for_update()
            )
            if state_changed:
                proposal = session.scalar(
                    select(ActionProposalRecord).where(
                        ActionProposalRecord.tenant_id == receipt.tenant_id,
                        ActionProposalRecord.proposal_id == delivery.proposal_id,
                    )
                )
                if proposal is None:
                    raise CustomerNotificationReceiptRejected
                succeeded = receipt.status == "DELIVERED"
                proposal.status = "EXECUTED" if succeeded else "EXECUTION_FAILED"
                if reconciliation is not None and reconciliation.status == "PENDING":
                    attempt_number = (
                        int(
                            session.scalar(
                                select(func.count())
                                .select_from(ExecutionAttemptRecord)
                                .where(
                                    ExecutionAttemptRecord.tenant_id == receipt.tenant_id,
                                    ExecutionAttemptRecord.operation_id == receipt.operation_id,
                                )
                            )
                            or 0
                        )
                        + 1
                    )
                    attempt_id = f"attempt-{uuid4().hex}"
                    session.add(
                        ExecutionAttemptRecord(
                            attempt_id=attempt_id,
                            tenant_id=receipt.tenant_id,
                            proposal_id=proposal.proposal_id,
                            approval_id=delivery.approval_id,
                            operation_id=receipt.operation_id,
                            attempt_number=attempt_number,
                            status="SUCCEEDED" if succeeded else "FAILED",
                            reason=receipt.reason,
                            result={
                                "notification_delivery_id": delivery.delivery_id,
                                "provider": provider,
                                "provider_message_id": receipt.provider_message_id,
                                "status": receipt.status,
                            },
                            started_at=now,
                            finished_at=now,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    reconciliation.status = "RESOLVED"
                    reconciliation.outcome = "SUCCEEDED" if succeeded else "FAILED"
                    reconciliation.reason = "provider_receipt_confirmed"
                    reconciliation.external_reference_id = receipt.provider_message_id
                    reconciliation.resolved_execution_attempt_id = attempt_id
                    reconciliation.last_checked_at = now
                    reconciliation.resolved_at = now
                    reconciliation.version += 1
                    reconciliation.updated_at = now
                    session.add(
                        ExecutionReconciliationEventRecord(
                            event_id=f"reconciliation-event-{uuid4().hex}",
                            tenant_id=receipt.tenant_id,
                            reconciliation_id=reconciliation.reconciliation_id,
                            sequence=reconciliation.version,
                            event_type="RESOLVED_BY_RECEIPT",
                            actor_subject_id=context.subject_id,
                            reason=reconciliation.reason,
                            outcome=reconciliation.outcome,
                            created_at=now,
                            updated_at=now,
                        )
                    )
            return CustomerNotificationReceiptResult(
                delivery.delivery_id,
                delivery.status,
                state_changed,
                reconciliation.reconciliation_id if reconciliation else None,
            )

    def process_purchase_request_receipt(
        self,
        *,
        provider: str,
        raw_body: bytes,
        signature: str,
    ) -> PurchaseRequestReceiptResult:
        """Verify, deduplicate and monotonically apply one ERP receipt."""

        adapter = self._procurement
        if adapter is None or provider != adapter.provider_id:
            raise PurchaseRequestReceiptRejected
        try:
            raw_receipt = adapter.verify_receipt(raw_body, signature)
            receipt = _validated_procurement_receipt(raw_receipt)
            context = TenantContext(receipt.tenant_id, f"provider-{provider}")
        except (EnterpriseToolError, ValueError, TypeError, KeyError) as exc:
            raise PurchaseRequestReceiptRejected from exc
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            duplicate = session.scalar(
                select(PurchaseRequestReceiptRecord).where(
                    PurchaseRequestReceiptRecord.tenant_id == receipt.tenant_id,
                    PurchaseRequestReceiptRecord.provider == provider,
                    PurchaseRequestReceiptRecord.provider_event_id
                    == receipt.provider_event_id,
                )
            )
            if duplicate is not None:
                if (
                    duplicate.operation_id != receipt.operation_id
                    or duplicate.external_request_id != receipt.external_request_id
                    or duplicate.status != receipt.status
                ):
                    raise PurchaseRequestReceiptRejected
                purchase = session.scalar(
                    select(PurchaseRequestRecord).where(
                        PurchaseRequestRecord.tenant_id == receipt.tenant_id,
                        PurchaseRequestRecord.purchase_request_id
                        == duplicate.purchase_request_id,
                    )
                )
                if purchase is None:
                    raise PurchaseRequestReceiptRejected
                reconciliation = session.scalar(
                    select(ExecutionReconciliationRecord).where(
                        ExecutionReconciliationRecord.tenant_id == receipt.tenant_id,
                        ExecutionReconciliationRecord.operation_id == receipt.operation_id,
                    )
                )
                return PurchaseRequestReceiptResult(
                    purchase.purchase_request_id,
                    purchase.status,
                    False,
                    reconciliation.reconciliation_id if reconciliation else None,
                )
            purchase = session.scalar(
                select(PurchaseRequestRecord)
                .where(
                    PurchaseRequestRecord.tenant_id == receipt.tenant_id,
                    PurchaseRequestRecord.operation_id == receipt.operation_id,
                    PurchaseRequestRecord.provider == provider,
                    PurchaseRequestRecord.delivery_target == "ENTERPRISE_ERP",
                )
                .with_for_update()
            )
            if (
                purchase is None
                or purchase.external_request_id is not None
                and purchase.external_request_id != receipt.external_request_id
            ):
                raise PurchaseRequestReceiptRejected
            duplicate_after_lock = session.scalar(
                select(PurchaseRequestReceiptRecord).where(
                    PurchaseRequestReceiptRecord.tenant_id == receipt.tenant_id,
                    PurchaseRequestReceiptRecord.provider == provider,
                    PurchaseRequestReceiptRecord.provider_event_id
                    == receipt.provider_event_id,
                )
            )
            if duplicate_after_lock is not None:
                if (
                    duplicate_after_lock.purchase_request_id
                    != purchase.purchase_request_id
                    or duplicate_after_lock.operation_id != receipt.operation_id
                    or duplicate_after_lock.external_request_id
                    != receipt.external_request_id
                    or duplicate_after_lock.status != receipt.status
                ):
                    raise PurchaseRequestReceiptRejected
                reconciliation = session.scalar(
                    select(ExecutionReconciliationRecord).where(
                        ExecutionReconciliationRecord.tenant_id == receipt.tenant_id,
                        ExecutionReconciliationRecord.operation_id
                        == receipt.operation_id,
                    )
                )
                return PurchaseRequestReceiptResult(
                    purchase.purchase_request_id,
                    purchase.status,
                    False,
                    reconciliation.reconciliation_id if reconciliation else None,
                )
            terminal = {"ACCEPTED", "REJECTED", "CANCELLED"}
            state_changed = False
            processing_result = "IGNORED_NONTERMINAL"
            if purchase.status not in terminal and receipt.status in terminal:
                purchase.status = receipt.status
                purchase.external_request_id = receipt.external_request_id
                purchase.reason = receipt.reason
                purchase.terminal_at = receipt.occurred_at
                purchase.last_receipt_at = receipt.occurred_at
                purchase.version += 1
                purchase.updated_at = now
                state_changed = True
                processing_result = "APPLIED"
            elif (
                purchase.status not in terminal
                and purchase.external_request_id is None
            ):
                purchase.external_request_id = receipt.external_request_id
                purchase.last_receipt_at = receipt.occurred_at
                purchase.version += 1
                purchase.updated_at = now
                processing_result = "BOUND_EXTERNAL_ID"
            elif purchase.status in terminal:
                processing_result = "IGNORED_TERMINAL"
            session.add(
                PurchaseRequestReceiptRecord(
                    receipt_id=f"purchase-receipt-{uuid4().hex}",
                    tenant_id=receipt.tenant_id,
                    purchase_request_id=purchase.purchase_request_id,
                    provider=provider,
                    provider_event_id=receipt.provider_event_id,
                    operation_id=receipt.operation_id,
                    external_request_id=receipt.external_request_id,
                    status=receipt.status,
                    reason=receipt.reason,
                    occurred_at=receipt.occurred_at,
                    signature_digest=receipt.signature_digest,
                    processing_result=processing_result,
                    created_at=now,
                    updated_at=now,
                )
            )
            reconciliation = session.scalar(
                select(ExecutionReconciliationRecord)
                .where(
                    ExecutionReconciliationRecord.tenant_id == receipt.tenant_id,
                    ExecutionReconciliationRecord.operation_id == receipt.operation_id,
                )
                .with_for_update()
            )
            if state_changed:
                proposal = session.scalar(
                    select(ActionProposalRecord).where(
                        ActionProposalRecord.tenant_id == receipt.tenant_id,
                        ActionProposalRecord.proposal_id == purchase.proposal_id,
                    )
                )
                approval = session.scalar(
                    select(ApprovalRequestRecord).where(
                        ApprovalRequestRecord.tenant_id == receipt.tenant_id,
                        ApprovalRequestRecord.proposal_id == purchase.proposal_id,
                    )
                )
                if proposal is None or approval is None:
                    raise PurchaseRequestReceiptRejected
                succeeded = receipt.status == "ACCEPTED"
                proposal.status = "EXECUTED" if succeeded else "EXECUTION_FAILED"
                attempt_number = (
                    int(
                        session.scalar(
                            select(func.count())
                            .select_from(ExecutionAttemptRecord)
                            .where(
                                ExecutionAttemptRecord.tenant_id == receipt.tenant_id,
                                ExecutionAttemptRecord.operation_id == receipt.operation_id,
                            )
                        )
                        or 0
                    )
                    + 1
                )
                attempt_id = f"attempt-{uuid4().hex}"
                session.add(
                    _procurement_execution_attempt(
                        receipt.tenant_id,
                        proposal,
                        approval_id=approval.approval_id,
                        attempt_id=attempt_id,
                        attempt_number=attempt_number,
                        status="SUCCEEDED" if succeeded else "FAILED",
                        reason=receipt.reason,
                        purchase=purchase,
                        now=now,
                        idempotency_key=None,
                    )
                )
                if reconciliation is not None and reconciliation.status == "PENDING":
                    reconciliation.status = "RESOLVED"
                    reconciliation.outcome = "SUCCEEDED" if succeeded else "FAILED"
                    reconciliation.reason = "provider_receipt_confirmed"
                    reconciliation.external_reference_id = receipt.external_request_id
                    reconciliation.resolved_execution_attempt_id = attempt_id
                    reconciliation.last_checked_at = now
                    reconciliation.resolved_at = now
                    reconciliation.version += 1
                    reconciliation.updated_at = now
                    session.add(
                        ExecutionReconciliationEventRecord(
                            event_id=f"reconciliation-event-{uuid4().hex}",
                            tenant_id=receipt.tenant_id,
                            reconciliation_id=reconciliation.reconciliation_id,
                            sequence=reconciliation.version,
                            event_type="RESOLVED_BY_RECEIPT",
                            actor_subject_id=context.subject_id,
                            reason=reconciliation.reason,
                            outcome=reconciliation.outcome,
                            created_at=now,
                            updated_at=now,
                        )
                    )
            return PurchaseRequestReceiptResult(
                purchase.purchase_request_id,
                purchase.status,
                state_changed,
                reconciliation.reconciliation_id if reconciliation else None,
            )

    def process_procurement_fulfillment_receipt(
        self,
        *,
        provider: str,
        raw_body: bytes,
        signature: str,
    ) -> ProcurementFulfillmentReceiptResult:
        """Authenticate and monotonically apply one ERP fulfillment event."""

        adapter = self._procurement
        if adapter is None or provider != adapter.provider_id:
            raise ProcurementFulfillmentReceiptRejected
        try:
            raw_receipt = adapter.verify_fulfillment_receipt(raw_body, signature)
            receipt = _validated_procurement_fulfillment_receipt(raw_receipt)
            context = TenantContext(receipt.tenant_id, f"provider-{provider}")
        except (
            AttributeError,
            EnterpriseToolError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            raise ProcurementFulfillmentReceiptRejected from exc
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            duplicate = session.scalar(
                select(PurchaseFulfillmentReceiptRecord).where(
                    PurchaseFulfillmentReceiptRecord.tenant_id == receipt.tenant_id,
                    PurchaseFulfillmentReceiptRecord.provider == provider,
                    PurchaseFulfillmentReceiptRecord.provider_event_id
                    == receipt.provider_event_id,
                )
            )
            if duplicate is not None:
                if not _same_procurement_fulfillment_receipt(duplicate, receipt):
                    raise ProcurementFulfillmentReceiptRejected
                fulfillment = session.scalar(
                    select(PurchaseFulfillmentRecord).where(
                        PurchaseFulfillmentRecord.tenant_id == receipt.tenant_id,
                        PurchaseFulfillmentRecord.fulfillment_id
                        == duplicate.fulfillment_id,
                    )
                )
                if fulfillment is None:
                    raise ProcurementFulfillmentReceiptRejected
                return ProcurementFulfillmentReceiptResult(
                    fulfillment.purchase_request_id,
                    fulfillment.fulfillment_id,
                    fulfillment.status,
                    fulfillment.received_quantity,
                    False,
                )

            purchase = session.scalar(
                select(PurchaseRequestRecord)
                .where(
                    PurchaseRequestRecord.tenant_id == receipt.tenant_id,
                    PurchaseRequestRecord.operation_id == receipt.operation_id,
                    PurchaseRequestRecord.provider == provider,
                    PurchaseRequestRecord.delivery_target == "ENTERPRISE_ERP",
                )
                .with_for_update()
            )
            if (
                purchase is None
                or purchase.status != "ACCEPTED"
                or purchase.external_request_id is None
                or purchase.external_request_id != receipt.external_request_id
            ):
                raise ProcurementFulfillmentReceiptRejected

            fulfillment = session.scalar(
                select(PurchaseFulfillmentRecord)
                .where(
                    PurchaseFulfillmentRecord.tenant_id == receipt.tenant_id,
                    PurchaseFulfillmentRecord.purchase_request_id
                    == purchase.purchase_request_id,
                )
                .with_for_update()
            )
            duplicate_after_lock = session.scalar(
                select(PurchaseFulfillmentReceiptRecord).where(
                    PurchaseFulfillmentReceiptRecord.tenant_id == receipt.tenant_id,
                    PurchaseFulfillmentReceiptRecord.provider == provider,
                    PurchaseFulfillmentReceiptRecord.provider_event_id
                    == receipt.provider_event_id,
                )
            )
            if duplicate_after_lock is not None:
                if (
                    not _same_procurement_fulfillment_receipt(
                        duplicate_after_lock,
                        receipt,
                    )
                    or fulfillment is None
                    or duplicate_after_lock.fulfillment_id
                    != fulfillment.fulfillment_id
                ):
                    raise ProcurementFulfillmentReceiptRejected
                return ProcurementFulfillmentReceiptResult(
                    purchase.purchase_request_id,
                    fulfillment.fulfillment_id,
                    fulfillment.status,
                    fulfillment.received_quantity,
                    False,
                )

            _validate_procurement_fulfillment_quantity(
                receipt.status,
                receipt.received_quantity,
                purchase.quantity,
            )
            if fulfillment is None:
                fulfillment = PurchaseFulfillmentRecord(
                    fulfillment_id=f"purchase-fulfillment-{uuid4().hex}",
                    tenant_id=receipt.tenant_id,
                    purchase_request_id=purchase.purchase_request_id,
                    provider=provider,
                    operation_id=receipt.operation_id,
                    external_request_id=receipt.external_request_id,
                    external_order_id=receipt.external_order_id,
                    status=receipt.status,
                    requested_quantity=purchase.quantity,
                    received_quantity=receipt.received_quantity,
                    expected_delivery_at=receipt.expected_delivery_at,
                    last_event_occurred_at=receipt.occurred_at,
                    last_receipt_at=now,
                    reason=receipt.reason,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
                session.add(fulfillment)
            else:
                if (
                    fulfillment.provider != provider
                    or fulfillment.operation_id != receipt.operation_id
                    or fulfillment.external_request_id
                    != receipt.external_request_id
                    or fulfillment.external_order_id != receipt.external_order_id
                    or fulfillment.requested_quantity != purchase.quantity
                ):
                    raise ProcurementFulfillmentReceiptRejected
                _validate_procurement_fulfillment_transition(fulfillment, receipt)
                fulfillment.status = receipt.status
                fulfillment.received_quantity = receipt.received_quantity
                if receipt.expected_delivery_at is not None:
                    fulfillment.expected_delivery_at = receipt.expected_delivery_at
                fulfillment.last_event_occurred_at = receipt.occurred_at
                fulfillment.last_receipt_at = now
                if receipt.reason is not None:
                    fulfillment.reason = receipt.reason
                fulfillment.version += 1
                fulfillment.updated_at = now

            session.add(
                PurchaseFulfillmentReceiptRecord(
                    receipt_id=f"purchase-fulfillment-receipt-{uuid4().hex}",
                    tenant_id=receipt.tenant_id,
                    fulfillment_id=fulfillment.fulfillment_id,
                    purchase_request_id=purchase.purchase_request_id,
                    provider=provider,
                    provider_event_id=receipt.provider_event_id,
                    operation_id=receipt.operation_id,
                    external_request_id=receipt.external_request_id,
                    external_order_id=receipt.external_order_id,
                    status=receipt.status,
                    received_quantity=receipt.received_quantity,
                    expected_delivery_at=receipt.expected_delivery_at,
                    occurred_at=receipt.occurred_at,
                    reason=receipt.reason,
                    signature_digest=receipt.signature_digest,
                    processing_result="APPLIED",
                    created_at=now,
                    updated_at=now,
                )
            )
            return ProcurementFulfillmentReceiptResult(
                purchase.purchase_request_id,
                fulfillment.fulfillment_id,
                fulfillment.status,
                fulfillment.received_quantity,
                True,
            )

    def process_refund_request_receipt(
        self,
        *,
        provider: str,
        raw_body: bytes,
        signature: str,
    ) -> RefundRequestReceiptResult:
        """Authenticate, deduplicate and monotonically apply one finance receipt."""

        adapter = self._refunds
        if adapter is None or provider != adapter.provider_id:
            raise RefundRequestReceiptRejected
        try:
            raw_receipt = adapter.verify_receipt(raw_body, signature)
            receipt = _validated_refund_receipt(raw_receipt)
            context = TenantContext(receipt.tenant_id, f"provider-{provider}")
        except (EnterpriseToolError, ValueError, TypeError, KeyError) as exc:
            raise RefundRequestReceiptRejected from exc
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            duplicate = session.scalar(
                select(RefundRequestReceiptRecord).where(
                    RefundRequestReceiptRecord.tenant_id == receipt.tenant_id,
                    RefundRequestReceiptRecord.provider == provider,
                    RefundRequestReceiptRecord.provider_event_id
                    == receipt.provider_event_id,
                )
            )
            if duplicate is not None:
                if not _same_refund_receipt(duplicate, receipt):
                    raise RefundRequestReceiptRejected
                refund = session.scalar(
                    select(RefundRequestRecord).where(
                        RefundRequestRecord.tenant_id == receipt.tenant_id,
                        RefundRequestRecord.refund_request_id
                        == duplicate.refund_request_id,
                    )
                )
                if refund is None:
                    raise RefundRequestReceiptRejected
                reconciliation = session.scalar(
                    select(ExecutionReconciliationRecord).where(
                        ExecutionReconciliationRecord.tenant_id == receipt.tenant_id,
                        ExecutionReconciliationRecord.operation_id == receipt.operation_id,
                    )
                )
                return RefundRequestReceiptResult(
                    refund.refund_request_id,
                    refund.status,
                    refund.payment_status,
                    False,
                    reconciliation.reconciliation_id if reconciliation else None,
                )
            refund = session.scalar(
                select(RefundRequestRecord)
                .where(
                    RefundRequestRecord.tenant_id == receipt.tenant_id,
                    RefundRequestRecord.operation_id == receipt.operation_id,
                    RefundRequestRecord.provider == provider,
                    RefundRequestRecord.delivery_target == "ENTERPRISE_FINANCE",
                )
                .with_for_update()
            )
            if (
                refund is None
                or refund.external_request_id is not None
                and refund.external_request_id != receipt.external_request_id
            ):
                raise RefundRequestReceiptRejected
            duplicate_after_lock = session.scalar(
                select(RefundRequestReceiptRecord).where(
                    RefundRequestReceiptRecord.tenant_id == receipt.tenant_id,
                    RefundRequestReceiptRecord.provider == provider,
                    RefundRequestReceiptRecord.provider_event_id
                    == receipt.provider_event_id,
                )
            )
            if duplicate_after_lock is not None:
                if (
                    duplicate_after_lock.refund_request_id != refund.refund_request_id
                    or not _same_refund_receipt(duplicate_after_lock, receipt)
                ):
                    raise RefundRequestReceiptRejected
                reconciliation = session.scalar(
                    select(ExecutionReconciliationRecord).where(
                        ExecutionReconciliationRecord.tenant_id == receipt.tenant_id,
                        ExecutionReconciliationRecord.operation_id == receipt.operation_id,
                    )
                )
                return RefundRequestReceiptResult(
                    refund.refund_request_id,
                    refund.status,
                    refund.payment_status,
                    False,
                    reconciliation.reconciliation_id if reconciliation else None,
                )

            request_terminal = {"ACCEPTED", "REJECTED", "CANCELLED"}
            request_changed = False
            payment_changed = False
            processing_result = "IGNORED_MONOTONIC_STATE"
            if refund.external_request_id is None:
                refund.external_request_id = receipt.external_request_id
                processing_result = "BOUND_EXTERNAL_ID"
            if refund.status not in request_terminal and receipt.status in request_terminal:
                refund.status = receipt.status
                refund.provider_reason = receipt.reason
                refund.terminal_at = receipt.occurred_at
                request_changed = True
                processing_result = "APPLIED"
            if (
                refund.status == receipt.status == "ACCEPTED"
                and _refund_payment_transition_allowed(
                    refund.payment_status, receipt.payment_status
                )
            ):
                refund.payment_status = receipt.payment_status
                refund.provider_reason = receipt.reason or refund.provider_reason
                payment_changed = True
                processing_result = "APPLIED"
            elif (
                request_changed
                and refund.status in {"REJECTED", "CANCELLED"}
                and refund.payment_status != "NOT_STARTED"
            ):
                refund.payment_status = "NOT_STARTED"
                payment_changed = True
            state_changed = request_changed or payment_changed
            refund.last_receipt_at = receipt.occurred_at
            if state_changed or processing_result == "BOUND_EXTERNAL_ID":
                refund.version += 1
                refund.updated_at = now
            session.add(
                RefundRequestReceiptRecord(
                    receipt_id=f"refund-receipt-{uuid4().hex}",
                    tenant_id=receipt.tenant_id,
                    refund_request_id=refund.refund_request_id,
                    provider=provider,
                    provider_event_id=receipt.provider_event_id,
                    operation_id=receipt.operation_id,
                    external_request_id=receipt.external_request_id,
                    status=receipt.status,
                    payment_status=receipt.payment_status,
                    reason=receipt.reason,
                    occurred_at=receipt.occurred_at,
                    signature_digest=receipt.signature_digest,
                    processing_result=processing_result,
                    created_at=now,
                    updated_at=now,
                )
            )
            reconciliation = session.scalar(
                select(ExecutionReconciliationRecord)
                .where(
                    ExecutionReconciliationRecord.tenant_id == receipt.tenant_id,
                    ExecutionReconciliationRecord.operation_id == receipt.operation_id,
                )
                .with_for_update()
            )
            if request_changed:
                proposal = session.scalar(
                    select(ActionProposalRecord).where(
                        ActionProposalRecord.tenant_id == receipt.tenant_id,
                        ActionProposalRecord.proposal_id == refund.proposal_id,
                    )
                )
                approval = session.scalar(
                    select(ApprovalRequestRecord).where(
                        ApprovalRequestRecord.tenant_id == receipt.tenant_id,
                        ApprovalRequestRecord.proposal_id == refund.proposal_id,
                    )
                )
                if proposal is None or approval is None:
                    raise RefundRequestReceiptRejected
                succeeded = receipt.status == "ACCEPTED"
                proposal.status = "EXECUTED" if succeeded else "EXECUTION_FAILED"
                attempt_number = (
                    int(
                        session.scalar(
                            select(func.count())
                            .select_from(ExecutionAttemptRecord)
                            .where(
                                ExecutionAttemptRecord.tenant_id == receipt.tenant_id,
                                ExecutionAttemptRecord.operation_id == receipt.operation_id,
                            )
                        )
                        or 0
                    )
                    + 1
                )
                attempt_id = f"attempt-{uuid4().hex}"
                session.add(
                    _refund_execution_attempt(
                        receipt.tenant_id,
                        proposal,
                        approval_id=approval.approval_id,
                        attempt_id=attempt_id,
                        attempt_number=attempt_number,
                        status="SUCCEEDED" if succeeded else "FAILED",
                        reason=receipt.reason,
                        refund=refund,
                        now=now,
                        idempotency_key=None,
                    )
                )
                if reconciliation is not None and reconciliation.status == "PENDING":
                    reconciliation.status = "RESOLVED"
                    reconciliation.outcome = "SUCCEEDED" if succeeded else "FAILED"
                    reconciliation.reason = "provider_receipt_confirmed"
                    reconciliation.external_reference_id = receipt.external_request_id
                    reconciliation.resolved_execution_attempt_id = attempt_id
                    reconciliation.last_checked_at = now
                    reconciliation.resolved_at = now
                    reconciliation.version += 1
                    reconciliation.updated_at = now
                    session.add(
                        ExecutionReconciliationEventRecord(
                            event_id=f"reconciliation-event-{uuid4().hex}",
                            tenant_id=receipt.tenant_id,
                            reconciliation_id=reconciliation.reconciliation_id,
                            sequence=reconciliation.version,
                            event_type="RESOLVED_BY_RECEIPT",
                            actor_subject_id=context.subject_id,
                            reason=reconciliation.reason,
                            outcome=reconciliation.outcome,
                            created_at=now,
                            updated_at=now,
                        )
                    )
            return RefundRequestReceiptResult(
                refund.refund_request_id,
                refund.status,
                refund.payment_status,
                state_changed,
                reconciliation.reconciliation_id if reconciliation else None,
            )

    def process_fsm_assignment_receipt(
        self,
        *,
        provider: str,
        raw_body: bytes,
        signature: str,
    ) -> FsmAssignmentReceiptResult:
        """Authenticate, deduplicate and monotonically converge one FSM receipt."""

        adapter = self._fsm_assignments
        if adapter is None or provider != adapter.provider_id:
            raise FsmAssignmentReceiptRejected
        try:
            raw_receipt = adapter.verify_receipt(raw_body, signature)
            receipt = _validated_fsm_assignment_receipt(raw_receipt)
            context = TenantContext(receipt.tenant_id, f"provider-{provider}")
        except (EnterpriseToolError, ValueError, TypeError, KeyError) as exc:
            raise FsmAssignmentReceiptRejected from exc
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            duplicate = session.scalar(
                select(FsmAssignmentReceiptRecord).where(
                    FsmAssignmentReceiptRecord.tenant_id == receipt.tenant_id,
                    FsmAssignmentReceiptRecord.provider == provider,
                    FsmAssignmentReceiptRecord.provider_event_id
                    == receipt.provider_event_id,
                )
            )
            if duplicate is not None:
                if not _same_fsm_assignment_receipt(duplicate, receipt):
                    raise FsmAssignmentReceiptRejected
                reconciliation = session.scalar(
                    select(ExecutionReconciliationRecord).where(
                        ExecutionReconciliationRecord.tenant_id == receipt.tenant_id,
                        ExecutionReconciliationRecord.operation_id == receipt.operation_id,
                    )
                )
                return FsmAssignmentReceiptResult(
                    duplicate.delivery_id,
                    duplicate.status,
                    False,
                    reconciliation.reconciliation_id if reconciliation else None,
                )
            delivery = session.scalar(
                select(FsmAssignmentDeliveryRecord)
                .where(
                    FsmAssignmentDeliveryRecord.tenant_id == receipt.tenant_id,
                    FsmAssignmentDeliveryRecord.operation_id == receipt.operation_id,
                    FsmAssignmentDeliveryRecord.provider == provider,
                )
                .with_for_update()
            )
            if (
                delivery is None
                or delivery.candidate_profile_id != receipt.candidate_profile_id
                or delivery.assignee_subject_id != receipt.assignee_subject_id
                or delivery.external_assignment_id is not None
                and delivery.external_assignment_id != receipt.external_assignment_id
            ):
                raise FsmAssignmentReceiptRejected
            if delivery.status in {"ACCEPTED", "REJECTED", "CANCELLED"} and (
                delivery.status != receipt.status
            ):
                raise FsmAssignmentReceiptRejected
            proposal = session.scalar(
                select(ActionProposalRecord).where(
                    ActionProposalRecord.tenant_id == receipt.tenant_id,
                    ActionProposalRecord.proposal_id == delivery.proposal_id,
                )
            )
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == receipt.tenant_id,
                    IncidentRecord.incident_id == delivery.incident_id,
                )
            )
            if proposal is None or incident is None:
                raise FsmAssignmentReceiptRejected
            reconciliation = session.scalar(
                select(ExecutionReconciliationRecord)
                .where(
                    ExecutionReconciliationRecord.tenant_id == receipt.tenant_id,
                    ExecutionReconciliationRecord.operation_id == receipt.operation_id,
                )
                .with_for_update()
            )
            system_identity = IdentityContext(
                subject_id=f"provider-{provider}",
                oidc_subject=f"provider-{provider}",
                tenant_id=receipt.tenant_id,
                roles=frozenset({Role.AFTER_SALES_ENGINEER}),
                asset_ids=frozenset({incident.asset_id}),
                site_ids=frozenset(),
                issued_at=now,
                expires_at=now + timedelta(minutes=5),
            )
            state_changed = False
            processing_result = "RECORDED_PENDING"
            if receipt.status in {"ACCEPTED", "REJECTED", "CANCELLED"}:
                try:
                    state_changed = self._apply_fsm_assignment_terminal_result(
                        session,
                        system_identity,
                        proposal,
                        delivery,
                        FsmAssignmentDeliveryResult(
                            status=receipt.status,
                            external_assignment_id=receipt.external_assignment_id,
                            reason=receipt.reason,
                            occurred_at=receipt.occurred_at,
                        ),
                        reconciliation=reconciliation,
                        now=now,
                    )
                except ExecutionReconciliationConflict as exc:
                    raise FsmAssignmentReceiptRejected from exc
                processing_result = "APPLIED" if state_changed else "IGNORED_MONOTONIC_STATE"
            elif delivery.external_assignment_id is None:
                delivery.external_assignment_id = receipt.external_assignment_id
                delivery.version += 1
                delivery.updated_at = now
                processing_result = "BOUND_EXTERNAL_ID"
            if (
                delivery.last_receipt_at is None
                or _utc(delivery.last_receipt_at) < receipt.occurred_at
            ):
                delivery.last_receipt_at = receipt.occurred_at
            session.add(
                FsmAssignmentReceiptRecord(
                    receipt_id=f"fsm-receipt-{uuid4().hex}",
                    tenant_id=receipt.tenant_id,
                    delivery_id=delivery.delivery_id,
                    provider=provider,
                    provider_event_id=receipt.provider_event_id,
                    operation_id=receipt.operation_id,
                    candidate_profile_id=receipt.candidate_profile_id,
                    assignee_subject_id=receipt.assignee_subject_id,
                    external_assignment_id=receipt.external_assignment_id,
                    status=receipt.status,
                    reason=receipt.reason,
                    occurred_at=receipt.occurred_at,
                    signature_digest=receipt.signature_digest,
                    processing_result=processing_result,
                    created_at=now,
                    updated_at=now,
                )
            )
            return FsmAssignmentReceiptResult(
                delivery.delivery_id,
                delivery.status,
                state_changed,
                reconciliation.reconciliation_id if reconciliation else None,
            )

    def propose_customer_notification(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        incident_version: int,
        template_id: str,
        headline: str,
        detail: str,
        next_step: str,
        request_id: str,
        channel: str = "PORTAL",
        contact_point_id: str | None = None,
    ) -> CustomerNotificationProposal:
        """Create a reviewable T2 proposal without publishing customer content."""

        now = datetime.now(UTC)
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident is None:
                raise ApprovalNotVisible
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                ResourceContext(identity.tenant_id, incident_id, incident.asset_id),
                request_id=request_id,
            )
            if incident.version != incident_version or incident.status in {"CLOSED", "CANCELLED"}:
                raise ApprovalConflict
            if channel == "PORTAL":
                if contact_point_id is not None:
                    raise CustomerNotificationContactConflict("portal_contact_not_allowed")
                contact_binding = None
                tool_version = "1.0.0"
            elif channel in {"PORTAL_AND_EMAIL", "PORTAL_AND_SMS"}:
                if not contact_point_id:
                    raise CustomerNotificationContactConflict("contact_point_required")
                contact_binding = self._resolve_customer_notification_binding(
                    tenant_id=identity.tenant_id,
                    recipient_subject_id=incident.reporter_subject_id,
                    contact_point_id=contact_point_id,
                    channel=channel,
                    now=now,
                )
                tool_version = "1.1.0"
            else:
                raise CustomerNotificationContactConflict("unsupported_notification_channel")
            parameters: dict[str, Any] = {
                "incident_id": incident_id,
                "asset_id": incident.asset_id,
                "recipient_scope": "INCIDENT_REPORTER",
                "recipient_subject_id": incident.reporter_subject_id,
                "channel": channel,
                "locale": "zh-CN",
                "template_id": template_id,
                "content": {
                    "headline": headline,
                    "detail": detail,
                    "next_step": next_step,
                },
            }
            if contact_binding is not None:
                parameters["contact_binding"] = contact_binding
            definition = self._registry.get("customer.notify", tool_version)
            definition.validate(parameters)
            proposal_id = f"proposal-{uuid4().hex}"
            approval_id = f"approval-{uuid4().hex}"
            operation_id = f"operation-{uuid4().hex}"
            session.add(
                ActionProposalRecord(
                    proposal_id=proposal_id,
                    tenant_id=identity.tenant_id,
                    incident_id=incident_id,
                    initiator_subject_id=identity.subject_id,
                    tool_id=definition.tool_id,
                    tool_version=definition.version,
                    risk_tier=definition.risk_tier,
                    parameters=parameters,
                    parameters_hash=_digest(parameters),
                    resource_version=incident_version,
                    operation_id=operation_id,
                    status="PENDING_APPROVAL",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ApprovalRequestRecord(
                    approval_id=approval_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    policy_version="m2-t2-v1",
                    status="PENDING",
                    expires_at=now + T2_APPROVAL_TTL,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            return CustomerNotificationProposal(
                proposal_id,
                approval_id,
                parameters,
                operation_id,
            )

    def _resolve_customer_notification_binding(
        self,
        *,
        tenant_id: str,
        recipient_subject_id: str,
        contact_point_id: str,
        channel: str,
        now: datetime,
    ) -> dict[str, Any]:
        _, binding = self._resolve_customer_notification_contact(
            tenant_id=tenant_id,
            recipient_subject_id=recipient_subject_id,
            contact_point_id=contact_point_id,
            channel=channel,
            now=now,
        )
        return binding

    def _resolve_customer_notification_contact(
        self,
        *,
        tenant_id: str,
        recipient_subject_id: str,
        contact_point_id: str,
        channel: str,
        now: datetime,
    ) -> tuple[NotificationContact, dict[str, Any]]:
        if self._notifications is None:
            raise CustomerNotificationAuthorityUnavailable
        try:
            contacts = self._notifications.resolve_contacts(
                tenant_id=tenant_id,
                recipient_subject_id=recipient_subject_id,
            )
        except EnterpriseToolError as exc:
            raise CustomerNotificationAuthorityUnavailable(exc.reason) from exc
        for contact in contacts:
            if contact.contact_point_id == contact_point_id:
                return contact, _notification_contact_binding(contact, channel=channel, now=now)
        raise CustomerNotificationContactConflict("contact_not_eligible")

    def _resolve_procurement_profile(
        self,
        *,
        tenant_id: str,
        now: datetime,
    ) -> ProcurementProfile:
        adapter = self._procurement
        if adapter is None:
            raise PurchaseDeliveryProfileUnavailable(
                "enterprise_procurement_adapter_disabled"
            )
        try:
            raw = adapter.resolve_profile(tenant_id=tenant_id)
        except EnterpriseToolError as exc:
            raise PurchaseDeliveryProfileUnavailable(exc.reason) from exc
        except (OSError, TimeoutError) as exc:
            raise PurchaseDeliveryProfileUnavailable(
                "enterprise_procurement_profile_unavailable"
            ) from exc
        if raw is None:
            raise PurchaseDeliveryProfileUnavailable(
                "enterprise_procurement_profile_unavailable"
            )
        values = (
            raw.provider,
            raw.profile_id,
            raw.profile_version,
            raw.display_name,
        )
        if (
            raw.provider != adapter.provider_id
            or any(not isinstance(value, str) or not value or len(value) > 128 for value in values)
            or any(not value.isprintable() for value in values)
            or not all(
                char.isascii() and (char.isalnum() or char in "._-")
                for char in raw.provider
            )
            or not isinstance(raw.as_of, datetime)
            or not isinstance(raw.expires_at, datetime)
            or raw.as_of.tzinfo is None
            or raw.as_of.utcoffset() is None
            or raw.expires_at.tzinfo is None
            or raw.expires_at.utcoffset() is None
        ):
            raise PurchaseDeliveryProfileUnavailable(
                "enterprise_procurement_profile_invalid"
            )
        as_of = raw.as_of.astimezone(UTC)
        expires_at = raw.expires_at.astimezone(UTC)
        if expires_at <= now:
            raise PurchaseDeliveryProfileUnavailable(
                "enterprise_procurement_profile_expired"
            )
        if as_of > now + timedelta(minutes=1) or as_of >= expires_at:
            raise PurchaseDeliveryProfileUnavailable(
                "enterprise_procurement_profile_invalid"
            )
        return ProcurementProfile(
            provider=raw.provider,
            profile_id=raw.profile_id,
            profile_version=raw.profile_version,
            display_name=raw.display_name,
            as_of=as_of,
            expires_at=expires_at,
        )

    def _resolve_fsm_assignment_candidates(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
        asset_id: str,
        site_id: str,
        service_window_start: datetime | None,
        service_window_end: datetime | None,
        now: datetime,
    ) -> tuple[FsmAssignmentCandidate, ...]:
        adapter = self._fsm_assignments
        if adapter is None:
            raise FsmAssignmentProfileUnavailable("fsm_assignment_adapter_disabled")
        try:
            raw_candidates = adapter.list_candidates(
                tenant_id=tenant_id,
                work_order_id=work_order_id,
                asset_id=asset_id,
                site_id=site_id,
                service_window_start=service_window_start,
                service_window_end=service_window_end,
            )
        except EnterpriseToolError as exc:
            raise FsmAssignmentProfileUnavailable(exc.reason) from exc
        except (OSError, TimeoutError) as exc:
            raise FsmAssignmentProfileUnavailable(
                "fsm_assignment_candidates_unavailable"
            ) from exc
        if not isinstance(raw_candidates, tuple):
            raise FsmAssignmentProfileUnavailable("fsm_assignment_candidates_invalid")

        candidates: list[FsmAssignmentCandidate] = []
        seen_profiles: set[str] = set()
        for raw in raw_candidates:
            candidate = _validated_fsm_assignment_candidate(
                raw,
                provider_id=adapter.provider_id,
                now=now,
            )
            if candidate.candidate_profile_id in seen_profiles:
                raise FsmAssignmentProfileUnavailable(
                    "fsm_assignment_candidates_invalid"
                )
            seen_profiles.add(candidate.candidate_profile_id)
            candidates.append(candidate)
        return tuple(candidates)

    def _resolve_refund_profile(
        self,
        *,
        tenant_id: str,
        customer_subject_id: str,
        now: datetime,
    ) -> RefundProfile:
        adapter = self._refunds
        if adapter is None:
            raise RefundDeliveryProfileUnavailable("enterprise_refund_adapter_disabled")
        try:
            raw = adapter.resolve_profile(
                tenant_id=tenant_id,
                customer_subject_id=customer_subject_id,
            )
        except EnterpriseToolError as exc:
            raise RefundDeliveryProfileUnavailable(exc.reason) from exc
        except (OSError, TimeoutError) as exc:
            raise RefundDeliveryProfileUnavailable(
                "enterprise_refund_profile_unavailable"
            ) from exc
        if raw is None:
            raise RefundDeliveryProfileUnavailable("enterprise_refund_profile_unavailable")
        provider = getattr(raw, "provider", None)
        profile_id = getattr(raw, "profile_id", None)
        profile_version = getattr(raw, "profile_version", None)
        display_name = getattr(raw, "display_name", None)
        raw_as_of = getattr(raw, "as_of", None)
        raw_expires_at = getattr(raw, "expires_at", None)
        values = (
            provider,
            profile_id,
            profile_version,
            display_name,
        )
        if any(
            not isinstance(value, str) or not value or len(value) > 128
            for value in values
        ):
            raise RefundDeliveryProfileUnavailable("enterprise_refund_profile_invalid")
        provider = cast(str, provider)
        profile_id = cast(str, profile_id)
        profile_version = cast(str, profile_version)
        display_name = cast(str, display_name)
        values = (provider, profile_id, profile_version, display_name)
        if (
            provider != adapter.provider_id
            or any(not value.isprintable() for value in values)
            or not all(
                char.isascii() and (char.isalnum() or char in "._-")
                for char in provider
            )
            or not isinstance(raw_as_of, datetime)
            or not isinstance(raw_expires_at, datetime)
            or raw_as_of.tzinfo is None
            or raw_as_of.utcoffset() is None
            or raw_expires_at.tzinfo is None
            or raw_expires_at.utcoffset() is None
        ):
            raise RefundDeliveryProfileUnavailable("enterprise_refund_profile_invalid")
        raw_as_of = cast(datetime, raw_as_of)
        raw_expires_at = cast(datetime, raw_expires_at)
        as_of = raw_as_of.astimezone(UTC)
        expires_at = raw_expires_at.astimezone(UTC)
        if expires_at <= now:
            raise RefundDeliveryProfileUnavailable("enterprise_refund_profile_expired")
        if as_of > now + timedelta(minutes=1) or as_of >= expires_at:
            raise RefundDeliveryProfileUnavailable("enterprise_refund_profile_invalid")
        return RefundProfile(
            provider=provider,
            profile_id=profile_id,
            profile_version=profile_version,
            display_name=display_name,
            as_of=as_of,
            expires_at=expires_at,
        )

    def propose_purchase_request(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        incident_version: int,
        part_number: str,
        quantity: int,
        estimated_unit_cost_minor: int,
        currency: str,
        cost_center: str,
        justification: str,
        request_id: str,
        delivery_target: str = "INTERNAL",
        profile_id: str | None = None,
    ) -> PurchaseRequestProposal:
        """Create a purchase proposal only when authoritative inventory is insufficient."""

        now = datetime.now(UTC)
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident is None:
                raise ApprovalNotVisible
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                ResourceContext(identity.tenant_id, incident_id, incident.asset_id),
                request_id=request_id,
            )
            if incident.version != incident_version or incident.status != "DIAGNOSED":
                raise ApprovalConflict
            try:
                availability = self._parts.check_availability(
                    part_number=part_number,
                    quantity=quantity,
                )
                snapshot = _purchase_inventory_snapshot(
                    availability,
                    expected_part_number=part_number,
                    requested_quantity=quantity,
                    now=now,
                )
            except EnterpriseToolError as exc:
                raise PurchaseDependencyUnavailable(exc.reason) from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise PurchaseDependencyUnavailable("inventory_fact_invalid") from exc
            if bool(availability["available"]):
                raise PurchaseNotRequired
            total_cost = estimated_unit_cost_minor * quantity
            parameters: dict[str, Any] = {
                "incident_id": incident_id,
                "asset_id": incident.asset_id,
                "part_number": part_number,
                "quantity": quantity,
                "estimated_unit_cost_minor": estimated_unit_cost_minor,
                "estimated_total_cost_minor": total_cost,
                "currency": currency,
                "cost_center": cost_center,
                "justification": justification,
                "inventory_snapshot": snapshot,
            }
            if delivery_target == "INTERNAL":
                if profile_id is not None:
                    raise PurchaseDeliveryProfileConflict(
                        "internal_purchase_profile_not_allowed"
                    )
                tool_version = "1.0.0"
            elif delivery_target == "ENTERPRISE_ERP":
                if not profile_id:
                    raise PurchaseDeliveryProfileConflict(
                        "enterprise_procurement_profile_required"
                    )
                profile = self._resolve_procurement_profile(
                    tenant_id=identity.tenant_id,
                    now=now,
                )
                if profile.profile_id != profile_id:
                    raise PurchaseDeliveryProfileConflict(
                        "enterprise_procurement_profile_not_available"
                    )
                parameters["delivery_target"] = "ENTERPRISE_ERP"
                parameters["erp_binding"] = _procurement_profile_binding(profile)
                tool_version = "1.1.0"
            else:
                raise PurchaseDeliveryProfileConflict(
                    "purchase_delivery_target_unsupported"
                )
            definition = self._registry.get("purchase.request", tool_version)
            definition.validate(parameters)
            proposal_id = f"proposal-{uuid4().hex}"
            approval_id = f"approval-{uuid4().hex}"
            operation_id = f"operation-{uuid4().hex}"
            session.add(
                ActionProposalRecord(
                    proposal_id=proposal_id,
                    tenant_id=identity.tenant_id,
                    incident_id=incident_id,
                    initiator_subject_id=identity.subject_id,
                    tool_id=definition.tool_id,
                    tool_version=definition.version,
                    risk_tier=definition.risk_tier,
                    parameters=parameters,
                    parameters_hash=_digest(parameters),
                    resource_version=incident_version,
                    operation_id=operation_id,
                    status="PENDING_APPROVAL",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ApprovalRequestRecord(
                    approval_id=approval_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    policy_version="m2-t2-v1",
                    status="PENDING",
                    expires_at=now + T2_APPROVAL_TTL,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            return PurchaseRequestProposal(
                proposal_id,
                approval_id,
                parameters,
                operation_id,
            )

    def list_purchase_requests(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        request_id: str,
    ) -> list[PurchaseRequestProjection]:
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident is None:
                raise ApprovalNotVisible
            self._authorizer.require(
                identity,
                Action.READ_PURCHASE_REQUEST,
                ResourceContext(identity.tenant_id, incident_id, incident.asset_id),
                request_id=request_id,
            )
            rows = session.execute(
                select(
                    PurchaseRequestRecord,
                    ExecutionReconciliationRecord.reconciliation_id,
                    PurchaseFulfillmentRecord,
                )
                .outerjoin(
                    ExecutionReconciliationRecord,
                    (
                        ExecutionReconciliationRecord.tenant_id
                        == PurchaseRequestRecord.tenant_id
                    )
                    & (
                        ExecutionReconciliationRecord.operation_id
                        == PurchaseRequestRecord.operation_id
                    ),
                )
                .outerjoin(
                    PurchaseFulfillmentRecord,
                    (
                        PurchaseFulfillmentRecord.tenant_id
                        == PurchaseRequestRecord.tenant_id
                    )
                    & (
                        PurchaseFulfillmentRecord.purchase_request_id
                        == PurchaseRequestRecord.purchase_request_id
                    ),
                )
                .where(
                    PurchaseRequestRecord.tenant_id == identity.tenant_id,
                    PurchaseRequestRecord.incident_id == incident_id,
                )
                .order_by(
                    PurchaseRequestRecord.created_at.desc(),
                    PurchaseRequestRecord.purchase_request_id,
                )
            )
            purchase_rows = list(rows)
        return [
            PurchaseRequestProjection(
                request=record,
                reconciliation_id=reconciliation_id,
                fulfillment=fulfillment,
                reservation_readiness=self._procurement_reservation_readiness(
                    identity,
                    record=record,
                    fulfillment=fulfillment,
                    request_id=request_id,
                ),
            )
            for record, reconciliation_id, fulfillment in purchase_rows
        ]

    def _procurement_reservation_readiness(
        self,
        identity: IdentityContext,
        *,
        record: PurchaseRequestRecord,
        fulfillment: PurchaseFulfillmentRecord | None,
        request_id: str,
    ) -> ProcurementReservationReadiness | None:
        if record.delivery_target != "ENTERPRISE_ERP":
            return None
        if record.status != "ACCEPTED":
            return ProcurementReservationReadiness(
                "NOT_RECEIVED",
                "erp_purchase_not_accepted",
                "WAIT_FOR_ERP_ACCEPTANCE",
            )
        if fulfillment is None:
            return ProcurementReservationReadiness(
                "AWAITING_ERP_FULFILLMENT",
                "erp_fulfillment_not_reported",
                "WAIT_FOR_ERP_FULFILLMENT",
            )
        if fulfillment.status != "RECEIVED":
            return ProcurementReservationReadiness(
                "NOT_RECEIVED",
                "erp_goods_not_received",
                "WAIT_FOR_ERP_RECEIPT",
            )
        try:
            invocation = self.invoke(
                identity,
                tool_id="parts.availability",
                version="1.0.0",
                parameters={"asset_id": record.asset_id},
                request_id=request_id,
            )
        except (
            EnterpriseToolError,
            ToolRateLimitExceeded,
            TimeoutError,
            OSError,
            TypeError,
            ValueError,
        ):
            return ProcurementReservationReadiness(
                "FACTS_UNAVAILABLE",
                "wms_facts_unavailable",
                "RETRY_WMS_FACTS",
            )
        if invocation.status != "SUCCEEDED" or invocation.data is None:
            return ProcurementReservationReadiness(
                "FACTS_UNAVAILABLE",
                "wms_facts_unavailable",
                "RETRY_WMS_FACTS",
            )
        try:
            return _procurement_reservation_readiness_from_wms(
                record,
                fulfillment,
                invocation.data,
            )
        except (KeyError, TypeError, ValueError):
            return ProcurementReservationReadiness(
                "FACTS_UNAVAILABLE",
                "wms_facts_invalid",
                "RETRY_WMS_FACTS",
            )

    def propose_refund_request(
        self,
        identity: IdentityContext,
        *,
        work_order_id: str,
        work_order_version: int,
        amount_minor: int,
        currency: str,
        cost_center: str,
        reason: str,
        request_id: str,
        delivery_target: str = "INTERNAL",
        profile_id: str | None = None,
        reason_code: str | None = None,
    ) -> RefundRequestProposal:
        """Create a T2 compensation proposal from current customer dissatisfaction facts."""

        now = datetime.now(UTC)
        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            row = (
                session.execute(
                    select(WorkOrderRecord, IncidentRecord, AssetSiteLinkRecord.site_id)
                    .join(IncidentRecord, IncidentRecord.incident_id == WorkOrderRecord.incident_id)
                    .outerjoin(
                        AssetSiteLinkRecord,
                        AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                    )
                    .where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.work_order_id == work_order_id,
                    )
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                raise ApprovalNotVisible
            work, incident, site_id = row
            self._authorizer.require(
                identity,
                Action.PROPOSE_REFUND_REQUEST,
                _scoped_resource(
                    identity,
                    resource_id=work_order_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                ),
                request_id=request_id,
            )
            if work.version != work_order_version or work.status not in {
                "COMPLETED",
                "VERIFIED",
                "CLOSED",
            }:
                raise ApprovalConflict
            if (
                session.scalar(
                    select(RefundRequestRecord.refund_request_id).where(
                        RefundRequestRecord.tenant_id == identity.tenant_id,
                        RefundRequestRecord.work_order_id == work_order_id,
                    )
                )
                is not None
            ):
                raise RefundAlreadyRequested
            confirmation = _latest_customer_result_confirmation(
                session,
                tenant_id=identity.tenant_id,
                work_order_id=work_order_id,
            )
            if not _refund_eligible(confirmation):
                raise RefundNotEligible
            assert confirmation is not None
            assert confirmation.satisfaction_rating is not None
            assert confirmation.result_accepted is not None
            parameters: dict[str, Any] = {
                "incident_id": incident.incident_id,
                "asset_id": incident.asset_id,
                "work_order_id": work_order_id,
                "customer_subject_id": incident.reporter_subject_id,
                "customer_confirmation_update_id": confirmation.update_id,
                "customer_result_accepted": confirmation.result_accepted,
                "satisfaction_rating": confirmation.satisfaction_rating,
                "amount_minor": amount_minor,
                "currency": currency,
                "cost_center": cost_center,
                "reason": reason,
                "compensation_policy_id": "customer-service-compensation-v1",
            }
            if delivery_target == "INTERNAL":
                if profile_id is not None or reason_code is not None:
                    raise RefundDeliveryProfileConflict(
                        "internal_refund_profile_not_allowed"
                    )
                tool_version = "1.0.0"
            elif delivery_target == "ENTERPRISE_FINANCE":
                if not profile_id:
                    raise RefundDeliveryProfileConflict(
                        "enterprise_refund_profile_required"
                    )
                if reason_code not in {
                    "SERVICE_NOT_ACCEPTED",
                    "LOW_SATISFACTION",
                    "REWORK_COMPENSATION",
                    "OTHER_APPROVED_COMPENSATION",
                }:
                    raise RefundDeliveryProfileConflict(
                        "enterprise_refund_reason_code_invalid"
                    )
                profile = self._resolve_refund_profile(
                    tenant_id=identity.tenant_id,
                    customer_subject_id=incident.reporter_subject_id,
                    now=now,
                )
                if profile.profile_id != profile_id:
                    raise RefundDeliveryProfileConflict(
                        "enterprise_refund_profile_not_available"
                    )
                parameters["delivery_target"] = "ENTERPRISE_FINANCE"
                parameters["reason_code"] = reason_code
                parameters["finance_binding"] = _refund_profile_binding(profile)
                tool_version = "1.1.0"
            else:
                raise RefundDeliveryProfileConflict(
                    "refund_delivery_target_unsupported"
                )
            definition = self._registry.get("refund.request", tool_version)
            definition.validate(parameters)
            proposal_id = f"proposal-{uuid4().hex}"
            approval_id = f"approval-{uuid4().hex}"
            operation_id = f"operation-{uuid4().hex}"
            session.add(
                ActionProposalRecord(
                    proposal_id=proposal_id,
                    tenant_id=identity.tenant_id,
                    incident_id=incident.incident_id,
                    initiator_subject_id=identity.subject_id,
                    tool_id=definition.tool_id,
                    tool_version=definition.version,
                    risk_tier=definition.risk_tier,
                    parameters=parameters,
                    parameters_hash=_digest(parameters),
                    resource_version=work_order_version,
                    operation_id=operation_id,
                    status="PENDING_APPROVAL",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ApprovalRequestRecord(
                    approval_id=approval_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    policy_version="m2-t2-v1",
                    status="PENDING",
                    expires_at=now + T2_APPROVAL_TTL,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            return RefundRequestProposal(
                proposal_id,
                approval_id,
                parameters,
                operation_id,
            )

    def list_refund_requests(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        request_id: str,
    ) -> list[tuple[RefundRequestRecord, str | None]]:
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            row = (
                session.execute(
                    select(IncidentRecord, AssetSiteLinkRecord.site_id)
                    .outerjoin(
                        AssetSiteLinkRecord,
                        AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                    )
                    .where(
                        IncidentRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.incident_id == incident_id,
                    )
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                raise ApprovalNotVisible
            incident, site_id = row
            self._authorizer.require(
                identity,
                Action.READ_REFUND_REQUEST,
                _scoped_resource(
                    identity,
                    resource_id=incident_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                ),
                request_id=request_id,
            )
            rows = session.execute(
                select(
                    RefundRequestRecord,
                    ExecutionReconciliationRecord.reconciliation_id,
                )
                .outerjoin(
                    ExecutionReconciliationRecord,
                    (
                        ExecutionReconciliationRecord.tenant_id
                        == RefundRequestRecord.tenant_id
                    )
                    & (
                        ExecutionReconciliationRecord.operation_id
                        == RefundRequestRecord.operation_id
                    ),
                )
                .where(
                    RefundRequestRecord.tenant_id == identity.tenant_id,
                    RefundRequestRecord.incident_id == incident_id,
                )
                .order_by(
                    RefundRequestRecord.created_at.desc(),
                    RefundRequestRecord.refund_request_id,
                )
            )
            return [(record, reconciliation_id) for record, reconciliation_id in rows]

    def propose_work_order_assignment(
        self,
        identity: IdentityContext,
        *,
        work_order_id: str,
        work_order_version: int,
        reason: str,
        request_id: str,
        assignee_subject_id: str | None = None,
        assignment_target: str = "LOCAL_DIRECTORY",
        candidate_profile_id: str | None = None,
    ) -> WorkOrderAssignmentProposal:
        """Create a local v1.0 or server-bound enterprise FSM v1.1 proposal."""

        now = datetime.now(UTC)
        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            row = (
                session.execute(
                    select(WorkOrderRecord, IncidentRecord, AssetSiteLinkRecord.site_id)
                    .join(IncidentRecord, IncidentRecord.incident_id == WorkOrderRecord.incident_id)
                    .outerjoin(
                        AssetSiteLinkRecord,
                        AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                    )
                    .where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.work_order_id == work_order_id,
                    )
                    .with_for_update(of=WorkOrderRecord)
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                raise ApprovalNotVisible
            work, incident, site_id = row
            if incident.asset_id in identity.asset_ids:
                resource = ResourceContext(
                    identity.tenant_id,
                    work_order_id,
                    asset_id=incident.asset_id,
                )
            elif site_id is not None and site_id in identity.site_ids:
                resource = ResourceContext(
                    identity.tenant_id,
                    work_order_id,
                    site_id=site_id,
                )
            else:
                resource = ResourceContext(
                    identity.tenant_id,
                    work_order_id,
                    asset_id=incident.asset_id,
                )
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                resource,
                request_id=request_id,
            )
            if (
                work.version != work_order_version
                or work.status != "READY"
                or work.pending_assignment_operation_id is not None
            ):
                raise ApprovalConflict
            parameters: dict[str, Any] = {
                "incident_id": incident.incident_id,
                "asset_id": incident.asset_id,
                "work_order_id": work_order_id,
                "reason": reason,
            }
            if assignment_target == "LOCAL_DIRECTORY":
                if candidate_profile_id is not None or not assignee_subject_id:
                    raise FsmAssignmentProfileConflict(
                        "local_assignment_subject_required"
                    )
                if not is_eligible_field_engineer(
                    session,
                    tenant_id=identity.tenant_id,
                    subject_id=assignee_subject_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                ):
                    raise ApprovalConflict
                parameters["assignee_subject_id"] = assignee_subject_id
                tool_version = "1.0.0"
            elif assignment_target == "ENTERPRISE_FSM":
                if assignee_subject_id is not None or not candidate_profile_id:
                    raise FsmAssignmentProfileConflict(
                        "enterprise_fsm_candidate_required"
                    )
                if site_id is None:
                    raise FsmAssignmentProfileUnavailable(
                        "fsm_assignment_site_unavailable"
                    )
                candidates = self._resolve_fsm_assignment_candidates(
                    tenant_id=identity.tenant_id,
                    work_order_id=work_order_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                    service_window_start=work.service_window_start,
                    service_window_end=work.service_window_end,
                    now=now,
                )
                candidate = next(
                    (
                        item
                        for item in candidates
                        if item.candidate_profile_id == candidate_profile_id
                    ),
                    None,
                )
                if candidate is None:
                    raise FsmAssignmentProfileConflict(
                        "fsm_assignment_candidate_not_available"
                    )
                if not is_eligible_field_engineer(
                    session,
                    tenant_id=identity.tenant_id,
                    subject_id=candidate.assignee_subject_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                ):
                    raise FsmAssignmentProfileConflict(
                        "fsm_assignment_candidate_not_eligible"
                    )
                parameters.update(
                    {
                        "site_id": site_id,
                        "assignee_subject_id": candidate.assignee_subject_id,
                        "assignment_target": "ENTERPRISE_FSM",
                        "fsm_binding": _fsm_assignment_candidate_binding(candidate),
                    }
                )
                tool_version = "1.1.0"
            else:
                raise FsmAssignmentProfileConflict(
                    "assignment_target_unsupported"
                )
            definition = self._registry.get("work_order.assign", tool_version)
            definition.validate(parameters)
            parameters_hash = _digest(parameters)
            existing = (
                session.execute(
                    select(ActionProposalRecord, ApprovalRequestRecord)
                    .join(
                        ApprovalRequestRecord,
                        ApprovalRequestRecord.proposal_id
                        == ActionProposalRecord.proposal_id,
                    )
                    .where(
                        ActionProposalRecord.tenant_id == identity.tenant_id,
                        ApprovalRequestRecord.tenant_id == identity.tenant_id,
                        ActionProposalRecord.incident_id == incident.incident_id,
                        ActionProposalRecord.initiator_subject_id
                        == identity.subject_id,
                        ActionProposalRecord.tool_id == definition.tool_id,
                        ActionProposalRecord.tool_version == definition.version,
                        ActionProposalRecord.parameters_hash == parameters_hash,
                        ActionProposalRecord.resource_version == work_order_version,
                        ActionProposalRecord.status.in_(
                            ("PENDING_APPROVAL", "APPROVED")
                        ),
                        ApprovalRequestRecord.status.in_(("PENDING", "APPROVED")),
                        ApprovalRequestRecord.expires_at > now,
                    )
                    .order_by(ActionProposalRecord.created_at.desc())
                )
                .tuples()
                .first()
            )
            if existing is not None:
                existing_proposal, existing_approval = existing
                return WorkOrderAssignmentProposal(
                    existing_proposal.proposal_id,
                    existing_approval.approval_id,
                    dict(existing_proposal.parameters),
                    existing_proposal.operation_id,
                )
            proposal_id = f"proposal-{uuid4().hex}"
            approval_id = f"approval-{uuid4().hex}"
            operation_id = f"operation-{uuid4().hex}"
            session.add(
                ActionProposalRecord(
                    proposal_id=proposal_id,
                    tenant_id=identity.tenant_id,
                    incident_id=incident.incident_id,
                    initiator_subject_id=identity.subject_id,
                    tool_id=definition.tool_id,
                    tool_version=definition.version,
                    risk_tier=definition.risk_tier,
                    parameters=parameters,
                    parameters_hash=parameters_hash,
                    resource_version=work_order_version,
                    operation_id=operation_id,
                    status="PENDING_APPROVAL",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ApprovalRequestRecord(
                    approval_id=approval_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    policy_version="m2-t2-v1",
                    status="PENDING",
                    expires_at=now + T2_APPROVAL_TTL,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            return WorkOrderAssignmentProposal(
                proposal_id,
                approval_id,
                parameters,
                operation_id,
            )

    def propose_work_order_closure(
        self,
        identity: IdentityContext,
        *,
        work_order_id: str,
        work_order_version: int,
        reason: str,
        request_id: str,
    ) -> WorkOrderClosureProposal:
        """Create a T2 closure proposal only from complete, verified facts."""

        now = datetime.now(UTC)
        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            row = (
                session.execute(
                    select(WorkOrderRecord, IncidentRecord, AssetSiteLinkRecord.site_id)
                    .join(IncidentRecord, IncidentRecord.incident_id == WorkOrderRecord.incident_id)
                    .outerjoin(
                        AssetSiteLinkRecord,
                        AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                    )
                    .where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.work_order_id == work_order_id,
                    )
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                raise ApprovalNotVisible
            work, incident, site_id = row
            if incident.asset_id in identity.asset_ids:
                resource = ResourceContext(
                    identity.tenant_id,
                    work_order_id,
                    asset_id=incident.asset_id,
                )
            elif site_id is not None and site_id in identity.site_ids:
                resource = ResourceContext(
                    identity.tenant_id,
                    work_order_id,
                    site_id=site_id,
                )
            else:
                resource = ResourceContext(
                    identity.tenant_id,
                    work_order_id,
                    asset_id=incident.asset_id,
                )
            self._authorizer.require(
                identity,
                Action.PROPOSE_ACTION,
                resource,
                request_id=request_id,
            )
            if work.version != work_order_version or work.status != "VERIFIED":
                raise ApprovalConflict
            try:
                WorkOrderService.closure_facts_in_session(
                    session,
                    tenant_id=identity.tenant_id,
                    work_order_id=work_order_id,
                    current_version=work.version,
                )
            except WorkOrderConflict as exc:
                raise ApprovalConflict from exc
            parameters = {
                "incident_id": incident.incident_id,
                "asset_id": incident.asset_id,
                "work_order_id": work_order_id,
                "reason": reason,
            }
            definition = self._registry.get("work_order.close", "1.0.0")
            definition.validate(parameters)
            proposal_id = f"proposal-{uuid4().hex}"
            approval_id = f"approval-{uuid4().hex}"
            operation_id = f"operation-{uuid4().hex}"
            session.add(
                ActionProposalRecord(
                    proposal_id=proposal_id,
                    tenant_id=identity.tenant_id,
                    incident_id=incident.incident_id,
                    initiator_subject_id=identity.subject_id,
                    tool_id=definition.tool_id,
                    tool_version=definition.version,
                    risk_tier=definition.risk_tier,
                    parameters=parameters,
                    parameters_hash=_digest(parameters),
                    resource_version=work_order_version,
                    operation_id=operation_id,
                    status="PENDING_APPROVAL",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ApprovalRequestRecord(
                    approval_id=approval_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal_id,
                    policy_version="m2-t2-v1",
                    status="PENDING",
                    expires_at=now + T2_APPROVAL_TTL,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            return WorkOrderClosureProposal(
                proposal_id,
                approval_id,
                parameters,
                operation_id,
            )

    def list_approvals(
        self, identity: IdentityContext, *, request_id: str
    ) -> list[tuple[ApprovalRequestRecord, ActionProposalRecord]]:
        self._authorizer.require(
            identity,
            Action.READ_APPROVAL,
            ResourceContext(identity.tenant_id),
            request_id=request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            rows = list(
                session.execute(
                    select(ApprovalRequestRecord, ActionProposalRecord)
                    .join(
                        ActionProposalRecord,
                        ActionProposalRecord.proposal_id == ApprovalRequestRecord.proposal_id,
                    )
                    .where(
                        ApprovalRequestRecord.tenant_id == identity.tenant_id,
                        ActionProposalRecord.tenant_id == identity.tenant_id,
                    )
                    .order_by(ApprovalRequestRecord.created_at)
                )
                .tuples()
                .all()
            )
            record_incident_disclosure(
                session,
                identity,
                (proposal.incident_id for _, proposal in rows),
                resource_kind="approval-list",
                resource_id="approvals",
                request_id=request_id,
            )
            return rows

    def list_execution_reconciliations(
        self,
        identity: IdentityContext,
        *,
        status: str | None,
        request_id: str,
    ) -> list[ExecutionReconciliationRecord]:
        """List tenant-scoped unknown-outcome tasks without exposing tool parameters."""

        self._authorizer.require(
            identity,
            Action.READ_EXECUTION_RECONCILIATION,
            ResourceContext(identity.tenant_id, "execution-reconciliations"),
            request_id=request_id,
        )
        if status is not None and status not in {"PENDING", "RESOLVED", "ESCALATED"}:
            raise ValueError("invalid_reconciliation_status")
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            statement = select(ExecutionReconciliationRecord).where(
                ExecutionReconciliationRecord.tenant_id == identity.tenant_id
            )
            if status is not None:
                statement = statement.where(ExecutionReconciliationRecord.status == status)
            records = list(
                session.scalars(
                    statement.order_by(
                        ExecutionReconciliationRecord.created_at.desc(),
                        ExecutionReconciliationRecord.reconciliation_id,
                    )
                )
            )

            self._record_reconciliation_disclosure(session, identity, records, request_id)
            return records
    def get_execution_reconciliation(
        self,
        identity: IdentityContext,
        reconciliation_id: str,
        *,
        request_id: str,
    ) -> tuple[
        ExecutionReconciliationRecord,
        list[ExecutionReconciliationEventRecord],
    ]:
        self._authorizer.require(
            identity,
            Action.READ_EXECUTION_RECONCILIATION,
            ResourceContext(identity.tenant_id, reconciliation_id),
            request_id=request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = session.scalar(
                select(ExecutionReconciliationRecord).where(
                    ExecutionReconciliationRecord.tenant_id == identity.tenant_id,
                    ExecutionReconciliationRecord.reconciliation_id == reconciliation_id,
                )
            )
            if record is None:
                raise ExecutionReconciliationNotVisible
            events = list(
                session.scalars(
                    select(ExecutionReconciliationEventRecord)
                    .where(
                        ExecutionReconciliationEventRecord.tenant_id == identity.tenant_id,
                        ExecutionReconciliationEventRecord.reconciliation_id == reconciliation_id,
                    )
                    .order_by(ExecutionReconciliationEventRecord.sequence)
                )
            )
            self._record_reconciliation_disclosure(session, identity, [record], request_id)
            return record, events

    @staticmethod
    def _record_reconciliation_disclosure(
        session: Session,
        identity: IdentityContext,
        records: Iterable[ExecutionReconciliationRecord],
        request_id: str,
    ) -> None:
        records = tuple(records)
        if not records:
            return
        proposals = dict(session.execute(
            select(ActionProposalRecord.proposal_id, ActionProposalRecord.incident_id).where(
                ActionProposalRecord.tenant_id == identity.tenant_id,
                ActionProposalRecord.proposal_id.in_(r.proposal_id for r in records),
            )
        ).all())
        if any(proposals.get(r.proposal_id) != r.incident_id for r in records):
            raise ReviewIsolationConflict("source_binding_invalid")
        record_incident_disclosure(
            session, identity, (r.incident_id for r in records),
            resource_kind="execution-reconciliation",
            resource_id="execution-reconciliations",
            request_id=request_id,
        )

    def _authorize_reconciliation_execution(
        self, identity: IdentityContext, reconciliation_id: str, request_id: str
    ) -> None:
        # Persist exposure before external I/O; a subsequent claim must see it.
        # Keep the original execution transaction/version checks after this short boundary.
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = session.scalar(select(ExecutionReconciliationRecord).where(
                ExecutionReconciliationRecord.tenant_id == identity.tenant_id,
                ExecutionReconciliationRecord.reconciliation_id == reconciliation_id,
            ))
            if record is None:
                raise ExecutionReconciliationNotVisible
            self._authorizer.require(
                identity, Action.RECONCILE_EXECUTION,
                ResourceContext(identity.tenant_id, reconciliation_id), request_id=request_id,
            )
            self._record_reconciliation_disclosure(session, identity, [record], request_id)

    def reconcile_execution(
        self,
        identity: IdentityContext,
        *,
        reconciliation_id: str,
        expected_version: int,
        request_id: str,
    ) -> ExecutionReconciliationRecord:
        """Query the external idempotency identity without replaying the side effect."""

        self._authorize_reconciliation_execution(identity, reconciliation_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(ExecutionReconciliationRecord)
                .where(
                    ExecutionReconciliationRecord.tenant_id == identity.tenant_id,
                    ExecutionReconciliationRecord.reconciliation_id == reconciliation_id,
                )
                .with_for_update()
            )
            if record is None:
                raise ExecutionReconciliationNotVisible
            self._authorizer.require(
                identity,
                Action.RECONCILE_EXECUTION,
                ResourceContext(identity.tenant_id, reconciliation_id),
                request_id=request_id,
            )
            if record.version != expected_version or record.status != "PENDING":
                raise ExecutionReconciliationConflict
            proposal = session.scalar(
                select(ActionProposalRecord).where(
                    ActionProposalRecord.tenant_id == identity.tenant_id,
                    ActionProposalRecord.proposal_id == record.proposal_id,
                )
            )
            if proposal is None:
                raise ExecutionReconciliationConflict
            if record.tool_id == "work_order.assign":
                return self._reconcile_enterprise_fsm_assignment(
                    session,
                    identity,
                    record,
                    proposal,
                    now=now,
                )
            if record.tool_id == "refund.request":
                return self._reconcile_enterprise_refund(
                    session,
                    identity,
                    record,
                    proposal,
                    now=now,
                )
            if record.tool_id == "purchase.request":
                return self._reconcile_enterprise_purchase(
                    session,
                    identity,
                    record,
                    proposal,
                    now=now,
                )
            if record.tool_id == "customer.notify":
                return self._reconcile_customer_notification(
                    session,
                    identity,
                    record,
                    proposal,
                    now=now,
                )
            if record.tool_id == "parts.issue":
                return self._reconcile_part_issue(
                    session,
                    identity,
                    record,
                    proposal,
                    now=now,
                )
            if record.tool_id in {"parts.consume", "parts.return"}:
                return self._reconcile_part_movement(
                    session,
                    identity,
                    record,
                    proposal,
                    now=now,
                )
            if record.tool_id != "parts.reserve":
                raise ExecutionReconciliationConflict
            try:
                result = self._parts.reconcile(record.operation_id)
            except EnterpriseToolError:
                self._record_reconciliation_check(
                    session,
                    identity,
                    record,
                    now=now,
                    event_type="CHECK_FAILED",
                    reason="external_reconciliation_unavailable",
                )
                return record
            if result is None:
                self._record_reconciliation_check(
                    session,
                    identity,
                    record,
                    now=now,
                    event_type="CHECK_PENDING",
                    reason="external_result_not_yet_visible",
                )
                return record
            attempt_number = (
                int(
                    session.scalar(
                        select(func.count())
                        .select_from(ExecutionAttemptRecord)
                        .where(
                            ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                            ExecutionAttemptRecord.operation_id == record.operation_id,
                        )
                    )
                    or 0
                )
                + 1
            )
            execution = self._persist_success(
                session,
                identity,
                record.approval_id,
                proposal,
                result,
                attempt_number,
                now,
                reason="reconciled_unknown_result",
            )
            if execution.status != "SUCCEEDED":
                raise ExecutionReconciliationConflict
            assert execution.execution_attempt_id is not None
            record.status = "RESOLVED"
            record.outcome = "SUCCEEDED"
            record.reason = "external_result_confirmed"
            record.external_reference_id = result.reservation_id
            record.resolved_execution_attempt_id = execution.execution_attempt_id
            record.last_checked_at = now
            record.resolved_at = now
            record.version += 1
            record.updated_at = now
            session.add(
                ExecutionReconciliationEventRecord(
                    event_id=f"reconciliation-event-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    reconciliation_id=record.reconciliation_id,
                    sequence=record.version,
                    event_type="RESOLVED",
                    actor_subject_id=identity.subject_id,
                    reason=record.reason,
                    outcome=record.outcome,
                    created_at=now,
                    updated_at=now,
                )
            )
            return record

    def _reconcile_part_issue(
        self,
        session: Session,
        identity: IdentityContext,
        record: ExecutionReconciliationRecord,
        proposal: ActionProposalRecord,
        *,
        now: datetime,
    ) -> ExecutionReconciliationRecord:
        issue = session.scalar(
            select(PartIssueRecord)
            .where(
                PartIssueRecord.tenant_id == identity.tenant_id,
                PartIssueRecord.operation_id == record.operation_id,
            )
            .with_for_update()
        )
        if (
            issue is None
            or issue.status != "RECONCILING"
            or issue.reconciliation_id != record.reconciliation_id
        ):
            raise ExecutionReconciliationConflict
        try:
            result = self._parts.reconcile_issue(record.operation_id)
        except EnterpriseToolError:
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_FAILED",
                reason="external_reconciliation_unavailable",
            )
            return record
        if result is None:
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_PENDING",
                reason="external_result_not_yet_visible",
            )
            return record
        if not _part_issue_result_matches(issue, result):
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_FAILED",
                reason="external_result_identity_mismatch",
            )
            return record
        reservation = session.scalar(
            select(PartReservationRecord)
            .where(
                PartReservationRecord.tenant_id == identity.tenant_id,
                PartReservationRecord.reservation_id == issue.reservation_id,
            )
            .with_for_update()
        )
        if reservation is None or reservation.status != "RESERVED":
            raise ExecutionReconciliationConflict
        attempt_number = (
            int(
                session.scalar(
                    select(func.count())
                    .select_from(ExecutionAttemptRecord)
                    .where(
                        ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                        ExecutionAttemptRecord.operation_id == record.operation_id,
                    )
                )
                or 0
            )
            + 1
        )
        execution = self._persist_part_issue_success(
            session,
            identity,
            record.approval_id,
            proposal,
            issue,
            reservation,
            result,
            attempt_number,
            now,
            reason="reconciled_unknown_result",
        )
        if execution.status != "SUCCEEDED" or execution.execution_attempt_id is None:
            raise ExecutionReconciliationConflict
        record.status = "RESOLVED"
        record.outcome = "SUCCEEDED"
        record.reason = "external_result_confirmed"
        record.external_reference_id = result.issue_id
        record.resolved_execution_attempt_id = execution.execution_attempt_id
        record.last_checked_at = now
        record.resolved_at = now
        record.version += 1
        record.updated_at = now
        session.add(
            ExecutionReconciliationEventRecord(
                event_id=f"reconciliation-event-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                reconciliation_id=record.reconciliation_id,
                sequence=record.version,
                event_type="RESOLVED",
                actor_subject_id=identity.subject_id,
                reason=record.reason,
                outcome=record.outcome,
                created_at=now,
                updated_at=now,
            )
        )
        return record

    def _reconcile_part_movement(
        self,
        session: Session,
        identity: IdentityContext,
        record: ExecutionReconciliationRecord,
        proposal: ActionProposalRecord,
        *,
        now: datetime,
    ) -> ExecutionReconciliationRecord:
        movement = session.scalar(
            select(PartMaterialMovementRecord)
            .where(
                PartMaterialMovementRecord.tenant_id == identity.tenant_id,
                PartMaterialMovementRecord.operation_id == record.operation_id,
            )
            .with_for_update()
        )
        if (
            movement is None
            or movement.status != "RECONCILING"
            or movement.reconciliation_id != record.reconciliation_id
            or proposal.tool_id
            != ("parts.consume" if movement.movement_kind == "CONSUME" else "parts.return")
        ):
            raise ExecutionReconciliationConflict
        try:
            result = self._parts.reconcile_movement(
                record.operation_id,
                movement.movement_kind,
            )
        except (EnterpriseToolError, TimeoutError, OSError):
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_FAILED",
                reason="external_reconciliation_unavailable",
            )
            return record
        if result is None:
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_PENDING",
                reason="external_result_not_yet_visible",
            )
            return record
        if not _part_movement_result_matches(movement, result):
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_FAILED",
                reason="external_result_identity_mismatch",
            )
            return record
        attempt_number = (
            int(
                session.scalar(
                    select(func.count())
                    .select_from(ExecutionAttemptRecord)
                    .where(
                        ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                        ExecutionAttemptRecord.operation_id == record.operation_id,
                    )
                )
                or 0
            )
            + 1
        )
        execution = self._persist_part_movement_success(
            session,
            identity,
            record.approval_id,
            proposal,
            movement,
            result,
            attempt_number,
            now,
            reason="reconciled_unknown_result",
        )
        if execution.status != "SUCCEEDED" or execution.execution_attempt_id is None:
            raise ExecutionReconciliationConflict
        record.status = "RESOLVED"
        record.outcome = "SUCCEEDED"
        record.reason = "external_result_confirmed"
        record.external_reference_id = result.movement_id
        record.resolved_execution_attempt_id = execution.execution_attempt_id
        record.last_checked_at = now
        record.resolved_at = now
        record.version += 1
        record.updated_at = now
        session.add(
            ExecutionReconciliationEventRecord(
                event_id=f"reconciliation-event-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                reconciliation_id=record.reconciliation_id,
                sequence=record.version,
                event_type="RESOLVED",
                actor_subject_id=identity.subject_id,
                reason=record.reason,
                outcome=record.outcome,
                created_at=now,
                updated_at=now,
            )
        )
        return record

    def _reconcile_enterprise_fsm_assignment(
        self,
        session: Session,
        identity: IdentityContext,
        record: ExecutionReconciliationRecord,
        proposal: ActionProposalRecord,
        *,
        now: datetime,
    ) -> ExecutionReconciliationRecord:
        adapter = self._fsm_assignments
        delivery = session.scalar(
            select(FsmAssignmentDeliveryRecord)
            .where(
                FsmAssignmentDeliveryRecord.tenant_id == identity.tenant_id,
                FsmAssignmentDeliveryRecord.operation_id == record.operation_id,
            )
            .with_for_update()
        )
        if adapter is None or delivery is None or delivery.provider != adapter.provider_id:
            raise ExecutionReconciliationConflict
        try:
            raw_result = adapter.reconcile(record.operation_id)
            result = (
                _validated_fsm_assignment_delivery_result(raw_result, now=now)
                if raw_result is not None
                else None
            )
        except (EnterpriseToolError, TimeoutError, OSError, TypeError, ValueError):
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_FAILED",
                reason="external_reconciliation_unavailable",
            )
            return record
        if result is None or result.status == "RECONCILING":
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_PENDING",
                reason="external_result_not_yet_visible",
            )
            return record
        self._apply_fsm_assignment_terminal_result(
            session,
            identity,
            proposal,
            delivery,
            result,
            reconciliation=record,
            now=now,
        )
        return record

    def _apply_fsm_assignment_terminal_result(
        self,
        session: Session,
        identity: IdentityContext,
        proposal: ActionProposalRecord,
        delivery: FsmAssignmentDeliveryRecord,
        result: FsmAssignmentDeliveryResult,
        *,
        reconciliation: ExecutionReconciliationRecord | None,
        now: datetime,
    ) -> bool:
        if result.status not in {"ACCEPTED", "REJECTED", "CANCELLED"}:
            return False
        if result.external_assignment_id is None:
            raise ExecutionReconciliationConflict
        if (
            delivery.external_assignment_id is not None
            and delivery.external_assignment_id != result.external_assignment_id
        ):
            raise ExecutionReconciliationConflict
        terminal = {"ACCEPTED", "REJECTED", "CANCELLED"}
        if delivery.status in terminal:
            if delivery.status != result.status:
                raise ExecutionReconciliationConflict
            return False
        work = session.scalar(
            select(WorkOrderRecord)
            .where(
                WorkOrderRecord.tenant_id == identity.tenant_id,
                WorkOrderRecord.work_order_id == delivery.work_order_id,
            )
            .with_for_update()
        )
        if (
            work is None
            or work.status != "READY"
            or work.pending_assignment_operation_id != delivery.operation_id
        ):
            if reconciliation is not None:
                reconciliation.status = "ESCALATED"
                reconciliation.reason = "fsm_assignment_manual_intervention_required"
                reconciliation.last_checked_at = now
                reconciliation.version += 1
                reconciliation.updated_at = now
                session.add(
                    ExecutionReconciliationEventRecord(
                        event_id=f"reconciliation-event-{uuid4().hex}",
                        tenant_id=identity.tenant_id,
                        reconciliation_id=reconciliation.reconciliation_id,
                        sequence=reconciliation.version,
                        event_type="ESCALATED",
                        actor_subject_id=identity.subject_id,
                        reason=reconciliation.reason,
                        outcome="UNKNOWN",
                        created_at=now,
                        updated_at=now,
                    )
                )
                return False
            raise ExecutionReconciliationConflict
        delivery.external_assignment_id = result.external_assignment_id
        delivery.status = result.status
        delivery.reason = result.reason
        delivery.terminal_at = result.occurred_at
        delivery.version += 1
        delivery.updated_at = now
        attempt_number = (
            int(
                session.scalar(
                    select(func.count())
                    .select_from(ExecutionAttemptRecord)
                    .where(
                        ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                        ExecutionAttemptRecord.operation_id == delivery.operation_id,
                    )
                )
                or 0
            )
            + 1
        )
        attempt = self._fsm_assignment_attempt(
            identity.tenant_id,
            proposal,
            approval_id=delivery.approval_id,
            delivery=delivery,
            status="SUCCEEDED" if result.status == "ACCEPTED" else "FAILED",
            reason=result.reason,
            now=now,
            idempotency_key=None,
        )
        attempt.attempt_number = attempt_number
        session.add(attempt)
        if result.status == "ACCEPTED":
            try:
                WorkOrderService(self._database, self._authorizer).assign_in_session(
                    session,
                    identity,
                    delivery.work_order_id,
                    assignee_subject_id=delivery.assignee_subject_id,
                    expected_version=work.version,
                    request_id="fsm-assignment-convergence",
                    reason=str(proposal.parameters.get("reason", "FSM accepted assignment")),
                    operation_id=delivery.operation_id,
                    proposal_id=proposal.proposal_id,
                    fsm_assignment_delivery_id=delivery.delivery_id,
                )
            except WorkOrderConflict as exc:
                if reconciliation is None:
                    raise ExecutionReconciliationConflict from exc
                reconciliation.status = "ESCALATED"
                reconciliation.reason = "fsm_assignment_manual_intervention_required"
                reconciliation.last_checked_at = now
                reconciliation.version += 1
                reconciliation.updated_at = now
                return False
            proposal.status = "EXECUTED"
        else:
            work.pending_assignment_operation_id = None
            work.version += 1
            work.updated_at = now
            session.add(
                WorkOrderEventRecord(
                    event_id=f"work-event-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    work_order_id=work.work_order_id,
                    sequence=work.version,
                    event_type="work_order.assignment_released",
                    actor_subject_id=identity.subject_id,
                    payload={
                        "operation_id": delivery.operation_id,
                        "delivery_id": delivery.delivery_id,
                        "status": result.status,
                        "reason": result.reason,
                    },
                    occurred_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            proposal.status = "EXECUTION_FAILED"
        if reconciliation is not None:
            reconciliation.status = "RESOLVED"
            reconciliation.outcome = (
                "SUCCEEDED" if result.status == "ACCEPTED" else "FAILED"
            )
            reconciliation.reason = "provider_query_confirmed"
            reconciliation.external_reference_id = result.external_assignment_id
            reconciliation.resolved_execution_attempt_id = attempt.attempt_id
            reconciliation.last_checked_at = now
            reconciliation.resolved_at = now
            reconciliation.version += 1
            reconciliation.updated_at = now
            session.add(
                ExecutionReconciliationEventRecord(
                    event_id=f"reconciliation-event-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    reconciliation_id=reconciliation.reconciliation_id,
                    sequence=reconciliation.version,
                    event_type="RESOLVED",
                    actor_subject_id=identity.subject_id,
                    reason=reconciliation.reason,
                    outcome=reconciliation.outcome,
                    created_at=now,
                    updated_at=now,
                )
            )
        return True

    def _reconcile_enterprise_refund(
        self,
        session: Session,
        identity: IdentityContext,
        record: ExecutionReconciliationRecord,
        proposal: ActionProposalRecord,
        *,
        now: datetime,
    ) -> ExecutionReconciliationRecord:
        adapter = self._refunds
        if adapter is None:
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_FAILED",
                reason="refund_adapter_unavailable",
            )
            return record
        refund = session.scalar(
            select(RefundRequestRecord)
            .where(
                RefundRequestRecord.tenant_id == identity.tenant_id,
                RefundRequestRecord.operation_id == record.operation_id,
            )
            .with_for_update()
        )
        if (
            refund is None
            or refund.delivery_target != "ENTERPRISE_FINANCE"
            or refund.provider != adapter.provider_id
        ):
            raise ExecutionReconciliationConflict
        try:
            raw_result = adapter.reconcile(record.operation_id)
            result = (
                _validated_refund_delivery_result(raw_result, now=now)
                if raw_result is not None
                else None
            )
        except (EnterpriseToolError, TimeoutError, OSError, TypeError, ValueError):
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_FAILED",
                reason="external_reconciliation_unavailable",
            )
            return record
        if result is None or result.status == "RECONCILING":
            if result is not None and result.external_request_id is not None:
                if (
                    refund.external_request_id is not None
                    and refund.external_request_id != result.external_request_id
                ):
                    raise ExecutionReconciliationConflict
                refund.external_request_id = result.external_request_id
                refund.updated_at = now
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_PENDING",
                reason="external_result_not_yet_visible",
            )
            return record
        if (
            refund.external_request_id is not None
            and refund.external_request_id != result.external_request_id
        ):
            raise ExecutionReconciliationConflict
        attempt_number = (
            int(
                session.scalar(
                    select(func.count())
                    .select_from(ExecutionAttemptRecord)
                    .where(
                        ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                        ExecutionAttemptRecord.operation_id == record.operation_id,
                    )
                )
                or 0
            )
            + 1
        )
        attempt_id = f"attempt-{uuid4().hex}"
        succeeded = result.status == "ACCEPTED"
        refund.status = result.status
        refund.payment_status = (
            result.payment_status if result.status == "ACCEPTED" else "NOT_STARTED"
        )
        refund.external_request_id = result.external_request_id
        refund.provider_reason = result.reason
        refund.terminal_at = result.occurred_at
        refund.version += 1
        refund.updated_at = now
        session.add(
            _refund_execution_attempt(
                identity.tenant_id,
                proposal,
                approval_id=record.approval_id,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                status="SUCCEEDED" if succeeded else "FAILED",
                reason=result.reason,
                refund=refund,
                now=now,
                idempotency_key=None,
            )
        )
        proposal.status = "EXECUTED" if succeeded else "EXECUTION_FAILED"
        record.status = "RESOLVED"
        record.outcome = "SUCCEEDED" if succeeded else "FAILED"
        record.reason = "provider_query_confirmed"
        record.external_reference_id = result.external_request_id
        record.resolved_execution_attempt_id = attempt_id
        record.last_checked_at = now
        record.resolved_at = now
        record.version += 1
        record.updated_at = now
        session.add(
            ExecutionReconciliationEventRecord(
                event_id=f"reconciliation-event-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                reconciliation_id=record.reconciliation_id,
                sequence=record.version,
                event_type="RESOLVED",
                actor_subject_id=identity.subject_id,
                reason=record.reason,
                outcome=record.outcome,
                created_at=now,
                updated_at=now,
            )
        )
        return record

    def _reconcile_enterprise_purchase(
        self,
        session: Session,
        identity: IdentityContext,
        record: ExecutionReconciliationRecord,
        proposal: ActionProposalRecord,
        *,
        now: datetime,
    ) -> ExecutionReconciliationRecord:
        adapter = self._procurement
        if adapter is None:
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_FAILED",
                reason="procurement_adapter_unavailable",
            )
            return record
        purchase = session.scalar(
            select(PurchaseRequestRecord)
            .where(
                PurchaseRequestRecord.tenant_id == identity.tenant_id,
                PurchaseRequestRecord.operation_id == record.operation_id,
            )
            .with_for_update()
        )
        if (
            purchase is None
            or purchase.delivery_target != "ENTERPRISE_ERP"
            or purchase.provider != adapter.provider_id
        ):
            raise ExecutionReconciliationConflict
        try:
            raw_result = adapter.reconcile(record.operation_id)
            result = (
                _validated_procurement_delivery_result(raw_result, now=now)
                if raw_result is not None
                else None
            )
        except (EnterpriseToolError, TimeoutError, OSError, TypeError, ValueError):
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_FAILED",
                reason="external_reconciliation_unavailable",
            )
            return record
        if result is None or result.status == "RECONCILING":
            if result is not None and result.external_request_id is not None:
                if (
                    purchase.external_request_id is not None
                    and purchase.external_request_id != result.external_request_id
                ):
                    raise ExecutionReconciliationConflict
                purchase.external_request_id = result.external_request_id
                purchase.updated_at = now
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_PENDING",
                reason="external_result_not_yet_visible",
            )
            return record
        if (
            purchase.external_request_id is not None
            and purchase.external_request_id != result.external_request_id
        ):
            raise ExecutionReconciliationConflict
        attempt_number = (
            int(
                session.scalar(
                    select(func.count())
                    .select_from(ExecutionAttemptRecord)
                    .where(
                        ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                        ExecutionAttemptRecord.operation_id == record.operation_id,
                    )
                )
                or 0
            )
            + 1
        )
        attempt_id = f"attempt-{uuid4().hex}"
        succeeded = result.status == "ACCEPTED"
        purchase.status = result.status
        purchase.external_request_id = result.external_request_id
        purchase.reason = result.reason
        purchase.terminal_at = result.occurred_at
        purchase.version += 1
        purchase.updated_at = now
        session.add(
            _procurement_execution_attempt(
                identity.tenant_id,
                proposal,
                approval_id=record.approval_id,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                status="SUCCEEDED" if succeeded else "FAILED",
                reason=result.reason,
                purchase=purchase,
                now=now,
                idempotency_key=None,
            )
        )
        proposal.status = "EXECUTED" if succeeded else "EXECUTION_FAILED"
        record.status = "RESOLVED"
        record.outcome = "SUCCEEDED" if succeeded else "FAILED"
        record.reason = "provider_query_confirmed"
        record.external_reference_id = result.external_request_id
        record.resolved_execution_attempt_id = attempt_id
        record.last_checked_at = now
        record.resolved_at = now
        record.version += 1
        record.updated_at = now
        session.add(
            ExecutionReconciliationEventRecord(
                event_id=f"reconciliation-event-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                reconciliation_id=record.reconciliation_id,
                sequence=record.version,
                event_type="RESOLVED",
                actor_subject_id=identity.subject_id,
                reason=record.reason,
                outcome=record.outcome,
                created_at=now,
                updated_at=now,
            )
        )
        return record

    def _reconcile_customer_notification(
        self,
        session: Session,
        identity: IdentityContext,
        record: ExecutionReconciliationRecord,
        proposal: ActionProposalRecord,
        *,
        now: datetime,
    ) -> ExecutionReconciliationRecord:
        if self._notifications is None:
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_FAILED",
                reason="notification_adapter_unavailable",
            )
            return record
        delivery = session.scalar(
            select(CustomerNotificationDeliveryRecord)
            .where(
                CustomerNotificationDeliveryRecord.tenant_id == identity.tenant_id,
                CustomerNotificationDeliveryRecord.operation_id == record.operation_id,
            )
            .with_for_update()
        )
        if delivery is None:
            raise ExecutionReconciliationConflict
        try:
            raw_result = self._notifications.reconcile(record.operation_id)
            result = (
                _validated_notification_delivery_result(raw_result, now=now)
                if raw_result is not None
                else None
            )
        except (EnterpriseToolError, TimeoutError, OSError, ValueError):
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_FAILED",
                reason="external_reconciliation_unavailable",
            )
            return record
        if result is None or result.status in {"SUBMITTED", "RECONCILING"}:
            self._record_reconciliation_check(
                session,
                identity,
                record,
                now=now,
                event_type="CHECK_PENDING",
                reason="external_result_not_yet_visible",
            )
            return record
        if (
            delivery.provider_message_id is not None
            and result.provider_message_id != delivery.provider_message_id
        ):
            raise ExecutionReconciliationConflict
        if delivery.status in {"DELIVERED", "FAILED", "UNSUBSCRIBED"}:
            return record
        attempt_number = (
            int(
                session.scalar(
                    select(func.count())
                    .select_from(ExecutionAttemptRecord)
                    .where(
                        ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                        ExecutionAttemptRecord.operation_id == record.operation_id,
                    )
                )
                or 0
            )
            + 1
        )
        attempt_id = f"attempt-{uuid4().hex}"
        delivery.status = result.status
        delivery.provider_message_id = result.provider_message_id
        delivery.reason = result.reason
        delivery.terminal_at = result.occurred_at
        delivery.updated_at = now
        delivery.version += 1
        succeeded = result.status == "DELIVERED"
        session.add(
            ExecutionAttemptRecord(
                attempt_id=attempt_id,
                tenant_id=identity.tenant_id,
                proposal_id=proposal.proposal_id,
                approval_id=record.approval_id,
                operation_id=record.operation_id,
                attempt_number=attempt_number,
                status="SUCCEEDED" if succeeded else "FAILED",
                reason=result.reason,
                result={
                    "notification_delivery_id": delivery.delivery_id,
                    "provider": delivery.provider,
                    "provider_message_id": delivery.provider_message_id,
                    "status": result.status,
                },
                started_at=now,
                finished_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        proposal.status = "EXECUTED" if succeeded else "EXECUTION_FAILED"
        record.status = "RESOLVED"
        record.outcome = "SUCCEEDED" if succeeded else "FAILED"
        record.reason = "external_result_confirmed"
        record.external_reference_id = result.provider_message_id
        record.resolved_execution_attempt_id = attempt_id
        record.last_checked_at = now
        record.resolved_at = now
        record.version += 1
        record.updated_at = now
        session.add(
            ExecutionReconciliationEventRecord(
                event_id=f"reconciliation-event-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                reconciliation_id=record.reconciliation_id,
                sequence=record.version,
                event_type="RESOLVED",
                actor_subject_id=identity.subject_id,
                reason=record.reason,
                outcome=record.outcome,
                created_at=now,
                updated_at=now,
            )
        )
        return record

    def escalate_execution_reconciliation(
        self,
        identity: IdentityContext,
        *,
        reconciliation_id: str,
        expected_version: int,
        reason: str,
        request_id: str,
    ) -> ExecutionReconciliationRecord:
        """Stop automatic resolution and hand the unknown result to a human owner."""

        self._authorize_reconciliation_execution(identity, reconciliation_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(ExecutionReconciliationRecord)
                .where(
                    ExecutionReconciliationRecord.tenant_id == identity.tenant_id,
                    ExecutionReconciliationRecord.reconciliation_id == reconciliation_id,
                )
                .with_for_update()
            )
            if record is None:
                raise ExecutionReconciliationNotVisible
            self._authorizer.require(
                identity,
                Action.RECONCILE_EXECUTION,
                ResourceContext(identity.tenant_id, reconciliation_id),
                request_id=request_id,
            )
            if record.version != expected_version or record.status != "PENDING":
                raise ExecutionReconciliationConflict
            proposal = session.scalar(
                select(ActionProposalRecord).where(
                    ActionProposalRecord.tenant_id == identity.tenant_id,
                    ActionProposalRecord.proposal_id == record.proposal_id,
                )
            )
            if proposal is None:
                raise ExecutionReconciliationConflict
            record.status = "ESCALATED"
            record.outcome = "UNKNOWN"
            record.reason = reason
            record.resolved_at = now
            record.version += 1
            record.updated_at = now
            proposal.status = "RECONCILIATION_ESCALATED"
            session.add(
                ExecutionReconciliationEventRecord(
                    event_id=f"reconciliation-event-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    reconciliation_id=record.reconciliation_id,
                    sequence=record.version,
                    event_type="ESCALATED",
                    actor_subject_id=identity.subject_id,
                    reason=reason,
                    outcome=record.outcome,
                    created_at=now,
                    updated_at=now,
                )
            )
            return record

    def get_approval(
        self,
        identity: IdentityContext,
        approval_id: str,
        *,
        request_id: str,
    ) -> tuple[ApprovalRequestRecord, ActionProposalRecord]:
        self._authorizer.require(
            identity,
            Action.READ_APPROVAL,
            ResourceContext(identity.tenant_id, approval_id),
            request_id=request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            approval, proposal = self._load(session, identity.tenant_id, approval_id)
            record_incident_disclosure(
                session,
                identity,
                (proposal.incident_id,),
                resource_kind="approval",
                resource_id=approval_id,
                request_id=request_id,
            )
            return approval, proposal

    def decide(
        self,
        identity: IdentityContext,
        *,
        approval_id: str,
        expected_version: int,
        decision: str,
        reason: str,
        request_id: str,
    ) -> ApprovalRequestRecord:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            approval, proposal = self._load(session, identity.tenant_id, approval_id)
            self._authorizer.require(
                identity,
                Action.APPROVE_ACTION,
                ResourceContext(identity.tenant_id, approval_id),
                request_id=request_id,
            )
            record_incident_disclosure(
                session, identity, (proposal.incident_id,),
                resource_kind="approval", resource_id=approval_id, request_id=request_id,
            )
            ApprovalBindingGuard.require_decidable(
                initiator_subject_id=proposal.initiator_subject_id,
                decider_subject_id=identity.subject_id,
                current_status=approval.status,
                current_version=approval.version,
                expected_version=expected_version,
                expires_at=_utc(approval.expires_at),
                now=now,
                decision=decision,
            )
            session.add(
                ApprovalDecisionRecord(
                    decision_id=f"decision-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    approval_id=approval_id,
                    decision=decision,
                    reason=reason,
                    decider_subject_id=identity.subject_id,
                    parameters_hash=proposal.parameters_hash,
                    resource_version=proposal.resource_version,
                    tool_version=proposal.tool_version,
                    decided_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            approval.status, approval.version, approval.updated_at = (
                decision,
                approval.version + 1,
                now,
            )
            proposal.status = decision
            if proposal.tool_id == "parts.issue":
                issue = session.scalar(
                    select(PartIssueRecord)
                    .where(
                        PartIssueRecord.tenant_id == identity.tenant_id,
                        PartIssueRecord.proposal_id == proposal.proposal_id,
                    )
                    .with_for_update()
                )
                if issue is None or issue.status != "PENDING_APPROVAL":
                    raise ApprovalConflict
                issue.status = decision
                issue.reason = reason if decision == "REJECTED" else None
                issue.version += 1
                issue.updated_at = now
            if proposal.tool_id in {"parts.consume", "parts.return"}:
                movement = session.scalar(
                    select(PartMaterialMovementRecord)
                    .where(
                        PartMaterialMovementRecord.tenant_id == identity.tenant_id,
                        PartMaterialMovementRecord.proposal_id == proposal.proposal_id,
                    )
                    .with_for_update()
                )
                if movement is None or movement.status != "PENDING_APPROVAL":
                    raise ApprovalConflict
                movement.status = decision
                movement.reason = reason if decision == "REJECTED" else None
                movement.version += 1
                movement.updated_at = now
            return approval

    def execute_approved_proposal(
        self,
        identity: IdentityContext,
        *,
        approval_id: str,
        parameters: dict[str, Any],
        request_id: str,
        expected_approval_version: int | None = None,
        idempotency_key: str | None = None,
    ) -> ExecutionResult:
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            _, proposal = self._load(session, identity.tenant_id, approval_id)
            self._authorizer.require(
                identity, Action.EXECUTE_ACTION, ResourceContext(identity.tenant_id, approval_id),
                request_id=request_id,
            )
            record_incident_disclosure(
                session, identity, (proposal.incident_id,), resource_kind="approval-execution",
                resource_id=approval_id, request_id=request_id,
            )
        if self._approval_is_repair_work_order(identity, approval_id):
            return self._execute_repair_work_order(
                identity,
                approval_id=approval_id,
                parameters=parameters,
                request_id=request_id,
                expected_approval_version=expected_approval_version,
                idempotency_key=idempotency_key,
            )
        if self._approval_is_service_quotation(identity, approval_id):
            return self._execute_service_quotation(
                identity,
                approval_id=approval_id,
                parameters=parameters,
                request_id=request_id,
                expected_approval_version=expected_approval_version,
                idempotency_key=idempotency_key,
            )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            approval, proposal = self._load(session, identity.tenant_id, approval_id)
            rejection = ApprovalBindingGuard.execution_rejection(
                status=approval.status,
                current_version=approval.version,
                expected_version=expected_approval_version,
                expires_at=_utc(approval.expires_at),
                now=now,
                approved_parameters_hash=proposal.parameters_hash,
                supplied_parameters_hash=_digest(parameters),
            )
            if rejection is not None:
                return ExecutionResult(rejection)
            self._authorizer.require(
                identity,
                Action.EXECUTE_ACTION,
                ResourceContext(identity.tenant_id, approval_id),
                request_id=request_id,
            )
            definition = self._registry.get(proposal.tool_id, proposal.tool_version)
            definition.validate(parameters)
            if definition.risk_tier != "T2" or not definition.approval_required:
                return ExecutionResult("RISK_CONTRACT_CHANGED")
            if proposal.tool_id == "parts.issue":
                return self._execute_part_issue(
                    session,
                    identity,
                    approval_id,
                    proposal,
                    parameters,
                    now,
                    idempotency_key=idempotency_key,
                )
            if proposal.tool_id in {"parts.consume", "parts.return"}:
                return self._execute_part_movement(
                    session,
                    identity,
                    approval_id,
                    proposal,
                    parameters,
                    now,
                    idempotency_key=idempotency_key,
                )
            if proposal.tool_id == "work_order.assign":
                return self._execute_work_order_assignment(
                    session,
                    identity,
                    approval_id,
                    proposal,
                    parameters,
                    now,
                    request_id=request_id,
                    idempotency_key=idempotency_key,
                )
            if proposal.tool_id == "work_order.close":
                return self._execute_work_order_closure(
                    session,
                    identity,
                    approval_id,
                    proposal,
                    parameters,
                    now,
                    request_id=request_id,
                    idempotency_key=idempotency_key,
                )
            if proposal.tool_id == "refund.request":
                return self._submit_refund_request(
                    session,
                    identity,
                    approval_id,
                    proposal,
                    parameters,
                    now,
                    idempotency_key=idempotency_key,
                )
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == proposal.incident_id,
                )
            )
            if proposal.tool_id == "parts.reserve":
                existing = session.scalar(
                    select(PartReservationRecord).where(
                        PartReservationRecord.tenant_id == identity.tenant_id,
                        PartReservationRecord.operation_id == proposal.operation_id,
                    )
                )
                if existing is not None:
                    work = session.scalar(
                        select(WorkOrderRecord).where(
                            WorkOrderRecord.tenant_id == identity.tenant_id,
                            WorkOrderRecord.reservation_id == existing.reservation_id,
                        )
                    )
                    return ExecutionResult(
                        "SUCCEEDED", existing.reservation_id, work.work_order_id if work else None
                    )
            if incident is None or incident.version != proposal.resource_version:
                return ExecutionResult("RESOURCE_CHANGED")
            if proposal.tool_id == "customer.notify":
                if incident.status in {"CLOSED", "CANCELLED"}:
                    return ExecutionResult("RESOURCE_CHANGED")
                if proposal.tool_version == "1.1.0":
                    binding = parameters.get("contact_binding")
                    if not isinstance(binding, dict):
                        return ExecutionResult("APPROVAL_BINDING_MISMATCH")
                    try:
                        contact, current_binding = self._resolve_customer_notification_contact(
                            tenant_id=identity.tenant_id,
                            recipient_subject_id=incident.reporter_subject_id,
                            contact_point_id=str(binding.get("contact_point_id", "")),
                            channel=str(parameters.get("channel", "")),
                            now=now,
                        )
                    except CustomerNotificationAuthorityUnavailable as exc:
                        return ExecutionResult(
                            "CONTACT_AUTHORITY_UNAVAILABLE",
                            reason=exc.reason,
                        )
                    except CustomerNotificationContactConflict as exc:
                        return ExecutionResult("CONTACT_NOT_ELIGIBLE", reason=exc.reason)
                    if current_binding != binding:
                        return ExecutionResult("CONTACT_BINDING_CHANGED")
                    return self._deliver_customer_notification(
                        session,
                        identity,
                        approval_id,
                        proposal,
                        parameters,
                        contact,
                        now,
                        idempotency_key=idempotency_key,
                    )
                return self._publish_customer_notification(
                    session,
                    identity,
                    approval_id,
                    proposal,
                    parameters,
                    now,
                    idempotency_key=idempotency_key,
                )
            if proposal.tool_id == "purchase.request":
                if incident.status != "DIAGNOSED":
                    return ExecutionResult("RESOURCE_CHANGED")
                if proposal.tool_version == "1.1.0":
                    binding = parameters.get("erp_binding")
                    if not isinstance(binding, dict):
                        return ExecutionResult("APPROVAL_BINDING_MISMATCH")
                    try:
                        profile = self._resolve_procurement_profile(
                            tenant_id=identity.tenant_id,
                            now=now,
                        )
                    except PurchaseDeliveryProfileUnavailable as exc:
                        return ExecutionResult(
                            "DELIVERY_PROFILE_UNAVAILABLE",
                            reason=exc.reason,
                        )
                    if _procurement_profile_binding(profile) != binding:
                        return ExecutionResult(
                            "DELIVERY_PROFILE_CHANGED",
                            reason="enterprise_procurement_profile_changed",
                        )
                    return self._deliver_enterprise_purchase(
                        session,
                        identity,
                        approval_id,
                        proposal,
                        parameters,
                        profile,
                        now,
                        idempotency_key=idempotency_key,
                    )
                return self._submit_purchase_request(
                    session,
                    identity,
                    approval_id,
                    proposal,
                    parameters,
                    now,
                    idempotency_key=idempotency_key,
                )
            if proposal.tool_id != "parts.reserve":
                return ExecutionResult("TOOL_EXECUTION_NOT_SUPPORTED")
            if incident.status != "DIAGNOSED":
                return ExecutionResult("RESOURCE_CHANGED")
            latest_attempt = session.scalar(
                select(ExecutionAttemptRecord)
                .where(
                    ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                    ExecutionAttemptRecord.operation_id == proposal.operation_id,
                )
                .order_by(ExecutionAttemptRecord.attempt_number.desc())
            )
            if latest_attempt is not None and latest_attempt.status == "RECONCILING":
                reconciliation = self._ensure_execution_reconciliation(
                    session,
                    identity,
                    proposal,
                    approval_id=approval_id,
                    initial_attempt_id=latest_attempt.attempt_id,
                    now=now,
                )
                return ExecutionResult(
                    "RECONCILING",
                    reason="query_reconciliation_task",
                    execution_attempt_id=latest_attempt.attempt_id,
                    reconciliation_id=reconciliation.reconciliation_id,
                )
            try:
                availability = self._parts.check_availability(
                    part_number=str(parameters["part_number"]),
                    quantity=int(parameters["quantity"]),
                )
            except EnterpriseToolError as exc:
                return ExecutionResult("DEPENDENCY_UNAVAILABLE", reason=exc.reason)
            if not availability["available"]:
                return ExecutionResult("INVENTORY_UNAVAILABLE")
            attempt_number = (
                int(
                    session.scalar(
                        select(func.count())
                        .select_from(ExecutionAttemptRecord)
                        .where(
                            ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                            ExecutionAttemptRecord.operation_id == proposal.operation_id,
                        )
                    )
                    or 0
                )
                + 1
            )
            try:
                result = self._parts.reserve(
                    operation_id=proposal.operation_id,
                    part_number=str(parameters["part_number"]),
                    quantity=int(parameters["quantity"]),
                )
            except ReservationOutcomeUnknown:
                attempt_id = f"attempt-{uuid4().hex}"
                session.add(
                    ExecutionAttemptRecord(
                        attempt_id=attempt_id,
                        tenant_id=identity.tenant_id,
                        proposal_id=proposal.proposal_id,
                        approval_id=approval_id,
                        operation_id=proposal.operation_id,
                        attempt_number=attempt_number,
                        status="RECONCILING",
                        reason="external_result_unknown",
                        result={
                            "idempotency_key_digest": (
                                sha256(idempotency_key.encode()).hexdigest()
                                if idempotency_key is not None
                                else None
                            )
                        },
                        started_at=now,
                        finished_at=now,
                        created_at=now,
                        updated_at=now,
                    )
                )
                proposal.status = "RECONCILING"
                reconciliation = self._ensure_execution_reconciliation(
                    session,
                    identity,
                    proposal,
                    approval_id=approval_id,
                    initial_attempt_id=attempt_id,
                    now=now,
                )
                return ExecutionResult(
                    "RECONCILING",
                    reason="external_result_unknown",
                    execution_attempt_id=attempt_id,
                    reconciliation_id=reconciliation.reconciliation_id,
                )
            except EnterpriseToolError as exc:
                return ExecutionResult("DEPENDENCY_REJECTED", reason=exc.reason)
            return self._persist_success(
                session,
                identity,
                approval_id,
                proposal,
                result,
                attempt_number,
                now,
                idempotency_key=idempotency_key,
            )

    def _execute_part_issue(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        parameters: dict[str, Any],
        now: datetime,
        *,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        issue = session.scalar(
            select(PartIssueRecord)
            .where(
                PartIssueRecord.tenant_id == identity.tenant_id,
                PartIssueRecord.proposal_id == proposal.proposal_id,
            )
            .with_for_update()
        )
        if issue is None or issue.approval_id != approval_id:
            return ExecutionResult("APPROVAL_BINDING_MISMATCH")
        latest_attempt = session.scalar(
            select(ExecutionAttemptRecord)
            .where(
                ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                ExecutionAttemptRecord.operation_id == proposal.operation_id,
            )
            .order_by(ExecutionAttemptRecord.attempt_number.desc())
        )
        if issue.status == "ISSUED":
            return ExecutionResult(
                status="SUCCEEDED",
                reservation_id=issue.reservation_id,
                work_order_id=issue.work_order_id,
                part_issue_id=issue.part_issue_id,
                execution_attempt_id=(
                    latest_attempt.attempt_id if latest_attempt is not None else None
                ),
            )
        if issue.status == "RECONCILING":
            if latest_attempt is None or latest_attempt.status != "RECONCILING":
                return ExecutionResult(
                    "EXECUTION_STATE_INVALID",
                    part_issue_id=issue.part_issue_id,
                )
            reconciliation = self._ensure_execution_reconciliation(
                session,
                identity,
                proposal,
                approval_id=approval_id,
                initial_attempt_id=latest_attempt.attempt_id,
                now=now,
            )
            issue.reconciliation_id = reconciliation.reconciliation_id
            return ExecutionResult(
                status="RECONCILING",
                reservation_id=issue.reservation_id,
                work_order_id=issue.work_order_id,
                reason="query_reconciliation_task",
                part_issue_id=issue.part_issue_id,
                execution_attempt_id=latest_attempt.attempt_id,
                reconciliation_id=reconciliation.reconciliation_id,
            )
        if issue.status != "APPROVED":
            return ExecutionResult(
                "APPROVAL_STATE_INVALID",
                part_issue_id=issue.part_issue_id,
            )

        work = session.scalar(
            select(WorkOrderRecord)
            .where(
                WorkOrderRecord.tenant_id == identity.tenant_id,
                WorkOrderRecord.work_order_id == issue.work_order_id,
            )
            .with_for_update()
        )
        reservation = session.scalar(
            select(PartReservationRecord)
            .where(
                PartReservationRecord.tenant_id == identity.tenant_id,
                PartReservationRecord.reservation_id == issue.reservation_id,
            )
            .with_for_update()
        )
        expected_parameters = (
            {
                "incident_id": issue.incident_id,
                "asset_id": str(parameters.get("asset_id", "")),
                "work_order_id": issue.work_order_id,
                "reservation_id": issue.reservation_id,
                "part_number": issue.part_number,
                "quantity": issue.quantity,
            }
            if work is not None
            else {}
        )
        current_binding_valid = (
            work is not None
            and reservation is not None
            and work.version == proposal.resource_version
            and work.status in {"ACCEPTED", "IN_PROGRESS", "ON_HOLD"}
            and work.assigned_subject_id == proposal.initiator_subject_id
            and work.part_issue_required
            and work.incident_id == issue.incident_id
            and work.reservation_id == issue.reservation_id
            and reservation.incident_id == issue.incident_id
            and reservation.status == "RESERVED"
            and reservation.part_number == issue.part_number
            and reservation.quantity == issue.quantity
            and parameters == proposal.parameters
            and expected_parameters == proposal.parameters
        )
        if not current_binding_valid:
            issue.status = "FAILED"
            issue.reason = "resource_changed"
            issue.version += 1
            issue.updated_at = now
            proposal.status = "EXECUTION_FAILED"
            return ExecutionResult(
                "RESOURCE_CHANGED",
                work_order_id=issue.work_order_id,
                reason="part_issue_binding_changed",
                part_issue_id=issue.part_issue_id,
            )

        attempt_number = (
            int(
                session.scalar(
                    select(func.count())
                    .select_from(ExecutionAttemptRecord)
                    .where(
                        ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                        ExecutionAttemptRecord.operation_id == proposal.operation_id,
                    )
                )
                or 0
            )
            + 1
        )
        issue.submitted_at = now
        issue.updated_at = now
        try:
            result = self._parts.issue(
                operation_id=proposal.operation_id,
                reservation_id=issue.reservation_id,
                part_number=issue.part_number,
                quantity=issue.quantity,
            )
            if not _part_issue_result_matches(issue, result):
                raise PartIssueOutcomeUnknown(proposal.operation_id)
        except PartIssueOutcomeUnknown:
            attempt_id = f"attempt-{uuid4().hex}"
            session.add(
                ExecutionAttemptRecord(
                    attempt_id=attempt_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal.proposal_id,
                    approval_id=approval_id,
                    operation_id=proposal.operation_id,
                    attempt_number=attempt_number,
                    status="RECONCILING",
                    reason="external_result_unknown",
                    result={
                        "part_issue_id": issue.part_issue_id,
                        "idempotency_key_digest": (
                            sha256(idempotency_key.encode()).hexdigest()
                            if idempotency_key is not None
                            else None
                        ),
                    },
                    started_at=now,
                    finished_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            issue.status = "RECONCILING"
            issue.reason = "external_result_unknown"
            issue.version += 1
            proposal.status = "RECONCILING"
            reconciliation = self._ensure_execution_reconciliation(
                session,
                identity,
                proposal,
                approval_id=approval_id,
                initial_attempt_id=attempt_id,
                now=now,
            )
            issue.reconciliation_id = reconciliation.reconciliation_id
            return ExecutionResult(
                status="RECONCILING",
                reservation_id=issue.reservation_id,
                work_order_id=issue.work_order_id,
                reason="external_result_unknown",
                part_issue_id=issue.part_issue_id,
                execution_attempt_id=attempt_id,
                reconciliation_id=reconciliation.reconciliation_id,
            )
        except EnterpriseToolError as exc:
            attempt_id = f"attempt-{uuid4().hex}"
            session.add(
                ExecutionAttemptRecord(
                    attempt_id=attempt_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal.proposal_id,
                    approval_id=approval_id,
                    operation_id=proposal.operation_id,
                    attempt_number=attempt_number,
                    status="FAILED",
                    reason=exc.reason,
                    result={"part_issue_id": issue.part_issue_id},
                    started_at=now,
                    finished_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            issue.status = "FAILED"
            issue.reason = exc.reason
            issue.version += 1
            proposal.status = "EXECUTION_FAILED"
            return ExecutionResult(
                status="DEPENDENCY_REJECTED",
                reservation_id=issue.reservation_id,
                work_order_id=issue.work_order_id,
                reason=exc.reason,
                part_issue_id=issue.part_issue_id,
                execution_attempt_id=attempt_id,
            )
        return self._persist_part_issue_success(
            session,
            identity,
            approval_id,
            proposal,
            issue,
            reservation,
            result,
            attempt_number,
            now,
            idempotency_key=idempotency_key,
        )

    def _execute_part_movement(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        parameters: dict[str, Any],
        now: datetime,
        *,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        movement = session.scalar(
            select(PartMaterialMovementRecord)
            .where(
                PartMaterialMovementRecord.tenant_id == identity.tenant_id,
                PartMaterialMovementRecord.proposal_id == proposal.proposal_id,
            )
            .with_for_update()
        )
        if movement is None or movement.approval_id != approval_id:
            return ExecutionResult("APPROVAL_BINDING_MISMATCH")
        latest_attempt = session.scalar(
            select(ExecutionAttemptRecord)
            .where(
                ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                ExecutionAttemptRecord.operation_id == proposal.operation_id,
            )
            .order_by(ExecutionAttemptRecord.attempt_number.desc())
        )
        if movement.status == "APPLIED":
            return ExecutionResult(
                status="SUCCEEDED",
                reservation_id=movement.reservation_id,
                work_order_id=movement.work_order_id,
                part_movement_id=movement.movement_id,
                execution_attempt_id=(
                    latest_attempt.attempt_id if latest_attempt is not None else None
                ),
            )
        if movement.status == "RECONCILING":
            if latest_attempt is None or latest_attempt.status != "RECONCILING":
                return ExecutionResult(
                    "EXECUTION_STATE_INVALID",
                    part_movement_id=movement.movement_id,
                )
            reconciliation = self._ensure_execution_reconciliation(
                session,
                identity,
                proposal,
                approval_id=approval_id,
                initial_attempt_id=latest_attempt.attempt_id,
                now=now,
            )
            movement.reconciliation_id = reconciliation.reconciliation_id
            return ExecutionResult(
                status="RECONCILING",
                reservation_id=movement.reservation_id,
                work_order_id=movement.work_order_id,
                reason="query_reconciliation_task",
                part_movement_id=movement.movement_id,
                execution_attempt_id=latest_attempt.attempt_id,
                reconciliation_id=reconciliation.reconciliation_id,
            )
        if movement.status != "APPROVED":
            return ExecutionResult(
                "APPROVAL_STATE_INVALID",
                work_order_id=movement.work_order_id,
                part_movement_id=movement.movement_id,
            )

        work = session.scalar(
            select(WorkOrderRecord)
            .where(
                WorkOrderRecord.tenant_id == identity.tenant_id,
                WorkOrderRecord.work_order_id == movement.work_order_id,
            )
            .with_for_update()
        )
        incident = session.scalar(
            select(IncidentRecord).where(
                IncidentRecord.tenant_id == identity.tenant_id,
                IncidentRecord.incident_id == movement.incident_id,
            )
        )
        reservation = session.scalar(
            select(PartReservationRecord)
            .where(
                PartReservationRecord.tenant_id == identity.tenant_id,
                PartReservationRecord.reservation_id == movement.reservation_id,
            )
            .with_for_update()
        )
        issue = session.scalar(
            select(PartIssueRecord)
            .where(
                PartIssueRecord.tenant_id == identity.tenant_id,
                PartIssueRecord.part_issue_id == movement.part_issue_id,
            )
            .with_for_update()
        )
        entry = (
            session.scalar(
                select(WorkOrderFieldEntryRecord).where(
                    WorkOrderFieldEntryRecord.tenant_id == identity.tenant_id,
                    WorkOrderFieldEntryRecord.entry_id == movement.field_entry_id,
                    WorkOrderFieldEntryRecord.work_order_id == movement.work_order_id,
                    WorkOrderFieldEntryRecord.entry_type == "PART",
                )
            )
            if movement.field_entry_id is not None
            else None
        )
        expected_parameters = {
            "incident_id": movement.incident_id,
            "asset_id": incident.asset_id if incident is not None else "",
            "work_order_id": movement.work_order_id,
            "reservation_id": movement.reservation_id,
            "part_issue_id": movement.part_issue_id,
            "part_number": movement.part_number,
            **(
                {"field_entry_id": movement.field_entry_id}
                if movement.movement_kind == "CONSUME"
                else {}
            ),
            "movement_kind": movement.movement_kind,
            "quantity": movement.quantity,
        }
        binding_valid = (
            work is not None
            and incident is not None
            and reservation is not None
            and issue is not None
            and work.version == proposal.resource_version
            and work.status in {"ACCEPTED", "IN_PROGRESS", "ON_HOLD"}
            and work.assigned_subject_id == proposal.initiator_subject_id
            and work.part_accounting_required
            and work.incident_id == movement.incident_id
            and work.reservation_id == movement.reservation_id
            and reservation.status == "ISSUED"
            and reservation.incident_id == movement.incident_id
            and issue.status == "ISSUED"
            and issue.work_order_id == movement.work_order_id
            and issue.reservation_id == movement.reservation_id
            and issue.part_number == movement.part_number
            and issue.quantity == reservation.quantity
            and parameters == proposal.parameters == expected_parameters
            and (
                movement.movement_kind == "RETURN"
                or entry is not None
                and entry.payload.get("part_reservation_id") == movement.reservation_id
                and entry.payload.get("part_number") == movement.part_number
                and entry.payload.get("quantity") == movement.quantity
            )
        )
        applied_total = int(
            session.scalar(
                select(func.coalesce(func.sum(PartMaterialMovementRecord.quantity), 0)).where(
                    PartMaterialMovementRecord.tenant_id == identity.tenant_id,
                    PartMaterialMovementRecord.work_order_id == movement.work_order_id,
                    PartMaterialMovementRecord.status == "APPLIED",
                )
            )
            or 0
        )
        if not binding_valid or issue is None or applied_total + movement.quantity > issue.quantity:
            movement.status = "FAILED"
            movement.reason = "resource_changed"
            movement.version += 1
            movement.updated_at = now
            proposal.status = "EXECUTION_FAILED"
            return ExecutionResult(
                "RESOURCE_CHANGED",
                work_order_id=movement.work_order_id,
                reason="part_movement_binding_changed",
                part_movement_id=movement.movement_id,
            )

        attempt_number = (
            int(
                session.scalar(
                    select(func.count())
                    .select_from(ExecutionAttemptRecord)
                    .where(
                        ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                        ExecutionAttemptRecord.operation_id == proposal.operation_id,
                    )
                )
                or 0
            )
            + 1
        )
        movement.submitted_at = now
        movement.updated_at = now
        try:
            result = self._parts.apply_movement(
                operation_id=proposal.operation_id,
                movement_kind=movement.movement_kind,
                part_issue_id=movement.part_issue_id,
                reservation_id=movement.reservation_id,
                part_number=movement.part_number,
                quantity=movement.quantity,
            )
            if not _part_movement_result_matches(movement, result):
                raise PartMovementOutcomeUnknown(proposal.operation_id)
        except (PartMovementOutcomeUnknown, TimeoutError, OSError):
            attempt_id = f"attempt-{uuid4().hex}"
            session.add(
                ExecutionAttemptRecord(
                    attempt_id=attempt_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal.proposal_id,
                    approval_id=approval_id,
                    operation_id=proposal.operation_id,
                    attempt_number=attempt_number,
                    status="RECONCILING",
                    reason="external_result_unknown",
                    result={
                        "part_movement_id": movement.movement_id,
                        "idempotency_key_digest": (
                            sha256(idempotency_key.encode()).hexdigest()
                            if idempotency_key is not None
                            else None
                        ),
                    },
                    started_at=now,
                    finished_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            movement.status = "RECONCILING"
            movement.reason = "external_result_unknown"
            movement.version += 1
            proposal.status = "RECONCILING"
            reconciliation = self._ensure_execution_reconciliation(
                session,
                identity,
                proposal,
                approval_id=approval_id,
                initial_attempt_id=attempt_id,
                now=now,
            )
            movement.reconciliation_id = reconciliation.reconciliation_id
            return ExecutionResult(
                status="RECONCILING",
                reservation_id=movement.reservation_id,
                work_order_id=movement.work_order_id,
                reason="external_result_unknown",
                part_movement_id=movement.movement_id,
                execution_attempt_id=attempt_id,
                reconciliation_id=reconciliation.reconciliation_id,
            )
        except EnterpriseToolError as exc:
            attempt_id = f"attempt-{uuid4().hex}"
            session.add(
                ExecutionAttemptRecord(
                    attempt_id=attempt_id,
                    tenant_id=identity.tenant_id,
                    proposal_id=proposal.proposal_id,
                    approval_id=approval_id,
                    operation_id=proposal.operation_id,
                    attempt_number=attempt_number,
                    status="FAILED",
                    reason=exc.reason,
                    result={"part_movement_id": movement.movement_id},
                    started_at=now,
                    finished_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            movement.status = "FAILED"
            movement.reason = exc.reason
            movement.version += 1
            proposal.status = "EXECUTION_FAILED"
            return ExecutionResult(
                "DEPENDENCY_REJECTED",
                work_order_id=movement.work_order_id,
                reason=exc.reason,
                part_movement_id=movement.movement_id,
                execution_attempt_id=attempt_id,
            )
        return self._persist_part_movement_success(
            session,
            identity,
            approval_id,
            proposal,
            movement,
            result,
            attempt_number,
            now,
            idempotency_key=idempotency_key,
        )

    def _deliver_customer_notification(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        parameters: dict[str, Any],
        contact: NotificationContact,
        now: datetime,
        *,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        if self._notifications is None:
            return ExecutionResult(
                "CONTACT_AUTHORITY_UNAVAILABLE",
                reason="external_notification_adapter_disabled",
            )
        existing = session.scalar(
            select(CustomerNotificationDeliveryRecord).where(
                CustomerNotificationDeliveryRecord.tenant_id == identity.tenant_id,
                CustomerNotificationDeliveryRecord.operation_id == proposal.operation_id,
            )
        )
        if existing is not None:
            return self._existing_customer_notification_delivery(
                session,
                identity,
                proposal,
                approval_id=approval_id,
                delivery=existing,
                now=now,
            )
        content = parameters.get("content")
        binding = parameters.get("contact_binding")
        if not isinstance(content, dict) or not isinstance(binding, dict):
            return ExecutionResult("APPROVAL_BINDING_MISMATCH")
        message = _render_customer_notification(
            template_id=str(parameters["template_id"]),
            headline=str(content["headline"]),
            detail=str(content["detail"]),
            next_step=str(content["next_step"]),
        )
        notification_id = f"customer-notification-{uuid4().hex}"
        delivery_id = f"notification-delivery-{uuid4().hex}"
        attempt_id = f"attempt-{uuid4().hex}"
        session.add(
            CustomerServiceUpdateRecord(
                update_id=notification_id,
                tenant_id=identity.tenant_id,
                incident_id=proposal.incident_id,
                work_order_id=None,
                client_operation_id=proposal.operation_id,
                update_type="STAFF_NOTIFICATION",
                message=message,
                evidence_ids=[],
                result_accepted=None,
                satisfaction_rating=None,
                actor_subject_id=identity.subject_id,
                occurred_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        delivery = CustomerNotificationDeliveryRecord(
            delivery_id=delivery_id,
            tenant_id=identity.tenant_id,
            proposal_id=proposal.proposal_id,
            approval_id=approval_id,
            incident_id=proposal.incident_id,
            portal_update_id=notification_id,
            operation_id=proposal.operation_id,
            contact_point_id=str(binding["contact_point_id"]),
            contact_channel=str(binding["channel"]),
            source_system=str(binding["source_system"]),
            source_record_id=str(binding["source_record_id"]),
            source_version=str(binding["source_version"]),
            source_as_of=datetime.fromisoformat(str(binding["as_of"])),
            masked_destination=str(binding["masked_destination"]),
            channel=str(parameters["channel"]),
            provider=self._notifications.provider_id,
            provider_message_id=None,
            status="PENDING",
            reason=None,
            submitted_at=None,
            terminal_at=None,
            last_receipt_at=None,
            version=1,
            created_at=now,
            updated_at=now,
        )
        session.add(delivery)
        try:
            raw_result = self._notifications.deliver(
                operation_id=proposal.operation_id,
                contact=contact,
                rendered_message=message,
            )
            result = _validated_notification_delivery_result(raw_result, now=now)
        except (EnterpriseToolError, TimeoutError, OSError, ValueError):
            result = NotificationDeliveryResult(
                status="RECONCILING",
                provider_message_id=None,
                reason="external_result_unknown",
                occurred_at=now,
            )
        delivery.provider_message_id = result.provider_message_id
        delivery.reason = result.reason
        delivery.updated_at = now
        delivery.version += 1
        status = result.status
        if status == "DELIVERED":
            delivery.status = "DELIVERED"
            delivery.submitted_at = result.occurred_at
            delivery.terminal_at = result.occurred_at
            session.add(
                _notification_execution_attempt(
                    identity,
                    proposal,
                    approval_id=approval_id,
                    attempt_id=attempt_id,
                    status="SUCCEEDED",
                    reason=None,
                    delivery=delivery,
                    now=now,
                    idempotency_key=idempotency_key,
                )
            )
            proposal.status = "EXECUTED"
            return ExecutionResult(
                "SUCCEEDED",
                notification_id=notification_id,
                notification_delivery_id=delivery_id,
                execution_attempt_id=attempt_id,
            )
        if status in {"FAILED", "UNSUBSCRIBED"}:
            delivery.status = status
            delivery.submitted_at = result.occurred_at
            delivery.terminal_at = result.occurred_at
            session.add(
                _notification_execution_attempt(
                    identity,
                    proposal,
                    approval_id=approval_id,
                    attempt_id=attempt_id,
                    status="FAILED",
                    reason=result.reason or status.lower(),
                    delivery=delivery,
                    now=now,
                    idempotency_key=idempotency_key,
                )
            )
            proposal.status = "EXECUTION_FAILED"
            return ExecutionResult(
                status,
                reason=delivery.reason,
                notification_id=notification_id,
                notification_delivery_id=delivery_id,
                execution_attempt_id=attempt_id,
            )
        delivery.status = "SUBMITTED" if status == "SUBMITTED" else "RECONCILING"
        delivery.submitted_at = result.occurred_at if status == "SUBMITTED" else None
        reason = result.reason or (
            "provider_accepted" if status == "SUBMITTED" else "external_result_unknown"
        )
        session.add(
            _notification_execution_attempt(
                identity,
                proposal,
                approval_id=approval_id,
                attempt_id=attempt_id,
                status="RECONCILING",
                reason=reason,
                delivery=delivery,
                now=now,
                idempotency_key=idempotency_key,
            )
        )
        proposal.status = "RECONCILING"
        reconciliation = self._ensure_execution_reconciliation(
            session,
            identity,
            proposal,
            approval_id=approval_id,
            initial_attempt_id=attempt_id,
            now=now,
        )
        return ExecutionResult(
            "RECONCILING",
            reason=reason,
            notification_id=notification_id,
            notification_delivery_id=delivery_id,
            execution_attempt_id=attempt_id,
            reconciliation_id=reconciliation.reconciliation_id,
        )

    def _existing_customer_notification_delivery(
        self,
        session: Session,
        identity: IdentityContext,
        proposal: ActionProposalRecord,
        *,
        approval_id: str,
        delivery: CustomerNotificationDeliveryRecord,
        now: datetime,
    ) -> ExecutionResult:
        latest_attempt = session.scalar(
            select(ExecutionAttemptRecord)
            .where(
                ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                ExecutionAttemptRecord.operation_id == proposal.operation_id,
            )
            .order_by(ExecutionAttemptRecord.attempt_number.desc())
        )
        attempt_id = latest_attempt.attempt_id if latest_attempt is not None else None
        if delivery.status == "DELIVERED":
            return ExecutionResult(
                "SUCCEEDED",
                notification_id=delivery.portal_update_id,
                notification_delivery_id=delivery.delivery_id,
                execution_attempt_id=attempt_id,
            )
        if delivery.status in {"FAILED", "UNSUBSCRIBED"}:
            return ExecutionResult(
                delivery.status,
                reason=delivery.reason,
                notification_id=delivery.portal_update_id,
                notification_delivery_id=delivery.delivery_id,
                execution_attempt_id=attempt_id,
            )
        if latest_attempt is None:
            return ExecutionResult(
                "RECONCILING",
                reason="delivery_state_requires_manual_reconciliation",
                notification_id=delivery.portal_update_id,
                notification_delivery_id=delivery.delivery_id,
            )
        reconciliation = self._ensure_execution_reconciliation(
            session,
            identity,
            proposal,
            approval_id=approval_id,
            initial_attempt_id=latest_attempt.attempt_id,
            now=now,
        )
        return ExecutionResult(
            "RECONCILING",
            reason=delivery.reason or "query_reconciliation_task",
            notification_id=delivery.portal_update_id,
            notification_delivery_id=delivery.delivery_id,
            execution_attempt_id=latest_attempt.attempt_id,
            reconciliation_id=reconciliation.reconciliation_id,
        )

    def _publish_customer_notification(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        parameters: dict[str, Any],
        now: datetime,
        *,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        existing = session.scalar(
            select(CustomerServiceUpdateRecord).where(
                CustomerServiceUpdateRecord.tenant_id == identity.tenant_id,
                CustomerServiceUpdateRecord.incident_id == proposal.incident_id,
                CustomerServiceUpdateRecord.client_operation_id == proposal.operation_id,
            )
        )
        if existing is not None:
            return ExecutionResult("SUCCEEDED", notification_id=existing.update_id)
        content = parameters["content"]
        if not isinstance(content, dict):
            return ExecutionResult("APPROVAL_BINDING_MISMATCH")
        notification_id = f"customer-notification-{uuid4().hex}"
        message = _render_customer_notification(
            template_id=str(parameters["template_id"]),
            headline=str(content["headline"]),
            detail=str(content["detail"]),
            next_step=str(content["next_step"]),
        )
        attempt_number = (
            int(
                session.scalar(
                    select(func.count())
                    .select_from(ExecutionAttemptRecord)
                    .where(
                        ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                        ExecutionAttemptRecord.operation_id == proposal.operation_id,
                    )
                )
                or 0
            )
            + 1
        )
        session.add(
            CustomerServiceUpdateRecord(
                update_id=notification_id,
                tenant_id=identity.tenant_id,
                incident_id=proposal.incident_id,
                work_order_id=None,
                client_operation_id=proposal.operation_id,
                update_type="STAFF_NOTIFICATION",
                message=message,
                evidence_ids=[],
                result_accepted=None,
                satisfaction_rating=None,
                actor_subject_id=identity.subject_id,
                occurred_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            ExecutionAttemptRecord(
                attempt_id=f"attempt-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                proposal_id=proposal.proposal_id,
                approval_id=approval_id,
                operation_id=proposal.operation_id,
                attempt_number=attempt_number,
                status="SUCCEEDED",
                reason=None,
                result={
                    "notification_id": notification_id,
                    "channel": parameters["channel"],
                    "template_id": parameters["template_id"],
                    "recipient_scope": parameters["recipient_scope"],
                    "idempotency_key_digest": (
                        sha256(idempotency_key.encode()).hexdigest()
                        if idempotency_key is not None
                        else None
                    ),
                },
                started_at=now,
                finished_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        proposal.status = "EXECUTED"
        return ExecutionResult("SUCCEEDED", notification_id=notification_id)

    def _deliver_enterprise_purchase(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        parameters: dict[str, Any],
        profile: ProcurementProfile,
        now: datetime,
        *,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        existing = session.scalar(
            select(PurchaseRequestRecord).where(
                PurchaseRequestRecord.tenant_id == identity.tenant_id,
                PurchaseRequestRecord.operation_id == proposal.operation_id,
            )
        )
        if existing is not None:
            return self._existing_enterprise_purchase(
                session,
                identity,
                proposal,
                approval_id=approval_id,
                purchase=existing,
                now=now,
                idempotency_key=idempotency_key,
            )
        try:
            availability = self._parts.check_availability(
                part_number=str(parameters["part_number"]),
                quantity=int(parameters["quantity"]),
            )
            inventory = _purchase_inventory_snapshot(
                availability,
                expected_part_number=str(parameters["part_number"]),
                requested_quantity=int(parameters["quantity"]),
                now=now,
            )
        except EnterpriseToolError as exc:
            return ExecutionResult("DEPENDENCY_UNAVAILABLE", reason=exc.reason)
        except (KeyError, TypeError, ValueError):
            return ExecutionResult("DEPENDENCY_UNAVAILABLE", reason="inventory_fact_invalid")
        if bool(availability["available"]):
            return ExecutionResult("PURCHASE_NOT_REQUIRED")
        unit_cost = int(parameters["estimated_unit_cost_minor"])
        quantity = int(parameters["quantity"])
        if int(parameters["estimated_total_cost_minor"]) != unit_cost * quantity:
            return ExecutionResult("APPROVAL_BINDING_MISMATCH")
        adapter = self._procurement
        if adapter is None:
            return ExecutionResult(
                "DELIVERY_PROFILE_UNAVAILABLE",
                reason="enterprise_procurement_adapter_disabled",
            )
        purchase_request_id = f"purchase-request-{uuid4().hex}"
        attempt_id = f"attempt-{uuid4().hex}"
        purchase = PurchaseRequestRecord(
            purchase_request_id=purchase_request_id,
            tenant_id=identity.tenant_id,
            operation_id=proposal.operation_id,
            proposal_id=proposal.proposal_id,
            incident_id=proposal.incident_id,
            asset_id=str(parameters["asset_id"]),
            part_number=str(parameters["part_number"]),
            quantity=quantity,
            available_quantity_at_submission=int(inventory["available_quantity"]),
            estimated_unit_cost_minor=unit_cost,
            estimated_total_cost_minor=unit_cost * quantity,
            currency=str(parameters["currency"]),
            cost_center=str(parameters["cost_center"]),
            justification=str(parameters["justification"]),
            delivery_target="ENTERPRISE_ERP",
            provider=profile.provider,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            profile_display_name=profile.display_name,
            profile_as_of=profile.as_of,
            profile_expires_at=profile.expires_at,
            external_request_id=None,
            status="SUBMITTING",
            reason=None,
            submitted_at=None,
            terminal_at=None,
            last_receipt_at=None,
            version=1,
            requested_by_subject_id=proposal.initiator_subject_id,
            inventory_source=str(inventory["source"]),
            inventory_source_record_id=str(inventory["source_record_id"]),
            inventory_as_of=datetime.fromisoformat(
                str(inventory["as_of"]).replace("Z", "+00:00")
            ),
            created_at=now,
            updated_at=now,
        )
        session.add(purchase)
        session.flush()
        payload: dict[str, object] = {
            "incident_id": proposal.incident_id,
            "asset_id": str(parameters["asset_id"]),
            "part_number": str(parameters["part_number"]),
            "quantity": quantity,
            "estimated_unit_cost_minor": unit_cost,
            "estimated_total_cost_minor": unit_cost * quantity,
            "currency": str(parameters["currency"]),
            "cost_center": str(parameters["cost_center"]),
            "justification": str(parameters["justification"]),
        }
        try:
            raw_result = adapter.submit(
                operation_id=proposal.operation_id,
                profile=profile,
                payload=payload,
            )
            result = _validated_procurement_delivery_result(raw_result, now=now)
        except (EnterpriseToolError, TimeoutError, OSError, TypeError, ValueError):
            result = ProcurementDeliveryResult(
                status="RECONCILING",
                external_request_id=None,
                reason="external_result_unknown",
                occurred_at=now,
            )
        purchase.status = result.status
        purchase.external_request_id = result.external_request_id
        purchase.reason = result.reason
        purchase.submitted_at = result.occurred_at
        purchase.version += 1
        purchase.updated_at = now
        if result.status == "ACCEPTED":
            purchase.terminal_at = result.occurred_at
            session.add(
                _procurement_execution_attempt(
                    identity.tenant_id,
                    proposal,
                    approval_id=approval_id,
                    attempt_id=attempt_id,
                    status="SUCCEEDED",
                    reason=None,
                    purchase=purchase,
                    now=now,
                    idempotency_key=idempotency_key,
                )
            )
            proposal.status = "EXECUTED"
            return ExecutionResult(
                "SUCCEEDED",
                purchase_request_id=purchase.purchase_request_id,
                execution_attempt_id=attempt_id,
            )
        if result.status in {"REJECTED", "CANCELLED"}:
            purchase.terminal_at = result.occurred_at
            session.add(
                _procurement_execution_attempt(
                    identity.tenant_id,
                    proposal,
                    approval_id=approval_id,
                    attempt_id=attempt_id,
                    status="FAILED",
                    reason=result.reason or result.status.lower(),
                    purchase=purchase,
                    now=now,
                    idempotency_key=idempotency_key,
                )
            )
            proposal.status = "EXECUTION_FAILED"
            return ExecutionResult(
                result.status,
                reason=purchase.reason,
                purchase_request_id=purchase.purchase_request_id,
                execution_attempt_id=attempt_id,
            )
        purchase.status = "RECONCILING"
        reason = result.reason or "external_result_unknown"
        session.add(
            _procurement_execution_attempt(
                identity.tenant_id,
                proposal,
                approval_id=approval_id,
                attempt_id=attempt_id,
                status="RECONCILING",
                reason=reason,
                purchase=purchase,
                now=now,
                idempotency_key=idempotency_key,
            )
        )
        proposal.status = "RECONCILING"
        reconciliation = self._ensure_execution_reconciliation(
            session,
            identity,
            proposal,
            approval_id=approval_id,
            initial_attempt_id=attempt_id,
            now=now,
        )
        return ExecutionResult(
            "RECONCILING",
            reason=reason,
            purchase_request_id=purchase.purchase_request_id,
            execution_attempt_id=attempt_id,
            reconciliation_id=reconciliation.reconciliation_id,
        )

    def _existing_enterprise_purchase(
        self,
        session: Session,
        identity: IdentityContext,
        proposal: ActionProposalRecord,
        *,
        approval_id: str,
        purchase: PurchaseRequestRecord,
        now: datetime,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        latest_attempt = session.scalar(
            select(ExecutionAttemptRecord)
            .where(
                ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                ExecutionAttemptRecord.operation_id == proposal.operation_id,
            )
            .order_by(ExecutionAttemptRecord.attempt_number.desc())
        )
        if purchase.status == "ACCEPTED":
            return ExecutionResult(
                "SUCCEEDED",
                purchase_request_id=purchase.purchase_request_id,
                execution_attempt_id=latest_attempt.attempt_id if latest_attempt else None,
            )
        if purchase.status in {"REJECTED", "CANCELLED"}:
            return ExecutionResult(
                purchase.status,
                reason=purchase.reason,
                purchase_request_id=purchase.purchase_request_id,
                execution_attempt_id=latest_attempt.attempt_id if latest_attempt else None,
            )
        if latest_attempt is None:
            latest_attempt = _procurement_execution_attempt(
                identity.tenant_id,
                proposal,
                approval_id=approval_id,
                attempt_id=f"attempt-{uuid4().hex}",
                status="RECONCILING",
                reason="delivery_state_requires_manual_reconciliation",
                purchase=purchase,
                now=now,
                idempotency_key=idempotency_key,
            )
            session.add(latest_attempt)
            purchase.status = "RECONCILING"
            purchase.reason = "delivery_state_requires_manual_reconciliation"
            purchase.version += 1
            purchase.updated_at = now
            proposal.status = "RECONCILING"
        reconciliation = self._ensure_execution_reconciliation(
            session,
            identity,
            proposal,
            approval_id=approval_id,
            initial_attempt_id=latest_attempt.attempt_id,
            now=now,
        )
        return ExecutionResult(
            "RECONCILING",
            reason=purchase.reason or "query_reconciliation_task",
            purchase_request_id=purchase.purchase_request_id,
            execution_attempt_id=latest_attempt.attempt_id,
            reconciliation_id=reconciliation.reconciliation_id,
        )

    def _submit_purchase_request(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        parameters: dict[str, Any],
        now: datetime,
        *,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        existing = session.scalar(
            select(PurchaseRequestRecord).where(
                PurchaseRequestRecord.tenant_id == identity.tenant_id,
                PurchaseRequestRecord.operation_id == proposal.operation_id,
            )
        )
        if existing is not None:
            return ExecutionResult(
                "SUCCEEDED",
                purchase_request_id=existing.purchase_request_id,
            )
        try:
            availability = self._parts.check_availability(
                part_number=str(parameters["part_number"]),
                quantity=int(parameters["quantity"]),
            )
            inventory = _purchase_inventory_snapshot(
                availability,
                expected_part_number=str(parameters["part_number"]),
                requested_quantity=int(parameters["quantity"]),
                now=now,
            )
        except EnterpriseToolError as exc:
            return ExecutionResult("DEPENDENCY_UNAVAILABLE", reason=exc.reason)
        except (KeyError, TypeError, ValueError):
            return ExecutionResult("DEPENDENCY_UNAVAILABLE", reason="inventory_fact_invalid")
        if bool(availability["available"]):
            return ExecutionResult("PURCHASE_NOT_REQUIRED")
        purchase_request_id = f"purchase-request-{uuid4().hex}"
        unit_cost = int(parameters["estimated_unit_cost_minor"])
        quantity = int(parameters["quantity"])
        if int(parameters["estimated_total_cost_minor"]) != unit_cost * quantity:
            return ExecutionResult("APPROVAL_BINDING_MISMATCH")
        session.add(
            PurchaseRequestRecord(
                purchase_request_id=purchase_request_id,
                tenant_id=identity.tenant_id,
                operation_id=proposal.operation_id,
                proposal_id=proposal.proposal_id,
                incident_id=proposal.incident_id,
                asset_id=str(parameters["asset_id"]),
                part_number=str(parameters["part_number"]),
                quantity=quantity,
                available_quantity_at_submission=int(inventory["available_quantity"]),
                estimated_unit_cost_minor=unit_cost,
                estimated_total_cost_minor=unit_cost * quantity,
                currency=str(parameters["currency"]),
                cost_center=str(parameters["cost_center"]),
                justification=str(parameters["justification"]),
                status="SUBMITTED",
                requested_by_subject_id=proposal.initiator_subject_id,
                inventory_source=str(inventory["source"]),
                inventory_source_record_id=str(inventory["source_record_id"]),
                inventory_as_of=datetime.fromisoformat(
                    str(inventory["as_of"]).replace("Z", "+00:00")
                ),
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            ExecutionAttemptRecord(
                attempt_id=f"attempt-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                proposal_id=proposal.proposal_id,
                approval_id=approval_id,
                operation_id=proposal.operation_id,
                attempt_number=1,
                status="SUCCEEDED",
                reason=None,
                result={
                    "purchase_request_id": purchase_request_id,
                    "status": "SUBMITTED",
                    "inventory_source": inventory["source"],
                    "inventory_as_of": inventory["as_of"],
                    "idempotency_key_digest": (
                        sha256(idempotency_key.encode()).hexdigest()
                        if idempotency_key is not None
                        else None
                    ),
                },
                started_at=now,
                finished_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        proposal.status = "EXECUTED"
        return ExecutionResult(
            "SUCCEEDED",
            purchase_request_id=purchase_request_id,
        )

    def _deliver_enterprise_refund(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        parameters: dict[str, Any],
        profile: RefundProfile,
        confirmation: CustomerServiceUpdateRecord,
        now: datetime,
        *,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        existing = session.scalar(
            select(RefundRequestRecord).where(
                RefundRequestRecord.tenant_id == identity.tenant_id,
                RefundRequestRecord.operation_id == proposal.operation_id,
            )
        )
        if existing is not None:
            return self._existing_enterprise_refund(
                session,
                identity,
                proposal,
                approval_id=approval_id,
                refund=existing,
                now=now,
                idempotency_key=idempotency_key,
            )
        adapter = self._refunds
        if adapter is None:
            return ExecutionResult(
                "DELIVERY_PROFILE_UNAVAILABLE",
                reason="enterprise_refund_adapter_disabled",
            )
        refund = RefundRequestRecord(
            refund_request_id=f"refund-request-{uuid4().hex}",
            tenant_id=identity.tenant_id,
            operation_id=proposal.operation_id,
            proposal_id=proposal.proposal_id,
            incident_id=proposal.incident_id,
            work_order_id=str(parameters["work_order_id"]),
            asset_id=str(parameters["asset_id"]),
            customer_subject_id=str(parameters["customer_subject_id"]),
            customer_confirmation_update_id=confirmation.update_id,
            amount_minor=int(parameters["amount_minor"]),
            currency=str(parameters["currency"]),
            cost_center=str(parameters["cost_center"]),
            reason=str(parameters["reason"]),
            compensation_policy_id=str(parameters["compensation_policy_id"]),
            reason_code=str(parameters["reason_code"]),
            delivery_target="ENTERPRISE_FINANCE",
            provider=profile.provider,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            profile_display_name=profile.display_name,
            profile_as_of=profile.as_of,
            profile_expires_at=profile.expires_at,
            external_request_id=None,
            status="SUBMITTING",
            payment_status="NOT_STARTED",
            provider_reason=None,
            submitted_at=None,
            terminal_at=None,
            last_receipt_at=None,
            version=1,
            requested_by_subject_id=proposal.initiator_subject_id,
            created_at=now,
            updated_at=now,
        )
        session.add(refund)
        session.flush()
        payload: dict[str, object] = {
            "refund_request_id": refund.refund_request_id,
            "incident_id": proposal.incident_id,
            "work_order_id": refund.work_order_id,
            "customer_subject_id": refund.customer_subject_id,
            "amount_minor": refund.amount_minor,
            "currency": refund.currency,
            "cost_center": refund.cost_center,
            "reason_code": refund.reason_code or "OTHER_APPROVED_COMPENSATION",
            "compensation_policy_id": refund.compensation_policy_id,
        }
        try:
            raw_result = adapter.submit(
                operation_id=proposal.operation_id,
                profile=profile,
                payload=payload,
            )
            result = _validated_refund_delivery_result(raw_result, now=now)
        except (EnterpriseToolError, TimeoutError, OSError, TypeError, ValueError):
            result = RefundDeliveryResult(
                status="RECONCILING",
                payment_status="NOT_STARTED",
                external_request_id=None,
                reason="external_result_unknown",
                occurred_at=now,
            )
        refund.status = result.status
        refund.payment_status = "NOT_STARTED"
        refund.external_request_id = result.external_request_id
        refund.provider_reason = result.reason
        refund.submitted_at = result.occurred_at
        refund.version += 1
        refund.updated_at = now
        attempt_id = f"attempt-{uuid4().hex}"
        if result.status == "ACCEPTED":
            refund.terminal_at = result.occurred_at
            session.add(
                _refund_execution_attempt(
                    identity.tenant_id,
                    proposal,
                    approval_id=approval_id,
                    attempt_id=attempt_id,
                    status="SUCCEEDED",
                    reason=result.reason,
                    refund=refund,
                    now=now,
                    idempotency_key=idempotency_key,
                )
            )
            proposal.status = "EXECUTED"
            return ExecutionResult(
                "SUCCEEDED",
                refund_request_id=refund.refund_request_id,
                execution_attempt_id=attempt_id,
            )
        if result.status in {"REJECTED", "CANCELLED"}:
            refund.terminal_at = result.occurred_at
            session.add(
                _refund_execution_attempt(
                    identity.tenant_id,
                    proposal,
                    approval_id=approval_id,
                    attempt_id=attempt_id,
                    status="FAILED",
                    reason=result.reason or result.status.lower(),
                    refund=refund,
                    now=now,
                    idempotency_key=idempotency_key,
                )
            )
            proposal.status = "EXECUTION_FAILED"
            return ExecutionResult(
                result.status,
                reason=refund.provider_reason,
                refund_request_id=refund.refund_request_id,
                execution_attempt_id=attempt_id,
            )
        refund.status = "RECONCILING"
        reason = result.reason or "external_result_unknown"
        session.add(
            _refund_execution_attempt(
                identity.tenant_id,
                proposal,
                approval_id=approval_id,
                attempt_id=attempt_id,
                status="RECONCILING",
                reason=reason,
                refund=refund,
                now=now,
                idempotency_key=idempotency_key,
            )
        )
        proposal.status = "RECONCILING"
        reconciliation = self._ensure_execution_reconciliation(
            session,
            identity,
            proposal,
            approval_id=approval_id,
            initial_attempt_id=attempt_id,
            now=now,
        )
        return ExecutionResult(
            "RECONCILING",
            reason=reason,
            refund_request_id=refund.refund_request_id,
            execution_attempt_id=attempt_id,
            reconciliation_id=reconciliation.reconciliation_id,
        )

    def _existing_enterprise_refund(
        self,
        session: Session,
        identity: IdentityContext,
        proposal: ActionProposalRecord,
        *,
        approval_id: str,
        refund: RefundRequestRecord,
        now: datetime,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        latest_attempt = session.scalar(
            select(ExecutionAttemptRecord)
            .where(
                ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                ExecutionAttemptRecord.operation_id == proposal.operation_id,
            )
            .order_by(ExecutionAttemptRecord.attempt_number.desc())
        )
        attempt_id = latest_attempt.attempt_id if latest_attempt else None
        if refund.status == "ACCEPTED":
            return ExecutionResult(
                "SUCCEEDED",
                refund_request_id=refund.refund_request_id,
                execution_attempt_id=attempt_id,
            )
        if refund.status in {"REJECTED", "CANCELLED"}:
            return ExecutionResult(
                refund.status,
                reason=refund.provider_reason,
                refund_request_id=refund.refund_request_id,
                execution_attempt_id=attempt_id,
            )
        if latest_attempt is None:
            latest_attempt = _refund_execution_attempt(
                identity.tenant_id,
                proposal,
                approval_id=approval_id,
                attempt_id=f"attempt-{uuid4().hex}",
                status="RECONCILING",
                reason="delivery_state_requires_manual_reconciliation",
                refund=refund,
                now=now,
                idempotency_key=idempotency_key,
            )
            session.add(latest_attempt)
            refund.status = "RECONCILING"
            refund.provider_reason = "delivery_state_requires_manual_reconciliation"
            refund.version += 1
            refund.updated_at = now
            proposal.status = "RECONCILING"
        reconciliation = self._ensure_execution_reconciliation(
            session,
            identity,
            proposal,
            approval_id=approval_id,
            initial_attempt_id=latest_attempt.attempt_id,
            now=now,
        )
        return ExecutionResult(
            "RECONCILING",
            reason=refund.provider_reason or "query_reconciliation_task",
            refund_request_id=refund.refund_request_id,
            execution_attempt_id=latest_attempt.attempt_id,
            reconciliation_id=reconciliation.reconciliation_id,
        )

    def _submit_refund_request(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        parameters: dict[str, Any],
        now: datetime,
        *,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        work_order_id = str(parameters["work_order_id"])
        work = session.scalar(
            select(WorkOrderRecord)
            .where(
                WorkOrderRecord.tenant_id == identity.tenant_id,
                WorkOrderRecord.work_order_id == work_order_id,
            )
            .with_for_update()
        )
        if work is None:
            return ExecutionResult("RESOURCE_CHANGED")
        existing = session.scalar(
            select(RefundRequestRecord).where(
                RefundRequestRecord.tenant_id == identity.tenant_id,
                RefundRequestRecord.operation_id == proposal.operation_id,
            )
        )
        if existing is not None:
            if existing.delivery_target == "ENTERPRISE_FINANCE":
                return self._existing_enterprise_refund(
                    session,
                    identity,
                    proposal,
                    approval_id=approval_id,
                    refund=existing,
                    now=now,
                    idempotency_key=idempotency_key,
                )
            return ExecutionResult(
                "SUCCEEDED",
                refund_request_id=existing.refund_request_id,
            )
        another = session.scalar(
            select(RefundRequestRecord.refund_request_id).where(
                RefundRequestRecord.tenant_id == identity.tenant_id,
                RefundRequestRecord.work_order_id == work_order_id,
            )
        )
        if another is not None:
            return ExecutionResult("REFUND_ALREADY_REQUESTED")
        if work.version != proposal.resource_version or work.status not in {
            "COMPLETED",
            "VERIFIED",
            "CLOSED",
        }:
            return ExecutionResult("RESOURCE_CHANGED")
        confirmation = _latest_customer_result_confirmation(
            session,
            tenant_id=identity.tenant_id,
            work_order_id=work_order_id,
        )
        if (
            not _refund_eligible(confirmation)
            or confirmation is None
            or confirmation.update_id != str(parameters["customer_confirmation_update_id"])
            or confirmation.result_accepted != bool(parameters["customer_result_accepted"])
            or confirmation.satisfaction_rating != int(parameters["satisfaction_rating"])
        ):
            return ExecutionResult("REFUND_ELIGIBILITY_CHANGED")
        if parameters.get("delivery_target") == "ENTERPRISE_FINANCE":
            binding = parameters.get("finance_binding")
            try:
                profile = self._resolve_refund_profile(
                    tenant_id=identity.tenant_id,
                    customer_subject_id=str(parameters["customer_subject_id"]),
                    now=now,
                )
            except RefundDeliveryProfileUnavailable as exc:
                return ExecutionResult(
                    "DELIVERY_PROFILE_CHANGED",
                    reason=exc.reason,
                )
            if _refund_profile_binding(profile) != binding:
                return ExecutionResult(
                    "DELIVERY_PROFILE_CHANGED",
                    reason="enterprise_refund_profile_changed",
                )
            return self._deliver_enterprise_refund(
                session,
                identity,
                approval_id,
                proposal,
                parameters,
                profile,
                confirmation,
                now,
                idempotency_key=idempotency_key,
            )
        refund_request_id = f"refund-request-{uuid4().hex}"
        session.add(
            RefundRequestRecord(
                refund_request_id=refund_request_id,
                tenant_id=identity.tenant_id,
                operation_id=proposal.operation_id,
                proposal_id=proposal.proposal_id,
                incident_id=proposal.incident_id,
                work_order_id=work_order_id,
                asset_id=str(parameters["asset_id"]),
                customer_subject_id=str(parameters["customer_subject_id"]),
                customer_confirmation_update_id=confirmation.update_id,
                amount_minor=int(parameters["amount_minor"]),
                currency=str(parameters["currency"]),
                cost_center=str(parameters["cost_center"]),
                reason=str(parameters["reason"]),
                compensation_policy_id=str(parameters["compensation_policy_id"]),
                reason_code=None,
                delivery_target="INTERNAL",
                provider=None,
                profile_id=None,
                profile_version=None,
                profile_display_name=None,
                profile_as_of=None,
                profile_expires_at=None,
                external_request_id=None,
                status="SUBMITTED",
                payment_status="NOT_APPLICABLE",
                provider_reason=None,
                submitted_at=now,
                terminal_at=None,
                last_receipt_at=None,
                version=1,
                requested_by_subject_id=proposal.initiator_subject_id,
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            ExecutionAttemptRecord(
                attempt_id=f"attempt-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                proposal_id=proposal.proposal_id,
                approval_id=approval_id,
                operation_id=proposal.operation_id,
                attempt_number=1,
                status="SUCCEEDED",
                reason=None,
                result={
                    "refund_request_id": refund_request_id,
                    "status": "SUBMITTED",
                    "work_order_id": work_order_id,
                    "customer_confirmation_update_id": confirmation.update_id,
                    "idempotency_key_digest": (
                        sha256(idempotency_key.encode()).hexdigest()
                        if idempotency_key is not None
                        else None
                    ),
                },
                started_at=now,
                finished_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        proposal.status = "EXECUTED"
        return ExecutionResult(
            "SUCCEEDED",
            refund_request_id=refund_request_id,
        )

    def _execute_work_order_assignment(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        parameters: dict[str, Any],
        now: datetime,
        *,
        request_id: str,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        if proposal.tool_version == "1.1.0":
            binding = parameters.get("fsm_binding")
            if not isinstance(binding, dict):
                return ExecutionResult(
                    "APPROVAL_BINDING_MISMATCH",
                    reason="fsm_assignment_binding_missing",
                )
            existing_delivery = session.scalar(
                select(FsmAssignmentDeliveryRecord).where(
                    FsmAssignmentDeliveryRecord.tenant_id == identity.tenant_id,
                    FsmAssignmentDeliveryRecord.operation_id == proposal.operation_id,
                )
            )
            if existing_delivery is not None:
                return self._existing_fsm_assignment_delivery(
                    session,
                    identity,
                    proposal,
                    approval_id=approval_id,
                    delivery=existing_delivery,
                    now=now,
                )
            work_order_id = str(parameters["work_order_id"])
            row = (
                session.execute(
                    select(WorkOrderRecord, IncidentRecord, AssetSiteLinkRecord.site_id)
                    .join(IncidentRecord, IncidentRecord.incident_id == WorkOrderRecord.incident_id)
                    .outerjoin(
                        AssetSiteLinkRecord,
                        AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                    )
                    .where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.work_order_id == work_order_id,
                    )
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                return ExecutionResult("RESOURCE_CHANGED")
            work, incident, site_id = row
            if (
                work.status != "READY"
                or work.version != proposal.resource_version
                or site_id is None
                or site_id != parameters.get("site_id")
            ):
                return ExecutionResult("RESOURCE_CHANGED")
            try:
                candidates = self._resolve_fsm_assignment_candidates(
                    tenant_id=identity.tenant_id,
                    work_order_id=work_order_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                    service_window_start=work.service_window_start,
                    service_window_end=work.service_window_end,
                    now=now,
                )
            except FsmAssignmentProfileUnavailable as exc:
                return ExecutionResult("DEPENDENCY_UNAVAILABLE", reason=exc.reason)
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.candidate_profile_id
                    == binding.get("candidate_profile_id")
                ),
                None,
            )
            if selected is None:
                return ExecutionResult(
                    "DEPENDENCY_UNAVAILABLE",
                    reason="fsm_assignment_candidate_not_available",
                )
            if not is_eligible_field_engineer(
                session,
                tenant_id=identity.tenant_id,
                subject_id=selected.assignee_subject_id,
                asset_id=incident.asset_id,
                site_id=site_id,
            ):
                return ExecutionResult(
                    "DEPENDENCY_UNAVAILABLE",
                    reason="fsm_assignment_candidate_not_eligible",
                )
            if _fsm_assignment_candidate_binding(selected) != binding:
                return ExecutionResult(
                    "DEPENDENCY_UNAVAILABLE",
                    reason="fsm_assignment_candidate_changed",
                )
            return self._deliver_enterprise_fsm_assignment(
                session,
                identity,
                approval_id,
                proposal,
                parameters,
                selected,
                now,
                request_id=request_id,
                idempotency_key=idempotency_key,
            )
        existing = session.scalar(
            select(WorkOrderAssignmentRecord).where(
                WorkOrderAssignmentRecord.tenant_id == identity.tenant_id,
                WorkOrderAssignmentRecord.operation_id == proposal.operation_id,
            )
        )
        if existing is not None:
            return ExecutionResult("SUCCEEDED", work_order_id=existing.work_order_id)
        work_order_id = str(parameters["work_order_id"])
        try:
            assigned = WorkOrderService(self._database, self._authorizer).assign_in_session(
                session,
                identity,
                work_order_id,
                assignee_subject_id=str(parameters["assignee_subject_id"]),
                expected_version=proposal.resource_version,
                request_id=request_id,
                reason=str(parameters["reason"]),
                operation_id=proposal.operation_id,
                proposal_id=proposal.proposal_id,
            )
        except WorkOrderConflict as exc:
            return ExecutionResult("RESOURCE_CHANGED", reason=exc.reason)
        session.add(
            ExecutionAttemptRecord(
                attempt_id=f"attempt-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                proposal_id=proposal.proposal_id,
                approval_id=approval_id,
                operation_id=proposal.operation_id,
                attempt_number=1,
                status="SUCCEEDED",
                reason=None,
                result={
                    "work_order_id": assigned.work_order_id,
                    "assignee_subject_id": assigned.assigned_subject_id,
                    "work_order_version": assigned.version,
                    "idempotency_key_digest": (
                        sha256(idempotency_key.encode()).hexdigest()
                        if idempotency_key is not None
                        else None
                    ),
                },
                started_at=now,
                finished_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        proposal.status = "EXECUTED"
        return ExecutionResult("SUCCEEDED", work_order_id=assigned.work_order_id)

    def _deliver_enterprise_fsm_assignment(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        parameters: dict[str, Any],
        candidate: FsmAssignmentCandidate,
        now: datetime,
        *,
        request_id: str,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        adapter = self._fsm_assignments
        if adapter is None:
            return ExecutionResult(
                "DEPENDENCY_UNAVAILABLE", reason="fsm_assignment_adapter_disabled"
            )
        existing = session.scalar(
            select(FsmAssignmentDeliveryRecord).where(
                FsmAssignmentDeliveryRecord.tenant_id == identity.tenant_id,
                FsmAssignmentDeliveryRecord.operation_id == proposal.operation_id,
            )
        )
        if existing is not None:
            return self._existing_fsm_assignment_delivery(
                session,
                identity,
                proposal,
                approval_id=approval_id,
                delivery=existing,
                now=now,
            )
        delivery = FsmAssignmentDeliveryRecord(
            delivery_id=f"fsm-delivery-{uuid4().hex}",
            tenant_id=identity.tenant_id,
            operation_id=proposal.operation_id,
            proposal_id=proposal.proposal_id,
            approval_id=approval_id,
            work_order_id=str(parameters["work_order_id"]),
            incident_id=proposal.incident_id,
            provider=candidate.provider,
            candidate_profile_id=candidate.candidate_profile_id,
            profile_version=candidate.profile_version,
            assignee_subject_id=candidate.assignee_subject_id,
            candidate_binding=_fsm_assignment_candidate_binding(candidate),
            external_assignment_id=None,
            status="SUBMITTING",
            reason=None,
            submitted_at=now,
            terminal_at=None,
            last_receipt_at=None,
            version=1,
            created_at=now,
            updated_at=now,
        )
        session.add(delivery)
        session.flush()
        payload: dict[str, object] = {
            "work_order_id": str(parameters["work_order_id"]),
            "incident_id": proposal.incident_id,
            "asset_id": str(parameters["asset_id"]),
            "site_id": str(parameters["site_id"]),
            "assignee_subject_id": candidate.assignee_subject_id,
            "candidate_profile_id": candidate.candidate_profile_id,
            "profile_version": candidate.profile_version,
            "service_window_start": candidate.service_window_start.isoformat(),
            "service_window_end": candidate.service_window_end.isoformat(),
        }
        try:
            raw_result = adapter.submit(
                operation_id=proposal.operation_id,
                candidate=candidate,
                payload=payload,
            )
            result = _validated_fsm_assignment_delivery_result(raw_result, now=now)
        except (EnterpriseToolError, TimeoutError, OSError, TypeError, ValueError):
            result = FsmAssignmentDeliveryResult(
                status="RECONCILING",
                external_assignment_id=None,
                reason="fsm_assignment_result_unknown",
                occurred_at=now,
            )
        if result.status == "ACCEPTED":
            assert result.external_assignment_id is not None
            delivery.status = "ACCEPTED"
            delivery.external_assignment_id = result.external_assignment_id
            delivery.reason = result.reason
            delivery.terminal_at = result.occurred_at
            delivery.version += 1
            delivery.updated_at = now
            try:
                assigned = WorkOrderService(
                    self._database, self._authorizer
                ).assign_in_session(
                    session,
                    identity,
                    delivery.work_order_id,
                    assignee_subject_id=candidate.assignee_subject_id,
                    expected_version=proposal.resource_version,
                    request_id=request_id,
                    reason=str(parameters["reason"]),
                    operation_id=proposal.operation_id,
                    proposal_id=proposal.proposal_id,
                    fsm_assignment_delivery_id=delivery.delivery_id,
                )
            except WorkOrderConflict as exc:
                return self._fsm_assignment_manual_escalation(
                    session,
                    identity,
                    proposal,
                    approval_id=approval_id,
                    delivery=delivery,
                    now=now,
                    reason=exc.reason,
                )
            attempt = self._fsm_assignment_attempt(
                identity.tenant_id,
                proposal,
                approval_id=approval_id,
                delivery=delivery,
                status="SUCCEEDED",
                reason=result.reason,
                now=now,
                idempotency_key=idempotency_key,
            )
            session.add(attempt)
            proposal.status = "EXECUTED"
            return ExecutionResult(
                "SUCCEEDED",
                work_order_id=assigned.work_order_id,
                assignment_delivery_id=delivery.delivery_id,
                execution_attempt_id=attempt.attempt_id,
            )
        if result.status in {"REJECTED", "CANCELLED"}:
            delivery.status = result.status
            delivery.external_assignment_id = result.external_assignment_id
            delivery.reason = result.reason
            delivery.terminal_at = result.occurred_at
            delivery.version += 1
            delivery.updated_at = now
            attempt = self._fsm_assignment_attempt(
                identity.tenant_id,
                proposal,
                approval_id=approval_id,
                delivery=delivery,
                status="FAILED",
                reason=result.reason,
                now=now,
                idempotency_key=idempotency_key,
            )
            session.add(attempt)
            proposal.status = "EXECUTION_FAILED"
            return ExecutionResult(
                result.status,
                work_order_id=delivery.work_order_id,
                reason=result.reason,
                assignment_delivery_id=delivery.delivery_id,
                execution_attempt_id=attempt.attempt_id,
            )
        delivery.status = "RECONCILING"
        delivery.external_assignment_id = result.external_assignment_id
        delivery.reason = result.reason or "fsm_assignment_result_unknown"
        delivery.version += 1
        delivery.updated_at = now
        work = session.scalar(
            select(WorkOrderRecord)
            .where(
                WorkOrderRecord.tenant_id == identity.tenant_id,
                WorkOrderRecord.work_order_id == delivery.work_order_id,
            )
            .with_for_update()
        )
        if (
            work is None
            or work.status != "READY"
            or work.version != proposal.resource_version
            or work.pending_assignment_operation_id not in {
                None,
                proposal.operation_id,
            }
        ):
            return self._fsm_assignment_manual_escalation(
                session,
                identity,
                proposal,
                approval_id=approval_id,
                delivery=delivery,
                now=now,
                reason="fsm_assignment_pending_gate_conflict",
            )
        if work.pending_assignment_operation_id is None:
            work.pending_assignment_operation_id = proposal.operation_id
            work.version += 1
            work.updated_at = now
            session.add(
                WorkOrderEventRecord(
                    event_id=f"work-event-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    work_order_id=work.work_order_id,
                    sequence=work.version,
                    event_type="work_order.assignment_pending",
                    actor_subject_id=identity.subject_id,
                    payload={
                        "operation_id": proposal.operation_id,
                        "delivery_id": delivery.delivery_id,
                        "candidate_profile_id": delivery.candidate_profile_id,
                    },
                    occurred_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
        attempt = self._fsm_assignment_attempt(
            identity.tenant_id,
            proposal,
            approval_id=approval_id,
            delivery=delivery,
            status="RECONCILING",
            reason=delivery.reason,
            now=now,
            idempotency_key=idempotency_key,
        )
        session.add(attempt)
        proposal.status = "RECONCILING"
        reconciliation = self._ensure_execution_reconciliation(
            session,
            identity,
            proposal,
            approval_id=approval_id,
            initial_attempt_id=attempt.attempt_id,
            now=now,
        )
        return ExecutionResult(
            "RECONCILING",
            work_order_id=delivery.work_order_id,
            reason=delivery.reason,
            assignment_delivery_id=delivery.delivery_id,
            execution_attempt_id=attempt.attempt_id,
            reconciliation_id=reconciliation.reconciliation_id,
        )

    def _existing_fsm_assignment_delivery(
        self,
        session: Session,
        identity: IdentityContext,
        proposal: ActionProposalRecord,
        *,
        approval_id: str,
        delivery: FsmAssignmentDeliveryRecord,
        now: datetime,
    ) -> ExecutionResult:
        latest_attempt = session.scalar(
            select(ExecutionAttemptRecord)
            .where(
                ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                ExecutionAttemptRecord.operation_id == proposal.operation_id,
            )
            .order_by(ExecutionAttemptRecord.attempt_number.desc())
        )
        if delivery.status == "ACCEPTED":
            return ExecutionResult(
                "SUCCEEDED",
                work_order_id=delivery.work_order_id,
                assignment_delivery_id=delivery.delivery_id,
                execution_attempt_id=latest_attempt.attempt_id if latest_attempt else None,
            )
        if delivery.status in {"REJECTED", "CANCELLED"}:
            return ExecutionResult(
                delivery.status,
                work_order_id=delivery.work_order_id,
                reason=delivery.reason,
                assignment_delivery_id=delivery.delivery_id,
                execution_attempt_id=latest_attempt.attempt_id if latest_attempt else None,
            )
        if latest_attempt is None:
            latest_attempt = self._fsm_assignment_attempt(
                identity.tenant_id,
                proposal,
                approval_id=approval_id,
                delivery=delivery,
                status="RECONCILING",
                reason="fsm_assignment_result_unknown",
                now=now,
                idempotency_key=None,
            )
            session.add(latest_attempt)
        reconciliation = self._ensure_execution_reconciliation(
            session,
            identity,
            proposal,
            approval_id=approval_id,
            initial_attempt_id=latest_attempt.attempt_id,
            now=now,
        )
        return ExecutionResult(
            "RECONCILING",
            work_order_id=delivery.work_order_id,
            reason=delivery.reason or "query_reconciliation_task",
            assignment_delivery_id=delivery.delivery_id,
            execution_attempt_id=latest_attempt.attempt_id,
            reconciliation_id=reconciliation.reconciliation_id,
        )

    @staticmethod
    def _fsm_assignment_attempt(
        tenant_id: str,
        proposal: ActionProposalRecord,
        *,
        approval_id: str,
        delivery: FsmAssignmentDeliveryRecord,
        status: str,
        reason: str | None,
        now: datetime,
        idempotency_key: str | None,
    ) -> ExecutionAttemptRecord:
        return ExecutionAttemptRecord(
            attempt_id=f"attempt-{uuid4().hex}",
            tenant_id=tenant_id,
            proposal_id=proposal.proposal_id,
            approval_id=approval_id,
            operation_id=proposal.operation_id,
            attempt_number=1,
            status=status,
            reason=reason,
            result={
                "assignment_delivery_id": delivery.delivery_id,
                "work_order_id": delivery.work_order_id,
                "provider": delivery.provider,
                "candidate_profile_id": delivery.candidate_profile_id,
                "external_assignment_id": delivery.external_assignment_id,
                "status": delivery.status,
                "idempotency_key_digest": (
                    sha256(idempotency_key.encode()).hexdigest()
                    if idempotency_key is not None
                    else None
                ),
            },
            started_at=now,
            finished_at=now,
            created_at=now,
            updated_at=now,
        )

    def _fsm_assignment_manual_escalation(
        self,
        session: Session,
        identity: IdentityContext,
        proposal: ActionProposalRecord,
        *,
        approval_id: str,
        delivery: FsmAssignmentDeliveryRecord,
        now: datetime,
        reason: str,
    ) -> ExecutionResult:
        delivery.status = "RECONCILING"
        delivery.reason = "fsm_assignment_manual_intervention_required"
        delivery.version += 1
        delivery.updated_at = now
        attempt = self._fsm_assignment_attempt(
            identity.tenant_id,
            proposal,
            approval_id=approval_id,
            delivery=delivery,
            status="RECONCILING",
            reason=reason,
            now=now,
            idempotency_key=None,
        )
        session.add(attempt)
        proposal.status = "RECONCILING"
        reconciliation = self._ensure_execution_reconciliation(
            session,
            identity,
            proposal,
            approval_id=approval_id,
            initial_attempt_id=attempt.attempt_id,
            now=now,
        )
        reconciliation.status = "ESCALATED"
        reconciliation.reason = delivery.reason
        reconciliation.last_checked_at = now
        reconciliation.version += 1
        reconciliation.updated_at = now
        session.add(
            ExecutionReconciliationEventRecord(
                event_id=f"reconciliation-event-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                reconciliation_id=reconciliation.reconciliation_id,
                sequence=reconciliation.version,
                event_type="ESCALATED",
                actor_subject_id=identity.subject_id,
                reason=delivery.reason,
                outcome="UNKNOWN",
                created_at=now,
                updated_at=now,
            )
        )
        return ExecutionResult(
            "RECONCILING",
            work_order_id=delivery.work_order_id,
            reason=delivery.reason,
            assignment_delivery_id=delivery.delivery_id,
            execution_attempt_id=attempt.attempt_id,
            reconciliation_id=reconciliation.reconciliation_id,
        )

    def _execute_work_order_closure(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        parameters: dict[str, Any],
        now: datetime,
        *,
        request_id: str,
        idempotency_key: str | None,
    ) -> ExecutionResult:
        work_order_id = str(parameters["work_order_id"])
        work = session.scalar(
            select(WorkOrderRecord)
            .where(
                WorkOrderRecord.tenant_id == identity.tenant_id,
                WorkOrderRecord.work_order_id == work_order_id,
            )
            .with_for_update()
        )
        if work is None:
            return ExecutionResult("RESOURCE_CHANGED")
        existing = session.scalar(
            select(ExecutionAttemptRecord).where(
                ExecutionAttemptRecord.tenant_id == identity.tenant_id,
                ExecutionAttemptRecord.operation_id == proposal.operation_id,
                ExecutionAttemptRecord.status == "SUCCEEDED",
            )
        )
        if existing is not None:
            return ExecutionResult("SUCCEEDED", work_order_id=work_order_id)
        try:
            closed = WorkOrderService(self._database, self._authorizer).close_in_session(
                session,
                identity,
                work_order_id,
                expected_version=proposal.resource_version,
                request_id=request_id,
                reason=str(parameters["reason"]),
                operation_id=proposal.operation_id,
                proposal_id=proposal.proposal_id,
            )
        except WorkOrderConflict as exc:
            return ExecutionResult("RESOURCE_CHANGED", reason=exc.reason)
        session.add(
            ExecutionAttemptRecord(
                attempt_id=f"attempt-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                proposal_id=proposal.proposal_id,
                approval_id=approval_id,
                operation_id=proposal.operation_id,
                attempt_number=1,
                status="SUCCEEDED",
                reason=None,
                result={
                    "work_order_id": closed.work_order_id,
                    "work_order_version": closed.version,
                    "incident_id": closed.incident_id,
                    "idempotency_key_digest": (
                        sha256(idempotency_key.encode()).hexdigest()
                        if idempotency_key is not None
                        else None
                    ),
                },
                started_at=now,
                finished_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        proposal.status = "EXECUTED"
        return ExecutionResult("SUCCEEDED", work_order_id=closed.work_order_id)

    def _ensure_execution_reconciliation(
        self,
        session: Session,
        identity: IdentityContext,
        proposal: ActionProposalRecord,
        *,
        approval_id: str,
        initial_attempt_id: str,
        now: datetime,
    ) -> ExecutionReconciliationRecord:
        existing = session.scalar(
            select(ExecutionReconciliationRecord).where(
                ExecutionReconciliationRecord.tenant_id == identity.tenant_id,
                ExecutionReconciliationRecord.operation_id == proposal.operation_id,
            )
        )
        if existing is not None:
            return existing
        reconciliation_id = f"reconciliation-{uuid4().hex}"
        record = ExecutionReconciliationRecord(
            reconciliation_id=reconciliation_id,
            tenant_id=identity.tenant_id,
            proposal_id=proposal.proposal_id,
            approval_id=approval_id,
            incident_id=proposal.incident_id,
            initial_execution_attempt_id=initial_attempt_id,
            resolved_execution_attempt_id=None,
            operation_id=proposal.operation_id,
            tool_id=proposal.tool_id,
            status="PENDING",
            outcome="UNKNOWN",
            reason="external_result_unknown",
            external_reference_id=None,
            version=1,
            last_checked_at=None,
            resolved_at=None,
            created_at=now,
            updated_at=now,
        )
        session.add(record)
        session.add(
            ExecutionReconciliationEventRecord(
                event_id=f"reconciliation-event-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                reconciliation_id=reconciliation_id,
                sequence=1,
                event_type="CREATED",
                actor_subject_id=identity.subject_id,
                reason=record.reason,
                outcome=record.outcome,
                created_at=now,
                updated_at=now,
            )
        )
        return record

    @staticmethod
    def _record_reconciliation_check(
        session: Session,
        identity: IdentityContext,
        record: ExecutionReconciliationRecord,
        *,
        now: datetime,
        event_type: str,
        reason: str,
    ) -> None:
        record.reason = reason
        record.last_checked_at = now
        record.version += 1
        record.updated_at = now
        session.add(
            ExecutionReconciliationEventRecord(
                event_id=f"reconciliation-event-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                reconciliation_id=record.reconciliation_id,
                sequence=record.version,
                event_type=event_type,
                actor_subject_id=identity.subject_id,
                reason=reason,
                outcome=record.outcome,
                created_at=now,
                updated_at=now,
            )
        )

    def _persist_success(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        result: ReservationResult,
        attempt_number: int,
        now: datetime,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ExecutionResult:
        incident = session.scalar(
            select(IncidentRecord)
            .where(
                IncidentRecord.tenant_id == identity.tenant_id,
                IncidentRecord.incident_id == proposal.incident_id,
            )
            .with_for_update()
        )
        if (
            incident is None
            or incident.version != proposal.resource_version
            or incident.status != "DIAGNOSED"
        ):
            return ExecutionResult("RESOURCE_CHANGED")
        work_order_id = f"work-order-{uuid4().hex}"
        attempt_id = f"attempt-{uuid4().hex}"
        session.add(
            ExecutionAttemptRecord(
                attempt_id=attempt_id,
                tenant_id=identity.tenant_id,
                proposal_id=proposal.proposal_id,
                approval_id=approval_id,
                operation_id=proposal.operation_id,
                attempt_number=attempt_number,
                status="SUCCEEDED",
                reason=reason,
                result={
                    "reservation_id": result.reservation_id,
                    "idempotency_key_digest": (
                        sha256(idempotency_key.encode()).hexdigest()
                        if idempotency_key is not None
                        else None
                    ),
                },
                started_at=now,
                finished_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            PartReservationRecord(
                reservation_id=result.reservation_id,
                tenant_id=identity.tenant_id,
                operation_id=result.operation_id,
                incident_id=proposal.incident_id,
                part_number=result.part_number,
                quantity=result.quantity,
                status="RESERVED",
                source=result.source,
                source_record_id=result.source_record_id,
                as_of=result.as_of,
                created_at=now,
                updated_at=now,
            )
        )
        # Persist each newly-created parent before records that reference it.
        session.flush()
        session.add(
            WorkOrderRecord(
                work_order_id=work_order_id,
                tenant_id=identity.tenant_id,
                incident_id=proposal.incident_id,
                proposal_id=proposal.proposal_id,
                reservation_id=result.reservation_id,
                status="READY",
                assigned_subject_id=None,
                part_issue_required=True,
                part_accounting_required=True,
                priority="NORMAL",
                sla_due_at=now + timedelta(hours=8),
                service_window_start=None,
                service_window_end=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            WorkOrderEventRecord(
                event_id=f"work-event-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                work_order_id=work_order_id,
                sequence=1,
                event_type="work_order.ready",
                actor_subject_id=identity.subject_id,
                payload={"reservation_id": result.reservation_id},
                occurred_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        previous_status = incident.status
        incident.status = "WORK_ORDER_CREATED"
        incident.version += 1
        incident.updated_at = now
        session.add(
            IncidentControlRecord(
                control_id=f"incident-control-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                incident_id=incident.incident_id,
                incident_version=incident.version,
                command_type="CREATE_WORK_ORDER",
                previous_status=previous_status,
                target_status=incident.status,
                reason="Approved parts reservation created work order",
                actor_subject_id=identity.subject_id,
                responsible_subject_id=None,
                recovery_condition=None,
                related_incident_id=None,
                details_json={
                    "work_order_id": work_order_id,
                    "proposal_id": proposal.proposal_id,
                    "approval_id": approval_id,
                    "operation_id": proposal.operation_id,
                    "reservation_id": result.reservation_id,
                },
                occurred_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        proposal.status = "EXECUTED"
        return ExecutionResult(
            "SUCCEEDED",
            result.reservation_id,
            work_order_id,
            execution_attempt_id=attempt_id,
        )

    def _persist_part_issue_success(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        issue: PartIssueRecord,
        reservation: PartReservationRecord,
        result: PartIssueResult,
        attempt_number: int,
        now: datetime,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ExecutionResult:
        if not _part_issue_result_matches(issue, result):
            return ExecutionResult(
                "DEPENDENCY_RESULT_INVALID",
                work_order_id=issue.work_order_id,
                part_issue_id=issue.part_issue_id,
            )
        if reservation.reservation_id != issue.reservation_id or reservation.status != "RESERVED":
            return ExecutionResult(
                "RESOURCE_CHANGED",
                work_order_id=issue.work_order_id,
                part_issue_id=issue.part_issue_id,
            )
        attempt_id = f"attempt-{uuid4().hex}"
        session.add(
            ExecutionAttemptRecord(
                attempt_id=attempt_id,
                tenant_id=identity.tenant_id,
                proposal_id=proposal.proposal_id,
                approval_id=approval_id,
                operation_id=proposal.operation_id,
                attempt_number=attempt_number,
                status="SUCCEEDED",
                reason=reason,
                result={
                    "part_issue_id": issue.part_issue_id,
                    "external_issue_id": result.issue_id,
                    "reservation_id": result.reservation_id,
                    "idempotency_key_digest": (
                        sha256(idempotency_key.encode()).hexdigest()
                        if idempotency_key is not None
                        else None
                    ),
                },
                started_at=now,
                finished_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        issue.status = "ISSUED"
        issue.external_issue_id = result.issue_id
        issue.source = result.source
        issue.source_record_id = result.source_record_id
        issue.as_of = _utc(result.as_of)
        issue.reason = reason
        issue.issued_at = now
        issue.version += 1
        issue.updated_at = now
        reservation.status = "ISSUED"
        reservation.updated_at = now
        proposal.status = "EXECUTED"
        return ExecutionResult(
            status="SUCCEEDED",
            reservation_id=reservation.reservation_id,
            work_order_id=issue.work_order_id,
            part_issue_id=issue.part_issue_id,
            execution_attempt_id=attempt_id,
        )

    def _persist_part_movement_success(
        self,
        session: Session,
        identity: IdentityContext,
        approval_id: str,
        proposal: ActionProposalRecord,
        movement: PartMaterialMovementRecord,
        result: PartMovementResult,
        attempt_number: int,
        now: datetime,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ExecutionResult:
        if not _part_movement_result_matches(movement, result):
            return ExecutionResult(
                "DEPENDENCY_RESULT_INVALID",
                work_order_id=movement.work_order_id,
                part_movement_id=movement.movement_id,
            )
        attempt_id = f"attempt-{uuid4().hex}"
        session.add(
            ExecutionAttemptRecord(
                attempt_id=attempt_id,
                tenant_id=identity.tenant_id,
                proposal_id=proposal.proposal_id,
                approval_id=approval_id,
                operation_id=proposal.operation_id,
                attempt_number=attempt_number,
                status="SUCCEEDED",
                reason=reason,
                result={
                    "part_movement_id": movement.movement_id,
                    "external_movement_id": result.movement_id,
                    "idempotency_key_digest": (
                        sha256(idempotency_key.encode()).hexdigest()
                        if idempotency_key is not None
                        else None
                    ),
                },
                started_at=now,
                finished_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        movement.status = "APPLIED"
        movement.external_movement_id = result.movement_id
        movement.source = result.source
        movement.source_record_id = result.source_record_id
        movement.as_of = _utc(result.as_of)
        movement.reason = reason
        movement.applied_at = now
        movement.version += 1
        movement.updated_at = now
        proposal.status = "EXECUTED"
        return ExecutionResult(
            "SUCCEEDED",
            reservation_id=movement.reservation_id,
            work_order_id=movement.work_order_id,
            part_movement_id=movement.movement_id,
            execution_attempt_id=attempt_id,
        )

    def _record_tool_call(
        self,
        identity: IdentityContext,
        *,
        tool_call_id: str,
        agent_run_id: str | None,
        tool_id: str,
        version: str,
        risk_tier: str,
        status: str,
        parameters: dict[str, Any],
        output_metadata: dict[str, Any],
        now: datetime,
    ) -> None:
        with self._database.transaction(identity.tenant_context) as session:
            session.add(
                ToolCallRecord(
                    tool_call_id=tool_call_id,
                    tenant_id=identity.tenant_id,
                    agent_run_id=agent_run_id,
                    tool_id=tool_id,
                    tool_version=version,
                    risk_tier=risk_tier,
                    status=status,
                    input_digest=_digest(parameters),
                    output_metadata=output_metadata,
                    created_at=now,
                    updated_at=now,
                )
            )

    @staticmethod
    def _load(
        session: Session,
        tenant_id: str,
        approval_id: str,
    ) -> tuple[ApprovalRequestRecord, ActionProposalRecord]:
        row = (
            session.execute(
                select(ApprovalRequestRecord, ActionProposalRecord)
                .join(
                    ActionProposalRecord,
                    ActionProposalRecord.proposal_id == ApprovalRequestRecord.proposal_id,
                )
                .where(
                    ApprovalRequestRecord.tenant_id == tenant_id,
                    ActionProposalRecord.tenant_id == tenant_id,
                    ApprovalRequestRecord.approval_id == approval_id,
                )
            )
            .tuples()
            .one_or_none()
        )
        if row is None:
            raise ApprovalNotVisible
        return row


def _digest(value: dict[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _part_issue_result_matches(
    issue: PartIssueRecord,
    result: PartIssueResult,
) -> bool:
    return (
        result.operation_id == issue.operation_id
        and result.reservation_id == issue.reservation_id
        and result.part_number == issue.part_number
        and result.quantity == issue.quantity
        and result.status == "ISSUED"
        and bool(result.issue_id)
        and bool(result.source)
        and bool(result.source_record_id)
        and result.as_of.tzinfo is not None
        and result.as_of.utcoffset() is not None
    )


def _part_movement_result_matches(
    movement: PartMaterialMovementRecord,
    result: PartMovementResult,
) -> bool:
    return (
        result.operation_id == movement.operation_id
        and result.movement_kind == movement.movement_kind
        and result.part_issue_id == movement.part_issue_id
        and result.reservation_id == movement.reservation_id
        and result.part_number == movement.part_number
        and result.quantity == movement.quantity
        and result.status == "APPLIED"
        and bool(result.movement_id)
        and bool(result.source)
        and bool(result.source_record_id)
        and result.as_of.tzinfo is not None
        and result.as_of.utcoffset() is not None
    )


def _validated_notification_delivery_result(
    result: Any,
    *,
    now: datetime,
) -> NotificationDeliveryResult:
    status = getattr(result, "status", None)
    provider_message_id = getattr(result, "provider_message_id", None)
    reason = getattr(result, "reason", None)
    occurred_at = getattr(result, "occurred_at", None)
    if status not in {"SUBMITTED", "RECONCILING", "DELIVERED", "FAILED", "UNSUBSCRIBED"}:
        raise ValueError("notification_delivery_status_invalid")
    if status != "RECONCILING" and (
        not isinstance(provider_message_id, str) or not provider_message_id
    ):
        raise ValueError("notification_provider_message_id_invalid")
    if reason is not None and (not isinstance(reason, str) or len(reason) > 255):
        raise ValueError("notification_delivery_reason_invalid")
    if not isinstance(occurred_at, datetime):
        raise ValueError("notification_delivery_time_invalid")
    occurred_at = _utc(occurred_at)
    if occurred_at > now + timedelta(minutes=5):
        raise ValueError("notification_delivery_time_invalid")
    return NotificationDeliveryResult(status, provider_message_id, reason, occurred_at)


def _validated_notification_receipt(receipt: Any) -> NotificationReceipt:
    values = {
        name: getattr(receipt, name, None)
        for name in (
            "tenant_id",
            "provider_event_id",
            "operation_id",
            "provider_message_id",
            "status",
            "occurred_at",
            "reason",
            "signature_digest",
        )
    }
    for name in (
        "tenant_id",
        "provider_event_id",
        "operation_id",
        "provider_message_id",
        "signature_digest",
    ):
        value = values[name]
        if not isinstance(value, str) or not value or len(value) > 255:
            raise ValueError(f"notification_receipt_{name}_invalid")
    if values["status"] not in {"SUBMITTED", "DELIVERED", "FAILED", "UNSUBSCRIBED"}:
        raise ValueError("notification_receipt_status_invalid")
    occurred_at = values["occurred_at"]
    if not isinstance(occurred_at, datetime):
        raise ValueError("notification_receipt_time_invalid")
    occurred_at = _utc(occurred_at)
    now = datetime.now(UTC)
    if occurred_at < now - timedelta(days=7) or occurred_at > now + timedelta(minutes=5):
        raise ValueError("notification_receipt_time_invalid")
    reason = values["reason"]
    if reason is not None and (not isinstance(reason, str) or len(reason) > 255):
        raise ValueError("notification_receipt_reason_invalid")
    return NotificationReceipt(
        tenant_id=cast(str, values["tenant_id"]),
        provider_event_id=cast(str, values["provider_event_id"]),
        operation_id=cast(str, values["operation_id"]),
        provider_message_id=cast(str, values["provider_message_id"]),
        status=cast(str, values["status"]),
        occurred_at=occurred_at,
        reason=reason,
        signature_digest=cast(str, values["signature_digest"]),
    )


def _validated_procurement_delivery_result(
    result: Any,
    *,
    now: datetime,
) -> ProcurementDeliveryResult:
    status = getattr(result, "status", None)
    external_request_id = getattr(result, "external_request_id", None)
    reason = getattr(result, "reason", None)
    occurred_at = getattr(result, "occurred_at", None)
    if status not in {"ACCEPTED", "REJECTED", "CANCELLED", "RECONCILING"}:
        raise ValueError("procurement_delivery_status_invalid")
    if status != "RECONCILING" and (
        not isinstance(external_request_id, str)
        or not external_request_id
        or len(external_request_id) > 255
    ):
        raise ValueError("procurement_external_request_id_invalid")
    if external_request_id is not None and (
        not isinstance(external_request_id, str)
        or not external_request_id
        or len(external_request_id) > 255
    ):
        raise ValueError("procurement_external_request_id_invalid")
    if reason is not None and (not isinstance(reason, str) or len(reason) > 255):
        raise ValueError("procurement_delivery_reason_invalid")
    if not isinstance(occurred_at, datetime):
        raise ValueError("procurement_delivery_time_invalid")
    occurred_at = _utc(occurred_at)
    if occurred_at > now + timedelta(minutes=5):
        raise ValueError("procurement_delivery_time_invalid")
    return ProcurementDeliveryResult(
        status=cast(str, status),
        external_request_id=external_request_id,
        reason=reason,
        occurred_at=occurred_at,
    )


def _validated_procurement_receipt(receipt: Any) -> ProcurementReceipt:
    values = {
        name: getattr(receipt, name, None)
        for name in (
            "tenant_id",
            "provider_event_id",
            "operation_id",
            "external_request_id",
            "status",
            "occurred_at",
            "reason",
            "signature_digest",
        )
    }
    for name in (
        "tenant_id",
        "provider_event_id",
        "operation_id",
        "external_request_id",
        "signature_digest",
    ):
        value = values[name]
        if not isinstance(value, str) or not value or len(value) > 255:
            raise ValueError(f"procurement_receipt_{name}_invalid")
    if values["status"] not in {"RECONCILING", "ACCEPTED", "REJECTED", "CANCELLED"}:
        raise ValueError("procurement_receipt_status_invalid")
    occurred_at = values["occurred_at"]
    if not isinstance(occurred_at, datetime):
        raise ValueError("procurement_receipt_time_invalid")
    occurred_at = _utc(occurred_at)
    now = datetime.now(UTC)
    if occurred_at < now - timedelta(days=7) or occurred_at > now + timedelta(minutes=5):
        raise ValueError("procurement_receipt_time_invalid")
    reason = values["reason"]
    if reason is not None and (not isinstance(reason, str) or len(reason) > 255):
        raise ValueError("procurement_receipt_reason_invalid")
    return ProcurementReceipt(
        tenant_id=cast(str, values["tenant_id"]),
        provider_event_id=cast(str, values["provider_event_id"]),
        operation_id=cast(str, values["operation_id"]),
        external_request_id=cast(str, values["external_request_id"]),
        status=cast(str, values["status"]),
        occurred_at=occurred_at,
        reason=reason,
        signature_digest=cast(str, values["signature_digest"]),
    )


def _validated_procurement_fulfillment_receipt(
    receipt: Any,
) -> ProcurementFulfillmentReceipt:
    names = (
        "tenant_id",
        "provider_event_id",
        "operation_id",
        "external_request_id",
        "external_order_id",
        "status",
        "received_quantity",
        "expected_delivery_at",
        "occurred_at",
        "reason",
        "signature_digest",
    )
    values = {name: getattr(receipt, name, None) for name in names}
    for name in (
        "tenant_id",
        "provider_event_id",
        "operation_id",
        "external_request_id",
        "external_order_id",
        "signature_digest",
    ):
        value = values[name]
        if not isinstance(value, str) or not value or len(value) > 255:
            raise ValueError(f"procurement_fulfillment_{name}_invalid")
    if values["status"] not in {
        "APPROVED",
        "ORDERED",
        "IN_TRANSIT",
        "PARTIALLY_RECEIVED",
        "RECEIVED",
        "REJECTED",
        "CANCELLED",
    }:
        raise ValueError("procurement_fulfillment_status_invalid")
    quantity = values["received_quantity"]
    if (
        not isinstance(quantity, int)
        or isinstance(quantity, bool)
        or quantity < 0
        or quantity > 1_000_000
    ):
        raise ValueError("procurement_fulfillment_quantity_invalid")
    occurred_at = values["occurred_at"]
    if not isinstance(occurred_at, datetime):
        raise ValueError("procurement_fulfillment_time_invalid")
    occurred_at = _utc(occurred_at)
    now = datetime.now(UTC)
    if occurred_at > now + timedelta(minutes=5):
        raise ValueError("procurement_fulfillment_time_invalid")
    expected_delivery_at = values["expected_delivery_at"]
    if expected_delivery_at is not None:
        if not isinstance(expected_delivery_at, datetime):
            raise ValueError("procurement_fulfillment_expected_delivery_invalid")
        expected_delivery_at = _utc(expected_delivery_at)
    reason = values["reason"]
    if reason is not None and (not isinstance(reason, str) or len(reason) > 255):
        raise ValueError("procurement_fulfillment_reason_invalid")
    return ProcurementFulfillmentReceipt(
        tenant_id=cast(str, values["tenant_id"]),
        provider_event_id=cast(str, values["provider_event_id"]),
        operation_id=cast(str, values["operation_id"]),
        external_request_id=cast(str, values["external_request_id"]),
        external_order_id=cast(str, values["external_order_id"]),
        status=cast(str, values["status"]),
        received_quantity=quantity,
        expected_delivery_at=expected_delivery_at,
        occurred_at=occurred_at,
        reason=reason,
        signature_digest=cast(str, values["signature_digest"]),
    )


def _same_procurement_fulfillment_receipt(
    stored: PurchaseFulfillmentReceiptRecord,
    receipt: ProcurementFulfillmentReceipt,
) -> bool:
    stored_expected = (
        _utc(stored.expected_delivery_at)
        if stored.expected_delivery_at is not None
        else None
    )
    return (
        stored.operation_id == receipt.operation_id
        and stored.external_request_id == receipt.external_request_id
        and stored.external_order_id == receipt.external_order_id
        and stored.status == receipt.status
        and stored.received_quantity == receipt.received_quantity
        and stored_expected == receipt.expected_delivery_at
        and _utc(stored.occurred_at) == receipt.occurred_at
        and stored.reason == receipt.reason
    )


def _validate_procurement_fulfillment_quantity(
    status: str,
    received_quantity: int,
    requested_quantity: int,
) -> None:
    if status == "PARTIALLY_RECEIVED":
        valid = 0 < received_quantity < requested_quantity
    elif status == "RECEIVED":
        valid = received_quantity == requested_quantity
    else:
        valid = received_quantity == 0
    if not valid:
        raise ProcurementFulfillmentReceiptRejected(
            "procurement_fulfillment_quantity_invalid"
        )


def _validate_procurement_fulfillment_transition(
    current: PurchaseFulfillmentRecord,
    receipt: ProcurementFulfillmentReceipt,
) -> None:
    if current.status in {"RECEIVED", "REJECTED", "CANCELLED"}:
        raise ProcurementFulfillmentReceiptRejected(
            "procurement_fulfillment_terminal"
        )
    ranks = {
        "APPROVED": 1,
        "ORDERED": 2,
        "IN_TRANSIT": 3,
        "PARTIALLY_RECEIVED": 4,
        "RECEIVED": 5,
        "REJECTED": 5,
        "CANCELLED": 5,
    }
    if ranks[receipt.status] < ranks[current.status]:
        raise ProcurementFulfillmentReceiptRejected(
            "procurement_fulfillment_status_regression"
        )
    if receipt.received_quantity < current.received_quantity:
        raise ProcurementFulfillmentReceiptRejected(
            "procurement_fulfillment_quantity_regression"
        )
    if (
        receipt.status == current.status
        and receipt.received_quantity == current.received_quantity
        and (
            receipt.expected_delivery_at
            != (
                _utc(current.expected_delivery_at)
                if current.expected_delivery_at is not None
                else None
            )
            or receipt.reason != current.reason
        )
    ):
        raise ProcurementFulfillmentReceiptRejected(
            "procurement_fulfillment_same_state_conflict"
        )
    if receipt.occurred_at < _utc(current.last_event_occurred_at):
        raise ProcurementFulfillmentReceiptRejected(
            "procurement_fulfillment_time_regression"
        )


def _procurement_reservation_readiness_from_wms(
    purchase: PurchaseRequestRecord,
    fulfillment: PurchaseFulfillmentRecord,
    facts: dict[str, Any],
) -> ProcurementReservationReadiness:
    part_number = facts["part_number"]
    available_quantity = facts["available_quantity"]
    source = facts["source"]
    source_record_id = facts["source_record_id"]
    raw_as_of = facts["as_of"]
    if not isinstance(part_number, str) or not part_number:
        raise ValueError("wms_part_number_invalid")
    if (
        not isinstance(available_quantity, int)
        or isinstance(available_quantity, bool)
        or available_quantity < 0
    ):
        raise ValueError("wms_available_quantity_invalid")
    if not isinstance(source, str) or not source:
        raise ValueError("wms_source_invalid")
    if not isinstance(source_record_id, str) or not source_record_id:
        raise ValueError("wms_source_record_id_invalid")
    try:
        as_of = (
            raw_as_of
            if isinstance(raw_as_of, datetime)
            else datetime.fromisoformat(str(raw_as_of).replace("Z", "+00:00"))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("wms_as_of_invalid") from exc
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("wms_as_of_invalid")
    as_of = _utc(as_of)
    metadata = {
        "source": source,
        "source_record_id": source_record_id,
        "as_of": as_of,
        "available_quantity": available_quantity,
    }
    if part_number != purchase.part_number:
        return ProcurementReservationReadiness(
            "REVIEW_REQUIRED",
            "wms_part_number_conflict",
            "REVIEW_WMS_PART_MAPPING",
            **metadata,
        )
    if as_of < _utc(fulfillment.last_event_occurred_at):
        return ProcurementReservationReadiness(
            "AWAITING_WMS_SYNC",
            "wms_fact_precedes_final_receipt",
            "WAIT_FOR_WMS_SYNC",
            **metadata,
        )
    if available_quantity < purchase.quantity:
        return ProcurementReservationReadiness(
            "AWAITING_WMS_SYNC",
            "wms_available_quantity_insufficient",
            "WAIT_FOR_WMS_SYNC",
            **metadata,
        )
    return ProcurementReservationReadiness(
        "READY_FOR_RESERVATION",
        "wms_inventory_ready",
        "OPEN_PARTS_AND_REQUEST_RESERVATION_APPROVAL",
        **metadata,
    )


def _procurement_execution_attempt(
    tenant_id: str,
    proposal: ActionProposalRecord,
    *,
    approval_id: str,
    attempt_id: str,
    status: str,
    reason: str | None,
    purchase: PurchaseRequestRecord,
    now: datetime,
    idempotency_key: str | None,
    attempt_number: int = 1,
) -> ExecutionAttemptRecord:
    return ExecutionAttemptRecord(
        attempt_id=attempt_id,
        tenant_id=tenant_id,
        proposal_id=proposal.proposal_id,
        approval_id=approval_id,
        operation_id=proposal.operation_id,
        attempt_number=attempt_number,
        status=status,
        reason=reason,
        result={
            "purchase_request_id": purchase.purchase_request_id,
            "delivery_target": purchase.delivery_target,
            "provider": purchase.provider,
            "profile_id": purchase.profile_id,
            "profile_version": purchase.profile_version,
            "external_request_id": purchase.external_request_id,
            "status": purchase.status,
            "idempotency_key_digest": (
                sha256(idempotency_key.encode()).hexdigest()
                if idempotency_key is not None
                else None
            ),
        },
        started_at=now,
        finished_at=now,
        created_at=now,
        updated_at=now,
    )


def _validated_refund_delivery_result(
    result: Any,
    *,
    now: datetime,
) -> RefundDeliveryResult:
    status = getattr(result, "status", None)
    payment_status = getattr(result, "payment_status", None)
    external_request_id = getattr(result, "external_request_id", None)
    reason = getattr(result, "reason", None)
    occurred_at = getattr(result, "occurred_at", None)
    if status not in {"ACCEPTED", "REJECTED", "CANCELLED", "RECONCILING"}:
        raise ValueError("refund_delivery_status_invalid")
    if payment_status not in {
        "NOT_STARTED",
        "PROCESSING",
        "PAID",
        "FAILED",
    }:
        raise ValueError("refund_payment_status_invalid")
    if status != "RECONCILING" and (
        not isinstance(external_request_id, str)
        or not external_request_id
        or len(external_request_id) > 255
    ):
        raise ValueError("refund_external_request_id_invalid")
    if external_request_id is not None and (
        not isinstance(external_request_id, str)
        or not external_request_id
        or len(external_request_id) > 255
    ):
        raise ValueError("refund_external_request_id_invalid")
    if status in {"REJECTED", "CANCELLED", "RECONCILING"} and payment_status in {
        "PROCESSING",
        "PAID",
    }:
        raise ValueError("refund_request_payment_state_invalid")
    if reason is not None and (not isinstance(reason, str) or len(reason) > 255):
        raise ValueError("refund_delivery_reason_invalid")
    if not isinstance(occurred_at, datetime):
        raise ValueError("refund_delivery_time_invalid")
    occurred_at = _utc(occurred_at)
    if occurred_at > now + timedelta(minutes=5):
        raise ValueError("refund_delivery_time_invalid")
    return RefundDeliveryResult(
        status=cast(str, status),
        payment_status=cast(str, payment_status),
        external_request_id=external_request_id,
        reason=reason,
        occurred_at=occurred_at,
    )


def _validated_refund_receipt(receipt: Any) -> RefundReceipt:
    values = {
        name: getattr(receipt, name, None)
        for name in (
            "tenant_id",
            "provider_event_id",
            "operation_id",
            "external_request_id",
            "status",
            "payment_status",
            "occurred_at",
            "reason",
            "signature_digest",
        )
    }
    for name in (
        "tenant_id",
        "provider_event_id",
        "operation_id",
        "external_request_id",
        "signature_digest",
    ):
        value = values[name]
        if not isinstance(value, str) or not value or len(value) > 255:
            raise ValueError(f"refund_receipt_{name}_invalid")
    if values["status"] not in {
        "RECONCILING",
        "ACCEPTED",
        "REJECTED",
        "CANCELLED",
    }:
        raise ValueError("refund_receipt_status_invalid")
    if values["payment_status"] not in {
        "NOT_STARTED",
        "PROCESSING",
        "PAID",
        "FAILED",
    }:
        raise ValueError("refund_receipt_payment_status_invalid")
    if values["status"] != "ACCEPTED" and values["payment_status"] in {
        "PROCESSING",
        "PAID",
    }:
        raise ValueError("refund_receipt_request_payment_state_invalid")
    occurred_at = values["occurred_at"]
    if not isinstance(occurred_at, datetime):
        raise ValueError("refund_receipt_time_invalid")
    occurred_at = _utc(occurred_at)
    now = datetime.now(UTC)
    if occurred_at < now - timedelta(days=7) or occurred_at > now + timedelta(minutes=5):
        raise ValueError("refund_receipt_time_invalid")
    reason = values["reason"]
    if reason is not None and (not isinstance(reason, str) or len(reason) > 255):
        raise ValueError("refund_receipt_reason_invalid")
    return RefundReceipt(
        tenant_id=cast(str, values["tenant_id"]),
        provider_event_id=cast(str, values["provider_event_id"]),
        operation_id=cast(str, values["operation_id"]),
        external_request_id=cast(str, values["external_request_id"]),
        status=cast(str, values["status"]),
        payment_status=cast(str, values["payment_status"]),
        occurred_at=occurred_at,
        reason=reason,
        signature_digest=cast(str, values["signature_digest"]),
    )


def _same_refund_receipt(
    stored: RefundRequestReceiptRecord,
    receipt: RefundReceipt,
) -> bool:
    return (
        stored.operation_id == receipt.operation_id
        and stored.external_request_id == receipt.external_request_id
        and stored.status == receipt.status
        and stored.payment_status == receipt.payment_status
        and stored.reason == receipt.reason
        and _utc(stored.occurred_at) == receipt.occurred_at
        and stored.signature_digest == receipt.signature_digest
    )


def _refund_payment_transition_allowed(current: str, target: str) -> bool:
    if current == target or current in {"PAID", "FAILED"}:
        return False
    ranks = {"NOT_STARTED": 0, "PROCESSING": 1}
    if target in {"PAID", "FAILED"}:
        return current in ranks
    return current in ranks and target in ranks and ranks[target] > ranks[current]


def _refund_execution_attempt(
    tenant_id: str,
    proposal: ActionProposalRecord,
    *,
    approval_id: str,
    attempt_id: str,
    status: str,
    reason: str | None,
    refund: RefundRequestRecord,
    now: datetime,
    idempotency_key: str | None,
    attempt_number: int = 1,
) -> ExecutionAttemptRecord:
    return ExecutionAttemptRecord(
        attempt_id=attempt_id,
        tenant_id=tenant_id,
        proposal_id=proposal.proposal_id,
        approval_id=approval_id,
        operation_id=proposal.operation_id,
        attempt_number=attempt_number,
        status=status,
        reason=reason,
        result={
            "refund_request_id": refund.refund_request_id,
            "delivery_target": refund.delivery_target,
            "provider": refund.provider,
            "profile_id": refund.profile_id,
            "profile_version": refund.profile_version,
            "external_request_id": refund.external_request_id,
            "status": refund.status,
            "payment_status": refund.payment_status,
            "idempotency_key_digest": (
                sha256(idempotency_key.encode()).hexdigest()
                if idempotency_key is not None
                else None
            ),
        },
        started_at=now,
        finished_at=now,
        created_at=now,
        updated_at=now,
    )


def _notification_execution_attempt(
    identity: IdentityContext,
    proposal: ActionProposalRecord,
    *,
    approval_id: str,
    attempt_id: str,
    status: str,
    reason: str | None,
    delivery: CustomerNotificationDeliveryRecord,
    now: datetime,
    idempotency_key: str | None,
) -> ExecutionAttemptRecord:
    return ExecutionAttemptRecord(
        attempt_id=attempt_id,
        tenant_id=identity.tenant_id,
        proposal_id=proposal.proposal_id,
        approval_id=approval_id,
        operation_id=proposal.operation_id,
        attempt_number=1,
        status=status,
        reason=reason,
        result={
            "notification_id": delivery.portal_update_id,
            "notification_delivery_id": delivery.delivery_id,
            "provider": delivery.provider,
            "provider_message_id": delivery.provider_message_id,
            "channel": delivery.channel,
            "idempotency_key_digest": (
                sha256(idempotency_key.encode()).hexdigest()
                if idempotency_key is not None
                else None
            ),
        },
        started_at=now,
        finished_at=now,
        created_at=now,
        updated_at=now,
    )


def _portal_notification_option() -> CustomerNotificationOption:
    return CustomerNotificationOption(
        channel="PORTAL",
        contact_point_id=None,
        masked_destination="客户门户",
        source_system="PLATFORM",
        source_record_id=None,
        source_version=None,
        as_of=None,
        consent_basis_digest=None,
        expires_at=None,
    )


def _notification_contact_binding(
    contact: NotificationContact,
    *,
    channel: str,
    now: datetime,
) -> dict[str, Any]:
    expected_contact_channel = {
        "PORTAL_AND_EMAIL": "EMAIL",
        "PORTAL_AND_SMS": "SMS",
    }.get(channel)
    if expected_contact_channel is None or contact.channel != expected_contact_channel:
        raise CustomerNotificationContactConflict("contact_channel_mismatch")
    string_fields = (
        contact.contact_point_id,
        contact.masked_destination,
        contact.source_system,
        contact.source_record_id,
        contact.source_version,
        contact.consent_basis,
    )
    if any(
        not isinstance(value, str) or not value or not value.isprintable()
        for value in string_fields
    ):
        raise CustomerNotificationContactConflict("contact_authority_invalid")
    if contact.verification_status != "VERIFIED" or contact.consent_status != "OPTED_IN":
        raise CustomerNotificationContactConflict("contact_not_eligible")
    try:
        raw_destination = contact.destination.reveal()
    except Exception as exc:
        raise CustomerNotificationContactConflict("contact_authority_invalid") from exc
    if (
        not isinstance(raw_destination, str)
        or not raw_destination
        or "*" not in contact.masked_destination
        or contact.masked_destination == raw_destination
        or raw_destination in contact.masked_destination
    ):
        raise CustomerNotificationContactConflict("contact_mask_invalid")
    if contact.as_of.tzinfo is None or contact.as_of.utcoffset() is None:
        raise CustomerNotificationContactConflict("contact_authority_invalid")
    normalized_as_of = _utc(contact.as_of)
    if normalized_as_of > now + timedelta(minutes=1):
        raise CustomerNotificationContactConflict("contact_authority_invalid")
    normalized_expiry: datetime | None = None
    if contact.expires_at is not None:
        if contact.expires_at.tzinfo is None or contact.expires_at.utcoffset() is None:
            raise CustomerNotificationContactConflict("contact_authority_invalid")
        normalized_expiry = _utc(contact.expires_at)
        if normalized_expiry <= now:
            raise CustomerNotificationContactConflict("contact_not_eligible")
    return {
        "contact_point_id": contact.contact_point_id,
        "channel": contact.channel,
        "source_system": contact.source_system,
        "source_record_id": contact.source_record_id,
        "source_version": contact.source_version,
        "as_of": normalized_as_of.isoformat(),
        "masked_destination": contact.masked_destination,
        "verification_status": "VERIFIED",
        "consent_status": "OPTED_IN",
        "consent_basis_digest": sha256(contact.consent_basis.encode()).hexdigest(),
        "expires_at": normalized_expiry.isoformat() if normalized_expiry else None,
    }


def _procurement_profile_binding(profile: ProcurementProfile) -> dict[str, str]:
    return {
        "provider": profile.provider,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "display_name": profile.display_name,
        "as_of": profile.as_of.isoformat(),
        "expires_at": profile.expires_at.isoformat(),
    }


def _refund_profile_binding(profile: RefundProfile) -> dict[str, str]:
    return {
        "provider": profile.provider,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "display_name": profile.display_name,
        "as_of": profile.as_of.isoformat(),
        "expires_at": profile.expires_at.isoformat(),
    }


def _validated_fsm_assignment_delivery_result(
    result: Any,
    *,
    now: datetime,
) -> FsmAssignmentDeliveryResult:
    status = getattr(result, "status", None)
    external_assignment_id = getattr(result, "external_assignment_id", None)
    reason = getattr(result, "reason", None)
    occurred_at = getattr(result, "occurred_at", None)
    if status not in {"ACCEPTED", "REJECTED", "CANCELLED", "RECONCILING"}:
        raise ValueError("fsm_assignment_delivery_status_invalid")
    if status != "RECONCILING" and (
        not isinstance(external_assignment_id, str)
        or not external_assignment_id
        or len(external_assignment_id) > 255
        or not external_assignment_id.isprintable()
    ):
        raise ValueError("fsm_assignment_external_id_invalid")
    if external_assignment_id is not None and (
        not isinstance(external_assignment_id, str)
        or not external_assignment_id
        or len(external_assignment_id) > 255
        or not external_assignment_id.isprintable()
    ):
        raise ValueError("fsm_assignment_external_id_invalid")
    if reason is not None and (
        not isinstance(reason, str) or len(reason) > 255 or not reason.isprintable()
    ):
        raise ValueError("fsm_assignment_delivery_reason_invalid")
    if (
        not isinstance(occurred_at, datetime)
        or occurred_at.tzinfo is None
        or occurred_at.utcoffset() is None
    ):
        raise ValueError("fsm_assignment_delivery_time_invalid")
    occurred_at = occurred_at.astimezone(UTC)
    if occurred_at > now + timedelta(minutes=5):
        raise ValueError("fsm_assignment_delivery_time_invalid")
    return FsmAssignmentDeliveryResult(
        status=cast(str, status),
        external_assignment_id=external_assignment_id,
        reason=reason,
        occurred_at=occurred_at,
    )


def _validated_fsm_assignment_receipt(receipt: Any) -> FsmAssignmentReceipt:
    limits = {
        "tenant_id": 128,
        "provider_event_id": 255,
        "operation_id": 128,
        "candidate_profile_id": 128,
        "assignee_subject_id": 128,
        "external_assignment_id": 255,
        "signature_digest": 128,
    }
    values = {
        name: getattr(receipt, name, None)
        for name in (*limits, "status", "occurred_at", "reason")
    }
    for name, limit in limits.items():
        value = values[name]
        if (
            not isinstance(value, str)
            or not value
            or len(value) > limit
            or not value.isprintable()
        ):
            raise ValueError(f"fsm_assignment_receipt_{name}_invalid")
    tenant_id = cast(str, values["tenant_id"])
    if not all(
        char.isascii() and (char.isalnum() or char in "._-") for char in tenant_id
    ):
        raise ValueError("fsm_assignment_receipt_tenant_id_invalid")
    if values["status"] not in {
        "RECONCILING",
        "ACCEPTED",
        "REJECTED",
        "CANCELLED",
    }:
        raise ValueError("fsm_assignment_receipt_status_invalid")
    occurred_at = values["occurred_at"]
    if (
        not isinstance(occurred_at, datetime)
        or occurred_at.tzinfo is None
        or occurred_at.utcoffset() is None
    ):
        raise ValueError("fsm_assignment_receipt_time_invalid")
    occurred_at = occurred_at.astimezone(UTC)
    now = datetime.now(UTC)
    if occurred_at < now - timedelta(days=7) or occurred_at > now + timedelta(minutes=5):
        raise ValueError("fsm_assignment_receipt_time_invalid")
    reason = values["reason"]
    if reason is not None and (
        not isinstance(reason, str) or len(reason) > 255 or not reason.isprintable()
    ):
        raise ValueError("fsm_assignment_receipt_reason_invalid")
    return FsmAssignmentReceipt(
        tenant_id=tenant_id,
        provider_event_id=cast(str, values["provider_event_id"]),
        operation_id=cast(str, values["operation_id"]),
        candidate_profile_id=cast(str, values["candidate_profile_id"]),
        assignee_subject_id=cast(str, values["assignee_subject_id"]),
        external_assignment_id=cast(str, values["external_assignment_id"]),
        status=cast(str, values["status"]),
        occurred_at=occurred_at,
        reason=reason,
        signature_digest=cast(str, values["signature_digest"]),
    )


def _same_fsm_assignment_receipt(
    stored: FsmAssignmentReceiptRecord,
    receipt: FsmAssignmentReceipt,
) -> bool:
    return (
        stored.operation_id == receipt.operation_id
        and stored.candidate_profile_id == receipt.candidate_profile_id
        and stored.assignee_subject_id == receipt.assignee_subject_id
        and stored.external_assignment_id == receipt.external_assignment_id
        and stored.status == receipt.status
        and stored.reason == receipt.reason
        and _utc(stored.occurred_at) == receipt.occurred_at
        and stored.signature_digest == receipt.signature_digest
    )


def _validated_fsm_assignment_candidate(
    raw: Any,
    *,
    provider_id: str,
    now: datetime,
) -> FsmAssignmentCandidate:
    string_limits = {
        "provider": 128,
        "candidate_profile_id": 128,
        "profile_version": 128,
        "assignee_subject_id": 128,
        "display_name": 128,
        "eligibility_code": 64,
        "source_record_id": 255,
    }
    values = {name: getattr(raw, name, None) for name in string_limits}
    if any(
        not isinstance(value, str)
        or not value
        or len(value) > string_limits[name]
        or not value.isprintable()
        for name, value in values.items()
    ):
        raise FsmAssignmentProfileUnavailable("fsm_assignment_candidate_invalid")
    provider = cast(str, values["provider"])
    if (
        provider != provider_id
        or not all(char.isascii() and (char.isalnum() or char in "._-") for char in provider)
        or values["eligibility_code"] != "ELIGIBLE"
    ):
        raise FsmAssignmentProfileUnavailable("fsm_assignment_candidate_invalid")

    skills = getattr(raw, "matched_skill_codes", None)
    if (
        not isinstance(skills, tuple)
        or not 1 <= len(skills) <= 32
        or len(skills) != len(set(skills))
        or any(
            not isinstance(skill, str)
            or not skill
            or len(skill) > 64
            or not skill.isprintable()
            for skill in skills
        )
    ):
        raise FsmAssignmentProfileUnavailable("fsm_assignment_candidate_invalid")

    datetimes = {
        name: getattr(raw, name, None)
        for name in (
            "service_window_start",
            "service_window_end",
            "as_of",
            "expires_at",
        )
    }
    if any(
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
        for value in datetimes.values()
    ):
        raise FsmAssignmentProfileUnavailable("fsm_assignment_candidate_invalid")
    normalized = {
        name: cast(datetime, value).astimezone(UTC)
        for name, value in datetimes.items()
    }
    if (
        normalized["service_window_start"] >= normalized["service_window_end"]
        or normalized["as_of"] > now + timedelta(minutes=1)
        or normalized["as_of"] >= normalized["expires_at"]
    ):
        raise FsmAssignmentProfileUnavailable("fsm_assignment_candidate_invalid")
    if normalized["expires_at"] <= now:
        raise FsmAssignmentProfileUnavailable("fsm_assignment_candidate_expired")

    travel_minutes = getattr(raw, "travel_minutes", None)
    remaining_work_minutes = getattr(raw, "remaining_work_minutes", None)
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > 10080
        for value in (travel_minutes, remaining_work_minutes)
    ):
        raise FsmAssignmentProfileUnavailable("fsm_assignment_candidate_invalid")
    return FsmAssignmentCandidate(
        provider=provider,
        candidate_profile_id=cast(str, values["candidate_profile_id"]),
        profile_version=cast(str, values["profile_version"]),
        assignee_subject_id=cast(str, values["assignee_subject_id"]),
        display_name=cast(str, values["display_name"]),
        matched_skill_codes=skills,
        service_window_start=normalized["service_window_start"],
        service_window_end=normalized["service_window_end"],
        travel_minutes=cast(int, travel_minutes),
        remaining_work_minutes=cast(int, remaining_work_minutes),
        eligibility_code="ELIGIBLE",
        source_record_id=cast(str, values["source_record_id"]),
        as_of=normalized["as_of"],
        expires_at=normalized["expires_at"],
    )


def _fsm_assignment_candidate_binding(
    candidate: FsmAssignmentCandidate,
) -> dict[str, Any]:
    return {
        "provider": candidate.provider,
        "candidate_profile_id": candidate.candidate_profile_id,
        "profile_version": candidate.profile_version,
        "assignee_subject_id": candidate.assignee_subject_id,
        "display_name": candidate.display_name,
        "matched_skill_codes": list(candidate.matched_skill_codes),
        "service_window_start": candidate.service_window_start.isoformat(),
        "service_window_end": candidate.service_window_end.isoformat(),
        "travel_minutes": candidate.travel_minutes,
        "remaining_work_minutes": candidate.remaining_work_minutes,
        "eligibility_code": candidate.eligibility_code,
        "source_record_id": candidate.source_record_id,
        "as_of": candidate.as_of.isoformat(),
        "expires_at": candidate.expires_at.isoformat(),
    }


def _service_entitlement_binding(entitlement: Any) -> dict[str, Any]:
    master = entitlement.master_warranty
    live = entitlement.live_entitlement
    return {
        "decision": entitlement.decision,
        "reason_codes": list(entitlement.reason_codes),
        "master_warranty": (
            {
                "warranty_id": master.warranty_id,
                "status": master.status,
                "source": master.source,
                "source_record_id": master.source_record_id,
                "as_of": master.as_of.isoformat() if master.as_of is not None else None,
                "freshness": master.freshness,
            }
            if master is not None
            else None
        ),
        "live_entitlement": (
            {
                "coverage": live.coverage,
                "valid": live.valid,
                "source": live.source,
                "source_record_id": live.source_record_id,
                "as_of": live.as_of.isoformat(),
                "authority_owner": live.authority_owner,
                "field_sources": dict(sorted(live.field_sources.items())),
                "tool_call_id": live.tool_call_id,
            }
            if live is not None
            else None
        ),
    }


def _repair_quotation_binding(
    quotation: ServiceQuotationRecord,
    decision: ServiceQuotationDecisionRecord,
) -> dict[str, Any]:
    return {
        "quotation_id": quotation.quotation_id,
        "quotation_version": quotation.quotation_version,
        "state_version": quotation.state_version,
        "status": quotation.status,
        "diagnosis_run_id": quotation.diagnosis_run_id,
        "diagnosis_version": quotation.diagnosis_version,
        "currency": quotation.currency,
        "total": quotation.total,
        "publication_proposal_id": quotation.proposal_id,
        "publication_approval_id": quotation.approval_id,
        "customer_decision": {
            "decision_id": decision.decision_id,
            "decision": decision.decision,
            "reporter_subject_id": decision.reporter_subject_id,
            "quotation_version": decision.quotation_version,
            "decided_at": _utc(decision.decided_at).isoformat(),
        },
    }


def _stable_entitlement_binding(binding: dict[str, Any]) -> dict[str, Any]:
    stable = json.loads(json.dumps(binding))
    live = stable.get("live_entitlement")
    if isinstance(live, dict):
        live.pop("tool_call_id", None)
        live.pop("as_of", None)
    return cast(dict[str, Any], stable)


def _scoped_resource(
    identity: IdentityContext,
    *,
    resource_id: str,
    asset_id: str,
    site_id: str | None,
) -> ResourceContext:
    if asset_id in identity.asset_ids:
        return ResourceContext(
            identity.tenant_id,
            resource_id=resource_id,
            asset_id=asset_id,
        )
    if site_id is not None and site_id in identity.site_ids:
        return ResourceContext(
            identity.tenant_id,
            resource_id=resource_id,
            site_id=site_id,
        )
    return ResourceContext(
        identity.tenant_id,
        resource_id=resource_id,
        asset_id=asset_id,
    )


def _latest_customer_result_confirmation(
    session: Session,
    *,
    tenant_id: str,
    work_order_id: str,
) -> CustomerServiceUpdateRecord | None:
    return session.scalar(
        select(CustomerServiceUpdateRecord)
        .where(
            CustomerServiceUpdateRecord.tenant_id == tenant_id,
            CustomerServiceUpdateRecord.work_order_id == work_order_id,
            CustomerServiceUpdateRecord.update_type == "RESULT_CONFIRMATION",
        )
        .order_by(
            CustomerServiceUpdateRecord.occurred_at.desc(),
            CustomerServiceUpdateRecord.created_at.desc(),
        )
    )


def _refund_eligible(confirmation: CustomerServiceUpdateRecord | None) -> bool:
    return bool(
        confirmation is not None
        and confirmation.result_accepted is not None
        and confirmation.satisfaction_rating is not None
        and (confirmation.result_accepted is False or confirmation.satisfaction_rating <= 2)
    )


def _render_customer_notification(
    *, template_id: str, headline: str, detail: str, next_step: str
) -> str:
    labels = {
        "DIAGNOSIS_SUMMARY": "诊断进展",
        "INFORMATION_REQUEST": "资料补充请求",
        "SERVICE_PROGRESS": "服务进展",
    }
    return f"【{labels[template_id]}】{headline}\n{detail}\n下一步：{next_step}"


def _purchase_inventory_snapshot(
    availability: dict[str, object],
    *,
    expected_part_number: str,
    requested_quantity: int,
    now: datetime,
) -> dict[str, int | str]:
    part_number = str(availability["part_number"])
    raw_requested_quantity = availability["requested_quantity"]
    if (
        not isinstance(raw_requested_quantity, int)
        or isinstance(raw_requested_quantity, bool)
        or raw_requested_quantity != requested_quantity
    ):
        raise ValueError("inventory requested quantity is invalid")
    raw_available_quantity = availability["available_quantity"]
    if not isinstance(raw_available_quantity, int) or isinstance(raw_available_quantity, bool):
        raise TypeError("inventory quantity is invalid")
    available_quantity = raw_available_quantity
    available = availability["available"]
    if (
        part_number != expected_part_number
        or available_quantity < 0
        or not isinstance(available, bool)
        or available != (available_quantity >= requested_quantity)
    ):
        raise ValueError("inventory identity or availability mismatch")
    raw_as_of = availability["as_of"]
    as_of = (
        raw_as_of
        if isinstance(raw_as_of, datetime)
        else datetime.fromisoformat(str(raw_as_of).replace("Z", "+00:00"))
    )
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("inventory timestamp must be timezone-aware")
    normalized_as_of = as_of.astimezone(UTC)
    if normalized_as_of < now - timedelta(minutes=15) or normalized_as_of > now + timedelta(
        minutes=1
    ):
        raise ValueError("inventory snapshot is stale")
    source = availability["source"]
    source_record_id = availability["source_record_id"]
    if (
        not isinstance(source, str)
        or not source
        or not isinstance(source_record_id, str)
        or not source_record_id
    ):
        raise ValueError("inventory source is invalid")
    return {
        "available_quantity": available_quantity,
        "source": source,
        "source_record_id": source_record_id,
        "as_of": normalized_as_of.isoformat(),
    }


def _record_handoff_sources(
    session: Session,
    identity: IdentityContext,
    handoffs: Iterable[EquipmentControlHandoffRecord],
    request_id: str,
) -> None:
    """Handoff evidence can refer to other drafts on the same authorized asset."""
    evidence_ids = {value for handoff in handoffs for value in handoff.evidence_ids}
    bundles = session.scalars(
        select(EvidenceBundleRecord).where(
            EvidenceBundleRecord.tenant_id == identity.tenant_id,
            EvidenceBundleRecord.bundle_id.in_(evidence_ids),
        )
    ).all()
    if len(bundles) != len(evidence_ids):
        raise ReviewIsolationConflict("source_unavailable")
    for draft_id in sorted({bundle.draft_id for bundle in bundles}):
        record_draft_disclosure(session, identity, draft_id, request_id=request_id)
