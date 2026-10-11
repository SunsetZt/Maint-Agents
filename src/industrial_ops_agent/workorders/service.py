"""Role-checked optimistic WorkOrder state transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.edge.contracts import FieldEdgeDiagnosisCandidate, canonical_digest
from industrial_ops_agent.eventing.envelope import build_work_order_closed_event
from industrial_ops_agent.eventing.trace_context import capture_current_trace_context
from industrial_ops_agent.guardrails.prompt_injection import PromptInjectionGuard
from industrial_ops_agent.incident_operations.service import (
    IncidentOperationConflict,
    record_work_order_resolution_in_session,
)
from industrial_ops_agent.maintenance_planning.review_isolation import (
    record_incident_disclosure,
    work_order_read_transaction,
)
from industrial_ops_agent.multimodal.models import REVIEWABLE_VIDEO_EVENT_TYPES
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetRecord,
    AssetSiteLinkRecord,
    CustomerServiceUpdateRecord,
    EvidenceBundleRecord,
    IncidentRecord,
    MediaObjectRecord,
    PartIssueRecord,
    PartMaterialMovementRecord,
    PartReservationRecord,
    RecognitionCorrectionRecord,
    RecognitionResultRecord,
    RecognitionRunRecord,
    WorkOrderAssignmentRecord,
    WorkOrderCompletionRecord,
    WorkOrderControlRecord,
    WorkOrderEdgeDiagnosisCandidateRecord,
    WorkOrderEdgeDiagnosisPackRecord,
    WorkOrderEventRecord,
    WorkOrderFieldEntryRecord,
    WorkOrderFieldEvidenceUploadRecord,
    WorkOrderRecord,
    WorkOrderReworkRecord,
    WorkOrderVerificationRecord,
)
from industrial_ops_agent.workorders.assignment import is_eligible_field_engineer


class WorkOrderNotVisible(Exception):
    pass


class WorkOrderConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        super().__init__(reason)
        self.reason, self.current_version = reason, current_version


@dataclass(frozen=True, slots=True)
class WorkOrderView:
    work_order_id: str
    incident_id: str
    proposal_id: str
    reservation_id: str | None
    creation_mode: Literal["PARTS_RESERVATION", "SERVICE_AUTHORIZATION"]
    authorization_type: Literal["COVERED_SERVICE", "ACCEPTED_QUOTATION"] | None
    diagnosis_run_id: str | None
    diagnosis_version: int | None
    service_quotation_id: str | None
    initial_parts_required: bool
    part_issue_required: bool
    part_accounting_required: bool
    status: str
    assigned_subject_id: str | None
    pending_assignment_operation_id: str | None
    priority: str
    sla_due_at: datetime | None
    sla_status: str
    service_window_start: datetime | None
    service_window_end: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class WorkOrderClosureFacts:
    completion_id: str
    verification_id: str


@dataclass(frozen=True, slots=True)
class DispatchQueueItem:
    work_order: WorkOrderView
    asset_id: str
    asset_display_name: str | None
    model_code: str | None
    site_id: str | None
    site_name: str | None
    incident_description: str


@dataclass(frozen=True, slots=True)
class WorkOrderControlView:
    control_id: str
    work_order_version: int
    command_type: str
    previous_status: str
    target_status: str
    reason: str
    recovery_condition: str | None
    responsible_subject_id: str | None
    service_window_start: datetime | None
    service_window_end: datetime | None
    sla_due_at: datetime | None
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class WorkOrderFieldEntryView:
    entry_id: str
    sequence: int
    client_operation_id: str
    entry_type: str
    payload: dict[str, Any]
    actor_subject_id: str
    occurred_at: datetime
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class WorkOrderPartAllocationView:
    reservation_id: str
    part_number: str
    approved_quantity: int
    status: str
    source: str
    source_record_id: str
    as_of: datetime
    issue_required: bool
    issue_status: str
    usage_enabled: bool
    legal_actions: list[str]
    part_issue_id: str | None
    proposal_id: str | None
    approval_id: str | None
    operation_id: str | None
    reconciliation_id: str | None
    external_issue_id: str | None
    issue_source: str | None
    issue_source_record_id: str | None
    issue_as_of: datetime | None


@dataclass(frozen=True, slots=True)
class WorkOrderPartAccountingEntryView:
    entry_id: str
    quantity: int
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class WorkOrderPartMovementView:
    movement_id: str
    operation_id: str
    proposal_id: str
    approval_id: str
    movement_kind: str
    field_entry_id: str | None
    quantity: int
    status: str
    external_movement_id: str | None
    source: str | None
    source_record_id: str | None
    as_of: datetime | None
    reconciliation_id: str | None
    reason: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class WorkOrderPartAccountingView:
    mode: str
    work_order_id: str
    work_order_version: int
    reservation_id: str
    part_issue_id: str | None
    part_number: str
    issued_quantity: int
    recorded_usage_quantity: int
    consumed_quantity: int
    returned_quantity: int
    active_consumption_quantity: int
    active_return_quantity: int
    unaccounted_quantity: int
    returnable_quantity: int
    eligible_entries: list[WorkOrderPartAccountingEntryView]
    movements: list[WorkOrderPartMovementView]
    close_ready: bool
    blocking_reasons: list[str]
    legal_actions: list[str]


@dataclass(frozen=True, slots=True)
class WorkOrderRepairRoundView:
    round_number: int
    status: str
    completion_id: str | None
    completed_by_subject_id: str | None
    root_cause: str | None
    actions: list[str]
    evidence_ids: list[str]
    part_reservation_ids: list[str]
    cost_amount: str | None
    customer_confirmation: str | None
    field_entry_sequence_start: int | None
    field_entry_sequence_end: int | None
    completed_at: datetime | None
    verification_id: str | None
    verifier_subject_id: str | None
    verification_passed: bool | None
    verification_reason: str | None
    verified_at: datetime | None
    rework_id: str | None
    rework_status: str | None
    rework_reason: str | None
    entry_sequence_checkpoint: int | None
    rework_opened_at: datetime | None
    rework_resolved_at: datetime | None


@dataclass(frozen=True, slots=True)
class WorkOrderRepairHistoryView:
    work_order_id: str
    current_round: int
    rounds: list[WorkOrderRepairRoundView]


@dataclass(frozen=True, slots=True)
class WorkOrderClosurePartView:
    reservation_id: str
    part_number: str
    quantity: int
    status: str
    source: str
    as_of: datetime


@dataclass(frozen=True, slots=True)
class WorkOrderClosureReportView:
    report_contract_version: str
    closure_facts_digest: str
    work_order_id: str
    incident_id: str
    asset_id: str
    asset_display_name: str | None
    model_code: str | None
    serial_number: str | None
    site_id: str | None
    site_name: str | None
    closed_at: datetime
    closed_by_subject_id: str
    close_reason: str | None
    close_operation_id: str | None
    close_proposal_id: str | None
    repair_round_count: int
    rework_count: int
    completion_id: str
    completion_round_number: int
    completed_by_subject_id: str
    completed_at: datetime
    root_cause: str
    actions: list[str]
    evidence_ids: list[str]
    parts: list[WorkOrderClosurePartView]
    cost_amount: str | None
    field_customer_confirmation: str | None
    verification_id: str
    verifier_subject_id: str
    verification_reason: str
    verified_at: datetime
    customer_result_confirmation_id: str | None
    customer_result_accepted: bool
    customer_confirmed_by_subject_id: str | None
    customer_confirmed_at: datetime | None
    customer_satisfaction_rating: int | None


class WorkOrderService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database, self._authorizer = database, authorizer

    def get(
        self, identity: IdentityContext, work_order_id: str, *, request_id: str
    ) -> WorkOrderView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(session, identity.tenant_id, work_order_id)
            self._require(
                identity,
                Action.READ_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            return _view(work)

    def list_dispatch_queue(
        self,
        identity: IdentityContext,
        *,
        status: str | None,
        assigned_subject_id: str | None,
        priority: str | None = None,
        limit: int,
        offset: int,
        request_id: str,
    ) -> tuple[list[DispatchQueueItem], int]:
        """Return only work orders whose asset or site is in the caller's current scope."""

        self._authorizer.require(
            identity,
            Action.READ_WORK_ORDER,
            ResourceContext(identity.tenant_id),
            request_id=request_id,
        )
        scope_conditions = []
        if identity.asset_ids:
            scope_conditions.append(IncidentRecord.asset_id.in_(identity.asset_ids))
        if identity.site_ids:
            scope_conditions.append(AssetSiteLinkRecord.site_id.in_(identity.site_ids))
        if not scope_conditions:
            return [], 0

        statement = (
            select(
                WorkOrderRecord,
                IncidentRecord,
                AssetRecord,
                AssetSiteLinkRecord,
            )
            .join(
                IncidentRecord,
                IncidentRecord.incident_id == WorkOrderRecord.incident_id,
            )
            .join(AssetRecord, AssetRecord.asset_id == IncidentRecord.asset_id)
            .outerjoin(
                AssetSiteLinkRecord,
                AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
            )
            .where(
                WorkOrderRecord.tenant_id == identity.tenant_id,
                IncidentRecord.tenant_id == identity.tenant_id,
                AssetRecord.tenant_id == identity.tenant_id,
                or_(*scope_conditions),
            )
        )
        if status is not None:
            statement = statement.where(WorkOrderRecord.status == status)
        if assigned_subject_id is not None:
            statement = statement.where(WorkOrderRecord.assigned_subject_id == assigned_subject_id)
        if priority is not None:
            statement = statement.where(WorkOrderRecord.priority == priority)

        with self._database.transaction(identity.tenant_context) as session:
            total = int(session.scalar(select(func.count()).select_from(statement.subquery())) or 0)
            rows = (
                session.execute(
                    statement.order_by(
                        WorkOrderRecord.updated_at.desc(), WorkOrderRecord.work_order_id
                    )
                    .limit(limit)
                    .offset(offset)
                )
                .tuples()
                .all()
            )
            record_incident_disclosure(
                session,
                identity,
                (incident.incident_id for _work, incident, _asset, _site in rows),
                resource_kind="work-order-list",
                resource_id="work-orders",
                request_id=request_id,
            )
            return [
                DispatchQueueItem(
                    work_order=_view(work),
                    asset_id=incident.asset_id,
                    asset_display_name=asset.display_name,
                    model_code=asset.model_code,
                    site_id=site.site_id if site is not None else None,
                    site_name=site.site_name if site is not None else None,
                    incident_description=incident.description,
                )
                for work, incident, asset, site in rows
            ], total

    def assign(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        assignee_subject_id: str,
        expected_version: int,
        request_id: str,
    ) -> WorkOrderView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            return self.assign_in_session(
                session,
                identity,
                work_order_id,
                assignee_subject_id=assignee_subject_id,
                expected_version=expected_version,
                request_id=request_id,
            )

    def assign_in_session(
        self,
        session: Session,
        identity: IdentityContext,
        work_order_id: str,
        *,
        assignee_subject_id: str,
        expected_version: int,
        request_id: str,
        reason: str | None = None,
        operation_id: str | None = None,
        proposal_id: str | None = None,
        fsm_assignment_delivery_id: str | None = None,
    ) -> WorkOrderView:
        """Apply the canonical assignment transition inside the caller's transaction."""

        work, incident, site_id = self._load(
            session,
            identity.tenant_id,
            work_order_id,
            for_update=True,
        )
        self._require(
            identity,
            Action.ASSIGN_WORK_ORDER,
            work_order_id,
            incident.asset_id,
            site_id,
            request_id,
        )
        if work.pending_assignment_operation_id is not None and (
            work.pending_assignment_operation_id != operation_id
        ):
            raise WorkOrderConflict("assignment_pending", work.version)
        if not is_eligible_field_engineer(
            session,
            tenant_id=identity.tenant_id,
            subject_id=assignee_subject_id,
            asset_id=incident.asset_id,
            site_id=site_id,
        ):
            raise WorkOrderConflict("assignee_not_eligible", work.version)
        event_payload = {"assignee_subject_id": assignee_subject_id}
        if reason is not None:
            event_payload["reason"] = reason
        self._transition(
            work,
            expected_version,
            "READY",
            "ASSIGNED",
            identity.subject_id,
            session,
            event_payload,
        )
        work.pending_assignment_operation_id = None
        work.assigned_subject_id = assignee_subject_id
        now = datetime.now(UTC)
        session.add(
            WorkOrderAssignmentRecord(
                assignment_id=f"assignment-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                work_order_id=work_order_id,
                assignee_subject_id=assignee_subject_id,
                assigned_by_subject_id=identity.subject_id,
                assigned_at=now,
                operation_id=operation_id,
                proposal_id=proposal_id,
                fsm_assignment_delivery_id=fsm_assignment_delivery_id,
                created_at=now,
                updated_at=now,
            )
        )
        return _view(work)

    def hold(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        reason: str,
        recovery_condition: str,
        request_id: str,
    ) -> WorkOrderView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session, identity.tenant_id, work_order_id, for_update=True
            )
            self._require(
                identity,
                Action.CONTROL_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            self._require_assignee_or_coordinator(identity, work)
            previous_status = work.status
            self._transition(
                work,
                expected_version,
                "IN_PROGRESS",
                "ON_HOLD",
                identity.subject_id,
                session,
                {"reason": reason, "recovery_condition": recovery_condition},
            )
            self._record_control(
                session,
                work,
                command_type="HOLD",
                previous_status=previous_status,
                reason=reason,
                recovery_condition=recovery_condition,
                responsible_subject_id=work.assigned_subject_id,
            )
            return _view(work)

    def resume(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        reason: str,
        request_id: str,
    ) -> WorkOrderView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session, identity.tenant_id, work_order_id, for_update=True
            )
            self._require(
                identity,
                Action.CONTROL_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            self._require_assignee_or_coordinator(identity, work)
            previous_status = work.status
            self._transition(
                work,
                expected_version,
                "ON_HOLD",
                "IN_PROGRESS",
                identity.subject_id,
                session,
                {"reason": reason},
            )
            self._record_control(
                session,
                work,
                command_type="RESUME",
                previous_status=previous_status,
                reason=reason,
                recovery_condition=None,
                responsible_subject_id=work.assigned_subject_id,
            )
            return _view(work)

    def escalate(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        reason: str,
        recovery_condition: str,
        responsible_subject_id: str,
        request_id: str,
    ) -> WorkOrderView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session, identity.tenant_id, work_order_id, for_update=True
            )
            self._require(
                identity,
                Action.ESCALATE_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            if work.status not in {"IN_PROGRESS", "ON_HOLD"}:
                raise WorkOrderConflict("illegal_state_transition", work.version)
            previous_status = work.status
            self._transition(
                work,
                expected_version,
                previous_status,
                "ESCALATED",
                identity.subject_id,
                session,
                {
                    "reason": reason,
                    "recovery_condition": recovery_condition,
                    "responsible_subject_id": responsible_subject_id,
                },
            )
            work.assigned_subject_id = responsible_subject_id
            self._record_control(
                session,
                work,
                command_type="ESCALATE",
                previous_status=previous_status,
                reason=reason,
                recovery_condition=recovery_condition,
                responsible_subject_id=responsible_subject_id,
            )
            return _view(work)

    def replan(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        target_status: str,
        reason: str,
        responsible_subject_id: str | None,
        service_window_start: datetime,
        service_window_end: datetime,
        sla_due_at: datetime,
        request_id: str,
    ) -> WorkOrderView:
        _validate_service_window(service_window_start, service_window_end, sla_due_at)
        if target_status not in {"READY", "IN_PROGRESS"}:
            raise WorkOrderConflict("replan_target_invalid")
        if target_status == "IN_PROGRESS" and not responsible_subject_id:
            raise WorkOrderConflict("replan_responsible_subject_required")
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session, identity.tenant_id, work_order_id, for_update=True
            )
            self._require(
                identity,
                Action.ESCALATE_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            previous_status = work.status
            self._transition(
                work,
                expected_version,
                "ESCALATED",
                target_status,
                identity.subject_id,
                session,
                {
                    "reason": reason,
                    "responsible_subject_id": responsible_subject_id,
                    "service_window_start": service_window_start.isoformat(),
                    "service_window_end": service_window_end.isoformat(),
                    "sla_due_at": sla_due_at.isoformat(),
                },
            )
            work.assigned_subject_id = (
                responsible_subject_id if target_status == "IN_PROGRESS" else None
            )
            work.service_window_start = service_window_start
            work.service_window_end = service_window_end
            work.sla_due_at = sla_due_at
            self._record_control(
                session,
                work,
                command_type="REPLAN",
                previous_status=previous_status,
                reason=reason,
                recovery_condition=None,
                responsible_subject_id=responsible_subject_id,
            )
            return _view(work)

    def reschedule(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        priority: str,
        reason: str,
        service_window_start: datetime,
        service_window_end: datetime,
        sla_due_at: datetime,
        request_id: str,
    ) -> WorkOrderView:
        _validate_service_window(service_window_start, service_window_end, sla_due_at)
        if priority not in {"CRITICAL", "HIGH", "NORMAL", "LOW"}:
            raise WorkOrderConflict("priority_invalid")
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session, identity.tenant_id, work_order_id, for_update=True
            )
            self._require(
                identity,
                Action.CONTROL_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            self._require_assignee_or_coordinator(identity, work)
            if work.version != expected_version:
                raise WorkOrderConflict("version_conflict", work.version)
            if work.pending_assignment_operation_id is not None:
                raise WorkOrderConflict("assignment_pending", work.version)
            if work.status in {"CLOSED", "CANCELLED"}:
                raise WorkOrderConflict("illegal_state_transition", work.version)
            previous_status = work.status
            now = datetime.now(UTC)
            work.priority = priority
            work.service_window_start = service_window_start
            work.service_window_end = service_window_end
            work.sla_due_at = sla_due_at
            work.version += 1
            work.updated_at = now
            session.add(
                WorkOrderEventRecord(
                    event_id=f"work-event-{uuid4().hex}",
                    tenant_id=work.tenant_id,
                    work_order_id=work.work_order_id,
                    sequence=work.version,
                    event_type="work_order.rescheduled",
                    actor_subject_id=identity.subject_id,
                    payload={
                        "priority": priority,
                        "reason": reason,
                        "service_window_start": service_window_start.isoformat(),
                        "service_window_end": service_window_end.isoformat(),
                        "sla_due_at": sla_due_at.isoformat(),
                    },
                    occurred_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            self._record_control(
                session,
                work,
                command_type="RESCHEDULE",
                previous_status=previous_status,
                reason=reason,
                recovery_condition=None,
                responsible_subject_id=work.assigned_subject_id,
            )
            return _view(work)

    def list_controls(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> list[WorkOrderControlView]:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            _work, incident, site_id = self._load(session, identity.tenant_id, work_order_id)
            self._require(
                identity,
                Action.READ_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            records = session.scalars(
                select(WorkOrderControlRecord)
                .where(
                    WorkOrderControlRecord.tenant_id == identity.tenant_id,
                    WorkOrderControlRecord.work_order_id == work_order_id,
                )
                .order_by(WorkOrderControlRecord.work_order_version)
            )
            return [_control_view(record) for record in records]

    def list_field_entries(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> list[WorkOrderFieldEntryView]:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(session, identity.tenant_id, work_order_id)
            self._require(
                identity,
                Action.EXECUTE_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            if work.assigned_subject_id != identity.subject_id:
                raise WorkOrderConflict("assignee_mismatch", work.version)
            records = session.scalars(
                select(WorkOrderFieldEntryRecord)
                .where(
                    WorkOrderFieldEntryRecord.tenant_id == identity.tenant_id,
                    WorkOrderFieldEntryRecord.work_order_id == work_order_id,
                )
                .order_by(WorkOrderFieldEntryRecord.sequence)
            )
            return [_field_entry_view(record) for record in records]

    def get_part_allocation(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> WorkOrderPartAllocationView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session,
                identity.tenant_id,
                work_order_id,
            )
            self._require(
                identity,
                Action.EXECUTE_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            if work.assigned_subject_id != identity.subject_id:
                raise WorkOrderConflict("assignee_mismatch", work.version)
            if work.creation_mode == "SERVICE_AUTHORIZATION":
                raise WorkOrderConflict("parts_not_required", work.version)
            reservation = self._load_available_part_allocation(session, work)
            issue = session.scalar(
                select(PartIssueRecord).where(
                    PartIssueRecord.tenant_id == work.tenant_id,
                    PartIssueRecord.work_order_id == work.work_order_id,
                )
            )
            if not work.part_issue_required:
                issue_status = "LEGACY_COMPATIBLE"
                usage_enabled = True
                legal_actions: list[str] = []
            elif issue is None:
                issue_status = "NOT_REQUESTED"
                usage_enabled = False
                legal_actions = (
                    ["PROPOSE_ISSUE"]
                    if work.status in {"ACCEPTED", "IN_PROGRESS", "ON_HOLD"}
                    and reservation.status == "RESERVED"
                    else []
                )
            else:
                issue_status = issue.status
                usage_enabled = issue.status == "ISSUED" and reservation.status == "ISSUED"
                legal_actions = []
            return WorkOrderPartAllocationView(
                reservation_id=reservation.reservation_id,
                part_number=reservation.part_number,
                approved_quantity=reservation.quantity,
                status=reservation.status,
                source=reservation.source,
                source_record_id=reservation.source_record_id,
                as_of=_utc(reservation.as_of),
                issue_required=work.part_issue_required,
                issue_status=issue_status,
                usage_enabled=usage_enabled,
                legal_actions=legal_actions,
                part_issue_id=issue.part_issue_id if issue is not None else None,
                proposal_id=issue.proposal_id if issue is not None else None,
                approval_id=issue.approval_id if issue is not None else None,
                operation_id=issue.operation_id if issue is not None else None,
                reconciliation_id=issue.reconciliation_id if issue is not None else None,
                external_issue_id=issue.external_issue_id if issue is not None else None,
                issue_source=issue.source if issue is not None else None,
                issue_source_record_id=issue.source_record_id if issue is not None else None,
                issue_as_of=(
                    _utc(issue.as_of) if issue is not None and issue.as_of is not None else None
                ),
            )

    def get_part_accounting(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> WorkOrderPartAccountingView:
        """Return the server-authoritative issue/usage/consumption/return ledger."""

        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session,
                identity.tenant_id,
                work_order_id,
            )
            self._require(
                identity,
                Action.EXECUTE_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            if work.assigned_subject_id != identity.subject_id:
                raise WorkOrderConflict("assignee_mismatch", work.version)
            return self.part_accounting_in_session(session, work)

    @classmethod
    def part_accounting_in_session(
        cls,
        session: Session,
        work: WorkOrderRecord,
    ) -> WorkOrderPartAccountingView:
        if work.creation_mode == "SERVICE_AUTHORIZATION":
            raise WorkOrderConflict("parts_not_required", work.version)
        reservation = cls._load_available_part_allocation(session, work)
        issue = session.scalar(
            select(PartIssueRecord).where(
                PartIssueRecord.tenant_id == work.tenant_id,
                PartIssueRecord.work_order_id == work.work_order_id,
                PartIssueRecord.reservation_id == reservation.reservation_id,
            )
        )
        if not work.part_accounting_required:
            return WorkOrderPartAccountingView(
                mode="LEGACY_ACCOUNTING_COMPATIBLE",
                work_order_id=work.work_order_id,
                work_order_version=work.version,
                reservation_id=reservation.reservation_id,
                part_issue_id=issue.part_issue_id if issue is not None else None,
                part_number=reservation.part_number,
                issued_quantity=(
                    issue.quantity
                    if issue is not None and issue.status == "ISSUED"
                    else reservation.quantity
                ),
                recorded_usage_quantity=0,
                consumed_quantity=0,
                returned_quantity=0,
                active_consumption_quantity=0,
                active_return_quantity=0,
                unaccounted_quantity=0,
                returnable_quantity=0,
                eligible_entries=[],
                movements=[],
                close_ready=True,
                blocking_reasons=[],
                legal_actions=[],
            )
        if (
            issue is None
            or issue.status != "ISSUED"
            or reservation.status != "ISSUED"
            or issue.quantity != reservation.quantity
            or issue.part_number != reservation.part_number
        ):
            raise WorkOrderConflict("part_issue_not_issued", work.version)

        entries = list(
            session.scalars(
                select(WorkOrderFieldEntryRecord)
                .where(
                    WorkOrderFieldEntryRecord.tenant_id == work.tenant_id,
                    WorkOrderFieldEntryRecord.work_order_id == work.work_order_id,
                    WorkOrderFieldEntryRecord.entry_type == "PART",
                )
                .order_by(WorkOrderFieldEntryRecord.sequence)
            )
        )
        movements = list(
            session.scalars(
                select(PartMaterialMovementRecord)
                .where(
                    PartMaterialMovementRecord.tenant_id == work.tenant_id,
                    PartMaterialMovementRecord.work_order_id == work.work_order_id,
                    PartMaterialMovementRecord.part_issue_id == issue.part_issue_id,
                )
                .order_by(PartMaterialMovementRecord.created_at)
            )
        )
        active_statuses = {"PENDING_APPROVAL", "APPROVED", "RECONCILING"}
        recorded_usage = sum(cls._part_entry_quantity(item, work.version) for item in entries)
        consumed = sum(
            item.quantity
            for item in movements
            if item.movement_kind == "CONSUME" and item.status == "APPLIED"
        )
        returned = sum(
            item.quantity
            for item in movements
            if item.movement_kind == "RETURN" and item.status == "APPLIED"
        )
        active_consumption = sum(
            item.quantity
            for item in movements
            if item.movement_kind == "CONSUME" and item.status in active_statuses
        )
        active_return = sum(
            item.quantity
            for item in movements
            if item.movement_kind == "RETURN" and item.status in active_statuses
        )
        occupied_entries = {
            item.field_entry_id
            for item in movements
            if (
                item.movement_kind == "CONSUME"
                and item.field_entry_id is not None
                and item.status in active_statuses | {"APPLIED"}
            )
        }
        eligible = [
            WorkOrderPartAccountingEntryView(
                entry_id=item.entry_id,
                quantity=cls._part_entry_quantity(item, work.version),
                occurred_at=_utc(item.occurred_at),
            )
            for item in entries
            if item.entry_id not in occupied_entries
        ]
        in_progress = any(item.status in active_statuses for item in movements)
        unconsumed = any(
            not any(
                movement.movement_kind == "CONSUME"
                and movement.field_entry_id == entry.entry_id
                and movement.status == "APPLIED"
                for movement in movements
            )
            for entry in entries
        )
        balance = issue.quantity - consumed - returned
        unaccounted = max(balance, 0)
        blocking_reasons: list[str] = []
        if unconsumed:
            blocking_reasons.append("unconsumed_field_entries")
        if balance != 0:
            blocking_reasons.append("unaccounted_issued_quantity")
        if in_progress:
            blocking_reasons.append("part_movement_in_progress")
        legal_actions: list[str] = []
        if work.status in {"ACCEPTED", "IN_PROGRESS", "ON_HOLD"}:
            if eligible:
                legal_actions.append("PROPOSE_CONSUMPTION")
            if issue.quantity - recorded_usage - returned - active_return > 0:
                legal_actions.append("PROPOSE_RETURN")
        return WorkOrderPartAccountingView(
            mode="AUTHORITATIVE",
            work_order_id=work.work_order_id,
            work_order_version=work.version,
            reservation_id=reservation.reservation_id,
            part_issue_id=issue.part_issue_id,
            part_number=issue.part_number,
            issued_quantity=issue.quantity,
            recorded_usage_quantity=recorded_usage,
            consumed_quantity=consumed,
            returned_quantity=returned,
            active_consumption_quantity=active_consumption,
            active_return_quantity=active_return,
            unaccounted_quantity=unaccounted,
            returnable_quantity=max(
                issue.quantity - recorded_usage - returned - active_return,
                0,
            ),
            eligible_entries=eligible,
            movements=[cls._part_movement_view(item) for item in movements],
            close_ready=not blocking_reasons,
            blocking_reasons=blocking_reasons,
            legal_actions=legal_actions,
        )

    @staticmethod
    def _part_entry_quantity(
        entry: WorkOrderFieldEntryRecord,
        current_version: int,
    ) -> int:
        quantity = entry.payload.get("quantity")
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            raise WorkOrderConflict("part_entry_quantity_invalid", current_version)
        return quantity

    @staticmethod
    def _part_movement_view(
        item: PartMaterialMovementRecord,
    ) -> WorkOrderPartMovementView:
        return WorkOrderPartMovementView(
            movement_id=item.movement_id,
            operation_id=item.operation_id,
            proposal_id=item.proposal_id,
            approval_id=item.approval_id,
            movement_kind=item.movement_kind,
            field_entry_id=item.field_entry_id,
            quantity=item.quantity,
            status=item.status,
            external_movement_id=item.external_movement_id,
            source=item.source,
            source_record_id=item.source_record_id,
            as_of=_utc(item.as_of) if item.as_of is not None else None,
            reconciliation_id=item.reconciliation_id,
            reason=item.reason,
            created_at=_utc(item.created_at),
            updated_at=_utc(item.updated_at),
        )

    def repair_history(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> WorkOrderRepairHistoryView:
        """Project immutable completion, verification, and rework facts by round."""

        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            _work, incident, site_id = self._load(
                session,
                identity.tenant_id,
                work_order_id,
            )
            self._require(
                identity,
                Action.READ_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            completions = list(
                session.scalars(
                    select(WorkOrderCompletionRecord)
                    .where(
                        WorkOrderCompletionRecord.tenant_id == identity.tenant_id,
                        WorkOrderCompletionRecord.work_order_id == work_order_id,
                    )
                    .order_by(WorkOrderCompletionRecord.round_number)
                )
            )
            verifications = list(
                session.scalars(
                    select(WorkOrderVerificationRecord)
                    .where(
                        WorkOrderVerificationRecord.tenant_id == identity.tenant_id,
                        WorkOrderVerificationRecord.work_order_id == work_order_id,
                    )
                    .order_by(WorkOrderVerificationRecord.round_number)
                )
            )
            reworks = list(
                session.scalars(
                    select(WorkOrderReworkRecord)
                    .where(
                        WorkOrderReworkRecord.tenant_id == identity.tenant_id,
                        WorkOrderReworkRecord.work_order_id == work_order_id,
                    )
                    .order_by(WorkOrderReworkRecord.round_number)
                )
            )
            completion_by_round = {item.round_number: item for item in completions}
            verification_by_round = {item.round_number: item for item in verifications}
            rework_by_round = {item.round_number: item for item in reworks}
            current_round = max([1, *completion_by_round.keys(), *rework_by_round.keys()])
            rounds: list[WorkOrderRepairRoundView] = []
            for round_number in range(1, current_round + 1):
                completion = completion_by_round.get(round_number)
                verification = verification_by_round.get(round_number)
                rework = rework_by_round.get(round_number)
                if completion is None:
                    status = "REWORK_IN_PROGRESS" if rework is not None else "IN_PROGRESS"
                elif verification is None:
                    status = "AWAITING_VERIFICATION"
                elif verification.passed:
                    status = "VERIFIED"
                else:
                    status = "REWORK_REQUIRED"
                rounds.append(
                    WorkOrderRepairRoundView(
                        round_number=round_number,
                        status=status,
                        completion_id=(completion.completion_id if completion else None),
                        completed_by_subject_id=(
                            completion.completed_by_subject_id if completion else None
                        ),
                        root_cause=(completion.root_cause if completion else None),
                        actions=(completion.actions if completion else []),
                        evidence_ids=(completion.evidence_ids if completion else []),
                        part_reservation_ids=(
                            completion.part_reservation_ids if completion else []
                        ),
                        cost_amount=(completion.cost_amount if completion else None),
                        customer_confirmation=(
                            completion.customer_confirmation if completion else None
                        ),
                        field_entry_sequence_start=(
                            completion.field_entry_sequence_start if completion else None
                        ),
                        field_entry_sequence_end=(
                            completion.field_entry_sequence_end if completion else None
                        ),
                        completed_at=(completion.completed_at if completion else None),
                        verification_id=(verification.verification_id if verification else None),
                        verifier_subject_id=(
                            verification.verifier_subject_id if verification else None
                        ),
                        verification_passed=(verification.passed if verification else None),
                        verification_reason=(verification.reason if verification else None),
                        verified_at=(verification.verified_at if verification else None),
                        rework_id=(rework.rework_id if rework else None),
                        rework_status=(rework.status if rework else None),
                        rework_reason=(rework.reason if rework else None),
                        entry_sequence_checkpoint=(
                            rework.entry_sequence_checkpoint if rework else None
                        ),
                        rework_opened_at=(rework.opened_at if rework else None),
                        rework_resolved_at=(rework.resolved_at if rework else None),
                    )
                )
            return WorkOrderRepairHistoryView(
                work_order_id=work_order_id,
                current_round=current_round,
                rounds=rounds,
            )

    def closure_report(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> WorkOrderClosureReportView:
        """Project the immutable facts accepted by the final close transition."""

        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session,
                identity.tenant_id,
                work_order_id,
            )
            self._require(
                identity,
                Action.READ_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            if work.status != "CLOSED":
                raise WorkOrderConflict("closure_report_not_ready", work.version)
            asset = session.scalar(
                select(AssetRecord).where(
                    AssetRecord.tenant_id == identity.tenant_id,
                    AssetRecord.asset_id == incident.asset_id,
                )
            )
            site = session.scalar(
                select(AssetSiteLinkRecord).where(
                    AssetSiteLinkRecord.tenant_id == identity.tenant_id,
                    AssetSiteLinkRecord.asset_id == incident.asset_id,
                )
            )
            close_event = session.scalar(
                select(WorkOrderEventRecord).where(
                    WorkOrderEventRecord.tenant_id == identity.tenant_id,
                    WorkOrderEventRecord.work_order_id == work_order_id,
                    WorkOrderEventRecord.event_type == "work_order.closed",
                )
            )
            completion = session.scalar(
                select(WorkOrderCompletionRecord)
                .where(
                    WorkOrderCompletionRecord.tenant_id == identity.tenant_id,
                    WorkOrderCompletionRecord.work_order_id == work_order_id,
                )
                .order_by(WorkOrderCompletionRecord.round_number.desc())
            )
            verification = (
                session.scalar(
                    select(WorkOrderVerificationRecord).where(
                        WorkOrderVerificationRecord.tenant_id == identity.tenant_id,
                        WorkOrderVerificationRecord.completion_id == completion.completion_id,
                        WorkOrderVerificationRecord.passed.is_(True),
                    )
                )
                if completion is not None
                else None
            )
            customer_confirmation = session.scalar(
                select(CustomerServiceUpdateRecord)
                .where(
                    CustomerServiceUpdateRecord.tenant_id == identity.tenant_id,
                    CustomerServiceUpdateRecord.work_order_id == work_order_id,
                    CustomerServiceUpdateRecord.update_type == "RESULT_CONFIRMATION",
                    CustomerServiceUpdateRecord.result_accepted.is_(True),
                )
                .order_by(
                    CustomerServiceUpdateRecord.occurred_at.desc(),
                    CustomerServiceUpdateRecord.created_at.desc(),
                )
            )
            if asset is None or close_event is None or completion is None or verification is None:
                raise WorkOrderConflict("closure_report_facts_incomplete", work.version)
            part_records = list(
                session.scalars(
                    select(PartReservationRecord)
                    .where(
                        PartReservationRecord.tenant_id == identity.tenant_id,
                        PartReservationRecord.reservation_id.in_(completion.part_reservation_ids),
                    )
                    .order_by(PartReservationRecord.reservation_id)
                )
            )
            if len(part_records) != len(set(completion.part_reservation_ids)):
                raise WorkOrderConflict("closure_report_facts_incomplete", work.version)
            customer_result_accepted = customer_confirmation is not None or bool(
                completion.customer_confirmation
            )
            if not customer_result_accepted:
                raise WorkOrderConflict("closure_report_facts_incomplete", work.version)
            parts = [
                WorkOrderClosurePartView(
                    reservation_id=item.reservation_id,
                    part_number=item.part_number,
                    quantity=item.quantity,
                    status=item.status,
                    source=item.source,
                    as_of=_utc(item.as_of),
                )
                for item in part_records
            ]
            close_reason = _optional_payload_text(close_event.payload, "reason")
            close_operation_id = _optional_payload_text(
                close_event.payload,
                "operation_id",
            )
            close_proposal_id = _optional_payload_text(
                close_event.payload,
                "proposal_id",
            )
            repair_round_count = int(
                session.scalar(
                    select(func.count(WorkOrderCompletionRecord.completion_id)).where(
                        WorkOrderCompletionRecord.tenant_id == identity.tenant_id,
                        WorkOrderCompletionRecord.work_order_id == work_order_id,
                    )
                )
                or 0
            )
            rework_count = int(
                session.scalar(
                    select(func.count(WorkOrderReworkRecord.rework_id)).where(
                        WorkOrderReworkRecord.tenant_id == identity.tenant_id,
                        WorkOrderReworkRecord.work_order_id == work_order_id,
                    )
                )
                or 0
            )
            digest_values: dict[str, Any] = {
                "work_order_id": work_order_id,
                "incident_id": incident.incident_id,
                "asset_id": incident.asset_id,
                "closed_at": _utc(close_event.occurred_at),
                "closed_by_subject_id": close_event.actor_subject_id,
                "close_reason": close_reason,
                "close_operation_id": close_operation_id,
                "close_proposal_id": close_proposal_id,
                "repair_round_count": repair_round_count,
                "rework_count": rework_count,
                "completion_id": completion.completion_id,
                "completion_round_number": completion.round_number,
                "completed_by_subject_id": completion.completed_by_subject_id,
                "completed_at": _utc(completion.completed_at),
                "root_cause": completion.root_cause,
                "actions": completion.actions,
                "evidence_ids": completion.evidence_ids,
                "parts": [
                    {
                        "reservation_id": item.reservation_id,
                        "part_number": item.part_number,
                        "quantity": item.quantity,
                        "status": item.status,
                        "source": item.source,
                        "as_of": item.as_of,
                    }
                    for item in parts
                ],
                "cost_amount": completion.cost_amount,
                "field_customer_confirmation": completion.customer_confirmation,
                "verification_id": verification.verification_id,
                "verifier_subject_id": verification.verifier_subject_id,
                "verification_reason": verification.reason,
                "verified_at": _utc(verification.verified_at),
                "customer_result_confirmation_id": (
                    customer_confirmation.update_id if customer_confirmation else None
                ),
                "customer_result_accepted": customer_result_accepted,
                "customer_confirmed_by_subject_id": (
                    customer_confirmation.actor_subject_id if customer_confirmation else None
                ),
                "customer_confirmed_at": (
                    _utc(customer_confirmation.occurred_at) if customer_confirmation else None
                ),
                "customer_satisfaction_rating": (
                    customer_confirmation.satisfaction_rating if customer_confirmation else None
                ),
            }
            return WorkOrderClosureReportView(
                report_contract_version="work-order-closure-report/v1",
                closure_facts_digest=_closure_report_digest(digest_values),
                work_order_id=work_order_id,
                incident_id=incident.incident_id,
                asset_id=incident.asset_id,
                asset_display_name=asset.display_name,
                model_code=asset.model_code,
                serial_number=asset.serial_number,
                site_id=site.site_id if site else None,
                site_name=site.site_name if site else None,
                closed_at=_utc(close_event.occurred_at),
                closed_by_subject_id=close_event.actor_subject_id,
                close_reason=close_reason,
                close_operation_id=close_operation_id,
                close_proposal_id=close_proposal_id,
                repair_round_count=repair_round_count,
                rework_count=rework_count,
                completion_id=completion.completion_id,
                completion_round_number=completion.round_number,
                completed_by_subject_id=completion.completed_by_subject_id,
                completed_at=_utc(completion.completed_at),
                root_cause=completion.root_cause,
                actions=completion.actions,
                evidence_ids=completion.evidence_ids,
                parts=parts,
                cost_amount=completion.cost_amount,
                field_customer_confirmation=completion.customer_confirmation,
                verification_id=verification.verification_id,
                verifier_subject_id=verification.verifier_subject_id,
                verification_reason=verification.reason,
                verified_at=_utc(verification.verified_at),
                customer_result_confirmation_id=(
                    customer_confirmation.update_id if customer_confirmation else None
                ),
                customer_result_accepted=customer_result_accepted,
                customer_confirmed_by_subject_id=(
                    customer_confirmation.actor_subject_id if customer_confirmation else None
                ),
                customer_confirmed_at=(
                    _utc(customer_confirmation.occurred_at) if customer_confirmation else None
                ),
                customer_satisfaction_rating=(
                    customer_confirmation.satisfaction_rating if customer_confirmation else None
                ),
            )

    def append_field_entry(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        client_operation_id: str,
        entry_type: str,
        payload: dict[str, Any],
        occurred_at: datetime,
        request_id: str,
    ) -> WorkOrderFieldEntryView:
        if entry_type not in {
            "STEP",
            "PART",
            "EVIDENCE",
            "NOTE",
            "SIGNATURE",
            "AI_OBSERVATION",
        }:
            raise WorkOrderConflict("field_entry_type_invalid")
        occurred_at = _utc(occurred_at)
        now = datetime.now(UTC)
        if occurred_at > now + timedelta(minutes=5):
            raise WorkOrderConflict("field_entry_time_in_future")
        if occurred_at < now - timedelta(days=30):
            raise WorkOrderConflict("field_entry_offline_window_expired")
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session,
                identity.tenant_id,
                work_order_id,
                for_update=True,
            )
            self._require(
                identity,
                Action.EXECUTE_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            if work.assigned_subject_id != identity.subject_id:
                raise WorkOrderConflict("assignee_mismatch", work.version)
            existing = session.scalar(
                select(WorkOrderFieldEntryRecord).where(
                    WorkOrderFieldEntryRecord.tenant_id == identity.tenant_id,
                    WorkOrderFieldEntryRecord.work_order_id == work_order_id,
                    WorkOrderFieldEntryRecord.client_operation_id == client_operation_id,
                )
            )
            if existing is not None:
                if (
                    existing.entry_type != entry_type
                    or not self._field_entry_payload_matches(
                        existing.payload,
                        entry_type,
                        payload,
                    )
                    or _utc(existing.occurred_at) != occurred_at
                ):
                    raise WorkOrderConflict("idempotency_payload_mismatch", work.version)
                return _field_entry_view(existing)
            if work.status not in {"ACCEPTED", "IN_PROGRESS", "ON_HOLD"}:
                raise WorkOrderConflict("field_entry_state_invalid", work.version)
            if entry_type == "PART":
                if work.creation_mode == "SERVICE_AUTHORIZATION":
                    raise WorkOrderConflict("parts_not_required", work.version)
                self._validate_part_entry(session, work, payload)
            elif entry_type == "EVIDENCE":
                self._validate_evidence_entry(
                    session,
                    work,
                    incident,
                    identity,
                    payload,
                )
            elif entry_type == "AI_OBSERVATION":
                payload = self._normalize_ai_observation_entry(
                    session,
                    work,
                    incident,
                    identity,
                    payload,
                )
            sequence = (
                int(
                    session.scalar(
                        select(func.max(WorkOrderFieldEntryRecord.sequence)).where(
                            WorkOrderFieldEntryRecord.tenant_id == identity.tenant_id,
                            WorkOrderFieldEntryRecord.work_order_id == work_order_id,
                        )
                    )
                    or 0
                )
                + 1
            )
            record = WorkOrderFieldEntryRecord(
                entry_id=f"field-entry-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                work_order_id=work_order_id,
                sequence=sequence,
                client_operation_id=client_operation_id,
                entry_type=entry_type,
                payload=payload,
                actor_subject_id=identity.subject_id,
                occurred_at=occurred_at,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            return _field_entry_view(record)

    @staticmethod
    def _field_entry_payload_matches(
        stored_payload: dict[str, Any],
        entry_type: str,
        requested_payload: dict[str, Any],
    ) -> bool:
        if entry_type != "AI_OBSERVATION":
            return stored_payload == requested_payload
        stored_sources = stored_payload.get("sources")
        requested_sources = requested_payload.get("source_items")
        if not isinstance(stored_sources, list) or not isinstance(requested_sources, list):
            return False
        stored_source_items = sorted(
            (
                {
                    "source_type": item.get("source_type"),
                    "source_id": item.get("source_id"),
                }
                for item in stored_sources
                if isinstance(item, dict)
            ),
            key=lambda item: (str(item["source_type"]), str(item["source_id"])),
        )
        normalized_requested_sources = sorted(
            (
                {
                    "source_type": item.get("source_type"),
                    "source_id": item.get("source_id"),
                }
                for item in requested_sources
                if isinstance(item, dict)
            ),
            key=lambda item: (str(item["source_type"]), str(item["source_id"])),
        )
        observation = requested_payload.get("observation")
        return (
            len(stored_source_items) == len(stored_sources)
            and len(normalized_requested_sources) == len(requested_sources)
            and stored_payload.get("observation")
            == (observation.strip() if isinstance(observation, str) else observation)
            and stored_payload.get("field_evidence_upload_id")
            == requested_payload.get("field_evidence_upload_id")
            and stored_payload.get("recognition_bundle_id")
            == requested_payload.get("recognition_bundle_id")
            and stored_payload.get("recognition_bundle_version")
            == requested_payload.get("recognition_bundle_version")
            and stored_payload.get("edge_diagnosis_candidate_id")
            == requested_payload.get("edge_diagnosis_candidate_id")
            and stored_payload.get("edge_diagnosis_candidate_version")
            == requested_payload.get("edge_diagnosis_candidate_version")
            and stored_source_items == normalized_requested_sources
        )

    @staticmethod
    def _normalize_ai_observation_entry(
        session: Session,
        work: WorkOrderRecord,
        incident: IncidentRecord,
        identity: IdentityContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        observation = payload.get("observation")
        upload_id = payload.get("field_evidence_upload_id")
        bundle_id = payload.get("recognition_bundle_id")
        bundle_version = payload.get("recognition_bundle_version")
        candidate_id = payload.get("edge_diagnosis_candidate_id")
        candidate_version = payload.get("edge_diagnosis_candidate_version")
        source_items = payload.get("source_items")
        recognition_binding_present = any(
            value is not None for value in (upload_id, bundle_id, bundle_version)
        )
        edge_binding_present = any(value is not None for value in (candidate_id, candidate_version))
        source_types = {
            item.get("source_type") for item in source_items or [] if isinstance(item, dict)
        }
        edge_source_present = "EDGE_DIAGNOSIS_CANDIDATE" in source_types
        recognition_source_present = bool(
            source_types & {"OCR_BLOCK", "ASR_SEGMENT", "VISUAL_FINDING", "VIDEO_EVENT"}
        )
        if edge_binding_present or edge_source_present:
            if recognition_binding_present or recognition_source_present:
                raise WorkOrderConflict(
                    "field_ai_observation_source_invalid",
                    work.version,
                )
            return WorkOrderService._normalize_edge_ai_observation_entry(
                session,
                work,
                incident,
                identity,
                payload,
            )
        if (
            not isinstance(observation, str)
            or not observation.strip()
            or not isinstance(upload_id, str)
            or not upload_id
            or not isinstance(bundle_id, str)
            or not bundle_id
            or not isinstance(bundle_version, int)
            or isinstance(bundle_version, bool)
            or not isinstance(source_items, list)
            or not 1 <= len(source_items) <= 32
        ):
            raise WorkOrderConflict("field_ai_observation_source_invalid", work.version)

        requested_sources: list[tuple[str, str]] = []
        allowed_source_types = {
            "OCR_BLOCK",
            "ASR_SEGMENT",
            "VISUAL_FINDING",
            "VIDEO_EVENT",
        }
        for item in source_items:
            if not isinstance(item, dict):
                raise WorkOrderConflict("field_ai_observation_source_invalid", work.version)
            source_type = item.get("source_type")
            source_id = item.get("source_id")
            if (
                source_type not in allowed_source_types
                or not isinstance(source_id, str)
                or not source_id
            ):
                raise WorkOrderConflict("field_ai_observation_source_invalid", work.version)
            requested_sources.append((str(source_type), source_id))
        if len(set(requested_sources)) != len(requested_sources):
            raise WorkOrderConflict("field_ai_observation_source_duplicate", work.version)

        upload = session.scalar(
            select(WorkOrderFieldEvidenceUploadRecord).where(
                WorkOrderFieldEvidenceUploadRecord.tenant_id == work.tenant_id,
                WorkOrderFieldEvidenceUploadRecord.evidence_upload_id == upload_id,
                WorkOrderFieldEvidenceUploadRecord.work_order_id == work.work_order_id,
                WorkOrderFieldEvidenceUploadRecord.incident_id == incident.incident_id,
                WorkOrderFieldEvidenceUploadRecord.source_draft_id == incident.source_draft_id,
            )
        )
        latest_run = session.scalar(
            select(RecognitionRunRecord)
            .where(
                RecognitionRunRecord.tenant_id == work.tenant_id,
                RecognitionRunRecord.context_kind == "FIELD_EVIDENCE",
                RecognitionRunRecord.work_order_id == work.work_order_id,
                RecognitionRunRecord.field_evidence_upload_id == upload_id,
            )
            .order_by(
                RecognitionRunRecord.created_at.desc(),
                RecognitionRunRecord.recognition_run_id.desc(),
            )
            .limit(1)
        )
        if (
            upload is None
            or latest_run is None
            or latest_run.requested_by_subject_id != identity.subject_id
            or latest_run.work_order_version != work.version
            or latest_run.status != "SUCCEEDED"
            or latest_run.evidence_bundle_id != bundle_id
        ):
            raise WorkOrderConflict("field_ai_observation_state_changed", work.version)

        bundle = session.scalar(
            select(EvidenceBundleRecord).where(
                EvidenceBundleRecord.tenant_id == work.tenant_id,
                EvidenceBundleRecord.bundle_id == bundle_id,
                EvidenceBundleRecord.context_kind == "FIELD_EVIDENCE",
                EvidenceBundleRecord.work_order_id == work.work_order_id,
                EvidenceBundleRecord.field_evidence_upload_id == upload_id,
            )
        )
        if (
            bundle is None
            or bundle.status != "CONFIRMED"
            or bundle.version != bundle_version
            or bundle.draft_id != incident.source_draft_id
            or bundle.asset_id != incident.asset_id
            or bundle.media_id != upload.media_id
            or bundle.source_sha256 != upload.content_hash
        ):
            raise WorkOrderConflict("field_ai_observation_state_changed", work.version)

        result_records = list(
            session.scalars(
                select(RecognitionResultRecord)
                .where(
                    RecognitionResultRecord.tenant_id == work.tenant_id,
                    RecognitionResultRecord.bundle_id == bundle.bundle_id,
                )
                .order_by(
                    RecognitionResultRecord.result_type,
                    RecognitionResultRecord.ordinal,
                )
            )
        )
        correction_records = list(
            session.scalars(
                select(RecognitionCorrectionRecord)
                .where(
                    RecognitionCorrectionRecord.tenant_id == work.tenant_id,
                    RecognitionCorrectionRecord.bundle_id == bundle.bundle_id,
                    RecognitionCorrectionRecord.evidence_version == bundle.version,
                )
                .order_by(
                    RecognitionCorrectionRecord.target_type,
                    RecognitionCorrectionRecord.target_id,
                    RecognitionCorrectionRecord.created_at.desc(),
                )
            )
        )
        results_by_source: dict[tuple[str, str], dict[str, Any]] = {}
        source_id_fields = {
            "OCR_BLOCK": "block_id",
            "ASR_SEGMENT": "segment_id",
            "VISUAL_FINDING": "finding_id",
            "VIDEO_EVENT": "event_id",
        }
        for result in result_records:
            source_id_field = source_id_fields.get(result.result_type)
            if source_id_field is None:
                continue
            source_id = result.payload.get(source_id_field)
            if isinstance(source_id, str):
                results_by_source[(result.result_type, source_id)] = result.payload

        review_target_types = {
            "OCR_BLOCK": "OCR_BLOCK",
            "ASR_SEGMENT": "ASR_SEGMENT",
            "VISUAL_FINDING": "FINDING",
            "VIDEO_EVENT": "VIDEO_EVENT",
        }
        corrections_by_source = {
            (correction.target_type, correction.target_id): correction
            for correction in correction_records
            if correction.reviewer_subject_id == identity.subject_id
        }
        normalized_sources: list[dict[str, str]] = []
        for source_type, source_id in sorted(requested_sources):
            result = results_by_source.get((source_type, source_id))
            correction = corrections_by_source.get((review_target_types[source_type], source_id))
            if result is None or correction is None:
                raise WorkOrderConflict("field_ai_observation_source_not_accepted", work.version)
            disposition = correction.disposition
            if source_type in {"VISUAL_FINDING", "VIDEO_EVENT"}:
                accepted = disposition == "ACCEPTED"
            else:
                accepted = disposition in {"ACCEPTED", "CORRECTED"}
            if not accepted:
                raise WorkOrderConflict("field_ai_observation_source_not_accepted", work.version)
            if (
                source_type == "VIDEO_EVENT"
                and result.get("event_type") not in REVIEWABLE_VIDEO_EVENT_TYPES
            ):
                raise WorkOrderConflict("field_ai_observation_source_not_accepted", work.version)
            value_field = "text" if source_type in {"OCR_BLOCK", "ASR_SEGMENT"} else "description"
            stored_value = result.get(value_field)
            if disposition == "CORRECTED":
                stored_value = correction.corrected_value
            if not isinstance(stored_value, str) or not stored_value.strip():
                raise WorkOrderConflict("field_ai_observation_source_not_accepted", work.version)
            value = stored_value.strip()
            normalized_sources.append(
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "decision": str(disposition),
                    "value": value,
                    "value_sha256": f"sha256:{sha256(value.encode()).hexdigest()}",
                }
            )

        source_sha256 = bundle.source_sha256
        if not source_sha256.startswith("sha256:"):
            source_sha256 = f"sha256:{source_sha256}"
        return {
            "observation": observation.strip(),
            "field_evidence_upload_id": upload.evidence_upload_id,
            "recognition_bundle_id": bundle.bundle_id,
            "recognition_bundle_version": bundle.version,
            "source_sha256": source_sha256,
            "processor_versions": dict(sorted(bundle.processor_versions.items())),
            "sources": normalized_sources,
        }

    @staticmethod
    def _normalize_edge_ai_observation_entry(
        session: Session,
        work: WorkOrderRecord,
        incident: IncidentRecord,
        identity: IdentityContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        observation = payload.get("observation")
        candidate_id = payload.get("edge_diagnosis_candidate_id")
        candidate_version = payload.get("edge_diagnosis_candidate_version")
        source_items = payload.get("source_items")
        if (
            not isinstance(observation, str)
            or not observation.strip()
            or not isinstance(candidate_id, str)
            or not candidate_id
            or not isinstance(candidate_version, int)
            or isinstance(candidate_version, bool)
            or not isinstance(source_items, list)
            or len(source_items) != 1
            or source_items[0]
            != {
                "source_type": "EDGE_DIAGNOSIS_CANDIDATE",
                "source_id": candidate_id,
            }
        ):
            raise WorkOrderConflict("field_ai_observation_source_invalid", work.version)

        candidate = session.scalar(
            select(WorkOrderEdgeDiagnosisCandidateRecord)
            .where(
                WorkOrderEdgeDiagnosisCandidateRecord.tenant_id == identity.tenant_id,
                WorkOrderEdgeDiagnosisCandidateRecord.subject_id == identity.subject_id,
                WorkOrderEdgeDiagnosisCandidateRecord.work_order_id == work.work_order_id,
                WorkOrderEdgeDiagnosisCandidateRecord.candidate_id == candidate_id,
            )
            .with_for_update()
        )
        if candidate is None:
            raise WorkOrderNotVisible
        if (
            candidate.version != candidate_version
            or candidate.status != "ACCEPTED"
            or candidate.reviewed_by_subject_id != identity.subject_id
            or candidate.reviewed_at is None
            or not candidate.review_reason
        ):
            raise WorkOrderConflict("field_ai_observation_state_changed", work.version)

        pack = session.scalar(
            select(WorkOrderEdgeDiagnosisPackRecord)
            .where(
                WorkOrderEdgeDiagnosisPackRecord.tenant_id == identity.tenant_id,
                WorkOrderEdgeDiagnosisPackRecord.subject_id == identity.subject_id,
                WorkOrderEdgeDiagnosisPackRecord.work_order_id == work.work_order_id,
                WorkOrderEdgeDiagnosisPackRecord.pack_id == candidate.pack_id,
            )
            .with_for_update()
        )
        if pack is None:
            raise WorkOrderNotVisible
        asset = session.scalar(
            select(AssetRecord)
            .where(
                AssetRecord.tenant_id == identity.tenant_id,
                AssetRecord.asset_id == incident.asset_id,
            )
            .with_for_update()
        )
        assignment = session.scalar(
            select(WorkOrderAssignmentRecord)
            .where(
                WorkOrderAssignmentRecord.tenant_id == identity.tenant_id,
                WorkOrderAssignmentRecord.work_order_id == work.work_order_id,
                WorkOrderAssignmentRecord.assignee_subject_id == identity.subject_id,
            )
            .order_by(
                WorkOrderAssignmentRecord.assigned_at.desc(),
                WorkOrderAssignmentRecord.assignment_id.desc(),
            )
            .limit(1)
            .with_for_update()
        )
        runtime_evidence = candidate.runtime_evidence_json
        required_pack_values = (
            pack.release_id,
            pack.manifest_hash,
            pack.model_file,
            pack.model_content_hash,
            pack.prompt_bundle_id,
            pack.prompt_bundle_hash,
            pack.index_release_id,
            pack.index_content_checksum,
            candidate.result_content_hash,
            candidate.security_policy_version,
        )
        if (
            asset is None
            or assignment is None
            or pack.work_order_version != work.version
            or pack.asset_id != incident.asset_id
            or pack.asset_version != asset.version
            or pack.assignment_id != assignment.assignment_id
            or _utc(pack.assignment_assigned_at) != _utc(assignment.assigned_at)
            or not all(isinstance(value, str) and value for value in required_pack_values)
            or not isinstance(candidate.candidate_json, dict)
            or not isinstance(runtime_evidence, dict)
            or runtime_evidence.get("runtime_attestation") != "UNATTESTED"
            or runtime_evidence.get("model_file") != pack.model_file
            or runtime_evidence.get("model_content_hash") != pack.model_content_hash
            or runtime_evidence.get("prompt_bundle_hash") != pack.prompt_bundle_hash
        ):
            raise WorkOrderConflict("field_ai_observation_state_changed", work.version)

        guard = PromptInjectionGuard().inspect_output(candidate.candidate_json)
        if guard.decision != "ALLOWED":
            raise WorkOrderConflict(
                "field_ai_observation_source_rejected",
                work.version,
            )
        try:
            candidate_value = FieldEdgeDiagnosisCandidate.model_validate(candidate.candidate_json)
        except (TypeError, ValueError) as exc:
            raise WorkOrderConflict(
                "field_ai_observation_state_changed",
                work.version,
            ) from exc

        candidate_content_hash = canonical_digest(candidate.candidate_json)
        return {
            "observation": observation.strip(),
            "source_kind": "EDGE_DIAGNOSIS_CANDIDATE",
            "edge_diagnosis_candidate_id": candidate.candidate_id,
            "edge_diagnosis_candidate_version": candidate.version,
            "edge_diagnosis_pack_id": pack.pack_id,
            "result_content_hash": candidate.result_content_hash,
            "candidate_content_hash": candidate_content_hash,
            "release_id": pack.release_id,
            "manifest_hash": pack.manifest_hash,
            "model_file": pack.model_file,
            "model_content_hash": pack.model_content_hash,
            "prompt_bundle_id": pack.prompt_bundle_id,
            "prompt_bundle_hash": pack.prompt_bundle_hash,
            "index_release_id": pack.index_release_id,
            "index_content_checksum": pack.index_content_checksum,
            "citation_ids": sorted(set(candidate_value.citation_ids)),
            "reviewed_by_subject_id": candidate.reviewed_by_subject_id,
            "reviewed_at": _utc(candidate.reviewed_at).isoformat(),
            "import_security_policy_version": candidate.security_policy_version,
            "current_security_policy_version": guard.policy_version,
            "runtime_attestation": "UNATTESTED",
            "sources": [
                {
                    "source_type": "EDGE_DIAGNOSIS_CANDIDATE",
                    "source_id": candidate.candidate_id,
                    "decision": "ACCEPTED",
                    "content_sha256": candidate_content_hash,
                }
            ],
        }

    @classmethod
    def _validate_part_entry(
        cls,
        session: Session,
        work: WorkOrderRecord,
        payload: dict[str, Any],
    ) -> None:
        reservation = cls._load_available_part_allocation(session, work)
        if work.part_issue_required:
            issue = session.scalar(
                select(PartIssueRecord).where(
                    PartIssueRecord.tenant_id == work.tenant_id,
                    PartIssueRecord.work_order_id == work.work_order_id,
                    PartIssueRecord.reservation_id == reservation.reservation_id,
                )
            )
            if issue is None or issue.status != "ISSUED" or reservation.status != "ISSUED":
                raise WorkOrderConflict("part_issue_required", work.version)
        if (
            payload.get("part_reservation_id") != reservation.reservation_id
            or payload.get("part_number") != reservation.part_number
        ):
            raise WorkOrderConflict("part_allocation_mismatch", work.version)
        quantity = payload.get("quantity")
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            raise WorkOrderConflict("part_allocation_mismatch", work.version)
        if quantity > reservation.quantity:
            raise WorkOrderConflict("part_quantity_exceeds_allocation", work.version)
        if work.part_accounting_required:
            issue = session.scalar(
                select(PartIssueRecord).where(
                    PartIssueRecord.tenant_id == work.tenant_id,
                    PartIssueRecord.work_order_id == work.work_order_id,
                    PartIssueRecord.reservation_id == reservation.reservation_id,
                )
            )
            if issue is None or issue.status != "ISSUED" or reservation.status != "ISSUED":
                raise WorkOrderConflict("part_issue_not_issued", work.version)
            existing_usage = sum(
                cls._part_entry_quantity(entry, work.version)
                for entry in session.scalars(
                    select(WorkOrderFieldEntryRecord).where(
                        WorkOrderFieldEntryRecord.tenant_id == work.tenant_id,
                        WorkOrderFieldEntryRecord.work_order_id == work.work_order_id,
                        WorkOrderFieldEntryRecord.entry_type == "PART",
                    )
                )
            )
            occupied_return = int(
                session.scalar(
                    select(func.coalesce(func.sum(PartMaterialMovementRecord.quantity), 0)).where(
                        PartMaterialMovementRecord.tenant_id == work.tenant_id,
                        PartMaterialMovementRecord.work_order_id == work.work_order_id,
                        PartMaterialMovementRecord.part_issue_id == issue.part_issue_id,
                        PartMaterialMovementRecord.movement_kind == "RETURN",
                        PartMaterialMovementRecord.status.in_(
                            {"PENDING_APPROVAL", "APPROVED", "RECONCILING", "APPLIED"}
                        ),
                    )
                )
                or 0
            )
            if existing_usage + quantity + occupied_return > issue.quantity:
                raise WorkOrderConflict("part_accounting_capacity_exceeded", work.version)

    @staticmethod
    def _validate_evidence_entry(
        session: Session,
        work: WorkOrderRecord,
        incident: IncidentRecord,
        identity: IdentityContext,
        payload: dict[str, Any],
    ) -> None:
        evidence_id = payload.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise WorkOrderConflict("field_evidence_not_ready_or_visible", work.version)

        field_upload = (
            session.execute(
                select(WorkOrderFieldEvidenceUploadRecord, MediaObjectRecord)
                .join(
                    MediaObjectRecord,
                    MediaObjectRecord.media_id == WorkOrderFieldEvidenceUploadRecord.media_id,
                )
                .where(
                    WorkOrderFieldEvidenceUploadRecord.tenant_id == work.tenant_id,
                    MediaObjectRecord.tenant_id == work.tenant_id,
                    WorkOrderFieldEvidenceUploadRecord.evidence_upload_id == evidence_id,
                    WorkOrderFieldEvidenceUploadRecord.work_order_id == work.work_order_id,
                    WorkOrderFieldEvidenceUploadRecord.incident_id == incident.incident_id,
                    WorkOrderFieldEvidenceUploadRecord.source_draft_id == incident.source_draft_id,
                    WorkOrderFieldEvidenceUploadRecord.uploaded_by_subject_id
                    == identity.subject_id,
                )
            )
            .tuples()
            .one_or_none()
        )
        if field_upload is not None:
            binding, media = field_upload
            if (
                media.draft_id == binding.source_draft_id
                and media.scan_state == "CLEAN"
                and media.content_hash == binding.content_hash
                and media.declared_mime == binding.declared_mime
                and media.size_bytes == binding.size_bytes
            ):
                return
            raise WorkOrderConflict("field_evidence_not_ready_or_visible", work.version)

        confirmed_bundle = session.scalar(
            select(EvidenceBundleRecord).where(
                EvidenceBundleRecord.tenant_id == work.tenant_id,
                EvidenceBundleRecord.bundle_id == evidence_id,
                EvidenceBundleRecord.draft_id == incident.source_draft_id,
                EvidenceBundleRecord.asset_id == incident.asset_id,
                EvidenceBundleRecord.context_kind == "INCIDENT_DRAFT",
                EvidenceBundleRecord.status == "CONFIRMED",
            )
        )
        if confirmed_bundle is None:
            raise WorkOrderConflict("field_evidence_not_ready_or_visible", work.version)

    @staticmethod
    def _load_available_part_allocation(
        session: Session,
        work: WorkOrderRecord,
    ) -> PartReservationRecord:
        reservation = session.scalar(
            select(PartReservationRecord).where(
                PartReservationRecord.tenant_id == work.tenant_id,
                PartReservationRecord.reservation_id == work.reservation_id,
            )
        )
        if (
            reservation is None
            or reservation.incident_id != work.incident_id
            or reservation.status not in {"RESERVED", "ISSUED"}
        ):
            raise WorkOrderConflict("part_allocation_unavailable", work.version)
        return reservation

    def complete_from_field_entries(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        root_cause: str,
        cost_amount: str,
        request_id: str,
    ) -> WorkOrderView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session,
                identity.tenant_id,
                work_order_id,
                for_update=True,
            )
            self._require(
                identity,
                Action.EXECUTE_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            if work.assigned_subject_id != identity.subject_id:
                raise WorkOrderConflict("assignee_mismatch", work.version)
            active_rework = session.scalar(
                select(WorkOrderReworkRecord).where(
                    WorkOrderReworkRecord.tenant_id == identity.tenant_id,
                    WorkOrderReworkRecord.work_order_id == work_order_id,
                    WorkOrderReworkRecord.status == "OPEN",
                )
            )
            entry_checkpoint = (
                active_rework.entry_sequence_checkpoint if active_rework is not None else 0
            )
            entries = list(
                session.scalars(
                    select(WorkOrderFieldEntryRecord)
                    .where(
                        WorkOrderFieldEntryRecord.tenant_id == identity.tenant_id,
                        WorkOrderFieldEntryRecord.work_order_id == work_order_id,
                        WorkOrderFieldEntryRecord.sequence > entry_checkpoint,
                    )
                    .order_by(WorkOrderFieldEntryRecord.sequence)
                )
            )
            actions = [
                str(entry.payload["description"])
                for entry in entries
                if entry.entry_type == "STEP"
                and entry.payload.get("outcome") == "COMPLETED"
                and entry.payload.get("description")
            ]
            evidence_ids = [
                str(entry.payload["evidence_id"])
                for entry in entries
                if entry.entry_type == "EVIDENCE" and entry.payload.get("evidence_id")
            ]
            part_reservation_ids = [
                str(entry.payload["part_reservation_id"])
                for entry in entries
                if entry.entry_type == "PART" and entry.payload.get("part_reservation_id")
            ]
            if work.creation_mode == "SERVICE_AUTHORIZATION" and part_reservation_ids:
                raise WorkOrderConflict("parts_not_required", work.version)
            signature = next(
                (
                    entry
                    for entry in reversed(entries)
                    if entry.entry_type == "SIGNATURE"
                    and entry.payload.get("signature_role") == "CUSTOMER"
                    and entry.payload.get("signed_by")
                ),
                None,
            )
            if not actions or not evidence_ids or signature is None:
                raise WorkOrderConflict("field_completion_facts_incomplete", work.version)
            signed_by = str(signature.payload["signed_by"])
            confirmation_text = str(signature.payload.get("confirmation_text") or "confirmed")
            return self._complete_in_session(
                session,
                identity,
                work,
                expected_version=expected_version,
                active_rework=active_rework,
                completion=WorkOrderCompletionRecord(
                    field_entry_sequence_start=entries[0].sequence,
                    field_entry_sequence_end=entries[-1].sequence,
                    root_cause=root_cause,
                    actions=actions,
                    evidence_ids=evidence_ids,
                    part_reservation_ids=part_reservation_ids,
                    cost_amount=cost_amount,
                    customer_confirmation=f"{signed_by}: {confirmation_text}",
                ),
                event_payload={
                    "field_entry_count": len(entries),
                    "signature_entry_id": signature.entry_id,
                },
            )

    def accept(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> WorkOrderView:
        return self._engineer_transition(
            identity, work_order_id, expected_version, "ASSIGNED", "ACCEPTED", request_id
        )

    def start(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> WorkOrderView:
        return self._engineer_transition(
            identity, work_order_id, expected_version, "ACCEPTED", "IN_PROGRESS", request_id
        )

    def complete(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        root_cause: str,
        actions: list[str],
        evidence_ids: list[str],
        part_reservation_ids: list[str] | None = None,
        cost_amount: str | None = None,
        customer_confirmation: str | None = None,
        request_id: str,
    ) -> WorkOrderView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session, identity.tenant_id, work_order_id, for_update=True
            )
            if work.status != "IN_PROGRESS":
                raise WorkOrderConflict("illegal_state_transition", work.version)
            self._require(
                identity,
                Action.EXECUTE_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            if (
                work.assigned_subject_id != identity.subject_id
                or not root_cause
                or not actions
                or not evidence_ids
            ):
                raise WorkOrderConflict("completion_precondition_failed", work.version)
            requested_part_reservations = part_reservation_ids or []
            if work.creation_mode == "SERVICE_AUTHORIZATION":
                if requested_part_reservations:
                    raise WorkOrderConflict("parts_not_required", work.version)
                if not cost_amount:
                    raise WorkOrderConflict("completion_cost_required", work.version)
            active_rework = session.scalar(
                select(WorkOrderReworkRecord).where(
                    WorkOrderReworkRecord.tenant_id == identity.tenant_id,
                    WorkOrderReworkRecord.work_order_id == work_order_id,
                    WorkOrderReworkRecord.status == "OPEN",
                )
            )
            return self._complete_in_session(
                session,
                identity,
                work,
                expected_version=expected_version,
                active_rework=active_rework,
                completion=WorkOrderCompletionRecord(
                    field_entry_sequence_start=None,
                    field_entry_sequence_end=None,
                    root_cause=root_cause,
                    actions=actions,
                    evidence_ids=evidence_ids,
                    part_reservation_ids=requested_part_reservations,
                    cost_amount=cost_amount,
                    customer_confirmation=customer_confirmation,
                ),
            )

    def _complete_in_session(
        self,
        session: Session,
        identity: IdentityContext,
        work: WorkOrderRecord,
        *,
        expected_version: int,
        active_rework: WorkOrderReworkRecord | None,
        completion: WorkOrderCompletionRecord,
        event_payload: dict[str, Any] | None = None,
    ) -> WorkOrderView:
        """Persist prepared facts; callers own locking, transaction and preconditions."""
        completion.round_number = (
            active_rework.round_number
            if active_rework is not None
            else int(
                session.scalar(
                    select(func.max(WorkOrderCompletionRecord.round_number)).where(
                        WorkOrderCompletionRecord.tenant_id == identity.tenant_id,
                        WorkOrderCompletionRecord.work_order_id == work.work_order_id,
                    )
                )
                or 0
            )
            + 1
        )
        self._transition(
            work,
            expected_version,
            "IN_PROGRESS",
            "COMPLETED",
            identity.subject_id,
            session,
            {
                "evidence_ids": completion.evidence_ids,
                **(event_payload or {}),
                "round_number": completion.round_number,
            },
        )
        now = datetime.now(UTC)
        completion.completion_id = f"completion-{uuid4().hex}"
        completion.tenant_id = identity.tenant_id
        completion.work_order_id = work.work_order_id
        completion.completed_by_subject_id = identity.subject_id
        completion.completed_at = completion.created_at = completion.updated_at = now
        session.add(completion)
        if active_rework is not None:
            active_rework.status = "RESUBMITTED"
            active_rework.resolved_completion_id = completion.completion_id
            active_rework.resolved_at = now
            active_rework.updated_at = now
        return _view(work)

    def verify(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        passed: bool,
        reason: str,
        request_id: str,
    ) -> WorkOrderView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session, identity.tenant_id, work_order_id, for_update=True
            )
            self._require(
                identity,
                Action.VERIFY_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            if work.assigned_subject_id == identity.subject_id:
                raise WorkOrderConflict("verifier_must_be_independent", work.version)
            completion = session.scalar(
                select(WorkOrderCompletionRecord)
                .where(
                    WorkOrderCompletionRecord.tenant_id == identity.tenant_id,
                    WorkOrderCompletionRecord.work_order_id == work_order_id,
                )
                .order_by(
                    WorkOrderCompletionRecord.round_number.desc(),
                    WorkOrderCompletionRecord.completed_at.desc(),
                )
            )
            if completion is None:
                raise WorkOrderConflict("verification_completion_missing", work.version)
            next_status = "VERIFIED" if passed else "IN_PROGRESS"
            self._transition(
                work,
                expected_version,
                "COMPLETED",
                next_status,
                identity.subject_id,
                session,
                {
                    "passed": passed,
                    "reason": reason,
                    "completion_id": completion.completion_id,
                    "round_number": completion.round_number,
                },
            )
            now = datetime.now(UTC)
            verification_id = f"verification-{uuid4().hex}"
            session.add(
                WorkOrderVerificationRecord(
                    verification_id=verification_id,
                    tenant_id=identity.tenant_id,
                    work_order_id=work_order_id,
                    completion_id=completion.completion_id,
                    round_number=completion.round_number,
                    verifier_subject_id=identity.subject_id,
                    passed=passed,
                    reason=reason,
                    verified_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            if not passed:
                entry_checkpoint = int(
                    session.scalar(
                        select(func.max(WorkOrderFieldEntryRecord.sequence)).where(
                            WorkOrderFieldEntryRecord.tenant_id == identity.tenant_id,
                            WorkOrderFieldEntryRecord.work_order_id == work_order_id,
                        )
                    )
                    or 0
                )
                session.add(
                    WorkOrderReworkRecord(
                        rework_id=f"rework-{uuid4().hex}",
                        tenant_id=identity.tenant_id,
                        work_order_id=work_order_id,
                        source_completion_id=completion.completion_id,
                        failed_verification_id=verification_id,
                        resolved_completion_id=None,
                        round_number=completion.round_number + 1,
                        entry_sequence_checkpoint=entry_checkpoint,
                        reason=reason,
                        status="OPEN",
                        opened_at=now,
                        resolved_at=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
            return _view(work)

    def close(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> WorkOrderView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            return self.close_in_session(
                session,
                identity,
                work_order_id,
                expected_version=expected_version,
                request_id=request_id,
            )

    def close_in_session(
        self,
        session: Session,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_version: int,
        request_id: str,
        reason: str | None = None,
        operation_id: str | None = None,
        proposal_id: str | None = None,
    ) -> WorkOrderView:
        """Close a verified work order in an existing transaction.

        Both the direct command and approved Agent execution use this boundary so
        authorization, closure evidence, state changes, and Outbox publication
        cannot drift into separate implementations.
        """

        work, incident, site_id = self._load(
            session,
            identity.tenant_id,
            work_order_id,
            for_update=True,
        )
        self._require(
            identity,
            Action.CLOSE_WORK_ORDER,
            work_order_id,
            incident.asset_id,
            site_id,
            request_id,
        )
        facts = self.closure_facts_in_session(
            session,
            tenant_id=identity.tenant_id,
            work_order_id=work_order_id,
            current_version=work.version,
        )
        transition_payload: dict[str, Any] = {}
        if reason is not None:
            transition_payload["reason"] = reason
        if operation_id is not None:
            transition_payload["operation_id"] = operation_id
        if proposal_id is not None:
            transition_payload["proposal_id"] = proposal_id
        self._transition(
            work,
            expected_version,
            "VERIFIED",
            "CLOSED",
            identity.subject_id,
            session,
            transition_payload,
        )
        completion = session.scalar(
            select(WorkOrderCompletionRecord).where(
                WorkOrderCompletionRecord.tenant_id == identity.tenant_id,
                WorkOrderCompletionRecord.completion_id == facts.completion_id,
            )
        )
        verification = session.scalar(
            select(WorkOrderVerificationRecord).where(
                WorkOrderVerificationRecord.tenant_id == identity.tenant_id,
                WorkOrderVerificationRecord.verification_id == facts.verification_id,
            )
        )
        if completion is None or verification is None:
            raise WorkOrderConflict("closure_facts_incomplete", work.version)
        try:
            record_work_order_resolution_in_session(
                session,
                identity,
                incident,
                work,
                completion,
                verification,
                occurred_at=datetime.now(UTC),
            )
        except IncidentOperationConflict as exc:
            raise WorkOrderConflict(exc.reason, work.version) from exc
        trace_context = capture_current_trace_context()
        session.add(
            build_work_order_closed_event(
                tenant_id=identity.tenant_id,
                work_order_id=work.work_order_id,
                incident_id=incident.incident_id,
                completion_id=facts.completion_id,
                verification_id=facts.verification_id,
                aggregate_version=work.version,
                occurred_at=work.updated_at,
                request_id=request_id,
                trace_id=(trace_context.trace_id if trace_context is not None else None),
            ).to_outbox_record(trace_context=trace_context)
        )
        return _view(work)

    @staticmethod
    def closure_facts_in_session(
        session: Session,
        *,
        tenant_id: str,
        work_order_id: str,
        current_version: int,
    ) -> WorkOrderClosureFacts:
        """Validate the authoritative evidence required to close a work order."""

        completion = session.scalar(
            select(WorkOrderCompletionRecord)
            .where(
                WorkOrderCompletionRecord.tenant_id == tenant_id,
                WorkOrderCompletionRecord.work_order_id == work_order_id,
            )
            .order_by(WorkOrderCompletionRecord.completed_at.desc())
        )
        verification = session.scalar(
            select(WorkOrderVerificationRecord)
            .where(
                WorkOrderVerificationRecord.tenant_id == tenant_id,
                WorkOrderVerificationRecord.work_order_id == work_order_id,
                WorkOrderVerificationRecord.passed.is_(True),
            )
            .order_by(WorkOrderVerificationRecord.verified_at.desc())
        )
        customer_confirmation = session.scalar(
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
        customer_result_accepted = (
            customer_confirmation.result_accepted is True
            if customer_confirmation is not None
            else bool(completion and completion.customer_confirmation)
        )
        if (
            completion is None
            or verification is None
            or verification.completion_id != completion.completion_id
            or not completion.cost_amount
            or not customer_result_accepted
        ):
            raise WorkOrderConflict("closure_facts_incomplete", current_version)
        work = session.scalar(
            select(WorkOrderRecord).where(
                WorkOrderRecord.tenant_id == tenant_id,
                WorkOrderRecord.work_order_id == work_order_id,
            )
        )
        if work is None:
            raise WorkOrderConflict("work_order_not_found", current_version)
        if work.creation_mode == "SERVICE_AUTHORIZATION":
            if completion.part_reservation_ids:
                raise WorkOrderConflict("parts_not_required", current_version)
        else:
            if not completion.part_reservation_ids:
                raise WorkOrderConflict("closure_facts_incomplete", current_version)
            accounting = WorkOrderService.part_accounting_in_session(session, work)
            if accounting.mode == "AUTHORITATIVE" and not accounting.close_ready:
                raise WorkOrderConflict("part_accounting_incomplete", current_version)
        return WorkOrderClosureFacts(
            completion_id=completion.completion_id,
            verification_id=verification.verification_id,
        )

    def _engineer_transition(
        self,
        identity: IdentityContext,
        work_order_id: str,
        expected_version: int,
        current: str,
        target: str,
        request_id: str,
    ) -> WorkOrderView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            work, incident, site_id = self._load(
                session, identity.tenant_id, work_order_id, for_update=True
            )
            self._require(
                identity,
                Action.EXECUTE_WORK_ORDER,
                work_order_id,
                incident.asset_id,
                site_id,
                request_id,
            )
            if work.assigned_subject_id != identity.subject_id:
                raise WorkOrderConflict("assignee_mismatch", work.version)
            self._transition(
                work, expected_version, current, target, identity.subject_id, session, {}
            )
            return _view(work)

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        asset_id: str,
        site_id: str | None,
        request_id: str,
    ) -> None:
        if asset_id in identity.asset_ids:
            resource = ResourceContext(identity.tenant_id, resource_id, asset_id=asset_id)
        elif site_id is not None and site_id in identity.site_ids:
            resource = ResourceContext(identity.tenant_id, resource_id, site_id=site_id)
        else:
            resource = ResourceContext(identity.tenant_id, resource_id, asset_id=asset_id)
        self._authorizer.require(
            identity,
            action,
            resource,
            request_id=request_id,
        )

    @staticmethod
    def _require_assignee_or_coordinator(
        identity: IdentityContext,
        work: WorkOrderRecord,
    ) -> None:
        coordinator_roles = {
            Role.AFTER_SALES_ENGINEER,
            Role.DOMAIN_EXPERT,
            Role.TENANT_ADMIN,
        }
        if identity.roles.isdisjoint(coordinator_roles) and (
            work.assigned_subject_id != identity.subject_id
        ):
            raise WorkOrderConflict("assignee_mismatch", work.version)

    @staticmethod
    def _record_control(
        session: Session,
        work: WorkOrderRecord,
        *,
        command_type: str,
        previous_status: str,
        reason: str,
        recovery_condition: str | None,
        responsible_subject_id: str | None,
    ) -> None:
        occurred_at = _utc(work.updated_at)
        session.add(
            WorkOrderControlRecord(
                control_id=f"work-control-{uuid4().hex}",
                tenant_id=work.tenant_id,
                work_order_id=work.work_order_id,
                work_order_version=work.version,
                command_type=command_type,
                previous_status=previous_status,
                target_status=work.status,
                reason=reason,
                recovery_condition=recovery_condition,
                responsible_subject_id=responsible_subject_id,
                service_window_start=work.service_window_start,
                service_window_end=work.service_window_end,
                sla_due_at=work.sla_due_at,
                occurred_at=occurred_at,
                created_at=occurred_at,
                updated_at=occurred_at,
            )
        )

    @staticmethod
    def _load(
        session: Session,
        tenant_id: str,
        work_order_id: str,
        *,
        for_update: bool = False,
    ) -> tuple[WorkOrderRecord, IncidentRecord, str | None]:
        statement = (
            select(WorkOrderRecord, IncidentRecord, AssetSiteLinkRecord.site_id)
            .join(IncidentRecord, IncidentRecord.incident_id == WorkOrderRecord.incident_id)
            .outerjoin(
                AssetSiteLinkRecord,
                AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
            )
            .where(
                WorkOrderRecord.tenant_id == tenant_id,
                IncidentRecord.tenant_id == tenant_id,
                WorkOrderRecord.work_order_id == work_order_id,
            )
        )
        if for_update:
            statement = statement.with_for_update(of=(WorkOrderRecord, IncidentRecord))
        row = session.execute(statement).tuples().one_or_none()
        if row is None:
            raise WorkOrderNotVisible
        return row

    @staticmethod
    def _transition(
        work: WorkOrderRecord,
        expected_version: int,
        current: str,
        target: str,
        actor: str,
        session: Session,
        payload: dict[str, Any],
    ) -> None:
        if work.version != expected_version:
            raise WorkOrderConflict("version_conflict", work.version)
        if work.status != current:
            raise WorkOrderConflict("illegal_state_transition", work.version)
        now, sequence = datetime.now(UTC), work.version + 1
        work.status, work.version, work.updated_at = target, sequence, now
        session.add(
            WorkOrderEventRecord(
                event_id=f"work-event-{uuid4().hex}",
                tenant_id=work.tenant_id,
                work_order_id=work.work_order_id,
                sequence=sequence,
                event_type=f"work_order.{target.lower()}",
                actor_subject_id=actor,
                payload=payload,
                occurred_at=now,
                created_at=now,
                updated_at=now,
            )
        )


def _view(record: WorkOrderRecord) -> WorkOrderView:
    return WorkOrderView(
        work_order_id=record.work_order_id,
        incident_id=record.incident_id,
        proposal_id=record.proposal_id,
        reservation_id=record.reservation_id,
        creation_mode=record.creation_mode,  # type: ignore[arg-type]
        authorization_type=record.authorization_type,  # type: ignore[arg-type]
        diagnosis_run_id=record.diagnosis_run_id,
        diagnosis_version=record.diagnosis_version,
        service_quotation_id=record.service_quotation_id,
        initial_parts_required=record.creation_mode == "PARTS_RESERVATION",
        part_issue_required=record.part_issue_required,
        part_accounting_required=record.part_accounting_required,
        status=record.status,
        assigned_subject_id=record.assigned_subject_id,
        pending_assignment_operation_id=record.pending_assignment_operation_id,
        priority=record.priority,
        sla_due_at=_utc(record.sla_due_at) if record.sla_due_at is not None else None,
        sla_status=_sla_status(record),
        service_window_start=(
            _utc(record.service_window_start) if record.service_window_start is not None else None
        ),
        service_window_end=(
            _utc(record.service_window_end) if record.service_window_end is not None else None
        ),
        version=record.version,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
    )


def _control_view(record: WorkOrderControlRecord) -> WorkOrderControlView:
    return WorkOrderControlView(
        control_id=record.control_id,
        work_order_version=record.work_order_version,
        command_type=record.command_type,
        previous_status=record.previous_status,
        target_status=record.target_status,
        reason=record.reason,
        recovery_condition=record.recovery_condition,
        responsible_subject_id=record.responsible_subject_id,
        service_window_start=(
            _utc(record.service_window_start) if record.service_window_start is not None else None
        ),
        service_window_end=(
            _utc(record.service_window_end) if record.service_window_end is not None else None
        ),
        sla_due_at=_utc(record.sla_due_at) if record.sla_due_at is not None else None,
        occurred_at=_utc(record.occurred_at),
    )


def _field_entry_view(record: WorkOrderFieldEntryRecord) -> WorkOrderFieldEntryView:
    return WorkOrderFieldEntryView(
        entry_id=record.entry_id,
        sequence=record.sequence,
        client_operation_id=record.client_operation_id,
        entry_type=record.entry_type,
        payload=record.payload,
        actor_subject_id=record.actor_subject_id,
        occurred_at=_utc(record.occurred_at),
        recorded_at=_utc(record.created_at),
    )


def _optional_payload_text(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get(field)
    return value if isinstance(value, str) and value else None


def _closure_report_digest(values: dict[str, Any]) -> str:
    canonical = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_closure_report_json_default,
    ).encode("utf-8")
    return f"sha256:{sha256(canonical).hexdigest()}"


def _closure_report_json_default(value: object) -> str:
    if isinstance(value, datetime):
        return _utc(value).isoformat()
    raise TypeError(f"closure report value is not serializable: {type(value).__name__}")


def _validate_service_window(
    service_window_start: datetime,
    service_window_end: datetime,
    sla_due_at: datetime,
) -> None:
    start = _utc(service_window_start)
    end = _utc(service_window_end)
    due = _utc(sla_due_at)
    if start >= end:
        raise WorkOrderConflict("service_window_invalid")
    if end > due:
        raise WorkOrderConflict("service_window_exceeds_sla")


def _sla_status(record: WorkOrderRecord) -> str:
    if record.sla_due_at is None:
        return "UNPLANNED"
    due = _utc(record.sla_due_at)
    if record.status in {"COMPLETED", "VERIFIED", "CLOSED"}:
        return "MET" if _utc(record.updated_at) <= due else "BREACHED"
    remaining_seconds = (due - datetime.now(UTC)).total_seconds()
    if remaining_seconds < 0:
        return "BREACHED"
    if remaining_seconds <= 2 * 60 * 60:
        return "AT_RISK"
    return "ON_TRACK"
