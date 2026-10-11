"""Role-scoped incident intake queue and append-only manual controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.maintenance_planning.review_isolation import (
    incident_read_transaction,
    record_incident_disclosure,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetRecord,
    AssetSiteLinkRecord,
    CustomerServiceUpdateRecord,
    DiagnosisRunRecord,
    IncidentControlRecord,
    IncidentRecord,
    IncidentRelationRecord,
    IncidentResolutionRecord,
    WorkOrderCompletionRecord,
    WorkOrderRecord,
    WorkOrderVerificationRecord,
)


class IncidentNotVisible(Exception):
    pass


class IncidentOperationConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.current_version = current_version


@dataclass(frozen=True, slots=True)
class IncidentOperationsView:
    incident_id: str
    asset_id: str
    description: str
    status: str
    severity: str | None
    category: str | None
    responsible_queue: str | None
    information_due_at: datetime | None
    escalation_responsible_subject_id: str | None
    version: int
    created_at: datetime
    updated_at: datetime
    legal_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IncidentQueueItem:
    incident: IncidentOperationsView
    asset_display_name: str | None
    model_code: str | None
    site_id: str | None
    site_name: str | None


@dataclass(frozen=True, slots=True)
class IncidentControlView:
    control_id: str
    incident_version: int
    command_type: str
    previous_status: str
    target_status: str
    reason: str
    actor_subject_id: str
    responsible_subject_id: str | None
    recovery_condition: str | None
    related_incident_id: str | None
    details: dict[str, Any]
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class IncidentResolutionView:
    resolution_id: str
    incident_id: str
    mode: str
    diagnosis_run_id: str | None
    work_order_id: str | None
    completion_id: str | None
    verification_id: str | None
    summary: str
    evidence_ids: tuple[str, ...]
    resolved_by_subject_id: str
    resolved_at: datetime


class IncidentOperationsService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def list_queue(
        self,
        identity: IdentityContext,
        *,
        status: str | None,
        severity: str | None,
        responsible_queue: str | None,
        limit: int,
        offset: int,
        request_id: str,
    ) -> tuple[list[IncidentQueueItem], int]:
        self._authorizer.require(
            identity,
            Action.READ_INCIDENT_QUEUE,
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
            select(IncidentRecord, AssetRecord, AssetSiteLinkRecord)
            .join(AssetRecord, AssetRecord.asset_id == IncidentRecord.asset_id)
            .outerjoin(
                AssetSiteLinkRecord,
                AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
            )
            .where(
                IncidentRecord.tenant_id == identity.tenant_id,
                AssetRecord.tenant_id == identity.tenant_id,
                or_(*scope_conditions),
            )
        )
        if status is not None:
            statement = statement.where(IncidentRecord.status == status)
        if severity is not None:
            statement = statement.where(IncidentRecord.severity == severity)
        if responsible_queue is not None:
            statement = statement.where(IncidentRecord.responsible_queue == responsible_queue)

        with self._database.transaction(identity.tenant_context) as session:
            total = int(session.scalar(select(func.count()).select_from(statement.subquery())) or 0)
            rows = (
                session.execute(
                    statement.order_by(IncidentRecord.updated_at.desc(), IncidentRecord.incident_id)
                    .limit(limit)
                    .offset(offset)
                )
                .tuples()
                .all()
            )
            record_incident_disclosure(
                session,
                identity,
                (incident.incident_id for incident, *_rest in rows),
                resource_kind="incident-list",
                resource_id="incident-list",
                request_id=request_id,
            )
            return [
                IncidentQueueItem(
                    incident=_view(incident),
                    asset_display_name=asset.display_name,
                    model_code=asset.model_code,
                    site_id=site.site_id if site is not None else None,
                    site_name=site.site_name if site is not None else None,
                )
                for incident, asset, site in rows
            ], total

    def get(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        request_id: str,
    ) -> IncidentOperationsView:
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            incident, site_id = self._load(session, identity.tenant_id, incident_id)
            self._require(
                identity,
                Action.READ_INCIDENT_QUEUE,
                incident,
                site_id,
                request_id,
            )
            return _view(incident)

    def list_controls(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        request_id: str,
    ) -> list[IncidentControlView]:
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            incident, site_id = self._load(session, identity.tenant_id, incident_id)
            self._require(
                identity,
                Action.READ_INCIDENT_QUEUE,
                incident,
                site_id,
                request_id,
            )
            records = session.scalars(
                select(IncidentControlRecord)
                .where(
                    IncidentControlRecord.tenant_id == identity.tenant_id,
                    IncidentControlRecord.incident_id == incident_id,
                )
                .order_by(
                    IncidentControlRecord.incident_version,
                    IncidentControlRecord.occurred_at,
                )
            ).all()
            return [_control_view(record) for record in records]

    def get_resolution(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        request_id: str,
    ) -> IncidentResolutionView | None:
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            incident, site_id = self._load(session, identity.tenant_id, incident_id)
            self._require(
                identity,
                Action.READ_INCIDENT_QUEUE,
                incident,
                site_id,
                request_id,
            )
            record = session.scalar(
                select(IncidentResolutionRecord).where(
                    IncidentResolutionRecord.tenant_id == identity.tenant_id,
                    IncidentResolutionRecord.incident_id == incident_id,
                )
            )
            return _resolution_view(record) if record is not None else None

    def resolve_remote(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        summary: str,
        evidence_ids: list[str],
        expected_version: int,
        request_id: str,
    ) -> IncidentOperationsView:
        normalized_summary = summary.strip()
        normalized_evidence = tuple(
            dict.fromkeys(item.strip() for item in evidence_ids if item.strip())
        )
        if not normalized_summary:
            raise IncidentOperationConflict("resolution_summary_required")
        if not normalized_evidence:
            raise IncidentOperationConflict("resolution_evidence_required")

        now = datetime.now(UTC)
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            incident, site_id = self._load(session, identity.tenant_id, incident_id)
            self._require(
                identity,
                Action.CONTROL_INCIDENT,
                incident,
                site_id,
                request_id,
            )
            self._check_version(incident, expected_version)
            if incident.status != "DIAGNOSED":
                raise IncidentOperationConflict(
                    f"incident in {incident.status} cannot execute RESOLVE_REMOTE",
                    incident.version,
                )
            existing = session.scalar(
                select(IncidentResolutionRecord).where(
                    IncidentResolutionRecord.tenant_id == identity.tenant_id,
                    IncidentResolutionRecord.incident_id == incident_id,
                )
            )
            if existing is not None:
                raise IncidentOperationConflict("incident_resolution_already_exists")
            work_order = session.scalar(
                select(WorkOrderRecord.work_order_id).where(
                    WorkOrderRecord.tenant_id == identity.tenant_id,
                    WorkOrderRecord.incident_id == incident_id,
                )
            )
            if work_order is not None:
                raise IncidentOperationConflict("work_order_already_exists")
            diagnosis = session.scalar(
                select(DiagnosisRunRecord)
                .where(
                    DiagnosisRunRecord.tenant_id == identity.tenant_id,
                    DiagnosisRunRecord.incident_id == incident_id,
                    DiagnosisRunRecord.status == "COMPLETED",
                )
                .order_by(DiagnosisRunRecord.updated_at.desc())
            )
            if diagnosis is None:
                raise IncidentOperationConflict("completed_diagnosis_required")

            resolution_id = f"incident-resolution-{uuid4().hex}"
            self._apply_update(
                session,
                incident,
                expected_version=expected_version,
                target_status="RESOLVED",
                values={},
                now=now,
            )
            session.add(
                IncidentResolutionRecord(
                    resolution_id=resolution_id,
                    tenant_id=identity.tenant_id,
                    incident_id=incident_id,
                    mode="REMOTE",
                    diagnosis_run_id=diagnosis.diagnosis_run_id,
                    work_order_id=None,
                    completion_id=None,
                    verification_id=None,
                    summary=normalized_summary,
                    evidence_ids=list(normalized_evidence),
                    resolved_by_subject_id=identity.subject_id,
                    resolved_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            self._append_control(
                session,
                identity,
                incident,
                command_type="RESOLVE_REMOTE",
                target_status="RESOLVED",
                reason=normalized_summary,
                details={
                    "resolution_id": resolution_id,
                    "diagnosis_run_id": diagnosis.diagnosis_run_id,
                    "evidence_ids": list(normalized_evidence),
                    "mode": "REMOTE",
                },
                occurred_at=now,
            )
            return _view_after(incident, status="RESOLVED", values={}, occurred_at=now)

    def close(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        confirmation_type: str,
        reason: str,
        expected_version: int,
        request_id: str,
        customer_update_id: str | None = None,
    ) -> IncidentOperationsView:
        normalized_type = confirmation_type.strip().upper()
        normalized_reason = reason.strip()
        if normalized_type not in {"CUSTOMER", "EXPERT"}:
            raise IncidentOperationConflict("invalid_resolution_confirmation_type")
        if not normalized_reason:
            raise IncidentOperationConflict("incident_close_reason_required")
        if normalized_type == "EXPERT" and identity.roles.isdisjoint(
            {Role.DOMAIN_EXPERT, Role.TENANT_ADMIN}
        ):
            raise IncidentOperationConflict("expert_confirmation_required")
        if normalized_type == "CUSTOMER" and not customer_update_id:
            raise IncidentOperationConflict("customer_confirmation_required")

        now = datetime.now(UTC)
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            incident, site_id = self._load(session, identity.tenant_id, incident_id)
            self._require(
                identity,
                Action.CONTROL_INCIDENT,
                incident,
                site_id,
                request_id,
            )
            self._check_version(incident, expected_version)
            if incident.status != "RESOLVED":
                raise IncidentOperationConflict(
                    f"incident in {incident.status} cannot execute CLOSE",
                    incident.version,
                )
            resolution = session.scalar(
                select(IncidentResolutionRecord).where(
                    IncidentResolutionRecord.tenant_id == identity.tenant_id,
                    IncidentResolutionRecord.incident_id == incident_id,
                )
            )
            if resolution is None:
                raise IncidentOperationConflict("incident_resolution_required")

            confirmation_id: str | None = None
            if normalized_type == "CUSTOMER":
                confirmation = session.scalar(
                    select(CustomerServiceUpdateRecord).where(
                        CustomerServiceUpdateRecord.tenant_id == identity.tenant_id,
                        CustomerServiceUpdateRecord.update_id == customer_update_id,
                        CustomerServiceUpdateRecord.incident_id == incident_id,
                        CustomerServiceUpdateRecord.update_type == "RESULT_CONFIRMATION",
                        CustomerServiceUpdateRecord.result_accepted.is_(True),
                    )
                )
                if (
                    confirmation is None
                    or confirmation.actor_subject_id != incident.reporter_subject_id
                ):
                    raise IncidentOperationConflict("accepted_customer_confirmation_required")
                confirmation_id = confirmation.update_id

            self._apply_update(
                session,
                incident,
                expected_version=expected_version,
                target_status="CLOSED",
                values={},
                now=now,
            )
            self._append_control(
                session,
                identity,
                incident,
                command_type="CLOSE",
                target_status="CLOSED",
                reason=normalized_reason,
                details={
                    "resolution_id": resolution.resolution_id,
                    "confirmation_type": normalized_type,
                    "customer_update_id": confirmation_id,
                },
                occurred_at=now,
            )
            return _view_after(incident, status="CLOSED", values={}, occurred_at=now)

    def classify(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        severity: str,
        category: str,
        responsible_queue: str,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> IncidentOperationsView:
        return self._transition(
            identity,
            incident_id,
            expected_version=expected_version,
            allowed_statuses=("SUBMITTED",),
            target_status="TRIAGED",
            command_type="CLASSIFY",
            reason=reason,
            values={
                "severity": severity,
                "category": category,
                "responsible_queue": responsible_queue,
                "information_due_at": None,
                "escalation_responsible_subject_id": None,
            },
            details={
                "severity": severity,
                "category": category,
                "responsible_queue": responsible_queue,
            },
            request_id=request_id,
        )

    def request_information(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        reason: str,
        information_due_at: datetime,
        expected_version: int,
        request_id: str,
    ) -> IncidentOperationsView:
        return self._transition(
            identity,
            incident_id,
            expected_version=expected_version,
            allowed_statuses=("SUBMITTED", "DIAGNOSING"),
            target_status="NEEDS_INFORMATION",
            command_type="REQUEST_INFORMATION",
            reason=reason,
            values={"information_due_at": information_due_at},
            details={"information_due_at": information_due_at.isoformat()},
            recovery_condition="客户补充所需故障资料并由受理人员确认",
            request_id=request_id,
        )

    def mark_information_received(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> IncidentOperationsView:
        return self._transition(
            identity,
            incident_id,
            expected_version=expected_version,
            allowed_statuses=("NEEDS_INFORMATION",),
            target_status="SUBMITTED",
            command_type="INFORMATION_RECEIVED",
            reason=reason,
            values={"information_due_at": None},
            details={},
            request_id=request_id,
        )

    def escalate(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        reason: str,
        responsible_subject_id: str,
        recovery_condition: str,
        expected_version: int,
        request_id: str,
    ) -> IncidentOperationsView:
        return self._transition(
            identity,
            incident_id,
            expected_version=expected_version,
            allowed_statuses=(
                "TRIAGED",
                "DIAGNOSING",
                "DIAGNOSED",
                "WORK_ORDER_CREATED",
            ),
            target_status="ESCALATED",
            command_type="ESCALATE",
            reason=reason,
            values={
                "escalation_responsible_subject_id": responsible_subject_id,
            },
            details={},
            responsible_subject_id=responsible_subject_id,
            recovery_condition=recovery_condition,
            request_id=request_id,
        )

    def resume(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        return_status: str,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> IncidentOperationsView:
        if return_status not in {"TRIAGED", "DIAGNOSING"}:
            raise IncidentOperationConflict("invalid incident recovery status")
        return self._transition(
            identity,
            incident_id,
            expected_version=expected_version,
            allowed_statuses=("ESCALATED",),
            target_status=return_status,
            command_type="RESUME",
            reason=reason,
            values={"escalation_responsible_subject_id": None},
            details={"return_status": return_status},
            request_id=request_id,
        )

    def merge_duplicate(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        canonical_incident_id: str,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> IncidentOperationsView:
        if incident_id == canonical_incident_id:
            raise IncidentOperationConflict("an incident cannot be merged into itself")
        now = datetime.now(UTC)
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id, canonical_incident_id),
            request_id=request_id,
            lineage_write=True,
        ) as session:
            source, site_id = self._load(session, identity.tenant_id, incident_id)
            target, target_site_id = self._load(session, identity.tenant_id, canonical_incident_id)
            self._require(
                identity,
                Action.CONTROL_INCIDENT,
                source,
                site_id,
                request_id,
            )
            self._require(
                identity,
                Action.CONTROL_INCIDENT,
                target,
                target_site_id,
                request_id,
            )
            self._check_version(source, expected_version)
            if source.status not in {
                "SUBMITTED",
                "NEEDS_INFORMATION",
                "TRIAGED",
                "DIAGNOSING",
            }:
                raise IncidentOperationConflict(
                    f"incident in {source.status} cannot be merged",
                    source.version,
                )
            if target.status in {"CANCELLED", "CLOSED"}:
                raise IncidentOperationConflict("canonical incident is not active")
            if source.asset_id != target.asset_id:
                raise IncidentOperationConflict("duplicate incidents must reference the same asset")
            existing_relation = session.scalar(
                select(IncidentRelationRecord).where(
                    IncidentRelationRecord.tenant_id == identity.tenant_id,
                    IncidentRelationRecord.source_incident_id == incident_id,
                    IncidentRelationRecord.relation_type == "DUPLICATE_OF",
                )
            )
            if existing_relation is not None:
                raise IncidentOperationConflict("incident was already merged")
            self._apply_update(
                session,
                source,
                expected_version=expected_version,
                target_status="CANCELLED",
                values={},
                now=now,
            )
            session.add(
                IncidentRelationRecord(
                    relation_id=f"incident-relation-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    source_incident_id=incident_id,
                    target_incident_id=canonical_incident_id,
                    relation_type="DUPLICATE_OF",
                    reason=reason,
                    created_by_subject_id=identity.subject_id,
                    occurred_at=now,
                )
            )
            self._append_control(
                session,
                identity,
                source,
                command_type="MERGE_DUPLICATE",
                target_status="CANCELLED",
                reason=reason,
                details={},
                related_incident_id=canonical_incident_id,
                occurred_at=now,
            )
            return _view_after(source, status="CANCELLED", values={}, occurred_at=now)

    def _transition(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        expected_version: int,
        allowed_statuses: tuple[str, ...],
        target_status: str,
        command_type: str,
        reason: str,
        values: dict[str, Any],
        details: dict[str, Any],
        request_id: str,
        responsible_subject_id: str | None = None,
        recovery_condition: str | None = None,
    ) -> IncidentOperationsView:
        now = datetime.now(UTC)
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            incident, site_id = self._load(session, identity.tenant_id, incident_id)
            self._require(
                identity,
                Action.CONTROL_INCIDENT,
                incident,
                site_id,
                request_id,
            )
            self._check_version(incident, expected_version)
            if incident.status not in allowed_statuses:
                raise IncidentOperationConflict(
                    f"incident in {incident.status} cannot execute {command_type}",
                    incident.version,
                )
            self._apply_update(
                session,
                incident,
                expected_version=expected_version,
                target_status=target_status,
                values=values,
                now=now,
            )
            self._append_control(
                session,
                identity,
                incident,
                command_type=command_type,
                target_status=target_status,
                reason=reason,
                details=details,
                responsible_subject_id=responsible_subject_id,
                recovery_condition=recovery_condition,
                occurred_at=now,
            )
            return _view_after(incident, status=target_status, values=values, occurred_at=now)

    @staticmethod
    def _load(
        session: Session, tenant_id: str, incident_id: str
    ) -> tuple[IncidentRecord, str | None]:
        row = session.execute(
            select(IncidentRecord, AssetSiteLinkRecord.site_id)
            .outerjoin(
                AssetSiteLinkRecord,
                AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
            )
            .where(
                IncidentRecord.tenant_id == tenant_id,
                IncidentRecord.incident_id == incident_id,
            )
        ).one_or_none()
        if row is None:
            raise IncidentNotVisible
        return row[0], row[1]

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        incident: IncidentRecord,
        site_id: str | None,
        request_id: str,
    ) -> None:
        resource_asset_id = incident.asset_id if incident.asset_id in identity.asset_ids else None
        resource_site_id = (
            site_id
            if resource_asset_id is None and site_id is not None and site_id in identity.site_ids
            else None
        )
        if resource_asset_id is None and resource_site_id is None:
            # Supplying the actual asset makes the common Authorizer boundary deny
            # without revealing whether the caller missed asset or site scope.
            resource_asset_id = incident.asset_id
        self._authorizer.require(
            identity,
            action,
            ResourceContext(
                identity.tenant_id,
                resource_id=incident.incident_id,
                asset_id=resource_asset_id,
                site_id=resource_site_id,
            ),
            request_id=request_id,
        )

    @staticmethod
    def _check_version(incident: IncidentRecord, expected_version: int) -> None:
        if incident.version != expected_version:
            raise IncidentOperationConflict("incident version conflict", incident.version)

    @staticmethod
    def _apply_update(
        session: Session,
        incident: IncidentRecord,
        *,
        expected_version: int,
        target_status: str,
        values: dict[str, Any],
        now: datetime,
    ) -> None:
        result = session.execute(
            update(IncidentRecord)
            .where(
                IncidentRecord.tenant_id == incident.tenant_id,
                IncidentRecord.incident_id == incident.incident_id,
                IncidentRecord.version == expected_version,
                IncidentRecord.status == incident.status,
            )
            .values(
                status=target_status,
                version=expected_version + 1,
                updated_at=now,
                **values,
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise IncidentOperationConflict("incident changed concurrently")

    @staticmethod
    def _append_control(
        session: Session,
        identity: IdentityContext,
        incident: IncidentRecord,
        *,
        command_type: str,
        target_status: str,
        reason: str,
        details: dict[str, Any],
        occurred_at: datetime,
        responsible_subject_id: str | None = None,
        recovery_condition: str | None = None,
        related_incident_id: str | None = None,
    ) -> None:
        session.add(
            IncidentControlRecord(
                control_id=f"incident-control-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                incident_id=incident.incident_id,
                incident_version=incident.version + 1,
                command_type=command_type,
                previous_status=incident.status,
                target_status=target_status,
                reason=reason,
                actor_subject_id=identity.subject_id,
                responsible_subject_id=responsible_subject_id,
                recovery_condition=recovery_condition,
                related_incident_id=related_incident_id,
                details_json=details,
                occurred_at=occurred_at,
            )
        )


def record_work_order_resolution_in_session(
    session: Session,
    identity: IdentityContext,
    incident: IncidentRecord,
    work_order: WorkOrderRecord,
    completion: WorkOrderCompletionRecord,
    verification: WorkOrderVerificationRecord,
    *,
    occurred_at: datetime,
) -> IncidentResolutionRecord:
    """Append the Incident-owned result of a verified WorkOrder transaction."""

    if incident.status not in {"WORK_ORDER_CREATED", "RESOLVED"}:
        raise IncidentOperationConflict(
            f"incident in {incident.status} cannot accept work order resolution",
            incident.version,
        )
    if work_order.incident_id != incident.incident_id:
        raise IncidentOperationConflict("work_order_incident_mismatch", incident.version)
    if (
        completion.work_order_id != work_order.work_order_id
        or verification.work_order_id != work_order.work_order_id
        or verification.completion_id != completion.completion_id
        or not verification.passed
    ):
        raise IncidentOperationConflict("work_order_resolution_facts_invalid", incident.version)
    existing = session.scalar(
        select(IncidentResolutionRecord).where(
            IncidentResolutionRecord.tenant_id == identity.tenant_id,
            IncidentResolutionRecord.incident_id == incident.incident_id,
        )
    )
    if existing is not None:
        raise IncidentOperationConflict("incident_resolution_already_exists", incident.version)

    resolution = IncidentResolutionRecord(
        resolution_id=f"incident-resolution-{uuid4().hex}",
        tenant_id=identity.tenant_id,
        incident_id=incident.incident_id,
        mode="WORK_ORDER",
        diagnosis_run_id=None,
        work_order_id=work_order.work_order_id,
        completion_id=completion.completion_id,
        verification_id=verification.verification_id,
        summary=completion.root_cause,
        evidence_ids=list(dict.fromkeys(completion.evidence_ids)),
        resolved_by_subject_id=identity.subject_id,
        resolved_at=occurred_at,
        created_at=occurred_at,
        updated_at=occurred_at,
    )
    previous_status = incident.status
    incident.status = "RESOLVED"
    incident.version += 1
    incident.updated_at = occurred_at
    session.add(resolution)
    session.add(
        IncidentControlRecord(
            control_id=f"incident-control-{uuid4().hex}",
            tenant_id=identity.tenant_id,
            incident_id=incident.incident_id,
            incident_version=incident.version,
            command_type="RESOLVE_WORK_ORDER",
            previous_status=previous_status,
            target_status="RESOLVED",
            reason=completion.root_cause,
            actor_subject_id=identity.subject_id,
            responsible_subject_id=None,
            recovery_condition=None,
            related_incident_id=None,
            details_json={
                "resolution_id": resolution.resolution_id,
                "mode": "WORK_ORDER",
                "work_order_id": work_order.work_order_id,
                "completion_id": completion.completion_id,
                "verification_id": verification.verification_id,
            },
            occurred_at=occurred_at,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
    )
    return resolution


def _view(record: IncidentRecord) -> IncidentOperationsView:
    return IncidentOperationsView(
        incident_id=record.incident_id,
        asset_id=record.asset_id,
        description=record.description,
        status=record.status,
        severity=record.severity,
        category=record.category,
        responsible_queue=record.responsible_queue,
        information_due_at=record.information_due_at,
        escalation_responsible_subject_id=record.escalation_responsible_subject_id,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
        legal_actions=_legal_actions(record.status),
    )


def _view_after(
    record: IncidentRecord,
    *,
    status: str,
    values: dict[str, Any],
    occurred_at: datetime,
) -> IncidentOperationsView:
    return IncidentOperationsView(
        incident_id=record.incident_id,
        asset_id=record.asset_id,
        description=record.description,
        status=status,
        severity=values.get("severity", record.severity),
        category=values.get("category", record.category),
        responsible_queue=values.get("responsible_queue", record.responsible_queue),
        information_due_at=values.get("information_due_at", record.information_due_at),
        escalation_responsible_subject_id=values.get(
            "escalation_responsible_subject_id",
            record.escalation_responsible_subject_id,
        ),
        version=record.version + 1,
        created_at=record.created_at,
        updated_at=occurred_at,
        legal_actions=_legal_actions(status),
    )


def _control_view(record: IncidentControlRecord) -> IncidentControlView:
    return IncidentControlView(
        control_id=record.control_id,
        incident_version=record.incident_version,
        command_type=record.command_type,
        previous_status=record.previous_status,
        target_status=record.target_status,
        reason=record.reason,
        actor_subject_id=record.actor_subject_id,
        responsible_subject_id=record.responsible_subject_id,
        recovery_condition=record.recovery_condition,
        related_incident_id=record.related_incident_id,
        details=record.details_json,
        occurred_at=record.occurred_at,
    )


def _resolution_view(record: IncidentResolutionRecord) -> IncidentResolutionView:
    return IncidentResolutionView(
        resolution_id=record.resolution_id,
        incident_id=record.incident_id,
        mode=record.mode,
        diagnosis_run_id=record.diagnosis_run_id,
        work_order_id=record.work_order_id,
        completion_id=record.completion_id,
        verification_id=record.verification_id,
        summary=record.summary,
        evidence_ids=tuple(record.evidence_ids),
        resolved_by_subject_id=record.resolved_by_subject_id,
        resolved_at=record.resolved_at,
    )


def _legal_actions(status: str) -> tuple[str, ...]:
    return {
        "SUBMITTED": ("CLASSIFY", "REQUEST_INFORMATION", "MERGE_DUPLICATE"),
        "NEEDS_INFORMATION": ("INFORMATION_RECEIVED", "MERGE_DUPLICATE"),
        "TRIAGED": ("ESCALATE", "MERGE_DUPLICATE"),
        "DIAGNOSING": ("REQUEST_INFORMATION", "ESCALATE", "MERGE_DUPLICATE"),
        "DIAGNOSED": ("RESOLVE_REMOTE", "ESCALATE"),
        "WORK_ORDER_CREATED": ("ESCALATE",),
        "RESOLVED": ("CLOSE",),
        "ESCALATED": ("RESUME",),
    }.get(status, ())
