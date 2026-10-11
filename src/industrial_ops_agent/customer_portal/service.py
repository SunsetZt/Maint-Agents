"""Customer-owned incident projection and append-only service updates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import overload
from uuid import uuid4

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
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
    IncidentRecord,
    IncidentResolutionRecord,
    WorkOrderRecord,
)


class CustomerCaseNotVisible(Exception):
    pass


class CustomerCaseConflict(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class CustomerWorkOrderView:
    work_order_id: str
    status: str
    priority: str
    sla_due_at: datetime | None
    assigned_subject_id: str | None
    version: int
    result_accepted: bool | None
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CustomerResolutionView:
    resolution_id: str
    mode: str
    summary: str
    evidence_ids: tuple[str, ...]
    resolved_at: datetime
    result_accepted: bool | None
    confirmation_update_id: str | None
    confirmation_occurred_at: datetime | None


@dataclass(frozen=True, slots=True)
class CustomerCaseView:
    incident_id: str
    asset_id: str
    asset_display_name: str | None
    model_code: str | None
    site_id: str | None
    site_name: str | None
    description: str
    status: str
    version: int
    created_at: datetime
    updated_at: datetime
    work_orders: tuple[CustomerWorkOrderView, ...]
    resolution: CustomerResolutionView | None


@dataclass(frozen=True, slots=True)
class CustomerUpdateView:
    update_id: str
    incident_id: str
    work_order_id: str | None
    client_operation_id: str
    update_type: str
    message: str
    evidence_ids: tuple[str, ...]
    result_accepted: bool | None
    satisfaction_rating: int | None
    actor_subject_id: str
    occurred_at: datetime


class CustomerPortalService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def list_cases(
        self,
        identity: IdentityContext,
        *,
        status: str | None,
        limit: int,
        offset: int,
        request_id: str,
    ) -> tuple[list[CustomerCaseView], int]:
        self._authorizer.require(
            identity,
            Action.READ_CUSTOMER_CASE,
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
                IncidentRecord.reporter_subject_id == identity.subject_id,
                or_(*scope_conditions),
            )
        )
        if status is not None:
            statement = statement.where(IncidentRecord.status == status)
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
            work_orders = self._work_orders_by_incident(
                session,
                identity.tenant_id,
                [incident.incident_id for incident, _asset, _site in rows],
            )
            resolutions = self._resolutions_by_incident(
                session,
                identity.tenant_id,
                [incident.incident_id for incident, _asset, _site in rows],
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
                _case_view(
                    incident,
                    asset,
                    site,
                    work_orders.get(incident.incident_id, ()),
                    resolutions.get(incident.incident_id),
                )
                for incident, asset, site in rows
            ], total

    def get_case(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        request_id: str,
    ) -> CustomerCaseView:
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            incident, asset, site = self._load_case(
                session, identity.tenant_id, identity.subject_id, incident_id
            )
            self._require(
                identity,
                Action.READ_CUSTOMER_CASE,
                incident,
                site.site_id if site is not None else None,
                request_id,
            )
            work_orders = self._work_orders_by_incident(session, identity.tenant_id, [incident_id])
            resolutions = self._resolutions_by_incident(session, identity.tenant_id, [incident_id])
            return _case_view(
                incident,
                asset,
                site,
                work_orders.get(incident_id, ()),
                resolutions.get(incident_id),
            )

    def list_updates(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        request_id: str,
    ) -> list[CustomerUpdateView]:
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            incident, _asset, site = self._load_case(
                session, identity.tenant_id, identity.subject_id, incident_id
            )
            self._require(
                identity,
                Action.READ_CUSTOMER_CASE,
                incident,
                site.site_id if site is not None else None,
                request_id,
            )
            records = session.scalars(
                select(CustomerServiceUpdateRecord)
                .where(
                    CustomerServiceUpdateRecord.tenant_id == identity.tenant_id,
                    CustomerServiceUpdateRecord.incident_id == incident_id,
                )
                .order_by(
                    CustomerServiceUpdateRecord.occurred_at,
                    CustomerServiceUpdateRecord.update_id,
                )
            )
            return [_update_view(record) for record in records]

    def list_updates_for_staff(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        request_id: str,
    ) -> list[CustomerUpdateView]:
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            row = (
                session.execute(
                    select(IncidentRecord, AssetSiteLinkRecord)
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
                raise CustomerCaseNotVisible
            incident, site = row
            self._require(
                identity,
                Action.READ_CUSTOMER_UPDATE,
                incident,
                site.site_id if site is not None else None,
                request_id,
            )
            records = session.scalars(
                select(CustomerServiceUpdateRecord)
                .where(
                    CustomerServiceUpdateRecord.tenant_id == identity.tenant_id,
                    CustomerServiceUpdateRecord.incident_id == incident_id,
                )
                .order_by(
                    CustomerServiceUpdateRecord.occurred_at,
                    CustomerServiceUpdateRecord.update_id,
                )
            )
            return [_update_view(record) for record in records]

    def append_update(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        client_operation_id: str,
        update_type: str,
        message: str,
        evidence_ids: list[str],
        work_order_id: str | None,
        result_accepted: bool | None,
        satisfaction_rating: int | None,
        request_id: str,
    ) -> CustomerUpdateView:
        if update_type not in {"INFORMATION", "EVIDENCE", "RESULT_CONFIRMATION"}:
            raise CustomerCaseConflict("customer_update_type_invalid")
        with incident_read_transaction(
            self._database,
            identity,
            (incident_id,),
            request_id=request_id,
        ) as session:
            incident, _asset, site = self._load_case(
                session, identity.tenant_id, identity.subject_id, incident_id
            )
            action = (
                Action.CONFIRM_SERVICE_RESULT
                if update_type == "RESULT_CONFIRMATION"
                else Action.UPDATE_CUSTOMER_CASE
            )
            self._require(
                identity,
                action,
                incident,
                site.site_id if site is not None else None,
                request_id,
            )
            existing = session.scalar(
                select(CustomerServiceUpdateRecord).where(
                    CustomerServiceUpdateRecord.tenant_id == identity.tenant_id,
                    CustomerServiceUpdateRecord.incident_id == incident_id,
                    CustomerServiceUpdateRecord.client_operation_id == client_operation_id,
                )
            )
            if existing is not None:
                if not _same_update(
                    existing,
                    update_type=update_type,
                    message=message,
                    evidence_ids=evidence_ids,
                    work_order_id=work_order_id,
                    result_accepted=result_accepted,
                    satisfaction_rating=satisfaction_rating,
                ):
                    raise CustomerCaseConflict("idempotency_payload_mismatch")
                return _update_view(existing)
            if incident.status in {"CLOSED", "CANCELLED"}:
                raise CustomerCaseConflict("customer_case_is_terminal")
            if update_type == "EVIDENCE" and not evidence_ids:
                raise CustomerCaseConflict("customer_evidence_required")
            if update_type == "RESULT_CONFIRMATION":
                if result_accepted is None:
                    raise CustomerCaseConflict("service_result_confirmation_incomplete")
                if satisfaction_rating is None or not 1 <= satisfaction_rating <= 5:
                    raise CustomerCaseConflict("satisfaction_rating_invalid")
                if work_order_id is None:
                    resolution = session.scalar(
                        select(IncidentResolutionRecord).where(
                            IncidentResolutionRecord.tenant_id == identity.tenant_id,
                            IncidentResolutionRecord.incident_id == incident_id,
                        )
                    )
                    if resolution is None or resolution.mode != "REMOTE":
                        raise CustomerCaseConflict("remote_resolution_required")
                    if incident.status != "RESOLVED":
                        raise CustomerCaseConflict("service_result_not_ready")
                else:
                    work_order = session.scalar(
                        select(WorkOrderRecord).where(
                            WorkOrderRecord.tenant_id == identity.tenant_id,
                            WorkOrderRecord.work_order_id == work_order_id,
                            WorkOrderRecord.incident_id == incident_id,
                        )
                    )
                    if work_order is None:
                        raise CustomerCaseNotVisible
                    if work_order.status not in {"COMPLETED", "VERIFIED"}:
                        raise CustomerCaseConflict("service_result_not_ready")
            elif work_order_id is not None:
                belongs = session.scalar(
                    select(WorkOrderRecord.work_order_id).where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.work_order_id == work_order_id,
                        WorkOrderRecord.incident_id == incident_id,
                    )
                )
                if belongs is None:
                    raise CustomerCaseNotVisible
            now = datetime.now(UTC)
            record = CustomerServiceUpdateRecord(
                update_id=f"customer-update-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                incident_id=incident_id,
                work_order_id=work_order_id,
                client_operation_id=client_operation_id,
                update_type=update_type,
                message=message,
                evidence_ids=evidence_ids,
                result_accepted=result_accepted,
                satisfaction_rating=satisfaction_rating,
                actor_subject_id=identity.subject_id,
                occurred_at=now,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            return _update_view(record)

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        incident: IncidentRecord,
        site_id: str | None,
        request_id: str,
    ) -> None:
        if incident.asset_id in identity.asset_ids:
            resource = ResourceContext(
                identity.tenant_id,
                incident.incident_id,
                asset_id=incident.asset_id,
            )
        elif site_id is not None and site_id in identity.site_ids:
            resource = ResourceContext(
                identity.tenant_id,
                incident.incident_id,
                site_id=site_id,
            )
        else:
            resource = ResourceContext(
                identity.tenant_id,
                incident.incident_id,
                asset_id=incident.asset_id,
            )
        self._authorizer.require(identity, action, resource, request_id=request_id)

    @staticmethod
    def _load_case(
        session: Session,
        tenant_id: str,
        subject_id: str,
        incident_id: str,
    ) -> tuple[IncidentRecord, AssetRecord, AssetSiteLinkRecord | None]:
        row = (
            session.execute(
                select(IncidentRecord, AssetRecord, AssetSiteLinkRecord)
                .join(AssetRecord, AssetRecord.asset_id == IncidentRecord.asset_id)
                .outerjoin(
                    AssetSiteLinkRecord,
                    AssetSiteLinkRecord.asset_id == IncidentRecord.asset_id,
                )
                .where(
                    IncidentRecord.tenant_id == tenant_id,
                    AssetRecord.tenant_id == tenant_id,
                    IncidentRecord.incident_id == incident_id,
                    IncidentRecord.reporter_subject_id == subject_id,
                )
            )
            .tuples()
            .one_or_none()
        )
        if row is None:
            raise CustomerCaseNotVisible
        return row

    @staticmethod
    def _work_orders_by_incident(
        session: Session,
        tenant_id: str,
        incident_ids: list[str],
    ) -> dict[str, tuple[CustomerWorkOrderView, ...]]:
        if not incident_ids:
            return {}
        work_orders = list(
            session.scalars(
                select(WorkOrderRecord)
                .where(
                    WorkOrderRecord.tenant_id == tenant_id,
                    WorkOrderRecord.incident_id.in_(incident_ids),
                )
                .order_by(WorkOrderRecord.created_at, WorkOrderRecord.work_order_id)
            )
        )
        confirmations = (
            list(
                session.scalars(
                    select(CustomerServiceUpdateRecord)
                    .where(
                        CustomerServiceUpdateRecord.tenant_id == tenant_id,
                        CustomerServiceUpdateRecord.work_order_id.in_(
                            [work.work_order_id for work in work_orders]
                        ),
                        CustomerServiceUpdateRecord.update_type == "RESULT_CONFIRMATION",
                    )
                    .order_by(CustomerServiceUpdateRecord.occurred_at)
                )
            )
            if work_orders
            else []
        )
        latest_confirmation = {
            confirmation.work_order_id: confirmation.result_accepted
            for confirmation in confirmations
        }
        grouped: dict[str, list[CustomerWorkOrderView]] = {}
        for work in work_orders:
            grouped.setdefault(work.incident_id, []).append(
                CustomerWorkOrderView(
                    work_order_id=work.work_order_id,
                    status=work.status,
                    priority=work.priority,
                    sla_due_at=_utc(work.sla_due_at),
                    assigned_subject_id=work.assigned_subject_id,
                    version=work.version,
                    result_accepted=latest_confirmation.get(work.work_order_id),
                    updated_at=_utc(work.updated_at),
                )
            )
        return {key: tuple(value) for key, value in grouped.items()}

    @staticmethod
    def _resolutions_by_incident(
        session: Session,
        tenant_id: str,
        incident_ids: list[str],
    ) -> dict[str, CustomerResolutionView]:
        if not incident_ids:
            return {}
        resolutions = list(
            session.scalars(
                select(IncidentResolutionRecord).where(
                    IncidentResolutionRecord.tenant_id == tenant_id,
                    IncidentResolutionRecord.incident_id.in_(incident_ids),
                )
            )
        )
        confirmations = list(
            session.scalars(
                select(CustomerServiceUpdateRecord)
                .where(
                    CustomerServiceUpdateRecord.tenant_id == tenant_id,
                    CustomerServiceUpdateRecord.incident_id.in_(incident_ids),
                    CustomerServiceUpdateRecord.work_order_id.is_(None),
                    CustomerServiceUpdateRecord.update_type == "RESULT_CONFIRMATION",
                )
                .order_by(
                    CustomerServiceUpdateRecord.occurred_at,
                    CustomerServiceUpdateRecord.update_id,
                )
            )
        )
        latest_confirmation = {
            confirmation.incident_id: confirmation for confirmation in confirmations
        }
        return {
            resolution.incident_id: _resolution_view(
                resolution,
                latest_confirmation.get(resolution.incident_id),
            )
            for resolution in resolutions
        }


def _case_view(
    incident: IncidentRecord,
    asset: AssetRecord,
    site: AssetSiteLinkRecord | None,
    work_orders: tuple[CustomerWorkOrderView, ...],
    resolution: CustomerResolutionView | None,
) -> CustomerCaseView:
    return CustomerCaseView(
        incident_id=incident.incident_id,
        asset_id=incident.asset_id,
        asset_display_name=asset.display_name,
        model_code=asset.model_code,
        site_id=site.site_id if site is not None else None,
        site_name=site.site_name if site is not None else None,
        description=incident.description,
        status=incident.status,
        version=incident.version,
        created_at=_utc(incident.created_at),
        updated_at=_utc(incident.updated_at),
        work_orders=work_orders,
        resolution=resolution,
    )


def _resolution_view(
    record: IncidentResolutionRecord,
    confirmation: CustomerServiceUpdateRecord | None,
) -> CustomerResolutionView:
    return CustomerResolutionView(
        resolution_id=record.resolution_id,
        mode=record.mode,
        summary=record.summary,
        evidence_ids=tuple(record.evidence_ids),
        resolved_at=_utc(record.resolved_at),
        result_accepted=(confirmation.result_accepted if confirmation is not None else None),
        confirmation_update_id=(confirmation.update_id if confirmation is not None else None),
        confirmation_occurred_at=(
            _utc(confirmation.occurred_at) if confirmation is not None else None
        ),
    )


def _update_view(record: CustomerServiceUpdateRecord) -> CustomerUpdateView:
    return CustomerUpdateView(
        update_id=record.update_id,
        incident_id=record.incident_id,
        work_order_id=record.work_order_id,
        client_operation_id=record.client_operation_id,
        update_type=record.update_type,
        message=record.message,
        evidence_ids=tuple(record.evidence_ids),
        result_accepted=record.result_accepted,
        satisfaction_rating=record.satisfaction_rating,
        actor_subject_id=record.actor_subject_id,
        occurred_at=_utc(record.occurred_at),
    )


def _same_update(
    record: CustomerServiceUpdateRecord,
    *,
    update_type: str,
    message: str,
    evidence_ids: list[str],
    work_order_id: str | None,
    result_accepted: bool | None,
    satisfaction_rating: int | None,
) -> bool:
    return (
        record.update_type == update_type
        and record.message == message
        and record.evidence_ids == evidence_ids
        and record.work_order_id == work_order_id
        and record.result_accepted == result_accepted
        and record.satisfaction_rating == satisfaction_rating
    )


@overload
def _utc(value: datetime) -> datetime: ...


@overload
def _utc(value: None) -> None: ...


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
