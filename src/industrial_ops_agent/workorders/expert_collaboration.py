"""Governed WorkOrder invitation and participant binding for expert audio."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.maintenance_planning.review_isolation import (
    ReviewIsolationConflict,
    lock_subject,
    record_incident_disclosure,
    work_order_read_transaction,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetScopeRecord,
    AssetSiteLinkRecord,
    IncidentRecord,
    RealtimeMediaSessionRecord,
    SubjectRecord,
    SubjectRoleRecord,
    WorkOrderExpertCollaborationEventRecord,
    WorkOrderExpertCollaborationRecommendationRecord,
    WorkOrderExpertCollaborationRecord,
    WorkOrderFieldEntryRecord,
    WorkOrderRecord,
)

COLLABORATIVE_WORK_ORDER_STATUSES = frozenset({"ACCEPTED", "IN_PROGRESS", "ON_HOLD", "ESCALATED"})
OPEN_COLLABORATION_STATUSES = frozenset({"WAITING_EXPERT", "ACTIVE"})


class ExpertCollaborationNotVisible(Exception):
    pass


class ExpertCollaborationConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ExpertCandidate:
    subject_id: str
    display_name: str


@dataclass(frozen=True, slots=True)
class ExpertCollaborationEvent:
    sequence: int
    event_type: str
    actor_subject_id: str
    from_status: str | None
    to_status: str
    reason_code: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class ExpertRecommendation:
    recommendation_id: str
    collaboration_id: str
    work_order_id: str
    work_order_version: int
    incident_id: str
    asset_id: str
    expert_subject_id: str
    recommendation_type: str
    summary: str
    basis: str
    recommended_checks: tuple[str, ...]
    safety_notice: str | None
    evidence_entry_ids: tuple[str, ...]
    status: str
    reviewed_by_subject_id: str | None
    review_reason: str | None
    reviewed_at: datetime | None
    field_entry_id: str | None
    version: int
    legal_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpertCollaboration:
    collaboration_id: str
    work_order_id: str
    work_order_version: int
    incident_id: str
    asset_id: str
    requested_by_subject_id: str
    invited_subject_id: str
    reason: str
    status: str
    accepted_at: datetime | None
    ended_by_subject_id: str | None
    ended_reason: str | None
    ended_at: datetime | None
    expires_at: datetime
    version: int
    recommendation: ExpertRecommendation | None
    legal_actions: tuple[str, ...]
    events: tuple[ExpertCollaborationEvent, ...]


@dataclass(frozen=True, slots=True)
class ExpertCollaborationOverview:
    work_order_id: str
    work_order_version: int
    collaboration: ExpertCollaboration | None
    candidates: tuple[ExpertCandidate, ...]
    legal_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpertCollaborationBinding:
    collaboration_id: str
    work_order_id: str
    work_order_version: int
    incident_id: str
    asset_id: str
    participant_subject_id: str


class ExpertCollaborationService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def field_overview(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> ExpertCollaborationOverview:
        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            work, incident, site_id = self._field_work_order(
                session, identity, work_order_id, request_id=request_id
            )
            record = _current_collaboration(
                session, identity.tenant_id, work_order_id, for_update=True
            )
            if record is None:
                record = _latest_recommendation_collaboration(
                    session,
                    identity.tenant_id,
                    work_order_id,
                    for_update=True,
                )
            if record is not None:
                _expire_if_needed(session, record)
            collaboration = (
                self._view(session, identity, record, request_id=request_id)
                if record is not None
                else None
            )
            candidates = tuple(
                ExpertCandidate(item.subject_id, item.display_name or item.subject_id)
                for item in _eligible_experts(
                    session,
                    tenant_id=identity.tenant_id,
                    asset_id=incident.asset_id,
                    site_id=site_id,
                    exclude_subject_id=identity.subject_id,
                )
            )
            has_open_collaboration = bool(
                record is not None and record.status in OPEN_COLLABORATION_STATUSES
            )
            legal_actions = (
                ("INVITE",) if not has_open_collaboration and candidates else tuple()
            )
            return ExpertCollaborationOverview(
                work.work_order_id,
                work.version,
                collaboration,
                candidates,
                legal_actions,
            )

    def create(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        work_order_version: int,
        invited_subject_id: str,
        reason: str,
        idempotency_key: str,
        request_id: str,
    ) -> tuple[ExpertCollaboration, bool]:
        normalized_reason = " ".join(reason.split())
        digest = _request_digest(
            work_order_id,
            work_order_version,
            identity.subject_id,
            invited_subject_id,
            normalized_reason,
        )
        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            work, incident, site_id = self._field_work_order(
                session, identity, work_order_id, request_id=request_id
            )
            now = datetime.now(UTC)
            existing = session.scalar(
                select(WorkOrderExpertCollaborationRecord)
                .where(
                    WorkOrderExpertCollaborationRecord.tenant_id == identity.tenant_id,
                    WorkOrderExpertCollaborationRecord.requested_by_subject_id
                    == identity.subject_id,
                    WorkOrderExpertCollaborationRecord.idempotency_key == idempotency_key,
                )
                .with_for_update()
            )
            if existing is not None:
                if existing.request_digest != digest:
                    raise ExpertCollaborationConflict(
                        "expert_collaboration_idempotency_conflict", existing.version
                    )
                _expire_if_needed(session, existing, now=now)
                return self._view(session, identity, existing, request_id=request_id), False
            if work.version != work_order_version:
                raise ExpertCollaborationConflict(
                    "expert_collaboration_work_order_version_conflict", work.version
                )
            if not _directory_eligible(
                session,
                tenant_id=identity.tenant_id,
                subject_id=invited_subject_id,
                role=Role.DOMAIN_EXPERT.value,
                asset_id=incident.asset_id,
                site_id=site_id,
            ):
                raise ExpertCollaborationNotVisible
            current = _current_collaboration(
                session, identity.tenant_id, work_order_id, for_update=True
            )
            if current is not None:
                _expire_if_needed(session, current, now=now)
                if current.status in OPEN_COLLABORATION_STATUSES:
                    raise ExpertCollaborationConflict(
                        "expert_collaboration_already_open", current.version
                    )
            record = WorkOrderExpertCollaborationRecord(
                collaboration_id=f"expert-collab-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                work_order_id=work.work_order_id,
                work_order_version=work.version,
                incident_id=incident.incident_id,
                asset_id=incident.asset_id,
                requested_by_subject_id=identity.subject_id,
                invited_subject_id=invited_subject_id,
                reason=normalized_reason,
                status="WAITING_EXPERT",
                idempotency_key=idempotency_key,
                request_digest=digest,
                accepted_at=None,
                ended_by_subject_id=None,
                ended_reason=None,
                ended_at=None,
                expires_at=now + timedelta(minutes=30),
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            _append_event(
                session,
                record,
                sequence=1,
                event_type="CREATED",
                actor_subject_id=identity.subject_id,
                from_status=None,
                to_status="WAITING_EXPERT",
                reason_code="field_owner_invited_expert",
                occurred_at=now,
            )
            session.flush()
            return self._view(session, identity, record, request_id=request_id), True

    def list_for_expert(
        self, identity: IdentityContext, *, request_id: str
    ) -> tuple[ExpertCollaboration, ...]:
        if Role.DOMAIN_EXPERT not in identity.roles:
            raise ExpertCollaborationNotVisible
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            records = list(
                session.scalars(
                    select(WorkOrderExpertCollaborationRecord)
                    .where(
                        WorkOrderExpertCollaborationRecord.tenant_id == identity.tenant_id,
                        WorkOrderExpertCollaborationRecord.invited_subject_id
                        == identity.subject_id,
                        or_(
                            WorkOrderExpertCollaborationRecord.status.in_(
                                OPEN_COLLABORATION_STATUSES
                            ),
                            WorkOrderExpertCollaborationRecord.accepted_at.is_not(None),
                        ),
                    )
                    .order_by(WorkOrderExpertCollaborationRecord.created_at.desc())
                    .limit(100)
                    .with_for_update()
                )
            )
            result: list[ExpertCollaboration] = []
            for record in records:
                _expire_if_needed(session, record)
                if (
                    record.status not in OPEN_COLLABORATION_STATUSES
                    and record.accepted_at is None
                ):
                    continue
                self._require(
                    identity,
                    Action.READ_EXPERT_COLLABORATION,
                    record,
                    request_id=request_id,
                )
                result.append(
                    self._view(session, identity, record, request_id=request_id)
                )
            return tuple(result)

    def get(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        request_id: str,
    ) -> ExpertCollaboration:
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _load_collaboration(
                session, identity.tenant_id, collaboration_id, for_update=True
            )
            self._require_participant(identity, record, request_id=request_id)
            _expire_if_needed(session, record)
            return self._view(session, identity, record, request_id=request_id)

    def accept(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> ExpertCollaboration:
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _load_collaboration(
                session, identity.tenant_id, collaboration_id, for_update=True
            )
            now = datetime.now(UTC)
            if record.invited_subject_id != identity.subject_id:
                raise ExpertCollaborationNotVisible
            self._require(
                identity,
                Action.ACCEPT_EXPERT_COLLABORATION,
                record,
                request_id=request_id,
            )
            _expire_if_needed(session, record, now=now)
            if record.version != expected_version:
                raise ExpertCollaborationConflict(
                    "expert_collaboration_version_conflict", record.version
                )
            if record.status != "WAITING_EXPERT":
                raise ExpertCollaborationConflict(
                    "expert_collaboration_not_waiting", record.version
                )
            if not expert_collaboration_binding_current(
                session,
                tenant_id=identity.tenant_id,
                collaboration_id=record.collaboration_id,
                participant_subject_id=identity.subject_id,
                require_active=False,
            ):
                raise ExpertCollaborationConflict(
                    "expert_collaboration_binding_changed", record.version
                )
            record.status = "ACTIVE"
            record.accepted_at = now
            record.expires_at = now + timedelta(minutes=60)
            record.version += 1
            record.updated_at = now
            _append_event(
                session,
                record,
                sequence=record.version,
                event_type="ACCEPTED",
                actor_subject_id=identity.subject_id,
                from_status="WAITING_EXPERT",
                to_status="ACTIVE",
                reason_code="invited_expert_accepted",
                occurred_at=now,
            )
            session.flush()
            return self._view(session, identity, record, request_id=request_id)

    def submit_recommendation(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        recommendation_type: str,
        summary: str,
        basis: str,
        recommended_checks: tuple[str, ...],
        safety_notice: str | None,
        evidence_entry_ids: tuple[str, ...],
        idempotency_key: str,
        request_id: str,
    ) -> tuple[ExpertCollaboration, bool]:
        if recommendation_type not in {
            "ADVICE",
            "REQUEST_MORE_EVIDENCE",
            "STOP_AND_ESCALATE",
        }:
            raise ExpertCollaborationConflict("expert_recommendation_type_invalid")
        if not recommended_checks or len(recommended_checks) > 10:
            raise ExpertCollaborationConflict("expert_recommendation_checks_invalid")
        normalized_summary = _normalize_text(summary)
        normalized_basis = _normalize_text(basis)
        normalized_checks = tuple(_normalize_text(item) for item in recommended_checks)
        normalized_safety = _normalize_optional_text(safety_notice)
        if len(set(evidence_entry_ids)) != len(evidence_entry_ids):
            raise ExpertCollaborationConflict("expert_recommendation_evidence_invalid")
        normalized_evidence = tuple(sorted(evidence_entry_ids))
        digest = _recommendation_request_digest(
            collaboration_id=collaboration_id,
            expert_subject_id=identity.subject_id,
            recommendation_type=recommendation_type,
            summary=normalized_summary,
            basis=normalized_basis,
            recommended_checks=normalized_checks,
            safety_notice=normalized_safety,
            evidence_entry_ids=normalized_evidence,
        )
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _load_collaboration(
                session, identity.tenant_id, collaboration_id, for_update=True
            )
            now = datetime.now(UTC)
            if record.invited_subject_id != identity.subject_id:
                raise ExpertCollaborationNotVisible
            self._require(
                identity,
                Action.SUBMIT_EXPERT_RECOMMENDATION,
                record,
                request_id=request_id,
            )
            _expire_if_needed(session, record, now=now)
            idempotent_recommendation = session.scalar(
                select(WorkOrderExpertCollaborationRecommendationRecord)
                .where(
                    WorkOrderExpertCollaborationRecommendationRecord.tenant_id
                    == identity.tenant_id,
                    WorkOrderExpertCollaborationRecommendationRecord.expert_subject_id
                    == identity.subject_id,
                    WorkOrderExpertCollaborationRecommendationRecord.idempotency_key
                    == idempotency_key,
                )
                .with_for_update()
            )
            if idempotent_recommendation is not None:
                if (
                    idempotent_recommendation.collaboration_id != collaboration_id
                    or idempotent_recommendation.request_digest != digest
                ):
                    raise ExpertCollaborationConflict(
                        "expert_recommendation_idempotency_conflict",
                        idempotent_recommendation.version,
                    )
                return self._view(session, identity, record, request_id=request_id), False
            existing = _load_recommendation(
                session,
                identity.tenant_id,
                collaboration_id,
                for_update=True,
                required=False,
            )
            if existing is not None:
                raise ExpertCollaborationConflict(
                    "expert_recommendation_already_exists", existing.version
                )
            if record.accepted_at is None or not _business_binding_current(
                session, record, for_update=True
            ):
                raise ExpertCollaborationConflict(
                    "expert_recommendation_binding_changed", record.version
                )
            if normalized_evidence:
                found_ids = set(
                    session.scalars(
                        select(WorkOrderFieldEntryRecord.entry_id).where(
                            WorkOrderFieldEntryRecord.tenant_id == identity.tenant_id,
                            WorkOrderFieldEntryRecord.work_order_id == record.work_order_id,
                            WorkOrderFieldEntryRecord.entry_id.in_(normalized_evidence),
                        )
                    )
                )
                if found_ids != set(normalized_evidence):
                    raise ExpertCollaborationConflict(
                        "expert_recommendation_evidence_invalid"
                    )
            recommendation = WorkOrderExpertCollaborationRecommendationRecord(
                recommendation_id=f"expert-recommendation-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                collaboration_id=record.collaboration_id,
                work_order_id=record.work_order_id,
                work_order_version=record.work_order_version,
                incident_id=record.incident_id,
                asset_id=record.asset_id,
                expert_subject_id=identity.subject_id,
                recommendation_type=recommendation_type,
                summary=normalized_summary,
                basis=normalized_basis,
                recommended_checks_json=list(normalized_checks),
                safety_notice=normalized_safety,
                evidence_entry_ids_json=list(normalized_evidence),
                status="PENDING_FIELD_REVIEW",
                idempotency_key=idempotency_key,
                request_digest=digest,
                reviewed_by_subject_id=None,
                review_reason=None,
                reviewed_at=None,
                field_entry_id=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(recommendation)
            session.flush()
            return self._view(session, identity, record, request_id=request_id), True

    def decide_recommendation(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        expected_version: int,
        decision: str,
        reason: str,
        request_id: str,
    ) -> ExpertCollaboration:
        if decision not in {"ACCEPT", "REJECT"}:
            raise ExpertCollaborationConflict("expert_recommendation_decision_invalid")
        normalized_reason = _normalize_text(reason)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _load_collaboration(
                session, identity.tenant_id, collaboration_id, for_update=True
            )
            now = datetime.now(UTC)
            if record.requested_by_subject_id != identity.subject_id:
                raise ExpertCollaborationNotVisible
            self._require(
                identity,
                Action.REVIEW_EXPERT_RECOMMENDATION,
                record,
                request_id=request_id,
            )
            recommendation = _load_recommendation(
                session,
                identity.tenant_id,
                collaboration_id,
                for_update=True,
                required=True,
            )
            if recommendation is None:
                raise ExpertCollaborationNotVisible
            if recommendation.version != expected_version:
                raise ExpertCollaborationConflict(
                    "expert_recommendation_version_conflict", recommendation.version
                )
            if recommendation.status != "PENDING_FIELD_REVIEW":
                raise ExpertCollaborationConflict(
                    "expert_recommendation_not_pending", recommendation.version
                )
            if not _business_binding_current(session, record, for_update=True):
                raise ExpertCollaborationConflict(
                    "expert_recommendation_binding_changed", recommendation.version
                )
            recommendation.status = "ACCEPTED" if decision == "ACCEPT" else "REJECTED"
            recommendation.reviewed_by_subject_id = identity.subject_id
            recommendation.review_reason = normalized_reason
            recommendation.reviewed_at = now
            recommendation.version += 1
            recommendation.updated_at = now
            if decision == "ACCEPT":
                sequence = (
                    int(
                        session.scalar(
                            select(func.max(WorkOrderFieldEntryRecord.sequence)).where(
                                WorkOrderFieldEntryRecord.tenant_id == identity.tenant_id,
                                WorkOrderFieldEntryRecord.work_order_id == record.work_order_id,
                            )
                        )
                        or 0
                    )
                    + 1
                )
                field_entry = WorkOrderFieldEntryRecord(
                    entry_id=f"field-entry-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    work_order_id=record.work_order_id,
                    sequence=sequence,
                    client_operation_id=(
                        f"expert-recommendation:{recommendation.recommendation_id}"
                    ),
                    entry_type="EXPERT_RECOMMENDATION",
                    payload=_recommendation_field_payload(recommendation),
                    actor_subject_id=identity.subject_id,
                    occurred_at=now,
                    created_at=now,
                    updated_at=now,
                )
                session.add(field_entry)
                session.flush()
                recommendation.field_entry_id = field_entry.entry_id
            session.flush()
            return self._view(session, identity, record, request_id=request_id)

    def end(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        expected_version: int,
        reason: str,
        request_id: str,
    ) -> ExpertCollaboration:
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _load_collaboration(
                session, identity.tenant_id, collaboration_id, for_update=True
            )
            now = datetime.now(UTC)
            self._require_participant(identity, record, request_id=request_id)
            _expire_if_needed(session, record, now=now)
            if record.version != expected_version:
                raise ExpertCollaborationConflict(
                    "expert_collaboration_version_conflict", record.version
                )
            if record.status not in OPEN_COLLABORATION_STATUSES:
                raise ExpertCollaborationConflict("expert_collaboration_not_open", record.version)
            previous = record.status
            record.status = "ENDED"
            record.ended_by_subject_id = identity.subject_id
            record.ended_reason = " ".join(reason.split())
            record.ended_at = now
            record.version += 1
            record.updated_at = now
            _append_event(
                session,
                record,
                sequence=record.version,
                event_type="ENDED",
                actor_subject_id=identity.subject_id,
                from_status=previous,
                to_status="ENDED",
                reason_code="participant_ended",
                occurred_at=now,
            )
            _revoke_media(session, record.collaboration_id, "collaboration_ended")
            session.flush()
            return self._view(session, identity, record, request_id=request_id)

    def require_active_participant(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        request_id: str,
    ) -> ExpertCollaborationBinding:
        with self._database.transaction(identity.tenant_context) as session:
            record = _load_collaboration(
                session, identity.tenant_id, collaboration_id, for_update=True
            )
            self._require_participant(identity, record, request_id=request_id)
            _expire_if_needed(session, record)
            if not expert_collaboration_binding_current(
                session,
                tenant_id=identity.tenant_id,
                collaboration_id=collaboration_id,
                participant_subject_id=identity.subject_id,
                require_active=True,
            ):
                _revoke_media(session, collaboration_id, "collaboration_binding_changed")
                raise ExpertCollaborationConflict(
                    "expert_collaboration_binding_changed", record.version
                )
            return ExpertCollaborationBinding(
                record.collaboration_id,
                record.work_order_id,
                record.work_order_version,
                record.incident_id,
                record.asset_id,
                identity.subject_id,
            )

    def _field_work_order(
        self,
        session: Session,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> tuple[WorkOrderRecord, IncidentRecord, str | None]:
        row = (
            session.execute(
                select(WorkOrderRecord, IncidentRecord)
                .join(IncidentRecord, IncidentRecord.incident_id == WorkOrderRecord.incident_id)
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
            raise ExpertCollaborationNotVisible
        work, incident = row
        site_id = session.scalar(
            select(AssetSiteLinkRecord.site_id).where(
                AssetSiteLinkRecord.tenant_id == identity.tenant_id,
                AssetSiteLinkRecord.asset_id == incident.asset_id,
            )
        )
        if work.assigned_subject_id != identity.subject_id:
            raise ExpertCollaborationNotVisible
        if work.status not in COLLABORATIVE_WORK_ORDER_STATUSES:
            raise ExpertCollaborationConflict(
                "expert_collaboration_work_order_not_collaborative", work.version
            )
        resource = WorkOrderExpertCollaborationRecord(
            collaboration_id="authorization-probe",
            tenant_id=identity.tenant_id,
            work_order_id=work.work_order_id,
            work_order_version=work.version,
            incident_id=incident.incident_id,
            asset_id=incident.asset_id,
            requested_by_subject_id=identity.subject_id,
            invited_subject_id="authorization-probe",
            reason="authorization-probe",
            status="WAITING_EXPERT",
            idempotency_key="authorization-probe",
            request_digest="authorization-probe",
            expires_at=datetime.now(UTC),
            version=1,
        )
        self._require(
            identity,
            Action.CREATE_EXPERT_COLLABORATION,
            resource,
            request_id=request_id,
            site_id=site_id,
        )
        return work, incident, site_id

    def _require_participant(
        self,
        identity: IdentityContext,
        record: WorkOrderExpertCollaborationRecord,
        *,
        request_id: str,
    ) -> None:
        if identity.subject_id not in {
            record.requested_by_subject_id,
            record.invited_subject_id,
        }:
            raise ExpertCollaborationNotVisible
        self._require(
            identity,
            Action.READ_EXPERT_COLLABORATION,
            record,
            request_id=request_id,
        )

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        record: WorkOrderExpertCollaborationRecord,
        *,
        request_id: str,
        site_id: str | None = None,
    ) -> None:
        resource = ResourceContext(
            identity.tenant_id,
            record.collaboration_id,
            site_id=site_id if site_id in identity.site_ids else None,
            asset_id=record.asset_id if site_id not in identity.site_ids else None,
        )
        try:
            self._authorizer.require(identity, action, resource, request_id=request_id)
        except AuthorizationDenied as exc:
            raise ExpertCollaborationNotVisible from exc

    def _view(
        self,
        session: Session,
        identity: IdentityContext,
        record: WorkOrderExpertCollaborationRecord,
        *,
        request_id: str,
    ) -> ExpertCollaboration:
        events = tuple(
            ExpertCollaborationEvent(
                item.sequence,
                item.event_type,
                item.actor_subject_id,
                item.from_status,
                item.to_status,
                item.reason_code,
                _aware(item.occurred_at),
            )
            for item in session.scalars(
                select(WorkOrderExpertCollaborationEventRecord)
                .where(
                    WorkOrderExpertCollaborationEventRecord.tenant_id == record.tenant_id,
                    WorkOrderExpertCollaborationEventRecord.collaboration_id
                    == record.collaboration_id,
                )
                .order_by(WorkOrderExpertCollaborationEventRecord.sequence)
            )
        )
        recommendation_record = _load_recommendation(
            session,
            record.tenant_id,
            record.collaboration_id,
            for_update=False,
            required=False,
        )
        # Compare immutable source bindings, not the current WorkOrder state/version.
        if recommendation_record is not None and (
            recommendation_record.work_order_id != record.work_order_id
            or recommendation_record.work_order_version != record.work_order_version
            or recommendation_record.incident_id != record.incident_id
            or recommendation_record.asset_id != record.asset_id
        ):
            raise ReviewIsolationConflict("source_binding_invalid")
        record_incident_disclosure(
            session,
            identity,
            (record.incident_id,),
            resource_kind="expert-collaboration",
            resource_id=record.collaboration_id,
            request_id=request_id,
        )
        binding_current = _business_binding_current(session, record)
        recommendation = (
            _recommendation_view(
                identity.subject_id,
                record,
                recommendation_record,
                binding_current=binding_current,
            )
            if recommendation_record is not None
            else None
        )
        return ExpertCollaboration(
            record.collaboration_id,
            record.work_order_id,
            record.work_order_version,
            record.incident_id,
            record.asset_id,
            record.requested_by_subject_id,
            record.invited_subject_id,
            record.reason,
            record.status,
            _optional_aware(record.accepted_at),
            record.ended_by_subject_id,
            record.ended_reason,
            _optional_aware(record.ended_at),
            _aware(record.expires_at),
            record.version,
            recommendation,
            _legal_actions(
                identity.subject_id,
                record,
                recommendation_record,
                binding_current=binding_current,
            ),
            events,
        )


def expert_collaboration_binding_current(
    session: Session,
    *,
    tenant_id: str,
    collaboration_id: str,
    participant_subject_id: str,
    require_active: bool = True,
) -> bool:
    """Revalidate collaboration, WorkOrder and current directory projections."""

    record = session.scalar(
        select(WorkOrderExpertCollaborationRecord).where(
            WorkOrderExpertCollaborationRecord.tenant_id == tenant_id,
            WorkOrderExpertCollaborationRecord.collaboration_id == collaboration_id,
        )
    )
    if record is None:
        return False
    allowed_statuses = {"ACTIVE"} if require_active else {"WAITING_EXPERT"}
    if record.status not in allowed_statuses or datetime.now(UTC) >= _aware(record.expires_at):
        return False
    if participant_subject_id not in {
        record.requested_by_subject_id,
        record.invited_subject_id,
    }:
        return False
    return _business_binding_current(session, record)


def _business_binding_current(
    session: Session,
    record: WorkOrderExpertCollaborationRecord,
    *,
    for_update: bool = False,
) -> bool:
    statement = (
        select(WorkOrderRecord, IncidentRecord)
        .join(IncidentRecord, IncidentRecord.incident_id == WorkOrderRecord.incident_id)
        .where(
            WorkOrderRecord.tenant_id == record.tenant_id,
            IncidentRecord.tenant_id == record.tenant_id,
            WorkOrderRecord.work_order_id == record.work_order_id,
        )
    )
    if for_update:
        statement = statement.with_for_update(of=WorkOrderRecord)
    row = session.execute(statement).tuples().one_or_none()
    if row is None:
        return False
    work, incident = row
    if (
        work.version != record.work_order_version
        or work.incident_id != record.incident_id
        or work.assigned_subject_id != record.requested_by_subject_id
        or work.status not in COLLABORATIVE_WORK_ORDER_STATUSES
        or incident.asset_id != record.asset_id
    ):
        return False
    site_id = session.scalar(
        select(AssetSiteLinkRecord.site_id).where(
            AssetSiteLinkRecord.tenant_id == record.tenant_id,
            AssetSiteLinkRecord.asset_id == record.asset_id,
        )
    )
    return _directory_eligible(
        session,
        tenant_id=record.tenant_id,
        subject_id=record.requested_by_subject_id,
        role=Role.FIELD_ENGINEER.value,
        asset_id=record.asset_id,
        site_id=site_id,
    ) and _directory_eligible(
        session,
        tenant_id=record.tenant_id,
        subject_id=record.invited_subject_id,
        role=Role.DOMAIN_EXPERT.value,
        asset_id=record.asset_id,
        site_id=site_id,
    )


def _eligible_experts(
    session: Session,
    *,
    tenant_id: str,
    asset_id: str,
    site_id: str | None,
    exclude_subject_id: str,
) -> tuple[SubjectRecord, ...]:
    subjects = list(
        session.scalars(
            select(SubjectRecord)
            .join(
                SubjectRoleRecord,
                (SubjectRoleRecord.tenant_id == SubjectRecord.tenant_id)
                & (SubjectRoleRecord.subject_id == SubjectRecord.subject_id),
            )
            .where(
                SubjectRecord.tenant_id == tenant_id,
                SubjectRecord.status == "active",
                SubjectRecord.subject_id != exclude_subject_id,
                SubjectRoleRecord.role == Role.DOMAIN_EXPERT.value,
            )
            .order_by(SubjectRecord.display_name, SubjectRecord.subject_id)
        )
    )
    return tuple(
        subject
        for subject in subjects
        if _directory_eligible(
            session,
            tenant_id=tenant_id,
            subject_id=subject.subject_id,
            role=Role.DOMAIN_EXPERT.value,
            asset_id=asset_id,
            site_id=site_id,
        )
    )


def _directory_eligible(
    session: Session,
    *,
    tenant_id: str,
    subject_id: str,
    role: str,
    asset_id: str,
    site_id: str | None,
) -> bool:
    subject = session.scalar(
        select(SubjectRecord).where(
            SubjectRecord.tenant_id == tenant_id,
            SubjectRecord.subject_id == subject_id,
            SubjectRecord.status == "active",
        )
    )
    if subject is None:
        return False
    assigned_role = session.scalar(
        select(SubjectRoleRecord).where(
            SubjectRoleRecord.tenant_id == tenant_id,
            SubjectRoleRecord.subject_id == subject_id,
            SubjectRoleRecord.role == role,
        )
    )
    if assigned_role is None:
        return False
    scope = session.scalar(
        select(AssetScopeRecord).where(
            AssetScopeRecord.tenant_id == tenant_id,
            or_(
                AssetScopeRecord.subject_id == subject_id,
                (AssetScopeRecord.subject_id.is_(None)) & (AssetScopeRecord.role == role),
            ),
            or_(
                AssetScopeRecord.asset_id == asset_id,
                (AssetScopeRecord.site_id == site_id) if site_id is not None else False,
            ),
        )
    )
    return scope is not None


def _load_collaboration(
    session: Session,
    tenant_id: str,
    collaboration_id: str,
    *,
    for_update: bool,
) -> WorkOrderExpertCollaborationRecord:
    statement = select(WorkOrderExpertCollaborationRecord).where(
        WorkOrderExpertCollaborationRecord.tenant_id == tenant_id,
        WorkOrderExpertCollaborationRecord.collaboration_id == collaboration_id,
    )
    if for_update:
        statement = statement.with_for_update()
    record = session.scalar(statement)
    if record is None:
        raise ExpertCollaborationNotVisible
    return record


def _current_collaboration(
    session: Session,
    tenant_id: str,
    work_order_id: str,
    *,
    for_update: bool,
) -> WorkOrderExpertCollaborationRecord | None:
    statement = (
        select(WorkOrderExpertCollaborationRecord)
        .where(
            WorkOrderExpertCollaborationRecord.tenant_id == tenant_id,
            WorkOrderExpertCollaborationRecord.work_order_id == work_order_id,
            WorkOrderExpertCollaborationRecord.status.in_(OPEN_COLLABORATION_STATUSES),
        )
        .order_by(WorkOrderExpertCollaborationRecord.created_at.desc())
        .limit(1)
    )
    if for_update:
        statement = statement.with_for_update()
    return session.scalar(statement)


def _latest_recommendation_collaboration(
    session: Session,
    tenant_id: str,
    work_order_id: str,
    *,
    for_update: bool,
) -> WorkOrderExpertCollaborationRecord | None:
    statement = (
        select(WorkOrderExpertCollaborationRecord)
        .where(
            WorkOrderExpertCollaborationRecord.tenant_id == tenant_id,
            WorkOrderExpertCollaborationRecord.work_order_id == work_order_id,
            WorkOrderExpertCollaborationRecord.collaboration_id.in_(
                select(
                    WorkOrderExpertCollaborationRecommendationRecord.collaboration_id
                ).where(
                    WorkOrderExpertCollaborationRecommendationRecord.tenant_id
                    == tenant_id,
                    WorkOrderExpertCollaborationRecommendationRecord.work_order_id
                    == work_order_id,
                )
            ),
        )
        .order_by(WorkOrderExpertCollaborationRecord.created_at.desc())
        .limit(1)
    )
    if for_update:
        statement = statement.with_for_update()
    return session.scalar(statement)


def _load_recommendation(
    session: Session,
    tenant_id: str,
    collaboration_id: str,
    *,
    for_update: bool,
    required: bool,
) -> WorkOrderExpertCollaborationRecommendationRecord | None:
    statement = select(WorkOrderExpertCollaborationRecommendationRecord).where(
        WorkOrderExpertCollaborationRecommendationRecord.tenant_id == tenant_id,
        WorkOrderExpertCollaborationRecommendationRecord.collaboration_id
        == collaboration_id,
    )
    if for_update:
        statement = statement.with_for_update()
    recommendation = session.scalar(statement)
    if recommendation is None and required:
        raise ExpertCollaborationNotVisible
    return recommendation


def _expire_if_needed(
    session: Session,
    record: WorkOrderExpertCollaborationRecord,
    *,
    now: datetime | None = None,
) -> None:
    current_time = now or datetime.now(UTC)
    if record.status not in OPEN_COLLABORATION_STATUSES or current_time < _aware(record.expires_at):
        return
    previous = record.status
    record.status = "EXPIRED"
    record.ended_reason = "collaboration_expired"
    record.ended_at = current_time
    record.version += 1
    record.updated_at = current_time
    _append_event(
        session,
        record,
        sequence=record.version,
        event_type="EXPIRED",
        actor_subject_id="system",
        from_status=previous,
        to_status="EXPIRED",
        reason_code="collaboration_expired",
        occurred_at=current_time,
    )
    _revoke_media(session, record.collaboration_id, "collaboration_expired")


def _revoke_media(session: Session, collaboration_id: str, reason: str) -> None:
    records = list(
        session.scalars(
            select(RealtimeMediaSessionRecord)
            .where(
                RealtimeMediaSessionRecord.expert_collaboration_id == collaboration_id,
                RealtimeMediaSessionRecord.status.in_({"ISSUED", "CONNECTED", "RECONNECTING"}),
            )
            .with_for_update()
        )
    )
    for media in records:
        media.status = "ENDED"
        media.ended_reason = reason
        media.media_token_digest = None
        media.version += 1


def _append_event(
    session: Session,
    record: WorkOrderExpertCollaborationRecord,
    *,
    sequence: int,
    event_type: str,
    actor_subject_id: str,
    from_status: str | None,
    to_status: str,
    reason_code: str,
    occurred_at: datetime,
) -> None:
    session.add(
        WorkOrderExpertCollaborationEventRecord(
            event_id=f"expert-collab-event-{uuid4().hex}",
            tenant_id=record.tenant_id,
            collaboration_id=record.collaboration_id,
            sequence=sequence,
            event_type=event_type,
            actor_subject_id=actor_subject_id,
            from_status=from_status,
            to_status=to_status,
            reason_code=reason_code,
            metadata_json={},
            occurred_at=occurred_at,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
    )


def _legal_actions(
    subject_id: str,
    record: WorkOrderExpertCollaborationRecord,
    recommendation: WorkOrderExpertCollaborationRecommendationRecord | None,
    *,
    binding_current: bool,
) -> tuple[str, ...]:
    actions: list[str] = []
    if record.status == "WAITING_EXPERT":
        if subject_id == record.invited_subject_id:
            actions.extend(("ACCEPT", "END"))
        elif subject_id == record.requested_by_subject_id:
            actions.append("END")
    if record.status == "ACTIVE" and subject_id in {
        record.requested_by_subject_id,
        record.invited_subject_id,
    }:
        actions.extend(("JOIN_AUDIO", "END"))
    if (
        subject_id == record.invited_subject_id
        and record.accepted_at is not None
        and recommendation is None
        and binding_current
    ):
        actions.append("SUBMIT_RECOMMENDATION")
    return tuple(actions)


def _recommendation_view(
    subject_id: str,
    collaboration: WorkOrderExpertCollaborationRecord,
    recommendation: WorkOrderExpertCollaborationRecommendationRecord,
    *,
    binding_current: bool,
) -> ExpertRecommendation:
    legal_actions: tuple[str, ...] = tuple()
    if (
        recommendation.status == "PENDING_FIELD_REVIEW"
        and subject_id == collaboration.requested_by_subject_id
        and binding_current
    ):
        legal_actions = ("ACCEPT", "REJECT")
    return ExpertRecommendation(
        recommendation.recommendation_id,
        recommendation.collaboration_id,
        recommendation.work_order_id,
        recommendation.work_order_version,
        recommendation.incident_id,
        recommendation.asset_id,
        recommendation.expert_subject_id,
        recommendation.recommendation_type,
        recommendation.summary,
        recommendation.basis,
        tuple(recommendation.recommended_checks_json),
        recommendation.safety_notice,
        tuple(recommendation.evidence_entry_ids_json),
        recommendation.status,
        recommendation.reviewed_by_subject_id,
        recommendation.review_reason,
        _optional_aware(recommendation.reviewed_at),
        recommendation.field_entry_id,
        recommendation.version,
        legal_actions,
    )


def _recommendation_field_payload(
    recommendation: WorkOrderExpertCollaborationRecommendationRecord,
) -> dict[str, object]:
    return {
        "collaboration_id": recommendation.collaboration_id,
        "recommendation_id": recommendation.recommendation_id,
        "expert_subject_id": recommendation.expert_subject_id,
        "reviewed_by_subject_id": recommendation.reviewed_by_subject_id,
        "recommendation_type": recommendation.recommendation_type,
        "summary": recommendation.summary,
        "basis": recommendation.basis,
        "recommended_checks": list(recommendation.recommended_checks_json),
        "safety_notice": recommendation.safety_notice,
        "evidence_entry_ids": list(recommendation.evidence_entry_ids_json),
        "review_reason": recommendation.review_reason,
    }


def _request_digest(
    work_order_id: str,
    work_order_version: int,
    requested_by_subject_id: str,
    invited_subject_id: str,
    reason: str,
) -> str:
    payload = json.dumps(
        {
            "work_order_id": work_order_id,
            "work_order_version": work_order_version,
            "requested_by_subject_id": requested_by_subject_id,
            "invited_subject_id": invited_subject_id,
            "reason": reason,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + sha256(payload.encode()).hexdigest()


def _recommendation_request_digest(
    *,
    collaboration_id: str,
    expert_subject_id: str,
    recommendation_type: str,
    summary: str,
    basis: str,
    recommended_checks: tuple[str, ...],
    safety_notice: str | None,
    evidence_entry_ids: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "collaboration_id": collaboration_id,
            "expert_subject_id": expert_subject_id,
            "recommendation_type": recommendation_type,
            "summary": summary,
            "basis": basis,
            "recommended_checks": recommended_checks,
            "safety_notice": safety_notice,
            "evidence_entry_ids": evidence_entry_ids,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + sha256(payload.encode()).hexdigest()


def _normalize_text(value: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise ExpertCollaborationConflict("expert_recommendation_text_invalid")
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _optional_aware(value: datetime | None) -> datetime | None:
    return _aware(value) if value is not None else None
