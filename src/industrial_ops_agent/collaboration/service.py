"""Tenant-scoped supplier Agent collaboration lifecycle and human review."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import false, or_, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.collaboration.a2a import (
    A2ATaskOutcome,
    SupplierA2AError,
    SupplierAgentClient,
)
from industrial_ops_agent.data_governance.dlp import POLICY_VERSION, PresidioDlpProcessor
from industrial_ops_agent.maintenance_planning.review_isolation import (
    lock_subject,
    record_exposure,
    resolve_source,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetRecord,
    AssetSiteLinkRecord,
    DiagnosisRunRecord,
    IncidentRecord,
    SupplierCollaborationEventRecord,
    SupplierCollaborationRecord,
)

CollaborationReviewDecision = Literal["ACCEPTED", "REJECTED"]
_REMOTE_PENDING = {"SUBMITTED", "WORKING", "INPUT_REQUIRED", "AUTH_REQUIRED"}


class CollaborationNotVisible(Exception):
    pass


class CollaborationConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class CollaborationEventView:
    sequence: int
    event_type: str
    actor_subject_id: str
    reason_code: str
    metadata: dict[str, Any]
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class SupplierCollaborationView:
    collaboration_id: str
    incident_id: str
    diagnosis_run_id: str
    asset_id: str
    agent_name: str
    agent_version: str
    skill_id: str
    status: str
    sharing_reason: str
    request_payload: dict[str, Any]
    request_payload_digest: str
    dlp_policy_version: str
    dlp_finding_count: int
    remote_task_id: str | None
    response_payload: dict[str, Any] | None
    response_digest: str | None
    guardrail_policy_version: str | None
    requested_by_subject_id: str
    reviewed_by_subject_id: str | None
    review_decision: str | None
    review_reason: str | None
    reviewed_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    events: tuple[CollaborationEventView, ...]
    legal_actions: tuple[str, ...]


class SupplierCollaborationService:
    """Create redacted requests, synchronize A2A tasks, and require human review."""

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        dlp: PresidioDlpProcessor | None,
        client: SupplierAgentClient | None,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._dlp = dlp
        self._client = client

    def list(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> tuple[SupplierCollaborationView, ...]:
        self._authorizer.require(
            identity,
            Action.READ_AGENT_COLLABORATION,
            ResourceContext(identity.tenant_id, resource_id="supplier-collaborations"),
            request_id=request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            query = (
                select(SupplierCollaborationRecord)
                .where(
                    SupplierCollaborationRecord.tenant_id == identity.tenant_id,
                    _scope_clause(identity),
                )
                .order_by(
                    SupplierCollaborationRecord.updated_at.desc(),
                    SupplierCollaborationRecord.collaboration_id,
                )
                .limit(limit)
            )
            if status is not None:
                query = query.where(SupplierCollaborationRecord.status == status)
            records = tuple(session.scalars(query))
            events = _events_by_collaboration(session, identity, records)
            return tuple(
                self._view(
                    session, identity, item, events.get(item.collaboration_id, ()), request_id
                )
                for item in records
            )

    def get(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        request_id: str,
    ) -> SupplierCollaborationView:
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _load(session, identity.tenant_id, collaboration_id)
            self._require_record(identity, Action.READ_AGENT_COLLABORATION, record, request_id)
            events = tuple(
                session.scalars(
                    select(SupplierCollaborationEventRecord)
                    .where(
                        SupplierCollaborationEventRecord.tenant_id == identity.tenant_id,
                        SupplierCollaborationEventRecord.collaboration_id == collaboration_id,
                    )
                    .order_by(SupplierCollaborationEventRecord.sequence)
                )
            )
            return self._view(
                session, identity, record, tuple(_event(item) for item in events), request_id
            )

    def create(
        self,
        identity: IdentityContext,
        *,
        incident_id: str,
        diagnosis_run_id: str,
        question: str,
        sharing_reason: str,
        idempotency_key: str,
        request_id: str,
    ) -> SupplierCollaborationView:
        request_fingerprint = _digest(
            {
                "incident_id": incident_id,
                "diagnosis_run_id": diagnosis_run_id,
                "question": question,
                "sharing_reason": sharing_reason,
            }
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            existing = session.scalar(
                select(SupplierCollaborationRecord).where(
                    SupplierCollaborationRecord.tenant_id == identity.tenant_id,
                    SupplierCollaborationRecord.requested_by_subject_id == identity.subject_id,
                    SupplierCollaborationRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                self._require_record(
                    identity,
                    Action.CREATE_AGENT_COLLABORATION,
                    existing,
                    request_id,
                )
                if existing.request_fingerprint != request_fingerprint:
                    raise CollaborationConflict("collaboration_idempotency_conflict")
                return self._view(session, identity, existing, (), request_id)
            row = (
                session.execute(
                    select(IncidentRecord, DiagnosisRunRecord, AssetRecord)
                    .join(
                        DiagnosisRunRecord,
                        DiagnosisRunRecord.incident_id == IncidentRecord.incident_id,
                    )
                    .join(AssetRecord, AssetRecord.asset_id == IncidentRecord.asset_id)
                    .where(
                        IncidentRecord.tenant_id == identity.tenant_id,
                        DiagnosisRunRecord.tenant_id == identity.tenant_id,
                        AssetRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.incident_id == incident_id,
                        DiagnosisRunRecord.diagnosis_run_id == diagnosis_run_id,
                    )
                )
                .tuples()
                .one_or_none()
            )
            if row is None:
                raise CollaborationNotVisible
            incident, diagnosis, asset = row
            self._authorizer.require(
                identity,
                Action.CREATE_AGENT_COLLABORATION,
                ResourceContext(identity.tenant_id, incident_id, incident.asset_id),
                request_id=request_id,
            )
            if self._client is None:
                raise CollaborationConflict("supplier_a2a_not_configured")
            if diagnosis.status not in {"COMPLETED", "NEEDS_INFORMATION"} or not isinstance(
                diagnosis.report, dict
            ):
                raise CollaborationConflict("diagnosis_not_ready_for_collaboration")
            collaboration_id = f"a2a-collaboration-{uuid4().hex}"
            payload, finding_count = self._request_payload(
                collaboration_id,
                incident,
                diagnosis,
                asset,
                question,
            )
            record = SupplierCollaborationRecord(
                collaboration_id=collaboration_id,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                tenant_id=identity.tenant_id,
                incident_id=incident_id,
                diagnosis_run_id=diagnosis_run_id,
                asset_id=incident.asset_id,
                agent_name=self._client.agent_name,
                agent_version=self._client.agent_version,
                skill_id=self._client.skill_id,
                status="READY",
                sharing_reason=sharing_reason,
                request_payload=payload,
                request_payload_digest=_digest(payload),
                dlp_policy_version=POLICY_VERSION,
                dlp_finding_count=finding_count,
                remote_task_id=None,
                remote_context_id=None,
                response_payload=None,
                response_digest=None,
                guardrail_policy_version=None,
                requested_by_subject_id=identity.subject_id,
                reviewed_by_subject_id=None,
                review_decision=None,
                review_reason=None,
                reviewed_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            # No ORM relationship joins the append-only event mapper to its parent.
            # Persist the collaboration first so PostgreSQL cannot schedule the
            # event INSERT ahead of the referenced row in the same unit of work.
            session.flush()
            event = _new_event(
                record,
                identity.subject_id,
                event_type="collaboration.created",
                reason_code="redacted_evidence_package_ready",
                metadata={
                    "request_payload_digest": record.request_payload_digest,
                    "dlp_policy_version": POLICY_VERSION,
                    "dlp_finding_count": finding_count,
                    "agent_name": record.agent_name,
                    "agent_version": record.agent_version,
                    "skill_id": record.skill_id,
                },
                occurred_at=now,
            )
            session.add(event)
            return self._view(session, identity, record, (_event(event),), request_id)

    async def dispatch(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> SupplierCollaborationView:
        payload = await asyncio.to_thread(
            self._start_command,
            identity,
            collaboration_id,
            action=Action.DISPATCH_AGENT_COLLABORATION,
            expected_version=expected_version,
            allowed_statuses={"READY", "DELIVERY_FAILED"},
            command_status="DISPATCHING",
            event_type="collaboration.dispatch_started",
            reason_code="supplier_agent_dispatch_started",
            request_id=request_id,
        )
        client = self._require_client()
        try:
            outcome = await client.submit(payload)
        except SupplierA2AError as exc:
            await asyncio.to_thread(
                self._record_delivery_failure, identity, collaboration_id, exc.reason
            )
            raise
        return await asyncio.to_thread(
            self._apply_outcome,
            identity,
            collaboration_id,
            outcome,
            event_type="collaboration.dispatched",
            actor_subject_id=identity.subject_id,
            request_id=request_id,
        )

    async def refresh(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> SupplierCollaborationView:
        remote_task_id, previous_status = await asyncio.to_thread(
            self._start_remote_command,
            identity,
            collaboration_id,
            action=Action.DISPATCH_AGENT_COLLABORATION,
            expected_version=expected_version,
            allowed_statuses=_REMOTE_PENDING,
            command_status="REFRESHING",
            event_type="collaboration.refresh_started",
            request_id=request_id,
        )
        client = self._require_client()
        try:
            outcome = await client.refresh(remote_task_id)
        except SupplierA2AError as exc:
            await asyncio.to_thread(
                self._record_sync_failure,
                identity,
                collaboration_id,
                exc.reason,
                previous_status=previous_status,
            )
            raise
        try:
            return await asyncio.to_thread(
                self._apply_outcome,
                identity,
                collaboration_id,
                outcome,
                event_type="collaboration.refreshed",
                actor_subject_id=identity.subject_id,
                request_id=request_id,
            )
        except CollaborationConflict as exc:
            await asyncio.to_thread(
                self._record_sync_failure,
                identity,
                collaboration_id,
                exc.reason,
                previous_status=previous_status,
            )
            raise

    async def cancel(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> SupplierCollaborationView:
        remote_task_id, previous_status = await asyncio.to_thread(
            self._start_remote_command,
            identity,
            collaboration_id,
            action=Action.DISPATCH_AGENT_COLLABORATION,
            expected_version=expected_version,
            allowed_statuses=_REMOTE_PENDING,
            command_status="CANCELING",
            event_type="collaboration.cancel_started",
            request_id=request_id,
        )
        client = self._require_client()
        try:
            outcome = await client.cancel(remote_task_id)
        except SupplierA2AError as exc:
            await asyncio.to_thread(
                self._record_sync_failure,
                identity,
                collaboration_id,
                exc.reason,
                previous_status=previous_status,
            )
            raise
        try:
            return await asyncio.to_thread(
                self._apply_outcome,
                identity,
                collaboration_id,
                outcome,
                event_type="collaboration.cancel_requested",
                actor_subject_id=identity.subject_id,
                request_id=request_id,
            )
        except CollaborationConflict as exc:
            await asyncio.to_thread(
                self._record_sync_failure,
                identity,
                collaboration_id,
                exc.reason,
                previous_status=previous_status,
            )
            raise

    def review(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        decision: CollaborationReviewDecision,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> SupplierCollaborationView:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _load(session, identity.tenant_id, collaboration_id)
            self._require_record(identity, Action.REVIEW_AGENT_COLLABORATION, record, request_id)
            _require_version(record, expected_version)
            if record.status != "COMPLETED" or record.response_payload is None:
                raise CollaborationConflict("collaboration_not_ready_for_review", record.version)
            if record.requested_by_subject_id == identity.subject_id:
                raise CollaborationConflict(
                    "collaboration_review_separation_required", record.version
                )
            record.status = decision
            record.review_decision = decision
            record.review_reason = reason
            record.reviewed_by_subject_id = identity.subject_id
            record.reviewed_at = now
            record.version += 1
            record.updated_at = now
            event = _new_event(
                record,
                identity.subject_id,
                event_type="collaboration.reviewed",
                reason_code=(
                    "supplier_advice_accepted"
                    if decision == "ACCEPTED"
                    else "supplier_advice_rejected"
                ),
                metadata={
                    "decision": decision,
                    "response_digest": record.response_digest,
                },
                occurred_at=now,
            )
            session.add(event)
            return self._view(session, identity, record, (_event(event),), request_id)

    def _request_payload(
        self,
        collaboration_id: str,
        incident: IncidentRecord,
        diagnosis: DiagnosisRunRecord,
        asset: AssetRecord,
        question: str,
    ) -> tuple[dict[str, Any], int]:
        if self._dlp is None:
            raise CollaborationConflict("collaboration_dlp_not_configured")
        report = diagnosis.report or {}
        fields = {
            "incident_description": incident.description,
            "question": question,
            "conclusion": _text(report.get("conclusion")),
            "contradictions": _joined(report.get("contradictions")),
            "missing_information": _joined(report.get("missing_information")),
            "next_checks": _joined(report.get("next_checks")),
        }
        result = self._dlp.process(fields)
        if result.status != "PASSED":
            raise CollaborationConflict("collaboration_dlp_failed")
        payload = {
            "contract_version": "supplier-diagnosis-request/v1",
            "collaboration_id": collaboration_id,
            "skill_id": self._client.skill_id if self._client is not None else "",
            "asset": {"model_code": asset.model_code or "UNKNOWN"},
            "incident": {
                "severity": incident.severity or "UNCLASSIFIED",
                "category": incident.category or "UNCLASSIFIED",
                "description": result.redacted_content["incident_description"],
            },
            "question": result.redacted_content["question"],
            "diagnosis": {
                "status": diagnosis.status,
                "conclusion": result.redacted_content["conclusion"],
                "contradictions": _split(result.redacted_content["contradictions"]),
                "missing_information": _split(result.redacted_content["missing_information"]),
                "next_checks": _split(result.redacted_content["next_checks"]),
                "confidence": _confidence(report.get("confidence")),
                "manifest_digest": _digest(diagnosis.manifest),
                "report_digest": _digest(report),
            },
            "constraints": {
                "advisory_only": True,
                "may_execute_tools": False,
                "may_change_work_order": False,
                "may_control_equipment": False,
            },
        }
        return payload, len(result.findings)

    def _start_command(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        action: Action,
        expected_version: int,
        allowed_statuses: set[str],
        command_status: str,
        event_type: str,
        reason_code: str,
        request_id: str,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _load(session, identity.tenant_id, collaboration_id)
            self._require_record(identity, action, record, request_id)
            self._record_disclosure(session, identity, record, request_id)
            self._require_client()
            _require_version(record, expected_version)
            if record.status not in allowed_statuses:
                raise CollaborationConflict("collaboration_transition_not_allowed", record.version)
            record.status = command_status
            record.version += 1
            record.updated_at = now
            session.add(
                _new_event(
                    record,
                    identity.subject_id,
                    event_type=event_type,
                    reason_code=reason_code,
                    metadata={"request_payload_digest": record.request_payload_digest},
                    occurred_at=now,
                )
            )
            return dict(record.request_payload)

    def _start_remote_command(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        action: Action,
        expected_version: int,
        allowed_statuses: set[str],
        command_status: str,
        event_type: str,
        request_id: str,
    ) -> tuple[str, str]:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _load(session, identity.tenant_id, collaboration_id)
            self._require_record(identity, action, record, request_id)
            self._record_disclosure(session, identity, record, request_id)
            self._require_client()
            _require_version(record, expected_version)
            if record.status not in allowed_statuses or record.remote_task_id is None:
                raise CollaborationConflict("collaboration_remote_task_not_ready", record.version)
            previous_status = record.status
            record.status = command_status
            record.version += 1
            record.updated_at = now
            session.add(
                _new_event(
                    record,
                    identity.subject_id,
                    event_type=event_type,
                    reason_code=f"supplier_agent_{command_status.lower()}",
                    metadata={"remote_task_id": record.remote_task_id},
                    occurred_at=now,
                )
            )
            return record.remote_task_id, previous_status

    def _record_delivery_failure(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        reason: str,
    ) -> None:
        self._failure(identity, collaboration_id, reason, status="DELIVERY_FAILED")

    def _record_sync_failure(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        reason: str,
        *,
        previous_status: str,
    ) -> None:
        self._failure(identity, collaboration_id, reason, status=previous_status)

    def _failure(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        reason: str,
        *,
        status: str | None,
    ) -> None:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = _load(session, identity.tenant_id, collaboration_id)
            if status is not None:
                record.status = status
            record.version += 1
            record.updated_at = now
            session.add(
                _new_event(
                    record,
                    identity.subject_id,
                    event_type="collaboration.remote_failure",
                    reason_code=reason,
                    metadata={},
                    occurred_at=now,
                )
            )

    def _apply_outcome(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        outcome: A2ATaskOutcome,
        *,
        event_type: str,
        actor_subject_id: str,
        request_id: str,
    ) -> SupplierCollaborationView:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _load(session, identity.tenant_id, collaboration_id)
            if (
                record.remote_task_id is not None
                and record.remote_task_id != outcome.remote_task_id
            ):
                raise CollaborationConflict("collaboration_remote_task_identity_changed")
            record.remote_task_id = outcome.remote_task_id
            record.remote_context_id = outcome.remote_context_id
            record.status = outcome.status
            record.response_payload = outcome.advice
            record.response_digest = outcome.response_digest
            record.guardrail_policy_version = outcome.guardrail_policy_version
            record.version += 1
            record.updated_at = now
            event = _new_event(
                record,
                actor_subject_id,
                event_type=event_type,
                reason_code=f"supplier_task_{outcome.status.lower()}",
                metadata={
                    "remote_task_id": outcome.remote_task_id,
                    "status": outcome.status,
                    "response_digest": outcome.response_digest,
                    "guardrail_policy_version": outcome.guardrail_policy_version,
                },
                occurred_at=now,
            )
            session.add(event)
            return self._view(session, identity, record, (_event(event),), request_id)

    def _require_client(self) -> SupplierAgentClient:
        if self._client is None:
            raise CollaborationConflict("supplier_a2a_not_configured")
        return self._client

    def _require_record(
        self,
        identity: IdentityContext,
        action: Action,
        record: SupplierCollaborationRecord,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(
                identity.tenant_id,
                resource_id=record.collaboration_id,
                asset_id=record.asset_id,
            ),
            request_id=request_id,
        )

    @staticmethod
    def _record_disclosure(
        session: Session,
        identity: IdentityContext,
        record: SupplierCollaborationRecord,
        request_id: str,
    ) -> None:
        source = resolve_source(session, identity.tenant_id, record.diagnosis_run_id)
        if source.incident_id != record.incident_id or source.asset_id != record.asset_id:
            raise CollaborationConflict("collaboration_source_binding_changed")
        record_exposure(
            session,
            identity,
            source,
            reason="IDENTIFIED_READ",
            resource_kind="supplier-collaboration",
            resource_id=record.collaboration_id,
            request_id=request_id,
        )

    def _view(
        self,
        session: Session,
        identity: IdentityContext,
        record: SupplierCollaborationRecord,
        events: tuple[CollaborationEventView, ...],
        request_id: str,
    ) -> SupplierCollaborationView:
        self._record_disclosure(session, identity, record, request_id)
        return SupplierCollaborationView(
            collaboration_id=record.collaboration_id,
            incident_id=record.incident_id,
            diagnosis_run_id=record.diagnosis_run_id,
            asset_id=record.asset_id,
            agent_name=record.agent_name,
            agent_version=record.agent_version,
            skill_id=record.skill_id,
            status=record.status,
            sharing_reason=record.sharing_reason,
            request_payload=dict(record.request_payload),
            request_payload_digest=record.request_payload_digest,
            dlp_policy_version=record.dlp_policy_version,
            dlp_finding_count=record.dlp_finding_count,
            remote_task_id=record.remote_task_id,
            response_payload=(
                dict(record.response_payload) if record.response_payload is not None else None
            ),
            response_digest=record.response_digest,
            guardrail_policy_version=record.guardrail_policy_version,
            requested_by_subject_id=record.requested_by_subject_id,
            reviewed_by_subject_id=record.reviewed_by_subject_id,
            review_decision=record.review_decision,
            review_reason=record.review_reason,
            reviewed_at=record.reviewed_at,
            version=record.version,
            created_at=record.created_at,
            updated_at=record.updated_at,
            events=events,
            legal_actions=self._legal_actions(identity, record),
        )

    def _legal_actions(
        self,
        identity: IdentityContext,
        record: SupplierCollaborationRecord,
    ) -> tuple[str, ...]:
        resource = ResourceContext(
            identity.tenant_id,
            resource_id=record.collaboration_id,
            asset_id=record.asset_id,
        )
        actions: list[str] = []
        if (
            record.status in {"READY", "DELIVERY_FAILED"}
            and self._authorizer.decide(
                identity, Action.DISPATCH_AGENT_COLLABORATION, resource
            ).allowed
        ):
            actions.append("DISPATCH")
        if (
            record.status in _REMOTE_PENDING
            and self._authorizer.decide(
                identity, Action.DISPATCH_AGENT_COLLABORATION, resource
            ).allowed
        ):
            actions.extend(("REFRESH", "CANCEL"))
        if (
            record.status == "COMPLETED"
            and record.requested_by_subject_id != identity.subject_id
            and self._authorizer.decide(
                identity, Action.REVIEW_AGENT_COLLABORATION, resource
            ).allowed
        ):
            actions.append("REVIEW")
        return tuple(actions)


def _load(
    session: Session,
    tenant_id: str,
    collaboration_id: str,
) -> SupplierCollaborationRecord:
    record = session.scalar(
        select(SupplierCollaborationRecord).where(
            SupplierCollaborationRecord.tenant_id == tenant_id,
            SupplierCollaborationRecord.collaboration_id == collaboration_id,
        )
    )
    if record is None:
        raise CollaborationNotVisible
    return record


def _scope_clause(identity: IdentityContext) -> Any:
    clauses: list[Any] = []
    if identity.asset_ids:
        clauses.append(SupplierCollaborationRecord.asset_id.in_(identity.asset_ids))
    if identity.site_ids:
        clauses.append(
            SupplierCollaborationRecord.asset_id.in_(
                select(AssetSiteLinkRecord.asset_id).where(
                    AssetSiteLinkRecord.tenant_id == identity.tenant_id,
                    AssetSiteLinkRecord.site_id.in_(identity.site_ids),
                )
            )
        )
    return or_(*clauses) if clauses else false()


def _events_by_collaboration(
    session: Session,
    identity: IdentityContext,
    records: tuple[SupplierCollaborationRecord, ...],
) -> dict[str, tuple[CollaborationEventView, ...]]:
    ids = tuple(item.collaboration_id for item in records)
    if not ids:
        return {}
    rows = tuple(
        session.scalars(
            select(SupplierCollaborationEventRecord)
            .where(
                SupplierCollaborationEventRecord.tenant_id == identity.tenant_id,
                SupplierCollaborationEventRecord.collaboration_id.in_(ids),
            )
            .order_by(
                SupplierCollaborationEventRecord.collaboration_id,
                SupplierCollaborationEventRecord.sequence,
            )
        )
    )
    result: dict[str, list[CollaborationEventView]] = {}
    for row in rows:
        result.setdefault(row.collaboration_id, []).append(_event(row))
    return {key: tuple(value) for key, value in result.items()}


def _new_event(
    record: SupplierCollaborationRecord,
    actor_subject_id: str,
    *,
    event_type: str,
    reason_code: str,
    metadata: dict[str, Any],
    occurred_at: datetime,
) -> SupplierCollaborationEventRecord:
    return SupplierCollaborationEventRecord(
        event_id=f"a2a-event-{uuid4().hex}",
        tenant_id=record.tenant_id,
        collaboration_id=record.collaboration_id,
        sequence=record.version,
        event_type=event_type,
        actor_subject_id=actor_subject_id,
        reason_code=reason_code,
        metadata_json=metadata,
        occurred_at=occurred_at,
        created_at=occurred_at,
        updated_at=occurred_at,
    )


def _event(record: SupplierCollaborationEventRecord) -> CollaborationEventView:
    return CollaborationEventView(
        sequence=record.sequence,
        event_type=record.event_type,
        actor_subject_id=record.actor_subject_id,
        reason_code=record.reason_code,
        metadata=dict(record.metadata_json),
        occurred_at=record.occurred_at,
    )


def _require_version(record: SupplierCollaborationRecord, expected: int) -> None:
    if record.version != expected:
        raise CollaborationConflict("collaboration_version_conflict", record.version)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _joined(value: object) -> str:
    if not isinstance(value, list):
        return ""
    return "\n".join(item for item in value if isinstance(item, str))


def _split(value: str) -> list[str]:
    return [item for item in value.splitlines() if item]


def _confidence(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(1.0, max(0.0, float(value)))
    return 0.0


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{sha256(encoded.encode()).hexdigest()}"
