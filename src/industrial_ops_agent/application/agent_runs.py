"""Optimistic, idempotent cancellation and human-confirmed Agent resumption."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.application.diagnoses import (
    DiagnosisDispatcher,
    DiagnosisStartResult,
    DiagnosisWorkflowInput,
    _run,
)
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AgentEventRecord,
    AgentRunRecord,
    DiagnosisRunRecord,
    IdempotencyRecord,
    IncidentRecord,
)
from industrial_ops_agent.persistence.repositories import (
    AgentRunRepository,
    DiagnosisRunRepository,
    EvidenceBundleRepository,
    IdempotencyRepository,
    IncidentRepository,
)


class AgentRunVersionConflict(Exception):
    def __init__(self, current_version: int) -> None:
        super().__init__("agent run version conflict")
        self.current_version = current_version


class AgentRunPreconditionFailed(Exception):
    def __init__(self, reason_code: str) -> None:
        super().__init__("agent run precondition failed")
        self.reason_code = reason_code


class AgentRunCancellationDispatchFailed(Exception):
    """Temporal rejected cancellation while the persisted request remains retryable."""


class AgentRunResumeDispatchFailed(Exception):
    """Temporal did not accept a newly queued successor run."""


@dataclass(frozen=True, slots=True)
class AgentRunControlView:
    agent_run_id: str
    diagnosis_run_id: str
    status: str
    version: int
    last_sequence: int
    created: bool


class AgentRunControlService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        dispatcher: DiagnosisDispatcher,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._dispatcher = dispatcher

    async def cancel(
        self,
        identity: IdentityContext,
        agent_run_id: str,
        *,
        reason: str,
        expected_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> AgentRunControlView:
        normalized_reason = reason.strip()
        if not 8 <= len(normalized_reason) <= 1_024:
            raise AgentRunPreconditionFailed("cancel_reason_invalid")
        request_hash = _canonical_hash(
            {
                "agent_run_id": agent_run_id,
                "expected_version": expected_version,
                "reason": normalized_reason,
            }
        )
        storage_key = f"agent-cancel:{idempotency_key}"
        created = False
        workflow_id: str
        with self._database.transaction(identity.tenant_context) as session:
            agent, diagnosis, _incident = self._authorized_records(
                session,
                identity,
                agent_run_id,
                request_id=request_id,
            )
            workflow_id = diagnosis.workflow_id
            idempotency = IdempotencyRepository(session, identity.tenant_context)
            existing = idempotency.get(subject_id=identity.subject_id, key=storage_key)
            if existing is not None:
                if existing.request_hash != request_hash or existing.result_ref != agent_run_id:
                    raise AgentRunPreconditionFailed("idempotency_conflict")
                if agent.status == "CANCELLED":
                    return _control_view(agent, created=False)
                if agent.status != "CANCEL_REQUESTED":
                    raise AgentRunPreconditionFailed("agent_cancel_not_pending")
            else:
                if agent.version != expected_version:
                    raise AgentRunVersionConflict(agent.version)
                if agent.status not in {"QUEUED", "RUNNING"}:
                    raise AgentRunPreconditionFailed("agent_run_not_cancellable")
                now = datetime.now(UTC)
                _append_event(
                    session,
                    identity.tenant_id,
                    agent,
                    event_type="diagnosis.cancel_requested",
                    payload={
                        "diagnosis_run_id": diagnosis.diagnosis_run_id,
                        "reason": normalized_reason,
                        "requested_by_subject_id": identity.subject_id,
                    },
                    now=now,
                    status="CANCEL_REQUESTED",
                )
                diagnosis.status = "CANCEL_REQUESTED"
                diagnosis.stop_reason = "cancel_requested_by_user"
                diagnosis.version += 1
                diagnosis.updated_at = now
                idempotency.add(
                    IdempotencyRecord(
                        record_id=f"idem-{uuid4().hex}",
                        tenant_id=identity.tenant_id,
                        subject_id=identity.subject_id,
                        key=storage_key,
                        request_hash=request_hash,
                        result_ref=agent_run_id,
                        expires_at=now + timedelta(hours=24),
                        created_at=now,
                        updated_at=now,
                    )
                )
                created = True

        try:
            await self._dispatcher.cancel(workflow_id)
        except Exception as exc:
            raise AgentRunCancellationDispatchFailed from exc

        with self._database.transaction(identity.tenant_context) as session:
            final_agent = AgentRunRepository(session, identity.tenant_context).get(agent_run_id)
            if final_agent is None:
                raise ResourceNotVisible
            final_diagnosis = DiagnosisRunRepository(session, identity.tenant_context).get(
                final_agent.diagnosis_run_id
            )
            if final_diagnosis is None:
                raise ResourceNotVisible
            if final_agent.status == "CANCELLED":
                return _control_view(final_agent, created=created)
            if final_agent.status != "CANCEL_REQUESTED":
                raise AgentRunPreconditionFailed("agent_cancel_raced_with_terminal")
            now = datetime.now(UTC)
            _append_event(
                session,
                identity.tenant_id,
                final_agent,
                event_type="diagnosis.cancelled",
                payload={
                    "diagnosis_run_id": final_diagnosis.diagnosis_run_id,
                    "stop_reason": "cancelled_by_user",
                },
                now=now,
                status="CANCELLED",
            )
            final_diagnosis.status = "CANCELLED"
            final_diagnosis.stop_reason = "cancelled_by_user"
            final_diagnosis.version += 1
            final_diagnosis.updated_at = now
            return _control_view(final_agent, created=created)

    async def resume(
        self,
        identity: IdentityContext,
        agent_run_id: str,
        *,
        clarification: str,
        expected_version: int,
        idempotency_key: str,
        request_id: str,
        requested_agent_run_id: str | None = None,
        interrupt_id: str | None = None,
        source_message_id: str | None = None,
        expected_incident_id: str | None = None,
    ) -> DiagnosisStartResult:
        normalized = clarification.strip()
        if not 2 <= len(normalized) <= 4_000:
            raise AgentRunPreconditionFailed("clarification_invalid")
        request_facts: dict[str, object] = {
            "agent_run_id": agent_run_id,
            "expected_version": expected_version,
            "clarification_sha256": sha256(normalized.encode()).hexdigest(),
        }
        if requested_agent_run_id is not None:
            request_facts["requested_agent_run_id"] = requested_agent_run_id
        if interrupt_id is not None:
            request_facts["interrupt_id"] = interrupt_id
        if source_message_id is not None:
            request_facts["source_message_id"] = source_message_id
        request_hash = _canonical_hash(request_facts)
        storage_key = f"agent-resume:{idempotency_key}"
        command: DiagnosisWorkflowInput | None = None
        with self._database.transaction(identity.tenant_context) as session:
            source_agent, source, incident = self._authorized_records(
                session,
                identity,
                agent_run_id,
                request_id=request_id,
            )
            if expected_incident_id is not None and incident.incident_id != expected_incident_id:
                raise AgentRunPreconditionFailed("ag_ui_thread_mismatch")
            idempotency = IdempotencyRepository(session, identity.tenant_context)
            existing = idempotency.get(subject_id=identity.subject_id, key=storage_key)
            diagnoses = DiagnosisRunRepository(session, identity.tenant_context)
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise AgentRunPreconditionFailed("idempotency_conflict")
                replay = diagnoses.get(existing.result_ref)
                if replay is None:
                    raise RuntimeError("agent resume idempotency references a missing diagnosis")
                if (
                    requested_agent_run_id is not None
                    and replay.agent_run_id != requested_agent_run_id
                ):
                    raise AgentRunPreconditionFailed("idempotency_conflict")
                input_context = replay.manifest.get("input_context")
                confirmed_input_event_id = (
                    input_context.get("source_agent_event_id")
                    if isinstance(input_context, dict)
                    else None
                )
                if not isinstance(confirmed_input_event_id, str):
                    raise RuntimeError("agent resume input binding is missing")
                command = _workflow_command(
                    identity,
                    replay,
                    request_id=request_id,
                    source_diagnosis_run_id=source.diagnosis_run_id,
                    confirmed_input_event_id=confirmed_input_event_id,
                )
                try:
                    await self._dispatcher.dispatch(command)
                except Exception as exc:
                    raise AgentRunResumeDispatchFailed from exc
                return DiagnosisStartResult(_run(replay), created=False)
            if source_agent.version != expected_version:
                raise AgentRunVersionConflict(source_agent.version)
            if (
                source_agent.status != "NEEDS_INFORMATION"
                or source.status != "NEEDS_INFORMATION"
            ):
                raise AgentRunPreconditionFailed("agent_run_not_resumable")
            open_interrupt = _open_input_interrupt(
                session,
                identity.tenant_id,
                source_agent,
            )
            if open_interrupt is None:
                raise AgentRunPreconditionFailed("agent_input_interrupt_already_resolved")
            if interrupt_id is not None and interrupt_id != open_interrupt.event_id:
                raise AgentRunPreconditionFailed("ag_ui_interrupt_not_found")
            evidence = EvidenceBundleRepository(session, identity.tenant_context).get(
                source.evidence_bundle_id
            )
            if evidence is None or evidence.status != "CONFIRMED":
                raise AgentRunPreconditionFailed("evidence_not_confirmed")
            if (
                not evidence.automation_eligible
                or evidence.processor_versions.get("vlm_status") == "unsupported_model"
            ):
                raise AgentRunPreconditionFailed("evidence_not_automation_eligible")

            now = datetime.now(UTC)
            confirmed_event = _append_event(
                session,
                identity.tenant_id,
                source_agent,
                event_type="agent.user_input.confirmed",
                payload={
                    "diagnosis_run_id": source.diagnosis_run_id,
                    "source_type": "manual_clarification",
                    "source_id": source_message_id or f"agent-resume-input-{uuid4().hex}",
                    "interrupt_id": open_interrupt.event_id,
                    "text": normalized,
                    "reviewer_subject_id": identity.subject_id,
                },
                now=now,
            )
            diagnosis_run_id = f"diagnosis-{uuid4().hex}"
            successor_agent_run_id = requested_agent_run_id or f"agent-{uuid4().hex}"
            workflow_id = f"m2-diagnosis-resume/{identity.tenant_id}/{diagnosis_run_id}"
            manifest = deepcopy(source.manifest)
            manifest["input_context"] = {
                "kind": "human_confirmed_clarification",
                "source_diagnosis_run_id": source.diagnosis_run_id,
                "source_agent_run_id": source.agent_run_id,
                "source_agent_event_id": confirmed_event.event_id,
                "source_agent_event_sequence": confirmed_event.sequence,
                "source_interrupt_id": open_interrupt.event_id,
                "confirmed_text_sha256": sha256(normalized.encode()).hexdigest(),
            }
            run_record = DiagnosisRunRecord(
                diagnosis_run_id=diagnosis_run_id,
                tenant_id=identity.tenant_id,
                incident_id=source.incident_id,
                evidence_bundle_id=source.evidence_bundle_id,
                agent_run_id=successor_agent_run_id,
                workflow_id=workflow_id,
                status="QUEUED",
                manifest=manifest,
                report=None,
                stop_reason=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            diagnoses.add(run_record)
            session.flush()
            successor = AgentRunRecord(
                agent_run_id=successor_agent_run_id,
                tenant_id=identity.tenant_id,
                diagnosis_run_id=diagnosis_run_id,
                status="QUEUED",
                budget=dict(source_agent.budget),
                last_sequence=0,
                version=0,
                created_at=now,
                updated_at=now,
            )
            AgentRunRepository(session, identity.tenant_context).add(successor)
            session.flush()
            _append_event(
                session,
                identity.tenant_id,
                successor,
                event_type="diagnosis.resume_queued",
                payload={
                    "diagnosis_run_id": diagnosis_run_id,
                    "source_diagnosis_run_id": source.diagnosis_run_id,
                    "parent_agent_run_id": source.agent_run_id,
                    "confirmed_input_event_id": confirmed_event.event_id,
                    "workflow_id": workflow_id,
                },
                now=now,
                status="QUEUED",
            )
            idempotency.add(
                IdempotencyRecord(
                    record_id=f"idem-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=diagnosis_run_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            command = _workflow_command(
                identity,
                run_record,
                request_id=request_id,
                source_diagnosis_run_id=source.diagnosis_run_id,
                confirmed_input_event_id=confirmed_event.event_id,
            )
            result = DiagnosisStartResult(_run(run_record), created=True)
        if command is None:
            raise RuntimeError("agent resume did not create a workflow command")
        try:
            await self._dispatcher.dispatch(command)
        except Exception as exc:
            raise AgentRunResumeDispatchFailed from exc
        return result

    def legal_actions(
        self,
        identity: IdentityContext,
        agent_run_id: str,
    ) -> tuple[list[str], int]:
        with self._database.transaction(identity.tenant_context) as session:
            agent = AgentRunRepository(session, identity.tenant_context).get(agent_run_id)
            if agent is None:
                raise ResourceNotVisible
            diagnosis = DiagnosisRunRepository(session, identity.tenant_context).get(
                agent.diagnosis_run_id
            )
            if diagnosis is None:
                raise ResourceNotVisible
            incident = IncidentRepository(session, identity.tenant_context).get(
                diagnosis.incident_id
            )
            if incident is None:
                raise ResourceNotVisible
            allowed = self._authorizer.decide(
                identity,
                Action.CONTROL_AGENT,
                ResourceContext(
                    tenant_id=identity.tenant_id,
                    resource_id=agent_run_id,
                    asset_id=incident.asset_id,
                ),
            ).allowed
            actions: list[str] = []
            if allowed and agent.status in {"QUEUED", "RUNNING"}:
                actions.append("CANCEL_AGENT")
            if (
                allowed
                and agent.status == "NEEDS_INFORMATION"
                and _open_input_interrupt(session, identity.tenant_id, agent) is not None
            ):
                actions.append("RESUME_AGENT")
            return actions, agent.version

    def _authorized_records(
        self,
        session: Session,
        identity: IdentityContext,
        agent_run_id: str,
        *,
        request_id: str,
    ) -> tuple[AgentRunRecord, DiagnosisRunRecord, IncidentRecord]:
        agent = AgentRunRepository(session, identity.tenant_context).get(agent_run_id)
        if agent is None:
            raise ResourceNotVisible
        diagnosis = DiagnosisRunRepository(session, identity.tenant_context).get(
            agent.diagnosis_run_id
        )
        if diagnosis is None:
            raise ResourceNotVisible
        incident = IncidentRepository(session, identity.tenant_context).get(diagnosis.incident_id)
        if incident is None:
            raise ResourceNotVisible
        try:
            self._authorizer.require(
                identity,
                Action.CONTROL_AGENT,
                ResourceContext(
                    tenant_id=identity.tenant_id,
                    resource_id=agent_run_id,
                    asset_id=incident.asset_id,
                ),
                request_id=request_id,
            )
        except AuthorizationDenied as exc:
            raise ResourceNotVisible from exc
        return agent, diagnosis, incident


def _append_event(
    session: Session,
    tenant_id: str,
    agent: AgentRunRecord,
    *,
    event_type: str,
    payload: dict[str, object],
    now: datetime,
    status: str | None = None,
) -> AgentEventRecord:
    sequence = agent.last_sequence + 1
    record = AgentEventRecord(
        event_id=f"agent-event-{uuid4().hex}",
        tenant_id=tenant_id,
        agent_run_id=agent.agent_run_id,
        sequence=sequence,
        event_type=event_type,
        payload=payload,
        visibility="authorized",
        occurred_at=now,
        created_at=now,
        updated_at=now,
    )
    session.add(record)
    agent.last_sequence = sequence
    agent.version += 1
    agent.updated_at = now
    if status is not None:
        agent.status = status
    return record


def _control_view(agent: AgentRunRecord, *, created: bool) -> AgentRunControlView:
    return AgentRunControlView(
        agent_run_id=agent.agent_run_id,
        diagnosis_run_id=agent.diagnosis_run_id,
        status=agent.status,
        version=agent.version,
        last_sequence=agent.last_sequence,
        created=created,
    )


def _canonical_hash(value: dict[str, object]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(canonical.encode()).hexdigest()


def _open_input_interrupt(
    session: Session,
    tenant_id: str,
    agent: AgentRunRecord,
) -> AgentEventRecord | None:
    return session.scalar(
        select(AgentEventRecord).where(
            AgentEventRecord.tenant_id == tenant_id,
            AgentEventRecord.agent_run_id == agent.agent_run_id,
            AgentEventRecord.sequence == agent.last_sequence,
            AgentEventRecord.event_type == "diagnosis.needs_information",
        )
    )


def _workflow_command(
    identity: IdentityContext,
    run: DiagnosisRunRecord,
    *,
    request_id: str,
    source_diagnosis_run_id: str,
    confirmed_input_event_id: str,
) -> DiagnosisWorkflowInput:
    return DiagnosisWorkflowInput(
        tenant_id=identity.tenant_id,
        subject_id=identity.subject_id,
        roles=tuple(sorted(role.value for role in identity.roles)),
        asset_ids=tuple(sorted(identity.asset_ids)),
        site_ids=tuple(sorted(identity.site_ids)),
        incident_id=run.incident_id,
        diagnosis_run_id=run.diagnosis_run_id,
        agent_run_id=run.agent_run_id,
        workflow_id=run.workflow_id,
        request_id=request_id,
        source_diagnosis_run_id=source_diagnosis_run_id,
        confirmed_input_event_id=confirmed_input_event_id,
        oidc_subject=identity.oidc_subject,
        issued_at=identity.issued_at.isoformat(),
        expires_at=identity.expires_at.isoformat(),
    )
