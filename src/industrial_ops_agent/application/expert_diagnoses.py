"""DIA-004 expert takeover and append-only diagnosis revision workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.application.diagnoses import (
    advance_incident_after_completed_diagnosis,
)
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.knowledge.service import CitationService, KnowledgeNotVisible
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AgentEventRecord,
    AgentRunRecord,
    DiagnosisExpertInterventionRecord,
    DiagnosisExpertRevisionRecord,
    DiagnosisRunRecord,
    IdempotencyRecord,
    IncidentRecord,
)

ExpertOutcome = Literal["COMPLETED", "ESCALATED"]
_TAKEOVER_STATUSES = frozenset({"RUNNING", "COMPLETED", "NEEDS_INFORMATION"})


class ExpertDiagnosisError(RuntimeError):
    def __init__(self, reason: str, *, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ExpertDiagnosisRevisionInput:
    outcome: ExpertOutcome
    conclusion: str | None
    citation_ids: tuple[str, ...]
    contradictions: tuple[str, ...]
    missing_information: tuple[str, ...]
    next_checks: tuple[str, ...]
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class ExpertDiagnosisRevisionView:
    revision_id: str
    ordinal: int
    outcome: str
    report: dict[str, Any]
    report_digest: str
    source_ai_report_digest: str
    revision_reason: str
    authored_by_subject_id: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ExpertDiagnosisInterventionView:
    intervention_id: str
    diagnosis_run_id: str
    status: str
    source_run_version: int
    ai_report_snapshot: dict[str, Any] | None
    ai_report_digest: str | None
    takeover_reason: str
    requested_by_subject_id: str
    completed_at: datetime | None
    version: int
    revisions: tuple[ExpertDiagnosisRevisionView, ...]
    created_at: datetime
    updated_at: datetime


class ExpertDiagnosisService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def take_over(
        self,
        identity: IdentityContext,
        diagnosis_run_id: str,
        *,
        reason: str,
        expected_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> ExpertDiagnosisInterventionView:
        normalized_reason = _bounded_text(reason, "expert_takeover_reason_invalid", 8, 1_024)
        _validate_idempotency_key(idempotency_key)
        request_hash = _digest(
            {
                "diagnosis_run_id": diagnosis_run_id,
                "expected_version": expected_version,
                "reason": normalized_reason,
            }
        )
        storage_key = f"diagnosis-expert-takeover:{idempotency_key}"
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            replay = _idempotency_replay(
                session,
                identity,
                storage_key=storage_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return _intervention_view(
                    session,
                    _intervention_or_hidden(session, identity.tenant_id, replay),
                )
            run, incident = _locked_run_and_incident(
                session, identity.tenant_id, diagnosis_run_id
            )
            self._require(
                identity,
                Action.TAKEOVER_DIAGNOSIS,
                run,
                incident,
                request_id,
            )
            if run.version != expected_version:
                raise ExpertDiagnosisError(
                    "expert_diagnosis_version_conflict",
                    current_version=run.version,
                )
            if run.status not in _TAKEOVER_STATUSES:
                raise ExpertDiagnosisError("expert_diagnosis_takeover_state_invalid")
            existing = session.scalar(
                select(DiagnosisExpertInterventionRecord).where(
                    DiagnosisExpertInterventionRecord.tenant_id == identity.tenant_id,
                    DiagnosisExpertInterventionRecord.diagnosis_run_id == diagnosis_run_id,
                    DiagnosisExpertInterventionRecord.status == "OPEN",
                )
            )
            if existing is not None:
                raise ExpertDiagnosisError("expert_diagnosis_takeover_already_open")
            snapshot = dict(run.report) if isinstance(run.report, dict) else None
            intervention = DiagnosisExpertInterventionRecord(
                tenant_id=identity.tenant_id,
                intervention_id=f"diagnosis-expert-intervention-{uuid4().hex}",
                diagnosis_run_id=diagnosis_run_id,
                status="OPEN",
                source_run_version=run.version,
                ai_report_snapshot=snapshot,
                ai_report_digest=_digest(snapshot) if snapshot is not None else None,
                takeover_reason=normalized_reason,
                requested_by_subject_id=identity.subject_id,
                completed_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(intervention)
            run.status = "WAITING_EXPERT"
            run.stop_reason = "expert_takeover_requested"
            run.version += 1
            run.updated_at = now
            _append_agent_event(
                session,
                identity.tenant_id,
                run.agent_run_id,
                event_type="diagnosis.expert_takeover_requested",
                payload={
                    "diagnosis_run_id": diagnosis_run_id,
                    "intervention_id": intervention.intervention_id,
                    "expert_subject_id": identity.subject_id,
                },
                agent_status="WAITING_EXPERT",
                now=now,
            )
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=intervention.intervention_id,
                    expires_at=now + timedelta(days=30),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _intervention_view(session, intervention)

    def submit_revision(
        self,
        identity: IdentityContext,
        diagnosis_run_id: str,
        intervention_id: str,
        revision: ExpertDiagnosisRevisionInput,
        *,
        expected_version: int,
        request_id: str,
    ) -> ExpertDiagnosisInterventionView:
        normalized = _validated_revision(revision)
        authorized_citations: dict[str, dict[str, Any]] = {}
        for citation_id in normalized.citation_ids:
            try:
                citation = CitationService(self._database, self._authorizer).get(
                    identity,
                    citation_id,
                    request_id=request_id,
                )
            except KnowledgeNotVisible as exc:
                raise ExpertDiagnosisError(
                    "expert_diagnosis_citation_not_visible"
                ) from exc
            authorized_citations[citation_id] = {
                "citation_id": citation.citation_id,
                "chunk_id": citation.chunk_id,
                "document_id": citation.document_id,
                "document_version_id": citation.document_version_id,
                "document_version": citation.document_version,
                "title": citation.title,
                "source_checksum": citation.source_checksum,
                "content_checksum": citation.content_checksum,
                "page_number": citation.page_number,
            }

        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            run, incident = _locked_run_and_incident(
                session, identity.tenant_id, diagnosis_run_id
            )
            self._require(
                identity,
                Action.REVISE_DIAGNOSIS,
                run,
                incident,
                request_id,
            )
            intervention = session.scalar(
                select(DiagnosisExpertInterventionRecord)
                .where(
                    DiagnosisExpertInterventionRecord.tenant_id == identity.tenant_id,
                    DiagnosisExpertInterventionRecord.intervention_id == intervention_id,
                    DiagnosisExpertInterventionRecord.diagnosis_run_id == diagnosis_run_id,
                )
                .with_for_update()
            )
            if intervention is None:
                raise ResourceNotVisible
            if intervention.version != expected_version:
                raise ExpertDiagnosisError(
                    "expert_intervention_version_conflict",
                    current_version=intervention.version,
                )
            if (
                intervention.status != "OPEN"
                or run.status != "WAITING_EXPERT"
                or intervention.requested_by_subject_id != identity.subject_id
            ):
                raise ExpertDiagnosisError("expert_diagnosis_revision_state_invalid")
            if intervention.ai_report_snapshot is None or intervention.ai_report_digest is None:
                raise ExpertDiagnosisError("expert_diagnosis_ai_draft_not_ready")
            original_citations = _citation_map(intervention.ai_report_snapshot)
            if any(
                citation_id not in original_citations
                for citation_id in normalized.citation_ids
            ):
                raise ExpertDiagnosisError("expert_diagnosis_citation_not_in_ai_draft")
            report = _expert_report(normalized, authorized_citations)
            ordinal = int(
                session.scalar(
                    select(func.coalesce(func.max(DiagnosisExpertRevisionRecord.ordinal), 0)).where(
                        DiagnosisExpertRevisionRecord.tenant_id == identity.tenant_id,
                        DiagnosisExpertRevisionRecord.intervention_id == intervention_id,
                    )
                )
                or 0
            ) + 1
            record = DiagnosisExpertRevisionRecord(
                tenant_id=identity.tenant_id,
                revision_id=f"diagnosis-expert-revision-{uuid4().hex}",
                intervention_id=intervention_id,
                diagnosis_run_id=diagnosis_run_id,
                ordinal=ordinal,
                outcome=normalized.outcome,
                report_json=report,
                report_digest=_digest(report),
                source_ai_report_digest=intervention.ai_report_digest,
                revision_reason=normalized.reason,
                authored_by_subject_id=identity.subject_id,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            intervention.status = "COMPLETED"
            intervention.completed_at = now
            intervention.version += 1
            intervention.updated_at = now
            run.status = normalized.outcome
            run.stop_reason = (
                "expert_revision_completed"
                if normalized.outcome == "COMPLETED"
                else "expert_escalated"
            )
            run.version += 1
            run.updated_at = now
            if normalized.outcome == "COMPLETED":
                advance_incident_after_completed_diagnosis(
                    session,
                    tenant_id=identity.tenant_id,
                    incident_id=incident.incident_id,
                    now=now,
                )
            _append_agent_event(
                session,
                identity.tenant_id,
                run.agent_run_id,
                event_type=(
                    "diagnosis.expert_revision_completed"
                    if normalized.outcome == "COMPLETED"
                    else "diagnosis.expert_escalated"
                ),
                payload={
                    "diagnosis_run_id": diagnosis_run_id,
                    "intervention_id": intervention_id,
                    "revision_id": record.revision_id,
                    "report_digest": record.report_digest,
                    "source_ai_report_digest": record.source_ai_report_digest,
                },
                agent_status=normalized.outcome,
                now=now,
            )
            session.flush()
            return _intervention_view(session, intervention)

    def list_for_run(
        self,
        identity: IdentityContext,
        diagnosis_run_id: str,
        *,
        request_id: str,
    ) -> tuple[ExpertDiagnosisInterventionView, ...]:
        with self._database.transaction(identity.tenant_context) as session:
            run, incident = _run_and_incident(session, identity.tenant_id, diagnosis_run_id)
            self._require(identity, Action.READ_DIAGNOSIS, run, incident, request_id)
            records = tuple(
                session.scalars(
                    select(DiagnosisExpertInterventionRecord)
                    .where(
                        DiagnosisExpertInterventionRecord.tenant_id == identity.tenant_id,
                        DiagnosisExpertInterventionRecord.diagnosis_run_id == diagnosis_run_id,
                    )
                    .order_by(
                        DiagnosisExpertInterventionRecord.created_at,
                        DiagnosisExpertInterventionRecord.intervention_id,
                    )
                )
            )
            return tuple(_intervention_view(session, item) for item in records)

    def effective_report(
        self,
        identity: IdentityContext,
        diagnosis_run_id: str,
        *,
        request_id: str,
    ) -> dict[str, Any] | None:
        with self._database.transaction(identity.tenant_context) as session:
            run, incident = _run_and_incident(session, identity.tenant_id, diagnosis_run_id)
            self._require(identity, Action.READ_DIAGNOSIS, run, incident, request_id)
            return effective_report(session, identity.tenant_id, run)

    def legal_actions(
        self,
        identity: IdentityContext,
        diagnosis_run_id: str,
    ) -> tuple[str, ...]:
        with self._database.transaction(identity.tenant_context) as session:
            run, incident = _run_and_incident(session, identity.tenant_id, diagnosis_run_id)
            resource = ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id=diagnosis_run_id,
                asset_id=incident.asset_id,
            )
            actions: list[str] = []
            if run.status in _TAKEOVER_STATUSES and self._authorizer.decide(
                identity, Action.TAKEOVER_DIAGNOSIS, resource
            ).allowed:
                actions.append("TAKE_OVER_DIAGNOSIS")
            open_intervention = session.scalar(
                select(DiagnosisExpertInterventionRecord).where(
                    DiagnosisExpertInterventionRecord.tenant_id == identity.tenant_id,
                    DiagnosisExpertInterventionRecord.diagnosis_run_id == diagnosis_run_id,
                    DiagnosisExpertInterventionRecord.status == "OPEN",
                    DiagnosisExpertInterventionRecord.requested_by_subject_id
                    == identity.subject_id,
                )
            )
            if (
                run.status == "WAITING_EXPERT"
                and open_intervention is not None
                and open_intervention.ai_report_digest is not None
                and self._authorizer.decide(
                    identity, Action.REVISE_DIAGNOSIS, resource
                ).allowed
            ):
                actions.append("SUBMIT_EXPERT_REVISION")
            return tuple(actions)

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        run: DiagnosisRunRecord,
        incident: IncidentRecord,
        request_id: str,
    ) -> None:
        try:
            self._authorizer.require(
                identity,
                action,
                ResourceContext(
                    tenant_id=identity.tenant_id,
                    resource_id=run.diagnosis_run_id,
                    asset_id=incident.asset_id,
                ),
                request_id=request_id,
            )
        except AuthorizationDenied as exc:
            raise ResourceNotVisible from exc


def effective_report(
    session: Session,
    tenant_id: str,
    run: DiagnosisRunRecord,
) -> dict[str, Any] | None:
    revision = session.scalar(
        select(DiagnosisExpertRevisionRecord)
        .where(
            DiagnosisExpertRevisionRecord.tenant_id == tenant_id,
            DiagnosisExpertRevisionRecord.diagnosis_run_id == run.diagnosis_run_id,
        )
        .order_by(
            DiagnosisExpertRevisionRecord.created_at.desc(),
            DiagnosisExpertRevisionRecord.revision_id.desc(),
        )
        .limit(1)
    )
    if revision is not None:
        return dict(revision.report_json)
    return dict(run.report) if isinstance(run.report, dict) else None


def preserve_waiting_expert_ai_draft(
    session: Session,
    run: DiagnosisRunRecord,
    report: dict[str, Any],
    *,
    now: datetime,
) -> bool:
    """Persist a late AI draft while preserving the expert-owned run state."""

    if run.status != "WAITING_EXPERT":
        return False
    intervention = session.scalar(
        select(DiagnosisExpertInterventionRecord)
        .where(
            DiagnosisExpertInterventionRecord.tenant_id == run.tenant_id,
            DiagnosisExpertInterventionRecord.diagnosis_run_id == run.diagnosis_run_id,
            DiagnosisExpertInterventionRecord.status == "OPEN",
        )
        .order_by(DiagnosisExpertInterventionRecord.created_at.desc())
        .limit(1)
    )
    if intervention is None:
        raise RuntimeError("waiting expert diagnosis has no open intervention")
    digest = _digest(report)
    if intervention.ai_report_digest is None:
        intervention.ai_report_snapshot = dict(report)
        intervention.ai_report_digest = digest
        intervention.version += 1
        intervention.updated_at = now
    elif intervention.ai_report_digest != digest:
        raise RuntimeError("expert intervention AI draft is immutable")
    run.report = dict(report)
    run.stop_reason = "expert_review_required"
    run.version += 1
    run.updated_at = now
    return True


def _validated_revision(value: ExpertDiagnosisRevisionInput) -> ExpertDiagnosisRevisionInput:
    if value.outcome not in {"COMPLETED", "ESCALATED"}:
        raise ExpertDiagnosisError("expert_diagnosis_outcome_invalid")
    reason = _bounded_text(value.reason, "expert_revision_reason_invalid", 8, 1_024)
    confidence = float(value.confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ExpertDiagnosisError("expert_diagnosis_confidence_invalid")
    citation_ids = tuple(dict.fromkeys(value.citation_ids))
    if len(citation_ids) != len(value.citation_ids) or len(citation_ids) > 50:
        raise ExpertDiagnosisError("expert_diagnosis_citations_invalid")
    conclusion = value.conclusion.strip() if value.conclusion else None
    contradictions = _text_items(value.contradictions, "expert_diagnosis_contradictions_invalid")
    missing = _text_items(value.missing_information, "expert_diagnosis_missing_invalid")
    next_checks = _text_items(value.next_checks, "expert_diagnosis_checks_invalid")
    if value.outcome == "COMPLETED" and (
        not conclusion or len(conclusion) > 4_000 or not citation_ids or missing or contradictions
    ):
        raise ExpertDiagnosisError("expert_completed_report_gate_failed")
    if value.outcome == "ESCALATED" and (conclusion is not None or not missing):
        raise ExpertDiagnosisError("expert_escalated_report_gate_failed")
    return ExpertDiagnosisRevisionInput(
        outcome=value.outcome,
        conclusion=conclusion,
        citation_ids=citation_ids,
        contradictions=contradictions,
        missing_information=missing,
        next_checks=next_checks,
        confidence=confidence,
        reason=reason,
    )


def _expert_report(
    revision: ExpertDiagnosisRevisionInput,
    citations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": revision.outcome,
        "conclusion": revision.conclusion,
        "citations": [citations[item] for item in revision.citation_ids],
        "contradictions": list(revision.contradictions),
        "missing_information": list(revision.missing_information),
        "next_checks": list(revision.next_checks),
        "confidence": revision.confidence,
        "stop_reason": (
            "expert_revision_completed"
            if revision.outcome == "COMPLETED"
            else "expert_escalated"
        ),
    }


def _citation_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    citations = report.get("citations")
    if not isinstance(citations, list):
        return {}
    return {
        str(item["citation_id"]): item
        for item in citations
        if isinstance(item, dict) and isinstance(item.get("citation_id"), str)
    }


def _append_agent_event(
    session: Session,
    tenant_id: str,
    agent_run_id: str,
    *,
    event_type: str,
    payload: dict[str, Any],
    agent_status: str,
    now: datetime,
) -> None:
    agent_run = session.scalar(
        select(AgentRunRecord)
        .where(
            AgentRunRecord.tenant_id == tenant_id,
            AgentRunRecord.agent_run_id == agent_run_id,
        )
        .with_for_update()
    )
    if agent_run is None:
        raise RuntimeError("diagnosis expert intervention has no agent run")
    sequence = agent_run.last_sequence + 1
    session.add(
        AgentEventRecord(
            tenant_id=tenant_id,
            event_id=f"agent-event-{uuid4().hex}",
            agent_run_id=agent_run_id,
            sequence=sequence,
            event_type=event_type,
            payload=payload,
            visibility="authorized",
            occurred_at=now,
            created_at=now,
            updated_at=now,
        )
    )
    agent_run.last_sequence = sequence
    agent_run.status = agent_status
    agent_run.version += 1
    agent_run.updated_at = now


def _run_and_incident(
    session: Session,
    tenant_id: str,
    diagnosis_run_id: str,
) -> tuple[DiagnosisRunRecord, IncidentRecord]:
    run = session.scalar(
        select(DiagnosisRunRecord).where(
            DiagnosisRunRecord.tenant_id == tenant_id,
            DiagnosisRunRecord.diagnosis_run_id == diagnosis_run_id,
        )
    )
    if run is None:
        raise ResourceNotVisible
    incident = session.scalar(
        select(IncidentRecord).where(
            IncidentRecord.tenant_id == tenant_id,
            IncidentRecord.incident_id == run.incident_id,
        )
    )
    if incident is None:
        raise ResourceNotVisible
    return run, incident


def _locked_run_and_incident(
    session: Session,
    tenant_id: str,
    diagnosis_run_id: str,
) -> tuple[DiagnosisRunRecord, IncidentRecord]:
    run = session.scalar(
        select(DiagnosisRunRecord)
        .where(
            DiagnosisRunRecord.tenant_id == tenant_id,
            DiagnosisRunRecord.diagnosis_run_id == diagnosis_run_id,
        )
        .with_for_update()
    )
    if run is None:
        raise ResourceNotVisible
    incident = session.scalar(
        select(IncidentRecord).where(
            IncidentRecord.tenant_id == tenant_id,
            IncidentRecord.incident_id == run.incident_id,
        )
    )
    if incident is None:
        raise ResourceNotVisible
    return run, incident


def _intervention_or_hidden(
    session: Session,
    tenant_id: str,
    intervention_id: str,
) -> DiagnosisExpertInterventionRecord:
    record = session.scalar(
        select(DiagnosisExpertInterventionRecord).where(
            DiagnosisExpertInterventionRecord.tenant_id == tenant_id,
            DiagnosisExpertInterventionRecord.intervention_id == intervention_id,
        )
    )
    if record is None:
        raise ResourceNotVisible
    return record


def _intervention_view(
    session: Session,
    record: DiagnosisExpertInterventionRecord,
) -> ExpertDiagnosisInterventionView:
    revisions = tuple(
        session.scalars(
            select(DiagnosisExpertRevisionRecord)
            .where(
                DiagnosisExpertRevisionRecord.tenant_id == record.tenant_id,
                DiagnosisExpertRevisionRecord.intervention_id == record.intervention_id,
            )
            .order_by(
                DiagnosisExpertRevisionRecord.ordinal,
                DiagnosisExpertRevisionRecord.revision_id,
            )
        )
    )
    return ExpertDiagnosisInterventionView(
        intervention_id=record.intervention_id,
        diagnosis_run_id=record.diagnosis_run_id,
        status=record.status,
        source_run_version=record.source_run_version,
        ai_report_snapshot=(
            dict(record.ai_report_snapshot) if record.ai_report_snapshot is not None else None
        ),
        ai_report_digest=record.ai_report_digest,
        takeover_reason=record.takeover_reason,
        requested_by_subject_id=record.requested_by_subject_id,
        completed_at=_utc(record.completed_at) if record.completed_at else None,
        version=record.version,
        revisions=tuple(
            ExpertDiagnosisRevisionView(
                revision_id=item.revision_id,
                ordinal=item.ordinal,
                outcome=item.outcome,
                report=dict(item.report_json),
                report_digest=item.report_digest,
                source_ai_report_digest=item.source_ai_report_digest,
                revision_reason=item.revision_reason,
                authored_by_subject_id=item.authored_by_subject_id,
                created_at=_utc(item.created_at),
            )
            for item in revisions
        ),
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
    )


def _idempotency_replay(
    session: Session,
    identity: IdentityContext,
    *,
    storage_key: str,
    request_hash: str,
) -> str | None:
    record = session.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.tenant_id == identity.tenant_id,
            IdempotencyRecord.subject_id == identity.subject_id,
            IdempotencyRecord.key == storage_key,
        )
    )
    if record is None:
        return None
    if record.request_hash != request_hash:
        raise ExpertDiagnosisError("expert_diagnosis_idempotency_conflict")
    return str(record.result_ref)


def _validate_idempotency_key(value: str) -> None:
    if not 1 <= len(value) <= 200:
        raise ExpertDiagnosisError("expert_diagnosis_idempotency_invalid")


def _bounded_text(value: str, reason: str, minimum: int, maximum: int) -> str:
    normalized = value.strip()
    if not minimum <= len(normalized) <= maximum or "\x00" in normalized:
        raise ExpertDiagnosisError(reason)
    return normalized


def _text_items(values: tuple[str, ...], reason: str) -> tuple[str, ...]:
    normalized = tuple(item.strip() for item in values)
    if len(normalized) > 50 or any(not item or len(item) > 1_000 for item in normalized):
        raise ExpertDiagnosisError(reason)
    return normalized


def _digest(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(canonical.encode()).hexdigest()


__all__ = [
    "ExpertDiagnosisError",
    "ExpertDiagnosisInterventionView",
    "ExpertDiagnosisRevisionInput",
    "ExpertDiagnosisRevisionView",
    "ExpertDiagnosisService",
    "effective_report",
    "preserve_waiting_expert_ai_draft",
]
