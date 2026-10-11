"""Append-only diagnosis quality feedback bound to authoritative AI run facts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import select

from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.maintenance_planning.review_isolation import (
    lock_subject,
    record_disclosure,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    DiagnosisQualityFeedbackRecord,
    DiagnosisRunRecord,
    IncidentRecord,
)

DiagnosisFeedbackVerdict = Literal[
    "HELPFUL",
    "PARTIALLY_HELPFUL",
    "NOT_HELPFUL",
    "UNSAFE",
]

DIAGNOSIS_FEEDBACK_ISSUE_CODES = frozenset(
    {
        "INCORRECT_ROOT_CAUSE",
        "MISSING_EVIDENCE",
        "WRONG_CITATION",
        "OUTDATED_KNOWLEDGE",
        "TOOL_FACT_MISMATCH",
        "INCOMPLETE_NEXT_STEPS",
        "UNSAFE_RECOMMENDATION",
        "OTHER",
    }
)
_TERMINAL_STATUSES = frozenset({"COMPLETED", "NEEDS_INFORMATION"})
_SUBMITTER_ROLES = frozenset(
    {Role.FIELD_ENGINEER, Role.AFTER_SALES_ENGINEER, Role.DOMAIN_EXPERT}
)


class DiagnosisFeedbackNotVisible(RuntimeError):
    """The diagnosis does not exist inside the caller's authorized scope."""


class DiagnosisFeedbackRejected(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class DiagnosisFeedbackInput:
    diagnosis_run_id: str
    verdict: DiagnosisFeedbackVerdict
    issue_codes: tuple[str, ...]
    comment: str | None


@dataclass(frozen=True, slots=True)
class DiagnosisFeedbackView:
    feedback_id: str
    diagnosis_run_id: str
    incident_id: str
    asset_id: str
    agent_run_id: str
    diagnosis_status: str
    diagnosis_version: int
    report_digest: str
    manifest_digest: str
    model_release_id: str | None
    prompt_bundle_id: str | None
    index_release_id: str | None
    context_hash: str | None
    verdict: str
    issue_codes: tuple[str, ...]
    comment: str | None
    content_digest: str
    submitted_by_subject_id: str
    submitted_by_role: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DiagnosisFeedbackCreateResult:
    feedback: DiagnosisFeedbackView
    created: bool


@dataclass(frozen=True, slots=True)
class DiagnosisFeedbackListResult:
    feedback: tuple[DiagnosisFeedbackView, ...]
    can_submit: bool


class DiagnosisFeedbackService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def create(
        self,
        identity: IdentityContext,
        value: DiagnosisFeedbackInput,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> DiagnosisFeedbackCreateResult:
        normalized = _normalize_input(value)
        _validate_idempotency_key(idempotency_key)
        request_digest = canonical_digest(
            {
                "diagnosis_run_id": normalized.diagnosis_run_id,
                "verdict": normalized.verdict,
                "issue_codes": list(normalized.issue_codes),
                "comment": normalized.comment,
            }
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            run, incident = _run_and_incident(
                session,
                identity.tenant_id,
                normalized.diagnosis_run_id,
                lock=True,
            )
            _require_visible(
                self._authorizer,
                identity,
                Action.CREATE_DIAGNOSIS_FEEDBACK,
                run,
                incident,
                request_id=request_id,
            )
            record_disclosure(
                session,
                identity,
                (run.diagnosis_run_id,),
                resource_kind="diagnosis-feedback",
                resource_id=run.diagnosis_run_id,
                request_id=request_id,
            )
            replay = session.scalar(
                select(DiagnosisQualityFeedbackRecord).where(
                    DiagnosisQualityFeedbackRecord.tenant_id == identity.tenant_id,
                    DiagnosisQualityFeedbackRecord.submitted_by_subject_id
                    == identity.subject_id,
                    DiagnosisQualityFeedbackRecord.idempotency_key == idempotency_key,
                )
            )
            if replay is not None:
                if (
                    replay.request_digest != request_digest
                    or replay.diagnosis_run_id != normalized.diagnosis_run_id
                ):
                    raise DiagnosisFeedbackRejected(
                        "diagnosis_feedback_idempotency_conflict"
                    )
                return DiagnosisFeedbackCreateResult(_view(replay), created=False)

            existing = session.scalar(
                select(DiagnosisQualityFeedbackRecord).where(
                    DiagnosisQualityFeedbackRecord.tenant_id == identity.tenant_id,
                    DiagnosisQualityFeedbackRecord.diagnosis_run_id
                    == normalized.diagnosis_run_id,
                    DiagnosisQualityFeedbackRecord.submitted_by_subject_id
                    == identity.subject_id,
                )
            )
            if existing is not None:
                raise DiagnosisFeedbackRejected("diagnosis_feedback_already_exists")
            if run.status not in _TERMINAL_STATUSES:
                raise DiagnosisFeedbackRejected("diagnosis_feedback_run_not_terminal")
            if not isinstance(run.report, dict) or not run.report:
                raise DiagnosisFeedbackRejected("diagnosis_feedback_report_not_available")
            if not isinstance(run.manifest, dict) or not run.manifest:
                raise DiagnosisFeedbackRejected("diagnosis_feedback_manifest_not_available")

            report_digest = canonical_digest(run.report)
            manifest_digest = canonical_digest(run.manifest)
            context = run.manifest.get("input_context")
            context_hash = canonical_digest(context) if context is not None else None
            model_release_id = _manifest_reference(run.manifest, "model_release")
            prompt_bundle_id = _manifest_reference(run.manifest, "prompt_bundle")
            index_release_id = _manifest_reference(run.manifest, "index_release_id")
            content_digest = diagnosis_feedback_content_digest(
                diagnosis_run_id=run.diagnosis_run_id,
                incident_id=incident.incident_id,
                asset_id=incident.asset_id,
                agent_run_id=run.agent_run_id,
                diagnosis_status=run.status,
                diagnosis_version=run.version,
                report_digest=report_digest,
                manifest_digest=manifest_digest,
                model_release_id=model_release_id,
                prompt_bundle_id=prompt_bundle_id,
                index_release_id=index_release_id,
                context_hash=context_hash,
                verdict=normalized.verdict,
                issue_codes=normalized.issue_codes,
                comment=normalized.comment,
            )
            submitted_by_role = next(
                role.value
                for role in sorted(identity.roles, key=lambda candidate: candidate.value)
                if role in _SUBMITTER_ROLES
            )
            record = DiagnosisQualityFeedbackRecord(
                tenant_id=identity.tenant_id,
                feedback_id=f"diagnosis-feedback-{uuid4().hex}",
                idempotency_key=idempotency_key,
                request_digest=request_digest,
                diagnosis_run_id=run.diagnosis_run_id,
                incident_id=incident.incident_id,
                asset_id=incident.asset_id,
                agent_run_id=run.agent_run_id,
                diagnosis_status=run.status,
                diagnosis_version=run.version,
                report_digest=report_digest,
                manifest_digest=manifest_digest,
                model_release_id=model_release_id,
                prompt_bundle_id=prompt_bundle_id,
                index_release_id=index_release_id,
                context_hash=context_hash,
                verdict=normalized.verdict,
                issue_codes=list(normalized.issue_codes),
                comment=normalized.comment,
                content_digest=content_digest,
                submitted_by_subject_id=identity.subject_id,
                submitted_by_role=submitted_by_role,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            return DiagnosisFeedbackCreateResult(_view(record), created=True)

    def list_for_run(
        self,
        identity: IdentityContext,
        diagnosis_run_id: str,
        *,
        request_id: str,
    ) -> DiagnosisFeedbackListResult:
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            run, incident = _run_and_incident(
                session, identity.tenant_id, diagnosis_run_id, lock=False
            )
            _require_visible(
                self._authorizer,
                identity,
                Action.READ_DIAGNOSIS,
                run,
                incident,
                request_id=request_id,
            )
            record_disclosure(
                session,
                identity,
                (run.diagnosis_run_id,),
                resource_kind="diagnosis-feedback",
                resource_id=run.diagnosis_run_id,
                request_id=request_id,
            )
            records = tuple(
                session.scalars(
                    select(DiagnosisQualityFeedbackRecord)
                    .where(
                        DiagnosisQualityFeedbackRecord.tenant_id == identity.tenant_id,
                        DiagnosisQualityFeedbackRecord.diagnosis_run_id
                        == diagnosis_run_id,
                    )
                    .order_by(
                        DiagnosisQualityFeedbackRecord.created_at,
                        DiagnosisQualityFeedbackRecord.feedback_id,
                    )
                )
            )
            already_submitted = any(
                item.submitted_by_subject_id == identity.subject_id for item in records
            )
            can_submit = (
                not already_submitted
                and run.status in _TERMINAL_STATUSES
                and isinstance(run.report, dict)
                and bool(run.report)
                and self._authorizer.decide(
                    identity,
                    Action.CREATE_DIAGNOSIS_FEEDBACK,
                    _resource(identity.tenant_id, run, incident),
                ).allowed
            )
            return DiagnosisFeedbackListResult(
                feedback=tuple(_view(item) for item in records),
                can_submit=can_submit,
            )


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{sha256(payload).hexdigest()}"


def diagnosis_feedback_content_digest(
    *,
    diagnosis_run_id: str,
    incident_id: str,
    asset_id: str,
    agent_run_id: str,
    diagnosis_status: str,
    diagnosis_version: int,
    report_digest: str,
    manifest_digest: str,
    model_release_id: str | None,
    prompt_bundle_id: str | None,
    index_release_id: str | None,
    context_hash: str | None,
    verdict: str,
    issue_codes: tuple[str, ...],
    comment: str | None,
) -> str:
    return canonical_digest(
        {
            "diagnosis_run_id": diagnosis_run_id,
            "incident_id": incident_id,
            "asset_id": asset_id,
            "agent_run_id": agent_run_id,
            "diagnosis_status": diagnosis_status,
            "diagnosis_version": diagnosis_version,
            "report_digest": report_digest,
            "manifest_digest": manifest_digest,
            "model_release_id": model_release_id,
            "prompt_bundle_id": prompt_bundle_id,
            "index_release_id": index_release_id,
            "context_hash": context_hash,
            "verdict": verdict,
            "issue_codes": list(issue_codes),
            "comment": comment,
        }
    )


def record_content_digest(record: DiagnosisQualityFeedbackRecord) -> str:
    return diagnosis_feedback_content_digest(
        diagnosis_run_id=record.diagnosis_run_id,
        incident_id=record.incident_id,
        asset_id=record.asset_id,
        agent_run_id=record.agent_run_id,
        diagnosis_status=record.diagnosis_status,
        diagnosis_version=record.diagnosis_version,
        report_digest=record.report_digest,
        manifest_digest=record.manifest_digest,
        model_release_id=record.model_release_id,
        prompt_bundle_id=record.prompt_bundle_id,
        index_release_id=record.index_release_id,
        context_hash=record.context_hash,
        verdict=record.verdict,
        issue_codes=tuple(record.issue_codes),
        comment=record.comment,
    )


def _normalize_input(value: DiagnosisFeedbackInput) -> DiagnosisFeedbackInput:
    if value.verdict not in {
        "HELPFUL",
        "PARTIALLY_HELPFUL",
        "NOT_HELPFUL",
        "UNSAFE",
    }:
        raise DiagnosisFeedbackRejected("diagnosis_feedback_verdict_invalid")
    issue_codes = tuple(sorted(set(value.issue_codes)))
    if len(issue_codes) != len(value.issue_codes) or not set(issue_codes).issubset(
        DIAGNOSIS_FEEDBACK_ISSUE_CODES
    ):
        raise DiagnosisFeedbackRejected("diagnosis_feedback_issue_codes_invalid")
    comment = value.comment.strip() if value.comment is not None else None
    comment = comment or None
    if comment is not None and len(comment) > 4_000:
        raise DiagnosisFeedbackRejected("diagnosis_feedback_comment_invalid")
    if value.verdict in {"NOT_HELPFUL", "UNSAFE"} and (
        not issue_codes or comment is None or len(comment) < 8
    ):
        raise DiagnosisFeedbackRejected("diagnosis_feedback_negative_basis_required")
    if value.verdict == "UNSAFE" and "UNSAFE_RECOMMENDATION" not in issue_codes:
        raise DiagnosisFeedbackRejected("diagnosis_feedback_unsafe_issue_required")
    return DiagnosisFeedbackInput(
        diagnosis_run_id=value.diagnosis_run_id,
        verdict=value.verdict,
        issue_codes=issue_codes,
        comment=comment,
    )


def _validate_idempotency_key(value: str) -> None:
    if not 1 <= len(value) <= 255 or value != value.strip():
        raise DiagnosisFeedbackRejected("diagnosis_feedback_idempotency_key_invalid")


def _manifest_reference(manifest: dict[str, Any], key: str) -> str | None:
    value = manifest.get(key)
    return value if isinstance(value, str) and value else None


def _run_and_incident(
    session: Any,
    tenant_id: str,
    diagnosis_run_id: str,
    *,
    lock: bool,
) -> tuple[DiagnosisRunRecord, IncidentRecord]:
    statement = select(DiagnosisRunRecord).where(
        DiagnosisRunRecord.tenant_id == tenant_id,
        DiagnosisRunRecord.diagnosis_run_id == diagnosis_run_id,
    )
    if lock:
        statement = statement.with_for_update()
    run = session.scalar(statement)
    if run is None:
        raise DiagnosisFeedbackNotVisible
    incident = session.scalar(
        select(IncidentRecord).where(
            IncidentRecord.tenant_id == tenant_id,
            IncidentRecord.incident_id == run.incident_id,
        )
    )
    if incident is None:
        raise DiagnosisFeedbackNotVisible
    return run, incident


def _resource(
    tenant_id: str,
    run: DiagnosisRunRecord,
    incident: IncidentRecord,
) -> ResourceContext:
    return ResourceContext(
        tenant_id=tenant_id,
        resource_id=run.diagnosis_run_id,
        asset_id=incident.asset_id,
    )


def _require_visible(
    authorizer: Authorizer,
    identity: IdentityContext,
    action: Action,
    run: DiagnosisRunRecord,
    incident: IncidentRecord,
    *,
    request_id: str,
) -> None:
    try:
        authorizer.require(
            identity,
            action,
            _resource(identity.tenant_id, run, incident),
            request_id=request_id,
        )
    except AuthorizationDenied as exc:
        raise DiagnosisFeedbackNotVisible from exc


def _view(record: DiagnosisQualityFeedbackRecord) -> DiagnosisFeedbackView:
    return DiagnosisFeedbackView(
        feedback_id=record.feedback_id,
        diagnosis_run_id=record.diagnosis_run_id,
        incident_id=record.incident_id,
        asset_id=record.asset_id,
        agent_run_id=record.agent_run_id,
        diagnosis_status=record.diagnosis_status,
        diagnosis_version=record.diagnosis_version,
        report_digest=record.report_digest,
        manifest_digest=record.manifest_digest,
        model_release_id=record.model_release_id,
        prompt_bundle_id=record.prompt_bundle_id,
        index_release_id=record.index_release_id,
        context_hash=record.context_hash,
        verdict=record.verdict,
        issue_codes=tuple(record.issue_codes),
        comment=record.comment,
        content_digest=record.content_digest,
        submitted_by_subject_id=record.submitted_by_subject_id,
        submitted_by_role=record.submitted_by_role,
        created_at=_utc(record.created_at),
    )


__all__ = [
    "DIAGNOSIS_FEEDBACK_ISSUE_CODES",
    "DiagnosisFeedbackCreateResult",
    "DiagnosisFeedbackInput",
    "DiagnosisFeedbackListResult",
    "DiagnosisFeedbackNotVisible",
    "DiagnosisFeedbackRejected",
    "DiagnosisFeedbackService",
    "DiagnosisFeedbackVerdict",
    "DiagnosisFeedbackView",
    "canonical_digest",
    "diagnosis_feedback_content_digest",
    "record_content_digest",
]
