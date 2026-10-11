"""Create an unapproved feedback reference without copying sensitive business text."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.application.diagnosis_feedback import record_content_digest
from industrial_ops_agent.data_governance.dlp import PresidioDlpProcessor
from industrial_ops_agent.eventing.envelope import (
    EventEnvelope,
    work_order_source_content_hash,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    DataEligibilityDecisionRecord,
    DiagnosisQualityFeedbackRecord,
    DlpProcessingResultRecord,
    FeedbackCandidateRecord,
    IncidentRecord,
    TenantRetentionPolicyRecord,
    WorkOrderCompletionRecord,
    WorkOrderRecord,
    WorkOrderVerificationRecord,
)


class FeedbackSourceRejected(Exception):
    pass


class GovernanceNotVisible(Exception):
    pass


class GovernanceConflict(Exception):
    def __init__(self, reason: str, current_version: int) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


class GovernanceGateDenied(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


GovernanceBlockerCode = Literal[
    "incident_not_closed",
    "training_not_authorized",
    "consent_basis_missing",
    "license_not_approved",
    "retention_missing",
    "retention_expired",
    "dlp_not_run",
    "dlp_review_required",
    "annotation_not_started",
    "annotation_review_pending",
    "annotation_review_required",
    "annotation_conflict",
    "governance_status_blocked",
]


@dataclass(frozen=True, slots=True)
class CandidateGovernanceView:
    candidate_id: str
    status: str
    eligibility_status: str
    allow_training: bool | None
    consent_basis: str | None
    license_status: str
    retention_until: datetime | None
    governance_blockers: tuple[GovernanceBlockerCode, ...]
    version: int


@dataclass(frozen=True, slots=True)
class DlpRunView:
    dlp_result_id: str
    candidate_id: str
    status: str
    policy_version: str
    source_content_hash: str
    content_hash: str
    redacted_content: dict[str, str]
    findings: list[dict[str, Any]]
    residual_entity_types: list[str]
    candidate_version: int


class FeedbackCandidateService:
    def create_from_closed_work_order(
        self,
        session: Session,
        envelope: EventEnvelope,
        *,
        consumer_name: str,
    ) -> FeedbackCandidateRecord:
        payload = envelope.payload
        work_order = session.scalar(
            select(WorkOrderRecord).where(
                WorkOrderRecord.tenant_id == envelope.tenant_id,
                WorkOrderRecord.work_order_id == payload.work_order_id,
            )
        )
        incident = session.scalar(
            select(IncidentRecord).where(
                IncidentRecord.tenant_id == envelope.tenant_id,
                IncidentRecord.incident_id == payload.incident_id,
            )
        )
        completion = session.scalar(
            select(WorkOrderCompletionRecord).where(
                WorkOrderCompletionRecord.tenant_id == envelope.tenant_id,
                WorkOrderCompletionRecord.completion_id == payload.completion_id,
                WorkOrderCompletionRecord.work_order_id == payload.work_order_id,
            )
        )
        verification = session.scalar(
            select(WorkOrderVerificationRecord).where(
                WorkOrderVerificationRecord.tenant_id == envelope.tenant_id,
                WorkOrderVerificationRecord.verification_id == payload.verification_id,
                WorkOrderVerificationRecord.work_order_id == payload.work_order_id,
            )
        )
        expected_hash = work_order_source_content_hash(
            tenant_id=envelope.tenant_id,
            work_order_id=payload.work_order_id,
            incident_id=payload.incident_id,
            completion_id=payload.completion_id,
            verification_id=payload.verification_id,
            aggregate_version=envelope.aggregate_version,
        )
        if (
            envelope.aggregate_id != payload.work_order_id
            or work_order is None
            or work_order.status != "CLOSED"
            or work_order.version != envelope.aggregate_version
            or incident is None
            or incident.status not in {"RESOLVED", "CLOSED"}
            or work_order.incident_id != incident.incident_id
            or completion is None
            or verification is None
            or not verification.passed
            or payload.source_content_hash != expected_hash
        ):
            raise FeedbackSourceRejected("authoritative_source_missing_or_changed")

        existing = session.scalar(
            select(FeedbackCandidateRecord).where(
                FeedbackCandidateRecord.tenant_id == envelope.tenant_id,
                FeedbackCandidateRecord.source_event_id == envelope.event_id,
            )
        )
        if existing is not None:
            return existing

        now = datetime.now(UTC)
        feedback_cutoff = _as_utc(envelope.occurred_at)
        diagnosis_feedback = tuple(
            session.scalars(
                select(DiagnosisQualityFeedbackRecord)
                .where(
                    DiagnosisQualityFeedbackRecord.tenant_id == envelope.tenant_id,
                    DiagnosisQualityFeedbackRecord.incident_id == payload.incident_id,
                    DiagnosisQualityFeedbackRecord.created_at <= feedback_cutoff,
                )
                .order_by(
                    DiagnosisQualityFeedbackRecord.created_at,
                    DiagnosisQualityFeedbackRecord.feedback_id,
                )
            )
        )
        candidate_id = (
            "feedback-"
            + sha256(f"{envelope.tenant_id}\0{envelope.event_id}".encode()).hexdigest()[:32]
        )
        candidate = FeedbackCandidateRecord(
            candidate_id=candidate_id,
            tenant_id=envelope.tenant_id,
            source_event_id=envelope.event_id,
            work_order_id=payload.work_order_id,
            incident_id=payload.incident_id,
            completion_id=payload.completion_id,
            verification_id=payload.verification_id,
            source_content_hash=payload.source_content_hash,
            data_classification=envelope.data_classification,
            status="PENDING_GOVERNANCE",
            eligibility_status="UNDECIDED",
            allow_training=None,
            consent_basis=None,
            license_status="UNKNOWN",
            retention_until=None,
            lineage_origin={
                "consumer_name": consumer_name,
                "event_id": envelope.event_id,
                "event_type": envelope.event_type,
                "event_version": envelope.event_version,
                "aggregate_type": envelope.aggregate_type,
                "aggregate_id": envelope.aggregate_id,
                "aggregate_version": envelope.aggregate_version,
                "work_order_id": payload.work_order_id,
                "incident_id": payload.incident_id,
                "completion_id": payload.completion_id,
                "verification_id": payload.verification_id,
                "source_content_hash": payload.source_content_hash,
                "diagnosis_feedback_cutoff_at": feedback_cutoff.isoformat().replace(
                    "+00:00", "Z"
                ),
                "diagnosis_feedback_refs": [
                    {
                        "feedback_id": item.feedback_id,
                        "content_digest": item.content_digest,
                    }
                    for item in diagnosis_feedback
                ],
            },
            version=1,
            created_at=now,
            updated_at=now,
        )
        session.add(candidate)
        session.flush()
        return candidate


class DataGovernanceService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        dlp_processor: PresidioDlpProcessor,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._dlp_processor = dlp_processor

    def decide_eligibility(
        self,
        identity: IdentityContext,
        candidate_id: str,
        *,
        expected_version: int,
        purpose: str,
        allow_training: bool,
        consent_basis: str | None,
        license_status: str,
        retention_until: datetime | None,
        policy_version: str,
        request_id: str,
    ) -> CandidateGovernanceView:
        self._authorizer.require(
            identity,
            Action.DECIDE_DATA_ELIGIBILITY,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=candidate_id),
            request_id=request_id,
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            candidate = _candidate(session, identity.tenant_id, candidate_id)
            _require_version(candidate, expected_version)
            _require_incident_closed(session, candidate)
            if candidate.status not in {
                "PENDING_GOVERNANCE",
                "ELIGIBLE",
                "ELIGIBILITY_DENIED",
            }:
                raise GovernanceGateDenied("eligibility_decision_locked_after_derivation")
            tenant_retention = session.scalar(
                select(TenantRetentionPolicyRecord).where(
                    TenantRetentionPolicyRecord.tenant_id == identity.tenant_id
                )
            )
            if (
                retention_until is not None
                and tenant_retention is not None
                and _as_utc(retention_until)
                > now + timedelta(days=tenant_retention.training_data_days)
            ):
                raise GovernanceGateDenied("retention_exceeds_tenant_policy")
            effective_allow = bool(
                allow_training
                and consent_basis
                and license_status == "APPROVED"
                and retention_until is not None
                and _as_utc(retention_until) > now
            )
            decision = DataEligibilityDecisionRecord(
                decision_id=f"eligibility-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                candidate_id=candidate_id,
                purpose=purpose,
                allow_training=effective_allow,
                consent_basis=consent_basis,
                license_status=license_status,
                retention_until=retention_until,
                decided_by_subject_id=identity.subject_id,
                policy_version=policy_version,
                candidate_version=candidate.version,
                created_at=now,
                updated_at=now,
            )
            session.add(decision)
            candidate.allow_training = effective_allow
            candidate.consent_basis = consent_basis
            candidate.license_status = license_status
            candidate.retention_until = retention_until
            candidate.eligibility_status = "ELIGIBLE" if effective_allow else "INELIGIBLE"
            candidate.status = "ELIGIBLE" if effective_allow else "ELIGIBILITY_DENIED"
            candidate.version += 1
            candidate.updated_at = now
            session.flush()
            return _candidate_view(candidate)

    def run_dlp(
        self,
        identity: IdentityContext,
        candidate_id: str,
        *,
        expected_version: int,
        policy_version: str,
        request_id: str,
    ) -> DlpRunView:
        self._authorizer.require(
            identity,
            Action.RUN_DATA_DLP,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=candidate_id),
            request_id=request_id,
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            candidate = _candidate(session, identity.tenant_id, candidate_id)
            _require_version(candidate, expected_version)
            _require_incident_closed(session, candidate)
            gate_reason = eligibility_gate_reason(candidate, now)
            if gate_reason is not None:
                self._authorizer.record_guard_decision(
                    identity,
                    action="data_dlp.eligibility_gate",
                    decision="deny",
                    reason_code=gate_reason,
                    request_id=request_id,
                    resource_id=candidate_id,
                )
                raise GovernanceGateDenied(gate_reason)
            content, source_content_hash = authoritative_candidate_content(session, candidate)
            existing = session.scalar(
                select(DlpProcessingResultRecord).where(
                    DlpProcessingResultRecord.tenant_id == identity.tenant_id,
                    DlpProcessingResultRecord.candidate_id == candidate_id,
                    DlpProcessingResultRecord.source_content_hash == source_content_hash,
                    DlpProcessingResultRecord.policy_version == policy_version,
                )
            )
            if existing is not None:
                return _dlp_view(existing, candidate.version)

            output = self._dlp_processor.process(content)
            redacted_hash = _content_hash(output.redacted_content)
            result = DlpProcessingResultRecord(
                dlp_result_id="dlp-"
                + sha256(
                    f"{identity.tenant_id}\0{candidate_id}\0{source_content_hash}\0{policy_version}".encode()
                ).hexdigest()[:32],
                tenant_id=identity.tenant_id,
                candidate_id=candidate_id,
                source_content_hash=source_content_hash,
                policy_version=policy_version,
                status=output.status,
                redacted_content=output.redacted_content,
                findings=list(output.findings),
                residual_entity_types=list(output.residual_entity_types),
                content_hash=redacted_hash,
                processed_by_subject_id=identity.subject_id,
                candidate_version=candidate.version,
                created_at=now,
                updated_at=now,
            )
            session.add(result)
            candidate.status = (
                "DLP_APPROVED" if output.status == "PASSED" else "DLP_REVIEW_REQUIRED"
            )
            candidate.version += 1
            candidate.updated_at = now
            session.flush()
            return _dlp_view(result, candidate.version)


def eligibility_gate_reason(
    candidate: FeedbackCandidateRecord,
    now: datetime,
) -> GovernanceBlockerCode | None:
    if candidate.allow_training is not True or candidate.eligibility_status != "ELIGIBLE":
        return "training_not_authorized"
    if not candidate.consent_basis:
        return "consent_basis_missing"
    if candidate.license_status != "APPROVED":
        return "license_not_approved"
    if candidate.retention_until is None:
        return "retention_missing"
    if _as_utc(candidate.retention_until) <= now:
        return "retention_expired"
    return None


def candidate_governance_blockers(
    candidate: FeedbackCandidateRecord,
    now: datetime,
) -> list[GovernanceBlockerCode]:
    """Return the current fail-closed reason before a candidate can be curated."""

    eligibility_reason = eligibility_gate_reason(candidate, now)
    if eligibility_reason is not None:
        return [eligibility_reason]
    stage_reasons: dict[str, GovernanceBlockerCode] = {
        "ELIGIBLE": "dlp_not_run",
        "DLP_REVIEW_REQUIRED": "dlp_review_required",
        "DLP_APPROVED": "annotation_not_started",
        "ANNOTATION_PENDING": "annotation_review_pending",
        "ANNOTATION_REVIEW_REQUIRED": "annotation_review_required",
        "ANNOTATION_CONFLICT": "annotation_conflict",
    }
    stage_reason = stage_reasons.get(candidate.status)
    if stage_reason is not None:
        return [stage_reason]
    if candidate.status == "READY_FOR_CURATION":
        return []
    return ["governance_status_blocked"]


def authoritative_candidate_content(
    session: Session,
    candidate: FeedbackCandidateRecord,
) -> tuple[dict[str, str], str]:
    incident = session.scalar(
        select(IncidentRecord).where(
            IncidentRecord.tenant_id == candidate.tenant_id,
            IncidentRecord.incident_id == candidate.incident_id,
        )
    )
    completion = session.scalar(
        select(WorkOrderCompletionRecord).where(
            WorkOrderCompletionRecord.tenant_id == candidate.tenant_id,
            WorkOrderCompletionRecord.completion_id == candidate.completion_id,
            WorkOrderCompletionRecord.work_order_id == candidate.work_order_id,
        )
    )
    verification = session.scalar(
        select(WorkOrderVerificationRecord).where(
            WorkOrderVerificationRecord.tenant_id == candidate.tenant_id,
            WorkOrderVerificationRecord.verification_id == candidate.verification_id,
            WorkOrderVerificationRecord.work_order_id == candidate.work_order_id,
        )
    )
    work_order = session.scalar(
        select(WorkOrderRecord).where(
            WorkOrderRecord.tenant_id == candidate.tenant_id,
            WorkOrderRecord.work_order_id == candidate.work_order_id,
        )
    )
    if incident is not None and incident.status != "CLOSED":
        raise GovernanceGateDenied("incident_not_closed")
    if (
        incident is None
        or completion is None
        or verification is None
        or work_order is None
        or incident.status != "CLOSED"
        or work_order.status != "CLOSED"
        or not verification.passed
    ):
        raise GovernanceGateDenied("authoritative_source_missing_or_changed")
    content = {
        "incident_description": incident.description,
        "root_cause": completion.root_cause,
        "customer_confirmation": completion.customer_confirmation or "",
        "verification_reason": verification.reason,
        **{f"action_{index}": action for index, action in enumerate(completion.actions)},
    }
    for index, feedback in enumerate(
        _authoritative_diagnosis_feedback(session, candidate)
    ):
        content[f"diagnosis_feedback_{index}_verdict"] = feedback.verdict
        content[f"diagnosis_feedback_{index}_issue_codes"] = ",".join(
            feedback.issue_codes
        )
        content[f"diagnosis_feedback_{index}_comment"] = feedback.comment or ""
    return content, _content_hash(content)


def diagnosis_feedback_count(candidate: FeedbackCandidateRecord) -> int:
    lineage = candidate.lineage_origin
    if not isinstance(lineage, dict):
        return 0
    refs = lineage.get("diagnosis_feedback_refs")
    return len(refs) if isinstance(refs, list) else 0


def _authoritative_diagnosis_feedback(
    session: Session,
    candidate: FeedbackCandidateRecord,
) -> tuple[DiagnosisQualityFeedbackRecord, ...]:
    lineage = candidate.lineage_origin
    if not isinstance(lineage, dict):
        raise GovernanceGateDenied("diagnosis_feedback_source_missing_or_changed")
    has_cutoff = "diagnosis_feedback_cutoff_at" in lineage
    has_refs = "diagnosis_feedback_refs" in lineage
    if not has_cutoff and not has_refs:
        return ()
    cutoff_value = lineage.get("diagnosis_feedback_cutoff_at")
    refs = lineage.get("diagnosis_feedback_refs")
    if not isinstance(cutoff_value, str) or not isinstance(refs, list):
        raise GovernanceGateDenied("diagnosis_feedback_source_missing_or_changed")
    try:
        cutoff = datetime.fromisoformat(cutoff_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GovernanceGateDenied(
            "diagnosis_feedback_source_missing_or_changed"
        ) from exc
    if cutoff.tzinfo is None:
        raise GovernanceGateDenied("diagnosis_feedback_source_missing_or_changed")
    cutoff = _as_utc(cutoff)

    normalized_refs: list[tuple[str, str]] = []
    for ref in refs:
        if (
            not isinstance(ref, dict)
            or set(ref) != {"feedback_id", "content_digest"}
            or not isinstance(ref.get("feedback_id"), str)
            or not isinstance(ref.get("content_digest"), str)
        ):
            raise GovernanceGateDenied(
                "diagnosis_feedback_source_missing_or_changed"
            )
        normalized_refs.append((ref["feedback_id"], ref["content_digest"]))
    feedback_ids = [feedback_id for feedback_id, _ in normalized_refs]
    if len(feedback_ids) != len(set(feedback_ids)):
        raise GovernanceGateDenied("diagnosis_feedback_source_missing_or_changed")
    if not feedback_ids:
        return ()

    records = tuple(
        session.scalars(
            select(DiagnosisQualityFeedbackRecord).where(
                DiagnosisQualityFeedbackRecord.feedback_id.in_(feedback_ids)
            )
        )
    )
    by_id = {record.feedback_id: record for record in records}
    if len(by_id) != len(feedback_ids):
        raise GovernanceGateDenied("diagnosis_feedback_source_missing_or_changed")
    ordered = tuple(by_id[feedback_id] for feedback_id in feedback_ids)
    if ordered != tuple(
        sorted(
            ordered,
            key=lambda item: (_as_utc(item.created_at), item.feedback_id),
        )
    ):
        raise GovernanceGateDenied("diagnosis_feedback_source_missing_or_changed")
    for record, (_, expected_digest) in zip(ordered, normalized_refs, strict=True):
        if (
            record.tenant_id != candidate.tenant_id
            or record.incident_id != candidate.incident_id
            or _as_utc(record.created_at) > cutoff
            or record.content_digest != expected_digest
            or record_content_digest(record) != expected_digest
        ):
            raise GovernanceGateDenied(
                "diagnosis_feedback_source_missing_or_changed"
            )
    return ordered


def _require_incident_closed(
    session: Session,
    candidate: FeedbackCandidateRecord,
) -> None:
    status = session.scalar(
        select(IncidentRecord.status).where(
            IncidentRecord.tenant_id == candidate.tenant_id,
            IncidentRecord.incident_id == candidate.incident_id,
        )
    )
    if status is not None and status != "CLOSED":
        raise GovernanceGateDenied("incident_not_closed")
    if status is None:
        raise GovernanceGateDenied("authoritative_source_missing_or_changed")


def _candidate(
    session: Session,
    tenant_id: str,
    candidate_id: str,
) -> FeedbackCandidateRecord:
    candidate = session.scalar(
        select(FeedbackCandidateRecord).where(
            FeedbackCandidateRecord.tenant_id == tenant_id,
            FeedbackCandidateRecord.candidate_id == candidate_id,
        )
    )
    if candidate is None:
        raise GovernanceNotVisible
    return candidate


def _require_version(candidate: FeedbackCandidateRecord, expected_version: int) -> None:
    if candidate.version != expected_version:
        raise GovernanceConflict("candidate_version_conflict", candidate.version)


def _candidate_view(candidate: FeedbackCandidateRecord) -> CandidateGovernanceView:
    return CandidateGovernanceView(
        candidate_id=candidate.candidate_id,
        status=candidate.status,
        eligibility_status=candidate.eligibility_status,
        allow_training=candidate.allow_training,
        consent_basis=candidate.consent_basis,
        license_status=candidate.license_status,
        retention_until=candidate.retention_until,
        governance_blockers=tuple(candidate_governance_blockers(candidate, datetime.now(UTC))),
        version=candidate.version,
    )


def _dlp_view(result: DlpProcessingResultRecord, candidate_version: int) -> DlpRunView:
    return DlpRunView(
        dlp_result_id=result.dlp_result_id,
        candidate_id=result.candidate_id,
        status=result.status,
        policy_version=result.policy_version,
        source_content_hash=result.source_content_hash,
        content_hash=result.content_hash,
        redacted_content=dict(result.redacted_content),
        findings=list(result.findings),
        residual_entity_types=list(result.residual_entity_types),
        candidate_version=candidate_version,
    )


def _content_hash(content: Any) -> str:
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + sha256(encoded.encode()).hexdigest()


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
