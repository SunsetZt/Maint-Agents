"""Source-scoped maintenance review facts; callers own authorization and transactions.

Synchronous operations belong inside a short tenant transaction in a worker thread.
Subject locking precedes evaluation/assignment locks. Never hold it while streaming.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.domain import as_utc
from industrial_ops_agent.domain.json import strict_document_digest
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    DiagnosisExpertInterventionRecord,
    DiagnosisExpertRevisionRecord,
    DiagnosisQualityFeedbackRecord,
    DiagnosisRunRecord,
    IdempotencyRecord,
    IncidentDraftRecord,
    IncidentRecord,
    IncidentRelationRecord,
    MaintenancePlanningBlindJudgmentRecord,
    MaintenancePlanningCouncilRecord,
    MaintenancePlanningEvaluationRunRecord,
    MaintenancePlanningEvaluationSuiteRecord,
    MaintenanceReviewAssignmentRecord,
    MaintenanceReviewExposureRecord,
    MaintenanceReviewPolicyRecord,
    ModelInferenceRecord,
    SubjectRecord,
    WorkOrderExpertCollaborationRecommendationRecord,
    WorkOrderExpertCollaborationRecord,
    WorkOrderRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext

REVIEW_POLICY_VERSION = "maintenance-planning-blind-ab-v2"
READER_COVERAGE_DIGEST = strict_document_digest(
    {
        "schema": "maintenance-review-reader-coverage-v2",
        "source_root": "incident-and-draft",
        "revision": "identified-readers-v2",
    }
)


class ReviewIsolationConflict(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = f"maintenance_review_{reason}"
        super().__init__(self.reason)


@dataclass(frozen=True, slots=True)
class ReviewSource:
    tenant_id: str
    incident_id: str | None
    diagnosis_run_id: str | None
    asset_id: str
    family_key: str
    version_digest: str
    created_at: datetime


def lock_subject(
    session: Session, identity: IdentityContext | TenantContext, *, lineage_write: bool = False
) -> SubjectRecord:
    """Lock a verified OIDC or capability-bound subject; never grants authorization.

    TenantContext callers must first validate their capability and derive its actual
    subject from the persisted binding, not from a token hint or request parameter.
    """
    subject = session.scalar(
        select(SubjectRecord)
        .where(
            SubjectRecord.tenant_id == identity.tenant_id,
            SubjectRecord.subject_id == identity.subject_id,
        )
        .with_for_update()
    )
    if subject is None or subject.status != "active":
        raise ReviewIsolationConflict("subject_unavailable")
    # Readers share this lock; duplicate merges take it exclusively before any source read.
    # A stable tenant key also serializes concurrent first edges across disjoint roots.
    if session.get_bind().dialect.name == "postgresql":
        digest = strict_document_digest(
            {"schema": "maintenance-review-lineage-lock-v1", "tenant": identity.tenant_id}
        )
        key = int(digest.removeprefix("sha256:")[:16], 16)
        if key >= 2**63:
            key -= 2**64
        query = (
            "SELECT pg_advisory_xact_lock(:key)"
            if lineage_write
            else "SELECT pg_advisory_xact_lock_shared(:key)"
        )
        session.execute(text(query), {"key": key})
    return subject


def resolve_source(session: Session, tenant_id: str, diagnosis_run_id: str) -> ReviewSource:
    """Derive identity from persisted lineage, never a client-provided family key."""
    pair = session.execute(
        select(DiagnosisRunRecord, IncidentRecord)
        .join(
            IncidentRecord,
            (IncidentRecord.tenant_id == DiagnosisRunRecord.tenant_id)
            & (IncidentRecord.incident_id == DiagnosisRunRecord.incident_id),
        )
        .where(
            DiagnosisRunRecord.tenant_id == tenant_id,
            DiagnosisRunRecord.diagnosis_run_id == diagnosis_run_id,
        )
    ).one_or_none()
    if pair is None:
        raise ReviewIsolationConflict("source_unavailable")
    diagnosis, incident = pair
    councils = session.scalars(
        select(MaintenancePlanningCouncilRecord)
        .where(
            MaintenancePlanningCouncilRecord.tenant_id == tenant_id,
            MaintenancePlanningCouncilRecord.diagnosis_run_id == diagnosis_run_id,
        )
        .order_by(MaintenancePlanningCouncilRecord.council_id)
    ).all()
    if any(
        c.incident_id != incident.incident_id or c.asset_id != incident.asset_id for c in councils
    ):
        raise ReviewIsolationConflict("source_binding_invalid")
    family_key = _incident_family_key(tenant_id, incident.incident_id)
    version_digest = strict_document_digest(
        {
            "schema": "maintenance-review-source-v1",
            "family_key": family_key,
            "diagnosis_run_id": diagnosis_run_id,
            "diagnosis_version": diagnosis.version,
            "report": diagnosis.report,
            "manifest": diagnosis.manifest,
            "councils": [
                {
                    "council_id": c.council_id,
                    "version": c.version,
                    "attempt": c.attempt_count,
                    "plan": c.plan_json,
                    "result_digest": c.result_digest,
                    "diagnosis_report_digest": c.diagnosis_report_digest,
                    "diagnosis_manifest_digest": c.diagnosis_manifest_digest,
                    "model_release_id": c.model_release_id,
                    "model_manifest_hash": c.model_manifest_hash,
                }
                for c in councils
            ],
        }
    )
    return ReviewSource(
        tenant_id,
        incident.incident_id,
        diagnosis_run_id,
        incident.asset_id,
        family_key,
        version_digest,
        as_utc(incident.created_at),
    )


def _incident_family_key(tenant_id: str, incident_id: str) -> str:
    return strict_document_digest(
        {
            "schema": "maintenance-review-family-v1",
            "tenant_id": tenant_id,
            "incident_id": incident_id,
        }
    )


def _draft_family_key(tenant_id: str, draft_id: str) -> str:
    return strict_document_digest(
        {
            "schema": "maintenance-review-draft-family-v1",
            "tenant_id": tenant_id,
            "draft_id": draft_id,
        }
    )


def _incident_family(session: Session, source: ReviewSource) -> tuple[IncidentRecord, ...]:
    """Read both sides of duplicate merges without replacing historical family keys."""
    if source.incident_id is None:
        raise ReviewIsolationConflict("source_unavailable")
    visited = {source.incident_id}
    frontier = set(visited)
    while frontier:
        edges = session.execute(
            select(
                IncidentRelationRecord.source_incident_id,
                IncidentRelationRecord.target_incident_id,
            ).where(
                IncidentRelationRecord.tenant_id == source.tenant_id,
                IncidentRelationRecord.relation_type == "DUPLICATE_OF",
                IncidentRelationRecord.source_incident_id.in_(frontier)
                | IncidentRelationRecord.target_incident_id.in_(frontier),
            )
        ).all()
        frontier = {value for edge in edges for value in edge} - visited
        visited.update(frontier)
    incidents = tuple(
        session.scalars(
            select(IncidentRecord)
            .where(
                IncidentRecord.tenant_id == source.tenant_id,
                IncidentRecord.incident_id.in_(visited),
            )
            .order_by(IncidentRecord.incident_id)
        )
    )
    if len(incidents) != len(visited) or any(
        incident.asset_id != source.asset_id for incident in incidents
    ):
        raise ReviewIsolationConflict("source_binding_invalid")
    return incidents


def source_family_keys(session: Session, source: ReviewSource) -> tuple[str, ...]:
    """Include every merged Incident and pre-submission root, retaining the original keys."""
    if source.incident_id is None:
        return (source.family_key,)
    return tuple(
        sorted(
            {
                key
                for incident in _incident_family(session, source)
                for key in (
                    _incident_family_key(source.tenant_id, incident.incident_id),
                    _draft_family_key(source.tenant_id, incident.source_draft_id),
                )
            }
        )
    )


def record_draft_disclosure(
    session: Session,
    identity: IdentityContext,
    draft_id: str,
    *,
    request_id: str,
) -> None:
    lock_subject(session, identity)
    draft = session.scalar(
        select(IncidentDraftRecord)
        .where(
            IncidentDraftRecord.tenant_id == identity.tenant_id,
            IncidentDraftRecord.draft_id == draft_id,
        )
        .execution_options(populate_existing=True)
    )
    if draft is None:
        raise ReviewIsolationConflict("source_unavailable")
    source = ReviewSource(
        identity.tenant_id,
        None,
        None,
        draft.asset_id,
        _draft_family_key(identity.tenant_id, draft_id),
        strict_document_digest(
            {
                "schema": "maintenance-review-draft-content-v1",
                "draft_id": draft_id,
                "version": draft.version,
                "description": draft.description,
            }
        ),
        as_utc(draft.created_at),
    )
    record_exposure(
        session,
        identity,
        source,
        reason="IDENTIFIED_READ",
        resource_kind="incident-draft",
        resource_id=draft_id,
        request_id=request_id,
    )
    # An already-submitted draft must also respect claims made against the incident key.
    incidents = session.scalars(
        select(IncidentRecord.incident_id).where(
            IncidentRecord.tenant_id == identity.tenant_id,
            IncidentRecord.source_draft_id == draft_id,
        )
    ).all()
    record_incident_disclosure(
        session,
        identity,
        incidents,
        resource_kind="incident-draft",
        resource_id=draft_id,
        request_id=request_id,
    )


@contextmanager
def draft_read_transaction(
    database: Database,
    identity: IdentityContext,
    draft_id: str,
    *,
    request_id: str,
) -> Iterator[Session]:
    """Human-only boundary; authorization remains in the caller, writes roll back on denial."""
    with database.transaction(identity.tenant_context) as session:
        lock_subject(session, identity)
        yield session
        record_draft_disclosure(session, identity, draft_id, request_id=request_id)


@contextmanager
def incident_read_transaction(
    database: Database,
    identity: IdentityContext,
    incident_ids: Iterable[str],
    *,
    request_id: str,
    lineage_write: bool = False,
) -> Iterator[Session]:
    ids = tuple(sorted(set(incident_ids)))
    with database.transaction(identity.tenant_context) as session:
        lock_subject(session, identity, lineage_write=lineage_write)
        yield session
        record_incident_disclosure(
            session,
            identity,
            ids,
            resource_kind="incident",
            resource_id=ids[0] if len(ids) == 1 else strict_document_digest(list(ids)),
            request_id=request_id,
        )


def record_incident_disclosure(
    session: Session,
    identity: IdentityContext | TenantContext,
    incident_ids: Iterable[str],
    *,
    resource_kind: str,
    resource_id: str,
    request_id: str,
) -> None:
    """Record the actual Incident lineage, including legacy work without a diagnosis FK."""
    ids = sorted(set(incident_ids))
    if not ids:
        return
    lock_subject(session, identity)
    for incident_id in ids:
        incident = session.scalar(
            select(IncidentRecord)
            .where(
                IncidentRecord.tenant_id == identity.tenant_id,
                IncidentRecord.incident_id == incident_id,
            )
            .execution_options(populate_existing=True)
        )
        if incident is None:
            raise ReviewIsolationConflict("source_unavailable")
        diagnosis_ids = session.scalars(
            select(DiagnosisRunRecord.diagnosis_run_id)
            .where(
                DiagnosisRunRecord.tenant_id == identity.tenant_id,
                DiagnosisRunRecord.incident_id == incident_id,
            )
            .order_by(DiagnosisRunRecord.diagnosis_run_id)
        )
        sources = tuple(
            resolve_source(session, identity.tenant_id, value) for value in diagnosis_ids
        )
        source = ReviewSource(
            identity.tenant_id,
            incident_id,
            None,
            incident.asset_id,
            _incident_family_key(identity.tenant_id, incident_id),
            strict_document_digest(
                {
                    "schema": "maintenance-review-incident-v1",
                    "incident_id": incident_id,
                    "version": incident.version,
                    "description": incident.description,
                    "evidence_bundle_id": incident.evidence_bundle_id,
                    "diagnoses": [item.version_digest for item in sources],
                }
            ),
            as_utc(incident.created_at),
        )
        record_exposure(
            session,
            identity,
            source,
            reason="IDENTIFIED_READ",
            resource_kind=resource_kind,
            resource_id=resource_id,
            request_id=request_id,
        )


@contextmanager
def work_order_read_transaction(
    database: Database,
    identity: IdentityContext,
    work_order_id: str,
    *,
    request_id: str,
) -> Iterator[Session]:
    """Human-facing work-order calls only; internal Workers keep their own transactions.

    The caller authorizes and constructs the result before leaving the context.
    A blocked disclosure rolls back its writes too. Subject lock precedes work locks.
    """
    with database.transaction(identity.tenant_context) as session:
        lock_subject(session, identity)
        yield session
        work = session.scalar(
            select(WorkOrderRecord).where(
                WorkOrderRecord.tenant_id == identity.tenant_id,
                WorkOrderRecord.work_order_id == work_order_id,
            )
        )
        if work is None:
            raise ReviewIsolationConflict("source_unavailable")
        if work.diagnosis_run_id is not None:
            source = resolve_source(session, identity.tenant_id, work.diagnosis_run_id)
            if source.incident_id != work.incident_id:
                raise ReviewIsolationConflict("source_binding_invalid")
        record_incident_disclosure(
            session,
            identity,
            (work.incident_id,),
            resource_kind="work-order",
            resource_id=work_order_id,
            request_id=request_id,
        )


def _assignments(
    session: Session, identity: IdentityContext | TenantContext, family_keys: Iterable[str]
) -> list[MaintenanceReviewAssignmentRecord]:
    wanted = set(family_keys)
    rows = session.scalars(
        select(MaintenanceReviewAssignmentRecord)
        .where(
            MaintenanceReviewAssignmentRecord.tenant_id == identity.tenant_id,
            MaintenanceReviewAssignmentRecord.subject_id == identity.subject_id,
        )
        .order_by(MaintenanceReviewAssignmentRecord.claim_id)
    ).all()
    return [row for row in rows if not wanted.isdisjoint(row.source_family_keys)]


def check_disclosure(
    session: Session, identity: IdentityContext, diagnosis_ids: Iterable[str]
) -> tuple[ReviewSource, ...]:
    """Serialize with claims and resolve the complete response before any release."""
    ids = sorted(set(diagnosis_ids))
    if not ids:
        return ()
    lock_subject(session, identity)
    sources = tuple(resolve_source(session, identity.tenant_id, value) for value in ids)
    for source in sources:
        if any(
            row.state == "ACTIVE"
            for row in _assignments(session, identity, source_family_keys(session, source))
        ):
            raise ReviewIsolationConflict("source_read_blocked")
    return sources


def record_disclosure(
    session: Session,
    identity: IdentityContext,
    diagnosis_ids: Iterable[str],
    *,
    resource_kind: str,
    resource_id: str,
    request_id: str,
) -> None:
    """Commit these facts in the caller's transaction before returning identified data."""
    for source in check_disclosure(session, identity, diagnosis_ids):
        record_exposure(
            session,
            identity,
            source,
            reason="IDENTIFIED_READ",
            resource_kind=resource_kind,
            resource_id=resource_id,
            request_id=request_id,
        )


def diagnosis_for_agent(session: Session, tenant_id: str, agent_run_id: str) -> str:
    diagnosis_id = session.scalar(
        select(DiagnosisRunRecord.diagnosis_run_id).where(
            DiagnosisRunRecord.tenant_id == tenant_id,
            DiagnosisRunRecord.agent_run_id == agent_run_id,
        )
    )
    if diagnosis_id is None:
        raise ReviewIsolationConflict("source_unavailable")
    return diagnosis_id


def record_exposure(
    session: Session,
    identity: IdentityContext | TenantContext,
    source: ReviewSource,
    *,
    reason: str,
    resource_kind: str,
    resource_id: str,
    request_id: str,
) -> MaintenanceReviewExposureRecord:
    """Write before releasing content. Errors propagate so callers cannot leak it."""
    lock_subject(session, identity)
    if source.tenant_id != identity.tenant_id:
        raise ReviewIsolationConflict("source_unavailable")
    if any(
        row.state == "ACTIVE"
        for row in _assignments(session, identity, source_family_keys(session, source))
    ):
        raise ReviewIsolationConflict("source_read_blocked")
    digest = strict_document_digest(
        {
            "subject_id": identity.subject_id,
            "family": source.family_key,
            "version": source.version_digest,
            "reason": reason,
            "resource_kind": resource_kind,
            "resource_id": resource_id,
            "request_id": request_id,
        }
    )
    existing = session.scalar(
        select(MaintenanceReviewExposureRecord).where(
            MaintenanceReviewExposureRecord.tenant_id == identity.tenant_id,
            MaintenanceReviewExposureRecord.subject_id == identity.subject_id,
            MaintenanceReviewExposureRecord.dedup_key == digest,
        )
    )
    if existing is not None:
        return existing
    now = datetime.now(UTC)
    record = MaintenanceReviewExposureRecord(
        exposure_id=f"review-exposure-{uuid4().hex}",
        tenant_id=identity.tenant_id,
        subject_id=identity.subject_id,
        source_family_key=source.family_key,
        source_version_digest=source.version_digest,
        reason=reason,
        resource_kind=resource_kind,
        resource_id=resource_id,
        request_id=request_id,
        occurred_at=now,
        dedup_key=digest,
        created_at=now,
        updated_at=now,
    )
    session.add(record)
    session.flush()
    return record


def _known_participation(
    session: Session,
    identity: IdentityContext,
    source: ReviewSource,
    incidents: tuple[IncidentRecord, ...],
) -> tuple[str, str, str] | None:
    """Supplement actual authorship/review facts, not an invented historical read."""
    if source.incident_id is None:
        raise ReviewIsolationConflict("source_unavailable")
    tenant, subject = identity.tenant_id, identity.subject_id
    incident_ids = {incident.incident_id for incident in incidents}
    diagnoses = select(DiagnosisRunRecord.diagnosis_run_id).where(
        DiagnosisRunRecord.tenant_id == tenant,
        DiagnosisRunRecord.incident_id.in_(incident_ids),
    )
    for incident in incidents:
        if incident.reporter_subject_id == subject:
            return "INCIDENT_REPORTER", "incident", incident.incident_id
    drafts = session.scalars(
        select(IncidentDraftRecord).where(
            IncidentDraftRecord.tenant_id == tenant,
            IncidentDraftRecord.draft_id.in_({incident.source_draft_id for incident in incidents}),
        )
    )
    for draft in drafts:
        if draft.author_subject_id == subject:
            return "DRAFT_AUTHOR", "incident-draft", draft.draft_id
    collaboration = session.scalar(
        select(WorkOrderExpertCollaborationRecord)
        .where(
            WorkOrderExpertCollaborationRecord.tenant_id == tenant,
            WorkOrderExpertCollaborationRecord.incident_id.in_(incident_ids),
            (WorkOrderExpertCollaborationRecord.requested_by_subject_id == subject)
            | (
                (WorkOrderExpertCollaborationRecord.invited_subject_id == subject)
                & WorkOrderExpertCollaborationRecord.accepted_at.is_not(None)
            )
            | (WorkOrderExpertCollaborationRecord.ended_by_subject_id == subject),
        )
        .order_by(WorkOrderExpertCollaborationRecord.collaboration_id)
        .limit(1)
    )
    if collaboration is not None:
        if collaboration.requested_by_subject_id == subject:
            reason = "EXPERT_COLLABORATION_REQUESTER"
        elif collaboration.invited_subject_id == subject and collaboration.accepted_at is not None:
            reason = "EXPERT_COLLABORATION_PARTICIPANT"
        else:
            reason = "EXPERT_COLLABORATION_ENDER"
        return reason, "expert-collaboration", collaboration.collaboration_id
    recommendation = session.scalar(
        select(WorkOrderExpertCollaborationRecommendationRecord)
        .where(
            WorkOrderExpertCollaborationRecommendationRecord.tenant_id == tenant,
            WorkOrderExpertCollaborationRecommendationRecord.incident_id.in_(incident_ids),
            (WorkOrderExpertCollaborationRecommendationRecord.expert_subject_id == subject)
            | (WorkOrderExpertCollaborationRecommendationRecord.reviewed_by_subject_id == subject),
        )
        .order_by(WorkOrderExpertCollaborationRecommendationRecord.recommendation_id)
        .limit(1)
    )
    if recommendation is not None:
        reason = (
            "EXPERT_RECOMMENDATION_AUTHOR"
            if recommendation.expert_subject_id == subject
            else "EXPERT_RECOMMENDATION_REVIEWER"
        )
        return reason, "expert-recommendation", recommendation.recommendation_id
    councils = session.scalars(
        select(MaintenancePlanningCouncilRecord)
        .where(
            MaintenancePlanningCouncilRecord.tenant_id == tenant,
            MaintenancePlanningCouncilRecord.diagnosis_run_id.in_(diagnoses),
        )
        .order_by(MaintenancePlanningCouncilRecord.council_id)
    ).all()
    for council in councils:
        if council.requested_by_subject_id == subject:
            return "COUNCIL_REQUESTER", "council", council.council_id
        if council.reviewed_by_subject_id == subject:
            return "COUNCIL_REVIEWER", "council", council.council_id
    intervention_id = session.scalar(
        select(DiagnosisExpertInterventionRecord.intervention_id)
        .where(
            DiagnosisExpertInterventionRecord.tenant_id == tenant,
            DiagnosisExpertInterventionRecord.diagnosis_run_id.in_(diagnoses),
            DiagnosisExpertInterventionRecord.requested_by_subject_id == subject,
        )
        .order_by(DiagnosisExpertInterventionRecord.intervention_id)
        .limit(1)
    )
    if intervention_id is not None:
        return "DIAGNOSIS_EXPERT_REQUESTER", "diagnosis-expert-intervention", intervention_id
    revision_id = session.scalar(
        select(DiagnosisExpertRevisionRecord.revision_id)
        .where(
            DiagnosisExpertRevisionRecord.tenant_id == tenant,
            DiagnosisExpertRevisionRecord.diagnosis_run_id.in_(diagnoses),
            DiagnosisExpertRevisionRecord.authored_by_subject_id == subject,
        )
        .order_by(DiagnosisExpertRevisionRecord.revision_id)
        .limit(1)
    )
    if revision_id is not None:
        return "DIAGNOSIS_EXPERT_AUTHOR", "diagnosis-expert-revision", revision_id
    feedback = session.scalar(
        select(DiagnosisQualityFeedbackRecord.feedback_id)
        .where(
            DiagnosisQualityFeedbackRecord.tenant_id == tenant,
            DiagnosisQualityFeedbackRecord.diagnosis_run_id.in_(diagnoses),
            DiagnosisQualityFeedbackRecord.submitted_by_subject_id == subject,
        )
        .order_by(DiagnosisQualityFeedbackRecord.feedback_id)
        .limit(1)
    )
    if feedback is not None:
        return "DIAGNOSIS_REVIEWER", "diagnosis-feedback", feedback
    inference = session.scalar(
        select(ModelInferenceRecord.inference_request_id)
        .where(
            ModelInferenceRecord.tenant_id == tenant,
            ModelInferenceRecord.diagnosis_run_id.in_(diagnoses),
            ModelInferenceRecord.subject_id == subject,
        )
        .order_by(ModelInferenceRecord.inference_request_id)
        .limit(1)
    )
    if inference is not None:
        return "INFERENCE_REQUESTER", "inference", inference
    requested = session.scalar(
        select(IdempotencyRecord.result_ref)
        .where(
            IdempotencyRecord.tenant_id == tenant,
            IdempotencyRecord.subject_id == subject,
            IdempotencyRecord.result_ref.in_(diagnoses),
        )
        .order_by(IdempotencyRecord.result_ref)
        .limit(1)
    )
    if requested is not None:
        return "DIAGNOSIS_REQUESTER", "diagnosis", requested
    for suite in session.scalars(
        select(MaintenancePlanningEvaluationSuiteRecord)
        .where(
            MaintenancePlanningEvaluationSuiteRecord.tenant_id == tenant,
            MaintenancePlanningEvaluationSuiteRecord.created_by_subject_id == subject,
        )
        .order_by(MaintenancePlanningEvaluationSuiteRecord.suite_id)
    ):
        if any(case["incident_id"] in incident_ids for case in suite.cases_json):
            return "SUITE_AUTHOR", "evaluation-suite", suite.suite_id
    for run in session.scalars(
        select(MaintenancePlanningEvaluationRunRecord)
        .where(
            MaintenancePlanningEvaluationRunRecord.tenant_id == tenant,
        )
        .order_by(MaintenancePlanningEvaluationRunRecord.evaluation_id)
    ):
        cases = {case["case_id"] for case in run.pairs_json if case["incident_id"] in incident_ids}
        if not cases:
            continue
        if run.requested_by_subject_id == subject:
            return "EVALUATION_REQUESTER", "evaluation", run.evaluation_id
        judgment = session.scalar(
            select(MaintenancePlanningBlindJudgmentRecord.judgment_id)
            .where(
                MaintenancePlanningBlindJudgmentRecord.tenant_id == tenant,
                MaintenancePlanningBlindJudgmentRecord.evaluation_id == run.evaluation_id,
                MaintenancePlanningBlindJudgmentRecord.case_id.in_(cases),
                MaintenancePlanningBlindJudgmentRecord.judge_subject_id == subject,
            )
            .limit(1)
        )
        if judgment is not None:
            return "PREVIOUS_JUDGMENT", "judgment", judgment
    return None


def eligibility(
    session: Session,
    identity: IdentityContext,
    source: ReviewSource,
    *,
    reader_coverage_digest: str,
    active_claim_id: str | None = None,
) -> str:
    """Admission input, not authorization or a replacement for claiming/submitting.

    The expected reader coverage is supplied by server code, never an HTTP payload.
    Known participation is stored in the caller's transaction even when ineligible.
    """
    lock_subject(session, identity)
    if source.tenant_id != identity.tenant_id:
        raise ReviewIsolationConflict("source_unavailable")
    families = source_family_keys(session, source)
    exposed = session.scalar(
        select(MaintenanceReviewExposureRecord.exposure_id)
        .where(
            MaintenanceReviewExposureRecord.tenant_id == identity.tenant_id,
            MaintenanceReviewExposureRecord.subject_id == identity.subject_id,
            MaintenanceReviewExposureRecord.source_family_key.in_(families),
        )
        .limit(1)
    )
    if exposed is not None:
        return "EXPOSED"
    if any(
        row.claim_id != active_claim_id
        for row in _assignments(session, identity, families)
    ):
        return "ALREADY_PARTICIPATED"
    incidents = _incident_family(session, source)
    known = _known_participation(session, identity, source, incidents)
    if known is not None:
        reason, kind, resource_id = known
        record_exposure(
            session,
            identity,
            source,
            reason=reason,
            resource_kind=kind,
            resource_id=resource_id,
            request_id="known-participation",
        )
        return "EXPOSED"
    policy = session.scalar(
        select(MaintenanceReviewPolicyRecord).where(
            MaintenanceReviewPolicyRecord.tenant_id == identity.tenant_id,
        )
    )
    if (
        policy is None
        or policy.cutover_at is None
        or policy.policy_version != REVIEW_POLICY_VERSION
        or policy.reader_coverage_digest != reader_coverage_digest
    ):
        return "POLICY_INACTIVE"
    from industrial_ops_agent.maintenance_planning.rollout_evidence import require_policy_activation

    try:
        require_policy_activation(session, policy)
    except ReviewIsolationConflict:
        return "POLICY_INACTIVE"
    draft_ids = {incident.source_draft_id for incident in incidents}
    drafts = session.scalars(
        select(IncidentDraftRecord).where(
            IncidentDraftRecord.tenant_id == identity.tenant_id,
            IncidentDraftRecord.draft_id.in_(draft_ids),
        )
    ).all()
    if (
        len(drafts) != len(draft_ids)
        or any(draft.asset_id != source.asset_id for draft in drafts)
        or any(
            as_utc(created_at) <= as_utc(policy.cutover_at)
            for created_at in (
                *(incident.created_at for incident in incidents),
                *(draft.created_at for draft in drafts),
            )
        )
    ):
        return "UNKNOWN_HISTORY"
    return "ELIGIBLE"
