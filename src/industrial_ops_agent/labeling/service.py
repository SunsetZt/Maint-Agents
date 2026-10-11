"""Eligibility-gated Label Studio creation and immutable consensus synchronization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, NoReturn

from sqlalchemy import select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.data_governance.dlp import PresidioDlpProcessor
from industrial_ops_agent.data_governance.service import (
    GovernanceConflict,
    GovernanceGateDenied,
    GovernanceNotVisible,
    authoritative_candidate_content,
    eligibility_gate_reason,
)
from industrial_ops_agent.labeling.label_studio import (
    LabelAnnotation,
    LabelStudioAdapter,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AnnotationReviewRecord,
    AnnotationRevisionRecord,
    AnnotationTaskRecord,
    DlpProcessingResultRecord,
    FeedbackCandidateRecord,
)


@dataclass(frozen=True, slots=True)
class AnnotationTaskView:
    task_id: str
    candidate_id: str
    status: str
    risk_level: str
    required_reviews: int
    completed_reviews: int
    external_task_id: str
    external_version: str
    version: int
    candidate_version: int
    requires_arbitration: bool


class LabelingService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        adapter: LabelStudioAdapter,
        dlp_processor: PresidioDlpProcessor,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._adapter = adapter
        self._dlp_processor = dlp_processor

    def create_task(
        self,
        identity: IdentityContext,
        candidate_id: str,
        *,
        expected_version: int,
        project_id: str,
        schema_version: str,
        risk_level: str,
        idempotency_key: str,
        request_id: str,
    ) -> AnnotationTaskView:
        self._authorizer.require(
            identity,
            Action.CREATE_ANNOTATION_TASK,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=candidate_id),
            request_id=request_id,
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(AnnotationTaskRecord).where(
                    AnnotationTaskRecord.tenant_id == identity.tenant_id,
                    AnnotationTaskRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.candidate_id != candidate_id:
                    raise GovernanceConflict("idempotency_key_reused", expected_version)
                candidate = _candidate(session, identity.tenant_id, candidate_id)
                return _task_view(session, existing, candidate.version)

            candidate = _candidate(session, identity.tenant_id, candidate_id)
            _require_version(candidate, expected_version)
            self._require_export_gate(identity, candidate, now, request_id)
            if candidate.status != "DLP_APPROVED":
                self._deny(identity, candidate_id, "dlp_not_approved", request_id)
            dlp_result = session.scalar(
                select(DlpProcessingResultRecord)
                .where(
                    DlpProcessingResultRecord.tenant_id == identity.tenant_id,
                    DlpProcessingResultRecord.candidate_id == candidate_id,
                    DlpProcessingResultRecord.status == "PASSED",
                )
                .order_by(DlpProcessingResultRecord.created_at.desc())
            )
            if dlp_result is None or dlp_result.residual_entity_types:
                self._deny(identity, candidate_id, "dlp_not_approved", request_id)
            try:
                _, current_source_hash = authoritative_candidate_content(session, candidate)
            except GovernanceGateDenied as exc:
                self._deny(identity, candidate_id, exc.reason, request_id)
            if current_source_hash != dlp_result.source_content_hash:
                self._deny(
                    identity,
                    candidate_id,
                    "authoritative_source_changed_after_dlp",
                    request_id,
                )

            payload = {
                "content": json.dumps(
                    dlp_result.redacted_content,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "schema_version": schema_version,
                "source_content_hash": dlp_result.source_content_hash,
            }
            payload_hash = _digest(payload)
            created = self._adapter.create_task(
                project_id=project_id,
                data=payload,
                source_data_version=payload_hash,
                idempotency_key=idempotency_key,
            )
            if created.payload_hash != payload_hash:
                self._deny(identity, candidate_id, "label_studio_payload_mismatch", request_id)

            task_id = (
                "annotation-task-"
                + sha256(f"{identity.tenant_id}\0{idempotency_key}".encode()).hexdigest()[:32]
            )
            required_reviews = 2 if risk_level == "HIGH" else 1
            task = AnnotationTaskRecord(
                task_id=task_id,
                tenant_id=identity.tenant_id,
                candidate_id=candidate_id,
                dlp_result_id=dlp_result.dlp_result_id,
                idempotency_key=idempotency_key,
                project_id=project_id,
                external_task_id=created.external_task_id,
                external_version=created.external_version,
                payload_hash=payload_hash,
                schema_version=schema_version,
                risk_level=risk_level,
                required_reviews=required_reviews,
                status="PENDING_REVIEW",
                consistency_status="PENDING",
                created_by_subject_id=identity.subject_id,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(task)
            candidate.status = "ANNOTATION_PENDING"
            candidate.version += 1
            candidate.updated_at = now
            session.flush()
            return _task_view(session, task, candidate.version)

    def sync_task(
        self,
        identity: IdentityContext,
        task_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> AnnotationTaskView:
        self._authorizer.require(
            identity,
            Action.SYNC_ANNOTATION_TASK,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=task_id),
            request_id=request_id,
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            task = _task(session, identity.tenant_id, task_id)
            if task.version != expected_version:
                raise GovernanceConflict("annotation_task_version_conflict", task.version)
            candidate = _candidate(session, identity.tenant_id, task.candidate_id)
            self._require_export_gate(identity, candidate, now, request_id)
            dlp_result = session.scalar(
                select(DlpProcessingResultRecord).where(
                    DlpProcessingResultRecord.tenant_id == identity.tenant_id,
                    DlpProcessingResultRecord.dlp_result_id == task.dlp_result_id,
                    DlpProcessingResultRecord.status == "PASSED",
                )
            )
            if dlp_result is None:
                self._deny(identity, candidate.candidate_id, "dlp_not_approved", request_id)
            try:
                _, current_source_hash = authoritative_candidate_content(session, candidate)
            except GovernanceGateDenied as exc:
                self._deny(identity, candidate.candidate_id, exc.reason, request_id)
            if current_source_hash != dlp_result.source_content_hash:
                self._deny(
                    identity,
                    candidate.candidate_id,
                    "authoritative_source_changed_after_dlp",
                    request_id,
                )

            snapshot = self._adapter.fetch_task(external_task_id=task.external_task_id)
            if snapshot.payload_hash != task.payload_hash:
                task.status = "REVIEW_REQUIRED"
                task.consistency_status = "EXTERNAL_PAYLOAD_DRIFT"
                task.external_version = snapshot.external_version
                task.version += 1
                task.updated_at = now
                candidate.status = "ANNOTATION_REVIEW_REQUIRED"
                candidate.version += 1
                candidate.updated_at = now
                session.flush()
                return _task_view(session, task, candidate.version, completed_reviews=0)

            active_annotations = _latest_by_reviewer(snapshot.annotations)
            for annotation in active_annotations:
                try:
                    _validate_labels(annotation.labels, self._dlp_processor)
                    revision = self._save_revision(session, task, annotation, now)
                    self._save_review(session, task, revision, annotation, now)
                except GovernanceGateDenied as exc:
                    self._deny(identity, task.task_id, exc.reason, request_id)
            session.flush()

            completed_reviews = len(active_annotations)
            label_digests = {_digest(annotation.labels) for annotation in active_annotations}
            if completed_reviews < task.required_reviews:
                status = "PENDING_REVIEW"
                consistency = "PENDING"
                candidate_status = "ANNOTATION_PENDING"
            elif len(label_digests) == 1:
                status = "APPROVED"
                consistency = "CONSENSUS"
                candidate_status = "READY_FOR_CURATION"
            else:
                status = "CONFLICT"
                consistency = "CONFLICT"
                candidate_status = "ANNOTATION_CONFLICT"
            task.status = status
            task.consistency_status = consistency
            task.external_version = snapshot.external_version
            task.version += 1
            task.updated_at = now
            candidate.status = candidate_status
            candidate.version += 1
            candidate.updated_at = now
            session.flush()
            return _task_view(
                session,
                task,
                candidate.version,
                completed_reviews=completed_reviews,
            )

    def _require_export_gate(
        self,
        identity: IdentityContext,
        candidate: FeedbackCandidateRecord,
        now: datetime,
        request_id: str,
    ) -> NoReturn:
        reason = eligibility_gate_reason(candidate, now)
        if reason is not None:
            self._deny(identity, candidate.candidate_id, reason, request_id)

    def _deny(
        self,
        identity: IdentityContext,
        resource_id: str,
        reason: str,
        request_id: str,
    ) -> None:
        self._authorizer.record_guard_decision(
            identity,
            action="annotation_task.governance_gate",
            decision="deny",
            reason_code=reason,
            request_id=request_id,
            resource_id=resource_id,
        )
        raise GovernanceGateDenied(reason)

    @staticmethod
    def _save_revision(
        session: Any,
        task: AnnotationTaskRecord,
        annotation: LabelAnnotation,
        now: datetime,
    ) -> AnnotationRevisionRecord:
        revision_id = (
            "annotation-revision-"
            + sha256(
                f"{task.tenant_id}\0{task.task_id}\0{annotation.external_annotation_id}\0"
                f"{annotation.external_version}".encode()
            ).hexdigest()[:32]
        )
        existing = session.get(AnnotationRevisionRecord, revision_id)
        labels_digest = _digest(annotation.labels)
        if existing is not None:
            if (
                existing.labels_digest != labels_digest
                or existing.reviewer_subject_id != annotation.reviewer_subject_id
                or existing.task_payload_hash != task.payload_hash
            ):
                raise GovernanceGateDenied("external_annotation_version_reused")
            return existing
        revision = AnnotationRevisionRecord(
            revision_id=revision_id,
            tenant_id=task.tenant_id,
            task_id=task.task_id,
            external_annotation_id=annotation.external_annotation_id,
            external_version=annotation.external_version,
            reviewer_subject_id=annotation.reviewer_subject_id,
            labels=annotation.labels,
            labels_digest=labels_digest,
            task_payload_hash=task.payload_hash,
            submitted_at=annotation.submitted_at,
            created_at=now,
            updated_at=now,
        )
        session.add(revision)
        return revision

    @staticmethod
    def _save_review(
        session: Any,
        task: AnnotationTaskRecord,
        revision: AnnotationRevisionRecord,
        annotation: LabelAnnotation,
        now: datetime,
    ) -> None:
        review_id = (
            "annotation-review-"
            + sha256(
                f"{task.tenant_id}\0{task.task_id}\0{revision.revision_id}".encode()
            ).hexdigest()[:32]
        )
        if session.get(AnnotationReviewRecord, review_id) is not None:
            return
        session.add(
            AnnotationReviewRecord(
                review_id=review_id,
                tenant_id=task.tenant_id,
                task_id=task.task_id,
                revision_id=revision.revision_id,
                reviewer_subject_id=annotation.reviewer_subject_id,
                outcome="SUBMITTED",
                consistency_key=revision.labels_digest,
                task_payload_hash=task.payload_hash,
                reviewed_at=annotation.submitted_at,
                created_at=now,
                updated_at=now,
            )
        )


def _candidate(session: Any, tenant_id: str, candidate_id: str) -> FeedbackCandidateRecord:
    candidate = session.scalar(
        select(FeedbackCandidateRecord).where(
            FeedbackCandidateRecord.tenant_id == tenant_id,
            FeedbackCandidateRecord.candidate_id == candidate_id,
        )
    )
    if candidate is None:
        raise GovernanceNotVisible
    return candidate


def _task(session: Any, tenant_id: str, task_id: str) -> AnnotationTaskRecord:
    task = session.scalar(
        select(AnnotationTaskRecord).where(
            AnnotationTaskRecord.tenant_id == tenant_id,
            AnnotationTaskRecord.task_id == task_id,
        )
    )
    if task is None:
        raise GovernanceNotVisible
    return task


def _require_version(candidate: FeedbackCandidateRecord, expected_version: int) -> None:
    if candidate.version != expected_version:
        raise GovernanceConflict("candidate_version_conflict", candidate.version)


def _latest_by_reviewer(
    annotations: tuple[LabelAnnotation, ...],
) -> tuple[LabelAnnotation, ...]:
    latest: dict[str, LabelAnnotation] = {}
    for annotation in annotations:
        current = latest.get(annotation.reviewer_subject_id)
        if current is None or _aware(annotation.submitted_at) > _aware(current.submitted_at):
            latest[annotation.reviewer_subject_id] = annotation
    return tuple(latest[key] for key in sorted(latest))


def _validate_labels(
    labels: dict[str, Any],
    dlp_processor: PresidioDlpProcessor,
) -> None:
    encoded = json.dumps(labels, ensure_ascii=False, sort_keys=True)
    forbidden_keys = {"raw", "raw_text", "prompt", "source_text", "secret"}
    if len(encoded) > 16_384 or forbidden_keys.intersection(labels):
        raise GovernanceGateDenied("annotation_payload_forbidden")
    output = dlp_processor.process({"annotation": encoded})
    if output.findings or output.status != "PASSED":
        raise GovernanceGateDenied("annotation_contains_sensitive_content")


def _task_view(
    session: Any,
    task: AnnotationTaskRecord,
    candidate_version: int,
    *,
    completed_reviews: int | None = None,
) -> AnnotationTaskView:
    if completed_reviews is None:
        completed_reviews = len(
            {
                value
                for value in session.scalars(
                    select(AnnotationReviewRecord.reviewer_subject_id).where(
                        AnnotationReviewRecord.tenant_id == task.tenant_id,
                        AnnotationReviewRecord.task_id == task.task_id,
                        AnnotationReviewRecord.task_payload_hash == task.payload_hash,
                    )
                )
            }
        )
    return AnnotationTaskView(
        task_id=task.task_id,
        candidate_id=task.candidate_id,
        status=task.status,
        risk_level=task.risk_level,
        required_reviews=task.required_reviews,
        completed_reviews=completed_reviews,
        external_task_id=task.external_task_id,
        external_version=task.external_version,
        version=task.version,
        candidate_version=candidate_version,
        requires_arbitration=task.status == "CONFLICT",
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(encoded.encode()).hexdigest()


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
