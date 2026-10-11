"""Governed tenant-level activation for evaluated maintenance-planning councils."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.maintenance_planning.evaluation import require_review_report
from industrial_ops_agent.maintenance_planning.review_isolation import (
    lock_subject,
    record_disclosure,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    MaintenancePlanningActivationRecord,
    MaintenancePlanningEvaluationRunRecord,
    MaintenancePlanningEvaluationSuiteRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext

ActivationTarget = Literal["PROJECT_STAGING", "PRODUCTION"]
ActivationDecision = Literal["APPROVE", "REJECT"]

TARGET_ENVIRONMENTS = frozenset({"PROJECT_STAGING", "PRODUCTION"})
ACTIVATION_DECISIONS = frozenset({"APPROVE", "REJECT"})


class MaintenancePlanningActivationNotVisible(Exception):
    """The activation is absent or outside the caller's tenant boundary."""


class MaintenancePlanningActivationConflict(RuntimeError):
    """The requested activation transition violates a closed governance rule."""

    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class MaintenancePlanningActivationBinding:
    activation_id: str
    activation_policy_hash: str
    target_environment: str
    evaluation_id: str
    evaluation_report_hash: str


@dataclass(frozen=True, slots=True)
class MaintenancePlanningActivationView:
    activation_id: str
    evaluation_id: str
    suite_id: str
    target_environment: str
    status: str
    evaluation_decision: str
    evaluation_report_hash: str
    evaluation_policy_hash: str
    suite_manifest_hash: str
    activation_policy_hash: str
    request_reason: str
    requested_by_subject_id: str
    decided_by_subject_id: str | None
    decision: str | None
    decision_reason: str | None
    activated_at: datetime | None
    rolled_back_by_subject_id: str | None
    rollback_reason: str | None
    rolled_back_at: datetime | None
    version: int
    legal_actions: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


class MaintenancePlanningActivationService:
    """Request, approve, resolve, and roll back one tenant-scoped rollout."""

    def __init__(self, database: Database, authorizer: Authorizer | None = None) -> None:
        self._database = database
        self._authorizer = authorizer

    def request(
        self,
        identity: IdentityContext,
        *,
        evaluation_id: str,
        target_environment: str,
        reason: str,
        idempotency_key: str,
        request_id: str,
    ) -> tuple[MaintenancePlanningActivationView, bool]:
        normalized_evaluation_id = evaluation_id.strip()
        target = _target(target_environment)
        normalized_reason = _reason(reason)
        if not normalized_evaluation_id or not 8 <= len(idempotency_key) <= 255:
            raise MaintenancePlanningActivationConflict(
                "maintenance_planning_activation_request_invalid"
            )
        self._require(
            identity,
            Action.REQUEST_MAINTENANCE_COUNCIL,
            normalized_evaluation_id,
            request_id,
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            evaluation, suite = _eligible_evaluation(
                session,
                identity.tenant_id,
                normalized_evaluation_id,
                target,
                require_isolation=True,
            )
            policy_hash = _activation_policy_hash(evaluation, suite, target)
            request_hash = _digest(
                {
                    "evaluation_id": evaluation.evaluation_id,
                    "target_environment": target,
                    "activation_policy_hash": policy_hash,
                    "reason": normalized_reason,
                }
            )
            replay = session.scalar(
                select(MaintenancePlanningActivationRecord).where(
                    MaintenancePlanningActivationRecord.tenant_id == identity.tenant_id,
                    MaintenancePlanningActivationRecord.requested_by_subject_id
                    == identity.subject_id,
                    MaintenancePlanningActivationRecord.idempotency_key == idempotency_key,
                )
            )
            if replay is not None:
                if replay.request_hash != request_hash:
                    raise MaintenancePlanningActivationConflict(
                        "maintenance_planning_activation_idempotency_conflict",
                        replay.version,
                    )
                return self._view(replay, identity, session=session, request_id=request_id), False
            open_request = session.scalar(
                select(MaintenancePlanningActivationRecord).where(
                    MaintenancePlanningActivationRecord.tenant_id == identity.tenant_id,
                    MaintenancePlanningActivationRecord.target_environment == target,
                    MaintenancePlanningActivationRecord.status.in_(("PENDING_APPROVAL", "ACTIVE")),
                )
            )
            if open_request is not None:
                raise MaintenancePlanningActivationConflict(
                    "maintenance_planning_activation_target_already_open",
                    open_request.version,
                )
            record = MaintenancePlanningActivationRecord(
                activation_id=f"maintenance-planning-activation-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                evaluation_id=evaluation.evaluation_id,
                suite_id=suite.suite_id,
                target_environment=target,
                status="PENDING_APPROVAL",
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                evaluation_decision=evaluation.decision,
                evaluation_report_hash=str(evaluation.report_hash),
                evaluation_policy_hash=evaluation.policy_hash,
                suite_manifest_hash=suite.manifest_hash,
                activation_policy_hash=policy_hash,
                request_reason=normalized_reason,
                requested_by_subject_id=identity.subject_id,
                decided_by_subject_id=None,
                decision=None,
                decision_reason=None,
                activated_at=None,
                rolled_back_by_subject_id=None,
                rollback_reason=None,
                rolled_back_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            return self._view(record, identity, session=session, request_id=request_id), True

    def list(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        target_environment: str | None = None,
    ) -> tuple[MaintenancePlanningActivationView, ...]:
        self._require(
            identity,
            Action.READ_MAINTENANCE_COUNCIL,
            "maintenance-planning-activations",
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            statement = select(MaintenancePlanningActivationRecord).where(
                MaintenancePlanningActivationRecord.tenant_id == identity.tenant_id
            )
            if target_environment is not None:
                statement = statement.where(
                    MaintenancePlanningActivationRecord.target_environment
                    == _target(target_environment)
                )
            records = session.scalars(
                statement.order_by(
                    MaintenancePlanningActivationRecord.created_at.desc(),
                    MaintenancePlanningActivationRecord.activation_id,
                )
            )
            return tuple(
                self._view(record, identity, session=session, request_id=request_id)
                for record in records
            )

    def get(
        self,
        identity: IdentityContext,
        activation_id: str,
        *,
        request_id: str,
    ) -> MaintenancePlanningActivationView:
        self._require(
            identity,
            Action.READ_MAINTENANCE_COUNCIL,
            activation_id,
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            return self._view(
                _activation_or_hidden(session, identity.tenant_id, activation_id),
                identity,
                session=session,
                request_id=request_id,
            )

    def decide(
        self,
        identity: IdentityContext,
        activation_id: str,
        *,
        decision: str,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> MaintenancePlanningActivationView:
        normalized_decision = decision.strip().upper()
        normalized_reason = _reason(reason)
        if normalized_decision not in ACTIVATION_DECISIONS:
            raise MaintenancePlanningActivationConflict(
                "maintenance_planning_activation_decision_invalid"
            )
        self._require(
            identity,
            Action.MANAGE_TENANT_ADMINISTRATION,
            activation_id,
            request_id,
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _activation_or_hidden(
                session,
                identity.tenant_id,
                activation_id,
                lock=True,
            )
            _require_version(record, expected_version)
            if record.status != "PENDING_APPROVAL":
                raise MaintenancePlanningActivationConflict(
                    "maintenance_planning_activation_is_not_pending",
                    record.version,
                )
            if record.requested_by_subject_id == identity.subject_id:
                raise MaintenancePlanningActivationConflict(
                    "maintenance_planning_activation_requester_cannot_decide",
                    record.version,
                )
            if normalized_decision == "APPROVE":
                evaluation, suite = _eligible_evaluation(
                    session,
                    identity.tenant_id,
                    record.evaluation_id,
                    record.target_environment,
                    require_isolation=True,
                )
                _require_source_binding(record, evaluation, suite)
                active = session.scalar(
                    select(MaintenancePlanningActivationRecord).where(
                        MaintenancePlanningActivationRecord.tenant_id == identity.tenant_id,
                        MaintenancePlanningActivationRecord.target_environment
                        == record.target_environment,
                        MaintenancePlanningActivationRecord.status == "ACTIVE",
                        MaintenancePlanningActivationRecord.activation_id != record.activation_id,
                    )
                )
                if active is not None:
                    raise MaintenancePlanningActivationConflict(
                        "maintenance_planning_activation_target_already_active",
                        active.version,
                    )
                record.status = "ACTIVE"
                record.activated_at = now
            else:
                record.status = "REJECTED"
            record.decided_by_subject_id = identity.subject_id
            record.decision = normalized_decision
            record.decision_reason = normalized_reason
            record.version += 1
            session.flush()
            return self._view(record, identity, session=session, request_id=request_id)

    def rollback(
        self,
        identity: IdentityContext,
        activation_id: str,
        *,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> MaintenancePlanningActivationView:
        normalized_reason = _reason(reason)
        self._require(
            identity,
            Action.MANAGE_TENANT_ADMINISTRATION,
            activation_id,
            request_id,
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _activation_or_hidden(
                session,
                identity.tenant_id,
                activation_id,
                lock=True,
            )
            _require_version(record, expected_version)
            if record.status != "ACTIVE":
                raise MaintenancePlanningActivationConflict(
                    "maintenance_planning_activation_is_not_active",
                    record.version,
                )
            record.status = "ROLLED_BACK"
            record.rolled_back_by_subject_id = identity.subject_id
            record.rollback_reason = normalized_reason
            record.rolled_back_at = now
            record.version += 1
            session.flush()
            return self._view(record, identity, session=session, request_id=request_id)

    def resolve_active(
        self,
        tenant_id: str,
        target_environment: str,
    ) -> MaintenancePlanningActivationBinding:
        target = _target(target_environment)
        context = TenantContext(
            tenant_id=tenant_id,
            subject_id="maintenance-planning-runtime",
        )
        with self._database.transaction(context) as session:
            records = tuple(
                session.scalars(
                    select(MaintenancePlanningActivationRecord).where(
                        MaintenancePlanningActivationRecord.tenant_id == tenant_id,
                        MaintenancePlanningActivationRecord.target_environment == target,
                        MaintenancePlanningActivationRecord.status == "ACTIVE",
                    )
                )
            )
            if len(records) != 1:
                raise MaintenancePlanningActivationConflict(
                    "maintenance_planning_active_activation_required"
                )
            record = records[0]
            evaluation, suite = _eligible_evaluation(
                session,
                tenant_id,
                record.evaluation_id,
                target,
            )
            _require_source_binding(record, evaluation, suite)
            return MaintenancePlanningActivationBinding(
                activation_id=record.activation_id,
                activation_policy_hash=record.activation_policy_hash,
                target_environment=record.target_environment,
                evaluation_id=record.evaluation_id,
                evaluation_report_hash=record.evaluation_report_hash,
            )

    def _view(
        self,
        record: MaintenancePlanningActivationRecord,
        identity: IdentityContext,
        *,
        session: Session,
        request_id: str,
    ) -> MaintenancePlanningActivationView:
        evaluation = session.scalar(
            select(MaintenancePlanningEvaluationRunRecord).where(
                MaintenancePlanningEvaluationRunRecord.tenant_id == identity.tenant_id,
                MaintenancePlanningEvaluationRunRecord.evaluation_id == record.evaluation_id,
            )
        )
        if evaluation is None:
            raise MaintenancePlanningActivationNotVisible
        record_disclosure(
            session,
            identity,
            (str(pair["diagnosis_run_id"]) for pair in evaluation.pairs_json),
            resource_kind="maintenance-activation",
            resource_id=record.activation_id,
            request_id=request_id,
        )
        can_manage = self._can(
            identity,
            Action.MANAGE_TENANT_ADMINISTRATION,
            record.activation_id,
        )
        legal_actions: tuple[str, ...] = ()
        if (
            record.status == "PENDING_APPROVAL"
            and record.requested_by_subject_id != identity.subject_id
            and can_manage
        ):
            legal_actions = ("APPROVE", "REJECT")
        elif record.status == "ACTIVE" and can_manage:
            legal_actions = ("ROLLBACK",)
        return MaintenancePlanningActivationView(
            activation_id=record.activation_id,
            evaluation_id=record.evaluation_id,
            suite_id=record.suite_id,
            target_environment=record.target_environment,
            status=record.status,
            evaluation_decision=record.evaluation_decision,
            evaluation_report_hash=record.evaluation_report_hash,
            evaluation_policy_hash=record.evaluation_policy_hash,
            suite_manifest_hash=record.suite_manifest_hash,
            activation_policy_hash=record.activation_policy_hash,
            request_reason=record.request_reason,
            requested_by_subject_id=record.requested_by_subject_id,
            decided_by_subject_id=record.decided_by_subject_id,
            decision=record.decision,
            decision_reason=record.decision_reason,
            activated_at=_optional_utc(record.activated_at),
            rolled_back_by_subject_id=record.rolled_back_by_subject_id,
            rollback_reason=record.rollback_reason,
            rolled_back_at=_optional_utc(record.rolled_back_at),
            version=record.version,
            legal_actions=legal_actions,
            created_at=_as_utc(record.created_at),
            updated_at=_as_utc(record.updated_at),
        )

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
    ) -> None:
        if self._authorizer is None:
            raise MaintenancePlanningActivationConflict(
                "maintenance_planning_activation_authorizer_unavailable"
            )
        self._authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )

    def _can(self, identity: IdentityContext, action: Action, resource_id: str) -> bool:
        return (
            self._authorizer is not None
            and self._authorizer.decide(
                identity,
                action,
                ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            ).allowed
        )


def _eligible_evaluation(
    session: Session,
    tenant_id: str,
    evaluation_id: str,
    target: str,
    *,
    require_isolation: bool = False,
) -> tuple[MaintenancePlanningEvaluationRunRecord, MaintenancePlanningEvaluationSuiteRecord]:
    evaluation = session.scalar(
        select(MaintenancePlanningEvaluationRunRecord).where(
            MaintenancePlanningEvaluationRunRecord.tenant_id == tenant_id,
            MaintenancePlanningEvaluationRunRecord.evaluation_id == evaluation_id,
        )
    )
    if evaluation is None:
        raise MaintenancePlanningActivationNotVisible
    suite = session.scalar(
        select(MaintenancePlanningEvaluationSuiteRecord).where(
            MaintenancePlanningEvaluationSuiteRecord.tenant_id == tenant_id,
            MaintenancePlanningEvaluationSuiteRecord.suite_id == evaluation.suite_id,
        )
    )
    if suite is None:
        raise MaintenancePlanningActivationNotVisible
    report_hash = evaluation.report_hash
    if (
        evaluation.status != "COMPLETED"
        or not isinstance(report_hash, str)
        or not _is_sha256(report_hash)
        or suite.status != "FROZEN"
    ):
        raise MaintenancePlanningActivationConflict(
            "maintenance_planning_activation_completed_evaluation_required",
            evaluation.version,
        )
    if target == "PROJECT_STAGING":
        if (
            evaluation.decision
            not in {"PROJECT_CANDIDATE_ELIGIBLE", "PRODUCTION_CANDIDATE_ELIGIBLE"}
            or suite.sample_count < 3
        ):
            raise MaintenancePlanningActivationConflict(
                "maintenance_planning_activation_project_candidate_required",
                evaluation.version,
            )
    elif (
        evaluation.decision != "PRODUCTION_CANDIDATE_ELIGIBLE"
        or suite.evidence_tier != "ENTERPRISE_GOLD"
        or suite.sample_count < 30
    ):
        raise MaintenancePlanningActivationConflict(
            "maintenance_planning_activation_production_candidate_required",
            evaluation.version,
        )
    if require_isolation:
        require_review_report(session, evaluation, suite)
    return evaluation, suite


def _activation_policy_hash(
    evaluation: MaintenancePlanningEvaluationRunRecord,
    suite: MaintenancePlanningEvaluationSuiteRecord,
    target: str,
) -> str:
    return _digest(
        {
            "schema_version": "maintenance-planning-activation-policy/v1",
            "target_environment": target,
            "evaluation_id": evaluation.evaluation_id,
            "evaluation_decision": evaluation.decision,
            "evaluation_report_hash": evaluation.report_hash,
            "evaluation_policy_hash": evaluation.policy_hash,
            "suite_id": suite.suite_id,
            "suite_manifest_hash": suite.manifest_hash,
            "suite_evidence_tier": suite.evidence_tier,
            "suite_sample_count": suite.sample_count,
        }
    )


def _require_source_binding(
    record: MaintenancePlanningActivationRecord,
    evaluation: MaintenancePlanningEvaluationRunRecord,
    suite: MaintenancePlanningEvaluationSuiteRecord,
) -> None:
    if (
        record.suite_id != suite.suite_id
        or record.evaluation_decision != evaluation.decision
        or record.evaluation_report_hash != evaluation.report_hash
        or record.evaluation_policy_hash != evaluation.policy_hash
        or record.suite_manifest_hash != suite.manifest_hash
        or record.activation_policy_hash
        != _activation_policy_hash(evaluation, suite, record.target_environment)
    ):
        raise MaintenancePlanningActivationConflict(
            "maintenance_planning_activation_source_binding_changed",
            record.version,
        )


def require_bound_active_activation(
    session: Session,
    *,
    tenant_id: str,
    activation_id: str,
    activation_policy_hash: str | None,
    target_environment: str | None,
) -> None:
    """Revalidate an activation and its frozen evaluation source in one transaction."""

    if activation_policy_hash is None or target_environment is None:
        raise MaintenancePlanningActivationConflict(
            "maintenance_planning_activation_binding_changed"
        )
    target = _target(target_environment)
    record = _activation_or_hidden(session, tenant_id, activation_id)
    if (
        record.status != "ACTIVE"
        or record.activation_policy_hash != activation_policy_hash
        or record.target_environment != target
    ):
        raise MaintenancePlanningActivationConflict(
            "maintenance_planning_activation_binding_changed",
            record.version,
        )
    evaluation, suite = _eligible_evaluation(
        session,
        tenant_id,
        record.evaluation_id,
        target,
    )
    _require_source_binding(record, evaluation, suite)


def _activation_or_hidden(
    session: Session,
    tenant_id: str,
    activation_id: str,
    *,
    lock: bool = False,
) -> MaintenancePlanningActivationRecord:
    statement = select(MaintenancePlanningActivationRecord).where(
        MaintenancePlanningActivationRecord.tenant_id == tenant_id,
        MaintenancePlanningActivationRecord.activation_id == activation_id,
    )
    if lock:
        statement = statement.with_for_update()
    record = session.scalar(statement)
    if record is None:
        raise MaintenancePlanningActivationNotVisible
    return record


def _require_version(record: MaintenancePlanningActivationRecord, expected: int) -> None:
    if record.version != expected:
        raise MaintenancePlanningActivationConflict(
            "maintenance_planning_activation_version_conflict",
            record.version,
        )


def _target(value: str) -> str:
    target = value.strip().upper()
    if target not in TARGET_ENVIRONMENTS:
        raise MaintenancePlanningActivationConflict(
            "maintenance_planning_activation_target_invalid"
        )
    return target


def _reason(value: str) -> str:
    reason = " ".join(value.split())
    if not 8 <= len(reason) <= 1_000:
        raise MaintenancePlanningActivationConflict(
            "maintenance_planning_activation_reason_invalid"
        )
    return reason


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _optional_utc(value: datetime | None) -> datetime | None:
    return _as_utc(value) if value is not None else None
