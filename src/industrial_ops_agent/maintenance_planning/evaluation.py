"""Blinded Gold evaluation for single-agent versus four-agent maintenance plans."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from statistics import fmean
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc
from industrial_ops_agent.maintenance_planning.review_isolation import (
    READER_COVERAGE_DIGEST,
    REVIEW_POLICY_VERSION,
    ReviewIsolationConflict,
    eligibility,
    lock_subject,
    record_disclosure,
    record_exposure,
    resolve_source,
    source_family_keys,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    DiagnosisRunRecord,
    MaintenancePlanningBlindJudgmentRecord,
    MaintenancePlanningContributionRecord,
    MaintenancePlanningCouncilRecord,
    MaintenancePlanningEvaluationRunRecord,
    MaintenancePlanningEvaluationSuiteRecord,
    MaintenanceReviewAssignmentRecord,
    MaintenanceReviewExposureRecord,
    MaintenanceReviewPolicyRecord,
    ModelInferenceRecord,
)

EVALUATION_POLICY_VERSION = "maintenance-planning-blind-ab-v1"
BASELINE_KIND = "BASELINE_SINGLE_AGENT"
CANDIDATE_KIND = "MULTI_AGENT_COUNCIL"
EVIDENCE_TIERS = frozenset({"PROJECT_STAGING_GOLD", "ENTERPRISE_GOLD"})
_SCORE_FIELDS = (
    "plan_quality",
    "factual_error_count",
    "safety_omission_count",
    "parts_false_positive_count",
    "dispatch_executability",
    "expert_review_seconds",
)
_POLICY: dict[str, Any] = {
    "schema_version": "maintenance-planning-blind-ab-policy/v1",
    "required_judgments_per_case": 2,
    "adjudication_on_preference_disagreement": True,
    "project_minimum_cases": 3,
    "production_minimum_cases": 30,
    "quality_lift_min": 0.25,
    "quality_ci_lower_exclusive_min": 0.0,
    "maximum_candidate_latency_ratio": 4.5,
    "maximum_candidate_token_ratio": 5.0,
    "bootstrap_iterations": 2_000,
    "confidence_level": 0.95,
}
_POLICY_HASH = sha256(
    json.dumps(_POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


class MaintenancePlanningEvaluationNotVisible(Exception):
    pass


class MaintenancePlanningEvaluationConflict(RuntimeError):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class MaintenancePlanningGoldCaseInput:
    diagnosis_run_id: str
    expected_findings: tuple[str, ...]
    required_safety_hold_points: tuple[str, ...]
    allowed_parts: tuple[str, ...]
    required_dispatch_constraints: tuple[str, ...]
    high_risk: bool = True


@dataclass(frozen=True, slots=True)
class MaintenancePlanningPlanScore:
    plan_quality: float
    factual_error_count: int
    safety_omission_count: int
    parts_false_positive_count: int
    dispatch_executability: float
    expert_review_seconds: float


@dataclass(frozen=True, slots=True)
class MaintenancePlanningEvaluationSuiteView:
    suite_id: str
    name: str
    suite_version: str
    evidence_tier: str
    status: str
    cases: tuple[dict[str, Any], ...]
    sample_count: int
    manifest_hash: str
    created_by_subject_id: str
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class MaintenancePlanningBlindPairView:
    case_id: str
    diagnosis_run_id: str
    incident_id: str
    asset_id: str
    high_risk: bool
    rubric: dict[str, Any]
    variant_a: dict[str, Any]
    variant_b: dict[str, Any]
    judgment_count: int
    required_judgment_count: int
    needs_adjudication: bool
    judged_by_current_subject: bool
    revealed_assignment: dict[str, str] | None


@dataclass(frozen=True, slots=True)
class MaintenancePlanningEvaluationRunView:
    evaluation_id: str
    suite_id: str
    suite_name: str
    suite_version: str
    evidence_tier: str
    status: str
    policy_version: str
    policy_hash: str
    policy: dict[str, Any]
    pairs: tuple[MaintenancePlanningBlindPairView, ...]
    required_judgments_per_case: int
    aggregate_metrics: dict[str, Any]
    gate_results: dict[str, bool]
    decision: str
    report_hash: str | None
    failure_reason: str | None
    requested_by_subject_id: str
    completed_at: datetime | None
    version: int
    legal_actions: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


class MaintenancePlanningEvaluationService:
    """Freeze cases, randomize plans, collect judgments, and decide net benefit."""

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def create_suite(
        self,
        identity: IdentityContext,
        *,
        name: str,
        suite_version: str,
        evidence_tier: str,
        cases: tuple[MaintenancePlanningGoldCaseInput, ...],
        request_id: str,
    ) -> MaintenancePlanningEvaluationSuiteView:
        normalized_name = name.strip()
        normalized_version = suite_version.strip()
        normalized_tier = evidence_tier.strip().upper()
        if (
            not 3 <= len(normalized_name) <= 255
            or not 1 <= len(normalized_version) <= 64
            or normalized_tier not in EVIDENCE_TIERS
            or not cases
            or len(cases) > 500
        ):
            raise MaintenancePlanningEvaluationConflict(
                "maintenance_planning_evaluation_suite_input_invalid"
            )
        diagnosis_ids = [item.diagnosis_run_id.strip() for item in cases]
        if any(not item for item in diagnosis_ids) or len(set(diagnosis_ids)) != len(cases):
            raise MaintenancePlanningEvaluationConflict(
                "maintenance_planning_evaluation_suite_cases_invalid"
            )

        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            try:
                lock_subject(session, identity)
            except ReviewIsolationConflict as exc:
                raise MaintenancePlanningEvaluationConflict(exc.reason) from exc
            frozen_cases: list[dict[str, Any]] = []
            for ordinal, case_input in enumerate(cases, start=1):
                diagnosis = _diagnosis_or_hidden(
                    session,
                    identity.tenant_id,
                    case_input.diagnosis_run_id.strip(),
                )
                if diagnosis.status != "COMPLETED" or not isinstance(diagnosis.report, dict):
                    raise MaintenancePlanningEvaluationConflict(
                        "maintenance_planning_evaluation_completed_diagnosis_required",
                        diagnosis.version,
                    )
                council = session.scalar(
                    select(MaintenancePlanningCouncilRecord).where(
                        MaintenancePlanningCouncilRecord.tenant_id == identity.tenant_id,
                        MaintenancePlanningCouncilRecord.diagnosis_run_id
                        == diagnosis.diagnosis_run_id,
                    )
                )
                asset_id = council.asset_id if council is not None else None
                incident_id = diagnosis.incident_id
                if asset_id is None:
                    raise MaintenancePlanningEvaluationConflict(
                        "maintenance_planning_evaluation_council_required",
                        diagnosis.version,
                    )
                self._require(
                    identity,
                    Action.REVIEW_MAINTENANCE_COUNCIL,
                    diagnosis.diagnosis_run_id,
                    request_id,
                    asset_id=asset_id,
                )
                rubric = _validated_rubric(case_input)
                report_digest = _digest(diagnosis.report)
                manifest_digest = _digest(diagnosis.manifest)
                case_id = (
                    "maintenance-gold-case-"
                    + sha256(
                        f"{identity.tenant_id}:{diagnosis.diagnosis_run_id}:"
                        f"{report_digest}:{_digest(rubric)}".encode()
                    ).hexdigest()[:24]
                )
                frozen_cases.append(
                    {
                        "case_id": case_id,
                        "ordinal": ordinal,
                        "diagnosis_run_id": diagnosis.diagnosis_run_id,
                        "incident_id": incident_id,
                        "asset_id": asset_id,
                        "diagnosis_version": diagnosis.version,
                        "diagnosis_report_digest": report_digest,
                        "diagnosis_manifest_digest": manifest_digest,
                        "high_risk": case_input.high_risk,
                        "rubric": rubric,
                        "rubric_digest": _digest(rubric),
                    }
                )

            manifest_document = {
                "schema_version": "maintenance-planning-gold-suite/v1",
                "tenant_id": identity.tenant_id,
                "name": normalized_name,
                "suite_version": normalized_version,
                "evidence_tier": normalized_tier,
                "cases": frozen_cases,
            }
            manifest_hash = _digest(manifest_document)
            existing = session.scalar(
                select(MaintenancePlanningEvaluationSuiteRecord).where(
                    MaintenancePlanningEvaluationSuiteRecord.tenant_id == identity.tenant_id,
                    MaintenancePlanningEvaluationSuiteRecord.name == normalized_name,
                    MaintenancePlanningEvaluationSuiteRecord.suite_version == normalized_version,
                )
            )
            if existing is not None:
                if existing.manifest_hash != manifest_hash:
                    raise MaintenancePlanningEvaluationConflict(
                        "maintenance_planning_evaluation_suite_version_conflict",
                        existing.version,
                    )
                _record_suite_exposure(session, identity, existing, request_id)
                return _suite_view(existing)
            record = MaintenancePlanningEvaluationSuiteRecord(
                suite_id=f"maintenance-eval-suite-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                name=normalized_name,
                suite_version=normalized_version,
                evidence_tier=normalized_tier,
                status="FROZEN",
                cases_json=frozen_cases,
                sample_count=len(frozen_cases),
                manifest_hash=manifest_hash,
                created_by_subject_id=identity.subject_id,
                retired_by_subject_id=None,
                retired_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            _record_suite_exposure(session, identity, record, request_id)
            return _suite_view(record)

    def list_suites(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> tuple[MaintenancePlanningEvaluationSuiteView, ...]:
        self._require(
            identity,
            Action.READ_MAINTENANCE_COUNCIL,
            "maintenance-planning-evaluation-suites",
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            records = session.scalars(
                select(MaintenancePlanningEvaluationSuiteRecord)
                .where(MaintenancePlanningEvaluationSuiteRecord.tenant_id == identity.tenant_id)
                .order_by(
                    MaintenancePlanningEvaluationSuiteRecord.created_at.desc(),
                    MaintenancePlanningEvaluationSuiteRecord.suite_id,
                )
            )
            visible = tuple(
                record for record in records if _assets_visible(identity, record.cases_json)
            )
            record_disclosure(
                session,
                identity,
                (str(case["diagnosis_run_id"]) for record in visible for case in record.cases_json),
                resource_kind="maintenance-evaluation-suite-list",
                resource_id="maintenance-evaluation-suites",
                request_id=request_id,
            )
            return tuple(_suite_view(record) for record in visible)

    def create_run(
        self,
        identity: IdentityContext,
        *,
        suite_id: str,
        council_ids: tuple[str, ...],
        idempotency_key: str,
        request_id: str,
    ) -> MaintenancePlanningEvaluationRunView:
        if (
            not 8 <= len(idempotency_key) <= 255
            or not suite_id.strip()
            or not council_ids
            or len(set(council_ids)) != len(council_ids)
        ):
            raise MaintenancePlanningEvaluationConflict(
                "maintenance_planning_evaluation_run_input_invalid"
            )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            policy_epoch = _review_policy(session, identity.tenant_id)
            policy_document = {
                **_POLICY,
                "schema_version": "maintenance-planning-blind-ab-policy/v2",
                "reader_coverage_digest": policy_epoch.reader_coverage_digest,
                "policy_epoch_version": policy_epoch.version,
                "rollout_receipt_digest": policy_epoch.receipt_digest,
                "cutover_at": as_utc(cast(datetime, policy_epoch.cutover_at)).isoformat(),
            }
            suite = _suite_or_hidden(session, identity.tenant_id, suite_id)
            if suite.status != "FROZEN" or suite.sample_count != len(council_ids):
                raise MaintenancePlanningEvaluationConflict(
                    "maintenance_planning_evaluation_suite_not_runnable", suite.version
                )
            councils = tuple(
                session.scalars(
                    select(MaintenancePlanningCouncilRecord).where(
                        MaintenancePlanningCouncilRecord.tenant_id == identity.tenant_id,
                        MaintenancePlanningCouncilRecord.council_id.in_(council_ids),
                    )
                )
            )
            if len(councils) != len(council_ids):
                raise MaintenancePlanningEvaluationNotVisible
            council_by_diagnosis = {item.diagnosis_run_id: item for item in councils}
            if len(council_by_diagnosis) != len(councils):
                raise MaintenancePlanningEvaluationConflict(
                    "maintenance_planning_evaluation_duplicate_diagnosis"
                )

            pairs: list[dict[str, Any]] = []
            for case in suite.cases_json:
                diagnosis_id = str(case.get("diagnosis_run_id", ""))
                council = council_by_diagnosis.get(diagnosis_id)
                if council is None:
                    raise MaintenancePlanningEvaluationConflict(
                        "maintenance_planning_evaluation_case_council_missing"
                    )
                self._require(
                    identity,
                    Action.REVIEW_MAINTENANCE_COUNCIL,
                    council.council_id,
                    request_id,
                    asset_id=council.asset_id,
                )
                pairs.append(_build_pair(session, suite, case, council))

            request_document = {
                "suite_id": suite.suite_id,
                "suite_manifest_hash": suite.manifest_hash,
                "council_ids": sorted(council_ids),
                "policy_hash": _digest(policy_document),
            }
            request_hash = _digest(request_document)
            replay = session.scalar(
                select(MaintenancePlanningEvaluationRunRecord).where(
                    MaintenancePlanningEvaluationRunRecord.tenant_id == identity.tenant_id,
                    MaintenancePlanningEvaluationRunRecord.requested_by_subject_id
                    == identity.subject_id,
                    MaintenancePlanningEvaluationRunRecord.idempotency_key == idempotency_key,
                )
            )
            if replay is not None:
                if replay.request_hash != request_hash:
                    raise MaintenancePlanningEvaluationConflict(
                        "maintenance_planning_evaluation_idempotency_conflict",
                        replay.version,
                    )
                record_disclosure(
                    session,
                    identity,
                    (str(pair["diagnosis_run_id"]) for pair in replay.pairs_json),
                    resource_kind="maintenance-evaluation",
                    resource_id=replay.evaluation_id,
                    request_id=request_id,
                )
                return _run_view(session, replay, suite, identity.subject_id)

            for pair in pairs:
                pair["review_case_id"] = f"review-case-{uuid4().hex}"
            record = MaintenancePlanningEvaluationRunRecord(
                evaluation_id=f"maintenance-evaluation-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                suite_id=suite.suite_id,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                status="JUDGING",
                policy_version=REVIEW_POLICY_VERSION,
                policy_hash=_digest(policy_document),
                policy_json=policy_document,
                pairs_json=pairs,
                required_judgments_per_case=int(_POLICY["required_judgments_per_case"]),
                aggregate_metrics_json={},
                gate_results_json={},
                decision="PENDING",
                report_hash=None,
                failure_reason=None,
                requested_by_subject_id=identity.subject_id,
                completed_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            record_disclosure(
                session,
                identity,
                (str(pair["diagnosis_run_id"]) for pair in record.pairs_json),
                resource_kind="maintenance-evaluation",
                resource_id=record.evaluation_id,
                request_id=request_id,
            )
            return _run_view(session, record, suite, identity.subject_id)

    def list_runs(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> tuple[MaintenancePlanningEvaluationRunView, ...]:
        self._require(
            identity,
            Action.READ_MAINTENANCE_COUNCIL,
            "maintenance-planning-evaluations",
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            records = tuple(
                session.scalars(
                    select(MaintenancePlanningEvaluationRunRecord)
                    .where(MaintenancePlanningEvaluationRunRecord.tenant_id == identity.tenant_id)
                    .order_by(
                        MaintenancePlanningEvaluationRunRecord.created_at.desc(),
                        MaintenancePlanningEvaluationRunRecord.evaluation_id,
                    )
                )
            )
            suites = {
                suite.suite_id: suite
                for suite in session.scalars(
                    select(MaintenancePlanningEvaluationSuiteRecord).where(
                        MaintenancePlanningEvaluationSuiteRecord.tenant_id == identity.tenant_id
                    )
                )
            }
            visible = tuple(
                record
                for record in records
                if record.suite_id in suites
                and _assets_visible(identity, suites[record.suite_id].cases_json)
            )
            record_disclosure(
                session,
                identity,
                (str(pair["diagnosis_run_id"]) for record in visible for pair in record.pairs_json),
                resource_kind="maintenance-evaluation-list",
                resource_id="maintenance-evaluations",
                request_id=request_id,
            )
            return tuple(
                _run_view(session, record, suites[record.suite_id], identity.subject_id)
                for record in visible
            )

    def get_run(
        self,
        identity: IdentityContext,
        evaluation_id: str,
        *,
        request_id: str,
    ) -> MaintenancePlanningEvaluationRunView:
        with self._database.transaction(identity.tenant_context) as session:
            record = _run_or_hidden(session, identity.tenant_id, evaluation_id)
            suite = _suite_or_hidden(session, identity.tenant_id, record.suite_id)
            if not _assets_visible(identity, suite.cases_json):
                raise MaintenancePlanningEvaluationNotVisible
            self._require(
                identity,
                Action.READ_MAINTENANCE_COUNCIL,
                evaluation_id,
                request_id,
            )
            record_disclosure(
                session,
                identity,
                (str(pair["diagnosis_run_id"]) for pair in record.pairs_json),
                resource_kind="maintenance-evaluation",
                resource_id=record.evaluation_id,
                request_id=request_id,
            )
            return _run_view(session, record, suite, identity.subject_id)

    def review_queue(
        self, identity: IdentityContext, *, request_id: str
    ) -> tuple[dict[str, Any], ...]:
        self._require(
            identity, Action.REVIEW_MAINTENANCE_COUNCIL, "maintenance-review-queue", request_id
        )
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            runs = session.scalars(
                select(MaintenancePlanningEvaluationRunRecord)
                .where(
                    MaintenancePlanningEvaluationRunRecord.tenant_id == identity.tenant_id,
                    MaintenancePlanningEvaluationRunRecord.policy_version == REVIEW_POLICY_VERSION,
                    (
                        (MaintenancePlanningEvaluationRunRecord.status == "JUDGING")
                        | MaintenancePlanningEvaluationRunRecord.evaluation_id.in_(
                            select(MaintenanceReviewAssignmentRecord.evaluation_id).where(
                                MaintenanceReviewAssignmentRecord.tenant_id == identity.tenant_id,
                                MaintenanceReviewAssignmentRecord.subject_id == identity.subject_id,
                                MaintenanceReviewAssignmentRecord.state == "ACTIVE",
                            )
                        )
                    ),
                )
                .order_by(MaintenancePlanningEvaluationRunRecord.created_at)
            )
            items = []
            for run in runs:
                for pair in run.pairs_json:
                    if not _assets_visible(identity, [pair]):
                        continue
                    claim = _case_claim(session, identity, run.evaluation_id, str(pair["case_id"]))
                    try:
                        source = resolve_source(
                            session, identity.tenant_id, str(pair["diagnosis_run_id"])
                        )
                        reason = eligibility(
                            session,
                            identity,
                            source,
                            reader_coverage_digest=READER_COVERAGE_DIGEST,
                            active_claim_id=claim.claim_id
                            if claim and claim.state == "ACTIVE"
                            else None,
                        )
                        if reason == "ELIGIBLE":
                            try:
                                _check_run_epoch(run, _review_policy(session, identity.tenant_id))
                            except ReviewIsolationConflict:
                                reason = "POLICY_CHANGED"
                        if reason == "ELIGIBLE":
                            if _source_failure(session, run):
                                reason = "SOURCE_CHANGED"
                            elif not _review_case_open(session, run, pair, claim):
                                reason = "CASE_CLOSED"
                            elif claim and claim.state == "ACTIVE":
                                reason = "CLAIMED"
                    except ReviewIsolationConflict as exc:
                        # Keep the opaque handle so a stale active claim remains withdrawable.
                        reason = exc.reason.removeprefix("maintenance_review_").upper()
                    items.append(
                        {
                            "evaluation_id": run.evaluation_id,
                            "case_id": pair["review_case_id"],
                            "eligibility": reason,
                            "claim_id": claim.claim_id
                            if claim and claim.state == "ACTIVE"
                            else None,
                            "claim_version": claim.version
                            if claim and claim.state == "ACTIVE"
                            else None,
                        }
                    )
            return tuple(items)

    def claim_review(
        self,
        identity: IdentityContext,
        evaluation_id: str,
        case_id: str,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> dict[str, Any]:
        if not 8 <= len(idempotency_key) <= 255:
            raise ReviewIsolationConflict("idempotency_key_invalid")
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            run, pair = self._review_context(session, identity, evaluation_id, case_id, request_id)
            digest = _digest({"evaluation_id": evaluation_id, "case_id": case_id})
            existing = session.scalar(
                select(MaintenanceReviewAssignmentRecord).where(
                    MaintenanceReviewAssignmentRecord.tenant_id == identity.tenant_id,
                    MaintenanceReviewAssignmentRecord.subject_id == identity.subject_id,
                    MaintenanceReviewAssignmentRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.request_digest != digest:
                    raise ReviewIsolationConflict("idempotency_conflict")
                if existing.state == "ACTIVE":
                    _validate_claim(session, identity, run, pair, existing)
                return _claim_view(run, pair, existing)
            existing = _case_claim(session, identity, evaluation_id, str(pair["case_id"]))
            if existing is not None:
                # Resume an active claim after a reload without allocating another identity.
                if existing.state != "ACTIVE":
                    raise ReviewIsolationConflict("already_participated")
                _validate_claim(session, identity, run, pair, existing)
                return _claim_view(run, pair, existing)
            if run.status != "JUDGING" or not _review_case_open(session, run, pair, None):
                raise ReviewIsolationConflict("case_closed")
            if _source_failure(session, run):
                raise ReviewIsolationConflict("source_changed")
            source = resolve_source(session, identity.tenant_id, str(pair["diagnosis_run_id"]))
            reason = eligibility(
                session, identity, source, reader_coverage_digest=READER_COVERAGE_DIGEST
            )
            if reason != "ELIGIBLE":
                raise ReviewIsolationConflict(reason.lower())
            epoch = _review_policy(session, identity.tenant_id)
            _check_run_epoch(run, epoch)
            now = datetime.now(UTC)
            claim = MaintenanceReviewAssignmentRecord(
                claim_id=f"review-claim-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                subject_id=identity.subject_id,
                evaluation_id=evaluation_id,
                case_id=str(pair["case_id"]),
                source_family_keys=list(source_family_keys(session, source)),
                source_binding_digest=_digest(
                    {"source": source.version_digest, "pair": pair["source_binding"]}
                ),
                idempotency_key=idempotency_key,
                request_digest=digest,
                policy_version=REVIEW_POLICY_VERSION,
                policy_epoch_version=epoch.version,
                reader_coverage_digest=epoch.reader_coverage_digest,
                state="ACTIVE",
                version=1,
                claimed_at=now,
                submitted_at=None,
                withdrawn_at=None,
                submission_digest=None,
                created_at=now,
                updated_at=now,
            )
            session.add(claim)
            session.flush()
            return _claim_view(run, pair, claim)

    def withdraw_review(
        self,
        identity: IdentityContext,
        evaluation_id: str,
        case_id: str,
        *,
        claim_id: str,
        claim_version: int,
        request_id: str,
    ) -> dict[str, Any]:
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            run, pair = self._review_context(session, identity, evaluation_id, case_id, request_id)
            claim = _owned_claim(session, identity, run, pair, claim_id)
            if claim.state == "WITHDRAWN" and claim_version in {claim.version, claim.version - 1}:
                return _claim_view(run, pair, claim)
            if claim.state != "ACTIVE" or claim.version != claim_version:
                raise ReviewIsolationConflict("claim_version_conflict")
            claim.state = "WITHDRAWN"
            claim.withdrawn_at = datetime.now(UTC)
            claim.updated_at = claim.withdrawn_at
            claim.version += 1
            session.flush()
            return _claim_view(run, pair, claim)

    def _review_context(
        self,
        session: Session,
        identity: IdentityContext,
        evaluation_id: str,
        case_id: str,
        request_id: str,
    ) -> tuple[MaintenancePlanningEvaluationRunRecord, dict[str, Any]]:
        run = _run_or_hidden(session, identity.tenant_id, evaluation_id, lock=True)
        if run.policy_version != REVIEW_POLICY_VERSION:
            raise ReviewIsolationConflict("legacy_read_only")
        pair = next((p for p in run.pairs_json if p.get("review_case_id") == case_id), None)
        if pair is None:
            raise MaintenancePlanningEvaluationNotVisible
        self._require(
            identity,
            Action.REVIEW_MAINTENANCE_COUNCIL,
            evaluation_id,
            request_id,
            asset_id=str(pair["asset_id"]),
        )
        return run, pair

    def submit_judgment(
        self,
        identity: IdentityContext,
        evaluation_id: str,
        case_id: str,
        *,
        claim_id: str,
        claim_version: int,
        variant_a: MaintenancePlanningPlanScore,
        variant_b: MaintenancePlanningPlanScore,
        preferred_variant: str,
        rationale: str,
        request_id: str,
    ) -> dict[str, Any]:
        normalized_preference = preferred_variant.strip().upper()
        normalized_rationale = rationale.strip()
        scores_a = _validated_scores(variant_a)
        scores_b = _validated_scores(variant_b)
        if (
            normalized_preference not in {"A", "B", "TIE"}
            or not 8 <= len(normalized_rationale) <= 1_000
        ):
            raise MaintenancePlanningEvaluationConflict(
                "maintenance_planning_evaluation_judgment_invalid"
            )
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record, pair = self._review_context(
                session, identity, evaluation_id, case_id, request_id
            )
            suite = _suite_or_hidden(session, identity.tenant_id, record.suite_id)
            claim = _owned_claim(session, identity, record, pair, claim_id)
            case_id = str(pair["case_id"])
            submission = {
                "variant_a": scores_a,
                "variant_b": scores_b,
                "preferred_variant": normalized_preference,
                "rationale": normalized_rationale,
            }
            if claim.state == "SUBMITTED":
                if claim.submission_digest != _digest(submission) or claim_version not in {
                    claim.version,
                    claim.version - 1,
                }:
                    raise ReviewIsolationConflict("submission_conflict")
                return _claim_view(record, pair, claim)
            if claim.state != "ACTIVE" or claim.version != claim_version:
                raise ReviewIsolationConflict("claim_version_conflict")
            _validate_claim(session, identity, record, pair, claim)
            if record.requested_by_subject_id == identity.subject_id:
                raise MaintenancePlanningEvaluationConflict(
                    "maintenance_planning_evaluation_independent_judge_required",
                    record.version,
                )
            if record.status != "JUDGING":
                raise MaintenancePlanningEvaluationConflict(
                    "maintenance_planning_evaluation_not_judging", record.version
                )
            existing = session.scalar(
                select(MaintenancePlanningBlindJudgmentRecord).where(
                    MaintenancePlanningBlindJudgmentRecord.tenant_id == identity.tenant_id,
                    MaintenancePlanningBlindJudgmentRecord.evaluation_id == evaluation_id,
                    MaintenancePlanningBlindJudgmentRecord.case_id == case_id,
                    MaintenancePlanningBlindJudgmentRecord.judge_subject_id == identity.subject_id,
                )
            )
            score_digest = _digest(submission)
            if existing is not None:
                if existing.score_digest != score_digest:
                    raise MaintenancePlanningEvaluationConflict(
                        "maintenance_planning_evaluation_judgment_conflict",
                        record.version,
                    )
                raise ReviewIsolationConflict("submission_record_conflict")

            judgments = _case_judgments(session, identity.tenant_id, evaluation_id, case_id)
            if len(judgments) >= 3 or (
                len(judgments) >= record.required_judgments_per_case
                and not _needs_adjudication(judgments)
            ):
                raise MaintenancePlanningEvaluationConflict(
                    "maintenance_planning_evaluation_case_already_decided",
                    record.version,
                )
            assignment_digest = _digest(pair["assignment"])
            now = datetime.now(UTC)
            session.add(
                MaintenancePlanningBlindJudgmentRecord(
                    judgment_id=f"maintenance-judgment-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    evaluation_id=evaluation_id,
                    case_id=case_id,
                    judge_subject_id=identity.subject_id,
                    blind_assignment_digest=assignment_digest,
                    variant_a_scores_json=scores_a,
                    variant_b_scores_json=scores_b,
                    preferred_variant=normalized_preference,
                    rationale=normalized_rationale,
                    score_digest=score_digest,
                    is_adjudication=len(judgments) >= record.required_judgments_per_case,
                    submitted_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            claim.state = "SUBMITTED"
            claim.submitted_at = now
            claim.updated_at = now
            claim.submission_digest = score_digest
            claim.version += 1
            record.version += 1
            session.flush()
            _maybe_finalize(session, record, suite)
            session.flush()
            return _claim_view(record, pair, claim)

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
        *,
        asset_id: str | None = None,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id=resource_id,
                asset_id=asset_id,
            ),
            request_id=request_id,
        )


def _review_policy(session: Session, tenant_id: str) -> MaintenanceReviewPolicyRecord:
    from industrial_ops_agent.maintenance_planning.rollout_evidence import require_policy_activation

    policy = session.scalar(
        select(MaintenanceReviewPolicyRecord)
        .where(
            MaintenanceReviewPolicyRecord.tenant_id == tenant_id,
        )
        .with_for_update(read=True)
    )
    if (
        policy is None
        or policy.cutover_at is None
        or policy.policy_version != REVIEW_POLICY_VERSION
        or policy.reader_coverage_digest != READER_COVERAGE_DIGEST
    ):
        raise ReviewIsolationConflict("policy_inactive")
    require_policy_activation(session, policy)
    return policy


def _check_run_epoch(
    run: MaintenancePlanningEvaluationRunRecord, epoch: MaintenanceReviewPolicyRecord
) -> None:
    if (
        epoch.cutover_at is None
        or run.policy_version != REVIEW_POLICY_VERSION
        or run.policy_json.get("policy_epoch_version") != epoch.version
        or run.policy_json.get("rollout_receipt_digest") != epoch.receipt_digest
        or run.policy_json.get("reader_coverage_digest") != epoch.reader_coverage_digest
        or run.policy_json.get("cutover_at") != as_utc(epoch.cutover_at).isoformat()
        or run.policy_hash != _digest(run.policy_json)
    ):
        raise ReviewIsolationConflict("policy_changed")


def _case_claim(
    session: Session, identity: IdentityContext, evaluation_id: str, case_id: str
) -> MaintenanceReviewAssignmentRecord | None:
    return session.scalar(
        select(MaintenanceReviewAssignmentRecord).where(
            MaintenanceReviewAssignmentRecord.tenant_id == identity.tenant_id,
            MaintenanceReviewAssignmentRecord.subject_id == identity.subject_id,
            MaintenanceReviewAssignmentRecord.evaluation_id == evaluation_id,
            MaintenanceReviewAssignmentRecord.case_id == case_id,
        )
    )


def _owned_claim(
    session: Session,
    identity: IdentityContext,
    run: MaintenancePlanningEvaluationRunRecord,
    pair: dict[str, Any],
    claim_id: str,
) -> MaintenanceReviewAssignmentRecord:
    claim = _case_claim(session, identity, run.evaluation_id, str(pair["case_id"]))
    if claim is None or claim.claim_id != claim_id:
        raise ReviewIsolationConflict("claim_required")
    return claim


def _review_case_open(
    session: Session,
    run: MaintenancePlanningEvaluationRunRecord,
    pair: dict[str, Any],
    claim: MaintenanceReviewAssignmentRecord | None,
) -> bool:
    judgments = _case_judgments(session, run.tenant_id, run.evaluation_id, str(pair["case_id"]))
    capacity = 3 if _needs_adjudication(judgments) else run.required_judgments_per_case
    active = session.scalars(
        select(MaintenanceReviewAssignmentRecord).where(
            MaintenanceReviewAssignmentRecord.tenant_id == run.tenant_id,
            MaintenanceReviewAssignmentRecord.evaluation_id == run.evaluation_id,
            MaintenanceReviewAssignmentRecord.case_id == str(pair["case_id"]),
            MaintenanceReviewAssignmentRecord.state == "ACTIVE",
        )
    ).all()
    occupied = sum(item.claim_id != getattr(claim, "claim_id", None) for item in active)
    return run.status == "JUDGING" and len(judgments) + occupied < capacity


def _validate_claim(
    session: Session,
    identity: IdentityContext,
    run: MaintenancePlanningEvaluationRunRecord,
    pair: dict[str, Any],
    claim: MaintenanceReviewAssignmentRecord,
) -> None:
    epoch = _review_policy(session, identity.tenant_id)
    _check_run_epoch(run, epoch)
    if (
        claim.policy_version != REVIEW_POLICY_VERSION
        or claim.policy_epoch_version != epoch.version
        or claim.reader_coverage_digest != epoch.reader_coverage_digest
    ):
        raise ReviewIsolationConflict("policy_changed")
    if run.status != "JUDGING" or not _review_case_open(session, run, pair, claim):
        raise ReviewIsolationConflict("case_closed")
    if _source_failure(session, run):
        raise ReviewIsolationConflict("source_changed")
    source = resolve_source(session, identity.tenant_id, str(pair["diagnosis_run_id"]))
    if claim.source_family_keys != list(
        source_family_keys(session, source)
    ) or claim.source_binding_digest != _digest(
        {"source": source.version_digest, "pair": pair["source_binding"]}
    ):
        raise ReviewIsolationConflict("source_changed")
    reason = eligibility(
        session,
        identity,
        source,
        reader_coverage_digest=READER_COVERAGE_DIGEST,
        active_claim_id=claim.claim_id,
    )
    if reason != "ELIGIBLE":
        raise ReviewIsolationConflict(reason.lower())


def _claim_view(
    run: MaintenancePlanningEvaluationRunRecord,
    pair: dict[str, Any],
    claim: MaintenanceReviewAssignmentRecord,
) -> dict[str, Any]:
    # Deliberately not _run_view: no source IDs, rubric, mappings or another judge's scores.
    result: dict[str, Any] = {
        "evaluation_id": run.evaluation_id,
        "case_id": pair["review_case_id"],
        "claim_id": claim.claim_id,
        "claim_version": claim.version,
        "state": claim.state,
        "variant_a": None,
        "variant_b": None,
    }
    if claim.state == "ACTIVE":
        for label in ("A", "B"):
            content = _anonymize_plan(pair["variants"][pair["assignment"][label]])
            if content["summary"] == "单 Agent 未提供结构化维修摘要":
                content["summary"] = "未提供结构化维修摘要"
            serialized = json.dumps(content, ensure_ascii=False)
            identifiers = [
                pair["diagnosis_run_id"],
                pair["incident_id"],
                pair["asset_id"],
                pair["source_binding"]["council_model_release_id"],
            ]
            if "://" in serialized or any(
                value and str(value) in serialized for value in identifiers
            ):
                raise ReviewIsolationConflict("anonymous_content_unavailable")
            result[f"variant_{label.lower()}"] = content
    return result


def _isolation_evidence(
    session: Session,
    run: MaintenancePlanningEvaluationRunRecord,
    judgments_by_case: dict[str, tuple[MaintenancePlanningBlindJudgmentRecord, ...]],
) -> tuple[bool, list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    valid = run.policy_version == REVIEW_POLICY_VERSION
    for pair in run.pairs_json:
        source = resolve_source(session, run.tenant_id, str(pair["diagnosis_run_id"]))
        families = list(source_family_keys(session, source))
        binding = _digest({"source": source.version_digest, "pair": pair["source_binding"]})
        for judgment in judgments_by_case[str(pair["case_id"])]:
            claim = session.scalar(
                select(MaintenanceReviewAssignmentRecord).where(
                    MaintenanceReviewAssignmentRecord.tenant_id == run.tenant_id,
                    MaintenanceReviewAssignmentRecord.subject_id == judgment.judge_subject_id,
                    MaintenanceReviewAssignmentRecord.evaluation_id == run.evaluation_id,
                    MaintenanceReviewAssignmentRecord.case_id == judgment.case_id,
                )
            )
            if claim is None or claim.submitted_at is None:
                valid = False
                continue
            exposed = session.scalar(
                select(MaintenanceReviewExposureRecord.exposure_id)
                .where(
                    MaintenanceReviewExposureRecord.tenant_id == run.tenant_id,
                    MaintenanceReviewExposureRecord.subject_id == judgment.judge_subject_id,
                    MaintenanceReviewExposureRecord.source_family_key.in_(families),
                    MaintenanceReviewExposureRecord.occurred_at <= claim.submitted_at,
                )
                .limit(1)
            )
            valid = valid and (
                claim.state == "SUBMITTED"
                and claim.policy_version == REVIEW_POLICY_VERSION
                and claim.source_family_keys == families
                and claim.source_binding_digest == binding
                and claim.submission_digest == judgment.score_digest
                and exposed is None
                and claim.policy_epoch_version == run.policy_json.get("policy_epoch_version")
                and claim.reader_coverage_digest == run.policy_json.get("reader_coverage_digest")
                and as_utc(claim.claimed_at)
                <= as_utc(claim.submitted_at)
                == as_utc(judgment.submitted_at)
            )
            evidence.append(
                {
                    "claim_id": claim.claim_id,
                    "subject_id": claim.subject_id,
                    "case_id": claim.case_id,
                    "source_family_keys": families,
                    "source_binding_digest": claim.source_binding_digest,
                    "submission_digest": claim.submission_digest,
                    "policy_epoch_version": claim.policy_epoch_version,
                    "reader_coverage_digest": claim.reader_coverage_digest,
                    "claimed_at": as_utc(claim.claimed_at).isoformat(),
                    "submitted_at": as_utc(claim.submitted_at).isoformat(),
                }
            )
    return valid and bool(evidence), sorted(evidence, key=lambda item: item["claim_id"])


def _evaluation_report_document(
    record: MaintenancePlanningEvaluationRunRecord,
    suite: MaintenancePlanningEvaluationSuiteRecord,
    judgments_by_case: dict[str, tuple[MaintenancePlanningBlindJudgmentRecord, ...]],
    metrics: dict[str, Any],
    gates: dict[str, bool],
    decision: str,
) -> dict[str, Any]:
    return {
        "schema_version": (
            "maintenance-planning-blind-ab-report/v2"
            if record.policy_version == REVIEW_POLICY_VERSION
            else "maintenance-planning-blind-ab-report/v1"
        ),
        "evaluation_id": record.evaluation_id,
        "suite_manifest_hash": suite.manifest_hash,
        "policy_hash": record.policy_hash,
        "pair_bindings": [item["source_binding"] for item in record.pairs_json],
        "judgment_digests": sorted(
            j.score_digest for items in judgments_by_case.values() for j in items
        ),
        "metrics": metrics,
        "gates": gates,
        "decision": decision,
    }


def require_review_report(
    session: Session,
    record: MaintenancePlanningEvaluationRunRecord,
    suite: MaintenancePlanningEvaluationSuiteRecord,
) -> None:
    """For new activations only; never rewrite or retroactively revoke legacy activations."""
    epoch = _review_policy(session, record.tenant_id)
    _check_run_epoch(record, epoch)
    judgments = {
        str(pair["case_id"]): _case_judgments(
            session, record.tenant_id, record.evaluation_id, str(pair["case_id"])
        )
        for pair in record.pairs_json
    }
    valid, evidence = _isolation_evidence(session, record, judgments)
    if (
        not valid
        or not all(_blind_protocol_results(record, judgments).values())
        or _source_failure(session, record) is not None
        or record.gate_results_json.get("historical_exposure_isolated") is not True
        or record.aggregate_metrics_json.get("review_isolation") != evidence
        or record.report_hash
        != _digest(
            _evaluation_report_document(
                record,
                suite,
                judgments,
                record.aggregate_metrics_json,
                record.gate_results_json,
                record.decision,
            )
        )
    ):
        raise ReviewIsolationConflict("qualified_report_required")


def _record_suite_exposure(
    session: Session,
    identity: IdentityContext,
    suite: MaintenancePlanningEvaluationSuiteRecord,
    request_id: str,
) -> None:
    # Keep frozen v1 cases/manifests byte-for-byte compatible; exposure is separate.
    try:
        for case in suite.cases_json:
            source = resolve_source(session, identity.tenant_id, str(case["diagnosis_run_id"]))
            record_exposure(
                session,
                identity,
                source,
                reason="SUITE_AUTHOR",
                resource_kind="evaluation-suite",
                resource_id=suite.suite_id,
                request_id=request_id,
            )
    except ReviewIsolationConflict as exc:
        raise MaintenancePlanningEvaluationConflict(exc.reason) from exc


def _build_pair(
    session: Session,
    suite: MaintenancePlanningEvaluationSuiteRecord,
    case: dict[str, Any],
    council: MaintenancePlanningCouncilRecord,
) -> dict[str, Any]:
    diagnosis = _diagnosis_or_hidden(session, council.tenant_id, council.diagnosis_run_id)
    if (
        council.status not in {"REVIEW_PENDING", "ACCEPTED", "REJECTED"}
        or not isinstance(council.plan_json, dict)
        or council.result_digest is None
        or _digest(council.plan_json) != council.result_digest
        or diagnosis.version != case.get("diagnosis_version")
        or _digest(diagnosis.report) != case.get("diagnosis_report_digest")
        or _digest(diagnosis.manifest) != case.get("diagnosis_manifest_digest")
        or council.diagnosis_report_digest != case.get("diagnosis_report_digest")
        or council.diagnosis_manifest_digest != case.get("diagnosis_manifest_digest")
    ):
        raise MaintenancePlanningEvaluationConflict(
            "maintenance_planning_evaluation_source_binding_invalid", council.version
        )
    contributions = tuple(
        session.scalars(
            select(MaintenancePlanningContributionRecord)
            .where(
                MaintenancePlanningContributionRecord.tenant_id == council.tenant_id,
                MaintenancePlanningContributionRecord.council_id == council.council_id,
                MaintenancePlanningContributionRecord.attempt_number == council.attempt_count,
            )
            .order_by(MaintenancePlanningContributionRecord.agent_role)
        )
    )
    if {item.agent_role for item in contributions} != {
        "SAFETY",
        "PARTS",
        "DISPATCH",
        "COORDINATOR",
    }:
        raise MaintenancePlanningEvaluationConflict(
            "maintenance_planning_evaluation_contributions_incomplete", council.version
        )
    baseline_measurement = _baseline_measurement(session, diagnosis)
    candidate_measurement = _candidate_measurement(session, contributions)
    controlled_model_binding = bool(
        baseline_measurement["release_ids"]
        and candidate_measurement["release_ids"]
        and baseline_measurement["release_ids"]
        == candidate_measurement["release_ids"]
        == [council.model_release_id]
    )
    baseline_plan = _normalize_baseline_plan(dict(diagnosis.report or {}))
    candidate_plan = _anonymize_plan(council.plan_json)
    assignment_seed = _digest(
        {
            "suite_manifest_hash": suite.manifest_hash,
            "case_id": case["case_id"],
            "council_result_digest": council.result_digest,
        }
    )
    assignment = (
        {"A": BASELINE_KIND, "B": CANDIDATE_KIND}
        if int(assignment_seed[-1], 16) % 2 == 0
        else {"A": CANDIDATE_KIND, "B": BASELINE_KIND}
    )
    binding = {
        "diagnosis_run_id": diagnosis.diagnosis_run_id,
        "diagnosis_version": diagnosis.version,
        "diagnosis_report_digest": _digest(diagnosis.report),
        "diagnosis_manifest_digest": _digest(diagnosis.manifest),
        "council_id": council.council_id,
        "council_attempt_count": council.attempt_count,
        "council_result_digest": council.result_digest,
        "council_model_release_id": council.model_release_id,
        "council_model_manifest_hash": council.model_manifest_hash,
        "contribution_digests": sorted(item.output_digest for item in contributions),
        "measurement_digest": _digest(
            {"baseline": baseline_measurement, "candidate": candidate_measurement}
        ),
    }
    return {
        "case_id": case["case_id"],
        "diagnosis_run_id": diagnosis.diagnosis_run_id,
        "incident_id": case["incident_id"],
        "asset_id": case["asset_id"],
        "high_risk": bool(case["high_risk"]),
        "rubric": dict(case["rubric"]),
        "rubric_digest": case["rubric_digest"],
        "assignment": assignment,
        "variants": {
            BASELINE_KIND: baseline_plan,
            CANDIDATE_KIND: candidate_plan,
        },
        "variant_digests": {
            BASELINE_KIND: _digest(baseline_plan),
            CANDIDATE_KIND: _digest(candidate_plan),
        },
        "measurements": {
            BASELINE_KIND: baseline_measurement,
            CANDIDATE_KIND: candidate_measurement,
            "controlled_model_binding": controlled_model_binding,
        },
        "source_binding": binding,
    }


def _maybe_finalize(
    session: Session,
    record: MaintenancePlanningEvaluationRunRecord,
    suite: MaintenancePlanningEvaluationSuiteRecord,
) -> None:
    judgments_by_case: dict[str, tuple[MaintenancePlanningBlindJudgmentRecord, ...]] = {}
    for pair in record.pairs_json:
        case_id = str(pair["case_id"])
        judgments = _case_judgments(session, record.tenant_id, record.evaluation_id, case_id)
        if len(judgments) < record.required_judgments_per_case:
            return
        if _needs_adjudication(judgments) and len(judgments) < 3:
            return
        judgments_by_case[case_id] = judgments

    failure = _source_failure(session, record)
    if failure is not None:
        record.status = "FAILED"
        record.failure_reason = failure
        record.decision = "KEEP_SINGLE_AGENT"
        record.completed_at = datetime.now(UTC)
        return

    metrics = _aggregate_metrics(record, judgments_by_case)
    gates = _gate_results(metrics, suite)
    if record.policy_version == REVIEW_POLICY_VERSION:
        valid_isolation, isolation_evidence = _isolation_evidence(
            session, record, judgments_by_case
        )
        metrics["review_isolation"] = isolation_evidence
        gates["historical_exposure_isolated"] = valid_isolation
    core_gates: tuple[str, ...] = (
        "blind_protocol_complete",
        "independent_judges",
        "controlled_model_binding",
        "performance_measurements_complete",
        "quality_lift",
        "quality_ci_lower_positive",
        "no_factual_regression",
        "no_safety_regression",
        "no_parts_regression",
        "dispatch_non_regression",
        "latency_within_budget",
        "token_cost_within_budget",
        "expert_time_non_regression",
    )
    if record.policy_version == REVIEW_POLICY_VERSION:
        core_gates += ("historical_exposure_isolated",)
    project_eligible = gates["project_sample_size_sufficient"] and all(
        gates[item] for item in core_gates
    )
    production_eligible = (
        project_eligible
        and suite.evidence_tier == "ENTERPRISE_GOLD"
        and gates["production_sample_size_sufficient"]
    )
    decision = (
        "PRODUCTION_CANDIDATE_ELIGIBLE"
        if production_eligible
        else "PROJECT_CANDIDATE_ELIGIBLE"
        if project_eligible
        else "KEEP_SINGLE_AGENT"
    )
    report_document = _evaluation_report_document(
        record,
        suite,
        judgments_by_case,
        metrics,
        gates,
        decision,
    )
    record.aggregate_metrics_json = metrics
    record.gate_results_json = gates
    record.decision = decision
    record.report_hash = _digest(report_document)
    record.failure_reason = None
    record.status = "COMPLETED"
    record.completed_at = datetime.now(UTC)
    record.version += 1


def _blind_protocol_results(
    record: MaintenancePlanningEvaluationRunRecord,
    judgments_by_case: dict[str, tuple[MaintenancePlanningBlindJudgmentRecord, ...]],
) -> dict[str, bool]:
    """Verify persisted judging protocol, not a reviewer\'s prior exposure to sources."""
    case_ids = [str(pair["case_id"]) for pair in record.pairs_json]
    complete = bool(case_ids) and len(set(case_ids)) == len(case_ids)
    complete = complete and set(judgments_by_case) == set(case_ids)
    independent = complete
    required = record.required_judgments_per_case
    complete = complete and required == int(_POLICY["required_judgments_per_case"])
    for pair in record.pairs_json:
        judgments = judgments_by_case.get(str(pair["case_id"]), ())
        subjects = {item.judge_subject_id for item in judgments}
        independent = independent and (
            len(subjects) == len(judgments) >= required
            and record.requested_by_subject_id not in subjects
        )
        expected_count = 3 if _needs_adjudication(judgments) else required
        assignment = pair["assignment"]
        complete = complete and (
            len(judgments) == expected_count
            and set(assignment) == {"A", "B"}
            and set(assignment.values()) == {BASELINE_KIND, CANDIDATE_KIND}
            and all(
                item.tenant_id == record.tenant_id
                and item.evaluation_id == record.evaluation_id
                and item.case_id == str(pair["case_id"])
                and item.blind_assignment_digest == _digest(assignment)
                and item.is_adjudication == (index >= required)
                and item.score_digest
                == _digest(
                    {
                        "variant_a": item.variant_a_scores_json,
                        "variant_b": item.variant_b_scores_json,
                        "preferred_variant": item.preferred_variant,
                        "rationale": item.rationale,
                    }
                )
                for index, item in enumerate(judgments)
            )
        )
    return {
        "blind_protocol_complete": complete and independent,
        "independent_judges": independent,
    }


def _aggregate_metrics(
    record: MaintenancePlanningEvaluationRunRecord,
    judgments_by_case: dict[str, tuple[MaintenancePlanningBlindJudgmentRecord, ...]],
) -> dict[str, Any]:
    case_scores: list[dict[str, dict[str, float]]] = []
    candidate_wins = 0
    baseline_wins = 0
    ties = 0
    for pair in record.pairs_json:
        assignment = pair["assignment"]
        buckets: dict[str, list[dict[str, Any]]] = {BASELINE_KIND: [], CANDIDATE_KIND: []}
        for judgment in judgments_by_case[str(pair["case_id"])]:
            buckets[str(assignment["A"])].append(judgment.variant_a_scores_json)
            buckets[str(assignment["B"])].append(judgment.variant_b_scores_json)
            preferred_kind = (
                None
                if judgment.preferred_variant == "TIE"
                else str(assignment[judgment.preferred_variant])
            )
            if preferred_kind == CANDIDATE_KIND:
                candidate_wins += 1
            elif preferred_kind == BASELINE_KIND:
                baseline_wins += 1
            else:
                ties += 1
        case_scores.append(
            {
                kind: {
                    field: fmean(float(score[field]) for score in buckets[kind])
                    for field in _SCORE_FIELDS
                }
                for kind in (BASELINE_KIND, CANDIDATE_KIND)
            }
        )

    def average(kind: str, field: str) -> float:
        return fmean(item[kind][field] for item in case_scores)

    quality_deltas = [
        item[CANDIDATE_KIND]["plan_quality"] - item[BASELINE_KIND]["plan_quality"]
        for item in case_scores
    ]
    ci_low, ci_high = _paired_bootstrap_interval(
        quality_deltas,
        iterations=int(record.policy_json["bootstrap_iterations"]),
        seed_material=f"{record.policy_hash}:{record.evaluation_id}",
    )
    baseline_latency = fmean(
        float(pair["measurements"][BASELINE_KIND]["latency_ms"]) for pair in record.pairs_json
    )
    candidate_latency = fmean(
        float(pair["measurements"][CANDIDATE_KIND]["latency_ms"]) for pair in record.pairs_json
    )
    baseline_tokens = fmean(
        float(pair["measurements"][BASELINE_KIND]["total_tokens"]) for pair in record.pairs_json
    )
    candidate_tokens = fmean(
        float(pair["measurements"][CANDIDATE_KIND]["total_tokens"]) for pair in record.pairs_json
    )
    return {
        **_blind_protocol_results(record, judgments_by_case),
        "sample_count": len(case_scores),
        "judgment_count": sum(len(item) for item in judgments_by_case.values()),
        "candidate_preference_wins": candidate_wins,
        "baseline_preference_wins": baseline_wins,
        "preference_ties": ties,
        "baseline_plan_quality": average(BASELINE_KIND, "plan_quality"),
        "candidate_plan_quality": average(CANDIDATE_KIND, "plan_quality"),
        "quality_delta": fmean(quality_deltas),
        "quality_ci_low": ci_low,
        "quality_ci_high": ci_high,
        "baseline_factual_errors": average(BASELINE_KIND, "factual_error_count"),
        "candidate_factual_errors": average(CANDIDATE_KIND, "factual_error_count"),
        "baseline_safety_omissions": average(BASELINE_KIND, "safety_omission_count"),
        "candidate_safety_omissions": average(CANDIDATE_KIND, "safety_omission_count"),
        "baseline_parts_false_positives": average(BASELINE_KIND, "parts_false_positive_count"),
        "candidate_parts_false_positives": average(CANDIDATE_KIND, "parts_false_positive_count"),
        "baseline_dispatch_executability": average(BASELINE_KIND, "dispatch_executability"),
        "candidate_dispatch_executability": average(CANDIDATE_KIND, "dispatch_executability"),
        "baseline_expert_review_seconds": average(BASELINE_KIND, "expert_review_seconds"),
        "candidate_expert_review_seconds": average(CANDIDATE_KIND, "expert_review_seconds"),
        "expert_time_savings_seconds": average(BASELINE_KIND, "expert_review_seconds")
        - average(CANDIDATE_KIND, "expert_review_seconds"),
        "baseline_latency_ms": baseline_latency,
        "candidate_latency_ms": candidate_latency,
        "candidate_latency_ratio": _safe_ratio(candidate_latency, baseline_latency),
        "baseline_total_tokens": baseline_tokens,
        "candidate_total_tokens": candidate_tokens,
        "candidate_token_ratio": _safe_ratio(candidate_tokens, baseline_tokens),
        "measurements_complete": all(
            bool(pair["measurements"][kind]["measured"])
            for pair in record.pairs_json
            for kind in (BASELINE_KIND, CANDIDATE_KIND)
        ),
        "controlled_model_binding": all(
            bool(pair["measurements"]["controlled_model_binding"]) for pair in record.pairs_json
        ),
    }


def _gate_results(
    metrics: dict[str, Any],
    suite: MaintenancePlanningEvaluationSuiteRecord,
) -> dict[str, bool]:
    sample_count = int(metrics["sample_count"])
    return {
        "blind_protocol_complete": metrics.get("blind_protocol_complete") is True,
        "independent_judges": metrics.get("independent_judges") is True,
        "project_sample_size_sufficient": sample_count >= int(_POLICY["project_minimum_cases"]),
        "production_sample_size_sufficient": sample_count
        >= int(_POLICY["production_minimum_cases"]),
        "enterprise_gold_tier": suite.evidence_tier == "ENTERPRISE_GOLD",
        "controlled_model_binding": bool(metrics["controlled_model_binding"]),
        "performance_measurements_complete": bool(metrics["measurements_complete"]),
        "quality_lift": float(metrics["quality_delta"]) >= float(_POLICY["quality_lift_min"]),
        "quality_ci_lower_positive": float(metrics["quality_ci_low"])
        > float(_POLICY["quality_ci_lower_exclusive_min"]),
        "no_factual_regression": float(metrics["candidate_factual_errors"])
        <= float(metrics["baseline_factual_errors"]),
        "no_safety_regression": float(metrics["candidate_safety_omissions"])
        <= float(metrics["baseline_safety_omissions"]),
        "no_parts_regression": float(metrics["candidate_parts_false_positives"])
        <= float(metrics["baseline_parts_false_positives"]),
        "dispatch_non_regression": float(metrics["candidate_dispatch_executability"])
        >= float(metrics["baseline_dispatch_executability"]),
        "latency_within_budget": float(metrics["candidate_latency_ratio"])
        <= float(_POLICY["maximum_candidate_latency_ratio"]),
        "token_cost_within_budget": float(metrics["candidate_token_ratio"])
        <= float(_POLICY["maximum_candidate_token_ratio"]),
        "expert_time_non_regression": float(metrics["expert_time_savings_seconds"]) >= 0,
    }


def _source_failure(
    session: Session,
    record: MaintenancePlanningEvaluationRunRecord,
) -> str | None:
    for pair in record.pairs_json:
        binding = pair["source_binding"]
        diagnosis = session.scalar(
            select(DiagnosisRunRecord).where(
                DiagnosisRunRecord.tenant_id == record.tenant_id,
                DiagnosisRunRecord.diagnosis_run_id == binding["diagnosis_run_id"],
            )
        )
        council = session.scalar(
            select(MaintenancePlanningCouncilRecord).where(
                MaintenancePlanningCouncilRecord.tenant_id == record.tenant_id,
                MaintenancePlanningCouncilRecord.council_id == binding["council_id"],
            )
        )
        if (
            diagnosis is None
            or council is None
            or diagnosis.version != binding["diagnosis_version"]
            or _digest(diagnosis.report) != binding["diagnosis_report_digest"]
            or _digest(diagnosis.manifest) != binding["diagnosis_manifest_digest"]
            or council.attempt_count != binding["council_attempt_count"]
            or council.result_digest != binding["council_result_digest"]
            or council.model_release_id != binding["council_model_release_id"]
            or council.model_manifest_hash != binding["council_model_manifest_hash"]
            or not isinstance(council.plan_json, dict)
            or _digest(council.plan_json) != council.result_digest
        ):
            return "maintenance_planning_evaluation_source_binding_changed"
        contributions = tuple(
            session.scalars(
                select(MaintenancePlanningContributionRecord).where(
                    MaintenancePlanningContributionRecord.tenant_id == record.tenant_id,
                    MaintenancePlanningContributionRecord.council_id == council.council_id,
                    MaintenancePlanningContributionRecord.attempt_number == council.attempt_count,
                )
            )
        )
        if sorted(item.output_digest for item in contributions) != binding["contribution_digests"]:
            return "maintenance_planning_evaluation_contribution_binding_changed"
        measurements = {
            "baseline": _baseline_measurement(session, diagnosis),
            "candidate": _candidate_measurement(session, contributions),
        }
        if _digest(measurements) != binding["measurement_digest"]:
            return "maintenance_planning_evaluation_measurement_binding_changed"
    return None


def _baseline_measurement(session: Session, diagnosis: DiagnosisRunRecord) -> dict[str, Any]:
    records = tuple(
        session.scalars(
            select(ModelInferenceRecord).where(
                ModelInferenceRecord.tenant_id == diagnosis.tenant_id,
                ModelInferenceRecord.diagnosis_run_id == diagnosis.diagnosis_run_id,
                ModelInferenceRecord.request_class != "MAINTENANCE_PLANNING",
                ModelInferenceRecord.status == "SUCCEEDED",
            )
        )
    )
    if not records:
        records = tuple(
            session.scalars(
                select(ModelInferenceRecord).where(
                    ModelInferenceRecord.tenant_id == diagnosis.tenant_id,
                    ModelInferenceRecord.agent_run_id == diagnosis.agent_run_id,
                    ModelInferenceRecord.request_class != "MAINTENANCE_PLANNING",
                    ModelInferenceRecord.status == "SUCCEEDED",
                )
            )
        )
    return _serial_measurement(records)


def _candidate_measurement(
    session: Session,
    contributions: tuple[MaintenancePlanningContributionRecord, ...],
) -> dict[str, Any]:
    inference_ids = tuple(item.inference_request_id for item in contributions)
    records = tuple(
        session.scalars(
            select(ModelInferenceRecord).where(
                ModelInferenceRecord.tenant_id == contributions[0].tenant_id,
                ModelInferenceRecord.inference_request_id.in_(inference_ids),
                ModelInferenceRecord.status == "SUCCEEDED",
            )
        )
    )
    by_id = {item.inference_request_id: item for item in records}
    role_latency = {
        item.agent_role: _record_latency(by_id.get(item.inference_request_id))
        for item in contributions
    }
    measured = len(records) == len(contributions) and all(
        value > 0 for value in role_latency.values()
    )
    specialist_latency = max(
        role_latency.get("SAFETY", 0.0),
        role_latency.get("PARTS", 0.0),
        role_latency.get("DISPATCH", 0.0),
    )
    coordinator_latency = role_latency.get("COORDINATOR", 0.0)
    return {
        "measured": measured,
        "request_count": len(records),
        "prompt_tokens": sum(item.usage_prompt_tokens for item in records),
        "completion_tokens": sum(item.usage_completion_tokens for item in records),
        "total_tokens": sum(
            item.usage_prompt_tokens + item.usage_completion_tokens for item in records
        ),
        "latency_ms": specialist_latency + coordinator_latency if measured else 0.0,
        "release_ids": sorted({item.resolved_release_id for item in records}),
        "inference_ids": sorted(item.inference_request_id for item in records),
    }


def _serial_measurement(records: tuple[ModelInferenceRecord, ...]) -> dict[str, Any]:
    latencies = [_record_latency(item) for item in records]
    measured = bool(records) and all(value > 0 for value in latencies)
    return {
        "measured": measured,
        "request_count": len(records),
        "prompt_tokens": sum(item.usage_prompt_tokens for item in records),
        "completion_tokens": sum(item.usage_completion_tokens for item in records),
        "total_tokens": sum(
            item.usage_prompt_tokens + item.usage_completion_tokens for item in records
        ),
        "latency_ms": sum(latencies) if measured else 0.0,
        "release_ids": sorted({item.resolved_release_id for item in records}),
        "inference_ids": sorted(item.inference_request_id for item in records),
    }


def _record_latency(record: ModelInferenceRecord | None) -> float:
    if record is None or not record.latency_breakdown_json:
        return 0.0
    values = [float(value) for value in record.latency_breakdown_json.values()]
    return (
        max(values)
        if values and all(math.isfinite(value) and value >= 0 for value in values)
        else 0.0
    )


def _normalize_baseline_plan(report: dict[str, Any]) -> dict[str, Any]:
    summary = next(
        (
            str(report[key]).strip()
            for key in ("summary", "conclusion", "diagnosis", "status")
            if isinstance(report.get(key), str) and str(report[key]).strip()
        ),
        "单 Agent 未提供结构化维修摘要",
    )
    recommendations = _string_list_from_keys(
        report, ("recommendations", "recommended_actions", "next_checks", "actions")
    )
    if not recommendations:
        recommendations = [summary]
    return {
        "summary": summary,
        "recommendations": recommendations,
        "constraints": _string_list_from_keys(report, ("constraints", "limitations")),
        "missing_facts": _string_list_from_keys(report, ("missing_facts", "missing_information")),
        "safety_hold_points": _string_list_from_keys(
            report, ("safety_hold_points", "safety_constraints")
        ),
        "required_parts": _parts(report.get("required_parts", report.get("parts", []))),
        "dispatch_constraints": _string_list_from_keys(
            report, ("dispatch_constraints", "skills_required")
        ),
        "readiness": "READY_FOR_BLIND_REVIEW",
    }


def _anonymize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": str(plan.get("summary", "")).strip(),
        "recommendations": _strings(plan.get("recommendations", [])),
        "constraints": _strings(plan.get("constraints", [])),
        "missing_facts": _strings(plan.get("missing_facts", [])),
        "safety_hold_points": _strings(plan.get("safety_hold_points", [])),
        "required_parts": _parts(plan.get("required_parts", [])),
        "dispatch_constraints": _strings(plan.get("dispatch_constraints", [])),
        "readiness": "READY_FOR_BLIND_REVIEW",
    }


def _validated_rubric(case: MaintenancePlanningGoldCaseInput) -> dict[str, Any]:
    rubric = {
        "expected_findings": _validated_reference_items(case.expected_findings, required=True),
        "required_safety_hold_points": _validated_reference_items(
            case.required_safety_hold_points, required=case.high_risk
        ),
        "allowed_parts": _validated_reference_items(case.allowed_parts, required=False),
        "required_dispatch_constraints": _validated_reference_items(
            case.required_dispatch_constraints, required=True
        ),
    }
    return rubric


def _validated_reference_items(values: tuple[str, ...], *, required: bool) -> list[str]:
    normalized = [item.strip() for item in values]
    if (
        (required and not normalized)
        or len(normalized) > 20
        or any(not item or len(item) > 255 for item in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise MaintenancePlanningEvaluationConflict(
            "maintenance_planning_evaluation_rubric_invalid"
        )
    return normalized


def _validated_scores(score: MaintenancePlanningPlanScore) -> dict[str, Any]:
    values = {
        "plan_quality": float(score.plan_quality),
        "factual_error_count": score.factual_error_count,
        "safety_omission_count": score.safety_omission_count,
        "parts_false_positive_count": score.parts_false_positive_count,
        "dispatch_executability": float(score.dispatch_executability),
        "expert_review_seconds": float(score.expert_review_seconds),
    }
    if (
        not 1 <= values["plan_quality"] <= 5
        or not 1 <= values["dispatch_executability"] <= 5
        or not 1 <= values["expert_review_seconds"] <= 3_600
        or any(
            isinstance(values[field], bool)
            or not isinstance(values[field], int)
            or not 0 <= values[field] <= 100
            for field in (
                "factual_error_count",
                "safety_omission_count",
                "parts_false_positive_count",
            )
        )
        or any(
            not math.isfinite(float(values[field]))
            for field in ("plan_quality", "dispatch_executability", "expert_review_seconds")
        )
    ):
        raise MaintenancePlanningEvaluationConflict(
            "maintenance_planning_evaluation_scores_invalid"
        )
    return values


def _paired_bootstrap_interval(
    deltas: list[float],
    *,
    iterations: int,
    seed_material: str,
) -> tuple[float, float]:
    if not deltas:
        raise MaintenancePlanningEvaluationConflict(
            "maintenance_planning_evaluation_scores_missing"
        )
    rng = random.Random(int(sha256(seed_material.encode()).hexdigest()[:16], 16))
    samples = sorted(
        fmean(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(iterations)
    )
    lower = samples[max(0, int(iterations * 0.025) - 1)]
    upper = samples[min(iterations - 1, int(iterations * 0.975))]
    return lower, upper


def _run_view(
    session: Session,
    record: MaintenancePlanningEvaluationRunRecord,
    suite: MaintenancePlanningEvaluationSuiteRecord,
    current_subject_id: str,
) -> MaintenancePlanningEvaluationRunView:
    all_judgments = tuple(
        session.scalars(
            select(MaintenancePlanningBlindJudgmentRecord).where(
                MaintenancePlanningBlindJudgmentRecord.tenant_id == record.tenant_id,
                MaintenancePlanningBlindJudgmentRecord.evaluation_id == record.evaluation_id,
            )
        )
    )
    by_case: dict[str, list[MaintenancePlanningBlindJudgmentRecord]] = {}
    for judgment in all_judgments:
        by_case.setdefault(judgment.case_id, []).append(judgment)
    reveal = record.status == "COMPLETED"
    pair_views: list[MaintenancePlanningBlindPairView] = []
    can_judge = False
    for pair in record.pairs_json:
        case_id = str(pair["case_id"])
        judgments = tuple(by_case.get(case_id, []))
        assignment = pair["assignment"]
        judged = any(item.judge_subject_id == current_subject_id for item in judgments)
        adjudication = _needs_adjudication(judgments)
        case_open = len(judgments) < record.required_judgments_per_case or (
            adjudication and len(judgments) < 3
        )
        can_judge = can_judge or (case_open and not judged)
        pair_views.append(
            MaintenancePlanningBlindPairView(
                case_id=case_id,
                diagnosis_run_id=str(pair["diagnosis_run_id"]),
                incident_id=str(pair["incident_id"]),
                asset_id=str(pair["asset_id"]),
                high_risk=bool(pair["high_risk"]),
                rubric=dict(pair["rubric"]),
                variant_a=dict(pair["variants"][assignment["A"]]),
                variant_b=dict(pair["variants"][assignment["B"]]),
                judgment_count=len(judgments),
                required_judgment_count=(3 if adjudication else 2),
                needs_adjudication=adjudication and len(judgments) < 3,
                judged_by_current_subject=judged,
                revealed_assignment=dict(assignment) if reveal else None,
            )
        )
    legal_actions: tuple[str, ...] = ()
    return MaintenancePlanningEvaluationRunView(
        evaluation_id=record.evaluation_id,
        suite_id=suite.suite_id,
        suite_name=suite.name,
        suite_version=suite.suite_version,
        evidence_tier=suite.evidence_tier,
        status=record.status,
        policy_version=record.policy_version,
        policy_hash=record.policy_hash,
        policy=dict(record.policy_json),
        pairs=tuple(pair_views),
        required_judgments_per_case=record.required_judgments_per_case,
        aggregate_metrics=dict(record.aggregate_metrics_json),
        gate_results=dict(record.gate_results_json),
        decision=record.decision,
        report_hash=record.report_hash,
        failure_reason=record.failure_reason,
        requested_by_subject_id=record.requested_by_subject_id,
        completed_at=record.completed_at,
        version=record.version,
        legal_actions=legal_actions,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _case_judgments(
    session: Session,
    tenant_id: str,
    evaluation_id: str,
    case_id: str,
) -> tuple[MaintenancePlanningBlindJudgmentRecord, ...]:
    return tuple(
        session.scalars(
            select(MaintenancePlanningBlindJudgmentRecord)
            .where(
                MaintenancePlanningBlindJudgmentRecord.tenant_id == tenant_id,
                MaintenancePlanningBlindJudgmentRecord.evaluation_id == evaluation_id,
                MaintenancePlanningBlindJudgmentRecord.case_id == case_id,
            )
            .order_by(
                MaintenancePlanningBlindJudgmentRecord.submitted_at,
                MaintenancePlanningBlindJudgmentRecord.judgment_id,
            )
        )
    )


def _needs_adjudication(
    judgments: tuple[MaintenancePlanningBlindJudgmentRecord, ...],
) -> bool:
    if len(judgments) < 2:
        return False
    return judgments[0].preferred_variant != judgments[1].preferred_variant


def _suite_view(
    record: MaintenancePlanningEvaluationSuiteRecord,
) -> MaintenancePlanningEvaluationSuiteView:
    return MaintenancePlanningEvaluationSuiteView(
        suite_id=record.suite_id,
        name=record.name,
        suite_version=record.suite_version,
        evidence_tier=record.evidence_tier,
        status=record.status,
        cases=tuple(dict(item) for item in record.cases_json),
        sample_count=record.sample_count,
        manifest_hash=record.manifest_hash,
        created_by_subject_id=record.created_by_subject_id,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _diagnosis_or_hidden(
    session: Session,
    tenant_id: str,
    diagnosis_run_id: str,
) -> DiagnosisRunRecord:
    record = session.scalar(
        select(DiagnosisRunRecord).where(
            DiagnosisRunRecord.tenant_id == tenant_id,
            DiagnosisRunRecord.diagnosis_run_id == diagnosis_run_id,
        )
    )
    if record is None:
        raise MaintenancePlanningEvaluationNotVisible
    return record


def _suite_or_hidden(
    session: Session,
    tenant_id: str,
    suite_id: str,
) -> MaintenancePlanningEvaluationSuiteRecord:
    record = session.scalar(
        select(MaintenancePlanningEvaluationSuiteRecord).where(
            MaintenancePlanningEvaluationSuiteRecord.tenant_id == tenant_id,
            MaintenancePlanningEvaluationSuiteRecord.suite_id == suite_id,
        )
    )
    if record is None:
        raise MaintenancePlanningEvaluationNotVisible
    return record


def _run_or_hidden(
    session: Session,
    tenant_id: str,
    evaluation_id: str,
    *,
    lock: bool = False,
) -> MaintenancePlanningEvaluationRunRecord:
    statement = select(MaintenancePlanningEvaluationRunRecord).where(
        MaintenancePlanningEvaluationRunRecord.tenant_id == tenant_id,
        MaintenancePlanningEvaluationRunRecord.evaluation_id == evaluation_id,
    )
    if lock:
        statement = statement.with_for_update()
    record = session.scalar(statement)
    if record is None:
        raise MaintenancePlanningEvaluationNotVisible
    return record


def _assets_visible(identity: IdentityContext, cases: list[dict[str, Any]]) -> bool:
    return all(str(item.get("asset_id", "")) in identity.asset_ids for item in cases)


def _string_list_from_keys(report: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        value = report.get(key)
        if isinstance(value, list):
            return _strings(value)
    return []


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()][:50]


def _parts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value[:50]:
        if isinstance(item, str) and item.strip():
            result.append({"part_reference": item.strip(), "quantity": 1})
        elif isinstance(item, dict):
            reference = item.get("part_reference", item.get("part_id", item.get("name")))
            if isinstance(reference, str) and reference.strip():
                result.append(
                    {
                        "part_reference": reference.strip(),
                        "quantity": item.get("quantity", 1),
                        "reason": str(item.get("reason", "")).strip(),
                    }
                )
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 1_000_000_000.0


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
