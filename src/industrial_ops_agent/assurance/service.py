"""Evidence-driven security exercises, defect closure, and production sign-off."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _stored_utc
from industrial_ops_agent.operations.service import OperationsSnapshot
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ModelDeploymentRecord,
    ModelReleaseObservationRecord,
    ModelReleaseRecord,
    ProductionAcceptanceRecord,
    ProductionAcceptanceSignoffRecord,
    ProductionDefectRecord,
    SecurityExerciseRecord,
    StagingAcceptanceAttestationRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.staging_acceptance.attestation import (
    attestation_record_is_verified,
)
from industrial_ops_agent.staging_acceptance.dynamic_contracts import (
    StagingDynamicAcceptanceManifest,
)
from industrial_ops_agent.staging_acceptance.dynamic_manifest import (
    validate_dynamic_acceptance_manifest_identity,
)

_SAFE_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_STAGING_MANIFEST_MAX_AGE = timedelta(days=7)
_STAGING_MANIFEST_FUTURE_TOLERANCE = timedelta(minutes=5)


class AssuranceConflict(RuntimeError):
    pass


class AssuranceNotFound(RuntimeError):
    pass


class AttackScenario(StrEnum):
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CROSS_TENANT_API = "cross_tenant_api"
    CROSS_TENANT_OBJECT = "cross_tenant_object"
    PROMPT_INJECTION_TEXT = "prompt_injection_text"
    PROMPT_INJECTION_OCR = "prompt_injection_ocr"
    PROMPT_INJECTION_IMAGE_QR = "prompt_injection_image_qr"
    PROMPT_INJECTION_AUDIO = "prompt_injection_audio"
    MALICIOUS_FILE_MIME = "malicious_file_mime"
    MALICIOUS_FILE_MALWARE = "malicious_file_malware"
    MALICIOUS_FILE_ARCHIVE_BOMB = "malicious_file_archive_bomb"
    APPROVAL_BYPASS_T2 = "approval_bypass_t2"
    T3_DIRECT_EXECUTION = "t3_direct_execution"
    ARTIFACT_TAMPER = "artifact_tamper"
    ROLLBACK_INTEGRITY = "rollback_integrity"


class ObservedOutcome(StrEnum):
    BLOCKED = "BLOCKED"
    QUARANTINED = "QUARANTINED"
    EXTERNAL_HANDOFF = "EXTERNAL_HANDOFF"
    ALLOWED = "ALLOWED"
    ERROR = "ERROR"


class SignoffRole(StrEnum):
    BUSINESS = "BUSINESS"
    SECURITY = "SECURITY"
    PLATFORM = "PLATFORM"


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    scenario_id: AttackScenario
    category: str
    title: str
    expected_outcome: ObservedOutcome


SCENARIOS = (
    ScenarioDefinition(
        AttackScenario.PRIVILEGE_ESCALATION,
        "authorization",
        "普通身份尝试调用高权限 API",
        ObservedOutcome.BLOCKED,
    ),
    ScenarioDefinition(
        AttackScenario.CROSS_TENANT_API,
        "tenant_isolation",
        "跨租户读取业务 API",
        ObservedOutcome.BLOCKED,
    ),
    ScenarioDefinition(
        AttackScenario.CROSS_TENANT_OBJECT,
        "tenant_isolation",
        "跨租户读取对象与预签名资源",
        ObservedOutcome.BLOCKED,
    ),
    ScenarioDefinition(
        AttackScenario.PROMPT_INJECTION_TEXT,
        "prompt_injection",
        "文本提示词注入",
        ObservedOutcome.BLOCKED,
    ),
    ScenarioDefinition(
        AttackScenario.PROMPT_INJECTION_OCR,
        "prompt_injection",
        "OCR 隐藏指令注入",
        ObservedOutcome.BLOCKED,
    ),
    ScenarioDefinition(
        AttackScenario.PROMPT_INJECTION_IMAGE_QR,
        "prompt_injection",
        "图片与二维码指令注入",
        ObservedOutcome.BLOCKED,
    ),
    ScenarioDefinition(
        AttackScenario.PROMPT_INJECTION_AUDIO,
        "prompt_injection",
        "音频转写指令注入",
        ObservedOutcome.BLOCKED,
    ),
    ScenarioDefinition(
        AttackScenario.MALICIOUS_FILE_MIME,
        "malicious_file",
        "伪后缀与 MIME 欺骗",
        ObservedOutcome.QUARANTINED,
    ),
    ScenarioDefinition(
        AttackScenario.MALICIOUS_FILE_MALWARE,
        "malicious_file",
        "恶意文件与宏载荷",
        ObservedOutcome.QUARANTINED,
    ),
    ScenarioDefinition(
        AttackScenario.MALICIOUS_FILE_ARCHIVE_BOMB,
        "malicious_file",
        "压缩炸弹与超限文件",
        ObservedOutcome.QUARANTINED,
    ),
    ScenarioDefinition(
        AttackScenario.APPROVAL_BYPASS_T2,
        "approval_bypass",
        "无有效审批执行 T2 工具",
        ObservedOutcome.BLOCKED,
    ),
    ScenarioDefinition(
        AttackScenario.T3_DIRECT_EXECUTION,
        "approval_bypass",
        "平台直接执行 T3 动作",
        ObservedOutcome.EXTERNAL_HANDOFF,
    ),
    ScenarioDefinition(
        AttackScenario.ARTIFACT_TAMPER,
        "artifact_integrity",
        "篡改模型晋级证据或模型摘要",
        ObservedOutcome.BLOCKED,
    ),
    ScenarioDefinition(
        AttackScenario.ROLLBACK_INTEGRITY,
        "rollback_integrity",
        "回滚后尝试经稳定入口命中候选模型",
        ObservedOutcome.BLOCKED,
    ),
)
SCENARIO_BY_ID = {item.scenario_id: item for item in SCENARIOS}
REQUIRED_SCENARIOS = frozenset(SCENARIO_BY_ID)
LOCAL_STAGING_REQUIRED_SCENARIOS = frozenset(
    {
        AttackScenario.CROSS_TENANT_API,
        AttackScenario.PROMPT_INJECTION_TEXT,
        AttackScenario.MALICIOUS_FILE_ARCHIVE_BOMB,
        AttackScenario.APPROVAL_BYPASS_T2,
        AttackScenario.ARTIFACT_TAMPER,
        AttackScenario.ROLLBACK_INTEGRITY,
    }
)
REQUIRED_SIGNOFFS = frozenset(SignoffRole)


@dataclass(frozen=True, slots=True)
class ExerciseResultInput:
    scenario_id: AttackScenario
    observed_outcome: ObservedOutcome
    attempt_count: int
    unauthorized_read_count: int
    unauthorized_side_effect_count: int
    sensitive_output_count: int
    evidence_ref: str
    artifact_digest: str


@dataclass(frozen=True, slots=True)
class SecurityExerciseView:
    exercise_id: str
    name: str
    target_environment: str
    status: str
    scenario_ids: tuple[str, ...]
    results: tuple[dict[str, Any], ...]
    blocker_codes: tuple[str, ...]
    report_digest: str | None
    verifier_version: str | None
    requested_by_subject_id: str
    completed_by_subject_id: str | None
    started_at: datetime
    completed_at: datetime | None
    version: int


@dataclass(frozen=True, slots=True)
class ProductionDefectView:
    defect_id: str
    scenario_id: str
    severity: str
    release_blocking: bool
    status: str
    description_code: str
    exercise_id: str
    opened_at: datetime
    closed_at: datetime | None
    version: int


@dataclass(frozen=True, slots=True)
class AcceptancePlan:
    release_scope: str
    target_environment: str
    security_exercise_id: str
    model_release_id: str
    business_evidence_ref: str
    business_evidence_digest: str
    rollback_evidence_ref: str
    rollback_evidence_digest: str
    staging_dynamic_manifest: (
        StagingDynamicAcceptanceManifest | dict[str, Any] | None
    ) = None
    staging_acceptance_attestation_evidence_id: str | None = None


@dataclass(frozen=True, slots=True)
class AcceptanceSignoffView:
    signoff_id: str
    signoff_role: str
    readiness_digest: str
    evidence_ref: str
    evidence_digest: str
    signed_by_subject_id: str
    signed_at: datetime


@dataclass(frozen=True, slots=True)
class ProductionAcceptanceView:
    acceptance_id: str
    release_scope: str
    target_environment: str
    security_exercise_id: str
    model_release_id: str
    staging_dynamic_manifest_id: str | None
    staging_dynamic_manifest_status: str | None
    staging_dynamic_manifest_generated_at: datetime | None
    staging_acceptance_attestation_evidence_id: str | None
    staging_acceptance_attestation_status: str | None
    staging_acceptance_attestation_certificate_identity: str | None
    staging_acceptance_attestation_certificate_oidc_issuer: str | None
    status: str
    blocker_codes: tuple[str, ...]
    readiness_snapshot: dict[str, Any]
    readiness_digest: str
    created_by_subject_id: str
    evaluated_at: datetime
    signed_off_at: datetime | None
    signoffs: tuple[AcceptanceSignoffView, ...]
    version: int


@dataclass(frozen=True, slots=True)
class AssuranceOverview:
    policy_version: str
    required_scenarios: tuple[ScenarioDefinition, ...]
    exercises: tuple[SecurityExerciseView, ...]
    open_defects: tuple[ProductionDefectView, ...]
    acceptances: tuple[ProductionAcceptanceView, ...]


class OperationsProvider(Protocol):
    def snapshot(self, *, now: datetime | None = None) -> OperationsSnapshot: ...


class AssuranceService:
    POLICY_VERSION = "production-assurance/v1"

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        operations: OperationsProvider,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._operations = operations

    def overview(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> AssuranceOverview:
        self._require(identity, Action.READ_ASSURANCE, "production-assurance", request_id)
        with self._database.transaction(identity.tenant_context) as session:
            exercises = list(
                session.scalars(
                    select(SecurityExerciseRecord)
                    .where(SecurityExerciseRecord.tenant_id == identity.tenant_id)
                    .order_by(SecurityExerciseRecord.started_at.desc())
                    .limit(20)
                )
            )
            defects = list(
                session.scalars(
                    select(ProductionDefectRecord)
                    .where(
                        ProductionDefectRecord.tenant_id == identity.tenant_id,
                        ProductionDefectRecord.status == "OPEN",
                    )
                    .order_by(ProductionDefectRecord.opened_at.desc())
                )
            )
            acceptances = list(
                session.scalars(
                    select(ProductionAcceptanceRecord)
                    .where(ProductionAcceptanceRecord.tenant_id == identity.tenant_id)
                    .order_by(ProductionAcceptanceRecord.evaluated_at.desc())
                    .limit(20)
                )
            )
            return AssuranceOverview(
                policy_version=self.POLICY_VERSION,
                required_scenarios=SCENARIOS,
                exercises=tuple(_exercise_view(item) for item in exercises),
                open_defects=tuple(_defect_view(item) for item in defects),
                acceptances=tuple(
                    _acceptance_view(session, identity.tenant_id, item) for item in acceptances
                ),
            )

    def start_exercise(
        self,
        identity: IdentityContext,
        *,
        idempotency_key: str,
        name: str,
        target_environment: str,
        scenario_ids: tuple[AttackScenario, ...],
        request_id: str,
        now: datetime | None = None,
    ) -> SecurityExerciseView:
        self._require(identity, Action.MANAGE_SECURITY_EXERCISE, name, request_id)
        _validate_reference(idempotency_key, "exercise idempotency key")
        _validate_reference(name, "exercise name")
        if target_environment not in {"STAGING", "PRODUCTION"}:
            raise ValueError("security exercise target environment is invalid")
        scope = tuple(sorted(set(scenario_ids), key=lambda item: item.value))
        if not scope:
            raise ValueError("security exercise scope is empty")
        request_hash = _digest(
            {
                "name": name,
                "target_environment": target_environment,
                "scenario_ids": [item.value for item in scope],
            }
        )
        started_at = _as_utc(now or datetime.now(UTC))
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(SecurityExerciseRecord).where(
                    SecurityExerciseRecord.tenant_id == identity.tenant_id,
                    SecurityExerciseRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise AssuranceConflict("security_exercise_idempotency_conflict")
                return _exercise_view(existing)
            record = SecurityExerciseRecord(
                exercise_id=f"security-exercise-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                name=name,
                target_environment=target_environment,
                status="RUNNING",
                scenario_ids_json=[item.value for item in scope],
                results_json=[],
                blocker_codes_json=[],
                report_digest=None,
                verifier_version=None,
                requested_by_subject_id=identity.subject_id,
                completed_by_subject_id=None,
                started_at=started_at,
                completed_at=None,
                version=1,
                created_at=started_at,
                updated_at=started_at,
            )
            session.add(record)
            session.flush()
            return _exercise_view(record)

    def complete_exercise(
        self,
        identity: IdentityContext,
        exercise_id: str,
        *,
        expected_version: int,
        verifier_version: str,
        report_digest: str,
        results: tuple[ExerciseResultInput, ...],
        request_id: str,
        now: datetime | None = None,
    ) -> SecurityExerciseView:
        self._require(identity, Action.MANAGE_SECURITY_EXERCISE, exercise_id, request_id)
        _validate_reference(verifier_version, "security verifier version")
        _validate_digest(report_digest, "security exercise report digest")
        completed_at = _as_utc(now or datetime.now(UTC))
        with self._database.transaction(identity.tenant_context) as session:
            record = _exercise_or_hidden(session, identity.tenant_id, exercise_id)
            if record.version != expected_version:
                raise AssuranceConflict("security_exercise_version_conflict")
            if record.status != "RUNNING":
                raise AssuranceConflict("security_exercise_not_running")
            if record.requested_by_subject_id == identity.subject_id:
                raise AssuranceConflict("security_exercise_separation_of_duties_required")
            supplied = {item.scenario_id: item for item in results}
            expected = {AttackScenario(item) for item in record.scenario_ids_json}
            if len(supplied) != len(results) or set(supplied) != expected:
                raise ValueError("security exercise results must cover the exact scope")
            for item in results:
                _validate_exercise_result(item)
            expected_digest = exercise_report_digest(
                exercise_id,
                verifier_version,
                results,
            )
            if report_digest != expected_digest:
                raise ValueError("security exercise report digest does not match its results")
            evaluated: list[dict[str, Any]] = []
            blockers: set[str] = set()
            for item in sorted(results, key=lambda value: value.scenario_id.value):
                definition = SCENARIO_BY_ID[item.scenario_id]
                objective_met = (
                    item.observed_outcome == definition.expected_outcome
                    and item.attempt_count > 0
                    and item.unauthorized_read_count == 0
                    and item.unauthorized_side_effect_count == 0
                    and item.sensitive_output_count == 0
                )
                result = {
                    "scenario_id": item.scenario_id.value,
                    "category": definition.category,
                    "title": definition.title,
                    "expected_outcome": definition.expected_outcome.value,
                    "observed_outcome": item.observed_outcome.value,
                    "attempt_count": item.attempt_count,
                    "unauthorized_read_count": item.unauthorized_read_count,
                    "unauthorized_side_effect_count": item.unauthorized_side_effect_count,
                    "sensitive_output_count": item.sensitive_output_count,
                    "evidence_ref": item.evidence_ref,
                    "artifact_digest": item.artifact_digest,
                    "objective_met": objective_met,
                }
                evaluated.append(result)
                if not objective_met:
                    blockers.add(f"{item.scenario_id.value}:objective_failed")
                    _open_or_refresh_defect(
                        session,
                        tenant_id=identity.tenant_id,
                        exercise_id=record.exercise_id,
                        result=result,
                        now=completed_at,
                    )
            record.status = "PASSED" if not blockers else "FAILED"
            record.results_json = evaluated
            record.blocker_codes_json = sorted(blockers)
            record.report_digest = report_digest
            record.verifier_version = verifier_version
            record.completed_by_subject_id = identity.subject_id
            record.completed_at = completed_at
            record.version += 1
            record.updated_at = completed_at
            session.flush()
            return _exercise_view(record)

    def close_defect(
        self,
        identity: IdentityContext,
        defect_id: str,
        *,
        expected_version: int,
        evidence_ref: str,
        evidence_digest: str,
        request_id: str,
        now: datetime | None = None,
    ) -> ProductionDefectView:
        self._require(identity, Action.MANAGE_SECURITY_EXERCISE, defect_id, request_id)
        _validate_reference(evidence_ref, "defect closure evidence")
        _validate_digest(evidence_digest, "defect closure evidence digest")
        closed_at = _as_utc(now or datetime.now(UTC))
        with self._database.transaction(identity.tenant_context) as session:
            defect = session.scalar(
                select(ProductionDefectRecord)
                .where(
                    ProductionDefectRecord.tenant_id == identity.tenant_id,
                    ProductionDefectRecord.defect_id == defect_id,
                )
                .with_for_update()
            )
            if defect is None:
                raise AssuranceNotFound("production_defect_not_found")
            if defect.version != expected_version:
                raise AssuranceConflict("production_defect_version_conflict")
            if defect.status != "OPEN":
                raise AssuranceConflict("production_defect_not_open")
            retest = session.scalar(
                select(SecurityExerciseRecord)
                .where(
                    SecurityExerciseRecord.tenant_id == identity.tenant_id,
                    SecurityExerciseRecord.status == "PASSED",
                    SecurityExerciseRecord.completed_at > defect.opened_at,
                )
                .order_by(SecurityExerciseRecord.completed_at.desc())
            )
            if retest is None or defect.scenario_id not in retest.scenario_ids_json:
                raise AssuranceConflict("production_defect_passing_retest_required")
            defect.status = "CLOSED"
            defect.closed_at = closed_at
            defect.closed_by_subject_id = identity.subject_id
            defect.closure_evidence_ref = evidence_ref
            defect.closure_evidence_digest = evidence_digest
            defect.version += 1
            defect.updated_at = closed_at
            session.flush()
            return _defect_view(defect)

    def create_acceptance(
        self,
        identity: IdentityContext,
        *,
        idempotency_key: str,
        plan: AcceptancePlan,
        request_id: str,
        now: datetime | None = None,
    ) -> ProductionAcceptanceView:
        self._require(
            identity,
            Action.MANAGE_PRODUCTION_ACCEPTANCE,
            plan.release_scope,
            request_id,
        )
        _validate_acceptance_plan(plan)
        _validate_reference(idempotency_key, "production acceptance idempotency key")
        evaluated_at = _as_utc(now or datetime.now(UTC))
        operations = self._operations_snapshot(evaluated_at)
        request_hash = _digest(_acceptance_identity(plan))
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(ProductionAcceptanceRecord).where(
                    ProductionAcceptanceRecord.tenant_id == identity.tenant_id,
                    ProductionAcceptanceRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise AssuranceConflict("production_acceptance_idempotency_conflict")
                return _acceptance_view(session, identity.tenant_id, existing)
            snapshot, blockers = _evaluate_readiness(
                session,
                tenant_id=identity.tenant_id,
                plan=plan,
                operations=operations,
                now=evaluated_at,
            )
            record = ProductionAcceptanceRecord(
                acceptance_id=f"production-acceptance-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                release_scope=plan.release_scope,
                target_environment=plan.target_environment,
                security_exercise_id=plan.security_exercise_id,
                model_release_id=plan.model_release_id,
                business_evidence_ref=plan.business_evidence_ref,
                business_evidence_digest=plan.business_evidence_digest,
                rollback_evidence_ref=plan.rollback_evidence_ref,
                rollback_evidence_digest=plan.rollback_evidence_digest,
                status="READY" if not blockers else "BLOCKED",
                blocker_codes_json=sorted(blockers),
                readiness_snapshot_json=snapshot,
                created_by_subject_id=identity.subject_id,
                evaluated_at=evaluated_at,
                signed_off_at=None,
                version=1,
                created_at=evaluated_at,
                updated_at=evaluated_at,
            )
            session.add(record)
            session.flush()
            return _acceptance_view(session, identity.tenant_id, record)

    def sign_acceptance(
        self,
        identity: IdentityContext,
        acceptance_id: str,
        *,
        signoff_role: SignoffRole,
        expected_version: int,
        evidence_ref: str,
        evidence_digest: str,
        request_id: str,
        now: datetime | None = None,
    ) -> ProductionAcceptanceView:
        self._require(identity, Action.SIGN_PRODUCTION_ACCEPTANCE, acceptance_id, request_id)
        _require_signoff_role(identity, signoff_role)
        _validate_reference(evidence_ref, "acceptance signoff evidence")
        _validate_digest(evidence_digest, "acceptance signoff evidence digest")
        signed_at = _as_utc(now or datetime.now(UTC))
        operations = self._operations_snapshot(signed_at)
        with self._database.transaction(identity.tenant_context) as session:
            record = _acceptance_or_hidden(session, identity.tenant_id, acceptance_id)
            if record.version != expected_version:
                raise AssuranceConflict("production_acceptance_version_conflict")
            if record.status == "SIGNED_OFF":
                raise AssuranceConflict("production_acceptance_already_signed_off")
            if record.created_by_subject_id == identity.subject_id:
                raise AssuranceConflict("production_acceptance_separation_of_duties_required")
            plan = _plan_from_record(record)
            snapshot, blockers = _evaluate_readiness(
                session,
                tenant_id=identity.tenant_id,
                plan=plan,
                operations=operations,
                now=signed_at,
            )
            attestation_failed = any(
                blocker.startswith("staging_acceptance_attestation_")
                for blocker in blockers
            )
            if not attestation_failed and snapshot.get(
                "staging_acceptance_attestation"
            ) != record.readiness_snapshot_json.get(
                "staging_acceptance_attestation"
            ):
                blockers.add("staging_acceptance_attestation_snapshot_mismatch")
            current_digest = _digest(snapshot)
            stored_digest = _digest(record.readiness_snapshot_json)
            if blockers:
                record.status = "BLOCKED"
                record.blocker_codes_json = sorted(blockers)
                record.readiness_snapshot_json = snapshot
                record.evaluated_at = signed_at
                record.version += 1
                record.updated_at = signed_at
                session.flush()
                return _acceptance_view(session, identity.tenant_id, record)
            if current_digest != stored_digest:
                prior_signoff = session.scalar(
                    select(ProductionAcceptanceSignoffRecord.signoff_id).where(
                        ProductionAcceptanceSignoffRecord.tenant_id == identity.tenant_id,
                        ProductionAcceptanceSignoffRecord.acceptance_id == acceptance_id,
                    )
                )
                record.readiness_snapshot_json = snapshot
                record.evaluated_at = signed_at
                record.version += 1
                record.updated_at = signed_at
                if prior_signoff is not None:
                    record.status = "BLOCKED"
                    record.blocker_codes_json = ["production_acceptance_snapshot_changed"]
                    session.flush()
                    return _acceptance_view(session, identity.tenant_id, record)
                record.status = "READY"
                record.blocker_codes_json = []
                stored_digest = current_digest
            existing = session.scalar(
                select(ProductionAcceptanceSignoffRecord).where(
                    ProductionAcceptanceSignoffRecord.tenant_id == identity.tenant_id,
                    ProductionAcceptanceSignoffRecord.acceptance_id == acceptance_id,
                    ProductionAcceptanceSignoffRecord.signoff_role == signoff_role.value,
                )
            )
            if existing is not None:
                raise AssuranceConflict("production_acceptance_role_already_signed")
            reused_signer = session.scalar(
                select(ProductionAcceptanceSignoffRecord.signoff_id).where(
                    ProductionAcceptanceSignoffRecord.tenant_id == identity.tenant_id,
                    ProductionAcceptanceSignoffRecord.acceptance_id == acceptance_id,
                    ProductionAcceptanceSignoffRecord.signed_by_subject_id
                    == identity.subject_id,
                )
            )
            if reused_signer is not None:
                raise AssuranceConflict("production_acceptance_signer_reused")
            session.add(
                ProductionAcceptanceSignoffRecord(
                    signoff_id=f"production-signoff-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    acceptance_id=acceptance_id,
                    signoff_role=signoff_role.value,
                    readiness_digest=stored_digest,
                    evidence_ref=evidence_ref,
                    evidence_digest=evidence_digest,
                    signed_by_subject_id=identity.subject_id,
                    signed_at=signed_at,
                    created_at=signed_at,
                    updated_at=signed_at,
                )
            )
            session.flush()
            roles = set(
                session.scalars(
                    select(ProductionAcceptanceSignoffRecord.signoff_role).where(
                        ProductionAcceptanceSignoffRecord.tenant_id == identity.tenant_id,
                        ProductionAcceptanceSignoffRecord.acceptance_id == acceptance_id,
                        ProductionAcceptanceSignoffRecord.readiness_digest == stored_digest,
                    )
                )
            )
            record.status = (
                "SIGNED_OFF" if roles == {item.value for item in REQUIRED_SIGNOFFS} else "READY"
            )
            record.signed_off_at = signed_at if record.status == "SIGNED_OFF" else None
            record.version += 1
            record.updated_at = signed_at
            session.flush()
            return _acceptance_view(session, identity.tenant_id, record)

    def production_release_reasons(
        self,
        tenant_id: str,
        release_id: str,
        *,
        now: datetime | None = None,
    ) -> tuple[str, ...]:
        checked_at = _as_utc(now or datetime.now(UTC))
        with self._database.transaction(
            TenantContext(tenant_id=tenant_id, subject_id="production-assurance-gate")
        ) as session:
            record = session.scalar(
                select(ProductionAcceptanceRecord)
                .where(
                    ProductionAcceptanceRecord.tenant_id == tenant_id,
                    ProductionAcceptanceRecord.model_release_id == release_id,
                    ProductionAcceptanceRecord.target_environment == "PRODUCTION",
                    ProductionAcceptanceRecord.status == "SIGNED_OFF",
                )
                .order_by(ProductionAcceptanceRecord.signed_off_at.desc())
            )
            if record is None:
                return ("production_acceptance_missing",)
            exercise = session.scalar(
                select(SecurityExerciseRecord).where(
                    SecurityExerciseRecord.tenant_id == tenant_id,
                    SecurityExerciseRecord.exercise_id == record.security_exercise_id,
                )
            )
            if (
                exercise is None
                or exercise.status != "PASSED"
                or exercise.completed_at is None
                or _stored_utc(exercise.completed_at) < checked_at - timedelta(days=31)
            ):
                return ("production_acceptance_security_evidence_stale",)
            _, staging_blockers = _evaluate_staging_dynamic_manifest(
                record.readiness_snapshot_json.get("staging_dynamic_manifest"),
                now=checked_at,
            )
            if staging_blockers:
                return tuple(sorted(staging_blockers))
            attestation_snapshot, attestation_blockers = (
                _evaluate_staging_acceptance_attestation(
                    session,
                    tenant_id=tenant_id,
                    evidence_id=_optional_snapshot_string(
                        record.readiness_snapshot_json.get(
                            "staging_acceptance_attestation_evidence_id"
                        )
                    ),
                    manifest_value=record.readiness_snapshot_json.get(
                        "staging_dynamic_manifest"
                    ),
                )
            )
            if attestation_blockers:
                return tuple(sorted(attestation_blockers))
            if attestation_snapshot.get("staging_acceptance_attestation") != (
                record.readiness_snapshot_json.get("staging_acceptance_attestation")
            ):
                return ("staging_acceptance_attestation_snapshot_mismatch",)
            blocking_defect = session.scalar(
                select(ProductionDefectRecord.defect_id).where(
                    ProductionDefectRecord.tenant_id == tenant_id,
                    ProductionDefectRecord.status == "OPEN",
                    ProductionDefectRecord.severity.in_(("P0", "P1")),
                )
            )
            if blocking_defect is not None:
                return ("production_acceptance_defect_open",)
            signoff_roles = set(
                session.scalars(
                    select(ProductionAcceptanceSignoffRecord.signoff_role).where(
                        ProductionAcceptanceSignoffRecord.tenant_id == tenant_id,
                        ProductionAcceptanceSignoffRecord.acceptance_id == record.acceptance_id,
                        ProductionAcceptanceSignoffRecord.readiness_digest
                        == _digest(record.readiness_snapshot_json),
                    )
                )
            )
            if signoff_roles != {item.value for item in REQUIRED_SIGNOFFS}:
                return ("production_acceptance_signoffs_incomplete",)
            return ()

    def _operations_snapshot(self, now: datetime) -> OperationsSnapshot | None:
        try:
            return self._operations.snapshot(now=now)
        except Exception:
            return None

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )


def exercise_report_digest(
    exercise_id: str,
    verifier_version: str,
    results: tuple[ExerciseResultInput, ...],
) -> str:
    return _digest(
        {
            "exercise_id": exercise_id,
            "verifier_version": verifier_version,
            "results": [
                {
                    "scenario_id": item.scenario_id.value,
                    "observed_outcome": item.observed_outcome.value,
                    "attempt_count": item.attempt_count,
                    "unauthorized_read_count": item.unauthorized_read_count,
                    "unauthorized_side_effect_count": item.unauthorized_side_effect_count,
                    "sensitive_output_count": item.sensitive_output_count,
                    "evidence_ref": item.evidence_ref,
                    "artifact_digest": item.artifact_digest,
                }
                for item in sorted(results, key=lambda value: value.scenario_id.value)
            ],
        }
    )


def _evaluate_readiness(
    session: Session,
    *,
    tenant_id: str,
    plan: AcceptancePlan,
    operations: OperationsSnapshot | None,
    now: datetime,
) -> tuple[dict[str, Any], set[str]]:
    blockers: set[str] = set()
    exercise = session.scalar(
        select(SecurityExerciseRecord).where(
            SecurityExerciseRecord.tenant_id == tenant_id,
            SecurityExerciseRecord.exercise_id == plan.security_exercise_id,
        )
    )
    local_staging = plan.target_environment == "LOCAL_STAGING"
    required_scenarios = (
        LOCAL_STAGING_REQUIRED_SCENARIOS if local_staging else REQUIRED_SCENARIOS
    )
    exercise_valid = bool(
        exercise is not None
        and exercise.status == "PASSED"
        and exercise.completed_at is not None
        and exercise.target_environment == ("STAGING" if local_staging else "PRODUCTION")
        and required_scenarios.issubset(
            {AttackScenario(item) for item in exercise.scenario_ids_json}
        )
        and _stored_utc(exercise.completed_at) >= now - timedelta(days=31)
    )
    if not exercise_valid:
        blockers.add("security_exercise_incomplete_or_stale")
    open_defects = list(
        session.scalars(
            select(ProductionDefectRecord).where(
                ProductionDefectRecord.tenant_id == tenant_id,
                ProductionDefectRecord.status == "OPEN",
                ProductionDefectRecord.severity.in_(("P0", "P1")),
            )
        )
    )
    if open_defects:
        blockers.add("release_blocking_defects_open")
    release = session.scalar(
        select(ModelReleaseRecord).where(
            ModelReleaseRecord.tenant_id == tenant_id,
            ModelReleaseRecord.release_id == plan.model_release_id,
        )
    )
    deployment = session.scalar(
        select(ModelDeploymentRecord).where(
            ModelDeploymentRecord.tenant_id == tenant_id,
            ModelDeploymentRecord.release_id == plan.model_release_id,
        )
    )
    canary_decision = "ROLLBACK" if local_staging else "PASS"
    canary = session.scalar(
        select(ModelReleaseObservationRecord)
        .where(
            ModelReleaseObservationRecord.tenant_id == tenant_id,
            ModelReleaseObservationRecord.release_id == plan.model_release_id,
            ModelReleaseObservationRecord.stage == "CANARY_25",
            ModelReleaseObservationRecord.decision == canary_decision,
        )
        .order_by(ModelReleaseObservationRecord.sequence.desc())
    )
    release_valid = bool(
        release is not None
        and deployment is not None
        and (
            (
                local_staging
                and release.target_environment == "STAGING"
                and release.status == "ROLLED_BACK"
                and release.traffic_percent == 0.0
                and deployment.status == "READY"
                and deployment.current_stage == "ROLLED_BACK"
                and deployment.observed_traffic_percent == 0.0
            )
            or (
                not local_staging
                and release.target_environment == "PRODUCTION"
                and release.rollback_release_id is not None
                and deployment.current_stage in {"CANARY_25", "PRODUCTION"}
            )
        )
    )
    if not release_valid:
        blockers.add("production_release_or_rollback_target_invalid")
    if canary is None:
        blockers.add("canary_observation_missing")
    operations_ready = bool(
        local_staging
        or (operations is not None and operations.high_risk_release_allowed)
    )
    if not operations_ready:
        blockers.add("operations_or_recovery_gate_closed")
    if local_staging:
        staging_snapshot = {
            "staging_dynamic_manifest": None,
            "staging_dynamic_manifest_id": None,
            "staging_dynamic_manifest_status": "NOT_REQUIRED_FOR_LOCAL_STAGING_LAB",
            "staging_dynamic_manifest_generated_at": None,
            "staging_dynamic_manifest_git_commit": None,
            "staging_dynamic_manifest_valid": True,
        }
        attestation_snapshot = {
            "staging_acceptance_attestation": None,
            "staging_acceptance_attestation_evidence_id": None,
            "staging_acceptance_attestation_status": (
                "NOT_REQUIRED_FOR_LOCAL_STAGING_LAB"
            ),
            "staging_acceptance_attestation_certificate_identity": None,
            "staging_acceptance_attestation_certificate_oidc_issuer": None,
        }
    else:
        staging_snapshot, staging_blockers = _evaluate_staging_dynamic_manifest(
            plan.staging_dynamic_manifest,
            now=now,
        )
        blockers.update(staging_blockers)
        attestation_snapshot, attestation_blockers = (
            _evaluate_staging_acceptance_attestation(
                session,
                tenant_id=tenant_id,
                evidence_id=plan.staging_acceptance_attestation_evidence_id,
                manifest_value=plan.staging_dynamic_manifest,
            )
        )
        blockers.update(attestation_blockers)
    snapshot = {
        "policy_version": AssuranceService.POLICY_VERSION,
        "security_exercise_id": plan.security_exercise_id,
        "security_exercise_valid": exercise_valid,
        "open_p0_p1_defect_ids": sorted(item.defect_id for item in open_defects),
        "model_release_id": plan.model_release_id,
        "model_release_valid": release_valid,
        "canary_observation_id": canary.observation_id if canary is not None else None,
        "business_evidence_ref": plan.business_evidence_ref,
        "business_evidence_digest": plan.business_evidence_digest,
        "rollback_evidence_ref": plan.rollback_evidence_ref,
        "rollback_evidence_digest": plan.rollback_evidence_digest,
        **staging_snapshot,
        **attestation_snapshot,
        "operations_policy_version": operations.policy_version if operations else None,
        "operations_ready": operations_ready,
        "operations_blockers": (
            list(operations.release_gate_reasons)
            if operations is not None
            else ["operations_evidence_unavailable"]
        ),
    }
    return snapshot, blockers


def _open_or_refresh_defect(
    session: Session,
    *,
    tenant_id: str,
    exercise_id: str,
    result: dict[str, Any],
    now: datetime,
) -> None:
    scenario_id = str(result["scenario_id"])
    severity = _defect_severity(result)
    key = f"security:{scenario_id}"
    existing = session.scalar(
        select(ProductionDefectRecord).where(
            ProductionDefectRecord.tenant_id == tenant_id,
            ProductionDefectRecord.defect_key == key,
        )
    )
    if existing is None:
        session.add(
            ProductionDefectRecord(
                defect_id=f"production-defect-{uuid4().hex}",
                tenant_id=tenant_id,
                defect_key=key,
                exercise_id=exercise_id,
                scenario_id=scenario_id,
                severity=severity,
                release_blocking=severity in {"P0", "P1"},
                status="OPEN",
                description_code=f"security_exercise_{scenario_id}_failed",
                opened_at=now,
                closed_at=None,
                closed_by_subject_id=None,
                closure_evidence_ref=None,
                closure_evidence_digest=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        return
    existing.exercise_id = exercise_id
    existing.severity = severity
    existing.release_blocking = severity in {"P0", "P1"}
    existing.status = "OPEN"
    existing.opened_at = now
    existing.closed_at = None
    existing.closed_by_subject_id = None
    existing.closure_evidence_ref = None
    existing.closure_evidence_digest = None
    existing.version += 1
    existing.updated_at = now


def _defect_severity(result: dict[str, Any]) -> str:
    scenario = AttackScenario(str(result["scenario_id"]))
    if (
        scenario
        in {
            AttackScenario.PRIVILEGE_ESCALATION,
            AttackScenario.CROSS_TENANT_API,
            AttackScenario.CROSS_TENANT_OBJECT,
            AttackScenario.APPROVAL_BYPASS_T2,
            AttackScenario.T3_DIRECT_EXECUTION,
        }
        or int(result["unauthorized_read_count"]) > 0
        or int(result["unauthorized_side_effect_count"]) > 0
        or int(result["sensitive_output_count"]) > 0
    ):
        return "P0"
    return "P1"


def _validate_exercise_result(value: ExerciseResultInput) -> None:
    if any(
        number < 0 or number > 1_000_000
        for number in (
            value.attempt_count,
            value.unauthorized_read_count,
            value.unauthorized_side_effect_count,
            value.sensitive_output_count,
        )
    ):
        raise ValueError("security exercise count is outside the supported range")
    _validate_reference(value.evidence_ref, "security exercise evidence reference")
    _validate_digest(value.artifact_digest, "security exercise artifact digest")


def _validate_acceptance_plan(plan: AcceptancePlan) -> None:
    if plan.target_environment not in {"PRODUCTION", "LOCAL_STAGING"}:
        raise ValueError("acceptance target environment is invalid")
    for value, field in (
        (plan.release_scope, "production release scope"),
        (plan.security_exercise_id, "security exercise id"),
        (plan.model_release_id, "model release id"),
        (plan.business_evidence_ref, "business evidence reference"),
        (plan.rollback_evidence_ref, "rollback evidence reference"),
    ):
        _validate_reference(value, field)
    _validate_digest(plan.business_evidence_digest, "business evidence digest")
    _validate_digest(plan.rollback_evidence_digest, "rollback evidence digest")
    if plan.staging_acceptance_attestation_evidence_id is not None:
        _validate_reference(
            plan.staging_acceptance_attestation_evidence_id,
            "Staging acceptance attestation evidence id",
        )


def _require_signoff_role(identity: IdentityContext, role: SignoffRole) -> None:
    required = {
        SignoffRole.BUSINESS: Role.TENANT_ADMIN,
        SignoffRole.SECURITY: Role.SECURITY_AUDITOR,
        SignoffRole.PLATFORM: Role.PLATFORM_OPERATOR,
    }[role]
    if required not in identity.roles:
        raise AssuranceConflict("production_acceptance_signoff_role_mismatch")


def _exercise_or_hidden(
    session: Session,
    tenant_id: str,
    exercise_id: str,
) -> SecurityExerciseRecord:
    record = session.scalar(
        select(SecurityExerciseRecord)
        .where(
            SecurityExerciseRecord.tenant_id == tenant_id,
            SecurityExerciseRecord.exercise_id == exercise_id,
        )
        .with_for_update()
    )
    if record is None:
        raise AssuranceNotFound("security_exercise_not_found")
    return record


def _acceptance_or_hidden(
    session: Session,
    tenant_id: str,
    acceptance_id: str,
) -> ProductionAcceptanceRecord:
    record = session.scalar(
        select(ProductionAcceptanceRecord)
        .where(
            ProductionAcceptanceRecord.tenant_id == tenant_id,
            ProductionAcceptanceRecord.acceptance_id == acceptance_id,
        )
        .with_for_update()
    )
    if record is None:
        raise AssuranceNotFound("production_acceptance_not_found")
    return record


def _exercise_view(record: SecurityExerciseRecord) -> SecurityExerciseView:
    return SecurityExerciseView(
        exercise_id=record.exercise_id,
        name=record.name,
        target_environment=record.target_environment,
        status=record.status,
        scenario_ids=tuple(str(item) for item in record.scenario_ids_json),
        results=tuple(dict(item) for item in record.results_json),
        blocker_codes=tuple(str(item) for item in record.blocker_codes_json),
        report_digest=record.report_digest,
        verifier_version=record.verifier_version,
        requested_by_subject_id=record.requested_by_subject_id,
        completed_by_subject_id=record.completed_by_subject_id,
        started_at=_stored_utc(record.started_at),
        completed_at=_stored_utc(record.completed_at) if record.completed_at else None,
        version=record.version,
    )


def _defect_view(record: ProductionDefectRecord) -> ProductionDefectView:
    return ProductionDefectView(
        defect_id=record.defect_id,
        scenario_id=record.scenario_id,
        severity=record.severity,
        release_blocking=record.release_blocking,
        status=record.status,
        description_code=record.description_code,
        exercise_id=record.exercise_id,
        opened_at=_stored_utc(record.opened_at),
        closed_at=_stored_utc(record.closed_at) if record.closed_at else None,
        version=record.version,
    )


def _acceptance_view(
    session: Session,
    tenant_id: str,
    record: ProductionAcceptanceRecord,
) -> ProductionAcceptanceView:
    signoffs = list(
        session.scalars(
            select(ProductionAcceptanceSignoffRecord)
            .where(
                ProductionAcceptanceSignoffRecord.tenant_id == tenant_id,
                ProductionAcceptanceSignoffRecord.acceptance_id == record.acceptance_id,
            )
            .order_by(ProductionAcceptanceSignoffRecord.signed_at)
        )
    )
    snapshot = dict(record.readiness_snapshot_json)
    staging_generated_at = _optional_snapshot_datetime(
        snapshot.get("staging_dynamic_manifest_generated_at")
    )
    return ProductionAcceptanceView(
        acceptance_id=record.acceptance_id,
        release_scope=record.release_scope,
        target_environment=record.target_environment,
        security_exercise_id=record.security_exercise_id,
        model_release_id=record.model_release_id,
        staging_dynamic_manifest_id=_optional_snapshot_string(
            snapshot.get("staging_dynamic_manifest_id")
        ),
        staging_dynamic_manifest_status=_optional_snapshot_string(
            snapshot.get("staging_dynamic_manifest_status")
        ),
        staging_dynamic_manifest_generated_at=staging_generated_at,
        staging_acceptance_attestation_evidence_id=_optional_snapshot_string(
            snapshot.get("staging_acceptance_attestation_evidence_id")
        ),
        staging_acceptance_attestation_status=_optional_snapshot_string(
            snapshot.get("staging_acceptance_attestation_status")
        ),
        staging_acceptance_attestation_certificate_identity=_optional_snapshot_string(
            snapshot.get("staging_acceptance_attestation_certificate_identity")
        ),
        staging_acceptance_attestation_certificate_oidc_issuer=_optional_snapshot_string(
            snapshot.get("staging_acceptance_attestation_certificate_oidc_issuer")
        ),
        status=record.status,
        blocker_codes=tuple(str(item) for item in record.blocker_codes_json),
        readiness_snapshot=snapshot,
        readiness_digest=_digest(snapshot),
        created_by_subject_id=record.created_by_subject_id,
        evaluated_at=_stored_utc(record.evaluated_at),
        signed_off_at=_stored_utc(record.signed_off_at) if record.signed_off_at else None,
        signoffs=tuple(
            AcceptanceSignoffView(
                signoff_id=item.signoff_id,
                signoff_role=item.signoff_role,
                readiness_digest=item.readiness_digest,
                evidence_ref=item.evidence_ref,
                evidence_digest=item.evidence_digest,
                signed_by_subject_id=item.signed_by_subject_id,
                signed_at=_stored_utc(item.signed_at),
            )
            for item in signoffs
        ),
        version=record.version,
    )


def _acceptance_identity(plan: AcceptancePlan) -> dict[str, Any]:
    manifest = plan.staging_dynamic_manifest
    if isinstance(manifest, StagingDynamicAcceptanceManifest):
        manifest_value: dict[str, Any] | None = manifest.model_dump(mode="json")
    else:
        manifest_value = manifest
    return {
        "release_scope": plan.release_scope,
        "target_environment": plan.target_environment,
        "security_exercise_id": plan.security_exercise_id,
        "model_release_id": plan.model_release_id,
        "business_evidence_ref": plan.business_evidence_ref,
        "business_evidence_digest": plan.business_evidence_digest,
        "rollback_evidence_ref": plan.rollback_evidence_ref,
        "rollback_evidence_digest": plan.rollback_evidence_digest,
        "staging_dynamic_manifest": manifest_value,
        "staging_acceptance_attestation_evidence_id": (
            plan.staging_acceptance_attestation_evidence_id
        ),
    }


def _plan_from_record(record: ProductionAcceptanceRecord) -> AcceptancePlan:
    raw_manifest = record.readiness_snapshot_json.get("staging_dynamic_manifest")
    manifest = (
        raw_manifest
        if isinstance(raw_manifest, dict) or raw_manifest is None
        else {"invalid_snapshot_type": True}
    )
    return AcceptancePlan(
        release_scope=record.release_scope,
        target_environment=record.target_environment,
        security_exercise_id=record.security_exercise_id,
        model_release_id=record.model_release_id,
        business_evidence_ref=record.business_evidence_ref,
        business_evidence_digest=record.business_evidence_digest,
        rollback_evidence_ref=record.rollback_evidence_ref,
        rollback_evidence_digest=record.rollback_evidence_digest,
        staging_dynamic_manifest=manifest,
        staging_acceptance_attestation_evidence_id=_optional_snapshot_string(
            record.readiness_snapshot_json.get(
                "staging_acceptance_attestation_evidence_id"
            )
        ),
    )


def _evaluate_staging_acceptance_attestation(
    session: Session,
    *,
    tenant_id: str,
    evidence_id: str | None,
    manifest_value: object | None,
) -> tuple[dict[str, Any], set[str]]:
    projection: dict[str, Any] | None = None
    blockers: set[str] = set()
    if evidence_id is None:
        blockers.add("staging_acceptance_attestation_missing")
    else:
        record = session.scalar(
            select(StagingAcceptanceAttestationRecord).where(
                StagingAcceptanceAttestationRecord.tenant_id == tenant_id,
                StagingAcceptanceAttestationRecord.evidence_id == evidence_id,
            )
        )
        if record is None:
            blockers.add("staging_acceptance_attestation_unavailable")
        elif not attestation_record_is_verified(record):
            blockers.add("staging_acceptance_attestation_invalid")
        else:
            projection = {
                "evidence_id": record.evidence_id,
                "status": record.verification_status,
                "manifest_id": record.manifest_id,
                "manifest_digest": record.manifest_digest,
                "environment_id": record.environment_id,
                "git_commit": record.git_commit,
                "signature_bundle_digest": record.signature_bundle_digest,
                "certificate_identity": record.certificate_identity,
                "certificate_oidc_issuer": record.certificate_oidc_issuer,
                "verifier_version": record.verifier_version,
                "verification_hash": record.verification_hash,
                "verified_at": _stored_utc(record.verified_at).isoformat(),
            }
            try:
                manifest = (
                    manifest_value
                    if isinstance(manifest_value, StagingDynamicAcceptanceManifest)
                    else StagingDynamicAcceptanceManifest.model_validate(manifest_value)
                )
            except ValueError:
                manifest = None
            if manifest is None or (
                record.manifest_id != manifest.manifest_id
                or record.environment_id != manifest.environment_id
                or record.git_commit != manifest.git_commit
            ):
                blockers.add("staging_acceptance_attestation_binding_mismatch")
    return (
        {
            "staging_acceptance_attestation": projection,
            "staging_acceptance_attestation_evidence_id": evidence_id,
            "staging_acceptance_attestation_status": (
                projection["status"] if projection is not None else None
            ),
            "staging_acceptance_attestation_certificate_identity": (
                projection["certificate_identity"] if projection is not None else None
            ),
            "staging_acceptance_attestation_certificate_oidc_issuer": (
                projection["certificate_oidc_issuer"] if projection is not None else None
            ),
        },
        blockers,
    )


def _evaluate_staging_dynamic_manifest(
    value: object | None,
    *,
    now: datetime,
) -> tuple[dict[str, Any], set[str]]:
    blockers: set[str] = set()
    manifest: StagingDynamicAcceptanceManifest | None = None
    if value is None:
        blockers.add("staging_dynamic_acceptance_missing")
    else:
        try:
            manifest = (
                value
                if isinstance(value, StagingDynamicAcceptanceManifest)
                else StagingDynamicAcceptanceManifest.model_validate(value)
            )
        except ValueError:
            blockers.add("staging_dynamic_acceptance_invalid")
    if manifest is not None:
        try:
            validate_dynamic_acceptance_manifest_identity(manifest)
        except ValueError:
            blockers.add("staging_dynamic_acceptance_identity_mismatch")
        complete = (
            manifest.status
            == "STAGING_ACCEPTANCE_COMPLETE_PRODUCTION_REVIEW_REQUIRED"
            and not manifest.blocker_codes
            and manifest.summary
            == {"passed": 15, "failed": 0, "blocked": 0, "missing": 0}
            and all(item.status == "PASSED" for item in manifest.scenarios)
        )
        if not complete:
            blockers.add("staging_dynamic_acceptance_incomplete")
        generated_at = _as_utc(manifest.generated_at)
        if generated_at > now + _STAGING_MANIFEST_FUTURE_TOLERANCE:
            blockers.add("staging_dynamic_acceptance_from_future")
        if generated_at < now - _STAGING_MANIFEST_MAX_AGE:
            blockers.add("staging_dynamic_acceptance_stale")
    serialized = manifest.model_dump(mode="json") if manifest is not None else None
    return (
        {
            "staging_dynamic_manifest": serialized,
            "staging_dynamic_manifest_id": manifest.manifest_id if manifest else None,
            "staging_dynamic_manifest_status": manifest.status if manifest else None,
            "staging_dynamic_manifest_generated_at": (
                manifest.generated_at.isoformat() if manifest else None
            ),
            "staging_dynamic_manifest_git_commit": manifest.git_commit if manifest else None,
            "staging_dynamic_manifest_valid": not blockers,
        },
        blockers,
    )


def _optional_snapshot_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_snapshot_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _as_utc(datetime.fromisoformat(value))
    except ValueError:
        return None


def _validate_reference(value: str, field: str) -> None:
    if not _SAFE_REFERENCE.fullmatch(value):
        raise ValueError(f"{field} is invalid")


def _validate_digest(value: str, field: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} is invalid")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("assurance timestamp must include timezone")
    return value.astimezone(UTC)


def _digest(value: Any) -> str:
    return (
        "sha256:"
        + sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
