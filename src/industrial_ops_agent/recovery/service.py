"""Persisted RPO/RTO evidence and recovery-drill consistency gates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _stored_utc
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    RecoveryDrillRecord,
    RecoveryEvidenceRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext

_SAFE_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_ALLOWED_DETAIL_KEYS = frozenset(
    {
        "bytes",
        "image_count",
        "message_count",
        "model_count",
        "object_count",
        "replica_count",
        "verified_object_count",
        "wal_segment_count",
        "workflow_count",
    }
)


class RecoveryConflict(RuntimeError):
    pass


class RecoveryNotFound(RuntimeError):
    pass


class RecoveryComponent(StrEnum):
    POSTGRESQL = "postgresql"
    TEMPORAL = "temporal"
    MINIO = "minio"
    KAFKA = "kafka"
    REDIS = "redis"
    GITOPS_REGISTRY = "gitops_registry"
    MODEL_SERVICE = "model_service"


class EvidenceStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class RecoveryTarget:
    component: RecoveryComponent
    rpo_seconds: int | None
    rto_seconds: int
    evidence_max_age_seconds: int
    recovery_method: str
    verification_methods: frozenset[str]
    drill_verification_method: str
    required_checks: frozenset[str]


TARGETS = (
    RecoveryTarget(
        RecoveryComponent.POSTGRESQL,
        300,
        1800,
        900,
        "multi-zone database, WAL archive and point-in-time restore",
        frozenset({"WAL_ARCHIVE_VERIFY", "PITR_RESTORE"}),
        "PITR_RESTORE",
        frozenset({"DATABASE_RECORDS", "TENANT_RLS", "OUTBOX_INTEGRITY"}),
    ),
    RecoveryTarget(
        RecoveryComponent.TEMPORAL,
        300,
        1800,
        900,
        "restore persistence and replay workflow history",
        frozenset({"PERSISTENCE_CHECKPOINT", "WORKFLOW_REPLAY"}),
        "WORKFLOW_REPLAY",
        frozenset({"WORKFLOW_REPLAY", "PENDING_TASKS", "EXTERNAL_SIDE_EFFECTS"}),
    ),
    RecoveryTarget(
        RecoveryComponent.MINIO,
        900,
        3600,
        1800,
        "versioned replication and object checksum validation",
        frozenset({"REPLICATION_CHECKPOINT", "OBJECT_CHECKSUM"}),
        "OBJECT_CHECKSUM",
        frozenset({"OBJECT_HASHES", "VERSIONING", "MEDIA_REFERENCES"}),
    ),
    RecoveryTarget(
        RecoveryComponent.KAFKA,
        300,
        3600,
        900,
        "multi-replica recovery followed by controlled event replay",
        frozenset({"REPLICA_CHECKPOINT", "OFFSET_REPLAY"}),
        "OFFSET_REPLAY",
        frozenset({"OFFSETS", "OUTBOX_REPLAY", "INBOX_IDEMPOTENCY"}),
    ),
    RecoveryTarget(
        RecoveryComponent.REDIS,
        None,
        900,
        86_400,
        "discard and rebuild cache from authoritative stores",
        frozenset({"CACHE_REBUILD"}),
        "CACHE_REBUILD",
        frozenset({"CACHE_REBUILD", "DATABASE_FALLBACK"}),
    ),
    RecoveryTarget(
        RecoveryComponent.GITOPS_REGISTRY,
        0,
        7200,
        86_400,
        "rebuild from immutable Git, Registry and Terraform evidence",
        frozenset({"GIT_REBUILD"}),
        "GIT_REBUILD",
        frozenset({"GIT_REVISION", "IMAGE_DIGEST", "SIGNATURE_POLICY"}),
    ),
    RecoveryTarget(
        RecoveryComponent.MODEL_SERVICE,
        0,
        900,
        86_400,
        "restore stable replicas or roll back the approved manifest",
        frozenset({"MODEL_ROLLBACK"}),
        "MODEL_ROLLBACK",
        frozenset({"MODEL_MANIFEST", "INFERENCE_SMOKE", "ROUTE_ALIAS"}),
    ),
)
TARGET_BY_COMPONENT = {item.component: item for item in TARGETS}
QUARTERLY_SCOPE = frozenset(
    {
        RecoveryComponent.POSTGRESQL,
        RecoveryComponent.TEMPORAL,
        RecoveryComponent.MINIO,
        RecoveryComponent.KAFKA,
        RecoveryComponent.GITOPS_REGISTRY,
        RecoveryComponent.MODEL_SERVICE,
    }
)


@dataclass(frozen=True, slots=True)
class DegradationPolicy:
    failure: str
    system_behavior: str
    prohibited_behavior: str


DEGRADATION_POLICIES = (
    DegradationPolicy(
        "postgresql_unavailable",
        "reject business writes, fail readiness and alert operators",
        "approve work or mutate equipment state from cached data",
    ),
    DegradationPolicy(
        "kafka_unavailable",
        "commit valid online transactions with Outbox and replay after recovery",
        "roll back an already committed business transaction only because publish failed",
    ),
    DegradationPolicy(
        "temporal_unavailable",
        "hold new orchestration requests and stop high-risk actions",
        "execute workflow side effects directly inside the API",
    ),
    DegradationPolicy(
        "minio_unavailable",
        "block attachment confirmation while retaining queryable metadata",
        "treat unhashed temporary files as formal evidence",
    ),
    DegradationPolicy(
        "redis_unavailable",
        "read from authoritative stores, apply conservative limits and reauthenticate if needed",
        "interpret missing cache entries as missing work orders or approvals",
    ),
    DegradationPolicy(
        "model_service_unavailable",
        "use an approved stable model or retrieval-assisted human mode",
        "silently activate an unevaluated model",
    ),
    DegradationPolicy(
        "opa_unavailable",
        "fail closed for writes and high-risk tools",
        "allow every request to preserve availability",
    ),
    DegradationPolicy(
        "vault_unavailable",
        "let leased instances run only until expiry and keep new instances unready",
        "fall back to plaintext secrets in configuration",
    ),
    DegradationPolicy(
        "sse_disconnected",
        "resume with Last-Event-ID or fetch the authoritative snapshot",
        "restart the complete run and duplicate side effects",
    ),
    DegradationPolicy(
        "gpu_out_of_memory",
        "apply bounded queuing, reject over-budget work or use an approved fallback",
        "retry without bounds and create a retry storm",
    ),
)


@dataclass(frozen=True, slots=True)
class EvidenceRegistration:
    component: RecoveryComponent
    backup_id: str
    backup_type: str
    recovery_point_at: datetime
    captured_at: datetime
    verification_status: EvidenceStatus
    verification_method: str
    evidence_ref: str
    artifact_digest: str
    immutable: bool
    encrypted: bool
    details: dict[str, int | float | bool]


@dataclass(frozen=True, slots=True)
class RecoveryEvidenceView:
    evidence_id: str
    component: str
    backup_id: str
    backup_type: str
    recovery_point_at: datetime
    captured_at: datetime
    verification_status: str
    verification_method: str
    evidence_ref: str
    artifact_digest: str
    immutable: bool
    encrypted: bool
    details: dict[str, Any]
    verified_by_subject_id: str
    verified_at: datetime
    version: int


@dataclass(frozen=True, slots=True)
class ConsistencyCheck:
    check_id: str
    status: str
    evidence_ref: str


@dataclass(frozen=True, slots=True)
class ComponentRecoveryResult:
    component: RecoveryComponent
    recovered_to_at: datetime
    service_restored_at: datetime
    evidence_ids: tuple[str, ...]
    checks: tuple[ConsistencyCheck, ...]


@dataclass(frozen=True, slots=True)
class RecoveryDrillView:
    drill_id: str
    scenario: str
    scope: tuple[str, ...]
    status: str
    outage_detected_at: datetime
    started_at: datetime
    completed_at: datetime | None
    results: tuple[dict[str, Any], ...]
    blocker_codes: tuple[str, ...]
    requested_by_subject_id: str
    completed_by_subject_id: str | None
    objective_met: bool
    version: int


@dataclass(frozen=True, slots=True)
class ComponentRecoveryPosture:
    component: str
    rpo_seconds: int | None
    rto_seconds: int
    recovery_method: str
    evidence_status: str
    latest_evidence: RecoveryEvidenceView | None


@dataclass(frozen=True, slots=True)
class RecoveryOverview:
    policy_version: str
    generated_at: datetime
    release_gate_enforced: bool
    release_allowed: bool
    release_gate_reasons: tuple[str, ...]
    monthly_postgres_restore_due: bool
    quarterly_cross_component_drill_due: bool
    components: tuple[ComponentRecoveryPosture, ...]
    recent_drills: tuple[RecoveryDrillView, ...]
    degradation_policies: tuple[DegradationPolicy, ...]


class RecoveryService:
    POLICY_VERSION = "recovery-governance/v1"

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        *,
        enforce_release_gate: bool,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._enforce_release_gate = enforce_release_gate

    def overview(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        now: datetime | None = None,
    ) -> RecoveryOverview:
        self._require(identity, Action.READ_RECOVERY, "recovery-overview", request_id)
        return self._overview(identity.tenant_context, now=now)

    def register_evidence(
        self,
        identity: IdentityContext,
        registration: EvidenceRegistration,
        *,
        request_id: str,
        now: datetime | None = None,
    ) -> RecoveryEvidenceView:
        self._require(identity, Action.MANAGE_RECOVERY, registration.backup_id, request_id)
        _validate_registration(registration)
        verified_at = _as_utc(now or datetime.now(UTC))
        recovery_point_at = _as_utc(registration.recovery_point_at)
        captured_at = _as_utc(registration.captured_at)
        if recovery_point_at > captured_at or captured_at > verified_at + timedelta(minutes=5):
            raise ValueError("recovery evidence timestamps are inconsistent")
        evidence_id = (
            "recovery-evidence-"
            + sha256(
                f"{identity.tenant_id}\0{registration.component.value}\0{registration.backup_id}".encode()
            ).hexdigest()[:32]
        )
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(RecoveryEvidenceRecord).where(
                    RecoveryEvidenceRecord.tenant_id == identity.tenant_id,
                    RecoveryEvidenceRecord.component == registration.component.value,
                    RecoveryEvidenceRecord.backup_id == registration.backup_id,
                )
            )
            if existing is not None:
                if _evidence_identity(existing) != _registration_identity(registration):
                    raise RecoveryConflict("recovery_evidence_idempotency_conflict")
                return _evidence_view(existing)
            record = RecoveryEvidenceRecord(
                evidence_id=evidence_id,
                tenant_id=identity.tenant_id,
                component=registration.component.value,
                backup_id=registration.backup_id,
                backup_type=registration.backup_type,
                recovery_point_at=recovery_point_at,
                captured_at=captured_at,
                verification_status=registration.verification_status.value,
                verification_method=registration.verification_method,
                evidence_ref=registration.evidence_ref,
                artifact_digest=registration.artifact_digest,
                immutable=registration.immutable,
                encrypted=registration.encrypted,
                details_json=dict(registration.details),
                verified_by_subject_id=identity.subject_id,
                verified_at=verified_at,
                version=1,
                created_at=verified_at,
                updated_at=verified_at,
            )
            session.add(record)
            session.flush()
            return _evidence_view(record)

    def start_drill(
        self,
        identity: IdentityContext,
        *,
        idempotency_key: str,
        scenario: str,
        scope: tuple[RecoveryComponent, ...],
        outage_detected_at: datetime,
        request_id: str,
        now: datetime | None = None,
    ) -> RecoveryDrillView:
        self._require(identity, Action.MANAGE_RECOVERY, "recovery-drill", request_id)
        if not _SAFE_REFERENCE.fullmatch(idempotency_key):
            raise ValueError("recovery drill idempotency key is invalid")
        if not _SAFE_REFERENCE.fullmatch(scenario):
            raise ValueError("recovery drill scenario is invalid")
        normalized_scope = tuple(sorted(set(scope), key=lambda item: item.value))
        if not normalized_scope:
            raise ValueError("recovery drill scope is empty")
        started_at = _as_utc(now or datetime.now(UTC))
        outage = _as_utc(outage_detected_at)
        if outage > started_at + timedelta(minutes=5):
            raise ValueError("recovery drill outage timestamp is in the future")
        request_hash = _digest(
            {
                "scenario": scenario,
                "scope": [item.value for item in normalized_scope],
                "outage_detected_at": outage.isoformat(),
            }
        )
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(RecoveryDrillRecord).where(
                    RecoveryDrillRecord.tenant_id == identity.tenant_id,
                    RecoveryDrillRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise RecoveryConflict("recovery_drill_idempotency_conflict")
                return _drill_view(existing)
            record = RecoveryDrillRecord(
                drill_id=f"recovery-drill-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                scenario=scenario,
                scope_json=[item.value for item in normalized_scope],
                status="RUNNING",
                outage_detected_at=outage,
                started_at=started_at,
                completed_at=None,
                results_json=[],
                blocker_codes_json=[],
                requested_by_subject_id=identity.subject_id,
                completed_by_subject_id=None,
                objective_met=False,
                version=1,
                created_at=started_at,
                updated_at=started_at,
            )
            session.add(record)
            session.flush()
            return _drill_view(record)

    def complete_drill(
        self,
        identity: IdentityContext,
        drill_id: str,
        *,
        expected_version: int,
        component_results: tuple[ComponentRecoveryResult, ...],
        request_id: str,
        now: datetime | None = None,
    ) -> RecoveryDrillView:
        self._require(identity, Action.MANAGE_RECOVERY, drill_id, request_id)
        completed_at = _as_utc(now or datetime.now(UTC))
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(RecoveryDrillRecord)
                .where(
                    RecoveryDrillRecord.tenant_id == identity.tenant_id,
                    RecoveryDrillRecord.drill_id == drill_id,
                )
                .with_for_update()
            )
            if record is None:
                raise RecoveryNotFound("recovery_drill_not_found")
            if record.version != expected_version:
                raise RecoveryConflict("recovery_drill_version_conflict")
            if record.status != "RUNNING":
                raise RecoveryConflict("recovery_drill_not_running")
            if completed_at < _stored_utc(record.started_at):
                raise ValueError("recovery drill completion precedes its start")
            expected_scope = {RecoveryComponent(value) for value in record.scope_json}
            supplied_scope = {item.component for item in component_results}
            if len(component_results) != len(supplied_scope) or supplied_scope != expected_scope:
                raise ValueError("recovery drill results must cover the exact drill scope")
            evidence_ids = {
                evidence_id for item in component_results for evidence_id in item.evidence_ids
            }
            evidence = {
                item.evidence_id: item
                for item in session.scalars(
                    select(RecoveryEvidenceRecord).where(
                        RecoveryEvidenceRecord.tenant_id == identity.tenant_id,
                        RecoveryEvidenceRecord.evidence_id.in_(evidence_ids),
                    )
                )
            }
            results: list[dict[str, Any]] = []
            blockers: set[str] = set()
            for submission in component_results:
                result, component_blockers = _evaluate_component_result(
                    submission,
                    outage_detected_at=_stored_utc(record.outage_detected_at),
                    completed_at=completed_at,
                    evidence=evidence,
                )
                results.append(result)
                blockers.update(component_blockers)
            record.status = "PASSED" if not blockers else "FAILED"
            record.completed_at = completed_at
            record.results_json = sorted(results, key=lambda item: str(item["component"]))
            record.blocker_codes_json = sorted(blockers)
            record.completed_by_subject_id = identity.subject_id
            record.objective_met = not blockers
            record.version += 1
            record.updated_at = completed_at
            session.flush()
            return _drill_view(record)

    def release_gate_reasons(
        self,
        tenant_id: str,
        *,
        now: datetime | None = None,
    ) -> tuple[str, ...]:
        if not self._enforce_release_gate:
            return ()
        overview = self._overview(
            TenantContext(tenant_id=tenant_id, subject_id="recovery-release-gate"),
            now=now,
        )
        return overview.release_gate_reasons

    def _overview(
        self,
        context: TenantContext,
        *,
        now: datetime | None,
    ) -> RecoveryOverview:
        generated_at = _as_utc(now or datetime.now(UTC))
        with self._database.transaction(context) as session:
            evidence_records = list(
                session.scalars(
                    select(RecoveryEvidenceRecord)
                    .where(RecoveryEvidenceRecord.tenant_id == context.tenant_id)
                    .order_by(RecoveryEvidenceRecord.verified_at.desc())
                )
            )
            drills = list(
                session.scalars(
                    select(RecoveryDrillRecord)
                    .where(RecoveryDrillRecord.tenant_id == context.tenant_id)
                    .order_by(RecoveryDrillRecord.started_at.desc())
                    .limit(20)
                )
            )
        latest_by_component: dict[str, RecoveryEvidenceRecord] = {}
        for record in evidence_records:
            latest_by_component.setdefault(record.component, record)
        postures = tuple(
            _posture(target, latest_by_component.get(target.component.value), generated_at)
            for target in TARGETS
        )
        passed = [item for item in drills if item.status == "PASSED" and item.objective_met]
        monthly_postgres_due = not any(
            RecoveryComponent.POSTGRESQL.value in item.scope_json
            and item.completed_at is not None
            and _stored_utc(item.completed_at) >= generated_at - timedelta(days=31)
            for item in passed
        )
        quarterly_due = not any(
            QUARTERLY_SCOPE.issubset({RecoveryComponent(value) for value in item.scope_json})
            and item.completed_at is not None
            and _stored_utc(item.completed_at) >= generated_at - timedelta(days=92)
            for item in passed
        )
        reasons: list[str] = []
        if any(
            item.component != RecoveryComponent.REDIS.value and item.evidence_status != "COMPLIANT"
            for item in postures
        ):
            reasons.append("recovery_evidence_noncompliant")
        if monthly_postgres_due:
            reasons.append("postgres_restore_drill_overdue")
        if quarterly_due:
            reasons.append("cross_component_recovery_drill_overdue")
        effective_reasons = tuple(reasons) if self._enforce_release_gate else ()
        return RecoveryOverview(
            policy_version=self.POLICY_VERSION,
            generated_at=generated_at,
            release_gate_enforced=self._enforce_release_gate,
            release_allowed=not effective_reasons,
            release_gate_reasons=effective_reasons,
            monthly_postgres_restore_due=monthly_postgres_due,
            quarterly_cross_component_drill_due=quarterly_due,
            components=postures,
            recent_drills=tuple(_drill_view(item) for item in drills),
            degradation_policies=DEGRADATION_POLICIES,
        )

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


def _evaluate_component_result(
    submission: ComponentRecoveryResult,
    *,
    outage_detected_at: datetime,
    completed_at: datetime,
    evidence: dict[str, RecoveryEvidenceRecord],
) -> tuple[dict[str, Any], set[str]]:
    target = TARGET_BY_COMPONENT[submission.component]
    recovered_to = _as_utc(submission.recovered_to_at)
    restored_at = _as_utc(submission.service_restored_at)
    if recovered_to > outage_detected_at or restored_at < outage_detected_at:
        raise ValueError("recovery drill component timestamps are inconsistent")
    if restored_at > completed_at + timedelta(minutes=5):
        raise ValueError("recovery drill restoration timestamp is in the future")
    supplied_checks = {item.check_id: item for item in submission.checks}
    if len(supplied_checks) != len(submission.checks):
        raise ValueError("recovery drill contains duplicate consistency checks")
    if not set(supplied_checks).issubset(target.required_checks):
        raise ValueError("recovery drill contains an unsupported consistency check")
    if any(
        item.status not in {"PASSED", "FAILED"}
        or not _SAFE_REFERENCE.fullmatch(item.check_id)
        or not _SAFE_REFERENCE.fullmatch(item.evidence_ref)
        for item in submission.checks
    ):
        raise ValueError("recovery drill contains an invalid consistency check")
    blockers: set[str] = set()
    for check_id in target.required_checks:
        check = supplied_checks.get(check_id)
        if check is None or check.status != "PASSED":
            blockers.add(f"{submission.component.value}:{check_id.lower()}_failed")
    component_evidence = [evidence.get(item) for item in submission.evidence_ids]
    if not component_evidence or any(item is None for item in component_evidence):
        blockers.add(f"{submission.component.value}:evidence_missing")
    else:
        if not any(
            item is not None
            and item.component == submission.component.value
            and item.verification_method == target.drill_verification_method
            for item in component_evidence
        ):
            blockers.add(f"{submission.component.value}:restore_evidence_missing")
        for item in component_evidence:
            assert item is not None
            if (
                item.component != submission.component.value
                or item.verification_status != EvidenceStatus.PASSED.value
                or not item.immutable
                or not item.encrypted
            ):
                blockers.add(f"{submission.component.value}:evidence_noncompliant")
    actual_rpo = max(0, int((outage_detected_at - recovered_to).total_seconds()))
    actual_rto = max(0, int((restored_at - outage_detected_at).total_seconds()))
    if target.rpo_seconds is not None and actual_rpo > target.rpo_seconds:
        blockers.add(f"{submission.component.value}:rpo_exceeded")
    if actual_rto > target.rto_seconds:
        blockers.add(f"{submission.component.value}:rto_exceeded")
    return (
        {
            "component": submission.component.value,
            "recovered_to_at": recovered_to.isoformat(),
            "service_restored_at": restored_at.isoformat(),
            "actual_rpo_seconds": actual_rpo,
            "actual_rto_seconds": actual_rto,
            "objective_met": not blockers,
            "evidence_ids": sorted(submission.evidence_ids),
            "checks": [
                {
                    "check_id": item.check_id,
                    "status": item.status,
                    "evidence_ref": item.evidence_ref,
                }
                for item in sorted(submission.checks, key=lambda value: value.check_id)
            ],
        },
        blockers,
    )


def _posture(
    target: RecoveryTarget,
    record: RecoveryEvidenceRecord | None,
    now: datetime,
) -> ComponentRecoveryPosture:
    status = "MISSING"
    view = None
    if record is not None:
        view = _evidence_view(record)
        if (
            record.verification_status != EvidenceStatus.PASSED.value
            or not record.immutable
            or not record.encrypted
        ):
            status = "FAILED"
        elif now - _stored_utc(record.verified_at) > timedelta(
            seconds=target.evidence_max_age_seconds
        ) or now - _stored_utc(record.recovery_point_at) > timedelta(
            seconds=target.evidence_max_age_seconds
        ):
            status = "STALE"
        else:
            status = "COMPLIANT"
    return ComponentRecoveryPosture(
        component=target.component.value,
        rpo_seconds=target.rpo_seconds,
        rto_seconds=target.rto_seconds,
        recovery_method=target.recovery_method,
        evidence_status=status,
        latest_evidence=view,
    )


def _validate_registration(registration: EvidenceRegistration) -> None:
    target = TARGET_BY_COMPONENT[registration.component]
    if not _SAFE_REFERENCE.fullmatch(registration.backup_id):
        raise ValueError("recovery backup id is invalid")
    if not _SAFE_REFERENCE.fullmatch(registration.backup_type):
        raise ValueError("recovery backup type is invalid")
    if registration.verification_method not in target.verification_methods:
        raise ValueError("recovery verification method is invalid for component")
    if not _SAFE_REFERENCE.fullmatch(registration.evidence_ref):
        raise ValueError("recovery evidence reference is invalid")
    if not _SHA256.fullmatch(registration.artifact_digest):
        raise ValueError("recovery artifact digest is invalid")
    if not set(registration.details).issubset(_ALLOWED_DETAIL_KEYS):
        raise ValueError("recovery evidence details contain an unsupported field")
    if any(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (value < 0 or value > 10**15)
        for value in registration.details.values()
    ):
        raise ValueError("recovery evidence detail is outside the supported range")


def _registration_identity(registration: EvidenceRegistration) -> tuple[Any, ...]:
    return (
        registration.backup_type,
        _as_utc(registration.recovery_point_at),
        _as_utc(registration.captured_at),
        registration.verification_status.value,
        registration.verification_method,
        registration.evidence_ref,
        registration.artifact_digest,
        registration.immutable,
        registration.encrypted,
        registration.details,
    )


def _evidence_identity(record: RecoveryEvidenceRecord) -> tuple[Any, ...]:
    return (
        record.backup_type,
        _stored_utc(record.recovery_point_at),
        _stored_utc(record.captured_at),
        record.verification_status,
        record.verification_method,
        record.evidence_ref,
        record.artifact_digest,
        record.immutable,
        record.encrypted,
        record.details_json,
    )


def _evidence_view(record: RecoveryEvidenceRecord) -> RecoveryEvidenceView:
    return RecoveryEvidenceView(
        evidence_id=record.evidence_id,
        component=record.component,
        backup_id=record.backup_id,
        backup_type=record.backup_type,
        recovery_point_at=_stored_utc(record.recovery_point_at),
        captured_at=_stored_utc(record.captured_at),
        verification_status=record.verification_status,
        verification_method=record.verification_method,
        evidence_ref=record.evidence_ref,
        artifact_digest=record.artifact_digest,
        immutable=record.immutable,
        encrypted=record.encrypted,
        details=dict(record.details_json),
        verified_by_subject_id=record.verified_by_subject_id,
        verified_at=_stored_utc(record.verified_at),
        version=record.version,
    )


def _drill_view(record: RecoveryDrillRecord) -> RecoveryDrillView:
    return RecoveryDrillView(
        drill_id=record.drill_id,
        scenario=record.scenario,
        scope=tuple(str(item) for item in record.scope_json),
        status=record.status,
        outage_detected_at=_stored_utc(record.outage_detected_at),
        started_at=_stored_utc(record.started_at),
        completed_at=_stored_utc(record.completed_at) if record.completed_at else None,
        results=tuple(dict(item) for item in record.results_json),
        blocker_codes=tuple(str(item) for item in record.blocker_codes_json),
        requested_by_subject_id=record.requested_by_subject_id,
        completed_by_subject_id=record.completed_by_subject_id,
        objective_met=record.objective_met,
        version=record.version,
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("recovery timestamp must include timezone")
    return value.astimezone(UTC)


def _digest(value: dict[str, Any]) -> str:
    return (
        "sha256:"
        + sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
