"""Administrator-confirmed reader deployment and auditable tenant cutover."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc
from industrial_ops_agent.domain.json import strict_document_digest
from industrial_ops_agent.maintenance_planning.review_isolation import (
    READER_COVERAGE_DIGEST,
    REVIEW_POLICY_VERSION,
    ReviewIsolationConflict,
    lock_subject,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    MaintenanceReviewPolicyCommandRecord,
    MaintenanceReviewPolicyRecord,
    MaintenanceReviewRolloutRecord,
)

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
MAX_RECEIPT_BYTES = 262_144
MAX_RECEIPT_AGE = timedelta(hours=1)
ACTIVATION_ASSURANCE = "PROJECT_ADMIN_CONFIRMED"


class ReaderInstance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    instance_id: str = Field(min_length=1, max_length=128)
    workload: Name
    image_digest: Digest
    git_revision: str = Field(max_length=128)
    reader_coverage_digest: str = Field(max_length=128)
    ready: bool


class ReaderScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tenant_id: Name
    environment_id: str = Field(min_length=1, max_length=128)
    runtime: Literal["compose", "kubernetes"]
    runtime_identity: str = Field(min_length=1, max_length=255)
    scope: Name
    workloads: dict[Name, Annotated[int, Field(ge=1, le=100)]] = Field(min_length=1, max_length=32)


class ReaderInventory(ReaderScope):
    observed_at: AwareDatetime
    instances: list[ReaderInstance] = Field(min_length=1, max_length=3200)


class ReaderRolloutReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["maintenance-review-rollout-v2"]
    policy_version: Literal["maintenance-planning-blind-ab-v2"]
    # Existing review protocol uses a bare canonical SHA-256, unlike blob/image digests.
    reader_coverage_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    before: ReaderInventory
    after: ReaderInventory

    def require_complete(self, tenant_id: str, now: datetime) -> None:
        # Deployment scope is explicitly attested by the administrator, not a signer.
        scope = self.after.model_dump(exclude={"observed_at", "instances"})
        for inventory in (self.before, self.after):
            if (
                inventory.tenant_id != tenant_id
                or inventory.model_dump(exclude={"observed_at", "instances"}) != scope
                or len({i.instance_id for i in inventory.instances}) != len(inventory.instances)
                or any(i.workload not in self.after.workloads for i in inventory.instances)
            ):
                raise ReviewIsolationConflict("rollout_scope_mismatch")
        if (
            self.reader_coverage_digest != READER_COVERAGE_DIGEST
            or not self.before.observed_at < self.after.observed_at <= now
            or now - self.after.observed_at > MAX_RECEIPT_AGE
        ):
            raise ReviewIsolationConflict("rollout_stale_or_wrong_version")
        if Counter(i.workload for i in self.after.instances) != self.after.workloads:
            raise ReviewIsolationConflict("reader_inventory_incomplete")
        images = {i.workload: i.image_digest for i in self.after.instances}
        current_ids = {i.instance_id for i in self.after.instances}
        for old in self.before.instances:
            old_code = (
                old.git_revision != self.git_revision
                or old.reader_coverage_digest != self.reader_coverage_digest
                or old.image_digest != images[old.workload]
            )
            if old_code and old.instance_id in current_ids:
                raise ReviewIsolationConflict("old_readers_not_drained")
        if any(
            not i.ready
            or i.git_revision != self.git_revision
            or i.reader_coverage_digest != self.reader_coverage_digest
            or i.image_digest != images[i.workload]
            for i in self.after.instances
        ):
            raise ReviewIsolationConflict("readers_not_ready")


def _digest(raw: bytes) -> str:
    return f"sha256:{sha256(raw).hexdigest()}"


def _verification_document(record: MaintenanceReviewRolloutRecord) -> dict[str, Any]:
    return {
        "tenant_id": record.tenant_id,
        "receipt_id": record.receipt_id,
        "receipt_digest": record.receipt_digest,
        "proof": record.proof_json,
        "registered_by": record.registered_by,
        "registered_at": as_utc(record.registered_at).isoformat(),
        "request_id": record.request_id,
    }


def _receipt(
    session: Session,
    tenant_id: str,
    receipt_id: str,
    at: datetime,
) -> MaintenanceReviewRolloutRecord:
    record = session.scalar(
        select(MaintenanceReviewRolloutRecord).where(
            MaintenanceReviewRolloutRecord.tenant_id == tenant_id,
            MaintenanceReviewRolloutRecord.receipt_id == receipt_id,
        )
    )
    if record is None:
        raise ReviewIsolationConflict("rollout_receipt_missing")
    proof = record.proof_json
    if (
        record.verification_digest != strict_document_digest(_verification_document(record))
        or record.receipt_digest != _digest(record.receipt_text.encode("utf-8"))
        or proof.get("assurance") != ACTIVATION_ASSURANCE
        or proof.get("deployment_confirmed") is not True
        or proof.get("deployment_digest") != record.receipt_digest
        or proof.get("confirmed_by") != record.registered_by
        or not isinstance(proof.get("reason"), str)
        or not 8 <= len(proof["reason"].strip()) <= 1000
        or record.signature_bundle != ""
    ):
        raise ReviewIsolationConflict("rollout_receipt_integrity")
    try:
        document = ReaderRolloutReceipt.model_validate_json(record.receipt_text)
    except ValidationError as exc:
        raise ReviewIsolationConflict("rollout_receipt_invalid") from exc
    document.require_complete(tenant_id, at)
    return record


def require_policy_activation(session: Session, policy: MaintenanceReviewPolicyRecord) -> None:
    """Recheck immutable binding, not freshness against every later claim's wall clock."""
    if session.get_bind().dialect.name != "postgresql":
        raise ReviewIsolationConflict("postgresql_required")
    if (
        not policy.enabled
        or policy.cutover_at is None
        or not policy.receipt_id
        or policy.policy_version != REVIEW_POLICY_VERSION
        or policy.reader_coverage_digest != READER_COVERAGE_DIGEST
    ):
        raise ReviewIsolationConflict("policy_inactive")
    record = _receipt(
        session,
        policy.tenant_id,
        policy.receipt_id,
        as_utc(policy.cutover_at),
    )
    if record.receipt_digest != policy.receipt_digest:
        raise ReviewIsolationConflict("policy_receipt_changed")
    command = session.scalar(
        select(MaintenanceReviewPolicyCommandRecord).where(
            MaintenanceReviewPolicyCommandRecord.tenant_id == policy.tenant_id,
            MaintenanceReviewPolicyCommandRecord.subject_id == policy.activated_by,
            MaintenanceReviewPolicyCommandRecord.idempotency_key == policy.idempotency_key,
        )
    )
    if (
        command is None
        or command.request_digest != policy.request_digest
        or {k: v for k, v in command.result_json.items() if k != "reason"} != _policy_view(policy)
        or as_utc(record.registered_at) > as_utc(policy.cutover_at)
    ):
        raise ReviewIsolationConflict("policy_activation_record_missing")


class ReviewPolicyService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database, self._authorizer = database, authorizer

    def _require(self, identity: IdentityContext, action: Action, request_id: str) -> None:
        # This administrative operation is not a model approval or emergency role grant.
        if Role.TENANT_ADMIN not in identity.roles:
            raise AuthorizationDenied("role_denied")
        self._authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id="maintenance-review-policy"),
            request_id=request_id,
        )

    def register(
        self,
        identity: IdentityContext,
        receipt_text: str,
        *,
        deployment_confirmed: bool,
        reason: str,
        request_id: str,
    ) -> dict[str, str]:
        self._require(identity, Action.REGISTER_MAINTENANCE_REVIEW_ROLLOUT, request_id)
        if deployment_confirmed is not True or not 8 <= len(reason.strip()) <= 1000:
            raise ReviewIsolationConflict("administrator_confirmation_required")
        raw = receipt_text.encode("utf-8")
        if not 0 < len(raw) <= MAX_RECEIPT_BYTES:
            raise ReviewIsolationConflict("rollout_size_invalid")
        try:
            receipt = ReaderRolloutReceipt.model_validate_json(raw)
        except ValidationError as exc:
            raise ReviewIsolationConflict("rollout_receipt_invalid") from exc
        receipt.require_complete(identity.tenant_id, datetime.now(UTC))
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity, lineage_write=True)
            now = datetime.now(UTC)
            receipt.require_complete(identity.tenant_id, now)
            digest = _digest(raw)
            existing = session.scalar(
                select(MaintenanceReviewRolloutRecord).where(
                    MaintenanceReviewRolloutRecord.tenant_id == identity.tenant_id,
                    MaintenanceReviewRolloutRecord.receipt_digest == digest,
                )
            )
            if existing is not None:
                _receipt(session, identity.tenant_id, existing.receipt_id, now)
                return {"receipt_id": existing.receipt_id, "receipt_digest": digest}
            record = MaintenanceReviewRolloutRecord(
                receipt_id=f"review-rollout-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                receipt_digest=digest,
                receipt_text=receipt_text,
                # Preserve the historical column/rows; new activations have no signature.
                signature_bundle="",
                proof_json={
                    "assurance": ACTIVATION_ASSURANCE,
                    "deployment_confirmed": True,
                    "deployment_digest": digest,
                    "confirmed_by": identity.subject_id,
                    "reason": reason.strip(),
                },
                registered_by=identity.subject_id,
                registered_at=now,
                request_id=request_id,
            )
            record.verification_digest = strict_document_digest(_verification_document(record))
            session.add(record)
            session.flush()
            return {"receipt_id": record.receipt_id, "receipt_digest": digest}

    def status(self, identity: IdentityContext, *, request_id: str) -> dict[str, Any]:
        self._require(identity, Action.READ_TENANT_ADMINISTRATION, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            policy = session.get(MaintenanceReviewPolicyRecord, identity.tenant_id)
            reasons: list[str] = []
            if session.get_bind().dialect.name != "postgresql":
                reasons.append("maintenance_review_postgresql_required")
            if policy is not None and policy.enabled:
                try:
                    require_policy_activation(session, policy)
                except ReviewIsolationConflict as exc:
                    reasons.append(exc.reason)
            receipts = []
            environment_id = None
            for record in session.scalars(
                select(MaintenanceReviewRolloutRecord)
                .where(MaintenanceReviewRolloutRecord.tenant_id == identity.tenant_id)
                .order_by(
                    MaintenanceReviewRolloutRecord.registered_at.desc(),
                    MaintenanceReviewRolloutRecord.receipt_id,
                )
                .limit(20)
            ):
                blockers = list(reasons)
                try:
                    at = (
                        as_utc(policy.cutover_at)
                        if policy
                        and policy.enabled
                        and policy.receipt_id == record.receipt_id
                        and policy.cutover_at
                        else datetime.now(UTC)
                    )
                    _receipt(session, identity.tenant_id, record.receipt_id, at)
                    if environment_id is None:
                        environment_id = ReaderRolloutReceipt.model_validate_json(
                            record.receipt_text
                        ).after.environment_id
                except ReviewIsolationConflict as exc:
                    blockers.append(exc.reason)
                receipts.append(
                    {
                        "receipt_id": record.receipt_id,
                        "receipt_digest": record.receipt_digest,
                        "registered_at": record.registered_at,
                        "blockers": blockers,
                    }
                )
            if not receipts:
                reasons.append("maintenance_review_rollout_receipt_missing")
            return {
                **_policy_view(policy),
                "blockers": reasons,
                "receipts": receipts,
                "environment_id": environment_id,
                "assurance": ACTIVATION_ASSURANCE,
            }

    def change(
        self,
        identity: IdentityContext,
        *,
        enabled: bool,
        receipt_id: str | None,
        expected_version: int,
        idempotency_key: str,
        reason: str,
        request_id: str,
    ) -> dict[str, Any]:
        self._require(identity, Action.ACTIVATE_MAINTENANCE_REVIEW_POLICY, request_id)
        digest = strict_document_digest(
            {
                "enabled": enabled,
                "receipt_id": receipt_id,
                "expected_version": expected_version,
                "reason": reason,
                "policy_version": REVIEW_POLICY_VERSION,
            }
        )
        with self._database.transaction(identity.tenant_context) as session:
            if session.get_bind().dialect.name != "postgresql":
                raise ReviewIsolationConflict("postgresql_required")
            # Exclusive tenant lineage lock serializes even the first policy and different admins.
            lock_subject(session, identity, lineage_write=True)
            replay = session.scalar(
                select(MaintenanceReviewPolicyCommandRecord).where(
                    MaintenanceReviewPolicyCommandRecord.tenant_id == identity.tenant_id,
                    MaintenanceReviewPolicyCommandRecord.subject_id == identity.subject_id,
                    MaintenanceReviewPolicyCommandRecord.idempotency_key == idempotency_key,
                )
            )
            if replay is not None:
                if replay.request_digest != digest:
                    raise ReviewIsolationConflict("policy_idempotency_conflict")
                return replay.result_json
            policy = session.get(
                MaintenanceReviewPolicyRecord, identity.tenant_id, with_for_update=True
            )
            version = policy.version if policy else 0
            if version != expected_version:
                raise ReviewIsolationConflict("policy_version_conflict")
            now = datetime.now(UTC)
            if enabled:
                if policy is not None and policy.enabled:
                    raise ReviewIsolationConflict("policy_already_active")
                record = _receipt(session, identity.tenant_id, receipt_id or "", now)
                # A new epoch must follow the last gap, not reuse an old observation.
                document = ReaderRolloutReceipt.model_validate_json(record.receipt_text)
                if policy is not None and document.before.observed_at <= as_utc(policy.updated_at):
                    raise ReviewIsolationConflict("rollout_precedes_deactivation")
                if policy is None:
                    policy = MaintenanceReviewPolicyRecord(tenant_id=identity.tenant_id)
                    session.add(policy)
                policy.cutover_at = now
                policy.activated_by = identity.subject_id
                policy.receipt_id, policy.receipt_digest = record.receipt_id, record.receipt_digest
                policy.policy_version, policy.reader_coverage_digest = (
                    REVIEW_POLICY_VERSION,
                    READER_COVERAGE_DIGEST,
                )
            elif policy is None or not policy.enabled:
                raise ReviewIsolationConflict("policy_inactive")
            policy.enabled, policy.version, policy.updated_at = enabled, version + 1, now
            policy.idempotency_key, policy.request_digest = idempotency_key, digest
            session.flush()
            result = _policy_view(policy)
            session.add(
                MaintenanceReviewPolicyCommandRecord(
                    command_id=f"review-policy-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    idempotency_key=idempotency_key,
                    request_digest=digest,
                    result_json={**result, "reason": reason},
                    request_id=request_id,
                )
            )
            session.flush()
            return {**result, "reason": reason}


def _policy_view(policy: MaintenanceReviewPolicyRecord | None) -> dict[str, Any]:
    return {
        "version": policy.version if policy else 0,
        "enabled": bool(policy and policy.enabled),
        "policy_version": policy.policy_version if policy else REVIEW_POLICY_VERSION,
        "reader_coverage_digest": policy.reader_coverage_digest
        if policy
        else READER_COVERAGE_DIGEST,
        "cutover_at": as_utc(policy.cutover_at).isoformat()
        if policy and policy.cutover_at
        else None,
        "receipt_id": policy.receipt_id if policy else None,
        "receipt_digest": policy.receipt_digest if policy else None,
    }
