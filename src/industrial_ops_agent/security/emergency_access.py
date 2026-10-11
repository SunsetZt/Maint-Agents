"""IAM-004 tenant-bound emergency access workflow and policy resolver."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import (
    Action,
    Authorizer,
    EmergencyGrantResolution,
    ResourceContext,
)
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    EmergencyAccessDecisionRecord,
    EmergencyAccessGrantRecord,
    EmergencyAccessUsageRecord,
)

EMERGENCY_ACTION_RESOURCES: dict[Action, str] = {
    Action.READ_SECURITY_AUDIT: "security-audit",
}
EMERGENCY_GRANT_STATUSES = frozenset(
    {"PENDING", "APPROVED", "REJECTED", "REVOKED", "EXPIRED"}
)
_GRANT_ID_PATTERN = re.compile(r"^emg-[0-9a-f]{32}$")
_INCIDENT_NUMBER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{2,127}$")
Clock = Callable[[], datetime]


class EmergencyAccessNotVisible(LookupError):
    pass


class EmergencyAccessConflict(RuntimeError):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.current_version = current_version


@dataclass(frozen=True, slots=True)
class EmergencyAccessGrantAggregate:
    grant: EmergencyAccessGrantRecord
    decisions: tuple[EmergencyAccessDecisionRecord, ...]
    usages: tuple[EmergencyAccessUsageRecord, ...]

    @property
    def usage_count(self) -> int:
        return len(self.usages)


class EmergencyAccessService:
    """Create and govern short-lived grants; never performs the protected action."""

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._clock = clock or (lambda: datetime.now(UTC))

    def request_grant(
        self,
        identity: IdentityContext,
        *,
        incident_number: str,
        action: Action,
        resource_id: str,
        justification: str,
        ttl_minutes: int,
        request_id: str,
    ) -> EmergencyAccessGrantAggregate:
        self._authorizer.require(
            identity,
            Action.REQUEST_EMERGENCY_ACCESS,
            ResourceContext(identity.tenant_id, "emergency-access"),
            request_id=request_id,
        )
        incident_number = incident_number.strip()
        justification = justification.strip()
        if _INCIDENT_NUMBER_PATTERN.fullmatch(incident_number) is None:
            raise EmergencyAccessConflict("emergency_incident_number_invalid")
        expected_resource = EMERGENCY_ACTION_RESOURCES.get(action)
        if expected_resource is None or resource_id != expected_resource:
            raise EmergencyAccessConflict("emergency_action_or_resource_not_allowed")
        if not 10 <= len(justification) <= 1000:
            raise EmergencyAccessConflict("emergency_justification_invalid")
        if not 5 <= ttl_minutes <= 60:
            raise EmergencyAccessConflict("emergency_ttl_invalid")

        now = _utc(self._clock())
        grant = EmergencyAccessGrantRecord(
            grant_id=f"emg-{uuid4().hex}",
            tenant_id=identity.tenant_id,
            incident_number=incident_number,
            requester_subject_id=identity.subject_id,
            action=action.value,
            resource_id=resource_id,
            justification=justification,
            status="PENDING",
            requested_expires_at=now + timedelta(minutes=ttl_minutes),
            approved_at=None,
            approver_subject_id=None,
            revoked_at=None,
            revoked_by_subject_id=None,
            version=1,
            created_at=now,
            updated_at=now,
        )
        with self._database.transaction(identity.tenant_context) as session:
            session.add(grant)
            session.flush()
            return _aggregate(session, grant)

    def list_grants(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        status: str | None = None,
    ) -> tuple[EmergencyAccessGrantAggregate, ...]:
        self._authorizer.require(
            identity,
            Action.READ_EMERGENCY_ACCESS,
            ResourceContext(identity.tenant_id, "emergency-access"),
            request_id=request_id,
        )
        if status is not None and status not in EMERGENCY_GRANT_STATUSES:
            raise EmergencyAccessConflict("emergency_status_filter_invalid")
        now = _utc(self._clock())
        with self._database.transaction(identity.tenant_context) as session:
            statement = select(EmergencyAccessGrantRecord).where(
                EmergencyAccessGrantRecord.tenant_id == identity.tenant_id
            )
            if Role.SECURITY_AUDITOR not in identity.roles:
                statement = statement.where(
                    EmergencyAccessGrantRecord.requester_subject_id
                    == identity.subject_id
                )
            grants = list(
                session.scalars(
                    statement.order_by(EmergencyAccessGrantRecord.created_at.desc())
                )
            )
            for grant in grants:
                _expire_if_needed(session, grant, now)
            if status is not None:
                grants = [grant for grant in grants if grant.status == status]
            return tuple(_aggregate(session, grant) for grant in grants)

    def decide_grant(
        self,
        identity: IdentityContext,
        grant_id: str,
        *,
        expected_version: int,
        decision: str,
        reason: str,
        request_id: str,
    ) -> EmergencyAccessGrantAggregate:
        self._authorizer.require(
            identity,
            Action.DECIDE_EMERGENCY_ACCESS,
            ResourceContext(identity.tenant_id, grant_id),
            request_id=request_id,
        )
        normalized_decision = decision.upper()
        normalized_reason = reason.strip()
        if normalized_decision not in {"APPROVE", "REJECT"}:
            raise EmergencyAccessConflict("emergency_decision_invalid")
        _validate_reason(normalized_reason)
        now = _utc(self._clock())
        with self._database.transaction(identity.tenant_context) as session:
            grant = _load_grant(session, identity.tenant_id, grant_id, for_update=True)
            _expire_if_needed(session, grant, now)
            if grant.version != expected_version:
                raise EmergencyAccessConflict("version_conflict", grant.version)
            if grant.status != "PENDING":
                raise EmergencyAccessConflict("emergency_grant_not_pending", grant.version)
            if grant.requester_subject_id == identity.subject_id:
                raise EmergencyAccessConflict("emergency_self_approval_denied", grant.version)
            grant.status = "APPROVED" if normalized_decision == "APPROVE" else "REJECTED"
            grant.approver_subject_id = identity.subject_id
            grant.approved_at = now if normalized_decision == "APPROVE" else None
            grant.version += 1
            grant.updated_at = now
            _append_decision(
                session,
                grant,
                decision=normalized_decision,
                actor_subject_id=identity.subject_id,
                reason=normalized_reason,
                occurred_at=now,
            )
            session.flush()
            return _aggregate(session, grant)

    def revoke_grant(
        self,
        identity: IdentityContext,
        grant_id: str,
        *,
        expected_version: int,
        reason: str,
        request_id: str,
    ) -> EmergencyAccessGrantAggregate:
        self._authorizer.require(
            identity,
            Action.REVOKE_EMERGENCY_ACCESS,
            ResourceContext(identity.tenant_id, grant_id),
            request_id=request_id,
        )
        normalized_reason = reason.strip()
        _validate_reason(normalized_reason)
        now = _utc(self._clock())
        with self._database.transaction(identity.tenant_context) as session:
            grant = _load_grant(session, identity.tenant_id, grant_id, for_update=True)
            _expire_if_needed(session, grant, now)
            if grant.version != expected_version:
                raise EmergencyAccessConflict("version_conflict", grant.version)
            if grant.status != "APPROVED":
                raise EmergencyAccessConflict("emergency_grant_not_active", grant.version)
            grant.status = "REVOKED"
            grant.revoked_at = now
            grant.revoked_by_subject_id = identity.subject_id
            grant.version += 1
            grant.updated_at = now
            _append_decision(
                session,
                grant,
                decision="REVOKE",
                actor_subject_id=identity.subject_id,
                reason=normalized_reason,
                occurred_at=now,
            )
            session.flush()
            return _aggregate(session, grant)


class DatabaseEmergencyGrantPolicy:
    """Resolve explicit grant headers at the shared server-side policy boundary."""

    def __init__(self, database: Database, *, clock: Clock | None = None) -> None:
        self._database = database
        self._clock = clock or (lambda: datetime.now(UTC))

    def resolve(
        self,
        grant_id: str,
        identity: IdentityContext,
        action: Action,
        resource: ResourceContext,
    ) -> EmergencyGrantResolution:
        if _GRANT_ID_PATTERN.fullmatch(grant_id) is None:
            return EmergencyGrantResolution(False, "emergency_grant_id_invalid")
        now = _utc(self._clock())
        if _utc(identity.expires_at) <= now:
            return EmergencyGrantResolution(False, "identity_expired")
        with self._database.transaction(identity.tenant_context) as session:
            grant = session.scalar(
                select(EmergencyAccessGrantRecord).where(
                    EmergencyAccessGrantRecord.tenant_id == identity.tenant_id,
                    EmergencyAccessGrantRecord.grant_id == grant_id,
                )
            )
            if grant is None:
                return EmergencyGrantResolution(
                    False, "emergency_grant_not_found_or_not_visible"
                )
            _expire_if_needed(session, grant, now)
            return _resolve_record(grant, identity, action, resource, now)

    def record_usage(
        self,
        grant_id: str,
        identity: IdentityContext,
        action: Action,
        resource: ResourceContext,
        *,
        request_id: str,
    ) -> None:
        now = _utc(self._clock())
        resource_id = resource.resource_id
        if resource_id is None:
            raise RuntimeError("emergency_resource_id_missing")
        with self._database.transaction(identity.tenant_context) as session:
            grant = _load_grant(
                session,
                identity.tenant_id,
                grant_id,
                for_update=True,
            )
            _expire_if_needed(session, grant, now)
            resolution = _resolve_record(grant, identity, action, resource, now)
            if not resolution.allowed:
                raise RuntimeError(resolution.reason_code)
            usage_id = _stable_usage_id(
                identity.tenant_id,
                grant_id,
                request_id,
                action.value,
                resource_id,
            )
            if session.get(EmergencyAccessUsageRecord, usage_id) is None:
                session.add(
                    EmergencyAccessUsageRecord(
                        usage_id=usage_id,
                        tenant_id=identity.tenant_id,
                        grant_id=grant_id,
                        subject_id=identity.subject_id,
                        action=action.value,
                        resource_id=resource_id,
                        request_id=request_id,
                        occurred_at=now,
                        created_at=now,
                        updated_at=now,
                    )
                )


def _resolve_record(
    grant: EmergencyAccessGrantRecord,
    identity: IdentityContext,
    action: Action,
    resource: ResourceContext,
    now: datetime,
) -> EmergencyGrantResolution:
    if grant.status != "APPROVED" or _utc(grant.requested_expires_at) <= now:
        return EmergencyGrantResolution(False, "emergency_grant_not_active")
    if grant.requester_subject_id != identity.subject_id:
        return EmergencyGrantResolution(False, "emergency_grant_subject_mismatch")
    expected_resource = EMERGENCY_ACTION_RESOURCES.get(action)
    if (
        expected_resource is None
        or grant.action != action.value
        or grant.resource_id != resource.resource_id
        or grant.resource_id != expected_resource
    ):
        return EmergencyGrantResolution(False, "emergency_grant_scope_mismatch")
    return EmergencyGrantResolution(True, "emergency_grant_allowed")


def _load_grant(
    session: Session,
    tenant_id: str,
    grant_id: str,
    *,
    for_update: bool,
) -> EmergencyAccessGrantRecord:
    statement = select(EmergencyAccessGrantRecord).where(
        EmergencyAccessGrantRecord.tenant_id == tenant_id,
        EmergencyAccessGrantRecord.grant_id == grant_id,
    )
    if for_update:
        statement = statement.with_for_update()
    grant = session.scalar(statement)
    if grant is None:
        raise EmergencyAccessNotVisible
    return grant


def _expire_if_needed(
    session: Session,
    grant: EmergencyAccessGrantRecord,
    now: datetime,
) -> None:
    if grant.status not in {"PENDING", "APPROVED"}:
        return
    if _utc(grant.requested_expires_at) > now:
        return
    grant.status = "EXPIRED"
    grant.version += 1
    grant.updated_at = now
    _append_decision(
        session,
        grant,
        decision="EXPIRE",
        actor_subject_id="system-emergency-expiry",
        reason="configured_emergency_window_elapsed",
        occurred_at=now,
    )


def _append_decision(
    session: Session,
    grant: EmergencyAccessGrantRecord,
    *,
    decision: str,
    actor_subject_id: str,
    reason: str,
    occurred_at: datetime,
) -> None:
    session.add(
        EmergencyAccessDecisionRecord(
            decision_id=f"emergency-decision-{uuid4().hex}",
            tenant_id=grant.tenant_id,
            grant_id=grant.grant_id,
            decision=decision,
            actor_subject_id=actor_subject_id,
            reason=reason,
            occurred_at=occurred_at,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
    )


def _aggregate(
    session: Session,
    grant: EmergencyAccessGrantRecord,
) -> EmergencyAccessGrantAggregate:
    decisions = tuple(
        session.scalars(
            select(EmergencyAccessDecisionRecord)
            .where(
                EmergencyAccessDecisionRecord.tenant_id == grant.tenant_id,
                EmergencyAccessDecisionRecord.grant_id == grant.grant_id,
            )
            .order_by(EmergencyAccessDecisionRecord.occurred_at)
        )
    )
    usages = tuple(
        session.scalars(
            select(EmergencyAccessUsageRecord)
            .where(
                EmergencyAccessUsageRecord.tenant_id == grant.tenant_id,
                EmergencyAccessUsageRecord.grant_id == grant.grant_id,
            )
            .order_by(EmergencyAccessUsageRecord.occurred_at)
        )
    )
    return EmergencyAccessGrantAggregate(grant, decisions, usages)


def _validate_reason(reason: str) -> None:
    if not 3 <= len(reason) <= 500:
        raise EmergencyAccessConflict("emergency_decision_reason_invalid")


def _stable_usage_id(*parts: str) -> str:
    digest = sha256("\0".join(parts).encode()).hexdigest()[:32]
    return f"emergency-usage-{digest}"
