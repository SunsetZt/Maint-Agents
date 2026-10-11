"""Governed memory lifecycle and prompt-context selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.maintenance_planning.review_isolation import (
    ReviewIsolationConflict,
    lock_subject,
    record_incident_disclosure,
)
from industrial_ops_agent.model_gateway.context_manifest import (
    ContextTruncation,
    canonical_sha256,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AgentEventRecord,
    AgentRunRecord,
    DiagnosisRunRecord,
    GovernedMemoryRecord,
    GovernedMemoryTransitionRecord,
    IncidentRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext

MemoryType = Literal["SESSION_NOTE", "USER_PREFERENCE"]

_MEMORY_TYPES = frozenset({"SESSION_NOTE", "USER_PREFERENCE"})
_MAX_CONTENT = {"SESSION_NOTE": 2_000, "USER_PREFERENCE": 500}
_DIAGNOSIS_MEMORY_LIMIT = 6
_DIAGNOSIS_MEMORY_CHARACTERS = 600
_DIAGNOSIS_CANDIDATE_LIMIT = 32


class MemoryNotVisible(LookupError):
    pass


class MemoryConflict(RuntimeError):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class MemoryTransition:
    transition_id: str
    sequence: int
    from_status: str
    to_status: str
    actor_subject_id: str
    reason_code: str
    evidence_hash: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryView:
    memory_id: str
    memory_type: str
    purpose: str
    status: str
    owner_subject_id: str
    incident_id: str | None
    asset_id: str | None
    content: str
    content_hash: str
    source_reference_type: str
    source_reference_id: str
    confirmed_by_subject_id: str
    revoked_by_subject_id: str | None
    revocation_reason: str | None
    revoked_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    legal_actions: tuple[str, ...]
    transitions: tuple[MemoryTransition, ...]


@dataclass(frozen=True, slots=True)
class MemoryContextItem:
    memory_id: str
    memory_type: str
    content: str
    source_reference_id: str

    def prompt_payload(self) -> dict[str, str]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "source_reference_id": self.source_reference_id,
        }


@dataclass(frozen=True, slots=True)
class MemoryContextSelection:
    items: tuple[MemoryContextItem, ...]
    truncations: tuple[ContextTruncation, ...]
    rejected_memory_ids: tuple[str, ...] = ()


class GovernedMemoryService:
    POLICY_VERSION = "human-confirmed-memory/v1"

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def list(
        self,
        identity: IdentityContext,
        *,
        memory_type: str | None = None,
        status: str | None = None,
        incident_id: str | None = None,
        request_id: str,
    ) -> tuple[MemoryView, ...]:
        self._authorizer.require(
            identity,
            Action.READ_MEMORY,
            ResourceContext(identity.tenant_id, "governed-memories"),
            request_id=request_id,
        )
        if memory_type is not None and memory_type not in _MEMORY_TYPES:
            raise ValueError("memory_type_invalid")
        if status is not None and status not in {"ACTIVE", "REVOKED"}:
            raise ValueError("memory_status_invalid")
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            statement = select(GovernedMemoryRecord).where(
                GovernedMemoryRecord.tenant_id == identity.tenant_id,
                or_(
                    GovernedMemoryRecord.owner_subject_id == identity.subject_id,
                    GovernedMemoryRecord.memory_type == "SESSION_NOTE",
                ),
            )
            if memory_type is not None:
                statement = statement.where(GovernedMemoryRecord.memory_type == memory_type)
            if status is not None:
                statement = statement.where(GovernedMemoryRecord.status == status)
            if incident_id is not None:
                statement = statement.where(GovernedMemoryRecord.incident_id == incident_id)
            records = tuple(
                session.scalars(
                    statement.order_by(
                        GovernedMemoryRecord.created_at.desc(),
                        GovernedMemoryRecord.memory_id,
                    )
                )
            )
            transitions = _transitions_by_memory(session, identity.tenant_id)
            return tuple(
                _view(
                    session, identity, self._authorizer, item,
                    transitions.get(item.memory_id, ()), request_id=request_id,
                )
                for item in records
                if _can_read(identity, self._authorizer, item)
            )

    def create(
        self,
        identity: IdentityContext,
        *,
        memory_type: str,
        content: str,
        source_reference_id: str,
        incident_id: str | None,
        confirmed: bool,
        idempotency_key: str,
        request_id: str,
    ) -> MemoryView:
        if memory_type not in _MEMORY_TYPES:
            raise ValueError("memory_type_invalid")
        if not confirmed:
            raise ValueError("memory_human_confirmation_required")
        normalized = " ".join(content.split())
        if not normalized or len(normalized) > _MAX_CONTENT[memory_type]:
            raise ValueError("memory_content_invalid")
        if not 3 <= len(source_reference_id) <= 255:
            raise ValueError("memory_source_reference_invalid")
        if not 8 <= len(idempotency_key) <= 255:
            raise ValueError("memory_idempotency_key_invalid")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            asset_id, source_type = _validated_scope_and_source(
                session,
                identity,
                memory_type=memory_type,
                incident_id=incident_id,
                source_reference_id=source_reference_id,
            )
            self._authorizer.require(
                identity,
                Action.WRITE_MEMORY,
                ResourceContext(
                    identity.tenant_id,
                    incident_id or identity.subject_id,
                    asset_id=asset_id,
                ),
                request_id=request_id,
            )
            content_hash = canonical_sha256({"content": normalized})
            existing = session.scalar(
                select(GovernedMemoryRecord).where(
                    GovernedMemoryRecord.tenant_id == identity.tenant_id,
                    GovernedMemoryRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if not _same_create_request(
                    existing,
                    memory_type=memory_type,
                    incident_id=incident_id,
                    content_hash=content_hash,
                    source_reference_id=source_reference_id,
                ):
                    raise MemoryConflict("memory_idempotency_key_reused")
                return _view(
                    session,
                    identity,
                    self._authorizer,
                    existing,
                    _transitions_for(session, identity.tenant_id, existing.memory_id),
                    request_id=request_id,
                )
            duplicate_source = session.scalar(
                select(GovernedMemoryRecord).where(
                    GovernedMemoryRecord.tenant_id == identity.tenant_id,
                    GovernedMemoryRecord.memory_type == memory_type,
                    GovernedMemoryRecord.source_reference_id == source_reference_id,
                )
            )
            if duplicate_source is not None:
                raise MemoryConflict("memory_source_reference_already_used")
            record = GovernedMemoryRecord(
                memory_id=f"memory-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                idempotency_key=idempotency_key,
                memory_type=memory_type,
                purpose="DIAGNOSIS",
                status="ACTIVE",
                owner_subject_id=identity.subject_id,
                incident_id=incident_id,
                asset_id=asset_id,
                content=normalized,
                content_hash=content_hash,
                source_reference_type=source_type,
                source_reference_id=source_reference_id,
                confirmed_by_subject_id=identity.subject_id,
                revoked_by_subject_id=None,
                revocation_reason=None,
                revoked_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            transition = _append_transition(
                session,
                record,
                sequence=1,
                from_status="NONE",
                to_status="ACTIVE",
                actor_subject_id=identity.subject_id,
                reason_code="human_confirmation_recorded",
                occurred_at=now,
            )
            return _view(
                session, identity, self._authorizer, record, (transition,),
                request_id=request_id,
            )

    def revoke(
        self,
        identity: IdentityContext,
        memory_id: str,
        *,
        expected_version: int,
        reason: str,
        request_id: str,
    ) -> MemoryView:
        normalized_reason = " ".join(reason.split())
        if not 8 <= len(normalized_reason) <= 1_000:
            raise ValueError("memory_revocation_reason_invalid")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = session.scalar(
                select(GovernedMemoryRecord)
                .where(
                    GovernedMemoryRecord.tenant_id == identity.tenant_id,
                    GovernedMemoryRecord.memory_id == memory_id,
                )
                .with_for_update()
            )
            if record is None or not _can_read(identity, self._authorizer, record):
                raise MemoryNotVisible(memory_id)
            self._authorizer.require(
                identity,
                Action.REVOKE_MEMORY,
                ResourceContext(
                    identity.tenant_id,
                    memory_id,
                    asset_id=record.asset_id,
                ),
                request_id=request_id,
            )
            if not _can_revoke(identity, record):
                raise MemoryNotVisible(memory_id)
            if record.version != expected_version:
                raise MemoryConflict("memory_version_conflict", record.version)
            if record.status != "ACTIVE":
                raise MemoryConflict("memory_not_active", record.version)
            record.status = "REVOKED"
            record.revoked_by_subject_id = identity.subject_id
            record.revocation_reason = normalized_reason
            record.revoked_at = now
            record.version += 1
            record.updated_at = now
            _append_transition(
                session,
                record,
                sequence=2,
                from_status="ACTIVE",
                to_status="REVOKED",
                actor_subject_id=identity.subject_id,
                reason_code="memory_revoked",
                occurred_at=now,
            )
            return _view(
                session,
                identity,
                self._authorizer,
                record,
                _transitions_for(session, identity.tenant_id, memory_id),
                request_id=request_id,
            )


class MemoryContextReader:
    """Internal read boundary used only by a tenant-bound diagnosis activity."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def select_for_diagnosis(
        self,
        context: TenantContext,
        *,
        subject_id: str,
        incident_id: str,
    ) -> MemoryContextSelection:
        with self._database.transaction(context) as session:
            incident_asset_id = session.scalar(
                select(IncidentRecord.asset_id).where(
                    IncidentRecord.tenant_id == context.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident_asset_id is None:
                raise RuntimeError("diagnosis memory incident is unavailable")
            records = tuple(
                session.scalars(
                    select(GovernedMemoryRecord)
                    .where(
                        GovernedMemoryRecord.tenant_id == context.tenant_id,
                        GovernedMemoryRecord.status == "ACTIVE",
                        GovernedMemoryRecord.purpose == "DIAGNOSIS",
                        or_(
                            and_(
                                GovernedMemoryRecord.memory_type == "SESSION_NOTE",
                                GovernedMemoryRecord.incident_id == incident_id,
                                GovernedMemoryRecord.asset_id == incident_asset_id,
                            ),
                            and_(
                                GovernedMemoryRecord.memory_type == "USER_PREFERENCE",
                                GovernedMemoryRecord.owner_subject_id == subject_id,
                            ),
                        ),
                    )
                    .order_by(
                        GovernedMemoryRecord.created_at.desc(),
                        GovernedMemoryRecord.memory_id,
                    )
                    .limit(_DIAGNOSIS_CANDIDATE_LIMIT)
                )
            )
        selected: list[MemoryContextItem] = []
        truncations: list[ContextTruncation] = []
        rejected: list[str] = []
        for record in records:
            if canonical_sha256({"content": record.content}) != record.content_hash:
                rejected.append(record.memory_id)
                continue
            if len(selected) >= _DIAGNOSIS_MEMORY_LIMIT:
                truncations.append(
                    ContextTruncation(
                        "MEMORY",
                        record.memory_id,
                        "COUNT_LIMIT",
                        len(record.content),
                        0,
                    )
                )
                continue
            content = record.content[:_DIAGNOSIS_MEMORY_CHARACTERS]
            if len(record.content) > len(content):
                truncations.append(
                    ContextTruncation(
                        "MEMORY",
                        record.memory_id,
                        "CHARACTER_LIMIT",
                        len(record.content),
                        len(content),
                    )
                )
            selected.append(
                MemoryContextItem(
                    memory_id=record.memory_id,
                    memory_type=record.memory_type,
                    content=content,
                    source_reference_id=record.source_reference_id,
                )
            )
        return MemoryContextSelection(tuple(selected), tuple(truncations), tuple(rejected))


def _validated_scope_and_source(
    session: Session,
    identity: IdentityContext,
    *,
    memory_type: str,
    incident_id: str | None,
    source_reference_id: str,
) -> tuple[str | None, str]:
    if memory_type == "USER_PREFERENCE":
        if incident_id is not None:
            raise ValueError("user_preference_incident_not_allowed")
        return None, "USER_CONFIRMATION"
    if incident_id is None:
        raise ValueError("session_memory_incident_required")
    incident = session.scalar(
        select(IncidentRecord).where(
            IncidentRecord.tenant_id == identity.tenant_id,
            IncidentRecord.incident_id == incident_id,
        )
    )
    event = session.scalar(
        select(AgentEventRecord)
        .join(AgentRunRecord, AgentRunRecord.agent_run_id == AgentEventRecord.agent_run_id)
        .join(
            DiagnosisRunRecord,
            DiagnosisRunRecord.diagnosis_run_id == AgentRunRecord.diagnosis_run_id,
        )
        .where(
            AgentEventRecord.tenant_id == identity.tenant_id,
            AgentEventRecord.event_id == source_reference_id,
            AgentEventRecord.event_type == "agent.user_input.confirmed",
            AgentRunRecord.tenant_id == identity.tenant_id,
            DiagnosisRunRecord.tenant_id == identity.tenant_id,
            DiagnosisRunRecord.incident_id == incident_id,
        )
    )
    if (
        incident is None
        or event is None
        or event.payload.get("reviewer_subject_id") != identity.subject_id
    ):
        raise MemoryNotVisible(source_reference_id)
    return incident.asset_id, "AGENT_EVENT"


def _same_create_request(
    record: GovernedMemoryRecord,
    *,
    memory_type: str,
    incident_id: str | None,
    content_hash: str,
    source_reference_id: str,
) -> bool:
    return (
        record.memory_type == memory_type
        and record.incident_id == incident_id
        and record.content_hash == content_hash
        and record.source_reference_id == source_reference_id
    )


def _can_read(
    identity: IdentityContext,
    authorizer: Authorizer,
    record: GovernedMemoryRecord,
) -> bool:
    if record.memory_type == "USER_PREFERENCE" and record.owner_subject_id != identity.subject_id:
        return False
    return authorizer.decide(
        identity,
        Action.READ_MEMORY,
        ResourceContext(identity.tenant_id, record.memory_id, asset_id=record.asset_id),
    ).allowed


def _can_revoke(identity: IdentityContext, record: GovernedMemoryRecord) -> bool:
    if record.owner_subject_id == identity.subject_id:
        return True
    return record.memory_type == "SESSION_NOTE" and bool(
        identity.roles & {Role.DOMAIN_EXPERT, Role.TENANT_ADMIN}
    )


def _legal_actions(
    identity: IdentityContext,
    authorizer: Authorizer,
    record: GovernedMemoryRecord,
) -> tuple[str, ...]:
    if record.status != "ACTIVE" or not _can_revoke(identity, record):
        return ()
    allowed = authorizer.decide(
        identity,
        Action.REVOKE_MEMORY,
        ResourceContext(identity.tenant_id, record.memory_id, asset_id=record.asset_id),
    ).allowed
    return ("REVOKE",) if allowed else ()


def _view(
    session: Session,
    identity: IdentityContext,
    authorizer: Authorizer,
    record: GovernedMemoryRecord,
    transitions: tuple[GovernedMemoryTransitionRecord, ...],
    *,
    request_id: str,
) -> MemoryView:
    # Only human-facing projections record exposure, never internal prompt selection.
    if record.memory_type == "SESSION_NOTE":
        incident = session.scalar(
            select(IncidentRecord).where(
                IncidentRecord.tenant_id == identity.tenant_id,
                IncidentRecord.incident_id == record.incident_id,
                IncidentRecord.asset_id == record.asset_id,
            )
        )
        if incident is None:
            raise ReviewIsolationConflict("source_binding_invalid")
        record_incident_disclosure(
            session,
            identity,
            (incident.incident_id,),
            resource_kind="governed-memory",
            resource_id=record.memory_id,
            request_id=request_id,
        )
    return MemoryView(
        memory_id=record.memory_id,
        memory_type=record.memory_type,
        purpose=record.purpose,
        status=record.status,
        owner_subject_id=record.owner_subject_id,
        incident_id=record.incident_id,
        asset_id=record.asset_id,
        content=record.content,
        content_hash=record.content_hash,
        source_reference_type=record.source_reference_type,
        source_reference_id=record.source_reference_id,
        confirmed_by_subject_id=record.confirmed_by_subject_id,
        revoked_by_subject_id=record.revoked_by_subject_id,
        revocation_reason=record.revocation_reason,
        revoked_at=_utc_optional(record.revoked_at),
        version=record.version,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
        legal_actions=_legal_actions(identity, authorizer, record),
        transitions=tuple(_transition(item) for item in transitions),
    )


def _append_transition(
    session: Session,
    record: GovernedMemoryRecord,
    *,
    sequence: int,
    from_status: str,
    to_status: str,
    actor_subject_id: str,
    reason_code: str,
    occurred_at: datetime,
) -> GovernedMemoryTransitionRecord:
    evidence_hash = canonical_sha256(
        {
            "memory_id": record.memory_id,
            "content_hash": record.content_hash,
            "from_status": from_status,
            "to_status": to_status,
            "actor_subject_id": actor_subject_id,
            "reason_code": reason_code,
        }
    )
    transition = GovernedMemoryTransitionRecord(
        transition_id=f"memory-transition-{uuid4().hex}",
        tenant_id=record.tenant_id,
        memory_id=record.memory_id,
        sequence=sequence,
        from_status=from_status,
        to_status=to_status,
        actor_subject_id=actor_subject_id,
        reason_code=reason_code,
        evidence_hash=evidence_hash,
        occurred_at=occurred_at,
        created_at=occurred_at,
        updated_at=occurred_at,
    )
    session.add(transition)
    session.flush()
    return transition


def _transitions_by_memory(
    session: Session,
    tenant_id: str,
) -> dict[str, tuple[GovernedMemoryTransitionRecord, ...]]:
    result: dict[str, list[GovernedMemoryTransitionRecord]] = {}
    for item in session.scalars(
        select(GovernedMemoryTransitionRecord)
        .where(GovernedMemoryTransitionRecord.tenant_id == tenant_id)
        .order_by(
            GovernedMemoryTransitionRecord.memory_id,
            GovernedMemoryTransitionRecord.sequence,
        )
    ):
        result.setdefault(item.memory_id, []).append(item)
    return {key: tuple(value) for key, value in result.items()}


def _transitions_for(
    session: Session,
    tenant_id: str,
    memory_id: str,
) -> tuple[GovernedMemoryTransitionRecord, ...]:
    return tuple(
        session.scalars(
            select(GovernedMemoryTransitionRecord)
            .where(
                GovernedMemoryTransitionRecord.tenant_id == tenant_id,
                GovernedMemoryTransitionRecord.memory_id == memory_id,
            )
            .order_by(GovernedMemoryTransitionRecord.sequence)
        )
    )


def _transition(record: GovernedMemoryTransitionRecord) -> MemoryTransition:
    return MemoryTransition(
        transition_id=record.transition_id,
        sequence=record.sequence,
        from_status=record.from_status,
        to_status=record.to_status,
        actor_subject_id=record.actor_subject_id,
        reason_code=record.reason_code,
        evidence_hash=record.evidence_hash,
        occurred_at=_utc(record.occurred_at),
    )


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _utc_optional(value: datetime | None) -> datetime | None:
    return _utc(value) if value is not None else None
