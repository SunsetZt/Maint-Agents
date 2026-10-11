"""Minimal structured security audit with keyed subject and tenant hashes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from hmac import new as hmac_new
from typing import Protocol
from uuid import uuid4

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import SecurityAuditEventRecord
from industrial_ops_agent.persistence.tenant import TenantContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SecurityAuditEvent:
    event_id: str
    occurred_at: datetime
    subject_hash: str
    tenant_hash: str
    action: str
    decision: str
    reason_code: str
    request_id: str
    resource_hash: str | None
    tenant_id: str | None


class SecurityAuditSink(Protocol):
    def write(self, event: SecurityAuditEvent) -> None: ...


class InMemorySecurityAuditSink:
    def __init__(self) -> None:
        self.events: list[SecurityAuditEvent] = []

    def write(self, event: SecurityAuditEvent) -> None:
        self.events.append(event)


class StructuredLoggingSecurityAuditSink:
    """Emit a content-free fallback record for central log collection."""

    def write(self, event: SecurityAuditEvent) -> None:
        logger.info(
            "security_audit event_id=%s tenant_hash=%s subject_hash=%s action=%s "
            "decision=%s reason_code=%s request_id=%s resource_hash=%s",
            event.event_id,
            event.tenant_hash,
            event.subject_hash,
            event.action,
            event.decision,
            event.reason_code,
            event.request_id,
            event.resource_hash or "none",
        )


class DatabaseSecurityAuditSink:
    """Persist tenant-bound audit events under the same PostgreSQL RLS context."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def write(self, event: SecurityAuditEvent) -> None:
        if event.tenant_id is None:
            return
        context = TenantContext(event.tenant_id, "security-audit-writer")
        with self._database.transaction(context) as session:
            session.add(
                SecurityAuditEventRecord(
                    event_id=event.event_id,
                    tenant_id=event.tenant_id,
                    subject_hash=event.subject_hash,
                    tenant_hash=event.tenant_hash,
                    action=event.action,
                    decision=event.decision,
                    reason=event.reason_code,
                    request_id=event.request_id,
                    resource_hash=event.resource_hash,
                    created_at=event.occurred_at,
                    updated_at=event.occurred_at,
                )
            )


class CompositeSecurityAuditSink:
    def __init__(self, *sinks: SecurityAuditSink) -> None:
        if not sinks:
            raise ValueError("at least one security audit sink is required")
        self._sinks = sinks

    def write(self, event: SecurityAuditEvent) -> None:
        for sink in self._sinks:
            sink.write(event)


class SecurityAuditor:
    def __init__(self, sink: SecurityAuditSink, *, hash_key: bytes) -> None:
        if len(hash_key) < 8:
            raise ValueError("security audit hash key is too short")
        self._sink = sink
        self._hash_key = hash_key

    def record(
        self,
        *,
        identity: IdentityContext | None,
        action: str,
        decision: str,
        reason_code: str,
        request_id: str,
        resource_id: str | None = None,
    ) -> None:
        self.record_context(
            tenant_id=identity.tenant_id if identity else None,
            subject_id=identity.subject_id if identity else None,
            action=action,
            decision=decision,
            reason_code=reason_code,
            request_id=request_id,
            resource_id=resource_id,
        )

    def record_context(
        self,
        *,
        tenant_id: str | None,
        subject_id: str | None,
        action: str,
        decision: str,
        reason_code: str,
        request_id: str,
        resource_id: str | None = None,
    ) -> None:
        self._sink.write(
            SecurityAuditEvent(
                event_id=f"audit-{uuid4().hex}",
                occurred_at=datetime.now(UTC),
                subject_hash=self._hash(subject_id or "anonymous"),
                tenant_hash=self._hash(tenant_id or "unresolved"),
                action=action,
                decision=decision,
                reason_code=reason_code,
                request_id=request_id,
                resource_hash=self._hash(resource_id) if resource_id else None,
                tenant_id=tenant_id,
            )
        )

    def _hash(self, value: str) -> str:
        return hmac_new(self._hash_key, value.encode(), sha256).hexdigest()


def build_persistent_security_auditor(
    database: Database, *, hash_key: bytes
) -> SecurityAuditor:
    return SecurityAuditor(
        CompositeSecurityAuditSink(
            StructuredLoggingSecurityAuditSink(),
            DatabaseSecurityAuditSink(database),
        ),
        hash_key=hash_key,
    )
