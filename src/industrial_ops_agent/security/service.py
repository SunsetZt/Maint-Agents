"""Read-only, tenant-isolated security audit queries for the formal workspace."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, func, or_, select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import SecurityAuditEventRecord


@dataclass(frozen=True, slots=True)
class SecurityAuditPage:
    events: tuple[SecurityAuditEventRecord, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class SecurityAuditSummary:
    window_start: datetime
    window_end: datetime
    total_events: int
    allowed_events: int
    denied_events: int
    guardrail_blocks: int
    action_counts: dict[str, int]


class SecurityAuditService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def list_events(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        limit: int = 50,
        cursor: str | None = None,
        decision: str | None = None,
        action: str | None = None,
    ) -> SecurityAuditPage:
        self._require(identity, request_id)
        if not 1 <= limit <= 100:
            raise ValueError("security audit limit is invalid")
        if decision is not None and decision not in {"allow", "deny"}:
            raise ValueError("security audit decision is invalid")
        if action is not None and (not action or len(action) > 128):
            raise ValueError("security audit action is invalid")
        cursor_value = _decode_cursor(cursor) if cursor else None
        with self._database.transaction(identity.tenant_context) as session:
            statement = select(SecurityAuditEventRecord).where(
                SecurityAuditEventRecord.tenant_id == identity.tenant_id
            )
            if decision is not None:
                statement = statement.where(
                    SecurityAuditEventRecord.decision == decision
                )
            if action is not None:
                statement = statement.where(SecurityAuditEventRecord.action == action)
            if cursor_value is not None:
                created_at, event_id = cursor_value
                statement = statement.where(
                    or_(
                        SecurityAuditEventRecord.created_at < created_at,
                        and_(
                            SecurityAuditEventRecord.created_at == created_at,
                            SecurityAuditEventRecord.event_id < event_id,
                        ),
                    )
                )
            records = list(
                session.scalars(
                    statement.order_by(
                        SecurityAuditEventRecord.created_at.desc(),
                        SecurityAuditEventRecord.event_id.desc(),
                    ).limit(limit + 1)
                )
            )
        visible = records[:limit]
        return SecurityAuditPage(
            events=tuple(visible),
            next_cursor=(
                _encode_cursor(visible[-1].created_at, visible[-1].event_id)
                if len(records) > limit and visible
                else None
            ),
        )

    def summarize(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        window_hours: int = 24,
    ) -> SecurityAuditSummary:
        self._require(identity, request_id)
        if not 1 <= window_hours <= 24 * 31:
            raise ValueError("security audit summary window is invalid")
        window_end = datetime.now(UTC)
        window_start = window_end - timedelta(hours=window_hours)
        base_conditions = (
            SecurityAuditEventRecord.tenant_id == identity.tenant_id,
            SecurityAuditEventRecord.created_at >= window_start,
            SecurityAuditEventRecord.created_at <= window_end,
        )
        with self._database.transaction(identity.tenant_context) as session:
            total = int(
                session.scalar(
                    select(func.count(SecurityAuditEventRecord.event_id)).where(
                        *base_conditions
                    )
                )
                or 0
            )
            allowed = int(
                session.scalar(
                    select(func.count(SecurityAuditEventRecord.event_id)).where(
                        *base_conditions,
                        SecurityAuditEventRecord.decision == "allow",
                    )
                )
                or 0
            )
            denied = int(
                session.scalar(
                    select(func.count(SecurityAuditEventRecord.event_id)).where(
                        *base_conditions,
                        SecurityAuditEventRecord.decision == "deny",
                    )
                )
                or 0
            )
            guardrail_blocks = int(
                session.scalar(
                    select(func.count(SecurityAuditEventRecord.event_id)).where(
                        *base_conditions,
                        SecurityAuditEventRecord.action == "model_gateway.guardrail",
                        SecurityAuditEventRecord.decision == "deny",
                    )
                )
                or 0
            )
            action_counts = {
                str(action): int(count)
                for action, count in session.execute(
                    select(
                        SecurityAuditEventRecord.action,
                        func.count(SecurityAuditEventRecord.event_id).label("event_count"),
                    )
                    .where(*base_conditions)
                    .group_by(SecurityAuditEventRecord.action)
                    .order_by(func.count(SecurityAuditEventRecord.event_id).desc())
                    .limit(10)
                )
            }
        return SecurityAuditSummary(
            window_start=window_start,
            window_end=window_end,
            total_events=total,
            allowed_events=allowed,
            denied_events=denied,
            guardrail_blocks=guardrail_blocks,
            action_counts=action_counts,
        )

    def _require(self, identity: IdentityContext, request_id: str) -> None:
        self._authorizer.require(
            identity,
            Action.READ_SECURITY_AUDIT,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="security-audit",
            ),
            request_id=request_id,
        )


def _encode_cursor(created_at: datetime, event_id: str) -> str:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    payload = json.dumps(
        {"created_at": created_at.isoformat(), "event_id": event_id},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(value: str) -> tuple[datetime, str]:
    try:
        padding = "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(value + padding))
        created_at = datetime.fromisoformat(payload["created_at"])
        event_id = payload["event_id"]
    except (binascii.Error, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("security audit cursor is invalid") from exc
    if created_at.tzinfo is None or not isinstance(event_id, str) or not event_id:
        raise ValueError("security audit cursor is invalid")
    return created_at, event_id
