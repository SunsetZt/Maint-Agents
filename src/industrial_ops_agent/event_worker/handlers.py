"""Transactional Inbox and aggregate-version handling for feedback events."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from opentelemetry.trace import SpanKind
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from industrial_ops_agent.data_governance.service import (
    FeedbackCandidateService,
    FeedbackSourceRejected,
)
from industrial_ops_agent.eventing.envelope import EventEnvelope
from industrial_ops_agent.eventing.trace_context import ExtractedEventTraceContext
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    EventAggregateProjectionRecord,
    EventInboxRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.telemetry import safe_tenant_hash, traced_operation

CONSUMER_NAME = "m3-feedback-candidate-v1"


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    disposition: str
    event_id: str | None
    candidate_id: str | None
    commit_offset: bool
    reason: str | None = None
    occurred_at: datetime | None = None
    source_trace_id: str | None = None
    tenant_hash: str | None = None
    trace_context_propagation: str = "LEGACY_ROOT"


class FeedbackEventProcessor:
    def __init__(
        self,
        database: Database,
        candidate_service: FeedbackCandidateService | None = None,
    ) -> None:
        self._database = database
        self._candidate_service = candidate_service or FeedbackCandidateService()

    def process(
        self,
        raw_event: bytes | str | dict[str, Any],
        *,
        trace_context: ExtractedEventTraceContext | None = None,
    ) -> ProcessingResult:
        value: Any = None
        try:
            value = _decode(raw_event)
            envelope = EventEnvelope.model_validate(value)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError) as exc:
            event_id = value.get("event_id") if isinstance(value, dict) else None
            return ProcessingResult(
                "REJECTED_CONTRACT",
                event_id if isinstance(event_id, str) else None,
                None,
                False,
                type(exc).__name__,
            )

        event_digest = _digest(envelope.model_dump(mode="json"))
        context = TenantContext(
            tenant_id=envelope.tenant_id,
            subject_id="service-m3-feedback-consumer",
        )
        tenant_hash = safe_tenant_hash(envelope.tenant_id)
        with traced_operation(
            "messaging.process work_order.closed",
            kind=SpanKind.CONSUMER,
            attributes={
                "messaging.system": "kafka",
                "messaging.operation.type": "process",
                "messaging.message.id": envelope.event_id,
                "ioap.event.type": envelope.event_type,
                "ioap.aggregate.type": envelope.aggregate_type,
                "ioap.aggregate.id": envelope.aggregate_id,
                "ioap.aggregate.version": envelope.aggregate_version,
                "ioap.correlation.trace_id": envelope.trace_id,
                "ioap.tenant_hash": tenant_hash,
                "ioap.trace.propagation": (
                    trace_context.propagation
                    if trace_context is not None
                    else "LEGACY_ROOT"
                ),
            },
            parent_context=(
                trace_context.parent_context if trace_context is not None else None
            ),
        ) as span:
            try:
                with self._database.transaction(context) as session:
                    result = self._process_transaction(
                        session,
                        envelope,
                        event_digest,
                        trace_context,
                    )
            except IntegrityError:
                result = self._resolve_concurrent_delivery(context, envelope, event_digest)
            except FeedbackSourceRejected as exc:
                result = ProcessingResult(
                    "REJECTED_SOURCE",
                    envelope.event_id,
                    None,
                    False,
                    str(exc),
                )
            span.set_attribute("ioap.event.disposition", result.disposition)
            span.set_attribute("ioap.kafka.commit_offset", result.commit_offset)
            if result.reason is not None:
                span.set_attribute("ioap.event.reason", result.reason)
            return replace(
                result,
                occurred_at=envelope.occurred_at,
                source_trace_id=envelope.trace_id,
                tenant_hash=tenant_hash,
                trace_context_propagation=(
                    trace_context.propagation
                    if trace_context is not None
                    else "LEGACY_ROOT"
                ),
            )

    def _process_transaction(
        self,
        session: Session,
        envelope: EventEnvelope,
        event_digest: str,
        trace_context: ExtractedEventTraceContext | None,
    ) -> ProcessingResult:
        existing = self._load_inbox(session, envelope)
        if existing is not None:
            return self._existing_result(existing, event_digest)

        now = datetime.now(UTC)
        inbox = EventInboxRecord(
            inbox_id=_stable_id("inbox", CONSUMER_NAME, envelope.tenant_id, envelope.event_id),
            tenant_id=envelope.tenant_id,
            consumer_name=CONSUMER_NAME,
            event_id=envelope.event_id,
            event_digest=event_digest,
            event_type=envelope.event_type,
            aggregate_type=envelope.aggregate_type,
            aggregate_id=envelope.aggregate_id,
            aggregate_version=envelope.aggregate_version,
            trace_context_propagation=(
                trace_context.propagation
                if trace_context is not None
                else "LEGACY_ROOT"
            ),
            parent_trace_id=(
                trace_context.carrier.trace_id if trace_context is not None else None
            ),
            status="RECEIVED",
            result_ref=None,
            failure_reason=None,
            received_at=now,
            processed_at=None,
            created_at=now,
            updated_at=now,
        )
        session.add(inbox)
        session.flush()

        projection = self._load_projection(session, envelope)
        if projection is not None and envelope.aggregate_version <= (
            projection.last_aggregate_version
        ):
            inbox.status = "STALE_SKIPPED"
            inbox.processed_at = now
            inbox.updated_at = now
            return ProcessingResult("STALE_SKIPPED", envelope.event_id, None, True)
        if projection is not None and envelope.aggregate_version > (
            projection.last_aggregate_version + 1
        ):
            projection.status = "GAP_BLOCKED"
            projection.gap_expected_version = projection.last_aggregate_version + 1
            projection.gap_received_version = envelope.aggregate_version
            projection.updated_at = now
            inbox.status = "GAP_BLOCKED"
            inbox.failure_reason = "aggregate_version_gap"
            inbox.processed_at = now
            inbox.updated_at = now
            return ProcessingResult(
                "GAP_BLOCKED",
                envelope.event_id,
                None,
                False,
                "aggregate_version_gap",
            )

        candidate = self._candidate_service.create_from_closed_work_order(
            session,
            envelope,
            consumer_name=CONSUMER_NAME,
        )
        if projection is None:
            projection = EventAggregateProjectionRecord(
                projection_id=_stable_id(
                    "projection",
                    CONSUMER_NAME,
                    envelope.tenant_id,
                    envelope.aggregate_type,
                    envelope.aggregate_id,
                ),
                tenant_id=envelope.tenant_id,
                consumer_name=CONSUMER_NAME,
                aggregate_type=envelope.aggregate_type,
                aggregate_id=envelope.aggregate_id,
                last_aggregate_version=envelope.aggregate_version,
                status="ACTIVE",
                gap_expected_version=None,
                gap_received_version=None,
                created_at=now,
                updated_at=now,
            )
            session.add(projection)
        else:
            projection.last_aggregate_version = envelope.aggregate_version
            projection.status = "ACTIVE"
            projection.gap_expected_version = None
            projection.gap_received_version = None
            projection.updated_at = now
        inbox.status = "PROCESSED"
        inbox.result_ref = candidate.candidate_id
        inbox.processed_at = now
        inbox.updated_at = now
        return ProcessingResult(
            "PROCESSED",
            envelope.event_id,
            candidate.candidate_id,
            True,
        )

    def _resolve_concurrent_delivery(
        self,
        context: TenantContext,
        envelope: EventEnvelope,
        event_digest: str,
    ) -> ProcessingResult:
        with self._database.transaction(context) as session:
            existing = self._load_inbox(session, envelope)
            if existing is None:
                return ProcessingResult(
                    "REJECTED_SOURCE",
                    envelope.event_id,
                    None,
                    False,
                    "concurrent_delivery_unresolved",
                )
            return self._existing_result(existing, event_digest)

    @staticmethod
    def _load_inbox(
        session: Session,
        envelope: EventEnvelope,
    ) -> EventInboxRecord | None:
        return session.scalar(
            select(EventInboxRecord).where(
                EventInboxRecord.tenant_id == envelope.tenant_id,
                EventInboxRecord.consumer_name == CONSUMER_NAME,
                EventInboxRecord.event_id == envelope.event_id,
            )
        )

    @staticmethod
    def _load_projection(
        session: Session,
        envelope: EventEnvelope,
    ) -> EventAggregateProjectionRecord | None:
        return session.scalar(
            select(EventAggregateProjectionRecord).where(
                EventAggregateProjectionRecord.tenant_id == envelope.tenant_id,
                EventAggregateProjectionRecord.consumer_name == CONSUMER_NAME,
                EventAggregateProjectionRecord.aggregate_type == envelope.aggregate_type,
                EventAggregateProjectionRecord.aggregate_id == envelope.aggregate_id,
            )
        )

    @staticmethod
    def _existing_result(existing: EventInboxRecord, event_digest: str) -> ProcessingResult:
        if existing.event_digest != event_digest:
            return ProcessingResult(
                "REJECTED_CONTRACT",
                existing.event_id,
                None,
                False,
                "event_id_payload_mismatch",
            )
        if existing.status == "GAP_BLOCKED":
            return ProcessingResult(
                "GAP_BLOCKED",
                existing.event_id,
                None,
                False,
                existing.failure_reason,
            )
        return ProcessingResult(
            "DUPLICATE",
            existing.event_id,
            existing.result_ref,
            True,
            trace_context_propagation=(
                existing.trace_context_propagation or "LEGACY_ROOT"
            ),
        )


def _decode(raw_event: bytes | str | dict[str, Any]) -> Any:
    if isinstance(raw_event, bytes):
        value = json.loads(raw_event.decode("utf-8"))
    elif isinstance(raw_event, str):
        value = json.loads(raw_event)
    elif isinstance(raw_event, dict):
        value = raw_event
    else:
        raise TypeError("event payload must be bytes, text or object")

    # Accept Kafka Connect's schema-enabled JSON envelope during rolling
    # converter migrations. The inner payload still passes the strict domain
    # event validation below, so this does not broaden the event contract.
    if (
        isinstance(value, dict)
        and isinstance(value.get("schema"), dict)
        and isinstance(value.get("payload"), dict)
    ):
        return value["payload"]
    return value


def _digest(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + sha256(canonical.encode()).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-" + sha256("\0".join(parts).encode()).hexdigest()[:32]
