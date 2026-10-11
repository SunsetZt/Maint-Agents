"""Minimal versioned event envelope owned by the business application."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.eventing.trace_context import (
    TRACE_CONTEXT_PROFILE,
    W3CTraceContext,
)
from industrial_ops_agent.persistence.models import EventOutboxRecord

PRODUCER = "industrial-ops-api@0.1.0"


class WorkOrderClosedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    work_order_id: str = Field(min_length=1, max_length=128)
    incident_id: str = Field(min_length=1, max_length=128)
    completion_id: str = Field(min_length=1, max_length=128)
    verification_id: str = Field(min_length=1, max_length=128)
    source_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(pattern=r"^event-[0-9a-f]{32}$")
    event_type: Literal["work_order.closed"]
    event_version: Literal[1]
    tenant_id: str = Field(min_length=1, max_length=64)
    aggregate_type: Literal["work_order"]
    aggregate_id: str = Field(min_length=1, max_length=128)
    aggregate_version: int = Field(ge=1)
    sequence: int = Field(ge=1)
    occurred_at: datetime
    published_at: datetime | None = None
    producer: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=128)
    causation_id: str = Field(min_length=1, max_length=128)
    data_classification: Literal["CONFIDENTIAL"]
    payload: WorkOrderClosedPayload

    def to_outbox_record(
        self,
        *,
        trace_context: W3CTraceContext | None = None,
    ) -> EventOutboxRecord:
        """Create the ORM row without performing I/O or publishing a message."""

        return EventOutboxRecord(
            event_id=self.event_id,
            event_type=self.event_type,
            event_version=self.event_version,
            tenant_id=self.tenant_id,
            aggregate_type=self.aggregate_type,
            aggregate_id=self.aggregate_id,
            aggregate_version=self.aggregate_version,
            sequence=self.sequence,
            occurred_at=self.occurred_at,
            published_at=self.published_at,
            producer=self.producer,
            partition_key=f"{self.tenant_id}:{self.aggregate_id}",
            trace_id=self.trace_id,
            trace_context_profile=(
                TRACE_CONTEXT_PROFILE if trace_context is not None else None
            ),
            source_traceparent=(
                trace_context.traceparent if trace_context is not None else None
            ),
            source_tracestate=(
                trace_context.tracestate if trace_context is not None else None
            ),
            tracingspancontext=(
                trace_context.as_debezium_properties()
                if trace_context is not None
                else None
            ),
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            data_classification=self.data_classification,
            payload=self.payload.model_dump(mode="json"),
            created_at=self.occurred_at,
            updated_at=self.occurred_at,
        )


def build_work_order_closed_event(
    *,
    tenant_id: str,
    work_order_id: str,
    incident_id: str,
    completion_id: str,
    verification_id: str,
    aggregate_version: int,
    occurred_at: datetime,
    request_id: str,
    trace_id: str | None = None,
) -> EventEnvelope:
    source_content_hash = work_order_source_content_hash(
        tenant_id=tenant_id,
        work_order_id=work_order_id,
        incident_id=incident_id,
        completion_id=completion_id,
        verification_id=verification_id,
        aggregate_version=aggregate_version,
    )
    return EventEnvelope(
        event_id=f"event-{uuid4().hex}",
        event_type="work_order.closed",
        event_version=1,
        tenant_id=tenant_id,
        aggregate_type="work_order",
        aggregate_id=work_order_id,
        aggregate_version=aggregate_version,
        sequence=aggregate_version,
        occurred_at=occurred_at,
        published_at=None,
        producer=PRODUCER,
        trace_id=trace_id or request_id,
        correlation_id=incident_id,
        causation_id=request_id,
        data_classification="CONFIDENTIAL",
        payload=WorkOrderClosedPayload(
            work_order_id=work_order_id,
            incident_id=incident_id,
            completion_id=completion_id,
            verification_id=verification_id,
            source_content_hash=source_content_hash,
        ),
    )


def work_order_source_content_hash(
    *,
    tenant_id: str,
    work_order_id: str,
    incident_id: str,
    completion_id: str,
    verification_id: str,
    aggregate_version: int,
) -> str:
    source_identity = {
        "aggregate_version": aggregate_version,
        "completion_id": completion_id,
        "incident_id": incident_id,
        "tenant_id": tenant_id,
        "verification_id": verification_id,
        "work_order_id": work_order_id,
    }
    return "sha256:" + sha256(
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
