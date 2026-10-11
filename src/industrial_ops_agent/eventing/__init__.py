"""Versioned domain event contracts and transactional Outbox helpers."""

from industrial_ops_agent.eventing.envelope import (
    EventEnvelope,
    WorkOrderClosedPayload,
    build_work_order_closed_event,
    work_order_source_content_hash,
)

__all__ = [
    "EventEnvelope",
    "WorkOrderClosedPayload",
    "build_work_order_closed_event",
    "work_order_source_content_hash",
]
