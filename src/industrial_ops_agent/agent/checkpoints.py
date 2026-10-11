"""Persistent graph checkpoints independent from Temporal workflow history."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.agent.events import (
    AgentEventStore,
    AgentRunNotFound,
    AgentRunWriteStopped,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import AgentCheckpointRecord, AgentRunRecord
from industrial_ops_agent.persistence.repositories import AgentCheckpointRepository
from industrial_ops_agent.persistence.tenant import TenantContext


class AgentCheckpointStore:
    def __init__(self, database: Database) -> None:
        self._database = database

    def save(
        self,
        context: TenantContext,
        agent_run_id: str,
        *,
        sequence: int,
        graph_version: str,
        state: dict[str, Any],
    ) -> None:
        with self._database.transaction(context) as session:
            self.save_in_session(
                session,
                context,
                agent_run_id,
                sequence=sequence,
                graph_version=graph_version,
                state=state,
            )

    @staticmethod
    def save_in_session(
        session: Session,
        context: TenantContext,
        agent_run_id: str,
        *,
        sequence: int,
        graph_version: str,
        state: dict[str, Any],
    ) -> None:
        canonical = json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        now = datetime.now(UTC)
        AgentCheckpointRepository(session, context).add(
            AgentCheckpointRecord(
                checkpoint_id=f"checkpoint-{uuid4().hex}",
                tenant_id=context.tenant_id,
                agent_run_id=agent_run_id,
                sequence=sequence,
                graph_version=graph_version,
                state=state,
                state_checksum=sha256(canonical.encode()).hexdigest(),
                created_at=now,
                updated_at=now,
            )
        )

    def latest(self, context: TenantContext, agent_run_id: str) -> dict[str, Any] | None:
        with self._database.transaction(context) as session:
            record = AgentCheckpointRepository(session, context).latest(agent_run_id)
            return None if record is None else record.state

    def load_graph(
        self,
        context: TenantContext,
        agent_run_id: str,
        *,
        graph_version: str,
    ) -> dict[str, Any] | None:
        with self._database.transaction(context) as session:
            record = AgentCheckpointRepository(session, context).latest(agent_run_id)
            if record is None:
                return None
            encoded = json.dumps(
                record.state, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
            if (
                record.graph_version != graph_version
                or sha256(encoded.encode()).hexdigest() != record.state_checksum
            ):
                raise ValueError("diagnosis_checkpoint_binding_invalid")
            state = dict(record.state)
            state["checkpoint_sequence"] = record.sequence
            return state

    def advance(
        self,
        context: TenantContext,
        agent_run_id: str,
        *,
        expected_sequence: int,
        graph_version: str,
        state: dict[str, Any],
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Commit one graph transition and its event atomically, using existing tables."""
        with self._database.transaction(context) as session:
            self.require_current(session, context, agent_run_id, expected_sequence)
            event = AgentEventStore.append_in_session(
                session,
                context,
                agent_run_id,
                event_type=event_type,
                payload=payload,
            )
            # Return exactly the JSON representation that a later retry will read.
            stored: dict[str, Any] = json.loads(json.dumps(state, ensure_ascii=False))
            stored["checkpoint_sequence"] = event.sequence
            self.save_in_session(
                session,
                context,
                agent_run_id,
                sequence=event.sequence,
                graph_version=graph_version,
                state=stored,
            )
            return stored

    @staticmethod
    def require_current(
        session: Session,
        context: TenantContext,
        agent_run_id: str,
        expected_sequence: int,
    ) -> None:
        run = session.scalar(
            select(AgentRunRecord)
            .where(
                AgentRunRecord.tenant_id == context.tenant_id,
                AgentRunRecord.agent_run_id == agent_run_id,
            )
            .with_for_update()
        )
        if run is None:
            raise AgentRunNotFound
        if run.status in {
            "CANCEL_REQUESTED",
            "CANCELLED",
            "COMPLETED",
            "FAILED",
            "NEEDS_INFORMATION",
            "ESCALATED",
        }:
            raise AgentRunWriteStopped
        record = AgentCheckpointRepository(session, context).latest(agent_run_id)
        if (record.sequence if record else 0) != expected_sequence:
            raise GraphCheckpointConflict("diagnosis_graph_concurrently_advanced")


class GraphCheckpointConflict(RuntimeError):
    """Another task attempt has already committed a newer node state."""
