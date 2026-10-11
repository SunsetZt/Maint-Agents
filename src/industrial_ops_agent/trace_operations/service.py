"""Tenant-scoped, prompt-free Agent and AI business trace projections."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from sqlalchemy import func, select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.maintenance_planning.review_isolation import record_disclosure
from industrial_ops_agent.model_gateway.context_manifest import context_manifest_is_valid
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AgentEventRecord,
    AgentRunRecord,
    DiagnosisRunRecord,
    ModelInferenceRecord,
    ToolCallRecord,
    WorkOrderRecord,
)


class TraceHealth(StrEnum):
    HEALTHY = "HEALTHY"
    RUNNING = "RUNNING"
    ATTENTION = "ATTENTION"
    FAILED = "FAILED"


class TraceEvidenceStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


TraceNodeCategory = Literal[
    "WORKFLOW",
    "AGENT",
    "AGENT_EVENT",
    "RETRIEVAL",
    "TOOL",
    "MODEL",
    "BUSINESS",
]


class TraceNotFound(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class TraceSummary:
    agent_run_id: str
    diagnosis_run_id: str
    incident_id: str
    workflow_id: str
    agent_status: str
    diagnosis_status: str
    stop_reason: str | None
    health: TraceHealth
    evidence_status: TraceEvidenceStatus
    started_at: datetime
    finished_at: datetime | None
    duration_ms: float | None
    latest_activity_at: datetime
    event_count: int
    tool_call_count: int
    inference_count: int
    failed_node_count: int
    prompt_tokens: int
    completion_tokens: int
    trace_ids: tuple[str, ...]
    model_release_ids: tuple[str, ...]
    request_classes: tuple[str, ...]
    work_order_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TraceNode:
    node_id: str
    parent_node_id: str | None
    category: TraceNodeCategory
    name: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    duration_ms: float | None
    attributes: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TraceDetail:
    summary: TraceSummary
    nodes: tuple[TraceNode, ...]
    missing_correlations: tuple[str, ...]
    privacy_notice: str


@dataclass(frozen=True, slots=True)
class TracePage:
    items: tuple[TraceSummary, ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class _TraceRecords:
    run: AgentRunRecord
    diagnosis: DiagnosisRunRecord
    events: tuple[AgentEventRecord, ...]
    tools: tuple[ToolCallRecord, ...]
    inferences: tuple[ModelInferenceRecord, ...]
    work_orders: tuple[WorkOrderRecord, ...]


class TraceOperationsService:
    """Builds a durable business trace without exposing model or customer content."""

    PROJECTION_VERSION = "agent-business-trace/v2"
    PRIVACY_NOTICE = (
        "仅展示结构化审计元数据；Prompt、模型输出、客户正文、检索内容、"
        "工具参数/结果和 Chain-of-Thought 均不会进入此投影。"
    )

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def list_traces(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> TracePage:
        self._require(identity, request_id, "agent-traces")
        with self._database.transaction(identity.tenant_context) as session:
            conditions = [AgentRunRecord.tenant_id == identity.tenant_id]
            if status is not None:
                conditions.append(AgentRunRecord.status == status)
            total = int(
                session.scalar(select(func.count()).select_from(AgentRunRecord).where(*conditions))
                or 0
            )
            runs = tuple(
                session.scalars(
                    select(AgentRunRecord)
                    .where(*conditions)
                    .order_by(AgentRunRecord.created_at.desc(), AgentRunRecord.agent_run_id)
                    .limit(limit)
                    .offset(offset)
                )
            )
            records = _load_records(session, identity.tenant_id, runs)
            record_disclosure(
                session,
                identity,
                (run.diagnosis_run_id for run in runs),
                resource_kind="agent-trace-list",
                resource_id="agent-traces",
                request_id=request_id,
            )
        return TracePage(
            items=tuple(_summary(item) for item in records),
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_trace(
        self,
        identity: IdentityContext,
        agent_run_id: str,
        *,
        request_id: str,
    ) -> TraceDetail:
        self._require(identity, request_id, agent_run_id)
        with self._database.transaction(identity.tenant_context) as session:
            run = session.scalar(
                select(AgentRunRecord).where(
                    AgentRunRecord.tenant_id == identity.tenant_id,
                    AgentRunRecord.agent_run_id == agent_run_id,
                )
            )
            if run is None:
                raise TraceNotFound(agent_run_id)
            records = _load_records(session, identity.tenant_id, (run,))
            record_disclosure(
                session,
                identity,
                (run.diagnosis_run_id,),
                resource_kind="agent-trace",
                resource_id=agent_run_id,
                request_id=request_id,
            )
        if not records:
            raise TraceNotFound(agent_run_id)
        trace = records[0]
        missing = _missing_correlations(trace)
        return TraceDetail(
            summary=_summary(trace, missing=missing),
            nodes=_nodes(trace),
            missing_correlations=missing,
            privacy_notice=self.PRIVACY_NOTICE,
        )

    def _require(self, identity: IdentityContext, request_id: str, resource_id: str) -> None:
        self._authorizer.require(
            identity,
            Action.READ_OPERATIONS,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )


def _load_records(
    session: Any,
    tenant_id: str,
    runs: tuple[AgentRunRecord, ...],
) -> tuple[_TraceRecords, ...]:
    if not runs:
        return ()
    agent_ids = {item.agent_run_id for item in runs}
    diagnosis_ids = {item.diagnosis_run_id for item in runs}
    diagnoses = {
        item.diagnosis_run_id: item
        for item in session.scalars(
            select(DiagnosisRunRecord).where(
                DiagnosisRunRecord.tenant_id == tenant_id,
                DiagnosisRunRecord.diagnosis_run_id.in_(diagnosis_ids),
            )
        )
    }
    events = _group_by_agent(
        session.scalars(
            select(AgentEventRecord)
            .where(
                AgentEventRecord.tenant_id == tenant_id,
                AgentEventRecord.agent_run_id.in_(agent_ids),
            )
            .order_by(AgentEventRecord.occurred_at, AgentEventRecord.sequence)
        )
    )
    tools = _group_by_agent(
        session.scalars(
            select(ToolCallRecord)
            .where(
                ToolCallRecord.tenant_id == tenant_id,
                ToolCallRecord.agent_run_id.in_(agent_ids),
            )
            .order_by(ToolCallRecord.created_at, ToolCallRecord.tool_call_id)
        )
    )
    inferences = _group_by_agent(
        session.scalars(
            select(ModelInferenceRecord)
            .where(
                ModelInferenceRecord.tenant_id == tenant_id,
                ModelInferenceRecord.agent_run_id.in_(agent_ids),
            )
            .order_by(ModelInferenceRecord.created_at, ModelInferenceRecord.inference_request_id)
        )
    )
    incident_ids = {item.incident_id for item in diagnoses.values()}
    work_orders_by_incident: dict[str, list[WorkOrderRecord]] = {}
    if incident_ids:
        for item in session.scalars(
            select(WorkOrderRecord)
            .where(
                WorkOrderRecord.tenant_id == tenant_id,
                WorkOrderRecord.incident_id.in_(incident_ids),
            )
            .order_by(WorkOrderRecord.created_at, WorkOrderRecord.work_order_id)
        ):
            work_orders_by_incident.setdefault(item.incident_id, []).append(item)
    result: list[_TraceRecords] = []
    for run in runs:
        diagnosis = diagnoses.get(run.diagnosis_run_id)
        if diagnosis is None:
            continue
        result.append(
            _TraceRecords(
                run=run,
                diagnosis=diagnosis,
                events=tuple(events.get(run.agent_run_id, ())),
                tools=tuple(tools.get(run.agent_run_id, ())),
                inferences=tuple(inferences.get(run.agent_run_id, ())),
                work_orders=tuple(work_orders_by_incident.get(diagnosis.incident_id, ())),
            )
        )
    return tuple(result)


def _group_by_agent(records: Any) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for item in records:
        agent_run_id = item.agent_run_id
        if agent_run_id is not None:
            grouped.setdefault(agent_run_id, []).append(item)
    return grouped


def _summary(
    records: _TraceRecords,
    *,
    missing: tuple[str, ...] | None = None,
) -> TraceSummary:
    missing_correlations = missing if missing is not None else _missing_correlations(records)
    started_at = min(
        [
            _utc(records.run.created_at),
            _utc(records.diagnosis.created_at),
            *(_utc(item.occurred_at) for item in records.events),
            *(_utc(item.created_at) for item in records.tools),
            *(_utc(item.created_at) for item in records.inferences),
        ]
    )
    terminal = records.run.status in {
        "COMPLETED",
        "FAILED",
        "NEEDS_INFORMATION",
        "WAITING_EXPERT",
        "CANCELLED",
    }
    activity_times = [
        _utc(records.run.updated_at),
        _utc(records.diagnosis.updated_at),
        *(_utc(item.occurred_at) for item in records.events),
        *(_utc(item.updated_at) for item in records.tools),
        *(_utc(item.completed_at or item.updated_at) for item in records.inferences),
    ]
    latest = max(activity_times)
    finished_at = latest if terminal else None
    failed_nodes = sum(item.status == "FAILED" for item in records.tools) + sum(
        item.status == "FAILED" for item in records.inferences
    )
    return TraceSummary(
        agent_run_id=records.run.agent_run_id,
        diagnosis_run_id=records.diagnosis.diagnosis_run_id,
        incident_id=records.diagnosis.incident_id,
        workflow_id=records.diagnosis.workflow_id,
        agent_status=records.run.status,
        diagnosis_status=records.diagnosis.status,
        stop_reason=records.diagnosis.stop_reason,
        health=_health(records, failed_nodes, bool(missing_correlations)),
        evidence_status=(
            TraceEvidenceStatus.PARTIAL if missing_correlations else TraceEvidenceStatus.COMPLETE
        ),
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=_duration_ms(started_at, finished_at),
        latest_activity_at=latest,
        event_count=len(records.events),
        tool_call_count=len(records.tools),
        inference_count=len(records.inferences),
        failed_node_count=failed_nodes,
        prompt_tokens=sum(item.usage_prompt_tokens for item in records.inferences),
        completion_tokens=sum(item.usage_completion_tokens for item in records.inferences),
        trace_ids=tuple(sorted({item.trace_id for item in records.inferences if item.trace_id})),
        model_release_ids=tuple(sorted({item.resolved_release_id for item in records.inferences})),
        request_classes=tuple(sorted({item.request_class for item in records.inferences})),
        work_order_ids=tuple(item.work_order_id for item in records.work_orders),
    )


def _health(records: _TraceRecords, failed_nodes: int, partial: bool) -> TraceHealth:
    if records.run.status in {"FAILED", "CANCELLED"} or records.diagnosis.status == "FAILED":
        return TraceHealth.FAILED
    if records.run.status in {"QUEUED", "PENDING", "RUNNING"}:
        return TraceHealth.RUNNING
    if (
        records.run.status in {"NEEDS_INFORMATION", "WAITING_EXPERT"}
        or records.diagnosis.status in {"NEEDS_INFORMATION", "WAITING_EXPERT"}
        or failed_nodes > 0
        or partial
    ):
        return TraceHealth.ATTENTION
    return TraceHealth.HEALTHY


def _missing_correlations(records: _TraceRecords) -> tuple[str, ...]:
    missing: list[str] = []
    if not records.events:
        missing.append("agent_events_missing")
    event_types = {item.event_type for item in records.events}
    if (
        "diagnosis.retrieval_started" in event_types
        and "diagnosis.retrieval_completed" not in event_types
    ):
        missing.append("retrieval_completion_missing")
    referenced_inferences = {
        str(item.payload["inference_request_id"])
        for item in records.events
        if item.event_type in {"diagnosis.model_completed", "diagnosis.model_unavailable"}
        and isinstance(item.payload.get("inference_request_id"), str)
    }
    persisted_inferences = {item.inference_request_id for item in records.inferences}
    if referenced_inferences - persisted_inferences:
        missing.append("model_inference_correlation_missing")
    referenced_tools: set[str] = set()
    for event in records.events:
        value = event.payload.get("tool_call_id")
        if isinstance(value, str):
            referenced_tools.add(value)
        values = event.payload.get("successful_tool_call_ids")
        if isinstance(values, list):
            referenced_tools.update(str(item) for item in values if isinstance(item, str))
    if referenced_tools - {item.tool_call_id for item in records.tools}:
        missing.append("tool_call_correlation_missing")
    if records.inferences and not any(item.trace_id for item in records.inferences):
        missing.append("telemetry_trace_id_missing")
    if records.inferences and any(
        item.context_hash is None
        or item.context_manifest_json.get("context_hash") != item.context_hash
        or item.context_manifest_json.get("trace_id") != item.trace_id
        or not context_manifest_is_valid(item.context_manifest_json)
        for item in records.inferences
    ):
        missing.append("context_manifest_missing")
    return tuple(missing)


def _nodes(records: _TraceRecords) -> tuple[TraceNode, ...]:
    summary = _summary(records)
    workflow_node_id = f"workflow:{records.diagnosis.workflow_id}"
    agent_node_id = f"agent:{records.run.agent_run_id}"
    nodes: list[TraceNode] = [
        TraceNode(
            node_id=workflow_node_id,
            parent_node_id=None,
            category="WORKFLOW",
            name="Temporal 诊断工作流",
            status=records.diagnosis.status,
            started_at=summary.started_at,
            finished_at=summary.finished_at,
            duration_ms=summary.duration_ms,
            attributes={
                "workflow_id": records.diagnosis.workflow_id,
                "diagnosis_run_id": records.diagnosis.diagnosis_run_id,
                "incident_id": records.diagnosis.incident_id,
                "stop_reason": records.diagnosis.stop_reason,
            },
        ),
        TraceNode(
            node_id=agent_node_id,
            parent_node_id=workflow_node_id,
            category="AGENT",
            name="诊断 Agent Run",
            status=records.run.status,
            started_at=_utc(records.run.created_at),
            finished_at=summary.finished_at,
            duration_ms=_duration_ms(_utc(records.run.created_at), summary.finished_at),
            attributes={
                "agent_run_id": records.run.agent_run_id,
                "event_count": len(records.events),
                "last_sequence": records.run.last_sequence,
            },
        ),
    ]
    nodes.extend(_event_nodes(records.events, agent_node_id))
    nodes.extend(_retrieval_nodes(records.events, agent_node_id))
    nodes.extend(_tool_nodes(records.tools, agent_node_id))
    nodes.extend(_inference_nodes(records.inferences, agent_node_id))
    nodes.extend(_work_order_nodes(records.work_orders, workflow_node_id))
    return tuple(
        sorted(
            nodes,
            key=lambda item: (
                item.started_at,
                _category_order(item.category),
                item.node_id,
            ),
        )
    )


def _event_nodes(events: tuple[AgentEventRecord, ...], parent_id: str) -> list[TraceNode]:
    hidden = {"diagnosis.retrieval_started", "diagnosis.retrieval_completed"}
    return [
        TraceNode(
            node_id=f"event:{event.event_id}",
            parent_node_id=parent_id,
            category="AGENT_EVENT",
            name=event.event_type,
            status=_event_status(event.event_type),
            started_at=_utc(event.occurred_at),
            finished_at=_utc(event.occurred_at),
            duration_ms=0.0,
            attributes=_safe_event_attributes(event),
        )
        for event in events
        if event.event_type not in hidden
    ]


def _retrieval_nodes(events: tuple[AgentEventRecord, ...], parent_id: str) -> list[TraceNode]:
    starts = [item for item in events if item.event_type == "diagnosis.retrieval_started"]
    completions = [item for item in events if item.event_type == "diagnosis.retrieval_completed"]
    if not starts and not completions:
        return []
    start = starts[0] if starts else completions[0]
    completed = completions[-1] if completions else None
    attributes: dict[str, Any] = {}
    index_release = start.payload.get("index_release_id")
    if isinstance(index_release, str):
        attributes["index_release_id"] = index_release
    if completed is not None:
        evidence_count = completed.payload.get("evidence_count")
        if isinstance(evidence_count, int):
            attributes["evidence_count"] = evidence_count
        guard = completed.payload.get("retrieval_guard")
        if isinstance(guard, dict):
            for key in (
                "filter_stage",
                "sparse_candidate_count",
                "vector_candidate_count",
                "fused_candidate_count",
                "returned_count",
                "unauthorized_candidate_count",
            ):
                value = guard.get(key)
                if isinstance(value, str | int):
                    attributes[key] = value
            dimensions = guard.get("policy_dimensions")
            if isinstance(dimensions, list):
                attributes["policy_dimensions"] = [
                    str(item)[:64] for item in dimensions if isinstance(item, str)
                ][:16]
    finished_at = _utc(completed.occurred_at) if completed is not None else None
    return [
        TraceNode(
            node_id=f"retrieval:{start.event_id}",
            parent_node_id=parent_id,
            category="RETRIEVAL",
            name="授权过滤与混合检索",
            status="SUCCEEDED" if completed is not None else "RUNNING",
            started_at=_utc(start.occurred_at),
            finished_at=finished_at,
            duration_ms=_duration_ms(_utc(start.occurred_at), finished_at),
            attributes=attributes,
        )
    ]


def _tool_nodes(tools: tuple[ToolCallRecord, ...], parent_id: str) -> list[TraceNode]:
    return [
        TraceNode(
            node_id=f"tool:{item.tool_call_id}",
            parent_node_id=parent_id,
            category="TOOL",
            name=item.tool_id,
            status=item.status,
            started_at=_utc(item.created_at),
            finished_at=_utc(item.updated_at),
            duration_ms=_duration_ms(_utc(item.created_at), _utc(item.updated_at)),
            attributes={
                "tool_call_id": item.tool_call_id,
                "tool_version": item.tool_version,
                "risk_tier": item.risk_tier,
            },
        )
        for item in tools
    ]


def _inference_nodes(
    inferences: tuple[ModelInferenceRecord, ...], parent_id: str
) -> list[TraceNode]:
    result: list[TraceNode] = []
    for item in inferences:
        finished = _utc(item.completed_at) if item.completed_at is not None else None
        latency = {
            str(key)[:64]: round(float(value), 3)
            for key, value in item.latency_breakdown_json.items()
            if isinstance(value, int | float) and math.isfinite(value) and value >= 0
        }
        result.append(
            TraceNode(
                node_id=f"model:{item.inference_request_id}",
                parent_node_id=parent_id,
                category="MODEL",
                name=f"{item.model_alias} / {item.resolved_release_id}",
                status=item.status,
                started_at=_utc(item.created_at),
                finished_at=finished,
                duration_ms=_duration_ms(_utc(item.created_at), finished),
                attributes={
                    "inference_request_id": item.inference_request_id,
                    "trace_id": item.trace_id,
                    "request_class": item.request_class,
                    "prompt_bundle_id": item.prompt_bundle_id,
                    "index_release_id": item.index_release_id,
                    "prompt_tokens": item.usage_prompt_tokens,
                    "completion_tokens": item.usage_completion_tokens,
                    "finish_reason": item.finish_reason,
                    "safety_decision": item.safety_decision,
                    "guardrail_policy_version": item.guardrail_policy_version,
                    "guardrail_finding_count": len(item.guardrail_findings_json),
                    "degraded_from": item.degraded_from,
                    "failure_reason": item.failure_reason,
                    "latency_breakdown_ms": latency,
                    **_safe_context_attributes(item),
                },
            )
        )
    return result


def _safe_context_attributes(item: ModelInferenceRecord) -> dict[str, Any]:
    manifest = item.context_manifest_json
    if (
        not isinstance(manifest, dict)
        or item.context_hash is None
        or manifest.get("context_hash") != item.context_hash
        or manifest.get("trace_id") != item.trace_id
        or not context_manifest_is_valid(manifest)
    ):
        return {
            "context_manifest_version": None,
            "context_hash": None,
        }
    return {
        "context_manifest_version": _safe_scalar(manifest.get("schema_version")),
        "context_hash": item.context_hash,
        "context_token_budget": _safe_nonnegative_integer(manifest.get("token_budget")),
        "context_evidence_ids": _safe_reference_ids(manifest.get("evidence")),
        "context_memory_ids": _safe_reference_ids(manifest.get("memories")),
        "context_tool_result_ids": _safe_reference_ids(manifest.get("tool_results")),
        "context_truncated_items": _safe_truncations(manifest.get("truncated_items")),
    }


def _safe_reference_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[:64]:
        if not isinstance(item, dict):
            continue
        reference_id = item.get("id")
        if isinstance(reference_id, str):
            result.append(reference_id[:255])
    return result


def _safe_truncations(value: object) -> list[dict[str, str | int]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str | int]] = []
    for item in value[:64]:
        if not isinstance(item, dict):
            continue
        source_type = _safe_scalar(item.get("source_type"))
        source_id = _safe_scalar(item.get("source_id"))
        reason = _safe_scalar(item.get("reason"))
        original = _safe_nonnegative_integer(item.get("original_units"))
        included = _safe_nonnegative_integer(item.get("included_units"))
        if (
            source_type is not None
            and source_id is not None
            and reason is not None
            and original is not None
            and included is not None
        ):
            result.append(
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "reason": reason,
                    "original_units": original,
                    "included_units": included,
                }
            )
    return result


def _safe_scalar(value: object) -> str | None:
    return value[:255] if isinstance(value, str) else None


def _safe_nonnegative_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _work_order_nodes(work_orders: tuple[WorkOrderRecord, ...], parent_id: str) -> list[TraceNode]:
    return [
        TraceNode(
            node_id=f"work-order:{item.work_order_id}",
            parent_node_id=parent_id,
            category="BUSINESS",
            name=f"工单 {item.work_order_id}",
            status=item.status,
            started_at=_utc(item.created_at),
            finished_at=_utc(item.updated_at),
            duration_ms=_duration_ms(_utc(item.created_at), _utc(item.updated_at)),
            attributes={
                "work_order_id": item.work_order_id,
                "priority": item.priority,
                "sla_due_at": _utc(item.sla_due_at).isoformat() if item.sla_due_at else None,
            },
        )
        for item in work_orders
    ]


def _safe_event_attributes(event: AgentEventRecord) -> dict[str, Any]:
    payload = event.payload
    attributes: dict[str, Any] = {
        "sequence": event.sequence,
        "visibility": event.visibility,
    }
    scalar_keys = {
        "diagnosis_run_id",
        "source_diagnosis_run_id",
        "inference_request_id",
        "resolved_release_id",
        "required_release_id",
        "finish_reason",
        "context_hash",
        "degraded",
        "replayed",
        "confidence",
        "tool_id",
        "tool_call_id",
        "max_tool_calls",
        "report_event_sequence",
    }
    for key in scalar_keys:
        value = payload.get(key)
        if isinstance(value, str | int | float | bool) or value is None:
            attributes[key] = value
    if event.event_type in {
        "diagnosis.model_unavailable",
        "diagnosis.tool_failed",
        "diagnosis.tools_unavailable",
    }:
        reason = payload.get("reason")
        if isinstance(reason, str):
            attributes["reason"] = reason
    if event.event_type in {
        "diagnosis.completed",
        "diagnosis.needs_information",
    }:
        stop_reason = payload.get("stop_reason")
        if isinstance(stop_reason, str):
            attributes["stop_reason"] = stop_reason
    count_keys = {
        "citation_ids": "citation_count",
        "tool_call_ids": "tool_call_count",
        "tool_failures": "tool_failure_count",
        "tool_ids": "selected_tool_count",
        "successful_tool_call_ids": "successful_tool_count",
        "failed_tool_ids": "failed_tool_count",
        "missing_information": "missing_information_count",
    }
    for source, target in count_keys.items():
        value = payload.get(source)
        if isinstance(value, list):
            attributes[target] = len(value)
    return attributes


def _event_status(event_type: str) -> str:
    if event_type.endswith(("failed", "unavailable")):
        return "FAILED"
    if event_type.endswith(("completed", "ready", "confirmed")):
        return "SUCCEEDED"
    if event_type.endswith(("queued", "started")):
        return "RUNNING"
    if event_type.endswith(("needs_information", "ready_for_expert")):
        return "ATTENTION"
    return "RECORDED"


def _duration_ms(started_at: datetime, finished_at: datetime | None) -> float | None:
    if finished_at is None:
        return None
    return round(max(0.0, (finished_at - started_at).total_seconds() * 1000), 3)


def _category_order(category: TraceNodeCategory) -> int:
    return {
        "WORKFLOW": 0,
        "AGENT": 1,
        "RETRIEVAL": 2,
        "TOOL": 3,
        "MODEL": 4,
        "AGENT_EVENT": 5,
        "BUSINESS": 6,
    }.get(category, 99)
