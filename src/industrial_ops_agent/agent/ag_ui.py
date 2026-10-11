"""Safe, one-to-one AG-UI projection of persistent Agent events.

The persistent project event remains the source of truth.  This adapter deliberately
emits one AG-UI event for every source sequence so ``Last-Event-ID`` can use the
database sequence without a compound cursor.  Application-specific facts use
``CUSTOM`` events; raw model state, tool arguments/results and memory content are
never copied through.
"""

from __future__ import annotations

from typing import Any, Final

from industrial_ops_agent.agent.events import AgentEvent

AG_UI_PROFILE: Final = "industrial-ops/ag-ui-events-v1"
INTERNAL_EVENT_SCHEMA_VERSION: Final = "industrial-agent-event/v1"

_START_EVENTS = frozenset(
    {"diagnosis.queued", "diagnosis.reanalysis_queued", "diagnosis.resume_queued"}
)
_FINISH_STATUS = {
    "diagnosis.completed": "COMPLETED",
    "diagnosis.needs_information": "NEEDS_INFORMATION",
    "diagnosis.cancelled": "CANCELLED",
    "diagnosis.expert_revision_completed": "COMPLETED",
    "diagnosis.expert_escalated": "ESCALATED",
}
_STEP_EVENTS = {
    "diagnosis.retrieval_started": ("STEP_STARTED", "knowledge_retrieval"),
    "diagnosis.retrieval_completed": ("STEP_FINISHED", "knowledge_retrieval"),
    "diagnosis.tools_started": ("STEP_STARTED", "enterprise_tools"),
    "diagnosis.tools_completed": ("STEP_FINISHED", "enterprise_tools"),
}


def adapt_agent_event(
    event: AgentEvent,
    *,
    thread_id: str,
    diagnosis_run_id: str | None = None,
) -> dict[str, Any]:
    """Convert a visible persistent event to the locked AG-UI event profile."""

    common = {
        "rawEvent": _source_reference(
            event,
            thread_id=thread_id,
            diagnosis_run_id=diagnosis_run_id or thread_id,
        )
    }
    source_type = event.event_type

    if source_type in _START_EVENTS:
        started: dict[str, Any] = {
            "type": "RUN_STARTED",
            "threadId": thread_id,
            "runId": event.agent_run_id,
            **common,
        }
        parent_run_id = event.payload.get("parent_agent_run_id")
        if isinstance(parent_run_id, str) and parent_run_id:
            started["parentRunId"] = parent_run_id
        return started

    if source_type in {"diagnosis.graph_node_started", "diagnosis.graph_node_completed"}:
        return {
            "type": ("STEP_STARTED" if source_type.endswith("_started") else "STEP_FINISHED"),
            "stepName": f"{event.payload.get('node', 'unknown')}:{event.payload.get('step', 0)}",
            "metadata": _selected(
                event.payload,
                "node",
                "step",
                "graph_version",
                "model_steps",
                "tool_calls",
                "tokens_used",
                "inference_request_id",
                "tool_id",
                "tool_call_id",
                "action",
                "next_node",
            ),
            **common,
        }

    if source_type in {"diagnosis.model_started", "diagnosis.model_attempt_failed"}:
        return _custom(
            source_type,
            _selected(
                event.payload,
                "inference_request_id",
                "node",
                "step",
                "model_steps",
                "attempt",
                "reason",
                "graph_version",
            ),
            common,
        )

    step = _STEP_EVENTS.get(source_type)
    if step is not None:
        event_type, step_name = step
        return {"type": event_type, "stepName": step_name, **common}

    if source_type in _FINISH_STATUS:
        finished: dict[str, Any] = {
            "type": "RUN_FINISHED",
            "threadId": thread_id,
            "runId": event.agent_run_id,
            "result": _finish_result(event, _FINISH_STATUS[source_type]),
            **common,
        }
        if source_type == "diagnosis.needs_information":
            finished["outcome"] = {
                "type": "interrupt",
                "interrupts": [
                    {
                        "id": event.event_id,
                        "reason": "input_required",
                        "message": "请补充并人工确认缺失的现场信息后继续诊断。",
                        "responseSchema": {
                            "type": "object",
                            "properties": {
                                "clarification": {
                                    "type": "string",
                                    "minLength": 2,
                                    "maxLength": 4_000,
                                }
                            },
                            "required": ["clarification"],
                            "additionalProperties": False,
                        },
                        "metadata": {
                            "profile": AG_UI_PROFILE,
                            "diagnosisRunId": diagnosis_run_id or thread_id,
                        },
                    }
                ],
            }
        else:
            finished["outcome"] = {"type": "success"}
        return finished

    if source_type == "diagnosis.failed":
        return {
            "type": "RUN_ERROR",
            "message": "诊断运行失败，请根据请求标识联系管理员或重新发起。",
            "code": _text(event.payload.get("reason"), default="diagnosis_failed")[:128],
            **common,
        }

    if source_type == "diagnosis.cancel_requested":
        return _custom(
            "industrial.run.cancel_requested",
            {"status": "CANCEL_REQUESTED"},
            common,
        )

    if source_type == "diagnosis.input_required_state":
        return {
            "type": "STATE_SNAPSHOT",
            "snapshot": {
                "phase": "input_required",
                **_selected(event.payload, "missing_information", "stop_reason"),
            },
            **common,
        }

    if source_type == "diagnosis.messages_snapshot":
        return {
            "type": "MESSAGES_SNAPSHOT",
            "messages": [
                {
                    "id": _text(event.payload.get("message_id"), default=event.event_id),
                    "role": "assistant",
                    "content": _text(
                        event.payload.get("content"),
                        default="需要补充现场信息后才能继续诊断。",
                    ),
                }
            ],
            **common,
        }

    if source_type == "diagnosis.text_started":
        return {
            "type": "TEXT_MESSAGE_START",
            "messageId": _text(event.payload.get("message_id"), default=event.event_id),
            "role": "assistant",
            **common,
        }

    if source_type == "diagnosis.text_delta":
        delta = event.payload.get("delta")
        if not isinstance(delta, str) or not delta:
            return _custom(
                "industrial.event.invalid",
                {"sourceType": source_type, "reason": "empty_text_delta"},
                common,
            )
        return {
            "type": "TEXT_MESSAGE_CONTENT",
            "messageId": _text(event.payload.get("message_id"), default=event.event_id),
            "delta": delta,
            **common,
        }

    if source_type == "diagnosis.text_ended":
        return {
            "type": "TEXT_MESSAGE_END",
            "messageId": _text(event.payload.get("message_id"), default=event.event_id),
            **common,
        }

    if source_type == "diagnosis.citations_added":
        return _custom(
            "industrial.citation.added",
            _selected(event.payload, "citation_ids"),
            common,
        )

    if source_type == "diagnosis.tool_completed":
        return _custom(
            "industrial.tool.completed",
            _selected(
                event.payload,
                "tool_id",
                "tool_call_id",
                "source",
                "source_record_id",
                "as_of",
                "authority",
            ),
            common,
        )

    if source_type == "diagnosis.tool_failed":
        return _custom(
            "industrial.tool.failed",
            _selected(event.payload, "tool_id", "reason"),
            common,
        )

    if source_type == "diagnosis.tools_unavailable":
        return _custom(
            "industrial.tools.unavailable",
            _selected(event.payload, "tool_id", "reason"),
            common,
        )

    if source_type == "diagnosis.memories_selected":
        return _custom(
            "industrial.memory.selection",
            _selected(
                event.payload,
                "memory_count",
                "truncated_count",
                "policy_version",
            ),
            common,
        )

    if source_type == "diagnosis.report_ready":
        return {
            "type": "STATE_SNAPSHOT",
            "snapshot": {
                "phase": "report_ready",
                **_selected(event.payload, "citation_ids", "confidence"),
            },
            **common,
        }

    if source_type in {
        "diagnosis.ai_draft_ready_for_expert",
        "diagnosis.expert_takeover_requested",
    }:
        return _custom(
            "industrial.approval.required",
            {
                "kind": (
                    "expert_takeover"
                    if source_type == "diagnosis.expert_takeover_requested"
                    else "expert_review"
                ),
                **_selected(
                    event.payload,
                    "diagnosis_run_id",
                    "intervention_id",
                    "report_event_sequence",
                    "missing_information",
                ),
            },
            common,
        )

    if source_type == "agent.user_input.confirmed":
        return _custom(
            "industrial.user_input.confirmed",
            _selected(
                event.payload,
                "source_type",
                "source_id",
                "session_id",
                "language",
                "asr_release_id",
            ),
            common,
        )

    if source_type in {"diagnosis.model_completed", "diagnosis.model_unavailable"}:
        return _custom(
            "industrial.model.completed",
            {
                "degraded": source_type == "diagnosis.model_unavailable",
                **_selected(
                    event.payload,
                    "finish_reason",
                    "reason",
                    "replayed",
                    "resolved_release_id",
                    "required_release_id",
                    "inference_request_ids",
                    "graph_version",
                    "steps",
                ),
            },
            common,
        )

    return _custom(
        "industrial.agent.event",
        {"sourceType": source_type},
        common,
    )


def _finish_result(event: AgentEvent, status: str) -> dict[str, Any]:
    return {
        "status": status,
        **_selected(
            event.payload,
            "diagnosis_run_id",
            "report_event_sequence",
            "stop_reason",
            "missing_information",
        ),
    }


def _source_reference(
    event: AgentEvent,
    *,
    thread_id: str,
    diagnosis_run_id: str,
) -> dict[str, Any]:
    """Return correlation metadata, never the unfiltered source payload."""

    return {
        "profile": AG_UI_PROFILE,
        "schemaVersion": INTERNAL_EVENT_SCHEMA_VERSION,
        "sourceEventId": event.event_id,
        "sourceType": event.event_type,
        "diagnosisRunId": diagnosis_run_id,
        "agentRunId": event.agent_run_id,
        "threadId": thread_id,
        "sequence": event.sequence,
        "stateVersion": event.sequence,
        "occurredAt": event.occurred_at.isoformat(),
    }


def _custom(
    name: str,
    value: dict[str, Any],
    common: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {"type": "CUSTOM", "name": name, "value": value, **common}


def _selected(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload}


def _text(value: object, *, default: str) -> str:
    return value if isinstance(value, str) else default
