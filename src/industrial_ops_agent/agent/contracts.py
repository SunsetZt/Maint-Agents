"""Infrastructure-independent diagnosis graph versions and output contracts."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from industrial_ops_agent.agent.models import DiagnosisReport

DIAGNOSIS_GRAPH_VERSION = "diagnosis-graph-v2"
DIAGNOSIS_AGENT_PROMPT_ID = "industrial-diagnosis-agent-v2"
DIAGNOSIS_STEP_SCHEMA = "industrial_diagnosis_step_v2"
DIAGNOSIS_READONLY_TOOLS = (
    "asset.get",
    "warranty.get",
    "parts.availability",
    "schedule.availability",
    "work_orders.history",
)
DIAGNOSIS_TOOL_FIELDS = {
    "asset.get": ("model_code", "lifecycle_status"),
    "warranty.get": ("coverage", "valid"),
    "parts.availability": ("part_number", "available_quantity"),
    "schedule.availability": (
        "site_id",
        "next_available_start",
        "next_available_end",
        "available_technician_count",
    ),
    "work_orders.history": ("recent_count", "latest_resolution"),
}


def report_schema(citation_ids: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "status",
            "conclusion",
            "citation_ids",
            "contradictions",
            "missing_information",
            "next_checks",
            "confidence",
        ],
        "properties": {
            "status": {"enum": ["COMPLETED", "NEEDS_INFORMATION"]},
            "conclusion": {"type": ["string", "null"], "maxLength": 4000},
            "citation_ids": {
                "type": "array",
                "items": {"type": "string", **({"enum": citation_ids} if citation_ids else {})},
                "uniqueItems": True,
                "maxItems": min(6, len(citation_ids)),
            },
            "contradictions": _bounded_string_array(8),
            "missing_information": _bounded_string_array(8),
            "next_checks": {
                **_bounded_string_array(8),
                "minItems": 1,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "COMPLETED"}}},
                "then": {
                    "properties": {
                        "conclusion": {"type": "string", "minLength": 1},
                        "citation_ids": {"minItems": 1},
                    }
                },
                "else": {
                    "properties": {
                        "conclusion": {"type": "null"},
                        "missing_information": {"minItems": 1},
                    }
                },
            }
        ],
    }
    return schema


def step_schema(citation_ids: list[str], tools: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "query", "tool_id", "report"],
        "properties": {
            "action": {"enum": ["RETRIEVE", "CALL_TOOL", "FINAL", "NEEDS_INFORMATION"]},
            "query": {"type": ["string", "null"], "minLength": 2, "maxLength": 1500},
            "tool_id": {"enum": [None, *tools]},
            "report": {"anyOf": [report_schema(citation_ids), {"type": "null"}]},
        },
    }


def validate_step(value: dict[str, Any], citation_ids: list[str], tools: list[str]) -> None:
    Draft202012Validator(step_schema(citation_ids, tools)).validate(value)
    action = value["action"]
    query, tool_id, report = value["query"], value["tool_id"], value["report"]
    valid = (
        action == "RETRIEVE"
        and isinstance(query, str)
        and bool(query.strip())
        and tool_id is None
        and report is None
        or action == "CALL_TOOL"
        and tool_id in tools
        and query is None
        and report is None
        or action in {"FINAL", "NEEDS_INFORMATION"}
        and query is None
        and tool_id is None
        and isinstance(report, dict)
        and report["status"] == ("COMPLETED" if action == "FINAL" else "NEEDS_INFORMATION")
    )
    if not valid:
        raise ValueError("diagnosis_step_action_mismatch")


def report_from_dict(value: dict[str, Any]) -> DiagnosisReport:
    return DiagnosisReport(
        status=value["status"],
        conclusion=value["conclusion"],
        citations=tuple(value["citations"]),
        contradictions=tuple(value["contradictions"]),
        missing_information=tuple(value["missing_information"]),
        next_checks=tuple(value["next_checks"]),
        confidence=value["confidence"],
        stop_reason=value["stop_reason"],
    )


def budget_report(reason: str) -> dict[str, Any]:
    return DiagnosisReport(
        status="NEEDS_INFORMATION",
        conclusion=None,
        citations=(),
        contradictions=(),
        missing_information=("诊断执行预算已耗尽，尚不能形成完整结论。",),
        next_checks=("请由现场人员或专家复核已有依据，补充信息后重新发起诊断。",),
        confidence=0.0,
        stop_reason=reason,
    ).as_dict()


def _bounded_string_array(max_items: int) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "minLength": 1, "maxLength": 1000},
        "maxItems": max_items,
    }
