"""Synthetic enterprise read adapters with explicit source freshness metadata."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from industrial_ops_agent.tools.authority import attest_tool_fact, authoritative_source
from industrial_ops_agent.tools.contracts import EnterpriseToolError, ToolTransportBinding


class SyntheticReadonlyAdapter:
    """Small deterministic adapter standing in for EAM, warranty, WMS, and FSM APIs."""

    def invoke(self, tool_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        asset_id = str(parameters["asset_id"])
        now = datetime.now(UTC)
        as_of = now.isoformat()
        facts: dict[str, dict[str, Any]] = {
            "asset.get": {
                "model_code": "PUMP-X100",
                "lifecycle_status": "IN_SERVICE",
                "source_record_id": asset_id,
            },
            "warranty.get": {
                "coverage": "PARTS_AND_LABOR",
                "valid": True,
                "source_record_id": f"warranty-{asset_id}",
            },
            "parts.availability": {
                "part_number": "FILTER-X100",
                "available_quantity": 8,
                "source_record_id": f"stock-{asset_id}",
            },
            "schedule.availability": {
                "site_id": "site-demo-east",
                "next_available_start": (now + timedelta(hours=2)).isoformat(),
                "next_available_end": (now + timedelta(hours=6)).isoformat(),
                "available_technician_count": 3,
                "source_record_id": f"schedule-{asset_id}",
            },
            "work_orders.history": {
                "recent_count": 2,
                "latest_resolution": "Replaced inlet filter",
                "source_record_id": f"history-{asset_id}",
            },
            "work_order.draft": {
                "title": f"Inspect {asset_id}",
                "status": "DRAFT",
                "source_record_id": f"draft-{asset_id}",
            },
        }
        return attest_tool_fact(
            tool_id,
            {
                **facts[tool_id],
                "source": authoritative_source(tool_id, "synthetic"),
                "as_of": as_of,
            },
            profile="synthetic",
        )

    def transport_binding(self, tool_id: str) -> ToolTransportBinding:
        if tool_id not in {
            "asset.get",
            "warranty.get",
            "parts.availability",
            "schedule.availability",
            "work_orders.history",
            "work_order.draft",
        }:
            raise EnterpriseToolError("enterprise_tool_not_supported")
        return ToolTransportBinding(
            transport="LOCAL_SYNTHETIC",
            local_tool_id=tool_id,
            remote_tool_name=tool_id,
            server_id="compose-lite-synthetic-tools",
            server_version="1.0.0",
        )
