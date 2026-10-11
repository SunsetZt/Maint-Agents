"""AST-002 authoritative-source declarations for enterprise tool facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from industrial_ops_agent.tools.contracts import EnterpriseToolError

SourceProfile = Literal["enterprise", "synthetic"]

AUTHORITY_CONTRACT_VERSION = "ast002-authoritative-facts-v1"


@dataclass(frozen=True, slots=True)
class ToolFactAuthority:
    """One owner and one source for every field returned by a governed tool."""

    domain: str
    owner: str
    fields: tuple[str, ...]
    enterprise_source: str
    synthetic_source: str

    def source_for(self, profile: SourceProfile) -> str:
        return self.enterprise_source if profile == "enterprise" else self.synthetic_source

    def declaration(self, profile: SourceProfile) -> dict[str, Any]:
        source = self.source_for(profile)
        return {
            "contract_version": AUTHORITY_CONTRACT_VERSION,
            "domain": self.domain,
            "owner": self.owner,
            "field_sources": {field: source for field in self.fields},
        }


TOOL_FACT_AUTHORITIES: Mapping[str, ToolFactAuthority] = {
    "asset.get": ToolFactAuthority(
        domain="asset",
        owner="enterprise-eam-cmdb",
        fields=("model_code", "lifecycle_status"),
        enterprise_source="enterprise-eam",
        synthetic_source="synthetic-eam",
    ),
    "warranty.get": ToolFactAuthority(
        domain="service_contract",
        owner="enterprise-contract-entitlement",
        fields=("coverage", "valid"),
        enterprise_source="enterprise-warranty",
        synthetic_source="synthetic-warranty",
    ),
    "parts.availability": ToolFactAuthority(
        domain="inventory",
        owner="enterprise-erp-wms",
        fields=("part_number", "available_quantity"),
        enterprise_source="enterprise-wms",
        synthetic_source="synthetic-wms",
    ),
    "schedule.availability": ToolFactAuthority(
        domain="schedule",
        owner="enterprise-field-service-scheduling",
        fields=(
            "site_id",
            "next_available_start",
            "next_available_end",
            "available_technician_count",
        ),
        enterprise_source="enterprise-fsm",
        synthetic_source="synthetic-fsm",
    ),
    "work_orders.history": ToolFactAuthority(
        domain="work_order",
        owner="enterprise-field-service-management",
        fields=("recent_count", "latest_resolution"),
        enterprise_source="enterprise-fsm",
        synthetic_source="synthetic-fsm",
    ),
    "work_order.draft": ToolFactAuthority(
        domain="work_order_template",
        owner="enterprise-field-service-management",
        fields=("title", "status"),
        enterprise_source="enterprise-fsm",
        synthetic_source="synthetic-fsm",
    ),
}


def attest_tool_fact(
    tool_id: str,
    payload: Mapping[str, Any],
    *,
    profile: SourceProfile,
) -> dict[str, Any]:
    """Validate the adapter result and attach its immutable authority declaration."""

    authority = _authority(tool_id)
    expected_source = authority.source_for(profile)
    if payload.get("source") != expected_source:
        raise EnterpriseToolError("enterprise_tool_authority_source_mismatch")
    _validate_fact_metadata(payload)
    if any(field not in payload for field in authority.fields):
        raise EnterpriseToolError("enterprise_tool_authoritative_fields_missing")
    return {**payload, "authority": authority.declaration(profile)}


def authoritative_source(tool_id: str, profile: SourceProfile) -> str:
    """Return the single configured source for a tool in one runtime profile."""

    return _authority(tool_id).source_for(profile)


def validate_attested_tool_fact(tool_id: str, payload: Mapping[str, Any]) -> None:
    """Fail closed at the Gateway if an adapter bypasses or alters the declaration."""

    authority = _authority(tool_id)
    source = payload.get("source")
    if source == authority.enterprise_source:
        profile: SourceProfile = "enterprise"
    elif source == authority.synthetic_source:
        profile = "synthetic"
    else:
        raise EnterpriseToolError("enterprise_tool_authority_source_mismatch")
    _validate_fact_metadata(payload)
    if any(field not in payload for field in authority.fields):
        raise EnterpriseToolError("enterprise_tool_authoritative_fields_missing")
    if payload.get("authority") != authority.declaration(profile):
        raise EnterpriseToolError("enterprise_tool_authority_declaration_invalid")


def _authority(tool_id: str) -> ToolFactAuthority:
    try:
        return TOOL_FACT_AUTHORITIES[tool_id]
    except KeyError as exc:
        raise EnterpriseToolError("enterprise_tool_authority_contract_missing") from exc


def _validate_fact_metadata(payload: Mapping[str, Any]) -> None:
    source_record_id = payload.get("source_record_id")
    if not isinstance(source_record_id, str) or not 1 <= len(source_record_id) <= 255:
        raise EnterpriseToolError("enterprise_tool_source_record_invalid")
    raw_as_of = payload.get("as_of")
    try:
        as_of = (
            raw_as_of
            if isinstance(raw_as_of, datetime)
            else datetime.fromisoformat(str(raw_as_of).replace("Z", "+00:00"))
        )
    except (TypeError, ValueError) as exc:
        raise EnterpriseToolError("enterprise_tool_as_of_invalid") from exc
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise EnterpriseToolError("enterprise_tool_as_of_invalid")
