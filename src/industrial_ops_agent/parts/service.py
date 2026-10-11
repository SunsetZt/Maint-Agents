"""Aggregate authorized EAM component facts with live WMS availability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from industrial_ops_agent.application.assets import AssetQueryService
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.tools.contracts import EnterpriseToolError
from industrial_ops_agent.tools.gateway import ToolGateway, ToolRateLimitExceeded


@dataclass(frozen=True, slots=True)
class PartComponentView:
    component_id: str
    part_number: str
    part_name: str
    installed_quantity: int
    component_status: str
    source: str
    source_record_id: str
    as_of: datetime | None
    freshness: str


@dataclass(frozen=True, slots=True)
class PartAvailabilityView:
    part_number: str
    available_quantity: int
    stock_status: str
    compatibility_evidence: str
    source: str
    source_record_id: str
    as_of: str
    authority_owner: str
    field_sources: dict[str, str]
    tool_call_id: str


@dataclass(frozen=True, slots=True)
class PartsLookupView:
    asset_id: str
    asset_display_name: str | None
    model_code: str | None
    serial_number: str | None
    site_id: str | None
    site_name: str | None
    components: tuple[PartComponentView, ...]
    availability: PartAvailabilityView | None
    availability_failure: str | None


class PartsLookupService:
    def __init__(
        self,
        asset_query: AssetQueryService,
        tool_gateway: ToolGateway,
    ) -> None:
        self._asset_query = asset_query
        self._tool_gateway = tool_gateway

    def lookup(
        self,
        identity: IdentityContext,
        asset_id: str,
        *,
        request_id: str,
    ) -> PartsLookupView:
        asset = self._asset_query.get(identity, asset_id, request_id=request_id)
        components = tuple(
            PartComponentView(
                component_id=component.component_id,
                part_number=component.part_number,
                part_name=component.part_name,
                installed_quantity=component.quantity,
                component_status=component.status,
                source=component.source.source_system,
                source_record_id=component.source.source_record_id,
                as_of=component.source.as_of,
                freshness=component.source.freshness.value,
            )
            for component in asset.components
        )
        availability: PartAvailabilityView | None = None
        failure: str | None = None
        try:
            result = self._tool_gateway.invoke(
                identity,
                tool_id="parts.availability",
                version="1.0.0",
                parameters={"asset_id": asset_id},
                request_id=request_id,
            )
            if result.status != "SUCCEEDED" or result.data is None:
                failure = "tool_result_not_usable"
            else:
                availability = _availability_view(result.tool_call_id, result.data, components)
        except AuthorizationDenied:
            failure = "tool_authorization_denied"
        except EnterpriseToolError as exc:
            failure = exc.reason
        except ToolRateLimitExceeded:
            failure = "tool_rate_limit_exceeded"
        except (KeyError, TypeError, ValueError):
            failure = "inventory_fact_invalid"
        return PartsLookupView(
            asset_id=asset.summary.asset_id,
            asset_display_name=asset.summary.display_name,
            model_code=asset.summary.model_code,
            serial_number=asset.summary.serial_number,
            site_id=asset.summary.site_id,
            site_name=asset.summary.site_name,
            components=components,
            availability=availability,
            availability_failure=failure,
        )


def _availability_view(
    tool_call_id: str,
    data: dict[str, Any],
    components: tuple[PartComponentView, ...],
) -> PartAvailabilityView:
    authority = data["authority"]
    if not isinstance(authority, dict):
        raise TypeError
    field_sources = authority["field_sources"]
    if not isinstance(field_sources, dict):
        raise TypeError
    part_number = str(data["part_number"])
    available_quantity = int(data["available_quantity"])
    installed_part_numbers = {component.part_number for component in components}
    compatibility_evidence = (
        "INSTALLED_COMPONENT_MATCH"
        if part_number in installed_part_numbers
        else "REVIEW_REQUIRED"
    )
    return PartAvailabilityView(
        part_number=part_number,
        available_quantity=available_quantity,
        stock_status="IN_STOCK" if available_quantity > 0 else "OUT_OF_STOCK",
        compatibility_evidence=compatibility_evidence,
        source=str(data["source"]),
        source_record_id=str(data["source_record_id"]),
        as_of=str(data["as_of"]),
        authority_owner=str(authority["owner"]),
        field_sources={str(field): str(source) for field, source in field_sources.items()},
        tool_call_id=tool_call_id,
    )
