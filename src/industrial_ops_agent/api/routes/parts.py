"""Authorized spare-parts compatibility and live inventory lookup API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_tool_gateway,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.application.assets import AssetQueryService, ResourceNotVisible
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.parts.service import (
    PartAvailabilityView,
    PartComponentView,
    PartsLookupService,
    PartsLookupView,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.tools.gateway import ToolGateway

router = APIRouter(prefix="/parts", tags=["parts"])


class PartComponentResponse(BaseModel):
    component_id: str
    part_number: str
    part_name: str
    installed_quantity: int
    component_status: str
    source: str
    source_record_id: str
    as_of: str | None
    freshness: str


class PartAvailabilityResponse(BaseModel):
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


class PartReservationPolicyResponse(BaseModel):
    risk_tier: str
    direct_reservation_allowed: bool
    required_flow: str


class PartsLookupResponse(BaseModel):
    asset_id: str
    asset_display_name: str | None
    model_code: str | None
    serial_number: str | None
    site_id: str | None
    site_name: str | None
    components: list[PartComponentResponse]
    availability: PartAvailabilityResponse | None
    availability_failure: str | None
    reservation_policy: PartReservationPolicyResponse


class PartsLookupEnvelope(BaseModel):
    data: PartsLookupResponse
    meta: dict[str, str]


@router.get("/lookup", response_model=PartsLookupEnvelope)
async def lookup_parts(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    tool_gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    asset_id: Annotated[str, Query(min_length=1, max_length=128)],
) -> PartsLookupEnvelope:
    try:
        lookup = PartsLookupService(
            AssetQueryService(database, authorizer),
            tool_gateway,
        ).lookup(identity, asset_id, request_id=request.state.request_id)
    except ResourceNotVisible as exc:
        raise AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        ) from exc
    return PartsLookupEnvelope(
        data=_lookup_response(lookup),
        meta={"request_id": request.state.request_id},
    )


def _lookup_response(lookup: PartsLookupView) -> PartsLookupResponse:
    return PartsLookupResponse(
        asset_id=lookup.asset_id,
        asset_display_name=lookup.asset_display_name,
        model_code=lookup.model_code,
        serial_number=lookup.serial_number,
        site_id=lookup.site_id,
        site_name=lookup.site_name,
        components=[_component_response(component) for component in lookup.components],
        availability=(
            _availability_response(lookup.availability)
            if lookup.availability is not None
            else None
        ),
        availability_failure=lookup.availability_failure,
        reservation_policy=PartReservationPolicyResponse(
            risk_tier="T2",
            direct_reservation_allowed=False,
            required_flow="diagnosis_proposal_approval_execute",
        ),
    )


def _component_response(component: PartComponentView) -> PartComponentResponse:
    return PartComponentResponse(
        component_id=component.component_id,
        part_number=component.part_number,
        part_name=component.part_name,
        installed_quantity=component.installed_quantity,
        component_status=component.component_status,
        source=component.source,
        source_record_id=component.source_record_id,
        as_of=component.as_of.isoformat() if component.as_of is not None else None,
        freshness=component.freshness,
    )


def _availability_response(
    availability: PartAvailabilityView,
) -> PartAvailabilityResponse:
    return PartAvailabilityResponse(
        part_number=availability.part_number,
        available_quantity=availability.available_quantity,
        stock_status=availability.stock_status,
        compatibility_evidence=availability.compatibility_evidence,
        source=availability.source,
        source_record_id=availability.source_record_id,
        as_of=availability.as_of,
        authority_owner=availability.authority_owner,
        field_sources=availability.field_sources,
        tool_call_id=availability.tool_call_id,
    )
