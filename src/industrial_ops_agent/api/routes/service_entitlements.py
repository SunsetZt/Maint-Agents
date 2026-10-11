"""Authorized after-sales service entitlement verification API."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Literal, cast

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
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.service_entitlements.service import (
    LiveEntitlementView,
    MasterWarrantyView,
    ServiceActionPolicyView,
    ServiceEntitlementService,
    ServiceEntitlementView,
)
from industrial_ops_agent.tools.gateway import ToolGateway

router = APIRouter(prefix="/service-entitlements", tags=["service-entitlements"])


class MasterWarrantyResponse(BaseModel):
    warranty_id: str
    contract_number: str
    status: str
    coverage_start: datetime
    coverage_end: datetime | None
    service_level: str | None
    source: str
    source_record_id: str
    as_of: datetime | None
    freshness: str


class LiveEntitlementResponse(BaseModel):
    coverage: str
    valid: bool
    source: str
    source_record_id: str
    as_of: datetime
    authority_owner: str
    field_sources: dict[str, str]
    tool_call_id: str


class ServiceActionPolicyResponse(BaseModel):
    automatic_service_denial_allowed: bool
    automatic_billing_allowed: bool
    automatic_work_order_change_allowed: bool
    required_flow: str


class ServiceEntitlementResponse(BaseModel):
    asset_id: str
    asset_display_name: str | None
    model_code: str | None
    serial_number: str | None
    lifecycle_status: str | None
    customer_name: str | None
    site_id: str | None
    site_name: str | None
    decision: Literal["COVERED", "NOT_COVERED", "REVIEW_REQUIRED", "FACTS_UNAVAILABLE"]
    reason_codes: list[str]
    next_action: str
    master_warranty: MasterWarrantyResponse | None
    live_entitlement: LiveEntitlementResponse | None
    live_fact_failure: str | None
    action_policy: ServiceActionPolicyResponse


class ServiceEntitlementEnvelope(BaseModel):
    data: ServiceEntitlementResponse
    meta: dict[str, str]


@router.get("/lookup", response_model=ServiceEntitlementEnvelope)
async def lookup_service_entitlement(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    tool_gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    asset_id: Annotated[str, Query(min_length=1, max_length=128)],
) -> ServiceEntitlementEnvelope:
    try:
        view = ServiceEntitlementService(
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
    return ServiceEntitlementEnvelope(
        data=_response(view),
        meta={"request_id": request.state.request_id},
    )


def _response(view: ServiceEntitlementView) -> ServiceEntitlementResponse:
    return ServiceEntitlementResponse(
        asset_id=view.asset_id,
        asset_display_name=view.asset_display_name,
        model_code=view.model_code,
        serial_number=view.serial_number,
        lifecycle_status=view.lifecycle_status,
        customer_name=view.customer_name,
        site_id=view.site_id,
        site_name=view.site_name,
        decision=cast(
            Literal["COVERED", "NOT_COVERED", "REVIEW_REQUIRED", "FACTS_UNAVAILABLE"],
            view.decision,
        ),
        reason_codes=list(view.reason_codes),
        next_action=view.next_action,
        master_warranty=_master_response(view.master_warranty),
        live_entitlement=_live_response(view.live_entitlement),
        live_fact_failure=view.live_fact_failure,
        action_policy=_policy_response(view.action_policy),
    )


def _master_response(value: MasterWarrantyView | None) -> MasterWarrantyResponse | None:
    return MasterWarrantyResponse(**asdict(value)) if value is not None else None


def _live_response(value: LiveEntitlementView | None) -> LiveEntitlementResponse | None:
    return LiveEntitlementResponse(**asdict(value)) if value is not None else None


def _policy_response(value: ServiceActionPolicyView) -> ServiceActionPolicyResponse:
    return ServiceActionPolicyResponse(**asdict(value))
