"""Authorized M1 asset selector API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
)
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.application.assets import (
    AssetQueryService,
    ResourceNotVisible,
)
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.domain.assets import (
    AssetComponent,
    AssetCustomer,
    AssetDetail,
    AssetSite,
    AssetSourceReference,
    AssetSummary,
    AssetWarranty,
)
from industrial_ops_agent.persistence.database import Database

router = APIRouter(prefix="/assets", tags=["assets"])


class AssetResponse(BaseModel):
    asset_id: str
    source_system: str
    source_record_id: str
    as_of: datetime | None
    version: int
    source_kind: str
    freshness: str
    model_code: str | None = None
    display_name: str | None = None
    serial_number: str | None = None
    lifecycle_status: str | None = None
    customer_name: str | None = None
    site_id: str | None = None
    site_name: str | None = None
    warranty_status: str | None = None
    warranty_end: datetime | None = None
    access_basis: str | None = None


class AssetSourceResponse(BaseModel):
    scope: str
    source_system: str
    source_record_id: str
    as_of: datetime | None
    version: int
    source_kind: str
    freshness: str


class AssetCustomerResponse(BaseModel):
    customer_id: str
    customer_name: str
    source: AssetSourceResponse


class AssetSiteResponse(BaseModel):
    site_id: str
    site_name: str
    address: str | None
    source: AssetSourceResponse


class AssetWarrantyResponse(BaseModel):
    warranty_id: str
    contract_number: str
    status: str
    coverage_start: datetime
    coverage_end: datetime | None
    service_level: str | None
    source: AssetSourceResponse


class AssetComponentResponse(BaseModel):
    component_id: str
    part_number: str
    part_name: str
    serial_number: str | None
    quantity: int
    status: str
    installed_at: datetime | None
    source: AssetSourceResponse


class AssetDetailResponse(AssetResponse):
    customer: AssetCustomerResponse | None = None
    site: AssetSiteResponse | None = None
    warranty: AssetWarrantyResponse | None = None
    components: list[AssetComponentResponse] = Field(default_factory=list)
    source_trace: list[AssetSourceResponse] = Field(default_factory=list)


class AssetEnvelope(BaseModel):
    data: AssetDetailResponse
    meta: dict[str, str]


class AssetListEnvelope(BaseModel):
    data: list[AssetResponse]
    meta: dict[str, str]


@router.get("", response_model=AssetListEnvelope, responses=STANDARD_ERROR_RESPONSES)
async def list_assets(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> AssetListEnvelope:
    service = AssetQueryService(database, authorizer)
    try:
        assets = service.list(identity, request_id=request.state.request_id)
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    return AssetListEnvelope(
        data=[_asset_response(asset) for asset in assets],
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/{asset_id}",
    response_model=AssetEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
async def get_asset(
    asset_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> AssetEnvelope:
    service = AssetQueryService(database, authorizer)
    try:
        asset = service.get(
            identity,
            asset_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    return AssetEnvelope(
        data=_asset_detail_response(asset),
        meta={"request_id": request.state.request_id},
    )


def _asset_response(asset: AssetSummary) -> AssetResponse:
    return AssetResponse(
        asset_id=asset.asset_id,
        source_system=asset.source_system,
        source_record_id=asset.source_record_id,
        as_of=asset.as_of,
        version=asset.version,
        source_kind=asset.source_kind.value,
        freshness=asset.freshness.value,
        model_code=asset.model_code,
        display_name=asset.display_name,
        serial_number=asset.serial_number,
        lifecycle_status=asset.lifecycle_status,
        customer_name=asset.customer_name,
        site_id=asset.site_id,
        site_name=asset.site_name,
        warranty_status=asset.warranty_status,
        warranty_end=asset.warranty_end,
        access_basis=asset.access_basis,
    )


def _asset_detail_response(asset: AssetDetail) -> AssetDetailResponse:
    return AssetDetailResponse(
        **_asset_response(asset.summary).model_dump(),
        customer=_customer_response(asset.customer) if asset.customer is not None else None,
        site=_site_response(asset.site) if asset.site is not None else None,
        warranty=(_warranty_response(asset.warranty) if asset.warranty is not None else None),
        components=[_component_response(item) for item in asset.components],
        source_trace=[_source_response(item) for item in asset.source_trace],
    )


def _source_response(source: AssetSourceReference) -> AssetSourceResponse:
    return AssetSourceResponse(
        scope=source.scope,
        source_system=source.source_system,
        source_record_id=source.source_record_id,
        as_of=source.as_of,
        version=source.version,
        source_kind=source.source_kind.value,
        freshness=source.freshness.value,
    )


def _customer_response(customer: AssetCustomer) -> AssetCustomerResponse:
    return AssetCustomerResponse(
        customer_id=customer.customer_id,
        customer_name=customer.customer_name,
        source=_source_response(customer.source),
    )


def _site_response(site: AssetSite) -> AssetSiteResponse:
    return AssetSiteResponse(
        site_id=site.site_id,
        site_name=site.site_name,
        address=site.address,
        source=_source_response(site.source),
    )


def _warranty_response(warranty: AssetWarranty) -> AssetWarrantyResponse:
    return AssetWarrantyResponse(
        warranty_id=warranty.warranty_id,
        contract_number=warranty.contract_number,
        status=warranty.status,
        coverage_start=warranty.coverage_start,
        coverage_end=warranty.coverage_end,
        service_level=warranty.service_level,
        source=_source_response(warranty.source),
    )


def _component_response(component: AssetComponent) -> AssetComponentResponse:
    return AssetComponentResponse(
        component_id=component.component_id,
        part_number=component.part_number,
        part_name=component.part_name,
        serial_number=component.serial_number,
        quantity=component.quantity,
        status=component.status,
        installed_at=component.installed_at,
        source=_source_response(component.source),
    )


def _not_visible() -> AppError:
    return AppError(
        status_code=404,
        code="resource_not_found_or_not_visible",
        category="not_found",
        message="Resource not found or not visible",
    )
