"""Authorized asset queries for the M1 device selector."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain.assets import (
    AssetComponent,
    AssetCustomer,
    AssetDetail,
    AssetSite,
    AssetSourceReference,
    AssetSummary,
    AssetWarranty,
    classify_freshness,
    classify_source,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetComponentRecord,
    AssetCustomerLinkRecord,
    AssetRecord,
    AssetSiteLinkRecord,
    AssetWarrantyRecord,
)
from industrial_ops_agent.persistence.repositories import AssetRepository


class ResourceNotVisible(Exception):
    """A resource is absent from the caller's effective authorized view."""


class AssetQueryService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def list(self, identity: IdentityContext, *, request_id: str) -> list[AssetSummary]:
        self._require(identity, asset_id=None, site_id=None, request_id=request_id)
        with self._database.transaction(identity.tenant_context) as session:
            records = AssetRepository(session, identity.tenant_context).list_allowed(
                identity.asset_ids,
                identity.site_ids,
            )
            summaries: list[AssetSummary] = []
            for record in records:
                customer, site, warranty, _components = _relations(
                    session,
                    identity.tenant_id,
                    record.asset_id,
                )
                summaries.append(
                    _to_summary(
                        record,
                        customer,
                        site,
                        warranty,
                        access_basis=_access_basis(
                            identity,
                            record.asset_id,
                            site.site_id if site is not None else None,
                        ),
                    )
                )
            return summaries

    def get(
        self,
        identity: IdentityContext,
        asset_id: str,
        *,
        request_id: str,
    ) -> AssetDetail:
        with self._database.transaction(identity.tenant_context) as session:
            record = AssetRepository(session, identity.tenant_context).get(asset_id)
            if record is None:
                raise ResourceNotVisible
            customer, site, warranty, components = _relations(
                session,
                identity.tenant_id,
                asset_id,
            )
            access_basis = _access_basis(
                identity,
                asset_id,
                site.site_id if site is not None else None,
            )
            if access_basis == "ASSET_SCOPE":
                self._require(
                    identity,
                    asset_id=asset_id,
                    site_id=None,
                    request_id=request_id,
                )
            elif access_basis == "SITE_SCOPE" and site is not None:
                self._require(
                    identity,
                    asset_id=None,
                    site_id=site.site_id,
                    request_id=request_id,
                )
            else:
                self._require(
                    identity,
                    asset_id=asset_id,
                    site_id=None,
                    request_id=request_id,
                )
                raise ResourceNotVisible
            summary = _to_summary(
                record,
                customer,
                site,
                warranty,
                access_basis=access_basis,
            )
            sources = [
                _source(
                    "asset",
                    record.source_system,
                    record.source_record_id,
                    record.as_of,
                    record.version,
                )
            ]
            sources.extend(
                value.source for value in (customer, site, warranty) if value is not None
            )
            sources.extend(component.source for component in components)
            return AssetDetail(
                summary=summary,
                customer=customer,
                site=site,
                warranty=warranty,
                components=components,
                source_trace=tuple(sources),
            )

    def _require(
        self,
        identity: IdentityContext,
        *,
        asset_id: str | None,
        site_id: str | None,
        request_id: str,
    ) -> None:
        try:
            self._authorizer.require(
                identity,
                Action.READ_ASSET,
                ResourceContext(
                    tenant_id=identity.tenant_id,
                    resource_id=asset_id,
                    asset_id=asset_id,
                    site_id=site_id,
                ),
                request_id=request_id,
            )
        except AuthorizationDenied as exc:
            raise ResourceNotVisible from exc


def _to_summary(
    record: AssetRecord,
    customer: AssetCustomer | None,
    site: AssetSite | None,
    warranty: AssetWarranty | None,
    *,
    access_basis: str,
) -> AssetSummary:
    return AssetSummary(
        asset_id=record.asset_id,
        source_system=record.source_system,
        source_record_id=record.source_record_id,
        as_of=record.as_of,
        version=record.version,
        source_kind=classify_source(record.source_system),
        freshness=classify_freshness(record.as_of),
        model_code=record.model_code,
        display_name=record.display_name,
        serial_number=record.serial_number,
        lifecycle_status=record.lifecycle_status,
        customer_name=customer.customer_name if customer is not None else None,
        site_id=site.site_id if site is not None else None,
        site_name=site.site_name if site is not None else None,
        warranty_status=warranty.status if warranty is not None else None,
        warranty_end=warranty.coverage_end if warranty is not None else None,
        access_basis=access_basis,
    )


def _relations(
    session: Session,
    tenant_id: str,
    asset_id: str,
) -> tuple[
    AssetCustomer | None,
    AssetSite | None,
    AssetWarranty | None,
    tuple[AssetComponent, ...],
]:
    customer_record = session.scalar(
        select(AssetCustomerLinkRecord).where(
            AssetCustomerLinkRecord.tenant_id == tenant_id,
            AssetCustomerLinkRecord.asset_id == asset_id,
        )
    )
    site_record = session.scalar(
        select(AssetSiteLinkRecord).where(
            AssetSiteLinkRecord.tenant_id == tenant_id,
            AssetSiteLinkRecord.asset_id == asset_id,
        )
    )
    warranty_record = session.scalar(
        select(AssetWarrantyRecord).where(
            AssetWarrantyRecord.tenant_id == tenant_id,
            AssetWarrantyRecord.asset_id == asset_id,
        )
    )
    component_records = list(
        session.scalars(
            select(AssetComponentRecord)
            .where(
                AssetComponentRecord.tenant_id == tenant_id,
                AssetComponentRecord.asset_id == asset_id,
            )
            .order_by(AssetComponentRecord.part_number, AssetComponentRecord.component_id)
        )
    )
    customer = (
        AssetCustomer(
            customer_id=customer_record.customer_id,
            customer_name=customer_record.customer_name,
            source=_record_source("customer", customer_record),
        )
        if customer_record is not None
        else None
    )
    site = (
        AssetSite(
            site_id=site_record.site_id,
            site_name=site_record.site_name,
            address=site_record.address,
            source=_record_source("site", site_record),
        )
        if site_record is not None
        else None
    )
    warranty = (
        AssetWarranty(
            warranty_id=warranty_record.warranty_id,
            contract_number=warranty_record.contract_number,
            status=warranty_record.status,
            coverage_start=warranty_record.coverage_start,
            coverage_end=warranty_record.coverage_end,
            service_level=warranty_record.service_level,
            source=_record_source("warranty", warranty_record),
        )
        if warranty_record is not None
        else None
    )
    components = tuple(
        AssetComponent(
            component_id=item.component_id,
            part_number=item.part_number,
            part_name=item.part_name,
            serial_number=item.serial_number,
            quantity=item.quantity,
            status=item.status,
            installed_at=item.installed_at,
            source=_record_source(f"component:{item.component_id}", item),
        )
        for item in component_records
    )
    return customer, site, warranty, components


def _record_source(
    scope: str,
    record: AssetCustomerLinkRecord
    | AssetSiteLinkRecord
    | AssetWarrantyRecord
    | AssetComponentRecord,
) -> AssetSourceReference:
    return _source(
        scope,
        record.source_system,
        record.source_record_id,
        record.as_of,
        record.version,
    )


def _source(
    scope: str,
    source_system: str,
    source_record_id: str,
    as_of: datetime | None,
    version: int,
) -> AssetSourceReference:
    return AssetSourceReference(
        scope=scope,
        source_system=source_system,
        source_record_id=source_record_id,
        as_of=as_of,
        version=version,
        source_kind=classify_source(source_system),
        freshness=classify_freshness(as_of),
    )


def _access_basis(
    identity: IdentityContext,
    asset_id: str,
    site_id: str | None,
) -> str:
    if asset_id in identity.asset_ids:
        return "ASSET_SCOPE"
    if site_id is not None and site_id in identity.site_ids:
        return "SITE_SCOPE"
    return "NONE"
