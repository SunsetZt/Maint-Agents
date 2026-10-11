"""Asset source and freshness semantics exposed by the M1 API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum


class AssetSourceKind(StrEnum):
    ENTERPRISE = "enterprise"
    SYNTHETIC = "synthetic"


class AssetFreshness(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AssetSummary:
    asset_id: str
    source_system: str
    source_record_id: str
    as_of: datetime | None
    version: int
    source_kind: AssetSourceKind
    freshness: AssetFreshness
    model_code: str | None = None
    display_name: str | None = None
    serial_number: str | None = None
    lifecycle_status: str | None = None
    customer_name: str | None = None
    site_id: str | None = None
    site_name: str | None = None
    warranty_status: str | None = None
    warranty_end: datetime | None = None
    access_basis: str = "ASSET_SCOPE"


@dataclass(frozen=True, slots=True)
class AssetSourceReference:
    scope: str
    source_system: str
    source_record_id: str
    as_of: datetime | None
    version: int
    source_kind: AssetSourceKind
    freshness: AssetFreshness


@dataclass(frozen=True, slots=True)
class AssetCustomer:
    customer_id: str
    customer_name: str
    source: AssetSourceReference


@dataclass(frozen=True, slots=True)
class AssetSite:
    site_id: str
    site_name: str
    address: str | None
    source: AssetSourceReference


@dataclass(frozen=True, slots=True)
class AssetWarranty:
    warranty_id: str
    contract_number: str
    status: str
    coverage_start: datetime
    coverage_end: datetime | None
    service_level: str | None
    source: AssetSourceReference


@dataclass(frozen=True, slots=True)
class AssetComponent:
    component_id: str
    part_number: str
    part_name: str
    serial_number: str | None
    quantity: int
    status: str
    installed_at: datetime | None
    source: AssetSourceReference


@dataclass(frozen=True, slots=True)
class AssetDetail:
    summary: AssetSummary
    customer: AssetCustomer | None
    site: AssetSite | None
    warranty: AssetWarranty | None
    components: tuple[AssetComponent, ...]
    source_trace: tuple[AssetSourceReference, ...]


def classify_source(source_system: str) -> AssetSourceKind:
    """Keep locally generated seed data distinguishable from enterprise facts."""

    if source_system.casefold().startswith("synthetic"):
        return AssetSourceKind.SYNTHETIC
    return AssetSourceKind.ENTERPRISE


def classify_freshness(
    as_of: datetime | None,
    *,
    now: datetime | None = None,
    stale_after: timedelta = timedelta(hours=24),
) -> AssetFreshness:
    """Classify a source observation without inventing a timestamp when absent."""

    if as_of is None:
        return AssetFreshness.UNKNOWN
    reference = now or datetime.now(UTC)
    observed = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=UTC)
    return AssetFreshness.CURRENT if observed >= reference - stale_after else AssetFreshness.STALE
