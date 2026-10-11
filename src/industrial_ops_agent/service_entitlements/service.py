"""Cross-check mastered warranty data with the live contract entitlement owner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from industrial_ops_agent.application.assets import AssetQueryService
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.domain.assets import AssetFreshness, AssetWarranty
from industrial_ops_agent.tools.contracts import EnterpriseToolError
from industrial_ops_agent.tools.gateway import ToolInvocationResult, ToolRateLimitExceeded

LIVE_FACT_MAX_AGE = timedelta(hours=24)
LIVE_FACT_FUTURE_TOLERANCE = timedelta(minutes=5)


class EntitlementFactGateway(Protocol):
    def invoke(
        self,
        identity: IdentityContext,
        *,
        tool_id: str,
        version: str,
        parameters: dict[str, Any],
        request_id: str,
        agent_run_id: str | None = None,
    ) -> ToolInvocationResult: ...


@dataclass(frozen=True, slots=True)
class MasterWarrantyView:
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


@dataclass(frozen=True, slots=True)
class LiveEntitlementView:
    coverage: str
    valid: bool
    source: str
    source_record_id: str
    as_of: datetime
    authority_owner: str
    field_sources: dict[str, str]
    tool_call_id: str


@dataclass(frozen=True, slots=True)
class ServiceActionPolicyView:
    automatic_service_denial_allowed: bool
    automatic_billing_allowed: bool
    automatic_work_order_change_allowed: bool
    required_flow: str


@dataclass(frozen=True, slots=True)
class ServiceEntitlementView:
    asset_id: str
    asset_display_name: str | None
    model_code: str | None
    serial_number: str | None
    lifecycle_status: str | None
    customer_name: str | None
    site_id: str | None
    site_name: str | None
    decision: str
    reason_codes: tuple[str, ...]
    next_action: str
    master_warranty: MasterWarrantyView | None
    live_entitlement: LiveEntitlementView | None
    live_fact_failure: str | None
    action_policy: ServiceActionPolicyView


class ServiceEntitlementService:
    def __init__(
        self,
        asset_query: AssetQueryService,
        fact_gateway: EntitlementFactGateway,
    ) -> None:
        self._asset_query = asset_query
        self._fact_gateway = fact_gateway

    def lookup(
        self,
        identity: IdentityContext,
        asset_id: str,
        *,
        request_id: str,
        now: datetime | None = None,
    ) -> ServiceEntitlementView:
        reference_time = now or datetime.now(UTC)
        asset = self._asset_query.get(identity, asset_id, request_id=request_id)
        live: LiveEntitlementView | None = None
        failure: str | None = None
        try:
            result = self._fact_gateway.invoke(
                identity,
                tool_id="warranty.get",
                version="1.0.0",
                parameters={"asset_id": asset_id},
                request_id=request_id,
            )
            if result.status != "SUCCEEDED" or result.data is None:
                failure = "tool_result_not_usable"
            else:
                live = _live_entitlement(result.tool_call_id, result.data)
        except AuthorizationDenied:
            failure = "tool_authorization_denied"
        except EnterpriseToolError as exc:
            failure = exc.reason
        except ToolRateLimitExceeded:
            failure = "tool_rate_limit_exceeded"
        except (KeyError, TypeError, ValueError):
            failure = "warranty_fact_invalid"

        decision, reasons, next_action = _decision(
            asset.warranty,
            live,
            reference_time,
        )
        summary = asset.summary
        return ServiceEntitlementView(
            asset_id=summary.asset_id,
            asset_display_name=summary.display_name,
            model_code=summary.model_code,
            serial_number=summary.serial_number,
            lifecycle_status=summary.lifecycle_status,
            customer_name=summary.customer_name,
            site_id=summary.site_id,
            site_name=summary.site_name,
            decision=decision,
            reason_codes=reasons,
            next_action=next_action,
            master_warranty=_master_warranty(asset.warranty),
            live_entitlement=live,
            live_fact_failure=failure,
            action_policy=ServiceActionPolicyView(
                automatic_service_denial_allowed=False,
                automatic_billing_allowed=False,
                automatic_work_order_change_allowed=False,
                required_flow="HUMAN_CONFIRM_BEFORE_QUOTE_DISPATCH_OR_DENIAL",
            ),
        )


def _master_warranty(warranty: AssetWarranty | None) -> MasterWarrantyView | None:
    if warranty is None:
        return None
    return MasterWarrantyView(
        warranty_id=warranty.warranty_id,
        contract_number=warranty.contract_number,
        status=warranty.status,
        coverage_start=warranty.coverage_start,
        coverage_end=warranty.coverage_end,
        service_level=warranty.service_level,
        source=warranty.source.source_system,
        source_record_id=warranty.source.source_record_id,
        as_of=warranty.source.as_of,
        freshness=warranty.source.freshness.value,
    )


def _live_entitlement(tool_call_id: str, data: dict[str, Any]) -> LiveEntitlementView:
    authority = data["authority"]
    if not isinstance(authority, dict):
        raise TypeError
    field_sources = authority["field_sources"]
    if not isinstance(field_sources, dict):
        raise TypeError
    coverage = data["coverage"]
    valid = data["valid"]
    if not isinstance(coverage, str) or not coverage.strip() or not isinstance(valid, bool):
        raise TypeError
    as_of = _timestamp(data["as_of"])
    return LiveEntitlementView(
        coverage=coverage.strip(),
        valid=valid,
        source=str(data["source"]),
        source_record_id=str(data["source_record_id"]),
        as_of=as_of,
        authority_owner=str(authority["owner"]),
        field_sources={str(field): str(source) for field, source in field_sources.items()},
        tool_call_id=tool_call_id,
    )


def _timestamp(value: object) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include timezone")
    return parsed


def _decision(
    master: AssetWarranty | None,
    live: LiveEntitlementView | None,
    now: datetime,
) -> tuple[str, tuple[str, ...], str]:
    if live is None:
        return (
            "FACTS_UNAVAILABLE",
            ("LIVE_ENTITLEMENT_UNAVAILABLE",),
            "RETRY_OR_CONTACT_CONTRACT_OWNER",
        )

    reasons: list[str] = []
    if live.as_of < now - LIVE_FACT_MAX_AGE:
        reasons.append("LIVE_ENTITLEMENT_STALE")
    if live.as_of > now + LIVE_FACT_FUTURE_TOLERANCE:
        reasons.append("LIVE_ENTITLEMENT_FUTURE_TIMESTAMP")
    if master is None:
        reasons.append("MASTER_WARRANTY_MISSING")
    else:
        if master.source.freshness is not AssetFreshness.CURRENT:
            reasons.append("MASTER_WARRANTY_NOT_CURRENT")
        mastered_valid = _mastered_valid(master, now)
        if mastered_valid != live.valid:
            reasons.append("WARRANTY_SOURCES_CONFLICT")
    if reasons:
        return (
            "REVIEW_REQUIRED",
            tuple(reasons),
            "CONTACT_CONTRACT_OWNER_AND_CONFIRM_SERVICE_SCOPE",
        )
    if live.valid:
        return (
            "COVERED",
            ("AUTHORITATIVE_AND_MASTERED_FACTS_AGREE",),
            "HUMAN_CONFIRM_COVERAGE_BEFORE_DISPATCH",
        )
    return (
        "NOT_COVERED",
        ("AUTHORITATIVE_AND_MASTERED_FACTS_AGREE",),
        "HUMAN_CONFIRM_BEFORE_QUOTE_OR_SERVICE_DENIAL",
    )


def _mastered_valid(master: AssetWarranty, now: datetime) -> bool:
    status_active = master.status.strip().upper() in {"ACTIVE", "VALID", "IN_FORCE"}
    coverage_start = _aware(master.coverage_start)
    coverage_end = _aware(master.coverage_end) if master.coverage_end is not None else None
    return (
        status_active
        and coverage_start <= now
        and (coverage_end is None or now < coverage_end)
    )


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
