"""Closed, non-production acceptance for all enterprise HTTP adapters."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.secrets import SecretSource, SecretValue
from industrial_ops_agent.tools.contracts import (
    EnterpriseToolError,
    ReservationOutcomeUnknown,
)
from industrial_ops_agent.tools.customer_notifications import (
    HttpNotificationDeliveryAdapter,
)
from industrial_ops_agent.tools.enterprise_http import EnterpriseHttpAdapter
from industrial_ops_agent.tools.fsm_assignments import HttpFsmAssignmentAdapter
from industrial_ops_agent.tools.procurement import HttpProcurementDeliveryAdapter
from industrial_ops_agent.tools.quotations import HttpServiceQuotationCatalogAdapter
from industrial_ops_agent.tools.refunds import HttpRefundDeliveryAdapter

CLASSIFICATION = "SIMULATED_NON_PRODUCTION"
SCHEMA_VERSION = "enterprise-integration-sandbox-acceptance/v1"
STATUS = "ENTERPRISE_SANDBOX_CLOSED_LOOP_PASSED"

_TENANT_ID = "tenant-m1-demo"
_ASSET_ID = "asset-m1-pump"
_SITE_ID = "site-m1-demo"
_CUSTOMER_ID = "customer-m1-demo"
_PART_NUMBER = "BRG-6312-C3"


class SandboxAcceptanceError(RuntimeError):
    """The sandbox did not satisfy its closed integration contract."""


@dataclass(frozen=True, slots=True)
class SandboxAcceptanceCredentials:
    enterprise_token: str = field(repr=False)
    notification_token: str = field(repr=False)
    notification_webhook_secret: str = field(repr=False)
    procurement_token: str = field(repr=False)
    procurement_webhook_secret: str = field(repr=False)
    refund_token: str = field(repr=False)
    refund_webhook_secret: str = field(repr=False)
    fsm_token: str = field(repr=False)
    fsm_webhook_secret: str = field(repr=False)
    quotation_token: str = field(repr=False)
    control_token: str = field(repr=False)

    @classmethod
    def from_environment(cls) -> SandboxAcceptanceCredentials:
        names = {
            "enterprise_token": "IOAP_ENTERPRISE_TOOL_GATEWAY_TOKEN",
            "notification_token": "IOAP_NOTIFICATION_PROVIDER_API_TOKEN",
            "notification_webhook_secret": "IOAP_NOTIFICATION_WEBHOOK_SECRET",
            "procurement_token": "IOAP_PROCUREMENT_PROVIDER_API_TOKEN",
            "procurement_webhook_secret": "IOAP_PROCUREMENT_WEBHOOK_SECRET",
            "refund_token": "IOAP_REFUND_PROVIDER_API_TOKEN",
            "refund_webhook_secret": "IOAP_REFUND_WEBHOOK_SECRET",
            "fsm_token": "IOAP_FSM_ASSIGNMENT_PROVIDER_API_TOKEN",
            "fsm_webhook_secret": "IOAP_FSM_ASSIGNMENT_WEBHOOK_SECRET",
            "quotation_token": "IOAP_SERVICE_QUOTATION_PROVIDER_API_TOKEN",
            "control_token": "IOAP_SANDBOX_CONTROL_TOKEN",
        }
        values = {name: os.environ.get(variable, "").strip() for name, variable in names.items()}
        if any(not value for value in values.values()):
            raise SandboxAcceptanceError("sandbox_acceptance_credentials_missing")
        return cls(
            enterprise_token=values["enterprise_token"],
            notification_token=values["notification_token"],
            notification_webhook_secret=values["notification_webhook_secret"],
            procurement_token=values["procurement_token"],
            procurement_webhook_secret=values["procurement_webhook_secret"],
            refund_token=values["refund_token"],
            refund_webhook_secret=values["refund_webhook_secret"],
            fsm_token=values["fsm_token"],
            fsm_webhook_secret=values["fsm_webhook_secret"],
            quotation_token=values["quotation_token"],
            control_token=values["control_token"],
        )


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SandboxAcceptanceCase(_ClosedModel):
    case_id: str = Field(min_length=1, max_length=128)
    domain: str = Field(min_length=1, max_length=64)
    capability: str = Field(min_length=1, max_length=160)
    status: Literal["PASSED"] = "PASSED"
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SandboxAcceptanceReport(_ClosedModel):
    schema_version: Literal["enterprise-integration-sandbox-acceptance/v1"] = (
        "enterprise-integration-sandbox-acceptance/v1"
    )
    classification: Literal["SIMULATED_NON_PRODUCTION"] = "SIMULATED_NON_PRODUCTION"
    production_claim: Literal[False] = False
    enterprise_production_data: Literal[False] = False
    data_source: Literal["PROJECT_GENERATED_SYNTHETIC_ENTERPRISE_RECORDS"] = (
        "PROJECT_GENERATED_SYNTHETIC_ENTERPRISE_RECORDS"
    )
    generated_at: datetime
    status: Literal["ENTERPRISE_SANDBOX_CLOSED_LOOP_PASSED"] = (
        "ENTERPRISE_SANDBOX_CLOSED_LOOP_PASSED"
    )
    tenant_id: str
    adapter_transport: Literal["REAL_HTTP_WITH_PROVIDER_AUTH"] = "REAL_HTTP_WITH_PROVIDER_AUTH"
    cases: tuple[SandboxAcceptanceCase, ...]
    provider_record_counts: dict[str, int]
    idempotency_replay_verified: Literal[True] = True
    unknown_outcome_reconciliation_verified: Literal[True] = True
    unauthorized_request_rejected: Literal[True] = True
    sensitive_values_persisted: Literal[False] = False
    excluded_production_scope: tuple[str, ...]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def execute_sandbox_acceptance(
    base_url: str,
    credentials: SandboxAcceptanceCredentials,
) -> SandboxAcceptanceReport:
    """Exercise every enterprise adapter over HTTP and return non-secret evidence."""

    endpoint = base_url.rstrip("/")
    allowed_prefixes = (
        "http://127.0.0.1:",
        "http://localhost:",
        "http://enterprise-sandbox:",
    )
    if not endpoint.startswith(allowed_prefixes):
        raise SandboxAcceptanceError("sandbox_acceptance_endpoint_is_not_local")
    _control(endpoint, credentials.control_token, "POST", "/sandbox/v1/reset", {})

    enterprise = EnterpriseHttpAdapter(
        endpoint,
        _secret(credentials.enterprise_token),
        allow_plain_http=True,
    )
    cases: list[SandboxAcceptanceCase] = []

    readonly_results = {
        tool_id: enterprise.invoke(tool_id, {"asset_id": _ASSET_ID})
        for tool_id in (
            "asset.get",
            "warranty.get",
            "parts.availability",
            "schedule.availability",
            "work_orders.history",
            "work_order.draft",
        )
    }
    if (
        readonly_results["asset.get"].get("model_code") != "PUMP-X100"
        or readonly_results["warranty.get"].get("valid") is not True
        or readonly_results["parts.availability"].get("part_number") != _PART_NUMBER
        or readonly_results["schedule.availability"].get("site_id") != _SITE_ID
    ):
        raise SandboxAcceptanceError("sandbox_authoritative_read_contract_failed")
    cases.append(_case("authoritative-reads", "EAM_ENTITLEMENT_WMS_FSM", readonly_results))

    _control(
        endpoint,
        credentials.control_token,
        "PUT",
        "/sandbox/v1/faults/wms",
        {"mode": "OUTCOME_UNKNOWN", "remaining": 1},
    )
    reservation_operation = "simulated-acceptance-reservation"
    try:
        enterprise.reserve(
            operation_id=reservation_operation,
            part_number=_PART_NUMBER,
            quantity=2,
        )
    except ReservationOutcomeUnknown:
        pass
    else:
        raise SandboxAcceptanceError("sandbox_unknown_outcome_was_not_observed")
    reservation = enterprise.reconcile(reservation_operation)
    if reservation is None:
        raise SandboxAcceptanceError("sandbox_reservation_reconciliation_failed")
    if (
        enterprise.reserve(
            operation_id=reservation_operation,
            part_number=_PART_NUMBER,
            quantity=2,
        )
        != reservation
    ):
        raise SandboxAcceptanceError("sandbox_reservation_replay_changed")
    issue = enterprise.issue(
        operation_id="simulated-acceptance-issue",
        reservation_id=reservation.reservation_id,
        part_number=reservation.part_number,
        quantity=reservation.quantity,
    )
    consumption = enterprise.apply_movement(
        operation_id="simulated-acceptance-consume",
        movement_kind="CONSUME",
        part_issue_id=issue.issue_id,
        reservation_id=reservation.reservation_id,
        part_number=reservation.part_number,
        quantity=1,
    )
    returned = enterprise.apply_movement(
        operation_id="simulated-acceptance-return",
        movement_kind="RETURN",
        part_issue_id=issue.issue_id,
        reservation_id=reservation.reservation_id,
        part_number=reservation.part_number,
        quantity=1,
    )
    if (
        enterprise.reconcile_issue(issue.operation_id) != issue
        or enterprise.reconcile_movement(consumption.operation_id, "CONSUME") != consumption
        or enterprise.reconcile_movement(returned.operation_id, "RETURN") != returned
    ):
        raise SandboxAcceptanceError("sandbox_wms_reconciliation_failed")
    cases.append(
        _case(
            "wms-lifecycle",
            "WMS",
            {
                "reservation_id": reservation.reservation_id,
                "issue_id": issue.issue_id,
                "consume_id": consumption.movement_id,
                "return_id": returned.movement_id,
            },
        )
    )

    fsm = HttpFsmAssignmentAdapter(
        endpoint,
        provider_id="enterprise-fsm",
        api_token=_secret(credentials.fsm_token),
        webhook_secret=_secret(credentials.fsm_webhook_secret),
        allow_plain_http=True,
    )
    candidates = fsm.list_candidates(
        tenant_id=_TENANT_ID,
        work_order_id="simulated-acceptance-work-order",
        asset_id=_ASSET_ID,
        site_id=_SITE_ID,
        service_window_start=None,
        service_window_end=None,
    )
    if len(candidates) != 1:
        raise SandboxAcceptanceError("sandbox_fsm_candidate_contract_failed")
    fsm_result = fsm.submit(
        operation_id="simulated-acceptance-fsm",
        candidate=candidates[0],
        payload={"work_order_id": "simulated-acceptance-work-order"},
    )
    if fsm.reconcile("simulated-acceptance-fsm") != fsm_result:
        raise SandboxAcceptanceError("sandbox_fsm_reconciliation_failed")
    cases.append(_case("fsm-assignment", "FSM", {"status": fsm_result.status}))

    procurement = HttpProcurementDeliveryAdapter(
        endpoint,
        provider_id="enterprise-erp",
        api_token=_secret(credentials.procurement_token),
        webhook_secret=_secret(credentials.procurement_webhook_secret),
        allow_plain_http=True,
    )
    procurement_profile = procurement.resolve_profile(tenant_id=_TENANT_ID)
    if procurement_profile is None:
        raise SandboxAcceptanceError("sandbox_procurement_profile_missing")
    procurement_result = procurement.submit(
        operation_id="simulated-acceptance-procurement",
        profile=procurement_profile,
        payload={"asset_id": _ASSET_ID, "part_number": _PART_NUMBER, "quantity": 1},
    )
    if procurement.reconcile("simulated-acceptance-procurement") != procurement_result:
        raise SandboxAcceptanceError("sandbox_procurement_reconciliation_failed")
    cases.append(_case("erp-procurement", "ERP", {"status": procurement_result.status}))

    refunds = HttpRefundDeliveryAdapter(
        endpoint,
        provider_id="enterprise-finance",
        api_token=_secret(credentials.refund_token),
        webhook_secret=_secret(credentials.refund_webhook_secret),
        allow_plain_http=True,
    )
    refund_profile = refunds.resolve_profile(
        tenant_id=_TENANT_ID,
        customer_subject_id=_CUSTOMER_ID,
    )
    if refund_profile is None:
        raise SandboxAcceptanceError("sandbox_refund_profile_missing")
    refund_result = refunds.submit(
        operation_id="simulated-acceptance-refund",
        profile=refund_profile,
        payload={"customer_subject_id": _CUSTOMER_ID, "amount_minor": 1200},
    )
    if refunds.reconcile("simulated-acceptance-refund") != refund_result:
        raise SandboxAcceptanceError("sandbox_refund_reconciliation_failed")
    cases.append(
        _case(
            "finance-refund",
            "FINANCE",
            {
                "status": refund_result.status,
                "payment_status": refund_result.payment_status,
            },
        )
    )

    quotation = HttpServiceQuotationCatalogAdapter(
        endpoint,
        provider_id="enterprise-cpq",
        api_token=_secret(credentials.quotation_token),
        allow_plain_http=True,
    )
    quotation_candidates = quotation.list_candidates(
        tenant_id=_TENANT_ID,
        incident_ref="simulated-acceptance-incident",
        asset_ref=_ASSET_ID,
        diagnosis_ref="simulated-acceptance-diagnosis",
    )
    if len(quotation_candidates) != 1:
        raise SandboxAcceptanceError("sandbox_cpq_candidate_contract_failed")
    resolved_quotation = quotation.resolve_candidate(
        tenant_id=_TENANT_ID,
        candidate_id=quotation_candidates[0].candidate_id,
        incident_ref="simulated-acceptance-incident",
        asset_ref=_ASSET_ID,
        diagnosis_ref="simulated-acceptance-diagnosis",
    )
    if resolved_quotation is None or resolved_quotation != quotation_candidates[0]:
        raise SandboxAcceptanceError("sandbox_cpq_resolution_failed")
    cases.append(
        _case(
            "cpq-quotation",
            "CPQ",
            {
                "candidate_id": resolved_quotation.candidate_id,
                "total": resolved_quotation.total,
            },
        )
    )

    notifications = HttpNotificationDeliveryAdapter(
        endpoint,
        provider_id="enterprise-notify",
        api_token=_secret(credentials.notification_token),
        webhook_secret=_secret(credentials.notification_webhook_secret),
        allow_plain_http=True,
    )
    contacts = notifications.resolve_contacts(
        tenant_id=_TENANT_ID,
        recipient_subject_id=_CUSTOMER_ID,
    )
    if {contact.channel for contact in contacts} != {"EMAIL", "SMS"}:
        raise SandboxAcceptanceError("sandbox_notification_contacts_invalid")
    delivery = notifications.deliver(
        operation_id="simulated-acceptance-notification",
        contact=contacts[0],
        rendered_message="Synthetic industrial maintenance acceptance update.",
    )
    if notifications.reconcile("simulated-acceptance-notification") != delivery:
        raise SandboxAcceptanceError("sandbox_notification_reconciliation_failed")
    cases.append(_case("customer-notification", "EMAIL_SMS", {"status": delivery.status}))

    unauthorized = EnterpriseHttpAdapter(
        endpoint,
        _secret("invalid-simulated-acceptance-token"),
        allow_plain_http=True,
    )
    try:
        unauthorized.invoke("asset.get", {"asset_id": _ASSET_ID})
    except EnterpriseToolError:
        pass
    else:
        raise SandboxAcceptanceError("sandbox_unauthorized_request_was_accepted")
    cases.append(_case("provider-authentication", "SECURITY", {"rejected": True}))

    state = _control(
        endpoint,
        credentials.control_token,
        "GET",
        "/sandbox/v1/state",
        None,
    )
    counts = state.get("record_counts")
    expected_counts = {
        "fsm": 1,
        "notification": 1,
        "procurement": 1,
        "refund": 1,
        "wms": 1,
        "wms-consumption": 1,
        "wms-issue": 1,
        "wms-return": 1,
    }
    if counts != expected_counts or state.get("faults") != {}:
        raise SandboxAcceptanceError("sandbox_provider_ledger_is_incomplete")

    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "production_claim": False,
        "enterprise_production_data": False,
        "data_source": "PROJECT_GENERATED_SYNTHETIC_ENTERPRISE_RECORDS",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "tenant_id": _TENANT_ID,
        "adapter_transport": "REAL_HTTP_WITH_PROVIDER_AUTH",
        "cases": [item.model_dump(mode="json") for item in cases],
        "provider_record_counts": expected_counts,
        "idempotency_replay_verified": True,
        "unknown_outcome_reconciliation_verified": True,
        "unauthorized_request_rejected": True,
        "sensitive_values_persisted": False,
        "excluded_production_scope": [
            "enterprise_production_data",
            "production_https_and_mtls",
            "enterprise_oidc",
            "production_secret_delivery",
            "provider_capacity_and_disaster_recovery",
            "business_security_platform_signoff",
        ],
    }
    draft = SandboxAcceptanceReport.model_validate({**unsigned, "evidence_chain_sha256": "0" * 64})
    normalized = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return draft.model_copy(update={"evidence_chain_sha256": _digest(normalized)})


def _secret(value: str) -> SecretValue:
    return SecretValue(value, source=SecretSource.ENVIRONMENT)


def _case(case_id: str, domain: str, evidence: object) -> SandboxAcceptanceCase:
    capabilities = {
        "authoritative-reads": "authoritative read contracts and field ownership",
        "wms-lifecycle": ("reservation, issue, consumption, return, replay and reconciliation"),
        "fsm-assignment": "eligible assignment and operation reconciliation",
        "erp-procurement": "procurement profile, submission and reconciliation",
        "finance-refund": ("refund profile, submission and payment-state reconciliation"),
        "cpq-quotation": "quotation candidate listing and identity-bound resolution",
        "customer-notification": ("authorized contact resolution, delivery and reconciliation"),
        "provider-authentication": "invalid provider credential rejection",
    }
    return SandboxAcceptanceCase(
        case_id=case_id,
        domain=domain,
        capability=capabilities[case_id],
        evidence_sha256=_digest(evidence),
    )


def _control(
    base_url: str,
    token: str,
    method: str,
    path: str,
    payload: dict[str, object] | None,
) -> dict[str, Any]:
    raw = None if payload is None else _canonical(payload)
    request = Request(  # noqa: S310 - restricted local/test sandbox endpoint
        f"{base_url}{path}",
        data=raw,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310
            value = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise SandboxAcceptanceError("sandbox_control_request_failed") from exc
    if not isinstance(value, dict) or value.get("classification") != CLASSIFICATION:
        raise SandboxAcceptanceError("sandbox_control_response_is_invalid")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()
