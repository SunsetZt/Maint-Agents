"""Unified local HTTP peer for the platform's existing enterprise adapters."""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.sandbox.state import (
    IdempotencyConflict,
    SandboxState,
    WriteDecision,
)

_ASSET_ID = "asset-m1-pump"
_TENANT_ID = "tenant-m1-demo"
_SITE_ID = "site-m1-demo"
_ENGINEER_ID = "subject-m1-engineer"
_CUSTOMER_ID = "customer-m1-demo"
_PART_NUMBER = "BRG-6312-C3"
_PROVIDERS = frozenset({"eam", "wms", "fsm", "procurement", "refund", "notification", "quotation"})
_READ_FAULTS = frozenset({"UNAVAILABLE", "STALE"})


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    """Validated settings with secret fields excluded from representations."""

    enabled: bool
    environment: str
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
    platform_api_url: str
    tenant_id: str = _TENANT_ID

    @classmethod
    def from_environment(cls) -> SandboxSettings:
        enabled = os.environ.get("IOAP_SANDBOX_ENABLED", "").strip().lower() == "true"
        environment = os.environ.get("IOAP_ENVIRONMENT", "").strip().lower()
        if not enabled:
            raise ValueError("enterprise sandbox must be explicitly enabled")
        if environment not in {"local", "test"}:
            raise ValueError("enterprise sandbox is restricted to local/test environments")

        required = {
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
            "platform_api_url": "IOAP_SANDBOX_PLATFORM_API_URL",
        }
        values = {name: os.environ.get(key, "").strip() for name, key in required.items()}
        if any(not value for value in values.values()):
            raise ValueError("missing required sandbox secret or endpoint configuration")
        access_tokens = [
            values["enterprise_token"],
            values["notification_token"],
            values["procurement_token"],
            values["refund_token"],
            values["fsm_token"],
            values["quotation_token"],
            values["control_token"],
        ]
        if len(set(access_tokens)) != len(access_tokens):
            raise ValueError("sandbox provider and control tokens must be distinct")
        return cls(
            enabled=enabled,
            environment=environment,
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
            platform_api_url=values["platform_api_url"].rstrip("/"),
            tenant_id=os.environ.get("IOAP_SANDBOX_TENANT_ID", _TENANT_ID).strip(),
        )


class _StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ReservationRequest(_StrictRequest):
    operation_id: str = Field(min_length=1, max_length=255)
    part_number: str = Field(pattern=r"^[A-Z0-9-]{3,64}$")
    quantity: int = Field(ge=1, le=20)


class _PartIssueRequest(_ReservationRequest):
    reservation_id: str = Field(min_length=1, max_length=255)


class _PartMovementRequest(_PartIssueRequest):
    movement_kind: Literal["CONSUME", "RETURN"]
    part_issue_id: str = Field(min_length=1, max_length=255)


class _AssignmentRequest(_StrictRequest):
    operation_id: str = Field(min_length=1, max_length=128)
    candidate_profile_id: str = Field(min_length=1, max_length=128)
    profile_version: str = Field(min_length=1, max_length=128)
    assignment: dict[str, object]


class _ProcurementRequest(_StrictRequest):
    operation_id: str = Field(min_length=1, max_length=128)
    profile_id: str = Field(min_length=1, max_length=128)
    profile_version: str = Field(min_length=1, max_length=128)
    purchase_request: dict[str, object]


class _RefundRequest(_StrictRequest):
    operation_id: str = Field(min_length=1, max_length=128)
    profile_id: str = Field(min_length=1, max_length=128)
    profile_version: str = Field(min_length=1, max_length=128)
    refund_request: dict[str, object]


class _DeliveryRequest(_StrictRequest):
    operation_id: str = Field(min_length=1, max_length=128)
    channel: Literal["EMAIL", "SMS"]
    destination: str = Field(min_length=1, max_length=320)
    message: str = Field(min_length=1, max_length=20_000)


class _FaultRequest(_StrictRequest):
    mode: Literal["UNAVAILABLE", "REJECT", "OUTCOME_UNKNOWN", "STALE"]
    remaining: int = Field(ge=1, le=10)


class _CompletionRequest(_StrictRequest):
    status: str = Field(min_length=1, max_length=32)
    payment_status: str | None = Field(default=None, min_length=1, max_length=32)
    reason: str | None = Field(default=None, max_length=255)


class _FulfillmentRequest(_StrictRequest):
    status: Literal[
        "APPROVED",
        "ORDERED",
        "IN_TRANSIT",
        "PARTIALLY_RECEIVED",
        "RECEIVED",
        "REJECTED",
        "CANCELLED",
    ]
    received_quantity: int = Field(ge=0, le=1_000_000)
    expected_delivery_at: datetime | None = None
    reason: str | None = Field(default=None, max_length=255)


def create_app(
    settings: SandboxSettings,
    *,
    state: SandboxState | None = None,
) -> FastAPI:
    """Create an explicitly non-production sandbox application."""

    if not settings.enabled or settings.environment not in {"local", "test"}:
        raise ValueError("enterprise sandbox is restricted to local/test environments")
    sandbox_state = state or SandboxState()
    started_at = datetime.now(UTC).replace(microsecond=0)
    application = FastAPI(
        title="Enterprise Integration Sandbox",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.sandbox = sandbox_state
    application.state.classification = "SIMULATED_NON_PRODUCTION"

    @application.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live", "classification": "SIMULATED_NON_PRODUCTION"}

    @application.get("/health/ready")
    def ready() -> dict[str, str]:
        return {"status": "ready", "classification": "SIMULATED_NON_PRODUCTION"}

    @application.get("/sandbox/v1/state")
    def control_state(request: Request) -> dict[str, object]:
        _require_bearer(request, settings.control_token)
        return sandbox_state.snapshot()

    @application.post("/sandbox/v1/reset")
    def control_reset(request: Request) -> dict[str, object]:
        _require_bearer(request, settings.control_token)
        sandbox_state.reset()
        return sandbox_state.snapshot()

    @application.put("/sandbox/v1/faults/{provider}")
    def control_fault(
        provider: str,
        body: _FaultRequest,
        request: Request,
    ) -> dict[str, object]:
        _require_bearer(request, settings.control_token)
        _require_provider(provider)
        sandbox_state.configure_fault(provider, body.mode, body.remaining)
        return sandbox_state.snapshot()

    @application.delete("/sandbox/v1/faults/{provider}")
    def control_clear_fault(provider: str, request: Request) -> dict[str, object]:
        _require_bearer(request, settings.control_token)
        _require_provider(provider)
        sandbox_state.clear_fault(provider)
        return sandbox_state.snapshot()

    @application.post("/sandbox/v1/completions/{provider}/{operation_id}")
    def control_complete(
        provider: str,
        operation_id: str,
        body: _CompletionRequest,
        request: Request,
    ) -> dict[str, object]:
        _require_bearer(request, settings.control_token)
        return _send_completion(
            settings=settings,
            state=sandbox_state,
            provider=provider,
            operation_id=operation_id,
            completion=body,
        )

    @application.post("/sandbox/v1/procurement-fulfillments/{operation_id}")
    def control_fulfill(
        operation_id: str,
        body: _FulfillmentRequest,
        request: Request,
    ) -> dict[str, object]:
        _require_bearer(request, settings.control_token)
        return _send_procurement_fulfillment(
            settings=settings,
            state=sandbox_state,
            operation_id=operation_id,
            fulfillment=body,
            fallback_time=started_at,
        )

    @application.get("/v1/assets/{asset_id}")
    def asset(asset_id: str, request: Request) -> dict[str, object]:
        _require_bearer(request, settings.enterprise_token)
        _require_asset(asset_id)
        return {
            "source_record_id": f"sandbox-eam-{asset_id}",
            "as_of": _time(_read_time(sandbox_state, "eam", started_at)),
            "model_code": "PUMP-X100",
            "lifecycle_status": "ACTIVE",
        }

    @application.get("/v1/warranties/by-asset/{asset_id}")
    def warranty(asset_id: str, request: Request) -> dict[str, object]:
        _require_bearer(request, settings.enterprise_token)
        _require_asset(asset_id)
        return {
            "source_record_id": f"sandbox-warranty-{asset_id}",
            "as_of": _time(_read_time(sandbox_state, "eam", started_at)),
            "coverage": "PARTS_AND_LABOR",
            "valid": True,
        }

    @application.get("/v1/parts/availability")
    def parts_for_asset(request: Request, asset_id: str) -> dict[str, object]:
        _require_bearer(request, settings.enterprise_token)
        _require_asset(asset_id)
        return {
            "source_record_id": f"sandbox-wms-{asset_id}-{_PART_NUMBER}",
            "as_of": _time(_read_time(sandbox_state, "wms", started_at)),
            "part_number": _PART_NUMBER,
            "available_quantity": 24,
        }

    @application.get("/v1/schedules/availability")
    def schedule(request: Request, asset_id: str) -> dict[str, object]:
        _require_bearer(request, settings.enterprise_token)
        _require_asset(asset_id)
        return {
            "source_record_id": f"sandbox-fsm-schedule-{asset_id}",
            "as_of": _time(_read_time(sandbox_state, "fsm", started_at)),
            "site_id": _SITE_ID,
            "next_available_start": _time(started_at + timedelta(hours=2)),
            "next_available_end": _time(started_at + timedelta(hours=6)),
            "available_technician_count": 3,
        }

    @application.get("/v1/work-orders/history")
    def work_order_history(request: Request, asset_id: str) -> dict[str, object]:
        _require_bearer(request, settings.enterprise_token)
        _require_asset(asset_id)
        return {
            "source_record_id": f"sandbox-fsm-history-{asset_id}",
            "as_of": _time(_read_time(sandbox_state, "fsm", started_at)),
            "recent_count": 2,
            "latest_resolution": "Bearing replaced and alignment verified",
        }

    @application.get("/v1/work-orders/draft-template")
    def work_order_draft(request: Request, asset_id: str) -> dict[str, object]:
        _require_bearer(request, settings.enterprise_token)
        _require_asset(asset_id)
        return {
            "source_record_id": f"sandbox-fsm-draft-{asset_id}",
            "as_of": _time(_read_time(sandbox_state, "fsm", started_at)),
            "title": "Inspect pump vibration and bearing condition",
            "status": "DRAFT",
        }

    @application.get("/v1/parts/{part_number}/availability")
    def part_availability(
        part_number: str,
        request: Request,
        quantity: int = Query(ge=1, le=20),
    ) -> dict[str, object]:
        _require_bearer(request, settings.enterprise_token)
        if part_number != _PART_NUMBER:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "synthetic part not found")
        return {
            "source_record_id": f"sandbox-wms-part-{part_number}",
            "as_of": _time(_read_time(sandbox_state, "wms", started_at)),
            "part_number": part_number,
            "requested_quantity": quantity,
            "available_quantity": 24,
            "available": quantity <= 24,
        }

    @application.put("/v1/part-reservations/{operation_id}")
    def reserve(
        operation_id: str,
        body: _ReservationRequest,
        request: Request,
    ) -> JSONResponse:
        _require_bearer(request, settings.enterprise_token)
        _require_write_identity(request, operation_id, body.operation_id)
        response = {
            "reservation_id": _external_id("reservation", operation_id),
            "operation_id": operation_id,
            "part_number": body.part_number,
            "quantity": body.quantity,
            "source_record_id": _external_id("wms-record", operation_id),
            "as_of": _time(started_at),
        }
        decision = _decide(
            sandbox_state,
            "wms",
            operation_id,
            body.model_dump(mode="json"),
            response,
        )
        return _write_response(decision, created_status=status.HTTP_201_CREATED)

    @application.get("/v1/part-reservations/by-operation/{operation_id}")
    def reservation_by_operation(operation_id: str, request: Request) -> dict[str, object]:
        _require_bearer(request, settings.enterprise_token)
        return _reconciled(sandbox_state, "wms", operation_id)

    @application.put("/v1/part-issues/{operation_id}")
    def issue(
        operation_id: str,
        body: _PartIssueRequest,
        request: Request,
    ) -> JSONResponse:
        _require_bearer(request, settings.enterprise_token)
        _require_write_identity(request, operation_id, body.operation_id)
        response = {
            "issue_id": _external_id("issue", operation_id),
            "operation_id": operation_id,
            "reservation_id": body.reservation_id,
            "part_number": body.part_number,
            "quantity": body.quantity,
            "status": "ISSUED",
            "source": "enterprise-wms",
            "source_record_id": _external_id("wms-issue-record", operation_id),
            "as_of": _time(started_at),
        }
        decision = _decide(
            sandbox_state,
            "wms-issue",
            operation_id,
            body.model_dump(mode="json"),
            response,
        )
        return _write_response(decision, created_status=status.HTTP_201_CREATED)

    @application.get("/v1/part-issues/by-operation/{operation_id}")
    def issue_by_operation(operation_id: str, request: Request) -> dict[str, object]:
        _require_bearer(request, settings.enterprise_token)
        return _reconciled(sandbox_state, "wms-issue", operation_id)

    def movement_write(
        *,
        provider: str,
        operation_id: str,
        expected_kind: Literal["CONSUME", "RETURN"],
        body: _PartMovementRequest,
        request: Request,
    ) -> JSONResponse:
        _require_bearer(request, settings.enterprise_token)
        _require_write_identity(request, operation_id, body.operation_id)
        if body.movement_kind != expected_kind:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "movement kind mismatch")
        response = {
            "movement_id": _external_id("movement", operation_id),
            "operation_id": operation_id,
            "movement_kind": body.movement_kind,
            "part_issue_id": body.part_issue_id,
            "reservation_id": body.reservation_id,
            "part_number": body.part_number,
            "quantity": body.quantity,
            "status": "APPLIED",
            "source": "enterprise-wms",
            "source_record_id": _external_id("wms-movement-record", operation_id),
            "as_of": _time(started_at),
        }
        decision = _decide(
            sandbox_state,
            provider,
            operation_id,
            body.model_dump(mode="json"),
            response,
        )
        return _write_response(decision, created_status=status.HTTP_201_CREATED)

    @application.put("/v1/part-consumptions/{operation_id}")
    def consume(
        operation_id: str,
        body: _PartMovementRequest,
        request: Request,
    ) -> JSONResponse:
        return movement_write(
            provider="wms-consumption",
            operation_id=operation_id,
            expected_kind="CONSUME",
            body=body,
            request=request,
        )

    @application.get("/v1/part-consumptions/by-operation/{operation_id}")
    def consumption_by_operation(operation_id: str, request: Request) -> dict[str, object]:
        _require_bearer(request, settings.enterprise_token)
        return _reconciled(sandbox_state, "wms-consumption", operation_id)

    @application.put("/v1/part-returns/{operation_id}")
    def return_part(
        operation_id: str,
        body: _PartMovementRequest,
        request: Request,
    ) -> JSONResponse:
        return movement_write(
            provider="wms-return",
            operation_id=operation_id,
            expected_kind="RETURN",
            body=body,
            request=request,
        )

    @application.get("/v1/part-returns/by-operation/{operation_id}")
    def return_by_operation(operation_id: str, request: Request) -> dict[str, object]:
        _require_bearer(request, settings.enterprise_token)
        return _reconciled(sandbox_state, "wms-return", operation_id)

    @application.get("/v1/assignment-candidates")
    def assignment_candidates(
        request: Request,
        tenant_id: str,
        work_order_id: str,
        asset_id: str,
        site_id: str,
    ) -> dict[str, object]:
        _require_bearer(request, settings.fsm_token)
        _require_context(tenant_id=tenant_id, asset_id=asset_id, site_id=site_id)
        fact_time = _read_time(sandbox_state, "fsm", started_at)
        return {
            "candidates": [
                {
                    "candidate_profile_id": "sandbox-fsm-profile",
                    "profile_version": "v1",
                    "assignee_subject_id": _ENGINEER_ID,
                    "display_name": "M1 Synthetic Field Engineer",
                    "matched_skill_codes": ["PUMP", "VIBRATION"],
                    "service_window_start": _time(started_at + timedelta(hours=2)),
                    "service_window_end": _time(started_at + timedelta(hours=6)),
                    "travel_minutes": 20,
                    "remaining_work_minutes": 240,
                    "eligibility_code": "ELIGIBLE",
                    "source_record_id": f"sandbox-fsm-candidate-{work_order_id}",
                    "as_of": _time(fact_time),
                    "expires_at": _time(fact_time + timedelta(days=1)),
                }
            ]
        }

    @application.post("/v1/assignments")
    def assign(body: _AssignmentRequest, request: Request) -> JSONResponse:
        _require_bearer(request, settings.fsm_token)
        _require_write_identity(request, body.operation_id, body.operation_id)
        response = {
            "status": "ACCEPTED",
            "external_assignment_id": _external_id("fsm", body.operation_id),
            "reason": None,
            "occurred_at": _time(started_at),
        }
        decision = _decide(
            sandbox_state,
            "fsm",
            body.operation_id,
            body.model_dump(mode="json"),
            response,
        )
        return _write_response(decision, unknown_status="RECONCILING")

    @application.get("/v1/assignments/{operation_id}")
    def assignment_by_operation(
        operation_id: str, request: Request
    ) -> dict[str, object] | None:
        _require_bearer(request, settings.fsm_token)
        return _nullable_reconciled(sandbox_state, "fsm", operation_id)

    @application.get("/v1/procurement/profiles")
    def procurement_profile(request: Request, tenant_id: str) -> dict[str, object] | None:
        _require_bearer(request, settings.procurement_token)
        if tenant_id != settings.tenant_id:
            return None
        fact_time = _read_time(sandbox_state, "procurement", started_at)
        return {
            "profile_id": "sandbox-procurement-profile",
            "profile_version": "v1",
            "display_name": "M1 Synthetic ERP Procurement",
            "as_of": _time(fact_time),
            "expires_at": _time(fact_time + timedelta(days=1)),
        }

    @application.post("/v1/purchase-requisitions")
    def purchase_requisition(
        body: _ProcurementRequest, request: Request
    ) -> JSONResponse:
        _require_bearer(request, settings.procurement_token)
        _require_write_identity(request, body.operation_id, body.operation_id)
        response = {
            "status": "ACCEPTED",
            "external_request_id": _external_id("erp", body.operation_id),
            "reason": None,
            "occurred_at": _time(started_at),
        }
        decision = _decide(
            sandbox_state,
            "procurement",
            body.operation_id,
            body.model_dump(mode="json"),
            response,
        )
        return _write_response(decision, unknown_status="RECONCILING")

    @application.get("/v1/purchase-requisitions/{operation_id}")
    def purchase_requisition_by_operation(
        operation_id: str, request: Request
    ) -> dict[str, object] | None:
        _require_bearer(request, settings.procurement_token)
        return _nullable_reconciled(sandbox_state, "procurement", operation_id)

    @application.get("/v1/refund-profiles")
    def refund_profile(
        request: Request,
        tenant_id: str,
        customer_subject_id: str,
    ) -> dict[str, object] | None:
        _require_bearer(request, settings.refund_token)
        if tenant_id != settings.tenant_id or customer_subject_id != _CUSTOMER_ID:
            return None
        fact_time = _read_time(sandbox_state, "refund", started_at)
        return {
            "profile_id": "sandbox-refund-profile",
            "profile_version": "v1",
            "display_name": "M1 Synthetic Finance Refund",
            "as_of": _time(fact_time),
            "expires_at": _time(fact_time + timedelta(days=1)),
        }

    @application.post("/v1/refund-requests")
    def refund(body: _RefundRequest, request: Request) -> JSONResponse:
        _require_bearer(request, settings.refund_token)
        _require_write_identity(request, body.operation_id, body.operation_id)
        response = {
            "status": "ACCEPTED",
            "payment_status": "PROCESSING",
            "external_request_id": _external_id("refund", body.operation_id),
            "reason": None,
            "occurred_at": _time(started_at),
        }
        decision = _decide(
            sandbox_state,
            "refund",
            body.operation_id,
            body.model_dump(mode="json"),
            response,
        )
        return _write_response(decision, unknown_status="RECONCILING")

    @application.get("/v1/refund-requests/{operation_id}")
    def refund_by_operation(
        operation_id: str, request: Request
    ) -> dict[str, object] | None:
        _require_bearer(request, settings.refund_token)
        return _nullable_reconciled(sandbox_state, "refund", operation_id)

    def quotation_payload(
        *, incident_ref: str, asset_ref: str, diagnosis_ref: str
    ) -> dict[str, object]:
        fact_time = _read_time(sandbox_state, "quotation", started_at)
        return {
            "candidate_id": "sandbox-cpq-standard-repair",
            "source_version": "v1",
            "display_name": "Synthetic pump bearing repair",
            "incident_ref": incident_ref,
            "asset_ref": asset_ref,
            "diagnosis_ref": diagnosis_ref,
            "currency": "USD",
            "line_items": [
                {
                    "code": "LABOR",
                    "description": "Synthetic field service labor",
                    "quantity": 1,
                    "unit_price": "480.00",
                    "line_total": "480.00",
                },
                {
                    "code": "PART-BRG",
                    "description": "Synthetic bearing kit",
                    "quantity": 1,
                    "unit_price": "220.00",
                    "line_total": "220.00",
                },
            ],
            "subtotal": "700.00",
            "discount": "0.00",
            "tax": "80.00",
            "total": "780.00",
            "as_of": _time(fact_time),
            "expires_at": _time(fact_time + timedelta(days=1)),
        }

    @application.get("/v1/service-quotation-candidates")
    def quotation_candidates(
        request: Request,
        tenant_id: str,
        incident_ref: str,
        asset_ref: str,
        diagnosis_ref: str,
    ) -> dict[str, object]:
        _require_bearer(request, settings.quotation_token)
        if tenant_id != settings.tenant_id or asset_ref != _ASSET_ID:
            return {"candidates": []}
        return {
            "candidates": [
                quotation_payload(
                    incident_ref=incident_ref,
                    asset_ref=asset_ref,
                    diagnosis_ref=diagnosis_ref,
                )
            ]
        }

    @application.get("/v1/service-quotation-candidates/{candidate_id}")
    def quotation_candidate(
        candidate_id: str,
        request: Request,
        tenant_id: str,
        incident_ref: str,
        asset_ref: str,
        diagnosis_ref: str,
    ) -> dict[str, object]:
        _require_bearer(request, settings.quotation_token)
        if (
            tenant_id != settings.tenant_id
            or asset_ref != _ASSET_ID
            or candidate_id != "sandbox-cpq-standard-repair"
        ):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "quotation candidate not found")
        return quotation_payload(
            incident_ref=incident_ref,
            asset_ref=asset_ref,
            diagnosis_ref=diagnosis_ref,
        )

    @application.get("/v1/contacts")
    def contacts(
        request: Request,
        tenant_id: str,
        recipient_subject_id: str,
    ) -> dict[str, object]:
        _require_bearer(request, settings.notification_token)
        if tenant_id != settings.tenant_id or recipient_subject_id != _CUSTOMER_ID:
            return {"contacts": []}
        fact_time = _read_time(sandbox_state, "notification", started_at)
        expires_at = _time(fact_time + timedelta(days=1))
        common = {
            "source_system": "sandbox-customer-master",
            "source_version": "v1",
            "as_of": _time(fact_time),
            "verification_status": "VERIFIED",
            "consent_status": "OPTED_IN",
            "consent_basis": "Synthetic local test consent",
            "expires_at": expires_at,
        }
        return {
            "contacts": [
                {
                    **common,
                    "contact_point_id": "sandbox-contact-email",
                    "channel": "EMAIL",
                    "destination": "customer@example.invalid",
                    "masked_destination": "c***@example.invalid",
                    "source_record_id": "sandbox-contact-email-record",
                },
                {
                    **common,
                    "contact_point_id": "sandbox-contact-sms",
                    "channel": "SMS",
                    "destination": "+15550100101",
                    "masked_destination": "+1******0101",
                    "source_record_id": "sandbox-contact-sms-record",
                },
            ]
        }

    @application.post("/v1/deliveries")
    def deliver(body: _DeliveryRequest, request: Request) -> JSONResponse:
        _require_bearer(request, settings.notification_token)
        _require_write_identity(request, body.operation_id, body.operation_id)
        response = {
            "status": "DELIVERED",
            "provider_message_id": _external_id("notification", body.operation_id),
            "reason": None,
            "occurred_at": _time(started_at),
        }
        decision = _decide(
            sandbox_state,
            "notification",
            body.operation_id,
            body.model_dump(mode="json"),
            response,
        )
        return _write_response(decision, unknown_status="SUBMITTED")

    @application.get("/v1/deliveries/{operation_id}")
    def delivery_by_operation(
        operation_id: str, request: Request
    ) -> dict[str, object] | None:
        _require_bearer(request, settings.notification_token)
        return _nullable_reconciled(sandbox_state, "notification", operation_id)

    return application


def _require_bearer(request: Request, expected_token: str) -> None:
    authorization = request.headers.get("Authorization", "")
    expected = f"Bearer {expected_token}"
    if not hmac.compare_digest(authorization, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid provider credential")


def _require_write_identity(request: Request, path_id: str, body_id: str) -> None:
    idempotency_key = request.headers.get("Idempotency-Key")
    if not idempotency_key or idempotency_key != body_id or path_id != body_id:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Idempotency-Key, route operation_id, and body operation_id must match",
        )


def _require_asset(asset_id: str) -> None:
    if asset_id != _ASSET_ID:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "synthetic asset not found")


def _require_context(*, tenant_id: str, asset_id: str, site_id: str) -> None:
    if tenant_id != _TENANT_ID or asset_id != _ASSET_ID or site_id != _SITE_ID:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "synthetic context not found")


def _require_provider(provider: str) -> None:
    if provider not in _PROVIDERS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sandbox provider not found")


def _read_time(state: SandboxState, provider: str, current: datetime) -> datetime:
    fault = state.consume_fault(provider, _READ_FAULTS)
    if fault == "UNAVAILABLE":
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "simulated provider unavailable")
    return current - timedelta(days=2) if fault == "STALE" else current


def _send_completion(*, settings: SandboxSettings, state: SandboxState, provider: str,
                     operation_id: str, completion: _CompletionRequest) -> dict[str, object]:
    terminal = {"ACCEPTED", "REJECTED", "CANCELLED", "RECONCILING"}
    bindings = {
        "fsm": (terminal, "/api/v1/fsm-assignment-receipts/enterprise-fsm", "X-FSM-Signature",
                settings.fsm_webhook_secret, "external_assignment_id"),
        "procurement": (terminal, "/api/v1/purchase-requisition-receipts/enterprise-erp",
                        "X-Procurement-Signature", settings.procurement_webhook_secret, "external_request_id"),
        "refund": (terminal, "/api/v1/refund-request-receipts/enterprise-finance", "X-Refund-Signature",
                   settings.refund_webhook_secret, "external_request_id"),
        "notification": ({"SUBMITTED", "DELIVERED", "FAILED", "UNSUBSCRIBED"},
                         "/api/v1/customer-notification-receipts/enterprise-notify", "X-Notification-Signature",
                         settings.notification_webhook_secret, "provider_message_id"),
    }
    binding = bindings.get(provider)
    if binding is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "completion provider not found")
    allowed, path, header, secret, identity_key = binding
    if completion.status not in allowed:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "completion status invalid")
    operation = state.get(provider, operation_id)
    if operation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sandbox operation not found")
    response = operation.response
    outcome = completion.status
    payload: dict[str, object] = {
        "tenant_id": settings.tenant_id,
        "operation_id": operation_id,
        "status": completion.status,
        "occurred_at": _required_response_text(response, "occurred_at"),
        "reason": completion.reason,
        identity_key: _required_response_text(response, identity_key),
    }
    if provider == "fsm":
        payload.update(candidate_profile_id="sandbox-fsm-profile",
                       assignee_subject_id=_ENGINEER_ID)
    if provider == "refund":
        payment_status = completion.payment_status or "PAID"
        if payment_status not in {"NOT_STARTED", "PROCESSING", "PAID", "FAILED"}:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "refund payment status invalid"
            )
        outcome = f"{outcome}:{payment_status}"
        payload["payment_status"] = payment_status
    payload["provider_event_id"] = state.receipt_event_id(provider, operation_id, outcome)
    return _deliver_callback(settings, provider, operation_id, path, header, secret, payload)


def _send_procurement_fulfillment(*, settings: SandboxSettings, state: SandboxState,
                                  operation_id: str, fulfillment: _FulfillmentRequest,
                                  fallback_time: datetime) -> dict[str, object]:
    operation = state.get("procurement", operation_id)
    if operation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sandbox operation not found")
    outcome = f"{fulfillment.status}:{fulfillment.received_quantity}"
    payload: dict[str, object] = {
        "tenant_id": settings.tenant_id,
        "provider_event_id": state.receipt_event_id("procurement-fulfillment", operation_id, outcome),
        "operation_id": operation_id,
        "external_request_id": _required_response_text(operation.response, "external_request_id"),
        "external_order_id": _external_id("erp-order", operation_id),
        "status": fulfillment.status,
        "received_quantity": fulfillment.received_quantity,
        "expected_delivery_at": _time(fulfillment.expected_delivery_at or fallback_time + timedelta(days=1)),
        "occurred_at": _required_response_text(operation.response, "occurred_at"),
        "reason": fulfillment.reason,
    }
    return _deliver_callback(settings, "procurement-fulfillment", operation_id,
                             "/api/v1/purchase-fulfillment-receipts/enterprise-erp",
                             "X-Procurement-Signature", settings.procurement_webhook_secret, payload)


def _deliver_callback(settings: SandboxSettings, provider: str, operation_id: str,
                      path: str, signature_header: str, secret: str,
                      payload: Mapping[str, object]) -> dict[str, object]:
    raw_body = json.dumps(dict(payload), ensure_ascii=True, separators=(",", ":"),
                          sort_keys=True).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), raw_body, sha256).hexdigest()
    request = UrlRequest(  # noqa: S310 - local/test endpoint is explicitly configured
        f"{settings.platform_api_url}{path}",
        data=raw_body,
        headers={"Accept": "application/json", "Content-Type": "application/json",
                 signature_header: signature},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:  # noqa: S310
            platform_status = response.status
            response_body = response.read(64 * 1024 + 1)
    except HTTPError as exc:
        platform_status = exc.code
        response_body = exc.read(64 * 1024 + 1)
    except (URLError, TimeoutError, OSError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            "sandbox callback target unavailable") from exc
    if len(response_body) > 64 * 1024:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "sandbox callback response too large")
    return {
        "classification": "SIMULATED_NON_PRODUCTION",
        "provider": provider,
        "operation_id": operation_id,
        "platform_status": platform_status,
        "response_digest": sha256(response_body).hexdigest(),
    }


def _required_response_text(response: Mapping[str, object], key: str) -> str:
    value = response.get(key)
    if not isinstance(value, str) or not value:
        raise HTTPException(status.HTTP_409_CONFLICT, "sandbox operation identity incomplete")
    return value


def _decide(
    state: SandboxState,
    provider: str,
    operation_id: str,
    payload: Mapping[str, object],
    response: Mapping[str, object],
) -> WriteDecision:
    try:
        return state.decide_write(
            provider=provider,
            operation_id=operation_id,
            payload=payload,
            response=response,
        )
    except IdempotencyConflict as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "operation_id already belongs to a different canonical payload",
        ) from exc


def _write_response(
    decision: WriteDecision,
    *,
    created_status: int = status.HTTP_200_OK,
    unknown_status: str | None = None,
) -> JSONResponse:
    if decision.fault == "UNAVAILABLE":
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "simulated provider unavailable")
    if decision.fault == "REJECT":
        raise HTTPException(status.HTTP_409_CONFLICT, "simulated provider rejection")
    if decision.operation is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "sandbox write decision invalid")
    response = decision.operation.response
    if decision.fault == "OUTCOME_UNKNOWN":
        if unknown_status is None:
            return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=response)
        pending = {**response, "status": unknown_status}
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=pending)
    if decision.replayed:
        return JSONResponse(status_code=status.HTTP_200_OK, content=response)
    return JSONResponse(status_code=created_status, content=response)


def _reconciled(state: SandboxState, provider: str, operation_id: str) -> dict[str, object]:
    operation = state.get(provider, operation_id)
    if operation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sandbox operation not found")
    return operation.response


def _nullable_reconciled(
    state: SandboxState, provider: str, operation_id: str
) -> dict[str, object] | None:
    operation = state.get(provider, operation_id)
    return None if operation is None else operation.response


def _external_id(prefix: str, operation_id: str) -> str:
    digest = sha256(operation_id.encode("utf-8")).hexdigest()[:20]
    return f"sandbox-{prefix}-{digest}"


def _time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


app = create_app(SandboxSettings.from_environment())
