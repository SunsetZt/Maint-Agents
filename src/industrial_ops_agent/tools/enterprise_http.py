"""Strict anti-corruption adapter for an enterprise EAM/WMS/FSM integration gateway."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError
from typing import Any, Protocol, TypeVar, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from industrial_ops_agent.secrets import SecretValue
from industrial_ops_agent.tools.authority import attest_tool_fact, authoritative_source
from industrial_ops_agent.tools.contracts import (
    EnterpriseToolError,
    PartIssueOutcomeUnknown,
    PartIssueResult,
    PartMovementOutcomeUnknown,
    PartMovementResult,
    ReservationOutcomeUnknown,
    ReservationResult,
    ToolTransportBinding,
)

_MAX_RESPONSE_BYTES = 64 * 1024
_ModelT = TypeVar("_ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class EnterpriseHttpResponse:
    status_code: int
    body: bytes


class EnterpriseHttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> EnterpriseHttpResponse: ...


class EnterpriseTransportUnavailable(RuntimeError):
    pass


class UrllibEnterpriseHttpTransport:
    """One-request transport that does not retain credentials or response bodies."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> EnterpriseHttpResponse:
        body = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            if payload is not None
            else None
        )
        request_headers = {"Accept": "application/json", **headers}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        request = Request(  # noqa: S310 - base URL is validated by the adapter
            url,
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                return EnterpriseHttpResponse(response.status, _bounded_read(response))
        except HTTPError as exc:
            return EnterpriseHttpResponse(exc.code, _bounded_read(exc))
        except (URLError, TimeoutError, OSError) as exc:
            raise EnterpriseTransportUnavailable from exc


class _SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_record_id: str = Field(min_length=1, max_length=255)
    as_of: datetime

    @field_validator("as_of")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must include a timezone")
        return value


class _AssetResponse(_SourceRecord):
    model_code: str = Field(min_length=1, max_length=128)
    lifecycle_status: str = Field(min_length=1, max_length=64)


class _WarrantyResponse(_SourceRecord):
    coverage: str = Field(min_length=1, max_length=128)
    valid: bool


class _PartsByAssetResponse(_SourceRecord):
    part_number: str = Field(pattern=r"^[A-Z0-9-]{3,64}$")
    available_quantity: int = Field(ge=0, le=1_000_000)


class _ScheduleAvailabilityResponse(_SourceRecord):
    site_id: str = Field(min_length=1, max_length=255)
    next_available_start: datetime
    next_available_end: datetime
    available_technician_count: int = Field(ge=0, le=10_000)

    @field_validator("next_available_start", "next_available_end")
    @classmethod
    def require_window_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("schedule window must include a timezone")
        return value


class _WorkOrderHistoryResponse(_SourceRecord):
    recent_count: int = Field(ge=0, le=10_000)
    latest_resolution: str | None = Field(default=None, max_length=2_000)


class _WorkOrderDraftResponse(_SourceRecord):
    title: str = Field(min_length=1, max_length=255)
    status: str = Field(pattern="^DRAFT$")


_READONLY_RESPONSE_TYPES: dict[str, type[_SourceRecord]] = {
    "asset.get": _AssetResponse,
    "warranty.get": _WarrantyResponse,
    "parts.availability": _PartsByAssetResponse,
    "schedule.availability": _ScheduleAvailabilityResponse,
    "work_orders.history": _WorkOrderHistoryResponse,
    "work_order.draft": _WorkOrderDraftResponse,
}


class _AvailabilityResponse(_SourceRecord):
    part_number: str = Field(pattern=r"^[A-Z0-9-]{3,64}$")
    requested_quantity: int = Field(ge=1, le=20)
    available_quantity: int = Field(ge=0, le=1_000_000)
    available: bool


class _ReservationResponse(_SourceRecord):
    reservation_id: str = Field(min_length=1, max_length=255)
    operation_id: str = Field(min_length=1, max_length=255)
    part_number: str = Field(pattern=r"^[A-Z0-9-]{3,64}$")
    quantity: int = Field(ge=1, le=20)


class _PartIssueResponse(_SourceRecord):
    issue_id: str = Field(min_length=1, max_length=255)
    operation_id: str = Field(min_length=1, max_length=255)
    reservation_id: str = Field(min_length=1, max_length=255)
    part_number: str = Field(pattern=r"^[A-Z0-9-]{3,64}$")
    quantity: int = Field(ge=1, le=20)
    status: str = Field(pattern="^ISSUED$")
    source: str = Field(min_length=1, max_length=255)


class _PartMovementResponse(_SourceRecord):
    movement_id: str = Field(min_length=1, max_length=255)
    operation_id: str = Field(min_length=1, max_length=255)
    movement_kind: str = Field(pattern="^(CONSUME|RETURN)$")
    part_issue_id: str = Field(min_length=1, max_length=255)
    reservation_id: str = Field(min_length=1, max_length=255)
    part_number: str = Field(pattern=r"^[A-Z0-9-]{3,64}$")
    quantity: int = Field(ge=1, le=20)
    status: str = Field(pattern="^APPLIED$")
    source: str = Field(min_length=1, max_length=255)


class EnterpriseHttpAdapter:
    """Whitelisted tool operations over one authenticated enterprise gateway."""

    def __init__(
        self,
        base_url: str,
        api_token: SecretValue,
        *,
        timeout_seconds: float = 5.0,
        allow_plain_http: bool = False,
        transport: EnterpriseHttpTransport | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("enterprise tool gateway URL is invalid")
        if parsed.scheme != "https" and not (
            allow_plain_http
            and parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost", "enterprise-tool-gateway"}
        ):
            raise ValueError("enterprise tool gateway must use HTTPS")
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._timeout_seconds = timeout_seconds
        self._transport = transport or UrllibEnterpriseHttpTransport()

    def invoke(self, tool_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        asset_id = quote(str(parameters["asset_id"]), safe="")
        raw_asset_id = str(parameters["asset_id"])
        routes: dict[str, str] = {
            "asset.get": f"/v1/assets/{asset_id}",
            "warranty.get": (
                f"/v1/warranties/by-asset/{asset_id}"
            ),
            "parts.availability": (
                f"/v1/parts/availability?{urlencode({'asset_id': raw_asset_id})}"
            ),
            "schedule.availability": (
                f"/v1/schedules/availability?{urlencode({'asset_id': raw_asset_id})}"
            ),
            "work_orders.history": (
                f"/v1/work-orders/history?{urlencode({'asset_id': raw_asset_id})}"
            ),
            "work_order.draft": (
                f"/v1/work-orders/draft-template?{urlencode({'asset_id': raw_asset_id})}"
            ),
        }
        try:
            route = routes[tool_id]
        except KeyError as exc:
            raise EnterpriseToolError("enterprise_tool_not_supported") from exc
        response = self._request("GET", route)
        if response.status_code != 200:
            raise EnterpriseToolError("enterprise_tool_http_status")
        try:
            payload = json.loads(response.body)
        except (JSONDecodeError, UnicodeDecodeError) as exc:
            raise EnterpriseToolError("enterprise_tool_response_invalid") from exc
        if not isinstance(payload, dict):
            raise EnterpriseToolError("enterprise_tool_response_invalid")
        return normalize_enterprise_readonly_payload(tool_id, payload)

    def transport_binding(self, tool_id: str) -> ToolTransportBinding:
        if tool_id not in _READONLY_RESPONSE_TYPES:
            raise EnterpriseToolError("enterprise_tool_not_supported")
        return ToolTransportBinding(
            transport="ENTERPRISE_HTTP",
            local_tool_id=tool_id,
            remote_tool_name=tool_id,
            server_id="enterprise-tool-gateway",
            server_version="rest-v1",
        )

    def check_availability(self, *, part_number: str, quantity: int) -> dict[str, object]:
        encoded_part = quote(part_number, safe="")
        response = self._request(
            "GET",
            f"/v1/parts/{encoded_part}/availability?{urlencode({'quantity': quantity})}",
        )
        model = _decode_model(response, _AvailabilityResponse)
        if model.part_number != part_number or model.requested_quantity != quantity:
            raise EnterpriseToolError("enterprise_availability_identity_mismatch")
        return attest_tool_fact(
            "parts.availability",
            {
                **model.model_dump(mode="json"),
                "source": authoritative_source("parts.availability", "enterprise"),
            },
            profile="enterprise",
        )

    def reserve(
        self,
        *,
        operation_id: str,
        part_number: str,
        quantity: int,
    ) -> ReservationResult:
        try:
            response = self._request(
                "PUT",
                f"/v1/part-reservations/{quote(operation_id, safe='')}",
                payload={
                    "operation_id": operation_id,
                    "part_number": part_number,
                    "quantity": quantity,
                },
                idempotency_key=operation_id,
            )
        except EnterpriseToolError as exc:
            if exc.reason == "enterprise_tool_unavailable":
                raise ReservationOutcomeUnknown(operation_id) from exc
            raise
        if response.status_code == 202:
            raise ReservationOutcomeUnknown(operation_id)
        if response.status_code not in {200, 201}:
            raise EnterpriseToolError("enterprise_reservation_rejected")
        try:
            model = _decode_model(response, _ReservationResponse, expected_statuses={200, 201})
        except EnterpriseToolError as exc:
            raise ReservationOutcomeUnknown(operation_id) from exc
        if (
            model.operation_id != operation_id
            or model.part_number != part_number
            or model.quantity != quantity
        ):
            raise ReservationOutcomeUnknown(operation_id)
        return _reservation_result(model)

    def reconcile(self, operation_id: str) -> ReservationResult | None:
        response = self._request(
            "GET",
            f"/v1/part-reservations/by-operation/{quote(operation_id, safe='')}",
        )
        if response.status_code == 404:
            return None
        model = _decode_model(response, _ReservationResponse)
        if model.operation_id != operation_id:
            raise EnterpriseToolError("enterprise_reservation_identity_mismatch")
        return _reservation_result(model)

    def issue(
        self,
        *,
        operation_id: str,
        reservation_id: str,
        part_number: str,
        quantity: int,
    ) -> PartIssueResult:
        try:
            response = self._request(
                "PUT",
                f"/v1/part-issues/{quote(operation_id, safe='')}",
                payload={
                    "operation_id": operation_id,
                    "reservation_id": reservation_id,
                    "part_number": part_number,
                    "quantity": quantity,
                },
                idempotency_key=operation_id,
            )
        except EnterpriseToolError as exc:
            if exc.reason == "enterprise_tool_unavailable":
                raise PartIssueOutcomeUnknown(operation_id) from exc
            raise
        if response.status_code in {202, 408, 425, 429} or response.status_code >= 500:
            raise PartIssueOutcomeUnknown(operation_id)
        if 400 <= response.status_code < 500:
            raise EnterpriseToolError("enterprise_part_issue_rejected")
        if response.status_code not in {200, 201}:
            raise PartIssueOutcomeUnknown(operation_id)
        try:
            model = _decode_model(
                response,
                _PartIssueResponse,
                expected_statuses={200, 201},
            )
        except EnterpriseToolError as exc:
            raise PartIssueOutcomeUnknown(operation_id) from exc
        if (
            model.operation_id != operation_id
            or model.reservation_id != reservation_id
            or model.part_number != part_number
            or model.quantity != quantity
            or model.source != authoritative_source("parts.availability", "enterprise")
        ):
            raise PartIssueOutcomeUnknown(operation_id)
        return _part_issue_result(model)

    def reconcile_issue(self, operation_id: str) -> PartIssueResult | None:
        response = self._request(
            "GET",
            f"/v1/part-issues/by-operation/{quote(operation_id, safe='')}",
        )
        if response.status_code == 404:
            return None
        model = _decode_model(response, _PartIssueResponse)
        if model.operation_id != operation_id:
            raise EnterpriseToolError("enterprise_part_issue_identity_mismatch")
        return _part_issue_result(model)

    def apply_movement(
        self,
        *,
        operation_id: str,
        movement_kind: str,
        part_issue_id: str,
        reservation_id: str,
        part_number: str,
        quantity: int,
    ) -> PartMovementResult:
        route = _part_movement_route(movement_kind)
        try:
            response = self._request(
                "PUT",
                f"/v1/{route}/{quote(operation_id, safe='')}",
                payload={
                    "operation_id": operation_id,
                    "movement_kind": movement_kind,
                    "part_issue_id": part_issue_id,
                    "reservation_id": reservation_id,
                    "part_number": part_number,
                    "quantity": quantity,
                },
                idempotency_key=operation_id,
            )
        except EnterpriseToolError as exc:
            if exc.reason == "enterprise_tool_unavailable":
                raise PartMovementOutcomeUnknown(operation_id) from exc
            raise
        if response.status_code in {202, 408, 425, 429} or response.status_code >= 500:
            raise PartMovementOutcomeUnknown(operation_id)
        if 400 <= response.status_code < 500:
            raise EnterpriseToolError("enterprise_part_movement_rejected")
        if response.status_code not in {200, 201}:
            raise PartMovementOutcomeUnknown(operation_id)
        try:
            model = _decode_model(
                response,
                _PartMovementResponse,
                expected_statuses={200, 201},
            )
        except EnterpriseToolError as exc:
            raise PartMovementOutcomeUnknown(operation_id) from exc
        if not _part_movement_identity_matches(
            model,
            operation_id=operation_id,
            movement_kind=movement_kind,
            part_issue_id=part_issue_id,
            reservation_id=reservation_id,
            part_number=part_number,
            quantity=quantity,
        ):
            raise PartMovementOutcomeUnknown(operation_id)
        return _part_movement_result(model)

    def reconcile_movement(
        self,
        operation_id: str,
        movement_kind: str,
    ) -> PartMovementResult | None:
        route = _part_movement_route(movement_kind)
        response = self._request(
            "GET",
            f"/v1/{route}/by-operation/{quote(operation_id, safe='')}",
        )
        if response.status_code == 404:
            return None
        model = _decode_model(response, _PartMovementResponse)
        if model.operation_id != operation_id or model.movement_kind != movement_kind:
            raise EnterpriseToolError("enterprise_part_movement_identity_mismatch")
        return _part_movement_result(model)

    def _request(
        self,
        method: str,
        route: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> EnterpriseHttpResponse:
        headers = {"Authorization": f"Bearer {self._api_token.reveal()}"}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        try:
            return self._transport.request(
                method,
                f"{self._base_url}{route}",
                headers=headers,
                payload=payload,
                timeout_seconds=self._timeout_seconds,
            )
        except EnterpriseTransportUnavailable as exc:
            raise EnterpriseToolError("enterprise_tool_unavailable") from exc


def _decode_model(
    response: EnterpriseHttpResponse,
    model_type: type[_ModelT],
    *,
    expected_statuses: set[int] | None = None,
) -> _ModelT:
    if response.status_code not in (expected_statuses or {200}):
        raise EnterpriseToolError("enterprise_tool_http_status")
    try:
        payload = json.loads(response.body)
        if not isinstance(payload, dict):
            raise TypeError
        return model_type.model_validate(payload)
    except (JSONDecodeError, UnicodeDecodeError, TypeError, ValidationError) as exc:
        raise EnterpriseToolError("enterprise_tool_response_invalid") from exc


def normalize_enterprise_readonly_payload(
    tool_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate an untrusted enterprise payload before authority attestation."""

    try:
        response_type = _READONLY_RESPONSE_TYPES[tool_id]
    except KeyError as exc:
        raise EnterpriseToolError("enterprise_tool_not_supported") from exc
    try:
        model = response_type.model_validate(payload)
    except ValidationError as exc:
        raise EnterpriseToolError("enterprise_tool_response_invalid") from exc
    return attest_tool_fact(
        tool_id,
        {
            **model.model_dump(mode="json"),
            "source": authoritative_source(tool_id, "enterprise"),
        },
        profile="enterprise",
    )


def _reservation_result(model: _ReservationResponse) -> ReservationResult:
    return ReservationResult(
        reservation_id=model.reservation_id,
        operation_id=model.operation_id,
        part_number=model.part_number,
        quantity=model.quantity,
        source=authoritative_source("parts.availability", "enterprise"),
        source_record_id=model.source_record_id,
        as_of=model.as_of,
    )


def _part_issue_result(model: _PartIssueResponse) -> PartIssueResult:
    expected_source = authoritative_source("parts.availability", "enterprise")
    if model.source != expected_source:
        raise EnterpriseToolError("enterprise_part_issue_source_mismatch")
    return PartIssueResult(
        issue_id=model.issue_id,
        operation_id=model.operation_id,
        reservation_id=model.reservation_id,
        part_number=model.part_number,
        quantity=model.quantity,
        status=model.status,
        source=model.source,
        source_record_id=model.source_record_id,
        as_of=model.as_of,
    )


def _part_movement_route(movement_kind: str) -> str:
    try:
        return {
            "CONSUME": "part-consumptions",
            "RETURN": "part-returns",
        }[movement_kind]
    except KeyError as exc:
        raise EnterpriseToolError("enterprise_part_movement_kind_invalid") from exc


def _part_movement_identity_matches(
    model: _PartMovementResponse,
    *,
    operation_id: str,
    movement_kind: str,
    part_issue_id: str,
    reservation_id: str,
    part_number: str,
    quantity: int,
) -> bool:
    return (
        model.operation_id == operation_id
        and model.movement_kind == movement_kind
        and model.part_issue_id == part_issue_id
        and model.reservation_id == reservation_id
        and model.part_number == part_number
        and model.quantity == quantity
        and model.source == authoritative_source("parts.availability", "enterprise")
    )


def _part_movement_result(model: _PartMovementResponse) -> PartMovementResult:
    if model.source != authoritative_source("parts.availability", "enterprise"):
        raise EnterpriseToolError("enterprise_part_movement_source_mismatch")
    return PartMovementResult(
        movement_id=model.movement_id,
        operation_id=model.operation_id,
        movement_kind=model.movement_kind,
        part_issue_id=model.part_issue_id,
        reservation_id=model.reservation_id,
        part_number=model.part_number,
        quantity=model.quantity,
        status=model.status,
        source=model.source,
        source_record_id=model.source_record_id,
        as_of=model.as_of,
    )


def _bounded_read(response: Any) -> bytes:
    body = cast(bytes, response.read(_MAX_RESPONSE_BYTES + 1))
    if len(body) > _MAX_RESPONSE_BYTES:
        raise EnterpriseTransportUnavailable
    return body
