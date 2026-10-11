"""Provider-neutral HTTPS adapter for governed enterprise refund delivery."""

from __future__ import annotations

import hmac
from datetime import datetime
from hashlib import sha256
from urllib.parse import urlencode, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from industrial_ops_agent.secrets import SecretValue
from industrial_ops_agent.tools.contracts import (
    EnterpriseToolError,
    RefundDeliveryResult,
    RefundProfile,
    RefundReceipt,
)
from industrial_ops_agent.tools.enterprise_http import (
    EnterpriseHttpTransport,
    EnterpriseTransportUnavailable,
    UrllibEnterpriseHttpTransport,
)


class _ProfileResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1, max_length=128)
    profile_version: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=128)
    as_of: datetime
    expires_at: datetime

    @field_validator("as_of", "expires_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("refund profile time must include timezone")
        return value


class _DeliveryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(pattern=r"^(ACCEPTED|REJECTED|CANCELLED|RECONCILING)$")
    payment_status: str = Field(
        pattern=r"^(NOT_STARTED|PROCESSING|PAID|FAILED)$"
    )
    external_request_id: str | None = Field(default=None, min_length=1, max_length=255)
    reason: str | None = Field(default=None, max_length=255)
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("refund result time must include timezone")
        return value


class _Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    provider_event_id: str = Field(min_length=1, max_length=255)
    operation_id: str = Field(min_length=1, max_length=128)
    external_request_id: str = Field(min_length=1, max_length=255)
    status: str = Field(pattern=r"^(RECONCILING|ACCEPTED|REJECTED|CANCELLED)$")
    payment_status: str = Field(
        pattern=r"^(NOT_STARTED|PROCESSING|PAID|FAILED)$"
    )
    occurred_at: datetime
    reason: str | None = Field(default=None, max_length=255)

    @field_validator("occurred_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("refund receipt time must include timezone")
        return value


class HttpRefundDeliveryAdapter:
    """Strict finance boundary; secrets and payee mappings never leave the adapter."""

    def __init__(
        self,
        base_url: str,
        *,
        provider_id: str,
        api_token: SecretValue,
        webhook_secret: SecretValue,
        timeout_seconds: float = 5.0,
        allow_plain_http: bool = False,
        transport: EnterpriseHttpTransport | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        allowed_schemes = {"http", "https"} if allow_plain_http else {"https"}
        if not parsed.netloc or parsed.scheme not in allowed_schemes:
            raise ValueError("refund provider URL must use an allowed HTTP scheme")
        if (
            not provider_id
            or len(provider_id) > 128
            or not all(
                char.isascii() and (char.isalnum() or char in "._-")
                for char in provider_id
            )
        ):
            raise ValueError("refund provider ID is invalid")
        self._base_url = base_url.rstrip("/")
        self.provider_id = provider_id
        self._api_token = api_token
        self._webhook_secret = webhook_secret
        self._timeout_seconds = timeout_seconds
        self._transport = transport or UrllibEnterpriseHttpTransport()

    def resolve_profile(
        self, *, tenant_id: str, customer_subject_id: str
    ) -> RefundProfile | None:
        query = urlencode(
            {"tenant_id": tenant_id, "customer_subject_id": customer_subject_id}
        )
        payload = self._request("GET", f"/v1/refund-profiles?{query}", None)
        if payload == b"null":
            return None
        try:
            parsed = _ProfileResponse.model_validate_json(payload)
        except ValidationError as exc:
            raise EnterpriseToolError("enterprise_refund_profile_invalid") from exc
        return RefundProfile(
            provider=self.provider_id,
            profile_id=parsed.profile_id,
            profile_version=parsed.profile_version,
            display_name=parsed.display_name,
            as_of=parsed.as_of,
            expires_at=parsed.expires_at,
        )

    def submit(
        self,
        *,
        operation_id: str,
        profile: RefundProfile,
        payload: dict[str, object],
    ) -> RefundDeliveryResult:
        if profile.provider != self.provider_id:
            raise EnterpriseToolError("enterprise_refund_profile_invalid")
        response = self._request(
            "POST",
            "/v1/refund-requests",
            {
                "operation_id": operation_id,
                "profile_id": profile.profile_id,
                "profile_version": profile.profile_version,
                "refund_request": payload,
            },
            idempotency_key=operation_id,
        )
        return self._delivery_result(response)

    def reconcile(self, operation_id: str) -> RefundDeliveryResult | None:
        payload = self._request("GET", f"/v1/refund-requests/{operation_id}", None)
        if payload == b"null":
            return None
        return self._delivery_result(payload)

    def verify_receipt(self, raw_body: bytes, signature: str) -> RefundReceipt:
        expected = hmac.new(
            self._webhook_secret.reveal().encode(), raw_body, sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise EnterpriseToolError("refund_receipt_signature_invalid")
        try:
            parsed = _Receipt.model_validate_json(raw_body)
        except ValidationError as exc:
            raise EnterpriseToolError("refund_receipt_invalid") from exc
        return RefundReceipt(
            tenant_id=parsed.tenant_id,
            provider_event_id=parsed.provider_event_id,
            operation_id=parsed.operation_id,
            external_request_id=parsed.external_request_id,
            status=parsed.status,
            payment_status=parsed.payment_status,
            occurred_at=parsed.occurred_at,
            reason=parsed.reason,
            signature_digest=sha256(signature.encode()).hexdigest(),
        )

    @staticmethod
    def _delivery_result(payload: bytes) -> RefundDeliveryResult:
        try:
            parsed = _DeliveryResponse.model_validate_json(payload)
        except ValidationError as exc:
            raise EnterpriseToolError("refund_provider_invalid_response") from exc
        return RefundDeliveryResult(
            status=parsed.status,
            payment_status=parsed.payment_status,
            external_request_id=parsed.external_request_id,
            reason=parsed.reason,
            occurred_at=parsed.occurred_at,
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
        *,
        idempotency_key: str | None = None,
    ) -> bytes:
        headers = {"Authorization": f"Bearer {self._api_token.reveal()}"}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        try:
            response = self._transport.request(
                method,
                f"{self._base_url}{path}",
                headers=headers,
                payload=payload,
                timeout_seconds=self._timeout_seconds,
            )
        except EnterpriseTransportUnavailable as exc:
            raise EnterpriseToolError("refund_provider_unavailable") from exc
        if response.status_code == 404 and method == "GET":
            return b"null"
        if response.status_code < 200 or response.status_code >= 300:
            raise EnterpriseToolError("refund_provider_rejected")
        return response.body
