"""Provider-neutral HTTP customer notification adapter with redacted boundaries."""

from __future__ import annotations

import hmac
from datetime import datetime
from hashlib import sha256
from urllib.parse import urlencode, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from industrial_ops_agent.secrets import SecretValue
from industrial_ops_agent.tools.contracts import (
    EnterpriseToolError,
    NotificationContact,
    NotificationDeliveryResult,
    NotificationDestination,
    NotificationReceipt,
)
from industrial_ops_agent.tools.enterprise_http import (
    EnterpriseHttpTransport,
    EnterpriseTransportUnavailable,
    UrllibEnterpriseHttpTransport,
)


class _Contact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact_point_id: str = Field(min_length=1, max_length=128)
    channel: str = Field(pattern=r"^(EMAIL|SMS)$")
    destination: str = Field(min_length=1, max_length=320)
    masked_destination: str = Field(min_length=3, max_length=320)
    source_system: str = Field(min_length=1, max_length=128)
    source_record_id: str = Field(min_length=1, max_length=128)
    source_version: str = Field(min_length=1, max_length=128)
    as_of: datetime
    verification_status: str = Field(pattern=r"^(VERIFIED|UNVERIFIED)$")
    consent_status: str = Field(pattern=r"^(OPTED_IN|OPTED_OUT)$")
    consent_basis: str = Field(min_length=1, max_length=255)
    expires_at: datetime | None = None

    @field_validator("as_of", "expires_at")
    @classmethod
    def timezone_required(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("notification authority time must include timezone")
        return value


class _Contacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contacts: list[_Contact] = Field(max_length=20)


class _DeliveryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = Field(pattern=r"^(SUBMITTED|DELIVERED|FAILED|UNSUBSCRIBED)$")
    provider_message_id: str = Field(min_length=1, max_length=255)
    reason: str | None = Field(default=None, max_length=255)
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("notification result time must include timezone")
        return value


class _Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    provider_event_id: str = Field(min_length=1, max_length=255)
    operation_id: str = Field(min_length=1, max_length=128)
    provider_message_id: str = Field(min_length=1, max_length=255)
    status: str = Field(pattern=r"^(SUBMITTED|DELIVERED|FAILED|UNSUBSCRIBED)$")
    occurred_at: datetime
    reason: str | None = Field(default=None, max_length=255)

    @field_validator("occurred_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("notification receipt time must include timezone")
        return value


class HttpNotificationDeliveryAdapter:
    """Strict HTTP adapter; raw destinations exist only in contact/deliver calls."""

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
            raise ValueError("notification provider URL must use an allowed HTTP scheme")
        if not provider_id or len(provider_id) > 128:
            raise ValueError("notification provider ID is invalid")
        self._base_url = base_url.rstrip("/")
        self.provider_id = provider_id
        self._api_token = api_token
        self._webhook_secret = webhook_secret
        self._timeout_seconds = timeout_seconds
        self._transport = transport or UrllibEnterpriseHttpTransport()

    def resolve_contacts(
        self,
        *,
        tenant_id: str,
        recipient_subject_id: str,
    ) -> tuple[NotificationContact, ...]:
        query = urlencode(
            {"tenant_id": tenant_id, "recipient_subject_id": recipient_subject_id}
        )
        payload = self._request("GET", f"/v1/contacts?{query}", None)
        try:
            parsed = _Contacts.model_validate_json(payload)
        except ValidationError as exc:
            raise EnterpriseToolError("contact_authority_invalid_response") from exc
        return tuple(
            NotificationContact(
                contact_point_id=item.contact_point_id,
                channel=item.channel,
                destination=NotificationDestination(item.destination),
                masked_destination=item.masked_destination,
                source_system=item.source_system,
                source_record_id=item.source_record_id,
                source_version=item.source_version,
                as_of=item.as_of,
                verification_status=item.verification_status,
                consent_status=item.consent_status,
                consent_basis=item.consent_basis,
                expires_at=item.expires_at,
            )
            for item in parsed.contacts
        )

    def deliver(
        self,
        *,
        operation_id: str,
        contact: NotificationContact,
        rendered_message: str,
    ) -> NotificationDeliveryResult:
        payload = self._request(
            "POST",
            "/v1/deliveries",
            {
                "operation_id": operation_id,
                "channel": contact.channel,
                "destination": contact.destination.reveal(),
                "message": rendered_message,
            },
            idempotency_key=operation_id,
        )
        return self._delivery_result(payload)

    def reconcile(self, operation_id: str) -> NotificationDeliveryResult | None:
        payload = self._request("GET", f"/v1/deliveries/{operation_id}", None)
        if payload == b"null":
            return None
        return self._delivery_result(payload)

    def verify_receipt(self, raw_body: bytes, signature: str) -> NotificationReceipt:
        expected = hmac.new(self._webhook_secret.reveal().encode(), raw_body, sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise EnterpriseToolError("notification_receipt_signature_invalid")
        try:
            parsed = _Receipt.model_validate_json(raw_body)
        except ValidationError as exc:
            raise EnterpriseToolError("notification_receipt_invalid") from exc
        return NotificationReceipt(
            tenant_id=parsed.tenant_id,
            provider_event_id=parsed.provider_event_id,
            operation_id=parsed.operation_id,
            provider_message_id=parsed.provider_message_id,
            status=parsed.status,
            occurred_at=parsed.occurred_at,
            reason=parsed.reason,
            signature_digest=sha256(signature.encode()).hexdigest(),
        )

    def _delivery_result(self, payload: bytes) -> NotificationDeliveryResult:
        try:
            parsed = _DeliveryResponse.model_validate_json(payload)
        except ValidationError as exc:
            raise EnterpriseToolError("notification_provider_invalid_response") from exc
        return NotificationDeliveryResult(
            parsed.status,
            parsed.provider_message_id,
            parsed.reason,
            parsed.occurred_at,
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, str] | None,
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
            raise EnterpriseToolError("notification_provider_unavailable") from exc
        if response.status_code == 404 and method == "GET" and "/deliveries/" in path:
            return b"null"
        if response.status_code < 200 or response.status_code >= 300:
            raise EnterpriseToolError("notification_provider_rejected")
        return response.body
