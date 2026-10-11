"""Provider-neutral HTTPS adapter for governed enterprise FSM assignment."""

from __future__ import annotations

import hmac
from datetime import datetime
from hashlib import sha256
from urllib.parse import urlencode, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from industrial_ops_agent.secrets import SecretValue
from industrial_ops_agent.tools.contracts import (
    EnterpriseToolError,
    FsmAssignmentCandidate,
    FsmAssignmentDeliveryResult,
    FsmAssignmentReceipt,
)
from industrial_ops_agent.tools.enterprise_http import (
    EnterpriseHttpTransport,
    EnterpriseTransportUnavailable,
    UrllibEnterpriseHttpTransport,
)


class _CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_profile_id: str = Field(min_length=1, max_length=128)
    profile_version: str = Field(min_length=1, max_length=128)
    assignee_subject_id: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=128)
    matched_skill_codes: tuple[str, ...] = Field(min_length=1, max_length=32)
    service_window_start: datetime
    service_window_end: datetime
    travel_minutes: int = Field(ge=0, le=10080)
    remaining_work_minutes: int = Field(ge=0, le=10080)
    eligibility_code: str = Field(pattern=r"^ELIGIBLE$")
    source_record_id: str = Field(min_length=1, max_length=255)
    as_of: datetime
    expires_at: datetime

    @field_validator(
        "service_window_start",
        "service_window_end",
        "as_of",
        "expires_at",
    )
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("FSM candidate time must include timezone")
        return value


class _CandidateListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: tuple[_CandidateResponse, ...]


class _DeliveryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(pattern=r"^(ACCEPTED|REJECTED|CANCELLED|RECONCILING)$")
    external_assignment_id: str | None = Field(
        default=None, min_length=1, max_length=255
    )
    reason: str | None = Field(default=None, max_length=255)
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("FSM assignment result time must include timezone")
        return value


class _Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    provider_event_id: str = Field(min_length=1, max_length=255)
    operation_id: str = Field(min_length=1, max_length=128)
    candidate_profile_id: str = Field(min_length=1, max_length=128)
    assignee_subject_id: str = Field(min_length=1, max_length=128)
    external_assignment_id: str = Field(min_length=1, max_length=255)
    status: str = Field(pattern=r"^(RECONCILING|ACCEPTED|REJECTED|CANCELLED)$")
    occurred_at: datetime
    reason: str | None = Field(default=None, max_length=255)

    @field_validator("occurred_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("FSM assignment receipt time must include timezone")
        return value


class HttpFsmAssignmentAdapter:
    """Strict FSM boundary; credentials and provider-private HR facts stay internal."""

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
            raise ValueError("FSM provider URL must use an allowed HTTP scheme")
        if not provider_id or len(provider_id) > 128 or not all(
            char.isascii() and (char.isalnum() or char in "._-")
            for char in provider_id
        ):
            raise ValueError("FSM provider ID is invalid")
        self._base_url = base_url.rstrip("/")
        self.provider_id = provider_id
        self._api_token = api_token
        self._webhook_secret = webhook_secret
        self._timeout_seconds = timeout_seconds
        self._transport = transport or UrllibEnterpriseHttpTransport()

    def list_candidates(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
        asset_id: str,
        site_id: str,
        service_window_start: datetime | None,
        service_window_end: datetime | None,
    ) -> tuple[FsmAssignmentCandidate, ...]:
        query_values = {
            "tenant_id": tenant_id,
            "work_order_id": work_order_id,
            "asset_id": asset_id,
            "site_id": site_id,
        }
        if service_window_start is not None:
            query_values["service_window_start"] = service_window_start.isoformat()
        if service_window_end is not None:
            query_values["service_window_end"] = service_window_end.isoformat()
        payload = self._request(
            "GET", f"/v1/assignment-candidates?{urlencode(query_values)}", None
        )
        try:
            parsed = _CandidateListResponse.model_validate_json(payload)
        except ValidationError as exc:
            raise EnterpriseToolError("fsm_assignment_candidates_invalid") from exc
        return tuple(
            FsmAssignmentCandidate(
                provider=self.provider_id,
                candidate_profile_id=item.candidate_profile_id,
                profile_version=item.profile_version,
                assignee_subject_id=item.assignee_subject_id,
                display_name=item.display_name,
                matched_skill_codes=item.matched_skill_codes,
                service_window_start=item.service_window_start,
                service_window_end=item.service_window_end,
                travel_minutes=item.travel_minutes,
                remaining_work_minutes=item.remaining_work_minutes,
                eligibility_code=item.eligibility_code,
                source_record_id=item.source_record_id,
                as_of=item.as_of,
                expires_at=item.expires_at,
            )
            for item in parsed.candidates
        )

    def submit(
        self,
        *,
        operation_id: str,
        candidate: FsmAssignmentCandidate,
        payload: dict[str, object],
    ) -> FsmAssignmentDeliveryResult:
        if candidate.provider != self.provider_id:
            raise EnterpriseToolError("fsm_assignment_candidate_invalid")
        response = self._request(
            "POST",
            "/v1/assignments",
            {
                "operation_id": operation_id,
                "candidate_profile_id": candidate.candidate_profile_id,
                "profile_version": candidate.profile_version,
                "assignment": payload,
            },
            idempotency_key=operation_id,
        )
        return self._delivery_result(response)

    def reconcile(self, operation_id: str) -> FsmAssignmentDeliveryResult | None:
        payload = self._request("GET", f"/v1/assignments/{operation_id}", None)
        if payload == b"null":
            return None
        return self._delivery_result(payload)

    def verify_receipt(self, raw_body: bytes, signature: str) -> FsmAssignmentReceipt:
        expected = hmac.new(
            self._webhook_secret.reveal().encode(), raw_body, sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise EnterpriseToolError("fsm_assignment_receipt_signature_invalid")
        try:
            parsed = _Receipt.model_validate_json(raw_body)
        except ValidationError as exc:
            raise EnterpriseToolError("fsm_assignment_receipt_invalid") from exc
        return FsmAssignmentReceipt(
            tenant_id=parsed.tenant_id,
            provider_event_id=parsed.provider_event_id,
            operation_id=parsed.operation_id,
            candidate_profile_id=parsed.candidate_profile_id,
            assignee_subject_id=parsed.assignee_subject_id,
            external_assignment_id=parsed.external_assignment_id,
            status=parsed.status,
            occurred_at=parsed.occurred_at,
            reason=parsed.reason,
            signature_digest=sha256(signature.encode()).hexdigest(),
        )

    @staticmethod
    def _delivery_result(payload: bytes) -> FsmAssignmentDeliveryResult:
        try:
            parsed = _DeliveryResponse.model_validate_json(payload)
        except ValidationError as exc:
            raise EnterpriseToolError("fsm_assignment_provider_invalid_response") from exc
        return FsmAssignmentDeliveryResult(
            status=parsed.status,
            external_assignment_id=parsed.external_assignment_id,
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
            raise EnterpriseToolError("fsm_assignment_provider_unavailable") from exc
        if response.status_code == 404 and method == "GET":
            return b"null"
        if response.status_code < 200 or response.status_code >= 300:
            raise EnterpriseToolError("fsm_assignment_provider_rejected")
        return response.body
