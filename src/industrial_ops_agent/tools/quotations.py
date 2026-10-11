"""Strict provider-neutral HTTPS boundary for enterprise service quotations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from urllib.parse import quote, urlencode, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from industrial_ops_agent.secrets import SecretValue
from industrial_ops_agent.tools.contracts import (
    EnterpriseToolError,
    ServiceQuotationCandidate,
    ServiceQuotationLineItem,
)
from industrial_ops_agent.tools.enterprise_http import (
    EnterpriseHttpTransport,
    EnterpriseTransportUnavailable,
    UrllibEnterpriseHttpTransport,
)

_FUTURE_TOLERANCE = timedelta(minutes=5)
_MONEY_LIMIT = Decimal("999999999999.99")


class _LineItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    description: str = Field(min_length=1, max_length=255)
    quantity: int = Field(ge=1, le=10000)
    unit_price: str = Field(pattern=r"^(?:0|[1-9][0-9]{0,11})\.[0-9]{2}$")
    line_total: str = Field(pattern=r"^(?:0|[1-9][0-9]{0,11})\.[0-9]{2}$")


class _CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    source_version: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=128)
    incident_ref: str = Field(min_length=1, max_length=128)
    asset_ref: str = Field(min_length=1, max_length=128)
    diagnosis_ref: str = Field(min_length=1, max_length=128)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    line_items: tuple[_LineItemResponse, ...] = Field(min_length=1, max_length=100)
    subtotal: str = Field(pattern=r"^(?:0|[1-9][0-9]{0,11})\.[0-9]{2}$")
    discount: str = Field(pattern=r"^(?:0|[1-9][0-9]{0,11})\.[0-9]{2}$")
    tax: str = Field(pattern=r"^(?:0|[1-9][0-9]{0,11})\.[0-9]{2}$")
    total: str = Field(pattern=r"^(?:0|[1-9][0-9]{0,11})\.[0-9]{2}$")
    as_of: datetime
    expires_at: datetime

    @field_validator("as_of", "expires_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("CPQ candidate time must include timezone")
        return value


class _CandidateListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: tuple[_CandidateResponse, ...] = Field(max_length=100)


def validate_service_quotation_candidate(
    candidate: object,
    *,
    provider_id: str,
    incident_ref: str,
    asset_ref: str,
    diagnosis_ref: str,
    now: datetime,
) -> ServiceQuotationCandidate:
    """Normalize and validate a candidate received from any adapter implementation."""

    if not isinstance(candidate, ServiceQuotationCandidate):
        raise EnterpriseToolError("service_quotation_candidate_invalid")
    identifiers = (
        candidate.provider,
        candidate.candidate_id,
        candidate.source_version,
        candidate.display_name,
        candidate.incident_ref,
        candidate.asset_ref,
        candidate.diagnosis_ref,
    )
    if any(not value or len(value) > 128 or not value.isprintable() for value in identifiers):
        raise EnterpriseToolError("service_quotation_candidate_invalid")
    if (
        candidate.provider != provider_id
        or candidate.incident_ref != incident_ref
        or candidate.asset_ref != asset_ref
        or candidate.diagnosis_ref != diagnosis_ref
        or len(candidate.currency) != 3
        or not candidate.currency.isascii()
        or not candidate.currency.isalpha()
        or candidate.currency != candidate.currency.upper()
        or not 1 <= len(candidate.line_items) <= 100
    ):
        raise EnterpriseToolError("service_quotation_candidate_invalid")
    normalized_items: list[ServiceQuotationLineItem] = []
    line_sum = Decimal("0.00")
    for item in candidate.line_items:
        if (
            not item.code
            or len(item.code) > 64
            or not item.description
            or len(item.description) > 255
            or not item.code.isprintable()
            or not item.description.isprintable()
            or item.quantity < 1
            or item.quantity > 10000
        ):
            raise EnterpriseToolError("service_quotation_candidate_invalid")
        unit_price = _money(item.unit_price)
        line_total = _money(item.line_total)
        if unit_price * item.quantity != line_total:
            raise EnterpriseToolError("service_quotation_amounts_invalid")
        line_sum += line_total
        normalized_items.append(
            ServiceQuotationLineItem(
                code=item.code,
                description=item.description,
                quantity=item.quantity,
                unit_price=_money_text(unit_price),
                line_total=_money_text(line_total),
            )
        )
    subtotal = _money(candidate.subtotal)
    discount = _money(candidate.discount)
    tax = _money(candidate.tax)
    total = _money(candidate.total)
    if line_sum != subtotal or subtotal - discount + tax != total:
        raise EnterpriseToolError("service_quotation_amounts_invalid")
    as_of = _aware(candidate.as_of)
    expires_at = _aware(candidate.expires_at)
    if as_of > now + _FUTURE_TOLERANCE or expires_at <= now or as_of >= expires_at:
        raise EnterpriseToolError("service_quotation_candidate_expired_or_invalid")
    return ServiceQuotationCandidate(
        provider=candidate.provider,
        candidate_id=candidate.candidate_id,
        source_version=candidate.source_version,
        display_name=candidate.display_name,
        incident_ref=candidate.incident_ref,
        asset_ref=candidate.asset_ref,
        diagnosis_ref=candidate.diagnosis_ref,
        currency=candidate.currency,
        line_items=tuple(normalized_items),
        subtotal=_money_text(subtotal),
        discount=_money_text(discount),
        tax=_money_text(tax),
        total=_money_text(total),
        as_of=as_of,
        expires_at=expires_at,
    )


def service_quotation_candidate_binding(
    candidate: ServiceQuotationCandidate,
) -> dict[str, object]:
    return {
        "provider": candidate.provider,
        "candidate_id": candidate.candidate_id,
        "source_version": candidate.source_version,
        "display_name": candidate.display_name,
        "incident_ref": candidate.incident_ref,
        "asset_ref": candidate.asset_ref,
        "diagnosis_ref": candidate.diagnosis_ref,
        "currency": candidate.currency,
        "line_items": [
            {
                "code": item.code,
                "description": item.description,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "line_total": item.line_total,
            }
            for item in candidate.line_items
        ],
        "subtotal": candidate.subtotal,
        "discount": candidate.discount,
        "tax": candidate.tax,
        "total": candidate.total,
        "as_of": candidate.as_of.isoformat(),
        "expires_at": candidate.expires_at.isoformat(),
    }


class HttpServiceQuotationCatalogAdapter:
    """Read-only CPQ client; URL, token and provider-private fields remain internal."""

    def __init__(
        self,
        base_url: str,
        *,
        provider_id: str,
        api_token: SecretValue,
        timeout_seconds: float = 5.0,
        allow_plain_http: bool = False,
        transport: EnterpriseHttpTransport | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        allowed_schemes = {"http", "https"} if allow_plain_http else {"https"}
        if not parsed.netloc or parsed.scheme not in allowed_schemes:
            raise ValueError("service quotation provider URL must use an allowed HTTP scheme")
        if not provider_id or len(provider_id) > 128 or not all(
            char.isascii() and (char.isalnum() or char in "._-") for char in provider_id
        ):
            raise ValueError("service quotation provider ID is invalid")
        self._base_url = base_url.rstrip("/")
        self.provider_id = provider_id
        self._api_token = api_token
        self._timeout_seconds = timeout_seconds
        self._transport = transport or UrllibEnterpriseHttpTransport()

    def list_candidates(
        self,
        *,
        tenant_id: str,
        incident_ref: str,
        asset_ref: str,
        diagnosis_ref: str,
    ) -> tuple[ServiceQuotationCandidate, ...]:
        query = urlencode(
            {
                "tenant_id": tenant_id,
                "incident_ref": incident_ref,
                "asset_ref": asset_ref,
                "diagnosis_ref": diagnosis_ref,
            }
        )
        payload = self._request(f"/v1/service-quotation-candidates?{query}")
        try:
            response = _CandidateListResponse.model_validate_json(payload)
        except ValidationError as exc:
            raise EnterpriseToolError("service_quotation_catalog_invalid") from exc
        return tuple(self._candidate(item) for item in response.candidates)

    def resolve_candidate(
        self,
        *,
        tenant_id: str,
        candidate_id: str,
        incident_ref: str,
        asset_ref: str,
        diagnosis_ref: str,
    ) -> ServiceQuotationCandidate | None:
        query = urlencode(
            {
                "tenant_id": tenant_id,
                "incident_ref": incident_ref,
                "asset_ref": asset_ref,
                "diagnosis_ref": diagnosis_ref,
            }
        )
        payload = self._request(
            f"/v1/service-quotation-candidates/{quote(candidate_id, safe='')}?{query}",
            not_found_allowed=True,
        )
        if payload is None:
            return None
        try:
            response = _CandidateResponse.model_validate_json(payload)
        except ValidationError as exc:
            raise EnterpriseToolError("service_quotation_candidate_invalid") from exc
        return self._candidate(response)

    def _request(self, path: str, *, not_found_allowed: bool = False) -> bytes | None:
        try:
            response = self._transport.request(
                "GET",
                f"{self._base_url}{path}",
                headers={"Authorization": f"Bearer {self._api_token.reveal()}"},
                payload=None,
                timeout_seconds=self._timeout_seconds,
            )
        except EnterpriseTransportUnavailable as exc:
            raise EnterpriseToolError("service_quotation_catalog_unavailable") from exc
        if response.status_code == 404 and not_found_allowed:
            return None
        if response.status_code < 200 or response.status_code >= 300:
            raise EnterpriseToolError("service_quotation_catalog_rejected")
        return response.body

    def _candidate(self, value: _CandidateResponse) -> ServiceQuotationCandidate:
        return ServiceQuotationCandidate(
            provider=self.provider_id,
            candidate_id=value.candidate_id,
            source_version=value.source_version,
            display_name=value.display_name,
            incident_ref=value.incident_ref,
            asset_ref=value.asset_ref,
            diagnosis_ref=value.diagnosis_ref,
            currency=value.currency,
            line_items=tuple(
                ServiceQuotationLineItem(
                    code=item.code,
                    description=item.description,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    line_total=item.line_total,
                )
                for item in value.line_items
            ),
            subtotal=value.subtotal,
            discount=value.discount,
            tax=value.tax,
            total=value.total,
            as_of=value.as_of,
            expires_at=value.expires_at,
        )


def _money(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise EnterpriseToolError("service_quotation_amounts_invalid") from exc
    if (
        parsed < 0
        or parsed > _MONEY_LIMIT
        or parsed.as_tuple().exponent != -2
        or not parsed.is_finite()
    ):
        raise EnterpriseToolError("service_quotation_amounts_invalid")
    return parsed


def _money_text(value: Decimal) -> str:
    return f"{value:.2f}"


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise EnterpriseToolError("service_quotation_candidate_invalid")
    return value.astimezone(UTC)
