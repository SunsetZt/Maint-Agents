"""Shared authenticated transport for ERP, finance and FSM delivery adapters."""

from __future__ import annotations

from typing import ClassVar

from industrial_ops_agent.secrets import SecretValue
from industrial_ops_agent.tools.contracts import EnterpriseToolError
from industrial_ops_agent.tools.enterprise_http import (
    EnterpriseHttpTransport,
    EnterpriseTransportUnavailable,
)


class EnterpriseDeliveryHttp:
    """Transport only; each adapter owns its schemas, receipts and business rules."""

    _base_url: str
    _api_token: SecretValue
    _timeout_seconds: float
    _transport: EnterpriseHttpTransport
    _unavailable_reason: ClassVar[str]
    _rejected_reason: ClassVar[str]

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
            raise EnterpriseToolError(self._unavailable_reason) from exc
        if response.status_code == 404 and method == "GET":
            return b"null"
        if response.status_code < 200 or response.status_code >= 300:
            raise EnterpriseToolError(self._rejected_reason)
        return response.body
