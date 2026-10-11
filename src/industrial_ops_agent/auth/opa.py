"""OPA decision client used for high-risk authorization decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import httpx


class OpaUnavailable(RuntimeError):
    """Raised when a remote policy decision cannot be obtained safely."""


@dataclass(frozen=True, slots=True)
class OpaDecisionInput:
    subject_id: str
    tenant_id: str
    roles: tuple[str, ...]
    asset_ids: tuple[str, ...]
    site_ids: tuple[str, ...]
    action: str
    resource_tenant_id: str
    resource_id: str | None
    resource_asset_id: str | None
    resource_site_id: str | None

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OpaDecision:
    allowed: bool
    reason_code: str
    obligations: tuple[str, ...] = ()
    policy_version: str = "unknown"


class PolicyDecisionClient(Protocol):
    def decide(self, decision_input: OpaDecisionInput) -> OpaDecision: ...


class OpaPolicyClient:
    """Bounded HTTP client for the OPA REST decision API."""

    def __init__(self, decision_url: str, *, timeout_seconds: float = 1.5) -> None:
        self._decision_url = decision_url
        self._timeout = httpx.Timeout(timeout_seconds)

    def decide(self, decision_input: OpaDecisionInput) -> OpaDecision:
        try:
            response = httpx.post(
                self._decision_url,
                json={"input": decision_input.as_payload()},
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json().get("result")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise OpaUnavailable("opa_decision_unavailable") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("allow"), bool):
            raise OpaUnavailable("opa_decision_invalid")
        obligations = payload.get("obligations", [])
        if not isinstance(obligations, list) or not all(
            isinstance(item, str) for item in obligations
        ):
            raise OpaUnavailable("opa_obligations_invalid")
        return OpaDecision(
            allowed=payload["allow"],
            reason_code=str(payload.get("reason", "opa_policy_denied")),
            obligations=tuple(obligations),
            policy_version=str(payload.get("policy_version", "unknown")),
        )
