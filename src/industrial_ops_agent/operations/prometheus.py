"""Strict, read-only Prometheus adapter used by the operations BFF."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_.:-]{1,128}$")
_ALLOWED_SEVERITIES = frozenset({"critical", "warning", "info"})


class PrometheusUnavailable(RuntimeError):
    """Prometheus could not provide trustworthy operational evidence."""


@dataclass(frozen=True, slots=True)
class PrometheusAlert:
    name: str
    severity: str
    state: str
    active_at: datetime
    summary: str
    runbook_id: str
    owner: str
    security_hard_gate: bool


class PrometheusReader(Protocol):
    def query_scalar(self, query: str) -> float | None: ...

    def active_alerts(self) -> tuple[PrometheusAlert, ...]: ...


class PrometheusHttpReader:
    """Execute only server-owned PromQL and expose a bounded alert projection."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 5.0,
        ca_file: Path | None = None,
        token_file: Path | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = client is None
        if client is not None:
            self._client = client
            return
        headers: dict[str, str] = {}
        if token_file is not None:
            token = token_file.read_text(encoding="utf-8").strip()
            if not token or "\n" in token or "\r" in token:
                raise ValueError("Prometheus bearer token file is invalid")
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
            verify=str(ca_file) if ca_file is not None else True,
        )

    def query_scalar(self, query: str) -> float | None:
        payload = self._request("/api/v1/query", params={"query": query})
        data = _mapping(payload.get("data"), "Prometheus query data")
        result_type = data.get("resultType")
        result = data.get("result")
        raw_value: Any
        if result_type == "scalar":
            raw_value = result
        elif result_type == "vector":
            if not isinstance(result, list):
                raise PrometheusUnavailable("Prometheus vector result is invalid")
            if not result:
                return None
            if len(result) != 1:
                raise PrometheusUnavailable("Prometheus query returned multiple series")
            raw_value = _mapping(result[0], "Prometheus vector sample").get("value")
        else:
            raise PrometheusUnavailable("Prometheus result type is unsupported")
        if not isinstance(raw_value, list) or len(raw_value) != 2:
            raise PrometheusUnavailable("Prometheus scalar sample is invalid")
        try:
            value = float(raw_value[1])
        except (TypeError, ValueError) as exc:
            raise PrometheusUnavailable("Prometheus scalar value is invalid") from exc
        if not math.isfinite(value):
            raise PrometheusUnavailable("Prometheus scalar value is not finite")
        return value

    def active_alerts(self) -> tuple[PrometheusAlert, ...]:
        payload = self._request("/api/v1/alerts")
        alerts = _mapping(payload.get("data"), "Prometheus alert data").get("alerts")
        if not isinstance(alerts, list):
            raise PrometheusUnavailable("Prometheus alerts result is invalid")
        projected: list[PrometheusAlert] = []
        for raw in alerts:
            if not isinstance(raw, dict):
                continue
            state = raw.get("state")
            if state not in {"firing", "pending"}:
                continue
            labels = raw.get("labels")
            annotations = raw.get("annotations")
            if not isinstance(labels, dict) or not isinstance(annotations, dict):
                continue
            name = _safe_identifier(labels.get("alertname"), fallback="UnknownAlert")
            severity = str(labels.get("severity", "warning")).lower()
            if severity not in _ALLOWED_SEVERITIES:
                severity = "warning"
            active_at = _parse_timestamp(raw.get("activeAt"))
            if active_at is None:
                continue
            projected.append(
                PrometheusAlert(
                    name=name,
                    severity=severity,
                    state=state,
                    active_at=active_at,
                    summary=_bounded_text(annotations.get("summary"), 240),
                    runbook_id=_safe_identifier(
                        labels.get("runbook_id"), fallback="operations-generic"
                    ),
                    owner=_safe_identifier(labels.get("owner"), fallback="platform-sre"),
                    security_hard_gate=labels.get("security_hard_gate") == "true",
                )
            )
        return tuple(
            sorted(projected, key=lambda alert: (alert.severity != "critical", alert.active_at))
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(self, path: str, *, params: dict[str, str] | None = None) -> dict[str, Any]:
        try:
            response = self._client.get(path, params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise PrometheusUnavailable("Prometheus request failed") from exc
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise PrometheusUnavailable("Prometheus returned an unsuccessful response")
        return payload


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PrometheusUnavailable(f"{label} is invalid")
    return value


def _safe_identifier(value: Any, *, fallback: str) -> str:
    return value if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) else fallback


def _bounded_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return "No alert summary supplied"
    normalized = " ".join(value.split())
    return normalized[:limit] or "No alert summary supplied"


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)
