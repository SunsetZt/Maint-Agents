"""Request correlation, structured logging, and bounded HTTP metrics."""

from __future__ import annotations

import json
import re
import sys
from contextvars import ContextVar, Token
from dataclasses import dataclass
from logging import getLevelNamesMapping
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any, TextIO
from uuid import uuid4

import structlog
from opentelemetry.trace import SpanKind, Status, StatusCode
from prometheus_client import CollectorRegistry, Counter, Histogram
from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars

from industrial_ops_agent.telemetry import (
    current_otel_trace_id,
    current_traceparent,
    extract_trace_context,
    traced_operation,
)

if TYPE_CHECKING:
    from fastapi import Request, Response
    from starlette.middleware.base import RequestResponseEndpoint

REQUEST_ID_HEADER = "X-Request-ID"
TRACE_ID_HEADER = "X-Trace-ID"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TRACEPARENT_PATTERN = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}$")

_request_id: ContextVar[str] = ContextVar("request_id", default="")
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_log_mirror: _LogMirror | None = None


@dataclass(frozen=True, slots=True)
class RequestContextTokens:
    """Tokens used to restore request context after a response."""

    request_id: Token[str]
    trace_id: Token[str]


class HttpMetrics:
    """Per-application metrics registry to keep tests and app factories isolated."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry(auto_describe=True)
        self.requests = Counter(
            "ioap_http_requests_total",
            "HTTP requests processed by the API.",
            ("method", "path", "status"),
            registry=self.registry,
        )
        self.duration = Histogram(
            "ioap_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            ("method", "path"),
            registry=self.registry,
        )

    def observe(self, *, method: str, path: str, status_code: int, seconds: float) -> None:
        """Record a request using bounded route labels."""

        self.requests.labels(method=method, path=path, status=str(status_code)).inc()
        self.duration.labels(method=method, path=path).observe(seconds)


class _LogMirror:
    """Mirror the already-minimized structured event to a collector-mounted file."""

    def __init__(self, destination: TextIO) -> None:
        self._destination = destination
        self._lock = Lock()

    def __call__(
        self,
        _logger: Any,
        _method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        rendered = json.dumps(
            event_dict,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        with self._lock:
            self._destination.write(rendered + "\n")
            self._destination.flush()
        return event_dict

    def close(self) -> None:
        self._destination.close()


def configure_logging(
    log_level: str,
    *,
    service_name: str = "industrial-ops",
    log_file: Path | None = None,
) -> None:
    """Configure deterministic JSON logs without request bodies or query values."""

    normalized_level = log_level.upper()
    level = getLevelNamesMapping().get(normalized_level)
    if level is None:
        raise ValueError(f"Unsupported log level: {log_level}")

    processors: list[Any] = [
        merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_service_name(service_name),
    ]
    global _log_mirror
    if _log_mirror is not None:
        _log_mirror.close()
        _log_mirror = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _log_mirror = _LogMirror(log_file.open("a", encoding="utf-8"))
        processors.append(_log_mirror)
    processors.append(structlog.processors.JSONRenderer(sort_keys=True))
    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=False,
    )


def _add_service_name(service_name: str) -> Any:
    def add_service(
        _logger: Any,
        _method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        event_dict.setdefault("service", service_name)
        return event_dict

    return add_service


def normalize_request_id(value: str | None) -> str:
    """Accept a bounded safe caller ID or create a new opaque ID."""

    if value and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return f"req-{uuid4().hex}"


def trace_id_from_header(value: str | None) -> str:
    """Extract a W3C trace ID without trusting other traceparent fields."""

    if value:
        match = _TRACEPARENT_PATTERN.fullmatch(value.lower())
        if match and match.group(1) != "0" * 32:
            return match.group(1)
    return uuid4().hex


def bind_request_context(request_id: str, trace_id: str) -> RequestContextTokens:
    """Bind request correlation to stdlib context and structlog."""

    clear_contextvars()
    bind_contextvars(request_id=request_id, trace_id=trace_id)
    return RequestContextTokens(
        request_id=_request_id.set(request_id),
        trace_id=_trace_id.set(trace_id),
    )


def reset_request_context(tokens: RequestContextTokens) -> None:
    """Restore context when a request completes."""

    _request_id.reset(tokens.request_id)
    _trace_id.reset(tokens.trace_id)
    clear_contextvars()


def current_request_id() -> str:
    """Return the active opaque request identifier."""

    return _request_id.get()


def current_trace_id() -> str:
    """Return the active trace identifier."""

    return _trace_id.get()


async def observe_request(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Correlate and observe one ASGI request without logging sensitive values."""

    request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
    supplied_traceparent = request.headers.get("traceparent")
    parent_context = extract_trace_context(supplied_traceparent)
    with traced_operation(
        f"{request.method} HTTP",
        kind=SpanKind.SERVER,
        attributes={
            "http.request.method": request.method,
            "url.scheme": request.url.scheme,
        },
        parent_context=parent_context,
    ) as span:
        trace_id = current_otel_trace_id() or trace_id_from_header(supplied_traceparent)
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        request.state.traceparent = current_traceparent()
        tokens = bind_request_context(request_id, trace_id)
        started = perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers[TRACE_ID_HEADER] = trace_id
            return response
        finally:
            elapsed = perf_counter() - started
            route = request.scope.get("route")
            route_path = getattr(route, "path", "<unmatched>")
            span.update_name(f"{request.method} {route_path}")
            span.set_attribute("http.route", route_path)
            span.set_attribute("http.response.status_code", status_code)
            if status_code >= 500:
                span.set_status(Status(StatusCode.ERROR))
            metrics: HttpMetrics = request.app.state.http_metrics
            metrics.observe(
                method=request.method,
                path=route_path,
                status_code=status_code,
                seconds=elapsed,
            )
            structlog.get_logger().info(
                "http_request",
                method=request.method,
                path=route_path,
                status_code=status_code,
                duration_ms=round(elapsed * 1000, 3),
            )
            reset_request_context(tokens)
