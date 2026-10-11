"""OpenTelemetry runtime and safe manual spans for business and AI operations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    from industrial_ops_agent.config import Settings

_TRACER_NAME = "industrial_ops_agent"


@dataclass(slots=True)
class TelemetryRuntime:
    """Own the SDK provider so buffered spans are flushed during graceful shutdown."""

    provider: TracerProvider | None = None

    def shutdown(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()


class _AiSpanProcessor(SpanProcessor):
    """Forward only reviewed Agent/Tool/Generation spans to Langfuse."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def on_start(self, span: Any, parent_context: Context | None = None) -> None:
        del span, parent_context

    def on_end(self, span: Any) -> None:
        attributes = span.attributes or {}
        if "langfuse.observation.type" in attributes or "gen_ai.operation.name" in attributes:
            self._delegate.on_end(span)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return bool(self._delegate.force_flush(timeout_millis))


def configure_telemetry(settings: Settings) -> TelemetryRuntime:
    """Configure one OTLP/HTTP exporter without making telemetry a core dependency."""

    if not settings.otel_enabled:
        return TelemetryRuntime()

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "service.namespace": settings.otel_service_namespace,
            "service.version": settings.release_version,
            "deployment.environment.name": settings.environment.value,
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(settings.otel_trace_sample_ratio)),
    )
    exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_traces_endpoint,
        certificate_file=(
            str(settings.otel_exporter_certificate_file)
            if settings.otel_exporter_certificate_file is not None
            else None
        ),
        timeout=settings.otel_exporter_timeout_seconds,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    if settings.langfuse_otel_traces_endpoint is not None:
        authorization_file = settings.langfuse_otel_authorization_file
        if authorization_file is None:
            raise ValueError("Langfuse OTLP authorization file is required")
        authorization = authorization_file.read_text(encoding="utf-8").strip()
        if not authorization:
            raise ValueError("Langfuse OTLP authorization file is empty")
        if not authorization.startswith("Basic "):
            authorization = f"Basic {authorization}"
        langfuse_exporter = OTLPSpanExporter(
            endpoint=settings.langfuse_otel_traces_endpoint,
            headers={
                "Authorization": authorization,
                "x-langfuse-ingestion-version": "4",
            },
            timeout=settings.otel_exporter_timeout_seconds,
        )
        provider.add_span_processor(_AiSpanProcessor(BatchSpanProcessor(langfuse_exporter)))
    trace.set_tracer_provider(provider)
    return TelemetryRuntime(provider)


def safe_tenant_hash(tenant_id: str) -> str:
    """Provide a stable non-plaintext tenant correlation attribute."""

    return sha256(tenant_id.encode()).hexdigest()[:16]


def current_otel_trace_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    return f"{context.trace_id:032x}" if context.is_valid else None


def current_traceparent() -> str | None:
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return f"00-{context.trace_id:032x}-{context.span_id:016x}-{int(context.trace_flags):02x}"


def extract_trace_context(
    traceparent: str | None,
    tracestate: str | None = None,
) -> Context | None:
    if traceparent is None:
        return None
    carrier: dict[str, str] = {"traceparent": traceparent}
    if tracestate is not None:
        carrier["tracestate"] = tracestate
    context = TraceContextTextMapPropagator().extract(carrier=carrier)
    return context if trace.get_current_span(context).get_span_context().is_valid else None


@contextmanager
def traced_operation(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Mapping[str, str | int | float | bool] | None = None,
    parent_context: Context | None = None,
) -> Iterator[Span]:
    """Create a span containing only caller-reviewed scalar attributes."""

    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(
        name,
        context=parent_context,
        kind=kind,
        attributes=dict(attributes or {}),
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
