"""Versioned W3C Trace Context transport for transactional Outbox events."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from opentelemetry.context import Context
from opentelemetry.trace import TraceState
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from industrial_ops_agent.telemetry import extract_trace_context

TRACE_CONTEXT_PROFILE = "ioap.kafka.trace-context.v1"
TRACE_CONTEXT_PROFILE_HEADER = "ioap-trace-context-profile"
SOURCE_TRACEPARENT_HEADER = "ioap-source-traceparent"
SOURCE_TRACESTATE_HEADER = "ioap-source-tracestate"

_TRACEPARENT_PATTERN = re.compile(
    r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
)
_TRACE_HEADER_NAMES = frozenset(
    {
        "traceparent",
        "tracestate",
        TRACE_CONTEXT_PROFILE_HEADER,
        SOURCE_TRACEPARENT_HEADER,
        SOURCE_TRACESTATE_HEADER,
    }
)


class InvalidEventTraceContext(ValueError):
    """Kafka Trace Context headers are malformed, ambiguous, or unsupported."""


@dataclass(frozen=True, slots=True)
class W3CTraceContext:
    """Validated source context persisted with an Outbox row."""

    traceparent: str
    tracestate: str | None

    @property
    def trace_id(self) -> str:
        return self.traceparent.split("-", 3)[1]

    def as_debezium_properties(self) -> str:
        """Serialize the carrier in the java.util.Properties format Debezium expects."""

        lines = [f"traceparent={_escape_java_property(self.traceparent)}"]
        if self.tracestate is not None:
            lines.append(f"tracestate={_escape_java_property(self.tracestate)}")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class ExtractedEventTraceContext:
    """A validated remote parent and its low-cardinality provenance."""

    carrier: W3CTraceContext
    parent_context: Context
    propagation: str


KafkaHeaders = Sequence[tuple[str, bytes | str | None]]


def capture_current_trace_context() -> W3CTraceContext | None:
    """Capture the current sampled or unsampled W3C context without baggage."""

    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier=carrier)
    traceparent = carrier.get("traceparent")
    if not isinstance(traceparent, str):
        return None
    tracestate = carrier.get("tracestate")
    return validate_trace_context(
        traceparent,
        tracestate if isinstance(tracestate, str) else None,
    )


def extract_event_trace_context(
    headers: KafkaHeaders | None,
) -> ExtractedEventTraceContext | None:
    """Prefer Debezium's standard parent and safely fall back to the source header."""

    values = _trace_header_values(headers)
    profile = values.get(TRACE_CONTEXT_PROFILE_HEADER)
    if profile is not None and profile != TRACE_CONTEXT_PROFILE:
        raise InvalidEventTraceContext("unsupported_trace_context_profile")

    standard_traceparent = values.get("traceparent")
    source_traceparent = values.get(SOURCE_TRACEPARENT_HEADER)
    if standard_traceparent is not None:
        traceparent = standard_traceparent
        tracestate = values.get("tracestate")
        propagation = "DEBEZIUM_PARENT"
    elif source_traceparent is not None:
        if profile is None:
            raise InvalidEventTraceContext("source_trace_context_profile_missing")
        traceparent = source_traceparent
        tracestate = values.get(SOURCE_TRACESTATE_HEADER)
        propagation = "SOURCE_FALLBACK"
    else:
        if profile is not None:
            raise InvalidEventTraceContext("traceparent_header_missing")
        if "tracestate" in values or SOURCE_TRACESTATE_HEADER in values:
            raise InvalidEventTraceContext("orphan_tracestate_header")
        return None

    carrier = validate_trace_context(traceparent, tracestate)
    parent_context = extract_trace_context(carrier.traceparent, carrier.tracestate)
    if parent_context is None:  # Defensive: validate_trace_context already checks this.
        raise InvalidEventTraceContext("traceparent_context_invalid")
    return ExtractedEventTraceContext(carrier, parent_context, propagation)


def validate_trace_context(
    traceparent: str,
    tracestate: str | None,
) -> W3CTraceContext:
    """Validate the bounded W3C subset supported by the platform transport profile."""

    if traceparent != traceparent.strip() or len(traceparent) != 55:
        raise InvalidEventTraceContext("traceparent_format_invalid")
    match = _TRACEPARENT_PATTERN.fullmatch(traceparent)
    if match is None or match.group(1) == "0" * 32 or match.group(2) == "0" * 16:
        raise InvalidEventTraceContext("traceparent_format_invalid")

    normalized_tracestate: str | None = None
    if tracestate is not None:
        normalized_tracestate = tracestate.strip()
        if (
            not normalized_tracestate
            or len(normalized_tracestate) > 512
            or any(ord(character) < 0x20 or ord(character) > 0x7E for character in tracestate)
        ):
            raise InvalidEventTraceContext("tracestate_format_invalid")
        parsed = TraceState.from_header([normalized_tracestate])
        if len(parsed) == 0:
            raise InvalidEventTraceContext("tracestate_format_invalid")

    if extract_trace_context(traceparent, normalized_tracestate) is None:
        raise InvalidEventTraceContext("traceparent_context_invalid")
    return W3CTraceContext(traceparent, normalized_tracestate)


def trace_headers(context: W3CTraceContext) -> tuple[tuple[str, str], ...]:
    """Return the source headers emitted by the Debezium Outbox router fallback path."""

    values = [
        (TRACE_CONTEXT_PROFILE_HEADER, TRACE_CONTEXT_PROFILE),
        (SOURCE_TRACEPARENT_HEADER, context.traceparent),
    ]
    if context.tracestate is not None:
        values.append((SOURCE_TRACESTATE_HEADER, context.tracestate))
    return tuple(values)


def _trace_header_values(headers: KafkaHeaders | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_name, raw_value in headers or ():
        name = raw_name.lower()
        if name not in _TRACE_HEADER_NAMES or raw_value is None:
            continue
        if name in values:
            raise InvalidEventTraceContext("duplicate_trace_context_header")
        if isinstance(raw_value, bytes):
            try:
                value = raw_value.decode("ascii")
            except UnicodeDecodeError as exc:
                raise InvalidEventTraceContext("trace_context_header_not_ascii") from exc
        elif isinstance(raw_value, str):
            value = raw_value
        else:
            raise InvalidEventTraceContext("trace_context_header_type_invalid")
        if len(value) > 512:
            raise InvalidEventTraceContext("trace_context_header_too_long")
        values[name] = value
    return values


def _escape_java_property(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("=", "\\=")
        .replace(":", "\\:")
    )
