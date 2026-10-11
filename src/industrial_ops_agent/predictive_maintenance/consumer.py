"""Replay-safe Kafka entrance for the governed telemetry contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Protocol

import structlog
from prometheus_client import start_http_server

from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.config import Settings, get_settings
from industrial_ops_agent.observability import configure_logging
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.predictive_maintenance.service import (
    PredictiveMaintenanceConflict,
    PredictiveMaintenanceNotFound,
    PredictiveMaintenanceService,
    TelemetryEventInput,
)
from industrial_ops_agent.runtime_metrics import EventConsumerMetrics
from industrial_ops_agent.secrets import SecretName, build_secret_provider
from industrial_ops_agent.security_audit import build_persistent_security_auditor
from industrial_ops_agent.telemetry import configure_telemetry


class KafkaMessage(Protocol):
    def value(self) -> bytes | str | dict[str, Any] | None: ...

    def error(self) -> Any: ...

    def topic(self) -> str: ...

    def partition(self) -> int: ...


class KafkaClient(Protocol):
    def poll(self, timeout: float) -> KafkaMessage | None: ...

    def commit(self, *, message: KafkaMessage, asynchronous: bool) -> Any: ...

    def pause(self, partitions: list[Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class TelemetryProcessingResult:
    disposition: str
    event_id: str | None
    occurred_at: datetime | None
    commit_offset: bool
    reason: str | None = None


class TelemetryEventProcessor:
    def __init__(self, service: PredictiveMaintenanceService) -> None:
        self._service = service

    def process(self, payload: bytes | str | dict[str, Any]) -> TelemetryProcessingResult:
        event_id: str | None = None
        try:
            value = _decode(payload)
            tenant_id = _required_string(value, "tenant_id")
            event_id = _required_string(value, "event_id")
            asset_id = _required_string(value, "asset_id")
            now = datetime.now(UTC)
            identity = IdentityContext(
                subject_id="m7-telemetry-gateway",
                oidc_subject="internal:m7-telemetry-gateway",
                tenant_id=tenant_id,
                roles=frozenset({Role.PLATFORM_OPERATOR}),
                asset_ids=frozenset({asset_id}),
                site_ids=frozenset(),
                issued_at=now,
                expires_at=now + timedelta(minutes=5),
            )
            measurements = value.get("measurements")
            quality_flags = value.get("quality_flags", [])
            if not isinstance(measurements, dict) or not isinstance(quality_flags, list):
                raise ValueError("telemetry measurements or quality flags are invalid")
            result = self._service.ingest_event(
                identity,
                TelemetryEventInput(
                    event_id=event_id,
                    asset_id=asset_id,
                    schema_version=_required_string(value, "schema_version"),
                    sequence=_required_integer(value, "sequence"),
                    source_system=_required_string(value, "source_system"),
                    source_ref=_required_string(value, "source_ref"),
                    event_time=_required_timestamp(value, "event_time"),
                    measurements=dict(measurements),
                    quality_flags=tuple(quality_flags),
                ),
                request_id=f"kafka:{event_id}",
                received_at=now,
            )
            return TelemetryProcessingResult(
                "DUPLICATE" if result.duplicate else result.event.ingestion_status,
                event_id,
                result.event.event_time,
                True,
            )
        except (
            AuthorizationDenied,
            PredictiveMaintenanceConflict,
            PredictiveMaintenanceNotFound,
            TypeError,
            ValueError,
        ) as exc:
            return TelemetryProcessingResult(
                "REJECTED_CONTRACT",
                event_id,
                None,
                True,
                type(exc).__name__,
            )
        except Exception as exc:  # dependency faults are retried without committing the offset
            return TelemetryProcessingResult(
                "RETRY_DEPENDENCY",
                event_id,
                None,
                False,
                type(exc).__name__,
            )


class ReplaySafeTelemetryConsumer:
    def __init__(
        self,
        client: KafkaClient,
        processor: TelemetryEventProcessor,
        *,
        partition_factory: Any,
        metrics: EventConsumerMetrics | None = None,
    ) -> None:
        self._client = client
        self._processor = processor
        self._partition_factory = partition_factory
        self._metrics = metrics

    def run_once(self, *, timeout_seconds: float) -> TelemetryProcessingResult:
        message = self._client.poll(timeout_seconds)
        if message is None:
            return TelemetryProcessingResult("NO_MESSAGE", None, None, False)
        if message.error() is not None:
            self._pause(message)
            return TelemetryProcessingResult("KAFKA_ERROR", None, None, False, str(message.error()))
        payload = message.value()
        result = (
            self._processor.process(payload)
            if payload is not None
            else TelemetryProcessingResult("REJECTED_CONTRACT", None, None, True, "empty")
        )
        if result.commit_offset:
            self._client.commit(message=message, asynchronous=False)
        else:
            self._pause(message)
        if self._metrics is not None:
            self._metrics.observe(
                consumer="m7-telemetry-v1",
                disposition=result.disposition,
                occurred_at=result.occurred_at,
            )
        structlog.get_logger().info(
            "telemetry_event_processed",
            event_id=result.event_id,
            disposition=result.disposition,
            commit_offset=result.commit_offset,
            reason=result.reason,
        )
        return result

    def _pause(self, message: KafkaMessage) -> None:
        self._client.pause([self._partition_factory(message.topic(), message.partition())])


def _consumer_config(settings: Settings) -> dict[str, Any]:
    return {
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "group.id": settings.kafka_telemetry_consumer_group,
        "enable.auto.commit": False,
        "enable.auto.offset.store": False,
        "auto.offset.reset": "earliest",
        "isolation.level": "read_committed",
    }


def run() -> None:
    """Consume the telemetry topic until interrupted."""

    from confluent_kafka import Consumer, TopicPartition

    settings = get_settings()
    configure_logging(
        settings.log_level,
        service_name="industrial-ops-telemetry-consumer",
        log_file=settings.log_file,
    )
    telemetry = configure_telemetry(settings)
    provider = build_secret_provider(settings)
    database = Database(provider.get(SecretName.DATABASE_URL).reveal())
    audit_secret = provider.get(SecretName.OIDC_CLIENT_SECRET).reveal()
    audit_key = sha256(b"industrial-ops-audit-v1\0" + audit_secret.encode()).digest()
    authorizer = Authorizer(build_persistent_security_auditor(database, hash_key=audit_key))
    client = Consumer(_consumer_config(settings))
    client.subscribe([settings.kafka_telemetry_topic])
    metrics = EventConsumerMetrics()
    metrics_server, _metrics_thread = start_http_server(
        settings.telemetry_worker_metrics_port,
        addr=settings.worker_metrics_host,
        registry=metrics.registry,
    )
    consumer = ReplaySafeTelemetryConsumer(
        client,
        TelemetryEventProcessor(PredictiveMaintenanceService(database, authorizer)),
        partition_factory=TopicPartition,
        metrics=metrics,
    )
    try:
        while True:
            consumer.run_once(timeout_seconds=settings.worker_poll_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        metrics_server.shutdown()
        metrics_server.server_close()
        client.close()
        database.dispose()
        telemetry.shutdown()


def _decode(payload: bytes | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    decoded = json.loads(payload.decode() if isinstance(payload, bytes) else payload)
    if not isinstance(decoded, dict):
        raise ValueError("telemetry envelope must be an object")
    return decoded


def _required_string(value: dict[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"telemetry field is invalid: {name}")
    return item


def _required_integer(value: dict[str, Any], name: str) -> int:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"telemetry field is invalid: {name}")
    return item


def _required_timestamp(value: dict[str, Any], name: str) -> datetime:
    item = _required_string(value, name)
    parsed = datetime.fromisoformat(item.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"telemetry timestamp requires timezone: {name}")
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    run()
