"""Kafka adapter that commits offsets only after replay-safe processing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast

import structlog
from prometheus_client import start_http_server

from industrial_ops_agent.config import Settings, get_settings
from industrial_ops_agent.event_worker.handlers import (
    CONSUMER_NAME,
    FeedbackEventProcessor,
    ProcessingResult,
)
from industrial_ops_agent.eventing.trace_context import (
    InvalidEventTraceContext,
    extract_event_trace_context,
)
from industrial_ops_agent.observability import configure_logging
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.runtime_metrics import EventConsumerMetrics
from industrial_ops_agent.secrets import SecretName, build_secret_provider
from industrial_ops_agent.telemetry import configure_telemetry


class KafkaMessage(Protocol):
    def value(self) -> bytes | str | dict[str, Any] | None: ...

    def error(self) -> Any: ...

    def topic(self) -> str: ...

    def partition(self) -> int: ...

    def headers(self) -> list[tuple[str, bytes | str | None]] | None: ...


class KafkaClient(Protocol):
    def poll(self, timeout: float) -> KafkaMessage | None: ...

    def commit(self, *, message: KafkaMessage, asynchronous: bool) -> Any: ...

    def pause(self, partitions: list[Any]) -> Any: ...


PartitionFactory = Callable[[str, int], Any]


class ReplaySafeKafkaConsumer:
    """Translate one Kafka delivery into an explicit offset decision."""

    def __init__(
        self,
        client: KafkaClient,
        processor: FeedbackEventProcessor,
        *,
        partition_factory: PartitionFactory,
        metrics: EventConsumerMetrics | None = None,
    ) -> None:
        self._client = client
        self._processor = processor
        self._partition_factory = partition_factory
        self._metrics = metrics

    def run_once(self, *, timeout_seconds: float) -> ProcessingResult:
        message = self._client.poll(timeout_seconds)
        if message is None:
            return ProcessingResult("NO_MESSAGE", None, None, False)
        if message.error() is not None:
            self._pause(message)
            return ProcessingResult(
                "KAFKA_ERROR",
                None,
                None,
                False,
                str(message.error()),
            )

        try:
            header_reader = getattr(message, "headers", None)
            raw_headers = header_reader() if callable(header_reader) else None
            trace_context = extract_event_trace_context(raw_headers)
        except InvalidEventTraceContext as exc:
            result = ProcessingResult(
                "REJECTED_CONTRACT",
                None,
                None,
                False,
                str(exc),
            )
        else:
            payload = message.value()
            if payload is None:
                result = ProcessingResult(
                    "REJECTED_CONTRACT",
                    None,
                    None,
                    False,
                    "empty_message_value",
                )
            else:
                result = self._processor.process(payload, trace_context=trace_context)

        if result.commit_offset:
            self._client.commit(message=message, asynchronous=False)
        else:
            self._pause(message)
        if self._metrics is not None:
            self._metrics.observe(
                consumer=CONSUMER_NAME,
                disposition=result.disposition,
                occurred_at=result.occurred_at,
            )
        structlog.get_logger().info(
            "feedback_event_processed",
            event_id=result.event_id,
            disposition=result.disposition,
            commit_offset=result.commit_offset,
            reason=result.reason,
            source_trace_id=result.source_trace_id,
            tenant_hash=result.tenant_hash,
            trace_context_propagation=result.trace_context_propagation,
        )
        return result

    def _pause(self, message: KafkaMessage) -> None:
        partition = self._partition_factory(message.topic(), message.partition())
        self._client.pause([partition])


def _consumer_config(settings: Settings) -> dict[str, Any]:
    return {
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "group.id": settings.kafka_feedback_consumer_group,
        "enable.auto.commit": False,
        "enable.auto.offset.store": False,
        "auto.offset.reset": "earliest",
        "isolation.level": "read_committed",
    }


def run() -> None:
    """Run the production FeedbackCandidate consumer until interrupted."""

    from confluent_kafka import Consumer, TopicPartition

    settings = get_settings()
    configure_logging(
        settings.log_level,
        service_name=settings.service_name,
        log_file=settings.log_file,
    )
    telemetry = configure_telemetry(settings)
    database_url = build_secret_provider(settings).get(SecretName.DATABASE_URL).reveal()
    database = Database(database_url)
    client = Consumer(_consumer_config(settings))
    client.subscribe([settings.kafka_work_order_topic])
    metrics = EventConsumerMetrics()
    metrics_server, _metrics_thread = start_http_server(
        settings.event_worker_metrics_port,
        addr=settings.worker_metrics_host,
        registry=metrics.registry,
    )
    consumer = ReplaySafeKafkaConsumer(
        cast(KafkaClient, client),
        FeedbackEventProcessor(database),
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


if __name__ == "__main__":
    run()
