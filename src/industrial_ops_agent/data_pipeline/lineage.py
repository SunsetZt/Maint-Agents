"""OpenLineage delivery boundary and deterministic behavior-test adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LineageEvent:
    event_type: str
    event_time: datetime
    run_id: str
    snapshot_id: str
    tenant_id: str
    source_work_order_ids: tuple[str, ...]
    output_dataset: str
    input_manifest_hash: str
    source_dataset_names: tuple[str, ...] = ()


class LineageEmitter(Protocol):
    namespace: str
    job_name: str

    def emit(self, event: LineageEvent) -> None: ...


@dataclass(slots=True)
class RecordingLineageEmitter:
    namespace: str = "industrial-ops-test"
    job_name: str = "build_dataset_snapshot"
    events: list[LineageEvent] = field(default_factory=list)

    def emit(self, event: LineageEvent) -> None:
        self.events.append(event)


@dataclass(slots=True)
class UnavailableLineageEmitter:
    reason: str
    namespace: str = "industrial-ops"
    job_name: str = "build_dataset_snapshot"

    def emit(self, event: LineageEvent) -> None:
        raise LineageUnavailable(self.reason)


class LineageUnavailable(RuntimeError):
    pass


class OpenLineageEmitter:
    """Emit START/COMPLETE events through the official OpenLineage client."""

    def __init__(
        self,
        *,
        url: str,
        namespace: str,
        job_name: str = "build_dataset_snapshot",
    ) -> None:
        from openlineage.client import OpenLineageClient

        self.namespace = namespace
        self.job_name = job_name
        self._client = OpenLineageClient(url=url)

    def emit(self, event: LineageEvent) -> None:
        from openlineage.client.run import Dataset, Job, Run, RunEvent, RunState

        state = RunState.START if event.event_type == "START" else RunState.COMPLETE
        inputs = (
            [
                Dataset(
                    namespace=f"{self.namespace}.telemetry-windows",
                    name=name,
                    facets={},
                )
                for name in event.source_dataset_names
            ]
            if event.source_dataset_names
            else [
                Dataset(
                    namespace=f"{self.namespace}.work-orders",
                    name=work_order_id,
                    facets={},
                )
                for work_order_id in event.source_work_order_ids
            ]
        )
        output = Dataset(
            namespace=f"{self.namespace}.datasets",
            name=event.output_dataset,
            facets={},
        )
        run_event = RunEvent(
            eventType=state,
            eventTime=event.event_time.isoformat(),
            run=Run(runId=event.run_id, facets={}),
            job=Job(namespace=self.namespace, name=self.job_name, facets={}),
            inputs=inputs,
            outputs=[output],
            producer="https://industrial-ops-agent.local/data-pipeline",
        )
        try:
            self._client.emit(run_event)
        except Exception as exc:  # transport implementations expose different errors
            raise LineageUnavailable(str(exc)) from exc


def event_now(
    event_type: str,
    *,
    run_id: str,
    snapshot_id: str,
    tenant_id: str,
    source_work_order_ids: tuple[str, ...],
    output_dataset: str,
    input_manifest_hash: str,
    source_dataset_names: tuple[str, ...] = (),
) -> LineageEvent:
    return LineageEvent(
        event_type=event_type,
        event_time=datetime.now(UTC),
        run_id=run_id,
        snapshot_id=snapshot_id,
        tenant_id=tenant_id,
        source_work_order_ids=source_work_order_ids,
        output_dataset=output_dataset,
        input_manifest_hash=input_manifest_hash,
        source_dataset_names=source_dataset_names,
    )
