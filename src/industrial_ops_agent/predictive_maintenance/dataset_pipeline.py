"""Publish immutable telemetry sequence snapshots for governed model training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import pyarrow as arrow  # type: ignore[import-untyped]
import pyarrow.dataset as arrow_dataset  # type: ignore[import-untyped]
import pyarrow.parquet as parquet  # type: ignore[import-untyped]
from sqlalchemy import select

from industrial_ops_agent.data_pipeline.lineage import LineageEmitter, LineageUnavailable, event_now
from industrial_ops_agent.data_pipeline.storage import DatasetStore, ImmutableObjectExists
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AlertCandidateRecord,
    CurationRunRecord,
    DataLineageRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
    PredictiveOutcomeRecord,
    TelemetryEventRecord,
    TelemetryWindowRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.predictive_maintenance.dataset_contract import (
    CONTRACT_VERSION,
    DATASET_COLUMNS,
    SIGNAL_ORDER,
    SPLITS,
    prepare_rows,
)

MIN_TRAINING_WINDOWS = 20


class TelemetryDatasetPipelineError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TelemetryDatasetBuildResult:
    run_id: str
    snapshot_id: str
    status: str
    training_eligible: bool
    row_count: int
    split_counts: dict[str, int]
    input_manifest_hash: str
    manifest_key: str | None
    blocker_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Artifact:
    kind: str
    split: str | None
    object_key: str
    content: bytes
    row_count: int
    schema_hash: str

    @property
    def content_hash(self) -> str:
        return _digest_bytes(self.content)

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "split": self.split,
            "object_key": self.object_key,
            "content_hash": self.content_hash,
            "size_bytes": len(self.content),
            "row_count": self.row_count,
            "schema_hash": self.schema_hash,
        }


class TelemetryDatasetPipelineService:
    def __init__(
        self,
        database: Database,
        store: DatasetStore,
        lineage: LineageEmitter,
        *,
        code_version: str,
    ) -> None:
        self._database = database
        self._store = store
        self._lineage = lineage
        self._code_version = code_version

    def build_snapshot(
        self,
        context: TenantContext,
        *,
        window_start: datetime,
        window_end: datetime,
        engine: str,
    ) -> TelemetryDatasetBuildResult:
        start, end = _utc(window_start), _utc(window_end)
        if start >= end:
            raise ValueError("window_start must be before window_end")
        if engine not in {"local", "spark"}:
            raise ValueError("engine must be local or spark")
        now = datetime.now(UTC)
        run_id = str(uuid4())
        snapshot_id = f"telemetry-dataset-{uuid4().hex}"
        rows, exclusions, source_features = self._freeze_rows(
            context,
            window_start=start,
            window_end=end,
        )
        if not rows:
            self._record_failed(
                context,
                run_id=run_id,
                window_start=start,
                window_end=end,
                engine=engine,
                input_manifest_hash="sha256:empty",
                exclusions=exclusions,
                reason="telemetry_dataset_has_no_eligible_windows",
                now=now,
            )
            raise TelemetryDatasetPipelineError("telemetry_dataset_has_no_eligible_windows")
        prepared = prepare_rows(rows)
        input_manifest_hash = _digest_json(
            [
                {
                    "feature_snapshot_id": item["feature_snapshot_id"],
                    "sequence_hash": _digest_text(item["sequence_json"]),
                    "label": item["label"],
                    "outcome_id": item["outcome_id"],
                }
                for item in prepared
            ]
        )
        config_hash = _digest_json(
            {
                "contract_version": CONTRACT_VERSION,
                "engine": engine,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "signal_order": SIGNAL_ORDER,
            }
        )
        split_counts = {split: sum(item["split"] == split for item in prepared) for split in SPLITS}
        label_counts = {
            "unlabeled": sum(item["label"] == -1 for item in prepared),
            "normal": sum(item["label"] == 0 for item in prepared),
            "anomaly": sum(item["label"] == 1 for item in prepared),
        }
        blockers = _training_blockers(prepared, split_counts)
        training_eligible = not blockers
        published_prefix = f"tenant/{context.tenant_id}/telemetry-datasets/published/{snapshot_id}"
        staging_prefix = f"tenant/{context.tenant_id}/telemetry-datasets/staging/{run_id}"
        output_dataset = f"{published_prefix}/manifest.json"
        schema_hash = _digest_text("\0".join(DATASET_COLUMNS))
        try:
            self._lineage.emit(
                event_now(
                    "START",
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    tenant_id=context.tenant_id,
                    source_work_order_ids=(),
                    source_dataset_names=source_features,
                    output_dataset=output_dataset,
                    input_manifest_hash=input_manifest_hash,
                )
            )
            payloads = _parquet_payloads(prepared, engine=engine)
            quality_report = {
                "contract_version": CONTRACT_VERSION,
                "pandera_validation": "PASSED",
                "row_count": len(prepared),
                "split_counts": split_counts,
                "group_leakage_count": _group_leakage_count(prepared),
                "exclusions": exclusions,
                "label_counts": label_counts,
                "signal_order": list(SIGNAL_ORDER),
                "training_eligible": training_eligible,
                "blocker_codes": list(blockers),
            }
            if quality_report["group_leakage_count"]:
                raise TelemetryDatasetPipelineError("equipment_group_leakage_detected")
            quality_bytes = _json_bytes(quality_report)
            for split, content in payloads.items():
                _put_immutable(self._store, f"{staging_prefix}/{split}.parquet", content)
            _put_immutable(
                self._store,
                f"{staging_prefix}/quality-report.json",
                quality_bytes,
            )
            self._revalidate_windows(context, prepared)
            self._lineage.emit(
                event_now(
                    "COMPLETE",
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    tenant_id=context.tenant_id,
                    source_work_order_ids=(),
                    source_dataset_names=source_features,
                    output_dataset=output_dataset,
                    input_manifest_hash=input_manifest_hash,
                )
            )
        except Exception as exc:
            reason = _safe_reason(exc)
            self._record_failed(
                context,
                run_id=run_id,
                window_start=start,
                window_end=end,
                engine=engine,
                input_manifest_hash=input_manifest_hash,
                exclusions=exclusions,
                reason=reason,
                now=now,
            )
            raise TelemetryDatasetPipelineError(reason) from exc

        artifacts: list[_Artifact] = []
        for split in SPLITS:
            key = f"{published_prefix}/{split}.parquet"
            content = payloads[split]
            _put_immutable(self._store, key, content)
            artifacts.append(
                _Artifact("parquet", split, key, content, split_counts[split], schema_hash)
            )
        quality_key = f"{published_prefix}/quality-report.json"
        _put_immutable(self._store, quality_key, quality_bytes)
        artifacts.append(
            _Artifact(
                "quality_report",
                None,
                quality_key,
                quality_bytes,
                len(prepared),
                _digest_text(CONTRACT_VERSION),
            )
        )
        manifest = {
            "snapshot_id": snapshot_id,
            "run_id": run_id,
            "tenant_id": context.tenant_id,
            "contract_version": CONTRACT_VERSION,
            "code_version": self._code_version,
            "input_manifest_hash": input_manifest_hash,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "row_count": len(prepared),
            "split_counts": split_counts,
            "label_counts": label_counts,
            "signal_order": list(SIGNAL_ORDER),
            "source_feature_snapshot_ids": list(source_features),
            "lineage_status": "CONFIRMED",
            "training_eligible": training_eligible,
            "training_blockers": list(blockers),
            "artifacts": [item.manifest_entry() for item in artifacts],
        }
        manifest_bytes = _json_bytes(manifest)
        manifest_key = f"{published_prefix}/manifest.json"
        _put_immutable(self._store, manifest_key, manifest_bytes)
        self._record_published(
            context,
            run_id=run_id,
            snapshot_id=snapshot_id,
            window_start=start,
            window_end=end,
            engine=engine,
            config_hash=config_hash,
            input_manifest_hash=input_manifest_hash,
            exclusions=exclusions,
            manifest_key=manifest_key,
            manifest_hash=_digest_bytes(manifest_bytes),
            quality_key=quality_key,
            artifacts=artifacts,
            split_counts=split_counts,
            row_count=len(prepared),
            training_eligible=training_eligible,
            output_dataset=output_dataset,
            now=now,
        )
        return TelemetryDatasetBuildResult(
            run_id=run_id,
            snapshot_id=snapshot_id,
            status="CANDIDATE",
            training_eligible=training_eligible,
            row_count=len(prepared),
            split_counts=split_counts,
            input_manifest_hash=input_manifest_hash,
            manifest_key=manifest_key,
            blocker_codes=blockers,
        )

    def _freeze_rows(
        self,
        context: TenantContext,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]], tuple[str, ...]]:
        rows: list[dict[str, Any]] = []
        exclusions: list[dict[str, str]] = []
        source_features: list[str] = []
        with self._database.transaction(context) as session:
            windows = list(
                session.scalars(
                    select(TelemetryWindowRecord)
                    .where(
                        TelemetryWindowRecord.tenant_id == context.tenant_id,
                        TelemetryWindowRecord.window_start >= window_start,
                        TelemetryWindowRecord.window_end <= window_end,
                    )
                    .order_by(
                        TelemetryWindowRecord.asset_id,
                        TelemetryWindowRecord.window_start,
                    )
                )
            )
            outcomes = list(
                session.scalars(
                    select(PredictiveOutcomeRecord).where(
                        PredictiveOutcomeRecord.tenant_id == context.tenant_id,
                        PredictiveOutcomeRecord.alert_candidate_id.is_not(None),
                    )
                )
            )
            candidate_ids = {
                item.alert_candidate_id for item in outcomes if item.alert_candidate_id
            }
            candidates = (
                list(
                    session.scalars(
                        select(AlertCandidateRecord).where(
                            AlertCandidateRecord.tenant_id == context.tenant_id,
                            AlertCandidateRecord.alert_candidate_id.in_(candidate_ids),
                        )
                    )
                )
                if candidate_ids
                else []
            )
            candidate_by_id = {item.alert_candidate_id: item for item in candidates}
            outcome_by_window = {
                candidate_by_id[item.alert_candidate_id].window_id: item
                for item in outcomes
                if item.alert_candidate_id in candidate_by_id
            }
            for window in windows:
                if window.quality_status == "INSUFFICIENT" or window.sample_count < 3:
                    exclusions.append(
                        {"window_id": window.window_id, "reason": "window_insufficient"}
                    )
                    continue
                event_ids = tuple(str(item) for item in window.source_event_ids_json)
                events = list(
                    session.scalars(
                        select(TelemetryEventRecord)
                        .where(
                            TelemetryEventRecord.tenant_id == context.tenant_id,
                            TelemetryEventRecord.event_id.in_(event_ids),
                        )
                        .order_by(
                            TelemetryEventRecord.event_time,
                            TelemetryEventRecord.sequence,
                        )
                    )
                )
                if len(events) != len(event_ids):
                    exclusions.append(
                        {"window_id": window.window_id, "reason": "source_event_missing"}
                    )
                    continue
                sequence, mask = _sequence(events)
                outcome = outcome_by_window.get(window.window_id)
                label = _label(outcome)
                rows.append(
                    {
                        "tenant_id": context.tenant_id,
                        "window_id": window.window_id,
                        "feature_snapshot_id": window.feature_snapshot_id,
                        "asset_id": window.asset_id,
                        "schema_version": window.schema_version,
                        "window_start": _utc(window.window_start).isoformat(),
                        "window_end": _utc(window.window_end).isoformat(),
                        "sample_count": len(events),
                        "sequence_json": _json_text(sequence),
                        "mask_json": _json_text(mask),
                        "label": label,
                        "label_source": "OBSERVED_OUTCOME" if outcome else "UNLABELED",
                        "outcome_id": outcome.outcome_id if outcome else "",
                    }
                )
                source_features.append(window.feature_snapshot_id)
        return rows, exclusions, tuple(sorted(source_features))

    def _revalidate_windows(
        self,
        context: TenantContext,
        rows: list[dict[str, Any]],
    ) -> None:
        with self._database.transaction(context) as session:
            for row in rows:
                window = session.scalar(
                    select(TelemetryWindowRecord).where(
                        TelemetryWindowRecord.tenant_id == context.tenant_id,
                        TelemetryWindowRecord.window_id == row["window_id"],
                    )
                )
                if (
                    window is None
                    or window.feature_snapshot_id != row["feature_snapshot_id"]
                    or window.sample_count != row["sample_count"]
                ):
                    raise TelemetryDatasetPipelineError("telemetry_window_changed_before_publish")

    def _record_published(
        self,
        context: TenantContext,
        *,
        run_id: str,
        snapshot_id: str,
        window_start: datetime,
        window_end: datetime,
        engine: str,
        config_hash: str,
        input_manifest_hash: str,
        exclusions: list[dict[str, str]],
        manifest_key: str,
        manifest_hash: str,
        quality_key: str,
        artifacts: list[_Artifact],
        split_counts: dict[str, int],
        row_count: int,
        training_eligible: bool,
        output_dataset: str,
        now: datetime,
    ) -> None:
        with self._database.transaction(context) as session:
            session.add(
                CurationRunRecord(
                    run_id=run_id,
                    tenant_id=context.tenant_id,
                    window_start=window_start,
                    window_end=window_end,
                    engine=engine,
                    contract_version=CONTRACT_VERSION,
                    code_version=self._code_version,
                    config_hash=config_hash,
                    input_manifest_hash=input_manifest_hash,
                    input_count=row_count,
                    exclusion_report=exclusions,
                    status="COMPLETED",
                    failure_reason=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            session.add(
                DatasetSnapshotRecord(
                    snapshot_id=snapshot_id,
                    tenant_id=context.tenant_id,
                    run_id=run_id,
                    status="CANDIDATE",
                    contract_version=CONTRACT_VERSION,
                    input_manifest_hash=input_manifest_hash,
                    manifest_key=manifest_key,
                    manifest_hash=manifest_hash,
                    quality_report_key=quality_key,
                    row_count=row_count,
                    split_counts=split_counts,
                    source_work_order_ids=[],
                    lineage_status="CONFIRMED",
                    training_eligible=training_eligible,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            for artifact in artifacts:
                session.add(
                    DatasetArtifactRecord(
                        artifact_id=f"artifact-{uuid4().hex}",
                        tenant_id=context.tenant_id,
                        snapshot_id=snapshot_id,
                        kind=artifact.kind,
                        split=artifact.split,
                        object_key=artifact.object_key,
                        content_hash=artifact.content_hash,
                        size_bytes=len(artifact.content),
                        row_count=artifact.row_count,
                        schema_hash=artifact.schema_hash,
                        created_at=now,
                        updated_at=now,
                    )
                )
            session.add(
                DataLineageRunRecord(
                    lineage_id=f"lineage-{uuid4().hex}",
                    tenant_id=context.tenant_id,
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    openlineage_run_id=run_id,
                    job_namespace=self._lineage.namespace,
                    job_name=self._lineage.job_name,
                    status="CONFIRMED",
                    source_work_order_ids=[],
                    output_dataset=output_dataset,
                    failure_reason=None,
                    created_at=now,
                    updated_at=now,
                )
            )

    def _record_failed(
        self,
        context: TenantContext,
        *,
        run_id: str,
        window_start: datetime,
        window_end: datetime,
        engine: str,
        input_manifest_hash: str,
        exclusions: list[dict[str, str]],
        reason: str,
        now: datetime,
    ) -> None:
        with self._database.transaction(context) as session:
            session.add(
                CurationRunRecord(
                    run_id=run_id,
                    tenant_id=context.tenant_id,
                    window_start=window_start,
                    window_end=window_end,
                    engine=engine,
                    contract_version=CONTRACT_VERSION,
                    code_version=self._code_version,
                    config_hash=_digest_text(f"{engine}:{window_start}:{window_end}"),
                    input_manifest_hash=input_manifest_hash,
                    input_count=0,
                    exclusion_report=exclusions,
                    status="FAILED",
                    failure_reason=reason,
                    created_at=now,
                    updated_at=now,
                )
            )


def _training_blockers(
    rows: list[dict[str, Any]],
    split_counts: dict[str, int],
) -> tuple[str, ...]:
    blockers: list[str] = []
    if len(rows) < MIN_TRAINING_WINDOWS:
        blockers.append("insufficient_window_count")
    if split_counts["train"] == 0:
        blockers.append("training_split_empty")
    if split_counts["validation"] == 0:
        blockers.append("validation_split_empty")
    if len({item["asset_id"] for item in rows}) < 2:
        blockers.append("insufficient_equipment_diversity")
    return tuple(blockers)


def _sequence(events: list[TelemetryEventRecord]) -> tuple[list[list[float]], list[list[int]]]:
    sequence: list[list[float]] = []
    mask: list[list[int]] = []
    for event in events:
        values: list[float] = []
        present: list[int] = []
        invalid = set(str(item) for item in event.quality_flags_json) & {"SENSOR_FAULT", "MISSING"}
        for signal in SIGNAL_ORDER:
            available = not invalid and signal in event.measurements_json
            values.append(float(event.measurements_json[signal]) if available else 0.0)
            present.append(1 if available else 0)
        sequence.append(values)
        mask.append(present)
    return sequence, mask


def _label(outcome: PredictiveOutcomeRecord | None) -> int:
    if outcome is None:
        return -1
    if outcome.outcome_type == "TRUE_POSITIVE":
        return 1
    if outcome.outcome_type == "FALSE_POSITIVE":
        return 0
    return -1


def _parquet_payloads(rows: list[dict[str, Any]], *, engine: str) -> dict[str, bytes]:
    prepared = prepare_rows(rows)
    if engine == "local":
        return {
            split: _parquet_bytes([item for item in prepared if item["split"] == split])
            for split in SPLITS
        }
    with TemporaryDirectory(prefix="industrial-ops-telemetry-spark-") as directory:
        from industrial_ops_agent.predictive_maintenance.telemetry_spark import (
            build_telemetry_spark_snapshot,
        )

        root = build_telemetry_spark_snapshot(prepared, Path(directory) / "dataset")
        payloads: dict[str, bytes] = {}
        for split in SPLITS:
            partition = root / f"split={split}"
            if not partition.exists():
                payloads[split] = _parquet_bytes([])
                continue
            table = arrow_dataset.dataset(partition, format="parquet").to_table()
            frame_rows = table.to_pylist()
            for item in frame_rows:
                item["split"] = split
            payloads[split] = _parquet_bytes(frame_rows)
        return payloads


def _parquet_bytes(rows: list[dict[str, Any]]) -> bytes:
    if rows:
        table = arrow.Table.from_pylist(rows, schema=_arrow_schema())
    else:
        table = arrow.Table.from_pylist([], schema=_arrow_schema())
    target = BytesIO()
    parquet.write_table(table, target, compression="zstd", use_dictionary=True)
    return target.getvalue()


def _arrow_schema() -> arrow.Schema:
    return arrow.schema(
        [
            arrow.field(
                name, arrow.int64() if name in {"sample_count", "label"} else arrow.string()
            )
            for name in DATASET_COLUMNS
        ]
    )


def _group_leakage_count(rows: list[dict[str, Any]]) -> int:
    assignments: dict[str, set[str]] = {}
    for row in rows:
        assignments.setdefault(str(row["group_key"]), set()).add(str(row["split"]))
    return sum(len(splits) > 1 for splits in assignments.values())


def _put_immutable(store: DatasetStore, key: str, content: bytes) -> None:
    try:
        store.put_bytes(key, content)
    except ImmutableObjectExists:
        if store.get_bytes(key) != content:
            raise TelemetryDatasetPipelineError("immutable_telemetry_artifact_conflict") from None


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: Any) -> bytes:
    return _json_text(value).encode()


def _digest_json(value: Any) -> str:
    return _digest_bytes(_json_bytes(value))


def _digest_text(value: str) -> str:
    return "sha256:" + sha256(value.encode()).hexdigest()


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + sha256(value).hexdigest()


def _safe_reason(exc: Exception) -> str:
    if isinstance(exc, LineageUnavailable):
        return "openlineage_unavailable"
    value = str(exc).strip().lower().replace(" ", "_")
    safe = "".join(character for character in value if character.isalnum() or character in "_:-")
    return safe[:255] or "telemetry_dataset_pipeline_failed"
