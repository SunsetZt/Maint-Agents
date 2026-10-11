"""Checksum-verified loader for governed supervised RUL snapshots."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import Any

import pyarrow.parquet as parquet  # type: ignore[import-untyped]
from sqlalchemy import select

from industrial_ops_agent.data_pipeline.quality import (
    DatasetQualityReportError,
    verify_dataset_quality_report,
)
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.data_pipeline.training_gate import dataset_training_blockers
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    DataLineageRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.predictive_maintenance.rul_dataset_contract import (
    CONTRACT_VERSION,
    DATASET_COLUMNS,
    DATASET_SCHEMA,
    SIGNAL_ORDER,
)


class RulTrainingDatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RulSequenceSample:
    window_id: str
    asset_id: str
    sequence: tuple[tuple[float, ...], ...]
    mask: tuple[tuple[bool, ...], ...]
    lead_time_minutes: float


@dataclass(frozen=True, slots=True)
class RulTrainingDatasetBundle:
    method: str
    snapshot_id: str
    manifest_hash: str
    contract_version: str
    signal_order: tuple[str, ...]
    train: tuple[RulSequenceSample, ...]
    validation: tuple[RulSequenceSample, ...]
    split_hashes: dict[str, str]


class GovernedRulDatasetLoader:
    def __init__(self, database: Database, store: DatasetStore) -> None:
        self._database = database
        self._store = store

    def load(
        self,
        context: TenantContext,
        experiment: TrainingExperimentRecord,
    ) -> RulTrainingDatasetBundle:
        if context.tenant_id != experiment.tenant_id:
            raise RulTrainingDatasetError("experiment_tenant_mismatch")
        if experiment.method != "RUL_TRANSFORMER":
            raise RulTrainingDatasetError("rul_dataset_requires_rul_transformer_method")
        with self._database.transaction(context) as session:
            snapshot = session.scalar(
                select(DatasetSnapshotRecord).where(
                    DatasetSnapshotRecord.tenant_id == context.tenant_id,
                    DatasetSnapshotRecord.snapshot_id == experiment.dataset_snapshot_id,
                )
            )
            if snapshot is None:
                raise RulTrainingDatasetError("dataset_snapshot_not_visible")
            lineage = session.scalar(
                select(DataLineageRunRecord).where(
                    DataLineageRunRecord.tenant_id == context.tenant_id,
                    DataLineageRunRecord.snapshot_id == snapshot.snapshot_id,
                )
            )
            artifacts = list(
                session.scalars(
                    select(DatasetArtifactRecord).where(
                        DatasetArtifactRecord.tenant_id == context.tenant_id,
                        DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
                    )
                )
            )
            quality_artifacts = [item for item in artifacts if item.kind == "quality_report"]
            if dataset_training_blockers(snapshot, lineage, quality_artifacts):
                raise RulTrainingDatasetError("dataset_snapshot_is_not_training_eligible")
            if snapshot.contract_version != CONTRACT_VERSION:
                raise RulTrainingDatasetError("rul_dataset_contract_is_not_supported")
            if snapshot.manifest_hash != experiment.dataset_manifest_hash:
                raise RulTrainingDatasetError("experiment_manifest_binding_changed")
            manifest_key = snapshot.manifest_key
            manifest_hash = str(snapshot.manifest_hash)
            quality_key = snapshot.quality_report_key
            snapshot_identity = (
                snapshot.snapshot_id,
                snapshot.row_count,
                dict(snapshot.split_counts),
            )

        assert manifest_key is not None
        assert quality_key is not None
        assert len(quality_artifacts) == 1
        try:
            quality_payload = self._store.get_bytes(quality_key)
            verify_dataset_quality_report(snapshot, quality_artifacts[0], quality_payload)
        except (FileNotFoundError, DatasetQualityReportError) as exc:
            raise RulTrainingDatasetError("dataset_quality_report_integrity_failed") from exc
        manifest_payload = self._store.get_bytes(manifest_key)
        if not _digest_matches(manifest_payload, manifest_hash):
            raise RulTrainingDatasetError("dataset_manifest_hash_mismatch")
        manifest = _object(manifest_payload, "rul_dataset_manifest_is_invalid")
        expected = {
            "snapshot_id": snapshot_identity[0],
            "tenant_id": context.tenant_id,
            "contract_version": CONTRACT_VERSION,
            "target": "lead_time_minutes_from_alert_detection",
            "row_count": snapshot_identity[1],
            "split_counts": snapshot_identity[2],
            "lineage_status": "CONFIRMED",
            "signal_order": list(SIGNAL_ORDER),
            "training_eligible": True,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RulTrainingDatasetError("rul_dataset_manifest_identity_mismatch")
        contracts = _parquet_contracts(manifest)
        _cross_check_catalog(contracts, artifacts)
        samples: dict[str, tuple[RulSequenceSample, ...]] = {}
        split_hashes: dict[str, str] = {}
        group_keys: dict[str, set[str]] = {}
        for split in ("train", "validation"):
            contract = contracts[split]
            payload = self._store.get_bytes(contract[0])
            if len(payload) != contract[2] or not _digest_matches(payload, contract[1]):
                raise RulTrainingDatasetError(f"{split}_parquet_integrity_failed")
            frame = parquet.read_table(BytesIO(payload)).to_pandas().loc[:, list(DATASET_COLUMNS)]
            DATASET_SCHEMA.validate(frame, lazy=True)
            if len(frame) != contract[3]:
                raise RulTrainingDatasetError(f"{split}_row_count_mismatch")
            if not frame.empty and set(frame["split"].astype(str)) != {split}:
                raise RulTrainingDatasetError(f"{split}_contains_foreign_split_rows")
            if not frame.empty and set(frame["tenant_id"].astype(str)) != {context.tenant_id}:
                raise RulTrainingDatasetError(f"{split}_contains_foreign_tenant_rows")
            group_keys[split] = set(frame["group_key"].astype(str))
            samples[split] = tuple(_sample(row) for row in frame.to_dict("records"))
            split_hashes[split] = contract[1]
        if not samples["train"]:
            raise RulTrainingDatasetError("training_split_is_empty")
        if not samples["validation"]:
            raise RulTrainingDatasetError("validation_split_is_empty")
        if group_keys["train"] & group_keys["validation"]:
            raise RulTrainingDatasetError("equipment_leakage_between_splits")
        return RulTrainingDatasetBundle(
            method=experiment.method,
            snapshot_id=experiment.dataset_snapshot_id,
            manifest_hash=manifest_hash,
            contract_version=CONTRACT_VERSION,
            signal_order=SIGNAL_ORDER,
            train=samples["train"],
            validation=samples["validation"],
            split_hashes=split_hashes,
        )


def _sample(row: dict[str, Any]) -> RulSequenceSample:
    try:
        sequence_raw = json.loads(str(row["sequence_json"]))
        mask_raw = json.loads(str(row["mask_json"]))
    except json.JSONDecodeError as exc:
        raise RulTrainingDatasetError("rul_sequence_json_is_invalid") from exc
    if (
        not isinstance(sequence_raw, list)
        or not isinstance(mask_raw, list)
        or len(sequence_raw) != len(mask_raw)
        or len(sequence_raw) < 3
        or len(sequence_raw) != int(row["sample_count"])
    ):
        raise RulTrainingDatasetError("rul_sequence_shape_is_invalid")
    sequence: list[tuple[float, ...]] = []
    mask: list[tuple[bool, ...]] = []
    for values_raw, present_raw in zip(sequence_raw, mask_raw, strict=True):
        if (
            not isinstance(values_raw, list)
            or not isinstance(present_raw, list)
            or len(values_raw) != len(SIGNAL_ORDER)
            or len(present_raw) != len(SIGNAL_ORDER)
        ):
            raise RulTrainingDatasetError("rul_signal_shape_is_invalid")
        values = tuple(float(value) for value in values_raw)
        if not all(math.isfinite(value) for value in values):
            raise RulTrainingDatasetError("rul_sequence_contains_non_finite_value")
        if any(value not in {0, 1} for value in present_raw):
            raise RulTrainingDatasetError("rul_mask_is_invalid")
        sequence.append(values)
        mask.append(tuple(bool(value) for value in present_raw))
    target = float(row["lead_time_minutes"])
    if not math.isfinite(target) or not 0 < target <= 5_256_000:
        raise RulTrainingDatasetError("rul_target_is_invalid")
    return RulSequenceSample(
        window_id=str(row["window_id"]),
        asset_id=str(row["asset_id"]),
        sequence=tuple(sequence),
        mask=tuple(mask),
        lead_time_minutes=target,
    )


def _parquet_contracts(manifest: dict[str, Any]) -> dict[str, tuple[str, str, int, int, str]]:
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise RulTrainingDatasetError("rul_dataset_manifest_artifacts_are_invalid")
    contracts: dict[str, tuple[str, str, int, int, str]] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, dict) or raw.get("kind") != "parquet":
            continue
        split = raw.get("split")
        if split not in {"train", "validation", "test"} or split in contracts:
            raise RulTrainingDatasetError("rul_dataset_manifest_parquet_splits_are_invalid")
        try:
            contracts[str(split)] = (
                str(raw["object_key"]),
                str(raw["content_hash"]),
                int(raw["size_bytes"]),
                int(raw["row_count"]),
                str(raw["schema_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RulTrainingDatasetError("rul_dataset_manifest_artifact_is_invalid") from exc
    if set(contracts) != {"train", "validation", "test"}:
        raise RulTrainingDatasetError("rul_dataset_manifest_requires_all_three_splits")
    return contracts


def _cross_check_catalog(
    contracts: dict[str, tuple[str, str, int, int, str]],
    artifacts: list[DatasetArtifactRecord],
) -> None:
    catalog = {
        str(item.split): item
        for item in artifacts
        if item.kind == "parquet" and item.split is not None
    }
    if set(catalog) != set(contracts):
        raise RulTrainingDatasetError("rul_dataset_catalog_and_manifest_differ")
    expected_schema_hash = "sha256:" + sha256("\0".join(DATASET_COLUMNS).encode()).hexdigest()
    for split, contract in contracts.items():
        record = catalog[split]
        if (
            record.object_key != contract[0]
            or record.content_hash != contract[1]
            or record.size_bytes != contract[2]
            or record.row_count != contract[3]
            or record.schema_hash != expected_schema_hash
            or contract[4] != expected_schema_hash
        ):
            raise RulTrainingDatasetError("rul_dataset_catalog_and_manifest_differ")


def _object(payload: bytes, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RulTrainingDatasetError(reason) from exc
    if not isinstance(value, dict):
        raise RulTrainingDatasetError(reason)
    return value


def _digest_matches(payload: bytes, expected: str) -> bool:
    return expected == "sha256:" + sha256(payload).hexdigest()
