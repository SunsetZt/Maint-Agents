"""Versioned Parquet contract for equipment telemetry sequence training."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pandera.pandas as pa

CONTRACT_VERSION = "industrial-telemetry-sequence-v1"
SPLITS = ("train", "validation", "test")
SIGNAL_ORDER = (
    "vibration_rms_mm_s",
    "bearing_temperature_c",
    "motor_current_a",
    "rpm",
)
DATASET_COLUMNS = (
    "tenant_id",
    "window_id",
    "feature_snapshot_id",
    "asset_id",
    "schema_version",
    "window_start",
    "window_end",
    "sample_count",
    "sequence_json",
    "mask_json",
    "label",
    "label_source",
    "outcome_id",
    "group_key",
    "split",
)

DATASET_SCHEMA = pa.DataFrameSchema(
    {
        "tenant_id": pa.Column(str, nullable=False),
        "window_id": pa.Column(str, nullable=False, unique=True),
        "feature_snapshot_id": pa.Column(str, nullable=False, unique=True),
        "asset_id": pa.Column(str, nullable=False),
        "schema_version": pa.Column(str, nullable=False),
        "window_start": pa.Column(str, nullable=False),
        "window_end": pa.Column(str, nullable=False),
        "sample_count": pa.Column(int, nullable=False, checks=pa.Check.ge(3)),
        "sequence_json": pa.Column(str, nullable=False),
        "mask_json": pa.Column(str, nullable=False),
        "label": pa.Column(int, nullable=False, checks=pa.Check.isin((-1, 0, 1))),
        "label_source": pa.Column(
            str,
            nullable=False,
            checks=pa.Check.isin(("UNLABELED", "OBSERVED_OUTCOME")),
        ),
        "outcome_id": pa.Column(str, nullable=False),
        "group_key": pa.Column(str, nullable=False),
        "split": pa.Column(str, nullable=False, checks=pa.Check.isin(SPLITS)),
    },
    strict=True,
    ordered=True,
    coerce=True,
    name=CONTRACT_VERSION,
)


def group_key_for(tenant_id: str, asset_id: str) -> str:
    if not tenant_id or not asset_id:
        raise ValueError("telemetry dataset grouping identity is missing")
    return "equipment-group:" + sha256(f"{tenant_id}\0{asset_id}".encode()).hexdigest()


def split_for_group(group_key: str) -> str:
    bucket = int(sha256(group_key.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def prepare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["group_key"] = group_key_for(str(row["tenant_id"]), str(row["asset_id"]))
        row["split"] = split_for_group(row["group_key"])
        prepared.append({name: row[name] for name in DATASET_COLUMNS})
    prepared.sort(key=lambda item: (item["group_key"], item["window_start"], item["window_id"]))
    frame = pd.DataFrame(prepared, columns=DATASET_COLUMNS)
    DATASET_SCHEMA.validate(frame, lazy=True)
    return prepared
