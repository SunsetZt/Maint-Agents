"""Versioned Parquet contract for supervised remaining-useful-life training."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pandera.pandas as pa

CONTRACT_VERSION = "industrial-rul-sequence-v1"
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
    "alert_candidate_id",
    "asset_id",
    "schema_version",
    "window_start",
    "window_end",
    "detected_at",
    "failure_observed_at",
    "sample_count",
    "sequence_json",
    "mask_json",
    "lead_time_minutes",
    "outcome_id",
    "outcome_evidence_digest",
    "group_key",
    "split",
)

DATASET_SCHEMA = pa.DataFrameSchema(
    {
        "tenant_id": pa.Column(str, nullable=False),
        "window_id": pa.Column(str, nullable=False, unique=True),
        "feature_snapshot_id": pa.Column(str, nullable=False, unique=True),
        "alert_candidate_id": pa.Column(str, nullable=False, unique=True),
        "asset_id": pa.Column(str, nullable=False),
        "schema_version": pa.Column(str, nullable=False),
        "window_start": pa.Column(str, nullable=False),
        "window_end": pa.Column(str, nullable=False),
        "detected_at": pa.Column(str, nullable=False),
        "failure_observed_at": pa.Column(str, nullable=False),
        "sample_count": pa.Column(int, nullable=False, checks=pa.Check.ge(3)),
        "sequence_json": pa.Column(str, nullable=False),
        "mask_json": pa.Column(str, nullable=False),
        "lead_time_minutes": pa.Column(
            float,
            nullable=False,
            checks=[pa.Check.gt(0), pa.Check.le(5_256_000)],
        ),
        "outcome_id": pa.Column(str, nullable=False, unique=True),
        "outcome_evidence_digest": pa.Column(
            str,
            nullable=False,
            checks=pa.Check.str_matches(r"^sha256:[0-9a-f]{64}$"),
        ),
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
        raise ValueError("RUL dataset grouping identity is missing")
    return "rul-equipment-group:" + sha256(f"{tenant_id}\0{asset_id}".encode()).hexdigest()


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
    prepared.sort(key=lambda item: (item["group_key"], item["detected_at"], item["window_id"]))
    frame = pd.DataFrame(prepared, columns=DATASET_COLUMNS)
    DATASET_SCHEMA.validate(frame, lazy=True)
    return prepared
