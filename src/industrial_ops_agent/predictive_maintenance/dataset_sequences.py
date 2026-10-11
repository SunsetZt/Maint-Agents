"""Shared telemetry sequence decoding and deterministic dataset partitioning."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from industrial_ops_agent.persistence.models import TelemetryEventRecord


def split_for_group(group_key: str) -> str:
    bucket = int(sha256(group_key.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def sequence_from_events(
    events: list[TelemetryEventRecord], *, signal_order: tuple[str, ...],
) -> tuple[list[list[float]], list[list[int]]]:
    sequence: list[list[float]] = []
    mask: list[list[int]] = []
    for event in events:
        values: list[float] = []
        present: list[int] = []
        invalid = set(str(item) for item in event.quality_flags_json) & {"SENSOR_FAULT", "MISSING"}
        for signal in signal_order:
            available = not invalid and signal in event.measurements_json
            values.append(float(event.measurements_json[signal]) if available else 0.0)
            present.append(1 if available else 0)
        sequence.append(values)
        mask.append(present)
    return sequence, mask


def parse_sequence_row(
    row: dict[str, Any], *, signal_order: tuple[str, ...],
    error_type: type[ValueError], prefix: str,
) -> tuple[list[tuple[float, ...]], list[tuple[bool, ...]]]:
    try:
        sequence_raw = json.loads(str(row["sequence_json"]))
        mask_raw = json.loads(str(row["mask_json"]))
    except json.JSONDecodeError as exc:
        raise error_type(f"{prefix}_sequence_json_is_invalid") from exc
    if (
        not isinstance(sequence_raw, list)
        or not isinstance(mask_raw, list)
        or len(sequence_raw) != len(mask_raw)
        or len(sequence_raw) < 3
        or len(sequence_raw) != int(row["sample_count"])
    ):
        raise error_type(f"{prefix}_sequence_shape_is_invalid")
    sequence: list[tuple[float, ...]] = []
    mask: list[tuple[bool, ...]] = []
    for values_raw, present_raw in zip(sequence_raw, mask_raw, strict=True):
        if (
            not isinstance(values_raw, list)
            or not isinstance(present_raw, list)
            or len(values_raw) != len(signal_order)
            or len(present_raw) != len(signal_order)
        ):
            raise error_type(f"{prefix}_signal_shape_is_invalid")
        values = tuple(float(value) for value in values_raw)
        if not all(math.isfinite(value) for value in values):
            raise error_type(f"{prefix}_sequence_contains_non_finite_value")
        if any(value not in {0, 1} for value in present_raw):
            raise error_type(f"{prefix}_mask_is_invalid")
        sequence.append(values)
        mask.append(tuple(bool(value) for value in present_raw))
    return sequence, mask


def parquet_contracts(
    manifest: dict[str, Any], *, error_type: type[ValueError], prefix: str,
) -> dict[str, tuple[str, str, int, int, str]]:
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise error_type(f"{prefix}dataset_manifest_artifacts_are_invalid")
    contracts: dict[str, tuple[str, str, int, int, str]] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, dict) or raw.get("kind") != "parquet":
            continue
        split = raw.get("split")
        if split not in {"train", "validation", "test"} or split in contracts:
            raise error_type(f"{prefix}dataset_manifest_parquet_splits_are_invalid")
        try:
            contracts[str(split)] = (
                str(raw["object_key"]),
                str(raw["content_hash"]),
                int(raw["size_bytes"]),
                int(raw["row_count"]),
                str(raw["schema_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise error_type(f"{prefix}dataset_manifest_artifact_is_invalid") from exc
    if set(contracts) != {"train", "validation", "test"}:
        raise error_type(f"{prefix}dataset_manifest_requires_all_three_splits")
    return contracts
