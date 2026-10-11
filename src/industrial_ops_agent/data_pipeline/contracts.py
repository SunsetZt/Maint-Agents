"""One versioned transformation contract shared by local and Spark execution."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pandera.pandas as pa

CONTRACT_VERSION = "industrial-feedback-dataset-v1"
SYNTHETIC_AUGMENTATION_CONTRACT = "industrial-synthetic-augmentation-v1"
SIMULATED_MULTIMODAL_CONTRACT = "industrial-simulated-multimodal-v1"
SPLITS = ("train", "validation", "test")


def production_release_blocker_for_dataset_contract(
    contract_version: str,
) -> str | None:
    """Return the shared production blocker for simulation-only datasets."""

    if contract_version == SIMULATED_MULTIMODAL_CONTRACT:
        return "simulated_multimodal_snapshot_is_not_production_releasable"
    return None


DATASET_COLUMNS = (
    "tenant_id",
    "candidate_id",
    "candidate_version",
    "work_order_id",
    "incident_id",
    "device_id",
    "source_event_id",
    "source_content_hash",
    "redacted_content_json",
    "labels_json",
    "lineage_json",
    "occurred_at",
    "near_duplicate_group",
    "group_key",
    "split",
)

DATASET_SCHEMA = pa.DataFrameSchema(
    {
        "tenant_id": pa.Column(str, nullable=False),
        "candidate_id": pa.Column(str, nullable=False, unique=True),
        "candidate_version": pa.Column(int, nullable=False, checks=pa.Check.ge(1)),
        "work_order_id": pa.Column(str, nullable=False),
        "incident_id": pa.Column(str, nullable=False),
        "device_id": pa.Column(str, nullable=False),
        "source_event_id": pa.Column(str, nullable=False),
        "source_content_hash": pa.Column(str, nullable=False),
        "redacted_content_json": pa.Column(str, nullable=False),
        "labels_json": pa.Column(str, nullable=False),
        "lineage_json": pa.Column(str, nullable=False),
        "occurred_at": pa.Column(str, nullable=False),
        "near_duplicate_group": pa.Column(str, nullable=False),
        "group_key": pa.Column(str, nullable=False),
        "split": pa.Column(str, nullable=False, checks=pa.Check.isin(SPLITS)),
    },
    strict=True,
    ordered=True,
    coerce=True,
    name=CONTRACT_VERSION,
)


class UnsafeGrouping(ValueError):
    """A row cannot be split without risking leakage between dataset splits."""


def group_key_for(row: dict[str, Any]) -> str:
    required = ("tenant_id", "device_id", "work_order_id", "occurred_at")
    missing = [name for name in required if not str(row.get(name, "")).strip()]
    if missing:
        raise UnsafeGrouping("missing grouping fields: " + ",".join(missing))
    near_duplicate_group = str(row.get("near_duplicate_group", "")).strip()
    if not near_duplicate_group:
        raise UnsafeGrouping("near_duplicate_group is required")
    # A near-duplicate cluster takes precedence over case/device grouping so that
    # equivalent samples can never leak across splits, even across work orders.
    if near_duplicate_group:
        basis = f"{row['tenant_id']}\0near-duplicate\0{near_duplicate_group}"
    else:  # pragma: no cover - guarded above; documents the composite contract
        month = str(row["occurred_at"])[:7]
        basis = f"{row['tenant_id']}\0{row['device_id']}\0{row['work_order_id']}\0{month}"
    return "group:" + sha256(basis.encode()).hexdigest()


def split_for_group(group_key: str) -> str:
    bucket = int(sha256(group_key.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def prepare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(source) for source in rows]
    parent = list(range(len(normalized)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    token_owner: dict[str, int] = {}
    row_tokens: list[tuple[str, str]] = []
    for index, row in enumerate(normalized):
        # Validate every grouping field through the public single-row contract.
        group_key_for(row)
        month = str(row["occurred_at"])[:7]
        tokens = (
            f"case:{row['tenant_id']}:{row['device_id']}:{row['work_order_id']}:{month}",
            f"near:{row['tenant_id']}:{row['near_duplicate_group']}",
        )
        row_tokens.append(tokens)
        for token in tokens:
            owner = token_owner.setdefault(token, index)
            union(index, owner)

    component_tokens: dict[int, set[str]] = {}
    for index, tokens in enumerate(row_tokens):
        component_tokens.setdefault(find(index), set()).update(tokens)

    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(normalized):
        grouped_tokens = sorted(component_tokens[find(index)])
        row["group_key"] = "group:" + sha256("\0".join(grouped_tokens).encode()).hexdigest()
        row["split"] = split_for_group(row["group_key"])
        prepared.append({name: row[name] for name in DATASET_COLUMNS})
    prepared.sort(key=lambda item: (item["group_key"], item["candidate_id"]))
    frame = pd.DataFrame(prepared, columns=DATASET_COLUMNS)
    DATASET_SCHEMA.validate(frame, lazy=True)
    return prepared
