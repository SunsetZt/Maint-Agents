"""Build governed training snapshots augmented with reviewed synthetic images."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

import pyarrow as arrow  # type: ignore[import-untyped]
import pyarrow.parquet as parquet  # type: ignore[import-untyped]
from sqlalchemy import select

from industrial_ops_agent.data_pipeline.contracts import (
    CONTRACT_VERSION,
    DATASET_COLUMNS,
    DATASET_SCHEMA,
    SPLITS,
    SYNTHETIC_AUGMENTATION_CONTRACT,
)
from industrial_ops_agent.data_pipeline.lineage import LineageEmitter, event_now
from industrial_ops_agent.data_pipeline.quality import (
    DatasetQualityReportError,
    verify_dataset_quality_report,
)
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.data_pipeline.training_gate import dataset_training_blockers
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CurationRunRecord,
    DataLineageRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.training.synthetic_media import (
    SyntheticMediaError,
    verify_synthetic_media_bundle,
)

MAX_SYNTHETIC_BUNDLES_PER_SNAPSHOT = 1_000


class SyntheticSnapshotError(RuntimeError):
    """A base snapshot or synthetic bundle cannot be safely published."""


@dataclass(frozen=True, slots=True)
class ReviewedSyntheticBundle:
    manifest: dict[str, Any]
    image_png: bytes
    source_image: bytes
    mask_image: bytes


@dataclass(frozen=True, slots=True)
class SyntheticSnapshotResult:
    run_id: str
    snapshot_id: str
    base_snapshot_id: str
    manifest_key: str
    manifest_hash: str
    row_count: int
    split_counts: dict[str, int]
    real_sample_count: int
    synthetic_sample_count: int


@dataclass(slots=True)
class _Artifact:
    kind: str
    split: str | None
    object_key: str
    content: bytes
    row_count: int
    schema_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return _digest(self.content)

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "split": self.split,
            "object_key": self.object_key,
            "content_hash": self.content_hash,
            "size_bytes": len(self.content),
            "row_count": self.row_count,
            "schema_hash": self.schema_hash,
            **self.metadata,
        }


@dataclass(frozen=True, slots=True)
class _BaseSnapshot:
    record: DatasetSnapshotRecord
    run: CurationRunRecord
    rows_by_split: dict[str, list[dict[str, Any]]]
    inherited_artifacts: tuple[_Artifact, ...]
    manifest: dict[str, Any]
    real_sample_count: int
    synthetic_sample_count: int


class SyntheticDatasetSnapshotService:
    """Derive a train-only synthetic augmentation from one governed snapshot."""

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

    def augment_snapshot(
        self,
        context: TenantContext,
        *,
        base_snapshot_id: str,
        bundles: list[ReviewedSyntheticBundle],
        occurred_at: datetime | None = None,
    ) -> SyntheticSnapshotResult:
        if not bundles or len(bundles) > MAX_SYNTHETIC_BUNDLES_PER_SNAPSHOT:
            raise SyntheticSnapshotError("synthetic_bundle_count_is_invalid")
        base = self._load_base_snapshot(context, base_snapshot_id)
        reviewed = _validate_bundles(bundles)
        existing_provenance = _existing_provenance_ids(base.manifest)
        if existing_provenance & set(reviewed):
            raise SyntheticSnapshotError("synthetic_bundle_already_exists_in_base_snapshot")

        now = occurred_at or datetime.now(UTC)
        if now.tzinfo is None:
            raise SyntheticSnapshotError("synthetic_snapshot_time_must_be_timezone_aware")
        now = now.astimezone(UTC)
        run_id = str(uuid4())
        snapshot_id = f"dataset-{uuid4().hex}"
        published_prefix = (
            f"tenant/{context.tenant_id}/datasets/snapshots/{snapshot_id}/published"
        )
        rows = {split: list(base.rows_by_split[split]) for split in SPLITS}
        inherited, key_mapping = _relocate_inherited_artifacts(
            base.inherited_artifacts,
            published_prefix=published_prefix,
        )
        synthetic_artifacts: list[_Artifact] = []
        synthetic_manifest_entries: list[dict[str, Any]] = []
        for provenance_id, bundle in reviewed.items():
            source_row = _governed_source_row(base, bundle)
            row, artifacts, entry = _synthetic_sample(
                context,
                bundle,
                source_row=source_row,
                provenance_id=provenance_id,
                published_prefix=published_prefix,
            )
            rows["train"].append(row)
            synthetic_artifacts.extend(artifacts)
            synthetic_manifest_entries.append(entry)

        _validate_rows(rows, context.tenant_id)
        split_counts = {split: len(rows[split]) for split in SPLITS}
        synthetic_count = base.synthetic_sample_count + len(reviewed)
        real_count = base.real_sample_count
        row_count = sum(split_counts.values())
        if real_count + synthetic_count != row_count:
            raise SyntheticSnapshotError("synthetic_origin_counts_do_not_match_rows")

        schema_hash = _digest("\0".join(DATASET_COLUMNS).encode())
        parquet_artifacts = [
            _Artifact(
                kind="parquet",
                split=split,
                object_key=f"{published_prefix}/{split}.parquet",
                content=_parquet_bytes(rows[split]),
                row_count=split_counts[split],
                schema_hash=schema_hash,
            )
            for split in SPLITS
        ]
        quality_document = {
            "contract_version": CONTRACT_VERSION,
            "pandera_validation": "PASSED",
            "row_count": row_count,
            "split_counts": split_counts,
            "group_leakage_count": 0,
            "exclusions": [],
            "sample_origin_counts": {"real": real_count, "synthetic": synthetic_count},
            "synthetic_split_counts": {
                "train": synthetic_count,
                "validation": 0,
                "test": 0,
            },
        }
        quality_bytes = _json_bytes(quality_document)
        quality_artifact = _Artifact(
            kind="quality_report",
            split=None,
            object_key=f"{published_prefix}/quality-report.json",
            content=quality_bytes,
            row_count=row_count,
            schema_hash=_digest(CONTRACT_VERSION.encode()),
        )
        artifacts = [
            *parquet_artifacts,
            quality_artifact,
            *inherited,
            *synthetic_artifacts,
        ]
        input_manifest_hash = _canonical_digest(
            {
                "base_snapshot_id": base.record.snapshot_id,
                "base_manifest_hash": base.record.manifest_hash,
                "synthetic_manifest_hashes": sorted(reviewed),
            }
        )
        output_dataset = f"{published_prefix}/manifest.json"
        inherited_synthetic_entries = _relocate_synthetic_entries(
            base.manifest,
            key_mapping,
        )
        manifest = {
            "snapshot_id": snapshot_id,
            "run_id": run_id,
            "tenant_id": context.tenant_id,
            "contract_version": CONTRACT_VERSION,
            "augmentation_contract_version": SYNTHETIC_AUGMENTATION_CONTRACT,
            "code_version": self._code_version,
            "input_manifest_hash": input_manifest_hash,
            "base_snapshot_id": base.record.snapshot_id,
            "base_manifest_hash": base.record.manifest_hash,
            "requested_by_subject_id": context.subject_id,
            "window_start": _iso(base.run.window_start),
            "window_end": _iso(base.run.window_end),
            "row_count": row_count,
            "split_counts": split_counts,
            "sample_origin_counts": {"real": real_count, "synthetic": synthetic_count},
            "synthetic_split_counts": {
                "train": synthetic_count,
                "validation": 0,
                "test": 0,
            },
            "evaluation_eligible": False,
            "source_work_order_ids": list(base.record.source_work_order_ids),
            "lineage_status": "CONFIRMED",
            "synthetic_media": [
                *inherited_synthetic_entries,
                *synthetic_manifest_entries,
            ],
            "artifacts": [artifact.manifest_entry() for artifact in artifacts],
        }
        manifest_bytes = _json_bytes(manifest)
        manifest_hash = _digest(manifest_bytes)

        self._lineage.emit(
            event_now(
                "START",
                run_id=run_id,
                snapshot_id=snapshot_id,
                tenant_id=context.tenant_id,
                source_work_order_ids=tuple(base.record.source_work_order_ids),
                source_dataset_names=(
                    base.record.snapshot_id,
                    *(f"synthetic:{value}" for value in sorted(reviewed)),
                ),
                output_dataset=output_dataset,
                input_manifest_hash=input_manifest_hash,
            )
        )
        for artifact in artifacts:
            self._store.put_bytes(artifact.object_key, artifact.content)
        self._store.put_bytes(output_dataset, manifest_bytes)
        self._lineage.emit(
            event_now(
                "COMPLETE",
                run_id=run_id,
                snapshot_id=snapshot_id,
                tenant_id=context.tenant_id,
                source_work_order_ids=tuple(base.record.source_work_order_ids),
                source_dataset_names=(
                    base.record.snapshot_id,
                    *(f"synthetic:{value}" for value in sorted(reviewed)),
                ),
                output_dataset=output_dataset,
                input_manifest_hash=input_manifest_hash,
            )
        )
        self._record_snapshot(
            context,
            base=base,
            run_id=run_id,
            snapshot_id=snapshot_id,
            input_manifest_hash=input_manifest_hash,
            manifest_key=output_dataset,
            manifest_hash=manifest_hash,
            quality_key=quality_artifact.object_key,
            artifacts=artifacts,
            split_counts=split_counts,
            now=now,
        )
        return SyntheticSnapshotResult(
            run_id=run_id,
            snapshot_id=snapshot_id,
            base_snapshot_id=base.record.snapshot_id,
            manifest_key=output_dataset,
            manifest_hash=manifest_hash,
            row_count=row_count,
            split_counts=split_counts,
            real_sample_count=real_count,
            synthetic_sample_count=synthetic_count,
        )

    def _load_base_snapshot(
        self,
        context: TenantContext,
        snapshot_id: str,
    ) -> _BaseSnapshot:
        with self._database.transaction(context) as session:
            snapshot = session.scalar(
                select(DatasetSnapshotRecord).where(
                    DatasetSnapshotRecord.tenant_id == context.tenant_id,
                    DatasetSnapshotRecord.snapshot_id == snapshot_id,
                )
            )
            if snapshot is None:
                raise SyntheticSnapshotError("base_snapshot_is_not_visible")
            run = session.scalar(
                select(CurationRunRecord).where(
                    CurationRunRecord.tenant_id == context.tenant_id,
                    CurationRunRecord.run_id == snapshot.run_id,
                )
            )
            lineage = session.scalar(
                select(DataLineageRunRecord).where(
                    DataLineageRunRecord.tenant_id == context.tenant_id,
                    DataLineageRunRecord.snapshot_id == snapshot.snapshot_id,
                )
            )
            catalog = list(
                session.scalars(
                    select(DatasetArtifactRecord).where(
                        DatasetArtifactRecord.tenant_id == context.tenant_id,
                        DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
                    )
                )
            )
        quality_artifacts = [item for item in catalog if item.kind == "quality_report"]
        if (
            run is None
            or snapshot.contract_version != CONTRACT_VERSION
            or dataset_training_blockers(snapshot, lineage, quality_artifacts)
        ):
            raise SyntheticSnapshotError("base_snapshot_is_not_training_eligible")
        assert snapshot.manifest_key is not None
        assert snapshot.manifest_hash is not None
        assert snapshot.quality_report_key is not None
        try:
            verify_dataset_quality_report(
                snapshot,
                quality_artifacts[0],
                self._store.get_bytes(snapshot.quality_report_key),
            )
        except (IndexError, FileNotFoundError, DatasetQualityReportError) as exc:
            raise SyntheticSnapshotError("base_snapshot_quality_report_is_invalid") from exc
        manifest_bytes = self._store.get_bytes(snapshot.manifest_key)
        if not _digest_matches(manifest_bytes, snapshot.manifest_hash):
            raise SyntheticSnapshotError("base_snapshot_manifest_integrity_failed")
        manifest = _json_object(manifest_bytes, "base_snapshot_manifest_is_invalid")
        expected = {
            "snapshot_id": snapshot.snapshot_id,
            "tenant_id": context.tenant_id,
            "contract_version": CONTRACT_VERSION,
            "row_count": snapshot.row_count,
            "split_counts": snapshot.split_counts,
            "lineage_status": "CONFIRMED",
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise SyntheticSnapshotError("base_snapshot_manifest_binding_failed")
        entries = _catalog_entries(manifest, catalog)
        rows_by_split, inherited = self._load_base_artifacts(context, entries)
        _validate_base_vlm_contract(rows_by_split, inherited)
        real_count, synthetic_count = _origin_counts(manifest, snapshot.row_count)
        return _BaseSnapshot(
            record=snapshot,
            run=run,
            rows_by_split=rows_by_split,
            inherited_artifacts=tuple(inherited),
            manifest=manifest,
            real_sample_count=real_count,
            synthetic_sample_count=synthetic_count,
        )

    def _load_base_artifacts(
        self,
        context: TenantContext,
        entries: list[dict[str, Any]],
    ) -> tuple[dict[str, list[dict[str, Any]]], list[_Artifact]]:
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        inherited: list[_Artifact] = []
        for entry in entries:
            kind = str(entry["kind"])
            if kind == "quality_report":
                continue
            content = self._store.get_bytes(str(entry["object_key"]))
            if len(content) != int(entry["size_bytes"]) or not _digest_matches(
                content, str(entry["content_hash"])
            ):
                raise SyntheticSnapshotError("base_snapshot_artifact_integrity_failed")
            if kind == "parquet":
                split = str(entry["split"])
                if split not in SPLITS or split in rows_by_split:
                    raise SyntheticSnapshotError("base_snapshot_parquet_contract_is_invalid")
                frame = parquet.read_table(BytesIO(content)).to_pandas()
                frame = frame.loc[:, list(DATASET_COLUMNS)]
                DATASET_SCHEMA.validate(frame, lazy=True)
                if (
                    len(frame) != int(entry["row_count"])
                    or (not frame.empty and set(frame["split"].astype(str)) != {split})
                    or (
                        not frame.empty
                        and set(frame["tenant_id"].astype(str)) != {context.tenant_id}
                    )
                ):
                    raise SyntheticSnapshotError("base_snapshot_parquet_contract_is_invalid")
                rows_by_split[split] = list(frame.to_dict("records"))
                continue
            metadata = {
                key: value
                for key, value in entry.items()
                if key
                not in {
                    "kind",
                    "split",
                    "object_key",
                    "content_hash",
                    "size_bytes",
                    "row_count",
                    "schema_hash",
                }
            }
            inherited.append(
                _Artifact(
                    kind=kind,
                    split=entry.get("split"),
                    object_key=str(entry["object_key"]),
                    content=content,
                    row_count=int(entry["row_count"]),
                    schema_hash=str(entry["schema_hash"]),
                    metadata=metadata,
                )
            )
        if set(rows_by_split) != set(SPLITS):
            raise SyntheticSnapshotError("base_snapshot_requires_all_splits")
        return rows_by_split, inherited

    def _record_snapshot(
        self,
        context: TenantContext,
        *,
        base: _BaseSnapshot,
        run_id: str,
        snapshot_id: str,
        input_manifest_hash: str,
        manifest_key: str,
        manifest_hash: str,
        quality_key: str,
        artifacts: list[_Artifact],
        split_counts: dict[str, int],
        now: datetime,
    ) -> None:
        with self._database.transaction(context) as session:
            session.add(
                CurationRunRecord(
                    run_id=run_id,
                    tenant_id=context.tenant_id,
                    window_start=base.run.window_start,
                    window_end=base.run.window_end,
                    engine="synthetic-augmentation",
                    contract_version=CONTRACT_VERSION,
                    code_version=self._code_version,
                    config_hash=_canonical_digest(
                        {
                            "augmentation_contract": SYNTHETIC_AUGMENTATION_CONTRACT,
                            "code_version": self._code_version,
                        }
                    ),
                    input_manifest_hash=input_manifest_hash,
                    input_count=sum(split_counts.values()),
                    exclusion_report=[],
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
                    row_count=sum(split_counts.values()),
                    split_counts=split_counts,
                    source_work_order_ids=list(base.record.source_work_order_ids),
                    lineage_status="CONFIRMED",
                    training_eligible=True,
                    base_snapshot_id=base.record.snapshot_id,
                    augmentation_contract_version=SYNTHETIC_AUGMENTATION_CONTRACT,
                    sample_origin_counts={
                        "real": base.real_sample_count,
                        "synthetic": sum(
                            1 for item in artifacts if item.kind == "synthetic_provenance"
                        ),
                    },
                    synthetic_split_counts={
                        "train": sum(
                            1 for item in artifacts if item.kind == "synthetic_provenance"
                        ),
                        "validation": 0,
                        "test": 0,
                    },
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            for artifact in artifacts:
                session.add(
                    DatasetArtifactRecord(
                        artifact_id=(
                            "artifact-"
                            + sha256(
                                f"{snapshot_id}\0{artifact.object_key}".encode()
                            ).hexdigest()[:32]
                        ),
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
                    source_work_order_ids=list(base.record.source_work_order_ids),
                    output_dataset=manifest_key,
                    failure_reason=None,
                    created_at=now,
                    updated_at=now,
                )
            )


def _validate_bundles(
    bundles: list[ReviewedSyntheticBundle],
) -> dict[str, ReviewedSyntheticBundle]:
    reviewed: dict[str, ReviewedSyntheticBundle] = {}
    output_hashes: set[str] = set()
    for bundle in bundles:
        try:
            verify_synthetic_media_bundle(
                bundle.manifest,
                image_png=bundle.image_png,
                source_image=bundle.source_image,
                mask_image=bundle.mask_image,
            )
        except SyntheticMediaError as exc:
            raise SyntheticSnapshotError(str(exc)) from exc
        review = bundle.manifest["review"]
        if (
            review["status"] != "APPROVED"
            or bundle.manifest["training_candidate_eligible"] is not True
            or bundle.manifest["evidence_eligible"] is not False
            or bundle.manifest["golden_dataset_eligible"] is not False
        ):
            raise SyntheticSnapshotError("synthetic_bundle_is_not_approved_for_training")
        provenance_id = str(bundle.manifest["manifest_sha256"])
        output_hash = str(bundle.manifest["output_image_sha256"])
        if provenance_id in reviewed or output_hash in output_hashes:
            raise SyntheticSnapshotError("synthetic_bundle_is_duplicated")
        reviewed[provenance_id] = bundle
        output_hashes.add(output_hash)
    return reviewed


def _synthetic_sample(
    context: TenantContext,
    bundle: ReviewedSyntheticBundle,
    *,
    source_row: dict[str, Any],
    provenance_id: str,
    published_prefix: str,
) -> tuple[dict[str, Any], list[_Artifact], dict[str, Any]]:
    manifest = bundle.manifest
    identity = provenance_id.removeprefix("sha256:")
    candidate_id = f"synthetic-{identity[:32]}"
    media_id = f"synthetic-media-{identity[:32]}"
    base_key = f"{published_prefix}/synthetic-provenance/{identity}"
    image_key = f"{published_prefix}/media/train/{identity}.png"
    manifest_key = f"{base_key}/reviewed-manifest.json"
    source_key = f"{base_key}/source-image.bin"
    mask_key = f"{base_key}/mask-image.bin"
    instruction = (
        f"请识别设备型号 {manifest['asset_model']} 图像中的受控缺陷类型。"
        "该图片是专家审核后的合成训练素材，不得作为现场故障证据。"
    )
    answer = json.dumps(
        {"defect_label": manifest["defect_label"], "synthetic": True},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    reviewed_at = str(manifest["review"]["reviewed_at"])
    row = {
        "tenant_id": context.tenant_id,
        "candidate_id": candidate_id,
        "candidate_version": 1,
        "work_order_id": source_row["work_order_id"],
        "incident_id": source_row["incident_id"],
        "device_id": source_row["device_id"],
        "source_event_id": f"synthetic-media:{identity}",
        "source_content_hash": manifest["output_image_sha256"],
        "redacted_content_json": json.dumps(
            {
                "asset_model": manifest["asset_model"],
                "generator_release_id": manifest["generator_release_id"],
                "prompt_hash": manifest["prompt_hash"],
                "synthetic": True,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "labels_json": json.dumps(
            {
                "defect_label": manifest["defect_label"],
                "synthetic": True,
                "_multimodal_training": {
                    "review_status": "APPROVED",
                    "modality": "image",
                    "media_id": media_id,
                    "instruction": instruction,
                    "answer": answer,
                    "synthetic_media": {
                        "schema_version": manifest["schema_version"],
                        "manifest_sha256": provenance_id,
                        "generator_release_id": manifest["generator_release_id"],
                        "reviewer_subject_id": manifest["review"][
                            "reviewer_subject_id"
                        ],
                    },
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "lineage_json": json.dumps(
            {
                "kind": "synthetic-media/v1",
                "manifest_sha256": provenance_id,
                "source_image_sha256": manifest["source_image_sha256"],
                "mask_image_sha256": manifest["mask_image_sha256"],
                "output_image_sha256": manifest["output_image_sha256"],
                "source_candidate_id": source_row["candidate_id"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "occurred_at": reviewed_at,
        "near_duplicate_group": source_row["near_duplicate_group"],
        "group_key": source_row["group_key"],
        "split": "train",
    }
    schema_hash = _digest(b"industrial-governed-media-v1")
    provenance_schema_hash = _digest(b"synthetic-media/v1")
    artifacts = [
        _Artifact(
            kind="media_image",
            split="train",
            object_key=image_key,
            content=bundle.image_png,
            row_count=1,
            schema_hash=schema_hash,
            metadata={
                "source_media_id": media_id,
                "candidate_ids": [candidate_id],
                "mime_type": "image/png",
                "synthetic": True,
                "synthetic_manifest_sha256": provenance_id,
                "evidence_eligible": False,
                "golden_dataset_eligible": False,
            },
        ),
        _Artifact(
            kind="synthetic_provenance",
            split="train",
            object_key=manifest_key,
            content=_json_bytes(manifest),
            row_count=1,
            schema_hash=provenance_schema_hash,
        ),
        _Artifact(
            kind="synthetic_source_image",
            split="train",
            object_key=source_key,
            content=bundle.source_image,
            row_count=1,
            schema_hash=provenance_schema_hash,
        ),
        _Artifact(
            kind="synthetic_mask_image",
            split="train",
            object_key=mask_key,
            content=bundle.mask_image,
            row_count=1,
            schema_hash=provenance_schema_hash,
        ),
    ]
    entry = {
        "manifest_sha256": provenance_id,
        "candidate_id": candidate_id,
        "media_id": media_id,
        "source_candidate_id": source_row["candidate_id"],
        "split": "train",
        "manifest_key": manifest_key,
        "source_image_key": source_key,
        "mask_image_key": mask_key,
        "output_image_key": image_key,
        "generator_release_id": manifest["generator_release_id"],
        "reviewer_subject_id": manifest["review"]["reviewer_subject_id"],
        "evidence_eligible": False,
        "golden_dataset_eligible": False,
    }
    return row, artifacts, entry


def _governed_source_row(
    base: _BaseSnapshot,
    bundle: ReviewedSyntheticBundle,
) -> dict[str, Any]:
    source_hash = str(bundle.manifest["source_image_sha256"])
    candidate_ids: set[str] = set()
    for artifact in base.inherited_artifacts:
        if (
            artifact.kind != "media_image"
            or artifact.split != "train"
            or artifact.metadata.get("synthetic") is True
            or artifact.content_hash != source_hash
        ):
            continue
        bound = artifact.metadata.get("candidate_ids")
        if isinstance(bound, list):
            candidate_ids.update(str(value) for value in bound)
    source_rows = [
        row
        for row in base.rows_by_split["train"]
        if str(row["candidate_id"]) in candidate_ids
    ]
    if not source_rows:
        raise SyntheticSnapshotError(
            "synthetic_source_image_is_not_governed_by_base_train_split"
        )
    if len({str(row["group_key"]) for row in source_rows}) != 1:
        raise SyntheticSnapshotError("synthetic_source_image_group_binding_is_ambiguous")
    return source_rows[0]


def _catalog_entries(
    manifest: dict[str, Any],
    catalog: list[DatasetArtifactRecord],
) -> list[dict[str, Any]]:
    raw_entries = manifest.get("artifacts")
    if not isinstance(raw_entries, list) or not all(
        isinstance(entry, dict) for entry in raw_entries
    ):
        raise SyntheticSnapshotError("base_snapshot_artifact_catalog_is_invalid")
    entries = [dict(entry) for entry in raw_entries]
    by_key = {str(entry.get("object_key")): entry for entry in entries}
    catalog_by_key = {item.object_key: item for item in catalog}
    if len(by_key) != len(entries) or set(by_key) != set(catalog_by_key):
        raise SyntheticSnapshotError("base_snapshot_artifact_catalog_is_invalid")
    for key, record in catalog_by_key.items():
        entry = by_key[key]
        expected = {
            "kind": record.kind,
            "split": record.split,
            "object_key": record.object_key,
            "content_hash": record.content_hash,
            "size_bytes": record.size_bytes,
            "row_count": record.row_count,
            "schema_hash": record.schema_hash,
        }
        if any(entry.get(name) != value for name, value in expected.items()):
            raise SyntheticSnapshotError("base_snapshot_artifact_catalog_is_invalid")
    return entries


def _relocate_inherited_artifacts(
    artifacts: tuple[_Artifact, ...],
    *,
    published_prefix: str,
) -> tuple[list[_Artifact], dict[str, str]]:
    relocated: list[_Artifact] = []
    key_mapping: dict[str, str] = {}
    for artifact in artifacts:
        name = PurePosixPath(artifact.object_key).name
        key_id = sha256(artifact.object_key.encode()).hexdigest()[:16]
        new_key = f"{published_prefix}/inherited/{key_id}-{name}"
        key_mapping[artifact.object_key] = new_key
        relocated.append(
            _Artifact(
                kind=artifact.kind,
                split=artifact.split,
                object_key=new_key,
                content=artifact.content,
                row_count=artifact.row_count,
                schema_hash=artifact.schema_hash,
                metadata=dict(artifact.metadata),
            )
        )
    return relocated, key_mapping


def _relocate_synthetic_entries(
    manifest: dict[str, Any],
    key_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    raw = manifest.get("synthetic_media", [])
    if not isinstance(raw, list) or not all(isinstance(entry, dict) for entry in raw):
        raise SyntheticSnapshotError("base_snapshot_synthetic_manifest_is_invalid")
    relocated: list[dict[str, Any]] = []
    for entry in raw:
        updated = dict(entry)
        for field_name in (
            "manifest_key",
            "source_image_key",
            "mask_image_key",
            "output_image_key",
        ):
            old_key = updated.get(field_name)
            if not isinstance(old_key, str) or old_key not in key_mapping:
                raise SyntheticSnapshotError("base_snapshot_synthetic_manifest_is_invalid")
            updated[field_name] = key_mapping[old_key]
        relocated.append(updated)
    return relocated


def _existing_provenance_ids(manifest: dict[str, Any]) -> set[str]:
    entries = manifest.get("synthetic_media", [])
    if not isinstance(entries, list):
        raise SyntheticSnapshotError("base_snapshot_synthetic_manifest_is_invalid")
    identifiers: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("manifest_sha256"), str):
            raise SyntheticSnapshotError("base_snapshot_synthetic_manifest_is_invalid")
        identifiers.add(entry["manifest_sha256"])
    if len(identifiers) != len(entries):
        raise SyntheticSnapshotError("base_snapshot_synthetic_manifest_is_invalid")
    return identifiers


def _origin_counts(manifest: dict[str, Any], row_count: int) -> tuple[int, int]:
    counts = manifest.get("sample_origin_counts")
    if counts is None:
        return row_count, 0
    if (
        not isinstance(counts, dict)
        or isinstance(counts.get("real"), bool)
        or not isinstance(counts.get("real"), int)
        or isinstance(counts.get("synthetic"), bool)
        or not isinstance(counts.get("synthetic"), int)
        or counts["real"] < 0
        or counts["synthetic"] < 0
        or counts["real"] + counts["synthetic"] != row_count
    ):
        raise SyntheticSnapshotError("base_snapshot_origin_counts_are_invalid")
    synthetic_splits = manifest.get("synthetic_split_counts")
    if synthetic_splits != {
        "train": counts["synthetic"],
        "validation": 0,
        "test": 0,
    }:
        raise SyntheticSnapshotError("base_snapshot_contains_synthetic_evaluation_samples")
    return counts["real"], counts["synthetic"]


def _validate_rows(rows: dict[str, list[dict[str, Any]]], tenant_id: str) -> None:
    candidate_ids: set[str] = set()
    groups: dict[str, set[str]] = {}
    for split in SPLITS:
        frame = arrow.Table.from_pylist(rows[split], schema=_arrow_schema()).to_pandas()
        DATASET_SCHEMA.validate(frame, lazy=True)
        if not frame.empty and (
            set(frame["tenant_id"].astype(str)) != {tenant_id}
            or set(frame["split"].astype(str)) != {split}
        ):
            raise SyntheticSnapshotError("synthetic_snapshot_row_contract_failed")
        for row in rows[split]:
            candidate_id = str(row["candidate_id"])
            if candidate_id in candidate_ids:
                raise SyntheticSnapshotError("synthetic_snapshot_candidate_is_duplicated")
            candidate_ids.add(candidate_id)
            groups.setdefault(str(row["group_key"]), set()).add(split)
    if any(len(splits) != 1 for splits in groups.values()):
        raise SyntheticSnapshotError("synthetic_snapshot_group_leakage_detected")


def _validate_base_vlm_contract(
    rows: dict[str, list[dict[str, Any]]],
    artifacts: list[_Artifact],
) -> None:
    media_bindings: dict[str, tuple[str, set[str]]] = {}
    for artifact in artifacts:
        if artifact.kind != "media_image":
            continue
        media_id = artifact.metadata.get("source_media_id")
        candidate_ids = artifact.metadata.get("candidate_ids")
        if (
            not isinstance(media_id, str)
            or not isinstance(candidate_ids, list)
            or not all(isinstance(value, str) for value in candidate_ids)
            or artifact.split not in {"train", "validation"}
        ):
            continue
        media_bindings[media_id] = (artifact.split, set(candidate_ids))
    for split in ("train", "validation"):
        if not rows[split]:
            raise SyntheticSnapshotError("base_snapshot_vlm_split_is_empty")
        for row in rows[split]:
            try:
                labels = json.loads(str(row["labels_json"]))
                contract = labels["_multimodal_training"]
                media_id = contract["media_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise SyntheticSnapshotError("base_snapshot_is_not_vlm_trainable") from exc
            binding = media_bindings.get(str(media_id))
            if (
                not isinstance(labels, dict)
                or not isinstance(contract, dict)
                or contract.get("review_status") != "APPROVED"
                or contract.get("modality") != "image"
                or not isinstance(contract.get("instruction"), str)
                or not str(contract["instruction"]).strip()
                or not isinstance(contract.get("answer"), str)
                or not str(contract["answer"]).strip()
                or binding is None
                or binding[0] != split
                or str(row["candidate_id"]) not in binding[1]
            ):
                raise SyntheticSnapshotError("base_snapshot_is_not_vlm_trainable")


def _parquet_bytes(rows: list[dict[str, Any]]) -> bytes:
    target = BytesIO()
    parquet.write_table(
        arrow.Table.from_pylist(rows, schema=_arrow_schema()),
        target,
        compression="zstd",
    )
    return target.getvalue()


def _arrow_schema() -> arrow.Schema:
    return arrow.schema(
        [
            arrow.field(name, arrow.int64() if name == "candidate_version" else arrow.string())
            for name in DATASET_COLUMNS
        ]
    )


def _json_object(content: bytes, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticSnapshotError(reason) from exc
    if not isinstance(value, dict):
        raise SyntheticSnapshotError(reason)
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _canonical_digest(value: Any) -> str:
    return _digest(_json_bytes(value))


def _digest(content: bytes) -> str:
    return f"sha256:{sha256(content).hexdigest()}"


def _digest_matches(content: bytes, expected: str) -> bool:
    return sha256(content).hexdigest() == expected.removeprefix("sha256:")


def _iso(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat()
