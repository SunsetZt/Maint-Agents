"""Airflow-facing command line entrypoint for governed dataset backfills."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from minio import Minio

from industrial_ops_agent.config import get_settings
from industrial_ops_agent.data_pipeline.lineage import OpenLineageEmitter
from industrial_ops_agent.data_pipeline.service import DatasetPipelineService
from industrial_ops_agent.data_pipeline.storage import MinioDatasetStore
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.secrets import SecretName, build_secret_provider

if TYPE_CHECKING:
    from industrial_ops_agent.data_pipeline.synthetic_snapshot import ReviewedSyntheticBundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-data-pipeline")
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="build one immutable interval snapshot")
    build.add_argument("--tenant-id", required=True)
    build.add_argument("--window-start", required=True, type=datetime.fromisoformat)
    build.add_argument("--window-end", required=True, type=datetime.fromisoformat)
    build.add_argument("--engine", choices=("local", "spark"), default=None)
    build.add_argument("--code-version", default="runtime")
    augment = subcommands.add_parser(
        "augment-synthetic",
        help="derive a training snapshot from independently reviewed synthetic bundles",
    )
    augment.add_argument("--tenant-id", required=True)
    augment.add_argument("--base-snapshot-id", required=True)
    augment.add_argument("--bundle-spec", type=Path, required=True)
    augment.add_argument("--requested-by", required=True)
    augment.add_argument("--code-version", default="runtime")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    secrets = build_secret_provider(settings)
    database = Database(secrets.get(SecretName.DATABASE_URL).reveal())
    client = Minio(
        settings.minio_endpoint,
        access_key=secrets.get(SecretName.MINIO_ACCESS_KEY).reveal(),
        secret_key=secrets.get(SecretName.MINIO_SECRET_KEY).reveal(),
        secure=settings.minio_secure,
    )
    store = MinioDatasetStore(client, settings.dataset_bucket)
    try:
        if args.command == "augment-synthetic":
            from industrial_ops_agent.data_pipeline.synthetic_snapshot import (
                SyntheticDatasetSnapshotService,
            )

            synthetic_result = SyntheticDatasetSnapshotService(
                database,
                store,
                OpenLineageEmitter(
                    url=settings.openlineage_url,
                    namespace=settings.openlineage_namespace,
                    job_name="build_synthetic_dataset_snapshot",
                ),
                code_version=args.code_version,
            ).augment_snapshot(
                TenantContext(args.tenant_id, args.requested_by),
                base_snapshot_id=args.base_snapshot_id,
                bundles=_load_reviewed_bundles(args.bundle_spec),
            )
            output = {
                "run_id": synthetic_result.run_id,
                "snapshot_id": synthetic_result.snapshot_id,
                "base_snapshot_id": synthetic_result.base_snapshot_id,
                "status": "CANDIDATE",
                "training_eligible": True,
                "row_count": synthetic_result.row_count,
                "real_sample_count": synthetic_result.real_sample_count,
                "synthetic_sample_count": synthetic_result.synthetic_sample_count,
                "manifest_key": synthetic_result.manifest_key,
                "manifest_hash": synthetic_result.manifest_hash,
            }
        else:
            build_result = DatasetPipelineService(
                database,
                store,
                OpenLineageEmitter(
                    url=settings.openlineage_url,
                    namespace=settings.openlineage_namespace,
                ),
                code_version=args.code_version,
                media_store=MinioDatasetStore(client, settings.minio_bucket),
            ).build_snapshot(
                TenantContext(args.tenant_id, "airflow-data-pipeline"),
                window_start=args.window_start,
                window_end=args.window_end,
                engine=args.engine or settings.dataset_pipeline_engine,
            )
            output = {
                "run_id": build_result.run_id,
                "snapshot_id": build_result.snapshot_id,
                "status": build_result.status,
                "training_eligible": build_result.training_eligible,
                "row_count": build_result.row_count,
                "manifest_key": build_result.manifest_key,
            }
    finally:
        database.dispose()
    print(json.dumps(output, sort_keys=True))
    # An empty fixed window is an expected no-op for scheduled curation, not an
    # infrastructure failure. Lineage-pending and other blocked outcomes keep
    # the non-zero exit code so orchestration can retry or alert.
    return 0 if output["training_eligible"] or output["status"] == "NO_DATA" else 3


def _load_reviewed_bundles(spec_path: Path) -> list[ReviewedSyntheticBundle]:
    from industrial_ops_agent.data_pipeline.synthetic_snapshot import (
        ReviewedSyntheticBundle,
        SyntheticSnapshotError,
    )

    try:
        document = json.loads(spec_path.read_text(encoding="utf-8"))
        entries = document["bundles"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SyntheticSnapshotError("synthetic_bundle_spec_is_invalid") from exc
    if not isinstance(entries, list) or not entries:
        raise SyntheticSnapshotError("synthetic_bundle_spec_is_invalid")
    base = spec_path.resolve().parent
    bundles: list[ReviewedSyntheticBundle] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise SyntheticSnapshotError("synthetic_bundle_spec_is_invalid")
        manifest_path = _bundle_path(base, entry, "manifest")
        try:
            manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise TypeError
            bundles.append(
                ReviewedSyntheticBundle(
                    manifest=manifest,
                    image_png=_bundle_path(base, entry, "image").read_bytes(),
                    source_image=_bundle_path(base, entry, "source_image").read_bytes(),
                    mask_image=_bundle_path(base, entry, "mask_image").read_bytes(),
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            raise SyntheticSnapshotError("synthetic_bundle_spec_is_invalid") from exc
    return bundles


def _bundle_path(base: Path, entry: dict[str, Any], field_name: str) -> Path:
    from industrial_ops_agent.data_pipeline.synthetic_snapshot import SyntheticSnapshotError

    value = entry.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise SyntheticSnapshotError("synthetic_bundle_spec_is_invalid")
    path = Path(value)
    return path if path.is_absolute() else base / path


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
