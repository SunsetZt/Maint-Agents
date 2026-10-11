"""Airflow-facing entrypoint for governed supervised RUL snapshots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from minio import Minio

from industrial_ops_agent.config import get_settings
from industrial_ops_agent.data_pipeline.lineage import OpenLineageEmitter
from industrial_ops_agent.data_pipeline.storage import MinioDatasetStore
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.predictive_maintenance.rul_dataset_pipeline import (
    RulDatasetPipelineService,
)
from industrial_ops_agent.secrets import SecretName, build_secret_provider


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-rul-dataset")
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="build one immutable supervised RUL snapshot")
    build.add_argument("--tenant-id", required=True)
    build.add_argument("--window-start", required=True, type=datetime.fromisoformat)
    build.add_argument("--window-end", required=True, type=datetime.fromisoformat)
    build.add_argument("--engine", choices=("local", "spark"), default=None)
    build.add_argument("--code-version", default="runtime")
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
    service = RulDatasetPipelineService(
        database,
        MinioDatasetStore(client, settings.dataset_bucket),
        OpenLineageEmitter(
            url=settings.openlineage_url,
            namespace=settings.openlineage_namespace,
            job_name="build_rul_dataset_snapshot",
        ),
        code_version=args.code_version,
    )
    try:
        result = service.build_snapshot(
            TenantContext(args.tenant_id, "airflow-rul-data-pipeline"),
            window_start=args.window_start,
            window_end=args.window_end,
            engine=args.engine or settings.dataset_pipeline_engine,
        )
    finally:
        database.dispose()
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "snapshot_id": result.snapshot_id,
                "status": result.status,
                "training_eligible": result.training_eligible,
                "row_count": result.row_count,
                "split_counts": result.split_counts,
                "blocker_codes": result.blocker_codes,
                "manifest_key": result.manifest_key,
            },
            sort_keys=True,
        )
    )
    return 0 if result.training_eligible else 3


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
