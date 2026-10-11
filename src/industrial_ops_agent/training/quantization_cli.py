"""One-shot worker CLI for a registered quantization experiment."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from minio import Minio

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.config import get_settings
from industrial_ops_agent.data_pipeline.storage import MinioDatasetStore
from industrial_ops_agent.experiments.mlflow import MlflowRestTracker
from industrial_ops_agent.experiments.service import ExperimentRegistryService
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.secrets import SecretName, build_secret_provider
from industrial_ops_agent.security_audit import build_persistent_security_auditor
from industrial_ops_agent.training.image_acceptance import load_gpu_acceptance_report
from industrial_ops_agent.training.quantization import ProductionQuantizationBackend
from industrial_ops_agent.training.quantization_runner import QuantizationRunner
from industrial_ops_agent.training.runner import RuntimeFingerprint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-quantization-worker")
    parser.add_argument("--experiment-id", default=os.getenv("IOAP_TRAINING_EXPERIMENT_ID"))
    parser.add_argument("--tenant-id", default=os.getenv("IOAP_TRAINING_TENANT_ID"))
    parser.add_argument(
        "--subject-id",
        default=os.getenv("IOAP_TRAINING_SUBJECT_ID", "quantization-worker"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.experiment_id or not args.tenant_id or not args.subject_id:
        raise SystemExit("experiment-id, tenant-id and subject-id are required")
    runtime = RuntimeFingerprint(
        git_commit=os.getenv("IOAP_TRAINING_GIT_COMMIT", ""),
        container_digest=os.getenv("IOAP_TRAINING_CONTAINER_DIGEST", ""),
        gpu_acceptance_report=load_gpu_acceptance_report(
            os.getenv(
                "IOAP_TRAINING_GPU_ACCEPTANCE_REPORT",
                "/models/huggingface/ioap-gpu-acceptance.json",
            )
        ),
    )
    settings = get_settings()
    secrets = build_secret_provider(settings)
    database = Database(secrets.get(SecretName.DATABASE_URL).reveal())
    client = Minio(
        settings.minio_endpoint,
        access_key=secrets.get(SecretName.MINIO_ACCESS_KEY).reveal(),
        secret_key=secrets.get(SecretName.MINIO_SECRET_KEY).reveal(),
        secure=settings.minio_secure,
    )
    authorizer = Authorizer(
        build_persistent_security_auditor(
            database,
            hash_key=secrets.get(SecretName.OIDC_CLIENT_SECRET).reveal().encode(),
        )
    )
    registry = ExperimentRegistryService(
        database,
        authorizer,
        MlflowRestTracker(
            settings.mlflow_tracking_url,
            timeout_seconds=settings.mlflow_timeout_seconds,
        ),
    )
    runner = QuantizationRunner(
        database,
        MinioDatasetStore(client, settings.dataset_bucket),
        registry,
        ProductionQuantizationBackend(),
    )
    now = datetime.now(UTC)
    identity = IdentityContext(
        subject_id=args.subject_id,
        oidc_subject=f"workload:{args.subject_id}",
        tenant_id=args.tenant_id,
        roles=frozenset({Role.MODEL_ENGINEER}),
        asset_ids=frozenset(),
        site_ids=frozenset(),
        issued_at=now,
        expires_at=now + timedelta(hours=24),
    )
    try:
        result = runner.run(
            identity,
            args.experiment_id,
            runtime=runtime,
            dry_run=args.dry_run,
            request_id=f"quantization-worker-{uuid4().hex}",
        )
    finally:
        database.dispose()
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
