"""One-shot CLI for a registered governed SFT or post-training experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from tempfile import TemporaryDirectory
from uuid import uuid4

from minio import Minio
from sqlalchemy import select

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.config import get_settings
from industrial_ops_agent.data_pipeline.storage import MinioDatasetStore
from industrial_ops_agent.experiments.mlflow import MlflowRestTracker
from industrial_ops_agent.experiments.service import ExperimentRegistryService
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import TrainingExperimentRecord
from industrial_ops_agent.secrets import SecretName, build_secret_provider
from industrial_ops_agent.security_audit import build_persistent_security_auditor
from industrial_ops_agent.training.backend import HuggingFaceTrainingBackend
from industrial_ops_agent.training.config import (
    CompiledPeftConfig,
    TrainingConfigurationError,
    compile_training_config,
)
from industrial_ops_agent.training.distributed import (
    CompiledDistributedProfile,
    build_torchrun_command,
    is_distributed_child,
    process_rank,
)
from industrial_ops_agent.training.image_acceptance import load_gpu_acceptance_report
from industrial_ops_agent.training.runner import RuntimeFingerprint, TrainingRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-training-worker")
    parser.add_argument("--experiment-id", default=os.getenv("IOAP_TRAINING_EXPERIMENT_ID"))
    parser.add_argument("--tenant-id", default=os.getenv("IOAP_TRAINING_TENANT_ID"))
    parser.add_argument(
        "--subject-id",
        default=os.getenv("IOAP_TRAINING_SUBJECT_ID", "m4-training-worker"),
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
    audit_key = secrets.get(SecretName.OIDC_CLIENT_SECRET).reveal().encode()
    authorizer = Authorizer(
        build_persistent_security_auditor(database, hash_key=audit_key)
    )
    tracker = MlflowRestTracker(
        settings.mlflow_tracking_url,
        timeout_seconds=settings.mlflow_timeout_seconds,
    )
    store = MinioDatasetStore(client, settings.dataset_bucket)
    registry = ExperimentRegistryService(database, authorizer, tracker)
    runner = TrainingRunner(
        database,
        store,
        registry,
        HuggingFaceTrainingBackend(),
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
    if not args.dry_run and not is_distributed_child():
        distributed = _load_distributed_config(database, identity, args.experiment_id)
        if distributed is not None and distributed.distributed:
            try:
                return _launch_distributed_workers(
                    args=args,
                    distributed=distributed,
                    identity=identity,
                    database=database,
                    registry=registry,
                )
            finally:
                database.dispose()
    try:
        result = runner.run(
            identity,
            args.experiment_id,
            runtime=runtime,
            dry_run=args.dry_run,
            request_id=f"training-worker-{uuid4().hex}",
        )
    finally:
        database.dispose()
    if process_rank() == 0:
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


def _load_distributed_config(
    database: Database,
    identity: IdentityContext,
    experiment_id: str,
) -> CompiledDistributedProfile | None:
    with database.transaction(identity.tenant_context) as session:
        experiment = session.scalar(
            select(TrainingExperimentRecord).where(
                TrainingExperimentRecord.tenant_id == identity.tenant_id,
                TrainingExperimentRecord.experiment_id == experiment_id,
            )
        )
        if experiment is None:
            return None
        try:
            compiled = compile_training_config(experiment)
        except TrainingConfigurationError:
            return None
        return compiled.distributed if isinstance(compiled, CompiledPeftConfig) else None


def _launch_distributed_workers(
    *,
    args: argparse.Namespace,
    distributed: CompiledDistributedProfile,
    identity: IdentityContext,
    database: Database,
    registry: ExperimentRegistryService,
) -> int:
    module_arguments = [
        "--experiment-id",
        str(args.experiment_id),
        "--tenant-id",
        str(args.tenant_id),
        "--subject-id",
        str(args.subject_id),
    ]
    command = build_torchrun_command(
        distributed,
        module="industrial_ops_agent.training.cli",
        module_arguments=module_arguments,
    )
    with TemporaryDirectory(prefix="industrial-ops-distributed-training-") as directory:
        environment = os.environ.copy()
        environment["IOAP_DISTRIBUTED_TRAINING_CHILD"] = "1"
        environment["IOAP_DISTRIBUTED_OUTPUT_DIRECTORY"] = f"{directory}/model-output"
        result = subprocess.run(command, env=environment, check=False)
    if result.returncode != 0:
        _close_failed_distributed_run(
            database,
            registry,
            identity,
            str(args.experiment_id),
        )
    return int(result.returncode)


def _close_failed_distributed_run(
    database: Database,
    registry: ExperimentRegistryService,
    identity: IdentityContext,
    experiment_id: str,
) -> None:
    with database.transaction(identity.tenant_context) as session:
        experiment = session.scalar(
            select(TrainingExperimentRecord).where(
                TrainingExperimentRecord.tenant_id == identity.tenant_id,
                TrainingExperimentRecord.experiment_id == experiment_id,
            )
        )
        if experiment is None or experiment.status != "RUNNING":
            return
        version = experiment.version
    registry.fail(
        identity,
        experiment_id,
        expected_version=version,
        reason_code="distributed_training_subprocess_failed",
        request_id=f"distributed-training-parent-{uuid4().hex}:failed",
    )


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
