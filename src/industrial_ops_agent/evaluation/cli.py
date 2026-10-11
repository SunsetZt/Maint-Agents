"""One-shot independent evaluator CLI for a registered evaluation job."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from minio import Minio
from sqlalchemy import select

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.config import get_settings
from industrial_ops_agent.data_pipeline.storage import MinioDatasetStore
from industrial_ops_agent.evaluation.backend import (
    AsrComponentEvaluationBackend,
    EdgeQuantizationEvaluationBackend,
    ModelEvaluationBackend,
    PpoResearchSafetyEvaluationBackend,
    RetrievalComponentEvaluationBackend,
    TransformersModelEvaluationBackend,
    TtsComponentEvaluationBackend,
    VlmComponentEvaluationBackend,
)
from industrial_ops_agent.evaluation.runner import (
    EvaluationRunner,
    EvaluationRuntimeFingerprint,
)
from industrial_ops_agent.evaluation.runtime_backend import (
    AgentRuntimeEvaluationBackend,
    DatabaseCitationVerifier,
)
from industrial_ops_agent.experiments.service import (
    EvaluationGovernanceService,
    EvaluationJobService,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ModelEvaluationJobRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.predictive_maintenance.evaluation_backend import (
    RulComponentEvaluationBackend,
    TimeseriesComponentEvaluationBackend,
)
from industrial_ops_agent.secrets import SecretName, build_secret_provider
from industrial_ops_agent.security_audit import build_persistent_security_auditor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-evaluation-worker")
    parser.add_argument("--job-id", default=os.getenv("IOAP_EVALUATION_JOB_ID"))
    parser.add_argument("--tenant-id", default=os.getenv("IOAP_EVALUATION_TENANT_ID"))
    parser.add_argument(
        "--subject-id",
        default=os.getenv("IOAP_EVALUATION_SUBJECT_ID", "m4-model-evaluator"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _component_backend(
    *,
    target_profile: str,
    candidate_method: str,
    candidate_config: dict[str, object],
    model_backend: ModelEvaluationBackend,
    reward_scorer: object | None = None,
    runner_config: dict[str, object] | None = None,
) -> ModelEvaluationBackend:
    if target_profile == "PPO_RESEARCH_SAFETY":
        if candidate_method != "PPO":
            raise ValueError("ppo_safety_profile_requires_ppo_candidate")
        return PpoResearchSafetyEvaluationBackend(
            candidate_config,
            policy_backend=model_backend,
            reward_scorer=reward_scorer,  # type: ignore[arg-type]
        )
    if target_profile == "RETRIEVAL_COMPONENT":
        return RetrievalComponentEvaluationBackend(candidate_method)
    if target_profile == "VLM_COMPONENT":
        return VlmComponentEvaluationBackend()
    if target_profile == "ASR_COMPONENT":
        return AsrComponentEvaluationBackend()
    if target_profile == "TTS_COMPONENT":
        if candidate_method != "TTS" or runner_config is None:
            raise ValueError("tts_profile_requires_tts_candidate_and_runner_config")
        return TtsComponentEvaluationBackend(runner_config)
    if target_profile == "TIMESERIES_COMPONENT":
        return TimeseriesComponentEvaluationBackend()
    if target_profile == "RUL_COMPONENT":
        return RulComponentEvaluationBackend()
    if target_profile == "EDGE_MODEL_COMPONENT":
        return EdgeQuantizationEvaluationBackend()
    return model_backend


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.job_id or not args.tenant_id or not args.subject_id:
        raise SystemExit("job-id, tenant-id and subject-id are required")
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
    authorizer = Authorizer(
        build_persistent_security_auditor(
            database,
            hash_key=secrets.get(SecretName.OIDC_CLIENT_SECRET).reveal().encode(),
        )
    )
    now = datetime.now(UTC)
    identity = IdentityContext(
        subject_id=args.subject_id,
        oidc_subject=f"workload:{args.subject_id}",
        tenant_id=args.tenant_id,
        roles=frozenset({Role.MODEL_EVALUATOR}),
        asset_ids=frozenset(),
        site_ids=frozenset(),
        issued_at=now,
        expires_at=now + timedelta(hours=24),
    )
    with database.transaction(identity.tenant_context) as session:
        job_profile = session.execute(
            select(
                ModelEvaluationJobRecord.target_profile,
                TrainingExperimentRecord.method,
                TrainingExperimentRecord.training_config,
                ModelEvaluationJobRecord.runner_config,
            )
            .join(
                TrainingExperimentRecord,
                TrainingExperimentRecord.experiment_id
                == ModelEvaluationJobRecord.candidate_experiment_id,
            )
            .where(
                ModelEvaluationJobRecord.tenant_id == identity.tenant_id,
                ModelEvaluationJobRecord.job_id == args.job_id,
            )
        ).one_or_none()
    if job_profile is None:
        raise SystemExit("evaluation job is not visible")
    target_profile, candidate_method, candidate_config, runner_config = job_profile
    model_backend: ModelEvaluationBackend = TransformersModelEvaluationBackend()
    if candidate_config.get("target_runtime") == "LLAMA_CPP":
        model_backend = EdgeQuantizationEvaluationBackend()
    backend: ModelEvaluationBackend
    if target_profile == "AGENT_RUNTIME":
        backend = AgentRuntimeEvaluationBackend(
            model_backend,
            DatabaseCitationVerifier(
                database,
                authorizer,
                tenant_id=identity.tenant_id,
                subject_id=identity.subject_id,
            ),
        )
    else:
        backend = _component_backend(
            target_profile=target_profile,
            candidate_method=candidate_method,
            candidate_config=candidate_config,
            model_backend=model_backend,
            runner_config=runner_config,
        )
    runner = EvaluationRunner(
        database,
        store,
        EvaluationJobService(database, authorizer),
        EvaluationGovernanceService(database, authorizer),
        backend,
    )
    try:
        result = runner.run(
            identity,
            args.job_id,
            runtime=EvaluationRuntimeFingerprint(
                git_commit=os.getenv("IOAP_EVALUATION_GIT_COMMIT", ""),
                container_digest=os.getenv("IOAP_EVALUATION_CONTAINER_DIGEST", ""),
            ),
            dry_run=args.dry_run,
            request_id=f"evaluation-worker-{uuid4().hex}",
        )
    finally:
        database.dispose()
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
