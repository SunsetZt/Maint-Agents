"""Independent evaluator: frozen suite -> paired inference -> immutable evidence -> verdict."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.evaluation.artifacts import (
    extract_adapter_bundle,
    store_case_evidence,
    verify_artifact_bytes,
)
from industrial_ops_agent.evaluation.backend import (
    EvaluationRuntimeConfig,
    ModelEvaluationBackend,
)
from industrial_ops_agent.evaluation.dataset import FrozenEvaluationDatasetLoader
from industrial_ops_agent.evaluation.frameworks import (
    DeterministicFrameworkBridge,
    EvaluationFrameworkBridge,
    RagasDeepEvalBridge,
)
from industrial_ops_agent.evaluation.metrics import (
    score_asr_paired_observations,
    score_paired_observations,
    score_ppo_safety_paired_observations,
    score_retrieval_paired_observations,
    score_rul_paired_observations,
    score_timeseries_paired_observations,
    score_tts_paired_observations,
    score_vlm_paired_observations,
)
from industrial_ops_agent.experiments.service import (
    EvaluationGovernanceService,
    EvaluationJobService,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    EvaluationPolicyRecord,
    EvaluationSuiteRecord,
    ModelEvaluationJobRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.predictive_maintenance.rul_training_dataset import (
    GovernedRulDatasetLoader,
)


class EvaluationExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeFingerprint:
    git_commit: str
    container_digest: str


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    job_id: str
    status: str
    dry_run: bool
    suite_manifest_hash: str
    sample_count: int
    evaluation_id: str | None
    decision: str | None
    evidence_object_key: str | None


@dataclass(frozen=True, slots=True)
class CandidateAdapter:
    artifact_id: str
    kind: str
    object_key: str
    content_hash: str
    size_bytes: int
    content: bytes


class EvaluationRunner:
    def __init__(
        self,
        database: Database,
        store: DatasetStore,
        job_service: EvaluationJobService,
        governance: EvaluationGovernanceService,
        backend: ModelEvaluationBackend,
    ) -> None:
        self._database = database
        self._store = store
        self._job_service = job_service
        self._governance = governance
        self._backend = backend
        self._dataset_loader = FrozenEvaluationDatasetLoader(database, store)
        self._rul_training_loader = GovernedRulDatasetLoader(database, store)

    def run(
        self,
        identity: IdentityContext,
        job_id: str,
        *,
        runtime: EvaluationRuntimeFingerprint,
        dry_run: bool,
        request_id: str,
    ) -> EvaluationRunResult:
        job, candidate, baseline, suite, policy = self._load(identity, job_id)
        if dry_run:
            if job.status != "PLANNED":
                raise EvaluationExecutionError("dry_run_requires_planned_evaluation_job")
        else:
            job = self._job_service.start(
                identity,
                job_id,
                expected_version=job.version,
                request_id=f"{request_id}:started",
            )
        try:
            _verify_runtime(job, runtime)
            if self._backend.target_profile != job.target_profile:
                raise EvaluationExecutionError("evaluation_backend_profile_mismatch")
            dataset = self._dataset_loader.load(identity.tenant_context, suite)
            if job.target_profile == "RUL_COMPONENT":
                self._verify_rul_baseline(identity, candidate, baseline)
            candidate_adapter = self._candidate_adapter(identity, candidate)
            baseline_adapter = (
                candidate_adapter
                if candidate.method == "TTS"
                else self._candidate_adapter(identity, baseline)
                if candidate.method == "QUANTIZATION"
                else None
            )
            if dry_run:
                return EvaluationRunResult(
                    job_id=job_id,
                    status="VALIDATED",
                    dry_run=True,
                    suite_manifest_hash=dataset.manifest_hash,
                    sample_count=len(dataset.cases),
                    evaluation_id=None,
                    decision=None,
                    evidence_object_key=None,
                )
            config = _runtime_config(job)
            with TemporaryDirectory(prefix="industrial-ops-evaluation-") as directory:
                adapter_directory = Path(directory) / "candidate-adapter"
                extract_adapter_bundle(candidate_adapter.content, adapter_directory)
                baseline_adapter_directory = None
                if baseline_adapter is not None:
                    baseline_adapter_directory = Path(directory) / "baseline-adapter"
                    extract_adapter_bundle(
                        baseline_adapter.content, baseline_adapter_directory
                    )
                baseline_observations = self._backend.evaluate(
                    experiment=baseline,
                    cases=dataset.cases,
                    config=config,
                    adapter_directory=baseline_adapter_directory,
                )
                candidate_observations = self._backend.evaluate(
                    experiment=candidate,
                    cases=dataset.cases,
                    config=config,
                    adapter_directory=adapter_directory,
                )
            if job.target_profile == "RETRIEVAL_COMPONENT":
                metrics = score_retrieval_paired_observations(
                    dataset.cases,
                    candidate_observations,
                    baseline_observations,
                    top_k=int(job.runner_config["top_k"]),
                    primary_metric=policy.primary_metric,
                )
                framework_metrics: dict[str, object] = {
                    "status": "DETERMINISTIC_RETRIEVAL_METRICS",
                    "judge": "NOT_CONFIGURED",
                }
            elif job.target_profile == "VLM_COMPONENT":
                metrics = score_vlm_paired_observations(
                    dataset.cases,
                    candidate_observations,
                    baseline_observations,
                    region_iou_min=float(policy.thresholds["region_iou_min"]),
                    hallucination_rate_max=float(policy.thresholds["hallucination_rate_max"]),
                )
                framework_metrics = {
                    "status": "DETERMINISTIC_VLM_METRICS",
                    "judge": "NOT_CONFIGURED",
                }
            elif job.target_profile == "ASR_COMPONENT":
                metrics = score_asr_paired_observations(
                    dataset.cases,
                    candidate_observations,
                    baseline_observations,
                    wer_max=float(policy.thresholds["wer_max"]),
                    cer_max=float(policy.thresholds["cer_max"]),
                    noise_wer_max=float(policy.thresholds["noise_wer_max"]),
                )
                framework_metrics = {
                    "status": "DETERMINISTIC_ASR_METRICS",
                    "judge": "NOT_CONFIGURED",
                }
            elif job.target_profile == "TTS_COMPONENT":
                metrics = score_tts_paired_observations(
                    dataset.cases,
                    candidate_observations,
                    baseline_observations,
                    safety_phrase_completeness_min=float(
                        policy.thresholds["safety_phrase_completeness_min"]
                    ),
                    terminology_recall_min=float(
                        policy.thresholds["terminology_recall_min"]
                    ),
                    intelligibility_min=float(
                        policy.thresholds["intelligibility_min"]
                    ),
                    audio_integrity_rate_min=float(
                        policy.thresholds["audio_integrity_rate_min"]
                    ),
                )
                framework_metrics = {
                    "status": "DETERMINISTIC_TTS_OBJECTIVE_METRICS",
                    "judge": "NOT_CONFIGURED",
                    "staging_mos": "REQUIRED_SEPARATELY",
                }
            elif job.target_profile == "TIMESERIES_COMPONENT":
                metrics = score_timeseries_paired_observations(
                    dataset.cases,
                    candidate_observations,
                    baseline_observations,
                    false_positive_rate_max=float(policy.thresholds["false_positive_rate_max"]),
                    miss_rate_max=float(policy.thresholds["miss_rate_max"]),
                )
                framework_metrics = {
                    "status": "DETERMINISTIC_TIMESERIES_METRICS",
                    "judge": "NOT_CONFIGURED",
                }
            elif job.target_profile == "RUL_COMPONENT":
                metrics = score_rul_paired_observations(
                    dataset.cases,
                    candidate_observations,
                    baseline_observations,
                    median_absolute_error_minutes_max=float(
                        policy.thresholds["median_absolute_error_minutes_max"]
                    ),
                    interval_coverage_min=float(
                        policy.thresholds["interval_coverage_min"]
                    ),
                    interval_coverage_max=float(
                        policy.thresholds["interval_coverage_max"]
                    ),
                    pinball_loss_max=float(policy.thresholds["pinball_loss_max"]),
                )
                framework_metrics = {
                    "status": "DETERMINISTIC_RUL_QUANTILE_METRICS",
                    "judge": "NOT_CONFIGURED",
                }
            elif job.target_profile == "PPO_RESEARCH_SAFETY":
                reward_model_id, reward_model_revision, reward_model_digest = (
                    _ppo_reward_model_binding(candidate)
                )
                metrics = score_ppo_safety_paired_observations(
                    dataset.cases,
                    candidate_observations,
                    baseline_observations,
                    reward_model_id=reward_model_id,
                    reward_model_revision=reward_model_revision,
                    reward_model_digest=reward_model_digest,
                    reward_consistency_rate_min=float(
                        policy.thresholds["reward_consistency_rate_min"]
                    ),
                    attack_success_rate_max=float(
                        policy.thresholds["attack_success_rate_max"]
                    ),
                    safe_response_rate_min=float(
                        policy.thresholds["safe_response_rate_min"]
                    ),
                )
                framework_metrics = {
                    "status": "DETERMINISTIC_PPO_SAFETY_METRICS",
                    "judge": "NOT_CONFIGURED",
                    "research_only": True,
                }
            else:
                metrics = score_paired_observations(
                    dataset.cases, candidate_observations, baseline_observations
                )
                bridge = _framework(job.runner_config["framework_mode"])
                framework_metrics = {
                    "candidate": bridge.summarize(dataset.cases, candidate_observations),
                    "baseline": bridge.summarize(dataset.cases, baseline_observations),
                }
            evidence = replace(
                metrics.evidence,
                slice_metrics={
                    **metrics.evidence.slice_metrics,
                    "frameworks": framework_metrics,
                },
            )
            report = {
                "schema_version": "industrial-ops-evaluation-evidence-v1",
                "job_id": job.job_id,
                "suite_id": suite.suite_id,
                "suite_manifest_hash": dataset.manifest_hash,
                "policy_id": policy.policy_id,
                "policy_hash": policy.policy_hash,
                "candidate_experiment_id": candidate.experiment_id,
                "candidate_method": candidate.method,
                "candidate_artifact": {
                    "artifact_id": candidate_adapter.artifact_id,
                    "kind": candidate_adapter.kind,
                    "object_key": candidate_adapter.object_key,
                    "content_hash": candidate_adapter.content_hash,
                    "size_bytes": candidate_adapter.size_bytes,
                },
                "baseline_experiment_id": baseline.experiment_id,
                "baseline_artifact": (
                    {
                        "artifact_id": baseline_adapter.artifact_id,
                        "kind": baseline_adapter.kind,
                        "object_key": baseline_adapter.object_key,
                        "content_hash": baseline_adapter.content_hash,
                        "size_bytes": baseline_adapter.size_bytes,
                    }
                    if baseline_adapter is not None
                    else None
                ),
                "target_profile": job.target_profile,
                "research_only": job.target_profile == "PPO_RESEARCH_SAFETY",
                "runner_git_commit": job.runner_git_commit,
                "container_digest": job.container_digest,
                "runner_config_hash": job.config_hash,
                "aggregate_metrics": metrics.aggregate_metrics,
                "framework_metrics": framework_metrics,
                "hard_gate_results": evidence.hard_gate_results,
                "gate_coverage": metrics.gate_coverage,
                "cases": [asdict(item) for item in metrics.case_metrics],
            }
            artifact = store_case_evidence(
                self._store,
                tenant_id=identity.tenant_id,
                job_id=job.job_id,
                report=report,
                metadata={
                    "suite_manifest_hash": dataset.manifest_hash,
                    "policy_hash": policy.policy_hash,
                    "runner_config_hash": job.config_hash,
                    "candidate_method": candidate.method,
                    "candidate_artifact_hash": candidate_adapter.content_hash,
                    "research_only": job.target_profile == "PPO_RESEARCH_SAFETY",
                },
            )
            evaluation = self._governance.evaluate(
                identity,
                candidate_experiment_id=candidate.experiment_id,
                baseline_experiment_id=baseline.experiment_id,
                suite_id=suite.suite_id,
                policy_id=policy.policy_id,
                evidence=evidence,
                idempotency_key=f"evaluation-job:{job.job_id}",
                request_id=f"{request_id}:verdict",
                artifacts=[artifact],
            )
            completed = self._job_service.complete(
                identity,
                job.job_id,
                expected_version=job.version,
                evaluation_id=evaluation.evaluation_id,
                request_id=f"{request_id}:completed",
            )
            return EvaluationRunResult(
                job_id=job.job_id,
                status=completed.status,
                dry_run=False,
                suite_manifest_hash=dataset.manifest_hash,
                sample_count=len(dataset.cases),
                evaluation_id=evaluation.evaluation_id,
                decision=evaluation.decision,
                evidence_object_key=artifact.object_key,
            )
        except Exception as exc:
            if not dry_run:
                reason = _safe_reason(exc)
                try:
                    current, *_ = self._load(identity, job_id)
                    if current.status == "RUNNING":
                        self._job_service.fail(
                            identity,
                            job_id,
                            expected_version=current.version,
                            reason_code=reason,
                            request_id=f"{request_id}:failed",
                        )
                except Exception as close_error:
                    raise EvaluationExecutionError(
                        f"{reason}:failure_state_could_not_be_closed"
                    ) from close_error
            if isinstance(exc, EvaluationExecutionError):
                raise
            raise EvaluationExecutionError(_safe_reason(exc)) from exc

    def _verify_rul_baseline(
        self,
        identity: IdentityContext,
        candidate: TrainingExperimentRecord,
        baseline: TrainingExperimentRecord,
    ) -> None:
        if baseline.method != "RUL_EMPIRICAL_BASELINE":
            raise EvaluationExecutionError("rul_empirical_baseline_is_required")
        training = self._rul_training_loader.load(identity.tenant_context, candidate)
        targets = sorted(sample.lead_time_minutes for sample in training.train)
        configured = baseline.training_config.get("empirical_quantiles_minutes")
        if not isinstance(configured, list) or len(configured) != 3:
            raise EvaluationExecutionError("rul_empirical_baseline_quantiles_are_invalid")
        expected = [_percentile(targets, quantile) for quantile in (0.1, 0.5, 0.9)]
        try:
            observed = [float(value) for value in configured]
        except (TypeError, ValueError) as exc:
            raise EvaluationExecutionError(
                "rul_empirical_baseline_quantiles_are_invalid"
            ) from exc
        if (
            len(targets) != baseline.training_config.get("source_sample_count")
            or baseline.training_config.get("source_manifest_hash") != training.manifest_hash
            or any(abs(left - right) > 1e-6 for left, right in zip(observed, expected, strict=True))
        ):
            raise EvaluationExecutionError("rul_empirical_baseline_binding_changed")

    def _load(
        self, identity: IdentityContext, job_id: str
    ) -> tuple[
        ModelEvaluationJobRecord,
        TrainingExperimentRecord,
        TrainingExperimentRecord,
        EvaluationSuiteRecord,
        EvaluationPolicyRecord,
    ]:
        with self._database.transaction(identity.tenant_context) as session:
            job = session.scalar(
                select(ModelEvaluationJobRecord).where(
                    ModelEvaluationJobRecord.tenant_id == identity.tenant_id,
                    ModelEvaluationJobRecord.job_id == job_id,
                )
            )
            if job is None:
                raise EvaluationExecutionError("evaluation_job_not_visible")
            candidate = session.get(TrainingExperimentRecord, job.candidate_experiment_id)
            baseline = session.get(TrainingExperimentRecord, job.baseline_experiment_id)
            suite = session.get(EvaluationSuiteRecord, job.suite_id)
            policy = session.get(EvaluationPolicyRecord, job.policy_id)
            if candidate is None or baseline is None or suite is None or policy is None:
                raise EvaluationExecutionError("evaluation_job_reference_not_visible")
            if any(
                item.tenant_id != identity.tenant_id
                for item in (candidate, baseline, suite, policy)
            ):
                raise EvaluationExecutionError("evaluation_job_reference_tenant_mismatch")
            return job, candidate, baseline, suite, policy

    def _candidate_adapter(
        self, identity: IdentityContext, candidate: TrainingExperimentRecord
    ) -> CandidateAdapter:
        with self._database.transaction(identity.tenant_context) as session:
            required_kind = {
                "EMBEDDING": "embedding_model_bundle",
                "RERANKER": "reranker_model_bundle",
                "VLM": "vlm_adapter_bundle",
                "ASR": "asr_adapter_bundle",
                "TTS": "tts_model_bundle",
                "TIMESERIES_TRANSFORMER": "timeseries_transformer_bundle",
                "RUL_TRANSFORMER": "rul_transformer_bundle",
                "QUANTIZATION": "quantized_model_bundle",
                "PPO": "ppo_policy_adapter_bundle",
            }.get(candidate.method, "adapter_bundle")
            artifacts = list(
                session.scalars(
                    select(TrainingArtifactRecord).where(
                        TrainingArtifactRecord.tenant_id == identity.tenant_id,
                        TrainingArtifactRecord.experiment_id == candidate.experiment_id,
                        TrainingArtifactRecord.kind == required_kind,
                    )
                )
            )
            if len(artifacts) != 1:
                raise EvaluationExecutionError(f"candidate_requires_one_{required_kind}")
            artifact = artifacts[0]
        content = self._store.get_bytes(artifact.object_key)
        verify_artifact_bytes(
            content, content_hash=artifact.content_hash, size_bytes=artifact.size_bytes
        )
        return CandidateAdapter(
            artifact_id=artifact.artifact_id,
            kind=artifact.kind,
            object_key=artifact.object_key,
            content_hash=artifact.content_hash,
            size_bytes=artifact.size_bytes,
            content=content,
        )


def _ppo_reward_model_binding(
    candidate: TrainingExperimentRecord,
) -> tuple[str, str, str]:
    if candidate.method != "PPO":
        raise EvaluationExecutionError("ppo_safety_profile_requires_ppo_candidate")
    reward_model_id = candidate.training_config.get("reward_model_id")
    reward_model_revision = candidate.training_config.get("reward_model_revision")
    reward_model_digest = candidate.training_config.get("reward_model_digest")
    if (
        not isinstance(reward_model_id, str)
        or not reward_model_id.strip()
        or len(reward_model_id) > 255
        or not isinstance(reward_model_revision, str)
        or len(reward_model_revision) != 40
        or any(character not in "0123456789abcdef" for character in reward_model_revision)
        or reward_model_digest != f"hf-revision:{reward_model_revision}"
    ):
        raise EvaluationExecutionError("ppo_reward_model_binding_is_invalid")
    return reward_model_id.strip(), reward_model_revision, str(reward_model_digest)


def _verify_runtime(job: ModelEvaluationJobRecord, runtime: EvaluationRuntimeFingerprint) -> None:
    if runtime.git_commit != job.runner_git_commit:
        raise EvaluationExecutionError("evaluation_runtime_git_commit_mismatch")
    if runtime.container_digest != job.container_digest:
        raise EvaluationExecutionError("evaluation_runtime_container_digest_mismatch")


def _runtime_config(job: ModelEvaluationJobRecord) -> EvaluationRuntimeConfig:
    config = job.runner_config
    return EvaluationRuntimeConfig(
        max_new_tokens=(int(config["max_new_tokens"]) if "max_new_tokens" in config else None),
        precision=str(config["precision"]),
        gpu_hourly_cost_usd=float(config["gpu_hourly_cost_usd"]),
        top_k=int(config["top_k"]) if "top_k" in config else None,
        max_image_pixels=(
            int(config["max_image_pixels"]) if "max_image_pixels" in config else None
        ),
        sampling_rate=(int(config["sampling_rate"]) if "sampling_rate" in config else None),
        language=str(config["language"]) if "language" in config else None,
        max_audio_seconds=(
            float(config["max_audio_seconds"]) if "max_audio_seconds" in config else None
        ),
        anomaly_threshold=(
            float(config["anomaly_threshold"]) if "anomaly_threshold" in config else None
        ),
        rule_threshold=(float(config["rule_threshold"]) if "rule_threshold" in config else None),
        cpu_hourly_cost_usd=(
            float(config["cpu_hourly_cost_usd"])
            if "cpu_hourly_cost_usd" in config
            else None
        ),
        threads=int(config["threads"]) if "threads" in config else None,
        context_size=int(config["context_size"]) if "context_size" in config else None,
        asr_verifier_model_id=(
            str(config["asr_verifier_model_id"])
            if "asr_verifier_model_id" in config
            else None
        ),
        asr_verifier_revision=(
            str(config["asr_verifier_revision"])
            if "asr_verifier_revision" in config
            else None
        ),
        asr_verifier_digest=(
            str(config["asr_verifier_digest"])
            if "asr_verifier_digest" in config
            else None
        ),
    )


def _framework(mode: str) -> EvaluationFrameworkBridge:
    if mode == "DETERMINISTIC":
        return DeterministicFrameworkBridge()
    if mode == "RAGAS_DEEPEVAL":
        return RagasDeepEvalBridge()
    raise EvaluationExecutionError("evaluation_framework_mode_is_unsupported")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise EvaluationExecutionError("rul_empirical_baseline_training_split_is_empty")
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _safe_reason(exc: Exception) -> str:
    if isinstance(exc, (ValueError, RuntimeError)):
        value = str(exc).lower().strip().replace(" ", "_")
        safe = "".join(
            character
            for character in value
            if character in "abcdefghijklmnopqrstuvwxyz0123456789_:-"
        )
        if safe:
            return safe[:255]
    return "evaluation_worker_internal_error"
