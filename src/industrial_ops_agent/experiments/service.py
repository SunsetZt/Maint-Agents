"""Governed training registry and deterministic, baseline-paired evaluation gates."""

from __future__ import annotations

import json
import math
import random
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal, cast
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.data_pipeline.contracts import SIMULATED_MULTIMODAL_CONTRACT
from industrial_ops_agent.data_pipeline.training_gate import dataset_training_blockers
from industrial_ops_agent.experiments.comparison import (
    MODEL_METHOD_COMPARISON,
    QUANTIZATION_ABLATION,
    SYNTHETIC_DATASET_ABLATION,
    QuantizationAblationContractError,
    SyntheticAblationContractError,
    quantization_ablation_context,
    snapshots_form_synthetic_ablation_pair,
    synthetic_ablation_context,
)
from industrial_ops_agent.experiments.mlflow import ExperimentTracker, TrackingUnavailable
from industrial_ops_agent.model_methods import (
    ASR_EVALUATION_CANDIDATE_METHODS,
    EVALUATION_CANDIDATE_METHODS,
    PPO_RESEARCH_EVALUATION_CANDIDATE_METHODS,
    RETRIEVAL_EVALUATION_CANDIDATE_METHODS,
    RUL_BASELINE_METHODS,
    RUL_EVALUATION_CANDIDATE_METHODS,
    TIMESERIES_BASELINE_METHODS,
    TIMESERIES_EVALUATION_CANDIDATE_METHODS,
    TTS_EVALUATION_CANDIDATE_METHODS,
    VLM_EVALUATION_CANDIDATE_METHODS,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    DataLineageRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
    EvaluationArtifactRecord,
    EvaluationPolicyRecord,
    EvaluationSuiteRecord,
    ModelEvaluationJobRecord,
    ModelEvaluationRunRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.training.reward_contracts import (
    RewardContractError,
    require_current_reward_contract,
)

TrainingMethod = Literal[
    "BASELINE",
    "LORA",
    "QLORA",
    "DPO",
    "GRPO",
    "PPO",
    "EMBEDDING",
    "RERANKER",
    "VLM",
    "ASR",
    "TTS",
    "QUANTIZATION",
    "TIMESERIES_TRANSFORMER",
    "TIMESERIES_RULE_BASELINE",
    "RUL_TRANSFORMER",
    "RUL_EMPIRICAL_BASELINE",
]
SuiteTier = Literal["SMOKE", "GOLD", "SIMULATION_REFERENCE", "PROJECT_AUTHORIZED"]
EvaluationTargetProfile = Literal[
    "MODEL_COMPONENT",
    "EDGE_MODEL_COMPONENT",
    "AGENT_RUNTIME",
    "RETRIEVAL_COMPONENT",
    "VLM_COMPONENT",
    "ASR_COMPONENT",
    "TTS_COMPONENT",
    "TIMESERIES_COMPONENT",
    "RUL_COMPONENT",
    "PPO_RESEARCH_SAFETY",
]

ALLOWED_METHODS = frozenset(
    {
        "BASELINE",
        "LORA",
        "QLORA",
        "DPO",
        "GRPO",
        "PPO",
        "EMBEDDING",
        "RERANKER",
        "VLM",
        "ASR",
        "TTS",
        "QUANTIZATION",
        "TIMESERIES_TRANSFORMER",
        "TIMESERIES_RULE_BASELINE",
        "RUL_TRANSFORMER",
        "RUL_EMPIRICAL_BASELINE",
    }
)
ALLOWED_EVALUATION_TARGET_PROFILES = frozenset(
    {
        "MODEL_COMPONENT",
        "EDGE_MODEL_COMPONENT",
        "AGENT_RUNTIME",
        "RETRIEVAL_COMPONENT",
        "VLM_COMPONENT",
        "ASR_COMPONENT",
        "TTS_COMPONENT",
        "TIMESERIES_COMPONENT",
        "RUL_COMPONENT",
        "PPO_RESEARCH_SAFETY",
    }
)
MANDATORY_HARD_GATES = frozenset(
    {
        "cross_tenant_isolation",
        "unauthorized_tool_execution",
        "t3_control_execution",
        "tool_schema_success",
        "high_risk_approval",
        "valid_citations",
        "data_governance",
        "replay_reproducibility",
        "no_blocking_regressions",
    }
)
RETRIEVAL_HARD_GATES = frozenset(
    {
        "cross_tenant_isolation",
        "data_governance",
        "retrieval_index_compatibility",
        "no_blocking_regressions",
    }
)
RETRIEVAL_PRIMARY_METRICS = frozenset({"recall_at_k", "mrr", "ndcg_at_k"})
VLM_HARD_GATES = frozenset(
    {
        "data_governance",
        "vlm_media_integrity",
        "vlm_region_grounding",
        "vlm_hallucination",
        "no_blocking_regressions",
    }
)
VLM_PRIMARY_METRICS = frozenset({"vlm_diagnostic_accuracy"})
ASR_HARD_GATES = frozenset(
    {
        "data_governance",
        "asr_media_integrity",
        "asr_transcript_quality",
        "asr_noise_robustness",
        "no_blocking_regressions",
    }
)
ASR_PRIMARY_METRICS = frozenset({"asr_word_accuracy"})
TTS_HARD_GATES = frozenset(
    {
        "data_governance",
        "tts_voice_usage_authorization",
        "tts_audio_integrity",
        "tts_safety_warning_completeness",
        "tts_terminology_accuracy",
        "tts_intelligibility",
        "no_blocking_regressions",
    }
)
TTS_PRIMARY_METRICS = frozenset({"tts_intelligibility"})
TIMESERIES_HARD_GATES = frozenset(
    {
        "data_governance",
        "telemetry_artifact_integrity",
        "label_coverage",
        "false_positive_rate",
        "miss_rate",
        "no_blocking_regressions",
    }
)
TIMESERIES_PRIMARY_METRICS = frozenset({"timeseries_accuracy"})
RUL_HARD_GATES = frozenset(
    {
        "data_governance",
        "rul_artifact_integrity",
        "rul_target_coverage",
        "rul_interval_ordering",
        "rul_median_mae",
        "rul_pinball_loss",
        "rul_interval_coverage",
        "no_blocking_regressions",
    }
)
RUL_PRIMARY_METRICS = frozenset({"rul_relative_accuracy"})
PPO_SAFETY_HARD_GATES = frozenset(
    {
        "data_governance",
        "reward_model_binding",
        "reward_preference_consistency",
        "reward_hacking_resistance",
        "no_blocking_regressions",
    }
)
PPO_SAFETY_PRIMARY_METRICS = frozenset({"ppo_safe_response_rate"})
FAIR_COMPARISON_CONFIG_KEYS = (
    "base_model_revision",
    "max_steps",
    "effective_batch_size",
    "evaluation_interval",
)


class ExperimentNotVisible(Exception):
    pass


class ExperimentConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


class ExperimentGateDenied(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    comparison_group_id: str
    method: TrainingMethod
    task_type: str
    dataset_snapshot_id: str
    base_model_id: str
    base_model_digest: str
    tokenizer_digest: str
    chat_template_digest: str
    git_commit: str
    container_digest: str
    training_config: dict[str, Any]
    distributed_profile: dict[str, Any]
    random_seeds: list[int]
    hardware_topology: dict[str, Any]
    mlflow_experiment_name: str
    license_status: str


@dataclass(frozen=True, slots=True)
class ArtifactInput:
    kind: str
    object_key: str
    content_hash: str
    size_bytes: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationEvidence:
    candidate_outcomes: list[float]
    baseline_outcomes: list[float]
    candidate_p95_latency_ms: float
    baseline_p95_latency_ms: float
    candidate_cost_per_case: float
    baseline_cost_per_case: float
    hard_gate_results: dict[str, bool]
    slice_metrics: dict[str, Any]
    high_risk_defect_resolved: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationJobPlan:
    candidate_experiment_id: str
    baseline_experiment_id: str
    suite_id: str
    policy_id: str
    target_profile: EvaluationTargetProfile
    runner_git_commit: str
    container_digest: str
    runner_config: dict[str, Any]


class ExperimentRegistryService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        tracker: ExperimentTracker,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._tracker = tracker

    def create(
        self,
        identity: IdentityContext,
        plan: ExperimentPlan,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> TrainingExperimentRecord:
        self._require(
            identity,
            Action.CREATE_TRAINING_EXPERIMENT,
            "training-experiments",
            request_id,
        )
        _validate_plan(plan)
        plan_digest = _plan_digest(plan)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(TrainingExperimentRecord).where(
                    TrainingExperimentRecord.tenant_id == identity.tenant_id,
                    TrainingExperimentRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if _record_plan_digest(existing) != plan_digest:
                    raise ExperimentConflict("idempotency_key_reused", existing.version)
                return existing
            snapshot = _snapshot_or_denied(session, identity.tenant_id, plan.dataset_snapshot_id)
            if (
                plan.method in {"TIMESERIES_TRANSFORMER", "TIMESERIES_RULE_BASELINE"}
                and snapshot.contract_version != "industrial-telemetry-sequence-v1"
            ):
                raise ExperimentGateDenied("timeseries_method_requires_telemetry_snapshot")
            if (
                plan.method in {"RUL_TRANSFORMER", "RUL_EMPIRICAL_BASELINE"}
                and snapshot.contract_version != "industrial-rul-sequence-v1"
            ):
                raise ExperimentGateDenied("rul_method_requires_rul_snapshot")
            if (
                plan.method == "RUL_EMPIRICAL_BASELINE"
                and plan.training_config.get("source_manifest_hash") != snapshot.manifest_hash
            ):
                raise ExperimentGateDenied("rul_baseline_manifest_binding_changed")
            evaluation_use = session.scalar(
                select(EvaluationSuiteRecord.suite_id).where(
                    EvaluationSuiteRecord.tenant_id == identity.tenant_id,
                    EvaluationSuiteRecord.source_snapshot_id == plan.dataset_snapshot_id,
                    EvaluationSuiteRecord.status == "FROZEN",
                )
            )
            if evaluation_use is not None:
                raise ExperimentGateDenied("evaluation_snapshot_cannot_be_used_for_training")
            if plan.method == "QUANTIZATION":
                _validate_quantization_source(session, identity.tenant_id, plan)
            self._enforce_fair_group(session, identity.tenant_id, plan)
            experiment = TrainingExperimentRecord(
                experiment_id=f"experiment-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                idempotency_key=idempotency_key,
                comparison_group_id=plan.comparison_group_id,
                method=plan.method,
                task_type=plan.task_type,
                status="PLANNED",
                dataset_snapshot_id=snapshot.snapshot_id,
                dataset_manifest_hash=str(snapshot.manifest_hash),
                base_model_id=plan.base_model_id,
                base_model_digest=plan.base_model_digest,
                tokenizer_digest=plan.tokenizer_digest,
                chat_template_digest=plan.chat_template_digest,
                git_commit=plan.git_commit,
                container_digest=plan.container_digest,
                training_config=plan.training_config,
                config_hash=_digest_json(plan.training_config),
                distributed_profile=plan.distributed_profile,
                random_seeds=plan.random_seeds,
                hardware_topology=plan.hardware_topology,
                mlflow_experiment_name=plan.mlflow_experiment_name,
                mlflow_run_id=None,
                metrics={},
                cost_summary={},
                license_status=plan.license_status,
                created_by_subject_id=identity.subject_id,
                completed_at=None,
                failure_reason=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(experiment)
            session.flush()
            return experiment

    def start(
        self,
        identity: IdentityContext,
        experiment_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> TrainingExperimentRecord:
        self._require(identity, Action.START_TRAINING_EXPERIMENT, experiment_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            experiment = _experiment_or_hidden(
                session, identity.tenant_id, experiment_id, lock=True
            )
            if experiment.version != expected_version:
                raise ExperimentConflict("version_mismatch", experiment.version)
            if experiment.status not in {"PLANNED", "TRACKING_FAILED"}:
                raise ExperimentConflict("experiment_not_startable", experiment.version)
            experiment.status = "TRACKING_PENDING"
            experiment.failure_reason = None
            experiment.version += 1
            pending_version = experiment.version
            tracker_payload = _tracking_payload(experiment)
        try:
            tracking = self._tracker.start_run(**tracker_payload)
        except TrackingUnavailable:
            self._mark_tracking_failure(identity, experiment_id, pending_version)
            raise
        with self._database.transaction(identity.tenant_context) as session:
            experiment = _experiment_or_hidden(
                session, identity.tenant_id, experiment_id, lock=True
            )
            if experiment.status != "TRACKING_PENDING" or experiment.version != pending_version:
                raise ExperimentConflict("tracking_state_changed", experiment.version)
            experiment.mlflow_run_id = tracking.run_id
            experiment.status = "RUNNING"
            experiment.version += 1
            session.flush()
            return experiment

    def complete(
        self,
        identity: IdentityContext,
        experiment_id: str,
        *,
        expected_version: int,
        metrics: dict[str, float],
        cost_summary: dict[str, Any],
        artifacts: list[ArtifactInput],
        request_id: str,
    ) -> TrainingExperimentRecord:
        self._require(identity, Action.COMPLETE_TRAINING_EXPERIMENT, experiment_id, request_id)
        _validate_completion(metrics, cost_summary, artifacts)
        with self._database.transaction(identity.tenant_context) as session:
            experiment = _experiment_or_hidden(
                session, identity.tenant_id, experiment_id, lock=True
            )
            if experiment.version != expected_version:
                raise ExperimentConflict("version_mismatch", experiment.version)
            if experiment.status not in {"RUNNING", "COMPLETION_FAILED"}:
                raise ExperimentConflict("experiment_not_completable", experiment.version)
            if experiment.mlflow_run_id is None:
                raise ExperimentGateDenied("mlflow_run_missing")
            experiment.status = "COMPLETION_PENDING"
            experiment.failure_reason = None
            experiment.version += 1
            pending_version = experiment.version
            mlflow_run_id = experiment.mlflow_run_id
        try:
            self._tracker.complete_run(mlflow_run_id, metrics=metrics)
        except TrackingUnavailable:
            self._mark_completion_failure(identity, experiment_id, pending_version)
            raise
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            experiment = _experiment_or_hidden(
                session, identity.tenant_id, experiment_id, lock=True
            )
            if experiment.status != "COMPLETION_PENDING" or experiment.version != pending_version:
                raise ExperimentConflict("completion_state_changed", experiment.version)
            for artifact in artifacts:
                session.add(
                    TrainingArtifactRecord(
                        artifact_id=f"artifact-{uuid4().hex}",
                        tenant_id=identity.tenant_id,
                        experiment_id=experiment_id,
                        kind=artifact.kind,
                        object_key=artifact.object_key,
                        content_hash=artifact.content_hash,
                        size_bytes=artifact.size_bytes,
                        metadata_json=artifact.metadata,
                        created_at=now,
                        updated_at=now,
                    )
                )
            experiment.metrics = metrics
            experiment.cost_summary = cost_summary
            experiment.status = "COMPLETED"
            experiment.completed_at = now
            experiment.version += 1
            session.flush()
            return experiment

    def fail(
        self,
        identity: IdentityContext,
        experiment_id: str,
        *,
        expected_version: int,
        reason_code: str,
        request_id: str,
    ) -> TrainingExperimentRecord:
        """Close a failed worker run locally, then best-effort close its MLflow run."""

        self._require(identity, Action.COMPLETE_TRAINING_EXPERIMENT, experiment_id, request_id)
        if (
            not reason_code
            or len(reason_code) > 255
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_:-"
                for character in reason_code
            )
        ):
            raise ValueError("failure reason code is invalid")
        with self._database.transaction(identity.tenant_context) as session:
            experiment = _experiment_or_hidden(
                session, identity.tenant_id, experiment_id, lock=True
            )
            if experiment.version != expected_version:
                raise ExperimentConflict("version_mismatch", experiment.version)
            if experiment.status not in {"RUNNING", "COMPLETION_FAILED"}:
                raise ExperimentConflict("experiment_not_failable", experiment.version)
            experiment.status = "FAILED"
            experiment.failure_reason = reason_code
            experiment.completed_at = datetime.now(UTC)
            experiment.version += 1
            mlflow_run_id = experiment.mlflow_run_id
            session.flush()
        if mlflow_run_id is not None:
            # The durable registry remains authoritative. Operators can replay
            # MLflow reconciliation without reopening a failed training run.
            with suppress(TrackingUnavailable):
                self._tracker.fail_run(mlflow_run_id, reason_code=reason_code)
        return experiment

    def _enforce_fair_group(self, session: Session, tenant_id: str, plan: ExperimentPlan) -> None:
        peers = list(
            session.scalars(
                select(TrainingExperimentRecord).where(
                    TrainingExperimentRecord.tenant_id == tenant_id,
                    TrainingExperimentRecord.comparison_group_id == plan.comparison_group_id,
                )
            )
        )
        for peer in peers:
            same_dataset = peer.dataset_snapshot_id == plan.dataset_snapshot_id
            synthetic_ablation_pair = False
            if not same_dataset and peer.method == plan.method == "VLM":
                try:
                    synthetic_ablation_pair = snapshots_form_synthetic_ablation_pair(
                        session,
                        tenant_id,
                        peer.dataset_snapshot_id,
                        plan.dataset_snapshot_id,
                    )
                except SyntheticAblationContractError as exc:
                    raise ExperimentGateDenied(
                        "synthetic_ablation_dataset_contract_invalid"
                    ) from exc
            same_fixed_inputs = (
                peer.task_type == plan.task_type
                and (same_dataset or synthetic_ablation_pair)
                and peer.base_model_digest == plan.base_model_digest
                and peer.tokenizer_digest == plan.tokenizer_digest
                and peer.chat_template_digest == plan.chat_template_digest
                and peer.random_seeds == plan.random_seeds
            )
            same_budget = all(
                peer.training_config.get(key) == plan.training_config.get(key)
                for key in FAIR_COMPARISON_CONFIG_KEYS
            )
            if synthetic_ablation_pair:
                same_budget = (
                    same_budget
                    and peer.training_config == plan.training_config
                    and peer.distributed_profile == plan.distributed_profile
                    and peer.hardware_topology == plan.hardware_topology
                )
            if not same_fixed_inputs or not same_budget:
                raise ExperimentGateDenied("comparison_group_inputs_or_budget_differ")

    def _mark_tracking_failure(
        self, identity: IdentityContext, experiment_id: str, pending_version: int
    ) -> None:
        with self._database.transaction(identity.tenant_context) as session:
            experiment = _experiment_or_hidden(
                session, identity.tenant_id, experiment_id, lock=True
            )
            if experiment.status == "TRACKING_PENDING" and experiment.version == pending_version:
                experiment.status = "TRACKING_FAILED"
                experiment.failure_reason = "mlflow_start_failed"
                experiment.version += 1

    def _mark_completion_failure(
        self, identity: IdentityContext, experiment_id: str, pending_version: int
    ) -> None:
        with self._database.transaction(identity.tenant_context) as session:
            experiment = _experiment_or_hidden(
                session, identity.tenant_id, experiment_id, lock=True
            )
            if experiment.status == "COMPLETION_PENDING" and experiment.version == pending_version:
                experiment.status = "COMPLETION_FAILED"
                experiment.failure_reason = "mlflow_completion_failed"
                experiment.version += 1

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )


class EvaluationGovernanceService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def register_suite(
        self,
        identity: IdentityContext,
        *,
        name: str,
        version: str,
        tier: SuiteTier,
        source_snapshot_id: str,
        manifest_hash: str,
        sample_count: int,
        slice_counts: dict[str, int],
        request_id: str,
    ) -> EvaluationSuiteRecord:
        self._require(identity, Action.MANAGE_EVALUATION_SUITE, source_snapshot_id, request_id)
        _validate_suite(name, version, tier, manifest_hash, sample_count, slice_counts)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(EvaluationSuiteRecord).where(
                    EvaluationSuiteRecord.tenant_id == identity.tenant_id,
                    EvaluationSuiteRecord.name == name,
                    EvaluationSuiteRecord.version == version,
                )
            )
            suite_digest = _digest_json(
                {
                    "name": name,
                    "version": version,
                    "tier": tier,
                    "source_snapshot_id": source_snapshot_id,
                    "manifest_hash": manifest_hash,
                    "sample_count": sample_count,
                    "slice_counts": slice_counts,
                }
            )
            if existing is not None:
                if _suite_digest(existing) != suite_digest:
                    raise ExperimentConflict("evaluation_suite_version_is_immutable")
                return existing
            snapshot = _snapshot_or_denied(session, identity.tenant_id, source_snapshot_id)
            is_simulation_reference = tier == "SIMULATION_REFERENCE"
            is_simulated_snapshot = snapshot.contract_version == SIMULATED_MULTIMODAL_CONTRACT
            if is_simulation_reference != is_simulated_snapshot:
                raise ExperimentGateDenied(
                    "simulation_reference_suite_requires_simulated_multimodal_snapshot"
                )
            synthetic_artifact = session.scalar(
                select(DatasetArtifactRecord.artifact_id).where(
                    DatasetArtifactRecord.tenant_id == identity.tenant_id,
                    DatasetArtifactRecord.snapshot_id == source_snapshot_id,
                    DatasetArtifactRecord.kind == "synthetic_provenance",
                )
            )
            if synthetic_artifact is not None and not is_simulation_reference:
                raise ExperimentGateDenied(
                    "synthetic_training_snapshot_cannot_be_frozen_for_evaluation"
                )
            if sample_count != snapshot.row_count or manifest_hash != snapshot.manifest_hash:
                raise ExperimentGateDenied("evaluation_suite_must_match_governed_snapshot")
            training_use = session.scalar(
                select(TrainingExperimentRecord.experiment_id).where(
                    TrainingExperimentRecord.tenant_id == identity.tenant_id,
                    TrainingExperimentRecord.dataset_snapshot_id == source_snapshot_id,
                )
            )
            if training_use is not None:
                raise ExperimentGateDenied("training_snapshot_cannot_be_frozen_for_evaluation")
            suite = EvaluationSuiteRecord(
                suite_id=f"eval-suite-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                name=name,
                version=version,
                source_snapshot_id=source_snapshot_id,
                tier=tier,
                purpose="EVALUATION_ONLY",
                status="FROZEN",
                manifest_hash=manifest_hash,
                sample_count=sample_count,
                slice_counts=slice_counts,
                created_by_subject_id=identity.subject_id,
                created_at=now,
                updated_at=now,
            )
            session.add(suite)
            session.flush()
            return suite

    def register_policy(
        self,
        identity: IdentityContext,
        *,
        name: str,
        version: str,
        primary_metric: str,
        hard_gates: dict[str, bool],
        thresholds: dict[str, float],
        request_id: str,
    ) -> EvaluationPolicyRecord:
        self._require(identity, Action.MANAGE_EVALUATION_POLICY, "evaluation-policies", request_id)
        _validate_policy(name, version, primary_metric, hard_gates, thresholds)
        policy_document = {
            "name": name,
            "version": version,
            "primary_metric": primary_metric,
            "hard_gates": hard_gates,
            "thresholds": thresholds,
        }
        policy_hash = _digest_json(policy_document)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(EvaluationPolicyRecord).where(
                    EvaluationPolicyRecord.tenant_id == identity.tenant_id,
                    EvaluationPolicyRecord.name == name,
                    EvaluationPolicyRecord.version == version,
                )
            )
            if existing is not None:
                if existing.policy_hash != policy_hash:
                    raise ExperimentConflict("evaluation_policy_version_is_immutable")
                return existing
            policy = EvaluationPolicyRecord(
                policy_id=f"eval-policy-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                status="ACTIVE",
                policy_hash=policy_hash,
                created_by_subject_id=identity.subject_id,
                created_at=now,
                updated_at=now,
                **policy_document,
            )
            session.add(policy)
            session.flush()
            return policy

    def evaluate(
        self,
        identity: IdentityContext,
        *,
        candidate_experiment_id: str,
        baseline_experiment_id: str,
        suite_id: str,
        policy_id: str,
        evidence: EvaluationEvidence,
        idempotency_key: str,
        request_id: str,
        artifacts: list[ArtifactInput] | None = None,
    ) -> ModelEvaluationRunRecord:
        self._require(identity, Action.RUN_MODEL_EVALUATION, candidate_experiment_id, request_id)
        now = datetime.now(UTC)
        evidence_artifacts = artifacts or []
        _validate_evaluation_artifacts(evidence_artifacts)
        with self._database.transaction(identity.tenant_context) as session:
            candidate = _experiment_or_hidden(
                session, identity.tenant_id, candidate_experiment_id, lock=True
            )
            baseline = _experiment_or_hidden(session, identity.tenant_id, baseline_experiment_id)
            suite = _suite_or_hidden(session, identity.tenant_id, suite_id)
            policy = _policy_or_hidden(session, identity.tenant_id, policy_id)
            existing = session.scalar(
                select(ModelEvaluationRunRecord).where(
                    ModelEvaluationRunRecord.tenant_id == identity.tenant_id,
                    ModelEvaluationRunRecord.idempotency_key == idempotency_key,
                )
            )
            _validate_evaluation_inputs(
                session,
                identity.tenant_id,
                candidate,
                baseline,
                suite,
                policy,
                evidence,
                allow_closed_no_gain=existing is not None,
            )
            comparison_kind, comparison_context = _comparison_contract(
                session,
                identity.tenant_id,
                candidate,
                baseline,
                policy,
            )
            report = _evaluate(policy, suite, evidence)
            comparison_report = (
                {
                    "comparison_kind": comparison_kind,
                    "comparison_context": comparison_context,
                }
                if comparison_kind != MODEL_METHOD_COMPARISON
                else {}
            )
            report_hash = _digest_json(
                {
                    "candidate_experiment_id": candidate_experiment_id,
                    "baseline_experiment_id": baseline_experiment_id,
                    "suite_id": suite_id,
                    "policy_id": policy_id,
                    **comparison_report,
                    "evidence_digest": _evaluation_evidence_digest(evidence),
                    "artifacts": [
                        {
                            "kind": artifact.kind,
                            "object_key": artifact.object_key,
                            "content_hash": artifact.content_hash,
                            "size_bytes": artifact.size_bytes,
                            "metadata": artifact.metadata,
                        }
                        for artifact in evidence_artifacts
                    ],
                    **report,
                }
            )
            if existing is not None:
                if existing.report_hash != report_hash:
                    raise ExperimentConflict("evaluation_idempotency_key_reused")
                return existing
            evaluation = ModelEvaluationRunRecord(
                evaluation_id=f"evaluation-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                idempotency_key=idempotency_key,
                candidate_experiment_id=candidate_experiment_id,
                baseline_experiment_id=baseline_experiment_id,
                suite_id=suite_id,
                policy_id=policy_id,
                status="COMPLETED",
                decision=str(report["decision"]),
                primary_metric=policy.primary_metric,
                candidate_score=float(report["candidate_score"]),
                baseline_score=float(report["baseline_score"]),
                quality_delta=float(report["quality_delta"]),
                ci_low=float(report["ci_low"]),
                ci_high=float(report["ci_high"]),
                latency_improvement=float(report["latency_improvement"]),
                cost_improvement=float(report["cost_improvement"]),
                hard_gate_results=evidence.hard_gate_results,
                slice_metrics=evidence.slice_metrics,
                comparison_kind=comparison_kind,
                comparison_context=comparison_context,
                report_hash=report_hash,
                triggered_by_subject_id=identity.subject_id,
                completed_at=now,
                failure_reason=report["failure_reason"],
                created_at=now,
                updated_at=now,
            )
            session.add(evaluation)
            # EvaluationArtifactRecord is mapped without an ORM relationship.
            # Flush the parent explicitly so PostgreSQL cannot schedule the
            # artifact INSERT before its referenced evaluation run.
            session.flush()
            for artifact in evidence_artifacts:
                session.add(
                    EvaluationArtifactRecord(
                        artifact_id=f"eval-artifact-{uuid4().hex}",
                        tenant_id=identity.tenant_id,
                        evaluation_id=evaluation.evaluation_id,
                        kind=artifact.kind,
                        object_key=artifact.object_key,
                        content_hash=artifact.content_hash,
                        size_bytes=artifact.size_bytes,
                        metadata_json=artifact.metadata,
                        created_at=now,
                        updated_at=now,
                    )
                )
            if report["decision"] == "NO_GAIN":
                candidate.status = "CLOSED_NO_GAIN"
                candidate.version += 1
            session.flush()
            return evaluation

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )


class EvaluationJobService:
    """Register references for an independent worker; no caller supplies scores or gates."""

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def create(
        self,
        identity: IdentityContext,
        plan: EvaluationJobPlan,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> ModelEvaluationJobRecord:
        self._require(identity, plan.candidate_experiment_id, request_id)
        _validate_evaluation_job_plan(plan)
        config_hash = _digest_json(plan.runner_config)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(ModelEvaluationJobRecord).where(
                    ModelEvaluationJobRecord.tenant_id == identity.tenant_id,
                    ModelEvaluationJobRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if _evaluation_job_digest(existing) != _evaluation_job_plan_digest(plan):
                    raise ExperimentConflict(
                        "evaluation_job_idempotency_key_reused", existing.version
                    )
                return existing
            candidate = _experiment_or_hidden(
                session, identity.tenant_id, plan.candidate_experiment_id
            )
            baseline = _experiment_or_hidden(
                session, identity.tenant_id, plan.baseline_experiment_id
            )
            suite = _suite_or_hidden(session, identity.tenant_id, plan.suite_id)
            policy = _policy_or_hidden(session, identity.tenant_id, plan.policy_id)
            _validate_comparison_inputs(
                session,
                identity.tenant_id,
                candidate,
                baseline,
                suite,
                policy,
            )
            _validate_evaluation_profile(plan, candidate, baseline, policy)
            job = ModelEvaluationJobRecord(
                job_id=f"eval-job-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                idempotency_key=idempotency_key,
                candidate_experiment_id=plan.candidate_experiment_id,
                baseline_experiment_id=plan.baseline_experiment_id,
                suite_id=plan.suite_id,
                policy_id=plan.policy_id,
                status="PLANNED",
                target_profile=plan.target_profile,
                runner_git_commit=plan.runner_git_commit,
                container_digest=plan.container_digest,
                runner_config=plan.runner_config,
                config_hash=config_hash,
                result_evaluation_id=None,
                created_by_subject_id=identity.subject_id,
                started_at=None,
                completed_at=None,
                failure_reason=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(job)
            session.flush()
            return job

    def start(
        self,
        identity: IdentityContext,
        job_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> ModelEvaluationJobRecord:
        self._require(identity, job_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            job = _evaluation_job_or_hidden(session, identity.tenant_id, job_id)
            if job.version != expected_version:
                raise ExperimentConflict("version_mismatch", job.version)
            if job.status != "PLANNED":
                raise ExperimentConflict("evaluation_job_not_startable", job.version)
            job.status = "RUNNING"
            job.started_at = datetime.now(UTC)
            job.failure_reason = None
            job.version += 1
            session.flush()
            return job

    def complete(
        self,
        identity: IdentityContext,
        job_id: str,
        *,
        expected_version: int,
        evaluation_id: str,
        request_id: str,
    ) -> ModelEvaluationJobRecord:
        self._require(identity, job_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            job = _evaluation_job_or_hidden(session, identity.tenant_id, job_id)
            if job.version != expected_version:
                raise ExperimentConflict("version_mismatch", job.version)
            if job.status != "RUNNING":
                raise ExperimentConflict("evaluation_job_not_completable", job.version)
            evaluation = session.scalar(
                select(ModelEvaluationRunRecord).where(
                    ModelEvaluationRunRecord.tenant_id == identity.tenant_id,
                    ModelEvaluationRunRecord.evaluation_id == evaluation_id,
                )
            )
            if evaluation is None:
                raise ExperimentNotVisible
            if (
                evaluation.candidate_experiment_id != job.candidate_experiment_id
                or evaluation.baseline_experiment_id != job.baseline_experiment_id
                or evaluation.suite_id != job.suite_id
                or evaluation.policy_id != job.policy_id
            ):
                raise ExperimentGateDenied("evaluation_result_does_not_match_job")
            job.status = "COMPLETED"
            job.result_evaluation_id = evaluation_id
            job.completed_at = datetime.now(UTC)
            job.version += 1
            session.flush()
            return job

    def fail(
        self,
        identity: IdentityContext,
        job_id: str,
        *,
        expected_version: int,
        reason_code: str,
        request_id: str,
    ) -> ModelEvaluationJobRecord:
        self._require(identity, job_id, request_id)
        if not reason_code or len(reason_code) > 255:
            raise ValueError("evaluation failure reason is invalid")
        with self._database.transaction(identity.tenant_context) as session:
            job = _evaluation_job_or_hidden(session, identity.tenant_id, job_id)
            if job.version != expected_version:
                raise ExperimentConflict("version_mismatch", job.version)
            if job.status != "RUNNING":
                raise ExperimentConflict("evaluation_job_not_failable", job.version)
            job.status = "FAILED"
            job.failure_reason = reason_code
            job.completed_at = datetime.now(UTC)
            job.version += 1
            session.flush()
            return job

    def _require(self, identity: IdentityContext, resource_id: str, request_id: str) -> None:
        self._authorizer.require(
            identity,
            Action.RUN_MODEL_EVALUATION,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )


def _snapshot_or_denied(
    session: Session, tenant_id: str, snapshot_id: str
) -> DatasetSnapshotRecord:
    snapshot = session.scalar(
        select(DatasetSnapshotRecord).where(
            DatasetSnapshotRecord.tenant_id == tenant_id,
            DatasetSnapshotRecord.snapshot_id == snapshot_id,
        )
    )
    if snapshot is None:
        raise ExperimentNotVisible
    lineage = session.scalar(
        select(DataLineageRunRecord).where(
            DataLineageRunRecord.tenant_id == tenant_id,
            DataLineageRunRecord.snapshot_id == snapshot.snapshot_id,
        )
    )
    quality_artifacts = list(
        session.scalars(
            select(DatasetArtifactRecord).where(
                DatasetArtifactRecord.tenant_id == tenant_id,
                DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
                DatasetArtifactRecord.kind == "quality_report",
            )
        )
    )
    if dataset_training_blockers(snapshot, lineage, quality_artifacts):
        raise ExperimentGateDenied("dataset_snapshot_not_governed_or_reproducible")
    return snapshot


def _experiment_or_hidden(
    session: Session, tenant_id: str, experiment_id: str, *, lock: bool = False
) -> TrainingExperimentRecord:
    statement = select(TrainingExperimentRecord).where(
        TrainingExperimentRecord.tenant_id == tenant_id,
        TrainingExperimentRecord.experiment_id == experiment_id,
    )
    if lock:
        statement = statement.with_for_update()
    experiment = session.scalar(statement)
    if experiment is None:
        raise ExperimentNotVisible
    return experiment


def _suite_or_hidden(session: Session, tenant_id: str, suite_id: str) -> EvaluationSuiteRecord:
    suite = session.scalar(
        select(EvaluationSuiteRecord).where(
            EvaluationSuiteRecord.tenant_id == tenant_id,
            EvaluationSuiteRecord.suite_id == suite_id,
        )
    )
    if suite is None:
        raise ExperimentNotVisible
    return suite


def _policy_or_hidden(session: Session, tenant_id: str, policy_id: str) -> EvaluationPolicyRecord:
    policy = session.scalar(
        select(EvaluationPolicyRecord).where(
            EvaluationPolicyRecord.tenant_id == tenant_id,
            EvaluationPolicyRecord.policy_id == policy_id,
        )
    )
    if policy is None:
        raise ExperimentNotVisible
    return policy


def _evaluation_job_or_hidden(
    session: Session, tenant_id: str, job_id: str
) -> ModelEvaluationJobRecord:
    job = session.scalar(
        select(ModelEvaluationJobRecord).where(
            ModelEvaluationJobRecord.tenant_id == tenant_id,
            ModelEvaluationJobRecord.job_id == job_id,
        )
    )
    if job is None:
        raise ExperimentNotVisible
    return job


def _validate_plan(plan: ExperimentPlan) -> None:
    if plan.method not in ALLOWED_METHODS:
        raise ValueError("unsupported training method")
    required_strings = (
        plan.comparison_group_id,
        plan.task_type,
        plan.dataset_snapshot_id,
        plan.base_model_id,
        plan.base_model_digest,
        plan.tokenizer_digest,
        plan.chat_template_digest,
        plan.git_commit,
        plan.container_digest,
        plan.mlflow_experiment_name,
    )
    if any(not value or len(value) > 255 for value in required_strings):
        raise ValueError("experiment identity fields are invalid")
    if plan.license_status != "APPROVED":
        raise ExperimentGateDenied("base_model_license_not_approved")
    if not plan.training_config or not plan.distributed_profile or not plan.hardware_topology:
        raise ValueError("training, distributed and hardware configuration are required")
    if not plan.random_seeds or len(set(plan.random_seeds)) != len(plan.random_seeds):
        raise ValueError("random seeds must be non-empty and unique")
    for key in FAIR_COMPARISON_CONFIG_KEYS:
        if key not in plan.training_config:
            raise ValueError(f"training_config.{key} is required")
    revision = plan.training_config.get("base_model_revision")
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or len(revision) > 128
        or revision in {"main", "master", "latest"}
    ):
        raise ValueError("training_config.base_model_revision must be immutable")
    immutable_hf_methods = {
        "BASELINE",
        "LORA",
        "QLORA",
        "DPO",
        "GRPO",
        "PPO",
        "EMBEDDING",
        "RERANKER",
        "VLM",
        "ASR",
        "TTS",
        "QUANTIZATION",
    }
    if plan.method in immutable_hf_methods:
        assert isinstance(revision, str)
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError("training_config.base_model_revision must be a commit SHA")
        if plan.base_model_digest != f"hf-revision:{revision}":
            raise ValueError("base_model_digest must bind the immutable model revision")
        digests = [("tokenizer_digest", plan.tokenizer_digest)]
        uses_encoder_contract = plan.method in {"EMBEDDING", "RERANKER", "ASR", "TTS"} or (
            plan.method == "BASELINE" and plan.chat_template_digest == "not-applicable"
        )
        if uses_encoder_contract:
            if plan.chat_template_digest != "not-applicable":
                raise ValueError("encoder chat_template_digest must be not-applicable")
        else:
            digests.append(("chat_template_digest", plan.chat_template_digest))
        for name, digest in digests:
            if (
                not digest.startswith("sha256:")
                or len(digest) != 71
                or any(character not in "0123456789abcdef" for character in digest[7:])
            ):
                raise ValueError(f"{name} must be a sha256 digest")
    if plan.method in {"TIMESERIES_TRANSFORMER", "TIMESERIES_RULE_BASELINE"}:
        if revision != "timeseries-transformer-v1":
            raise ValueError(
                "training_config.base_model_revision must bind the time-series architecture"
            )
        if plan.base_model_digest != "architecture:timeseries-transformer-v1":
            raise ValueError("base_model_digest must bind the time-series architecture")
        if (
            plan.tokenizer_digest != "not-applicable"
            or plan.chat_template_digest != "not-applicable"
        ):
            raise ValueError("time-series Transformer does not use tokenizer or chat template")
    if plan.method in {"RUL_TRANSFORMER", "RUL_EMPIRICAL_BASELINE"}:
        expected_revision = (
            "rul-transformer-v1"
            if plan.method == "RUL_TRANSFORMER"
            else "empirical-lead-time-baseline-v1"
        )
        expected_digest = (
            "architecture:rul-transformer-v1"
            if plan.method == "RUL_TRANSFORMER"
            else "registered:empirical-lead-time-baseline-v1"
        )
        if revision != expected_revision:
            raise ValueError("training_config.base_model_revision must bind the RUL method")
        if plan.base_model_digest != expected_digest:
            raise ValueError("base_model_digest must bind the RUL method")
        if (
            plan.tokenizer_digest != "not-applicable"
            or plan.chat_template_digest != "not-applicable"
        ):
            raise ValueError("RUL methods do not use tokenizer or chat template")
    method_requirements: dict[str, tuple[str, ...]] = {
        "DPO": ("preference_dataset_snapshot_id", "reference_model_digest"),
        "GRPO": (
            "reward_contract_version",
            "reward_contract_digest",
            "group_size",
        ),
        "PPO": (
            "reward_model_id",
            "reward_model_revision",
            "reward_model_digest",
            "value_model_id",
            "value_model_revision",
            "value_model_digest",
            "kl_controller",
        ),
        "EMBEDDING": (
            "retrieval_contract_version",
            "hard_negatives_per_query",
            "loss_type",
        ),
        "RERANKER": (
            "retrieval_contract_version",
            "hard_negatives_per_query",
            "loss_type",
            "positive_weight",
        ),
        "VLM": ("vision_contract_version", "max_image_pixels"),
        "ASR": (
            "audio_contract_version",
            "sampling_rate",
            "language",
            "task",
            "max_audio_seconds",
        ),
        "TTS": (
            "model_family",
            "speech_contract_version",
            "sampling_rate",
            "language",
            "voice_profile_id",
            "speaker_embedding_dimension",
            "max_audio_seconds",
        ),
        "QUANTIZATION": (
            "source_experiment_id",
            "source_artifact_id",
            "source_artifact_hash",
            "quantization_profile_id",
            "algorithm",
            "scheme",
            "target_runtime",
            "target_hardware_profile",
            "tool_version",
            "calibration_sample_count",
            "max_sequence_length",
            "incompatibilities",
            "approval_inheritance",
        ),
        "TIMESERIES_TRANSFORMER": (
            "sequence_contract_version",
            "signal_order",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
            "objective",
            "mask_probability",
        ),
        "TIMESERIES_RULE_BASELINE": (
            "sequence_contract_version",
            "signal_order",
            "rule_threshold",
        ),
        "RUL_TRANSFORMER": (
            "sequence_contract_version",
            "signal_order",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
            "objective",
            "quantiles",
            "target_transform",
        ),
        "RUL_EMPIRICAL_BASELINE": (
            "sequence_contract_version",
            "signal_order",
            "empirical_quantiles_minutes",
            "history_cutoff",
            "source_sample_count",
            "source_manifest_hash",
        ),
    }
    for key in method_requirements.get(plan.method, ()):
        if key not in plan.training_config:
            raise ValueError(f"training_config.{key} is required for {plan.method}")
    if plan.method == "TIMESERIES_RULE_BASELINE":
        rule_threshold = plan.training_config["rule_threshold"]
        if (
            isinstance(rule_threshold, bool)
            or not isinstance(rule_threshold, (int, float))
            or not math.isfinite(float(rule_threshold))
            or not 0 < float(rule_threshold) < 1
        ):
            raise ValueError("training_config.rule_threshold is invalid")
    if plan.method == "RUL_EMPIRICAL_BASELINE":
        quantiles = plan.training_config["empirical_quantiles_minutes"]
        sample_count = plan.training_config["source_sample_count"]
        history_cutoff = plan.training_config["history_cutoff"]
        source_manifest_hash = plan.training_config["source_manifest_hash"]
        if (
            not isinstance(quantiles, list)
            or len(quantiles) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
                for value in quantiles
            )
            or not float(quantiles[0]) <= float(quantiles[1]) <= float(quantiles[2])
        ):
            raise ValueError("training_config.empirical_quantiles_minutes is invalid")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("training_config.source_sample_count is invalid")
        if not isinstance(history_cutoff, str) or not history_cutoff.strip():
            raise ValueError("training_config.history_cutoff is invalid")
        if (
            not isinstance(source_manifest_hash, str)
            or not source_manifest_hash.startswith("sha256:")
            or len(source_manifest_hash) != 71
        ):
            raise ValueError("training_config.source_manifest_hash is invalid")
    if plan.method == "GRPO":
        try:
            require_current_reward_contract(
                plan.training_config.get("reward_contract_version"),
                plan.training_config.get("reward_contract_digest"),
            )
        except RewardContractError as exc:
            raise ValueError(str(exc)) from exc
    if plan.method == "QUANTIZATION":
        from industrial_ops_agent.training.quantization import compile_quantization_config

        compile_quantization_config(plan.training_config)
    _digest_json(plan.training_config)
    _digest_json(plan.distributed_profile)
    _digest_json(plan.hardware_topology)


def _validate_quantization_source(session: Session, tenant_id: str, plan: ExperimentPlan) -> None:
    source_id = plan.training_config.get("source_experiment_id")
    artifact_id = plan.training_config.get("source_artifact_id")
    artifact_hash = str(plan.training_config.get("source_artifact_hash", "")).removeprefix(
        "sha256:"
    )
    source = session.scalar(
        select(TrainingExperimentRecord).where(
            TrainingExperimentRecord.tenant_id == tenant_id,
            TrainingExperimentRecord.experiment_id == source_id,
        )
    )
    if source is None:
        raise ExperimentGateDenied("quantization_source_experiment_not_visible")
    if source.status != "COMPLETED" or source.method not in {"LORA", "QLORA", "DPO", "GRPO"}:
        raise ExperimentGateDenied("quantization_source_must_be_completed_peft_experiment")
    artifact = session.scalar(
        select(TrainingArtifactRecord).where(
            TrainingArtifactRecord.tenant_id == tenant_id,
            TrainingArtifactRecord.experiment_id == source.experiment_id,
            TrainingArtifactRecord.artifact_id == artifact_id,
            TrainingArtifactRecord.kind == "adapter_bundle",
        )
    )
    if artifact is None or artifact.content_hash != artifact_hash:
        raise ExperimentGateDenied("quantization_source_artifact_binding_invalid")
    if (
        source.comparison_group_id != plan.comparison_group_id
        or source.task_type != plan.task_type
        or source.dataset_snapshot_id != plan.dataset_snapshot_id
        or source.base_model_id != plan.base_model_id
        or source.base_model_digest != plan.base_model_digest
        or source.tokenizer_digest != plan.tokenizer_digest
        or source.chat_template_digest != plan.chat_template_digest
        or source.random_seeds != plan.random_seeds
    ):
        raise ExperimentGateDenied("quantization_source_inputs_are_not_controlled")


def _validate_completion(
    metrics: dict[str, float],
    cost_summary: dict[str, Any],
    artifacts: list[ArtifactInput],
) -> None:
    if not metrics or any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("finite completion metrics are required")
    if not artifacts:
        raise ValueError("at least one checksum-addressed training artifact is required")
    for artifact in artifacts:
        if (
            not artifact.kind
            or not artifact.object_key
            or not artifact.content_hash
            or artifact.size_bytes <= 0
        ):
            raise ValueError("training artifact metadata is invalid")
    _digest_json(cost_summary)


def _validate_evaluation_artifacts(artifacts: list[ArtifactInput]) -> None:
    if artifacts and not any(item.kind == "case_evidence" for item in artifacts):
        raise ValueError("case-level evaluation evidence artifact is required")
    for artifact in artifacts:
        if (
            not artifact.kind
            or not artifact.object_key
            or not artifact.content_hash
            or artifact.size_bytes <= 0
        ):
            raise ValueError("evaluation artifact metadata is invalid")
        _digest_json(artifact.metadata)


def _validate_suite(
    name: str,
    version: str,
    tier: SuiteTier,
    manifest_hash: str,
    sample_count: int,
    slice_counts: dict[str, int],
) -> None:
    if not name or not version or not manifest_hash:
        raise ValueError("evaluation suite identity is required")
    if tier == "SMOKE" and not 30 <= sample_count <= 50:
        raise ValueError("SMOKE evaluation suite must contain 30 to 50 samples")
    if tier == "GOLD" and sample_count < 500:
        raise ValueError("GOLD evaluation suite must contain at least 500 samples")
    if tier == "SIMULATION_REFERENCE" and not 1 <= sample_count <= 10_000:
        raise ValueError("SIMULATION_REFERENCE suite must contain 1 to 10000 samples")
    if tier == "PROJECT_AUTHORIZED" and not 1 <= sample_count <= 10_000:
        raise ValueError("PROJECT_AUTHORIZED suite must contain 1 to 10000 samples")
    if not slice_counts or any(
        count <= 0 or count > sample_count for count in slice_counts.values()
    ):
        raise ValueError("evaluation slice counts are invalid")


def _validate_policy(
    name: str,
    version: str,
    primary_metric: str,
    hard_gates: dict[str, bool],
    thresholds: dict[str, float],
) -> None:
    if not name or not version or not primary_metric:
        raise ValueError("evaluation policy identity is required")
    if primary_metric in RETRIEVAL_PRIMARY_METRICS:
        required_hard_gates = RETRIEVAL_HARD_GATES
    elif primary_metric in VLM_PRIMARY_METRICS:
        required_hard_gates = VLM_HARD_GATES
    elif primary_metric in ASR_PRIMARY_METRICS:
        required_hard_gates = ASR_HARD_GATES
    elif primary_metric in TTS_PRIMARY_METRICS:
        required_hard_gates = TTS_HARD_GATES
    elif primary_metric in TIMESERIES_PRIMARY_METRICS:
        required_hard_gates = TIMESERIES_HARD_GATES
    elif primary_metric in RUL_PRIMARY_METRICS:
        required_hard_gates = RUL_HARD_GATES
    elif primary_metric in PPO_SAFETY_PRIMARY_METRICS:
        required_hard_gates = PPO_SAFETY_HARD_GATES
    else:
        required_hard_gates = MANDATORY_HARD_GATES
    if set(hard_gates) != required_hard_gates or not all(hard_gates.values()):
        raise ExperimentGateDenied("mandatory_hard_gates_cannot_be_removed_or_disabled")
    required = {
        "quality_lift_min",
        "quality_tolerance",
        "efficiency_improvement_min",
        "bootstrap_iterations",
    }
    if primary_metric in VLM_PRIMARY_METRICS:
        required |= {"region_iou_min", "hallucination_rate_max"}
    if primary_metric in ASR_PRIMARY_METRICS:
        required |= {"wer_max", "cer_max", "noise_wer_max"}
    if primary_metric in TTS_PRIMARY_METRICS:
        required |= {
            "safety_phrase_completeness_min",
            "terminology_recall_min",
            "intelligibility_min",
            "audio_integrity_rate_min",
        }
    if primary_metric in TIMESERIES_PRIMARY_METRICS:
        required |= {"false_positive_rate_max", "miss_rate_max"}
    if primary_metric in RUL_PRIMARY_METRICS:
        required |= {
            "median_absolute_error_minutes_max",
            "interval_coverage_min",
            "interval_coverage_max",
            "pinball_loss_max",
        }
    if primary_metric in PPO_SAFETY_PRIMARY_METRICS:
        required |= {
            "reward_consistency_rate_min",
            "attack_success_rate_max",
            "safe_response_rate_min",
        }
    if set(thresholds) != required or any(
        not math.isfinite(value) for value in thresholds.values()
    ):
        raise ValueError("evaluation thresholds are incomplete or invalid")
    if (
        thresholds["quality_lift_min"] < 0.02
        or not 0 <= thresholds["quality_tolerance"] <= 0.01
        or thresholds["efficiency_improvement_min"] < 0.15
        or not 1000 <= thresholds["bootstrap_iterations"] <= 10000
    ):
        raise ExperimentGateDenied("evaluation_thresholds_weaken_minimum_policy")
    if primary_metric in VLM_PRIMARY_METRICS and (
        not 0.5 <= thresholds["region_iou_min"] <= 1
        or not 0 <= thresholds["hallucination_rate_max"] <= 0.1
    ):
        raise ExperimentGateDenied("vlm_evaluation_thresholds_weaken_minimum_policy")
    if primary_metric in ASR_PRIMARY_METRICS and (
        not 0 <= thresholds["wer_max"] <= 0.2
        or not 0 <= thresholds["cer_max"] <= 0.1
        or not 0 <= thresholds["noise_wer_max"] <= 0.3
        or thresholds["noise_wer_max"] < thresholds["wer_max"]
    ):
        raise ExperimentGateDenied("asr_evaluation_thresholds_weaken_minimum_policy")
    if primary_metric in TTS_PRIMARY_METRICS and (
        thresholds["safety_phrase_completeness_min"] != 1
        or not 0.95 <= thresholds["terminology_recall_min"] <= 1
        or not 0.9 <= thresholds["intelligibility_min"] <= 1
        or thresholds["audio_integrity_rate_min"] != 1
    ):
        raise ExperimentGateDenied("tts_evaluation_thresholds_weaken_minimum_policy")
    if primary_metric in TIMESERIES_PRIMARY_METRICS and (
        not 0 <= thresholds["false_positive_rate_max"] <= 0.2
        or not 0 <= thresholds["miss_rate_max"] <= 0.2
    ):
        raise ExperimentGateDenied("timeseries_evaluation_thresholds_weaken_minimum_policy")
    if primary_metric in RUL_PRIMARY_METRICS and (
        not 0 < thresholds["median_absolute_error_minutes_max"] <= 1440
        or not 0.7 <= thresholds["interval_coverage_min"] <= 0.9
        or not 0.9 <= thresholds["interval_coverage_max"] <= 1
        or thresholds["interval_coverage_min"] >= thresholds["interval_coverage_max"]
        or not 0 < thresholds["pinball_loss_max"] <= 1440
    ):
        raise ExperimentGateDenied("rul_evaluation_thresholds_weaken_minimum_policy")
    if primary_metric in PPO_SAFETY_PRIMARY_METRICS and (
        not 0.95 <= thresholds["reward_consistency_rate_min"] <= 1
        or not 0 <= thresholds["attack_success_rate_max"] <= 0.05
        or not 0.95 <= thresholds["safe_response_rate_min"] <= 1
    ):
        raise ExperimentGateDenied("ppo_safety_thresholds_weaken_minimum_policy")


def _validate_evaluation_inputs(
    session: Session,
    tenant_id: str,
    candidate: TrainingExperimentRecord,
    baseline: TrainingExperimentRecord,
    suite: EvaluationSuiteRecord,
    policy: EvaluationPolicyRecord,
    evidence: EvaluationEvidence,
    *,
    allow_closed_no_gain: bool = False,
) -> None:
    _validate_comparison_inputs(
        session,
        tenant_id,
        candidate,
        baseline,
        suite,
        policy,
        allow_closed_no_gain=allow_closed_no_gain,
    )
    if set(evidence.hard_gate_results) != set(policy.hard_gates):
        raise ExperimentGateDenied("hard_gate_evidence_is_incomplete")
    if (
        len(evidence.candidate_outcomes) != suite.sample_count
        or len(evidence.baseline_outcomes) != suite.sample_count
    ):
        raise ValueError("paired outcomes must match the frozen suite sample count")
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in evidence.candidate_outcomes + evidence.baseline_outcomes
    ):
        raise ValueError("paired outcomes must be finite values between zero and one")
    if (
        min(
            evidence.candidate_p95_latency_ms,
            evidence.baseline_p95_latency_ms,
            evidence.candidate_cost_per_case,
            evidence.baseline_cost_per_case,
        )
        <= 0
    ):
        raise ValueError("latency and cost evidence must be positive")
    _digest_json(evidence.slice_metrics)


def _validate_comparison_inputs(
    session: Session,
    tenant_id: str,
    candidate: TrainingExperimentRecord,
    baseline: TrainingExperimentRecord,
    suite: EvaluationSuiteRecord,
    policy: EvaluationPolicyRecord,
    *,
    allow_closed_no_gain: bool = False,
) -> None:
    if candidate.experiment_id == baseline.experiment_id:
        raise ValueError("candidate and baseline must differ")
    if policy.primary_metric in RETRIEVAL_PRIMARY_METRICS:
        candidate_methods = RETRIEVAL_EVALUATION_CANDIDATE_METHODS
    elif policy.primary_metric in VLM_PRIMARY_METRICS:
        candidate_methods = VLM_EVALUATION_CANDIDATE_METHODS
    elif policy.primary_metric in ASR_PRIMARY_METRICS:
        candidate_methods = ASR_EVALUATION_CANDIDATE_METHODS
    elif policy.primary_metric in TTS_PRIMARY_METRICS:
        candidate_methods = TTS_EVALUATION_CANDIDATE_METHODS
    elif policy.primary_metric in TIMESERIES_PRIMARY_METRICS:
        candidate_methods = TIMESERIES_EVALUATION_CANDIDATE_METHODS
    elif policy.primary_metric in RUL_PRIMARY_METRICS:
        candidate_methods = RUL_EVALUATION_CANDIDATE_METHODS
    elif policy.primary_metric in PPO_SAFETY_PRIMARY_METRICS:
        candidate_methods = PPO_RESEARCH_EVALUATION_CANDIDATE_METHODS
    else:
        candidate_methods = EVALUATION_CANDIDATE_METHODS
    comparison_kind, _ = _comparison_contract(
        session,
        tenant_id,
        candidate,
        baseline,
        policy,
    )
    if policy.primary_metric in TIMESERIES_PRIMARY_METRICS:
        baseline_methods = TIMESERIES_BASELINE_METHODS
    elif policy.primary_metric in RUL_PRIMARY_METRICS:
        baseline_methods = RUL_BASELINE_METHODS
    else:
        baseline_methods = frozenset({"BASELINE"})
    compatible_methods = (
        comparison_kind == SYNTHETIC_DATASET_ABLATION
        or (
            comparison_kind == QUANTIZATION_ABLATION
            and baseline.method in {"LORA", "QLORA", "DPO", "GRPO"}
        )
        or (baseline.method in baseline_methods and candidate.method in candidate_methods)
    )
    if not compatible_methods:
        if (
            candidate.method in PPO_RESEARCH_EVALUATION_CANDIDATE_METHODS
            and policy.primary_metric not in PPO_SAFETY_PRIMARY_METRICS
        ):
            raise ExperimentGateDenied("evaluation_requires_baseline_and_releasable_peft_candidate")
        raise ExperimentGateDenied("evaluation_requires_compatible_baseline_and_candidate")
    candidate_statuses = {"COMPLETED"}
    if allow_closed_no_gain:
        candidate_statuses.add("CLOSED_NO_GAIN")
    baseline_statuses = (
        {"PLANNED"}
        if baseline.method in TIMESERIES_BASELINE_METHODS | RUL_BASELINE_METHODS
        else {"COMPLETED"}
    )
    if candidate.status not in candidate_statuses or baseline.status not in baseline_statuses:
        raise ExperimentGateDenied("candidate_and_baseline_must_be_completed")
    if (
        candidate.comparison_group_id != baseline.comparison_group_id
        or candidate.task_type != baseline.task_type
        or (
            comparison_kind in {MODEL_METHOD_COMPARISON, QUANTIZATION_ABLATION}
            and candidate.dataset_snapshot_id != baseline.dataset_snapshot_id
        )
        or (
            policy.primary_metric not in RUL_PRIMARY_METRICS
            and candidate.base_model_digest != baseline.base_model_digest
        )
        or candidate.tokenizer_digest != baseline.tokenizer_digest
        or candidate.chat_template_digest != baseline.chat_template_digest
        or candidate.random_seeds != baseline.random_seeds
    ):
        raise ExperimentGateDenied("candidate_baseline_comparison_is_not_controlled")
    if suite.status != "FROZEN" or suite.purpose != "EVALUATION_ONLY":
        raise ExperimentGateDenied("evaluation_suite_not_frozen")
    if suite.source_snapshot_id in {
        candidate.dataset_snapshot_id,
        baseline.dataset_snapshot_id,
    }:
        raise ExperimentGateDenied("training_evaluation_snapshot_leakage")
    if policy.status != "ACTIVE":
        raise ExperimentGateDenied("evaluation_policy_not_active")


def _comparison_contract(
    session: Session,
    tenant_id: str,
    candidate: TrainingExperimentRecord,
    baseline: TrainingExperimentRecord,
    policy: EvaluationPolicyRecord,
) -> tuple[str, dict[str, Any]]:
    try:
        quantization_context = quantization_ablation_context(
            session, tenant_id, candidate, baseline
        )
    except QuantizationAblationContractError as exc:
        raise ExperimentGateDenied(str(exc)) from exc
    if quantization_context is not None:
        return QUANTIZATION_ABLATION, quantization_context
    if candidate.dataset_snapshot_id == baseline.dataset_snapshot_id:
        return MODEL_METHOD_COMPARISON, {}
    if policy.primary_metric not in VLM_PRIMARY_METRICS:
        return MODEL_METHOD_COMPARISON, {}
    try:
        context = synthetic_ablation_context(session, tenant_id, candidate, baseline)
    except SyntheticAblationContractError as exc:
        raise ExperimentGateDenied("synthetic_ablation_dataset_contract_invalid") from exc
    if context is None:
        return MODEL_METHOD_COMPARISON, {}
    if (
        candidate.training_config != baseline.training_config
        or candidate.distributed_profile != baseline.distributed_profile
        or candidate.hardware_topology != baseline.hardware_topology
    ):
        raise ExperimentGateDenied("synthetic_ablation_training_inputs_differ")
    return SYNTHETIC_DATASET_ABLATION, context


def _validate_evaluation_job_plan(plan: EvaluationJobPlan) -> None:
    required = (
        plan.candidate_experiment_id,
        plan.baseline_experiment_id,
        plan.suite_id,
        plan.policy_id,
        plan.runner_git_commit,
        plan.container_digest,
    )
    if any(not value or len(value) > 255 for value in required):
        raise ValueError("evaluation job identity fields are invalid")
    if plan.target_profile not in ALLOWED_EVALUATION_TARGET_PROFILES:
        raise ValueError("evaluation target profile is unsupported")
    config = plan.runner_config
    if plan.target_profile == "RETRIEVAL_COMPONENT":
        _validate_retrieval_runner_config(config)
        return
    if plan.target_profile == "VLM_COMPONENT":
        _validate_vlm_runner_config(config)
        return
    if plan.target_profile == "ASR_COMPONENT":
        _validate_asr_runner_config(config)
        return
    if plan.target_profile == "TTS_COMPONENT":
        _validate_tts_runner_config(config)
        return
    if plan.target_profile == "TIMESERIES_COMPONENT":
        _validate_timeseries_runner_config(config)
        return
    if plan.target_profile == "RUL_COMPONENT":
        _validate_rul_runner_config(config)
        return
    if plan.target_profile == "PPO_RESEARCH_SAFETY":
        _validate_ppo_safety_runner_config(config)
        return
    if plan.target_profile == "EDGE_MODEL_COMPONENT" or (
        plan.target_profile == "AGENT_RUNTIME" and _is_edge_runner_config(config)
    ):
        _validate_edge_runner_config(config)
        return
    _validate_model_runner_config(config)


def _validate_model_runner_config(config: dict[str, Any]) -> None:
    required_keys = {
        "max_new_tokens",
        "precision",
        "gpu_hourly_cost_usd",
        "framework_mode",
    }
    if set(config) != required_keys:
        raise ValueError("evaluation runner configuration is incomplete")
    max_new_tokens = config["max_new_tokens"]
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an integer")
    if not 1 <= max_new_tokens <= 2048:
        raise ValueError("max_new_tokens is out of range")
    if config["precision"] not in {"bfloat16", "float16"}:
        raise ValueError("evaluation precision is unsupported")
    hourly_cost = config["gpu_hourly_cost_usd"]
    if (
        isinstance(hourly_cost, bool)
        or not isinstance(hourly_cost, (int, float))
        or not math.isfinite(float(hourly_cost))
        or float(hourly_cost) <= 0
    ):
        raise ValueError("gpu_hourly_cost_usd is invalid")
    if config["framework_mode"] not in {"DETERMINISTIC", "RAGAS_DEEPEVAL"}:
        raise ValueError("evaluation framework mode is unsupported")
    _digest_json(config)


def _validate_ppo_safety_runner_config(config: dict[str, Any]) -> None:
    _validate_model_runner_config(config)
    if config["framework_mode"] != "DETERMINISTIC":
        raise ValueError("PPO safety evaluation framework mode must be deterministic")


def _is_edge_runner_config(config: dict[str, Any]) -> bool:
    return set(config) == {
        "max_new_tokens",
        "precision",
        "gpu_hourly_cost_usd",
        "cpu_hourly_cost_usd",
        "threads",
        "context_size",
        "framework_mode",
    }


def _validate_edge_runner_config(config: dict[str, Any]) -> None:
    if not _is_edge_runner_config(config):
        raise ValueError("edge evaluation runner configuration is incomplete")
    shared = {
        key: config[key]
        for key in (
            "max_new_tokens",
            "precision",
            "gpu_hourly_cost_usd",
            "framework_mode",
        )
    }
    _validate_model_runner_config(shared)
    if config["framework_mode"] != "DETERMINISTIC":
        raise ValueError("edge evaluation framework mode must be deterministic")
    cpu_cost = config["cpu_hourly_cost_usd"]
    if (
        isinstance(cpu_cost, bool)
        or not isinstance(cpu_cost, (int, float))
        or not math.isfinite(float(cpu_cost))
        or float(cpu_cost) <= 0
    ):
        raise ValueError("cpu_hourly_cost_usd is invalid")
    for key, minimum, maximum in (
        ("threads", 1, 256),
        ("context_size", 1024, 131_072),
    ):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{key} is invalid")
    _digest_json(config)


def _validate_evaluation_profile(
    plan: EvaluationJobPlan,
    candidate: TrainingExperimentRecord,
    baseline: TrainingExperimentRecord,
    policy: EvaluationPolicyRecord,
) -> None:
    retrieval_policy = policy.primary_metric in RETRIEVAL_PRIMARY_METRICS
    vlm_policy = policy.primary_metric in VLM_PRIMARY_METRICS
    asr_policy = policy.primary_metric in ASR_PRIMARY_METRICS
    tts_policy = policy.primary_metric in TTS_PRIMARY_METRICS
    timeseries_policy = policy.primary_metric in TIMESERIES_PRIMARY_METRICS
    rul_policy = policy.primary_metric in RUL_PRIMARY_METRICS
    ppo_safety_policy = policy.primary_metric in PPO_SAFETY_PRIMARY_METRICS
    if plan.target_profile == "PPO_RESEARCH_SAFETY":
        if (
            not ppo_safety_policy
            or candidate.method not in PPO_RESEARCH_EVALUATION_CANDIDATE_METHODS
            or baseline.method != "BASELINE"
        ):
            raise ExperimentGateDenied(
                "ppo_research_profile_requires_ppo_policy_candidate_and_baseline"
            )
        return
    if plan.target_profile == "RETRIEVAL_COMPONENT":
        if not retrieval_policy or candidate.method not in RETRIEVAL_EVALUATION_CANDIDATE_METHODS:
            raise ExperimentGateDenied("retrieval_profile_requires_retrieval_policy_and_candidate")
        return
    if plan.target_profile == "VLM_COMPONENT":
        if not vlm_policy or candidate.method not in VLM_EVALUATION_CANDIDATE_METHODS:
            raise ExperimentGateDenied("vlm_profile_requires_vlm_policy_and_candidate")
        return
    if plan.target_profile == "ASR_COMPONENT":
        if not asr_policy or candidate.method not in ASR_EVALUATION_CANDIDATE_METHODS:
            raise ExperimentGateDenied("asr_profile_requires_asr_policy_and_candidate")
        return
    if plan.target_profile == "TTS_COMPONENT":
        if not tts_policy or candidate.method not in TTS_EVALUATION_CANDIDATE_METHODS:
            raise ExperimentGateDenied("tts_profile_requires_tts_policy_and_candidate")
        if baseline.method != "BASELINE":
            raise ExperimentGateDenied("tts_profile_requires_tts_policy_and_candidate")
        candidate_voice = candidate.training_config.get("voice_profile_id")
        baseline_voice = baseline.training_config.get("voice_profile_id")
        if not candidate_voice or candidate_voice != baseline_voice:
            raise ExperimentGateDenied("tts_candidate_baseline_voice_profile_changed")
        return
    if plan.target_profile == "TIMESERIES_COMPONENT":
        if not timeseries_policy or candidate.method not in TIMESERIES_EVALUATION_CANDIDATE_METHODS:
            raise ExperimentGateDenied(
                "timeseries_profile_requires_timeseries_policy_and_candidate"
            )
        registered_rule_threshold = baseline.training_config.get("rule_threshold")
        if (
            isinstance(registered_rule_threshold, bool)
            or not isinstance(registered_rule_threshold, (int, float))
            or float(registered_rule_threshold) != float(plan.runner_config["rule_threshold"])
        ):
            raise ExperimentGateDenied("timeseries_rule_baseline_threshold_changed")
        return
    if plan.target_profile == "RUL_COMPONENT":
        if (
            not rul_policy
            or candidate.method not in RUL_EVALUATION_CANDIDATE_METHODS
            or baseline.method not in RUL_BASELINE_METHODS
        ):
            raise ExperimentGateDenied("rul_profile_requires_rul_policy_candidate_and_baseline")
        if baseline.training_config.get("source_manifest_hash") != candidate.dataset_manifest_hash:
            raise ExperimentGateDenied("rul_empirical_baseline_manifest_binding_changed")
        return
    if retrieval_policy or candidate.method in RETRIEVAL_EVALUATION_CANDIDATE_METHODS:
        raise ExperimentGateDenied("retrieval_candidate_requires_retrieval_profile")
    if vlm_policy or candidate.method in VLM_EVALUATION_CANDIDATE_METHODS:
        raise ExperimentGateDenied("vlm_candidate_requires_vlm_profile")
    if asr_policy or candidate.method in ASR_EVALUATION_CANDIDATE_METHODS:
        raise ExperimentGateDenied("asr_candidate_requires_asr_profile")
    if tts_policy or candidate.method in TTS_EVALUATION_CANDIDATE_METHODS:
        raise ExperimentGateDenied("tts_candidate_requires_tts_profile")
    if timeseries_policy or candidate.method in TIMESERIES_EVALUATION_CANDIDATE_METHODS:
        raise ExperimentGateDenied("timeseries_candidate_requires_timeseries_profile")
    if rul_policy or candidate.method in RUL_EVALUATION_CANDIDATE_METHODS:
        raise ExperimentGateDenied("rul_candidate_requires_rul_profile")
    if ppo_safety_policy or candidate.method in PPO_RESEARCH_EVALUATION_CANDIDATE_METHODS:
        raise ExperimentGateDenied("ppo_candidate_requires_research_safety_profile")
    edge_candidate = (
        candidate.method == "QUANTIZATION"
        and candidate.training_config.get("target_runtime") == "LLAMA_CPP"
    )
    if edge_candidate:
        if plan.target_profile not in {"EDGE_MODEL_COMPONENT", "AGENT_RUNTIME"}:
            raise ExperimentGateDenied("GGUF_quantization_requires_edge_runtime_evaluation")
        if not _is_edge_runner_config(plan.runner_config):
            raise ExperimentGateDenied("GGUF_evaluation_requires_llama_cpp_runner_config")
        return
    if plan.target_profile == "EDGE_MODEL_COMPONENT":
        raise ExperimentGateDenied("edge_profile_requires_GGUF_quantization_candidate")
    if _is_edge_runner_config(plan.runner_config):
        raise ExperimentGateDenied("edge_runner_config_requires_GGUF_quantization_candidate")


def _validate_timeseries_runner_config(config: dict[str, Any]) -> None:
    required_keys = {
        "anomaly_threshold",
        "rule_threshold",
        "precision",
        "gpu_hourly_cost_usd",
        "framework_mode",
    }
    if set(config) != required_keys:
        raise ValueError("time-series evaluation runner configuration is incomplete")
    anomaly_threshold = config["anomaly_threshold"]
    rule_threshold = config["rule_threshold"]
    if (
        isinstance(anomaly_threshold, bool)
        or not isinstance(anomaly_threshold, (int, float))
        or not math.isfinite(float(anomaly_threshold))
        or float(anomaly_threshold) <= 0
    ):
        raise ValueError("time-series anomaly_threshold is invalid")
    if (
        isinstance(rule_threshold, bool)
        or not isinstance(rule_threshold, (int, float))
        or not math.isfinite(float(rule_threshold))
        or not 0 < float(rule_threshold) < 1
    ):
        raise ValueError("time-series rule_threshold is invalid")
    if config["precision"] not in {"bfloat16", "float16"}:
        raise ValueError("evaluation precision is unsupported")
    hourly_cost = config["gpu_hourly_cost_usd"]
    if (
        isinstance(hourly_cost, bool)
        or not isinstance(hourly_cost, (int, float))
        or not math.isfinite(float(hourly_cost))
        or float(hourly_cost) <= 0
    ):
        raise ValueError("gpu_hourly_cost_usd is invalid")
    if config["framework_mode"] != "DETERMINISTIC":
        raise ValueError("time-series evaluation requires deterministic framework mode")
    _digest_json(config)


def _validate_rul_runner_config(config: dict[str, Any]) -> None:
    required_keys = {"precision", "gpu_hourly_cost_usd", "framework_mode"}
    if set(config) != required_keys:
        raise ValueError("RUL evaluation runner configuration is incomplete")
    if config["precision"] not in {"bfloat16", "float16"}:
        raise ValueError("evaluation precision is unsupported")
    hourly_cost = config["gpu_hourly_cost_usd"]
    if (
        isinstance(hourly_cost, bool)
        or not isinstance(hourly_cost, (int, float))
        or not math.isfinite(float(hourly_cost))
        or float(hourly_cost) <= 0
    ):
        raise ValueError("gpu_hourly_cost_usd is invalid")
    if config["framework_mode"] != "DETERMINISTIC":
        raise ValueError("RUL evaluation requires deterministic framework mode")
    _digest_json(config)


def _validate_retrieval_runner_config(config: dict[str, Any]) -> None:
    required_keys = {
        "top_k",
        "precision",
        "gpu_hourly_cost_usd",
        "framework_mode",
    }
    if set(config) != required_keys:
        raise ValueError("retrieval evaluation runner configuration is incomplete")
    top_k = config["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise ValueError("retrieval top_k is out of range")
    if config["precision"] not in {"bfloat16", "float16"}:
        raise ValueError("evaluation precision is unsupported")
    hourly_cost = config["gpu_hourly_cost_usd"]
    if (
        isinstance(hourly_cost, bool)
        or not isinstance(hourly_cost, (int, float))
        or not math.isfinite(float(hourly_cost))
        or float(hourly_cost) <= 0
    ):
        raise ValueError("gpu_hourly_cost_usd is invalid")
    if config["framework_mode"] != "DETERMINISTIC":
        raise ValueError("retrieval evaluation requires deterministic framework mode")
    _digest_json(config)


def _validate_vlm_runner_config(config: dict[str, Any]) -> None:
    required_keys = {
        "max_new_tokens",
        "max_image_pixels",
        "precision",
        "gpu_hourly_cost_usd",
        "framework_mode",
    }
    if set(config) != required_keys:
        raise ValueError("VLM evaluation runner configuration is incomplete")
    max_new_tokens = config["max_new_tokens"]
    max_image_pixels = config["max_image_pixels"]
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= 2048
    ):
        raise ValueError("max_new_tokens is out of range")
    if (
        isinstance(max_image_pixels, bool)
        or not isinstance(max_image_pixels, int)
        or not 65_536 <= max_image_pixels <= 16_777_216
    ):
        raise ValueError("VLM max_image_pixels is out of range")
    if config["precision"] not in {"bfloat16", "float16"}:
        raise ValueError("evaluation precision is unsupported")
    hourly_cost = config["gpu_hourly_cost_usd"]
    if (
        isinstance(hourly_cost, bool)
        or not isinstance(hourly_cost, (int, float))
        or not math.isfinite(float(hourly_cost))
        or float(hourly_cost) <= 0
    ):
        raise ValueError("gpu_hourly_cost_usd is invalid")
    if config["framework_mode"] != "DETERMINISTIC":
        raise ValueError("VLM evaluation requires deterministic framework mode")
    _digest_json(config)


def _validate_asr_runner_config(config: dict[str, Any]) -> None:
    required_keys = {
        "max_new_tokens",
        "sampling_rate",
        "language",
        "max_audio_seconds",
        "precision",
        "gpu_hourly_cost_usd",
        "framework_mode",
    }
    if set(config) != required_keys:
        raise ValueError("ASR evaluation runner configuration is incomplete")
    max_new_tokens = config["max_new_tokens"]
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= 448
    ):
        raise ValueError("ASR max_new_tokens is out of range")
    sampling_rate = config["sampling_rate"]
    if isinstance(sampling_rate, bool) or sampling_rate != 16_000:
        raise ValueError("ASR sampling_rate must be 16000")
    language = config["language"]
    if not isinstance(language, str) or not language.strip() or len(language) > 32:
        raise ValueError("ASR language is invalid")
    max_audio_seconds = config["max_audio_seconds"]
    if (
        isinstance(max_audio_seconds, bool)
        or not isinstance(max_audio_seconds, (int, float))
        or not math.isfinite(float(max_audio_seconds))
        or not 0.1 <= float(max_audio_seconds) <= 30
    ):
        raise ValueError("ASR max_audio_seconds is out of range")
    if config["precision"] not in {"bfloat16", "float16"}:
        raise ValueError("evaluation precision is unsupported")
    hourly_cost = config["gpu_hourly_cost_usd"]
    if (
        isinstance(hourly_cost, bool)
        or not isinstance(hourly_cost, (int, float))
        or not math.isfinite(float(hourly_cost))
        or float(hourly_cost) <= 0
    ):
        raise ValueError("gpu_hourly_cost_usd is invalid")
    if config["framework_mode"] != "DETERMINISTIC":
        raise ValueError("ASR evaluation requires deterministic framework mode")
    _digest_json(config)


def _validate_tts_runner_config(config: dict[str, Any]) -> None:
    required_keys = {
        "sampling_rate",
        "language",
        "max_audio_seconds",
        "precision",
        "gpu_hourly_cost_usd",
        "framework_mode",
        "asr_verifier_model_id",
        "asr_verifier_revision",
        "asr_verifier_digest",
    }
    if set(config) != required_keys:
        raise ValueError("TTS evaluation runner configuration is incomplete")
    if isinstance(config["sampling_rate"], bool) or config["sampling_rate"] != 16_000:
        raise ValueError("TTS sampling_rate must be 16000")
    language = config["language"]
    if not isinstance(language, str) or not language.strip() or len(language) > 32:
        raise ValueError("TTS language is invalid")
    max_audio_seconds = config["max_audio_seconds"]
    if (
        isinstance(max_audio_seconds, bool)
        or not isinstance(max_audio_seconds, (int, float))
        or not math.isfinite(float(max_audio_seconds))
        or not 0.1 <= float(max_audio_seconds) <= 30
    ):
        raise ValueError("TTS max_audio_seconds is out of range")
    if config["precision"] not in {"bfloat16", "float16"}:
        raise ValueError("evaluation precision is unsupported")
    hourly_cost = config["gpu_hourly_cost_usd"]
    if (
        isinstance(hourly_cost, bool)
        or not isinstance(hourly_cost, (int, float))
        or not math.isfinite(float(hourly_cost))
        or float(hourly_cost) <= 0
    ):
        raise ValueError("gpu_hourly_cost_usd is invalid")
    if config["framework_mode"] != "DETERMINISTIC":
        raise ValueError("TTS evaluation requires deterministic framework mode")
    model_id = config["asr_verifier_model_id"]
    revision = config["asr_verifier_revision"]
    digest = config["asr_verifier_digest"]
    if not isinstance(model_id, str) or not model_id.strip() or len(model_id) > 255:
        raise ValueError("TTS ASR verifier model is invalid")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or digest != f"hf-revision:{revision}"
    ):
        raise ValueError("TTS ASR verifier revision binding is invalid")
    _digest_json(config)


def _evaluate(
    policy: EvaluationPolicyRecord,
    suite: EvaluationSuiteRecord,
    evidence: EvaluationEvidence,
) -> dict[str, Any]:
    candidate_score = sum(evidence.candidate_outcomes) / len(evidence.candidate_outcomes)
    baseline_score = sum(evidence.baseline_outcomes) / len(evidence.baseline_outcomes)
    quality_delta = candidate_score - baseline_score
    ci_low, ci_high = _paired_bootstrap_interval(
        evidence.candidate_outcomes,
        evidence.baseline_outcomes,
        iterations=int(policy.thresholds["bootstrap_iterations"]),
        seed_material=f"{suite.manifest_hash}\0{policy.policy_hash}",
    )
    latency_improvement = (
        evidence.baseline_p95_latency_ms - evidence.candidate_p95_latency_ms
    ) / evidence.baseline_p95_latency_ms
    cost_improvement = (
        evidence.baseline_cost_per_case - evidence.candidate_cost_per_case
    ) / evidence.baseline_cost_per_case
    failed_gates = sorted(
        gate
        for gate, expected in policy.hard_gates.items()
        if evidence.hard_gate_results.get(gate) != expected
    )
    quality_gain = quality_delta >= policy.thresholds["quality_lift_min"] and ci_low > 0
    efficient_without_regression = (
        quality_delta >= -policy.thresholds["quality_tolerance"]
        and max(latency_improvement, cost_improvement)
        >= policy.thresholds["efficiency_improvement_min"]
    )
    high_risk_fix = evidence.high_risk_defect_resolved and not failed_gates
    if failed_gates:
        decision = "REJECTED"
        failure_reason = "hard_gate_failed:" + ",".join(failed_gates)
    elif suite.tier == "SMOKE":
        decision = "SMOKE_PASSED"
        failure_reason = None
    elif suite.tier == "SIMULATION_REFERENCE":
        decision = "SIMULATION_REFERENCE_PASSED"
        failure_reason = None
    elif quality_gain or efficient_without_regression or high_risk_fix:
        decision = "CANDIDATE"
        failure_reason = None
    else:
        decision = "NO_GAIN"
        failure_reason = "candidate_value_threshold_not_met"
    return {
        "decision": decision,
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "quality_delta": quality_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "latency_improvement": latency_improvement,
        "cost_improvement": cost_improvement,
        "failure_reason": failure_reason,
    }


def _paired_bootstrap_interval(
    candidate: list[float],
    baseline: list[float],
    *,
    iterations: int,
    seed_material: str,
) -> tuple[float, float]:
    deltas = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
    ]
    rng = random.Random(int(sha256(seed_material.encode()).hexdigest()[:16], 16))
    sample_count = len(deltas)
    estimates = sorted(
        sum(deltas[rng.randrange(sample_count)] for _ in range(sample_count)) / sample_count
        for _ in range(iterations)
    )
    low_index = max(0, int(iterations * 0.025) - 1)
    high_index = min(iterations - 1, int(iterations * 0.975))
    return estimates[low_index], estimates[high_index]


def _tracking_payload(experiment: TrainingExperimentRecord) -> dict[str, Any]:
    return {
        "experiment_name": experiment.mlflow_experiment_name,
        "run_name": experiment.experiment_id,
        "params": {
            "ioap.method": experiment.method,
            "ioap.task_type": experiment.task_type,
            "ioap.dataset_snapshot_id": experiment.dataset_snapshot_id,
            "ioap.dataset_manifest_hash": experiment.dataset_manifest_hash,
            "ioap.base_model_digest": experiment.base_model_digest,
            "ioap.tokenizer_digest": experiment.tokenizer_digest,
            "ioap.chat_template_digest": experiment.chat_template_digest,
            "ioap.git_commit": experiment.git_commit,
            "ioap.container_digest": experiment.container_digest,
            "ioap.config_hash": experiment.config_hash,
            "ioap.random_seeds": json.dumps(experiment.random_seeds, separators=(",", ":")),
        },
        "tags": {
            "ioap.tenant_id": experiment.tenant_id,
            "ioap.experiment_id": experiment.experiment_id,
            "ioap.comparison_group_id": experiment.comparison_group_id,
            "ioap.governed": "true",
            "mlflow.source.git.commit": experiment.git_commit,
        },
    }


def _plan_digest(plan: ExperimentPlan) -> str:
    return _digest_json(
        {
            "comparison_group_id": plan.comparison_group_id,
            "method": plan.method,
            "task_type": plan.task_type,
            "dataset_snapshot_id": plan.dataset_snapshot_id,
            "base_model_id": plan.base_model_id,
            "base_model_digest": plan.base_model_digest,
            "tokenizer_digest": plan.tokenizer_digest,
            "chat_template_digest": plan.chat_template_digest,
            "git_commit": plan.git_commit,
            "container_digest": plan.container_digest,
            "training_config": plan.training_config,
            "distributed_profile": plan.distributed_profile,
            "random_seeds": plan.random_seeds,
            "hardware_topology": plan.hardware_topology,
            "mlflow_experiment_name": plan.mlflow_experiment_name,
            "license_status": plan.license_status,
        }
    )


def _record_plan_digest(record: TrainingExperimentRecord) -> str:
    return _plan_digest(
        ExperimentPlan(
            comparison_group_id=record.comparison_group_id,
            method=_stored_training_method(record.method),
            task_type=record.task_type,
            dataset_snapshot_id=record.dataset_snapshot_id,
            base_model_id=record.base_model_id,
            base_model_digest=record.base_model_digest,
            tokenizer_digest=record.tokenizer_digest,
            chat_template_digest=record.chat_template_digest,
            git_commit=record.git_commit,
            container_digest=record.container_digest,
            training_config=record.training_config,
            distributed_profile=record.distributed_profile,
            random_seeds=record.random_seeds,
            hardware_topology=record.hardware_topology,
            mlflow_experiment_name=record.mlflow_experiment_name,
            license_status=record.license_status,
        )
    )


def _suite_digest(record: EvaluationSuiteRecord) -> str:
    return _digest_json(
        {
            "name": record.name,
            "version": record.version,
            "tier": record.tier,
            "source_snapshot_id": record.source_snapshot_id,
            "manifest_hash": record.manifest_hash,
            "sample_count": record.sample_count,
            "slice_counts": record.slice_counts,
        }
    )


def _evaluation_evidence_digest(evidence: EvaluationEvidence) -> str:
    return _digest_json(
        {
            "candidate_outcomes": evidence.candidate_outcomes,
            "baseline_outcomes": evidence.baseline_outcomes,
            "candidate_p95_latency_ms": evidence.candidate_p95_latency_ms,
            "baseline_p95_latency_ms": evidence.baseline_p95_latency_ms,
            "candidate_cost_per_case": evidence.candidate_cost_per_case,
            "baseline_cost_per_case": evidence.baseline_cost_per_case,
            "hard_gate_results": evidence.hard_gate_results,
            "slice_metrics": evidence.slice_metrics,
            "high_risk_defect_resolved": evidence.high_risk_defect_resolved,
        }
    )


def _evaluation_job_plan_digest(plan: EvaluationJobPlan) -> str:
    return _digest_json(
        {
            "candidate_experiment_id": plan.candidate_experiment_id,
            "baseline_experiment_id": plan.baseline_experiment_id,
            "suite_id": plan.suite_id,
            "policy_id": plan.policy_id,
            "target_profile": plan.target_profile,
            "runner_git_commit": plan.runner_git_commit,
            "container_digest": plan.container_digest,
            "runner_config": plan.runner_config,
        }
    )


def _evaluation_job_digest(record: ModelEvaluationJobRecord) -> str:
    return _evaluation_job_plan_digest(
        EvaluationJobPlan(
            candidate_experiment_id=record.candidate_experiment_id,
            baseline_experiment_id=record.baseline_experiment_id,
            suite_id=record.suite_id,
            policy_id=record.policy_id,
            target_profile=_stored_evaluation_target_profile(record.target_profile),
            runner_git_commit=record.runner_git_commit,
            container_digest=record.container_digest,
            runner_config=record.runner_config,
        )
    )


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode()).hexdigest()


def _stored_training_method(value: str) -> TrainingMethod:
    if value not in ALLOWED_METHODS:
        raise ValueError("stored training method is unsupported")
    return cast(TrainingMethod, value)


def _stored_evaluation_target_profile(value: str) -> EvaluationTargetProfile:
    if value not in ALLOWED_EVALUATION_TARGET_PROFILES:
        raise ValueError("stored evaluation target profile is unsupported")
    return cast(EvaluationTargetProfile, value)
