"""Explicit STAGING-only admission of governed 14-case diagnosis SMOKE evidence.

This reads authoritative records only. It does not promote an evaluation to
CANDIDATE, rewrite scores, approve a release, or confer production eligibility.
"""

from __future__ import annotations

import math
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.experiments.service import MANDATORY_HARD_GATES
from industrial_ops_agent.model_methods import PRODUCTION_RELEASE_METHODS
from industrial_ops_agent.persistence.model_import_integrity import enterprise_model_import_hash
from industrial_ops_agent.persistence.models import (
    DatasetSnapshotRecord,
    EnterpriseModelAssetImportRecord,
    EvaluationArtifactRecord,
    EvaluationPolicyRecord,
    EvaluationSuiteRecord,
    ModelEvaluationJobRecord,
    ModelEvaluationRunRecord,
    ModelReleaseRecord,
    PromptBundleRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.prompting import DEFAULT_PROMPT_BUNDLE_ID, default_prompt_registry
from industrial_ops_agent.prompting.registry import PromptBundleNotDeployed

STAGING_DIAGNOSIS_SMOKE_14 = "STAGING_DIAGNOSIS_SMOKE_14"


class StagingSmokeAdmissionError(ValueError):
    pass


def _record(session: Session, tenant_id: str, model: Any, column: Any, value: str) -> Any:
    record = session.scalar(select(model).where(model.tenant_id == tenant_id, column == value))
    if record is None:
        raise StagingSmokeAdmissionError("staging_smoke_records_not_visible")
    return record


def _digest(value: str) -> str:
    return value.removeprefix("sha256:")


def build_staging_smoke_admission(
    session: Session,
    tenant_id: str,
    *,
    evaluation_id: str,
    prompt_bundle_id: str,
    target_environment: str,
) -> dict[str, Any]:
    if target_environment != "STAGING":
        raise StagingSmokeAdmissionError("staging_smoke_is_staging_only")
    evaluation = _record(
        session,
        tenant_id,
        ModelEvaluationRunRecord,
        ModelEvaluationRunRecord.evaluation_id,
        evaluation_id,
    )
    candidate = _record(
        session,
        tenant_id,
        TrainingExperimentRecord,
        TrainingExperimentRecord.experiment_id,
        evaluation.candidate_experiment_id,
    )
    baseline = _record(
        session,
        tenant_id,
        TrainingExperimentRecord,
        TrainingExperimentRecord.experiment_id,
        evaluation.baseline_experiment_id,
    )
    suite = _record(
        session,
        tenant_id,
        EvaluationSuiteRecord,
        EvaluationSuiteRecord.suite_id,
        evaluation.suite_id,
    )
    policy = _record(
        session,
        tenant_id,
        EvaluationPolicyRecord,
        EvaluationPolicyRecord.policy_id,
        evaluation.policy_id,
    )
    snapshot = _record(
        session,
        tenant_id,
        DatasetSnapshotRecord,
        DatasetSnapshotRecord.snapshot_id,
        suite.source_snapshot_id,
    )
    if (
        evaluation.status != "COMPLETED"
        or evaluation.decision != "SMOKE_PASSED"
        or evaluation.failure_reason is not None
        # The registered component runner uses diagnostic_exact_match. Keep its
        # metric and score intact; do not relabel it as task_success on import.
        or evaluation.primary_metric not in {"task_success", "diagnostic_exact_match"}
        or candidate.status != "COMPLETED"
        or candidate.task_type != "DIAGNOSIS"
        or candidate.method not in PRODUCTION_RELEASE_METHODS
        or candidate.method == "QUANTIZATION"
        or baseline.status != "COMPLETED"
        or baseline.method != "BASELINE"
        or baseline.task_type != candidate.task_type
        or any(
            getattr(candidate, key) != getattr(baseline, key)
            for key in (
                "comparison_group_id",
                "dataset_snapshot_id",
                "base_model_id",
                "base_model_digest",
                "tokenizer_digest",
                "chat_template_digest",
                "random_seeds",
            )
        )
    ):
        raise StagingSmokeAdmissionError("staging_smoke_diagnosis_evaluation_required")
    if (
        suite.tier != "SMOKE"
        or suite.sample_count != 14
        or suite.status != "FROZEN"
        or suite.purpose != "EVALUATION_ONLY"
        or suite.source_snapshot_id == candidate.dataset_snapshot_id
        or snapshot.lineage_status != "CONFIRMED"
        or snapshot.row_count != 14
        or snapshot.split_counts.get("test") != 14
        or any(value != 0 for key, value in snapshot.split_counts.items() if key != "test")
        or _digest(snapshot.manifest_hash) != _digest(suite.manifest_hash)
    ):
        raise StagingSmokeAdmissionError("staging_smoke_frozen_14_cases_required")
    if (
        policy.status != "ACTIVE"
        or policy.primary_metric != evaluation.primary_metric
        or not set(MANDATORY_HARD_GATES).issubset(evaluation.hard_gate_results)
        or not evaluation.hard_gate_results
        or any(value is not True for value in evaluation.hard_gate_results.values())
        or any(
            evaluation.hard_gate_results.get(key) != value
            for key, value in policy.hard_gates.items()
        )
    ):
        raise StagingSmokeAdmissionError("staging_smoke_hard_gates_failed")
    scores = (evaluation.candidate_score, evaluation.baseline_score)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
        raise StagingSmokeAdmissionError("staging_smoke_scores_invalid")
    if not all(
        math.isfinite(value)
        for value in (
            evaluation.quality_delta,
            evaluation.ci_low,
            evaluation.ci_high,
        )
    ):
        raise StagingSmokeAdmissionError("staging_smoke_scores_invalid")
    jobs = list(
        session.scalars(
            select(ModelEvaluationJobRecord).where(
                ModelEvaluationJobRecord.tenant_id == tenant_id,
                ModelEvaluationJobRecord.result_evaluation_id == evaluation.evaluation_id,
            )
        )
    )
    if len(jobs) != 1:
        raise StagingSmokeAdmissionError("staging_smoke_one_completed_job_required")
    job = jobs[0]
    if (
        job.status != "COMPLETED"
        or job.target_profile != "MODEL_COMPONENT"
        or job.failure_reason is not None
        or not job.config_hash
        or any(
            getattr(job, key) != getattr(evaluation, key)
            for key in (
                "candidate_experiment_id",
                "baseline_experiment_id",
                "suite_id",
                "policy_id",
            )
        )
    ):
        raise StagingSmokeAdmissionError("staging_smoke_component_job_binding_invalid")
    try:
        definition = default_prompt_registry().get(prompt_bundle_id)
    except PromptBundleNotDeployed as exc:
        raise StagingSmokeAdmissionError("staging_smoke_prompt_not_deployed") from exc
    if (
        definition.task_type != "DIAGNOSIS"
        or job.runner_config.get("prompt_bundle_id") != prompt_bundle_id
        or job.runner_config.get("prompt_content_hash") != definition.content_hash
    ):
        raise StagingSmokeAdmissionError("staging_smoke_prompt_binding_mismatch")
    evidence = list(
        session.scalars(
            select(EvaluationArtifactRecord).where(
                EvaluationArtifactRecord.tenant_id == tenant_id,
                EvaluationArtifactRecord.evaluation_id == evaluation.evaluation_id,
                EvaluationArtifactRecord.kind == "case_evidence",
            )
        )
    )
    adapters = list(
        session.scalars(
            select(TrainingArtifactRecord).where(
                TrainingArtifactRecord.tenant_id == tenant_id,
                TrainingArtifactRecord.experiment_id == candidate.experiment_id,
                TrainingArtifactRecord.kind == "adapter_bundle",
            )
        )
    )
    if len(evidence) != 1 or len(adapters) != 1:
        raise StagingSmokeAdmissionError("staging_smoke_evidence_or_adapter_missing")
    artifact, adapter = evidence[0], adapters[0]
    metadata = artifact.metadata_json
    if (
        artifact.size_bytes <= 0
        or not artifact.content_hash
        or not evaluation.report_hash
        or adapter.size_bytes <= 0
        or not adapter.content_hash
        or metadata.get("research_only") is not False
        or metadata.get("candidate_method") != candidate.method
        or metadata.get("candidate_artifact_hash") != adapter.content_hash
        or metadata.get("suite_manifest_hash") != _digest(suite.manifest_hash)
        or metadata.get("policy_hash") != policy.policy_hash
        or metadata.get("runner_config_hash") != job.config_hash
        or metadata.get("prompt_bundle_id") != prompt_bundle_id
        or metadata.get("prompt_content_hash") != definition.content_hash
    ):
        raise StagingSmokeAdmissionError("staging_smoke_evidence_binding_invalid")
    transfer = None
    import_id = evaluation.comparison_context.get("staging_smoke_import_id")
    if import_id is not None:
        receipt = _record(
            session,
            tenant_id,
            EnterpriseModelAssetImportRecord,
            EnterpriseModelAssetImportRecord.import_id,
            import_id,
        )
        if (
            receipt.status != "ACTIVE"
            or receipt.import_hash != enterprise_model_import_hash(receipt)
            or receipt.operational_classification != "STAGING_DIAGNOSIS_SMOKE_TRANSFER"
            or receipt.release_scope != "STAGING_ONLY"
            or receipt.actual_sample_count != 14
            or receipt.source_decision != evaluation.decision
            or receipt.imported_candidate_experiment_id != candidate.experiment_id
            or receipt.imported_evaluation_id != evaluation.evaluation_id
            or receipt.imported_suite_id != suite.suite_id
        ):
            raise StagingSmokeAdmissionError("staging_smoke_transfer_invalid")
        transfer = {
            "import_id": receipt.import_id,
            "import_hash": receipt.import_hash,
            "source_tenant_id": receipt.imported_records_json["source_tenant_id"],
            "source_records_hash": receipt.source_evidence_chain_sha256,
            "training_authorization_inherited": False,
        }
        source_scores = receipt.imported_records_json.get("source_scores", {})
        if any(
            source_scores.get(key) != getattr(evaluation, key)
            for key in (
                "candidate_score",
                "baseline_score",
                "quality_delta",
                "ci_low",
                "ci_high",
                "hard_gate_results",
                "slice_metrics",
                "report_hash",
            )
        ):
            raise StagingSmokeAdmissionError("staging_smoke_import_scores_changed")
    result = {
        "mode": STAGING_DIAGNOSIS_SMOKE_14,
        "schema_version": 1,
        "tenant_id": tenant_id,
        "target_environment": "STAGING",
        "release_scope": "STAGING_ONLY",
        "production_eligible": False,
        "quality_status": "LIMITED_REGRESSION_ONLY",
        "independent_approval_required": True,
        "evaluation_id": evaluation.evaluation_id,
        "decision": evaluation.decision,
        "report_hash": evaluation.report_hash,
        "primary_metric": evaluation.primary_metric,
        "candidate_score": evaluation.candidate_score,
        "baseline_score": evaluation.baseline_score,
        "quality_delta": evaluation.quality_delta,
        "ci_low": evaluation.ci_low,
        "ci_high": evaluation.ci_high,
        "hard_gate_results": dict(evaluation.hard_gate_results),
        "slice_metrics": evaluation.slice_metrics,
        "suite_id": suite.suite_id,
        "suite_tier": suite.tier,
        "suite_version": suite.version,
        "sample_count": suite.sample_count,
        "snapshot_id": snapshot.snapshot_id,
        "suite_manifest_hash": suite.manifest_hash,
        "policy_id": policy.policy_id,
        "policy_hash": policy.policy_hash,
        "candidate_experiment_id": candidate.experiment_id,
        "candidate_config_hash": candidate.config_hash,
        "baseline_experiment_id": baseline.experiment_id,
        "baseline_config_hash": baseline.config_hash,
        "adapter_artifact_id": adapter.artifact_id,
        "adapter_content_hash": adapter.content_hash,
        "job_id": job.job_id,
        "runner_config_hash": job.config_hash,
        "target_profile": job.target_profile,
        "agent_runtime_evaluated": False,
        "prompt_bundle_id": definition.prompt_bundle_id,
        "prompt_content_hash": definition.content_hash,
        "case_evidence_id": artifact.artifact_id,
        "case_evidence_hash": artifact.content_hash,
    }
    if transfer is not None:
        result["transfer"] = transfer
    return result


def require_staging_smoke_admission_current(session: Session, release: ModelReleaseRecord) -> None:
    """Rebuild the admission before approval/deployment; never trust caller-provided scores."""
    marker = release.manifest_json.get("evaluation_admission")
    if not isinstance(marker, dict) or marker.get("mode") != STAGING_DIAGNOSIS_SMOKE_14:
        raise StagingSmokeAdmissionError("staging_smoke_admission_invalid")
    expected = build_staging_smoke_admission(
        session,
        release.tenant_id,
        evaluation_id=release.evaluation_id,
        prompt_bundle_id=release.manifest_json["prompt_bundle_id"],
        target_environment=release.target_environment,
    )
    if (
        release.manifest_json.get("target_environment") != "STAGING"
        or release.candidate_experiment_id != expected["candidate_experiment_id"]
        or marker != expected
    ):
        raise StagingSmokeAdmissionError("staging_smoke_admission_changed")
    from industrial_ops_agent.releases.local_adapter import require_local_adapter_current

    require_local_adapter_current(session, release, expected)


def require_publishable_staging_admission(
    session: Session,
    release: ModelReleaseRecord,
    *,
    deployment_provider: str | None = None,
) -> None:
    """Recheck special admission at publication and actual serving boundaries."""
    manifest = release.manifest_json
    if (
        not any(
            key in manifest
            for key in ("evaluation_admission", "prompt_evaluation_only", "local_staging_adapter")
        )
        and manifest.get("supply_chain", {}).get("kind") != "LOCAL_PINNED"
    ):
        return  # Historical STANDARD manifests have no staging-admission marker.
    require_staging_smoke_admission_current(session, release)
    if manifest.get("prompt_evaluation_only") is not False:
        raise StagingSmokeAdmissionError("evaluation_only_not_publishable")
    prompt = session.scalar(
        select(PromptBundleRecord).where(
            PromptBundleRecord.tenant_id == release.tenant_id,
            PromptBundleRecord.prompt_bundle_id == manifest["prompt_bundle_id"],
        )
    )
    binding = manifest.get("prompt_bundle", {})
    if (
        binding.get("id") != manifest["prompt_bundle_id"]
        or binding.get("content_hash") != manifest["evaluation_admission"]["prompt_content_hash"]
        or (
            prompt is None
            and (
                manifest["prompt_bundle_id"] != DEFAULT_PROMPT_BUNDLE_ID
                or binding.get("governance_status") != "LEGACY_BUILTIN"
            )
        )
        or (
            prompt is not None
            and (
                prompt.status != "APPROVED"
                or binding.get("governance_status") != "APPROVED"
                or prompt.content_hash != binding.get("content_hash")
            )
        )
    ):
        raise StagingSmokeAdmissionError("staging_prompt_approval_changed")
    if (
        manifest.get("local_staging_adapter")
        and deployment_provider is not None
        and deployment_provider != "COMPOSE_VLLM_GATEWAY"
    ):
        raise StagingSmokeAdmissionError("local_adapter_requires_compose_gateway")
