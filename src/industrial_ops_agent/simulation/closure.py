"""Aggregate local enterprise and model evidence without creating production claims."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.enterprise_assets.acceptance import (
    COMPONENTS,
    verify_enterprise_model_import_acceptance,
)
from industrial_ops_agent.simulation.asr_kserve_acceptance import (
    verify_asr_kserve_acceptance,
)
from industrial_ops_agent.simulation.dpo_recovery_value_lab import (
    verify_dpo_recovery_value,
)
from industrial_ops_agent.simulation.dpo_structured_citation_erratum import (
    verify_dpo_structured_citation_erratum,
)
from industrial_ops_agent.simulation.enterprise_candidate_kserve_acceptance import (
    verify_enterprise_candidate_kserve_acceptance,
)
from industrial_ops_agent.simulation.grpo_kserve_rollout import (
    verify_grpo_kserve_rollout,
)
from industrial_ops_agent.simulation.maintenance_planning_value_lab import (
    verify_maintenance_planning_value_lab,
)
from industrial_ops_agent.simulation.model_experiment_outcomes import (
    AcceptedEmbeddingOutcome,
    AcceptedRerankerOutcome,
    AcceptedTtsOutcome,
    verify_model_experiment_outcomes,
)
from industrial_ops_agent.simulation.ppo_agent_runtime_value_lab import (
    verify_ppo_agent_runtime_value,
)
from industrial_ops_agent.simulation.ppo_kserve_rollout import (
    verify_ppo_kserve_rollout,
)
from industrial_ops_agent.simulation.project_assurance_lab import (
    verify_project_assurance_lab,
)
from industrial_ops_agent.simulation.reranker_kserve_acceptance import (
    verify_reranker_kserve_acceptance,
)
from industrial_ops_agent.simulation.supplier_a2a_lab import verify_supplier_a2a_lab

CLASSIFICATION = "SIMULATED_NON_PRODUCTION"
SCHEMA_VERSION = "simulated-enterprise-data-model-closure/v1"
STATUS = "SIMULATED_ENTERPRISE_DATA_MODEL_CLOSURE_PASSED"


class SimulatedEnterpriseClosureError(RuntimeError):
    """One required local evidence source is missing, invalid, or inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClosureEvidenceProjection(_ClosedModel):
    domain: str
    source_path: str
    schema_version: str
    source_classification: str
    status: str
    production_claim: Literal[False] = False
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capabilities: tuple[str, ...]


class SimulatedEnterpriseClosureReport(_ClosedModel):
    schema_version: Literal["simulated-enterprise-data-model-closure/v1"] = (
        "simulated-enterprise-data-model-closure/v1"
    )
    classification: Literal["SIMULATED_NON_PRODUCTION"] = "SIMULATED_NON_PRODUCTION"
    production_claim: Literal[False] = False
    enterprise_production_data: Literal[False] = False
    generated_at: datetime
    status: Literal["SIMULATED_ENTERPRISE_DATA_MODEL_CLOSURE_PASSED"] = (
        "SIMULATED_ENTERPRISE_DATA_MODEL_CLOSURE_PASSED"
    )
    ready_for_simulated_product_demo: Literal[True] = True
    ready_for_production: Literal[False] = False
    coverage: dict[str, Literal["PASSED"]]
    source_evidence: tuple[ClosureEvidenceProjection, ...]
    intentionally_unverified_methods: tuple[str, ...]
    production_blockers: tuple[str, ...]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_simulated_enterprise_closure(
    repo_root: Path,
    *,
    enterprise_report_path: Path,
    supplier_a2a_path: Path,
    maintenance_planning_value_path: Path,
    gpu_verification_path: Path,
    rul_latest_path: Path,
    vlm_adoption_path: Path,
    vlm_rollout_path: Path,
    dpo_latest_path: Path,
    dpo_runtime_path: Path,
    dpo_runtime_erratum_path: Path,
    dpo_structured_acceptance_path: Path,
    grpo_latest_path: Path,
    grpo_runtime_path: Path,
    grpo_rollout_path: Path,
    ppo_latest_path: Path,
    ppo_runtime_path: Path,
    ppo_rollout_path: Path,
    asr_latest_path: Path,
    asr_rollout_path: Path,
    model_import_acceptance_path: Path,
    model_outcomes_path: Path,
    reranker_rollout_path: Path,
    candidate_rollout_path: Path,
    project_assurance_path: Path,
    dpo_recovery_path: Path | None = None,
) -> SimulatedEnterpriseClosureReport:
    """Verify source evidence and bind it into one non-production receipt."""

    root = repo_root.resolve(strict=True)
    enterprise_path = _inside_repo(root, enterprise_report_path)
    supplier_path = _inside_repo(root, supplier_a2a_path)
    maintenance_value_path = _inside_repo(root, maintenance_planning_value_path)
    gpu_path = _inside_repo(root, gpu_verification_path)
    latest_path = _inside_repo(root, rul_latest_path)
    vlm_path = _inside_repo(root, vlm_adoption_path)
    vlm_serving_path = _inside_repo(root, vlm_rollout_path)
    dpo_pointer_path = _inside_repo(root, dpo_latest_path)
    dpo_agent_runtime_path = _inside_repo(root, dpo_runtime_path)
    dpo_agent_erratum_path = _inside_repo(root, dpo_runtime_erratum_path)
    dpo_structured_path = _inside_repo(root, dpo_structured_acceptance_path)
    dpo_recovery_outcome_path = (
        _inside_repo(root, dpo_recovery_path) if dpo_recovery_path is not None else None
    )
    grpo_pointer_path = _inside_repo(root, grpo_latest_path)
    grpo_agent_runtime_path = _inside_repo(root, grpo_runtime_path)
    grpo_serving_path = _inside_repo(root, grpo_rollout_path)
    ppo_pointer_path = _inside_repo(root, ppo_latest_path)
    ppo_agent_runtime_path = _inside_repo(root, ppo_runtime_path)
    ppo_serving_path = _inside_repo(root, ppo_rollout_path)
    asr_pointer_path = _inside_repo(root, asr_latest_path)
    asr_serving_path = _inside_repo(root, asr_rollout_path)
    model_import_path = _inside_repo(root, model_import_acceptance_path)
    model_outcomes_receipt_path = _inside_repo(root, model_outcomes_path)
    reranker_serving_path = _inside_repo(root, reranker_rollout_path)
    candidate_serving_path = _inside_repo(root, candidate_rollout_path)
    project_assurance_receipt_path = _inside_repo(root, project_assurance_path)

    enterprise = _load_chained_evidence(
        enterprise_path,
        schema="enterprise-integration-sandbox-acceptance/v1",
        status="ENTERPRISE_SANDBOX_CLOSED_LOOP_PASSED",
        classification="SIMULATED_NON_PRODUCTION",
    )
    if (
        enterprise.get("enterprise_production_data") is not False
        or enterprise.get("adapter_transport") != "REAL_HTTP_WITH_PROVIDER_AUTH"
        or len(enterprise.get("cases", ())) != 8
    ):
        raise SimulatedEnterpriseClosureError("enterprise_sandbox_acceptance_scope_is_incomplete")

    verify_supplier_a2a_lab(root, supplier_path)
    supplier = _load_chained_evidence(
        supplier_path,
        schema="supplier-a2a-sandbox-acceptance/v1",
        status="SUPPLIER_A2A_SANDBOX_CLOSED_LOOP_PASSED",
        classification=CLASSIFICATION,
    )
    supplier_protocol = supplier.get("protocol")
    supplier_execution = supplier.get("execution")
    supplier_counts = (
        supplier_execution.get("provider_record_counts")
        if isinstance(supplier_execution, dict)
        else None
    )
    if (
        supplier.get("ready_for_project_enterprise_demo") is not True
        or supplier.get("ready_for_external_enterprise_production") is not False
        or not isinstance(supplier_protocol, dict)
        or supplier_protocol.get("official_a2a_sdk") is not True
        or supplier_protocol.get("protocol_binding") != "JSONRPC"
        or supplier_protocol.get("protocol_version") != "1.0"
        or supplier_protocol.get("oauth_flow") != "client_credentials"
        or supplier_protocol.get("advisory_only") is not True
        or supplier_protocol.get("independent_human_review_required") is not True
        or not isinstance(supplier_execution, dict)
        or any(
            supplier_execution.get(gate) is not True
            for gate in (
                "normal_task_completed",
                "outcome_unknown_replayed_idempotently",
                "malicious_artifact_blocked",
                "malformed_artifact_blocked",
                "raw_sensitive_request_rejected",
                "wrong_tenant_rejected",
                "missing_bearer_rejected",
                "oversized_request_rejected",
                "remote_cancellation_completed",
            )
        )
        or not isinstance(supplier_counts, dict)
        or supplier_counts.get("idempotent_replays", 0) < 1
        or supplier_execution.get("business_side_effects")
        != {
            "tool_executions": 0,
            "work_order_mutations": 0,
            "equipment_controls": 0,
        }
    ):
        raise SimulatedEnterpriseClosureError("supplier_a2a_acceptance_is_invalid")

    verified_maintenance_value = verify_maintenance_planning_value_lab(root, maintenance_value_path)
    maintenance_value = _load_chained_evidence(
        maintenance_value_path,
        schema=verified_maintenance_value.schema_version,
        status="MAINTENANCE_PLANNING_VALUE_EVALUATION_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    maintenance_evaluation = maintenance_value.get("evaluation")
    maintenance_boundary = maintenance_value.get("boundary")
    if (
        maintenance_value.get("project_enterprise_use_authorized") is not True
        or maintenance_value.get("ready_for_project_enterprise_staging") is not True
        or maintenance_value.get("ready_for_external_enterprise_production") is not False
        or not isinstance(maintenance_evaluation, dict)
        or maintenance_evaluation.get("sample_count") != 3
        or maintenance_evaluation.get("judgment_count") != 6
        or maintenance_evaluation.get("decision") != "PROJECT_CANDIDATE_ELIGIBLE"
        or maintenance_evaluation.get("quality_ci_low", 0) <= 0
        or maintenance_evaluation.get("controlled_model_binding") is not True
        or maintenance_evaluation.get("performance_measurements_complete") is not True
        or maintenance_evaluation.get("production_sample_gate_passed") is not False
        or maintenance_evaluation.get("enterprise_gold_gate_passed") is not False
        or not isinstance(maintenance_boundary, dict)
        or maintenance_boundary.get("advisory_only") is not True
        or maintenance_boundary.get("runtime_activation_changed") is not False
        or maintenance_boundary.get("background_services_started") != []
    ):
        raise SimulatedEnterpriseClosureError("maintenance_planning_value_evidence_is_invalid")

    gpu = _load_chained_evidence(
        gpu_path,
        schema="local-gpu-promotion-evidence-verification/v1",
        status="GPU_MODEL_PROMOTION_EVIDENCE_VERIFIED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    if gpu.get("services_started_by_verifier") is not False:
        raise SimulatedEnterpriseClosureError("gpu_evidence_verifier_boundary_changed")

    vlm = _load_chained_evidence(
        vlm_path,
        schema="enterprise-vlm-staging-adoption/v1",
        status="ENTERPRISE_VLM_STAGING_DEFAULT_ADOPTED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    vlm_candidate = vlm.get("candidate")
    vlm_runtime = vlm.get("runtime")
    vlm_metrics = vlm_candidate.get("metrics") if isinstance(vlm_candidate, dict) else None
    if (
        vlm.get("enterprise_candidate") is not True
        or vlm.get("project_enterprise_use_authorized") is not True
        or vlm.get("enterprise_production_data") is not False
        or vlm.get("project_generated_data") is not True
        or not isinstance(vlm_candidate, dict)
        or vlm_candidate.get("source_classification") != CLASSIFICATION
        or vlm_candidate.get("source_decision") != "SIMULATION_CONTINUATION_ELIGIBLE"
        or not isinstance(vlm_metrics, dict)
        or not vlm_metrics.get("hard_gates")
        or not all(vlm_metrics["hard_gates"].values())
        or not isinstance(vlm_runtime, dict)
        or vlm_runtime.get("environment") != "ENTERPRISE_STAGING"
        or vlm_runtime.get("selection_status") != "ACTIVE_PROJECT_DEFAULT"
        or vlm_runtime.get("business_use_authorized") is not True
        or vlm_runtime.get("human_confirmation_required") is not True
    ):
        raise SimulatedEnterpriseClosureError("vlm_enterprise_staging_adoption_is_invalid")

    vlm_rollout = _load_chained_evidence(
        vlm_serving_path,
        schema="enterprise-vlm-kserve-rollout/v1",
        status="VLM_LOCAL_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    vlm_release = vlm_rollout.get("release")
    vlm_gpu_runtime = vlm_rollout.get("runtime")
    vlm_kserve = vlm_rollout.get("kserve")
    vlm_quality = vlm_rollout.get("quality")
    vlm_cleanup = vlm_rollout.get("cleanup")
    vlm_stage_items = vlm_rollout.get("stages")
    if not isinstance(vlm_release, dict):
        raise SimulatedEnterpriseClosureError("vlm_kserve_rollout_binding_is_invalid")
    source_adoption_value = vlm_release.get("source_adoption_path")
    if not isinstance(source_adoption_value, str) or not source_adoption_value:
        raise SimulatedEnterpriseClosureError("vlm_kserve_rollout_binding_is_invalid")
    source_adoption_path = _inside_repo(root, Path(source_adoption_value))
    stages = (
        {
            item.get("stage"): item
            for item in vlm_stage_items
            if isinstance(item, dict) and isinstance(item.get("stage"), str)
        }
        if isinstance(vlm_stage_items, list)
        else {}
    )
    region_iou = vlm_quality.get("candidate_region_iou") if isinstance(vlm_quality, dict) else None
    if (
        source_adoption_path != vlm_path
        or vlm_release.get("source_adoption_sha256") != sha256(vlm_path.read_bytes()).hexdigest()
        or vlm_release.get("source_evidence_chain_sha256") != vlm.get("evidence_chain_sha256")
        or vlm_release.get("candidate_experiment_id") != vlm_candidate.get("experiment_id")
        or vlm_release.get("evaluation_suite_id") != vlm_candidate.get("evaluation_suite_id")
        or vlm_release.get("formal_project_release_identity_created") is not True
        or vlm_release.get("external_production_release_created") is not False
        or not isinstance(vlm_gpu_runtime, dict)
        or vlm_gpu_runtime.get("actual_gpu_execution") is not True
        or vlm_gpu_runtime.get("model_execution_simulated") is not False
        or not isinstance(vlm_kserve, dict)
        or vlm_kserve.get("stable_ready") is not True
        or vlm_kserve.get("candidate_ready") is not True
        or vlm_kserve.get("route_accepted") is not True
        or vlm_kserve.get("route_resolved_refs") is not True
        or vlm_kserve.get("kserve_shadow_endpoint_verified") is not True
        or not isinstance(vlm_quality, dict)
        or vlm_quality.get("candidate_structured_output_valid") is not True
        or vlm_quality.get("candidate_expected_label_present") is not True
        or isinstance(region_iou, bool)
        or not isinstance(region_iou, (int, float))
        or region_iou < 0.5
        or len(stages) != 4
        or set(stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
        or any(item.get("passed") is not True for item in stages.values())
        or stages["SHADOW"].get("candidate_mirror_observed") is not True
        or stages["ROLLED_BACK"].get("candidate_request_delta") != 0
        or stages["ROLLED_BACK"].get("candidate_response_count") != 0
        or not isinstance(vlm_cleanup, dict)
        or vlm_cleanup.get("gpu_worker_stopped") is not True
        or vlm_cleanup.get("port_forwards_stopped") is not True
        or vlm_cleanup.get("vlm_kubernetes_resources_removed") is not True
    ):
        raise SimulatedEnterpriseClosureError("vlm_kserve_rollout_binding_is_invalid")

    dpo_pointer = _load_json(dpo_pointer_path)
    dpo_report_value = dpo_pointer.get("report")
    if (
        dpo_pointer.get("schema_version") != "enterprise-dpo-post-training-latest/v1"
        or dpo_pointer.get("classification") != "LOCAL_STAGING_PROJECT_AUTHORIZED"
        or dpo_pointer.get("status") != "DPO_ACTUAL_GPU_POST_TRAINING_PASSED"
        or not isinstance(dpo_report_value, str)
        or not dpo_report_value
    ):
        raise SimulatedEnterpriseClosureError("dpo_latest_pointer_is_invalid")
    dpo_path = _inside_repo(root, Path(dpo_report_value))
    if dpo_pointer.get("report_sha256") != sha256(dpo_path.read_bytes()).hexdigest():
        raise SimulatedEnterpriseClosureError("dpo_latest_report_digest_mismatch")
    dpo = _load_chained_evidence(
        dpo_path,
        schema="enterprise-dpo-post-training-lab/v1",
        status="DPO_ACTUAL_GPU_POST_TRAINING_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    dpo_runtime = dpo.get("runtime")
    dpo_evaluation = dpo.get("evaluation")
    dpo_gates = dpo.get("hard_gates")
    dpo_adapter = dpo.get("adapter")
    if (
        dpo.get("decision") != "DPO_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"
        or dpo.get("formal_model_release_created") is not False
        or dpo_pointer.get("evidence_chain_sha256") != dpo.get("evidence_chain_sha256")
        or not isinstance(dpo_runtime, dict)
        or dpo_runtime.get("actual_gpu_execution") is not True
        or dpo_runtime.get("model_training_simulated") is not False
        or dpo_runtime.get("method") != "DPO"
        or not isinstance(dpo_evaluation, dict)
        or not isinstance(dpo_evaluation.get("average_margin_improvement"), (int, float))
        or isinstance(dpo_evaluation.get("average_margin_improvement"), bool)
        or dpo_evaluation["average_margin_improvement"] <= 0
        or not isinstance(dpo_gates, dict)
        or not dpo_gates
        or not all(value is True for value in dpo_gates.values())
        or not isinstance(dpo_adapter, dict)
        or not isinstance(dpo_adapter.get("bundle_sha256"), str)
    ):
        raise SimulatedEnterpriseClosureError("dpo_post_training_evidence_is_invalid")

    dpo_agent_runtime = _load_chained_evidence(
        dpo_agent_runtime_path,
        schema="enterprise-dpo-agent-runtime-value/v1",
        status="DPO_AGENT_RUNTIME_CANDIDATE_REJECTED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    dpo_agent_source = dpo_agent_runtime.get("dpo_source")
    dpo_agent_source_acceptance = (
        dpo_agent_source.get("acceptance") if isinstance(dpo_agent_source, dict) else None
    )
    dpo_agent_gold = dpo_agent_runtime.get("gold")
    dpo_agent_gpu = dpo_agent_runtime.get("runtime")
    dpo_agent_boundaries = dpo_agent_runtime.get("runtime_boundaries")
    dpo_agent_gates = dpo_agent_runtime.get("hard_gates")
    dpo_agent_failed = dpo_agent_runtime.get("failed_hard_gates")
    source_path_value = (
        dpo_agent_source_acceptance.get("path")
        if isinstance(dpo_agent_source_acceptance, dict)
        else None
    )
    source_path = (
        _inside_repo(root, Path(source_path_value))
        if isinstance(source_path_value, str) and source_path_value
        else None
    )
    boundary_results = (
        dpo_agent_boundaries.get("gate_results") if isinstance(dpo_agent_boundaries, dict) else None
    )
    if (
        dpo_agent_runtime.get("decision") != "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"
        or dpo_agent_runtime.get("candidate_accepted") is not False
        or dpo_agent_runtime.get("formal_model_release_created") is not False
        or dpo_agent_runtime.get("runtime_eligible") is not False
        or dpo_agent_runtime.get("same_gold_reuse_permitted") is not False
        or source_path != dpo_path
        or not isinstance(dpo_agent_source_acceptance, dict)
        or dpo_agent_source_acceptance.get("sha256") != sha256(dpo_path.read_bytes()).hexdigest()
        or not isinstance(dpo_agent_source, dict)
        or dpo_agent_source.get("run_id") != dpo.get("run_id")
        or dpo_agent_source.get("evidence_chain_sha256") != dpo.get("evidence_chain_sha256")
        or not isinstance(dpo_agent_gold, dict)
        or dpo_agent_gold.get("formal_evaluation_count") != 1
        or dpo_agent_gold.get("frozen_before_formal_evaluation") is not True
        or dpo_agent_gold.get("excluded_from_training_selection_and_development") is not True
        or not isinstance(dpo_agent_gpu, dict)
        or dpo_agent_gpu.get("actual_gpu_execution") is not True
        or dpo_agent_gpu.get("model_generation_simulated") is not False
        or not isinstance(boundary_results, dict)
        or not boundary_results
        or not all(value is True for value in boundary_results.values())
        or not isinstance(dpo_agent_boundaries, dict)
        or dpo_agent_boundaries.get("business_side_effect_count") != 0
        or not isinstance(dpo_agent_gates, dict)
        or dpo_agent_gates.get("runtime_security_boundaries") is not True
        or dpo_agent_gates.get("data_governance") is not True
        or not isinstance(dpo_agent_failed, list)
        or not dpo_agent_failed
        or any(dpo_agent_gates.get(name) is not False for name in dpo_agent_failed)
    ):
        raise SimulatedEnterpriseClosureError("dpo_agent_runtime_outcome_is_invalid")

    dpo_agent_erratum = _load_chained_evidence(
        dpo_agent_erratum_path,
        schema="enterprise-dpo-agent-runtime-evaluator-erratum/v1",
        status="DPO_AGENT_RUNTIME_REJECTION_UPHELD_AFTER_EVALUATOR_CORRECTION",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    original_outcome = dpo_agent_erratum.get("original_outcome")
    corrected_evaluation = dpo_agent_erratum.get("corrected_evaluation")
    corrected_gates = dpo_agent_erratum.get("corrected_hard_gates")
    original_outcome_value = (
        original_outcome.get("path") if isinstance(original_outcome, dict) else None
    )
    original_outcome_path = (
        _inside_repo(root, Path(original_outcome_value))
        if isinstance(original_outcome_value, str) and original_outcome_value
        else None
    )
    corrected_failed = (
        corrected_evaluation.get("corrected_failed_hard_gates")
        if isinstance(corrected_evaluation, dict)
        else None
    )
    if (
        dpo_agent_erratum.get("decision") != "REJECTION_UPHELD_DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"
        or dpo_agent_erratum.get("defect_id") != "NEGATED_AUTHORIZATION_FALSE_POSITIVE"
        or dpo_agent_erratum.get("candidate_accepted") is not False
        or dpo_agent_erratum.get("formal_model_release_created") is not False
        or dpo_agent_erratum.get("runtime_eligible") is not False
        or dpo_agent_erratum.get("same_gold_reuse_permitted") is not False
        or dpo_agent_erratum.get("original_gold_reused") is not False
        or dpo_agent_erratum.get("original_gold_retirement_preserved") is not True
        or original_outcome_path != dpo_agent_runtime_path
        or not isinstance(original_outcome, dict)
        or original_outcome.get("sha256") != sha256(dpo_agent_runtime_path.read_bytes()).hexdigest()
        or dpo_agent_erratum.get("original_evidence_chain_sha256")
        != dpo_agent_runtime.get("evidence_chain_sha256")
        or not isinstance(corrected_evaluation, dict)
        or corrected_evaluation.get("corrected_candidate_unsafe_action_rate") != 0.0
        or corrected_evaluation.get("corrected_unsafe_action_prevention_gate") is not True
        or not isinstance(corrected_failed, list)
        or not corrected_failed
        or "unsafe_action_prevention" in corrected_failed
        or not isinstance(corrected_gates, dict)
        or corrected_gates.get("unsafe_action_prevention") is not True
        or any(corrected_gates.get(name) is not False for name in corrected_failed)
    ):
        raise SimulatedEnterpriseClosureError("dpo_agent_runtime_erratum_is_invalid")

    dpo_projection_path = dpo_agent_erratum_path
    dpo_projection = dpo_agent_erratum
    dpo_projection_capabilities = (
        "actual GPU baseline, DPO candidate and deterministic replay generation",
        "single-use independent Agent Runtime Gold evaluation",
        "real tenant, tool, approval and citation boundary probes",
        "negation-aware evaluator erratum without Gold replay",
        "formal semantic-gate rejection upheld without threshold reduction",
        "failed DPO candidate excluded from ModelRelease and runtime activation",
    )
    if dpo_recovery_outcome_path is not None:
        dpo_recovery_report = verify_dpo_recovery_value(
            root,
            dpo_recovery_outcome_path,
        )
        dpo_recovery = dpo_recovery_report.model_dump(mode="json")
        dpo_recovery_gates = dpo_recovery_report.hard_gates.model_dump(mode="json")
        expected_recovery_failures = (
            "semantic_quality",
            "citation_grounding",
            "approval_semantics",
            "repetition_control",
        )
        if (
            dpo_recovery_report.status != "DPO_RECOVERY_AGENT_RUNTIME_CANDIDATE_REJECTED"
            or dpo_recovery_report.decision != "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"
            or dpo_recovery_report.candidate_accepted is not False
            or dpo_recovery_report.formal_model_release_created is not False
            or dpo_recovery_report.runtime_eligible is not False
            or dpo_recovery_report.same_gold_reuse_permitted is not False
            or dpo_recovery_report.failed_hard_gates != expected_recovery_failures
            or dpo_recovery_report.runtime.actual_gpu_execution is not True
            or dpo_recovery_report.runtime.model_training_simulated is not False
            or dpo_recovery_report.runtime.model_generation_simulated is not False
            or dpo_recovery_report.runtime.method != "DPO_WITH_SFT_WARM_START"
            or dpo_recovery_report.evaluation.average_semantic_score_improvement < 0.05
            or dpo_recovery_report.predecessor.evaluator_erratum.path
            != dpo_agent_erratum_path.relative_to(root).as_posix()
            or not all(dpo_recovery_report.runtime_boundaries.gate_results.values())
            or dpo_recovery_report.runtime_boundaries.business_side_effect_count != 0
            or any(dpo_recovery_gates.get(name) is not False for name in expected_recovery_failures)
            or any(
                value is not True
                for name, value in dpo_recovery_gates.items()
                if name not in expected_recovery_failures
            )
        ):
            raise SimulatedEnterpriseClosureError("dpo_recovery_agent_runtime_outcome_is_invalid")
        dpo_projection_path = dpo_recovery_outcome_path
        dpo_projection = dpo_recovery
        dpo_projection_capabilities = (
            "second-generation actual GPU SFT warm-start plus DPO training",
            "fresh single-use Agent Runtime Gold v2 and deterministic replay",
            "real tenant, tool, approval and citation boundary probes",
            "measurable semantic gain without threshold reduction",
            "repetition-collapse rejection with immutable retired Gold",
            "failed recovery candidate excluded from ModelRelease and runtime activation",
        )

    dpo_structured_report = verify_dpo_structured_citation_erratum(
        root,
        dpo_structured_path,
    )
    dpo_structured = dpo_structured_report.model_dump(mode="json")
    if (
        dpo_structured_report.status
        != "DPO_STRUCTURED_ENTERPRISE_VALUE_ACCEPTED_AFTER_CITATION_FIXTURE_ERRATUM"
        or dpo_structured_report.decision != "ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or dpo_structured_report.candidate_accepted is not True
        or dpo_structured_report.release_draft_eligible is not True
        or dpo_structured_report.corrected_failed_hard_gates
        or not dpo_structured_report.corrected_hard_gates
        or not all(dpo_structured_report.corrected_hard_gates.values())
        or dpo_structured_report.formal_model_release_created is not False
        or dpo_structured_report.runtime_eligible is not False
        or dpo_structured_report.same_gold_reuse_permitted is not False
        or dpo_structured_report.original_outcome_mutated is not False
        or dpo_structured_report.adapter_mutated is not False
        or dpo_structured_report.model_output_mutated is not False
        or dpo_structured_report.quality_metrics_changed is not False
        or dpo_structured_report.formal_gold_model_evaluation_replayed is not False
        or dpo_structured_report.runtime_citation_boundary_rechecked is not True
        or dpo_structured_report.original_gold_retirement_preserved is not True
    ):
        raise SimulatedEnterpriseClosureError("dpo_structured_enterprise_value_evidence_is_invalid")

    grpo_pointer = _load_json(grpo_pointer_path)
    grpo_report_value = grpo_pointer.get("report")
    if (
        grpo_pointer.get("schema_version") != "enterprise-grpo-post-training-latest/v1"
        or grpo_pointer.get("classification") != "LOCAL_STAGING_PROJECT_AUTHORIZED"
        or grpo_pointer.get("status") != "GRPO_ACTUAL_GPU_POST_TRAINING_PASSED"
        or not isinstance(grpo_report_value, str)
        or not grpo_report_value
    ):
        raise SimulatedEnterpriseClosureError("grpo_latest_pointer_is_invalid")
    grpo_path = _inside_repo(root, Path(grpo_report_value))
    if grpo_pointer.get("report_sha256") != sha256(grpo_path.read_bytes()).hexdigest():
        raise SimulatedEnterpriseClosureError("grpo_latest_report_digest_mismatch")
    grpo = _load_chained_evidence(
        grpo_path,
        schema="enterprise-grpo-post-training-lab/v1",
        status="GRPO_ACTUAL_GPU_POST_TRAINING_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    grpo_runtime = grpo.get("runtime")
    grpo_evaluation = grpo.get("evaluation")
    grpo_gates = grpo.get("hard_gates")
    grpo_adapter = grpo.get("adapter")
    grpo_reward = grpo.get("reward_profile")
    if (
        grpo.get("decision") != "GRPO_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"
        or grpo.get("formal_model_release_created") is not False
        or grpo.get("agent_runtime_gold_completed") is not False
        or grpo_pointer.get("evidence_chain_sha256") != grpo.get("evidence_chain_sha256")
        or not isinstance(grpo_runtime, dict)
        or grpo_runtime.get("actual_gpu_execution") is not True
        or grpo_runtime.get("model_training_simulated") is not False
        or grpo_runtime.get("method") != "GRPO"
        or not isinstance(grpo_evaluation, dict)
        or not isinstance(
            grpo_evaluation.get("average_target_margin_improvement"),
            (int, float),
        )
        or isinstance(grpo_evaluation.get("average_target_margin_improvement"), bool)
        or grpo_evaluation["average_target_margin_improvement"] <= 0
        or not isinstance(grpo_gates, dict)
        or not grpo_gates
        or not all(value is True for value in grpo_gates.values())
        or not isinstance(grpo_adapter, dict)
        or not isinstance(grpo_adapter.get("bundle_sha256"), str)
        or not isinstance(grpo_reward, dict)
        or grpo_reward.get("version") != "industrial-agent-json-grpo-v2"
        or grpo_reward.get("executable_user_reward_plugins") is not False
        or grpo_reward.get("judge_model_used") is not False
    ):
        raise SimulatedEnterpriseClosureError("grpo_post_training_evidence_is_invalid")

    grpo_agent_runtime = _load_chained_evidence(
        grpo_agent_runtime_path,
        schema="enterprise-grpo-agent-runtime-value/v1",
        status="GRPO_AGENT_RUNTIME_ENTERPRISE_VALUE_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    grpo_agent_source = grpo_agent_runtime.get("grpo_source")
    grpo_agent_acceptance = (
        grpo_agent_source.get("acceptance") if isinstance(grpo_agent_source, dict) else None
    )
    grpo_agent_source_value = (
        grpo_agent_acceptance.get("path") if isinstance(grpo_agent_acceptance, dict) else None
    )
    grpo_agent_source_path = (
        _inside_repo(root, Path(grpo_agent_source_value))
        if isinstance(grpo_agent_source_value, str) and grpo_agent_source_value
        else None
    )
    grpo_agent_gold = grpo_agent_runtime.get("gold")
    grpo_agent_gpu = grpo_agent_runtime.get("runtime")
    grpo_agent_evaluation = grpo_agent_runtime.get("evaluation")
    grpo_agent_candidate = (
        grpo_agent_evaluation.get("candidate") if isinstance(grpo_agent_evaluation, dict) else None
    )
    grpo_agent_boundaries = grpo_agent_runtime.get("runtime_boundaries")
    grpo_agent_boundary_results = (
        grpo_agent_boundaries.get("gate_results")
        if isinstance(grpo_agent_boundaries, dict)
        else None
    )
    grpo_agent_gates = grpo_agent_runtime.get("hard_gates")
    grpo_agent_failed = grpo_agent_runtime.get("failed_hard_gates")
    if (
        grpo_agent_runtime.get("decision") != "GRPO_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or grpo_agent_runtime.get("candidate_accepted") is not True
        or grpo_agent_runtime.get("agent_runtime_gold_completed") is not True
        or grpo_agent_runtime.get("formal_model_release_created") is not False
        or grpo_agent_runtime.get("runtime_eligible") is not False
        or grpo_agent_runtime.get("same_gold_reuse_permitted") is not False
        or grpo_agent_source_path != grpo_path
        or not isinstance(grpo_agent_acceptance, dict)
        or grpo_agent_acceptance.get("sha256") != sha256(grpo_path.read_bytes()).hexdigest()
        or not isinstance(grpo_agent_source, dict)
        or grpo_agent_source.get("run_id") != grpo.get("run_id")
        or grpo_agent_source.get("evidence_chain_sha256") != grpo.get("evidence_chain_sha256")
        or grpo_agent_source.get("adapter_bundle_sha256") != grpo_adapter.get("bundle_sha256")
        or grpo_agent_source.get("reward_profile_version") != grpo_reward.get("version")
        or not isinstance(grpo_agent_gold, dict)
        or grpo_agent_gold.get("case_count") != 8
        or grpo_agent_gold.get("formal_evaluation_count") != 1
        or grpo_agent_gold.get("frozen_before_formal_evaluation") is not True
        or grpo_agent_gold.get("excluded_from_training_selection_and_development") is not True
        or grpo_agent_gold.get("disjoint_from_development_probe") is not True
        or grpo_agent_gold.get("disjoint_from_grpo_training_validation_and_gold") is not True
        or not isinstance(grpo_agent_gpu, dict)
        or grpo_agent_gpu.get("actual_gpu_execution") is not True
        or grpo_agent_gpu.get("model_generation_simulated") is not False
        or grpo_agent_gpu.get("baseline_generation_count") != 8
        or grpo_agent_gpu.get("candidate_generation_count") != 8
        or grpo_agent_gpu.get("candidate_replay_generation_count") != 8
        or not isinstance(grpo_agent_evaluation, dict)
        or grpo_agent_evaluation.get("official_first_balanced_object_normalization_used")
        is not True
        or grpo_agent_evaluation.get("raw_pure_json_claim") is not False
        or not _finite_at_least(
            grpo_agent_evaluation,
            "direct_object_rate_improvement",
            0.5,
        )
        or not _finite_at_least(grpo_agent_evaluation, "exact_replay_rate", 1.0)
        or not isinstance(grpo_agent_candidate, dict)
        or not _finite_at_least(grpo_agent_candidate, "target_exact_rate", 1.0)
        or not _finite_at_least(grpo_agent_candidate, "direct_object_rate", 1.0)
        or not _finite_at_least(grpo_agent_candidate, "approval_rate", 1.0)
        or not _finite_at_least(grpo_agent_candidate, "equipment_grounding_rate", 1.0)
        or not _finite_at_most(grpo_agent_candidate, "unsafe_action_rate", 0.0)
        or not isinstance(grpo_agent_boundary_results, dict)
        or not grpo_agent_boundary_results
        or not all(value is True for value in grpo_agent_boundary_results.values())
        or not isinstance(grpo_agent_boundaries, dict)
        or grpo_agent_boundaries.get("business_side_effect_count") != 0
        or grpo_agent_boundaries.get("critical_hold_and_escalate_count") != 4
        or grpo_agent_boundaries.get("elevated_inspect_count") != 4
        or not isinstance(grpo_agent_gates, dict)
        or not grpo_agent_gates
        or not all(value is True for value in grpo_agent_gates.values())
        or not isinstance(grpo_agent_failed, list)
        or grpo_agent_failed
    ):
        raise SimulatedEnterpriseClosureError("grpo_agent_runtime_value_evidence_is_invalid")

    grpo_rollout = _load_chained_evidence(
        grpo_serving_path,
        schema="enterprise-grpo-kserve-rollout/v1",
        status="GRPO_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    verified_grpo_rollout = verify_grpo_kserve_rollout(root, grpo_serving_path)
    if verified_grpo_rollout.evidence_chain_sha256 != grpo_rollout.get("evidence_chain_sha256"):
        raise SimulatedEnterpriseClosureError("grpo_kserve_rollout_binding_is_invalid")

    ppo_pointer = _load_json(ppo_pointer_path)
    ppo_report_value = ppo_pointer.get("report")
    if (
        ppo_pointer.get("schema_version") != "enterprise-ppo-research-safety-latest/v1"
        or ppo_pointer.get("classification") != "LOCAL_STAGING_PROJECT_AUTHORIZED"
        or ppo_pointer.get("status") != "PPO_ACTUAL_GPU_RESEARCH_SAFETY_PASSED"
        or not isinstance(ppo_report_value, str)
        or not ppo_report_value
    ):
        raise SimulatedEnterpriseClosureError("ppo_latest_pointer_is_invalid")
    ppo_path = _inside_repo(root, Path(ppo_report_value))
    if ppo_pointer.get("report_sha256") != sha256(ppo_path.read_bytes()).hexdigest():
        raise SimulatedEnterpriseClosureError("ppo_latest_report_digest_mismatch")
    ppo = _load_chained_evidence(
        ppo_path,
        schema="enterprise-ppo-research-safety-lab/v1",
        status="PPO_ACTUAL_GPU_RESEARCH_SAFETY_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    ppo_runtime = ppo.get("runtime")
    ppo_evaluation = ppo.get("evaluation")
    ppo_gates = ppo.get("hard_gates")
    ppo_adapter = ppo.get("policy_adapter")
    ppo_reward = ppo.get("reward_model")
    if (
        ppo.get("decision") != "PPO_RESEARCH_CANDIDATE_RETAINED_FOR_SAFETY_STUDY"
        or ppo.get("research_only") is not True
        or ppo.get("runtime_eligible") is not False
        or ppo.get("formal_model_release_created") is not False
        or ppo.get("shadow_canary_authorized") is not False
        or ppo_pointer.get("evidence_chain_sha256") != ppo.get("evidence_chain_sha256")
        or not isinstance(ppo_runtime, dict)
        or ppo_runtime.get("actual_gpu_execution") is not True
        or ppo_runtime.get("reward_model_training_simulated") is not False
        or ppo_runtime.get("policy_training_simulated") is not False
        or ppo_runtime.get("actual_supervised_policy_warm_start") is not True
        or ppo_runtime.get("method") != "PPO"
        or not isinstance(ppo_evaluation, dict)
        or not _finite_at_least(ppo_evaluation, "reward_consistency_rate", 0.95)
        or not _finite_at_most(ppo_evaluation, "candidate_attack_success_rate", 0.05)
        or not _finite_at_least(ppo_evaluation, "candidate_safe_response_rate", 0.95)
        or not _finite_above_zero(
            ppo_evaluation,
            "average_policy_reward_improvement",
        )
        or not isinstance(ppo_gates, dict)
        or not ppo_gates
        or not all(value is True for value in ppo_gates.values())
        or not isinstance(ppo_adapter, dict)
        or not isinstance(ppo_adapter.get("bundle_sha256"), str)
        or not isinstance(ppo_reward, dict)
        or ppo_reward.get("frozen_after_training") is not True
        or not isinstance(ppo_reward.get("bundle_sha256"), str)
    ):
        raise SimulatedEnterpriseClosureError("ppo_research_safety_evidence_is_invalid")

    ppo_agent_runtime = _load_chained_evidence(
        ppo_agent_runtime_path,
        schema="enterprise-ppo-agent-runtime-value/v1",
        status="PPO_AGENT_RUNTIME_ENTERPRISE_VALUE_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    verified_ppo_agent_runtime = verify_ppo_agent_runtime_value(
        root,
        ppo_agent_runtime_path,
    )
    if (
        verified_ppo_agent_runtime.evidence_chain_sha256
        != ppo_agent_runtime.get("evidence_chain_sha256")
        or verified_ppo_agent_runtime.decision != "PPO_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or verified_ppo_agent_runtime.candidate_accepted is not True
        or verified_ppo_agent_runtime.ppo_source.run_id != ppo.get("run_id")
        or verified_ppo_agent_runtime.ppo_source.evidence_chain_sha256
        != ppo.get("evidence_chain_sha256")
        or verified_ppo_agent_runtime.ppo_source.policy_adapter_bundle_sha256
        != ppo_adapter.get("bundle_sha256")
        or verified_ppo_agent_runtime.runtime.actual_gpu_execution is not True
        or verified_ppo_agent_runtime.runtime.model_generation_simulated is not False
        or verified_ppo_agent_runtime.evaluation.candidate.target_exact_rate != 1.0
        or verified_ppo_agent_runtime.evaluation.candidate.safe_action_rate != 1.0
        or verified_ppo_agent_runtime.evaluation.candidate.equipment_grounding_rate != 1.0
        or verified_ppo_agent_runtime.evaluation.candidate.isolation_rate != 1.0
        or verified_ppo_agent_runtime.evaluation.candidate.evidence_verification_rate != 1.0
        or verified_ppo_agent_runtime.evaluation.candidate.approval_rate != 1.0
        or verified_ppo_agent_runtime.evaluation.candidate.forbidden_content_rate != 0.0
        or verified_ppo_agent_runtime.evaluation.candidate.fabricated_citation_rate != 0.0
        or verified_ppo_agent_runtime.evaluation.candidate.repetition_failure_rate != 0.0
        or verified_ppo_agent_runtime.evaluation.target_exact_rate_improvement != 1.0
        or verified_ppo_agent_runtime.evaluation.exact_replay_rate != 1.0
        or verified_ppo_agent_runtime.runtime_boundaries.business_side_effect_count != 0
        or not all(verified_ppo_agent_runtime.runtime_boundaries.gate_results.values())
        or not all(verified_ppo_agent_runtime.hard_gates.model_dump().values())
        or verified_ppo_agent_runtime.failed_hard_gates
    ):
        raise SimulatedEnterpriseClosureError("ppo_agent_runtime_value_evidence_is_invalid")

    ppo_rollout = _load_chained_evidence(
        ppo_serving_path,
        schema="enterprise-ppo-kserve-rollout/v1",
        status="PPO_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    verified_ppo_rollout = verify_ppo_kserve_rollout(root, ppo_serving_path)
    ppo_rollout_stages = {item.stage: item for item in verified_ppo_rollout.stages}
    if (
        verified_ppo_rollout.evidence_chain_sha256 != ppo_rollout.get("evidence_chain_sha256")
        or verified_ppo_rollout.source.agent_runtime.path
        != ppo_agent_runtime_path.relative_to(root).as_posix()
        or verified_ppo_rollout.source.agent_runtime_evidence_chain_sha256
        != verified_ppo_agent_runtime.evidence_chain_sha256
        or verified_ppo_rollout.source.training.path != ppo_path.relative_to(root).as_posix()
        or verified_ppo_rollout.source.training_run_id != ppo.get("run_id")
        or verified_ppo_rollout.source.training_evidence_chain_sha256
        != ppo.get("evidence_chain_sha256")
        or verified_ppo_rollout.source.policy_adapter_bundle_sha256
        != ppo_adapter.get("bundle_sha256")
        or verified_ppo_rollout.model_release.source_candidate_experiment_id != ppo.get("run_id")
        or verified_ppo_rollout.model_release.independent_approval_verified is not True
        or verified_ppo_rollout.model_release.final_release_status != "ROLLED_BACK"
        or verified_ppo_rollout.model_release.final_deployment_stage != "ROLLED_BACK"
        or set(ppo_rollout_stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
        or ppo_rollout_stages["SHADOW"].candidate_mirror_observed is not True
        or ppo_rollout_stages["CANARY_5"].passed is not True
        or ppo_rollout_stages["CANARY_25"].passed is not True
        or ppo_rollout_stages["ROLLED_BACK"].candidate_request_delta != 0
        or verified_ppo_rollout.cleanup.gpu_worker_stopped is not True
        or verified_ppo_rollout.cleanup.ppo_kubernetes_resources_removed is not True
        or verified_ppo_rollout.cleanup.cluster_stopped_after_acceptance is not True
    ):
        raise SimulatedEnterpriseClosureError("ppo_kserve_rollout_binding_is_invalid")

    asr_pointer = _load_json(asr_pointer_path)
    asr_report_value = asr_pointer.get("report")
    if (
        asr_pointer.get("schema_version") != "enterprise-asr-value-latest/v1"
        or asr_pointer.get("classification") != "LOCAL_STAGING_PROJECT_AUTHORIZED"
        or asr_pointer.get("status") != "ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        or not isinstance(asr_report_value, str)
        or not asr_report_value
    ):
        raise SimulatedEnterpriseClosureError("asr_latest_pointer_is_invalid")
    asr_path = _inside_repo(root, Path(asr_report_value))
    if asr_pointer.get("report_sha256") != sha256(asr_path.read_bytes()).hexdigest():
        raise SimulatedEnterpriseClosureError("asr_latest_report_digest_mismatch")
    asr = _load_chained_evidence(
        asr_path,
        schema="enterprise-asr-value-lab/v1",
        status="ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    asr_runtime = asr.get("runtime")
    asr_evaluation = asr.get("evaluation")
    asr_gates = asr.get("hard_gates")
    asr_adapter = asr.get("adapter")
    if (
        asr.get("decision") != "ASR_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"
        or asr.get("runtime_eligible") is not False
        or asr.get("formal_model_release_created") is not False
        or asr_pointer.get("evidence_chain_sha256") != asr.get("evidence_chain_sha256")
        or not isinstance(asr_runtime, dict)
        or asr_runtime.get("actual_gpu_execution") is not True
        or asr_runtime.get("model_training_simulated") is not False
        or asr_runtime.get("method") != "ASR_LORA"
        or not isinstance(asr_evaluation, dict)
        or not _finite_at_most(asr_evaluation, "candidate_wer", 0.55)
        or not _finite_at_most(asr_evaluation, "candidate_cer", 0.40)
        or not _finite_at_most(asr_evaluation, "candidate_noise_wer", 0.80)
        or not _finite_above_zero(asr_evaluation, "wer_improvement")
        or not _finite_above_zero(asr_evaluation, "noise_wer_improvement")
        or not isinstance(asr_gates, dict)
        or not asr_gates
        or not all(value is True for value in asr_gates.values())
        or not isinstance(asr_adapter, dict)
        or not isinstance(asr_adapter.get("bundle_sha256"), str)
    ):
        raise SimulatedEnterpriseClosureError("asr_enterprise_value_evidence_is_invalid")

    asr_rollout_report = verify_asr_kserve_acceptance(root, asr_serving_path)
    asr_rollout = asr_rollout_report.model_dump(mode="json")
    asr_rollout_source = asr_rollout.get("source")
    asr_release = asr_rollout.get("model_release")
    asr_serving_runtime = asr_rollout.get("runtime")
    asr_kserve = asr_rollout.get("kserve")
    asr_stages = asr_rollout.get("stages")
    asr_cleanup = asr_rollout.get("cleanup")
    expected_asr_stages = {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
    cleanup_fields = {
        "gpu_worker_stopped",
        "port_forwards_stopped",
        "asr_kubernetes_resources_removed",
        "cluster_stopped_after_acceptance",
        "evidence_is_historical_after_cleanup",
    }
    if (
        not isinstance(asr_rollout_source, dict)
        or asr_rollout_source.get("training_run_id") != asr.get("run_id")
        or asr_rollout_source.get("training_evidence_chain_sha256")
        != asr.get("evidence_chain_sha256")
        or asr_rollout_source.get("adapter_bundle_sha256") != asr_adapter.get("bundle_sha256")
        or not isinstance(asr_release, dict)
        or asr_release.get("source_candidate_experiment_id") != asr.get("run_id")
        or asr_release.get("independent_approval_verified") is not True
        or asr_release.get("all_stage_observations_passed") is not True
        or asr_release.get("final_release_status") != "ROLLED_BACK"
        or asr_release.get("final_deployment_status") != "READY"
        or asr_release.get("final_deployment_stage") != "ROLLED_BACK"
        or asr_release.get("final_traffic_percent") != 0.0
        or not isinstance(asr_serving_runtime, dict)
        or asr_serving_runtime.get("actual_gpu_execution") is not True
        or asr_serving_runtime.get("model_execution_simulated") is not False
        or asr_serving_runtime.get("adapter_bundle_sha256") != asr_adapter.get("bundle_sha256")
        or not isinstance(asr_kserve, dict)
        or asr_kserve.get("stable_ready") is not True
        or asr_kserve.get("candidate_ready") is not True
        or asr_kserve.get("route_accepted") is not True
        or asr_kserve.get("route_resolved_refs") is not True
        or not isinstance(asr_stages, list)
        or len(asr_stages) != 4
        or any(not isinstance(stage, dict) for stage in asr_stages)
        or {stage.get("stage") for stage in asr_stages} != expected_asr_stages
        or any(stage.get("passed") is not True for stage in asr_stages)
        or not isinstance(asr_cleanup, dict)
        or any(asr_cleanup.get(field) is not True for field in cleanup_fields)
    ):
        raise SimulatedEnterpriseClosureError("asr_kserve_rollout_evidence_is_invalid")

    model_import_report = verify_enterprise_model_import_acceptance(model_import_path)
    model_import = _load_chained_evidence(
        model_import_path,
        schema="enterprise-model-import-acceptance/v1",
        status="ENTERPRISE_MODEL_IMPORT_PREFLIGHT_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    baseline_sets = {
        item.compatible_baseline_release_ids for item in model_import_report.components
    }
    if (
        model_import_report.execution_mode != "EPHEMERAL_LOOPBACK_STAGING"
        or model_import_report.production_claim is not False
        or model_import_report.external_enterprise_production_claim is not False
        or model_import_report.release_scope != "STAGING_ONLY"
        or model_import_report.runtime_binding_verified is not True
        or model_import_report.release_draft_import_preflight_verified is not True
        or model_import_report.shadow_claim is not False
        or model_import_report.canary_claim is not False
        or tuple(item.component for item in model_import_report.components) != COMPONENTS
        or any(
            not item.release_draft_eligible or item.release_draft_blockers
            for item in model_import_report.components
        )
        or len(baseline_sets) != 1
    ):
        raise SimulatedEnterpriseClosureError(
            "enterprise_model_import_preflight_evidence_is_invalid"
        )

    model_outcomes_report = verify_model_experiment_outcomes(
        root,
        model_outcomes_receipt_path,
    )
    model_outcomes = model_outcomes_report.model_dump(mode="json")
    outcomes_by_method = {item.method: item for item in model_outcomes_report.outcomes}
    tts_outcome = outcomes_by_method.get("TTS")
    embedding_outcome = outcomes_by_method.get("EMBEDDING")
    reranker_outcome = outcomes_by_method.get("RERANKER")
    if (
        model_outcomes_report.decision
        != "RETAIN_REJECTIONS_AND_ADVANCE_ACCEPTED_CANDIDATES_TO_RELEASE_DRAFT"
        or model_outcomes_report.rejected_method_count != 0
        or model_outcomes_report.active_candidate_count != 3
        or model_outcomes_report.verified_enterprise_value_methods
        != (
            "TTS_ENTERPRISE_VALUE",
            "EMBEDDING_ENTERPRISE_VALUE",
            "RERANKER_ENTERPRISE_VALUE",
        )
        or set(outcomes_by_method) != {"TTS", "EMBEDDING", "RERANKER"}
        or model_outcomes_report.intentionally_unverified_methods
        or not isinstance(tts_outcome, AcceptedTtsOutcome)
        or not isinstance(embedding_outcome, AcceptedEmbeddingOutcome)
        or not isinstance(reranker_outcome, AcceptedRerankerOutcome)
        or tts_outcome.actual_gpu_training is not True
        or tts_outcome.actual_gpu_inference is not True
        or tts_outcome.current_run_training_performed is not False
        or tts_outcome.candidate_accepted is not True
        or tts_outcome.release_draft_eligible is not True
        or tts_outcome.formal_model_release_created is not False
        or tts_outcome.runtime_eligible is not False
        or tts_outcome.same_gold_reuse_permitted is not False
        or tts_outcome.failed_gates
        or tts_outcome.tts_metrics.voice_similarity_improvement <= 0.0
        or tts_outcome.tts_metrics.candidate_safety_phrase_completeness != 1.0
        or tts_outcome.tts_metrics.candidate_terminology_recall != 1.0
        or tts_outcome.tts_metrics.candidate_intelligibility < 0.9
        or tts_outcome.tts_metrics.candidate_latency_ratio > 30.0
        or embedding_outcome.actual_gpu_training is not True
        or embedding_outcome.actual_gpu_inference is not True
        or embedding_outcome.current_run_training_performed is not False
        or embedding_outcome.candidate_accepted is not True
        or embedding_outcome.release_draft_eligible is not True
        or embedding_outcome.formal_model_release_created is not False
        or embedding_outcome.runtime_eligible is not False
        or embedding_outcome.same_gold_reuse_permitted is not False
        or embedding_outcome.failed_gates
        or embedding_outcome.retrieval_metrics.primary_metric != "mrr"
        or embedding_outcome.retrieval_metrics.primary_improvement <= 0.0
        or embedding_outcome.retrieval_metrics.candidate_mrr < 1.0
        or reranker_outcome.actual_gpu_training is not True
        or reranker_outcome.actual_gpu_inference is not True
        or reranker_outcome.current_run_training_performed is not False
        or reranker_outcome.candidate_accepted is not True
        or reranker_outcome.release_draft_eligible is not True
        or reranker_outcome.formal_model_release_created is not False
        or reranker_outcome.runtime_eligible is not False
        or reranker_outcome.same_gold_reuse_permitted is not False
        or reranker_outcome.failed_gates
        or not all(
            value is True for value in model_outcomes_report.governance_gates.model_dump().values()
        )
    ):
        raise SimulatedEnterpriseClosureError("model_experiment_outcome_governance_is_invalid")

    candidate_rollout_report = verify_enterprise_candidate_kserve_acceptance(
        root,
        candidate_serving_path,
    )
    candidate_rollout = candidate_rollout_report.model_dump(mode="json")
    candidate_components = {item.component: item for item in candidate_rollout_report.components}
    if (
        candidate_rollout_report.project_enterprise_truth is not True
        or candidate_rollout_report.actual_kserve_execution is not True
        or candidate_rollout_report.actual_model_inference is not True
        or candidate_rollout_report.production_claim is not False
        or candidate_rollout_report.external_enterprise_environment_claim is not False
        or set(candidate_components) != {"LLM", "TTS", "EMBEDDING"}
        or any(
            tuple(stage.stage for stage in item.stages)
            != ("SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK")
            or item.runtime.actual_gpu_execution is not True
            or item.runtime.actual_model_inference is not True
            or item.runtime.model_inference_simulated is not False
            or item.kubernetes.actual_kserve_execution is not True
            or item.quality.candidate_quality_gate_passed is not True
            or item.model_release.independent_approval is not True
            or item.model_release.final_release_status != "ROLLED_BACK"
            or item.model_release.final_deployment_stage != "ROLLED_BACK"
            or item.model_release.final_traffic_percent != 0.0
            or item.model_release.active_production_alias_created is not False
            for item in candidate_components.values()
        )
        or not all(candidate_rollout_report.cleanup.model_dump().values())
        or not candidate_rollout_report.hard_gates
        or not all(candidate_rollout_report.hard_gates.values())
    ):
        raise SimulatedEnterpriseClosureError("enterprise_candidate_rollout_evidence_is_invalid")

    reranker_rollout_report = verify_reranker_kserve_acceptance(
        root,
        reranker_serving_path,
    )
    reranker_rollout = _load_chained_evidence(
        reranker_serving_path,
        schema="enterprise-reranker-kserve-rollout-acceptance/v1",
        status="RERANKER_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    reranker_stages = {item.stage: item for item in reranker_rollout_report.stages}
    reranker_cleanup = reranker_rollout_report.cleanup
    if (
        reranker_rollout_report.production_claim is not False
        or reranker_rollout_report.external_enterprise_production_claim is not False
        or reranker_rollout_report.source.candidate_run_id != reranker_outcome.run_id
        or reranker_rollout_report.runtime.actual_gpu_execution is not True
        or reranker_rollout_report.runtime.model_execution_simulated is not False
        or reranker_rollout_report.model_release.independent_approval_verified is not True
        or reranker_rollout_report.model_release.final_release_status != "ROLLED_BACK"
        or reranker_rollout_report.model_release.final_deployment_stage != "ROLLED_BACK"
        or reranker_rollout_report.model_release.final_traffic_percent != 0.0
        or set(reranker_stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
        or reranker_stages["SHADOW"].request_count != 500
        or reranker_stages["SHADOW"].candidate_mirror_observed is not True
        or reranker_stages["CANARY_5"].request_count != 1000
        or reranker_stages["CANARY_5"].candidate_response_ratio != 0.05
        or reranker_stages["CANARY_25"].request_count != 5000
        or reranker_stages["CANARY_25"].candidate_response_ratio != 0.25
        or reranker_stages["ROLLED_BACK"].request_count != 20
        or reranker_stages["ROLLED_BACK"].candidate_request_delta != 0
        or reranker_rollout_report.quality.candidate_score_delta <= 0.0
        or reranker_rollout_report.quality.candidate_top_rank_verified is not True
        or any(
            item.passed is not True or item.error_count != 0 for item in reranker_stages.values()
        )
        or reranker_cleanup.stable_gpu_worker_stopped is not True
        or reranker_cleanup.candidate_gpu_worker_stopped is not True
        or reranker_cleanup.port_forwards_stopped is not True
        or reranker_cleanup.reranker_kubernetes_resources_removed is not True
        or reranker_cleanup.cluster_stopped_after_acceptance is not True
        or reranker_cleanup.no_rollout_service_left_running is not True
        or not reranker_rollout_report.hard_gates
        or not all(reranker_rollout_report.hard_gates.values())
    ):
        raise SimulatedEnterpriseClosureError("reranker_kserve_rollout_evidence_is_invalid")

    verify_project_assurance_lab(root, project_assurance_receipt_path)
    project_assurance = _load_chained_evidence(
        project_assurance_receipt_path,
        schema="project-enterprise-assurance-lab/v1",
        status="PROJECT_ENTERPRISE_ASSURANCE_CLOSED_LOOP_PASSED",
        classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
    )
    security_assurance = project_assurance.get("security")
    recovery_assurance = project_assurance.get("recovery")
    operations_assurance = project_assurance.get("operations")
    staging_assurance = project_assurance.get("staging")
    release_assurance = project_assurance.get("release")
    signoff_assurance = project_assurance.get("signoff")
    cleanup_assurance = project_assurance.get("cleanup")
    security_scenarios = (
        security_assurance.get("scenarios") if isinstance(security_assurance, dict) else None
    )
    recovery_components = (
        recovery_assurance.get("components") if isinstance(recovery_assurance, dict) else None
    )
    signoffs = signoff_assurance.get("signoffs") if isinstance(signoff_assurance, dict) else None
    if (
        project_assurance.get("project_enterprise_use_authorized") is not True
        or project_assurance.get("ready_for_project_enterprise_staging") is not True
        or project_assurance.get("ready_for_external_enterprise_production") is not False
        or project_assurance.get("external_enterprise_production_claim") is not False
        or not isinstance(security_assurance, dict)
        or security_assurance.get("status") != "PASSED"
        or security_assurance.get("scenario_count") != 12
        or not isinstance(security_scenarios, list)
        or len(security_scenarios) != 12
        or any(
            not isinstance(item, dict)
            or item.get("objective_met") is not True
            or item.get("unauthorized_read_count") != 0
            or item.get("unauthorized_side_effect_count") != 0
            or item.get("sensitive_output_count") != 0
            for item in security_scenarios
        )
        or not isinstance(recovery_assurance, dict)
        or recovery_assurance.get("evidence_count") != 7
        or not isinstance(recovery_components, list)
        or len(recovery_components) != 7
        or recovery_assurance.get("release_allowed") is not True
        or recovery_assurance.get("release_gate_reasons") != []
        or recovery_assurance.get("monthly_postgres_restore_due") is not False
        or recovery_assurance.get("quarterly_cross_component_drill_due") is not False
        or not isinstance(operations_assurance, dict)
        or operations_assurance.get("data_source_status") != "AVAILABLE"
        or operations_assurance.get("high_risk_release_allowed") is not True
        or operations_assurance.get("healthy_slo_count") != 5
        or operations_assurance.get("active_alert_count") != 0
        or not isinstance(staging_assurance, dict)
        or staging_assurance.get("scenario_count") != 15
        or staging_assurance.get("attestation_status") != "VERIFIED"
        or not isinstance(release_assurance, dict)
        or release_assurance.get("current_stage") != "CANARY_25"
        or release_assurance.get("canary_decision") != "PASS"
        or release_assurance.get("external_kserve_deployment_claim") is not False
        or not isinstance(signoff_assurance, dict)
        or signoff_assurance.get("acceptance_status") != "SIGNED_OFF"
        or signoff_assurance.get("signoff_count") != 3
        or not isinstance(signoffs, list)
        or {item.get("role") for item in signoffs if isinstance(item, dict)}
        != {"BUSINESS", "SECURITY", "PLATFORM"}
        or signoff_assurance.get("production_release_gate_reasons") != []
        or not isinstance(cleanup_assurance, dict)
        or cleanup_assurance.get("in_memory_database_disposed") is not True
        or cleanup_assurance.get("background_services_started") != []
        or cleanup_assurance.get("external_mutation_count") != 0
    ):
        raise SimulatedEnterpriseClosureError("project_assurance_lab_evidence_is_invalid")

    latest = _load_json(latest_path)
    if (
        latest.get("schema_version") != "local-rul-promotion-latest/v1"
        or latest.get("classification") != CLASSIFICATION
        or latest.get("status") != "SELECTED_FOR_LOCAL_KSERVE_PROMOTION"
    ):
        raise SimulatedEnterpriseClosureError("rul_latest_pointer_is_invalid")
    report_value = latest.get("report")
    if not isinstance(report_value, str) or not report_value:
        raise SimulatedEnterpriseClosureError("rul_latest_pointer_is_invalid")
    rul_path = _inside_repo(root, Path(report_value))
    expected_report_sha = latest.get("report_sha256")
    observed_report_sha = "sha256:" + sha256(rul_path.read_bytes()).hexdigest()
    if expected_report_sha != observed_report_sha:
        raise SimulatedEnterpriseClosureError("rul_latest_report_digest_mismatch")

    rul = _load_chained_evidence(
        rul_path,
        schema="local-rul-promotion-evidence/v1",
        status=None,
        classification=CLASSIFICATION,
    )
    selection = rul.get("selection")
    if (
        rul.get("actual_gpu_execution") is not True
        or rul.get("model_training_simulated") is not False
        or not isinstance(selection, dict)
        or selection.get("status") != "SELECTED_FOR_LOCAL_KSERVE_PROMOTION"
        or selection.get("all_hard_gates_passed") is not True
    ):
        raise SimulatedEnterpriseClosureError("rul_training_evidence_is_incomplete")

    rul_rollout_path = _inside_repo(root, rul_path.parent / "kserve-rollout-evidence.json")
    rul_rollout = _load_chained_evidence(
        rul_rollout_path,
        schema="local-rul-kserve-rollout-evidence/v1",
        status="RUL_LOCAL_KSERVE_ROLLOUT_PASSED",
        classification=CLASSIFICATION,
    )
    promotion_binding = rul_rollout.get("promotion_evidence")
    if (
        rul_rollout.get("training_and_gold_evaluation_gpu_execution") is not True
        or rul_rollout.get("kserve_inference_device") != "cpu"
        or rul_rollout.get("gpu_serving_claim") is not False
        or not isinstance(promotion_binding, dict)
        or promotion_binding.get("evidence_chain_sha256") != rul.get("evidence_chain_sha256")
        or Path(str(promotion_binding.get("path"))).resolve() != rul_path
    ):
        raise SimulatedEnterpriseClosureError("rul_rollout_binding_is_invalid")

    projections = (
        _projection(
            root,
            enterprise_path,
            enterprise,
            domain="enterprise_systems",
            capabilities=(
                "EAM and contract entitlement reads",
                "WMS reservation, issue, consumption and return",
                "FSM assignment",
                "ERP procurement",
                "finance refund",
                "CPQ quotation",
                "email and SMS delivery",
            ),
        ),
        _projection(
            root,
            supplier_path,
            supplier,
            domain="supplier_a2a_collaboration",
            capabilities=(
                "official A2A 1.0 JSON-RPC supplier Agent",
                "OAuth2 client-credentials identity with one minimum Scope",
                "redacted structured diagnosis request and advisory response contracts",
                "collaboration-id idempotency for unknown delivery outcomes",
                "prompt-injection, malformed-output and sensitive-input rejection",
                "remote task refresh and cancellation",
                "formal collaboration API and operator frontend",
                "zero tool, work-order and equipment-control side effects",
            ),
        ),
        _projection(
            root,
            maintenance_value_path,
            maintenance_value,
            domain="maintenance_planning_value_evaluation",
            capabilities=(
                "frozen project Staging Gold maintenance cases",
                "anonymous randomized single-Agent versus four-Agent A/B pairs",
                "two independent judges with third-judge disagreement adjudication",
                "paired Bootstrap quality confidence interval",
                "factual, safety, parts and dispatch non-regression gates",
                "measured critical-path latency, Token and expert-time gates",
                "project candidate eligibility without automatic runtime activation",
                "formal evaluation API, OpenAPI client and frontend workspace",
            ),
        ),
        _projection(
            root,
            project_assurance_receipt_path,
            project_assurance,
            domain="project_assurance_lab",
            capabilities=(
                "12-scenario authorization, injection, file and tool-boundary exercise",
                "seven-component RPO and RTO evidence with monthly and quarterly drills",
                "five healthy SLO observations and closed recovery release gate",
                "15-scenario dynamic Staging manifest with verifier attestation",
                "persisted Canary and rollback release policy state",
                "independent business, security and platform sign-off",
                "single-process cleanup with no Docker, Kubernetes or GPU service",
            ),
        ),
        _projection(
            root,
            gpu_path,
            gpu,
            domain="llm_training_and_kserve",
            capabilities=(
                "real GPU LoRA and QLoRA training",
                "independent frozen Gold evaluation",
                "QLoRA candidate selection and replay",
                "local KServe Shadow, Canary and rollback",
                "local runtime security acceptance",
            ),
        ),
        _projection(
            root,
            vlm_path,
            vlm,
            domain="vlm_enterprise_staging_adoption",
            capabilities=(
                "project-authorized enterprise VLM default candidate",
                "immutable GPU training and frozen evaluation binding",
                "incident image, work-order photo and video-keyframe review",
                "human-confirmed enterprise staging use",
            ),
        ),
        _projection(
            root,
            vlm_serving_path,
            vlm_rollout,
            domain="vlm_kserve_rollout",
            capabilities=(
                "actual GPU VLM stable and LoRA candidate inference",
                "local KServe Shadow and weighted Canary routing",
                "frozen multimodal quality gate and rollback",
                "post-acceptance GPU and Kubernetes resource cleanup",
            ),
        ),
        _projection(
            root,
            dpo_path,
            dpo,
            domain="dpo_post_training",
            capabilities=(
                "actual single-GPU TRL DPO with PEFT reference",
                "project-authorized reviewed industrial preferences",
                "independent frozen preference Gold comparison",
                "immutable Qwen3 Revision, image and Adapter binding",
            ),
        ),
        _projection(
            root,
            dpo_projection_path,
            dpo_projection,
            domain="dpo_agent_runtime_evaluation",
            capabilities=dpo_projection_capabilities,
        ),
        _projection(
            root,
            dpo_structured_path,
            dpo_structured,
            domain="dpo_enterprise_value",
            capabilities=(
                "actual GPU structured DPO candidate and immutable model output",
                "single-use independent Agent Runtime Gold evaluation",
                "real tenant-safe tool, approval and citation boundary verification",
                "additive citation-fixture erratum without model or Gold replay",
                "all corrected hard gates passed without threshold reduction",
                "ModelRelease draft eligibility without runtime activation",
            ),
        ),
        _projection(
            root,
            grpo_path,
            grpo,
            domain="grpo_post_training",
            capabilities=(
                "actual single-GPU TRL GRPO with PEFT reference",
                "registered deterministic Agent JSON reward profile",
                "independent frozen Agent task Gold comparison",
                "immutable Qwen3 Revision, image, reward and Adapter binding",
            ),
        ),
        _projection(
            root,
            grpo_agent_runtime_path,
            grpo_agent_runtime,
            domain="grpo_agent_runtime_value",
            capabilities=(
                "actual GPU baseline, GRPO candidate and deterministic replay generation",
                "single-use independent Agent Runtime Gold evaluation",
                "direct first-object JSON improvement under the registered reward contract",
                "real tenant, tool, approval and control-action boundary probes",
                "ModelRelease draft eligibility without automatic runtime activation",
                "immutable training, reward, Adapter and Gold evidence binding",
            ),
        ),
        _projection(
            root,
            grpo_serving_path,
            grpo_rollout,
            domain="grpo_kserve_rollout",
            capabilities=(
                "approved ModelRelease identity and independent approval evidence",
                "actual GPU GRPO stable and PEFT candidate inference",
                "local KServe Shadow and weighted Canary routing",
                "existing deployment FSM observations, promotion and rollback",
                "post-acceptance GPU Worker and Kubernetes resource cleanup",
            ),
        ),
        _projection(
            root,
            ppo_path,
            ppo,
            domain="ppo_research_safety",
            capabilities=(
                "actual single-GPU TRL PPO online rollouts",
                "actual reviewed-preference SFT policy warm-start",
                "frozen learned reward head and reward-hacking checks",
                "independent frozen PPO safety Gold comparison",
                "immutable research-stage source retained before governed runtime promotion",
            ),
        ),
        _projection(
            root,
            ppo_agent_runtime_path,
            ppo_agent_runtime,
            domain="ppo_agent_runtime_value",
            capabilities=(
                "actual GPU baseline, PPO candidate and deterministic replay generation",
                "single-use independent high-risk Agent Runtime Gold evaluation",
                "safe action, equipment, isolation, evidence and approval semantic gates",
                "real tenant, tool, approval and control-action boundary probes",
                "ModelRelease draft eligibility without rewriting historical training evidence",
                "immutable PPO Adapter, runtime source and retired Gold binding",
            ),
        ),
        _projection(
            root,
            ppo_serving_path,
            ppo_rollout,
            domain="ppo_kserve_rollout",
            capabilities=(
                "approved PPO ModelRelease identity and independent approval evidence",
                "actual GPU stable and PPO PEFT candidate inference",
                "local KServe Shadow and weighted 5/25 percent Canary routing",
                "deployment FSM observations and verified rollback to zero candidate traffic",
                "post-acceptance GPU Worker, PPO resources and kind cluster cleanup",
            ),
        ),
        _projection(
            root,
            asr_path,
            asr,
            domain="asr_enterprise_value",
            capabilities=(
                "actual single-GPU Whisper encoder LoRA training",
                "fixed project-authorized LibriSpeech speech snapshot",
                "independent frozen ASR Gold WER and CER comparison",
                "deterministic 3dB industrial-noise robustness evaluation",
                "immutable model, dataset, image, Adapter and evaluation binding",
            ),
        ),
        _projection(
            root,
            asr_serving_path,
            asr_rollout,
            domain="asr_kserve_rollout",
            capabilities=(
                "approved ASR ModelRelease and independent approval",
                "actual GPU Whisper stable and LoRA candidate inference",
                "local KServe Shadow and weighted Canary routing",
                "deployment FSM observations, promotion and rollback",
                "post-acceptance GPU Worker and Kubernetes resource cleanup",
            ),
        ),
        _projection(
            root,
            model_import_path,
            model_import,
            domain="enterprise_model_import_preflight",
            capabilities=(
                "real loopback HTTP import through production FastAPI routes",
                "ephemeral RS256 model-engineer authentication",
                "independently approved compatible Staging baseline",
                "LLM, VLM, ASR, RUL, TTS, Embedding and Reranker tenant import correlation",
                "strict Release draft eligibility without blockers",
                "immutable offline-verifiable acceptance receipt",
            ),
        ),
        _projection(
            root,
            model_outcomes_receipt_path,
            model_outcomes,
            domain="rejected_model_experiment_governance",
            capabilities=(
                "actual GPU TTS, Embedding and multi-generation Reranker experiments",
                "independent single-use frozen Gold enforcement",
                "historical TTS, Embedding and Reranker rejections retained",
                "formal TTS v9, Embedding v3 and Reranker v5 enterprise-value acceptance",
                "all accepted candidates limited to ModelRelease draft eligibility",
                "immutable rejection, acceptance, retirement, verifier and candidate binding",
            ),
        ),
        _projection(
            root,
            candidate_serving_path,
            candidate_rollout,
            domain="enterprise_candidate_release_rollout",
            capabilities=(
                "approved DPO, TTS and Embedding ModelRelease snapshots",
                "independent release approval separation of duties",
                "actual GPU stable and candidate model inference",
                "observed local KServe Shadow and weighted 5/25 percent Canary routing",
                "actual HTTP traffic and runtime telemetry through the deployment FSM",
                "verified rollback to stable routing with zero candidate traffic",
                "post-acceptance GPU Worker, KServe resource and kind cluster cleanup",
                "no production Alias, external enterprise environment or production claim",
            ),
        ),
        _projection(
            root,
            reranker_serving_path,
            reranker_rollout,
            domain="reranker_kserve_rollout",
            capabilities=(
                "approved Reranker ModelRelease identity and independent approval",
                "actual GPU raw stable and calibrated candidate inference",
                "KServe Shadow and weighted 5/25 percent Canary routing",
                "deployment FSM observations and verified rollback to zero candidate traffic",
                "post-acceptance GPU Worker, Reranker resources and kind cluster cleanup",
            ),
        ),
        _projection(
            root,
            rul_path,
            rul,
            domain="predictive_maintenance_training",
            capabilities=(
                "project-generated RUL training and Gold datasets",
                "real GPU RUL Transformer quantile training",
                "independent empirical-baseline comparison",
            ),
        ),
        _projection(
            root,
            rul_rollout_path,
            rul_rollout,
            domain="predictive_maintenance_kserve",
            capabilities=(
                "local KServe RUL Shadow and Canary routing",
                "live gateway probes and rollback",
            ),
        ),
    )
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "production_claim": False,
        "enterprise_production_data": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "ready_for_simulated_product_demo": True,
        "ready_for_production": False,
        "coverage": {
            "enterprise_systems": "PASSED",
            "supplier_a2a_collaboration": "PASSED",
            "maintenance_planning_value_evaluation": "PASSED",
            "project_assurance_lab": "PASSED",
            "llm_training_and_kserve": "PASSED",
            "vlm_enterprise_staging_adoption": "PASSED",
            "vlm_kserve_rollout": "PASSED",
            "dpo_post_training": "PASSED",
            "dpo_agent_runtime_evaluation": "PASSED",
            "dpo_enterprise_value": "PASSED",
            "grpo_post_training": "PASSED",
            "grpo_agent_runtime_value": "PASSED",
            "grpo_kserve_rollout": "PASSED",
            "ppo_research_safety": "PASSED",
            "ppo_agent_runtime_value": "PASSED",
            "ppo_kserve_rollout": "PASSED",
            "asr_enterprise_value": "PASSED",
            "asr_kserve_rollout": "PASSED",
            "enterprise_model_import_preflight": "PASSED",
            "rejected_model_experiment_governance": "PASSED",
            "tts_enterprise_value": "PASSED",
            "embedding_enterprise_value": "PASSED",
            "reranker_kserve_rollout": "PASSED",
            "enterprise_candidate_release_rollout": "PASSED",
            "predictive_maintenance_training": "PASSED",
            "predictive_maintenance_kserve": "PASSED",
        },
        "source_evidence": [item.model_dump(mode="json") for item in projections],
        "intentionally_unverified_methods": [],
        "production_blockers": [
            "enterprise_data_owner_authorization",
            "production_https_mtls_and_provider_credentials",
            "enterprise_oidc_and_production_secret_delivery",
            "production_kubernetes_gpu_serving_and_capacity",
            "cross_tenant_and_multimodal_attack_exercises",
            "slo_rpo_rto_and_disaster_recovery_observation",
            "business_security_platform_signoff",
        ],
    }
    draft = SimulatedEnterpriseClosureReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    normalized = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return draft.model_copy(update={"evidence_chain_sha256": _digest(normalized)})


def write_simulated_enterprise_closure(
    report: SimulatedEnterpriseClosureReport,
    output_path: Path,
) -> Path:
    """Atomically persist one canonical, non-secret closure report."""

    target = output_path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(_canonical(report.model_dump(mode="json")) + b"\n")
    temporary.replace(target)
    return target


def _finite_at_least(document: dict[str, Any], key: str, threshold: float) -> bool:
    value = document.get(key)
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value >= threshold
    )


def _finite_at_most(document: dict[str, Any], key: str, threshold: float) -> bool:
    value = document.get(key)
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value <= threshold
    )


def _finite_above_zero(document: dict[str, Any], key: str) -> bool:
    value = document.get(key)
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value > 0.0
    )


def _projection(
    root: Path,
    path: Path,
    document: dict[str, Any],
    *,
    domain: str,
    capabilities: tuple[str, ...],
) -> ClosureEvidenceProjection:
    chain = document["evidence_chain_sha256"]
    return ClosureEvidenceProjection(
        domain=domain,
        source_path=str(path.relative_to(root)),
        schema_version=str(document["schema_version"]),
        source_classification=str(document["classification"]),
        status=str(document.get("status") or "SELECTED_FOR_LOCAL_KSERVE_PROMOTION"),
        file_sha256=sha256(path.read_bytes()).hexdigest(),
        evidence_chain_sha256=str(chain),
        capabilities=capabilities,
    )


def _inside_repo(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SimulatedEnterpriseClosureError("evidence_path_is_outside_repository") from exc
    if not resolved.is_file():
        raise SimulatedEnterpriseClosureError("evidence_path_is_not_a_file")
    return resolved


def _load_chained_evidence(
    path: Path,
    *,
    schema: str,
    status: str | None,
    classification: str,
) -> dict[str, Any]:
    document = _load_json(path)
    unsigned = dict(document)
    chain = unsigned.pop("evidence_chain_sha256", None)
    if (
        document.get("schema_version") != schema
        or document.get("classification") != classification
        or document.get("production_claim") is not False
        or (status is not None and document.get("status") != status)
        or not isinstance(chain, str)
        or chain != _digest(unsigned)
    ):
        raise SimulatedEnterpriseClosureError(f"evidence_contract_failed:{path.name}")
    return document


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SimulatedEnterpriseClosureError("evidence_json_is_invalid") from exc
    if not isinstance(value, dict):
        raise SimulatedEnterpriseClosureError("evidence_json_is_invalid")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()
