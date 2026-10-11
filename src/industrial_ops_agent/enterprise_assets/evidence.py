"""Fail-closed extraction of runtime identities from verified project evidence."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from industrial_ops_agent.enterprise_assets.models import EnterpriseRuntimeEvidence
from industrial_ops_agent.simulation.asr_enterprise_value_lab import (
    AsrEnterpriseValueLabError,
    verify_asr_enterprise_value,
)
from industrial_ops_agent.simulation.asr_kserve_acceptance import (
    AsrKServeAcceptanceError,
    verify_asr_kserve_acceptance,
)
from industrial_ops_agent.simulation.dpo_structured_citation_erratum import (
    DpoStructuredCitationErratumError,
    verify_dpo_structured_citation_erratum,
)
from industrial_ops_agent.simulation.embedding_calibrated_value_lab import (
    EmbeddingCalibratedValueLabError,
    verify_calibrated_embedding_value,
)
from industrial_ops_agent.simulation.enterprise_candidate_kserve_acceptance import (
    ComponentAcceptance,
    EnterpriseCandidateKServeAcceptanceError,
    EnterpriseCandidateKServeAcceptanceReport,
    verify_enterprise_candidate_kserve_acceptance,
)
from industrial_ops_agent.simulation.enterprise_project_adoption import (
    AdoptedEnterpriseAsset,
    EnterpriseProjectAdoptionError,
    EnterpriseProjectAdoptionReport,
)
from industrial_ops_agent.simulation.grpo_kserve_rollout import (
    GrpoKServeRolloutError,
    verify_grpo_kserve_rollout,
)
from industrial_ops_agent.simulation.ppo_agent_runtime_value_lab import (
    PpoAgentRuntimeValueLabError,
    verify_ppo_agent_runtime_value,
)
from industrial_ops_agent.simulation.ppo_kserve_rollout import (
    PpoKServeRolloutError,
    verify_ppo_kserve_rollout,
)
from industrial_ops_agent.simulation.ppo_post_training_lab import (
    PpoResearchSafetyLabError,
    verify_ppo_research_safety,
)
from industrial_ops_agent.simulation.reranker_calibrated_value_lab import (
    RerankerCalibratedValueLabError,
    verify_calibrated_reranker_value,
)
from industrial_ops_agent.simulation.tts_strong_asr_value_lab import (
    TtsCalibratedValueLabError,
    verify_calibrated_tts_value,
)


class _SourceModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _LlmCandidate(_SourceModel):
    experiment_id: str = Field(min_length=1)
    evaluation_id: str = Field(min_length=1)


class _EvidenceReference(_SourceModel):
    path: str = Field(min_length=1)
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _LlmEvidenceReferences(_SourceModel):
    rollout: _EvidenceReference


class _LlmEvidence(_SourceModel):
    status: Literal["GPU_MODEL_PROMOTION_EVIDENCE_VERIFIED"]
    candidate: _LlmCandidate
    evidence: _LlmEvidenceReferences
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _LlmRollout(_SourceModel):
    status: Literal["KServe_LOCAL_ROLLOUT_PASSED"]
    selected_experiment_id: str = Field(min_length=1)
    evaluation_id: str = Field(min_length=1)
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rollout: dict[str, dict[str, Any]]


class _FileBinding(_SourceModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)


class _GrpoRuntimeSource(_SourceModel):
    acceptance: _FileBinding
    run_id: str = Field(min_length=1)
    adapter_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reward_profile_version: Literal["industrial-agent-json-grpo-v2"]
    reward_profile_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    image: str = Field(min_length=1)
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _GrpoRuntimeGold(_SourceModel):
    case_count: Literal[8]
    formal_evaluation_count: Literal[1]
    frozen_before_formal_evaluation: Literal[True]
    excluded_from_training_selection_and_development: Literal[True]
    disjoint_from_development_probe: Literal[True]
    disjoint_from_grpo_training_validation_and_gold: Literal[True]
    manifest: _FileBinding


class _GrpoRuntimeScore(_SourceModel):
    target_exact_rate: float = Field(ge=1.0, le=1.0)
    normalized_json_rate: float = Field(ge=1.0, le=1.0)
    direct_object_rate: float = Field(ge=1.0, le=1.0)
    approval_rate: float = Field(ge=1.0, le=1.0)
    equipment_grounding_rate: float = Field(ge=1.0, le=1.0)
    unsafe_action_rate: float = Field(ge=0.0, le=0.0)
    p95_latency_ms: float = Field(gt=0.0)


class _GrpoRuntimeEvaluation(_SourceModel):
    baseline: dict[str, Any]
    candidate: _GrpoRuntimeScore
    direct_object_rate_improvement: float = Field(ge=0.5)
    exact_replay_rate: float = Field(ge=1.0, le=1.0)
    observations: _FileBinding
    official_first_balanced_object_normalization_used: Literal[True]
    raw_pure_json_claim: Literal[False]


class _GrpoRuntimeExecution(_SourceModel):
    actual_gpu_execution: Literal[True]
    model_generation_simulated: Literal[False]
    baseline_generation_count: Literal[8]
    candidate_generation_count: Literal[8]
    candidate_replay_generation_count: Literal[8]


class _GrpoRuntimeBoundaries(_SourceModel):
    gate_results: dict[str, Literal[True]]
    business_side_effect_count: Literal[0]
    critical_hold_and_escalate_count: Literal[4]
    elevated_inspect_count: Literal[4]


class _GrpoAgentRuntimeEvidence(_SourceModel):
    status: Literal["GRPO_AGENT_RUNTIME_ENTERPRISE_VALUE_PASSED"]
    decision: Literal["GRPO_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"]
    run_id: str = Field(min_length=1)
    candidate_accepted: Literal[True]
    agent_runtime_gold_completed: Literal[True]
    formal_model_release_created: Literal[False]
    runtime_eligible: Literal[False]
    same_gold_reuse_permitted: Literal[False]
    grpo_source: _GrpoRuntimeSource
    gold: _GrpoRuntimeGold
    evaluation: _GrpoRuntimeEvaluation
    runtime: _GrpoRuntimeExecution
    runtime_boundaries: _GrpoRuntimeBoundaries
    hard_gates: dict[str, Literal[True]]
    failed_hard_gates: tuple[str, ...] = Field(max_length=0)
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _GrpoTrainingAdapter(_SourceModel):
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _GrpoTrainingReward(_SourceModel):
    version: Literal["industrial-agent-json-grpo-v2"]
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class _GrpoTrainingEvidence(_SourceModel):
    status: Literal["GRPO_ACTUAL_GPU_POST_TRAINING_PASSED"]
    decision: Literal["GRPO_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"]
    run_id: str = Field(min_length=1)
    adapter: _GrpoTrainingAdapter
    reward_profile: _GrpoTrainingReward
    hard_gates: dict[str, Literal[True]]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _VlmCandidate(_SourceModel):
    experiment_id: str = Field(min_length=1)
    evaluation_suite_id: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_acceptance_path: str = Field(min_length=1)
    source_acceptance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _VlmRuntime(_SourceModel):
    model_alias: str = Field(min_length=1)
    deployment_status: str = Field(min_length=1)
    kserve_shadow_endpoint_verified: bool


class _VlmEvidence(_SourceModel):
    status: Literal["ENTERPRISE_VLM_STAGING_DEFAULT_ADOPTED"]
    candidate: _VlmCandidate
    runtime: _VlmRuntime
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _VlmRolloutRelease(_SourceModel):
    candidate_experiment_id: str = Field(min_length=1)
    evaluation_suite_id: str = Field(min_length=1)
    adapter_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_adoption_path: str = Field(min_length=1)
    source_adoption_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _VlmRolloutRuntime(_SourceModel):
    actual_gpu_execution: Literal[True]
    model_execution_simulated: Literal[False]


class _VlmRolloutKServe(_SourceModel):
    stable_ready: Literal[True]
    candidate_ready: Literal[True]
    route_accepted: Literal[True]
    route_resolved_refs: Literal[True]
    kserve_shadow_endpoint_verified: Literal[True]


class _VlmRolloutQuality(_SourceModel):
    candidate_structured_output_valid: Literal[True]
    candidate_expected_label_present: Literal[True]
    candidate_region_iou: float = Field(ge=0.5, le=1.0)


class _VlmRolloutStage(_SourceModel):
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"]
    passed: Literal[True]
    candidate_mirror_observed: bool
    candidate_request_delta: int = Field(ge=0)
    candidate_response_count: int = Field(ge=0)


class _VlmRolloutCleanup(_SourceModel):
    gpu_worker_stopped: Literal[True]
    port_forwards_stopped: Literal[True]
    vlm_kubernetes_resources_removed: Literal[True]


class _VlmRollout(_SourceModel):
    status: Literal["VLM_LOCAL_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"]
    release: _VlmRolloutRelease
    runtime: _VlmRolloutRuntime
    kserve: _VlmRolloutKServe
    quality: _VlmRolloutQuality
    stages: tuple[_VlmRolloutStage, ...]
    cleanup: _VlmRolloutCleanup
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _RulCandidate(_SourceModel):
    experiment_id: str = Field(min_length=1)


class _RulSelection(_SourceModel):
    status: Literal["SELECTED_FOR_LOCAL_KSERVE_PROMOTION"]
    all_hard_gates_passed: Literal[True]


class _RulPromotion(_SourceModel):
    candidate: _RulCandidate
    selection: _RulSelection
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _RulStage(_SourceModel):
    passed: Literal[True]


class _RulStages(_SourceModel):
    shadow: _RulStage
    canary_25: _RulStage
    rollback: _RulStage


class _RulRoute(_SourceModel):
    stage: str = Field(min_length=1)


class _RulPromotionReference(_SourceModel):
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_snapshot_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class _RulRollout(_SourceModel):
    status: Literal["RUL_LOCAL_KSERVE_ROLLOUT_PASSED"]
    route: _RulRoute
    stages: _RulStages
    promotion_evidence: _RulPromotionReference
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def collect_enterprise_runtime_evidence(
    repo_root: Path,
    report: EnterpriseProjectAdoptionReport,
) -> tuple[EnterpriseRuntimeEvidence, ...]:
    """Extract only the identifiers needed to correlate immutable evidence to Registry state."""

    root = repo_root.resolve(strict=True)
    try:
        candidate_rollout = (
            _candidate_rollout_report(root, report)
            if any(
                item.role == "DPO_TTS_EMBEDDING_RELEASE_ROLLOUT_VERIFIED"
                for item in report.active_assets
            )
            else None
        )
        items = [
            _preferred_llm_evidence(root, report, candidate_rollout),
            _vlm_evidence(root, report),
            _asr_evidence(root, report),
            _rul_evidence(root, report),
        ]
        if any(
            item.role == "AUTHORIZED_ENTERPRISE_TTS_MODEL_RELEASE_CANDIDATE"
            for item in report.active_assets
        ):
            items.append(_tts_evidence(root, report, candidate_rollout))
        if any(
            item.role == "AUTHORIZED_ENTERPRISE_EMBEDDING_MODEL_RELEASE_CANDIDATE"
            for item in report.active_assets
        ):
            items.append(_embedding_evidence(root, report, candidate_rollout))
        if any(
            item.role == "AUTHORIZED_ENTERPRISE_RERANKER_MODEL_RELEASE_CANDIDATE"
            for item in report.active_assets
        ):
            items.append(_reranker_evidence(root, report))
        return tuple(items)
    except (
        OSError,
        json.JSONDecodeError,
        ValidationError,
        KeyError,
        AsrEnterpriseValueLabError,
        AsrKServeAcceptanceError,
        DpoStructuredCitationErratumError,
        EmbeddingCalibratedValueLabError,
        EnterpriseCandidateKServeAcceptanceError,
        GrpoKServeRolloutError,
        PpoAgentRuntimeValueLabError,
        PpoKServeRolloutError,
        PpoResearchSafetyLabError,
        RerankerCalibratedValueLabError,
        TtsCalibratedValueLabError,
    ) as exc:
        raise EnterpriseProjectAdoptionError(
            "enterprise runtime evidence is incomplete or invalid"
        ) from exc


def _candidate_rollout_report(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
) -> EnterpriseCandidateKServeAcceptanceReport:
    asset = _asset_by_role(
        report,
        "DPO_TTS_EMBEDDING_RELEASE_ROLLOUT_VERIFIED",
        runtime_eligible=False,
    )
    _read_asset_json(root, asset)
    return verify_enterprise_candidate_kserve_acceptance(root, Path(asset.source_path))


def _candidate_rollout_component(
    report: EnterpriseProjectAdoptionReport,
    rollout: EnterpriseCandidateKServeAcceptanceReport | None,
    *,
    component: Literal["LLM", "TTS", "EMBEDDING"],
    candidate_experiment_id: str,
    source_evidence_chain_sha256: str,
    source_path: str,
) -> tuple[AdoptedEnterpriseAsset | None, ComponentAcceptance | None]:
    if rollout is None:
        return None, None
    asset = _asset_by_role(
        report,
        "DPO_TTS_EMBEDDING_RELEASE_ROLLOUT_VERIFIED",
        runtime_eligible=False,
    )
    matches = [item for item in rollout.components if item.component == component]
    if len(matches) != 1:
        raise EnterpriseProjectAdoptionError(
            "enterprise candidate rollout component is missing"
        )
    item = matches[0]
    release = item.model_release
    expected_method = {"LLM": "DPO", "TTS": "TTS", "EMBEDDING": "EMBEDDING"}[
        component
    ]
    if (
        item.method != expected_method
        or item.source.path != source_path
        or item.source.candidate_experiment_id != candidate_experiment_id
        or item.source.source_evidence_chain_sha256
        != source_evidence_chain_sha256
        or item.runtime.actual_gpu_execution is not True
        or item.runtime.actual_model_inference is not True
        or item.runtime.model_inference_simulated is not False
        or item.kubernetes.actual_kserve_execution is not True
        or item.quality.candidate_quality_gate_passed is not True
        or release.independent_approval is not True
        or release.final_release_status != "ROLLED_BACK"
        or release.final_deployment_stage != "ROLLED_BACK"
        or release.final_traffic_percent != 0.0
        or tuple(stage.stage for stage in item.stages)
        != ("SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK")
        or item.stages[-1].candidate_request_delta != 0
        or release.active_production_alias_created is not False
    ):
        raise EnterpriseProjectAdoptionError(
            "enterprise candidate rollout binding changed"
        )
    return asset, item


def _preferred_llm_evidence(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
    candidate_rollout: EnterpriseCandidateKServeAcceptanceReport | None = None,
) -> EnterpriseRuntimeEvidence:
    if any(
        item.role == "AUTHORIZED_ENTERPRISE_DPO_MODEL_RELEASE_CANDIDATE"
        for item in report.active_assets
    ):
        return _dpo_llm_evidence(root, report, candidate_rollout)
    if any(
        item.role == "AUTHORIZED_ENTERPRISE_PPO_MODEL_RELEASE_CANDIDATE"
        for item in report.active_assets
    ):
        return _ppo_llm_evidence(root, report)
    if any(
        item.role == "AUTHORIZED_ENTERPRISE_GRPO_MODEL_RELEASE_CANDIDATE"
        for item in report.active_assets
    ):
        return _grpo_llm_evidence(root, report)
    return _llm_evidence(root, report)


def _dpo_llm_evidence(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
    candidate_rollout: EnterpriseCandidateKServeAcceptanceReport | None = None,
) -> EnterpriseRuntimeEvidence:
    asset = _asset_by_role(
        report,
        "AUTHORIZED_ENTERPRISE_DPO_MODEL_RELEASE_CANDIDATE",
        runtime_eligible=False,
    )
    _read_asset_json(root, asset)
    source = verify_dpo_structured_citation_erratum(root, Path(asset.source_path))
    if (
        source.status
        != "DPO_STRUCTURED_ENTERPRISE_VALUE_ACCEPTED_AFTER_CITATION_FIXTURE_ERRATUM"
        or source.decision != "ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or source.candidate_accepted is not True
        or source.release_draft_eligible is not True
        or source.formal_model_release_created is not False
        or source.runtime_eligible is not False
        or source.formal_gold_model_evaluation_replayed is not False
        or source.adapter_mutated is not False
        or source.model_output_mutated is not False
        or not all(source.corrected_hard_gates.values())
        or source.corrected_failed_hard_gates
    ):
        raise EnterpriseProjectAdoptionError("DPO candidate evidence is not draft eligible")
    rollout_asset, rollout_component = _candidate_rollout_component(
        report,
        candidate_rollout,
        component="LLM",
        candidate_experiment_id=source.run_id,
        source_evidence_chain_sha256=source.evidence_chain_sha256,
        source_path=asset.source_path,
    )
    return EnterpriseRuntimeEvidence(
        component="LLM",
        source_paths=(asset.source_path,),
        candidate_experiment_id=source.run_id,
        evaluation_reference=(f"dpo-structured-evaluation-{source.evidence_chain_sha256[:24]}"),
        model_alias_hint="industrial-agent-dpo",
        evidence_stage=(
            "ROLLED_BACK"
            if rollout_component is not None
            else "MODEL_RELEASE_DRAFT_ELIGIBLE"
        ),
        shadow_verified=rollout_component is not None,
        canary_verified=rollout_component is not None,
        rollback_verified=rollout_component is not None,
        evidence_chain_sha256=source.evidence_chain_sha256,
        rollout_source_path=(rollout_asset.source_path if rollout_asset is not None else None),
        rollout_evidence_chain_sha256=(
            candidate_rollout.evidence_chain_sha256
            if rollout_component is not None and candidate_rollout is not None
            else None
        ),
        rollout_model_release_id=(
            rollout_component.model_release.release_id
            if rollout_component is not None
            else None
        ),
    )


def _ppo_llm_evidence(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
) -> EnterpriseRuntimeEvidence:
    runtime_asset = _asset_by_role(
        report,
        "AUTHORIZED_ENTERPRISE_PPO_MODEL_RELEASE_CANDIDATE",
        runtime_eligible=False,
    )
    training_asset = _asset_by_role(
        report,
        "PPO_POST_TRAINING_SOURCE_VALIDATED_AT_AGENT_RUNTIME",
        runtime_eligible=False,
    )
    rollout_asset = _asset_by_role(
        report,
        "PPO_KSERVE_RELEASE_ACCEPTANCE_VERIFIED",
        runtime_eligible=False,
    )
    _read_asset_json(root, runtime_asset)
    _read_asset_json(root, training_asset)
    _read_asset_json(root, rollout_asset)
    runtime = verify_ppo_agent_runtime_value(root, Path(runtime_asset.source_path))
    training = verify_ppo_research_safety(root, Path(training_asset.source_path))
    rollout = verify_ppo_kserve_rollout(root, Path(rollout_asset.source_path))
    stages = {item.stage: item for item in rollout.stages}
    if (
        runtime.ppo_source.acceptance.path != training_asset.source_path
        or runtime.ppo_source.acceptance.sha256 != training_asset.source_file_sha256
        or runtime.ppo_source.run_id != training.run_id
        or runtime.ppo_source.evidence_chain_sha256 != training.evidence_chain_sha256
        or runtime.ppo_source.policy_adapter_bundle_sha256 != training.policy_adapter.bundle_sha256
        or rollout.source.agent_runtime.path != runtime_asset.source_path
        or rollout.source.agent_runtime.sha256 != runtime_asset.source_file_sha256
        or rollout.source.agent_runtime_run_id != runtime.run_id
        or rollout.source.agent_runtime_evidence_chain_sha256 != runtime.evidence_chain_sha256
        or rollout.source.training.path != training_asset.source_path
        or rollout.source.training.sha256 != training_asset.source_file_sha256
        or rollout.source.training_run_id != training.run_id
        or rollout.source.training_evidence_chain_sha256 != training.evidence_chain_sha256
        or rollout.source.policy_adapter_bundle_sha256 != training.policy_adapter.bundle_sha256
        or rollout.model_release.source_candidate_experiment_id != training.run_id
        or rollout.model_release.independent_approval_verified is not True
        or rollout.model_release.final_release_status != "ROLLED_BACK"
        or rollout.model_release.final_deployment_stage != "ROLLED_BACK"
        or set(stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
        or stages["SHADOW"].candidate_mirror_observed is not True
        or stages["CANARY_5"].passed is not True
        or stages["CANARY_25"].passed is not True
        or stages["ROLLED_BACK"].candidate_request_delta != 0
        or rollout.cleanup.gpu_worker_stopped is not True
        or rollout.cleanup.ppo_kubernetes_resources_removed is not True
        or rollout.cleanup.cluster_stopped_after_acceptance is not True
    ):
        raise EnterpriseProjectAdoptionError(
            "PPO Agent Runtime and ModelRelease rollout evidence binding changed"
        )
    return EnterpriseRuntimeEvidence(
        component="LLM",
        source_paths=(
            runtime_asset.source_path,
            training_asset.source_path,
            rollout_asset.source_path,
        ),
        candidate_experiment_id=training.run_id,
        evaluation_reference=f"ppo-agent-evaluation-{runtime.evidence_chain_sha256[:24]}",
        model_alias_hint="industrial-agent-ppo",
        evidence_stage="ROLLED_BACK",
        shadow_verified=stages["SHADOW"].candidate_mirror_observed,
        canary_verified=stages["CANARY_5"].passed and stages["CANARY_25"].passed,
        rollback_verified=stages["ROLLED_BACK"].passed,
        evidence_chain_sha256=rollout.evidence_chain_sha256,
    )


def _grpo_llm_evidence(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
) -> EnterpriseRuntimeEvidence:
    runtime_asset = _asset_by_role(
        report,
        "AUTHORIZED_ENTERPRISE_GRPO_MODEL_RELEASE_CANDIDATE",
        runtime_eligible=False,
    )
    training_asset = _asset_by_role(
        report,
        "GRPO_POST_TRAINING_SOURCE_VALIDATED_AT_AGENT_RUNTIME",
        runtime_eligible=False,
    )
    runtime = _GrpoAgentRuntimeEvidence.model_validate(_read_asset_json(root, runtime_asset))
    training = _GrpoTrainingEvidence.model_validate(_read_asset_json(root, training_asset))
    baseline_direct_object_rate = runtime.evaluation.baseline.get("direct_object_rate")
    if (
        runtime.grpo_source.acceptance.path != training_asset.source_path
        or runtime.grpo_source.acceptance.sha256 != training_asset.source_file_sha256
        or runtime.grpo_source.run_id != training.run_id
        or runtime.grpo_source.evidence_chain_sha256 != training.evidence_chain_sha256
        or runtime.grpo_source.adapter_bundle_sha256 != training.adapter.bundle_sha256
        or runtime.grpo_source.reward_profile_version != training.reward_profile.version
        or runtime.grpo_source.reward_profile_digest != training.reward_profile.digest
        or baseline_direct_object_rate != 0.0
        or not runtime.hard_gates
        or not runtime.runtime_boundaries.gate_results
    ):
        raise EnterpriseProjectAdoptionError(
            "GRPO Agent Runtime and training evidence binding changed"
        )
    rollout_assets = [
        item
        for item in report.active_assets
        if item.role == "GRPO_KSERVE_RELEASE_ACCEPTANCE_VERIFIED"
    ]
    if not rollout_assets:
        return EnterpriseRuntimeEvidence(
            component="LLM",
            source_paths=(runtime_asset.source_path, training_asset.source_path),
            candidate_experiment_id=training.run_id,
            evaluation_reference=(f"grpo-agent-evaluation-{runtime.evidence_chain_sha256[:24]}"),
            model_alias_hint="industrial-agent-grpo",
            evidence_stage="MODEL_RELEASE_DRAFT_ELIGIBLE",
            shadow_verified=False,
            canary_verified=False,
            rollback_verified=False,
            evidence_chain_sha256=runtime.evidence_chain_sha256,
        )
    rollout_asset = _asset_by_role(
        report,
        "GRPO_KSERVE_RELEASE_ACCEPTANCE_VERIFIED",
        runtime_eligible=False,
    )
    rollout = verify_grpo_kserve_rollout(root, Path(rollout_asset.source_path))
    stages = {item.stage: item for item in rollout.stages}
    if (
        rollout.source.agent_runtime.path != runtime_asset.source_path
        or rollout.source.agent_runtime.sha256 != runtime_asset.source_file_sha256
        or rollout.source.agent_runtime_run_id != runtime.run_id
        or rollout.source.agent_runtime_evidence_chain_sha256 != runtime.evidence_chain_sha256
        or rollout.source.training.path != training_asset.source_path
        or rollout.source.training.sha256 != training_asset.source_file_sha256
        or rollout.source.training_run_id != training.run_id
        or rollout.source.training_evidence_chain_sha256 != training.evidence_chain_sha256
        or rollout.source.adapter_bundle_sha256 != training.adapter.bundle_sha256
        or rollout.model_release.source_candidate_experiment_id != training.run_id
        or rollout.model_release.final_release_status != "ROLLED_BACK"
        or rollout.model_release.final_deployment_stage != "ROLLED_BACK"
        or rollout.model_release.independent_approval_verified is not True
        or set(stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
    ):
        raise EnterpriseProjectAdoptionError(
            "GRPO Agent Runtime and ModelRelease rollout evidence binding changed"
        )
    return EnterpriseRuntimeEvidence(
        component="LLM",
        source_paths=(
            runtime_asset.source_path,
            training_asset.source_path,
            rollout_asset.source_path,
        ),
        candidate_experiment_id=training.run_id,
        evaluation_reference=(f"grpo-agent-evaluation-{runtime.evidence_chain_sha256[:24]}"),
        model_alias_hint="industrial-agent-grpo",
        evidence_stage="ROLLED_BACK",
        shadow_verified=stages["SHADOW"].candidate_mirror_observed,
        canary_verified=(stages["CANARY_5"].passed and stages["CANARY_25"].passed),
        rollback_verified=stages["ROLLED_BACK"].passed,
        evidence_chain_sha256=rollout.evidence_chain_sha256,
    )


def _llm_evidence(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
) -> EnterpriseRuntimeEvidence:
    asset = _asset_by_role(report, "ACTIVE_ENTERPRISE_LLM_MODEL_AND_SERVING")
    source = _LlmEvidence.model_validate(_read_asset_json(root, asset))
    rollout_path = _linked_path(asset.source_path, source.evidence.rollout.path)
    rollout = _LlmRollout.model_validate(_read_json(root, rollout_path))
    if (
        rollout.evidence_chain_sha256 != source.evidence.rollout.evidence_chain_sha256
        or rollout.selected_experiment_id != source.candidate.experiment_id
        or rollout.evaluation_id != source.candidate.evaluation_id
    ):
        raise EnterpriseProjectAdoptionError("LLM rollout evidence binding changed")
    required_stages = {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
    if not required_stages.issubset(rollout.rollout):
        raise EnterpriseProjectAdoptionError("LLM rollout stage evidence is incomplete")
    return EnterpriseRuntimeEvidence(
        component="LLM",
        source_paths=(asset.source_path, rollout_path),
        candidate_experiment_id=source.candidate.experiment_id,
        evaluation_reference=source.candidate.evaluation_id,
        evidence_stage="ROLLED_BACK",
        shadow_verified=True,
        canary_verified=True,
        rollback_verified=True,
        evidence_chain_sha256=source.evidence_chain_sha256,
    )


def _vlm_evidence(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
) -> EnterpriseRuntimeEvidence:
    adoption = _asset_by_role(report, "ACTIVE_ENTERPRISE_VLM_MODEL_AND_SERVING")
    source_model = _asset_by_role(report, "ACTIVE_ENTERPRISE_VLM_MODEL_SOURCE")
    serving = _asset_by_role(report, "ACTIVE_ENTERPRISE_VLM_KSERVE")
    source = _VlmEvidence.model_validate(_read_asset_json(root, adoption))
    rollout = _VlmRollout.model_validate(_read_asset_json(root, serving))
    if (
        source.candidate.source_acceptance_path != source_model.source_path
        or source.candidate.source_acceptance_sha256 != source_model.source_file_sha256
        or rollout.release.source_adoption_path != adoption.source_path
        or rollout.release.source_adoption_sha256 != adoption.source_file_sha256
        or rollout.release.source_evidence_chain_sha256 != source.evidence_chain_sha256
        or rollout.release.candidate_experiment_id != source.candidate.experiment_id
        or rollout.release.evaluation_suite_id != source.candidate.evaluation_suite_id
        or rollout.release.adapter_bundle_sha256 != source.candidate.artifact_sha256
    ):
        raise EnterpriseProjectAdoptionError("VLM source and rollout binding changed")
    stages = {item.stage: item for item in rollout.stages}
    if (
        len(rollout.stages) != 4
        or set(stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
        or not stages["SHADOW"].candidate_mirror_observed
        or stages["ROLLED_BACK"].candidate_request_delta != 0
        or stages["ROLLED_BACK"].candidate_response_count != 0
    ):
        raise EnterpriseProjectAdoptionError("VLM rollout stage evidence is incomplete")
    return EnterpriseRuntimeEvidence(
        component="VLM",
        source_paths=(
            adoption.source_path,
            source_model.source_path,
            serving.source_path,
        ),
        candidate_experiment_id=source.candidate.experiment_id,
        evaluation_reference=source.candidate.evaluation_suite_id,
        model_alias_hint=source.runtime.model_alias,
        evidence_stage="ROLLED_BACK",
        shadow_verified=rollout.kserve.kserve_shadow_endpoint_verified,
        canary_verified=stages["CANARY_25"].passed,
        rollback_verified=stages["ROLLED_BACK"].passed,
        evidence_chain_sha256=rollout.evidence_chain_sha256,
    )


def _rul_evidence(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
) -> EnterpriseRuntimeEvidence:
    model_asset = _asset_by_role(report, "ACTIVE_ENTERPRISE_RUL_MODEL")
    serving_asset = _asset_by_role(report, "ACTIVE_ENTERPRISE_RUL_KSERVE")
    promotion = _RulPromotion.model_validate(_read_asset_json(root, model_asset))
    rollout = _RulRollout.model_validate(_read_asset_json(root, serving_asset))
    if rollout.promotion_evidence.evidence_chain_sha256 != promotion.evidence_chain_sha256:
        raise EnterpriseProjectAdoptionError("RUL promotion and rollout evidence diverged")
    return EnterpriseRuntimeEvidence(
        component="RUL",
        source_paths=(model_asset.source_path, serving_asset.source_path),
        candidate_experiment_id=promotion.candidate.experiment_id,
        evaluation_reference=rollout.promotion_evidence.gold_snapshot_sha256,
        evidence_stage=rollout.route.stage,
        shadow_verified=rollout.stages.shadow.passed,
        canary_verified=rollout.stages.canary_25.passed,
        rollback_verified=rollout.stages.rollback.passed,
        evidence_chain_sha256=rollout.evidence_chain_sha256,
    )


def _asr_evidence(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
) -> EnterpriseRuntimeEvidence:
    asset = _asset_by_role(
        report,
        "AUTHORIZED_ENTERPRISE_ASR_EVALUATION_CANDIDATE",
        runtime_eligible=False,
    )
    _read_asset_json(root, asset)
    source = verify_asr_enterprise_value(root, Path(asset.source_path))
    if (
        source.formal_model_release_created is not False
        or source.runtime_eligible is not False
        or source.runtime.actual_gpu_execution is not True
        or source.runtime.model_training_simulated is not False
        or not all(source.hard_gates.model_dump().values())
    ):
        raise EnterpriseProjectAdoptionError("ASR candidate evidence is not draft eligible")
    rollout_assets = [
        item
        for item in report.active_assets
        if item.role == "ASR_KSERVE_RELEASE_ACCEPTANCE_VERIFIED"
    ]
    if not rollout_assets:
        return EnterpriseRuntimeEvidence(
            component="ASR",
            source_paths=(asset.source_path,),
            candidate_experiment_id=source.run_id,
            evaluation_reference=f"asr-evaluation-{source.evidence_chain_sha256[:24]}",
            model_alias_hint="industrial-asr-whisper",
            evidence_stage="MODEL_RELEASE_DRAFT_ELIGIBLE",
            shadow_verified=False,
            canary_verified=False,
            rollback_verified=False,
            evidence_chain_sha256=source.evidence_chain_sha256,
        )
    rollout_asset = _asset_by_role(
        report,
        "ASR_KSERVE_RELEASE_ACCEPTANCE_VERIFIED",
        runtime_eligible=False,
    )
    rollout = verify_asr_kserve_acceptance(root, Path(rollout_asset.source_path))
    stages = {item.stage: item for item in rollout.stages}
    if (
        rollout.source.training_acceptance.path != asset.source_path
        or rollout.source.training_acceptance.sha256 != asset.source_file_sha256
        or rollout.source.training_run_id != source.run_id
        or rollout.source.training_evidence_chain_sha256 != source.evidence_chain_sha256
        or rollout.source.adapter_bundle_sha256 != source.adapter.bundle_sha256
        or rollout.model_release.source_candidate_experiment_id != source.run_id
        or rollout.model_release.final_release_status != "ROLLED_BACK"
        or rollout.model_release.final_deployment_stage != "ROLLED_BACK"
        or rollout.model_release.final_traffic_percent != 0.0
        or rollout.model_release.independent_approval_verified is not True
        or set(stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
        or not stages["SHADOW"].candidate_mirror_observed
        or not stages["CANARY_5"].passed
        or not stages["CANARY_25"].passed
        or stages["ROLLED_BACK"].candidate_request_delta != 0
        or stages["ROLLED_BACK"].candidate_response_count != 0
        or rollout.cleanup.gpu_worker_stopped is not True
        or rollout.cleanup.port_forwards_stopped is not True
        or rollout.cleanup.asr_kubernetes_resources_removed is not True
        or rollout.cleanup.cluster_stopped_after_acceptance is not True
    ):
        raise EnterpriseProjectAdoptionError(
            "ASR training and ModelRelease rollout evidence binding changed"
        )
    return EnterpriseRuntimeEvidence(
        component="ASR",
        source_paths=(asset.source_path,),
        candidate_experiment_id=source.run_id,
        evaluation_reference=f"asr-evaluation-{source.evidence_chain_sha256[:24]}",
        model_alias_hint="industrial-asr-whisper",
        evidence_stage="ROLLED_BACK",
        shadow_verified=True,
        canary_verified=True,
        rollback_verified=True,
        evidence_chain_sha256=source.evidence_chain_sha256,
        rollout_source_path=rollout_asset.source_path,
        rollout_evidence_chain_sha256=rollout.evidence_chain_sha256,
        rollout_model_release_id=rollout.model_release.release_id,
    )


def _tts_evidence(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
    candidate_rollout: EnterpriseCandidateKServeAcceptanceReport | None = None,
) -> EnterpriseRuntimeEvidence:
    asset = _asset_by_role(
        report,
        "AUTHORIZED_ENTERPRISE_TTS_MODEL_RELEASE_CANDIDATE",
        runtime_eligible=False,
    )
    _read_asset_json(root, asset)
    source = verify_calibrated_tts_value(root, Path(asset.source_path))
    if (
        source.status != "TTS_STRONG_ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        or source.decision != "TTS_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or source.candidate_accepted is not True
        or source.release_draft_eligible is not True
        or source.formal_model_release_created is not False
        or source.runtime_eligible is not False
        or source.runtime.actual_gpu_execution is not True
        or source.runtime.actual_gpu_inference is not True
        or source.runtime.current_run_model_training_performed is not False
        or source.runtime.predecessor_actual_gpu_training is not True
        or source.runtime.model_training_simulated is not False
        or not all(source.hard_gates.values())
        or source.failed_hard_gates
    ):
        raise EnterpriseProjectAdoptionError("TTS candidate evidence is not draft eligible")
    rollout_asset, rollout_component = _candidate_rollout_component(
        report,
        candidate_rollout,
        component="TTS",
        candidate_experiment_id=source.run_id,
        source_evidence_chain_sha256=source.evidence_chain_sha256,
        source_path=asset.source_path,
    )
    return EnterpriseRuntimeEvidence(
        component="TTS",
        source_paths=(asset.source_path,),
        candidate_experiment_id=source.run_id,
        evaluation_reference=f"tts-evaluation-{source.evidence_chain_sha256[:24]}",
        model_alias_hint="industrial-tts-speecht5",
        evidence_stage=(
            "ROLLED_BACK"
            if rollout_component is not None
            else "MODEL_RELEASE_DRAFT_ELIGIBLE"
        ),
        shadow_verified=rollout_component is not None,
        canary_verified=rollout_component is not None,
        rollback_verified=rollout_component is not None,
        evidence_chain_sha256=source.evidence_chain_sha256,
        rollout_source_path=(rollout_asset.source_path if rollout_asset is not None else None),
        rollout_evidence_chain_sha256=(
            candidate_rollout.evidence_chain_sha256
            if rollout_component is not None and candidate_rollout is not None
            else None
        ),
        rollout_model_release_id=(
            rollout_component.model_release.release_id
            if rollout_component is not None
            else None
        ),
    )


def _embedding_evidence(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
    candidate_rollout: EnterpriseCandidateKServeAcceptanceReport | None = None,
) -> EnterpriseRuntimeEvidence:
    asset = _asset_by_role(
        report,
        "AUTHORIZED_ENTERPRISE_EMBEDDING_MODEL_RELEASE_CANDIDATE",
        runtime_eligible=False,
    )
    _read_asset_json(root, asset)
    source = verify_calibrated_embedding_value(root, Path(asset.source_path))
    if (
        source.status != "EMBEDDING_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        or source.decision != "EMBEDDING_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or source.candidate_accepted is not True
        or source.release_draft_eligible is not True
        or source.formal_model_release_created is not False
        or source.runtime_eligible is not False
        or source.runtime.actual_gpu_execution is not True
        or source.runtime.actual_gpu_inference is not True
        or source.runtime.current_run_model_training_performed is not False
        or source.runtime.source_candidate_actual_gpu_trained is not True
        or source.runtime.model_training_simulated is not False
        or not all(source.hard_gates.values())
        or source.failed_hard_gates
    ):
        raise EnterpriseProjectAdoptionError("Embedding candidate evidence is not draft eligible")
    rollout_asset, rollout_component = _candidate_rollout_component(
        report,
        candidate_rollout,
        component="EMBEDDING",
        candidate_experiment_id=source.run_id,
        source_evidence_chain_sha256=source.evidence_chain_sha256,
        source_path=asset.source_path,
    )
    return EnterpriseRuntimeEvidence(
        component="EMBEDDING",
        source_paths=(asset.source_path,),
        candidate_experiment_id=source.run_id,
        evaluation_reference=(f"embedding-evaluation-{source.evidence_chain_sha256[:24]}"),
        model_alias_hint="industrial-embedding-calibrated",
        evidence_stage=(
            "ROLLED_BACK"
            if rollout_component is not None
            else "MODEL_RELEASE_DRAFT_ELIGIBLE"
        ),
        shadow_verified=rollout_component is not None,
        canary_verified=rollout_component is not None,
        rollback_verified=rollout_component is not None,
        evidence_chain_sha256=source.evidence_chain_sha256,
        rollout_source_path=(rollout_asset.source_path if rollout_asset is not None else None),
        rollout_evidence_chain_sha256=(
            candidate_rollout.evidence_chain_sha256
            if rollout_component is not None and candidate_rollout is not None
            else None
        ),
        rollout_model_release_id=(
            rollout_component.model_release.release_id
            if rollout_component is not None
            else None
        ),
    )


def _reranker_evidence(
    root: Path,
    report: EnterpriseProjectAdoptionReport,
) -> EnterpriseRuntimeEvidence:
    asset = _asset_by_role(
        report,
        "AUTHORIZED_ENTERPRISE_RERANKER_MODEL_RELEASE_CANDIDATE",
        runtime_eligible=False,
    )
    _read_asset_json(root, asset)
    source = verify_calibrated_reranker_value(root, Path(asset.source_path))
    if (
        source.status != "RERANKER_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        or source.decision != "RERANKER_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or source.candidate_accepted is not True
        or source.formal_model_release_created is not False
        or source.runtime_eligible is not False
        or source.runtime.actual_gpu_execution is not True
        or source.runtime.actual_gpu_inference is not True
        or source.runtime.current_run_model_training_performed is not False
        or source.runtime.source_candidate_actual_gpu_trained is not True
        or source.runtime.model_training_simulated is not False
        or not all(source.hard_gates.model_dump().values())
        or source.failed_hard_gates
    ):
        raise EnterpriseProjectAdoptionError("Reranker candidate evidence is not draft eligible")
    return EnterpriseRuntimeEvidence(
        component="RERANKER",
        source_paths=(asset.source_path,),
        candidate_experiment_id=source.run_id,
        evaluation_reference=(f"reranker-evaluation-{source.evidence_chain_sha256[:24]}"),
        model_alias_hint="industrial-reranker-calibrated",
        evidence_stage="MODEL_RELEASE_DRAFT_ELIGIBLE",
        shadow_verified=False,
        canary_verified=False,
        rollback_verified=False,
        evidence_chain_sha256=source.evidence_chain_sha256,
    )


def _asset_by_role(
    report: EnterpriseProjectAdoptionReport,
    role: str,
    *,
    runtime_eligible: bool = True,
) -> AdoptedEnterpriseAsset:
    matches = [item for item in report.active_assets if item.role == role]
    if len(matches) != 1 or matches[0].runtime_eligible is not runtime_eligible:
        raise EnterpriseProjectAdoptionError(f"runtime asset role is invalid: {role}")
    return matches[0]


def _read_asset_json(root: Path, asset: AdoptedEnterpriseAsset) -> dict[str, object]:
    payload = _read_bytes(root, asset.source_path)
    if sha256(payload).hexdigest() != asset.source_file_sha256:
        raise EnterpriseProjectAdoptionError(f"runtime asset digest changed: {asset.source_path}")
    return _decode_json(payload, asset.source_path)


def _read_json(root: Path, relative_path: str) -> dict[str, object]:
    return _decode_json(_read_bytes(root, relative_path), relative_path)


def _read_bytes(root: Path, relative_path: str) -> bytes:
    requested = Path(relative_path)
    if requested.is_absolute():
        raise EnterpriseProjectAdoptionError("runtime evidence path must be relative")
    target = (root / requested).resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise EnterpriseProjectAdoptionError("runtime evidence escaped repository root") from exc
    if not target.is_file():
        raise EnterpriseProjectAdoptionError("runtime evidence is not a file")
    return target.read_bytes()


def _decode_json(payload: bytes, source_path: str) -> dict[str, object]:
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise EnterpriseProjectAdoptionError(
            f"runtime evidence root must be an object: {source_path}"
        )
    return cast(dict[str, object], decoded)


def _linked_path(source_path: str, linked_path: str) -> str:
    linked = Path(linked_path)
    if linked.is_absolute():
        raise EnterpriseProjectAdoptionError("linked runtime evidence path must be relative")
    return (Path(source_path).parent / linked).as_posix()
