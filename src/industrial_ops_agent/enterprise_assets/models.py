"""Typed projections for enterprise model evidence and runtime registration state."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

EnterpriseModelComponent = Literal[
    "LLM",
    "VLM",
    "ASR",
    "TTS",
    "RUL",
    "EMBEDDING",
    "RERANKER",
]
EnterpriseDeploymentAction = Literal[
    "REQUEST_MODEL_DEPLOYMENT",
    "PROMOTE_MODEL_RELEASE",
    "ROLLBACK_MODEL_RELEASE",
]
EnterpriseDeploymentNextAction = Literal[
    "WAIT_FOR_APPROVAL",
    "REQUEST_SHADOW",
    "WAIT_FOR_CONTROLLER",
    "COLLECT_STAGE_OBSERVATION",
    "PROMOTE_NEXT_STAGE",
    "OPERATE_PRODUCTION",
    "REMEDIATE_DEPLOYMENT",
    "REVIEW_ROLLBACK",
]
EnterpriseReleaseAutomationStatus = Literal[
    "WAITING_APPROVAL",
    "SHADOW_REQUESTED",
    "CANCELLED",
    "BLOCKED",
    "DISABLED",
]


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EnterpriseRuntimeEvidence(_ClosedModel):
    component: EnterpriseModelComponent
    source_paths: tuple[str, ...] = Field(min_length=1)
    candidate_experiment_id: str = Field(min_length=1)
    evaluation_reference: str | None = None
    model_alias_hint: str | None = None
    evidence_stage: str = Field(min_length=1)
    shadow_verified: bool
    canary_verified: bool
    rollback_verified: bool
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rollout_source_path: str | None = Field(default=None, min_length=1)
    rollout_evidence_chain_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    rollout_model_release_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_complete_rollout_binding(self) -> Self:
        rollout_values = (
            self.rollout_source_path,
            self.rollout_evidence_chain_sha256,
            self.rollout_model_release_id,
        )
        if any(value is not None for value in rollout_values) and not all(
            value is not None for value in rollout_values
        ):
            raise ValueError("rollout evidence binding must be complete")
        return self


class EnterpriseDeploymentBinding(_ClosedModel):
    deployment_id: str
    provider: str
    status: str
    desired_stage: str
    current_stage: str
    desired_traffic_percent: float = Field(ge=0, le=100)
    observed_traffic_percent: float = Field(ge=0, le=100)
    failure_reason: str | None
    version: int = Field(ge=1)
    latest_observation_stage: str | None
    latest_observation_decision: str | None


class EnterpriseAliasBinding(_ClosedModel):
    alias: str
    status: str
    runtime_profile: str
    version: int = Field(ge=1)


class EnterpriseReleaseAutomationBinding(_ClosedModel):
    onboarding_id: str
    baseline_release_id: str
    auto_shadow_enabled: bool
    automation_status: EnterpriseReleaseAutomationStatus
    attempt_count: int = Field(ge=0)
    last_error: str | None
    last_attempt_at: datetime | None
    version: int = Field(ge=1)


class EnterpriseModelAssetImport(_ClosedModel):
    import_id: str
    component: EnterpriseModelComponent
    source_candidate_experiment_id: str
    imported_candidate_experiment_id: str
    imported_evaluation_id: str
    imported_suite_id: str
    source_paths: tuple[str, ...]
    source_file_hashes: dict[str, str]
    source_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_classification: str
    source_decision: str
    operational_classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"]
    actual_sample_count: int = Field(ge=1)
    source_artifact_size_recorded: bool
    release_scope: Literal["STAGING_ONLY"]
    import_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["ACTIVE", "REVOKED"]
    version: int = Field(ge=1)
    created_at: datetime


class EnterpriseReleaseRuntimeRegistration(_ClosedModel):
    release_id: str
    status: str
    target_environment: str
    manifest_hash: str
    rollback_release_id: str | None
    traffic_percent: float = Field(ge=0, le=100)
    failure_reason: str | None
    version: int = Field(ge=1)
    created_at: datetime
    deployment: EnterpriseDeploymentBinding | None
    aliases: tuple[EnterpriseAliasBinding, ...]
    automation: EnterpriseReleaseAutomationBinding | None


class EnterpriseRuntimeBinding(_ClosedModel):
    evidence: EnterpriseRuntimeEvidence
    import_state: Literal["IMPORTED", "NOT_IMPORTED"] = "NOT_IMPORTED"
    model_import: EnterpriseModelAssetImport | None = None
    registry_state: Literal["REGISTERED", "NOT_REGISTERED"]
    deployment_state: Literal["DEPLOYMENT_RECORDED", "NOT_DEPLOYED"]
    alias_state: Literal["ACTIVE_ALIAS_BOUND", "NO_ACTIVE_ALIAS"]
    registrations: tuple[EnterpriseReleaseRuntimeRegistration, ...]


class EnterpriseReleaseDraftBaseline(_ClosedModel):
    release_id: str
    status: str
    target_environment: str
    candidate_experiment_id: str
    manifest_hash: str
    version: int = Field(ge=1)
    created_at: datetime


class EnterpriseReleaseDraftPreview(_ClosedModel):
    evidence: EnterpriseRuntimeEvidence
    import_state: Literal["IMPORTED", "NOT_IMPORTED"] = "NOT_IMPORTED"
    model_import: EnterpriseModelAssetImport | None = None
    eligible: bool
    blockers: tuple[str, ...]
    evaluation_id: str | None
    suggested_supply_chain_evidence_id: str | None
    baselines: tuple[EnterpriseReleaseDraftBaseline, ...]
    target_environment: Literal["STAGING"] = "STAGING"
    auto_shadow_supported: bool = True


class EnterpriseStagingBaselineDraft(_ClosedModel):
    release_id: str
    status: str
    version: int = Field(ge=1)
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_environment: Literal["STAGING"] = "STAGING"
    operational_classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"]
    source: Literal["PROJECT_GENERATED_ENTERPRISE_STAGING_BASELINE"]
    bootstrap_schema_version: Literal["enterprise-staging-release-baseline/v1"]
    approval_status: str | None
    approval_required: Literal[True] = True
    created: bool
    next_action: Literal[
        "VALIDATE_RELEASE",
        "SUBMIT_INDEPENDENT_APPROVAL",
        "WAIT_INDEPENDENT_APPROVAL",
        "READY_FOR_CANDIDATE_DRAFT",
        "REMEDIATE_RELEASE",
    ]
    component_candidate_ids: dict[str, str]


class EnterpriseReleaseOnboarding(_ClosedModel):
    onboarding_id: str
    release_id: str
    release_status: str
    release_version: int = Field(ge=1)
    baseline_release_id: str
    component: EnterpriseModelComponent
    candidate_experiment_id: str
    evaluation_id: str
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    auto_shadow_enabled: bool
    automation_status: EnterpriseReleaseAutomationStatus
    attempt_count: int = Field(ge=0)
    last_error: str | None
    last_attempt_at: datetime | None
    version: int = Field(ge=1)
    created_at: datetime


class EnterpriseCandidateReleaseBatch(_ClosedModel):
    batch_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_release_id: str
    target_environment: Literal["STAGING"] = "STAGING"
    operational_classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"]
    status: Literal["BATCH_REGISTERED"]
    releases: tuple[EnterpriseReleaseOnboarding, ...] = Field(
        min_length=3,
        max_length=3,
    )
    created_count: int = Field(ge=0, le=3)
    replayed_count: int = Field(ge=0, le=3)
    next_action: Literal["VALIDATE_AND_SUBMIT_INDEPENDENT_APPROVAL"]


class EnterpriseCandidateReleaseProgress(_ClosedModel):
    component: EnterpriseModelComponent
    onboarding_id: str
    release_id: str
    release_status: str
    release_version: int = Field(ge=1)
    approval_id: str | None
    approval_status: str | None
    approval_version: int | None = Field(default=None, ge=1)
    approval_requested_by_subject_id: str | None
    can_decide_approval: bool
    automation_status: EnterpriseReleaseAutomationStatus
    failure_reason: str | None
    deployment_id: str | None
    deployment_status: str | None
    deployment_version: int | None = Field(default=None, ge=1)
    current_stage: str | None
    desired_stage: str | None
    observed_traffic_percent: float | None = Field(default=None, ge=0, le=100)
    latest_observation_decision: str | None
    deployment_failure_reason: str | None
    deployment_legal_actions: tuple[EnterpriseDeploymentAction, ...]
    deployment_next_action: EnterpriseDeploymentNextAction
    next_action: Literal[
        "VALIDATE_RELEASE",
        "SUBMIT_APPROVAL",
        "WAIT_INDEPENDENT_APPROVAL",
        "REQUEST_SHADOW_IN_RELEASE_CENTER",
        "MONITOR_ROLLOUT",
        "REMEDIATE_RELEASE",
        "REVIEW_ROLLBACK",
    ]
    created_at: datetime


class EnterpriseCandidateReleaseBatchProgress(_ClosedModel):
    batch_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_release_id: str
    requested_by_subject_id: str
    target_environment: Literal["STAGING"] = "STAGING"
    operational_classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"]
    status: Literal[
        "DRAFT_REGISTRATION_INCOMPLETE",
        "DRAFTS_REGISTERED",
        "APPROVAL_SUBMISSION_IN_PROGRESS",
        "AWAITING_INDEPENDENT_APPROVAL",
        "APPROVED",
        "ROLLOUT_IN_PROGRESS",
        "REMEDIATION_REQUIRED",
        "ROLLED_BACK",
    ]
    components: tuple[EnterpriseCandidateReleaseProgress, ...] = Field(
        min_length=1,
        max_length=3,
    )
    registered_count: int = Field(ge=1, le=3)
    approval_pending_count: int = Field(ge=0, le=3)
    approved_count: int = Field(ge=0, le=3)
    decidable_count: int = Field(ge=0, le=3)
    rollout_count: int = Field(ge=0, le=3)
    deployment_requested_count: int = Field(ge=0, le=3)
    shadow_ready_count: int = Field(ge=0, le=3)
    canary_count: int = Field(ge=0, le=3)
    production_count: int = Field(ge=0, le=3)
    rolled_back_count: int = Field(ge=0, le=3)
    can_advance_to_approval: bool
    next_action: Literal[
        "RESUME_DRAFT_REGISTRATION",
        "VALIDATE_AND_SUBMIT_APPROVAL",
        "CONTINUE_VALIDATE_AND_SUBMIT_APPROVAL",
        "INDEPENDENT_APPROVER_DECISION",
        "REQUEST_SHADOW_IN_RELEASE_CENTER",
        "MONITOR_ROLLOUT",
        "REMEDIATE_RELEASE",
        "REVIEW_ROLLBACK",
    ]
    created_at: datetime
