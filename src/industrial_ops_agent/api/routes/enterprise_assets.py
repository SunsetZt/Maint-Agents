"""Read-only governance API for project-authorized enterprise staging assets."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Path, Query, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_enterprise_project_adoption_service,
    get_identity,
    get_tool_gateway,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.api.routes.model_deployments import DeploymentPlanBody
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.deployment.service import (
    DeploymentConflict,
    DeploymentNotVisible,
    DeploymentPlan,
)
from industrial_ops_agent.enterprise_assets import (
    EnterpriseCandidateReleaseBatch,
    EnterpriseCandidateReleaseBatchConflict,
    EnterpriseCandidateReleaseBatchProgress,
    EnterpriseCandidateReleaseBatchService,
    EnterpriseModelAssetImport,
    EnterpriseModelAssetImportConflict,
    EnterpriseModelAssetImportService,
    EnterpriseModelImportAcceptanceGovernanceError,
    EnterpriseModelImportAcceptanceGovernanceService,
    EnterpriseProjectAdoptionReader,
    EnterpriseProjectAssuranceGovernanceError,
    EnterpriseProjectAssuranceGovernanceService,
    EnterpriseProjectClosureGovernanceError,
    EnterpriseProjectClosureGovernanceService,
    EnterpriseReleaseDraftPreview,
    EnterpriseReleaseOnboarding,
    EnterpriseReleaseOnboardingConflict,
    EnterpriseReleaseOnboardingService,
    EnterpriseRuntimeBinding,
    EnterpriseRuntimeBindingService,
    EnterpriseStagingBaselineConflict,
    EnterpriseStagingBaselineDraft,
    EnterpriseStagingBaselineService,
    common_enterprise_model_import_baselines,
)
from industrial_ops_agent.enterprise_assets.acceptance import (
    EnterpriseModelImportAcceptanceReport,
)
from industrial_ops_agent.enterprise_assets.models import EnterpriseModelComponent
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.releases.service import ReleaseConflict, ReleaseNotVisible
from industrial_ops_agent.simulation.closure import SimulatedEnterpriseClosureReport
from industrial_ops_agent.simulation.enterprise_project_adoption import (
    EnterpriseProjectAdoptionError,
    EnterpriseProjectAdoptionReport,
)
from industrial_ops_agent.simulation.project_assurance_lab import (
    ProjectAssuranceLabReport,
)
from industrial_ops_agent.simulation.reranker_kserve_acceptance import (
    ACCEPTANCE_RELATIVE as RERANKER_KSERVE_ACCEPTANCE_RELATIVE,
)
from industrial_ops_agent.simulation.reranker_kserve_acceptance import (
    EnterpriseRerankerKServeAcceptanceReport,
    RerankerKServeAcceptanceError,
    verify_reranker_kserve_acceptance,
)
from industrial_ops_agent.tools.gateway import ToolGateway

router = APIRouter(tags=["enterprise-assets"])


class EnterpriseProjectAdoptionMeta(BaseModel):
    request_id: str
    active_assets: int = Field(ge=0)
    runtime_assets: int = Field(ge=0)
    experiment_history: int = Field(ge=0)


class EnterpriseProjectAdoptionEnvelope(BaseModel):
    data: EnterpriseProjectAdoptionReport
    meta: EnterpriseProjectAdoptionMeta


class EnterpriseModelImportAcceptanceMeta(BaseModel):
    request_id: str
    components: int = Field(ge=0)
    eligible_components: int = Field(ge=0)
    total_actual_samples: int = Field(ge=0)
    common_baseline_release_ids: tuple[str, ...]


class EnterpriseModelImportAcceptanceEnvelope(BaseModel):
    data: EnterpriseModelImportAcceptanceReport
    meta: EnterpriseModelImportAcceptanceMeta


class EnterpriseProjectClosureMeta(BaseModel):
    request_id: str
    coverage_domains: int = Field(ge=0)
    source_evidence: int = Field(ge=0)
    intentionally_unverified_methods: int = Field(ge=0)
    production_blockers: int = Field(ge=0)


class EnterpriseProjectClosureEnvelope(BaseModel):
    data: SimulatedEnterpriseClosureReport
    meta: EnterpriseProjectClosureMeta


class EnterpriseProjectAssuranceMeta(BaseModel):
    request_id: str
    security_scenarios: int = Field(ge=0)
    recovery_components: int = Field(ge=0)
    healthy_slos: int = Field(ge=0)
    staging_scenarios: int = Field(ge=0)
    signoffs: int = Field(ge=0)


class EnterpriseProjectAssuranceEnvelope(BaseModel):
    data: ProjectAssuranceLabReport
    meta: EnterpriseProjectAssuranceMeta


class EnterpriseRerankerKServeAcceptanceMeta(BaseModel):
    request_id: str
    stages: int = Field(ge=0)
    governed_requests: int = Field(ge=0)
    actual_gpu_execution: Literal[True] = True
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnterpriseRerankerKServeAcceptanceEnvelope(BaseModel):
    data: EnterpriseRerankerKServeAcceptanceReport
    meta: EnterpriseRerankerKServeAcceptanceMeta


class EnterpriseRuntimeBindingMeta(BaseModel):
    request_id: str
    components: int = Field(ge=0)
    registered_components: int = Field(ge=0)
    deployed_components: int = Field(ge=0)
    active_alias_components: int = Field(ge=0)


class EnterpriseRuntimeBindingEnvelope(BaseModel):
    data: list[EnterpriseRuntimeBinding]
    meta: EnterpriseRuntimeBindingMeta


class EnterpriseReleaseDraftBody(BaseModel):
    baseline_release_id: str = Field(min_length=1, max_length=128)
    target_environment: Literal["STAGING"] = "STAGING"
    auto_shadow_enabled: bool = True
    deployment_plan: DeploymentPlanBody | None = None


class EnterpriseReleaseDraftPreviewEnvelope(BaseModel):
    data: EnterpriseReleaseDraftPreview
    meta: dict[str, str]


class EnterpriseModelAssetImportEnvelope(BaseModel):
    data: EnterpriseModelAssetImport
    meta: dict[str, str]


class EnterpriseReleaseOnboardingEnvelope(BaseModel):
    data: EnterpriseReleaseOnboarding
    meta: dict[str, str]


class EnterpriseStagingBaselineDraftEnvelope(BaseModel):
    data: EnterpriseStagingBaselineDraft
    meta: dict[str, str]


class EnterpriseCandidateReleaseBatchEnvelope(BaseModel):
    data: EnterpriseCandidateReleaseBatch
    meta: dict[str, str]


class EnterpriseCandidateReleaseBatchProgressEnvelope(BaseModel):
    data: EnterpriseCandidateReleaseBatchProgress
    meta: dict[str, str]


class EnterpriseCandidateReleaseBatchProgressMeta(BaseModel):
    request_id: str
    batches: int = Field(ge=0)
    awaiting_approval: int = Field(ge=0)
    approved: int = Field(ge=0)


class EnterpriseCandidateReleaseBatchProgressListEnvelope(BaseModel):
    data: list[EnterpriseCandidateReleaseBatchProgress]
    meta: EnterpriseCandidateReleaseBatchProgressMeta


class EnterpriseCandidateReleaseReviewQueueMeta(BaseModel):
    request_id: str
    batches: int = Field(ge=0)
    pending_decisions: int = Field(ge=0)
    decidable_decisions: int = Field(ge=0)
    blocked_by_separation_of_duties: int = Field(ge=0)


class EnterpriseCandidateReleaseReviewQueueEnvelope(BaseModel):
    data: list[EnterpriseCandidateReleaseBatchProgress]
    meta: EnterpriseCandidateReleaseReviewQueueMeta


class EnterpriseCandidateReleaseDeploymentQueueMeta(BaseModel):
    request_id: str
    batches: int = Field(ge=0)
    deployable_components: int = Field(ge=0)
    active_deployments: int = Field(ge=0)
    promotable_components: int = Field(ge=0)
    rollback_available_components: int = Field(ge=0)


class EnterpriseCandidateReleaseDeploymentQueueEnvelope(BaseModel):
    data: list[EnterpriseCandidateReleaseBatchProgress]
    meta: EnterpriseCandidateReleaseDeploymentQueueMeta


@router.get(
    "/enterprise-assets/adoption",
    response_model=EnterpriseProjectAdoptionEnvelope,
)
async def get_enterprise_project_adoption(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
) -> EnterpriseProjectAdoptionEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        report = await asyncio.to_thread(service.snapshot)
    except EnterpriseProjectAdoptionError as exc:
        raise _unavailable() from exc
    return EnterpriseProjectAdoptionEnvelope(
        data=report,
        meta=EnterpriseProjectAdoptionMeta(
            request_id=request.state.request_id,
            active_assets=len(report.active_assets),
            runtime_assets=sum(item.runtime_eligible for item in report.active_assets),
            experiment_history=len(report.experiment_history),
        ),
    )


@router.get(
    "/enterprise-assets/model-import-acceptance",
    response_model=EnterpriseModelImportAcceptanceEnvelope,
)
async def get_enterprise_model_import_acceptance(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
) -> EnterpriseModelImportAcceptanceEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        report = await asyncio.to_thread(
            EnterpriseModelImportAcceptanceGovernanceService(
                request.app.state.settings.enterprise_project_repo_root,
                service,
            ).snapshot
        )
        common_baselines = common_enterprise_model_import_baselines(report)
    except (
        EnterpriseModelImportAcceptanceGovernanceError,
        EnterpriseProjectAdoptionError,
    ) as exc:
        raise _unavailable() from exc
    return EnterpriseModelImportAcceptanceEnvelope(
        data=report,
        meta=EnterpriseModelImportAcceptanceMeta(
            request_id=request.state.request_id,
            components=len(report.components),
            eligible_components=sum(
                component.release_draft_eligible for component in report.components
            ),
            total_actual_samples=sum(
                component.actual_sample_count for component in report.components
            ),
            common_baseline_release_ids=common_baselines,
        ),
    )


@router.get(
    "/enterprise-assets/closure-readiness",
    response_model=EnterpriseProjectClosureEnvelope,
)
async def get_enterprise_project_closure_readiness(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
) -> EnterpriseProjectClosureEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        report = await asyncio.to_thread(
            EnterpriseProjectClosureGovernanceService(
                request.app.state.settings.enterprise_project_repo_root,
                service,
            ).snapshot
        )
    except (
        EnterpriseProjectAdoptionError,
        EnterpriseProjectClosureGovernanceError,
    ) as exc:
        raise _unavailable() from exc
    return EnterpriseProjectClosureEnvelope(
        data=report,
        meta=EnterpriseProjectClosureMeta(
            request_id=request.state.request_id,
            coverage_domains=len(report.coverage),
            source_evidence=len(report.source_evidence),
            intentionally_unverified_methods=len(report.intentionally_unverified_methods),
            production_blockers=len(report.production_blockers),
        ),
    )


@router.get(
    "/enterprise-assets/project-assurance",
    response_model=EnterpriseProjectAssuranceEnvelope,
)
async def get_enterprise_project_assurance(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
) -> EnterpriseProjectAssuranceEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        report = await asyncio.to_thread(
            EnterpriseProjectAssuranceGovernanceService(
                request.app.state.settings.enterprise_project_repo_root,
                service,
            ).snapshot
        )
    except (
        EnterpriseProjectAdoptionError,
        EnterpriseProjectAssuranceGovernanceError,
    ) as exc:
        raise _unavailable() from exc
    return EnterpriseProjectAssuranceEnvelope(
        data=report,
        meta=EnterpriseProjectAssuranceMeta(
            request_id=request.state.request_id,
            security_scenarios=report.security.scenario_count,
            recovery_components=report.recovery.evidence_count,
            healthy_slos=report.operations.healthy_slo_count,
            staging_scenarios=report.staging.scenario_count,
            signoffs=report.signoff.signoff_count,
        ),
    )


@router.get(
    "/enterprise-assets/reranker-kserve-acceptance",
    response_model=EnterpriseRerankerKServeAcceptanceEnvelope,
)
async def get_enterprise_reranker_kserve_acceptance(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> EnterpriseRerankerKServeAcceptanceEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        report = await asyncio.to_thread(
            verify_reranker_kserve_acceptance,
            request.app.state.settings.enterprise_project_repo_root,
            RERANKER_KSERVE_ACCEPTANCE_RELATIVE,
        )
    except (OSError, RerankerKServeAcceptanceError) as exc:
        raise AppError(
            status_code=503,
            code="enterprise_reranker_kserve_acceptance_unavailable",
            category="dependency",
            message="Governed Reranker KServe acceptance evidence is unavailable",
            retryable=True,
        ) from exc
    return EnterpriseRerankerKServeAcceptanceEnvelope(
        data=report,
        meta=EnterpriseRerankerKServeAcceptanceMeta(
            request_id=request.state.request_id,
            stages=len(report.stages),
            governed_requests=sum(stage.request_count for stage in report.stages),
            evidence_chain_sha256=report.evidence_chain_sha256,
        ),
    )


@router.post(
    "/enterprise-assets/release-baseline-drafts",
    response_model=EnterpriseStagingBaselineDraftEnvelope,
    status_code=201,
)
async def create_enterprise_staging_baseline_draft(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> EnterpriseStagingBaselineDraftEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        draft = await asyncio.to_thread(
            EnterpriseStagingBaselineService(database, authorizer).create_draft,
            identity,
            active_mcp_server_versions=gateway.mcp_server_versions(),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        EnterpriseStagingBaselineConflict,
        ReleaseConflict,
        ReleaseNotVisible,
    ) as exc:
        raise _translate_baseline(exc) from exc
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="invalid_enterprise_staging_baseline",
            category="validation",
            message=str(exc),
        ) from exc
    return EnterpriseStagingBaselineDraftEnvelope(
        data=draft,
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/enterprise-assets/release-batches",
    response_model=EnterpriseCandidateReleaseBatchProgressListEnvelope,
)
async def list_enterprise_candidate_release_batches(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> EnterpriseCandidateReleaseBatchProgressListEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        batches = await asyncio.to_thread(
            EnterpriseCandidateReleaseBatchService(
                database,
                authorizer,
                service,
            ).list_progress,
            identity,
            request_id=request.state.request_id,
            limit=limit,
        )
    except (
        AuthorizationDenied,
        EnterpriseCandidateReleaseBatchConflict,
        EnterpriseProjectAdoptionError,
        ReleaseConflict,
        ReleaseNotVisible,
    ) as exc:
        raise _translate_batch(exc) from exc
    return EnterpriseCandidateReleaseBatchProgressListEnvelope(
        data=list(batches),
        meta=EnterpriseCandidateReleaseBatchProgressMeta(
            request_id=request.state.request_id,
            batches=len(batches),
            awaiting_approval=sum(
                item.status == "AWAITING_INDEPENDENT_APPROVAL" for item in batches
            ),
            approved=sum(item.status == "APPROVED" for item in batches),
        ),
    )


@router.get(
    "/enterprise-assets/release-batches/review-queue",
    response_model=EnterpriseCandidateReleaseReviewQueueEnvelope,
)
async def list_enterprise_candidate_release_review_queue(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> EnterpriseCandidateReleaseReviewQueueEnvelope:
    try:
        batches = await asyncio.to_thread(
            EnterpriseCandidateReleaseBatchService(
                database,
                authorizer,
                service,
            ).list_review_queue,
            identity,
            request_id=request.state.request_id,
            limit=limit,
        )
    except AuthorizationDenied as exc:
        raise AppError(
            status_code=403,
            code="enterprise_candidate_release_review_forbidden",
            category="authorization",
            message="Enterprise candidate release review access was denied",
        ) from exc
    except (
        EnterpriseCandidateReleaseBatchConflict,
        EnterpriseProjectAdoptionError,
        ReleaseConflict,
        ReleaseNotVisible,
    ) as exc:
        raise _translate_batch(exc) from exc
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="invalid_enterprise_candidate_release_review_queue",
            category="validation",
            message=str(exc),
        ) from exc
    pending_decisions = sum(item.approval_pending_count for item in batches)
    decidable_decisions = sum(item.decidable_count for item in batches)
    return EnterpriseCandidateReleaseReviewQueueEnvelope(
        data=list(batches),
        meta=EnterpriseCandidateReleaseReviewQueueMeta(
            request_id=request.state.request_id,
            batches=len(batches),
            pending_decisions=pending_decisions,
            decidable_decisions=decidable_decisions,
            blocked_by_separation_of_duties=pending_decisions - decidable_decisions,
        ),
    )


@router.get(
    "/enterprise-assets/release-batches/deployment-queue",
    response_model=EnterpriseCandidateReleaseDeploymentQueueEnvelope,
)
async def list_enterprise_candidate_release_deployment_queue(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> EnterpriseCandidateReleaseDeploymentQueueEnvelope:
    try:
        batches = await asyncio.to_thread(
            EnterpriseCandidateReleaseBatchService(
                database,
                authorizer,
                service,
            ).list_deployment_queue,
            identity,
            request_id=request.state.request_id,
            limit=limit,
        )
    except (
        AuthorizationDenied,
        DeploymentConflict,
        DeploymentNotVisible,
        EnterpriseCandidateReleaseBatchConflict,
        EnterpriseProjectAdoptionError,
        ReleaseConflict,
        ReleaseNotVisible,
    ) as exc:
        raise _translate_batch(exc) from exc
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="invalid_enterprise_candidate_release_deployment_queue",
            category="validation",
            message=str(exc),
        ) from exc
    components = tuple(component for batch in batches for component in batch.components)
    return EnterpriseCandidateReleaseDeploymentQueueEnvelope(
        data=list(batches),
        meta=EnterpriseCandidateReleaseDeploymentQueueMeta(
            request_id=request.state.request_id,
            batches=len(batches),
            deployable_components=sum(
                "REQUEST_MODEL_DEPLOYMENT" in component.deployment_legal_actions
                for component in components
            ),
            active_deployments=sum(component.deployment_id is not None for component in components),
            promotable_components=sum(
                "PROMOTE_MODEL_RELEASE" in component.deployment_legal_actions
                for component in components
            ),
            rollback_available_components=sum(
                "ROLLBACK_MODEL_RELEASE" in component.deployment_legal_actions
                for component in components
            ),
        ),
    )


@router.post(
    "/enterprise-assets/release-batches",
    response_model=EnterpriseCandidateReleaseBatchEnvelope,
    status_code=201,
)
async def create_enterprise_candidate_release_batch(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> EnterpriseCandidateReleaseBatchEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        batch = await asyncio.to_thread(
            EnterpriseCandidateReleaseBatchService(
                database,
                authorizer,
                service,
            ).create_drafts,
            identity,
            active_mcp_server_versions=gateway.mcp_server_versions(),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        EnterpriseCandidateReleaseBatchConflict,
        EnterpriseReleaseOnboardingConflict,
        EnterpriseProjectAdoptionError,
        ReleaseConflict,
        ReleaseNotVisible,
    ) as exc:
        raise _translate_batch(exc) from exc
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="invalid_enterprise_candidate_release_batch",
            category="validation",
            message=str(exc),
        ) from exc
    return EnterpriseCandidateReleaseBatchEnvelope(
        data=batch,
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/enterprise-assets/release-batches/{batch_key_sha256}/approval-submissions",
    response_model=EnterpriseCandidateReleaseBatchProgressEnvelope,
)
async def advance_enterprise_candidate_release_batch_to_approval(
    batch_key_sha256: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")],
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
) -> EnterpriseCandidateReleaseBatchProgressEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        progress = await asyncio.to_thread(
            EnterpriseCandidateReleaseBatchService(
                database,
                authorizer,
                service,
            ).advance_to_approval,
            identity,
            batch_key_sha256,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        EnterpriseCandidateReleaseBatchConflict,
        EnterpriseProjectAdoptionError,
        ReleaseConflict,
        ReleaseNotVisible,
    ) as exc:
        raise _translate_batch(exc) from exc
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="invalid_enterprise_candidate_release_batch_advance",
            category="validation",
            message=str(exc),
        ) from exc
    return EnterpriseCandidateReleaseBatchProgressEnvelope(
        data=progress,
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/enterprise-assets/release-batches/{batch_key_sha256}/components/{component}/shadow-requests",
    response_model=EnterpriseCandidateReleaseBatchProgressEnvelope,
    status_code=202,
)
async def request_enterprise_candidate_release_component_shadow(
    batch_key_sha256: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")],
    component: EnterpriseModelComponent,
    body: DeploymentPlanBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
    requested_by_subject_id: Annotated[
        str,
        Query(min_length=1, max_length=128),
    ],
) -> EnterpriseCandidateReleaseBatchProgressEnvelope:
    try:
        progress = await asyncio.to_thread(
            EnterpriseCandidateReleaseBatchService(
                database,
                authorizer,
                service,
            ).request_component_shadow,
            identity,
            batch_key_sha256,
            component,
            DeploymentPlan(**body.model_dump()),
            requested_by_subject_id=requested_by_subject_id,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        DeploymentConflict,
        DeploymentNotVisible,
        EnterpriseCandidateReleaseBatchConflict,
        EnterpriseProjectAdoptionError,
        ReleaseConflict,
        ReleaseNotVisible,
    ) as exc:
        raise _translate_batch(exc) from exc
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="invalid_enterprise_candidate_release_shadow_request",
            category="validation",
            message=str(exc),
        ) from exc
    return EnterpriseCandidateReleaseBatchProgressEnvelope(
        data=progress,
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/enterprise-assets/runtime-bindings/{component}/imports",
    response_model=EnterpriseModelAssetImportEnvelope,
    status_code=201,
)
async def import_enterprise_model_asset(
    component: EnterpriseModelComponent,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> EnterpriseModelAssetImportEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        imported = await asyncio.to_thread(
            EnterpriseModelAssetImportService(
                database,
                authorizer,
                service,
                request.app.state.settings.enterprise_project_repo_root,
            ).import_component,
            identity,
            component,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        EnterpriseModelAssetImportConflict,
        EnterpriseProjectAdoptionError,
    ) as exc:
        raise _translate_import(exc) from exc
    except OSError as exc:
        raise _unavailable() from exc
    return EnterpriseModelAssetImportEnvelope(
        data=imported,
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/enterprise-assets/runtime-bindings",
    response_model=EnterpriseRuntimeBindingEnvelope,
)
async def list_enterprise_runtime_bindings(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
) -> EnterpriseRuntimeBindingEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        bindings = await asyncio.to_thread(
            EnterpriseRuntimeBindingService(database, service).snapshot,
            identity,
        )
    except EnterpriseProjectAdoptionError as exc:
        raise _unavailable() from exc
    return EnterpriseRuntimeBindingEnvelope(
        data=list(bindings),
        meta=EnterpriseRuntimeBindingMeta(
            request_id=request.state.request_id,
            components=len(bindings),
            registered_components=sum(item.registry_state == "REGISTERED" for item in bindings),
            deployed_components=sum(
                item.deployment_state == "DEPLOYMENT_RECORDED" for item in bindings
            ),
            active_alias_components=sum(
                item.alias_state == "ACTIVE_ALIAS_BOUND" for item in bindings
            ),
        ),
    )


@router.get(
    "/enterprise-assets/runtime-bindings/{component}/release-draft-preview",
    response_model=EnterpriseReleaseDraftPreviewEnvelope,
)
async def preview_enterprise_release_draft(
    component: EnterpriseModelComponent,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
) -> EnterpriseReleaseDraftPreviewEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    try:
        preview = await asyncio.to_thread(
            EnterpriseReleaseOnboardingService(
                database,
                authorizer,
                service,
            ).preview,
            identity,
            component,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        EnterpriseReleaseOnboardingConflict,
        EnterpriseProjectAdoptionError,
    ) as exc:
        raise _translate_onboarding(exc) from exc
    return EnterpriseReleaseDraftPreviewEnvelope(
        data=preview,
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/enterprise-assets/runtime-bindings/{component}/release-drafts",
    response_model=EnterpriseReleaseOnboardingEnvelope,
    status_code=201,
)
async def create_enterprise_release_draft(
    component: EnterpriseModelComponent,
    body: EnterpriseReleaseDraftBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    service: Annotated[
        EnterpriseProjectAdoptionReader,
        Depends(get_enterprise_project_adoption_service),
    ],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> EnterpriseReleaseOnboardingEnvelope:
    _require_access(identity, authorizer, request.state.request_id)
    deployment_plan = (
        DeploymentPlan(**body.deployment_plan.model_dump())
        if body.deployment_plan is not None
        else None
    )
    try:
        onboarding = await asyncio.to_thread(
            EnterpriseReleaseOnboardingService(
                database,
                authorizer,
                service,
            ).create_draft,
            identity,
            component,
            baseline_release_id=body.baseline_release_id,
            auto_shadow_enabled=body.auto_shadow_enabled,
            deployment_plan=deployment_plan,
            active_mcp_server_versions=gateway.mcp_server_versions(),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        EnterpriseReleaseOnboardingConflict,
        EnterpriseProjectAdoptionError,
        ReleaseConflict,
        ReleaseNotVisible,
    ) as exc:
        raise _translate_onboarding(exc) from exc
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="invalid_enterprise_release_draft",
            category="validation",
            message=str(exc),
        ) from exc
    return EnterpriseReleaseOnboardingEnvelope(
        data=onboarding,
        meta={"request_id": request.state.request_id},
    )


def _require_access(
    identity: IdentityContext,
    authorizer: Authorizer,
    request_id: str,
) -> None:
    try:
        authorizer.require(
            identity,
            Action.READ_ENTERPRISE_PROJECT_ADOPTION,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="enterprise-project-adoption",
            ),
            request_id=request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(
            status_code=403,
            code="enterprise_project_adoption_forbidden",
            category="authorization",
            message="Enterprise project adoption access was denied",
        ) from exc


def _unavailable() -> AppError:
    return AppError(
        status_code=503,
        code="enterprise_project_adoption_unavailable",
        category="dependency",
        message="Enterprise project adoption evidence is unavailable",
        retryable=True,
    )


def _translate_onboarding(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(
            status_code=403,
            code="enterprise_release_onboarding_forbidden",
            category="authorization",
            message="Enterprise release onboarding access was denied",
        )
    if isinstance(exc, EnterpriseProjectAdoptionError):
        return _unavailable()
    if isinstance(exc, ReleaseNotVisible):
        return AppError(
            status_code=404,
            code="resource_not_found_or_not_visible",
            category="not_found",
            message="Resource not found or not visible",
        )
    if isinstance(exc, EnterpriseReleaseOnboardingConflict):
        return AppError(
            status_code=409,
            code=exc.reason.lower(),
            category="release_governance",
            message="Enterprise release onboarding is blocked by governance",
            details={"current_version": exc.current_version},
        )
    if isinstance(exc, ReleaseConflict):
        return AppError(
            status_code=409,
            code=exc.reason,
            category="release_governance",
            message="Model release governance rejected the draft",
            details={"current_version": exc.current_version},
        )
    raise TypeError("unsupported enterprise release onboarding error")


def _translate_import(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(
            status_code=403,
            code="enterprise_model_asset_import_forbidden",
            category="authorization",
            message="Enterprise model asset import access was denied",
        )
    if isinstance(exc, EnterpriseProjectAdoptionError):
        return _unavailable()
    if isinstance(exc, EnterpriseModelAssetImportConflict):
        return AppError(
            status_code=409,
            code=exc.reason.lower().replace(":", "_"),
            category="model_governance",
            message="Enterprise model asset import was rejected by governance",
            details={"current_version": exc.current_version},
        )
    raise TypeError("unsupported enterprise model asset import error")


def _translate_baseline(exc: Exception) -> AppError:
    if isinstance(exc, EnterpriseStagingBaselineConflict):
        return AppError(
            status_code=409,
            code=exc.reason.lower(),
            category="release_governance",
            message="Enterprise Staging baseline bootstrap was rejected by governance",
            details={"current_version": exc.current_version},
        )
    return _translate_onboarding(exc)


def _translate_batch(exc: Exception) -> AppError:
    if isinstance(exc, DeploymentNotVisible):
        return AppError(
            status_code=404,
            code="resource_not_found_or_not_visible",
            category="not_found",
            message="Resource not found or not visible",
        )
    if isinstance(exc, DeploymentConflict):
        return AppError(
            status_code=409,
            code=exc.reason,
            category="model_deployment_governance",
            message="Enterprise candidate deployment was rejected by governance",
            details={"current_version": exc.current_version},
        )
    if isinstance(exc, EnterpriseCandidateReleaseBatchConflict):
        return AppError(
            status_code=409,
            code=exc.reason.lower().replace(":", "_"),
            category="release_governance",
            message="Enterprise candidate release batch was rejected by governance",
            details={"current_version": exc.current_version},
        )
    return _translate_onboarding(exc)
