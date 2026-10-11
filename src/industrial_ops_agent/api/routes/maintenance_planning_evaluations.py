"""Gold-suite and blinded A/B evaluation API for multi-agent maintenance planning."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.api.versions import numeric_precondition
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.maintenance_planning import (
    MaintenancePlanningEvaluationConflict,
    MaintenancePlanningEvaluationNotVisible,
    MaintenancePlanningEvaluationRunView,
    MaintenancePlanningEvaluationService,
    MaintenancePlanningEvaluationSuiteView,
    MaintenancePlanningGoldCaseInput,
    MaintenancePlanningPlanScore,
)
from industrial_ops_agent.maintenance_planning.rollout_evidence import ReviewPolicyService
from industrial_ops_agent.persistence.database import Database

suite_router = APIRouter(
    prefix="/maintenance-planning-evaluation-suites",
    tags=["maintenance-planning-evaluation"],
)
evaluation_router = APIRouter(
    prefix="/maintenance-planning-evaluations",
    tags=["maintenance-planning-evaluation"],
)


class MaintenancePlanningGoldCaseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis_run_id: str = Field(min_length=3, max_length=128)
    expected_findings: list[str] = Field(min_length=1, max_length=20)
    required_safety_hold_points: list[str] = Field(default_factory=list, max_length=20)
    allowed_parts: list[str] = Field(default_factory=list, max_length=20)
    required_dispatch_constraints: list[str] = Field(min_length=1, max_length=20)
    high_risk: bool = True


class CreateMaintenancePlanningEvaluationSuiteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=255)
    suite_version: str = Field(min_length=1, max_length=64)
    evidence_tier: Literal["PROJECT_STAGING_GOLD", "ENTERPRISE_GOLD"]
    cases: list[MaintenancePlanningGoldCaseBody] = Field(min_length=1, max_length=500)


class MaintenancePlanningEvaluationSuiteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_id: str
    name: str
    suite_version: str
    evidence_tier: Literal["PROJECT_STAGING_GOLD", "ENTERPRISE_GOLD"]
    status: Literal["FROZEN", "RETIRED"]
    cases: list[dict[str, Any]]
    sample_count: int
    manifest_hash: str
    created_by_subject_id: str
    version: int
    created_at: datetime
    updated_at: datetime


class MaintenancePlanningEvaluationSuiteEnvelope(BaseModel):
    data: MaintenancePlanningEvaluationSuiteResponse
    meta: dict[str, str]


class MaintenancePlanningEvaluationSuiteListEnvelope(BaseModel):
    data: list[MaintenancePlanningEvaluationSuiteResponse]
    meta: dict[str, str | int]


class CreateMaintenancePlanningEvaluationRunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_id: str = Field(min_length=3, max_length=128)
    council_ids: list[str] = Field(min_length=1, max_length=500)


class MaintenancePlanningPlanScoreBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_quality: float = Field(ge=1, le=5)
    factual_error_count: int = Field(ge=0, le=100)
    safety_omission_count: int = Field(ge=0, le=100)
    parts_false_positive_count: int = Field(ge=0, le=100)
    dispatch_executability: float = Field(ge=1, le=5)
    expert_review_seconds: float = Field(ge=1, le=3_600)


class SubmitMaintenancePlanningBlindJudgmentBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=3, max_length=128)
    claim_version: int = Field(ge=1)
    variant_a: MaintenancePlanningPlanScoreBody
    variant_b: MaintenancePlanningPlanScoreBody
    preferred_variant: Literal["A", "B", "TIE"]
    rationale: str = Field(min_length=8, max_length=1_000)


class MaintenancePlanningBlindPairResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    diagnosis_run_id: str
    incident_id: str
    asset_id: str
    high_risk: bool
    rubric: dict[str, Any]
    variant_a: dict[str, Any]
    variant_b: dict[str, Any]
    judgment_count: int
    required_judgment_count: int
    needs_adjudication: bool
    judged_by_current_subject: bool
    revealed_assignment: dict[str, str] | None


class MaintenancePlanningEvaluationRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    suite_id: str
    suite_name: str
    suite_version: str
    evidence_tier: Literal["PROJECT_STAGING_GOLD", "ENTERPRISE_GOLD"]
    status: Literal["JUDGING", "COMPLETED", "FAILED"]
    policy_version: str
    policy_hash: str
    policy: dict[str, Any]
    pairs: list[MaintenancePlanningBlindPairResponse]
    required_judgments_per_case: int
    aggregate_metrics: dict[str, Any]
    gate_results: dict[str, bool]
    decision: Literal[
        "PENDING",
        "KEEP_SINGLE_AGENT",
        "PROJECT_CANDIDATE_ELIGIBLE",
        "PRODUCTION_CANDIDATE_ELIGIBLE",
    ]
    report_hash: str | None
    failure_reason: str | None
    requested_by_subject_id: str
    completed_at: datetime | None
    version: int
    legal_actions: list[Literal["SUBMIT_BLIND_JUDGMENT"]]
    created_at: datetime
    updated_at: datetime
    historical_exposure_status: Literal["LEGACY_UNVERIFIED", "TRACKED", "VERIFIED"]
    comparison_blinded: bool
    activation_changed: Literal[False] = False


class MaintenancePlanningEvaluationRunEnvelope(BaseModel):
    data: MaintenancePlanningEvaluationRunResponse
    meta: dict[str, str]


class MaintenancePlanningEvaluationRunListEnvelope(BaseModel):
    data: list[MaintenancePlanningEvaluationRunResponse]
    meta: dict[str, str | int]


class ReviewClaimVersionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(min_length=3, max_length=128)
    claim_version: int = Field(ge=1)


class AnonymousReviewClaimResponse(ReviewClaimVersionBody):
    evaluation_id: str
    case_id: str
    state: Literal["ACTIVE", "SUBMITTED", "WITHDRAWN"]
    variant_a: dict[str, Any] | None
    variant_b: dict[str, Any] | None


class AnonymousReviewClaimEnvelope(BaseModel):
    data: AnonymousReviewClaimResponse
    meta: dict[str, str]


class AnonymousReviewQueueItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluation_id: str
    case_id: str
    eligibility: str
    claim_id: str | None
    claim_version: int | None


class AnonymousReviewQueueEnvelope(BaseModel):
    data: list[AnonymousReviewQueueItem]
    meta: dict[str, str]


class ReviewRolloutRegistrationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    receipt_text: str = Field(min_length=1, max_length=262144)
    deployment_confirmed: Literal[True]
    reason: str = Field(min_length=8, max_length=1000)


class ReviewRolloutRegistrationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    receipt_id: str
    receipt_digest: str


class ReviewRolloutRegistrationEnvelope(BaseModel):
    data: ReviewRolloutRegistrationResponse
    meta: dict[str, str]


class ReviewPolicyCommandBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_version: Literal["maintenance-planning-blind-ab-v2"]
    reason: str = Field(min_length=8, max_length=1000)


class ReviewPolicyActivationBody(ReviewPolicyCommandBody):
    receipt_id: str = Field(min_length=3, max_length=128)


class ReviewPolicyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    version: int
    policy_version: str
    reader_coverage_digest: str
    cutover_at: datetime | None
    receipt_id: str | None
    receipt_digest: str | None
    reason: str | None = None


class ReviewRolloutStatus(ReviewRolloutRegistrationResponse):
    registered_at: datetime
    blockers: list[str]


class ReviewPolicyStatus(ReviewPolicyResponse):
    assurance: Literal["PROJECT_ADMIN_CONFIRMED"]
    environment_id: str | None
    blockers: list[str]
    receipts: list[ReviewRolloutStatus]


class ReviewPolicyStatusEnvelope(BaseModel):
    data: ReviewPolicyStatus
    meta: dict[str, str]


class ReviewPolicyEnvelope(BaseModel):
    data: ReviewPolicyResponse
    meta: dict[str, str]


def _review_policy_service(
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ReviewPolicyService:
    return ReviewPolicyService(database, authorizer)


ReviewPolicyDependency = Annotated[ReviewPolicyService, Depends(_review_policy_service)]
ReviewVersionHeader = Annotated[str, Header(alias="If-Match", min_length=1, max_length=32)]
ReviewKeyHeader = Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=255)]


@evaluation_router.get("/review-policy", response_model=ReviewPolicyStatusEnvelope)
def review_policy_status(
    request: Request,
    service: ReviewPolicyDependency,
    identity: Annotated[IdentityContext, Depends(get_identity)],
) -> ReviewPolicyStatusEnvelope:
    try:
        result = service.status(identity, request_id=request.state.request_id)
    except AuthorizationDenied as exc:
        raise _error(exc) from exc
    return ReviewPolicyStatusEnvelope(
        data=ReviewPolicyStatus(**result), meta={"request_id": request.state.request_id}
    )


@evaluation_router.post("/review-policy/receipts", response_model=ReviewRolloutRegistrationEnvelope)
def register_review_rollout(
    body: ReviewRolloutRegistrationBody,
    request: Request,
    service: ReviewPolicyDependency,
    identity: Annotated[IdentityContext, Depends(get_identity)],
) -> ReviewRolloutRegistrationEnvelope:
    try:
        result = service.register(
            identity,
            body.receipt_text,
            deployment_confirmed=body.deployment_confirmed,
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(exc) from exc
    return ReviewRolloutRegistrationEnvelope(
        data=ReviewRolloutRegistrationResponse(**result),
        meta={"request_id": request.state.request_id},
    )


@evaluation_router.post("/review-policy/activate", response_model=ReviewPolicyEnvelope)
def activate_review_policy(
    body: ReviewPolicyActivationBody,
    request: Request,
    service: ReviewPolicyDependency,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    if_match: ReviewVersionHeader,
    idempotency_key: ReviewKeyHeader,
) -> ReviewPolicyEnvelope:
    try:
        result = service.change(
            identity,
            enabled=True,
            receipt_id=body.receipt_id,
            expected_version=numeric_precondition(if_match),
            idempotency_key=idempotency_key,
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(exc) from exc
    return ReviewPolicyEnvelope(
        data=ReviewPolicyResponse(**result), meta={"request_id": request.state.request_id}
    )


@evaluation_router.post("/review-policy/deactivate", response_model=ReviewPolicyEnvelope)
def deactivate_review_policy(
    body: ReviewPolicyCommandBody,
    request: Request,
    service: ReviewPolicyDependency,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    if_match: ReviewVersionHeader,
    idempotency_key: ReviewKeyHeader,
) -> ReviewPolicyEnvelope:
    try:
        result = service.change(
            identity,
            enabled=False,
            receipt_id=None,
            expected_version=numeric_precondition(if_match),
            idempotency_key=idempotency_key,
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(exc) from exc
    return ReviewPolicyEnvelope(
        data=ReviewPolicyResponse(**result), meta={"request_id": request.state.request_id}
    )


@evaluation_router.get("/review-queue", response_model=AnonymousReviewQueueEnvelope)
def list_anonymous_reviews(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> AnonymousReviewQueueEnvelope:
    try:
        items = MaintenancePlanningEvaluationService(database, authorizer).review_queue(
            identity, request_id=request.state.request_id
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningEvaluationNotVisible,
        MaintenancePlanningEvaluationConflict,
    ) as exc:
        raise _error(exc) from exc
    return AnonymousReviewQueueEnvelope(
        data=[AnonymousReviewQueueItem(**item) for item in items],
        meta={"request_id": request.state.request_id},
    )


@evaluation_router.post(
    "/{evaluation_id}/cases/{case_id}/claim", response_model=AnonymousReviewClaimEnvelope
)
def claim_anonymous_review(
    evaluation_id: str,
    case_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=255)],
) -> AnonymousReviewClaimEnvelope:
    try:
        item = MaintenancePlanningEvaluationService(database, authorizer).claim_review(
            identity,
            evaluation_id,
            case_id,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningEvaluationNotVisible,
        MaintenancePlanningEvaluationConflict,
    ) as exc:
        raise _error(exc) from exc
    return AnonymousReviewClaimEnvelope(
        data=AnonymousReviewClaimResponse(**item), meta={"request_id": request.state.request_id}
    )


@evaluation_router.post(
    "/{evaluation_id}/cases/{case_id}/withdraw", response_model=AnonymousReviewClaimEnvelope
)
def withdraw_anonymous_review(
    evaluation_id: str,
    case_id: str,
    body: ReviewClaimVersionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> AnonymousReviewClaimEnvelope:
    try:
        item = MaintenancePlanningEvaluationService(database, authorizer).withdraw_review(
            identity,
            evaluation_id,
            case_id,
            claim_id=body.claim_id,
            claim_version=body.claim_version,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningEvaluationNotVisible,
        MaintenancePlanningEvaluationConflict,
    ) as exc:
        raise _error(exc) from exc
    return AnonymousReviewClaimEnvelope(
        data=AnonymousReviewClaimResponse(**item), meta={"request_id": request.state.request_id}
    )


@suite_router.get("", response_model=MaintenancePlanningEvaluationSuiteListEnvelope)
async def list_maintenance_planning_evaluation_suites(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> MaintenancePlanningEvaluationSuiteListEnvelope:
    try:
        items = await asyncio.to_thread(
            MaintenancePlanningEvaluationService(database, authorizer).list_suites,
            identity,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, MaintenancePlanningEvaluationNotVisible) as exc:
        raise _error(exc) from exc
    return MaintenancePlanningEvaluationSuiteListEnvelope(
        data=[_suite_response(item) for item in items],
        meta={"request_id": request.state.request_id, "total": len(items)},
    )


@suite_router.post(
    "",
    response_model=MaintenancePlanningEvaluationSuiteEnvelope,
    status_code=201,
)
async def create_maintenance_planning_evaluation_suite(
    body: CreateMaintenancePlanningEvaluationSuiteBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> MaintenancePlanningEvaluationSuiteEnvelope:
    try:
        item = await asyncio.to_thread(
            MaintenancePlanningEvaluationService(database, authorizer).create_suite,
            identity,
            name=body.name,
            suite_version=body.suite_version,
            evidence_tier=body.evidence_tier,
            cases=tuple(
                MaintenancePlanningGoldCaseInput(
                    diagnosis_run_id=case.diagnosis_run_id,
                    expected_findings=tuple(case.expected_findings),
                    required_safety_hold_points=tuple(case.required_safety_hold_points),
                    allowed_parts=tuple(case.allowed_parts),
                    required_dispatch_constraints=tuple(case.required_dispatch_constraints),
                    high_risk=case.high_risk,
                )
                for case in body.cases
            ),
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningEvaluationNotVisible,
        MaintenancePlanningEvaluationConflict,
    ) as exc:
        raise _error(exc) from exc
    return MaintenancePlanningEvaluationSuiteEnvelope(
        data=_suite_response(item),
        meta={"request_id": request.state.request_id},
    )


@evaluation_router.get("", response_model=MaintenancePlanningEvaluationRunListEnvelope)
async def list_maintenance_planning_evaluations(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> MaintenancePlanningEvaluationRunListEnvelope:
    try:
        items = await asyncio.to_thread(
            MaintenancePlanningEvaluationService(database, authorizer).list_runs,
            identity,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, MaintenancePlanningEvaluationNotVisible) as exc:
        raise _error(exc) from exc
    return MaintenancePlanningEvaluationRunListEnvelope(
        data=[_run_response(item, identity, authorizer) for item in items],
        meta={"request_id": request.state.request_id, "total": len(items)},
    )


@evaluation_router.post(
    "",
    response_model=MaintenancePlanningEvaluationRunEnvelope,
    status_code=201,
)
async def create_maintenance_planning_evaluation(
    body: CreateMaintenancePlanningEvaluationRunBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=255),
    ],
) -> MaintenancePlanningEvaluationRunEnvelope:
    try:
        item = await asyncio.to_thread(
            MaintenancePlanningEvaluationService(database, authorizer).create_run,
            identity,
            suite_id=body.suite_id,
            council_ids=tuple(body.council_ids),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningEvaluationNotVisible,
        MaintenancePlanningEvaluationConflict,
    ) as exc:
        raise _error(exc) from exc
    return MaintenancePlanningEvaluationRunEnvelope(
        data=_run_response(item, identity, authorizer),
        meta={"request_id": request.state.request_id},
    )


@evaluation_router.get(
    "/{evaluation_id}",
    response_model=MaintenancePlanningEvaluationRunEnvelope,
)
async def get_maintenance_planning_evaluation(
    evaluation_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> MaintenancePlanningEvaluationRunEnvelope:
    try:
        item = await asyncio.to_thread(
            MaintenancePlanningEvaluationService(database, authorizer).get_run,
            identity,
            evaluation_id,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningEvaluationNotVisible,
        MaintenancePlanningEvaluationConflict,
    ) as exc:
        raise _error(exc) from exc
    return MaintenancePlanningEvaluationRunEnvelope(
        data=_run_response(item, identity, authorizer),
        meta={"request_id": request.state.request_id},
    )


@evaluation_router.post(
    "/{evaluation_id}/cases/{case_id}/judgments",
    response_model=AnonymousReviewClaimEnvelope,
)
async def submit_maintenance_planning_blind_judgment(
    evaluation_id: str,
    case_id: str,
    body: SubmitMaintenancePlanningBlindJudgmentBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> AnonymousReviewClaimEnvelope:
    try:
        item = await asyncio.to_thread(
            MaintenancePlanningEvaluationService(database, authorizer).submit_judgment,
            identity,
            evaluation_id,
            case_id,
            claim_id=body.claim_id,
            claim_version=body.claim_version,
            variant_a=_score(body.variant_a),
            variant_b=_score(body.variant_b),
            preferred_variant=body.preferred_variant,
            rationale=body.rationale,
            request_id=request.state.request_id,
        )
    except (
        AuthorizationDenied,
        MaintenancePlanningEvaluationNotVisible,
        MaintenancePlanningEvaluationConflict,
    ) as exc:
        raise _error(exc) from exc
    return AnonymousReviewClaimEnvelope(
        data=AnonymousReviewClaimResponse(**item),
        meta={"request_id": request.state.request_id},
    )


def _score(body: MaintenancePlanningPlanScoreBody) -> MaintenancePlanningPlanScore:
    return MaintenancePlanningPlanScore(
        plan_quality=body.plan_quality,
        factual_error_count=body.factual_error_count,
        safety_omission_count=body.safety_omission_count,
        parts_false_positive_count=body.parts_false_positive_count,
        dispatch_executability=body.dispatch_executability,
        expert_review_seconds=body.expert_review_seconds,
    )


def _suite_response(
    item: MaintenancePlanningEvaluationSuiteView,
) -> MaintenancePlanningEvaluationSuiteResponse:
    return MaintenancePlanningEvaluationSuiteResponse(
        suite_id=item.suite_id,
        name=item.name,
        suite_version=item.suite_version,
        evidence_tier=item.evidence_tier,
        status=item.status,
        cases=[dict(case) for case in item.cases],
        sample_count=item.sample_count,
        manifest_hash=item.manifest_hash,
        created_by_subject_id=item.created_by_subject_id,
        version=item.version,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def _run_response(
    item: MaintenancePlanningEvaluationRunView,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> MaintenancePlanningEvaluationRunResponse:
    actions: list[str] = []
    if "SUBMIT_BLIND_JUDGMENT" in item.legal_actions and item.pairs:
        resource = ResourceContext(
            tenant_id=identity.tenant_id,
            resource_id=item.evaluation_id,
            asset_id=item.pairs[0].asset_id,
        )
        if authorizer.decide(identity, Action.REVIEW_MAINTENANCE_COUNCIL, resource).allowed:
            actions.append("SUBMIT_BLIND_JUDGMENT")
    return MaintenancePlanningEvaluationRunResponse(
        evaluation_id=item.evaluation_id,
        suite_id=item.suite_id,
        suite_name=item.suite_name,
        suite_version=item.suite_version,
        evidence_tier=item.evidence_tier,
        status=item.status,
        historical_exposure_status=(
            "LEGACY_UNVERIFIED"
            if item.policy_version != "maintenance-planning-blind-ab-v2"
            else "VERIFIED"
            if item.gate_results.get("historical_exposure_isolated") is True
            else "TRACKED"
        ),
        policy_version=item.policy_version,
        policy_hash=item.policy_hash,
        policy=item.policy,
        pairs=[
            MaintenancePlanningBlindPairResponse(
                case_id=pair.case_id,
                diagnosis_run_id=pair.diagnosis_run_id,
                incident_id=pair.incident_id,
                asset_id=pair.asset_id,
                high_risk=pair.high_risk,
                rubric=pair.rubric,
                variant_a=pair.variant_a,
                variant_b=pair.variant_b,
                judgment_count=pair.judgment_count,
                required_judgment_count=pair.required_judgment_count,
                needs_adjudication=pair.needs_adjudication,
                judged_by_current_subject=pair.judged_by_current_subject,
                revealed_assignment=pair.revealed_assignment,
            )
            for pair in item.pairs
        ],
        required_judgments_per_case=item.required_judgments_per_case,
        aggregate_metrics=item.aggregate_metrics,
        gate_results=item.gate_results,
        decision=item.decision,
        report_hash=item.report_hash,
        failure_reason=item.failure_reason,
        requested_by_subject_id=item.requested_by_subject_id,
        completed_at=item.completed_at,
        version=item.version,
        legal_actions=actions,
        created_at=item.created_at,
        updated_at=item.updated_at,
        comparison_blinded=item.status != "COMPLETED",
    )


def _error(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(
            403,
            "maintenance_planning_evaluation_forbidden",
            "authorization",
            "Forbidden",
        )
    if isinstance(exc, MaintenancePlanningEvaluationNotVisible):
        return AppError(
            404,
            "maintenance_planning_evaluation_not_found",
            "visibility",
            "Not found",
        )
    if isinstance(exc, MaintenancePlanningEvaluationConflict):
        details = (
            {"current_version": exc.current_version} if exc.current_version is not None else None
        )
        return AppError(
            409,
            exc.reason,
            "conflict",
            "Maintenance planning evaluation conflicts with current evidence",
            details=details,
        )
    return AppError(
        500,
        "maintenance_planning_evaluation_failed",
        "internal",
        "Request failed",
    )
