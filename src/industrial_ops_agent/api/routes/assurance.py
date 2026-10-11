"""Governed security exercise and production acceptance API."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field, StrictInt

from industrial_ops_agent.api.dependencies import get_assurance_service, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.assurance.service import (
    AcceptancePlan,
    AssuranceConflict,
    AssuranceNotFound,
    AssuranceOverview,
    AssuranceService,
    AttackScenario,
    ExerciseResultInput,
    ObservedOutcome,
    ProductionAcceptanceView,
    ProductionDefectView,
    SecurityExerciseView,
    SignoffRole,
)
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.staging_acceptance.dynamic_contracts import (
    StagingDynamicAcceptanceManifest,
)

router = APIRouter(tags=["production-assurance"])


class StartExerciseBody(BaseModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
    target_environment: str = Field(pattern=r"^(STAGING|PRODUCTION)$")
    scenario_ids: list[AttackScenario] = Field(min_length=1, max_length=12)


class ExerciseResultBody(BaseModel):
    scenario_id: AttackScenario
    observed_outcome: ObservedOutcome
    attempt_count: StrictInt = Field(ge=0, le=1_000_000)
    unauthorized_read_count: StrictInt = Field(ge=0, le=1_000_000)
    unauthorized_side_effect_count: StrictInt = Field(ge=0, le=1_000_000)
    sensitive_output_count: StrictInt = Field(ge=0, le=1_000_000)
    evidence_ref: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    artifact_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class CompleteExerciseBody(BaseModel):
    verifier_version: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    report_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    results: list[ExerciseResultBody] = Field(min_length=1, max_length=12)


class CloseDefectBody(BaseModel):
    evidence_ref: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    evidence_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class CreateAcceptanceBody(BaseModel):
    release_scope: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    target_environment: str = Field(pattern=r"^PRODUCTION$")
    security_exercise_id: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    model_release_id: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    business_evidence_ref: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    business_evidence_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    rollback_evidence_ref: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    rollback_evidence_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    staging_dynamic_manifest: StagingDynamicAcceptanceManifest | None = None
    staging_acceptance_attestation_evidence_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )


class SignAcceptanceBody(BaseModel):
    signoff_role: SignoffRole
    evidence_ref: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    evidence_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ScenarioResponse(BaseModel):
    scenario_id: str
    category: str
    title: str
    expected_outcome: str


class SecurityExerciseResponse(BaseModel):
    exercise_id: str
    name: str
    target_environment: str
    status: str
    scenario_ids: list[str]
    results: list[dict[str, Any]]
    blocker_codes: list[str]
    report_digest: str | None
    verifier_version: str | None
    requested_by_subject_id: str
    completed_by_subject_id: str | None
    started_at: datetime
    completed_at: datetime | None
    version: int


class ProductionDefectResponse(BaseModel):
    defect_id: str
    scenario_id: str
    severity: str
    release_blocking: bool
    status: str
    description_code: str
    exercise_id: str
    opened_at: datetime
    closed_at: datetime | None
    version: int


class AcceptanceSignoffResponse(BaseModel):
    signoff_id: str
    signoff_role: str
    readiness_digest: str
    evidence_ref: str
    evidence_digest: str
    signed_by_subject_id: str
    signed_at: datetime


class ProductionAcceptanceResponse(BaseModel):
    acceptance_id: str
    release_scope: str
    target_environment: str
    security_exercise_id: str
    model_release_id: str
    staging_dynamic_manifest_id: str | None
    staging_dynamic_manifest_status: str | None
    staging_dynamic_manifest_generated_at: datetime | None
    staging_acceptance_attestation_evidence_id: str | None
    staging_acceptance_attestation_status: str | None
    staging_acceptance_attestation_certificate_identity: str | None
    staging_acceptance_attestation_certificate_oidc_issuer: str | None
    status: str
    blocker_codes: list[str]
    readiness_snapshot: dict[str, Any]
    readiness_digest: str
    created_by_subject_id: str
    evaluated_at: datetime
    signed_off_at: datetime | None
    signoffs: list[AcceptanceSignoffResponse]
    version: int


class AssuranceOverviewResponse(BaseModel):
    policy_version: str
    required_scenarios: list[ScenarioResponse]
    exercises: list[SecurityExerciseResponse]
    open_defects: list[ProductionDefectResponse]
    acceptances: list[ProductionAcceptanceResponse]


class AssuranceOverviewEnvelope(BaseModel):
    data: AssuranceOverviewResponse
    legal_actions: list[str]
    meta: dict[str, str]


class ExerciseEnvelope(BaseModel):
    data: SecurityExerciseResponse
    meta: dict[str, str]


class DefectEnvelope(BaseModel):
    data: ProductionDefectResponse
    meta: dict[str, str]


class AcceptanceEnvelope(BaseModel):
    data: ProductionAcceptanceResponse
    meta: dict[str, str]


@router.get("/assurance/overview", response_model=AssuranceOverviewEnvelope)
async def get_assurance_overview(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[AssuranceService, Depends(get_assurance_service)],
) -> AssuranceOverviewEnvelope:
    try:
        overview = await asyncio.to_thread(
            service.overview,
            identity,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _translate(exc) from exc
    return AssuranceOverviewEnvelope(
        data=_overview_response(overview),
        legal_actions=_legal_actions(identity),
        meta={"request_id": request.state.request_id},
    )


@router.post("/security-exercises", response_model=ExerciseEnvelope, status_code=201)
async def start_security_exercise(
    body: StartExerciseBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[AssuranceService, Depends(get_assurance_service)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> ExerciseEnvelope:
    try:
        exercise = await asyncio.to_thread(
            service.start_exercise,
            identity,
            idempotency_key=idempotency_key,
            name=body.name,
            target_environment=body.target_environment,
            scenario_ids=tuple(body.scenario_ids),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, AssuranceConflict, ValueError) as exc:
        raise _translate(exc) from exc
    return ExerciseEnvelope(
        data=_exercise_response(exercise),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/security-exercises/{exercise_id}/complete",
    response_model=ExerciseEnvelope,
)
async def complete_security_exercise(
    exercise_id: str,
    body: CompleteExerciseBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[AssuranceService, Depends(get_assurance_service)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ExerciseEnvelope:
    try:
        exercise = await asyncio.to_thread(
            service.complete_exercise,
            identity,
            exercise_id,
            expected_version=_version(if_match),
            verifier_version=body.verifier_version,
            report_digest=body.report_digest,
            results=tuple(
                ExerciseResultInput(
                    scenario_id=item.scenario_id,
                    observed_outcome=item.observed_outcome,
                    attempt_count=item.attempt_count,
                    unauthorized_read_count=item.unauthorized_read_count,
                    unauthorized_side_effect_count=item.unauthorized_side_effect_count,
                    sensitive_output_count=item.sensitive_output_count,
                    evidence_ref=item.evidence_ref,
                    artifact_digest=item.artifact_digest,
                )
                for item in body.results
            ),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, AssuranceConflict, AssuranceNotFound, ValueError) as exc:
        raise _translate(exc) from exc
    return ExerciseEnvelope(
        data=_exercise_response(exercise),
        meta={"request_id": request.state.request_id},
    )


@router.post("/production-defects/{defect_id}/close", response_model=DefectEnvelope)
async def close_production_defect(
    defect_id: str,
    body: CloseDefectBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[AssuranceService, Depends(get_assurance_service)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> DefectEnvelope:
    try:
        defect = await asyncio.to_thread(
            service.close_defect,
            identity,
            defect_id,
            expected_version=_version(if_match),
            evidence_ref=body.evidence_ref,
            evidence_digest=body.evidence_digest,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, AssuranceConflict, AssuranceNotFound, ValueError) as exc:
        raise _translate(exc) from exc
    return DefectEnvelope(
        data=_defect_response(defect),
        meta={"request_id": request.state.request_id},
    )


@router.post("/production-acceptances", response_model=AcceptanceEnvelope, status_code=201)
async def create_production_acceptance(
    body: CreateAcceptanceBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[AssuranceService, Depends(get_assurance_service)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> AcceptanceEnvelope:
    try:
        acceptance = await asyncio.to_thread(
            service.create_acceptance,
            identity,
            idempotency_key=idempotency_key,
            plan=AcceptancePlan(**body.model_dump(mode="json")),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, AssuranceConflict, AssuranceNotFound, ValueError) as exc:
        raise _translate(exc) from exc
    return AcceptanceEnvelope(
        data=_acceptance_response(acceptance),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/production-acceptances/{acceptance_id}/signoffs",
    response_model=AcceptanceEnvelope,
)
async def sign_production_acceptance(
    acceptance_id: str,
    body: SignAcceptanceBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[AssuranceService, Depends(get_assurance_service)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> AcceptanceEnvelope:
    try:
        acceptance = await asyncio.to_thread(
            service.sign_acceptance,
            identity,
            acceptance_id,
            signoff_role=body.signoff_role,
            expected_version=_version(if_match),
            evidence_ref=body.evidence_ref,
            evidence_digest=body.evidence_digest,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, AssuranceConflict, AssuranceNotFound, ValueError) as exc:
        raise _translate(exc) from exc
    return AcceptanceEnvelope(
        data=_acceptance_response(acceptance),
        meta={"request_id": request.state.request_id},
    )


def _overview_response(value: AssuranceOverview) -> AssuranceOverviewResponse:
    return AssuranceOverviewResponse(
        policy_version=value.policy_version,
        required_scenarios=[
            ScenarioResponse(
                scenario_id=item.scenario_id.value,
                category=item.category,
                title=item.title,
                expected_outcome=item.expected_outcome.value,
            )
            for item in value.required_scenarios
        ],
        exercises=[_exercise_response(item) for item in value.exercises],
        open_defects=[_defect_response(item) for item in value.open_defects],
        acceptances=[_acceptance_response(item) for item in value.acceptances],
    )


def _exercise_response(value: SecurityExerciseView) -> SecurityExerciseResponse:
    return SecurityExerciseResponse(
        exercise_id=value.exercise_id,
        name=value.name,
        target_environment=value.target_environment,
        status=value.status,
        scenario_ids=list(value.scenario_ids),
        results=[dict(item) for item in value.results],
        blocker_codes=list(value.blocker_codes),
        report_digest=value.report_digest,
        verifier_version=value.verifier_version,
        requested_by_subject_id=value.requested_by_subject_id,
        completed_by_subject_id=value.completed_by_subject_id,
        started_at=value.started_at,
        completed_at=value.completed_at,
        version=value.version,
    )


def _defect_response(value: ProductionDefectView) -> ProductionDefectResponse:
    return ProductionDefectResponse(
        **{field: getattr(value, field) for field in ProductionDefectResponse.model_fields}
    )


def _acceptance_response(value: ProductionAcceptanceView) -> ProductionAcceptanceResponse:
    return ProductionAcceptanceResponse(
        acceptance_id=value.acceptance_id,
        release_scope=value.release_scope,
        target_environment=value.target_environment,
        security_exercise_id=value.security_exercise_id,
        model_release_id=value.model_release_id,
        staging_dynamic_manifest_id=value.staging_dynamic_manifest_id,
        staging_dynamic_manifest_status=value.staging_dynamic_manifest_status,
        staging_dynamic_manifest_generated_at=value.staging_dynamic_manifest_generated_at,
        staging_acceptance_attestation_evidence_id=(
            value.staging_acceptance_attestation_evidence_id
        ),
        staging_acceptance_attestation_status=(
            value.staging_acceptance_attestation_status
        ),
        staging_acceptance_attestation_certificate_identity=(
            value.staging_acceptance_attestation_certificate_identity
        ),
        staging_acceptance_attestation_certificate_oidc_issuer=(
            value.staging_acceptance_attestation_certificate_oidc_issuer
        ),
        status=value.status,
        blocker_codes=list(value.blocker_codes),
        readiness_snapshot=dict(value.readiness_snapshot),
        readiness_digest=value.readiness_digest,
        created_by_subject_id=value.created_by_subject_id,
        evaluated_at=value.evaluated_at,
        signed_off_at=value.signed_off_at,
        signoffs=[
            AcceptanceSignoffResponse(
                **{field: getattr(item, field) for field in AcceptanceSignoffResponse.model_fields}
            )
            for item in value.signoffs
        ],
        version=value.version,
    )


def _legal_actions(identity: IdentityContext) -> list[str]:
    actions: list[str] = []
    if Role.SECURITY_AUDITOR in identity.roles:
        actions.extend(["MANAGE_SECURITY_EXERCISE", "SIGN_SECURITY_ACCEPTANCE"])
    if Role.PLATFORM_OPERATOR in identity.roles:
        actions.extend(["CREATE_PRODUCTION_ACCEPTANCE", "SIGN_PLATFORM_ACCEPTANCE"])
    if Role.TENANT_ADMIN in identity.roles:
        actions.append("SIGN_BUSINESS_ACCEPTANCE")
    return actions


def _version(value: str) -> int:
    match = re.fullmatch(r'"?([1-9][0-9]*)"?', value.strip())
    if match is None:
        raise AppError(400, "invalid_version_precondition", "validation", "If-Match is invalid")
    return int(match.group(1))


def _translate(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(
            403, "assurance_action_forbidden", "authorization", "Assurance action was denied"
        )
    if isinstance(exc, AssuranceNotFound):
        return AppError(404, str(exc), "not_found", "Assurance resource was not found")
    if isinstance(exc, AssuranceConflict):
        return AppError(409, str(exc), "conflict", "Assurance state rejected this action")
    return AppError(400, "invalid_assurance_request", "validation", str(exc))
