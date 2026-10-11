"""Recovery evidence and drill operations API."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt

from industrial_ops_agent.api.dependencies import get_identity, get_recovery_service
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.recovery.service import (
    ComponentRecoveryResult,
    ConsistencyCheck,
    EvidenceRegistration,
    EvidenceStatus,
    RecoveryComponent,
    RecoveryConflict,
    RecoveryDrillView,
    RecoveryEvidenceView,
    RecoveryNotFound,
    RecoveryOverview,
    RecoveryService,
)

router = APIRouter(tags=["recovery"])


class RecoveryEvidenceBody(BaseModel):
    component: RecoveryComponent
    backup_id: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
    backup_type: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
    recovery_point_at: datetime
    captured_at: datetime
    verification_status: EvidenceStatus
    verification_method: str = Field(min_length=1, max_length=64)
    evidence_ref: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    artifact_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    immutable: bool
    encrypted: bool
    details: dict[str, StrictInt | StrictFloat | StrictBool] = Field(default_factory=dict)


class StartRecoveryDrillBody(BaseModel):
    scenario: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
    scope: list[RecoveryComponent] = Field(min_length=1, max_length=7)
    outage_detected_at: datetime


class ConsistencyCheckBody(BaseModel):
    check_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    status: Literal["PASSED", "FAILED"]
    evidence_ref: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )


class ComponentRecoveryResultBody(BaseModel):
    component: RecoveryComponent
    recovered_to_at: datetime
    service_restored_at: datetime
    evidence_ids: list[
        Annotated[
            str,
            Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"),
        ]
    ] = Field(min_length=1, max_length=20)
    checks: list[ConsistencyCheckBody] = Field(min_length=1, max_length=20)


class CompleteRecoveryDrillBody(BaseModel):
    component_results: list[ComponentRecoveryResultBody] = Field(min_length=1, max_length=7)


class RecoveryEvidenceResponse(BaseModel):
    evidence_id: str
    component: str
    backup_id: str
    backup_type: str
    recovery_point_at: datetime
    captured_at: datetime
    verification_status: str
    verification_method: str
    evidence_ref: str
    artifact_digest: str
    immutable: bool
    encrypted: bool
    details: dict[str, Any]
    verified_by_subject_id: str
    verified_at: datetime
    version: int


class RecoveryDrillResponse(BaseModel):
    drill_id: str
    scenario: str
    scope: list[str]
    status: str
    outage_detected_at: datetime
    started_at: datetime
    completed_at: datetime | None
    results: list[dict[str, Any]]
    blocker_codes: list[str]
    requested_by_subject_id: str
    completed_by_subject_id: str | None
    objective_met: bool
    version: int


class ComponentRecoveryPostureResponse(BaseModel):
    component: str
    rpo_seconds: int | None
    rto_seconds: int
    recovery_method: str
    evidence_status: Literal["COMPLIANT", "MISSING", "STALE", "FAILED"]
    latest_evidence: RecoveryEvidenceResponse | None


class DegradationPolicyResponse(BaseModel):
    failure: str
    system_behavior: str
    prohibited_behavior: str


class RecoveryOverviewResponse(BaseModel):
    policy_version: str
    generated_at: datetime
    release_gate_enforced: bool
    release_allowed: bool
    release_gate_reasons: list[str]
    monthly_postgres_restore_due: bool
    quarterly_cross_component_drill_due: bool
    components: list[ComponentRecoveryPostureResponse]
    recent_drills: list[RecoveryDrillResponse]
    degradation_policies: list[DegradationPolicyResponse]


class RecoveryOverviewEnvelope(BaseModel):
    data: RecoveryOverviewResponse
    legal_actions: list[str]
    meta: dict[str, str]


class RecoveryEvidenceEnvelope(BaseModel):
    data: RecoveryEvidenceResponse
    meta: dict[str, str]


class RecoveryDrillEnvelope(BaseModel):
    data: RecoveryDrillResponse
    meta: dict[str, str]


@router.get("/recovery/overview", response_model=RecoveryOverviewEnvelope)
async def get_recovery_overview(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[RecoveryService, Depends(get_recovery_service)],
) -> RecoveryOverviewEnvelope:
    try:
        overview = await asyncio.to_thread(
            service.overview,
            identity,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _translate(exc) from exc
    return RecoveryOverviewEnvelope(
        data=_overview_response(overview),
        legal_actions=["REGISTER_RECOVERY_EVIDENCE", "MANAGE_RECOVERY_DRILL"]
        if any(role.value == "platform_operator" for role in identity.roles)
        else [],
        meta={"request_id": request.state.request_id},
    )


@router.post("/recovery/evidence", response_model=RecoveryEvidenceEnvelope, status_code=201)
async def register_recovery_evidence(
    body: RecoveryEvidenceBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[RecoveryService, Depends(get_recovery_service)],
) -> RecoveryEvidenceEnvelope:
    try:
        evidence = await asyncio.to_thread(
            service.register_evidence,
            identity,
            EvidenceRegistration(
                component=body.component,
                backup_id=body.backup_id,
                backup_type=body.backup_type,
                recovery_point_at=body.recovery_point_at,
                captured_at=body.captured_at,
                verification_status=body.verification_status,
                verification_method=body.verification_method,
                evidence_ref=body.evidence_ref,
                artifact_digest=body.artifact_digest,
                immutable=body.immutable,
                encrypted=body.encrypted,
                details=dict(body.details),
            ),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, RecoveryConflict, ValueError) as exc:
        raise _translate(exc) from exc
    return RecoveryEvidenceEnvelope(
        data=_evidence_response(evidence),
        meta={"request_id": request.state.request_id},
    )


@router.post("/recovery/drills", response_model=RecoveryDrillEnvelope, status_code=201)
async def start_recovery_drill(
    body: StartRecoveryDrillBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[RecoveryService, Depends(get_recovery_service)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ],
) -> RecoveryDrillEnvelope:
    try:
        drill = await asyncio.to_thread(
            service.start_drill,
            identity,
            idempotency_key=idempotency_key,
            scenario=body.scenario,
            scope=tuple(body.scope),
            outage_detected_at=body.outage_detected_at,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, RecoveryConflict, ValueError) as exc:
        raise _translate(exc) from exc
    return RecoveryDrillEnvelope(
        data=_drill_response(drill),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/recovery/drills/{drill_id}/complete",
    response_model=RecoveryDrillEnvelope,
)
async def complete_recovery_drill(
    drill_id: str,
    body: CompleteRecoveryDrillBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    service: Annotated[RecoveryService, Depends(get_recovery_service)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> RecoveryDrillEnvelope:
    try:
        drill = await asyncio.to_thread(
            service.complete_drill,
            identity,
            drill_id,
            expected_version=_version(if_match),
            component_results=tuple(
                ComponentRecoveryResult(
                    component=item.component,
                    recovered_to_at=item.recovered_to_at,
                    service_restored_at=item.service_restored_at,
                    evidence_ids=tuple(item.evidence_ids),
                    checks=tuple(
                        ConsistencyCheck(
                            check_id=check.check_id,
                            status=check.status,
                            evidence_ref=check.evidence_ref,
                        )
                        for check in item.checks
                    ),
                )
                for item in body.component_results
            ),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, RecoveryConflict, RecoveryNotFound, ValueError) as exc:
        raise _translate(exc) from exc
    return RecoveryDrillEnvelope(
        data=_drill_response(drill),
        meta={"request_id": request.state.request_id},
    )


def _overview_response(value: RecoveryOverview) -> RecoveryOverviewResponse:
    return RecoveryOverviewResponse(
        policy_version=value.policy_version,
        generated_at=value.generated_at,
        release_gate_enforced=value.release_gate_enforced,
        release_allowed=value.release_allowed,
        release_gate_reasons=list(value.release_gate_reasons),
        monthly_postgres_restore_due=value.monthly_postgres_restore_due,
        quarterly_cross_component_drill_due=value.quarterly_cross_component_drill_due,
        components=[
            ComponentRecoveryPostureResponse(
                component=item.component,
                rpo_seconds=item.rpo_seconds,
                rto_seconds=item.rto_seconds,
                recovery_method=item.recovery_method,
                evidence_status=item.evidence_status,
                latest_evidence=(
                    _evidence_response(item.latest_evidence)
                    if item.latest_evidence is not None
                    else None
                ),
            )
            for item in value.components
        ],
        recent_drills=[_drill_response(item) for item in value.recent_drills],
        degradation_policies=[
            DegradationPolicyResponse(
                failure=item.failure,
                system_behavior=item.system_behavior,
                prohibited_behavior=item.prohibited_behavior,
            )
            for item in value.degradation_policies
        ],
    )


def _evidence_response(value: RecoveryEvidenceView) -> RecoveryEvidenceResponse:
    return RecoveryEvidenceResponse(
        **{field: getattr(value, field) for field in RecoveryEvidenceResponse.model_fields}
    )


def _drill_response(value: RecoveryDrillView) -> RecoveryDrillResponse:
    return RecoveryDrillResponse(
        drill_id=value.drill_id,
        scenario=value.scenario,
        scope=list(value.scope),
        status=value.status,
        outage_detected_at=value.outage_detected_at,
        started_at=value.started_at,
        completed_at=value.completed_at,
        results=[dict(item) for item in value.results],
        blocker_codes=list(value.blocker_codes),
        requested_by_subject_id=value.requested_by_subject_id,
        completed_by_subject_id=value.completed_by_subject_id,
        objective_met=value.objective_met,
        version=value.version,
    )


def _version(value: str) -> int:
    match = re.fullmatch(r'"?([1-9][0-9]*)"?', value.strip())
    if match is None:
        raise AppError(
            400,
            "invalid_version_precondition",
            "validation",
            "If-Match is invalid",
        )
    return int(match.group(1))


def _translate(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(
            403,
            "recovery_action_forbidden",
            "authorization",
            "Recovery action was denied",
        )
    if isinstance(exc, RecoveryNotFound):
        return AppError(404, str(exc), "not_found", "Recovery drill was not found")
    if isinstance(exc, RecoveryConflict):
        return AppError(409, str(exc), "conflict", "Recovery state changed; refresh and retry")
    return AppError(400, "invalid_recovery_request", "validation", str(exc))
