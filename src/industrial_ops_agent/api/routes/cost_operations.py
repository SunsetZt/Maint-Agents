"""Tenant-scoped FinOps attribution, budgets, and versioned rate policy API."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Request, Response, status
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.cost_operations.service import (
    CostLedgerBatchReceipt,
    CostLedgerConflict,
    CostLedgerEntryInput,
    CostOperationsService,
    CostOperationsSnapshot,
    CostPolicy,
    CostPolicyConflict,
)
from industrial_ops_agent.persistence.database import Database

router = APIRouter(tags=["cost-operations"])

BudgetStatusValue = Literal[
    "ON_TRACK",
    "WARNING",
    "EXCEEDED",
    "EVIDENCE_INCOMPLETE",
    "UNCONFIGURED",
]


class CostPolicyBody(BaseModel):
    prompt_tokens_per_million_usd: float = Field(ge=0, le=10_000)
    completion_tokens_per_million_usd: float = Field(ge=0, le=10_000)
    gpu_hourly_cost_usd: float = Field(ge=0, le=10_000)
    monthly_budget_usd: float = Field(gt=0, le=1_000_000_000)
    scenario_budgets_usd: dict[str, float] = Field(default_factory=dict)
    warning_ratio: float = Field(ge=0.1, le=1.0)
    reason: str = Field(min_length=8, max_length=1_000)


class CostPolicyResponse(CostPolicyBody):
    policy_id: str
    version: int
    effective_at: datetime
    updated_by_subject_id: str


class CostBreakdownResponse(BaseModel):
    key: str
    label: str
    request_count: int
    succeeded_count: int
    failed_count: int
    prompt_tokens: int
    completion_tokens: int
    estimated_gpu_hours: float
    ledger_entry_count: int
    ledger_known_cost_usd: float
    known_cost_usd: float
    cost_complete: bool


class DailyCostResponse(BaseModel):
    day: date
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    ledger_entry_count: int
    ledger_known_cost_usd: float
    known_cost_usd: float
    cost_complete: bool


class ScenarioBudgetResponse(BaseModel):
    scenario: str
    budget_usd: float
    known_cost_usd: float
    remaining_usd: float
    consumed_ratio: float
    status: BudgetStatusValue


class BudgetPostureResponse(BaseModel):
    status: BudgetStatusValue
    monthly_budget_usd: float | None
    known_cost_usd: float
    remaining_usd: float | None
    consumed_ratio: float | None
    warning_ratio: float | None
    scenarios: list[ScenarioBudgetResponse]


class CostTotalsResponse(BaseModel):
    inference_request_count: int
    succeeded_inference_count: int
    failed_inference_count: int
    pending_inference_count: int
    prompt_tokens: int
    completion_tokens: int
    estimated_gpu_hours: float
    inference_known_cost_usd: float
    training_known_cost_usd: float
    ledger_entry_count: int
    ledger_known_cost_usd: float
    total_known_cost_usd: float
    failed_inference_known_cost_usd: float
    missing_cost_record_count: int
    business_unattributed_request_count: int


class CostOperationsResponse(BaseModel):
    policy_version: str
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    ledger_coverage_cutoff: datetime
    evidence_status: Literal["COMPLETE", "PARTIAL", "UNCONFIGURED"]
    current_policy: CostPolicyResponse | None
    totals: CostTotalsResponse
    budget: BudgetPostureResponse
    anomalies: list[str]
    by_release: list[CostBreakdownResponse]
    by_scenario: list[CostBreakdownResponse]
    by_business_task: list[CostBreakdownResponse]
    daily: list[DailyCostResponse]
    ledger_by_category: list["LedgerCategoryCostResponse"]
    ledger_batches: list["CostLedgerBatchSummaryResponse"]
    unmetered_categories: list[str]


class CostOperationsEnvelope(BaseModel):
    data: CostOperationsResponse
    legal_actions: list[Literal["UPDATE_POLICY", "IMPORT_LEDGER"]]
    meta: dict[str, str]


class CostPolicyEnvelope(BaseModel):
    data: CostPolicyResponse
    meta: dict[str, str]


class CostLedgerEntryBody(BaseModel):
    external_entry_id: str = Field(min_length=1, max_length=128)
    category: Literal[
        "object_storage",
        "network_egress",
        "search_service",
        "external_tool_api",
        "manual_annotation",
    ]
    occurred_at: datetime
    amount_usd: float = Field(ge=0, le=1_000_000_000)
    quantity: float | None = Field(default=None, ge=0, le=1_000_000_000_000)
    unit: str | None = Field(default=None, min_length=1, max_length=32)
    release_id: str | None = Field(default=None, min_length=1, max_length=128)
    scenario: Literal["DIAGNOSIS", "VLM", "ASR", "TTS"] | None = None
    diagnosis_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    work_order_id: str | None = Field(default=None, min_length=1, max_length=128)


class CostLedgerBatchBody(BaseModel):
    source_system: str = Field(min_length=1, max_length=128)
    external_batch_id: str = Field(min_length=1, max_length=128)
    coverage_start: datetime
    coverage_end: datetime
    covered_categories: list[
        Literal[
            "object_storage",
            "network_egress",
            "search_service",
            "external_tool_api",
            "manual_annotation",
        ]
    ] = Field(min_length=1, max_length=5)
    currency: Literal["USD"]
    evidence_uri: str = Field(min_length=6, max_length=1_024)
    evidence_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    batch_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    entries: list[CostLedgerEntryBody] = Field(min_length=1, max_length=2_000)


class CostLedgerBatchReceiptResponse(BaseModel):
    batch_id: str
    source_system: str
    external_batch_id: str
    coverage_start: datetime
    coverage_end: datetime
    covered_categories: list[str]
    currency: str
    evidence_uri: str
    evidence_sha256: str
    batch_digest: str
    entry_count: int
    total_amount_usd: float
    imported_by_subject_id: str
    imported_at: datetime


class CostLedgerBatchEnvelope(BaseModel):
    data: CostLedgerBatchReceiptResponse
    meta: dict[str, str]


class LedgerCategoryCostResponse(BaseModel):
    category: str
    entry_count: int
    known_cost_usd: float
    coverage_complete: bool


class CostLedgerBatchSummaryResponse(BaseModel):
    batch_id: str
    source_system: str
    external_batch_id: str
    coverage_start: datetime
    coverage_end: datetime
    covered_categories: list[str]
    batch_digest: str
    evidence_uri: str
    evidence_sha256: str
    entry_count: int
    total_amount_usd: float
    imported_by_subject_id: str
    imported_at: datetime


@router.get("/operations/cost", response_model=CostOperationsEnvelope)
async def get_cost_operations(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> CostOperationsEnvelope:
    service = CostOperationsService(database, authorizer)
    try:
        snapshot = await asyncio.to_thread(
            service.snapshot,
            identity,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "cost_operations_forbidden") from exc
    can_update = authorizer.decide(
        identity,
        Action.MANAGE_COST_POLICY,
        ResourceContext(tenant_id=identity.tenant_id, resource_id="cost-operations"),
    ).allowed
    return CostOperationsEnvelope(
        data=_snapshot(snapshot),
        legal_actions=["UPDATE_POLICY", "IMPORT_LEDGER"] if can_update else [],
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/operations/cost-ledger/batches",
    response_model=CostLedgerBatchEnvelope,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_200_OK: {
            "model": CostLedgerBatchEnvelope,
            "description": "Idempotent replay of the existing immutable batch",
        }
    },
)
async def import_cost_ledger_batch(
    body: CostLedgerBatchBody,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)],
) -> CostLedgerBatchEnvelope:
    try:
        result = await asyncio.to_thread(
            CostOperationsService(database, authorizer).import_ledger_batch,
            identity,
            source_system=body.source_system,
            external_batch_id=body.external_batch_id,
            coverage_start=body.coverage_start,
            coverage_end=body.coverage_end,
            covered_categories=tuple(body.covered_categories),
            currency=body.currency,
            evidence_uri=body.evidence_uri,
            evidence_sha256=body.evidence_sha256,
            batch_digest=body.batch_digest,
            entries=tuple(
                CostLedgerEntryInput(**item.model_dump()) for item in body.entries
            ),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "cost_ledger_import_forbidden") from exc
    except CostLedgerConflict as exc:
        raise AppError(
            status_code=409,
            code=exc.reason,
            category="conflict",
            message=exc.reason,
        ) from exc
    except ValueError as exc:
        code = str(exc) if str(exc).startswith("cost_ledger_") else "cost_ledger_invalid"
        raise _error(422, code) from exc
    if result.replayed:
        response.status_code = status.HTTP_200_OK
    return CostLedgerBatchEnvelope(
        data=_ledger_receipt(result.receipt),
        meta={
            "request_id": request.state.request_id,
            "replayed": "true" if result.replayed else "false",
        },
    )


@router.put("/operations/cost/policy", response_model=CostPolicyEnvelope)
async def update_cost_policy(
    body: CostPolicyBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> CostPolicyEnvelope:
    try:
        policy = await asyncio.to_thread(
            CostOperationsService(database, authorizer).update_policy,
            identity,
            **body.model_dump(),
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "cost_policy_forbidden") from exc
    except CostPolicyConflict as exc:
        raise AppError(
            status_code=409,
            code=exc.reason,
            category="conflict",
            message="Cost policy update conflicted with current state",
            details={"current_version": exc.current_version},
        ) from exc
    except ValueError as exc:
        raise _error(422, "cost_policy_invalid") from exc
    return CostPolicyEnvelope(
        data=_policy(policy),
        meta={"request_id": request.state.request_id},
    )


def _snapshot(snapshot: CostOperationsSnapshot) -> CostOperationsResponse:
    return CostOperationsResponse(
        policy_version=snapshot.policy_version,
        generated_at=snapshot.generated_at,
        window_start=snapshot.window_start,
        window_end=snapshot.window_end,
        ledger_coverage_cutoff=snapshot.ledger_coverage_cutoff,
        evidence_status=snapshot.evidence_status,
        current_policy=(
            _policy(snapshot.current_policy) if snapshot.current_policy is not None else None
        ),
        totals=CostTotalsResponse.model_validate(snapshot.totals, from_attributes=True),
        budget=BudgetPostureResponse(
            status=snapshot.budget.status,
            monthly_budget_usd=snapshot.budget.monthly_budget_usd,
            known_cost_usd=snapshot.budget.known_cost_usd,
            remaining_usd=snapshot.budget.remaining_usd,
            consumed_ratio=snapshot.budget.consumed_ratio,
            warning_ratio=snapshot.budget.warning_ratio,
            scenarios=[
                ScenarioBudgetResponse.model_validate(item, from_attributes=True)
                for item in snapshot.budget.scenarios
            ],
        ),
        anomalies=list(snapshot.anomalies),
        by_release=[
            CostBreakdownResponse.model_validate(item, from_attributes=True)
            for item in snapshot.by_release
        ],
        by_scenario=[
            CostBreakdownResponse.model_validate(item, from_attributes=True)
            for item in snapshot.by_scenario
        ],
        by_business_task=[
            CostBreakdownResponse.model_validate(item, from_attributes=True)
            for item in snapshot.by_business_task
        ],
        daily=[
            DailyCostResponse.model_validate(item, from_attributes=True) for item in snapshot.daily
        ],
        ledger_by_category=[
            LedgerCategoryCostResponse.model_validate(item, from_attributes=True)
            for item in snapshot.ledger_by_category
        ],
        ledger_batches=[
            CostLedgerBatchSummaryResponse.model_validate(item, from_attributes=True)
            for item in snapshot.ledger_batches
        ],
        unmetered_categories=list(snapshot.unmetered_categories),
    )


def _ledger_receipt(receipt: CostLedgerBatchReceipt) -> CostLedgerBatchReceiptResponse:
    return CostLedgerBatchReceiptResponse(
        batch_id=receipt.batch_id,
        source_system=receipt.source_system,
        external_batch_id=receipt.external_batch_id,
        coverage_start=receipt.coverage_start,
        coverage_end=receipt.coverage_end,
        covered_categories=list(receipt.covered_categories),
        currency=receipt.currency,
        evidence_uri=receipt.evidence_uri,
        evidence_sha256=receipt.evidence_sha256,
        batch_digest=receipt.batch_digest,
        entry_count=receipt.entry_count,
        total_amount_usd=receipt.total_amount_usd,
        imported_by_subject_id=receipt.imported_by_subject_id,
        imported_at=receipt.imported_at,
    )


def _policy(policy: CostPolicy) -> CostPolicyResponse:
    return CostPolicyResponse(
        policy_id=policy.policy_id,
        version=policy.version,
        effective_at=policy.effective_at,
        prompt_tokens_per_million_usd=policy.prompt_tokens_per_million_usd,
        completion_tokens_per_million_usd=policy.completion_tokens_per_million_usd,
        gpu_hourly_cost_usd=policy.gpu_hourly_cost_usd,
        monthly_budget_usd=policy.monthly_budget_usd,
        scenario_budgets_usd=policy.scenario_budgets_usd,
        warning_ratio=policy.warning_ratio,
        reason=policy.reason,
        updated_by_subject_id=policy.updated_by_subject_id,
    )


def _version(value: str) -> int:
    try:
        version = int(value.strip().strip('"'))
    except ValueError as exc:
        raise _error(422, "cost_policy_version_invalid") from exc
    if version < 0:
        raise _error(422, "cost_policy_version_invalid")
    return version


def _error(status_code: int, code: str) -> AppError:
    return AppError(
        status_code=status_code,
        code=code,
        category="authorization" if status_code == 403 else "validation",
        message="Cost operations request was rejected",
    )
