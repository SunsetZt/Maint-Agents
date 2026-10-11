"""Versioned rates, tenant cost attribution, and budget posture."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.maintenance_planning.review_isolation import (
    ReviewIsolationConflict,
    diagnosis_for_agent,
    lock_subject,
    record_disclosure,
    record_incident_disclosure,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CostLedgerBatchRecord,
    CostLedgerEntryRecord,
    CostPolicyRecord,
    DiagnosisRunRecord,
    ModelDeploymentRecord,
    ModelInferenceRecord,
    ModelReleaseRecord,
    TenantRecord,
    TrainingExperimentRecord,
    WorkOrderRecord,
)

_SCENARIO = re.compile(r"^[A-Z][A-Z0-9_-]{0,63}$")
_LEDGER_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LEDGER_UNIT = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_EVIDENCE_URI = re.compile(r"^(?:s3|https)://[^\s]{1,1016}$")
_SUPPORTED_SCENARIOS = frozenset({"DIAGNOSIS", "VLM", "ASR", "TTS"})
_LEDGER_CATEGORIES = (
    "object_storage",
    "network_egress",
    "search_service",
    "external_tool_api",
    "manual_annotation",
)
_LEDGER_CATEGORY_SET = frozenset(_LEDGER_CATEGORIES)


class CostEvidenceStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNCONFIGURED = "UNCONFIGURED"


class BudgetStatus(StrEnum):
    ON_TRACK = "ON_TRACK"
    WARNING = "WARNING"
    EXCEEDED = "EXCEEDED"
    EVIDENCE_INCOMPLETE = "EVIDENCE_INCOMPLETE"
    UNCONFIGURED = "UNCONFIGURED"


class CostPolicyConflict(RuntimeError):
    def __init__(self, reason: str, current_version: int) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


class CostLedgerConflict(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class CostLedgerEntryInput:
    external_entry_id: str
    category: str
    occurred_at: datetime
    amount_usd: float
    quantity: float | None = None
    unit: str | None = None
    release_id: str | None = None
    scenario: str | None = None
    diagnosis_run_id: str | None = None
    work_order_id: str | None = None


@dataclass(frozen=True, slots=True)
class CostLedgerBatchReceipt:
    batch_id: str
    source_system: str
    external_batch_id: str
    coverage_start: datetime
    coverage_end: datetime
    covered_categories: tuple[str, ...]
    currency: str
    evidence_uri: str
    evidence_sha256: str
    batch_digest: str
    entry_count: int
    total_amount_usd: float
    imported_by_subject_id: str
    imported_at: datetime


@dataclass(frozen=True, slots=True)
class CostLedgerImportResult:
    receipt: CostLedgerBatchReceipt
    replayed: bool


@dataclass(frozen=True, slots=True)
class LedgerCategoryCost:
    category: str
    entry_count: int
    known_cost_usd: float
    coverage_complete: bool


@dataclass(frozen=True, slots=True)
class CostLedgerBatchSummary:
    batch_id: str
    source_system: str
    external_batch_id: str
    coverage_start: datetime
    coverage_end: datetime
    covered_categories: tuple[str, ...]
    batch_digest: str
    evidence_uri: str
    evidence_sha256: str
    entry_count: int
    total_amount_usd: float
    imported_by_subject_id: str
    imported_at: datetime


@dataclass(frozen=True, slots=True)
class CostPolicy:
    policy_id: str
    version: int
    effective_at: datetime
    prompt_tokens_per_million_usd: float
    completion_tokens_per_million_usd: float
    gpu_hourly_cost_usd: float
    monthly_budget_usd: float
    scenario_budgets_usd: dict[str, float]
    warning_ratio: float
    reason: str
    updated_by_subject_id: str


@dataclass(frozen=True, slots=True)
class CostBreakdown:
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


@dataclass(frozen=True, slots=True)
class DailyCost:
    day: date
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    ledger_entry_count: int
    ledger_known_cost_usd: float
    known_cost_usd: float
    cost_complete: bool


@dataclass(frozen=True, slots=True)
class ScenarioBudget:
    scenario: str
    budget_usd: float
    known_cost_usd: float
    remaining_usd: float
    consumed_ratio: float
    status: BudgetStatus


@dataclass(frozen=True, slots=True)
class BudgetPosture:
    status: BudgetStatus
    monthly_budget_usd: float | None
    known_cost_usd: float
    remaining_usd: float | None
    consumed_ratio: float | None
    warning_ratio: float | None
    scenarios: tuple[ScenarioBudget, ...]


@dataclass(frozen=True, slots=True)
class CostTotals:
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


@dataclass(frozen=True, slots=True)
class CostOperationsSnapshot:
    policy_version: str
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    ledger_coverage_cutoff: datetime
    evidence_status: CostEvidenceStatus
    current_policy: CostPolicy | None
    totals: CostTotals
    budget: BudgetPosture
    anomalies: tuple[str, ...]
    by_release: tuple[CostBreakdown, ...]
    by_scenario: tuple[CostBreakdown, ...]
    by_business_task: tuple[CostBreakdown, ...]
    daily: tuple[DailyCost, ...]
    ledger_by_category: tuple[LedgerCategoryCost, ...]
    ledger_batches: tuple[CostLedgerBatchSummary, ...]
    unmetered_categories: tuple[str, ...]


@dataclass(slots=True)
class _Accumulator:
    key: str
    label: str
    request_count: int = 0
    succeeded_count: int = 0
    failed_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_gpu_hours: float = 0.0
    ledger_entry_count: int = 0
    ledger_known_cost_usd: float = 0.0
    known_cost_usd: float = 0.0
    cost_complete: bool = True

    def add(self, allocation: _InferenceAllocation) -> None:
        self.request_count += 1
        self.succeeded_count += int(allocation.status == "SUCCEEDED")
        self.failed_count += int(allocation.status == "FAILED")
        self.prompt_tokens += allocation.prompt_tokens
        self.completion_tokens += allocation.completion_tokens
        self.estimated_gpu_hours += allocation.estimated_gpu_hours
        self.known_cost_usd += allocation.known_cost_usd
        self.cost_complete = self.cost_complete and allocation.cost_complete

    def add_ledger(self, allocation: _LedgerAllocation) -> None:
        self.ledger_entry_count += 1
        self.ledger_known_cost_usd += allocation.amount_usd
        self.known_cost_usd += allocation.amount_usd

    def freeze(self) -> CostBreakdown:
        return CostBreakdown(
            key=self.key,
            label=self.label,
            request_count=self.request_count,
            succeeded_count=self.succeeded_count,
            failed_count=self.failed_count,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            estimated_gpu_hours=_quantity(self.estimated_gpu_hours),
            ledger_entry_count=self.ledger_entry_count,
            ledger_known_cost_usd=_money(self.ledger_known_cost_usd),
            known_cost_usd=_money(self.known_cost_usd),
            cost_complete=self.cost_complete,
        )


@dataclass(frozen=True, slots=True)
class _InferenceAllocation:
    release_id: str
    model_alias: str
    scenario: str
    business_key: str
    business_label: str
    day: date
    status: str
    prompt_tokens: int
    completion_tokens: int
    estimated_gpu_hours: float
    known_cost_usd: float
    cost_complete: bool


@dataclass(frozen=True, slots=True)
class _LedgerAllocation:
    category: str
    day: date
    amount_usd: float
    release_id: str | None
    scenario: str | None
    business_key: str
    business_label: str


@dataclass(slots=True)
class _DailyAccumulator:
    day: date
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ledger_entry_count: int = 0
    ledger_known_cost_usd: float = 0.0
    known_cost_usd: float = 0.0
    cost_complete: bool = True

    def add(self, allocation: _InferenceAllocation) -> None:
        self.request_count += 1
        self.prompt_tokens += allocation.prompt_tokens
        self.completion_tokens += allocation.completion_tokens
        self.known_cost_usd += allocation.known_cost_usd
        self.cost_complete = self.cost_complete and allocation.cost_complete

    def add_ledger(self, allocation: _LedgerAllocation) -> None:
        self.ledger_entry_count += 1
        self.ledger_known_cost_usd += allocation.amount_usd
        self.known_cost_usd += allocation.amount_usd

    def freeze(self) -> DailyCost:
        return DailyCost(
            day=self.day,
            request_count=self.request_count,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            ledger_entry_count=self.ledger_entry_count,
            ledger_known_cost_usd=_money(self.ledger_known_cost_usd),
            known_cost_usd=_money(self.known_cost_usd),
            cost_complete=self.cost_complete,
        )


class CostOperationsService:
    POLICY_VERSION = "cost-attribution/v1"

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def snapshot(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        now: datetime | None = None,
    ) -> CostOperationsSnapshot:
        self._require(identity, Action.READ_OPERATIONS, request_id)
        generated_at = _as_utc(now or datetime.now(UTC))
        window_start = _month_start(generated_at)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            policies = tuple(
                session.scalars(
                    select(CostPolicyRecord)
                    .where(
                        CostPolicyRecord.tenant_id == identity.tenant_id,
                        CostPolicyRecord.effective_at <= generated_at,
                    )
                    .order_by(CostPolicyRecord.effective_at, CostPolicyRecord.version)
                )
            )
            inferences = tuple(
                session.scalars(
                    select(ModelInferenceRecord)
                    .where(
                        ModelInferenceRecord.tenant_id == identity.tenant_id,
                        ModelInferenceRecord.created_at >= window_start,
                        ModelInferenceRecord.created_at < generated_at,
                    )
                    .order_by(ModelInferenceRecord.created_at)
                )
            )
            release_ids = {item.resolved_release_id for item in inferences}
            deployments = (
                tuple(
                    session.scalars(
                        select(ModelDeploymentRecord).where(
                            ModelDeploymentRecord.tenant_id == identity.tenant_id,
                            ModelDeploymentRecord.release_id.in_(release_ids),
                        )
                    )
                )
                if release_ids
                else ()
            )
            diagnosis_ids = {
                item.diagnosis_run_id for item in inferences if item.diagnosis_run_id is not None
            }
            diagnoses = (
                tuple(
                    session.scalars(
                        select(DiagnosisRunRecord).where(
                            DiagnosisRunRecord.tenant_id == identity.tenant_id,
                            DiagnosisRunRecord.diagnosis_run_id.in_(diagnosis_ids),
                        )
                    )
                )
                if diagnosis_ids
                else ()
            )
            incident_ids = {item.incident_id for item in diagnoses}
            work_orders = (
                tuple(
                    session.scalars(
                        select(WorkOrderRecord)
                        .where(
                            WorkOrderRecord.tenant_id == identity.tenant_id,
                            WorkOrderRecord.incident_id.in_(incident_ids),
                        )
                        .order_by(WorkOrderRecord.created_at.desc())
                    )
                )
                if incident_ids
                else ()
            )
            experiments = tuple(
                session.scalars(
                    select(TrainingExperimentRecord).where(
                        TrainingExperimentRecord.tenant_id == identity.tenant_id,
                        TrainingExperimentRecord.completed_at >= window_start,
                        TrainingExperimentRecord.completed_at < generated_at,
                    )
                )
            )
            ledger_batches = tuple(
                session.scalars(
                    select(CostLedgerBatchRecord)
                    .where(
                        CostLedgerBatchRecord.tenant_id == identity.tenant_id,
                        CostLedgerBatchRecord.coverage_start < generated_at,
                        CostLedgerBatchRecord.coverage_end > window_start,
                    )
                    .order_by(
                        CostLedgerBatchRecord.coverage_start.desc(),
                        CostLedgerBatchRecord.created_at.desc(),
                    )
                )
            )
            ledger_entries = tuple(
                session.scalars(
                    select(CostLedgerEntryRecord)
                    .where(
                        CostLedgerEntryRecord.tenant_id == identity.tenant_id,
                        CostLedgerEntryRecord.occurred_at >= window_start,
                        CostLedgerEntryRecord.occurred_at < generated_at,
                    )
                    .order_by(CostLedgerEntryRecord.occurred_at)
                )
            )

            disclosed_diagnoses = diagnosis_ids | {
                item.diagnosis_run_id for item in ledger_entries if item.diagnosis_run_id
            }
            disclosed_diagnoses.update(
                diagnosis_for_agent(session, identity.tenant_id, item.agent_run_id)
                for item in inferences
                if item.diagnosis_run_id is None and item.agent_run_id is not None
            )
            record_disclosure(
                session,
                identity,
                disclosed_diagnoses,
                resource_kind="cost-snapshot",
                resource_id="operations-cost",
                request_id=request_id,
            )
            ledger_work_order_ids = {
                item.work_order_id for item in ledger_entries if item.work_order_id
            }
            if ledger_work_order_ids:
                ledger_orders = session.scalars(
                    select(WorkOrderRecord).where(
                        WorkOrderRecord.tenant_id == identity.tenant_id,
                        WorkOrderRecord.work_order_id.in_(ledger_work_order_ids),
                    )
                ).all()
                if len(ledger_orders) != len(ledger_work_order_ids):
                    raise ReviewIsolationConflict("source_unavailable")
                record_incident_disclosure(
                    session,
                    identity,
                    (item.incident_id for item in ledger_orders),
                    resource_kind="cost-snapshot",
                    resource_id="operations-cost",
                    request_id=request_id,
                )

        gpu_counts = {item.release_id: _gpu_count(item.desired_spec_json) for item in deployments}
        diagnosis_incidents = {item.diagnosis_run_id: item.incident_id for item in diagnoses}
        incident_work_orders: dict[str, list[str]] = {}
        for work_order in work_orders:
            incident_work_orders.setdefault(work_order.incident_id, []).append(
                work_order.work_order_id
            )
        allocations = tuple(
            _allocate_inference(
                item,
                policies,
                gpu_counts.get(item.resolved_release_id),
                diagnosis_incidents,
                incident_work_orders,
            )
            for item in inferences
        )
        current_policy = _cost_policy(policies[-1]) if policies else None
        training_known_cost, training_missing = _training_cost(experiments)
        ledger_allocations = tuple(_ledger_allocation(item) for item in ledger_entries)
        return _snapshot(
            generated_at=generated_at,
            window_start=window_start,
            current_policy=current_policy,
            allocations=allocations,
            training_known_cost=training_known_cost,
            training_missing=training_missing,
            ledger_allocations=ledger_allocations,
            ledger_batches=ledger_batches,
        )

    def import_ledger_batch(
        self,
        identity: IdentityContext,
        *,
        source_system: str,
        external_batch_id: str,
        coverage_start: datetime,
        coverage_end: datetime,
        covered_categories: tuple[str, ...],
        currency: str,
        evidence_uri: str,
        evidence_sha256: str,
        batch_digest: str,
        entries: tuple[CostLedgerEntryInput, ...],
        idempotency_key: str,
        request_id: str,
        now: datetime | None = None,
    ) -> CostLedgerImportResult:
        self._require(identity, Action.MANAGE_COST_POLICY, request_id)
        imported_at = _as_utc(now or datetime.now(UTC))
        normalized_start, normalized_end, categories = _validate_ledger_batch(
            source_system=source_system,
            external_batch_id=external_batch_id,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            covered_categories=covered_categories,
            currency=currency,
            evidence_uri=evidence_uri,
            evidence_sha256=evidence_sha256,
            batch_digest=batch_digest,
            entries=entries,
            idempotency_key=idempotency_key,
            now=imported_at,
        )
        expected_digest = canonical_cost_ledger_digest(
            source_system=source_system,
            external_batch_id=external_batch_id,
            coverage_start=normalized_start,
            coverage_end=normalized_end,
            covered_categories=categories,
            currency=currency,
            evidence_uri=evidence_uri,
            evidence_sha256=evidence_sha256,
            entries=entries,
        )
        if batch_digest != expected_digest:
            raise ValueError("cost_ledger_digest_mismatch")

        with self._database.transaction(identity.tenant_context) as session:
            tenant = session.scalar(
                select(TenantRecord).where(TenantRecord.id == identity.tenant_id).with_for_update()
            )
            if tenant is None:
                raise CostLedgerConflict("tenant_not_found")
            replay = session.scalar(
                select(CostLedgerBatchRecord).where(
                    CostLedgerBatchRecord.tenant_id == identity.tenant_id,
                    CostLedgerBatchRecord.imported_by_subject_id == identity.subject_id,
                    CostLedgerBatchRecord.idempotency_key == idempotency_key,
                )
            )
            if replay is not None:
                return _ledger_import_replay(replay, batch_digest)
            external = session.scalar(
                select(CostLedgerBatchRecord).where(
                    CostLedgerBatchRecord.tenant_id == identity.tenant_id,
                    CostLedgerBatchRecord.source_system == source_system,
                    CostLedgerBatchRecord.external_batch_id == external_batch_id,
                )
            )
            if external is not None:
                return _ledger_import_replay(external, batch_digest)
            _validate_ledger_references(session, identity.tenant_id, entries)
            overlapping = tuple(
                session.scalars(
                    select(CostLedgerBatchRecord).where(
                        CostLedgerBatchRecord.tenant_id == identity.tenant_id,
                        CostLedgerBatchRecord.coverage_start < normalized_end,
                        CostLedgerBatchRecord.coverage_end > normalized_start,
                    )
                )
            )
            if any(
                set(item.covered_categories_json).intersection(categories) for item in overlapping
            ):
                raise CostLedgerConflict("cost_ledger_coverage_overlap")

            batch_id = f"cost-ledger-{uuid4()}"
            total = _money(sum(item.amount_usd for item in entries))
            batch = CostLedgerBatchRecord(
                batch_id=batch_id,
                tenant_id=identity.tenant_id,
                source_system=source_system,
                external_batch_id=external_batch_id,
                idempotency_key=idempotency_key,
                coverage_start=normalized_start,
                coverage_end=normalized_end,
                covered_categories_json=list(categories),
                currency=currency,
                evidence_uri=evidence_uri,
                evidence_sha256=evidence_sha256,
                batch_digest=batch_digest,
                entry_count=len(entries),
                total_amount_usd=total,
                imported_by_subject_id=identity.subject_id,
                created_at=imported_at,
                updated_at=imported_at,
            )
            session.add(batch)
            session.add_all(
                [
                    CostLedgerEntryRecord(
                        entry_id=f"cost-entry-{uuid4()}",
                        batch_id=batch_id,
                        tenant_id=identity.tenant_id,
                        external_entry_id=item.external_entry_id,
                        category=item.category,
                        occurred_at=_as_utc(item.occurred_at),
                        amount_usd=item.amount_usd,
                        quantity=item.quantity,
                        unit=item.unit,
                        release_id=item.release_id,
                        scenario=item.scenario,
                        diagnosis_run_id=item.diagnosis_run_id,
                        work_order_id=item.work_order_id,
                        created_at=imported_at,
                        updated_at=imported_at,
                    )
                    for item in entries
                ]
            )
            session.flush()
            return CostLedgerImportResult(_ledger_receipt(batch), replayed=False)

    def update_policy(
        self,
        identity: IdentityContext,
        *,
        prompt_tokens_per_million_usd: float,
        completion_tokens_per_million_usd: float,
        gpu_hourly_cost_usd: float,
        monthly_budget_usd: float,
        scenario_budgets_usd: dict[str, float],
        warning_ratio: float,
        reason: str,
        expected_version: int,
        request_id: str,
        now: datetime | None = None,
    ) -> CostPolicy:
        self._require(identity, Action.MANAGE_COST_POLICY, request_id)
        _validate_policy(
            prompt_tokens_per_million_usd,
            completion_tokens_per_million_usd,
            gpu_hourly_cost_usd,
            monthly_budget_usd,
            scenario_budgets_usd,
            warning_ratio,
            reason,
            expected_version,
        )
        effective_now = _as_utc(now or datetime.now(UTC))
        with self._database.transaction(identity.tenant_context) as session:
            tenant = session.scalar(
                select(TenantRecord).where(TenantRecord.id == identity.tenant_id).with_for_update()
            )
            if tenant is None:
                raise CostPolicyConflict("tenant_not_found", 0)
            current = session.scalar(
                select(CostPolicyRecord)
                .where(CostPolicyRecord.tenant_id == identity.tenant_id)
                .order_by(CostPolicyRecord.version.desc())
                .limit(1)
            )
            current_version = current.version if current is not None else 0
            if current_version != expected_version:
                raise CostPolicyConflict("cost_policy_version_mismatch", current_version)
            record = CostPolicyRecord(
                policy_id=f"cost-policy-{uuid4()}",
                tenant_id=identity.tenant_id,
                version=current_version + 1,
                effective_at=_month_start(effective_now) if current is None else effective_now,
                prompt_tokens_per_million_usd=prompt_tokens_per_million_usd,
                completion_tokens_per_million_usd=completion_tokens_per_million_usd,
                gpu_hourly_cost_usd=gpu_hourly_cost_usd,
                monthly_budget_usd=monthly_budget_usd,
                scenario_budgets_json=dict(sorted(scenario_budgets_usd.items())),
                warning_ratio=warning_ratio,
                reason=reason.strip(),
                updated_by_subject_id=identity.subject_id,
            )
            session.add(record)
            session.flush()
            return _cost_policy(record)

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id="cost-operations"),
            request_id=request_id,
        )


def _snapshot(
    *,
    generated_at: datetime,
    window_start: datetime,
    current_policy: CostPolicy | None,
    allocations: tuple[_InferenceAllocation, ...],
    training_known_cost: float,
    training_missing: int,
    ledger_allocations: tuple[_LedgerAllocation, ...],
    ledger_batches: tuple[CostLedgerBatchRecord, ...],
) -> CostOperationsSnapshot:
    coverage_cutoff = _day_start(generated_at)
    coverage = _ledger_coverage(
        ledger_batches,
        window_start=window_start,
        cutoff=coverage_cutoff,
    )
    unmetered = tuple(category for category in _LEDGER_CATEGORIES if not coverage[category])
    by_release = _group(
        allocations,
        lambda item: (item.release_id, f"{item.release_id} / {item.model_alias}"),
        ledger_allocations,
        lambda item: (item.release_id, item.release_id) if item.release_id is not None else None,
    )
    by_scenario = _group(
        allocations,
        lambda item: (item.scenario, item.scenario),
        ledger_allocations,
        lambda item: (item.scenario, item.scenario) if item.scenario is not None else None,
    )
    by_business = _group(
        allocations,
        lambda item: (item.business_key, item.business_label),
        ledger_allocations,
        lambda item: (item.business_key, item.business_label),
    )
    daily_items: dict[date, _DailyAccumulator] = {}
    for allocation in allocations:
        daily_items.setdefault(allocation.day, _DailyAccumulator(allocation.day)).add(allocation)
    for allocation in ledger_allocations:
        daily_items.setdefault(allocation.day, _DailyAccumulator(allocation.day)).add_ledger(
            allocation
        )
    daily = tuple(item.freeze() for _, item in sorted(daily_items.items()))

    missing = sum(not item.cost_complete for item in allocations) + training_missing
    if current_policy is None:
        evidence_status = CostEvidenceStatus.UNCONFIGURED
    elif missing or unmetered:
        evidence_status = CostEvidenceStatus.PARTIAL
    else:
        evidence_status = CostEvidenceStatus.COMPLETE
    inference_cost = sum(item.known_cost_usd for item in allocations)
    ledger_cost = sum(item.amount_usd for item in ledger_allocations)
    total_cost = inference_cost + training_known_cost + ledger_cost
    failed_cost = sum(item.known_cost_usd for item in allocations if item.status == "FAILED")
    business_unattributed = sum(item.business_key == "unattributed" for item in allocations)
    totals = CostTotals(
        inference_request_count=len(allocations),
        succeeded_inference_count=sum(item.status == "SUCCEEDED" for item in allocations),
        failed_inference_count=sum(item.status == "FAILED" for item in allocations),
        pending_inference_count=sum(item.status == "PENDING" for item in allocations),
        prompt_tokens=sum(item.prompt_tokens for item in allocations),
        completion_tokens=sum(item.completion_tokens for item in allocations),
        estimated_gpu_hours=_quantity(sum(item.estimated_gpu_hours for item in allocations)),
        inference_known_cost_usd=_money(inference_cost),
        training_known_cost_usd=_money(training_known_cost),
        ledger_entry_count=len(ledger_allocations),
        ledger_known_cost_usd=_money(ledger_cost),
        total_known_cost_usd=_money(total_cost),
        failed_inference_known_cost_usd=_money(failed_cost),
        missing_cost_record_count=missing,
        business_unattributed_request_count=business_unattributed,
    )
    budget = _budget(current_policy, evidence_status, total_cost, by_scenario)
    anomalies: list[str] = []
    if budget.status is BudgetStatus.EXCEEDED:
        anomalies.append("monthly_budget_exceeded")
    elif budget.status is BudgetStatus.WARNING:
        anomalies.append("monthly_budget_warning")
    if evidence_status is CostEvidenceStatus.PARTIAL:
        anomalies.append("cost_evidence_partial")
    if inference_cost > 0 and failed_cost / inference_cost >= 0.20:
        anomalies.append("failed_inference_cost_high")
    if business_unattributed:
        anomalies.append("business_attribution_incomplete")
    return CostOperationsSnapshot(
        policy_version=CostOperationsService.POLICY_VERSION,
        generated_at=generated_at,
        window_start=window_start,
        window_end=generated_at,
        ledger_coverage_cutoff=coverage_cutoff,
        evidence_status=evidence_status,
        current_policy=current_policy,
        totals=totals,
        budget=budget,
        anomalies=tuple(anomalies),
        by_release=by_release,
        by_scenario=by_scenario,
        by_business_task=by_business,
        daily=daily,
        ledger_by_category=_ledger_category_costs(ledger_allocations, coverage),
        ledger_batches=tuple(
            _ledger_batch_summary(item)
            for item in sorted(
                ledger_batches,
                key=lambda value: (_as_utc(value.coverage_start), value.batch_id),
                reverse=True,
            )
        ),
        unmetered_categories=unmetered,
    )


def _allocate_inference(
    record: ModelInferenceRecord,
    policies: tuple[CostPolicyRecord, ...],
    gpu_count: float | None,
    diagnosis_incidents: dict[str, str],
    incident_work_orders: dict[str, list[str]],
) -> _InferenceAllocation:
    created_at = _as_utc(record.created_at)
    policy = next(
        (item for item in reversed(policies) if _as_utc(item.effective_at) <= created_at),
        None,
    )
    completed_at = _as_utc(record.completed_at) if record.completed_at is not None else None
    gpu_consumed = (
        completed_at is not None
        and record.status != "PENDING"
        and record.failure_reason != "model_input_guardrail_blocked"
    )
    duration_hours = (
        max(0.0, (completed_at - created_at).total_seconds()) / 3600.0
        if gpu_consumed and completed_at is not None
        else 0.0
    )
    estimated_gpu_hours = duration_hours * gpu_count if gpu_count is not None else 0.0
    cost_complete = record.status == "PENDING" or (
        policy is not None and (not gpu_consumed or gpu_count is not None)
    )
    known_cost = 0.0
    if policy is not None:
        known_cost += (
            record.usage_prompt_tokens * policy.prompt_tokens_per_million_usd
            + record.usage_completion_tokens * policy.completion_tokens_per_million_usd
        ) / 1_000_000
        if gpu_count is not None:
            known_cost += estimated_gpu_hours * policy.gpu_hourly_cost_usd
    business_key, business_label = _business_reference(
        record,
        diagnosis_incidents,
        incident_work_orders,
    )
    return _InferenceAllocation(
        release_id=record.resolved_release_id,
        model_alias=record.model_alias,
        scenario=record.request_class,
        business_key=business_key,
        business_label=business_label,
        day=created_at.date(),
        status=record.status,
        prompt_tokens=record.usage_prompt_tokens,
        completion_tokens=record.usage_completion_tokens,
        estimated_gpu_hours=estimated_gpu_hours,
        known_cost_usd=known_cost,
        cost_complete=cost_complete,
    )


def _business_reference(
    record: ModelInferenceRecord,
    diagnosis_incidents: dict[str, str],
    incident_work_orders: dict[str, list[str]],
) -> tuple[str, str]:
    if record.diagnosis_run_id is not None:
        incident_id = diagnosis_incidents.get(record.diagnosis_run_id)
        work_orders = incident_work_orders.get(incident_id, []) if incident_id else []
        if len(work_orders) == 1:
            return f"work-order:{work_orders[0]}", f"工单 {work_orders[0]}"
        if incident_id is not None:
            return f"incident:{incident_id}", f"故障 {incident_id}"
        return (
            f"diagnosis:{record.diagnosis_run_id}",
            f"诊断 {record.diagnosis_run_id}",
        )
    if record.agent_run_id is not None:
        return f"agent-run:{record.agent_run_id}", f"Agent Run {record.agent_run_id}"
    return "unattributed", "未关联业务任务"


def _group(
    allocations: tuple[_InferenceAllocation, ...],
    key: Callable[[_InferenceAllocation], tuple[str, str]],
    ledger_allocations: tuple[_LedgerAllocation, ...],
    ledger_key: Callable[[_LedgerAllocation], tuple[str, str] | None],
) -> tuple[CostBreakdown, ...]:
    grouped: dict[str, _Accumulator] = {}
    for allocation in allocations:
        item_key, label = key(allocation)
        grouped.setdefault(item_key, _Accumulator(item_key, label)).add(allocation)
    for allocation in ledger_allocations:
        resolved = ledger_key(allocation)
        if resolved is None:
            continue
        item_key, label = resolved
        grouped.setdefault(item_key, _Accumulator(item_key, label)).add_ledger(allocation)
    return tuple(
        item.freeze()
        for item in sorted(
            grouped.values(),
            key=lambda item: (-item.known_cost_usd, item.key),
        )
    )


def _budget(
    policy: CostPolicy | None,
    evidence_status: CostEvidenceStatus,
    known_cost: float,
    by_scenario: tuple[CostBreakdown, ...],
) -> BudgetPosture:
    if policy is None:
        return BudgetPosture(
            status=BudgetStatus.UNCONFIGURED,
            monthly_budget_usd=None,
            known_cost_usd=_money(known_cost),
            remaining_usd=None,
            consumed_ratio=None,
            warning_ratio=None,
            scenarios=(),
        )
    status = _budget_status(
        known_cost,
        policy.monthly_budget_usd,
        policy.warning_ratio,
        evidence_status is not CostEvidenceStatus.COMPLETE,
    )
    scenario_costs = {item.key: item for item in by_scenario}
    scenarios: list[ScenarioBudget] = []
    for scenario, scenario_budget in sorted(policy.scenario_budgets_usd.items()):
        breakdown = scenario_costs.get(scenario)
        scenario_cost = breakdown.known_cost_usd if breakdown is not None else 0.0
        scenarios.append(
            ScenarioBudget(
                scenario=scenario,
                budget_usd=scenario_budget,
                known_cost_usd=_money(scenario_cost),
                remaining_usd=_money(scenario_budget - scenario_cost),
                consumed_ratio=scenario_cost / scenario_budget,
                status=_budget_status(
                    scenario_cost,
                    scenario_budget,
                    policy.warning_ratio,
                    breakdown is not None and not breakdown.cost_complete,
                ),
            )
        )
    return BudgetPosture(
        status=status,
        monthly_budget_usd=policy.monthly_budget_usd,
        known_cost_usd=_money(known_cost),
        remaining_usd=_money(policy.monthly_budget_usd - known_cost),
        consumed_ratio=known_cost / policy.monthly_budget_usd,
        warning_ratio=policy.warning_ratio,
        scenarios=tuple(scenarios),
    )


def _budget_status(
    cost: float,
    budget: float,
    warning_ratio: float,
    incomplete: bool,
) -> BudgetStatus:
    if incomplete:
        return BudgetStatus.EVIDENCE_INCOMPLETE
    if cost > budget:
        return BudgetStatus.EXCEEDED
    if cost / budget >= warning_ratio:
        return BudgetStatus.WARNING
    return BudgetStatus.ON_TRACK


def _training_cost(
    experiments: tuple[TrainingExperimentRecord, ...],
) -> tuple[float, int]:
    total = 0.0
    missing = 0
    for experiment in experiments:
        value = experiment.cost_summary.get("estimated_compute_cost_usd")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            missing += 1
            continue
        amount = float(value)
        if not math.isfinite(amount) or amount < 0:
            missing += 1
            continue
        total += amount
    return total, missing


def canonical_cost_ledger_digest(
    *,
    source_system: str,
    external_batch_id: str,
    coverage_start: datetime,
    coverage_end: datetime,
    covered_categories: tuple[str, ...],
    currency: str,
    evidence_uri: str,
    evidence_sha256: str,
    entries: tuple[CostLedgerEntryInput, ...],
) -> str:
    ordered_entries = sorted(
        entries,
        key=lambda item: (_as_utc(item.occurred_at), item.external_entry_id),
    )
    value = {
        "source_system": source_system,
        "external_batch_id": external_batch_id,
        "coverage_start": _iso_z(coverage_start),
        "coverage_end": _iso_z(coverage_end),
        "covered_categories": sorted(covered_categories),
        "currency": currency,
        "evidence_uri": evidence_uri,
        "evidence_sha256": evidence_sha256,
        "entries": [
            {
                "external_entry_id": item.external_entry_id,
                "category": item.category,
                "occurred_at": _iso_z(item.occurred_at),
                "amount_usd": item.amount_usd,
                "quantity": item.quantity,
                "unit": item.unit,
                "release_id": item.release_id,
                "scenario": item.scenario,
                "diagnosis_run_id": item.diagnosis_run_id,
                "work_order_id": item.work_order_id,
            }
            for item in ordered_entries
        ],
    }
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validate_ledger_batch(
    *,
    source_system: str,
    external_batch_id: str,
    coverage_start: datetime,
    coverage_end: datetime,
    covered_categories: tuple[str, ...],
    currency: str,
    evidence_uri: str,
    evidence_sha256: str,
    batch_digest: str,
    entries: tuple[CostLedgerEntryInput, ...],
    idempotency_key: str,
    now: datetime,
) -> tuple[datetime, datetime, tuple[str, ...]]:
    if not all(
        _LEDGER_IDENTIFIER.fullmatch(value)
        for value in (source_system, external_batch_id, idempotency_key)
    ):
        raise ValueError("cost_ledger_identifier_invalid")
    start = _as_utc(coverage_start)
    end = _as_utc(coverage_end)
    if (
        start >= end
        or end > now
        or _month_start(end.replace(microsecond=0))
        not in {
            _month_start(start),
            _next_month_start(start),
        }
    ):
        raise ValueError("cost_ledger_coverage_invalid")
    if _month_start(end) != _month_start(start) and end != _next_month_start(start):
        raise ValueError("cost_ledger_coverage_invalid")
    categories = tuple(sorted(set(covered_categories)))
    if (
        not categories
        or len(categories) != len(covered_categories)
        or set(categories) - _LEDGER_CATEGORY_SET
    ):
        raise ValueError("cost_ledger_categories_invalid")
    if currency != "USD":
        raise ValueError("cost_ledger_currency_invalid")
    if not _EVIDENCE_URI.fullmatch(evidence_uri) or not _SHA256.fullmatch(evidence_sha256):
        raise ValueError("cost_ledger_evidence_invalid")
    if not _SHA256.fullmatch(batch_digest):
        raise ValueError("cost_ledger_digest_invalid")
    if not entries or len(entries) > 2_000:
        raise ValueError("cost_ledger_entries_invalid")
    entry_ids: set[str] = set()
    for item in entries:
        occurred_at = _as_utc(item.occurred_at)
        if (
            not _LEDGER_IDENTIFIER.fullmatch(item.external_entry_id)
            or item.external_entry_id in entry_ids
            or item.category not in categories
            or not start <= occurred_at < end
            or isinstance(item.amount_usd, bool)
            or not math.isfinite(item.amount_usd)
            or not 0 <= item.amount_usd <= 1_000_000_000
        ):
            raise ValueError("cost_ledger_entry_invalid")
        entry_ids.add(item.external_entry_id)
        if item.quantity is not None and (
            isinstance(item.quantity, bool)
            or not math.isfinite(item.quantity)
            or not 0 <= item.quantity <= 1_000_000_000_000
        ):
            raise ValueError("cost_ledger_entry_invalid")
        if (item.quantity is None) != (item.unit is None) or (
            item.unit is not None and not _LEDGER_UNIT.fullmatch(item.unit)
        ):
            raise ValueError("cost_ledger_entry_invalid")
        if item.scenario is not None and item.scenario not in _SUPPORTED_SCENARIOS:
            raise ValueError("cost_ledger_entry_invalid")
        for reference in (
            item.release_id,
            item.diagnosis_run_id,
            item.work_order_id,
        ):
            if reference is not None and not _LEDGER_IDENTIFIER.fullmatch(reference):
                raise ValueError("cost_ledger_entry_invalid")
    return start, end, categories


def _ledger_import_replay(
    record: CostLedgerBatchRecord,
    batch_digest: str,
) -> CostLedgerImportResult:
    if record.batch_digest != batch_digest:
        raise CostLedgerConflict("cost_ledger_idempotency_conflict")
    return CostLedgerImportResult(_ledger_receipt(record), replayed=True)


def _validate_ledger_references(
    session: Session,
    tenant_id: str,
    entries: tuple[CostLedgerEntryInput, ...],
) -> None:
    reference_sets = (
        (
            {item.release_id for item in entries if item.release_id is not None},
            ModelReleaseRecord.release_id,
            ModelReleaseRecord.tenant_id,
        ),
        (
            {item.diagnosis_run_id for item in entries if item.diagnosis_run_id is not None},
            DiagnosisRunRecord.diagnosis_run_id,
            DiagnosisRunRecord.tenant_id,
        ),
        (
            {item.work_order_id for item in entries if item.work_order_id is not None},
            WorkOrderRecord.work_order_id,
            WorkOrderRecord.tenant_id,
        ),
    )
    for requested, identifier_column, tenant_column in reference_sets:
        if not requested:
            continue
        visible = set(
            session.scalars(
                select(identifier_column).where(
                    tenant_column == tenant_id,
                    identifier_column.in_(requested),
                )
            )
        )
        if visible != requested:
            raise ValueError("cost_ledger_reference_invalid")


def _ledger_receipt(record: CostLedgerBatchRecord) -> CostLedgerBatchReceipt:
    return CostLedgerBatchReceipt(
        batch_id=record.batch_id,
        source_system=record.source_system,
        external_batch_id=record.external_batch_id,
        coverage_start=_as_utc(record.coverage_start),
        coverage_end=_as_utc(record.coverage_end),
        covered_categories=tuple(record.covered_categories_json),
        currency=record.currency,
        evidence_uri=record.evidence_uri,
        evidence_sha256=record.evidence_sha256,
        batch_digest=record.batch_digest,
        entry_count=record.entry_count,
        total_amount_usd=_money(record.total_amount_usd),
        imported_by_subject_id=record.imported_by_subject_id,
        imported_at=_as_utc(record.created_at),
    )


def _ledger_batch_summary(record: CostLedgerBatchRecord) -> CostLedgerBatchSummary:
    receipt = _ledger_receipt(record)
    return CostLedgerBatchSummary(
        batch_id=receipt.batch_id,
        source_system=receipt.source_system,
        external_batch_id=receipt.external_batch_id,
        coverage_start=receipt.coverage_start,
        coverage_end=receipt.coverage_end,
        covered_categories=receipt.covered_categories,
        batch_digest=receipt.batch_digest,
        evidence_uri=receipt.evidence_uri,
        evidence_sha256=receipt.evidence_sha256,
        entry_count=receipt.entry_count,
        total_amount_usd=receipt.total_amount_usd,
        imported_by_subject_id=receipt.imported_by_subject_id,
        imported_at=receipt.imported_at,
    )


def _ledger_allocation(record: CostLedgerEntryRecord) -> _LedgerAllocation:
    if record.work_order_id is not None:
        business_key = f"work-order:{record.work_order_id}"
        business_label = f"工单 {record.work_order_id}"
    elif record.diagnosis_run_id is not None:
        business_key = f"diagnosis:{record.diagnosis_run_id}"
        business_label = f"诊断 {record.diagnosis_run_id}"
    else:
        business_key = "ledger-unattributed"
        business_label = "未关联业务任务的 Ledger"
    return _LedgerAllocation(
        category=record.category,
        day=_as_utc(record.occurred_at).date(),
        amount_usd=record.amount_usd,
        release_id=record.release_id,
        scenario=record.scenario,
        business_key=business_key,
        business_label=business_label,
    )


def _ledger_coverage(
    batches: tuple[CostLedgerBatchRecord, ...],
    *,
    window_start: datetime,
    cutoff: datetime,
) -> dict[str, bool]:
    result = {category: False for category in _LEDGER_CATEGORIES}
    if cutoff <= window_start:
        return result
    for category in _LEDGER_CATEGORIES:
        intervals = sorted(
            (
                max(_as_utc(item.coverage_start), window_start),
                min(_as_utc(item.coverage_end), cutoff),
            )
            for item in batches
            if category in item.covered_categories_json
            and _as_utc(item.coverage_start) < cutoff
            and _as_utc(item.coverage_end) > window_start
        )
        cursor = window_start
        for start, end in intervals:
            if end <= cursor:
                continue
            if start > cursor:
                break
            cursor = max(cursor, end)
            if cursor >= cutoff:
                result[category] = True
                break
    return result


def _ledger_category_costs(
    allocations: tuple[_LedgerAllocation, ...],
    coverage: dict[str, bool],
) -> tuple[LedgerCategoryCost, ...]:
    return tuple(
        LedgerCategoryCost(
            category=category,
            entry_count=sum(item.category == category for item in allocations),
            known_cost_usd=_money(
                sum(item.amount_usd for item in allocations if item.category == category)
            ),
            coverage_complete=coverage[category],
        )
        for category in _LEDGER_CATEGORIES
    )


def _gpu_count(desired_spec: dict[str, Any]) -> float | None:
    profile = desired_spec.get("hardware_profile")
    value = profile.get("gpu_count") if isinstance(profile, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    count = float(value)
    return count if math.isfinite(count) and count > 0 else None


def _cost_policy(record: CostPolicyRecord) -> CostPolicy:
    return CostPolicy(
        policy_id=record.policy_id,
        version=record.version,
        effective_at=_as_utc(record.effective_at),
        prompt_tokens_per_million_usd=record.prompt_tokens_per_million_usd,
        completion_tokens_per_million_usd=record.completion_tokens_per_million_usd,
        gpu_hourly_cost_usd=record.gpu_hourly_cost_usd,
        monthly_budget_usd=record.monthly_budget_usd,
        scenario_budgets_usd=dict(record.scenario_budgets_json),
        warning_ratio=record.warning_ratio,
        reason=record.reason,
        updated_by_subject_id=record.updated_by_subject_id,
    )


def _validate_policy(
    prompt_rate: float,
    completion_rate: float,
    gpu_rate: float,
    monthly_budget: float,
    scenario_budgets: dict[str, float],
    warning_ratio: float,
    reason: str,
    expected_version: int,
) -> None:
    for value, upper in (
        (prompt_rate, 10_000.0),
        (completion_rate, 10_000.0),
        (gpu_rate, 10_000.0),
        (monthly_budget, 1_000_000_000.0),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0 or value > upper:
            raise ValueError("cost policy rate or budget is invalid")
    if monthly_budget <= 0 or not 0.1 <= warning_ratio <= 1.0:
        raise ValueError("cost policy budget threshold is invalid")
    if not 8 <= len(reason.strip()) <= 1_000 or expected_version < 0:
        raise ValueError("cost policy reason or version is invalid")
    if set(scenario_budgets) - _SUPPORTED_SCENARIOS:
        raise ValueError("cost policy scenario is invalid")
    for scenario, value in scenario_budgets.items():
        if (
            not _SCENARIO.fullmatch(scenario)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            or value > monthly_budget
        ):
            raise ValueError("cost policy scenario budget is invalid")


def _month_start(value: datetime) -> datetime:
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_month_start(value: datetime) -> datetime:
    current = _month_start(value)
    if current.month == 12:
        return current.replace(year=current.year + 1, month=1)
    return current.replace(month=current.month + 1)


def _day_start(value: datetime) -> datetime:
    return _as_utc(value).replace(hour=0, minute=0, second=0, microsecond=0)


def _iso_z(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _money(value: float) -> float:
    return round(value, 8)


def _quantity(value: float) -> float:
    return round(value, 10)
