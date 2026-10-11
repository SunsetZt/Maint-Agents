"""Source-bound, review-only multi-agent maintenance planning lifecycle."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.maintenance_planning.activation import (
    MaintenancePlanningActivationConflict,
    MaintenancePlanningActivationNotVisible,
    MaintenancePlanningActivationService,
    require_bound_active_activation,
)
from industrial_ops_agent.maintenance_planning.review_isolation import (
    check_disclosure,
    lock_subject,
    record_disclosure,
)
from industrial_ops_agent.model_gateway.context_manifest import ContextReference
from industrial_ops_agent.model_gateway.service import (
    GatewayRequest,
    GatewayResponse,
    ModelGatewayError,
    ProductionModelResolver,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetRecord,
    DiagnosisRunRecord,
    IncidentRecord,
    MaintenancePlanningContributionRecord,
    MaintenancePlanningCouncilRecord,
    PromptBundleRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.prompting import MAINTENANCE_PLANNING_PROMPT_BUNDLE_ID
from industrial_ops_agent.prompting.registry import PromptBundleNotDeployed, default_prompt_registry

MAINTENANCE_PLANNING_PROFILE = "multi-agent-maintenance-planning/v1"
MAINTENANCE_PLANNING_SCHEMA_VERSION = "maintenance_planning_council_v1"
SPECIALIST_ROLES = ("SAFETY", "PARTS", "DISPATCH")
ALL_AGENT_ROLES = (*SPECIALIST_ROLES, "COORDINATOR")
TERMINAL_STATUSES = frozenset({"ACCEPTED", "REJECTED"})
RETRYABLE_FAILURES = frozenset(
    {
        "maintenance_planning_dispatch_unavailable",
        "maintenance_planning_dependency_unavailable",
        "maintenance_planning_inference_retryable",
    }
)
_OUTPUT_FIELDS = frozenset(
    {
        "agent_role",
        "summary",
        "recommendations",
        "constraints",
        "missing_facts",
        "safety_hold_points",
        "required_parts",
        "dispatch_constraints",
        "source_roles",
        "readiness",
    }
)


class MaintenancePlanningNotVisible(Exception):
    pass


class MaintenancePlanningConflict(RuntimeError):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


class MaintenancePlanningDispatchUnavailable(RuntimeError):
    """Temporal could not durably accept a council command."""


@dataclass(frozen=True, slots=True)
class MaintenancePlanningRuntimeBinding:
    planning_profile: str
    model_alias: str
    model_release_id: str
    model_manifest_hash: str
    prompt_bundle_id: str
    prompt_bundle_hash: str
    response_schema_version: str
    activation_id: str | None = None
    activation_policy_hash: str | None = None
    activation_target_environment: str | None = None
    activation_evaluation_id: str | None = None
    activation_evaluation_report_hash: str | None = None


class MaintenancePlanningBindingResolver(Protocol):
    def resolve(self, identity: IdentityContext) -> MaintenancePlanningRuntimeBinding: ...


@dataclass(frozen=True, slots=True)
class MaintenancePlanningActivityInput:
    council_id: str
    workflow_id: str
    identity: IdentityContext
    request_id: str


class MaintenancePlanningDispatcher(Protocol):
    async def dispatch(self, command: MaintenancePlanningActivityInput) -> None: ...


class MaintenancePlanningModelGateway(Protocol):
    async def complete(
        self,
        context: TenantContext,
        request: GatewayRequest,
    ) -> GatewayResponse: ...


@dataclass(frozen=True, slots=True)
class MaintenancePlanningContributionView:
    contribution_id: str
    attempt_number: int
    agent_role: str
    inference_request_id: str
    input_digest: str
    output: dict[str, Any]
    output_digest: str
    model_release_id: str
    model_manifest_hash: str
    context_hash: str
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class MaintenancePlanningView:
    council_id: str
    incident_id: str
    diagnosis_run_id: str
    asset_id: str
    diagnosis_version: int
    diagnosis_report_digest: str
    diagnosis_manifest_digest: str
    workflow_id: str
    model_alias: str
    model_release_id: str
    model_manifest_hash: str
    prompt_bundle_id: str
    prompt_bundle_hash: str
    response_schema_version: str
    activation_id: str | None
    activation_policy_hash: str | None
    activation_target_environment: str | None
    status: str
    stage: str
    attempt_count: int
    plan: dict[str, Any] | None
    result_digest: str | None
    coordinator_context_hash: str | None
    failure_code: str | None
    requested_by_subject_id: str
    reviewed_by_subject_id: str | None
    review_decision: str | None
    review_reason: str | None
    started_at: datetime | None
    completed_at: datetime | None
    reviewed_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    contributions: tuple[MaintenancePlanningContributionView, ...]
    dispatch_required: bool = False

    @property
    def legal_actions(self) -> tuple[str, ...]:
        if self.status == "FAILED" and self.failure_code in RETRYABLE_FAILURES:
            return ("RETRY",)
        if self.status == "REVIEW_PENDING":
            return ("REVIEW_ACCEPT", "REVIEW_REJECT")
        return ()


@dataclass(frozen=True, slots=True)
class _PlanningSource:
    incident: dict[str, Any]
    diagnosis_report: dict[str, Any]
    diagnosis_manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _InferenceResult:
    role: str
    inference_request_id: str
    input_digest: str
    output: dict[str, Any]
    response: GatewayResponse


class _PlanningFailure(RuntimeError):
    def __init__(self, stage: str, failure_code: str) -> None:
        self.stage = stage
        self.failure_code = failure_code
        super().__init__(failure_code)


class ProductionMaintenancePlanningBindingResolver:
    """Freeze one approved image Prompt and exact production model release."""

    def __init__(
        self,
        database: Database,
        model_resolver: ProductionModelResolver,
        *,
        model_alias: str,
    ) -> None:
        self._database = database
        self._model_resolver = model_resolver
        self._model_alias = model_alias

    def resolve(self, identity: IdentityContext) -> MaintenancePlanningRuntimeBinding:
        try:
            definition = default_prompt_registry().get(MAINTENANCE_PLANNING_PROMPT_BUNDLE_ID)
        except PromptBundleNotDeployed as exc:
            raise MaintenancePlanningConflict("maintenance_planning_prompt_not_deployed") from exc
        if (
            definition.output_schema_name != MAINTENANCE_PLANNING_SCHEMA_VERSION
            or self._model_alias not in definition.compatible_model_aliases
        ):
            raise MaintenancePlanningConflict("maintenance_planning_prompt_binding_invalid")
        try:
            model = self._model_resolver.resolve(identity.tenant_context, self._model_alias)
        except ModelGatewayError as exc:
            raise MaintenancePlanningConflict("maintenance_planning_model_unavailable") from exc
        with self._database.transaction(identity.tenant_context) as session:
            governed = session.scalar(
                select(PromptBundleRecord).where(
                    PromptBundleRecord.tenant_id == identity.tenant_id,
                    PromptBundleRecord.prompt_bundle_id == definition.prompt_bundle_id,
                )
            )
            if governed is None or governed.status != "APPROVED":
                raise MaintenancePlanningConflict("maintenance_planning_prompt_not_approved")
            if (
                governed.content_hash != definition.content_hash
                or governed.output_schema_name != definition.output_schema_name
                or self._model_alias not in governed.compatible_model_aliases_json
            ):
                raise MaintenancePlanningConflict("maintenance_planning_prompt_hash_mismatch")
        return MaintenancePlanningRuntimeBinding(
            planning_profile=MAINTENANCE_PLANNING_PROFILE,
            model_alias=model.alias,
            model_release_id=model.release_id,
            model_manifest_hash=model.manifest_hash,
            prompt_bundle_id=definition.prompt_bundle_id,
            prompt_bundle_hash=definition.content_hash,
            response_schema_version=definition.output_schema_name,
        )


class GovernedMaintenancePlanningBindingResolver:
    """Require an approved tenant rollout before resolving model and Prompt bindings."""

    def __init__(
        self,
        delegate: MaintenancePlanningBindingResolver,
        activations: MaintenancePlanningActivationService,
        *,
        target_environment: str,
    ) -> None:
        self._delegate = delegate
        self._activations = activations
        self._target_environment = target_environment

    def resolve(self, identity: IdentityContext) -> MaintenancePlanningRuntimeBinding:
        activation = self._activations.resolve_active(
            identity.tenant_id,
            self._target_environment,
        )
        binding = self._delegate.resolve(identity)
        return MaintenancePlanningRuntimeBinding(
            planning_profile=binding.planning_profile,
            model_alias=binding.model_alias,
            model_release_id=binding.model_release_id,
            model_manifest_hash=binding.model_manifest_hash,
            prompt_bundle_id=binding.prompt_bundle_id,
            prompt_bundle_hash=binding.prompt_bundle_hash,
            response_schema_version=binding.response_schema_version,
            activation_id=activation.activation_id,
            activation_policy_hash=activation.activation_policy_hash,
            activation_target_environment=activation.target_environment,
            activation_evaluation_id=activation.evaluation_id,
            activation_evaluation_report_hash=activation.evaluation_report_hash,
        )


class MaintenancePlanningActivity:
    """Run three independent specialists and one synthesis-only coordinator."""

    def __init__(
        self,
        database: Database,
        model_gateway: MaintenancePlanningModelGateway,
        *,
        timeout_seconds: float,
    ) -> None:
        self._database = database
        self._model_gateway = model_gateway
        self._timeout_seconds = timeout_seconds

    async def execute(self, command: MaintenancePlanningActivityInput) -> MaintenancePlanningView:
        current = self._begin(command)
        if current.status != "RUNNING":
            return current
        try:
            source = self._load_source(command, current)
            specialist_results = await asyncio.gather(
                *(
                    self._complete_role(command, current, source, role, ())
                    for role in SPECIALIST_ROLES
                )
            )
            coordinator = await self._complete_role(
                command,
                current,
                source,
                "COORDINATOR",
                tuple(item.output for item in specialist_results),
            )
            return self._finish(
                command,
                current,
                (*specialist_results, coordinator),
            )
        except _PlanningFailure as exc:
            return self._fail(command, stage=exc.stage, failure_code=exc.failure_code)
        except (MaintenancePlanningConflict, MaintenancePlanningNotVisible):
            return self._fail(
                command,
                stage="SOURCE_VALIDATION_FAILED",
                failure_code="maintenance_planning_source_binding_changed",
            )
        except ModelGatewayError as exc:
            retryable = exc.reason in {
                "model_quota_unavailable",
                "model_request_rate_exceeded",
                "model_daily_token_quota_exceeded",
                "production_model_unavailable",
            }
            return self._fail(
                command,
                stage="MODEL_INFERENCE_FAILED",
                failure_code=(
                    "maintenance_planning_inference_retryable"
                    if retryable
                    else "maintenance_planning_inference_rejected"
                ),
            )
        except Exception:
            return self._fail(
                command,
                stage="DEPENDENCY_FAILED",
                failure_code="maintenance_planning_dependency_unavailable",
            )

    def _begin(self, command: MaintenancePlanningActivityInput) -> MaintenancePlanningView:
        now = datetime.now(UTC)
        with self._database.transaction(command.identity.tenant_context) as session:
            record = _council_or_hidden(
                session, command.identity.tenant_id, command.council_id, lock=True
            )
            if record.workflow_id != command.workflow_id:
                raise MaintenancePlanningConflict("maintenance_planning_workflow_binding_changed")
            if record.status in {"REVIEW_PENDING", "ACCEPTED", "REJECTED", "FAILED"}:
                return _view(session, record)
            if record.status == "QUEUED":
                record.status = "RUNNING"
                record.stage = "SPECIALIST_INFERENCE"
                record.attempt_count += 1
                record.started_at = now
                record.completed_at = None
                record.failure_code = None
                record.plan_json = None
                record.result_digest = None
                record.coordinator_context_hash = None
                record.version += 1
                session.flush()
            elif record.status != "RUNNING":
                raise MaintenancePlanningConflict(
                    "maintenance_planning_activity_state_invalid", record.version
                )
            return _view(session, record)

    def _load_source(
        self,
        command: MaintenancePlanningActivityInput,
        current: MaintenancePlanningView,
    ) -> _PlanningSource:
        with self._database.transaction(command.identity.tenant_context) as session:
            record = _council_or_hidden(session, command.identity.tenant_id, command.council_id)
            if record.status != "RUNNING" or record.workflow_id != command.workflow_id:
                raise MaintenancePlanningConflict("maintenance_planning_activity_state_changed")
            return _validated_source(session, record, current)

    async def _complete_role(
        self,
        command: MaintenancePlanningActivityInput,
        council: MaintenancePlanningView,
        source: _PlanningSource,
        role: str,
        specialist_contributions: tuple[dict[str, Any], ...],
    ) -> _InferenceResult:
        request, input_digest = _gateway_request(
            council,
            source,
            role,
            specialist_contributions,
            self._timeout_seconds,
        )
        response = await self._model_gateway.complete(
            command.identity.tenant_context,
            request,
        )
        if (
            response.inference_request_id != request.inference_request_id
            or response.resolved_release_id != council.model_release_id
            or response.manifest_hash != council.model_manifest_hash
        ):
            raise _PlanningFailure(
                "MODEL_BINDING_FAILED",
                "maintenance_planning_model_binding_changed",
            )
        return _InferenceResult(
            role=role,
            inference_request_id=request.inference_request_id,
            input_digest=input_digest,
            output=_validated_output(response.content, role),
            response=response,
        )

    def _finish(
        self,
        command: MaintenancePlanningActivityInput,
        current: MaintenancePlanningView,
        results: tuple[_InferenceResult, ...],
    ) -> MaintenancePlanningView:
        now = datetime.now(UTC)
        if tuple(item.role for item in results) != ALL_AGENT_ROLES:
            raise _PlanningFailure(
                "COORDINATION_FAILED", "maintenance_planning_contributions_incomplete"
            )
        coordinator = results[-1]
        with self._database.transaction(command.identity.tenant_context) as session:
            record = _council_or_hidden(
                session, command.identity.tenant_id, command.council_id, lock=True
            )
            if record.status == "REVIEW_PENDING":
                return _view(session, record)
            if record.status != "RUNNING" or record.workflow_id != command.workflow_id:
                raise MaintenancePlanningConflict(
                    "maintenance_planning_activity_state_changed", record.version
                )
            _validated_source(session, record, current)
            for result in results:
                session.add(
                    MaintenancePlanningContributionRecord(
                        contribution_id=f"planning-contribution-{uuid4().hex}",
                        tenant_id=record.tenant_id,
                        council_id=record.council_id,
                        attempt_number=record.attempt_count,
                        agent_role=result.role,
                        inference_request_id=result.inference_request_id,
                        input_digest=result.input_digest,
                        output_json=result.output,
                        output_digest=_digest(result.output),
                        model_release_id=result.response.resolved_release_id,
                        model_manifest_hash=result.response.manifest_hash,
                        context_hash=result.response.context_hash,
                        completed_at=now,
                    )
                )
            record.plan_json = coordinator.output
            record.result_digest = _digest(coordinator.output)
            record.coordinator_context_hash = coordinator.response.context_hash
            record.status = "REVIEW_PENDING"
            record.stage = "REVIEW_PENDING"
            record.failure_code = None
            record.completed_at = now
            record.version += 1
            session.flush()
            return _view(session, record)

    def _fail(
        self,
        command: MaintenancePlanningActivityInput,
        *,
        stage: str,
        failure_code: str,
    ) -> MaintenancePlanningView:
        with self._database.transaction(command.identity.tenant_context) as session:
            record = _council_or_hidden(
                session, command.identity.tenant_id, command.council_id, lock=True
            )
            if record.status in {"REVIEW_PENDING", "ACCEPTED", "REJECTED", "FAILED"}:
                return _view(session, record)
            record.status = "FAILED"
            record.stage = stage[:64]
            record.failure_code = failure_code[:128]
            record.plan_json = None
            record.result_digest = None
            record.coordinator_context_hash = None
            record.completed_at = datetime.now(UTC)
            record.version += 1
            session.flush()
            return _view(session, record)


class MaintenancePlanningService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def request(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        diagnosis_run_id: str,
        idempotency_key: str,
        binding: MaintenancePlanningRuntimeBinding,
        request_id: str,
    ) -> MaintenancePlanningView:
        if not 8 <= len(idempotency_key) <= 255:
            raise MaintenancePlanningConflict("maintenance_planning_idempotency_invalid")
        binding_document = _binding_document(binding)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            incident, diagnosis, _asset = _request_source(
                session, identity.tenant_id, incident_id, diagnosis_run_id
            )
            self._require(
                identity,
                Action.REQUEST_MAINTENANCE_COUNCIL,
                incident_id,
                request_id,
                asset_id=incident.asset_id,
            )
            report = dict(diagnosis.report or {})
            manifest = dict(diagnosis.manifest)
            request_hash = _digest(
                {
                    "incident_id": incident_id,
                    "diagnosis_run_id": diagnosis_run_id,
                    "diagnosis_version": diagnosis.version,
                    "report_digest": _digest(report),
                    "manifest_digest": _digest(manifest),
                    "binding": binding_document,
                }
            )
            replay = session.scalar(
                select(MaintenancePlanningCouncilRecord).where(
                    MaintenancePlanningCouncilRecord.tenant_id == identity.tenant_id,
                    MaintenancePlanningCouncilRecord.requested_by_subject_id == identity.subject_id,
                    MaintenancePlanningCouncilRecord.idempotency_key == idempotency_key,
                )
            )
            if replay is not None:
                if replay.request_hash != request_hash:
                    raise MaintenancePlanningConflict(
                        "maintenance_planning_idempotency_conflict", replay.version
                    )
                record_disclosure(
                    session,
                    identity,
                    (replay.diagnosis_run_id,),
                    resource_kind="maintenance-council",
                    resource_id=replay.council_id,
                    request_id=request_id,
                )
                return _view(session, replay)
            existing = session.scalar(
                select(MaintenancePlanningCouncilRecord).where(
                    MaintenancePlanningCouncilRecord.tenant_id == identity.tenant_id,
                    MaintenancePlanningCouncilRecord.diagnosis_run_id == diagnosis_run_id,
                )
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise MaintenancePlanningConflict(
                        "maintenance_planning_diagnosis_already_bound", existing.version
                    )
                record_disclosure(
                    session,
                    identity,
                    (existing.diagnosis_run_id,),
                    resource_kind="maintenance-council",
                    resource_id=existing.council_id,
                    request_id=request_id,
                )
                return _view(session, existing)
            council_id = f"maintenance-council-{uuid4().hex}"
            workflow_id = f"maintenance-council-workflow-{uuid4().hex}"
            record = MaintenancePlanningCouncilRecord(
                council_id=council_id,
                tenant_id=identity.tenant_id,
                incident_id=incident_id,
                diagnosis_run_id=diagnosis_run_id,
                asset_id=incident.asset_id,
                diagnosis_version=diagnosis.version,
                diagnosis_report_digest=_digest(report),
                diagnosis_manifest_digest=_digest(manifest),
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                workflow_id=workflow_id,
                model_alias=binding.model_alias,
                model_release_id=binding.model_release_id,
                model_manifest_hash=binding.model_manifest_hash,
                prompt_bundle_id=binding.prompt_bundle_id,
                prompt_bundle_hash=binding.prompt_bundle_hash,
                response_schema_version=binding.response_schema_version,
                activation_id=binding.activation_id,
                activation_policy_hash=binding.activation_policy_hash,
                activation_target_environment=binding.activation_target_environment,
                status="QUEUED",
                stage="QUEUED",
                attempt_count=0,
                plan_json=None,
                result_digest=None,
                coordinator_context_hash=None,
                failure_code=None,
                requested_by_subject_id=identity.subject_id,
                reviewed_by_subject_id=None,
                review_decision=None,
                review_reason=None,
                started_at=None,
                completed_at=None,
                reviewed_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            record_disclosure(
                session,
                identity,
                (record.diagnosis_run_id,),
                resource_kind="maintenance-council",
                resource_id=record.council_id,
                request_id=request_id,
            )
            return _view(session, record, dispatch_required=True)

    def list(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        incident_id: str | None = None,
    ) -> tuple[MaintenancePlanningView, ...]:
        self._require(
            identity,
            Action.READ_MAINTENANCE_COUNCIL,
            incident_id or "maintenance-councils",
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            statement = select(MaintenancePlanningCouncilRecord).where(
                MaintenancePlanningCouncilRecord.tenant_id == identity.tenant_id,
                MaintenancePlanningCouncilRecord.asset_id.in_(identity.asset_ids),
            )
            if incident_id is not None:
                statement = statement.where(
                    MaintenancePlanningCouncilRecord.incident_id == incident_id
                )
            records = session.scalars(
                statement.order_by(
                    MaintenancePlanningCouncilRecord.created_at.desc(),
                    MaintenancePlanningCouncilRecord.council_id,
                )
            )
            records = tuple(records)
            record_disclosure(
                session,
                identity,
                (item.diagnosis_run_id for item in records),
                resource_kind="maintenance-council-list",
                resource_id=incident_id or "maintenance-councils",
                request_id=request_id,
            )
            return tuple(_view(session, item) for item in records)

    def get(
        self,
        identity: IdentityContext,
        council_id: str,
        *,
        request_id: str,
    ) -> MaintenancePlanningView:
        with self._database.transaction(identity.tenant_context) as session:
            record = _council_or_hidden(session, identity.tenant_id, council_id)
            self._require(
                identity,
                Action.READ_MAINTENANCE_COUNCIL,
                council_id,
                request_id,
                asset_id=record.asset_id,
            )
            record_disclosure(
                session,
                identity,
                (record.diagnosis_run_id,),
                resource_kind="maintenance-council",
                resource_id=record.council_id,
                request_id=request_id,
            )
            return _view(session, record)

    def retry(
        self,
        identity: IdentityContext,
        council_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> MaintenancePlanningView:
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _council_or_hidden(session, identity.tenant_id, council_id, lock=True)
            self._require(
                identity,
                Action.REQUEST_MAINTENANCE_COUNCIL,
                council_id,
                request_id,
                asset_id=record.asset_id,
            )
            check_disclosure(session, identity, (record.diagnosis_run_id,))
            if (
                record.version != expected_version
                or record.status != "FAILED"
                or record.failure_code not in RETRYABLE_FAILURES
            ):
                raise MaintenancePlanningConflict(
                    "maintenance_planning_retry_state_invalid", record.version
                )
            current = _view(session, record)
            _validated_source(session, record, current)
            record.status = "QUEUED"
            record.stage = "QUEUED"
            record.failure_code = None
            record.completed_at = None
            record.version += 1
            session.flush()
            record_disclosure(
                session,
                identity,
                (record.diagnosis_run_id,),
                resource_kind="maintenance-council",
                resource_id=record.council_id,
                request_id=request_id,
            )
            return _view(session, record, dispatch_required=True)

    def mark_dispatch_failed(
        self,
        identity: IdentityContext,
        council_id: str,
        *,
        workflow_id: str,
    ) -> MaintenancePlanningView:
        with self._database.transaction(identity.tenant_context) as session:
            record = _council_or_hidden(session, identity.tenant_id, council_id, lock=True)
            if record.workflow_id != workflow_id:
                raise MaintenancePlanningConflict("maintenance_planning_workflow_binding_changed")
            if record.status == "FAILED":
                return _view(session, record)
            if record.status != "QUEUED":
                raise MaintenancePlanningConflict(
                    "maintenance_planning_dispatch_state_invalid", record.version
                )
            record.status = "FAILED"
            record.stage = "DISPATCH_FAILED"
            record.failure_code = "maintenance_planning_dispatch_unavailable"
            record.completed_at = datetime.now(UTC)
            record.version += 1
            session.flush()
            return _view(session, record)

    def review(
        self,
        identity: IdentityContext,
        council_id: str,
        *,
        decision: str,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> MaintenancePlanningView:
        normalized_decision = decision.strip().upper()
        normalized_reason = reason.strip()
        if (
            normalized_decision not in {"ACCEPT", "REJECT"}
            or not 8 <= len(normalized_reason) <= 1_000
        ):
            raise MaintenancePlanningConflict("maintenance_planning_review_input_invalid")
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _council_or_hidden(session, identity.tenant_id, council_id, lock=True)
            self._require(
                identity,
                Action.REVIEW_MAINTENANCE_COUNCIL,
                council_id,
                request_id,
                asset_id=record.asset_id,
            )
            check_disclosure(session, identity, (record.diagnosis_run_id,))
            if record.requested_by_subject_id == identity.subject_id:
                raise MaintenancePlanningConflict(
                    "maintenance_planning_review_separation_required", record.version
                )
            if record.version != expected_version:
                raise MaintenancePlanningConflict(
                    "maintenance_planning_review_state_changed", record.version
                )
            if record.status in TERMINAL_STATUSES:
                if record.review_decision == normalized_decision:
                    record_disclosure(
                        session,
                        identity,
                        (record.diagnosis_run_id,),
                        resource_kind="maintenance-council",
                        resource_id=record.council_id,
                        request_id=request_id,
                    )
                    return _view(session, record)
                raise MaintenancePlanningConflict(
                    "maintenance_planning_review_already_completed", record.version
                )
            if record.status != "REVIEW_PENDING":
                raise MaintenancePlanningConflict(
                    "maintenance_planning_review_state_changed", record.version
                )
            current = _view(session, record)
            _validated_source(session, record, current)
            if (
                record.plan_json is None
                or record.result_digest is None
                or _digest(record.plan_json) != record.result_digest
            ):
                raise MaintenancePlanningConflict(
                    "maintenance_planning_result_binding_changed", record.version
                )
            _validated_output(record.plan_json, "COORDINATOR")
            record.status = "ACCEPTED" if normalized_decision == "ACCEPT" else "REJECTED"
            record.stage = record.status
            record.reviewed_by_subject_id = identity.subject_id
            record.review_decision = normalized_decision
            record.review_reason = normalized_reason
            record.reviewed_at = datetime.now(UTC)
            record.version += 1
            session.flush()
            record_disclosure(
                session,
                identity,
                (record.diagnosis_run_id,),
                resource_kind="maintenance-council",
                resource_id=record.council_id,
                request_id=request_id,
            )
            return _view(session, record)

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
        *,
        asset_id: str | None = None,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id=resource_id,
                asset_id=asset_id,
            ),
            request_id=request_id,
        )


def _binding_document(binding: MaintenancePlanningRuntimeBinding) -> dict[str, str]:
    document = {
        "planning_profile": str(binding.planning_profile),
        "model_alias": str(binding.model_alias),
        "model_release_id": str(binding.model_release_id),
        "model_manifest_hash": str(binding.model_manifest_hash),
        "prompt_bundle_id": str(binding.prompt_bundle_id),
        "prompt_bundle_hash": str(binding.prompt_bundle_hash),
        "response_schema_version": str(binding.response_schema_version),
    }
    activation_values = (
        binding.activation_id,
        binding.activation_policy_hash,
        binding.activation_target_environment,
        binding.activation_evaluation_id,
        binding.activation_evaluation_report_hash,
    )
    if any(value is not None for value in activation_values):
        if not all(isinstance(value, str) and value for value in activation_values):
            raise MaintenancePlanningConflict("maintenance_planning_activation_binding_invalid")
        document.update(
            {
                "activation_id": str(binding.activation_id),
                "activation_policy_hash": str(binding.activation_policy_hash),
                "activation_target_environment": str(binding.activation_target_environment),
                "activation_evaluation_id": str(binding.activation_evaluation_id),
                "activation_evaluation_report_hash": str(binding.activation_evaluation_report_hash),
            }
        )
    if (
        document["planning_profile"] != MAINTENANCE_PLANNING_PROFILE
        or document["response_schema_version"] != MAINTENANCE_PLANNING_SCHEMA_VERSION
        or any(not value or len(value) > 255 for value in document.values())
    ):
        raise MaintenancePlanningConflict("maintenance_planning_runtime_binding_invalid")
    return document


def _request_source(
    session: Session,
    tenant_id: str,
    incident_id: str,
    diagnosis_run_id: str,
) -> tuple[IncidentRecord, DiagnosisRunRecord, AssetRecord]:
    incident = session.scalar(
        select(IncidentRecord).where(
            IncidentRecord.tenant_id == tenant_id,
            IncidentRecord.incident_id == incident_id,
        )
    )
    diagnosis = session.scalar(
        select(DiagnosisRunRecord).where(
            DiagnosisRunRecord.tenant_id == tenant_id,
            DiagnosisRunRecord.diagnosis_run_id == diagnosis_run_id,
        )
    )
    if incident is None or diagnosis is None or diagnosis.incident_id != incident_id:
        raise MaintenancePlanningNotVisible
    asset = session.scalar(
        select(AssetRecord).where(
            AssetRecord.tenant_id == tenant_id,
            AssetRecord.asset_id == incident.asset_id,
        )
    )
    if asset is None:
        raise MaintenancePlanningNotVisible
    if diagnosis.status != "COMPLETED" or not isinstance(diagnosis.report, dict):
        raise MaintenancePlanningConflict(
            "maintenance_planning_completed_diagnosis_required", diagnosis.version
        )
    return incident, diagnosis, asset


def _validated_source(
    session: Session,
    record: MaintenancePlanningCouncilRecord,
    current: MaintenancePlanningView,
) -> _PlanningSource:
    incident, diagnosis, asset = _request_source(
        session,
        record.tenant_id,
        record.incident_id,
        record.diagnosis_run_id,
    )
    report = dict(diagnosis.report or {})
    manifest = dict(diagnosis.manifest)
    if record.activation_id is not None:
        try:
            require_bound_active_activation(
                session,
                tenant_id=record.tenant_id,
                activation_id=record.activation_id,
                activation_policy_hash=record.activation_policy_hash,
                target_environment=record.activation_target_environment,
            )
        except (
            MaintenancePlanningActivationConflict,
            MaintenancePlanningActivationNotVisible,
        ) as exc:
            raise MaintenancePlanningConflict(
                "maintenance_planning_source_binding_changed", record.version
            ) from exc
    if (
        diagnosis.version != record.diagnosis_version
        or diagnosis.version != current.diagnosis_version
        or incident.asset_id != record.asset_id
        or asset.asset_id != record.asset_id
        or _digest(report) != record.diagnosis_report_digest
        or _digest(manifest) != record.diagnosis_manifest_digest
        or record.diagnosis_report_digest != current.diagnosis_report_digest
        or record.diagnosis_manifest_digest != current.diagnosis_manifest_digest
    ):
        raise MaintenancePlanningConflict(
            "maintenance_planning_source_binding_changed", record.version
        )
    return _PlanningSource(
        incident={
            "incident_id": incident.incident_id,
            "asset_id": incident.asset_id,
            "severity": incident.severity,
            "category": incident.category,
        },
        diagnosis_report=report,
        diagnosis_manifest=manifest,
    )


def _gateway_request(
    council: MaintenancePlanningView,
    source: _PlanningSource,
    role: str,
    specialist_contributions: tuple[dict[str, Any], ...],
    timeout_seconds: float,
) -> tuple[GatewayRequest, str]:
    if role not in ALL_AGENT_ROLES:
        raise _PlanningFailure("ROLE_VALIDATION_FAILED", "maintenance_planning_agent_role_invalid")
    try:
        prompt = default_prompt_registry().get(council.prompt_bundle_id)
    except PromptBundleNotDeployed as exc:
        raise _PlanningFailure(
            "PROMPT_BINDING_FAILED", "maintenance_planning_prompt_not_deployed"
        ) from exc
    if (
        council.response_schema_version != MAINTENANCE_PLANNING_SCHEMA_VERSION
        or prompt.prompt_bundle_id != MAINTENANCE_PLANNING_PROMPT_BUNDLE_ID
        or prompt.content_hash != council.prompt_bundle_hash
        or prompt.output_schema_name != council.response_schema_version
        or council.model_alias not in prompt.compatible_model_aliases
    ):
        raise _PlanningFailure(
            "PROMPT_BINDING_FAILED", "maintenance_planning_prompt_binding_changed"
        )
    if role == "COORDINATOR":
        diagnosis_payload: dict[str, Any] = {
            "diagnosis_report_digest": council.diagnosis_report_digest,
            "diagnosis_manifest_digest": council.diagnosis_manifest_digest,
        }
    else:
        diagnosis_payload = source.diagnosis_report
    payload = {
        "agent_role": role,
        "incident": source.incident,
        "diagnosis_report": diagnosis_payload,
        "specialist_contributions": list(specialist_contributions),
        "advisory_only": True,
        "human_review_required": True,
        "business_side_effects_allowed": False,
    }
    input_digest = _digest(payload)
    inference_request_id = f"inference-{council.council_id}-{council.attempt_count}-{role.lower()}"
    request = GatewayRequest(
        inference_request_id=inference_request_id,
        model_alias=council.model_alias,
        required_release_id=council.model_release_id,
        subject_id=council.requested_by_subject_id,
        trace_id=f"maintenance-planning:{council.council_id}",
        request_class="MAINTENANCE_PLANNING",
        data_classification="CONFIDENTIAL",
        messages=(
            {"role": "system", "content": prompt.render_system()},
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ),
        response_schema_name=council.response_schema_version,
        response_schema=_response_schema(role),
        deadline=datetime.now(UTC) + timedelta(seconds=timeout_seconds),
        max_output_tokens=2_048,
        temperature=0.0,
        diagnosis_run_id=council.diagnosis_run_id,
        prompt_bundle_id=council.prompt_bundle_id,
        prompt_bundle_hash=council.prompt_bundle_hash,
        context_evidence=(
            ContextReference(
                reference_id=council.diagnosis_run_id,
                content_hash=council.diagnosis_report_digest,
            ),
        ),
        context_token_budget=4_096,
    )
    return request, input_digest


def _response_schema(role: str) -> dict[str, Any]:
    string_array: dict[str, Any] = {
        "type": "array",
        "maxItems": 20,
        "items": {"type": "string", "minLength": 1, "maxLength": 500},
    }
    part_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["part_reference", "quantity", "reason"],
        "properties": {
            "part_reference": {"type": "string", "minLength": 1, "maxLength": 128},
            "quantity": {"type": ["integer", "null"], "minimum": 1, "maximum": 10_000},
            "reason": {"type": "string", "minLength": 1, "maxLength": 500},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_OUTPUT_FIELDS),
        "properties": {
            "agent_role": {"type": "string", "const": role},
            "summary": {"type": "string", "minLength": 1, "maxLength": 2_000},
            "recommendations": string_array,
            "constraints": string_array,
            "missing_facts": string_array,
            "safety_hold_points": string_array,
            "required_parts": {
                "type": "array",
                "maxItems": 32,
                "items": part_schema,
            },
            "dispatch_constraints": string_array,
            "source_roles": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string", "enum": list(SPECIALIST_ROLES)},
            },
            "readiness": {
                "type": "string",
                "enum": ["READY_FOR_HUMAN_REVIEW", "BLOCKED_BY_MISSING_FACTS"],
            },
        },
    }


def _validated_output(value: object, expected_role: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != _OUTPUT_FIELDS:
        raise _PlanningFailure("OUTPUT_VALIDATION_FAILED", "maintenance_planning_output_invalid")
    role = value.get("agent_role")
    summary = value.get("summary")
    readiness = value.get("readiness")
    if (
        role != expected_role
        or not isinstance(summary, str)
        or not 1 <= len(summary.strip()) <= 2_000
        or readiness
        not in {
            "READY_FOR_HUMAN_REVIEW",
            "BLOCKED_BY_MISSING_FACTS",
        }
    ):
        raise _PlanningFailure("OUTPUT_VALIDATION_FAILED", "maintenance_planning_output_invalid")
    normalized: dict[str, Any] = {
        "agent_role": expected_role,
        "summary": summary.strip(),
        "readiness": readiness,
    }
    for field in (
        "recommendations",
        "constraints",
        "missing_facts",
        "safety_hold_points",
        "dispatch_constraints",
    ):
        normalized[field] = _string_list(value.get(field))
    parts = value.get("required_parts")
    if not isinstance(parts, list) or len(parts) > 32:
        raise _PlanningFailure("OUTPUT_VALIDATION_FAILED", "maintenance_planning_output_invalid")
    normalized_parts: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict) or set(part) != {
            "part_reference",
            "quantity",
            "reason",
        }:
            raise _PlanningFailure(
                "OUTPUT_VALIDATION_FAILED", "maintenance_planning_output_invalid"
            )
        reference = part.get("part_reference")
        quantity = part.get("quantity")
        reason = part.get("reason")
        if (
            not isinstance(reference, str)
            or not 1 <= len(reference.strip()) <= 128
            or (
                quantity is not None
                and (not isinstance(quantity, int) or not 1 <= quantity <= 10_000)
            )
            or not isinstance(reason, str)
            or not 1 <= len(reason.strip()) <= 500
        ):
            raise _PlanningFailure(
                "OUTPUT_VALIDATION_FAILED", "maintenance_planning_output_invalid"
            )
        normalized_parts.append(
            {
                "part_reference": reference.strip(),
                "quantity": quantity,
                "reason": reason.strip(),
            }
        )
    normalized["required_parts"] = normalized_parts
    source_roles = value.get("source_roles")
    if not isinstance(source_roles, list) or any(
        not isinstance(item, str) for item in source_roles
    ):
        raise _PlanningFailure("OUTPUT_VALIDATION_FAILED", "maintenance_planning_output_invalid")
    if expected_role == "COORDINATOR":
        if len(source_roles) != len(SPECIALIST_ROLES) or set(source_roles) != set(SPECIALIST_ROLES):
            raise _PlanningFailure("COORDINATION_FAILED", "maintenance_planning_sources_incomplete")
        normalized["source_roles"] = list(SPECIALIST_ROLES)
    else:
        if source_roles:
            raise _PlanningFailure(
                "OUTPUT_VALIDATION_FAILED", "maintenance_planning_output_invalid"
            )
        normalized["source_roles"] = []
    return normalized


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list) or len(value) > 20:
        raise _PlanningFailure("OUTPUT_VALIDATION_FAILED", "maintenance_planning_output_invalid")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not 1 <= len(item.strip()) <= 500:
            raise _PlanningFailure(
                "OUTPUT_VALIDATION_FAILED", "maintenance_planning_output_invalid"
            )
        normalized.append(item.strip())
    return normalized


def _council_or_hidden(
    session: Session,
    tenant_id: str,
    council_id: str,
    *,
    lock: bool = False,
) -> MaintenancePlanningCouncilRecord:
    statement = select(MaintenancePlanningCouncilRecord).where(
        MaintenancePlanningCouncilRecord.tenant_id == tenant_id,
        MaintenancePlanningCouncilRecord.council_id == council_id,
    )
    if lock:
        statement = statement.with_for_update()
    record = session.scalar(statement)
    if record is None:
        raise MaintenancePlanningNotVisible
    return record


def _view(
    session: Session,
    record: MaintenancePlanningCouncilRecord,
    *,
    dispatch_required: bool = False,
) -> MaintenancePlanningView:
    contributions = tuple(
        _contribution_view(item)
        for item in session.scalars(
            select(MaintenancePlanningContributionRecord)
            .where(
                MaintenancePlanningContributionRecord.tenant_id == record.tenant_id,
                MaintenancePlanningContributionRecord.council_id == record.council_id,
            )
            .order_by(
                MaintenancePlanningContributionRecord.attempt_number,
                MaintenancePlanningContributionRecord.agent_role,
            )
        )
    )
    return MaintenancePlanningView(
        council_id=record.council_id,
        incident_id=record.incident_id,
        diagnosis_run_id=record.diagnosis_run_id,
        asset_id=record.asset_id,
        diagnosis_version=record.diagnosis_version,
        diagnosis_report_digest=record.diagnosis_report_digest,
        diagnosis_manifest_digest=record.diagnosis_manifest_digest,
        workflow_id=record.workflow_id,
        model_alias=record.model_alias,
        model_release_id=record.model_release_id,
        model_manifest_hash=record.model_manifest_hash,
        prompt_bundle_id=record.prompt_bundle_id,
        prompt_bundle_hash=record.prompt_bundle_hash,
        response_schema_version=record.response_schema_version,
        activation_id=record.activation_id,
        activation_policy_hash=record.activation_policy_hash,
        activation_target_environment=record.activation_target_environment,
        status=record.status,
        stage=record.stage,
        attempt_count=record.attempt_count,
        plan=dict(record.plan_json) if record.plan_json is not None else None,
        result_digest=record.result_digest,
        coordinator_context_hash=record.coordinator_context_hash,
        failure_code=record.failure_code,
        requested_by_subject_id=record.requested_by_subject_id,
        reviewed_by_subject_id=record.reviewed_by_subject_id,
        review_decision=record.review_decision,
        review_reason=record.review_reason,
        started_at=_optional_utc(record.started_at),
        completed_at=_optional_utc(record.completed_at),
        reviewed_at=_optional_utc(record.reviewed_at),
        version=record.version,
        created_at=_as_utc(record.created_at),
        updated_at=_as_utc(record.updated_at),
        contributions=contributions,
        dispatch_required=dispatch_required,
    )


def _contribution_view(
    record: MaintenancePlanningContributionRecord,
) -> MaintenancePlanningContributionView:
    return MaintenancePlanningContributionView(
        contribution_id=record.contribution_id,
        attempt_number=record.attempt_number,
        agent_role=record.agent_role,
        inference_request_id=record.inference_request_id,
        input_digest=record.input_digest,
        output=dict(record.output_json),
        output_digest=record.output_digest,
        model_release_id=record.model_release_id,
        model_manifest_hash=record.model_manifest_hash,
        context_hash=record.context_hash,
        completed_at=_as_utc(record.completed_at),
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{sha256(encoded.encode()).hexdigest()}"


def _optional_utc(value: datetime | None) -> datetime | None:
    return _as_utc(value) if value is not None else None
