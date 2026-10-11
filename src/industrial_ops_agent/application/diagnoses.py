"""Explicit diagnosis commands and fixed-run manifest construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.agent.contracts import (
    DIAGNOSIS_AGENT_PROMPT_ID,
    DIAGNOSIS_GRAPH_VERSION,
    DIAGNOSIS_READONLY_TOOLS,
)
from industrial_ops_agent.agent.models import AgentBudget, DiagnosisManifest, DiagnosisRun
from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.model_gateway.service import (
    ModelGatewayError,
    ProductionModelResolver,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AgentEventRecord,
    AgentRunRecord,
    DiagnosisRunRecord,
    IdempotencyRecord,
    IncidentRecord,
    RealtimeMediaSessionRecord,
    RealtimeTranscriptSegmentRecord,
    WorkOrderFieldEntryRecord,
    WorkOrderRecord,
)
from industrial_ops_agent.persistence.repositories import (
    AgentEventRepository,
    AgentRunRepository,
    AssetRepository,
    DiagnosisRunRepository,
    EvidenceBundleRepository,
    IdempotencyRepository,
    IncidentRepository,
    IndexReleaseRepository,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.prompting.registry import default_prompt_registry


class DiagnosisVersionConflict(Exception):
    def __init__(self, current_version: int) -> None:
        super().__init__("incident version conflict")
        self.current_version = current_version


class DiagnosisPreconditionFailed(Exception):
    def __init__(self, reason_code: str) -> None:
        super().__init__("diagnosis precondition failed")
        self.reason_code = reason_code


class DiagnosisDispatchFailed(Exception):
    """The durable workflow dependency rejected a newly queued run."""


def advance_incident_after_completed_diagnosis(
    session: Session,
    *,
    tenant_id: str,
    incident_id: str,
    now: datetime,
) -> bool:
    """Advance only an active diagnosis lifecycle; never overwrite later decisions."""

    incident = session.scalar(
        select(IncidentRecord)
        .where(
            IncidentRecord.tenant_id == tenant_id,
            IncidentRecord.incident_id == incident_id,
        )
        .with_for_update()
    )
    if incident is None:
        raise RuntimeError("diagnosis incident disappeared before completion")
    if incident.status not in {"TRIAGED", "DIAGNOSING"}:
        return False
    incident.status = "DIAGNOSED"
    incident.version += 1
    incident.updated_at = now
    return True


@dataclass(frozen=True, slots=True)
class DiagnosisWorkflowInput:
    tenant_id: str
    subject_id: str
    roles: tuple[str, ...]
    asset_ids: tuple[str, ...]
    site_ids: tuple[str, ...]
    incident_id: str
    diagnosis_run_id: str
    agent_run_id: str
    workflow_id: str
    request_id: str
    source_diagnosis_run_id: str | None = None
    confirmed_input_event_id: str | None = None
    oidc_subject: str | None = None
    issued_at: str | None = None
    expires_at: str | None = None


class DiagnosisDispatcher(Protocol):
    async def dispatch(self, command: DiagnosisWorkflowInput) -> None: ...

    async def cancel(self, workflow_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class DiagnosisStartResult:
    run: DiagnosisRun
    created: bool


@dataclass(frozen=True, slots=True)
class ConfirmedDiagnosisInput:
    message_id: str
    text: str


FIELD_REDIAGNOSIS_ACTIVE_STATUSES = frozenset({"QUEUED", "RUNNING"})


class DiagnosisService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        dispatcher: DiagnosisDispatcher,
        *,
        model_resolver: ProductionModelResolver | None = None,
        model_alias: str = "industrial-diagnosis",
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._dispatcher = dispatcher
        self._model_resolver = model_resolver
        self._model_alias = model_alias

    def _resolve_model_binding(
        self, context: TenantContext, index_release_id: str
    ) -> dict[str, Any]:
        if self._model_resolver is None:
            raise DiagnosisPreconditionFailed("model_gateway_required")
        try:
            resolved = self._model_resolver.resolve(context, self._model_alias)
        except ModelGatewayError as exc:
            raise DiagnosisPreconditionFailed(exc.reason) from exc
        binding = resolved.diagnosis_binding or {}
        definition = default_prompt_registry().get(DIAGNOSIS_AGENT_PROMPT_ID)
        prompt = binding.get("prompt_bundle")
        retrieval = binding.get("retrieval")
        tools = binding.get("tool_registry")
        if (
            binding.get("agent_graph_id") != DIAGNOSIS_GRAPH_VERSION
            or binding.get("prompt_bundle_id") != DIAGNOSIS_AGENT_PROMPT_ID
            or not isinstance(prompt, dict)
            or prompt.get("content_hash") != definition.content_hash
            or not isinstance(retrieval, dict)
            or retrieval.get("index_release_id") != index_release_id
            or not isinstance(tools, dict)
            or any(
                not isinstance(tools.get(name), str) or not tools[name]
                for name in DIAGNOSIS_READONLY_TOOLS
            )
        ):
            raise DiagnosisPreconditionFailed("diagnosis_release_binding_mismatch")
        return {
            "model_release": resolved.release_id,
            "prompt_bundle": binding["prompt_bundle_id"],
            "prompt_bundle_hash": definition.content_hash,
            "agent_graph_version": binding["agent_graph_id"],
            "index_release_id": retrieval["index_release_id"],
            "tool_versions": {name: tools[name] for name in DIAGNOSIS_READONLY_TOOLS},
        }

    async def start(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
        request_id: str,
        requested_agent_run_id: str | None = None,
        confirmed_input: ConfirmedDiagnosisInput | None = None,
    ) -> DiagnosisStartResult:
        normalized_input: ConfirmedDiagnosisInput | None = None
        if confirmed_input is not None:
            text = confirmed_input.text.strip()
            if not 1 <= len(confirmed_input.message_id) <= 128 or not 2 <= len(text) <= 4_000:
                raise DiagnosisPreconditionFailed("confirmed_input_invalid")
            normalized_input = ConfirmedDiagnosisInput(
                message_id=confirmed_input.message_id,
                text=text,
            )
        now = datetime.now(UTC)
        request_facts: dict[str, str | int] = {
            "incident_id": incident_id,
            "expected_version": expected_version,
        }
        if requested_agent_run_id is not None:
            request_facts["requested_agent_run_id"] = requested_agent_run_id
        if normalized_input is not None:
            request_facts["confirmed_input_sha256"] = sha256(
                normalized_input.text.encode()
            ).hexdigest()
            request_facts["confirmed_input_message_id"] = normalized_input.message_id
        request_hash = _canonical_hash(request_facts)
        storage_key = f"diagnosis:{idempotency_key}"
        command: DiagnosisWorkflowInput | None = None
        with self._database.transaction(identity.tenant_context) as session:
            incidents = IncidentRepository(session, identity.tenant_context)
            incident = incidents.get(incident_id)
            if incident is None:
                raise ResourceNotVisible
            try:
                self._authorizer.require(
                    identity,
                    Action.START_DIAGNOSIS,
                    ResourceContext(
                        tenant_id=identity.tenant_id,
                        resource_id=incident_id,
                        asset_id=incident.asset_id,
                    ),
                    request_id=request_id,
                )
            except AuthorizationDenied as exc:
                raise ResourceNotVisible from exc
            asset = AssetRepository(session, identity.tenant_context).get(incident.asset_id)
            if asset is None:
                raise ResourceNotVisible
            if not (asset.model_code or "").strip():
                raise DiagnosisPreconditionFailed("asset_model_unavailable")
            idempotency = IdempotencyRepository(session, identity.tenant_context)
            existing = idempotency.get(subject_id=identity.subject_id, key=storage_key)
            runs = DiagnosisRunRepository(session, identity.tenant_context)
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise DiagnosisPreconditionFailed("idempotency_conflict")
                run = runs.get(existing.result_ref)
                if run is None:
                    raise RuntimeError("idempotency record references a missing diagnosis")
                if (
                    requested_agent_run_id is not None
                    and run.agent_run_id != requested_agent_run_id
                ):
                    raise DiagnosisPreconditionFailed("idempotency_conflict")
                command = _workflow_command(identity, run, request_id=request_id)
                try:
                    await self._dispatcher.dispatch(command)
                except Exception as exc:
                    raise DiagnosisDispatchFailed from exc
                return DiagnosisStartResult(_run(run), created=False)
            if incident.version != expected_version:
                raise DiagnosisVersionConflict(incident.version)
            if incident.status != "TRIAGED":
                raise DiagnosisPreconditionFailed("incident_not_triaged")
            evidence = EvidenceBundleRepository(session, identity.tenant_context).get(
                incident.evidence_bundle_id
            )
            if evidence is None or evidence.status != "CONFIRMED":
                raise DiagnosisPreconditionFailed("evidence_not_confirmed")
            if (
                not evidence.automation_eligible
                or evidence.processor_versions.get("vlm_status") == "unsupported_model"
            ):
                raise DiagnosisPreconditionFailed("evidence_not_automation_eligible")
            release = IndexReleaseRepository(session, identity.tenant_context).active()
            if release is None:
                raise DiagnosisPreconditionFailed("active_index_unavailable")
            binding = self._resolve_model_binding(identity.tenant_context, release.release_id)
            diagnosis_run_id = f"diagnosis-{uuid4().hex}"
            agent_run_id = requested_agent_run_id or f"agent-{uuid4().hex}"
            workflow_id = f"m2-diagnosis/{identity.tenant_id}/{diagnosis_run_id}"
            budget = AgentBudget()
            manifest = DiagnosisManifest(
                incident_version=incident.version,
                evidence_bundle_id=evidence.bundle_id,
                evidence_version=evidence.version,
                **binding,
                budget=budget,
                created_at=now,
            ).as_dict()
            confirmed_event_id: str | None = None
            if normalized_input is not None:
                confirmed_event_id = f"agent-event-{uuid4().hex}"
                manifest["input_context"] = {
                    "kind": "human_confirmed_initial_message",
                    "source_diagnosis_run_id": diagnosis_run_id,
                    "source_agent_run_id": agent_run_id,
                    "source_agent_event_id": confirmed_event_id,
                    "source_agent_event_sequence": 2,
                    "source_message_id": normalized_input.message_id,
                    "confirmed_text_sha256": sha256(normalized_input.text.encode()).hexdigest(),
                }
            run_record = DiagnosisRunRecord(
                diagnosis_run_id=diagnosis_run_id,
                tenant_id=identity.tenant_id,
                incident_id=incident_id,
                evidence_bundle_id=evidence.bundle_id,
                agent_run_id=agent_run_id,
                workflow_id=workflow_id,
                status="QUEUED",
                manifest=manifest,
                report=None,
                stop_reason=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            runs.add(run_record)
            # Persist each parent before adding its FK-dependent child.  Bare scalar
            # foreign-key ids do not give the ORM an object relationship to order by.
            session.flush()
            AgentRunRepository(session, identity.tenant_context).add(
                AgentRunRecord(
                    agent_run_id=agent_run_id,
                    tenant_id=identity.tenant_id,
                    diagnosis_run_id=diagnosis_run_id,
                    status="QUEUED",
                    budget=budget.as_dict(),
                    last_sequence=2 if normalized_input is not None else 1,
                    version=2 if normalized_input is not None else 1,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            AgentEventRepository(session, identity.tenant_context).add(
                AgentEventRecord(
                    event_id=f"agent-event-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    agent_run_id=agent_run_id,
                    sequence=1,
                    event_type="diagnosis.queued",
                    payload={
                        "diagnosis_run_id": diagnosis_run_id,
                        "incident_id": incident_id,
                        "workflow_id": workflow_id,
                    },
                    visibility="authorized",
                    occurred_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            if normalized_input is not None and confirmed_event_id is not None:
                AgentEventRepository(session, identity.tenant_context).add(
                    AgentEventRecord(
                        event_id=confirmed_event_id,
                        tenant_id=identity.tenant_id,
                        agent_run_id=agent_run_id,
                        sequence=2,
                        event_type="agent.user_input.confirmed",
                        payload={
                            "diagnosis_run_id": diagnosis_run_id,
                            "source_type": "ag_ui_user_message",
                            "source_id": normalized_input.message_id,
                            "text": normalized_input.text,
                            "reviewer_subject_id": identity.subject_id,
                        },
                        visibility="authorized",
                        occurred_at=now,
                        created_at=now,
                        updated_at=now,
                    )
                )
            idempotency.add(
                IdempotencyRecord(
                    record_id=f"idem-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=diagnosis_run_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            command = _workflow_command(identity, run_record, request_id=request_id)
            result = DiagnosisStartResult(_run(run_record), created=True)
        if command is None:
            raise RuntimeError("new diagnosis did not create a workflow command")
        try:
            await self._dispatcher.dispatch(command)
        except Exception as exc:
            raise DiagnosisDispatchFailed from exc
        return result

    async def reanalyze_with_confirmed_transcript(
        self,
        identity: IdentityContext,
        diagnosis_run_id: str,
        *,
        transcript_segment_id: str,
        expected_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> DiagnosisStartResult:
        """Create a new immutable run causally bound to one confirmed ASR input."""

        now = datetime.now(UTC)
        request_hash = _canonical_hash(
            {
                "source_diagnosis_run_id": diagnosis_run_id,
                "transcript_segment_id": transcript_segment_id,
                "expected_version": expected_version,
            }
        )
        storage_key = f"diagnosis-reanalysis:{idempotency_key}"
        command: DiagnosisWorkflowInput | None = None
        with self._database.transaction(identity.tenant_context) as session:
            source = DiagnosisRunRepository(session, identity.tenant_context).get(diagnosis_run_id)
            if source is None:
                raise ResourceNotVisible
            incident = IncidentRepository(session, identity.tenant_context).get(source.incident_id)
            if incident is None:
                raise ResourceNotVisible
            try:
                self._authorizer.require(
                    identity,
                    Action.START_DIAGNOSIS,
                    ResourceContext(
                        tenant_id=identity.tenant_id,
                        resource_id=diagnosis_run_id,
                        asset_id=incident.asset_id,
                    ),
                    request_id=request_id,
                )
            except AuthorizationDenied as exc:
                raise ResourceNotVisible from exc
            if source.version != expected_version:
                raise DiagnosisVersionConflict(source.version)
            if source.status not in {"COMPLETED", "NEEDS_INFORMATION"}:
                raise DiagnosisPreconditionFailed("source_diagnosis_not_terminal")
            transcript = session.scalar(
                select(RealtimeTranscriptSegmentRecord).where(
                    RealtimeTranscriptSegmentRecord.tenant_id == identity.tenant_id,
                    RealtimeTranscriptSegmentRecord.segment_id == transcript_segment_id,
                )
            )
            media_session = (
                session.scalar(
                    select(RealtimeMediaSessionRecord).where(
                        RealtimeMediaSessionRecord.tenant_id == identity.tenant_id,
                        RealtimeMediaSessionRecord.session_id == transcript.session_id,
                    )
                )
                if transcript is not None
                else None
            )
            confirmed_event = (
                session.scalar(
                    select(AgentEventRecord).where(
                        AgentEventRecord.tenant_id == identity.tenant_id,
                        AgentEventRecord.event_id == transcript.agent_event_id,
                        AgentEventRecord.agent_run_id == source.agent_run_id,
                    )
                )
                if transcript is not None and transcript.agent_event_id is not None
                else None
            )
            if (
                transcript is None
                or media_session is None
                or transcript.status != "ACCEPTED"
                or not transcript.confirmed_text
                or transcript.reviewer_subject_id != identity.subject_id
                or transcript.agent_event_id is None
                or transcript.agent_event_sequence is None
                or confirmed_event is None
                or confirmed_event.event_type != "agent.user_input.confirmed"
                or confirmed_event.sequence != transcript.agent_event_sequence
                or confirmed_event.payload.get("source_id") != transcript.segment_id
                or confirmed_event.payload.get("text") != transcript.confirmed_text
                or media_session.diagnosis_run_id != diagnosis_run_id
                or media_session.incident_id != source.incident_id
                or media_session.subject_id != identity.subject_id
            ):
                raise DiagnosisPreconditionFailed("confirmed_transcript_input_invalid")
            idempotency = IdempotencyRepository(session, identity.tenant_context)
            existing = idempotency.get(subject_id=identity.subject_id, key=storage_key)
            runs = DiagnosisRunRepository(session, identity.tenant_context)
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise DiagnosisPreconditionFailed("idempotency_conflict")
                replay = runs.get(existing.result_ref)
                if replay is None:
                    raise RuntimeError("idempotency record references a missing diagnosis")
                replay_command = _workflow_command(
                    identity,
                    replay,
                    request_id=request_id,
                )
                try:
                    await self._dispatcher.dispatch(replay_command)
                except Exception as exc:
                    raise DiagnosisDispatchFailed from exc
                return DiagnosisStartResult(_run(replay), created=False)
            evidence = EvidenceBundleRepository(session, identity.tenant_context).get(
                source.evidence_bundle_id
            )
            if evidence is None or evidence.status != "CONFIRMED":
                raise DiagnosisPreconditionFailed("evidence_not_confirmed")
            if (
                not evidence.automation_eligible
                or evidence.processor_versions.get("vlm_status") == "unsupported_model"
            ):
                raise DiagnosisPreconditionFailed("evidence_not_automation_eligible")
            release = IndexReleaseRepository(session, identity.tenant_context).active()
            if release is None:
                raise DiagnosisPreconditionFailed("active_index_unavailable")
            binding = self._resolve_model_binding(identity.tenant_context, release.release_id)
            new_diagnosis_run_id = f"diagnosis-{uuid4().hex}"
            new_agent_run_id = f"agent-{uuid4().hex}"
            workflow_id = f"m2-diagnosis-reanalysis/{identity.tenant_id}/{new_diagnosis_run_id}"
            budget = AgentBudget()
            manifest = DiagnosisManifest(
                incident_version=incident.version,
                evidence_bundle_id=evidence.bundle_id,
                evidence_version=evidence.version,
                **binding,
                budget=budget,
                created_at=now,
            ).as_dict()
            manifest["input_context"] = {
                "kind": "human_confirmed_realtime_transcript",
                "source_diagnosis_run_id": diagnosis_run_id,
                "source_agent_event_id": transcript.agent_event_id,
                "source_agent_event_sequence": transcript.agent_event_sequence,
                "source_segment_id": transcript.segment_id,
                "confirmed_text_sha256": sha256(transcript.confirmed_text.encode()).hexdigest(),
            }
            run_record = DiagnosisRunRecord(
                diagnosis_run_id=new_diagnosis_run_id,
                tenant_id=identity.tenant_id,
                incident_id=source.incident_id,
                evidence_bundle_id=evidence.bundle_id,
                agent_run_id=new_agent_run_id,
                workflow_id=workflow_id,
                status="QUEUED",
                manifest=manifest,
                report=None,
                stop_reason=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            runs.add(run_record)
            session.flush()
            AgentRunRepository(session, identity.tenant_context).add(
                AgentRunRecord(
                    agent_run_id=new_agent_run_id,
                    tenant_id=identity.tenant_id,
                    diagnosis_run_id=new_diagnosis_run_id,
                    status="QUEUED",
                    budget=budget.as_dict(),
                    last_sequence=1,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            AgentEventRepository(session, identity.tenant_context).add(
                AgentEventRecord(
                    event_id=f"agent-event-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    agent_run_id=new_agent_run_id,
                    sequence=1,
                    event_type="diagnosis.reanalysis_queued",
                    payload={
                        "diagnosis_run_id": new_diagnosis_run_id,
                        "source_diagnosis_run_id": diagnosis_run_id,
                        "parent_agent_run_id": source.agent_run_id,
                        "confirmed_input_event_id": transcript.agent_event_id,
                        "workflow_id": workflow_id,
                    },
                    visibility="authorized",
                    occurred_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            idempotency.add(
                IdempotencyRecord(
                    record_id=f"idem-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=new_diagnosis_run_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            command = DiagnosisWorkflowInput(
                tenant_id=identity.tenant_id,
                subject_id=identity.subject_id,
                roles=tuple(sorted(role.value for role in identity.roles)),
                asset_ids=tuple(sorted(identity.asset_ids)),
                site_ids=tuple(sorted(identity.site_ids)),
                incident_id=source.incident_id,
                diagnosis_run_id=new_diagnosis_run_id,
                agent_run_id=new_agent_run_id,
                workflow_id=workflow_id,
                request_id=request_id,
                source_diagnosis_run_id=diagnosis_run_id,
                confirmed_input_event_id=transcript.agent_event_id,
                oidc_subject=identity.oidc_subject,
                issued_at=identity.issued_at.isoformat(),
                expires_at=identity.expires_at.isoformat(),
            )
            result = DiagnosisStartResult(_run(run_record), created=True)
        if command is None:
            raise RuntimeError("new diagnosis did not create a workflow command")
        try:
            await self._dispatcher.dispatch(command)
        except Exception as exc:
            raise DiagnosisDispatchFailed from exc
        return result

    async def reanalyze_with_field_observations(
        self,
        identity: IdentityContext,
        diagnosis_run_id: str,
        *,
        work_order_id: str,
        work_order_version: int,
        field_entry_ids: tuple[str, ...],
        expected_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> DiagnosisStartResult:
        """Create an immutable successor from server-resolved field observations."""

        normalized_entry_ids = tuple(sorted(field_entry_ids))
        if (
            not 1 <= len(normalized_entry_ids) <= 20
            or len(set(normalized_entry_ids)) != len(normalized_entry_ids)
            or any(not 1 <= len(item) <= 128 for item in normalized_entry_ids)
        ):
            raise DiagnosisPreconditionFailed("field_observations_invalid")
        now = datetime.now(UTC)
        request_hash = _canonical_hash(
            {
                "source_diagnosis_run_id": diagnosis_run_id,
                "expected_version": expected_version,
                "work_order_id": work_order_id,
                "work_order_version": work_order_version,
                "field_entry_ids": normalized_entry_ids,
            }
        )
        storage_key = f"diagnosis-reanalysis:{idempotency_key}"
        command: DiagnosisWorkflowInput | None = None
        with self._database.transaction(identity.tenant_context) as session:
            source = session.scalar(
                select(DiagnosisRunRecord)
                .where(
                    DiagnosisRunRecord.tenant_id == identity.tenant_id,
                    DiagnosisRunRecord.diagnosis_run_id == diagnosis_run_id,
                )
                .with_for_update()
            )
            if source is None:
                raise ResourceNotVisible
            incident = IncidentRepository(session, identity.tenant_context).get(source.incident_id)
            if incident is None:
                raise ResourceNotVisible
            try:
                self._authorizer.require(
                    identity,
                    Action.START_DIAGNOSIS,
                    ResourceContext(
                        tenant_id=identity.tenant_id,
                        resource_id=diagnosis_run_id,
                        asset_id=incident.asset_id,
                    ),
                    request_id=request_id,
                )
            except AuthorizationDenied as exc:
                raise ResourceNotVisible from exc
            if source.version != expected_version:
                raise DiagnosisVersionConflict(source.version)
            if source.status not in {"COMPLETED", "NEEDS_INFORMATION"}:
                raise DiagnosisPreconditionFailed("source_diagnosis_not_terminal")

            idempotency = IdempotencyRepository(session, identity.tenant_context)
            existing = idempotency.get(subject_id=identity.subject_id, key=storage_key)
            runs = DiagnosisRunRepository(session, identity.tenant_context)
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise DiagnosisPreconditionFailed("idempotency_conflict")
                replay = runs.get(existing.result_ref)
                if replay is None:
                    raise RuntimeError("idempotency record references a missing diagnosis")
                replay_command = _workflow_command(
                    identity,
                    replay,
                    request_id=request_id,
                )
                try:
                    await self._dispatcher.dispatch(replay_command)
                except Exception as exc:
                    raise DiagnosisDispatchFailed from exc
                return DiagnosisStartResult(_run(replay), created=False)

            work = session.scalar(
                select(WorkOrderRecord)
                .where(
                    WorkOrderRecord.tenant_id == identity.tenant_id,
                    WorkOrderRecord.work_order_id == work_order_id,
                )
                .with_for_update()
            )
            if work is None or work.incident_id != source.incident_id:
                raise ResourceNotVisible
            try:
                self._authorizer.require(
                    identity,
                    Action.EXECUTE_WORK_ORDER,
                    ResourceContext(
                        tenant_id=identity.tenant_id,
                        resource_id=work_order_id,
                        asset_id=incident.asset_id,
                    ),
                    request_id=request_id,
                )
            except AuthorizationDenied as exc:
                raise ResourceNotVisible from exc
            if work.assigned_subject_id != identity.subject_id:
                raise ResourceNotVisible
            if work.status not in {"ACCEPTED", "IN_PROGRESS", "ON_HOLD"}:
                raise DiagnosisPreconditionFailed("field_work_order_not_executable")
            if work.version != work_order_version:
                raise DiagnosisPreconditionFailed("field_work_order_version_conflict")
            latest_source = session.scalar(
                select(DiagnosisRunRecord)
                .where(
                    DiagnosisRunRecord.tenant_id == identity.tenant_id,
                    DiagnosisRunRecord.incident_id == source.incident_id,
                    DiagnosisRunRecord.status.in_({"COMPLETED", "NEEDS_INFORMATION"}),
                )
                .order_by(
                    DiagnosisRunRecord.updated_at.desc(),
                    DiagnosisRunRecord.diagnosis_run_id.desc(),
                )
            )
            if latest_source is None or latest_source.diagnosis_run_id != diagnosis_run_id:
                raise DiagnosisPreconditionFailed("source_diagnosis_not_current")
            if _active_field_rediagnosis_exists(
                session,
                tenant_id=identity.tenant_id,
                incident_id=source.incident_id,
                source_diagnosis_run_id=diagnosis_run_id,
                source_work_order_id=work_order_id,
            ):
                raise DiagnosisPreconditionFailed("field_rediagnosis_in_progress")

            entries = list(
                session.scalars(
                    select(WorkOrderFieldEntryRecord)
                    .where(
                        WorkOrderFieldEntryRecord.tenant_id == identity.tenant_id,
                        WorkOrderFieldEntryRecord.work_order_id == work_order_id,
                        WorkOrderFieldEntryRecord.entry_id.in_(normalized_entry_ids),
                    )
                    .order_by(
                        WorkOrderFieldEntryRecord.sequence,
                        WorkOrderFieldEntryRecord.entry_id,
                    )
                )
            )
            if len(entries) != len(normalized_entry_ids):
                raise DiagnosisPreconditionFailed("field_observations_invalid")
            try:
                field_references = [_field_observation_reference(item) for item in entries]
                _field_observation_text(entries)
            except ValueError as exc:
                raise DiagnosisPreconditionFailed("field_observations_invalid") from exc
            field_context_sha256 = _canonical_hash(field_references)

            evidence = EvidenceBundleRepository(session, identity.tenant_context).get(
                source.evidence_bundle_id
            )
            if evidence is None or evidence.status != "CONFIRMED":
                raise DiagnosisPreconditionFailed("evidence_not_confirmed")
            if (
                not evidence.automation_eligible
                or evidence.processor_versions.get("vlm_status") == "unsupported_model"
            ):
                raise DiagnosisPreconditionFailed("evidence_not_automation_eligible")
            release = IndexReleaseRepository(session, identity.tenant_context).active()
            if release is None:
                raise DiagnosisPreconditionFailed("active_index_unavailable")
            binding = self._resolve_model_binding(identity.tenant_context, release.release_id)

            source_agent = session.scalar(
                select(AgentRunRecord)
                .where(
                    AgentRunRecord.tenant_id == identity.tenant_id,
                    AgentRunRecord.agent_run_id == source.agent_run_id,
                )
                .with_for_update()
            )
            if source_agent is None or source_agent.diagnosis_run_id != diagnosis_run_id:
                raise DiagnosisPreconditionFailed("source_agent_run_invalid")
            confirmed_event_id = f"agent-event-{uuid4().hex}"
            confirmed_event_sequence = source_agent.last_sequence + 1
            input_context = {
                "kind": "human_confirmed_field_observations",
                "source_diagnosis_run_id": diagnosis_run_id,
                "source_agent_run_id": source.agent_run_id,
                "source_agent_event_id": confirmed_event_id,
                "source_agent_event_sequence": confirmed_event_sequence,
                "source_work_order_id": work_order_id,
                "source_work_order_version": work_order_version,
                "field_entries": field_references,
                "field_context_sha256": field_context_sha256,
            }
            new_diagnosis_run_id = f"diagnosis-{uuid4().hex}"
            new_agent_run_id = f"agent-{uuid4().hex}"
            workflow_id = (
                f"m2-diagnosis-field-reanalysis/{identity.tenant_id}/{new_diagnosis_run_id}"
            )
            budget = AgentBudget()
            manifest = DiagnosisManifest(
                incident_version=incident.version,
                evidence_bundle_id=evidence.bundle_id,
                evidence_version=evidence.version,
                **binding,
                budget=budget,
                created_at=now,
            ).as_dict()
            manifest["input_context"] = input_context
            run_record = DiagnosisRunRecord(
                diagnosis_run_id=new_diagnosis_run_id,
                tenant_id=identity.tenant_id,
                incident_id=source.incident_id,
                evidence_bundle_id=evidence.bundle_id,
                agent_run_id=new_agent_run_id,
                workflow_id=workflow_id,
                status="QUEUED",
                manifest=manifest,
                report=None,
                stop_reason=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            runs.add(run_record)
            session.flush()
            AgentRunRepository(session, identity.tenant_context).add(
                AgentRunRecord(
                    agent_run_id=new_agent_run_id,
                    tenant_id=identity.tenant_id,
                    diagnosis_run_id=new_diagnosis_run_id,
                    status="QUEUED",
                    budget=budget.as_dict(),
                    last_sequence=1,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            AgentEventRepository(session, identity.tenant_context).add(
                AgentEventRecord(
                    event_id=confirmed_event_id,
                    tenant_id=identity.tenant_id,
                    agent_run_id=source.agent_run_id,
                    sequence=confirmed_event_sequence,
                    event_type="agent.user_input.confirmed",
                    payload={
                        "diagnosis_run_id": diagnosis_run_id,
                        "source_type": "field_observations",
                        "source_id": work_order_id,
                        "source_work_order_version": work_order_version,
                        "field_entries": field_references,
                        "field_context_sha256": field_context_sha256,
                        "reviewer_subject_id": identity.subject_id,
                    },
                    visibility="authorized",
                    occurred_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            source_agent.last_sequence = confirmed_event_sequence
            source_agent.version += 1
            source_agent.updated_at = now
            AgentEventRepository(session, identity.tenant_context).add(
                AgentEventRecord(
                    event_id=f"agent-event-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    agent_run_id=new_agent_run_id,
                    sequence=1,
                    event_type="diagnosis.reanalysis_queued",
                    payload={
                        "diagnosis_run_id": new_diagnosis_run_id,
                        "source_diagnosis_run_id": diagnosis_run_id,
                        "parent_agent_run_id": source.agent_run_id,
                        "confirmed_input_event_id": confirmed_event_id,
                        "workflow_id": workflow_id,
                    },
                    visibility="authorized",
                    occurred_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            idempotency.add(
                IdempotencyRecord(
                    record_id=f"idem-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=new_diagnosis_run_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            command = DiagnosisWorkflowInput(
                tenant_id=identity.tenant_id,
                subject_id=identity.subject_id,
                roles=tuple(sorted(role.value for role in identity.roles)),
                asset_ids=tuple(sorted(identity.asset_ids)),
                site_ids=tuple(sorted(identity.site_ids)),
                incident_id=source.incident_id,
                diagnosis_run_id=new_diagnosis_run_id,
                agent_run_id=new_agent_run_id,
                workflow_id=workflow_id,
                request_id=request_id,
                source_diagnosis_run_id=diagnosis_run_id,
                confirmed_input_event_id=confirmed_event_id,
                oidc_subject=identity.oidc_subject,
                issued_at=identity.issued_at.isoformat(),
                expires_at=identity.expires_at.isoformat(),
            )
            result = DiagnosisStartResult(_run(run_record), created=True)
        if command is None:
            raise RuntimeError("new field reanalysis did not create a workflow command")
        try:
            await self._dispatcher.dispatch(command)
        except Exception as exc:
            raise DiagnosisDispatchFailed from exc
        return result

    def get(
        self,
        identity: IdentityContext,
        diagnosis_run_id: str,
        *,
        request_id: str,
    ) -> DiagnosisRun:
        from industrial_ops_agent.maintenance_planning.review_isolation import record_disclosure

        with self._database.transaction(identity.tenant_context) as session:
            run = DiagnosisRunRepository(session, identity.tenant_context).get(diagnosis_run_id)
            if run is None:
                raise ResourceNotVisible
            incident = IncidentRepository(session, identity.tenant_context).get(run.incident_id)
            if incident is None:
                raise ResourceNotVisible
            try:
                self._authorizer.require(
                    identity,
                    Action.READ_DIAGNOSIS,
                    ResourceContext(
                        tenant_id=identity.tenant_id,
                        resource_id=diagnosis_run_id,
                        asset_id=incident.asset_id,
                    ),
                    request_id=request_id,
                )
            except AuthorizationDenied as exc:
                raise ResourceNotVisible from exc
            record_disclosure(
                session,
                identity,
                (diagnosis_run_id,),
                resource_kind="diagnosis",
                resource_id=diagnosis_run_id,
                request_id=request_id,
            )
            return _run(run)


def _active_field_rediagnosis_exists(
    session: Session,
    *,
    tenant_id: str,
    incident_id: str,
    source_diagnosis_run_id: str,
    source_work_order_id: str,
) -> bool:
    candidates = session.scalars(
        select(DiagnosisRunRecord).where(
            DiagnosisRunRecord.tenant_id == tenant_id,
            DiagnosisRunRecord.incident_id == incident_id,
            DiagnosisRunRecord.status.in_(FIELD_REDIAGNOSIS_ACTIVE_STATUSES),
        )
    )
    for candidate in candidates:
        manifest = candidate.manifest
        context = manifest.get("input_context") if isinstance(manifest, dict) else None
        if (
            isinstance(context, dict)
            and context.get("kind") == "human_confirmed_field_observations"
            and context.get("source_diagnosis_run_id") == source_diagnosis_run_id
            and context.get("source_work_order_id") == source_work_order_id
        ):
            return True
    return False


def _run(record: DiagnosisRunRecord) -> DiagnosisRun:
    return DiagnosisRun(
        diagnosis_run_id=record.diagnosis_run_id,
        incident_id=record.incident_id,
        evidence_bundle_id=record.evidence_bundle_id,
        agent_run_id=record.agent_run_id,
        workflow_id=record.workflow_id,
        status=record.status,
        manifest=record.manifest,
        report=record.report,
        stop_reason=record.stop_reason,
        version=record.version,
        created_at=_as_utc(record.created_at),
        updated_at=_as_utc(record.updated_at),
    )


def _canonical_hash(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(canonical.encode()).hexdigest()


def _field_observation_reference(entry: WorkOrderFieldEntryRecord) -> dict[str, str | int]:
    _field_observation_value(entry)
    return {
        "entry_id": entry.entry_id,
        "sequence": entry.sequence,
        "entry_type": entry.entry_type,
        "content_sha256": _canonical_hash(
            {"entry_type": entry.entry_type, "payload": entry.payload}
        ),
    }


def _field_observation_text(entries: list[WorkOrderFieldEntryRecord]) -> str:
    lines = [
        f"#{entry.sequence} {entry.entry_type}: {_field_observation_value(entry)}"
        for entry in entries
    ]
    text = "\n".join(lines)
    if not text or len(text) > 4_000:
        raise ValueError("field observation text exceeds the governed context limit")
    return text


def _field_observation_value(entry: WorkOrderFieldEntryRecord) -> str:
    field = "description" if entry.entry_type == "NOTE" else "observation"
    if entry.entry_type not in {"NOTE", "AI_OBSERVATION"}:
        raise ValueError("field entry is not an observation")
    value = entry.payload.get(field)
    if not isinstance(value, str) or not (text := value.strip()) or len(text) > 4_000:
        raise ValueError("field observation content is invalid")
    return text


def _workflow_command(
    identity: IdentityContext,
    run: DiagnosisRunRecord,
    *,
    request_id: str,
) -> DiagnosisWorkflowInput:
    input_context = run.manifest.get("input_context")
    source_diagnosis_run_id = (
        input_context.get("source_diagnosis_run_id") if isinstance(input_context, dict) else None
    )
    confirmed_input_event_id = (
        input_context.get("source_agent_event_id") if isinstance(input_context, dict) else None
    )
    return DiagnosisWorkflowInput(
        tenant_id=identity.tenant_id,
        subject_id=identity.subject_id,
        roles=tuple(sorted(role.value for role in identity.roles)),
        asset_ids=tuple(sorted(identity.asset_ids)),
        site_ids=tuple(sorted(identity.site_ids)),
        incident_id=run.incident_id,
        diagnosis_run_id=run.diagnosis_run_id,
        agent_run_id=run.agent_run_id,
        workflow_id=run.workflow_id,
        request_id=request_id,
        source_diagnosis_run_id=(
            source_diagnosis_run_id if isinstance(source_diagnosis_run_id, str) else None
        ),
        confirmed_input_event_id=(
            confirmed_input_event_id if isinstance(confirmed_input_event_id, str) else None
        ),
        oidc_subject=identity.oidc_subject,
        issued_at=identity.issued_at.isoformat(),
        expires_at=identity.expires_at.isoformat(),
    )
