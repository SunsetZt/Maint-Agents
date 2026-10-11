"""Persistent, resumable Server-Sent Events transport for Agent runs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from industrial_ops_agent.agent.ag_ui import AG_UI_PROFILE, adapt_agent_event
from industrial_ops_agent.agent.events import AgentEvent, AgentEventStore, AgentRunNotFound
from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_diagnosis_dispatcher,
    get_identity,
    get_model_resolver,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.application.agent_runs import (
    AgentRunControlService,
    AgentRunPreconditionFailed,
    AgentRunResumeDispatchFailed,
    AgentRunVersionConflict,
)
from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.application.diagnoses import (
    ConfirmedDiagnosisInput,
    DiagnosisDispatcher,
    DiagnosisDispatchFailed,
    DiagnosisPreconditionFailed,
    DiagnosisService,
    DiagnosisVersionConflict,
)
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.maintenance_planning.review_isolation import (
    ReviewIsolationConflict,
    check_disclosure,
    diagnosis_for_agent,
    record_disclosure,
)
from industrial_ops_agent.model_gateway.service import ProductionModelResolver
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.repositories import DiagnosisRunRepository, IncidentRepository

router = APIRouter(tags=["agent-events"])
_TERMINAL = frozenset({"COMPLETED", "NEEDS_INFORMATION", "FAILED", "CANCELLED", "ESCALATED"})
_HEARTBEAT_POLLS = 40


class AgUiTextInputContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str = Field(min_length=1, max_length=4_000)


class AgUiInputMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    role: Literal[
        "developer",
        "system",
        "assistant",
        "user",
        "tool",
        "activity",
        "reasoning",
    ]
    content: str | list[AgUiTextInputContent] | None = None
    name: str | None = Field(default=None, max_length=128)
    tool_call_id: str | None = Field(default=None, alias="toolCallId", max_length=128)
    tool_calls: list[dict[str, Any]] | None = Field(default=None, alias="toolCalls")


class AgUiClientTool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    description: str = Field(max_length=1_024)
    parameters: dict[str, Any]


class AgUiContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(max_length=512)
    value: str = Field(max_length=4_000)


class AgUiResumeInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    interrupt_id: str = Field(alias="interruptId", min_length=1, max_length=128)
    status: Literal["resolved", "cancelled"]
    payload: dict[str, Any] | None = None


class RunAgentInput(BaseModel):
    """Project-locked AG-UI RunAgentInput transport contract."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    thread_id: str = Field(alias="threadId", min_length=1, max_length=128)
    run_id: UUID = Field(alias="runId")
    parent_run_id: str | None = Field(default=None, alias="parentRunId", max_length=128)
    state: dict[str, Any]
    messages: list[AgUiInputMessage] = Field(max_length=100)
    tools: list[AgUiClientTool] = Field(max_length=32)
    context: list[AgUiContext] = Field(max_length=32)
    forwarded_props: dict[str, Any] = Field(alias="forwardedProps")
    resume: list[AgUiResumeInput] = Field(default_factory=list, max_length=8)


@router.get("/agent-runs/{agent_run_id}/events")
async def stream_agent_events(
    agent_run_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    sequence = _parse_last_event_id(last_event_id)
    await run_in_threadpool(
        _authorize_run,
        database,
        authorizer,
        identity,
        agent_run_id,
        request_id=request.state.request_id,
    )
    return _event_stream(
        request,
        AgentEventStore(database),
        database=database,
        identity=identity,
        agent_run_id=agent_run_id,
        sequence=sequence,
        encode=_encode,
    )


@router.get("/agent-runs/{agent_run_id}/ag-ui/events")
async def stream_ag_ui_events(
    agent_run_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    """Project-locked AG-UI projection over the durable Agent event log."""

    sequence = _parse_last_event_id(last_event_id)
    stream_context = await run_in_threadpool(
        _authorize_run,
        database,
        authorizer,
        identity,
        agent_run_id,
        request_id=request.state.request_id,
    )
    return _event_stream(
        request,
        AgentEventStore(database),
        database=database,
        identity=identity,
        agent_run_id=agent_run_id,
        sequence=sequence,
        encode=partial(
            _encode_ag_ui,
            thread_id=stream_context.thread_id,
            diagnosis_run_id=stream_context.diagnosis_run_id,
        ),
        ag_ui=True,
    )


@router.post(
    "/ag-ui/agents/industrial-diagnosis/runs",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "AG-UI BaseEvent stream",
            "content": {"text/event-stream": {}},
        }
    },
)
async def run_diagnosis_agent(
    body: RunAgentInput,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[DiagnosisDispatcher, Depends(get_diagnosis_dispatcher)],
    model_resolver: Annotated[ProductionModelResolver | None, Depends(get_model_resolver)],
) -> StreamingResponse:
    """Accept standard RunAgentInput and return the resulting durable AG-UI stream."""

    sequence = _parse_last_event_id(request.headers.get("Last-Event-ID"))
    run_id = str(body.run_id)
    if body.tools:
        return _ag_ui_error_stream(
            body,
            code="ag_ui_client_tools_not_allowed",
            message="Client-provided tools are not enabled for this governed agent.",
        )
    try:
        user_message = _latest_user_message(body.messages)
        if body.parent_run_id is None:
            if body.resume:
                raise _AgUiInputRejected("ag_ui_parent_run_required")
            incident_version = _state_version(body.state, "incidentVersion")
            result = await DiagnosisService(
                database,
                authorizer,
                dispatcher,
                model_resolver=model_resolver,
                model_alias=request.app.state.settings.model_gateway_alias,
            ).start(
                identity,
                body.thread_id,
                expected_version=incident_version,
                idempotency_key=f"ag-ui-run:{run_id}",
                request_id=request.state.request_id,
                requested_agent_run_id=run_id,
                confirmed_input=(
                    ConfirmedDiagnosisInput(user_message.id, user_message.text)
                    if user_message is not None
                    else None
                ),
            )
        else:
            resume = _resolved_resume(body.resume)
            clarification = _resume_clarification(resume, user_message)
            result = await AgentRunControlService(
                database,
                authorizer,
                dispatcher,
            ).resume(
                identity,
                body.parent_run_id,
                clarification=clarification,
                expected_version=_state_version(body.state, "agentRunVersion"),
                idempotency_key=f"ag-ui-run:{run_id}",
                request_id=request.state.request_id,
                requested_agent_run_id=run_id,
                interrupt_id=resume.interrupt_id,
                source_message_id=user_message.id if user_message is not None else None,
                expected_incident_id=body.thread_id,
            )
    except _AgUiInputRejected as exc:
        return _ag_ui_error_stream(body, code=exc.reason_code, message=exc.safe_message)
    except DiagnosisVersionConflict as exc:
        return _ag_ui_error_stream(
            body,
            code="incident_version_conflict",
            message=f"Incident version conflict; current version is {exc.current_version}.",
        )
    except AgentRunVersionConflict as exc:
        return _ag_ui_error_stream(
            body,
            code="agent_run_version_conflict",
            message=f"Agent run version conflict; current version is {exc.current_version}.",
        )
    except (DiagnosisPreconditionFailed, AgentRunPreconditionFailed) as exc:
        return _ag_ui_error_stream(
            body,
            code=exc.reason_code,
            message="Agent input preconditions are not satisfied.",
        )
    except ResourceNotVisible as exc:
        raise AppError(
            status_code=404,
            code="resource_not_found_or_not_visible",
            category="not_found",
            message="Resource not found or not visible",
        ) from exc
    except (DiagnosisDispatchFailed, AgentRunResumeDispatchFailed) as exc:
        raise AppError(
            status_code=503,
            code="agent_workflow_unavailable",
            category="dependency",
            message="Agent workflow unavailable",
            retryable=True,
        ) from exc

    if result.run.agent_run_id != run_id or result.run.incident_id != body.thread_id:
        raise RuntimeError("AG-UI run identity diverged from persisted diagnosis state")
    return _event_stream(
        request,
        AgentEventStore(database),
        database=database,
        identity=identity,
        agent_run_id=result.run.agent_run_id,
        sequence=sequence,
        encode=partial(
            _encode_ag_ui,
            diagnosis_run_id=result.run.diagnosis_run_id,
            thread_id=result.run.incident_id,
        ),
        ag_ui=True,
    )


@dataclass(frozen=True, slots=True)
class _RunStreamContext:
    diagnosis_run_id: str
    thread_id: str


@dataclass(frozen=True, slots=True)
class _UserMessage:
    id: str
    text: str


class _AgUiInputRejected(Exception):
    def __init__(self, reason_code: str, safe_message: str | None = None) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.safe_message = safe_message or "AG-UI input is invalid for this agent."


def _authorize_run(
    database: Database,
    authorizer: Authorizer,
    identity: IdentityContext,
    agent_run_id: str,
    *,
    request_id: str,
) -> _RunStreamContext:
    with database.transaction(identity.tenant_context) as session:
        diagnosis = DiagnosisRunRepository(session, identity.tenant_context).get_by_agent_run(
            agent_run_id
        )
        if diagnosis is None:
            raise ResourceNotVisible
        incident = IncidentRepository(session, identity.tenant_context).get(diagnosis.incident_id)
        if incident is None:
            raise ResourceNotVisible
        try:
            authorizer.require(
                identity,
                Action.READ_DIAGNOSIS,
                ResourceContext(
                    tenant_id=identity.tenant_id,
                    resource_id=agent_run_id,
                    asset_id=incident.asset_id,
                ),
                request_id=request_id,
            )
        except AuthorizationDenied as exc:
            raise ResourceNotVisible from exc
        check_disclosure(session, identity, (diagnosis.diagnosis_run_id,))
        return _RunStreamContext(
            diagnosis_run_id=diagnosis.diagnosis_run_id,
            thread_id=diagnosis.incident_id,
        )


def _parse_last_event_id(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        sequence = int(value)
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="invalid_last_event_id",
            category="validation",
            message="Last-Event-ID must be a non-negative event sequence",
        ) from exc
    if sequence < 0:
        raise AppError(
            status_code=400,
            code="invalid_last_event_id",
            category="validation",
            message="Last-Event-ID must be a non-negative event sequence",
        )
    return sequence


def _encode(event: AgentEvent) -> str:
    payload = json.dumps(
        {
            "sequence": event.sequence,
            "occurred_at": event.occurred_at.isoformat(),
            "payload": event.payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"id: {event.sequence}\nevent: {event.event_type}\ndata: {payload}\n\n"


def _encode_ag_ui(
    event: AgentEvent,
    *,
    thread_id: str,
    diagnosis_run_id: str,
) -> str:
    projected = adapt_agent_event(
        event,
        thread_id=thread_id,
        diagnosis_run_id=diagnosis_run_id,
    )
    payload = json.dumps(
        projected,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"id: {event.sequence}\nevent: {projected['type']}\ndata: {payload}\n\n"


def _record_stream_disclosure(
    database: Database,
    identity: IdentityContext,
    agent_run_id: str,
    *,
    request_id: str,
) -> None:
    # A short worker-thread transaction; no subject/DB lock crosses a network yield.
    with database.transaction(identity.tenant_context) as session:
        diagnosis_id = diagnosis_for_agent(session, identity.tenant_id, agent_run_id)
        record_disclosure(
            session,
            identity,
            (diagnosis_id,),
            resource_kind="agent-event-stream",
            resource_id=agent_run_id,
            request_id=request_id,
        )


def _event_stream(
    request: Request,
    event_store: AgentEventStore,
    *,
    database: Database,
    identity: IdentityContext,
    agent_run_id: str,
    sequence: int,
    encode: Callable[[AgentEvent], str],
    ag_ui: bool = False,
) -> StreamingResponse:
    async def generate() -> AsyncIterator[str]:
        cursor = sequence
        idle_polls = 0
        yield "retry: 1000\n\n"
        while not await request.is_disconnected():
            try:
                events, status = await run_in_threadpool(
                    event_store.list_after,
                    identity.tenant_context,
                    agent_run_id,
                    cursor,
                )
            except AgentRunNotFound:
                return
            if any(event.visibility == "authorized" for event in events):
                try:
                    await run_in_threadpool(
                        _record_stream_disclosure,
                        database,
                        identity,
                        agent_run_id,
                        request_id=request.state.request_id,
                    )
                except ReviewIsolationConflict as exc:
                    event_type = "RUN_ERROR" if ag_ui else "error"
                    payload = json.dumps(
                        {
                            "type": event_type,
                            "code": exc.reason,
                            "message": "Maintenance review source access is unavailable.",
                        },
                        separators=(",", ":"),
                    )
                    yield f"event: {event_type}\ndata: {payload}\n\n"
                    return
            for event in events:
                cursor = event.sequence
                if event.visibility == "authorized":
                    yield encode(event)
            if status in _TERMINAL and not events:
                return
            if events:
                idle_polls = 0
                continue
            idle_polls += 1
            if idle_polls % _HEARTBEAT_POLLS == 0:
                yield ": keepalive\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            **({"X-AG-UI-Profile": AG_UI_PROFILE} if ag_ui else {}),
        },
    )


def _ag_ui_error_stream(
    body: RunAgentInput,
    *,
    code: str,
    message: str,
) -> StreamingResponse:
    async def generate() -> AsyncIterator[str]:
        payload = json.dumps(
            {
                "type": "RUN_ERROR",
                "message": message,
                "code": code,
                "rawEvent": {
                    "profile": AG_UI_PROFILE,
                    "threadId": body.thread_id,
                    "runId": str(body.run_id),
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        yield f"event: RUN_ERROR\ndata: {payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-AG-UI-Profile": AG_UI_PROFILE,
        },
    )


def _latest_user_message(messages: list[AgUiInputMessage]) -> _UserMessage | None:
    for message in reversed(messages):
        if message.role != "user":
            continue
        if isinstance(message.content, str):
            text = message.content.strip()
        elif isinstance(message.content, list):
            text = "\n".join(item.text.strip() for item in message.content).strip()
        else:
            raise _AgUiInputRejected("ag_ui_user_message_empty")
        if not 2 <= len(text) <= 4_000:
            raise _AgUiInputRejected("ag_ui_user_message_invalid")
        return _UserMessage(id=message.id, text=text)
    return None


def _resolved_resume(values: list[AgUiResumeInput]) -> AgUiResumeInput:
    if len(values) != 1:
        raise _AgUiInputRejected("ag_ui_resume_must_cover_open_interrupt")
    value = values[0]
    if value.status != "resolved":
        raise _AgUiInputRejected("ag_ui_cancelled_resume_not_supported")
    return value


def _resume_clarification(
    resume: AgUiResumeInput,
    user_message: _UserMessage | None,
) -> str:
    payload = resume.payload
    if payload is None or set(payload) != {"clarification"}:
        raise _AgUiInputRejected("ag_ui_resume_payload_invalid")
    clarification = payload.get("clarification")
    if not isinstance(clarification, str):
        raise _AgUiInputRejected("ag_ui_resume_payload_invalid")
    normalized = clarification.strip()
    if not 2 <= len(normalized) <= 4_000:
        raise _AgUiInputRejected("ag_ui_resume_payload_invalid")
    if user_message is not None and user_message.text != normalized:
        raise _AgUiInputRejected("ag_ui_resume_message_mismatch")
    return normalized


def _state_version(state: dict[str, Any], key: str) -> int:
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _AgUiInputRejected(f"ag_ui_{key}_required")
    return value
