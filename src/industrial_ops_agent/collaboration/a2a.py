"""Official A2A SDK boundary for a fixed supplier diagnosis-review skill."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.client.errors import A2AClientError, A2AClientTimeoutError
from a2a.types import (
    AgentCard,
    Artifact,
    CancelTaskRequest,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SecurityRequirement,
    SendMessageConfiguration,
    SendMessageRequest,
    Task,
    TaskState,
)
from a2a.utils import TransportProtocol
from a2a.utils.errors import A2AError
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Value
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from industrial_ops_agent.guardrails import PromptInjectionGuard
from industrial_ops_agent.secrets import SecretValue

_MAX_TOKEN_BYTES = 16 * 1024
_MAX_MESSAGE_BYTES = 128 * 1024
_MAX_ARTIFACT_BYTES = 128 * 1024
_MAX_TOKEN_LIFETIME_SECONDS = 15 * 60
_TOKEN_REFRESH_SKEW_SECONDS = 30
_TERMINAL_STATES = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_REJECTED,
}


class SupplierA2AError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True, repr=False)
class A2AAccessToken:
    value: SecretValue
    scope: str
    expires_in_seconds: int


@dataclass(frozen=True, slots=True)
class A2AOAuthResponse:
    status_code: int
    body: bytes


class A2AOAuthTransport(Protocol):
    async def request_token(
        self,
        url: str,
        *,
        form: dict[str, str],
        timeout_seconds: float,
    ) -> A2AOAuthResponse: ...


class HttpxA2AOAuthTransport:
    async def request_token(
        self,
        url: str,
        *,
        form: dict[str, str],
        timeout_seconds: float,
    ) -> A2AOAuthResponse:
        try:
            async with httpx.AsyncClient(
                timeout=timeout_seconds,
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    url,
                    data=form,
                    headers={"Accept": "application/json"},
                )
        except httpx.HTTPError as exc:
            raise SupplierA2AError("supplier_a2a_token_unavailable") from exc
        if len(response.content) > _MAX_TOKEN_BYTES:
            raise SupplierA2AError("supplier_a2a_token_response_too_large")
        return A2AOAuthResponse(response.status_code, response.content)


class A2AClientCredentialsTokenProvider:
    """Issue one short-lived token with exactly one collaboration Scope."""

    def __init__(
        self,
        *,
        token_url: str,
        client_id: str,
        client_secret: SecretValue,
        resource: str,
        scope: str,
        timeout_seconds: float,
        allow_plain_http: bool = False,
        transport: A2AOAuthTransport | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        _validate_url(token_url, allow_plain_http=allow_plain_http, local_host="keycloak")
        _validate_url(
            resource,
            allow_plain_http=allow_plain_http,
            local_host="supplier-agent",
        )
        if not client_id.strip() or not scope.strip() or any(item.isspace() for item in scope):
            raise ValueError("A2A client identity and one minimum Scope are required")
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._resource = resource
        self._scope = scope
        self._timeout_seconds = timeout_seconds
        self._transport = transport or HttpxA2AOAuthTransport()
        self._clock = clock
        self._cached: A2AAccessToken | None = None
        self._refresh_at = 0.0
        self._lock = asyncio.Lock()

    async def get(self) -> A2AAccessToken:
        async with self._lock:
            now = self._clock()
            if self._cached is not None and self._refresh_at > now:
                return self._cached
            response = await self._transport.request_token(
                self._token_url,
                form={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret.reveal(),
                    "scope": self._scope,
                    "resource": self._resource,
                },
                timeout_seconds=self._timeout_seconds,
            )
            token = self._decode(response)
            self._cached = token
            self._refresh_at = now + max(
                1,
                token.expires_in_seconds - _TOKEN_REFRESH_SKEW_SECONDS,
            )
            return token

    @property
    def scope(self) -> str:
        return self._scope

    def _decode(self, response: A2AOAuthResponse) -> A2AAccessToken:
        if response.status_code != 200:
            raise SupplierA2AError("supplier_a2a_token_rejected")
        try:
            payload = json.loads(response.body)
            access_token = payload["access_token"]
            token_type = payload["token_type"]
            expires_in = payload["expires_in"]
            scope = payload["scope"]
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
            raise SupplierA2AError("supplier_a2a_token_response_invalid") from exc
        if (
            not isinstance(access_token, str)
            or not access_token
            or not isinstance(token_type, str)
            or token_type.casefold() != "bearer"
            or not isinstance(expires_in, int)
            or isinstance(expires_in, bool)
            or expires_in <= _TOKEN_REFRESH_SKEW_SECONDS
            or expires_in > _MAX_TOKEN_LIFETIME_SECONDS
            or scope != self._scope
        ):
            raise SupplierA2AError("supplier_a2a_token_response_invalid")
        return A2AAccessToken(
            SecretValue(access_token, source=self._client_secret.source),
            self._scope,
            expires_in,
        )


class SupplierReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=255)
    kind: str = Field(pattern=r"^(MANUAL|SERVICE_BULLETIN|CASE|OTHER)$")


class SupplierAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = Field(pattern=r"^supplier-diagnosis-advice/v1$")
    recommendation: str = Field(min_length=1, max_length=4_000)
    confidence: float = Field(ge=0.0, le=1.0)
    required_information: list[str] = Field(default_factory=list, max_length=20)
    safety_notices: list[str] = Field(default_factory=list, max_length=20)
    references: list[SupplierReference] = Field(default_factory=list, max_length=20)

    @field_validator("required_information", "safety_notices")
    @classmethod
    def validate_bounded_strings(cls, values: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 1_000 for item in values):
            raise ValueError("supplier advice strings are invalid")
        return values


@dataclass(frozen=True, slots=True)
class A2ATaskOutcome:
    remote_task_id: str
    remote_context_id: str | None
    status: str
    advice: dict[str, Any] | None
    response_digest: str | None
    guardrail_policy_version: str | None


class SupplierAgentClient(Protocol):
    agent_name: str
    agent_version: str
    skill_id: str

    async def submit(self, payload: dict[str, Any]) -> A2ATaskOutcome: ...

    async def refresh(self, remote_task_id: str) -> A2ATaskOutcome: ...

    async def cancel(self, remote_task_id: str) -> A2ATaskOutcome: ...


class OfficialA2ASupplierAgentClient:
    """A2A 1.0 JSON-RPC client with pinned card identity and structured output."""

    def __init__(
        self,
        *,
        base_url: str,
        agent_name: str,
        agent_version: str,
        skill_id: str,
        remote_tenant: str,
        timeout_seconds: float,
        token_provider: A2AClientCredentialsTokenProvider,
        allow_plain_http: bool = False,
        guardrail: PromptInjectionGuard | None = None,
    ) -> None:
        _validate_url(
            base_url,
            allow_plain_http=allow_plain_http,
            local_host="supplier-agent",
        )
        self._base_url = base_url.rstrip("/")
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.skill_id = skill_id
        self._remote_tenant = remote_tenant
        self._timeout_seconds = timeout_seconds
        self._token_provider = token_provider
        self._guardrail = guardrail or PromptInjectionGuard()

    async def submit(self, payload: dict[str, Any]) -> A2ATaskOutcome:
        collaboration_id = payload.get("collaboration_id")
        if not isinstance(collaboration_id, str) or not collaboration_id:
            raise SupplierA2AError("supplier_a2a_collaboration_identity_required")
        if len(json.dumps(payload, ensure_ascii=False).encode()) > _MAX_MESSAGE_BYTES:
            raise SupplierA2AError("supplier_a2a_request_too_large")
        token = await self._token_provider.get()
        try:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token.value.reveal()}"},
                timeout=self._timeout_seconds,
                follow_redirects=False,
            ) as http_client:
                card = await A2ACardResolver(http_client, self._base_url).get_agent_card()
                self._validate_card(card)
                client = ClientFactory(
                    ClientConfig(
                        streaming=False,
                        polling=True,
                        httpx_client=http_client,
                        supported_protocol_bindings=[TransportProtocol.JSONRPC.value],
                        accepted_output_modes=["application/json"],
                    )
                ).create(card)
                async with client:
                    request = SendMessageRequest(
                        tenant=self._remote_tenant,
                        message=Message(
                            message_id=f"a2a-message-{collaboration_id}",
                            role=Role.ROLE_USER,
                            parts=[Part(data=ParseDict(payload, Value()))],
                        ),
                        configuration=SendMessageConfiguration(
                            accepted_output_modes=["application/json"],
                            return_immediately=True,
                        ),
                    )
                    responses = [item async for item in client.send_message(request)]
        except SupplierA2AError:
            raise
        except (httpx.TimeoutException, A2AClientTimeoutError) as exc:
            raise SupplierA2AError("supplier_a2a_timeout") from exc
        except (httpx.HTTPError, A2AClientError) as exc:
            raise SupplierA2AError("supplier_a2a_unavailable") from exc
        except A2AError as exc:
            raise SupplierA2AError("supplier_a2a_remote_request_rejected") from exc
        if len(responses) != 1 or not responses[0].HasField("task"):
            raise SupplierA2AError("supplier_a2a_task_response_required")
        return self._outcome(responses[0].task)

    async def refresh(self, remote_task_id: str) -> A2ATaskOutcome:
        return await self._task_command(remote_task_id, cancel=False)

    async def cancel(self, remote_task_id: str) -> A2ATaskOutcome:
        return await self._task_command(remote_task_id, cancel=True)

    async def _task_command(self, remote_task_id: str, *, cancel: bool) -> A2ATaskOutcome:
        token = await self._token_provider.get()
        try:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token.value.reveal()}"},
                timeout=self._timeout_seconds,
                follow_redirects=False,
            ) as http_client:
                card = await A2ACardResolver(http_client, self._base_url).get_agent_card()
                self._validate_card(card)
                client = ClientFactory(
                    ClientConfig(
                        streaming=False,
                        polling=True,
                        httpx_client=http_client,
                        supported_protocol_bindings=[TransportProtocol.JSONRPC.value],
                    )
                ).create(card)
                async with client:
                    task = (
                        await client.cancel_task(
                            CancelTaskRequest(tenant=self._remote_tenant, id=remote_task_id)
                        )
                        if cancel
                        else await client.get_task(
                            GetTaskRequest(
                                tenant=self._remote_tenant,
                                id=remote_task_id,
                                history_length=0,
                            )
                        )
                    )
        except SupplierA2AError:
            raise
        except (httpx.TimeoutException, A2AClientTimeoutError) as exc:
            raise SupplierA2AError("supplier_a2a_timeout") from exc
        except (httpx.HTTPError, A2AClientError) as exc:
            raise SupplierA2AError("supplier_a2a_unavailable") from exc
        except A2AError as exc:
            raise SupplierA2AError("supplier_a2a_remote_request_rejected") from exc
        return self._outcome(task)

    def _validate_card(self, card: AgentCard) -> None:
        if card.name != self.agent_name or card.version != self.agent_version:
            raise SupplierA2AError("supplier_a2a_agent_identity_mismatch")
        if sum(skill.id == self.skill_id for skill in card.skills) != 1:
            raise SupplierA2AError("supplier_a2a_skill_not_available")
        interfaces = [
            item
            for item in card.supported_interfaces
            if item.protocol_binding == TransportProtocol.JSONRPC.value
            and item.protocol_version == "1.0"
        ]
        if not interfaces or any(
            _origin(item.url) != _origin(self._base_url) for item in interfaces
        ):
            raise SupplierA2AError("supplier_a2a_interface_not_allowed")
        if not any(
            _supported_security_requirement(card, requirement, self._token_provider.scope)
            for requirement in card.security_requirements
        ):
            raise SupplierA2AError("supplier_a2a_enterprise_identity_required")

    def _outcome(self, task: Task) -> A2ATaskOutcome:
        if not task.id or len(task.id) > 255:
            raise SupplierA2AError("supplier_a2a_task_identity_invalid")
        if len(task.context_id) > 255:
            raise SupplierA2AError("supplier_a2a_context_identity_invalid")
        state = task.status.state
        if state == TaskState.TASK_STATE_UNSPECIFIED:
            raise SupplierA2AError("supplier_a2a_task_state_invalid")
        try:
            status = TaskState.Name(state).removeprefix("TASK_STATE_")
        except ValueError as exc:
            raise SupplierA2AError("supplier_a2a_task_state_invalid") from exc
        advice: dict[str, Any] | None = None
        digest: str | None = None
        guardrail_version: str | None = None
        if state == TaskState.TASK_STATE_COMPLETED:
            advice = _extract_advice(task.artifacts)
            decision = self._guardrail.inspect_output(advice)
            guardrail_version = decision.policy_version
            if decision.decision == "BLOCKED":
                raise SupplierA2AError("supplier_a2a_prompt_injection_blocked")
            digest = _digest(advice)
        elif task.artifacts:
            reason = (
                "supplier_a2a_terminal_artifact_rejected"
                if state in _TERMINAL_STATES
                else "supplier_a2a_noncompleted_artifact_rejected"
            )
            raise SupplierA2AError(reason)
        return A2ATaskOutcome(
            remote_task_id=task.id,
            remote_context_id=task.context_id or None,
            status=status,
            advice=advice,
            response_digest=digest,
            guardrail_policy_version=guardrail_version,
        )


def _extract_advice(artifacts: Any) -> dict[str, Any]:
    if len(artifacts) != 1:
        raise SupplierA2AError("supplier_a2a_single_artifact_required")
    artifact: Artifact = artifacts[0]
    if artifact.ByteSize() > _MAX_ARTIFACT_BYTES:
        raise SupplierA2AError("supplier_a2a_artifact_too_large")
    if len(artifact.parts) != 1 or artifact.parts[0].WhichOneof("content") != "data":
        raise SupplierA2AError("supplier_a2a_structured_artifact_required")
    raw = MessageToDict(
        artifact.parts[0].data,
        preserving_proto_field_name=True,
    )
    if not isinstance(raw, dict):
        raise SupplierA2AError("supplier_a2a_structured_artifact_required")
    try:
        return SupplierAdvice.model_validate(raw).model_dump(mode="json")
    except ValidationError as exc:
        raise SupplierA2AError("supplier_a2a_advice_contract_invalid") from exc


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{sha256(encoded.encode()).hexdigest()}"


def _validate_url(url: str, *, allow_plain_http: bool, local_host: str) -> None:
    parsed = urlparse(url)
    if (
        not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.query
    ):
        raise ValueError("A2A URL is invalid")
    if parsed.scheme == "https":
        return
    if (
        allow_plain_http
        and parsed.scheme == "http"
        and parsed.hostname
        in {
            "127.0.0.1",
            "localhost",
            local_host,
        }
    ):
        return
    raise ValueError("A2A URL must use HTTPS")


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlparse(url)
    return parsed.scheme, parsed.hostname or "", parsed.port


def _supported_security_requirement(
    card: AgentCard,
    requirement: SecurityRequirement,
    required_scope: str,
) -> bool:
    if len(requirement.schemes) != 1:
        return False
    scheme_name = next(iter(requirement.schemes))
    scheme = card.security_schemes.get(scheme_name)
    if scheme is None:
        return False
    kind = scheme.WhichOneof("scheme")
    scopes = tuple(requirement.schemes[scheme_name].list)
    if kind == "http_auth_security_scheme":
        return not scopes and scheme.http_auth_security_scheme.scheme.casefold() == "bearer"
    return kind in {"oauth2_security_scheme", "open_id_connect_security_scheme"} and (
        not scopes or required_scope in scopes
    )
