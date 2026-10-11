"""Authenticated, schema-bound MCP adapter for enterprise read-only tools."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from time import monotonic
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
import httpx2
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, Tool

from industrial_ops_agent.secrets import SecretValue
from industrial_ops_agent.tools.contracts import (
    EnterpriseToolError,
    ToolTransportBinding,
)
from industrial_ops_agent.tools.enterprise_http import normalize_enterprise_readonly_payload
from industrial_ops_agent.tools.registry import ToolDefinition

_MAX_TOKEN_RESPONSE_BYTES = 16 * 1024
_MAX_TOKEN_LIFETIME_SECONDS = 15 * 60
_TOKEN_REFRESH_SKEW_SECONDS = 30

MCP_TOOL_SCOPES: Mapping[str, tuple[str, ...]] = {
    "asset.get": ("industrial.asset.read",),
    "warranty.get": ("industrial.warranty.read",),
    "parts.availability": ("industrial.parts.read",),
    "schedule.availability": ("industrial.schedule.read",),
    "work_orders.history": ("industrial.work-order.read",),
    "work_order.draft": ("industrial.work-order.read",),
}


@dataclass(frozen=True, slots=True, repr=False)
class McpAccessToken:
    value: SecretValue
    scopes: tuple[str, ...]
    expires_in_seconds: int


@dataclass(frozen=True, slots=True)
class McpOAuthResponse:
    status_code: int
    body: bytes


class McpOAuthTransport(Protocol):
    def request_token(
        self,
        url: str,
        *,
        form: dict[str, str],
        timeout_seconds: float,
    ) -> McpOAuthResponse: ...


class HttpxMcpOAuthTransport:
    """One-shot OAuth transport that never retains credentials or token bodies."""

    def request_token(
        self,
        url: str,
        *,
        form: dict[str, str],
        timeout_seconds: float,
    ) -> McpOAuthResponse:
        try:
            with httpx.Client(
                timeout=timeout_seconds,
                follow_redirects=False,
            ) as client:
                response = client.post(
                    url,
                    data=form,
                    headers={"Accept": "application/json"},
                )
        except httpx.HTTPError as exc:
            raise EnterpriseToolError("enterprise_mcp_token_unavailable") from exc
        if len(response.content) > _MAX_TOKEN_RESPONSE_BYTES:
            raise EnterpriseToolError("enterprise_mcp_token_response_too_large")
        return McpOAuthResponse(response.status_code, response.content)


@dataclass(slots=True)
class _CachedToken:
    token: McpAccessToken
    refresh_at: float


class ClientCredentialsMcpTokenProvider:
    """Acquire resource-bound, per-tool short-lived OAuth access tokens."""

    def __init__(
        self,
        *,
        token_url: str,
        client_id: str,
        client_secret: SecretValue,
        resource: str,
        allowed_scopes: tuple[str, ...],
        timeout_seconds: float,
        allow_plain_http: bool = False,
        transport: McpOAuthTransport | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        _validate_remote_url(token_url, allow_plain_http=allow_plain_http, local_host="keycloak")
        _validate_remote_url(
            resource,
            allow_plain_http=allow_plain_http,
            local_host="enterprise-mcp-server",
        )
        if not client_id.strip():
            raise ValueError("MCP OAuth client ID is required")
        normalized_scopes = tuple(sorted(set(allowed_scopes)))
        if not normalized_scopes or any(not item.strip() for item in normalized_scopes):
            raise ValueError("MCP OAuth allowed scopes are invalid")
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._resource = resource
        self._allowed_scopes = frozenset(normalized_scopes)
        self._timeout_seconds = timeout_seconds
        self._transport = transport or HttpxMcpOAuthTransport()
        self._clock = clock
        self._cache: dict[tuple[str, ...], _CachedToken] = {}
        self._lock = Lock()

    def get(self, required_scopes: tuple[str, ...]) -> McpAccessToken:
        requested = tuple(sorted(set(required_scopes)))
        if not requested or not set(requested) <= self._allowed_scopes:
            raise EnterpriseToolError("enterprise_mcp_scope_not_allowed")
        with self._lock:
            cached = self._cache.get(requested)
            now = self._clock()
            if cached is not None and cached.refresh_at > now:
                return cached.token
            token = self._request(requested)
            self._cache[requested] = _CachedToken(
                token=token,
                refresh_at=now
                + max(1, token.expires_in_seconds - _TOKEN_REFRESH_SKEW_SECONDS),
            )
            return token

    def _request(self, scopes: tuple[str, ...]) -> McpAccessToken:
        response = self._transport.request_token(
            self._token_url,
            form={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret.reveal(),
                "scope": " ".join(scopes),
                "resource": self._resource,
            },
            timeout_seconds=self._timeout_seconds,
        )
        if response.status_code != 200:
            raise EnterpriseToolError("enterprise_mcp_token_rejected")
        try:
            payload = json.loads(response.body)
            access_token = payload["access_token"]
            token_type = payload["token_type"]
            expires_in = payload["expires_in"]
            granted_scopes = payload["scope"]
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
            raise EnterpriseToolError("enterprise_mcp_token_response_invalid") from exc
        if (
            not isinstance(access_token, str)
            or not access_token
            or not isinstance(token_type, str)
            or token_type.casefold() != "bearer"
            or not isinstance(expires_in, int)
            or isinstance(expires_in, bool)
            or expires_in <= _TOKEN_REFRESH_SKEW_SECONDS
            or expires_in > _MAX_TOKEN_LIFETIME_SECONDS
            or not isinstance(granted_scopes, str)
            or tuple(sorted(set(granted_scopes.split()))) != scopes
        ):
            raise EnterpriseToolError("enterprise_mcp_token_response_invalid")
        return McpAccessToken(
            value=SecretValue(access_token, source=self._client_secret.source),
            scopes=scopes,
            expires_in_seconds=expires_in,
        )


class McpInvocationTransport(Protocol):
    def invoke(
        self,
        *,
        binding: ToolTransportBinding,
        parameters: dict[str, Any],
        token: McpAccessToken,
    ) -> dict[str, Any]: ...


class SdkMcpInvocationTransport:
    """MCP SDK v2 Streamable HTTP boundary with a fresh authenticated session."""

    def __init__(
        self,
        server_url: str,
        *,
        timeout_seconds: float,
        allow_plain_http: bool = False,
    ) -> None:
        _validate_remote_url(
            server_url,
            allow_plain_http=allow_plain_http,
            local_host="enterprise-mcp-server",
        )
        self._server_url = server_url
        self._timeout_seconds = timeout_seconds

    def invoke(
        self,
        *,
        binding: ToolTransportBinding,
        parameters: dict[str, Any],
        token: McpAccessToken,
    ) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(self._invoke(binding, parameters, token))
            except EnterpriseToolError:
                raise
            except Exception as exc:
                raise EnterpriseToolError("enterprise_mcp_unavailable") from exc
        raise EnterpriseToolError("enterprise_mcp_sync_boundary_required")

    async def _invoke(
        self,
        binding: ToolTransportBinding,
        parameters: dict[str, Any],
        token: McpAccessToken,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {token.value.reveal()}",
            "Accept": "application/json, text/event-stream",
        }
        async with httpx2.AsyncClient(
            headers=headers,
            timeout=self._timeout_seconds,
            follow_redirects=False,
        ) as http_client:
            transport = streamable_http_client(
                self._server_url,
                http_client=http_client,
            )
            async with Client(
                transport,
                read_timeout_seconds=self._timeout_seconds,
                cache=None,
            ) as client:
                tools: list[Tool] = []
                cursor: str | None = None
                for _page in range(20):
                    catalog = await client.list_tools(cursor=cursor)
                    tools.extend(catalog.tools)
                    cursor = catalog.next_cursor
                    if cursor is None:
                        break
                else:
                    raise EnterpriseToolError("enterprise_mcp_catalog_too_large")
                _require_bound_tool(tools, binding)
                result = await client.call_tool(
                    binding.remote_tool_name,
                    parameters,
                    read_timeout_seconds=self._timeout_seconds,
                )
        return _validated_structured_result(result)


class McpReadonlyAdapter:
    """Expose only locally registered, schema-pinned read tools over MCP."""

    def __init__(
        self,
        *,
        server_id: str,
        server_version: str,
        definitions: tuple[ToolDefinition, ...],
        token_provider: ClientCredentialsMcpTokenProvider,
        transport: McpInvocationTransport,
        allowed_scopes: tuple[str, ...],
    ) -> None:
        definitions_by_id = {
            item.tool_id: item
            for item in definitions
            if item.risk_tier in {"T0", "T1"} and item.tool_id in MCP_TOOL_SCOPES
        }
        if set(definitions_by_id) != set(MCP_TOOL_SCOPES):
            raise ValueError("MCP bindings do not cover the read-only Tool Registry")
        required_scopes = {scope for scopes in MCP_TOOL_SCOPES.values() for scope in scopes}
        if set(allowed_scopes) != required_scopes:
            raise ValueError("MCP scopes must exactly match the minimum registry scope set")
        self._bindings = {
            tool_id: ToolTransportBinding(
                transport="MCP_STREAMABLE_HTTP",
                local_tool_id=tool_id,
                remote_tool_name=tool_id,
                server_id=server_id,
                server_version=server_version,
                required_scopes=MCP_TOOL_SCOPES[tool_id],
                input_schema_digest=_schema_digest(definition.input_schema),
            )
            for tool_id, definition in definitions_by_id.items()
        }
        self._token_provider = token_provider
        self._transport = transport

    def invoke(self, tool_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            binding = self._bindings[tool_id]
        except KeyError as exc:
            raise EnterpriseToolError("enterprise_mcp_tool_not_allowed") from exc
        token = self._token_provider.get(binding.required_scopes)
        payload = self._transport.invoke(
            binding=binding,
            parameters=parameters,
            token=token,
        )
        return normalize_enterprise_readonly_payload(tool_id, payload)

    def transport_binding(self, tool_id: str) -> ToolTransportBinding:
        try:
            return self._bindings[tool_id]
        except KeyError as exc:
            raise EnterpriseToolError("enterprise_mcp_tool_not_allowed") from exc


def _schema_digest(schema: dict[str, Any]) -> str:
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"


def _require_bound_tool(
    tools: list[Tool],
    binding: ToolTransportBinding,
) -> None:
    matches = [tool for tool in tools if tool.name == binding.remote_tool_name]
    if len(matches) != 1:
        raise EnterpriseToolError("enterprise_mcp_tool_not_discovered")
    if binding.input_schema_digest is None or _schema_digest(
        matches[0].input_schema
    ) != binding.input_schema_digest:
        raise EnterpriseToolError("enterprise_mcp_tool_schema_mismatch")


def _validated_structured_result(result: CallToolResult) -> dict[str, Any]:
    if result.is_error:
        raise EnterpriseToolError("enterprise_mcp_tool_failed")
    # Content blocks are model-oriented and may carry prompt injection, URLs,
    # media or resources. This enterprise adapter accepts structured facts only.
    if result.content:
        raise EnterpriseToolError("enterprise_mcp_untrusted_content_rejected")
    if not isinstance(result.structured_content, dict):
        raise EnterpriseToolError("enterprise_mcp_structured_result_required")
    return result.structured_content


def _validate_remote_url(url: str, *, allow_plain_http: bool, local_host: str) -> None:
    parsed = urlparse(url)
    if (
        not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.query
    ):
        raise ValueError("MCP remote URL is invalid")
    if parsed.scheme == "https":
        return
    if allow_plain_http and parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        local_host,
    }:
        return
    raise ValueError("MCP remote URL must use HTTPS")
