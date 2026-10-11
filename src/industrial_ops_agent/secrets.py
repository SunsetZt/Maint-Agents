"""Typed, redacted runtime-secret providers for environment and Vault KV v2."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Protocol, Self
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


class SecretName(StrEnum):
    DATABASE_URL = "database_url"
    OIDC_CLIENT_SECRET = "oidc_client_secret"
    MINIO_ACCESS_KEY = "minio_access_key"
    MINIO_SECRET_KEY = "minio_secret_key"
    NEXTAUTH_SECRET = "nextauth_secret"
    VAULT_TOKEN = "vault_token"
    LABEL_STUDIO_API_TOKEN = "label_studio_api_token"
    MODEL_GATEWAY_API_KEY = "model_gateway_api_key"
    EDGE_PACK_SIGNING_PRIVATE_KEY = "edge_pack_signing_private_key"
    REALTIME_TURN_SHARED_SECRET = "realtime_turn_shared_secret"
    ENTERPRISE_TOOL_GATEWAY_TOKEN = "enterprise_tool_gateway_token"
    ENTERPRISE_MCP_CLIENT_SECRET = "enterprise_mcp_client_secret"
    NOTIFICATION_PROVIDER_API_TOKEN = "notification_provider_api_token"
    NOTIFICATION_WEBHOOK_SECRET = "notification_webhook_secret"
    PROCUREMENT_PROVIDER_API_TOKEN = "procurement_provider_api_token"
    PROCUREMENT_WEBHOOK_SECRET = "procurement_webhook_secret"
    REFUND_PROVIDER_API_TOKEN = "refund_provider_api_token"
    REFUND_WEBHOOK_SECRET = "refund_webhook_secret"
    FSM_ASSIGNMENT_PROVIDER_API_TOKEN = "fsm_assignment_provider_api_token"
    FSM_ASSIGNMENT_WEBHOOK_SECRET = "fsm_assignment_webhook_secret"
    SERVICE_QUOTATION_PROVIDER_API_TOKEN = "service_quotation_provider_api_token"
    SUPPLIER_A2A_CLIENT_SECRET = "supplier_a2a_client_secret"
    NEO4J_PASSWORD = "neo4j_password"
    OPENSEARCH_PASSWORD = "opensearch_password"
    TAVILY_API_KEY = "tavily_api_key"


class SecretSource(StrEnum):
    ENVIRONMENT = "environment"
    VAULT = "vault"
    FILE = "file"


@dataclass(frozen=True, repr=False)
class SecretValue:
    """A value that is revealed only at an explicit adapter boundary."""

    _value: str
    source: SecretSource

    def __post_init__(self) -> None:
        if not self._value:
            raise ValueError("secret value must not be empty")

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"SecretValue(<redacted:{self.source.value}>)"

    def __str__(self) -> str:
        return f"<redacted:{self.source.value}>"


class SecretReadError(RuntimeError):
    """A stable diagnostic that never stores the rejected secret value."""

    def __init__(
        self,
        name: SecretName,
        source: SecretSource,
        category: str,
        *,
        locator: str | None = None,
    ) -> None:
        self.name = name
        self.source = source
        self.category = category
        self.locator = locator
        suffix = f" locator={locator}" if locator else ""
        super().__init__(
            f"secret read failed: name={name.value} source={source.value} "
            f"category={category}{suffix}"
        )


class SecretProvider(Protocol):
    def get(self, name: SecretName) -> SecretValue: ...

    def refresh(self) -> None: ...


ENVIRONMENT_VARIABLES: Mapping[SecretName, str] = {
    SecretName.DATABASE_URL: "IOAP_DATABASE_URL",
    SecretName.OIDC_CLIENT_SECRET: "IOAP_OIDC_CLIENT_SECRET",
    SecretName.MINIO_ACCESS_KEY: "IOAP_MINIO_ACCESS_KEY",
    SecretName.MINIO_SECRET_KEY: "IOAP_MINIO_SECRET_KEY",
    SecretName.NEXTAUTH_SECRET: "IOAP_NEXTAUTH_SECRET",
    SecretName.VAULT_TOKEN: "VAULT_TOKEN",
    SecretName.LABEL_STUDIO_API_TOKEN: "IOAP_LABEL_STUDIO_API_TOKEN",
    SecretName.MODEL_GATEWAY_API_KEY: "IOAP_MODEL_GATEWAY_API_KEY",
    SecretName.EDGE_PACK_SIGNING_PRIVATE_KEY: "IOAP_EDGE_PACK_SIGNING_PRIVATE_KEY",
    SecretName.REALTIME_TURN_SHARED_SECRET: "IOAP_REALTIME_TURN_SHARED_SECRET",
    SecretName.ENTERPRISE_TOOL_GATEWAY_TOKEN: "IOAP_ENTERPRISE_TOOL_GATEWAY_TOKEN",
    SecretName.ENTERPRISE_MCP_CLIENT_SECRET: "IOAP_ENTERPRISE_MCP_CLIENT_SECRET",
    SecretName.NOTIFICATION_PROVIDER_API_TOKEN: "IOAP_NOTIFICATION_PROVIDER_API_TOKEN",
    SecretName.NOTIFICATION_WEBHOOK_SECRET: "IOAP_NOTIFICATION_WEBHOOK_SECRET",
    SecretName.PROCUREMENT_PROVIDER_API_TOKEN: "IOAP_PROCUREMENT_PROVIDER_API_TOKEN",
    SecretName.PROCUREMENT_WEBHOOK_SECRET: "IOAP_PROCUREMENT_WEBHOOK_SECRET",
    SecretName.REFUND_PROVIDER_API_TOKEN: "IOAP_REFUND_PROVIDER_API_TOKEN",
    SecretName.REFUND_WEBHOOK_SECRET: "IOAP_REFUND_WEBHOOK_SECRET",
    SecretName.FSM_ASSIGNMENT_PROVIDER_API_TOKEN: (
        "IOAP_FSM_ASSIGNMENT_PROVIDER_API_TOKEN"
    ),
    SecretName.FSM_ASSIGNMENT_WEBHOOK_SECRET: "IOAP_FSM_ASSIGNMENT_WEBHOOK_SECRET",
    SecretName.SERVICE_QUOTATION_PROVIDER_API_TOKEN: (
        "IOAP_SERVICE_QUOTATION_PROVIDER_API_TOKEN"
    ),
    SecretName.SUPPLIER_A2A_CLIENT_SECRET: "IOAP_SUPPLIER_A2A_CLIENT_SECRET",
    SecretName.NEO4J_PASSWORD: "IOAP_NEO4J_PASSWORD",
    SecretName.OPENSEARCH_PASSWORD: "IOAP_OPENSEARCH_PASSWORD",
    SecretName.TAVILY_API_KEY: "IOAP_TAVILY_API_KEY",
}


class EnvironmentSecretProvider:
    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def get(self, name: SecretName) -> SecretValue:
        variable = ENVIRONMENT_VARIABLES[name]
        value = self._environment.get(variable)
        if not value:
            raise SecretReadError(
                name,
                SecretSource.ENVIRONMENT,
                "missing",
                locator=variable,
            )
        return SecretValue(value, source=SecretSource.ENVIRONMENT)

    def refresh(self) -> None:
        """Environment values are read on every request; no cache is retained."""


class VaultTransport(Protocol):
    def read_kv_v2(
        self,
        *,
        address: str,
        mount: str,
        path: str,
        token: SecretValue,
    ) -> Mapping[str, str]: ...


class UrllibVaultTransport:
    """Small Vault HTTP boundary with no SDK types leaking into business code."""

    def __init__(self, *, timeout_seconds: float = 3.0) -> None:
        self._timeout_seconds = timeout_seconds

    def read_kv_v2(
        self,
        *,
        address: str,
        mount: str,
        path: str,
        token: SecretValue,
    ) -> Mapping[str, str]:
        endpoint = self._endpoint(address, mount, path)
        request = Request(  # noqa: S310 - scheme and local exception are validated below
            endpoint,
            headers={
                "Accept": "application/json",
                "X-Vault-Token": token.reveal(),
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                payload: Any = json.load(response)
        except HTTPError as exc:
            category = "unauthorized" if exc.code in {401, 403} else "unavailable"
            raise SecretReadError(
                SecretName.VAULT_TOKEN,
                SecretSource.VAULT,
                category,
                locator=f"{mount}/{path}",
            ) from exc
        except (URLError, TimeoutError, OSError, JSONDecodeError) as exc:
            raise SecretReadError(
                SecretName.VAULT_TOKEN,
                SecretSource.VAULT,
                "unavailable",
                locator=f"{mount}/{path}",
            ) from exc
        try:
            values = payload["data"]["data"]
        except (KeyError, TypeError) as exc:
            raise SecretReadError(
                SecretName.VAULT_TOKEN,
                SecretSource.VAULT,
                "invalid_response",
                locator=f"{mount}/{path}",
            ) from exc
        if not isinstance(values, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in values.items()
        ):
            raise SecretReadError(
                SecretName.VAULT_TOKEN,
                SecretSource.VAULT,
                "invalid_response",
                locator=f"{mount}/{path}",
            )
        return values

    @staticmethod
    def _endpoint(address: str, mount: str, path: str) -> str:
        parsed = urlparse(address)
        local_hosts = {"127.0.0.1", "localhost", "vault"}
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in local_hosts
        ):
            raise ValueError("Vault address must use HTTPS outside the local Lite network")
        if not parsed.netloc or not mount or not path:
            raise ValueError("Vault address, mount, and path are required")
        safe_mount = quote(mount.strip("/"), safe="")
        safe_path = "/".join(quote(part, safe="") for part in path.strip("/").split("/"))
        return f"{address.rstrip('/')}/v1/{safe_mount}/data/{safe_path}"


class VaultSecretProvider:
    def __init__(
        self,
        *,
        address: str,
        mount: str,
        path: str,
        token: SecretValue,
        transport: VaultTransport | None = None,
        token_loader: Callable[[], SecretValue] | None = None,
    ) -> None:
        self._address = address
        self._mount = mount
        self._path = path
        self._token = token
        self._transport = transport or UrllibVaultTransport()
        self._token_loader = token_loader
        self._cache: dict[SecretName, SecretValue] = {}

    def get(self, name: SecretName) -> SecretValue:
        if name is SecretName.VAULT_TOKEN:
            raise SecretReadError(name, SecretSource.VAULT, "not_exportable")
        if name not in self._cache:
            values = self._read_values()
            for candidate in SecretName:
                if candidate is SecretName.VAULT_TOKEN:
                    continue
                raw = values.get(candidate.value)
                if raw:
                    self._cache[candidate] = SecretValue(raw, source=SecretSource.VAULT)
        try:
            return self._cache[name]
        except KeyError as exc:
            raise SecretReadError(
                name,
                SecretSource.VAULT,
                "missing",
                locator=f"{self._mount}/{self._path}",
            ) from exc

    def refresh(self) -> None:
        self._cache.clear()
        if self._token_loader is not None:
            self._token = self._token_loader()

    def _read_values(self) -> Mapping[str, str]:
        try:
            return self._transport.read_kv_v2(
                address=self._address,
                mount=self._mount,
                path=self._path,
                token=self._token,
            )
        except SecretReadError as error:
            token_loader = self._token_loader
            if error.category != "unauthorized" or token_loader is None:
                raise
        self._token = token_loader()
        return self._transport.read_kv_v2(
            address=self._address,
            mount=self._mount,
            path=self._path,
            token=self._token,
        )

    def __repr__(self) -> str:
        return (
            "VaultSecretProvider("
            f"address={self._address!r}, mount={self._mount!r}, path={self._path!r}, "
            "token=<redacted>)"
        )


@dataclass(frozen=True, repr=False)
class RuntimeSecrets:
    database_url: SecretValue
    oidc_client_secret: SecretValue
    minio_access_key: SecretValue
    minio_secret_key: SecretValue
    nextauth_secret: SecretValue

    @classmethod
    def load(cls, provider: SecretProvider) -> Self:
        return cls(
            database_url=provider.get(SecretName.DATABASE_URL),
            oidc_client_secret=provider.get(SecretName.OIDC_CLIENT_SECRET),
            minio_access_key=provider.get(SecretName.MINIO_ACCESS_KEY),
            minio_secret_key=provider.get(SecretName.MINIO_SECRET_KEY),
            nextauth_secret=provider.get(SecretName.NEXTAUTH_SECRET),
        )

    def redacted(self) -> dict[str, str]:
        return {
            name: f"<redacted:{value.source.value}>"
            for name, value in sorted(self.__dict__.items())
        }

    def __repr__(self) -> str:
        return f"RuntimeSecrets({self.redacted()!r})"


def secret_from_file(path: Path) -> SecretValue:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SecretReadError(
            SecretName.VAULT_TOKEN,
            SecretSource.FILE,
            "unavailable",
            locator=str(path),
        ) from exc
    if not value:
        raise SecretReadError(
            SecretName.VAULT_TOKEN,
            SecretSource.FILE,
            "missing",
            locator=str(path),
        )
    return SecretValue(value, source=SecretSource.FILE)


def build_secret_provider(settings: Any) -> SecretProvider:
    """Build the selected provider without placing credential fields on Settings."""

    from industrial_ops_agent.config import SecretBackend

    if settings.secret_backend is SecretBackend.ENVIRONMENT:
        return EnvironmentSecretProvider()
    def token_loader() -> SecretValue:
        return secret_from_file(settings.vault_token_file)

    token = token_loader()
    return VaultSecretProvider(
        address=settings.vault_address,
        mount=settings.vault_mount,
        path=settings.vault_path,
        token=token,
        token_loader=token_loader,
    )
