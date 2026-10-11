"""Resource-bound OAuth access-token verification for the supplier sandbox."""

from __future__ import annotations

from typing import Any, Protocol

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from industrial_ops_agent.auth.oidc import JwksSigningKeyProvider, SigningKeyProvider

_MAX_TOKEN_BYTES = 16 * 1024


class SupplierSandboxAuthenticationFailure(RuntimeError):
    """The caller did not present the configured supplier service identity."""


class SupplierSandboxTokenVerifier(Protocol):
    def verify(self, token: str) -> dict[str, Any]: ...


class OidcSupplierSandboxTokenVerifier:
    """Verify RS256 issuer, audience, client and one exact collaboration Scope."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        client_id: str,
        required_scope: str,
        signing_keys: SigningKeyProvider,
        leeway_seconds: int = 10,
    ) -> None:
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._client_id = client_id
        self._required_scope = required_scope
        self._signing_keys = signing_keys
        self._leeway_seconds = leeway_seconds

    @classmethod
    def from_jwks(
        cls,
        *,
        issuer: str,
        jwks_url: str,
        audience: str,
        client_id: str,
        required_scope: str,
        timeout_seconds: float,
    ) -> OidcSupplierSandboxTokenVerifier:
        return cls(
            issuer=issuer,
            audience=audience,
            client_id=client_id,
            required_scope=required_scope,
            signing_keys=JwksSigningKeyProvider(
                jwks_url,
                timeout_seconds=timeout_seconds,
            ),
        )

    def verify(self, token: str) -> dict[str, Any]:
        if not token or len(token.encode("utf-8")) > _MAX_TOKEN_BYTES:
            raise SupplierSandboxAuthenticationFailure("supplier_access_token_invalid")
        try:
            claims = jwt.decode(
                token,
                key=self._signing_keys.signing_key(token),
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway_seconds,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except ExpiredSignatureError as exc:
            raise SupplierSandboxAuthenticationFailure(
                "supplier_access_token_expired"
            ) from exc
        except (InvalidTokenError, ValueError, TypeError) as exc:
            raise SupplierSandboxAuthenticationFailure(
                "supplier_access_token_invalid"
            ) from exc
        scope = claims.get("scope")
        authorized_party = claims.get("azp")
        if (
            not isinstance(scope, str)
            or set(scope.split()) != {self._required_scope}
            or authorized_party != self._client_id
        ):
            raise SupplierSandboxAuthenticationFailure(
                "supplier_service_identity_scope_denied"
            )
        return dict(claims)


class StaticSupplierSandboxTokenVerifier:
    """Deterministic verifier used only by the loopback acceptance server."""

    def __init__(self, accepted_token: str) -> None:
        if not accepted_token:
            raise ValueError("accepted token is required")
        self._accepted_token = accepted_token

    def verify(self, token: str) -> dict[str, Any]:
        if token != self._accepted_token:
            raise SupplierSandboxAuthenticationFailure("supplier_access_token_invalid")
        return {"sub": "loopback-supplier-client", "scope": "supplier.diagnosis.review"}
