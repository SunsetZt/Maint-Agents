"""OIDC access-token verification independent of browser session state."""

from __future__ import annotations

from typing import Any, Protocol

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError, PyJWKClient

from industrial_ops_agent.auth.errors import AuthenticationFailure
from industrial_ops_agent.auth.identity import IdentityContext


class SigningKeyProvider(Protocol):
    def signing_key(self, token: str) -> Any:
        """Resolve an approved verification key without returning token contents."""


class StaticSigningKeyProvider:
    """Injected key provider for deterministic tests and offline verification."""

    def __init__(self, key: Any) -> None:
        self._key = key

    def signing_key(self, token: str) -> Any:
        del token
        return self._key


class JwksSigningKeyProvider:
    """Cached standard JWKS resolver; no Keycloak-specific API is required."""

    def __init__(self, jwks_url: str, *, timeout_seconds: float = 2.0) -> None:
        self._client = PyJWKClient(
            jwks_url,
            cache_keys=True,
            cache_jwk_set=True,
            timeout=timeout_seconds,
        )

    def signing_key(self, token: str) -> Any:
        return self._client.get_signing_key_from_jwt(token).key


class OidcVerifier:
    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        signing_keys: SigningKeyProvider,
        algorithms: tuple[str, ...] = ("RS256",),
        leeway_seconds: int = 10,
    ) -> None:
        if not issuer.startswith("https://") and not issuer.startswith("http://localhost"):
            raise ValueError("OIDC issuer must use HTTPS outside localhost")
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._signing_keys = signing_keys
        self._algorithms = algorithms
        self._leeway_seconds = leeway_seconds

    def verify(self, token: str) -> IdentityContext:
        if not token or len(token) > 16_384:
            raise AuthenticationFailure("token_invalid")
        try:
            key = self._signing_keys.signing_key(token)
            claims = jwt.decode(
                token,
                key=key,
                algorithms=list(self._algorithms),
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway_seconds,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except ExpiredSignatureError as exc:
            raise AuthenticationFailure("token_expired") from exc
        except (InvalidTokenError, ValueError, TypeError) as exc:
            raise AuthenticationFailure("token_invalid") from exc
        try:
            return IdentityContext.from_claims(claims)
        except (ValueError, TypeError) as exc:
            raise AuthenticationFailure("claims_invalid") from exc
