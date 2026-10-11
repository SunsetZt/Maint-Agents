"""Ed25519 JWS boundary for server-issued FIELD edge packages."""

from __future__ import annotations

from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from industrial_ops_agent.edge.contracts import (
    FIELD_EDGE_AUDIENCE,
    FIELD_EDGE_ISSUER,
    FieldEdgeDiagnosisPackClaims,
)


class FieldEdgeSignatureError(ValueError):
    """The package is unsigned, malformed, expired or signed by another key."""


class Ed25519PackSigner:
    """Sign and verify only the closed FIELD-008 claim set with one named key."""

    def __init__(self, private_key: Ed25519PrivateKey, *, key_id: str) -> None:
        if not key_id or len(key_id) > 128:
            raise ValueError("field_edge_signing_key_id_invalid")
        self._private_key = private_key
        self._public_key = private_key.public_key()
        self.key_id = key_id

    @classmethod
    def from_private_pem(cls, value: str, *, key_id: str) -> Ed25519PackSigner:
        try:
            loaded = serialization.load_pem_private_key(value.encode(), password=None)
        except (TypeError, ValueError) as exc:
            raise ValueError("field_edge_signing_private_key_invalid") from exc
        if not isinstance(loaded, Ed25519PrivateKey):
            raise ValueError("field_edge_signing_private_key_must_be_ed25519")
        return cls(loaded, key_id=key_id)

    def sign(self, claims: FieldEdgeDiagnosisPackClaims) -> str:
        return str(
            jwt.encode(
                claims.model_dump(mode="json"),
                self._private_key,
                algorithm="EdDSA",
                headers={"alg": "EdDSA", "kid": self.key_id, "typ": "JWT"},
            )
        )

    def verify(self, token: str) -> FieldEdgeDiagnosisPackClaims:
        return _verify(token, self._public_key, self.key_id)

    def public_pem(self) -> str:
        return self._public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()


def load_ed25519_public_pem(value: str) -> Ed25519PublicKey:
    """Load the explicit deployment trust anchor used later by the offline CLI."""

    try:
        loaded = serialization.load_pem_public_key(value.encode())
    except (TypeError, ValueError) as exc:
        raise ValueError("field_edge_trusted_public_key_invalid") from exc
    if not isinstance(loaded, Ed25519PublicKey):
        raise ValueError("field_edge_trusted_public_key_must_be_ed25519")
    return loaded


class Ed25519PackVerifier:
    """Offline verifier bound to one deployment-provided key ID and public key."""

    def __init__(self, public_key: Ed25519PublicKey, *, key_id: str) -> None:
        if not key_id or len(key_id) > 128:
            raise ValueError("field_edge_trusted_key_id_invalid")
        self._public_key = public_key
        self.key_id = key_id

    @classmethod
    def from_public_pem(cls, value: str, *, key_id: str) -> Ed25519PackVerifier:
        return cls(load_ed25519_public_pem(value), key_id=key_id)

    def verify(self, token: str) -> FieldEdgeDiagnosisPackClaims:
        return _verify(token, self._public_key, self.key_id)


def _verify(
    token: str,
    public_key: Ed25519PublicKey,
    key_id: str,
) -> FieldEdgeDiagnosisPackClaims:
    try:
        header = jwt.get_unverified_header(token)
        if header != {"alg": "EdDSA", "kid": key_id, "typ": "JWT"}:
            raise FieldEdgeSignatureError("field_edge_jws_header_invalid")
        payload: dict[str, Any] = jwt.decode(
            token,
            public_key,
            algorithms=["EdDSA"],
            audience=FIELD_EDGE_AUDIENCE,
            issuer=FIELD_EDGE_ISSUER,
            options={
                "require": ["aud", "exp", "iat", "iss", "jti", "nbf", "sub"],
                "verify_aud": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_nbf": True,
                "verify_signature": True,
            },
        )
        return FieldEdgeDiagnosisPackClaims.model_validate(payload)
    except FieldEdgeSignatureError:
        raise
    except (jwt.PyJWTError, TypeError, ValueError) as exc:
        raise FieldEdgeSignatureError("field_edge_jws_invalid") from exc
