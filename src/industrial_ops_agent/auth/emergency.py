"""Request-local selection of an explicitly supplied emergency access grant."""

from __future__ import annotations

from contextvars import ContextVar, Token

_emergency_grant_id: ContextVar[str | None] = ContextVar(
    "emergency_grant_id",
    default=None,
)


def bind_emergency_grant_id(grant_id: str | None) -> Token[str | None]:
    """Bind the untrusted grant selector; the policy resolver validates it later."""

    return _emergency_grant_id.set(grant_id)


def current_emergency_grant_id() -> str | None:
    return _emergency_grant_id.get()


def reset_emergency_grant_id(token: Token[str | None]) -> None:
    _emergency_grant_id.reset(token)
