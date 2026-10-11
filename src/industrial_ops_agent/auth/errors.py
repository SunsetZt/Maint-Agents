"""Safe authentication and authorization failures."""

from __future__ import annotations


class AuthenticationFailure(Exception):
    """Internal authentication classification with a constant external message."""

    def __init__(self, reason_code: str, message: str = "Access token is invalid") -> None:
        self.reason_code = reason_code
        super().__init__(message)


class AuthorizationDenied(PermissionError):
    """A deny that deliberately does not reveal resource existence."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__("Resource not found or not visible")
