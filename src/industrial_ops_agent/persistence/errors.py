"""Stable infrastructure and tenant-boundary failures."""

from __future__ import annotations

from collections.abc import Callable
from time import sleep
from typing import TypeVar

T = TypeVar("T")


class TenantBoundaryViolation(PermissionError):
    """An infrastructure key belongs to a different tenant."""


class InfrastructureUnavailable(RuntimeError):
    """A named dependency failed after its bounded retry budget."""

    def __init__(self, dependency: str) -> None:
        self.dependency = dependency
        super().__init__(f"{dependency} temporarily unavailable")


def execute_with_retry(
    *,
    dependency: str,
    operation: Callable[[], T],
    attempts: int,
    retryable: tuple[type[Exception], ...],
    delay_seconds: float = 0,
) -> T:
    """Run an infrastructure call with a small explicit retry budget."""

    if attempts < 1:
        raise ValueError("attempts must be at least one")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except retryable as exc:
            if attempt == attempts:
                raise InfrastructureUnavailable(dependency) from exc
            if delay_seconds > 0:
                sleep(delay_seconds)
    raise AssertionError("bounded retry loop did not terminate")
