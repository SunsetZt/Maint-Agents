"""Durable workflow and Activity boundaries for M2 business processes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from industrial_ops_agent.orchestration.activities import (
        RecognitionActivity,
        RecognitionActivityInput,
        RecognitionDispatcher,
        RecognitionDispatchUnavailable,
    )

__all__ = [
    "RecognitionActivity",
    "RecognitionActivityInput",
    "RecognitionDispatcher",
    "RecognitionDispatchUnavailable",
]


def __getattr__(name: str) -> Any:
    """Load Activity exports lazily so Temporal can import workflow modules safely."""

    if name in __all__:
        from industrial_ops_agent.orchestration import activities

        return getattr(activities, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
