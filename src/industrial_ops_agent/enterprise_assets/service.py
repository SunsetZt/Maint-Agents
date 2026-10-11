"""Verified, cached access to the project-wide enterprise adoption receipt."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Protocol

from industrial_ops_agent.enterprise_assets.evidence import (
    collect_enterprise_runtime_evidence,
)
from industrial_ops_agent.enterprise_assets.models import EnterpriseRuntimeEvidence
from industrial_ops_agent.simulation.enterprise_project_adoption import (
    EnterpriseProjectAdoptionReport,
    verify_enterprise_project_adoption,
)


class EnterpriseProjectAdoptionReader(Protocol):
    def snapshot(self) -> EnterpriseProjectAdoptionReport: ...

    def runtime_evidence(self) -> tuple[EnterpriseRuntimeEvidence, ...]: ...


class EnterpriseProjectAdoptionService:
    """Verify the complete source chain and cache only the typed projection."""

    def __init__(
        self,
        repo_root: Path,
        *,
        acceptance_path: Path = Path("artifacts/enterprise-project-adoption/acceptance.json"),
        cache_seconds: float = 15.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if cache_seconds < 0 or cache_seconds > 300:
            raise ValueError("enterprise adoption cache must be between 0 and 300 seconds")
        self._repo_root = repo_root.resolve(strict=True)
        self._acceptance_path = acceptance_path
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._lock = Lock()
        self._cached: EnterpriseProjectAdoptionReport | None = None
        self._cache_expires_at = 0.0

    def snapshot(self) -> EnterpriseProjectAdoptionReport:
        now = self._clock()
        with self._lock:
            if self._cached is not None and now < self._cache_expires_at:
                return self._cached
            report = verify_enterprise_project_adoption(
                self._repo_root,
                self._acceptance_path,
            )
            self._cached = report
            self._cache_expires_at = now + self._cache_seconds
            return report

    def runtime_evidence(self) -> tuple[EnterpriseRuntimeEvidence, ...]:
        return collect_enterprise_runtime_evidence(self._repo_root, self.snapshot())
