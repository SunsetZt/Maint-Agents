"""Explicit diagnosis state, version manifest, budget, and report contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentBudget:
    max_steps: int = 8
    max_tool_calls: int = 5
    max_tokens: int = 6000
    max_cost_usd: float = 1.0
    timeout_seconds: int = 180

    def as_dict(self) -> dict[str, int | float]:
        return {
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class DiagnosisManifest:
    incident_version: int
    evidence_bundle_id: str
    evidence_version: int
    model_release: str
    prompt_bundle: str
    index_release_id: str
    agent_graph_version: str
    tool_versions: dict[str, str]
    budget: AgentBudget
    created_at: datetime
    prompt_bundle_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_version": self.incident_version,
            "evidence_bundle_id": self.evidence_bundle_id,
            "evidence_version": self.evidence_version,
            "model_release": self.model_release,
            "prompt_bundle": self.prompt_bundle,
            "index_release_id": self.index_release_id,
            "agent_graph_version": self.agent_graph_version,
            "tool_versions": dict(sorted(self.tool_versions.items())),
            "budget": self.budget.as_dict(),
            "created_at": self.created_at.isoformat(),
            **({"prompt_bundle_hash": self.prompt_bundle_hash} if self.prompt_bundle_hash else {}),
        }


@dataclass(frozen=True, slots=True)
class DiagnosisReport:
    status: str
    conclusion: str | None
    citations: tuple[dict[str, Any], ...]
    contradictions: tuple[str, ...]
    missing_information: tuple[str, ...]
    next_checks: tuple[str, ...]
    confidence: float
    stop_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "conclusion": self.conclusion,
            "citations": [dict(item) for item in self.citations],
            "contradictions": list(self.contradictions),
            "missing_information": list(self.missing_information),
            "next_checks": list(self.next_checks),
            "confidence": self.confidence,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True, slots=True)
class DiagnosisRun:
    diagnosis_run_id: str
    incident_id: str
    evidence_bundle_id: str
    agent_run_id: str
    workflow_id: str
    status: str
    manifest: dict[str, Any]
    report: dict[str, Any] | None
    stop_reason: str | None
    version: int
    created_at: datetime
    updated_at: datetime
