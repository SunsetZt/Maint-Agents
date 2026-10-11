"""Adoption-bound projection of project closure and remaining production gaps."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from industrial_ops_agent.enterprise_assets.service import EnterpriseProjectAdoptionReader

if TYPE_CHECKING:
    from industrial_ops_agent.simulation.closure import SimulatedEnterpriseClosureReport

ADOPTED_ROLE = "ENTERPRISE_PROJECT_EVIDENCE_ROOT"
ADOPTED_DOMAIN = "enterprise_project_closure"


class EnterpriseProjectClosureGovernanceError(RuntimeError):
    """The adopted project closure evidence cannot be projected safely."""


class EnterpriseProjectClosureGovernanceService:
    """Read and re-verify the closure receipt selected by project adoption."""

    def __init__(
        self,
        repo_root: Path,
        adoption_reader: EnterpriseProjectAdoptionReader,
    ) -> None:
        self._repo_root = repo_root.resolve(strict=True)
        self._adoption_reader = adoption_reader

    def snapshot(self) -> SimulatedEnterpriseClosureReport:
        # Import lazily so simulation.closure can import the acceptance verifier without
        # re-entering this projection through enterprise_assets.__init__.
        from industrial_ops_agent.simulation.closure import SimulatedEnterpriseClosureReport

        adoption = self._adoption_reader.snapshot()
        assets = tuple(asset for asset in adoption.active_assets if asset.role == ADOPTED_ROLE)
        if len(assets) != 1:
            raise EnterpriseProjectClosureGovernanceError(
                "adopted_enterprise_project_closure_not_unique"
            )
        asset = assets[0]
        if (
            asset.domain != ADOPTED_DOMAIN
            or asset.runtime_eligible
            or asset.source_schema_version != "simulated-enterprise-data-model-closure/v1"
            or asset.source_status != "SIMULATED_ENTERPRISE_DATA_MODEL_CLOSURE_PASSED"
            or asset.source_evidence_chain_sha256 is None
        ):
            raise EnterpriseProjectClosureGovernanceError(
                "adopted_enterprise_project_closure_contract_invalid"
            )

        source = self._resolve_source(asset.source_path)
        if _sha256_file(source) != asset.source_file_sha256:
            raise EnterpriseProjectClosureGovernanceError(
                "adopted_enterprise_project_closure_digest_mismatch"
            )
        try:
            report = SimulatedEnterpriseClosureReport.model_validate_json(
                source.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise EnterpriseProjectClosureGovernanceError(
                "adopted_enterprise_project_closure_invalid"
            ) from exc
        unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
        if (
            report.evidence_chain_sha256 != _digest(unsigned)
            or report.evidence_chain_sha256 != asset.source_evidence_chain_sha256
            or report.classification != asset.source_classification
            or set(report.coverage) != {evidence.domain for evidence in report.source_evidence}
            or len(report.source_evidence)
            != len({evidence.domain for evidence in report.source_evidence})
        ):
            raise EnterpriseProjectClosureGovernanceError(
                "adopted_enterprise_project_closure_identity_mismatch"
            )
        for evidence in report.source_evidence:
            evidence_path = self._resolve_source(evidence.source_path)
            if _sha256_file(evidence_path) != evidence.file_sha256:
                raise EnterpriseProjectClosureGovernanceError(
                    f"enterprise_project_closure_source_digest_mismatch:{evidence.domain}"
                )
        return report

    def _resolve_source(self, source_path: str) -> Path:
        relative = Path(source_path)
        if relative.is_absolute():
            raise EnterpriseProjectClosureGovernanceError(
                "adopted_enterprise_project_closure_path_invalid"
            )
        try:
            source = (self._repo_root / relative).resolve(strict=True)
            source.relative_to(self._repo_root)
        except (OSError, ValueError) as exc:
            raise EnterpriseProjectClosureGovernanceError(
                "adopted_enterprise_project_closure_path_invalid"
            ) from exc
        if not source.is_file():
            raise EnterpriseProjectClosureGovernanceError(
                "adopted_enterprise_project_closure_path_invalid"
            )
        return source


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
