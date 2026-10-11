"""Adoption-bound projection of the project enterprise assurance receipt."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from industrial_ops_agent.enterprise_assets.service import EnterpriseProjectAdoptionReader
from industrial_ops_agent.simulation.project_assurance_lab import (
    ProjectAssuranceLabError,
    ProjectAssuranceLabReport,
    verify_project_assurance_lab,
)

ADOPTED_ROLE = "PROJECT_ENTERPRISE_ASSURANCE_ACCEPTANCE_VERIFIED"
ADOPTED_DOMAIN = "project_assurance_lab"


class EnterpriseProjectAssuranceGovernanceError(RuntimeError):
    """The adopted project assurance evidence cannot be projected safely."""


class EnterpriseProjectAssuranceGovernanceService:
    """Read and re-verify the assurance receipt selected by project adoption."""

    def __init__(
        self,
        repo_root: Path,
        adoption_reader: EnterpriseProjectAdoptionReader,
    ) -> None:
        self._repo_root = repo_root.resolve(strict=True)
        self._adoption_reader = adoption_reader

    def snapshot(self) -> ProjectAssuranceLabReport:
        adoption = self._adoption_reader.snapshot()
        assets = tuple(asset for asset in adoption.active_assets if asset.role == ADOPTED_ROLE)
        if len(assets) != 1:
            raise EnterpriseProjectAssuranceGovernanceError(
                "adopted_project_assurance_receipt_not_unique"
            )
        asset = assets[0]
        if (
            asset.domain != ADOPTED_DOMAIN
            or asset.runtime_eligible
            or asset.source_schema_version != "project-enterprise-assurance-lab/v1"
            or asset.source_status != "PROJECT_ENTERPRISE_ASSURANCE_CLOSED_LOOP_PASSED"
            or asset.source_evidence_chain_sha256 is None
        ):
            raise EnterpriseProjectAssuranceGovernanceError(
                "adopted_project_assurance_contract_invalid"
            )
        source = self._resolve_source(asset.source_path)
        if _sha256_file(source) != asset.source_file_sha256:
            raise EnterpriseProjectAssuranceGovernanceError(
                "adopted_project_assurance_digest_mismatch"
            )
        try:
            report = verify_project_assurance_lab(self._repo_root, source)
        except (ProjectAssuranceLabError, OSError, ValueError) as exc:
            raise EnterpriseProjectAssuranceGovernanceError(
                "adopted_project_assurance_verification_failed"
            ) from exc
        if (
            report.evidence_chain_sha256 != asset.source_evidence_chain_sha256
            or report.classification != asset.source_classification
            or not report.project_enterprise_use_authorized
            or not report.ready_for_project_enterprise_staging
            or report.external_enterprise_production_claim
        ):
            raise EnterpriseProjectAssuranceGovernanceError(
                "adopted_project_assurance_identity_mismatch"
            )
        return report

    def _resolve_source(self, source_path: str) -> Path:
        relative = Path(source_path)
        if relative.is_absolute():
            raise EnterpriseProjectAssuranceGovernanceError(
                "adopted_project_assurance_path_invalid"
            )
        try:
            source = (self._repo_root / relative).resolve(strict=True)
            source.relative_to(self._repo_root)
        except (OSError, ValueError) as exc:
            raise EnterpriseProjectAssuranceGovernanceError(
                "adopted_project_assurance_path_invalid"
            ) from exc
        if not source.is_file():
            raise EnterpriseProjectAssuranceGovernanceError(
                "adopted_project_assurance_path_invalid"
            )
        return source


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
