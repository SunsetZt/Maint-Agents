"""Adoption-bound projection of the enterprise model import acceptance receipt."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from industrial_ops_agent.enterprise_assets.acceptance import (
    COMPONENTS,
    EnterpriseModelImportAcceptanceError,
    EnterpriseModelImportAcceptanceReport,
    verify_enterprise_model_import_acceptance,
)
from industrial_ops_agent.enterprise_assets.service import EnterpriseProjectAdoptionReader

ADOPTED_ROLE = "ENTERPRISE_MODEL_IMPORT_PREFLIGHT_ACCEPTANCE_VERIFIED"
ADOPTED_DOMAIN = "enterprise_model_import_preflight"


class EnterpriseModelImportAcceptanceGovernanceError(RuntimeError):
    """The adopted import acceptance evidence cannot be projected safely."""


class EnterpriseModelImportAcceptanceGovernanceService:
    """Read the receipt selected by the verified project adoption manifest."""

    def __init__(
        self,
        repo_root: Path,
        adoption_reader: EnterpriseProjectAdoptionReader,
    ) -> None:
        self._repo_root = repo_root.resolve(strict=True)
        self._adoption_reader = adoption_reader

    def snapshot(self) -> EnterpriseModelImportAcceptanceReport:
        adoption = self._adoption_reader.snapshot()
        assets = tuple(asset for asset in adoption.active_assets if asset.role == ADOPTED_ROLE)
        if len(assets) != 1:
            raise EnterpriseModelImportAcceptanceGovernanceError(
                "adopted_enterprise_model_import_acceptance_not_unique"
            )
        asset = assets[0]
        if (
            asset.domain != ADOPTED_DOMAIN
            or asset.runtime_eligible
            or asset.source_schema_version != "enterprise-model-import-acceptance/v1"
            or asset.source_status != "ENTERPRISE_MODEL_IMPORT_PREFLIGHT_PASSED"
            or asset.source_evidence_chain_sha256 is None
        ):
            raise EnterpriseModelImportAcceptanceGovernanceError(
                "adopted_enterprise_model_import_acceptance_contract_invalid"
            )

        source = self._resolve_source(asset.source_path)
        if _sha256_file(source) != asset.source_file_sha256:
            raise EnterpriseModelImportAcceptanceGovernanceError(
                "adopted_enterprise_model_import_acceptance_digest_mismatch"
            )
        try:
            report = verify_enterprise_model_import_acceptance(source)
        except (EnterpriseModelImportAcceptanceError, OSError, ValueError) as exc:
            raise EnterpriseModelImportAcceptanceGovernanceError(
                "adopted_enterprise_model_import_acceptance_verification_failed"
            ) from exc
        if (
            report.execution_mode != "EPHEMERAL_LOOPBACK_STAGING"
            or report.evidence_chain_sha256 != asset.source_evidence_chain_sha256
            or tuple(item.component for item in report.components) != COMPONENTS
        ):
            raise EnterpriseModelImportAcceptanceGovernanceError(
                "adopted_enterprise_model_import_acceptance_identity_mismatch"
            )
        common_enterprise_model_import_baselines(report)
        return report

    def _resolve_source(self, source_path: str) -> Path:
        relative = Path(source_path)
        if relative.is_absolute():
            raise EnterpriseModelImportAcceptanceGovernanceError(
                "adopted_enterprise_model_import_acceptance_path_invalid"
            )
        try:
            source = (self._repo_root / relative).resolve(strict=True)
            source.relative_to(self._repo_root)
        except (OSError, ValueError) as exc:
            raise EnterpriseModelImportAcceptanceGovernanceError(
                "adopted_enterprise_model_import_acceptance_path_invalid"
            ) from exc
        if not source.is_file():
            raise EnterpriseModelImportAcceptanceGovernanceError(
                "adopted_enterprise_model_import_acceptance_path_invalid"
            )
        return source


def common_enterprise_model_import_baselines(
    report: EnterpriseModelImportAcceptanceReport,
) -> tuple[str, ...]:
    """Return baselines compatible with every accepted component."""

    baseline_sets = [
        set(component.compatible_baseline_release_ids) for component in report.components
    ]
    common = set.intersection(*baseline_sets) if baseline_sets else set()
    if not common:
        raise EnterpriseModelImportAcceptanceGovernanceError(
            "enterprise_model_import_acceptance_common_baseline_missing"
        )
    return tuple(sorted(common))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
