"""Tenant-scoped correlation between adopted model evidence and live release records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.enterprise_assets.importer import (
    active_import_for_source,
    enterprise_model_asset_import_projection,
)
from industrial_ops_agent.enterprise_assets.models import (
    EnterpriseAliasBinding,
    EnterpriseDeploymentBinding,
    EnterpriseReleaseAutomationBinding,
    EnterpriseReleaseRuntimeRegistration,
    EnterpriseRuntimeBinding,
    EnterpriseRuntimeEvidence,
)
from industrial_ops_agent.enterprise_assets.service import EnterpriseProjectAdoptionReader
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    EnterpriseModelAssetImportRecord,
    EnterpriseModelReleaseOnboardingRecord,
    ModelAliasRecord,
    ModelDeploymentRecord,
    ModelReleaseObservationRecord,
    ModelReleaseRecord,
)


class EnterpriseRuntimeBindingService:
    """Build a read-only join without changing Release or deployment state."""

    def __init__(
        self,
        database: Database,
        adoption_reader: EnterpriseProjectAdoptionReader,
    ) -> None:
        self._database = database
        self._adoption_reader = adoption_reader

    def snapshot(self, identity: IdentityContext) -> tuple[EnterpriseRuntimeBinding, ...]:
        evidence_items = self._adoption_reader.runtime_evidence()
        with self._database.transaction(identity.tenant_context) as session:
            releases = list(
                session.scalars(
                    select(ModelReleaseRecord)
                    .where(ModelReleaseRecord.tenant_id == identity.tenant_id)
                    .order_by(ModelReleaseRecord.created_at.desc())
                )
            )
            deployments = list(
                session.scalars(
                    select(ModelDeploymentRecord).where(
                        ModelDeploymentRecord.tenant_id == identity.tenant_id
                    )
                )
            )
            observations = list(
                session.scalars(
                    select(ModelReleaseObservationRecord)
                    .where(ModelReleaseObservationRecord.tenant_id == identity.tenant_id)
                    .order_by(
                        ModelReleaseObservationRecord.release_id,
                        ModelReleaseObservationRecord.sequence,
                    )
                )
            )
            aliases = list(
                session.scalars(
                    select(ModelAliasRecord)
                    .where(ModelAliasRecord.tenant_id == identity.tenant_id)
                    .order_by(ModelAliasRecord.alias)
                )
            )
            onboardings = list(
                session.scalars(
                    select(EnterpriseModelReleaseOnboardingRecord).where(
                        EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id
                    )
                )
            )
            model_imports = {
                evidence.component: active_import_for_source(
                    session,
                    identity.tenant_id,
                    evidence.component,
                    evidence.candidate_experiment_id,
                )
                for evidence in evidence_items
            }

        deployment_by_release = {item.release_id: item for item in deployments}
        latest_observation_by_release = {item.release_id: item for item in observations}
        aliases_by_release: dict[str, list[ModelAliasRecord]] = defaultdict(list)
        for alias in aliases:
            aliases_by_release[alias.active_release_id].append(alias)
        onboarding_by_release = {item.release_id: item for item in onboardings}

        return tuple(
            _binding(
                evidence,
                releases,
                deployment_by_release,
                latest_observation_by_release,
                aliases_by_release,
                onboarding_by_release,
                model_imports.get(evidence.component),
            )
            for evidence in evidence_items
        )


def _binding(
    evidence: EnterpriseRuntimeEvidence,
    releases: list[ModelReleaseRecord],
    deployment_by_release: dict[str, ModelDeploymentRecord],
    latest_observation_by_release: dict[str, ModelReleaseObservationRecord],
    aliases_by_release: dict[str, list[ModelAliasRecord]],
    onboarding_by_release: dict[str, EnterpriseModelReleaseOnboardingRecord],
    model_import: EnterpriseModelAssetImportRecord | None,
) -> EnterpriseRuntimeBinding:
    runtime_candidate_id = (
        model_import.imported_candidate_experiment_id
        if model_import is not None
        else evidence.candidate_experiment_id
    )
    matched_releases = [
        item
        for item in releases
        if _release_matches(item, evidence, runtime_candidate_id=runtime_candidate_id)
    ]
    registrations = tuple(
        _registration(
            release,
            deployment_by_release.get(release.release_id),
            latest_observation_by_release.get(release.release_id),
            aliases_by_release.get(release.release_id, []),
            onboarding_by_release.get(release.release_id),
        )
        for release in matched_releases
    )
    has_deployment = any(item.deployment is not None for item in registrations)
    has_active_alias = any(
        alias.status == "ACTIVE" for registration in registrations for alias in registration.aliases
    )
    return EnterpriseRuntimeBinding(
        evidence=evidence,
        import_state="IMPORTED" if model_import is not None else "NOT_IMPORTED",
        model_import=(
            enterprise_model_asset_import_projection(model_import)
            if model_import is not None
            else None
        ),
        registry_state="REGISTERED" if registrations else "NOT_REGISTERED",
        deployment_state="DEPLOYMENT_RECORDED" if has_deployment else "NOT_DEPLOYED",
        alias_state="ACTIVE_ALIAS_BOUND" if has_active_alias else "NO_ACTIVE_ALIAS",
        registrations=registrations,
    )


def _registration(
    release: ModelReleaseRecord,
    deployment: ModelDeploymentRecord | None,
    observation: ModelReleaseObservationRecord | None,
    aliases: list[ModelAliasRecord],
    onboarding: EnterpriseModelReleaseOnboardingRecord | None,
) -> EnterpriseReleaseRuntimeRegistration:
    deployment_projection = (
        EnterpriseDeploymentBinding(
            deployment_id=deployment.deployment_id,
            provider=deployment.provider,
            status=deployment.status,
            desired_stage=deployment.desired_stage,
            current_stage=deployment.current_stage,
            desired_traffic_percent=deployment.desired_traffic_percent,
            observed_traffic_percent=deployment.observed_traffic_percent,
            failure_reason=deployment.failure_reason,
            version=deployment.version,
            latest_observation_stage=observation.stage if observation is not None else None,
            latest_observation_decision=(observation.decision if observation is not None else None),
        )
        if deployment is not None
        else None
    )
    return EnterpriseReleaseRuntimeRegistration(
        release_id=release.release_id,
        status=release.status,
        target_environment=release.target_environment,
        manifest_hash=release.manifest_hash,
        rollback_release_id=release.rollback_release_id,
        traffic_percent=release.traffic_percent,
        failure_reason=release.failure_reason,
        version=release.version,
        created_at=release.created_at,
        deployment=deployment_projection,
        aliases=tuple(
            EnterpriseAliasBinding(
                alias=item.alias,
                status=item.status,
                runtime_profile=item.runtime_profile,
                version=item.version,
            )
            for item in aliases
        ),
        automation=(
            EnterpriseReleaseAutomationBinding(
                onboarding_id=onboarding.onboarding_id,
                baseline_release_id=onboarding.baseline_release_id,
                auto_shadow_enabled=onboarding.auto_shadow_enabled,
                automation_status=onboarding.automation_status,
                attempt_count=onboarding.attempt_count,
                last_error=onboarding.last_error,
                last_attempt_at=onboarding.last_attempt_at,
                version=onboarding.version,
            )
            if onboarding is not None
            else None
        ),
    )


def _release_matches(
    release: ModelReleaseRecord,
    evidence: EnterpriseRuntimeEvidence,
    *,
    runtime_candidate_id: str,
) -> bool:
    if evidence.component == "LLM":
        return release.candidate_experiment_id == runtime_candidate_id
    component = _specialized_component(
        release.manifest_json,
        evidence.component.lower(),
    )
    return component is not None and component.get("runtime_model_id") == runtime_candidate_id


def _specialized_component(
    manifest: dict[str, Any],
    component_name: str,
) -> dict[str, Any] | None:
    components = manifest.get("specialized_components")
    if not isinstance(components, dict):
        return None
    component = components.get(component_name)
    if not isinstance(component, dict):
        return None
    return component
