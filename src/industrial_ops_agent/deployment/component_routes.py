"""Read and validate an active production component route inside its caller transaction."""

from __future__ import annotations

from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.domain.json import strict_document_digest as _digest_json
from industrial_ops_agent.persistence.models import (
    ModelAliasRecord,
    ModelDeploymentRecord,
    ModelReleaseRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.supply_chain.service import require_media_deployment_supply_chain


class ComponentRouteErrorFactory(Protocol):
    def __call__(self, reason: str, *, release_id: str | None = None) -> Exception: ...


def resolve_production_component_route(
    session: Session,
    context: TenantContext,
    *,
    alias_name: str,
    error_factory: ComponentRouteErrorFactory,
) -> tuple[ModelAliasRecord, ModelReleaseRecord]:
    alias = session.scalar(
        select(ModelAliasRecord).where(
            ModelAliasRecord.tenant_id == context.tenant_id,
            ModelAliasRecord.alias == alias_name,
            ModelAliasRecord.status == "ACTIVE",
        )
    )
    if alias is None:
        raise error_factory("production_release_unavailable")
    release = session.scalar(
        select(ModelReleaseRecord).where(
            ModelReleaseRecord.tenant_id == context.tenant_id,
            ModelReleaseRecord.release_id == alias.active_release_id,
            ModelReleaseRecord.status == "PRODUCTION",
        )
    )
    deployment = session.scalar(
        select(ModelDeploymentRecord).where(
            ModelDeploymentRecord.tenant_id == context.tenant_id,
            ModelDeploymentRecord.deployment_id == alias.deployment_id,
            ModelDeploymentRecord.status == "READY",
        )
    )
    deployment_release = (
        session.scalar(
            select(ModelReleaseRecord).where(
                ModelReleaseRecord.tenant_id == context.tenant_id,
                ModelReleaseRecord.release_id == deployment.release_id,
            )
        )
        if deployment is not None
        else None
    )
    route_matches_release = bool(
        deployment is not None
        and (
            (
                deployment.current_stage == "PRODUCTION"
                and deployment.desired_stage == "PRODUCTION"
                and deployment.observed_traffic_percent == 100.0
                and deployment.desired_traffic_percent == 100.0
                and deployment.release_id == alias.active_release_id
            )
            or (
                deployment.current_stage == "ROLLED_BACK"
                and deployment.desired_stage == "ROLLED_BACK"
                and deployment.observed_traffic_percent == 0.0
                and deployment.desired_traffic_percent == 0.0
                and deployment.release_id != alias.active_release_id
                and deployment_release is not None
                and deployment_release.rollback_release_id == alias.active_release_id
            )
        )
    )
    if (
        release is None
        or deployment is None
        or deployment_release is None
        or not route_matches_release
        or deployment_release.target_environment != release.target_environment
        or _digest_json(deployment_release.manifest_json) != deployment_release.manifest_hash
        or deployment.desired_spec_json.get(
            "manifest_hash", deployment.desired_spec_json.get("release_manifest_hash")
        )
        != deployment_release.manifest_hash
        or deployment.applied_spec_hash != deployment.desired_spec_hash
        or _digest_json(deployment.desired_spec_json) != deployment.desired_spec_hash
        or not alias.endpoint_url.startswith(("http://", "https://"))
        or alias.runtime_profile != release.manifest_json.get("runtime", {}).get("profile_id")
        or alias.manifest_hash != release.manifest_hash
        or alias.endpoint_url != deployment.endpoint_url
        or _digest_json(release.manifest_json) != release.manifest_hash
    ):
        raise error_factory(
            "production_release_route_inconsistent",
            release_id=alias.active_release_id,
        )
    try:
        # Import after package initialization: releases and deployment expose eager facades.
        from industrial_ops_agent.releases.staging_smoke import (
            require_publishable_staging_admission,
        )

        require_publishable_staging_admission(
            session, release, deployment_provider=deployment.provider
        )
        require_media_deployment_supply_chain(
            session, release, deployment, allow_exact_recovery=True
        )
    except ValueError as exc:
        raise error_factory(str(exc), release_id=release.release_id) from exc
    return alias, release
