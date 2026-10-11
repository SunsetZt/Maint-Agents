"""Only this module may construct Redis tenant keys."""

from __future__ import annotations

from dataclasses import dataclass

from industrial_ops_agent.persistence.errors import TenantBoundaryViolation
from industrial_ops_agent.persistence.tenant import TenantContext, validate_boundary_identifier


@dataclass(frozen=True, slots=True)
class TenantCacheKey:
    tenant_id: str
    value: str

    def assert_owned_by(self, context: TenantContext) -> None:
        if self.tenant_id != context.tenant_id:
            raise TenantBoundaryViolation("cache key is outside the active tenant")


def tenant_cache_key(
    context: TenantContext, namespace: str, resource_id: str
) -> TenantCacheKey:
    namespace = validate_boundary_identifier(namespace, field="cache namespace")
    resource_id = validate_boundary_identifier(resource_id, field="cache resource_id")
    return TenantCacheKey(
        tenant_id=context.tenant_id,
        value=f"ioap:v1:tenant:{context.tenant_id}:{namespace}:{resource_id}",
    )
