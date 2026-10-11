"""Deterministic cache adapter used by domain and isolation tests."""

from __future__ import annotations

from industrial_ops_agent.cache.keys import TenantCacheKey
from industrial_ops_agent.persistence.tenant import TenantContext


class InMemoryTenantCache:
    def __init__(self) -> None:
        self._values: dict[str, bytes] = {}

    async def get(self, context: TenantContext, key: TenantCacheKey) -> bytes | None:
        key.assert_owned_by(context)
        return self._values.get(key.value)

    async def set(
        self,
        context: TenantContext,
        key: TenantCacheKey,
        value: bytes,
        *,
        ttl_seconds: int,
    ) -> None:
        key.assert_owned_by(context)
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        self._values[key.value] = value

    async def invalidate_namespace(
        self,
        context: TenantContext,
        namespace: str,
    ) -> int:
        prefix = f"ioap:v1:tenant:{context.tenant_id}:{namespace}:"
        keys = [key for key in self._values if key.startswith(prefix)]
        for key in keys:
            del self._values[key]
        return len(keys)
