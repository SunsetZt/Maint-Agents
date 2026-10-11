"""Deterministic object-store adapter for isolation and media tests."""

from __future__ import annotations

from industrial_ops_agent.object_store.keys import TenantObjectKey
from industrial_ops_agent.persistence.tenant import TenantContext


class InMemoryTenantObjectStore:
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    async def put(
        self, context: TenantContext, key: TenantObjectKey, content: bytes
    ) -> None:
        key.assert_owned_by(context)
        self._objects[key.value] = content

    async def get(self, context: TenantContext, key: TenantObjectKey) -> bytes | None:
        key.assert_owned_by(context)
        return self._objects.get(key.value)

    async def delete(self, context: TenantContext, key: TenantObjectKey) -> None:
        key.assert_owned_by(context)
        self._objects.pop(key.value, None)
