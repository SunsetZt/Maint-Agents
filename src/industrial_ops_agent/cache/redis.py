"""Redis implementation that never accepts an unscoped raw key."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import partial

from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from industrial_ops_agent.cache.keys import TenantCacheKey
from industrial_ops_agent.persistence.errors import InfrastructureUnavailable
from industrial_ops_agent.persistence.tenant import TenantContext, validate_boundary_identifier


class RedisTenantCache:
    def __init__(self, client: Redis, *, attempts: int = 2) -> None:
        self._client = client
        self._attempts = attempts

    @classmethod
    def from_url(cls, url: str, *, timeout_seconds: float = 1.0) -> RedisTenantCache:
        client = Redis.from_url(
            url,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
            decode_responses=False,
        )
        return cls(client)

    async def _run(self, operation: Callable[[], Awaitable[object]]) -> object:
        for attempt in range(1, self._attempts + 1):
            try:
                return await operation()
            except (RedisConnectionError, RedisTimeoutError) as exc:
                if attempt == self._attempts:
                    raise InfrastructureUnavailable("cache") from exc
        raise AssertionError("bounded retry loop did not terminate")

    async def get(self, context: TenantContext, key: TenantCacheKey) -> bytes | None:
        key.assert_owned_by(context)
        value = await self._run(lambda: self._client.get(key.value))
        return value if isinstance(value, bytes) else None

    async def set(
        self,
        context: TenantContext,
        key: TenantCacheKey,
        value: bytes,
        *,
        ttl_seconds: int,
    ) -> None:
        key.assert_owned_by(context)
        await self._run(lambda: self._client.set(key.value, value, ex=ttl_seconds))

    async def invalidate_namespace(
        self,
        context: TenantContext,
        namespace: str,
    ) -> int:
        """Delete a bounded tenant namespace without accepting an arbitrary Redis pattern."""

        safe_namespace = validate_boundary_identifier(namespace, field="cache namespace")
        pattern = f"ioap:v1:tenant:{context.tenant_id}:{safe_namespace}:*"
        cursor = 0
        deleted = 0
        while True:
            result = await self._run(partial(self._scan_page, cursor, pattern))
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("cache scan returned an invalid payload")
            next_cursor, keys = result
            owned_prefix = f"ioap:v1:tenant:{context.tenant_id}:{safe_namespace}:"
            owned_keys = [
                key
                for key in keys
                if isinstance(key, bytes)
                and key.decode("utf-8", errors="strict").startswith(owned_prefix)
            ]
            if owned_keys:
                count = await self._run(partial(self._delete_keys, tuple(owned_keys)))
                if not isinstance(count, int):
                    raise TypeError("cache delete returned an invalid payload")
                deleted += count
            cursor = int(next_cursor)
            if cursor == 0:
                return deleted

    async def _scan_page(self, cursor: int, pattern: str) -> object:
        return await self._client.scan(cursor=cursor, match=pattern, count=200)

    async def _delete_keys(self, keys: tuple[bytes, ...]) -> object:
        return await self._client.delete(*keys)

    async def ping(self) -> bool:
        return bool(await self._run(self._client.ping))

    async def close(self) -> None:
        await self._client.aclose()
