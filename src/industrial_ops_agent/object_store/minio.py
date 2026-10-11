"""MinIO implementation guarded by canonical tenant object keys."""

from __future__ import annotations

from asyncio import to_thread
from collections.abc import Callable
from io import BytesIO

from minio import Minio
from minio.error import MinioException, S3Error

from industrial_ops_agent.object_store.keys import TenantObjectKey
from industrial_ops_agent.persistence.errors import InfrastructureUnavailable
from industrial_ops_agent.persistence.tenant import TenantContext


class MinioTenantObjectStore:
    def __init__(self, client: Minio, bucket: str, *, attempts: int = 2) -> None:
        self._client = client
        self._bucket = bucket
        self._attempts = attempts

    async def put(
        self, context: TenantContext, key: TenantObjectKey, content: bytes
    ) -> None:
        key.assert_owned_by(context)

        def operation() -> None:
            self._client.put_object(
                self._bucket,
                key.value,
                BytesIO(content),
                length=len(content),
                content_type="application/octet-stream",
            )

        await self._run(operation)

    async def get(self, context: TenantContext, key: TenantObjectKey) -> bytes | None:
        key.assert_owned_by(context)

        def operation() -> bytes | None:
            try:
                response = self._client.get_object(self._bucket, key.value)
            except S3Error as exc:
                if exc.code in {"NoSuchKey", "NoSuchObject"}:
                    return None
                raise
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        result = await self._run(operation)
        if result is not None and not isinstance(result, bytes):
            raise TypeError("object-store read returned an invalid payload")
        return result

    async def delete(self, context: TenantContext, key: TenantObjectKey) -> None:
        key.assert_owned_by(context)
        await self._run(lambda: self._client.remove_object(self._bucket, key.value))

    async def _run(self, operation: Callable[[], object]) -> object:
        for attempt in range(1, self._attempts + 1):
            try:
                return await to_thread(operation)
            except (MinioException, OSError) as exc:
                if attempt == self._attempts:
                    raise InfrastructureUnavailable("object_store") from exc
        raise AssertionError("bounded retry loop did not terminate")
