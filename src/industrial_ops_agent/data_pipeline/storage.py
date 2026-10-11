"""Immutable dataset artifact storage for local files and MinIO."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from minio import Minio
from minio.error import S3Error


class ImmutableObjectExists(RuntimeError):
    pass


class DatasetStore(Protocol):
    def put_bytes(self, object_key: str, content: bytes) -> None: ...

    def get_bytes(self, object_key: str) -> bytes: ...


def _safe_key(object_key: str) -> str:
    path = Path(object_key)
    if path.is_absolute() or not object_key or ".." in path.parts:
        raise ValueError("dataset object key is unsafe")
    return path.as_posix()


class FilesystemDatasetStore:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def put_bytes(self, object_key: str, content: bytes) -> None:
        target = self._root / _safe_key(object_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ImmutableObjectExists(object_key)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        temporary.write_bytes(content)
        temporary.replace(target)

    def get_bytes(self, object_key: str) -> bytes:
        return (self._root / _safe_key(object_key)).read_bytes()


class MinioDatasetStore:
    def __init__(self, client: Minio, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def put_bytes(self, object_key: str, content: bytes) -> None:
        key = _safe_key(object_key)
        try:
            self._client.stat_object(self._bucket, key)
        except S3Error as exc:
            if exc.code not in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
                raise
        else:
            raise ImmutableObjectExists(key)
        self._client.put_object(
            self._bucket,
            key,
            BytesIO(content),
            length=len(content),
            content_type="application/octet-stream",
        )

    def get_bytes(self, object_key: str) -> bytes:
        response = self._client.get_object(self._bucket, _safe_key(object_key))
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
