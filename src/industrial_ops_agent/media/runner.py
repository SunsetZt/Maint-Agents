"""Runnable M1 scan-worker loop with tenant-scoped claims and fail-closed startup."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from hashlib import sha256
from pathlib import Path

import structlog
from minio import Minio
from sqlalchemy import select, text

from industrial_ops_agent.config import get_settings
from industrial_ops_agent.media.models import MediaStateConflict
from industrial_ops_agent.media.worker import ClamAvScanner, MediaScanWorker
from industrial_ops_agent.object_store.minio import MinioTenantObjectStore
from industrial_ops_agent.observability import configure_logging
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.errors import InfrastructureUnavailable
from industrial_ops_agent.persistence.models import MediaObjectRecord, TenantRecord
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.secrets import SecretName, build_secret_provider

LOGGER = structlog.get_logger("m1-media-worker")


def _tenant_label(context: TenantContext) -> str:
    return sha256(context.tenant_id.encode()).hexdigest()[:16]


async def _dependency_probe(
    database: Database,
    minio_client: Minio,
    bucket: str,
    clamav_host: str,
    clamav_port: int,
) -> None:
    def sync_probe() -> None:
        with database.engine.connect() as connection:
            if connection.scalar(text("SELECT 1")) != 1:
                raise InfrastructureUnavailable("database")
        if not minio_client.bucket_exists(bucket):
            raise InfrastructureUnavailable("object_store")

    await asyncio.to_thread(sync_probe)
    _reader, writer = await asyncio.open_connection(clamav_host, clamav_port)
    writer.close()
    await writer.wait_closed()


def _pending(database: Database) -> list[tuple[TenantContext, str]]:
    with database.engine.connect() as connection:
        tenant_ids = list(connection.scalars(select(TenantRecord.id)))
    pending: list[tuple[TenantContext, str]] = []
    for tenant_id in tenant_ids:
        context = TenantContext(tenant_id=tenant_id, subject_id="m1-media-worker")
        with database.transaction(context) as session:
            media_ids = session.scalars(
                select(MediaObjectRecord.media_id)
                .where(MediaObjectRecord.scan_state == "PENDING")
                .order_by(MediaObjectRecord.created_at)
                .limit(25)
            )
            pending.extend((context, media_id) for media_id in media_ids)
    return pending


async def run() -> None:
    settings = get_settings()
    configure_logging(
        settings.log_level,
        service_name=settings.service_name,
        log_file=settings.log_file,
    )
    provider = build_secret_provider(settings)
    database_url = provider.get(SecretName.DATABASE_URL)
    minio_access_key = provider.get(SecretName.MINIO_ACCESS_KEY)
    minio_secret_key = provider.get(SecretName.MINIO_SECRET_KEY)
    database = Database(database_url.reveal())
    minio_client = Minio(
        settings.minio_endpoint,
        access_key=minio_access_key.reveal(),
        secret_key=minio_secret_key.reveal(),
        secure=settings.minio_secure,
    )
    store = MinioTenantObjectStore(minio_client, settings.minio_bucket)
    scanner = ClamAvScanner(settings.clamav_host, settings.clamav_port)
    worker = MediaScanWorker(database, store, scanner)
    try:
        await _dependency_probe(
            database,
            minio_client,
            settings.minio_bucket,
            settings.clamav_host,
            settings.clamav_port,
        )
        ready_file = Path(settings.worker_ready_file)
        ready_file.touch(mode=0o600)
        LOGGER.info("worker_ready", service="m1-media-worker")
        while True:
            for context, media_id in await asyncio.to_thread(_pending, database):
                try:
                    await worker.process(context, media_id)
                except MediaStateConflict:
                    LOGGER.info(
                        "media_claim_lost",
                        service="m1-media-worker",
                        tenant_hash=_tenant_label(context),
                    )
                except InfrastructureUnavailable:
                    LOGGER.error(
                        "media_dependency_unavailable",
                        service="m1-media-worker",
                        tenant_hash=_tenant_label(context),
                    )
            await asyncio.sleep(settings.worker_poll_seconds)
    finally:
        database.dispose()


def main() -> None:
    with suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    main()
