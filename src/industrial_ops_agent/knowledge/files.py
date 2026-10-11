"""Durable, malware-scanned file ingestion into governed knowledge drafts."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import PurePath
from typing import Protocol
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.knowledge.ingestion import (
    ExtractedKnowledgeStructure,
    KnowledgeIngestionError,
    KnowledgeIngestionService,
)
from industrial_ops_agent.media.service import TenantObjectStore
from industrial_ops_agent.media.validation import MediaValidationFailure, inspect_media
from industrial_ops_agent.media.worker import MalwareScanner, ScanVerdict
from industrial_ops_agent.object_store.keys import (
    ObjectZone,
    TenantObjectKey,
    knowledge_source_object_key,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    IdempotencyRecord,
    KnowledgeIngestionAttemptRecord,
    KnowledgeIngestionJobRecord,
)
from industrial_ops_agent.persistence.tenant import validate_boundary_identifier

MAX_KNOWLEDGE_FILE_BYTES = 25 * 1024 * 1024
_TERMINAL_STATUSES = frozenset({"DRAFT_READY", "REJECTED", "FAILED"})
_SUPPORTED_MIME_EXTENSIONS = {
    "text/plain": frozenset({".txt"}),
    "text/markdown": frozenset({".md", ".markdown"}),
    "application/pdf": frozenset({".pdf"}),
}


class KnowledgeFileError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class KnowledgeDocumentParseError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ParsedKnowledgeDocument:
    pages: tuple[tuple[int, str, str], ...]
    parser_version: str
    structures: tuple[ExtractedKnowledgeStructure, ...] = ()


class KnowledgeDocumentParser(Protocol):
    async def parse(
        self,
        content: bytes,
        *,
        mime_type: str,
        identity: IdentityContext,
        ingestion_id: str,
        request_id: str,
        device_models: tuple[str, ...] = (),
    ) -> ParsedKnowledgeDocument: ...


@dataclass(frozen=True, slots=True)
class KnowledgeFilePlan:
    title: str
    source_filename: str
    declared_mime: str
    classification: str
    acl_subject_ids: tuple[str, ...]
    acl_roles: tuple[str, ...]
    device_families: tuple[str, ...]
    device_models: tuple[str, ...]
    valid_from: datetime
    valid_to: datetime | None


@dataclass(frozen=True, slots=True)
class KnowledgeFileAttemptView:
    attempt_id: str
    attempt_number: int
    workflow_id: str
    status: str
    failure_reason: str | None
    parser_version: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class KnowledgeFileView:
    ingestion_id: str
    workflow_id: str
    title: str
    source_filename: str
    declared_mime: str
    detected_mime: str | None
    classification: str
    acl_subject_ids: tuple[str, ...]
    acl_roles: tuple[str, ...]
    device_families: tuple[str, ...]
    device_models: tuple[str, ...]
    valid_from: datetime
    valid_to: datetime | None
    status: str
    failure_reason: str | None
    source_checksum: str | None
    size_bytes: int | None
    parser_version: str | None
    document_id: str | None
    document_version_id: str | None
    attempt_count: int
    attempts: tuple[KnowledgeFileAttemptView, ...]
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class KnowledgeIngestionActivityInput:
    ingestion_id: str
    identity: IdentityContext
    request_id: str
    workflow_id: str | None = None


class KnowledgeIngestionDispatcher(Protocol):
    async def dispatch(self, command: KnowledgeIngestionActivityInput) -> None: ...


class KnowledgeIngestionDispatchUnavailable(RuntimeError):
    """Temporal could not durably accept a knowledge ingestion workflow."""


class KnowledgeFileIngestionService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        object_store: TenantObjectStore,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._object_store = object_store

    def create_upload(
        self,
        identity: IdentityContext,
        plan: KnowledgeFilePlan,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> KnowledgeFileView:
        self._require_manage(identity, "knowledge-source-files", request_id)
        normalized = _validate_plan(plan)
        _validate_idempotency_key(idempotency_key)
        request_hash = _digest(
            json.dumps(
                {
                    "title": normalized.title,
                    "source_filename": normalized.source_filename,
                    "declared_mime": normalized.declared_mime,
                    "classification": normalized.classification,
                    "acl_subject_ids": sorted(normalized.acl_subject_ids),
                    "acl_roles": sorted(normalized.acl_roles),
                    "device_families": sorted(normalized.device_families),
                    "device_models": sorted(normalized.device_models),
                    "valid_from": normalized.valid_from.isoformat(),
                    "valid_to": normalized.valid_to.isoformat() if normalized.valid_to else None,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        storage_key = f"knowledge-file-intent:{idempotency_key}"
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            replay = session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.tenant_id == identity.tenant_id,
                    IdempotencyRecord.subject_id == identity.subject_id,
                    IdempotencyRecord.key == storage_key,
                )
            )
            if replay is not None:
                if replay.request_hash != request_hash:
                    raise KnowledgeFileError("knowledge_file_idempotency_conflict")
                job = session.scalar(
                    select(KnowledgeIngestionJobRecord).where(
                        KnowledgeIngestionJobRecord.tenant_id == identity.tenant_id,
                        KnowledgeIngestionJobRecord.ingestion_id == replay.result_ref,
                    )
                )
                if job is None:
                    raise KnowledgeFileError("knowledge_file_idempotency_state_invalid")
                return _view_with_attempts(session, identity, job)
            ingestion_id = f"knowledge-ingestion-{uuid4().hex}"
            workflow_id = f"knowledge-ingestion-workflow-{uuid4().hex}"
            quarantine_key = knowledge_source_object_key(
                identity.tenant_context,
                ObjectZone.QUARANTINE,
                ingestion_id,
            )
            job = KnowledgeIngestionJobRecord(
                tenant_id=identity.tenant_id,
                ingestion_id=ingestion_id,
                workflow_id=workflow_id,
                title=normalized.title,
                source_filename=normalized.source_filename,
                declared_mime=normalized.declared_mime,
                detected_mime=None,
                classification=normalized.classification,
                acl_subject_ids=list(sorted(normalized.acl_subject_ids)),
                acl_roles=list(sorted(normalized.acl_roles)),
                device_families=list(sorted(normalized.device_families)),
                device_models=list(sorted(normalized.device_models)),
                valid_from=normalized.valid_from,
                valid_to=normalized.valid_to,
                quarantine_key=quarantine_key.value,
                clean_key=None,
                source_uri=None,
                source_checksum=None,
                size_bytes=None,
                status="AWAITING_UPLOAD",
                failure_reason=None,
                parser_version=None,
                document_id=None,
                document_version_id=None,
                created_by_subject_id=identity.subject_id,
                attempt_count=0,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(job)
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=ingestion_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _view_with_attempts(session, identity, job)

    async def upload_content(
        self,
        identity: IdentityContext,
        ingestion_id: str,
        *,
        content: bytes,
        expected_version: int,
        request_id: str,
    ) -> KnowledgeFileView:
        def prepare_upload() -> KnowledgeFileView | tuple[TenantObjectKey, str, str]:
            self._require_manage(identity, ingestion_id, request_id)
            with self._database.transaction(identity.tenant_context) as session:
                job = _job(session, identity, ingestion_id, lock=False)
                if job.status in {"QUEUED", "RUNNING"}:
                    if job.source_checksum == _digest_bytes(content):
                        return _view_with_attempts(session, identity, job)
                    raise KnowledgeFileError("knowledge_file_content_conflict")
                if job.status != "AWAITING_UPLOAD" or job.version != expected_version:
                    raise KnowledgeFileError("knowledge_file_state_changed")
                declared_mime = job.declared_mime
                quarantine_key_value = job.quarantine_key
            detected_mime, source_checksum = _inspect_file(content, declared_mime)
            expected_key = knowledge_source_object_key(
                identity.tenant_context,
                ObjectZone.QUARANTINE,
                ingestion_id,
            )
            if expected_key.value != quarantine_key_value:
                raise KnowledgeFileError("knowledge_file_object_key_invalid")
            return expected_key, detected_mime, source_checksum

        prepared = await asyncio.to_thread(prepare_upload)
        if isinstance(prepared, KnowledgeFileView):
            return prepared
        expected_key, detected_mime, source_checksum = prepared
        await self._object_store.put(identity.tenant_context, expected_key, content)
        now = datetime.now(UTC)
        def persist_upload() -> KnowledgeFileView:
            with self._database.transaction(identity.tenant_context) as session:
                job = _job(session, identity, ingestion_id, lock=True)
                if job.status in {"QUEUED", "RUNNING"} and job.source_checksum == source_checksum:
                    return _view_with_attempts(session, identity, job)
                if job.status != "AWAITING_UPLOAD" or job.version != expected_version:
                    raise KnowledgeFileError("knowledge_file_state_changed")
                job.detected_mime = detected_mime
                job.source_checksum = source_checksum
                job.size_bytes = len(content)
                job.status = "QUEUED"
                _queue_attempt(session, job, now)
                job.version += 1
                job.updated_at = now
                session.flush()
                return _view_with_attempts(session, identity, job)

        return await asyncio.to_thread(persist_upload)

    def get(
        self,
        identity: IdentityContext,
        ingestion_id: str,
        *,
        request_id: str,
    ) -> KnowledgeFileView:
        self._require_manage(identity, ingestion_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            job = _job(session, identity, ingestion_id, lock=False)
            return _view_with_attempts(session, identity, job)

    def list(
        self,
        identity: IdentityContext,
        *,
        status: str | None,
        limit: int,
        offset: int,
        request_id: str,
    ) -> tuple[tuple[KnowledgeFileView, ...], int]:
        self._require_manage(identity, "knowledge-source-files", request_id)
        if status is not None and status not in {
            "AWAITING_UPLOAD",
            "QUEUED",
            "RUNNING",
            "DRAFT_READY",
            "REJECTED",
            "FAILED",
            "PURGED",
        }:
            raise KnowledgeFileError("knowledge_file_status_invalid")
        if limit < 1 or limit > 100 or offset < 0:
            raise KnowledgeFileError("knowledge_file_pagination_invalid")
        filters = [KnowledgeIngestionJobRecord.tenant_id == identity.tenant_id]
        if status is not None:
            filters.append(KnowledgeIngestionJobRecord.status == status)
        with self._database.transaction(identity.tenant_context) as session:
            total = int(
                session.scalar(
                    select(func.count()).select_from(KnowledgeIngestionJobRecord).where(*filters)
                )
                or 0
            )
            jobs = tuple(
                session.scalars(
                    select(KnowledgeIngestionJobRecord)
                    .where(*filters)
                    .order_by(
                        KnowledgeIngestionJobRecord.updated_at.desc(),
                        KnowledgeIngestionJobRecord.ingestion_id,
                    )
                    .offset(offset)
                    .limit(limit)
                )
            )
            attempts_by_ingestion: dict[str, list[KnowledgeFileAttemptView]] = {
                job.ingestion_id: [] for job in jobs
            }
            if jobs:
                attempts = session.scalars(
                    select(KnowledgeIngestionAttemptRecord)
                    .where(
                        KnowledgeIngestionAttemptRecord.tenant_id == identity.tenant_id,
                        KnowledgeIngestionAttemptRecord.ingestion_id.in_(
                            [job.ingestion_id for job in jobs]
                        ),
                    )
                    .order_by(
                        KnowledgeIngestionAttemptRecord.ingestion_id,
                        KnowledgeIngestionAttemptRecord.attempt_number.desc(),
                    )
                )
                for attempt in attempts:
                    attempts_by_ingestion[attempt.ingestion_id].append(_attempt_view(attempt))
            return (
                tuple(
                    _view(
                        job,
                        attempts=tuple(attempts_by_ingestion[job.ingestion_id]),
                    )
                    for job in jobs
                ),
                total,
            )

    def reprocess(
        self,
        identity: IdentityContext,
        ingestion_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> KnowledgeFileView:
        self._require_manage(identity, ingestion_id, request_id)
        _validate_idempotency_key(idempotency_key)
        request_hash = _digest(
            json.dumps(
                {
                    "ingestion_id": ingestion_id,
                    "expected_version": expected_version,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        storage_key = f"knowledge-file-reprocess:{idempotency_key}"
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            job = _job(session, identity, ingestion_id, lock=True)
            replay = session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.tenant_id == identity.tenant_id,
                    IdempotencyRecord.subject_id == identity.subject_id,
                    IdempotencyRecord.key == storage_key,
                )
            )
            if replay is not None:
                if replay.request_hash != request_hash:
                    raise KnowledgeFileError("knowledge_file_idempotency_conflict")
                attempt = session.scalar(
                    select(KnowledgeIngestionAttemptRecord).where(
                        KnowledgeIngestionAttemptRecord.tenant_id == identity.tenant_id,
                        KnowledgeIngestionAttemptRecord.attempt_id == replay.result_ref,
                        KnowledgeIngestionAttemptRecord.ingestion_id == ingestion_id,
                    )
                )
                if attempt is None:
                    raise KnowledgeFileError("knowledge_file_idempotency_state_invalid")
                return _view_with_attempts(session, identity, job)
            if job.status == "QUEUED":
                return _view_with_attempts(session, identity, job)
            if job.status != "FAILED":
                raise KnowledgeFileError("knowledge_file_reprocess_forbidden")
            if job.version != expected_version:
                raise KnowledgeFileError("knowledge_file_state_changed")
            if job.source_checksum is None or job.size_bytes is None:
                raise KnowledgeFileError("knowledge_file_source_state_invalid")
            job.workflow_id = f"knowledge-ingestion-workflow-{uuid4().hex}"
            job.status = "QUEUED"
            job.failure_reason = None
            job.parser_version = None
            job.clean_key = None
            job.source_uri = None
            job.document_id = None
            job.document_version_id = None
            attempt = _queue_attempt(session, job, now)
            job.version += 1
            job.updated_at = now
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=attempt.attempt_id,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _view_with_attempts(session, identity, job)

    def _require_manage(
        self,
        identity: IdentityContext,
        resource_id: str,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            Action.PUBLISH_KNOWLEDGE,
            ResourceContext(identity.tenant_id, resource_id),
            request_id=request_id,
        )


class KnowledgeIngestionActivity:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        object_store: TenantObjectStore,
        scanner: MalwareScanner,
        parser: KnowledgeDocumentParser,
        *,
        bucket: str,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._object_store = object_store
        self._scanner = scanner
        self._parser = parser
        self._bucket = bucket

    async def execute(self, command: KnowledgeIngestionActivityInput) -> KnowledgeFileView:
        identity = command.identity
        current = self._start(identity, command.ingestion_id, command.workflow_id)
        if current.status in _TERMINAL_STATUSES:
            return current
        quarantine_key = knowledge_source_object_key(
            identity.tenant_context,
            ObjectZone.QUARANTINE,
            command.ingestion_id,
        )
        content = await self._object_store.get(identity.tenant_context, quarantine_key)
        if (
            content is None
            or current.source_checksum is None
            or current.size_bytes != len(content)
            or current.source_checksum != _digest_bytes(content)
        ):
            return self._finish(
                identity,
                command.ingestion_id,
                "FAILED",
                "source_integrity_failed",
                workflow_id=current.workflow_id,
            )
        verdict = await self._scanner.scan(content)
        if verdict is ScanVerdict.INFECTED:
            return self._finish(
                identity,
                command.ingestion_id,
                "REJECTED",
                "malware_detected",
                workflow_id=current.workflow_id,
            )
        clean_key = knowledge_source_object_key(
            identity.tenant_context,
            ObjectZone.CLEAN,
            command.ingestion_id,
        )
        await self._object_store.put(identity.tenant_context, clean_key, content)
        promoted = await self._object_store.get(identity.tenant_context, clean_key)
        if promoted is None or _digest_bytes(promoted) != current.source_checksum:
            return self._finish(
                identity,
                command.ingestion_id,
                "FAILED",
                "clean_copy_failed",
                workflow_id=current.workflow_id,
            )
        source_uri = f"minio://{self._bucket}/{clean_key.value}"
        try:
            parsed = await self._parser.parse(
                promoted,
                mime_type=current.detected_mime or current.declared_mime,
                identity=identity,
                ingestion_id=command.ingestion_id,
                request_id=command.request_id,
                device_models=current.device_models,
            )
            version = KnowledgeIngestionService(
                self._database,
                self._authorizer,
            ).create_extracted_document(
                identity,
                title=current.title,
                source_uri=source_uri,
                pages=parsed.pages,
                structures=parsed.structures,
                parser_version=parsed.parser_version,
                source_checksum=current.source_checksum,
                classification=current.classification,
                acl_subject_ids=current.acl_subject_ids,
                acl_roles=current.acl_roles,
                device_families=current.device_families,
                device_models=current.device_models,
                valid_from=current.valid_from,
                valid_to=current.valid_to,
                idempotency_key=f"file-{command.ingestion_id}",
                request_id=command.request_id,
            )
        except (KnowledgeDocumentParseError, KnowledgeIngestionError) as exc:
            reason = exc.reason
            return self._finish(
                identity,
                command.ingestion_id,
                "FAILED",
                reason,
                workflow_id=current.workflow_id,
                clean_key=clean_key.value,
                source_uri=source_uri,
            )
        return self._finish(
            identity,
            command.ingestion_id,
            "DRAFT_READY",
            None,
            workflow_id=current.workflow_id,
            clean_key=clean_key.value,
            source_uri=source_uri,
            parser_version=parsed.parser_version,
            document_id=version.document_id,
            document_version_id=version.document_version_id,
        )

    def _start(
        self,
        identity: IdentityContext,
        ingestion_id: str,
        workflow_id: str | None,
    ) -> KnowledgeFileView:
        with self._database.transaction(identity.tenant_context) as session:
            job = _job(session, identity, ingestion_id, lock=True)
            if workflow_id is not None and job.workflow_id != workflow_id:
                raise KnowledgeFileError("knowledge_file_workflow_stale")
            if job.status in _TERMINAL_STATUSES or job.status == "RUNNING":
                return _view_with_attempts(session, identity, job)
            if job.status != "QUEUED":
                raise KnowledgeFileError("knowledge_file_not_ready")
            now = datetime.now(UTC)
            attempt = _current_attempt(session, identity, job)
            attempt.status = "RUNNING"
            attempt.started_at = now
            attempt.updated_at = now
            job.status = "RUNNING"
            job.version += 1
            job.updated_at = now
            session.flush()
            return _view_with_attempts(session, identity, job)

    def _finish(
        self,
        identity: IdentityContext,
        ingestion_id: str,
        status: str,
        failure_reason: str | None,
        *,
        workflow_id: str,
        clean_key: str | None = None,
        source_uri: str | None = None,
        parser_version: str | None = None,
        document_id: str | None = None,
        document_version_id: str | None = None,
    ) -> KnowledgeFileView:
        with self._database.transaction(identity.tenant_context) as session:
            job = _job(session, identity, ingestion_id, lock=True)
            if job.workflow_id != workflow_id:
                raise KnowledgeFileError("knowledge_file_workflow_stale")
            if job.status in _TERMINAL_STATUSES:
                return _view_with_attempts(session, identity, job)
            now = datetime.now(UTC)
            attempt = _current_attempt(session, identity, job)
            attempt.status = status
            attempt.failure_reason = failure_reason
            attempt.parser_version = parser_version
            attempt.completed_at = now
            attempt.updated_at = now
            job.status = status
            job.failure_reason = failure_reason
            job.clean_key = clean_key
            job.source_uri = source_uri
            job.parser_version = parser_version
            job.document_id = document_id
            job.document_version_id = document_version_id
            job.version += 1
            job.updated_at = now
            session.flush()
            return _view_with_attempts(session, identity, job)


def _validate_plan(plan: KnowledgeFilePlan) -> KnowledgeFilePlan:
    title = plan.title.strip()
    filename = plan.source_filename.strip()
    declared_mime = plan.declared_mime.split(";", 1)[0].strip().casefold()
    if not title or len(title) > 255:
        raise KnowledgeFileError("knowledge_file_title_invalid")
    if (
        not filename
        or len(filename) > 255
        or PurePath(filename).name != filename
        or any(ord(character) < 32 for character in filename)
    ):
        raise KnowledgeFileError("knowledge_source_filename_invalid")
    extensions = _SUPPORTED_MIME_EXTENSIONS.get(declared_mime)
    if extensions is None or PurePath(filename).suffix.casefold() not in extensions:
        raise KnowledgeFileError("knowledge_file_type_invalid")
    if plan.classification not in {"internal", "restricted"}:
        raise KnowledgeFileError("knowledge_classification_invalid")
    if plan.classification == "restricted" and not plan.acl_roles:
        raise KnowledgeFileError("restricted_knowledge_role_acl_required")
    if any(
        len(values) > 100
        for values in (
            plan.acl_subject_ids,
            plan.acl_roles,
            plan.device_families,
            plan.device_models,
        )
    ):
        raise KnowledgeFileError("knowledge_scope_invalid")
    try:
        for subject_id in plan.acl_subject_ids:
            validate_boundary_identifier(subject_id, field="acl_subject_id")
        for role in plan.acl_roles:
            Role(role)
    except ValueError as exc:
        raise KnowledgeFileError("knowledge_acl_invalid") from exc
    collections = (
        plan.acl_subject_ids,
        plan.acl_roles,
        plan.device_families,
        plan.device_models,
    )
    if any(len(set(values)) != len(values) for values in collections) or any(
        not value or len(value) > 128 for value in (*plan.device_families, *plan.device_models)
    ):
        raise KnowledgeFileError("knowledge_scope_invalid")
    if plan.valid_from.tzinfo is None or plan.valid_from.utcoffset() is None:
        raise KnowledgeFileError("knowledge_validity_invalid")
    if plan.valid_to is not None and (
        plan.valid_to.tzinfo is None
        or plan.valid_to.utcoffset() is None
        or plan.valid_to <= plan.valid_from
    ):
        raise KnowledgeFileError("knowledge_validity_invalid")
    return KnowledgeFilePlan(
        title=title,
        source_filename=filename,
        declared_mime=declared_mime,
        classification=plan.classification,
        acl_subject_ids=plan.acl_subject_ids,
        acl_roles=plan.acl_roles,
        device_families=plan.device_families,
        device_models=plan.device_models,
        valid_from=plan.valid_from,
        valid_to=plan.valid_to,
    )


def _inspect_file(content: bytes, declared_mime: str) -> tuple[str, str]:
    if not content or len(content) > MAX_KNOWLEDGE_FILE_BYTES:
        raise KnowledgeFileError("knowledge_file_size_invalid")
    if declared_mime == "application/pdf":
        try:
            inspection = inspect_media(
                content,
                declared_mime=declared_mime,
                max_bytes=MAX_KNOWLEDGE_FILE_BYTES,
            )
        except MediaValidationFailure as exc:
            raise KnowledgeFileError(exc.reason_code) from exc
        return inspection.detected_mime, inspection.content_hash
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KnowledgeFileError("knowledge_text_encoding_invalid") from exc
    if "\x00" in decoded or len(decoded.strip()) < 20:
        raise KnowledgeFileError("knowledge_content_invalid")
    return declared_mime, _digest_bytes(content)


def _job(
    session: Session,
    identity: IdentityContext,
    ingestion_id: str,
    *,
    lock: bool,
) -> KnowledgeIngestionJobRecord:
    statement = select(KnowledgeIngestionJobRecord).where(
        KnowledgeIngestionJobRecord.tenant_id == identity.tenant_id,
        KnowledgeIngestionJobRecord.ingestion_id == ingestion_id,
    )
    if lock:
        statement = statement.with_for_update()
    job = session.scalar(statement)
    if job is None:
        raise KnowledgeFileError("knowledge_file_not_visible")
    return job


def _queue_attempt(
    session: Session,
    job: KnowledgeIngestionJobRecord,
    now: datetime,
) -> KnowledgeIngestionAttemptRecord:
    job.attempt_count += 1
    attempt = KnowledgeIngestionAttemptRecord(
        tenant_id=job.tenant_id,
        attempt_id=f"knowledge-attempt-{uuid4().hex}",
        ingestion_id=job.ingestion_id,
        attempt_number=job.attempt_count,
        workflow_id=job.workflow_id,
        status="QUEUED",
        failure_reason=None,
        parser_version=None,
        started_at=None,
        completed_at=None,
        created_at=now,
        updated_at=now,
    )
    session.add(attempt)
    return attempt


def _current_attempt(
    session: Session,
    identity: IdentityContext,
    job: KnowledgeIngestionJobRecord,
) -> KnowledgeIngestionAttemptRecord:
    attempt = session.scalar(
        select(KnowledgeIngestionAttemptRecord).where(
            KnowledgeIngestionAttemptRecord.tenant_id == identity.tenant_id,
            KnowledgeIngestionAttemptRecord.ingestion_id == job.ingestion_id,
            KnowledgeIngestionAttemptRecord.workflow_id == job.workflow_id,
            KnowledgeIngestionAttemptRecord.attempt_number == job.attempt_count,
        )
    )
    if attempt is None:
        raise KnowledgeFileError("knowledge_file_attempt_state_invalid")
    return attempt


def _attempt_views(
    session: Session,
    identity: IdentityContext,
    ingestion_id: str,
) -> tuple[KnowledgeFileAttemptView, ...]:
    attempts = tuple(
        session.scalars(
            select(KnowledgeIngestionAttemptRecord)
            .where(
                KnowledgeIngestionAttemptRecord.tenant_id == identity.tenant_id,
                KnowledgeIngestionAttemptRecord.ingestion_id == ingestion_id,
            )
            .order_by(KnowledgeIngestionAttemptRecord.attempt_number.desc())
        )
    )
    return tuple(_attempt_view(attempt) for attempt in attempts)


def _attempt_view(attempt: KnowledgeIngestionAttemptRecord) -> KnowledgeFileAttemptView:
    return KnowledgeFileAttemptView(
        attempt_id=attempt.attempt_id,
        attempt_number=attempt.attempt_number,
        workflow_id=attempt.workflow_id,
        status=attempt.status,
        failure_reason=attempt.failure_reason,
        parser_version=attempt.parser_version,
        started_at=_utc(attempt.started_at) if attempt.started_at else None,
        completed_at=_utc(attempt.completed_at) if attempt.completed_at else None,
        created_at=_utc(attempt.created_at),
        updated_at=_utc(attempt.updated_at),
    )


def _view_with_attempts(
    session: Session,
    identity: IdentityContext,
    job: KnowledgeIngestionJobRecord,
) -> KnowledgeFileView:
    return _view(job, attempts=_attempt_views(session, identity, job.ingestion_id))


def _view(
    job: KnowledgeIngestionJobRecord,
    *,
    attempts: tuple[KnowledgeFileAttemptView, ...],
) -> KnowledgeFileView:
    return KnowledgeFileView(
        ingestion_id=job.ingestion_id,
        workflow_id=job.workflow_id,
        title=job.title,
        source_filename=job.source_filename,
        declared_mime=job.declared_mime,
        detected_mime=job.detected_mime,
        classification=job.classification,
        acl_subject_ids=tuple(job.acl_subject_ids),
        acl_roles=tuple(job.acl_roles),
        device_families=tuple(job.device_families),
        device_models=tuple(job.device_models),
        valid_from=_utc(job.valid_from),
        valid_to=_utc(job.valid_to) if job.valid_to else None,
        status=job.status,
        failure_reason=job.failure_reason,
        source_checksum=job.source_checksum,
        size_bytes=job.size_bytes,
        parser_version=job.parser_version,
        document_id=job.document_id,
        document_version_id=job.document_version_id,
        attempt_count=job.attempt_count,
        attempts=attempts,
        version=job.version,
        created_at=_utc(job.created_at),
        updated_at=_utc(job.updated_at),
    )


def _validate_idempotency_key(value: str) -> None:
    if not value or len(value) > 200 or any(character.isspace() for character in value):
        raise KnowledgeFileError("knowledge_idempotency_key_invalid")


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()
