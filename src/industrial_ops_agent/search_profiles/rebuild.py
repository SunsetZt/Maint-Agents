"""Durable Temporal command model for OpenSearch profile rebuilds."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    IdempotencyRecord,
    KnowledgeSearchProfileRecord,
    KnowledgeSearchRebuildJobRecord,
)
from industrial_ops_agent.search_profiles.embedding import EmbeddingUnavailable
from industrial_ops_agent.search_profiles.service import (
    KnowledgeSearchConflict,
    KnowledgeSearchNotVisible,
    KnowledgeSearchProfileService,
)
from industrial_ops_agent.search_profiles.store import SearchStoreUnavailable

TERMINAL_REBUILD_STATUSES = frozenset({"COMPLETED", "FAILED"})


@dataclass(frozen=True, slots=True)
class SearchProfileRebuildView:
    rebuild_job_id: str
    search_profile_id: str
    workflow_id: str
    expected_profile_version: int
    status: str
    stage: str
    progress_percent: int
    completed_items: int
    total_items: int
    attempt_count: int
    requested_by_subject_id: str
    failure_code: str | None
    started_at: datetime | None
    completed_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SearchProfileRebuildActivityInput:
    rebuild_job_id: str
    workflow_id: str
    identity: IdentityContext
    request_id: str


class SearchProfileRebuildDispatcher(Protocol):
    async def dispatch(self, command: SearchProfileRebuildActivityInput) -> None: ...


class SearchProfileRebuildDispatchUnavailable(RuntimeError):
    """Temporal could not durably accept a profile rebuild command."""


class SearchProfileRebuildService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def request(
        self,
        identity: IdentityContext,
        search_profile_id: str,
        *,
        expected_profile_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> SearchProfileRebuildView:
        self._require(identity, search_profile_id, request_id)
        if not 8 <= len(idempotency_key) <= 255:
            raise KnowledgeSearchConflict("search_profile_rebuild_idempotency_invalid")
        request_hash = sha256(
            f"{search_profile_id}\0{expected_profile_version}".encode()
        ).hexdigest()
        storage_key = f"knowledge-search-rebuild:{idempotency_key}"
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
                    raise KnowledgeSearchConflict("search_profile_rebuild_idempotency_conflict")
                return _view(
                    _job_or_hidden(session, identity.tenant_id, str(replay.result_ref))
                )

            profile = session.scalar(
                select(KnowledgeSearchProfileRecord)
                .where(
                    KnowledgeSearchProfileRecord.tenant_id == identity.tenant_id,
                    KnowledgeSearchProfileRecord.search_profile_id == search_profile_id,
                )
                .with_for_update()
            )
            if profile is None:
                raise KnowledgeSearchNotVisible
            if (
                profile.state_version != expected_profile_version
                or profile.status not in {"DRAFT", "SYNCED"}
            ):
                raise KnowledgeSearchConflict(
                    "search_profile_rebuild_state_invalid", profile.state_version
                )
            active = session.scalar(
                select(KnowledgeSearchRebuildJobRecord).where(
                    KnowledgeSearchRebuildJobRecord.tenant_id == identity.tenant_id,
                    KnowledgeSearchRebuildJobRecord.search_profile_id == search_profile_id,
                    KnowledgeSearchRebuildJobRecord.status.in_(("QUEUED", "RUNNING")),
                )
            )
            if active is not None:
                raise KnowledgeSearchConflict(
                    "search_profile_rebuild_already_active", active.version
                )

            rebuild_job_id = f"search-rebuild-{uuid4().hex}"
            workflow_id = f"search-rebuild-workflow-{uuid4().hex}"
            record = KnowledgeSearchRebuildJobRecord(
                rebuild_job_id=rebuild_job_id,
                tenant_id=identity.tenant_id,
                search_profile_id=search_profile_id,
                workflow_id=workflow_id,
                expected_profile_version=expected_profile_version,
                status="QUEUED",
                stage="QUEUED",
                progress_percent=0,
                completed_items=0,
                total_items=0,
                attempt_count=0,
                requested_by_subject_id=identity.subject_id,
                failure_code=None,
                started_at=None,
                completed_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=request_hash,
                    result_ref=rebuild_job_id,
                    expires_at=now + timedelta(days=30),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _view(record)

    def list(
        self,
        identity: IdentityContext,
        *,
        search_profile_id: str | None,
        limit: int,
        request_id: str,
    ) -> tuple[SearchProfileRebuildView, ...]:
        self._require(identity, search_profile_id or "knowledge-search-rebuilds", request_id)
        if not 1 <= limit <= 100:
            raise KnowledgeSearchConflict("search_profile_rebuild_limit_invalid")
        statement = select(KnowledgeSearchRebuildJobRecord).where(
            KnowledgeSearchRebuildJobRecord.tenant_id == identity.tenant_id
        )
        if search_profile_id is not None:
            statement = statement.where(
                KnowledgeSearchRebuildJobRecord.search_profile_id == search_profile_id
            )
        with self._database.transaction(identity.tenant_context) as session:
            records = session.scalars(
                statement.order_by(
                    KnowledgeSearchRebuildJobRecord.updated_at.desc(),
                    KnowledgeSearchRebuildJobRecord.rebuild_job_id,
                ).limit(limit)
            )
            return tuple(_view(item) for item in records)

    def get(
        self,
        identity: IdentityContext,
        rebuild_job_id: str,
        *,
        request_id: str,
    ) -> SearchProfileRebuildView:
        self._require(identity, rebuild_job_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            return _view(_job_or_hidden(session, identity.tenant_id, rebuild_job_id))

    def retry(
        self,
        identity: IdentityContext,
        rebuild_job_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> SearchProfileRebuildView:
        self._require(identity, rebuild_job_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = _job_or_hidden(
                session, identity.tenant_id, rebuild_job_id, lock=True
            )
            if record.version != expected_version or record.status != "FAILED":
                raise KnowledgeSearchConflict(
                    "search_profile_rebuild_retry_state_invalid", record.version
                )
            profile = session.scalar(
                select(KnowledgeSearchProfileRecord)
                .where(
                    KnowledgeSearchProfileRecord.tenant_id == identity.tenant_id,
                    KnowledgeSearchProfileRecord.search_profile_id == record.search_profile_id,
                )
                .with_for_update()
            )
            if profile is None:
                raise KnowledgeSearchNotVisible
            if (
                profile.state_version != record.expected_profile_version
                or profile.status not in {"DRAFT", "SYNCED"}
            ):
                raise KnowledgeSearchConflict(
                    "search_profile_rebuild_profile_changed", profile.state_version
                )
            active = session.scalar(
                select(KnowledgeSearchRebuildJobRecord).where(
                    KnowledgeSearchRebuildJobRecord.tenant_id == identity.tenant_id,
                    KnowledgeSearchRebuildJobRecord.search_profile_id
                    == record.search_profile_id,
                    KnowledgeSearchRebuildJobRecord.rebuild_job_id != rebuild_job_id,
                    KnowledgeSearchRebuildJobRecord.status.in_(("QUEUED", "RUNNING")),
                )
            )
            if active is not None:
                raise KnowledgeSearchConflict(
                    "search_profile_rebuild_already_active", active.version
                )
            record.workflow_id = f"search-rebuild-workflow-{uuid4().hex}"
            record.status = "QUEUED"
            record.stage = "QUEUED"
            record.progress_percent = 0
            record.completed_items = 0
            record.total_items = 0
            record.failure_code = None
            record.started_at = None
            record.completed_at = None
            record.version += 1
            record.updated_at = now
            session.flush()
            return _view(record)

    def mark_dispatch_failed(
        self,
        identity: IdentityContext,
        rebuild_job_id: str,
        *,
        workflow_id: str,
    ) -> SearchProfileRebuildView:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = _job_or_hidden(
                session, identity.tenant_id, rebuild_job_id, lock=True
            )
            if record.workflow_id == workflow_id and record.status == "QUEUED":
                record.status = "FAILED"
                record.stage = "FAILED"
                record.failure_code = "search_profile_rebuild_dispatch_unavailable"
                record.completed_at = now
                record.version += 1
                record.updated_at = now
                session.flush()
            return _view(record)

    def _require(self, identity: IdentityContext, resource_id: str, request_id: str) -> None:
        self._authorizer.require(
            identity,
            Action.MANAGE_KNOWLEDGE_SEARCH,
            ResourceContext(identity.tenant_id, resource_id),
            request_id=request_id,
        )


class SearchProfileRebuildActivity:
    def __init__(
        self,
        database: Database,
        profile_service: KnowledgeSearchProfileService,
    ) -> None:
        self._database = database
        self._profile_service = profile_service

    async def execute(
        self, command: SearchProfileRebuildActivityInput
    ) -> SearchProfileRebuildView:
        context = command.identity.tenant_context
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            record = _job_or_hidden(
                session, command.identity.tenant_id, command.rebuild_job_id, lock=True
            )
            if record.workflow_id != command.workflow_id:
                raise KnowledgeSearchConflict("search_profile_rebuild_workflow_stale")
            if record.status == "COMPLETED":
                return _view(record)
            if record.status not in {"QUEUED", "RUNNING"}:
                raise KnowledgeSearchConflict("search_profile_rebuild_activity_state_invalid")
            profile = session.scalar(
                select(KnowledgeSearchProfileRecord).where(
                    KnowledgeSearchProfileRecord.tenant_id == command.identity.tenant_id,
                    KnowledgeSearchProfileRecord.search_profile_id == record.search_profile_id,
                )
            )
            if profile is None:
                raise KnowledgeSearchNotVisible
            if (
                record.status == "RUNNING"
                and profile.state_version == record.expected_profile_version + 1
                and profile.status == "SYNCED"
            ):
                record.status = "COMPLETED"
                record.stage = "COMPLETED"
                record.progress_percent = 100
                record.completed_items = record.total_items
                record.completed_at = now
                record.failure_code = None
                record.version += 1
                record.updated_at = now
                session.flush()
                return _view(record)
            record.status = "RUNNING"
            record.stage = "VALIDATING"
            record.progress_percent = 5
            record.attempt_count += 1
            record.started_at = record.started_at or now
            record.completed_at = None
            record.failure_code = None
            record.version += 1
            record.updated_at = now
            profile_id = record.search_profile_id
            expected_profile_version = record.expected_profile_version

        def report(stage: str, completed: int, total: int, percent: int) -> None:
            self._report(command, stage, completed, total, percent)

        try:
            await asyncio.to_thread(
                self._profile_service.sync,
                command.identity,
                profile_id,
                expected_version=expected_profile_version,
                request_id=command.request_id,
                progress=report,
                rebuild_job_id=command.rebuild_job_id,
            )
        except Exception as exc:
            return self._fail(command, _failure_code(exc))

        completed_at = datetime.now(UTC)
        with self._database.transaction(context) as session:
            record = _job_or_hidden(
                session, command.identity.tenant_id, command.rebuild_job_id, lock=True
            )
            if record.workflow_id != command.workflow_id:
                raise KnowledgeSearchConflict("search_profile_rebuild_workflow_stale")
            record.status = "COMPLETED"
            record.stage = "COMPLETED"
            record.progress_percent = 100
            record.completed_items = record.total_items
            record.failure_code = None
            record.completed_at = completed_at
            record.version += 1
            record.updated_at = completed_at
            session.flush()
            return _view(record)

    def _report(
        self,
        command: SearchProfileRebuildActivityInput,
        stage: str,
        completed: int,
        total: int,
        percent: int,
    ) -> None:
        now = datetime.now(UTC)
        with self._database.transaction(command.identity.tenant_context) as session:
            record = _job_or_hidden(
                session, command.identity.tenant_id, command.rebuild_job_id, lock=True
            )
            if record.workflow_id != command.workflow_id or record.status != "RUNNING":
                raise KnowledgeSearchConflict("search_profile_rebuild_workflow_stale")
            record.stage = stage
            record.completed_items = max(0, completed)
            record.total_items = max(0, total)
            record.progress_percent = min(99, max(record.progress_percent, percent))
            record.version += 1
            record.updated_at = now

    def _fail(
        self,
        command: SearchProfileRebuildActivityInput,
        failure_code: str,
    ) -> SearchProfileRebuildView:
        now = datetime.now(UTC)
        with self._database.transaction(command.identity.tenant_context) as session:
            record = _job_or_hidden(
                session, command.identity.tenant_id, command.rebuild_job_id, lock=True
            )
            if record.workflow_id != command.workflow_id:
                raise KnowledgeSearchConflict("search_profile_rebuild_workflow_stale")
            record.status = "FAILED"
            record.stage = "FAILED"
            record.failure_code = failure_code
            record.completed_at = now
            record.version += 1
            record.updated_at = now
            session.flush()
            return _view(record)


def _job_or_hidden(
    session: Session,
    tenant_id: str,
    rebuild_job_id: str,
    *,
    lock: bool = False,
) -> KnowledgeSearchRebuildJobRecord:
    statement = select(KnowledgeSearchRebuildJobRecord).where(
        KnowledgeSearchRebuildJobRecord.tenant_id == tenant_id,
        KnowledgeSearchRebuildJobRecord.rebuild_job_id == rebuild_job_id,
    )
    if lock:
        statement = statement.with_for_update()
    record = session.scalar(statement)
    if record is None:
        raise KnowledgeSearchNotVisible
    return record


def _failure_code(exc: Exception) -> str:
    if isinstance(exc, (KnowledgeSearchConflict, EmbeddingUnavailable)):
        return exc.reason
    if isinstance(exc, AuthorizationDenied):
        return "search_profile_rebuild_authorization_denied"
    if isinstance(exc, KnowledgeSearchNotVisible):
        return "search_profile_rebuild_not_visible"
    if isinstance(exc, SearchStoreUnavailable):
        return "search_profile_rebuild_store_unavailable"
    return "search_profile_rebuild_failed"


def _view(record: KnowledgeSearchRebuildJobRecord) -> SearchProfileRebuildView:
    return SearchProfileRebuildView(
        rebuild_job_id=record.rebuild_job_id,
        search_profile_id=record.search_profile_id,
        workflow_id=record.workflow_id,
        expected_profile_version=record.expected_profile_version,
        status=record.status,
        stage=record.stage,
        progress_percent=record.progress_percent,
        completed_items=record.completed_items,
        total_items=record.total_items,
        attempt_count=record.attempt_count,
        requested_by_subject_id=record.requested_by_subject_id,
        failure_code=record.failure_code,
        started_at=record.started_at,
        completed_at=record.completed_at,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
