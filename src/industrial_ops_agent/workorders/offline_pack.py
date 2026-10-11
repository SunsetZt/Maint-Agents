"""Governed, expiring read snapshots for assigned field engineers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.maintenance_planning.review_isolation import (
    ReviewIsolationConflict,
    resolve_source,
    work_order_read_transaction,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetRecord,
    AssetSiteLinkRecord,
    CitationAnchorRecord,
    DiagnosisRunRecord,
    IncidentRecord,
    WorkOrderAssignmentRecord,
    WorkOrderOfflinePackRecord,
    WorkOrderRecord,
)
from industrial_ops_agent.workorders.service import WorkOrderConflict, WorkOrderNotVisible

OFFLINE_PACK_SCHEMA_VERSION = "field-offline-pack-v1"
OFFLINE_PACK_TTL = timedelta(hours=8)
OFFLINE_PACK_EXECUTABLE_STATUSES = frozenset({"ASSIGNED", "ACCEPTED", "IN_PROGRESS", "ON_HOLD"})
MAX_DIAGNOSIS_LIST_ITEMS = 12
MAX_CITATIONS = 20
MAX_DIAGNOSIS_TEXT = 2000
MAX_DIAGNOSIS_LIST_TEXT = 500
MAX_CITATION_EXCERPT = 1000


class OfflinePackNotAvailable(Exception):
    """The caller has no current usable offline pack."""


@dataclass(frozen=True, slots=True)
class WorkOrderOfflinePackView:
    pack_id: str
    status: str
    schema_version: str
    subject_id: str
    work_order_id: str
    work_order_version: int
    asset_id: str
    asset_version: int
    assignment_id: str
    assignment_assigned_at: datetime
    snapshot: dict[str, Any]
    content_hash: str
    issued_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    revoked_by_subject_id: str | None
    revocation_reason: str | None
    version: int
    legal_actions: list[str]


@dataclass(frozen=True, slots=True)
class WorkOrderOfflinePackResult:
    pack: WorkOrderOfflinePackView
    created: bool


@dataclass(frozen=True, slots=True)
class _OfflinePackContext:
    work: WorkOrderRecord
    incident: IncidentRecord
    asset: AssetRecord
    site: AssetSiteLinkRecord | None
    assignment: WorkOrderAssignmentRecord


class WorkOrderOfflinePackService:
    """Create and revalidate data-minimized offline read snapshots."""

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def create(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_work_order_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> WorkOrderOfflinePackResult:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            context = self._load_context(
                session,
                identity,
                work_order_id,
                request_id=request_id,
                for_update=True,
            )
            if context.work.version != expected_work_order_version:
                raise WorkOrderConflict("version_conflict", context.work.version)
            binding = _request_binding(context)
            request_digest = _digest(binding)
            existing = session.scalar(
                select(WorkOrderOfflinePackRecord).where(
                    WorkOrderOfflinePackRecord.tenant_id == identity.tenant_id,
                    WorkOrderOfflinePackRecord.subject_id == identity.subject_id,
                    WorkOrderOfflinePackRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.request_digest != request_digest:
                    raise WorkOrderConflict(
                        "idempotency_payload_mismatch",
                        context.work.version,
                    )
                _validate_snapshot_source(session, identity, context, existing)
                return WorkOrderOfflinePackResult(pack=_view(existing), created=False)

            now = datetime.now(UTC)
            active = session.scalars(
                select(WorkOrderOfflinePackRecord)
                .where(
                    WorkOrderOfflinePackRecord.tenant_id == identity.tenant_id,
                    WorkOrderOfflinePackRecord.subject_id == identity.subject_id,
                    WorkOrderOfflinePackRecord.work_order_id == work_order_id,
                    WorkOrderOfflinePackRecord.status == "ACTIVE",
                )
                .with_for_update()
            )
            for prior in active:
                prior.status = "SUPERSEDED"
                prior.version += 1
                prior.updated_at = now

            snapshot = self._snapshot(session, identity, context, request_id=request_id)
            record = WorkOrderOfflinePackRecord(
                pack_id=f"offline-pack-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                subject_id=identity.subject_id,
                work_order_id=work_order_id,
                work_order_version=context.work.version,
                asset_id=context.asset.asset_id,
                asset_version=context.asset.version,
                assignment_id=context.assignment.assignment_id,
                assignment_assigned_at=_utc(context.assignment.assigned_at),
                schema_version=OFFLINE_PACK_SCHEMA_VERSION,
                snapshot_json=snapshot,
                content_hash=_digest(snapshot),
                idempotency_key=idempotency_key,
                request_digest=request_digest,
                status="ACTIVE",
                issued_at=now,
                expires_at=now + OFFLINE_PACK_TTL,
                revoked_at=None,
                revoked_by_subject_id=None,
                revocation_reason=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            return WorkOrderOfflinePackResult(pack=_view(record), created=True)

    def current(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> WorkOrderOfflinePackView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            context = self._load_context(
                session,
                identity,
                work_order_id,
                request_id=request_id,
            )
            record = session.scalar(
                select(WorkOrderOfflinePackRecord)
                .where(
                    WorkOrderOfflinePackRecord.tenant_id == identity.tenant_id,
                    WorkOrderOfflinePackRecord.subject_id == identity.subject_id,
                    WorkOrderOfflinePackRecord.work_order_id == work_order_id,
                    WorkOrderOfflinePackRecord.status == "ACTIVE",
                )
                .order_by(
                    WorkOrderOfflinePackRecord.issued_at.desc(),
                    WorkOrderOfflinePackRecord.pack_id.desc(),
                )
            )
            if record is None or _utc(record.expires_at) <= datetime.now(UTC):
                raise OfflinePackNotAvailable
            if (
                record.work_order_version != context.work.version
                or record.asset_id != context.asset.asset_id
                or record.asset_version != context.asset.version
                or record.assignment_id != context.assignment.assignment_id
                or _utc(record.assignment_assigned_at) != _utc(context.assignment.assigned_at)
            ):
                raise WorkOrderConflict("offline_pack_binding_changed", context.work.version)
            _validate_snapshot_source(session, identity, context, record)
            return _view(record)

    def revoke(
        self,
        identity: IdentityContext,
        work_order_id: str,
        pack_id: str,
        *,
        expected_pack_version: int,
        reason: str,
        request_id: str,
    ) -> WorkOrderOfflinePackView:
        with work_order_read_transaction(
            self._database,
            identity,
            work_order_id,
            request_id=request_id,
        ) as session:
            context = self._load_context(
                session,
                identity,
                work_order_id,
                request_id=request_id,
                for_update=True,
            )
            record = session.scalar(
                select(WorkOrderOfflinePackRecord)
                .where(
                    WorkOrderOfflinePackRecord.tenant_id == identity.tenant_id,
                    WorkOrderOfflinePackRecord.subject_id == identity.subject_id,
                    WorkOrderOfflinePackRecord.work_order_id == work_order_id,
                    WorkOrderOfflinePackRecord.pack_id == pack_id,
                )
                .with_for_update()
            )
            if record is None:
                raise OfflinePackNotAvailable
            _validate_snapshot_source(session, identity, context, record)
            if record.status == "REVOKED":
                if (
                    record.revoked_by_subject_id == identity.subject_id
                    and record.revocation_reason == reason
                ):
                    return _view(record)
                raise WorkOrderConflict("offline_pack_revoke_conflict", record.version)
            if record.version != expected_pack_version:
                raise WorkOrderConflict("offline_pack_version_conflict", record.version)
            if record.status != "ACTIVE":
                raise WorkOrderConflict("offline_pack_not_active", record.version)
            now = datetime.now(UTC)
            record.status = "REVOKED"
            record.revoked_at = now
            record.revoked_by_subject_id = identity.subject_id
            record.revocation_reason = reason
            record.version += 1
            record.updated_at = now
            session.flush()
            return _view(record)

    def _load_context(
        self,
        session: Session,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
        for_update: bool = False,
    ) -> _OfflinePackContext:
        statement = (
            select(
                WorkOrderRecord,
                IncidentRecord,
                AssetRecord,
                AssetSiteLinkRecord,
            )
            .join(
                IncidentRecord,
                IncidentRecord.incident_id == WorkOrderRecord.incident_id,
            )
            .join(AssetRecord, AssetRecord.asset_id == IncidentRecord.asset_id)
            .outerjoin(
                AssetSiteLinkRecord,
                AssetSiteLinkRecord.asset_id == AssetRecord.asset_id,
            )
            .where(
                WorkOrderRecord.tenant_id == identity.tenant_id,
                IncidentRecord.tenant_id == identity.tenant_id,
                AssetRecord.tenant_id == identity.tenant_id,
                WorkOrderRecord.work_order_id == work_order_id,
            )
        )
        if for_update:
            statement = statement.with_for_update(of=WorkOrderRecord)
        row = session.execute(statement).tuples().one_or_none()
        if row is None:
            raise WorkOrderNotVisible
        work, incident, asset, site = row
        self._require(
            identity,
            Action.EXECUTE_WORK_ORDER,
            work_order_id,
            asset.asset_id,
            site.site_id if site is not None else None,
            request_id,
        )
        if work.assigned_subject_id != identity.subject_id:
            raise WorkOrderConflict("assignee_mismatch", work.version)
        if work.status not in OFFLINE_PACK_EXECUTABLE_STATUSES:
            raise WorkOrderConflict("offline_pack_illegal_work_order_status", work.version)
        assignment = session.scalar(
            select(WorkOrderAssignmentRecord)
            .where(
                WorkOrderAssignmentRecord.tenant_id == identity.tenant_id,
                WorkOrderAssignmentRecord.work_order_id == work_order_id,
                WorkOrderAssignmentRecord.assignee_subject_id == identity.subject_id,
            )
            .order_by(
                WorkOrderAssignmentRecord.assigned_at.desc(),
                WorkOrderAssignmentRecord.assignment_id.desc(),
            )
        )
        if assignment is None:
            raise WorkOrderConflict("offline_pack_assignment_missing", work.version)
        return _OfflinePackContext(
            work=work,
            incident=incident,
            asset=asset,
            site=site,
            assignment=assignment,
        )

    def _snapshot(
        self,
        session: Session,
        identity: IdentityContext,
        context: _OfflinePackContext,
        *,
        request_id: str,
    ) -> dict[str, Any]:
        diagnosis = session.scalar(
            select(DiagnosisRunRecord)
            .where(
                DiagnosisRunRecord.tenant_id == identity.tenant_id,
                DiagnosisRunRecord.incident_id == context.incident.incident_id,
                DiagnosisRunRecord.status.in_({"COMPLETED", "NEEDS_INFORMATION"}),
            )
            .order_by(
                DiagnosisRunRecord.updated_at.desc(),
                DiagnosisRunRecord.diagnosis_run_id.desc(),
            )
        )
        diagnosis_snapshot: dict[str, Any] | None = None
        citations: list[dict[str, Any]] = []
        if diagnosis is not None:
            self._require(
                identity,
                Action.READ_DIAGNOSIS,
                diagnosis.diagnosis_run_id,
                context.asset.asset_id,
                context.site.site_id if context.site is not None else None,
                request_id,
            )
            diagnosis_snapshot = _project_diagnosis(diagnosis)
            citation_ids = diagnosis_snapshot["citation_ids"]
            if citation_ids:
                self._require(
                    identity,
                    Action.READ_CITATION,
                    context.incident.incident_id,
                    context.asset.asset_id,
                    context.site.site_id if context.site is not None else None,
                    request_id,
                )
                rows = session.scalars(
                    select(CitationAnchorRecord).where(
                        CitationAnchorRecord.tenant_id == identity.tenant_id,
                        CitationAnchorRecord.citation_id.in_(citation_ids),
                    )
                )
                by_id = {row.citation_id: row for row in rows}
                citations = [
                    _project_citation(by_id[citation_id])
                    for citation_id in citation_ids
                    if citation_id in by_id
                ]
        return {
            "work_order": {
                "work_order_id": context.work.work_order_id,
                "incident_id": context.work.incident_id,
                "status": context.work.status,
                "priority": context.work.priority,
                "sla_due_at": _optional_iso(context.work.sla_due_at),
                "service_window_start": _optional_iso(context.work.service_window_start),
                "service_window_end": _optional_iso(context.work.service_window_end),
                "version": context.work.version,
            },
            "asset": {
                "asset_id": context.asset.asset_id,
                "display_name": context.asset.display_name,
                "model_code": context.asset.model_code,
                "serial_number": context.asset.serial_number,
                "lifecycle_status": context.asset.lifecycle_status,
                "version": context.asset.version,
            },
            "site": (
                {
                    "site_id": context.site.site_id,
                    "site_name": context.site.site_name,
                }
                if context.site is not None
                else None
            ),
            "incident": {
                "incident_id": context.incident.incident_id,
                "description": _bounded_text(
                    context.incident.description,
                    MAX_DIAGNOSIS_TEXT,
                ),
                "severity": context.incident.severity,
                "category": context.incident.category,
            },
            "diagnosis": diagnosis_snapshot,
            "citations": citations,
            "legal_actions": ["READ_OFFLINE_SNAPSHOT"],
        }

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        asset_id: str,
        site_id: str | None,
        request_id: str,
    ) -> None:
        if asset_id in identity.asset_ids:
            resource = ResourceContext(identity.tenant_id, resource_id, asset_id=asset_id)
        elif site_id is not None and site_id in identity.site_ids:
            resource = ResourceContext(identity.tenant_id, resource_id, site_id=site_id)
        else:
            resource = ResourceContext(identity.tenant_id, resource_id, asset_id=asset_id)
        self._authorizer.require(identity, action, resource, request_id=request_id)


def _validate_snapshot_source(
    session: Session,
    identity: IdentityContext,
    context: _OfflinePackContext,
    record: WorkOrderOfflinePackRecord,
) -> None:
    if record.content_hash != _digest(record.snapshot_json):
        raise WorkOrderConflict("offline_pack_integrity_failed", record.version)
    snapshot = record.snapshot_json
    work, incident, diagnosis = (
        snapshot.get("work_order"),
        snapshot.get("incident"),
        snapshot.get("diagnosis"),
    )
    if (
        not isinstance(work, dict)
        or work.get("work_order_id") != context.work.work_order_id
        or work.get("incident_id") != context.incident.incident_id
        or not isinstance(incident, dict)
        or incident.get("incident_id") != context.incident.incident_id
    ):
        raise ReviewIsolationConflict("source_binding_invalid")
    if diagnosis is not None:
        if not isinstance(diagnosis, dict) or not isinstance(
            diagnosis.get("diagnosis_run_id"), str
        ):
            raise ReviewIsolationConflict("source_unavailable")
        source = resolve_source(session, identity.tenant_id, diagnosis["diagnosis_run_id"])
        if source.incident_id != context.incident.incident_id:
            raise ReviewIsolationConflict("source_binding_invalid")


def _request_binding(context: _OfflinePackContext) -> dict[str, Any]:
    return {
        "schema_version": OFFLINE_PACK_SCHEMA_VERSION,
        "ttl_seconds": int(OFFLINE_PACK_TTL.total_seconds()),
        "work_order_id": context.work.work_order_id,
        "work_order_version": context.work.version,
        "asset_id": context.asset.asset_id,
        "asset_version": context.asset.version,
        "assignment_id": context.assignment.assignment_id,
        "assignment_assigned_at": _utc(context.assignment.assigned_at).isoformat(),
    }


def _project_diagnosis(record: DiagnosisRunRecord) -> dict[str, Any]:
    report = record.report if isinstance(record.report, dict) else {}
    conclusion = _bounded_text(
        report.get("conclusion") or report.get("summary"),
        MAX_DIAGNOSIS_TEXT,
    )
    return {
        "diagnosis_run_id": record.diagnosis_run_id,
        "status": record.status,
        "version": record.version,
        "conclusion": conclusion,
        "next_checks": _bounded_string_list(report.get("next_checks")),
        "recommended_actions": _bounded_string_list(report.get("recommended_actions")),
        "confidence": _bounded_confidence(report.get("confidence")),
        "citation_ids": _citation_ids(report.get("citation_ids")),
    }


def _project_citation(record: CitationAnchorRecord) -> dict[str, Any]:
    return {
        "citation_id": record.citation_id,
        "anchor_kind": record.anchor_kind,
        "page_number": record.page_number,
        "excerpt": _bounded_text(record.excerpt, MAX_CITATION_EXCERPT),
        "excerpt_checksum": record.excerpt_checksum,
    }


def _bounded_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized[:limit]


def _bounded_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    projected: list[str] = []
    for item in value:
        text = _bounded_text(item, MAX_DIAGNOSIS_LIST_TEXT)
        if text is not None and text not in projected:
            projected.append(text)
        if len(projected) == MAX_DIAGNOSIS_LIST_ITEMS:
            break
    return projected


def _citation_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and 0 < len(item) <= 128 and item not in result:
            result.append(item)
        if len(result) == MAX_CITATIONS:
            break
    return result


def _bounded_confidence(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    projected = float(value)
    if not math.isfinite(projected) or not 0.0 <= projected <= 1.0:
        return None
    return projected


def _view(record: WorkOrderOfflinePackRecord) -> WorkOrderOfflinePackView:
    active = (
        record.status == "ACTIVE"
        and record.revoked_at is None
        and _utc(record.expires_at) > datetime.now(UTC)
    )
    return WorkOrderOfflinePackView(
        pack_id=record.pack_id,
        status=record.status,
        schema_version=record.schema_version,
        subject_id=record.subject_id,
        work_order_id=record.work_order_id,
        work_order_version=record.work_order_version,
        asset_id=record.asset_id,
        asset_version=record.asset_version,
        assignment_id=record.assignment_id,
        assignment_assigned_at=_utc(record.assignment_assigned_at),
        snapshot=record.snapshot_json,
        content_hash=record.content_hash,
        issued_at=_utc(record.issued_at),
        expires_at=_utc(record.expires_at),
        revoked_at=_utc(record.revoked_at) if record.revoked_at is not None else None,
        revoked_by_subject_id=record.revoked_by_subject_id,
        revocation_reason=record.revocation_reason,
        version=record.version,
        legal_actions=["REVOKE"] if active else [],
    )


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def _optional_iso(value: datetime | None) -> str | None:
    return _utc(value).isoformat() if value is not None else None


__all__ = [
    "AuthorizationDenied",
    "OfflinePackNotAvailable",
    "WorkOrderOfflinePackResult",
    "WorkOrderOfflinePackService",
    "WorkOrderOfflinePackView",
]
