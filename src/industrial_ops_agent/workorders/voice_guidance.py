"""Authoritative field WorkOrder projection and realtime voice binding checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.application.speech import TTS_SAFETY_WARNING
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.maintenance_planning.review_isolation import work_order_read_transaction
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetSiteLinkRecord,
    DiagnosisRunRecord,
    IncidentRecord,
    WorkOrderRecord,
)
from industrial_ops_agent.workorders.service import WorkOrderConflict, WorkOrderNotVisible

FIELD_VOICE_EXECUTABLE_STATUSES = frozenset({"ACCEPTED", "IN_PROGRESS", "ON_HOLD"})
FIELD_VOICE_DIAGNOSIS_STATUSES = frozenset({"COMPLETED", "NEEDS_INFORMATION"})
FIELD_REDIAGNOSIS_ACTIVE_STATUSES = frozenset({"QUEUED", "RUNNING"})


@dataclass(frozen=True, slots=True)
class FieldVoiceDiagnosis:
    diagnosis_run_id: str
    status: str
    version: int
    conclusion: str | None
    next_checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FieldRediagnosisProgress:
    diagnosis_run_id: str
    status: str
    version: int
    source_diagnosis_run_id: str
    source_work_order_version: int
    field_entry_count: int
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class FieldVoiceGuidance:
    work_order_id: str
    work_order_version: int
    work_order_status: str
    incident_id: str
    diagnosis: FieldVoiceDiagnosis | None
    field_reanalysis: FieldRediagnosisProgress | None
    safety_warning: str
    legal_actions: tuple[str, ...]


class FieldVoiceGuidanceService:
    """Resolve the sole server-approved diagnosis usable by a field voice client."""

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def get(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> FieldVoiceGuidance:
        with work_order_read_transaction(
            self._database, identity, work_order_id, request_id=request_id
        ) as session:
            work, incident, site = self._load(
                session,
                identity,
                work_order_id,
                request_id=request_id,
            )
            diagnosis = _latest_diagnosis(session, identity.tenant_id, incident.incident_id)
            field_reanalysis = _latest_field_rediagnosis(
                session,
                identity.tenant_id,
                incident.incident_id,
                work.work_order_id,
            )
            if diagnosis is not None:
                self._require(
                    identity,
                    Action.READ_DIAGNOSIS,
                    diagnosis.diagnosis_run_id,
                    incident.asset_id,
                    site.site_id if site is not None else None,
                    request_id,
                )
            if (
                field_reanalysis is not None
                and (
                    diagnosis is None
                    or field_reanalysis.diagnosis_run_id != diagnosis.diagnosis_run_id
                )
            ):
                self._require(
                    identity,
                    Action.READ_DIAGNOSIS,
                    field_reanalysis.diagnosis_run_id,
                    incident.asset_id,
                    site.site_id if site is not None else None,
                    request_id,
                )
            return _guidance(work, incident, diagnosis, field_reanalysis)

    def require_exact(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_work_order_version: int,
        diagnosis_run_id: str,
        require_completed: bool,
        request_id: str,
    ) -> FieldVoiceGuidance:
        guidance = self.get(identity, work_order_id, request_id=request_id)
        if guidance.work_order_version != expected_work_order_version:
            raise WorkOrderConflict("field_voice_version_conflict", guidance.work_order_version)
        diagnosis = guidance.diagnosis
        if diagnosis is None or diagnosis.diagnosis_run_id != diagnosis_run_id:
            raise WorkOrderConflict(
                "field_voice_diagnosis_changed", guidance.work_order_version
            )
        if require_completed and diagnosis.status != "COMPLETED":
            raise WorkOrderConflict(
                "field_voice_diagnosis_not_speakable", guidance.work_order_version
            )
        return guidance

    def _load(
        self,
        session: Session,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> tuple[WorkOrderRecord, IncidentRecord, AssetSiteLinkRecord | None]:
        row = session.execute(
            select(WorkOrderRecord, IncidentRecord)
            .join(IncidentRecord, IncidentRecord.incident_id == WorkOrderRecord.incident_id)
            .where(
                WorkOrderRecord.tenant_id == identity.tenant_id,
                IncidentRecord.tenant_id == identity.tenant_id,
                WorkOrderRecord.work_order_id == work_order_id,
            )
        ).tuples().one_or_none()
        if row is None:
            raise WorkOrderNotVisible
        work, incident = row
        site = session.scalar(
            select(AssetSiteLinkRecord).where(
                AssetSiteLinkRecord.tenant_id == identity.tenant_id,
                AssetSiteLinkRecord.asset_id == incident.asset_id,
            )
        )
        self._require(
            identity,
            Action.EXECUTE_WORK_ORDER,
            work_order_id,
            incident.asset_id,
            site.site_id if site is not None else None,
            request_id,
        )
        if work.assigned_subject_id != identity.subject_id:
            raise WorkOrderNotVisible
        if work.status not in FIELD_VOICE_EXECUTABLE_STATUSES:
            raise WorkOrderConflict("field_voice_work_order_not_executable", work.version)
        return work, incident, site

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        asset_id: str,
        site_id: str | None,
        request_id: str,
    ) -> None:
        resource = ResourceContext(
            identity.tenant_id,
            resource_id,
            site_id=site_id if site_id in identity.site_ids else None,
            asset_id=asset_id if site_id not in identity.site_ids else None,
        )
        try:
            self._authorizer.require(identity, action, resource, request_id=request_id)
        except AuthorizationDenied as exc:
            raise WorkOrderNotVisible from exc


def field_voice_binding_current(
    session: Session,
    *,
    tenant_id: str,
    subject_id: str,
    work_order_id: str | None,
    work_order_version: int | None,
    incident_id: str,
    diagnosis_run_id: str | None,
) -> bool:
    """Revalidate a stored FIELD binding without granting API identity powers."""

    if work_order_id is None and work_order_version is None:
        return True
    if work_order_id is None or work_order_version is None or diagnosis_run_id is None:
        return False
    work = session.scalar(
        select(WorkOrderRecord)
        .where(
            WorkOrderRecord.tenant_id == tenant_id,
            WorkOrderRecord.work_order_id == work_order_id,
        )
        .with_for_update()
    )
    if (
        work is None
        or work.incident_id != incident_id
        or work.assigned_subject_id != subject_id
        or work.status not in FIELD_VOICE_EXECUTABLE_STATUSES
        or work.version != work_order_version
    ):
        return False
    latest = _latest_diagnosis(session, tenant_id, incident_id)
    return latest is not None and latest.diagnosis_run_id == diagnosis_run_id


def _latest_diagnosis(
    session: Session,
    tenant_id: str,
    incident_id: str,
) -> DiagnosisRunRecord | None:
    return session.scalar(
        select(DiagnosisRunRecord)
        .where(
            DiagnosisRunRecord.tenant_id == tenant_id,
            DiagnosisRunRecord.incident_id == incident_id,
            DiagnosisRunRecord.status.in_(FIELD_VOICE_DIAGNOSIS_STATUSES),
        )
        .order_by(
            DiagnosisRunRecord.updated_at.desc(),
            DiagnosisRunRecord.diagnosis_run_id.desc(),
        )
    )


def _latest_field_rediagnosis(
    session: Session,
    tenant_id: str,
    incident_id: str,
    work_order_id: str,
) -> DiagnosisRunRecord | None:
    candidates = session.scalars(
        select(DiagnosisRunRecord)
        .where(
            DiagnosisRunRecord.tenant_id == tenant_id,
            DiagnosisRunRecord.incident_id == incident_id,
        )
        .order_by(
            DiagnosisRunRecord.updated_at.desc(),
            DiagnosisRunRecord.diagnosis_run_id.desc(),
        )
    )
    for candidate in candidates:
        if _field_rediagnosis_context(candidate, work_order_id) is not None:
            return candidate
    return None


def _field_rediagnosis_context(
    diagnosis: DiagnosisRunRecord,
    work_order_id: str,
) -> tuple[str, int, int] | None:
    manifest = diagnosis.manifest
    context = manifest.get("input_context") if isinstance(manifest, dict) else None
    if not isinstance(context, dict):
        return None
    source_run_id = context.get("source_diagnosis_run_id")
    source_work_order_version = context.get("source_work_order_version")
    field_entries = context.get("field_entries")
    if (
        context.get("kind") != "human_confirmed_field_observations"
        or context.get("source_work_order_id") != work_order_id
        or not isinstance(source_run_id, str)
        or not source_run_id
        or not isinstance(source_work_order_version, int)
        or isinstance(source_work_order_version, bool)
        or source_work_order_version < 1
        or not isinstance(field_entries, list)
        or not 1 <= len(field_entries) <= 20
        or not all(isinstance(item, dict) for item in field_entries)
    ):
        return None
    return source_run_id, source_work_order_version, len(field_entries)


def _guidance(
    work: WorkOrderRecord,
    incident: IncidentRecord,
    diagnosis: DiagnosisRunRecord | None,
    field_reanalysis: DiagnosisRunRecord | None,
) -> FieldVoiceGuidance:
    projected: FieldVoiceDiagnosis | None = None
    reanalysis_progress: FieldRediagnosisProgress | None = None
    legal_actions: list[str] = []
    if field_reanalysis is not None:
        context = _field_rediagnosis_context(field_reanalysis, work.work_order_id)
        if context is not None:
            source_run_id, source_work_order_version, field_entry_count = context
            reanalysis_progress = FieldRediagnosisProgress(
                diagnosis_run_id=field_reanalysis.diagnosis_run_id,
                status=field_reanalysis.status,
                version=field_reanalysis.version,
                source_diagnosis_run_id=source_run_id,
                source_work_order_version=source_work_order_version,
                field_entry_count=field_entry_count,
                updated_at=field_reanalysis.updated_at,
            )
    if diagnosis is not None:
        report = diagnosis.report if isinstance(diagnosis.report, dict) else {}
        projected = FieldVoiceDiagnosis(
            diagnosis_run_id=diagnosis.diagnosis_run_id,
            status=diagnosis.status,
            version=diagnosis.version,
            conclusion=_bounded_text(report.get("conclusion") or report.get("summary")),
            next_checks=_bounded_list(report.get("next_checks")),
        )
        legal_actions.append("START_REALTIME_VOICE")
        if (
            reanalysis_progress is None
            or reanalysis_progress.status not in FIELD_REDIAGNOSIS_ACTIVE_STATUSES
        ):
            legal_actions.append("REQUEST_FIELD_REANALYSIS")
        if diagnosis.status == "COMPLETED" and _report_speakable(report):
            legal_actions.append("SYNTHESIZE_VOICE_GUIDANCE")
    return FieldVoiceGuidance(
        work_order_id=work.work_order_id,
        work_order_version=work.version,
        work_order_status=work.status,
        incident_id=incident.incident_id,
        diagnosis=projected,
        field_reanalysis=reanalysis_progress,
        safety_warning=TTS_SAFETY_WARNING,
        legal_actions=tuple(legal_actions),
    )


def _report_speakable(report: dict[str, Any]) -> bool:
    conclusion = report.get("conclusion")
    checks = report.get("next_checks")
    return (
        report.get("status") == "COMPLETED"
        and isinstance(conclusion, str)
        and bool(conclusion.strip())
        and report.get("contradictions") in ([], ())
        and report.get("missing_information") in ([], ())
        and isinstance(checks, (list, tuple))
        and all(isinstance(item, str) and bool(item.strip()) for item in checks)
    )


def _bounded_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:2_000] if value else None


def _bounded_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        text
        for item in value[:20]
        if (text := _bounded_text(item)) is not None
    )


__all__ = [
    "FIELD_VOICE_EXECUTABLE_STATUSES",
    "FieldRediagnosisProgress",
    "FieldVoiceDiagnosis",
    "FieldVoiceGuidance",
    "FieldVoiceGuidanceService",
    "field_voice_binding_current",
]
