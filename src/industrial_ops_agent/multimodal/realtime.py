"""Tenant-bound WebRTC control plane with one-time tokens and renewable leases."""

from __future__ import annotations

import base64
import binascii
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha1, sha256
from secrets import token_urlsafe
from urllib.parse import urlparse
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.maintenance_planning.review_isolation import (
    ReviewIsolationConflict,
    incident_read_transaction,
    lock_subject,
    record_incident_disclosure,
    resolve_source,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    DiagnosisRunRecord,
    IncidentRecord,
    RealtimeMediaSessionRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.workorders.expert_collaboration import (
    ExpertCollaborationConflict,
    ExpertCollaborationNotVisible,
    ExpertCollaborationService,
    expert_collaboration_binding_current,
)
from industrial_ops_agent.workorders.service import WorkOrderConflict, WorkOrderNotVisible
from industrial_ops_agent.workorders.voice_guidance import (
    FieldVoiceGuidanceService,
    field_voice_binding_current,
)

ACTIVE_STATUSES = frozenset({"ISSUED", "CONNECTED", "RECONNECTING"})


class RealtimeSessionError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class IceServer:
    urls: tuple[str, ...]
    username: str | None = None
    credential: str | None = None


@dataclass(frozen=True, slots=True)
class RealtimeSession:
    session_id: str
    incident_id: str
    diagnosis_run_id: str | None
    work_order_id: str | None
    work_order_version: int | None
    expert_collaboration_id: str | None
    participant_subject_id: str
    status: str
    media_kind: str
    token_expires_at: datetime
    lease_expires_at: datetime
    max_expires_at: datetime
    reconnect_count: int
    version: int


@dataclass(frozen=True, slots=True)
class RealtimeSessionGrant:
    session: RealtimeSession
    access_token: str
    signaling_url: str
    ice_servers: tuple[IceServer, ...]
    heartbeat_interval_seconds: int


@dataclass(frozen=True, slots=True)
class RealtimeMediaAdmission:
    session: RealtimeSession
    media_access_token: str


@dataclass(frozen=True, slots=True)
class RealtimeMediaContext:
    tenant_context: TenantContext
    session: RealtimeSession


class RealtimeSessionService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        *,
        signaling_url: str,
        stun_urls: tuple[str, ...],
        turn_url: str | None = None,
        turn_shared_secret: str | None = None,
        token_ttl_seconds: int = 60,
        lease_seconds: int = 90,
        max_session_seconds: int = 3_600,
        heartbeat_interval_seconds: int = 20,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._signaling_url = _signaling_url(signaling_url)
        self._stun_urls = tuple(_ice_url(value, "stun") for value in stun_urls)
        self._turn_url = _ice_url(turn_url, "turn") if turn_url else None
        if self._turn_url and not turn_shared_secret:
            raise ValueError("TURN shared secret is required when TURN is configured")
        self._turn_shared_secret = turn_shared_secret
        self._token_ttl = timedelta(seconds=token_ttl_seconds)
        self._lease = timedelta(seconds=lease_seconds)
        self._maximum = timedelta(seconds=max_session_seconds)
        self._heartbeat_interval_seconds = heartbeat_interval_seconds

    def create(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        diagnosis_run_id: str | None,
        work_order_id: str | None = None,
        work_order_version: int | None = None,
        request_id: str,
    ) -> RealtimeSessionGrant:
        if (work_order_id is None) != (work_order_version is None):
            raise RealtimeSessionError("field_realtime_binding_incomplete")
        if work_order_id is not None:
            if diagnosis_run_id is None or work_order_version is None:
                raise RealtimeSessionError("field_realtime_diagnosis_required")
            try:
                guidance = FieldVoiceGuidanceService(
                    self._database, self._authorizer
                ).require_exact(
                    identity,
                    work_order_id,
                    expected_work_order_version=work_order_version,
                    diagnosis_run_id=diagnosis_run_id,
                    require_completed=False,
                    request_id=request_id,
                )
            except WorkOrderNotVisible as exc:
                raise ResourceNotVisible from exc
            except WorkOrderConflict as exc:
                raise RealtimeSessionError(exc.reason) from exc
            if guidance.incident_id != incident_id:
                raise ResourceNotVisible
        self._authorize_incident(
            identity,
            incident_id,
            diagnosis_run_id=diagnosis_run_id,
            request_id=request_id,
        )
        now = datetime.now(UTC)
        max_expires_at = min(now + self._maximum, identity.expires_at)
        if max_expires_at <= now + timedelta(seconds=5):
            raise RealtimeSessionError("identity_expiry_too_close")
        session_id = f"realtime-{uuid4().hex}"
        access_token = _new_token(identity.tenant_id)
        token_expires_at = min(now + self._token_ttl, max_expires_at)
        lease_expires_at = min(now + self._lease, max_expires_at)
        with incident_read_transaction(
            self._database, identity, (incident_id,), request_id=request_id
        ) as session:
            if work_order_id is not None and not field_voice_binding_current(
                session,
                tenant_id=identity.tenant_id,
                subject_id=identity.subject_id,
                work_order_id=work_order_id,
                work_order_version=work_order_version,
                incident_id=incident_id,
                diagnosis_run_id=diagnosis_run_id,
            ):
                raise RealtimeSessionError("realtime_field_binding_changed")
            existing = list(
                session.scalars(
                    select(RealtimeMediaSessionRecord)
                    .where(
                        RealtimeMediaSessionRecord.tenant_id == identity.tenant_id,
                        RealtimeMediaSessionRecord.subject_id == identity.subject_id,
                        RealtimeMediaSessionRecord.incident_id == incident_id,
                        RealtimeMediaSessionRecord.status.in_(ACTIVE_STATUSES),
                    )
                    .with_for_update()
                )
            )
            for record in existing:
                record.status = "ENDED"
                record.ended_reason = "superseded_by_new_session"
                record.version += 1
            record = RealtimeMediaSessionRecord(
                session_id=session_id,
                tenant_id=identity.tenant_id,
                incident_id=incident_id,
                work_order_id=work_order_id,
                work_order_version=work_order_version,
                expert_collaboration_id=None,
                diagnosis_run_id=diagnosis_run_id,
                subject_id=identity.subject_id,
                status="ISSUED",
                media_kind="audio",
                token_digest=_token_digest(access_token),
                token_expires_at=token_expires_at,
                token_consumed_at=None,
                media_token_digest=None,
                media_token_issued_at=None,
                lease_expires_at=lease_expires_at,
                max_expires_at=max_expires_at,
                reconnect_count=0,
                ended_reason=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            return self._grant(record, access_token)

    def create_expert_collaboration(
        self,
        identity: IdentityContext,
        collaboration_id: str,
        *,
        request_id: str,
    ) -> RealtimeSessionGrant:
        try:
            binding = ExpertCollaborationService(
                self._database, self._authorizer
            ).require_active_participant(
                identity,
                collaboration_id,
                request_id=request_id,
            )
        except ExpertCollaborationNotVisible as exc:
            raise ResourceNotVisible from exc
        except ExpertCollaborationConflict as exc:
            raise RealtimeSessionError(exc.reason) from exc
        self._authorize_incident(
            identity,
            binding.incident_id,
            diagnosis_run_id=None,
            request_id=request_id,
        )
        now = datetime.now(UTC)
        max_expires_at = min(now + self._maximum, identity.expires_at)
        if max_expires_at <= now + timedelta(seconds=5):
            raise RealtimeSessionError("identity_expiry_too_close")
        session_id = f"realtime-{uuid4().hex}"
        access_token = _new_token(identity.tenant_id)
        token_expires_at = min(now + self._token_ttl, max_expires_at)
        lease_expires_at = min(now + self._lease, max_expires_at)
        with incident_read_transaction(
            self._database, identity, (binding.incident_id,), request_id=request_id
        ) as session:
            if not expert_collaboration_binding_current(
                session,
                tenant_id=identity.tenant_id,
                collaboration_id=collaboration_id,
                participant_subject_id=identity.subject_id,
            ):
                raise RealtimeSessionError("realtime_expert_collaboration_changed")
            existing = list(
                session.scalars(
                    select(RealtimeMediaSessionRecord)
                    .where(
                        RealtimeMediaSessionRecord.tenant_id == identity.tenant_id,
                        RealtimeMediaSessionRecord.subject_id == identity.subject_id,
                        RealtimeMediaSessionRecord.expert_collaboration_id == collaboration_id,
                        RealtimeMediaSessionRecord.status.in_(ACTIVE_STATUSES),
                    )
                    .with_for_update()
                )
            )
            for previous in existing:
                previous.status = "ENDED"
                previous.ended_reason = "superseded_by_new_session"
                previous.media_token_digest = None
                previous.version += 1
            record = RealtimeMediaSessionRecord(
                session_id=session_id,
                tenant_id=identity.tenant_id,
                incident_id=binding.incident_id,
                work_order_id=binding.work_order_id,
                work_order_version=binding.work_order_version,
                expert_collaboration_id=collaboration_id,
                diagnosis_run_id=None,
                subject_id=identity.subject_id,
                status="ISSUED",
                media_kind="audio",
                token_digest=_token_digest(access_token),
                token_expires_at=token_expires_at,
                token_consumed_at=None,
                media_token_digest=None,
                media_token_issued_at=None,
                lease_expires_at=lease_expires_at,
                max_expires_at=max_expires_at,
                reconnect_count=0,
                ended_reason=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            return self._grant(record, access_token)

    def consume(self, session_id: str, access_token: str) -> RealtimeMediaAdmission:
        context = _context_from_token(access_token)
        media_access_token = _new_token(context.tenant_id, version="rtm1")
        error: str | None = None
        result: RealtimeMediaAdmission | None = None
        with self._database.transaction(context) as session:
            actor = _capability_subject(
                session, context, session_id, access_token, media=False
            )
            record = _locked_session(session, context.tenant_id, session_id)
            now = datetime.now(UTC)
            if (
                record is None or record.subject_id != actor.subject_id
                or not hmac.compare_digest(record.token_digest, _token_digest(access_token))
            ):
                error = "realtime_session_token_invalid"
            elif record.status not in {"ISSUED", "RECONNECTING"}:
                error = "realtime_session_token_consumed"
            elif now >= _aware(record.token_expires_at) or now >= _aware(record.max_expires_at):
                record.status = "EXPIRED"
                record.ended_reason = "token_expired"
                record.version += 1
                error = "realtime_session_token_expired"
            elif not _binding_current(record, session):
                error = _binding_changed_reason(record)
                _end_for_binding_drift(record)
            else:
                _record_session_disclosure(
                    session, actor, record, request_id=f"realtime-consume:{session_id}"
                )
                record.status = "CONNECTED"
                record.token_consumed_at = now
                record.media_token_digest = _token_digest(media_access_token)
                record.media_token_issued_at = now
                record.lease_expires_at = min(now + self._lease, _aware(record.max_expires_at))
                record.version += 1
                result = RealtimeMediaAdmission(_view(record), media_access_token)
        if error is not None or result is None:
            raise RealtimeSessionError(error or "realtime_session_token_invalid")
        return result

    def reconnect(
        self,
        identity: IdentityContext,
        session_id: str,
        *,
        request_id: str,
    ) -> RealtimeSessionGrant:
        self._authorize_session(identity, session_id, request_id=request_id)
        access_token = _new_token(identity.tenant_id)
        error: str | None = None
        grant: RealtimeSessionGrant | None = None
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _locked_session(session, identity.tenant_id, session_id)
            now = datetime.now(UTC)
            if record is None or record.subject_id != identity.subject_id:
                raise ResourceNotVisible
            if not _binding_current(record, session):
                error = _binding_changed_reason(record)
                _end_for_binding_drift(record)
            elif record.status not in ACTIVE_STATUSES:
                error = "realtime_session_not_active"
            elif now >= _aware(record.lease_expires_at) or now >= _aware(record.max_expires_at):
                record.status = "EXPIRED"
                record.ended_reason = "lease_expired"
                record.version += 1
                error = "realtime_session_lease_expired"
            else:
                record.status = "RECONNECTING"
                record.token_digest = _token_digest(access_token)
                record.token_expires_at = min(now + self._token_ttl, _aware(record.max_expires_at))
                record.token_consumed_at = None
                record.media_token_digest = None
                record.media_token_issued_at = None
                record.reconnect_count += 1
                record.version += 1
                _record_session_disclosure(session, identity, record, request_id=request_id)
                grant = self._grant(record, access_token)
        if error is not None or grant is None:
            raise RealtimeSessionError(error or "realtime_session_not_active")
        return grant

    def heartbeat(
        self,
        identity: IdentityContext,
        session_id: str,
        *,
        request_id: str,
    ) -> RealtimeSession:
        self._authorize_session(identity, session_id, request_id=request_id)
        error: str | None = None
        result: RealtimeSession | None = None
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _locked_session(session, identity.tenant_id, session_id)
            now = datetime.now(UTC)
            if record is None or record.subject_id != identity.subject_id:
                raise ResourceNotVisible
            if not _binding_current(record, session):
                error = _binding_changed_reason(record)
                _end_for_binding_drift(record)
            elif record.status != "CONNECTED":
                error = "realtime_session_not_connected"
            elif now >= _aware(record.lease_expires_at) or now >= _aware(record.max_expires_at):
                record.status = "EXPIRED"
                record.ended_reason = "lease_expired"
                record.version += 1
                error = "realtime_session_lease_expired"
            else:
                record.lease_expires_at = min(now + self._lease, _aware(record.max_expires_at))
                record.version += 1
                _record_session_disclosure(session, identity, record, request_id=request_id)
                result = _view(record)
        if error is not None or result is None:
            raise RealtimeSessionError(error or "realtime_session_not_connected")
        return result

    def end(
        self,
        identity: IdentityContext,
        session_id: str,
        *,
        request_id: str,
        reason: str = "user_ended",
    ) -> RealtimeSession:
        self._authorize_session(identity, session_id, request_id=request_id)
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            record = _locked_session(session, identity.tenant_id, session_id)
            if record is None or record.subject_id != identity.subject_id:
                raise ResourceNotVisible
            if record.status != "ENDED":
                record.status = "ENDED"
                record.ended_reason = reason[:128]
                record.media_token_digest = None
                record.version += 1
            _record_session_disclosure(session, identity, record, request_id=request_id)
            return _view(record)

    def _authorize_incident(
        self,
        identity: IdentityContext,
        incident_id: str,
        *,
        diagnosis_run_id: str | None,
        request_id: str,
    ) -> None:
        with self._database.transaction(identity.tenant_context) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                )
            )
            if incident is None:
                raise ResourceNotVisible
            if diagnosis_run_id is not None:
                diagnosis = session.scalar(
                    select(DiagnosisRunRecord).where(
                        DiagnosisRunRecord.tenant_id == identity.tenant_id,
                        DiagnosisRunRecord.diagnosis_run_id == diagnosis_run_id,
                        DiagnosisRunRecord.incident_id == incident_id,
                    )
                )
                if diagnosis is None:
                    raise ResourceNotVisible
            asset_id = incident.asset_id
        try:
            self._authorizer.require(
                identity,
                Action.START_REALTIME_MEDIA,
                ResourceContext(
                    tenant_id=identity.tenant_id,
                    resource_id=incident_id,
                    asset_id=asset_id,
                ),
                request_id=request_id,
            )
        except AuthorizationDenied as exc:
            raise ResourceNotVisible from exc

    def _authorize_session(
        self,
        identity: IdentityContext,
        session_id: str,
        *,
        request_id: str,
    ) -> None:
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(RealtimeMediaSessionRecord).where(
                    RealtimeMediaSessionRecord.tenant_id == identity.tenant_id,
                    RealtimeMediaSessionRecord.session_id == session_id,
                )
            )
            if record is None:
                raise ResourceNotVisible
            incident_id = record.incident_id
            diagnosis_run_id = record.diagnosis_run_id
            expert_collaboration_id = record.expert_collaboration_id
        if expert_collaboration_id is not None:
            try:
                ExpertCollaborationService(
                    self._database, self._authorizer
                ).require_active_participant(
                    identity,
                    expert_collaboration_id,
                    request_id=request_id,
                )
            except ExpertCollaborationNotVisible as exc:
                raise ResourceNotVisible from exc
            except ExpertCollaborationConflict as exc:
                raise RealtimeSessionError("realtime_expert_collaboration_changed") from exc
            if record.subject_id != identity.subject_id:
                raise ResourceNotVisible
            return
        if record.subject_id != identity.subject_id:
            raise ResourceNotVisible
        self._authorize_incident(
            identity,
            incident_id,
            diagnosis_run_id=diagnosis_run_id,
            request_id=request_id,
        )

    def _grant(
        self,
        record: RealtimeMediaSessionRecord,
        access_token: str,
    ) -> RealtimeSessionGrant:
        ice_servers = [IceServer(self._stun_urls)] if self._stun_urls else []
        if self._turn_url and self._turn_shared_secret:
            username = f"{int(_aware(record.token_expires_at).timestamp())}:{record.session_id}"
            credential = base64.b64encode(
                hmac.new(
                    self._turn_shared_secret.encode(),
                    username.encode(),
                    sha1,
                ).digest()
            ).decode()
            ice_servers.append(IceServer((self._turn_url,), username, credential))
        return RealtimeSessionGrant(
            session=_view(record),
            access_token=access_token,
            signaling_url=self._signaling_url,
            ice_servers=tuple(ice_servers),
            heartbeat_interval_seconds=self._heartbeat_interval_seconds,
        )


def _capability_subject(
    session: Session,
    context: TenantContext,
    session_id: str,
    access_token: str,
    *,
    media: bool,
) -> TenantContext:
    """Authenticate before locking the bound person; never turn a capability into OIDC."""
    record = session.scalar(
        select(RealtimeMediaSessionRecord).where(
            RealtimeMediaSessionRecord.tenant_id == context.tenant_id,
            RealtimeMediaSessionRecord.session_id == session_id,
        )
    )
    invalid = "realtime_media_capability_invalid" if media else "realtime_session_token_invalid"
    if record is None:
        raise RealtimeSessionError(invalid)
    expected = record.media_token_digest if media else record.token_digest
    if expected is None or not hmac.compare_digest(expected, _token_digest(access_token)):
        raise RealtimeSessionError(invalid)
    actor = TenantContext(record.tenant_id, record.subject_id)
    lock_subject(session, actor)
    # The caller MUST reload and revalidate token/state/binding after waiting for the lock.
    return actor


def _record_session_disclosure(
    session: Session,
    actor: IdentityContext | TenantContext,
    record: RealtimeMediaSessionRecord,
    *,
    request_id: str,
) -> None:
    if record.tenant_id != actor.tenant_id or record.subject_id != actor.subject_id:
        raise ReviewIsolationConflict("source_binding_invalid")
    if record.diagnosis_run_id is not None:
        source = resolve_source(session, actor.tenant_id, record.diagnosis_run_id)
        if source.incident_id != record.incident_id:
            raise ReviewIsolationConflict("source_binding_invalid")
    record_incident_disclosure(
        session, actor, (record.incident_id,),
        resource_kind="realtime-session",
        resource_id=record.session_id,
        request_id=request_id,
    )


def _locked_session(
    session: Session,
    tenant_id: str,
    session_id: str,
) -> RealtimeMediaSessionRecord | None:
    return session.scalar(
        select(RealtimeMediaSessionRecord)
        .where(
            RealtimeMediaSessionRecord.tenant_id == tenant_id,
            RealtimeMediaSessionRecord.session_id == session_id,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )


def _view(record: RealtimeMediaSessionRecord) -> RealtimeSession:
    return RealtimeSession(
        session_id=record.session_id,
        incident_id=record.incident_id,
        diagnosis_run_id=record.diagnosis_run_id,
        work_order_id=record.work_order_id,
        work_order_version=record.work_order_version,
        expert_collaboration_id=record.expert_collaboration_id,
        participant_subject_id=record.subject_id,
        status=record.status,
        media_kind=record.media_kind,
        token_expires_at=_aware(record.token_expires_at),
        lease_expires_at=_aware(record.lease_expires_at),
        max_expires_at=_aware(record.max_expires_at),
        reconnect_count=record.reconnect_count,
        version=record.version,
    )


def _new_token(tenant_id: str, *, version: str = "rt1") -> str:
    tenant = base64.urlsafe_b64encode(tenant_id.encode()).decode().rstrip("=")
    return f"{version}.{tenant}.{token_urlsafe(32)}"


def context_from_media_token(access_token: str) -> TenantContext:
    return _context_from_token(access_token, expected_version="rtm1")


def authorize_media_capability(
    database: Database,
    access_token: str,
    session_id: str,
) -> RealtimeMediaContext:
    """Resolve a live media capability without granting browser/API identity powers."""

    context = context_from_media_token(access_token)
    with database.transaction(context) as session:
        actor = _capability_subject(
            session, context, session_id, access_token, media=True
        )
        now = datetime.now(UTC)
        record = session.scalar(
            select(RealtimeMediaSessionRecord).where(
                RealtimeMediaSessionRecord.tenant_id == context.tenant_id,
                RealtimeMediaSessionRecord.session_id == session_id,
            ).execution_options(populate_existing=True)
        )
        if (
            record is None
            or record.subject_id != actor.subject_id
            or record.status != "CONNECTED"
            or record.media_token_digest is None
            or not hmac.compare_digest(
                record.media_token_digest,
                _token_digest(access_token),
            )
            or now >= _aware(record.lease_expires_at)
            or now >= _aware(record.max_expires_at)
        ):
            raise RealtimeSessionError("realtime_media_capability_invalid")
        if not _binding_current(record, session):
            reason = _binding_changed_reason(record)
            _end_for_binding_drift(record)
            raise RealtimeSessionError(reason)
        _record_session_disclosure(
            session, actor, record, request_id=f"realtime-audio:{session_id}"
        )
        return RealtimeMediaContext(context, _view(record))


def _binding_current(
    record: RealtimeMediaSessionRecord,
    session: Session,
) -> bool:
    if record.expert_collaboration_id is not None:
        return expert_collaboration_binding_current(
            session,
            tenant_id=record.tenant_id,
            collaboration_id=record.expert_collaboration_id,
            participant_subject_id=record.subject_id,
        )
    return bool(
        field_voice_binding_current(
            session,
            tenant_id=record.tenant_id,
            subject_id=record.subject_id,
            work_order_id=record.work_order_id,
            work_order_version=record.work_order_version,
            incident_id=record.incident_id,
            diagnosis_run_id=record.diagnosis_run_id,
        )
    )


def _binding_changed_reason(record: RealtimeMediaSessionRecord) -> str:
    if record.expert_collaboration_id is not None:
        return "realtime_expert_collaboration_changed"
    return "realtime_field_binding_changed"


def _end_for_binding_drift(record: RealtimeMediaSessionRecord) -> None:
    record.status = "ENDED"
    record.ended_reason = (
        "expert_collaboration_changed"
        if record.expert_collaboration_id is not None
        else "field_binding_changed"
    )
    record.media_token_digest = None
    record.version += 1


def _context_from_token(
    access_token: str,
    *,
    expected_version: str = "rt1",
) -> TenantContext:
    try:
        version, encoded_tenant, nonce = access_token.split(".", 2)
        if version != expected_version or len(nonce) < 32:
            raise ValueError
        padding = "=" * (-len(encoded_tenant) % 4)
        tenant_id = base64.urlsafe_b64decode(encoded_tenant + padding).decode()
        return TenantContext(tenant_id, "realtime-media-service")
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise RealtimeSessionError("realtime_session_token_invalid") from exc


def _token_digest(access_token: str) -> str:
    return "sha256:" + sha256(access_token.encode()).hexdigest()


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _signaling_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname or parsed.query:
        raise ValueError("signaling URL must be a ws/wss endpoint without query parameters")
    return value.rstrip("/")


def _ice_url(value: str, kind: str) -> str:
    normalized = value.strip()
    allowed = {kind, f"{kind}s"}
    if not normalized or normalized.split(":", 1)[0].casefold() not in allowed:
        raise ValueError(f"{kind.upper()} URL is invalid")
    return normalized
