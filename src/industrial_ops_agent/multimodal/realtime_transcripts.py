"""Persist governed realtime ASR segments without retaining raw audio."""

from __future__ import annotations

import asyncio
import hmac
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.guardrails.prompt_injection import PromptInjectionGuard
from industrial_ops_agent.model_gateway.service import ModelGatewayError
from industrial_ops_agent.multimodal.realtime import (
    RealtimeSessionError,
    context_from_media_token,
)
from industrial_ops_agent.multimodal.transcription import (
    TranscriptionGateway,
    TranscriptionRequest,
    build_transcription_hotword_profile,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AgentEventRecord,
    AgentRunRecord,
    DiagnosisRunRecord,
    IncidentRecord,
    RealtimeMediaSessionRecord,
    RealtimeTranscriptSegmentRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext, validate_boundary_identifier
from industrial_ops_agent.workorders.voice_guidance import field_voice_binding_current

ENTITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "alarm_code",
        re.compile(r"(?:报警(?:码)?|故障码|alarm)\s*[:：]?\s*([A-Z]{1,4}[-_ ]?\d{2,6})", re.I),
    ),
    (
        "serial_number",
        re.compile(r"(?:设备号|序列号|serial|s/n)\s*[:：#]?\s*([A-Z0-9][A-Z0-9._/-]{3,})", re.I),
    ),
    (
        "part_number",
        re.compile(r"(?:部件号|零件号|part|p/n)\s*[:：#]?\s*([A-Z0-9][A-Z0-9._/-]{3,})", re.I),
    ),
)


@dataclass(frozen=True, slots=True)
class RealtimeTranscriptSegment:
    segment_id: str
    session_id: str
    sequence: int
    start_ms: int
    end_ms: int
    text: str
    confirmed_text: str | None
    confidence: float
    language: str
    entities: tuple[dict[str, object], ...]
    resolved_release_id: str
    manifest_hash: str
    status: str
    agent_event_id: str | None
    agent_event_sequence: int | None
    security_policy_version: str | None
    security_findings: tuple[dict[str, str], ...]
    version: int


class RealtimeTranscriptService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        gateway: TranscriptionGateway,
        *,
        model_alias: str,
        timeout_seconds: float,
        hotwords: tuple[str, ...],
        prompt_injection_guard: PromptInjectionGuard | None = None,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._gateway = gateway
        self._model_alias = model_alias
        self._timeout_seconds = timeout_seconds
        self._hotword_profile = build_transcription_hotword_profile(hotwords)
        self._prompt_injection_guard = prompt_injection_guard or PromptInjectionGuard()

    async def transcribe(
        self,
        media_access_token: str,
        session_id: str,
        *,
        segment_id: str,
        sequence: int,
        start_ms: int,
        end_ms: int,
        audio_wav: bytes,
        request_id: str,
    ) -> RealtimeTranscriptSegment:
        validate_boundary_identifier(segment_id, field="segment_id")
        if sequence < 1 or start_ms < 0 or end_ms <= start_ms:
            raise RealtimeSessionError("realtime_segment_timeline_invalid")
        context = context_from_media_token(media_access_token)
        now = datetime.now(UTC)
        audio_hash = sha256(audio_wav).hexdigest()
        subject_id, existing = await asyncio.to_thread(
            self._admit_media_segment,
            context,
            media_access_token,
            session_id,
            segment_id=segment_id,
            sequence=sequence,
            start_ms=start_ms,
            end_ms=end_ms,
            audio_hash=audio_hash,
            now=now,
        )
        if existing is not None:
            return existing
        inference_id = "asr-" + sha256(
            f"{context.tenant_id}\0{session_id}\0{segment_id}".encode()
        ).hexdigest()
        result = await self._gateway.transcribe(
            TenantContext(context.tenant_id, subject_id),
            TranscriptionRequest(
                inference_request_id=inference_id,
                model_alias=self._model_alias,
                session_id=session_id,
                segment_id=segment_id,
                subject_id=subject_id,
                trace_id=request_id,
                audio_wav=audio_wav,
                language="zh",
                hotwords=self._hotword_profile.hotwords,
                deadline=now + timedelta(seconds=self._timeout_seconds),
                hotword_profile_id=self._hotword_profile.profile_id,
            ),
        )
        expected_duration_ms = end_ms - start_ms
        actual_duration_ms = round(result.duration_seconds * 1_000)
        if abs(expected_duration_ms - actual_duration_ms) > 1_500:
            raise ModelGatewayError("asr_segment_timeline_mismatch")
        entities = _extract_entities(result.text, result.confidence)
        security_decision = self._prompt_injection_guard.inspect_text(
            result.text,
            source_role="asr_segment",
        )
        security_findings = tuple(
            {
                "source_type": "ASR_SEGMENT",
                "source_id": segment_id,
                "policy_version": security_decision.policy_version,
                "pattern_id": finding.pattern_id,
                "category": finding.category,
                "severity": finding.severity,
                "content_hash": finding.content_hash,
            }
            for finding in security_decision.findings
        )
        status = (
            "SECURITY_REVIEW_REQUIRED"
            if security_findings
            else "REQUIRES_CONFIRMATION"
            if result.confidence < 0.75
            or any(bool(item["requires_confirmation"]) for item in entities)
            else "PROVISIONAL"
        )
        def persist_segment() -> RealtimeTranscriptSegment:
            with self._database.transaction(context) as session:
                media_session = session.scalar(
                    select(RealtimeMediaSessionRecord)
                    .where(
                        RealtimeMediaSessionRecord.tenant_id == context.tenant_id,
                        RealtimeMediaSessionRecord.session_id == session_id,
                    )
                    .with_for_update()
                )
                if media_session is None or media_session.subject_id != subject_id:
                    raise RealtimeSessionError("realtime_media_capability_invalid")
                if not _field_binding_current(media_session, session):
                    raise RealtimeSessionError("realtime_field_binding_changed")
                existing_record = session.scalar(
                    select(RealtimeTranscriptSegmentRecord).where(
                        RealtimeTranscriptSegmentRecord.tenant_id == context.tenant_id,
                        RealtimeTranscriptSegmentRecord.segment_id == segment_id,
                    )
                )
                if existing_record is not None:
                    return _view(existing_record)
                record = RealtimeTranscriptSegmentRecord(
                    segment_id=segment_id,
                    tenant_id=context.tenant_id,
                    session_id=session_id,
                    sequence=sequence,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=result.text,
                    confirmed_text=None,
                    confidence=result.confidence,
                    language=result.language,
                    entities_json=list(entities),
                    security_policy_version=security_decision.policy_version,
                    security_findings_json=list(security_findings),
                    audio_sha256=result.audio_sha256,
                    resolved_release_id=result.resolved_release_id,
                    manifest_hash=result.manifest_hash,
                    status=status,
                    reviewer_subject_id=None,
                    reviewed_at=None,
                    agent_event_id=None,
                    agent_event_sequence=None,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
                session.flush()
                return _view(record)

        return await asyncio.to_thread(persist_segment)

    def confirm(
        self,
        identity: IdentityContext,
        segment_id: str,
        *,
        decision: str,
        corrected_text: str | None,
        expected_version: int,
        request_id: str,
    ) -> RealtimeTranscriptSegment:
        if decision not in {"ACCEPTED", "REJECTED"}:
            raise ValueError("transcript decision is invalid")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(RealtimeTranscriptSegmentRecord).where(
                    RealtimeTranscriptSegmentRecord.tenant_id == identity.tenant_id,
                    RealtimeTranscriptSegmentRecord.segment_id == segment_id,
                )
            )
            if record is None:
                raise ResourceNotVisible
            media_session = session.scalar(
                select(RealtimeMediaSessionRecord).where(
                    RealtimeMediaSessionRecord.tenant_id == identity.tenant_id,
                    RealtimeMediaSessionRecord.session_id == record.session_id,
                )
            )
            incident = (
                session.scalar(
                    select(IncidentRecord).where(
                        IncidentRecord.tenant_id == identity.tenant_id,
                        IncidentRecord.incident_id == media_session.incident_id,
                    )
                )
                if media_session is not None
                else None
            )
            if media_session is None or incident is None:
                raise ResourceNotVisible
            if media_session.subject_id != identity.subject_id:
                raise ResourceNotVisible
            if not _field_binding_current(media_session, session):
                raise RealtimeSessionError("realtime_field_binding_changed")
            if decision == "ACCEPTED" and media_session.diagnosis_run_id is None:
                raise RealtimeSessionError("realtime_transcript_diagnosis_required")
            asset_id = incident.asset_id
        try:
            self._authorizer.require(
                identity,
                Action.CONFIRM_REALTIME_TRANSCRIPT,
                ResourceContext(
                    tenant_id=identity.tenant_id,
                    resource_id=segment_id,
                    asset_id=asset_id,
                ),
                request_id=request_id,
            )
        except AuthorizationDenied as exc:
            raise ResourceNotVisible from exc
        normalized = corrected_text.strip() if corrected_text else None
        if decision == "ACCEPTED" and normalized is not None and len(normalized) > 4_000:
            raise ValueError("corrected transcript is too long")
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(RealtimeTranscriptSegmentRecord)
                .where(
                    RealtimeTranscriptSegmentRecord.tenant_id == identity.tenant_id,
                    RealtimeTranscriptSegmentRecord.segment_id == segment_id,
                )
                .with_for_update()
            )
            if record is None:
                raise ResourceNotVisible
            if record.version != expected_version:
                raise RealtimeSessionError("realtime_transcript_version_conflict")
            if record.status in {"ACCEPTED", "REJECTED"}:
                raise RealtimeSessionError("realtime_transcript_already_reviewed")
            if not record.security_policy_version:
                raise RealtimeSessionError("realtime_transcript_security_check_missing")
            if (
                decision == "ACCEPTED"
                and record.security_findings_json
                and (
                    normalized is None
                    or self._prompt_injection_guard.inspect_text(
                        normalized,
                        source_role="asr_segment",
                    ).decision != "ALLOWED"
                )
            ):
                raise RealtimeSessionError("realtime_transcript_security_blocked")
            media_session = session.scalar(
                select(RealtimeMediaSessionRecord)
                .where(
                    RealtimeMediaSessionRecord.tenant_id == identity.tenant_id,
                    RealtimeMediaSessionRecord.session_id == record.session_id,
                )
                .with_for_update()
            )
            if media_session is None or media_session.subject_id != identity.subject_id:
                raise ResourceNotVisible
            if not _field_binding_current(media_session, session):
                raise RealtimeSessionError("realtime_field_binding_changed")
            event: AgentEventRecord | None = None
            if decision == "ACCEPTED":
                if media_session.diagnosis_run_id is None:
                    raise RealtimeSessionError("realtime_transcript_diagnosis_required")
                diagnosis = session.scalar(
                    select(DiagnosisRunRecord).where(
                        DiagnosisRunRecord.tenant_id == identity.tenant_id,
                        DiagnosisRunRecord.diagnosis_run_id
                        == media_session.diagnosis_run_id,
                        DiagnosisRunRecord.incident_id == media_session.incident_id,
                    )
                )
                if diagnosis is None:
                    raise ResourceNotVisible
                agent_run = session.scalar(
                    select(AgentRunRecord)
                    .where(
                        AgentRunRecord.tenant_id == identity.tenant_id,
                        AgentRunRecord.agent_run_id == diagnosis.agent_run_id,
                    )
                    .with_for_update()
                )
                if agent_run is None:
                    raise ResourceNotVisible
                event = _confirmed_input_event(
                    identity,
                    record,
                    diagnosis,
                    agent_run,
                    confirmed_text=normalized or record.text,
                    occurred_at=now,
                )
                session.add(event)
                agent_run.last_sequence = event.sequence
                agent_run.version += 1
                agent_run.updated_at = now
            record.status = decision
            record.confirmed_text = (
                normalized or record.text if decision == "ACCEPTED" else None
            )
            record.reviewer_subject_id = identity.subject_id
            record.reviewed_at = now
            record.agent_event_id = event.event_id if event is not None else None
            record.agent_event_sequence = event.sequence if event is not None else None
            record.version += 1
            record.updated_at = now
            return _view(record)

    def _admit_media_segment(
        self,
        context: TenantContext,
        media_access_token: str,
        session_id: str,
        *,
        segment_id: str,
        sequence: int,
        start_ms: int,
        end_ms: int,
        audio_hash: str,
        now: datetime,
    ) -> tuple[str, RealtimeTranscriptSegment | None]:
        with self._database.transaction(context) as session:
            media_session = session.scalar(
                select(RealtimeMediaSessionRecord).where(
                    RealtimeMediaSessionRecord.tenant_id == context.tenant_id,
                    RealtimeMediaSessionRecord.session_id == session_id,
                )
            )
            if (
                media_session is None
                or media_session.status != "CONNECTED"
                or media_session.media_token_digest is None
                or not hmac.compare_digest(
                    media_session.media_token_digest,
                    "sha256:" + sha256(media_access_token.encode()).hexdigest(),
                )
                or now >= _aware(media_session.lease_expires_at)
                or now >= _aware(media_session.max_expires_at)
            ):
                raise RealtimeSessionError("realtime_media_capability_invalid")
            if not _field_binding_current(media_session, session):
                raise RealtimeSessionError("realtime_field_binding_changed")
            existing = session.scalar(
                select(RealtimeTranscriptSegmentRecord).where(
                    RealtimeTranscriptSegmentRecord.tenant_id == context.tenant_id,
                    RealtimeTranscriptSegmentRecord.segment_id == segment_id,
                )
            )
            if existing is not None:
                if (
                    existing.session_id != session_id
                    or existing.sequence != sequence
                    or existing.start_ms != start_ms
                    or existing.end_ms != end_ms
                    or existing.audio_sha256 != audio_hash
                ):
                    raise RealtimeSessionError("realtime_segment_idempotency_conflict")
                return media_session.subject_id, _view(existing)
            sequence_collision = session.scalar(
                select(RealtimeTranscriptSegmentRecord).where(
                    RealtimeTranscriptSegmentRecord.tenant_id == context.tenant_id,
                    RealtimeTranscriptSegmentRecord.session_id == session_id,
                    RealtimeTranscriptSegmentRecord.sequence == sequence,
                )
            )
            if sequence_collision is not None:
                raise RealtimeSessionError("realtime_segment_sequence_conflict")
            return media_session.subject_id, None


def _field_binding_current(
    record: RealtimeMediaSessionRecord,
    session: Session,
) -> bool:
    return field_voice_binding_current(
        session,
        tenant_id=record.tenant_id,
        subject_id=record.subject_id,
        work_order_id=record.work_order_id,
        work_order_version=record.work_order_version,
        incident_id=record.incident_id,
        diagnosis_run_id=record.diagnosis_run_id,
    )


def _extract_entities(text: str, confidence: float) -> tuple[dict[str, object], ...]:
    entities: list[dict[str, object]] = []
    for entity_type, pattern in ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1).upper().replace("_", "-").replace(" ", "-")
            entity_confidence = max(0.0, confidence - 0.05)
            entities.append(
                {
                    "entity_id": "entity-"
                    + sha256(f"{entity_type}\0{match.start()}\0{value}".encode()).hexdigest()[:16],
                    "entity_type": entity_type,
                    "value": value,
                    "confidence": entity_confidence,
                    "requires_confirmation": entity_confidence < 0.9,
                }
            )
    return tuple(entities)


def _view(record: RealtimeTranscriptSegmentRecord) -> RealtimeTranscriptSegment:
    return RealtimeTranscriptSegment(
        segment_id=record.segment_id,
        session_id=record.session_id,
        sequence=record.sequence,
        start_ms=record.start_ms,
        end_ms=record.end_ms,
        text=record.text,
        confirmed_text=record.confirmed_text,
        confidence=record.confidence,
        language=record.language,
        entities=tuple(dict(item) for item in record.entities_json),
        resolved_release_id=record.resolved_release_id,
        manifest_hash=record.manifest_hash,
        status=record.status,
        agent_event_id=record.agent_event_id,
        agent_event_sequence=record.agent_event_sequence,
        security_policy_version=record.security_policy_version,
        security_findings=tuple(dict(item) for item in record.security_findings_json),
        version=record.version,
    )


def _confirmed_input_event(
    identity: IdentityContext,
    transcript: RealtimeTranscriptSegmentRecord,
    diagnosis: DiagnosisRunRecord,
    agent_run: AgentRunRecord,
    *,
    confirmed_text: str,
    occurred_at: datetime,
) -> AgentEventRecord:
    """Build the sole promotion boundary from provisional ASR to Agent input."""

    return AgentEventRecord(
        event_id=f"agent-event-{uuid4().hex}",
        tenant_id=identity.tenant_id,
        agent_run_id=agent_run.agent_run_id,
        sequence=agent_run.last_sequence + 1,
        event_type="agent.user_input.confirmed",
        payload={
            "diagnosis_run_id": diagnosis.diagnosis_run_id,
            "source_type": "realtime_transcript",
            "source_id": transcript.segment_id,
            "session_id": transcript.session_id,
            "text": confirmed_text,
            "language": transcript.language,
            "reviewer_subject_id": identity.subject_id,
            "asr_release_id": transcript.resolved_release_id,
            "asr_manifest_hash": transcript.manifest_hash,
        },
        visibility="authorized",
        occurred_at=occurred_at,
        created_at=occurred_at,
        updated_at=occurred_at,
    )


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
