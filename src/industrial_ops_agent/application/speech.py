"""Business-bound speech synthesis for completed, authorized diagnoses."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.application.expert_diagnoses import effective_report
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.maintenance_planning.review_isolation import (
    lock_subject,
    record_disclosure,
)
from industrial_ops_agent.multimodal.speech import (
    SpeechSynthesisGateway,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.repositories import (
    DiagnosisRunRepository,
    IncidentRepository,
)

TTS_SAFETY_WARNING = (
    "安全提示：在检查、拆卸或维修设备前，必须停机、断电、泄压并执行上锁挂牌；"
    "确认设备无残余能量，佩戴适用的个人防护装备。未经现场负责人确认，不得执行维修动作。"
)


class SpeechSynthesisDenied(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class DiagnosisSpeechService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        gateway: SpeechSynthesisGateway,
        *,
        model_alias: str,
        timeout_seconds: float,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._gateway = gateway
        self._model_alias = model_alias
        self._timeout_seconds = timeout_seconds

    async def synthesize(
        self,
        identity: IdentityContext,
        diagnosis_run_id: str,
        *,
        safety_acknowledged: bool,
        voice: str,
        idempotency_key: str,
        request_id: str,
    ) -> SpeechSynthesisResult:
        report = await asyncio.to_thread(
            self._authorized_report, identity, diagnosis_run_id, request_id=request_id
        )
        if not safety_acknowledged:
            await asyncio.to_thread(
                self._authorizer.record_guard_decision,
                identity,
                action="diagnosis_speech.safety_confirmation",
                decision="deny",
                reason_code="screen_safety_confirmation_required",
                request_id=request_id,
                resource_id=diagnosis_run_id,
            )
            raise SpeechSynthesisDenied("screen_safety_confirmation_required")
        text = _speech_text(report)
        warning_digest = "sha256:" + sha256(TTS_SAFETY_WARNING.encode()).hexdigest()
        synthesis_id = _synthesis_id(identity, idempotency_key)
        await asyncio.to_thread(
            self._authorizer.record_guard_decision,
            identity,
            action="diagnosis_speech.safety_confirmation",
            decision="allow",
            reason_code="screen_safety_confirmation_recorded",
            request_id=request_id,
            resource_id=diagnosis_run_id,
        )
        return await self._gateway.synthesize(
            identity.tenant_context,
            SpeechSynthesisRequest(
                synthesis_id=synthesis_id,
                diagnosis_run_id=diagnosis_run_id,
                model_alias=self._model_alias,
                subject_id=identity.subject_id,
                trace_id=request_id,
                text=text,
                voice=voice,
                deadline=datetime.now(UTC) + timedelta(seconds=self._timeout_seconds),
                safety_warning_digest=warning_digest,
            ),
        )

    async def audio(
        self,
        identity: IdentityContext,
        synthesis_id: str,
        *,
        request_id: str,
    ) -> tuple[SpeechSynthesisResult, bytes]:
        result = await asyncio.to_thread(
            self._gateway.describe, identity.tenant_context, synthesis_id
        )
        await asyncio.to_thread(
            self._authorized_report, identity, result.diagnosis_run_id, request_id=request_id
        )
        return result, await self._gateway.read_audio(identity.tenant_context, result)

    def _authorized_report(
        self,
        identity: IdentityContext,
        diagnosis_run_id: str,
        *,
        request_id: str,
    ) -> dict[str, object]:
        with self._database.transaction(identity.tenant_context) as session:
            lock_subject(session, identity)
            run = DiagnosisRunRepository(session, identity.tenant_context).get(
                diagnosis_run_id
            )
            if run is None:
                raise ResourceNotVisible
            incident = IncidentRepository(session, identity.tenant_context).get(
                run.incident_id
            )
            if incident is None:
                raise ResourceNotVisible
            try:
                self._authorizer.require(
                    identity,
                    Action.SYNTHESIZE_DIAGNOSIS_SPEECH,
                    ResourceContext(
                        tenant_id=identity.tenant_id,
                        resource_id=diagnosis_run_id,
                        asset_id=incident.asset_id,
                    ),
                    request_id=request_id,
                )
            except AuthorizationDenied as exc:
                raise ResourceNotVisible from exc
            report = effective_report(session, identity.tenant_id, run)
            if run.status != "COMPLETED" or not isinstance(report, dict):
                raise SpeechSynthesisDenied("completed_diagnosis_is_required")
            if report.get("status") != "COMPLETED":
                raise SpeechSynthesisDenied("safe_completed_report_is_required")
            record_disclosure(
                session, identity, (diagnosis_run_id,),
                resource_kind="diagnosis-speech",
                resource_id=diagnosis_run_id,
                request_id=request_id,
            )
            return dict(report)


def _speech_text(report: dict[str, object]) -> str:
    conclusion = report.get("conclusion")
    if not isinstance(conclusion, str) or not conclusion.strip():
        raise SpeechSynthesisDenied("diagnosis_conclusion_is_required")
    contradictions = report.get("contradictions")
    missing = report.get("missing_information")
    if contradictions not in ([], ()) or missing not in ([], ()):
        raise SpeechSynthesisDenied("uncertain_diagnosis_cannot_be_spoken")
    checks = report.get("next_checks")
    if not isinstance(checks, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in checks
    ):
        raise SpeechSynthesisDenied("diagnosis_checks_are_invalid")
    sections = [TTS_SAFETY_WARNING, f"诊断结论：{conclusion.strip()}。"]
    if checks:
        sections.append("建议由具备资质的人员执行以下检查：" + "；".join(checks) + "。")
    text = "".join(sections)
    if len(text) > 6_000 or any(character in text for character in {"\x00", "\r"}):
        raise SpeechSynthesisDenied("diagnosis_speech_text_is_invalid")
    return text


def _synthesis_id(identity: IdentityContext, idempotency_key: str) -> str:
    if not idempotency_key or len(idempotency_key) > 255:
        raise SpeechSynthesisDenied("idempotency_key_is_invalid")
    basis = f"{identity.tenant_id}\0{identity.subject_id}\0{idempotency_key}"
    return "tts-" + sha256(basis.encode()).hexdigest()
