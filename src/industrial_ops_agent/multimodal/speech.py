"""Governed TTS gateway with immutable audio and minimized inference audit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from time import monotonic
from typing import Protocol
from urllib.parse import urlparse

import httpx
from sqlalchemy import select

from industrial_ops_agent.domain.json import legacy_canonical_digest as _digest_json
from industrial_ops_agent.media.service import TenantObjectStore
from industrial_ops_agent.model_gateway.context_manifest import (
    ContextManifest,
    ContextReference,
    build_context_manifest,
    canonical_sha256,
)
from industrial_ops_agent.model_gateway.media_quota import require_media_quota
from industrial_ops_agent.model_gateway.service import (
    ModelGatewayError,
    ModelTransportError,
    ProductionModelResolver,
    ResolvedModel,
)
from industrial_ops_agent.object_store.keys import TenantObjectKey, generated_object_key
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ModelInferenceRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext

MAX_TTS_INPUT_CHARACTERS = 6_000
MAX_TTS_AUDIO_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SpeechSynthesisRequest:
    synthesis_id: str
    diagnosis_run_id: str
    model_alias: str
    subject_id: str
    trace_id: str
    text: str
    voice: str
    deadline: datetime
    safety_warning_digest: str


@dataclass(frozen=True, slots=True)
class SpeechTransportResult:
    content: bytes
    media_type: str
    latency_breakdown: dict[str, float]


@dataclass(frozen=True, slots=True)
class SpeechSynthesisResult:
    synthesis_id: str
    diagnosis_run_id: str
    resolved_release_id: str
    manifest_hash: str
    media_type: str
    size_bytes: int
    audio_sha256: str
    object_key: str
    voice: str
    safety_warning_digest: str
    latency_breakdown: dict[str, float]
    replayed: bool = False


class SpeechTransport(Protocol):
    async def synthesize(
        self,
        model: ResolvedModel,
        request: SpeechSynthesisRequest,
        *,
        timeout_seconds: float,
    ) -> SpeechTransportResult: ...


@dataclass(slots=True)
class OpenAiCompatibleSpeechTransport:
    """Binary `/v1/audio/speech` adapter for approved TTS serving endpoints."""

    api_key: str
    allow_plain_http: bool = False

    async def synthesize(
        self,
        model: ResolvedModel,
        request: SpeechSynthesisRequest,
        *,
        timeout_seconds: float,
    ) -> SpeechTransportResult:
        endpoint = _speech_endpoint(model.endpoint_url, self.allow_plain_http)
        started = monotonic()
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds),
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": _tts_model_id(model),
                        "input": request.text,
                        "voice": request.voice,
                        "response_format": "wav",
                    },
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ModelTransportError("tts_endpoint_unavailable") from exc
        elapsed_ms = (monotonic() - started) * 1000.0
        if response.status_code != 200:
            reason = (
                "tts_endpoint_rejected" if response.status_code < 500 else "tts_endpoint_failed"
            )
            raise ModelTransportError(reason)
        media_type = response.headers.get("content-type", "").split(";", 1)[0].casefold()
        if media_type in {"audio/x-wav", "audio/wave", "application/octet-stream"}:
            media_type = "audio/wav"
        return SpeechTransportResult(
            content=response.content,
            media_type=media_type,
            latency_breakdown={"upstream_total_ms": elapsed_ms},
        )


class SpeechSynthesisGateway:
    def __init__(
        self,
        database: Database,
        resolver: ProductionModelResolver,
        transport: SpeechTransport,
        object_store: TenantObjectStore,
        *,
        allowed_voices: frozenset[str] = frozenset({"default"}),
    ) -> None:
        self._database = database
        self._resolver = resolver
        self._transport = transport
        self._object_store = object_store
        self._allowed_voices = allowed_voices

    async def synthesize(
        self,
        context: TenantContext,
        request: SpeechSynthesisRequest,
    ) -> SpeechSynthesisResult:
        now = datetime.now(UTC)
        self._validate_request(request, now)
        model = self._resolver.resolve(context, request.model_alias)
        prompt_hash = _digest_json(
            {
                "diagnosis_run_id": request.diagnosis_run_id,
                "text": request.text,
                "voice": request.voice,
                "safety_warning_digest": request.safety_warning_digest,
            }
        )
        estimated_tokens = max(1, len(request.text) // 4)
        context_manifest = build_context_manifest(
            trace_id=request.trace_id,
            prompt_bundle_id=None,
            prompt_bundle_hash=None,
            model_release_id=model.release_id,
            retrieval_index_id=None,
            evidence=(
                ContextReference(
                    request.diagnosis_run_id,
                    canonical_sha256(
                        {
                            "text": request.text,
                            "safety_warning_digest": request.safety_warning_digest,
                        }
                    ),
                ),
            ),
            memories=(),
            tool_results=(),
            token_budget=estimated_tokens,
            truncated_items=(),
            messages=({"role": "user", "content": request.text},),
            response_schema={"type": "audio/wav", "voice": request.voice},
        )
        replay = self._admit(
            context,
            request,
            model,
            prompt_hash,
            context_manifest,
            estimated_tokens,
            now,
        )
        if replay is not None:
            return replay
        remaining = max(0.001, (request.deadline - now).total_seconds())
        try:
            result = await self._transport.synthesize(
                model,
                request,
                timeout_seconds=remaining,
            )
            _validate_transport_result(result)
            artifact_id = f"speech-{request.synthesis_id}"
            object_key = generated_object_key(
                context,
                request.diagnosis_run_id,
                artifact_id,
            )
            await self._object_store.put(context, object_key, result.content)
            return self._finish_success(
                context,
                request,
                model,
                result,
                object_key,
            )
        except ModelGatewayError as exc:
            self._finish_failure(context, request.synthesis_id, exc.reason)
            raise
        except Exception as exc:
            self._finish_failure(context, request.synthesis_id, "tts_unexpected_failure")
            raise ModelGatewayError("tts_unexpected_failure") from exc

    def describe(
        self,
        context: TenantContext,
        synthesis_id: str,
    ) -> SpeechSynthesisResult:
        with self._database.transaction(context) as session:
            record = session.scalar(
                select(ModelInferenceRecord).where(
                    ModelInferenceRecord.tenant_id == context.tenant_id,
                    ModelInferenceRecord.inference_request_id == synthesis_id,
                    ModelInferenceRecord.request_class == "TTS",
                )
            )
            if record is None or record.status != "SUCCEEDED":
                raise ModelGatewayError("speech_synthesis_not_visible")
            return _result_from_record(record, replayed=True)

    async def read_audio(
        self,
        context: TenantContext,
        result: SpeechSynthesisResult,
    ) -> bytes:
        expected_key = generated_object_key(
            context,
            result.diagnosis_run_id,
            f"speech-{result.synthesis_id}",
        )
        if result.object_key != expected_key.value:
            raise ModelGatewayError("speech_audio_object_key_changed")
        key = TenantObjectKey(
            tenant_id=context.tenant_id,
            zone=expected_key.zone,
            value=expected_key.value,
        )
        content = await self._object_store.get(context, key)
        if (
            content is None
            or len(content) != result.size_bytes
            or sha256(content).hexdigest() != result.audio_sha256
        ):
            raise ModelGatewayError("speech_audio_integrity_failed")
        return content

    def _validate_request(self, request: SpeechSynthesisRequest, now: datetime) -> None:
        if not request.subject_id or not request.trace_id:
            raise ModelGatewayError("tts_request_identity_invalid")
        if request.deadline.tzinfo is None or request.deadline <= now:
            raise ModelGatewayError("tts_request_deadline_expired")
        if (
            not request.text.strip()
            or len(request.text) > MAX_TTS_INPUT_CHARACTERS
            or request.voice not in self._allowed_voices
        ):
            raise ModelGatewayError("tts_request_contract_invalid")
        if not request.safety_warning_digest.startswith("sha256:"):
            raise ModelGatewayError("tts_safety_warning_not_bound")

    def _admit(
        self,
        context: TenantContext,
        request: SpeechSynthesisRequest,
        model: ResolvedModel,
        prompt_hash: str,
        context_manifest: ContextManifest,
        estimated_tokens: int,
        now: datetime,
    ) -> SpeechSynthesisResult | None:
        with self._database.transaction(context) as session:
            existing = session.scalar(
                select(ModelInferenceRecord).where(
                    ModelInferenceRecord.tenant_id == context.tenant_id,
                    ModelInferenceRecord.inference_request_id == request.synthesis_id,
                )
            )
            if existing is not None:
                if (
                    existing.prompt_hash != prompt_hash
                    or existing.resolved_release_id != model.release_id
                    or existing.context_hash != context_manifest.context_hash
                ):
                    raise ModelGatewayError("tts_idempotency_conflict")
                if existing.status == "SUCCEEDED":
                    return _result_from_record(existing, replayed=True)
                raise ModelGatewayError("tts_request_already_consumed")
            require_media_quota(
                session,
                tenant_id=context.tenant_id,
                model_alias=request.model_alias,
                request_class="TTS",
                reservation=estimated_tokens,
                limit_reason="tts_input_limit_exceeded",
                include_pending=True,
                now=now,
            )
            session.add(
                ModelInferenceRecord(
                    inference_request_id=request.synthesis_id,
                    tenant_id=context.tenant_id,
                    model_alias=request.model_alias,
                    resolved_release_id=model.release_id,
                    manifest_hash=model.manifest_hash,
                    subject_id=request.subject_id,
                    trace_id=request.trace_id,
                    agent_run_id=None,
                    diagnosis_run_id=request.diagnosis_run_id,
                    prompt_bundle_id=None,
                    index_release_id=None,
                    context_manifest_json=context_manifest.as_dict(),
                    context_hash=context_manifest.context_hash,
                    request_class="TTS",
                    data_classification="CONFIDENTIAL",
                    status="PENDING",
                    deadline=request.deadline,
                    max_output_tokens=estimated_tokens,
                    temperature=0.0,
                    prompt_hash=prompt_hash,
                    response_json=None,
                    response_hash=None,
                    usage_prompt_tokens=0,
                    usage_completion_tokens=0,
                    finish_reason=None,
                    latency_breakdown_json={},
                    safety_decision="PENDING",
                    degraded_from=None,
                    failure_reason=None,
                    completed_at=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        return None

    def _finish_success(
        self,
        context: TenantContext,
        request: SpeechSynthesisRequest,
        model: ResolvedModel,
        result: SpeechTransportResult,
        object_key: TenantObjectKey,
    ) -> SpeechSynthesisResult:
        now = datetime.now(UTC)
        audio_hash = sha256(result.content).hexdigest()
        payload = {
            "diagnosis_run_id": request.diagnosis_run_id,
            "object_key": object_key.value,
            "media_type": result.media_type,
            "size_bytes": len(result.content),
            "audio_sha256": audio_hash,
            "voice": request.voice,
            "safety_warning_digest": request.safety_warning_digest,
        }
        with self._database.transaction(context) as session:
            record = session.scalar(
                select(ModelInferenceRecord).where(
                    ModelInferenceRecord.tenant_id == context.tenant_id,
                    ModelInferenceRecord.inference_request_id == request.synthesis_id,
                )
            )
            if record is None or record.status != "PENDING":
                raise ModelGatewayError("tts_audit_state_changed")
            record.status = "SUCCEEDED"
            record.response_json = payload
            record.response_hash = _digest_json(payload)
            record.usage_prompt_tokens = record.max_output_tokens
            record.finish_reason = "synthesized"
            record.latency_breakdown_json = result.latency_breakdown
            record.safety_decision = "ALLOWED"
            record.completed_at = now
            return _result_from_record(record, replayed=False)

    def _finish_failure(
        self,
        context: TenantContext,
        synthesis_id: str,
        reason: str,
    ) -> None:
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            record = session.scalar(
                select(ModelInferenceRecord).where(
                    ModelInferenceRecord.tenant_id == context.tenant_id,
                    ModelInferenceRecord.inference_request_id == synthesis_id,
                )
            )
            if record is not None and record.status == "PENDING":
                record.status = "FAILED"
                record.safety_decision = "REJECTED"
                record.failure_reason = reason[:255]
                record.completed_at = now


def _validate_transport_result(result: SpeechTransportResult) -> None:
    if (
        result.media_type != "audio/wav"
        or not result.content.startswith(b"RIFF")
        or len(result.content) < 44
        or len(result.content) > MAX_TTS_AUDIO_BYTES
        or result.content[8:12] != b"WAVE"
        or any(not math.isfinite(value) or value < 0 for value in result.latency_breakdown.values())
    ):
        raise ModelGatewayError("tts_response_invalid")


def _result_from_record(
    record: ModelInferenceRecord,
    *,
    replayed: bool,
) -> SpeechSynthesisResult:
    payload = record.response_json
    if not isinstance(payload, dict) or record.finish_reason != "synthesized":
        raise ModelGatewayError("speech_synthesis_audit_incomplete")
    try:
        return SpeechSynthesisResult(
            synthesis_id=record.inference_request_id,
            diagnosis_run_id=str(payload["diagnosis_run_id"]),
            resolved_release_id=record.resolved_release_id,
            manifest_hash=record.manifest_hash,
            media_type=str(payload["media_type"]),
            size_bytes=int(payload["size_bytes"]),
            audio_sha256=str(payload["audio_sha256"]),
            object_key=str(payload["object_key"]),
            voice=str(payload["voice"]),
            safety_warning_digest=str(payload["safety_warning_digest"]),
            latency_breakdown=dict(record.latency_breakdown_json),
            replayed=replayed,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelGatewayError("speech_synthesis_audit_incomplete") from exc


def _speech_endpoint(base_url: str, allow_plain_http: bool) -> str:
    parsed = urlparse(base_url)
    allowed = {"https"} | ({"http"} if allow_plain_http else set())
    if (
        parsed.scheme not in allowed
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ModelTransportError("tts_endpoint_invalid")
    return f"{base_url.rstrip('/')}/v1/audio/speech"


def _tts_model_id(model: ResolvedModel) -> str:
    model_id = model.multimodal_model_ids.get("tts", "").strip()
    if not model_id:
        raise ModelTransportError("tts_model_not_bound_to_release")
    return model_id
