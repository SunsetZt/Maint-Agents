"""Governed online ASR for short VAD segments."""

from __future__ import annotations

import asyncio
import io
import json
import math
import unicodedata
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from time import monotonic
from typing import Protocol

import httpx

from industrial_ops_agent.domain.json import legacy_canonical_digest as _digest
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
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ModelInferenceRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext

MAX_ASR_AUDIO_BYTES = 5 * 1024 * 1024
MAX_ASR_SEGMENT_SECONDS = 30.0
MAX_ASR_HOTWORDS = 64
MAX_ASR_HOTWORD_LENGTH = 64


@dataclass(frozen=True, slots=True)
class TranscriptionHotwordProfile:
    profile_id: str
    hotwords: tuple[str, ...]


def build_transcription_hotword_profile(
    base_hotwords: tuple[str, ...],
    *,
    contextual_terms: tuple[str, ...] = (),
) -> TranscriptionHotwordProfile:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in (*base_hotwords, *contextual_terms):
        value = unicodedata.normalize("NFKC", raw_value).strip()
        if not value:
            continue
        if (
            len(value) > MAX_ASR_HOTWORD_LENGTH
            or any(unicodedata.category(character).startswith("C") for character in value)
        ):
            raise ModelGatewayError("asr_hotword_profile_invalid")
        deduplication_key = value.casefold()
        if deduplication_key not in seen:
            normalized.append(value)
            seen.add(deduplication_key)
    if len(normalized) > MAX_ASR_HOTWORDS or len("，".join(normalized)) > 1_000:
        raise ModelGatewayError("asr_hotword_profile_invalid")
    hotwords = tuple(normalized)
    profile_hash = canonical_sha256(hotwords).removeprefix("sha256:")
    return TranscriptionHotwordProfile(
        profile_id=f"asr-hotwords-{profile_hash[:20]}",
        hotwords=hotwords,
    )


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    inference_request_id: str
    model_alias: str
    session_id: str
    segment_id: str
    subject_id: str
    trace_id: str
    audio_wav: bytes
    language: str
    hotwords: tuple[str, ...]
    deadline: datetime
    hotword_profile_id: str = "unversioned-hotwords"


@dataclass(frozen=True, slots=True)
class TranscriptionTransportResult:
    text: str
    language: str
    confidence: float
    duration_seconds: float
    latency_breakdown: dict[str, float]


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    duration_seconds: float
    resolved_release_id: str
    manifest_hash: str
    audio_sha256: str
    latency_breakdown: dict[str, float]
    hotword_profile_id: str


class TranscriptionTransport(Protocol):
    async def transcribe(
        self,
        model: ResolvedModel,
        request: TranscriptionRequest,
        *,
        timeout_seconds: float,
    ) -> TranscriptionTransportResult: ...


@dataclass(slots=True)
class OpenAiCompatibleTranscriptionTransport:
    api_key: str
    allow_plain_http: bool = False

    async def transcribe(
        self,
        model: ResolvedModel,
        request: TranscriptionRequest,
        *,
        timeout_seconds: float,
    ) -> TranscriptionTransportResult:
        endpoint = _transcription_endpoint(model.endpoint_url, self.allow_plain_http)
        model_id = model.multimodal_model_ids.get("asr", "").strip()
        if not model_id:
            raise ModelTransportError("asr_model_not_bound_to_release")
        started = monotonic()
        prompt = "，".join(request.hotwords)
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds),
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data={
                        "model": model_id,
                        "language": request.language,
                        "response_format": "verbose_json",
                        "prompt": prompt,
                    },
                    files={"file": ("segment.wav", request.audio_wav, "audio/wav")},
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ModelTransportError("asr_endpoint_unavailable") from exc
        if response.status_code != 200:
            reason = (
                "asr_endpoint_rejected" if response.status_code < 500 else "asr_endpoint_failed"
            )
            raise ModelTransportError(reason)
        try:
            body = response.json()
            text = str(body["text"]).strip()
            language = str(body.get("language") or request.language).strip()
            duration = float(body.get("duration") or _wav_contract(request.audio_wav))
            confidence = _response_confidence(body)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelTransportError("asr_response_invalid") from exc
        return TranscriptionTransportResult(
            text=text,
            language=language,
            confidence=confidence,
            duration_seconds=duration,
            latency_breakdown={"upstream_total_ms": (monotonic() - started) * 1000.0},
        )


class TranscriptionGateway:
    def __init__(
        self,
        database: Database,
        resolver: ProductionModelResolver,
        transport: TranscriptionTransport,
    ) -> None:
        self._database = database
        self._resolver = resolver
        self._transport = transport

    async def transcribe(
        self,
        context: TenantContext,
        request: TranscriptionRequest,
    ) -> TranscriptionResult:
        now = datetime.now(UTC)
        duration = _validate_request(request, now)
        model = await asyncio.to_thread(self._resolver.resolve, context, request.model_alias)
        audio_hash = sha256(request.audio_wav).hexdigest()
        prompt_hash = _digest(
            {
                "audio_sha256": audio_hash,
                "language": request.language,
                "hotwords": request.hotwords,
                "hotword_profile_id": request.hotword_profile_id,
                "segment_id": request.segment_id,
            }
        )
        reservation = max(1, math.ceil(duration * 4))
        context_manifest = build_context_manifest(
            trace_id=request.trace_id,
            prompt_bundle_id=None,
            prompt_bundle_hash=None,
            model_release_id=model.release_id,
            retrieval_index_id=None,
            evidence=(
                ContextReference(
                    request.segment_id,
                    f"sha256:{audio_hash}",
                ),
            ),
            memories=(),
            tool_results=(),
            token_budget=reservation,
            truncated_items=(),
            messages=(
                {
                    "role": "user",
                    "content": {
                        "audio_sha256": f"sha256:{audio_hash}",
                        "language": request.language,
                        "hotwords_hash": canonical_sha256(request.hotwords),
                        "hotword_profile_id": request.hotword_profile_id,
                    },
                },
            ),
            response_schema={"type": "asr-transcript"},
        )
        await asyncio.to_thread(
            self._admit,
            context,
            request,
            model,
            prompt_hash,
            context_manifest,
            reservation,
            now,
        )
        try:
            result = await self._transport.transcribe(
                model,
                request,
                timeout_seconds=max(0.001, (request.deadline - now).total_seconds()),
            )
            _validate_result(result, duration)
        except ModelGatewayError as exc:
            await asyncio.to_thread(
                self._finish_failure, context, request.inference_request_id, exc.reason
            )
            raise
        except Exception as exc:
            await asyncio.to_thread(
                self._finish_failure,
                context,
                request.inference_request_id,
                "asr_unexpected_failure",
            )
            raise ModelGatewayError("asr_unexpected_failure") from exc
        await asyncio.to_thread(self._finish_success, context, request, model, result, audio_hash)
        return TranscriptionResult(
            text=result.text,
            language=result.language,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            resolved_release_id=model.release_id,
            manifest_hash=model.manifest_hash,
            audio_sha256=audio_hash,
            latency_breakdown=dict(result.latency_breakdown),
            hotword_profile_id=request.hotword_profile_id,
        )

    def _admit(
        self,
        context: TenantContext,
        request: TranscriptionRequest,
        model: ResolvedModel,
        prompt_hash: str,
        context_manifest: ContextManifest,
        reservation: int,
        now: datetime,
    ) -> None:
        with self._database.transaction(context) as session:
            existing = session.get(ModelInferenceRecord, request.inference_request_id)
            if existing is not None:
                if (
                    existing.prompt_hash != prompt_hash
                    or existing.context_hash != context_manifest.context_hash
                ):
                    raise ModelGatewayError("asr_idempotency_conflict")
                raise ModelGatewayError("asr_request_already_consumed")
            require_media_quota(
                session,
                tenant_id=context.tenant_id,
                model_alias=request.model_alias,
                request_class="ASR",
                reservation=reservation,
                limit_reason="asr_segment_limit_exceeded",
                include_pending=False,
                now=now,
            )
            session.add(
                ModelInferenceRecord(
                    inference_request_id=request.inference_request_id,
                    tenant_id=context.tenant_id,
                    model_alias=request.model_alias,
                    resolved_release_id=model.release_id,
                    manifest_hash=model.manifest_hash,
                    subject_id=request.subject_id,
                    trace_id=request.trace_id,
                    agent_run_id=None,
                    diagnosis_run_id=None,
                    prompt_bundle_id=None,
                    index_release_id=None,
                    context_manifest_json=context_manifest.as_dict(),
                    context_hash=context_manifest.context_hash,
                    request_class="ASR",
                    data_classification="CONFIDENTIAL",
                    status="PENDING",
                    deadline=request.deadline,
                    max_output_tokens=reservation,
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

    def _finish_success(
        self,
        context: TenantContext,
        request: TranscriptionRequest,
        model: ResolvedModel,
        result: TranscriptionTransportResult,
        audio_hash: str,
    ) -> None:
        now = datetime.now(UTC)
        payload = {
            "session_id": request.session_id,
            "segment_id": request.segment_id,
            "audio_sha256": audio_hash,
            "transcript_sha256": sha256(result.text.encode()).hexdigest(),
            "confidence": result.confidence,
            "language": result.language,
            "duration_seconds": result.duration_seconds,
            "hotword_profile_id": request.hotword_profile_id,
        }
        with self._database.transaction(context) as session:
            record = session.get(ModelInferenceRecord, request.inference_request_id)
            if record is None or record.status != "PENDING":
                raise ModelGatewayError("asr_audit_state_changed")
            record.status = "SUCCEEDED"
            record.response_json = payload
            record.response_hash = _digest(payload)
            record.usage_prompt_tokens = record.max_output_tokens
            record.finish_reason = "transcribed"
            record.latency_breakdown_json = result.latency_breakdown
            record.safety_decision = "ALLOWED"
            record.completed_at = now
            record.updated_at = now

    def _finish_failure(self, context: TenantContext, request_id: str, reason: str) -> None:
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            record = session.get(ModelInferenceRecord, request_id)
            if record is not None and record.status == "PENDING":
                record.status = "FAILED"
                record.safety_decision = "REJECTED"
                record.failure_reason = reason[:255]
                record.completed_at = now
                record.updated_at = now


def _validate_request(request: TranscriptionRequest, now: datetime) -> float:
    if (
        not request.subject_id
        or not request.trace_id
        or not request.segment_id
        or not request.hotword_profile_id.strip()
    ):
        raise ModelGatewayError("asr_request_identity_invalid")
    if request.deadline.tzinfo is None or request.deadline <= now:
        raise ModelGatewayError("asr_request_deadline_expired")
    if request.language not in {"zh", "en"} or len("，".join(request.hotwords)) > 1_000:
        raise ModelGatewayError("asr_request_contract_invalid")
    return _wav_contract(request.audio_wav)


def _wav_contract(content: bytes) -> float:
    if len(content) < 44 or len(content) > MAX_ASR_AUDIO_BYTES:
        raise ModelGatewayError("asr_audio_contract_invalid")
    try:
        with wave.open(io.BytesIO(content), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != 16_000
            ):
                raise ModelGatewayError("asr_audio_contract_invalid")
            duration = source.getnframes() / 16_000.0
    except (EOFError, wave.Error) as exc:
        raise ModelGatewayError("asr_audio_contract_invalid") from exc
    if not 0.1 <= duration <= MAX_ASR_SEGMENT_SECONDS:
        raise ModelGatewayError("asr_audio_duration_invalid")
    return duration


def _validate_result(result: TranscriptionTransportResult, expected_duration: float) -> None:
    if (
        not result.text.strip()
        or len(result.text) > 4_000
        or result.language not in {"zh", "en"}
        or not 0 <= result.confidence <= 1
        or abs(result.duration_seconds - expected_duration) > 2.0
        or any(not math.isfinite(value) or value < 0 for value in result.latency_breakdown.values())
    ):
        raise ModelTransportError("asr_response_invalid")


def _response_confidence(body: object) -> float:
    if not isinstance(body, dict):
        raise ValueError("ASR body must be an object")
    direct = body.get("confidence")
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return max(0.0, min(1.0, float(direct)))
    segments = body.get("segments")
    scores: list[float] = []
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            confidence = segment.get("confidence")
            log_probability = segment.get("avg_logprob")
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                scores.append(float(confidence))
            elif isinstance(log_probability, (int, float)) and not isinstance(
                log_probability, bool
            ):
                scores.append(math.exp(float(log_probability)))
    return max(0.0, min(1.0, sum(scores) / len(scores))) if scores else 0.5


def _transcription_endpoint(base_url: str, allow_plain_http: bool) -> str:
    parsed = httpx.URL(base_url)
    allowed = {"https"} | ({"http"} if allow_plain_http else set())
    if parsed.scheme not in allowed or parsed.host is None or parsed.query:
        raise ModelTransportError("asr_endpoint_invalid")
    return f"{base_url.rstrip('/')}/v1/audio/transcriptions"
