"""Short-lived GPU runtime for governed enterprise TTS releases."""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
import time
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

MAX_INPUT_CHARACTERS = 6_000
MAX_INPUT_TOKENS = 1_024
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 90.0
SAMPLING_RATE = 16_000


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TtsRuntimeRequest(_ClosedModel):
    """OpenAI-compatible request plus the internal rollout variant."""

    model: str = Field(min_length=1, max_length=256)
    input: str = Field(min_length=1, max_length=MAX_INPUT_CHARACTERS)
    voice: str = Field(default="default", min_length=1, max_length=64)
    response_format: Literal["wav"] = "wav"
    speed: float = Field(default=1.0, ge=1.0, le=1.0)
    variant: Literal["stable", "candidate"]


class TtsHttpRequest(_ClosedModel):
    """Public speech request; a dedicated release runtime may lock the variant."""

    model: str = Field(min_length=1, max_length=256)
    input: str = Field(min_length=1, max_length=MAX_INPUT_CHARACTERS)
    voice: str = Field(default="default", min_length=1, max_length=64)
    response_format: Literal["wav"] = "wav"
    speed: float = Field(default=1.0, ge=1.0, le=1.0)
    variant: Literal["stable", "candidate"] | None = None


@dataclass(frozen=True, slots=True)
class TtsAudioResult:
    content: bytes
    variant: Literal["stable", "candidate"]
    model_revision: str
    cache_hit: bool
    audio_sha256: str
    duration_seconds: float
    input_tokens: int
    latency_ms: float


class EnterpriseTtsRuntime:
    """Serve one SpeechT5 base and an approved postnet candidate with bounded memory."""

    def __init__(
        self,
        *,
        served_model_name: str,
        model_revision: str,
        model_directory: Path,
        candidate_directory: Path,
        vocoder_directory: Path,
        speaker_embedding_path: Path,
        allowed_voices: frozenset[str] = frozenset({"default"}),
    ) -> None:
        if not served_model_name.strip() or not model_revision.strip():
            raise ValueError("tts_runtime_model_identity_is_invalid")
        if not allowed_voices or any(not voice.strip() for voice in allowed_voices):
            raise ValueError("tts_runtime_voice_policy_is_invalid")
        self.served_model_name = served_model_name
        self.model_revision = model_revision
        self.model_directory = model_directory.resolve(strict=True)
        self.candidate_directory = candidate_directory.resolve(strict=True)
        self.vocoder_directory = vocoder_directory.resolve(strict=True)
        self.speaker_embedding_path = speaker_embedding_path.resolve(strict=True)
        self.allowed_voices = allowed_voices
        self._lock = threading.Lock()
        self._cache: dict[str, TtsAudioResult] = {}
        self._request_counts = {"stable": 0, "candidate": 0}
        self._inference_counts = {"stable": 0, "candidate": 0}
        self._cache_hits = {"stable": 0, "candidate": 0}
        self._error_counts = {"stable": 0, "candidate": 0}
        self._variant_switches = 0
        self._active_variant: Literal["stable", "candidate"] = "stable"
        self._started_at = time.time()
        self._torch: Any = None
        self._numpy: Any = None
        self._soundfile: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._vocoder: Any = None
        self._speaker_embedding: Any = None
        self._postnet_states: dict[str, dict[str, Any]] = {}
        self._gpu_name = ""
        self._gpu_total_memory_bytes = 0
        self._torch_version = ""
        self._transformers_version = ""

    def load(self) -> None:
        """Load and verify stable/candidate weights without retaining two GPU models."""

        try:
            import numpy
            import soundfile  # type: ignore[import-untyped]
            import torch
            from transformers import (
                SpeechT5ForTextToSpeech,
                SpeechT5HifiGan,
                SpeechT5Processor,
            )
        except ImportError as exc:  # pragma: no cover - runtime image only
            raise RuntimeError("tts_runtime_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly_one_cuda_gpu_is_required")
        torch.set_num_threads(2)
        speaker = numpy.load(self.speaker_embedding_path, allow_pickle=False)
        speaker = numpy.asarray(speaker, dtype="float32").reshape(-1)
        if speaker.shape != (512,) or not bool(numpy.isfinite(speaker).all()):
            raise RuntimeError("tts_runtime_speaker_embedding_is_invalid")
        processor = SpeechT5Processor.from_pretrained(
            str(self.model_directory), local_files_only=True, trust_remote_code=False
        )
        model: Any = SpeechT5ForTextToSpeech.from_pretrained(
            str(self.model_directory),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
        )
        model = model.to("cuda:0")
        model.eval()
        model.config.use_cache = True
        stable_state = _cpu_state(model.speech_decoder_postnet.layers.state_dict())
        candidate: Any = SpeechT5ForTextToSpeech.from_pretrained(
            str(self.candidate_directory),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
        )
        candidate_state = _cpu_state(candidate.speech_decoder_postnet.layers.state_dict())
        if set(stable_state) != set(candidate_state):
            raise RuntimeError("tts_candidate_postnet_contract_changed")
        candidate = None
        gc.collect()
        vocoder: Any = SpeechT5HifiGan.from_pretrained(
            str(self.vocoder_directory),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        vocoder = vocoder.to("cuda:0")
        vocoder.eval()
        properties = torch.cuda.get_device_properties(0)
        self._torch = torch
        self._numpy = numpy
        self._soundfile = soundfile
        self._processor = processor
        self._model = model
        self._vocoder = vocoder
        self._speaker_embedding = speaker
        self._postnet_states = {"stable": stable_state, "candidate": candidate_state}
        self._gpu_name = torch.cuda.get_device_name(0)
        self._gpu_total_memory_bytes = int(properties.total_memory)
        self._torch_version = str(torch.__version__)
        self._transformers_version = version("transformers")

    def infer(self, request: TtsRuntimeRequest) -> TtsAudioResult:
        if self._model is None:
            raise RuntimeError("tts_runtime_is_not_loaded")
        self._validate_request(request)
        variant = request.variant
        with self._lock:
            self._request_counts[variant] += 1
            try:
                cache_key = _request_cache_key(request)
                existing = self._cache.get(cache_key)
                if existing is not None:
                    self._cache_hits[variant] += 1
                    return replace(existing, cache_hit=True)
                result = self._execute(request)
                self._cache[cache_key] = result
                self._inference_counts[variant] += 1
                return result
            except Exception:
                self._error_counts[variant] += 1
                raise

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            torch = self._torch
            return {
                "schema_version": "enterprise-tts-runtime-metrics/v1",
                "ready": self._model is not None and self._vocoder is not None,
                "served_model_name": self.served_model_name,
                "model_revision": self.model_revision,
                "active_variant": self._active_variant,
                "variant_switches": self._variant_switches,
                "gpu_name": self._gpu_name,
                "gpu_total_memory_bytes": self._gpu_total_memory_bytes,
                "torch_version": self._torch_version,
                "transformers_version": self._transformers_version,
                "gpu_memory_allocated_bytes": (
                    int(torch.cuda.memory_allocated(0)) if torch is not None else 0
                ),
                "gpu_memory_reserved_bytes": (
                    int(torch.cuda.memory_reserved(0)) if torch is not None else 0
                ),
                "gpu_peak_memory_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(0)) if torch is not None else 0
                ),
                "request_counts": dict(self._request_counts),
                "actual_inference_counts": dict(self._inference_counts),
                "cache_hits": dict(self._cache_hits),
                "error_counts": dict(self._error_counts),
                "cache_entries": len(self._cache),
                "uptime_seconds": max(time.time() - self._started_at, 0.0),
            }

    def _validate_request(self, request: TtsRuntimeRequest) -> None:
        if request.model != self.served_model_name:
            raise ValueError("tts_requested_model_is_not_served")
        if request.voice not in self.allowed_voices:
            raise ValueError("tts_requested_voice_is_not_authorized")
        if not request.input.strip():
            raise ValueError("tts_input_is_blank")

    def _execute(self, request: TtsRuntimeRequest) -> TtsAudioResult:
        torch = self._torch
        numpy = self._numpy
        soundfile = self._soundfile
        processor = self._processor
        model = self._model
        vocoder = self._vocoder
        if request.variant != self._active_variant:
            model.speech_decoder_postnet.layers.load_state_dict(
                self._postnet_states[request.variant], strict=True
            )
            self._active_variant = request.variant
            self._variant_switches += 1
        encoded = processor(text=request.input, return_tensors="pt")
        input_tokens = int(encoded.input_ids.shape[-1])
        if input_tokens <= 0 or input_tokens > MAX_INPUT_TOKENS:
            raise ValueError("tts_input_token_count_is_invalid")
        embedding = torch.from_numpy(self._speaker_embedding[None, :]).to(
            device="cuda:0", dtype=model.dtype
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            waveform = model.generate_speech(
                input_ids=encoded.input_ids.to("cuda:0"),
                attention_mask=encoded.attention_mask.to("cuda:0"),
                speaker_embeddings=embedding,
                vocoder=vocoder,
                threshold=0.5,
                minlenratio=0.0,
                maxlenratio=20.0,
            )
        torch.cuda.synchronize()
        latency_ms = max((time.perf_counter() - started) * 1_000.0, 1e-9)
        audio = waveform.detach().float().cpu().numpy().reshape(-1)
        duration_seconds = len(audio) / float(SAMPLING_RATE)
        if (
            len(audio) == 0
            or not bool(numpy.isfinite(audio).all())
            or duration_seconds > MAX_AUDIO_DURATION_SECONDS
            or float(numpy.mean(numpy.abs(audio) <= 0.0001)) >= 0.98
            or float(numpy.mean(numpy.abs(audio) >= 0.999)) > 0.01
        ):
            raise RuntimeError("tts_generated_audio_integrity_failed")
        buffer = BytesIO()
        soundfile.write(buffer, audio, SAMPLING_RATE, format="WAV", subtype="PCM_16")
        content = buffer.getvalue()
        if not content or len(content) > MAX_AUDIO_BYTES:
            raise RuntimeError("tts_generated_audio_size_is_invalid")
        return TtsAudioResult(
            content=content,
            variant=request.variant,
            model_revision=self.model_revision,
            cache_hit=False,
            audio_sha256=sha256(content).hexdigest(),
            duration_seconds=duration_seconds,
            input_tokens=input_tokens,
            latency_ms=latency_ms,
        )


def create_app(
    runtime: EnterpriseTtsRuntime,
    *,
    fixed_variant: Literal["stable", "candidate"] | None = None,
) -> FastAPI:
    app = FastAPI(title="Enterprise TTS GPU Worker", version="1.0.0")

    @app.get("/health/ready")
    def ready() -> dict[str, object]:
        metrics = runtime.metrics()
        if metrics["ready"] is not True:
            raise HTTPException(status_code=503, detail="tts_runtime_not_ready")
        return {
            "ready": True,
            "served_model_name": metrics["served_model_name"],
            "model_revision": metrics["model_revision"],
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return runtime.metrics()

    @app.post("/v1/audio/speech")
    def synthesize(request: TtsHttpRequest) -> Response:
        try:
            variant = fixed_variant or request.variant
            if variant is None:
                raise ValueError("tts_runtime_variant_is_required")
            if fixed_variant is not None and request.variant not in {None, fixed_variant}:
                raise ValueError("tts_runtime_variant_is_locked")
            runtime_request = TtsRuntimeRequest(
                **request.model_dump(exclude={"variant"}),
                variant=variant,
            )
            result = runtime.infer(runtime_request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            content=result.content,
            media_type="audio/wav",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
                "X-IOAP-TTS-Variant": result.variant,
                "X-IOAP-TTS-Model-Revision": result.model_revision,
                "X-IOAP-TTS-Audio-SHA256": result.audio_sha256,
                "X-IOAP-TTS-Cache-Hit": str(result.cache_hit).lower(),
                "X-IOAP-TTS-Duration-Seconds": f"{result.duration_seconds:.6f}",
                "X-IOAP-TTS-Input-Tokens": str(result.input_tokens),
                "X-IOAP-TTS-Latency-Ms": f"{result.latency_ms:.6f}",
            },
        )

    return app


def _cpu_state(state: dict[str, Any]) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _request_cache_key(request: TtsRuntimeRequest) -> str:
    return _digest(request.model_dump(mode="json"))


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-enterprise-tts-runtime")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18095)
    parser.add_argument("--served-model-name")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--vocoder-dir", type=Path)
    parser.add_argument("--speaker-embedding-path", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--allowed-voice", action="append", default=[])
    parser.add_argument("--fixed-variant", choices=("stable", "candidate"))
    return parser


def main(argv: list[str] | None = None) -> int:
    from industrial_ops_agent.simulation.tts_enterprise_value_lab import (
        BASE_MODEL_RELATIVE,
        VOCODER_RELATIVE,
        verify_tts_enterprise_value,
    )

    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be between 1 and 65535")
    direct_paths = (
        args.model_dir,
        args.candidate_dir,
        args.vocoder_dir,
        args.speaker_embedding_path,
    )
    if any(value is not None for value in direct_paths):
        if (
            any(value is None for value in direct_paths)
            or not args.served_model_name
            or not args.model_revision
        ):
            raise ValueError("direct TTS runtime requires complete immutable model coordinates")
        runtime = EnterpriseTtsRuntime(
            served_model_name=args.served_model_name,
            model_revision=args.model_revision,
            model_directory=args.model_dir,
            candidate_directory=args.candidate_dir,
            vocoder_directory=args.vocoder_dir,
            speaker_embedding_path=args.speaker_embedding_path,
            allowed_voices=frozenset(args.allowed_voice or ["default"]),
        )
    else:
        root = args.repo_root.resolve(strict=True)
        report = verify_tts_enterprise_value(root)
        if not all(report.hard_gates.model_dump().values()):
            raise RuntimeError("tts_authoritative_hard_gates_are_incomplete")
        runtime = EnterpriseTtsRuntime(
            served_model_name=args.served_model_name or report.run_id,
            model_revision=next(
                item.revision for item in report.input_components if item.role == "base_model"
            ),
            model_directory=root / BASE_MODEL_RELATIVE,
            candidate_directory=root / report.candidate.path,
            vocoder_directory=root / VOCODER_RELATIVE,
            speaker_embedding_path=root / report.dataset.speaker_embedding.path,
        )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    runtime.load()
    import uvicorn

    uvicorn.run(
        create_app(runtime, fixed_variant=args.fixed_variant),
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
