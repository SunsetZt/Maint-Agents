"""Short-lived GPU runtime for governed enterprise ASR rollout acceptance."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import threading
import time
import unicodedata
from contextlib import nullcontext
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.simulation.asr_enterprise_value_lab import (
    verify_asr_enterprise_value,
)

MAX_AUDIO_BYTES = 2_000_000
MAX_AUDIO_DURATION_SECONDS = 30.0


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AsrRuntimeRequest(_ClosedModel):
    schema_version: Literal["enterprise-asr-runtime-request/v1"] = (
        "enterprise-asr-runtime-request/v1"
    )
    request_id: str = Field(min_length=1, max_length=128)
    case_id: str = Field(min_length=1, max_length=256)
    audio_base64: str = Field(min_length=1, max_length=2_700_000)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_mime_type: Literal["audio/flac"] = "audio/flac"
    sampling_rate: Literal[16000] = 16000
    language: Literal["en"] = "en"
    max_length: int = Field(default=64, ge=1, le=64)
    variant: Literal["stable", "candidate"]


class AsrRuntimeResponse(_ClosedModel):
    schema_version: Literal["enterprise-asr-runtime-response/v1"] = (
        "enterprise-asr-runtime-response/v1"
    )
    request_id: str
    case_id: str
    variant: Literal["stable", "candidate"]
    model_id: str
    model_revision: str
    adapter_enabled: bool
    cache_hit: bool
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audio_size_bytes: int = Field(gt=0)
    sampling_rate: Literal[16000] = 16000
    sample_count: int = Field(gt=0)
    duration_seconds: float = Field(gt=0.0, le=MAX_AUDIO_DURATION_SECONDS)
    transcript_text: str
    normalized_transcript: str
    latency_ms: float = Field(gt=0.0)
    input_frames: int = Field(ge=1)
    output_tokens: int = Field(ge=0)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnterpriseAsrRuntime:
    """Load one immutable Whisper base and switch its approved LoRA per request."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_directory: Path,
        adapter_directory: Path,
        max_audio_bytes: int = MAX_AUDIO_BYTES,
        max_audio_duration_seconds: float = MAX_AUDIO_DURATION_SECONDS,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_directory = model_directory.resolve(strict=True)
        self.adapter_directory = adapter_directory.resolve(strict=True)
        self.max_audio_bytes = max_audio_bytes
        self.max_audio_duration_seconds = max_audio_duration_seconds
        self._lock = threading.Lock()
        self._cache: dict[str, AsrRuntimeResponse] = {}
        self._request_counts = {"stable": 0, "candidate": 0}
        self._inference_counts = {"stable": 0, "candidate": 0}
        self._cache_hits = {"stable": 0, "candidate": 0}
        self._error_counts = {"stable": 0, "candidate": 0}
        self._started_at = time.time()
        self._torch: Any = None
        self._numpy: Any = None
        self._soundfile: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._gpu_name = ""
        self._gpu_total_memory_bytes = 0
        self._torch_version = ""
        self._transformers_version = ""
        self._peft_version = ""

    def load(self) -> None:
        try:
            import numpy
            import soundfile  # type: ignore[import-untyped]
            import torch
            from peft import PeftModel
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:  # pragma: no cover - runtime image only
            raise RuntimeError("asr_runtime_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly_one_cuda_gpu_is_required")
        torch.set_num_threads(2)
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            str(self.model_directory),
            local_files_only=True,
            language="en",
            task="transcribe",
            trust_remote_code=False,
        )
        model: Any = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(self.model_directory),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(
            model,
            str(self.adapter_directory),
            is_trainable=False,
        )
        model.eval()
        model.config.use_cache = True
        self._torch = torch
        self._numpy = numpy
        self._soundfile = soundfile
        self._processor = processor
        self._model = model
        properties = torch.cuda.get_device_properties(0)
        self._gpu_name = torch.cuda.get_device_name(0)
        self._gpu_total_memory_bytes = int(properties.total_memory)
        self._torch_version = str(torch.__version__)
        self._transformers_version = version("transformers")
        self._peft_version = version("peft")

    def infer(self, request: AsrRuntimeRequest) -> AsrRuntimeResponse:
        if self._model is None:
            raise RuntimeError("asr_runtime_is_not_loaded")
        variant = request.variant
        with self._lock:
            self._request_counts[variant] += 1
            try:
                audio_bytes = _decode_audio(request, max_bytes=self.max_audio_bytes)
                cache_key = _request_cache_key(request)
                existing = self._cache.get(cache_key)
                if existing is not None:
                    self._cache_hits[variant] += 1
                    return _copy_cache_hit(existing, request_id=request.request_id)
                result = self._execute(request, audio_bytes)
                self._cache[cache_key] = result
                self._inference_counts[variant] += 1
                return result
            except Exception:
                self._error_counts[variant] += 1
                raise

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            torch = self._torch
            memory_allocated = (
                int(torch.cuda.memory_allocated(0)) if torch is not None else 0
            )
            memory_reserved = int(torch.cuda.memory_reserved(0)) if torch is not None else 0
            peak_memory_reserved = (
                int(torch.cuda.max_memory_reserved(0)) if torch is not None else 0
            )
            return {
                "schema_version": "enterprise-asr-runtime-metrics/v1",
                "ready": self._model is not None,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "gpu_name": self._gpu_name,
                "gpu_total_memory_bytes": self._gpu_total_memory_bytes,
                "torch_version": self._torch_version,
                "transformers_version": self._transformers_version,
                "peft_version": self._peft_version,
                "gpu_memory_allocated_bytes": memory_allocated,
                "gpu_memory_reserved_bytes": memory_reserved,
                "gpu_peak_memory_reserved_bytes": peak_memory_reserved,
                "request_counts": dict(self._request_counts),
                "actual_inference_counts": dict(self._inference_counts),
                "cache_hits": dict(self._cache_hits),
                "error_counts": dict(self._error_counts),
                "cache_entries": len(self._cache),
                "uptime_seconds": max(time.time() - self._started_at, 0.0),
            }

    def _execute(
        self,
        request: AsrRuntimeRequest,
        audio_bytes: bytes,
    ) -> AsrRuntimeResponse:
        torch = self._torch
        numpy = self._numpy
        soundfile = self._soundfile
        processor = self._processor
        model = self._model
        audio, sampling_rate = soundfile.read(
            BytesIO(audio_bytes),
            dtype="float32",
            always_2d=False,
        )
        if sampling_rate != request.sampling_rate:
            raise ValueError("asr_audio_sampling_rate_mismatch")
        if getattr(audio, "ndim", 0) == 2:
            audio = numpy.mean(audio, axis=1, dtype=numpy.float32)
        if getattr(audio, "ndim", 0) != 1 or len(audio) == 0:
            raise ValueError("asr_audio_shape_is_invalid")
        if not bool(numpy.isfinite(audio).all()):
            raise ValueError("asr_audio_contains_non_finite_samples")
        sample_count = int(len(audio))
        duration_seconds = sample_count / float(sampling_rate)
        if not 0.05 <= duration_seconds <= self.max_audio_duration_seconds:
            raise ValueError("asr_audio_duration_is_invalid")
        inputs = processor.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        features = inputs.input_features.to(device="cuda:0", dtype=torch.bfloat16)
        attention_mask = inputs.attention_mask.to(device="cuda:0")
        input_frames = int(features.shape[-1])
        adapter_context = (
            model.disable_adapter() if request.variant == "stable" else nullcontext()
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with adapter_context, torch.inference_mode():
            generated = model.generate(
                input_features=features,
                attention_mask=attention_mask,
                max_length=request.max_length,
                do_sample=False,
                language=request.language,
                task="transcribe",
            )
        torch.cuda.synchronize()
        latency_ms = max((time.perf_counter() - started) * 1_000.0, 1e-9)
        transcript = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        unsigned: dict[str, Any] = {
            "schema_version": "enterprise-asr-runtime-response/v1",
            "request_id": request.request_id,
            "case_id": request.case_id,
            "variant": request.variant,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "adapter_enabled": request.variant == "candidate",
            "cache_hit": False,
            "audio_sha256": request.audio_sha256,
            "audio_size_bytes": len(audio_bytes),
            "sampling_rate": sampling_rate,
            "sample_count": sample_count,
            "duration_seconds": duration_seconds,
            "transcript_text": transcript,
            "normalized_transcript": normalize_transcript(transcript),
            "latency_ms": latency_ms,
            "input_frames": input_frames,
            "output_tokens": int(generated.shape[-1]),
        }
        return AsrRuntimeResponse(**unsigned, response_sha256=_digest(unsigned))


def create_app(runtime: EnterpriseAsrRuntime) -> FastAPI:
    app = FastAPI(title="Enterprise ASR GPU Worker", version="1.0.0")

    @app.get("/health/ready")
    def ready() -> dict[str, object]:
        metrics = runtime.metrics()
        if metrics["ready"] is not True:
            raise HTTPException(status_code=503, detail="asr_runtime_not_ready")
        return {
            "ready": True,
            "model_id": metrics["model_id"],
            "model_revision": metrics["model_revision"],
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return runtime.metrics()

    @app.post("/v1/asr/transcribe", response_model=AsrRuntimeResponse)
    def transcribe(request: AsrRuntimeRequest) -> AsrRuntimeResponse:
        try:
            return runtime.infer(request)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def _decode_audio(request: AsrRuntimeRequest, *, max_bytes: int) -> bytes:
    try:
        payload = base64.b64decode(request.audio_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("asr_audio_base64_is_invalid") from exc
    if not payload or len(payload) > max_bytes:
        raise ValueError("asr_audio_size_is_invalid")
    if sha256(payload).hexdigest() != request.audio_sha256:
        raise ValueError("asr_audio_sha256_mismatch")
    return payload


def _request_cache_key(request: AsrRuntimeRequest) -> str:
    payload = request.model_dump(mode="json", exclude={"request_id", "audio_base64"})
    return _digest(payload)


def _copy_cache_hit(
    response: AsrRuntimeResponse,
    *,
    request_id: str,
) -> AsrRuntimeResponse:
    unsigned = response.model_dump(
        mode="json",
        exclude={"request_id", "cache_hit", "response_sha256"},
    )
    unsigned.update({"request_id": request_id, "cache_hit": True})
    return AsrRuntimeResponse(**unsigned, response_sha256=_digest(unsigned))


def normalize_transcript(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    words = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z0-9]+", normalized)
    return " ".join(words)


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
    parser = argparse.ArgumentParser(prog="industrial-ops-enterprise-asr-runtime")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18094)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be between 1 and 65535")
    root = args.repo_root.resolve(strict=True)
    report = verify_asr_enterprise_value(root)
    if not all(report.hard_gates.model_dump().values()):
        raise RuntimeError("asr_authoritative_hard_gates_are_incomplete")
    runtime = EnterpriseAsrRuntime(
        model_id=report.base_model.model_id,
        model_revision=report.base_model.revision,
        model_directory=root / report.base_model.snapshot_path,
        adapter_directory=root / report.adapter.path,
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    runtime.load()
    import uvicorn

    uvicorn.run(
        create_app(runtime),
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
