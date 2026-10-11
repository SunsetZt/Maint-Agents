"""Short-lived GPU runtime for governed enterprise VLM rollout acceptance."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import threading
import time
from contextlib import nullcontext
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.multimodal.vlm_output import build_vlm_findings_instruction
from industrial_ops_agent.simulation.vlm_enterprise_staging import (
    verify_enterprise_vlm_staging_adoption,
)

MODEL_CACHE_RELATIVE = Path(
    ".cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots"
)
ADAPTER_RELATIVE = Path(
    "artifacts/m7-vlm-checkpoint-continuation-rescored-lab/"
    "adapters/lora-continuation"
)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VlmRuntimeRequest(_ClosedModel):
    request_id: str = Field(min_length=1, max_length=128)
    case_id: str = Field(min_length=1, max_length=256)
    instruction: str = Field(min_length=1, max_length=8_000)
    allowed_labels: tuple[str, ...] = Field(min_length=1, max_length=100)
    image_base64: str = Field(min_length=1, max_length=8_000_000)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_new_tokens: int = Field(default=96, ge=1, le=256)
    variant: Literal["stable", "candidate"]


class VlmFindingRegion(_ClosedModel):
    x: float
    y: float
    width: float
    height: float


class VlmRuntimeFinding(_ClosedModel):
    label: str = Field(min_length=1, max_length=128)
    region: VlmFindingRegion


class VlmRuntimeResponse(_ClosedModel):
    schema_version: Literal["enterprise-vlm-runtime-response/v1"] = (
        "enterprise-vlm-runtime-response/v1"
    )
    request_id: str
    case_id: str
    variant: Literal["stable", "candidate"]
    model_id: str
    model_revision: str
    adapter_enabled: bool
    cache_hit: bool
    output_text: str
    structured_output_valid: bool
    structured_output_repair: Literal["NONE", "BALANCED_CLOSING_DELIMITERS"]
    findings: tuple[VlmRuntimeFinding, ...]
    latency_ms: float = Field(gt=0)
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=0)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnterpriseVlmRuntime:
    """Load one immutable base model and switch the approved adapter per request."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_directory: Path,
        adapter_directory: Path,
        max_image_pixels: int = 307_200,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_directory = model_directory.resolve(strict=True)
        self.adapter_directory = adapter_directory.resolve(strict=True)
        self.max_image_pixels = max_image_pixels
        self._lock = threading.Lock()
        self._cache: dict[str, VlmRuntimeResponse] = {}
        self._request_counts = {"stable": 0, "candidate": 0}
        self._inference_counts = {"stable": 0, "candidate": 0}
        self._cache_hits = {"stable": 0, "candidate": 0}
        self._error_counts = {"stable": 0, "candidate": 0}
        self._started_at = time.time()
        self._torch: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._gpu_name = ""
        self._gpu_total_memory_bytes = 0

    def load(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - runtime environment only
            raise RuntimeError("vlm_runtime_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly_one_cuda_gpu_is_required")
        torch.set_num_threads(2)
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            str(self.model_directory),
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("vlm_processor_has_no_tokenizer")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model: Any = AutoModelForImageTextToText.from_pretrained(
            str(self.model_directory),
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=False,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(
            model,
            str(self.adapter_directory),
            is_trainable=False,
        )
        model.eval()
        self._torch = torch
        self._processor = processor
        self._tokenizer = tokenizer
        self._model = model
        properties = torch.cuda.get_device_properties(0)
        self._gpu_name = torch.cuda.get_device_name(0)
        self._gpu_total_memory_bytes = int(properties.total_memory)

    def infer(self, request: VlmRuntimeRequest) -> VlmRuntimeResponse:
        if self._model is None:
            raise RuntimeError("vlm_runtime_is_not_loaded")
        variant = request.variant
        with self._lock:
            self._request_counts[variant] += 1
            try:
                image_bytes = _decode_image(request)
                cache_key = _request_cache_key(request)
                existing = self._cache.get(cache_key)
                if existing is not None:
                    self._cache_hits[variant] += 1
                    return _copy_cache_hit(existing)
                result = self._execute(request, image_bytes)
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
                "schema_version": "enterprise-vlm-runtime-metrics/v1",
                "ready": self._model is not None,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "gpu_name": self._gpu_name,
                "gpu_total_memory_bytes": self._gpu_total_memory_bytes,
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
        request: VlmRuntimeRequest,
        image_bytes: bytes,
    ) -> VlmRuntimeResponse:
        from PIL import Image

        torch = self._torch
        processor = self._processor
        tokenizer = self._tokenizer
        model = self._model
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            image = source.convert("RGB")
        if image.width * image.height > self.max_image_pixels:
            raise ValueError("vlm_image_exceeds_registered_pixel_limit")
        instruction = build_vlm_findings_instruction(
            request.instruction,
            allowed_labels=request.allowed_labels,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])
        adapter_context = (
            model.disable_adapter() if request.variant == "stable" else nullcontext()
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with adapter_context, torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        latency_ms = max((time.perf_counter() - started) * 1_000, 1e-9)
        output_ids = generated[0][input_tokens:]
        output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        findings, valid, repair = _parse_findings(output)
        unsigned: dict[str, Any] = {
            "schema_version": "enterprise-vlm-runtime-response/v1",
            "request_id": request.request_id,
            "case_id": request.case_id,
            "variant": request.variant,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "adapter_enabled": request.variant == "candidate",
            "cache_hit": False,
            "output_text": output,
            "structured_output_valid": valid,
            "structured_output_repair": repair,
            "findings": [item.model_dump(mode="json") for item in findings],
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": int(output_ids.shape[-1]),
        }
        return VlmRuntimeResponse(
            **unsigned,
            response_sha256=_digest(unsigned),
        )


def create_app(runtime: EnterpriseVlmRuntime) -> FastAPI:
    app = FastAPI(title="Enterprise VLM GPU Worker", version="1.0.0")

    @app.get("/health/ready")
    def ready() -> dict[str, object]:
        metrics = runtime.metrics()
        if metrics["ready"] is not True:
            raise HTTPException(status_code=503, detail="vlm_runtime_not_ready")
        return {
            "ready": True,
            "model_id": metrics["model_id"],
            "model_revision": metrics["model_revision"],
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return runtime.metrics()

    @app.post("/v1/vlm/predict", response_model=VlmRuntimeResponse)
    def predict(request: VlmRuntimeRequest) -> VlmRuntimeResponse:
        try:
            return runtime.infer(request)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def _decode_image(request: VlmRuntimeRequest) -> bytes:
    try:
        payload = base64.b64decode(request.image_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("vlm_image_base64_is_invalid") from exc
    if not payload or len(payload) > 8_000_000:
        raise ValueError("vlm_image_size_is_invalid")
    if sha256(payload).hexdigest() != request.image_sha256:
        raise ValueError("vlm_image_sha256_mismatch")
    return payload


def _request_cache_key(request: VlmRuntimeRequest) -> str:
    payload = request.model_dump(mode="json", exclude={"request_id", "image_base64"})
    return _digest(payload)


def _copy_cache_hit(response: VlmRuntimeResponse) -> VlmRuntimeResponse:
    unsigned = response.model_dump(
        mode="json",
        exclude={"cache_hit", "response_sha256"},
    )
    unsigned["cache_hit"] = True
    return VlmRuntimeResponse(**unsigned, response_sha256=_digest(unsigned))


def _parse_findings(
    output: str,
) -> tuple[
    tuple[VlmRuntimeFinding, ...],
    bool,
    Literal["NONE", "BALANCED_CLOSING_DELIMITERS"],
]:
    candidate = output.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1])
            if candidate.lstrip().startswith("json"):
                candidate = candidate.lstrip()[4:].lstrip()
    repair: Literal["NONE", "BALANCED_CLOSING_DELIMITERS"] = "NONE"
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        completed = _complete_json(candidate)
        if completed is None:
            return (), False, repair
        try:
            value = json.loads(completed)
        except json.JSONDecodeError:
            return (), False, repair
        repair = "BALANCED_CLOSING_DELIMITERS"
    if not isinstance(value, dict) or set(value) != {"findings"}:
        return (), False, repair
    raw_findings = value["findings"]
    if not isinstance(raw_findings, list) or len(raw_findings) > 100:
        return (), False, repair
    findings: list[VlmRuntimeFinding] = []
    for item in raw_findings:
        if not isinstance(item, dict) or set(item) != {"label", "region"}:
            return (), False, repair
        label = item["label"]
        region = item["region"]
        if (
            not isinstance(label, str)
            or not label.strip()
            or len(label) > 128
            or not isinstance(region, dict)
            or set(region) != {"x", "y", "width", "height"}
        ):
            return (), False, repair
        coordinates: dict[str, float] = {}
        for key in ("x", "y", "width", "height"):
            coordinate = region[key]
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                return (), False, repair
            coordinates[key] = float(coordinate)
        x = coordinates["x"]
        y = coordinates["y"]
        width = coordinates["width"]
        height = coordinates["height"]
        if (
            not 0 <= x < 1
            or not 0 <= y < 1
            or not 0 < width <= 1
            or not 0 < height <= 1
            or x + width > 1
            or y + height > 1
        ):
            return (), False, repair
        findings.append(
            VlmRuntimeFinding(
                label=label.strip(),
                region=VlmFindingRegion(**coordinates),
            )
        )
    return tuple(findings), True, repair


def _complete_json(candidate: str) -> str | None:
    if not candidate or len(candidate) > 65_536:
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for character in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}":
            expected = "[" if character == "]" else "{"
            if not stack or stack.pop() != expected:
                return None
    if in_string or escaped:
        return None
    return candidate + "".join("]" if item == "[" else "}" for item in reversed(stack))


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
    parser = argparse.ArgumentParser(prog="industrial-ops-enterprise-vlm-runtime")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18091)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be between 1 and 65535")
    root = args.repo_root.resolve(strict=True)
    adoption = verify_enterprise_vlm_staging_adoption(
        root,
        root / "artifacts/m7-vlm-enterprise-staging/acceptance.json",
    )
    model_directory = (
        Path.home() / MODEL_CACHE_RELATIVE / adoption.candidate.model_revision
    )
    runtime = EnterpriseVlmRuntime(
        model_id=adoption.candidate.model_id,
        model_revision=adoption.candidate.model_revision,
        model_directory=model_directory,
        adapter_directory=root / ADAPTER_RELATIVE,
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
