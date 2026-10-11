"""Short-lived actual-GPU runtime for the governed GRPO Agent release probe."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from contextlib import nullcontext
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.simulation.grpo_agent_runtime_value_lab import (
    verify_grpo_agent_runtime_value,
)
from industrial_ops_agent.simulation.grpo_post_training_lab import (
    verify_grpo_post_training,
)
from industrial_ops_agent.training.grpo_reward_profile import (
    CURRENT_GRPO_REWARD_PROFILE,
    normalize_grpo_completion,
)

ADAPTER_RELATIVE = Path(
    "artifacts/m7-grpo-post-training-lab/runs/grpo-ee9f7768aee8528dfa8d/adapter"
)
MODEL_PATH = Path("/models/base")


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GrpoAgentRuntimeRequest(_ClosedModel):
    request_id: str = Field(min_length=1, max_length=128)
    case_id: str = Field(min_length=1, max_length=256)
    equipment_id: str = Field(pattern=r"^[A-Z0-9][A-Z0-9-]{2,63}$")
    risk: Literal["ELEVATED", "CRITICAL"]
    prompt: str = Field(min_length=1, max_length=8_000)
    max_new_tokens: int = Field(default=48, ge=1, le=64)
    variant: Literal["stable", "candidate"]


class GrpoAgentRuntimeResponse(_ClosedModel):
    schema_version: Literal["enterprise-grpo-agent-runtime-response/v1"] = (
        "enterprise-grpo-agent-runtime-response/v1"
    )
    request_id: str
    case_id: str
    equipment_id: str
    risk: Literal["ELEVATED", "CRITICAL"]
    variant: Literal["stable", "candidate"]
    model_id: str
    model_revision: str
    adapter_enabled: bool
    cache_hit: bool
    output_text: str
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_json: dict[str, Any] | None
    structured_output_valid: bool
    direct_object: bool
    policy_consistent: bool
    human_approval_required: bool
    latency_ms: float = Field(gt=0)
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=0)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnterpriseGrpoAgentRuntime:
    """Load one immutable Qwen base and switch the approved GRPO adapter per call."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_directory: Path,
        adapter_directory: Path,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_directory = model_directory.resolve(strict=True)
        self.adapter_directory = adapter_directory.resolve(strict=True)
        self._lock = threading.Lock()
        self._cache: dict[str, GrpoAgentRuntimeResponse] = {}
        self._request_counts = {"stable": 0, "candidate": 0}
        self._inference_counts = {"stable": 0, "candidate": 0}
        self._cache_hits = {"stable": 0, "candidate": 0}
        self._error_counts = {"stable": 0, "candidate": 0}
        self._started_at = time.time()
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._gpu_name = ""
        self._gpu_total_memory_bytes = 0

    def load(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - runtime image only
            raise RuntimeError("grpo_runtime_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly_one_cuda_gpu_is_required")
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_directory),
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base: Any = AutoModelForCausalLM.from_pretrained(
            str(self.model_directory),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        model: Any = PeftModel.from_pretrained(
            base,
            str(self.adapter_directory),
            is_trainable=False,
        )
        model.eval()
        properties = torch.cuda.get_device_properties(0)
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._gpu_name = torch.cuda.get_device_name(0)
        self._gpu_total_memory_bytes = int(properties.total_memory)

    def infer(self, request: GrpoAgentRuntimeRequest) -> GrpoAgentRuntimeResponse:
        if self._model is None:
            raise RuntimeError("grpo_agent_runtime_is_not_loaded")
        variant = request.variant
        with self._lock:
            self._request_counts[variant] += 1
            try:
                cache_key = _request_cache_key(request)
                existing = self._cache.get(cache_key)
                if existing is not None:
                    self._cache_hits[variant] += 1
                    return _copy_cache_hit(existing, request.request_id)
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
                "schema_version": "enterprise-grpo-agent-runtime-metrics/v1",
                "ready": self._model is not None,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "gpu_name": self._gpu_name,
                "gpu_total_memory_bytes": self._gpu_total_memory_bytes,
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
                "torch_version": version("torch"),
                "transformers_version": version("transformers"),
                "peft_version": version("peft"),
                "uptime_seconds": max(time.time() - self._started_at, 0.0),
            }

    def _execute(
        self,
        request: GrpoAgentRuntimeRequest,
    ) -> GrpoAgentRuntimeResponse:
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        encoded = tokenizer(
            request.prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        inputs = {key: value.to("cuda:0") for key, value in encoded.items()}
        adapter_context = (
            nullcontext() if request.variant == "candidate" else model.disable_adapter()
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
        input_tokens = int(inputs["input_ids"].shape[-1])
        output_ids = generated[0][input_tokens:]
        output = str(tokenizer.decode(output_ids, skip_special_tokens=True)).strip()
        normalized = normalize_grpo_completion(output)
        normalized_value = _json_object(normalized)
        valid = _valid_agent_object(normalized_value)
        expected_action = "HOLD_AND_ESCALATE" if request.risk == "CRITICAL" else "INSPECT"
        expected_priority = "P1" if request.risk == "CRITICAL" else "P2"
        policy_consistent = bool(
            valid
            and normalized_value is not None
            and normalized_value.get("equipment_id") == request.equipment_id
            and normalized_value.get("action") == expected_action
            and normalized_value.get("priority") == expected_priority
        )
        unsigned: dict[str, Any] = {
            "schema_version": "enterprise-grpo-agent-runtime-response/v1",
            "request_id": request.request_id,
            "case_id": request.case_id,
            "equipment_id": request.equipment_id,
            "risk": request.risk,
            "variant": request.variant,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "adapter_enabled": request.variant == "candidate",
            "cache_hit": False,
            "output_text": output,
            "output_sha256": sha256(output.encode()).hexdigest(),
            "normalized_json": normalized_value,
            "structured_output_valid": valid,
            "direct_object": bool(normalized is not None and output.lstrip().startswith("{")),
            "policy_consistent": policy_consistent,
            "human_approval_required": bool(
                normalized_value is not None and normalized_value.get("approval_required") is True
            ),
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": int(output_ids.shape[-1]),
        }
        return GrpoAgentRuntimeResponse(
            **unsigned,
            response_sha256=_digest(unsigned),
        )


def create_app(runtime: EnterpriseGrpoAgentRuntime) -> FastAPI:
    app = FastAPI(title="Enterprise GRPO Agent GPU Worker", version="1.0.0")

    @app.get("/health/ready")
    def ready() -> dict[str, object]:
        metrics = runtime.metrics()
        if metrics["ready"] is not True:
            raise HTTPException(status_code=503, detail="grpo_agent_runtime_not_ready")
        return {
            "ready": True,
            "model_id": metrics["model_id"],
            "model_revision": metrics["model_revision"],
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return runtime.metrics()

    @app.post("/v1/agent/decision", response_model=GrpoAgentRuntimeResponse)
    def decision(request: GrpoAgentRuntimeRequest) -> GrpoAgentRuntimeResponse:
        try:
            return runtime.infer(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def _request_cache_key(request: GrpoAgentRuntimeRequest) -> str:
    return _digest(request.model_dump(mode="json", exclude={"request_id"}))


def _copy_cache_hit(
    response: GrpoAgentRuntimeResponse,
    request_id: str,
) -> GrpoAgentRuntimeResponse:
    unsigned = response.model_dump(
        mode="json",
        exclude={"request_id", "cache_hit", "latency_ms", "response_sha256"},
    )
    unsigned.update(
        {
            "request_id": request_id,
            "cache_hit": True,
            "latency_ms": 0.001,
        }
    )
    return GrpoAgentRuntimeResponse(**unsigned, response_sha256=_digest(unsigned))


def _json_object(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _valid_agent_object(value: dict[str, Any] | None) -> bool:
    if value is None or set(value) != set(CURRENT_GRPO_REWARD_PROFILE.required_keys):
        return False
    return bool(
        value.get("action") in CURRENT_GRPO_REWARD_PROFILE.allowed_actions
        and value.get("priority") in CURRENT_GRPO_REWARD_PROFILE.allowed_priorities
        and value.get("approval_required") is True
        and isinstance(value.get("equipment_id"), str)
        and value["equipment_id"]
    )


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
    parser = argparse.ArgumentParser(prog="industrial-ops-enterprise-grpo-agent-runtime")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18093)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be between 1 and 65535")
    root = args.repo_root.resolve(strict=True)
    value = verify_grpo_agent_runtime_value(root)
    training = verify_grpo_post_training(root, Path(value.grpo_source.acceptance.path))
    if (
        not value.candidate_accepted
        or value.grpo_source.run_id != training.run_id
        or value.grpo_source.adapter_bundle_sha256 != training.adapter.bundle_sha256
    ):
        raise RuntimeError("grpo_agent_release_source_binding_changed")
    runtime = EnterpriseGrpoAgentRuntime(
        model_id=training.base_model.model_id,
        model_revision=training.base_model.revision,
        model_directory=args.model_path,
        adapter_directory=root / Path(training.adapter.path),
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
