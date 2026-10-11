"""Small governed Transformers runtime used by the local KServe promotion lab."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from industrial_ops_agent.guardrails.prompt_injection import PromptInjectionGuard
from industrial_ops_agent.training.dataset import SYSTEM_INSTRUCTION


class RuntimeContractError(RuntimeError):
    pass


@dataclass(slots=True)
class RuntimeState:
    variant: str
    identity: dict[str, Any]
    tokenizer: Any
    model: Any
    torch: Any
    guardrail: PromptInjectionGuard
    generation_lock: threading.Lock
    request_lock: threading.Lock
    predict_requests: int = 0
    blocked_requests: int = 0
    route_probe_requests: int = 0

    def predict(self, evidence: dict[str, Any]) -> dict[str, Any]:
        canonical = json.dumps(
            evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        input_decision = self.guardrail.inspect_text(canonical, source_role="evidence")
        if input_decision.decision == "BLOCKED":
            with self.request_lock:
                self.blocked_requests += 1
            raise RuntimeContractError("prompt_injection_blocked")
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": "请根据以下脱敏工业工单证据完成根因分类。\n"
                f"evidence={canonical}",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        started = perf_counter()
        with self.generation_lock, self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=96,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        input_tokens = int(encoded["input_ids"].shape[-1])
        output = self.tokenizer.decode(
            generated[0][input_tokens:], skip_special_tokens=True
        ).strip()
        normalized = _normalized_json(output)
        try:
            decoded = json.loads(normalized)
        except json.JSONDecodeError:
            parsed = {"unparsed_output_sha256": sha256(output.encode()).hexdigest()}
        else:
            parsed = (
                decoded
                if isinstance(decoded, dict)
                else {"unparsed_output_sha256": sha256(output.encode()).hexdigest()}
            )
        output_decision = self.guardrail.inspect_output(parsed)
        if output_decision.decision == "BLOCKED":
            with self.request_lock:
                self.blocked_requests += 1
            raise RuntimeContractError("model_output_guardrail_blocked")
        with self.request_lock:
            self.predict_requests += 1
        return {
            "output": parsed,
            "variant": self.variant,
            "latency_ms": (perf_counter() - started) * 1000,
            "input_tokens": input_tokens,
            "output_tokens": int(generated[0].shape[-1]) - input_tokens,
            "production_claim": False,
        }

    def route_probe(self) -> dict[str, Any]:
        with self.request_lock:
            self.route_probe_requests += 1
            count = self.route_probe_requests
        return {"variant": self.variant, "route_probe_count": count}

    def stats(self) -> dict[str, Any]:
        with self.request_lock:
            predict_requests = self.predict_requests
            blocked_requests = self.blocked_requests
            route_probe_requests = self.route_probe_requests
        return {
            "variant": self.variant,
            "predict_requests": predict_requests,
            "blocked_requests": blocked_requests,
            "route_probe_requests": route_probe_requests,
            "guardrail_policy_version": self.guardrail.policy_version,
            "identity": self.identity,
            "production_claim": False,
        }


class _Handler(BaseHTTPRequestHandler):
    server: _RuntimeServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/health", "/health/ready", "/v2/health/ready"}:
            self._json(HTTPStatus.OK, {"status": "ready"})
            return
        if self.path == "/identity":
            self._json(HTTPStatus.OK, self.server.state.stats())
            return
        if self.path == "/stats":
            self._json(HTTPStatus.OK, self.server.state.stats())
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/route-probe":
            self._json(HTTPStatus.OK, self.server.state.route_probe())
            return
        if not self.path.endswith(":predict"):
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            body = self._body()
            raw_instances = body.get("instances")
            if not isinstance(raw_instances, list) or len(raw_instances) != 1:
                raise RuntimeContractError("exactly_one_instance_is_required")
            evidence = raw_instances[0]
            if not isinstance(evidence, dict) or not evidence:
                raise RuntimeContractError("prediction_evidence_is_invalid")
            prediction = self.server.state.predict(cast(dict[str, Any], evidence))
        except (RuntimeContractError, UnicodeError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._json(HTTPStatus.OK, {"predictions": [prediction]})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "")
        if not raw_length.isdigit() or not 0 < int(raw_length) <= 64 * 1024:
            raise RuntimeContractError("request_size_is_invalid")
        value = json.loads(self.rfile.read(int(raw_length)).decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeContractError("request_body_must_be_an_object")
        return cast(dict[str, Any], value)

    def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        content = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class _RuntimeServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: RuntimeState) -> None:
        super().__init__(address, _Handler)
        self.state = state


def _load_state() -> RuntimeState:
    try:
        import torch  # type: ignore[import-not-found]
        from peft import PeftModel  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeContractError("runtime_dependencies_are_missing") from exc
    base = Path(os.environ.get("IOAP_GPU_LAB_BASE_MODEL_PATH", "/models/base"))
    adapter = Path(os.environ.get("IOAP_GPU_LAB_ADAPTER_PATH", "/models/adapter"))
    identity_path = Path(
        os.environ.get("IOAP_GPU_LAB_IDENTITY_PATH", "/models/identity.json")
    )
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeContractError("runtime_identity_is_invalid") from exc
    if not isinstance(identity, dict) or identity.get("production_claim") is not False:
        raise RuntimeContractError("runtime_identity_is_invalid")
    use_adapter = os.environ.get("IOAP_GPU_LAB_USE_ADAPTER", "0") == "1"
    variant = "candidate" if use_adapter else "stable"
    torch.set_num_threads(max(1, int(os.environ.get("IOAP_GPU_LAB_CPU_THREADS", "4"))))
    tokenizer = AutoTokenizer.from_pretrained(
        str(base), trust_remote_code=False, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        dtype=torch.float32,
        trust_remote_code=False,
        local_files_only=True,
        attn_implementation="eager",
    )
    if use_adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return RuntimeState(
        variant=variant,
        identity=cast(dict[str, Any], identity),
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        guardrail=PromptInjectionGuard(),
        generation_lock=threading.Lock(),
        request_lock=threading.Lock(),
    )


def _normalized_json(value: str) -> str:
    candidate = value.strip()
    if "</think>" in candidate:
        candidate = candidate.rsplit("</think>", 1)[1].strip()
    decoder = json.JSONDecoder()
    for offset, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return json.dumps(
                parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
    return candidate


def main() -> int:
    port = int(os.environ.get("IOAP_GPU_LAB_PORT", "8080"))
    server = _RuntimeServer(("0.0.0.0", port), _load_state())
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
