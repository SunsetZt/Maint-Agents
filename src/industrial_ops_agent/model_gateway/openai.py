"""Narrow OpenAI-compatible HTTP adapter for vLLM and SGLang endpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import monotonic
from typing import Any
from urllib.parse import urlparse

import httpx

from industrial_ops_agent.model_gateway.service import (
    GatewayRequest,
    ModelTransportError,
    ResolvedModel,
    TransportResult,
)


@dataclass(slots=True)
class OpenAiCompatibleTransport:
    api_key: str
    allow_plain_http: bool = False
    trust_env: bool = True

    async def complete(
        self,
        model: ResolvedModel,
        request: GatewayRequest,
        *,
        timeout_seconds: float,
    ) -> TransportResult:
        endpoint = _completion_endpoint(model.endpoint_url, self.allow_plain_http)
        payload: dict[str, Any] = {
            "model": _served_model_id(model, request.request_class),
            "messages": list(request.messages),
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_schema_name,
                    "schema": request.response_schema,
                    "strict": True,
                },
            },
        }
        started = monotonic()
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds),
                follow_redirects=False,
                trust_env=self.trust_env,
            ) as client:
                response = await client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ModelTransportError("model_endpoint_unavailable") from exc
        elapsed_ms = (monotonic() - started) * 1000.0
        if response.status_code != 200:
            category = (
                "model_endpoint_rejected" if response.status_code < 500 else "model_endpoint_failed"
            )
            raise ModelTransportError(category)
        try:
            body = response.json()
            choice = body["choices"][0]
            raw_content = choice["message"]["content"]
            content = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
            usage = body.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            finish_reason = str(choice.get("finish_reason", "unknown"))
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelTransportError("model_response_invalid") from exc
        if not isinstance(content, dict):
            raise ModelTransportError("model_response_invalid")
        return TransportResult(
            content=content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_breakdown={"upstream_total_ms": elapsed_ms},
        )


def _completion_endpoint(base_url: str, allow_plain_http: bool) -> str:
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
        raise ModelTransportError("model_endpoint_invalid")
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def _served_model_id(model: ResolvedModel, request_class: str) -> str:
    if request_class == "VLM":
        component_id = model.multimodal_model_ids.get("vlm")
        if not component_id:
            raise ModelTransportError("vlm_model_not_bound")
        return component_id
    return model.release_id
