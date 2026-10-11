"""Small KServe adapter for a host-side governed Reranker GPU runtime."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Literal, cast

ENDPOINT_PATH = "/v1/retrieval/rerank"
MAX_REQUEST_BYTES = 1_048_576


class ProxyConfigurationError(RuntimeError):
    """The KServe proxy configuration is missing or unsafe."""


class ProxyMetrics:
    def __init__(self, variant: Literal["stable", "candidate"]) -> None:
        self.variant = variant
        self._lock = threading.Lock()
        self._requests = 0
        self._successes = 0
        self._rejections = 0
        self._errors = 0

    def record(self, status_code: int) -> None:
        with self._lock:
            self._requests += 1
            if 200 <= status_code < 300:
                self._successes += 1
            elif 400 <= status_code < 500:
                self._rejections += 1
            else:
                self._errors += 1

    def snapshot(self) -> dict[str, str | int]:
        with self._lock:
            return {
                "schema_version": "enterprise-reranker-kserve-proxy-metrics/v1",
                "variant": self.variant,
                "request_count": self._requests,
                "success_count": self._successes,
                "rejected_count": self._rejections,
                "error_count": self._errors,
            }


class ProxyConfiguration:
    def __init__(self) -> None:
        variant = os.environ.get("IOAP_RERANKER_VARIANT", "")
        if variant not in {"stable", "candidate"}:
            raise ProxyConfigurationError("reranker_proxy_variant_is_invalid")
        upstream = os.environ.get("IOAP_RERANKER_UPSTREAM_URL", "").rstrip("/")
        if not upstream.startswith("http://") or any(item.isspace() for item in upstream):
            raise ProxyConfigurationError("reranker_proxy_upstream_is_invalid")
        timeout_text = os.environ.get("IOAP_RERANKER_UPSTREAM_TIMEOUT_SECONDS", "30")
        try:
            timeout = float(timeout_text)
        except ValueError as exc:
            raise ProxyConfigurationError("reranker_proxy_timeout_is_invalid") from exc
        if not 1.0 <= timeout <= 120.0:
            raise ProxyConfigurationError("reranker_proxy_timeout_is_invalid")
        self.variant = cast(Literal["stable", "candidate"], variant)
        self.upstream_url = upstream
        self.timeout_seconds = timeout
        self.release_id = _required("IOAP_RERANKER_RELEASE_ID")
        self.manifest_hash = _digest("IOAP_RERANKER_MANIFEST_HASH")
        self.public_component_model_id = _required("IOAP_RERANKER_PUBLIC_COMPONENT_MODEL_ID")
        self.public_artifact_hash = _digest("IOAP_RERANKER_PUBLIC_ARTIFACT_CONTENT_HASH")
        self.upstream_component_model_id = _required("IOAP_RERANKER_UPSTREAM_COMPONENT_MODEL_ID")
        self.upstream_artifact_hash = _digest("IOAP_RERANKER_UPSTREAM_ARTIFACT_CONTENT_HASH")


class RerankerProxyHandler(BaseHTTPRequestHandler):
    server_version = "IndustrialRerankerKServeProxy/1"
    protocol_version = "HTTP/1.1"
    configuration: ClassVar[ProxyConfiguration]
    metrics: ClassVar[ProxyMetrics]

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        if self.path == "/health/ready":
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ready",
                    "variant": self.configuration.variant,
                    "upstream_scope": "host_gpu_runtime",
                },
            )
            return
        if self.path == "/metrics":
            self._json(HTTPStatus.OK, self.metrics.snapshot())
            return
        self._json(HTTPStatus.NOT_FOUND, {"detail": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        if self.path != ENDPOINT_PATH:
            self._json(HTTPStatus.NOT_FOUND, {"detail": "not_found"})
            return
        try:
            payload = self._request_payload()
            upstream_payload = self._bound_upstream_payload(payload)
            status, body = self._forward(upstream_payload)
        except ValueError as exc:
            status, body = (
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {
                    "detail": str(exc),
                },
            )
        except (OSError, TimeoutError, urllib.error.URLError):
            status, body = (
                HTTPStatus.BAD_GATEWAY,
                {
                    "detail": "reranker_proxy_upstream_unavailable",
                },
            )
        self.metrics.record(int(status))
        self._json(status, body)

    def log_message(self, format: str, *args: object) -> None:
        safe_path = self.path.split("?", maxsplit=1)[0]
        print(
            json.dumps(
                {
                    "event": "reranker_proxy_request",
                    "variant": self.configuration.variant,
                    "path": safe_path,
                    "message": format % args,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _request_payload(self) -> dict[str, Any]:
        length_text = self.headers.get("Content-Length")
        try:
            length = int(length_text or "0")
        except ValueError as exc:
            raise ValueError("reranker_proxy_request_size_is_invalid") from exc
        if not 1 <= length <= MAX_REQUEST_BYTES:
            raise ValueError("reranker_proxy_request_size_is_invalid")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("reranker_proxy_request_json_is_invalid") from exc
        if not isinstance(value, dict):
            raise ValueError("reranker_proxy_request_json_is_invalid")
        return cast(dict[str, Any], value)

    def _bound_upstream_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        configuration = self.configuration
        if (
            payload.get("expected_release_id") != configuration.release_id
            or payload.get("expected_manifest_hash") != configuration.manifest_hash
            or payload.get("expected_component_model_id") != configuration.public_component_model_id
            or payload.get("expected_artifact_content_hash") != configuration.public_artifact_hash
        ):
            raise ValueError("reranker_proxy_release_binding_mismatch")
        return {
            **payload,
            "expected_component_model_id": (configuration.upstream_component_model_id),
            "expected_artifact_content_hash": configuration.upstream_artifact_hash,
        }

    def _forward(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        request = urllib.request.Request(
            self.configuration.upstream_url + ENDPOINT_PATH,
            data=raw,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.configuration.timeout_seconds,
            ) as response:
                status = response.status
                result = response.read(MAX_REQUEST_BYTES + 1)
        except urllib.error.HTTPError as error:
            status = error.code
            result = error.read(MAX_REQUEST_BYTES + 1)
        if len(result) > MAX_REQUEST_BYTES:
            raise ValueError("reranker_proxy_upstream_response_is_too_large")
        try:
            decoded = json.loads(result)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("reranker_proxy_upstream_response_is_invalid") from exc
        if not isinstance(decoded, dict):
            raise ValueError("reranker_proxy_upstream_response_is_invalid")
        return status, cast(dict[str, Any], decoded)

    def _json(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "X-IOAP-Reranker-Variant",
            self.configuration.variant,
        )
        self.end_headers()
        self.wfile.write(raw)


def serve() -> None:
    configuration = ProxyConfiguration()
    metrics = ProxyMetrics(configuration.variant)
    RerankerProxyHandler.configuration = configuration
    RerankerProxyHandler.metrics = metrics
    try:
        port = int(os.environ.get("PORT", "8080"))
    except ValueError as exc:
        raise ProxyConfigurationError("reranker_proxy_port_is_invalid") from exc
    if not 1 <= port <= 65_535:
        raise ProxyConfigurationError("reranker_proxy_port_is_invalid")
    server = ThreadingHTTPServer(("0.0.0.0", port), RerankerProxyHandler)
    server.daemon_threads = True
    server.serve_forever()


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or any(item.isspace() for item in value):
        raise ProxyConfigurationError(f"{name.lower()}_is_invalid")
    return value


def _digest(name: str) -> str:
    value = _required(name)
    if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise ProxyConfigurationError(f"{name.lower()}_is_invalid")
    return value


if __name__ == "__main__":
    serve()
