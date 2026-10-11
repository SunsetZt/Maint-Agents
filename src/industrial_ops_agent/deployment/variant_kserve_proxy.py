"""Dependency-free KServe proxy for a governed stable/candidate GPU worker."""

from __future__ import annotations

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_REQUEST_BYTES = 128 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_HEADER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,63}$")


class VariantProxyHandler(BaseHTTPRequestHandler):
    """Forward one configured inference path and lock the rollout variant."""

    server_version = "IOAPVariantProxy/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/health/ready", "/metrics"}:
            self._json_response(404, {"detail": "not_found"})
            return
        self._forward("GET", self.path, None)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != _endpoint_path():
            self._json_response(404, {"detail": "not_found"})
            return
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError:
            self._json_response(400, {"detail": "content_length_invalid"})
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._json_response(413, {"detail": "request_size_invalid"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json_response(400, {"detail": "request_json_invalid"})
            return
        if not isinstance(payload, dict) or "variant" in payload:
            self._json_response(400, {"detail": "request_variant_is_proxy_controlled"})
            return
        payload["variant"] = _variant()
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self._forward("POST", self.path, encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _forward(self, method: str, path: str, body: bytes | None) -> None:
        request = Request(
            _upstream() + path,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "*/*"},
            method=method,
        )
        try:
            with urlopen(request, timeout=_timeout_seconds()) as response:  # noqa: S310
                response_body = response.read(_max_response_bytes() + 1)
                if len(response_body) > _max_response_bytes():
                    self._json_response(502, {"detail": "gpu_worker_response_too_large"})
                    return
                self.send_response(response.status)
                self._response_headers(response.headers, len(response_body))
                self.end_headers()
                self.wfile.write(response_body)
        except HTTPError as exc:
            response_body = exc.read(min(_max_response_bytes(), 16_384))
            self.send_response(exc.code)
            self._response_headers(exc.headers, len(response_body))
            self.end_headers()
            self.wfile.write(response_body)
        except (URLError, TimeoutError, OSError):
            self._json_response(502, {"detail": "gpu_worker_unavailable"})

    def _response_headers(self, headers: Any, content_length: int) -> None:
        self.send_header(
            "Content-Type",
            headers.get("Content-Type", "application/octet-stream"),
        )
        self.send_header("Content-Length", str(content_length))
        self.send_header(_variant_header(), _variant())
        self.send_header("X-Content-Type-Options", "nosniff")
        cache_control = headers.get("Cache-Control")
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        for name, value in headers.items():
            if name.casefold().startswith("x-ioap-") and name.casefold() != (
                _variant_header().casefold()
            ):
                self.send_header(name, value)

    def _json_response(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(_variant_header(), _variant())
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def _variant() -> str:
    value = os.environ.get("IOAP_VARIANT", "").strip()
    if value not in {"stable", "candidate"}:
        raise RuntimeError("IOAP_VARIANT must be stable or candidate")
    return value


def _upstream() -> str:
    value = os.environ.get("IOAP_UPSTREAM_URL", "").strip().rstrip("/")
    if not value.startswith("http://") or value.count(":") < 2:
        raise RuntimeError("IOAP_UPSTREAM_URL must be an explicit HTTP endpoint")
    return value


def _endpoint_path() -> str:
    value = os.environ.get("IOAP_ENDPOINT_PATH", "").strip()
    if not value.startswith("/") or value in {"/", "/health/ready", "/metrics"}:
        raise RuntimeError("IOAP_ENDPOINT_PATH is invalid")
    return value


def _variant_header() -> str:
    value = os.environ.get("IOAP_VARIANT_HEADER", "X-IOAP-Model-Variant").strip()
    if not _HEADER_NAME.fullmatch(value):
        raise RuntimeError("IOAP_VARIANT_HEADER is invalid")
    return value


def _timeout_seconds() -> float:
    value = float(os.environ.get("IOAP_UPSTREAM_TIMEOUT_SECONDS", "120"))
    if not 1 <= value <= 300:
        raise RuntimeError("IOAP_UPSTREAM_TIMEOUT_SECONDS is invalid")
    return value


def _max_response_bytes() -> int:
    value = int(os.environ.get("IOAP_MAX_RESPONSE_BYTES", str(DEFAULT_MAX_RESPONSE_BYTES)))
    if not 1_024 <= value <= 64 * 1024 * 1024:
        raise RuntimeError("IOAP_MAX_RESPONSE_BYTES is invalid")
    return value


def main() -> int:
    _variant()
    _upstream()
    _endpoint_path()
    _variant_header()
    _timeout_seconds()
    _max_response_bytes()
    port = int(os.environ.get("PORT", "8080"))
    if not 1 <= port <= 65_535:
        raise RuntimeError("PORT is invalid")
    ThreadingHTTPServer(("0.0.0.0", port), VariantProxyHandler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
