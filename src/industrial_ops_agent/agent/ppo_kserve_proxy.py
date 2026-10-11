"""Dependency-free KServe proxy for the short-lived PPO Agent GPU worker."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_REQUEST_BYTES = 65_536


class PpoAgentProxyHandler(BaseHTTPRequestHandler):
    server_version = "IOAPPpoAgentProxy/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health/ready":
            self._json_response(404, {"detail": "not_found"})
            return
        self._forward("GET", "/health/ready", None)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/agent/decision":
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
        if not isinstance(payload, dict):
            self._json_response(400, {"detail": "request_json_invalid"})
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
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=_timeout_seconds()) as response:  # noqa: S310
                response_body = response.read(MAX_REQUEST_BYTES)
                self.send_response(response.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.send_header("X-IOAP-PPO-Variant", _variant())
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(response_body)
        except HTTPError as exc:
            response_body = exc.read(16_384)
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header("X-IOAP-PPO-Variant", _variant())
            self.end_headers()
            self.wfile.write(response_body)
        except (URLError, TimeoutError, OSError):
            self._json_response(502, {"detail": "ppo_agent_gpu_worker_unavailable"})

    def _json_response(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def _variant() -> str:
    value = os.environ.get("IOAP_PPO_VARIANT", "").strip()
    if value not in {"stable", "candidate"}:
        raise RuntimeError("IOAP_PPO_VARIANT must be stable or candidate")
    return value


def _upstream() -> str:
    value = os.environ.get("IOAP_PPO_UPSTREAM_URL", "").strip().rstrip("/")
    if not value.startswith("http://") or value.count(":") < 2:
        raise RuntimeError("IOAP_PPO_UPSTREAM_URL must be an explicit HTTP endpoint")
    return value


def _timeout_seconds() -> float:
    value = float(os.environ.get("IOAP_PPO_UPSTREAM_TIMEOUT_SECONDS", "120"))
    if not 1 <= value <= 300:
        raise RuntimeError("IOAP_PPO_UPSTREAM_TIMEOUT_SECONDS is invalid")
    return value


def main() -> int:
    _variant()
    _upstream()
    port = int(os.environ.get("PORT", "8080"))
    if not 1 <= port <= 65_535:
        raise RuntimeError("PORT is invalid")
    server = ThreadingHTTPServer(("0.0.0.0", port), PpoAgentProxyHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
