"""Closed internal OpenAI-compatible router for project-staging model services."""

from __future__ import annotations

import hmac
import http.client
import json
import os
import re
import socket
import ssl
import stat
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

_DEFAULT_REQUEST_LIMIT = 16 * 1024 * 1024
_DEFAULT_RESPONSE_LIMIT = 8 * 1024 * 1024
_MAX_BODY_LIMIT = 32 * 1024 * 1024
_MAX_SECRET_BYTES = 4096
_READINESS_BODY_LIMIT = 1024 * 1024
_MAX_HTTP_HANDLERS = 16
# Shared with the PID-1 owner and prepare client; keep cleanup inside the caller's wait.
_MODEL_LOAD_SECONDS = 90.0
_MODEL_LOAD_CONTROL_SECONDS = _MODEL_LOAD_SECONDS + 12.0


class RouterConfigurationError(RuntimeError):
    """Raised when the immutable router contract is unsafe or incomplete."""


@dataclass(frozen=True, slots=True)
class UpstreamTarget:
    scheme: str
    host: str
    port: int
    api_path: str
    model_id: str


@dataclass(frozen=True, slots=True)
class DeploymentBinding:
    release_id: str
    manifest_hash: str
    target_environment: str
    diagnosis_serving_image_digest: str
    vlm_serving_image_digest: str
    router_image_digest: str
    diagnosis_package_digest: str
    vlm_package_digest: str

    def readiness_view(self) -> dict[str, object]:
        return {
            "release_id": self.release_id,
            "manifest_hash": self.manifest_hash,
            "target_environment": self.target_environment,
            "serving_image_digests": {
                "diagnosis": self.diagnosis_serving_image_digest,
                "vlm": self.vlm_serving_image_digest,
            },
            "router_image_digest": self.router_image_digest,
            "package_digests": {
                "diagnosis": self.diagnosis_package_digest,
                "vlm": self.vlm_package_digest,
            },
        }


@dataclass(frozen=True, slots=True)
class RouterConfig:
    diagnosis: UpstreamTarget
    vlm: UpstreamTarget
    credential: str = field(repr=False)
    request_max_bytes: int = _DEFAULT_REQUEST_LIMIT
    response_max_bytes: int = _DEFAULT_RESPONSE_LIMIT
    timeout_seconds: float = 30.0
    deployment_binding: DeploymentBinding | None = None
    max_pending_requests: int = 4
    queue_timeout_seconds: float = 5.0

    def target_for(self, model_id: str) -> UpstreamTarget | None:
        if hmac.compare_digest(model_id, self.diagnosis.model_id):
            return self.diagnosis
        if hmac.compare_digest(model_id, self.vlm.model_id):
            return self.vlm
        return None


class _AdmissionError(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status = status
        self.code = code
        super().__init__(code)


class _Admission:
    """One forwarding exchange; waiting requests hold no decoded body."""

    def __init__(self, config: RouterConfig) -> None:
        self._condition = threading.Condition()
        self._waiting: deque[object] = deque()
        self._active = False
        self._limit = config.max_pending_requests
        self._timeout = config.queue_timeout_seconds

    def acquire(self) -> None:
        deadline = time.monotonic() + self._timeout
        with self._condition:
            if not self._active and not self._waiting:
                self._active = True
                return
            if len(self._waiting) >= self._limit:
                raise _AdmissionError(429, "router_queue_full")
            ticket = object()
            self._waiting.append(ticket)
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _AdmissionError(504, "router_queue_timeout")
                    if not self._active and self._waiting[0] is ticket:
                        self._active = True
                        return
                    self._condition.wait(timeout=remaining)
            finally:
                self._waiting.remove(ticket)
                self._condition.notify_all()

    def release(self) -> None:
        with self._condition:
            self._active = False
            self._condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "active_requests": int(self._active),
                "waiting_requests": len(self._waiting),
                "max_active_requests": 1,
                "max_pending_requests": self._limit,
                "queue_timeout_seconds": self._timeout,
                "max_http_handlers": _MAX_HTTP_HANDLERS,
            }


class ModelRouterServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], config: RouterConfig) -> None:
        self.router_config = config
        self.admission = _Admission(config)
        self._handler_slots = threading.BoundedSemaphore(_MAX_HTTP_HANDLERS)
        self.lifecycle_error: str | None = None
        super().__init__(address, ModelRouterRequestHandler)

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        if not self._handler_slots.acquire(blocking=False):
            # Do not allocate another handler just to report overload. This
            # socket is owned by this accept call, never an existing client.
            body = json.dumps({"error": {
                "message": "Model router request failed",
                "type": "model_router_error",
                "code": "router_connection_limit",
            }}, separators=(",", ":")).encode("utf-8")
            headers = (
                "HTTP/1.1 503 Service Unavailable\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Retry-After: 1\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
            ).encode("ascii")
            try:
                request.settimeout(0.1)
                request.sendall(headers + body)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._handler_slots.release()
            raise

    def process_request_thread(
        self, request: socket.socket, client_address: tuple[str, int]
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handler_slots.release()


class ModelRouterRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "industrial-ops-model-router"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.config.timeout_seconds)

    @property
    def router(self) -> ModelRouterServer:
        server = self.server
        if not isinstance(server, ModelRouterServer):
            raise RuntimeError("router_server_invalid")
        return server

    @property
    def config(self) -> RouterConfig:
        return self.router.router_config

    def log_message(self, _format: str, *args: object) -> None:
        del args

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health/ready":
            self._send_error(404, "route_not_found")
            return
        try:
            if self.router.lifecycle_error:
                raise _AdmissionError(503, self.router.lifecycle_error)
            lifecycle = {
                name: _runtime_control(self.config, target, "status", timeout=0.5)
                for name, target in (("diagnosis", self.config.diagnosis), ("vlm", self.config.vlm))
            }
            if not all(row["prepared"] and row["state"] in {
                "UNLOADED", "LOADING", "READY", "UNLOADING",
            } for row in lifecycle.values()):
                raise _AdmissionError(503, "runtime_not_prepared")
        except _AdmissionError as exc:
            self._send_json(503, {"status": "NOT_READY", "reason": exc.code,
                                  "admission": self.router.admission.snapshot()})
            return
        if lifecycle:
            payload: dict[str, object] = {
                "status": "READY" if all(row["state"] == "READY" for row in lifecycle.values()) else "ACCEPTING",
                "admission": self.router.admission.snapshot(),
                "lifecycle": lifecycle,
                "models": {
                    "diagnosis": self.config.diagnosis.model_id,
                    "vlm": self.config.vlm.model_id,
                },
            }
            if self.config.deployment_binding is not None:
                payload["deployment_binding"] = self.config.deployment_binding.readiness_view()
            self._send_json(
                200,
                payload,
            )
            return

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send_error(404, "route_not_found")
            return
        if not self._authenticated():
            self._send_error(401, "router_authentication_required")
            return
        try:
            self.router.admission.acquire()
        except _AdmissionError as exc:
            self._send_error(exc.status, exc.code)
            return
        try:
            self._complete_request()
        finally:
            self.router.admission.release()

    def _complete_request(self) -> None:
        body = self._read_bounded_body()
        if body is None:
            return
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_error(400, "request_json_invalid")
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
            self._send_error(400, "request_model_invalid")
            return
        target = self.config.target_for(payload["model"])
        if target is None:
            self._send_error(404, "model_not_allowed")
            return
        self._forward(target, body)

    def do_PUT(self) -> None:  # noqa: N802
        self._send_error(404, "route_not_found")

    def do_PATCH(self) -> None:  # noqa: N802
        self._send_error(404, "route_not_found")

    def do_DELETE(self) -> None:  # noqa: N802
        self._send_error(404, "route_not_found")

    def _authenticated(self) -> bool:
        values = self.headers.get_all("Authorization", failobj=[])
        if len(values) != 1:
            return False
        return hmac.compare_digest(values[0], f"Bearer {self.config.credential}")

    def _read_bounded_body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding") is not None:
            self.close_connection = True
            self._send_error(400, "request_framing_invalid")
            return None
        values = self.headers.get_all("Content-Length", failobj=[])
        if len(values) != 1 or not values[0].isdigit():
            self.close_connection = True
            self._send_error(411, "content_length_required")
            return None
        size = int(values[0])
        if size <= 0:
            self.close_connection = True
            self._send_error(400, "request_body_required")
            return None
        if size > self.config.request_max_bytes:
            self.close_connection = True
            self._send_error(413, "request_too_large")
            return None
        try:
            body = self.rfile.read(size)
        except TimeoutError:
            self.close_connection = True
            self._send_error(408, "request_body_timeout")
            return None
        if len(body) != size:
            self.close_connection = True
            self._send_error(400, "request_body_incomplete")
            return None
        return body

    def _forward(self, target: UpstreamTarget, body: bytes) -> None:
        if self.router.lifecycle_error:
            self._send_error(503, self.router.lifecycle_error)
            return
        result: tuple[int, bytes, str] | None = None
        failure: _AdmissionError | None = None
        try:
            # Never trust a cached 'unloaded' observation across requests/restarts.
            for upstream in (self.config.diagnosis, self.config.vlm):
                _runtime_control(self.config, upstream, "unload", timeout=12)
            _runtime_control(self.config, target, "load", timeout=_MODEL_LOAD_CONTROL_SECONDS)
            result = self._exchange(target, body, credential=self.config.credential)
        except _AdmissionError as exc:
            failure = exc
            if exc.code.startswith("runtime_"):
                self.router.lifecycle_error = exc.code
        finally:
            # Buffer the result until the owning supervisor confirms exit.
            # A timeout/409 here is uncertainty, never permission to load another model.
            try:
                _runtime_control(self.config, target, "unload", timeout=12)
            except _AdmissionError as cleanup_error:
                primary = failure.code if failure is not None else (
                    "upstream_http_error" if result is not None and result[0] >= 400 else "none")
                print(f"runtime_cleanup_failed primary={primary} cleanup={cleanup_error.code}",
                      file=sys.stderr, flush=True)
                self.router.lifecycle_error = "runtime_cleanup_unconfirmed"
                failure = _AdmissionError(503, "runtime_cleanup_unconfirmed")
        if failure is not None:
            self._send_error(failure.status, failure.code)
        elif result is not None:
            status, response_body, content_type = result
            self._send_bytes(status, response_body, content_type=content_type)

    def _exchange(
        self, target: UpstreamTarget, body: bytes, *, credential: str | None = None,
    ) -> tuple[int, bytes, str]:
        connection = _connection(target, self.config.timeout_seconds)
        try:
            headers = {
                "Accept": "application/json", "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            }
            # Only router -> supervisor is authenticated. The loopback hop
            # deliberately calls this method without the internal credential.
            if credential is not None:
                headers["Authorization"] = f"Bearer {credential}"
            connection.request(
                "POST",
                f"{target.api_path}/chat/completions",
                body=body,
                headers=headers,
            )
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise _AdmissionError(502, "upstream_redirect_rejected")
            declared_length = response.getheader("Content-Length")
            if declared_length is not None:
                try:
                    declared_size = int(declared_length)
                except ValueError:
                    raise _AdmissionError(502, "upstream_response_invalid") from None
                if declared_size < 0 or declared_size > self.config.response_max_bytes:
                    raise _AdmissionError(502, "upstream_response_too_large")
            response_body = response.read(self.config.response_max_bytes + 1)
            if len(response_body) > self.config.response_max_bytes:
                raise _AdmissionError(502, "upstream_response_too_large")
            content_type = response.getheader("Content-Type", "application/json")
            return response.status, response_body, content_type
        except (OSError, http.client.HTTPException, TimeoutError, ssl.SSLError):
            raise _AdmissionError(502, "upstream_unavailable") from None
        finally:
            connection.close()

    def _send_error(self, status_code: int, code: str) -> None:
        self._send_json(
            status_code,
            {
                "error": {
                    "message": "Model router request failed",
                    "type": "model_router_error",
                    "code": code,
                }
            },
        )

    def _send_json(self, status_code: int, payload: object) -> None:
        self._send_bytes(
            status_code,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            content_type="application/json",
        )

    def _send_bytes(self, status_code: int, payload: bytes, *, content_type: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if status_code in {429, 503, 504}:
            self.send_header("Retry-After", "1")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True


def _runtime_control(
    config: RouterConfig, target: UpstreamTarget, action: str, *, timeout: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            return _runtime_control_once(config, target, action, timeout=remaining)
        except _AdmissionError as exc:
            if action != "unload" or exc.code != "runtime_busy":
                raise
        # The same owner may still be unwinding its request. Never reset the
        # deadline, retry inference, or interpret busy as permission to load.
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    raise _AdmissionError(503, "runtime_control_unavailable")


def _runtime_control_once(
    config: RouterConfig, target: UpstreamTarget, action: str, *, timeout: float,
) -> dict[str, object]:
    binding = config.deployment_binding
    if binding is None or action not in {"status", "load", "unload"}:
        raise _AdmissionError(503, "runtime_binding_missing")
    digest = (binding.diagnosis_package_digest if target == config.diagnosis
              else binding.vlm_package_digest)
    connection = _connection(target, timeout)
    try:
        connection.request(
            "GET" if action == "status" else "POST", f"/runtime/{action}",
            body=None if action == "status" else b"",
            headers={"Authorization": f"Bearer {config.credential}", "Accept": "application/json"},
        )
        response = connection.getresponse()
        raw = response.read(65537)
        if action == "unload" and response.status == 409 and len(raw) <= 65536:
            busy = json.loads(raw)
            if (isinstance(busy, dict) and isinstance(busy.get("error"), dict)
                    and busy["error"].get("code") == "runtime_busy"):
                raise _AdmissionError(503, "runtime_busy")
        if response.status != 200 or len(raw) > 65536:
            raise ValueError("invalid control response")
        row = json.loads(raw)
        if (not isinstance(row, dict)
                or row.get("schema_version") != "ioap-owned-model-runtime/v1"
                or row.get("model_id") != target.model_id
                or row.get("package_digest") != digest
                or type(row.get("prepared")) is not bool
                or row.get("state") not in {"UNLOADED", "LOADING", "READY", "UNLOADING", "FAILED"}):
            raise ValueError("invalid control identity")
        expected = {"load": "READY", "unload": "UNLOADED"}.get(action)
        if expected is not None and (row["state"] != expected or row["prepared"] is not True):
            raise ValueError("unconfirmed control state")
        return {key: row[key] for key in ("schema_version", "model_id", "package_digest", "state", "prepared")}
    except (OSError, http.client.HTTPException, ValueError, TypeError):
        raise _AdmissionError(503, "runtime_control_unavailable") from None
    finally:
        connection.close()


def _connection(target: UpstreamTarget, timeout_seconds: float) -> http.client.HTTPConnection:
    if target.scheme == "https":
        return http.client.HTTPSConnection(
            target.host,
            target.port,
            timeout=timeout_seconds,
            context=ssl.create_default_context(),
        )
    return http.client.HTTPConnection(target.host, target.port, timeout=timeout_seconds)


def _read_upstream(
    target: UpstreamTarget,
    method: str,
    path: str,
    timeout_seconds: float,
    *,
    limit: int,
) -> tuple[int, bytes]:
    connection = _connection(target, timeout_seconds)
    try:
        connection.request(method, path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        if 300 <= response.status < 400:
            return 0, b""
        body = response.read(limit + 1)
        if len(body) > limit:
            return 0, b""
        return response.status, body
    except (OSError, http.client.HTTPException, TimeoutError, ssl.SSLError):
        return 0, b""
    finally:
        connection.close()


def _probe_target(target: UpstreamTarget, timeout_seconds: float) -> bool:
    health_status, _ = _read_upstream(
        target,
        "GET",
        "/health",
        timeout_seconds,
        limit=_READINESS_BODY_LIMIT,
    )
    if health_status != 200:
        return False
    model_status, model_body = _read_upstream(
        target,
        "GET",
        f"{target.api_path}/models",
        timeout_seconds,
        limit=_READINESS_BODY_LIMIT,
    )
    if model_status != 200:
        return False
    try:
        payload = json.loads(model_body)
        entries = payload["data"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return False
    if not isinstance(entries, list):
        return False
    return any(isinstance(item, dict) and item.get("id") == target.model_id for item in entries)


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value or len(value) > 4096 or any(ord(character) < 32 for character in value):
        raise RouterConfigurationError(f"{name.lower()}_invalid")
    return value


def _bounded_int(
    environment: Mapping[str, str], name: str, default: int, minimum: int, maximum: int
) -> int:
    raw = environment.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RouterConfigurationError(f"{name.lower()}_invalid") from exc
    if value < minimum or value > maximum:
        raise RouterConfigurationError(f"{name.lower()}_invalid")
    return value


def _bounded_float(
    environment: Mapping[str, str], name: str, default: float, minimum: float, maximum: float
) -> float:
    raw = environment.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RouterConfigurationError(f"{name.lower()}_invalid") from exc
    if value != value or value < minimum or value > maximum:
        raise RouterConfigurationError(f"{name.lower()}_invalid")
    return value


def _target(url: str, model_id: str, name: str) -> UpstreamTarget:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise RouterConfigurationError(f"{name}_upstream_invalid")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise RouterConfigurationError(f"{name}_upstream_invalid") from exc
    return UpstreamTarget(
        scheme=parsed.scheme,
        host=parsed.hostname,
        port=port,
        api_path="/v1",
        model_id=model_id,
    )


def _read_credential(path_value: str) -> str:
    try:
        metadata = os.lstat(path_value)
    except OSError as exc:
        raise RouterConfigurationError("secret_file_invalid") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o077
        or not metadata.st_mode & stat.S_IRUSR
        or metadata.st_size <= 0
        or metadata.st_size > _MAX_SECRET_BYTES
    ):
        raise RouterConfigurationError("secret_file_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path_value, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise RouterConfigurationError("secret_file_invalid")
            raw = os.read(descriptor, _MAX_SECRET_BYTES + 1)
        finally:
            os.close(descriptor)
    except RouterConfigurationError:
        raise
    except OSError as exc:
        raise RouterConfigurationError("secret_file_invalid") from exc
    try:
        credential = raw.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise RouterConfigurationError("secret_file_invalid") from exc
    if not credential or len(raw) > _MAX_SECRET_BYTES or "\n" in credential or "\r" in credential:
        raise RouterConfigurationError("secret_file_invalid")
    return credential


def load_router_config(environment: Mapping[str, str] | None = None) -> RouterConfig:
    values = os.environ if environment is None else environment
    diagnosis_model_id = _required(values, "IOAP_DIAGNOSIS_MODEL_ID")
    vlm_model_id = _required(values, "IOAP_VLM_MODEL_ID")
    if hmac.compare_digest(diagnosis_model_id, vlm_model_id):
        raise RouterConfigurationError("model_ids_must_be_distinct")
    credential_path = _required(values, "IOAP_MODEL_GATEWAY_API_KEY_FILE")
    return RouterConfig(
        diagnosis=_target(
            _required(values, "IOAP_DIAGNOSIS_UPSTREAM_URL"),
            diagnosis_model_id,
            "diagnosis",
        ),
        vlm=_target(
            _required(values, "IOAP_VLM_UPSTREAM_URL"),
            vlm_model_id,
            "vlm",
        ),
        credential=_read_credential(credential_path),
        request_max_bytes=_bounded_int(
            values,
            "IOAP_MODEL_ROUTER_REQUEST_MAX_BYTES",
            _DEFAULT_REQUEST_LIMIT,
            128,
            _MAX_BODY_LIMIT,
        ),
        response_max_bytes=_bounded_int(
            values,
            "IOAP_MODEL_ROUTER_RESPONSE_MAX_BYTES",
            _DEFAULT_RESPONSE_LIMIT,
            128,
            _MAX_BODY_LIMIT,
        ),
        timeout_seconds=_bounded_float(
            values,
            "IOAP_MODEL_ROUTER_TIMEOUT_SECONDS",
            30.0,
            0.1,
            120.0,
        ),
        deployment_binding=_deployment_binding(values),
        max_pending_requests=_bounded_int(
            values, "IOAP_MODEL_ROUTER_MAX_PENDING_REQUESTS", 4, 0, 8,
        ),
        queue_timeout_seconds=_bounded_float(
            values, "IOAP_MODEL_ROUTER_QUEUE_TIMEOUT_SECONDS", 5.0, 0.1, 30.0,
        ),
    )


def _deployment_binding(environment: Mapping[str, str]) -> DeploymentBinding | None:
    names = (
        "IOAP_MODEL_RELEASE_ID",
        "IOAP_MODEL_RELEASE_MANIFEST_HASH",
        "IOAP_MODEL_TARGET_ENVIRONMENT",
        "IOAP_DIAGNOSIS_SERVING_IMAGE_DIGEST",
        "IOAP_VLM_SERVING_IMAGE_DIGEST",
        "IOAP_MODEL_ROUTER_IMAGE_DIGEST",
        "IOAP_DIAGNOSIS_PACKAGE_DIGEST",
        "IOAP_VLM_PACKAGE_DIGEST",
    )
    values = {name: environment.get(name, "").strip() for name in names}
    if not any(values.values()):
        return None
    if not all(values.values()):
        raise RouterConfigurationError("deployment_binding_incomplete")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", values["IOAP_MODEL_RELEASE_ID"]):
        raise RouterConfigurationError("deployment_release_id_invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", values["IOAP_MODEL_RELEASE_MANIFEST_HASH"]):
        raise RouterConfigurationError("deployment_manifest_hash_invalid")
    if values["IOAP_MODEL_TARGET_ENVIRONMENT"] != "STAGING":
        raise RouterConfigurationError("deployment_target_environment_invalid")
    for name in (
        "IOAP_DIAGNOSIS_SERVING_IMAGE_DIGEST",
        "IOAP_VLM_SERVING_IMAGE_DIGEST",
        "IOAP_MODEL_ROUTER_IMAGE_DIGEST",
    ):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", values[name]):
            raise RouterConfigurationError("deployment_image_digest_invalid")
    for name in ("IOAP_DIAGNOSIS_PACKAGE_DIGEST", "IOAP_VLM_PACKAGE_DIGEST"):
        if not re.fullmatch(r"[0-9a-f]{64}", values[name]):
            raise RouterConfigurationError("deployment_package_digest_invalid")
    return DeploymentBinding(
        release_id=values["IOAP_MODEL_RELEASE_ID"],
        manifest_hash=values["IOAP_MODEL_RELEASE_MANIFEST_HASH"],
        target_environment=values["IOAP_MODEL_TARGET_ENVIRONMENT"],
        diagnosis_serving_image_digest=values["IOAP_DIAGNOSIS_SERVING_IMAGE_DIGEST"],
        vlm_serving_image_digest=values["IOAP_VLM_SERVING_IMAGE_DIGEST"],
        router_image_digest=values["IOAP_MODEL_ROUTER_IMAGE_DIGEST"],
        diagnosis_package_digest=values["IOAP_DIAGNOSIS_PACKAGE_DIGEST"],
        vlm_package_digest=values["IOAP_VLM_PACKAGE_DIGEST"],
    )


def build_router_server(
    config: RouterConfig, *, host: str = "0.0.0.0", port: int = 8080
) -> ModelRouterServer:
    if not host or not 0 <= port <= 65535:
        raise RouterConfigurationError("listen_address_invalid")
    return ModelRouterServer((host, port), config)


def run() -> None:
    try:
        config = load_router_config()
        host = os.environ.get("IOAP_MODEL_ROUTER_HOST", "0.0.0.0")
        port = _bounded_int(os.environ, "IOAP_MODEL_ROUTER_PORT", 8080, 1, 65535)
        server = build_router_server(config, host=host, port=port)
    except RouterConfigurationError as exc:
        print(f"MODEL_ROUTER_STARTUP_FAILED:{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
