"""PID-1 owner for one immutable offline vLLM model inside its dedicated container."""

from __future__ import annotations

import json
import http.client
import base64
import io
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import tarfile
from hashlib import sha256
from pathlib import Path

if __package__:
    from .router_runtime import (
        ModelRouterRequestHandler, ModelRouterServer, RouterConfig, RouterConfigurationError,
        UpstreamTarget, _AdmissionError, _probe_target, _read_credential, _read_upstream,
        _MODEL_LOAD_SECONDS as _LOAD_SECONDS, _MODEL_LOAD_CONTROL_SECONDS,
    )
else:
    from router_runtime import (
        ModelRouterRequestHandler, ModelRouterServer, RouterConfig, RouterConfigurationError,
        UpstreamTarget, _AdmissionError, _probe_target, _read_credential, _read_upstream,
        _MODEL_LOAD_SECONDS as _LOAD_SECONDS, _MODEL_LOAD_CONTROL_SECONDS,
    )

_SCHEMA = "ioap-owned-model-runtime/v1"
_SMOL_REQUEST_SECONDS = 60.0
_DIAGNOSIS_REQUEST_SECONDS = 60.0
_VALUE_FLAGS = {
    "--model", "--served-model-name", "--lora-modules", "--max-lora-rank",
    "--gpu-memory-utilization", "--max-model-len", "--max-num-seqs",
    "--max-num-batched-tokens", "--limit-mm-per-prompt", "--mm-processor-kwargs", "--dtype",
}
_BOOL_FLAGS = {"--enable-lora", "--enforce-eager"}
# Native patch counting assumes this resize, including the startup dummy image.
# Request thumbnailing below still preserves aspect ratio and never upscales.
_SMOL_PROCESSOR = {"do_resize": True, "size": {"longest_edge": 1024},
                   "max_image_size": {"longest_edge": 512}, "do_image_splitting": True}
_BASE_VLM = {
    "schema_version": 2, "artifact_kind": "BASE_CHECKPOINT", "component": "vlm",
    "classification": "PROJECT_STAGING_REAL",
    "base_model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "base_revision": "a7da5b986cb59b408707209984f360a5f4ad7e47",
    "source_manifest_sha256": "276d9a77513b54202bcb3db3753203ca7a1e9016bdf4ea6b243da324671ac9de",
}


def _model_arguments(argv: list[str], model_id: str, package: dict | None = None) -> list[str]:
    values: dict[str, str] = {}
    seen: set[str] = set()
    offset = 0
    while offset < len(argv):
        flag = argv[offset]
        if flag in seen or flag not in _VALUE_FLAGS | _BOOL_FLAGS:
            raise RouterConfigurationError("runtime_arguments_invalid")
        seen.add(flag)
        offset += 1
        if flag in _VALUE_FLAGS:
            if offset == len(argv) or argv[offset].startswith("--"):
                raise RouterConfigurationError("runtime_arguments_invalid")
            values[flag] = argv[offset]
            offset += 1
    if values.get("--served-model-name") == "Qwen/Qwen2-VL-2B-Instruct":
        raise RouterConfigurationError("OLD_VLM_RETIRED")
    base_only = package is not None and package.get("schema_version") == 2
    if base_only:
        if (any(package.get(key) != value for key, value in _BASE_VLM.items())
                or package.get("packaged_model_id") != model_id
                or values.get("--model") != "/models/runtime/base"
                or values.get("--served-model-name") != model_id
                or seen & {"--enable-lora", "--lora-modules", "--max-lora-rank"}):
            raise RouterConfigurationError("runtime_model_binding_invalid")
        try:
            processor = json.loads(values.get("--mm-processor-kwargs", "null"))
            media = json.loads(values.get("--limit-mm-per-prompt", "null"))
            bounded = (
                json.dumps(processor, sort_keys=True) == json.dumps(_SMOL_PROCESSOR, sort_keys=True)
                and json.dumps(media, sort_keys=True) == '{"image": 1, "video": 0}'
                and values.get("--max-num-seqs") == "1"
                and values.get("--max-model-len") == "4096"
                and values.get("--max-num-batched-tokens") == "4096"
                and values.get("--dtype") == "bfloat16"
                and "--enforce-eager" in seen
                and 0 < float(values.get("--gpu-memory-utilization", "0")) <= 0.5
            )
        except (ValueError, TypeError):
            bounded = False
        if not bounded:
            raise RouterConfigurationError("runtime_smol_limits_invalid")
    elif (
        values.get("--model") != "/models/runtime/base"
        or values.get("--lora-modules") != f"{model_id}=/models/runtime/adapter"
        or not values.get("--served-model-name")
        or "--enable-lora" not in seen
    ):
        raise RouterConfigurationError("runtime_model_binding_invalid")
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--host", "127.0.0.1", "--port", "8001", *argv]


def _smol_request(body: bytes) -> bytes:
    # Model image already supplies Pillow. Never add it to the dependency-free router.
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise _AdmissionError(503, "runtime_image_processor_unavailable") from None
    try:
        payload = json.loads(body)
        tokens = payload.get("max_tokens", 512)
        if (type(tokens) is not int or not 0 < tokens <= 512
                or any(type(payload.get(key, 1)) is not int or payload.get(key, 1) != 1
                       for key in ("n", "best_of"))
                or payload.get("stream", False) is not False
                or {"max_completion_tokens", "mm_processor_kwargs", "multi_modal_data", "extra_body"} & payload.keys()):
            raise ValueError("request limits")
        messages = payload["messages"]
        if not isinstance(messages, list):
            raise ValueError("messages")
        images = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                continue
            if not isinstance(content, list):
                raise ValueError("content")
            for item in content:
                if item["type"] == "image_url":
                    images.append(item["image_url"])
                elif item["type"] != "text" or not isinstance(item.get("text"), str):
                    raise ValueError("media type")
        if len(images) != 1:
            raise ValueError("image count")
        url = images[0]["url"]
        if not isinstance(url, str) or len(url) > 14 * 1024 * 1024:
            raise _AdmissionError(413, "runtime_image_too_large")
        prefix, encoded = url.split(",", 1)
        formats = {"data:image/png;base64": "PNG", "data:image/jpeg;base64": "JPEG"}
        if prefix not in formats:
            raise ValueError("image source")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 10 * 1024 * 1024:
            raise _AdmissionError(413, "runtime_image_too_large")
        with Image.open(io.BytesIO(raw)) as source:
            if (source.format != formats[prefix] or getattr(source, "n_frames", 1) != 1
                    or min(source.size) <= 0):
                raise ValueError("image format")
            if source.width * source.height > 8_294_400:
                raise _AdmissionError(413, "runtime_image_too_large")
            # This derived inference copy does not replace the original evidence.
            # Native Idefics3 patch sizing is separate from input-image resampling.
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="PNG")
        images[0]["url"] = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
        payload["max_tokens"] = tokens
        return json.dumps(payload, separators=(",", ":")).encode()
    except (KeyError, TypeError, ValueError, OSError, Image.DecompressionBombError):
        raise _AdmissionError(400, "runtime_image_request_invalid") from None


def _mounted_package(model_id: str, digest: str) -> dict:
    path = Path("/models/runtime/ioap-package-manifest.json")
    if path.is_symlink() or path.stat().st_size > 65536:
        raise RuntimeError("runtime_package_binding_invalid")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise RuntimeError("runtime_package_binding_invalid")
    package = json.loads(raw)
    if (not isinstance(package, dict) or package.get("package_digest") != digest
            or package.get("packaged_model_id") != model_id):
        raise RuntimeError("runtime_package_binding_invalid")
    if package.get("base_model_id") == "Qwen/Qwen2-VL-2B-Instruct":
        raise RuntimeError("OLD_VLM_RETIRED")
    return package


def _verify_adapter_identity(package: dict, root: Path = Path("/models/runtime")) -> str | None:
    """Bind a new candidate to the registered tar bytes AND the actual mounted files.

    Existing packages stay compatible, but cannot satisfy candidate evaluation's
    required Adapter hash. No archive is extracted and no model is loaded here.
    """
    expected = package.get("source_adapter_artifact_hash")
    if expected is None:
        return None
    try:
        if not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
            raise ValueError("hash")
        bundle = root / "ioap-adapter-bundle.tar.gz"
        adapter = root / "adapter"
        if (root.is_symlink() or bundle.is_symlink() or adapter.is_symlink()
                or not bundle.is_file() or not adapter.is_dir()
                or bundle.stat().st_size > 512 * 1024 * 1024):
            raise ValueError("path")
        digest = sha256()
        with bundle.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("content")
        names, total = set(), 0
        with tarfile.open(bundle, "r|gz") as archive:
            for member in archive:
                relative = Path(member.name)
                total += member.size
                if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                        or "\\" in member.name or str(relative) != member.name
                        or member.name in names or len(names) >= 1024
                        or total > 512 * 1024 * 1024):
                    raise ValueError("member")
                names.add(member.name)
                path = adapter / relative
                if (any(p.is_symlink() for p in (path, *path.parents))
                        or not path.is_file() or path.stat().st_size != member.size):
                    raise ValueError("mounted member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("member stream")
                with stream, path.open("rb") as mounted:
                    while block := stream.read(1024 * 1024):
                        if block != mounted.read(len(block)):
                            raise ValueError("mounted content")
        actual = {p.relative_to(adapter).as_posix() for p in adapter.rglob("*") if p.is_file()}
        if not names or names != actual:
            raise ValueError("file set")
        return expected
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise RuntimeError("runtime_adapter_binding_invalid") from exc


def _process_rows() -> dict[int, tuple[int, str]]:
    rows: dict[int, tuple[int, str]] = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            rows[int(path.name)] = (int(fields[1]), fields[19])
        except FileNotFoundError:
            continue
    return rows


def _descendants(rows: dict[int, tuple[int, str]], parent: int) -> dict[int, str]:
    owned = {parent}
    while True:
        following = owned | {pid for pid, (ppid, _) in rows.items() if ppid in owned}
        if following == owned:
            return {pid: rows[pid][1] for pid in owned if pid != parent}
        owned = following


def _signal_owned(pid: int, start: str, sig: signal.Signals) -> None:
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        # Read AFTER opening the pidfd; PID reuse must never select a new process.
        current = _process_rows()
        if pid not in current:
            return
        if current.get(pid, (None, None))[1] != start:
            raise RuntimeError("runtime_process_identity_changed")
        if pid not in _descendants(current, os.getpid()):
            raise RuntimeError("runtime_process_ownership_lost")
        signal.pidfd_send_signal(descriptor, sig)
    except ProcessLookupError:
        return
    finally:
        os.close(descriptor)


class OwnedRuntime:
    def __init__(self, model_id: str, digest: str, argv: list[str]) -> None:
        self.model_id = model_id
        self.digest = digest
        self.argv = list(argv)
        package = _mounted_package(model_id, digest) if "--enable-lora" not in argv else None
        self.command = _model_arguments(self.argv, model_id, package)
        self.base_only = package is not None and package.get("schema_version") == 2
        self.target = UpstreamTarget("http", "127.0.0.1", 8001, "/v1", model_id)
        self.operation = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = "UNLOADED"
        self._reason: str | None = None
        self._prepared = False
        self._adapter_content_hash = None
        self._child: subprocess.Popen[bytes] | None = None
        self._startup_log = b""
        self._startup_reader: threading.Thread | None = None
        self._owns_generation = False
        self._unexpected_exit = False
        self.idle_deadline = 0.0

    def _capture_startup(self, stream) -> None:
        # Drain throughout the child lifetime so its pipe cannot block. Retain
        # only a bounded startup tail in memory, never inference/request content.
        try:
            with stream:
                while block := stream.read1(4096):
                    with self._state_lock:
                        if self._state == "LOADING":
                            self._startup_log = (self._startup_log + block)[-65536:]
        except (OSError, ValueError):
            pass

    def _startup_failure(self, fallback: str) -> str:
        if fallback not in {"runtime_load_failed", "runtime_load_timeout"}:
            return fallback
        with self._state_lock:
            tail = self._startup_log
        # Expose library source locations, never traceback text or exception values.
        # These bounded locations make a native engine failure diagnosable locally.
        frames = re.findall(rb'File "[^"\r\n]*/((?:vllm|triton|torch)/[a-z0-9_/]{1,96}\.py)", line ([0-9]{1,6}),', tail)
        # Preserve native root locations before repeated vLLM wrapper frames.
        locations = list(dict.fromkeys(path.decode() + ":" + line.decode() for path, line in frames))
        locations = sorted(locations, key=lambda value: value.startswith("vllm/"))[:16]
        if locations:
            print("RUNTIME_STARTUP_DIAGNOSTIC " + ";".join(locations), flush=True)
        # Fixed exception names and numeric tensor dimensions only: exception
        # messages can contain paths, credentials or arbitrary provider text.
        kinds = re.findall(rb"\b(RuntimeError|ValueError|TypeError|AssertionError|IndexError|KeyError|OSError):", tail)
        if kinds:
            print("RUNTIME_STARTUP_EXCEPTION " + ";".join(dict.fromkeys(kind.decode() for kind in kinds)), flush=True)
        split = re.search(
            rb"split_with_sizes expects split_sizes to sum exactly to ([0-9]{1,8}) "
            rb"\(input tensor's size at dimension 0\), but got split_sizes=\[([0-9]{1,8}(?:, [0-9]{1,8}){0,15})\]",
            tail,
        )
        if split:
            print("RUNTIME_STARTUP_SHAPE split_total=" + split[1].decode()
                  + " split_sizes=" + split[2].decode().replace(" ", ""), flush=True)
        # HTTP callers receive fixed codes; never return/log raw child output.
        for marker, code in (
            (b"CUDA out of memory", "cuda_out_of_memory"),
            (b"Resource temporarily unavailable", "process_limit"),
            (b"Permission denied", "permission_denied"),
            (b"No space left on device", "storage_full"),
            (b"No available memory for the cache blocks", "kv_cache_memory"),
            (b"ModuleNotFoundError", "dependency_missing"),
        ):
            if marker in tail:
                return "runtime_load_" + code
        for marker, phase in (
            (b"Loading model weights took", "weights_loaded"),
            (b"Starting to load model", "weights_loading"),
            (b"Initializing a V1 LLM engine", "engine_initializing"),
            (b"Using max model len", "config_loaded"),
        ):
            if marker in tail:
                return fallback + "_" + phase
        return fallback

    def _set(self, state: str, reason: str | None = None) -> None:
        with self._state_lock:
            self._state, self._reason = state, reason
            if state == "FAILED":
                self._prepared = False

    def snapshot(self) -> dict[str, object]:
        with self._state_lock:
            if self._state == "READY" and (self._child is None or self._child.poll() is not None):
                self._state, self._reason, self._prepared = "FAILED", "runtime_process_exited", False
                self._unexpected_exit = True
            return {"schema_version": _SCHEMA, "model_id": self.model_id,
                    "package_digest": self.digest, "state": self._state,
                    "prepared": self._prepared, "reason": self._reason,
                    "adapter_content_hash": self._adapter_content_hash if self._prepared else None}

    def _reap(self) -> None:
        if self._child is not None:
            self._child.poll()
        # PID 1 adopts orphaned vLLM workers, even after the original child exits.
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break

    def _stop_owned(self) -> bool:
        started = time.monotonic()
        sent: set[tuple[int, str, signal.Signals]] = set()
        try:
            while time.monotonic() - started < 10:
                self._reap()
                children = _descendants(_process_rows(), os.getpid())
                if not children:
                    if self._startup_reader is not None:
                        self._startup_reader.join(timeout=1)
                        if self._startup_reader.is_alive():
                            return False
                        self._startup_reader = None
                    self._child = None
                    self._owns_generation = False
                    return True
                if not self._owns_generation:
                    return False  # Pre-existing children were never adopted as ours.
                sig = signal.SIGTERM if time.monotonic() - started < 5 else signal.SIGKILL
                for pid, identity in children.items():
                    token = (pid, identity, sig)
                    if token not in sent:
                        _signal_owned(pid, identity, sig)
                        sent.add(token)
                time.sleep(0.05)
        except (OSError, RuntimeError, ValueError, IndexError):
            pass
        return False

    def unload(self) -> None:
        failed = self.snapshot()["state"] == "FAILED"
        self._set("UNLOADING")
        if not self._stop_owned():
            self._set("FAILED", "runtime_cleanup_unconfirmed")
            raise RuntimeError("runtime_cleanup_unconfirmed")
        self._set("FAILED" if failed else "UNLOADED",
                  "runtime_recovery_required" if failed else None)

    def load(self) -> None:
        state = self.snapshot()["state"]
        if state == "READY":
            return
        if state != "UNLOADED":
            raise RuntimeError("runtime_recovery_required")
        self._set("LOADING")
        try:
            # Large artifact hashing belongs to the governed packaging gate.
            # Here check exact mounted identity; never claim a new integrity measurement.
            package = _mounted_package(self.model_id, self.digest)
            self._adapter_content_hash = _verify_adapter_identity(package)
            if package.get("schema_version") == 2:
                if _model_arguments(self.argv, self.model_id, package) != self.command:
                    raise RuntimeError("runtime_package_binding_invalid")
            elif (package.get("schema_version") != 1 or package.get("component") != "diagnosis"
                    or package.get("base_model_id") != "Qwen/Qwen3-0.6B"
                    or package.get("base_model_id") != self.command[self.command.index("--served-model-name") + 1]
                    or _model_arguments(self.argv, self.model_id) != self.command):
                raise RuntimeError("runtime_package_binding_invalid")
            if _descendants(_process_rows(), os.getpid()):
                raise RuntimeError("runtime_existing_children")
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       VLLM_SERVER_DEV_MODE="0")
            env.pop("IOAP_MODEL_GATEWAY_API_KEY_FILE", None)
            self._startup_log = b""
            self._child = subprocess.Popen(
                self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, env=env, start_new_session=True,
            )
            self._owns_generation = True
            self._startup_reader = threading.Thread(target=self._capture_startup,
                                                     args=(self._child.stdout,), daemon=True)
            self._startup_reader.start()
            deadline = time.monotonic() + _LOAD_SECONDS
            while time.monotonic() < deadline:
                if self._child.poll() is not None:
                    raise RuntimeError("runtime_load_failed")
                if _probe_target(self.target, 0.5):
                    with self._state_lock:
                        self._prepared = True
                        self._startup_log = b""
                    self._set("READY")
                    self.idle_deadline = time.monotonic() + 5
                    return
                time.sleep(0.2)
            raise RuntimeError("runtime_load_timeout")
        except Exception as exc:
            reason = str(exc) if isinstance(exc, RuntimeError) else "runtime_load_failed"
            if not self._stop_owned():
                reason = "runtime_cleanup_unconfirmed"
            reason = self._startup_failure(reason)
            self._set("FAILED", reason)
            self._startup_log = b""
            raise RuntimeError(reason) from exc


class RuntimeServer(ModelRouterServer):
    def __init__(self, config: RouterConfig, owner: OwnedRuntime) -> None:
        self.owner = owner
        super().__init__(("0.0.0.0", 8000), config)
        self.RequestHandlerClass = RuntimeHandler

    def service_actions(self) -> None:
        # Router loss after loading cannot leave an idle model resident forever.
        state = self.owner.snapshot()["state"]
        idle = state == "READY" and time.monotonic() >= self.owner.idle_deadline
        if ((idle or self.owner._unexpected_exit)
                and self.owner.operation.acquire(blocking=False)):
            try:
                self.owner._unexpected_exit = False
                self.owner.unload()
            except RuntimeError:
                pass  # FAILED is observable; never automatically reload.
            finally:
                self.owner.operation.release()


class RuntimeHandler(ModelRouterRequestHandler):
    @property
    def owner(self) -> OwnedRuntime:
        assert isinstance(self.server, RuntimeServer)
        return self.server.owner

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/runtime/status":
            self._send_json(200, self.owner.snapshot())
        elif self.path == "/health":
            ready = self.owner.snapshot()["state"] == "READY"
            self._send_json(200 if ready else 503, {"status": "READY" if ready else "NOT_READY"})
        elif self.path == "/v1/models" and self.owner.snapshot()["state"] == "READY":
            status, body = _read_upstream(self.owner.target, "GET", "/v1/models", 2, limit=65536)
            self._send_bytes(status or 503, body, content_type="application/json")
        else:
            self._send_error(404, "route_not_found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/runtime/load", "/runtime/unload", "/v1/chat/completions"}:
            self._send_error(404, "route_not_found")
            return
        if not self._authenticated():
            self._send_error(401, "router_authentication_required")
            return
        if not self.owner.operation.acquire(blocking=False):
            self._send_error(409, "runtime_busy")
            return
        timer = None
        try:
            if self.path == "/v1/chat/completions":
                if self.owner.snapshot()["state"] != "READY":
                    self._send_error(503, "runtime_model_not_loaded")
                    return
                seconds = (_SMOL_REQUEST_SECONDS if self.owner.base_only
                           else _DIAGNOSIS_REQUEST_SECONDS)
                self._request_failure = None
                self._request_expired = threading.Event()
                self._request_deadline = time.monotonic() + seconds
                self._upstream_socket = None
                timer = threading.Timer(seconds, self._expire_request)
                timer.daemon = True
                timer.start()
                super().do_POST()
            else:
                if (self.headers.get("Transfer-Encoding") is not None
                        or self.headers.get_all("Content-Length") != ["0"]):
                    self._send_error(400, "request_framing_invalid")
                    return
                if self.path == "/runtime/load":
                    self.owner.load()
                else:
                    self.owner.unload()
                self._send_json(200, self.owner.snapshot())
        except RuntimeError:
            self._send_error(503, str(self.owner.snapshot()["reason"] or "runtime_control_failed"))
        except (OSError, http.client.HTTPException):
            if timer is None:
                raise
            self._request_failure = "runtime_client_disconnected"
        finally:
            try:
                if timer is not None:
                    timer.cancel()
                    timer.join()
                    reason = ("runtime_request_timeout" if self._request_expired.is_set()
                              else self._request_failure)
                    if reason:
                        # Only the request thread cleans up, while still owning operation.
                        # The watchdog interrupts I/O; it never races PID ownership checks.
                        timed_out = reason == "runtime_request_timeout"
                        self.owner._set("UNLOADING" if timed_out else "FAILED", reason)
                        if not self.owner._stop_owned():
                            self.owner._set("FAILED", "runtime_cleanup_unconfirmed")
                        elif timed_out:
                            # Keep preparation provenance, not an active generation.
                            # A future independent request must still load explicitly.
                            self.owner._set("UNLOADED", reason)
                        print(f"runtime_request_failed primary={reason} "
                              f"cleanup={self.owner.snapshot()['state']}", file=sys.stderr, flush=True)
            finally:
                self.owner.operation.release()

    def _expire_request(self) -> None:
        self._request_expired.set()
        # shutdown also wakes buffered HTTPResponse/rfile reads. close alone does not.
        # Keep the exact socket objects, not fd numbers which can be reused.
        for channel in (self.connection, self._upstream_socket):
            if channel is not None:
                try:
                    channel.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _exchange(self, target: UpstreamTarget, body: bytes, *, credential: str | None = None
                  ) -> tuple[int, bytes, str]:
        # Bound loopback transport independently of the external router's timeout.
        # A saved socket is needed even when HTTPConnection clears sock at EOF.
        remaining = self._request_deadline - time.monotonic()
        if remaining <= 0 or self._request_expired.is_set():
            raise _AdmissionError(504, "runtime_request_timeout")
        connection = http.client.HTTPConnection(target.host, target.port,
                                                timeout=min(remaining, self.config.timeout_seconds))
        try:
            connection.connect()
            self._upstream_socket = connection.sock
            if self._request_expired.is_set():
                raise _AdmissionError(504, "runtime_request_timeout")
            connection.request("POST", f"{target.api_path}/chat/completions", body=body,
                               headers={"Accept": "application/json", "Content-Type": "application/json"})
            response = connection.getresponse()
            with response:
                if 300 <= response.status < 400:
                    raise _AdmissionError(502, "upstream_redirect_rejected")
                length = response.getheader("Content-Length")
                if length is not None and (not length.isascii() or not length.isdigit()):
                    raise _AdmissionError(502, "upstream_response_invalid")
                if length is not None and (len(length) > 20 or int(length) > self.config.response_max_bytes):
                    raise _AdmissionError(502, "upstream_response_too_large")
                result = response.read(self.config.response_max_bytes + 1)
                if len(result) > self.config.response_max_bytes:
                    raise _AdmissionError(502, "upstream_response_too_large")
                if length is not None and int(length) != len(result):
                    raise _AdmissionError(502, "upstream_response_invalid")
                if self._request_expired.is_set() or time.monotonic() >= self._request_deadline:
                    raise _AdmissionError(504, "runtime_request_timeout")
                return response.status, result, response.getheader("Content-Type", "application/json")
        except (OSError, http.client.HTTPException):
            raise _AdmissionError(502, "upstream_unavailable") from None
        finally:
            connection.close()
            self._upstream_socket = None

    def _forward(self, target: UpstreamTarget, body: bytes) -> None:
        # Only the router coordinates lifecycle. This hop talks to loopback vLLM.
        try:
            if self.owner.base_only:
                body = _smol_request(body)
            status, response, content_type = self._exchange(target, body)
            if status >= 500:
                self._request_failure = "upstream_unavailable"
            self._send_bytes(status, response, content_type=content_type)
        except _AdmissionError as exc:
            if exc.status >= 500:
                self._request_failure = exc.code
            self._send_error(exc.status, exc.code)


def _prepare(host: str) -> None:
    key = _read_credential(os.environ.get("IOAP_MODEL_GATEWAY_API_KEY_FILE", ""))
    for action, expected in (("load", "READY"), ("unload", "UNLOADED")):
        connection = http.client.HTTPConnection(
            host, 8000, timeout=_MODEL_LOAD_CONTROL_SECONDS if action == "load" else 12,
        )
        try:
            connection.request("POST", f"/runtime/{action}", body=b"",
                               headers={"Authorization": f"Bearer {key}"})
            response = connection.getresponse()
            raw = response.read(65537)
            if response.status != 200 or len(raw) > 65536:
                raise ValueError("control failed")
            row = json.loads(raw)
            prefix = "IOAP_DIAGNOSIS" if host == "diagnosis-model" else "IOAP_VLM"
            if (row.get("schema_version") != _SCHEMA or row.get("state") != expected
                    or row.get("model_id") != os.environ.get(prefix + "_MODEL_ID")
                    or row.get("package_digest") != os.environ.get(prefix + "_PACKAGE_DIGEST")
                    or row.get("prepared") is not True):
                raise ValueError("control identity failed")
        finally:
            connection.close()
    print("OWNED_MODEL_PREPARED")


def run() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--prepare" and sys.argv[2] in {"diagnosis-model", "vlm-model"}:
        try:
            _prepare(sys.argv[2])
        except Exception:
            raise SystemExit("RUNTIME_PREPARE_FAILED") from None
        return
    if os.getpid() != 1 or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise SystemExit("RUNTIME_STARTUP_FAILED:dedicated_pid_namespace_required")
    model_id = os.environ.get("IOAP_RUNTIME_MODEL_ID", "")
    digest = os.environ.get("IOAP_RUNTIME_PACKAGE_DIGEST", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", model_id):
        raise SystemExit("RUNTIME_STARTUP_FAILED:model_identity_invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SystemExit("RUNTIME_STARTUP_FAILED:package_identity_invalid")
    try:
        owner = OwnedRuntime(model_id, digest, sys.argv[1:])
        key = _read_credential(os.environ.get("IOAP_MODEL_GATEWAY_API_KEY_FILE", ""))
        config = RouterConfig(owner.target, owner.target, key, timeout_seconds=(
            _SMOL_REQUEST_SECONDS if owner.base_only else _DIAGNOSIS_REQUEST_SECONDS))
        server = RuntimeServer(config, owner)
    except (OSError, ValueError, RuntimeError, RouterConfigurationError):
        raise SystemExit("RUNTIME_STARTUP_FAILED:configuration_invalid") from None

    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        # Container shutdown must not race an in-flight load. Container PID-1
        # exit remains the namespace backstop if the bounded operation stalls.
        with owner.operation:
            owner.unload()
        server.server_close()


if __name__ == "__main__":
    run()
