"""Shared local KServe command and process operations; no rollout policy decisions."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from industrial_ops_agent.simulation.commands import (
    resource_wait_command,
    start_service_port_forward,
)


@dataclass(frozen=True, slots=True)
class ControlPlaneOperations:
    error_type: type[RuntimeError]
    kube_context: str
    kind_node: str
    namespace: str
    proxy_image: str
    kserve_namespace: str
    envoy_gateway_namespace: str
    gateway_name: str

    def ensure_cluster_running(self, root: Path) -> None:
        running = self.capture_checked(
            ["docker", "inspect", self.kind_node, "--format", "{{.State.Running}}"],
            cwd=root,
        )
        if running != "true":
            self.run_checked(["docker", "start", self.kind_node], cwd=root)
        observed_context = self.capture_checked(["kubectl", "config", "current-context"], cwd=root)
        if observed_context != self.kube_context:
            self.run_checked(["kubectl", "config", "use-context", self.kube_context], cwd=root)
        self.wait_for_cluster_api_access(root, timeout_seconds=90)
        self.run_checked(
            [
                "kubectl",
                "--context",
                self.kube_context,
                "wait",
                "--for=condition=Ready",
                f"node/{self.kind_node}",
                "--timeout=180s",
            ],
            cwd=root,
        )
        self.wait_control_plane_pods(
            root,
            namespace=self.kserve_namespace,
            selector="control-plane=kserve-controller-manager",
            timeout_seconds=120,
        )
        self.wait_control_plane_pods(
            root,
            namespace=self.envoy_gateway_namespace,
            selector=None,
            timeout_seconds=120,
        )
        self.run_checked(
            [
                "kubectl",
                "--context",
                self.kube_context,
                "get",
                "crd/inferenceservices.serving.kserve.io",
            ],
            cwd=root,
        )

    def wait_for_cluster_api_access(self, root: Path, *, timeout_seconds: int) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_output = "Kubernetes API did not answer"
        command = [
            "kubectl",
            "--context",
            self.kube_context,
            "--request-timeout=5s",
            "get",
            f"node/{self.kind_node}",
            "--output=name",
        ]
        while time.monotonic() < deadline:
            result = subprocess.run(
                command,
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if result.returncode == 0:
                return
            last_output = result.stdout.strip() or last_output
            time.sleep(2)
        raise self.error_type(
            f"Kind Kubernetes API did not become accessible: {last_output[-1_024:]}"
        )

    def wait_control_plane_pods(
        self,
        root: Path,
        *,
        namespace: str,
        selector: str | None,
        timeout_seconds: int,
    ) -> None:
        command = [
            "kubectl",
            "--context",
            self.kube_context,
            "--namespace",
            namespace,
            "wait",
            "pod",
            "--for=condition=Ready",
            f"--timeout={timeout_seconds}s",
        ]
        if selector is None:
            command.append("--all")
        else:
            command.extend(("--selector", selector))
        self.run_checked(command, cwd=root)

    def node_image_digest(self, root: Path) -> str:
        normalized_image = f"docker.io/{self.proxy_image}"
        raw_document = self.capture_checked(
            ["docker", "exec", self.kind_node, "crictl", "inspecti", normalized_image],
            cwd=root,
        )
        try:
            document = json.loads(raw_document)
        except json.JSONDecodeError as exc:
            raise self.error_type("Kind node returned invalid proxy image metadata") from exc
        status = document.get("status")
        if not isinstance(status, dict):
            raise self.error_type("Kind node proxy image status is missing")
        digest = status.get("id")
        repo_tags = status.get("repoTags")
        if (
            not isinstance(digest, str)
            or not valid_image_digest(digest)
            or not isinstance(repo_tags, list)
            or normalized_image not in repo_tags
        ):
            raise self.error_type("Kind node fixed proxy image binding changed")
        return digest

    def image_digest(self, root: Path, image: str) -> str:
        value = self.capture_checked(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            cwd=root,
        )
        if not valid_image_digest(value):
            raise self.error_type("worker image digest is invalid")
        return value

    def wait_resource(
        self,
        root: Path,
        resource: str,
        name: str,
        *,
        condition: str,
        timeout_seconds: int,
    ) -> None:
        self.run_checked(
            resource_wait_command(
                self.kube_context, self.namespace, resource, name,
                condition, timeout_seconds,
            ),
            cwd=root,
        )

    def wait_service_endpoint(self, root: Path, service: str, *, timeout_seconds: int) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            document = self.kubectl_json(root, "endpoints", service)
            subsets = document.get("subsets")
            if isinstance(subsets, list) and any(
                isinstance(item, dict) and item.get("addresses") for item in subsets
            ):
                return
            time.sleep(1)
        raise self.error_type(f"KServe endpoint is not ready: {service}")

    def gateway_service(self, root: Path) -> tuple[str, str]:
        document = json.loads(
            self.capture_checked(
                ["kubectl", "--context", self.kube_context, "get", "service", "-A", "-o", "json"],
                cwd=root,
            )
        )
        matches: list[tuple[str, str]] = []
        for item in document.get("items", []):
            metadata = item.get("metadata", {})
            labels = metadata.get("labels", {})
            if (
                labels.get("gateway.envoyproxy.io/owning-gateway-name") == self.gateway_name
                and labels.get("gateway.envoyproxy.io/owning-gateway-namespace") == self.namespace
            ):
                matches.append((metadata.get("namespace", ""), metadata.get("name", "")))
        if len(matches) != 1 or not all(matches[0]):
            raise self.error_type("Envoy Gateway service is not unique")
        return matches[0]

    @contextmanager
    def port_forward(
        self,
        root: Path,
        *,
        namespace: str,
        resource: str,
        local_port: int,
    ) -> Iterator[None]:
        process = start_service_port_forward(
            root, kube_context=self.kube_context, namespace=namespace,
            resource=resource, local_port=local_port,
        )
        try:
            self.wait_tcp_process(process, local_port, timeout_seconds=30)
            yield
        finally:
            stop_process(process)

    def wait_tcp_process(
        self,
        process: subprocess.Popen[str],
        port: int,
        *,
        timeout_seconds: int,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout is not None else ""
                raise self.error_type(f"port-forward failed: {output[-512:]}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.25)
        raise self.error_type("port-forward did not become ready")

    def kubectl_json(self, root: Path, resource: str, name: str) -> dict[str, Any]:
        raw = self.capture_checked(
            [
                "kubectl",
                "--context",
                self.kube_context,
                "--namespace",
                self.namespace,
                "get",
                resource,
                name,
                "-o",
                "json",
            ],
            cwd=root,
        )
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise self.error_type("Kubernetes resource response is invalid")
        return document

    def stop_cluster(self, root: Path, *, check: bool = True) -> None:
        result = subprocess.run(
            ["docker", "stop", "--time", "30", self.kind_node],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if check and result.returncode != 0:
            raise self.error_type(f"Kind cluster stop failed: {result.stdout[-512:]}")

    def run_checked(self, argv: list[str], *, cwd: Path) -> None:
        result = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            raise self.error_type(f"command failed ({argv[0]}): {result.stdout[-1_024:]}")

    def capture_checked(self, argv: list[str], *, cwd: Path) -> str:
        result = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            raise self.error_type(f"command failed ({argv[0]}): {result.stdout[-1_024:]}")
        return result.stdout.strip()

    def run_input_checked(self, argv: list[str], payload: bytes, *, cwd: Path) -> None:
        result = subprocess.run(
            argv,
            cwd=cwd,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            raise self.error_type(
                f"command failed ({argv[0]}): {result.stdout[-1_024:].decode(errors='replace')}"
            )


@dataclass(frozen=True, slots=True)
class RolloutOperations(ControlPlaneOperations):
    label: str
    worker_container: str
    worker_module: str
    field_manager: str
    route_name: str
    stable_service: str
    candidate_service: str
    config_map_name: str
    route_condition: Callable[[dict[str, Any], str], bool]

    def start_worker(
        self,
        root: Path,
        *,
        port: int,
        log: Any,
    ) -> subprocess.Popen[str]:
        if container_exists(root, self.worker_container):
            raise self.error_type(
                f"owned {self.label} worker container already exists: {self.worker_container}"
            )
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            self.worker_container,
            "--gpus",
            "device=0",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cpus",
            "2",
            "--memory",
            "4g",
            "--memory-swap",
            "4g",
            "--pids-limit",
            "256",
            "--shm-size",
            "256m",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=512m,mode=1777",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--publish",
            f"{port}:8080",
            "--env",
            "PYTHONPATH=/workspace/src",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "HOME=/tmp/home",
            "--env",
            "HF_HOME=/tmp/hf",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "TOKENIZERS_PARALLELISM=false",
            "--env",
            "OMP_NUM_THREADS=2",
            "--env",
            "MKL_NUM_THREADS=2",
            "--env",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "--mount",
            f"type=bind,src={root},dst=/workspace,readonly",
            "--workdir",
            "/workspace",
            "--entrypoint",
            "python",
            self.proxy_image,
            "-B",
            "-m",
            self.worker_module,
            "--repo-root",
            "/workspace",
            "--model-path",
            "/models/base",
            "--port",
            "8080",
        ]
        return subprocess.Popen(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def kubectl_apply(self, root: Path, document: dict[str, Any]) -> None:
        self.run_input_checked(
            [
                "kubectl",
                "--context",
                self.kube_context,
                "apply",
                "--server-side",
                "--field-manager",
                self.field_manager,
                "-f",
                "-",
            ],
            json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode(),
            cwd=root,
        )

    def wait_route(self, root: Path, *, timeout_seconds: int) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            route = self.kubectl_json(root, "httproute", self.route_name)
            if self.route_condition(route, "Accepted") and self.route_condition(
                route, "ResolvedRefs"
            ):
                time.sleep(0.5)
                return
            time.sleep(0.5)
        raise self.error_type(f"{self.label} HTTPRoute was not accepted and resolved")

    def wait_worker(
        self,
        process: subprocess.Popen[str],
        worker_url: str,
        *,
        timeout_seconds: int,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise self.error_type(f"{self.label} GPU worker exited during startup")
            try:
                document = self.get_json(f"{worker_url}/health/ready", timeout_seconds=2)
                if document.get("ready") is True:
                    return
            except (httpx.HTTPError, self.error_type):
                pass
            time.sleep(1)
        raise self.error_type(f"{self.label} GPU worker did not become ready")

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        with httpx.Client(timeout=httpx.Timeout(timeout_seconds), trust_env=False) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            document = response.json()
        if not isinstance(document, dict):
            raise self.error_type(f"{self.label} response is invalid")
        return document

    def get_json(self, url: str, *, timeout_seconds: int) -> dict[str, Any]:
        with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
            response = client.get(url)
            response.raise_for_status()
            document = response.json()
        if not isinstance(document, dict):
            raise self.error_type(f"{self.label} runtime response is invalid")
        return document

    def wait_candidate_delta(
        self,
        worker_url: str,
        before: dict[str, Any],
        *,
        minimum: int,
        timeout_seconds: int,
    ) -> None:
        original = self.variant_count(before, "request_counts", "candidate")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            current = self.get_json(f"{worker_url}/metrics", timeout_seconds=5)
            if self.variant_count(current, "request_counts", "candidate") - original >= minimum:
                return
            time.sleep(0.25)
        raise self.error_type(f"Shadow traffic was not mirrored to {self.label} candidate")

    def delete_resources(self, root: Path, *, check: bool = True) -> None:
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                self.kube_context,
                "--namespace",
                self.namespace,
                "delete",
                f"httproute/{self.route_name}",
                f"inferenceservice/{self.stable_service}",
                f"inferenceservice/{self.candidate_service}",
                f"configmap/{self.config_map_name}",
                "--ignore-not-found=true",
                "--wait=true",
                "--timeout=120s",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if check and result.returncode != 0:
            raise self.error_type(f"{self.label} resource cleanup failed: {result.stdout[-512:]}")

    def stop_worker(
        self,
        root: Path,
        process: subprocess.Popen[str],
        *,
        check: bool = True,
    ) -> None:
        if container_exists(root, self.worker_container):
            result = subprocess.run(
                ["docker", "stop", "--time", "20", self.worker_container],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if check and result.returncode != 0:
                raise self.error_type(f"{self.label} worker cleanup failed: {result.stdout[-512:]}")
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        if check and container_exists(root, self.worker_container):
            raise self.error_type(f"{self.label} worker container still exists after cleanup")

    def resolve_latest_outcome(self, root: Path, pointer_path: Path) -> Path:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict):
            raise self.error_type(f"{self.label} runtime pointer is invalid")
        value = pointer.get("outcome") or pointer.get("report")
        if not isinstance(value, str) or not value:
            raise self.error_type(f"{self.label} runtime pointer has no outcome")
        path = (root / value).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise self.error_type(f"{self.label} runtime pointer escaped repository") from exc
        return path

    def variant_count(self, document: dict[str, Any], field: str, variant: str) -> int:
        values = document.get(field)
        if not isinstance(values, dict):
            raise self.error_type(f"runtime metric is invalid: {field}")
        value = values.get(variant)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise self.error_type(f"runtime metric is invalid: {field}.{variant}")
        return value

    def text(self, document: dict[str, Any], field: str) -> str:
        value = document.get(field)
        if not isinstance(value, str) or not value:
            raise self.error_type(f"field is invalid: {field}")
        return value

    def positive_integer(self, document: dict[str, Any], field: str) -> int:
        value = document.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise self.error_type(f"field is invalid: {field}")
        return value


def container_exists(root: Path, name: str) -> bool:
    return (
        subprocess.run(
            ["docker", "container", "inspect", name],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def valid_image_digest(value: str) -> bool:
    return (
        value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return ordered[index]


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def write_json(path: Path, document: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_log_tail(path: Path, *, max_chars: int = 4_096) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""
