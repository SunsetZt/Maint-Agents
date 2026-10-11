"""Shared subprocess output contracts and bounded Reranker worker arguments."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def capture_required_output(command: list[str], *, cwd: Path, error_type: type[Exception]) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise error_type(f"command failed: {command[0]}") from exc
    value = completed.stdout.strip()
    if not value:
        raise error_type(f"command returned no output: {command[0]}")
    return value


def capture_merged_output(argv: list[str], *, cwd: Path, error_type: type[Exception]) -> str:
    result = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise error_type(
            f"command failed ({argv[0]}): {result.stdout[-1_024:]}"
        )
    return result.stdout.strip()


def reranker_worker_command(
    root: Path, *, container_name: str, image: str,
    worker_module: str, image_digest: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
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
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=512m,mode=1777",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--env",
        "PYTHONPATH=/workspace/src",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONUNBUFFERED=1",
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
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        "HF_HOME=/tmp/hf",
        "--env",
        "TRITON_CACHE_DIR=/tmp/triton",
        "--env",
        "TORCH_EXTENSIONS_DIR=/tmp/torch-extensions",
        "--env",
        "IOAP_NETWORK_DISABLED=1",
        "--env",
        "IOAP_READ_ONLY_ROOTFS=1",
        "--mount",
        f"type=bind,src={root},dst=/workspace",
        "--workdir",
        "/workspace",
        "--entrypoint",
        "python",
        image,
        "-B",
        "-m",
        worker_module,
        "worker",
        "--repo-root",
        "/workspace",
        "--image-digest",
        image_digest,
    ]


def start_service_port_forward(
    root: Path, *, kube_context: str, namespace: str,
    resource: str, local_port: int,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            "kubectl",
            "--context",
            kube_context,
            "--namespace",
            namespace,
            "port-forward",
            resource,
            f"{local_port}:80",
            "--address",
            "127.0.0.1",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def resource_wait_command(
    kube_context: str, namespace: str, resource: str, name: str,
    condition: str, timeout_seconds: int,
) -> list[str]:
    return [
        "kubectl",
        "--context",
        kube_context,
        "--namespace",
        namespace,
        "wait",
        f"{resource}/{name}",
        f"--for=condition={condition}",
        f"--timeout={timeout_seconds}s",
    ]
