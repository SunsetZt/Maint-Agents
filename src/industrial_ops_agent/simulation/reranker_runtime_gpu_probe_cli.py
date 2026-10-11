"""Execute one resource-bounded actual-GPU Reranker runtime probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, cast

from industrial_ops_agent.simulation.reranker_kserve_readiness import (
    RUNTIME_IMAGE_REPOSITORY,
    RUNTIME_IMAGE_TAG,
    verify_reranker_kserve_readiness,
)
from industrial_ops_agent.simulation.reranker_runtime_gpu_probe import (
    ACCEPTANCE_RELATIVE,
    CONTAINER_NAME,
    EVIDENCE_FILES,
    OUTPUT_RELATIVE,
    PROBE_MANIFEST_HASH,
    PROBE_RELEASE_ID,
    EnterpriseRerankerRuntimeGpuProbeReport,
    RerankerRuntimeGpuProbeError,
    build_runtime_gpu_probe_report,
    probe_request,
    verify_runtime_gpu_probe_report,
)

_HTTP_SCRIPT = r"""
import json, sys, urllib.error, urllib.request
request = json.loads(sys.stdin.read())
body = request.get("body")
data = (
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    if body is not None
    else None
)
headers = {"Content-Type": "application/json", "Accept": "application/json"}
target = urllib.request.Request(
    "http://127.0.0.1:8080" + request["path"],
    data=data,
    headers=headers,
    method=request["method"],
)
try:
    with urllib.request.urlopen(target, timeout=30) as response:
        raw = response.read()
        status = response.status
except urllib.error.HTTPError as error:
    raw = error.read()
    status = error.code
decoded = json.loads(raw) if raw else None
print(
    json.dumps(
        {
            "schema_version": "container-http-result/v1",
            "status_code": status,
            "body": decoded,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
""".strip()

_TEXT_SCRIPT = r"""
import sys, urllib.request
with urllib.request.urlopen("http://127.0.0.1:8080" + sys.argv[1], timeout=30) as response:
    sys.stdout.buffer.write(response.read())
""".strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Actual-GPU Reranker runtime probe")
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=ACCEPTANCE_RELATIVE)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "verify":
        report = verify_runtime_gpu_probe_report(root, args.output)
    else:
        report = _run_probe(root, args.output)
    print(report.model_dump_json(indent=2))
    return 0


def _run_probe(
    root: Path,
    output_path: Path,
) -> EnterpriseRerankerRuntimeGpuProbeReport:
    readiness = verify_reranker_kserve_readiness(root)
    output = root / OUTPUT_RELATIVE
    output.mkdir(parents=True, exist_ok=True)
    image_ref = f"{RUNTIME_IMAGE_REPOSITORY}:{RUNTIME_IMAGE_TAG}"
    candidate = (root / readiness.source.candidate_path).resolve(strict=True)
    _require_container_absent()
    started = time.monotonic()
    running = False
    try:
        _run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                CONTAINER_NAME,
                "--gpus",
                "device=0",
                "--cpus",
                "2",
                "--memory",
                "4g",
                "--memory-swap",
                "4g",
                "--pids-limit",
                "256",
                "--network",
                "none",
                "--read-only",
                "--security-opt",
                "no-new-privileges",
                "--cap-drop",
                "ALL",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=256m",
                "--env",
                "HF_HUB_OFFLINE=1",
                "--env",
                "TRANSFORMERS_OFFLINE=1",
                "--env",
                "TOKENIZERS_PARALLELISM=false",
                "--volume",
                f"{candidate}:/mnt/models/reranker:ro",
                image_ref,
                "--model-dir",
                "/mnt/models/reranker",
                "--release-id",
                PROBE_RELEASE_ID,
                "--manifest-hash",
                PROBE_MANIFEST_HASH,
                "--component-model-id",
                readiness.source.run_id,
                "--artifact-content-hash",
                readiness.source.candidate_bundle_sha256,
                "--precision",
                "bfloat16",
                "--batch-size",
                "4",
                "--max-sequence-length",
                "128",
                "--port",
                "8080",
            ],
            timeout=30,
        )
        running = True
        health = _wait_ready(timeout_seconds=120)
        _write_json(output / EVIDENCE_FILES["health"], health)
        request = probe_request()
        request.pop("probe_metadata")
        _write_json(output / EVIDENCE_FILES["probe_request"], probe_request())
        first = _http("POST", "/v1/retrieval/rerank", request)
        second = _http("POST", "/v1/retrieval/rerank", request)
        missing = _http(
            "POST",
            "/v1/retrieval/rerank",
            {**request, "query": "No reviewed field alias appears in this release probe."},
        )
        ambiguous = _http(
            "POST",
            "/v1/retrieval/rerank",
            {**request, "query": "Compare amber cascade with hollow pulse."},
        )
        wrong_binding = _http(
            "POST",
            "/v1/retrieval/rerank",
            {**request, "expected_manifest_hash": "0" * 64},
        )
        for name, value in (
            ("first_response", first),
            ("second_response", second),
            ("missing_alias_rejection", missing),
            ("ambiguous_alias_rejection", ambiguous),
            ("binding_rejection", wrong_binding),
        ):
            _write_json(output / EVIDENCE_FILES[name], value)
        metrics = _container_text("/metrics")
        (output / EVIDENCE_FILES["metrics"]).write_text(metrics, encoding="utf-8")
        inspect = json.loads(_run(["docker", "inspect", CONTAINER_NAME], timeout=20).stdout)
        _write_json(output / EVIDENCE_FILES["container_inspect"], inspect)
        gpu = _run(
            [
                "docker",
                "exec",
                CONTAINER_NAME,
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=20,
        ).stdout
        (output / EVIDENCE_FILES["gpu_inventory"]).write_text(gpu, encoding="utf-8")
        log_result = _run(["docker", "logs", CONTAINER_NAME], timeout=20)
        logs = log_result.stdout + log_result.stderr
        (output / EVIDENCE_FILES["container_logs"]).write_text(
            logs or "runtime completed without stdout/stderr log output\n",
            encoding="utf-8",
        )
    finally:
        if running:
            subprocess.run(
                ["docker", "stop", "--time", "10", CONTAINER_NAME],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            subprocess.run(
                ["docker", "rm", "--force", CONTAINER_NAME],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
    _require_container_absent()
    duration = time.monotonic() - started
    if duration > 180:
        raise RerankerRuntimeGpuProbeError("Reranker runtime probe exceeded its budget")
    return build_runtime_gpu_probe_report(
        root,
        duration_seconds=duration,
        output_path=output_path,
    )


def _wait_ready(*, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "runtime_not_started"
    while time.monotonic() < deadline:
        try:
            result = _http("GET", "/health/ready", None)
            if result.get("status_code") == 200 and isinstance(result.get("body"), dict):
                return cast(dict[str, Any], result["body"])
            last_error = json.dumps(result, sort_keys=True)
        except RerankerRuntimeGpuProbeError as exc:
            last_error = str(exc)
        state = _run(
            [
                "docker",
                "container",
                "inspect",
                "--format",
                "{{.State.Status}}",
                CONTAINER_NAME,
            ],
            check=False,
            timeout=10,
        )
        if state.returncode == 0 and state.stdout.strip() != "running":
            log_result = _run(["docker", "logs", CONTAINER_NAME], check=False, timeout=20)
            logs = log_result.stdout + log_result.stderr
            raise RerankerRuntimeGpuProbeError(
                f"Reranker runtime exited before readiness: {state.stdout.strip()}; "
                f"logs={logs[-4000:]}"
            )
        time.sleep(1)
    log_result = _run(["docker", "logs", CONTAINER_NAME], check=False, timeout=20)
    logs = log_result.stdout + log_result.stderr
    raise RerankerRuntimeGpuProbeError(
        f"Reranker runtime did not become ready: {last_error}; logs={logs[-2000:]}"
    )


def _http(method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
    result = _run(
        ["docker", "exec", "-i", CONTAINER_NAME, "python", "-c", _HTTP_SCRIPT],
        input_text=json.dumps({"method": method, "path": path, "body": body}),
        timeout=40,
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RerankerRuntimeGpuProbeError("Reranker container HTTP result is invalid") from exc
    if not isinstance(value, dict):
        raise RerankerRuntimeGpuProbeError("Reranker container HTTP result is invalid")
    return cast(dict[str, Any], value)


def _container_text(path: str) -> str:
    return _run(
        ["docker", "exec", CONTAINER_NAME, "python", "-c", _TEXT_SCRIPT, path],
        timeout=30,
    ).stdout


def _require_container_absent() -> None:
    result = _run(
        ["docker", "container", "inspect", CONTAINER_NAME],
        check=False,
        timeout=10,
    )
    if result.returncode == 0:
        raise RerankerRuntimeGpuProbeError(
            f"owned Reranker probe container already exists: {CONTAINER_NAME}"
        )


def _run(
    command: list[str],
    *,
    check: bool = True,
    input_text: str | None = None,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            input=input_text,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RerankerRuntimeGpuProbeError(f"command failed: {command[0]}") from exc
    if check and result.returncode != 0:
        raise RerankerRuntimeGpuProbeError(
            f"command failed: {' '.join(command[:3])}: {result.stderr[-2000:]}"
        )
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
