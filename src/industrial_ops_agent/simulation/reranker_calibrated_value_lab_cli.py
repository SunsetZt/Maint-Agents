"""CLI for the low-memory actual-GPU calibrated Reranker value lab v5."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from industrial_ops_agent.simulation.reranker_calibrated_value_lab import (
    CONTAINER_NAME,
    IMAGE,
    OUTPUT_RELATIVE,
    RerankerCalibratedValueLabError,
    RerankerCalibratedValueReport,
    assert_gold_available,
    execute_worker,
    planned_run_id,
    preflight_calibrated_reranker,
    prepare_inputs,
    verify_calibrated_reranker_value,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="industrial-ops-reranker-calibrated-value-lab"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "preflight", "run", "verify", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--repo-root", type=Path, default=Path.cwd())
        if name == "worker":
            command.add_argument("--image-digest", required=True)
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "prepare":
        print("RERANKER_CALIBRATED_INPUTS_PREPARED")
        print(prepare_inputs(root))
        return
    if args.command == "preflight":
        result = preflight_calibrated_reranker(root)
        print(result["status"])
        print(f"SOURCE_V4_RUN_ID={result['source_v4_run_id']}")
        print(f"GOLD_CONSUMED={str(result['gold_consumed']).lower()}")
        return
    if args.command == "worker":
        path = execute_worker(root, image_digest=args.image_digest)
        report = verify_calibrated_reranker_value(root, path)
    elif args.command == "verify":
        report = verify_calibrated_reranker_value(root)
        path = root / OUTPUT_RELATIVE / "latest.json"
    else:
        report, path = _run_container(root)
    _print_report(report, path)


def _print_report(report: RerankerCalibratedValueReport, path: Path) -> None:
    print(report.status)
    print(report.classification)
    print(f"ACTUAL_GPU_INFERENCE={report.runtime.actual_gpu_inference}")
    print(f"BASELINE_NDCG_AT_3={report.evaluation.baseline_ndcg_at_k:.6f}")
    print(f"CANDIDATE_NDCG_AT_3={report.evaluation.candidate_ndcg_at_k:.6f}")
    print(f"NDCG_AT_3_IMPROVEMENT={report.evaluation.primary_improvement:.6f}")
    print(f"CANDIDATE_ACCEPTED={report.candidate_accepted}")
    print(f"FAILED_HARD_GATES={','.join(report.failed_hard_gates)}")
    print(f"GPU_PEAK_RESERVED_BYTES={report.runtime.peak_gpu_memory_reserved_bytes}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


def _run_container(root: Path) -> tuple[RerankerCalibratedValueReport, Path]:
    image_digest = _capture_checked(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], cwd=root
    )
    run_id = planned_run_id(root, image_digest)
    acceptance = root / OUTPUT_RELATIVE / "runs" / run_id / "acceptance.json"
    rejection = root / OUTPUT_RELATIVE / "rejections" / run_id / "rejection.json"
    for existing in (acceptance, rejection):
        if existing.is_file():
            return verify_calibrated_reranker_value(root, existing), existing
    assert_gold_available(root)
    if _container_exists(root):
        raise RerankerCalibratedValueLabError(
            f"Reranker v5 lab container already exists: {CONTAINER_NAME}"
        )
    (root / OUTPUT_RELATIVE).mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        CONTAINER_NAME,
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
        IMAGE,
        "-B",
        "-m",
        "industrial_ops_agent.simulation.reranker_calibrated_value_lab_cli",
        "worker",
        "--repo-root",
        "/workspace",
        "--image-digest",
        image_digest,
    ]
    try:
        subprocess.run(command, cwd=root, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RerankerCalibratedValueLabError(
            "actual GPU Reranker v5 container failed; Gold is retired if evaluation began"
        ) from exc
    finally:
        if _container_exists(root):
            _remove_owned_container(root)
        if _container_exists(root):
            raise RerankerCalibratedValueLabError("Reranker v5 lab cleanup failed")
    for outcome in (acceptance, rejection):
        if outcome.is_file():
            return verify_calibrated_reranker_value(root, outcome), outcome
    raise RerankerCalibratedValueLabError(
        "Reranker v5 worker produced no governed outcome"
    )


def _container_exists(root: Path) -> bool:
    completed = subprocess.run(
        ["docker", "container", "inspect", CONTAINER_NAME],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _remove_owned_container(root: Path) -> None:
    try:
        subprocess.run(
            ["docker", "container", "rm", "--force", CONTAINER_NAME],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RerankerCalibratedValueLabError("Reranker v5 lab cleanup failed") from exc


def _capture_checked(command: list[str], *, cwd: Path) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RerankerCalibratedValueLabError(f"command failed: {command[0]}") from exc
    value = completed.stdout.strip()
    if not value:
        raise RerankerCalibratedValueLabError(
            f"command returned no output: {command[0]}"
        )
    return value


if __name__ == "__main__":
    run()
