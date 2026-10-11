"""CLI for the one-shot actual-GPU PPO Agent Runtime value evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from industrial_ops_agent.simulation.ppo_agent_runtime_value_lab import (
    CONTAINER_NAME,
    IMAGE,
    LATEST_RELATIVE,
    PpoAgentRuntimeValueLabError,
    PpoAgentRuntimeValueReport,
    execute_worker,
    planned_run_id,
    prepare_gold_manifest,
    verify_ppo_agent_runtime_value,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-ppo-agent-runtime-value-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "verify", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--repo-root", type=Path, default=Path.cwd())
        if name == "worker":
            command.add_argument("--image-digest", required=True)
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "prepare":
        path = prepare_gold_manifest(root)
        print("PPO_AGENT_RUNTIME_GOLD_FROZEN")
        print(path)
        return
    if args.command == "worker":
        path = execute_worker(root, image_digest=args.image_digest)
        report = verify_ppo_agent_runtime_value(root, path)
    elif args.command == "verify":
        report = verify_ppo_agent_runtime_value(root)
        path = root / LATEST_RELATIVE
    else:
        report, path = _run_container(root)
    _print_report(report, path)


def _run_container(root: Path) -> tuple[PpoAgentRuntimeValueReport, Path]:
    prepare_gold_manifest(root)
    image_digest = _capture_checked(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        cwd=root,
    )
    run_id = planned_run_id(root, image_digest)
    for path in (
        root / "artifacts/m7-ppo-agent-runtime-value-lab/runs" / run_id / "acceptance.json",
        root / "artifacts/m7-ppo-agent-runtime-value-lab/rejections" / run_id / "rejection.json",
    ):
        if path.is_file():
            return verify_ppo_agent_runtime_value(root, path), path
    if _container_exists(root):
        raise PpoAgentRuntimeValueLabError(
            f"PPO Agent Runtime lab container already exists: {CONTAINER_NAME}"
        )
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
        "--mount",
        f"type=bind,src={root},dst=/workspace",
        "--workdir",
        "/workspace",
        "--entrypoint",
        "python",
        IMAGE,
        "-B",
        "-m",
        "industrial_ops_agent.simulation.ppo_agent_runtime_value_lab_cli",
        "worker",
        "--repo-root",
        "/workspace",
        "--image-digest",
        image_digest,
    ]
    try:
        subprocess.run(command, cwd=root, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PpoAgentRuntimeValueLabError("actual GPU PPO Agent Runtime container failed") from exc
    finally:
        if _container_exists(root):
            _remove_owned_container(root)
        if _container_exists(root):
            raise PpoAgentRuntimeValueLabError("PPO Agent Runtime lab container cleanup failed")
    report = verify_ppo_agent_runtime_value(root)
    latest = root / LATEST_RELATIVE
    return report, latest


def _print_report(report: PpoAgentRuntimeValueReport, path: Path) -> None:
    print(report.status)
    print(report.classification)
    print(f"CANDIDATE_ACCEPTED={str(report.candidate_accepted).lower()}")
    print(f"BASELINE_TARGET_EXACT_RATE={report.evaluation.baseline.target_exact_rate:.6f}")
    print(f"CANDIDATE_TARGET_EXACT_RATE={report.evaluation.candidate.target_exact_rate:.6f}")
    print(f"FAILED_HARD_GATES={','.join(report.failed_hard_gates)}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


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
        raise PpoAgentRuntimeValueLabError(
            "PPO Agent Runtime lab container cleanup failed"
        ) from exc


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
        raise PpoAgentRuntimeValueLabError(f"command failed: {command[0]}") from exc
    value = completed.stdout.strip()
    if not value:
        raise PpoAgentRuntimeValueLabError(f"command returned no output: {command[0]}")
    return value


if __name__ == "__main__":
    run()
