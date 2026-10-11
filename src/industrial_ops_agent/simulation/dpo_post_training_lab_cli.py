"""CLI for the low-memory, actual-GPU DPO project lab."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from industrial_ops_agent.simulation.dpo_post_training_lab import (
    CONTAINER_NAME,
    IMAGE,
    OUTPUT_RELATIVE,
    DpoPostTrainingLabError,
    DpoPostTrainingReport,
    execute_worker,
    planned_run_id,
    verify_dpo_post_training,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-dpo-post-training-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--repo-root", type=Path, default=Path.cwd())
        if name == "worker":
            command.add_argument("--image-digest", required=True)
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "worker":
        path = execute_worker(root, image_digest=args.image_digest)
        report = verify_dpo_post_training(root, path)
    elif args.command == "verify":
        report = verify_dpo_post_training(root)
        path = root / OUTPUT_RELATIVE / "latest.json"
    else:
        report, path = _run_container(root)
    print(report.status)
    print(report.classification)
    print(f"ACTUAL_GPU_TRAINING={report.runtime.actual_gpu_execution}")
    print(f"MARGIN_IMPROVEMENT={report.evaluation.average_margin_improvement:.6f}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


def _run_container(root: Path) -> tuple[DpoPostTrainingReport, Path]:
    image_digest = _capture_checked(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        cwd=root,
    )
    run_id = planned_run_id(root, image_digest)
    acceptance = root / OUTPUT_RELATIVE / "runs" / run_id / "acceptance.json"
    if acceptance.is_file():
        return verify_dpo_post_training(root, acceptance), acceptance
    if _container_exists(root):
        raise DpoPostTrainingLabError(
            f"DPO lab container already exists: {CONTAINER_NAME}"
        )
    (root / OUTPUT_RELATIVE).mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        CONTAINER_NAME,
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cpus",
        "2",
        "--memory",
        "8g",
        "--pids-limit",
        "512",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--env",
        "PYTHONPATH=/workspace/src",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
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
        "industrial_ops_agent.simulation.dpo_post_training_lab_cli",
        "worker",
        "--repo-root",
        "/workspace",
        "--image-digest",
        image_digest,
    ]
    try:
        subprocess.run(command, cwd=root, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DpoPostTrainingLabError("actual GPU DPO container failed") from exc
    finally:
        if _container_exists(root):
            _remove_owned_container(root)
        if _container_exists(root):
            raise DpoPostTrainingLabError("DPO lab container cleanup failed")
    report = verify_dpo_post_training(root, acceptance)
    return report, acceptance


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
        raise DpoPostTrainingLabError("DPO lab container cleanup failed") from exc


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
        raise DpoPostTrainingLabError(f"command failed: {command[0]}") from exc
    value = completed.stdout.strip()
    if not value:
        raise DpoPostTrainingLabError(f"command returned no output: {command[0]}")
    return value


if __name__ == "__main__":
    run()
