"""CLI for the actual-GPU verifier-guided TTS enterprise-value lab v3."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from industrial_ops_agent.simulation.tts_final_value_lab import (
    CONTAINER_NAME,
    IMAGE,
    OUTPUT_RELATIVE,
    TtsCalibratedValueLabError,
    TtsCalibratedValueReport,
    assert_gold_available,
    execute_validation_probe,
    execute_worker,
    planned_run_id,
    preflight_calibrated_tts,
    verify_calibrated_tts_value,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-tts-final-value-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "preflight",
        "validate",
        "run",
        "verify",
        "validation-worker",
        "worker",
    ):
        command = commands.add_parser(name)
        command.add_argument("--repo-root", type=Path, default=Path.cwd())
        if name in {"validation-worker", "worker"}:
            command.add_argument("--image-digest", required=True)
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "preflight":
        result = preflight_calibrated_tts(root)
        print(result["status"])
        print(result["classification"])
        print(f"SOURCE_V2_ACTUAL_GPU_TRAINING={result['source_v2_actual_gpu_training']}")
        print(f"CANDIDATE_POOL_SIZE={result['candidate_pool_size']}")
        print(f"GOLD_CONSUMED={str(result['gold_consumed']).lower()}")
        return
    if args.command == "validation-worker":
        path = execute_validation_probe(root, image_digest=args.image_digest)
        result = json.loads(path.read_text(encoding="utf-8"))
        print(result["status"])
        print("FORMAL_GOLD_CONSUMED=false")
        print(path)
        return
    if args.command == "validate":
        path = _run_validation_container(root)
        result = json.loads(path.read_text(encoding="utf-8"))
        print(result["status"])
        print("FORMAL_GOLD_CONSUMED=false")
        summary = result.get("summary")
        if isinstance(summary, dict):
            print(
                "VALIDATION_VOICE_SIMILARITY_IMPROVEMENT="
                f"{float(summary['voice_similarity_improvement']):.9f}"
            )
            print(
                "VALIDATION_SAFETY_PHRASE_COMPLETENESS="
                f"{float(summary['candidate_safety_phrase_completeness']):.6f}"
            )
        print(path)
        return
    if args.command == "worker":
        path = execute_worker(root, image_digest=args.image_digest)
        report = verify_calibrated_tts_value(root, path)
    elif args.command == "verify":
        report = verify_calibrated_tts_value(root)
        path = root / OUTPUT_RELATIVE / "latest-outcome.json"
    else:
        report, path = _run_container(root)
    print(report.status)
    print(report.classification)
    print(f"ACTUAL_GPU_INFERENCE={report.runtime.actual_gpu_inference}")
    print(
        "VALIDATION_VOICE_SIMILARITY_IMPROVEMENT="
        f"{report.validation.voice_similarity_improvement:.9f}"
    )
    print(
        "GOLD_VOICE_SIMILARITY_IMPROVEMENT="
        f"{report.evaluation.voice_similarity_improvement:.9f}"
    )
    print(
        "GOLD_SAFETY_PHRASE_COMPLETENESS="
        f"{report.evaluation.candidate_safety_phrase_completeness:.6f}"
    )
    print(f"CANDIDATE_ACCEPTED={str(report.candidate_accepted).lower()}")
    print(f"FAILED_HARD_GATES={','.join(report.failed_hard_gates) or 'none'}")
    print(f"GPU_PEAK_RESERVED_BYTES={report.runtime.peak_gpu_memory_reserved_bytes}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


def _run_container(root: Path) -> tuple[TtsCalibratedValueReport, Path]:
    image_digest = _capture_checked(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], cwd=root
    )
    run_id = planned_run_id(root, image_digest)
    acceptance = root / OUTPUT_RELATIVE / "runs" / run_id / "acceptance.json"
    rejection = root / OUTPUT_RELATIVE / "rejections" / run_id / "rejection.json"
    for existing in (acceptance, rejection):
        if existing.is_file():
            return verify_calibrated_tts_value(root, existing), existing
    assert_gold_available(root)
    if _container_exists(root):
        raise TtsCalibratedValueLabError(f"TTS lab container already exists: {CONTAINER_NAME}")
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
        "HOME=/tmp/home",
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
        "--env",
        "IOAP_DIRECT_PYTHON_ENTRYPOINT=1",
        "--mount",
        f"type=bind,src={root},dst=/workspace",
        "--workdir",
        "/workspace",
        "--entrypoint",
        "python",
        IMAGE,
        "-B",
        "-m",
        "industrial_ops_agent.simulation.tts_final_value_lab_cli",
        "worker",
        "--repo-root",
        "/workspace",
        "--image-digest",
        image_digest,
    ]
    try:
        subprocess.run(command, cwd=root, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TtsCalibratedValueLabError("actual GPU TTS final container failed") from exc
    finally:
        if _container_exists(root):
            _remove_owned_container(root)
        if _container_exists(root):
            raise TtsCalibratedValueLabError("TTS final container cleanup failed")
    outcome = acceptance if acceptance.is_file() else rejection
    return verify_calibrated_tts_value(root, outcome), outcome


def _run_validation_container(root: Path) -> Path:
    image_digest = _capture_checked(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], cwd=root
    )
    run_id = planned_run_id(root, image_digest)
    output = (
        root / OUTPUT_RELATIVE / "development" / f"{run_id}-validation-probe.json"
    )
    assert_gold_available(root)
    if _container_exists(root):
        raise TtsCalibratedValueLabError(
            f"TTS lab container already exists: {CONTAINER_NAME}"
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
        "HOME=/tmp/home",
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
        "--env",
        "IOAP_DIRECT_PYTHON_ENTRYPOINT=1",
        "--mount",
        f"type=bind,src={root},dst=/workspace",
        "--workdir",
        "/workspace",
        "--entrypoint",
        "python",
        IMAGE,
        "-B",
        "-m",
        "industrial_ops_agent.simulation.tts_final_value_lab_cli",
        "validation-worker",
        "--repo-root",
        "/workspace",
        "--image-digest",
        image_digest,
    ]
    case_filter = os.environ.get("IOAP_TTS_VALIDATION_CASE", "").strip()
    if case_filter:
        image_index = command.index(IMAGE)
        command[image_index:image_index] = [
            "--env",
            f"IOAP_TTS_VALIDATION_CASE={case_filter}",
        ]
    try:
        subprocess.run(command, cwd=root, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TtsCalibratedValueLabError(
            "actual GPU TTS validation container failed"
        ) from exc
    finally:
        if _container_exists(root):
            _remove_owned_container(root)
        if _container_exists(root):
            raise TtsCalibratedValueLabError(
                "TTS validation container cleanup failed"
            )
    if not output.is_file():
        raise TtsCalibratedValueLabError("TTS validation evidence is missing")
    return output


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
        raise TtsCalibratedValueLabError("TTS final container cleanup failed") from exc


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
        raise TtsCalibratedValueLabError(f"command failed: {command[0]}") from exc
    value = completed.stdout.strip()
    if not value:
        raise TtsCalibratedValueLabError(f"command returned empty output: {command[0]}")
    return value


if __name__ == "__main__":
    run()
