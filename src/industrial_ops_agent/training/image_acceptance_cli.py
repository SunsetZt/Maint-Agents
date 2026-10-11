"""CLI entrypoint for real NVIDIA training-image acceptance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from industrial_ops_agent.training.image_acceptance import (
    run_gpu_image_acceptance,
    write_gpu_acceptance_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-training-image-acceptance")
    parser.add_argument(
        "--image-digest",
        default=os.getenv("IOAP_TRAINING_CONTAINER_DIGEST", ""),
    )
    parser.add_argument(
        "--git-commit",
        default=os.getenv("IOAP_TRAINING_GIT_COMMIT", ""),
    )
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16"),
        default=os.getenv("IOAP_GPU_ACCEPTANCE_PRECISION", "bfloat16"),
    )
    parser.add_argument(
        "--minimum-gpu-memory-gib",
        type=float,
        default=float(os.getenv("IOAP_GPU_ACCEPTANCE_MIN_MEMORY_GIB", "0")),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.minimum_gpu_memory_gib < 0:
        raise SystemExit("minimum-gpu-memory-gib must be non-negative")
    report = run_gpu_image_acceptance(
        image_digest=args.image_digest,
        git_commit=args.git_commit,
        precision=args.precision,
        minimum_gpu_memory_bytes=int(args.minimum_gpu_memory_gib * 1024**3),
    )
    if args.output is not None:
        write_gpu_acceptance_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASSED" else 2


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
