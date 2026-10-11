from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from industrial_ops_agent.simulation.vlm_training_lab import (
    build_simulated_vlm_training_lab,
    preflight_simulated_vlm_training_lab,
    rescore_simulated_vlm_training_lab,
    verify_simulated_vlm_training_lab,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-simulated-vlm-training-lab")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("preflight")
    build = subcommands.add_parser("build")
    build.add_argument("--repo-root", type=Path, default=Path.cwd())
    build.add_argument(
        "--platform",
        type=Path,
        default=Path("artifacts/m7-multimodal-platform-lab"),
    )
    build.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m7-simulated-vlm-training-lab"),
    )
    build.add_argument("--max-steps", type=int, default=12)
    verify = subcommands.add_parser("verify")
    verify.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m7-simulated-vlm-training-lab"),
    )
    rescore = subcommands.add_parser("rescore")
    rescore.add_argument("--source", type=Path, required=True)
    rescore.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload: dict[str, Any]
    if args.command == "preflight":
        payload = preflight_simulated_vlm_training_lab()
        created = False
    elif args.command == "build":
        report, created = build_simulated_vlm_training_lab(
            args.repo_root,
            platform_dir=args.platform,
            output_dir=args.output,
            max_steps=args.max_steps,
        )
        payload = report.model_dump(mode="json")
    elif args.command == "verify":
        report = verify_simulated_vlm_training_lab(args.output)
        payload = report.model_dump(mode="json")
        created = False
    else:
        report, created = rescore_simulated_vlm_training_lab(
            args.source,
            output_dir=args.output,
        )
        payload = report.model_dump(mode="json")
    print(json.dumps({"created": created, **payload}, ensure_ascii=False, sort_keys=True))
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
