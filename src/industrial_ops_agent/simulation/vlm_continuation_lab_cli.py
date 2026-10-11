from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from industrial_ops_agent.simulation.vlm_continuation_lab import (
    build_simulated_vlm_continuation_lab,
    preflight_simulated_vlm_continuation_lab,
    rescore_simulated_vlm_continuation_lab,
    verify_simulated_vlm_continuation_lab,
)

DEFAULT_TRAINING = Path("artifacts/m7-vlm-curriculum-training-lab")
DEFAULT_RESCORE = Path("artifacts/m7-vlm-curriculum-rescored-lab")
DEFAULT_OUTPUT = Path("artifacts/m7-vlm-checkpoint-continuation-lab")
DEFAULT_RESCORED_OUTPUT = Path("artifacts/m7-vlm-checkpoint-continuation-rescored-lab")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-simulated-vlm-continuation-lab")
    subcommands = parser.add_subparsers(dest="command", required=True)
    preflight = subcommands.add_parser("preflight")
    preflight.add_argument("--source-training", type=Path, default=DEFAULT_TRAINING)
    preflight.add_argument("--source-rescore", type=Path, default=DEFAULT_RESCORE)
    build = subcommands.add_parser("build")
    build.add_argument("--repo-root", type=Path, default=Path.cwd())
    build.add_argument("--source-training", type=Path, default=DEFAULT_TRAINING)
    build.add_argument("--source-rescore", type=Path, default=DEFAULT_RESCORE)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--additional-steps", type=int, default=20)
    build.add_argument("--repeat-factor", type=int, default=3)
    rescore = subcommands.add_parser("rescore")
    rescore.add_argument("--source", type=Path, default=DEFAULT_OUTPUT)
    rescore.add_argument("--output", type=Path, default=DEFAULT_RESCORED_OUTPUT)
    verify = subcommands.add_parser("verify")
    verify.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload: dict[str, Any]
    if args.command == "preflight":
        payload = preflight_simulated_vlm_continuation_lab(
            args.source_training,
            args.source_rescore,
        )
        created = False
    elif args.command == "build":
        report, created = build_simulated_vlm_continuation_lab(
            args.repo_root,
            source_training_dir=args.source_training,
            source_rescore_dir=args.source_rescore,
            output_dir=args.output,
            additional_steps=args.additional_steps,
            repeat_factor=args.repeat_factor,
        )
        payload = report.model_dump(mode="json")
    elif args.command == "rescore":
        report, created = rescore_simulated_vlm_continuation_lab(
            args.source,
            output_dir=args.output,
        )
        payload = report.model_dump(mode="json")
    else:
        report = verify_simulated_vlm_continuation_lab(args.output)
        payload = report.model_dump(mode="json")
        created = False
    print(json.dumps({"created": created, **payload}, ensure_ascii=False, sort_keys=True))
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
