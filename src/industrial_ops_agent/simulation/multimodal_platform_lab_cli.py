"""CLI for the M7 simulated multimodal platform import lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from industrial_ops_agent.simulation.multimodal_platform_lab import (
    SimulatedMultimodalPlatformLabError,
    build_simulated_multimodal_platform_lab,
    verify_simulated_multimodal_platform_lab,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-multimodal-platform-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--repo-root", type=Path, default=Path.cwd())
    build.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/m7-multimodal-fault-dataset"),
    )
    build.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m7-multimodal-platform-lab"),
    )
    verify = commands.add_parser("verify")
    verify.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m7-multimodal-platform-lab"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            report, created = build_simulated_multimodal_platform_lab(
                args.repo_root,
                source_dir=args.source,
                output_dir=args.output,
            )
        else:
            report = verify_simulated_multimodal_platform_lab(args.output)
            created = False
    except (OSError, SimulatedMultimodalPlatformLabError, ValueError) as exc:
        parser.error(str(exc))
    output = report.model_dump(mode="json")
    output["created"] = created
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
