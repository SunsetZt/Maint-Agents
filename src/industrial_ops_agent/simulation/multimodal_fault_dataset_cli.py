"""CLI for the project-generated M7 multimodal fault dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from industrial_ops_agent.simulation.multimodal_fault_dataset import (
    DEFAULT_CASES_PER_SPLIT,
    DEFAULT_SEED,
    MultimodalFaultDatasetError,
    build_multimodal_fault_dataset,
    verify_multimodal_fault_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-multimodal-dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build or reuse one immutable simulated dataset")
    build.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m7-multimodal-fault-dataset"),
    )
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--cases-per-split", type=int, default=DEFAULT_CASES_PER_SPLIT)
    build.add_argument("--train-cases", type=int)
    build.add_argument("--validation-cases", type=int)
    build.add_argument("--evaluation-cases", type=int)
    verify = commands.add_parser("verify", help="verify all files, bindings and leakage gates")
    verify.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m7-multimodal-fault-dataset"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_multimodal_fault_dataset(
                args.output,
                seed=args.seed,
                cases_per_split=args.cases_per_split,
                train_cases=args.train_cases,
                validation_cases=args.validation_cases,
                evaluation_cases=args.evaluation_cases,
            )
        else:
            result = verify_multimodal_fault_dataset(args.output)
    except (OSError, MultimodalFaultDatasetError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
