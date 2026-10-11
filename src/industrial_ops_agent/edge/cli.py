"""Real command-line entrypoint for governed FIELD edge diagnosis."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from industrial_ops_agent.edge.runner import FieldEdgeRunnerError, run_edge_diagnosis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m industrial_ops_agent.edge.cli",
        description="Run governed FIELD edge diagnosis with a signed package and llama.cpp.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="verify a signed package and create one UNATTESTED review result",
    )
    run.add_argument("--pack", required=True, type=Path)
    run.add_argument("--trusted-public-key", required=True, type=Path)
    run.add_argument("--trusted-key-id", required=True)
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--llama-cli", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command != "run":  # pragma: no cover - argparse closes this branch.
        return 2
    try:
        run_edge_diagnosis(
            pack_path=arguments.pack,
            trusted_public_key_path=arguments.trusted_public_key,
            trusted_key_id=arguments.trusted_key_id,
            model_path=arguments.model,
            llama_cli_path=arguments.llama_cli,
            output_path=arguments.output,
        )
    except (FieldEdgeRunnerError, ValidationError, OSError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, FieldEdgeRunnerError) else "field_edge_run_failed"
        print(f"governed FIELD edge diagnosis failed: {reason}", file=sys.stderr)
        return 2
    print(f"governed FIELD edge diagnosis result: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
