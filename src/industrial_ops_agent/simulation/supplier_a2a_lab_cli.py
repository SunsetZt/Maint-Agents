"""CLI for the low-resource supplier A2A closed-loop acceptance."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from industrial_ops_agent.simulation.supplier_a2a_lab import (
    OUTPUT_RELATIVE,
    run_supplier_a2a_lab,
    verify_supplier_a2a_lab,
    write_supplier_a2a_lab,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-supplier-a2a-lab")
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=OUTPUT_RELATIVE)
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    output = args.output if args.output.is_absolute() else root / args.output
    if args.command == "run":
        report = asyncio.run(run_supplier_a2a_lab(root))
        path = write_supplier_a2a_lab(report, output)
    else:
        report = verify_supplier_a2a_lab(root, output)
        path = output
    print(report.status)
    print(report.classification)
    print(path)


if __name__ == "__main__":
    run()
