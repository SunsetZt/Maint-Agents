"""Read-only CLI for historical AGT-007 value-evaluation receipts."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ops_agent.simulation.maintenance_planning_value_lab import (
    OUTPUT_RELATIVE,
    verify_maintenance_planning_value_lab,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-maintenance-planning-value-lab")
    parser.add_argument("command", choices=("verify",), nargs="?", default="verify")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=OUTPUT_RELATIVE)
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        relative_output = output.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("maintenance planning value output must stay inside repository") from exc
    report = verify_maintenance_planning_value_lab(root, relative_output)
    print(report.status)
    print(report.classification)
    print(f"SAMPLE_COUNT={report.evaluation.sample_count}")
    print(f"JUDGMENT_COUNT={report.evaluation.judgment_count}")
    print(f"DECISION={report.evaluation.decision}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(output)


if __name__ == "__main__":
    run()
