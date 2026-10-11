"""CLI for the rejected Embedding v2 experiment acceptance receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ops_agent.simulation.embedding_recovery_outcome import (
    OUTPUT_RELATIVE,
    build_embedding_recovery_rejection_acceptance,
    verify_embedding_recovery_rejection_acceptance,
    write_embedding_recovery_rejection_acceptance,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-embedding-recovery-outcome")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    verify = commands.add_parser("verify")
    for command in (build, verify):
        command.add_argument("--repo-root", type=Path, default=Path.cwd())
        command.add_argument("--output", type=Path, default=OUTPUT_RELATIVE)
    build.add_argument("--rejection", type=Path, required=True)
    build.add_argument("--retirement", type=Path, required=True)
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    output = args.output if args.output.is_absolute() else root / args.output
    if args.command == "build":
        report = build_embedding_recovery_rejection_acceptance(
            root,
            rejection_path=args.rejection,
            retirement_path=args.retirement,
        )
        path = write_embedding_recovery_rejection_acceptance(report, output)
    else:
        report = verify_embedding_recovery_rejection_acceptance(root, output)
        path = output
    print(report.status)
    print(report.classification)
    print(f"RUN_ID={report.run_id}")
    print(f"BASELINE_MRR={report.evaluation.baseline_mrr:.6f}")
    print(f"CANDIDATE_MRR={report.evaluation.candidate_mrr:.6f}")
    print(f"FAILED_GATES={','.join(report.failed_gates)}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


if __name__ == "__main__":
    run()
