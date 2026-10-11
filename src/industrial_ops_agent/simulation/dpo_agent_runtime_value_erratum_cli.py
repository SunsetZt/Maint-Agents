"""CLI for building and verifying the immutable DPO runtime evaluator erratum."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ops_agent.simulation.dpo_agent_runtime_value_erratum import (
    AUTHORITATIVE_LATEST_RELATIVE,
    build_dpo_agent_runtime_erratum,
    verify_dpo_agent_runtime_erratum,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-dpo-agent-runtime-erratum")
    parser.add_argument("command", choices=("build", "verify"), nargs="?", default="build")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "build":
        path = build_dpo_agent_runtime_erratum(root)
        report = verify_dpo_agent_runtime_erratum(root, path)
    else:
        report = verify_dpo_agent_runtime_erratum(root)
        path = root / AUTHORITATIVE_LATEST_RELATIVE
    print(report.status)
    print(report.classification)
    print(f"CANDIDATE_ACCEPTED={str(report.candidate_accepted).lower()}")
    print(
        "CORRECTED_FAILED_HARD_GATES="
        f"{','.join(report.corrected_evaluation.corrected_failed_hard_gates)}"
    )
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


if __name__ == "__main__":
    run()
