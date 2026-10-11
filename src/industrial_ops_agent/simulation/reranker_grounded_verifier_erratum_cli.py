"""CLI for the additive grounded Reranker v4 verifier erratum."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ops_agent.simulation.reranker_grounded_verifier_erratum import (
    build_reranker_grounded_verifier_erratum,
    verify_reranker_grounded_verifier_erratum,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="industrial-ops-reranker-grounded-verifier-erratum"
    )
    parser.add_argument(
        "command", choices=("build", "verify"), nargs="?", default="build"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "build":
        path = build_reranker_grounded_verifier_erratum(root)
        report = verify_reranker_grounded_verifier_erratum(root, path)
    else:
        report = verify_reranker_grounded_verifier_erratum(root)
        path = (
            root / "artifacts/m7-reranker-grounded-value-lab/authoritative-latest.json"
        )
    print(report.status)
    print(report.classification)
    print(f"RUN_ID={report.run_id}")
    print(f"CANDIDATE_ACCEPTED={report.corrected_verification.candidate_accepted}")
    print(
        f"FAILED_HARD_GATES={','.join(report.corrected_verification.failed_hard_gates)}"
    )
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


if __name__ == "__main__":
    run()
