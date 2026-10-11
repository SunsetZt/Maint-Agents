"""CLI for the additive Reranker v3 verifier erratum."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ops_agent.simulation.reranker_recovery_verifier_erratum import (
    AUTHORITATIVE_LATEST_RELATIVE,
    RerankerRecoveryVerifierErratumReport,
    build_reranker_recovery_verifier_erratum,
    verify_reranker_recovery_verifier_erratum,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-reranker-verifier-erratum")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "build":
        path = build_reranker_recovery_verifier_erratum(root)
        report = verify_reranker_recovery_verifier_erratum(root, path)
    else:
        report = verify_reranker_recovery_verifier_erratum(root)
        path = root / AUTHORITATIVE_LATEST_RELATIVE
    _print_report(report, path)


def _print_report(report: RerankerRecoveryVerifierErratumReport, path: Path) -> None:
    corrected = report.corrected_verification
    print(report.status)
    print(report.classification)
    print(f"RUN_ID={report.run_id}")
    print(f"CANDIDATE_ACCEPTED={corrected.candidate_accepted}")
    print(f"BASELINE_NDCG_AT_3={corrected.baseline_ndcg_at_k:.6f}")
    print(f"CANDIDATE_NDCG_AT_3={corrected.candidate_ndcg_at_k:.6f}")
    print(f"NDCG_AT_3_IMPROVEMENT={corrected.primary_improvement:.6f}")
    print(f"FAILED_HARD_GATES={','.join(corrected.failed_hard_gates)}")
    print(f"FORMAL_GOLD_REPLAYED={report.formal_gold_replayed}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


if __name__ == "__main__":
    run()
