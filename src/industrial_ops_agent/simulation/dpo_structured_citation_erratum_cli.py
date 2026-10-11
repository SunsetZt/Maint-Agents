"""CLI for the additive DPO v3 citation-fixture erratum."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ops_agent.simulation.dpo_structured_citation_erratum import (
    build_dpo_structured_citation_erratum,
    verify_dpo_structured_citation_erratum,
)


def run() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve(strict=True)
    path = (
        build_dpo_structured_citation_erratum(root)
        if args.command == "run"
        else None
    )
    report = verify_dpo_structured_citation_erratum(root, path)
    print(report.status)
    print(report.classification)
    print(f"CANDIDATE_ACCEPTED={str(report.candidate_accepted).lower()}")
    print(
        "CORRECTED_CITATION_PASSES="
        f"{report.corrected_citation_boundary.corrected_model_pass_count}/"
        f"{report.corrected_citation_boundary.case_count}"
    )
    print(f"FAILED_HARD_GATES={','.join(report.corrected_failed_hard_gates) or 'none'}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")


if __name__ == "__main__":
    run()
