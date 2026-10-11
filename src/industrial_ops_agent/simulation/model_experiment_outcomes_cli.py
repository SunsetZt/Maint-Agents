"""CLI for governed actual-GPU model experiment outcome receipts."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ops_agent.simulation.model_experiment_outcomes import (
    OUTPUT_RELATIVE,
    build_model_experiment_outcomes,
    verify_model_experiment_outcomes,
    write_model_experiment_outcomes,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-model-experiment-outcomes")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    verify = commands.add_parser("verify")
    for command in (build, verify):
        command.add_argument("--repo-root", type=Path, default=Path.cwd())
        command.add_argument("--output", type=Path, default=OUTPUT_RELATIVE)
    build.add_argument("--tts-rejection", type=Path, required=True)
    build.add_argument("--tts-retirement", type=Path, required=True)
    build.add_argument("--tts-strong-asr-acceptance", type=Path)
    build.add_argument("--retrieval-rejection", type=Path, required=True)
    build.add_argument("--retrieval-retirement", type=Path, required=True)
    build.add_argument("--embedding-recovery-acceptance", type=Path)
    build.add_argument("--embedding-calibrated-acceptance", type=Path)
    build.add_argument("--reranker-recovery-erratum", type=Path)
    build.add_argument("--reranker-calibrated-acceptance", type=Path)
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    output = args.output if args.output.is_absolute() else root / args.output
    if args.command == "build":
        report = build_model_experiment_outcomes(
            root,
            tts_rejection_path=args.tts_rejection,
            tts_retirement_path=args.tts_retirement,
            tts_strong_asr_acceptance_path=args.tts_strong_asr_acceptance,
            retrieval_rejection_path=args.retrieval_rejection,
            retrieval_retirement_path=args.retrieval_retirement,
            embedding_recovery_acceptance_path=args.embedding_recovery_acceptance,
            embedding_calibrated_acceptance_path=(
                args.embedding_calibrated_acceptance
            ),
            reranker_recovery_erratum_path=args.reranker_recovery_erratum,
            reranker_calibrated_acceptance_path=(args.reranker_calibrated_acceptance),
        )
        path = write_model_experiment_outcomes(report, output)
    else:
        report = verify_model_experiment_outcomes(root, output)
        path = output
    print(report.status)
    print(report.classification)
    print(f"REJECTED_METHOD_COUNT={report.rejected_method_count}")
    print(f"ACTIVE_CANDIDATE_COUNT={report.active_candidate_count}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


if __name__ == "__main__":
    run()
