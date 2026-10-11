"""CLI for the low-resource project enterprise assurance lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ops_agent.simulation.project_assurance_lab import (
    OUTPUT_RELATIVE,
    run_project_assurance_lab,
    verify_project_assurance_lab,
    write_project_assurance_lab,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-project-assurance-lab")
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=OUTPUT_RELATIVE)
    parser.add_argument("--git-commit")
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        relative_output = output.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("project assurance output must stay inside the repository") from exc
    if args.command == "run":
        report = run_project_assurance_lab(
            root,
            git_commit=args.git_commit,
        )
        write_project_assurance_lab(report, output)
        report = verify_project_assurance_lab(root, relative_output)
    else:
        report = verify_project_assurance_lab(root, relative_output)
    print(report.status)
    print(report.classification)
    print(f"SECURITY_SCENARIOS={report.security.scenario_count}")
    print(f"RECOVERY_COMPONENTS={report.recovery.evidence_count}")
    print(f"HEALTHY_SLOS={report.operations.healthy_slo_count}")
    print(f"STAGING_SCENARIOS={report.staging.scenario_count}")
    print(f"SIGNOFFS={report.signoff.signoff_count}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(output)


if __name__ == "__main__":
    run()
