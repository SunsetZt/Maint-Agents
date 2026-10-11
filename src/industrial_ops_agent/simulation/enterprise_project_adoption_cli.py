"""CLI for the project-wide authoritative enterprise staging adoption."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ops_agent.simulation.enterprise_project_adoption import (
    adopt_all_project_assets_as_enterprise,
    verify_enterprise_project_adoption,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-enterprise-project-adoption")
    parser.add_argument("command", choices=("adopt", "verify"), nargs="?", default="adopt")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=Path("artifacts/enterprise-project-adoption/acceptance.json"),
    )
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    acceptance = args.acceptance if args.acceptance.is_absolute() else root / args.acceptance
    if args.command == "adopt":
        report, created = adopt_all_project_assets_as_enterprise(
            root,
            output_path=acceptance,
        )
        print("CREATED" if created else "VERIFIED_EXISTING")
    else:
        report = verify_enterprise_project_adoption(root, acceptance)
    print(report.status)
    print(report.classification)
    print(f"ACTIVE_ASSETS={len(report.active_assets)}")
    print(f"EXPERIMENT_HISTORY={len(report.experiment_history)}")
    print(acceptance)


if __name__ == "__main__":
    run()
