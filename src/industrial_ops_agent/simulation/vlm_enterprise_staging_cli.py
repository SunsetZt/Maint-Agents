"""CLI for adopting and verifying the project-authorized enterprise VLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ops_agent.simulation.vlm_enterprise_staging import (
    adopt_vlm_for_enterprise_staging,
    verify_enterprise_vlm_staging_adoption,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-enterprise-vlm-staging")
    parser.add_argument("command", choices=("adopt", "verify"), nargs="?", default="adopt")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/m7-vlm-checkpoint-continuation-rescored-lab"),
    )
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=Path("artifacts/m7-vlm-enterprise-staging/acceptance.json"),
    )
    parser.add_argument("--model-alias", default="industrial-diagnosis")
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    acceptance = args.acceptance if args.acceptance.is_absolute() else root / args.acceptance
    if args.command == "adopt":
        report, created = adopt_vlm_for_enterprise_staging(
            root,
            source_dir=args.source_dir,
            output_path=acceptance,
            model_alias=args.model_alias,
        )
        print("CREATED" if created else "VERIFIED_EXISTING")
    else:
        report = verify_enterprise_vlm_staging_adoption(root, acceptance)
    print(report.status)
    print(report.classification)
    print(report.runtime.model_alias)
    print(acceptance)


if __name__ == "__main__":
    run()
