"""CLI for low-resource enterprise candidate rollout acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from industrial_ops_agent.simulation.enterprise_candidate_rollout import (
    DEFAULT_OUTPUT,
    run_enterprise_candidate_rollout,
    verify_enterprise_candidate_rollout,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-enterprise-candidate-rollout")
    parser.add_argument("action", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "verify":
        report = verify_enterprise_candidate_rollout(args.repo_root, args.output)
        created = False
    else:
        report, created = run_enterprise_candidate_rollout(
            args.repo_root,
            output_path=args.output,
        )
    print(
        json.dumps(
            {
                "status": report.status,
                "classification": report.classification,
                "components": [item.component for item in report.components],
                "evidence_chain_sha256": report.evidence_chain_sha256,
                "created": created,
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
