"""CLI for enterprise model import and Release preflight acceptance."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from industrial_ops_agent.enterprise_assets.acceptance import (
    CANONICAL_RECEIPT_PATH,
    execute_enterprise_model_import_acceptance,
    verify_enterprise_model_import_acceptance,
    write_enterprise_model_import_acceptance,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-enterprise-model-import-acceptance")
    parser.add_argument(
        "command",
        choices=("run", "run-local", "verify"),
        nargs="?",
        default="run",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("IOAP_API_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=CANONICAL_RECEIPT_PATH,
    )
    parser.add_argument("--timeout-seconds", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    output = args.output if args.output.is_absolute() else root / args.output
    timeout_seconds = args.timeout_seconds
    if timeout_seconds is None:
        timeout_seconds = 300.0 if args.command == "run-local" else 30.0
    if args.command == "verify":
        report = verify_enterprise_model_import_acceptance(output)
        created = False
    elif args.command == "run-local":
        from industrial_ops_agent.enterprise_assets.local_acceptance import (
            execute_local_enterprise_model_import_acceptance,
        )

        report = execute_local_enterprise_model_import_acceptance(
            root,
            timeout_seconds=timeout_seconds,
        )
        report, created = write_enterprise_model_import_acceptance(output, report)
    else:
        report = execute_enterprise_model_import_acceptance(
            args.base_url,
            os.getenv("IOAP_ENTERPRISE_MODEL_IMPORT_ACCEPTANCE_TOKEN", ""),
            timeout_seconds=timeout_seconds,
        )
        report, created = write_enterprise_model_import_acceptance(output, report)
    print("CREATED" if created else "VERIFIED_EXISTING")
    print(report.status)
    print(report.classification)
    print(f"EXECUTION_MODE={report.execution_mode}")
    print(f"COMPONENTS={len(report.components)}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(output)
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
