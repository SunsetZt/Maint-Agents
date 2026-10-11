"""Build or verify the low-resource TTS KServe rollout readiness package."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ops_agent.simulation.tts_kserve_rollout import (
    OUTPUT_RELATIVE,
    build_tts_kserve_readiness,
    verify_tts_kserve_readiness,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-tts-kserve-rollout")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--repo-root", type=Path, default=Path.cwd())
    plan.add_argument("--acceptance", type=Path, required=True)
    plan.add_argument("--host-gateway", default="172.19.0.1")
    plan.add_argument("--host-port", type=int, default=18095)
    plan.add_argument("--proxy-image-digest", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "plan":
        path = build_tts_kserve_readiness(
            root,
            acceptance_path=args.acceptance,
            host_gateway=args.host_gateway,
            host_port=args.host_port,
            proxy_image_digest=args.proxy_image_digest,
        )
    else:
        path = root / OUTPUT_RELATIVE / "readiness.json"
    report = verify_tts_kserve_readiness(root, path)
    print(report.status)
    print(report.classification)
    print(f"ACTUAL_ROLLOUT_EXECUTION_COMPLETED={report.runtime.actual_rollout_execution_completed}")
    print(f"FORMAL_MODEL_RELEASE_CREATED={report.model_release.formal_model_release_created}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


if __name__ == "__main__":
    run()
