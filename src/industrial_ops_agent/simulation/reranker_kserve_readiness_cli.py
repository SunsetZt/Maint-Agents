"""Build and verify low-resource Reranker KServe readiness evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from industrial_ops_agent.simulation.reranker_kserve_readiness import (
    ACCEPTANCE_RELATIVE,
    IMAGE_HELP_RELATIVE,
    IMAGE_INSPECT_RELATIVE,
    RUNTIME_IMAGE_REPOSITORY,
    RUNTIME_IMAGE_TAG,
    RerankerKServeReadinessError,
    build_reranker_kserve_readiness,
    verify_reranker_kserve_readiness,
)


def _capture_image_evidence(root: Path) -> tuple[Path, Path]:
    image_ref = f"{RUNTIME_IMAGE_REPOSITORY}:{RUNTIME_IMAGE_TAG}"
    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", image_ref],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        help_result = subprocess.run(
            ["docker", "run", "--rm", image_ref, "--help"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RerankerKServeReadinessError(
            "unable to inspect the pinned Reranker runtime image"
        ) from exc
    output = root / IMAGE_INSPECT_RELATIVE.parent
    output.mkdir(parents=True, exist_ok=True)
    inspect_path = root / IMAGE_INSPECT_RELATIVE
    help_path = root / IMAGE_HELP_RELATIVE
    decoded = json.loads(inspect.stdout)
    inspect_path.write_text(
        json.dumps(decoded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    help_path.write_text(help_result.stdout, encoding="utf-8")
    return inspect_path, help_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reranker KServe release readiness")
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=ACCEPTANCE_RELATIVE)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "build":
        inspect_path, help_path = _capture_image_evidence(root)
        report = build_reranker_kserve_readiness(
            root,
            image_inspect_path=inspect_path,
            entrypoint_help_path=help_path,
            output_path=args.output,
        )
    else:
        report = verify_reranker_kserve_readiness(root, args.output)
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
