#!/usr/bin/env python3
"""Create the normalized reports and signed-statement payload used by M6 verification."""

from __future__ import annotations

import argparse
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

SEVERITY_RANK = {
    "UNKNOWN": 0,
    "NEGLIGIBLE": 1,
    "LOW": 2,
    "MEDIUM": 3,
    "HIGH": 4,
    "CRITICAL": 5,
}
DEFAULT_DENIED_LICENSES = ("AGPL-3.0-only", "GPL-3.0-only", "SSPL-1.0")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-repository", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--model-artifact-digest", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--vulnerability-report", type=Path, required=True)
    parser.add_argument("--license-report", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scanner", default="anchore-grype/scan-action@v7.4.0")
    parser.add_argument(
        "--denied-license", action="append", default=list(DEFAULT_DENIED_LICENSES)
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _digest(args.image_digest, "image-digest")
    _digest(args.model_artifact_digest, "model-artifact-digest")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        raise SystemExit("source-revision must be a full 40-character Git commit")
    sbom = _object(args.sbom)
    if not str(sbom.get("spdxVersion", "")).startswith("SPDX-2."):
        raise SystemExit("SBOM must be SPDX JSON")
    vulnerability_report = _object(args.vulnerability_report)
    maximum_severity = _maximum_severity(vulnerability_report)
    vulnerability_status = (
        "PASSED"
        if SEVERITY_RANK[maximum_severity] <= SEVERITY_RANK["MEDIUM"]
        else "FAILED"
    )
    denied_licenses = _denied_licenses(sbom, set(args.denied_license))
    _write_json(
        args.license_report,
        {
            "schema_version": "industrial-ops-license-report/v1",
            "status": "APPROVED" if not denied_licenses else "REJECTED",
            "denied_licenses": denied_licenses,
            "package_count": len(sbom.get("packages", [])),
        },
    )
    _write_json(
        args.provenance,
        {
            "buildDefinition": {
                "buildType": "https://github.com/industrial-ops/model-runtime/v1",
                "externalParameters": {"dockerfile": "docker/model-runtime.Dockerfile"},
                "internalParameters": {},
                "resolvedDependencies": [
                    {
                        "uri": f"git+{args.source_repository}",
                        "digest": {"gitCommit": args.source_revision},
                    }
                ],
            },
            "runDetails": {
                "builder": {
                    "id": f"{args.source_repository}/.github/workflows/supply-chain.yml"
                },
                "metadata": {"invocationId": args.source_revision},
            },
        },
    )
    statement = {
        "schema_version": "industrial-ops-supply-chain/v1",
        "image": {"repository": args.image_repository, "digest": args.image_digest},
        "model": {"artifact_digest": args.model_artifact_digest},
        "source": {
            "repository": args.source_repository,
            "revision": args.source_revision,
        },
        "artifacts": {
            "sbom": {"digest": _file_digest(args.sbom), "format": "spdx-json"},
            "vulnerability_scan": {
                "digest": _file_digest(args.vulnerability_report),
                "status": vulnerability_status,
                "maximum_severity": maximum_severity,
                "scanner": args.scanner,
            },
            "license_report": {
                "digest": _file_digest(args.license_report),
                "status": "APPROVED" if not denied_licenses else "REJECTED",
            },
            "provenance": {"digest": _file_digest(args.provenance)},
        },
    }
    _write_json(args.output, statement)
    print(json.dumps(statement, ensure_ascii=False, sort_keys=True))
    return 0


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"JSON file must contain an object: {path}")
    return value


def _maximum_severity(report: dict[str, Any]) -> str:
    matches = report.get("matches")
    if not isinstance(matches, list):
        raise SystemExit("vulnerability report must be Grype JSON")
    maximum = "UNKNOWN"
    for match in matches:
        if not isinstance(match, dict) or not isinstance(match.get("vulnerability"), dict):
            raise SystemExit("vulnerability report must be Grype JSON")
        severity = str(match["vulnerability"].get("severity", "UNKNOWN")).upper()
        if severity not in SEVERITY_RANK:
            severity = "UNKNOWN"
        if SEVERITY_RANK[severity] > SEVERITY_RANK[maximum]:
            maximum = severity
    return maximum


def _denied_licenses(sbom: dict[str, Any], denied: set[str]) -> list[str]:
    found: set[str] = set()
    packages = sbom.get("packages", [])
    if not isinstance(packages, list):
        raise SystemExit("SPDX packages must be an array")
    for package in packages:
        if not isinstance(package, dict):
            raise SystemExit("SPDX package entry must be an object")
        expression = " ".join(
            str(package.get(field, ""))
            for field in ("licenseDeclared", "licenseConcluded")
        )
        for license_id in denied:
            pattern = rf"(?<![A-Za-z0-9.-]){re.escape(license_id)}(?![A-Za-z0-9.-])"
            if re.search(pattern, expression):
                found.add(license_id)
    return sorted(found)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_digest(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def _digest(value: str, field: str) -> None:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise SystemExit(f"{field} must be a sha256 digest")


if __name__ == "__main__":
    raise SystemExit(main())
