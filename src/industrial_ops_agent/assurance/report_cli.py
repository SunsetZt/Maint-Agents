"""Submit a bounded isolated red-team report through the governed API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from industrial_ops_agent.assurance.service import (
    AttackScenario,
    ExerciseResultInput,
    ObservedOutcome,
    exercise_report_digest,
)

MAX_REPORT_BYTES = 256 * 1024


def run() -> None:
    parser = argparse.ArgumentParser(description="Submit an isolated security exercise report.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--token-url", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret-file", required=True, type=Path)
    parser.add_argument("--ca-file", type=Path)
    args = parser.parse_args()
    report = _load_report(args.report)
    secret = _read_secret(args.client_secret_file)
    _validate_endpoint(args.api_url)
    _validate_endpoint(args.token_url)
    exercise_id = _required_string(report, "exercise_id")
    verifier_version = _required_string(report, "verifier_version")
    expected_version = _required_positive_integer(report, "expected_version")
    results = _results(report.get("results"))
    body = {
        "verifier_version": verifier_version,
        "report_digest": exercise_report_digest(
            exercise_id,
            verifier_version,
            results,
        ),
        "results": [
            {
                "scenario_id": item.scenario_id.value,
                "observed_outcome": item.observed_outcome.value,
                "attempt_count": item.attempt_count,
                "unauthorized_read_count": item.unauthorized_read_count,
                "unauthorized_side_effect_count": item.unauthorized_side_effect_count,
                "sensitive_output_count": item.sensitive_output_count,
                "evidence_ref": item.evidence_ref,
                "artifact_digest": item.artifact_digest,
            }
            for item in results
        ],
    }
    verify: bool | str = str(args.ca_file) if args.ca_file is not None else True
    with httpx.Client(timeout=30.0, verify=verify) as client:
        token_response = client.post(
            args.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": args.client_id,
                "client_secret": secret,
            },
            headers={"Accept": "application/json"},
        )
        token_response.raise_for_status()
        access_token = token_response.json().get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError("OIDC token response did not contain an access token")
        response = client.post(
            f"{args.api_url.rstrip('/')}/api/v1/security-exercises/{exercise_id}/complete",
            json=body,
            headers={
                "Authorization": f"Bearer {access_token}",
                "If-Match": str(expected_version),
                "X-Request-ID": f"security-exercise-{uuid4().hex}",
            },
        )
        response.raise_for_status()
        payload = response.json()
    result = payload.get("data", {})
    accepted_id = result.get("exercise_id")
    status = result.get("status")
    if accepted_id != exercise_id or status not in {"PASSED", "FAILED"}:
        raise RuntimeError("security exercise API returned an invalid completion receipt")
    print(f"Security exercise accepted: {accepted_id} status={status}")


def _load_report(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_REPORT_BYTES:
        raise ValueError("security exercise report size is invalid")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("security exercise report must be a JSON object")
    if set(payload) != {"exercise_id", "expected_version", "verifier_version", "results"}:
        raise ValueError("security exercise report fields are invalid")
    return payload


def _results(value: Any) -> tuple[ExerciseResultInput, ...]:
    if not isinstance(value, list) or not value or len(value) > 12:
        raise ValueError("security exercise report results are invalid")
    parsed: list[ExerciseResultInput] = []
    expected_fields = {
        "scenario_id",
        "observed_outcome",
        "attempt_count",
        "unauthorized_read_count",
        "unauthorized_side_effect_count",
        "sensitive_output_count",
        "evidence_ref",
        "artifact_digest",
    }
    for item in value:
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError("security exercise result fields are invalid")
        parsed.append(
            ExerciseResultInput(
                scenario_id=AttackScenario(_required_string(item, "scenario_id")),
                observed_outcome=ObservedOutcome(_required_string(item, "observed_outcome")),
                attempt_count=_required_nonnegative_integer(item, "attempt_count"),
                unauthorized_read_count=_required_nonnegative_integer(
                    item, "unauthorized_read_count"
                ),
                unauthorized_side_effect_count=_required_nonnegative_integer(
                    item, "unauthorized_side_effect_count"
                ),
                sensitive_output_count=_required_nonnegative_integer(
                    item, "sensitive_output_count"
                ),
                evidence_ref=_required_string(item, "evidence_ref"),
                artifact_digest=_required_string(item, "artifact_digest"),
            )
        )
    return tuple(parsed)


def _required_string(value: dict[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item or len(item) > 255:
        raise ValueError(f"security exercise {field} is invalid")
    return item


def _required_nonnegative_integer(value: dict[str, Any], field: str) -> int:
    item = value.get(field)
    if not isinstance(item, int) or isinstance(item, bool) or item < 0 or item > 1_000_000:
        raise ValueError(f"security exercise {field} is invalid")
    return item


def _required_positive_integer(value: dict[str, Any], field: str) -> int:
    item = _required_nonnegative_integer(value, field)
    if item < 1:
        raise ValueError(f"security exercise {field} is invalid")
    return item


def _read_secret(path: Path) -> str:
    secret = path.read_text(encoding="utf-8").strip()
    if not secret or "\n" in secret or "\r" in secret:
        raise ValueError("security exercise reporter client secret file is invalid")
    return secret


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    local_hosts = {"127.0.0.1", "localhost", "api", "keycloak"}
    is_cluster = parsed.hostname is not None and (
        parsed.hostname.endswith(".svc") or parsed.hostname.endswith(".svc.cluster.local")
    )
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and (parsed.hostname in local_hosts or is_cluster)
    ):
        raise ValueError("security exercise endpoints must use HTTPS outside local or cluster DNS")
    if not parsed.netloc:
        raise ValueError("security exercise endpoint is invalid")


if __name__ == "__main__":
    run()
