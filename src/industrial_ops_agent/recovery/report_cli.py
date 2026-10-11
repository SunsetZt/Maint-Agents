"""Submit a bounded verifier-produced recovery report through the governed API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx

MAX_REPORT_BYTES = 64 * 1024


def run() -> None:
    parser = argparse.ArgumentParser(description="Submit isolated recovery verification evidence.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--token-url", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret-file", required=True, type=Path)
    parser.add_argument("--ca-file", type=Path)
    args = parser.parse_args()
    report = _load_report(args.report)
    secret = _read_secret(args.client_secret_file)
    verify: bool | str = str(args.ca_file) if args.ca_file is not None else True
    _validate_endpoint(args.api_url)
    _validate_endpoint(args.token_url)
    with httpx.Client(timeout=15.0, verify=verify) as client:
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
        token_payload = token_response.json()
        access_token = token_payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError("OIDC token response did not contain an access token")
        response = client.post(
            f"{args.api_url.rstrip('/')}/api/v1/recovery/evidence",
            json=report,
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-Request-ID": f"recovery-evidence-{uuid4().hex}",
            },
        )
        response.raise_for_status()
        payload = response.json()
    evidence_id = payload.get("data", {}).get("evidence_id")
    if not isinstance(evidence_id, str):
        raise RuntimeError("recovery API response did not contain an evidence id")
    print(f"Recovery evidence accepted: {evidence_id}")


def _load_report(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_REPORT_BYTES:
        raise ValueError("recovery report size is invalid")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery report must be a JSON object")
    return payload


def _read_secret(path: Path) -> str:
    secret = path.read_text(encoding="utf-8").strip()
    if not secret or "\n" in secret or "\r" in secret:
        raise ValueError("recovery reporter client secret file is invalid")
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
        raise ValueError("recovery reporter endpoints must use HTTPS outside local or cluster DNS")
    if not parsed.netloc:
        raise ValueError("recovery reporter endpoint is invalid")


if __name__ == "__main__":
    run()
