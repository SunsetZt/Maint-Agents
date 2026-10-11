#!/usr/bin/env python3
"""Run a resumable-safe project-staging VLM and diagnosis acceptance path."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
ACCEPTANCE_CLIENT_ID = "real-model-acceptance"
ACCEPTANCE_SUBJECT_ID = "subject-real-model-acceptance"
ACCEPTANCE_TENANT_ID = "tenant-m1-demo"
ACCEPTANCE_SITE_IDS = frozenset({"site-m1-demo"})
ACCEPTANCE_ROLES = frozenset(
    {"model_engineer", "field_engineer", "after_sales_engineer"}
)
TERMINAL_RECOGNITION_FAILURES = {"FAILED", "CANCELLED", "CANCELED"}
TERMINAL_DIAGNOSIS_FAILURES = {
    "FAILED",
    "CANCELLED",
    "CANCELED",
    "NEEDS_DATA",
    "NEEDS_INPUT",
    "ESCALATED",
}


class AcceptanceFailure(RuntimeError):
    """A safe-to-display acceptance failure."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_: object, **__: object) -> None:
        return None


class ApiClient:
    def __init__(self, api_url: str, token: str, timeout_seconds: float) -> None:
        parsed = urllib.parse.urlsplit(api_url.rstrip("/"))
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise AcceptanceFailure("api_url_must_be_loopback_without_credentials")
        self._api_url = api_url.rstrip("/")
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(NoRedirect())

    def request(
        self,
        method: str,
        path: str,
        *,
        expected_statuses: set[int],
        json_body: dict[str, Any] | None = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not path.startswith("/") or "//" in path:
            raise AcceptanceFailure("api_path_invalid")
        body = content
        request_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
            "User-Agent": "ioap-real-model-acceptance/1",
        }
        if json_body is not None:
            body = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            self._api_url + path,
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self._timeout_seconds) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise AcceptanceFailure("api_response_too_large")
                if response.status not in expected_statuses:
                    raise AcceptanceFailure(f"api_unexpected_status:{response.status}")
        except urllib.error.HTTPError as exc:
            raw = exc.read(64 * 1024)
            try:
                error = json.loads(raw).get("error", {})
                code = str(error.get("code", "unknown_error"))
            except (AttributeError, json.JSONDecodeError, UnicodeDecodeError):
                code = "non_json_error"
            raise AcceptanceFailure(f"api_http_error:{exc.code}:{code}") from None
        except (TimeoutError, urllib.error.URLError, OSError):
            raise AcceptanceFailure("api_transport_unavailable") from None
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise AcceptanceFailure("api_response_invalid") from None
        if not isinstance(decoded, dict) or not isinstance(decoded.get("data"), (dict, list)):
            raise AcceptanceFailure("api_envelope_invalid")
        return decoded


def _read_token(path: Path, *, asset_id: str) -> str:
    if not path.is_file() or path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise AcceptanceFailure("access_token_file_invalid")
    token = path.read_text(encoding="utf-8").strip()
    parts = token.split(".")
    if len(parts) != 3 or len(token) > 16_384:
        raise AcceptanceFailure("access_token_invalid")
    try:
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        raise AcceptanceFailure("access_token_claims_invalid") from None
    if not isinstance(claims, dict):
        raise AcceptanceFailure("access_token_claims_invalid")

    def exact_string_list(value: object, expected: frozenset[str]) -> bool:
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(isinstance(item, str) for item in value)
            and frozenset(value) == expected
        )

    issued_at = claims.get("iat")
    expires_at = claims.get("exp")
    now = time.time()
    if (
        claims.get("azp") != ACCEPTANCE_CLIENT_ID
        or claims.get("subject_id") != ACCEPTANCE_SUBJECT_ID
        or claims.get("tenant_id") != ACCEPTANCE_TENANT_ID
        or not exact_string_list(claims.get("roles"), ACCEPTANCE_ROLES)
        or not exact_string_list(claims.get("asset_ids"), frozenset({asset_id}))
        or not exact_string_list(claims.get("site_ids"), ACCEPTANCE_SITE_IDS)
        or isinstance(issued_at, bool)
        or not isinstance(issued_at, (int, float))
        or issued_at > now + 30
        or isinstance(expires_at, bool)
        or not isinstance(expires_at, (int, float))
        or expires_at <= now + 30
    ):
        raise AcceptanceFailure("acceptance_identity_claims_incomplete")
    return token


def _data(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("data")
    if not isinstance(value, dict):
        raise AcceptanceFailure("api_data_invalid")
    return value


def _wait_for(
    client: ApiClient,
    path: str,
    *,
    status_field: str,
    success: set[str],
    failures: set[str],
    deadline: float,
    poll_seconds: float,
    stage: str,
) -> dict[str, Any]:
    transient_failures = 0
    while time.monotonic() < deadline:
        try:
            data = _data(client.request("GET", path, expected_statuses={200}))
            transient_failures = 0
        except AcceptanceFailure as exc:
            if str(exc) != "api_transport_unavailable" or transient_failures >= 2:
                raise AcceptanceFailure(f"{stage}:{exc}") from None
            transient_failures += 1
            time.sleep(poll_seconds)
            continue
        status = str(data.get(status_field, "")).upper()
        if status in success:
            return data
        if status in failures:
            reason = str(data.get("failure_reason") or data.get("stop_reason") or status)
            raise AcceptanceFailure(f"{stage}_failed:{reason[:128]}")
        time.sleep(poll_seconds)
    raise AcceptanceFailure(f"{stage}_timeout")


def _require_execution(
    execution: object,
    *,
    component: str,
    request_class: str,
    expected_release_id: str,
    expected_manifest_hash: str,
) -> dict[str, Any]:
    if not isinstance(execution, dict):
        raise AcceptanceFailure(f"{component}_execution_missing")
    if (
        execution.get("component") != component
        or execution.get("request_class") != request_class
        or execution.get("target_environment") != "STAGING"
        or execution.get("release_id") != expected_release_id
        or execution.get("manifest_hash") != expected_manifest_hash
        or execution.get("guardrail_decision") != "ALLOWED"
        or not execution.get("inference_request_id")
        or int(execution.get("prompt_tokens", 0)) <= 0
        or int(execution.get("completion_tokens", 0)) <= 0
        or float(execution.get("latency_ms", 0)) <= 0
    ):
        raise AcceptanceFailure(f"{component}_execution_mismatch")
    return execution


def _write_result(path: Path, payload: dict[str, str]) -> None:
    if path.is_symlink() or path.exists() and not path.is_file() or not path.parent.is_dir():
        raise AcceptanceFailure("result_path_invalid")
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def run(args: argparse.Namespace) -> dict[str, str]:
    image_path = Path(args.image)
    if not image_path.is_file() or image_path.is_symlink():
        raise AcceptanceFailure("acceptance_image_unavailable")
    image = image_path.read_bytes()
    if hashlib.sha256(image).hexdigest() != args.image_sha256:
        raise AcceptanceFailure("acceptance_image_digest_mismatch")
    token = _read_token(Path(args.token_file), asset_id=args.asset_id)
    client = ApiClient(args.api_url, token, args.request_timeout_seconds)
    deadline = time.monotonic() + args.workflow_timeout_seconds
    nonce = uuid4().hex

    print("[acceptance] creating isolated incident draft", file=sys.stderr, flush=True)
    draft = _data(
        client.request(
            "POST",
            "/incident-drafts",
            expected_statuses={200, 201},
            headers={"Idempotency-Key": f"real-model-{nonce}-draft"},
            json_body={
                "asset_id": args.asset_id,
                "description": "Project-staging real-model acceptance: pump seal leakage.",
            },
        )
    )
    draft_id = str(draft["draft_id"])
    draft_version = int(draft["version"])

    print("[acceptance] uploading immutable pump evidence", file=sys.stderr, flush=True)
    media = _data(
        client.request(
            "POST",
            f"/incident-drafts/{draft_id}/media",
            expected_statuses={201},
            content=image,
            headers={"Content-Type": "image/png"},
        )
    )
    media_id = str(media["media_id"])
    _wait_for(
        client,
        f"/media/{media_id}/status",
        status_field="scan_state",
        success={"CLEAN"},
        failures={"REJECTED", "INFECTED", "FAILED"},
        deadline=deadline,
        poll_seconds=args.poll_seconds,
        stage="media_scan",
    )

    print("[acceptance] waiting for governed VLM recognition", file=sys.stderr, flush=True)
    recognition = _data(
        client.request(
            "POST",
            f"/incident-drafts/{draft_id}/recognition-runs",
            expected_statuses={200, 202},
            headers={"Idempotency-Key": f"real-model-{nonce}-recognition"},
            json_body={"media_id": media_id, "processor_profile": "local-core-v1"},
        )
    )
    recognition_run_id = str(recognition["recognition_run_id"])
    _wait_for(
        client,
        f"/incident-drafts/{draft_id}/recognition-runs/{recognition_run_id}",
        status_field="status",
        success={"SUCCEEDED"},
        failures=TERMINAL_RECOGNITION_FAILURES,
        deadline=deadline,
        poll_seconds=args.poll_seconds,
        stage="recognition",
    )
    evidence = _data(
        client.request("GET", f"/incident-drafts/{draft_id}/evidence", expected_statuses={200})
    )
    processors = evidence.get("processor_versions", {})
    if (
        not isinstance(processors, dict)
        or str(processors.get("ocr", "")).lower().startswith("development-")
        or processors.get("ocr.release_id") != args.expected_release_id
    ):
        raise AcceptanceFailure("ocr_release_binding_mismatch")
    executions = evidence.get("model_executions", [])
    vlm_execution = next(
        (item for item in executions if isinstance(item, dict) and item.get("component") == "vlm"),
        None,
    )
    vlm_execution = _require_execution(
        vlm_execution,
        component="vlm",
        request_class="VLM",
        expected_release_id=args.expected_release_id,
        expected_manifest_hash=args.expected_manifest_hash,
    )

    print("[acceptance] confirming reviewed OCR and VLM candidates", file=sys.stderr, flush=True)
    corrections = {
        str(item["entity_id"]): str(
            item.get("original_value") or item.get("normalized_value") or ""
        )
        for item in evidence.get("extracted_entities", [])
        if isinstance(item, dict) and item.get("validation_status") != "VALID"
    }
    confirmation = _data(
        client.request(
            "POST",
            f"/incident-drafts/{draft_id}/recognition-confirmations",
            expected_statuses={200},
            headers={"If-Match": f'"{int(evidence["version"])}"'},
            json_body={
                "bundle_id": evidence["bundle_id"],
                "corrections": corrections,
                "finding_dispositions": {
                    str(item["finding_id"]): "ACCEPTED"
                    for item in evidence.get("visual_findings", [])
                    if isinstance(item, dict)
                },
                "ocr_block_decisions": {
                    str(item["block_id"]): {"disposition": "ACCEPTED"}
                    for item in evidence.get("ocr_blocks", [])
                    if isinstance(item, dict)
                },
                "qr_code_decisions": {
                    str(item["candidate_id"]): {
                        "disposition": "REJECTED" if item.get("security_findings") else "ACCEPTED"
                    }
                    for item in evidence.get("qr_codes", [])
                    if isinstance(item, dict)
                },
            },
        )
    )
    if confirmation.get("status") != "CONFIRMED":
        raise AcceptanceFailure("recognition_confirmation_incomplete")

    incident = _data(
        client.request(
            "POST",
            f"/incident-drafts/{draft_id}/submit",
            expected_statuses={200, 201},
            headers={
                "Idempotency-Key": f"real-model-{nonce}-submit",
                "If-Match": f'"{draft_version}"',
            },
            json_body={"evidence_bundle_id": evidence["bundle_id"]},
        )
    )
    incident_id = str(incident["incident_id"])
    triaged = _data(
        client.request(
            "POST",
            f"/incidents/{incident_id}/triage",
            expected_statuses={200},
            headers={"If-Match": f'"{int(incident["version"])}"'},
        )
    )

    print("[acceptance] waiting for governed fine-tuned diagnosis", file=sys.stderr, flush=True)
    diagnosis = _data(
        client.request(
            "POST",
            f"/incidents/{incident_id}/diagnoses",
            expected_statuses={200, 202},
            headers={
                "Idempotency-Key": f"real-model-{nonce}-diagnosis",
                "If-Match": f'"{int(triaged["version"])}"',
            },
        )
    )
    diagnosis_run_id = str(diagnosis["diagnosis_run_id"])
    diagnosis = _wait_for(
        client,
        f"/diagnosis-runs/{diagnosis_run_id}",
        status_field="status",
        success={"COMPLETED"},
        failures=TERMINAL_DIAGNOSIS_FAILURES,
        deadline=deadline,
        poll_seconds=args.poll_seconds,
        stage="diagnosis",
    )
    diagnosis_execution = _require_execution(
        diagnosis.get("model_execution"),
        component="diagnosis",
        request_class="DIAGNOSIS",
        expected_release_id=args.expected_release_id,
        expected_manifest_hash=args.expected_manifest_hash,
    )
    return {
        "draft_id": draft_id,
        "incident_id": incident_id,
        "recognition_run_id": recognition_run_id,
        "diagnosis_run_id": diagnosis_run_id,
        "vlm_inference_request_id": str(vlm_execution["inference_request_id"]),
        "diagnosis_inference_request_id": str(diagnosis_execution["inference_request_id"]),
        "release_id": args.expected_release_id,
        "manifest_hash": args.expected_manifest_hash,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--expected-release-id", required=True)
    parser.add_argument("--expected-manifest-hash", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--workflow-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if (
        args.request_timeout_seconds <= 0
        or args.workflow_timeout_seconds <= 0
        or args.poll_seconds <= 0
    ):
        parser.error("timeouts must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
        _write_result(Path(args.output), result)
    except (AcceptanceFailure, KeyError, TypeError, ValueError) as exc:
        print(f"Real-model business acceptance failed: {exc}", file=sys.stderr)
        return 3
    print(
        "REAL_MODEL_BUSINESS_EXERCISE_COMPLETE "
        f"recognition_run_id={result['recognition_run_id']} "
        f"diagnosis_run_id={result['diagnosis_run_id']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
