"""HTTP closed-loop acceptance for tenant-scoped enterprise model imports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.enterprise_assets.models import EnterpriseModelComponent

SCHEMA_VERSION: Literal["enterprise-model-import-acceptance/v1"] = (
    "enterprise-model-import-acceptance/v1"
)
STATUS: Literal["ENTERPRISE_MODEL_IMPORT_PREFLIGHT_PASSED"] = (
    "ENTERPRISE_MODEL_IMPORT_PREFLIGHT_PASSED"
)
CANONICAL_RECEIPT_PATH = Path(
    "artifacts/enterprise-model-import-acceptance/receipt-seven-components.json"
)
LEGACY_COMPONENTS: tuple[EnterpriseModelComponent, ...] = (
    "LLM",
    "VLM",
    "ASR",
    "RUL",
)
COMPONENTS: tuple[EnterpriseModelComponent, ...] = (
    *LEGACY_COMPONENTS,
    "TTS",
    "EMBEDDING",
    "RERANKER",
)
AcceptanceExecutionMode = Literal[
    "EXTERNAL_STAGING_API",
    "EPHEMERAL_LOOPBACK_STAGING",
]


class EnterpriseModelImportAcceptanceError(RuntimeError):
    """The API did not satisfy the model-import preflight contract."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EnterpriseModelImportAcceptanceComponent(_ClosedModel):
    component: EnterpriseModelComponent
    import_id: str = Field(min_length=1)
    import_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_candidate_experiment_id: str = Field(min_length=1)
    imported_candidate_experiment_id: str = Field(min_length=1)
    imported_evaluation_id: str = Field(min_length=1)
    imported_suite_id: str = Field(min_length=1)
    source_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_classification: str = Field(min_length=1)
    operational_classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"]
    actual_sample_count: int = Field(ge=1)
    release_scope: Literal["STAGING_ONLY"]
    registry_state: str = Field(min_length=1)
    release_draft_eligible: bool
    release_draft_blockers: tuple[str, ...]
    compatible_baseline_release_ids: tuple[str, ...] = Field(min_length=1)


class EnterpriseModelImportAcceptanceReport(_ClosedModel):
    schema_version: Literal["enterprise-model-import-acceptance/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    release_scope: Literal["STAGING_ONLY"] = "STAGING_ONLY"
    execution_mode: AcceptanceExecutionMode = "EXTERNAL_STAGING_API"
    generated_at: datetime
    status: Literal["ENTERPRISE_MODEL_IMPORT_PREFLIGHT_PASSED"] = STATUS
    components: tuple[EnterpriseModelImportAcceptanceComponent, ...] = Field(
        min_length=4,
        max_length=7,
    )
    runtime_binding_verified: Literal[True] = True
    release_draft_import_preflight_verified: Literal[True] = True
    shadow_claim: Literal[False] = False
    canary_claim: Literal[False] = False
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def execute_enterprise_model_import_acceptance(
    base_url: str,
    bearer_token: str,
    *,
    timeout_seconds: float = 30.0,
    execution_mode: AcceptanceExecutionMode = "EXTERNAL_STAGING_API",
) -> EnterpriseModelImportAcceptanceReport:
    """Import all authoritative candidates and verify API-to-preflight correlation."""

    endpoint = _validated_endpoint(base_url)
    token = bearer_token.strip()
    if not token:
        raise EnterpriseModelImportAcceptanceError("acceptance_bearer_token_missing")
    if not 1 <= timeout_seconds <= 300:
        raise ValueError("timeout_seconds must be between 1 and 300")

    imported: dict[str, dict[str, Any]] = {}
    for component in COMPONENTS:
        envelope = _request_json(
            endpoint,
            token,
            "POST",
            f"/api/v1/enterprise-assets/runtime-bindings/{component}/imports",
            timeout_seconds=timeout_seconds,
            idempotency_key=f"enterprise-model-import-acceptance-v1:{component}",
        )
        imported[component] = _object(envelope, "data")

    runtime_envelope = _request_json(
        endpoint,
        token,
        "GET",
        "/api/v1/enterprise-assets/runtime-bindings",
        timeout_seconds=timeout_seconds,
    )
    runtime_items = runtime_envelope.get("data")
    if not isinstance(runtime_items, list):
        raise EnterpriseModelImportAcceptanceError("runtime_binding_response_invalid")
    runtime_by_component = {
        _text(_object(item, "evidence"), "component"): cast(dict[str, Any], item)
        for item in runtime_items
        if isinstance(item, dict)
    }

    results: list[EnterpriseModelImportAcceptanceComponent] = []
    for component in COMPONENTS:
        preview_envelope = _request_json(
            endpoint,
            token,
            "GET",
            (f"/api/v1/enterprise-assets/runtime-bindings/{component}/release-draft-preview"),
            timeout_seconds=timeout_seconds,
        )
        preview = _object(preview_envelope, "data")
        runtime = runtime_by_component.get(component)
        if runtime is None:
            raise EnterpriseModelImportAcceptanceError(f"runtime_binding_missing:{component}")
        source_import = imported[component]
        runtime_import = _object(runtime, "model_import")
        preview_import = _object(preview, "model_import")
        blockers = preview.get("blockers")
        if not isinstance(blockers, list) or any(not isinstance(item, str) for item in blockers):
            raise EnterpriseModelImportAcceptanceError(
                f"release_draft_blockers_invalid:{component}"
            )
        baselines = preview.get("baselines")
        if (
            not isinstance(baselines, list)
            or not baselines
            or any(not isinstance(item, dict) for item in baselines)
        ):
            raise EnterpriseModelImportAcceptanceError(
                f"compatible_release_baselines_invalid:{component}"
            )
        baseline_release_ids = tuple(
            _text(cast(dict[str, Any], item), "release_id") for item in baselines
        )
        eligible = _boolean(preview, "eligible")
        if (
            runtime.get("import_state") != "IMPORTED"
            or preview.get("import_state") != "IMPORTED"
            or runtime_import != source_import
            or preview_import != source_import
            or not eligible
            or blockers
            or len(set(baseline_release_ids)) != len(baseline_release_ids)
        ):
            raise EnterpriseModelImportAcceptanceError(
                f"enterprise_model_import_preflight_failed:{component}"
            )
        results.append(
            EnterpriseModelImportAcceptanceComponent(
                component=cast(Any, component),
                import_id=_text(source_import, "import_id"),
                import_hash=_text(source_import, "import_hash"),
                source_candidate_experiment_id=_text(
                    source_import,
                    "source_candidate_experiment_id",
                ),
                imported_candidate_experiment_id=_text(
                    source_import,
                    "imported_candidate_experiment_id",
                ),
                imported_evaluation_id=_text(
                    source_import,
                    "imported_evaluation_id",
                ),
                imported_suite_id=_text(source_import, "imported_suite_id"),
                source_evidence_chain_sha256=_text(
                    source_import,
                    "source_evidence_chain_sha256",
                ),
                source_classification=_text(
                    source_import,
                    "source_classification",
                ),
                operational_classification=cast(
                    Any,
                    _text(source_import, "operational_classification"),
                ),
                actual_sample_count=_positive_integer(
                    source_import,
                    "actual_sample_count",
                ),
                release_scope=cast(Any, _text(source_import, "release_scope")),
                registry_state=_text(runtime, "registry_state"),
                release_draft_eligible=eligible,
                release_draft_blockers=tuple(blockers),
                compatible_baseline_release_ids=baseline_release_ids,
            )
        )

    generated_at = datetime.now(UTC)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": "LOCAL_STAGING_PROJECT_AUTHORIZED",
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "release_scope": "STAGING_ONLY",
        "execution_mode": execution_mode,
        "generated_at": generated_at.isoformat(),
        "status": STATUS,
        "components": [item.model_dump(mode="json") for item in results],
        "runtime_binding_verified": True,
        "release_draft_import_preflight_verified": True,
        "shadow_claim": False,
        "canary_claim": False,
    }
    draft = EnterpriseModelImportAcceptanceReport(
        **payload,
        evidence_chain_sha256="0" * 64,
    )
    canonical_payload = draft.model_dump(
        mode="json",
        exclude={"evidence_chain_sha256"},
    )
    return draft.model_copy(
        update={"evidence_chain_sha256": _digest(canonical_payload)},
    )


def write_enterprise_model_import_acceptance(
    path: Path,
    report: EnterpriseModelImportAcceptanceReport,
) -> tuple[EnterpriseModelImportAcceptanceReport, bool]:
    """Write once; an equivalent existing receipt remains immutable."""

    _validate_acceptance_report(report)
    target = path.resolve()
    if target.exists():
        return _equivalent_existing_receipt(target, report)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    try:
        with target.open("x", encoding="utf-8") as receipt:
            receipt.write(encoded)
    except FileExistsError:
        return _equivalent_existing_receipt(target, report)
    return report, True


def verify_enterprise_model_import_acceptance(
    path: Path,
) -> EnterpriseModelImportAcceptanceReport:
    report = EnterpriseModelImportAcceptanceReport.model_validate_json(
        path.resolve(strict=True).read_text(encoding="utf-8")
    )
    _validate_acceptance_report(report)
    return report


def _validate_acceptance_report(
    report: EnterpriseModelImportAcceptanceReport,
) -> None:
    payload = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(payload):
        raise EnterpriseModelImportAcceptanceError(
            "enterprise_model_import_acceptance_integrity_failed"
        )
    component_order = tuple(item.component for item in report.components)
    if component_order not in (LEGACY_COMPONENTS, COMPONENTS):
        raise EnterpriseModelImportAcceptanceError(
            "enterprise_model_import_acceptance_components_incomplete"
        )
    if any(
        not item.release_draft_eligible or item.release_draft_blockers for item in report.components
    ):
        raise EnterpriseModelImportAcceptanceError(
            "enterprise_model_import_release_preflight_not_passed"
        )


def _equivalent_existing_receipt(
    target: Path,
    report: EnterpriseModelImportAcceptanceReport,
) -> tuple[EnterpriseModelImportAcceptanceReport, bool]:
    existing = verify_enterprise_model_import_acceptance(target)
    existing_bindings = {
        (
            item.component,
            item.import_id,
            item.import_hash,
            item.compatible_baseline_release_ids,
        )
        for item in existing.components
    }
    current_bindings = {
        (
            item.component,
            item.import_id,
            item.import_hash,
            item.compatible_baseline_release_ids,
        )
        for item in report.components
    }
    if existing_bindings != current_bindings:
        raise EnterpriseModelImportAcceptanceError(
            "immutable_acceptance_receipt_conflicts_with_current_imports"
        )
    return existing, False


def _request_json(
    endpoint: str,
    token: str,
    method: str,
    path: str,
    *,
    timeout_seconds: float,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    request = Request(
        endpoint + path,
        data=b"" if method == "POST" else None,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read(4096).decode("utf-8", errors="replace")
        raise EnterpriseModelImportAcceptanceError(
            f"acceptance_api_http_error:{exc.code}:{details}"
        ) from exc
    except (URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnterpriseModelImportAcceptanceError(
            f"acceptance_api_unavailable:{type(exc).__name__}"
        ) from exc
    if not isinstance(decoded, dict):
        raise EnterpriseModelImportAcceptanceError("acceptance_api_response_invalid")
    return cast(dict[str, Any], decoded)


def _validated_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    parsed = urlparse(endpoint)
    local_http = parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }
    if parsed.scheme != "https" and not local_http:
        raise EnterpriseModelImportAcceptanceError(
            "acceptance_endpoint_requires_https_or_localhost"
        )
    if not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise EnterpriseModelImportAcceptanceError("acceptance_endpoint_invalid")
    return endpoint


def _object(document: object, key: str) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise EnterpriseModelImportAcceptanceError(f"acceptance_field_invalid:{key}")
    value = document.get(key)
    if not isinstance(value, dict):
        raise EnterpriseModelImportAcceptanceError(f"acceptance_field_invalid:{key}")
    return cast(dict[str, Any], value)


def _text(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise EnterpriseModelImportAcceptanceError(f"acceptance_field_invalid:{key}")
    return value


def _positive_integer(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EnterpriseModelImportAcceptanceError(f"acceptance_field_invalid:{key}")
    return value


def _boolean(document: dict[str, Any], key: str) -> bool:
    value = document.get(key)
    if not isinstance(value, bool):
        raise EnterpriseModelImportAcceptanceError(f"acceptance_field_invalid:{key}")
    return value


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()
