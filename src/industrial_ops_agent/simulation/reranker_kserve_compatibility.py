"""Fail-closed compatibility evidence for additive KServe provider changes."""

from __future__ import annotations

import argparse
import ast
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from industrial_ops_agent.simulation.reranker_kserve_readiness import (
    ACCEPTANCE_RELATIVE,
    KSERVE_PROVIDER_RELATIVE,
    EnterpriseRerankerKServeReadinessReport,
    FileBinding,
    document_sha256,
    file_binding,
)

SCHEMA_VERSION: Literal["enterprise-reranker-kserve-provider-compatibility/v1"] = (
    "enterprise-reranker-kserve-provider-compatibility/v1"
)
STATUS: Literal["RERANKER_KSERVE_PROVIDER_ADDITIVE_TTS_COMPATIBILITY_VERIFIED"] = (
    "RERANKER_KSERVE_PROVIDER_ADDITIVE_TTS_COMPATIBILITY_VERIFIED"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
OUTPUT_RELATIVE = Path(
    "artifacts/m7-reranker-kserve-readiness/provider-compatibility.json"
)

_TTS_REGION_NAMES = (
    "provider_reconcile_tts",
    "tts_serving_runtime",
    "tts_inference_service",
    "tts_http_route",
    "tts_labels",
)
_TTS_REGIONS = (
    (
        b'            if "tts" in target.desired_spec:\n',
        b'            if "timeseries" in target.desired_spec:\n',
    ),
    (
        b"def _tts_serving_runtime(target: ProviderTarget) -> dict[str, Any]:\n",
        b"def _embedding_inference_service(target: ProviderTarget) -> dict[str, Any]:\n",
    ),
    (
        b"def _tts_inference_service(target: ProviderTarget) -> dict[str, Any]:\n",
        b"def _reranker_inference_service(target: ProviderTarget) -> dict[str, Any]:\n",
    ),
    (
        b'    tts = target.desired_spec.get("tts")\n',
        b'    timeseries = target.desired_spec.get("timeseries")\n',
    ),
    (
        b"def _tts_labels(target: ProviderTarget) -> dict[str, str]:\n",
        b"def _resource_ready(document: dict[str, Any], "
        b"required_types: tuple[str, ...]) -> bool:\n",
    ),
)


class RerankerKServeCompatibilityError(RuntimeError):
    """The current provider is not an additive extension of the accepted source."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderCompatibilityProjection(_ClosedModel):
    policy: Literal["EXACT_HISTORICAL_SOURCE_PLUS_TTS_REGIONS_ONLY"]
    removed_regions: tuple[str, ...] = Field(min_length=5, max_length=5)
    projected_size_bytes: int = Field(gt=0)
    projected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    full_source_syntax_valid: Literal[True] = True
    historical_reranker_bytes_preserved: Literal[True] = True
    tts_regions_are_additive: Literal[True] = True


class EnterpriseRerankerKServeProviderCompatibilityReport(_ClosedModel):
    schema_version: Literal[
        "enterprise-reranker-kserve-provider-compatibility/v1"
    ] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal[
        "RERANKER_KSERVE_PROVIDER_ADDITIVE_TTS_COMPATIBILITY_VERIFIED"
    ] = STATUS
    readiness: FileBinding
    readiness_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    historical_provider: FileBinding
    current_provider: FileBinding
    projection: ProviderCompatibilityProjection
    hard_gates: dict[str, Literal[True]]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_reranker_kserve_provider_compatibility(
    repo_root: Path,
    *,
    readiness_path: Path = ACCEPTANCE_RELATIVE,
    output_path: Path = OUTPUT_RELATIVE,
) -> EnterpriseRerankerKServeProviderCompatibilityReport:
    root = repo_root.resolve(strict=True)
    readiness_file, readiness = _load_historical_readiness(root, readiness_path)
    historical = readiness.runtime_image.kserve_provider
    current_path = _inside_file(root, KSERVE_PROVIDER_RELATIVE)
    current = FileBinding.model_validate(file_binding(root, current_path))
    projection = _compatibility_projection(current_path)
    _require_projection_matches_historical(projection, historical, current)

    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "readiness": file_binding(root, readiness_file),
        "readiness_evidence_chain_sha256": readiness.evidence_chain_sha256,
        "historical_provider": historical.model_dump(mode="json"),
        "current_provider": current.model_dump(mode="json"),
        "projection": projection.model_dump(mode="json"),
        "hard_gates": {
            "historical_readiness_chain_verified": True,
            "historical_provider_binding_preserved": True,
            "current_provider_binding_verified": True,
            "full_provider_source_syntax_verified": True,
            "only_tts_additive_regions_removed": True,
            "reranker_provider_bytes_match_historical_acceptance": True,
            "no_production_claim_created": True,
        },
    }
    draft = EnterpriseRerankerKServeProviderCompatibilityReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    canonical = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    report = draft.model_copy(
        update={"evidence_chain_sha256": document_sha256(canonical)}
    )
    destination = _inside_output(root, output_path)
    _write_json(destination, report.model_dump(mode="json"))
    return verify_reranker_kserve_provider_compatibility(
        root,
        expected_historical_provider=historical,
        compatibility_path=destination,
    )


def verify_reranker_kserve_provider_compatibility(
    repo_root: Path,
    *,
    expected_historical_provider: FileBinding,
    compatibility_path: Path = OUTPUT_RELATIVE,
) -> EnterpriseRerankerKServeProviderCompatibilityReport:
    root = repo_root.resolve(strict=True)
    path = _inside_file(root, compatibility_path)
    try:
        report = EnterpriseRerankerKServeProviderCompatibilityReport.model_validate_json(
            path.read_bytes()
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise RerankerKServeCompatibilityError(
            "Reranker KServe compatibility evidence is invalid"
        ) from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != document_sha256(unsigned):
        raise RerankerKServeCompatibilityError(
            "Reranker KServe compatibility evidence chain changed"
        )
    readiness_path, readiness = _load_historical_readiness(
        root,
        Path(report.readiness.path),
    )
    if (
        file_binding(root, readiness_path) != report.readiness.model_dump(mode="json")
        or readiness.evidence_chain_sha256
        != report.readiness_evidence_chain_sha256
        or readiness.runtime_image.kserve_provider != report.historical_provider
        or report.historical_provider != expected_historical_provider
    ):
        raise RerankerKServeCompatibilityError(
            "Reranker historical provider binding changed"
        )
    current_path = _inside_file(root, KSERVE_PROVIDER_RELATIVE)
    current = FileBinding.model_validate(file_binding(root, current_path))
    if current != report.current_provider:
        raise RerankerKServeCompatibilityError(
            "Reranker current KServe provider binding changed"
        )
    projection = _compatibility_projection(current_path)
    if projection != report.projection:
        raise RerankerKServeCompatibilityError(
            "Reranker KServe compatibility projection changed"
        )
    _require_projection_matches_historical(
        projection,
        expected_historical_provider,
        current,
    )
    if set(report.hard_gates.values()) != {True}:
        raise RerankerKServeCompatibilityError(
            "Reranker KServe compatibility gate changed"
        )
    return report


def _load_historical_readiness(
    root: Path,
    path: Path,
) -> tuple[Path, EnterpriseRerankerKServeReadinessReport]:
    readiness_path = _inside_file(root, path)
    try:
        report = EnterpriseRerankerKServeReadinessReport.model_validate_json(
            readiness_path.read_bytes()
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise RerankerKServeCompatibilityError(
            "Historical Reranker readiness is invalid"
        ) from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != document_sha256(unsigned):
        raise RerankerKServeCompatibilityError(
            "Historical Reranker readiness evidence chain changed"
        )
    return readiness_path, report


def _compatibility_projection(path: Path) -> ProviderCompatibilityProjection:
    source = path.read_bytes()
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RerankerKServeCompatibilityError(
            "Current KServe provider source is not valid Python"
        ) from exc
    projected = source
    for start_marker, end_marker in reversed(_TTS_REGIONS):
        if projected.count(start_marker) != 1 or projected.count(end_marker) != 1:
            raise RerankerKServeCompatibilityError(
                "KServe TTS compatibility region is missing or ambiguous"
            )
        start = projected.index(start_marker)
        end = projected.index(end_marker)
        if end <= start:
            raise RerankerKServeCompatibilityError(
                "KServe TTS compatibility region order changed"
            )
        projected = projected[:start] + projected[end:]
    if b'_tts_' in projected or b'"tts"' in projected:
        raise RerankerKServeCompatibilityError(
            "TTS provider code escaped the approved additive regions"
        )
    return ProviderCompatibilityProjection(
        policy="EXACT_HISTORICAL_SOURCE_PLUS_TTS_REGIONS_ONLY",
        removed_regions=_TTS_REGION_NAMES,
        projected_size_bytes=len(projected),
        projected_sha256=sha256(projected).hexdigest(),
        full_source_syntax_valid=True,
        historical_reranker_bytes_preserved=True,
        tts_regions_are_additive=True,
    )


def _require_projection_matches_historical(
    projection: ProviderCompatibilityProjection,
    historical: FileBinding,
    current: FileBinding,
) -> None:
    if current.path != historical.path or current.path != KSERVE_PROVIDER_RELATIVE.as_posix():
        raise RerankerKServeCompatibilityError("KServe provider path changed")
    if (
        current.sha256 == historical.sha256
        or projection.projected_size_bytes != historical.size_bytes
        or projection.projected_sha256 != historical.sha256
        or projection.removed_regions != _TTS_REGION_NAMES
    ):
        raise RerankerKServeCompatibilityError(
            "Current provider is not the historical provider plus TTS-only regions"
        )


def _inside_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RerankerKServeCompatibilityError(
            "Reranker KServe compatibility path is invalid"
        ) from exc
    if not resolved.is_file():
        raise RerankerKServeCompatibilityError(
            "Reranker KServe compatibility file is missing"
        )
    return resolved


def _inside_output(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RerankerKServeCompatibilityError(
            "Reranker KServe compatibility output path is invalid"
        ) from exc
    return resolved


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify additive TTS compatibility with accepted Reranker KServe code"
    )
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--readiness", type=Path, default=ACCEPTANCE_RELATIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT_RELATIVE)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    _, readiness = _load_historical_readiness(root, args.readiness)
    if args.command == "build":
        report = build_reranker_kserve_provider_compatibility(
            root,
            readiness_path=args.readiness,
            output_path=args.output,
        )
    else:
        report = verify_reranker_kserve_provider_compatibility(
            root,
            expected_historical_provider=readiness.runtime_image.kserve_provider,
            compatibility_path=args.output,
        )
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OUTPUT_RELATIVE",
    "EnterpriseRerankerKServeProviderCompatibilityReport",
    "RerankerKServeCompatibilityError",
    "build_reranker_kserve_provider_compatibility",
    "verify_reranker_kserve_provider_compatibility",
]
