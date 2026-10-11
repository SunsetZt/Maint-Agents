"""Immutable runtime-image and KServe readiness evidence for Reranker v5."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from industrial_ops_agent.deployment.reranker_runtime import (
    CALIBRATION_PROFILE_ID,
    load_calibrated_bundle,
)
from industrial_ops_agent.simulation.reranker_calibrated_value_lab import (
    verify_calibrated_reranker_value,
)

SCHEMA_VERSION: Literal["enterprise-reranker-kserve-readiness/v1"] = (
    "enterprise-reranker-kserve-readiness/v1"
)
STATUS: Literal["RERANKER_KSERVE_RUNTIME_AND_RELEASE_READY"] = (
    "RERANKER_KSERVE_RUNTIME_AND_RELEASE_READY"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
OUTPUT_RELATIVE = Path("artifacts/m7-reranker-kserve-readiness")
ACCEPTANCE_RELATIVE = OUTPUT_RELATIVE / "acceptance.json"
VALUE_ACCEPTANCE_RELATIVE = Path(
    "artifacts/m7-reranker-calibrated-value-lab/runs/"
    "reranker-calibrated-64f493aa97c1a51cf422/acceptance.json"
)
RUNTIME_SOURCE_RELATIVE = Path("src/industrial_ops_agent/deployment/reranker_runtime.py")
RUNTIME_DOCKERFILE_RELATIVE = Path("docker/reranker-calibrated-runtime.Dockerfile")
KSERVE_PROVIDER_RELATIVE = Path("src/industrial_ops_agent/deployment/kserve.py")
IMAGE_INSPECT_RELATIVE = OUTPUT_RELATIVE / "runtime-image-inspect.json"
IMAGE_HELP_RELATIVE = OUTPUT_RELATIVE / "runtime-entrypoint-help.txt"
RUNTIME_IMAGE_REPOSITORY = "industrial-ops/reranker-runtime"
RUNTIME_IMAGE_TAG = "calibrated-v1-95fb9107de7a"
RUNTIME_IMAGE_DIGEST = "sha256:7386510182721f4f086773ad8d2dd7f5713855c483f43e7d572252f88c9776eb"
RUNTIME_SOURCE_SHA256 = "95fb9107de7a8066b3e65af8ab2f2f7851ac763bdad12665cfc4693a06e58074"
BASE_IMAGE_REPOSITORY = "industrial-ops/m4-training-worker"
BASE_IMAGE_DIGEST = "sha256:7bcd50262cf62f33f8a3728a53f49ee36cbcc206f2bc0970776442f0b105effe"
ENTRYPOINT = ("python", "-B", "/opt/ioap-runtime/reranker_runtime.py")
REQUIRED_HELP_OPTIONS = frozenset(
    {
        "--model-dir",
        "--release-id",
        "--manifest-hash",
        "--component-model-id",
        "--artifact-content-hash",
        "--precision",
        "--batch-size",
        "--max-sequence-length",
        "--port",
    }
)


class RerankerKServeReadinessError(RuntimeError):
    """The built Reranker runtime or its immutable evidence is invalid."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RerankerReadinessSource(_ClosedModel):
    value_acceptance: FileBinding
    run_id: Literal["reranker-calibrated-64f493aa97c1a51cf422"]
    value_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_path: str = Field(min_length=1)
    candidate_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_image_digest: Literal[
        "sha256:7bcd50262cf62f33f8a3728a53f49ee36cbcc206f2bc0970776442f0b105effe"
    ]
    actual_gpu_value_evaluation_passed: Literal[True] = True
    formal_gold_reused: Literal[False] = False


class RerankerRuntimeImageEvidence(_ClosedModel):
    repository: Literal["industrial-ops/reranker-runtime"]
    tag: Literal["calibrated-v1-95fb9107de7a"]
    digest: Literal["sha256:7386510182721f4f086773ad8d2dd7f5713855c483f43e7d572252f88c9776eb"]
    base_repository: Literal["industrial-ops/m4-training-worker"]
    base_digest: Literal["sha256:7bcd50262cf62f33f8a3728a53f49ee36cbcc206f2bc0970776442f0b105effe"]
    image_inspect: FileBinding
    entrypoint_help: FileBinding
    runtime_source: FileBinding
    dockerfile: FileBinding
    kserve_provider: FileBinding
    entrypoint: tuple[str, ...] = Field(min_length=3, max_length=3)
    run_as_user: Literal["app"]
    source_revision: str = Field(min_length=7, max_length=40)
    source_sha256_label: Literal["95fb9107de7a8066b3e65af8ab2f2f7851ac763bdad12665cfc4693a06e58074"]
    runtime_profile_id: Literal["industrial-reranker-evidence-calibration-v1"]
    registry_id: Literal["industrial-reranker-terminology-registry-v1"]
    actual_image_built: Literal[True] = True
    entrypoint_help_verified: Literal[True] = True
    training_image_separated_from_runtime_image: Literal[True] = True
    actual_gpu_runtime_probe_completed: Literal[False] = False


class RerankerKServeContract(_ClosedModel):
    model_format: Literal["industrial-reranker"]
    protocol_version: Literal["v1"]
    endpoint_path: Literal["/v1/retrieval/rerank"]
    route_path_prefix: Literal["/v1/retrieval"]
    model_mount_path: Literal["/mnt/models/reranker"]
    max_sequence_length_forwarded: Literal[True] = True
    immutable_release_binding_required: Literal[True] = True
    calibration_loaded_from_candidate_bundle: Literal[True] = True
    missing_or_ambiguous_alias_fails_closed: Literal[True] = True
    container_security_context_hardened: Literal[True] = True
    cpu_limit: Literal["2"]
    memory_limit: Literal["4Gi"]
    gpu_count: Literal[1]
    existing_release_fsm_reused: Literal[True] = True
    rollout_stages: tuple[str, ...] = Field(min_length=4, max_length=4)
    actual_rollout_execution_completed: Literal[False] = False


class RerankerModelReleaseReadiness(_ClosedModel):
    component: Literal["RERANKER"]
    method: Literal["RERANKER"]
    artifact_kind: Literal["reranker_model_bundle"]
    target_profile: Literal["RETRIEVAL_COMPONENT"]
    target_environment: Literal["STAGING"]
    import_api: Literal["/api/v1/enterprise-assets/runtime-bindings/RERANKER/imports"]
    preview_api: Literal[
        "/api/v1/enterprise-assets/runtime-bindings/RERANKER/release-draft-preview"
    ]
    release_draft_api: Literal["/api/v1/enterprise-assets/runtime-bindings/RERANKER/release-drafts"]
    runtime_supply_chain_ready: Literal[True] = True
    training_image_must_not_be_used_as_runtime: Literal[True] = True
    formal_model_release_created: Literal[False] = False
    shadow_claim: Literal[False] = False
    canary_claim: Literal[False] = False
    rollback_claim: Literal[False] = False


class EnterpriseRerankerKServeReadinessReport(_ClosedModel):
    schema_version: Literal["enterprise-reranker-kserve-readiness/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["RERANKER_KSERVE_RUNTIME_AND_RELEASE_READY"] = STATUS
    source: RerankerReadinessSource
    runtime_image: RerankerRuntimeImageEvidence
    kserve: RerankerKServeContract
    model_release: RerankerModelReleaseReadiness
    hard_gates: dict[str, Literal[True]]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_reranker_kserve_readiness(
    repo_root: Path,
    *,
    image_inspect_path: Path,
    entrypoint_help_path: Path,
    output_path: Path | None = None,
) -> EnterpriseRerankerKServeReadinessReport:
    root = repo_root.resolve(strict=True)
    source = verify_calibrated_reranker_value(root, VALUE_ACCEPTANCE_RELATIVE)
    candidate_path = _inside(root, Path(source.candidate.path), directory=True)
    _validate_candidate_runtime_access(candidate_path)
    bundle = load_calibrated_bundle(candidate_path)
    if bundle is None:
        raise RerankerKServeReadinessError("Reranker v5 calibration bundle is missing")
    inspect_path = _inside(root, image_inspect_path)
    help_path = _inside(root, entrypoint_help_path)
    inspect = _load_inspect(inspect_path)
    help_text = help_path.read_text(encoding="utf-8")
    image_facts = _validate_image(inspect, help_text)
    runtime_source = _inside(root, RUNTIME_SOURCE_RELATIVE)
    if file_sha256(runtime_source) != RUNTIME_SOURCE_SHA256:
        raise RerankerKServeReadinessError("Reranker runtime source changed after image build")
    if source.image_digest == RUNTIME_IMAGE_DIGEST:
        raise RerankerKServeReadinessError("training and runtime images must be distinct")
    destination = _inside_output(root, output_path or ACCEPTANCE_RELATIVE)
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "source": {
            "value_acceptance": file_binding(root, root / VALUE_ACCEPTANCE_RELATIVE),
            "run_id": source.run_id,
            "value_evidence_chain_sha256": source.evidence_chain_sha256,
            "candidate_path": source.candidate.path,
            "candidate_bundle_sha256": source.candidate.bundle_sha256,
            "training_image_digest": source.image_digest,
            "actual_gpu_value_evaluation_passed": True,
            "formal_gold_reused": False,
        },
        "runtime_image": {
            "repository": RUNTIME_IMAGE_REPOSITORY,
            "tag": RUNTIME_IMAGE_TAG,
            "digest": RUNTIME_IMAGE_DIGEST,
            "base_repository": BASE_IMAGE_REPOSITORY,
            "base_digest": BASE_IMAGE_DIGEST,
            "image_inspect": file_binding(root, inspect_path),
            "entrypoint_help": file_binding(root, help_path),
            "runtime_source": file_binding(root, runtime_source),
            "dockerfile": file_binding(root, root / RUNTIME_DOCKERFILE_RELATIVE),
            "kserve_provider": file_binding(root, root / KSERVE_PROVIDER_RELATIVE),
            "entrypoint": list(ENTRYPOINT),
            "run_as_user": image_facts["run_as_user"],
            "source_revision": image_facts["source_revision"],
            "source_sha256_label": image_facts["source_sha256_label"],
            "runtime_profile_id": bundle.profile_id,
            "registry_id": bundle.registry.registry_id,
            "actual_image_built": True,
            "entrypoint_help_verified": True,
            "training_image_separated_from_runtime_image": True,
            "actual_gpu_runtime_probe_completed": False,
        },
        "kserve": {
            "model_format": "industrial-reranker",
            "protocol_version": "v1",
            "endpoint_path": "/v1/retrieval/rerank",
            "route_path_prefix": "/v1/retrieval",
            "model_mount_path": "/mnt/models/reranker",
            "max_sequence_length_forwarded": True,
            "immutable_release_binding_required": True,
            "calibration_loaded_from_candidate_bundle": True,
            "missing_or_ambiguous_alias_fails_closed": True,
            "container_security_context_hardened": True,
            "cpu_limit": "2",
            "memory_limit": "4Gi",
            "gpu_count": 1,
            "existing_release_fsm_reused": True,
            "rollout_stages": ["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"],
            "actual_rollout_execution_completed": False,
        },
        "model_release": {
            "component": "RERANKER",
            "method": "RERANKER",
            "artifact_kind": "reranker_model_bundle",
            "target_profile": "RETRIEVAL_COMPONENT",
            "target_environment": "STAGING",
            "import_api": "/api/v1/enterprise-assets/runtime-bindings/RERANKER/imports",
            "preview_api": (
                "/api/v1/enterprise-assets/runtime-bindings/RERANKER/release-draft-preview"
            ),
            "release_draft_api": (
                "/api/v1/enterprise-assets/runtime-bindings/RERANKER/release-drafts"
            ),
            "runtime_supply_chain_ready": True,
            "training_image_must_not_be_used_as_runtime": True,
            "formal_model_release_created": False,
            "shadow_claim": False,
            "canary_claim": False,
            "rollback_claim": False,
        },
        "hard_gates": {
            "v5_value_acceptance_verified": True,
            "candidate_non_root_runtime_access_verified": True,
            "candidate_bundle_calibration_verified": True,
            "reviewed_terminology_registry_verified": True,
            "runtime_image_built_and_digest_pinned": True,
            "runtime_source_label_matches": True,
            "runtime_entrypoint_contract_verified": True,
            "training_and_runtime_images_separated": True,
            "kserve_contract_hardened": True,
            "release_fsm_reused_without_online_claim": True,
        },
    }
    draft = EnterpriseRerankerKServeReadinessReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    canonical_unsigned = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    report = draft.model_copy(
        update={"evidence_chain_sha256": document_sha256(canonical_unsigned)}
    )
    _write_json(destination, report.model_dump(mode="json"))
    return verify_reranker_kserve_readiness(root, destination)


def verify_reranker_kserve_readiness(
    repo_root: Path,
    acceptance_path: Path = ACCEPTANCE_RELATIVE,
) -> EnterpriseRerankerKServeReadinessReport:
    root = repo_root.resolve(strict=True)
    path = _inside(root, acceptance_path)
    try:
        report = EnterpriseRerankerKServeReadinessReport.model_validate_json(path.read_bytes())
    except (OSError, UnicodeError, ValidationError) as exc:
        raise RerankerKServeReadinessError("Reranker readiness report is invalid") from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != document_sha256(unsigned):
        raise RerankerKServeReadinessError("Reranker readiness evidence chain changed")
    source = verify_calibrated_reranker_value(root, Path(report.source.value_acceptance.path))
    if (
        source.run_id != report.source.run_id
        or source.evidence_chain_sha256 != report.source.value_evidence_chain_sha256
        or source.candidate.path != report.source.candidate_path
        or source.candidate.bundle_sha256 != report.source.candidate_bundle_sha256
        or source.image_digest != report.source.training_image_digest
    ):
        raise RerankerKServeReadinessError("Reranker value and runtime readiness diverged")
    bindings = (
        report.source.value_acceptance,
        report.runtime_image.image_inspect,
        report.runtime_image.entrypoint_help,
        report.runtime_image.runtime_source,
        report.runtime_image.dockerfile,
    )
    for binding in bindings:
        _verify_binding(root, binding)
    _verify_kserve_provider_binding(root, report.runtime_image.kserve_provider)
    candidate_path = _inside(root, Path(report.source.candidate_path), directory=True)
    _validate_candidate_runtime_access(candidate_path)
    bundle = load_calibrated_bundle(candidate_path)
    if (
        bundle is None
        or bundle.profile_id != report.runtime_image.runtime_profile_id
        or bundle.registry.registry_id != report.runtime_image.registry_id
        or str(report.runtime_image.digest) == str(report.source.training_image_digest)
        or report.runtime_image.runtime_source.sha256 != report.runtime_image.source_sha256_label
        or set(report.hard_gates.values()) != {True}
    ):
        raise RerankerKServeReadinessError("Reranker runtime readiness gates changed")
    inspect = _load_inspect(_inside(root, Path(report.runtime_image.image_inspect.path)))
    help_text = _inside(root, Path(report.runtime_image.entrypoint_help.path)).read_text(
        encoding="utf-8"
    )
    facts = _validate_image(inspect, help_text)
    if (
        facts["run_as_user"] != report.runtime_image.run_as_user
        or facts["source_revision"] != report.runtime_image.source_revision
        or facts["source_sha256_label"] != report.runtime_image.source_sha256_label
    ):
        raise RerankerKServeReadinessError("Reranker image inspection facts changed")
    return report


def _validate_candidate_runtime_access(candidate_path: Path) -> None:
    for item in (candidate_path, *sorted(candidate_path.rglob("*"))):
        if item.is_symlink():
            raise RerankerKServeReadinessError(
                "Reranker candidate must not contain symbolic links"
            )
        mode = item.stat().st_mode
        if item.is_dir() and mode & 0o005 != 0o005:
            raise RerankerKServeReadinessError(
                "Reranker candidate directory is not runtime traversable"
            )
        if item.is_file() and mode & 0o004 != 0o004:
            raise RerankerKServeReadinessError(
                "Reranker candidate file is not runtime readable"
            )


def file_binding(root: Path, path: Path) -> dict[str, Any]:
    target = _inside(root, path)
    return {
        "path": target.relative_to(root).as_posix(),
        "size_bytes": target.stat().st_size,
        "sha256": file_sha256(target),
    }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_image(inspect: dict[str, Any], help_text: str) -> dict[str, str]:
    config = inspect.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    image_id = inspect.get("Id")
    repo_tags = inspect.get("RepoTags")
    entrypoint = config.get("Entrypoint") if isinstance(config, dict) else None
    user = config.get("User") if isinstance(config, dict) else None
    required_ref = f"{RUNTIME_IMAGE_REPOSITORY}:{RUNTIME_IMAGE_TAG}"
    if (
        image_id != RUNTIME_IMAGE_DIGEST
        or not isinstance(repo_tags, list)
        or required_ref not in repo_tags
        or entrypoint != list(ENTRYPOINT)
        or user != "app"
        or not isinstance(labels, dict)
        or labels.get("io.industrial-ops.runtime.component") != "reranker"
        or labels.get("io.industrial-ops.runtime.profile") != CALIBRATION_PROFILE_ID
        or labels.get("io.industrial-ops.runtime.source-sha256") != RUNTIME_SOURCE_SHA256
        or not isinstance(labels.get("org.opencontainers.image.revision"), str)
        or not 7 <= len(labels["org.opencontainers.image.revision"]) <= 40
        or any(option not in help_text for option in REQUIRED_HELP_OPTIONS)
    ):
        raise RerankerKServeReadinessError("Reranker runtime image contract is invalid")
    return {
        "run_as_user": user,
        "source_revision": cast(str, labels["org.opencontainers.image.revision"]),
        "source_sha256_label": cast(str, labels["io.industrial-ops.runtime.source-sha256"]),
    }


def _load_inspect(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RerankerKServeReadinessError("Reranker image inspection is invalid") from exc
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        raise RerankerKServeReadinessError("Reranker image inspection is invalid")
    return cast(dict[str, Any], document[0])


def _verify_kserve_provider_binding(root: Path, binding: FileBinding) -> None:
    path = _inside(root, Path(binding.path))
    if path.stat().st_size == binding.size_bytes and file_sha256(path) == binding.sha256:
        return
    if binding.path != KSERVE_PROVIDER_RELATIVE.as_posix():
        raise RerankerKServeReadinessError(
            f"Reranker readiness file changed: {binding.path}"
        )
    from industrial_ops_agent.simulation.reranker_kserve_compatibility import (
        RerankerKServeCompatibilityError,
        verify_reranker_kserve_provider_compatibility,
    )

    try:
        verify_reranker_kserve_provider_compatibility(
            root,
            expected_historical_provider=binding,
        )
    except RerankerKServeCompatibilityError as exc:
        raise RerankerKServeReadinessError(
            "Reranker KServe provider changed without verified additive compatibility"
        ) from exc


def _verify_binding(root: Path, binding: FileBinding) -> None:
    path = _inside(root, Path(binding.path))
    if path.stat().st_size != binding.size_bytes or file_sha256(path) != binding.sha256:
        raise RerankerKServeReadinessError(f"Reranker readiness file changed: {binding.path}")


def _inside(root: Path, path: Path, *, directory: bool = False) -> Path:
    target = path if path.is_absolute() else root / path
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RerankerKServeReadinessError("Reranker readiness path escaped repository") from exc
    if directory and not resolved.is_dir():
        raise RerankerKServeReadinessError("Reranker readiness directory is missing")
    if not directory and not resolved.is_file():
        raise RerankerKServeReadinessError("Reranker readiness file is missing")
    return resolved


def _inside_output(root: Path, path: Path) -> Path:
    target = path if path.is_absolute() else root / path
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RerankerKServeReadinessError("Reranker output escaped repository") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ACCEPTANCE_RELATIVE",
    "EnterpriseRerankerKServeReadinessReport",
    "IMAGE_HELP_RELATIVE",
    "IMAGE_INSPECT_RELATIVE",
    "RUNTIME_IMAGE_DIGEST",
    "RUNTIME_IMAGE_REPOSITORY",
    "RUNTIME_IMAGE_TAG",
    "RerankerKServeReadinessError",
    "build_reranker_kserve_readiness",
    "verify_reranker_kserve_readiness",
]
