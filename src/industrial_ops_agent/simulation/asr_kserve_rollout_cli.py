"""Build and verify the low-resource ASR KServe rollout readiness package."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from industrial_ops_agent.simulation.asr_enterprise_value_lab import (
    AsrEnterpriseValueReport,
    verify_asr_enterprise_value,
)
from industrial_ops_agent.simulation.asr_kserve_rollout import (
    CANDIDATE_SERVICE,
    GATEWAY_NAME,
    HOSTNAME,
    NAMESPACE,
    OUTPUT_RELATIVE,
    PROBE_AUDIO_SHA256,
    PROBE_AUDIO_SIZE_BYTES,
    PROBE_CASE_ID,
    PROBE_DURATION_SECONDS,
    PROBE_SAMPLE_COUNT,
    PROBE_SAMPLE_RATE,
    PROBE_SOURCE_ID,
    PROBE_TRANSCRIPT,
    PROBE_TRANSCRIPT_SHA256,
    PROXY_SOURCE_RELATIVE,
    ROUTE_NAME,
    STABLE_SERVICE,
    WORKER_SOURCE_RELATIVE,
    file_binding,
    finalize_readiness,
    inference_service_document,
    proxy_config_map_document,
    release_probe_document,
    route_document,
    verify_asr_kserve_readiness,
    write_readiness_report,
)


def build_readiness_package(
    repo_root: Path,
    *,
    host_gateway: str,
    host_port: int,
) -> Path:
    root = repo_root.resolve(strict=True)
    source = verify_asr_enterprise_value(root)
    output = root / OUTPUT_RELATIVE
    kubernetes = output / "kubernetes"
    kubernetes.mkdir(parents=True, exist_ok=True)
    acceptance_path = root / _latest_acceptance_path(root)
    probe_audio_path = output / "release-probe.flac"
    _write_probe_audio(root, source, probe_audio_path)
    probe_manifest_path = output / "release-probe.json"
    _write_json(probe_manifest_path, release_probe_document())
    proxy_path = (root / PROXY_SOURCE_RELATIVE).resolve(strict=True)
    worker_path = (root / WORKER_SOURCE_RELATIVE).resolve(strict=True)
    proxy_source = proxy_path.read_text(encoding="utf-8")
    config_map = proxy_config_map_document(proxy_source)
    stable = inference_service_document(
        variant="stable",
        host_gateway=host_gateway,
        host_port=host_port,
        proxy_image_digest=source.base_model.image_digest,
    )
    candidate = inference_service_document(
        variant="candidate",
        host_gateway=host_gateway,
        host_port=host_port,
        proxy_image_digest=source.base_model.image_digest,
    )
    config_path = kubernetes / "proxy-configmap.json"
    stable_path = kubernetes / "stable-inferenceservice.json"
    candidate_path = kubernetes / "candidate-inferenceservice.json"
    _write_json(config_path, config_map)
    _write_json(stable_path, stable)
    _write_json(candidate_path, candidate)
    route_paths: dict[str, Path] = {}
    for stage in ("SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"):
        route_path = kubernetes / f"route-{stage.lower()}.json"
        _write_json(route_path, route_document(cast(Any, stage)))
        route_paths[stage] = route_path
    unsigned: dict[str, Any] = {
        "schema_version": "enterprise-asr-kserve-rollout-readiness/v1",
        "classification": "LOCAL_STAGING_PROJECT_AUTHORIZED",
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "ASR_MODEL_RELEASE_KSERVE_ROLLOUT_READY",
        "source": {
            "acceptance": file_binding(root, acceptance_path),
            "run_id": source.run_id,
            "evidence_chain_sha256": source.evidence_chain_sha256,
            "model_id": source.base_model.model_id,
            "model_revision": source.base_model.revision,
            "image": source.base_model.image,
            "image_digest": source.base_model.image_digest,
            "adapter_bundle_sha256": source.adapter.bundle_sha256,
            "actual_gpu_training": True,
            "frozen_gold_passed": True,
        },
        "probe": {
            "manifest": file_binding(root, probe_manifest_path),
            "audio": file_binding(root, probe_audio_path),
            "case_id": PROBE_CASE_ID,
            "source_id": PROBE_SOURCE_ID,
            "transcript_sha256": PROBE_TRANSCRIPT_SHA256,
            "sampling_rate": PROBE_SAMPLE_RATE,
            "sample_count": PROBE_SAMPLE_COUNT,
            "duration_seconds": PROBE_DURATION_SECONDS,
            "formal_gold": False,
            "formal_gold_reused": False,
            "excluded_from_training_and_model_selection": True,
            "source_id_disjoint_from_governed_splits": True,
            "project_enterprise_use_authorized": True,
        },
        "runtime": {
            "worker_source": file_binding(root, worker_path),
            "proxy_source": file_binding(root, proxy_path),
            "endpoint_path": "/v1/asr/transcribe",
            "stable_disables_adapter": True,
            "candidate_enables_adapter": True,
            "request_media_digest_required": True,
            "runtime_cache_variant_bound": True,
            "exactly_one_gpu_required": True,
            "cpu_threads": 2,
            "memory_limit_bytes": 4 * 1024**3,
            "actual_rollout_execution_completed": False,
        },
        "kubernetes": {
            "namespace": NAMESPACE,
            "gateway_name": GATEWAY_NAME,
            "hostname": HOSTNAME,
            "route_name": ROUTE_NAME,
            "stable_inference_service": STABLE_SERVICE,
            "candidate_inference_service": CANDIDATE_SERVICE,
            "config_map": file_binding(root, config_path),
            "stable_service": file_binding(root, stable_path),
            "candidate_service": file_binding(root, candidate_path),
            "routes": {
                stage: file_binding(root, path) for stage, path in route_paths.items()
            },
            "hardened_container_security_context": True,
            "proxy_has_no_gpu_request": True,
            "min_replicas_per_variant": 1,
            "max_replicas_per_variant": 1,
        },
        "model_release": {
            "component": "ASR",
            "method": "ASR",
            "artifact_kind": "asr_adapter_bundle",
            "target_profile": "ASR_COMPONENT",
            "target_environment": "STAGING",
            "import_api": "/api/v1/enterprise-assets/runtime-bindings/ASR/imports",
            "preview_api": (
                "/api/v1/enterprise-assets/runtime-bindings/ASR/release-draft-preview"
            ),
            "existing_approval_fsm_reused": True,
            "existing_deployment_fsm_reused": True,
            "shadow_canary_rollback_stages": [
                "SHADOW",
                "CANARY_5",
                "CANARY_25",
                "ROLLED_BACK",
            ],
            "formal_model_release_created": False,
            "shadow_claim": False,
            "canary_claim": False,
            "rollback_claim": False,
        },
        "hard_gates": {
            "authoritative_training_evidence": True,
            "release_probe_independence": True,
            "runtime_source_integrity": True,
            "stable_candidate_isolation": True,
            "hardened_proxy_security": True,
            "model_release_fsm_reuse": True,
            "no_unexecuted_rollout_claim": True,
        },
    }
    report = finalize_readiness(unsigned)
    readiness_path = output / "readiness.json"
    write_readiness_report(report, readiness_path)
    verify_asr_kserve_readiness(root, readiness_path)
    return readiness_path


def _write_probe_audio(
    root: Path,
    source: AsrEnterpriseValueReport,
    output_path: Path,
) -> None:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
        import soundfile  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - declared project dependencies
        raise RuntimeError("ASR release probe dependencies are not installed") from exc
    dataset_path = (root / source.dataset.source_file.path).resolve(strict=True)
    table = parquet.read_table(dataset_path, columns=["id", "text", "audio", "chapter_id"])
    rows = [item for item in table.to_pylist() if item.get("id") == PROBE_SOURCE_ID]
    if len(rows) != 1:
        raise RuntimeError("ASR release probe source row is missing or duplicated")
    row = rows[0]
    audio_value = row.get("audio")
    if (
        row.get("chapter_id") != 135031
        or row.get("text") != PROBE_TRANSCRIPT
        or not isinstance(audio_value, dict)
        or not isinstance(audio_value.get("bytes"), bytes)
    ):
        raise RuntimeError("ASR release probe source row changed")
    audio = audio_value["bytes"]
    info = soundfile.info(BytesIO(audio))
    if (
        len(audio) != PROBE_AUDIO_SIZE_BYTES
        or sha256(audio).hexdigest() != PROBE_AUDIO_SHA256
        or info.samplerate != PROBE_SAMPLE_RATE
        or info.frames != PROBE_SAMPLE_COUNT
        or abs(float(info.duration) - PROBE_DURATION_SECONDS) > 1e-9
        or sha256(PROBE_TRANSCRIPT.encode()).hexdigest() != PROBE_TRANSCRIPT_SHA256
    ):
        raise RuntimeError("ASR release probe media binding changed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_bytes(audio)
    temporary.replace(output_path)


def _latest_acceptance_path(root: Path) -> Path:
    pointer_path = root / "artifacts/m7-asr-enterprise-value-lab/latest.json"
    value = json.loads(pointer_path.read_text(encoding="utf-8"))
    report = value.get("report") if isinstance(value, dict) else None
    if not isinstance(report, str) or not report:
        raise RuntimeError("ASR latest pointer is invalid")
    target = Path(report)
    if target.is_absolute():
        raise RuntimeError("ASR latest pointer must be relative")
    resolved = (root / target).resolve(strict=True)
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("ASR latest pointer escaped repository root") from exc


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-asr-kserve-rollout")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--repo-root", type=Path, default=Path.cwd())
    plan.add_argument("--host-gateway", default="172.19.0.1")
    plan.add_argument("--host-port", type=int, default=18094)
    verify = commands.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "plan":
        path = build_readiness_package(
            root,
            host_gateway=args.host_gateway,
            host_port=args.host_port,
        )
    else:
        path = root / OUTPUT_RELATIVE / "readiness.json"
    report = verify_asr_kserve_readiness(root, path)
    print(report.status)
    print(report.classification)
    print(f"ACTUAL_ROLLOUT_EXECUTION_COMPLETED={report.runtime.actual_rollout_execution_completed}")
    print(f"FORMAL_MODEL_RELEASE_CREATED={report.model_release.formal_model_release_created}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


if __name__ == "__main__":
    run()
