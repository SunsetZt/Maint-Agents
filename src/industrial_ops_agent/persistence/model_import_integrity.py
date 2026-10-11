"""Canonical integrity contract for enterprise project model imports."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Protocol


class EnterpriseModelImportLike(Protocol):
    tenant_id: str
    component: str
    source_candidate_experiment_id: str
    imported_candidate_experiment_id: str
    imported_evaluation_id: str
    imported_suite_id: str
    source_paths_json: list[str]
    source_file_hashes_json: dict[str, str]
    source_evidence_chain_sha256: str
    source_classification: str
    source_decision: str
    operational_classification: str
    actual_sample_count: int
    source_artifact_size_recorded: bool
    release_scope: str
    imported_records_json: dict[str, str]
    request_hash: str
    status: str


def enterprise_model_import_hash(record: EnterpriseModelImportLike) -> str:
    payload = {
        "schema_version": "enterprise-model-asset-import/v1",
        "tenant_id": record.tenant_id,
        "component": record.component,
        "source_candidate_experiment_id": record.source_candidate_experiment_id,
        "imported_candidate_experiment_id": record.imported_candidate_experiment_id,
        "imported_evaluation_id": record.imported_evaluation_id,
        "imported_suite_id": record.imported_suite_id,
        "source_paths": record.source_paths_json,
        "source_file_hashes": record.source_file_hashes_json,
        "source_evidence_chain_sha256": record.source_evidence_chain_sha256,
        "source_classification": record.source_classification,
        "source_decision": record.source_decision,
        "operational_classification": record.operational_classification,
        "actual_sample_count": record.actual_sample_count,
        "source_artifact_size_recorded": record.source_artifact_size_recorded,
        "release_scope": record.release_scope,
        "imported_records": record.imported_records_json,
        "request_hash": record.request_hash,
        "status": record.status,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()
