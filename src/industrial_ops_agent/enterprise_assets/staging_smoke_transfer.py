"""Authorized, score-preserving transfer of a frozen STAGING diagnosis evaluation.

Both identities must be authenticated by the API. No source query runs in the
target tenant context, and no approval or inference record is imported.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256

from sqlalchemy import select

from industrial_ops_agent.auth.policy import Action, ResourceContext
from industrial_ops_agent.data_pipeline.storage import ImmutableObjectExists
from industrial_ops_agent.persistence.model_import_integrity import enterprise_model_import_hash
from industrial_ops_agent.persistence.models import (
    CurationRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
    EnterpriseModelAssetImportRecord,
    EvaluationArtifactRecord,
    EvaluationPolicyRecord,
    EvaluationSuiteRecord,
    ModelEvaluationJobRecord,
    ModelEvaluationRunRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.releases.staging_smoke import build_staging_smoke_admission

CLASSIFICATION = "STAGING_DIAGNOSIS_SMOKE_TRANSFER"
_MODELS = (
    CurationRunRecord,
    DatasetSnapshotRecord,
    DatasetArtifactRecord,
    TrainingExperimentRecord,
    TrainingArtifactRecord,
    EvaluationSuiteRecord,
    EvaluationPolicyRecord,
    ModelEvaluationRunRecord,
    ModelEvaluationJobRecord,
    EvaluationArtifactRecord,
)


def _bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.isoformat(),
        allow_nan=False,
    ).encode()


def _hash(value):
    return sha256(_bytes(value)).hexdigest()


def _document(record):
    return {
        column.name: deepcopy(getattr(record, column.name)) for column in record.__table__.columns
    }


class StagingSmokeTransferService:
    def __init__(self, database, authorizer, store):
        self.database, self.authorizer, self.store = database, authorizer, store

    def transfer(
        self, source, target, *, evaluation_id, prompt_bundle_id, idempotency_key, request_id
    ):
        for identity in (source, target):
            self.authorizer.require(
                identity,
                Action.IMPORT_ENTERPRISE_MODEL_ASSET,
                ResourceContext(identity.tenant_id, "staging-smoke-transfer"),
                request_id=request_id,
            )
        if source.tenant_id == target.tenant_id:
            raise ValueError("staging_smoke_transfer_requires_different_tenants")
        intent = {
            "source_tenant": source.tenant_id,
            "target_tenant": target.tenant_id,
            "evaluation_id": evaluation_id,
            "prompt_bundle_id": prompt_bundle_id,
        }
        intent_hash = _hash(intent)
        import_id = "smoke-import-" + intent_hash[:32]
        with self.database.transaction(source.tenant_context) as session:
            admission = build_staging_smoke_admission(
                session,
                source.tenant_id,
                evaluation_id=evaluation_id,
                prompt_bundle_id=prompt_bundle_id,
                target_environment="STAGING",
            )
            records = self._collect(session, source.tenant_id, admission)
        original = {
            "schema": "staging-smoke-transfer/v1",
            **intent,
            "admission": admission,
            "records": records,
            "exported_by": source.subject_id,
            "production_claim": False,
        }
        proof_hash = _hash(original)
        with self.database.transaction(target.tenant_context) as session:
            old = session.scalar(
                select(EnterpriseModelAssetImportRecord).where(
                    EnterpriseModelAssetImportRecord.tenant_id == target.tenant_id,
                    EnterpriseModelAssetImportRecord.requested_by_subject_id == target.subject_id,
                    EnterpriseModelAssetImportRecord.idempotency_key == idempotency_key,
                )
            )
            if old is None:
                old = session.get(EnterpriseModelAssetImportRecord, import_id)
            if old is not None:
                if (
                    old.tenant_id != target.tenant_id
                    or old.request_hash != intent_hash
                    or old.status != "ACTIVE"
                    or old.source_evidence_chain_sha256 != proof_hash
                    or old.import_hash != enterprise_model_import_hash(old)
                ):
                    raise ValueError("staging_smoke_transfer_idempotency_conflict")
                return self._view(old)

        # Copy only explicitly collected records and checksum-verified objects.
        # Original dataset manifests remain source provenance, not trainable target data.
        ids = {}
        for model in _MODELS:
            primary = next(iter(model.__table__.primary_key)).name
            for row in records[model.__tablename__]:
                old_id = row[primary]
                ids[old_id] = "smoke-" + _hash([target.tenant_id, intent_hash, old_id])[:40]
        root = f"staging-smoke-imports/{target.tenant_id}/{import_id}"
        object_map, hashes, object_checks, total = {}, {}, {}, 0
        for model in _MODELS:
            for row in records[model.__tablename__]:
                objects = []
                if "object_key" in row:
                    objects.append((row["object_key"], row["content_hash"], row["size_bytes"]))
                if model is DatasetSnapshotRecord:
                    objects.append((row["manifest_key"], row["manifest_hash"], None))
                    if row["quality_report_key"]:
                        objects.append((row["quality_report_key"], None, None))
                for key, expected_hash, size in objects:
                    if (
                        not isinstance(key, str)
                        or not key
                        or key.startswith("/")
                        or ".." in key.split("/")
                    ):
                        raise ValueError("staging_smoke_source_object_invalid")
                    if key in object_map:
                        digest, actual_size = object_checks[key]
                        if (expected_hash and digest != expected_hash.removeprefix("sha256:")) or (
                            size is not None and size != actual_size
                        ):
                            raise ValueError("staging_smoke_source_object_changed")
                        continue
                    blob = self.store.get_bytes(key)
                    total += len(blob)
                    if total > 512 * 1024 * 1024:
                        raise ValueError("staging_smoke_transfer_too_large")
                    digest = sha256(blob).hexdigest()
                    if (expected_hash and digest != expected_hash.removeprefix("sha256:")) or (
                        size is not None and size != len(blob)
                    ):
                        raise ValueError("staging_smoke_source_object_changed")
                    # Different source objects may have identical bytes (for
                    # example empty dataset splits). Artifact keys are unique
                    # within a tenant: preserve source identity as well as hash.
                    destination = root + "/objects/" + _hash(key)[:32] + "/" + digest
                    self._put(destination, blob)
                    object_map[key], hashes[destination] = destination, digest
                    object_checks[key] = digest, len(blob)
        proof_key = root + "/source-records.json"
        self._put(proof_key, _bytes(original))
        hashes[proof_key] = proof_hash
        now = datetime.now(UTC)
        with self.database.transaction(target.tenant_context) as session:
            if session.get(EnterpriseModelAssetImportRecord, import_id) is not None:
                raise ValueError("staging_smoke_transfer_concurrent_retry")
            for model in _MODELS:
                primary = next(iter(model.__table__.primary_key)).name
                rows = records[model.__tablename__]
                # Parent snapshots are exported before their descendants.
                for original_row in rows:
                    row = deepcopy(original_row)
                    row[primary] = ids[row[primary]]
                    row["tenant_id"] = target.tenant_id
                    for column in model.__table__.columns:
                        value = row[column.name]
                        if column.foreign_keys and column.name != "tenant_id" and value is not None:
                            if value not in ids:
                                raise ValueError("staging_smoke_external_record_reference")
                            row[column.name] = ids[value]
                    for key in ("object_key", "manifest_key", "quality_report_key"):
                        if row.get(key) is not None:
                            row[key] = object_map[row[key]]
                    if "idempotency_key" in row:
                        row["idempotency_key"] = import_id + ":" + row[primary]
                    if model is DatasetSnapshotRecord:
                        row["training_eligible"] = False
                        row["source_work_order_ids"] = []
                    if model is ModelEvaluationRunRecord:
                        row["comparison_context"] = {
                            **row["comparison_context"],
                            "staging_smoke_import_id": import_id,
                        }
                    session.add(model(**row))
                    session.flush()
            receipt = EnterpriseModelAssetImportRecord(
                tenant_id=target.tenant_id,
                import_id=import_id,
                component="LLM",
                source_candidate_experiment_id=admission["candidate_experiment_id"],
                imported_candidate_experiment_id=ids[admission["candidate_experiment_id"]],
                imported_evaluation_id=ids[evaluation_id],
                imported_suite_id=ids[admission["suite_id"]],
                source_paths_json=sorted(hashes),
                source_file_hashes_json=hashes,
                source_evidence_chain_sha256=proof_hash,
                source_classification="PROJECT_GENERATED",
                source_decision=admission["decision"],
                operational_classification=CLASSIFICATION,
                actual_sample_count=14,
                source_artifact_size_recorded=True,
                release_scope="STAGING_ONLY",
                imported_records_json={
                    "candidate": ids[admission["candidate_experiment_id"]],
                    "evaluation": ids[evaluation_id],
                    "suite": ids[admission["suite_id"]],
                    "source_tenant_id": source.tenant_id,
                    "exported_by": source.subject_id,
                    "source_scores": {
                        key: admission[key]
                        for key in (
                            "candidate_score",
                            "baseline_score",
                            "quality_delta",
                            "ci_low",
                            "ci_high",
                            "hard_gate_results",
                            "slice_metrics",
                            "report_hash",
                        )
                    },
                },
                idempotency_key=idempotency_key,
                request_hash=intent_hash,
                import_hash="",
                status="ACTIVE",
                requested_by_subject_id=target.subject_id,
                version=1,
                created_at=now,
                updated_at=now,
            )
            receipt.import_hash = enterprise_model_import_hash(receipt)
            session.add(receipt)
            session.flush()
            # Recheck the complete target graph without changing its scores/decision.
            build_staging_smoke_admission(
                session,
                target.tenant_id,
                evaluation_id=receipt.imported_evaluation_id,
                prompt_bundle_id=prompt_bundle_id,
                target_environment="STAGING",
            )
            return self._view(receipt)

    def _put(self, key, content):
        try:
            self.store.put_bytes(key, content)
        except ImmutableObjectExists:
            if self.store.get_bytes(key) != content:
                raise ValueError("staging_smoke_import_object_conflict") from None

    @staticmethod
    def _view(record):
        return {
            "import_id": record.import_id,
            "evaluation_id": record.imported_evaluation_id,
            "candidate_experiment_id": record.imported_candidate_experiment_id,
            "suite_id": record.imported_suite_id,
            "decision": record.source_decision,
            "sample_count": record.actual_sample_count,
            "release_scope": record.release_scope,
            "import_hash": record.import_hash,
            "approvals_imported": False,
            "training_authorization_inherited": False,
            "business_binding_changed": False,
        }

    @staticmethod
    def _collect(session, tenant_id, admission):
        def rows(model, condition):
            return list(
                session.scalars(select(model).where(model.tenant_id == tenant_id, condition))
            )

        experiments = rows(
            TrainingExperimentRecord,
            TrainingExperimentRecord.experiment_id.in_(
                [admission["candidate_experiment_id"], admission["baseline_experiment_id"]]
            ),
        )
        snapshots, seen = [], set()

        def snapshot(identifier):
            if identifier in seen:
                return
            if len(seen) >= 16:
                raise ValueError("staging_smoke_snapshot_graph_too_large")
            seen.add(identifier)
            found = rows(DatasetSnapshotRecord, DatasetSnapshotRecord.snapshot_id == identifier)
            if len(found) != 1:
                raise ValueError("staging_smoke_snapshot_not_visible")
            item = found[0]
            if item.base_snapshot_id:
                snapshot(item.base_snapshot_id)
            snapshots.append(item)

        for identifier in [admission["snapshot_id"], *(e.dataset_snapshot_id for e in experiments)]:
            snapshot(identifier)
        for experiment in experiments:
            training = next(s for s in snapshots if s.snapshot_id == experiment.dataset_snapshot_id)
            if (
                training.status != "CANDIDATE"
                or training.training_eligible is not True
                or training.lineage_status != "CONFIRMED"
                or training.manifest_hash != experiment.dataset_manifest_hash
            ):
                raise ValueError("staging_smoke_source_training_provenance_invalid")
        groups = [
            rows(CurationRunRecord, CurationRunRecord.run_id.in_([s.run_id for s in snapshots])),
            snapshots,
            rows(DatasetArtifactRecord, DatasetArtifactRecord.snapshot_id.in_(seen)),
            experiments,
            rows(
                TrainingArtifactRecord,
                (TrainingArtifactRecord.experiment_id == admission["candidate_experiment_id"])
                & (TrainingArtifactRecord.kind == "adapter_bundle"),
            ),
            rows(EvaluationSuiteRecord, EvaluationSuiteRecord.suite_id == admission["suite_id"]),
            rows(
                EvaluationPolicyRecord, EvaluationPolicyRecord.policy_id == admission["policy_id"]
            ),
            rows(
                ModelEvaluationRunRecord,
                ModelEvaluationRunRecord.evaluation_id == admission["evaluation_id"],
            ),
            rows(ModelEvaluationJobRecord, ModelEvaluationJobRecord.job_id == admission["job_id"]),
            rows(
                EvaluationArtifactRecord,
                EvaluationArtifactRecord.evaluation_id == admission["evaluation_id"],
            ),
        ]
        return {
            model.__tablename__: sorted(
                (_document(r) for r in group),
                key=lambda r: str(r[next(iter(model.__table__.primary_key)).name]),
            )
            if model is not DatasetSnapshotRecord
            else [_document(r) for r in group]
            for model, group in zip(_MODELS, groups, strict=True)
        }
