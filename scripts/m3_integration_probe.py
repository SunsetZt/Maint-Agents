#!/usr/bin/env python3
"""Durable integration driver for the real M3 offline service chain."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from urllib.request import Request, urlopen
from uuid import uuid4

from minio import Minio
from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.config import get_settings
from industrial_ops_agent.data_governance.dlp import PresidioDlpProcessor
from industrial_ops_agent.data_governance.service import DataGovernanceService
from industrial_ops_agent.data_pipeline.lineage import OpenLineageEmitter
from industrial_ops_agent.data_pipeline.service import DatasetPipelineService
from industrial_ops_agent.data_pipeline.storage import MinioDatasetStore
from industrial_ops_agent.labeling.label_studio import LabelStudioHttpAdapter
from industrial_ops_agent.labeling.service import LabelingService
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ActionProposalRecord,
    AnnotationTaskRecord,
    AssetRecord,
    EvidenceBundleRecord,
    FeedbackCandidateRecord,
    IncidentDraftRecord,
    IncidentRecord,
    MediaObjectRecord,
    PartReservationRecord,
    TenantRecord,
    WorkOrderCompletionRecord,
    WorkOrderRecord,
    WorkOrderVerificationRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.secrets import (
    SecretName,
    SecretSource,
    SecretValue,
    build_secret_provider,
)
from industrial_ops_agent.security_audit import InMemorySecurityAuditSink, SecurityAuditor
from industrial_ops_agent.workorders.service import WorkOrderService


def _identity(
    tenant_id: str,
    subject_id: str,
    role: Role,
    asset_id: str,
) -> IdentityContext:
    now = datetime.now(UTC)
    return IdentityContext(
        subject_id=subject_id,
        oidc_subject=f"integration-{subject_id}",
        tenant_id=tenant_id,
        roles=frozenset({role}),
        asset_ids=frozenset({asset_id}),
        site_ids=frozenset(),
        issued_at=now,
        expires_at=now + timedelta(hours=1),
    )


def _seed_verified_work_order(
    database: Database,
    *,
    tenant_id: str,
    asset_id: str,
    work_order_id: str,
    suffix: str,
    now: datetime,
) -> None:
    with Session(database.engine) as session:
        if session.get(TenantRecord, tenant_id) is None:
            session.add(TenantRecord(id=tenant_id, status="active"))
            session.commit()
    context = TenantContext(tenant_id, "m3-integration-seed")
    with database.transaction(context) as session:
        session.add(
            AssetRecord(
                tenant_id=tenant_id,
                asset_id=asset_id,
                source_system="m3-integration",
                source_record_id=f"pump-{suffix}",
                as_of=now,
                model_code="PUMP-X100",
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            IncidentDraftRecord(
                tenant_id=tenant_id,
                draft_id=f"draft-{suffix}",
                asset_id=asset_id,
                author_subject_id="integration-engineer",
                description="联系人张三，手机号13800138000，泵组出现 E-42 告警。",
                version=2,
                status="SUBMITTED",
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            MediaObjectRecord(
                tenant_id=tenant_id,
                media_id=f"media-{suffix}",
                draft_id=f"draft-{suffix}",
                quarantine_key=f"tenant/{tenant_id}/quarantine/{suffix}",
                clean_key=f"tenant/{tenant_id}/clean/{suffix}",
                content_hash=sha256(suffix.encode()).hexdigest(),
                declared_mime="image/jpeg",
                detected_mime="image/jpeg",
                size_bytes=128,
                scan_state="CLEAN",
                scan_version=2,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            EvidenceBundleRecord(
                tenant_id=tenant_id,
                bundle_id=f"bundle-{suffix}",
                draft_id=f"draft-{suffix}",
                asset_id=asset_id,
                media_id=f"media-{suffix}",
                source_sha256=sha256(f"bundle-{suffix}".encode()).hexdigest(),
                source_type="image/jpeg",
                processor_versions={"ocr": "integration-v1", "vlm": "integration-v1"},
                status="CONFIRMED",
                version=2,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            IncidentRecord(
                tenant_id=tenant_id,
                incident_id=f"incident-{suffix}",
                asset_id=asset_id,
                source_draft_id=f"draft-{suffix}",
                evidence_bundle_id=f"bundle-{suffix}",
                reporter_subject_id="integration-engineer",
                description="联系人张三，手机号13800138000，泵组出现 E-42 告警。",
                status="RESOLVED",
                version=4,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            ActionProposalRecord(
                tenant_id=tenant_id,
                proposal_id=f"proposal-{suffix}",
                incident_id=f"incident-{suffix}",
                initiator_subject_id="integration-engineer",
                tool_id="parts.reserve",
                tool_version="integration-v1",
                risk_tier="L2",
                parameters={"part_number": "FILTER-X100", "quantity": 1},
                parameters_hash=sha256(f"proposal-{suffix}".encode()).hexdigest(),
                resource_version=4,
                operation_id=f"operation-{suffix}",
                status="EXECUTED",
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            PartReservationRecord(
                tenant_id=tenant_id,
                reservation_id=f"reservation-{suffix}",
                operation_id=f"reservation-operation-{suffix}",
                incident_id=f"incident-{suffix}",
                part_number="FILTER-X100",
                quantity=1,
                status="RESERVED",
                source="m3-integration",
                source_record_id=f"reservation-source-{suffix}",
                as_of=now,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            WorkOrderRecord(
                tenant_id=tenant_id,
                work_order_id=work_order_id,
                incident_id=f"incident-{suffix}",
                proposal_id=f"proposal-{suffix}",
                reservation_id=f"reservation-{suffix}",
                status="VERIFIED",
                assigned_subject_id="integration-engineer",
                version=8,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            WorkOrderCompletionRecord(
                tenant_id=tenant_id,
                completion_id=f"completion-{suffix}",
                work_order_id=work_order_id,
                completed_by_subject_id="integration-engineer",
                round_number=1,
                field_entry_sequence_start=None,
                field_entry_sequence_end=None,
                root_cause="设备序列号 SN-9988 的入口滤芯堵塞。",
                actions=["更换入口滤芯", "复测压差"],
                evidence_ids=[f"bundle-{suffix}"],
                part_reservation_ids=[f"reservation-{suffix}"],
                cost_amount="125.00",
                customer_confirmation="张三通过 13800138000 确认完成",
                completed_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            WorkOrderVerificationRecord(
                tenant_id=tenant_id,
                verification_id=f"verification-{suffix}",
                work_order_id=work_order_id,
                completion_id=f"completion-{suffix}",
                round_number=1,
                verifier_subject_id="integration-expert",
                passed=True,
                reason="独立复核通过",
                verified_at=now,
                created_at=now,
                updated_at=now,
            )
        )


def _wait_candidate(
    database: Database,
    tenant_id: str,
    work_order_id: str,
) -> FeedbackCandidateRecord:
    context = TenantContext(tenant_id, "m3-integration-observer")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        with database.transaction(context) as session:
            candidate = session.scalar(
                select(FeedbackCandidateRecord).where(
                    FeedbackCandidateRecord.tenant_id == tenant_id,
                    FeedbackCandidateRecord.work_order_id == work_order_id,
                )
            )
            if candidate is not None:
                return candidate
        time.sleep(2)
    raise RuntimeError("Debezium/Kafka event did not produce FeedbackCandidate")


def _create_annotation(base_url: str, token: str, external_task_id: str) -> None:
    body = json.dumps(
        {
            "result": [
                {
                    "from_name": "root_cause",
                    "to_name": "content",
                    "type": "choices",
                    "value": {"choices": ["restricted_inlet_filter"]},
                }
            ],
            "was_cancelled": False,
            "ground_truth": True,
        }
    ).encode()
    request = Request(
        f"{base_url.rstrip('/')}/api/tasks/{external_task_id}/annotations/",
        data=body,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=15) as response:  # noqa: S310 - integration-local URL
        if response.status not in {200, 201}:
            raise RuntimeError("Label Studio annotation creation failed")


def main() -> int:
    settings = get_settings()
    secrets = build_secret_provider(settings)
    database = Database(secrets.get(SecretName.DATABASE_URL).reveal())
    suffix = uuid4().hex[:12]
    tenant_id = f"tenant-m3-{suffix}"
    asset_id = f"asset-m3-{suffix}"
    work_order_id = f"work-order-m3-{suffix}"
    now = datetime.now(UTC)
    auditor = SecurityAuditor(InMemorySecurityAuditSink(), hash_key=b"m3-integration-audit")
    authorizer = Authorizer(auditor)
    closer = _identity(
        tenant_id,
        "integration-dispatcher",
        Role.AFTER_SALES_ENGINEER,
        asset_id,
    )
    steward = _identity(tenant_id, "integration-steward", Role.DATA_STEWARD, asset_id)
    annotation_admin = _identity(
        tenant_id,
        "integration-annotation-admin",
        Role.ANNOTATION_ADMIN,
        asset_id,
    )
    try:
        _seed_verified_work_order(
            database,
            tenant_id=tenant_id,
            asset_id=asset_id,
            work_order_id=work_order_id,
            suffix=suffix,
            now=now,
        )
        closed = WorkOrderService(database, authorizer).close(
            closer,
            work_order_id,
            expected_version=8,
            request_id=f"m3-integration-{suffix}",
        )
        if closed.status != "CLOSED":
            raise RuntimeError("real WorkOrder close did not complete")
        candidate = _wait_candidate(database, tenant_id, work_order_id)

        dlp = PresidioDlpProcessor()
        governance = DataGovernanceService(database, authorizer, dlp)
        governance.decide_eligibility(
            steward,
            candidate.candidate_id,
            expected_version=candidate.version,
            purpose="industrial_model_improvement",
            allow_training=True,
            consent_basis="enterprise-maintenance-training-contract",
            license_status="APPROVED",
            retention_until=now + timedelta(days=365),
            policy_version="industrial-data-eligibility-v1",
            request_id=f"m3-eligibility-{suffix}",
        )
        governance.run_dlp(
            steward,
            candidate.candidate_id,
            expected_version=candidate.version + 1,
            policy_version="m3-presidio-zh-industrial-v1",
            request_id=f"m3-dlp-{suffix}",
        )

        label_token = os.environ["IOAP_LABEL_STUDIO_API_TOKEN"]
        label_url = settings.label_studio_url
        adapter = LabelStudioHttpAdapter(
            base_url=label_url,
            api_token=SecretValue(label_token, SecretSource.ENVIRONMENT),
        )
        labeling = LabelingService(database, authorizer, adapter, dlp)
        task = labeling.create_task(
            steward,
            candidate.candidate_id,
            expected_version=candidate.version + 2,
            project_id="1",
            schema_version="root-cause-label-v1",
            risk_level="STANDARD",
            idempotency_key=f"m3-label-{suffix}",
            request_id=f"m3-label-create-{suffix}",
        )
        _create_annotation(label_url, label_token, task.external_task_id)
        approved = labeling.sync_task(
            annotation_admin,
            task.task_id,
            expected_version=task.version,
            request_id=f"m3-label-sync-{suffix}",
        )
        if approved.status != "APPROVED":
            raise RuntimeError("Label Studio review did not approve the candidate")

        minio = Minio(
            settings.minio_endpoint,
            access_key=secrets.get(SecretName.MINIO_ACCESS_KEY).reveal(),
            secret_key=secrets.get(SecretName.MINIO_SECRET_KEY).reveal(),
            secure=settings.minio_secure,
        )
        pipeline = DatasetPipelineService(
            database,
            MinioDatasetStore(minio, settings.dataset_bucket),
            OpenLineageEmitter(
                url=settings.openlineage_url,
                namespace=settings.openlineage_namespace,
            ),
            code_version="m3-integration",
        )
        result = pipeline.build_snapshot(
            TenantContext(tenant_id, "m3-integration-pipeline"),
            window_start=now - timedelta(minutes=5),
            window_end=datetime.now(UTC) + timedelta(minutes=5),
            engine="spark",
        )
        if not result.training_eligible or result.status != "CANDIDATE":
            raise RuntimeError("governed DatasetSnapshot was not published")
        with database.transaction(TenantContext(tenant_id, "m3-integration-observer")) as session:
            task_row = session.scalar(
                select(AnnotationTaskRecord).where(
                    AnnotationTaskRecord.tenant_id == tenant_id,
                    AnnotationTaskRecord.task_id == task.task_id,
                )
            )
            if task_row is None or task_row.status != "APPROVED":
                raise RuntimeError("annotation approval is not durable")
        print(
            "M3_INTEGRATION_PROBE_OK "
            + json.dumps(
                {
                    "candidate_id": candidate.candidate_id,
                    "snapshot_id": result.snapshot_id,
                    "run_id": result.run_id,
                    "work_order_id": work_order_id,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        database.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
