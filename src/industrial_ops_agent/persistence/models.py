"""SQLAlchemy model contract for the tenant-scoped M1/M2 data plane."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class TenantRecord(TimestampMixin, Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class TenantScopedMixin(TimestampMixin):
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)


class SubjectRecord(TenantScopedMixin, Base):
    __tablename__ = "subjects"
    __table_args__ = (
        UniqueConstraint("tenant_id", "oidc_issuer", "oidc_subject", name="uq_subject_oidc"),
    )

    subject_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    oidc_issuer: Mapped[str] = mapped_column(String(512), nullable=False)
    oidc_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class SubjectRoleRecord(TenantScopedMixin, Base):
    __tablename__ = "subject_roles"

    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(64), primary_key=True)


class AssetRecord(TenantScopedMixin, Base):
    __tablename__ = "assets"
    __table_args__ = (
        UniqueConstraint("tenant_id", "source_system", "source_record_id", name="uq_asset_source"),
    )

    asset_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_system: Mapped[str] = mapped_column(String(128), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(255), nullable=False)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    model_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    serial_number: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lifecycle_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class AssetScopeRecord(TenantScopedMixin, Base):
    __tablename__ = "asset_scopes"

    scope_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    asset_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    site_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class TenantRetentionPolicyRecord(TenantScopedMixin, Base):
    """Versioned tenant defaults consumed by retention and deletion workflows."""

    __tablename__ = "tenant_retention_policies"
    __table_args__ = (UniqueConstraint("tenant_id", name="uq_tenant_retention_policy"),)

    policy_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    incident_days: Mapped[int] = mapped_column(Integer, nullable=False)
    media_days: Mapped[int] = mapped_column(Integer, nullable=False)
    knowledge_days: Mapped[int] = mapped_column(Integer, nullable=False)
    training_data_days: Mapped[int] = mapped_column(Integer, nullable=False)
    audit_days: Mapped[int] = mapped_column(Integer, nullable=False)
    legal_hold: Mapped[bool] = mapped_column(Boolean, nullable=False)
    updated_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class TenantAdministrationRecord(TenantScopedMixin, Base):
    """Append-only evidence for membership, access and retention policy changes."""

    __tablename__ = "tenant_administration_records"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "actor_subject_id",
            "client_operation_id",
            name="uq_tenant_administration_operation",
        ),
        Index(
            "ix_tenant_administration_occurred",
            "tenant_id",
            "occurred_at",
        ),
    )

    administration_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    client_operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    command_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    previous_version: Mapped[int] = mapped_column(Integer, nullable=False)
    target_version: Mapped[int] = mapped_column(Integer, nullable=False)
    before_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    after_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AssetCustomerLinkRecord(TenantScopedMixin, Base):
    __tablename__ = "asset_customer_links"
    __table_args__ = (
        UniqueConstraint("tenant_id", "asset_id", name="uq_asset_customer_link"),
        Index("ix_asset_customer_links_customer", "tenant_id", "customer_id"),
    )

    customer_link_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    customer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    customer_name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_system: Mapped[str] = mapped_column(String(128), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(255), nullable=False)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class AssetSiteLinkRecord(TenantScopedMixin, Base):
    __tablename__ = "asset_site_links"
    __table_args__ = (
        UniqueConstraint("tenant_id", "asset_id", name="uq_asset_site_link"),
        Index("ix_asset_site_links_site", "tenant_id", "site_id"),
    )

    site_link_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    site_id: Mapped[str] = mapped_column(String(128), nullable=False)
    site_name: Mapped[str] = mapped_column(String(255), nullable=False)
    address: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_system: Mapped[str] = mapped_column(String(128), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(255), nullable=False)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class AssetWarrantyRecord(TenantScopedMixin, Base):
    __tablename__ = "asset_warranties"
    __table_args__ = (
        UniqueConstraint("tenant_id", "asset_id", name="uq_asset_warranty"),
        Index("ix_asset_warranties_contract", "tenant_id", "contract_number"),
    )

    warranty_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    contract_number: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    coverage_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    coverage_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    service_level: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_system: Mapped[str] = mapped_column(String(128), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(255), nullable=False)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class AssetComponentRecord(TenantScopedMixin, Base):
    __tablename__ = "asset_components"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "asset_id",
            "source_system",
            "source_record_id",
            name="uq_asset_component_source",
        ),
        Index("ix_asset_components_asset", "tenant_id", "asset_id", "status"),
    )

    component_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    part_number: Mapped[str] = mapped_column(String(128), nullable=False)
    part_name: Mapped[str] = mapped_column(String(255), nullable=False)
    serial_number: Mapped[str | None] = mapped_column(String(255), nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    installed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_system: Mapped[str] = mapped_column(String(128), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(255), nullable=False)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class DeviceFamilyProfileRecord(TenantScopedMixin, Base):
    """Versioned, evidence-bound permission to route one equipment family."""

    __tablename__ = "device_family_profiles"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "family_code",
            "family_version",
            name="uq_device_family_profile_version",
        ),
        UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_device_family_profile_idempotency",
        ),
        Index(
            "ix_device_family_profiles_status",
            "tenant_id",
            "status",
            "family_code",
        ),
        Index(
            "uq_active_device_family_code",
            "tenant_id",
            "family_code",
            unique=True,
            postgresql_where=text("status = 'ACTIVE'"),
            sqlite_where=text("status = 'ACTIVE'"),
        ),
        CheckConstraint(
            "status IN ('BLOCKED', 'READY', 'ACTIVE', 'REJECTED', 'RETIRED')",
            name="ck_device_family_profile_status",
        ),
        CheckConstraint(
            "risk_class IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
            name="ck_device_family_profile_risk_class",
        ),
        CheckConstraint(
            "family_version >= 1 AND version >= 1 AND minimum_gold_samples >= 1",
            name="ck_device_family_profile_versions",
        ),
    )

    profile_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    family_code: Mapped[str] = mapped_column(String(64), nullable=False)
    family_version: Mapped[int] = mapped_column(Integer, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    risk_class: Mapped[str] = mapped_column(String(32), nullable=False)
    model_codes_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    source_system: Mapped[str] = mapped_column(String(128), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    knowledge_release_id: Mapped[str] = mapped_column(
        ForeignKey("index_releases.release_id"), nullable=False
    )
    evaluation_suite_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_suites.suite_id"), nullable=False
    )
    evaluation_id: Mapped[str] = mapped_column(
        ForeignKey("model_evaluation_runs.evaluation_id"), nullable=False
    )
    minimum_gold_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    readiness_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    evidence_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    configuration_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class DeviceFamilyModelBindingRecord(TenantScopedMixin, Base):
    """Normalized model mapping with a database-enforced single active owner."""

    __tablename__ = "device_family_model_bindings"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "profile_id",
            "model_code",
            name="uq_device_family_profile_model",
        ),
        Index(
            "uq_active_device_family_model",
            "tenant_id",
            "model_code",
            unique=True,
            postgresql_where=text("is_active"),
            sqlite_where=text("is_active = 1"),
        ),
        Index(
            "ix_device_family_model_profile",
            "tenant_id",
            "profile_id",
            "is_active",
        ),
        CheckConstraint("profile_version >= 1", name="ck_device_family_model_profile_version"),
    )

    binding_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("device_family_profiles.profile_id"), nullable=False
    )
    family_code: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_version: Mapped[int] = mapped_column(Integer, nullable=False)
    model_code: Mapped[str] = mapped_column(String(128), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False)


class IncidentDraftRecord(TenantScopedMixin, Base):
    __tablename__ = "incident_drafts"

    draft_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    author_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)


class IdempotencyRecord(TenantScopedMixin, Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        UniqueConstraint("tenant_id", "subject_id", "key", name="uq_idempotency_key"),
    )

    record_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    result_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EmergencyAccessGrantRecord(TenantScopedMixin, Base):
    """Current state of a short-lived, resource-bound break-glass request."""

    __tablename__ = "emergency_access_grants"

    grant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    incident_number: Mapped[str] = mapped_column(String(128), nullable=False)
    requester_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(128), nullable=False)
    justification: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approver_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class EmergencyAccessDecisionRecord(TenantScopedMixin, Base):
    """Append-only approval, rejection, revocation, and expiry evidence."""

    __tablename__ = "emergency_access_decisions"

    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    grant_id: Mapped[str] = mapped_column(
        ForeignKey("emergency_access_grants.grant_id"), nullable=False, index=True
    )
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EmergencyAccessUsageRecord(TenantScopedMixin, Base):
    """Append-only proof that a grant was actually used at a policy boundary."""

    __tablename__ = "emergency_access_usage"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "grant_id",
            "request_id",
            "action",
            "resource_id",
            name="uq_emergency_access_usage_request",
        ),
    )

    usage_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    grant_id: Mapped[str] = mapped_column(
        ForeignKey("emergency_access_grants.grant_id"), nullable=False, index=True
    )
    subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(128), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DraftTimelineEventRecord(TenantScopedMixin, Base):
    __tablename__ = "draft_timeline_events"
    __table_args__ = (
        UniqueConstraint("tenant_id", "draft_id", "sequence", name="uq_draft_timeline_sequence"),
    )

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    draft_id: Mapped[str] = mapped_column(ForeignKey("incident_drafts.draft_id"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class MediaObjectRecord(TenantScopedMixin, Base):
    __tablename__ = "media_objects"

    media_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    draft_id: Mapped[str] = mapped_column(ForeignKey("incident_drafts.draft_id"), nullable=False)
    quarantine_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    clean_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    declared_mime: Mapped[str] = mapped_column(String(255), nullable=False)
    detected_mime: Mapped[str | None] = mapped_column(String(255), nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    scan_state: Mapped[str] = mapped_column(String(32), nullable=False)
    scan_version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_credential_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PENDING_INSPECTION"
    )
    content_credential_report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    content_credential_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_credential_inspected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class RecognitionRunRecord(TenantScopedMixin, Base):
    __tablename__ = "recognition_runs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "workflow_id", name="uq_recognition_workflow"),
        UniqueConstraint(
            "tenant_id",
            "context_kind",
            "field_evidence_upload_id",
            "requested_by_subject_id",
            "work_order_version",
            name="uq_recognition_field_evidence_binding",
        ),
        CheckConstraint(
            "context_kind IN ('INCIDENT_DRAFT', 'FIELD_EVIDENCE')",
            name="ck_recognition_run_context_kind",
        ),
        CheckConstraint(
            "(context_kind = 'INCIDENT_DRAFT' AND work_order_id IS NULL "
            "AND field_evidence_upload_id IS NULL AND work_order_version IS NULL) OR "
            "(context_kind = 'FIELD_EVIDENCE' AND work_order_id IS NOT NULL "
            "AND field_evidence_upload_id IS NOT NULL AND work_order_version IS NOT NULL "
            "AND requested_by_subject_id IS NOT NULL)",
            name="ck_recognition_run_context_binding",
        ),
        Index(
            "ix_recognition_runs_field_context",
            "tenant_id",
            "work_order_id",
            "field_evidence_upload_id",
            "context_kind",
        ),
    )

    recognition_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    draft_id: Mapped[str] = mapped_column(ForeignKey("incident_drafts.draft_id"), nullable=False)
    media_id: Mapped[str] = mapped_column(ForeignKey("media_objects.media_id"), nullable=False)
    context_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default="INCIDENT_DRAFT", server_default="INCIDENT_DRAFT"
    )
    work_order_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=True
    )
    field_evidence_upload_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_order_field_evidence_uploads.evidence_upload_id"), nullable=True
    )
    work_order_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    requested_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    processor_profile: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_bundle_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class EvidenceBundleRecord(TenantScopedMixin, Base):
    __tablename__ = "evidence_bundles"
    __table_args__ = (
        Index(
            "uq_evidence_incident_draft_media",
            "tenant_id",
            "draft_id",
            "media_id",
            unique=True,
            postgresql_where=text("context_kind = 'INCIDENT_DRAFT'"),
            sqlite_where=text("context_kind = 'INCIDENT_DRAFT'"),
        ),
        CheckConstraint(
            "context_kind IN ('INCIDENT_DRAFT', 'FIELD_EVIDENCE')",
            name="ck_evidence_bundle_context_kind",
        ),
        CheckConstraint(
            "(context_kind = 'INCIDENT_DRAFT' AND work_order_id IS NULL "
            "AND field_evidence_upload_id IS NULL) OR "
            "(context_kind = 'FIELD_EVIDENCE' AND work_order_id IS NOT NULL "
            "AND field_evidence_upload_id IS NOT NULL)",
            name="ck_evidence_bundle_context_binding",
        ),
        Index(
            "ix_evidence_bundles_field_context",
            "tenant_id",
            "work_order_id",
            "field_evidence_upload_id",
            "context_kind",
        ),
    )

    bundle_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    draft_id: Mapped[str] = mapped_column(ForeignKey("incident_drafts.draft_id"), nullable=False)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    media_id: Mapped[str] = mapped_column(ForeignKey("media_objects.media_id"), nullable=False)
    context_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default="INCIDENT_DRAFT", server_default="INCIDENT_DRAFT"
    )
    work_order_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=True
    )
    field_evidence_upload_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_order_field_evidence_uploads.evidence_upload_id"), nullable=True
    )
    source_sha256: Mapped[str] = mapped_column(String(128), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    processor_versions: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)
    security_policy_version: Mapped[str | None] = mapped_column(
        String(128), nullable=True, default="m6-prompt-injection-v1"
    )
    security_findings_json: Mapped[list[dict[str, str]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    automation_eligible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="false"
    )
    automation_blockers_json: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class RecognitionResultRecord(TenantScopedMixin, Base):
    __tablename__ = "recognition_results"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "bundle_id",
            "result_type",
            "ordinal",
            name="uq_recognition_result_ordinal",
        ),
    )

    result_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    bundle_id: Mapped[str] = mapped_column(ForeignKey("evidence_bundles.bundle_id"), nullable=False)
    result_type: Mapped[str] = mapped_column(String(32), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class RecognitionCorrectionRecord(TenantScopedMixin, Base):
    __tablename__ = "recognition_corrections"

    correction_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    bundle_id: Mapped[str] = mapped_column(ForeignKey("evidence_bundles.bundle_id"), nullable=False)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[str] = mapped_column(String(128), nullable=False)
    original_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    disposition: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reviewer_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    evidence_version: Mapped[int] = mapped_column(Integer, nullable=False)


class IncidentRecord(TenantScopedMixin, Base):
    __tablename__ = "incidents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "source_draft_id", name="uq_incident_source_draft"),
    )

    incident_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    source_draft_id: Mapped[str] = mapped_column(
        ForeignKey("incident_drafts.draft_id"), nullable=False
    )
    evidence_bundle_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_bundles.bundle_id"), nullable=False
    )
    reporter_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str | None] = mapped_column(String(16), nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    responsible_queue: Mapped[str | None] = mapped_column(String(128), nullable=True)
    information_due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    escalation_responsible_subject_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class IncidentControlRecord(TenantScopedMixin, Base):
    """Append-only evidence for every manual incident-center decision."""

    __tablename__ = "incident_controls"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "incident_id",
            "incident_version",
            name="uq_incident_control_version",
        ),
        Index(
            "ix_incident_controls_incident",
            "tenant_id",
            "incident_id",
            "occurred_at",
        ),
    )

    control_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    incident_version: Mapped[int] = mapped_column(Integer, nullable=False)
    command_type: Mapped[str] = mapped_column(String(32), nullable=False)
    previous_status: Mapped[str] = mapped_column(String(32), nullable=False)
    target_status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    responsible_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    recovery_condition: Mapped[str | None] = mapped_column(Text, nullable=True)
    related_incident_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class IncidentResolutionRecord(TenantScopedMixin, Base):
    """Immutable fact describing how an incident reached RESOLVED."""

    __tablename__ = "incident_resolutions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "incident_id",
            name="uq_incident_resolution_incident",
        ),
        CheckConstraint(
            "(mode = 'REMOTE' AND diagnosis_run_id IS NOT NULL "
            "AND work_order_id IS NULL AND completion_id IS NULL "
            "AND verification_id IS NULL) OR "
            "(mode = 'WORK_ORDER' AND work_order_id IS NOT NULL "
            "AND completion_id IS NOT NULL AND verification_id IS NOT NULL)",
            name="ck_incident_resolution_source",
        ),
        Index(
            "ix_incident_resolutions_incident",
            "tenant_id",
            "incident_id",
            "resolved_at",
        ),
    )

    resolution_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    diagnosis_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("diagnosis_runs.diagnosis_run_id"), nullable=True
    )
    work_order_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=True
    )
    completion_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_order_completions.completion_id"), nullable=True
    )
    verification_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_order_verifications.verification_id"), nullable=True
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    resolved_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ServicePerformanceBaselineRecord(TenantScopedMixin, Base):
    """Immutable KPI evidence with an independently controlled lifecycle."""

    __tablename__ = "service_performance_baselines"
    __table_args__ = (
        CheckConstraint(
            "scope_type = 'TENANT'",
            name="ck_service_performance_baseline_scope",
        ),
        CheckConstraint(
            "status IN ('DRAFT', 'ACTIVE', 'RETIRED')",
            name="ck_service_performance_baseline_status",
        ),
        CheckConstraint(
            "data_quality_status = 'COMPLETE'",
            name="ck_service_performance_baseline_quality",
        ),
        CheckConstraint(
            "minimum_work_order_samples > 0 AND minimum_resolution_samples > 0",
            name="ck_service_performance_baseline_minimum_samples",
        ),
        CheckConstraint(
            "target_mttr_reduction_rate >= 0 AND target_mttr_reduction_rate <= 1 "
            "AND target_first_time_fix_lift >= 0 AND target_first_time_fix_lift <= 1 "
            "AND target_remote_resolution_lift >= 0 AND target_remote_resolution_lift <= 1",
            name="ck_service_performance_baseline_targets",
        ),
        Index(
            "uq_service_performance_baseline_active",
            "tenant_id",
            unique=True,
            postgresql_where=text("status = 'ACTIVE'"),
            sqlite_where=text("status = 'ACTIVE'"),
        ),
        Index(
            "ix_service_performance_baselines_tenant_created",
            "tenant_id",
            "created_at",
        ),
    )

    baseline_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    scope_type: Mapped[str] = mapped_column(String(16), nullable=False)
    reference_window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    reference_window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metric_contract_version: Mapped[str] = mapped_column(String(64), nullable=False)
    mean_mttr_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    first_time_fix_rate: Mapped[float] = mapped_column(Float, nullable=False)
    remote_resolution_rate: Mapped[float] = mapped_column(Float, nullable=False)
    closed_work_order_count: Mapped[int] = mapped_column(Integer, nullable=False)
    first_time_fix_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mttr_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    resolution_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_work_order_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_resolution_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    target_mttr_reduction_rate: Mapped[float] = mapped_column(Float, nullable=False)
    target_first_time_fix_lift: Mapped[float] = mapped_column(Float, nullable=False)
    target_remote_resolution_lift: Mapped[float] = mapped_column(Float, nullable=False)
    data_quality_status: Mapped[str] = mapped_column(String(16), nullable=False)
    data_quality_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    content_digest: Mapped[str] = mapped_column(String(71), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    activated_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    activation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class IncidentRelationRecord(TenantScopedMixin, Base):
    """Durable incident-to-incident relation, currently used for duplicate merges."""

    __tablename__ = "incident_relations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "source_incident_id",
            "relation_type",
            name="uq_incident_relation_source_type",
        ),
        Index(
            "ix_incident_relations_target",
            "tenant_id",
            "target_incident_id",
        ),
    )

    relation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_incident_id: Mapped[str] = mapped_column(
        ForeignKey("incidents.incident_id"), nullable=False
    )
    target_incident_id: Mapped[str] = mapped_column(
        ForeignKey("incidents.incident_id"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class KnowledgeDocumentRecord(TenantScopedMixin, Base):
    __tablename__ = "knowledge_documents"

    document_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    classification: Mapped[str] = mapped_column(String(32), nullable=False)
    source_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    current_version: Mapped[int] = mapped_column(Integer, nullable=False)


class KnowledgeDocumentVersionRecord(TenantScopedMixin, Base):
    __tablename__ = "knowledge_document_versions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "document_id",
            "version",
            name="uq_knowledge_document_version",
        ),
    )

    document_version_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.document_id"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    parser_version: Mapped[str] = mapped_column(String(128), nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(128), nullable=False)
    acl_subject_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    acl_roles: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    device_families: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    device_models: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    content_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    extraction_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(32), nullable=False)


class KnowledgeIngestionJobRecord(TenantScopedMixin, Base):
    __tablename__ = "knowledge_ingestion_jobs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "workflow_id", name="uq_knowledge_ingestion_workflow"),
    )

    ingestion_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    declared_mime: Mapped[str] = mapped_column(String(128), nullable=False)
    detected_mime: Mapped[str | None] = mapped_column(String(128), nullable=True)
    classification: Mapped[str] = mapped_column(String(32), nullable=False)
    acl_subject_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    acl_roles: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    device_families: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    device_models: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    quarantine_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    clean_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("knowledge_documents.document_id"), nullable=True
    )
    document_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("knowledge_document_versions.document_version_id"), nullable=True
    )
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class KnowledgeIngestionAttemptRecord(TenantScopedMixin, Base):
    __tablename__ = "knowledge_ingestion_attempts"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "ingestion_id",
            "attempt_number",
            name="uq_knowledge_ingestion_attempt_number",
        ),
        UniqueConstraint(
            "tenant_id",
            "workflow_id",
            name="uq_knowledge_ingestion_attempt_workflow",
        ),
    )

    attempt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    ingestion_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_ingestion_jobs.ingestion_id"), nullable=False
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class KnowledgeDeletionRequestRecord(TenantScopedMixin, Base):
    """Minimal audit proof and execution state for one knowledge purge request."""

    __tablename__ = "knowledge_deletion_requests"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workflow_id",
            name="uq_knowledge_deletion_workflow",
        ),
    )

    deletion_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    document_id: Mapped[str] = mapped_column(String(128), nullable=False)
    document_version_id: Mapped[str] = mapped_column(String(128), nullable=False)
    trigger: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    target_source_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    target_content_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    scope_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    impact_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    verification_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class IndexReleaseRecord(TenantScopedMixin, Base):
    __tablename__ = "index_releases"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", "version", name="uq_index_release_version"),
        Index(
            "uq_active_index_release",
            "tenant_id",
            "name",
            unique=True,
            postgresql_where=text("is_active"),
            sqlite_where=text("is_active = 1"),
        ),
    )

    release_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    created_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    published_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    activation_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class KnowledgeIndexActivationRecord(TenantScopedMixin, Base):
    """Append-only alias switch evidence for publication and rollback."""

    __tablename__ = "knowledge_index_activations"
    __table_args__ = (
        Index(
            "ix_knowledge_index_activations_name",
            "tenant_id",
            "name",
            "occurred_at",
        ),
    )

    activation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    from_release_id: Mapped[str | None] = mapped_column(
        ForeignKey("index_releases.release_id"), nullable=True
    )
    to_release_id: Mapped[str] = mapped_column(
        ForeignKey("index_releases.release_id"), nullable=False
    )
    from_content_checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    to_content_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    retired_search_profile_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    revoked_graph_release_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class KnowledgeIndexEvaluationRecord(TenantScopedMixin, Base):
    """Durable quality and authorization evidence for one immutable index release."""

    __tablename__ = "knowledge_index_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workflow_id",
            name="uq_knowledge_index_evaluation_workflow",
        ),
    )

    evaluation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("index_releases.release_id"), nullable=False)
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(128), nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    gate_results_json: Mapped[dict[str, bool]] = mapped_column(JSON, nullable=False)
    failure_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class KnowledgeChunkRecord(TenantScopedMixin, Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "release_id",
            "document_version_id",
            "ordinal",
            name="uq_knowledge_chunk_ordinal",
        ),
    )

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_document_versions.document_version_id"), nullable=False
    )
    release_id: Mapped[str] = mapped_column(ForeignKey("index_releases.release_id"), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        JSON().with_variant(Vector(16), "postgresql"), nullable=False
    )
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)


class CitationAnchorRecord(TenantScopedMixin, Base):
    __tablename__ = "citation_anchors"

    citation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("knowledge_chunks.chunk_id"), nullable=False)
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_document_versions.document_version_id"), nullable=False
    )
    anchor_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bounding_box: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    excerpt_checksum: Mapped[str] = mapped_column(String(128), nullable=False)


class KnowledgeGraphReleaseRecord(TenantScopedMixin, Base):
    """Governed GraphRAG projection of one already-published knowledge release."""

    __tablename__ = "knowledge_graph_releases"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", "version", name="uq_knowledge_graph_release_version"),
        Index(
            "uq_active_knowledge_graph_release",
            "tenant_id",
            "name",
            unique=True,
            postgresql_where=text("is_active"),
            sqlite_where=text("is_active = 1"),
        ),
    )

    graph_release_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_index_release_id: Mapped[str] = mapped_column(
        ForeignKey("index_releases.release_id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    gate_results_json: Mapped[dict[str, bool]] = mapped_column(JSON, nullable=False)
    failure_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluated_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    activated_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)


class KnowledgeGraphNodeRecord(TenantScopedMixin, Base):
    __tablename__ = "knowledge_graph_nodes"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "graph_release_id",
            "node_key",
            name="uq_knowledge_graph_node_key",
        ),
        Index(
            "ix_knowledge_graph_node_source",
            "tenant_id",
            "document_version_id",
        ),
    )

    graph_node_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    graph_release_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_graph_releases.graph_release_id"), nullable=False
    )
    node_key: Mapped[str] = mapped_column(String(128), nullable=False)
    node_type: Mapped[str] = mapped_column(String(32), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    citation_id: Mapped[str] = mapped_column(
        ForeignKey("citation_anchors.citation_id"), nullable=False
    )
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_document_versions.document_version_id"), nullable=False
    )
    classification: Mapped[str] = mapped_column(String(32), nullable=False)
    device_families: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    device_models: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class KnowledgeGraphEdgeRecord(TenantScopedMixin, Base):
    __tablename__ = "knowledge_graph_edges"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "graph_release_id",
            "source_node_id",
            "target_node_id",
            "relation_type",
            name="uq_knowledge_graph_edge_fact",
        ),
        Index(
            "ix_knowledge_graph_edge_source",
            "tenant_id",
            "document_version_id",
        ),
    )

    graph_edge_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    graph_release_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_graph_releases.graph_release_id"), nullable=False
    )
    source_node_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_graph_nodes.graph_node_id"), nullable=False
    )
    target_node_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_graph_nodes.graph_node_id"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    citation_id: Mapped[str] = mapped_column(
        ForeignKey("citation_anchors.citation_id"), nullable=False
    )
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_document_versions.document_version_id"), nullable=False
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)


class KnowledgeGraphEvaluationRecord(TenantScopedMixin, Base):
    __tablename__ = "knowledge_graph_evaluations"

    evaluation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    graph_release_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_graph_releases.graph_release_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    cases_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    gate_results_json: Mapped[dict[str, bool]] = mapped_column(JSON, nullable=False)
    failure_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class KnowledgeGraphExtractionJobRecord(TenantScopedMixin, Base):
    """Immutable source/inference binding and monotonic review candidate lifecycle."""

    __tablename__ = "knowledge_graph_extraction_jobs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workflow_id",
            name="uq_knowledge_graph_extraction_workflow",
        ),
        UniqueConstraint(
            "tenant_id",
            "inference_request_id",
            name="uq_knowledge_graph_extraction_inference",
        ),
        Index(
            "ix_knowledge_graph_extraction_release_status",
            "tenant_id",
            "graph_release_id",
            "status",
        ),
        CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'REVIEW_PENDING', 'ACCEPTED', 'REJECTED', 'FAILED')",
            name="ck_knowledge_graph_extraction_status",
        ),
        CheckConstraint("attempt_count >= 0", name="ck_knowledge_graph_extraction_attempt"),
        CheckConstraint("version > 0", name="ck_knowledge_graph_extraction_version"),
    )

    extraction_job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    graph_release_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_graph_releases.graph_release_id"), nullable=False
    )
    source_index_release_id: Mapped[str] = mapped_column(
        ForeignKey("index_releases.release_id"), nullable=False
    )
    expected_graph_version: Mapped[int] = mapped_column(Integer, nullable=False)
    citation_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    citation_bindings_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    source_binding_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    extraction_profile: Mapped[str] = mapped_column(String(128), nullable=False)
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    inference_request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_alias: Mapped[str] = mapped_column(String(128), nullable=False)
    model_release_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_bundle_id: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_bundle_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    response_schema_version: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_bundle_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    result_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    context_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class KnowledgeSearchProfileRecord(TenantScopedMixin, Base):
    """Governed OpenSearch projection of one immutable published index release."""

    __tablename__ = "knowledge_search_profiles"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "name", "version", name="uq_knowledge_search_profile_version"
        ),
        Index(
            "uq_active_knowledge_search_profile",
            "tenant_id",
            "name",
            unique=True,
            postgresql_where=text("is_active_shadow"),
            sqlite_where=text("is_active_shadow = 1"),
        ),
    )

    search_profile_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_index_release_id: Mapped[str] = mapped_column(
        ForeignKey("index_releases.release_id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    backend: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    is_active_shadow: Mapped[bool] = mapped_column(Boolean, nullable=False)
    index_name: Mapped[str] = mapped_column(String(255), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_model_release_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_manifest_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_component_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_artifact_content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    document_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    gate_results_json: Mapped[dict[str, bool]] = mapped_column(JSON, nullable=False)
    failure_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluated_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    activated_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)


class KnowledgeSearchRebuildJobRecord(TenantScopedMixin, Base):
    """Durable Temporal job for rebuilding one OpenSearch projection."""

    __tablename__ = "knowledge_search_rebuild_jobs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workflow_id",
            name="uq_knowledge_search_rebuild_workflow",
        ),
        Index(
            "uq_active_knowledge_search_rebuild",
            "tenant_id",
            "search_profile_id",
            unique=True,
            postgresql_where=text("status IN ('QUEUED', 'RUNNING')"),
            sqlite_where=text("status IN ('QUEUED', 'RUNNING')"),
        ),
        Index(
            "ix_knowledge_search_rebuild_status",
            "tenant_id",
            "status",
            "updated_at",
        ),
        CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100",
            name="ck_knowledge_search_rebuild_progress",
        ),
        CheckConstraint(
            "completed_items >= 0 AND total_items >= 0",
            name="ck_knowledge_search_rebuild_items",
        ),
    )

    rebuild_job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    search_profile_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_search_profiles.search_profile_id"), nullable=False
    )
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    expected_profile_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_items: Mapped[int] = mapped_column(Integer, nullable=False)
    total_items: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    failure_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ExternalSearchPolicyRecord(TenantScopedMixin, Base):
    """Tenant-owned allowlist for optional public web references."""

    __tablename__ = "external_search_policies"
    __table_args__ = (UniqueConstraint("tenant_id", name="uq_external_search_policy_tenant"),)

    policy_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    allowed_domains: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    official_domains: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    max_results: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ExternalSearchQueryRecord(TenantScopedMixin, Base):
    """Auditable query command; public search never becomes enterprise knowledge."""

    __tablename__ = "external_search_queries"
    __table_args__ = (
        Index("ix_external_search_query_status", "tenant_id", "status", "updated_at"),
        Index("ix_external_search_query_requested", "tenant_id", "requested_by_subject_id"),
    )

    external_search_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    use_case: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False)
    blocked_result_count: Mapped[int] = mapped_column(Integer, nullable=False)
    usage_credits: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    usage_conclusion: Mapped[str] = mapped_column(String(48), nullable=False)
    conclusion_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    concluded_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ExternalSearchReferenceRecord(TenantScopedMixin, Base):
    """Filtered, guardrailed source summary returned by one public search."""

    __tablename__ = "external_search_references"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "external_search_id",
            "rank",
            name="uq_external_search_reference_rank",
        ),
        UniqueConstraint(
            "tenant_id",
            "external_search_id",
            "url_hash",
            name="uq_external_search_reference_url",
        ),
    )

    external_reference_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    external_search_id: Mapped[str] = mapped_column(
        ForeignKey("external_search_queries.external_search_id"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    trust_level: Mapped[str] = mapped_column(String(32), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    guardrail_policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class KnowledgeSearchProjectionRecord(TenantScopedMixin, Base):
    """Auditable SQL manifest for every document projected to OpenSearch."""

    __tablename__ = "knowledge_search_projections"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "search_profile_id",
            "citation_id",
            name="uq_knowledge_search_projection_citation",
        ),
        Index(
            "ix_knowledge_search_projection_source",
            "tenant_id",
            "document_version_id",
        ),
    )

    projection_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    search_profile_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_search_profiles.search_profile_id"), nullable=False
    )
    citation_id: Mapped[str] = mapped_column(
        ForeignKey("citation_anchors.citation_id"), nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(ForeignKey("knowledge_chunks.chunk_id"), nullable=False)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.document_id"), nullable=False
    )
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_document_versions.document_version_id"), nullable=False
    )
    content_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    source_checksum: Mapped[str] = mapped_column(String(128), nullable=False)


class KnowledgeSearchEvaluationRecord(TenantScopedMixin, Base):
    __tablename__ = "knowledge_search_evaluations"

    evaluation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    search_profile_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_search_profiles.search_profile_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    cases_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    gate_results_json: Mapped[dict[str, bool]] = mapped_column(JSON, nullable=False)
    failure_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class DiagnosisRunRecord(TenantScopedMixin, Base):
    __tablename__ = "diagnosis_runs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "workflow_id", name="uq_diagnosis_workflow"),
        UniqueConstraint("tenant_id", "agent_run_id", name="uq_diagnosis_agent_run"),
    )

    diagnosis_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    evidence_bundle_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_bundles.bundle_id"), nullable=False
    )
    agent_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class DiagnosisQualityFeedbackRecord(TenantScopedMixin, Base):
    """Append-only human quality judgment bound to one immutable AI diagnosis."""

    __tablename__ = "diagnosis_quality_feedback"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "diagnosis_run_id",
            "submitted_by_subject_id",
            name="uq_diagnosis_feedback_subject_run",
        ),
        UniqueConstraint(
            "tenant_id",
            "submitted_by_subject_id",
            "idempotency_key",
            name="uq_diagnosis_feedback_subject_idempotency",
        ),
        CheckConstraint(
            "verdict IN ('HELPFUL', 'PARTIALLY_HELPFUL', 'NOT_HELPFUL', 'UNSAFE')",
            name="ck_diagnosis_feedback_verdict",
        ),
        CheckConstraint(
            "diagnosis_status IN ('COMPLETED', 'NEEDS_INFORMATION')",
            name="ck_diagnosis_feedback_terminal_status",
        ),
        Index(
            "ix_diagnosis_feedback_incident_created",
            "tenant_id",
            "incident_id",
            "created_at",
            "feedback_id",
        ),
    )

    feedback_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    diagnosis_run_id: Mapped[str] = mapped_column(
        ForeignKey("diagnosis_runs.diagnosis_run_id"), nullable=False
    )
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    agent_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    diagnosis_status: Mapped[str] = mapped_column(String(32), nullable=False)
    diagnosis_version: Mapped[int] = mapped_column(Integer, nullable=False)
    report_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    manifest_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    model_release_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_bundle_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    index_release_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    context_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    issue_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    submitted_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    submitted_by_role: Mapped[str] = mapped_column(String(64), nullable=False)


class SupplierCollaborationRecord(TenantScopedMixin, Base):
    """Durable A2A task containing only an approved, redacted evidence package."""

    __tablename__ = "supplier_collaborations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_supplier_collaboration_idempotency",
        ),
        Index(
            "ix_supplier_collaboration_incident",
            "tenant_id",
            "incident_id",
            "created_at",
        ),
        Index(
            "ix_supplier_collaboration_status",
            "tenant_id",
            "status",
            "updated_at",
        ),
    )

    collaboration_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    diagnosis_run_id: Mapped[str] = mapped_column(
        ForeignKey("diagnosis_runs.diagnosis_run_id"), nullable=False
    )
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False)
    agent_version: Mapped[str] = mapped_column(String(64), nullable=False)
    skill_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    sharing_reason: Mapped[str] = mapped_column(Text, nullable=False)
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    request_payload_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    dlp_policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    dlp_finding_count: Mapped[int] = mapped_column(Integer, nullable=False)
    remote_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    remote_context_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    response_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    response_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    guardrail_policy_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class SupplierCollaborationEventRecord(TenantScopedMixin, Base):
    """Append-only task and human-review evidence without raw remote messages."""

    __tablename__ = "supplier_collaboration_events"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "collaboration_id",
            "sequence",
            name="uq_supplier_collaboration_event_sequence",
        ),
        Index(
            "ix_supplier_collaboration_event_time",
            "tenant_id",
            "collaboration_id",
            "occurred_at",
        ),
    )

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    collaboration_id: Mapped[str] = mapped_column(
        ForeignKey("supplier_collaborations.collaboration_id"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MaintenancePlanningCouncilRecord(TenantScopedMixin, Base):
    """Governed multi-agent council bound to one immutable diagnosis run."""

    __tablename__ = "maintenance_planning_councils"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "diagnosis_run_id",
            name="uq_maintenance_council_diagnosis",
        ),
        UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_maintenance_council_idempotency",
        ),
        Index(
            "ix_maintenance_council_status",
            "tenant_id",
            "status",
            "updated_at",
        ),
        CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'REVIEW_PENDING', 'ACCEPTED', 'REJECTED', 'FAILED')",
            name="ck_maintenance_council_status",
        ),
        CheckConstraint(
            "attempt_count >= 0 AND version >= 1 AND diagnosis_version >= 1",
            name="ck_maintenance_council_versions",
        ),
    )

    council_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    diagnosis_run_id: Mapped[str] = mapped_column(
        ForeignKey("diagnosis_runs.diagnosis_run_id"), nullable=False
    )
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    diagnosis_version: Mapped[int] = mapped_column(Integer, nullable=False)
    diagnosis_report_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    diagnosis_manifest_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    model_alias: Mapped[str] = mapped_column(String(128), nullable=False)
    model_release_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_bundle_id: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_bundle_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    response_schema_version: Mapped[str] = mapped_column(String(128), nullable=False)
    activation_id: Mapped[str | None] = mapped_column(
        ForeignKey("maintenance_planning_activations.activation_id"), nullable=True
    )
    activation_policy_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    activation_target_environment: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    plan_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    result_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    coordinator_context_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class MaintenancePlanningContributionRecord(TenantScopedMixin, Base):
    """Append-only specialist or coordinator contribution for one council attempt."""

    __tablename__ = "maintenance_planning_contributions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "council_id",
            "attempt_number",
            "agent_role",
            name="uq_maintenance_contribution_role_attempt",
        ),
        UniqueConstraint(
            "tenant_id",
            "inference_request_id",
            name="uq_maintenance_contribution_inference",
        ),
        Index(
            "ix_maintenance_contribution_council",
            "tenant_id",
            "council_id",
            "attempt_number",
        ),
        CheckConstraint(
            "agent_role IN ('SAFETY', 'PARTS', 'DISPATCH', 'COORDINATOR')",
            name="ck_maintenance_contribution_role",
        ),
        CheckConstraint(
            "attempt_number >= 1",
            name="ck_maintenance_contribution_attempt",
        ),
    )

    contribution_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    council_id: Mapped[str] = mapped_column(
        ForeignKey("maintenance_planning_councils.council_id"), nullable=False
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_role: Mapped[str] = mapped_column(String(32), nullable=False)
    inference_request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    input_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    output_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    output_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    model_release_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    context_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MaintenancePlanningEvaluationSuiteRecord(TenantScopedMixin, Base):
    """Frozen Gold cases for a blinded single-agent versus council comparison."""

    __tablename__ = "maintenance_planning_evaluation_suites"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "name",
            "suite_version",
            name="uq_maintenance_planning_eval_suite_version",
        ),
        CheckConstraint(
            "evidence_tier IN ('PROJECT_STAGING_GOLD', 'ENTERPRISE_GOLD')",
            name="ck_maintenance_planning_eval_suite_tier",
        ),
        CheckConstraint(
            "status IN ('FROZEN', 'RETIRED')",
            name="ck_maintenance_planning_eval_suite_status",
        ),
        CheckConstraint(
            "sample_count > 0 AND version > 0",
            name="ck_maintenance_planning_eval_suite_counts",
        ),
    )

    suite_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    suite_version: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_tier: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    cases_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    retired_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class MaintenancePlanningEvaluationRunRecord(TenantScopedMixin, Base):
    """Immutable A/B assignments and aggregate net-benefit decision."""

    __tablename__ = "maintenance_planning_evaluation_runs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_maintenance_planning_eval_run_idempotency",
        ),
        Index(
            "ix_maintenance_planning_eval_run_status",
            "tenant_id",
            "status",
            "updated_at",
        ),
        CheckConstraint(
            "status IN ('JUDGING', 'COMPLETED', 'FAILED')",
            name="ck_maintenance_planning_eval_run_status",
        ),
        CheckConstraint(
            "decision IN ('PENDING', 'KEEP_SINGLE_AGENT', "
            "'PROJECT_CANDIDATE_ELIGIBLE', 'PRODUCTION_CANDIDATE_ELIGIBLE')",
            name="ck_maintenance_planning_eval_run_decision",
        ),
        CheckConstraint(
            "required_judgments_per_case >= 2 AND version > 0",
            name="ck_maintenance_planning_eval_run_counts",
        ),
    )

    evaluation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    suite_id: Mapped[str] = mapped_column(
        ForeignKey("maintenance_planning_evaluation_suites.suite_id"), nullable=False
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    policy_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    policy_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    pairs_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    required_judgments_per_case: Mapped[int] = mapped_column(Integer, nullable=False)
    aggregate_metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    gate_results_json: Mapped[dict[str, bool]] = mapped_column(JSON, nullable=False)
    decision: Mapped[str] = mapped_column(String(48), nullable=False)
    report_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class MaintenancePlanningBlindJudgmentRecord(TenantScopedMixin, Base):
    """Append-only independent score for one anonymous A/B planning pair."""

    __tablename__ = "maintenance_planning_blind_judgments"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "evaluation_id",
            "case_id",
            "judge_subject_id",
            name="uq_maintenance_planning_blind_judge_case",
        ),
        Index(
            "ix_maintenance_planning_blind_judgment_case",
            "tenant_id",
            "evaluation_id",
            "case_id",
            "submitted_at",
        ),
        CheckConstraint(
            "preferred_variant IN ('A', 'B', 'TIE')",
            name="ck_maintenance_planning_blind_judgment_preference",
        ),
    )

    judgment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_id: Mapped[str] = mapped_column(
        ForeignKey("maintenance_planning_evaluation_runs.evaluation_id"), nullable=False
    )
    case_id: Mapped[str] = mapped_column(String(128), nullable=False)
    judge_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    blind_assignment_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    variant_a_scores_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    variant_b_scores_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    preferred_variant: Mapped[str] = mapped_column(String(8), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    score_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    is_adjudication: Mapped[bool] = mapped_column(Boolean, nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MaintenanceReviewExposureRecord(TenantScopedMixin, Base):
    """Append-only platform-observed contact with an Incident source family."""

    __tablename__ = "maintenance_review_exposures"
    __table_args__ = (
        UniqueConstraint("tenant_id", "subject_id", "dedup_key", name="uq_review_exposure_dedup"),
        Index("ix_review_exposure_family", "tenant_id", "subject_id", "source_family_key"),
    )

    exposure_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"), nullable=False)
    source_family_key: Mapped[str] = mapped_column(String(128), nullable=False)
    source_version_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    dedup_key: Mapped[str] = mapped_column(String(128), nullable=False)


class MaintenanceReviewAssignmentRecord(TenantScopedMixin, Base):
    """Opaque reviewer ownership retained after submission or withdrawal."""

    __tablename__ = "maintenance_review_assignments"
    __table_args__ = (
        UniqueConstraint("tenant_id", "subject_id", "idempotency_key", name="uq_review_claim_key"),
        UniqueConstraint("tenant_id", "subject_id", "evaluation_id", "case_id",
                         name="uq_review_claim_case"),
        Index("ix_review_claim_subject_state", "tenant_id", "subject_id", "state"),
        CheckConstraint("state IN ('ACTIVE', 'SUBMITTED', 'WITHDRAWN')",
                        name="ck_maintenance_review_assignment_state"),
        CheckConstraint("version > 0 AND policy_epoch_version > 0",
                        name="ck_maintenance_review_assignment_version"),
        CheckConstraint(
            "(state = 'ACTIVE' AND submitted_at IS NULL AND withdrawn_at IS NULL "
            "AND submission_digest IS NULL) OR "
            "(state = 'SUBMITTED' AND submitted_at IS NOT NULL AND withdrawn_at IS NULL "
            "AND submission_digest IS NOT NULL) OR "
            "(state = 'WITHDRAWN' AND submitted_at IS NULL AND withdrawn_at IS NOT NULL "
            "AND submission_digest IS NULL)", name="ck_review_claim_timestamps"),
    )

    claim_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"), nullable=False)
    evaluation_id: Mapped[str] = mapped_column(
        ForeignKey("maintenance_planning_evaluation_runs.evaluation_id"), nullable=False)
    case_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_family_keys: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    source_binding_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    policy_epoch_version: Mapped[int] = mapped_column(Integer, nullable=False)
    reader_coverage_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    claimed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    submission_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)


class MaintenanceReviewRolloutRecord(TenantScopedMixin, Base):
    """Immutable original signed receipt and the server-verified proof, not a pass flag."""

    __tablename__ = "maintenance_review_rollouts"
    __table_args__ = (
        UniqueConstraint("tenant_id", "receipt_digest", name="uq_review_rollout_digest"),
    )
    receipt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    receipt_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    receipt_text: Mapped[str] = mapped_column(Text, nullable=False)
    signature_bundle: Mapped[str] = mapped_column(Text, nullable=False)
    proof_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    verification_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    registered_by: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"), nullable=False)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)


class MaintenanceReviewPolicyCommandRecord(TenantScopedMixin, Base):
    """Retained command results prevent replay from moving an epoch or erasing a rollback."""

    __tablename__ = "maintenance_review_policy_commands"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "subject_id", "idempotency_key", name="uq_review_policy_command"
        ),
    )
    command_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)


class MaintenanceReviewPolicyRecord(TenantScopedMixin, Base):
    """Tenant cutover; absence or a null epoch never means clean source history."""

    __tablename__ = "maintenance_review_policies"
    __table_args__ = (CheckConstraint("version > 0", name="ck_review_policy_version"),)

    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    receipt_id: Mapped[str | None] = mapped_column(
        ForeignKey("maintenance_review_rollouts.receipt_id"), nullable=True
    )
    receipt_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    reader_coverage_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    cutover_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    activated_by: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class MaintenancePlanningActivationRecord(TenantScopedMixin, Base):
    """Tenant rollout approval bound to one completed blinded evaluation."""

    __tablename__ = "maintenance_planning_activations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_maintenance_planning_activation_idempotency",
        ),
        Index(
            "ix_maintenance_planning_activation_status",
            "tenant_id",
            "target_environment",
            "status",
            "updated_at",
        ),
        Index(
            "uq_active_maintenance_planning_activation",
            "tenant_id",
            "target_environment",
            unique=True,
            postgresql_where=text("status = 'ACTIVE'"),
            sqlite_where=text("status = 'ACTIVE'"),
        ),
        CheckConstraint(
            "target_environment IN ('PROJECT_STAGING', 'PRODUCTION')",
            name="ck_maintenance_planning_activation_target",
        ),
        CheckConstraint(
            "status IN ('PENDING_APPROVAL', 'ACTIVE', 'REJECTED', 'ROLLED_BACK')",
            name="ck_maintenance_planning_activation_status",
        ),
        CheckConstraint(
            "decision IS NULL OR decision IN ('APPROVE', 'REJECT')",
            name="ck_maintenance_planning_activation_decision",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_maintenance_planning_activation_version",
        ),
    )

    activation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_id: Mapped[str] = mapped_column(
        ForeignKey("maintenance_planning_evaluation_runs.evaluation_id"), nullable=False
    )
    suite_id: Mapped[str] = mapped_column(
        ForeignKey("maintenance_planning_evaluation_suites.suite_id"), nullable=False
    )
    target_environment: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluation_decision: Mapped[str] = mapped_column(String(48), nullable=False)
    evaluation_report_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluation_policy_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    suite_manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    activation_policy_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    request_reason: Mapped[str] = mapped_column(Text, nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    decided_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rolled_back_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    rollback_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class GovernedMemoryRecord(TenantScopedMixin, Base):
    """Human-confirmed, revocable memory; never an authoritative enterprise fact."""

    __tablename__ = "governed_memories"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_governed_memory_idempotency",
        ),
        UniqueConstraint(
            "tenant_id",
            "memory_type",
            "source_reference_id",
            name="uq_governed_memory_source_reference",
        ),
        Index(
            "ix_governed_memory_incident_status",
            "tenant_id",
            "incident_id",
            "status",
        ),
        Index(
            "ix_governed_memory_owner_status",
            "tenant_id",
            "owner_subject_id",
            "status",
        ),
    )

    memory_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    memory_type: Mapped[str] = mapped_column(String(32), nullable=False)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    owner_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    incident_id: Mapped[str | None] = mapped_column(
        ForeignKey("incidents.incident_id"), nullable=True
    )
    asset_id: Mapped[str | None] = mapped_column(ForeignKey("assets.asset_id"), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    source_reference_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_reference_id: Mapped[str] = mapped_column(String(255), nullable=False)
    confirmed_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    revoked_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    revocation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class GovernedMemoryTransitionRecord(TenantScopedMixin, Base):
    """Append-only lifecycle evidence for a governed memory."""

    __tablename__ = "governed_memory_transitions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "memory_id",
            "sequence",
            name="uq_governed_memory_transition_sequence",
        ),
    )

    transition_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    memory_id: Mapped[str] = mapped_column(
        ForeignKey("governed_memories.memory_id"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    from_status: Mapped[str] = mapped_column(String(32), nullable=False)
    to_status: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DiagnosisExpertInterventionRecord(TenantScopedMixin, Base):
    """One expert takeover without mutating the model-authored report."""

    __tablename__ = "diagnosis_expert_interventions"
    __table_args__ = (
        Index(
            "uq_open_diagnosis_expert_intervention",
            "tenant_id",
            "diagnosis_run_id",
            unique=True,
            postgresql_where=text("status = 'OPEN'"),
            sqlite_where=text("status = 'OPEN'"),
        ),
    )

    intervention_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    diagnosis_run_id: Mapped[str] = mapped_column(
        ForeignKey("diagnosis_runs.diagnosis_run_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    source_run_version: Mapped[int] = mapped_column(Integer, nullable=False)
    ai_report_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    ai_report_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    takeover_reason: Mapped[str] = mapped_column(String(1024), nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class DiagnosisExpertRevisionRecord(TenantScopedMixin, Base):
    """Append-only expert-authored report revision."""

    __tablename__ = "diagnosis_expert_revisions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "intervention_id",
            "ordinal",
            name="uq_diagnosis_expert_revision_ordinal",
        ),
    )

    revision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    intervention_id: Mapped[str] = mapped_column(
        ForeignKey("diagnosis_expert_interventions.intervention_id"), nullable=False
    )
    diagnosis_run_id: Mapped[str] = mapped_column(
        ForeignKey("diagnosis_runs.diagnosis_run_id"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    report_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    source_ai_report_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    revision_reason: Mapped[str] = mapped_column(String(1024), nullable=False)
    authored_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)


class AgentRunRecord(TenantScopedMixin, Base):
    __tablename__ = "agent_runs"

    agent_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    diagnosis_run_id: Mapped[str] = mapped_column(
        ForeignKey("diagnosis_runs.diagnosis_run_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    budget: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    last_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class AgentEventRecord(TenantScopedMixin, Base):
    __tablename__ = "agent_events"
    __table_args__ = (
        UniqueConstraint("tenant_id", "agent_run_id", "sequence", name="uq_agent_event_sequence"),
    )

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.agent_run_id"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    visibility: Mapped[str] = mapped_column(String(32), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AgentCheckpointRecord(TenantScopedMixin, Base):
    __tablename__ = "agent_checkpoints"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "agent_run_id",
            "sequence",
            name="uq_agent_checkpoint_sequence",
        ),
    )

    checkpoint_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.agent_run_id"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    graph_version: Mapped[str] = mapped_column(String(128), nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    state_checksum: Mapped[str] = mapped_column(String(128), nullable=False)


class ToolCallRecord(TenantScopedMixin, Base):
    __tablename__ = "tool_calls"

    tool_call_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    agent_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_version: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    input_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    output_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class EquipmentControlHandoffRecord(TenantScopedMixin, Base):
    """Immutable evidence package for an external certified control system."""

    __tablename__ = "equipment_control_handoffs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "client_operation_id",
            name="uq_equipment_control_handoff_operation",
        ),
        Index(
            "ix_equipment_control_handoffs_incident",
            "tenant_id",
            "incident_id",
            "created_at",
        ),
    )

    handoff_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    client_operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    tool_call_id: Mapped[str] = mapped_column(ForeignKey("tool_calls.tool_call_id"), nullable=False)
    requested_tool_id: Mapped[str] = mapped_column(String(128), nullable=False)
    requested_operation: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    target_system: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    incident_version: Mapped[int] = mapped_column(Integer, nullable=False)


class ActionProposalRecord(TenantScopedMixin, Base):
    __tablename__ = "action_proposals"

    proposal_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    initiator_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_version: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(8), nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    parameters_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_version: Mapped[int] = mapped_column(Integer, nullable=False)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)


class ApprovalRequestRecord(TenantScopedMixin, Base):
    __tablename__ = "approval_requests"

    approval_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=False, unique=True
    )
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ApprovalDecisionRecord(TenantScopedMixin, Base):
    __tablename__ = "approval_decisions"

    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    approval_id: Mapped[str] = mapped_column(
        ForeignKey("approval_requests.approval_id"), nullable=False
    )
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    decider_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    parameters_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_version: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_version: Mapped[str] = mapped_column(String(64), nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ExecutionAttemptRecord(TenantScopedMixin, Base):
    __tablename__ = "execution_attempts"

    attempt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=False
    )
    approval_id: Mapped[str] = mapped_column(
        ForeignKey("approval_requests.approval_id"), nullable=False
    )
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ExecutionReconciliationRecord(TenantScopedMixin, Base):
    """Mutable projection for an external side effect whose outcome is unknown."""

    __tablename__ = "execution_reconciliations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "operation_id",
            name="uq_execution_reconciliation_operation",
        ),
        CheckConstraint("version > 0", name="ck_execution_reconciliation_version"),
        CheckConstraint(
            "status IN ('PENDING', 'RESOLVED', 'ESCALATED')",
            name="ck_execution_reconciliation_status",
        ),
        CheckConstraint(
            "outcome IN ('UNKNOWN', 'SUCCEEDED', 'FAILED')",
            name="ck_execution_reconciliation_outcome",
        ),
        Index(
            "ix_execution_reconciliations_status",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_execution_reconciliations_incident",
            "tenant_id",
            "incident_id",
            "created_at",
        ),
    )

    reconciliation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=False
    )
    approval_id: Mapped[str] = mapped_column(
        ForeignKey("approval_requests.approval_id"), nullable=False
    )
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    initial_execution_attempt_id: Mapped[str] = mapped_column(
        ForeignKey("execution_attempts.attempt_id"), nullable=False
    )
    resolved_execution_attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("execution_attempts.attempt_id"), nullable=True
    )
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    external_reference_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ExecutionReconciliationEventRecord(TenantScopedMixin, Base):
    """Append-only audit event for an execution reconciliation task."""

    __tablename__ = "execution_reconciliation_events"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "reconciliation_id",
            "sequence",
            name="uq_execution_reconciliation_event_sequence",
        ),
        CheckConstraint("sequence > 0", name="ck_execution_reconciliation_event_sequence"),
        Index(
            "ix_execution_reconciliation_events_task",
            "tenant_id",
            "reconciliation_id",
            "sequence",
        ),
    )

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    reconciliation_id: Mapped[str] = mapped_column(
        ForeignKey("execution_reconciliations.reconciliation_id"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)


class CustomerNotificationDeliveryRecord(TenantScopedMixin, Base):
    """Tenant-scoped external notification ledger without raw destinations."""

    __tablename__ = "customer_notification_deliveries"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "operation_id",
            name="uq_customer_notification_delivery_operation",
        ),
        CheckConstraint("version > 0", name="ck_customer_notification_delivery_version"),
        CheckConstraint(
            "status IN ('PENDING', 'SUBMITTED', 'RECONCILING', 'DELIVERED', "
            "'FAILED', 'UNSUBSCRIBED')",
            name="ck_customer_notification_delivery_status",
        ),
        Index(
            "ix_customer_notification_deliveries_incident",
            "tenant_id",
            "incident_id",
            "created_at",
        ),
    )

    delivery_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=False
    )
    approval_id: Mapped[str] = mapped_column(
        ForeignKey("approval_requests.approval_id"), nullable=False
    )
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    portal_update_id: Mapped[str] = mapped_column(
        ForeignKey("customer_service_updates.update_id"), nullable=False
    )
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    contact_point_id: Mapped[str] = mapped_column(String(128), nullable=False)
    contact_channel: Mapped[str] = mapped_column(String(16), nullable=False)
    source_system: Mapped[str] = mapped_column(String(128), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_version: Mapped[str] = mapped_column(String(128), nullable=False)
    source_as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    masked_destination: Mapped[str] = mapped_column(String(320), nullable=False)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_message_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    terminal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_receipt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class CustomerNotificationReceiptRecord(TenantScopedMixin, Base):
    """Normalized append-only receipt; raw bodies and credentials are excluded."""

    __tablename__ = "customer_notification_receipts"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_event_id",
            name="uq_customer_notification_receipt_event",
        ),
        Index(
            "ix_customer_notification_receipts_delivery",
            "tenant_id",
            "delivery_id",
            "occurred_at",
        ),
    )

    receipt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    delivery_id: Mapped[str] = mapped_column(
        ForeignKey("customer_notification_deliveries.delivery_id"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_message_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    signature_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    processing_result: Mapped[str] = mapped_column(String(64), nullable=False)


class PartReservationRecord(TenantScopedMixin, Base):
    __tablename__ = "part_reservations"
    __table_args__ = (
        UniqueConstraint("tenant_id", "operation_id", name="uq_part_reservation_operation"),
    )

    reservation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    part_number: Mapped[str] = mapped_column(String(128), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(128), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PartIssueRecord(TenantScopedMixin, Base):
    """Governed WMS issue lifecycle for one WorkOrder reservation."""

    __tablename__ = "part_issues"
    __table_args__ = (
        UniqueConstraint("tenant_id", "operation_id", name="uq_part_issue_operation"),
        UniqueConstraint("tenant_id", "work_order_id", name="uq_part_issue_work_order"),
        UniqueConstraint("tenant_id", "reservation_id", name="uq_part_issue_reservation"),
        CheckConstraint("quantity > 0", name="ck_part_issue_quantity"),
        CheckConstraint(
            "status IN ('PENDING_APPROVAL', 'APPROVED', 'REJECTED', "
            "'RECONCILING', 'ISSUED', 'FAILED')",
            name="ck_part_issue_status",
        ),
        CheckConstraint("version > 0", name="ck_part_issue_version"),
        Index("ix_part_issues_work_order", "tenant_id", "work_order_id", "created_at"),
    )

    part_issue_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=False
    )
    approval_id: Mapped[str] = mapped_column(
        ForeignKey("approval_requests.approval_id"), nullable=False
    )
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    reservation_id: Mapped[str] = mapped_column(
        ForeignKey("part_reservations.reservation_id"), nullable=False
    )
    part_number: Mapped[str] = mapped_column(String(128), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    external_issue_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_record_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reconciliation_id: Mapped[str | None] = mapped_column(
        ForeignKey("execution_reconciliations.reconciliation_id"), nullable=True
    )
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class PartMaterialMovementRecord(TenantScopedMixin, Base):
    """Append-only governed consumption or return lifecycle for one issued part."""

    __tablename__ = "part_material_movements"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "operation_id",
            name="uq_part_material_movement_operation",
        ),
        CheckConstraint("quantity > 0", name="ck_part_material_movement_quantity"),
        CheckConstraint(
            "movement_kind IN ('CONSUME', 'RETURN')",
            name="ck_part_material_movement_kind",
        ),
        CheckConstraint(
            "status IN ('PENDING_APPROVAL', 'APPROVED', 'REJECTED', "
            "'RECONCILING', 'APPLIED', 'FAILED')",
            name="ck_part_material_movement_status",
        ),
        CheckConstraint("version > 0", name="ck_part_material_movement_version"),
        Index(
            "ix_part_material_movements_work_order",
            "tenant_id",
            "work_order_id",
            "created_at",
        ),
        Index(
            "uq_part_material_movement_live_field_entry",
            "tenant_id",
            "field_entry_id",
            unique=True,
            postgresql_where=text(
                "field_entry_id IS NOT NULL AND movement_kind = 'CONSUME' AND "
                "status IN ('PENDING_APPROVAL', 'APPROVED', 'RECONCILING', 'APPLIED')"
            ),
            sqlite_where=text(
                "field_entry_id IS NOT NULL AND movement_kind = 'CONSUME' AND "
                "status IN ('PENDING_APPROVAL', 'APPROVED', 'RECONCILING', 'APPLIED')"
            ),
        ),
    )

    movement_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=False
    )
    approval_id: Mapped[str] = mapped_column(
        ForeignKey("approval_requests.approval_id"), nullable=False
    )
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    part_issue_id: Mapped[str] = mapped_column(
        ForeignKey("part_issues.part_issue_id"), nullable=False
    )
    reservation_id: Mapped[str] = mapped_column(
        ForeignKey("part_reservations.reservation_id"), nullable=False
    )
    field_entry_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_order_field_entries.entry_id"), nullable=True
    )
    movement_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    part_number: Mapped[str] = mapped_column(String(128), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    external_movement_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_record_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reconciliation_id: Mapped[str | None] = mapped_column(
        ForeignKey("execution_reconciliations.reconciliation_id"), nullable=True
    )
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class PurchaseRequestRecord(TenantScopedMixin, Base):
    __tablename__ = "purchase_requests"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "operation_id",
            name="uq_purchase_request_operation",
        ),
        CheckConstraint("quantity > 0", name="ck_purchase_request_quantity"),
        CheckConstraint(
            "available_quantity_at_submission >= 0",
            name="ck_purchase_request_available_quantity",
        ),
        CheckConstraint(
            "estimated_unit_cost_minor >= 0",
            name="ck_purchase_request_unit_cost",
        ),
        CheckConstraint(
            "estimated_total_cost_minor >= 0",
            name="ck_purchase_request_total_cost",
        ),
        CheckConstraint(
            "delivery_target IN ('INTERNAL', 'ENTERPRISE_ERP')",
            name="ck_purchase_request_delivery_target",
        ),
        CheckConstraint(
            "status IN ('SUBMITTED', 'SUBMITTING', 'RECONCILING', "
            "'ACCEPTED', 'REJECTED', 'CANCELLED')",
            name="ck_purchase_request_status",
        ),
        CheckConstraint("version > 0", name="ck_purchase_request_version"),
        Index(
            "ix_purchase_requests_incident",
            "tenant_id",
            "incident_id",
            "created_at",
        ),
        Index(
            "ix_purchase_requests_status",
            "tenant_id",
            "status",
            "created_at",
        ),
    )

    purchase_request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=False
    )
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    part_number: Mapped[str] = mapped_column(String(128), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    available_quantity_at_submission: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_unit_cost_minor: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_total_cost_minor: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    cost_center: Mapped[str] = mapped_column(String(64), nullable=False)
    justification: Mapped[str] = mapped_column(Text, nullable=False)
    delivery_target: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="INTERNAL",
        server_default=text("'INTERNAL'"),
    )
    provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    profile_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    profile_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    profile_display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    profile_as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    profile_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    external_request_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    terminal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_receipt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    inventory_source: Mapped[str] = mapped_column(String(128), nullable=False)
    inventory_source_record_id: Mapped[str] = mapped_column(String(255), nullable=False)
    inventory_as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ServiceQuotationRecord(TenantScopedMixin, Base):
    """Immutable customer-visible quotation published by one approved operation."""

    __tablename__ = "service_quotations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "operation_id",
            name="uq_service_quotation_operation",
        ),
        UniqueConstraint(
            "tenant_id",
            "incident_id",
            "quotation_version",
            name="uq_service_quotation_incident_version",
        ),
        CheckConstraint(
            "status IN ('PUBLISHED', 'ACCEPTED', 'REJECTED', 'EXPIRED', 'SUPERSEDED')",
            name="ck_service_quotation_status",
        ),
        CheckConstraint("quotation_version > 0", name="ck_service_quotation_version"),
        CheckConstraint("state_version > 0", name="ck_service_quotation_state_version"),
        Index(
            "ix_service_quotations_incident",
            "tenant_id",
            "incident_id",
            "quotation_version",
        ),
    )

    quotation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    diagnosis_run_id: Mapped[str] = mapped_column(
        ForeignKey("diagnosis_runs.diagnosis_run_id"), nullable=False
    )
    diagnosis_version: Mapped[int] = mapped_column(Integer, nullable=False)
    quotation_version: Mapped[int] = mapped_column(Integer, nullable=False)
    supersedes_quotation_id: Mapped[str | None] = mapped_column(
        ForeignKey("service_quotations.quotation_id"), nullable=True
    )
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=False
    )
    approval_id: Mapped[str] = mapped_column(
        ForeignKey("approval_requests.approval_id"), nullable=False
    )
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_version: Mapped[str] = mapped_column(String(128), nullable=False)
    source_as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    line_items_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    subtotal: Mapped[str] = mapped_column(String(32), nullable=False)
    discount: Mapped[str] = mapped_column(String(32), nullable=False)
    tax: Mapped[str] = mapped_column(String(32), nullable=False)
    total: Mapped[str] = mapped_column(String(32), nullable=False)
    entitlement_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class ServiceQuotationDecisionRecord(TenantScopedMixin, Base):
    """Append-only customer decision bound to one immutable quotation version."""

    __tablename__ = "service_quotation_decisions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "quotation_id",
            name="uq_service_quotation_decision_quotation",
        ),
        UniqueConstraint(
            "tenant_id",
            "reporter_subject_id",
            "client_operation_id",
            name="uq_service_quotation_decision_operation",
        ),
        CheckConstraint(
            "decision IN ('ACCEPTED', 'REJECTED')",
            name="ck_service_quotation_decision_value",
        ),
        Index(
            "ix_service_quotation_decisions_incident",
            "tenant_id",
            "incident_id",
            "decided_at",
        ),
    )

    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    quotation_id: Mapped[str] = mapped_column(
        ForeignKey("service_quotations.quotation_id"), nullable=False
    )
    quotation_version: Mapped[int] = mapped_column(Integer, nullable=False)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    reporter_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    client_operation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PurchaseRequestReceiptRecord(TenantScopedMixin, Base):
    """Normalized ERP purchase receipt without raw callback bodies or tenant references."""

    __tablename__ = "purchase_request_receipts"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_event_id",
            name="uq_purchase_request_receipt_event",
        ),
        Index(
            "ix_purchase_request_receipts_request",
            "tenant_id",
            "purchase_request_id",
            "occurred_at",
        ),
    )

    receipt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    purchase_request_id: Mapped[str] = mapped_column(
        ForeignKey("purchase_requests.purchase_request_id"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_request_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    signature_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    processing_result: Mapped[str] = mapped_column(String(64), nullable=False)


class PurchaseFulfillmentRecord(TenantScopedMixin, Base):
    """ERP order fulfillment kept separate from requisition acceptance."""

    __tablename__ = "purchase_fulfillments"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "purchase_request_id",
            name="uq_purchase_fulfillment_request",
        ),
        CheckConstraint(
            "status IN ('APPROVED', 'ORDERED', 'IN_TRANSIT', "
            "'PARTIALLY_RECEIVED', 'RECEIVED', 'REJECTED', 'CANCELLED')",
            name="ck_purchase_fulfillment_status",
        ),
        CheckConstraint(
            "requested_quantity > 0",
            name="ck_purchase_fulfillment_requested_quantity",
        ),
        CheckConstraint(
            "received_quantity >= 0 AND received_quantity <= requested_quantity",
            name="ck_purchase_fulfillment_received_quantity",
        ),
        CheckConstraint("version > 0", name="ck_purchase_fulfillment_version"),
        Index(
            "ix_purchase_fulfillments_request",
            "tenant_id",
            "purchase_request_id",
        ),
    )

    fulfillment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    purchase_request_id: Mapped[str] = mapped_column(
        ForeignKey("purchase_requests.purchase_request_id"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_request_id: Mapped[str] = mapped_column(String(255), nullable=False)
    external_order_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    received_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_delivery_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_event_occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_receipt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class PurchaseFulfillmentReceiptRecord(TenantScopedMixin, Base):
    """Normalized fulfillment callback; raw provider bodies are not retained."""

    __tablename__ = "purchase_fulfillment_receipts"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_event_id",
            name="uq_purchase_fulfillment_receipt_event",
        ),
        Index(
            "ix_purchase_fulfillment_receipts_fulfillment",
            "tenant_id",
            "fulfillment_id",
            "occurred_at",
        ),
    )

    receipt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    fulfillment_id: Mapped[str] = mapped_column(
        ForeignKey("purchase_fulfillments.fulfillment_id"), nullable=False
    )
    purchase_request_id: Mapped[str] = mapped_column(
        ForeignKey("purchase_requests.purchase_request_id"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_request_id: Mapped[str] = mapped_column(String(255), nullable=False)
    external_order_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    received_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_delivery_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    signature_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    processing_result: Mapped[str] = mapped_column(String(64), nullable=False)


class RefundRequestRecord(TenantScopedMixin, Base):
    """Customer compensation request with distinct finance and payment state."""

    __tablename__ = "refund_requests"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "operation_id",
            name="uq_refund_request_operation",
        ),
        UniqueConstraint(
            "tenant_id",
            "work_order_id",
            name="uq_refund_request_work_order",
        ),
        CheckConstraint("amount_minor > 0", name="ck_refund_request_amount"),
        CheckConstraint(
            "delivery_target IN ('INTERNAL', 'ENTERPRISE_FINANCE')",
            name="ck_refund_request_delivery_target",
        ),
        CheckConstraint(
            "status IN ('SUBMITTED', 'SUBMITTING', 'RECONCILING', "
            "'ACCEPTED', 'REJECTED', 'CANCELLED')",
            name="ck_refund_request_status",
        ),
        CheckConstraint(
            "payment_status IN ('NOT_APPLICABLE', 'NOT_STARTED', 'PROCESSING', 'PAID', 'FAILED')",
            name="ck_refund_request_payment_status",
        ),
        CheckConstraint("version > 0", name="ck_refund_request_version"),
        Index(
            "ix_refund_requests_incident",
            "tenant_id",
            "incident_id",
            "created_at",
        ),
        Index(
            "ix_refund_requests_status",
            "tenant_id",
            "status",
            "created_at",
        ),
    )

    refund_request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=False
    )
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    customer_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    customer_confirmation_update_id: Mapped[str] = mapped_column(
        ForeignKey("customer_service_updates.update_id"), nullable=False
    )
    amount_minor: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    cost_center: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    compensation_policy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    delivery_target: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="INTERNAL",
        server_default=text("'INTERNAL'"),
    )
    provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    profile_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    profile_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    profile_display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    profile_as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    profile_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    external_request_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    payment_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="NOT_APPLICABLE",
        server_default=text("'NOT_APPLICABLE'"),
    )
    provider_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    terminal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_receipt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)


class RefundRequestReceiptRecord(TenantScopedMixin, Base):
    """Normalized finance receipt without raw callback or customer payment data."""

    __tablename__ = "refund_request_receipts"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_event_id",
            name="uq_refund_request_receipt_event",
        ),
        Index(
            "ix_refund_request_receipts_request",
            "tenant_id",
            "refund_request_id",
            "occurred_at",
        ),
    )

    receipt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    refund_request_id: Mapped[str] = mapped_column(
        ForeignKey("refund_requests.refund_request_id"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_request_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    payment_status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    signature_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    processing_result: Mapped[str] = mapped_column(String(64), nullable=False)


class FsmAssignmentDeliveryRecord(TenantScopedMixin, Base):
    """Durable enterprise FSM assignment handoff without provider-private HR data."""

    __tablename__ = "fsm_assignment_deliveries"
    __table_args__ = (
        UniqueConstraint("tenant_id", "operation_id", name="uq_fsm_assignment_delivery_operation"),
        CheckConstraint(
            "status IN ('SUBMITTING', 'RECONCILING', 'ACCEPTED', 'REJECTED', 'CANCELLED')",
            name="ck_fsm_assignment_delivery_status",
        ),
        CheckConstraint("version > 0", name="ck_fsm_assignment_delivery_version"),
        Index(
            "ix_fsm_assignment_deliveries_work_order",
            "tenant_id",
            "work_order_id",
            "created_at",
        ),
    )

    delivery_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=False
    )
    approval_id: Mapped[str] = mapped_column(
        ForeignKey("approval_requests.approval_id"), nullable=False
    )
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_profile_id: Mapped[str] = mapped_column(String(128), nullable=False)
    profile_version: Mapped[str] = mapped_column(String(128), nullable=False)
    assignee_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_binding: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    external_assignment_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    terminal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_receipt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class FsmAssignmentReceiptRecord(TenantScopedMixin, Base):
    """Normalized signed FSM receipt; raw callbacks are intentionally excluded."""

    __tablename__ = "fsm_assignment_receipts"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_event_id",
            name="uq_fsm_assignment_receipt_event",
        ),
        Index(
            "ix_fsm_assignment_receipts_delivery",
            "tenant_id",
            "delivery_id",
            "occurred_at",
        ),
    )

    receipt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    delivery_id: Mapped[str] = mapped_column(
        ForeignKey("fsm_assignment_deliveries.delivery_id"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_profile_id: Mapped[str] = mapped_column(String(128), nullable=False)
    assignee_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_assignment_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    signature_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    processing_result: Mapped[str] = mapped_column(String(64), nullable=False)


class WorkOrderRecord(TenantScopedMixin, Base):
    __tablename__ = "work_orders"
    __table_args__ = (
        CheckConstraint(
            "creation_mode IN ('PARTS_RESERVATION', 'SERVICE_AUTHORIZATION')",
            name="ck_work_order_creation_mode",
        ),
        CheckConstraint(
            "(creation_mode = 'PARTS_RESERVATION' AND reservation_id IS NOT NULL "
            "AND authorization_type IS NULL AND service_quotation_id IS NULL) OR "
            "(creation_mode = 'SERVICE_AUTHORIZATION' AND reservation_id IS NULL "
            "AND diagnosis_run_id IS NOT NULL AND diagnosis_version IS NOT NULL "
            "AND ((authorization_type = 'COVERED_SERVICE' "
            "AND service_quotation_id IS NULL) OR "
            "(authorization_type = 'ACCEPTED_QUOTATION' "
            "AND service_quotation_id IS NOT NULL)))",
            name="ck_work_order_creation_binding",
        ),
        UniqueConstraint(
            "tenant_id",
            "proposal_id",
            name="uq_work_order_proposal",
        ),
        UniqueConstraint(
            "tenant_id",
            "pending_assignment_operation_id",
            name="uq_work_order_pending_assignment_operation",
        ),
    )

    work_order_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=False
    )
    reservation_id: Mapped[str | None] = mapped_column(
        ForeignKey("part_reservations.reservation_id"), nullable=True
    )
    creation_mode: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="PARTS_RESERVATION",
        server_default=text("'PARTS_RESERVATION'"),
    )
    authorization_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    diagnosis_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("diagnosis_runs.diagnosis_run_id"), nullable=True
    )
    diagnosis_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    service_quotation_id: Mapped[str | None] = mapped_column(
        ForeignKey("service_quotations.quotation_id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    assigned_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    pending_assignment_operation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    part_issue_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    part_accounting_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    priority: Mapped[str] = mapped_column(String(16), nullable=False, default="NORMAL")
    sla_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    service_window_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    service_window_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class WorkOrderEventRecord(TenantScopedMixin, Base):
    __tablename__ = "work_order_events"
    __table_args__ = (
        UniqueConstraint("tenant_id", "work_order_id", "sequence", name="uq_work_order_event"),
    )

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WorkOrderControlRecord(TenantScopedMixin, Base):
    __tablename__ = "work_order_controls"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "work_order_id",
            "work_order_version",
            name="uq_work_order_control_version",
        ),
        Index("ix_work_order_controls_work_order", "tenant_id", "work_order_id", "occurred_at"),
    )

    control_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    work_order_version: Mapped[int] = mapped_column(Integer, nullable=False)
    command_type: Mapped[str] = mapped_column(String(32), nullable=False)
    previous_status: Mapped[str] = mapped_column(String(32), nullable=False)
    target_status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    recovery_condition: Mapped[str | None] = mapped_column(Text, nullable=True)
    responsible_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    service_window_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    service_window_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sla_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WorkOrderFieldEntryRecord(TenantScopedMixin, Base):
    """Append-only field execution fact with an offline-safe operation identity."""

    __tablename__ = "work_order_field_entries"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "work_order_id",
            "sequence",
            name="uq_work_order_field_entry_sequence",
        ),
        UniqueConstraint(
            "tenant_id",
            "work_order_id",
            "client_operation_id",
            name="uq_work_order_field_entry_operation",
        ),
        Index(
            "ix_work_order_field_entries_work_order",
            "tenant_id",
            "work_order_id",
            "occurred_at",
        ),
    )

    entry_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    client_operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    entry_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WorkOrderFieldEvidenceUploadRecord(TenantScopedMixin, Base):
    """Immutable business binding for one quarantined field-media upload."""

    __tablename__ = "work_order_field_evidence_uploads"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "work_order_id",
            "uploaded_by_subject_id",
            "client_operation_id",
            name="uq_work_order_field_evidence_operation",
        ),
        UniqueConstraint(
            "tenant_id",
            "media_id",
            name="uq_work_order_field_evidence_media",
        ),
        Index(
            "ix_work_order_field_evidence_uploads_work_order",
            "tenant_id",
            "work_order_id",
            "occurred_at",
        ),
    )

    evidence_upload_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    source_draft_id: Mapped[str] = mapped_column(
        ForeignKey("incident_drafts.draft_id"), nullable=False
    )
    media_id: Mapped[str] = mapped_column(ForeignKey("media_objects.media_id"), nullable=False)
    uploaded_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    client_operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    declared_mime: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    work_order_version: Mapped[int] = mapped_column(Integer, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class CustomerServiceUpdateRecord(TenantScopedMixin, Base):
    """Append-only customer update or service-result confirmation."""

    __tablename__ = "customer_service_updates"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "incident_id",
            "client_operation_id",
            name="uq_customer_service_update_operation",
        ),
        Index(
            "ix_customer_service_updates_incident",
            "tenant_id",
            "incident_id",
            "occurred_at",
        ),
        Index(
            "ix_customer_service_updates_work_order",
            "tenant_id",
            "work_order_id",
            "occurred_at",
        ),
    )

    update_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    work_order_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=True
    )
    client_operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    update_type: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    result_accepted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    satisfaction_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WorkOrderAssignmentRecord(TenantScopedMixin, Base):
    __tablename__ = "work_order_assignments"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "operation_id",
            name="uq_work_order_assignment_operation",
        ),
        Index(
            "ix_work_order_assignments_proposal",
            "tenant_id",
            "proposal_id",
        ),
    )

    assignment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    assignee_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    assigned_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    operation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    proposal_id: Mapped[str | None] = mapped_column(
        ForeignKey("action_proposals.proposal_id"), nullable=True
    )
    fsm_assignment_delivery_id: Mapped[str | None] = mapped_column(
        ForeignKey("fsm_assignment_deliveries.delivery_id"), nullable=True
    )


class WorkOrderOfflinePackRecord(TenantScopedMixin, Base):
    """Immutable, expiring read snapshot for one assigned field engineer."""

    __tablename__ = "work_order_offline_packs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "subject_id",
            "idempotency_key",
            name="uq_work_order_offline_pack_idempotency",
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'SUPERSEDED', 'REVOKED')",
            name="ck_work_order_offline_pack_status",
        ),
        CheckConstraint("version > 0", name="ck_work_order_offline_pack_version"),
        Index(
            "ix_work_order_offline_pack_current",
            "tenant_id",
            "subject_id",
            "work_order_id",
            "status",
            "issued_at",
        ),
    )

    pack_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    work_order_version: Mapped[int] = mapped_column(Integer, nullable=False)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    asset_version: Mapped[int] = mapped_column(Integer, nullable=False)
    assignment_id: Mapped[str] = mapped_column(
        ForeignKey("work_order_assignments.assignment_id"), nullable=False
    )
    assignment_assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    revocation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class WorkOrderEdgeDiagnosisPackRecord(TenantScopedMixin, Base):
    """Short-lived signed bridge from a current offline pack to llama.cpp."""

    __tablename__ = "work_order_edge_diagnosis_packs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "subject_id",
            "idempotency_key",
            name="uq_work_order_edge_pack_idempotency",
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'SUPERSEDED')",
            name="ck_work_order_edge_pack_status",
        ),
        CheckConstraint("version > 0", name="ck_work_order_edge_pack_version"),
        Index(
            "ix_work_order_edge_pack_current",
            "tenant_id",
            "subject_id",
            "work_order_id",
            "status",
            "issued_at",
        ),
    )

    pack_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    work_order_version: Mapped[int] = mapped_column(Integer, nullable=False)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    asset_version: Mapped[int] = mapped_column(Integer, nullable=False)
    assignment_id: Mapped[str] = mapped_column(
        ForeignKey("work_order_assignments.assignment_id"), nullable=False
    )
    assignment_assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    offline_pack_id: Mapped[str] = mapped_column(
        ForeignKey("work_order_offline_packs.pack_id"), nullable=False
    )
    offline_pack_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.release_id"), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    model_file: Mapped[str] = mapped_column(String(255), nullable=False)
    model_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_bundle_id: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_bundle_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    index_release_id: Mapped[str] = mapped_column(
        ForeignKey("index_releases.release_id"), nullable=False
    )
    index_content_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    key_id: Mapped[str] = mapped_column(String(128), nullable=False)
    claims_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    signed_token: Mapped[str] = mapped_column(Text, nullable=False)
    pack_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class WorkOrderEdgeDiagnosisCandidateRecord(TenantScopedMixin, Base):
    """Untrusted edge result retained as an immutable human-review candidate."""

    __tablename__ = "work_order_edge_diagnosis_candidates"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "subject_id",
            "client_operation_id",
            name="uq_work_order_edge_candidate_operation",
        ),
        CheckConstraint(
            "status IN ('PENDING_REVIEW', 'ACCEPTED', 'REJECTED')",
            name="ck_work_order_edge_candidate_status",
        ),
        CheckConstraint("version > 0", name="ck_work_order_edge_candidate_version"),
        Index(
            "ix_work_order_edge_candidate_work_order",
            "tenant_id",
            "work_order_id",
            "created_at",
        ),
    )

    candidate_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    pack_id: Mapped[str] = mapped_column(
        ForeignKey("work_order_edge_diagnosis_packs.pack_id"), nullable=False
    )
    client_operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    result_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    runtime_evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    security_policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reviewed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class WorkOrderCompletionRecord(TenantScopedMixin, Base):
    __tablename__ = "work_order_completions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "work_order_id",
            "round_number",
            name="uq_work_order_completion_round",
        ),
        CheckConstraint("round_number > 0", name="ck_work_order_completion_round"),
    )

    completion_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    completed_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    field_entry_sequence_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    field_entry_sequence_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    root_cause: Mapped[str] = mapped_column(Text, nullable=False)
    actions: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    evidence_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    part_reservation_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    cost_amount: Mapped[str | None] = mapped_column(String(64), nullable=True)
    customer_confirmation: Mapped[str | None] = mapped_column(String(255), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WorkOrderVerificationRecord(TenantScopedMixin, Base):
    __tablename__ = "work_order_verifications"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "completion_id",
            name="uq_work_order_verification_completion",
        ),
        CheckConstraint("round_number > 0", name="ck_work_order_verification_round"),
    )

    verification_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    completion_id: Mapped[str] = mapped_column(
        ForeignKey("work_order_completions.completion_id"), nullable=False
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    verifier_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WorkOrderReworkRecord(TenantScopedMixin, Base):
    """Explicit rework round opened by an independent failed verification."""

    __tablename__ = "work_order_reworks"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "failed_verification_id",
            name="uq_work_order_rework_verification",
        ),
        UniqueConstraint(
            "tenant_id",
            "work_order_id",
            "round_number",
            name="uq_work_order_rework_round",
        ),
        CheckConstraint("round_number > 1", name="ck_work_order_rework_round"),
        CheckConstraint(
            "status IN ('OPEN', 'RESUBMITTED')",
            name="ck_work_order_rework_status",
        ),
        Index(
            "ix_work_order_reworks_work_order",
            "tenant_id",
            "work_order_id",
            "opened_at",
        ),
    )

    rework_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    source_completion_id: Mapped[str] = mapped_column(
        ForeignKey("work_order_completions.completion_id"), nullable=False
    )
    failed_verification_id: Mapped[str] = mapped_column(
        ForeignKey("work_order_verifications.verification_id"), nullable=False
    )
    resolved_completion_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_order_completions.completion_id"), nullable=True
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_sequence_checkpoint: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EventOutboxRecord(TenantScopedMixin, Base):
    """Committed domain event waiting for Debezium CDC publication."""

    __tablename__ = "event_outbox"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "aggregate_type",
            "aggregate_id",
            "sequence",
            name="uq_event_outbox_aggregate_sequence",
        ),
    )

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    event_version: Mapped[int] = mapped_column(Integer, nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(128), nullable=False)
    aggregate_version: Mapped[int] = mapped_column(Integer, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    producer: Mapped[str] = mapped_column(String(128), nullable=False)
    partition_key: Mapped[str] = mapped_column(String(255), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    trace_context_profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_traceparent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_tracestate: Mapped[str | None] = mapped_column(String(512), nullable=True)
    tracingspancontext: Mapped[str | None] = mapped_column(Text, nullable=True)
    correlation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    causation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    data_classification: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class EventInboxRecord(TenantScopedMixin, Base):
    __tablename__ = "event_inbox"
    __table_args__ = (
        UniqueConstraint(
            "consumer_name",
            "event_id",
            name="uq_event_inbox_consumer_event",
        ),
    )

    inbox_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    consumer_name: Mapped[str] = mapped_column(String(128), nullable=False)
    event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    event_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(128), nullable=False)
    aggregate_version: Mapped[int] = mapped_column(Integer, nullable=False)
    trace_context_propagation: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parent_trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    result_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EventAggregateProjectionRecord(TenantScopedMixin, Base):
    __tablename__ = "event_aggregate_projections"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "consumer_name",
            "aggregate_type",
            "aggregate_id",
            name="uq_event_projection_consumer_aggregate",
        ),
    )

    projection_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    consumer_name: Mapped[str] = mapped_column(String(128), nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(128), nullable=False)
    last_aggregate_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    gap_expected_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gap_received_version: Mapped[int | None] = mapped_column(Integer, nullable=True)


class FeedbackCandidateRecord(TenantScopedMixin, Base):
    __tablename__ = "feedback_candidates"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "source_event_id",
            name="uq_feedback_candidate_source_event",
        ),
    )

    candidate_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    completion_id: Mapped[str] = mapped_column(
        ForeignKey("work_order_completions.completion_id"), nullable=False
    )
    verification_id: Mapped[str] = mapped_column(
        ForeignKey("work_order_verifications.verification_id"), nullable=False
    )
    source_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    data_classification: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    eligibility_status: Mapped[str] = mapped_column(String(32), nullable=False)
    allow_training: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    consent_basis: Mapped[str | None] = mapped_column(String(255), nullable=True)
    license_status: Mapped[str] = mapped_column(String(32), nullable=False)
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lineage_origin: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class DataEligibilityDecisionRecord(TenantScopedMixin, Base):
    """Append-only purpose, license and retention decision for one candidate."""

    __tablename__ = "data_eligibility_decisions"

    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("feedback_candidates.candidate_id"), nullable=False
    )
    purpose: Mapped[str] = mapped_column(String(128), nullable=False)
    allow_training: Mapped[bool] = mapped_column(Boolean, nullable=False)
    consent_basis: Mapped[str | None] = mapped_column(String(255), nullable=True)
    license_status: Mapped[str] = mapped_column(String(32), nullable=False)
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_version: Mapped[int] = mapped_column(Integer, nullable=False)


class DlpProcessingResultRecord(TenantScopedMixin, Base):
    """Immutable redacted derivative; authoritative business text remains untouched."""

    __tablename__ = "dlp_processing_results"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "candidate_id",
            "source_content_hash",
            "policy_version",
            name="uq_dlp_candidate_source_policy",
        ),
    )

    dlp_result_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("feedback_candidates.candidate_id"), nullable=False
    )
    source_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    redacted_content: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    findings: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    residual_entity_types: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    processed_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_version: Mapped[int] = mapped_column(Integer, nullable=False)


class AnnotationTaskRecord(TenantScopedMixin, Base):
    __tablename__ = "annotation_tasks"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_annotation_task_idempotency",
        ),
    )

    task_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("feedback_candidates.candidate_id"), nullable=False
    )
    dlp_result_id: Mapped[str] = mapped_column(
        ForeignKey("dlp_processing_results.dlp_result_id"), nullable=False
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    project_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_task_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_version: Mapped[str] = mapped_column(String(128), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(128), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(32), nullable=False)
    required_reviews: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    consistency_status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class AnnotationRevisionRecord(TenantScopedMixin, Base):
    __tablename__ = "annotation_revisions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "task_id",
            "external_annotation_id",
            "external_version",
            name="uq_annotation_revision_external_version",
        ),
    )

    revision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("annotation_tasks.task_id"), nullable=False)
    external_annotation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_version: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewer_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    labels: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    labels_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    task_payload_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AnnotationReviewRecord(TenantScopedMixin, Base):
    __tablename__ = "annotation_reviews"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "task_id",
            "revision_id",
            name="uq_annotation_review_revision",
        ),
    )

    review_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("annotation_tasks.task_id"), nullable=False)
    revision_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_revisions.revision_id"), nullable=False
    )
    reviewer_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    consistency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    task_payload_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class CurationRunRecord(TenantScopedMixin, Base):
    """One fixed-window, immutable-input dataset curation attempt."""

    __tablename__ = "curation_runs"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    engine: Mapped[str] = mapped_column(String(32), nullable=False)
    contract_version: Mapped[str] = mapped_column(String(128), nullable=False)
    code_version: Mapped[str] = mapped_column(String(128), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    input_manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    input_count: Mapped[int] = mapped_column(Integer, nullable=False)
    exclusion_report: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)


class DatasetSnapshotRecord(TenantScopedMixin, Base):
    """Catalog entry for an immutable governed dataset snapshot."""

    __tablename__ = "dataset_snapshots"
    __table_args__ = (UniqueConstraint("tenant_id", "run_id", name="uq_dataset_snapshot_run"),)

    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("curation_runs.run_id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    contract_version: Mapped[str] = mapped_column(String(128), nullable=False)
    input_manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    manifest_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    manifest_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    quality_report_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    split_counts: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False)
    source_work_order_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    lineage_status: Mapped[str] = mapped_column(String(32), nullable=False)
    training_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    base_snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_snapshots.snapshot_id"), nullable=True
    )
    augmentation_contract_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sample_origin_counts: Mapped[dict[str, int] | None] = mapped_column(JSON, nullable=True)
    synthetic_split_counts: Mapped[dict[str, int] | None] = mapped_column(JSON, nullable=True)


class DatasetArtifactRecord(TenantScopedMixin, Base):
    """Checksum-addressed member of a published dataset snapshot."""

    __tablename__ = "dataset_artifacts"
    __table_args__ = (
        UniqueConstraint("tenant_id", "snapshot_id", "object_key", name="uq_dataset_artifact_key"),
    )

    artifact_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_snapshots.snapshot_id"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    split: Mapped[str | None] = mapped_column(String(32), nullable=True)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_hash: Mapped[str] = mapped_column(String(128), nullable=False)


class DataLineageRunRecord(TenantScopedMixin, Base):
    """OpenLineage delivery state; pending delivery is never trainable."""

    __tablename__ = "data_lineage_runs"
    __table_args__ = (UniqueConstraint("tenant_id", "run_id", name="uq_data_lineage_run"),)

    lineage_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("curation_runs.run_id"), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_snapshots.snapshot_id"), nullable=False
    )
    openlineage_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    job_namespace: Mapped[str] = mapped_column(String(255), nullable=False)
    job_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    source_work_order_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    output_dataset: Mapped[str] = mapped_column(String(1024), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)


class TrainingExperimentRecord(TenantScopedMixin, Base):
    """Immutable training intent plus the controlled state of one reproducible run."""

    __tablename__ = "training_experiments"
    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_training_experiment_idempotency"),
        UniqueConstraint("tenant_id", "mlflow_run_id", name="uq_training_experiment_mlflow_run"),
    )

    experiment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    comparison_group_id: Mapped[str] = mapped_column(String(128), nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_snapshots.snapshot_id"), nullable=False
    )
    dataset_manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    base_model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    base_model_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    tokenizer_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    chat_template_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    git_commit: Mapped[str] = mapped_column(String(128), nullable=False)
    container_digest: Mapped[str] = mapped_column(String(255), nullable=False)
    training_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    distributed_profile: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    random_seeds: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    hardware_topology: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    mlflow_experiment_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metrics: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    cost_summary: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    license_status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class TrainingArtifactRecord(TenantScopedMixin, Base):
    """Checksum-addressed checkpoint, adapter, log or report from an experiment."""

    __tablename__ = "training_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "experiment_id", "object_key", name="uq_training_artifact_key"
        ),
    )

    artifact_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("training_experiments.experiment_id"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class EvaluationSuiteRecord(TenantScopedMixin, Base):
    """Frozen evaluation-only sample set, deliberately separate from training intent."""

    __tablename__ = "evaluation_suites"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", "version", name="uq_evaluation_suite_name_version"),
    )

    suite_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    source_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_snapshots.snapshot_id"), nullable=False
    )
    tier: Mapped[str] = mapped_column(String(32), nullable=False)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    slice_counts: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)


class EvaluationPolicyRecord(TenantScopedMixin, Base):
    """Versioned deterministic release gate; aggregate scores cannot override hard gates."""

    __tablename__ = "evaluation_policies"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", "version", name="uq_evaluation_policy_name_version"),
    )

    policy_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    primary_metric: Mapped[str] = mapped_column(String(128), nullable=False)
    hard_gates: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    thresholds: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    policy_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)


class ModelEvaluationRunRecord(TenantScopedMixin, Base):
    """Immutable baseline comparison and its deterministic candidate decision."""

    __tablename__ = "model_evaluation_runs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_model_evaluation_idempotency"),
    )

    evaluation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    candidate_experiment_id: Mapped[str] = mapped_column(
        ForeignKey("training_experiments.experiment_id"), nullable=False
    )
    baseline_experiment_id: Mapped[str] = mapped_column(
        ForeignKey("training_experiments.experiment_id"), nullable=False
    )
    suite_id: Mapped[str] = mapped_column(ForeignKey("evaluation_suites.suite_id"), nullable=False)
    policy_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_policies.policy_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    primary_metric: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_score: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_score: Mapped[float] = mapped_column(Float, nullable=False)
    quality_delta: Mapped[float] = mapped_column(Float, nullable=False)
    ci_low: Mapped[float] = mapped_column(Float, nullable=False)
    ci_high: Mapped[float] = mapped_column(Float, nullable=False)
    latency_improvement: Mapped[float] = mapped_column(Float, nullable=False)
    cost_improvement: Mapped[float] = mapped_column(Float, nullable=False)
    hard_gate_results: Mapped[dict[str, bool]] = mapped_column(JSON, nullable=False)
    slice_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    comparison_kind: Mapped[str] = mapped_column(String(64), nullable=False, default="MODEL_METHOD")
    comparison_context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    report_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    triggered_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)


class ModelEvaluationJobRecord(TenantScopedMixin, Base):
    """Independent evaluator work item; callers register references, never scores."""

    __tablename__ = "model_evaluation_jobs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_model_eval_job_idempotency"),
    )

    job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    candidate_experiment_id: Mapped[str] = mapped_column(
        ForeignKey("training_experiments.experiment_id"), nullable=False
    )
    baseline_experiment_id: Mapped[str] = mapped_column(
        ForeignKey("training_experiments.experiment_id"), nullable=False
    )
    suite_id: Mapped[str] = mapped_column(ForeignKey("evaluation_suites.suite_id"), nullable=False)
    policy_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_policies.policy_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    target_profile: Mapped[str] = mapped_column(String(32), nullable=False)
    runner_git_commit: Mapped[str] = mapped_column(String(128), nullable=False)
    container_digest: Mapped[str] = mapped_column(String(255), nullable=False)
    runner_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    result_evaluation_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_evaluation_runs.evaluation_id"), nullable=True
    )
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class SupplyChainEvidenceRecord(TenantScopedMixin, Base):
    """Immutable, verifier-produced evidence for one runtime image and model bundle."""

    __tablename__ = "supply_chain_evidence"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "verification_hash", name="uq_supply_chain_evidence_verification"
        ),
    )

    evidence_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    image_repository: Mapped[str] = mapped_column(String(512), nullable=False)
    image_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    model_artifact_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    source_repository: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    sbom_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    sbom_format: Mapped[str] = mapped_column(String(32), nullable=False)
    vulnerability_report_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    vulnerability_scan_status: Mapped[str] = mapped_column(String(32), nullable=False)
    maximum_vulnerability_severity: Mapped[str] = mapped_column(String(32), nullable=False)
    vulnerability_scanner: Mapped[str] = mapped_column(String(128), nullable=False)
    license_report_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    license_status: Mapped[str] = mapped_column(String(32), nullable=False)
    provenance_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    image_signature_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    model_signature_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    certificate_identity: Mapped[str] = mapped_column(String(1024), nullable=False)
    certificate_oidc_issuer: Mapped[str] = mapped_column(String(1024), nullable=False)
    verifier_version: Mapped[str] = mapped_column(String(64), nullable=False)
    verification_status: Mapped[str] = mapped_column(String(32), nullable=False)
    verification_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    verified_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StagingAcceptanceAttestationRecord(TenantScopedMixin, Base):
    """Immutable verifier-owned proof for one signed Staging acceptance manifest."""

    __tablename__ = "staging_acceptance_attestations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "verification_hash",
            name="uq_staging_acceptance_attestation_verification",
        ),
        Index(
            "ix_staging_acceptance_attestation_manifest",
            "tenant_id",
            "manifest_id",
            "verified_at",
        ),
    )

    evidence_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_id: Mapped[str] = mapped_column(String(128), nullable=False)
    manifest_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(128), nullable=False)
    git_commit: Mapped[str] = mapped_column(String(64), nullable=False)
    signature_bundle_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    certificate_identity: Mapped[str] = mapped_column(String(1024), nullable=False)
    certificate_oidc_issuer: Mapped[str] = mapped_column(String(1024), nullable=False)
    verifier_version: Mapped[str] = mapped_column(String(64), nullable=False)
    verification_status: Mapped[str] = mapped_column(String(32), nullable=False)
    verification_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    verified_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EvaluationArtifactRecord(TenantScopedMixin, Base):
    """Immutable case-level evidence or framework report for a final evaluation."""

    __tablename__ = "evaluation_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "evaluation_id", "object_key", name="uq_evaluation_artifact_key"
        ),
    )

    artifact_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_id: Mapped[str] = mapped_column(
        ForeignKey("model_evaluation_runs.evaluation_id"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class PromptBundleRecord(TenantScopedMixin, Base):
    """Prompt metadata and review state; raw prompt text remains image-bound."""

    __tablename__ = "prompt_bundles"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "name",
            "bundle_version",
            name="uq_prompt_bundle_name_version",
        ),
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_prompt_bundle_idempotency",
        ),
        Index("ix_prompt_bundle_status", "tenant_id", "status"),
    )

    prompt_bundle_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    bundle_version: Mapped[str] = mapped_column(String(64), nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    source_commit: Mapped[str] = mapped_column(String(128), nullable=False)
    source_path: Mapped[str] = mapped_column(String(512), nullable=False)
    section_names_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    required_variables_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    output_schema_name: Mapped[str] = mapped_column(String(128), nullable=False)
    compatible_model_aliases_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    change_summary: Mapped[str] = mapped_column(Text, nullable=False)
    evaluation_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_evaluation_runs.evaluation_id"), nullable=True
    )
    evaluation_report_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class PromptBundleTransitionRecord(TenantScopedMixin, Base):
    """Append-only Prompt Bundle lifecycle evidence."""

    __tablename__ = "prompt_bundle_transitions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "prompt_bundle_id",
            "sequence",
            name="uq_prompt_bundle_transition_sequence",
        ),
    )

    transition_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    prompt_bundle_id: Mapped[str] = mapped_column(
        ForeignKey("prompt_bundles.prompt_bundle_id"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    from_status: Mapped[str] = mapped_column(String(32), nullable=False)
    to_status: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    evidence_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ModelReleaseRecord(TenantScopedMixin, Base):
    """Immutable AIReleaseManifest plus its monotonic deployment state."""

    __tablename__ = "model_releases"
    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_model_release_idempotency"),
        UniqueConstraint("tenant_id", "manifest_hash", name="uq_model_release_manifest"),
    )

    release_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    evaluation_id: Mapped[str] = mapped_column(
        ForeignKey("model_evaluation_runs.evaluation_id"), nullable=False
    )
    candidate_experiment_id: Mapped[str] = mapped_column(
        ForeignKey("training_experiments.experiment_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    target_environment: Mapped[str] = mapped_column(String(32), nullable=False)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    rollback_release_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_releases.release_id"), nullable=True
    )
    traffic_percent: Mapped[float] = mapped_column(Float, nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ModelReleaseTransitionRecord(TenantScopedMixin, Base):
    """Append-only release transition with bound evidence and actor."""

    __tablename__ = "model_release_transitions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "release_id", "sequence", name="uq_model_release_transition_sequence"
        ),
    )

    transition_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.release_id"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    from_status: Mapped[str] = mapped_column(String(32), nullable=False)
    to_status: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    evidence_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ModelReleaseApprovalRecord(TenantScopedMixin, Base):
    """Separation-of-duties decision bound to one immutable manifest hash."""

    __tablename__ = "model_release_approvals"
    __table_args__ = (
        UniqueConstraint("tenant_id", "release_id", name="uq_model_release_approval"),
    )

    approval_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.release_id"), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    decided_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class EnterpriseModelReleaseOnboardingRecord(TenantScopedMixin, Base):
    """Governed intent to register one adopted candidate and request real Shadow."""

    __tablename__ = "enterprise_model_release_onboardings"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "release_id",
            name="uq_enterprise_release_onboarding_release",
        ),
        UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_enterprise_release_onboarding_idempotency",
        ),
        Index(
            "ix_enterprise_release_onboarding_status",
            "tenant_id",
            "automation_status",
            "auto_shadow_enabled",
        ),
        CheckConstraint(
            "component IN ('LLM', 'VLM', 'ASR', 'TTS', 'RUL', 'EMBEDDING', 'RERANKER')",
            name="ck_enterprise_release_onboarding_component",
        ),
        CheckConstraint(
            "automation_status IN "
            "('WAITING_APPROVAL', 'SHADOW_REQUESTED', 'CANCELLED', 'BLOCKED', 'DISABLED')",
            name="ck_enterprise_release_onboarding_status",
        ),
        CheckConstraint(
            "attempt_count >= 0 AND version >= 1",
            name="ck_enterprise_release_onboarding_versions",
        ),
    )

    onboarding_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.release_id"), nullable=False)
    baseline_release_id: Mapped[str] = mapped_column(
        ForeignKey("model_releases.release_id"), nullable=False
    )
    component: Mapped[str] = mapped_column(String(16), nullable=False)
    candidate_experiment_id: Mapped[str] = mapped_column(
        ForeignKey("training_experiments.experiment_id"), nullable=False
    )
    evaluation_id: Mapped[str] = mapped_column(
        ForeignKey("model_evaluation_runs.evaluation_id"), nullable=False
    )
    evidence_chain_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    auto_shadow_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    deployment_plan_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    deployment_plan_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    automation_status: Mapped[str] = mapped_column(String(32), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    last_error: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class EnterpriseModelAssetImportRecord(TenantScopedMixin, Base):
    """Immutable tenant projection of one project-authoritative model candidate."""

    __tablename__ = "enterprise_model_asset_imports"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "component",
            "source_candidate_experiment_id",
            name="uq_enterprise_model_asset_import_source",
        ),
        UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_enterprise_model_asset_import_idempotency",
        ),
        Index(
            "ix_enterprise_model_asset_import_status",
            "tenant_id",
            "status",
            "component",
        ),
        CheckConstraint(
            "component IN ('LLM', 'VLM', 'ASR', 'TTS', 'RUL', 'EMBEDDING', 'RERANKER')",
            name="ck_enterprise_model_asset_import_component",
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'REVOKED')",
            name="ck_enterprise_model_asset_import_status",
        ),
        CheckConstraint(
            "release_scope = 'STAGING_ONLY'",
            name="ck_enterprise_model_asset_import_scope",
        ),
        CheckConstraint(
            "actual_sample_count >= 1 AND version >= 1",
            name="ck_enterprise_model_asset_import_counts",
        ),
    )

    import_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    component: Mapped[str] = mapped_column(String(16), nullable=False)
    source_candidate_experiment_id: Mapped[str] = mapped_column(String(128), nullable=False)
    imported_candidate_experiment_id: Mapped[str] = mapped_column(
        ForeignKey("training_experiments.experiment_id"), nullable=False
    )
    imported_evaluation_id: Mapped[str] = mapped_column(
        ForeignKey("model_evaluation_runs.evaluation_id"), nullable=False
    )
    imported_suite_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_suites.suite_id"), nullable=False
    )
    source_paths_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    source_file_hashes_json: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)
    source_evidence_chain_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_classification: Mapped[str] = mapped_column(String(64), nullable=False)
    source_decision: Mapped[str] = mapped_column(String(128), nullable=False)
    operational_classification: Mapped[str] = mapped_column(String(64), nullable=False)
    actual_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    source_artifact_size_recorded: Mapped[bool] = mapped_column(Boolean, nullable=False)
    release_scope: Mapped[str] = mapped_column(String(32), nullable=False)
    imported_records_json: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    import_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ModelDeploymentRecord(TenantScopedMixin, Base):
    """Desired and externally observed state for one idempotent model deployment."""

    __tablename__ = "model_deployments"
    __table_args__ = (
        UniqueConstraint("tenant_id", "release_id", name="uq_model_deployment_release"),
    )

    deployment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.release_id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False)
    service_name: Mapped[str] = mapped_column(String(128), nullable=False)
    route_name: Mapped[str] = mapped_column(String(128), nullable=False)
    stable_service_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    desired_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    desired_traffic_percent: Mapped[float] = mapped_column(Float, nullable=False)
    observed_traffic_percent: Mapped[float] = mapped_column(Float, nullable=False)
    desired_spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    desired_spec_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    applied_spec_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    provider_revision: Mapped[str | None] = mapped_column(String(255), nullable=True)
    endpoint_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reconciled_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ModelReleaseObservationRecord(TenantScopedMixin, Base):
    """Immutable Prometheus-backed evidence for one release stage and time window."""

    __tablename__ = "model_release_observations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "release_id",
            "stage",
            "sequence",
            name="uq_model_release_observation_sequence",
        ),
    )

    observation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.release_id"), nullable=False)
    deployment_id: Mapped[str] = mapped_column(
        ForeignKey("model_deployments.deployment_id"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    request_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    critical_case_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    metrics_json: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    thresholds_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source_refs_json: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)
    evidence_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    collected_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)


class ModelAliasRecord(TenantScopedMixin, Base):
    """Tenant model alias bound to one verified production release and route."""

    __tablename__ = "model_aliases"
    __table_args__ = (UniqueConstraint("tenant_id", "alias", name="uq_model_alias_tenant_name"),)

    alias_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    alias: Mapped[str] = mapped_column(String(128), nullable=False)
    active_release_id: Mapped[str] = mapped_column(
        ForeignKey("model_releases.release_id"), nullable=False
    )
    deployment_id: Mapped[str] = mapped_column(
        ForeignKey("model_deployments.deployment_id"), nullable=False
    )
    manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    endpoint_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    runtime_profile: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    updated_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ModelGatewayQuotaRecord(TenantScopedMixin, Base):
    """Serialized tenant quota policy used before an inference request is sent."""

    __tablename__ = "model_gateway_quotas"
    __table_args__ = (
        UniqueConstraint("tenant_id", "model_alias", name="uq_model_gateway_quota_alias"),
    )

    quota_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_alias: Mapped[str] = mapped_column(String(128), nullable=False)
    requests_per_minute: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_per_day: Mapped[int] = mapped_column(BigInteger, nullable=False)
    max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    allowed_request_classes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ModelInferenceRecord(TenantScopedMixin, Base):
    """Prompt-minimized, version-bound inference request and result audit record."""

    __tablename__ = "model_inference_requests"

    inference_request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_alias: Mapped[str] = mapped_column(String(128), nullable=False)
    resolved_release_id: Mapped[str] = mapped_column(
        ForeignKey("model_releases.release_id"), nullable=False
    )
    manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    agent_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    diagnosis_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_bundle_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    index_release_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    context_manifest_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    context_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    request_class: Mapped[str] = mapped_column(String(64), nullable=False)
    data_classification: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    response_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    response_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    usage_prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    usage_completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    finish_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_breakdown_json: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    safety_decision: Mapped[str] = mapped_column(String(64), nullable=False)
    guardrail_policy_version: Mapped[str] = mapped_column(
        String(128), nullable=False, default="not-evaluated"
    )
    guardrail_findings_json: Mapped[list[dict[str, str]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    degraded_from: Mapped[str | None] = mapped_column(String(128), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CostPolicyRecord(TenantScopedMixin, Base):
    """Append-only FinOps rates and budgets effective from one UTC instant."""

    __tablename__ = "cost_policies"
    __table_args__ = (
        UniqueConstraint("tenant_id", "version", name="uq_cost_policy_tenant_version"),
        Index("ix_cost_policy_effective", "tenant_id", "effective_at"),
    )

    policy_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    prompt_tokens_per_million_usd: Mapped[float] = mapped_column(Float, nullable=False)
    completion_tokens_per_million_usd: Mapped[float] = mapped_column(Float, nullable=False)
    gpu_hourly_cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    monthly_budget_usd: Mapped[float] = mapped_column(Float, nullable=False)
    scenario_budgets_json: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    warning_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    updated_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)


class CostLedgerBatchRecord(TenantScopedMixin, Base):
    """Immutable evidence envelope for one external FinOps ledger interval."""

    __tablename__ = "cost_ledger_batches"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "imported_by_subject_id",
            "idempotency_key",
            name="uq_cost_ledger_batch_idempotency",
        ),
        UniqueConstraint(
            "tenant_id",
            "source_system",
            "external_batch_id",
            name="uq_cost_ledger_batch_external",
        ),
        CheckConstraint("currency = 'USD'", name="ck_cost_ledger_batch_currency"),
        CheckConstraint(
            "coverage_end > coverage_start",
            name="ck_cost_ledger_batch_coverage",
        ),
        CheckConstraint("entry_count > 0", name="ck_cost_ledger_batch_entry_count"),
        CheckConstraint(
            "total_amount_usd >= 0",
            name="ck_cost_ledger_batch_total_amount",
        ),
        Index(
            "ix_cost_ledger_batch_coverage",
            "tenant_id",
            "coverage_start",
            "coverage_end",
        ),
    )

    batch_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_system: Mapped[str] = mapped_column(String(128), nullable=False)
    external_batch_id: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    coverage_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    coverage_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    covered_categories_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    evidence_uri: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_sha256: Mapped[str] = mapped_column(String(71), nullable=False)
    batch_digest: Mapped[str] = mapped_column(String(71), nullable=False)
    entry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_amount_usd: Mapped[float] = mapped_column(Float, nullable=False)
    imported_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)


class CostLedgerEntryRecord(TenantScopedMixin, Base):
    """Minimal cost fact retained beneath an immutable ledger batch."""

    __tablename__ = "cost_ledger_entries"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "batch_id",
            "external_entry_id",
            name="uq_cost_ledger_entry_external",
        ),
        CheckConstraint("amount_usd >= 0", name="ck_cost_ledger_entry_amount"),
        CheckConstraint(
            "quantity IS NULL OR quantity >= 0",
            name="ck_cost_ledger_entry_quantity",
        ),
        Index(
            "ix_cost_ledger_entry_window",
            "tenant_id",
            "occurred_at",
            "category",
        ),
    )

    entry_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    batch_id: Mapped[str] = mapped_column(
        ForeignKey("cost_ledger_batches.batch_id"), nullable=False, index=True
    )
    external_entry_id: Mapped[str] = mapped_column(String(128), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    amount_usd: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    release_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    scenario: Mapped[str | None] = mapped_column(String(64), nullable=True)
    diagnosis_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    work_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class WorkOrderExpertCollaborationRecord(TenantScopedMixin, Base):
    """WorkOrder-bound invitation for a fixed field/expert participant pair."""

    __tablename__ = "work_order_expert_collaborations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_expert_collaboration_idempotency",
        ),
        CheckConstraint(
            "status IN ('WAITING_EXPERT', 'ACTIVE', 'ENDED', 'EXPIRED')",
            name="ck_expert_collaboration_status",
        ),
        Index(
            "ix_expert_collaboration_work_order_status",
            "tenant_id",
            "work_order_id",
            "status",
        ),
        Index(
            "uq_expert_collaboration_open_work_order",
            "tenant_id",
            "work_order_id",
            unique=True,
            postgresql_where=text("status IN ('WAITING_EXPERT', 'ACTIVE')"),
            sqlite_where=text("status IN ('WAITING_EXPERT', 'ACTIVE')"),
        ),
        Index(
            "ix_expert_collaboration_invited_status",
            "tenant_id",
            "invited_subject_id",
            "status",
        ),
    )

    collaboration_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    work_order_version: Mapped[int] = mapped_column(Integer, nullable=False)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    invited_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ended_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class WorkOrderExpertCollaborationEventRecord(TenantScopedMixin, Base):
    """Append-only state-transition evidence for expert collaboration."""

    __tablename__ = "work_order_expert_collaboration_events"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "collaboration_id",
            "sequence",
            name="uq_expert_collaboration_event_sequence",
        ),
    )

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    collaboration_id: Mapped[str] = mapped_column(
        ForeignKey("work_order_expert_collaborations.collaboration_id"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WorkOrderExpertCollaborationRecommendationRecord(TenantScopedMixin, Base):
    """Immutable expert advice awaiting the field owner's independent review."""

    __tablename__ = "work_order_expert_collaboration_recommendations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "collaboration_id",
            name="uq_expert_recommendation_collaboration",
        ),
        UniqueConstraint(
            "tenant_id",
            "expert_subject_id",
            "idempotency_key",
            name="uq_expert_recommendation_idempotency",
        ),
        UniqueConstraint(
            "tenant_id",
            "field_entry_id",
            name="uq_expert_recommendation_field_entry",
        ),
        CheckConstraint(
            "recommendation_type IN ('ADVICE', 'REQUEST_MORE_EVIDENCE', 'STOP_AND_ESCALATE')",
            name="ck_expert_recommendation_type",
        ),
        CheckConstraint(
            "status IN ('PENDING_FIELD_REVIEW', 'ACCEPTED', 'REJECTED')",
            name="ck_expert_recommendation_status",
        ),
        Index(
            "ix_expert_recommendation_work_order_status",
            "tenant_id",
            "work_order_id",
            "status",
        ),
    )

    recommendation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    collaboration_id: Mapped[str] = mapped_column(
        ForeignKey("work_order_expert_collaborations.collaboration_id"), nullable=False
    )
    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=False
    )
    work_order_version: Mapped[int] = mapped_column(Integer, nullable=False)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    expert_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    recommendation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    basis: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_checks_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    safety_notice: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_entry_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    field_entry_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_order_field_entries.entry_id"), nullable=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class RealtimeMediaSessionRecord(TenantScopedMixin, Base):
    """Short-lived WebRTC control-plane lease; media never lives in this table."""

    __tablename__ = "realtime_media_sessions"
    __table_args__ = (
        UniqueConstraint("tenant_id", "token_digest", name="uq_realtime_media_token"),
    )

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    work_order_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=True
    )
    work_order_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expert_collaboration_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_order_expert_collaborations.collaboration_id"), nullable=True
    )
    diagnosis_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("diagnosis_runs.diagnosis_run_id"), nullable=True
    )
    subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    media_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    token_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    token_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    token_consumed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    media_token_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    media_token_issued_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    lease_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    max_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reconnect_count: Mapped[int] = mapped_column(Integer, nullable=False)
    ended_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class RealtimeTranscriptSegmentRecord(TenantScopedMixin, Base):
    """Time-aligned ASR result awaiting explicit user confirmation."""

    __tablename__ = "realtime_transcript_segments"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "session_id", "sequence", name="uq_realtime_transcript_sequence"
        ),
    )

    segment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("realtime_media_sessions.session_id"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confirmed_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    entities_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    security_policy_version: Mapped[str | None] = mapped_column(
        String(128), nullable=True, default="m6-prompt-injection-v1"
    )
    security_findings_json: Mapped[list[dict[str, str]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    audio_sha256: Mapped[str] = mapped_column(String(128), nullable=False)
    resolved_release_id: Mapped[str] = mapped_column(String(128), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reviewer_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    agent_event_id: Mapped[str | None] = mapped_column(
        ForeignKey("agent_events.event_id"), nullable=True, unique=True
    )
    agent_event_sequence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class RecoveryEvidenceRecord(TenantScopedMixin, Base):
    __tablename__ = "recovery_evidence"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "component",
            "backup_id",
            name="uq_recovery_evidence_backup",
        ),
        Index(
            "ix_recovery_evidence_latest",
            "tenant_id",
            "component",
            "verified_at",
        ),
    )

    evidence_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    component: Mapped[str] = mapped_column(String(64), nullable=False)
    backup_id: Mapped[str] = mapped_column(String(255), nullable=False)
    backup_type: Mapped[str] = mapped_column(String(64), nullable=False)
    recovery_point_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    verification_status: Mapped[str] = mapped_column(String(32), nullable=False)
    verification_method: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    artifact_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    immutable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    encrypted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    verified_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class RecoveryDrillRecord(TenantScopedMixin, Base):
    __tablename__ = "recovery_drills"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_recovery_drill_idempotency",
        ),
        Index(
            "ix_recovery_drill_status",
            "tenant_id",
            "status",
            "completed_at",
        ),
    )

    drill_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    scenario: Mapped[str] = mapped_column(String(128), nullable=False)
    scope_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    outage_detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    results_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    blocker_codes_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    completed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    objective_met: Mapped[bool] = mapped_column(Boolean, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class SecurityExerciseRecord(TenantScopedMixin, Base):
    """Immutable red-team campaign result produced by an isolated verifier."""

    __tablename__ = "security_exercises"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_security_exercise_idempotency",
        ),
        Index(
            "ix_security_exercise_status",
            "tenant_id",
            "status",
            "completed_at",
        ),
    )

    exercise_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    target_environment: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    scenario_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    results_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    blocker_codes_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    report_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    verifier_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    requested_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    completed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ProductionDefectRecord(TenantScopedMixin, Base):
    """Release-blocking finding opened by a failed governed exercise scenario."""

    __tablename__ = "production_defects"
    __table_args__ = (
        UniqueConstraint("tenant_id", "defect_key", name="uq_production_defect_key"),
        Index(
            "ix_production_defect_status",
            "tenant_id",
            "status",
            "severity",
        ),
    )

    defect_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    defect_key: Mapped[str] = mapped_column(String(255), nullable=False)
    exercise_id: Mapped[str] = mapped_column(
        ForeignKey("security_exercises.exercise_id"), nullable=False
    )
    scenario_id: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    release_blocking: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    description_code: Mapped[str] = mapped_column(String(128), nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    closure_evidence_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    closure_evidence_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ProductionAcceptanceRecord(TenantScopedMixin, Base):
    """Server-evaluated production dossier bound to one model release and evidence set."""

    __tablename__ = "production_acceptances"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_production_acceptance_idempotency",
        ),
        Index(
            "ix_production_acceptance_status",
            "tenant_id",
            "status",
            "evaluated_at",
        ),
    )

    acceptance_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    release_scope: Mapped[str] = mapped_column(String(128), nullable=False)
    target_environment: Mapped[str] = mapped_column(String(32), nullable=False)
    security_exercise_id: Mapped[str] = mapped_column(
        ForeignKey("security_exercises.exercise_id"), nullable=False
    )
    model_release_id: Mapped[str] = mapped_column(
        ForeignKey("model_releases.release_id"), nullable=False
    )
    business_evidence_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    business_evidence_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    rollback_evidence_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    rollback_evidence_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    blocker_codes_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    readiness_snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    signed_off_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ProductionAcceptanceSignoffRecord(TenantScopedMixin, Base):
    """Append-only role sign-off bound to a readiness snapshot digest."""

    __tablename__ = "production_acceptance_signoffs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "acceptance_id",
            "signoff_role",
            name="uq_production_acceptance_signoff_role",
        ),
    )

    signoff_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    acceptance_id: Mapped[str] = mapped_column(
        ForeignKey("production_acceptances.acceptance_id"), nullable=False
    )
    signoff_role: Mapped[str] = mapped_column(String(32), nullable=False)
    readiness_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    evidence_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    signed_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    signed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TelemetryEventRecord(TenantScopedMixin, Base):
    """Validated device observation retained independently from derived windows."""

    __tablename__ = "telemetry_events"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "asset_id",
            "schema_version",
            "sequence",
            name="uq_telemetry_stream_sequence",
        ),
        Index(
            "ix_telemetry_events_asset_time",
            "tenant_id",
            "asset_id",
            "event_time",
        ),
    )

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source_system: Mapped[str] = mapped_column(String(128), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    measurements_json: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    quality_flags_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    payload_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    ingestion_status: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class TelemetryStreamStateRecord(TenantScopedMixin, Base):
    """Monotonic stream watermark used to classify late and out-of-order events."""

    __tablename__ = "telemetry_stream_states"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "asset_id",
            "schema_version",
            name="uq_telemetry_stream_state",
        ),
    )

    stream_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    max_event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    max_sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    watermark_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class TelemetryWindowRecord(TenantScopedMixin, Base):
    """Immutable feature snapshot for one equipment event-time interval."""

    __tablename__ = "telemetry_windows"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "feature_snapshot_id",
            name="uq_telemetry_feature_snapshot",
        ),
        Index(
            "ix_telemetry_windows_asset_time",
            "tenant_id",
            "asset_id",
            "window_start",
            "window_end",
        ),
    )

    window_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    features_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    missing_signals_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    supporting_signal_refs_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    includes_too_late: Mapped[bool] = mapped_column(Boolean, nullable=False)
    supersedes_window_id: Mapped[str | None] = mapped_column(
        ForeignKey("telemetry_windows.window_id"), nullable=True
    )
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class AlertCandidateRecord(TenantScopedMixin, Base):
    """A prediction candidate that cannot directly stop equipment or create a work order."""

    __tablename__ = "alert_candidates"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "window_id",
            "threshold_policy_id",
            "model_release_id",
            name="uq_alert_candidate_detector",
        ),
        Index(
            "ix_alert_candidates_status",
            "tenant_id",
            "status",
            "detected_at",
        ),
    )

    alert_candidate_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    window_id: Mapped[str] = mapped_column(
        ForeignKey("telemetry_windows.window_id"), nullable=False
    )
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_release_id: Mapped[str] = mapped_column(String(128), nullable=False)
    detector_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_policy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    supporting_signal_refs_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    explanation_codes_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class AlertCandidateDecisionRecord(TenantScopedMixin, Base):
    """Append-only human decision explaining confirmation or dismissal."""

    __tablename__ = "alert_candidate_decisions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "alert_candidate_id",
            name="uq_alert_candidate_decision",
        ),
    )

    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    alert_candidate_id: Mapped[str] = mapped_column(
        ForeignKey("alert_candidates.alert_candidate_id"), nullable=False
    )
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(128), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    candidate_version: Mapped[int] = mapped_column(Integer, nullable=False)


class PredictiveMaintenanceReferralRecord(TenantScopedMixin, Base):
    """Controlled bridge from a confirmed alert into the existing Incident flow."""

    __tablename__ = "predictive_maintenance_referrals"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "alert_candidate_id",
            name="uq_predictive_referral_alert",
        ),
        UniqueConstraint(
            "tenant_id",
            "incident_draft_id",
            name="uq_predictive_referral_draft",
        ),
    )

    referral_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    alert_candidate_id: Mapped[str] = mapped_column(
        ForeignKey("alert_candidates.alert_candidate_id"), nullable=False
    )
    incident_draft_id: Mapped[str] = mapped_column(
        ForeignKey("incident_drafts.draft_id"), nullable=False
    )
    linked_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    linked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PredictiveOutcomeRecord(TenantScopedMixin, Base):
    """Observed maintenance outcome used for false-positive, miss, and lead-time reporting."""

    __tablename__ = "predictive_outcomes"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_predictive_outcome_idempotency",
        ),
        Index(
            "ix_predictive_outcomes_asset_time",
            "tenant_id",
            "asset_id",
            "observed_at",
        ),
        UniqueConstraint(
            "tenant_id",
            "alert_candidate_id",
            name="uq_predictive_outcome_alert",
        ),
    )

    outcome_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    alert_candidate_id: Mapped[str | None] = mapped_column(
        ForeignKey("alert_candidates.alert_candidate_id"), nullable=True
    )
    outcome_type: Mapped[str] = mapped_column(String(32), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    failure_observed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    work_order_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_orders.work_order_id"), nullable=True
    )
    avoided_downtime_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    evidence_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    evidence_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    recorded_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class RulForecastCandidateRecord(TenantScopedMixin, Base):
    """Human-reviewed remaining-useful-life estimate derived from governed outcomes."""

    __tablename__ = "rul_forecast_candidates"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "alert_candidate_id",
            name="uq_rul_forecast_alert_candidate",
        ),
        UniqueConstraint(
            "tenant_id",
            "created_by_subject_id",
            "idempotency_key",
            name="uq_rul_forecast_idempotency",
        ),
        Index(
            "ix_rul_forecast_status",
            "tenant_id",
            "status",
            "created_at",
        ),
    )

    rul_forecast_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    alert_candidate_id: Mapped[str] = mapped_column(
        ForeignKey("alert_candidates.alert_candidate_id"), nullable=False
    )
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    window_id: Mapped[str] = mapped_column(
        ForeignKey("telemetry_windows.window_id"), nullable=False
    )
    feature_snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_model_release_id: Mapped[str] = mapped_column(String(128), nullable=False)
    forecast_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    estimate_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    lower_bound_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    upper_bound_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    recommended_inspection_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    historical_failure_count: Mapped[int] = mapped_column(Integer, nullable=False)
    source_outcome_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    cohort_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    history_cutoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_alert_version: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_by_subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewed_by_subject_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class RulForecastCalibrationRecord(TenantScopedMixin, Base):
    """Append-only outcome evidence for one governed RUL forecast."""

    __tablename__ = "rul_forecast_calibrations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "rul_forecast_id",
            name="uq_rul_calibration_forecast",
        ),
        UniqueConstraint(
            "tenant_id",
            "outcome_id",
            name="uq_rul_calibration_outcome",
        ),
        Index(
            "ix_rul_calibration_release_time",
            "tenant_id",
            "source_model_release_id",
            "forecast_model_version",
            "evaluated_at",
        ),
        CheckConstraint(
            "actual_minutes >= 0 AND absolute_error_minutes >= 0",
            name="ck_rul_calibration_nonnegative",
        ),
        CheckConstraint(
            "forecast_review_status IN ('PENDING_REVIEW', 'ACCEPTED', 'REJECTED')",
            name="ck_rul_calibration_forecast_review_status",
        ),
        CheckConstraint(
            "calibration_status IN ('INSUFFICIENT_EVIDENCE', 'STABLE', 'RETRAINING_RECOMMENDED')",
            name="ck_rul_calibration_status",
        ),
    )

    calibration_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    rul_forecast_id: Mapped[str] = mapped_column(
        ForeignKey("rul_forecast_candidates.rul_forecast_id"), nullable=False
    )
    outcome_id: Mapped[str] = mapped_column(
        ForeignKey("predictive_outcomes.outcome_id"), nullable=False
    )
    alert_candidate_id: Mapped[str] = mapped_column(
        ForeignKey("alert_candidates.alert_candidate_id"), nullable=False
    )
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.asset_id"), nullable=False)
    source_model_release_id: Mapped[str] = mapped_column(String(128), nullable=False)
    forecast_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    forecast_review_status: Mapped[str] = mapped_column(String(32), nullable=False)
    actual_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    absolute_error_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    interval_covered: Mapped[bool] = mapped_column(Boolean, nullable=False)
    calibration_status: Mapped[str] = mapped_column(String(32), nullable=False)
    retraining_recommended: Mapped[bool] = mapped_column(Boolean, nullable=False)
    decision_reasons_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    evidence_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SecurityAuditEventRecord(TimestampMixin, Base):
    __tablename__ = "security_audit_events"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenants.id"), nullable=True, index=True
    )
    subject_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    tenant_hash: Mapped[str] = mapped_column(String(128), nullable=False, default="legacy")
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


TENANT_TABLE_NAMES = frozenset(
    table_name for table_name in Base.metadata.tables if table_name != TenantRecord.__tablename__
)
