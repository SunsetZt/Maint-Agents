"""Idempotent synthetic tenant and asset seed; no customer data is accepted."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import Role
from industrial_ops_agent.config import get_settings
from industrial_ops_agent.knowledge.seed import install_synthetic_knowledge
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetComponentRecord,
    AssetCustomerLinkRecord,
    AssetRecord,
    AssetScopeRecord,
    AssetSiteLinkRecord,
    AssetWarrantyRecord,
    SubjectRecord,
    SubjectRoleRecord,
    TenantRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.secrets import SecretName, build_secret_provider

TENANT_ID = "tenant-m1-demo"
SUBJECT_ID = "subject-m1-engineer"
TENANT_ADMIN_SUBJECT_ID = "subject-m1-tenant-admin"
DATA_STEWARD_SUBJECT_ID = "subject-m3-data-steward"
LOCAL_DEMO_ADMIN_SUBJECT_ID = "subject-local-demo-admin"
LOCAL_DEMO_ADMIN_ROLES = tuple(role.value for role in Role)
AFTER_SALES_SUBJECT_ID = "subject-m7-after-sales"
DOMAIN_EXPERT_SUBJECT_ID = "subject-m7-domain-expert"
SECURITY_AUDITOR_SUBJECT_ID = "subject-m6-security-auditor"
PLATFORM_OPERATOR_SUBJECT_ID = "subject-m6-platform-operator"
ASSET_ID = "asset-m1-pump"


def _synthetic_oidc_subject(subject_id: str) -> str:
    return f"managed-by-keycloak-import:{subject_id}"


def seed(database: Database) -> None:
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        tenant = session.get(TenantRecord, TENANT_ID)
        if tenant is None:
            session.add(
                TenantRecord(
                    id=TENANT_ID,
                    status="active",
                    display_name="合成示范制造企业",
                    version=1,
                )
            )
        else:
            tenant.display_name = tenant.display_name or "合成示范制造企业"
        session.commit()

    context = TenantContext(tenant_id=TENANT_ID, subject_id="m1-seed")
    with database.transaction(context) as session:
        if session.get(SubjectRecord, SUBJECT_ID) is None:
            session.add(
                SubjectRecord(
                    subject_id=SUBJECT_ID,
                    tenant_id=TENANT_ID,
                    oidc_issuer="http://localhost:8080/realms/industrial-ops",
                    oidc_subject=_synthetic_oidc_subject(SUBJECT_ID),
                    status="active",
                )
            )
        if session.get(SubjectRecord, TENANT_ADMIN_SUBJECT_ID) is None:
            session.add(
                SubjectRecord(
                    subject_id=TENANT_ADMIN_SUBJECT_ID,
                    tenant_id=TENANT_ID,
                    oidc_issuer="http://localhost:8080/realms/industrial-ops",
                    oidc_subject=_synthetic_oidc_subject(TENANT_ADMIN_SUBJECT_ID),
                    display_name="示范租户管理员",
                    email="tenant-admin@example.invalid",
                    status="active",
                    version=1,
                )
            )
        if session.get(SubjectRecord, DATA_STEWARD_SUBJECT_ID) is None:
            session.add(
                SubjectRecord(
                    subject_id=DATA_STEWARD_SUBJECT_ID,
                    tenant_id=TENANT_ID,
                    oidc_issuer="http://localhost:8080/realms/industrial-ops",
                    oidc_subject=_synthetic_oidc_subject(DATA_STEWARD_SUBJECT_ID),
                    display_name="示范数据管理员",
                    email="data-steward@example.invalid",
                    status="active",
                    version=1,
                )
            )
        if session.get(SubjectRecord, LOCAL_DEMO_ADMIN_SUBJECT_ID) is None:
            session.add(
                SubjectRecord(
                    subject_id=LOCAL_DEMO_ADMIN_SUBJECT_ID,
                    tenant_id=TENANT_ID,
                    oidc_issuer="http://localhost:8080/realms/industrial-ops",
                    oidc_subject=_synthetic_oidc_subject(LOCAL_DEMO_ADMIN_SUBJECT_ID),
                    display_name="本地全功能演示管理员",
                    email="local-demo-admin@industrial-ops.local",
                    status="active",
                    version=1,
                )
            )
        if session.get(SubjectRecord, AFTER_SALES_SUBJECT_ID) is None:
            session.add(
                SubjectRecord(
                    subject_id=AFTER_SALES_SUBJECT_ID,
                    tenant_id=TENANT_ID,
                    oidc_issuer="http://localhost:8080/realms/industrial-ops",
                    oidc_subject=_synthetic_oidc_subject(AFTER_SALES_SUBJECT_ID),
                    display_name="示范售后工程师",
                    email="m7-after-sales@industrial-ops.local",
                    status="active",
                    version=1,
                )
            )
        if session.get(SubjectRecord, DOMAIN_EXPERT_SUBJECT_ID) is None:
            session.add(
                SubjectRecord(
                    subject_id=DOMAIN_EXPERT_SUBJECT_ID,
                    tenant_id=TENANT_ID,
                    oidc_issuer="http://localhost:8080/realms/industrial-ops",
                    oidc_subject=_synthetic_oidc_subject(DOMAIN_EXPERT_SUBJECT_ID),
                    display_name="示范领域专家",
                    email="m7-domain-expert@industrial-ops.local",
                    status="active",
                    version=1,
                )
            )
        if session.get(SubjectRecord, SECURITY_AUDITOR_SUBJECT_ID) is None:
            session.add(
                SubjectRecord(
                    subject_id=SECURITY_AUDITOR_SUBJECT_ID,
                    tenant_id=TENANT_ID,
                    oidc_issuer="http://localhost:8080/realms/industrial-ops",
                    oidc_subject=_synthetic_oidc_subject(SECURITY_AUDITOR_SUBJECT_ID),
                    display_name="示范安全审计员",
                    email="security-auditor@example.invalid",
                    status="active",
                    version=1,
                )
            )
        if session.get(SubjectRecord, PLATFORM_OPERATOR_SUBJECT_ID) is None:
            session.add(
                SubjectRecord(
                    subject_id=PLATFORM_OPERATOR_SUBJECT_ID,
                    tenant_id=TENANT_ID,
                    oidc_issuer="http://localhost:8080/realms/industrial-ops",
                    oidc_subject=_synthetic_oidc_subject(PLATFORM_OPERATOR_SUBJECT_ID),
                    display_name="示范平台运维",
                    email="platform-operator@example.invalid",
                    status="active",
                    version=1,
                )
            )
        asset = session.get(AssetRecord, ASSET_ID)
        if asset is None:
            asset = AssetRecord(
                asset_id=ASSET_ID,
                tenant_id=TENANT_ID,
                source_system="synthetic-eam",
                source_record_id="m1-demo-pump-001",
                as_of=now,
                model_code="PUMP-X100",
                display_name="一号循环泵",
                serial_number="SYN-PX100-0001",
                lifecycle_status="IN_SERVICE",
                version=1,
            )
            session.add(asset)
        else:
            if not (asset.model_code or "").strip():
                asset.model_code = "PUMP-X100"
                asset.as_of = now
                asset.version += 1
            asset.display_name = asset.display_name or "一号循环泵"
            asset.serial_number = asset.serial_number or "SYN-PX100-0001"
            asset.lifecycle_status = asset.lifecycle_status or "IN_SERVICE"
        if session.get(AssetCustomerLinkRecord, "asset-customer-m1-pump") is None:
            session.add(
                AssetCustomerLinkRecord(
                    customer_link_id="asset-customer-m1-pump",
                    tenant_id=TENANT_ID,
                    asset_id=ASSET_ID,
                    customer_id="customer-m1-demo",
                    customer_name="合成示范制造企业",
                    source_system="synthetic-crm",
                    source_record_id="crm-customer-m1-demo",
                    as_of=now,
                    version=1,
                )
            )
        if session.get(AssetSiteLinkRecord, "asset-site-m1-pump") is None:
            session.add(
                AssetSiteLinkRecord(
                    site_link_id="asset-site-m1-pump",
                    tenant_id=TENANT_ID,
                    asset_id=ASSET_ID,
                    site_id="site-m1-demo",
                    site_name="合成示范工厂一号站点",
                    address="合成数据地址 A 区泵房",
                    source_system="synthetic-eam",
                    source_record_id="eam-site-m1-demo",
                    as_of=now,
                    version=1,
                )
            )
        warranty = session.get(AssetWarrantyRecord, "asset-warranty-m1-pump")
        if warranty is None:
            session.add(
                AssetWarrantyRecord(
                    warranty_id="asset-warranty-m1-pump",
                    tenant_id=TENANT_ID,
                    asset_id=ASSET_ID,
                    contract_number="SYN-WARRANTY-2026-001",
                    status="ACTIVE",
                    coverage_start=now - timedelta(days=180),
                    coverage_end=now + timedelta(days=185),
                    service_level="24X7_FOUR_HOUR_RESPONSE",
                    source_system="synthetic-contract",
                    source_record_id="contract-m1-pump",
                    as_of=now,
                    version=1,
                )
            )
        elif warranty.source_system.casefold().startswith("synthetic"):
            observed = warranty.as_of
            if observed is not None and observed.tzinfo is None:
                observed = observed.replace(tzinfo=UTC)
            if observed is None or observed < now - timedelta(hours=24):
                warranty.as_of = now
                warranty.version += 1
        for component in (
            AssetComponentRecord(
                component_id="component-m1-pump-motor",
                tenant_id=TENANT_ID,
                asset_id=ASSET_ID,
                part_number="MOTOR-45KW-A",
                part_name="45kW 驱动电机",
                serial_number="SYN-MOTOR-0001",
                quantity=1,
                status="INSTALLED",
                installed_at=now - timedelta(days=180),
                source_system="synthetic-eam",
                source_record_id="bom-m1-motor",
                as_of=now,
                version=1,
            ),
            AssetComponentRecord(
                component_id="component-m1-pump-bearing",
                tenant_id=TENANT_ID,
                asset_id=ASSET_ID,
                part_number="BRG-6312-C3",
                part_name="驱动端轴承",
                serial_number=None,
                quantity=1,
                status="INSTALLED",
                installed_at=now - timedelta(days=30),
                source_system="synthetic-eam",
                source_record_id="bom-m1-bearing",
                as_of=now,
                version=2,
            ),
        ):
            if session.get(AssetComponentRecord, component.component_id) is None:
                session.add(component)
        role_key = {"tenant_id": TENANT_ID, "subject_id": SUBJECT_ID, "role": "field_engineer"}
        if session.get(SubjectRoleRecord, role_key) is None:
            session.add(SubjectRoleRecord(**role_key))
        admin_role_key = {
            "tenant_id": TENANT_ID,
            "subject_id": TENANT_ADMIN_SUBJECT_ID,
            "role": "tenant_admin",
        }
        if session.get(SubjectRoleRecord, admin_role_key) is None:
            session.add(SubjectRoleRecord(**admin_role_key))
        for subject_id, role in (
            (DATA_STEWARD_SUBJECT_ID, "data_steward"),
            (AFTER_SALES_SUBJECT_ID, "after_sales_engineer"),
            (DOMAIN_EXPERT_SUBJECT_ID, "domain_expert"),
            (SECURITY_AUDITOR_SUBJECT_ID, "security_auditor"),
            (PLATFORM_OPERATOR_SUBJECT_ID, "platform_operator"),
        ):
            role_key = {"tenant_id": TENANT_ID, "subject_id": subject_id, "role": role}
            if session.get(SubjectRoleRecord, role_key) is None:
                session.add(SubjectRoleRecord(**role_key))
        for role in LOCAL_DEMO_ADMIN_ROLES:
            role_key = {
                "tenant_id": TENANT_ID,
                "subject_id": LOCAL_DEMO_ADMIN_SUBJECT_ID,
                "role": role,
            }
            if session.get(SubjectRoleRecord, role_key) is None:
                session.add(SubjectRoleRecord(**role_key))
        if session.get(AssetScopeRecord, "scope-m1-engineer-pump") is None:
            session.add(
                AssetScopeRecord(
                    scope_id="scope-m1-engineer-pump",
                    tenant_id=TENANT_ID,
                    subject_id=SUBJECT_ID,
                    role="field_engineer",
                    asset_id=ASSET_ID,
                    site_id="site-m1-demo",
                )
            )
        if session.get(AssetScopeRecord, "scope-local-demo-admin-pump") is None:
            session.add(
                AssetScopeRecord(
                    scope_id="scope-local-demo-admin-pump",
                    tenant_id=TENANT_ID,
                    subject_id=LOCAL_DEMO_ADMIN_SUBJECT_ID,
                    role=None,
                    asset_id=ASSET_ID,
                    site_id="site-m1-demo",
                )
            )
        for scope_id, subject_id, role in (
            (
                "scope-m7-after-sales-pump",
                AFTER_SALES_SUBJECT_ID,
                "after_sales_engineer",
            ),
            (
                "scope-m7-domain-expert-pump",
                DOMAIN_EXPERT_SUBJECT_ID,
                "domain_expert",
            ),
        ):
            if session.get(AssetScopeRecord, scope_id) is None:
                session.add(
                    AssetScopeRecord(
                        scope_id=scope_id,
                        tenant_id=TENANT_ID,
                        subject_id=subject_id,
                        role=role,
                        asset_id=ASSET_ID,
                        site_id="site-m1-demo",
                    )
                )
        install_synthetic_knowledge(session, tenant_id=TENANT_ID, now=now)


def main() -> None:
    settings = get_settings()
    database_url = build_secret_provider(settings).get(SecretName.DATABASE_URL)
    database = Database(database_url.reveal())
    try:
        seed(database)
    finally:
        database.dispose()
    print("M1 synthetic tenant and asset seed is current.")


if __name__ == "__main__":
    main()
