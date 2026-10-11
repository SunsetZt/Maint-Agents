"""Tenant-local identity directory and administration use cases."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetRecord,
    AssetScopeRecord,
    AssetSiteLinkRecord,
    SubjectRecord,
    SubjectRoleRecord,
    TenantAdministrationRecord,
    TenantRecord,
    TenantRetentionPolicyRecord,
)
from industrial_ops_agent.persistence.tenant import validate_boundary_identifier

HUMAN_ASSIGNABLE_ROLES = tuple(
    role for role in Role if role is not Role.MODEL_DEPLOYMENT_CONTROLLER
)


class ManagedIdentityDenied(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TenantAdministrationNotVisible(Exception):
    pass


class TenantAdministrationConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.current_version = current_version


@dataclass(frozen=True, slots=True)
class TenantOverview:
    tenant_id: str
    display_name: str | None
    status: str
    version: int
    member_count: int
    active_member_count: int
    asset_count: int
    site_count: int


@dataclass(frozen=True, slots=True)
class TenantAssetOption:
    asset_id: str
    display_name: str | None
    model_code: str | None


@dataclass(frozen=True, slots=True)
class TenantSiteOption:
    site_id: str
    site_name: str


@dataclass(frozen=True, slots=True)
class TenantMemberView:
    subject_id: str
    oidc_issuer: str
    oidc_subject: str
    display_name: str | None
    email: str | None
    status: str
    roles: tuple[str, ...]
    asset_ids: tuple[str, ...]
    site_ids: tuple[str, ...]
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class TenantRetentionPolicyView:
    policy_id: str | None
    configured: bool
    incident_days: int | None
    media_days: int | None
    knowledge_days: int | None
    training_data_days: int | None
    audit_days: int | None
    legal_hold: bool
    updated_by_subject_id: str | None
    version: int
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class TenantAdministrationView:
    administration_id: str
    command_type: str
    target_type: str
    target_id: str
    actor_subject_id: str
    reason: str
    previous_version: int
    target_version: int
    before: dict[str, Any]
    after: dict[str, Any]
    occurred_at: datetime


class DirectoryPolicyService:
    """Overlay managed directory assignments on a cryptographically verified token."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def resolve(self, token_identity: IdentityContext) -> IdentityContext:
        with self._database.transaction(token_identity.tenant_context) as session:
            tenant = session.get(TenantRecord, token_identity.tenant_id)
            if tenant is not None and tenant.status.upper() != "ACTIVE":
                raise ManagedIdentityDenied("tenant_inactive")
            member = session.scalar(
                select(SubjectRecord).where(
                    SubjectRecord.tenant_id == token_identity.tenant_id,
                    SubjectRecord.subject_id == token_identity.subject_id,
                )
            )
            if member is None:
                managed_member_count = int(
                    session.scalar(
                        select(func.count())
                        .select_from(SubjectRecord)
                        .where(SubjectRecord.tenant_id == token_identity.tenant_id)
                    )
                    or 0
                )
                if managed_member_count:
                    raise ManagedIdentityDenied("subject_not_registered")
                return token_identity
            if member.status.upper() != "ACTIVE":
                raise ManagedIdentityDenied("subject_inactive")
            role_values = tuple(
                session.scalars(
                    select(SubjectRoleRecord.role).where(
                        SubjectRoleRecord.tenant_id == token_identity.tenant_id,
                        SubjectRoleRecord.subject_id == token_identity.subject_id,
                    )
                ).all()
            )
            roles = frozenset(
                Role(value) for value in role_values if value in Role._value2member_map_
            )
            if not roles:
                raise ManagedIdentityDenied("subject_has_no_active_role")
            scopes = session.scalars(
                select(AssetScopeRecord).where(
                    AssetScopeRecord.tenant_id == token_identity.tenant_id,
                    AssetScopeRecord.subject_id == token_identity.subject_id,
                )
            ).all()
            applicable = [
                scope
                for scope in scopes
                if scope.role is None or scope.role in {role.value for role in roles}
            ]
            return replace(
                token_identity,
                roles=roles,
                asset_ids=frozenset(
                    scope.asset_id for scope in applicable if scope.asset_id is not None
                ),
                site_ids=frozenset(
                    scope.site_id for scope in applicable if scope.site_id is not None
                ),
            )


class TenantAdministrationService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def overview(
        self, identity: IdentityContext, *, request_id: str
    ) -> tuple[TenantOverview, TenantRetentionPolicyView]:
        self._require(identity, Action.READ_TENANT_ADMINISTRATION, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            tenant = session.get(TenantRecord, identity.tenant_id)
            if tenant is None:
                raise TenantAdministrationNotVisible
            member_count = int(
                session.scalar(
                    select(func.count())
                    .select_from(SubjectRecord)
                    .where(SubjectRecord.tenant_id == identity.tenant_id)
                )
                or 0
            )
            active_member_count = int(
                session.scalar(
                    select(func.count())
                    .select_from(SubjectRecord)
                    .where(
                        SubjectRecord.tenant_id == identity.tenant_id,
                        func.upper(SubjectRecord.status) == "ACTIVE",
                    )
                )
                or 0
            )
            asset_count = int(
                session.scalar(
                    select(func.count())
                    .select_from(AssetRecord)
                    .where(AssetRecord.tenant_id == identity.tenant_id)
                )
                or 0
            )
            site_count = int(
                session.scalar(
                    select(func.count(func.distinct(AssetSiteLinkRecord.site_id))).where(
                        AssetSiteLinkRecord.tenant_id == identity.tenant_id
                    )
                )
                or 0
            )
            policy = session.scalar(
                select(TenantRetentionPolicyRecord).where(
                    TenantRetentionPolicyRecord.tenant_id == identity.tenant_id
                )
            )
            return (
                TenantOverview(
                    tenant_id=tenant.id,
                    display_name=tenant.display_name,
                    status=tenant.status.upper(),
                    version=tenant.version,
                    member_count=member_count,
                    active_member_count=active_member_count,
                    asset_count=asset_count,
                    site_count=site_count,
                ),
                _retention_view(policy),
            )

    def scope_catalog(
        self, identity: IdentityContext, *, request_id: str
    ) -> tuple[list[TenantAssetOption], list[TenantSiteOption]]:
        self._require(identity, Action.READ_TENANT_ADMINISTRATION, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            assets = session.scalars(
                select(AssetRecord)
                .where(AssetRecord.tenant_id == identity.tenant_id)
                .order_by(AssetRecord.display_name, AssetRecord.asset_id)
            ).all()
            sites = (
                session.execute(
                    select(AssetSiteLinkRecord.site_id, AssetSiteLinkRecord.site_name)
                    .where(AssetSiteLinkRecord.tenant_id == identity.tenant_id)
                    .distinct()
                    .order_by(AssetSiteLinkRecord.site_name, AssetSiteLinkRecord.site_id)
                )
                .tuples()
                .all()
            )
            return (
                [
                    TenantAssetOption(
                        asset_id=asset.asset_id,
                        display_name=asset.display_name,
                        model_code=asset.model_code,
                    )
                    for asset in assets
                ],
                [
                    TenantSiteOption(site_id=site_id, site_name=site_name)
                    for site_id, site_name in sites
                ],
            )

    def list_members(
        self,
        identity: IdentityContext,
        *,
        status: str | None,
        role: str | None,
        limit: int,
        offset: int,
        request_id: str,
    ) -> tuple[list[TenantMemberView], int]:
        self._require(identity, Action.READ_TENANT_ADMINISTRATION, request_id)
        statement = select(SubjectRecord).where(SubjectRecord.tenant_id == identity.tenant_id)
        if status is not None:
            statement = statement.where(func.upper(SubjectRecord.status) == status.upper())
        if role is not None:
            role_subjects = select(SubjectRoleRecord.subject_id).where(
                SubjectRoleRecord.tenant_id == identity.tenant_id,
                SubjectRoleRecord.role == role,
            )
            statement = statement.where(SubjectRecord.subject_id.in_(role_subjects))
        with self._database.transaction(identity.tenant_context) as session:
            total = int(session.scalar(select(func.count()).select_from(statement.subquery())) or 0)
            members = session.scalars(
                statement.order_by(SubjectRecord.created_at, SubjectRecord.subject_id)
                .limit(limit)
                .offset(offset)
            ).all()
            return [self._member_view(session, member) for member in members], total

    def create_member(
        self,
        identity: IdentityContext,
        *,
        subject_id: str,
        oidc_issuer: str,
        oidc_subject: str,
        display_name: str | None,
        email: str | None,
        roles: tuple[str, ...],
        asset_ids: tuple[str, ...],
        site_ids: tuple[str, ...],
        reason: str,
        client_operation_id: str,
        request_id: str,
    ) -> tuple[TenantMemberView, bool]:
        self._require(identity, Action.MANAGE_TENANT_ADMINISTRATION, request_id)
        validate_boundary_identifier(subject_id, field="subject_id")
        role_values = self._validated_roles(roles)
        now = datetime.now(UTC)
        desired = _access_snapshot(
            status="ACTIVE",
            roles=role_values,
            asset_ids=asset_ids,
            site_ids=site_ids,
            display_name=display_name,
            email=email,
        )
        desired["oidc_issuer"] = oidc_issuer
        desired["oidc_subject"] = oidc_subject
        if not oidc_issuer.startswith("https://") and not oidc_issuer.startswith(
            "http://localhost"
        ):
            raise TenantAdministrationConflict("OIDC issuer must use HTTPS outside localhost")
        with self._database.transaction(identity.tenant_context) as session:
            existing_operation = self._existing_operation(session, identity, client_operation_id)
            if existing_operation is not None:
                if (
                    existing_operation.command_type != "CREATE_MEMBER"
                    or existing_operation.target_id != subject_id
                    or existing_operation.after_json != desired
                ):
                    raise TenantAdministrationConflict(
                        "client operation id was used for another tenant change"
                    )
                member = self._get_member(session, identity.tenant_id, subject_id)
                return self._member_view(session, member), False
            existing = session.scalar(
                select(SubjectRecord).where(
                    SubjectRecord.tenant_id == identity.tenant_id,
                    SubjectRecord.subject_id == subject_id,
                )
            )
            if existing is not None:
                raise TenantAdministrationConflict("tenant member already exists")
            existing_oidc_subject = session.scalar(
                select(SubjectRecord.subject_id).where(
                    SubjectRecord.tenant_id == identity.tenant_id,
                    SubjectRecord.oidc_issuer == oidc_issuer,
                    SubjectRecord.oidc_subject == oidc_subject,
                )
            )
            if existing_oidc_subject is not None:
                raise TenantAdministrationConflict(
                    "OIDC identity is already linked to another tenant member"
                )
            self._validate_scopes(session, identity.tenant_id, asset_ids, site_ids)
            member = SubjectRecord(
                subject_id=subject_id,
                tenant_id=identity.tenant_id,
                oidc_issuer=oidc_issuer,
                oidc_subject=oidc_subject,
                display_name=display_name,
                email=email,
                status="ACTIVE",
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(member)
            self._replace_assignments(
                session,
                identity.tenant_id,
                subject_id,
                role_values,
                asset_ids,
                site_ids,
            )
            self._append_administration(
                session,
                identity,
                client_operation_id=client_operation_id,
                command_type="CREATE_MEMBER",
                target_type="SUBJECT",
                target_id=subject_id,
                reason=reason,
                previous_version=0,
                target_version=1,
                before={},
                after=desired,
                occurred_at=now,
            )
            session.flush()
            return self._member_view(session, member), True

    def update_member(
        self,
        identity: IdentityContext,
        subject_id: str,
        *,
        display_name: str | None,
        email: str | None,
        status: str,
        roles: tuple[str, ...],
        asset_ids: tuple[str, ...],
        site_ids: tuple[str, ...],
        reason: str,
        expected_version: int,
        client_operation_id: str,
        request_id: str,
    ) -> TenantMemberView:
        self._require(identity, Action.MANAGE_TENANT_ADMINISTRATION, request_id)
        role_values = self._validated_roles(roles)
        normalized_status = status.upper()
        if normalized_status not in {"ACTIVE", "SUSPENDED"}:
            raise TenantAdministrationConflict("invalid member status")
        if subject_id == identity.subject_id and (
            normalized_status != "ACTIVE" or Role.TENANT_ADMIN.value not in role_values
        ):
            raise TenantAdministrationConflict(
                "an administrator cannot remove their own active tenant-admin access"
            )
        desired = _access_snapshot(
            status=normalized_status,
            roles=role_values,
            asset_ids=asset_ids,
            site_ids=site_ids,
            display_name=display_name,
            email=email,
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            existing_operation = self._existing_operation(session, identity, client_operation_id)
            if existing_operation is not None:
                if (
                    existing_operation.command_type != "UPDATE_MEMBER"
                    or existing_operation.target_id != subject_id
                    or existing_operation.after_json != desired
                ):
                    raise TenantAdministrationConflict(
                        "client operation id was used for another tenant change"
                    )
                return self._member_view(
                    session, self._get_member(session, identity.tenant_id, subject_id)
                )
            member = self._get_member(session, identity.tenant_id, subject_id)
            if member.version != expected_version:
                raise TenantAdministrationConflict("tenant member version conflict", member.version)
            before = self._member_access_snapshot(session, member)
            removes_admin = Role.TENANT_ADMIN.value in before["roles"] and (
                Role.TENANT_ADMIN.value not in role_values or normalized_status != "ACTIVE"
            )
            if removes_admin and self._active_admin_count(session, identity.tenant_id) <= 1:
                raise TenantAdministrationConflict(
                    "the tenant must retain at least one active administrator"
                )
            self._validate_scopes(session, identity.tenant_id, asset_ids, site_ids)
            changed = session.execute(
                update(SubjectRecord)
                .where(
                    SubjectRecord.tenant_id == identity.tenant_id,
                    SubjectRecord.subject_id == subject_id,
                    SubjectRecord.version == expected_version,
                )
                .values(
                    display_name=display_name,
                    email=email,
                    status=normalized_status,
                    version=expected_version + 1,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if changed.rowcount != 1:
                raise TenantAdministrationConflict("tenant member changed concurrently")
            self._replace_assignments(
                session,
                identity.tenant_id,
                subject_id,
                role_values,
                asset_ids,
                site_ids,
            )
            self._append_administration(
                session,
                identity,
                client_operation_id=client_operation_id,
                command_type="UPDATE_MEMBER",
                target_type="SUBJECT",
                target_id=subject_id,
                reason=reason,
                previous_version=expected_version,
                target_version=expected_version + 1,
                before=before,
                after=desired,
                occurred_at=now,
            )
            session.flush()
            session.expire_all()
            return self._member_view(
                session, self._get_member(session, identity.tenant_id, subject_id)
            )

    def update_retention_policy(
        self,
        identity: IdentityContext,
        *,
        incident_days: int,
        media_days: int,
        knowledge_days: int,
        training_data_days: int,
        audit_days: int,
        legal_hold: bool,
        reason: str,
        expected_version: int,
        client_operation_id: str,
        request_id: str,
    ) -> TenantRetentionPolicyView:
        self._require(identity, Action.MANAGE_TENANT_ADMINISTRATION, request_id)
        values = {
            "incident_days": incident_days,
            "media_days": media_days,
            "knowledge_days": knowledge_days,
            "training_data_days": training_data_days,
            "audit_days": audit_days,
            "legal_hold": legal_hold,
        }
        if any(value < 1 or value > 36_500 for key, value in values.items() if key != "legal_hold"):
            raise TenantAdministrationConflict("retention days must be between 1 and 36500")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            existing_operation = self._existing_operation(session, identity, client_operation_id)
            if existing_operation is not None:
                if (
                    existing_operation.command_type != "UPDATE_RETENTION"
                    or existing_operation.after_json != values
                ):
                    raise TenantAdministrationConflict(
                        "client operation id was used for another tenant change"
                    )
                policy = self._get_retention_policy(session, identity.tenant_id)
                return _retention_view(policy)
            policy = session.scalar(
                select(TenantRetentionPolicyRecord).where(
                    TenantRetentionPolicyRecord.tenant_id == identity.tenant_id
                )
            )
            if policy is None:
                if expected_version != 0:
                    raise TenantAdministrationConflict("retention policy version conflict", 0)
                policy = TenantRetentionPolicyRecord(
                    policy_id=f"retention-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    updated_by_subject_id=identity.subject_id,
                    version=1,
                    created_at=now,
                    updated_at=now,
                    **values,
                )
                session.add(policy)
                before: dict[str, Any] = {}
                target_version = 1
            else:
                if policy.version != expected_version:
                    raise TenantAdministrationConflict(
                        "retention policy version conflict", policy.version
                    )
                before = _retention_snapshot(policy)
                target_version = expected_version + 1
                changed = session.execute(
                    update(TenantRetentionPolicyRecord)
                    .where(
                        TenantRetentionPolicyRecord.tenant_id == identity.tenant_id,
                        TenantRetentionPolicyRecord.version == expected_version,
                    )
                    .values(
                        **values,
                        updated_by_subject_id=identity.subject_id,
                        version=target_version,
                        updated_at=now,
                    )
                    .execution_options(synchronize_session=False)
                )
                if changed.rowcount != 1:
                    raise TenantAdministrationConflict("retention policy changed concurrently")
            self._append_administration(
                session,
                identity,
                client_operation_id=client_operation_id,
                command_type="UPDATE_RETENTION",
                target_type="RETENTION_POLICY",
                target_id=identity.tenant_id,
                reason=reason,
                previous_version=expected_version,
                target_version=target_version,
                before=before,
                after=values,
                occurred_at=now,
            )
            session.flush()
            session.expire_all()
            return _retention_view(self._get_retention_policy(session, identity.tenant_id))

    def list_administration_records(
        self,
        identity: IdentityContext,
        *,
        limit: int,
        offset: int,
        request_id: str,
    ) -> tuple[list[TenantAdministrationView], int]:
        self._require(identity, Action.READ_TENANT_ADMINISTRATION, request_id)
        statement = select(TenantAdministrationRecord).where(
            TenantAdministrationRecord.tenant_id == identity.tenant_id
        )
        with self._database.transaction(identity.tenant_context) as session:
            total = int(session.scalar(select(func.count()).select_from(statement.subquery())) or 0)
            records = session.scalars(
                statement.order_by(
                    TenantAdministrationRecord.occurred_at.desc(),
                    TenantAdministrationRecord.administration_id,
                )
                .limit(limit)
                .offset(offset)
            ).all()
            return [_administration_view(record) for record in records], total

    def _require(self, identity: IdentityContext, action: Action, request_id: str) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(identity.tenant_id, resource_id=identity.tenant_id),
            request_id=request_id,
        )

    @staticmethod
    def _get_member(session: Session, tenant_id: str, subject_id: str) -> SubjectRecord:
        member = session.scalar(
            select(SubjectRecord).where(
                SubjectRecord.tenant_id == tenant_id,
                SubjectRecord.subject_id == subject_id,
            )
        )
        if member is None:
            raise TenantAdministrationNotVisible
        return member

    @staticmethod
    def _get_retention_policy(session: Session, tenant_id: str) -> TenantRetentionPolicyRecord:
        policy = session.scalar(
            select(TenantRetentionPolicyRecord).where(
                TenantRetentionPolicyRecord.tenant_id == tenant_id
            )
        )
        if policy is None:
            raise TenantAdministrationNotVisible
        return policy

    @staticmethod
    def _existing_operation(
        session: Session, identity: IdentityContext, client_operation_id: str
    ) -> TenantAdministrationRecord | None:
        return session.scalar(
            select(TenantAdministrationRecord).where(
                TenantAdministrationRecord.tenant_id == identity.tenant_id,
                TenantAdministrationRecord.actor_subject_id == identity.subject_id,
                TenantAdministrationRecord.client_operation_id == client_operation_id,
            )
        )

    @staticmethod
    def _validated_roles(values: tuple[str, ...]) -> tuple[str, ...]:
        allowed = {role.value for role in HUMAN_ASSIGNABLE_ROLES}
        normalized = tuple(sorted(set(values)))
        if not normalized or any(value not in allowed for value in normalized):
            raise TenantAdministrationConflict("member roles are invalid")
        return normalized

    @staticmethod
    def _validate_scopes(
        session: Session,
        tenant_id: str,
        asset_ids: tuple[str, ...],
        site_ids: tuple[str, ...],
    ) -> None:
        existing_assets = set(
            session.scalars(
                select(AssetRecord.asset_id).where(
                    AssetRecord.tenant_id == tenant_id,
                    AssetRecord.asset_id.in_(asset_ids or ("__none__",)),
                )
            ).all()
        )
        existing_sites = set(
            session.scalars(
                select(AssetSiteLinkRecord.site_id).where(
                    AssetSiteLinkRecord.tenant_id == tenant_id,
                    AssetSiteLinkRecord.site_id.in_(site_ids or ("__none__",)),
                )
            ).all()
        )
        if existing_assets != set(asset_ids) or existing_sites != set(site_ids):
            raise TenantAdministrationConflict(
                "member scope contains an unknown tenant asset or site"
            )

    @staticmethod
    def _replace_assignments(
        session: Session,
        tenant_id: str,
        subject_id: str,
        roles: tuple[str, ...],
        asset_ids: tuple[str, ...],
        site_ids: tuple[str, ...],
    ) -> None:
        session.execute(
            delete(SubjectRoleRecord).where(
                SubjectRoleRecord.tenant_id == tenant_id,
                SubjectRoleRecord.subject_id == subject_id,
            )
        )
        session.execute(
            delete(AssetScopeRecord).where(
                AssetScopeRecord.tenant_id == tenant_id,
                AssetScopeRecord.subject_id == subject_id,
            )
        )
        session.add_all(
            SubjectRoleRecord(tenant_id=tenant_id, subject_id=subject_id, role=role)
            for role in roles
        )
        session.add_all(
            [
                AssetScopeRecord(
                    scope_id=f"scope-{uuid4().hex}",
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                    role=None,
                    asset_id=asset_id,
                    site_id=None,
                )
                for asset_id in sorted(set(asset_ids))
            ]
            + [
                AssetScopeRecord(
                    scope_id=f"scope-{uuid4().hex}",
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                    role=None,
                    asset_id=None,
                    site_id=site_id,
                )
                for site_id in sorted(set(site_ids))
            ]
        )

    def _member_view(self, session: Session, member: SubjectRecord) -> TenantMemberView:
        roles = tuple(
            sorted(
                session.scalars(
                    select(SubjectRoleRecord.role).where(
                        SubjectRoleRecord.tenant_id == member.tenant_id,
                        SubjectRoleRecord.subject_id == member.subject_id,
                    )
                ).all()
            )
        )
        scopes = session.scalars(
            select(AssetScopeRecord).where(
                AssetScopeRecord.tenant_id == member.tenant_id,
                AssetScopeRecord.subject_id == member.subject_id,
            )
        ).all()
        return TenantMemberView(
            subject_id=member.subject_id,
            oidc_issuer=member.oidc_issuer,
            oidc_subject=member.oidc_subject,
            display_name=member.display_name,
            email=member.email,
            status=member.status.upper(),
            roles=roles,
            asset_ids=tuple(sorted({item.asset_id for item in scopes if item.asset_id})),
            site_ids=tuple(sorted({item.site_id for item in scopes if item.site_id})),
            version=member.version,
            created_at=member.created_at,
            updated_at=member.updated_at,
        )

    def _member_access_snapshot(self, session: Session, member: SubjectRecord) -> dict[str, Any]:
        view = self._member_view(session, member)
        return _access_snapshot(
            status=view.status,
            roles=view.roles,
            asset_ids=view.asset_ids,
            site_ids=view.site_ids,
            display_name=view.display_name,
            email=view.email,
        )

    @staticmethod
    def _active_admin_count(session: Session, tenant_id: str) -> int:
        return int(
            session.scalar(
                select(func.count(func.distinct(SubjectRecord.subject_id)))
                .join(
                    SubjectRoleRecord,
                    SubjectRoleRecord.subject_id == SubjectRecord.subject_id,
                )
                .where(
                    SubjectRecord.tenant_id == tenant_id,
                    SubjectRoleRecord.tenant_id == tenant_id,
                    func.upper(SubjectRecord.status) == "ACTIVE",
                    SubjectRoleRecord.role == Role.TENANT_ADMIN.value,
                )
            )
            or 0
        )

    @staticmethod
    def _append_administration(
        session: Session,
        identity: IdentityContext,
        *,
        client_operation_id: str,
        command_type: str,
        target_type: str,
        target_id: str,
        reason: str,
        previous_version: int,
        target_version: int,
        before: dict[str, Any],
        after: dict[str, Any],
        occurred_at: datetime,
    ) -> None:
        session.add(
            TenantAdministrationRecord(
                administration_id=f"tenant-admin-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                client_operation_id=client_operation_id,
                command_type=command_type,
                target_type=target_type,
                target_id=target_id,
                actor_subject_id=identity.subject_id,
                reason=reason,
                previous_version=previous_version,
                target_version=target_version,
                before_json=before,
                after_json=after,
                occurred_at=occurred_at,
            )
        )


def _access_snapshot(
    *,
    status: str,
    roles: tuple[str, ...],
    asset_ids: tuple[str, ...],
    site_ids: tuple[str, ...],
    display_name: str | None,
    email: str | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "roles": list(sorted(set(roles))),
        "asset_ids": list(sorted(set(asset_ids))),
        "site_ids": list(sorted(set(site_ids))),
        "display_name": display_name,
        "email": email,
    }


def _retention_snapshot(record: TenantRetentionPolicyRecord) -> dict[str, Any]:
    return {
        "incident_days": record.incident_days,
        "media_days": record.media_days,
        "knowledge_days": record.knowledge_days,
        "training_data_days": record.training_data_days,
        "audit_days": record.audit_days,
        "legal_hold": record.legal_hold,
    }


def _retention_view(
    record: TenantRetentionPolicyRecord | None,
) -> TenantRetentionPolicyView:
    if record is None:
        return TenantRetentionPolicyView(
            policy_id=None,
            configured=False,
            incident_days=None,
            media_days=None,
            knowledge_days=None,
            training_data_days=None,
            audit_days=None,
            legal_hold=False,
            updated_by_subject_id=None,
            version=0,
            updated_at=None,
        )
    return TenantRetentionPolicyView(
        policy_id=record.policy_id,
        configured=True,
        incident_days=record.incident_days,
        media_days=record.media_days,
        knowledge_days=record.knowledge_days,
        training_data_days=record.training_data_days,
        audit_days=record.audit_days,
        legal_hold=record.legal_hold,
        updated_by_subject_id=record.updated_by_subject_id,
        version=record.version,
        updated_at=record.updated_at,
    )


def _administration_view(
    record: TenantAdministrationRecord,
) -> TenantAdministrationView:
    return TenantAdministrationView(
        administration_id=record.administration_id,
        command_type=record.command_type,
        target_type=record.target_type,
        target_id=record.target_id,
        actor_subject_id=record.actor_subject_id,
        reason=record.reason,
        previous_version=record.previous_version,
        target_version=record.target_version,
        before=record.before_json,
        after=record.after_json,
        occurred_at=record.occurred_at,
    )
