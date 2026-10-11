"""Tenant policy, source filtering, guardrails and audit for public references."""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.external_search.provider import (
    ExternalSearchProvider,
    ExternalSearchProviderError,
    ExternalSearchProviderItem,
)
from industrial_ops_agent.guardrails import PromptInjectionGuard
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ExternalSearchPolicyRecord,
    ExternalSearchQueryRecord,
    ExternalSearchReferenceRecord,
    IdempotencyRecord,
)

EXTERNAL_SEARCH_USE_CASES = frozenset(
    {"GENERAL_REFERENCE", "REPAIR_REFERENCE", "SAFETY_REFERENCE", "WARRANTY_REFERENCE"}
)
SENSITIVE_EXTERNAL_SEARCH_USE_CASES = frozenset(
    {"REPAIR_REFERENCE", "SAFETY_REFERENCE", "WARRANTY_REFERENCE"}
)
USAGE_CONCLUSIONS = frozenset(
    {"NOT_USED", "REFERENCE_ONLY", "ESCALATED_TO_KNOWLEDGE_REVIEW"}
)


class ExternalSearchError(RuntimeError):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ExternalSearchPolicyView:
    enabled: bool
    allowed_domains: tuple[str, ...]
    official_domains: tuple[str, ...]
    max_results: int
    updated_by_subject_id: str | None
    version: int
    created_at: datetime | None
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class ExternalReferenceView:
    external_reference_id: str
    rank: int
    title: str
    url: str
    domain: str
    summary: str
    content_hash: str
    relevance_score: float
    trust_level: str
    published_at: datetime | None
    guardrail_policy_version: str
    fetched_at: datetime


@dataclass(frozen=True, slots=True)
class ExternalSearchQueryView:
    external_search_id: str
    query_text: str
    query_hash: str
    use_case: str
    status: str
    provider: str
    provider_request_id: str | None
    requested_by_subject_id: str
    result_count: int
    blocked_result_count: int
    usage_credits: int | None
    failure_code: str | None
    usage_conclusion: str
    conclusion_reason: str | None
    concluded_by_subject_id: str | None
    started_at: datetime
    completed_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    references: tuple[ExternalReferenceView, ...]


class ExternalSearchService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        provider: ExternalSearchProvider | None,
        *,
        guardrail: PromptInjectionGuard | None = None,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._provider = provider
        self._guardrail = guardrail or PromptInjectionGuard()

    def get_policy(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> ExternalSearchPolicyView:
        self._require(
            identity,
            Action.READ_EXTERNAL_REFERENCE,
            "external-search-policy",
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(ExternalSearchPolicyRecord).where(
                    ExternalSearchPolicyRecord.tenant_id == identity.tenant_id
                )
            )
            return _policy_view(record)

    def update_policy(
        self,
        identity: IdentityContext,
        *,
        enabled: bool,
        allowed_domains: tuple[str, ...],
        official_domains: tuple[str, ...],
        max_results: int,
        expected_version: int,
        request_id: str,
    ) -> ExternalSearchPolicyView:
        self._require(
            identity,
            Action.MANAGE_EXTERNAL_SEARCH_POLICY,
            "external-search-policy",
            request_id,
        )
        allowed = _domains(allowed_domains)
        official = _domains(official_domains)
        if enabled and not allowed:
            raise ExternalSearchError("external_search_allowed_domains_required")
        if not set(official).issubset(allowed):
            raise ExternalSearchError("external_search_official_domains_not_allowed")
        if not 1 <= max_results <= 10:
            raise ExternalSearchError("external_search_max_results_invalid")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(ExternalSearchPolicyRecord)
                .where(ExternalSearchPolicyRecord.tenant_id == identity.tenant_id)
                .with_for_update()
            )
            if record is None:
                if expected_version != 0:
                    raise ExternalSearchError("external_search_policy_state_changed", 0)
                record = ExternalSearchPolicyRecord(
                    tenant_id=identity.tenant_id,
                    policy_id=f"external-search-policy-{identity.tenant_id}",
                    enabled=enabled,
                    allowed_domains=list(allowed),
                    official_domains=list(official),
                    max_results=max_results,
                    updated_by_subject_id=identity.subject_id,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
            else:
                if record.version != expected_version:
                    raise ExternalSearchError(
                        "external_search_policy_state_changed", record.version
                    )
                record.enabled = enabled
                record.allowed_domains = list(allowed)
                record.official_domains = list(official)
                record.max_results = max_results
                record.updated_by_subject_id = identity.subject_id
                record.version += 1
                record.updated_at = now
            session.flush()
            return _policy_view(record)

    def search(
        self,
        identity: IdentityContext,
        *,
        query_text: str,
        use_case: str,
        idempotency_key: str,
        request_id: str,
    ) -> ExternalSearchQueryView:
        self._require(identity, Action.RUN_EXTERNAL_SEARCH, "external-search", request_id)
        query = " ".join(query_text.split())
        if not 3 <= len(query) <= 500 or use_case not in EXTERNAL_SEARCH_USE_CASES:
            raise ExternalSearchError("external_search_input_invalid")
        if not 8 <= len(idempotency_key) <= 255:
            raise ExternalSearchError("external_search_idempotency_invalid")
        fingerprint = sha256(f"{query}\0{use_case}".encode()).hexdigest()
        storage_key = f"external-search:{idempotency_key}"
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            replay = session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.tenant_id == identity.tenant_id,
                    IdempotencyRecord.subject_id == identity.subject_id,
                    IdempotencyRecord.key == storage_key,
                )
            )
            if replay is not None:
                if replay.request_hash != fingerprint:
                    raise ExternalSearchError("external_search_idempotency_conflict")
                return _query_view(
                    session,
                    _query_or_hidden(session, identity.tenant_id, str(replay.result_ref)),
                )
            policy = session.scalar(
                select(ExternalSearchPolicyRecord).where(
                    ExternalSearchPolicyRecord.tenant_id == identity.tenant_id
                )
            )
            if policy is None or not policy.enabled:
                raise ExternalSearchError("external_search_disabled_by_tenant_policy")
            domains = (
                tuple(policy.official_domains)
                if use_case in SENSITIVE_EXTERNAL_SEARCH_USE_CASES
                else tuple(policy.allowed_domains)
            )
            if not domains:
                raise ExternalSearchError("external_search_official_domain_required")
            external_search_id = f"external-search-{uuid4().hex}"
            provider_name = self._provider.provider_name if self._provider is not None else "NONE"
            record = ExternalSearchQueryRecord(
                tenant_id=identity.tenant_id,
                external_search_id=external_search_id,
                query_text=query,
                query_hash=sha256(query.encode()).hexdigest(),
                use_case=use_case,
                status="RUNNING",
                provider=provider_name,
                provider_request_id=None,
                requested_by_subject_id=identity.subject_id,
                result_count=0,
                blocked_result_count=0,
                usage_credits=None,
                failure_code=None,
                usage_conclusion="PENDING_REVIEW",
                conclusion_reason=None,
                concluded_by_subject_id=None,
                started_at=now,
                completed_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=fingerprint,
                    result_ref=external_search_id,
                    expires_at=now + timedelta(days=30),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()

        input_decision = self._guardrail.inspect_messages(
            ({"role": "user", "content": query},)
        )
        if input_decision.decision == "BLOCKED":
            self._authorizer.record_guard_decision(
                identity,
                action="external_search.query_guardrail",
                decision="deny",
                reason_code="external_search_query_guardrail_blocked",
                request_id=request_id,
                resource_id=external_search_id,
            )
            return self._fail(
                identity,
                external_search_id,
                "external_search_query_guardrail_blocked",
                status="BLOCKED",
            )
        if self._provider is None:
            return self._fail(
                identity,
                external_search_id,
                "external_search_provider_not_configured",
            )
        try:
            outcome = self._provider.search(
                query,
                include_domains=domains,
                max_results=policy.max_results,
            )
        except ExternalSearchProviderError as exc:
            return self._fail(identity, external_search_id, exc.reason)
        except Exception:
            # An adapter bug or an unclassified dependency failure must not leave
            # the durable query in RUNNING forever. Keep the persisted failure stable.
            return self._fail(
                identity,
                external_search_id,
                "external_search_provider_unexpected_failure",
            )

        references: list[tuple[ExternalSearchProviderItem, str, str, str]] = []
        blocked = 0
        seen_urls: set[str] = set()
        for item in outcome.items:
            normalized = _safe_reference(item, domains)
            if normalized is None:
                blocked += 1
                continue
            canonical_url, domain, title, summary = normalized
            if canonical_url in seen_urls:
                blocked += 1
                continue
            decision = self._guardrail.inspect_messages(
                ({"role": "tool", "content": f"{title}\n{summary}"},)
            )
            if decision.decision == "BLOCKED":
                blocked += 1
                continue
            seen_urls.add(canonical_url)
            references.append((item, canonical_url, domain, decision.policy_version))
        if blocked:
            self._authorizer.record_guard_decision(
                identity,
                action="external_search.result_guardrail",
                decision="deny",
                reason_code="external_search_result_blocked",
                request_id=request_id,
                resource_id=external_search_id,
            )
        return self._complete(
            identity,
            external_search_id,
            outcome_request_id=outcome.request_id,
            usage_credits=outcome.usage_credits,
            official_domains=tuple(policy.official_domains),
            references=references,
            blocked=blocked,
        )

    def list(
        self,
        identity: IdentityContext,
        *,
        limit: int,
        request_id: str,
    ) -> tuple[ExternalSearchQueryView, ...]:
        self._require(identity, Action.READ_EXTERNAL_REFERENCE, "external-search", request_id)
        if not 1 <= limit <= 100:
            raise ExternalSearchError("external_search_limit_invalid")
        with self._database.transaction(identity.tenant_context) as session:
            records = session.scalars(
                select(ExternalSearchQueryRecord)
                .where(ExternalSearchQueryRecord.tenant_id == identity.tenant_id)
                .order_by(
                    ExternalSearchQueryRecord.created_at.desc(),
                    ExternalSearchQueryRecord.external_search_id,
                )
                .limit(limit)
            )
            return tuple(_query_view(session, item) for item in records)

    def get(
        self,
        identity: IdentityContext,
        external_search_id: str,
        *,
        request_id: str,
    ) -> ExternalSearchQueryView:
        self._require(identity, Action.READ_EXTERNAL_REFERENCE, external_search_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            return _query_view(
                session,
                _query_or_hidden(session, identity.tenant_id, external_search_id),
            )

    def conclude(
        self,
        identity: IdentityContext,
        external_search_id: str,
        *,
        conclusion: str,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> ExternalSearchQueryView:
        self._require(identity, Action.RUN_EXTERNAL_SEARCH, external_search_id, request_id)
        normalized_reason = " ".join(reason.split())
        if conclusion not in USAGE_CONCLUSIONS or not 3 <= len(normalized_reason) <= 1000:
            raise ExternalSearchError("external_search_conclusion_invalid")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = _query_or_hidden(
                session, identity.tenant_id, external_search_id, lock=True
            )
            if record.version != expected_version or record.status == "RUNNING":
                raise ExternalSearchError("external_search_state_changed", record.version)
            if (
                identity.subject_id != record.requested_by_subject_id
                and Role.DOMAIN_EXPERT not in identity.roles
            ):
                raise ExternalSearchError("external_search_conclusion_forbidden")
            record.usage_conclusion = conclusion
            record.conclusion_reason = normalized_reason
            record.concluded_by_subject_id = identity.subject_id
            record.version += 1
            record.updated_at = now
            session.flush()
            return _query_view(session, record)

    def _complete(
        self,
        identity: IdentityContext,
        external_search_id: str,
        *,
        outcome_request_id: str | None,
        usage_credits: int | None,
        official_domains: tuple[str, ...],
        references: Sequence[tuple[ExternalSearchProviderItem, str, str, str]],
        blocked: int,
    ) -> ExternalSearchQueryView:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = _query_or_hidden(
                session, identity.tenant_id, external_search_id, lock=True
            )
            if record.status != "RUNNING":
                return _query_view(session, record)
            for rank, (item, url, domain, guardrail_version) in enumerate(
                references, start=1
            ):
                title = item.title.strip()
                summary = item.content.strip()
                session.add(
                    ExternalSearchReferenceRecord(
                        tenant_id=identity.tenant_id,
                        external_reference_id=f"external-reference-{uuid4().hex}",
                        external_search_id=external_search_id,
                        rank=rank,
                        title=title,
                        url=url,
                        url_hash=sha256(url.encode()).hexdigest(),
                        domain=domain,
                        summary=summary,
                        content_hash=sha256(f"{title}\0{summary}".encode()).hexdigest(),
                        relevance_score=item.score,
                        trust_level=(
                            "OFFICIAL"
                            if _matches_any(domain, official_domains)
                            else "ALLOWLISTED"
                        ),
                        published_at=item.published_at,
                        guardrail_policy_version=guardrail_version,
                        fetched_at=now,
                        created_at=now,
                        updated_at=now,
                    )
                )
            record.provider_request_id = outcome_request_id
            record.result_count = len(references)
            record.blocked_result_count = blocked
            record.usage_credits = usage_credits
            record.status = (
                "SUCCEEDED"
                if references
                else "BLOCKED"
                if blocked
                else "NO_RESULTS"
            )
            record.failure_code = (
                "external_search_all_results_blocked"
                if record.status == "BLOCKED"
                else None
            )
            record.completed_at = now
            record.version += 1
            record.updated_at = now
            session.flush()
            return _query_view(session, record)

    def _fail(
        self,
        identity: IdentityContext,
        external_search_id: str,
        reason: str,
        *,
        status: str = "FAILED",
    ) -> ExternalSearchQueryView:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = _query_or_hidden(
                session, identity.tenant_id, external_search_id, lock=True
            )
            if record.status == "RUNNING":
                record.status = status
                record.failure_code = reason
                record.completed_at = now
                record.version += 1
                record.updated_at = now
                session.flush()
            return _query_view(session, record)

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(identity.tenant_id, resource_id),
            request_id=request_id,
        )


def _policy_view(record: ExternalSearchPolicyRecord | None) -> ExternalSearchPolicyView:
    if record is None:
        return ExternalSearchPolicyView(False, (), (), 5, None, 0, None, None)
    return ExternalSearchPolicyView(
        enabled=record.enabled,
        allowed_domains=tuple(record.allowed_domains),
        official_domains=tuple(record.official_domains),
        max_results=record.max_results,
        updated_by_subject_id=record.updated_by_subject_id,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _query_or_hidden(
    session: Session,
    tenant_id: str,
    external_search_id: str,
    *,
    lock: bool = False,
) -> ExternalSearchQueryRecord:
    statement = select(ExternalSearchQueryRecord).where(
        ExternalSearchQueryRecord.tenant_id == tenant_id,
        ExternalSearchQueryRecord.external_search_id == external_search_id,
    )
    if lock:
        statement = statement.with_for_update()
    record = session.scalar(statement)
    if record is None:
        raise ExternalSearchError("external_search_not_visible")
    return record


def _query_view(session: Session, record: ExternalSearchQueryRecord) -> ExternalSearchQueryView:
    references = tuple(
        session.scalars(
            select(ExternalSearchReferenceRecord)
            .where(
                ExternalSearchReferenceRecord.tenant_id == record.tenant_id,
                ExternalSearchReferenceRecord.external_search_id == record.external_search_id,
            )
            .order_by(ExternalSearchReferenceRecord.rank)
        )
    )
    return ExternalSearchQueryView(
        external_search_id=record.external_search_id,
        query_text=record.query_text,
        query_hash=record.query_hash,
        use_case=record.use_case,
        status=record.status,
        provider=record.provider,
        provider_request_id=record.provider_request_id,
        requested_by_subject_id=record.requested_by_subject_id,
        result_count=record.result_count,
        blocked_result_count=record.blocked_result_count,
        usage_credits=record.usage_credits,
        failure_code=record.failure_code,
        usage_conclusion=record.usage_conclusion,
        conclusion_reason=record.conclusion_reason,
        concluded_by_subject_id=record.concluded_by_subject_id,
        started_at=record.started_at,
        completed_at=record.completed_at,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
        references=tuple(
            ExternalReferenceView(
                external_reference_id=item.external_reference_id,
                rank=item.rank,
                title=item.title,
                url=item.url,
                domain=item.domain,
                summary=item.summary,
                content_hash=item.content_hash,
                relevance_score=item.relevance_score,
                trust_level=item.trust_level,
                published_at=item.published_at,
                guardrail_policy_version=item.guardrail_policy_version,
                fetched_at=item.fetched_at,
            )
            for item in references
        ),
    )


def _domains(values: tuple[str, ...]) -> tuple[str, ...]:
    if len(values) > 50:
        raise ExternalSearchError("external_search_domains_invalid")
    normalized = tuple(sorted({_domain(value) for value in values}))
    if len(normalized) != len(values):
        raise ExternalSearchError("external_search_domains_invalid")
    return normalized


def _domain(value: str) -> str:
    raw = value.strip().rstrip(".").casefold()
    try:
        normalized = raw.encode("idna").decode("ascii")
        ipaddress.ip_address(normalized)
    except ValueError:
        pass
    except UnicodeError as exc:
        raise ExternalSearchError("external_search_domains_invalid") from exc
    else:
        raise ExternalSearchError("external_search_domains_invalid")
    labels = normalized.split(".")
    if (
        len(normalized) > 253
        or len(labels) < 2
        or any(
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or not all(character.isalnum() or character == "-" for character in label)
            for label in labels
        )
    ):
        raise ExternalSearchError("external_search_domains_invalid")
    return normalized


def _safe_reference(
    item: ExternalSearchProviderItem,
    allowed_domains: tuple[str, ...],
) -> tuple[str, str, str, str] | None:
    title = item.title.strip()
    summary = item.content.strip()
    if not 1 <= len(title) <= 512 or not 1 <= len(summary) <= 4000:
        return None
    try:
        parsed = urlparse(item.url)
        domain = _domain(parsed.hostname or "")
        port = parsed.port
    except (ExternalSearchError, ValueError):
        return None
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or not _matches_any(domain, allowed_domains)
    ):
        return None
    canonical = urlunparse(("https", domain, parsed.path or "/", "", parsed.query, ""))
    if len(canonical) > 2048 or any(ord(character) < 32 for character in canonical):
        return None
    return canonical, domain, title, summary


def _matches_any(domain: str, allowed_domains: tuple[str, ...]) -> bool:
    return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in allowed_domains)
