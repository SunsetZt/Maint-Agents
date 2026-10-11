"""Isolated DRAFT-only prompt evaluation, never a business route."""

from __future__ import annotations

import asyncio
from time import monotonic
from urllib.parse import urlparse

import httpx
from sqlalchemy import select

from industrial_ops_agent.model_gateway.openai import OpenAiCompatibleTransport
from industrial_ops_agent.model_gateway.service import ModelGatewayError, ResolvedModel
from industrial_ops_agent.persistence.models import ModelAliasRecord, ModelReleaseRecord
from industrial_ops_agent.releases.service import _digest_json, _evaluate_release_gates
from industrial_ops_agent.releases.staging_smoke import require_staging_smoke_admission_current

ALIAS = "industrial-diagnosis-staging"


class CandidatePromptResolver:
    def __init__(self, database, *, endpoint, release_id, package_digest):
        url = urlparse(endpoint or "")
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/", "/v1"}
            or not release_id
            or not package_digest
            or len(package_digest) != 64
            or any(c not in "0123456789abcdef" for c in package_digest)
        ):
            raise ValueError("candidate_prompt_runtime_not_configured")
        self.database, self.release_id = database, release_id
        self.endpoint = endpoint.rstrip("/").removesuffix("/v1") + "/v1"
        self.package_digest = package_digest
        self.adapter_hash = None

    def resolve(self, context, alias_name):
        if alias_name != ALIAS:
            raise ModelGatewayError("candidate_prompt_alias_invalid")
        with self.database.transaction(context) as session:
            release = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == context.tenant_id,
                    ModelReleaseRecord.release_id == self.release_id,
                )
            )
            if (
                release is None
                or release.status != "DRAFT"
                or release.traffic_percent != 0
                or release.target_environment != "STAGING"
                or release.manifest_json.get("prompt_evaluation_only") is not True
                or _digest_json(release.manifest_json) != release.manifest_hash
            ):
                raise ModelGatewayError("candidate_prompt_requires_evaluation_only_draft")
            # The candidate endpoint cannot be one of this tenant's live routes.
            aliases = session.scalars(
                select(ModelAliasRecord).where(
                    ModelAliasRecord.tenant_id == context.tenant_id,
                    ModelAliasRecord.status == "ACTIVE",
                )
            )
            if any(
                a.active_release_id == self.release_id
                or urlparse(a.endpoint_url).netloc == urlparse(self.endpoint).netloc
                for a in aliases
            ):
                raise ModelGatewayError("candidate_prompt_business_route_forbidden")
            try:
                require_staging_smoke_admission_current(session, release)
                gates = _evaluate_release_gates(session, context.tenant_id, release)
            except (ValueError, KeyError, TypeError) as exc:
                raise ModelGatewayError("candidate_prompt_evidence_invalid") from exc
            failures = {key for key, passed in gates.items() if not passed}
            if failures - {"prompt_bundle_governed", "evaluation_only_not_publishable"}:
                raise ModelGatewayError("candidate_prompt_release_prerequisites_failed")
            self.adapter_hash = release.manifest_json["adapter"]["content_hash"].removeprefix(
                "sha256:"
            )
            return ResolvedModel(
                alias=alias_name,
                release_id=release.release_id,
                manifest_hash=release.manifest_hash,
                endpoint_url=self.endpoint,
                runtime_profile=release.manifest_json["runtime"]["profile_id"],
                multimodal_model_ids={},
                served_model_id=release.release_id,
                served_model_digest=self.package_digest,
                target_environment="STAGING",
            )


class CandidateOwnedTransport:
    """Reuse the serving owner and normal Gateway validation, audit and tenant quota."""

    def __init__(self, resolver, api_key, *, allow_plain_http=False):
        if urlparse(resolver.endpoint).scheme == "http" and not allow_plain_http:
            raise ValueError("candidate_prompt_https_required")
        self.resolver, self.api_key = resolver, api_key
        # Candidate addresses are on the private serving network. Control and
        # completions must use the same direct path, never inherited host proxies.
        self.inner = OpenAiCompatibleTransport(
            api_key, allow_plain_http=allow_plain_http, trust_env=False
        )

    async def complete(self, model, request, *, timeout_seconds):
        if request.request_class != "DIAGNOSIS" or model.release_id != self.resolver.release_id:
            raise ModelGatewayError("candidate_prompt_request_forbidden")
        base = self.resolver.endpoint.removesuffix("/v1")
        headers = {"Authorization": "Bearer " + self.api_key}
        started = monotonic()
        async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as client:
            try:
                response = await client.post(
                    base + "/runtime/load", headers=headers, timeout=min(102.0, timeout_seconds)
                )
                if response.status_code != 200:
                    raise ModelGatewayError("candidate_prompt_load_failed")
                state = response.json()
                if (
                    state.get("state") != "READY"
                    or state.get("prepared") is not True
                    or state.get("model_id") != model.release_id
                    or state.get("package_digest") != self.resolver.package_digest
                    or state.get("adapter_content_hash") != self.resolver.adapter_hash
                ):
                    raise ModelGatewayError("candidate_prompt_runtime_binding_mismatch")
                remaining = timeout_seconds - (monotonic() - started)
                if remaining <= 0:
                    raise ModelGatewayError("candidate_prompt_deadline_exceeded")
                return await self.inner.complete(model, request, timeout_seconds=remaining)
            except (httpx.HTTPError, ValueError) as exc:
                raise ModelGatewayError("candidate_prompt_runtime_unavailable") from exc
            finally:
                # An uncertain load must also receive cleanup; never auto-retry inference.
                try:
                    await self._unload(client, base, headers)
                except (httpx.HTTPError, ValueError) as exc:
                    raise ModelGatewayError("candidate_prompt_cleanup_unconfirmed") from exc

    async def _unload(self, client, base, headers):
        # The response can reach the client before the owner releases its request
        # lock (or while idle cleanup is in progress). Match the normal router's
        # bounded runtime_busy handling; only unload is retried, never inference.
        deadline = monotonic() + 15.0
        while (remaining := deadline - monotonic()) > 0:
            response = await client.post(
                base + "/runtime/unload", headers=headers, timeout=remaining
            )
            state = response.json()
            if (response.status_code == 409 and isinstance(state, dict)
                    and isinstance(state.get("error"), dict)
                    and state["error"].get("code") == "runtime_busy"):
                await asyncio.sleep(min(0.1, max(0, deadline - monotonic())))
                continue
            if (response.status_code == 200 and isinstance(state, dict)
                    and state.get("state") == "UNLOADED"):
                return
            raise ModelGatewayError("candidate_prompt_cleanup_unconfirmed")
        raise ModelGatewayError("candidate_prompt_cleanup_unconfirmed")
