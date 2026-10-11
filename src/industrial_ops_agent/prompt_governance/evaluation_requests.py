"""Historical prompt request format retained only for paired prompt evaluation.

There is no diagnosis business execution entrypoint in this module.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from industrial_ops_agent.agent.contracts import report_schema
from industrial_ops_agent.agent.models import AgentBudget
from industrial_ops_agent.application.diagnoses import DiagnosisWorkflowInput
from industrial_ops_agent.knowledge.models import RetrievedEvidence
from industrial_ops_agent.memory_governance.service import MemoryContextItem
from industrial_ops_agent.model_gateway.context_manifest import (
    ContextReference,
    ContextTruncation,
    canonical_sha256,
)
from industrial_ops_agent.model_gateway.service import (
    MAX_MULTIMODAL_TEXT_CHARACTERS,
    GatewayRequest,
    ModelGatewayError,
)
from industrial_ops_agent.prompting.registry import PromptBundleNotDeployed, default_prompt_registry
from industrial_ops_agent.telemetry import current_otel_trace_id


def baseline_report_request(
    command: DiagnosisWorkflowInput,
    manifest: dict[str, Any],
    *,
    query_text: str,
    evidence: list[RetrievedEvidence],
    tool_facts: list[dict[str, Any]],
    tool_failures: list[dict[str, str]],
    inference_request_id: str,
    model_alias: str,
    budget: AgentBudget,
    timeout_seconds: float,
    memories: tuple[MemoryContextItem, ...] = (),
    memory_truncations: tuple[ContextTruncation, ...] = (),
) -> GatewayRequest:
    citation_ids = [item.citation_id for item in evidence]
    schema = report_schema(citation_ids)
    selected_evidence = evidence[:4]
    evidence_payload = [
        {
            "citation_id": item.citation_id,
            "title": item.title[:255],
            "content": item.content[:1000],
            "device_model": item.device_model[:128],
            "content_checksum": item.content_checksum[:128],
        }
        for item in selected_evidence
    ]
    memory_payload = [item.prompt_payload() for item in memories]
    prompt_bundle_id = str(manifest.get("prompt_bundle", ""))
    try:
        prompt_bundle = default_prompt_registry().get(prompt_bundle_id)
    except PromptBundleNotDeployed as exc:
        raise ModelGatewayError("prompt_bundle_not_deployed") from exc
    system = prompt_bundle.render_system()
    user = json.dumps(
        {
            "incident": query_text[:3000],
            "authorized_evidence": evidence_payload,
            "authorized_enterprise_facts": tool_facts,
            "enterprise_tool_failures": tool_failures,
            "untrusted_human_confirmed_memories": memory_payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(system) + len(user) > MAX_MULTIMODAL_TEXT_CHARACTERS:
        raise ModelGatewayError("diagnosis_context_limit_exceeded")
    allowed_timeout = min(timeout_seconds, float(budget.timeout_seconds))
    context_truncations = (
        *_diagnosis_context_truncations(query_text, evidence, selected_evidence),
        *memory_truncations,
    )
    return GatewayRequest(
        inference_request_id=inference_request_id,
        model_alias=model_alias,
        required_release_id=str(manifest["model_release"]),
        subject_id=command.subject_id,
        trace_id=current_otel_trace_id() or command.request_id,
        request_class="DIAGNOSIS",
        data_classification="CONFIDENTIAL",
        messages=(
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ),
        response_schema_name="industrial_diagnosis_report",
        response_schema=schema,
        deadline=datetime.now(UTC) + timedelta(seconds=allowed_timeout),
        max_output_tokens=min(2048, budget.max_tokens),
        temperature=0.0,
        agent_run_id=command.agent_run_id,
        diagnosis_run_id=command.diagnosis_run_id,
        prompt_bundle_id=prompt_bundle.prompt_bundle_id,
        prompt_bundle_hash=prompt_bundle.content_hash,
        index_release_id=(
            str(manifest["index_release_id"]) if "index_release_id" in manifest else None
        ),
        context_evidence=tuple(
            ContextReference(item.citation_id, canonical_sha256(payload))
            for item, payload in zip(selected_evidence, evidence_payload, strict=True)
        ),
        context_tool_results=tuple(
            ContextReference(str(item["tool_call_id"]), canonical_sha256(item))
            for item in tool_facts
        ),
        context_memories=tuple(
            ContextReference(item.memory_id, canonical_sha256(payload))
            for item, payload in zip(memories, memory_payload, strict=True)
        ),
        context_token_budget=budget.max_tokens,
        context_truncated_items=context_truncations,
    )


def _diagnosis_context_truncations(
    query_text: str,
    evidence: list[RetrievedEvidence],
    selected_evidence: list[RetrievedEvidence],
) -> tuple[ContextTruncation, ...]:
    truncated: list[ContextTruncation] = []
    if len(query_text) > 3_000:
        truncated.append(
            ContextTruncation(
                source_type="QUERY",
                source_id="diagnosis-query",
                reason="CHARACTER_LIMIT",
                original_units=len(query_text),
                included_units=3_000,
            )
        )
    for item in selected_evidence:
        if len(item.content) > 1_000:
            truncated.append(
                ContextTruncation(
                    source_type="EVIDENCE",
                    source_id=item.citation_id,
                    reason="CHARACTER_LIMIT",
                    original_units=len(item.content),
                    included_units=1_000,
                )
            )
    for item in evidence[len(selected_evidence) :]:
        truncated.append(
            ContextTruncation(
                source_type="EVIDENCE",
                source_id=item.citation_id,
                reason="COUNT_LIMIT",
                original_units=max(1, len(item.content)),
                included_units=0,
            )
        )
    return tuple(truncated)
