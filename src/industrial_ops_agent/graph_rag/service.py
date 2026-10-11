"""Governed causal-graph lifecycle, evaluation gates, and authorized path retrieval."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.domain.json import legacy_canonical_hex_digest as _digest
from industrial_ops_agent.graph_rag.store import GraphEdge, GraphNode, GraphPath, GraphStore
from industrial_ops_agent.graph_rag.validation import has_key_cycle as _has_key_cycle
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetRecord,
    CitationAnchorRecord,
    IdempotencyRecord,
    IndexReleaseRecord,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    KnowledgeDocumentVersionRecord,
    KnowledgeGraphEdgeRecord,
    KnowledgeGraphEvaluationRecord,
    KnowledgeGraphNodeRecord,
    KnowledgeGraphReleaseRecord,
)

NodeType = Literal["COMPONENT", "FAILURE_MODE", "SYMPTOM", "CAUSE", "ACTION", "MATERIAL"]
RelationType = Literal[
    "DEPENDS_ON",
    "CAUSES",
    "MANIFESTS_AS",
    "MITIGATED_BY",
    "COMPATIBLE_WITH",
    "INCOMPATIBLE_WITH",
    "PART_OF",
]

NODE_TYPES = frozenset({"COMPONENT", "FAILURE_MODE", "SYMPTOM", "CAUSE", "ACTION", "MATERIAL"})
RELATION_TYPES = frozenset(
    {
        "DEPENDS_ON",
        "CAUSES",
        "MANIFESTS_AS",
        "MITIGATED_BY",
        "COMPATIBLE_WITH",
        "INCOMPATIBLE_WITH",
        "PART_OF",
    }
)
POLICY_VERSION = "graphrag-causal-gates-v1"
SCHEMA_VERSION = "industrial-causal-graph-v1"
_KEY = re.compile(r"^[a-z0-9][a-z0-9_.:-]{1,127}$")


class KnowledgeGraphNotVisible(Exception):
    pass


class KnowledgeGraphConflict(RuntimeError):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class GraphNodeView:
    graph_node_id: str
    node_key: str
    node_type: str
    display_name: str
    citation_id: str
    document_version_id: str
    classification: str
    device_families: tuple[str, ...]
    device_models: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GraphEdgeView:
    graph_edge_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    confidence: float
    citation_id: str
    document_version_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GraphEvaluationView:
    evaluation_id: str
    status: str
    policy_version: str
    cases: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    gate_results: dict[str, bool]
    failure_codes: tuple[str, ...]
    requested_by_subject_id: str
    completed_at: datetime | None
    version: int


@dataclass(frozen=True, slots=True)
class GraphReleaseView:
    graph_release_id: str
    source_index_release_id: str
    name: str
    version: int
    status: str
    is_active: bool
    schema_version: str
    manifest_hash: str | None
    metrics: dict[str, Any]
    gate_results: dict[str, bool]
    failure_codes: tuple[str, ...]
    created_by_subject_id: str
    evaluated_by_subject_id: str | None
    activated_by_subject_id: str | None
    evaluated_at: datetime | None
    activated_at: datetime | None
    state_version: int
    created_at: datetime
    updated_at: datetime
    nodes: tuple[GraphNodeView, ...]
    edges: tuple[GraphEdgeView, ...]
    evaluation: GraphEvaluationView | None


@dataclass(frozen=True, slots=True)
class GraphPathView:
    nodes: tuple[GraphNodeView, ...]
    edges: tuple[GraphEdgeView, ...]


class KnowledgeGraphService:
    def __init__(self, database: Database, authorizer: Authorizer, store: GraphStore) -> None:
        self._database = database
        self._authorizer = authorizer
        self._store = store

    def list(self, identity: IdentityContext, *, request_id: str) -> tuple[GraphReleaseView, ...]:
        self._require(identity, Action.READ_KNOWLEDGE_GRAPH, "knowledge-graphs", request_id)
        with self._database.transaction(identity.tenant_context) as session:
            records = tuple(
                session.scalars(
                    select(KnowledgeGraphReleaseRecord)
                    .where(KnowledgeGraphReleaseRecord.tenant_id == identity.tenant_id)
                    .order_by(
                        KnowledgeGraphReleaseRecord.updated_at.desc(),
                        KnowledgeGraphReleaseRecord.graph_release_id,
                    )
                )
            )
            return tuple(_release_view(session, identity.tenant_id, record) for record in records)

    def get(
        self, identity: IdentityContext, graph_release_id: str, *, request_id: str
    ) -> GraphReleaseView:
        self._require(identity, Action.READ_KNOWLEDGE_GRAPH, graph_release_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            return _release_view(
                session,
                identity.tenant_id,
                _release_or_hidden(session, identity.tenant_id, graph_release_id),
            )

    def create(
        self,
        identity: IdentityContext,
        *,
        source_index_release_id: str,
        name: str,
        idempotency_key: str,
        request_id: str,
    ) -> GraphReleaseView:
        self._require(identity, Action.MANAGE_KNOWLEDGE_GRAPH, source_index_release_id, request_id)
        normalized_name = name.strip().lower()
        if not _KEY.fullmatch(normalized_name) or not 8 <= len(idempotency_key) <= 255:
            raise KnowledgeGraphConflict("graph_release_input_invalid")
        fingerprint = _digest(
            {"source_index_release_id": source_index_release_id, "name": normalized_name}
        )
        storage_key = f"knowledge-graph:{idempotency_key}"
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
                    raise KnowledgeGraphConflict("graph_release_idempotency_conflict")
                return _release_view(
                    session,
                    identity.tenant_id,
                    _release_or_hidden(session, identity.tenant_id, str(replay.result_ref)),
                )
            source = session.scalar(
                select(IndexReleaseRecord).where(
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.release_id == source_index_release_id,
                    IndexReleaseRecord.status == "PUBLISHED",
                )
            )
            if source is None:
                raise KnowledgeGraphNotVisible
            next_version = (
                int(
                    session.scalar(
                        select(
                            func.coalesce(func.max(KnowledgeGraphReleaseRecord.version), 0)
                        ).where(
                            KnowledgeGraphReleaseRecord.tenant_id == identity.tenant_id,
                            KnowledgeGraphReleaseRecord.name == normalized_name,
                        )
                    )
                    or 0
                )
                + 1
            )
            graph_release_id = f"knowledge-graph-{uuid4().hex}"
            record = KnowledgeGraphReleaseRecord(
                graph_release_id=graph_release_id,
                tenant_id=identity.tenant_id,
                source_index_release_id=source_index_release_id,
                name=normalized_name,
                version=next_version,
                status="DRAFT",
                is_active=False,
                schema_version=SCHEMA_VERSION,
                manifest_hash=None,
                metrics_json={},
                gate_results_json={},
                failure_codes=[],
                created_by_subject_id=identity.subject_id,
                evaluated_by_subject_id=None,
                activated_by_subject_id=None,
                evaluated_at=None,
                activated_at=None,
                state_version=1,
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
                    result_ref=graph_release_id,
                    expires_at=now + timedelta(days=30),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _release_view(session, identity.tenant_id, record)

    def add_node(
        self,
        identity: IdentityContext,
        graph_release_id: str,
        *,
        node_key: str,
        node_type: str,
        display_name: str,
        citation_id: str,
        metadata: dict[str, Any],
        expected_version: int,
        request_id: str,
    ) -> GraphReleaseView:
        normalized_key = node_key.strip().lower()
        normalized_name = display_name.strip()
        if (
            not _KEY.fullmatch(normalized_key)
            or node_type not in NODE_TYPES
            or not 2 <= len(normalized_name) <= 255
        ):
            raise KnowledgeGraphConflict("graph_node_input_invalid")
        _validate_metadata(metadata)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, graph_release_id, lock=True)
            self._require(identity, Action.MANAGE_KNOWLEDGE_GRAPH, graph_release_id, request_id)
            _require_draft_version(release, expected_version)
            _, version, document = _citation_source(
                session, identity.tenant_id, release.source_index_release_id, citation_id
            )
            session.add(
                KnowledgeGraphNodeRecord(
                    graph_node_id=f"graph-node-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    graph_release_id=graph_release_id,
                    node_key=normalized_key,
                    node_type=node_type,
                    display_name=normalized_name,
                    citation_id=citation_id,
                    document_version_id=version.document_version_id,
                    classification=document.classification,
                    device_families=list(version.device_families),
                    device_models=list(version.device_models),
                    metadata_json=metadata,
                    created_at=now,
                    updated_at=now,
                )
            )
            release.state_version += 1
            release.updated_at = now
            session.flush()
            return _release_view(session, identity.tenant_id, release)

    def add_edge(
        self,
        identity: IdentityContext,
        graph_release_id: str,
        *,
        source_node_id: str,
        target_node_id: str,
        relation_type: str,
        confidence: float,
        citation_id: str,
        metadata: dict[str, Any],
        expected_version: int,
        request_id: str,
    ) -> GraphReleaseView:
        if relation_type not in RELATION_TYPES or not 0.0 <= confidence <= 1.0:
            raise KnowledgeGraphConflict("graph_edge_input_invalid")
        _validate_metadata(metadata)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, graph_release_id, lock=True)
            self._require(identity, Action.MANAGE_KNOWLEDGE_GRAPH, graph_release_id, request_id)
            _require_draft_version(release, expected_version)
            nodes = tuple(
                session.scalars(
                    select(KnowledgeGraphNodeRecord).where(
                        KnowledgeGraphNodeRecord.tenant_id == identity.tenant_id,
                        KnowledgeGraphNodeRecord.graph_release_id == graph_release_id,
                        KnowledgeGraphNodeRecord.graph_node_id.in_(
                            (source_node_id, target_node_id)
                        ),
                    )
                )
            )
            if len(nodes) != 2 or source_node_id == target_node_id:
                raise KnowledgeGraphConflict("graph_edge_endpoint_invalid")
            _, version, _ = _citation_source(
                session, identity.tenant_id, release.source_index_release_id, citation_id
            )
            session.add(
                KnowledgeGraphEdgeRecord(
                    graph_edge_id=f"graph-edge-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    graph_release_id=graph_release_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    relation_type=relation_type,
                    confidence=confidence,
                    citation_id=citation_id,
                    document_version_id=version.document_version_id,
                    metadata_json=metadata,
                    created_by_subject_id=identity.subject_id,
                    created_at=now,
                    updated_at=now,
                )
            )
            release.state_version += 1
            release.updated_at = now
            session.flush()
            return _release_view(session, identity.tenant_id, release)

    def sync(
        self,
        identity: IdentityContext,
        graph_release_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> GraphReleaseView:
        self._require(identity, Action.MANAGE_KNOWLEDGE_GRAPH, graph_release_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, graph_release_id, lock=True)
            _require_draft_version(release, expected_version)
            node_records, edge_records = _records(session, identity.tenant_id, graph_release_id)
            if len(node_records) < 2 or not edge_records:
                raise KnowledgeGraphConflict("graph_release_content_insufficient")
            if _has_causal_cycle(node_records, edge_records):
                raise KnowledgeGraphConflict("graph_release_causal_cycle")
            nodes = tuple(_store_node(item) for item in node_records)
            edges = tuple(_store_edge(item) for item in edge_records)
            manifest_hash = _digest(
                {
                    "schema_version": release.schema_version,
                    "nodes": [_node_dict(node) for node in nodes],
                    "edges": [_edge_dict(edge) for edge in edges],
                }
            )
        self._store.replace_release(identity.tenant_id, graph_release_id, nodes, edges)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, graph_release_id, lock=True)
            _require_draft_version(release, expected_version)
            release.status = "SYNCED"
            release.manifest_hash = manifest_hash
            release.state_version += 1
            release.updated_at = now
            session.flush()
            return _release_view(session, identity.tenant_id, release)

    def evaluate(
        self,
        identity: IdentityContext,
        graph_release_id: str,
        *,
        cases: tuple[dict[str, Any], ...],
        expected_version: int,
        request_id: str,
    ) -> GraphReleaseView:
        self._require(identity, Action.EVALUATE_KNOWLEDGE_GRAPH, graph_release_id, request_id)
        normalized_cases = tuple(_validate_case(case) for case in cases)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, graph_release_id, lock=True)
            if release.state_version != expected_version or release.status != "SYNCED":
                raise KnowledgeGraphConflict("graph_release_state_changed", release.state_version)
            if release.created_by_subject_id == identity.subject_id:
                raise KnowledgeGraphConflict("graph_evaluation_separation_required")
            nodes, edges = _records(session, identity.tenant_id, graph_release_id)
            node_by_id = {node.graph_node_id: node for node in nodes}
            edge_by_id = {edge.graph_edge_id: edge for edge in edges}
        passed_cases = 0
        returned_paths = 0
        for case in normalized_cases:
            paths = self._store.query_paths(
                identity.tenant_id,
                graph_release_id,
                start_node_key=str(case["start_node_key"]),
                target_node_type=str(case["target_node_type"]),
                relation_types=tuple(case["relation_types"]),
                max_hops=int(case["max_hops"]),
                limit=20,
            )
            returned_paths += len(paths)
            if any(
                _path_matches_projection(
                    path,
                    node_by_id,
                    edge_by_id,
                    start_node_key=str(case["start_node_key"]),
                    target_node_type=str(case["target_node_type"]),
                    relation_types=frozenset(case["relation_types"]),
                )
                and node_by_id[path.node_ids[-1]].node_key == case["expected_target_node_key"]
                for path in paths
            ):
                passed_cases += 1
        recall = passed_cases / len(normalized_cases) if normalized_cases else 0.0
        gates = {
            "minimum_gold_cases": len(normalized_cases) >= 3,
            "expected_path_recall": recall >= 0.8,
            "citation_integrity": all(node.citation_id for node in nodes)
            and all(edge.citation_id for edge in edges),
            "causal_graph_acyclic": not _has_causal_cycle(nodes, edges),
            "independent_evaluator": True,
        }
        failures = sorted(name for name, passed in gates.items() if not passed)
        metrics: dict[str, Any] = {
            "gold_case_count": len(normalized_cases),
            "passed_case_count": passed_cases,
            "expected_path_recall": recall,
            "returned_path_count": returned_paths,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, graph_release_id, lock=True)
            if release.state_version != expected_version or release.status != "SYNCED":
                raise KnowledgeGraphConflict("graph_release_state_changed", release.state_version)
            session.add(
                KnowledgeGraphEvaluationRecord(
                    evaluation_id=f"graph-evaluation-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    graph_release_id=graph_release_id,
                    status="FAILED" if failures else "PASSED",
                    policy_version=POLICY_VERSION,
                    cases_json=list(normalized_cases),
                    metrics_json=metrics,
                    gate_results_json=gates,
                    failure_codes=failures,
                    requested_by_subject_id=identity.subject_id,
                    completed_at=now,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            release.status = "REJECTED" if failures else "EVALUATED"
            release.metrics_json = metrics
            release.gate_results_json = gates
            release.failure_codes = failures
            release.evaluated_by_subject_id = identity.subject_id
            release.evaluated_at = now
            release.state_version += 1
            release.updated_at = now
            session.flush()
            return _release_view(session, identity.tenant_id, release)

    def activate(
        self,
        identity: IdentityContext,
        graph_release_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> GraphReleaseView:
        self._require(identity, Action.ACTIVATE_KNOWLEDGE_GRAPH, graph_release_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, graph_release_id, lock=True)
            if (
                release.state_version != expected_version
                or release.status != "EVALUATED"
                or not release.gate_results_json
                or not all(release.gate_results_json.values())
            ):
                raise KnowledgeGraphConflict(
                    "graph_release_activation_gate_failed", release.state_version
                )
            session.execute(
                update(KnowledgeGraphReleaseRecord)
                .where(
                    KnowledgeGraphReleaseRecord.tenant_id == identity.tenant_id,
                    KnowledgeGraphReleaseRecord.name == release.name,
                    KnowledgeGraphReleaseRecord.is_active.is_(True),
                )
                .values(
                    is_active=False,
                    status="RETIRED",
                    state_version=KnowledgeGraphReleaseRecord.state_version + 1,
                    updated_at=now,
                )
            )
            release.status = "ACTIVE"
            release.is_active = True
            release.activated_by_subject_id = identity.subject_id
            release.activated_at = now
            release.state_version += 1
            release.updated_at = now
            session.flush()
            return _release_view(session, identity.tenant_id, release)

    def query(
        self,
        identity: IdentityContext,
        *,
        graph_name: str,
        start_node_key: str,
        target_node_type: str | None,
        relation_types: tuple[str, ...],
        max_hops: int,
        limit: int,
        request_id: str,
    ) -> tuple[GraphPathView, ...]:
        self._require(identity, Action.READ_KNOWLEDGE_GRAPH, "active-knowledge-graph", request_id)
        key = start_node_key.strip().lower()
        normalized_graph_name = graph_name.strip().lower()
        if (
            not _KEY.fullmatch(normalized_graph_name)
            or not _KEY.fullmatch(key)
            or target_node_type is not None
            and target_node_type not in NODE_TYPES
            or not relation_types
            or any(item not in RELATION_TYPES for item in relation_types)
            or not 1 <= max_hops <= 4
            or not 1 <= limit <= 20
        ):
            raise KnowledgeGraphConflict("graph_query_invalid")
        with self._database.transaction(identity.tenant_context) as session:
            release = session.scalar(
                select(KnowledgeGraphReleaseRecord)
                .join(
                    IndexReleaseRecord,
                    IndexReleaseRecord.release_id
                    == KnowledgeGraphReleaseRecord.source_index_release_id,
                )
                .where(
                    KnowledgeGraphReleaseRecord.tenant_id == identity.tenant_id,
                    KnowledgeGraphReleaseRecord.name == normalized_graph_name,
                    KnowledgeGraphReleaseRecord.status == "ACTIVE",
                    KnowledgeGraphReleaseRecord.is_active.is_(True),
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.status == "PUBLISHED",
                )
            )
            if release is None:
                raise KnowledgeGraphNotVisible
            node_records, edge_records = _records(
                session, identity.tenant_id, release.graph_release_id
            )
            node_map = {item.graph_node_id: item for item in node_records}
            edge_map = {item.graph_edge_id: item for item in edge_records}
            version_ids = {item.document_version_id for item in node_records}
            versions = {
                item.document_version_id: item
                for item in session.scalars(
                    select(KnowledgeDocumentVersionRecord).where(
                        KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id,
                        KnowledgeDocumentVersionRecord.document_version_id.in_(version_ids),
                    )
                )
            }
            visible_models = frozenset(
                value
                for value in session.scalars(
                    select(AssetRecord.model_code).where(
                        AssetRecord.tenant_id == identity.tenant_id,
                        AssetRecord.asset_id.in_(identity.asset_ids),
                    )
                )
                if value is not None
            )
        raw_paths = self._store.query_paths(
            identity.tenant_id,
            release.graph_release_id,
            start_node_key=key,
            target_node_type=target_node_type,
            relation_types=tuple(sorted(set(relation_types))),
            max_hops=max_hops,
            limit=limit,
        )
        results: list[GraphPathView] = []
        for path in raw_paths:
            graph_nodes = tuple(node_map.get(node_id) for node_id in path.node_ids)
            graph_edges = tuple(edge_map.get(edge_id) for edge_id in path.edge_ids)
            if any(item is None for item in graph_nodes) or any(
                item is None for item in graph_edges
            ):
                continue
            typed_nodes = tuple(item for item in graph_nodes if item is not None)
            typed_edges = tuple(item for item in graph_edges if item is not None)
            if (
                not typed_nodes
                or len(typed_nodes) != len(typed_edges) + 1
                or typed_nodes[0].node_key != key
                or target_node_type is not None
                and typed_nodes[-1].node_type != target_node_type
                or any(
                    edge.source_node_id != typed_nodes[index].graph_node_id
                    or edge.target_node_id != typed_nodes[index + 1].graph_node_id
                    or edge.relation_type not in relation_types
                    for index, edge in enumerate(typed_edges)
                )
            ):
                continue
            if not all(
                _node_visible(
                    identity,
                    node,
                    versions.get(node.document_version_id),
                    visible_models,
                )
                for node in typed_nodes
            ):
                continue
            results.append(
                GraphPathView(
                    nodes=tuple(_node_view(item) for item in typed_nodes),
                    edges=tuple(_edge_view(item) for item in typed_edges),
                )
            )
        return tuple(results)

    def _require(
        self, identity: IdentityContext, action: Action, resource_id: str, request_id: str
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(identity.tenant_id, resource_id),
            request_id=request_id,
        )


def purge_graph_source(
    session: Session,
    *,
    tenant_id: str,
    document_version_id: str,
) -> tuple[int, int, tuple[str, ...]]:
    """Remove SQL graph derivatives before their citation anchors are deleted."""

    nodes = tuple(
        session.scalars(
            select(KnowledgeGraphNodeRecord).where(
                KnowledgeGraphNodeRecord.tenant_id == tenant_id,
                KnowledgeGraphNodeRecord.document_version_id == document_version_id,
            )
        )
    )
    node_ids = tuple(item.graph_node_id for item in nodes)
    release_ids = {item.graph_release_id for item in nodes}
    edge_scope = KnowledgeGraphEdgeRecord.document_version_id == document_version_id
    if node_ids:
        edge_scope = (
            edge_scope
            | KnowledgeGraphEdgeRecord.source_node_id.in_(node_ids)
            | KnowledgeGraphEdgeRecord.target_node_id.in_(node_ids)
        )
    edge_filter = [KnowledgeGraphEdgeRecord.tenant_id == tenant_id, edge_scope]
    edges = tuple(session.scalars(select(KnowledgeGraphEdgeRecord).where(*edge_filter)))
    release_ids.update(item.graph_release_id for item in edges)
    edge_ids = tuple(item.graph_edge_id for item in edges)
    if edge_ids:
        session.execute(
            delete(KnowledgeGraphEdgeRecord).where(
                KnowledgeGraphEdgeRecord.tenant_id == tenant_id,
                KnowledgeGraphEdgeRecord.graph_edge_id.in_(edge_ids),
            )
        )
    if node_ids:
        session.execute(
            delete(KnowledgeGraphNodeRecord).where(
                KnowledgeGraphNodeRecord.tenant_id == tenant_id,
                KnowledgeGraphNodeRecord.graph_node_id.in_(node_ids),
            )
        )
    now = datetime.now(UTC)
    for release in session.scalars(
        select(KnowledgeGraphReleaseRecord).where(
            KnowledgeGraphReleaseRecord.tenant_id == tenant_id,
            KnowledgeGraphReleaseRecord.graph_release_id.in_(release_ids),
        )
    ):
        release.status = "REVOKED"
        release.is_active = False
        release.failure_codes = sorted(set(release.failure_codes) | {"source_document_removed"})
        release.state_version += 1
        release.updated_at = now
    return len(nodes), len(edges), tuple(sorted(release_ids))


def apply_reviewed_candidate(
    session: Session,
    identity: IdentityContext,
    release: KnowledgeGraphReleaseRecord,
    candidate: dict[str, Any],
) -> None:
    """Apply one validated candidate bundle inside the caller-owned transaction."""

    if release.status != "DRAFT":
        raise KnowledgeGraphConflict("graph_release_state_changed", release.state_version)
    nodes_value = candidate.get("nodes")
    edges_value = candidate.get("edges")
    if not isinstance(nodes_value, list) or not isinstance(edges_value, list):
        raise KnowledgeGraphConflict("graph_review_candidate_invalid")
    existing_nodes, existing_edges = _records(
        session,
        identity.tenant_id,
        release.graph_release_id,
    )
    node_ids_by_key = {item.node_key: item.graph_node_id for item in existing_nodes}
    existing_keys = set(node_ids_by_key)
    now = datetime.now(UTC)
    new_nodes: list[KnowledgeGraphNodeRecord] = []
    for item in nodes_value:
        if not isinstance(item, dict):
            raise KnowledgeGraphConflict("graph_review_candidate_invalid")
        key = item.get("node_key")
        node_type = item.get("node_type")
        display_name = item.get("display_name")
        citation_id = item.get("citation_id")
        metadata = item.get("metadata")
        if (
            not isinstance(key, str)
            or not _KEY.fullmatch(key)
            or key in node_ids_by_key
            or node_type not in NODE_TYPES
            or not isinstance(display_name, str)
            or not 2 <= len(display_name) <= 255
            or not isinstance(citation_id, str)
            or not isinstance(metadata, dict)
        ):
            raise KnowledgeGraphConflict("graph_review_candidate_conflict")
        _validate_metadata(metadata)
        _, version, document = _citation_source(
            session,
            identity.tenant_id,
            release.source_index_release_id,
            citation_id,
        )
        graph_node_id = f"graph-node-{uuid4().hex}"
        node_ids_by_key[key] = graph_node_id
        new_nodes.append(
            KnowledgeGraphNodeRecord(
                graph_node_id=graph_node_id,
                tenant_id=identity.tenant_id,
                graph_release_id=release.graph_release_id,
                node_key=key,
                node_type=node_type,
                display_name=display_name,
                citation_id=citation_id,
                document_version_id=version.document_version_id,
                classification=document.classification,
                device_families=list(version.device_families),
                device_models=list(version.device_models),
                metadata_json=metadata,
                created_at=now,
                updated_at=now,
            )
        )
    if set(node_ids_by_key) == existing_keys:
        raise KnowledgeGraphConflict("graph_review_candidate_invalid")

    node_keys_by_id = {node_id: key for key, node_id in node_ids_by_key.items()}
    edge_identities = {
        (
            node_keys_by_id[item.source_node_id],
            node_keys_by_id[item.target_node_id],
            item.relation_type,
        )
        for item in existing_edges
        if item.source_node_id in node_keys_by_id and item.target_node_id in node_keys_by_id
    }
    causes = [
        (node_keys_by_id[item.source_node_id], node_keys_by_id[item.target_node_id])
        for item in existing_edges
        if item.relation_type == "CAUSES"
        and item.source_node_id in node_keys_by_id
        and item.target_node_id in node_keys_by_id
    ]
    new_edges: list[KnowledgeGraphEdgeRecord] = []
    for item in edges_value:
        if not isinstance(item, dict):
            raise KnowledgeGraphConflict("graph_review_candidate_invalid")
        source_key = item.get("source_node_key")
        target_key = item.get("target_node_key")
        relation_type = item.get("relation_type")
        confidence = item.get("confidence")
        citation_id = item.get("citation_id")
        metadata = item.get("metadata")
        identity_key = (source_key, target_key, relation_type)
        if (
            not isinstance(source_key, str)
            or source_key not in node_ids_by_key
            or not isinstance(target_key, str)
            or target_key not in node_ids_by_key
            or source_key == target_key
            or relation_type not in RELATION_TYPES
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or not isinstance(citation_id, str)
            or not isinstance(metadata, dict)
            or identity_key in edge_identities
        ):
            raise KnowledgeGraphConflict("graph_review_candidate_conflict")
        _validate_metadata(metadata)
        _, version, _ = _citation_source(
            session,
            identity.tenant_id,
            release.source_index_release_id,
            citation_id,
        )
        edge_identities.add(identity_key)
        if relation_type == "CAUSES":
            causes.append((source_key, target_key))
        new_edges.append(
            KnowledgeGraphEdgeRecord(
                graph_edge_id=f"graph-edge-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                graph_release_id=release.graph_release_id,
                source_node_id=node_ids_by_key[source_key],
                target_node_id=node_ids_by_key[target_key],
                relation_type=relation_type,
                confidence=float(confidence),
                citation_id=citation_id,
                document_version_id=version.document_version_id,
                metadata_json=metadata,
                created_by_subject_id=identity.subject_id,
                created_at=now,
                updated_at=now,
            )
        )
    if _has_key_cycle(set(node_ids_by_key), causes):
        raise KnowledgeGraphConflict("graph_review_candidate_causal_cycle")
    session.add_all(new_nodes)
    session.flush()
    session.add_all(new_edges)
    session.flush()
    release.state_version += 1
    release.updated_at = now


def _release_or_hidden(
    session: Session, tenant_id: str, graph_release_id: str, *, lock: bool = False
) -> KnowledgeGraphReleaseRecord:
    query = select(KnowledgeGraphReleaseRecord).where(
        KnowledgeGraphReleaseRecord.tenant_id == tenant_id,
        KnowledgeGraphReleaseRecord.graph_release_id == graph_release_id,
    )
    record = session.scalar(query.with_for_update() if lock else query)
    if record is None:
        raise KnowledgeGraphNotVisible
    return record


def _records(
    session: Session, tenant_id: str, graph_release_id: str
) -> tuple[tuple[KnowledgeGraphNodeRecord, ...], tuple[KnowledgeGraphEdgeRecord, ...]]:
    nodes = tuple(
        session.scalars(
            select(KnowledgeGraphNodeRecord)
            .where(
                KnowledgeGraphNodeRecord.tenant_id == tenant_id,
                KnowledgeGraphNodeRecord.graph_release_id == graph_release_id,
            )
            .order_by(KnowledgeGraphNodeRecord.node_key)
        )
    )
    edges = tuple(
        session.scalars(
            select(KnowledgeGraphEdgeRecord)
            .where(
                KnowledgeGraphEdgeRecord.tenant_id == tenant_id,
                KnowledgeGraphEdgeRecord.graph_release_id == graph_release_id,
            )
            .order_by(KnowledgeGraphEdgeRecord.graph_edge_id)
        )
    )
    return nodes, edges


def _citation_source(
    session: Session, tenant_id: str, source_release_id: str, citation_id: str
) -> tuple[CitationAnchorRecord, KnowledgeDocumentVersionRecord, KnowledgeDocumentRecord]:
    row = session.execute(
        select(CitationAnchorRecord, KnowledgeDocumentVersionRecord, KnowledgeDocumentRecord)
        .join(KnowledgeChunkRecord, KnowledgeChunkRecord.chunk_id == CitationAnchorRecord.chunk_id)
        .join(
            KnowledgeDocumentVersionRecord,
            KnowledgeDocumentVersionRecord.document_version_id
            == CitationAnchorRecord.document_version_id,
        )
        .join(
            KnowledgeDocumentRecord,
            KnowledgeDocumentRecord.document_id == KnowledgeDocumentVersionRecord.document_id,
        )
        .where(
            CitationAnchorRecord.tenant_id == tenant_id,
            CitationAnchorRecord.citation_id == citation_id,
            KnowledgeChunkRecord.tenant_id == tenant_id,
            KnowledgeChunkRecord.release_id == source_release_id,
            KnowledgeDocumentVersionRecord.tenant_id == tenant_id,
            KnowledgeDocumentVersionRecord.status == "PUBLISHED",
            KnowledgeDocumentRecord.tenant_id == tenant_id,
        )
    ).one_or_none()
    if row is None:
        raise KnowledgeGraphConflict("graph_citation_not_in_source_release")
    anchor, version, document = row
    return anchor, version, document


def _require_draft_version(release: KnowledgeGraphReleaseRecord, expected_version: int) -> None:
    if release.state_version != expected_version or release.status != "DRAFT":
        raise KnowledgeGraphConflict("graph_release_state_changed", release.state_version)


def _validate_metadata(metadata: dict[str, Any]) -> None:
    try:
        encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise KnowledgeGraphConflict("graph_metadata_invalid") from exc
    if len(encoded.encode()) > 8_192:
        raise KnowledgeGraphConflict("graph_metadata_too_large")


def _validate_case(case: dict[str, Any]) -> dict[str, Any]:
    required = {
        "start_node_key",
        "target_node_type",
        "relation_types",
        "max_hops",
        "expected_target_node_key",
    }
    if set(case) != required:
        raise KnowledgeGraphConflict("graph_evaluation_case_invalid")
    start = str(case["start_node_key"]).strip().lower()
    target = str(case["expected_target_node_key"]).strip().lower()
    target_type = str(case["target_node_type"])
    relations = case["relation_types"]
    hops = case["max_hops"]
    if (
        not _KEY.fullmatch(start)
        or not _KEY.fullmatch(target)
        or target_type not in NODE_TYPES
        or not isinstance(relations, list)
        or not relations
        or any(item not in RELATION_TYPES for item in relations)
        or not isinstance(hops, int)
        or isinstance(hops, bool)
        or not 1 <= hops <= 4
    ):
        raise KnowledgeGraphConflict("graph_evaluation_case_invalid")
    return {
        "start_node_key": start,
        "target_node_type": target_type,
        "relation_types": sorted(set(relations)),
        "max_hops": hops,
        "expected_target_node_key": target,
    }


def _has_causal_cycle(
    nodes: tuple[KnowledgeGraphNodeRecord, ...], edges: tuple[KnowledgeGraphEdgeRecord, ...]
) -> bool:
    return _has_key_cycle(
        {node.graph_node_id for node in nodes},
        [
            (edge.source_node_id, edge.target_node_id)
            for edge in edges
            if edge.relation_type == "CAUSES"
        ],
    )



def _path_matches_projection(
    path: GraphPath,
    nodes: dict[str, KnowledgeGraphNodeRecord],
    edges: dict[str, KnowledgeGraphEdgeRecord],
    *,
    start_node_key: str,
    target_node_type: str,
    relation_types: frozenset[str],
) -> bool:
    if not path.node_ids or len(path.node_ids) != len(path.edge_ids) + 1:
        return False
    path_nodes = tuple(nodes.get(node_id) for node_id in path.node_ids)
    path_edges = tuple(edges.get(edge_id) for edge_id in path.edge_ids)
    if any(node is None for node in path_nodes) or any(edge is None for edge in path_edges):
        return False
    typed_nodes = tuple(node for node in path_nodes if node is not None)
    typed_edges = tuple(edge for edge in path_edges if edge is not None)
    return (
        typed_nodes[0].node_key == start_node_key
        and typed_nodes[-1].node_type == target_node_type
        and all(
            edge.source_node_id == typed_nodes[index].graph_node_id
            and edge.target_node_id == typed_nodes[index + 1].graph_node_id
            and edge.relation_type in relation_types
            for index, edge in enumerate(typed_edges)
        )
    )


def _node_visible(
    identity: IdentityContext,
    node: KnowledgeGraphNodeRecord,
    version: KnowledgeDocumentVersionRecord | None,
    visible_models: frozenset[str],
) -> bool:
    if version is None or version.status != "PUBLISHED":
        return False
    now = datetime.now(UTC)
    valid_from = _as_utc(version.valid_from)
    valid_to = _as_utc(version.valid_to) if version.valid_to is not None else None
    roles = {role.value for role in identity.roles}
    acl_allowed = (
        not version.acl_subject_ids
        and not version.acl_roles
        or identity.subject_id in version.acl_subject_ids
        or bool(roles.intersection(version.acl_roles))
    )
    restricted_allowed = node.classification != "restricted" or bool(
        roles.intersection(version.acl_roles)
    )
    model_allowed = not node.device_models or bool(visible_models.intersection(node.device_models))
    return (
        acl_allowed
        and model_allowed
        and restricted_allowed
        and valid_from <= now
        and (valid_to is None or valid_to > now)
    )


def _release_view(
    session: Session, tenant_id: str, record: KnowledgeGraphReleaseRecord
) -> GraphReleaseView:
    nodes, edges = _records(session, tenant_id, record.graph_release_id)
    evaluation = session.scalar(
        select(KnowledgeGraphEvaluationRecord)
        .where(
            KnowledgeGraphEvaluationRecord.tenant_id == tenant_id,
            KnowledgeGraphEvaluationRecord.graph_release_id == record.graph_release_id,
        )
        .order_by(KnowledgeGraphEvaluationRecord.created_at.desc())
        .limit(1)
    )
    return GraphReleaseView(
        graph_release_id=record.graph_release_id,
        source_index_release_id=record.source_index_release_id,
        name=record.name,
        version=record.version,
        status=record.status,
        is_active=record.is_active,
        schema_version=record.schema_version,
        manifest_hash=record.manifest_hash,
        metrics=dict(record.metrics_json),
        gate_results=dict(record.gate_results_json),
        failure_codes=tuple(record.failure_codes),
        created_by_subject_id=record.created_by_subject_id,
        evaluated_by_subject_id=record.evaluated_by_subject_id,
        activated_by_subject_id=record.activated_by_subject_id,
        evaluated_at=record.evaluated_at,
        activated_at=record.activated_at,
        state_version=record.state_version,
        created_at=record.created_at,
        updated_at=record.updated_at,
        nodes=tuple(_node_view(item) for item in nodes),
        edges=tuple(_edge_view(item) for item in edges),
        evaluation=_evaluation_view(evaluation) if evaluation is not None else None,
    )


def _node_view(record: KnowledgeGraphNodeRecord) -> GraphNodeView:
    return GraphNodeView(
        graph_node_id=record.graph_node_id,
        node_key=record.node_key,
        node_type=record.node_type,
        display_name=record.display_name,
        citation_id=record.citation_id,
        document_version_id=record.document_version_id,
        classification=record.classification,
        device_families=tuple(record.device_families),
        device_models=tuple(record.device_models),
        metadata=dict(record.metadata_json),
    )


def _edge_view(record: KnowledgeGraphEdgeRecord) -> GraphEdgeView:
    return GraphEdgeView(
        graph_edge_id=record.graph_edge_id,
        source_node_id=record.source_node_id,
        target_node_id=record.target_node_id,
        relation_type=record.relation_type,
        confidence=record.confidence,
        citation_id=record.citation_id,
        document_version_id=record.document_version_id,
        metadata=dict(record.metadata_json),
    )


def _evaluation_view(record: KnowledgeGraphEvaluationRecord) -> GraphEvaluationView:
    return GraphEvaluationView(
        evaluation_id=record.evaluation_id,
        status=record.status,
        policy_version=record.policy_version,
        cases=tuple(record.cases_json),
        metrics=dict(record.metrics_json),
        gate_results=dict(record.gate_results_json),
        failure_codes=tuple(record.failure_codes),
        requested_by_subject_id=record.requested_by_subject_id,
        completed_at=record.completed_at,
        version=record.version,
    )


def _store_node(record: KnowledgeGraphNodeRecord) -> GraphNode:
    return GraphNode(
        record.graph_node_id,
        record.node_key,
        record.node_type,
        record.display_name,
        record.citation_id,
        record.document_version_id,
    )


def _store_edge(record: KnowledgeGraphEdgeRecord) -> GraphEdge:
    return GraphEdge(
        record.graph_edge_id,
        record.source_node_id,
        record.target_node_id,
        record.relation_type,
        record.citation_id,
        record.document_version_id,
        record.confidence,
    )


def _node_dict(node: GraphNode) -> dict[str, Any]:
    return {
        "graph_node_id": node.graph_node_id,
        "node_key": node.node_key,
        "node_type": node.node_type,
        "display_name": node.display_name,
        "citation_id": node.citation_id,
        "document_version_id": node.document_version_id,
    }


def _edge_dict(edge: GraphEdge) -> dict[str, Any]:
    return {
        "graph_edge_id": edge.graph_edge_id,
        "source_node_id": edge.source_node_id,
        "target_node_id": edge.target_node_id,
        "relation_type": edge.relation_type,
        "citation_id": edge.citation_id,
        "document_version_id": edge.document_version_id,
        "confidence": edge.confidence,
    }
