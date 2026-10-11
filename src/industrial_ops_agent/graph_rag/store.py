"""Static, parameterized graph-store boundary for GraphRAG projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from neo4j import GraphDatabase, RoutingControl
from neo4j.exceptions import Neo4jError


class GraphStoreUnavailable(RuntimeError):
    """The sidecar is unavailable; callers must continue without graph evidence."""


@dataclass(frozen=True, slots=True)
class GraphNode:
    graph_node_id: str
    node_key: str
    node_type: str
    display_name: str
    citation_id: str
    document_version_id: str


@dataclass(frozen=True, slots=True)
class GraphEdge:
    graph_edge_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    citation_id: str
    document_version_id: str
    confidence: float


@dataclass(frozen=True, slots=True)
class GraphPath:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]


class GraphStore(Protocol):
    def replace_release(
        self,
        tenant_id: str,
        graph_release_id: str,
        nodes: tuple[GraphNode, ...],
        edges: tuple[GraphEdge, ...],
    ) -> None: ...

    def query_paths(
        self,
        tenant_id: str,
        graph_release_id: str,
        *,
        start_node_key: str,
        target_node_type: str | None,
        relation_types: tuple[str, ...],
        max_hops: int,
        limit: int,
    ) -> tuple[GraphPath, ...]: ...

    def remove_document_version(self, tenant_id: str, document_version_id: str) -> None: ...

    def ping(self) -> bool: ...

    def close(self) -> None: ...


class InMemoryGraphStore:
    """Deterministic adapter for local operation and contract tests."""

    def __init__(self) -> None:
        self._releases: dict[
            tuple[str, str], tuple[dict[str, GraphNode], tuple[GraphEdge, ...]]
        ] = {}

    def replace_release(
        self,
        tenant_id: str,
        graph_release_id: str,
        nodes: tuple[GraphNode, ...],
        edges: tuple[GraphEdge, ...],
    ) -> None:
        self._releases[(tenant_id, graph_release_id)] = (
            {node.graph_node_id: node for node in nodes},
            edges,
        )

    def query_paths(
        self,
        tenant_id: str,
        graph_release_id: str,
        *,
        start_node_key: str,
        target_node_type: str | None,
        relation_types: tuple[str, ...],
        max_hops: int,
        limit: int,
    ) -> tuple[GraphPath, ...]:
        nodes, edges = self._releases.get((tenant_id, graph_release_id), ({}, ()))
        starts = [node for node in nodes.values() if node.node_key == start_node_key]
        allowed = set(relation_types)
        adjacency: dict[str, list[GraphEdge]] = {}
        for edge in edges:
            if edge.relation_type in allowed:
                adjacency.setdefault(edge.source_node_id, []).append(edge)
        found: list[GraphPath] = []
        for start in starts:
            frontier: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
                (start.graph_node_id, (start.graph_node_id,), ())
            ]
            while frontier and len(found) < limit:
                current, node_path, edge_path = frontier.pop(0)
                if edge_path and (
                    target_node_type is None or nodes[current].node_type == target_node_type
                ):
                    found.append(GraphPath(node_path, edge_path))
                if len(edge_path) >= max_hops:
                    continue
                for edge in sorted(adjacency.get(current, ()), key=lambda item: item.graph_edge_id):
                    if edge.target_node_id not in nodes or edge.target_node_id in node_path:
                        continue
                    frontier.append(
                        (
                            edge.target_node_id,
                            (*node_path, edge.target_node_id),
                            (*edge_path, edge.graph_edge_id),
                        )
                    )
        return tuple(found[:limit])

    def remove_document_version(self, tenant_id: str, document_version_id: str) -> None:
        for key, (nodes, edges) in tuple(self._releases.items()):
            if key[0] != tenant_id:
                continue
            kept = {
                node_id: node
                for node_id, node in nodes.items()
                if node.document_version_id != document_version_id
            }
            kept_edges = tuple(
                edge
                for edge in edges
                if edge.document_version_id != document_version_id
                and edge.source_node_id in kept
                and edge.target_node_id in kept
            )
            self._releases[key] = kept, kept_edges

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        self._releases.clear()


class Neo4jGraphStore:
    """Official Neo4j-driver adapter; labels and Cypher stay application-owned."""

    _PATH_QUERIES = {
        hops: (
            "MATCH path=(start:KnowledgeEntity)-[rels:RELATES*1.."
            + str(hops)
            + "]->(target:KnowledgeEntity) "
            "WHERE start.tenant_id = $tenant_id "
            "AND start.graph_release_id = $graph_release_id "
            "AND start.node_key = $start_node_key "
            "AND ($target_node_type IS NULL OR target.node_type = $target_node_type) "
            "AND all(n IN nodes(path) WHERE n.tenant_id = $tenant_id "
            "AND n.graph_release_id = $graph_release_id) "
            "AND all(r IN rels WHERE r.tenant_id = $tenant_id "
            "AND r.graph_release_id = $graph_release_id "
            "AND r.relation_type IN $relation_types) "
            "AND all(n IN nodes(path) WHERE single(m IN nodes(path) WHERE m = n)) "
            "RETURN [n IN nodes(path) | n.graph_node_id] AS node_ids, "
            "[r IN rels | r.graph_edge_id] AS edge_ids "
            "ORDER BY length(path), edge_ids LIMIT $limit"
        )
        for hops in range(1, 5)
    }

    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        database: str,
        connection_timeout_seconds: float = 5.0,
    ) -> None:
        self._database = database
        self._driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            connection_timeout=connection_timeout_seconds,
        )

    def replace_release(
        self,
        tenant_id: str,
        graph_release_id: str,
        nodes: tuple[GraphNode, ...],
        edges: tuple[GraphEdge, ...],
    ) -> None:
        node_rows = [asdict(node) for node in nodes]
        edge_rows = [asdict(edge) for edge in edges]

        def write(tx: Any) -> None:
            tx.run(
                "MATCH (n:KnowledgeEntity {tenant_id: $tenant_id, "
                "graph_release_id: $graph_release_id}) DETACH DELETE n",
                tenant_id=tenant_id,
                graph_release_id=graph_release_id,
            ).consume()
            tx.run(
                "UNWIND $nodes AS row CREATE (n:KnowledgeEntity) SET n = row, "
                "n.tenant_id = $tenant_id, n.graph_release_id = $graph_release_id",
                nodes=node_rows,
                tenant_id=tenant_id,
                graph_release_id=graph_release_id,
            ).consume()
            tx.run(
                "UNWIND $edges AS row "
                "MATCH (source:KnowledgeEntity {tenant_id: $tenant_id, "
                "graph_release_id: $graph_release_id, graph_node_id: row.source_node_id}) "
                "MATCH (target:KnowledgeEntity {tenant_id: $tenant_id, "
                "graph_release_id: $graph_release_id, graph_node_id: row.target_node_id}) "
                "CREATE (source)-[r:RELATES]->(target) SET r = row, "
                "r.tenant_id = $tenant_id, r.graph_release_id = $graph_release_id",
                edges=edge_rows,
                tenant_id=tenant_id,
                graph_release_id=graph_release_id,
            ).consume()

        try:
            with self._driver.session(database=self._database) as session:
                session.execute_write(write)
        except Neo4jError as exc:
            raise GraphStoreUnavailable("graph_store_sync_failed") from exc

    def query_paths(
        self,
        tenant_id: str,
        graph_release_id: str,
        *,
        start_node_key: str,
        target_node_type: str | None,
        relation_types: tuple[str, ...],
        max_hops: int,
        limit: int,
    ) -> tuple[GraphPath, ...]:
        query = self._PATH_QUERIES.get(max_hops)
        if query is None:
            raise ValueError("max_hops must be between 1 and 4")
        try:
            records, _, _ = self._driver.execute_query(
                query,
                tenant_id=tenant_id,
                graph_release_id=graph_release_id,
                start_node_key=start_node_key,
                target_node_type=target_node_type,
                relation_types=list(relation_types),
                limit=limit,
                database_=self._database,
                routing_=RoutingControl.READ,
            )
        except Neo4jError as exc:
            raise GraphStoreUnavailable("graph_store_query_failed") from exc
        return tuple(
            GraphPath(tuple(record["node_ids"]), tuple(record["edge_ids"])) for record in records
        )

    def remove_document_version(self, tenant_id: str, document_version_id: str) -> None:
        try:
            self._driver.execute_query(
                "MATCH ()-[r:RELATES {tenant_id: $tenant_id, "
                "document_version_id: $document_version_id}]->() DELETE r",
                tenant_id=tenant_id,
                document_version_id=document_version_id,
                database_=self._database,
            )
            self._driver.execute_query(
                "MATCH (n:KnowledgeEntity {tenant_id: $tenant_id, "
                "document_version_id: $document_version_id}) DETACH DELETE n",
                tenant_id=tenant_id,
                document_version_id=document_version_id,
                database_=self._database,
            )
        except Neo4jError as exc:
            raise GraphStoreUnavailable("graph_store_deletion_failed") from exc

    def ping(self) -> bool:
        try:
            self._driver.verify_connectivity()
        except Neo4jError:
            return False
        return True

    def close(self) -> None:
        self._driver.close()
