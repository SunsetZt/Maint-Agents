"""Static OpenSearch boundary with mandatory authorization filters before recall."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlparse

from opensearchpy import OpenSearch
from opensearchpy.exceptions import OpenSearchException
from opensearchpy.helpers import bulk

from industrial_ops_agent.knowledge.embedding import cosine_similarity, tokenize


class SearchStoreUnavailable(RuntimeError):
    """The optional search sidecar is unavailable; PostgreSQL remains authoritative."""


@dataclass(frozen=True, slots=True)
class SearchDocument:
    chunk_id: str
    citation_id: str
    tenant_id: str
    search_profile_id: str
    source_index_release_id: str
    document_id: str
    document_version_id: str
    document_version: int
    title: str
    content: str
    content_checksum: str
    source_checksum: str
    page_number: int | None
    embedding: tuple[float, ...]
    acl_subject_ids: tuple[str, ...]
    acl_roles: tuple[str, ...]
    device_families: tuple[str, ...]
    device_models: tuple[str, ...]
    valid_from: datetime
    valid_to: datetime | None
    classification: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    document: SearchDocument
    sparse_rank: int | None
    vector_rank: int | None
    fused_score: float


class SearchStore(Protocol):
    def replace_release(self, index_name: str, documents: tuple[SearchDocument, ...]) -> None: ...

    def search(
        self,
        index_name: str,
        *,
        tenant_id: str,
        search_profile_id: str,
        subject_id: str,
        roles: tuple[str, ...],
        device_family: str | None,
        device_model: str,
        query_text: str,
        query_embedding: tuple[float, ...] | None,
        as_of: datetime,
        limit: int,
    ) -> tuple[SearchHit, ...]: ...

    def count(self, index_name: str) -> int: ...

    def remove_document_version(self, tenant_id: str, document_version_id: str) -> None: ...

    def ping(self) -> bool: ...

    def close(self) -> None: ...


class InMemorySearchStore:
    """Deterministic profile adapter for local use and contract tests."""

    def __init__(self) -> None:
        self._indices: dict[str, tuple[SearchDocument, ...]] = {}

    def replace_release(self, index_name: str, documents: tuple[SearchDocument, ...]) -> None:
        self._indices[index_name] = documents

    def search(
        self,
        index_name: str,
        *,
        tenant_id: str,
        search_profile_id: str,
        subject_id: str,
        roles: tuple[str, ...],
        device_family: str | None,
        device_model: str,
        query_text: str,
        query_embedding: tuple[float, ...] | None,
        as_of: datetime,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        candidates = [
            item
            for item in self._indices.get(index_name, ())
            if _authorized(
                item,
                tenant_id=tenant_id,
                search_profile_id=search_profile_id,
                subject_id=subject_id,
                roles=roles,
                device_family=device_family,
                device_model=device_model,
                as_of=as_of,
            )
        ]
        query_tokens = set(tokenize(query_text))
        sparse = sorted(
            (
                item
                for item in candidates
                if query_tokens.intersection(tokenize(f"{item.title} {item.content}"))
            ),
            key=lambda item: (
                -len(query_tokens.intersection(tokenize(f"{item.title} {item.content}"))),
                item.chunk_id,
            ),
        )
        vector = (
            sorted(
                candidates,
                key=lambda item: (
                    -cosine_similarity(list(query_embedding), list(item.embedding)),
                    item.chunk_id,
                ),
            )
            if query_embedding is not None
            else []
        )
        return _rrf(sparse, vector, limit)

    def count(self, index_name: str) -> int:
        return len(self._indices.get(index_name, ()))

    def remove_document_version(self, tenant_id: str, document_version_id: str) -> None:
        for index_name, documents in tuple(self._indices.items()):
            self._indices[index_name] = tuple(
                item
                for item in documents
                if item.tenant_id != tenant_id or item.document_version_id != document_version_id
            )

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        self._indices.clear()


class OpenSearchStore:
    """Official-client adapter; callers never provide raw DSL or arbitrary index names."""

    def __init__(
        self,
        *,
        url: str,
        index_prefix: str,
        username: str | None = None,
        password: str | None = None,
        ca_certs: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OpenSearch URL must be absolute HTTP(S)")
        if not index_prefix or not all(char.isalnum() or char == "-" for char in index_prefix):
            raise ValueError("OpenSearch index prefix is invalid")
        self._prefix = index_prefix
        auth = (username, password) if username and password else None
        self._client = OpenSearch(
            hosts=[
                {
                    "host": parsed.hostname,
                    "port": parsed.port or (443 if parsed.scheme == "https" else 80),
                }
            ],
            http_auth=auth,
            http_compress=True,
            use_ssl=parsed.scheme == "https",
            verify_certs=parsed.scheme == "https",
            ssl_assert_hostname=parsed.scheme == "https",
            ca_certs=ca_certs,
            timeout=timeout_seconds,
        )

    def replace_release(self, index_name: str, documents: tuple[SearchDocument, ...]) -> None:
        self._validate_index(index_name)
        dimensions = {len(document.embedding) for document in documents}
        if len(dimensions) != 1 or not 8 <= next(iter(dimensions), 0) <= 8192:
            raise ValueError("OpenSearch embedding dimension is invalid or inconsistent")
        try:
            if self._client.indices.exists(index=index_name):
                self._client.indices.delete(index=index_name)
            self._client.indices.create(
                index=index_name,
                body=_index_definition(next(iter(dimensions))),
            )
            if documents:
                succeeded, errors = bulk(
                    self._client,
                    (
                        {
                            "_op_type": "index",
                            "_index": index_name,
                            "_id": document.chunk_id,
                            "_source": _document_source(document),
                        }
                        for document in documents
                    ),
                    refresh=True,
                    raise_on_error=False,
                )
                if errors or succeeded != len(documents):
                    raise SearchStoreUnavailable("OpenSearch bulk projection was incomplete")
        except OpenSearchException as exc:
            raise SearchStoreUnavailable("OpenSearch projection failed") from exc

    def search(
        self,
        index_name: str,
        *,
        tenant_id: str,
        search_profile_id: str,
        subject_id: str,
        roles: tuple[str, ...],
        device_family: str | None,
        device_model: str,
        query_text: str,
        query_embedding: tuple[float, ...] | None,
        as_of: datetime,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        self._validate_index(index_name)
        candidate_limit = max(limit * 4, 12)
        filters = _authorization_filters(
            tenant_id=tenant_id,
            search_profile_id=search_profile_id,
            subject_id=subject_id,
            roles=roles,
            device_family=device_family,
            device_model=device_model,
            as_of=as_of,
        )
        try:
            sparse_response = self._client.search(
                index=index_name,
                body={
                    "size": candidate_limit,
                    "query": {
                        "bool": {
                            "filter": filters,
                            "must": [
                                {
                                    "multi_match": {
                                        "query": query_text,
                                        "fields": ["title^2", "content"],
                                    }
                                }
                            ],
                        }
                    },
                },
            )
            vector_response = (
                self._client.search(
                    index=index_name,
                    body={
                        "size": candidate_limit,
                        "query": {
                            "knn": {
                                "embedding": {
                                    "vector": list(query_embedding),
                                    "k": candidate_limit,
                                    "filter": {"bool": {"filter": filters}},
                                }
                            }
                        },
                    },
                )
                if query_embedding is not None
                else None
            )
        except OpenSearchException as exc:
            raise SearchStoreUnavailable("OpenSearch query failed") from exc
        sparse = [_source_document(hit["_source"]) for hit in sparse_response["hits"]["hits"]]
        vector = (
            [_source_document(hit["_source"]) for hit in vector_response["hits"]["hits"]]
            if vector_response is not None
            else []
        )
        return _rrf(sparse, vector, limit)

    def count(self, index_name: str) -> int:
        self._validate_index(index_name)
        try:
            return int(self._client.count(index=index_name)["count"])
        except OpenSearchException as exc:
            raise SearchStoreUnavailable("OpenSearch count failed") from exc

    def remove_document_version(self, tenant_id: str, document_version_id: str) -> None:
        try:
            self._client.delete_by_query(
                index=f"{self._prefix}-*",
                body={
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"tenant_id": tenant_id}},
                                {"term": {"document_version_id": document_version_id}},
                            ]
                        }
                    }
                },
                conflicts="proceed",
                refresh=True,
                ignore_unavailable=True,
            )
        except OpenSearchException as exc:
            raise SearchStoreUnavailable("OpenSearch deletion propagation failed") from exc

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except OpenSearchException:
            return False

    def close(self) -> None:
        self._client.close()

    def _validate_index(self, index_name: str) -> None:
        if not index_name.startswith(f"{self._prefix}-") or not all(
            char.isalnum() or char in {"-", "_"} for char in index_name
        ):
            raise ValueError("OpenSearch index is outside the application-owned prefix")


def _index_definition(dimension: int = 16) -> dict[str, Any]:
    if not 8 <= dimension <= 8192:
        raise ValueError("OpenSearch embedding dimension is invalid")
    return {
        "settings": {"index": {"knn": True, "number_of_shards": 1}},
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "citation_id": {"type": "keyword"},
                "tenant_id": {"type": "keyword"},
                "search_profile_id": {"type": "keyword"},
                "source_index_release_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "document_version_id": {"type": "keyword"},
                "document_version": {"type": "integer"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "content_checksum": {"type": "keyword"},
                "source_checksum": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                    },
                },
                "acl_subject_ids": {"type": "keyword"},
                "acl_roles": {"type": "keyword"},
                "acl_open": {"type": "boolean"},
                "device_families": {"type": "keyword"},
                "device_models": {"type": "keyword"},
                "family_open": {"type": "boolean"},
                "model_open": {"type": "boolean"},
                "valid_from": {"type": "date"},
                "valid_to": {"type": "date"},
                "classification": {"type": "keyword"},
            },
        },
    }


def _document_source(document: SearchDocument) -> dict[str, Any]:
    source = asdict(document)
    source["embedding"] = list(document.embedding)
    source["acl_open"] = not document.acl_subject_ids and not document.acl_roles
    source["family_open"] = not document.device_families
    source["model_open"] = not document.device_models
    source["valid_from"] = document.valid_from.isoformat()
    source["valid_to"] = document.valid_to.isoformat() if document.valid_to else None
    return source


def _source_document(source: dict[str, Any]) -> SearchDocument:
    return SearchDocument(
        chunk_id=str(source["chunk_id"]),
        citation_id=str(source["citation_id"]),
        tenant_id=str(source["tenant_id"]),
        search_profile_id=str(source["search_profile_id"]),
        source_index_release_id=str(source["source_index_release_id"]),
        document_id=str(source["document_id"]),
        document_version_id=str(source["document_version_id"]),
        document_version=int(source["document_version"]),
        title=str(source["title"]),
        content=str(source["content"]),
        content_checksum=str(source["content_checksum"]),
        source_checksum=str(source["source_checksum"]),
        page_number=int(source["page_number"]) if source.get("page_number") is not None else None,
        embedding=tuple(float(value) for value in source["embedding"]),
        acl_subject_ids=tuple(str(value) for value in source["acl_subject_ids"]),
        acl_roles=tuple(str(value) for value in source["acl_roles"]),
        device_families=tuple(str(value) for value in source["device_families"]),
        device_models=tuple(str(value) for value in source["device_models"]),
        valid_from=datetime.fromisoformat(str(source["valid_from"])),
        valid_to=(
            datetime.fromisoformat(str(source["valid_to"])) if source.get("valid_to") else None
        ),
        classification=str(source["classification"]),
    )


def _authorization_filters(
    *,
    tenant_id: str,
    search_profile_id: str,
    subject_id: str,
    roles: tuple[str, ...],
    device_family: str | None,
    device_model: str,
    as_of: datetime,
) -> list[dict[str, Any]]:
    acl_should: list[dict[str, Any]] = [
        {"term": {"acl_open": True}},
        {"term": {"acl_subject_ids": subject_id}},
    ]
    classification_should: list[dict[str, Any]] = [
        {"bool": {"must_not": [{"term": {"classification": "restricted"}}]}}
    ]
    if roles:
        acl_should.append({"terms": {"acl_roles": list(roles)}})
        classification_should.append({"terms": {"acl_roles": list(roles)}})
    family_should: list[dict[str, Any]] = [{"term": {"family_open": True}}]
    if device_family is not None:
        family_should.append({"term": {"device_families": device_family}})
    return [
        {"term": {"tenant_id": tenant_id}},
        {"term": {"search_profile_id": search_profile_id}},
        {"bool": {"should": acl_should, "minimum_should_match": 1}},
        {"bool": {"should": classification_should, "minimum_should_match": 1}},
        {
            "bool": {
                "should": [
                    {"term": {"model_open": True}},
                    {"term": {"device_models": device_model}},
                ],
                "minimum_should_match": 1,
            }
        },
        {"bool": {"should": family_should, "minimum_should_match": 1}},
        {"range": {"valid_from": {"lte": as_of.isoformat()}}},
        {
            "bool": {
                "should": [
                    {"bool": {"must_not": [{"exists": {"field": "valid_to"}}]}},
                    {"range": {"valid_to": {"gt": as_of.isoformat()}}},
                ],
                "minimum_should_match": 1,
            }
        },
    ]


def _authorized(
    document: SearchDocument,
    *,
    tenant_id: str,
    search_profile_id: str,
    subject_id: str,
    roles: tuple[str, ...],
    device_family: str | None,
    device_model: str,
    as_of: datetime,
) -> bool:
    role_set = set(roles)
    normalized_as_of = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    valid_from = (
        document.valid_from
        if document.valid_from.tzinfo
        else document.valid_from.replace(tzinfo=UTC)
    )
    valid_to = document.valid_to
    if valid_to is not None and valid_to.tzinfo is None:
        valid_to = valid_to.replace(tzinfo=UTC)
    role_allowed = bool(role_set.intersection(document.acl_roles))
    return (
        document.tenant_id == tenant_id
        and document.search_profile_id == search_profile_id
        and valid_from <= normalized_as_of
        and (valid_to is None or valid_to > normalized_as_of)
        and (
            (not document.acl_subject_ids and not document.acl_roles)
            or subject_id in document.acl_subject_ids
            or role_allowed
        )
        and (not document.device_models or device_model in document.device_models)
        and (not document.device_families or device_family in document.device_families)
        and (document.classification != "restricted" or role_allowed)
    )


def _rrf(
    sparse: list[SearchDocument], vector: list[SearchDocument], limit: int
) -> tuple[SearchHit, ...]:
    sparse_rank = {item.chunk_id: index for index, item in enumerate(sparse, 1)}
    vector_rank = {item.chunk_id: index for index, item in enumerate(vector, 1)}
    candidates = {item.chunk_id: item for item in [*sparse, *vector]}
    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            -(
                (1 / (60 + sparse_rank[item.chunk_id]) if item.chunk_id in sparse_rank else 0)
                + (1 / (60 + vector_rank[item.chunk_id]) if item.chunk_id in vector_rank else 0)
            ),
            item.chunk_id,
        ),
    )
    return tuple(
        SearchHit(
            document=item,
            sparse_rank=sparse_rank.get(item.chunk_id),
            vector_rank=vector_rank.get(item.chunk_id),
            fused_score=(
                (1 / (60 + sparse_rank[item.chunk_id]) if item.chunk_id in sparse_rank else 0)
                + (1 / (60 + vector_rank[item.chunk_id]) if item.chunk_id in vector_rank else 0)
            ),
        )
        for item in ordered[:limit]
    )
