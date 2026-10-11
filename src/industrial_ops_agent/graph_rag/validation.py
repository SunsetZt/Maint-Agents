"""Pure graph structure checks shared by graph ingestion and extraction."""

from __future__ import annotations


def has_key_cycle(keys: set[str], edges: list[tuple[str, str]]) -> bool:
    adjacency: dict[str, list[str]] = {}
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> bool:
        if key in visiting:
            return True
        if key in visited:
            return False
        visiting.add(key)
        if any(visit(target) for target in adjacency.get(key, ())):
            return True
        visiting.remove(key)
        visited.add(key)
        return False

    return any(visit(key) for key in keys)
