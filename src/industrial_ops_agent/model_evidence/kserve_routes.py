"""Read-only Gateway API evidence contracts; no cluster or model dependencies."""

from __future__ import annotations

from typing import Any


def route_condition_is_true(document: dict[str, Any], condition_type: str) -> bool:
    generation = document.get("metadata", {}).get("generation")
    parents = document.get("status", {}).get("parents", [])
    if not isinstance(generation, int) or not isinstance(parents, list):
        return False
    for parent in parents:
        if not isinstance(parent, dict):
            continue
        conditions = parent.get("conditions", [])
        if not isinstance(conditions, list):
            continue
        if any(
            isinstance(condition, dict)
            and condition.get("type") == condition_type
            and condition.get("status") == "True"
            and condition.get("observedGeneration") == generation
            for condition in conditions
        ):
            return True
    return False


def route_rules_semantically_equal(
    observed_document: dict[str, Any],
    desired_document: dict[str, Any],
) -> bool:
    observed_rules = _normalized_route_rules(observed_document)
    desired_rules = _normalized_route_rules(desired_document)
    return (
        observed_rules is not None and desired_rules is not None and observed_rules == desired_rules
    )


def _normalized_route_rules(document: dict[str, Any]) -> object | None:
    spec = document.get("spec")
    if not isinstance(spec, dict):
        return None
    rules = spec.get("rules")
    if not isinstance(rules, list):
        return None
    return _without_gateway_api_defaults(rules)


def _without_gateway_api_defaults(value: object) -> object:
    if isinstance(value, list):
        return [_without_gateway_api_defaults(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                continue
            if (raw_key == "group" and item == "") or (raw_key == "kind" and item == "Service"):
                continue
            normalized[raw_key] = _without_gateway_api_defaults(item)
        return normalized
    return value


def resource_ready(document: dict[str, Any]) -> bool:
    conditions = document.get("status", {}).get("conditions", [])
    return any(
        isinstance(item, dict) and item.get("type") == "Ready" and item.get("status") == "True"
        for item in conditions
    )
