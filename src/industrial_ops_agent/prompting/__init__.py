"""Versioned, image-bound Prompt Bundle registry."""

from industrial_ops_agent.prompting.registry import (
    DEFAULT_PROMPT_BUNDLE_ID,
    GRAPH_EXTRACTION_PROMPT_BUNDLE_ID,
    MAINTENANCE_PLANNING_PROMPT_BUNDLE_ID,
    PromptBundleDefinition,
    PromptBundleRegistry,
    default_prompt_registry,
)

__all__ = [
    "DEFAULT_PROMPT_BUNDLE_ID",
    "GRAPH_EXTRACTION_PROMPT_BUNDLE_ID",
    "MAINTENANCE_PLANNING_PROMPT_BUNDLE_ID",
    "PromptBundleDefinition",
    "PromptBundleRegistry",
    "default_prompt_registry",
]
