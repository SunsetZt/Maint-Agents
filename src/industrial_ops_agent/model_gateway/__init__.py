"""Version-bound, quota-controlled access to OpenAI-compatible model serving."""

from industrial_ops_agent.model_gateway.context_manifest import (
    ContextManifest,
    ContextReference,
    ContextTruncation,
    build_context_manifest,
    context_manifest_is_valid,
)
from industrial_ops_agent.model_gateway.service import (
    EnvironmentAwareModelResolver,
    GatewayRequest,
    GatewayResponse,
    ModelGateway,
    ModelGatewayError,
    ProductionModelResolver,
    ResolvedModel,
)

__all__ = [
    "ContextManifest",
    "ContextReference",
    "ContextTruncation",
    "EnvironmentAwareModelResolver",
    "GatewayRequest",
    "GatewayResponse",
    "ModelGateway",
    "ModelGatewayError",
    "ProductionModelResolver",
    "ResolvedModel",
    "build_context_manifest",
    "context_manifest_is_valid",
]
