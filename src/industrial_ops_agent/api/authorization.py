"""HTTP authorization boundary shared by tenant collection routes."""

from __future__ import annotations

from industrial_ops_agent.api.errors import authorization_denied
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext


def authorize_tenant_resource(
    authorizer: Authorizer,
    identity: IdentityContext,
    action: Action,
    resource_id: str,
    request_id: str,
) -> None:
    try:
        authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )
    except AuthorizationDenied as exc:
        raise authorization_denied() from exc
