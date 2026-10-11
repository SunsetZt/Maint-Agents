"""Select enterprise adapters without leaking credentials into typed settings."""

from __future__ import annotations

from dataclasses import dataclass

from industrial_ops_agent.config import Environment, Settings
from industrial_ops_agent.secrets import SecretName, SecretProvider
from industrial_ops_agent.tools.contracts import (
    FsmAssignmentAdapter,
    NotificationDeliveryAdapter,
    PartsAdapter,
    ProcurementDeliveryAdapter,
    ReadonlyToolAdapter,
    RefundDeliveryAdapter,
    ServiceQuotationCatalogAdapter,
)
from industrial_ops_agent.tools.customer_notifications import HttpNotificationDeliveryAdapter
from industrial_ops_agent.tools.enterprise_http import EnterpriseHttpAdapter
from industrial_ops_agent.tools.fsm_assignments import HttpFsmAssignmentAdapter
from industrial_ops_agent.tools.mcp import (
    ClientCredentialsMcpTokenProvider,
    McpReadonlyAdapter,
    SdkMcpInvocationTransport,
)
from industrial_ops_agent.tools.registry import ToolRegistry
from industrial_ops_agent.tools.procurement import HttpProcurementDeliveryAdapter
from industrial_ops_agent.tools.refunds import HttpRefundDeliveryAdapter
from industrial_ops_agent.tools.quotations import HttpServiceQuotationCatalogAdapter
from industrial_ops_agent.tools.synthetic_parts import SyntheticPartsAdapter
from industrial_ops_agent.tools.synthetic_readonly import SyntheticReadonlyAdapter


@dataclass(frozen=True, slots=True)
class EnterpriseAdapters:
    readonly: ReadonlyToolAdapter
    parts: PartsAdapter
    notifications: NotificationDeliveryAdapter | None
    procurement: ProcurementDeliveryAdapter | None
    refunds: RefundDeliveryAdapter | None
    fsm_assignments: FsmAssignmentAdapter | None
    quotations: ServiceQuotationCatalogAdapter | None


def build_enterprise_adapters(
    settings: Settings,
    secret_provider: SecretProvider,
    registry: ToolRegistry,
) -> EnterpriseAdapters:
    allow_plain_http = settings.environment in {Environment.LOCAL, Environment.TEST}
    notifications: NotificationDeliveryAdapter | None = None
    if settings.use_http_notification_delivery:
        notifications = HttpNotificationDeliveryAdapter(
            settings.notification_provider_url,
            provider_id=settings.notification_provider_id,
            api_token=secret_provider.get(SecretName.NOTIFICATION_PROVIDER_API_TOKEN),
            webhook_secret=secret_provider.get(SecretName.NOTIFICATION_WEBHOOK_SECRET),
            timeout_seconds=settings.notification_provider_timeout_seconds,
            allow_plain_http=allow_plain_http,
        )
    procurement: ProcurementDeliveryAdapter | None = None
    if settings.use_http_procurement_delivery:
        procurement = HttpProcurementDeliveryAdapter(
            settings.procurement_provider_url,
            provider_id=settings.procurement_provider_id,
            api_token=secret_provider.get(SecretName.PROCUREMENT_PROVIDER_API_TOKEN),
            webhook_secret=secret_provider.get(SecretName.PROCUREMENT_WEBHOOK_SECRET),
            timeout_seconds=settings.procurement_provider_timeout_seconds,
            allow_plain_http=allow_plain_http,
        )
    refunds: RefundDeliveryAdapter | None = None
    if settings.use_http_refund_delivery:
        refunds = HttpRefundDeliveryAdapter(
            settings.refund_provider_url,
            provider_id=settings.refund_provider_id,
            api_token=secret_provider.get(SecretName.REFUND_PROVIDER_API_TOKEN),
            webhook_secret=secret_provider.get(SecretName.REFUND_WEBHOOK_SECRET),
            timeout_seconds=settings.refund_provider_timeout_seconds,
            allow_plain_http=allow_plain_http,
        )
    fsm_assignments: FsmAssignmentAdapter | None = None
    if settings.use_http_fsm_assignment_delivery:
        fsm_assignments = HttpFsmAssignmentAdapter(
            settings.fsm_assignment_provider_url,
            provider_id=settings.fsm_assignment_provider_id,
            api_token=secret_provider.get(
                SecretName.FSM_ASSIGNMENT_PROVIDER_API_TOKEN
            ),
            webhook_secret=secret_provider.get(
                SecretName.FSM_ASSIGNMENT_WEBHOOK_SECRET
            ),
            timeout_seconds=settings.fsm_assignment_provider_timeout_seconds,
            allow_plain_http=allow_plain_http,
        )
    quotations: ServiceQuotationCatalogAdapter | None = None
    if settings.use_http_service_quotation_catalog:
        quotations = HttpServiceQuotationCatalogAdapter(
            settings.service_quotation_provider_url,
            provider_id=settings.service_quotation_provider_id,
            api_token=secret_provider.get(
                SecretName.SERVICE_QUOTATION_PROVIDER_API_TOKEN
            ),
            timeout_seconds=settings.service_quotation_provider_timeout_seconds,
            allow_plain_http=allow_plain_http,
        )
    http_adapter: EnterpriseHttpAdapter | None = None
    if settings.use_enterprise_http_parts:
        http_adapter = EnterpriseHttpAdapter(
            settings.enterprise_tool_gateway_url,
            secret_provider.get(SecretName.ENTERPRISE_TOOL_GATEWAY_TOKEN),
            timeout_seconds=settings.enterprise_tool_timeout_seconds,
            allow_plain_http=allow_plain_http,
        )
    if settings.use_mcp_enterprise_tools:
        token_provider = ClientCredentialsMcpTokenProvider(
            token_url=settings.enterprise_mcp_token_url,
            client_id=settings.enterprise_mcp_client_id,
            client_secret=secret_provider.get(SecretName.ENTERPRISE_MCP_CLIENT_SECRET),
            resource=settings.enterprise_mcp_resource,
            allowed_scopes=settings.enterprise_mcp_allowed_scopes,
            timeout_seconds=settings.enterprise_mcp_timeout_seconds,
            allow_plain_http=allow_plain_http,
        )
        readonly = McpReadonlyAdapter(
            server_id=settings.enterprise_mcp_server_id,
            server_version=settings.enterprise_mcp_server_version,
            definitions=registry.definitions(),
            token_provider=token_provider,
            transport=SdkMcpInvocationTransport(
                settings.enterprise_mcp_server_url,
                timeout_seconds=settings.enterprise_mcp_timeout_seconds,
                allow_plain_http=allow_plain_http,
            ),
            allowed_scopes=settings.enterprise_mcp_allowed_scopes,
        )
        if http_adapter is None:  # defensive: MCP read tools never own T2 reservation
            raise ValueError("MCP mode requires the enterprise HTTP parts adapter")
        return EnterpriseAdapters(
            readonly=readonly,
            parts=http_adapter,
            notifications=notifications,
            procurement=procurement,
            refunds=refunds,
            fsm_assignments=fsm_assignments,
            quotations=quotations,
        )
    if settings.use_http_enterprise_tools:
        if http_adapter is None:
            raise ValueError("HTTP mode requires the enterprise HTTP adapter")
        return EnterpriseAdapters(
            readonly=http_adapter,
            parts=http_adapter,
            notifications=notifications,
            procurement=procurement,
            refunds=refunds,
            fsm_assignments=fsm_assignments,
            quotations=quotations,
        )
    return EnterpriseAdapters(
        readonly=SyntheticReadonlyAdapter(),
        parts=SyntheticPartsAdapter(),
        notifications=notifications,
        procurement=procurement,
        refunds=refunds,
        fsm_assignments=fsm_assignments,
        quotations=quotations,
    )
