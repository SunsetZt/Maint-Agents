"""Typed process configuration with safe defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Supported deployment environments."""

    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class SecretBackend(StrEnum):
    ENVIRONMENT = "environment"
    VAULT = "vault"


class ModelTargetEnvironment(StrEnum):
    """Governed Release environment accepted by this deployment."""

    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"


class EnterpriseToolsMode(StrEnum):
    AUTO = "auto"
    SYNTHETIC = "synthetic"
    HTTP = "http"
    MCP = "mcp"


class NotificationDeliveryMode(StrEnum):
    DISABLED = "disabled"
    HTTP = "http"


class ProcurementDeliveryMode(StrEnum):
    DISABLED = "disabled"
    HTTP = "http"


class RefundDeliveryMode(StrEnum):
    DISABLED = "disabled"
    HTTP = "http"


class FsmAssignmentDeliveryMode(StrEnum):
    DISABLED = "disabled"
    HTTP = "http"


class ServiceQuotationCatalogMode(StrEnum):
    DISABLED = "disabled"
    HTTP = "http"


class Settings(BaseSettings):
    """Non-secret M1 process settings.

    Runtime credentials are intentionally absent. Task 7 introduces the
    dedicated SecretProvider boundary instead of growing this model into a
    credential container.
    """

    model_config = SettingsConfigDict(
        env_prefix="IOAP_",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    service_name: str = "industrial-ops-api"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    log_level: str = "INFO"
    log_file: Path | None = None
    release_version: str = "development"
    otel_enabled: bool = False
    otel_service_namespace: str = "industrial-ops"
    otel_exporter_otlp_traces_endpoint: str = "http://otel-collector:4318/v1/traces"
    otel_exporter_certificate_file: Path | None = None
    otel_exporter_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    otel_trace_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    langfuse_otel_traces_endpoint: str | None = None
    langfuse_otel_authorization_file: Path | None = None
    api_prefix: str = "/api/v1"
    api_host: str = "0.0.0.0"  # noqa: S104 - container listener, not an access policy
    api_port: int = Field(default=8000, ge=1, le=65535)
    secret_backend: SecretBackend = SecretBackend.ENVIRONMENT
    vault_address: str = "http://vault:8200"
    vault_mount: str = "secret"
    vault_path: str = "industrial-ops/m1"
    vault_token_file: Path = Path("/run/industrial-ops-secrets/vault-token")
    database_host: str = "postgres"
    redis_url: str = "redis://redis:6379/0"
    minio_endpoint: str = "minio:9000"
    minio_secure: bool = False
    minio_bucket: str = "industrial-ops-media"
    dataset_bucket: str = "industrial-ops-datasets"
    oidc_issuer: str = "http://localhost:8080/realms/industrial-ops"
    oidc_jwks_url: str = "http://keycloak:8080/realms/industrial-ops/protocol/openid-connect/certs"
    oidc_audience: str = "industrial-ops-web"
    clamav_host: str = "clamav"
    clamav_port: int = Field(default=3310, ge=1, le=65535)
    worker_poll_seconds: float = Field(default=2.0, gt=0, le=60)
    worker_ready_file: Path = Path("/tmp/worker-ready")
    worker_metrics_host: str = "0.0.0.0"  # noqa: S104 - internal container listener
    workflow_worker_metrics_port: int = Field(default=9101, ge=1, le=65535)
    event_worker_metrics_port: int = Field(default=9102, ge=1, le=65535)
    telemetry_worker_metrics_port: int = Field(default=9109, ge=1, le=65535)
    opa_decision_url: str = "http://opa:8181/v1/data/industrial_ops/authz/result"
    opa_timeout_seconds: float = Field(default=1.5, gt=0, le=10)
    temporal_address: str = "temporal:7233"
    temporal_task_queue: str = "industrial-ops-m2"
    kafka_bootstrap_servers: str = "kafka:9092"
    kafka_work_order_topic: str = "ops.work-order.v1"
    kafka_feedback_consumer_group: str = "industrial-ops-m3-feedback-v1"
    kafka_telemetry_topic: str = "ops.telemetry.v1"
    kafka_telemetry_consumer_group: str = "industrial-ops-m7-telemetry-v1"
    label_studio_url: str = "http://label-studio:8080"
    label_studio_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    label_studio_project_key: str = "industrial-root-cause"
    label_studio_project_id: int = Field(default=1, ge=1)
    openlineage_url: str = "http://marquez:5000"
    openlineage_namespace: str = "industrial-ops"
    dataset_pipeline_engine: str = "spark"
    mlflow_tracking_url: str = "http://mlflow:5000"
    mlflow_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    kubernetes_api_url: str = "https://kubernetes.default.svc"
    kubernetes_ca_file: Path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
    kubernetes_token_file: Path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    kubernetes_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    prometheus_url: str = "http://prometheus:9090"
    prometheus_ca_file: Path | None = None
    prometheus_token_file: Path | None = None
    prometheus_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    deployment_controller_tenant_id: str = "tenant-industrial"
    deployment_controller_subject_id: str = "m5-deployment-controller"
    deployment_controller_interval_seconds: float = Field(default=30.0, ge=5.0, le=300)
    deployment_observation_interval_seconds: int = Field(default=60, ge=30, le=3600)
    deployment_controller_leader_election_enabled: bool = False
    deployment_controller_lease_namespace: str = Field(
        default="industrial-ops",
        min_length=1,
        max_length=63,
        pattern=r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$",
    )
    deployment_controller_lease_name: str = Field(
        default="industrial-ops-release-controller",
        min_length=1,
        max_length=63,
        pattern=r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$",
    )
    deployment_controller_pod_uid: str | None = Field(default=None, max_length=128)
    deployment_controller_lease_duration_seconds: int = Field(default=15, gt=0, le=300)
    deployment_controller_renew_deadline_seconds: int = Field(default=10, gt=0, le=299)
    deployment_controller_retry_period_seconds: int = Field(default=2, gt=0, le=298)
    edge_diagnosis_enabled: bool = False
    edge_pack_signing_key_id: str = Field(
        default="field-edge-signing-v1",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    model_gateway_required_environment: ModelTargetEnvironment = ModelTargetEnvironment.STAGING
    model_gateway_alias: str = "industrial-diagnosis-staging"
    model_gateway_timeout_seconds: float = Field(default=90.0, gt=0, le=180)
    graph_extraction_model_alias: str = Field(
        default="industrial-graphrag-extraction",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    graph_extraction_timeout_seconds: float = Field(default=90.0, gt=0, le=180)
    timeseries_inference_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    rul_inference_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    vlm_model_alias: str = "industrial-diagnosis-staging"
    video_vlm_model_alias: str = "industrial-diagnosis-staging"
    video_vlm_timeout_seconds: float = Field(default=45.0, gt=0, le=120)
    enterprise_project_adoption_enabled: bool = False
    enterprise_project_repo_root: Path = Path(".")
    enterprise_project_adoption_path: Path = Path(
        "artifacts/enterprise-project-adoption/acceptance.json"
    )
    enterprise_project_adoption_cache_seconds: float = Field(
        default=15.0,
        ge=0,
        le=300,
    )
    paddle_ocr_language: str = Field(default="ch", min_length=2, max_length=16)
    paddle_ocr_component_id: str = Field(default="PaddleOCR@3.0", min_length=1, max_length=255)
    docling_artifacts_path: Path | None = None
    enterprise_tools_mode: EnterpriseToolsMode = EnterpriseToolsMode.AUTO
    enterprise_tool_gateway_url: str = "http://enterprise-tool-gateway:8090"
    enterprise_tool_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    notification_delivery_mode: NotificationDeliveryMode = NotificationDeliveryMode.DISABLED
    notification_provider_url: str = "http://notification-provider:8093"
    notification_provider_id: str = Field(
        default="enterprise-notify",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    notification_provider_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    procurement_delivery_mode: ProcurementDeliveryMode = ProcurementDeliveryMode.DISABLED
    procurement_provider_url: str = "http://procurement-provider:8094"
    procurement_provider_id: str = Field(
        default="enterprise-erp",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    procurement_provider_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    refund_delivery_mode: RefundDeliveryMode = RefundDeliveryMode.DISABLED
    refund_provider_url: str = "http://refund-provider:8095"
    refund_provider_id: str = Field(
        default="enterprise-finance",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    refund_provider_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    fsm_assignment_delivery_mode: FsmAssignmentDeliveryMode = FsmAssignmentDeliveryMode.DISABLED
    fsm_assignment_provider_url: str = "http://fsm-provider:8096"
    fsm_assignment_provider_id: str = Field(
        default="enterprise-fsm",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    fsm_assignment_provider_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    service_quotation_catalog_mode: ServiceQuotationCatalogMode = (
        ServiceQuotationCatalogMode.DISABLED
    )
    service_quotation_provider_url: str = "http://service-quotation-provider:8097"
    service_quotation_provider_id: str = Field(
        default="enterprise-cpq",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    service_quotation_provider_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        le=30,
    )
    enterprise_mcp_server_url: str = "http://enterprise-mcp-server:8091/mcp"
    enterprise_mcp_server_id: str = Field(
        default="industrial-enterprise-tools",
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$",
    )
    enterprise_mcp_server_version: str = Field(
        default="1.0.0",
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.+-]*$",
    )
    enterprise_mcp_token_url: str = (
        "http://keycloak:8080/realms/industrial-ops/protocol/openid-connect/token"
    )
    enterprise_mcp_client_id: str = Field(
        default="industrial-ops-mcp-client",
        min_length=1,
        max_length=128,
    )
    enterprise_mcp_resource: str = "http://enterprise-mcp-server:8091/mcp"
    enterprise_mcp_allowed_scopes: tuple[str, ...] = (
        "industrial.asset.read",
        "industrial.warranty.read",
        "industrial.parts.read",
        "industrial.schedule.read",
        "industrial.work-order.read",
    )
    enterprise_mcp_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    supplier_a2a_enabled: bool = False
    supplier_a2a_base_url: str = "http://supplier-agent:8092"
    supplier_a2a_agent_name: str = Field(
        default="industrial-supplier-diagnosis-agent",
        min_length=1,
        max_length=128,
    )
    supplier_a2a_agent_version: str = Field(
        default="1.0.0",
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.+-]*$",
    )
    supplier_a2a_skill_id: str = Field(
        default="industrial.vendor.diagnosis-review",
        min_length=1,
        max_length=128,
    )
    supplier_a2a_remote_tenant: str = Field(
        default="industrial-ops",
        min_length=1,
        max_length=128,
    )
    supplier_a2a_token_url: str = (
        "http://keycloak:8080/realms/industrial-ops/protocol/openid-connect/token"
    )
    supplier_a2a_client_id: str = Field(
        default="industrial-ops-a2a-client",
        min_length=1,
        max_length=128,
    )
    supplier_a2a_resource: str = "http://supplier-agent:8092"
    supplier_a2a_scope: str = Field(
        default="supplier.diagnosis.review",
        min_length=1,
        max_length=128,
    )
    supplier_a2a_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    graph_rag_enabled: bool = False
    multi_agent_planning_enabled: bool = False
    maintenance_planning_model_alias: str = Field(
        default="industrial-diagnosis",
        min_length=1,
        max_length=128,
    )
    maintenance_planning_timeout_seconds: float = Field(default=90.0, gt=0, le=180)
    neo4j_uri: str = "neo4j://neo4j:7687"
    neo4j_database: str = Field(default="neo4j", min_length=1, max_length=63)
    neo4j_username: str = Field(default="neo4j", min_length=1, max_length=128)
    neo4j_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    knowledge_search_profiles_enabled: bool = False
    opensearch_url: str = "http://opensearch:9200"
    opensearch_username: str | None = Field(default=None, min_length=1, max_length=128)
    opensearch_index_prefix: str = Field(
        default="ioap-knowledge", pattern=r"^[a-z0-9][a-z0-9-]{2,62}$"
    )
    opensearch_ca_file: str | None = None
    opensearch_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    reranker_inference_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    embedding_inference_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    external_search_enabled: bool = False
    tavily_search_url: str = "https://api.tavily.com/search"
    tavily_project_id: str | None = Field(default=None, min_length=1, max_length=128)
    tavily_timeout_seconds: float = Field(default=10.0, gt=0, le=30)
    tts_model_alias: str = "industrial-diagnosis"
    tts_timeout_seconds: float = Field(default=30.0, gt=0, le=60)
    tts_voice: str = Field(default="default", min_length=1, max_length=64)
    realtime_media_enabled: bool = False
    realtime_signaling_url: str = "ws://localhost:7880/realtime"
    realtime_stun_urls: tuple[str, ...] = ()
    realtime_turn_url: str | None = None
    realtime_token_ttl_seconds: int = Field(default=60, ge=15, le=300)
    realtime_lease_seconds: int = Field(default=90, ge=30, le=300)
    realtime_max_session_seconds: int = Field(default=3_600, ge=60, le=14_400)
    realtime_heartbeat_interval_seconds: int = Field(default=20, ge=5, le=60)
    realtime_asr_model_alias: str = "industrial-diagnosis"
    realtime_asr_timeout_seconds: float = Field(default=20.0, gt=0, le=60)
    realtime_asr_hotwords: tuple[str, ...] = (
        "报警码",
        "设备号",
        "序列号",
        "部件号",
        "停机",
        "断电",
        "泄压",
        "上锁挂牌",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_model_runtime_selection(cls, values: Any) -> Any:
        """Fail explicitly when a removed fixture or transport switch is supplied."""

        if not isinstance(values, Mapping):
            return values
        removed = {
            "model_gateway_enabled",
            "model_gateway_transport",
            "ocr_provider",
            "ollama_base_url",
            "ollama_model",
            "ollama_model_digest",
            "ollama_manifest_hash",
        }
        legacy = sorted(removed.intersection(values))
        legacy.extend(
            sorted(
                f"IOAP_{name.upper()}" for name in removed if f"IOAP_{name.upper()}" in os.environ
            )
        )
        if legacy:
            raise ValueError(
                "legacy model runtime selection is no longer supported: " + ", ".join(legacy)
            )
        return values

    @model_validator(mode="after")
    def enforce_safe_environment(self) -> Settings:
        """Reject unsafe production-only combinations."""

        if self.deployment_controller_leader_election_enabled:
            if not self.deployment_controller_pod_uid:
                raise ValueError("Deployment controller Pod UID is required for leader election")
            if not (
                self.deployment_controller_retry_period_seconds
                < self.deployment_controller_renew_deadline_seconds
                < self.deployment_controller_lease_duration_seconds
            ):
                raise ValueError(
                    "Deployment controller retry period must be less than renew deadline "
                    "and renew deadline must be less than lease duration"
                )
        if self.environment is Environment.PRODUCTION and self.debug:
            raise ValueError("debug mode is forbidden in production")
        if self.environment is Environment.PRODUCTION and not self.vault_address.startswith(
            "https://"
        ):
            raise ValueError("Vault must use HTTPS in production")
        if self.environment is Environment.PRODUCTION and not self.prometheus_url.startswith(
            "https://"
        ):
            raise ValueError("Prometheus must use HTTPS in production")
        if (
            self.environment is Environment.PRODUCTION
            and self.otel_enabled
            and not _is_https_or_cluster_service(self.otel_exporter_otlp_traces_endpoint)
        ):
            raise ValueError("production OTLP export must use HTTPS or an in-cluster Service")
        if self.langfuse_otel_traces_endpoint is not None:
            if not self.otel_enabled:
                raise ValueError("Langfuse OTLP export requires OpenTelemetry")
            if self.langfuse_otel_authorization_file is None:
                raise ValueError("Langfuse OTLP export requires an authorization file")
            if (
                self.environment is Environment.PRODUCTION
                and not self.langfuse_otel_traces_endpoint.startswith("https://")
            ):
                raise ValueError("Langfuse OTLP export must use HTTPS in production")
        expected_alias = (
            "industrial-diagnosis"
            if self.model_gateway_required_environment is ModelTargetEnvironment.PRODUCTION
            else "industrial-diagnosis-staging"
        )
        if self.environment is Environment.PRODUCTION and (
            self.model_gateway_required_environment is not ModelTargetEnvironment.PRODUCTION
        ):
            raise ValueError("production requires the PRODUCTION model target environment")
        if self.model_gateway_alias != expected_alias or self.vlm_model_alias != expected_alias:
            raise ValueError("model aliases must match the required Release environment")
        if (
            self.environment is Environment.PRODUCTION
            and self.enterprise_tools_mode is EnterpriseToolsMode.SYNTHETIC
        ):
            raise ValueError("Synthetic enterprise tools are forbidden in production")
        if (
            self.use_enterprise_http_parts
            and not self.enterprise_tool_gateway_url.startswith("https://")
            and self.environment not in {Environment.LOCAL, Environment.TEST}
        ):
            raise ValueError("Enterprise tool gateway must use HTTPS outside local/test")
        if self.use_http_notification_delivery:
            parsed_notification = urlparse(self.notification_provider_url)
            if (
                parsed_notification.scheme not in {"http", "https"}
                or not parsed_notification.hostname
            ):
                raise ValueError("Notification provider URL must use HTTP(S)")
            if (
                self.environment not in {Environment.LOCAL, Environment.TEST}
                and parsed_notification.scheme != "https"
            ):
                raise ValueError("Notification provider must use HTTPS outside local/test")
        if self.use_http_procurement_delivery:
            parsed_procurement = urlparse(self.procurement_provider_url)
            if (
                parsed_procurement.scheme not in {"http", "https"}
                or not parsed_procurement.hostname
            ):
                raise ValueError("Procurement provider URL must use HTTP(S)")
            if (
                self.environment not in {Environment.LOCAL, Environment.TEST}
                and parsed_procurement.scheme != "https"
            ):
                raise ValueError("Procurement provider must use HTTPS outside local/test")
        if self.use_http_refund_delivery:
            parsed_refund = urlparse(self.refund_provider_url)
            if parsed_refund.scheme not in {"http", "https"} or not parsed_refund.hostname:
                raise ValueError("Refund provider URL must use HTTP(S)")
            if (
                self.environment not in {Environment.LOCAL, Environment.TEST}
                and parsed_refund.scheme != "https"
            ):
                raise ValueError("Refund provider must use HTTPS outside local/test")
        if self.use_http_fsm_assignment_delivery:
            parsed_fsm = urlparse(self.fsm_assignment_provider_url)
            if parsed_fsm.scheme not in {"http", "https"} or not parsed_fsm.hostname:
                raise ValueError("FSM assignment provider URL must use HTTP(S)")
            if (
                self.environment not in {Environment.LOCAL, Environment.TEST}
                and parsed_fsm.scheme != "https"
            ):
                raise ValueError("FSM assignment provider must use HTTPS outside local/test")
        if self.use_http_service_quotation_catalog:
            parsed_cpq = urlparse(self.service_quotation_provider_url)
            if parsed_cpq.scheme not in {"http", "https"} or not parsed_cpq.hostname:
                raise ValueError("Service quotation provider URL must use HTTP(S)")
            if (
                self.environment not in {Environment.LOCAL, Environment.TEST}
                and parsed_cpq.scheme != "https"
            ):
                raise ValueError("Service quotation provider must use HTTPS outside local/test")
        if self.use_mcp_enterprise_tools:
            if len(set(self.enterprise_mcp_allowed_scopes)) != len(
                self.enterprise_mcp_allowed_scopes
            ) or any(not scope.strip() for scope in self.enterprise_mcp_allowed_scopes):
                raise ValueError("MCP allowed scopes must be unique and non-empty")
            if self.enterprise_mcp_resource.rstrip("/") != self.enterprise_mcp_server_url.rstrip(
                "/"
            ):
                raise ValueError("MCP resource indicator must identify the configured server")
            if self.environment not in {Environment.LOCAL, Environment.TEST}:
                if not self.enterprise_mcp_server_url.startswith("https://"):
                    raise ValueError("MCP server must use HTTPS outside local/test")
                if not self.enterprise_mcp_token_url.startswith("https://"):
                    raise ValueError("MCP token endpoint must use HTTPS outside local/test")
                if not self.enterprise_mcp_resource.startswith("https://"):
                    raise ValueError("MCP resource indicator must use HTTPS outside local/test")
        if self.supplier_a2a_enabled:
            if self.supplier_a2a_resource.rstrip("/") != self.supplier_a2a_base_url.rstrip("/"):
                raise ValueError("A2A resource indicator must identify the configured agent")
            if any(character.isspace() for character in self.supplier_a2a_scope):
                raise ValueError("A2A configuration must declare exactly one minimum scope")
            if self.environment not in {Environment.LOCAL, Environment.TEST}:
                if not self.supplier_a2a_base_url.startswith("https://"):
                    raise ValueError("A2A agent must use HTTPS outside local/test")
                if not self.supplier_a2a_token_url.startswith("https://"):
                    raise ValueError("A2A token endpoint must use HTTPS outside local/test")
                if not self.supplier_a2a_resource.startswith("https://"):
                    raise ValueError("A2A resource indicator must use HTTPS outside local/test")
        if self.graph_rag_enabled:
            if not self.neo4j_uri.startswith(("neo4j://", "neo4j+s://", "neo4j+ssc://")):
                raise ValueError("Neo4j must use a routing URI")
            if self.environment not in {
                Environment.LOCAL,
                Environment.TEST,
            } and not self.neo4j_uri.startswith("neo4j+s://"):
                raise ValueError("Neo4j must use verified TLS outside local/test")
        if self.knowledge_search_profiles_enabled:
            if not self.opensearch_url.startswith(("http://", "https://")):
                raise ValueError("OpenSearch URL must use HTTP(S)")
            if self.environment not in {
                Environment.LOCAL,
                Environment.TEST,
            } and not self.opensearch_url.startswith("https://"):
                raise ValueError("OpenSearch must use verified TLS outside local/test")
            if (
                self.environment
                not in {
                    Environment.LOCAL,
                    Environment.TEST,
                }
                and self.opensearch_username is None
            ):
                raise ValueError("OpenSearch authentication is required outside local/test")
        if self.external_search_enabled:
            parsed_tavily = urlparse(self.tavily_search_url)
            if parsed_tavily.scheme not in {"http", "https"} or not parsed_tavily.hostname:
                raise ValueError("Tavily Search URL must use HTTP(S)")
            if (
                self.environment
                not in {
                    Environment.LOCAL,
                    Environment.TEST,
                }
                and parsed_tavily.scheme != "https"
            ):
                raise ValueError("Tavily Search must use HTTPS outside local/test")
        if (
            self.environment is Environment.PRODUCTION
            and self.realtime_media_enabled
            and not self.realtime_signaling_url.startswith("wss://")
        ):
            raise ValueError("WebRTC signaling must use WSS in production")
        if self.realtime_heartbeat_interval_seconds * 2 >= self.realtime_lease_seconds:
            raise ValueError("WebRTC lease must exceed two heartbeat intervals")
        if not self.kubernetes_api_url.startswith("https://"):
            raise ValueError("Kubernetes API must use HTTPS")
        return self

    @property
    def expose_api_docs(self) -> bool:
        """Interactive docs are disabled by default in production."""

        return self.environment is not Environment.PRODUCTION

    @property
    def use_paddle_ocr(self) -> bool:
        return True

    @property
    def use_http_enterprise_tools(self) -> bool:
        return self.enterprise_tools_mode is EnterpriseToolsMode.HTTP or (
            self.enterprise_tools_mode is EnterpriseToolsMode.AUTO
            and self.environment in {Environment.STAGING, Environment.PRODUCTION}
        )

    @property
    def use_mcp_enterprise_tools(self) -> bool:
        return self.enterprise_tools_mode is EnterpriseToolsMode.MCP

    @property
    def use_enterprise_http_parts(self) -> bool:
        return self.use_http_enterprise_tools or self.use_mcp_enterprise_tools

    @property
    def use_http_notification_delivery(self) -> bool:
        return self.notification_delivery_mode is NotificationDeliveryMode.HTTP

    @property
    def use_http_procurement_delivery(self) -> bool:
        return self.procurement_delivery_mode is ProcurementDeliveryMode.HTTP

    @property
    def use_http_refund_delivery(self) -> bool:
        return self.refund_delivery_mode is RefundDeliveryMode.HTTP

    @property
    def use_http_fsm_assignment_delivery(self) -> bool:
        return self.fsm_assignment_delivery_mode is FsmAssignmentDeliveryMode.HTTP

    @property
    def use_http_service_quotation_catalog(self) -> bool:
        return self.service_quotation_catalog_mode is ServiceQuotationCatalogMode.HTTP


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable settings object per process."""

    return Settings()


def _is_https_or_cluster_service(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    return parsed.scheme == "https" or (
        parsed.scheme == "http"
        and parsed.hostname is not None
        and (parsed.hostname.endswith(".svc") or parsed.hostname.endswith(".svc.cluster.local"))
    )
