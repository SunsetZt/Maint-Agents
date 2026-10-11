"""Runtime dependency assembly kept outside the import-safe OpenAPI application."""

from __future__ import annotations

import asyncio
import socket
from base64 import b64decode
from binascii import Error as Base64Error
from hashlib import sha256
from urllib.request import urlopen

from fastapi import FastAPI
from minio import Minio
from sqlalchemy import text
from temporalio.contrib.opentelemetry import OpenTelemetryInterceptor

from industrial_ops_agent import __version__
from industrial_ops_agent.api.app import create_app
from industrial_ops_agent.assurance.service import AssuranceService
from industrial_ops_agent.auth.oidc import JwksSigningKeyProvider, OidcVerifier
from industrial_ops_agent.auth.opa import OpaPolicyClient
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.cache.redis import RedisTenantCache
from industrial_ops_agent.collaboration.a2a import (
    A2AClientCredentialsTokenProvider,
    OfficialA2ASupplierAgentClient,
)
from industrial_ops_agent.config import Settings, get_settings
from industrial_ops_agent.data_governance.dlp import PresidioDlpProcessor
from industrial_ops_agent.data_pipeline.lineage import OpenLineageEmitter
from industrial_ops_agent.data_pipeline.service import DatasetPipelineService
from industrial_ops_agent.data_pipeline.storage import MinioDatasetStore
from industrial_ops_agent.edge.signing import Ed25519PackSigner
from industrial_ops_agent.enterprise_assets import EnterpriseProjectAdoptionService
from industrial_ops_agent.experiments.mlflow import MlflowRestTracker
from industrial_ops_agent.external_search.provider import TavilyExternalSearchProvider
from industrial_ops_agent.gpu_operations import GpuOperationsService
from industrial_ops_agent.graph_rag.extraction import (
    ProductionGraphExtractionBindingResolver,
)
from industrial_ops_agent.graph_rag.store import Neo4jGraphStore
from industrial_ops_agent.labeling.label_studio import LazyLabelStudioAdapter
from industrial_ops_agent.maintenance_planning import (
    GovernedMaintenancePlanningBindingResolver,
    MaintenancePlanningActivationService,
    ProductionMaintenancePlanningBindingResolver,
)
from industrial_ops_agent.model_gateway.service import EnvironmentAwareModelResolver
from industrial_ops_agent.multimodal.realtime import RealtimeSessionService
from industrial_ops_agent.multimodal.realtime_transcripts import RealtimeTranscriptService
from industrial_ops_agent.multimodal.speech import (
    OpenAiCompatibleSpeechTransport,
    SpeechSynthesisGateway,
)
from industrial_ops_agent.multimodal.transcription import (
    OpenAiCompatibleTranscriptionTransport,
    TranscriptionGateway,
)
from industrial_ops_agent.object_store.minio import MinioTenantObjectStore
from industrial_ops_agent.observability import HttpMetrics
from industrial_ops_agent.operations import OperationsService, PrometheusHttpReader
from industrial_ops_agent.orchestration.workflows import (
    LazyTemporalClient,
    LazyTemporalDiagnosisDispatcher,
    LazyTemporalKnowledgeDeletionDispatcher,
    LazyTemporalKnowledgeGraphExtractionDispatcher,
    LazyTemporalKnowledgeIndexEvaluationDispatcher,
    LazyTemporalKnowledgeIngestionDispatcher,
    LazyTemporalKnowledgeSearchRebuildDispatcher,
    LazyTemporalMaintenancePlanningDispatcher,
    LazyTemporalRecognitionDispatcher,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.predictive_maintenance.dataset_pipeline import (
    TelemetryDatasetPipelineService,
)
from industrial_ops_agent.predictive_maintenance.inference import (
    HttpTimeseriesDetector,
    ProductionTimeseriesResolver,
)
from industrial_ops_agent.predictive_maintenance.rul_dataset_pipeline import (
    RulDatasetPipelineService,
)
from industrial_ops_agent.predictive_maintenance.rul_inference import (
    HttpRulForecaster,
    ProductionRulResolver,
)
from industrial_ops_agent.predictive_maintenance.service import PredictiveMaintenanceService
from industrial_ops_agent.recovery.service import RecoveryService
from industrial_ops_agent.runtime_metrics import (
    KnowledgeEmbeddingOnlineMetrics,
    KnowledgeRerankerOnlineMetrics,
    PredictiveMaintenanceOnlineMetrics,
)
from industrial_ops_agent.search_profiles.embedding import (
    HttpKnowledgeEmbedder,
    ProductionEmbeddingResolver,
)
from industrial_ops_agent.search_profiles.reranker import (
    HttpKnowledgeReranker,
    ProductionRerankerResolver,
)
from industrial_ops_agent.search_profiles.store import OpenSearchStore
from industrial_ops_agent.secrets import SecretName, build_secret_provider
from industrial_ops_agent.security.emergency_access import DatabaseEmergencyGrantPolicy
from industrial_ops_agent.security_audit import build_persistent_security_auditor
from industrial_ops_agent.telemetry import configure_telemetry
from industrial_ops_agent.tools.factory import build_enterprise_adapters
from industrial_ops_agent.tools.gateway import ToolGateway
from industrial_ops_agent.tools.registry import default_tool_registry


def create_runtime_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    enterprise_project_adoption_service = (
        EnterpriseProjectAdoptionService(
            resolved.enterprise_project_repo_root,
            acceptance_path=resolved.enterprise_project_adoption_path,
            cache_seconds=resolved.enterprise_project_adoption_cache_seconds,
        )
        if resolved.enterprise_project_adoption_enabled
        else None
    )
    telemetry = configure_telemetry(resolved)
    provider = build_secret_provider(resolved)
    database_url = provider.get(SecretName.DATABASE_URL)
    minio_access_key = provider.get(SecretName.MINIO_ACCESS_KEY)
    minio_secret_key = provider.get(SecretName.MINIO_SECRET_KEY)
    audit_hash_key = provider.get(SecretName.OIDC_CLIENT_SECRET)
    edge_pack_signer = None
    if resolved.edge_diagnosis_enabled:
        encoded_edge_key = provider.get(SecretName.EDGE_PACK_SIGNING_PRIVATE_KEY).reveal()
        try:
            edge_key_pem = b64decode(encoded_edge_key, validate=True).decode("utf-8")
        except (Base64Error, UnicodeDecodeError) as exc:
            raise ValueError("FIELD edge signing key must be base64-encoded PEM") from exc
        edge_pack_signer = Ed25519PackSigner.from_private_pem(
            edge_key_pem,
            key_id=resolved.edge_pack_signing_key_id,
        )
    database = Database(database_url.reveal())
    cache = RedisTenantCache.from_url(resolved.redis_url)
    minio_client = Minio(
        resolved.minio_endpoint,
        access_key=minio_access_key.reveal(),
        secret_key=minio_secret_key.reveal(),
        secure=resolved.minio_secure,
    )
    object_store = MinioTenantObjectStore(minio_client, resolved.minio_bucket)
    dataset_store = MinioDatasetStore(minio_client, resolved.dataset_bucket)
    audit_key = sha256(b"industrial-ops-audit-v1\0" + audit_hash_key.reveal().encode()).digest()
    auditor = build_persistent_security_auditor(database, hash_key=audit_key)
    authorizer = Authorizer(
        auditor,
        OpaPolicyClient(
            resolved.opa_decision_url,
            timeout_seconds=resolved.opa_timeout_seconds,
        ),
        DatabaseEmergencyGrantPolicy(database),
    )
    tool_registry = default_tool_registry()
    enterprise_adapters = build_enterprise_adapters(resolved, provider, tool_registry)
    tool_gateway = ToolGateway(
        database,
        authorizer,
        tool_registry,
        enterprise_adapters.parts,
        readonly=enterprise_adapters.readonly,
        notifications=enterprise_adapters.notifications,
        procurement=enterprise_adapters.procurement,
        refunds=enterprise_adapters.refunds,
        fsm_assignments=enterprise_adapters.fsm_assignments,
        quotations=enterprise_adapters.quotations,
    )
    verifier = OidcVerifier(
        issuer=resolved.oidc_issuer,
        audience=resolved.oidc_audience,
        signing_keys=JwksSigningKeyProvider(resolved.oidc_jwks_url),
    )
    temporal = LazyTemporalClient(
        resolved.temporal_address,
        interceptors=(OpenTelemetryInterceptor(add_temporal_spans=True),)
        if resolved.otel_enabled
        else (),
    )
    recognition_dispatcher = LazyTemporalRecognitionDispatcher(
        temporal,
        resolved.temporal_task_queue,
    )
    knowledge_ingestion_dispatcher = LazyTemporalKnowledgeIngestionDispatcher(
        temporal,
        resolved.temporal_task_queue,
    )
    knowledge_deletion_dispatcher = LazyTemporalKnowledgeDeletionDispatcher(
        temporal,
        resolved.temporal_task_queue,
    )
    knowledge_index_evaluation_dispatcher = LazyTemporalKnowledgeIndexEvaluationDispatcher(
        temporal,
        resolved.temporal_task_queue,
    )
    knowledge_search_rebuild_dispatcher = LazyTemporalKnowledgeSearchRebuildDispatcher(
        temporal,
        resolved.temporal_task_queue,
    )
    graph_extraction_dispatcher = LazyTemporalKnowledgeGraphExtractionDispatcher(
        temporal,
        resolved.temporal_task_queue,
    )
    maintenance_planning_dispatcher = LazyTemporalMaintenancePlanningDispatcher(
        temporal,
        resolved.temporal_task_queue,
    )
    diagnosis_dispatcher = LazyTemporalDiagnosisDispatcher(
        temporal,
        resolved.temporal_task_queue,
    )
    dlp_processor = PresidioDlpProcessor()
    label_studio_adapter = LazyLabelStudioAdapter(
        base_url=resolved.label_studio_url,
        secret_provider=provider,
        project_key=resolved.label_studio_project_key,
        project_id=resolved.label_studio_project_id,
        timeout_seconds=resolved.label_studio_timeout_seconds,
    )
    dataset_pipeline = DatasetPipelineService(
        database,
        dataset_store,
        OpenLineageEmitter(
            url=resolved.openlineage_url,
            namespace=resolved.openlineage_namespace,
        ),
        code_version=__version__,
        media_store=MinioDatasetStore(minio_client, resolved.minio_bucket),
    )
    telemetry_dataset_pipeline = TelemetryDatasetPipelineService(
        database,
        dataset_store,
        OpenLineageEmitter(
            url=resolved.openlineage_url,
            namespace=resolved.openlineage_namespace,
            job_name="build_telemetry_dataset_snapshot",
        ),
        code_version=__version__,
    )
    rul_dataset_pipeline = RulDatasetPipelineService(
        database,
        dataset_store,
        OpenLineageEmitter(
            url=resolved.openlineage_url,
            namespace=resolved.openlineage_namespace,
            job_name="build_rul_dataset_snapshot",
        ),
        code_version=__version__,
    )
    experiment_tracker = MlflowRestTracker(
        resolved.mlflow_tracking_url,
        timeout_seconds=resolved.mlflow_timeout_seconds,
    )
    model_resolver = EnvironmentAwareModelResolver(
        database, resolved.model_gateway_required_environment
    )
    graph_extraction_binding_resolver = (
        ProductionGraphExtractionBindingResolver(
            database,
            model_resolver,
            model_alias=resolved.graph_extraction_model_alias,
        )
        if resolved.graph_rag_enabled
        else None
    )
    maintenance_planning_base_resolver = (
        ProductionMaintenancePlanningBindingResolver(
            database,
            model_resolver,
            model_alias=resolved.maintenance_planning_model_alias,
        )
        if model_resolver is not None and resolved.multi_agent_planning_enabled
        else None
    )
    maintenance_planning_binding_resolver = (
        GovernedMaintenancePlanningBindingResolver(
            maintenance_planning_base_resolver,
            MaintenancePlanningActivationService(database),
            target_environment=(
                "PRODUCTION" if resolved.environment.value == "production" else "PROJECT_STAGING"
            ),
        )
        if maintenance_planning_base_resolver is not None
        else None
    )
    speech_gateway = None
    realtime_transcript_service = None
    timeseries_detector = None
    rul_forecaster = None
    if model_resolver is not None:
        model_gateway_key = provider.get(SecretName.MODEL_GATEWAY_API_KEY)
        timeseries_detector = HttpTimeseriesDetector(
            ProductionTimeseriesResolver(database),
            api_key=model_gateway_key.reveal(),
            timeout_seconds=resolved.timeseries_inference_timeout_seconds,
            allow_plain_http=resolved.environment.value in {"local", "test"},
        )
        rul_forecaster = HttpRulForecaster(
            ProductionRulResolver(database),
            api_key=model_gateway_key.reveal(),
            timeout_seconds=resolved.rul_inference_timeout_seconds,
            allow_plain_http=resolved.environment.value in {"local", "test"},
        )
        speech_gateway = SpeechSynthesisGateway(
            database,
            model_resolver,
            OpenAiCompatibleSpeechTransport(
                model_gateway_key.reveal(),
                allow_plain_http=resolved.environment.value in {"local", "test"},
            ),
            object_store,
            allowed_voices=frozenset({"default", resolved.tts_voice}),
        )
        realtime_transcript_service = RealtimeTranscriptService(
            database,
            authorizer,
            TranscriptionGateway(
                database,
                model_resolver,
                OpenAiCompatibleTranscriptionTransport(
                    model_gateway_key.reveal(),
                    allow_plain_http=resolved.environment.value in {"local", "test"},
                ),
            ),
            model_alias=resolved.realtime_asr_model_alias,
            timeout_seconds=resolved.realtime_asr_timeout_seconds,
            hotwords=resolved.realtime_asr_hotwords,
        )
    realtime_session_service = None
    if resolved.realtime_media_enabled:
        turn_secret = (
            provider.get(SecretName.REALTIME_TURN_SHARED_SECRET).reveal()
            if resolved.realtime_turn_url
            else None
        )
        realtime_session_service = RealtimeSessionService(
            database,
            authorizer,
            signaling_url=resolved.realtime_signaling_url,
            stun_urls=resolved.realtime_stun_urls,
            turn_url=resolved.realtime_turn_url,
            turn_shared_secret=turn_secret,
            token_ttl_seconds=resolved.realtime_token_ttl_seconds,
            lease_seconds=resolved.realtime_lease_seconds,
            max_session_seconds=resolved.realtime_max_session_seconds,
            heartbeat_interval_seconds=resolved.realtime_heartbeat_interval_seconds,
        )
    prometheus_reader = PrometheusHttpReader(
        resolved.prometheus_url,
        timeout_seconds=resolved.prometheus_timeout_seconds,
        ca_file=resolved.prometheus_ca_file,
        token_file=resolved.prometheus_token_file,
    )
    recovery_service = RecoveryService(
        database,
        authorizer,
        enforce_release_gate=resolved.environment.value == "production",
    )
    operations_service = OperationsService(
        prometheus_reader,
        recovery_gate=recovery_service,
        recovery_tenant_id=resolved.deployment_controller_tenant_id,
    )
    gpu_operations_service = GpuOperationsService(database, prometheus_reader)
    assurance_service = AssuranceService(database, authorizer, operations_service)
    http_metrics = HttpMetrics()
    predictive_online_metrics = PredictiveMaintenanceOnlineMetrics(http_metrics.registry)
    predictive_maintenance_service = PredictiveMaintenanceService(
        database,
        authorizer,
        timeseries_detector=timeseries_detector,
        online_metrics=predictive_online_metrics,
        rul_forecaster=rul_forecaster,
    )
    supplier_agent_client = None
    if resolved.supplier_a2a_enabled:
        supplier_a2a_secret = provider.get(SecretName.SUPPLIER_A2A_CLIENT_SECRET)
        if supplier_a2a_secret.reveal() == "disabled-local-a2a-secret":
            raise ValueError("supplier A2A client secret must be configured before enablement")
        supplier_agent_client = OfficialA2ASupplierAgentClient(
            base_url=resolved.supplier_a2a_base_url,
            agent_name=resolved.supplier_a2a_agent_name,
            agent_version=resolved.supplier_a2a_agent_version,
            skill_id=resolved.supplier_a2a_skill_id,
            remote_tenant=resolved.supplier_a2a_remote_tenant,
            timeout_seconds=resolved.supplier_a2a_timeout_seconds,
            token_provider=A2AClientCredentialsTokenProvider(
                token_url=resolved.supplier_a2a_token_url,
                client_id=resolved.supplier_a2a_client_id,
                client_secret=supplier_a2a_secret,
                resource=resolved.supplier_a2a_resource,
                scope=resolved.supplier_a2a_scope,
                timeout_seconds=resolved.supplier_a2a_timeout_seconds,
                allow_plain_http=resolved.environment.value in {"local", "test"},
            ),
            allow_plain_http=resolved.environment.value in {"local", "test"},
        )
    knowledge_graph_store = None
    if resolved.graph_rag_enabled:
        neo4j_password = provider.get(SecretName.NEO4J_PASSWORD)
        if neo4j_password.reveal() == "disabled-local-neo4j-password":
            raise ValueError("Neo4j password must be configured before GraphRAG enablement")
        knowledge_graph_store = Neo4jGraphStore(
            uri=resolved.neo4j_uri,
            username=resolved.neo4j_username,
            password=neo4j_password.reveal(),
            database=resolved.neo4j_database,
            connection_timeout_seconds=resolved.neo4j_timeout_seconds,
        )
    knowledge_search_store = None
    if resolved.knowledge_search_profiles_enabled:
        opensearch_password = (
            provider.get(SecretName.OPENSEARCH_PASSWORD).reveal()
            if resolved.opensearch_username is not None
            else None
        )
        knowledge_search_store = OpenSearchStore(
            url=resolved.opensearch_url,
            index_prefix=resolved.opensearch_index_prefix,
            username=resolved.opensearch_username,
            password=opensearch_password,
            ca_certs=resolved.opensearch_ca_file,
            timeout_seconds=resolved.opensearch_timeout_seconds,
        )
    knowledge_reranker = None
    knowledge_embedder = None
    external_search_provider = None
    if knowledge_search_store is not None and model_resolver is not None:
        model_gateway_key = provider.get(SecretName.MODEL_GATEWAY_API_KEY)
        knowledge_reranker = HttpKnowledgeReranker(
            ProductionRerankerResolver(database),
            api_key=model_gateway_key.reveal(),
            timeout_seconds=resolved.reranker_inference_timeout_seconds,
            allow_plain_http=resolved.environment.value in {"local", "test"},
            metrics=KnowledgeRerankerOnlineMetrics(http_metrics.registry),
        )
        knowledge_embedder = HttpKnowledgeEmbedder(
            ProductionEmbeddingResolver(database),
            api_key=model_gateway_key.reveal(),
            timeout_seconds=resolved.embedding_inference_timeout_seconds,
            allow_plain_http=resolved.environment.value in {"local", "test"},
            metrics=KnowledgeEmbeddingOnlineMetrics(http_metrics.registry),
        )
    if resolved.external_search_enabled:
        external_search_provider = TavilyExternalSearchProvider(
            endpoint=resolved.tavily_search_url,
            api_key=provider.get(SecretName.TAVILY_API_KEY),
            timeout_seconds=resolved.tavily_timeout_seconds,
            project_id=resolved.tavily_project_id,
        )

    async def database_ready() -> bool:
        def probe() -> bool:
            with database.engine.connect() as connection:
                return bool(connection.scalar(text("SELECT 1")) == 1)

        return await asyncio.to_thread(probe)

    async def object_store_ready() -> bool:
        return await asyncio.to_thread(minio_client.bucket_exists, resolved.minio_bucket)

    async def oidc_ready() -> bool:
        def probe() -> bool:
            with urlopen(resolved.oidc_jwks_url, timeout=2.0) as response:  # noqa: S310
                return bool(response.status == 200)

        return await asyncio.to_thread(probe)

    async def temporal_ready() -> bool:
        host, port = resolved.temporal_address.rsplit(":", maxsplit=1)

        def probe() -> bool:
            with socket.create_connection((host, int(port)), timeout=2.0):
                return True

        return await asyncio.to_thread(probe)

    async def graph_ready() -> bool:
        return knowledge_graph_store is not None and await asyncio.to_thread(
            knowledge_graph_store.ping
        )

    async def search_ready() -> bool:
        return knowledge_search_store is not None and await asyncio.to_thread(
            knowledge_search_store.ping
        )

    readiness_checks = {
        "database": database_ready,
        "object_store": object_store_ready,
        "oidc": oidc_ready,
        "redis": cache.ping,
        "temporal": temporal_ready,
    }
    if knowledge_graph_store is not None:
        readiness_checks["neo4j"] = graph_ready
    if knowledge_search_store is not None:
        readiness_checks["opensearch"] = search_ready

    app = create_app(
        resolved,
        readiness_checks=readiness_checks,
        oidc_verifier=verifier,
        security_auditor=auditor,
        database=database,
        authorizer=authorizer,
        object_store=object_store,
        tool_gateway=tool_gateway,
        recognition_dispatcher=recognition_dispatcher,
        knowledge_ingestion_dispatcher=knowledge_ingestion_dispatcher,
        knowledge_deletion_dispatcher=knowledge_deletion_dispatcher,
        knowledge_index_evaluation_dispatcher=knowledge_index_evaluation_dispatcher,
        knowledge_search_rebuild_dispatcher=knowledge_search_rebuild_dispatcher,
        graph_extraction_dispatcher=graph_extraction_dispatcher,
        graph_extraction_binding_resolver=graph_extraction_binding_resolver,
        maintenance_planning_dispatcher=maintenance_planning_dispatcher,
        maintenance_planning_binding_resolver=maintenance_planning_binding_resolver,
        diagnosis_dispatcher=diagnosis_dispatcher,
        dlp_processor=dlp_processor,
        label_studio_adapter=label_studio_adapter,
        dataset_pipeline=dataset_pipeline,
        dataset_store=dataset_store,
        experiment_tracker=experiment_tracker,
        model_resolver=model_resolver,
        speech_gateway=speech_gateway,
        realtime_session_service=realtime_session_service,
        realtime_transcript_service=realtime_transcript_service,
        operations_service=operations_service,
        gpu_operations_service=gpu_operations_service,
        recovery_service=recovery_service,
        assurance_service=assurance_service,
        enterprise_project_adoption_service=enterprise_project_adoption_service,
        predictive_maintenance_service=predictive_maintenance_service,
        telemetry_dataset_pipeline=telemetry_dataset_pipeline,
        rul_dataset_pipeline=rul_dataset_pipeline,
        supplier_agent_client=supplier_agent_client,
        knowledge_graph_store=knowledge_graph_store,
        knowledge_search_store=knowledge_search_store,
        knowledge_reranker=knowledge_reranker,
        knowledge_embedder=knowledge_embedder,
        external_search_provider=external_search_provider,
        edge_pack_signer=edge_pack_signer,
        http_metrics=http_metrics,
    )
    app.state.secret_provider = provider
    app.state.tenant_cache = cache

    async def close_runtime_dependencies() -> None:
        await cache.close()
        if timeseries_detector is not None:
            timeseries_detector.close()
        if rul_forecaster is not None:
            rul_forecaster.close()
        if knowledge_reranker is not None:
            knowledge_reranker.close()
        if knowledge_embedder is not None:
            knowledge_embedder.close()
        prometheus_reader.close()
        if knowledge_graph_store is not None:
            knowledge_graph_store.close()
        if knowledge_search_store is not None:
            knowledge_search_store.close()
        database.dispose()
        telemetry.shutdown()

    app.router.add_event_handler("shutdown", close_runtime_dependencies)
    return app
