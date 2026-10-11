"""M2 Temporal worker for durable recognition and diagnosis orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any

from minio import Minio
from prometheus_client import start_http_server
from temporalio.client import Client
from temporalio.contrib.opentelemetry import OpenTelemetryInterceptor
from temporalio.worker import Worker

from industrial_ops_agent.application.incidents import RecognitionService
from industrial_ops_agent.auth.opa import OpaPolicyClient
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.cache.redis import RedisTenantCache
from industrial_ops_agent.config import Settings, get_settings
from industrial_ops_agent.graph_rag.extraction import GraphExtractionActivity
from industrial_ops_agent.graph_rag.store import Neo4jGraphStore
from industrial_ops_agent.knowledge.deletion import KnowledgeDeletionActivity
from industrial_ops_agent.knowledge.document_parser import (
    DoclingStructuredPdfParser,
    GovernedKnowledgeDocumentParser,
)
from industrial_ops_agent.knowledge.files import KnowledgeIngestionActivity
from industrial_ops_agent.knowledge.index_evaluation import KnowledgeIndexEvaluationActivity
from industrial_ops_agent.maintenance_planning import MaintenancePlanningActivity
from industrial_ops_agent.media.worker import ClamAvScanner
from industrial_ops_agent.model_gateway.openai import OpenAiCompatibleTransport
from industrial_ops_agent.model_gateway.service import (
    EnvironmentAwareModelResolver,
    ModelGateway,
    ModelResolver,
)
from industrial_ops_agent.multimodal.processor import MultimodalProcessor
from industrial_ops_agent.multimodal.providers import (
    GatewayVlmProvider,
    PaddleOcrProvider,
    ReleaseBoundOcrProvider,
    ZxingQrCodeProvider,
)
from industrial_ops_agent.multimodal.transcription import (
    OpenAiCompatibleTranscriptionTransport,
    TranscriptionGateway,
)
from industrial_ops_agent.multimodal.video import (
    DevelopmentVideoAsrProvider,
    GatewayVideoAsrProvider,
    GatewayVideoTemporalProvider,
    PyAvVideoDecomposer,
    VideoAsrProvider,
    VideoTemporalProvider,
)
from industrial_ops_agent.object_store.minio import MinioTenantObjectStore
from industrial_ops_agent.observability import configure_logging
from industrial_ops_agent.orchestration.activities import RecognitionActivity
from industrial_ops_agent.orchestration.workflows import (
    DiagnosisActivity,
    DiagnosisWorkflow,
    KnowledgeDeletionWorkflow,
    KnowledgeGraphExtractionWorkflow,
    KnowledgeIndexEvaluationWorkflow,
    KnowledgeIngestionWorkflow,
    KnowledgeSearchRebuildWorkflow,
    MaintenancePlanningWorkflow,
    RecognitionWorkflow,
    TemporalKnowledgeDeletionActivity,
    TemporalKnowledgeGraphExtractionActivity,
    TemporalKnowledgeIndexEvaluationActivity,
    TemporalKnowledgeIngestionActivity,
    TemporalKnowledgeSearchRebuildActivity,
    TemporalMaintenancePlanningActivity,
    TemporalRecognitionActivity,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.runtime_metrics import InferenceMetrics, KnowledgeEmbeddingOnlineMetrics
from industrial_ops_agent.search_profiles.embedding import (
    HttpKnowledgeEmbedder,
    ProductionEmbeddingResolver,
)
from industrial_ops_agent.search_profiles.rebuild import SearchProfileRebuildActivity
from industrial_ops_agent.search_profiles.service import KnowledgeSearchProfileService
from industrial_ops_agent.search_profiles.store import OpenSearchStore
from industrial_ops_agent.secrets import SecretName, SecretProvider, build_secret_provider
from industrial_ops_agent.security_audit import SecurityAuditor, build_persistent_security_auditor
from industrial_ops_agent.telemetry import configure_telemetry
from industrial_ops_agent.tools.factory import build_enterprise_adapters
from industrial_ops_agent.tools.gateway import ToolGateway
from industrial_ops_agent.tools.registry import default_tool_registry


def _build_diagnosis_model_runtime(
    settings: Settings,
    database: Database,
    provider: SecretProvider,
    auditor: SecurityAuditor,
    metrics: InferenceMetrics,
) -> tuple[ModelGateway, ModelResolver, str]:
    model_resolver = EnvironmentAwareModelResolver(
        database, settings.model_gateway_required_environment
    )
    api_key = provider.get(SecretName.MODEL_GATEWAY_API_KEY).reveal()
    return (
        ModelGateway(
            database,
            model_resolver,
            OpenAiCompatibleTransport(
                api_key,
                allow_plain_http=settings.environment.value in {"local", "test"},
            ),
            security_auditor=auditor,
            metrics=metrics,
        ),
        model_resolver,
        api_key,
    )


async def run() -> None:
    settings = get_settings()
    configure_logging(
        settings.log_level,
        service_name=settings.service_name,
        log_file=settings.log_file,
    )
    telemetry = configure_telemetry(settings)
    provider = build_secret_provider(settings)
    database_url = provider.get(SecretName.DATABASE_URL)
    minio_access_key = provider.get(SecretName.MINIO_ACCESS_KEY)
    minio_secret_key = provider.get(SecretName.MINIO_SECRET_KEY)
    audit_secret = provider.get(SecretName.OIDC_CLIENT_SECRET)
    database = Database(database_url.reveal())
    minio_client = Minio(
        settings.minio_endpoint,
        access_key=minio_access_key.reveal(),
        secret_key=minio_secret_key.reveal(),
        secure=settings.minio_secure,
    )
    object_store = MinioTenantObjectStore(minio_client, settings.minio_bucket)
    cache = RedisTenantCache.from_url(settings.redis_url)
    audit_key = sha256(b"industrial-ops-audit-v1\0" + audit_secret.reveal().encode()).digest()
    auditor = build_persistent_security_auditor(database, hash_key=audit_key)
    authorizer = Authorizer(
        auditor,
        OpaPolicyClient(settings.opa_decision_url, timeout_seconds=settings.opa_timeout_seconds),
    )
    tool_registry = default_tool_registry()
    enterprise_adapters = build_enterprise_adapters(settings, provider, tool_registry)
    tool_gateway = ToolGateway(
        database,
        authorizer,
        tool_registry,
        enterprise_adapters.parts,
        readonly=enterprise_adapters.readonly,
    )
    inference_metrics = InferenceMetrics()
    metrics_server, _metrics_thread = start_http_server(
        settings.workflow_worker_metrics_port,
        addr=settings.worker_metrics_host,
        registry=inference_metrics.registry,
    )
    video_asr: VideoAsrProvider = DevelopmentVideoAsrProvider()
    model_gateway, resolver, model_gateway_key = _build_diagnosis_model_runtime(
        settings,
        database,
        provider,
        auditor,
        inference_metrics,
    )
    video_asr = GatewayVideoAsrProvider(
        TranscriptionGateway(
            database,
            resolver,
            OpenAiCompatibleTranscriptionTransport(
                model_gateway_key,
                allow_plain_http=settings.environment.value in {"local", "test"},
            ),
        ),
        model_alias=settings.realtime_asr_model_alias,
        timeout_seconds=settings.realtime_asr_timeout_seconds,
        hotwords=settings.realtime_asr_hotwords,
    )
    vlm = GatewayVlmProvider(
        model_gateway,
        resolver,
        database=database,
        model_alias=settings.vlm_model_alias,
        timeout_seconds=settings.video_vlm_timeout_seconds,
    )
    video_temporal: VideoTemporalProvider | None = GatewayVideoTemporalProvider(
        model_gateway,
        resolver,
        model_alias=settings.video_vlm_model_alias,
        timeout_seconds=settings.video_vlm_timeout_seconds,
    )
    ocr = ReleaseBoundOcrProvider(
        PaddleOcrProvider(
            language=settings.paddle_ocr_language,
            version=settings.paddle_ocr_component_id,
        ),
        resolver,
        model_alias=settings.model_gateway_alias,
        component_id=settings.paddle_ocr_component_id,
    )
    recognition = TemporalRecognitionActivity(
        RecognitionActivity(
            RecognitionService(database, authorizer),
            object_store,
            MultimodalProcessor(
                ocr,
                vlm,
                qr=ZxingQrCodeProvider(),
                video=PyAvVideoDecomposer(),
                video_asr=video_asr,
                video_temporal=video_temporal,
            ),
        )
    )
    knowledge_ingestion = TemporalKnowledgeIngestionActivity(
        KnowledgeIngestionActivity(
            database,
            authorizer,
            object_store,
            ClamAvScanner(settings.clamav_host, settings.clamav_port),
            GovernedKnowledgeDocumentParser(
                ocr if settings.use_paddle_ocr else None,
                DoclingStructuredPdfParser(settings.docling_artifacts_path),
                vlm,
            ),
            bucket=settings.minio_bucket,
        )
    )
    knowledge_graph_store = None
    if settings.graph_rag_enabled:
        neo4j_password = provider.get(SecretName.NEO4J_PASSWORD)
        knowledge_graph_store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=neo4j_password.reveal(),
            database=settings.neo4j_database,
            connection_timeout_seconds=settings.neo4j_timeout_seconds,
        )
        if not await asyncio.to_thread(knowledge_graph_store.ping):
            raise RuntimeError("Neo4j is required when GraphRAG is enabled")
    knowledge_search_store = None
    knowledge_embedder = None
    if settings.knowledge_search_profiles_enabled:
        opensearch_password = (
            provider.get(SecretName.OPENSEARCH_PASSWORD).reveal()
            if settings.opensearch_username is not None
            else None
        )
        knowledge_search_store = OpenSearchStore(
            url=settings.opensearch_url,
            index_prefix=settings.opensearch_index_prefix,
            username=settings.opensearch_username,
            password=opensearch_password,
            ca_certs=settings.opensearch_ca_file,
            timeout_seconds=settings.opensearch_timeout_seconds,
        )
        if not await asyncio.to_thread(knowledge_search_store.ping):
            raise RuntimeError("OpenSearch is required when search profiles are enabled")
        knowledge_embedder = HttpKnowledgeEmbedder(
            ProductionEmbeddingResolver(database),
            api_key=provider.get(SecretName.MODEL_GATEWAY_API_KEY).reveal(),
            timeout_seconds=settings.embedding_inference_timeout_seconds,
            allow_plain_http=settings.environment.value in {"local", "test"},
            metrics=KnowledgeEmbeddingOnlineMetrics(inference_metrics.registry),
        )
    knowledge_deletion = TemporalKnowledgeDeletionActivity(
        KnowledgeDeletionActivity(
            database,
            object_store,
            cache,
            knowledge_graph_store,
            knowledge_search_store,
        )
    )
    knowledge_index_evaluation = TemporalKnowledgeIndexEvaluationActivity(
        KnowledgeIndexEvaluationActivity(database)
    )
    knowledge_search_rebuild = (
        TemporalKnowledgeSearchRebuildActivity(
            SearchProfileRebuildActivity(
                database,
                KnowledgeSearchProfileService(
                    database,
                    authorizer,
                    knowledge_search_store,
                    index_prefix=settings.opensearch_index_prefix,
                    embedder=knowledge_embedder,
                ),
            )
        )
        if knowledge_search_store is not None
        else None
    )
    knowledge_graph_extraction = (
        TemporalKnowledgeGraphExtractionActivity(
            GraphExtractionActivity(
                database,
                model_gateway,
                timeout_seconds=settings.graph_extraction_timeout_seconds,
            )
        )
        if settings.graph_rag_enabled
        else None
    )
    maintenance_planning = (
        TemporalMaintenancePlanningActivity(
            MaintenancePlanningActivity(
                database,
                model_gateway,
                timeout_seconds=settings.maintenance_planning_timeout_seconds,
            )
        )
        if settings.multi_agent_planning_enabled
        else None
    )
    client = await Client.connect(
        settings.temporal_address,
        interceptors=(OpenTelemetryInterceptor(add_temporal_spans=True),)
        if settings.otel_enabled
        else (),
    )
    ready_file = Path(settings.worker_ready_file)
    ready_file.touch(mode=0o600)
    registered_activities: list[Callable[..., Any]] = [
        recognition.execute,
        knowledge_ingestion.execute,
        knowledge_deletion.execute,
        knowledge_index_evaluation.execute,
        DiagnosisActivity(
            database,
            authorizer,
            tool_gateway=tool_gateway,
            model_gateway=model_gateway,
            model_alias=settings.model_gateway_alias,
            timeout_seconds=settings.model_gateway_timeout_seconds,
        ).execute,
    ]
    if knowledge_search_rebuild is not None:
        registered_activities.append(knowledge_search_rebuild.execute)
    registered_workflows = [
        RecognitionWorkflow,
        DiagnosisWorkflow,
        KnowledgeIngestionWorkflow,
        KnowledgeDeletionWorkflow,
        KnowledgeIndexEvaluationWorkflow,
        KnowledgeSearchRebuildWorkflow,
    ]
    if knowledge_graph_extraction is not None:
        registered_activities.append(knowledge_graph_extraction.execute)
        registered_workflows.append(KnowledgeGraphExtractionWorkflow)
    if maintenance_planning is not None:
        registered_activities.append(maintenance_planning.execute)
        registered_workflows.append(MaintenancePlanningWorkflow)
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=registered_workflows,
        activities=registered_activities,
    )
    try:
        await worker.run()
    finally:
        ready_file.unlink(missing_ok=True)
        if knowledge_embedder is not None:
            knowledge_embedder.close()
        await cache.close()
        metrics_server.shutdown()
        metrics_server.server_close()
        if knowledge_graph_store is not None:
            knowledge_graph_store.close()
        if knowledge_search_store is not None:
            knowledge_search_store.close()
        database.dispose()
        telemetry.shutdown()


def main() -> None:
    with suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    main()
