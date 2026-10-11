"""FastAPI application factory for the M1 service foundation."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeAlias

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from industrial_ops_agent import __version__
from industrial_ops_agent.api.dependencies import AccessTokenVerifier
from industrial_ops_agent.api.errors import register_error_handlers
from industrial_ops_agent.api.routes.agent_events import router as agent_events_router
from industrial_ops_agent.api.routes.approvals import router as approvals_router
from industrial_ops_agent.api.routes.assets import router as assets_router
from industrial_ops_agent.api.routes.assurance import router as assurance_router
from industrial_ops_agent.api.routes.collaborations import router as collaborations_router
from industrial_ops_agent.api.routes.cost_operations import router as cost_operations_router
from industrial_ops_agent.api.routes.customer_notifications import (
    router as customer_notifications_router,
)
from industrial_ops_agent.api.routes.data_feedback import router as data_feedback_router
from industrial_ops_agent.api.routes.device_families import router as device_families_router
from industrial_ops_agent.api.routes.diagnoses import router as diagnoses_router
from industrial_ops_agent.api.routes.diagnosis_feedback import (
    router as diagnosis_feedback_router,
)
from industrial_ops_agent.api.routes.emergency_access import router as emergency_access_router
from industrial_ops_agent.api.routes.enterprise_assets import router as enterprise_assets_router
from industrial_ops_agent.api.routes.equipment_control_handoffs import (
    router as equipment_control_handoffs_router,
)
from industrial_ops_agent.api.routes.execution_reconciliations import (
    router as execution_reconciliations_router,
)
from industrial_ops_agent.api.routes.experiments import router as experiments_router
from industrial_ops_agent.api.routes.expert_collaborations import (
    router as expert_collaborations_router,
)
from industrial_ops_agent.api.routes.external_search import router as external_search_router
from industrial_ops_agent.api.routes.fsm_assignments import router as fsm_assignments_router
from industrial_ops_agent.api.routes.incident_lifecycle import router as incident_lifecycle_router
from industrial_ops_agent.api.routes.incident_operations import router as incident_operations_router
from industrial_ops_agent.api.routes.incidents import router as incidents_router
from industrial_ops_agent.api.routes.knowledge import router as knowledge_router
from industrial_ops_agent.api.routes.knowledge_graph import router as knowledge_graph_router
from industrial_ops_agent.api.routes.knowledge_search import router as knowledge_search_router
from industrial_ops_agent.api.routes.maintenance_planning import (
    router as maintenance_planning_router,
)
from industrial_ops_agent.api.routes.maintenance_planning_activations import (
    router as maintenance_planning_activation_router,
)
from industrial_ops_agent.api.routes.maintenance_planning_evaluations import (
    evaluation_router as maintenance_planning_evaluation_router,
)
from industrial_ops_agent.api.routes.maintenance_planning_evaluations import (
    suite_router as maintenance_planning_evaluation_suite_router,
)
from industrial_ops_agent.api.routes.me import router as me_router
from industrial_ops_agent.api.routes.memories import router as memories_router
from industrial_ops_agent.api.routes.model_deployments import router as model_deployments_router
from industrial_ops_agent.api.routes.model_gateway import router as model_gateway_router
from industrial_ops_agent.api.routes.model_releases import router as model_releases_router
from industrial_ops_agent.api.routes.operations import router as operations_router
from industrial_ops_agent.api.routes.parts import router as parts_router
from industrial_ops_agent.api.routes.portal import router as portal_router
from industrial_ops_agent.api.routes.portal import staff_router as customer_updates_router
from industrial_ops_agent.api.routes.predictive_maintenance import (
    router as predictive_maintenance_router,
)
from industrial_ops_agent.api.routes.procurement import (
    fulfillment_router as procurement_fulfillment_router,
)
from industrial_ops_agent.api.routes.procurement import router as procurement_router
from industrial_ops_agent.api.routes.prompt_governance import router as prompt_governance_router
from industrial_ops_agent.api.routes.realtime import router as realtime_router
from industrial_ops_agent.api.routes.recognition import router as recognition_router
from industrial_ops_agent.api.routes.recovery import router as recovery_router
from industrial_ops_agent.api.routes.refunds import router as refunds_router
from industrial_ops_agent.api.routes.security import router as security_router
from industrial_ops_agent.api.routes.service_entitlements import (
    router as service_entitlements_router,
)
from industrial_ops_agent.api.routes.service_performance import (
    router as service_performance_router,
)
from industrial_ops_agent.api.routes.speech import router as speech_router
from industrial_ops_agent.api.routes.supply_chain import router as supply_chain_router
from industrial_ops_agent.api.routes.tenant_administration import (
    router as tenant_administration_router,
)
from industrial_ops_agent.api.routes.tool_governance import router as tool_governance_router
from industrial_ops_agent.api.routes.trace_operations import router as trace_operations_router
from industrial_ops_agent.api.routes.uploads import router as uploads_router
from industrial_ops_agent.api.routes.workorders import router as workorders_router
from industrial_ops_agent.application.diagnoses import DiagnosisDispatcher
from industrial_ops_agent.assurance.service import AssuranceService
from industrial_ops_agent.auth.emergency import (
    bind_emergency_grant_id,
    reset_emergency_grant_id,
)
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.collaboration.a2a import SupplierAgentClient
from industrial_ops_agent.config import Settings, get_settings
from industrial_ops_agent.data_governance.dlp import PresidioDlpProcessor
from industrial_ops_agent.data_pipeline.service import DatasetPipelineService
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.edge.signing import Ed25519PackSigner
from industrial_ops_agent.enterprise_assets import EnterpriseProjectAdoptionReader
from industrial_ops_agent.experiments.mlflow import ExperimentTracker
from industrial_ops_agent.external_search.provider import ExternalSearchProvider
from industrial_ops_agent.gpu_operations.service import GpuOperationsService
from industrial_ops_agent.graph_rag.extraction import (
    GraphExtractionBindingResolver,
    GraphExtractionDispatcher,
)
from industrial_ops_agent.graph_rag.store import GraphStore
from industrial_ops_agent.knowledge.deletion import KnowledgeDeletionDispatcher
from industrial_ops_agent.knowledge.files import KnowledgeIngestionDispatcher
from industrial_ops_agent.knowledge.index_evaluation import KnowledgeIndexEvaluationDispatcher
from industrial_ops_agent.labeling.label_studio import LabelStudioAdapter
from industrial_ops_agent.maintenance_planning import (
    MaintenancePlanningBindingResolver,
    MaintenancePlanningDispatcher,
)
from industrial_ops_agent.media.service import TenantObjectStore
from industrial_ops_agent.model_gateway.service import EnvironmentAwareModelResolver
from industrial_ops_agent.multimodal.realtime import RealtimeSessionService
from industrial_ops_agent.multimodal.realtime_transcripts import RealtimeTranscriptService
from industrial_ops_agent.multimodal.speech import SpeechSynthesisGateway
from industrial_ops_agent.observability import HttpMetrics, configure_logging, observe_request
from industrial_ops_agent.operations.service import OperationsService
from industrial_ops_agent.orchestration.activities import RecognitionDispatcher
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.predictive_maintenance.dataset_pipeline import (
    TelemetryDatasetPipelineService,
)
from industrial_ops_agent.predictive_maintenance.rul_dataset_pipeline import (
    RulDatasetPipelineService,
)
from industrial_ops_agent.predictive_maintenance.service import PredictiveMaintenanceService
from industrial_ops_agent.recovery.service import RecoveryService
from industrial_ops_agent.search_profiles.embedding import KnowledgeEmbedder
from industrial_ops_agent.search_profiles.rebuild import SearchProfileRebuildDispatcher
from industrial_ops_agent.search_profiles.reranker import KnowledgeReranker
from industrial_ops_agent.search_profiles.store import SearchStore
from industrial_ops_agent.security_audit import SecurityAuditor
from industrial_ops_agent.tools.gateway import ToolGateway

ReadinessResult: TypeAlias = bool | Awaitable[bool]
ReadinessCheck: TypeAlias = Callable[[], ReadinessResult]


async def _run_readiness_checks(
    checks: Mapping[str, ReadinessCheck],
) -> tuple[bool, dict[str, str]]:
    results: dict[str, str] = {}
    for name in sorted(checks):
        try:
            outcome = checks[name]()
            ready = await outcome if inspect.isawaitable(outcome) else outcome
            results[name] = "ok" if ready else "failed"
        except Exception:  # readiness deliberately classifies without exposing dependency details
            results[name] = "failed"
    return all(value == "ok" for value in results.values()), results


def create_app(
    settings: Settings | None = None,
    *,
    readiness_checks: Mapping[str, ReadinessCheck] | None = None,
    oidc_verifier: AccessTokenVerifier | None = None,
    security_auditor: SecurityAuditor | None = None,
    database: Database | None = None,
    authorizer: Authorizer | None = None,
    object_store: TenantObjectStore | None = None,
    recognition_dispatcher: RecognitionDispatcher | None = None,
    knowledge_ingestion_dispatcher: KnowledgeIngestionDispatcher | None = None,
    knowledge_deletion_dispatcher: KnowledgeDeletionDispatcher | None = None,
    knowledge_index_evaluation_dispatcher: KnowledgeIndexEvaluationDispatcher | None = None,
    knowledge_search_rebuild_dispatcher: SearchProfileRebuildDispatcher | None = None,
    graph_extraction_dispatcher: GraphExtractionDispatcher | None = None,
    graph_extraction_binding_resolver: GraphExtractionBindingResolver | None = None,
    maintenance_planning_dispatcher: MaintenancePlanningDispatcher | None = None,
    maintenance_planning_binding_resolver: MaintenancePlanningBindingResolver | None = None,
    diagnosis_dispatcher: DiagnosisDispatcher | None = None,
    tool_gateway: ToolGateway | None = None,
    dlp_processor: PresidioDlpProcessor | None = None,
    label_studio_adapter: LabelStudioAdapter | None = None,
    dataset_pipeline: DatasetPipelineService | None = None,
    dataset_store: DatasetStore | None = None,
    experiment_tracker: ExperimentTracker | None = None,
    model_resolver: EnvironmentAwareModelResolver | None = None,
    speech_gateway: SpeechSynthesisGateway | None = None,
    realtime_session_service: RealtimeSessionService | None = None,
    realtime_transcript_service: RealtimeTranscriptService | None = None,
    operations_service: OperationsService | None = None,
    gpu_operations_service: GpuOperationsService | None = None,
    recovery_service: RecoveryService | None = None,
    assurance_service: AssuranceService | None = None,
    enterprise_project_adoption_service: EnterpriseProjectAdoptionReader | None = None,
    predictive_maintenance_service: PredictiveMaintenanceService | None = None,
    telemetry_dataset_pipeline: TelemetryDatasetPipelineService | None = None,
    rul_dataset_pipeline: RulDatasetPipelineService | None = None,
    supplier_agent_client: SupplierAgentClient | None = None,
    knowledge_graph_store: GraphStore | None = None,
    knowledge_search_store: SearchStore | None = None,
    knowledge_reranker: KnowledgeReranker | None = None,
    knowledge_embedder: KnowledgeEmbedder | None = None,
    external_search_provider: ExternalSearchProvider | None = None,
    edge_pack_signer: Ed25519PackSigner | None = None,
    http_metrics: HttpMetrics | None = None,
) -> FastAPI:
    """Create an isolated application with explicit runtime dependencies."""

    resolved = settings or get_settings()
    configure_logging(
        resolved.log_level,
        service_name=resolved.service_name,
        log_file=resolved.log_file,
    )
    app = FastAPI(
        title="Industrial Operations Agent Platform API",
        version=__version__,
        debug=resolved.debug,
        docs_url="/docs" if resolved.expose_api_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if resolved.expose_api_docs else None,
    )
    app.state.settings = resolved
    app.state.readiness_checks = dict(readiness_checks or {})
    app.state.http_metrics = http_metrics or HttpMetrics()
    app.state.oidc_verifier = oidc_verifier
    app.state.security_auditor = security_auditor
    app.state.database = database
    app.state.authorizer = authorizer
    app.state.object_store = object_store
    app.state.recognition_dispatcher = recognition_dispatcher
    app.state.knowledge_ingestion_dispatcher = knowledge_ingestion_dispatcher
    app.state.knowledge_deletion_dispatcher = knowledge_deletion_dispatcher
    app.state.knowledge_index_evaluation_dispatcher = knowledge_index_evaluation_dispatcher
    app.state.knowledge_search_rebuild_dispatcher = knowledge_search_rebuild_dispatcher
    app.state.graph_extraction_dispatcher = graph_extraction_dispatcher
    app.state.graph_extraction_binding_resolver = graph_extraction_binding_resolver
    app.state.maintenance_planning_dispatcher = maintenance_planning_dispatcher
    app.state.maintenance_planning_binding_resolver = maintenance_planning_binding_resolver
    app.state.diagnosis_dispatcher = diagnosis_dispatcher
    app.state.tool_gateway = tool_gateway
    app.state.dlp_processor = dlp_processor
    app.state.label_studio_adapter = label_studio_adapter
    app.state.dataset_pipeline = dataset_pipeline
    app.state.dataset_store = dataset_store
    app.state.experiment_tracker = experiment_tracker
    app.state.model_resolver = model_resolver
    app.state.speech_gateway = speech_gateway
    app.state.realtime_session_service = realtime_session_service
    app.state.realtime_transcript_service = realtime_transcript_service
    app.state.operations_service = operations_service
    app.state.gpu_operations_service = gpu_operations_service
    app.state.recovery_service = recovery_service
    app.state.assurance_service = assurance_service
    app.state.enterprise_project_adoption_service = enterprise_project_adoption_service
    app.state.predictive_maintenance_service = predictive_maintenance_service
    app.state.telemetry_dataset_pipeline = telemetry_dataset_pipeline
    app.state.rul_dataset_pipeline = rul_dataset_pipeline
    app.state.supplier_agent_client = supplier_agent_client
    app.state.knowledge_graph_store = knowledge_graph_store
    app.state.knowledge_search_store = knowledge_search_store
    app.state.knowledge_reranker = knowledge_reranker
    app.state.knowledge_embedder = knowledge_embedder
    app.state.external_search_provider = external_search_provider
    app.state.edge_pack_signer = edge_pack_signer

    @app.middleware("http")
    async def emergency_access_selection(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        token = bind_emergency_grant_id(request.headers.get("X-Emergency-Grant-ID"))
        try:
            return await call_next(request)
        finally:
            reset_emergency_grant_id(token)

    @app.middleware("http")
    async def request_observability(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        return await observe_request(request, call_next)

    @app.get("/health/live", include_in_schema=False)
    async def live() -> dict[str, str]:
        return {"service": resolved.service_name, "status": "ok", "version": __version__}

    @app.get("/health/ready", include_in_schema=False)
    async def ready(request: Request) -> Response:
        is_ready, checks = await _run_readiness_checks(app.state.readiness_checks)
        payload = {
            "checks": checks,
            "request_id": request.state.request_id,
            "status": "ready" if is_ready else "not_ready",
        }
        if not is_ready:
            return JSONResponse(status_code=503, content=payload)
        return JSONResponse(status_code=200, content=payload)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(
            content=generate_latest(app.state.http_metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    app.include_router(me_router, prefix=resolved.api_prefix)
    app.include_router(assets_router, prefix=resolved.api_prefix)
    app.include_router(incidents_router, prefix=resolved.api_prefix)
    app.include_router(uploads_router, prefix=resolved.api_prefix)
    app.include_router(recognition_router, prefix=resolved.api_prefix)
    app.include_router(incident_operations_router, prefix=resolved.api_prefix)
    app.include_router(incident_lifecycle_router, prefix=resolved.api_prefix)
    app.include_router(diagnoses_router, prefix=resolved.api_prefix)
    app.include_router(diagnosis_feedback_router, prefix=resolved.api_prefix)
    app.include_router(device_families_router, prefix=resolved.api_prefix)
    app.include_router(speech_router, prefix=resolved.api_prefix)
    app.include_router(realtime_router, prefix=resolved.api_prefix)
    app.include_router(expert_collaborations_router, prefix=resolved.api_prefix)
    app.include_router(agent_events_router, prefix=resolved.api_prefix)
    app.include_router(approvals_router, prefix=resolved.api_prefix)
    app.include_router(customer_notifications_router, prefix=resolved.api_prefix)
    app.include_router(procurement_router, prefix=resolved.api_prefix)
    app.include_router(procurement_fulfillment_router, prefix=resolved.api_prefix)
    app.include_router(refunds_router, prefix=resolved.api_prefix)
    app.include_router(fsm_assignments_router, prefix=resolved.api_prefix)
    app.include_router(workorders_router, prefix=resolved.api_prefix)
    app.include_router(portal_router, prefix=resolved.api_prefix)
    app.include_router(customer_updates_router, prefix=resolved.api_prefix)
    app.include_router(parts_router, prefix=resolved.api_prefix)
    app.include_router(service_entitlements_router, prefix=resolved.api_prefix)
    app.include_router(tenant_administration_router, prefix=resolved.api_prefix)
    app.include_router(data_feedback_router, prefix=resolved.api_prefix)
    app.include_router(knowledge_router, prefix=resolved.api_prefix)
    app.include_router(knowledge_graph_router, prefix=resolved.api_prefix)
    app.include_router(knowledge_search_router, prefix=resolved.api_prefix)
    app.include_router(external_search_router, prefix=resolved.api_prefix)
    app.include_router(experiments_router, prefix=resolved.api_prefix)
    app.include_router(model_releases_router, prefix=resolved.api_prefix)
    app.include_router(model_deployments_router, prefix=resolved.api_prefix)
    app.include_router(model_gateway_router, prefix=resolved.api_prefix)
    app.include_router(security_router, prefix=resolved.api_prefix)
    app.include_router(emergency_access_router, prefix=resolved.api_prefix)
    app.include_router(equipment_control_handoffs_router, prefix=resolved.api_prefix)
    app.include_router(execution_reconciliations_router, prefix=resolved.api_prefix)
    app.include_router(supply_chain_router, prefix=resolved.api_prefix)
    app.include_router(operations_router, prefix=resolved.api_prefix)
    app.include_router(service_performance_router, prefix=resolved.api_prefix)
    app.include_router(cost_operations_router, prefix=resolved.api_prefix)
    app.include_router(trace_operations_router, prefix=resolved.api_prefix)
    app.include_router(tool_governance_router, prefix=resolved.api_prefix)
    app.include_router(recovery_router, prefix=resolved.api_prefix)
    app.include_router(assurance_router, prefix=resolved.api_prefix)
    app.include_router(enterprise_assets_router, prefix=resolved.api_prefix)
    app.include_router(predictive_maintenance_router, prefix=resolved.api_prefix)
    app.include_router(prompt_governance_router, prefix=resolved.api_prefix)
    app.include_router(memories_router, prefix=resolved.api_prefix)
    app.include_router(collaborations_router, prefix=resolved.api_prefix)
    app.include_router(maintenance_planning_router, prefix=resolved.api_prefix)
    app.include_router(maintenance_planning_activation_router, prefix=resolved.api_prefix)
    app.include_router(maintenance_planning_evaluation_suite_router, prefix=resolved.api_prefix)
    app.include_router(maintenance_planning_evaluation_router, prefix=resolved.api_prefix)
    register_error_handlers(app)
    return app
