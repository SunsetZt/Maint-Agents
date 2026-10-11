"""FastAPI dependencies that establish only server-verified identity."""

from __future__ import annotations

from typing import Annotated, Protocol

from fastapi import Header, Request

from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.application.diagnoses import DiagnosisDispatcher
from industrial_ops_agent.assurance.service import AssuranceService
from industrial_ops_agent.auth.errors import AuthenticationFailure
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.data_governance.dlp import PresidioDlpProcessor
from industrial_ops_agent.data_pipeline.service import DatasetPipelineService
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.edge.signing import Ed25519PackSigner
from industrial_ops_agent.enterprise_assets import EnterpriseProjectAdoptionReader
from industrial_ops_agent.experiments.mlflow import ExperimentTracker
from industrial_ops_agent.gpu_operations.service import GpuOperationsService
from industrial_ops_agent.graph_rag.extraction import (
    GraphExtractionBindingResolver,
    GraphExtractionDispatcher,
)
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
from industrial_ops_agent.search_profiles.rebuild import SearchProfileRebuildDispatcher
from industrial_ops_agent.tenant_administration.service import (
    DirectoryPolicyService,
    ManagedIdentityDenied,
)
from industrial_ops_agent.tools.gateway import ToolGateway


class AccessTokenVerifier(Protocol):
    def verify(self, token: str) -> IdentityContext: ...


async def get_optional_edge_pack_signer(request: Request) -> Ed25519PackSigner | None:
    return getattr(request.app.state, "edge_pack_signer", None)


def get_identity(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> IdentityContext:
    if authorization is None or not authorization.startswith("Bearer "):
        _audit_failure(request, "authentication_required")
        raise AppError(
            status_code=401,
            code="authentication_required",
            category="authentication",
            message="Authentication required",
        )
    token = authorization.removeprefix("Bearer ").strip()
    verifier: AccessTokenVerifier | None = getattr(request.app.state, "oidc_verifier", None)
    if verifier is None:
        raise AppError(
            status_code=503,
            code="authentication_unavailable",
            category="dependency",
            message="Authentication service unavailable",
            retryable=True,
        )
    try:
        identity = verifier.verify(token)
    except AuthenticationFailure as exc:
        _audit_failure(request, exc.reason_code)
        raise AppError(
            status_code=401,
            code="invalid_access_token",
            category="authentication",
            message="Access token is invalid",
        ) from exc
    database: Database | None = getattr(request.app.state, "database", None)
    if database is not None:
        try:
            identity = DirectoryPolicyService(database).resolve(identity)
        except ManagedIdentityDenied as exc:
            _audit_failure(request, exc.reason)
            raise AppError(
                status_code=401,
                code="identity_access_revoked",
                category="authentication",
                message="Identity access has been revoked or suspended",
            ) from exc
    request.state.identity = identity
    return identity


def _audit_failure(request: Request, reason_code: str) -> None:
    auditor = getattr(request.app.state, "security_auditor", None)
    if auditor is not None:
        auditor.record(
            identity=None,
            action="authentication.verify",
            decision="deny",
            reason_code=reason_code,
            request_id=getattr(request.state, "request_id", "unavailable"),
        )


async def get_database(request: Request) -> Database:
    database: Database | None = getattr(request.app.state, "database", None)
    if database is None:
        raise AppError(
            status_code=503,
            code="database_unavailable",
            category="dependency",
            message="Database service unavailable",
            retryable=True,
        )
    return database


async def get_authorizer(request: Request) -> Authorizer:
    authorizer: Authorizer | None = getattr(request.app.state, "authorizer", None)
    if authorizer is None:
        raise AppError(
            status_code=503,
            code="authorization_unavailable",
            category="dependency",
            message="Authorization service unavailable",
            retryable=True,
        )
    return authorizer


async def get_assurance_service(request: Request) -> AssuranceService:
    service: AssuranceService | None = getattr(request.app.state, "assurance_service", None)
    if service is None:
        raise AppError(
            status_code=503,
            code="assurance_service_unavailable",
            category="dependency",
            message="Production assurance service unavailable",
            retryable=True,
        )
    return service


async def get_optional_assurance_service(request: Request) -> AssuranceService | None:
    return getattr(request.app.state, "assurance_service", None)


async def get_enterprise_project_adoption_service(
    request: Request,
) -> EnterpriseProjectAdoptionReader:
    service: EnterpriseProjectAdoptionReader | None = getattr(
        request.app.state,
        "enterprise_project_adoption_service",
        None,
    )
    if service is None:
        raise AppError(
            status_code=503,
            code="enterprise_project_adoption_service_unavailable",
            category="dependency",
            message="Enterprise project adoption service unavailable",
            retryable=True,
        )
    return service


async def get_predictive_maintenance_service(
    request: Request,
) -> PredictiveMaintenanceService:
    service: PredictiveMaintenanceService | None = getattr(
        request.app.state,
        "predictive_maintenance_service",
        None,
    )
    if service is None:
        raise AppError(
            status_code=503,
            code="predictive_maintenance_service_unavailable",
            category="dependency",
            message="Predictive-maintenance service unavailable",
            retryable=True,
        )
    return service


async def get_telemetry_dataset_pipeline(
    request: Request,
) -> TelemetryDatasetPipelineService:
    pipeline: TelemetryDatasetPipelineService | None = getattr(
        request.app.state,
        "telemetry_dataset_pipeline",
        None,
    )
    if pipeline is None:
        raise AppError(
            status_code=503,
            code="telemetry_dataset_pipeline_unavailable",
            category="dependency",
            message="Telemetry dataset pipeline unavailable",
            retryable=True,
        )
    return pipeline


async def get_rul_dataset_pipeline(request: Request) -> RulDatasetPipelineService:
    pipeline: RulDatasetPipelineService | None = getattr(
        request.app.state,
        "rul_dataset_pipeline",
        None,
    )
    if pipeline is None:
        raise AppError(
            status_code=503,
            code="rul_dataset_pipeline_unavailable",
            category="dependency",
            message="RUL dataset pipeline unavailable",
            retryable=True,
        )
    return pipeline


async def get_object_store(request: Request) -> TenantObjectStore:
    object_store: TenantObjectStore | None = getattr(
        request.app.state,
        "object_store",
        None,
    )
    if object_store is None:
        raise AppError(
            status_code=503,
            code="object_store_unavailable",
            category="dependency",
            message="Object store unavailable",
            retryable=True,
        )
    return object_store


async def get_recognition_dispatcher(request: Request) -> RecognitionDispatcher:
    dispatcher: RecognitionDispatcher | None = getattr(
        request.app.state,
        "recognition_dispatcher",
        None,
    )
    if dispatcher is None:
        raise AppError(
            status_code=503,
            code="recognition_workflow_unavailable",
            category="dependency",
            message="Recognition workflow unavailable",
            retryable=True,
        )
    return dispatcher


async def get_knowledge_ingestion_dispatcher(
    request: Request,
) -> KnowledgeIngestionDispatcher:
    dispatcher: KnowledgeIngestionDispatcher | None = getattr(
        request.app.state,
        "knowledge_ingestion_dispatcher",
        None,
    )
    if dispatcher is None:
        raise AppError(
            status_code=503,
            code="knowledge_ingestion_workflow_unavailable",
            category="dependency",
            message="Knowledge ingestion workflow unavailable",
            retryable=True,
        )
    return dispatcher


async def get_knowledge_deletion_dispatcher(
    request: Request,
) -> KnowledgeDeletionDispatcher:
    dispatcher: KnowledgeDeletionDispatcher | None = getattr(
        request.app.state,
        "knowledge_deletion_dispatcher",
        None,
    )
    if dispatcher is None:
        raise AppError(
            status_code=503,
            code="knowledge_deletion_workflow_unavailable",
            category="dependency",
            message="Knowledge deletion workflow unavailable",
            retryable=True,
        )
    return dispatcher


async def get_knowledge_index_evaluation_dispatcher(
    request: Request,
) -> KnowledgeIndexEvaluationDispatcher:
    dispatcher: KnowledgeIndexEvaluationDispatcher | None = getattr(
        request.app.state,
        "knowledge_index_evaluation_dispatcher",
        None,
    )
    if dispatcher is None:
        raise AppError(
            status_code=503,
            code="knowledge_index_evaluation_workflow_unavailable",
            category="dependency",
            message="Knowledge index evaluation workflow unavailable",
            retryable=True,
        )
    return dispatcher


async def get_knowledge_search_rebuild_dispatcher(
    request: Request,
) -> SearchProfileRebuildDispatcher:
    dispatcher: SearchProfileRebuildDispatcher | None = getattr(
        request.app.state,
        "knowledge_search_rebuild_dispatcher",
        None,
    )
    if dispatcher is None:
        raise AppError(
            status_code=503,
            code="knowledge_search_rebuild_workflow_unavailable",
            category="dependency",
            message="Knowledge search rebuild workflow unavailable",
            retryable=True,
        )
    return dispatcher


async def get_graph_extraction_dispatcher(request: Request) -> GraphExtractionDispatcher:
    dispatcher: GraphExtractionDispatcher | None = getattr(
        request.app.state,
        "graph_extraction_dispatcher",
        None,
    )
    if dispatcher is None:
        raise AppError(
            status_code=503,
            code="graph_extraction_workflow_unavailable",
            category="dependency",
            message="Graph extraction workflow unavailable",
            retryable=True,
        )
    return dispatcher


async def get_graph_extraction_binding_resolver(
    request: Request,
) -> GraphExtractionBindingResolver:
    resolver: GraphExtractionBindingResolver | None = getattr(
        request.app.state,
        "graph_extraction_binding_resolver",
        None,
    )
    if resolver is None:
        raise AppError(
            status_code=503,
            code="graph_extraction_runtime_binding_unavailable",
            category="dependency",
            message="Graph extraction runtime binding unavailable",
            retryable=True,
        )
    return resolver


async def get_maintenance_planning_dispatcher(
    request: Request,
) -> MaintenancePlanningDispatcher:
    dispatcher: MaintenancePlanningDispatcher | None = getattr(
        request.app.state,
        "maintenance_planning_dispatcher",
        None,
    )
    if dispatcher is None:
        raise AppError(
            status_code=503,
            code="maintenance_planning_workflow_unavailable",
            category="dependency",
            message="Maintenance planning workflow unavailable",
            retryable=True,
        )
    return dispatcher


async def get_maintenance_planning_binding_resolver(
    request: Request,
) -> MaintenancePlanningBindingResolver:
    resolver: MaintenancePlanningBindingResolver | None = getattr(
        request.app.state,
        "maintenance_planning_binding_resolver",
        None,
    )
    if resolver is None:
        raise AppError(
            status_code=503,
            code="maintenance_planning_runtime_binding_unavailable",
            category="dependency",
            message="Maintenance planning runtime binding unavailable",
            retryable=True,
        )
    return resolver


async def get_diagnosis_dispatcher(request: Request) -> DiagnosisDispatcher:
    dispatcher: DiagnosisDispatcher | None = getattr(
        request.app.state,
        "diagnosis_dispatcher",
        None,
    )
    if dispatcher is None:
        raise AppError(
            status_code=503,
            code="diagnosis_workflow_unavailable",
            category="dependency",
            message="Diagnosis workflow unavailable",
            retryable=True,
        )
    return dispatcher


async def get_model_resolver(request: Request) -> EnvironmentAwareModelResolver | None:
    """Return the configured fail-closed governed model resolver, when assembled."""

    return getattr(request.app.state, "model_resolver", None)


async def get_speech_gateway(request: Request) -> SpeechSynthesisGateway:
    gateway: SpeechSynthesisGateway | None = getattr(
        request.app.state,
        "speech_gateway",
        None,
    )
    if gateway is None:
        raise AppError(
            status_code=503,
            code="tts_service_unavailable",
            category="dependency",
            message="Speech synthesis service unavailable",
            retryable=True,
        )
    return gateway


async def get_realtime_session_service(request: Request) -> RealtimeSessionService:
    service: RealtimeSessionService | None = getattr(
        request.app.state,
        "realtime_session_service",
        None,
    )
    if service is None:
        raise AppError(
            status_code=503,
            code="realtime_media_unavailable",
            category="dependency",
            message="Realtime media service unavailable",
            retryable=True,
        )
    return service


async def get_realtime_transcript_service(request: Request) -> RealtimeTranscriptService:
    service: RealtimeTranscriptService | None = getattr(
        request.app.state,
        "realtime_transcript_service",
        None,
    )
    if service is None:
        raise AppError(
            status_code=503,
            code="realtime_asr_unavailable",
            category="dependency",
            message="Realtime transcription service unavailable",
            retryable=True,
        )
    return service


async def get_tool_gateway(request: Request) -> ToolGateway:
    gateway: ToolGateway | None = getattr(request.app.state, "tool_gateway", None)
    if gateway is None:
        raise AppError(
            status_code=503,
            code="tool_gateway_unavailable",
            category="dependency",
            message="Tool gateway unavailable",
            retryable=True,
        )
    return gateway


async def get_dlp_processor(request: Request) -> PresidioDlpProcessor:
    processor: PresidioDlpProcessor | None = getattr(
        request.app.state,
        "dlp_processor",
        None,
    )
    if processor is None:
        raise AppError(
            status_code=503,
            code="dlp_unavailable",
            category="dependency",
            message="DLP service unavailable",
            retryable=True,
        )
    return processor


async def get_label_studio_adapter(request: Request) -> LabelStudioAdapter:
    adapter: LabelStudioAdapter | None = getattr(
        request.app.state,
        "label_studio_adapter",
        None,
    )
    if adapter is None:
        raise AppError(
            status_code=503,
            code="label_studio_unavailable",
            category="dependency",
            message="Label Studio unavailable",
            retryable=True,
        )
    return adapter


async def get_dataset_pipeline(request: Request) -> DatasetPipelineService:
    pipeline: DatasetPipelineService | None = getattr(
        request.app.state,
        "dataset_pipeline",
        None,
    )
    if pipeline is None:
        raise AppError(
            status_code=503,
            code="dataset_pipeline_unavailable",
            category="dependency",
            message="Dataset pipeline unavailable",
            retryable=True,
        )
    return pipeline


async def get_dataset_store(request: Request) -> DatasetStore:
    store: DatasetStore | None = getattr(request.app.state, "dataset_store", None)
    if store is None:
        raise AppError(
            status_code=503,
            code="dataset_store_unavailable",
            category="dependency",
            message="Dataset store unavailable",
            retryable=True,
        )
    return store


async def get_experiment_tracker(request: Request) -> ExperimentTracker:
    tracker: ExperimentTracker | None = getattr(
        request.app.state,
        "experiment_tracker",
        None,
    )
    if tracker is None:
        raise AppError(
            status_code=503,
            code="experiment_tracking_unavailable",
            category="dependency",
            message="Experiment tracking unavailable",
            retryable=True,
        )
    return tracker


async def get_operations_service(request: Request) -> OperationsService:
    service: OperationsService | None = getattr(
        request.app.state,
        "operations_service",
        None,
    )
    if service is None:
        raise AppError(
            status_code=503,
            code="operations_service_unavailable",
            category="dependency",
            message="Operations service unavailable",
            retryable=True,
        )
    return service


async def get_gpu_operations_service(request: Request) -> GpuOperationsService:
    service: GpuOperationsService | None = getattr(
        request.app.state,
        "gpu_operations_service",
        None,
    )
    if service is None:
        raise AppError(
            status_code=503,
            code="gpu_operations_service_unavailable",
            category="dependency",
            message="GPU operations service unavailable",
            retryable=True,
        )
    return service


async def get_optional_operations_service(request: Request) -> OperationsService | None:
    """Allow isolated tests/local factories to omit monitoring; production always injects it."""

    return getattr(request.app.state, "operations_service", None)


async def get_recovery_service(request: Request) -> RecoveryService:
    service: RecoveryService | None = getattr(request.app.state, "recovery_service", None)
    if service is None:
        raise AppError(
            status_code=503,
            code="recovery_service_unavailable",
            category="dependency",
            message="Recovery governance service unavailable",
            retryable=True,
        )
    return service
