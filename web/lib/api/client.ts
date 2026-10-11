import createClient, { type Client } from "openapi-fetch";

import { readAgUiEvents, type AgUiTransportEvent } from "@/lib/ag-ui-stream";

import type { components, paths } from "@/generated/api/schema";

export type AssetSummary = components["schemas"]["AssetResponse"];
export type AssetDetail = components["schemas"]["AssetDetailResponse"];
export type IncidentDraft = components["schemas"]["IncidentDraftResponse"];
export type MediaStatus = components["schemas"]["MediaStatusResponse"];
export type TimelineEvent = components["schemas"]["TimelineEventResponse"];
export type EvidenceBundle = components["schemas"]["EvidenceResponse"];
export type RecognitionRun = components["schemas"]["RecognitionRunResponse"];
export type Incident = components["schemas"]["IncidentResponse"];
export type IncidentOperations = components["schemas"]["IncidentOperationsResponse"];
export type IncidentQueueItem = components["schemas"]["IncidentQueueItemResponse"];
export type IncidentControl = components["schemas"]["IncidentControlResponse"];
export type IncidentResolution = components["schemas"]["IncidentResolutionResponse"];
export type ClassifyIncidentInput = components["schemas"]["ClassifyIncidentRequest"];
export type ResolveRemoteIncidentInput =
  components["schemas"]["ResolveRemoteIncidentRequest"];
export type CloseIncidentInput = components["schemas"]["CloseIncidentRequest"];
export type RequestIncidentInformationInput =
  components["schemas"]["RequestInformationRequest"];
export type EscalateIncidentInput = components["schemas"]["EscalateIncidentRequest"];
export type ResumeIncidentInput = components["schemas"]["ResumeIncidentRequest"];
export type MergeDuplicateIncidentInput = components["schemas"]["MergeDuplicateRequest"];
export type TenantOverview = components["schemas"]["TenantOverviewResponse"];
export type TenantAssetOption = components["schemas"]["TenantAssetOptionResponse"];
export type TenantSiteOption = components["schemas"]["TenantSiteOptionResponse"];
export type TenantMember = components["schemas"]["TenantMemberResponse"];
export type TenantRetentionPolicy = components["schemas"]["RetentionPolicyResponse"];
export type TenantAdministrationRecord =
  components["schemas"]["TenantAdministrationResponse"];
export type CreateTenantMemberInput = components["schemas"]["CreateTenantMemberBody"];
export type UpdateTenantMemberInput = components["schemas"]["UpdateTenantMemberBody"];
export type UpdateTenantRetentionPolicyInput =
  components["schemas"]["UpdateRetentionPolicyBody"];
export type DiagnosisRun = components["schemas"]["DiagnosisRunResponse"];
export type DiagnosisQualityFeedback =
  components["schemas"]["DiagnosisFeedbackResponse"];
export type DiagnosisQualityFeedbackInput =
  components["schemas"]["DiagnosisFeedbackRequest"];
export type AgentRunControl = components["schemas"]["AgentRunControlResponse"];
export type ExpertDiagnosisIntervention =
  components["schemas"]["ExpertDiagnosisInterventionResponse"];
export type SpeechSynthesis = components["schemas"]["SpeechSynthesisResponse"];
export type FieldSpeechSynthesis = components["schemas"]["FieldSpeechSynthesisResponse"];
export type RealtimeSession = components["schemas"]["RealtimeSessionResponse"];
export type FieldVoiceGuidance = components["schemas"]["FieldVoiceGuidanceResponse"];
export type ExpertCollaboration = components["schemas"]["ExpertCollaborationResponse"];
export type ExpertRecommendation = components["schemas"]["ExpertRecommendationResponse"];
export type ExpertCollaborationOverview =
  components["schemas"]["ExpertCollaborationOverviewResponse"];
export type SubmitExpertRecommendationInput =
  components["schemas"]["SubmitExpertRecommendationRequest"];

export interface AgUiRunAgentInput {
  threadId: string;
  runId: string;
  parentRunId?: string;
  state: Record<string, unknown>;
  messages: Array<{
    id: string;
    role: "user" | "assistant" | "system" | "developer" | "tool";
    content: string;
  }>;
  tools: Array<{
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  }>;
  context: Array<{ description: string; value: string }>;
  forwardedProps: Record<string, unknown>;
  resume?: Array<{
    interruptId: string;
    status: "resolved" | "cancelled";
    payload?: Record<string, unknown>;
  }>;
}

export type { AgUiTransportEvent } from "@/lib/ag-ui-stream";
export type RealtimeSessionGrant = components["schemas"]["RealtimeSessionGrantResponse"];
export type RealtimeTranscript = components["schemas"]["RealtimeTranscriptResponse"];
export type Citation = components["schemas"]["CitationResponse"];
export type IndustrialApiClient = Client<paths>;

export type CitationSourceDownload = {
  blob: Blob;
  sourceChecksum: string;
  requestId?: string;
};

export type ApprovalView = components["schemas"]["ApprovalResponse"];
export type ApprovalExecution = components["schemas"]["ExecutionResponse"];
export type ExecutionReconciliation =
  components["schemas"]["ExecutionReconciliationResponse"];
export type ExecutionReconciliationEvent =
  components["schemas"]["ExecutionReconciliationEventResponse"];
export type CustomerNotificationProposal = components["schemas"]["ActionProposalResponse"];
export type CustomerNotificationProposalInput = Omit<components["schemas"]["CustomerNotificationProposalBody"], "channel"> & { channel?: components["schemas"]["CustomerNotificationProposalBody"]["channel"] };
export type CustomerNotificationOption =
  components["schemas"]["CustomerNotificationOptionResponse"];
export type PurchaseRequestProposal = components["schemas"]["ActionProposalResponse"];
export type PurchaseRequestProposalInput = Omit<
  components["schemas"]["PurchaseRequestProposalBody"],
  "delivery_target"
> & {
  delivery_target?: components["schemas"]["PurchaseRequestProposalBody"]["delivery_target"];
};
export type PurchaseRequestOption =
  components["schemas"]["PurchaseRequestOptionResponse"];
export type PurchaseRequest = components["schemas"]["PurchaseRequestResponse"];
export type PurchaseFulfillment = components["schemas"]["PurchaseFulfillmentResponse"];
export type PurchaseReservationReadiness =
  components["schemas"]["PurchaseReservationReadinessResponse"];
export type ServiceQuotationOption =
  components["schemas"]["ServiceQuotationOptionResponse"];
export type ServiceQuotation = components["schemas"]["ServiceQuotationResponse"];
export type ServiceQuotationProposal = components["schemas"]["ActionProposalResponse"];
export type CustomerServiceQuotationView =
  components["schemas"]["CustomerQuotationViewResponse"];
export type CustomerServiceQuotationDecision =
  components["schemas"]["CustomerQuotationDecisionResponse"];
export type RepairWorkOrderOption = components["schemas"]["RepairWorkOrderOptionResponse"];
export type RepairWorkOrderProposal = components["schemas"]["ActionProposalResponse"];
export type PartsReservationProposal = components["schemas"]["ActionProposalResponse"];
export type PartsReservationProposalInput =
  components["schemas"]["PartReservationProposalBody"];
export type RefundRequestProposal = components["schemas"]["ActionProposalResponse"];
export type RefundRequestProposalInput = Omit<
  components["schemas"]["RefundRequestProposalBody"],
  "delivery_target"
> & {
  delivery_target?: components["schemas"]["RefundRequestProposalBody"]["delivery_target"];
};
export type RefundRequestOption = components["schemas"]["RefundRequestOptionResponse"];
export type RefundRequest = components["schemas"]["RefundRequestResponse"];
export type WorkOrderAssignmentProposal = components["schemas"]["ActionProposalResponse"];
export type WorkOrderAssignmentProposalInput = Omit<components["schemas"]["WorkOrderAssignmentProposalBody"], "assignment_target"> & { assignment_target?: components["schemas"]["WorkOrderAssignmentProposalBody"]["assignment_target"] };
export type WorkOrderAssignmentOption =
  components["schemas"]["WorkOrderAssignmentOptionResponse"];
export type WorkOrderClosureProposal = components["schemas"]["ActionProposalResponse"];
export type WorkOrderClosureProposalInput =
  components["schemas"]["WorkOrderClosureProposalBody"];
type WorkOrderOriginFields = "creation_mode" | "authorization_type" | "diagnosis_run_id" | "diagnosis_version" | "service_quotation_id" | "initial_parts_required" | "part_issue_required" | "part_accounting_required";
type CompatibleWorkOrder<T extends Record<string, unknown>> =
  Omit<T, "pending_assignment_operation_id" | WorkOrderOriginFields>
  & Partial<Pick<T, WorkOrderOriginFields>>
  & { pending_assignment_operation_id?: string | null };
export type WorkOrderView = CompatibleWorkOrder<components["schemas"]["WorkOrderResponse"]>;
export type DispatchWorkOrder = CompatibleWorkOrder<components["schemas"]["DispatchWorkOrderResponse"]>;
export type DispatchFact = components["schemas"]["DispatchFactResponse"];
export type WorkOrderCenterItem = CompatibleWorkOrder<components["schemas"]["WorkOrderCenterItemResponse"]>;
export type WorkOrderControl = components["schemas"]["WorkOrderControlResponse"];
export type WorkOrderRepairHistory =
  components["schemas"]["WorkOrderRepairHistoryResponse"];
export type WorkOrderPartAllocation =
  components["schemas"]["WorkOrderPartAllocationResponse"];
export type WorkOrderOfflinePack =
  components["schemas"]["WorkOrderOfflinePackResponse"];
export type FieldEdgeDiagnosisPack =
  components["schemas"]["FieldEdgeDiagnosisPackResponse"];
export type FieldEdgeDiagnosisCandidate =
  components["schemas"]["FieldEdgeDiagnosisCandidateResponse"];
export type FieldEdgeDiagnosisResult =
  components["schemas"]["FieldEdgeDiagnosisResult"];
export type FieldEvidenceUpload =
  components["schemas"]["FieldEvidenceUploadResponse"];
export type FieldRecognitionRun =
  components["schemas"]["FieldRecognitionRunResponse"];
export type FieldOcrBlockDecision =
  components["schemas"]["FieldOcrBlockDecisionBody"];
export type FieldTranscriptDecision =
  components["schemas"]["FieldTranscriptDecisionBody"];
export type IncidentOcrBlockDecision =
  components["schemas"]["OcrDecisionRequest"];
export type IncidentQrCodeDecision =
  components["schemas"]["QrCodeDecisionRequest"];
export type FieldQrCodeDecision =
  components["schemas"]["FieldQrCodeDecisionBody"];
export type PartIssueProposal = components["schemas"]["PartIssueProposalResponse"];
export type WorkOrderPartAccounting =
  components["schemas"]["WorkOrderPartAccountingResponse"];
export type PartMaterialMovementProposal =
  components["schemas"]["PartMaterialMovementProposalResponse"];
export type WorkOrderClosureReport =
  components["schemas"]["WorkOrderClosureReportResponse"];
export type WorkOrderHoldInput = components["schemas"]["HoldBody"];
export type WorkOrderEscalateInput = components["schemas"]["EscalateBody"];
export type WorkOrderReplanInput = components["schemas"]["ReplanBody"];
export type WorkOrderRescheduleInput = components["schemas"]["RescheduleBody"];
export type FieldWorkOrderEntry = components["schemas"]["WorkOrderFieldEntryResponse"];
export type FieldEntryInput = components["schemas"]["FieldEntryBody"];
export type FieldCompletionInput = components["schemas"]["FieldCompletionBody"];
export type CustomerCase = components["schemas"]["CustomerCaseResponse"];
export type CustomerCaseUpdate = components["schemas"]["CustomerUpdateResponse"];
export type CustomerCaseUpdateInput = components["schemas"]["CustomerUpdateBody"];
export type EquipmentControlHandoff =
  components["schemas"]["EquipmentControlHandoffResponse"];
export type EquipmentControlHandoffInput =
  components["schemas"]["EquipmentControlHandoffBody"];
export type SparePartsLookup = components["schemas"]["PartsLookupResponse"];
export type ServiceEntitlement = components["schemas"]["ServiceEntitlementResponse"];
export type FeedbackCandidate = components["schemas"]["FeedbackCandidateSummaryResponse"];
export type FeedbackCandidateDetail = components["schemas"]["FeedbackCandidateDetailResponse"];
export type GovernanceBlockerCode = FeedbackCandidate["governance_blockers"][number];
export type EligibilityDecision = components["schemas"]["EligibilityDecisionResponse"];
export type DlpProcessingHistory = components["schemas"]["DlpProcessingHistoryResponse"];
export type AnnotationTaskHistory = components["schemas"]["AnnotationTaskHistoryResponse"];
export type CurationRun = components["schemas"]["CurationRunResponse"];
export type DatasetSnapshot = components["schemas"]["DatasetSnapshotResponse"];
export type DatasetTrainingBlockerCode = DatasetSnapshot["training_blockers"][number];
export type DatasetQualityReport = components["schemas"]["DatasetQualityReportResponse"];
export type DataLineage = components["schemas"]["DataLineageResponse"];
export type LineageStage = components["schemas"]["TimelineStageResponse"];
export type CandidateGovernance = components["schemas"]["CandidateGovernanceResponse"];
export type CandidateEligibilityInput = components["schemas"]["EligibilityDecisionBody"];
export type DlpRun = components["schemas"]["DlpRunResponse"];
export type AnnotationTask = components["schemas"]["AnnotationTaskResponse"];
export type KnowledgeVersion = components["schemas"]["KnowledgeVersionResponse"];
export type KnowledgeRelease = components["schemas"]["KnowledgeReleaseResponse"];
export type KnowledgeIndexActivation =
  components["schemas"]["KnowledgeIndexActivationResponse"];
export type KnowledgeIndexEvaluation = components["schemas"]["KnowledgeIndexEvaluationResponse"];
export type KnowledgeFile = components["schemas"]["KnowledgeFileResponse"];
export type KnowledgeDeletion = components["schemas"]["KnowledgeDeletionResponse"];
export type KnowledgeGraphRelease = components["schemas"]["GraphReleaseResponse"];
export type KnowledgeGraphNode = components["schemas"]["GraphNodeResponse"];
export type KnowledgeGraphEdge = components["schemas"]["GraphEdgeResponse"];
export type KnowledgeGraphPath = components["schemas"]["GraphPathResponse"];
export type KnowledgeGraphExtraction = components["schemas"]["GraphExtractionResponse"];
export type CreateKnowledgeGraphInput = components["schemas"]["CreateGraphReleaseBody"];
export type CreateKnowledgeGraphNodeInput = components["schemas"]["CreateGraphNodeBody"];
export type CreateKnowledgeGraphEdgeInput = components["schemas"]["CreateGraphEdgeBody"];
export type EvaluateKnowledgeGraphInput = components["schemas"]["EvaluateGraphBody"];
export type KnowledgeGraphQueryInput = components["schemas"]["GraphQueryBody"];
export type ReviewKnowledgeGraphExtractionInput =
  components["schemas"]["ReviewGraphExtractionBody"];
export type KnowledgeSearchProfile = components["schemas"]["SearchProfileResponse"];
export type KnowledgeSearchRebuild = components["schemas"]["SearchProfileRebuildResponse"];
export type KnowledgeSearchResult = components["schemas"]["SearchResultResponse"];
export type CreateKnowledgeSearchProfileInput = components["schemas"]["CreateSearchProfileBody"];
export type EvaluateKnowledgeSearchProfileInput = components["schemas"]["EvaluateSearchProfileBody"];
export type KnowledgeSearchQueryInput = components["schemas"]["SearchQueryBody"];
export type ExternalSearchPolicy = components["schemas"]["ExternalSearchPolicyResponse"];
export type ExternalSearchQuery = components["schemas"]["ExternalSearchQueryResponse"];
export type ExternalSearchPolicyInput = components["schemas"]["ExternalSearchPolicyBody"];
export type ExternalSearchInput = components["schemas"]["ExternalSearchBody"];
export type ExternalSearchConclusionInput =
  components["schemas"]["ExternalSearchConclusionBody"];
export type TrainingExperiment = components["schemas"]["TrainingExperimentResponse"];
export type EvaluationJob = components["schemas"]["EvaluationJobResponse"];
export type EvaluationSuite = components["schemas"]["EvaluationSuiteResponse"];
export type EvaluationPolicy = components["schemas"]["EvaluationPolicyResponse"];
export type ModelEvaluation = components["schemas"]["ModelEvaluationResponse"];
export type ModelRelease = components["schemas"]["ModelReleaseResponse"];
export type ModelDeployment = components["schemas"]["ModelDeploymentResponse"];
export type ModelDeploymentPlanInput = components["schemas"]["DeploymentPlanBody"];
export type ModelRoute = components["schemas"]["ModelRouteResponse"];
export type ModelRuntimeStatus = components["schemas"]["ModelRuntimeStatusResponse"];
export type ModelExecution = components["schemas"]["ModelExecutionResponse"];
export type ModelInference = components["schemas"]["ModelInferenceResponse"];
export type ModelQuotaInput = components["schemas"]["ModelQuotaBody"];
export type EnterpriseProjectAdoption =
  components["schemas"]["EnterpriseProjectAdoptionReport"];
export type EnterpriseProjectClosure =
  components["schemas"]["SimulatedEnterpriseClosureReport"];
export type EnterpriseProjectAssurance =
  components["schemas"]["ProjectAssuranceLabReport"];
export type EnterpriseProjectAsset = components["schemas"]["AdoptedEnterpriseAsset"];
export type EnterpriseProjectExperimentHistory =
  components["schemas"]["EnterpriseExperimentHistory"];
export type EnterpriseModelImportAcceptance =
  components["schemas"]["EnterpriseModelImportAcceptanceReport"];
export type EnterpriseRerankerKServeAcceptance =
  components["schemas"]["EnterpriseRerankerKServeAcceptanceReport"];
export type EnterpriseRuntimeBinding = components["schemas"]["EnterpriseRuntimeBinding"];
export type EnterpriseModelAssetImport =
  components["schemas"]["EnterpriseModelAssetImport"];
export type EnterpriseReleaseDraftPreview =
  components["schemas"]["EnterpriseReleaseDraftPreview"];
export type EnterpriseReleaseDraftInput =
  components["schemas"]["EnterpriseReleaseDraftBody"];
export type EnterpriseReleaseOnboarding =
  components["schemas"]["EnterpriseReleaseOnboarding"];
export type EnterpriseStagingBaselineDraft =
  components["schemas"]["EnterpriseStagingBaselineDraft"];
export type EnterpriseCandidateReleaseBatch =
  components["schemas"]["EnterpriseCandidateReleaseBatch"];
export type EnterpriseCandidateReleaseBatchProgress =
  components["schemas"]["EnterpriseCandidateReleaseBatchProgress"];
export type EnterpriseModelComponent =
  EnterpriseRuntimeBinding["evidence"]["component"];
export type SecurityAuditEvent = components["schemas"]["SecurityAuditEventResponse"];
export type SecurityAuditSummary = components["schemas"]["SecurityAuditSummaryResponse"];
export type EmergencyAccessGrant =
  components["schemas"]["EmergencyAccessGrantResponse"];
export type EmergencyAccessRequestInput =
  components["schemas"]["EmergencyAccessRequestBody"];
export type EmergencyAccessDecisionInput =
  components["schemas"]["EmergencyAccessDecisionBody"];
export type EmergencyAccessRevokeInput =
  components["schemas"]["EmergencyAccessRevokeBody"];
export type SupplyChainEvidence = components["schemas"]["SupplyChainEvidenceResponse"];
export type OperationsOverview = components["schemas"]["OperationsOverviewResponse"];
export type ServicePerformance = components["schemas"]["ServicePerformanceResponse"];
export type ServicePerformanceV3 = components["schemas"]["ServicePerformanceV3Response"];
export type ServicePerformanceBaseline = components["schemas"]["ServicePerformanceBaselineResponse"];
export type CreateServicePerformanceBaselineInput =
  components["schemas"]["ServicePerformanceBaselineCreateBody"];
export type SloObservation = components["schemas"]["SloObservationResponse"];
export type OperationsAlert = components["schemas"]["ActiveAlertResponse"];
export type OperationsRunbook = components["schemas"]["RunbookResponse"];
export type GpuOperations = components["schemas"]["GpuOperationsResponse"];
export type ClusterGpuPosture = components["schemas"]["ClusterGpuPostureResponse"];
export type RuntimeGpuPosture = components["schemas"]["RuntimeGpuPostureResponse"];
export type CostOperations = components["schemas"]["CostOperationsResponse"];
export type CostBreakdown = components["schemas"]["CostBreakdownResponse"];
export type CostPolicy = components["schemas"]["CostPolicyResponse"];
export type CostPolicyInput = components["schemas"]["CostPolicyBody"];
export type CostLedgerBatch = components["schemas"]["CostLedgerBatchReceiptResponse"];
export type CostLedgerBatchInput = components["schemas"]["CostLedgerBatchBody"];
export type AgentTraceSummary = components["schemas"]["TraceSummaryResponse"];
export type AgentTraceDetail = components["schemas"]["TraceDetailResponse"];
export type AgentTraceNode = components["schemas"]["TraceNodeResponse"];
export type ToolGovernance = components["schemas"]["ToolGovernanceResponse"];
export type GovernedTool = components["schemas"]["GovernedToolResponse"];
export type RecentToolCall = components["schemas"]["RecentToolCallResponse"];
export type PromptBundle = components["schemas"]["PromptBundleResponse"];
export type CreatePromptBundleInput = components["schemas"]["CreatePromptBundleBody"];
export type GovernedMemory = components["schemas"]["GovernedMemoryResponse"];
export type CreateMemoryInput = components["schemas"]["CreateMemoryBody"];
export type SupplierCollaboration = components["schemas"]["SupplierCollaborationResponse"];
export type CreateSupplierCollaborationInput =
  components["schemas"]["CreateSupplierCollaborationBody"];
export type ReviewSupplierCollaborationInput =
  components["schemas"]["ReviewSupplierCollaborationBody"];
export type MaintenancePlanningCouncil =
  components["schemas"]["MaintenancePlanningResponse"];
export type CreateMaintenancePlanningInput =
  components["schemas"]["CreateMaintenancePlanningBody"];
export type ReviewMaintenancePlanningInput =
  components["schemas"]["ReviewMaintenancePlanningBody"];
export type MaintenancePlanningEvaluationSuite =
  components["schemas"]["MaintenancePlanningEvaluationSuiteResponse"];
export type CreateMaintenancePlanningEvaluationSuiteInput =
  components["schemas"]["CreateMaintenancePlanningEvaluationSuiteBody"];
export type MaintenancePlanningEvaluationRun =
  components["schemas"]["MaintenancePlanningEvaluationRunResponse"];
export type CreateMaintenancePlanningEvaluationRunInput =
  components["schemas"]["CreateMaintenancePlanningEvaluationRunBody"];
export type SubmitMaintenancePlanningBlindJudgmentInput =
  components["schemas"]["SubmitMaintenancePlanningBlindJudgmentBody"];
export type MaintenancePlanningActivation =
  components["schemas"]["MaintenancePlanningActivationResponse"];
export type CreateMaintenancePlanningActivationInput =
  components["schemas"]["CreateMaintenancePlanningActivationBody"];
export type DecideMaintenancePlanningActivationInput =
  components["schemas"]["DecideMaintenancePlanningActivationBody"];
export type RollbackMaintenancePlanningActivationInput =
  components["schemas"]["RollbackMaintenancePlanningActivationBody"];
export type DeviceFamilyProfile = components["schemas"]["DeviceFamilyResponse"];
export type CreateDeviceFamilyInput = components["schemas"]["CreateDeviceFamilyBody"];
export type ReviewDeviceFamilyInput = components["schemas"]["ReviewDeviceFamilyBody"];
export type RecoveryOverview = components["schemas"]["RecoveryOverviewResponse"];
export type RecoveryDrill = components["schemas"]["RecoveryDrillResponse"];
export type RecoveryComponent = components["schemas"]["RecoveryComponent"];
export type StartRecoveryDrillInput = components["schemas"]["StartRecoveryDrillBody"];
export type AssuranceOverview = components["schemas"]["AssuranceOverviewResponse"];
export type SecurityExercise = components["schemas"]["SecurityExerciseResponse"];
export type ProductionAcceptance = components["schemas"]["ProductionAcceptanceResponse"];
export type AttackScenario = components["schemas"]["AttackScenario"];
export type StartSecurityExerciseInput = components["schemas"]["StartExerciseBody"];
export type CreateProductionAcceptanceInput = components["schemas"]["CreateAcceptanceBody"];
export type SignProductionAcceptanceInput = components["schemas"]["SignAcceptanceBody"];
export type PredictiveMaintenanceOverview = components["schemas"]["PredictiveMaintenanceOverviewResponse"];
export type PredictiveAlertCandidate = components["schemas"]["PredictiveAlertCandidateResponse"];
export type TelemetryWindow = components["schemas"]["TelemetryWindowResponse"];
export type PredictiveOutcome = components["schemas"]["PredictiveOutcomeResponse"];
export type RulForecast = components["schemas"]["RulForecastResponse"];
export type RulForecastCalibration = components["schemas"]["RulForecastCalibrationResponse"];
export type RulReleaseCalibration = components["schemas"]["RulReleaseCalibrationResponse"];
export type RulForecastDecisionInput = components["schemas"]["RulForecastDecisionBody"];
export type BuildTelemetryWindowInput = components["schemas"]["BuildTelemetryWindowBody"];
export type BuildTelemetryDatasetInput = components["schemas"]["BuildTelemetryDatasetBody"];
export type TelemetryDatasetSnapshot = components["schemas"]["TelemetryDatasetSnapshotResponse"];
export type AlertDecisionInput = components["schemas"]["AlertDecisionBody"];
export type PredictiveOutcomeInput = components["schemas"]["PredictiveOutcomeBody"];

export type SecurityAuditQuery = {
  limit?: number;
  cursor?: string;
  decision?: "allow" | "deny";
  action?: string;
};

export interface WorkOrderCompletion {
  root_cause: string;
  actions: string[];
  evidence_ids: string[];
  part_reservation_ids?: string[];
  cost_amount?: string | null;
  customer_confirmation?: string | null;
}

export type DispatchWorkOrderStatus =
  | "READY"
  | "ASSIGNED"
  | "ACCEPTED"
  | "IN_PROGRESS"
  | "ON_HOLD"
  | "COMPLETED"
  | "VERIFIED"
  | "ESCALATED"
  | "CLOSED"
  | "CANCELLED";

export type WorkOrderPriority = "CRITICAL" | "HIGH" | "NORMAL" | "LOW";

export type DispatchWorkOrderQuery = {
  status?: DispatchWorkOrderStatus;
  assigned_subject_id?: string;
  priority?: WorkOrderPriority;
  limit?: number;
  offset?: number;
};

export type FieldWorkOrderQuery = {
  status?: DispatchWorkOrderStatus;
  limit?: number;
  offset?: number;
};

export type CustomerCaseStatus =
  | "SUBMITTED"
  | "NEEDS_INFORMATION"
  | "TRIAGED"
  | "DIAGNOSING"
  | "DIAGNOSED"
  | "WORK_ORDER_CREATED"
  | "RESOLVED"
  | "ESCALATED"
  | "CLOSED"
  | "CANCELLED";

export type CustomerCaseQuery = {
  status?: CustomerCaseStatus;
  limit?: number;
  offset?: number;
};

export type IncidentQueueQuery = {
  status?: CustomerCaseStatus;
  severity?: "P1" | "P2" | "P3" | "P4";
  responsible_queue?: string;
  limit?: number;
  offset?: number;
};

export type FeedbackCandidateStatus =
  | "PENDING_GOVERNANCE"
  | "ELIGIBLE"
  | "ELIGIBILITY_DENIED"
  | "DLP_APPROVED"
  | "DLP_REVIEW_REQUIRED"
  | "ANNOTATION_PENDING"
  | "ANNOTATION_REVIEW_REQUIRED"
  | "ANNOTATION_CONFLICT"
  | "READY_FOR_CURATION";

export type FeedbackEligibilityStatus = "UNDECIDED" | "ELIGIBLE" | "INELIGIBLE";

export type FeedbackCandidateQuery = {
  status?: FeedbackCandidateStatus;
  eligibility_status?: FeedbackEligibilityStatus;
  work_order_id?: string;
  governance_blocker?: GovernanceBlockerCode;
  limit?: number;
  offset?: number;
};

export type CurationRunStatus = "COMPLETED" | "NO_DATA" | "FAILED" | "LINEAGE_PENDING";

export type CurationRunQuery = {
  run_id?: string;
  status?: CurationRunStatus;
  limit?: number;
  offset?: number;
};

export type DatasetSnapshotStatus = "CANDIDATE" | "LINEAGE_PENDING";
export type DatasetLineageStatus = "CONFIRMED" | "PENDING";

export type DatasetManifestDownload = {
  blob: Blob;
  filename: string;
  manifestHash: string;
};

export type ModelEvaluationEvidenceDownload = {
  blob: Blob;
  filename: string;
  evidenceHash: string;
  requestId?: string;
};

export type WithRequestId<T> = T & { requestId: string };

export type DatasetSnapshotQuery = {
  snapshot_id?: string;
  run_id?: string;
  status?: DatasetSnapshotStatus;
  lineage_status?: DatasetLineageStatus;
  training_eligible?: boolean;
  training_blocker?: DatasetTrainingBlockerCode;
  limit?: number;
  offset?: number;
};

export type TrainingExperimentMethod =
  | "BASELINE"
  | "LORA"
  | "QLORA"
  | "DPO"
  | "GRPO"
  | "PPO"
  | "EMBEDDING"
  | "RERANKER"
  | "VLM"
  | "ASR"
  | "TTS"
  | "QUANTIZATION"
  | "TIMESERIES_TRANSFORMER"
  | "TIMESERIES_RULE_BASELINE"
  | "RUL_TRANSFORMER"
  | "RUL_EMPIRICAL_BASELINE";

export type TrainingExperimentStatus =
  | "PLANNED"
  | "TRACKING_PENDING"
  | "TRACKING_FAILED"
  | "RUNNING"
  | "COMPLETION_PENDING"
  | "COMPLETION_FAILED"
  | "COMPLETED"
  | "FAILED"
  | "CLOSED_NO_GAIN";

export type TrainingExperimentQuery = {
  experiment_id?: string;
  dataset_snapshot_id?: string;
  method?: TrainingExperimentMethod;
  status?: TrainingExperimentStatus;
  limit?: number;
  offset?: number;
};

export type EvaluationJobStatus = "PLANNED" | "RUNNING" | "COMPLETED" | "FAILED";
export type EvaluationTargetProfile =
  components["schemas"]["EvaluationJobBody"]["target_profile"];

export type EvaluationJobQuery = {
  job_id?: string;
  candidate_experiment_id?: string;
  baseline_experiment_id?: string;
  status?: EvaluationJobStatus;
  limit?: number;
  offset?: number;
};

export type ModelEvaluationDecision =
  | "CANDIDATE"
  | "REJECTED"
  | "SMOKE_PASSED"
  | "NO_GAIN";

export type ModelEvaluationQuery = {
  evaluation_id?: string;
  candidate_experiment_id?: string;
  baseline_experiment_id?: string;
  decision?: ModelEvaluationDecision;
  limit?: number;
  offset?: number;
};

type ErrorEnvelope = components["schemas"]["ApiErrorEnvelope"];

export class ApiClientError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    readonly category: string,
    message: string,
    readonly retryable: boolean,
    readonly requestId?: string | null,
    readonly details?: Record<string, unknown> | null,
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

type SessionRefresher = () => Promise<boolean>;

let browserSessionRefreshPromise: Promise<boolean> | undefined;

function redirectToExpiredSessionLogin() {
  const callbackUrl =
    window.location.pathname + window.location.search + window.location.hash;
  const query = new URLSearchParams({
    error: "SessionExpired",
    callbackUrl,
  });
  window.location.assign(`/login?${query.toString()}`);
}

async function refreshBrowserSession(): Promise<boolean> {
  if (typeof window === "undefined") return false;
  if (!browserSessionRefreshPromise) {
    browserSessionRefreshPromise = (async () => {
      try {
        const response = await globalThis.fetch("/api/auth/session", {
          cache: "no-store",
          credentials: "same-origin",
        });
        if (!response.ok) return false;
        const session = (await response.json()) as {
          authError?: string;
          user?: unknown;
        };
        if (session.authError || !session.user) {
          redirectToExpiredSessionLogin();
          return false;
        }
        return true;
      } catch {
        return false;
      }
    })().finally(() => {
      browserSessionRefreshPromise = undefined;
    });
  }
  return browserSessionRefreshPromise;
}

async function isRefreshableAuthenticationFailure(
  response: Response,
): Promise<boolean> {
  if (response.status !== 401) return false;
  try {
    const payload = (await response.clone().json()) as {
      error?: { code?: unknown };
    };
    return payload.error?.code === "access_token_expired";
  } catch {
    return false;
  }
}

export function createIndustrialApiClient(
  fetchImplementation: typeof fetch = globalThis.fetch,
  baseUrl = typeof window === "undefined"
    ? "http://localhost/api/backend"
    : "/api/backend",
  sessionRefresher?: SessionRefresher,
): IndustrialApiClient {
  const refreshSession =
    sessionRefresher ??
    (typeof window !== "undefined" &&
    fetchImplementation === globalThis.fetch &&
    baseUrl === "/api/backend"
      ? refreshBrowserSession
      : undefined);
  const normalizedFetch: typeof fetch = async (input, init) => {
    let normalizedInput = input;
    let normalizedInit = init;
    if (input instanceof Request && init === undefined) {
      const hasBody = input.method !== "GET" && input.method !== "HEAD";
      const contentType = input.headers.get("content-type") ?? "";
      const body = hasBody
        ? contentType.includes("application/json")
          ? await input.text()
          : await input.blob()
        : undefined;
      normalizedInput = input.url;
      normalizedInit = {
        method: input.method,
        headers: input.headers,
        body,
        credentials: input.credentials,
        redirect: input.redirect,
        signal: input.signal,
      };
    }
    const response = await fetchImplementation(normalizedInput, normalizedInit);
    if (
      refreshSession &&
      (await isRefreshableAuthenticationFailure(response)) &&
      (await refreshSession())
    ) {
      return fetchImplementation(normalizedInput, normalizedInit);
    }
    return response;
  };
  return createClient<paths>({
    baseUrl,
    fetch: normalizedFetch,
    bodySerializer(body) {
      const value: unknown = body;
      if (value instanceof Blob || value instanceof ArrayBuffer) return value;
      return JSON.stringify(body);
    },
  });
}

export const industrialApi = createIndustrialApiClient();

export async function listAuthorizedAssets(
  client: IndustrialApiClient = industrialApi,
): Promise<{ assets: AssetSummary[]; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/assets");
  if (!data) throw toApiError(response, error);
  return { assets: data.data, requestId: data.meta.request_id };
}

export async function getAuthorizedAsset(
  assetId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ asset: AssetDetail; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/assets/{asset_id}",
    { params: { path: { asset_id: assetId } } },
  );
  if (!data) throw toApiError(response, error);
  return { asset: data.data, requestId: data.meta.request_id };
}

export async function createIncidentDraft(
  input: components["schemas"]["CreateDraftRequest"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ draft: IncidentDraft; requestId: string }> {
  const { data, error, response } = await client.POST("/api/v1/incident-drafts", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return { draft: data.data, requestId: String(data.meta.request_id) };
}

export async function getIncidentDraft(
  draftId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ draft: IncidentDraft; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incident-drafts/{draft_id}",
    { params: { path: { draft_id: draftId } } },
  );
  if (!data) throw toApiError(response, error);
  return { draft: data.data, requestId: String(data.meta.request_id) };
}

export async function updateIncidentDraft(
  draftId: string,
  description: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<{ draft: IncidentDraft; requestId: string }> {
  const { data, error, response } = await client.PATCH(
    "/api/v1/incident-drafts/{draft_id}",
    {
      params: {
        path: { draft_id: draftId },
        header: { "If-Match": `"${version}"` },
      },
      body: { description },
    },
  );
  if (!data) throw toApiError(response, error);
  return { draft: data.data, requestId: String(data.meta.request_id) };
}

export async function getDraftTimeline(
  draftId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ events: TimelineEvent[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incident-drafts/{draft_id}/timeline",
    { params: { path: { draft_id: draftId } } },
  );
  if (!data) throw toApiError(response, error);
  return { events: data.data, requestId: data.meta.request_id };
}

export async function uploadDraftMedia(
  draftId: string,
  file: File,
  client: IndustrialApiClient = industrialApi,
): Promise<{ media: MediaStatus; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incident-drafts/{draft_id}/media",
    {
      params: {
        path: { draft_id: draftId },
        header: { "Content-Type": file.type },
      },
      body: file as unknown as string,
    },
  );
  if (!data) throw toApiError(response, error);
  return { media: data.data, requestId: data.meta.request_id };
}

export async function getMediaStatus(
  mediaId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ media: MediaStatus; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/media/{media_id}/status", {
    params: { path: { media_id: mediaId } },
  });
  if (!data) throw toApiError(response, error);
  return { media: data.data, requestId: data.meta.request_id };
}

export async function startRecognition(
  draftId: string,
  mediaId: string,
  idempotencyKey: string,
  processorProfile: "local-core-v1" | "local-video-v1" | "video-temporal-v1" = "local-core-v1",
  client: IndustrialApiClient = industrialApi,
): Promise<{ recognition: RecognitionRun; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incident-drafts/{draft_id}/recognition-runs",
    {
      params: {
        path: { draft_id: draftId },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: { media_id: mediaId, processor_profile: processorProfile },
    },
  );
  if (!data) throw toApiError(response, error);
  return { recognition: data.data, requestId: String(data.meta.request_id) };
}

export async function getRecognitionRun(
  draftId: string,
  recognitionRunId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ recognition: RecognitionRun; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incident-drafts/{draft_id}/recognition-runs/{recognition_run_id}",
    {
      params: {
        path: {
          draft_id: draftId,
          recognition_run_id: recognitionRunId,
        },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { recognition: data.data, requestId: String(data.meta.request_id) };
}

export async function getDraftEvidence(
  draftId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ evidence: EvidenceBundle; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incident-drafts/{draft_id}/evidence",
    { params: { path: { draft_id: draftId } } },
  );
  if (!data) throw toApiError(response, error);
  return { evidence: data.data, requestId: String(data.meta.request_id) };
}

export async function confirmRecognition(
  draftId: string,
  evidence: EvidenceBundle,
  corrections: Record<string, string>,
  findingDispositions: Record<string, "ACCEPTED" | "REJECTED">,
  transcriptDecisions: Record<
    string,
    components["schemas"]["TranscriptDecisionRequest"]
  > = {},
  ocrBlockDecisionsOrClient: Record<string, IncidentOcrBlockDecision> | IndustrialApiClient = {},
  qrCodeDecisionsOrClient: Record<string, IncidentQrCodeDecision> | IndustrialApiClient = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{ evidence: EvidenceBundle; requestId: string }> {
  const legacyClient = [ocrBlockDecisionsOrClient, qrCodeDecisionsOrClient]
    .find((value) => typeof (value as IndustrialApiClient).POST === "function") as
      IndustrialApiClient | undefined;
  const ocrBlockDecisions = legacyClient
    ? {}
    : ocrBlockDecisionsOrClient as Record<string, IncidentOcrBlockDecision>;
  const qrCodeDecisions = legacyClient
    ? {}
    : qrCodeDecisionsOrClient as Record<string, IncidentQrCodeDecision>;
  const resolvedClient = legacyClient ?? client;
  const { data, error, response } = await resolvedClient.POST(
    "/api/v1/incident-drafts/{draft_id}/recognition-confirmations",
    {
      params: {
        path: { draft_id: draftId },
        header: { "If-Match": `"${evidence.version}"` },
      },
      body: {
        bundle_id: evidence.bundle_id,
        corrections,
        finding_dispositions: findingDispositions,
        transcript_decisions: transcriptDecisions,
        ocr_block_decisions: ocrBlockDecisions,
        qr_code_decisions: qrCodeDecisions,
        video_event_dispositions: Object.fromEntries(
          evidence.video_events
            .filter((event) => [
              "visual_candidate",
              "motion_candidate",
              "signal_pattern_candidate",
              "action_sequence_candidate",
            ].includes(event.event_type))
            .map((event) => [
              event.event_id,
              (evidence.security_findings ?? []).some((finding) => (
                finding.source_type === "VIDEO_EVENT"
                && finding.source_id === event.event_id
              )) ? "REJECTED" as const : "ACCEPTED" as const,
            ]),
        ),
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { evidence: data.data, requestId: String(data.meta.request_id) };
}

export async function submitIncident(
  draftId: string,
  draftVersion: number,
  evidenceBundleId: string,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: Incident; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incident-drafts/{draft_id}/submit",
    {
      params: {
        path: { draft_id: draftId },
        header: {
          "If-Match": `"${draftVersion}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: { evidence_bundle_id: evidenceBundleId },
    },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function getIncident(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: Incident; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function triageIncident(
  incidentId: string,
  incidentVersion: number,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: Incident; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/triage",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "If-Match": `"${incidentVersion}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function listEquipmentControlHandoffs(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ handoffs: EquipmentControlHandoff[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/equipment-control-handoffs",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { handoffs: data.data, requestId: data.meta.request_id };
}

export async function createEquipmentControlHandoff(
  incidentId: string,
  input: EquipmentControlHandoffInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ handoff: EquipmentControlHandoff; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/equipment-control-handoffs",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { handoff: data.data, requestId: data.meta.request_id };
}

export async function proposeCustomerNotification(
  incidentId: string,
  input: CustomerNotificationProposalInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ proposal: CustomerNotificationProposal; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/customer-notification-proposals",
    {
      params: { path: { incident_id: incidentId } },
      body: { ...input, channel: input.channel ?? "PORTAL" },
    },
  );
  if (!data) throw toApiError(response, error);
  return { proposal: data.data, requestId: data.meta.request_id };
}

export async function listCustomerNotificationOptions(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  options: CustomerNotificationOption[];
  externalUnavailableReason: string | null;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/customer-notification-options",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    options: data.data.options,
    externalUnavailableReason: data.data.external_unavailable_reason,
    requestId: data.meta.request_id,
  };
}

export async function proposePurchaseRequest(
  incidentId: string,
  input: PurchaseRequestProposalInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ proposal: PurchaseRequestProposal; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/purchase-request-proposals",
    {
      params: { path: { incident_id: incidentId } },
      body: { ...input, delivery_target: input.delivery_target ?? "INTERNAL" },
    },
  );
  if (!data) throw toApiError(response, error);
  return { proposal: data.data, requestId: data.meta.request_id };
}

export async function proposePartReservation(
  incidentId: string,
  input: PartsReservationProposalInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ proposal: PartsReservationProposal; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/part-reservation-proposals",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { proposal: data.data, requestId: data.meta.request_id };
}

export async function listPurchaseRequestOptions(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  options: PurchaseRequestOption[];
  erpUnavailableReason: string | null;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/purchase-request-options",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    options: data.data.options,
    erpUnavailableReason: data.data.erp_unavailable_reason,
    requestId: data.meta.request_id,
  };
}

export async function listPurchaseRequests(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ purchaseRequests: PurchaseRequest[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/purchase-requests",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { purchaseRequests: data.data, requestId: data.meta.request_id };
}

export async function listServiceQuotationOptions(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  incidentId: string;
  diagnosisRunId: string;
  diagnosisVersion: number;
  entitlementDecision: string;
  options: ServiceQuotationOption[];
  catalogUnavailableReason: string | null;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/service-quotation-options",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    incidentId: data.data.incident_id,
    diagnosisRunId: data.data.diagnosis_run_id,
    diagnosisVersion: data.data.diagnosis_version,
    entitlementDecision: data.data.entitlement_decision,
    options: data.data.options,
    catalogUnavailableReason: data.data.catalog_unavailable_reason,
    requestId: data.meta.request_id,
  };
}

export async function listRepairWorkOrderOptions(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  incidentId: string;
  incidentVersion: number;
  assetId: string;
  diagnosisRunId: string | null;
  diagnosisVersion: number | null;
  entitlementDecision: string | null;
  options: RepairWorkOrderOption[];
  unavailableReason: string | null;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/repair-work-order-options",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    incidentId: data.data.incident_id,
    incidentVersion: data.data.incident_version,
    assetId: data.data.asset_id,
    diagnosisRunId: data.data.diagnosis_run_id,
    diagnosisVersion: data.data.diagnosis_version,
    entitlementDecision: data.data.entitlement_decision,
    options: data.data.options,
    unavailableReason: data.data.unavailable_reason,
    requestId: String(data.meta.request_id),
  };
}

export async function proposeRepairWorkOrder(
  incidentId: string,
  incidentVersion: number,
  authorizationType: RepairWorkOrderOption["authorization_type"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ proposal: RepairWorkOrderProposal; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/repair-work-order-proposals",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: { incident_version: incidentVersion, authorization_type: authorizationType },
    },
  );
  if (!data) throw toApiError(response, error);
  return { proposal: data.data, requestId: String(data.meta.request_id) };
}

export async function listServiceQuotations(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ quotations: ServiceQuotation[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/service-quotations",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { quotations: data.data, requestId: data.meta.request_id };
}

export async function proposeServiceQuotation(
  incidentId: string,
  candidateId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ proposal: ServiceQuotationProposal; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/service-quotation-proposals",
    {
      params: { path: { incident_id: incidentId } },
      body: { candidate_id: candidateId },
    },
  );
  if (!data) throw toApiError(response, error);
  return { proposal: data.data, requestId: data.meta.request_id };
}

export async function proposeRefundRequest(
  workOrderId: string,
  input: RefundRequestProposalInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ proposal: RefundRequestProposal; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/work-orders/{work_order_id}/refund-request-proposals",
    {
      params: { path: { work_order_id: workOrderId } },
      body: { ...input, delivery_target: input.delivery_target ?? "INTERNAL" },
    },
  );
  if (!data) throw toApiError(response, error);
  return { proposal: data.data, requestId: data.meta.request_id };
}

export async function listRefundRequestOptions(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  options: RefundRequestOption[];
  financeUnavailableReason: string | null;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/work-orders/{work_order_id}/refund-request-options",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    options: data.data.options,
    financeUnavailableReason: data.data.finance_unavailable_reason,
    requestId: data.meta.request_id,
  };
}

export async function listRefundRequests(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ refundRequests: RefundRequest[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/refund-requests",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { refundRequests: data.data, requestId: data.meta.request_id };
}

export async function listIncidentQueue(
  query: IncidentQueueQuery = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{ incidents: IncidentQueueItem[]; total: number; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/incidents", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    incidents: data.data,
    total: Number(data.meta.total),
    requestId: String(data.meta.request_id),
  };
}

export async function listIncidentControls(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ controls: IncidentControl[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/controls",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { controls: data.data, requestId: data.meta.request_id };
}

export async function getIncidentOperations(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: IncidentOperations; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/operations",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function getIncidentResolution(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ resolution: IncidentResolution | null; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/resolution",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { resolution: data.data, requestId: data.meta.request_id };
}

export async function resolveIncidentRemote(
  incidentId: string,
  version: number,
  input: ResolveRemoteIncidentInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: IncidentOperations; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/resolution/remote",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "If-Match": `"${version}"` },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function closeIncident(
  incidentId: string,
  version: number,
  input: CloseIncidentInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: IncidentOperations; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/closure",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "If-Match": `"${version}"` },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function classifyIncident(
  incidentId: string,
  version: number,
  input: ClassifyIncidentInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: IncidentOperations; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/classification",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "If-Match": `"${version}"` },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function requestIncidentInformation(
  incidentId: string,
  version: number,
  input: RequestIncidentInformationInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: IncidentOperations; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/request-information",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "If-Match": `"${version}"` },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function markIncidentInformationReceived(
  incidentId: string,
  version: number,
  reason: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: IncidentOperations; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/information-received",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "If-Match": `"${version}"` },
      },
      body: { reason },
    },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function escalateIncident(
  incidentId: string,
  version: number,
  input: EscalateIncidentInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: IncidentOperations; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/escalation",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "If-Match": `"${version}"` },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function resumeIncident(
  incidentId: string,
  version: number,
  input: ResumeIncidentInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: IncidentOperations; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/resume",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "If-Match": `"${version}"` },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function mergeDuplicateIncident(
  incidentId: string,
  version: number,
  input: MergeDuplicateIncidentInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ incident: IncidentOperations; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/duplicate-merge",
    {
      params: {
        path: { incident_id: incidentId },
        header: { "If-Match": `"${version}"` },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { incident: data.data, requestId: String(data.meta.request_id) };
}

export async function getTenantAdministrationOverview(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  tenant: TenantOverview;
  retentionPolicy: TenantRetentionPolicy;
  assignableRoles: string[];
  assets: TenantAssetOption[];
  sites: TenantSiteOption[];
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/tenant");
  if (!data) throw toApiError(response, error);
  return {
    tenant: data.data,
    retentionPolicy: data.retention_policy,
    assignableRoles: data.assignable_roles,
    assets: data.assets,
    sites: data.sites,
    requestId: data.meta.request_id,
  };
}

export async function listTenantMembers(
  query: { status?: string; role?: string; limit?: number; offset?: number } = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{ members: TenantMember[]; total: number; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/tenant/members", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    members: data.data,
    total: Number(data.meta.total),
    requestId: String(data.meta.request_id),
  };
}

export async function createTenantMember(
  input: CreateTenantMemberInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ member: TenantMember; requestId: string }> {
  const { data, error, response } = await client.POST("/api/v1/tenant/members", {
    params: { header: { "Idempotency-Key": idempotencyKey } },
    body: input,
  });
  if (!data) throw toApiError(response, error);
  return { member: data.data, requestId: String(data.meta.request_id) };
}

export async function updateTenantMember(
  subjectId: string,
  version: number,
  input: UpdateTenantMemberInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ member: TenantMember; requestId: string }> {
  const { data, error, response } = await client.PUT(
    "/api/v1/tenant/members/{subject_id}",
    {
      params: {
        path: { subject_id: subjectId },
        header: {
          "If-Match": `"${version}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { member: data.data, requestId: String(data.meta.request_id) };
}

export async function updateTenantRetentionPolicy(
  version: number,
  input: UpdateTenantRetentionPolicyInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ retentionPolicy: TenantRetentionPolicy }> {
  const { data, error, response } = await client.PUT(
    "/api/v1/tenant/retention-policy",
    {
      params: {
        header: {
          "If-Match": `"${version}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { retentionPolicy: data };
}

export async function listTenantAdministrationRecords(
  query: { limit?: number; offset?: number } = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  records: TenantAdministrationRecord[];
  total: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/tenant/administration-records",
    { params: { query } },
  );
  if (!data) throw toApiError(response, error);
  return {
    records: data.data,
    total: Number(data.meta.total),
    requestId: String(data.meta.request_id),
  };
}

export async function startDiagnosis(
  incidentId: string,
  version: number,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ diagnosis: DiagnosisRun; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/diagnoses",
    {
      params: {
        path: { incident_id: incidentId },
        header: {
          "If-Match": `"${version}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { diagnosis: data.data, requestId: String(data.meta.request_id) };
}

export async function getDiagnosis(
  diagnosisRunId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ diagnosis: DiagnosisRun; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/diagnosis-runs/{diagnosis_run_id}",
    { params: { path: { diagnosis_run_id: diagnosisRunId } } },
  );
  if (!data) throw toApiError(response, error);
  return { diagnosis: data.data, requestId: String(data.meta.request_id) };
}

export async function listDiagnosisFeedback(
  diagnosisRunId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  feedback: DiagnosisQualityFeedback[];
  canSubmit: boolean;
  total: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/diagnosis-runs/{diagnosis_run_id}/feedback",
    { params: { path: { diagnosis_run_id: diagnosisRunId } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    feedback: data.data,
    canSubmit: Boolean(data.meta.can_submit),
    total: Number(data.meta.total ?? data.data.length),
    requestId: String(data.meta.request_id),
  };
}

export async function createDiagnosisFeedback(
  input: DiagnosisQualityFeedbackInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  feedback: DiagnosisQualityFeedback;
  created: boolean;
  requestId: string;
}> {
  const { data, error, response } = await client.POST("/api/v1/feedback", {
    params: { header: { "Idempotency-Key": idempotencyKey } },
    body: input,
  });
  if (!data) throw toApiError(response, error);
  return {
    feedback: data.data,
    created: Boolean(data.meta.created),
    requestId: String(data.meta.request_id),
  };
}

export async function reanalyzeDiagnosisWithTranscript(
  diagnosisRunId: string,
  transcriptSegmentId: string,
  version: number,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ diagnosis: DiagnosisRun; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/diagnosis-runs/{diagnosis_run_id}/reanalyses",
    {
      params: {
        path: { diagnosis_run_id: diagnosisRunId },
        header: {
          "If-Match": `"${version}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: { transcript_segment_id: transcriptSegmentId },
    },
  );
  if (!data) throw toApiError(response, error);
  return { diagnosis: data.data, requestId: String(data.meta.request_id) };
}

export async function reanalyzeDiagnosisWithFieldObservations(
  diagnosisRunId: string,
  workOrderId: string,
  workOrderVersion: number,
  fieldEntryIds: string[],
  version: number,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ diagnosis: DiagnosisRun; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/diagnosis-runs/{diagnosis_run_id}/reanalyses",
    {
      params: {
        path: { diagnosis_run_id: diagnosisRunId },
        header: {
          "If-Match": `"${version}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: {
        work_order_id: workOrderId,
        work_order_version: workOrderVersion,
        field_entry_ids: fieldEntryIds,
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { diagnosis: data.data, requestId: String(data.meta.request_id) };
}

export async function cancelAgentRun(
  agentRunId: string,
  version: number,
  reason: string,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ agentRun: AgentRunControl; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/agent-runs/{agent_run_id}/cancel",
    {
      params: {
        path: { agent_run_id: agentRunId },
        header: {
          "If-Match": `"${version}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: { reason },
    },
  );
  if (!data) throw toApiError(response, error);
  return { agentRun: data.data, requestId: String(data.meta.request_id) };
}

export async function resumeAgentRun(
  agentRunId: string,
  version: number,
  clarification: string,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ diagnosis: DiagnosisRun; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/agent-runs/{agent_run_id}/resume",
    {
      params: {
        path: { agent_run_id: agentRunId },
        header: {
          "If-Match": `"${version}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: { clarification },
    },
  );
  if (!data) throw toApiError(response, error);
  return { diagnosis: data.data, requestId: String(data.meta.request_id) };
}

export async function takeOverDiagnosis(
  diagnosisRunId: string,
  version: number,
  reason: string,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ diagnosis: DiagnosisRun; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/diagnosis-runs/{diagnosis_run_id}/expert-interventions",
    {
      params: {
        path: { diagnosis_run_id: diagnosisRunId },
        header: {
          "If-Match": `"${version}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: { reason },
    },
  );
  if (!data) throw toApiError(response, error);
  return { diagnosis: data.data, requestId: String(data.meta.request_id) };
}

export async function submitExpertDiagnosisRevision(
  diagnosisRunId: string,
  interventionId: string,
  interventionVersion: number,
  input: components["schemas"]["ExpertDiagnosisRevisionRequest"],
  client: IndustrialApiClient = industrialApi,
): Promise<{ diagnosis: DiagnosisRun; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/diagnosis-runs/{diagnosis_run_id}/expert-interventions/{intervention_id}/revisions",
    {
      params: {
        path: {
          diagnosis_run_id: diagnosisRunId,
          intervention_id: interventionId,
        },
        header: { "If-Match": `"${interventionVersion}"` },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { diagnosis: data.data, requestId: String(data.meta.request_id) };
}

export async function synthesizeDiagnosisSpeech(
  diagnosisRunId: string,
  safetyAcknowledged: boolean,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ speech: SpeechSynthesis; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/diagnosis-runs/{diagnosis_run_id}/speech",
    {
      params: {
        path: { diagnosis_run_id: diagnosisRunId },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: { safety_acknowledged: safetyAcknowledged, voice: "default" },
    },
  );
  if (!data) throw toApiError(response, error);
  return { speech: data.data, requestId: String(data.meta.request_id) };
}

export async function createRealtimeSession(
  incidentId: string,
  diagnosisRunId?: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ session: RealtimeSessionGrant; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/incidents/{incident_id}/realtime-sessions",
    {
      params: { path: { incident_id: incidentId } },
      body: { diagnosis_run_id: diagnosisRunId ?? null, media_kind: "audio" },
    },
  );
  if (!data) throw toApiError(response, error);
  return { session: data.data, requestId: String(data.meta.request_id) };
}

export async function getFieldVoiceGuidance(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ guidance: FieldVoiceGuidance; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/field/work-orders/{work_order_id}/voice-guidance",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return { guidance: data.data, requestId: String(data.meta.request_id) };
}

export async function createFieldRealtimeSession(
  workOrderId: string,
  workOrderVersion: number,
  diagnosisRunId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ session: RealtimeSessionGrant; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/realtime-sessions",
    {
      params: { path: { work_order_id: workOrderId } },
      body: {
        diagnosis_run_id: diagnosisRunId,
        work_order_version: workOrderVersion,
        media_kind: "audio",
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { session: data.data, requestId: String(data.meta.request_id) };
}

export async function synthesizeFieldVoiceGuidance(
  workOrderId: string,
  workOrderVersion: number,
  diagnosisRunId: string,
  safetyAcknowledged: boolean,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ speech: FieldSpeechSynthesis; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/voice-guidance/speech",
    {
      params: {
        path: { work_order_id: workOrderId },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: {
        diagnosis_run_id: diagnosisRunId,
        work_order_version: workOrderVersion,
        safety_acknowledged: safetyAcknowledged,
        voice: "default",
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { speech: data.data, requestId: String(data.meta.request_id) };
}

export async function reconnectRealtimeSession(
  sessionId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ session: RealtimeSessionGrant; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/realtime-sessions/{session_id}/reconnect",
    { params: { path: { session_id: sessionId } } },
  );
  if (!data) throw toApiError(response, error);
  return { session: data.data, requestId: String(data.meta.request_id) };
}

export async function heartbeatRealtimeSession(
  sessionId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ session: RealtimeSession; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/realtime-sessions/{session_id}/heartbeat",
    { params: { path: { session_id: sessionId } } },
  );
  if (!data) throw toApiError(response, error);
  return { session: data.data, requestId: String(data.meta.request_id) };
}

export async function endRealtimeSession(
  sessionId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ session: RealtimeSession; requestId: string }> {
  const { data, error, response } = await client.DELETE(
    "/api/v1/realtime-sessions/{session_id}",
    { params: { path: { session_id: sessionId } } },
  );
  if (!data) throw toApiError(response, error);
  return { session: data.data, requestId: String(data.meta.request_id) };
}

export async function getFieldExpertCollaboration(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ overview: ExpertCollaborationOverview; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/field/work-orders/{work_order_id}/expert-collaboration",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return { overview: data.data, requestId: String(data.meta.request_id) };
}

export async function createExpertCollaboration(
  workOrderId: string,
  workOrderVersion: number,
  invitedSubjectId: string,
  reason: string,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ collaboration: ExpertCollaboration; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/expert-collaborations",
    {
      params: {
        path: { work_order_id: workOrderId },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: {
        work_order_version: workOrderVersion,
        invited_subject_id: invitedSubjectId,
        reason,
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { collaboration: data.data, requestId: String(data.meta.request_id) };
}

export async function listExpertCollaborations(
  client: IndustrialApiClient = industrialApi,
): Promise<{ collaborations: ExpertCollaboration[]; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/expert-collaborations");
  if (!data) throw toApiError(response, error);
  return { collaborations: data.data, requestId: String(data.meta.request_id) };
}

export async function getExpertCollaboration(
  collaborationId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ collaboration: ExpertCollaboration; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/expert-collaborations/{collaboration_id}",
    { params: { path: { collaboration_id: collaborationId } } },
  );
  if (!data) throw toApiError(response, error);
  return { collaboration: data.data, requestId: String(data.meta.request_id) };
}

export async function acceptExpertCollaboration(
  collaborationId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<{ collaboration: ExpertCollaboration; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/expert-collaborations/{collaboration_id}/accept",
    {
      params: {
        path: { collaboration_id: collaborationId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { collaboration: data.data, requestId: String(data.meta.request_id) };
}

export async function endExpertCollaboration(
  collaborationId: string,
  version: number,
  reason: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ collaboration: ExpertCollaboration; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/expert-collaborations/{collaboration_id}/end",
    {
      params: {
        path: { collaboration_id: collaborationId },
        header: { "If-Match": `"${version}"` },
      },
      body: { reason },
    },
  );
  if (!data) throw toApiError(response, error);
  return { collaboration: data.data, requestId: String(data.meta.request_id) };
}

export async function submitExpertRecommendation(
  collaborationId: string,
  input: SubmitExpertRecommendationInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ collaboration: ExpertCollaboration; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/expert-collaborations/{collaboration_id}/recommendations",
    {
      params: {
        path: { collaboration_id: collaborationId },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { collaboration: data.data, requestId: String(data.meta.request_id) };
}

export async function decideExpertRecommendation(
  collaborationId: string,
  version: number,
  decision: "ACCEPT" | "REJECT",
  reason: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ collaboration: ExpertCollaboration; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/expert-collaborations/{collaboration_id}/recommendation/decision",
    {
      params: {
        path: { collaboration_id: collaborationId },
        header: { "If-Match": `"${version}"` },
      },
      body: { decision, reason },
    },
  );
  if (!data) throw toApiError(response, error);
  return { collaboration: data.data, requestId: String(data.meta.request_id) };
}

export async function createExpertCollaborationRealtimeSession(
  collaborationId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ session: RealtimeSessionGrant; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/expert-collaborations/{collaboration_id}/realtime-sessions",
    { params: { path: { collaboration_id: collaborationId } } },
  );
  if (!data) throw toApiError(response, error);
  return { session: data.data, requestId: String(data.meta.request_id) };
}

export async function confirmRealtimeTranscript(
  segmentId: string,
  decision: "ACCEPTED" | "REJECTED",
  correctedText: string | null,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<{ transcript: RealtimeTranscript; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/realtime-transcripts/{segment_id}/confirmation",
    {
      params: {
        path: { segment_id: segmentId },
        header: { "If-Match": `"${version}"` },
      },
      body: { decision, corrected_text: correctedText },
    },
  );
  if (!data) throw toApiError(response, error);
  return { transcript: data.data, requestId: String(data.meta.request_id) };
}

export async function getCitation(
  citationId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ citation: Citation; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/citations/{citation_id}",
    { params: { path: { citation_id: citationId } } },
  );
  if (!data) throw toApiError(response, error);
  return { citation: data.data, requestId: String(data.meta.request_id) };
}

export async function downloadCitationSource(
  citationId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<CitationSourceDownload> {
  const { data, error, response } = await client.GET(
    "/api/v1/citations/{citation_id}/source",
    {
      params: { path: { citation_id: citationId } },
      parseAs: "blob",
    },
  );
  if (!data) {
    let errorEnvelope: unknown = error;
    if (error instanceof Blob) {
      try {
        errorEnvelope = JSON.parse(await error.text()) as unknown;
      } catch {
        // Preserve the original Blob for the generic API error fallback.
      }
    }
    throw toApiError(response, errorEnvelope);
  }
  const expectedChecksum = response.headers
    .get("etag")
    ?.replace(/^W\//, "")
    .replace(/^"|"$/g, "");
  const sourceChecksum = await verifyCitationSourceBlob(data, expectedChecksum, response);
  return {
    blob: data,
    sourceChecksum,
    requestId: response.headers.get("x-request-id") ?? undefined,
  };
}

async function verifyCitationSourceBlob(
  blob: Blob,
  expectedChecksum: string | undefined,
  response: Response,
): Promise<string> {
  const failure = () => new ApiClientError(
    503,
    "citation_source_integrity_failed",
    "dependency",
    "Citation source integrity verification failed",
    false,
    response.headers.get("x-request-id"),
  );
  if (!expectedChecksum || !/^[0-9a-f]{64}$/.test(expectedChecksum)) throw failure();
  try {
    const digest = await globalThis.crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
    const actualChecksum = Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, "0")).join("");
    if (actualChecksum !== expectedChecksum) throw failure();
  } catch (error) {
    if (error instanceof ApiClientError) throw error;
    throw failure();
  }
  return expectedChecksum;
}

export async function listApprovals(
  client: IndustrialApiClient = industrialApi,
): Promise<{ approvals: ApprovalView[]; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/approvals");
  const envelope = unwrapGenericEnvelope<ApprovalView[]>(response, data, error);
  return { approvals: envelope.data, requestId: envelope.meta.request_id };
}

export async function getApproval(
  approvalId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ approval: ApprovalView; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/approvals/{approval_id}",
    { params: { path: { approval_id: approvalId } } },
  );
  const envelope = unwrapGenericEnvelope<ApprovalView>(response, data, error);
  return { approval: envelope.data, requestId: envelope.meta.request_id };
}

export async function decideApproval(
  approvalId: string,
  version: number,
  decision: "APPROVED" | "REJECTED",
  reason: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ approval: ApprovalView; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/approvals/{approval_id}/decisions",
    {
      params: {
        path: { approval_id: approvalId },
        header: { "If-Match": `"${version}"` },
      },
      body: { decision, reason },
    },
  );
  const envelope = unwrapGenericEnvelope<ApprovalView>(response, data, error);
  return { approval: envelope.data, requestId: envelope.meta.request_id };
}

export async function executeApproval(
  approvalId: string,
  version: number,
  parameters: Record<string, unknown>,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ execution: ApprovalExecution; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/approvals/{approval_id}/execute",
    {
      params: {
        path: { approval_id: approvalId },
        header: {
          "If-Match": `"${version}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: { parameters },
    },
  );
  const envelope = unwrapGenericEnvelope<ApprovalExecution>(response, data, error);
  return { execution: envelope.data, requestId: envelope.meta.request_id };
}

export async function listExecutionReconciliations(
  status?: "PENDING" | "RESOLVED" | "ESCALATED",
  client: IndustrialApiClient = industrialApi,
): Promise<{ reconciliations: ExecutionReconciliation[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/execution-reconciliations",
    { params: { query: { status } } },
  );
  const envelope = unwrapGenericEnvelope<ExecutionReconciliation[]>(response, data, error);
  return { reconciliations: envelope.data, requestId: String(envelope.meta.request_id) };
}

export async function getExecutionReconciliation(
  reconciliationId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  reconciliation: ExecutionReconciliation;
  events: ExecutionReconciliationEvent[];
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/execution-reconciliations/{reconciliation_id}",
    { params: { path: { reconciliation_id: reconciliationId } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    reconciliation: data.data.reconciliation,
    events: data.data.events,
    requestId: data.meta.request_id,
  };
}

export async function checkExecutionReconciliation(
  reconciliationId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<{ reconciliation: ExecutionReconciliation; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/execution-reconciliations/{reconciliation_id}/check",
    {
      params: {
        path: { reconciliation_id: reconciliationId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { reconciliation: data.data, requestId: data.meta.request_id };
}

export async function escalateExecutionReconciliation(
  reconciliationId: string,
  version: number,
  reason: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ reconciliation: ExecutionReconciliation; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/execution-reconciliations/{reconciliation_id}/escalate",
    {
      params: {
        path: { reconciliation_id: reconciliationId },
        header: { "If-Match": `"${version}"` },
      },
      body: { reason },
    },
  );
  if (!data) throw toApiError(response, error);
  return { reconciliation: data.data, requestId: data.meta.request_id };
}

export async function getWorkOrder(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ workOrder: WorkOrderView; requestId: string }> {
  return workOrderRequest("GET", workOrderId, "", undefined, undefined, client);
}

export async function getWorkOrderRepairHistory(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ history: WorkOrderRepairHistory; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/work-orders/{work_order_id}/repair-history",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return { history: data.data, requestId: data.meta.request_id };
}

export async function getWorkOrderClosureReport(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ report: WorkOrderClosureReport; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/work-orders/{work_order_id}/closure-report",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return { report: data.data, requestId: data.meta.request_id };
}

export async function listWorkOrders(
  query: DispatchWorkOrderQuery = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  workOrders: WorkOrderCenterItem[];
  total: number;
  limit: number;
  offset: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/work-orders", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    workOrders: data.data,
    total: data.meta.total,
    limit: data.meta.limit,
    offset: data.meta.offset,
    requestId: data.meta.request_id,
  };
}

export async function listMyFieldWorkOrders(
  query: FieldWorkOrderQuery = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  workOrders: WorkOrderCenterItem[];
  total: number;
  limit: number;
  offset: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/field/work-orders", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    workOrders: data.data,
    total: data.meta.total,
    limit: data.meta.limit,
    offset: data.meta.offset,
    requestId: data.meta.request_id,
  };
}

export async function listCustomerCases(
  query: CustomerCaseQuery = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  cases: CustomerCase[];
  total: number;
  limit: number;
  offset: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/portal/cases", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    cases: data.data,
    total: data.meta.total ?? data.data.length,
    limit: data.meta.limit ?? data.data.length,
    offset: data.meta.offset ?? 0,
    requestId: data.meta.request_id,
  };
}

export async function lookupSpareParts(
  assetId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ lookup: SparePartsLookup; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/parts/lookup", {
    params: { query: { asset_id: assetId } },
  });
  if (!data) throw toApiError(response, error);
  return { lookup: data.data, requestId: data.meta.request_id };
}

export async function lookupServiceEntitlement(
  assetId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ entitlement: ServiceEntitlement; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/service-entitlements/lookup",
    { params: { query: { asset_id: assetId } } },
  );
  if (!data) throw toApiError(response, error);
  return { entitlement: data.data, requestId: data.meta.request_id };
}

export async function getCustomerCase(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ customerCase: CustomerCase; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/portal/cases/{incident_id}",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { customerCase: data.data, requestId: data.meta.request_id };
}

export async function getCustomerServiceQuotation(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ quotation: CustomerServiceQuotationView; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/portal/cases/{incident_id}/service-quotation",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { quotation: data.data, requestId: data.meta.request_id };
}

export async function decideCustomerServiceQuotation(
  incidentId: string,
  decision: "ACCEPTED" | "REJECTED",
  stateVersion: number,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ decision: CustomerServiceQuotationDecision; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/portal/cases/{incident_id}/service-quotation/decision",
    {
      params: {
        path: { incident_id: incidentId },
        header: {
          "If-Match": `"${stateVersion}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: { decision },
    },
  );
  if (!data) throw toApiError(response, error);
  return { decision: data.data, requestId: data.meta.request_id };
}

export async function listCustomerCaseUpdates(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ updates: CustomerCaseUpdate[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/portal/cases/{incident_id}/updates",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { updates: data.data, requestId: data.meta.request_id };
}

export async function listIncidentCustomerUpdates(
  incidentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ updates: CustomerCaseUpdate[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/incidents/{incident_id}/customer-updates",
    { params: { path: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { updates: data.data, requestId: data.meta.request_id };
}

export async function appendCustomerCaseUpdate(
  incidentId: string,
  input: CustomerCaseUpdateInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ update: CustomerCaseUpdate; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/portal/cases/{incident_id}/updates",
    {
      params: { path: { incident_id: incidentId } },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { update: data.data, requestId: data.meta.request_id };
}

export async function listFieldWorkOrderEntries(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ entries: FieldWorkOrderEntry[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/field/work-orders/{work_order_id}/entries",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return { entries: data.data, requestId: data.meta.request_id };
}

export async function listFieldEvidenceUploads(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ uploads: FieldEvidenceUpload[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/field/work-orders/{work_order_id}/evidence-uploads",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    uploads: data.data,
    requestId: String(data.meta.request_id),
  };
}

export async function uploadFieldEvidence(
  workOrderId: string,
  workOrderVersion: number,
  clientOperationId: string,
  contentHash: string,
  declaredMime: string,
  content: Blob,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  upload: FieldEvidenceUpload;
  requestId: string;
  created: boolean;
}> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/evidence-uploads",
    {
      params: {
        path: { work_order_id: workOrderId },
        header: {
          "Content-Type": declaredMime,
          "If-Match": `"${workOrderVersion}"`,
          "Idempotency-Key": clientOperationId,
          "X-Content-SHA256": contentHash,
        },
      },
      body: content as unknown as string,
    },
  );
  if (!data) throw toApiError(response, error);
  return {
    upload: data.data,
    requestId: String(data.meta.request_id),
    created: Boolean(data.meta.created),
  };
}

export async function startFieldEvidenceRecognition(
  workOrderId: string,
  evidenceUploadId: string,
  workOrderVersion: number,
  idempotencyKey: string,
  temporalAnalysis = false,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  recognition: FieldRecognitionRun;
  requestId: string;
  created: boolean;
}> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/evidence-uploads/{evidence_upload_id}/recognitions",
    {
      params: {
        path: {
          work_order_id: workOrderId,
          evidence_upload_id: evidenceUploadId,
        },
        header: {
          "If-Match": `"${workOrderVersion}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: { temporal_analysis: temporalAnalysis },
    },
  );
  if (!data) throw toApiError(response, error);
  return {
    recognition: data.data,
    requestId: String(data.meta.request_id),
    created: Boolean(data.meta.created),
  };
}

export async function getFieldEvidenceRecognition(
  workOrderId: string,
  evidenceUploadId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ evidence: EvidenceBundle; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/field/work-orders/{work_order_id}/evidence-uploads/{evidence_upload_id}/recognitions",
    {
      params: {
        path: {
          work_order_id: workOrderId,
          evidence_upload_id: evidenceUploadId,
        },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { evidence: data.data, requestId: String(data.meta.request_id) };
}

export async function confirmFieldEvidenceRecognition(
  workOrderId: string,
  evidenceUploadId: string,
  evidence: EvidenceBundle,
  corrections: Record<string, string>,
  findingDispositions: Record<string, "ACCEPTED" | "REJECTED">,
  ocrBlockDecisions: Record<string, FieldOcrBlockDecision>,
  transcriptDecisions: Record<string, FieldTranscriptDecision>,
  videoEventDispositions: Record<string, "ACCEPTED" | "REJECTED">,
  qrCodeDecisionsOrClient: Record<string, FieldQrCodeDecision> | IndustrialApiClient = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{ evidence: EvidenceBundle; requestId: string }> {
  const legacyClient = typeof (qrCodeDecisionsOrClient as IndustrialApiClient).POST === "function"
    ? qrCodeDecisionsOrClient as IndustrialApiClient
    : undefined;
  const qrCodeDecisions = legacyClient
    ? {}
    : qrCodeDecisionsOrClient as Record<string, FieldQrCodeDecision>;
  const { data, error, response } = await (legacyClient ?? client).POST(
    "/api/v1/field/work-orders/{work_order_id}/evidence-uploads/{evidence_upload_id}/recognitions/confirmation",
    {
      params: {
        path: {
          work_order_id: workOrderId,
          evidence_upload_id: evidenceUploadId,
        },
        header: { "If-Match": `"${evidence.version}"` },
      },
      body: {
        corrections,
        finding_dispositions: findingDispositions,
        ocr_block_decisions: ocrBlockDecisions,
        transcript_decisions: transcriptDecisions,
        video_event_dispositions: videoEventDispositions,
        qr_code_decisions: qrCodeDecisions,
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { evidence: data.data, requestId: String(data.meta.request_id) };
}

export async function createFieldOfflinePack(
  workOrderId: string,
  workOrderVersion: number,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  pack: WorkOrderOfflinePack;
  requestId: string;
  created: boolean;
}> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/offline-packs",
    {
      params: {
        path: { work_order_id: workOrderId },
        header: {
          "If-Match": `"${workOrderVersion}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return {
    pack: data.data,
    requestId: String(data.meta.request_id),
    created: Boolean(data.meta.created),
  };
}

export async function getCurrentFieldOfflinePack(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ pack: WorkOrderOfflinePack; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/field/work-orders/{work_order_id}/offline-packs/current",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return { pack: data.data, requestId: String(data.meta.request_id) };
}

export async function revokeFieldOfflinePack(
  workOrderId: string,
  packId: string,
  packVersion: number,
  reason: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ pack: WorkOrderOfflinePack; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/offline-packs/{pack_id}/revoke",
    {
      params: {
        path: { work_order_id: workOrderId, pack_id: packId },
        header: { "If-Match": `"${packVersion}"` },
      },
      body: { reason },
    },
  );
  if (!data) throw toApiError(response, error);
  return { pack: data.data, requestId: String(data.meta.request_id) };
}

export async function createFieldEdgeDiagnosisPack(
  workOrderId: string,
  workOrderVersion: number,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  pack: FieldEdgeDiagnosisPack;
  requestId: string;
  created: boolean;
}> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/edge-diagnosis-packs",
    {
      params: {
        path: { work_order_id: workOrderId },
        header: {
          "If-Match": `"${workOrderVersion}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return {
    pack: data.data,
    requestId: String(data.meta.request_id),
    created: Boolean(data.meta.created),
  };
}

export async function listFieldEdgeDiagnosisCandidates(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ candidates: FieldEdgeDiagnosisCandidate[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/field/work-orders/{work_order_id}/edge-diagnosis-candidates",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return { candidates: data.data, requestId: String(data.meta.request_id) };
}

export async function importFieldEdgeDiagnosisCandidate(
  workOrderId: string,
  clientOperationId: string,
  result: FieldEdgeDiagnosisResult,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  candidate: FieldEdgeDiagnosisCandidate;
  requestId: string;
  created: boolean;
}> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/edge-diagnosis-candidates/import",
    {
      params: { path: { work_order_id: workOrderId } },
      body: { client_operation_id: clientOperationId, result },
    },
  );
  if (!data) throw toApiError(response, error);
  return {
    candidate: data.data,
    requestId: String(data.meta.request_id),
    created: Boolean(data.meta.created),
  };
}

export async function decideFieldEdgeDiagnosisCandidate(
  workOrderId: string,
  candidateId: string,
  version: number,
  decision: "ACCEPTED" | "REJECTED",
  reason: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ candidate: FieldEdgeDiagnosisCandidate; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/edge-diagnosis-candidates/{candidate_id}/decision",
    {
      params: {
        path: { work_order_id: workOrderId, candidate_id: candidateId },
        header: { "If-Match": `"${version}"` },
      },
      body: { decision, reason },
    },
  );
  if (!data) throw toApiError(response, error);
  return { candidate: data.data, requestId: String(data.meta.request_id) };
}

export async function getWorkOrderPartAllocation(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ allocation: WorkOrderPartAllocation; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/field/work-orders/{work_order_id}/part-allocation",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return { allocation: data.data, requestId: data.meta.request_id };
}

export async function getWorkOrderPartAccounting(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ accounting: WorkOrderPartAccounting; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/field/work-orders/{work_order_id}/part-accounting",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return { accounting: data.data, requestId: data.meta.request_id };
}

export async function proposeWorkOrderPartIssue(
  workOrderId: string,
  workOrderVersion: number,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ proposal: PartIssueProposal; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/part-issue-proposals",
    {
      params: {
        path: { work_order_id: workOrderId },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: { work_order_version: workOrderVersion },
    },
  );
  if (!data) throw toApiError(response, error);
  return { proposal: data.data, requestId: data.meta.request_id };
}

export async function proposeWorkOrderPartConsumption(
  workOrderId: string,
  workOrderVersion: number,
  fieldEntryId: string,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ proposal: PartMaterialMovementProposal; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/part-consumption-proposals",
    {
      params: {
        path: { work_order_id: workOrderId },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: {
        work_order_version: workOrderVersion,
        field_entry_id: fieldEntryId,
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { proposal: data.data, requestId: data.meta.request_id };
}

export async function proposeWorkOrderPartReturn(
  workOrderId: string,
  workOrderVersion: number,
  quantity: number,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ proposal: PartMaterialMovementProposal; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/part-return-proposals",
    {
      params: {
        path: { work_order_id: workOrderId },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: { work_order_version: workOrderVersion, quantity },
    },
  );
  if (!data) throw toApiError(response, error);
  return { proposal: data.data, requestId: data.meta.request_id };
}

export async function appendFieldWorkOrderEntry(
  workOrderId: string,
  input: FieldEntryInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ entry: FieldWorkOrderEntry; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/entries",
    {
      params: { path: { work_order_id: workOrderId } },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { entry: data.data, requestId: data.meta.request_id };
}

export async function completeFieldWorkOrder(
  workOrderId: string,
  version: number,
  input: FieldCompletionInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ workOrder: WorkOrderView; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/field/work-orders/{work_order_id}/complete",
    {
      params: {
        path: { work_order_id: workOrderId },
        header: { "If-Match": `"${version}"` },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { workOrder: data.data, requestId: data.meta.request_id };
}

export async function listWorkOrderControls(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ controls: WorkOrderControl[]; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/work-orders/{work_order_id}/controls",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return { controls: data.data, requestId: data.meta.request_id };
}

export async function listDispatchWorkOrders(
  query: DispatchWorkOrderQuery = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  workOrders: DispatchWorkOrder[];
  total: number;
  limit: number;
  offset: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/dispatch/work-orders", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    workOrders: data.data,
    total: data.meta.total,
    limit: data.meta.limit,
    offset: data.meta.offset,
    requestId: data.meta.request_id,
  };
}

export async function assignWorkOrder(
  workOrderId: string,
  version: number,
  assigneeSubjectId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ workOrder: WorkOrderView; requestId: string }> {
  return workOrderRequest(
    "POST", workOrderId, "/assign", version,
    { assignee_subject_id: assigneeSubjectId }, client,
  );
}

export async function proposeWorkOrderAssignment(
  workOrderId: string,
  input: WorkOrderAssignmentProposalInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ proposal: WorkOrderAssignmentProposal; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/work-orders/{work_order_id}/assignment-proposals",
    {
      params: { path: { work_order_id: workOrderId } },
      body: { ...input, assignment_target: input.assignment_target ?? "LOCAL_DIRECTORY" },
    },
  );
  if (!data) throw toApiError(response, error);
  return { proposal: data.data, requestId: data.meta.request_id };
}

export async function listWorkOrderAssignmentOptions(
  workOrderId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  options: WorkOrderAssignmentOption[];
  fsmUnavailableReason: string | null;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/work-orders/{work_order_id}/assignment-options",
    { params: { path: { work_order_id: workOrderId } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    options: data.data.options,
    fsmUnavailableReason: data.data.fsm_unavailable_reason,
    requestId: data.meta.request_id,
  };
}

export async function proposeWorkOrderClosure(
  workOrderId: string,
  input: WorkOrderClosureProposalInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ proposal: WorkOrderClosureProposal; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/work-orders/{work_order_id}/closure-proposals",
    {
      params: { path: { work_order_id: workOrderId } },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { proposal: data.data, requestId: data.meta.request_id };
}

export async function acceptWorkOrder(
  workOrderId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
) {
  return workOrderRequest("POST", workOrderId, "/accept", version, undefined, client);
}

export async function startWorkOrder(
  workOrderId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
) {
  return workOrderRequest("POST", workOrderId, "/start", version, undefined, client);
}

export async function holdWorkOrder(
  workOrderId: string,
  version: number,
  input: WorkOrderHoldInput,
  client: IndustrialApiClient = industrialApi,
) {
  return workOrderRequest("POST", workOrderId, "/hold", version, input, client);
}

export async function resumeWorkOrder(
  workOrderId: string,
  version: number,
  reason: string,
  client: IndustrialApiClient = industrialApi,
) {
  return workOrderRequest("POST", workOrderId, "/resume", version, { reason }, client);
}

export async function escalateWorkOrder(
  workOrderId: string,
  version: number,
  input: WorkOrderEscalateInput,
  client: IndustrialApiClient = industrialApi,
) {
  return workOrderRequest("POST", workOrderId, "/escalate", version, input, client);
}

export async function replanWorkOrder(
  workOrderId: string,
  version: number,
  input: WorkOrderReplanInput,
  client: IndustrialApiClient = industrialApi,
) {
  return workOrderRequest("POST", workOrderId, "/replan", version, input, client);
}

export async function rescheduleWorkOrder(
  workOrderId: string,
  version: number,
  input: WorkOrderRescheduleInput,
  client: IndustrialApiClient = industrialApi,
) {
  return workOrderRequest("POST", workOrderId, "/reschedule", version, input, client);
}

export async function completeWorkOrder(
  workOrderId: string,
  version: number,
  completion: WorkOrderCompletion | Record<string, unknown>,
  client: IndustrialApiClient = industrialApi,
) {
  return workOrderRequest("POST", workOrderId, "/complete", version, completion, client);
}

export async function verifyWorkOrder(
  workOrderId: string,
  version: number,
  passed: boolean,
  reason: string,
  client: IndustrialApiClient = industrialApi,
) {
  return workOrderRequest("POST", workOrderId, "/verify", version, { passed, reason }, client);
}

export async function closeWorkOrder(
  workOrderId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
) {
  return workOrderRequest("POST", workOrderId, "/close", version, undefined, client);
}

export async function listFeedbackCandidates(
  query: FeedbackCandidateQuery = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  candidates: FeedbackCandidate[];
  legalActions: string[];
  total: number;
  limit: number;
  offset: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/data-feedback/candidates", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    candidates: data.data,
    legalActions: data.legal_actions,
    total: Number(data.meta.total),
    limit: Number(data.meta.limit),
    offset: Number(data.meta.offset),
    requestId: String(data.meta.request_id),
  };
}

export async function getFeedbackCandidate(
  candidateId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ candidate: FeedbackCandidateDetail; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/data-feedback/candidates/{candidate_id}",
    { params: { path: { candidate_id: candidateId } } },
  );
  if (!data) throw toApiError(response, error);
  return { candidate: data.data, requestId: data.meta.request_id };
}

export async function decideCandidateEligibility(
  candidateId: string,
  version: number,
  input: components["schemas"]["EligibilityDecisionBody"],
  client: IndustrialApiClient = industrialApi,
): Promise<WithRequestId<CandidateGovernance>> {
  const { data, error, response } = await client.POST(
    "/api/v1/data-feedback/candidates/{candidate_id}/eligibility-decisions",
    {
      params: {
        path: { candidate_id: candidateId },
        header: { "If-Match": `"${version}"` },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { ...data.data, requestId: data.meta.request_id };
}

export async function runCandidateDlp(
  candidateId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<WithRequestId<DlpRun>> {
  const { data, error, response } = await client.POST(
    "/api/v1/data-feedback/candidates/{candidate_id}/dlp-runs",
    {
      params: {
        path: { candidate_id: candidateId },
        header: { "If-Match": `"${version}"` },
      },
      body: { policy_version: "m3-presidio-zh-industrial-v1" },
    },
  );
  if (!data) throw toApiError(response, error);
  return { ...data.data, requestId: data.meta.request_id };
}

export async function createCandidateAnnotationTask(
  candidateId: string,
  version: number,
  idempotencyKey: string,
  input: components["schemas"]["AnnotationTaskBody"],
  client: IndustrialApiClient = industrialApi,
): Promise<WithRequestId<AnnotationTask>> {
  const { data, error, response } = await client.POST(
    "/api/v1/data-feedback/candidates/{candidate_id}/annotation-tasks",
    {
      params: {
        path: { candidate_id: candidateId },
        header: {
          "If-Match": `"${version}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { ...data.data, requestId: data.meta.request_id };
}

export async function syncCandidateAnnotationTask(
  taskId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<WithRequestId<AnnotationTask>> {
  const { data, error, response } = await client.POST(
    "/api/v1/annotation-tasks/{task_id}/sync",
    {
      params: {
        path: { task_id: taskId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { ...data.data, requestId: data.meta.request_id };
}

export async function startCurationRun(
  input: components["schemas"]["CurationRunBody"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ run: CurationRun; requestId: string }> {
  const { data, error, response } = await client.POST("/api/v1/curation-runs", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return { run: data.data, requestId: data.meta.request_id };
}

export async function getCurationRun(
  runId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ run: CurationRun; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/curation-runs/{run_id}", {
    params: { path: { run_id: runId } },
  });
  if (!data) throw toApiError(response, error);
  return { run: data.data, requestId: data.meta.request_id };
}

export async function listCurationRuns(
  query: CurationRunQuery = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  runs: CurationRun[];
  legalActions: string[];
  total: number;
  limit: number;
  offset: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/curation-runs", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    runs: data.data,
    legalActions: data.legal_actions,
    total: Number(data.meta.total),
    limit: Number(data.meta.limit),
    offset: Number(data.meta.offset),
    requestId: String(data.meta.request_id),
  };
}

export async function listDatasetSnapshots(
  query: DatasetSnapshotQuery = { limit: 100, offset: 0 },
  client: IndustrialApiClient = industrialApi,
): Promise<{
  snapshots: DatasetSnapshot[];
  total: number;
  limit: number;
  offset: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/dataset-snapshots", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    snapshots: data.data,
    total: Number(data.meta.total),
    limit: Number(data.meta.limit),
    offset: Number(data.meta.offset),
    requestId: String(data.meta.request_id),
  };
}

export async function getDatasetSnapshot(
  snapshotId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ snapshot: DatasetSnapshot; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/dataset-snapshots/{snapshot_id}",
    { params: { path: { snapshot_id: snapshotId } } },
  );
  if (!data) throw toApiError(response, error);
  return { snapshot: data.data, requestId: data.meta.request_id };
}

export async function getDatasetQualityReport(
  snapshotId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ report: DatasetQualityReport; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/dataset-snapshots/{snapshot_id}/quality-report",
    { params: { path: { snapshot_id: snapshotId } } },
  );
  if (!data) throw toApiError(response, error);
  return { report: data.data, requestId: data.meta.request_id };
}

export async function downloadDatasetManifest(
  snapshotId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<DatasetManifestDownload> {
  const { data, error, response } = await client.GET(
    "/api/v1/dataset-snapshots/{snapshot_id}/manifest",
    {
      params: { path: { snapshot_id: snapshotId } },
      parseAs: "blob",
    },
  );
  if (!data) {
    let errorEnvelope: unknown = error;
    if (error instanceof Blob) {
      try {
        errorEnvelope = JSON.parse(await error.text()) as unknown;
      } catch {
        // Preserve the original Blob for the generic API error fallback.
      }
    }
    throw toApiError(response, errorEnvelope);
  }
  const etag = response.headers.get("etag")?.replace(/^W\//, "").replace(/^"|"$/g, "");
  const manifestHash = await verifyDatasetManifestBlob(data, etag, response);
  return {
    blob: data,
    filename: `${snapshotId}-manifest.json`,
    manifestHash,
  };
}

async function verifyDatasetManifestBlob(
  blob: Blob,
  expectedHash: string | undefined,
  response: Response,
): Promise<string> {
  const failure = () => new ApiClientError(
    503,
    "dataset_manifest_integrity_failed",
    "dependency",
    "Dataset manifest integrity check failed",
    false,
    response.headers.get("x-request-id"),
  );
  if (!expectedHash || !/^sha256:[0-9a-f]{64}$/.test(expectedHash)) throw failure();
  try {
    const digest = await globalThis.crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
    const actualHash = `sha256:${Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, "0")).join("")}`;
    if (actualHash !== expectedHash) throw failure();
  } catch (error) {
    if (error instanceof ApiClientError) throw error;
    throw failure();
  }
  return expectedHash;
}

export async function getDataLineage(
  lineageId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ lineage: DataLineage; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/data-lineage/{lineage_run_id}",
    { params: { path: { lineage_run_id: lineageId } } },
  );
  if (!data) throw toApiError(response, error);
  return { lineage: data.data, requestId: data.meta.request_id };
}

export async function listKnowledgeVersions(
  status?: "DRAFT" | "REVIEWED" | "PUBLISHED",
  client: IndustrialApiClient = industrialApi,
): Promise<{ versions: KnowledgeVersion[]; legalActions: string[]; total: number }> {
  const { data, error, response } = await client.GET("/api/v1/knowledge/document-versions", {
    params: { query: { status, limit: 100, offset: 0 } },
  });
  if (!data) throw toApiError(response, error);
  return {
    versions: data.data,
    legalActions: data.legal_actions,
    total: Number(data.meta.total),
  };
}

export async function createKnowledgeDocument(
  input: components["schemas"]["KnowledgeDocumentBody"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeVersion> {
  const { data, error, response } = await client.POST("/api/v1/knowledge/documents", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function createKnowledgeSourceFile(
  input: components["schemas"]["KnowledgeSourceFileBody"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeFile> {
  const { data, error, response } = await client.POST("/api/v1/knowledge/source-files", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function uploadKnowledgeSourceContent(
  ingestionId: string,
  version: number,
  file: File,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeFile> {
  const { data, error, response } = await client.PUT(
    "/api/v1/knowledge/source-files/{ingestion_id}/content",
    {
      body: file as unknown as string,
      params: {
        path: { ingestion_id: ingestionId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function getKnowledgeSourceFile(
  ingestionId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeFile> {
  const { data, error, response } = await client.GET(
    "/api/v1/knowledge/source-files/{ingestion_id}",
    { params: { path: { ingestion_id: ingestionId } } },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listKnowledgeSourceFiles(
  status?: "AWAITING_UPLOAD" | "QUEUED" | "RUNNING" | "DRAFT_READY" | "REJECTED" | "FAILED",
  client: IndustrialApiClient = industrialApi,
): Promise<{ files: KnowledgeFile[]; legalActions: string[]; total: number }> {
  const { data, error, response } = await client.GET("/api/v1/knowledge/source-files", {
    params: { query: { status, limit: 100, offset: 0 } },
  });
  if (!data) throw toApiError(response, error);
  return {
    files: data.data,
    legalActions: data.legal_actions,
    total: Number(data.meta.total),
  };
}

export async function reprocessKnowledgeSourceFile(
  ingestionId: string,
  version: number,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeFile> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge/source-files/{ingestion_id}/reprocess",
    {
      params: {
        path: { ingestion_id: ingestionId },
        header: {
          "If-Match": `"${version}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function createKnowledgeDocumentVersion(
  documentId: string,
  input: components["schemas"]["KnowledgeDocumentVersionBody"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeVersion> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge/documents/{document_id}/versions",
    {
      body: input,
      params: {
        path: { document_id: documentId },
        header: { "Idempotency-Key": idempotencyKey },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function reviewKnowledgeVersion(
  documentVersionId: string,
  stateVersion: number,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeVersion> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge/document-versions/{document_version_id}/reviews",
    {
      params: {
        path: { document_version_id: documentVersionId },
        header: { "If-Match": `"${stateVersion}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listKnowledgeReleases(
  status?:
    | "EVALUATION_PENDING"
    | "EVALUATING"
    | "CANDIDATE"
    | "REJECTED"
    | "PUBLISHED"
    | "REVOKED",
  client: IndustrialApiClient = industrialApi,
): Promise<{ releases: KnowledgeRelease[]; legalActions: string[]; total: number }> {
  const { data, error, response } = await client.GET("/api/v1/knowledge/index-releases", {
    params: { query: { status, limit: 100, offset: 0 } },
  });
  if (!data) throw toApiError(response, error);
  return {
    releases: data.data,
    legalActions: data.legal_actions,
    total: Number(data.meta.total),
  };
}

export async function createKnowledgeRelease(
  input: components["schemas"]["KnowledgeReleaseBody"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeRelease> {
  const { data, error, response } = await client.POST("/api/v1/knowledge/index-releases", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function promoteKnowledgeRelease(
  releaseId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeRelease> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge/index-releases/{release_id}/promote",
    { params: { path: { release_id: releaseId } } },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function rollbackKnowledgeRelease(
  releaseId: string,
  activationVersion: number,
  expectedActiveReleaseId: string,
  reason: string,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeRelease> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge/index-releases/{release_id}/rollback",
    {
      body: { reason, expected_active_release_id: expectedActiveReleaseId },
      params: {
        path: { release_id: releaseId },
        header: {
          "If-Match": `"${activationVersion}"`,
          "Idempotency-Key": idempotencyKey,
        },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listKnowledgeIndexActivations(
  name?: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ activations: KnowledgeIndexActivation[]; total: number }> {
  const { data, error, response } = await client.GET(
    "/api/v1/knowledge/index-activations",
    { params: { query: { name, limit: 100 } } },
  );
  if (!data) throw toApiError(response, error);
  return { activations: data.data, total: Number(data.meta.total) };
}

export async function startKnowledgeIndexEvaluation(
  releaseId: string,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeIndexEvaluation> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge/index-releases/{release_id}/evaluations",
    {
      params: {
        path: { release_id: releaseId },
        header: { "Idempotency-Key": idempotencyKey },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listKnowledgeDeletions(
  status?: "QUEUED" | "RUNNING" | "COMPLETED" | "PARTIAL" | "FAILED",
  client: IndustrialApiClient = industrialApi,
): Promise<{ deletions: KnowledgeDeletion[]; legalActions: string[]; total: number }> {
  const { data, error, response } = await client.GET("/api/v1/knowledge/deletions", {
    params: { query: { status, limit: 100, offset: 0 } },
  });
  if (!data) throw toApiError(response, error);
  return {
    deletions: data.data,
    legalActions: data.legal_actions,
    total: Number(data.meta.total),
  };
}

export async function listKnowledgeGraphs(
  client: IndustrialApiClient = industrialApi,
): Promise<{ releases: KnowledgeGraphRelease[]; total: number; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/knowledge-graphs");
  if (!data) throw toApiError(response, error);
  return {
    releases: data.data,
    total: Number(data.meta.total),
    requestId: String(data.meta.request_id),
  };
}

export async function createKnowledgeGraph(
  input: CreateKnowledgeGraphInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeGraphRelease> {
  const { data, error, response } = await client.POST("/api/v1/knowledge-graphs", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function addKnowledgeGraphNode(
  graphReleaseId: string,
  version: number,
  input: CreateKnowledgeGraphNodeInput,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeGraphRelease> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-graphs/{graph_release_id}/nodes",
    {
      body: input,
      params: {
        path: { graph_release_id: graphReleaseId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function addKnowledgeGraphEdge(
  graphReleaseId: string,
  version: number,
  input: CreateKnowledgeGraphEdgeInput,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeGraphRelease> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-graphs/{graph_release_id}/edges",
    {
      body: input,
      params: {
        path: { graph_release_id: graphReleaseId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function syncKnowledgeGraph(
  graphReleaseId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeGraphRelease> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-graphs/{graph_release_id}/sync",
    {
      params: {
        path: { graph_release_id: graphReleaseId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function evaluateKnowledgeGraph(
  graphReleaseId: string,
  version: number,
  input: EvaluateKnowledgeGraphInput,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeGraphRelease> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-graphs/{graph_release_id}/evaluations",
    {
      body: input,
      params: {
        path: { graph_release_id: graphReleaseId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function activateKnowledgeGraph(
  graphReleaseId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeGraphRelease> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-graphs/{graph_release_id}/activate",
    {
      params: {
        path: { graph_release_id: graphReleaseId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function queryKnowledgeGraph(
  input: KnowledgeGraphQueryInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ paths: KnowledgeGraphPath[]; total: number; requestId: string }> {
  const { data, error, response } = await client.POST("/api/v1/knowledge-graphs/query", {
    body: input,
  });
  if (!data) throw toApiError(response, error);
  return {
    paths: data.data,
    total: Number(data.meta.total),
    requestId: String(data.meta.request_id),
  };
}

export async function listKnowledgeGraphExtractions(
  graphReleaseId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ extractions: KnowledgeGraphExtraction[]; total: number; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/knowledge-graphs/{graph_release_id}/extractions",
    { params: { path: { graph_release_id: graphReleaseId } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    extractions: data.data,
    total: Number(data.meta.total),
    requestId: String(data.meta.request_id),
  };
}

export async function requestKnowledgeGraphExtraction(
  graphReleaseId: string,
  version: number,
  citationIds: string[],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeGraphExtraction> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-graphs/{graph_release_id}/extractions",
    {
      body: { citation_ids: citationIds },
      params: {
        path: { graph_release_id: graphReleaseId },
        header: {
          "Idempotency-Key": idempotencyKey,
          "If-Match": `"${version}"`,
        },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function retryKnowledgeGraphExtraction(
  extractionJobId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeGraphExtraction> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-graphs/extractions/{extraction_job_id}/retry",
    {
      params: {
        path: { extraction_job_id: extractionJobId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function reviewKnowledgeGraphExtraction(
  extractionJobId: string,
  version: number,
  decision: ReviewKnowledgeGraphExtractionInput["decision"],
  reason: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeGraphExtraction> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-graphs/extractions/{extraction_job_id}/review",
    {
      body: { decision, reason },
      params: {
        path: { extraction_job_id: extractionJobId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listKnowledgeSearchProfiles(
  client: IndustrialApiClient = industrialApi,
): Promise<{ profiles: KnowledgeSearchProfile[]; total: number; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/knowledge-search-profiles");
  if (!data) throw toApiError(response, error);
  return {
    profiles: data.data,
    total: Number(data.meta.total),
    requestId: String(data.meta.request_id),
  };
}

export async function createKnowledgeSearchProfile(
  input: CreateKnowledgeSearchProfileInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeSearchProfile> {
  const { data, error, response } = await client.POST("/api/v1/knowledge-search-profiles", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function syncKnowledgeSearchProfile(
  searchProfileId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeSearchProfile> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-search-profiles/{search_profile_id}/sync",
    {
      params: {
        path: { search_profile_id: searchProfileId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listKnowledgeSearchRebuilds(
  searchProfileId?: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ rebuilds: KnowledgeSearchRebuild[]; total: number; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/knowledge-search-profiles/rebuilds",
    { params: { query: { search_profile_id: searchProfileId, limit: 50 } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    rebuilds: data.data,
    total: Number(data.meta.total),
    requestId: String(data.meta.request_id),
  };
}

export async function requestKnowledgeSearchRebuild(
  searchProfileId: string,
  profileVersion: number,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeSearchRebuild> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-search-profiles/{search_profile_id}/rebuilds",
    {
      params: {
        path: { search_profile_id: searchProfileId },
        header: {
          "Idempotency-Key": idempotencyKey,
          "If-Match": `"${profileVersion}"`,
        },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function retryKnowledgeSearchRebuild(
  rebuildJobId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeSearchRebuild> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-search-profiles/rebuilds/{rebuild_job_id}/retry",
    {
      params: {
        path: { rebuild_job_id: rebuildJobId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function evaluateKnowledgeSearchProfile(
  searchProfileId: string,
  version: number,
  input: EvaluateKnowledgeSearchProfileInput,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeSearchProfile> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-search-profiles/{search_profile_id}/evaluations",
    {
      body: input,
      params: {
        path: { search_profile_id: searchProfileId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function activateKnowledgeSearchProfileShadow(
  searchProfileId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeSearchProfile> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-search-profiles/{search_profile_id}/activate-shadow",
    {
      params: {
        path: { search_profile_id: searchProfileId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function deactivateKnowledgeSearchProfileShadow(
  searchProfileId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeSearchProfile> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-search-profiles/{search_profile_id}/deactivate-shadow",
    {
      params: {
        path: { search_profile_id: searchProfileId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function queryKnowledgeSearchProfile(
  input: KnowledgeSearchQueryInput,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeSearchResult> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge-search-profiles/query",
    { body: input },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function getExternalSearchPolicy(
  client: IndustrialApiClient = industrialApi,
): Promise<ExternalSearchPolicy> {
  const { data, error, response } = await client.GET("/api/v1/external-search/policy");
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function updateExternalSearchPolicy(
  input: ExternalSearchPolicyInput,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<ExternalSearchPolicy> {
  const { data, error, response } = await client.PUT("/api/v1/external-search/policy", {
    body: input,
    params: { header: { "If-Match": `"${version}"` } },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function runExternalSearch(
  input: ExternalSearchInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<ExternalSearchQuery> {
  const { data, error, response } = await client.POST("/api/v1/external-search/queries", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listExternalSearches(
  client: IndustrialApiClient = industrialApi,
): Promise<{ searches: ExternalSearchQuery[]; total: number; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/external-search/queries", {
    params: { query: { limit: 50 } },
  });
  if (!data) throw toApiError(response, error);
  return {
    searches: data.data,
    total: Number(data.meta.total),
    requestId: String(data.meta.request_id),
  };
}

export async function concludeExternalSearch(
  externalSearchId: string,
  version: number,
  input: ExternalSearchConclusionInput,
  client: IndustrialApiClient = industrialApi,
): Promise<ExternalSearchQuery> {
  const { data, error, response } = await client.POST(
    "/api/v1/external-search/queries/{external_search_id}/conclusion",
    {
      body: input,
      params: {
        path: { external_search_id: externalSearchId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function requestKnowledgeDeletion(
  documentVersionId: string,
  input: components["schemas"]["KnowledgeDeletionBody"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeDeletion> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge/document-versions/{document_version_id}/deletions",
    {
      body: input,
      params: {
        path: { document_version_id: documentVersionId },
        header: { "Idempotency-Key": idempotencyKey },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function retryKnowledgeDeletion(
  deletionId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeDeletion> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge/deletions/{deletion_id}/retry",
    {
      params: {
        path: { deletion_id: deletionId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function sweepExpiredKnowledge(
  reason: string,
  limit = 100,
  client: IndustrialApiClient = industrialApi,
): Promise<KnowledgeDeletion[]> {
  const { data, error, response } = await client.POST(
    "/api/v1/knowledge/deletions/expiry-sweep",
    { body: { reason, limit } },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listTrainingExperiments(
  query: TrainingExperimentQuery = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  experiments: TrainingExperiment[];
  legalActions: string[];
  total: number;
  limit: number;
  offset: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/experiments", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    experiments: data.data,
    legalActions: data.legal_actions,
    total: Number(data.meta.total),
    limit: Number(data.meta.limit),
    offset: Number(data.meta.offset),
    requestId: String(data.meta.request_id),
  };
}

export async function getTrainingExperiment(
  experimentId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ experiment: TrainingExperiment; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/experiments/{experiment_id}",
    { params: { path: { experiment_id: experimentId } } },
  );
  if (!data) throw toApiError(response, error);
  return { experiment: data.data, requestId: String(data.meta.request_id) };
}

export async function listEvaluationJobs(
  query: EvaluationJobQuery = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  jobs: EvaluationJob[];
  legalActions: string[];
  total: number;
  limit: number;
  offset: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/model-evaluation-jobs", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    jobs: data.data,
    legalActions: data.legal_actions,
    total: Number(data.meta.total),
    limit: Number(data.meta.limit),
    offset: Number(data.meta.offset),
    requestId: String(data.meta.request_id),
  };
}

export async function createEvaluationJob(
  input: components["schemas"]["EvaluationJobBody"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<WithRequestId<EvaluationJob>> {
  const { data, error, response } = await client.POST("/api/v1/model-evaluation-jobs", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return { ...data.data, requestId: String(data.meta.request_id) };
}

export async function createTrainingExperiment(
  input: components["schemas"]["ExperimentPlanBody"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<WithRequestId<TrainingExperiment>> {
  const { data, error, response } = await client.POST("/api/v1/experiments", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return { ...data.data, requestId: String(data.meta.request_id) };
}

export async function startTrainingExperiment(
  experimentId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<WithRequestId<TrainingExperiment>> {
  const { data, error, response } = await client.POST(
    "/api/v1/experiments/{experiment_id}/start",
    {
      params: {
        path: { experiment_id: experimentId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { ...data.data, requestId: String(data.meta.request_id) };
}

export async function listEvaluationSuites(
  client: IndustrialApiClient = industrialApi,
): Promise<{ suites: EvaluationSuite[]; legalActions: string[]; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/evaluation-suites");
  if (!data) throw toApiError(response, error);
  return {
    suites: data.data,
    legalActions: data.legal_actions,
    requestId: String(data.meta.request_id),
  };
}

export async function createEvaluationSuite(
  input: components["schemas"]["EvaluationSuiteBody"],
  client: IndustrialApiClient = industrialApi,
): Promise<WithRequestId<EvaluationSuite>> {
  const { data, error, response } = await client.POST("/api/v1/evaluation-suites", {
    body: input,
  });
  if (!data) throw toApiError(response, error);
  return { ...data.data, requestId: String(data.meta.request_id) };
}

export async function listEvaluationPolicies(
  client: IndustrialApiClient = industrialApi,
): Promise<{ policies: EvaluationPolicy[]; legalActions: string[]; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/evaluation-policies");
  if (!data) throw toApiError(response, error);
  return {
    policies: data.data,
    legalActions: data.legal_actions,
    requestId: String(data.meta.request_id),
  };
}

export async function createEvaluationPolicy(
  input: components["schemas"]["EvaluationPolicyBody"],
  client: IndustrialApiClient = industrialApi,
): Promise<WithRequestId<EvaluationPolicy>> {
  const { data, error, response } = await client.POST("/api/v1/evaluation-policies", {
    body: input,
  });
  if (!data) throw toApiError(response, error);
  return { ...data.data, requestId: String(data.meta.request_id) };
}

export async function listModelEvaluations(
  query: ModelEvaluationQuery = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  evaluations: ModelEvaluation[];
  legalActions: string[];
  total: number;
  limit: number;
  offset: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/model-evaluations", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    evaluations: data.data,
    legalActions: data.legal_actions,
    total: Number(data.meta.total),
    limit: Number(data.meta.limit),
    offset: Number(data.meta.offset),
    requestId: String(data.meta.request_id),
  };
}

export async function getModelEvaluation(
  evaluationId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ evaluation: ModelEvaluation; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/model-evaluations/{evaluation_id}",
    { params: { path: { evaluation_id: evaluationId } } },
  );
  if (!data) throw toApiError(response, error);
  return { evaluation: data.data, requestId: String(data.meta.request_id) };
}

export async function downloadModelEvaluationEvidence(
  evaluationId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<ModelEvaluationEvidenceDownload> {
  const { data, error, response } = await client.GET(
    "/api/v1/model-evaluations/{evaluation_id}/evidence",
    {
      params: { path: { evaluation_id: evaluationId } },
      parseAs: "blob",
    },
  );
  if (!data) {
    let errorEnvelope: unknown = error;
    if (error instanceof Blob) {
      try {
        errorEnvelope = JSON.parse(await error.text()) as unknown;
      } catch {
        // Preserve the original Blob for the generic API error fallback.
      }
    }
    throw toApiError(response, errorEnvelope);
  }
  const etag = response.headers.get("etag")?.replace(/^W\//, "").replace(/^"|"$/g, "");
  const evidenceHash = await verifyModelEvaluationEvidenceBlob(data, etag, response);
  return {
    blob: data,
    filename: `evaluation-${evaluationId}-evidence.json`,
    evidenceHash,
    requestId: response.headers.get("x-request-id") ?? undefined,
  };
}

async function verifyModelEvaluationEvidenceBlob(
  blob: Blob,
  expectedHash: string | undefined,
  response: Response,
): Promise<string> {
  const failure = () => new ApiClientError(
    503,
    "evaluation_evidence_integrity_failed",
    "dependency",
    "Model evaluation evidence integrity check failed",
    false,
    response.headers.get("x-request-id"),
  );
  if (!expectedHash || !/^[0-9a-f]{64}$/.test(expectedHash)) throw failure();
  try {
    const digest = await globalThis.crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
    const actualHash = Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, "0")).join("");
    if (actualHash !== expectedHash) throw failure();
  } catch (error) {
    if (error instanceof ApiClientError) throw error;
    throw failure();
  }
  return `sha256:${expectedHash}`;
}

export async function listModelReleases(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  releases: ModelRelease[];
  legalActions: string[];
  mcpServerVersions: Record<string, string>;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/model-releases");
  if (!data) throw toApiError(response, error);
  return {
    releases: data.data,
    legalActions: data.legal_actions,
    mcpServerVersions: data.mcp_server_versions,
    requestId: String(data.meta.request_id),
  };
}

export async function listSupplyChainEvidence(
  releaseReadyOnly = false,
  client: IndustrialApiClient = industrialApi,
): Promise<{ evidence: SupplyChainEvidence[]; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/supply-chain/evidence", {
    params: { query: { release_ready_only: releaseReadyOnly } },
  });
  if (!data) throw toApiError(response, error);
  return {
    evidence: data.data,
    requestId: String(data.meta.request_id),
  };
}

export async function listComponentSupplyChainEvidence(
  component: "VLM" | "ASR",
  modelId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ evidence: SupplyChainEvidence[]; bindingStatus: string; reason?: string; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/supply-chain/evidence", {
    params: { query: { component, model_id: modelId, release_ready_only: true } },
  });
  if (!data) throw toApiError(response, error);
  return {
    evidence: data.data,
    bindingStatus: String(data.meta.binding_status ?? "UNKNOWN"),
    reason: data.meta.reason === undefined ? undefined : String(data.meta.reason),
    requestId: String(data.meta.request_id),
  };
}

export async function createSupplyChainSuccessor(
  releaseId: string,
  version: number,
  input: components["schemas"]["SupplyChainSuccessorBody"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<ModelRelease> {
  const { data, error, response } = await client.POST(
    "/api/v1/model-releases/{release_id}/supply-chain-successor", {
      body: input,
      params: { path: { release_id: releaseId },
        header: { "If-Match": `"${version}"`, "Idempotency-Key": idempotencyKey } },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function createModelRelease(
  input: components["schemas"]["ReleaseManifestBody"],
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<ModelRelease> {
  const { data, error, response } = await client.POST("/api/v1/model-releases", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function validateModelRelease(
  releaseId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<ModelRelease> {
  const { data, error, response } = await client.POST(
    "/api/v1/model-releases/{release_id}/validate",
    {
      params: {
        path: { release_id: releaseId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function submitModelReleaseApproval(
  releaseId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<ModelRelease> {
  const { data, error, response } = await client.POST(
    "/api/v1/model-releases/{release_id}/submit-approval",
    {
      params: {
        path: { release_id: releaseId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function decideModelReleaseApproval(
  releaseId: string,
  releaseVersion: number,
  approvalVersion: number,
  decision: "APPROVED" | "REJECTED",
  reason: string,
  client: IndustrialApiClient = industrialApi,
): Promise<ModelRelease> {
  const { data, error, response } = await client.POST(
    "/api/v1/model-releases/{release_id}/approval-decisions",
    {
      body: { approval_version: approvalVersion, decision, reason },
      params: {
        path: { release_id: releaseId },
        header: { "If-Match": `"${releaseVersion}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listModelDeployments(
  client: IndustrialApiClient = industrialApi,
): Promise<{ deployments: ModelDeployment[]; legalActions: string[]; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/model-deployments");
  if (!data) throw toApiError(response, error);
  return {
    deployments: data.data,
    legalActions: data.legal_actions,
    requestId: String(data.meta.request_id),
  };
}

export async function requestModelDeployment(
  releaseId: string,
  releaseVersion: number,
  input: ModelDeploymentPlanInput,
  client: IndustrialApiClient = industrialApi,
): Promise<ModelDeployment> {
  const { data, error, response } = await client.POST(
    "/api/v1/model-releases/{release_id}/deployments",
    {
      body: input,
      params: {
        path: { release_id: releaseId },
        header: { "If-Match": `"${releaseVersion}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function promoteModelRelease(
  releaseId: string,
  deploymentVersion: number,
  client: IndustrialApiClient = industrialApi,
): Promise<ModelDeployment> {
  const { data, error, response } = await client.POST(
    "/api/v1/model-releases/{release_id}/promotions",
    {
      params: {
        path: { release_id: releaseId },
        header: { "If-Match": `"${deploymentVersion}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function rollbackModelRelease(
  releaseId: string,
  deploymentVersion: number,
  reasonCode: string,
  client: IndustrialApiClient = industrialApi,
): Promise<ModelDeployment> {
  const { data, error, response } = await client.POST(
    "/api/v1/model-releases/{release_id}/rollbacks",
    {
      body: { reason_code: reasonCode },
      params: {
        path: { release_id: releaseId },
        header: { "If-Match": `"${deploymentVersion}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listModelRoutes(
  client: IndustrialApiClient = industrialApi,
): Promise<ModelRoute[]> {
  const { data, error, response } = await client.GET("/api/v1/model-gateway/routes");
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function getModelRuntimeStatus(
  client: IndustrialApiClient = industrialApi,
): Promise<ModelRuntimeStatus> {
  const { data, error, response } = await client.GET(
    "/api/v1/model-gateway/runtime-status",
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listModelInferences(
  limit = 50,
  client: IndustrialApiClient = industrialApi,
): Promise<ModelInference[]> {
  const { data, error, response } = await client.GET(
    "/api/v1/model-gateway/inferences",
    { params: { query: { limit } } },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function updateModelQuota(
  alias: string,
  version: number,
  input: ModelQuotaInput,
  client: IndustrialApiClient = industrialApi,
): Promise<ModelRoute> {
  const { data, error, response } = await client.PUT(
    "/api/v1/model-gateway/routes/{alias_name}/quota",
    {
      body: input,
      params: {
        path: { alias_name: alias },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listSecurityAuditEvents(
  query: SecurityAuditQuery = {},
  emergencyGrantId?: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  events: SecurityAuditEvent[];
  nextCursor?: string;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/security/audit-events",
    {
      params: {
        query,
        header: { "X-Emergency-Grant-ID": emergencyGrantId },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return {
    events: data.data,
    nextCursor: data.meta.next_cursor ?? undefined,
    requestId: data.meta.request_id,
  };
}

export async function getSecurityAuditSummary(
  windowHours = 24,
  emergencyGrantId?: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ summary: SecurityAuditSummary; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/security/summary", {
    params: {
      query: { window_hours: windowHours },
      header: { "X-Emergency-Grant-ID": emergencyGrantId },
    },
  });
  if (!data) throw toApiError(response, error);
  return { summary: data.data, requestId: data.meta.request_id };
}

export async function listEmergencyAccessGrants(
  status?: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ grants: EmergencyAccessGrant[]; requestId: string; total: number }> {
  const { data, error, response } = await client.GET(
    "/api/v1/security/emergency-access-grants",
    { params: { query: { status } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    grants: data.data,
    requestId: String(data.meta.request_id),
    total: Number(data.meta.total),
  };
}

export async function requestEmergencyAccessGrant(
  input: EmergencyAccessRequestInput,
  client: IndustrialApiClient = industrialApi,
): Promise<EmergencyAccessGrant> {
  const { data, error, response } = await client.POST(
    "/api/v1/security/emergency-access-grants",
    { body: input },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function decideEmergencyAccessGrant(
  grantId: string,
  version: number,
  input: EmergencyAccessDecisionInput,
  client: IndustrialApiClient = industrialApi,
): Promise<EmergencyAccessGrant> {
  const { data, error, response } = await client.POST(
    "/api/v1/security/emergency-access-grants/{grant_id}/decisions",
    {
      body: input,
      params: {
        path: { grant_id: grantId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function revokeEmergencyAccessGrant(
  grantId: string,
  version: number,
  input: EmergencyAccessRevokeInput,
  client: IndustrialApiClient = industrialApi,
): Promise<EmergencyAccessGrant> {
  const { data, error, response } = await client.POST(
    "/api/v1/security/emergency-access-grants/{grant_id}/revoke",
    {
      body: input,
      params: {
        path: { grant_id: grantId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function getOperationsOverview(
  client: IndustrialApiClient = industrialApi,
): Promise<{ overview: OperationsOverview; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/operations/overview");
  if (!data) throw toApiError(response, error);
  return { overview: data.data, requestId: data.meta.request_id };
}

export async function getEnterpriseProjectAdoption(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  adoption: EnterpriseProjectAdoption;
  activeAssets: number;
  runtimeAssets: number;
  experimentHistory: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/enterprise-assets/adoption",
  );
  if (!data) throw toApiError(response, error);
  return {
    adoption: data.data,
    activeAssets: data.meta.active_assets,
    runtimeAssets: data.meta.runtime_assets,
    experimentHistory: data.meta.experiment_history,
    requestId: data.meta.request_id,
  };
}

export async function getEnterpriseRuntimeBindings(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  bindings: EnterpriseRuntimeBinding[];
  components: number;
  registeredComponents: number;
  deployedComponents: number;
  activeAliasComponents: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/enterprise-assets/runtime-bindings",
  );
  if (!data) throw toApiError(response, error);
  return {
    bindings: data.data,
    components: data.meta.components,
    registeredComponents: data.meta.registered_components,
    deployedComponents: data.meta.deployed_components,
    activeAliasComponents: data.meta.active_alias_components,
    requestId: data.meta.request_id,
  };
}

export async function getEnterpriseModelImportAcceptance(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  acceptance: EnterpriseModelImportAcceptance;
  components: number;
  eligibleComponents: number;
  totalActualSamples: number;
  commonBaselineReleaseIds: string[];
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/enterprise-assets/model-import-acceptance",
  );
  if (!data) throw toApiError(response, error);
  return {
    acceptance: data.data,
    components: data.meta.components,
    eligibleComponents: data.meta.eligible_components,
    totalActualSamples: data.meta.total_actual_samples,
    commonBaselineReleaseIds: data.meta.common_baseline_release_ids,
    requestId: data.meta.request_id,
  };
}

export async function getEnterpriseRerankerKServeAcceptance(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  acceptance: EnterpriseRerankerKServeAcceptance;
  stages: number;
  governedRequests: number;
  actualGpuExecution: true;
  evidenceChainSha256: string;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/enterprise-assets/reranker-kserve-acceptance",
  );
  if (!data) throw toApiError(response, error);
  return {
    acceptance: data.data,
    stages: data.meta.stages,
    governedRequests: data.meta.governed_requests,
    actualGpuExecution: data.meta.actual_gpu_execution,
    evidenceChainSha256: data.meta.evidence_chain_sha256,
    requestId: data.meta.request_id,
  };
}

export async function getEnterpriseProjectClosureReadiness(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  closure: EnterpriseProjectClosure;
  coverageDomains: number;
  sourceEvidence: number;
  intentionallyUnverifiedMethods: number;
  productionBlockers: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/enterprise-assets/closure-readiness",
  );
  if (!data) throw toApiError(response, error);
  return {
    closure: data.data,
    coverageDomains: data.meta.coverage_domains,
    sourceEvidence: data.meta.source_evidence,
    intentionallyUnverifiedMethods: data.meta.intentionally_unverified_methods,
    productionBlockers: data.meta.production_blockers,
    requestId: data.meta.request_id,
  };
}

export async function getEnterpriseProjectAssurance(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  assurance: EnterpriseProjectAssurance;
  securityScenarios: number;
  recoveryComponents: number;
  healthySlos: number;
  stagingScenarios: number;
  signoffs: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/enterprise-assets/project-assurance",
  );
  if (!data) throw toApiError(response, error);
  return {
    assurance: data.data,
    securityScenarios: data.meta.security_scenarios,
    recoveryComponents: data.meta.recovery_components,
    healthySlos: data.meta.healthy_slos,
    stagingScenarios: data.meta.staging_scenarios,
    signoffs: data.meta.signoffs,
    requestId: data.meta.request_id,
  };
}

export async function previewEnterpriseReleaseDraft(
  component: EnterpriseModelComponent,
  client: IndustrialApiClient = industrialApi,
): Promise<{ preview: EnterpriseReleaseDraftPreview; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/enterprise-assets/runtime-bindings/{component}/release-draft-preview",
    { params: { path: { component } } },
  );
  if (!data) throw toApiError(response, error);
  return { preview: data.data, requestId: data.meta.request_id };
}

export async function createEnterpriseStagingBaselineDraft(
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ baseline: EnterpriseStagingBaselineDraft; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/enterprise-assets/release-baseline-drafts",
    {
      params: { header: { "Idempotency-Key": idempotencyKey } },
    },
  );
  if (!data) throw toApiError(response, error);
  return { baseline: data.data, requestId: data.meta.request_id };
}

export async function createEnterpriseCandidateReleaseBatch(
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ batch: EnterpriseCandidateReleaseBatch; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/enterprise-assets/release-batches",
    {
      params: { header: { "Idempotency-Key": idempotencyKey } },
    },
  );
  if (!data) throw toApiError(response, error);
  return { batch: data.data, requestId: data.meta.request_id };
}

export async function listEnterpriseCandidateReleaseBatches(
  limit = 20,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  batches: EnterpriseCandidateReleaseBatchProgress[];
  awaitingApproval: number;
  approved: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/enterprise-assets/release-batches",
    { params: { query: { limit } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    batches: data.data,
    awaitingApproval: data.meta.awaiting_approval,
    approved: data.meta.approved,
    requestId: data.meta.request_id,
  };
}

export async function listEnterpriseCandidateReleaseBatchReviewQueue(
  limit = 20,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  batches: EnterpriseCandidateReleaseBatchProgress[];
  pendingDecisions: number;
  decidableDecisions: number;
  blockedBySeparationOfDuties: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/enterprise-assets/release-batches/review-queue",
    { params: { query: { limit } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    batches: data.data,
    pendingDecisions: data.meta.pending_decisions,
    decidableDecisions: data.meta.decidable_decisions,
    blockedBySeparationOfDuties: data.meta.blocked_by_separation_of_duties,
    requestId: data.meta.request_id,
  };
}

export async function listEnterpriseCandidateReleaseBatchDeploymentQueue(
  limit = 20,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  batches: EnterpriseCandidateReleaseBatchProgress[];
  deployableComponents: number;
  activeDeployments: number;
  promotableComponents: number;
  rollbackAvailableComponents: number;
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/enterprise-assets/release-batches/deployment-queue",
    { params: { query: { limit } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    batches: data.data,
    deployableComponents: data.meta.deployable_components,
    activeDeployments: data.meta.active_deployments,
    promotableComponents: data.meta.promotable_components,
    rollbackAvailableComponents: data.meta.rollback_available_components,
    requestId: data.meta.request_id,
  };
}

export async function requestEnterpriseCandidateReleaseComponentShadow(
  batchKeySha256: string,
  component: EnterpriseModelComponent,
  requestedBySubjectId: string,
  input: ModelDeploymentPlanInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ batch: EnterpriseCandidateReleaseBatchProgress; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/enterprise-assets/release-batches/{batch_key_sha256}/components/{component}/shadow-requests",
    {
      body: input,
      params: {
        path: { batch_key_sha256: batchKeySha256, component },
        query: { requested_by_subject_id: requestedBySubjectId },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { batch: data.data, requestId: data.meta.request_id };
}

export async function advanceEnterpriseCandidateReleaseBatchToApproval(
  batchKeySha256: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ batch: EnterpriseCandidateReleaseBatchProgress; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/enterprise-assets/release-batches/{batch_key_sha256}/approval-submissions",
    { params: { path: { batch_key_sha256: batchKeySha256 } } },
  );
  if (!data) throw toApiError(response, error);
  return { batch: data.data, requestId: data.meta.request_id };
}

export async function importEnterpriseModelAsset(
  component: EnterpriseModelComponent,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ modelImport: EnterpriseModelAssetImport; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/enterprise-assets/runtime-bindings/{component}/imports",
    {
      params: {
        path: { component },
        header: { "Idempotency-Key": idempotencyKey },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { modelImport: data.data, requestId: data.meta.request_id };
}

export async function createEnterpriseReleaseDraft(
  component: EnterpriseModelComponent,
  input: EnterpriseReleaseDraftInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ onboarding: EnterpriseReleaseOnboarding; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/enterprise-assets/runtime-bindings/{component}/release-drafts",
    {
      params: {
        path: { component },
        header: { "Idempotency-Key": idempotencyKey },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { onboarding: data.data, requestId: data.meta.request_id };
}

export async function getServicePerformance(
  windowStart: string,
  windowEnd: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ performance: ServicePerformance; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/operations/service-performance",
    { params: { query: { window_start: windowStart, window_end: windowEnd } } },
  );
  if (!data) throw toApiError(response, error);
  return { performance: data.data as ServicePerformance, requestId: data.meta.request_id };
}

export async function getServicePerformanceV3(
  windowStart: string,
  windowEnd: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ performance: ServicePerformanceV3; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/operations/service-performance",
    { params: { query: { window_start: windowStart, window_end: windowEnd, contract_version: "v3" } } },
  );
  if (!data) throw toApiError(response, error);
  return { performance: data.data as ServicePerformanceV3, requestId: data.meta.request_id };
}

export async function listServicePerformanceBaselines(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  baselines: ServicePerformanceBaseline[];
  legalActions: string[];
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/operations/service-performance/baselines",
  );
  if (!data) throw toApiError(response, error);
  return {
    baselines: data.data,
    legalActions: data.legal_actions,
    requestId: data.meta.request_id,
  };
}

export async function createServicePerformanceBaseline(
  input: CreateServicePerformanceBaselineInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ baseline: ServicePerformanceBaseline; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/operations/service-performance/baselines", { body: input },
  );
  if (!data) throw toApiError(response, error);
  return { baseline: data.data, requestId: data.meta.request_id };
}

export async function activateServicePerformanceBaseline(
  baselineId: string,
  version: number,
  reason: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ baseline: ServicePerformanceBaseline; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/operations/service-performance/baselines/{baseline_id}/activation",
    { body: { reason }, params: {
      path: { baseline_id: baselineId }, header: { "If-Match": `"${version}"` },
    } },
  );
  if (!data) throw toApiError(response, error);
  return { baseline: data.data, requestId: data.meta.request_id };
}

export async function getGpuOperations(
  client: IndustrialApiClient = industrialApi,
): Promise<{ operations: GpuOperations; requestId: string }> {
  const { data, error, response } = await client.GET("/api/v1/operations/gpu");
  if (!data) throw toApiError(response, error);
  return { operations: data.data, requestId: data.meta.request_id };
}

export async function getCostOperations(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  operations: CostOperations;
  legalActions: string[];
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/operations/cost");
  if (!data) throw toApiError(response, error);
  return {
    operations: data.data,
    legalActions: data.legal_actions,
    requestId: data.meta.request_id,
  };
}

export async function importCostLedgerBatch(
  input: CostLedgerBatchInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ batch: CostLedgerBatch; replayed: boolean; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/operations/cost-ledger/batches",
    {
      body: input,
      params: { header: { "Idempotency-Key": idempotencyKey } },
    },
  );
  if (!data) throw toApiError(response, error);
  return {
    batch: data.data,
    replayed: data.meta.replayed === "true",
    requestId: data.meta.request_id,
  };
}

export async function updateCostPolicy(
  input: CostPolicyInput,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<{ policy: CostPolicy; requestId: string }> {
  const { data, error, response } = await client.PUT("/api/v1/operations/cost/policy", {
    body: input,
    params: { header: { "If-Match": `"${version}"` } },
  });
  if (!data) throw toApiError(response, error);
  return { policy: data.data, requestId: data.meta.request_id };
}

export async function listAgentTraces(
  query: { status?: string; limit?: number; offset?: number } = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  items: AgentTraceSummary[];
  total: number;
  limit: number;
  offset: number;
  projectionVersion: string;
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/operations/traces", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    items: data.data.items,
    total: data.data.total,
    limit: data.data.limit,
    offset: data.data.offset,
    projectionVersion: data.data.projection_version,
    requestId: data.meta.request_id,
  };
}

export async function getAgentTrace(
  agentRunId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ trace: AgentTraceDetail; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/operations/traces/{agent_run_id}",
    { params: { path: { agent_run_id: agentRunId } } },
  );
  if (!data) throw toApiError(response, error);
  return { trace: data.data, requestId: data.meta.request_id };
}

export async function getToolGovernance(
  query: { window_hours?: number; recent_limit?: number } = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{ governance: ToolGovernance; requestId: string }> {
  const { data, error, response } = await client.GET(
    "/api/v1/operations/tools/governance",
    { params: { query } },
  );
  if (!data) throw toApiError(response, error);
  return { governance: data.data, requestId: data.meta.request_id };
}

export async function listPromptBundles(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  bundles: PromptBundle[];
  legalActions: string[];
  requestId: string;
  policyVersion: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/prompt-bundles");
  if (!data) throw toApiError(response, error);
  return {
    bundles: data.data,
    legalActions: data.legal_actions,
    requestId: String(data.meta.request_id),
    policyVersion: String(data.meta.policy_version),
  };
}

export async function createPromptBundle(
  input: CreatePromptBundleInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<PromptBundle> {
  const { data, error, response } = await client.POST("/api/v1/prompt-bundles", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function submitPromptBundle(
  promptBundleId: string,
  evaluationId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<PromptBundle> {
  const { data, error, response } = await client.POST(
    "/api/v1/prompt-bundles/{prompt_bundle_id}/submit",
    {
      body: { evaluation_id: evaluationId },
      params: {
        path: { prompt_bundle_id: promptBundleId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function reviewPromptBundle(
  promptBundleId: string,
  decision: "APPROVED" | "REJECTED",
  reason: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<PromptBundle> {
  const { data, error, response } = await client.POST(
    "/api/v1/prompt-bundles/{prompt_bundle_id}/review",
    {
      body: { decision, reason },
      params: {
        path: { prompt_bundle_id: promptBundleId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function retirePromptBundle(
  promptBundleId: string,
  reason: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<PromptBundle> {
  const { data, error, response } = await client.POST(
    "/api/v1/prompt-bundles/{prompt_bundle_id}/retire",
    {
      body: { reason },
      params: {
        path: { prompt_bundle_id: promptBundleId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listGovernedMemories(
  query: {
    memory_type?: "SESSION_NOTE" | "USER_PREFERENCE";
    status?: "ACTIVE" | "REVOKED";
    incident_id?: string;
  } = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  memories: GovernedMemory[];
  legalActions: string[];
  requestId: string;
  policyVersion: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/memories", {
    params: { query },
  });
  if (!data) throw toApiError(response, error);
  return {
    memories: data.data,
    legalActions: data.legal_actions,
    requestId: String(data.meta.request_id),
    policyVersion: String(data.meta.policy_version),
  };
}

export async function createGovernedMemory(
  input: CreateMemoryInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<GovernedMemory> {
  const { data, error, response } = await client.POST("/api/v1/memories", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function revokeGovernedMemory(
  memoryId: string,
  version: number,
  reason: string,
  client: IndustrialApiClient = industrialApi,
): Promise<GovernedMemory> {
  const { data, error, response } = await client.POST(
    "/api/v1/memories/{memory_id}/revoke",
    {
      body: { reason },
      params: {
        path: { memory_id: memoryId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listSupplierCollaborations(
  query: { status?: string; limit?: number } = {},
  client: IndustrialApiClient = industrialApi,
): Promise<{
  collaborations: SupplierCollaboration[];
  requestId: string;
  legalActions: string[];
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/supplier-collaborations",
    { params: { query } },
  );
  if (!data) throw toApiError(response, error);
  return {
    collaborations: data.data,
    requestId: String(data.meta.request_id),
    legalActions: data.legal_actions,
  };
}

export async function getSupplierCollaboration(
  collaborationId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<SupplierCollaboration> {
  const { data, error, response } = await client.GET(
    "/api/v1/supplier-collaborations/{collaboration_id}",
    { params: { path: { collaboration_id: collaborationId } } },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function createSupplierCollaboration(
  input: CreateSupplierCollaborationInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<SupplierCollaboration> {
  const { data, error, response } = await client.POST(
    "/api/v1/supplier-collaborations",
    {
      body: input,
      params: { header: { "Idempotency-Key": idempotencyKey } },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function dispatchSupplierCollaboration(
  collaborationId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<SupplierCollaboration> {
  return supplierCollaborationCommand(collaborationId, "dispatch", version, client);
}

export async function refreshSupplierCollaboration(
  collaborationId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<SupplierCollaboration> {
  return supplierCollaborationCommand(collaborationId, "refresh", version, client);
}

export async function cancelSupplierCollaboration(
  collaborationId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<SupplierCollaboration> {
  return supplierCollaborationCommand(collaborationId, "cancel", version, client);
}

export async function reviewSupplierCollaboration(
  collaborationId: string,
  version: number,
  input: ReviewSupplierCollaborationInput,
  client: IndustrialApiClient = industrialApi,
): Promise<SupplierCollaboration> {
  const { data, error, response } = await client.POST(
    "/api/v1/supplier-collaborations/{collaboration_id}/reviews",
    {
      body: input,
      params: {
        path: { collaboration_id: collaborationId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

async function supplierCollaborationCommand(
  collaborationId: string,
  action: "dispatch" | "refresh" | "cancel",
  version: number,
  client: IndustrialApiClient,
): Promise<SupplierCollaboration> {
  const params = {
    path: { collaboration_id: collaborationId },
    header: { "If-Match": `"${version}"` },
  };
  const result = action === "dispatch"
    ? await client.POST(
        "/api/v1/supplier-collaborations/{collaboration_id}/dispatch",
        { params },
      )
    : action === "refresh"
      ? await client.POST(
          "/api/v1/supplier-collaborations/{collaboration_id}/refresh",
          { params },
        )
      : await client.POST(
          "/api/v1/supplier-collaborations/{collaboration_id}/cancel",
          { params },
        );
  if (!result.data) throw toApiError(result.response, result.error);
  return result.data.data;
}

export async function listMaintenancePlanningCouncils(
  incidentId?: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ councils: MaintenancePlanningCouncil[]; requestId: string; total: number }> {
  const { data, error, response } = await client.GET(
    "/api/v1/maintenance-planning-councils",
    { params: { query: { incident_id: incidentId } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    councils: data.data,
    requestId: String(data.meta.request_id),
    total: Number(data.meta.total),
  };
}

export async function createMaintenancePlanningCouncil(
  input: CreateMaintenancePlanningInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<MaintenancePlanningCouncil> {
  const { data, error, response } = await client.POST(
    "/api/v1/maintenance-planning-councils",
    {
      body: input,
      params: { header: { "Idempotency-Key": idempotencyKey } },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function getMaintenancePlanningCouncil(
  councilId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<MaintenancePlanningCouncil> {
  const { data, error, response } = await client.GET(
    "/api/v1/maintenance-planning-councils/{council_id}",
    { params: { path: { council_id: councilId } } },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function retryMaintenancePlanningCouncil(
  councilId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<MaintenancePlanningCouncil> {
  const { data, error, response } = await client.POST(
    "/api/v1/maintenance-planning-councils/{council_id}/retry",
    {
      params: {
        path: { council_id: councilId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function reviewMaintenancePlanningCouncil(
  councilId: string,
  version: number,
  input: ReviewMaintenancePlanningInput,
  client: IndustrialApiClient = industrialApi,
): Promise<MaintenancePlanningCouncil> {
  const { data, error, response } = await client.POST(
    "/api/v1/maintenance-planning-councils/{council_id}/review",
    {
      body: input,
      params: {
        path: { council_id: councilId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listMaintenancePlanningEvaluationSuites(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  suites: MaintenancePlanningEvaluationSuite[];
  requestId: string;
  total: number;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/maintenance-planning-evaluation-suites",
  );
  if (!data) throw toApiError(response, error);
  return {
    suites: data.data,
    requestId: String(data.meta.request_id),
    total: Number(data.meta.total),
  };
}

export async function createMaintenancePlanningEvaluationSuite(
  input: CreateMaintenancePlanningEvaluationSuiteInput,
  client: IndustrialApiClient = industrialApi,
): Promise<MaintenancePlanningEvaluationSuite> {
  const { data, error, response } = await client.POST(
    "/api/v1/maintenance-planning-evaluation-suites",
    { body: input },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listMaintenancePlanningEvaluations(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  evaluations: MaintenancePlanningEvaluationRun[];
  requestId: string;
  total: number;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/maintenance-planning-evaluations",
  );
  if (!data) throw toApiError(response, error);
  return {
    evaluations: data.data,
    requestId: String(data.meta.request_id),
    total: Number(data.meta.total),
  };
}

export async function createMaintenancePlanningEvaluation(
  input: CreateMaintenancePlanningEvaluationRunInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<MaintenancePlanningEvaluationRun> {
  const { data, error, response } = await client.POST(
    "/api/v1/maintenance-planning-evaluations",
    {
      body: input,
      params: { header: { "Idempotency-Key": idempotencyKey } },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function getMaintenancePlanningEvaluation(
  evaluationId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<MaintenancePlanningEvaluationRun> {
  const { data, error, response } = await client.GET(
    "/api/v1/maintenance-planning-evaluations/{evaluation_id}",
    { params: { path: { evaluation_id: evaluationId } } },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function submitMaintenancePlanningBlindJudgment(
  evaluationId: string,
  caseId: string,
  input: SubmitMaintenancePlanningBlindJudgmentInput,
  client: IndustrialApiClient = industrialApi,
): Promise<AnonymousReviewClaim> {
  const { data, error, response } = await client.POST(
    "/api/v1/maintenance-planning-evaluations/{evaluation_id}/cases/{case_id}/judgments",
    {
      body: input,
      params: { path: { evaluation_id: evaluationId, case_id: caseId } },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listMaintenancePlanningActivations(
  targetEnvironment?: "PROJECT_STAGING" | "PRODUCTION",
  client: IndustrialApiClient = industrialApi,
): Promise<{
  activations: MaintenancePlanningActivation[];
  requestId: string;
  total: number;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/maintenance-planning-activations",
    { params: { query: { target_environment: targetEnvironment } } },
  );
  if (!data) throw toApiError(response, error);
  return {
    activations: data.data,
    requestId: String(data.meta.request_id),
    total: Number(data.meta.total),
  };
}

export async function createMaintenancePlanningActivation(
  input: CreateMaintenancePlanningActivationInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<MaintenancePlanningActivation> {
  const { data, error, response } = await client.POST(
    "/api/v1/maintenance-planning-activations",
    {
      body: input,
      params: { header: { "Idempotency-Key": idempotencyKey } },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function getMaintenancePlanningActivation(
  activationId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<MaintenancePlanningActivation> {
  const { data, error, response } = await client.GET(
    "/api/v1/maintenance-planning-activations/{activation_id}",
    { params: { path: { activation_id: activationId } } },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function decideMaintenancePlanningActivation(
  activationId: string,
  version: number,
  input: DecideMaintenancePlanningActivationInput,
  client: IndustrialApiClient = industrialApi,
): Promise<MaintenancePlanningActivation> {
  const { data, error, response } = await client.POST(
    "/api/v1/maintenance-planning-activations/{activation_id}/decision",
    {
      body: input,
      params: {
        path: { activation_id: activationId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function rollbackMaintenancePlanningActivation(
  activationId: string,
  version: number,
  input: RollbackMaintenancePlanningActivationInput,
  client: IndustrialApiClient = industrialApi,
): Promise<MaintenancePlanningActivation> {
  const { data, error, response } = await client.POST(
    "/api/v1/maintenance-planning-activations/{activation_id}/rollback",
    {
      body: input,
      params: {
        path: { activation_id: activationId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function listDeviceFamilies(
  status?: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ profiles: DeviceFamilyProfile[]; requestId: string; total: number }> {
  const { data, error, response } = await client.GET("/api/v1/device-families", {
    params: { query: { status } },
  });
  if (!data) throw toApiError(response, error);
  return {
    profiles: data.data,
    requestId: String(data.meta.request_id),
    total: Number(data.meta.total),
  };
}

export async function createDeviceFamily(
  input: CreateDeviceFamilyInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<DeviceFamilyProfile> {
  const { data, error, response } = await client.POST("/api/v1/device-families", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function getDeviceFamily(
  profileId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<DeviceFamilyProfile> {
  const { data, error, response } = await client.GET(
    "/api/v1/device-families/{profile_id}",
    { params: { path: { profile_id: profileId } } },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function revalidateDeviceFamily(
  profileId: string,
  version: number,
  client: IndustrialApiClient = industrialApi,
): Promise<DeviceFamilyProfile> {
  const { data, error, response } = await client.POST(
    "/api/v1/device-families/{profile_id}/revalidate",
    {
      params: {
        path: { profile_id: profileId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function reviewDeviceFamily(
  profileId: string,
  version: number,
  input: ReviewDeviceFamilyInput,
  client: IndustrialApiClient = industrialApi,
): Promise<DeviceFamilyProfile> {
  const { data, error, response } = await client.POST(
    "/api/v1/device-families/{profile_id}/review",
    {
      body: input,
      params: {
        path: { profile_id: profileId },
        header: { "If-Match": `"${version}"` },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function getRecoveryOverview(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  overview: RecoveryOverview;
  legalActions: string[];
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/recovery/overview");
  if (!data) throw toApiError(response, error);
  return {
    overview: data.data,
    legalActions: data.legal_actions,
    requestId: data.meta.request_id,
  };
}

export async function startRecoveryDrill(
  input: StartRecoveryDrillInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ drill: RecoveryDrill; requestId: string }> {
  const { data, error, response } = await client.POST("/api/v1/recovery/drills", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return { drill: data.data, requestId: data.meta.request_id };
}

export async function getAssuranceOverview(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  overview: AssuranceOverview;
  legalActions: string[];
  requestId: string;
}> {
  const { data, error, response } = await client.GET("/api/v1/assurance/overview");
  if (!data) throw toApiError(response, error);
  return {
    overview: data.data,
    legalActions: data.legal_actions,
    requestId: data.meta.request_id,
  };
}

export async function startSecurityExercise(
  input: StartSecurityExerciseInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ exercise: SecurityExercise; requestId: string }> {
  const { data, error, response } = await client.POST("/api/v1/security-exercises", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return { exercise: data.data, requestId: data.meta.request_id };
}

export async function createProductionAcceptance(
  input: CreateProductionAcceptanceInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ acceptance: ProductionAcceptance; requestId: string }> {
  const { data, error, response } = await client.POST("/api/v1/production-acceptances", {
    body: input,
    params: { header: { "Idempotency-Key": idempotencyKey } },
  });
  if (!data) throw toApiError(response, error);
  return { acceptance: data.data, requestId: data.meta.request_id };
}

export async function signProductionAcceptance(
  acceptanceId: string,
  version: number,
  input: SignProductionAcceptanceInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ acceptance: ProductionAcceptance; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/production-acceptances/{acceptance_id}/signoffs",
    {
      params: {
        path: { acceptance_id: acceptanceId },
        header: { "If-Match": String(version) },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { acceptance: data.data, requestId: data.meta.request_id };
}

export async function getPredictiveMaintenanceOverview(
  client: IndustrialApiClient = industrialApi,
): Promise<{
  overview: PredictiveMaintenanceOverview;
  legalActions: string[];
  requestId: string;
}> {
  const { data, error, response } = await client.GET(
    "/api/v1/predictive-maintenance/overview",
  );
  if (!data) throw toApiError(response, error);
  return {
    overview: data.data,
    legalActions: data.legal_actions,
    requestId: data.meta.request_id,
  };
}

export async function buildTelemetryWindow(
  input: BuildTelemetryWindowInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ window: TelemetryWindow; requestId: string }> {
  const { data, error, response } = await client.POST("/api/v1/telemetry/windows", {
    body: input,
  });
  if (!data) throw toApiError(response, error);
  return { window: data.data, requestId: data.meta.request_id };
}

export async function buildTelemetryDatasetSnapshot(
  input: BuildTelemetryDatasetInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ snapshot: TelemetryDatasetSnapshot; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/predictive-maintenance/dataset-snapshots",
    { body: input },
  );
  if (!data) throw toApiError(response, error);
  return { snapshot: data.data, requestId: data.meta.request_id };
}

export async function buildRulDatasetSnapshot(
  input: BuildTelemetryDatasetInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ snapshot: TelemetryDatasetSnapshot; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/predictive-maintenance/rul-dataset-snapshots",
    { body: input },
  );
  if (!data) throw toApiError(response, error);
  return { snapshot: data.data, requestId: data.meta.request_id };
}

export async function detectTelemetryWindow(
  windowId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{
  result: components["schemas"]["DetectionResponse"];
  requestId: string;
}> {
  const { data, error, response } = await client.POST(
    "/api/v1/telemetry/windows/{window_id}/detect",
    { params: { path: { window_id: windowId } } },
  );
  if (!data) throw toApiError(response, error);
  return { result: data.data, requestId: data.meta.request_id };
}

export async function decidePredictiveAlert(
  alertCandidateId: string,
  version: number,
  input: AlertDecisionInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ candidate: PredictiveAlertCandidate; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/alert-candidates/{alert_candidate_id}/decisions",
    {
      params: {
        path: { alert_candidate_id: alertCandidateId },
        header: { "If-Match": String(version) },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { candidate: data.data, requestId: data.meta.request_id };
}

export async function referPredictiveAlert(
  alertCandidateId: string,
  version: number,
  incidentDraftId: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ candidate: PredictiveAlertCandidate; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/alert-candidates/{alert_candidate_id}/referrals",
    {
      params: {
        path: { alert_candidate_id: alertCandidateId },
        header: { "If-Match": String(version) },
      },
      body: { incident_draft_id: incidentDraftId },
    },
  );
  if (!data) throw toApiError(response, error);
  return { candidate: data.data, requestId: data.meta.request_id };
}

export async function recordPredictiveOutcome(
  input: PredictiveOutcomeInput,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ outcome: PredictiveOutcome; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/predictive-maintenance/outcomes",
    {
      params: { header: { "Idempotency-Key": idempotencyKey } },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { outcome: data.data, requestId: data.meta.request_id };
}

export async function generateRulForecast(
  alertCandidateId: string,
  alertVersion: number,
  idempotencyKey: string,
  client: IndustrialApiClient = industrialApi,
): Promise<{ forecast: RulForecast; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/alert-candidates/{alert_candidate_id}/rul-forecasts",
    {
      params: {
        path: { alert_candidate_id: alertCandidateId },
        header: {
          "If-Match": String(alertVersion),
          "Idempotency-Key": idempotencyKey,
        },
      },
    },
  );
  if (!data) throw toApiError(response, error);
  return { forecast: data.data, requestId: data.meta.request_id };
}

export async function reviewRulForecast(
  rulForecastId: string,
  version: number,
  input: RulForecastDecisionInput,
  client: IndustrialApiClient = industrialApi,
): Promise<{ forecast: RulForecast; requestId: string }> {
  const { data, error, response } = await client.POST(
    "/api/v1/rul-forecast-candidates/{rul_forecast_id}/decisions",
    {
      params: {
        path: { rul_forecast_id: rulForecastId },
        header: { "If-Match": String(version) },
      },
      body: input,
    },
  );
  if (!data) throw toApiError(response, error);
  return { forecast: data.data, requestId: data.meta.request_id };
}

export async function connectAgentEventStream(
  agentRunId: string,
  lastEventId?: string,
  fetchImplementation: typeof fetch = globalThis.fetch,
  signal?: AbortSignal,
): Promise<Response> {
  const baseUrl = typeof window === "undefined"
    ? "http://localhost/api/backend"
    : "/api/backend";
  const headers = new Headers({ Accept: "text/event-stream" });
  if (lastEventId) headers.set("Last-Event-ID", lastEventId);
  const response = await fetchImplementation(
    `${baseUrl}/api/v1/agent-runs/${encodeURIComponent(agentRunId)}/events`,
    { headers, credentials: "same-origin", cache: "no-store", signal },
  );
  if (!response.ok) {
    let error: unknown;
    try {
      error = await response.json();
    } catch {
      error = undefined;
    }
    throw toApiError(response, error);
  }
  return response;
}

export async function connectAgUiEventStream(
  agentRunId: string,
  lastEventId?: string,
  fetchImplementation: typeof fetch = globalThis.fetch,
  signal?: AbortSignal,
): Promise<Response> {
  const baseUrl = typeof window === "undefined"
    ? "http://localhost/api/backend"
    : "/api/backend";
  const headers = new Headers({ Accept: "text/event-stream" });
  if (lastEventId) headers.set("Last-Event-ID", lastEventId);
  const response = await fetchImplementation(
    `${baseUrl}/api/v1/agent-runs/${encodeURIComponent(agentRunId)}/ag-ui/events`,
    { headers, credentials: "same-origin", cache: "no-store", signal },
  );
  if (!response.ok) {
    let error: unknown;
    try {
      error = await response.json();
    } catch {
      error = undefined;
    }
    throw toApiError(response, error);
  }
  return response;
}

export async function runDiagnosisAgent(
  input: AgUiRunAgentInput,
  fetchImplementation: typeof fetch = globalThis.fetch,
  signal?: AbortSignal,
  onEvent?: (event: AgUiTransportEvent) => void,
): Promise<{ events: AgUiTransportEvent[]; profile: string }> {
  const baseUrl = typeof window === "undefined"
    ? "http://localhost/api/backend"
    : "/api/backend";
  const response = await fetchImplementation(
    `${baseUrl}/api/v1/ag-ui/agents/industrial-diagnosis/runs`,
    {
      method: "POST",
      headers: {
        Accept: "text/event-stream",
        "Content-Type": "application/json",
      },
      credentials: "same-origin",
      cache: "no-store",
      body: JSON.stringify(input),
      signal,
    },
  );
  if (!response.ok) {
    let error: unknown;
    try {
      error = await response.json();
    } catch {
      error = undefined;
    }
    throw toApiError(response, error);
  }
  if (!response.body) throw new Error("AG-UI 运行响应没有可读事件流");
  const events = await readAgUiTransportEvents(response.body, onEvent);
  if (!events.some((event) => event.type === "RUN_FINISHED" || event.type === "RUN_ERROR")) {
    throw new Error("AG-UI 运行在终态事件前结束");
  }
  return {
    events,
    profile: response.headers.get("X-AG-UI-Profile") ?? "unknown",
  };
}

async function readAgUiTransportEvents(
  body: ReadableStream<Uint8Array>,
  onEvent?: (event: AgUiTransportEvent) => void,
): Promise<AgUiTransportEvent[]> {
  const events: AgUiTransportEvent[] = [];
  for await (const event of readAgUiEvents(body)) {
    events.push(event);
    onEvent?.(event);
  }
  const failed = events.find((event) => event.type === "RUN_ERROR");
  if (failed) {
    const code = typeof failed.data.code === "string"
      ? failed.data.code
      : "ag_ui_agent_run_failed";
    const fallbackMessage = typeof failed.data.message === "string"
      ? failed.data.message
      : "AG-UI Agent 运行失败";
    const preconditionMessage = AG_UI_PRECONDITION_MESSAGES[code];
    throw new ApiClientError(
      preconditionMessage ? 409 : 500,
      code,
      preconditionMessage ? "conflict" : "agent",
      preconditionMessage ?? fallbackMessage,
      false,
    );
  }
  return events;
}

const AG_UI_PRECONDITION_MESSAGES: Readonly<Record<string, string>> = {
  asset_model_unavailable: "当前设备缺少权威型号信息，请先在“授权设备与售后权益”中补全型号后再启动诊断。",
  incident_not_triaged: "当前 Incident 尚未完成分诊。请先点击“完成分诊”，再启动诊断。",
  evidence_not_confirmed: "当前证据包尚未确认，完成证据确认后才能启动诊断。",
  evidence_not_automation_eligible: "当前证据不满足自动诊断条件，请补充或重新确认证据。",
  active_index_unavailable: "当前没有可用的知识索引版本，暂时无法启动诊断。",
  confirmed_input_invalid: "本次诊断补充关注点格式无效，请填写 2 至 4000 个字符。",
  idempotency_conflict: "本次诊断请求与已有请求冲突，请刷新页面后重试。",
};

async function workOrderRequest(
  method: "GET" | "POST",
  workOrderId: string,
  action:
    | ""
    | "/assign"
    | "/accept"
    | "/start"
    | "/hold"
    | "/resume"
    | "/escalate"
    | "/replan"
    | "/reschedule"
    | "/complete"
    | "/verify"
    | "/close",
  version: number | undefined,
  body: unknown,
  client: IndustrialApiClient,
): Promise<{ workOrder: WorkOrderView; requestId: string }> {
  const path = `/api/v1/work-orders/{work_order_id}${action}` as keyof paths;
  const params = {
    path: { work_order_id: workOrderId },
    ...(version === undefined ? {} : { header: { "If-Match": `"${version}"` } }),
  };
  const result = method === "GET"
    ? await client.GET(path as "/api/v1/work-orders/{work_order_id}", {
        params: { path: { work_order_id: workOrderId } },
      })
    : await client.POST(path as never, { params, ...(body === undefined ? {} : { body }) } as never);
  const envelope = unwrapGenericEnvelope<WorkOrderView>(
    result.response,
    result.data,
    result.error,
  );
  return { workOrder: envelope.data, requestId: envelope.meta.request_id };
}

function unwrapGenericEnvelope<T>(
  response: Response,
  value: unknown,
  error: unknown,
): { data: T; meta: { request_id: string } } {
  if (!value) throw toApiError(response, error);
  if (typeof value !== "object" || !("data" in value) || !("meta" in value)) {
    throw new ApiClientError(
      response.status,
      "invalid_response_envelope",
      "dependency",
      "The service returned an invalid response envelope",
      false,
    );
  }
  return value as { data: T; meta: { request_id: string } };
}

function toApiError(response: Response, value: unknown): ApiClientError {
  const envelope = isErrorEnvelope(value) ? value : undefined;
  return new ApiClientError(
    response.status,
    envelope?.error.code ?? "unexpected_api_error",
    envelope?.error.category ?? "network",
    envelope?.error.message ?? "The service request failed",
    envelope?.error.retryable ?? response.status >= 500,
    envelope?.request_id,
    envelope?.error.details,
  );
}

function isErrorEnvelope(value: unknown): value is ErrorEnvelope {
  if (!value || typeof value !== "object" || !("error" in value)) return false;
  const error = value.error;
  return Boolean(
    error &&
      typeof error === "object" &&
      "code" in error &&
      typeof error.code === "string" &&
      "category" in error &&
      typeof error.category === "string",
  );
}

export type AnonymousReviewClaim = components["schemas"]["AnonymousReviewClaimResponse"];
export type MaintenanceReviewPolicyStatus = components["schemas"]["ReviewPolicyStatus"];

export async function getMaintenanceReviewPolicy(client: IndustrialApiClient = industrialApi): Promise<MaintenanceReviewPolicyStatus> {
  const { data, error, response } = await client.GET("/api/v1/maintenance-planning-evaluations/review-policy");
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function registerMaintenanceReviewRollout(receiptText: string, reason: string,
  client: IndustrialApiClient = industrialApi): Promise<components["schemas"]["ReviewRolloutRegistrationResponse"]> {
  const { data, error, response } = await client.POST("/api/v1/maintenance-planning-evaluations/review-policy/receipts", {
    body: { receipt_text: receiptText, deployment_confirmed: true, reason },
  });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function changeMaintenanceReviewPolicy(enabled: boolean, version: number, reason: string,
  receiptId: string, idempotencyKey: string, client: IndustrialApiClient = industrialApi): Promise<components["schemas"]["ReviewPolicyResponse"]> {
  const params = { header: { "If-Match": String(version), "Idempotency-Key": idempotencyKey } };
  const body = { policy_version: "maintenance-planning-blind-ab-v2" as const, reason };
  const result = enabled
    ? await client.POST("/api/v1/maintenance-planning-evaluations/review-policy/activate", {
        params, body: { ...body, receipt_id: receiptId },
      })
    : await client.POST("/api/v1/maintenance-planning-evaluations/review-policy/deactivate", { params, body });
  if (!result.data) throw toApiError(result.response, result.error);
  return result.data.data;
}

export type AnonymousReviewQueueItem = components["schemas"]["AnonymousReviewQueueItem"];

export async function listAnonymousReviews(client: IndustrialApiClient = industrialApi): Promise<AnonymousReviewQueueItem[]> {
  const { data, error, response } = await client.GET("/api/v1/maintenance-planning-evaluations/review-queue");
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function claimAnonymousReview(evaluationId: string, caseId: string, idempotencyKey: string,
  client: IndustrialApiClient = industrialApi): Promise<AnonymousReviewClaim> {
  const { data, error, response } = await client.POST(
    "/api/v1/maintenance-planning-evaluations/{evaluation_id}/cases/{case_id}/claim", {
      params: { path: { evaluation_id: evaluationId, case_id: caseId },
        header: { "Idempotency-Key": idempotencyKey } },
    });
  if (!data) throw toApiError(response, error);
  return data.data;
}

export async function withdrawAnonymousReview(evaluationId: string, caseId: string,
  input: components["schemas"]["ReviewClaimVersionBody"],
  client: IndustrialApiClient = industrialApi): Promise<AnonymousReviewClaim> {
  const { data, error, response } = await client.POST(
    "/api/v1/maintenance-planning-evaluations/{evaluation_id}/cases/{case_id}/withdraw", {
      params: { path: { evaluation_id: evaluationId, case_id: caseId } }, body: input,
    });
  if (!data) throw toApiError(response, error);
  return data.data;
}
