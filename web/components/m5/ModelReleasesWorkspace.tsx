"use client";

import { releaseStatusColor as statusColor } from "@/lib/ai-status-color";
import { isRecord } from "@/lib/value-guards";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Drawer,
  Form,
  Input,
  InputNumber,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { DeploymentControlPanel } from "@/components/m5/DeploymentControlPanel";
import { EnterpriseReleaseBatchDeploymentPanel } from "@/components/m5/EnterpriseReleaseBatchDeploymentPanel";
import { ModelGatewayControlPanel } from "@/components/m5/ModelGatewayControlPanel";
import { RerankerKServeAcceptancePanel } from "@/components/m5/RerankerKServeAcceptancePanel";
import {
  type EnterpriseCandidateReleaseBatchProgress,
  type ModelEvaluation,
  type ModelRelease,
  type SupplyChainEvidence,
  type TrainingExperiment,
  createModelRelease,
  createSupplyChainSuccessor,
  listComponentSupplyChainEvidence,
  decideModelReleaseApproval,
  listModelEvaluations,
  listModelReleases,
  listEnterpriseCandidateReleaseBatchReviewQueue,
  listSupplyChainEvidence,
  listTrainingExperiments,
  submitModelReleaseApproval,
  validateModelRelease,
} from "@/lib/api/client";

type ReleaseFormValues = {
  evaluation_id: string;
  target_environment: "STAGING" | "PRODUCTION";
  evaluation_admission: "STANDARD" | "STAGING_DIAGNOSIS_SMOKE_14";
  prompt_bundle_id: string;
  rollback_release_id?: string;
  index_release_id: string;
  embedding_evaluation_id: string;
  embedding_supply_chain_evidence_id: string;
  reranker_evaluation_id: string;
  reranker_supply_chain_evidence_id: string;
  vlm_evaluation_id: string;
  vlm_supply_chain_evidence_id: string;
  asr_supply_chain_evidence_id: string;
  asr_evaluation_id: string;
  tts_evaluation_id?: string;
  timeseries_evaluation_id?: string;
  timeseries_supply_chain_evidence_id?: string;
  rul_evaluation_id?: string;
  rul_supply_chain_evidence_id?: string;
  quantization_profile_id: string;
  runtime_profile_id: string;
  supply_chain_evidence_id: string;
  ocr_model_id: string;
  tts_model_id?: string;
  tensor_parallel_size: number;
  max_model_len: number;
  dtype: "bfloat16" | "float16";
  max_num_seqs: number;
  max_lora_rank: number;
  gpu_count: number;
  gpu_model: string;
  gpu_memory_gb: number;
  cpu_architecture: "x86_64" | "aarch64";
  cpu_model: string;
  cpu_cores: number;
  memory_gb: number;
  instruction_set: string;
  llama_threads: number;
  llama_batch_size: number;
  llama_mmap: boolean;
  llama_mlock: boolean;
};

const toolVersions = {
  "asset.get": "1.0.0",
  "warranty.get": "1.0.0",
  "parts.availability": "1.0.0",
  "schedule.availability": "1.0.0",
  "work_orders.history": "1.0.0",
  "work_order.draft": "1.0.0",
  "parts.reserve": "1.0.0",
  "customer.notify": "1.0.0",
  "purchase.request": "1.0.0",
  "refund.request": "1.0.0",
  "work_order.assign": "1.0.0",
  "work_order.close": "1.0.0",
  "equipment.control.*": "1.1.0",
};

type GovernedTtsEvaluation = Pick<
  ModelEvaluation,
  "evaluation_id" | "candidate_experiment_id" | "status" | "decision"
>;
type GovernedTtsExperiment = Pick<TrainingExperiment, "experiment_id" | "method">;
type EnterpriseCandidateReleaseProgress =
  EnterpriseCandidateReleaseBatchProgress["components"][number];

type ApprovalDecisionTarget = {
  releaseId: string;
  releaseVersion: number;
  approvalId: string;
  approvalVersion: number;
  label: string;
  batchKeySha256?: string;
};

export function bindGovernedTtsRelease(
  evaluationId: string,
  evaluations: GovernedTtsEvaluation[],
  experiments: Map<string, GovernedTtsExperiment>,
) {
  const evaluation = evaluations.find((item) => item.evaluation_id === evaluationId);
  if (!evaluation || evaluation.status !== "COMPLETED" || evaluation.decision !== "CANDIDATE") {
    throw new Error("TTS 专项评测未通过或已不可见，请刷新后重试");
  }
  const candidate = experiments.get(evaluation.candidate_experiment_id);
  if (!candidate || candidate.method !== "TTS") {
    throw new Error("TTS 专项评测未绑定合法 TTS 候选实验");
  }
  return {
    componentEvaluationId: evaluation.evaluation_id,
    runtimeModelId: candidate.experiment_id,
  };
}

type MediaComponent = "VLM" | "ASR";
type MediaEvidence = Awaited<ReturnType<typeof listComponentSupplyChainEvidence>>;
const pendingEvidence: MediaEvidence = { evidence: [], bindingStatus: "LOADING", requestId: "" };

function useMediaEvidence(component: MediaComponent, modelId?: string, active = true) {
  const [state, setState] = useState<{ modelId: string; result: MediaEvidence }>();
  useEffect(() => {
    if (!active || !modelId) return;
    let cancelled = false;
    setState(undefined);
    void listComponentSupplyChainEvidence(component, modelId).then((result) => {
      if (!cancelled) setState({ modelId, result });
    }).catch(() => {
      if (!cancelled) setState({ modelId, result: { evidence: [], bindingStatus: "BLOCKED",
        requestId: "", reason: "组件证据查询失败，请刷新后重试" } });
    });
    return () => { cancelled = true; };
  }, [component, modelId, active]);
  return active && modelId && state?.modelId === modelId ? state.result : pendingEvidence;
}

function mediaModel(release: ModelRelease | undefined, component: MediaComponent) {
  const ids = release?.manifest.multimodal_model_ids;
  const value = isRecord(ids) ? ids[component.toLowerCase()] : undefined;
  return typeof value === "string" && value ? value : undefined;
}

function evidenceOptions(result: MediaEvidence) {
  return result.evidence.map((item) => ({
    value: item.evidence_id,
    label: `${item.evidence_id} · ${item.image_repository}@${item.image_digest.slice(0, 19)}…`,
  }));
}

export default function ModelReleasesPage() {
  const [releases, setReleases] = useState<ModelRelease[]>();
  const [evaluations, setEvaluations] = useState<ModelEvaluation[]>([]);
  const [experiments, setExperiments] = useState<TrainingExperiment[]>([]);
  const [supplyChainEvidence, setSupplyChainEvidence] = useState<SupplyChainEvidence[]>([]);
  const [mcpServerVersions, setMcpServerVersions] = useState<Record<string, string>>({});
  const [collectionActions, setCollectionActions] = useState<string[]>([]);
  const [reviewBatches, setReviewBatches] = useState<
    EnterpriseCandidateReleaseBatchProgress[]
  >([]);
  const [reviewQueueAvailable, setReviewQueueAvailable] = useState(false);
  const [reviewQueueSummary, setReviewQueueSummary] = useState({
    pendingDecisions: 0,
    decidableDecisions: 0,
    blockedBySeparationOfDuties: 0,
  });
  const [error, setError] = useState<unknown>();
  const [commandError, setCommandError] = useState<unknown>();
  const [busy, setBusy] = useState<string>();
  const [createOpen, setCreateOpen] = useState(false);
  const [selected, setSelected] = useState<ModelRelease>();
  const [requestedReleaseHandled, setRequestedReleaseHandled] = useState(false);
  const [decisionTarget, setDecisionTarget] = useState<ApprovalDecisionTarget>();
  const [decisionReason, setDecisionReason] = useState("");
  const [form] = Form.useForm<ReleaseFormValues>();
  const [migrationChoices, setMigrationChoices] = useState<Record<string, string | undefined>>({});
  const [migrationFeedback, setMigrationFeedback] = useState<string>();
  const migrationIntent = useRef<{ signature: string; key: string } | undefined>(undefined);
  const migrationVlm = useMediaEvidence("VLM", mediaModel(selected, "VLM"));
  const migrationAsr = useMediaEvidence("ASR", mediaModel(selected, "ASR"));
  const migrationResults = { VLM: migrationVlm, ASR: migrationAsr };
  const migrationIds = (["VLM", "ASR"] as const).filter((name) => mediaModel(selected, name));
  const migrationChoice = (name: MediaComponent) => {
    const choice = migrationChoices[`${selected?.release_id}:${name}`];
    const items = migrationResults[name].evidence;
    return items.find((item) => item.evidence_id === choice)?.evidence_id
      ?? (items.length === 1 ? items[0].evidence_id : undefined);
  };
  const migrationReady = migrationIds.length > 0 && migrationIds.every((name) =>
    migrationResults[name].bindingStatus === "READY" && migrationChoice(name));


  const load = useCallback(async () => {
    const [
      releaseResult,
      evaluationResult,
      experimentResult,
      evidenceResult,
      reviewQueueResult,
    ] = await Promise.allSettled([
      listModelReleases(),
      listModelEvaluations({ limit: 100 }),
      listTrainingExperiments({ limit: 100 }),
      listSupplyChainEvidence(true),
      listEnterpriseCandidateReleaseBatchReviewQueue(),
    ]);
    if (releaseResult.status === "fulfilled") {
      setReleases(releaseResult.value.releases);
      setCollectionActions(releaseResult.value.legalActions);
      setMcpServerVersions(releaseResult.value.mcpServerVersions);
      setError(undefined);
    } else {
      setError(releaseResult.reason);
    }
    if (evaluationResult.status === "fulfilled") {
      setEvaluations(evaluationResult.value.evaluations);
    }
    if (experimentResult.status === "fulfilled") {
      setExperiments(experimentResult.value.experiments);
    }
    if (evidenceResult.status === "fulfilled") {
      setSupplyChainEvidence(evidenceResult.value.evidence);
    }
    if (reviewQueueResult.status === "fulfilled") {
      setReviewBatches(reviewQueueResult.value.batches);
      setReviewQueueAvailable(true);
      setReviewQueueSummary({
        pendingDecisions: reviewQueueResult.value.pendingDecisions,
        decidableDecisions: reviewQueueResult.value.decidableDecisions,
        blockedBySeparationOfDuties:
          reviewQueueResult.value.blockedBySeparationOfDuties,
      });
    } else {
      setReviewBatches([]);
      setReviewQueueAvailable(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  useEffect(() => {
    if (!releases || requestedReleaseHandled) return;
    const requestedReleaseId = new URLSearchParams(window.location.search).get("release_id");
    if (requestedReleaseId) {
      setSelected(releases.find((item) => item.release_id === requestedReleaseId));
    }
    setRequestedReleaseHandled(true);
  }, [releases, requestedReleaseHandled]);

  const totals = useMemo(() => ({
    draft: releases?.filter((item) => item.status === "DRAFT").length ?? 0,
    candidate: releases?.filter((item) => item.status === "CANDIDATE").length ?? 0,
    approval: releases?.filter((item) => item.status === "APPROVAL_PENDING").length ?? 0,
    rejected: releases?.filter((item) => item.status === "REJECTED").length ?? 0,
  }), [releases]);
  const experimentById = useMemo(
    () => new Map(experiments.map((item) => [item.experiment_id, item])),
    [experiments],
  );
  const selectedReleaseEvaluationId = Form.useWatch("evaluation_id", form);
  const evaluationAdmission = Form.useWatch("evaluation_admission", form) ?? "STANDARD";
  const targetEnvironment = Form.useWatch("target_environment", form) ?? "STAGING";
  const selectedAdmission = selected?.manifest.evaluation_admission as Record<string, unknown> | undefined;
  const selectedReleaseEvaluation = evaluations.find(
    (item) => item.evaluation_id === selectedReleaseEvaluationId,
  );
  const selectedReleaseCandidate = selectedReleaseEvaluation
    ? experimentById.get(selectedReleaseEvaluation.candidate_experiment_id)
    : undefined;
  const vlmEvaluationId = Form.useWatch("vlm_evaluation_id", form);
  const asrEvaluationId = Form.useWatch("asr_evaluation_id", form);
  const vlmModelId = evaluations.find((item) => item.evaluation_id === vlmEvaluationId)?.candidate_experiment_id;
  const asrModelId = evaluations.find((item) => item.evaluation_id === asrEvaluationId)?.candidate_experiment_id;
  const vlmEvidence = useMediaEvidence("VLM", vlmModelId, createOpen);
  const asrEvidence = useMediaEvidence("ASR", asrModelId, createOpen);
  useEffect(() => { form.setFieldValue("vlm_supply_chain_evidence_id", undefined); }, [form, vlmEvaluationId]);
  useEffect(() => { form.setFieldValue("asr_supply_chain_evidence_id", undefined); }, [form, asrEvaluationId]);

  async function migrateSupplyChain() {
    if (!selected || !migrationReady || busy === "supply-chain-successor") return;
    const body = {
      vlm_supply_chain_evidence_id: migrationChoice("VLM") ?? null,
      asr_supply_chain_evidence_id: migrationChoice("ASR") ?? null,
    };
    const signature = JSON.stringify([selected.release_id, selected.version, body]);
    if (migrationIntent.current?.signature !== signature) {
      migrationIntent.current = { signature, key: crypto.randomUUID() };
    }
    setBusy("supply-chain-successor");
    setCommandError(undefined);
    setMigrationFeedback(undefined);
    try {
      const result = await createSupplyChainSuccessor(
        selected.release_id, selected.version, body, migrationIntent.current.key,
      );
      setSelected(result);
      setMigrationFeedback("补证版本已创建，仍需执行门禁和独立审批");
      await load();
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  const selectedReleaseIsQuantized = selectedReleaseCandidate?.method === "QUANTIZATION";
  const selectedReleaseIsEdge =
    selectedReleaseIsQuantized &&
    selectedReleaseCandidate?.training_config.target_runtime === "LLAMA_CPP";

  async function create(values: ReleaseFormValues) {
    setBusy("create");
    setCommandError(undefined);
    try {
      const componentEvaluationIds: Record<string, string> = {
        embedding: values.embedding_evaluation_id,
        reranker: values.reranker_evaluation_id,
        vlm: values.vlm_evaluation_id,
        asr: values.asr_evaluation_id,
      };
      if (values.timeseries_evaluation_id) {
        componentEvaluationIds.timeseries = values.timeseries_evaluation_id;
      }
      if (values.rul_evaluation_id) {
        componentEvaluationIds.rul = values.rul_evaluation_id;
      }
      const ttsBinding = values.tts_evaluation_id
        ? bindGovernedTtsRelease(values.tts_evaluation_id, evaluations, experimentById)
        : undefined;
      if (ttsBinding) componentEvaluationIds.tts = ttsBinding.componentEvaluationId;
      const ttsModelId = ttsBinding?.runtimeModelId ?? values.tts_model_id?.trim();
      if (!ttsModelId) {
        throw new Error("必须选择已通过的 TTS 专项评测或填写既有外部 TTS 运行时 ID");
      }
      const componentModelIds = Object.fromEntries(
        Object.entries(componentEvaluationIds).map(([component, evaluationId]) => {
          const evaluation = evaluations.find((item) => item.evaluation_id === evaluationId);
          const experiment = evaluation
            ? experimentById.get(evaluation.candidate_experiment_id)
            : undefined;
          if (!evaluation || !experiment) {
            throw new Error(`${component} 专项评测或候选实验已不可见，请刷新后重试`);
          }
          return [component, experiment.experiment_id];
        }),
      );
      for (const [name, result] of [["vlm", vlmEvidence], ["asr", asrEvidence]] as const) {
        const modelId = name === "vlm" ? vlmModelId : asrModelId;
        if (result.bindingStatus !== "READY" || componentModelIds[name] !== modelId
          || !result.evidence.some((item) => item.evidence_id === values[`${name}_supply_chain_evidence_id`])) {
          throw new Error(`${name.toUpperCase()} 证据未就绪或已不匹配，请重新选择`);
        }
      }
      const evidence = supplyChainEvidence.find(
        (item) => item.evidence_id === values.supply_chain_evidence_id,
      );
      if (!evidence) {
        throw new Error("所选供应链证据已不可见或未通过门禁，请刷新后重试");
      }
      const rerankerEvidence = supplyChainEvidence.find(
        (item) => item.evidence_id === values.reranker_supply_chain_evidence_id,
      );
      if (!rerankerEvidence) {
        throw new Error("Reranker 推理镜像与模型制品证据已不可见，请刷新后重试");
      }
      const embeddingEvidence = supplyChainEvidence.find(
        (item) => item.evidence_id === values.embedding_supply_chain_evidence_id,
      );
      if (!embeddingEvidence) {
        throw new Error("Embedding 推理镜像与模型制品证据已不可见，请刷新后重试");
      }
      const timeseriesEvidence = values.timeseries_supply_chain_evidence_id
        ? supplyChainEvidence.find(
            (item) => item.evidence_id === values.timeseries_supply_chain_evidence_id,
          )
        : undefined;
      if (values.timeseries_evaluation_id && !timeseriesEvidence) {
        throw new Error("选择时序候选后必须绑定其已验证运行镜像与模型制品证据");
      }
      const rulEvidence = values.rul_supply_chain_evidence_id
        ? supplyChainEvidence.find(
            (item) => item.evidence_id === values.rul_supply_chain_evidence_id,
          )
        : undefined;
      if (values.rul_evaluation_id && !rulEvidence) {
        throw new Error("选择 RUL 候选后必须绑定其已验证运行镜像与模型制品证据");
      }
      const mainEvaluation = evaluations.find(
        (item) => item.evaluation_id === values.evaluation_id,
      );
      const mainCandidate = mainEvaluation
        ? experimentById.get(mainEvaluation.candidate_experiment_id)
        : undefined;
      if (!mainEvaluation || !mainCandidate) {
        throw new Error("所选候选评测或训练实验已不可见，请刷新后重试");
      }
      const quantized = mainCandidate.method === "QUANTIZATION";
      const edge = quantized && mainCandidate.training_config.target_runtime === "LLAMA_CPP";
      const boundQuantizationProfile = quantized
        ? String(mainCandidate.training_config.quantization_profile_id ?? "")
        : values.quantization_profile_id;
      if (!boundQuantizationProfile) {
        throw new Error("量化候选缺少受控 quantization_profile_id");
      }
      await createModelRelease({
        evaluation_id: values.evaluation_id,
        component_evaluation_ids: componentEvaluationIds,
        target_environment: values.target_environment,
        rollback_release_id: values.rollback_release_id || null,
        index_release_id: values.index_release_id,
        quantization_profile_id: boundQuantizationProfile,
        runtime_profile_id: values.runtime_profile_id,
        runtime_image_repository: evidence.image_repository,
        runtime_image_digest: evidence.image_digest,
        prompt_bundle_id: values.prompt_bundle_id,
        evaluation_admission: values.evaluation_admission,
        // The business publishing form never creates an isolated evaluation draft.
        prompt_evaluation_only: false,
        agent_graph_id: "diagnosis-graph-v2",
        tool_versions: toolVersions,
        mcp_server_versions: mcpServerVersions,
        embedding_model_id: componentModelIds.embedding,
        reranker_model_id: componentModelIds.reranker,
        multimodal_model_ids: {
          ocr: values.ocr_model_id,
          vlm: componentModelIds.vlm,
          asr: componentModelIds.asr,
          tts: ttsModelId,
        },
        inference_config: edge
          ? {
              engine: "llama.cpp",
              context_size: values.max_model_len,
              threads: values.llama_threads,
              batch_size: values.llama_batch_size,
              mmap: values.llama_mmap,
              mlock: values.llama_mlock,
            }
          : {
              engine: "vllm",
              tensor_parallel_size: values.tensor_parallel_size,
              max_model_len: values.max_model_len,
              dtype: values.dtype,
              max_num_seqs: values.max_num_seqs,
              enable_lora: !quantized,
              max_lora_rank: quantized ? 0 : values.max_lora_rank,
            },
        hardware_profile: edge
          ? {
              accelerator: "CPU",
              cpu_architecture: values.cpu_architecture,
              cpu_model: values.cpu_model,
              cpu_cores: values.cpu_cores,
              memory_gb: values.memory_gb,
              instruction_set: values.instruction_set,
            }
          : {
              accelerator: "NVIDIA_GPU",
              gpu_count: values.gpu_count,
              gpu_model: values.gpu_model,
              gpu_memory_gb: values.gpu_memory_gb,
            },
        supply_chain_evidence_id: evidence.evidence_id,
        local_staging_adapter: false,
        vlm_supply_chain_evidence_id: values.vlm_supply_chain_evidence_id,
        asr_supply_chain_evidence_id: values.asr_supply_chain_evidence_id,
        timeseries_model_id: componentModelIds.timeseries ?? null,
        timeseries_supply_chain_evidence_id: timeseriesEvidence?.evidence_id ?? null,
        rul_model_id: componentModelIds.rul ?? null,
        rul_supply_chain_evidence_id: rulEvidence?.evidence_id ?? null,
        reranker_supply_chain_evidence_id: rerankerEvidence.evidence_id,
        embedding_supply_chain_evidence_id: embeddingEvidence.evidence_id,
      }, crypto.randomUUID());
      setCreateOpen(false);
      form.resetFields();
      await load();
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function transition(
    release: ModelRelease,
    command: "validate" | "submit",
  ) {
    setBusy(`${command}-${release.release_id}`);
    setCommandError(undefined);
    try {
      if (command === "validate") {
        await validateModelRelease(release.release_id, release.version);
      } else {
        await submitModelReleaseApproval(release.release_id, release.version);
      }
      await load();
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function decide(decision: "APPROVED" | "REJECTED") {
    const target = decisionTarget;
    if (!target) return;
    setBusy(`decide-${target.releaseId}`);
    setCommandError(undefined);
    try {
      await decideModelReleaseApproval(
        target.releaseId,
        target.releaseVersion,
        target.approvalVersion,
        decision,
        decisionReason,
      );
      setDecisionTarget(undefined);
      setDecisionReason("");
      await load();
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  function openReleaseDecision(release: ModelRelease) {
    if (!release.approval) return;
    setDecisionReason("");
    setDecisionTarget({
      releaseId: release.release_id,
      releaseVersion: release.version,
      approvalId: release.approval.approval_id,
      approvalVersion: release.approval.version,
      label: release.release_id,
    });
  }

  function openBatchDecision(
    batch: EnterpriseCandidateReleaseBatchProgress,
    component: EnterpriseCandidateReleaseProgress,
  ) {
    if (
      !component.can_decide_approval
      || !component.approval_id
      || component.approval_version == null
    ) return;
    setDecisionReason("");
    setDecisionTarget({
      releaseId: component.release_id,
      releaseVersion: component.release_version,
      approvalId: component.approval_id,
      approvalVersion: component.approval_version,
      label: `${component.component} · ${component.release_id}`,
      batchKeySha256: batch.batch_key_sha256,
    });
  }

  function closeDecision() {
    setDecisionTarget(undefined);
    setDecisionReason("");
  }

  return (
    <AppShell>
      <div className="page-stack">
        <div>
          <Typography.Title level={2}>模型发布与 AIReleaseManifest</Typography.Title>
          <Typography.Paragraph type="secondary">
            将 Adapter、冻结评测、Prompt、Agent 图、工具、知识索引、vLLM 镜像、硬件与供应链证据锁定为同一个最小灰度和回滚单元。
          </Typography.Paragraph>
        </div>
        <Alert
          showIcon
          type="info"
          message="审批通过不等于已经上线"
          description="本页完成离线门禁和职责分离审批。只有部署控制器收到真实 Shadow/Canary 路由与观察证据后，才允许进入后续状态。"
        />
        {commandError ? <ErrorState error={commandError} onRetry={() => setCommandError(undefined)} /> : null}
        <Row gutter={[16, 16]}>
          <Col xs={12} lg={6}><Card><Statistic title="待验证" value={totals.draft} /></Card></Col>
          <Col xs={12} lg={6}><Card><Statistic title="候选" value={totals.candidate} /></Card></Col>
          <Col xs={12} lg={6}><Card><Statistic title="待审批" value={totals.approval} /></Card></Col>
          <Col xs={12} lg={6}><Card><Statistic title="已拒绝" value={totals.rejected} /></Card></Col>
        </Row>
        <RerankerKServeAcceptancePanel />
        {reviewQueueAvailable ? (
          <Card
            title="三组件独立审批工作台"
            extra={(
              <Space wrap>
                <Tag color="blue">待审 {reviewQueueSummary.pendingDecisions}</Tag>
                <Tag color="green">可决策 {reviewQueueSummary.decidableDecisions}</Tag>
                {reviewQueueSummary.blockedBySeparationOfDuties > 0 ? (
                  <Tag color="orange">
                    职责分离阻止 {reviewQueueSummary.blockedBySeparationOfDuties}
                  </Tag>
                ) : null}
              </Space>
            )}
          >
            <Alert
              showIcon
              type="info"
              message="集中复核不等于批量批准"
              description="同一批次的 LLM、VLM、RUL 候选集中展示，但每个组件必须单独填写审批理由并独立作出批准或拒绝决定；申请人不能审批自己的候选。"
              style={{ marginBottom: 16 }}
            />
            {reviewBatches.length === 0 ? (
              <EmptyState description="当前没有待独立审批的三组件候选批次" />
            ) : (
              <Table<EnterpriseCandidateReleaseBatchProgress>
                rowKey={(batch) => (
                  `${batch.requested_by_subject_id}:${batch.batch_key_sha256}`
                )}
                dataSource={reviewBatches}
                pagination={false}
                scroll={{ x: 1100 }}
                columns={[
                  {
                    title: "批次与基线",
                    width: 280,
                    render: (_, batch) => (
                      <Space direction="vertical" size={2}>
                        <Typography.Text code>
                          {batch.batch_key_sha256.slice(0, 16)}
                        </Typography.Text>
                        <Typography.Text type="secondary">
                          基线 {batch.baseline_release_id}
                        </Typography.Text>
                      </Space>
                    ),
                  },
                  {
                    title: "申请人与进度",
                    width: 260,
                    render: (_, batch) => (
                      <Space direction="vertical" size={2}>
                        <Typography.Text>{batch.requested_by_subject_id}</Typography.Text>
                        <Space wrap>
                          <Tag color={statusColor(batch.status)}>{batch.status}</Tag>
                          <Typography.Text type="secondary">
                            可决策 {batch.decidable_count}/{batch.approval_pending_count}
                          </Typography.Text>
                        </Space>
                      </Space>
                    ),
                  },
                  {
                    title: "LLM / VLM / RUL 独立决策",
                    render: (_, batch) => (
                      <Space direction="vertical" size="small" style={{ width: "100%" }}>
                        {batch.components.map((component) => (
                          <Space key={component.release_id} wrap>
                            <Tag color="blue">{component.component}</Tag>
                            <Typography.Text code>{component.release_id}</Typography.Text>
                            <Tag color={statusColor(component.release_status)}>
                              {component.release_status}
                            </Tag>
                            <Tag color={component.approval_status === "PENDING" ? "gold" : "green"}>
                              审批 {component.approval_status ?? "未提交"}
                            </Tag>
                            {component.can_decide_approval ? (
                              <Button
                                type="primary"
                                size="small"
                                aria-label={`审批 ${component.component}`}
                                onClick={() => openBatchDecision(batch, component)}
                              >
                                审批 {component.component}
                              </Button>
                            ) : component.approval_status === "PENDING" ? (
                              <Tag color="orange">申请人不可自审</Tag>
                            ) : (
                              <Tag>无需决策</Tag>
                            )}
                          </Space>
                        ))}
                      </Space>
                    ),
                  },
                ]}
              />
            )}
          </Card>
        ) : null}
        {collectionActions.includes("CREATE_MODEL_RELEASE") ? (
          <div><Button type="primary" onClick={() => setCreateOpen(true)}>登记发布清单</Button></div>
        ) : null}
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : !releases ? (
          <LoadingState />
        ) : releases.length === 0 ? (
          <EmptyState description="尚未登记发布清单" />
        ) : (
          <Table
            rowKey="release_id"
            dataSource={releases}
            pagination={false}
            scroll={{ x: 1200 }}
            columns={[
              { title: "状态", dataIndex: "status", render: (value: string) => <Tag color={statusColor(value)}>{value}</Tag> },
              { title: "环境", dataIndex: "target_environment" },
              { title: "发布 ID", dataIndex: "release_id", ellipsis: true },
              { title: "候选实验", dataIndex: "candidate_experiment_id", ellipsis: true },
              { title: "Manifest", dataIndex: "manifest_hash", render: (value: string) => <Typography.Text code>{value.slice(0, 16)}</Typography.Text> },
              { title: "流量", dataIndex: "traffic_percent", render: (value: number) => `${value}%` },
              { title: "失败原因", dataIndex: "failure_reason", ellipsis: true, render: (value?: string) => value ?? "—" },
              {
                title: "操作",
                fixed: "right",
                width: 300,
                render: (_, row) => (
                  <Space wrap>
                    <Button type="link" onClick={() => { setCommandError(undefined); setMigrationFeedback(undefined); setSelected(row); }}>清单与历史</Button>
                    {row.legal_actions.includes("VALIDATE_MODEL_RELEASE") ? (
                      <Button loading={busy === `validate-${row.release_id}`} onClick={() => void transition(row, "validate")}>执行门禁</Button>
                    ) : null}
                    {row.legal_actions.includes("SUBMIT_MODEL_RELEASE_APPROVAL") ? (
                      <Button type="primary" loading={busy === `submit-${row.release_id}`} onClick={() => void transition(row, "submit")}>提交审批</Button>
                    ) : null}
                    {row.legal_actions.includes("DECIDE_MODEL_RELEASE_APPROVAL") ? (
                      <Button type="primary" onClick={() => openReleaseDecision(row)}>审批</Button>
                    ) : null}
                  </Space>
                ),
              },
            ]}
          />
        )}
        <EnterpriseReleaseBatchDeploymentPanel onChanged={load} />
        {releases ? <DeploymentControlPanel releases={releases} onChanged={load} /> : null}
        <ModelGatewayControlPanel />
      </div>

      <Modal
        title="登记不可变发布清单"
        open={createOpen}
        width={960}
        onCancel={() => setCreateOpen(false)}
        onOk={() => form.submit()}
        confirmLoading={busy === "create"}
        okText="登记 DRAFT"
      >
        <Alert type="info" showIcon message="运行镜像、SBOM、漏洞、许可证、签名与来源修订由已验证证据自动绑定，页面不再接收手工摘要。" style={{ marginBottom: 16 }} />
        <Form form={form} layout="vertical" onFinish={(values) => void create(values)} initialValues={{
          target_environment: "STAGING",
          evaluation_admission: "STANDARD",
          prompt_bundle_id: "industrial-diagnosis-agent-v2",
          quantization_profile_id: "none-bfloat16",
          runtime_profile_id: "vllm-peft-v1",
          tensor_parallel_size: 1,
          max_model_len: 32768,
          dtype: "bfloat16",
          max_num_seqs: 128,
          max_lora_rank: 64,
          gpu_count: 1,
          gpu_memory_gb: 80,
          cpu_architecture: "x86_64",
          cpu_model: "Intel Xeon / AMD EPYC",
          cpu_cores: 16,
          memory_gb: 64,
          instruction_set: "AVX2",
          llama_threads: 16,
          llama_batch_size: 512,
          llama_mmap: true,
          llama_mlock: false,
        }}>
          <Typography.Title level={5}>候选、知识与回滚</Typography.Title>
          <Row gutter={16}>
            <Col span={12}><Required name="evaluation_admission" label="评测接入方式"><Select onChange={() => form.setFieldValue("evaluation_id", undefined)} options={[
              { value: "STANDARD", label: "标准发布规则" },
              { value: "STAGING_DIAGNOSIS_SMOKE_14", label: "仅暂存：14条诊断 SMOKE", disabled: targetEnvironment !== "STAGING" },
            ]} /></Required></Col>
            <Col span={12}><Required name="prompt_bundle_id" label="提示词 ID（须与评测绑定一致并通过审批）"><Input /></Required></Col>
            {evaluationAdmission === "STAGING_DIAGNOSIS_SMOKE_14" ? <Col span={24}><Alert showIcon type="warning" message="仅允许暂存接入，不代表14条全部通过" description="保留原始分数、SMOKE判定与组件级评测类型；仍须独立审批。服务端核对冻结14条、提示词及Adapter绑定。" style={{ marginBottom: 16 }} /></Col> : null}
            <Col span={12}><Required name="evaluation_id" label={evaluationAdmission === "STANDARD" ? "Agent Runtime Gold 候选" : "14条诊断 SMOKE 结果"}><Select showSearch onChange={(evaluationId: string) => {
              const evaluation = evaluations.find((item) => item.evaluation_id === evaluationId);
              const candidate = evaluation ? experimentById.get(evaluation.candidate_experiment_id) : undefined;
              if (candidate?.method === "QUANTIZATION") {
                const edge = candidate.training_config.target_runtime === "LLAMA_CPP";
                form.setFieldsValue({
                  quantization_profile_id: String(candidate.training_config.quantization_profile_id ?? ""),
                  runtime_profile_id: edge ? "llama-cpp-edge-v1" : "vllm-quantized-v1",
                });
              }
            }} options={evaluationOptions(evaluations, experimentById, ["LORA", "QLORA", "DPO", "GRPO", "QUANTIZATION"], evaluationAdmission)} /></Required></Col>
            <Col span={6}><Required name="target_environment" label="目标环境"><Select onChange={(value) => {
              if (value === "PRODUCTION") form.setFieldsValue({ evaluation_admission: "STANDARD", evaluation_id: undefined });
            }} options={[{ value: "STAGING" }, { value: "PRODUCTION" }]} /></Required></Col>
            <Col span={6}><Form.Item name="rollback_release_id" label="回滚版本"><Select allowClear showSearch options={(releases ?? []).filter((item) => ["PRODUCTION", "RETIRED"].includes(item.status)).map((item) => ({ value: item.release_id, label: item.release_id }))} /></Form.Item></Col>
            <Col span={12}><Required name="index_release_id" label="已发布知识索引 ID"><Input /></Required></Col>
          </Row>
          <Typography.Title level={5}>专项评测与训练制品</Typography.Title>
          <Alert type="info" showIcon message="模型标识由服务端绑定到所选评测的候选实验和唯一训练制品，不能在页面手工填写。" style={{ marginBottom: 16 }} />
          <Row gutter={16}>
            <Col span={12}><Required name="embedding_evaluation_id" label="Embedding Retrieval Gold"><Select showSearch options={evaluationOptions(evaluations, experimentById, ["EMBEDDING"])} /></Required></Col>
            <Col span={12}>
              <Required name="embedding_supply_chain_evidence_id" label="Embedding 推理镜像与模型制品证据">
                <Select
                  showSearch
                  optionFilterProp="label"
                  options={supplyChainEvidence.map((item) => ({
                    value: item.evidence_id,
                    label: `${item.image_repository}@${item.image_digest.slice(0, 19)}…`,
                  }))}
                />
              </Required>
            </Col>
            <Col span={12}><Required name="reranker_evaluation_id" label="Reranker Retrieval Gold"><Select showSearch options={evaluationOptions(evaluations, experimentById, ["RERANKER"])} /></Required></Col>
            <Col span={12}>
              <Required name="reranker_supply_chain_evidence_id" label="Reranker 推理镜像与模型制品证据">
                <Select
                  showSearch
                  optionFilterProp="label"
                  options={supplyChainEvidence.map((item) => ({
                    value: item.evidence_id,
                    label: `${item.image_repository}@${item.image_digest.slice(0, 19)}…`,
                  }))}
                />
              </Required>
            </Col>
            <Col span={12}><Required name="vlm_evaluation_id" label="VLM Multimodal Gold"><Select showSearch options={evaluationOptions(evaluations, experimentById, ["VLM"])} /></Required></Col>
            <Col span={12}><Required name="asr_evaluation_id" label="ASR Gold"><Select showSearch options={evaluationOptions(evaluations, experimentById, ["ASR"])} /></Required></Col>
            {([["vlm", vlmEvidence], ["asr", asrEvidence]] as const).map(([name, result]) => (
              <Col span={12} key={name}>
                <Required name={`${name}_supply_chain_evidence_id`} label={`${name.toUpperCase()} 运行镜像与模型制品证据`}>
                  <Select showSearch optionFilterProp="label" loading={result.bindingStatus === "LOADING"}
                    disabled={result.bindingStatus !== "READY"} options={evidenceOptions(result)} />
                </Required>
                {result.reason ? <Alert type="warning" showIcon message={result.reason} /> : null}
              </Col>
            ))}
            <Col span={12}>
              <Form.Item name="tts_evaluation_id" label="TTS Gold（平台训练候选，可选）">
                <Select
                  allowClear
                  showSearch
                  onChange={(evaluationId?: string) => {
                    form.setFieldValue(
                      "tts_model_id",
                      evaluationId
                        ? bindGovernedTtsRelease(evaluationId, evaluations, experimentById).runtimeModelId
                        : undefined,
                    );
                  }}
                  options={evaluationOptions(evaluations, experimentById, ["TTS"])}
                />
              </Form.Item>
            </Col>
            <Col span={12}><Form.Item name="timeseries_evaluation_id" label="设备时序 Transformer Gold（可选）"><Select allowClear showSearch options={evaluationOptions(evaluations, experimentById, ["TIMESERIES_TRANSFORMER"])} /></Form.Item></Col>
            <Col span={12}>
              <Form.Item
                noStyle
                shouldUpdate={(previous, current) => previous.timeseries_evaluation_id !== current.timeseries_evaluation_id}
              >
                {({ getFieldValue }) => (
                  <Form.Item
                    name="timeseries_supply_chain_evidence_id"
                    label="时序推理镜像与模型制品证据"
                    rules={[{ required: Boolean(getFieldValue("timeseries_evaluation_id")) }]}
                  >
                    <Select
                      allowClear
                      disabled={!getFieldValue("timeseries_evaluation_id")}
                      showSearch
                      optionFilterProp="label"
                      options={supplyChainEvidence.map((item) => ({
                        value: item.evidence_id,
                        label: `${item.image_repository}@${item.image_digest.slice(0, 19)}…`,
                      }))}
                    />
                  </Form.Item>
                )}
              </Form.Item>
            </Col>
            <Col span={12}><Form.Item name="rul_evaluation_id" label="RUL Transformer Gold（可选）"><Select allowClear showSearch options={evaluationOptions(evaluations, experimentById, ["RUL_TRANSFORMER"])} /></Form.Item></Col>
            <Col span={12}>
              <Form.Item
                noStyle
                shouldUpdate={(previous, current) => previous.rul_evaluation_id !== current.rul_evaluation_id}
              >
                {({ getFieldValue }) => (
                  <Form.Item
                    name="rul_supply_chain_evidence_id"
                    label="RUL 推理镜像与模型制品证据"
                    rules={[{ required: Boolean(getFieldValue("rul_evaluation_id")) }]}
                  >
                    <Select
                      allowClear
                      disabled={!getFieldValue("rul_evaluation_id")}
                      showSearch
                      optionFilterProp="label"
                      options={supplyChainEvidence.map((item) => ({
                        value: item.evidence_id,
                        label: `${item.image_repository}@${item.image_digest.slice(0, 19)}…`,
                      }))}
                    />
                  </Form.Item>
                )}
              </Form.Item>
            </Col>
          </Row>
          <Typography.Title level={5}>推理运行时与{selectedReleaseIsEdge ? "边缘 CPU" : " GPU"}</Typography.Title>
          {selectedReleaseIsEdge ? <Alert type="info" showIcon message="该候选将发布最终 GGUF 制品，并由 llama.cpp ServingRuntime 加载；源 LoRA 仅保留为审计绑定。" style={{ marginBottom: 16 }} /> : null}
          <Row gutter={16}>
            <Col span={6}><Required name="runtime_profile_id" label="运行 Profile"><Input /></Required></Col>
            <Col span={6}><Required name="quantization_profile_id" label="量化 Profile"><Input /></Required></Col>
            {selectedReleaseIsEdge ? <>
              <Col span={6}><Required name="cpu_architecture" label="CPU 架构"><Select options={[{ value: "x86_64" }, { value: "aarch64" }]} /></Required></Col>
              <Col span={12}><Required name="cpu_model" label="CPU 型号"><Input /></Required></Col>
              <Col span={6}><Required name="instruction_set" label="指令集"><Input placeholder="AVX2 / ARMv8.2-A" /></Required></Col>
              <Col span={6}><Required name="cpu_cores" label="CPU 核数"><InputNumber min={1} max={512} style={{ width: "100%" }} /></Required></Col>
              <Col span={6}><Required name="memory_gb" label="内存 GB"><InputNumber min={1} max={4096} style={{ width: "100%" }} /></Required></Col>
              <Col span={6}><Required name="llama_threads" label="推理线程"><InputNumber min={1} max={256} style={{ width: "100%" }} /></Required></Col>
              <Col span={6}><Required name="llama_batch_size" label="Batch Token"><InputNumber min={1} max={4096} style={{ width: "100%" }} /></Required></Col>
              <Col span={6}><Required name="llama_mmap" label="mmap"><Select options={[{ value: true, label: "启用" }, { value: false, label: "关闭" }]} /></Required></Col>
              <Col span={6}><Required name="llama_mlock" label="mlock"><Select options={[{ value: false, label: "关闭" }, { value: true, label: "启用" }]} /></Required></Col>
            </> : <>
              <Col span={6}><Required name="gpu_model" label="GPU 型号"><Input placeholder="NVIDIA H100 80GB HBM3" /></Required></Col>
              <Col span={4}><Required name="gpu_count" label="GPU 数"><InputNumber min={1} style={{ width: "100%" }} /></Required></Col>
              <Col span={4}><Required name="gpu_memory_gb" label="单卡显存 GB"><InputNumber min={1} style={{ width: "100%" }} /></Required></Col>
              <Col span={4}><Required name="tensor_parallel_size" label="TP"><InputNumber min={1} style={{ width: "100%" }} /></Required></Col>
              <Col span={6}><Required name="dtype" label="精度"><Select options={[{ value: "bfloat16" }, { value: "float16" }]} /></Required></Col>
            </>}
            <Col span={8}><Required name="max_model_len" label="最大上下文"><InputNumber min={1024} style={{ width: "100%" }} /></Required></Col>
            {!selectedReleaseIsEdge ? <Col span={8}><Required name="max_num_seqs" label="最大并发序列"><InputNumber min={1} style={{ width: "100%" }} /></Required></Col> : null}
            {!selectedReleaseIsEdge && !selectedReleaseIsQuantized ? <Col span={8}><Required name="max_lora_rank" label="最大 LoRA Rank"><InputNumber min={1} style={{ width: "100%" }} /></Required></Col> : null}
          </Row>
          <Typography.Title level={5}>未进入专项训练的兼容组件</Typography.Title>
          <Row gutter={16}>
            <Col span={12}><Required name="ocr_model_id" label="OCR 模型"><Input /></Required></Col>
            <Col span={12}>
              <Form.Item noStyle shouldUpdate={(previous, current) => previous.tts_evaluation_id !== current.tts_evaluation_id}>
                {({ getFieldValue }) => getFieldValue("tts_evaluation_id") ? (
                  <Form.Item name="tts_model_id" label="TTS 运行时候选（由评测自动绑定）">
                    <Input disabled />
                  </Form.Item>
                ) : (
                  <Required name="tts_model_id" label="既有外部 TTS 运行时 ID"><Input /></Required>
                )}
              </Form.Item>
            </Col>
          </Row>
          <Typography.Title level={5}>供应链证据</Typography.Title>
          <Alert type="success" showIcon message="这里只显示 Cosign 签名有效、漏洞门禁通过且许可证已批准的不可变证据。" style={{ marginBottom: 16 }} />
          <Row gutter={16}>
            <Col span={24}><Required name="supply_chain_evidence_id" label="已验证证据"><Select showSearch optionFilterProp="label" options={supplyChainEvidence.map((item) => ({
              value: item.evidence_id,
              label: `${item.image_repository}@${item.image_digest.slice(0, 19)}… · ${item.maximum_vulnerability_severity} · ${item.source_revision.slice(0, 12)}`,
            }))} /></Required></Col>
          </Row>
        </Form>
      </Modal>

      <Drawer title="发布清单与审计历史" width={760} open={Boolean(selected)} onClose={() => setSelected(undefined)}>
        {selected ? <Space direction="vertical" size="large" style={{ width: "100%" }}>
          {migrationFeedback ? <Alert type="success" showIcon message={migrationFeedback} /> : null}
          {commandError ? <ErrorState error={commandError} /> : null}
          <Card size="small" title="VLM / ASR 供应链补证">
            <Alert type="info" showIcon
              message="运行中不等于供应链通过；补证创建新版本，不修改旧记录"
              description="先完成可信签名、扫描及模型来源登记。仅显示与该组件精确匹配的证据；创建后仍需执行门禁和独立审批，不会自动部署。" />
            {migrationIds.map((name) => {
              const result = migrationResults[name];
              const components = selected.manifest.specialized_components;
              const component = isRecord(components) ? components[name.toLowerCase()] : undefined;
              const rawReceipt = isRecord(component) ? component.supply_chain : undefined;
              const receipt = isRecord(rawReceipt) ? rawReceipt : undefined;
              const legacyMissing = !receipt && selected.manifest.schema_version === "ai-release-manifest/v4";
              const current = result.evidence.some((item) =>
                item.evidence_id === receipt?.evidence_id && item.verification_hash === receipt?.verification_hash);
              return <div key={name} style={{ marginTop: 12 }}>
                <Typography.Text strong>{name}：{legacyMissing ? "旧版本待补证" : current ? "已有匹配证据，发布前仍需复核" : "绑定待核对"}</Typography.Text>
                <Select aria-label={`${name} 补证证据`} style={{ width: "100%" }}
                  loading={result.bindingStatus === "LOADING"} disabled={result.bindingStatus !== "READY"}
                  value={migrationChoice(name)} placeholder="选择该模型的已验证证据"
                  onChange={(value) => setMigrationChoices((old) => ({ ...old, [`${selected.release_id}:${name}`]: value }))}
                  options={evidenceOptions(result)} />
                {result.reason ? <Alert type="warning" showIcon message={result.reason} /> : null}
              </div>;
            })}
            {migrationIds.length === 0 ? <Typography.Paragraph>此清单未声明 VLM / ASR 运行绑定，无需添加不存在的组件。</Typography.Paragraph> : null}
            {collectionActions.includes("CREATE_MODEL_RELEASE") && migrationIds.length > 0 ? (
              <Button type="primary" aria-label="创建同模型补证版本" aria-busy={busy === "supply-chain-successor"} style={{ marginTop: 12 }}
                disabled={!migrationReady || !["ai-release-manifest/v4", "ai-release-manifest/v5"].includes(String(selected.manifest.schema_version))}
                loading={busy === "supply-chain-successor"} onClick={() => void migrateSupplyChain()}>
                创建同模型补证版本
              </Button>
            ) : null}
          </Card>
          {selected.manifest.local_staging_adapter === true ? <Alert type="warning" showIcon
            message="本地暂存 Adapter · 未签名、未扫描"
            description="仅允许本地 STAGING 联调。制品摘要已绑定，不代表供应链签名验证或质量全部通过；独立审批和生产发布规则保持。" /> : null}
          <Descriptions bordered size="small" column={1}>
            <Descriptions.Item label="发布 ID">{selected.release_id}</Descriptions.Item>
            <Descriptions.Item label="Manifest hash"><Typography.Text code copyable>{selected.manifest_hash}</Typography.Text></Descriptions.Item>
            <Descriptions.Item label="审批">{selected.approval ? `${selected.approval.status} · v${selected.approval.version}` : "未提交"}</Descriptions.Item>
          </Descriptions>
          {selectedAdmission?.mode === "STAGING_DIAGNOSIS_SMOKE_14" ? <Alert type="warning" showIcon message="14条SMOKE暂存接入 · 不具备生产发布资格" description={`原始成绩 ${Number(selectedAdmission.candidate_score).toFixed(4)}，样本数 ${selectedAdmission.sample_count}；原判定 ${selectedAdmission.decision}；MODEL_COMPONENT不是业务运行评测。独立审批规则保持不变。`} /> : null}
          <Card size="small" title="AIReleaseManifest"><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(selected.manifest, null, 2)}</pre></Card>
          <Table rowKey="transition_id" size="small" pagination={false} dataSource={selected.transitions} columns={[
            { title: "序号", dataIndex: "sequence" },
            { title: "迁移", render: (_, row) => `${row.from_status} → ${row.to_status}` },
            { title: "原因", dataIndex: "reason_code" },
            { title: "操作者", dataIndex: "actor_subject_id", ellipsis: true },
            { title: "时间", dataIndex: "occurred_at", render: (value: string) => new Date(value).toLocaleString() },
          ]} />
        </Space> : null}
      </Drawer>

      <Modal title={decisionTarget ? `发布职责分离审批 · ${decisionTarget.label}` : "发布职责分离审批"} open={Boolean(decisionTarget)} onCancel={closeDecision} footer={[
        <Button key="reject" danger disabled={decisionReason.trim().length < 3} loading={busy === `decide-${decisionTarget?.releaseId}`} onClick={() => void decide("REJECTED")}>拒绝</Button>,
        <Button key="approve" type="primary" disabled={decisionReason.trim().length < 3} loading={busy === `decide-${decisionTarget?.releaseId}`} onClick={() => void decide("APPROVED")}>批准进入部署队列</Button>,
      ]}>
        <Alert type="info" showIcon message="审批绑定当前 Manifest hash；申请人与审批人必须是不同主体。" style={{ marginBottom: 16 }} />
        {decisionTarget ? (
          <Descriptions bordered size="small" column={1} style={{ marginBottom: 16 }}>
            <Descriptions.Item label="发布 ID">{decisionTarget.releaseId}</Descriptions.Item>
            <Descriptions.Item label="审批 ID">{decisionTarget.approvalId}</Descriptions.Item>
            {decisionTarget.batchKeySha256 ? (
              <Descriptions.Item label="三组件批次">
                {decisionTarget.batchKeySha256.slice(0, 16)}
              </Descriptions.Item>
            ) : null}
          </Descriptions>
        ) : null}
        <Input.TextArea rows={4} value={decisionReason} onChange={(event) => setDecisionReason(event.target.value)} placeholder="填写可审计的审批依据（至少 3 个字符）" />
      </Modal>
    </AppShell>
  );
}

function Required({ name, label, children }: { name: keyof ReleaseFormValues; label: string; children: ReactNode }) {
  return <Form.Item name={name} label={label} rules={[{ required: true }]}>{children}</Form.Item>;
}

function evaluationOptions(
  evaluations: ModelEvaluation[],
  experimentById: Map<string, TrainingExperiment>,
  allowedMethods: string[],
  admission: "STANDARD" | "STAGING_DIAGNOSIS_SMOKE_14" = "STANDARD",
) {
  return evaluations
    .filter((evaluation) => {
      const experiment = experimentById.get(evaluation.candidate_experiment_id);
      return evaluation.status === "COMPLETED"
        && (admission === "STANDARD"
          ? evaluation.decision === "CANDIDATE"
          : evaluation.decision === "SMOKE_PASSED" && experiment?.task_type === "DIAGNOSIS" && experiment.method !== "QUANTIZATION")
        && Boolean(experiment && allowedMethods.includes(experiment.method));
    })
    .map((evaluation) => {
      const method = experimentById.get(evaluation.candidate_experiment_id)?.method;
      return {
        value: evaluation.evaluation_id,
        label: `${method} · ${evaluation.evaluation_id} · ${evaluation.primary_metric}`,
      };
    });
}
