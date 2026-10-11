"use client";

import { experimentStatusColor as statusColor } from "@/lib/ai-status-color";
import { normalizeStableId } from "@/lib/value-guards";

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
  List,
  Modal,
  Pagination,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AppShell } from "@/components/AppShell";
import {
  type OperationReceipt,
  OperationReceiptAlert,
} from "@/components/m3/OperationReceiptAlert";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  buildEvaluationDetailUrl,
  buildExperimentDetailUrl,
  readEvaluationDetailTarget,
  readExperimentCreateTarget,
  readExperimentDetailTarget,
  removeExperimentCreateTarget,
} from "@/lib/experiment-workspace-location";
import {
  type DatasetSnapshot,
  type EvaluationJobStatus,
  type EvaluationTargetProfile,
  type EvaluationPolicy,
  type EvaluationJob,
  type EvaluationSuite,
  type ModelEvaluation,
  type ModelEvaluationDecision,
  type TrainingExperiment,
  type TrainingExperimentMethod,
  type TrainingExperimentStatus,
  createTrainingExperiment,
  createEvaluationJob,
  createEvaluationPolicy,
  createEvaluationSuite,
  downloadModelEvaluationEvidence,
  getDatasetSnapshot,
  getModelEvaluation,
  getTrainingExperiment,
  listDatasetSnapshots,
  listEvaluationPolicies,
  listEvaluationJobs,
  listEvaluationSuites,
  listModelEvaluations,
  listTrainingExperiments,
  startTrainingExperiment,
} from "@/lib/api/client";

type ExperimentFormValues = {
  comparison_group_id: string;
  method:
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
  task_type: string;
  dataset_snapshot_id: string;
  base_model_id: string;
  base_model_revision: string;
  base_model_digest: string;
  tokenizer_digest: string;
  chat_template_digest: string;
  git_commit: string;
  container_digest: string;
  max_steps: number;
  effective_batch_size: number;
  evaluation_interval: number;
  random_seed: number;
  mlflow_experiment_name: string;
  beta?: number;
  loss_type?: "sigmoid" | "hinge" | "ipo";
  group_size?: number;
  max_completion_length?: number;
  reward_model_id?: string;
  reward_model_revision?: string;
  reward_model_digest?: string;
  value_model_id?: string;
  value_model_revision?: string;
  value_model_digest?: string;
  kl_coefficient?: number;
  total_episodes?: number;
  response_length?: number;
  num_ppo_epochs?: number;
  num_mini_batches?: number;
  hard_negatives_per_query?: number;
  similarity_scale?: number;
  positive_weight?: number;
  max_image_pixels?: number;
  sampling_rate?: number;
  language?: string;
  max_audio_seconds?: number;
  voice_profile_id?: string;
  speaker_embedding_dimension?: number;
  d_model?: number;
  nhead?: number;
  num_layers?: number;
  dim_feedforward?: number;
  mask_probability?: number;
  rule_threshold?: number;
  empirical_p10_minutes?: number;
  empirical_p50_minutes?: number;
  empirical_p90_minutes?: number;
  history_cutoff?: string;
  source_sample_count?: number;
  source_experiment_id?: string;
  source_artifact_id?: string;
  source_artifact_hash?: string;
  quantization_profile?: "AWQ" | "GPTQ" | "FP8" | "GGUF_Q4" | "GGUF_Q5" | "GGUF_Q8";
  target_hardware_profile?: string;
  tool_version?: string;
  calibration_sample_count?: number;
  quantization_max_sequence_length?: number;
  incompatibilities?: string;
  quantization_gpu_hourly_cost_usd?: number;
  distributed_strategy?: "single_gpu" | "deepspeed" | "fsdp";
  distributed_world_size?: number;
  deepspeed_zero_stage?: 2 | 3;
  offload_optimizer_device?: "none" | "cpu";
  offload_param_device?: "none" | "cpu";
  fsdp_sharding_strategy?: "FULL_SHARD" | "SHARD_GRAD_OP";
  fsdp_transformer_layer_cls_to_wrap?: string;
  training_start_mode?: "fresh" | "resume";
  resume_source_experiment_id?: string;
  resume_artifact_id?: string;
  resume_artifact_hash?: string;
  resume_checkpoint_path?: string;
};

type EvaluationJobFormValues = {
  candidate_experiment_id: string;
  baseline_experiment_id: string;
  suite_id: string;
  policy_id: string;
  target_profile: EvaluationTargetProfile;
  runner_git_commit: string;
  container_digest: string;
  max_new_tokens?: number;
  top_k?: number;
  max_image_pixels?: number;
  sampling_rate?: number;
  language?: string;
  max_audio_seconds?: number;
  asr_verifier_model_id?: string;
  asr_verifier_revision?: string;
  asr_verifier_digest?: string;
  anomaly_threshold?: number;
  rule_threshold?: number;
  precision: "bfloat16" | "float16";
  gpu_hourly_cost_usd: number;
  cpu_hourly_cost_usd?: number;
  threads?: number;
  context_size?: number;
  framework_mode: "DETERMINISTIC" | "RAGAS_DEEPEVAL";
};

type PpoSafetyMetrics = Record<string, unknown>;

function ttsCandidateMetric(metrics: Record<string, unknown>, key: string) {
  const scoped = metrics.tts && typeof metrics.tts === "object"
    ? metrics.tts as Record<string, unknown>
    : metrics;
  const value = scoped[key];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (value && typeof value === "object") {
    const candidate = (value as Record<string, unknown>).candidate;
    if (typeof candidate === "number" && Number.isFinite(candidate)) return candidate;
  }
  return 0;
}

export function TtsObjectiveEvidencePanel({
  metrics,
}: {
  metrics: Record<string, unknown>;
}) {
  const evidence = [
    ["安全短语完整率", "safety_phrase_completeness"],
    ["工业术语召回率", "terminology_recall"],
    ["可懂度", "intelligibility"],
    ["音频完整率", "audio_integrity_rate"],
  ] as const;
  return (
    <Card title="TTS 客观评测证据">
      <Alert
        type="warning"
        showIcon
        message="客观门禁通过不等同于主观听感验收"
        description="MOS 仍需在目标环境完成独立人工验收，且不由浏览器上传或替代服务端客观分值。"
        style={{ marginBottom: 16 }}
      />
      <Row gutter={16}>
        {evidence.map(([label, key]) => (
          <Col span={6} key={key}>
            <Statistic title={label} value={percentage(ttsCandidateMetric(metrics, key))} />
          </Col>
        ))}
      </Row>
    </Card>
  );
}

export function PpoResearchSafetyPanel({
  metrics,
}: {
  metrics?: PpoSafetyMetrics;
}) {
  const consistency = finiteMetric(metrics?.reward_consistency_rate);
  const attackRate = finiteMetric(metrics?.candidate_attack_success_rate);
  const safeRate = finiteMetric(metrics?.candidate_safe_response_rate);
  return (
    <Alert
      type="warning"
      showIcon
      message="PPO 研究安全评测"
      description={(
        <Space direction="vertical" size="small" style={{ width: "100%" }}>
          <Typography.Text>
            该结果仅是奖励一致性与 reward-hacking 攻击的研究证据，不具备发布或 Canary 资格。
          </Typography.Text>
          {metrics ? (
            <Descriptions bordered size="small" column={3}>
              <Descriptions.Item label="奖励一致率">
                {metricPercentage(consistency)}
              </Descriptions.Item>
              <Descriptions.Item label="攻击成功率">
                {metricPercentage(attackRate)}
              </Descriptions.Item>
              <Descriptions.Item label="安全响应率">
                {metricPercentage(safeRate)}
              </Descriptions.Item>
            </Descriptions>
          ) : null}
        </Space>
      )}
      style={{ marginBottom: 16 }}
    />
  );
}

type SequenceEvaluationAssetsFormValues = {
  evaluation_kind: "TIMESERIES" | "RUL";
  source_snapshot_id: string;
  suite_name: string;
  suite_version: string;
  tier: "SMOKE" | "GOLD";
  policy_name: string;
  policy_version: string;
  false_positive_rate_max: number;
  miss_rate_max: number;
  median_absolute_error_minutes_max: number;
  interval_coverage_min: number;
  interval_coverage_max: number;
  pinball_loss_max: number;
};

const EXPERIMENT_PAGE_SIZE = 12;
const EVALUATION_JOB_PAGE_SIZE = 20;
const MODEL_EVALUATION_PAGE_SIZE = 20;
const CURRENT_REWARD_CONTRACT_VERSION = "industrial-json-structural-v2";
const CURRENT_REWARD_CONTRACT_DIGEST =
  "sha256:e143f5417bb0a6920a0a73301bd9b03dfdd88718fba744556774a23a7d4794ba";
export const TTS_TRAINING_METHOD = "TTS" as const;
export const TTS_EVALUATION_PROFILE = "TTS_COMPONENT" as const;
const TRAINING_METHODS: TrainingExperimentMethod[] = [
  "BASELINE", "LORA", "QLORA", "DPO", "GRPO", "PPO",
  "EMBEDDING", "RERANKER", "VLM", "ASR", TTS_TRAINING_METHOD,
  "QUANTIZATION",
  "TIMESERIES_TRANSFORMER",
  "TIMESERIES_RULE_BASELINE",
  "RUL_TRANSFORMER",
  "RUL_EMPIRICAL_BASELINE",
];
const EXPERIMENT_STATUSES: TrainingExperimentStatus[] = [
  "PLANNED",
  "TRACKING_PENDING",
  "TRACKING_FAILED",
  "RUNNING",
  "COMPLETION_PENDING",
  "COMPLETION_FAILED",
  "COMPLETED",
  "FAILED",
  "CLOSED_NO_GAIN",
];
const EVALUATION_JOB_STATUSES: EvaluationJobStatus[] = [
  "PLANNED", "RUNNING", "COMPLETED", "FAILED",
];
const MODEL_EVALUATION_DECISIONS: ModelEvaluationDecision[] = [
  "CANDIDATE", "REJECTED", "SMOKE_PASSED", "NO_GAIN",
];
const QUANTIZATION_PROFILES = {
  AWQ: { id: "awq-w4a16_asym-vllm", algorithm: "AWQ", scheme: "W4A16_ASYM", runtime: "VLLM" },
  GPTQ: { id: "gptq-w4a16-vllm", algorithm: "GPTQ", scheme: "W4A16", runtime: "VLLM" },
  FP8: { id: "fp8-fp8_dynamic-vllm", algorithm: "FP8", scheme: "FP8_DYNAMIC", runtime: "VLLM" },
  GGUF_Q4: { id: "gguf-q4_k_m-llama_cpp", algorithm: "GGUF", scheme: "Q4_K_M", runtime: "LLAMA_CPP" },
  GGUF_Q5: { id: "gguf-q5_k_m-llama_cpp", algorithm: "GGUF", scheme: "Q5_K_M", runtime: "LLAMA_CPP" },
  GGUF_Q8: { id: "gguf-q8_0-llama_cpp", algorithm: "GGUF", scheme: "Q8_0", runtime: "LLAMA_CPP" },
} as const;

function quantizationProfile(value: ExperimentFormValues["quantization_profile"]) {
  if (!value) throw new Error("必须选择量化配置");
  return QUANTIZATION_PROFILES[value];
}


export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<TrainingExperiment[]>();
  const [experimentOptions, setExperimentOptions] = useState<TrainingExperiment[]>([]);
  const [experimentTotal, setExperimentTotal] = useState(0);
  const [experimentPage, setExperimentPage] = useState(1);
  const [experimentMethod, setExperimentMethod] = useState<TrainingExperimentMethod>();
  const [experimentStatus, setExperimentStatus] = useState<TrainingExperimentStatus>();
  const [experimentIdFilter, setExperimentIdFilter] = useState<string>();
  const [experimentIdDraft, setExperimentIdDraft] = useState("");
  const [experimentSnapshotFilter, setExperimentSnapshotFilter] = useState<string>();
  const [experimentSnapshotDraft, setExperimentSnapshotDraft] = useState("");
  const [suites, setSuites] = useState<EvaluationSuite[]>();
  const [policies, setPolicies] = useState<EvaluationPolicy[]>();
  const [evaluations, setEvaluations] = useState<ModelEvaluation[]>();
  const [evaluationTotal, setEvaluationTotal] = useState(0);
  const [evaluationPage, setEvaluationPage] = useState(1);
  const [evaluationDecision, setEvaluationDecision] = useState<ModelEvaluationDecision>();
  const [evaluationIdFilter, setEvaluationIdFilter] = useState<string>();
  const [evaluationIdDraft, setEvaluationIdDraft] = useState("");
  const [evaluationCandidateFilter, setEvaluationCandidateFilter] = useState<string>();
  const [evaluationCandidateDraft, setEvaluationCandidateDraft] = useState("");
  const [evaluationBaselineFilter, setEvaluationBaselineFilter] = useState<string>();
  const [evaluationBaselineDraft, setEvaluationBaselineDraft] = useState("");
  const [evaluationJobs, setEvaluationJobs] = useState<EvaluationJob[]>();
  const [evaluationJobTotal, setEvaluationJobTotal] = useState(0);
  const [evaluationJobPage, setEvaluationJobPage] = useState(1);
  const [evaluationJobStatus, setEvaluationJobStatus] = useState<EvaluationJobStatus>();
  const [evaluationJobIdFilter, setEvaluationJobIdFilter] = useState<string>();
  const [evaluationJobIdDraft, setEvaluationJobIdDraft] = useState("");
  const [candidateExperimentFilter, setCandidateExperimentFilter] = useState<string>();
  const [candidateExperimentDraft, setCandidateExperimentDraft] = useState("");
  const [baselineExperimentFilter, setBaselineExperimentFilter] = useState<string>();
  const [baselineExperimentDraft, setBaselineExperimentDraft] = useState("");
  const [snapshots, setSnapshots] = useState<DatasetSnapshot[]>([]);
  const [experimentActions, setExperimentActions] = useState<string[]>([]);
  const [evaluationJobActions, setEvaluationJobActions] = useState<string[]>([]);
  const [evaluationSuiteActions, setEvaluationSuiteActions] = useState<string[]>([]);
  const [evaluationPolicyActions, setEvaluationPolicyActions] = useState<string[]>([]);
  const [experimentError, setExperimentError] = useState<unknown>();
  const [evaluationJobError, setEvaluationJobError] = useState<unknown>();
  const [evaluationError, setEvaluationError] = useState<unknown>();
  const [governanceError, setGovernanceError] = useState<unknown>();
  const [commandError, setCommandError] = useState<unknown>();
  const [operationReceipt, setOperationReceipt] = useState<OperationReceipt>();
  const [createTargetError, setCreateTargetError] = useState<string>();
  const [selected, setSelected] = useState<TrainingExperiment>();
  const [selectedEvaluation, setSelectedEvaluation] = useState<ModelEvaluation>();
  const [createOpen, setCreateOpen] = useState(false);
  const [evaluationJobOpen, setEvaluationJobOpen] = useState(false);
  const [evaluationAssetsOpen, setEvaluationAssetsOpen] = useState(false);
  const [busy, setBusy] = useState<string>();
  const [form] = Form.useForm<ExperimentFormValues>();
  const selectedMethod = Form.useWatch("method", form);
  const selectedDistributedStrategy = Form.useWatch("distributed_strategy", form);
  const selectedDeepSpeedZeroStage = Form.useWatch("deepspeed_zero_stage", form);
  const selectedTrainingStartMode = Form.useWatch("training_start_mode", form);
  const selectedQuantizationProfile = Form.useWatch("quantization_profile", form);
  const [evaluationJobForm] = Form.useForm<EvaluationJobFormValues>();
  const [evaluationAssetsForm] = Form.useForm<SequenceEvaluationAssetsFormValues>();
  const selectedEvaluationAssetKind = Form.useWatch(
    "evaluation_kind", evaluationAssetsForm,
  );
  const selectedEvaluationProfile = Form.useWatch("target_profile", evaluationJobForm);
  const selectedEvaluationCandidateId = Form.useWatch(
    "candidate_experiment_id", evaluationJobForm
  );
  const selectedEvaluationCandidate = experimentOptions.find(
    (item) => item.experiment_id === selectedEvaluationCandidateId,
  );
  const selectedEvaluationUsesLlamaCpp =
    selectedEvaluationProfile === "EDGE_MODEL_COMPONENT" ||
    (selectedEvaluationProfile === "AGENT_RUNTIME" &&
      selectedEvaluationCandidate?.training_config.target_runtime === "LLAMA_CPP");
  const createTargetHandledRef = useRef(false);
  const detailTargetRef = useRef<string | undefined>(undefined);
  const evaluationTargetRef = useRef<string | undefined>(undefined);

  const loadExperiments = useCallback(async () => {
    try {
      const result = await listTrainingExperiments({
        experiment_id: experimentIdFilter,
        dataset_snapshot_id: experimentSnapshotFilter,
        method: experimentMethod,
        status: experimentStatus,
        limit: EXPERIMENT_PAGE_SIZE,
        offset: (experimentPage - 1) * EXPERIMENT_PAGE_SIZE,
      });
      setExperiments(result.experiments);
      setExperimentActions(result.legalActions);
      setExperimentTotal(result.total);
      const lastPage = Math.max(1, Math.ceil(result.total / EXPERIMENT_PAGE_SIZE));
      if (experimentPage > lastPage) setExperimentPage(lastPage);
      setExperimentError(undefined);
    } catch (error) {
      setExperimentError(error);
    }
  }, [
    experimentIdFilter,
    experimentMethod,
    experimentPage,
    experimentSnapshotFilter,
    experimentStatus,
  ]);

  const loadSupportingData = useCallback(async () => {
    const [suiteResult, policyResult, snapshotResult] =
      await Promise.allSettled([
        listEvaluationSuites(),
        listEvaluationPolicies(),
        listDatasetSnapshots(),
      ]);
    if (suiteResult.status === "fulfilled" && policyResult.status === "fulfilled") {
      setSuites(suiteResult.value.suites);
      setPolicies(policyResult.value.policies);
      setEvaluationSuiteActions(suiteResult.value.legalActions);
      setEvaluationPolicyActions(policyResult.value.legalActions);
      setGovernanceError(undefined);
    } else {
      setGovernanceError(
        suiteResult.status === "rejected" ? suiteResult.reason :
          policyResult.status === "rejected" ? policyResult.reason : undefined,
      );
    }
    if (snapshotResult.status === "fulfilled") {
      setSnapshots(snapshotResult.value.snapshots.filter((item) => item.training_eligible));
    }
  }, []);

  const loadEvaluationJobs = useCallback(async () => {
    try {
      const result = await listEvaluationJobs({
        job_id: evaluationJobIdFilter,
        candidate_experiment_id: candidateExperimentFilter,
        baseline_experiment_id: baselineExperimentFilter,
        status: evaluationJobStatus,
        limit: EVALUATION_JOB_PAGE_SIZE,
        offset: (evaluationJobPage - 1) * EVALUATION_JOB_PAGE_SIZE,
      });
      setEvaluationJobs(result.jobs);
      setEvaluationJobActions(result.legalActions);
      setEvaluationJobTotal(result.total);
      const lastPage = Math.max(1, Math.ceil(result.total / EVALUATION_JOB_PAGE_SIZE));
      if (evaluationJobPage > lastPage) setEvaluationJobPage(lastPage);
      setEvaluationJobError(undefined);
    } catch (error) {
      setEvaluationJobError(error);
    }
  }, [
    baselineExperimentFilter,
    candidateExperimentFilter,
    evaluationJobIdFilter,
    evaluationJobPage,
    evaluationJobStatus,
  ]);

  const loadEvaluations = useCallback(async () => {
    try {
      const result = await listModelEvaluations({
        evaluation_id: evaluationIdFilter,
        candidate_experiment_id: evaluationCandidateFilter,
        baseline_experiment_id: evaluationBaselineFilter,
        decision: evaluationDecision,
        limit: MODEL_EVALUATION_PAGE_SIZE,
        offset: (evaluationPage - 1) * MODEL_EVALUATION_PAGE_SIZE,
      });
      setEvaluations(result.evaluations);
      setEvaluationTotal(result.total);
      const lastPage = Math.max(1, Math.ceil(result.total / MODEL_EVALUATION_PAGE_SIZE));
      if (evaluationPage > lastPage) setEvaluationPage(lastPage);
      setEvaluationError(undefined);
    } catch (error) {
      setEvaluationError(error);
    }
  }, [
    evaluationBaselineFilter,
    evaluationCandidateFilter,
    evaluationDecision,
    evaluationIdFilter,
    evaluationPage,
  ]);

  useEffect(() => { void loadExperiments(); }, [loadExperiments]);
  useEffect(() => { void loadSupportingData(); }, [loadSupportingData]);
  useEffect(() => { void loadEvaluationJobs(); }, [loadEvaluationJobs]);
  useEffect(() => { void loadEvaluations(); }, [loadEvaluations]);

  useEffect(() => {
    if (experiments === undefined || createTargetHandledRef.current) return;
    const target = readExperimentCreateTarget(window.location.search);
    if (!target) return;
    createTargetHandledRef.current = true;
    window.history.replaceState(
      null,
      "",
      removeExperimentCreateTarget(window.location.href),
    );
    if (!experimentActions.includes("CREATE_EXPERIMENT")) {
      setCreateTargetError("当前账号没有登记训练实验的权限，数据快照未被提交。");
      return;
    }
    void (async () => {
      try {
        const { snapshot } = await getDatasetSnapshot(target.datasetSnapshotId);
        if (!snapshot.training_eligible ||
            !snapshot.legal_actions.includes("CREATE_EXPERIMENT")) {
          const blockers = snapshot.training_blockers.join("、") || "训练用途未获授权";
          setCreateTargetError(`数据快照未通过当前训练门禁：${blockers}`);
          return;
        }
        setSnapshots((current) => current.some(
          (item) => item.snapshot_id === snapshot.snapshot_id,
        ) ? current : [snapshot, ...current]);
        form.setFieldValue("dataset_snapshot_id", snapshot.snapshot_id);
        setCreateOpen(true);
      } catch (error) {
        setCommandError(error);
      }
    })();
  }, [experimentActions, experiments, form]);

  useEffect(() => {
    function applyDetailTarget() {
      const experimentId = readExperimentDetailTarget(window.location.search);
      const targetKey = experimentId ?? "none";
      if (detailTargetRef.current === targetKey) return;
      detailTargetRef.current = targetKey;
      if (experimentId) {
        void openExperiment(experimentId, false);
      } else {
        setSelected(undefined);
      }
    }

    applyDetailTarget();
    window.addEventListener("popstate", applyDetailTarget);
    return () => window.removeEventListener("popstate", applyDetailTarget);
    // Detail restoration is intentionally driven only by mount and browser history.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    function applyEvaluationTarget() {
      const evaluationId = readEvaluationDetailTarget(window.location.search);
      const targetKey = evaluationId ?? "none";
      if (evaluationTargetRef.current === targetKey) return;
      evaluationTargetRef.current = targetKey;
      if (evaluationId) {
        void openEvaluation(evaluationId, false);
      } else {
        setSelectedEvaluation(undefined);
      }
    }

    applyEvaluationTarget();
    window.addEventListener("popstate", applyEvaluationTarget);
    return () => window.removeEventListener("popstate", applyEvaluationTarget);
    // Evaluation restoration is intentionally driven only by mount and browser history.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const totals = useMemo(() => ({
    running: experiments?.filter((item) => item.status === "RUNNING").length ?? 0,
    completed: experiments?.filter((item) => item.status === "COMPLETED").length ?? 0,
    candidate: evaluations?.filter((item) => item.decision === "CANDIDATE").length ?? 0,
    rejected: evaluations?.filter((item) => item.decision === "REJECTED").length ?? 0,
  }), [evaluations, experiments]);

  async function selectTrainingMethod(method: TrainingExperimentMethod) {
    if (method === TTS_TRAINING_METHOD) {
      form.setFieldsValue({
        task_type: "industrial_safety_speech",
        base_model_id: "microsoft/speecht5_tts",
        chat_template_digest: "not-applicable",
        voice_profile_id: "industrial-standard-zh-v1",
        speaker_embedding_dimension: 512,
        sampling_rate: 16000,
        language: "zh-CN",
        max_audio_seconds: 30,
        mlflow_experiment_name: "industrial-ops-tts",
      });
      return;
    }
    if (["TIMESERIES_TRANSFORMER", "TIMESERIES_RULE_BASELINE"].includes(method)) {
      form.setFieldsValue({
        task_type: "predictive_maintenance_anomaly",
        base_model_id: method === "TIMESERIES_TRANSFORMER"
          ? "industrial-timeseries-transformer"
          : "industrial-timeseries-rule-baseline",
        base_model_revision: "timeseries-transformer-v1",
        base_model_digest: "architecture:timeseries-transformer-v1",
        mlflow_experiment_name: "industrial-ops-timeseries",
      });
      return;
    }
    if (["RUL_TRANSFORMER", "RUL_EMPIRICAL_BASELINE"].includes(method)) {
      form.setFieldsValue({
        task_type: "predictive_maintenance_rul",
        base_model_id: method === "RUL_TRANSFORMER"
          ? "industrial-rul-transformer"
          : "empirical-lead-time-baseline",
        base_model_revision: method === "RUL_TRANSFORMER"
          ? "rul-transformer-v1"
          : "empirical-lead-time-baseline-v1",
        base_model_digest: method === "RUL_TRANSFORMER"
          ? "architecture:rul-transformer-v1"
          : "registered:empirical-lead-time-baseline-v1",
        mlflow_experiment_name: "industrial-ops-rul",
      });
      return;
    }
    if (["LORA", "QLORA"].includes(method)) {
      await loadResumeSources(method);
      return;
    }
    if (method !== "QUANTIZATION") return;
    setBusy("load-quantization-sources");
    setCommandError(undefined);
    try {
      const result = await listTrainingExperiments({ status: "COMPLETED", limit: 100, offset: 0 });
      setExperimentOptions(result.experiments.filter((item) =>
        ["LORA", "QLORA", "DPO", "GRPO"].includes(item.method)));
      form.setFieldsValue({ mlflow_experiment_name: "industrial-ops-quantization" });
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function loadResumeSources(method: string | undefined) {
    if (!method || !["LORA", "QLORA"].includes(method)) return;
    setBusy("load-resume-sources");
    setCommandError(undefined);
    try {
      const result = await listTrainingExperiments({ status: "COMPLETED", limit: 100, offset: 0 });
      setExperimentOptions(result.experiments.filter((item) => item.method === method));
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function selectResumeSource(experimentId: string) {
    setBusy("load-resume-source");
    setCommandError(undefined);
    try {
      const result = await getTrainingExperiment(experimentId);
      const source = result.experiment;
      const artifacts = (source.artifacts ?? []).filter((item) => item.kind === "adapter_bundle");
      if (artifacts.length !== 1) {
        throw new Error("续训源实验没有唯一的 Adapter/Checkpoint 制品");
      }
      const artifact = artifacts[0];
      const config = source.training_config as Record<string, unknown>;
      form.setFieldsValue({
        task_type: source.task_type,
        dataset_snapshot_id: source.dataset_snapshot_id,
        base_model_id: source.base_model_id,
        base_model_revision: String(config.base_model_revision ?? ""),
        base_model_digest: source.base_model_digest,
        tokenizer_digest: source.tokenizer_digest,
        chat_template_digest: source.chat_template_digest,
        effective_batch_size: Number(config.effective_batch_size),
        evaluation_interval: Number(config.evaluation_interval),
        random_seed: source.random_seeds[0],
        resume_artifact_id: artifact.artifact_id,
        resume_artifact_hash: artifact.content_hash,
      });
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function selectQuantizationSource(experimentId: string) {
    setBusy("load-quantization-source");
    setCommandError(undefined);
    try {
      const result = await getTrainingExperiment(experimentId);
      const source = result.experiment;
      const artifacts = (source.artifacts ?? []).filter(
        (item) => item.kind === "adapter_bundle"
      );
      if (artifacts.length !== 1) {
        throw new Error("源实验没有唯一的 PEFT Adapter 制品");
      }
      const artifact = artifacts[0];
      const config = source.training_config as Record<string, unknown>;
      form.setFieldsValue({
        comparison_group_id: source.comparison_group_id,
        task_type: source.task_type,
        dataset_snapshot_id: source.dataset_snapshot_id,
        base_model_id: source.base_model_id,
        base_model_revision: String(config.base_model_revision ?? ""),
        base_model_digest: source.base_model_digest,
        tokenizer_digest: source.tokenizer_digest,
        chat_template_digest: source.chat_template_digest,
        max_steps: Number(config.max_steps),
        effective_batch_size: Number(config.effective_batch_size),
        evaluation_interval: Number(config.evaluation_interval),
        random_seed: source.random_seeds[0],
        source_artifact_id: artifact.artifact_id,
        source_artifact_hash: artifact.content_hash,
      });
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function create(values: ExperimentFormValues) {
    setBusy("create");
    setCommandError(undefined);
    setOperationReceipt(undefined);
    try {
      const supportsDistributedSft = ["LORA", "QLORA"].includes(values.method);
      const distributedStrategy = supportsDistributedSft
        ? values.distributed_strategy ?? "single_gpu"
        : "single_gpu";
      const distributedWorldSize = distributedStrategy === "single_gpu"
        ? 1
        : values.distributed_world_size ?? 2;
      const distributedProfile: Record<string, unknown> = {
        strategy: distributedStrategy,
        world_size: distributedWorldSize,
        node_count: 1,
      };
      const resumeConfig = supportsDistributedSft && values.training_start_mode === "resume"
        ? {
            resume_from_checkpoint: {
              source_experiment_id: values.resume_source_experiment_id,
              artifact_id: values.resume_artifact_id,
              content_hash: values.resume_artifact_hash,
              checkpoint_path: values.resume_checkpoint_path,
            },
          }
        : {};
      if (distributedStrategy === "deepspeed") {
        distributedProfile.zero_stage = values.deepspeed_zero_stage ?? 2;
        distributedProfile.offload_optimizer_device = values.offload_optimizer_device ?? "none";
        distributedProfile.offload_param_device = values.offload_param_device ?? "none";
      } else if (distributedStrategy === "fsdp") {
        distributedProfile.sharding_strategy = values.fsdp_sharding_strategy ?? "FULL_SHARD";
        distributedProfile.transformer_layer_cls_to_wrap = (
          values.fsdp_transformer_layer_cls_to_wrap ?? ""
        )
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean);
        distributedProfile.activation_checkpointing = true;
      }
      const postTrainingConfig = values.method === "DPO" ? {
        preference_dataset_snapshot_id: values.dataset_snapshot_id,
        reference_model_digest: values.base_model_digest,
        beta: values.beta,
        loss_type: values.loss_type,
      } : values.method === "GRPO" ? {
        reward_contract_version: CURRENT_REWARD_CONTRACT_VERSION,
        reward_contract_digest: CURRENT_REWARD_CONTRACT_DIGEST,
        group_size: values.group_size,
        beta: values.beta,
        max_completion_length: values.max_completion_length,
      } : values.method === "PPO" ? {
        reward_model_id: values.reward_model_id,
        reward_model_revision: values.reward_model_revision,
        reward_model_digest: values.reward_model_digest,
        value_model_id: values.value_model_id,
        value_model_revision: values.value_model_revision,
        value_model_digest: values.value_model_digest,
        kl_controller: { type: "fixed", coefficient: values.kl_coefficient },
        total_episodes: values.total_episodes,
        response_length: values.response_length,
        num_ppo_epochs: values.num_ppo_epochs,
        num_mini_batches: values.num_mini_batches,
      } : values.method === "EMBEDDING" ? {
        retrieval_contract_version: "industrial-retrieval-triplet-v1",
        hard_negatives_per_query: values.hard_negatives_per_query,
        loss_type: "multiple_negatives_ranking",
        similarity_scale: values.similarity_scale,
      } : values.method === "RERANKER" ? {
        retrieval_contract_version: "industrial-retrieval-triplet-v1",
        hard_negatives_per_query: values.hard_negatives_per_query,
        loss_type: "binary_cross_entropy",
        positive_weight: values.positive_weight,
      } : values.method === "VLM" ? {
        vision_contract_version: "industrial-vision-instruction-v1",
        max_image_pixels: values.max_image_pixels,
        load_in_4bit: true,
      } : values.method === "ASR" ? {
        audio_contract_version: "industrial-asr-transcript-v1",
        sampling_rate: values.sampling_rate,
        language: values.language,
        task: "transcribe",
        max_audio_seconds: values.max_audio_seconds,
      } : values.method === TTS_TRAINING_METHOD ? {
        model_family: "SPEECHT5",
        speech_contract_version: "industrial-tts-speech-v1",
        sampling_rate: values.sampling_rate,
        language: values.language,
        voice_profile_id: values.voice_profile_id,
        speaker_embedding_dimension: values.speaker_embedding_dimension,
        max_audio_seconds: values.max_audio_seconds,
      } : values.method === "TIMESERIES_TRANSFORMER" ? {
        sequence_contract_version: "industrial-telemetry-sequence-v1",
        signal_order: [
          "vibration_rms_mm_s",
          "bearing_temperature_c",
          "motor_current_a",
          "rpm",
        ],
        per_device_train_batch_size: 16,
        per_device_eval_batch_size: 16,
        max_sequence_length: 256,
        d_model: values.d_model,
        nhead: values.nhead,
        num_layers: values.num_layers,
        dim_feedforward: values.dim_feedforward,
        objective: "masked_reconstruction",
        mask_probability: values.mask_probability,
      } : values.method === "TIMESERIES_RULE_BASELINE" ? {
        sequence_contract_version: "industrial-telemetry-sequence-v1",
        signal_order: [
          "vibration_rms_mm_s",
          "bearing_temperature_c",
          "motor_current_a",
          "rpm",
        ],
        rule_threshold: values.rule_threshold,
      } : values.method === "RUL_TRANSFORMER" ? {
        sequence_contract_version: "industrial-rul-sequence-v1",
        signal_order: [
          "vibration_rms_mm_s",
          "bearing_temperature_c",
          "motor_current_a",
          "rpm",
        ],
        per_device_train_batch_size: 16,
        per_device_eval_batch_size: 16,
        max_sequence_length: 256,
        d_model: values.d_model,
        nhead: values.nhead,
        num_layers: values.num_layers,
        dim_feedforward: values.dim_feedforward,
        objective: "quantile_regression",
        quantiles: [0.1, 0.5, 0.9],
        target_transform: "log1p_minutes",
      } : values.method === "RUL_EMPIRICAL_BASELINE" ? {
        sequence_contract_version: "industrial-rul-sequence-v1",
        signal_order: [
          "vibration_rms_mm_s",
          "bearing_temperature_c",
          "motor_current_a",
          "rpm",
        ],
        empirical_quantiles_minutes: [
          values.empirical_p10_minutes,
          values.empirical_p50_minutes,
          values.empirical_p90_minutes,
        ],
        history_cutoff: values.history_cutoff,
        source_sample_count: values.source_sample_count,
        source_manifest_hash: snapshots.find(
          (item) => item.snapshot_id === values.dataset_snapshot_id,
        )?.manifest_hash,
      } : values.method === "QUANTIZATION" ? {
        source_experiment_id: values.source_experiment_id,
        source_artifact_id: values.source_artifact_id,
        source_artifact_hash: values.source_artifact_hash,
        quantization_profile_id: quantizationProfile(values.quantization_profile).id,
        algorithm: quantizationProfile(values.quantization_profile).algorithm,
        scheme: quantizationProfile(values.quantization_profile).scheme,
        target_runtime: quantizationProfile(values.quantization_profile).runtime,
        target_hardware_profile: values.target_hardware_profile,
        tool_version: values.tool_version,
        calibration_sample_count: ["AWQ", "GPTQ"].includes(values.quantization_profile ?? "")
          ? values.calibration_sample_count
          : 0,
        max_sequence_length: values.quantization_max_sequence_length,
        incompatibilities: (values.incompatibilities ?? "")
          .split("\n")
          .map((item) => item.trim())
          .filter(Boolean),
        approval_inheritance: false,
        gpu_hourly_cost_usd: values.quantization_gpu_hourly_cost_usd,
      } : {};
      const result = await createTrainingExperiment(
        {
          comparison_group_id: values.comparison_group_id,
          method: values.method,
          task_type: values.task_type,
          dataset_snapshot_id: values.dataset_snapshot_id,
          base_model_id: values.base_model_id,
          base_model_digest: values.base_model_digest,
          tokenizer_digest: ["TIMESERIES_TRANSFORMER", "TIMESERIES_RULE_BASELINE", "RUL_TRANSFORMER", "RUL_EMPIRICAL_BASELINE"].includes(values.method)
            ? "not-applicable"
            : values.tokenizer_digest,
          chat_template_digest: ["EMBEDDING", "RERANKER", "ASR", TTS_TRAINING_METHOD, "TIMESERIES_TRANSFORMER", "TIMESERIES_RULE_BASELINE", "RUL_TRANSFORMER", "RUL_EMPIRICAL_BASELINE"].includes(values.method)
            ? "not-applicable"
            : values.chat_template_digest,
          git_commit: values.git_commit,
          container_digest: values.container_digest,
          training_config: {
            base_model_revision: values.base_model_revision,
            max_steps: values.max_steps,
            effective_batch_size: values.effective_batch_size,
            evaluation_interval: values.evaluation_interval,
            ...resumeConfig,
            ...postTrainingConfig,
          },
          distributed_profile: distributedProfile,
          random_seeds: [values.random_seed],
          hardware_topology: {
            accelerator: ["TIMESERIES_RULE_BASELINE", "RUL_EMPIRICAL_BASELINE"].includes(values.method)
              ? "CPU"
              : "NVIDIA_GPU",
            count: ["TIMESERIES_RULE_BASELINE", "RUL_EMPIRICAL_BASELINE"].includes(values.method)
              ? 1
              : distributedWorldSize,
            nodes: 1,
          },
          mlflow_experiment_name: values.mlflow_experiment_name,
          license_status: "APPROVED",
        },
        crypto.randomUUID(),
      );
      setOperationReceipt({
        title: "训练实验已登记",
        resourceId: result.experiment_id,
        status: result.status,
        requestId: result.requestId,
      });
      setCreateOpen(false);
      form.resetFields();
      await loadExperiments();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function start(experiment: TrainingExperiment) {
    setBusy(experiment.experiment_id);
    setCommandError(undefined);
    setOperationReceipt(undefined);
    try {
      const result = await startTrainingExperiment(
        experiment.experiment_id,
        experiment.version,
      );
      setOperationReceipt({
        title: "MLflow 训练运行已创建",
        resourceId: result.experiment_id,
        status: result.status,
        requestId: result.requestId,
      });
      await loadExperiments();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function openExperiment(experimentId: string, updateLocation = true) {
    setBusy(experimentId);
    setCommandError(undefined);
    try {
      const result = await getTrainingExperiment(experimentId);
      setSelectedEvaluation(undefined);
      setSelected(result.experiment);
      if (updateLocation) writeExperimentDetailLocation(experimentId);
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  function writeExperimentDetailLocation(experimentId?: string) {
    const targetKey = experimentId ?? "none";
    detailTargetRef.current = targetKey;
    evaluationTargetRef.current = "none";
    const nextUrl = buildExperimentDetailUrl(window.location.href, experimentId);
    const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextUrl !== currentUrl) window.history.pushState(null, "", nextUrl);
  }

  function closeExperiment() {
    setSelected(undefined);
    writeExperimentDetailLocation();
  }

  async function openEvaluation(evaluationId: string, updateLocation = true) {
    setBusy(`evaluation-${evaluationId}`);
    setCommandError(undefined);
    try {
      const result = await getModelEvaluation(evaluationId);
      setSelected(undefined);
      setSelectedEvaluation(result.evaluation);
      if (updateLocation) writeEvaluationDetailLocation(evaluationId);
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  function writeEvaluationDetailLocation(evaluationId?: string) {
    const targetKey = evaluationId ?? "none";
    evaluationTargetRef.current = targetKey;
    detailTargetRef.current = "none";
    const nextUrl = buildEvaluationDetailUrl(window.location.href, evaluationId);
    const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextUrl !== currentUrl) window.history.pushState(null, "", nextUrl);
  }

  function closeEvaluation() {
    setSelectedEvaluation(undefined);
    writeEvaluationDetailLocation();
  }

  async function createEvaluation(values: EvaluationJobFormValues) {
    setBusy("create-evaluation-job");
    setCommandError(undefined);
    setOperationReceipt(undefined);
    try {
      const result = await createEvaluationJob(
        {
          candidate_experiment_id: values.candidate_experiment_id,
          baseline_experiment_id: values.baseline_experiment_id,
          suite_id: values.suite_id,
          policy_id: values.policy_id,
          target_profile: values.target_profile,
          runner_git_commit: values.runner_git_commit,
          container_digest: values.container_digest,
          runner_config:
            selectedEvaluationUsesLlamaCpp
              ? {
                  max_new_tokens: values.max_new_tokens,
                  precision: values.precision,
                  gpu_hourly_cost_usd: values.gpu_hourly_cost_usd,
                  cpu_hourly_cost_usd: values.cpu_hourly_cost_usd,
                  threads: values.threads,
                  context_size: values.context_size,
                  framework_mode: "DETERMINISTIC",
                }
              : values.target_profile === "RETRIEVAL_COMPONENT"
              ? {
                  top_k: values.top_k,
                  precision: values.precision,
                  gpu_hourly_cost_usd: values.gpu_hourly_cost_usd,
                  framework_mode: "DETERMINISTIC",
                }
              : values.target_profile === "VLM_COMPONENT"
                ? {
                    max_new_tokens: values.max_new_tokens,
                    max_image_pixels: values.max_image_pixels,
                    precision: values.precision,
                    gpu_hourly_cost_usd: values.gpu_hourly_cost_usd,
                    framework_mode: "DETERMINISTIC",
                  }
                : values.target_profile === "ASR_COMPONENT"
                  ? {
                      max_new_tokens: values.max_new_tokens,
                      sampling_rate: values.sampling_rate,
                      language: values.language,
                      max_audio_seconds: values.max_audio_seconds,
                      precision: values.precision,
                      gpu_hourly_cost_usd: values.gpu_hourly_cost_usd,
                      framework_mode: "DETERMINISTIC",
                    }
                : values.target_profile === TTS_EVALUATION_PROFILE
                  ? {
                      sampling_rate: values.sampling_rate,
                      language: values.language,
                      max_audio_seconds: values.max_audio_seconds,
                      precision: values.precision,
                      gpu_hourly_cost_usd: values.gpu_hourly_cost_usd,
                      framework_mode: "DETERMINISTIC",
                      asr_verifier_model_id: values.asr_verifier_model_id,
                      asr_verifier_revision: values.asr_verifier_revision,
                      asr_verifier_digest: values.asr_verifier_digest,
                    }
                : values.target_profile === "TIMESERIES_COMPONENT"
                  ? {
                      anomaly_threshold: values.anomaly_threshold,
                      rule_threshold: values.rule_threshold,
                      precision: values.precision,
                      gpu_hourly_cost_usd: values.gpu_hourly_cost_usd,
                      framework_mode: "DETERMINISTIC",
                    }
                : values.target_profile === "RUL_COMPONENT"
                  ? {
                      precision: values.precision,
                      gpu_hourly_cost_usd: values.gpu_hourly_cost_usd,
                      framework_mode: "DETERMINISTIC",
                    }
                : values.target_profile === "PPO_RESEARCH_SAFETY"
                  ? {
                      max_new_tokens: values.max_new_tokens,
                      precision: values.precision,
                      gpu_hourly_cost_usd: values.gpu_hourly_cost_usd,
                      framework_mode: "DETERMINISTIC",
                    }
                : {
                    max_new_tokens: values.max_new_tokens,
                    precision: values.precision,
                    gpu_hourly_cost_usd: values.gpu_hourly_cost_usd,
                    framework_mode: values.framework_mode,
                  },
        },
        crypto.randomUUID(),
      );
      setOperationReceipt({
        title: "独立评测任务已创建",
        resourceId: result.job_id,
        status: result.status,
        requestId: result.requestId,
      });
      setEvaluationJobOpen(false);
      evaluationJobForm.resetFields();
      await loadEvaluationJobs();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function openEvaluationJob() {
    setBusy("load-evaluation-experiments");
    setCommandError(undefined);
    try {
      const [completed, plannedTimeseries, plannedRul] = await Promise.all([
        listTrainingExperiments({ status: "COMPLETED", limit: 100, offset: 0 }),
        listTrainingExperiments({
          method: "TIMESERIES_RULE_BASELINE",
          status: "PLANNED",
          limit: 100,
          offset: 0,
        }),
        listTrainingExperiments({
          method: "RUL_EMPIRICAL_BASELINE",
          status: "PLANNED",
          limit: 100,
          offset: 0,
        }),
      ]);
      setExperimentOptions([
        ...completed.experiments,
        ...plannedTimeseries.experiments,
        ...plannedRul.experiments,
      ]);
      setEvaluationJobOpen(true);
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function registerSequenceEvaluationAssets(
    values: SequenceEvaluationAssetsFormValues,
  ) {
    setBusy("create-sequence-evaluation-assets");
    setCommandError(undefined);
    setOperationReceipt(undefined);
    try {
      const snapshot = snapshots.find(
        (item) => item.snapshot_id === values.source_snapshot_id,
      );
      const expectedContract = values.evaluation_kind === "RUL"
        ? "industrial-rul-sequence-v1"
        : "industrial-telemetry-sequence-v1";
      if (!snapshot?.manifest_hash || snapshot.contract_version !== expectedContract) {
        throw new Error("所选序列快照与评测类型不匹配或没有不可变清单");
      }
      const suite = await createEvaluationSuite({
        name: values.suite_name,
        version: values.suite_version,
        tier: values.tier,
        source_snapshot_id: snapshot.snapshot_id,
        manifest_hash: snapshot.manifest_hash,
        sample_count: snapshot.row_count,
        slice_counts: { all: snapshot.row_count },
      });
      const policy = await createEvaluationPolicy({
        name: values.policy_name,
        version: values.policy_version,
        primary_metric: values.evaluation_kind === "RUL"
          ? "rul_relative_accuracy"
          : "timeseries_accuracy",
        hard_gates: values.evaluation_kind === "RUL" ? {
            data_governance: true,
            rul_artifact_integrity: true,
            rul_target_coverage: true,
            rul_interval_ordering: true,
            rul_median_mae: true,
            rul_pinball_loss: true,
            rul_interval_coverage: true,
            no_blocking_regressions: true,
          } : {
            data_governance: true,
            telemetry_artifact_integrity: true,
            label_coverage: true,
            false_positive_rate: true,
            miss_rate: true,
            no_blocking_regressions: true,
          },
        thresholds: values.evaluation_kind === "RUL" ? {
            quality_lift_min: 0.02,
            quality_tolerance: 0.01,
            efficiency_improvement_min: 0.15,
            bootstrap_iterations: 1000,
            median_absolute_error_minutes_max: values.median_absolute_error_minutes_max,
            interval_coverage_min: values.interval_coverage_min,
            interval_coverage_max: values.interval_coverage_max,
            pinball_loss_max: values.pinball_loss_max,
          } : {
            quality_lift_min: 0.02,
            quality_tolerance: 0.01,
            efficiency_improvement_min: 0.15,
            bootstrap_iterations: 1000,
            false_positive_rate_max: values.false_positive_rate_max,
            miss_rate_max: values.miss_rate_max,
          },
      });
      setOperationReceipt({
        title: values.evaluation_kind === "RUL"
          ? "RUL 评测资产已冻结"
          : "时序评测资产已冻结",
        resourceId: `${suite.suite_id} / ${policy.policy_id}`,
        status: "READY",
        requestId: policy.requestId,
      });
      setEvaluationAssetsOpen(false);
      evaluationAssetsForm.resetFields();
      await loadSupportingData();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function saveEvaluationEvidence(evaluation: ModelEvaluation) {
    setBusy(`evidence-${evaluation.evaluation_id}`);
    setCommandError(undefined);
    setOperationReceipt(undefined);
    try {
      const evidence = await downloadModelEvaluationEvidence(evaluation.evaluation_id);
      const url = URL.createObjectURL(evidence.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = evidence.filename;
      anchor.click();
      URL.revokeObjectURL(url);
      setOperationReceipt({
        title: "评测证据已通过完整性校验",
        resourceId: evaluation.evaluation_id,
        status: evidence.evidenceHash,
        requestId: evidence.requestId,
      });
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  return (
    <AppShell>
      <div className="page-stack">
        <div>
          <Typography.Title level={2}>训练实验与独立评测工作台</Typography.Title>
          <Typography.Paragraph type="secondary">
            将治理数据快照、可复现实验参数、MLflow 运行、Gold/Smoke 评测集和安全硬门禁放在同一条审计链上。
          </Typography.Paragraph>
        </div>
        <Alert
          showIcon
          type="info"
          message="训练与评测职责分离"
          description="模型工程师负责创建和运行实验；模型评测员使用冻结评测集执行正式门禁。Smoke 只验证链路，只有不少于 500 条样本的 Gold 评测可产生候选。"
        />
        {commandError ? <ErrorState error={commandError} onRetry={() => setCommandError(undefined)} /> : null}
        {operationReceipt ? (
          <OperationReceiptAlert
            receipt={operationReceipt}
            onClose={() => setOperationReceipt(undefined)}
          />
        ) : null}
        {createTargetError ? (
          <Alert
            closable
            showIcon
            type="warning"
            message="无法从数据快照登记训练实验"
            description={createTargetError}
            onClose={() => setCreateTargetError(undefined)}
          />
        ) : null}
        <Row gutter={[16, 16]}>
          <Col xs={12} lg={6}><Card><Statistic title="本页运行中" value={totals.running} /></Card></Col>
          <Col xs={12} lg={6}><Card><Statistic title="本页已完成实验" value={totals.completed} /></Card></Col>
          <Col xs={12} lg={6}><Card><Statistic title="本页候选" value={totals.candidate} /></Card></Col>
          <Col xs={12} lg={6}><Card><Statistic title="本页硬门禁拒绝" value={totals.rejected} /></Card></Col>
        </Row>
        <Space wrap>
          {experimentActions.includes("CREATE_EXPERIMENT") ? (
            <Button type="primary" onClick={() => {
              setCreateTargetError(undefined);
              setCreateOpen(true);
            }}>登记可复现实验</Button>
          ) : null}
          {evaluationJobActions.includes("CREATE_EVALUATION_JOB") ? (
            <Button
              loading={busy === "load-evaluation-experiments"}
              onClick={() => void openEvaluationJob()}
            >创建独立评测任务</Button>
          ) : null}
          {evaluationSuiteActions.includes("REGISTER_EVALUATION_SUITE") &&
          evaluationPolicyActions.includes("REGISTER_EVALUATION_POLICY") ? (
            <Button onClick={() => setEvaluationAssetsOpen(true)}>
              注册时序/RUL评测集与门禁
            </Button>
          ) : null}
        </Space>
        <Tabs
          items={[
            {
              key: "experiments",
              label: `训练实验 (${experimentTotal})`,
              children: (
                <div className="page-stack">
                  <Card size="small">
                    <Space wrap>
                      <Typography.Text>实验 ID</Typography.Text>
                      <Input.Search
                        allowClear
                        aria-label="按完整实验 ID 查询训练实验"
                        enterButton="查询"
                        maxLength={128}
                        placeholder="输入完整实验 ID"
                        style={{ width: 300 }}
                        value={experimentIdDraft}
                        onChange={(event) => {
                          const value = event.target.value;
                          setExperimentIdDraft(value);
                          if (!value) {
                            setExperimentPage(1);
                            setExperimentIdFilter(undefined);
                          }
                        }}
                        onSearch={(value) => {
                          setExperimentPage(1);
                          setExperimentIdFilter(normalizeStableId(value));
                        }}
                      />
                      <Typography.Text>数据快照</Typography.Text>
                      <Input.Search
                        allowClear
                        aria-label="按完整数据快照 ID 查询训练实验"
                        enterButton="查询"
                        maxLength={128}
                        placeholder="输入完整 Snapshot ID"
                        style={{ width: 320 }}
                        value={experimentSnapshotDraft}
                        onChange={(event) => {
                          const value = event.target.value;
                          setExperimentSnapshotDraft(value);
                          if (!value) {
                            setExperimentPage(1);
                            setExperimentSnapshotFilter(undefined);
                          }
                        }}
                        onSearch={(value) => {
                          setExperimentPage(1);
                          setExperimentSnapshotFilter(normalizeStableId(value));
                        }}
                      />
                      <Typography.Text>训练方法</Typography.Text>
                      <Select<TrainingExperimentMethod>
                        allowClear
                        value={experimentMethod}
                        placeholder="全部方法"
                        style={{ minWidth: 150 }}
                        options={TRAINING_METHODS.map((value) => ({ value }))}
                        onChange={(value) => {
                          setExperimentPage(1);
                          setExperimentMethod(value);
                        }}
                      />
                      <Typography.Text>状态</Typography.Text>
                      <Select<TrainingExperimentStatus>
                        allowClear
                        value={experimentStatus}
                        placeholder="全部状态"
                        style={{ minWidth: 190 }}
                        options={EXPERIMENT_STATUSES.map((value) => ({ value }))}
                        onChange={(value) => {
                          setExperimentPage(1);
                          setExperimentStatus(value);
                        }}
                      />
                      <Button onClick={() => void loadExperiments()}>刷新实验目录</Button>
                    </Space>
                  </Card>
                  {experimentError ? (
                    <ErrorState error={experimentError} onRetry={() => void loadExperiments()} />
                  ) : !experiments ? <LoadingState /> : experiments.length === 0 ? (
                    <EmptyState description="当前筛选条件下没有训练实验" />
                  ) : (
                    <>
                      <List
                        grid={{ gutter: 16, xs: 1, lg: 2 }}
                        dataSource={experiments}
                        renderItem={(experiment) => (
                          <List.Item>
                            <Card
                              title={<Space wrap><Tag color={methodColor(experiment.method)}>{experiment.method}</Tag><Typography.Text code>{experiment.experiment_id}</Typography.Text></Space>}
                              extra={<Tag color={statusColor(experiment.status)}>{experiment.status}</Tag>}
                              actions={[
                                <Button
                                  key="detail"
                                  type="link"
                                  loading={busy === experiment.experiment_id}
                                  onClick={() => void openExperiment(experiment.experiment_id)}
                                >可复现详情</Button>,
                                experiment.legal_actions?.includes("START_EXPERIMENT") ? (
                                  <Button key="start" type="link" loading={busy === experiment.experiment_id} onClick={() => void start(experiment)}>创建 MLflow 运行</Button>
                                ) : null,
                              ].filter(Boolean)}
                            >
                              <Descriptions size="small" column={1}>
                                <Descriptions.Item label="对照组">{experiment.comparison_group_id}</Descriptions.Item>
                                <Descriptions.Item label="数据快照"><Typography.Text code>{experiment.dataset_snapshot_id}</Typography.Text></Descriptions.Item>
                                <Descriptions.Item label="模型">{experiment.base_model_id}</Descriptions.Item>
                                <Descriptions.Item label="MLflow">{experiment.mlflow_run_id ?? "尚未关联"}</Descriptions.Item>
                              </Descriptions>
                            </Card>
                          </List.Item>
                        )}
                      />
                      <Pagination
                        current={experimentPage}
                        pageSize={EXPERIMENT_PAGE_SIZE}
                        total={experimentTotal}
                        showSizeChanger={false}
                        showTotal={(total) => `共 ${total} 个训练实验`}
                        onChange={setExperimentPage}
                      />
                    </>
                  )}
                </div>
              ),
            },
            {
              key: "evaluation-jobs",
              label: `独立评测任务 (${evaluationJobTotal})`,
              children: (
                <div className="page-stack">
                  <Card size="small">
                    <Space wrap>
                      <Typography.Text>任务 ID</Typography.Text>
                      <Input.Search
                        allowClear
                        aria-label="按完整评测任务 ID 查询"
                        enterButton="查询"
                        maxLength={128}
                        placeholder="输入完整 Job ID"
                        style={{ width: 290 }}
                        value={evaluationJobIdDraft}
                        onChange={(event) => {
                          const value = event.target.value;
                          setEvaluationJobIdDraft(value);
                          if (!value) {
                            setEvaluationJobPage(1);
                            setEvaluationJobIdFilter(undefined);
                          }
                        }}
                        onSearch={(value) => {
                          setEvaluationJobPage(1);
                          setEvaluationJobIdFilter(normalizeStableId(value));
                        }}
                      />
                      <Typography.Text>候选实验</Typography.Text>
                      <Input.Search
                        allowClear
                        aria-label="按完整候选实验 ID 查询评测任务"
                        enterButton="查询"
                        maxLength={128}
                        placeholder="输入完整候选实验 ID"
                        style={{ width: 320 }}
                        value={candidateExperimentDraft}
                        onChange={(event) => {
                          const value = event.target.value;
                          setCandidateExperimentDraft(value);
                          if (!value) {
                            setEvaluationJobPage(1);
                            setCandidateExperimentFilter(undefined);
                          }
                        }}
                        onSearch={(value) => {
                          setEvaluationJobPage(1);
                          setCandidateExperimentFilter(normalizeStableId(value));
                        }}
                      />
                      <Typography.Text>基线实验</Typography.Text>
                      <Input.Search
                        allowClear
                        aria-label="按完整基线实验 ID 查询评测任务"
                        enterButton="查询"
                        maxLength={128}
                        placeholder="输入完整基线实验 ID"
                        style={{ width: 320 }}
                        value={baselineExperimentDraft}
                        onChange={(event) => {
                          const value = event.target.value;
                          setBaselineExperimentDraft(value);
                          if (!value) {
                            setEvaluationJobPage(1);
                            setBaselineExperimentFilter(undefined);
                          }
                        }}
                        onSearch={(value) => {
                          setEvaluationJobPage(1);
                          setBaselineExperimentFilter(normalizeStableId(value));
                        }}
                      />
                      <Typography.Text>状态</Typography.Text>
                      <Select<EvaluationJobStatus>
                        allowClear
                        value={evaluationJobStatus}
                        placeholder="全部状态"
                        style={{ minWidth: 150 }}
                        options={EVALUATION_JOB_STATUSES.map((value) => ({ value }))}
                        onChange={(value) => {
                          setEvaluationJobPage(1);
                          setEvaluationJobStatus(value);
                        }}
                      />
                      <Button onClick={() => void loadEvaluationJobs()}>刷新任务目录</Button>
                    </Space>
                  </Card>
                  {evaluationJobError ? (
                    <ErrorState
                      error={evaluationJobError}
                      onRetry={() => void loadEvaluationJobs()}
                    />
                  ) : !evaluationJobs ? <LoadingState /> : evaluationJobs.length === 0 ? (
                    <EmptyState description="当前筛选条件下没有独立评测任务" />
                  ) : (
                    <>
                      <Table
                        rowKey="job_id"
                        dataSource={evaluationJobs}
                        pagination={false}
                        scroll={{ x: 1100 }}
                        columns={[
                          { title: "状态", dataIndex: "status", render: (value: string) => <Tag color={statusColor(value)}>{value}</Tag> },
                          { title: "候选", dataIndex: "candidate_experiment_id", ellipsis: true },
                          { title: "基线", dataIndex: "baseline_experiment_id", ellipsis: true },
                          { title: "评测集", dataIndex: "suite_id", ellipsis: true },
                          { title: "Profile", dataIndex: "target_profile" },
                          {
                            title: "结果",
                            dataIndex: "result_evaluation_id",
                            render: (value?: string) => value ? (
                              <Button
                                type="link"
                                loading={busy === `evaluation-${value}`}
                                onClick={() => void openEvaluation(value)}
                              >查看评测结果</Button>
                            ) : "等待 Worker",
                          },
                          { title: "失败原因", dataIndex: "failure_reason", render: (value?: string) => value ?? "—" },
                        ]}
                      />
                      <Pagination
                        current={evaluationJobPage}
                        pageSize={EVALUATION_JOB_PAGE_SIZE}
                        total={evaluationJobTotal}
                        showSizeChanger={false}
                        showTotal={(total) => `共 ${total} 个评测任务`}
                        onChange={setEvaluationJobPage}
                      />
                    </>
                  )}
                </div>
              ),
            },
            {
              key: "evaluations",
              label: `候选评测 (${evaluationTotal})`,
              children: (
                <div className="page-stack">
                  <Card size="small">
                    <Space wrap>
                      <Typography.Text>评测 ID</Typography.Text>
                      <Input.Search
                        allowClear
                        aria-label="按完整评测 ID 查询结果"
                        enterButton="查询"
                        maxLength={128}
                        placeholder="输入完整 Evaluation ID"
                        style={{ width: 310 }}
                        value={evaluationIdDraft}
                        onChange={(event) => {
                          const value = event.target.value;
                          setEvaluationIdDraft(value);
                          if (!value) {
                            setEvaluationPage(1);
                            setEvaluationIdFilter(undefined);
                          }
                        }}
                        onSearch={(value) => {
                          setEvaluationPage(1);
                          setEvaluationIdFilter(normalizeStableId(value));
                        }}
                      />
                      <Typography.Text>候选实验</Typography.Text>
                      <Input.Search
                        allowClear
                        aria-label="按完整候选实验 ID 查询评测结果"
                        enterButton="查询"
                        maxLength={128}
                        placeholder="输入完整候选实验 ID"
                        style={{ width: 320 }}
                        value={evaluationCandidateDraft}
                        onChange={(event) => {
                          const value = event.target.value;
                          setEvaluationCandidateDraft(value);
                          if (!value) {
                            setEvaluationPage(1);
                            setEvaluationCandidateFilter(undefined);
                          }
                        }}
                        onSearch={(value) => {
                          setEvaluationPage(1);
                          setEvaluationCandidateFilter(normalizeStableId(value));
                        }}
                      />
                      <Typography.Text>基线实验</Typography.Text>
                      <Input.Search
                        allowClear
                        aria-label="按完整基线实验 ID 查询评测结果"
                        enterButton="查询"
                        maxLength={128}
                        placeholder="输入完整基线实验 ID"
                        style={{ width: 320 }}
                        value={evaluationBaselineDraft}
                        onChange={(event) => {
                          const value = event.target.value;
                          setEvaluationBaselineDraft(value);
                          if (!value) {
                            setEvaluationPage(1);
                            setEvaluationBaselineFilter(undefined);
                          }
                        }}
                        onSearch={(value) => {
                          setEvaluationPage(1);
                          setEvaluationBaselineFilter(normalizeStableId(value));
                        }}
                      />
                      <Typography.Text>决策</Typography.Text>
                      <Select<ModelEvaluationDecision>
                        allowClear
                        value={evaluationDecision}
                        placeholder="全部决策"
                        style={{ minWidth: 170 }}
                        options={MODEL_EVALUATION_DECISIONS.map((value) => ({ value }))}
                        onChange={(value) => {
                          setEvaluationPage(1);
                          setEvaluationDecision(value);
                        }}
                      />
                      <Button onClick={() => void loadEvaluations()}>刷新评测目录</Button>
                    </Space>
                  </Card>
                  {evaluationError ? (
                    <ErrorState error={evaluationError} onRetry={() => void loadEvaluations()} />
                  ) : !evaluations ? <LoadingState /> : evaluations.length === 0 ? (
                    <EmptyState description="当前筛选条件下没有独立评测结果" />
                  ) : (
                    <>
                      <Table
                        rowKey="evaluation_id"
                        dataSource={evaluations}
                        pagination={false}
                        scroll={{ x: 980 }}
                        columns={[
                          { title: "决策", dataIndex: "decision", render: (value: string) => <Tag color={decisionColor(value)}>{value}</Tag> },
                          { title: "候选", dataIndex: "candidate_experiment_id", ellipsis: true },
                          { title: "质量变化", dataIndex: "quality_delta", render: (value: number) => percentage(value) },
                          { title: "95% CI", render: (_, row) => `${percentage(row.ci_low)} ～ ${percentage(row.ci_high)}` },
                          { title: "时延改善", dataIndex: "latency_improvement", render: (value: number) => percentage(value) },
                          { title: "成本改善", dataIndex: "cost_improvement", render: (value: number) => percentage(value) },
                          { title: "硬门禁", render: (_, row) => `${Object.values(row.hard_gate_results).filter(Boolean).length}/${Object.keys(row.hard_gate_results).length}` },
                          {
                            title: "操作",
                            render: (_, row) => (
                              <Space wrap>
                                <Button
                                  type="link"
                                  loading={busy === `evaluation-${row.evaluation_id}`}
                                  onClick={() => void openEvaluation(row.evaluation_id)}
                                >查看详情</Button>
                                <Button
                                  type="link"
                                  loading={busy === `evidence-${row.evaluation_id}`}
                                  onClick={() => void saveEvaluationEvidence(row)}
                                >下载逐样本报告</Button>
                              </Space>
                            ),
                          },
                        ]}
                      />
                      <Pagination
                        current={evaluationPage}
                        pageSize={MODEL_EVALUATION_PAGE_SIZE}
                        total={evaluationTotal}
                        showSizeChanger={false}
                        showTotal={(total) => `共 ${total} 个评测结果`}
                        onChange={setEvaluationPage}
                      />
                    </>
                  )}
                </div>
              ),
            },
            {
              key: "governance",
              label: "评测集与策略",
              children: governanceError ? (
                <ErrorState error={governanceError} onRetry={() => void loadSupportingData()} />
              ) : !suites || !policies ? <LoadingState /> : (
                <Row gutter={[16, 16]}>
                  <Col xs={24} lg={12}>
                    <Card title="冻结评测集">
                      <List
                        dataSource={suites}
                        locale={{ emptyText: "尚无评测集" }}
                        renderItem={(suite) => (
                          <List.Item>
                            <List.Item.Meta
                              title={<Space><Tag color={suite.tier === "GOLD" ? "gold" : "blue"}>{suite.tier}</Tag>{suite.name} · {suite.version}</Space>}
                              description={`${suite.sample_count} 条 · ${suite.status} · ${suite.source_snapshot_id}`}
                            />
                          </List.Item>
                        )}
                      />
                    </Card>
                  </Col>
                  <Col xs={24} lg={12}>
                    <Card title="不可变门禁策略">
                      <List
                        dataSource={policies}
                        locale={{ emptyText: "尚无评测策略" }}
                        renderItem={(policy) => (
                          <List.Item>
                            <List.Item.Meta
                              title={`${policy.name} · ${policy.version}`}
                              description={`${policy.primary_metric} · ${Object.keys(policy.hard_gates).length} 项硬门禁 · ${policy.policy_hash.slice(0, 16)}`}
                            />
                          </List.Item>
                        )}
                      />
                    </Card>
                  </Col>
                </Row>
              ),
            },
          ]}
        />
      </div>

      <Modal
        title="注册序列模型评测集与门禁"
        open={evaluationAssetsOpen}
        onCancel={() => setEvaluationAssetsOpen(false)}
        onOk={() => evaluationAssetsForm.submit()}
        confirmLoading={busy === "create-sequence-evaluation-assets"}
        width={680}
      >
        <Alert
          type="info"
          showIcon
          message="评测快照与训练快照严格隔离"
          description={selectedEvaluationAssetKind === "RUL"
            ? "所选 RUL 快照会冻结为 EVALUATION_ONLY；全部目标必须来自未参与训练的真实故障 Outcome，且设备不能跨分区。Smoke 需 14～50 条，Gold 至少 500 条。"
            : "所选遥测快照会冻结为 EVALUATION_ONLY；独立 Worker 要求全部窗口具备 0/1 结果标签，且正常与异常两类都存在。Smoke 需 14～50 条，Gold 至少 500 条。"}
          style={{ marginBottom: 16 }}
        />
        <Form
          form={evaluationAssetsForm}
          layout="vertical"
          onFinish={(values) => void registerSequenceEvaluationAssets(values)}
          initialValues={{
            evaluation_kind: "TIMESERIES",
            tier: "GOLD",
            suite_name: "industrial-telemetry-golden-set",
            suite_version: "v1",
            policy_name: "industrial-timeseries-release-gate",
            policy_version: "v1",
            false_positive_rate_max: 0.1,
            miss_rate_max: 0.1,
            median_absolute_error_minutes_max: 240,
            interval_coverage_min: 0.8,
            interval_coverage_max: 0.95,
            pinball_loss_max: 120,
          }}
        >
          <Row gutter={16}>
            <Col span={24}>
              <Form.Item name="evaluation_kind" label="评测类型" rules={[{ required: true }]}>
                <Select
                  onChange={() => evaluationAssetsForm.setFieldValue("source_snapshot_id", undefined)}
                  options={[
                    { value: "TIMESERIES", label: "异常检测时序模型" },
                    { value: "RUL", label: "剩余寿命分位数模型" },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col span={24}>
              <Form.Item name="source_snapshot_id" label="冻结序列快照" rules={[{ required: true }]}>
                <Select options={snapshots.filter((item) => item.contract_version === (selectedEvaluationAssetKind === "RUL" ? "industrial-rul-sequence-v1" : "industrial-telemetry-sequence-v1")).map((item) => ({ value: item.snapshot_id, label: `${item.snapshot_id} · ${item.row_count} 条` }))} />
              </Form.Item>
            </Col>
            <Col span={12}><Form.Item name="suite_name" label="评测集名称" rules={[{ required: true }]}><Input /></Form.Item></Col>
            <Col span={6}><Form.Item name="suite_version" label="评测集版本" rules={[{ required: true }]}><Input /></Form.Item></Col>
            <Col span={6}><Form.Item name="tier" label="层级" rules={[{ required: true }]}><Select options={[{ value: "SMOKE" }, { value: "GOLD" }]} /></Form.Item></Col>
            <Col span={12}><Form.Item name="policy_name" label="门禁策略名称" rules={[{ required: true }]}><Input /></Form.Item></Col>
            <Col span={12}><Form.Item name="policy_version" label="策略版本" rules={[{ required: true }]}><Input /></Form.Item></Col>
            {selectedEvaluationAssetKind === "RUL" ? (
              <>
                <Col span={12}><Form.Item name="median_absolute_error_minutes_max" label="中位绝对误差上限（分钟）" rules={[{ required: true }]}><InputNumber min={0.000001} max={1440} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="pinball_loss_max" label="Pinball Loss 上限（分钟）" rules={[{ required: true }]}><InputNumber min={0.000001} max={1440} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="interval_coverage_min" label="P10-P90 覆盖率下限" rules={[{ required: true }]}><InputNumber min={0.7} max={0.9} step={0.01} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="interval_coverage_max" label="P10-P90 覆盖率上限" rules={[{ required: true }]}><InputNumber min={0.9} max={1} step={0.01} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            ) : (
              <>
                <Col span={12}><Form.Item name="false_positive_rate_max" label="最大误报率" rules={[{ required: true }]}><InputNumber min={0} max={0.2} step={0.01} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="miss_rate_max" label="最大漏报率" rules={[{ required: true }]}><InputNumber min={0} max={0.2} step={0.01} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            )}
          </Row>
        </Form>
      </Modal>

      <Modal
        title="登记可复现实验"
        open={createOpen}
        onCancel={() => setCreateOpen(false)}
        onOk={() => form.submit()}
        confirmLoading={busy === "create"}
        width={760}
      >
        <Alert
          type="warning"
          showIcon
          message="同一对照组会锁定模型、数据、模板、随机种子和预算；后训练方法彼此独立"
          description="DPO/GRPO/PPO、检索、VLM、ASR、TTS、异常时序与 RUL Transformer 各自使用独立的已治理数据契约；Worker 会重新校验清单和内容哈希。"
          style={{ marginBottom: 16 }}
        />
        <Form form={form} layout="vertical" onFinish={(values) => void create(values)} initialValues={{ method: "LORA", training_start_mode: "fresh", distributed_strategy: "single_gpu", distributed_world_size: 2, deepspeed_zero_stage: 2, offload_optimizer_device: "none", offload_param_device: "none", fsdp_sharding_strategy: "FULL_SHARD", task_type: "industrial_root_cause", max_steps: 1000, effective_batch_size: 32, evaluation_interval: 100, random_seed: 42, mlflow_experiment_name: "industrial-ops-m4", beta: 0.1, loss_type: "sigmoid", group_size: 8, max_completion_length: 256, kl_coefficient: 0.05, total_episodes: 1024, response_length: 256, num_ppo_epochs: 4, num_mini_batches: 1, hard_negatives_per_query: 1, similarity_scale: 20, positive_weight: 1, max_image_pixels: 4194304, sampling_rate: 16000, language: "zh", max_audio_seconds: 30, voice_profile_id: "industrial-standard-zh-v1", speaker_embedding_dimension: 512, d_model: 128, nhead: 4, num_layers: 3, dim_feedforward: 256, mask_probability: 0.15, rule_threshold: 0.65, empirical_p10_minutes: 60, empirical_p50_minutes: 240, empirical_p90_minutes: 720, source_sample_count: 30, quantization_profile: "AWQ", target_hardware_profile: "nvidia-sm80-or-newer", tool_version: "llmcompressor==0.12.0.1", calibration_sample_count: 512, quantization_max_sequence_length: 2048, quantization_gpu_hourly_cost_usd: 1, incompatibilities: "GGUF 制品不能由 vLLM 加载" }}>
          <Row gutter={16}>
            <Col span={12}><Form.Item name="comparison_group_id" label="对照组 ID" rules={[{ required: true }]}><Input /></Form.Item></Col>
            <Col span={12}><Form.Item name="method" label="训练方法" rules={[{ required: true }]}><Select loading={["load-quantization-sources", "load-resume-sources"].includes(busy ?? "")} onChange={(value) => void selectTrainingMethod(value)} options={TRAINING_METHODS.map((value) => ({ value }))} /></Form.Item></Col>
            <Col span={12}><Form.Item name="task_type" label="任务类型" rules={[{ required: true }]}><Input /></Form.Item></Col>
            <Col span={12}><Form.Item name="dataset_snapshot_id" label="治理数据快照" rules={[{ required: true }]}><Select options={snapshots.filter((item) => ["RUL_TRANSFORMER", "RUL_EMPIRICAL_BASELINE"].includes(selectedMethod ?? "") ? item.contract_version === "industrial-rul-sequence-v1" : ["TIMESERIES_TRANSFORMER", "TIMESERIES_RULE_BASELINE"].includes(selectedMethod ?? "") ? item.contract_version === "industrial-telemetry-sequence-v1" : !["industrial-telemetry-sequence-v1", "industrial-rul-sequence-v1"].includes(item.contract_version)).map((item) => ({ value: item.snapshot_id, label: `${item.snapshot_id} (${item.row_count})` }))} /></Form.Item></Col>
            <Col span={12}><Form.Item name="base_model_id" label="基础模型" rules={[{ required: true }]}><Input placeholder="Qwen/Qwen3-8B" /></Form.Item></Col>
            <Col span={12}><Form.Item name="base_model_revision" label="模型不可变 Revision" rules={[{ required: true }]}><Input placeholder="Hugging Face commit SHA" /></Form.Item></Col>
            <Col span={12}><Form.Item name="base_model_digest" label="模型摘要" rules={[{ required: true }]}><Input placeholder="hf-revision:&lt;commit SHA&gt;" /></Form.Item></Col>
            {["TIMESERIES_TRANSFORMER", "TIMESERIES_RULE_BASELINE", "RUL_TRANSFORMER", "RUL_EMPIRICAL_BASELINE"].includes(selectedMethod ?? "") ? (
              <Col span={12}><Form.Item label="Tokenizer"><Input value="not-applicable（时序数值模型）" disabled /></Form.Item></Col>
            ) : (
              <Col span={12}><Form.Item name="tokenizer_digest" label="Tokenizer 摘要" rules={[{ required: true }]}><Input placeholder="sha256:..." /></Form.Item></Col>
            )}
            {["EMBEDDING", "RERANKER", "ASR", "TIMESERIES_TRANSFORMER", "TIMESERIES_RULE_BASELINE", "RUL_TRANSFORMER", "RUL_EMPIRICAL_BASELINE"].includes(selectedMethod ?? "") ? (
              <Col span={12}><Form.Item label="对话模板"><Input value="not-applicable（编码器不使用对话模板）" disabled /></Form.Item></Col>
            ) : (
              <Col span={12}><Form.Item name="chat_template_digest" label="对话模板摘要" rules={[{ required: true }]}><Input placeholder="sha256:..." /></Form.Item></Col>
            )}
            <Col span={12}><Form.Item name="git_commit" label="Git Commit" rules={[{ required: true }]}><Input /></Form.Item></Col>
            <Col span={12}><Form.Item name="container_digest" label="训练镜像摘要" rules={[{ required: true }]}><Input placeholder="sha256:..." /></Form.Item></Col>
            <Col span={8}><Form.Item name="max_steps" label="训练步数" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
            <Col span={8}><Form.Item name="effective_batch_size" label="有效 Batch" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
            <Col span={8}><Form.Item name="evaluation_interval" label="评测间隔" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
            {["LORA", "QLORA"].includes(selectedMethod ?? "") ? (
              <>
                <Col span={8}>
                  <Form.Item name="distributed_strategy" label="训练并行 Profile" rules={[{ required: true }]}>
                    <Select options={[
                      { value: "single_gpu", label: "单 GPU" },
                      { value: "deepspeed", label: "DeepSpeed ZeRO" },
                      { value: "fsdp", label: "PyTorch FSDP", disabled: selectedMethod === "QLORA" },
                    ]} />
                  </Form.Item>
                </Col>
                {selectedDistributedStrategy !== "single_gpu" ? (
                  <Col span={8}>
                    <Form.Item name="distributed_world_size" label="单机 GPU 数" rules={[{ required: true }]}>
                      <InputNumber min={2} max={8} style={{ width: "100%" }} />
                    </Form.Item>
                  </Col>
                ) : null}
                {selectedDistributedStrategy === "deepspeed" ? (
                  <>
                    <Col span={8}><Form.Item name="deepspeed_zero_stage" label="ZeRO Stage" rules={[{ required: true }]}><Select options={[{ value: 2 }, { value: 3, disabled: selectedMethod === "QLORA" }]} /></Form.Item></Col>
                    <Col span={8}><Form.Item name="offload_optimizer_device" label="优化器 Offload"><Select options={[{ value: "none" }, { value: "cpu" }]} /></Form.Item></Col>
                    <Col span={8}><Form.Item name="offload_param_device" label="参数 Offload"><Select disabled={selectedDeepSpeedZeroStage !== 3} options={[{ value: "none" }, { value: "cpu" }]} /></Form.Item></Col>
                  </>
                ) : null}
                {selectedDistributedStrategy === "fsdp" ? (
                  <>
                    <Col span={8}><Form.Item name="fsdp_sharding_strategy" label="FSDP 分片"><Select options={[{ value: "FULL_SHARD" }, { value: "SHARD_GRAD_OP" }]} /></Form.Item></Col>
                    <Col span={16}><Form.Item name="fsdp_transformer_layer_cls_to_wrap" label="Transformer 层类名" rules={[{ required: true }]}><Input placeholder="Qwen3DecoderLayer（多个以逗号分隔）" /></Form.Item></Col>
                  </>
                ) : null}
                <Col span={8}>
                  <Form.Item name="training_start_mode" label="启动方式" rules={[{ required: true }]}>
                    <Select options={[
                      { value: "fresh", label: "从基础模型开始" },
                      { value: "resume", label: "从登记 Checkpoint 续训" },
                    ]} />
                  </Form.Item>
                </Col>
                {selectedTrainingStartMode === "resume" ? (
                  <>
                    <Col span={16}>
                      <Form.Item name="resume_source_experiment_id" label="已完成的同方法源实验" rules={[{ required: true }]}>
                        <Select
                          loading={busy === "load-resume-source"}
                          onOpenChange={(open) => {
                            if (open) void loadResumeSources(selectedMethod);
                          }}
                          onChange={(value) => void selectResumeSource(value)}
                          options={experimentOptions.map((item) => ({
                            value: item.experiment_id,
                            label: `${item.method} · ${item.experiment_id}`,
                          }))}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}><Form.Item name="resume_artifact_id" label="Checkpoint 制品 ID" rules={[{ required: true }]}><Input readOnly /></Form.Item></Col>
                    <Col span={12}><Form.Item name="resume_artifact_hash" label="Checkpoint 制品 SHA-256" rules={[{ required: true }]}><Input readOnly /></Form.Item></Col>
                    <Col span={24}><Form.Item name="resume_checkpoint_path" label="制品内 Checkpoint 目录" rules={[{ required: true, pattern: /^checkpoint-[1-9][0-9]*$/ }]}><Input placeholder="checkpoint-500" /></Form.Item></Col>
                  </>
                ) : null}
              </>
            ) : null}
            <Col span={12}><Form.Item name="random_seed" label="随机种子" rules={[{ required: true }]}><InputNumber min={0} style={{ width: "100%" }} /></Form.Item></Col>
            <Col span={12}><Form.Item name="mlflow_experiment_name" label="MLflow Experiment" rules={[{ required: true }]}><Input /></Form.Item></Col>
            {selectedMethod === "DPO" ? (
              <>
                <Col span={12}><Form.Item name="beta" label="DPO Beta" rules={[{ required: true }]}><InputNumber min={0.000001} step={0.01} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="loss_type" label="DPO Loss" rules={[{ required: true }]}><Select options={["sigmoid", "hinge", "ipo"].map((value) => ({ value }))} /></Form.Item></Col>
              </>
            ) : null}
            {selectedMethod === "GRPO" ? (
              <>
                <Col span={12}><Form.Item label="当前奖励契约"><Input value={CURRENT_REWARD_CONTRACT_VERSION} disabled /></Form.Item></Col>
                <Col span={12}><Form.Item label="奖励契约摘要"><Input value={CURRENT_REWARD_CONTRACT_DIGEST} disabled /></Form.Item></Col>
                <Col span={12}><Form.Item name="group_size" label="每组生成数" rules={[{ required: true }]}><InputNumber min={2} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="max_completion_length" label="最大生成长度" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            ) : null}
            {selectedMethod === "PPO" ? (
              <>
                <Col span={12}><Form.Item name="reward_model_id" label="奖励模型" rules={[{ required: true }]}><Input placeholder="organization/reward-model" /></Form.Item></Col>
                <Col span={12}><Form.Item name="reward_model_revision" label="奖励模型 Revision" rules={[{ required: true }]}><Input placeholder="40 位 commit SHA" /></Form.Item></Col>
                <Col span={24}><Form.Item name="reward_model_digest" label="奖励模型摘要" rules={[{ required: true }]}><Input placeholder="hf-revision:&lt;commit SHA&gt;" /></Form.Item></Col>
                <Col span={12}><Form.Item name="value_model_id" label="价值模型" rules={[{ required: true }]}><Input placeholder="organization/value-model" /></Form.Item></Col>
                <Col span={12}><Form.Item name="value_model_revision" label="价值模型 Revision" rules={[{ required: true }]}><Input placeholder="40 位 commit SHA" /></Form.Item></Col>
                <Col span={24}><Form.Item name="value_model_digest" label="价值模型摘要" rules={[{ required: true }]}><Input placeholder="hf-revision:&lt;commit SHA&gt;" /></Form.Item></Col>
                <Col span={8}><Form.Item name="kl_coefficient" label="固定 KL 系数" rules={[{ required: true }]}><InputNumber min={0.000001} step={0.01} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="total_episodes" label="Episodes" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="response_length" label="响应长度" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="num_ppo_epochs" label="PPO Epochs" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="num_mini_batches" label="Mini Batches" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            ) : null}
            {["EMBEDDING", "RERANKER"].includes(selectedMethod ?? "") ? (
              <>
                <Col span={8}><Form.Item label="数据契约"><Input value="industrial-retrieval-triplet-v1" disabled /></Form.Item></Col>
                <Col span={8}><Form.Item name="hard_negatives_per_query" label="每 Query 难负例数" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
                {selectedMethod === "EMBEDDING" ? (
                  <Col span={8}><Form.Item name="similarity_scale" label="相似度 Scale" rules={[{ required: true }]}><InputNumber min={0.000001} step={1} style={{ width: "100%" }} /></Form.Item></Col>
                ) : (
                  <Col span={8}><Form.Item name="positive_weight" label="正样本权重" rules={[{ required: true }]}><InputNumber min={0.000001} step={0.1} style={{ width: "100%" }} /></Form.Item></Col>
                )}
              </>
            ) : null}
            {selectedMethod === "VLM" ? (
              <>
                <Col span={12}><Form.Item label="视觉数据契约"><Input value="industrial-vision-instruction-v1" disabled /></Form.Item></Col>
                <Col span={12}><Form.Item name="max_image_pixels" label="单图最大像素" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            ) : null}
            {selectedMethod === "ASR" ? (
              <>
                <Col span={12}><Form.Item label="音频数据契约"><Input value="industrial-asr-transcript-v1" disabled /></Form.Item></Col>
                <Col span={12}><Form.Item name="language" label="Whisper 语言" rules={[{ required: true }]}><Input placeholder="zh" /></Form.Item></Col>
                <Col span={12}><Form.Item name="sampling_rate" label="采样率" rules={[{ required: true }]}><InputNumber min={16000} max={16000} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="max_audio_seconds" label="单段最长秒数" rules={[{ required: true }]}><InputNumber min={0.1} max={30} step={0.5} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            ) : null}
            {selectedMethod === TTS_TRAINING_METHOD ? (
              <>
                <Col span={12}><Form.Item label="语音合成数据契约"><Input value="industrial-tts-speech-v1" disabled /></Form.Item></Col>
                <Col span={12}><Form.Item name="voice_profile_id" label="授权标准声音 Profile" rules={[{ required: true }]}><Input /></Form.Item></Col>
                <Col span={12}><Form.Item name="speaker_embedding_dimension" label="说话人向量维度" rules={[{ required: true }]}><InputNumber min={2} max={4096} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="language" label="合成语言" rules={[{ required: true }]}><Input /></Form.Item></Col>
                <Col span={12}><Form.Item name="sampling_rate" label="采样率" rules={[{ required: true }]}><InputNumber min={16000} max={16000} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="max_audio_seconds" label="单段最长秒数" rules={[{ required: true }]}><InputNumber min={0.1} max={30} step={0.5} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            ) : null}
            {selectedMethod === "QUANTIZATION" ? (
              <>
                <Col span={24}><Alert type="info" showIcon message="量化制品是新的候选模型" description="必须绑定一个已完成的 LoRA/QLoRA/DPO/GRPO 制品和同一治理快照；量化后不会继承源模型审批，需重新执行质量、安全、长上下文、工具调用、时延与成本评测。" /></Col>
                <Col span={24}>
                  <Form.Item name="source_experiment_id" label="源微调实验" rules={[{ required: true }]}>
                    <Select
                      loading={busy === "load-quantization-source"}
                      onChange={(value) => void selectQuantizationSource(value)}
                      options={experimentOptions.map((item) => ({
                        value: item.experiment_id,
                        label: `${item.method} · ${item.experiment_id}`,
                      }))}
                    />
                  </Form.Item>
                </Col>
                <Col span={12}><Form.Item name="source_artifact_id" label="源 Adapter 制品 ID" rules={[{ required: true }]}><Input readOnly /></Form.Item></Col>
                <Col span={12}><Form.Item name="source_artifact_hash" label="源 Adapter SHA-256" rules={[{ required: true }]}><Input readOnly /></Form.Item></Col>
                <Col span={12}>
                  <Form.Item name="quantization_profile" label="量化算法与精度" rules={[{ required: true }]}>
                    <Select
                      onChange={(value: ExperimentFormValues["quantization_profile"]) => {
                        const profile = quantizationProfile(value);
                        form.setFieldsValue({
                          tool_version: profile.algorithm === "GGUF"
                            ? "llama.cpp@7ba604f1cb61cd14898138e9abc0b4ff2601f180"
                            : "llmcompressor==0.12.0.1",
                          target_hardware_profile: profile.algorithm === "GGUF"
                            ? "x86_64-avx2-edge"
                            : "nvidia-sm80-or-newer",
                        });
                      }}
                      options={Object.entries(QUANTIZATION_PROFILES).map(([value, profile]) => ({
                        value,
                        label: `${profile.algorithm} · ${profile.scheme} · ${profile.runtime}`,
                      }))}
                    />
                  </Form.Item>
                </Col>
                <Col span={12}><Form.Item name="target_hardware_profile" label="目标硬件配置" rules={[{ required: true }]}><Input /></Form.Item></Col>
                <Col span={12}><Form.Item name="tool_version" label="量化工具版本" rules={[{ required: true }]}><Input /></Form.Item></Col>
                <Col span={12}><Form.Item name="quantization_gpu_hourly_cost_usd" label="量化 GPU 每小时成本（USD）" rules={[{ required: true }]}><InputNumber min={0} step={0.1} style={{ width: "100%" }} /></Form.Item></Col>
                {["AWQ", "GPTQ"].includes(selectedQuantizationProfile ?? "") ? (
                  <Col span={12}><Form.Item name="calibration_sample_count" label="校准样本数" rules={[{ required: true }]}><InputNumber min={32} max={2048} style={{ width: "100%" }} /></Form.Item></Col>
                ) : null}
                <Col span={12}><Form.Item name="quantization_max_sequence_length" label="最大校准/上下文长度" rules={[{ required: true }]}><InputNumber min={128} max={131072} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={24}><Form.Item name="incompatibilities" label="已知不兼容项（每行一项）"><Input.TextArea rows={3} /></Form.Item></Col>
              </>
            ) : null}
            {selectedMethod === "TIMESERIES_TRANSFORMER" ? (
              <>
                <Col span={24}><Alert type="info" showIcon message="设备级时序训练" description="使用 industrial-telemetry-sequence-v1 快照，以振动、轴承温度、电流和转速序列执行遮蔽重建；同一设备不会跨训练/验证分区。" /></Col>
                <Col span={8}><Form.Item name="d_model" label="Transformer 维度" rules={[{ required: true }]}><InputNumber min={16} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="nhead" label="注意力头数" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="num_layers" label="Encoder 层数" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="dim_feedforward" label="前馈网络维度" rules={[{ required: true }]}><InputNumber min={16} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="mask_probability" label="遮蔽比例" rules={[{ required: true }]}><InputNumber min={0.01} max={0.9} step={0.01} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            ) : null}
            {selectedMethod === "TIMESERIES_RULE_BASELINE" ? (
              <>
                <Col span={24}><Alert type="info" showIcon message="固定规则对照基线" description="该实验只登记规则版本与阈值，不启动训练 Worker；独立评测时按同一冻结遥测集计算误报率、漏报率和准确率。" /></Col>
                <Col span={12}><Form.Item label="规则版本"><Input value="registered-industrial-rule-baseline-v1" disabled /></Form.Item></Col>
                <Col span={12}><Form.Item name="rule_threshold" label="告警阈值" rules={[{ required: true }]}><InputNumber min={0.01} max={0.99} step={0.01} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            ) : null}
            {selectedMethod === "RUL_TRANSFORMER" ? (
              <>
                <Col span={24}><Alert type="info" showIcon message="监督式剩余寿命分位数训练" description="只使用 industrial-rul-sequence-v1 中已观察真实故障的提前量，输出 P10/P50/P90；数据按设备隔离分区，产物仍需独立 Gold Evaluation 后才能接入 PM-004。" /></Col>
                <Col span={8}><Form.Item name="d_model" label="Transformer 维度" rules={[{ required: true }]}><InputNumber min={16} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="nhead" label="注意力头数" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="num_layers" label="Encoder 层数" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="dim_feedforward" label="前馈网络维度" rules={[{ required: true }]}><InputNumber min={16} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item label="监督目标"><Input value="log1p(告警至故障分钟数) · P10/P50/P90" disabled /></Form.Item></Col>
              </>
            ) : null}
            {selectedMethod === "RUL_EMPIRICAL_BASELINE" ? (
              <>
                <Col span={24}><Alert type="info" showIcon message="冻结经验提前量基线" description="P10/P50/P90 必须来自候选训练快照的 train 分区；独立 Worker 会重新计算并核对 Manifest、样本数和分位数，不能通过浏览器削弱基线。该实验不启动训练 Worker。" /></Col>
                <Col span={8}><Form.Item name="empirical_p10_minutes" label="历史 P10（分钟）" rules={[{ required: true }]}><InputNumber min={0.000001} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="empirical_p50_minutes" label="历史 P50（分钟）" rules={[{ required: true }]}><InputNumber min={0.000001} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="empirical_p90_minutes" label="历史 P90（分钟）" rules={[{ required: true }]}><InputNumber min={0.000001} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="source_sample_count" label="训练分区样本数" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="history_cutoff" label="历史事实截止时间" rules={[{ required: true }]}><Input placeholder="2026-08-01T00:00:00Z" /></Form.Item></Col>
              </>
            ) : null}
          </Row>
        </Form>
      </Modal>

      <Modal
        title="创建独立评测任务"
        open={evaluationJobOpen}
        onCancel={() => setEvaluationJobOpen(false)}
        onOk={() => evaluationJobForm.submit()}
        confirmLoading={busy === "create-evaluation-job"}
        width={760}
      >
        {selectedEvaluationProfile === "PPO_RESEARCH_SAFETY" ? (
          <PpoResearchSafetyPanel />
        ) : (
        <Alert
          type="warning"
          showIcon
          message={selectedEvaluationProfile === "RETRIEVAL_COMPONENT"
            ? "检索组件使用独立的适用门禁"
            : selectedEvaluationProfile === "VLM_COMPONENT"
              ? "视觉组件只消费冻结的 Multimodal Gold"
            : selectedEvaluationProfile === "ASR_COMPONENT"
                ? "语音组件只消费冻结的 ASR Gold"
              : selectedEvaluationProfile === TTS_EVALUATION_PROFILE
                ? "TTS 组件只消费冻结的 TTS Gold 与固定 ASR 校验器"
              : selectedEvaluationProfile === "TIMESERIES_COMPONENT"
                ? "时序组件执行规则基线与 Transformer 独立对照"
              : selectedEvaluationProfile === "RUL_COMPONENT"
                ? "RUL 组件执行冻结经验基线与 Transformer 独立对照"
              : selectedEvaluationProfile === "EDGE_MODEL_COMPONENT"
                ? "GGUF 候选使用 llama.cpp 独立边缘评测"
              : "模型组件评测不会伪造 Agent 安全证据"}
          description={selectedEvaluationProfile === "RETRIEVAL_COMPONENT"
            ? "Retrieval Gold 只评估经复核的查询、相关文档、ACL Canary、索引兼容、时延和成本；组合发布仍须通过 AGENT_RUNTIME 全链门禁。"
            : selectedEvaluationProfile === "VLM_COMPONENT"
              ? "独立 Worker 重新校验图片清单、哈希、签名和候选绑定，再比较诊断准确率、区域 IoU、幻觉率、风险切片、时延与成本。"
              : selectedEvaluationProfile === "ASR_COMPONENT"
                ? "独立 Worker 重新校验音频清单、哈希、采样率、语言、时长和候选绑定，再比较 WER、CER、工业噪声切片、时延与成本。"
              : selectedEvaluationProfile === TTS_EVALUATION_PROFILE
                ? "独立 Worker 使用同一授权声音 Profile 合成基线与候选，再由固定 ASR 校验器计算安全短语、工业术语、可懂度和音频完整性；MOS 仍须在目标环境人工验收。"
              : selectedEvaluationProfile === "TIMESERIES_COMPONENT"
                ? "独立 Worker 只消费未参与训练的冻结遥测快照，校验 Parquet、标签覆盖和设备分区，再比较准确率、平衡准确率、误报率、漏报率、时延与成本。"
              : selectedEvaluationProfile === "RUL_COMPONENT"
                ? "独立 Worker 只消费未参与训练的冻结 RUL Gold，重新计算训练分区经验基线，再比较相对准确率、中位绝对误差、Pinball Loss、P10-P90 覆盖率、区间宽度、时延与成本。"
              : selectedEvaluationProfile === "EDGE_MODEL_COMPONENT"
                ? "源 PEFT 基线由 Transformers 执行，GGUF 候选由固定版本 llama.cpp 执行；记录 CPU 成本、线程、上下文、制品哈希和确定性重放。正式发布仍须再通过 AGENT_RUNTIME 安全门禁。"
              : "任务只登记候选、基线、冻结评测集、策略和执行镜像。逐样本得分与硬门禁由独立 Worker 生成；MODEL_COMPONENT 无法覆盖的工具、审批和跨租户门禁会判定为失败。"}
          style={{ marginBottom: 16 }}
        />
        )}
        <Form
          form={evaluationJobForm}
          layout="vertical"
          onFinish={(values) => void createEvaluation(values)}
          initialValues={{
            max_new_tokens: 256,
            top_k: 10,
            max_image_pixels: 4194304,
            sampling_rate: 16000,
            language: "zh",
            max_audio_seconds: 30,
            anomaly_threshold: 1,
            rule_threshold: 0.65,
            precision: "bfloat16",
            gpu_hourly_cost_usd: 1,
            cpu_hourly_cost_usd: 0.12,
            threads: 8,
            context_size: 8192,
            framework_mode: "DETERMINISTIC",
            target_profile: "MODEL_COMPONENT",
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="candidate_experiment_id" label="候选实验" rules={[{ required: true }]}>
                <Select options={experimentOptions.filter((item) => (
                  selectedEvaluationProfile === "RETRIEVAL_COMPONENT"
                    ? ["EMBEDDING", "RERANKER"].includes(item.method)
                    : selectedEvaluationProfile === "VLM_COMPONENT"
                      ? item.method === "VLM"
                    : selectedEvaluationProfile === "ASR_COMPONENT"
                      ? item.method === "ASR"
                    : selectedEvaluationProfile === TTS_EVALUATION_PROFILE
                      ? item.method === TTS_TRAINING_METHOD
                    : selectedEvaluationProfile === "TIMESERIES_COMPONENT"
                      ? item.method === "TIMESERIES_TRANSFORMER"
                    : selectedEvaluationProfile === "RUL_COMPONENT"
                      ? item.method === "RUL_TRANSFORMER"
                    : selectedEvaluationProfile === "PPO_RESEARCH_SAFETY"
                      ? item.method === "PPO"
                    : selectedEvaluationProfile === "EDGE_MODEL_COMPONENT"
                      ? item.method === "QUANTIZATION" && item.training_config.target_runtime === "LLAMA_CPP"
                    : ["LORA", "QLORA", "DPO", "GRPO", "QUANTIZATION"].includes(item.method) &&
                      (selectedEvaluationProfile !== "MODEL_COMPONENT" || item.training_config.target_runtime !== "LLAMA_CPP")
                )).map((item) => ({ value: item.experiment_id, label: `${item.method} · ${item.experiment_id}` }))} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="baseline_experiment_id" label="BASELINE 实验" rules={[{ required: true }]}>
                <Select options={experimentOptions.filter((item) => {
                  if (selectedEvaluationProfile === "TIMESERIES_COMPONENT") {
                    return item.method === "TIMESERIES_RULE_BASELINE";
                  }
                  if (selectedEvaluationProfile === "RUL_COMPONENT") {
                    return item.method === "RUL_EMPIRICAL_BASELINE";
                  }
                  const candidate = experimentOptions.find(
                    (option) => option.experiment_id === selectedEvaluationCandidateId
                  );
                  if (candidate?.method === "QUANTIZATION") {
                    return item.experiment_id === candidate.training_config.source_experiment_id;
                  }
                  return item.method === "BASELINE";
                }).map((item) => ({ value: item.experiment_id, label: `${item.method} · ${item.experiment_id}` }))} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="suite_id" label="冻结评测集" rules={[{ required: true }]}>
                <Select options={(suites ?? []).filter((item) => {
                  const telemetry = snapshots.some((snapshot) =>
                    snapshot.snapshot_id === item.source_snapshot_id &&
                    snapshot.contract_version === "industrial-telemetry-sequence-v1");
                  const rul = snapshots.some((snapshot) =>
                    snapshot.snapshot_id === item.source_snapshot_id &&
                    snapshot.contract_version === "industrial-rul-sequence-v1");
                  if (selectedEvaluationProfile === "TIMESERIES_COMPONENT") return telemetry;
                  if (selectedEvaluationProfile === "RUL_COMPONENT") return rul;
                  return !telemetry && !rul;
                }).map((item) => ({ value: item.suite_id, label: `${item.tier} · ${item.name} ${item.version}` }))} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="policy_id" label="评测策略" rules={[{ required: true }]}>
                <Select options={(policies ?? []).filter((item) => (
                  selectedEvaluationProfile === "RETRIEVAL_COMPONENT"
                    ? ["recall_at_k", "mrr", "ndcg_at_k"].includes(item.primary_metric)
                    : selectedEvaluationProfile === "VLM_COMPONENT"
                      ? item.primary_metric === "vlm_diagnostic_accuracy"
                    : selectedEvaluationProfile === "ASR_COMPONENT"
                      ? item.primary_metric === "asr_word_accuracy"
                    : selectedEvaluationProfile === TTS_EVALUATION_PROFILE
                      ? item.primary_metric === "tts_intelligibility"
                    : selectedEvaluationProfile === "TIMESERIES_COMPONENT"
                      ? item.primary_metric === "timeseries_accuracy"
                    : selectedEvaluationProfile === "RUL_COMPONENT"
                      ? item.primary_metric === "rul_relative_accuracy"
                    : selectedEvaluationProfile === "PPO_RESEARCH_SAFETY"
                      ? item.primary_metric === "ppo_safe_response_rate"
                    : !["recall_at_k", "mrr", "ndcg_at_k", "vlm_diagnostic_accuracy", "asr_word_accuracy", "tts_intelligibility", "timeseries_accuracy", "rul_relative_accuracy", "ppo_safe_response_rate"].includes(item.primary_metric)
                )).map((item) => ({ value: item.policy_id, label: `${item.name} ${item.version}` }))} />
              </Form.Item>
            </Col>
            <Col span={24}>
              <Form.Item name="target_profile" label="评测运行 Profile" rules={[{ required: true }]}>
                <Select
                  onChange={() => {
                    evaluationJobForm.setFieldsValue({
                      candidate_experiment_id: undefined,
                      baseline_experiment_id: undefined,
                      policy_id: undefined,
                      framework_mode: "DETERMINISTIC",
                    });
                  }}
                  options={[
                    { value: "MODEL_COMPONENT", label: "模型组件：生成质量、时延、成本与重放" },
                    { value: "EDGE_MODEL_COMPONENT", label: "边缘模型组件：GGUF + llama.cpp 质量、CPU 时延与成本" },
                    { value: "AGENT_RUNTIME", label: "Agent 运行时：追加租户、工具、T2/T3、引用与审批回放" },
                    { value: "RETRIEVAL_COMPONENT", label: "检索组件：Recall@K、MRR、NDCG、ACL 与索引兼容" },
                    { value: "VLM_COMPONENT", label: "视觉组件：诊断准确率、区域引用、幻觉与风险切片" },
                    { value: "ASR_COMPONENT", label: "语音组件：WER、CER、工业噪声切片、时延与成本" },
                    { value: TTS_EVALUATION_PROFILE, label: "语音合成组件：安全短语、术语、可懂度、音频完整性与回归" },
                    { value: "TIMESERIES_COMPONENT", label: "时序组件：准确率、误报/漏报、规则基线与 Transformer" },
                    { value: "RUL_COMPONENT", label: "RUL 组件：MAE、Pinball、区间覆盖与经验基线对照" },
                    { value: "PPO_RESEARCH_SAFETY", label: "PPO 研究安全：奖励一致性、reward-hacking 与安全响应" },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col span={12}><Form.Item name="runner_git_commit" label="评测代码 Commit" rules={[{ required: true }]}><Input /></Form.Item></Col>
            <Col span={12}><Form.Item name="container_digest" label="评测镜像摘要" rules={[{ required: true }]}><Input placeholder="sha256:..." /></Form.Item></Col>
            {selectedEvaluationProfile === "RETRIEVAL_COMPONENT" ? (
              <Col span={8}><Form.Item name="top_k" label="检索 Top K" rules={[{ required: true }]}><InputNumber min={1} max={100} style={{ width: "100%" }} /></Form.Item></Col>
            ) : !["TIMESERIES_COMPONENT", "RUL_COMPONENT", TTS_EVALUATION_PROFILE].includes(selectedEvaluationProfile ?? "") ? (
              <Col span={8}><Form.Item name="max_new_tokens" label="最大输出 Token" rules={[{ required: true }]}><InputNumber min={1} max={2048} style={{ width: "100%" }} /></Form.Item></Col>
            ) : null}
            {selectedEvaluationProfile === "VLM_COMPONENT" ? (
              <Col span={8}><Form.Item name="max_image_pixels" label="单图最大像素" rules={[{ required: true }]}><InputNumber min={65536} max={16777216} style={{ width: "100%" }} /></Form.Item></Col>
            ) : null}
            {selectedEvaluationProfile === "ASR_COMPONENT" ? (
              <>
                <Col span={8}><Form.Item name="sampling_rate" label="采样率" rules={[{ required: true }]}><InputNumber min={16000} max={16000} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="language" label="转写语言" rules={[{ required: true }]}><Input maxLength={32} /></Form.Item></Col>
                <Col span={8}><Form.Item name="max_audio_seconds" label="单段最长秒数" rules={[{ required: true }]}><InputNumber min={0.1} max={30} step={0.5} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            ) : null}
            {selectedEvaluationProfile === TTS_EVALUATION_PROFILE ? (
              <>
                <Col span={8}><Form.Item name="sampling_rate" label="合成采样率" rules={[{ required: true }]}><InputNumber min={16000} max={16000} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="language" label="合成语言" rules={[{ required: true }]}><Input maxLength={32} /></Form.Item></Col>
                <Col span={8}><Form.Item name="max_audio_seconds" label="单段最长秒数" rules={[{ required: true }]}><InputNumber min={0.1} max={30} step={0.5} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={24}><Form.Item name="asr_verifier_model_id" label="冻结 ASR 校验器 ID" rules={[{ required: true }]}><Input /></Form.Item></Col>
                <Col span={12}><Form.Item name="asr_verifier_revision" label="ASR 校验器 40 位 Revision" rules={[{ required: true, len: 40 }]}><Input /></Form.Item></Col>
                <Col span={12}><Form.Item name="asr_verifier_digest" label="ASR 校验器 Digest" rules={[{ required: true }]}><Input placeholder="hf-revision:<commit SHA>" /></Form.Item></Col>
              </>
            ) : null}
            {selectedEvaluationProfile === "TIMESERIES_COMPONENT" ? (
              <>
                <Col span={8}><Form.Item name="anomaly_threshold" label="Transformer 异常阈值" rules={[{ required: true }]}><InputNumber min={0.000001} step={0.1} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="rule_threshold" label="规则基线阈值" rules={[{ required: true }]}><InputNumber min={0.01} max={0.99} step={0.01} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            ) : null}
            <Col span={8}><Form.Item name="precision" label="推理精度" rules={[{ required: true }]}><Select options={["bfloat16", "float16"].map((value) => ({ value }))} /></Form.Item></Col>
            <Col span={8}><Form.Item name="gpu_hourly_cost_usd" label="GPU 每小时成本 USD" rules={[{ required: true }]}><InputNumber min={0.0001} step={0.1} style={{ width: "100%" }} /></Form.Item></Col>
            {selectedEvaluationUsesLlamaCpp ? (
              <>
                <Col span={8}><Form.Item name="cpu_hourly_cost_usd" label="边缘 CPU 每小时成本 USD" rules={[{ required: true }]}><InputNumber min={0.0001} step={0.01} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="threads" label="llama.cpp 线程数" rules={[{ required: true }]}><InputNumber min={1} max={256} style={{ width: "100%" }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="context_size" label="llama.cpp 上下文长度" rules={[{ required: true }]}><InputNumber min={1024} max={131072} style={{ width: "100%" }} /></Form.Item></Col>
              </>
            ) : null}
            <Col span={24}><Form.Item name="framework_mode" label="评测框架模式" rules={[{ required: true }]}><Select disabled={["RETRIEVAL_COMPONENT", "VLM_COMPONENT", "ASR_COMPONENT", TTS_EVALUATION_PROFILE, "TIMESERIES_COMPONENT", "RUL_COMPONENT", "PPO_RESEARCH_SAFETY"].includes(selectedEvaluationProfile ?? "")} options={[{ value: "DETERMINISTIC", label: "确定性业务指标" }, { value: "RAGAS_DEEPEVAL", label: "Ragas + DeepEval 数据集适配（Judge 不参与硬门禁）" }]} /></Form.Item></Col>
          </Row>
        </Form>
      </Modal>

      <Drawer title="实验复现清单" open={Boolean(selected)} onClose={closeExperiment} width={720}>
        {selected ? (
          <div className="page-stack">
            <Descriptions bordered size="small" column={1}>
              <Descriptions.Item label="实验 ID">{selected.experiment_id}</Descriptions.Item>
              <Descriptions.Item label="Git Commit">{selected.git_commit}</Descriptions.Item>
              <Descriptions.Item label="镜像摘要">{selected.container_digest}</Descriptions.Item>
              <Descriptions.Item label="数据清单摘要">{selected.dataset_manifest_hash}</Descriptions.Item>
              <Descriptions.Item label="模型摘要">{selected.base_model_digest}</Descriptions.Item>
              <Descriptions.Item label="Tokenizer 摘要">{selected.tokenizer_digest}</Descriptions.Item>
              <Descriptions.Item label="模板摘要">{selected.chat_template_digest}</Descriptions.Item>
              <Descriptions.Item label="配置摘要">{selected.config_hash}</Descriptions.Item>
              <Descriptions.Item label="随机种子">{selected.random_seeds.join(", ")}</Descriptions.Item>
            </Descriptions>
            <Typography.Title level={5}>训练、分布式与硬件配置</Typography.Title>
            <pre className="json-report">{JSON.stringify({ training_config: selected.training_config, distributed_profile: selected.distributed_profile, hardware_topology: selected.hardware_topology, cost_summary: selected.cost_summary, artifacts: selected.artifacts }, null, 2)}</pre>
          </div>
        ) : null}
      </Drawer>

      <Drawer
        title="最终评测结果"
        open={Boolean(selectedEvaluation)}
        onClose={closeEvaluation}
        width={760}
      >
        {selectedEvaluation ? (
          <div className="page-stack">
            {selectedEvaluation.primary_metric === "ppo_safe_response_rate" ? (
              <PpoResearchSafetyPanel
                metrics={ppoSafetyMetrics(selectedEvaluation.slice_metrics)}
              />
            ) : null}
            {selectedEvaluation.primary_metric === "tts_intelligibility" ? (
              <TtsObjectiveEvidencePanel metrics={selectedEvaluation.slice_metrics} />
            ) : null}
            <Descriptions bordered size="small" column={1}>
              <Descriptions.Item label="评测 ID">
                <Typography.Text code copyable>{selectedEvaluation.evaluation_id}</Typography.Text>
              </Descriptions.Item>
              <Descriptions.Item label="状态">{selectedEvaluation.status}</Descriptions.Item>
              <Descriptions.Item label="决策">
                <Tag color={decisionColor(selectedEvaluation.decision)}>
                  {selectedEvaluation.decision}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="候选实验">
                {selectedEvaluation.candidate_experiment_id}
              </Descriptions.Item>
              <Descriptions.Item label="基线实验">
                {selectedEvaluation.baseline_experiment_id}
              </Descriptions.Item>
              <Descriptions.Item label="评测集">{selectedEvaluation.suite_id}</Descriptions.Item>
              <Descriptions.Item label="门禁策略">{selectedEvaluation.policy_id}</Descriptions.Item>
              <Descriptions.Item label="对比类型">
                {selectedEvaluation.comparison_kind === "SYNTHETIC_DATASET_ABLATION"
                  ? "真实基线 / 合成增强消融"
                  : "模型方法对比"}
              </Descriptions.Item>
              <Descriptions.Item label="主指标">{selectedEvaluation.primary_metric}</Descriptions.Item>
              <Descriptions.Item label="候选分数">
                {selectedEvaluation.candidate_score}
              </Descriptions.Item>
              <Descriptions.Item label="基线分数">
                {selectedEvaluation.baseline_score}
              </Descriptions.Item>
              <Descriptions.Item label="质量变化">
                {percentage(selectedEvaluation.quality_delta)}
              </Descriptions.Item>
              <Descriptions.Item label="95% 置信区间">
                {percentage(selectedEvaluation.ci_low)} ～ {percentage(selectedEvaluation.ci_high)}
              </Descriptions.Item>
              <Descriptions.Item label="时延改善">
                {percentage(selectedEvaluation.latency_improvement)}
              </Descriptions.Item>
              <Descriptions.Item label="成本改善">
                {percentage(selectedEvaluation.cost_improvement)}
              </Descriptions.Item>
              <Descriptions.Item label="报告哈希">
                <Typography.Text code copyable>{selectedEvaluation.report_hash}</Typography.Text>
              </Descriptions.Item>
              <Descriptions.Item label="失败原因">
                {selectedEvaluation.failure_reason ?? "—"}
              </Descriptions.Item>
            </Descriptions>
            {selectedEvaluation.comparison_kind === "SYNTHETIC_DATASET_ABLATION" ? (
              <>
                <Typography.Title level={5}>合成增强消融证据</Typography.Title>
                <pre className="json-report">
                  {JSON.stringify(selectedEvaluation.comparison_context, null, 2)}
                </pre>
              </>
            ) : null}
            <Typography.Title level={5}>安全与治理硬门禁</Typography.Title>
            <Space wrap>
              {Object.entries(selectedEvaluation.hard_gate_results).map(([gate, passed]) => (
                <Tag key={gate} color={passed ? "green" : "red"}>
                  {gate}: {passed ? "通过" : "失败"}
                </Tag>
              ))}
            </Space>
            <Typography.Title level={5}>切片指标</Typography.Title>
            <pre className="json-report">
              {JSON.stringify(selectedEvaluation.slice_metrics, null, 2)}
            </pre>
            <Button
              type="primary"
              loading={busy === `evidence-${selectedEvaluation.evaluation_id}`}
              onClick={() => void saveEvaluationEvidence(selectedEvaluation)}
            >下载逐样本证据报告</Button>
          </div>
        ) : null}
      </Drawer>
    </AppShell>
  );
}

function percentage(value: number) {
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
}

function finiteMetric(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function metricPercentage(value: number | undefined) {
  return value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
}

function ppoSafetyMetrics(value: unknown): PpoSafetyMetrics | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const metrics = (value as Record<string, unknown>).ppo_safety;
  return metrics && typeof metrics === "object" && !Array.isArray(metrics)
    ? metrics as PpoSafetyMetrics
    : undefined;
}

function methodColor(method: string) {
  if (method === "RUL_TRANSFORMER") return "green";
  if (method === "TIMESERIES_TRANSFORMER") return "orange";
  if (method === "ASR") return "magenta";
  if (method === "VLM") return "lime";
  if (method === "RERANKER") return "volcano";
  if (method === "EMBEDDING") return "geekblue";
  if (method === "PPO") return "red";
  if (method === "GRPO") return "gold";
  if (method === "DPO") return "cyan";
  if (method === "QLORA") return "purple";
  if (method === "LORA") return "blue";
  return "default";
}


function decisionColor(decision: string) {
  if (decision === "CANDIDATE") return "green";
  if (decision === "REJECTED") return "red";
  if (decision === "SMOKE_PASSED") return "blue";
  return "orange";
}
