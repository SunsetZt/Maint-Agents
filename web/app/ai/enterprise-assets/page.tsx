"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  Input,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Switch,
  Table,
  Tabs,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  advanceEnterpriseCandidateReleaseBatchToApproval,
  createEnterpriseCandidateReleaseBatch,
  createEnterpriseReleaseDraft,
  createEnterpriseStagingBaselineDraft,
  type EnterpriseCandidateReleaseBatch,
  type EnterpriseCandidateReleaseBatchProgress,
  type EnterpriseModelComponent,
  type EnterpriseModelImportAcceptance,
  type EnterpriseProjectAdoption,
  type EnterpriseProjectAssurance,
  type EnterpriseProjectClosure,
  type EnterpriseProjectAsset,
  type EnterpriseProjectExperimentHistory,
  type EnterpriseReleaseDraftPreview,
  type EnterpriseRuntimeBinding,
  getEnterpriseProjectAdoption,
  getEnterpriseProjectAssurance,
  getEnterpriseProjectClosureReadiness,
  getEnterpriseModelImportAcceptance,
  getEnterpriseRuntimeBindings,
  importEnterpriseModelAsset,
  listEnterpriseCandidateReleaseBatches,
  previewEnterpriseReleaseDraft,
} from "@/lib/api/client";

type AdoptionCounts = {
  activeAssets: number;
  runtimeAssets: number;
  experimentHistory: number;
};

type RuntimeFilter = "ALL" | "RUNTIME" | "EVIDENCE";
type EnterpriseRuntimeRegistration = EnterpriseRuntimeBinding["registrations"][number];
type EnterpriseClosureEvidence = EnterpriseProjectClosure["source_evidence"][number];
type ProjectAssuranceSecurityScenario =
  EnterpriseProjectAssurance["security"]["scenarios"][number];
type ProjectAssuranceSlo = EnterpriseProjectAssurance["operations"]["observations"][number];

type RuntimeBindingCounts = {
  components: number;
  registeredComponents: number;
  deployedComponents: number;
  activeAliasComponents: number;
};

type ModelImportAcceptanceCounts = {
  components: number;
  eligibleComponents: number;
  totalActualSamples: number;
  commonBaselineReleaseIds: string[];
};

type ProjectClosureCounts = {
  coverageDomains: number;
  sourceEvidence: number;
  intentionallyUnverifiedMethods: number;
  productionBlockers: number;
};

type ProjectAssuranceCounts = {
  securityScenarios: number;
  recoveryComponents: number;
  healthySlos: number;
  stagingScenarios: number;
  signoffs: number;
};

type ReleaseDraftFormValues = {
  baseline_release_id: string;
  auto_shadow_enabled: boolean;
  namespace: string;
  gateway_name: string;
  hostname: string;
  route_name: string;
  stable_service_name: string;
  service_account_name: string;
  serving_runtime_name: string;
  artifact_uri_prefix: string;
};

const blockerLabels: Record<string, string> = {
  VLM_SUPPLY_CHAIN_EVIDENCE_REQUIRED: "VLM 缺少与实际运行镜像、模型制品及来源匹配的已验证证据",
  ASR_SUPPLY_CHAIN_EVIDENCE_REQUIRED: "ASR 缺少与实际运行镜像、模型制品及来源匹配的已验证证据",
  COMPONENT_RUNTIME_IDENTITY_REQUIRED: "缺少可信运行镜像或源码来源登记，不能用训练镜像代替",
  COMPONENT_SUPPLY_CHAIN_EVIDENCE_REQUIRED: "组件证据缺失或不匹配，请先完成可信验证登记",
  LEGACY_EVIDENCE_REQUIRED: "历史 Release 需要在发布中心创建同模型补证版本，再独立审批",
  ENTERPRISE_MODEL_ASSET_NOT_IMPORTED: "项目权威候选尚未导入当前租户模型台账",
  ENTERPRISE_MODEL_IMPORT_INTEGRITY_FAILED: "候选导入证明完整性校验失败，禁止创建草稿",
  CANDIDATE_EXPERIMENT_NOT_IMPORTED: "候选实验尚未导入当前租户训练台账",
  CANDIDATE_METHOD_MISMATCH: "候选训练方法与组件类型不一致",
  CANDIDATE_NOT_RELEASE_ELIGIBLE: "候选训练或许可证状态尚未达到发布要求",
  CANDIDATE_EVALUATION_NOT_IMPORTED: "未找到已完成且决策为 CANDIDATE 的独立评测",
  CANDIDATE_EVALUATION_AMBIGUOUS: "找到多个候选评测，需先收敛为唯一权威评测",
  CANDIDATE_ALREADY_REGISTERED: "该候选已经登记到当前租户 Model Registry",
  COMPATIBLE_BASELINE_RELEASE_REQUIRED: "缺少已批准且清单结构兼容的基线 Release",
  LLM_SUPPLY_CHAIN_EVIDENCE_REQUIRED: "缺少 LLM 模型制品的有效供应链证据",
  RUL_SUPPLY_CHAIN_EVIDENCE_REQUIRED: "缺少 RUL 模型制品的有效供应链证据",
  EMBEDDING_SUPPLY_CHAIN_EVIDENCE_REQUIRED: "缺少 Embedding 模型制品的有效供应链证据",
  RERANKER_SUPPLY_CHAIN_EVIDENCE_REQUIRED: "缺少 Reranker 模型制品的有效供应链证据",
  LLM_QUANTIZATION_REQUIRES_MANUAL_RELEASE_PLAN: "量化 LLM 需要在发布中心显式配置运行时",
};

const unverifiedMethodLabels: Record<string, string> = {
  DPO_ENTERPRISE_VALUE: "当前 DPO 候选经正式 Agent Runtime 评测被拒绝，需新候选和新 Gold",
  PPO_PRODUCTION_RELEASE: "历史闭环仍把 PPO 标为研究候选，请刷新到已完成发布验收的当前回执",
  TTS_ENTERPRISE_VALUE: "当前 TTS 实际 GPU 候选无企业价值增益，需新数据或新方案",
  EMBEDDING_ENTERPRISE_VALUE: "当前 Embedding 候选无检索收益，需新候选与独立 Gold",
  RERANKER_ENTERPRISE_VALUE: "当前 Reranker 候选无排序收益，需新候选与独立 Gold",
};

const productionBlockerLabels: Record<string, string> = {
  enterprise_data_owner_authorization: "企业数据所有者正式授权",
  production_https_mtls_and_provider_credentials: "生产 HTTPS/mTLS 与供应商凭据",
  enterprise_oidc_and_production_secret_delivery: "企业 OIDC 与生产 Secret 投递",
  production_kubernetes_gpu_serving_and_capacity: "生产 Kubernetes/GPU 推理与容量验收",
  cross_tenant_and_multimodal_attack_exercises: "目标环境跨租户与多模态攻击演练",
  slo_rpo_rto_and_disaster_recovery_observation: "SLO、RPO/RTO 与灾备观察窗口",
  business_security_platform_signoff: "业务、安全、平台三方正式签署",
};

export default function EnterpriseAssetsPage() {
  const [draftForm] = Form.useForm<ReleaseDraftFormValues>();
  const batchIdempotencyKey = useRef<string | undefined>(undefined);
  const [adoption, setAdoption] = useState<EnterpriseProjectAdoption>();
  const [counts, setCounts] = useState<AdoptionCounts>();
  const [runtimeBindings, setRuntimeBindings] = useState<EnterpriseRuntimeBinding[]>([]);
  const [modelImportAcceptance, setModelImportAcceptance] =
    useState<EnterpriseModelImportAcceptance>();
  const [modelImportAcceptanceCounts, setModelImportAcceptanceCounts] =
    useState<ModelImportAcceptanceCounts>();
  const [projectClosure, setProjectClosure] = useState<EnterpriseProjectClosure>();
  const [projectClosureCounts, setProjectClosureCounts] =
    useState<ProjectClosureCounts>();
  const [projectAssurance, setProjectAssurance] =
    useState<EnterpriseProjectAssurance>();
  const [projectAssuranceCounts, setProjectAssuranceCounts] =
    useState<ProjectAssuranceCounts>();
  const [runtimeCounts, setRuntimeCounts] = useState<RuntimeBindingCounts>();
  const [requestId, setRequestId] = useState<string>();
  const [runtimeRequestId, setRuntimeRequestId] = useState<string>();
  const [modelImportAcceptanceRequestId, setModelImportAcceptanceRequestId] =
    useState<string>();
  const [projectClosureRequestId, setProjectClosureRequestId] = useState<string>();
  const [projectAssuranceRequestId, setProjectAssuranceRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [refreshing, setRefreshing] = useState(false);
  const [domain, setDomain] = useState("ALL");
  const [runtimeFilter, setRuntimeFilter] = useState<RuntimeFilter>("ALL");
  const [search, setSearch] = useState("");
  const [draftOpen, setDraftOpen] = useState(false);
  const [draftComponent, setDraftComponent] = useState<EnterpriseModelComponent>();
  const [draftPreview, setDraftPreview] = useState<EnterpriseReleaseDraftPreview>();
  const [draftLoading, setDraftLoading] = useState(false);
  const [draftSubmitting, setDraftSubmitting] = useState(false);
  const [baselineCreating, setBaselineCreating] = useState(false);
  const [importingComponent, setImportingComponent] =
    useState<EnterpriseModelComponent>();
  const [draftError, setDraftError] = useState<unknown>();
  const [createdReleaseId, setCreatedReleaseId] = useState<string>();
  const [createdBaselineReleaseId, setCreatedBaselineReleaseId] =
    useState<string>();
  const [batchCreating, setBatchCreating] = useState(false);
  const [batchError, setBatchError] = useState<unknown>();
  const [createdBatch, setCreatedBatch] =
    useState<EnterpriseCandidateReleaseBatch>();
  const [releaseBatches, setReleaseBatches] =
    useState<EnterpriseCandidateReleaseBatchProgress[]>([]);
  const [releaseBatchCounts, setReleaseBatchCounts] = useState({
    awaitingApproval: 0,
    approved: 0,
  });
  const [advancingBatch, setAdvancingBatch] = useState<string>();
  const [batchAdvanceError, setBatchAdvanceError] = useState<unknown>();
  const [submittedBatch, setSubmittedBatch] =
    useState<EnterpriseCandidateReleaseBatchProgress>();

  const load = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true);
    try {
      const [
        result,
        closureResult,
        assuranceResult,
        runtimeResult,
        importAcceptanceResult,
        batchResult,
      ] =
        await Promise.all([
        getEnterpriseProjectAdoption(),
        getEnterpriseProjectClosureReadiness(),
        getEnterpriseProjectAssurance(),
        getEnterpriseRuntimeBindings(),
        getEnterpriseModelImportAcceptance(),
        listEnterpriseCandidateReleaseBatches(),
      ]);
      setAdoption(result.adoption);
      setCounts({
        activeAssets: result.activeAssets,
        runtimeAssets: result.runtimeAssets,
        experimentHistory: result.experimentHistory,
      });
      setProjectClosure(closureResult.closure);
      setProjectClosureCounts({
        coverageDomains: closureResult.coverageDomains,
        sourceEvidence: closureResult.sourceEvidence,
        intentionallyUnverifiedMethods: closureResult.intentionallyUnverifiedMethods,
        productionBlockers: closureResult.productionBlockers,
      });
      setProjectAssurance(assuranceResult.assurance);
      setProjectAssuranceCounts({
        securityScenarios: assuranceResult.securityScenarios,
        recoveryComponents: assuranceResult.recoveryComponents,
        healthySlos: assuranceResult.healthySlos,
        stagingScenarios: assuranceResult.stagingScenarios,
        signoffs: assuranceResult.signoffs,
      });
      setRuntimeBindings(runtimeResult.bindings);
      setRuntimeCounts({
        components: runtimeResult.components,
        registeredComponents: runtimeResult.registeredComponents,
        deployedComponents: runtimeResult.deployedComponents,
        activeAliasComponents: runtimeResult.activeAliasComponents,
      });
      setModelImportAcceptance(importAcceptanceResult.acceptance);
      setModelImportAcceptanceCounts({
        components: importAcceptanceResult.components,
        eligibleComponents: importAcceptanceResult.eligibleComponents,
        totalActualSamples: importAcceptanceResult.totalActualSamples,
        commonBaselineReleaseIds: importAcceptanceResult.commonBaselineReleaseIds,
      });
      setReleaseBatches(batchResult.batches);
      setReleaseBatchCounts({
        awaitingApproval: batchResult.awaitingApproval,
        approved: batchResult.approved,
      });
      setRequestId(result.requestId);
      setProjectClosureRequestId(closureResult.requestId);
      setProjectAssuranceRequestId(assuranceResult.requestId);
      setRuntimeRequestId(runtimeResult.requestId);
      setModelImportAcceptanceRequestId(importAcceptanceResult.requestId);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const domains = useMemo(
    () => Array.from(new Set(adoption?.active_assets.map((item) => item.domain) ?? [])).sort(),
    [adoption],
  );
  const filteredAssets = useMemo(() => {
    const keyword = search.trim().toLowerCase();
    return (adoption?.active_assets ?? []).filter((asset) => {
      if (domain !== "ALL" && asset.domain !== domain) return false;
      if (runtimeFilter === "RUNTIME" && !asset.runtime_eligible) return false;
      if (runtimeFilter === "EVIDENCE" && asset.runtime_eligible) return false;
      if (!keyword) return true;
      return [
        asset.domain,
        asset.role,
        asset.asset_kind,
        asset.source_path,
        asset.source_status,
        ...asset.capabilities,
      ].some((value) => value.toLowerCase().includes(keyword));
    });
  }, [adoption, domain, runtimeFilter, search]);
  const coverage = useMemo(
    () => Object.entries(adoption?.coverage ?? {}).map(([name, status]) => ({ name, status })),
    [adoption],
  );
  const candidateBatchReady = useMemo(
    () =>
      (["LLM", "VLM", "RUL"] as const).every((component) => {
        const binding = runtimeBindings.find(
          (item) => item.evidence.component === component,
        );
        return (
          binding !== undefined &&
          binding.import_state === "IMPORTED" &&
          binding.registry_state === "NOT_REGISTERED"
        );
      }),
    [runtimeBindings],
  );

  const openDraftWizard = useCallback(
    async (component: EnterpriseModelComponent) => {
      setDraftOpen(true);
      setDraftComponent(component);
      setDraftPreview(undefined);
      setDraftError(undefined);
      setDraftLoading(true);
      try {
        const result = await previewEnterpriseReleaseDraft(component);
        setDraftPreview(result.preview);
        const suffix = component.toLowerCase();
        const isAsr = component === "ASR";
        const isTts = component === "TTS";
        const isEmbedding = component === "EMBEDDING";
        const isReranker = component === "RERANKER";
        draftForm.setFieldsValue({
          baseline_release_id: result.preview.baselines[0]?.release_id,
          auto_shadow_enabled: true,
          namespace: "industrial-models",
          gateway_name: "industrial-agent-gateway",
          hostname: "models.ops.example.com",
          route_name: `enterprise-${suffix}-candidate`,
          stable_service_name: isAsr
            ? "ioap-asr-stable"
            : isTts
              ? "ioap-tts-stable"
              : isEmbedding
                ? "ioap-embedding-stable"
              : isReranker
                ? "ioap-reranker-stable"
              : "industrial-agent-stable",
          service_account_name: "model-storage-reader",
          serving_runtime_name: isAsr
            ? "asr-whisper-peft-runtime"
            : isTts
              ? "tts-speecht5-postnet-runtime"
              : isEmbedding
                ? "sentence-transformers-embedding-runtime"
              : isReranker
                ? "reranker-cross-encoder-runtime"
              : "vllm-lora-runtime",
          artifact_uri_prefix: isAsr
            ? "s3://industrial-models/asr"
            : isTts
              ? "s3://industrial-models/tts"
              : isEmbedding
                ? "s3://industrial-models/embedding"
              : isReranker
                ? "s3://industrial-models/reranker"
              : "s3://industrial-models",
        });
      } catch (cause) {
        setDraftError(cause);
      } finally {
        setDraftLoading(false);
      }
    },
    [draftForm],
  );

  const createDraft = useCallback(
    async (values: ReleaseDraftFormValues) => {
      if (!draftComponent || !draftPreview?.eligible) return;
      setDraftSubmitting(true);
      setDraftError(undefined);
      try {
        const result = await createEnterpriseReleaseDraft(
          draftComponent,
          {
            baseline_release_id: values.baseline_release_id,
            target_environment: "STAGING",
            auto_shadow_enabled: values.auto_shadow_enabled,
            deployment_plan: values.auto_shadow_enabled
              ? {
                  namespace: values.namespace,
                  gateway_name: values.gateway_name,
                  hostname: values.hostname,
                  route_name: values.route_name,
                  stable_service_name: values.stable_service_name,
                  service_account_name: values.service_account_name,
                  serving_runtime_name: values.serving_runtime_name,
                  artifact_uri_prefix: values.artifact_uri_prefix,
                  routing_mode: "STANDARD",
                  autoscaling_mode: "FIXED",
                }
              : null,
          },
          globalThis.crypto?.randomUUID?.() ?? `enterprise-release-${Date.now()}`,
        );
        setCreatedReleaseId(result.onboarding.release_id);
        setDraftOpen(false);
        await load(true);
      } catch (cause) {
        setDraftError(cause);
      } finally {
        setDraftSubmitting(false);
      }
    },
    [draftComponent, draftPreview, load],
  );

  const importCandidate = useCallback(async () => {
    if (!draftComponent) return;
    const component = draftComponent;
    setImportingComponent(component);
    setDraftError(undefined);
    try {
      await importEnterpriseModelAsset(
        component,
        globalThis.crypto?.randomUUID?.() ?? `enterprise-import-${Date.now()}`,
      );
      await Promise.all([load(true), openDraftWizard(component)]);
    } catch (cause) {
      setDraftError(cause);
    } finally {
      setImportingComponent(undefined);
    }
  }, [draftComponent, load, openDraftWizard]);

  const createStagingBaseline = useCallback(async () => {
    setBaselineCreating(true);
    setDraftError(undefined);
    try {
      const result = await createEnterpriseStagingBaselineDraft(
        globalThis.crypto?.randomUUID?.() ?? `enterprise-baseline-${Date.now()}`,
      );
      setCreatedBaselineReleaseId(result.baseline.release_id);
      setDraftOpen(false);
      await load(true);
    } catch (cause) {
      setDraftError(cause);
    } finally {
      setBaselineCreating(false);
    }
  }, [load]);

  const createCandidateReleaseBatch = useCallback(async () => {
    setBatchCreating(true);
    setBatchError(undefined);
    batchIdempotencyKey.current ??=
      globalThis.crypto?.randomUUID?.() ?? `enterprise-release-batch-${Date.now()}`;
    try {
      const result = await createEnterpriseCandidateReleaseBatch(
        batchIdempotencyKey.current,
      );
      setCreatedBatch(result.batch);
      batchIdempotencyKey.current = undefined;
      await load(true);
    } catch (cause) {
      setBatchError(cause);
    } finally {
      setBatchCreating(false);
    }
  }, [load]);

  const advanceCandidateReleaseBatch = useCallback(
    async (batchKeySha256: string) => {
      setAdvancingBatch(batchKeySha256);
      setBatchAdvanceError(undefined);
      try {
        const result = await advanceEnterpriseCandidateReleaseBatchToApproval(
          batchKeySha256,
        );
        setSubmittedBatch(result.batch);
        await load(true);
      } catch (cause) {
        setBatchAdvanceError(cause);
      } finally {
        setAdvancingBatch(undefined);
      }
    },
    [load],
  );

  return (
    <AppShell>
      <div className="page-stack">
        <Space align="start" style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>企业项目权威资产治理</Typography.Title>
            <Typography.Paragraph type="secondary">
              查看本项目已采用的企业系统、数据、训练候选和本地部署证据。来源标记用于保留血缘，项目运行资格以经过复核的采用清单为准。
            </Typography.Paragraph>
          </div>
          <Button loading={refreshing} onClick={() => void load(true)}>
            重新复核
          </Button>
        </Space>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {createdReleaseId ? (
          <Alert
            showIcon
            closable
            type="success"
            title="ModelRelease 草稿已创建"
            description="草稿仍需校验和独立审批；只有审批通过后，部署控制器才会自动创建真实 Shadow 请求。"
            action={
              <Button
                href={
                  "/ai/releases?release_id=" + encodeURIComponent(createdReleaseId)
                }
              >
                打开发布草稿
              </Button>
            }
            onClose={() => setCreatedReleaseId(undefined)}
          />
        ) : null}
        {createdBaselineReleaseId ? (
          <Alert
            showIcon
            closable
            type="success"
            title="首个企业 Staging 基线草稿已创建"
            description="请在发布中心完成校验、提交审批，并由不同主体独立批准。批准后返回本页，即可为 LLM、VLM、ASR、TTS、RUL、Embedding 或 Reranker 创建候选 Release 草稿。"
            action={
              <Button
                href={
                  "/ai/releases?release_id=" +
                  encodeURIComponent(createdBaselineReleaseId)
                }
              >
                打开基线草稿
              </Button>
            }
            onClose={() => setCreatedBaselineReleaseId(undefined)}
          />
        ) : null}
        {createdBatch ? (
          <Alert
            showIcon
            closable
            type="success"
            title="LLM/VLM/RUL Release 草稿已批量创建"
            description={
              <Space orientation="vertical" size={2}>
                <Typography.Text>
                  共同基线 {createdBatch.baseline_release_id}；新建 {createdBatch.created_count}
                  个，幂等回放 {createdBatch.replayed_count} 个。三个草稿仍需分别校验、提交并由独立审批人批准。
                </Typography.Text>
                {createdBatch.releases.map((release) => (
                  <Typography.Text key={release.release_id} copyable>
                    {release.component} · {release.release_id}
                  </Typography.Text>
                ))}
              </Space>
            }
            action={
              <Button
                href={
                  "/ai/releases?release_id=" +
                  encodeURIComponent(createdBatch.releases[0].release_id)
                }
              >
                打开首个草稿
              </Button>
            }
            onClose={() => setCreatedBatch(undefined)}
          />
        ) : null}
        {submittedBatch ? (
          <Alert
            showIcon
            closable
            type="success"
            title="三个候选 Release 已进入独立审批"
            description={`当前 ${submittedBatch.approval_pending_count} 个待审批、${submittedBatch.approved_count} 个已批准。系统已停止在审批门禁，没有自动批准或启动 Shadow。`}
            action={
              <Button
                href={
                  "/ai/releases?release_id=" +
                  encodeURIComponent(submittedBatch.components[0].release_id)
                }
              >
                打开首个审批
              </Button>
            }
            onClose={() => setSubmittedBatch(undefined)}
          />
        ) : null}
        {!adoption && !error ? (
          <LoadingState label="正在复核企业项目采用清单与来源证据" />
        ) : null}

        {adoption ? (
          <>
            <Alert
              showIcon
              type={adoption.ready_for_enterprise_project_use ? "success" : "warning"}
              title="企业项目权威 Staging 已采用"
              description="SIMULATED_NON_PRODUCTION 仅保留项目生成来源血缘，不再否定项目内使用。失败或无收益实验只进入审计历史，不具备运行资格。"
            />

            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="活动权威资产" value={counts?.activeAssets ?? 0} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="可运行资产" value={counts?.runtimeAssets ?? 0} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="实验审计历史" value={counts?.experimentHistory ?? 0} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="覆盖门禁通过" value={coverage.length} suffix="项" /></Card>
              </Col>
            </Row>

            <Card title="采用身份与策略">
              <Descriptions bordered size="small" column={{ xs: 1, md: 2, xl: 3 }}>
                <Descriptions.Item label="运行分类">
                  <Tag color="green">{adoption.classification}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="状态">
                  <Tag color="blue">{adoption.status}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="企业范围">{adoption.enterprise_scope}</Descriptions.Item>
                <Descriptions.Item label="项目企业事实">
                  {adoption.project_enterprise_truth ? "已确认" : "未确认"}
                </Descriptions.Item>
                <Descriptions.Item label="项目内可用">
                  {adoption.ready_for_enterprise_project_use ? "是" : "否"}
                </Descriptions.Item>
                <Descriptions.Item label="外部生产声明">
                  {adoption.production_claim ? "已声明" : "未声明"}
                </Descriptions.Item>
                <Descriptions.Item label="来源标记解释">
                  {adoption.policy.source_marker_interpretation}
                </Descriptions.Item>
                <Descriptions.Item label="失败实验策略">
                  {adoption.policy.failed_experiment_policy}
                </Descriptions.Item>
                <Descriptions.Item label="采用时间">
                  {new Date(adoption.adopted_at).toLocaleString("zh-CN")}
                </Descriptions.Item>
                <Descriptions.Item label="证据链" span={3}>
                  <Typography.Text copyable={{ text: adoption.evidence_chain_sha256 }}>
                    {adoption.evidence_chain_sha256}
                  </Typography.Text>
                </Descriptions.Item>
              </Descriptions>
            </Card>

            <Tabs
              items={[
                {
                  key: "active",
                  label: "活动企业资产",
                  children: (
                    <Card>
                      <Space wrap style={{ marginBottom: 16 }}>
                        <Input.Search
                          allowClear
                          placeholder="搜索领域、角色、能力或来源路径"
                          style={{ width: 360 }}
                          value={search}
                          onChange={(event) => setSearch(event.target.value)}
                        />
                        <Select
                          value={domain}
                          style={{ width: 260 }}
                          onChange={setDomain}
                          options={[
                            { value: "ALL", label: "全部领域" },
                            ...domains.map((value) => ({ value, label: value })),
                          ]}
                        />
                        <Select<RuntimeFilter>
                          value={runtimeFilter}
                          style={{ width: 180 }}
                          onChange={setRuntimeFilter}
                          options={[
                            { value: "ALL", label: "全部资格" },
                            { value: "RUNTIME", label: "可运行" },
                            { value: "EVIDENCE", label: "仅证据" },
                          ]}
                        />
                        <Typography.Text type="secondary">
                          当前显示 {filteredAssets.length} 项
                        </Typography.Text>
                      </Space>
                      <Table<EnterpriseProjectAsset>
                        rowKey="source_path"
                        dataSource={filteredAssets}
                        pagination={{ pageSize: 10, hideOnSinglePage: true }}
                        scroll={{ x: 1500 }}
                        locale={{ emptyText: "没有符合筛选条件的企业资产" }}
                        columns={[
                          {
                            title: "领域 / 类型",
                            fixed: "left",
                            width: 270,
                            render: (_, asset) => (
                              <Space orientation="vertical" size={2}>
                                <Typography.Text strong>{asset.domain}</Typography.Text>
                                <Typography.Text type="secondary">{asset.asset_kind}</Typography.Text>
                              </Space>
                            ),
                          },
                          {
                            title: "项目角色",
                            dataIndex: "role",
                            width: 330,
                            render: (value: string) => <Tag color="blue">{value}</Tag>,
                          },
                          {
                            title: "运行资格",
                            dataIndex: "runtime_eligible",
                            width: 110,
                            render: (value: boolean) => (
                              <Tag color={value ? "green" : "default"}>
                                {value ? "可运行" : "仅证据"}
                              </Tag>
                            ),
                          },
                          {
                            title: "来源状态",
                            width: 260,
                            render: (_, asset) => (
                              <Space orientation="vertical" size={2}>
                                <Tag>{asset.source_status}</Tag>
                                <Typography.Text type="secondary">
                                  {asset.source_classification}
                                </Typography.Text>
                              </Space>
                            ),
                          },
                          {
                            title: "能力",
                            width: 420,
                            render: (_, asset) => (
                              <Space wrap>
                                {asset.capabilities.map((capability) => (
                                  <Tag key={capability}>{capability}</Tag>
                                ))}
                              </Space>
                            ),
                          },
                          {
                            title: "来源路径",
                            dataIndex: "source_path",
                            width: 390,
                            render: (value: string) => (
                              <Typography.Text copyable={{ text: value }}>{value}</Typography.Text>
                            ),
                          },
                        ]}
                      />
                    </Card>
                  ),
                },
                {
                  key: "project-assurance",
                  label: "M6 联合验收",
                  children: projectAssurance ? (
                    <Card>
                      <Alert
                        showIcon
                        type="success"
                        title="安全、恢复、SLO、Canary 与三方签署已形成项目级闭环"
                        description="该回执由单进程低资源实验室调用现有正式状态机生成，属于项目企业 Staging 权威证据；未启动 Docker、Kubernetes 或 GPU，也不冒充外部企业生产观察。"
                        style={{ marginBottom: 16 }}
                      />
                      <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
                        <Col xs={12} lg={4}>
                          <Statistic
                            title="安全场景"
                            value={projectAssuranceCounts?.securityScenarios ?? 0}
                            suffix="项"
                          />
                        </Col>
                        <Col xs={12} lg={4}>
                          <Statistic
                            title="恢复组件"
                            value={projectAssuranceCounts?.recoveryComponents ?? 0}
                            suffix="类"
                          />
                        </Col>
                        <Col xs={12} lg={4}>
                          <Statistic
                            title="健康 SLO"
                            value={projectAssuranceCounts?.healthySlos ?? 0}
                            suffix="项"
                          />
                        </Col>
                        <Col xs={12} lg={4}>
                          <Statistic
                            title="动态 Staging"
                            value={projectAssuranceCounts?.stagingScenarios ?? 0}
                            suffix="项"
                          />
                        </Col>
                        <Col xs={12} lg={4}>
                          <Statistic
                            title="独立签署"
                            value={projectAssuranceCounts?.signoffs ?? 0}
                            suffix="方"
                          />
                        </Col>
                      </Row>
                      <Descriptions
                        bordered
                        size="small"
                        column={{ xs: 1, md: 2, xl: 3 }}
                        style={{ marginBottom: 16 }}
                      >
                        <Descriptions.Item label="验收状态">
                          <Tag color="green">{projectAssurance.status}</Tag>
                        </Descriptions.Item>
                        <Descriptions.Item label="执行方式">
                          <Tag color="cyan">{projectAssurance.execution_mode}</Tag>
                        </Descriptions.Item>
                        <Descriptions.Item label="项目企业 Staging">
                          {projectAssurance.ready_for_project_enterprise_staging ? "已授权" : "否"}
                        </Descriptions.Item>
                        <Descriptions.Item label="外部生产声明">
                          {projectAssurance.external_enterprise_production_claim ? "是" : "否"}
                        </Descriptions.Item>
                        <Descriptions.Item label="后台服务启动数">
                          {projectAssurance.cleanup.background_services_started.length}
                        </Descriptions.Item>
                        <Descriptions.Item label="生成时间">
                          {new Date(projectAssurance.generated_at).toLocaleString("zh-CN")}
                        </Descriptions.Item>
                        <Descriptions.Item label="联合验收证据链" span={3}>
                          <Typography.Text
                            copyable={{ text: projectAssurance.evidence_chain_sha256 }}
                          >
                            {projectAssurance.evidence_chain_sha256}
                          </Typography.Text>
                        </Descriptions.Item>
                      </Descriptions>
                      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                        <Col xs={24} xl={12}>
                          <Card type="inner" title="恢复、发布与签署门禁">
                            <Descriptions bordered size="small" column={1}>
                              <Descriptions.Item label="恢复发布门禁">
                                <Tag color="green">
                                  {projectAssurance.recovery.release_allowed
                                    ? "ALLOWED"
                                    : "BLOCKED"}
                                </Tag>
                              </Descriptions.Item>
                              <Descriptions.Item label="月度 PostgreSQL 演练">
                                {projectAssurance.recovery.monthly_postgres.status}
                              </Descriptions.Item>
                              <Descriptions.Item label="季度跨组件演练">
                                {projectAssurance.recovery.quarterly_cross_component.status}
                              </Descriptions.Item>
                              <Descriptions.Item label="Canary 状态">
                                <Space>
                                  <Tag color="blue">{projectAssurance.release.current_stage}</Tag>
                                  <Tag color="green">
                                    {projectAssurance.release.canary_decision}
                                  </Tag>
                                  <Typography.Text>
                                    {projectAssurance.release.observed_traffic_percent}%
                                  </Typography.Text>
                                </Space>
                              </Descriptions.Item>
                              <Descriptions.Item label="联合签署">
                                <Space wrap>
                                  {projectAssurance.signoff.signoffs.map((signoff) => (
                                    <Tag color="purple" key={signoff.role}>
                                      {signoff.role}
                                    </Tag>
                                  ))}
                                </Space>
                              </Descriptions.Item>
                              <Descriptions.Item label="恢复组件">
                                <Space wrap>
                                  {projectAssurance.recovery.components.map((component) => (
                                    <Tag color="green" key={component.component}>
                                      {component.component}: {component.evidence_status}
                                    </Tag>
                                  ))}
                                </Space>
                              </Descriptions.Item>
                            </Descriptions>
                          </Card>
                        </Col>
                        <Col xs={24} xl={12}>
                          <Card type="inner" title="仍需外部生产环境提供的条件">
                            <Space orientation="vertical" style={{ width: "100%" }}>
                              {projectAssurance.remaining_external_production_requirements.map(
                                (requirement) => (
                                  <Alert
                                    key={requirement}
                                    type="info"
                                    showIcon
                                    title={requirement}
                                  />
                                ),
                              )}
                            </Space>
                          </Card>
                        </Col>
                      </Row>
                      <Typography.Title level={5}>安全合同探测</Typography.Title>
                      <Table<ProjectAssuranceSecurityScenario>
                        rowKey="scenario"
                        dataSource={projectAssurance.security.scenarios}
                        pagination={{ pageSize: 6, hideOnSinglePage: true }}
                        scroll={{ x: 1050 }}
                        columns={[
                          {
                            title: "场景",
                            dataIndex: "scenario",
                            width: 260,
                            render: (value: string) => (
                              <Typography.Text strong>{value}</Typography.Text>
                            ),
                          },
                          {
                            title: "类别",
                            dataIndex: "category",
                            width: 180,
                          },
                          {
                            title: "实际结果",
                            dataIndex: "observed_outcome",
                            width: 170,
                            render: (value: string) => <Tag color="green">{value}</Tag>,
                          },
                          {
                            title: "探测实现",
                            dataIndex: "probe_method",
                            width: 280,
                          },
                          {
                            title: "尝试次数",
                            dataIndex: "attempt_count",
                            width: 120,
                          },
                        ]}
                      />
                      <Typography.Title level={5} style={{ marginTop: 16 }}>
                        SLO 观测
                      </Typography.Title>
                      <Table<ProjectAssuranceSlo>
                        rowKey="slo_id"
                        dataSource={projectAssurance.operations.observations}
                        pagination={false}
                        scroll={{ x: 900 }}
                        columns={[
                          { title: "SLO", dataIndex: "slo_id", width: 300 },
                          {
                            title: "状态",
                            dataIndex: "status",
                            width: 140,
                            render: (value: string) => <Tag color="green">{value}</Tag>,
                          },
                          { title: "当前值", dataIndex: "current_value", width: 140 },
                          { title: "目标", dataIndex: "objective", width: 140 },
                          { title: "Runbook", dataIndex: "runbook_id", width: 220 },
                        ]}
                      />
                      {projectAssuranceRequestId ? (
                        <Typography.Text className="request-id" type="secondary">
                          M6 联合验收请求标识：{projectAssuranceRequestId}
                        </Typography.Text>
                      ) : null}
                    </Card>
                  ) : (
                    <LoadingState label="正在复核 M6 联合验收证据" />
                  ),
                },
                {
                  key: "project-closure",
                  label: "项目闭环与生产差距",
                  children: projectClosure ? (
                    <Card>
                      <Alert
                        showIcon
                        type={
                          projectClosure.intentionally_unverified_methods.length === 0
                            ? "success"
                            : "info"
                        }
                        title="项目企业 Staging 功能闭环已通过，外部生产仍需环境验收"
                        description={
                          projectClosure.intentionally_unverified_methods.length === 0
                            ? "DPO、TTS、Embedding 与 Reranker 的计划内企业价值评测均已通过，候选当前只具备 ModelRelease 草稿资格；外部生产仍需目标企业身份、基础设施、观察窗口和责任人签署。"
                            : "已通过项来自采用清单绑定的不可变总闭环回执。待验证方法表示当前候选被拒绝或研究范围受限，不等于代码未实现；生产事项需要目标企业身份、基础设施、观察窗口和责任人签署。"
                        }
                        style={{ marginBottom: 16 }}
                      />
                      <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
                        <Col xs={12} lg={6}>
                          <Statistic
                            title="已通过能力域"
                            value={projectClosureCounts?.coverageDomains ?? 0}
                            suffix="项"
                          />
                        </Col>
                        <Col xs={12} lg={6}>
                          <Statistic
                            title="权威来源证据"
                            value={projectClosureCounts?.sourceEvidence ?? 0}
                            suffix="份"
                          />
                        </Col>
                        <Col xs={12} lg={6}>
                          <Statistic
                            title="未闭环模型方法"
                            value={projectClosureCounts?.intentionallyUnverifiedMethods ?? 0}
                            suffix="项"
                          />
                        </Col>
                        <Col xs={12} lg={6}>
                          <Statistic
                            title="外部生产事项"
                            value={projectClosureCounts?.productionBlockers ?? 0}
                            suffix="项"
                          />
                        </Col>
                      </Row>
                      <Descriptions
                        bordered
                        size="small"
                        column={{ xs: 1, md: 2, xl: 3 }}
                        style={{ marginBottom: 16 }}
                      >
                        <Descriptions.Item label="闭环状态">
                          <Tag color="green">{projectClosure.status}</Tag>
                        </Descriptions.Item>
                        <Descriptions.Item label="项目演示可用">
                          {projectClosure.ready_for_simulated_product_demo ? "是" : "否"}
                        </Descriptions.Item>
                        <Descriptions.Item label="外部生产就绪">
                          {projectClosure.ready_for_production ? "是" : "否"}
                        </Descriptions.Item>
                        <Descriptions.Item label="生成时间">
                          {new Date(projectClosure.generated_at).toLocaleString("zh-CN")}
                        </Descriptions.Item>
                        <Descriptions.Item label="运行分类">
                          {projectClosure.classification}
                        </Descriptions.Item>
                        <Descriptions.Item label="企业生产数据">
                          {projectClosure.enterprise_production_data ? "已接入" : "未接入"}
                        </Descriptions.Item>
                        <Descriptions.Item label="总闭环证据链" span={3}>
                          <Typography.Text
                            copyable={{ text: projectClosure.evidence_chain_sha256 }}
                          >
                            {projectClosure.evidence_chain_sha256}
                          </Typography.Text>
                        </Descriptions.Item>
                      </Descriptions>
                      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                        <Col xs={24} xl={12}>
                          <Card type="inner" title="模型企业价值评测状态">
                            <Space orientation="vertical" style={{ width: "100%" }}>
                              {projectClosure.intentionally_unverified_methods.length === 0 ? (
                                <Alert
                                  type="success"
                                  showIcon
                                  title="计划内模型企业价值评测已全部闭环"
                                  description="DPO、TTS、Embedding、Reranker 均已形成独立 Gold、正式门禁和不可变证据；发布仍须经过审批、Shadow、Canary 与回滚状态机。"
                                />
                              ) : (
                                projectClosure.intentionally_unverified_methods.map((method) => (
                                  <Alert
                                    key={method}
                                    type="warning"
                                    showIcon
                                    title={method}
                                    description={unverifiedMethodLabels[method] ?? method}
                                  />
                                ))
                              )}
                            </Space>
                          </Card>
                        </Col>
                        <Col xs={24} xl={12}>
                          <Card type="inner" title="目标企业生产环境事项">
                            <Space orientation="vertical" style={{ width: "100%" }}>
                              {projectClosure.production_blockers.map((blocker) => (
                                <Alert
                                  key={blocker}
                                  type="info"
                                  showIcon
                                  title={productionBlockerLabels[blocker] ?? blocker}
                                  description={blocker}
                                />
                              ))}
                            </Space>
                          </Card>
                        </Col>
                      </Row>
                      <Table<EnterpriseClosureEvidence>
                        rowKey="domain"
                        dataSource={projectClosure.source_evidence}
                        pagination={{ pageSize: 8, hideOnSinglePage: true }}
                        scroll={{ x: 1420 }}
                        columns={[
                          {
                            title: "能力域",
                            dataIndex: "domain",
                            fixed: "left",
                            width: 280,
                            render: (value: string) => (
                              <Space orientation="vertical" size={2}>
                                <Typography.Text strong>{value}</Typography.Text>
                                <Tag color="green">PASSED</Tag>
                              </Space>
                            ),
                          },
                          {
                            title: "来源状态",
                            width: 310,
                            render: (_, evidence) => (
                              <Space orientation="vertical" size={2}>
                                <Tag color="blue">{evidence.status}</Tag>
                                <Typography.Text type="secondary">
                                  {evidence.source_classification}
                                </Typography.Text>
                              </Space>
                            ),
                          },
                          {
                            title: "已证明能力",
                            width: 480,
                            render: (_, evidence) => (
                              <Space wrap>
                                {evidence.capabilities.map((capability) => (
                                  <Tag key={capability}>{capability}</Tag>
                                ))}
                              </Space>
                            ),
                          },
                          {
                            title: "来源路径",
                            dataIndex: "source_path",
                            width: 390,
                            render: (value: string) => (
                              <Typography.Text copyable={{ text: value }}>{value}</Typography.Text>
                            ),
                          },
                          {
                            title: "证据链",
                            dataIndex: "evidence_chain_sha256",
                            width: 260,
                            render: (value: string) => (
                              <Typography.Text copyable={{ text: value }}>
                                {value.slice(0, 24)}…
                              </Typography.Text>
                            ),
                          },
                        ]}
                      />
                      {projectClosureRequestId ? (
                        <Typography.Text className="request-id" type="secondary">
                          项目闭环请求标识：{projectClosureRequestId}
                        </Typography.Text>
                      ) : null}
                    </Card>
                  ) : (
                    <LoadingState label="正在复核项目总闭环与生产差距" />
                  ),
                },
                {
                  key: "model-import-acceptance",
                  label: "七组件导入验收",
                  children: modelImportAcceptance ? (
                    <Card>
                      <Alert
                        showIcon
                        type="success"
                        title="LLM/VLM/ASR/RUL/TTS/Embedding/Reranker 导入与 Release 预检已闭环"
                        description="七类项目权威候选已通过真实 FastAPI 路由导入临时租户治理台账，并共同绑定已独立批准的 Staging 基线。该回执证明导入与草稿预检，不声明 Production、Shadow 或 Canary。"
                        style={{ marginBottom: 16 }}
                      />
                      <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
                        <Col xs={12} lg={6}>
                          <Statistic
                            title="验收组件"
                            value={modelImportAcceptanceCounts?.components ?? 0}
                            suffix="/ 7"
                          />
                        </Col>
                        <Col xs={12} lg={6}>
                          <Statistic
                            title="预检可创建草稿"
                            value={modelImportAcceptanceCounts?.eligibleComponents ?? 0}
                            suffix="项"
                          />
                        </Col>
                        <Col xs={12} lg={6}>
                          <Statistic
                            title="实际评测样本"
                            value={modelImportAcceptanceCounts?.totalActualSamples ?? 0}
                            suffix="条"
                          />
                        </Col>
                        <Col xs={12} lg={6}>
                          <Statistic
                            title="共同兼容基线"
                            value={
                              modelImportAcceptanceCounts?.commonBaselineReleaseIds.length ?? 0
                            }
                            suffix="个"
                          />
                        </Col>
                      </Row>
                      <Descriptions
                        bordered
                        size="small"
                        column={{ xs: 1, md: 2, xl: 3 }}
                        style={{ marginBottom: 16 }}
                      >
                        <Descriptions.Item label="执行方式">
                          <Tag color="cyan">{modelImportAcceptance.execution_mode}</Tag>
                        </Descriptions.Item>
                        <Descriptions.Item label="发布范围">
                          <Tag color="blue">{modelImportAcceptance.release_scope}</Tag>
                        </Descriptions.Item>
                        <Descriptions.Item label="验收状态">
                          <Tag color="green">{modelImportAcceptance.status}</Tag>
                        </Descriptions.Item>
                        <Descriptions.Item label="运行绑定复核">
                          {modelImportAcceptance.runtime_binding_verified ? "通过" : "未通过"}
                        </Descriptions.Item>
                        <Descriptions.Item label="Release 预检">
                          {modelImportAcceptance.release_draft_import_preflight_verified
                            ? "通过"
                            : "未通过"}
                        </Descriptions.Item>
                        <Descriptions.Item label="外部生产声明">
                          {modelImportAcceptance.external_enterprise_production_claim
                            ? "已声明"
                            : "未声明"}
                        </Descriptions.Item>
                        <Descriptions.Item label="共同兼容基线" span={3}>
                          <Space wrap>
                            {modelImportAcceptanceCounts?.commonBaselineReleaseIds.map(
                              (baseline) => (
                                <Typography.Text key={baseline} copyable>
                                  {baseline}
                                </Typography.Text>
                              ),
                            )}
                          </Space>
                        </Descriptions.Item>
                        <Descriptions.Item label="验收证据链" span={3}>
                          <Typography.Text
                            copyable={{ text: modelImportAcceptance.evidence_chain_sha256 }}
                          >
                            {modelImportAcceptance.evidence_chain_sha256}
                          </Typography.Text>
                        </Descriptions.Item>
                      </Descriptions>
                      <Table
                        rowKey="component"
                        dataSource={modelImportAcceptance.components}
                        pagination={false}
                        scroll={{ x: 1640 }}
                        columns={[
                          {
                            title: "组件",
                            dataIndex: "component",
                            fixed: "left",
                            width: 90,
                            render: (value: string) => <Tag color="purple">{value}</Tag>,
                          },
                          {
                            title: "来源 / 租户候选",
                            width: 390,
                            render: (_, component) => (
                              <Space orientation="vertical" size={2}>
                                <Typography.Text copyable>
                                  来源 {component.source_candidate_experiment_id}
                                </Typography.Text>
                                <Typography.Text copyable type="secondary">
                                  租户 {component.imported_candidate_experiment_id}
                                </Typography.Text>
                              </Space>
                            ),
                          },
                          {
                            title: "导入证明",
                            width: 350,
                            render: (_, component) => (
                              <Space orientation="vertical" size={2}>
                                <Typography.Text copyable>{component.import_id}</Typography.Text>
                                <Typography.Text
                                  copyable={{ text: component.import_hash }}
                                  type="secondary"
                                >
                                  {component.import_hash.slice(0, 24)}…
                                </Typography.Text>
                              </Space>
                            ),
                          },
                          {
                            title: "评测 / 样本",
                            width: 390,
                            render: (_, component) => (
                              <Space orientation="vertical" size={2}>
                                <Typography.Text copyable>
                                  {component.imported_evaluation_id}
                                </Typography.Text>
                                <Typography.Text type="secondary">
                                  {component.imported_suite_id} · {component.actual_sample_count} 条
                                </Typography.Text>
                              </Space>
                            ),
                          },
                          {
                            title: "Registry / 预检",
                            width: 230,
                            render: (_, component) => (
                              <Space wrap>
                                <Tag>{component.registry_state}</Tag>
                                <Tag color={component.release_draft_eligible ? "green" : "red"}>
                                  {component.release_draft_eligible
                                    ? "草稿预检通过"
                                    : "草稿预检阻断"}
                                </Tag>
                                {component.release_draft_blockers.map((blocker) => (
                                  <Tag key={blocker} color="red">
                                    {blocker}
                                  </Tag>
                                ))}
                              </Space>
                            ),
                          },
                          {
                            title: "兼容基线",
                            width: 330,
                            render: (_, component) => (
                              <Space orientation="vertical" size={2}>
                                {component.compatible_baseline_release_ids.map((baseline) => (
                                  <Typography.Text key={baseline} copyable>
                                    {baseline}
                                  </Typography.Text>
                                ))}
                              </Space>
                            ),
                          },
                        ]}
                      />
                      {modelImportAcceptanceRequestId ? (
                        <Typography.Text className="request-id" type="secondary">
                          导入验收请求标识：{modelImportAcceptanceRequestId}
                        </Typography.Text>
                      ) : null}
                    </Card>
                  ) : (
                    <LoadingState label="正在复核七组件导入验收回执" />
                  ),
                },
                {
                  key: "runtime",
                  label: "模型运行联动",
                  children: (
                    <Card>
                      <Alert
                        showIcon
                        type="info"
                        title="项目证据与当前租户发布状态独立呈现"
                        description="左侧阶段表示项目已验证的本地证据；Registry、Deployment 和 Alias 来自当前租户实时数据库。没有正式登记时显示 NOT_REGISTERED，不会把历史本地 KServe 验收冒充在线发布。"
                        style={{ marginBottom: 16 }}
                      />
                      {candidateBatchReady ? (
                        <Alert
                          showIcon
                          type="success"
                          title="三个项目权威候选均已导入，可批量登记"
                          description="系统将先确认共同的已批准 Staging 基线，再创建三个相互独立的 DRAFT；中途失败可使用同一幂等键安全续跑。"
                          action={
                            <Button
                              type="primary"
                              loading={batchCreating}
                              onClick={() => void createCandidateReleaseBatch()}
                            >
                              批量创建 LLM/VLM/RUL 草稿
                            </Button>
                          }
                          style={{ marginBottom: 16 }}
                        />
                      ) : null}
                      {batchError ? (
                        <ErrorState
                          error={batchError}
                          onRetry={() => void createCandidateReleaseBatch()}
                        />
                      ) : null}
                      <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
                        <Col xs={12} lg={6}>
                          <Statistic title="模型组件" value={runtimeCounts?.components ?? 0} />
                        </Col>
                        <Col xs={12} lg={6}>
                          <Statistic
                            title="已登记 Registry"
                            value={runtimeCounts?.registeredComponents ?? 0}
                          />
                        </Col>
                        <Col xs={12} lg={6}>
                          <Statistic
                            title="已有部署记录"
                            value={runtimeCounts?.deployedComponents ?? 0}
                          />
                        </Col>
                        <Col xs={12} lg={6}>
                          <Statistic
                            title="活动 Alias"
                            value={runtimeCounts?.activeAliasComponents ?? 0}
                          />
                        </Col>
                      </Row>
                      <Table<EnterpriseRuntimeBinding>
                        rowKey={(binding) => binding.evidence.component}
                        dataSource={runtimeBindings}
                        pagination={false}
                        scroll={{ x: 1320 }}
                        expandable={{
                          expandedRowRender: (binding) =>
                            binding.registrations.length === 0 ? (
                              <Alert
                                type="warning"
                                showIcon
                                title="该权威候选尚未登记到当前租户 Model Registry"
                                description="项目内运行资格和本地验收证据仍然有效，但必须创建正式 ModelRelease 后才能进入统一 Shadow、Canary、Production 与回滚状态机。"
                              />
                            ) : (
                              <Table<EnterpriseRuntimeRegistration>
                                rowKey="release_id"
                                size="small"
                                pagination={false}
                                dataSource={binding.registrations}
                                columns={[
                                  {
                                    title: "Release / 环境",
                                    render: (_, registration) => (
                                      <Space orientation="vertical" size={2}>
                                        <Typography.Text copyable>
                                          {registration.release_id}
                                        </Typography.Text>
                                        <Tag>{registration.target_environment}</Tag>
                                      </Space>
                                    ),
                                  },
                                  {
                                    title: "发布状态",
                                    render: (_, registration) => (
                                      <Space orientation="vertical" size={2}>
                                        <Tag color="blue">{registration.status}</Tag>
                                        <Typography.Text type="secondary">
                                          流量 {registration.traffic_percent}%
                                        </Typography.Text>
                                      </Space>
                                    ),
                                  },
                                  {
                                    title: "KServe / 部署",
                                    render: (_, registration) =>
                                      registration.deployment ? (
                                        <Space orientation="vertical" size={2}>
                                          <Tag color="geekblue">
                                            {registration.deployment.current_stage}
                                          </Tag>
                                          <Typography.Text type="secondary">
                                            {registration.deployment.status} · 观测流量{" "}
                                            {registration.deployment.observed_traffic_percent}%
                                          </Typography.Text>
                                        </Space>
                                      ) : (
                                        <Tag>NOT_DEPLOYED</Tag>
                                      ),
                                  },
                                  {
                                    title: "活动别名",
                                    render: (_, registration) =>
                                      registration.aliases.length ? (
                                        <Space wrap>
                                          {registration.aliases.map((alias) => (
                                            <Tag
                                              key={alias.alias}
                                              color={alias.status === "ACTIVE" ? "green" : "default"}
                                            >
                                              {alias.alias} · {alias.status}
                                            </Tag>
                                          ))}
                                        </Space>
                                      ) : (
                                        <Tag>NO_ACTIVE_ALIAS</Tag>
                                      ),
                                  },
                                  {
                                    title: "自动 Shadow",
                                    render: (_, registration) =>
                                      registration.automation ? (
                                        <Space orientation="vertical" size={2}>
                                          <Tag
                                            color={
                                              registration.automation.automation_status ===
                                              "SHADOW_REQUESTED"
                                                ? "green"
                                                : registration.automation.automation_status ===
                                                    "BLOCKED"
                                                  ? "red"
                                                  : "gold"
                                            }
                                          >
                                            {registration.automation.automation_status}
                                          </Tag>
                                          <Typography.Text type="secondary">
                                            尝试 {registration.automation.attempt_count} 次
                                          </Typography.Text>
                                          {registration.automation.last_error ? (
                                            <Typography.Text type="danger">
                                              {registration.automation.last_error}
                                            </Typography.Text>
                                          ) : null}
                                        </Space>
                                      ) : (
                                        <Tag>未配置</Tag>
                                      ),
                                  },
                                  {
                                    title: "回滚目标",
                                    dataIndex: "rollback_release_id",
                                    render: (value: string | null) => value ?? "—",
                                  },
                                  {
                                    title: "操作",
                                    width: 120,
                                    render: (_, registration) => (
                                      <Button
                                        type="link"
                                        href={
                                          "/ai/releases?release_id=" +
                                          encodeURIComponent(registration.release_id)
                                        }
                                      >
                                        发布详情
                                      </Button>
                                    ),
                                  },
                                ]}
                              />
                            ),
                        }}
                        columns={[
                          {
                            title: "组件 / 候选",
                            fixed: "left",
                            width: 360,
                            render: (_, binding) => (
                              <Space orientation="vertical" size={2}>
                                <Tag color="purple">{binding.evidence.component}</Tag>
                                <Typography.Text copyable>
                                  {binding.evidence.candidate_experiment_id}
                                </Typography.Text>
                                <Typography.Text type="secondary">
                                  {binding.evidence.evaluation_reference ?? "无独立评测引用"}
                                </Typography.Text>
                                {binding.evidence.rollout_model_release_id ? (
                                  <Typography.Text type="secondary" copyable>
                                    验收 Release {binding.evidence.rollout_model_release_id}
                                  </Typography.Text>
                                ) : null}
                              </Space>
                            ),
                          },
                          {
                            title: "项目证据阶段",
                            width: 300,
                            render: (_, binding) => (
                              <Space orientation="vertical" size={4}>
                                <Tag color="cyan">{binding.evidence.evidence_stage}</Tag>
                                {binding.evidence.rollout_model_release_id ? (
                                  <Tag color="geekblue">实际发布回执</Tag>
                                ) : null}
                                <Space wrap>
                                  <Tag color={binding.evidence.shadow_verified ? "green" : "default"}>
                                    Shadow {binding.evidence.shadow_verified ? "通过" : "待执行"}
                                  </Tag>
                                  <Tag color={binding.evidence.canary_verified ? "green" : "default"}>
                                    Canary {binding.evidence.canary_verified ? "通过" : "待执行"}
                                  </Tag>
                                  <Tag color={binding.evidence.rollback_verified ? "green" : "default"}>
                                    回滚 {binding.evidence.rollback_verified ? "通过" : "待执行"}
                                  </Tag>
                                </Space>
                              </Space>
                            ),
                          },
                          {
                            title: "当前租户治理状态",
                            width: 290,
                            render: (_, binding) => (
                              <Space wrap>
                                <Tag color={binding.registry_state === "REGISTERED" ? "green" : "orange"}>
                                  {binding.registry_state}
                                </Tag>
                                <Tag color={binding.import_state === "IMPORTED" ? "cyan" : "gold"}>
                                  {binding.import_state}
                                </Tag>
                                <Tag color={binding.deployment_state === "DEPLOYMENT_RECORDED" ? "blue" : "default"}>
                                  {binding.deployment_state}
                                </Tag>
                                <Tag color={binding.alias_state === "ACTIVE_ALIAS_BOUND" ? "purple" : "default"}>
                                  {binding.alias_state}
                                </Tag>
                              </Space>
                            ),
                          },
                          {
                            title: "登记数量",
                            width: 110,
                            render: (_, binding) => binding.registrations.length,
                          },
                          {
                            title: "最新发布",
                            width: 120,
                            render: (_, binding) => {
                              const latest = binding.registrations[0];
                              return latest ? (
                                <Button
                                  type="link"
                                  href={
                                    "/ai/releases?release_id=" +
                                    encodeURIComponent(latest.release_id)
                                  }
                                >
                                  发布详情
                                </Button>
                              ) : (
                                <Button
                                  size="small"
                                  type="primary"
                                  onClick={() =>
                                    void openDraftWizard(binding.evidence.component)
                                  }
                                >
                                  {binding.import_state === "IMPORTED"
                                    ? "创建 Release 草稿"
                                    : "导入 / 创建草稿"}
                                </Button>
                              );
                            },
                          },
                          {
                            title: "证据链",
                            width: 260,
                            render: (_, binding) => (
                              <Space orientation="vertical" size={2}>
                                <Typography.Text
                                  copyable={{ text: binding.evidence.evidence_chain_sha256 }}
                                >
                                  主链 {binding.evidence.evidence_chain_sha256.slice(0, 20)}…
                                </Typography.Text>
                                {binding.evidence.rollout_evidence_chain_sha256 ? (
                                  <Typography.Text
                                    type="secondary"
                                    copyable={{
                                      text: binding.evidence.rollout_evidence_chain_sha256,
                                    }}
                                  >
                                    发布链 {binding.evidence.rollout_evidence_chain_sha256.slice(0, 20)}…
                                  </Typography.Text>
                                ) : null}
                              </Space>
                            ),
                          },
                        ]}
                      />
                      <Card
                        type="inner"
                        title="三组件晋级进度"
                        style={{ marginTop: 16 }}
                        extra={
                          <Space>
                            <Tag color="gold">
                              待独立审批 {releaseBatchCounts.awaitingApproval}
                            </Tag>
                            <Tag color="green">
                              已全部批准 {releaseBatchCounts.approved}
                            </Tag>
                          </Space>
                        }
                      >
                        <Alert
                          showIcon
                          type="info"
                          title="批量动作只负责校验和提交"
                          description="LLM、VLM、RUL 继续保持三个独立 Release。提交完成后必须由不同主体分别审批；批准、Shadow、Canary 和回滚不会由本页自动执行。"
                          style={{ marginBottom: 16 }}
                        />
                        {batchAdvanceError ? (
                          <ErrorState error={batchAdvanceError} />
                        ) : null}
                        <Table<EnterpriseCandidateReleaseBatchProgress>
                          rowKey="batch_key_sha256"
                          dataSource={releaseBatches}
                          pagination={false}
                          locale={{ emptyText: "尚未创建三组件候选 Release 批次" }}
                          scroll={{ x: 1180 }}
                          columns={[
                            {
                              title: "批次 / 状态",
                              width: 280,
                              render: (_, batch) => (
                                <Space orientation="vertical" size={2}>
                                  <Typography.Text
                                    copyable={{ text: batch.batch_key_sha256 }}
                                  >
                                    {batch.batch_key_sha256.slice(0, 16)}…
                                  </Typography.Text>
                                  <Tag
                                    color={
                                      batch.status === "APPROVED"
                                        ? "green"
                                        : batch.status === "REMEDIATION_REQUIRED"
                                          ? "red"
                                          : batch.status ===
                                              "AWAITING_INDEPENDENT_APPROVAL"
                                            ? "gold"
                                            : "blue"
                                    }
                                  >
                                    {batch.status}
                                  </Tag>
                                </Space>
                              ),
                            },
                            {
                              title: "三个独立 Release",
                              width: 390,
                              render: (_, batch) => (
                                <Space orientation="vertical" size={4}>
                                  {batch.components.map((component) => (
                                    <Space key={component.release_id} wrap>
                                      <Tag color="purple">{component.component}</Tag>
                                      <Tag color="blue">{component.release_status}</Tag>
                                      <Tag
                                        color={
                                          component.approval_status === "APPROVED"
                                            ? "green"
                                            : component.approval_status === "PENDING"
                                              ? "gold"
                                              : "default"
                                        }
                                      >
                                        {component.approval_status ?? "未提交审批"}
                                      </Tag>
                                    </Space>
                                  ))}
                                </Space>
                              ),
                            },
                            {
                              title: "晋级计数",
                              width: 190,
                              render: (_, batch) => (
                                <Space orientation="vertical" size={2}>
                                  <Typography.Text>
                                    已登记 {batch.registered_count}/3
                                  </Typography.Text>
                                  <Typography.Text>
                                    待审批 {batch.approval_pending_count}/3
                                  </Typography.Text>
                                  <Typography.Text>
                                    已批准 {batch.approved_count}/3
                                  </Typography.Text>
                                </Space>
                              ),
                            },
                            {
                              title: "下一动作",
                              dataIndex: "next_action",
                              width: 290,
                              render: (value: string) => <Tag>{value}</Tag>,
                            },
                            {
                              title: "操作",
                              width: 210,
                              fixed: "right",
                              render: (_, batch) =>
                                batch.can_advance_to_approval ? (
                                  <Button
                                    type="primary"
                                    loading={
                                      advancingBatch === batch.batch_key_sha256
                                    }
                                    onClick={() =>
                                      void advanceCandidateReleaseBatch(
                                        batch.batch_key_sha256,
                                      )
                                    }
                                  >
                                    校验并提交三个审批
                                  </Button>
                                ) : batch.components[0] ? (
                                  <Button
                                    href={
                                      "/ai/releases?release_id=" +
                                      encodeURIComponent(
                                        batch.components[0].release_id,
                                      )
                                    }
                                  >
                                    打开发布详情
                                  </Button>
                                ) : null,
                            },
                          ]}
                        />
                      </Card>
                      {runtimeRequestId ? (
                        <Typography.Text className="request-id" type="secondary">
                          运行联动请求标识：{runtimeRequestId}
                        </Typography.Text>
                      ) : null}
                    </Card>
                  ),
                },
                {
                  key: "history",
                  label: "实验审计历史",
                  children: (
                    <Card>
                      <Alert
                        showIcon
                        type="info"
                        title="历史事实不会被删除，也不会被部署"
                        description="被拒绝或被更优候选替代的实验仍是企业项目权威审计记录，所有记录的 runtime_eligible 均为 false。"
                        style={{ marginBottom: 16 }}
                      />
                      <Table<EnterpriseProjectExperimentHistory>
                        rowKey="source_path"
                        dataSource={adoption.experiment_history}
                        pagination={{ pageSize: 10, hideOnSinglePage: true }}
                        scroll={{ x: 1200 }}
                        columns={[
                          {
                            title: "处置",
                            dataIndex: "disposition",
                            width: 330,
                            render: (value: string) => (
                              <Tag color={value.startsWith("REJECTED") ? "red" : "orange"}>
                                {value}
                              </Tag>
                            ),
                          },
                          {
                            title: "状态 / 决策",
                            width: 320,
                            render: (_, item) => (
                              <Space orientation="vertical" size={2}>
                                <Typography.Text>{item.source_status}</Typography.Text>
                                <Typography.Text type="secondary">
                                  {item.source_decision ?? "无独立决策字段"}
                                </Typography.Text>
                              </Space>
                            ),
                          },
                          {
                            title: "运行资格",
                            width: 110,
                            render: () => <Tag>不可运行</Tag>,
                          },
                          {
                            title: "来源路径",
                            dataIndex: "source_path",
                            render: (value: string) => (
                              <Typography.Text copyable={{ text: value }}>{value}</Typography.Text>
                            ),
                          },
                        ]}
                      />
                    </Card>
                  ),
                },
                {
                  key: "coverage",
                  label: "覆盖与来源",
                  children: (
                    <Row gutter={[16, 16]}>
                      <Col xs={24} xl={14}>
                        <Card title="覆盖门禁">
                          <Table<{ name: string; status: string }>
                            rowKey="name"
                            dataSource={coverage}
                            pagination={false}
                            columns={[
                              { title: "能力域", dataIndex: "name" },
                              {
                                title: "状态",
                                dataIndex: "status",
                                width: 140,
                                render: (value: string) => <Tag color="green">{value}</Tag>,
                              },
                            ]}
                          />
                        </Card>
                      </Col>
                      <Col xs={24} xl={10}>
                        <Card title="来源标记计数">
                          <Descriptions bordered size="small" column={1}>
                            {Object.entries(adoption.source_marker_counts).map(([marker, count]) => (
                              <Descriptions.Item key={marker} label={marker}>
                                {count}
                              </Descriptions.Item>
                            ))}
                          </Descriptions>
                        </Card>
                      </Col>
                    </Row>
                  ),
                },
              ]}
            />

            {requestId ? (
              <Typography.Text className="request-id" type="secondary">
                请求标识：{requestId}
              </Typography.Text>
            ) : null}
          </>
        ) : null}

        <Modal
          title={
            draftComponent
              ? `创建 ${draftComponent} ModelRelease 草稿`
              : "创建 ModelRelease 草稿"
          }
          open={draftOpen}
          width={760}
          okText="创建治理草稿"
          cancelText="取消"
          confirmLoading={draftSubmitting}
          okButtonProps={{
            disabled:
              draftLoading ||
              !draftPreview?.eligible ||
              Boolean(draftError),
          }}
          onOk={() => draftForm.submit()}
          onCancel={() => {
            setDraftOpen(false);
            setDraftError(undefined);
          }}
        >
          {draftLoading ? <LoadingState label="正在核对候选、独立评测与基线 Release" /> : null}
          {draftError ? (
            <ErrorState
              error={draftError}
              onRetry={() => {
                if (draftComponent) void openDraftWizard(draftComponent);
              }}
            />
          ) : null}
          {draftPreview && !draftLoading ? (
            <Space orientation="vertical" size={16} style={{ width: "100%" }}>
              <Alert
                showIcon
                type="info"
                title="该操作只创建 DRAFT，不绕过校验与独立审批"
                description="勾选自动 Shadow 后，审批通过只会触发部署请求；KServe 必须真实就绪并收到镜像流量，Prometheus 观测才可能判定 PASS。系统不会自动晋级 Canary。"
              />
              <Descriptions bordered size="small" column={1}>
                <Descriptions.Item label="项目来源候选">
                  <Typography.Text copyable>
                    {draftPreview.evidence.candidate_experiment_id}
                  </Typography.Text>
                </Descriptions.Item>
                <Descriptions.Item label="租户导入状态">
                  <Tag color={draftPreview.import_state === "IMPORTED" ? "green" : "gold"}>
                    {draftPreview.import_state}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="租户候选实验">
                  {draftPreview.model_import ? (
                    <Typography.Text copyable>
                      {draftPreview.model_import.imported_candidate_experiment_id}
                    </Typography.Text>
                  ) : (
                    "导入后生成租户隔离标识"
                  )}
                </Descriptions.Item>
                <Descriptions.Item label="评测样本 / 发布范围">
                  {draftPreview.model_import
                    ? `${draftPreview.model_import.actual_sample_count} 条 · ${draftPreview.model_import.release_scope}`
                    : "待导入"}
                </Descriptions.Item>
                <Descriptions.Item label="独立评测">
                  {draftPreview.evaluation_id ?? "未解析"}
                </Descriptions.Item>
                <Descriptions.Item label="供应链证据">
                  {draftPreview.suggested_supply_chain_evidence_id ?? "该组件无需单独指定"}
                </Descriptions.Item>
                <Descriptions.Item label="目标环境">
                  <Tag color="blue">{draftPreview.target_environment}</Tag>
                </Descriptions.Item>
              </Descriptions>
              {draftPreview.blockers.length ? (
                <Alert
                  showIcon
                  type="warning"
                  title="当前不能创建草稿"
                  description={
                    <ul style={{ marginBottom: 0, paddingInlineStart: 20 }}>
                      {draftPreview.blockers.map((blocker) => (
                        <li key={blocker}>{blockerLabels[blocker] ?? blocker}</li>
                      ))}
                    </ul>
                  }
                  action={
                    draftPreview.import_state === "NOT_IMPORTED" ? (
                      <Button
                        type="primary"
                        loading={importingComponent === draftComponent}
                        onClick={() => void importCandidate()}
                      >
                        导入候选证据
                      </Button>
                    ) : draftPreview.blockers.includes(
                        "COMPATIBLE_BASELINE_RELEASE_REQUIRED",
                      ) ? (
                      <Button
                        type="primary"
                        loading={baselineCreating}
                        onClick={() => void createStagingBaseline()}
                      >
                        创建首个 Staging 基线草稿
                      </Button>
                    ) : undefined
                  }
                />
              ) : null}
              <Form<ReleaseDraftFormValues>
                form={draftForm}
                layout="vertical"
                onFinish={(values) => void createDraft(values)}
              >
                <Form.Item
                  name="baseline_release_id"
                  label="已批准基线 Release"
                  rules={[{ required: true, message: "请选择用于克隆完整清单的基线 Release" }]}
                >
                  <Select
                    placeholder="选择基线 Release"
                    options={draftPreview.baselines.map((baseline) => ({
                      value: baseline.release_id,
                      label: `${baseline.status} · ${baseline.release_id}`,
                    }))}
                  />
                </Form.Item>
                <Form.Item
                  name="auto_shadow_enabled"
                  label="审批通过后自动请求真实 Shadow"
                  valuePropName="checked"
                >
                  <Switch />
                </Form.Item>
                <Typography.Title level={5}>KServe Shadow 部署计划</Typography.Title>
                <Row gutter={12}>
                  <Col span={12}>
                    <Form.Item
                      name="namespace"
                      label="命名空间"
                      rules={[{ required: true }]}
                    >
                      <Input />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="gateway_name"
                      label="Gateway"
                      rules={[{ required: true }]}
                    >
                      <Input />
                    </Form.Item>
                  </Col>
                  <Col span={24}>
                    <Form.Item
                      name="hostname"
                      label="模型域名"
                      rules={[{ required: true }]}
                    >
                      <Input />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="route_name"
                      label="候选路由"
                      rules={[{ required: true }]}
                    >
                      <Input />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="stable_service_name"
                      label="稳定服务"
                      rules={[{ required: true }]}
                    >
                      <Input />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="service_account_name"
                      label="ServiceAccount"
                      rules={[{ required: true }]}
                    >
                      <Input />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="serving_runtime_name"
                      label="ServingRuntime"
                      rules={[{ required: true }]}
                    >
                      <Input />
                    </Form.Item>
                  </Col>
                  <Col span={24}>
                    <Form.Item
                      name="artifact_uri_prefix"
                      label="模型制品 URI 前缀"
                      rules={[{ required: true }]}
                    >
                      <Input />
                    </Form.Item>
                  </Col>
                </Row>
              </Form>
            </Space>
          ) : null}
        </Modal>
      </div>
    </AppShell>
  );
}
