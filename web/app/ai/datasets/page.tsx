"use client";

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
  List,
  Modal,
  Pagination,
  Row,
  Select,
  Space,
  Statistic,
  Tabs,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { AppShell } from "@/components/AppShell";
import { AnnotationReviewHistory } from "@/components/m3/AnnotationReviewHistory";
import { CandidateGovernanceActions } from "@/components/m3/CandidateGovernanceActions";
import { CurationRunCatalog } from "@/components/m3/CurationRunCatalog";
import { CurationExclusionReport } from "@/components/m3/CurationExclusionReport";
import { DataLineageTimeline } from "@/components/m3/DataLineageTimeline";
import { DatasetArtifactCatalog } from "@/components/m3/DatasetArtifactCatalog";
import { DatasetLineageExplorer } from "@/components/m3/DatasetLineageExplorer";
import { DatasetQualityReportCard } from "@/components/m3/DatasetQualityReportCard";
import { DatasetSnapshotActions } from "@/components/m3/DatasetSnapshotActions";
import {
  DATASET_TRAINING_BLOCKER_OPTIONS,
  DatasetTrainingBlockers,
} from "@/components/m3/DatasetTrainingBlockers";
import { DlpProcessingHistory } from "@/components/m3/DlpProcessingHistory";
import { EligibilityDecisionForm } from "@/components/m3/EligibilityDecisionForm";
import { EligibilityDecisionHistory } from "@/components/m3/EligibilityDecisionHistory";
import {
  GOVERNANCE_BLOCKER_OPTIONS,
  GovernanceBlockers,
} from "@/components/m3/GovernanceBlockers";
import {
  type OperationReceipt,
  OperationReceiptAlert,
} from "@/components/m3/OperationReceiptAlert";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type DatasetWorkspaceTarget,
  buildDatasetWorkspaceUrl,
  datasetWorkspaceTargetKey,
  readDatasetWorkspaceTarget,
} from "@/lib/dataset-workspace-location";
import { buildExperimentCreateUrl } from "@/lib/experiment-workspace-location";
import {
  type CandidateEligibilityInput,
  type CurationRun,
  type CurationRunStatus,
  type DataLineage,
  type DatasetLineageStatus,
  type DatasetSnapshot,
  type DatasetSnapshotStatus,
  type DatasetTrainingBlockerCode,
  type FeedbackCandidate,
  type FeedbackCandidateDetail,
  type FeedbackCandidateQuery,
  createCandidateAnnotationTask,
  decideCandidateEligibility,
  downloadDatasetManifest,
  getCurationRun,
  getDataLineage,
  getDatasetSnapshot,
  getFeedbackCandidate,
  listCurationRuns,
  listDatasetSnapshots,
  listFeedbackCandidates,
  runCandidateDlp,
  startCurationRun,
  syncCandidateAnnotationTask,
} from "@/lib/api/client";

type CurationValues = {
  window_start: string;
  window_end: string;
  engine: "local" | "spark";
};

type CandidateQueryValues = Pick<
  FeedbackCandidateQuery,
  "status" | "eligibility_status" | "work_order_id" | "governance_blocker"
>;

const CANDIDATE_PAGE_SIZE = 20;
const RUN_PAGE_SIZE = 12;
const SNAPSHOT_PAGE_SIZE = 12;


export default function GovernedDatasetsPage() {
  const router = useRouter();
  const [candidates, setCandidates] = useState<FeedbackCandidate[]>();
  const [candidateActions, setCandidateActions] = useState<string[]>([]);
  const [candidateTotal, setCandidateTotal] = useState(0);
  const [candidatePage, setCandidatePage] = useState(1);
  const [candidateFilters, setCandidateFilters] = useState<CandidateQueryValues>({});
  const [runs, setRuns] = useState<CurationRun[]>();
  const [runActions, setRunActions] = useState<string[]>([]);
  const [runTotal, setRunTotal] = useState(0);
  const [runPage, setRunPage] = useState(1);
  const [runStatus, setRunStatus] = useState<CurationRunStatus>();
  const [runIdFilter, setRunIdFilter] = useState<string>();
  const [runIdDraft, setRunIdDraft] = useState("");
  const [snapshots, setSnapshots] = useState<DatasetSnapshot[]>();
  const [snapshotTotal, setSnapshotTotal] = useState(0);
  const [snapshotPage, setSnapshotPage] = useState(1);
  const [snapshotStatus, setSnapshotStatus] = useState<DatasetSnapshotStatus>();
  const [snapshotLineageStatus, setSnapshotLineageStatus] = useState<DatasetLineageStatus>();
  const [snapshotTrainingEligible, setSnapshotTrainingEligible] = useState<boolean>();
  const [snapshotTrainingBlocker, setSnapshotTrainingBlocker] =
    useState<DatasetTrainingBlockerCode>();
  const [snapshotIdFilter, setSnapshotIdFilter] = useState<string>();
  const [snapshotIdDraft, setSnapshotIdDraft] = useState("");
  const [snapshotRunIdFilter, setSnapshotRunIdFilter] = useState<string>();
  const [snapshotRunIdDraft, setSnapshotRunIdDraft] = useState("");
  const [candidateError, setCandidateError] = useState<unknown>();
  const [runError, setRunError] = useState<unknown>();
  const [snapshotError, setSnapshotError] = useState<unknown>();
  const [operationError, setOperationError] = useState<unknown>();
  const [operationReceipt, setOperationReceipt] = useState<OperationReceipt>();
  const [selectedCandidate, setSelectedCandidate] = useState<FeedbackCandidateDetail>();
  const [selectedRun, setSelectedRun] = useState<CurationRun>();
  const [selectedSnapshot, setSelectedSnapshot] = useState<DatasetSnapshot>();
  const [selectedLineage, setSelectedLineage] = useState<DataLineage>();
  const [curationOpen, setCurationOpen] = useState(false);
  const [eligibilityCandidate, setEligibilityCandidate] = useState<FeedbackCandidate>();
  const [annotationCandidate, setAnnotationCandidate] = useState<FeedbackCandidate>();
  const [busy, setBusy] = useState<string>();
  const [curationForm] = Form.useForm<CurationValues>();
  const [candidateQueryForm] = Form.useForm<CandidateQueryValues>();
  const [annotationForm] = Form.useForm();
  const locationTargetRef = useRef<string | undefined>(undefined);

  const load = useCallback(async () => {
    const [candidateResult, runResult, snapshotResult] = await Promise.allSettled([
      listFeedbackCandidates({
        ...candidateFilters,
        limit: CANDIDATE_PAGE_SIZE,
        offset: (candidatePage - 1) * CANDIDATE_PAGE_SIZE,
      }),
      listCurationRuns({
        run_id: runIdFilter,
        status: runStatus,
        limit: RUN_PAGE_SIZE,
        offset: (runPage - 1) * RUN_PAGE_SIZE,
      }),
      listDatasetSnapshots({
        snapshot_id: snapshotIdFilter,
        run_id: snapshotRunIdFilter,
        status: snapshotStatus,
        lineage_status: snapshotLineageStatus,
        training_eligible: snapshotTrainingEligible,
        training_blocker: snapshotTrainingBlocker,
        limit: SNAPSHOT_PAGE_SIZE,
        offset: (snapshotPage - 1) * SNAPSHOT_PAGE_SIZE,
      }),
    ]);
    if (candidateResult.status === "fulfilled") {
      setCandidates(candidateResult.value.candidates);
      setCandidateActions(candidateResult.value.legalActions);
      setCandidateTotal(candidateResult.value.total);
      const lastPage = Math.max(
        1,
        Math.ceil(candidateResult.value.total / CANDIDATE_PAGE_SIZE),
      );
      if (candidatePage > lastPage) setCandidatePage(lastPage);
      setCandidateError(undefined);
    } else {
      setCandidateError(candidateResult.reason);
    }
    if (runResult.status === "fulfilled") {
      setRuns(runResult.value.runs);
      setRunActions(runResult.value.legalActions);
      setRunTotal(runResult.value.total);
      const lastPage = Math.max(1, Math.ceil(runResult.value.total / RUN_PAGE_SIZE));
      if (runPage > lastPage) setRunPage(lastPage);
      setRunError(undefined);
    } else {
      setRunError(runResult.reason);
    }
    if (snapshotResult.status === "fulfilled") {
      setSnapshots(snapshotResult.value.snapshots);
      setSnapshotTotal(snapshotResult.value.total);
      const lastPage = Math.max(
        1,
        Math.ceil(snapshotResult.value.total / SNAPSHOT_PAGE_SIZE),
      );
      if (snapshotPage > lastPage) setSnapshotPage(lastPage);
      setSnapshotError(undefined);
    } else {
      setSnapshotError(snapshotResult.reason);
    }
  }, [
    candidateFilters,
    candidatePage,
    runIdFilter,
    runPage,
    runStatus,
    snapshotIdFilter,
    snapshotLineageStatus,
    snapshotPage,
    snapshotRunIdFilter,
    snapshotStatus,
    snapshotTrainingBlocker,
    snapshotTrainingEligible,
  ]);

  useEffect(() => { void load(); }, [load]);

  const totals = useMemo(() => ({
    ready: candidates?.filter((item) => item.status === "READY_FOR_CURATION").length ?? 0,
    snapshots: snapshotTotal,
    eligible: snapshots?.filter((item) => item.training_eligible).length ?? 0,
  }), [candidates, snapshotTotal, snapshots]);

  async function openCandidate(candidateId: string, updateLocation = true) {
    setBusy(candidateId);
    setOperationError(undefined);
    try {
      const detail = (await getFeedbackCandidate(candidateId)).candidate;
      setSelectedRun(undefined);
      setSelectedSnapshot(undefined);
      setSelectedLineage(undefined);
      setSelectedCandidate(detail);
      if (updateLocation) writeWorkspaceLocation({ kind: "candidate", id: candidateId });
    } catch (error) {
      setOperationError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function openSnapshot(snapshotId: string, updateLocation = true) {
    setBusy(snapshotId);
    setOperationError(undefined);
    try {
      const detail = (await getDatasetSnapshot(snapshotId)).snapshot;
      setSelectedCandidate(undefined);
      setSelectedRun(undefined);
      setSelectedSnapshot(detail);
      setSelectedLineage(undefined);
      if (detail.lineage_id) {
        const lineage = (await getDataLineage(detail.lineage_id)).lineage;
        setSelectedLineage(lineage);
      }
      if (updateLocation) writeWorkspaceLocation({ kind: "snapshot", id: snapshotId });
    } catch (error) {
      setOperationError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function openRun(runId: string, updateLocation = true) {
    setBusy(runId);
    setOperationError(undefined);
    try {
      const detail = (await getCurationRun(runId)).run;
      setRuns((current) => current?.map((run) => run.run_id === runId ? detail : run));
      setSelectedCandidate(undefined);
      setSelectedSnapshot(undefined);
      setSelectedLineage(undefined);
      setSelectedRun(detail);
      if (updateLocation) writeWorkspaceLocation({ kind: "run", id: runId });
    } catch (error) {
      setOperationError(error);
    } finally {
      setBusy(undefined);
    }
  }

  function writeWorkspaceLocation(target?: DatasetWorkspaceTarget) {
    if (typeof window === "undefined") return;
    locationTargetRef.current = datasetWorkspaceTargetKey(target);
    const nextUrl = buildDatasetWorkspaceUrl(window.location.href, target);
    const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextUrl !== currentUrl) window.history.pushState(null, "", nextUrl);
  }

  function closeWorkspace(updateLocation = true) {
    setSelectedCandidate(undefined);
    setSelectedRun(undefined);
    setSelectedSnapshot(undefined);
    setSelectedLineage(undefined);
    if (updateLocation) writeWorkspaceLocation();
  }

  useEffect(() => {
    function applyLocationTarget() {
      const target = readDatasetWorkspaceTarget(window.location.search);
      const targetKey = datasetWorkspaceTargetKey(target);
      if (locationTargetRef.current === targetKey) return;
      locationTargetRef.current = targetKey;
      if (!target) {
        closeWorkspace(false);
      } else if (target.kind === "candidate") {
        void openCandidate(target.id, false);
      } else if (target.kind === "run") {
        void openRun(target.id, false);
      } else {
        void openSnapshot(target.id, false);
      }
    }

    applyLocationTarget();
    window.addEventListener("popstate", applyLocationTarget);
    return () => window.removeEventListener("popstate", applyLocationTarget);
    // Resource functions intentionally read the target only on mount and popstate.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function startCuration(values: CurationValues) {
    setBusy("curation");
    setOperationError(undefined);
    setOperationReceipt(undefined);
    try {
      const result = await startCurationRun(
        {
          window_start: new Date(values.window_start).toISOString(),
          window_end: new Date(values.window_end).toISOString(),
          engine: values.engine,
        },
        crypto.randomUUID(),
      );
      const noData = result.run.status === "NO_DATA";
      setOperationReceipt({
        title: noData ? "策展运行已完成（当前窗口无可用数据）" : "策展运行已创建",
        resourceId: result.run.run_id,
        status: noData ? "无可用数据（未生成快照）" : result.run.status,
        requestId: result.requestId,
      });
      setCurationOpen(false);
      setSelectedCandidate(undefined);
      setSelectedSnapshot(undefined);
      setSelectedLineage(undefined);
      setSelectedRun(result.run);
      await load();
    } catch (error) {
      setOperationError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function refreshCandidateIfOpen(candidateId: string) {
    if (selectedCandidate?.candidate_id !== candidateId) return;
    const detail = (await getFeedbackCandidate(candidateId)).candidate;
    setSelectedCandidate((current) =>
      current?.candidate_id === candidateId ? detail : current
    );
  }

  async function decideEligibility(input: CandidateEligibilityInput) {
    if (!eligibilityCandidate) return;
    const candidateId = eligibilityCandidate.candidate_id;
    setBusy(candidateId);
    setOperationError(undefined);
    setOperationReceipt(undefined);
    try {
      const result = await decideCandidateEligibility(
        candidateId,
        eligibilityCandidate.version,
        input,
      );
      setOperationReceipt({
        title: "用途治理决策已保存",
        resourceId: result.candidate_id,
        status: result.status,
        requestId: result.requestId,
      });
      setEligibilityCandidate(undefined);
      await load();
      await refreshCandidateIfOpen(candidateId);
    } catch (error) {
      setOperationError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function applyDlp(candidate: FeedbackCandidate) {
    setBusy(candidate.candidate_id);
    setOperationError(undefined);
    setOperationReceipt(undefined);
    try {
      const result = await runCandidateDlp(candidate.candidate_id, candidate.version);
      setOperationReceipt({
        title: "DLP 派生数据已生成",
        resourceId: result.dlp_result_id,
        status: result.status,
        requestId: result.requestId,
      });
      await load();
      await refreshCandidateIfOpen(candidate.candidate_id);
    } catch (error) {
      setOperationError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function createAnnotation(values: {
    project_id: string;
    schema_version: string;
    risk_level: "STANDARD" | "HIGH";
  }) {
    if (!annotationCandidate) return;
    setBusy(annotationCandidate.candidate_id);
    setOperationError(undefined);
    setOperationReceipt(undefined);
    try {
      const result = await createCandidateAnnotationTask(
        annotationCandidate.candidate_id,
        annotationCandidate.version,
        crypto.randomUUID(),
        values,
      );
      setOperationReceipt({
        title: "标注复核任务已创建",
        resourceId: result.task_id,
        status: result.status,
        requestId: result.requestId,
      });
      setAnnotationCandidate(undefined);
      await load();
      await refreshCandidateIfOpen(annotationCandidate.candidate_id);
    } catch (error) {
      setOperationError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function syncAnnotation(candidate: FeedbackCandidate) {
    setBusy(candidate.candidate_id);
    setOperationError(undefined);
    setOperationReceipt(undefined);
    try {
      const detail = (await getFeedbackCandidate(candidate.candidate_id)).candidate;
      const annotation = detail.timeline.find((item) => item.stage === "ANNOTATION");
      const version = Number(annotation?.facts?.version);
      if (!annotation || !Number.isInteger(version)) return;
      const result = await syncCandidateAnnotationTask(annotation.resource_id, version);
      setOperationReceipt({
        title: "标注复核结果已同步",
        resourceId: result.task_id,
        status: result.status,
        requestId: result.requestId,
      });
      await load();
      await refreshCandidateIfOpen(candidate.candidate_id);
    } catch (error) {
      setOperationError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function saveManifest(snapshot: DatasetSnapshot) {
    setBusy(snapshot.snapshot_id);
    setOperationError(undefined);
    setOperationReceipt(undefined);
    try {
      const manifest = await downloadDatasetManifest(snapshot.snapshot_id);
      const url = URL.createObjectURL(manifest.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = manifest.filename;
      anchor.click();
      URL.revokeObjectURL(url);
      setOperationReceipt({
        title: "数据集清单已通过完整性校验",
        resourceId: snapshot.snapshot_id,
        status: manifest.manifestHash,
      });
    } catch (error) {
      setOperationError(error);
    } finally {
      setBusy(undefined);
    }
  }

  function applyCandidateQuery(values: CandidateQueryValues) {
    setCandidatePage(1);
    setCandidateFilters({
      status: values.status,
      eligibility_status: values.eligibility_status,
      work_order_id: values.work_order_id?.trim() || undefined,
      governance_blocker: values.governance_blocker,
    });
  }

  function resetCandidateQuery() {
    candidateQueryForm.resetFields();
    setCandidatePage(1);
    setCandidateFilters({});
  }

  return (
    <AppShell>
      <div className="page-stack">
        <div>
          <Typography.Title level={2}>治理数据与训练候选工作台</Typography.Title>
          <Typography.Paragraph type="secondary">
            从已关闭工单到用途授权、DLP、双人复核、Parquet 快照和 OpenLineage 的可追溯闭环。
          </Typography.Paragraph>
        </div>
        <Alert
          showIcon
          type="info"
          message="服务端事实与操作门禁"
          description="页面不缓存候选原文、DLP 命中位置或标注正文；所有按钮均来自服务端 legal_actions，清单下载会再次执行租户和角色授权。"
        />
        {operationError ? (
          <ErrorState error={operationError} onRetry={() => setOperationError(undefined)} />
        ) : null}
        {operationReceipt ? (
          <OperationReceiptAlert
            receipt={operationReceipt}
            onClose={() => setOperationReceipt(undefined)}
          />
        ) : null}
        <Row gutter={[16, 16]}>
          <Col xs={24} md={8}><Card><Statistic title="本页可策展候选" value={totals.ready} /></Card></Col>
          <Col xs={24} md={8}><Card><Statistic title="快照目录结果" value={totals.snapshots} /></Card></Col>
          <Col xs={24} md={8}><Card><Statistic title="本页训练候选" value={totals.eligible} /></Card></Col>
        </Row>
        {[...candidateActions, ...runActions].includes("START_CURATION_RUN") ? (
          <Button type="primary" onClick={() => setCurationOpen(true)}>发起固定窗口策展</Button>
        ) : null}
        <Tabs
          items={[
            {
              key: "candidates",
              label: `反馈候选与治理门禁 (${candidateTotal})`,
              children: (
                <div className="page-stack">
                  <Card size="small">
                    <Form<CandidateQueryValues>
                      form={candidateQueryForm}
                      layout="inline"
                      onFinish={applyCandidateQuery}
                    >
                      <Form.Item name="status" label="治理状态">
                        <Select
                          allowClear
                          style={{ minWidth: 210 }}
                          options={[
                            { value: "PENDING_GOVERNANCE", label: "待治理" },
                            { value: "ELIGIBLE", label: "资格已批准" },
                            { value: "ELIGIBILITY_DENIED", label: "资格已拒绝" },
                            { value: "DLP_APPROVED", label: "DLP 已通过" },
                            { value: "DLP_REVIEW_REQUIRED", label: "DLP 需复核" },
                            { value: "ANNOTATION_PENDING", label: "等待标注复核" },
                            { value: "ANNOTATION_REVIEW_REQUIRED", label: "标注需复核" },
                            { value: "ANNOTATION_CONFLICT", label: "标注冲突" },
                            { value: "READY_FOR_CURATION", label: "可进入策展" },
                          ]}
                        />
                      </Form.Item>
                      <Form.Item name="eligibility_status" label="用途资格">
                        <Select
                          allowClear
                          style={{ minWidth: 150 }}
                          options={[
                            { value: "UNDECIDED", label: "未决定" },
                            { value: "ELIGIBLE", label: "已批准" },
                            { value: "INELIGIBLE", label: "不符合" },
                          ]}
                        />
                      </Form.Item>
                      <Form.Item name="governance_blocker" label="当前阻断">
                        <Select
                          allowClear
                          showSearch
                          optionFilterProp="label"
                          placeholder="全部阻断原因"
                          style={{ minWidth: 210 }}
                          options={GOVERNANCE_BLOCKER_OPTIONS}
                        />
                      </Form.Item>
                      <Form.Item name="work_order_id" label="源工单">
                        <Input allowClear placeholder="输入完整工单 ID" />
                      </Form.Item>
                      <Form.Item>
                        <Space>
                          <Button type="primary" htmlType="submit">查询</Button>
                          <Button onClick={resetCandidateQuery}>重置</Button>
                        </Space>
                      </Form.Item>
                    </Form>
                  </Card>
                  {candidateError ? (
                    <ErrorState error={candidateError} onRetry={() => void load()} />
                  ) : !candidates ? <LoadingState /> : candidates.length === 0 ? (
                    <EmptyState description="当前筛选条件下没有可见候选" />
                  ) : (
                    <>
                      <List
                        dataSource={candidates}
                        renderItem={(candidate) => (
                          <List.Item
                            actions={[
                              <CandidateGovernanceActions
                                key="governance-actions"
                                candidate={candidate}
                                busy={busy === candidate.candidate_id}
                                onInspect={() => void openCandidate(candidate.candidate_id)}
                                onEligibility={() => setEligibilityCandidate(candidate)}
                                onDlp={() => void applyDlp(candidate)}
                                onAnnotation={() => setAnnotationCandidate(candidate)}
                                onSync={() => void syncAnnotation(candidate)}
                              />,
                            ]}
                          >
                            <List.Item.Meta
                              title={(
                                <Space wrap>
                                  <Typography.Text code>{candidate.candidate_id}</Typography.Text>
                                  <Tag>{candidate.status}</Tag>
                                  <Tag color={candidate.allow_training ? "green" : candidate.allow_training === false ? "red" : "default"}>
                                    {candidate.eligibility_status}
                                  </Tag>
                                  <Tag>许可：{candidate.license_status}</Tag>
                                  <Tag color={candidate.diagnosis_feedback_count > 0 ? "blue" : "default"}>
                                    诊断反馈引用：{candidate.diagnosis_feedback_count} 条
                                  </Tag>
                                </Space>
                              )}
                              description={(
                                <Space direction="vertical" size={4}>
                                  <Typography.Text type="secondary">
                                    工单 {candidate.work_order_id} · v{candidate.version} · {candidate.source_content_hash}
                                  </Typography.Text>
                                  <GovernanceBlockers blockers={candidate.governance_blockers} />
                                </Space>
                              )}
                            />
                          </List.Item>
                        )}
                      />
                      <Pagination
                        current={candidatePage}
                        pageSize={CANDIDATE_PAGE_SIZE}
                        total={candidateTotal}
                        showSizeChanger={false}
                        showTotal={(total) => `共 ${total} 条候选`}
                        onChange={setCandidatePage}
                      />
                    </>
                  )}
                </div>
              ),
            },
            {
              key: "runs",
              label: `策展运行记录 (${runTotal})`,
              children: (
                <div className="page-stack">
                  <Card size="small">
                    <Space wrap>
                      <Typography.Text>Run ID</Typography.Text>
                      <Input.Search
                        allowClear
                        aria-label="按完整 Run ID 查询策展运行"
                        enterButton="查询"
                        maxLength={128}
                        placeholder="输入完整 Run ID"
                        style={{ width: 300 }}
                        value={runIdDraft}
                        onChange={(event) => {
                          const value = event.target.value;
                          setRunIdDraft(value);
                          if (!value) {
                            setRunPage(1);
                            setRunIdFilter(undefined);
                          }
                        }}
                        onSearch={(value) => {
                          setRunPage(1);
                          setRunIdFilter(normalizeStableId(value));
                        }}
                      />
                      <Typography.Text>运行状态</Typography.Text>
                      <Select<CurationRunStatus>
                        allowClear
                        value={runStatus}
                        placeholder="全部状态"
                        style={{ minWidth: 190 }}
                        options={[
                          { value: "COMPLETED", label: "已完成" },
                          { value: "NO_DATA", label: "无可用数据" },
                          { value: "FAILED", label: "失败" },
                          { value: "LINEAGE_PENDING", label: "等待血缘补发" },
                        ]}
                        onChange={(value) => {
                          setRunPage(1);
                          setRunStatus(value);
                        }}
                      />
                      <Button onClick={() => void load()}>刷新运行目录</Button>
                    </Space>
                  </Card>
                  {runError ? (
                    <ErrorState error={runError} onRetry={() => void load()} />
                  ) : !runs ? <LoadingState /> : runs.length === 0 ? (
                    <EmptyState description="当前筛选条件下没有策展运行" />
                  ) : (
                    <>
                      <CurationRunCatalog
                        runs={runs}
                        busyRunId={busy}
                        onInspect={(runId) => void openRun(runId)}
                        onOpenSnapshot={(snapshotId) => void openSnapshot(snapshotId)}
                      />
                      <Pagination
                        current={runPage}
                        pageSize={RUN_PAGE_SIZE}
                        total={runTotal}
                        showSizeChanger={false}
                        showTotal={(total) => `共 ${total} 次运行`}
                        onChange={setRunPage}
                      />
                    </>
                  )}
                </div>
              ),
            },
            {
              key: "snapshots",
              label: `数据集快照目录 (${snapshotTotal})`,
              children: (
                <div className="page-stack">
                  <Card size="small">
                    <Space wrap>
                      <Typography.Text>Snapshot ID</Typography.Text>
                      <Input.Search
                        allowClear
                        aria-label="按完整 Snapshot ID 查询数据集快照"
                        enterButton="查询"
                        maxLength={128}
                        placeholder="输入完整 Snapshot ID"
                        style={{ width: 320 }}
                        value={snapshotIdDraft}
                        onChange={(event) => {
                          const value = event.target.value;
                          setSnapshotIdDraft(value);
                          if (!value) {
                            setSnapshotPage(1);
                            setSnapshotIdFilter(undefined);
                          }
                        }}
                        onSearch={(value) => {
                          setSnapshotPage(1);
                          setSnapshotIdFilter(normalizeStableId(value));
                        }}
                      />
                      <Typography.Text>来源 Run ID</Typography.Text>
                      <Input.Search
                        allowClear
                        aria-label="按完整来源 Run ID 查询数据集快照"
                        enterButton="查询"
                        maxLength={128}
                        placeholder="输入完整来源 Run ID"
                        style={{ width: 300 }}
                        value={snapshotRunIdDraft}
                        onChange={(event) => {
                          const value = event.target.value;
                          setSnapshotRunIdDraft(value);
                          if (!value) {
                            setSnapshotPage(1);
                            setSnapshotRunIdFilter(undefined);
                          }
                        }}
                        onSearch={(value) => {
                          setSnapshotPage(1);
                          setSnapshotRunIdFilter(normalizeStableId(value));
                        }}
                      />
                      <Typography.Text>快照状态</Typography.Text>
                      <Select<DatasetSnapshotStatus>
                        allowClear
                        value={snapshotStatus}
                        placeholder="全部状态"
                        style={{ minWidth: 160 }}
                        options={[
                          { value: "CANDIDATE", label: "候选快照" },
                          { value: "LINEAGE_PENDING", label: "等待血缘" },
                        ]}
                        onChange={(value) => {
                          setSnapshotPage(1);
                          setSnapshotStatus(value);
                        }}
                      />
                      <Typography.Text>血缘状态</Typography.Text>
                      <Select<DatasetLineageStatus>
                        allowClear
                        value={snapshotLineageStatus}
                        placeholder="全部血缘"
                        style={{ minWidth: 150 }}
                        options={[
                          { value: "CONFIRMED", label: "已确认" },
                          { value: "PENDING", label: "待补发" },
                        ]}
                        onChange={(value) => {
                          setSnapshotPage(1);
                          setSnapshotLineageStatus(value);
                        }}
                      />
                      <Typography.Text>训练资格</Typography.Text>
                      <Select<"eligible" | "blocked">
                        allowClear
                        value={snapshotTrainingEligible === undefined
                          ? undefined
                          : snapshotTrainingEligible ? "eligible" : "blocked"}
                        placeholder="全部资格"
                        style={{ minWidth: 150 }}
                        options={[
                          { value: "eligible", label: "可用于训练" },
                          { value: "blocked", label: "不可用于训练" },
                        ]}
                        onChange={(value) => {
                          setSnapshotPage(1);
                          setSnapshotTrainingEligible(
                            value === undefined ? undefined : value === "eligible",
                          );
                        }}
                      />
                      <Typography.Text>阻断原因</Typography.Text>
                      <Select<DatasetTrainingBlockerCode>
                        allowClear
                        value={snapshotTrainingBlocker}
                        placeholder="全部原因"
                        style={{ minWidth: 210 }}
                        options={DATASET_TRAINING_BLOCKER_OPTIONS}
                        onChange={(value) => {
                          setSnapshotPage(1);
                          setSnapshotTrainingBlocker(value);
                        }}
                      />
                      <Button onClick={() => void load()}>刷新快照目录</Button>
                    </Space>
                  </Card>
                  {snapshotError ? (
                    <ErrorState error={snapshotError} onRetry={() => void load()} />
                  ) : !snapshots ? <LoadingState /> : snapshots.length === 0 ? (
                    <EmptyState description="当前筛选条件下没有数据集快照" />
                  ) : (
                    <>
                      <List
                        grid={{ gutter: 16, xs: 1, lg: 2 }}
                        dataSource={snapshots}
                        renderItem={(snapshot) => (
                          <List.Item>
                            <Card
                              title={snapshot.snapshot_id}
                              extra={<Tag color={snapshot.training_eligible ? "green" : "orange"}>{snapshot.status}</Tag>}
                            >
                              <Descriptions size="small" column={1} items={[
                                { key: "rows", label: "行数", children: snapshot.row_count },
                                { key: "contract", label: "数据契约", children: snapshot.contract_version },
                                { key: "lineage", label: "血缘", children: snapshot.lineage_status },
                                { key: "training", label: "训练资格", children: snapshot.training_eligible ? "可用" : "阻断" },
                                { key: "hash", label: "清单哈希", children: snapshot.manifest_hash ?? "未发布" },
                              ]} />
                              <DatasetTrainingBlockers blockers={snapshot.training_blockers} />
                              <DatasetSnapshotActions
                                snapshot={snapshot}
                                busy={busy === snapshot.snapshot_id}
                                onInspect={() => void openSnapshot(snapshot.snapshot_id)}
                                onCreateExperiment={() =>
                                  router.push(buildExperimentCreateUrl(snapshot.snapshot_id))
                                }
                                onDownload={() => void saveManifest(snapshot)}
                              />
                            </Card>
                          </List.Item>
                        )}
                      />
                      <Pagination
                        current={snapshotPage}
                        pageSize={SNAPSHOT_PAGE_SIZE}
                        total={snapshotTotal}
                        showSizeChanger={false}
                        showTotal={(total) => `共 ${total} 个快照`}
                        onChange={setSnapshotPage}
                      />
                    </>
                  )}
                </div>
              ),
            },
          ]}
        />
      </div>

      <Drawer
        open={Boolean(selectedCandidate || selectedRun || selectedSnapshot)}
        width={620}
        title={selectedCandidate?.candidate_id ?? selectedRun?.run_id ?? selectedSnapshot?.snapshot_id ?? "数据血缘"}
        onClose={() => {
          closeWorkspace();
        }}
      >
        {selectedRun ? (
          <div className="page-stack">
            <Descriptions
              bordered
              size="small"
              column={1}
              items={[
                { key: "status", label: "运行状态", children: <Tag>{selectedRun.status}</Tag> },
                { key: "engine", label: "执行引擎", children: selectedRun.engine },
                { key: "start", label: "窗口开始", children: new Date(selectedRun.window_start).toLocaleString("zh-CN") },
                { key: "end", label: "窗口结束", children: new Date(selectedRun.window_end).toLocaleString("zh-CN") },
                { key: "inputs", label: "输入数量", children: selectedRun.input_count },
                { key: "contract", label: "数据契约", children: selectedRun.contract_version },
                { key: "code", label: "代码版本", children: selectedRun.code_version },
                { key: "config", label: "配置哈希", children: <Typography.Text code copyable>{selectedRun.config_hash}</Typography.Text> },
                { key: "manifest", label: "输入清单哈希", children: <Typography.Text code copyable>{selectedRun.input_manifest_hash}</Typography.Text> },
              ]}
            />
            {selectedRun.failure_reason ? (
              <Alert
                showIcon
                type={selectedRun.status === "LINEAGE_PENDING" ? "warning" : "error"}
                title={selectedRun.status === "LINEAGE_PENDING" ? "等待血缘补发" : "运行阻断原因"}
                description={selectedRun.failure_reason}
              />
            ) : null}
            <CurationExclusionReport
              exclusions={selectedRun.exclusion_report}
              onOpenCandidate={(candidateId) => void openCandidate(candidateId)}
            />
            {selectedRun.snapshot_id ? (
              <Button
                type="primary"
                onClick={() => {
                  if (selectedRun.snapshot_id) void openSnapshot(selectedRun.snapshot_id);
                }}
              >
                打开关联数据集快照
              </Button>
            ) : null}
          </div>
        ) : selectedCandidate ? (
          <div className="page-stack">
            <Descriptions
              bordered
              size="small"
              column={1}
              items={[
                { key: "status", label: "候选状态", children: <Tag>{selectedCandidate.status}</Tag> },
                { key: "work-order", label: "源工单", children: selectedCandidate.work_order_id },
                { key: "version", label: "候选版本", children: selectedCandidate.version },
                { key: "diagnosis-feedback", label: "冻结诊断反馈引用", children: `${selectedCandidate.diagnosis_feedback_count} 条（仅计数；正文须经 DLP）` },
                { key: "hash", label: "源内容哈希", children: <Typography.Text code copyable>{selectedCandidate.source_content_hash}</Typography.Text> },
              ]}
            />
            <CandidateGovernanceActions
              candidate={selectedCandidate}
              busy={busy === selectedCandidate.candidate_id}
              onEligibility={() => setEligibilityCandidate(selectedCandidate)}
              onDlp={() => void applyDlp(selectedCandidate)}
              onAnnotation={() => setAnnotationCandidate(selectedCandidate)}
              onSync={() => void syncAnnotation(selectedCandidate)}
            />
            <GovernanceBlockers blockers={selectedCandidate.governance_blockers} />
            <EligibilityDecisionHistory decisions={selectedCandidate.eligibility_decisions} />
            <DlpProcessingHistory runs={selectedCandidate.dlp_runs} />
            <AnnotationReviewHistory tasks={selectedCandidate.annotation_tasks} />
            <Card title="治理与数据血缘" size="small">
              <DataLineageTimeline stages={selectedCandidate.timeline} />
            </Card>
          </div>
        ) : selectedSnapshot ? (
          <div className="page-stack">
            <Descriptions
              bordered
              size="small"
              column={1}
              items={[
                { key: "status", label: "快照状态", children: <Tag>{selectedSnapshot.status}</Tag> },
                { key: "training", label: "训练资格", children: selectedSnapshot.training_eligible ? <Tag color="green">可用</Tag> : <Tag color="orange">阻断</Tag> },
                { key: "rows", label: "总行数", children: selectedSnapshot.row_count },
                { key: "splits", label: "数据切分", children: <Space wrap>{Object.entries(selectedSnapshot.split_counts).map(([split, count]) => <Tag key={split}>{split}: {count}</Tag>)}</Space> },
                { key: "base-snapshot", label: "派生自快照", children: selectedSnapshot.base_snapshot_id ?? "—" },
                { key: "augmentation-contract", label: "增强契约", children: selectedSnapshot.augmentation_contract_version ?? "—" },
                { key: "sample-origins", label: "样本来源", children: selectedSnapshot.sample_origin_counts ? <Space wrap>{Object.entries(selectedSnapshot.sample_origin_counts).map(([origin, count]) => <Tag key={origin}>{origin}: {count}</Tag>)}</Space> : "纯真实数据" },
                { key: "synthetic-splits", label: "合成样本切分", children: selectedSnapshot.synthetic_split_counts ? <Space wrap>{Object.entries(selectedSnapshot.synthetic_split_counts).map(([split, count]) => <Tag key={split}>{split}: {count}</Tag>)}</Space> : "—" },
                { key: "contract", label: "数据契约", children: selectedSnapshot.contract_version },
                { key: "input-hash", label: "输入清单哈希", children: <Typography.Text code copyable>{selectedSnapshot.input_manifest_hash}</Typography.Text> },
                { key: "manifest-hash", label: "发布清单哈希", children: selectedSnapshot.manifest_hash ? <Typography.Text code copyable>{selectedSnapshot.manifest_hash}</Typography.Text> : "未发布" },
              ]}
            />
            <DatasetTrainingBlockers blockers={selectedSnapshot.training_blockers} />
            <DatasetSnapshotActions
              snapshot={selectedSnapshot}
              busy={busy === selectedSnapshot.snapshot_id || busy === selectedSnapshot.run_id}
              onOpenRun={() => void openRun(selectedSnapshot.run_id)}
              onCreateExperiment={() =>
                router.push(buildExperimentCreateUrl(selectedSnapshot.snapshot_id))
              }
              onDownload={() => void saveManifest(selectedSnapshot)}
            />
            {selectedLineage ? (
              <DatasetLineageExplorer lineage={selectedLineage} />
            ) : (
              <Alert
                showIcon
                type="warning"
                title="快照血缘尚未登记或未能加载"
                description="该快照不会因为缺少已确认血缘而获得训练资格。"
              />
            )}
            <DatasetQualityReportCard
              snapshotId={selectedSnapshot.snapshot_id}
              available={(selectedSnapshot.artifacts ?? []).some(
                (artifact) => artifact.kind === "quality_report",
              )}
            />
            <DatasetArtifactCatalog artifacts={selectedSnapshot.artifacts} />
          </div>
        ) : null}
      </Drawer>

      <Modal open={curationOpen} title="发起固定窗口策展" footer={null} onCancel={() => setCurationOpen(false)}>
        <Form<CurationValues>
          form={curationForm}
          layout="vertical"
          initialValues={{ window_start: localDateTime(-24), window_end: localDateTime(0), engine: "spark" }}
          onFinish={(values) => void startCuration(values)}
        >
          <Form.Item name="window_start" label="窗口开始" rules={[{ required: true }]}><Input type="datetime-local" /></Form.Item>
          <Form.Item name="window_end" label="窗口结束" rules={[{ required: true }]}><Input type="datetime-local" /></Form.Item>
          <Form.Item name="engine" label="执行引擎" rules={[{ required: true }]}><Select options={[{ value: "spark", label: "Spark 集群" }, { value: "local", label: "本地回归" }]} /></Form.Item>
          <Button htmlType="submit" type="primary" loading={busy === "curation"}>开始策展</Button>
        </Form>
      </Modal>

      <Modal open={Boolean(eligibilityCandidate)} title="用途与许可治理决策" footer={null} onCancel={() => setEligibilityCandidate(undefined)}>
        <EligibilityDecisionForm
          key={`${eligibilityCandidate?.candidate_id ?? "candidate"}-${eligibilityCandidate?.version ?? 0}`}
          submitting={Boolean(eligibilityCandidate && busy === eligibilityCandidate.candidate_id)}
          onSubmit={(input) => void decideEligibility(input)}
        />
      </Modal>

      <Modal open={Boolean(annotationCandidate)} title="创建 Label Studio 复核任务" footer={null} onCancel={() => setAnnotationCandidate(undefined)}>
        <Form
          form={annotationForm}
          layout="vertical"
          initialValues={{ project_id: "industrial-root-cause", schema_version: "root-cause-label-v1", risk_level: "HIGH" }}
          onFinish={(values) => void createAnnotation(values)}
        >
          <Form.Item name="project_id" label="项目" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="schema_version" label="标注模式版本" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="risk_level" label="风险级别"><Select options={[{ value: "HIGH", label: "高风险双人复核" }, { value: "STANDARD", label: "标准复核" }]} /></Form.Item>
          <Button
            htmlType="submit"
            type="primary"
            loading={Boolean(annotationCandidate && busy === annotationCandidate.candidate_id)}
          >
            创建复核任务
          </Button>
        </Form>
      </Modal>
    </AppShell>
  );
}

function localDateTime(offsetHours: number): string {
  const date = new Date(Date.now() + offsetHours * 60 * 60 * 1000);
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60 * 1000);
  return local.toISOString().slice(0, 16);
}
