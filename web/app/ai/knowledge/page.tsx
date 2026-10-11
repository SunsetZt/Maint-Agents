"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Drawer,
  Form,
  Input,
  Modal,
  Popconfirm,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
  Upload,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type KnowledgeDeletion,
  type KnowledgeIndexActivation,
  type KnowledgeRelease,
  type KnowledgeFile,
  type KnowledgeVersion,
  createKnowledgeDocument,
  createKnowledgeDocumentVersion,
  createKnowledgeRelease,
  createKnowledgeSourceFile,
  getKnowledgeSourceFile,
  listKnowledgeIndexActivations,
  listKnowledgeDeletions,
  listKnowledgeSourceFiles,
  listKnowledgeReleases,
  listKnowledgeVersions,
  promoteKnowledgeRelease,
  rollbackKnowledgeRelease,
  requestKnowledgeDeletion,
  reprocessKnowledgeSourceFile,
  retryKnowledgeDeletion,
  reviewKnowledgeVersion,
  startKnowledgeIndexEvaluation,
  sweepExpiredKnowledge,
  uploadKnowledgeSourceContent,
} from "@/lib/api/client";

type KnowledgeContentValues = {
  title: string;
  source_uri: string;
  content: string;
  classification: "internal" | "restricted";
  acl_subject_ids: string[];
  acl_roles: string[];
  device_families: string[];
  device_models: string[];
  valid_from: string;
  valid_to?: string;
};

type KnowledgeScopeValues = Pick<
  KnowledgeContentValues,
  | "acl_subject_ids"
  | "acl_roles"
  | "device_families"
  | "device_models"
  | "valid_from"
  | "valid_to"
>;

type KnowledgeVersionValues = KnowledgeScopeValues & { content: string };

type KnowledgeFileValues = Omit<KnowledgeContentValues, "source_uri" | "content">;

type ReleaseValues = {
  name: string;
  document_version_ids: string[];
};

type DeletionValues = {
  trigger: "LEGAL_DELETION" | "ACCESS_REVOKED" | "RETENTION_EXPIRED";
  reason: string;
};

type RollbackValues = {
  reason: string;
};

const roleOptions = [
  "field_engineer",
  "after_sales_engineer",
  "domain_expert",
  "tenant_admin",
].map((value) => ({ value, label: value }));

export default function KnowledgeWorkspacePage() {
  const [versions, setVersions] = useState<KnowledgeVersion[]>();
  const [releases, setReleases] = useState<KnowledgeRelease[]>();
  const [activations, setActivations] = useState<KnowledgeIndexActivation[]>();
  const [fileJobs, setFileJobs] = useState<KnowledgeFile[]>();
  const [deletions, setDeletions] = useState<KnowledgeDeletion[]>();
  const [collectionActions, setCollectionActions] = useState<string[]>([]);
  const [deletionActions, setDeletionActions] = useState<string[]>([]);
  const [versionError, setVersionError] = useState<unknown>();
  const [releaseError, setReleaseError] = useState<unknown>();
  const [activationError, setActivationError] = useState<unknown>();
  const [fileError, setFileError] = useState<unknown>();
  const [deletionError, setDeletionError] = useState<unknown>();
  const [commandError, setCommandError] = useState<unknown>();
  const [busy, setBusy] = useState<string>();
  const [createOpen, setCreateOpen] = useState(false);
  const [fileOpen, setFileOpen] = useState(false);
  const [sourceFile, setSourceFile] = useState<File>();
  const [fileIngestion, setFileIngestion] = useState<KnowledgeFile>();
  const [releaseOpen, setReleaseOpen] = useState(false);
  const [versionTarget, setVersionTarget] = useState<KnowledgeVersion>();
  const [selectedVersion, setSelectedVersion] = useState<KnowledgeVersion>();
  const [selectedRelease, setSelectedRelease] = useState<KnowledgeRelease>();
  const [rollbackTarget, setRollbackTarget] = useState<KnowledgeRelease>();
  const [deletionTarget, setDeletionTarget] = useState<KnowledgeVersion>();
  const [selectedDeletion, setSelectedDeletion] = useState<KnowledgeDeletion>();
  const [documentForm] = Form.useForm<KnowledgeContentValues>();
  const [fileForm] = Form.useForm<KnowledgeFileValues>();
  const [versionForm] = Form.useForm<KnowledgeVersionValues>();
  const [releaseForm] = Form.useForm<ReleaseValues>();
  const [rollbackForm] = Form.useForm<RollbackValues>();
  const [deletionForm] = Form.useForm<DeletionValues>();

  const load = useCallback(async () => {
    const [versionResult, releaseResult, activationResult, fileResult, deletionResult] =
      await Promise.allSettled([
        listKnowledgeVersions(),
        listKnowledgeReleases(),
        listKnowledgeIndexActivations(),
        listKnowledgeSourceFiles(),
        listKnowledgeDeletions(),
      ]);
    if (versionResult.status === "fulfilled") {
      setVersions(versionResult.value.versions);
      setCollectionActions(versionResult.value.legalActions);
      setVersionError(undefined);
    } else {
      setVersionError(versionResult.reason);
    }
    if (releaseResult.status === "fulfilled") {
      setReleases(releaseResult.value.releases);
      setReleaseError(undefined);
      setSelectedRelease((current) =>
        current
          ? releaseResult.value.releases.find(
              (item) => item.release_id === current.release_id,
            ) ?? current
          : current,
      );
    } else {
      setReleaseError(releaseResult.reason);
    }
    if (activationResult.status === "fulfilled") {
      setActivations(activationResult.value.activations);
      setActivationError(undefined);
    } else {
      setActivationError(activationResult.reason);
    }
    if (fileResult.status === "fulfilled") {
      setFileJobs(fileResult.value.files);
      setFileError(undefined);
      setFileIngestion((current) =>
        current
          ? fileResult.value.files.find((item) => item.ingestion_id === current.ingestion_id) ??
            current
          : current,
      );
    } else {
      setFileError(fileResult.reason);
    }
    if (deletionResult.status === "fulfilled") {
      setDeletions(deletionResult.value.deletions);
      setDeletionActions(deletionResult.value.legalActions);
      setDeletionError(undefined);
      setSelectedDeletion((current) =>
        current
          ? deletionResult.value.deletions.find(
              (item) => item.deletion_id === current.deletion_id,
            ) ?? current
          : current,
      );
    } else {
      setDeletionError(deletionResult.reason);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!fileIngestion || !["QUEUED", "RUNNING"].includes(fileIngestion.status)) return;
    const ingestionId = fileIngestion.ingestion_id;
    let cancelled = false;
    let inFlight = false;
    const poll = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const next = await getKnowledgeSourceFile(ingestionId);
        if (cancelled) return;
        setCommandError(undefined);
        setFileIngestion(next);
        if (next.status === "DRAFT_READY") await load();
      } catch (error) {
        if (!cancelled) setCommandError(error);
      } finally {
        inFlight = false;
      }
    };
    const timer = window.setInterval(() => void poll(), 1500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [fileIngestion?.ingestion_id, fileIngestion?.status, load]);

  useEffect(() => {
    if (!fileJobs?.some((item) => ["QUEUED", "RUNNING"].includes(item.status))) return;
    const timer = window.setInterval(() => void load(), 3000);
    return () => window.clearInterval(timer);
  }, [fileJobs, load]);

  useEffect(() => {
    if (!deletions?.some((item) => ["QUEUED", "RUNNING"].includes(item.status))) return;
    const timer = window.setInterval(() => void load(), 2500);
    return () => window.clearInterval(timer);
  }, [deletions, load]);

  useEffect(() => {
    if (!releases?.some((item) => item.status === "EVALUATING")) return;
    const timer = window.setInterval(() => void load(), 2500);
    return () => window.clearInterval(timer);
  }, [releases, load]);

  const totals = useMemo(
    () => ({
      draft: versions?.filter((item) => item.status === "DRAFT").length ?? 0,
      reviewed: versions?.filter((item) => item.status === "REVIEWED").length ?? 0,
      published: versions?.filter((item) => item.status === "PUBLISHED").length ?? 0,
      active: releases?.filter((item) => item.status === "PUBLISHED" && item.is_active).length ?? 0,
    }),
    [releases, versions],
  );

  async function createDocument(values: KnowledgeContentValues) {
    setBusy("create-document");
    setCommandError(undefined);
    try {
      await createKnowledgeDocument(
        {
          ...scopePayload(values),
          title: values.title,
          source_uri: values.source_uri,
          content: values.content,
          classification: values.classification,
        },
        crypto.randomUUID(),
      );
      setCreateOpen(false);
      documentForm.resetFields();
      await load();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function ingestFile(values: KnowledgeFileValues) {
    if (!sourceFile) {
      setCommandError(new Error("请选择 TXT、Markdown 或 PDF 文件"));
      return;
    }
    setBusy("ingest-file");
    setCommandError(undefined);
    try {
      const intent =
        fileIngestion?.status === "AWAITING_UPLOAD"
          ? fileIngestion
          : await createKnowledgeSourceFile(
              {
                ...scopePayload(values),
                title: values.title,
                source_filename: sourceFile.name,
                declared_mime: declaredKnowledgeMime(sourceFile),
                classification: values.classification,
              },
              crypto.randomUUID(),
            );
      setFileIngestion(intent);
      const queued = await uploadKnowledgeSourceContent(
        intent.ingestion_id,
        intent.version,
        sourceFile,
      );
      setFileIngestion(queued);
      await load();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function reprocessFile(target: KnowledgeFile) {
    setBusy(`reprocess-${target.ingestion_id}`);
    setCommandError(undefined);
    try {
      const updated = await reprocessKnowledgeSourceFile(
        target.ingestion_id,
        target.version,
        `reprocess-${target.ingestion_id}-${target.version}`,
      );
      setFileIngestion(updated);
      await load();
    } catch (error) {
      setCommandError(error);
      await load();
    } finally {
      setBusy(undefined);
    }
  }

  function openFileIngestion() {
    setFileIngestion(undefined);
    setSourceFile(undefined);
    fileForm.resetFields();
    setFileOpen(true);
  }

  function openNewVersion(target: KnowledgeVersion) {
    setVersionTarget(target);
    versionForm.setFieldsValue({
      content: "",
      acl_subject_ids: target.acl_subject_ids,
      acl_roles: target.acl_roles,
      device_families: target.device_families,
      device_models: target.device_models,
      valid_from: toLocalInput(target.valid_from),
      valid_to: target.valid_to ? toLocalInput(target.valid_to) : undefined,
    });
  }

  async function createVersion(values: KnowledgeVersionValues) {
    if (!versionTarget) return;
    setBusy(`version-${versionTarget.document_id}`);
    setCommandError(undefined);
    try {
      await createKnowledgeDocumentVersion(
        versionTarget.document_id,
        { ...scopePayload(values), content: values.content },
        crypto.randomUUID(),
      );
      setVersionTarget(undefined);
      versionForm.resetFields();
      await load();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function review(version: KnowledgeVersion) {
    setBusy(`review-${version.document_version_id}`);
    setCommandError(undefined);
    try {
      await reviewKnowledgeVersion(version.document_version_id, version.state_version);
      await load();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function buildRelease(values: ReleaseValues) {
    setBusy("create-release");
    setCommandError(undefined);
    try {
      await createKnowledgeRelease(values, crypto.randomUUID());
      setReleaseOpen(false);
      releaseForm.resetFields();
      await load();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function promote(release: KnowledgeRelease) {
    setBusy(`promote-${release.release_id}`);
    setCommandError(undefined);
    try {
      await promoteKnowledgeRelease(release.release_id);
      await load();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  function openRollback(release: KnowledgeRelease) {
    setRollbackTarget(release);
    rollbackForm.setFieldsValue({
      reason: `活动索引出现异常，回滚至已通过评测的稳定快照 v${release.version}`,
    });
  }

  async function rollback(values: RollbackValues) {
    if (!rollbackTarget) return;
    setBusy(`rollback-${rollbackTarget.release_id}`);
    setCommandError(undefined);
    try {
      const activeRelease = releases?.find(
        (item) => item.name === rollbackTarget.name && item.is_active,
      );
      if (!activeRelease) throw new Error("当前活动索引已变化，请刷新后重试");
      await rollbackKnowledgeRelease(
        rollbackTarget.release_id,
        rollbackTarget.activation_version,
        activeRelease.release_id,
        values.reason,
        crypto.randomUUID(),
      );
      setRollbackTarget(undefined);
      rollbackForm.resetFields();
      await load();
    } catch (error) {
      setCommandError(error);
      await load();
    } finally {
      setBusy(undefined);
    }
  }

  async function evaluateRelease(release: KnowledgeRelease) {
    setBusy(`evaluate-${release.release_id}`);
    setCommandError(undefined);
    try {
      await startKnowledgeIndexEvaluation(release.release_id, crypto.randomUUID());
      await load();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  function openDeletion(target: KnowledgeVersion) {
    setDeletionTarget(target);
    deletionForm.setFieldsValue({
      trigger: target.valid_to && new Date(target.valid_to) <= new Date()
        ? "RETENTION_EXPIRED"
        : "ACCESS_REVOKED",
      reason: "依据企业数据治理要求撤销该知识版本及全部检索派生数据",
    });
  }

  async function requestDeletion(values: DeletionValues) {
    if (!deletionTarget) return;
    setBusy(`delete-${deletionTarget.document_version_id}`);
    setCommandError(undefined);
    try {
      const deletion = await requestKnowledgeDeletion(
        deletionTarget.document_version_id,
        values,
        crypto.randomUUID(),
      );
      setSelectedDeletion(deletion);
      setDeletionTarget(undefined);
      deletionForm.resetFields();
      await load();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  async function retryDeletion(target: KnowledgeDeletion) {
    setBusy(`retry-deletion-${target.deletion_id}`);
    setCommandError(undefined);
    try {
      const deletion = await retryKnowledgeDeletion(target.deletion_id, target.version);
      setSelectedDeletion(deletion);
      await load();
    } catch (error) {
      setCommandError(error);
      await load();
    } finally {
      setBusy(undefined);
    }
  }

  async function runExpirySweep() {
    setBusy("expiry-sweep");
    setCommandError(undefined);
    try {
      await sweepExpiredKnowledge("按租户保留策略扫描并清理已到期知识版本");
      await load();
    } catch (error) {
      setCommandError(error);
    } finally {
      setBusy(undefined);
    }
  }

  const releasableVersions = (versions ?? []).filter((item) =>
    item.legal_actions.includes("INCLUDE_IN_INDEX_RELEASE"),
  );

  return (
    <AppShell>
      <div className="page-stack">
        <div>
          <Typography.Title level={2}>企业知识与索引发布工作台</Typography.Title>
          <Typography.Paragraph type="secondary">
            管理维修手册、服务公告和 SOP 的不可变版本、ACL、独立审核、候选索引评测及活动快照。
          </Typography.Paragraph>
        </div>
        <Alert
          showIcon
          type="info"
          title="未审核、未发布的知识不会进入诊断"
          description="创建人与审核人、生产发布人必须分离；候选索引通过召回、引用、ACL、设备范围、租户隔离、有效期和删除传播门禁后，发布事务才会原子切换活动索引。"
        />
        {commandError ? (
          <ErrorState error={commandError} onRetry={() => setCommandError(undefined)} />
        ) : null}
        <Row gutter={[16, 16]}>
          <Col xs={12} lg={6}><Card><Statistic title="待独立审核" value={totals.draft} /></Card></Col>
          <Col xs={12} lg={6}><Card><Statistic title="可构建快照" value={totals.reviewed} /></Card></Col>
          <Col xs={12} lg={6}><Card><Statistic title="已发布版本" value={totals.published} /></Card></Col>
          <Col xs={12} lg={6}><Card><Statistic title="活动索引" value={totals.active} /></Card></Col>
        </Row>
        <Space wrap>
          {collectionActions.includes("CREATE_KNOWLEDGE_DOCUMENT") ? (
            <Button type="primary" onClick={() => setCreateOpen(true)}>录入知识文档</Button>
          ) : null}
          {collectionActions.includes("CREATE_KNOWLEDGE_DOCUMENT") ? (
            <Button onClick={openFileIngestion}>上传 PDF / Markdown / TXT</Button>
          ) : null}
          {collectionActions.includes("CREATE_KNOWLEDGE_INDEX_RELEASE") ? (
            <Button
              disabled={releasableVersions.length === 0}
              onClick={() => setReleaseOpen(true)}
            >
              构建候选索引
            </Button>
          ) : null}
          {deletionActions.includes("RUN_KNOWLEDGE_EXPIRY_SWEEP") ? (
            <Popconfirm
              title="扫描并清理已到期知识"
              description="到期版本会立即退出检索，并异步清理对象、索引与缓存。"
              okText="开始扫描"
              cancelText="取消"
              onConfirm={() => void runExpirySweep()}
            >
              <Button loading={busy === "expiry-sweep"}>执行到期清理</Button>
            </Popconfirm>
          ) : null}
        </Space>
        {fileIngestion ? (
          <Alert
            showIcon
            type={knowledgeFileAlertType(fileIngestion.status)}
            title={
              <Space wrap>
                <span>{fileIngestion.source_filename}</span>
                <Tag color={statusColor(fileIngestion.status)}>{fileIngestion.status}</Tag>
              </Space>
            }
            description={
              fileIngestion.failure_reason
                ? `处理失败：${fileIngestion.failure_reason}`
                : fileIngestion.status === "DRAFT_READY"
                  ? `已生成待独立审核的知识版本 ${fileIngestion.document_version_id ?? ""}`
                  : "源文件已进入租户隔离区，后台 Worker 正在执行病毒扫描、PDF 原生文本提取与按需 PaddleOCR。"
            }
            action={
              <Button size="small" onClick={() => setFileOpen(true)}>
                查看任务
              </Button>
            }
          />
        ) : null}
        <Tabs
          items={[
            {
              key: "versions",
              label: "文档版本",
              children: versionError ? (
                <ErrorState error={versionError} onRetry={() => void load()} />
              ) : !versions ? (
                <LoadingState />
              ) : versions.length === 0 ? (
                <EmptyState description="当前租户尚未录入企业知识" />
              ) : (
                <Table
                  rowKey="document_version_id"
                  dataSource={versions}
                  pagination={{ pageSize: 10, showSizeChanger: false }}
                  scroll={{ x: 1300 }}
                  columns={[
                    {
                      title: "状态",
                      dataIndex: "status",
                      width: 110,
                      render: (value: string) => <Tag color={statusColor(value)}>{value}</Tag>,
                    },
                    { title: "标题", dataIndex: "title", width: 240, ellipsis: true },
                    { title: "版本", dataIndex: "version", width: 80, render: (value: number) => `v${value}` },
                    { title: "分类", dataIndex: "classification", width: 100 },
                    {
                      title: "设备型号",
                      dataIndex: "device_models",
                      width: 200,
                      render: (items: string[]) => items.length ? items.map((item) => <Tag key={item}>{item}</Tag>) : "全部",
                    },
                    { title: "创建人", dataIndex: "created_by_subject_id", width: 160, ellipsis: true },
                    { title: "审核人", dataIndex: "reviewed_by_subject_id", width: 160, ellipsis: true, render: (value?: string) => value ?? "—" },
                    {
                      title: "操作",
                      fixed: "right",
                      width: 390,
                      render: (_, row: KnowledgeVersion) => (
                        <Space wrap>
                          <Button type="link" onClick={() => setSelectedVersion(row)}>详情</Button>
                          {row.legal_actions.includes("CREATE_DOCUMENT_VERSION") ? (
                            <Button onClick={() => openNewVersion(row)}>追加版本</Button>
                          ) : null}
                          {row.legal_actions.includes("REVIEW_KNOWLEDGE_VERSION") ? (
                            <Button
                              type="primary"
                              loading={busy === `review-${row.document_version_id}`}
                              onClick={() => void review(row)}
                            >
                              审核通过
                            </Button>
                          ) : null}
                          {row.legal_actions.includes("REQUEST_KNOWLEDGE_DELETION") ? (
                            <Button danger onClick={() => openDeletion(row)}>
                              撤权 / 删除
                            </Button>
                          ) : null}
                        </Space>
                      ),
                    },
                  ]}
                />
              ),
            },
            {
              key: "file-jobs",
              label: `文件入库${fileJobs ? ` (${fileJobs.length})` : ""}`,
              children: fileError ? (
                <ErrorState error={fileError} onRetry={() => void load()} />
              ) : !fileJobs ? (
                <LoadingState />
              ) : fileJobs.length === 0 ? (
                <EmptyState description="尚未上传知识文件" />
              ) : (
                <Table
                  rowKey="ingestion_id"
                  dataSource={fileJobs}
                  pagination={{ pageSize: 10, showSizeChanger: false }}
                  scroll={{ x: 1250 }}
                  columns={[
                    {
                      title: "状态",
                      dataIndex: "status",
                      width: 130,
                      render: (value: string) => (
                        <Tag color={statusColor(value)}>{value}</Tag>
                      ),
                    },
                    { title: "标题", dataIndex: "title", width: 240, ellipsis: true },
                    { title: "文件", dataIndex: "source_filename", width: 220, ellipsis: true },
                    { title: "类型", dataIndex: "detected_mime", width: 150, render: (value?: string) => value ?? "—" },
                    { title: "处理次数", dataIndex: "attempt_count", width: 100 },
                    { title: "失败原因", dataIndex: "failure_reason", width: 250, ellipsis: true, render: (value?: string) => value ?? "—" },
                    { title: "更新时间", dataIndex: "updated_at", width: 190, render: formatTime },
                    {
                      title: "操作",
                      fixed: "right",
                      width: 230,
                      render: (_, row: KnowledgeFile) => (
                        <Space wrap>
                          <Button
                            type="link"
                            onClick={() => {
                              setFileIngestion(row);
                              setFileOpen(true);
                            }}
                          >
                            详情
                          </Button>
                          {row.legal_actions.some((action) =>
                            [
                              "REPROCESS_KNOWLEDGE_SOURCE_FILE",
                              "RETRY_KNOWLEDGE_INGESTION_DISPATCH",
                            ].includes(action),
                          ) ? (
                            <Button
                              loading={busy === `reprocess-${row.ingestion_id}`}
                              onClick={() => void reprocessFile(row)}
                            >
                              {row.status === "QUEUED" ? "重新派发" : "重新处理"}
                            </Button>
                          ) : null}
                        </Space>
                      ),
                    },
                  ]}
                />
              ),
            },
            {
              key: "releases",
              label: "索引发布",
              children: releaseError ? (
                <ErrorState error={releaseError} onRetry={() => void load()} />
              ) : !releases ? (
                <LoadingState />
              ) : releases.length === 0 ? (
                <EmptyState description="尚未构建索引快照" />
              ) : (
                <Table
                  rowKey="release_id"
                  dataSource={releases}
                  pagination={false}
                  scroll={{ x: 1380 }}
                  columns={[
                    {
                      title: "状态",
                      dataIndex: "status",
                      width: 120,
                      render: (value: string, row: KnowledgeRelease) => (
                        <Space><Tag color={statusColor(value)}>{value}</Tag>{row.is_active ? <Tag color="green">ACTIVE</Tag> : null}</Space>
                      ),
                    },
                    { title: "名称", dataIndex: "name", width: 220 },
                    { title: "版本", dataIndex: "version", width: 80, render: (value: number) => `v${value}` },
                    { title: "激活版本", dataIndex: "activation_version", width: 100 },
                    { title: "Chunk", dataIndex: "chunk_count", width: 90 },
                    {
                      title: "评测",
                      dataIndex: "evaluation_status",
                      width: 120,
                      render: (value?: string) =>
                        value ? <Tag color={statusColor(value)}>{value}</Tag> : "待运行",
                    },
                    {
                      title: "Recall@3",
                      width: 110,
                      render: (_, row: KnowledgeRelease) => {
                        const value = row.evaluation_metrics.recall_at_3;
                        return typeof value === "number" ? value.toFixed(3) : "—";
                      },
                    },
                    { title: "发布人", dataIndex: "published_by", width: 160, ellipsis: true, render: (value?: string) => value ?? "—" },
                    { title: "发布时间", dataIndex: "published_at", width: 190, render: formatTime },
                    {
                      title: "操作",
                      fixed: "right",
                      width: 410,
                      render: (_, row: KnowledgeRelease) => (
                        <Space wrap>
                          <Button type="link" onClick={() => setSelectedRelease(row)}>详情</Button>
                          {row.legal_actions.includes("RUN_KNOWLEDGE_INDEX_EVALUATION") ? (
                            <Button
                              loading={busy === `evaluate-${row.release_id}`}
                              onClick={() => void evaluateRelease(row)}
                            >
                              运行候选评测
                            </Button>
                          ) : null}
                          {row.legal_actions.includes("PROMOTE_KNOWLEDGE_RELEASE") ? (
                            <Button
                              type="primary"
                              loading={busy === `promote-${row.release_id}`}
                              onClick={() => void promote(row)}
                            >
                              发布活动索引
                            </Button>
                          ) : null}
                          {row.legal_actions.includes("ROLLBACK_KNOWLEDGE_RELEASE") ? (
                            <Button
                              danger
                              loading={busy === `rollback-${row.release_id}`}
                              onClick={() => openRollback(row)}
                            >
                              回滚至此版本
                            </Button>
                          ) : null}
                        </Space>
                      ),
                    },
                  ]}
                />
              ),
            },
            {
              key: "activation-history",
              label: `激活历史${activations ? ` (${activations.length})` : ""}`,
              children: activationError ? (
                <ErrorState error={activationError} onRetry={() => void load()} />
              ) : !activations ? (
                <LoadingState />
              ) : activations.length === 0 ? (
                <EmptyState description="尚无索引发布或回滚记录" />
              ) : (
                <Table
                  rowKey="activation_id"
                  dataSource={activations}
                  pagination={{ pageSize: 10, showSizeChanger: false }}
                  scroll={{ x: 1400 }}
                  columns={[
                    {
                      title: "动作",
                      dataIndex: "action",
                      width: 110,
                      render: (value: KnowledgeIndexActivation["action"]) => (
                        <Tag color={value === "ROLLBACK" ? "red" : "green"}>{value}</Tag>
                      ),
                    },
                    { title: "索引名称", dataIndex: "name", width: 210 },
                    {
                      title: "切换路径",
                      width: 390,
                      render: (_, row: KnowledgeIndexActivation) => (
                        <Typography.Text code copyable>
                          {row.from_release_id ?? "首次发布"} → {row.to_release_id}
                        </Typography.Text>
                      ),
                    },
                    { title: "操作人", dataIndex: "actor_subject_id", width: 180 },
                    { title: "原因", dataIndex: "reason", width: 280, ellipsis: true },
                    {
                      title: "派生面退役",
                      width: 150,
                      render: (_, row: KnowledgeIndexActivation) =>
                        `搜索 ${row.retired_search_profile_ids.length} / 图谱 ${row.revoked_graph_release_ids.length}`,
                    },
                    { title: "发生时间", dataIndex: "occurred_at", width: 190, render: formatTime },
                  ]}
                />
              ),
            },
            {
              key: "deletions",
              label: `删除传播${deletions ? ` (${deletions.length})` : ""}`,
              children: deletionError ? (
                <ErrorState error={deletionError} onRetry={() => void load()} />
              ) : !deletions ? (
                <LoadingState />
              ) : deletions.length === 0 ? (
                <EmptyState description="尚无删除、撤权或到期传播任务" />
              ) : (
                <Table
                  rowKey="deletion_id"
                  dataSource={deletions}
                  pagination={{ pageSize: 10, showSizeChanger: false }}
                  scroll={{ x: 1300 }}
                  columns={[
                    {
                      title: "状态",
                      dataIndex: "status",
                      width: 110,
                      render: (value: string) => (
                        <Tag color={statusColor(value)}>{value}</Tag>
                      ),
                    },
                    { title: "触发类型", dataIndex: "trigger", width: 170 },
                    {
                      title: "知识版本",
                      dataIndex: "document_version_id",
                      width: 260,
                      ellipsis: true,
                    },
                    {
                      title: "对象 / Chunk",
                      width: 140,
                      render: (_, row: KnowledgeDeletion) =>
                        `${Number(row.scope.source_object_count ?? 0)} / ${Number(row.scope.knowledge_chunk_count ?? 0)}`,
                    },
                    {
                      title: "模型影响",
                      width: 120,
                      render: (_, row: KnowledgeDeletion) => {
                        const ids = row.impact.model_release_ids;
                        return Array.isArray(ids) && ids.length > 0 ? (
                          <Tag color="red">需复核 {ids.length}</Tag>
                        ) : (
                          "无直接影响"
                        );
                      },
                    },
                    { title: "请求人", dataIndex: "requested_by_subject_id", width: 170 },
                    { title: "完成时间", dataIndex: "completed_at", width: 190, render: formatTime },
                    {
                      title: "操作",
                      fixed: "right",
                      width: 210,
                      render: (_, row: KnowledgeDeletion) => (
                        <Space>
                          <Button type="link" onClick={() => setSelectedDeletion(row)}>
                            影响报告
                          </Button>
                          {row.legal_actions.includes("RETRY_KNOWLEDGE_DELETION") ? (
                            <Button
                              loading={busy === `retry-deletion-${row.deletion_id}`}
                              onClick={() => void retryDeletion(row)}
                            >
                              重试
                            </Button>
                          ) : null}
                        </Space>
                      ),
                    },
                  ]}
                />
              ),
            },
          ]}
        />
      </div>

      <Modal
        title="录入文本或 Markdown 知识"
        open={createOpen}
        width={920}
        onCancel={() => setCreateOpen(false)}
        onOk={() => documentForm.submit()}
        confirmLoading={busy === "create-document"}
        okText="创建 DRAFT"
      >
        <Alert
          type="warning"
          showIcon
          title="此入口只接收已经确认来源的文本或 Markdown。PDF、扫描件和复杂表格必须走异步解析 Worker。"
          style={{ marginBottom: 16 }}
        />
        <Form
          form={documentForm}
          layout="vertical"
          onFinish={(values) => void createDocument(values)}
          initialValues={{
            classification: "internal",
            acl_subject_ids: [],
            acl_roles: ["field_engineer", "domain_expert"],
            device_families: [],
            device_models: [],
            valid_from: toLocalInput(new Date().toISOString()),
          }}
        >
          <Row gutter={16}>
            <Col span={12}><Required name="title" label="文档标题"><Input /></Required></Col>
            <Col span={12}><Required name="source_uri" label="来源 URI"><Input placeholder="https://、s3:// 或 minio://" /></Required></Col>
            <Col span={6}><Required name="classification" label="分类"><Select options={[{ value: "internal" }, { value: "restricted" }]} /></Required></Col>
            <Col span={9}><Required name="valid_from" label="生效时间"><Input type="datetime-local" /></Required></Col>
            <Col span={9}><Form.Item name="valid_to" label="失效时间"><Input type="datetime-local" /></Form.Item></Col>
            <Col span={12}><Form.Item name="acl_roles" label="角色 ACL"><Select mode="multiple" options={roleOptions} /></Form.Item></Col>
            <Col span={12}><Form.Item name="acl_subject_ids" label="用户 ACL"><Select mode="tags" tokenSeparators={[","]} /></Form.Item></Col>
            <Col span={12}><Form.Item name="device_families" label="设备族"><Select mode="tags" tokenSeparators={[","]} /></Form.Item></Col>
            <Col span={12}><Form.Item name="device_models" label="设备型号"><Select mode="tags" tokenSeparators={[","]} /></Form.Item></Col>
            <Col span={24}><Required name="content" label="正文"><Input.TextArea rows={12} showCount maxLength={500000} /></Required></Col>
          </Row>
        </Form>
      </Modal>

      <Modal
        title="异步解析企业知识文件"
        open={fileOpen}
        width={900}
        destroyOnHidden
        onCancel={() => setFileOpen(false)}
        onOk={() => fileForm.submit()}
        confirmLoading={busy === "ingest-file"}
        okButtonProps={{
          disabled: Boolean(
            fileIngestion && !["AWAITING_UPLOAD"].includes(fileIngestion.status),
          ),
        }}
        okText={fileIngestion?.status === "AWAITING_UPLOAD" ? "重试上传" : "上传并开始解析"}
      >
        <Alert
          type="info"
          showIcon
          title="文件先进入隔离区，扫描通过后才会复制到 clean 区并生成 DRAFT"
          description="支持 UTF-8 TXT/Markdown 和最多 200 页、25 MB 的 PDF。文本不足的 PDF 页面交给发布版本绑定的 PaddleOCR；解析结果仍须由另一位人员审核后才能发布。"
          style={{ marginBottom: 16 }}
        />
        {fileIngestion ? <FileIngestionDetails ingestion={fileIngestion} /> : null}
        <Form
          form={fileForm}
          layout="vertical"
          onFinish={(values) => void ingestFile(values)}
          initialValues={{
            classification: "internal",
            acl_subject_ids: [],
            acl_roles: ["field_engineer", "domain_expert"],
            device_families: [],
            device_models: [],
            valid_from: toLocalInput(new Date().toISOString()),
          }}
          disabled={Boolean(fileIngestion && fileIngestion.status !== "AWAITING_UPLOAD")}
        >
          <Row gutter={16}>
            <Col span={16}><Required name="title" label="文档标题"><Input /></Required></Col>
            <Col span={8}><Required name="classification" label="分类"><Select options={[{ value: "internal" }, { value: "restricted" }]} /></Required></Col>
          </Row>
          <ScopeFields />
          <Form.Item label="源文件" required>
            <Upload.Dragger
              accept=".txt,.md,.markdown,.pdf,text/plain,text/markdown,application/pdf"
              maxCount={1}
              beforeUpload={(file) => {
                setSourceFile(file);
                return false;
              }}
              onRemove={() => {
                setSourceFile(undefined);
              }}
            >
              <Typography.Text strong>点击或拖入文件</Typography.Text>
              <br />
              <Typography.Text type="secondary">单文件不超过 25 MB</Typography.Text>
            </Upload.Dragger>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={`为 ${versionTarget?.title ?? "文档"} 追加不可变版本`}
        open={Boolean(versionTarget)}
        width={900}
        onCancel={() => setVersionTarget(undefined)}
        onOk={() => versionForm.submit()}
        confirmLoading={Boolean(versionTarget && busy === `version-${versionTarget.document_id}`)}
        okText="创建新 DRAFT"
      >
        <Form form={versionForm} layout="vertical" onFinish={(values) => void createVersion(values)}>
          <ScopeFields />
          <Required name="content" label="新版本正文"><Input.TextArea rows={14} showCount maxLength={500000} /></Required>
        </Form>
      </Modal>

      <Modal
        title="构建候选知识索引"
        open={releaseOpen}
        onCancel={() => setReleaseOpen(false)}
        onOk={() => releaseForm.submit()}
        confirmLoading={busy === "create-release"}
        okText="构建 CANDIDATE"
      >
        <Alert
          type="info"
          showIcon
          title="同一文档只能选择一个版本；候选快照不会自动进入诊断。"
          style={{ marginBottom: 16 }}
        />
        <Form form={releaseForm} layout="vertical" onFinish={(values) => void buildRelease(values)}>
          <Required name="name" label="稳定发布名称">
            <Input placeholder="enterprise-service-manuals" />
          </Required>
          <Required name="document_version_ids" label="已审核文档版本">
            <Select
              mode="multiple"
              showSearch
              optionFilterProp="label"
              options={releasableVersions.map((item) => ({
                value: item.document_version_id,
                label: `${item.title} · v${item.version} · ${item.status}`,
              }))}
            />
          </Required>
        </Form>
      </Modal>

      <Modal
        title={`回滚索引：${rollbackTarget?.name ?? ""} v${rollbackTarget?.version ?? ""}`}
        open={Boolean(rollbackTarget)}
        width={760}
        onCancel={() => {
          setRollbackTarget(undefined);
          rollbackForm.resetFields();
        }}
        onOk={() => rollbackForm.submit()}
        confirmLoading={Boolean(
          rollbackTarget && busy === `rollback-${rollbackTarget.release_id}`,
        )}
        okButtonProps={{ danger: true }}
        okText="确认原子回滚"
      >
        <Alert
          type="warning"
          showIcon
          title="回滚前会重新校验目标快照的评测、文档状态和引用完整性"
          description="事务将原子切换活动别名；基于当前较新索引的 OpenSearch Shadow 与 GraphRAG 派生版本会被退役，查询立即回落到目标快照支持的 SQL / pgvector 路径。操作记录不可覆盖。"
          style={{ marginBottom: 16 }}
        />
        <Form
          form={rollbackForm}
          layout="vertical"
          onFinish={(values) => void rollback(values)}
        >
          <Required name="reason" label="回滚原因与处置依据">
            <Input.TextArea rows={4} minLength={10} maxLength={1000} showCount />
          </Required>
        </Form>
      </Modal>

      <Modal
        title={`撤权或删除：${deletionTarget?.title ?? "知识版本"}`}
        open={Boolean(deletionTarget)}
        width={760}
        onCancel={() => setDeletionTarget(undefined)}
        onOk={() => deletionForm.submit()}
        confirmLoading={Boolean(
          deletionTarget && busy === `delete-${deletionTarget.document_version_id}`,
        )}
        okButtonProps={{ danger: true }}
        okText="确认退出检索并开始传播"
      >
        <Alert
          type="error"
          showIcon
          title="这是合规删除操作，提交后知识会立即停止召回"
          description="Temporal 将继续清理 MinIO 原件、全文/向量 Chunk、引用锚点、Redis 缓存和直接关联的训练候选；审计仅保留哈希、范围与结果。若已发布模型受影响，任务会停在 PARTIAL 等待人工处置。"
          style={{ marginBottom: 16 }}
        />
        <Form
          form={deletionForm}
          layout="vertical"
          onFinish={(values) => void requestDeletion(values)}
        >
          <Required name="trigger" label="触发类型">
            <Select
              options={[
                { value: "ACCESS_REVOKED", label: "权限撤销" },
                { value: "LEGAL_DELETION", label: "合法删除请求" },
                {
                  value: "RETENTION_EXPIRED",
                  label: "保留期到期",
                  disabled: !(
                    deletionTarget?.valid_to &&
                    new Date(deletionTarget.valid_to) <= new Date()
                  ),
                },
              ]}
            />
          </Required>
          <Required name="reason" label="合规依据与原因">
            <Input.TextArea rows={4} maxLength={1024} showCount />
          </Required>
        </Form>
      </Modal>

      <Drawer
        title="文档版本与授权事实"
        size={720}
        open={Boolean(selectedVersion)}
        onClose={() => setSelectedVersion(undefined)}
      >
        {selectedVersion ? <VersionDetails version={selectedVersion} /> : null}
      </Drawer>
      <Drawer
        title="索引快照与评测证据"
        size={820}
        open={Boolean(selectedRelease)}
        onClose={() => setSelectedRelease(undefined)}
      >
        {selectedRelease ? (
          <ReleaseDetails release={selectedRelease} />
        ) : null}
      </Drawer>
      <Drawer
        title="删除传播影响与验证报告"
        size={820}
        open={Boolean(selectedDeletion)}
        onClose={() => setSelectedDeletion(undefined)}
      >
        {selectedDeletion ? <DeletionDetails deletion={selectedDeletion} /> : null}
      </Drawer>
    </AppShell>
  );
}

function ReleaseDetails({ release }: { release: KnowledgeRelease }) {
  const gateEntries = Object.entries(release.evaluation_gate_results);
  return (
    <Space direction="vertical" size="large" style={{ width: "100%" }}>
      {release.evaluation_status === "FAILED" ? (
        <Alert
          type="error"
          showIcon
          title="候选索引未通过发布门禁"
          description={release.evaluation_failure_codes.join("、") || "评测失败"}
        />
      ) : null}
      {release.status === "CANDIDATE" ? (
        <Alert
          type="success"
          showIcon
          title="候选索引已通过自动评测"
          description="仍需由非内容创建人执行生产发布，发布事务才会切换活动别名。"
        />
      ) : null}
      <Descriptions bordered size="small" column={1}>
        <Descriptions.Item label="Release ID">
          <Typography.Text copyable code>{release.release_id}</Typography.Text>
        </Descriptions.Item>
        <Descriptions.Item label="名称 / 版本">{release.name} / v{release.version}</Descriptions.Item>
        <Descriptions.Item label="状态"><Tag color={statusColor(release.status)}>{release.status}</Tag></Descriptions.Item>
        <Descriptions.Item label="构建人">{release.created_by_subject_id ?? "历史导入"}</Descriptions.Item>
        <Descriptions.Item label="活动快照">{release.is_active ? "是" : "否"}</Descriptions.Item>
        <Descriptions.Item label="激活并发版本">{release.activation_version}</Descriptions.Item>
        <Descriptions.Item label="允许回滚">{release.rollback_eligible ? "是" : "否"}</Descriptions.Item>
        <Descriptions.Item label="Chunk 数">{release.chunk_count}</Descriptions.Item>
        <Descriptions.Item label="内容哈希"><Typography.Text copyable code>{release.content_checksum}</Typography.Text></Descriptions.Item>
        <Descriptions.Item label="评测任务">
          {release.evaluation_id ? (
            <Typography.Text copyable code>{release.evaluation_id}</Typography.Text>
          ) : "—"}
        </Descriptions.Item>
        <Descriptions.Item label="评测状态">
          {release.evaluation_status ? (
            <Tag color={statusColor(release.evaluation_status)}>{release.evaluation_status}</Tag>
          ) : "未运行"}
        </Descriptions.Item>
        <Descriptions.Item label="评测完成">{formatTime(release.evaluated_at)}</Descriptions.Item>
        <Descriptions.Item label="发布人">{release.published_by ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="发布时间">{formatTime(release.published_at)}</Descriptions.Item>
      </Descriptions>
      {Object.keys(release.evaluation_metrics).length > 0 ? (
        <div>
          <Typography.Title level={5}>检索评测指标</Typography.Title>
          <FactDescriptions facts={release.evaluation_metrics} />
        </div>
      ) : null}
      {gateEntries.length > 0 ? (
        <div>
          <Typography.Title level={5}>发布门禁</Typography.Title>
          <Descriptions bordered size="small" column={1}>
            {gateEntries.map(([name, passed]) => (
              <Descriptions.Item key={name} label={factLabel(name)}>
                <Tag color={passed ? "green" : "red"}>{passed ? "PASSED" : "FAILED"}</Tag>
              </Descriptions.Item>
            ))}
          </Descriptions>
        </div>
      ) : null}
    </Space>
  );
}

function DeletionDetails({ deletion }: { deletion: KnowledgeDeletion }) {
  return (
    <Space direction="vertical" size="large" style={{ width: "100%" }}>
      {deletion.status === "PARTIAL" ? (
        <Alert
          type="warning"
          showIcon
          title="自动清理未全部闭环"
          description={deletion.failure_reason ?? "存在需要人工复核的下游影响"}
        />
      ) : null}
      <Descriptions bordered size="small" column={1}>
        <Descriptions.Item label="任务 ID">
          <Typography.Text copyable code>{deletion.deletion_id}</Typography.Text>
        </Descriptions.Item>
        <Descriptions.Item label="工作流">
          <Typography.Text copyable code>{deletion.workflow_id}</Typography.Text>
        </Descriptions.Item>
        <Descriptions.Item label="状态">
          <Tag color={statusColor(deletion.status)}>{deletion.status}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="触发类型">{deletion.trigger}</Descriptions.Item>
        <Descriptions.Item label="原因">{deletion.reason}</Descriptions.Item>
        <Descriptions.Item label="目标版本">
          <Typography.Text copyable code>{deletion.document_version_id}</Typography.Text>
        </Descriptions.Item>
        <Descriptions.Item label="正文哈希">
          <Typography.Text copyable code>{deletion.target_content_checksum}</Typography.Text>
        </Descriptions.Item>
        <Descriptions.Item label="执行次数">{deletion.attempt_count}</Descriptions.Item>
        <Descriptions.Item label="请求 / 完成">
          {formatTime(deletion.requested_at)} / {formatTime(deletion.completed_at)}
        </Descriptions.Item>
      </Descriptions>

      <div>
        <Typography.Title level={5}>传播范围</Typography.Title>
        <FactDescriptions facts={deletion.scope} />
      </div>
      <div>
        <Typography.Title level={5}>下游影响</Typography.Title>
        <FactDescriptions facts={deletion.impact} />
      </div>
      <div>
        <Typography.Title level={5}>分层验证</Typography.Title>
        <Descriptions bordered size="small" column={1}>
          {Object.entries(deletion.verification).map(([key, value]) => (
            <Descriptions.Item key={key} label={factLabel(key)}>
              <Tag color={verificationColor(String(value))}>{String(value)}</Tag>
            </Descriptions.Item>
          ))}
        </Descriptions>
      </div>
    </Space>
  );
}

function FactDescriptions({ facts }: { facts: Record<string, unknown> }) {
  return (
    <Descriptions bordered size="small" column={1}>
      {Object.entries(facts).map(([key, value]) => (
        <Descriptions.Item key={key} label={factLabel(key)}>
          {formatFact(value)}
        </Descriptions.Item>
      ))}
    </Descriptions>
  );
}

function ScopeFields() {
  return (
    <Row gutter={16}>
      <Col span={12}><Form.Item name="acl_roles" label="角色 ACL"><Select mode="multiple" options={roleOptions} /></Form.Item></Col>
      <Col span={12}><Form.Item name="acl_subject_ids" label="用户 ACL"><Select mode="tags" tokenSeparators={[","]} /></Form.Item></Col>
      <Col span={12}><Form.Item name="device_families" label="设备族"><Select mode="tags" tokenSeparators={[","]} /></Form.Item></Col>
      <Col span={12}><Form.Item name="device_models" label="设备型号"><Select mode="tags" tokenSeparators={[","]} /></Form.Item></Col>
      <Col span={12}><Required name="valid_from" label="生效时间"><Input type="datetime-local" /></Required></Col>
      <Col span={12}><Form.Item name="valid_to" label="失效时间"><Input type="datetime-local" /></Form.Item></Col>
    </Row>
  );
}

function VersionDetails({ version }: { version: KnowledgeVersion }) {
  return (
    <Descriptions bordered size="small" column={1}>
      <Descriptions.Item label="文档 ID"><Typography.Text copyable code>{version.document_id}</Typography.Text></Descriptions.Item>
      <Descriptions.Item label="版本 ID"><Typography.Text copyable code>{version.document_version_id}</Typography.Text></Descriptions.Item>
      <Descriptions.Item label="标题 / 版本">{version.title} / v{version.version}</Descriptions.Item>
      <Descriptions.Item label="状态"><Tag color={statusColor(version.status)}>{version.status}</Tag></Descriptions.Item>
      <Descriptions.Item label="来源"><Typography.Text copyable>{version.source_uri}</Typography.Text></Descriptions.Item>
      <Descriptions.Item label="分类">{version.classification}</Descriptions.Item>
      <Descriptions.Item label="角色 ACL">{version.acl_roles.join(", ") || "全部"}</Descriptions.Item>
      <Descriptions.Item label="用户 ACL">{version.acl_subject_ids.join(", ") || "未限定"}</Descriptions.Item>
      <Descriptions.Item label="设备范围">{[...version.device_families, ...version.device_models].join(", ") || "全部"}</Descriptions.Item>
      <Descriptions.Item label="有效期">{formatTime(version.valid_from)} 至 {formatTime(version.valid_to)}</Descriptions.Item>
      <Descriptions.Item label="源文件哈希"><Typography.Text copyable code>{version.source_checksum}</Typography.Text></Descriptions.Item>
      <Descriptions.Item label="规范化内容哈希"><Typography.Text copyable code>{version.content_checksum}</Typography.Text></Descriptions.Item>
      <Descriptions.Item label="解析器版本">{version.parser_version}</Descriptions.Item>
      <Descriptions.Item label="抽取方法">
        {formatExtractionMethods(version.extraction_metadata)}
      </Descriptions.Item>
      <Descriptions.Item label="创建人">{version.created_by_subject_id ?? "历史导入"}</Descriptions.Item>
      <Descriptions.Item label="审核人 / 时间">{version.reviewed_by_subject_id ?? "—"} / {formatTime(version.reviewed_at)}</Descriptions.Item>
      <Descriptions.Item label="状态版本">{version.state_version}</Descriptions.Item>
    </Descriptions>
  );
}

function FileIngestionDetails({ ingestion }: { ingestion: KnowledgeFile }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <Descriptions bordered size="small" column={2} style={{ marginBottom: 16 }}>
        <Descriptions.Item label="任务状态">
          <Tag color={statusColor(ingestion.status)}>{ingestion.status}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="状态版本">{ingestion.version}</Descriptions.Item>
        <Descriptions.Item label="任务 ID" span={2}>
          <Typography.Text copyable code>{ingestion.ingestion_id}</Typography.Text>
        </Descriptions.Item>
        <Descriptions.Item label="检测类型">{ingestion.detected_mime ?? "等待上传"}</Descriptions.Item>
        <Descriptions.Item label="文件大小">
          {ingestion.size_bytes === null ? "—" : `${(ingestion.size_bytes / 1024).toFixed(1)} KiB`}
        </Descriptions.Item>
        <Descriptions.Item label="解析器" span={2}>{ingestion.parser_version ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="结果" span={2}>
          {ingestion.failure_reason ?? ingestion.document_version_id ?? "处理中"}
        </Descriptions.Item>
      </Descriptions>
      <Typography.Title level={5}>处理尝试记录</Typography.Title>
      <Table
        size="small"
        rowKey="attempt_id"
        dataSource={ingestion.attempts}
        pagination={false}
        scroll={{ x: 760 }}
        columns={[
          { title: "次数", dataIndex: "attempt_number", width: 70, render: (value: number) => `#${value}` },
          { title: "状态", dataIndex: "status", width: 120, render: (value: string) => <Tag color={statusColor(value)}>{value}</Tag> },
          { title: "失败原因", dataIndex: "failure_reason", width: 230, ellipsis: true, render: (value?: string) => value ?? "—" },
          { title: "解析器", dataIndex: "parser_version", width: 200, ellipsis: true, render: (value?: string) => value ?? "—" },
          { title: "开始", dataIndex: "started_at", width: 180, render: formatTime },
          { title: "完成", dataIndex: "completed_at", width: 180, render: formatTime },
        ]}
      />
    </div>
  );
}

function Required({
  name,
  label,
  children,
}: {
  name: string;
  label: string;
  children: ReactNode;
}) {
  return <Form.Item name={name} label={label} rules={[{ required: true }]}>{children}</Form.Item>;
}

function scopePayload(values: KnowledgeScopeValues) {
  return {
    acl_subject_ids: values.acl_subject_ids ?? [],
    acl_roles: values.acl_roles ?? [],
    device_families: values.device_families ?? [],
    device_models: values.device_models ?? [],
    valid_from: new Date(values.valid_from).toISOString(),
    valid_to: values.valid_to ? new Date(values.valid_to).toISOString() : null,
  };
}

function declaredKnowledgeMime(file: File): "text/plain" | "text/markdown" | "application/pdf" {
  const extension = file.name.toLocaleLowerCase();
  if (extension.endsWith(".pdf") && (!file.type || file.type === "application/pdf")) {
    return "application/pdf";
  }
  if (
    (extension.endsWith(".md") || extension.endsWith(".markdown")) &&
    (!file.type || ["text/plain", "text/markdown"].includes(file.type))
  ) {
    return "text/markdown";
  }
  if (extension.endsWith(".txt") && (!file.type || file.type === "text/plain")) {
    return "text/plain";
  }
  throw new Error("文件扩展名与浏览器识别的 MIME 类型不匹配");
}

function formatExtractionMethods(metadata: Record<string, unknown> | null) {
  if (!metadata) return "直接录入";
  const pages = metadata.pages;
  if (!Array.isArray(pages)) return "—";
  const methods = new Set(
    pages
      .map((page) => (typeof page === "object" && page ? Reflect.get(page, "method") : undefined))
      .filter((method): method is string => typeof method === "string"),
  );
  const base = Array.from(methods).join(", ") || "—";
  const structures = metadata.structures;
  if (!Array.isArray(structures)) return base;
  const figures = structures.filter(
    (structure) => typeof structure === "object"
      && structure !== null
      && Reflect.get(structure, "kind") === "figure",
  );
  if (figures.length === 0) return base;
  const analyzed = figures.filter((structure) => {
    const analysis = Reflect.get(structure, "analysis");
    return typeof analysis === "object"
      && analysis !== null
      && ["ANALYZED", "NO_FINDINGS"].includes(String(Reflect.get(analysis, "analysis_status")));
  }).length;
  return `${base} · 内嵌图片语义 ${analyzed}/${figures.length}`;
}

function toLocalInput(value: string) {
  const date = new Date(value);
  const shifted = new Date(date.getTime() - date.getTimezoneOffset() * 60_000);
  return shifted.toISOString().slice(0, 16);
}

function formatTime(value?: string | null) {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "—";
}

function formatFact(value: unknown) {
  if (Array.isArray(value)) {
    return value.length > 0 ? value.map(String).join(", ") : "无";
  }
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

function factLabel(value: string) {
  const labels: Record<string, string> = {
    source_object_count: "源对象数",
    knowledge_chunk_count: "知识 Chunk 数",
    citation_anchor_count: "引用锚点数",
    index_release_count: "受影响索引数",
    training_candidate_count: "训练候选数",
    index_release_ids: "索引 Release",
    diagnosis_run_ids: "历史诊断任务",
    model_release_ids: "模型 Release",
    training_candidate_ids: "训练候选",
    revoked_training_candidate_ids: "已撤销训练候选",
    model_disposition: "模型处置",
    historical_audit_policy: "历史审计策略",
    deleted_source_object_count: "已删除源对象数",
    invalidated_cache_key_count: "已失效缓存键数",
    postgresql: "PostgreSQL 正文",
    object_store: "MinIO / S3",
    full_text_index: "全文索引",
    vector_index: "向量索引",
    citation_anchors: "引用锚点",
    tenant_cache: "租户缓存",
    training_candidates: "训练候选",
    graph_index: "GraphRAG",
    data_lake: "数据湖",
    label_platform: "标注平台",
    published_model_review: "已发布模型复核",
    document_count: "文档数",
    chunk_count: "Chunk 数",
    citation_count: "引用锚点数",
    retrieval_case_count: "召回用例数",
    recall_at_3: "Recall@3",
    acl_probe_count: "ACL 隔离用例数",
    unauthorized_result_count: "未授权泄漏数",
    device_scope_probe_count: "设备范围用例数",
    device_scope_leak_count: "设备范围泄漏数",
    pending_deletion_count: "未完成删除任务数",
    structure_integrity: "索引结构完整性",
    citation_integrity: "引用完整性",
    embedding_integrity: "向量完整性",
    retrieval_recall: "真实检索召回",
    acl_isolation: "ACL 隔离",
    device_scope_isolation: "设备范围隔离",
    validity_window: "知识有效期",
    tenant_isolation: "租户隔离",
    deletion_propagation: "删除传播闭环",
  };
  return labels[value] ?? value;
}

function verificationColor(value: string) {
  if (["VERIFIED", "NOT_REQUIRED", "NO_DIRECT_LINEAGE"].includes(value)) return "green";
  if (["REQUIRED", "PENDING", "NOT_CONFIGURED"].includes(value)) return "gold";
  return "red";
}

function statusColor(value: string) {
  if (["PUBLISHED", "ACTIVE", "DRAFT_READY", "COMPLETED", "PASSED"].includes(value)) return "green";
  if (["REVIEWED", "RUNNING", "EVALUATING"].includes(value)) return "blue";
  if (["CANDIDATE", "QUEUED", "AWAITING_UPLOAD", "PARTIAL", "EVALUATION_PENDING"].includes(value)) return "gold";
  if (["FAILED", "REJECTED", "DELETED", "REVOKED", "EXPIRED", "PURGED"].includes(value)) return "red";
  return "default";
}

function knowledgeFileAlertType(status: string): "info" | "success" | "error" {
  if (status === "DRAFT_READY") return "success";
  if (["FAILED", "REJECTED"].includes(status)) return "error";
  return "info";
}
