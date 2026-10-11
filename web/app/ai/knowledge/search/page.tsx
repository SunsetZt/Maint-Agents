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
  InputNumber,
  Modal,
  Progress,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type KnowledgeRelease,
  type KnowledgeSearchProfile,
  type KnowledgeSearchRebuild,
  type KnowledgeSearchResult,
  activateKnowledgeSearchProfileShadow,
  createKnowledgeSearchProfile,
  deactivateKnowledgeSearchProfileShadow,
  evaluateKnowledgeSearchProfile,
  listKnowledgeReleases,
  listKnowledgeSearchProfiles,
  listKnowledgeSearchRebuilds,
  queryKnowledgeSearchProfile,
  requestKnowledgeSearchRebuild,
  retryKnowledgeSearchRebuild,
} from "@/lib/api/client";

type CreateValues = { source_index_release_id: string; name: string };
type EvaluationValues = { cases_json: string };
type QueryValues = {
  profile_name: string;
  query_text: string;
  device_family?: string;
  device_model: string;
  limit: number;
};

const exampleCases = JSON.stringify(
  [
    {
      query: "轴承高温 润滑油压力",
      device_family: "pump",
      device_model: "PUMP-X100",
      expected_citation_ids: ["citation-bearing-overheat"],
      limit: 6,
    },
    {
      query: "E77 报警如何停机复核",
      device_family: "pump",
      device_model: "PUMP-X100",
      expected_citation_ids: ["citation-e77-safe-stop"],
      limit: 6,
    },
    {
      query: "更换主轴承需要哪些备件",
      device_family: "pump",
      device_model: "PUMP-X100",
      expected_citation_ids: ["citation-bearing-parts"],
      limit: 6,
    },
  ],
  null,
  2,
);

export default function KnowledgeSearchPage() {
  const [profiles, setProfiles] = useState<KnowledgeSearchProfile[]>();
  const [rebuilds, setRebuilds] = useState<KnowledgeSearchRebuild[]>([]);
  const [releases, setReleases] = useState<KnowledgeRelease[]>([]);
  const [selected, setSelected] = useState<KnowledgeSearchProfile>();
  const [searchResult, setSearchResult] = useState<KnowledgeSearchResult>();
  const [error, setError] = useState<unknown>();
  const [commandError, setCommandError] = useState<unknown>();
  const [busy, setBusy] = useState<string>();
  const [createOpen, setCreateOpen] = useState(false);
  const [evaluationOpen, setEvaluationOpen] = useState(false);
  const [createForm] = Form.useForm<CreateValues>();
  const [evaluationForm] = Form.useForm<EvaluationValues>();
  const [queryForm] = Form.useForm<QueryValues>();

  const load = useCallback(async () => {
    const [profileResult, releaseResult, rebuildResult] = await Promise.allSettled([
      listKnowledgeSearchProfiles(),
      listKnowledgeReleases("PUBLISHED"),
      listKnowledgeSearchRebuilds(),
    ]);
    if (profileResult.status === "fulfilled") {
      setProfiles(profileResult.value.profiles);
      setSelected((current) =>
        current
          ? profileResult.value.profiles.find(
              (item) => item.search_profile_id === current.search_profile_id,
            ) ?? current
          : current,
      );
      setError(undefined);
    } else {
      setError(profileResult.reason);
    }
    if (releaseResult.status === "fulfilled") {
      setReleases(releaseResult.value.releases);
    }
    if (rebuildResult.status === "fulfilled") {
      setRebuilds(rebuildResult.value.rebuilds);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!rebuilds.some((item) => item.status === "QUEUED" || item.status === "RUNNING")) {
      return undefined;
    }
    const timer = window.setInterval(() => void load(), 2500);
    return () => window.clearInterval(timer);
  }, [load, rebuilds]);

  const run = useCallback(
    async (key: string, command: () => Promise<KnowledgeSearchProfile>) => {
      setBusy(key);
      setCommandError(undefined);
      try {
        const result = await command();
        setSelected(result);
        await load();
        return result;
      } catch (cause) {
        setCommandError(cause);
        return undefined;
      } finally {
        setBusy(undefined);
      }
    },
    [load],
  );

  const runRebuild = useCallback(
    async (key: string, command: () => Promise<KnowledgeSearchRebuild>) => {
      setBusy(key);
      setCommandError(undefined);
      try {
        const result = await command();
        setRebuilds((current) => [
          result,
          ...current.filter((item) => item.rebuild_job_id !== result.rebuild_job_id),
        ]);
        return result;
      } catch (cause) {
        setCommandError(cause);
        return undefined;
      } finally {
        setBusy(undefined);
      }
    },
    [],
  );

  const activeProfiles = useMemo(
    () => profiles?.filter((item) => item.status === "ACTIVE_SHADOW") ?? [],
    [profiles],
  );
  const projectedDocuments = useMemo(
    () => profiles?.reduce((total, item) => total + item.document_count, 0) ?? 0,
    [profiles],
  );
  const passedProfiles = useMemo(
    () => profiles?.filter((item) => item.evaluation?.status === "PASSED").length ?? 0,
    [profiles],
  );
  const activeRebuildCount = useMemo(
    () => rebuilds.filter((item) => item.status === "QUEUED" || item.status === "RUNNING").length,
    [rebuilds],
  );
  const selectedActiveRebuild = useMemo(
    () =>
      selected
        ? rebuilds.find(
            (item) =>
              item.search_profile_id === selected.search_profile_id &&
              (item.status === "QUEUED" || item.status === "RUNNING"),
          )
        : undefined,
    [rebuilds, selected],
  );

  if (!profiles && !error) {
    return (
      <AppShell>
        <LoadingState label="正在加载企业知识搜索 Profile" />
      </AppShell>
    );
  }

  return (
    <AppShell>
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <Space align="start" style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>企业知识搜索 Profile</Typography.Title>
            <Typography.Paragraph type="secondary">
              将已发布知识索引通过生产 Embedding 模型重建到 OpenSearch，使用 BM25 与向量召回进行
              PostgreSQL 基线 A/B；查询向量必须与索引的发布、制品和维度一致，再由 CrossEncoder
              Reranker 重排已授权候选。通过独立评测后只进入 Shadow 查询面，不会自动替换诊断主链路。
            </Typography.Paragraph>
          </div>
          <Space>
            <Link href="/ai/knowledge">返回知识治理</Link>
            <Button type="primary" onClick={() => setCreateOpen(true)}>
              创建搜索候选
            </Button>
          </Space>
        </Space>

        <Alert
          type="info"
          showIcon
          message="默认检索事实源仍是 PostgreSQL 全文检索 + pgvector"
          description="OpenSearch 仅在真实语料规模、复杂筛选、分词、聚合或延迟证据证明收益后才可进一步申请生产切换。本页面的激活只开放受控 Shadow 查询；租户、ACL、密级、设备范围和有效期均在两条召回路径之前过滤，并由 PostgreSQL 防御性复核。"
        />
        {commandError ? <ErrorState error={commandError} /> : null}
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}

        <Row gutter={16}>
          <Col span={6}>
            <Card><Statistic title="候选 / 历史 Profile" value={profiles?.length ?? 0} /></Card>
          </Col>
          <Col span={6}>
            <Card><Statistic title="活动 Shadow" value={activeProfiles.length} /></Card>
          </Col>
          <Col span={6}>
            <Card><Statistic title="通过 A/B 评测" value={passedProfiles} /></Card>
          </Col>
          <Col span={6}>
            <Card><Statistic title="累计投影文档" value={projectedDocuments} /></Card>
          </Col>
        </Row>

        <Card
          title="Temporal 索引重建进度中心"
          extra={<Tag color={activeRebuildCount ? "processing" : "default"}>运行中 {activeRebuildCount}</Tag>}
        >
          {!rebuilds.length ? (
            <EmptyState description="尚无异步重建任务；从 Profile 详情提交后可在这里查看阶段、进度和失败原因。" />
          ) : (
            <Table
              rowKey="rebuild_job_id"
              dataSource={rebuilds.slice(0, 10)}
              pagination={false}
              columns={[
                {
                  title: "Profile",
                  dataIndex: "search_profile_id",
                  width: 210,
                  render: (value) =>
                    profiles?.find((item) => item.search_profile_id === value)?.name ?? value,
                },
                {
                  title: "状态 / 阶段",
                  width: 190,
                  render: (_, row) => (
                    <Space direction="vertical" size={2}>
                      <Tag color={rebuildStatusColor(row.status)}>{row.status}</Tag>
                      <Typography.Text type="secondary">{rebuildStageLabel(row.stage)}</Typography.Text>
                    </Space>
                  ),
                },
                {
                  title: "进度",
                  width: 260,
                  render: (_, row) => (
                    <Progress
                      percent={row.progress_percent}
                      status={
                        row.status === "FAILED"
                          ? "exception"
                          : row.status === "COMPLETED"
                            ? "success"
                            : "active"
                      }
                      format={(percent) =>
                        row.total_items
                          ? `${percent}% · ${row.completed_items}/${row.total_items}`
                          : `${percent}%`
                      }
                    />
                  ),
                },
                { title: "尝试", dataIndex: "attempt_count", width: 70 },
                {
                  title: "失败原因",
                  dataIndex: "failure_code",
                  ellipsis: true,
                  render: (value) => value ?? "—",
                },
                {
                  title: "操作",
                  width: 90,
                  render: (_, row) =>
                    row.legal_actions.includes("RETRY") ? (
                      <Button
                        size="small"
                        loading={busy === `retry-${row.rebuild_job_id}`}
                        onClick={() =>
                          void runRebuild(`retry-${row.rebuild_job_id}`, () =>
                            retryKnowledgeSearchRebuild(row.rebuild_job_id, row.version),
                          )
                        }
                      >
                        重试
                      </Button>
                    ) : null,
                },
              ]}
            />
          )}
        </Card>

        <Card title="Profile 发布流水线">
          {!profiles?.length ? (
            <EmptyState description="尚无 OpenSearch 候选；请从一个已发布知识索引创建首个搜索 Profile。" />
          ) : (
            <Table
              rowKey="search_profile_id"
              dataSource={profiles}
              pagination={false}
              onRow={(record) => ({ onClick: () => setSelected(record) })}
              columns={[
                {
                  title: "Profile",
                  dataIndex: "name",
                  render: (value, row) => (
                    <Space><Typography.Text strong>{value}</Typography.Text><Tag>v{row.version}</Tag></Space>
                  ),
                },
                {
                  title: "状态",
                  dataIndex: "status",
                  render: (value) => <Tag color={statusColor(value)}>{value}</Tag>,
                },
                { title: "来源索引", dataIndex: "source_index_release_id", ellipsis: true },
                { title: "投影文档", dataIndex: "document_count" },
                {
                  title: "基线 / 候选召回",
                  render: (_, row) =>
                    `${formatMetric(row.metrics.baseline_expected_recall_at_k)} / ${formatMetric(row.metrics.candidate_expected_recall_at_k)}`,
                },
                {
                  title: "越权候选",
                  render: (_, row) => String(row.metrics.candidate_unauthorized_count ?? "—"),
                },
                {
                  title: "操作",
                  render: (_, row) => (
                    <Button
                      size="small"
                      onClick={(event) => {
                        event.stopPropagation();
                        setSelected(row);
                      }}
                    >
                      查看与推进
                    </Button>
                  ),
                },
              ]}
            />
          )}
        </Card>

        <Card title="活动 Shadow 检索工作台">
          <Form<QueryValues>
            form={queryForm}
            layout="vertical"
            initialValues={{ limit: 6 }}
            onFinish={async (values) => {
              setBusy("query");
              setCommandError(undefined);
              try {
                setSearchResult(await queryKnowledgeSearchProfile(values));
              } catch (cause) {
                setCommandError(cause);
              } finally {
                setBusy(undefined);
              }
            }}
          >
            <Row gutter={16}>
              <Col span={6}>
                <Form.Item name="profile_name" label="活动 Profile" rules={[{ required: true }]}>
                  <Select options={activeProfiles.map((item) => ({ value: item.name, label: `${item.name} v${item.version}` }))} />
                </Form.Item>
              </Col>
              <Col span={5}>
                <Form.Item name="device_model" label="设备型号" rules={[{ required: true }]}>
                  <Input placeholder="PUMP-X100" />
                </Form.Item>
              </Col>
              <Col span={5}>
                <Form.Item name="device_family" label="设备家族">
                  <Input placeholder="pump" />
                </Form.Item>
              </Col>
              <Col span={4}>
                <Form.Item name="limit" label="结果上限">
                  <InputNumber min={1} max={20} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            </Row>
            <Form.Item name="query_text" label="检索问题" rules={[{ required: true }]}>
              <Input.TextArea rows={3} placeholder="例如：轴承高温时如何检查润滑油压力？" />
            </Form.Item>
            <Button type="primary" htmlType="submit" loading={busy === "query"}>
              执行受控混合检索
            </Button>
          </Form>

          {searchResult ? (
            <Space direction="vertical" size="middle" style={{ width: "100%", marginTop: 20 }}>
              <Alert
                type={
                  searchResult.embedding.status === "APPLIED"
                    ? "success"
                    : searchResult.embedding.status === "FALLBACK_SPARSE"
                      ? "warning"
                      : "info"
                }
                showIcon
                message={embeddingTitle(searchResult)}
                description={embeddingDescription(searchResult)}
              />
              <Alert
                type={
                  searchResult.rerank.status === "APPLIED"
                    ? "success"
                    : searchResult.rerank.status === "FALLBACK"
                      ? "warning"
                      : "info"
                }
                showIcon
                message={rerankTitle(searchResult)}
                description={rerankDescription(searchResult)}
              />
              <Table
                rowKey="citation_id"
                dataSource={searchResult.evidence}
                pagination={false}
                columns={[
                  { title: "标题", dataIndex: "title", width: 220 },
                  { title: "证据正文", dataIndex: "content" },
                  { title: "引用", dataIndex: "citation_id", width: 210, ellipsis: true },
                  { title: "Sparse", dataIndex: "sparse_rank", width: 80 },
                  { title: "Vector", dataIndex: "vector_rank", width: 80 },
                  {
                    title: "RRF",
                    dataIndex: "fused_score",
                    width: 90,
                    render: (value) => Number(value).toFixed(6),
                  },
                  {
                    title: "CrossEncoder",
                    dataIndex: "reranker_score",
                    width: 120,
                    render: (value) =>
                      typeof value === "number" ? value.toFixed(6) : "—",
                  },
                  {
                    title: "最终排名",
                    dataIndex: "final_rank",
                    width: 90,
                    render: (value) => value ?? "—",
                  },
                ]}
              />
            </Space>
          ) : null}
        </Card>
      </Space>

      <Modal
        title="创建 OpenSearch 候选 Profile"
        open={createOpen}
        onCancel={() => setCreateOpen(false)}
        onOk={() => createForm.submit()}
        confirmLoading={busy === "create"}
      >
        <Form
          form={createForm}
          layout="vertical"
          onFinish={async (values) => {
            const result = await run("create", () =>
              createKnowledgeSearchProfile(values, crypto.randomUUID()),
            );
            if (result) {
              setCreateOpen(false);
              createForm.resetFields();
            }
          }}
        >
          <Form.Item name="source_index_release_id" label="已发布知识索引" rules={[{ required: true }]}>
            <Select options={releases.map((item) => ({ value: item.release_id, label: `${item.name} v${item.version} · ${item.release_id}` }))} />
          </Form.Item>
          <Form.Item name="name" label="Profile 名称" rules={[{ required: true }]}>
            <Input placeholder="enterprise-manual-search" />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="提交 PostgreSQL / OpenSearch A/B 评测"
        width={820}
        open={evaluationOpen}
        onCancel={() => setEvaluationOpen(false)}
        onOk={() => evaluationForm.submit()}
        confirmLoading={busy === "evaluate"}
      >
        <Alert
          type="warning"
          showIcon
          message="评测人必须与候选创建人不同"
          description="至少提交 3 个黄金用例。门禁检查投影一致性、召回不退化、候选召回下限、零越权候选和独立评测人。"
          style={{ marginBottom: 16 }}
        />
        <Form
          form={evaluationForm}
          layout="vertical"
          initialValues={{ cases_json: exampleCases }}
          onFinish={async ({ cases_json }) => {
            if (!selected) return;
            try {
              const parsedCases = JSON.parse(cases_json) as Array<{
                query: string;
                device_family?: string;
                device_model: string;
                expected_citation_ids: string[];
                limit?: number;
              }>;
              const cases = parsedCases.map((item) => ({ ...item, limit: item.limit ?? 6 }));
              const result = await run("evaluate", () =>
                evaluateKnowledgeSearchProfile(
                  selected.search_profile_id,
                  selected.state_version,
                  { cases },
                ),
              );
              if (result) setEvaluationOpen(false);
            } catch {
              setCommandError(new Error("评测用例必须是合法 JSON 数组"));
            }
          }}
        >
          <Form.Item name="cases_json" label="黄金用例 JSON" rules={[{ required: true }]}>
            <Input.TextArea rows={18} style={{ fontFamily: "monospace" }} />
          </Form.Item>
        </Form>
      </Modal>

      <Drawer
        title={selected ? `${selected.name} v${selected.version}` : "Profile 详情"}
        width={720}
        open={Boolean(selected)}
        onClose={() => setSelected(undefined)}
        extra={selected ? (
          <Space>
            {selected.legal_actions.includes("SYNC") ? (
              <Button
                loading={busy === "rebuild"}
                disabled={Boolean(selectedActiveRebuild)}
                onClick={() =>
                  void runRebuild("rebuild", () =>
                    requestKnowledgeSearchRebuild(
                      selected.search_profile_id,
                      selected.state_version,
                      crypto.randomUUID(),
                    ),
                  )
                }
              >
                {selectedActiveRebuild ? "重建进行中" : "异步重建投影"}
              </Button>
            ) : null}
            {selected.legal_actions.includes("EVALUATE") ? (
              <Button onClick={() => setEvaluationOpen(true)}>提交 A/B 评测</Button>
            ) : null}
            {selected.legal_actions.includes("ACTIVATE_SHADOW") ? (
              <Button
                type="primary"
                loading={busy === "activate"}
                onClick={() => void run("activate", () => activateKnowledgeSearchProfileShadow(selected.search_profile_id, selected.state_version))}
              >
                激活 Shadow
              </Button>
            ) : null}
            {selected.legal_actions.includes("DEACTIVATE_SHADOW") ? (
              <Button
                danger
                loading={busy === "deactivate"}
                onClick={() => void run("deactivate", () => deactivateKnowledgeSearchProfileShadow(selected.search_profile_id, selected.state_version))}
              >
                回退到 PostgreSQL
              </Button>
            ) : null}
          </Space>
        ) : null}
      >
        {selected ? (
          <Space direction="vertical" size="large" style={{ width: "100%" }}>
            <Descriptions bordered column={1} size="small">
              <Descriptions.Item label="状态"><Tag color={statusColor(selected.status)}>{selected.status}</Tag></Descriptions.Item>
              <Descriptions.Item label="Profile ID">{selected.search_profile_id}</Descriptions.Item>
              <Descriptions.Item label="来源索引">{selected.source_index_release_id}</Descriptions.Item>
              <Descriptions.Item label="后端 / Schema">{selected.backend} · {selected.schema_version}</Descriptions.Item>
              <Descriptions.Item label="服务端索引名">{selected.index_name}</Descriptions.Item>
              <Descriptions.Item label="文档 / 状态版本">{selected.document_count} / {selected.state_version}</Descriptions.Item>
              <Descriptions.Item label="Manifest">{selected.manifest_hash ?? "尚未同步"}</Descriptions.Item>
              <Descriptions.Item label="Embedding 绑定">
                {selected.embedding_component_model_id
                  ? `${selected.embedding_component_model_id} · ${selected.embedding_dimension ?? "—"} 维 · ${selected.embedding_model_release_id ?? "—"}`
                  : "Legacy 16 维确定性向量"}
              </Descriptions.Item>
            </Descriptions>

            {selectedActiveRebuild ? (
              <Card size="small" title="当前 Temporal 重建任务">
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Space>
                    <Tag color="processing">{selectedActiveRebuild.status}</Tag>
                    <Typography.Text>{rebuildStageLabel(selectedActiveRebuild.stage)}</Typography.Text>
                    <Typography.Text type="secondary">
                      第 {selectedActiveRebuild.attempt_count} 次执行
                    </Typography.Text>
                  </Space>
                  <Progress
                    percent={selectedActiveRebuild.progress_percent}
                    status="active"
                  />
                </Space>
              </Card>
            ) : null}

            <Card size="small" title="A/B 质量与安全证据">
              <Row gutter={[16, 16]}>
                <Col span={8}><Statistic title="PostgreSQL 召回" value={metricPercent(selected.metrics.baseline_expected_recall_at_k)} suffix="%" /></Col>
                <Col span={8}><Statistic title="OpenSearch 召回" value={metricPercent(selected.metrics.candidate_expected_recall_at_k)} suffix="%" /></Col>
                <Col span={8}><Statistic title="召回变化" value={metricPercent(selected.metrics.recall_delta)} suffix="pp" /></Col>
              </Row>
              <Space wrap style={{ marginTop: 16 }}>
                {Object.entries(selected.gate_results).map(([name, passed]) => (
                  <Tag key={name} color={passed ? "green" : "red"}>{name}: {passed ? "PASS" : "FAIL"}</Tag>
                ))}
              </Space>
              {selected.failure_codes.length ? (
                <Alert type="error" showIcon message="门禁失败" description={selected.failure_codes.join("、")} style={{ marginTop: 16 }} />
              ) : null}
            </Card>

            <Alert
              type={selected.status === "ACTIVE_SHADOW" ? "success" : "info"}
              showIcon
              message={selected.status === "ACTIVE_SHADOW" ? "已进入受控 Shadow 查询面" : "尚未影响生产诊断主链路"}
              description="进一步替换 PostgreSQL 主检索前，仍需真实目标负载下的 p95 延迟、容量、故障回退与双写截止证据。"
            />
          </Space>
        ) : null}
      </Drawer>
    </AppShell>
  );
}

function statusColor(status: string) {
  if (status === "ACTIVE_SHADOW") return "green";
  if (status === "EVALUATED" || status === "SYNCED") return "blue";
  if (status === "REJECTED" || status === "REVOKED") return "red";
  if (status === "RETIRED") return "default";
  return "gold";
}

function rebuildStatusColor(status: KnowledgeSearchRebuild["status"]) {
  if (status === "COMPLETED") return "green";
  if (status === "FAILED") return "red";
  if (status === "RUNNING") return "processing";
  return "gold";
}

function rebuildStageLabel(stage: string) {
  const labels: Record<string, string> = {
    QUEUED: "等待 Temporal Worker",
    VALIDATING: "校验 Profile 与版本",
    LOADING_SOURCE: "读取发布知识切片",
    EMBEDDING: "生产 Embedding 批量推理",
    PREPARING_INDEX: "准备索引文档",
    INDEXING: "写入 OpenSearch",
    VERIFYING: "核对外部文档数量",
    PERSISTING: "固化 SQL 投影与模型绑定",
    COMPLETED: "重建完成",
    FAILED: "重建失败",
  };
  return labels[stage] ?? stage;
}

function formatMetric(value: unknown) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

function metricPercent(value: unknown) {
  return typeof value === "number" ? Number((value * 100).toFixed(1)) : 0;
}

function rerankTitle(result: KnowledgeSearchResult) {
  if (result.rerank.status === "APPLIED") {
    return `生产 Reranker 已应用 · ${result.evidence.length} 条证据`;
  }
  if (result.rerank.status === "FALLBACK") {
    return `Reranker 已显式回退到 RRF · ${result.evidence.length} 条证据`;
  }
  return `${result.backend} RRF · ${result.evidence.length} 条证据`;
}

function rerankDescription(result: KnowledgeSearchResult) {
  const base = `来源索引 ${result.source_index_release_id}；防护阶段 ${result.guard_stage}；候选 ${result.rerank.candidate_count} / 返回 ${result.rerank.returned_count}`;
  if (result.rerank.status === "APPLIED") {
    return `${base}；发布 ${result.rerank.model_release_id ?? "—"}；模型 ${result.rerank.component_model_id ?? "—"}；延迟 ${result.rerank.latency_ms?.toFixed(1) ?? "—"} ms`;
  }
  if (result.rerank.status === "FALLBACK") {
    return `${base}；回退原因 ${result.rerank.fallback_reason ?? "unknown"}`;
  }
  return `${base}；当前未配置生产 Reranker`;
}

function embeddingTitle(result: KnowledgeSearchResult) {
  if (result.embedding.status === "APPLIED") {
    return `生产 Embedding 已应用 · ${result.embedding.dimension ?? "—"} 维`;
  }
  if (result.embedding.status === "FALLBACK_SPARSE") {
    return "Embedding 不可用 · 本次仅执行 BM25 稀疏召回";
  }
  return "Legacy 确定性 Embedding";
}

function embeddingDescription(result: KnowledgeSearchResult) {
  if (result.embedding.status === "APPLIED") {
    return `发布 ${result.embedding.model_release_id ?? "—"}；模型 ${result.embedding.component_model_id ?? "—"}；制品 ${result.embedding.artifact_content_hash ?? "—"}；延迟 ${result.embedding.latency_ms?.toFixed(1) ?? "—"} ms`;
  }
  if (result.embedding.status === "FALLBACK_SPARSE") {
    return `索引仍保持不变；回退原因 ${result.embedding.fallback_reason ?? "unknown"}`;
  }
  return "该 Profile 在生产 Embedding 接入前构建，继续使用与旧索引一致的 16 维向量。";
}
