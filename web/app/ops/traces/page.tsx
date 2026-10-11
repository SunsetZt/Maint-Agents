"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Drawer,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  getAgentTrace,
  listAgentTraces,
  type AgentTraceDetail,
  type AgentTraceNode,
  type AgentTraceSummary,
} from "@/lib/api/client";

const healthLabels: Record<string, string> = {
  HEALTHY: "正常",
  RUNNING: "运行中",
  ATTENTION: "需关注",
  FAILED: "失败",
};

const healthColors: Record<string, string> = {
  HEALTHY: "green",
  RUNNING: "blue",
  ATTENTION: "orange",
  FAILED: "red",
};

const categoryLabels: Record<string, string> = {
  WORKFLOW: "工作流",
  AGENT: "Agent",
  AGENT_EVENT: "Agent 事件",
  RETRIEVAL: "检索",
  TOOL: "工具",
  MODEL: "模型",
  BUSINESS: "业务",
};

const statusOptions = [
  { label: "全部状态", value: "" },
  { label: "排队中", value: "QUEUED" },
  { label: "运行中", value: "RUNNING" },
  { label: "已完成", value: "COMPLETED" },
  { label: "等待专家", value: "WAITING_EXPERT" },
  { label: "需要补充信息", value: "NEEDS_INFORMATION" },
  { label: "失败", value: "FAILED" },
];

export default function AgentTraceOperationsPage() {
  const [items, setItems] = useState<AgentTraceSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [projectionVersion, setProjectionVersion] = useState<string>();
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [detail, setDetail] = useState<AgentTraceDetail>();
  const [detailError, setDetailError] = useState<unknown>();
  const [detailLoading, setDetailLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedAgentRunId, setSelectedAgentRunId] = useState<string>();

  const load = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true);
    else setLoading(true);
    try {
      const result = await listAgentTraces({
        status: status || undefined,
        limit: pageSize,
        offset: (page - 1) * pageSize,
      });
      setItems(result.items);
      setTotal(result.total);
      setProjectionVersion(result.projectionVersion);
      setRequestId(result.requestId);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [page, pageSize, status]);

  useEffect(() => {
    void load();
  }, [load]);

  const pageMetrics = useMemo(() => ({
    failed: items.filter((item) => item.health === "FAILED").length,
    attention: items.filter((item) => item.health === "ATTENTION").length,
    partial: items.filter((item) => item.evidence_status === "PARTIAL").length,
    tokens: items.reduce(
      (sum, item) => sum + item.prompt_tokens + item.completion_tokens,
      0,
    ),
  }), [items]);

  async function openTrace(agentRunId: string) {
    setSelectedAgentRunId(agentRunId);
    setDrawerOpen(true);
    setDetail(undefined);
    setDetailError(undefined);
    setDetailLoading(true);
    try {
      const result = await getAgentTrace(agentRunId);
      setDetail(result.trace);
    } catch (cause) {
      setDetailError(cause);
    } finally {
      setDetailLoading(false);
    }
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Space align="start" style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>Agent 与 AI 全链路观测中心</Typography.Title>
            <Typography.Paragraph type="secondary">
              将诊断、Temporal、Agent 事件、授权检索、工具调用、Context Manifest、模型推理和售后工单投影到同一条业务链路，支持失败节点定位、耗时分解与跨系统关联。
            </Typography.Paragraph>
          </div>
          <Button loading={refreshing} onClick={() => void load(true)}>刷新链路</Button>
        </Space>

        <Alert
          showIcon
          type="info"
          message="隐私最小化业务追踪"
          description="本中心只读取持久化审计元数据。Context Manifest 展示内容摘要、证据/工具引用、预算和裁剪事实，不展示 Prompt、模型输出、客户正文、检索内容、工具参数/结果或 Chain-of-Thought。Tempo 与 Langfuse 继续承担底层遥测，本页面不重复存储 Span。"
        />

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {loading && !error ? <LoadingState label="正在关联 Agent 与 AI 业务链路" /> : null}

        {!loading && !error ? (
          <>
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="匹配链路总数" value={total} /></Card></Col>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="本页失败 / 需关注" value={pageMetrics.failed + pageMetrics.attention} suffix={` / ${items.length}`} valueStyle={pageMetrics.failed ? { color: "#cf1322" } : undefined} /></Card></Col>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="本页推理 Token" value={pageMetrics.tokens} /></Card></Col>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="本页证据不完整" value={pageMetrics.partial} valueStyle={pageMetrics.partial ? { color: "#d46b08" } : undefined} /></Card></Col>
            </Row>

            <Card
              title="业务追踪列表"
              extra={(
                <Space>
                  <Typography.Text type="secondary">投影 {projectionVersion ?? "-"}</Typography.Text>
                  <Select
                    value={status}
                    options={statusOptions}
                    style={{ width: 160 }}
                    onChange={(value) => {
                      setStatus(value);
                      setPage(1);
                    }}
                  />
                </Space>
              )}
            >
              <Table<AgentTraceSummary>
                rowKey="agent_run_id"
                dataSource={items}
                scroll={{ x: 1250 }}
                pagination={{
                  current: page,
                  pageSize,
                  total,
                  showSizeChanger: true,
                  pageSizeOptions: [10, 20, 50, 100],
                  showTotal: (value) => `共 ${value} 条`,
                  onChange: (nextPage, nextSize) => {
                    setPage(nextSize !== pageSize ? 1 : nextPage);
                    setPageSize(nextSize);
                  },
                }}
                locale={{ emptyText: "当前筛选条件下没有 Agent 业务链路" }}
                columns={[
                  {
                    title: "健康度",
                    dataIndex: "health",
                    width: 100,
                    render: (value: string) => <Tag color={healthColors[value]}>{healthLabels[value] ?? value}</Tag>,
                  },
                  {
                    title: "Agent Run",
                    dataIndex: "agent_run_id",
                    width: 230,
                    render: (value: string) => <Button type="link" style={{ padding: 0 }} onClick={() => void openTrace(value)}>{value}</Button>,
                  },
                  { title: "诊断", dataIndex: "diagnosis_run_id", width: 190 },
                  { title: "状态", dataIndex: "agent_status", width: 140 },
                  {
                    title: "节点",
                    width: 170,
                    render: (_, item) => `${item.event_count} 事件 / ${item.tool_call_count} 工具 / ${item.inference_count} 推理`,
                  },
                  {
                    title: "耗时",
                    dataIndex: "duration_ms",
                    width: 110,
                    render: duration,
                  },
                  {
                    title: "模型 Release",
                    dataIndex: "model_release_ids",
                    width: 220,
                    render: (values: string[]) => values.length ? values.map((value) => <Tag key={value}>{value}</Tag>) : "-",
                  },
                  {
                    title: "关联工单",
                    dataIndex: "work_order_ids",
                    width: 120,
                    render: (values: string[]) => values.length,
                  },
                  {
                    title: "开始时间",
                    dataIndex: "started_at",
                    width: 180,
                    render: timestamp,
                  },
                  {
                    title: "证据",
                    dataIndex: "evidence_status",
                    width: 100,
                    render: (value: string) => <Tag color={value === "COMPLETE" ? "green" : "orange"}>{value === "COMPLETE" ? "完整" : "部分"}</Tag>,
                  },
                ]}
              />
            </Card>
            <Typography.Text type="secondary">最近请求 ID：{requestId ?? "-"}</Typography.Text>
          </>
        ) : null}
      </div>

      <Drawer
        title={detail ? `链路详情 · ${detail.summary.agent_run_id}` : "链路详情"}
        width={1000}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        destroyOnHidden
      >
        {detailLoading ? <LoadingState label="正在生成安全业务追踪投影" /> : null}
        {detailError ? <ErrorState error={detailError} onRetry={() => selectedAgentRunId && void openTrace(selectedAgentRunId)} /> : null}
        {detail ? <TraceDetailView detail={detail} /> : null}
      </Drawer>
    </AppShell>
  );
}

function TraceDetailView({ detail }: { detail: AgentTraceDetail }) {
  const summary = detail.summary;
  return (
    <Space direction="vertical" size="large" style={{ width: "100%" }}>
      <Alert showIcon type="info" message={detail.privacy_notice} />
      {detail.missing_correlations.length ? (
        <Alert
          showIcon
          type="warning"
          message="链路关联证据不完整"
          description={detail.missing_correlations.map(missingCorrelationLabel).join("；")}
        />
      ) : null}
      <Descriptions bordered size="small" column={2}>
        <Descriptions.Item label="健康度"><Tag color={healthColors[summary.health]}>{healthLabels[summary.health] ?? summary.health}</Tag></Descriptions.Item>
        <Descriptions.Item label="Agent 状态">{summary.agent_status}</Descriptions.Item>
        <Descriptions.Item label="诊断 Run">{summary.diagnosis_run_id}</Descriptions.Item>
        <Descriptions.Item label="事件 / 工具 / 推理">{summary.event_count} / {summary.tool_call_count} / {summary.inference_count}</Descriptions.Item>
        <Descriptions.Item label="Temporal Workflow">{summary.workflow_id}</Descriptions.Item>
        <Descriptions.Item label="Incident">{summary.incident_id}</Descriptions.Item>
        <Descriptions.Item label="Trace ID" span={2}>{summary.trace_ids.join(", ") || "未产生模型 Trace ID"}</Descriptions.Item>
        <Descriptions.Item label="Token">{summary.prompt_tokens} 输入 / {summary.completion_tokens} 输出</Descriptions.Item>
        <Descriptions.Item label="总耗时">{duration(summary.duration_ms)}</Descriptions.Item>
        <Descriptions.Item label="关联工单" span={2}>{summary.work_order_ids.join(", ") || "尚未关联工单"}</Descriptions.Item>
      </Descriptions>

      <Card title={`节点时间线（${detail.nodes.length}）`}>
        <Table<AgentTraceNode>
          rowKey="node_id"
          size="small"
          pagination={false}
          dataSource={detail.nodes}
          scroll={{ x: 850 }}
          expandable={{
            rowExpandable: (node) => Object.keys(node.attributes).length > 0,
            expandedRowRender: (node) => <AttributeTable attributes={node.attributes} />,
          }}
          columns={[
            {
              title: "类型",
              dataIndex: "category",
              width: 110,
              render: (value: string) => <Tag>{categoryLabels[value] ?? value}</Tag>,
            },
            { title: "节点", dataIndex: "name", width: 260 },
            {
              title: "状态",
              dataIndex: "status",
              width: 130,
              render: (value: string) => <Tag color={nodeStatusColor(value)}>{value}</Tag>,
            },
            { title: "开始", dataIndex: "started_at", width: 180, render: timestamp },
            { title: "耗时", dataIndex: "duration_ms", width: 110, render: duration },
            { title: "父节点", dataIndex: "parent_node_id", ellipsis: true },
          ]}
        />
      </Card>
    </Space>
  );
}

function AttributeTable({ attributes }: { attributes: Record<string, unknown> }) {
  const rows = Object.entries(attributes).map(([key, value]) => ({ key, value }));
  return (
    <Table
      rowKey="key"
      size="small"
      pagination={false}
      dataSource={rows}
      columns={[
        { title: "安全属性", dataIndex: "key", width: 250 },
        { title: "值", dataIndex: "value", render: safeValue },
      ]}
    />
  );
}

function safeValue(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

function timestamp(value: string | null | undefined): string {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "-";
}

function duration(value: number | null | undefined): string {
  if (value == null) return "运行中";
  if (value < 1000) return `${value.toFixed(0)} ms`;
  if (value < 60_000) return `${(value / 1000).toFixed(2)} s`;
  return `${(value / 60_000).toFixed(2)} min`;
}

function nodeStatusColor(value: string): string {
  if (["FAILED", "CANCELLED"].includes(value)) return "red";
  if (["RUNNING", "PENDING", "QUEUED"].includes(value)) return "blue";
  if (["ATTENTION", "NEEDS_INFORMATION", "WAITING_EXPERT"].includes(value)) return "orange";
  if (["SUCCEEDED", "COMPLETED", "RECORDED"].includes(value)) return "green";
  return "default";
}

function missingCorrelationLabel(value: string): string {
  return {
    agent_events_missing: "缺少 Agent 事件",
    retrieval_completion_missing: "检索尚未形成完成证据",
    model_inference_correlation_missing: "模型推理关联记录缺失",
    tool_call_correlation_missing: "工具调用关联记录缺失",
    telemetry_trace_id_missing: "模型遥测 Trace ID 缺失",
    context_manifest_missing: "模型调用缺少 Context Manifest",
  }[value] ?? value;
}
