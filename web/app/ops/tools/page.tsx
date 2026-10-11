"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  getToolGovernance,
  type GovernedTool,
  type RecentToolCall,
  type ToolGovernance,
} from "@/lib/api/client";

const riskColors: Record<string, string> = {
  T0: "green",
  T1: "blue",
  T2: "orange",
  T3: "red",
};

const riskLabels: Record<string, string> = {
  T0: "只读查询",
  T1: "低风险可逆",
  T2: "强制人工审批",
  T3: "仅外部安全交接",
};

const policyLabels: Record<string, string> = {
  AUTO_READ_ONLY: "授权后自动只读执行",
  CONTROLLED_REVERSIBLE: "受控可逆执行",
  HUMAN_APPROVAL_REQUIRED: "独立人工审批后执行",
  EXTERNAL_HANDOFF_ONLY: "平台禁止执行，仅外部交接",
  DENY_UNKNOWN_RISK: "未知风险，默认拒绝",
};

const anomalyLabels: Record<string, string> = {
  unregistered_tool_audit_present: "发现无法映射到当前 Registry 的历史调用记录",
  t2_execution_without_approval_path: "发现 T2 工具绕过审批路径的成功调用记录",
  t3_platform_execution_violation: "发现 T3 工具未按外部安全交接处理",
  authority_evidence_mismatch: "近期调用的数据源权威声明与合同不一致",
  tool_failure_ratio_high: "统计窗口内工具失败率达到 20%",
  mcp_schema_binding_missing: "MCP 工具缺少本地输入 Schema 摘要绑定",
};

export default function ToolGovernancePage() {
  const [governance, setGovernance] = useState<ToolGovernance>();
  const [windowHours, setWindowHours] = useState(24);
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true);
    else setLoading(true);
    try {
      const result = await getToolGovernance({ window_hours: windowHours, recent_limit: 100 });
      setGovernance(result.governance);
      setRequestId(result.requestId);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [windowHours]);

  useEffect(() => {
    void load();
  }, [load]);

  const failureRatio = useMemo(() => {
    if (!governance?.totals.call_count) return 0;
    return governance.totals.failed_count / governance.totals.call_count;
  }, [governance]);

  return (
    <AppShell>
      <div className="page-stack">
        <Space align="start" style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>企业工具治理中心</Typography.Title>
            <Typography.Paragraph type="secondary">
              审查 Agent 当前可见工具的版本、JSON Schema、风险等级、授权动作、审批与补偿策略、MCP 服务身份与最小 Scope、企业事实权威来源，以及当前租户的近期调用证据。
            </Typography.Paragraph>
          </div>
          <Space>
            <Select
              value={windowHours}
              style={{ width: 150 }}
              options={[
                { label: "最近 24 小时", value: 24 },
                { label: "最近 7 天", value: 168 },
                { label: "最近 30 天", value: 720 },
              ]}
              onChange={setWindowHours}
            />
            <Button loading={refreshing} onClick={() => void load(true)}>刷新治理证据</Button>
          </Space>
        </Space>

        <Alert
          showIcon
          type="info"
          message="只读治理入口"
          description="本页面不提供工具试运行或参数编辑。T2 必须继续走提案、独立审批和执行前复检；T3 无论普通审批结果如何，都只能交接到外部认证安全系统。"
        />

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {loading && !error ? <LoadingState label="正在加载 Tool Registry 与调用审计" /> : null}

        {governance && !loading && !error ? (
          <>
            {governance.anomalies.map((item) => (
              <Alert
                key={item}
                showIcon
                type={item.includes("violation") || item.includes("mismatch") ? "error" : "warning"}
                message={anomalyLabels[item] ?? item}
              />
            ))}

            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="已注册工具" value={governance.totals.registered_tool_count} suffix={` / ${governance.totals.authority_declared_count} 有权威合同`} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="窗口内调用" value={governance.totals.call_count} suffix={` / ${governance.totals.active_agent_run_count} Agent`} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="工具失败率" value={failureRatio * 100} precision={2} suffix="%" valueStyle={failureRatio >= 0.2 ? { color: "#cf1322" } : undefined} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="审批拦截 / 外部交接" value={governance.totals.approval_required_count} suffix={` / ${governance.totals.external_handoff_count}`} /></Card>
              </Col>
            </Row>

            <Card>
              <Space wrap>
                <Typography.Text strong>风险目录：</Typography.Text>
                {Object.entries(governance.totals.risk_counts).map(([risk, count]) => (
                  <Tag key={risk} color={riskColors[risk]}>{risk} {riskLabels[risk]} · {count}</Tag>
                ))}
                <Tag color={governance.totals.mcp_tool_count ? "cyan" : "default"}>
                  MCP {governance.totals.mcp_tool_count} 工具 / {governance.totals.mcp_server_count} 服务
                </Tag>
                <Typography.Text type="secondary">策略版本 {governance.policy_version}</Typography.Text>
              </Space>
            </Card>

            <Tabs
              items={[
                {
                  key: "registry",
                  label: `Registry（${governance.tools.length}）`,
                  children: <RegistryTable tools={governance.tools} />,
                },
                {
                  key: "calls",
                  label: `近期调用（${governance.recent_calls.length}）`,
                  children: <RecentCallsTable calls={governance.recent_calls} />,
                },
              ]}
            />
            <Typography.Text type="secondary">
              统计窗口：{timestamp(governance.window_start)} 至 {timestamp(governance.window_end)}；请求 ID：{requestId ?? "-"}
            </Typography.Text>
          </>
        ) : null}
      </div>
    </AppShell>
  );
}

function RegistryTable({ tools }: { tools: GovernedTool[] }) {
  return (
    <Table<GovernedTool>
      rowKey={(item) => `${item.tool_id}@${item.version}`}
      dataSource={tools}
      pagination={false}
      scroll={{ x: 1500 }}
      expandable={{ expandedRowRender: (item) => <ToolDefinitionDetail tool={item} /> }}
      columns={[
        {
          title: "风险",
          dataIndex: "risk_tier",
          width: 90,
          render: (value: string) => <Tag color={riskColors[value]}>{value}</Tag>,
        },
        { title: "工具", dataIndex: "tool_id", width: 220 },
        { title: "版本", dataIndex: "version", width: 100 },
        {
          title: "接入通道",
          dataIndex: "transport",
          width: 190,
          render: (value: string | null) => (
            <Tag color={value === "MCP_STREAMABLE_HTTP" ? "cyan" : "default"}>{value ?? "未声明"}</Tag>
          ),
        },
        { title: "授权动作", dataIndex: "required_action", width: 180 },
        {
          title: "执行策略",
          dataIndex: "execution_policy",
          width: 220,
          render: (value: string) => policyLabels[value] ?? value,
        },
        {
          title: "权威来源合同",
          dataIndex: "authority_coverage",
          width: 150,
          render: (value: string) => (
            <Tag color={value === "DECLARED" ? "green" : value === "MISSING" ? "red" : "default"}>
              {value === "DECLARED" ? "已声明" : value === "MISSING" ? "缺失" : "不适用"}
            </Tag>
          ),
        },
        {
          title: "调用 / 失败",
          width: 120,
          render: (_, item) => `${item.usage.call_count} / ${item.usage.failed_count}`,
        },
        {
          title: "超时 / 重试",
          width: 130,
          render: (_, item) => `${item.timeout_seconds}s / ${item.max_attempts}`,
        },
        {
          title: "每分钟限额",
          dataIndex: "rate_limit_per_minute",
          width: 110,
        },
        {
          title: "最近调用",
          dataIndex: ["usage", "latest_call_at"],
          width: 180,
          render: timestamp,
        },
      ]}
    />
  );
}

function ToolDefinitionDetail({ tool }: { tool: GovernedTool }) {
  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Descriptions bordered size="small" column={2}>
        <Descriptions.Item label="完整版本">{tool.tool_id}@{tool.version}</Descriptions.Item>
        <Descriptions.Item label="风险语义">{riskLabels[tool.risk_tier]}</Descriptions.Item>
        <Descriptions.Item label="幂等">{tool.idempotent ? "是" : "否"}</Descriptions.Item>
        <Descriptions.Item label="需要审批">{tool.approval_required ? "是" : "否"}</Descriptions.Item>
        <Descriptions.Item label="补偿策略">{tool.compensation}</Descriptions.Item>
        <Descriptions.Item label="执行策略">{policyLabels[tool.execution_policy]}</Descriptions.Item>
        <Descriptions.Item label="接入通道">{tool.transport ?? "未声明"}</Descriptions.Item>
        <Descriptions.Item label="远端工具名">{tool.remote_tool_name ?? "-"}</Descriptions.Item>
        <Descriptions.Item label="服务身份 / 版本">
          {tool.server_id ? `${tool.server_id}@${tool.server_version ?? "unknown"}` : "-"}
        </Descriptions.Item>
        <Descriptions.Item label="最小 Scope">
          {tool.required_scopes.length ? tool.required_scopes.join(", ") : "不适用"}
        </Descriptions.Item>
        <Descriptions.Item label="输入 Schema 摘要" span={2}>
          <Typography.Text code copyable={Boolean(tool.input_schema_digest)}>
            {tool.input_schema_digest ?? "-"}
          </Typography.Text>
        </Descriptions.Item>
        {tool.authority ? (
          <>
            <Descriptions.Item label="权威域 / 所有者">{tool.authority.domain} / {tool.authority.owner}</Descriptions.Item>
            <Descriptions.Item label="合同版本">{tool.authority.contract_version}</Descriptions.Item>
            <Descriptions.Item label="生产来源">{tool.authority.enterprise_source}</Descriptions.Item>
            <Descriptions.Item label="本地合成来源">{tool.authority.synthetic_source}</Descriptions.Item>
            <Descriptions.Item label="权威字段" span={2}>{tool.authority.fields.join(", ")}</Descriptions.Item>
          </>
        ) : null}
      </Descriptions>
      <div>
        <Typography.Text strong>输入 JSON Schema</Typography.Text>
        <pre style={{ marginTop: 8, padding: 12, background: "#f5f5f5", overflowX: "auto" }}>
          {JSON.stringify(tool.input_schema, null, 2)}
        </pre>
      </div>
    </Space>
  );
}

function RecentCallsTable({ calls }: { calls: RecentToolCall[] }) {
  return (
    <Table<RecentToolCall>
      rowKey="tool_call_id"
      dataSource={calls}
      pagination={{ pageSize: 20, showSizeChanger: true }}
      scroll={{ x: 1420 }}
      locale={{ emptyText: "统计窗口内没有工具调用" }}
      columns={[
        { title: "时间", dataIndex: "occurred_at", width: 180, render: timestamp },
        { title: "工具", dataIndex: "tool_id", width: 210 },
        { title: "版本", dataIndex: "tool_version", width: 100 },
        { title: "接入通道", dataIndex: "transport", width: 190, render: dash },
        {
          title: "远端服务",
          width: 220,
          render: (_, item) => item.server_id
            ? `${item.server_id}@${item.server_version ?? "unknown"}`
            : "-",
        },
        {
          title: "风险",
          dataIndex: "risk_tier",
          width: 80,
          render: (value: string) => <Tag color={riskColors[value]}>{value}</Tag>,
        },
        {
          title: "状态",
          dataIndex: "status",
          width: 150,
          render: (value: string) => <Tag color={callStatusColor(value)}>{value}</Tag>,
        },
        { title: "Agent Run", dataIndex: "agent_run_id", width: 210, render: dash },
        { title: "事实来源", dataIndex: "source", width: 150, render: dash },
        {
          title: "权威证据",
          dataIndex: "authority_evidence",
          width: 130,
          render: (value: string) => (
            <Tag color={value === "VERIFIED" ? "green" : value === "MISMATCH" ? "red" : "default"}>{value}</Tag>
          ),
        },
        { title: "来源时间", dataIndex: "source_as_of", width: 180, render: timestamp },
        { title: "失败原因码", dataIndex: "reason_code", width: 200, render: dash },
      ]}
    />
  );
}

function callStatusColor(value: string): string {
  if (value === "FAILED") return "red";
  if (value === "SUCCEEDED") return "green";
  if (value === "APPROVAL_REQUIRED") return "orange";
  if (value === "EXTERNAL_HANDOFF") return "purple";
  return "default";
}

function timestamp(value: string | null | undefined): string {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "-";
}

function dash(value: string | null | undefined): string {
  return value || "-";
}
