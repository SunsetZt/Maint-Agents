"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Modal,
  Progress,
  Row,
  Space,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
  Upload,
  message,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  getCostOperations,
  importCostLedgerBatch,
  updateCostPolicy,
  type CostBreakdown,
  type CostLedgerBatch,
  type CostLedgerBatchInput,
  type CostOperations,
  type CostPolicyInput,
} from "@/lib/api/client";

const budgetColors: Record<string, string> = {
  ON_TRACK: "green",
  WARNING: "orange",
  EXCEEDED: "red",
  EVIDENCE_INCOMPLETE: "default",
  UNCONFIGURED: "default",
};

const budgetLabels: Record<string, string> = {
  ON_TRACK: "预算正常",
  WARNING: "接近预算",
  EXCEEDED: "预算超支",
  EVIDENCE_INCOMPLETE: "证据不完整",
  UNCONFIGURED: "尚未配置",
};

const anomalyLabels: Record<string, string> = {
  monthly_budget_exceeded: "本月已超过总预算",
  monthly_budget_warning: "本月成本已达到预算预警线",
  cost_evidence_partial: "部分请求缺少有效费率或 GPU Profile",
  failed_inference_cost_high: "失败推理成本占比达到 20%",
  business_attribution_incomplete: "部分推理尚未关联诊断或工单",
};

const categoryLabels: Record<string, string> = {
  object_storage: "对象存储",
  network_egress: "网络流量",
  search_service: "搜索服务",
  external_tool_api: "外部工具 API",
  manual_annotation: "人工标注",
};

type CostPolicyFormValues = CostPolicyInput;

type LedgerImportReceipt = {
  batch: CostLedgerBatch;
  replayed: boolean;
  requestId: string;
};

export default function CostOperationsPage() {
  const [operations, setOperations] = useState<CostOperations>();
  const [legalActions, setLegalActions] = useState<string[]>([]);
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [refreshing, setRefreshing] = useState(false);
  const [policyOpen, setPolicyOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [ledgerOpen, setLedgerOpen] = useState(false);
  const [ledgerJson, setLedgerJson] = useState("");
  const [importingLedger, setImportingLedger] = useState(false);
  const [ledgerReceipt, setLedgerReceipt] = useState<LedgerImportReceipt>();
  const [form] = Form.useForm<CostPolicyFormValues>();
  const [messageApi, messageContext] = message.useMessage();

  const load = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true);
    try {
      const result = await getCostOperations();
      setOperations(result.operations);
      setLegalActions(result.legalActions);
      setRequestId(result.requestId);
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

  const failedCostRatio = useMemo(() => {
    const total = operations?.totals.inference_known_cost_usd ?? 0;
    return total > 0
      ? (operations?.totals.failed_inference_known_cost_usd ?? 0) / total
      : 0;
  }, [operations]);

  function openPolicy() {
    const policy = operations?.current_policy;
    form.setFieldsValue({
      prompt_tokens_per_million_usd: policy?.prompt_tokens_per_million_usd ?? 0,
      completion_tokens_per_million_usd: policy?.completion_tokens_per_million_usd ?? 0,
      gpu_hourly_cost_usd: policy?.gpu_hourly_cost_usd ?? 2.5,
      monthly_budget_usd: policy?.monthly_budget_usd ?? 1_000,
      scenario_budgets_usd: policy?.scenario_budgets_usd ?? {
        DIAGNOSIS: 600,
        VLM: 200,
        ASR: 100,
        TTS: 100,
      },
      warning_ratio: policy?.warning_ratio ?? 0.8,
      reason: "建立经审核的模型与 GPU 月度成本核算基线",
    });
    setPolicyOpen(true);
  }

  async function savePolicy() {
    if (!operations) return;
    try {
      const values = await form.validateFields();
      setSaving(true);
      await updateCostPolicy(values, operations.current_policy?.version ?? 0);
      messageApi.success("成本费率与预算新版本已生效");
      setPolicyOpen(false);
      await load(true);
    } catch (cause) {
      if (cause && typeof cause === "object" && "errorFields" in cause) return;
      messageApi.error(cause instanceof Error ? cause.message : "成本策略更新失败");
    } finally {
      setSaving(false);
    }
  }

  async function importLedger() {
    try {
      const parsed = JSON.parse(ledgerJson) as CostLedgerBatchInput;
      setImportingLedger(true);
      const nonce = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
      const result = await importCostLedgerBatch(parsed, `cost-ledger-${nonce}`);
      setLedgerReceipt(result);
      setLedgerOpen(false);
      setLedgerJson("");
      messageApi.success(result.replayed ? "成本 Ledger 批次已存在，已返回原回执" : "成本 Ledger 批次导入成功");
      await load(true);
    } catch (cause) {
      const detail = cause instanceof SyntaxError
        ? "Ledger JSON 格式无效，请修正后重试"
        : cause instanceof Error ? cause.message : "成本 Ledger 导入失败";
      messageApi.error(detail);
    } finally {
      setImportingLedger(false);
    }
  }

  function openLedgerImport() {
    setLedgerReceipt(undefined);
    setLedgerOpen(true);
  }

  return (
    <AppShell>
      {messageContext}
      <div className="page-stack">
        <Space align="start" style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>模型与基础设施成本归因中心</Typography.Title>
            <Typography.Paragraph type="secondary">
              按租户、Release、模型别名、业务场景以及诊断/工单关联汇总推理 Token、估算 GPU 时间、训练费用和失败重试成本。费率与预算均保留生效版本，不改写历史核算口径。
            </Typography.Paragraph>
          </div>
          <Space>
            {legalActions.includes("IMPORT_LEDGER") ? (
              <Button onClick={openLedgerImport}>导入成本 Ledger</Button>
            ) : null}
            {legalActions.includes("UPDATE_POLICY") ? (
              <Button type="primary" onClick={openPolicy}>配置费率与预算</Button>
            ) : null}
            <Button loading={refreshing} onClick={() => void load(true)}>刷新归因</Button>
          </Space>
        </Space>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!operations && !error ? <LoadingState label="正在核算本月模型与 GPU 成本" /> : null}

        {operations ? (
          <>
            {operations.evidence_status !== "COMPLETE" ? (
              <Alert
                showIcon
                type="warning"
                message={operations.evidence_status === "UNCONFIGURED" ? "尚未配置成本核算策略" : "成本证据不完整"}
                description={operations.evidence_status === "UNCONFIGURED"
                  ? "Token 与 GPU 用量仍会展示，但在明确录入内部费率和预算前不会生成虚假的美元成本。首次配置从本月月初生效。"
                  : `仍有 ${operations.totals.missing_cost_record_count} 条应用成本记录或 ${operations.unmetered_categories.length} 个外部成本类别证据不完整，已知成本只是下限，预算不会判定为正常。`}
              />
            ) : null}

            {ledgerReceipt ? (
              <Alert
                showIcon
                closable
                type="success"
                onClose={() => setLedgerReceipt(undefined)}
                message={ledgerReceipt.replayed ? "成本 Ledger 幂等重放回执" : "成本 Ledger 导入回执"}
                description={`批次 ${ledgerReceipt.batch.batch_id} · 外部批次 ${ledgerReceipt.batch.external_batch_id} · 金额 ${usd(ledgerReceipt.batch.total_amount_usd)} · 摘要 ${ledgerReceipt.batch.batch_digest} · 请求 ID ${ledgerReceipt.requestId}`}
              />
            ) : null}

            <Alert
              showIcon
              type={operations.budget.status === "EXCEEDED" ? "error" : operations.budget.status === "WARNING" ? "warning" : operations.budget.status === "ON_TRACK" ? "success" : "info"}
              message={`本月预算状态：${budgetLabels[operations.budget.status] ?? operations.budget.status}`}
              description={operations.budget.monthly_budget_usd == null
                ? "配置费率和预算后，系统会使用版本生效时间计算已知成本与预算消耗。"
                : `已知成本 ${usd(operations.budget.known_cost_usd)}，预算 ${usd(operations.budget.monthly_budget_usd)}，剩余 ${usd(operations.budget.remaining_usd ?? 0)}。`}
            />

            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="本月已知总成本" value={operations.totals.total_known_cost_usd} precision={4} prefix="$" /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="推理 Token" value={operations.totals.prompt_tokens + operations.totals.completion_tokens} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="估算推理 GPU 小时" value={operations.totals.estimated_gpu_hours} precision={4} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="外部 Ledger 已知成本" value={operations.totals.ledger_known_cost_usd} precision={4} prefix="$" suffix={` / ${operations.totals.ledger_entry_count} 条`} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="失败推理已知成本占比" value={failedCostRatio * 100} precision={2} suffix="%" valueStyle={failedCostRatio >= 0.2 ? { color: "#cf1322" } : undefined} /></Card>
              </Col>
            </Row>

            <Card
              title="外部成本 Ledger"
              extra={<Typography.Text type="secondary">完整性截止 {new Date(operations.ledger_coverage_cutoff).toLocaleString()}</Typography.Text>}
            >
              <Typography.Paragraph type="secondary">
                企业 FinOps 聚合器提交不可变、摘要绑定的 USD 批次。部分区间金额会计入已知成本，但只有从月初连续覆盖到最近完整 UTC 日的类别才标记为完整。
              </Typography.Paragraph>
              <Table
                rowKey="category"
                size="small"
                pagination={false}
                dataSource={operations.ledger_by_category}
                locale={{ emptyText: "当前月份尚无外部成本类别" }}
                columns={[
                  { title: "成本类别", dataIndex: "category", render: (value: string) => categoryLabels[value] ?? value },
                  { title: "Ledger 条目", dataIndex: "entry_count" },
                  { title: "已知成本", dataIndex: "known_cost_usd", render: usd },
                  { title: "月初至截止点覆盖", dataIndex: "coverage_complete", render: evidenceTag },
                ]}
              />
              <Typography.Title level={5} style={{ marginTop: 24 }}>不可变导入批次</Typography.Title>
              <Table
                rowKey="batch_id"
                size="small"
                dataSource={operations.ledger_batches}
                locale={{ emptyText: "当前月份尚无 Ledger 批次" }}
                pagination={{ pageSize: 8, hideOnSinglePage: true }}
                scroll={{ x: 1250 }}
                columns={[
                  { title: "来源", dataIndex: "source_system", width: 160 },
                  { title: "外部批次", dataIndex: "external_batch_id", width: 190 },
                  { title: "覆盖开始", dataIndex: "coverage_start", width: 190, render: localTime },
                  { title: "覆盖结束", dataIndex: "coverage_end", width: 190, render: localTime },
                  { title: "类别代码", dataIndex: "covered_categories", width: 220, render: (items: string[]) => items.join("、") },
                  { title: "条目", dataIndex: "entry_count", width: 80 },
                  { title: "金额", dataIndex: "total_amount_usd", width: 110, render: usd },
                  { title: "导入时间", dataIndex: "imported_at", width: 190, render: localTime },
                ]}
              />
            </Card>

            {operations.budget.consumed_ratio != null ? (
              <Card title="月度预算消耗">
                <Progress
                  percent={Math.min(100, operations.budget.consumed_ratio * 100)}
                  status={operations.budget.status === "EXCEEDED" ? "exception" : "active"}
                  format={() => `${(operations.budget.consumed_ratio! * 100).toFixed(2)}%`}
                />
                <Table
                  rowKey="scenario"
                  size="small"
                  pagination={false}
                  dataSource={operations.budget.scenarios}
                  locale={{ emptyText: "当前策略没有配置场景子预算" }}
                  columns={[
                    { title: "业务场景", dataIndex: "scenario" },
                    { title: "预算", dataIndex: "budget_usd", render: usd },
                    { title: "已知成本", dataIndex: "known_cost_usd", render: usd },
                    { title: "剩余", dataIndex: "remaining_usd", render: usd },
                    { title: "消耗率", dataIndex: "consumed_ratio", render: ratio },
                    { title: "状态", dataIndex: "status", render: (value: string) => <Tag color={budgetColors[value]}>{budgetLabels[value] ?? value}</Tag> },
                  ]}
                />
              </Card>
            ) : null}

            {operations.anomalies.length ? (
              <Alert
                showIcon
                type="warning"
                message="当前成本治理提示"
                description={operations.anomalies.map((item) => anomalyLabels[item] ?? item).join("；")}
              />
            ) : null}

            <Card title="归因明细">
              <Tabs
                items={[
                  { key: "release", label: "按 Release", children: <BreakdownTable items={operations.by_release} /> },
                  { key: "scenario", label: "按业务场景", children: <BreakdownTable items={operations.by_scenario} /> },
                  { key: "business", label: "按诊断/工单", children: <BreakdownTable items={operations.by_business_task} /> },
                ]}
              />
            </Card>

            <Row gutter={[16, 16]}>
              <Col xs={24} xl={14}>
                <Card title="本月每日用量与成本">
                  <Table
                    rowKey="day"
                    size="small"
                    pagination={false}
                    dataSource={operations.daily}
                    locale={{ emptyText: "本月尚无推理用量" }}
                    columns={[
                      { title: "日期", dataIndex: "day" },
                      { title: "请求", dataIndex: "request_count" },
                      { title: "输入 Token", dataIndex: "prompt_tokens" },
                      { title: "输出 Token", dataIndex: "completion_tokens" },
                      { title: "Ledger 条目", dataIndex: "ledger_entry_count" },
                      { title: "Ledger 成本", dataIndex: "ledger_known_cost_usd", render: usd },
                      { title: "已知成本", dataIndex: "known_cost_usd", render: usd },
                      { title: "证据", dataIndex: "cost_complete", render: evidenceTag },
                    ]}
                  />
                </Card>
              </Col>
              <Col xs={24} xl={10}>
                <Card title="当前生效核算口径">
                  {operations.current_policy ? (
                    <Descriptions column={1} bordered size="small">
                      <Descriptions.Item label="版本">v{operations.current_policy.version}</Descriptions.Item>
                      <Descriptions.Item label="输入 / 百万 Token">{usd(operations.current_policy.prompt_tokens_per_million_usd)}</Descriptions.Item>
                      <Descriptions.Item label="输出 / 百万 Token">{usd(operations.current_policy.completion_tokens_per_million_usd)}</Descriptions.Item>
                      <Descriptions.Item label="GPU / 小时">{usd(operations.current_policy.gpu_hourly_cost_usd)}</Descriptions.Item>
                      <Descriptions.Item label="生效时间">{new Date(operations.current_policy.effective_at).toLocaleString()}</Descriptions.Item>
                      <Descriptions.Item label="变更原因">{operations.current_policy.reason}</Descriptions.Item>
                    </Descriptions>
                  ) : <Typography.Text type="secondary">尚无经审核的成本策略。</Typography.Text>}
                </Card>
              </Col>
            </Row>

            <Alert
              showIcon
              type="info"
              message="未计量类别不会被填零"
              description={`当前尚未接入账单级计量：${operations.unmetered_categories.map((item) => categoryLabels[item] ?? item).join("、")}。这些项目不计入“已知总成本”，生产接入时必须通过真实账单或计量 Ledger 补齐。`}
            />
            <Typography.Text type="secondary">
              核算窗口 {new Date(operations.window_start).toLocaleString()} 至 {new Date(operations.window_end).toLocaleString()} · 策略 {operations.policy_version} · 请求 ID {requestId}
            </Typography.Text>
          </>
        ) : null}
      </div>

      <Modal
        title="导入受治理的成本 Ledger 批次"
        open={ledgerOpen}
        confirmLoading={importingLedger}
        okText="导入批次"
        cancelText="取消"
        okButtonProps={{ disabled: ledgerJson.trim().length === 0 }}
        onOk={() => void importLedger()}
        onCancel={() => setLedgerOpen(false)}
        width={820}
      >
        <Alert
          showIcon
          type="info"
          message="只提交经审核和脱敏的规范 JSON"
          description="服务端会重新计算规范摘要并校验币种、类别、证据哈希、时间覆盖、租户引用和幂等冲突。不要加入 tenant_id、凭据或原始账单正文。导入失败时输入会保留。"
          style={{ marginBottom: 16 }}
        />
        <Space direction="vertical" style={{ width: "100%" }} size="middle">
          <Upload
            accept="application/json,.json"
            maxCount={1}
            showUploadList={false}
            beforeUpload={(file) => {
              void file.text().then(setLedgerJson).catch(() => messageApi.error("无法读取 Ledger JSON 文件"));
              return false;
            }}
          >
            <Button>选择 JSON 文件</Button>
          </Upload>
          <label htmlFor="cost-ledger-json"><Typography.Text strong>Ledger JSON</Typography.Text></label>
          <Input.TextArea
            id="cost-ledger-json"
            value={ledgerJson}
            onChange={(event) => setLedgerJson(event.target.value)}
            rows={18}
            spellCheck={false}
            placeholder='{"source_system":"enterprise-finops", ...}'
          />
        </Space>
      </Modal>

      <Modal
        title="发布成本费率与预算新版本"
        open={policyOpen}
        confirmLoading={saving}
        okText="发布新版本"
        cancelText="取消"
        onOk={() => void savePolicy()}
        onCancel={() => setPolicyOpen(false)}
        width={760}
      >
        <Alert
          showIcon
          type="info"
          message="版本生效规则"
          description="首次策略从本月月初生效，用于回算当月；后续策略只影响发布时间之后的新请求，历史费率不会被覆盖。自建模型可将 Token 单价设为 0，只核算 GPU；外部模型可按合同设置 Token 单价。"
          style={{ marginBottom: 18 }}
        />
        <Form form={form} layout="vertical">
          <Row gutter={16}>
            <Col span={8}><Form.Item name="prompt_tokens_per_million_usd" label="输入 / 百万 Token（USD）" rules={[{ required: true }]}><InputNumber min={0} max={10_000} precision={6} style={{ width: "100%" }} /></Form.Item></Col>
            <Col span={8}><Form.Item name="completion_tokens_per_million_usd" label="输出 / 百万 Token（USD）" rules={[{ required: true }]}><InputNumber min={0} max={10_000} precision={6} style={{ width: "100%" }} /></Form.Item></Col>
            <Col span={8}><Form.Item name="gpu_hourly_cost_usd" label="GPU / 小时（USD）" rules={[{ required: true }]}><InputNumber min={0} max={10_000} precision={4} style={{ width: "100%" }} /></Form.Item></Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}><Form.Item name="monthly_budget_usd" label="租户月度总预算（USD）" rules={[{ required: true }]}><InputNumber min={0.01} max={1_000_000_000} precision={2} style={{ width: "100%" }} /></Form.Item></Col>
            <Col span={12}><Form.Item name="warning_ratio" label="预算预警比例" rules={[{ required: true }]}><InputNumber min={0.1} max={1} step={0.05} precision={2} style={{ width: "100%" }} /></Form.Item></Col>
          </Row>
          <Typography.Text strong>场景子预算（可选）</Typography.Text>
          <Row gutter={16} style={{ marginTop: 10 }}>
            {(["DIAGNOSIS", "VLM", "ASR", "TTS"] as const).map((scenario) => (
              <Col span={6} key={scenario}>
                <Form.Item name={["scenario_budgets_usd", scenario]} label={scenario}>
                  <InputNumber min={0.01} precision={2} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            ))}
          </Row>
          <Form.Item name="reason" label="发布原因" rules={[{ required: true, min: 8, max: 1_000 }]}>
            <Input.TextArea rows={3} maxLength={1_000} showCount />
          </Form.Item>
        </Form>
      </Modal>
    </AppShell>
  );
}

function BreakdownTable({ items }: { items: CostBreakdown[] }) {
  return (
    <Table<CostBreakdown>
      rowKey="key"
      dataSource={items}
      pagination={{ pageSize: 10, hideOnSinglePage: true }}
      locale={{ emptyText: "当前窗口没有可归因记录" }}
      scroll={{ x: 1050 }}
      columns={[
        { title: "归因对象", dataIndex: "label", width: 270 },
        { title: "请求", dataIndex: "request_count", width: 80 },
        { title: "成功", dataIndex: "succeeded_count", width: 80 },
        { title: "失败", dataIndex: "failed_count", width: 80 },
        { title: "输入 Token", dataIndex: "prompt_tokens", width: 120 },
        { title: "输出 Token", dataIndex: "completion_tokens", width: 120 },
        { title: "GPU 小时", dataIndex: "estimated_gpu_hours", width: 115, render: (value: number) => value.toFixed(4) },
        { title: "Ledger 条目", dataIndex: "ledger_entry_count", width: 110 },
        { title: "Ledger 成本", dataIndex: "ledger_known_cost_usd", width: 120, render: usd },
        { title: "已知成本", dataIndex: "known_cost_usd", width: 120, render: usd },
        { title: "成本证据", dataIndex: "cost_complete", width: 110, render: evidenceTag },
      ]}
    />
  );
}

function usd(value: number): string {
  return `$${value.toFixed(4)}`;
}

function ratio(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function localTime(value: string): string {
  return new Date(value).toLocaleString();
}

function evidenceTag(value: boolean) {
  return <Tag color={value ? "green" : "orange"}>{value ? "完整" : "部分"}</Tag>;
}
