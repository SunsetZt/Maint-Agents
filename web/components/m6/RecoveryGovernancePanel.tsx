"use client";

import {
  Alert,
  Button,
  Card,
  Checkbox,
  Descriptions,
  Form,
  Input,
  Modal,
  Space,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  getRecoveryOverview,
  startRecoveryDrill,
  type RecoveryComponent,
  type RecoveryDrill,
  type RecoveryOverview,
} from "@/lib/api/client";

const components: Array<{ label: string; value: RecoveryComponent }> = [
  { label: "PostgreSQL", value: "postgresql" },
  { label: "Temporal", value: "temporal" },
  { label: "MinIO", value: "minio" },
  { label: "Kafka", value: "kafka" },
  { label: "Redis", value: "redis" },
  { label: "GitOps / Registry", value: "gitops_registry" },
  { label: "模型服务", value: "model_service" },
];

const componentNames = new Map(components.map((item) => [item.value, item.label]));
const evidenceColors: Record<string, string> = {
  COMPLIANT: "green",
  MISSING: "default",
  STALE: "orange",
  FAILED: "red",
};
const evidenceLabels: Record<string, string> = {
  COMPLIANT: "证据合规",
  MISSING: "缺少证据",
  STALE: "证据过期",
  FAILED: "验证失败",
};
const failureLabels: Record<string, string> = {
  postgresql_unavailable: "PostgreSQL 不可用",
  kafka_unavailable: "Kafka 不可用",
  temporal_unavailable: "Temporal 不可用",
  minio_unavailable: "MinIO 不可用",
  redis_unavailable: "Redis 不可用",
  model_service_unavailable: "模型服务不可用",
  opa_unavailable: "OPA 不可用",
  vault_unavailable: "Vault 不可用",
  sse_disconnected: "SSE 断线",
  gpu_out_of_memory: "GPU OOM",
};
const policyTranslations: Record<string, { behavior: string; prohibited: string }> = {
  postgresql_unavailable: {
    behavior: "拒绝业务写入、Readiness 失败并告警",
    prohibited: "仅凭缓存继续审批或修改设备状态",
  },
  kafka_unavailable: {
    behavior: "合法在线事务与 Outbox 继续，恢复后重放追平",
    prohibited: "仅因消息发送失败回滚已提交的业务事务",
  },
  temporal_unavailable: {
    behavior: "新编排请求进入等待，高风险动作停止",
    prohibited: "在 API 内绕过工作流直接执行副作用",
  },
  minio_unavailable: {
    behavior: "阻止附件确认，保留可查询元数据",
    prohibited: "把无哈希临时文件当作正式证据",
  },
  redis_unavailable: {
    behavior: "回源权威存储、保守限流，必要时重新认证",
    prohibited: "把缓存缺失解释为工单或审批丢失",
  },
  model_service_unavailable: {
    behavior: "切换合格稳定模型或检索加人工模式",
    prohibited: "让未经评测的模型静默接管",
  },
  opa_unavailable: {
    behavior: "写操作和高风险工具失败关闭",
    prohibited: "为保持可用性放行全部请求",
  },
  vault_unavailable: {
    behavior: "已有租约实例仅运行至过期，新实例保持未就绪",
    prohibited: "在配置中回退到明文 Secret",
  },
  sse_disconnected: {
    behavior: "通过 Last-Event-ID 续传或读取权威快照",
    prohibited: "重启整个 Run 并重复产生副作用",
  },
  gpu_out_of_memory: {
    behavior: "有界排队、拒绝超预算任务或切换合格模型",
    prohibited: "无限重试形成雪崩",
  },
};

export function RecoveryGovernancePanel() {
  const [overview, setOverview] = useState<RecoveryOverview>();
  const [legalActions, setLegalActions] = useState<string[]>([]);
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [modalOpen, setModalOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [receipt, setReceipt] = useState<RecoveryDrill>();
  const [form] = Form.useForm<{
    scenario: string;
    scope: RecoveryComponent[];
    outageDetectedAt: string;
  }>();

  const load = useCallback(async () => {
    try {
      const result = await getRecoveryOverview();
      setOverview(result.overview);
      setLegalActions(result.legalActions);
      setRequestId(result.requestId);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const canManage = legalActions.includes("MANAGE_RECOVERY_DRILL");
  const failedComponents = useMemo(
    () => overview?.components.filter((item) => item.evidence_status !== "COMPLIANT").length ?? 0,
    [overview],
  );

  function openDrill() {
    const now = new Date();
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    form.setFieldsValue({
      scenario: "quarterly-cross-component",
      scope: components.filter((item) => item.value !== "redis").map((item) => item.value),
      outageDetectedAt: now.toISOString().slice(0, 16),
    });
    setModalOpen(true);
  }

  async function submitDrill() {
    const values = await form.validateFields();
    setSubmitting(true);
    try {
      const result = await startRecoveryDrill(
        {
          scenario: values.scenario,
          scope: values.scope,
          outage_detected_at: new Date(values.outageDetectedAt).toISOString(),
        },
        `recovery-drill-${Date.now()}-${crypto.randomUUID()}`,
      );
      setReceipt(result.drill);
      setRequestId(result.requestId);
      setModalOpen(false);
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card
      title="灾难恢复、备份证据与演练"
      extra={canManage ? <Button type="primary" onClick={openDrill}>发起恢复演练</Button> : null}
    >
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!overview && !error ? <LoadingState label="正在读取恢复治理证据" /> : null}
        {receipt ? (
          <Alert
            showIcon
            type="info"
            closable
            onClose={() => setReceipt(undefined)}
            message={`恢复演练 ${receipt.drill_id} 已进入 RUNNING`}
            description="备份与恢复执行器必须提交逐组件恢复结果；页面发起演练不会把未验证结果直接标记为成功。"
          />
        ) : null}
        {overview ? (
          <>
            <Alert
              showIcon
              type={overview.release_allowed ? "success" : "warning"}
              message={overview.release_gate_enforced
                ? overview.release_allowed ? "恢复发布门禁通过" : "恢复发布门禁未通过"
                : "当前环境只观察恢复姿态，不执行发布阻断"}
              description={`异常组件：${failedComponents}；PostgreSQL 月度恢复验证${overview.monthly_postgres_restore_due ? "已到期" : "有效"}；跨组件季度演练${overview.quarterly_cross_component_drill_due ? "已到期" : "有效"}。`}
            />
            <Table
              rowKey="component"
              pagination={false}
              dataSource={overview.components}
              scroll={{ x: 1050 }}
              columns={[
                {
                  title: "组件",
                  dataIndex: "component",
                  width: 160,
                  render: (value: RecoveryComponent) => componentNames.get(value) ?? value,
                },
                {
                  title: "RPO",
                  dataIndex: "rpo_seconds",
                  width: 130,
                  render: (value: number | null) => value == null ? "缓存可丢失" : formatSeconds(value),
                },
                {
                  title: "RTO",
                  dataIndex: "rto_seconds",
                  width: 120,
                  render: (value: number) => formatSeconds(value),
                },
                {
                  title: "证据状态",
                  dataIndex: "evidence_status",
                  width: 130,
                  render: (value: string) => (
                    <Tag color={evidenceColors[value]}>{evidenceLabels[value] ?? value}</Tag>
                  ),
                },
                {
                  title: "最新恢复点",
                  width: 200,
                  render: (_, item) => item.latest_evidence
                    ? new Date(item.latest_evidence.recovery_point_at).toLocaleString()
                    : "—",
                },
                { title: "恢复方式", dataIndex: "recovery_method" },
              ]}
            />
            <Card size="small" title={`最近恢复演练（${overview.recent_drills.length}）`}>
              <Table
                rowKey="drill_id"
                size="small"
                pagination={false}
                dataSource={overview.recent_drills}
                locale={{ emptyText: "尚未登记恢复演练" }}
                columns={[
                  { title: "场景", dataIndex: "scenario" },
                  {
                    title: "范围",
                    dataIndex: "scope",
                    render: (values: RecoveryComponent[]) => values
                      .map((value) => componentNames.get(value) ?? value)
                      .join("、"),
                  },
                  {
                    title: "状态",
                    dataIndex: "status",
                    width: 110,
                    render: (value: string) => (
                      <Tag color={value === "PASSED" ? "green" : value === "FAILED" ? "red" : "blue"}>
                        {value}
                      </Tag>
                    ),
                  },
                  {
                    title: "开始时间",
                    dataIndex: "started_at",
                    width: 190,
                    render: (value: string) => new Date(value).toLocaleString(),
                  },
                  {
                    title: "阻断项",
                    dataIndex: "blocker_codes",
                    render: (values: string[]) => values.length ? values.join("；") : "—",
                  },
                ]}
              />
            </Card>
            <Card size="small" title="受控降级策略">
              <Table
                rowKey="failure"
                size="small"
                pagination={false}
                dataSource={overview.degradation_policies}
                scroll={{ x: 1000 }}
                columns={[
                  {
                    title: "故障",
                    dataIndex: "failure",
                    width: 190,
                    render: (value: string) => failureLabels[value] ?? value,
                  },
                  {
                    title: "系统行为",
                    dataIndex: "system_behavior",
                    render: (value: string, item) => policyTranslations[item.failure]?.behavior ?? value,
                  },
                  {
                    title: "禁止行为",
                    dataIndex: "prohibited_behavior",
                    render: (value: string, item) => (
                      <Typography.Text type="danger">
                        {policyTranslations[item.failure]?.prohibited ?? value}
                      </Typography.Text>
                    ),
                  },
                ]}
              />
            </Card>
            <Typography.Text type="secondary">
              恢复策略 {overview.policy_version} · 请求 ID {requestId}
            </Typography.Text>
          </>
        ) : null}
      </Space>

      <Modal
        title="发起受控恢复演练"
        open={modalOpen}
        okText="创建演练"
        cancelText="取消"
        confirmLoading={submitting}
        onOk={() => void submitDrill()}
        onCancel={() => setModalOpen(false)}
      >
        <Alert
          showIcon
          type="warning"
          message="此操作只建立演练状态和证据边界，不会直接删除、恢复或切换生产数据。"
          style={{ marginBottom: 16 }}
        />
        <Form form={form} layout="vertical">
          <Form.Item
            name="scenario"
            label="演练场景标识"
            rules={[{ required: true }, { pattern: /^[A-Za-z0-9][A-Za-z0-9._:/-]*$/ }]}
          >
            <Input placeholder="quarterly-cross-component" />
          </Form.Item>
          <Form.Item name="scope" label="恢复范围" rules={[{ required: true }]}>
            <Checkbox.Group options={components} />
          </Form.Item>
          <Form.Item name="outageDetectedAt" label="模拟故障发现时间" rules={[{ required: true }]}>
            <Input type="datetime-local" />
          </Form.Item>
        </Form>
        <Descriptions size="small" column={1} bordered>
          <Descriptions.Item label="成功条件">
            每个组件均满足 RPO/RTO、备份证据和全部一致性检查。
          </Descriptions.Item>
          <Descriptions.Item label="失败行为">
            任何缺失或失败都记录明确 blocker，不能人工改写成 PASSED。
          </Descriptions.Item>
        </Descriptions>
      </Modal>
    </Card>
  );
}

function formatSeconds(seconds: number): string {
  if (seconds === 0) return "0 秒（版本不丢失）";
  if (seconds % 3600 === 0) return `${seconds / 3600} 小时`;
  if (seconds % 60 === 0) return `${seconds / 60} 分钟`;
  return `${seconds} 秒`;
}
