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
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  createGovernedMemory,
  listGovernedMemories,
  revokeGovernedMemory,
  type CreateMemoryInput,
  type GovernedMemory,
} from "@/lib/api/client";

type MemoryFilters = {
  memory_type?: "SESSION_NOTE" | "USER_PREFERENCE";
  status?: "ACTIVE" | "REVOKED";
  incident_id?: string;
};

export default function MemoryGovernancePage() {
  const [memories, setMemories] = useState<GovernedMemory[]>([]);
  const [filters, setFilters] = useState<MemoryFilters>({});
  const [legalActions, setLegalActions] = useState<string[]>([]);
  const [requestId, setRequestId] = useState<string>();
  const [policyVersion, setPolicyVersion] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [revokeTarget, setRevokeTarget] = useState<GovernedMemory>();
  const [createForm] = Form.useForm<CreateMemoryInput>();
  const [revokeForm] = Form.useForm<{ reason: string }>();
  const memoryType = Form.useWatch("memory_type", createForm);
  const [messageApi, messageContext] = message.useMessage();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const result = await listGovernedMemories(filters);
      setMemories(result.memories);
      setLegalActions(result.legalActions);
      setRequestId(result.requestId);
      setPolicyVersion(result.policyVersion);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    void load();
  }, [load]);

  const metrics = useMemo(
    () => ({
      active: memories.filter((item) => item.status === "ACTIVE").length,
      session: memories.filter((item) => item.memory_type === "SESSION_NOTE").length,
      preference: memories.filter((item) => item.memory_type === "USER_PREFERENCE").length,
      revoked: memories.filter((item) => item.status === "REVOKED").length,
    }),
    [memories],
  );

  function openCreate() {
    createForm.setFieldsValue({
      memory_type: "USER_PREFERENCE",
      content: "",
      source_reference_id: `profile-confirmation-${crypto.randomUUID()}`,
      incident_id: null,
      confirmed: true,
    });
    setCreateOpen(true);
  }

  async function saveCreate() {
    try {
      const values = await createForm.validateFields();
      setSaving(true);
      await createGovernedMemory(
        {
          ...values,
          incident_id: values.memory_type === "SESSION_NOTE" ? values.incident_id : null,
          confirmed: true,
        },
        `memory-create-${crypto.randomUUID()}`,
      );
      messageApi.success("已保存人类确认的受治理记忆");
      setCreateOpen(false);
      await load();
    } catch (cause) {
      if (cause && typeof cause === "object" && "errorFields" in cause) return;
      messageApi.error(cause instanceof Error ? cause.message : "记忆保存失败");
    } finally {
      setSaving(false);
    }
  }

  async function saveRevoke() {
    if (!revokeTarget) return;
    try {
      const values = await revokeForm.validateFields();
      setSaving(true);
      await revokeGovernedMemory(revokeTarget.memory_id, revokeTarget.version, values.reason);
      messageApi.success("记忆已撤销，后续诊断不会再读取");
      setRevokeTarget(undefined);
      await load();
    } catch (cause) {
      if (cause && typeof cause === "object" && "errorFields" in cause) return;
      messageApi.error(cause instanceof Error ? cause.message : "记忆撤销失败");
    } finally {
      setSaving(false);
    }
  }

  return (
    <AppShell>
      {messageContext}
      <div className="page-stack">
        <Space align="start" style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>Memory 治理</Typography.Title>
            <Typography.Paragraph type="secondary">
              管理当前故障会话记忆和当前用户的跨会话偏好；每条记忆必须由人确认、保留来源并可撤销。
            </Typography.Paragraph>
          </div>
          <Space>
            {legalActions.includes("CREATE") ? <Button type="primary" onClick={openCreate}>新增确认记忆</Button> : null}
            <Button onClick={() => void load()}>刷新</Button>
          </Space>
        </Space>

        <Alert
          showIcon
          type="warning"
          message="Memory 不是企业事实库"
          description="设备状态、库存、保修和排班必须实时调用权威业务系统。模型生成的推测不能写入 Memory；所有记忆内容进入模型时仍按不可信数据处理。"
        />

        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} xl={6}><Card><Statistic title="当前有效" value={metrics.active} /></Card></Col>
          <Col xs={24} sm={12} xl={6}><Card><Statistic title="会话记忆" value={metrics.session} /></Card></Col>
          <Col xs={24} sm={12} xl={6}><Card><Statistic title="用户偏好" value={metrics.preference} /></Card></Col>
          <Col xs={24} sm={12} xl={6}><Card><Statistic title="已撤销" value={metrics.revoked} /></Card></Col>
        </Row>

        <Card size="small" title="筛选">
          <Space wrap>
            <Select
              allowClear
              style={{ width: 180 }}
              placeholder="记忆类型"
              value={filters.memory_type}
              onChange={(value) => setFilters((current) => ({ ...current, memory_type: value }))}
              options={[
                { value: "SESSION_NOTE", label: "当前故障会话" },
                { value: "USER_PREFERENCE", label: "用户确认偏好" },
              ]}
            />
            <Select
              allowClear
              style={{ width: 140 }}
              placeholder="状态"
              value={filters.status}
              onChange={(value) => setFilters((current) => ({ ...current, status: value }))}
              options={[
                { value: "ACTIVE", label: "有效" },
                { value: "REVOKED", label: "已撤销" },
              ]}
            />
            <Input
              allowClear
              style={{ width: 260 }}
              placeholder="Incident ID"
              value={filters.incident_id}
              onChange={(event) => setFilters((current) => ({
                ...current,
                incident_id: event.target.value || undefined,
              }))}
            />
          </Space>
        </Card>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {loading && !error ? <LoadingState label="正在加载受治理记忆" /> : null}
        {!loading && !error ? (
          <>
            <Table<GovernedMemory>
              rowKey="memory_id"
              dataSource={memories}
              scroll={{ x: 1320 }}
              pagination={{ pageSize: 20 }}
              expandable={{ expandedRowRender: (item) => <MemoryDetail memory={item} /> }}
              locale={{ emptyText: "当前范围没有可见记忆" }}
              columns={[
                {
                  title: "状态",
                  dataIndex: "status",
                  width: 100,
                  render: (value: string) => <Tag color={value === "ACTIVE" ? "green" : "default"}>{value === "ACTIVE" ? "有效" : "已撤销"}</Tag>,
                },
                {
                  title: "类型",
                  dataIndex: "memory_type",
                  width: 150,
                  render: (value: string) => value === "SESSION_NOTE" ? "当前故障会话" : "用户确认偏好",
                },
                { title: "内容", dataIndex: "content", ellipsis: true, width: 360 },
                { title: "Incident", dataIndex: "incident_id", width: 210, render: dash },
                { title: "来源", dataIndex: "source_reference_id", width: 240 },
                { title: "确认人", dataIndex: "confirmed_by_subject_id", width: 160 },
                { title: "版本", dataIndex: "version", width: 80 },
                {
                  title: "操作",
                  fixed: "right",
                  width: 110,
                  render: (_, item) => item.legal_actions.includes("REVOKE") ? (
                    <Button danger size="small" onClick={() => {
                      revokeForm.resetFields();
                      setRevokeTarget(item);
                    }}>撤销</Button>
                  ) : "—",
                },
              ]}
            />
            <Typography.Text type="secondary">策略版本：{policyVersion ?? "-"}；请求 ID：{requestId ?? "-"}</Typography.Text>
          </>
        ) : null}
      </div>

      <Modal
        title="新增人类确认记忆"
        open={createOpen}
        confirmLoading={saving}
        onOk={() => void saveCreate()}
        onCancel={() => setCreateOpen(false)}
      >
        <Form form={createForm} layout="vertical">
          <Form.Item name="memory_type" label="记忆类型" rules={[{ required: true }]}>
            <Select options={[
              { value: "USER_PREFERENCE", label: "用户确认偏好（跨会话，仅本人可见）" },
              { value: "SESSION_NOTE", label: "当前故障会话（绑定已确认输入）" },
            ]} />
          </Form.Item>
          {memoryType === "SESSION_NOTE" ? (
            <Form.Item name="incident_id" label="Incident ID" rules={[{ required: true, min: 3 }]}><Input /></Form.Item>
          ) : null}
          <Form.Item
            name="source_reference_id"
            label={memoryType === "SESSION_NOTE" ? "已确认 Agent Event ID" : "用户确认表单引用"}
            rules={[{ required: true, min: 3 }]}
            extra={memoryType === "SESSION_NOTE" ? "必须引用该故障下由当前用户确认的 agent.user_input.confirmed 事件。" : undefined}
          ><Input /></Form.Item>
          <Form.Item name="content" label="确认内容" rules={[{ required: true, max: memoryType === "USER_PREFERENCE" ? 500 : 2000 }]}>
            <Input.TextArea rows={5} showCount maxLength={memoryType === "USER_PREFERENCE" ? 500 : 2000} />
          </Form.Item>
          <Form.Item name="confirmed" hidden><Input /></Form.Item>
          <Alert showIcon type="info" message="提交即表示当前用户明确确认写入；模型和后台任务没有该写入入口。" />
        </Form>
      </Modal>

      <Modal
        title="撤销记忆"
        open={Boolean(revokeTarget)}
        confirmLoading={saving}
        okButtonProps={{ danger: true }}
        onOk={() => void saveRevoke()}
        onCancel={() => setRevokeTarget(undefined)}
      >
        <Typography.Paragraph>撤销后，新诊断不会再选择该记忆，已有模型调用审计仍保留当时的 Memory ID 和摘要。</Typography.Paragraph>
        <Form form={revokeForm} layout="vertical">
          <Form.Item name="reason" label="撤销原因" rules={[{ required: true, min: 8, max: 1000 }]}>
            <Input.TextArea rows={4} />
          </Form.Item>
        </Form>
      </Modal>
    </AppShell>
  );
}

function MemoryDetail({ memory }: { memory: GovernedMemory }) {
  return (
    <Space direction="vertical" style={{ width: "100%" }}>
      <Descriptions bordered size="small" column={2}>
        <Descriptions.Item label="Memory ID">{memory.memory_id}</Descriptions.Item>
        <Descriptions.Item label="内容摘要">{memory.content_hash}</Descriptions.Item>
        <Descriptions.Item label="资产">{memory.asset_id ?? "-"}</Descriptions.Item>
        <Descriptions.Item label="所有者">{memory.owner_subject_id}</Descriptions.Item>
        <Descriptions.Item label="来源类型">{memory.source_reference_type}</Descriptions.Item>
        <Descriptions.Item label="来源引用">{memory.source_reference_id}</Descriptions.Item>
        <Descriptions.Item label="撤销人">{memory.revoked_by_subject_id ?? "-"}</Descriptions.Item>
        <Descriptions.Item label="撤销时间">{timestamp(memory.revoked_at)}</Descriptions.Item>
        <Descriptions.Item label="撤销原因" span={2}>{memory.revocation_reason ?? "-"}</Descriptions.Item>
      </Descriptions>
      <Table
        rowKey="transition_id"
        size="small"
        pagination={false}
        dataSource={memory.transitions}
        columns={[
          { title: "序号", dataIndex: "sequence", width: 70 },
          { title: "状态", render: (_, item) => `${item.from_status} → ${item.to_status}` },
          { title: "原因码", dataIndex: "reason_code" },
          { title: "操作人", dataIndex: "actor_subject_id" },
          { title: "时间", dataIndex: "occurred_at", render: timestamp },
        ]}
      />
    </Space>
  );
}

function timestamp(value: string | null | undefined): string {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "-";
}

function dash(value: string | null | undefined): string {
  return value || "-";
}
