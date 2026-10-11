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
  createPromptBundle,
  listPromptBundles,
  retirePromptBundle,
  reviewPromptBundle,
  submitPromptBundle,
  type CreatePromptBundleInput,
  type PromptBundle,
} from "@/lib/api/client";

const statusColors: Record<string, string> = {
  DEPLOYED: "green",
  DRAFT: "default",
  REVIEW_PENDING: "blue",
  APPROVED: "green",
  REJECTED: "red",
  RETIRED: "default",
};

const statusLabels: Record<string, string> = {
  DEPLOYED: "镜像内置",
  DRAFT: "草稿",
  REVIEW_PENDING: "等待独立复核",
  APPROVED: "治理已批准",
  REJECTED: "已拒绝",
  RETIRED: "已退役",
};

type CreateFormValues = Omit<
  CreatePromptBundleInput,
  "section_names" | "required_variables" | "compatible_model_aliases"
> & {
  section_names: string;
  required_variables: string;
  compatible_model_aliases: string;
};

type ActionState = {
  type: "SUBMIT" | "APPROVE" | "REJECT" | "RETIRE";
  bundle: PromptBundle;
};

export default function PromptGovernancePage() {
  const [bundles, setBundles] = useState<PromptBundle[]>([]);
  const [legalActions, setLegalActions] = useState<string[]>([]);
  const [requestId, setRequestId] = useState<string>();
  const [policyVersion, setPolicyVersion] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [action, setAction] = useState<ActionState>();
  const [saving, setSaving] = useState(false);
  const [createForm] = Form.useForm<CreateFormValues>();
  const [actionForm] = Form.useForm<{ evaluation_id?: string; reason?: string }>();
  const [messageApi, messageContext] = message.useMessage();

  const load = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true);
    else setLoading(true);
    try {
      const result = await listPromptBundles();
      setBundles(result.bundles);
      setLegalActions(result.legalActions);
      setRequestId(result.requestId);
      setPolicyVersion(result.policyVersion);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const metrics = useMemo(() => ({
    deployed: bundles.filter((item) => item.runtime_available && item.runtime_hash_matches).length,
    pending: bundles.filter((item) => item.status === "REVIEW_PENDING").length,
    approvedNotDeployed: bundles.filter(
      (item) => item.status === "APPROVED" && (!item.runtime_available || !item.runtime_hash_matches),
    ).length,
    production: bundles.filter((item) => item.active_production_reference).length,
  }), [bundles]);

  function openCreate() {
    createForm.setFieldsValue({
      task_type: "DIAGNOSIS",
      section_names: "system_core,tenant_policy,diagnosis_role,risk_policy,tool_instructions,output_schema",
      required_variables: "incident,authorized_evidence,authorized_enterprise_facts,enterprise_tool_failures,untrusted_human_confirmed_memories",
      output_schema_name: "industrial_diagnosis_report",
      compatible_model_aliases: "industrial-diagnosis",
    } as CreateFormValues);
    setCreateOpen(true);
  }

  async function saveCreate() {
    try {
      const values = await createForm.validateFields();
      setSaving(true);
      await createPromptBundle(
        {
          ...values,
          section_names: splitCsv(values.section_names),
          required_variables: splitCsv(values.required_variables),
          compatible_model_aliases: splitCsv(values.compatible_model_aliases),
        },
        `prompt-create-${crypto.randomUUID()}`,
      );
      messageApi.success("Prompt Bundle 元数据已登记为草稿");
      setCreateOpen(false);
      await load(true);
    } catch (cause) {
      if (cause && typeof cause === "object" && "errorFields" in cause) return;
      messageApi.error(cause instanceof Error ? cause.message : "Prompt Bundle 创建失败");
    } finally {
      setSaving(false);
    }
  }

  function openAction(next: ActionState) {
    actionForm.resetFields();
    setAction(next);
  }

  async function executeAction() {
    if (!action) return;
    try {
      const values = await actionForm.validateFields();
      setSaving(true);
      if (action.type === "SUBMIT") {
        await submitPromptBundle(
          action.bundle.prompt_bundle_id,
          values.evaluation_id!,
          action.bundle.version,
        );
      } else if (action.type === "RETIRE") {
        await retirePromptBundle(
          action.bundle.prompt_bundle_id,
          values.reason!,
          action.bundle.version,
        );
      } else {
        await reviewPromptBundle(
          action.bundle.prompt_bundle_id,
          action.type === "APPROVE" ? "APPROVED" : "REJECTED",
          values.reason!,
          action.bundle.version,
        );
      }
      messageApi.success(actionSuccess(action.type));
      setAction(undefined);
      await load(true);
    } catch (cause) {
      if (cause && typeof cause === "object" && "errorFields" in cause) return;
      messageApi.error(cause instanceof Error ? cause.message : "Prompt Bundle 操作失败");
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
            <Typography.Title level={2}>Prompt 与 Agent 配置治理</Typography.Title>
            <Typography.Paragraph type="secondary">
              以 Git Commit、源码路径和 SHA-256 管理 Prompt Bundle，绑定独立评测和双人复核；只有随运行镜像部署且摘要匹配的 Bundle 才能进入 AI Release。
            </Typography.Paragraph>
          </div>
          <Space>
            {legalActions.includes("CREATE") ? <Button type="primary" onClick={openCreate}>登记候选 Bundle</Button> : null}
            <Button loading={refreshing} onClick={() => void load(true)}>刷新</Button>
          </Space>
        </Space>

        <Alert
          showIcon
          type="info"
          message="GitOps 与明文隔离"
          description="数据库和本页面不保存、显示或在线编辑 Prompt 明文。候选必须先在受审查源码中形成，登记摘要后运行同口径评测；治理批准不等于已经部署，Runtime Registry 摘要匹配后才具备发布资格。"
        />

        {metrics.approvedNotDeployed ? (
          <Alert
            showIcon
            type="warning"
            message={`${metrics.approvedNotDeployed} 个已批准 Bundle 尚未进入当前运行镜像或摘要不一致`}
            description="这些候选不能用于创建生产 AI Release，需要先通过代码评审和镜像发布加入 Runtime Registry。"
          />
        ) : null}

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {loading && !error ? <LoadingState label="正在加载 Prompt Bundle 治理状态" /> : null}

        {!loading && !error ? (
          <>
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="Bundle 总数" value={bundles.length} /></Card></Col>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="当前镜像可用" value={metrics.deployed} /></Card></Col>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="等待独立复核" value={metrics.pending} /></Card></Col>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="生产 Release 使用中" value={metrics.production} /></Card></Col>
            </Row>

            <Table<PromptBundle>
              rowKey="prompt_bundle_id"
              dataSource={bundles}
              scroll={{ x: 1400 }}
              pagination={{ pageSize: 20, showSizeChanger: true }}
              expandable={{ expandedRowRender: (item) => <PromptBundleDetail bundle={item} /> }}
              locale={{ emptyText: "尚未登记 Prompt Bundle" }}
              columns={[
                {
                  title: "状态",
                  dataIndex: "status",
                  width: 130,
                  render: (value: string) => <Tag color={statusColors[value]}>{statusLabels[value] ?? value}</Tag>,
                },
                {
                  title: "Bundle",
                  width: 260,
                  render: (_, item) => <><Typography.Text strong>{item.prompt_bundle_id}</Typography.Text><br /><Typography.Text type="secondary">{item.name}@{item.bundle_version}</Typography.Text></>,
                },
                { title: "任务", dataIndex: "task_type", width: 120 },
                {
                  title: "运行时",
                  width: 140,
                  render: (_, item) => runtimeTag(item),
                },
                { title: "评测", dataIndex: "evaluation_id", width: 220, render: dash },
                { title: "Release 引用", dataIndex: "release_reference_count", width: 110 },
                {
                  title: "来源 Commit",
                  dataIndex: "source_commit",
                  width: 170,
                  render: (value: string) => value.length > 16 ? value.slice(0, 12) : value,
                },
                { title: "更新时间", dataIndex: "updated_at", width: 180, render: timestamp },
                {
                  title: "操作",
                  fixed: "right",
                  width: 260,
                  render: (_, item) => (
                    <Space wrap>
                      {item.legal_actions.includes("SUBMIT") ? <Button size="small" onClick={() => openAction({ type: "SUBMIT", bundle: item })}>提交评测证据</Button> : null}
                      {item.legal_actions.includes("APPROVE") ? <Button size="small" type="primary" onClick={() => openAction({ type: "APPROVE", bundle: item })}>批准</Button> : null}
                      {item.legal_actions.includes("REJECT") ? <Button size="small" danger onClick={() => openAction({ type: "REJECT", bundle: item })}>拒绝</Button> : null}
                      {item.legal_actions.includes("RETIRE") ? <Button size="small" danger onClick={() => openAction({ type: "RETIRE", bundle: item })}>退役</Button> : null}
                    </Space>
                  ),
                },
              ]}
            />
            <Typography.Text type="secondary">策略版本：{policyVersion ?? "-"}；请求 ID：{requestId ?? "-"}</Typography.Text>
          </>
        ) : null}
      </div>

      <Modal
        title="登记 Prompt Bundle 候选"
        open={createOpen}
        width={760}
        confirmLoading={saving}
        onOk={() => void saveCreate()}
        onCancel={() => setCreateOpen(false)}
      >
        <Alert
          showIcon
          type="warning"
          message="这里只登记元数据，不要粘贴 Prompt 明文"
          style={{ marginBottom: 16 }}
        />
        <Form form={createForm} layout="vertical">
          <Row gutter={16}>
            <Col span={12}><Form.Item name="prompt_bundle_id" label="Bundle ID" rules={[{ required: true }]}><Input placeholder="industrial-diagnosis-v2" /></Form.Item></Col>
            <Col span={12}><Form.Item name="name" label="名称" rules={[{ required: true }]}><Input placeholder="industrial-diagnosis" /></Form.Item></Col>
            <Col span={12}><Form.Item name="bundle_version" label="语义版本" rules={[{ required: true }]}><Input placeholder="2.0.0" /></Form.Item></Col>
            <Col span={12}><Form.Item name="task_type" label="任务类型" rules={[{ required: true }]}><Select options={["DIAGNOSIS", "VLM", "ASR", "TTS", "RERANKING"].map((value) => ({ value, label: value }))} /></Form.Item></Col>
          </Row>
          <Form.Item name="content_hash" label="规范化内容 SHA-256" rules={[{ required: true, pattern: /^sha256:[0-9a-f]{64}$/ }]}><Input placeholder="sha256:..." /></Form.Item>
          <Form.Item name="source_commit" label="源码 Commit" rules={[{ required: true, pattern: /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/ }]}><Input /></Form.Item>
          <Form.Item name="source_path" label="仓库内源码路径" rules={[{ required: true }]}><Input placeholder="src/.../prompt_bundle.py" /></Form.Item>
          <Form.Item name="section_names" label="Section 名称（逗号分隔）" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="required_variables" label="必需变量（逗号分隔）" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="output_schema_name" label="输出 Schema" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="compatible_model_aliases" label="兼容模型别名（逗号分隔）" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="change_summary" label="变更说明" rules={[{ required: true, min: 8, max: 1000 }]}><Input.TextArea rows={3} /></Form.Item>
        </Form>
      </Modal>

      <Modal
        title={action ? actionTitle(action.type) : "Prompt Bundle 操作"}
        open={Boolean(action)}
        confirmLoading={saving}
        okButtonProps={{ danger: action?.type === "REJECT" || action?.type === "RETIRE" }}
        onOk={() => void executeAction()}
        onCancel={() => setAction(undefined)}
      >
        {action ? <Typography.Paragraph>目标：{action.bundle.prompt_bundle_id}@{action.bundle.bundle_version}</Typography.Paragraph> : null}
        <Form form={actionForm} layout="vertical">
          {action?.type === "SUBMIT" ? (
            <Form.Item name="evaluation_id" label="通过门禁且绑定该摘要的 Evaluation ID" rules={[{ required: true }]}><Input /></Form.Item>
          ) : (
            <Form.Item name="reason" label="复核/退役理由" rules={[{ required: true, min: 8, max: 1000 }]}><Input.TextArea rows={4} /></Form.Item>
          )}
        </Form>
      </Modal>
    </AppShell>
  );
}

function PromptBundleDetail({ bundle }: { bundle: PromptBundle }) {
  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Descriptions bordered size="small" column={2}>
        <Descriptions.Item label="来源">{bundle.origin}</Descriptions.Item>
        <Descriptions.Item label="内容摘要">{bundle.content_hash}</Descriptions.Item>
        <Descriptions.Item label="源码路径" span={2}>{bundle.source_path}</Descriptions.Item>
        <Descriptions.Item label="Section" span={2}>{bundle.section_names.map((value) => <Tag key={value}>{value}</Tag>)}</Descriptions.Item>
        <Descriptions.Item label="必需变量" span={2}>{bundle.required_variables.map((value) => <Tag key={value}>{value}</Tag>)}</Descriptions.Item>
        <Descriptions.Item label="输出 Schema">{bundle.output_schema_name}</Descriptions.Item>
        <Descriptions.Item label="兼容模型">{bundle.compatible_model_aliases.join(", ")}</Descriptions.Item>
        <Descriptions.Item label="变更说明" span={2}>{bundle.change_summary}</Descriptions.Item>
        <Descriptions.Item label="评测报告摘要" span={2}>{bundle.evaluation_report_hash ?? "-"}</Descriptions.Item>
        <Descriptions.Item label="创建者">{bundle.created_by_subject_id}</Descriptions.Item>
        <Descriptions.Item label="复核者">{bundle.reviewed_by_subject_id ?? "-"}</Descriptions.Item>
        <Descriptions.Item label="复核理由" span={2}>{bundle.review_reason ?? "-"}</Descriptions.Item>
      </Descriptions>
      <Table
        rowKey="transition_id"
        size="small"
        pagination={false}
        dataSource={bundle.transitions}
        locale={{ emptyText: "内置 Bundle 的治理历史由镜像供应链证据承载" }}
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

function runtimeTag(item: PromptBundle) {
  if (!item.runtime_available) return <Tag color="orange">未进入当前镜像</Tag>;
  if (!item.runtime_hash_matches) return <Tag color="red">运行时摘要不匹配</Tag>;
  return <Tag color="green">可用且摘要一致</Tag>;
}

function splitCsv(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function timestamp(value: string | null | undefined): string {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "-";
}

function dash(value: string | null | undefined): string {
  return value || "-";
}

function actionTitle(value: ActionState["type"]): string {
  return {
    SUBMIT: "提交 Prompt 评测证据",
    APPROVE: "批准 Prompt Bundle",
    REJECT: "拒绝 Prompt Bundle",
    RETIRE: "退役 Prompt Bundle",
  }[value];
}

function actionSuccess(value: ActionState["type"]): string {
  return {
    SUBMIT: "已提交独立复核",
    APPROVE: "Prompt Bundle 已批准",
    REJECT: "Prompt Bundle 已拒绝",
    RETIRE: "Prompt Bundle 已退役",
  }[value];
}
