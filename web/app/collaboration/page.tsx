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
  List,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Timeline,
  Typography,
  message,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  cancelSupplierCollaboration,
  createSupplierCollaboration,
  dispatchSupplierCollaboration,
  getSupplierCollaboration,
  listSupplierCollaborations,
  refreshSupplierCollaboration,
  reviewSupplierCollaboration,
  type CreateSupplierCollaborationInput,
  type ReviewSupplierCollaborationInput,
  type SupplierCollaboration,
} from "@/lib/api/client";

type ReviewValues = ReviewSupplierCollaborationInput;

const STATUS_OPTIONS = [
  "READY",
  "DISPATCHING",
  "SUBMITTED",
  "WORKING",
  "INPUT_REQUIRED",
  "AUTH_REQUIRED",
  "REFRESHING",
  "CANCELING",
  "COMPLETED",
  "DELIVERY_FAILED",
  "FAILED",
  "CANCELED",
  "REJECTED",
  "ACCEPTED",
] as const;

export default function SupplierCollaborationPage() {
  const [items, setItems] = useState<SupplierCollaboration[]>([]);
  const [legalActions, setLegalActions] = useState<string[]>([]);
  const [status, setStatus] = useState<string>();
  const [selected, setSelected] = useState<SupplierCollaboration>();
  const [reviewTarget, setReviewTarget] = useState<SupplierCollaboration>();
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [busyId, setBusyId] = useState<string>();
  const [createOpen, setCreateOpen] = useState(false);
  const [createForm] = Form.useForm<CreateSupplierCollaborationInput>();
  const [reviewForm] = Form.useForm<ReviewValues>();
  const [messageApi, messageContext] = message.useMessage();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const result = await listSupplierCollaborations({ status, limit: 100 });
      setItems(result.collaborations);
      setLegalActions(result.legalActions);
      if (!result.legalActions.includes("CREATE")) setCreateOpen(false);
      setRequestId(result.requestId);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    } finally {
      setLoading(false);
    }
  }, [status]);

  useEffect(() => {
    const query = new URLSearchParams(window.location.search);
    const incidentId = query.get("incident_id");
    const diagnosisRunId = query.get("diagnosis_run_id");
    if (incidentId && diagnosisRunId) {
      createForm.setFieldsValue({
        incident_id: incidentId,
        diagnosis_run_id: diagnosisRunId,
        question: "请基于厂商手册和服务通告，复核当前诊断并给出建议、补充信息需求及安全提示。",
        sharing_reason: "当前诊断需要设备厂商专有知识进行独立复核，发送前由平台执行 DLP 脱敏。",
      });
      setCreateOpen(true);
    }
  }, [createForm]);

  useEffect(() => {
    void load();
  }, [load]);

  const metrics = useMemo(() => ({
    ready: items.filter((item) => item.status === "READY").length,
    remote: items.filter((item) => [
      "DISPATCHING",
      "SUBMITTED",
      "WORKING",
      "INPUT_REQUIRED",
      "AUTH_REQUIRED",
      "REFRESHING",
      "CANCELING",
    ].includes(item.status)).length,
    review: items.filter((item) => item.status === "COMPLETED").length,
    accepted: items.filter((item) => item.status === "ACCEPTED").length,
  }), [items]);

  function openCreate() {
    createForm.setFieldsValue({
      question: "请基于厂商手册和服务通告，复核当前诊断并给出建议、补充信息需求及安全提示。",
      sharing_reason: "当前诊断需要设备厂商专有知识进行独立复核，发送前由平台执行 DLP 脱敏。",
    });
    setCreateOpen(true);
  }

  async function saveCreate() {
    try {
      const values = await createForm.validateFields();
      setSaving(true);
      const created = await createSupplierCollaboration(
        values,
        `supplier-collaboration-${crypto.randomUUID()}`,
      );
      messageApi.success("脱敏协作证据包已创建，尚未发送给供应商 Agent");
      setCreateOpen(false);
      setSelected(created);
      await load();
    } catch (cause) {
      if (cause && typeof cause === "object" && "errorFields" in cause) return;
      messageApi.error(errorMessage(cause, "协作任务创建失败"));
    } finally {
      setSaving(false);
    }
  }

  async function runCommand(
    item: SupplierCollaboration,
    action: "DISPATCH" | "REFRESH" | "CANCEL",
  ) {
    setBusyId(item.collaboration_id);
    try {
      const changed = action === "DISPATCH"
        ? await dispatchSupplierCollaboration(item.collaboration_id, item.version)
        : action === "REFRESH"
          ? await refreshSupplierCollaboration(item.collaboration_id, item.version)
          : await cancelSupplierCollaboration(item.collaboration_id, item.version);
      setSelected(changed);
      messageApi.success(action === "DISPATCH" ? "协作任务已受控发送" : action === "REFRESH" ? "已同步远端任务状态" : "已发送取消请求");
      await load();
    } catch (cause) {
      messageApi.error(errorMessage(cause, "供应商 Agent 指令执行失败"));
    } finally {
      setBusyId(undefined);
    }
  }

  async function openDetail(item: SupplierCollaboration) {
    setBusyId(item.collaboration_id);
    try {
      setSelected(await getSupplierCollaboration(item.collaboration_id));
    } catch (cause) {
      messageApi.error(errorMessage(cause, "协作详情加载失败"));
    } finally {
      setBusyId(undefined);
    }
  }

  async function saveReview() {
    if (!reviewTarget) return;
    try {
      const values = await reviewForm.validateFields();
      setSaving(true);
      const changed = await reviewSupplierCollaboration(
        reviewTarget.collaboration_id,
        reviewTarget.version,
        values,
      );
      setSelected(changed);
      setReviewTarget(undefined);
      messageApi.success(values.decision === "ACCEPTED" ? "供应商建议已由独立复核人接纳" : "供应商建议已驳回");
      await load();
    } catch (cause) {
      if (cause && typeof cause === "object" && "errorFields" in cause) return;
      messageApi.error(errorMessage(cause, "人工复核提交失败"));
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
            <Typography.Title level={2}>供应商 Agent 协作</Typography.Title>
            <Typography.Paragraph type="secondary">
              将证据不足或需要厂商专有知识的诊断，升级为受控 A2A 协作任务；远端输出只能作为待复核建议。
            </Typography.Paragraph>
          </div>
          <Space>
            {legalActions.includes("CREATE") ? <Button type="primary" onClick={openCreate}>新建协作</Button> : null}
            <Button onClick={() => void load()}>刷新</Button>
          </Space>
        </Space>

        <Alert
          showIcon
          type="warning"
          message="供应商 Agent 没有本平台工具权限"
          description="只发送经 Presidio DLP 脱敏的最小证据包。远端 Agent Card、消息和 Artifact 均按不可信输入处理；建议经过结构校验、提示注入防护和不同人员复核后，仍不会自动改诊断、工单或控制设备。"
        />

        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} xl={6}><Card><Statistic title="待发送" value={metrics.ready} /></Card></Col>
          <Col xs={24} sm={12} xl={6}><Card><Statistic title="远端处理中" value={metrics.remote} /></Card></Col>
          <Col xs={24} sm={12} xl={6}><Card><Statistic title="待人工复核" value={metrics.review} /></Card></Col>
          <Col xs={24} sm={12} xl={6}><Card><Statistic title="已接纳" value={metrics.accepted} /></Card></Col>
        </Row>

        <Card size="small" title="任务筛选">
          <Select
            allowClear
            style={{ width: 240 }}
            placeholder="全部状态"
            value={status}
            onChange={setStatus}
            options={STATUS_OPTIONS.map((value) => ({ value, label: statusLabel(value) }))}
          />
        </Card>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {loading && !error ? <LoadingState label="正在加载供应商协作任务" /> : null}
        {!loading && !error ? (
          <>
            <Table<SupplierCollaboration>
              rowKey="collaboration_id"
              dataSource={items}
              pagination={{ pageSize: 20 }}
              scroll={{ x: 1540 }}
              locale={{ emptyText: "当前范围没有供应商协作任务" }}
              columns={[
                {
                  title: "状态",
                  dataIndex: "status",
                  width: 130,
                  render: (value: string) => <Tag color={statusColor(value)}>{statusLabel(value)}</Tag>,
                },
                { title: "Incident", dataIndex: "incident_id", width: 210 },
                { title: "诊断运行", dataIndex: "diagnosis_run_id", width: 220 },
                { title: "设备", dataIndex: "asset_id", width: 180 },
                {
                  title: "供应商 Agent",
                  width: 280,
                  render: (_, item) => `${item.agent_name}@${item.agent_version}`,
                },
                { title: "DLP 命中", dataIndex: "dlp_finding_count", width: 100 },
                { title: "请求人", dataIndex: "requested_by_subject_id", width: 180 },
                { title: "更新时间", dataIndex: "updated_at", width: 190, render: formatTime },
                {
                  title: "操作",
                  fixed: "right",
                  width: 300,
                  render: (_, item) => (
                    <Space wrap>
                      <Button size="small" loading={busyId === item.collaboration_id} onClick={() => void openDetail(item)}>详情</Button>
                      {item.legal_actions.includes("DISPATCH") ? <Button size="small" type="primary" loading={busyId === item.collaboration_id} onClick={() => void runCommand(item, "DISPATCH")}>发送</Button> : null}
                      {item.legal_actions.includes("REFRESH") ? <Button size="small" loading={busyId === item.collaboration_id} onClick={() => void runCommand(item, "REFRESH")}>同步</Button> : null}
                      {item.legal_actions.includes("CANCEL") ? <Button size="small" danger loading={busyId === item.collaboration_id} onClick={() => void runCommand(item, "CANCEL")}>取消</Button> : null}
                      {item.legal_actions.includes("REVIEW") ? <Button size="small" type="primary" onClick={() => {
                        reviewForm.setFieldsValue({ decision: "ACCEPTED", reason: "已独立核对引用证据、适用机型与安全提示，建议可作为后续人工处置参考。" });
                        setReviewTarget(item);
                      }}>复核</Button> : null}
                    </Space>
                  ),
                },
              ]}
            />
            <Typography.Text type="secondary">请求 ID：{requestId ?? "-"}</Typography.Text>
          </>
        ) : null}
      </div>

      <Modal
        title="新建供应商协作"
        open={createOpen}
        confirmLoading={saving}
        okText="创建脱敏证据包"
        onOk={() => void saveCreate()}
        onCancel={() => setCreateOpen(false)}
        width={720}
      >
        <Alert showIcon type="info" message="创建与发送是两个独立步骤；创建后可先检查脱敏载荷，再点击发送。" style={{ marginBottom: 16 }} />
        <Form form={createForm} layout="vertical">
          <Form.Item name="incident_id" label="Incident ID" rules={[{ required: true, min: 3 }]}><Input /></Form.Item>
          <Form.Item name="diagnosis_run_id" label="诊断运行 ID" rules={[{ required: true, min: 3 }]}><Input /></Form.Item>
          <Form.Item name="question" label="向厂商专家提出的问题" rules={[{ required: true, min: 8 }]}><Input.TextArea rows={4} showCount maxLength={2000} /></Form.Item>
          <Form.Item name="sharing_reason" label="数据共享业务理由" rules={[{ required: true, min: 8 }]}><Input.TextArea rows={3} showCount maxLength={1024} /></Form.Item>
        </Form>
      </Modal>

      <Modal
        title="人工复核供应商建议"
        open={Boolean(reviewTarget)}
        confirmLoading={saving}
        okText="提交复核"
        onOk={() => void saveReview()}
        onCancel={() => setReviewTarget(undefined)}
      >
        <Alert showIcon type="warning" message="请求人不能复核自己的协作任务；接纳建议也不会自动执行任何设备或工单动作。" style={{ marginBottom: 16 }} />
        <Form form={reviewForm} layout="vertical">
          <Form.Item name="decision" label="复核决定" rules={[{ required: true }]}>
            <Select options={[{ value: "ACCEPTED", label: "接纳为人工处置参考" }, { value: "REJECTED", label: "驳回建议" }]} />
          </Form.Item>
          <Form.Item name="reason" label="复核依据" rules={[{ required: true, min: 8 }]}><Input.TextArea rows={4} showCount maxLength={2000} /></Form.Item>
        </Form>
      </Modal>

      <Drawer
        title="供应商协作详情"
        open={Boolean(selected)}
        onClose={() => setSelected(undefined)}
        width={760}
      >
        {selected ? <CollaborationDetail item={selected} /> : null}
      </Drawer>
    </AppShell>
  );
}

function CollaborationDetail({ item }: { item: SupplierCollaboration }) {
  const advice = item.response_payload;
  return (
    <div className="page-stack">
      <Descriptions bordered size="small" column={1}>
        <Descriptions.Item label="状态"><Tag color={statusColor(item.status)}>{statusLabel(item.status)}</Tag></Descriptions.Item>
        <Descriptions.Item label="供应商 Agent">{item.agent_name}@{item.agent_version}</Descriptions.Item>
        <Descriptions.Item label="固定 Skill">{item.skill_id}</Descriptions.Item>
        <Descriptions.Item label="远端 Task">{item.remote_task_id ?? "尚未发送"}</Descriptions.Item>
        <Descriptions.Item label="DLP 策略">{item.dlp_policy_version}（命中 {item.dlp_finding_count} 项）</Descriptions.Item>
        <Descriptions.Item label="请求摘要">{item.request_payload_digest}</Descriptions.Item>
        <Descriptions.Item label="响应摘要">{item.response_digest ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="输出护栏">{item.guardrail_policy_version ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="共享理由">{item.sharing_reason}</Descriptions.Item>
        <Descriptions.Item label="复核">{item.review_decision ? `${item.review_decision} · ${item.review_reason ?? ""}` : "待复核"}</Descriptions.Item>
      </Descriptions>

      <Card size="small" title="实际发送的脱敏证据包">
        <pre className="json-report">{JSON.stringify(item.request_payload, null, 2)}</pre>
      </Card>

      <Card size="small" title="结构化供应商建议">
        {advice ? (
          <div className="page-stack">
            <Typography.Paragraph>{textField(advice, "recommendation")}</Typography.Paragraph>
            <Descriptions size="small" column={1}>
              <Descriptions.Item label="置信度">{String(advice.confidence ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="需补充信息">{stringList(advice.required_information).join("；") || "无"}</Descriptions.Item>
              <Descriptions.Item label="安全提示">{stringList(advice.safety_notices).join("；") || "无"}</Descriptions.Item>
            </Descriptions>
            <List
              size="small"
              header="证据引用"
              dataSource={referenceList(advice.references)}
              locale={{ emptyText: "供应商未提供引用" }}
              renderItem={(reference) => <List.Item>{reference.kind} · {reference.reference_id} · {reference.title}</List.Item>}
            />
          </div>
        ) : <Typography.Text type="secondary">远端尚未返回完成态结构化建议。</Typography.Text>}
      </Card>

      <Card size="small" title="追加式审计事件">
        <Timeline items={item.events.map((event) => ({
          children: (
            <div>
              <Typography.Text strong>{event.event_type}</Typography.Text>
              <br />
              <Typography.Text type="secondary">#{event.sequence} · {event.reason_code} · {event.actor_subject_id} · {formatTime(event.occurred_at)}</Typography.Text>
            </div>
          ),
        }))} />
      </Card>
    </div>
  );
}

function statusLabel(value: string): string {
  return ({
    READY: "待发送",
    DISPATCHING: "发送中",
    SUBMITTED: "已提交",
    WORKING: "处理中",
    INPUT_REQUIRED: "需补充信息",
    AUTH_REQUIRED: "远端需授权",
    REFRESHING: "同步中",
    CANCELING: "取消中",
    COMPLETED: "待人工复核",
    DELIVERY_FAILED: "发送失败",
    FAILED: "远端失败",
    CANCELED: "已取消",
    REJECTED: "远端拒绝",
    ACCEPTED: "已接纳",
  } as Record<string, string>)[value] ?? value;
}

function statusColor(value: string): string {
  if (value === "ACCEPTED") return "green";
  if (value === "COMPLETED") return "blue";
  if (["FAILED", "DELIVERY_FAILED", "REJECTED"].includes(value)) return "red";
  if (["WORKING", "SUBMITTED", "DISPATCHING", "REFRESHING", "CANCELING"].includes(value)) return "processing";
  return "default";
}

function textField(value: Record<string, unknown>, key: string): string {
  return typeof value[key] === "string" ? value[key] : "—";
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function referenceList(value: unknown): Array<{ reference_id: string; title: string; kind: string }> {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const reference = item as Record<string, unknown>;
    return typeof reference.reference_id === "string" && typeof reference.title === "string" && typeof reference.kind === "string"
      ? [{ reference_id: reference.reference_id, title: reference.title, kind: reference.kind }]
      : [];
  });
}

function formatTime(value: string): string {
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function errorMessage(value: unknown, fallback: string): string {
  return value instanceof Error ? value.message : fallback;
}
