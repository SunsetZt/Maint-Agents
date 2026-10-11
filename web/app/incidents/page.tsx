"use client";

import {
  Alert,
  Button,
  Card,
  Drawer,
  Form,
  Input,
  List,
  Modal,
  Pagination,
  Select,
  Space,
  Tag,
  Typography,
} from "antd";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import {
  type CustomerCaseStatus,
  type IncidentControl,
  type IncidentQueueItem,
  classifyIncident,
  escalateIncident,
  listIncidentControls,
  listIncidentQueue,
  markIncidentInformationReceived,
  mergeDuplicateIncident,
  requestIncidentInformation,
  resumeIncident,
} from "@/lib/api/client";

const PAGE_SIZE = 20;
const STATUS_OPTIONS: Array<{ label: string; value: CustomerCaseStatus }> = [
  { label: "待受理", value: "SUBMITTED" },
  { label: "待补资料", value: "NEEDS_INFORMATION" },
  { label: "已分诊", value: "TRIAGED" },
  { label: "诊断中", value: "DIAGNOSING" },
  { label: "已诊断", value: "DIAGNOSED" },
  { label: "已建工单", value: "WORK_ORDER_CREATED" },
  { label: "已升级", value: "ESCALATED" },
  { label: "已解决", value: "RESOLVED" },
  { label: "已关闭", value: "CLOSED" },
  { label: "已取消", value: "CANCELLED" },
];

type Command =
  | "CLASSIFY"
  | "REQUEST_INFORMATION"
  | "INFORMATION_RECEIVED"
  | "ESCALATE"
  | "RESUME"
  | "MERGE_DUPLICATE";

type CommandValues = {
  severity?: "P1" | "P2" | "P3" | "P4";
  category?: string;
  responsible_queue?: string;
  reason: string;
  information_due_at?: string;
  responsible_subject_id?: string;
  recovery_condition?: string;
  return_status?: "TRIAGED" | "DIAGNOSING";
  canonical_incident_id?: string;
};

export default function IncidentCenterPage() {
  const [incidents, setIncidents] = useState<IncidentQueueItem[]>();
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState<CustomerCaseStatus>();
  const [severity, setSeverity] = useState<"P1" | "P2" | "P3" | "P4">();
  const [queueInput, setQueueInput] = useState("");
  const [queueFilter, setQueueFilter] = useState("");
  const [offset, setOffset] = useState(0);
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [active, setActive] = useState<{ incident: IncidentQueueItem; command: Command }>();
  const [busy, setBusy] = useState(false);
  const [historyIncident, setHistoryIncident] = useState<IncidentQueueItem>();
  const [controls, setControls] = useState<IncidentControl[]>();
  const [form] = Form.useForm<CommandValues>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const result = await listIncidentQueue({
        ...(status ? { status } : {}),
        ...(severity ? { severity } : {}),
        ...(queueFilter ? { responsible_queue: queueFilter } : {}),
        limit: PAGE_SIZE,
        offset,
      });
      setIncidents(result.incidents);
      setTotal(result.total);
      setRequestId(result.requestId);
    } catch (caught) {
      setError(caught);
    }
  }, [offset, queueFilter, severity, status]);

  useEffect(() => { void load(); }, [load]);

  const summary = useMemo(() => ({
    pending: incidents?.filter((item) => item.status === "SUBMITTED").length ?? 0,
    needsInformation: incidents?.filter((item) => item.status === "NEEDS_INFORMATION").length ?? 0,
    escalated: incidents?.filter((item) => item.status === "ESCALATED").length ?? 0,
    critical: incidents?.filter((item) => item.severity === "P1").length ?? 0,
  }), [incidents]);

  function openCommand(incident: IncidentQueueItem, command: Command) {
    setActive({ incident, command });
    form.resetFields();
    form.setFieldsValue({
      severity: "P2",
      return_status: "TRIAGED",
      responsible_queue: incident.responsible_queue ?? "after-sales-l1",
    });
  }

  async function submitCommand(values: CommandValues) {
    if (!active) return;
    const { incident, command } = active;
    setBusy(true);
    setError(undefined);
    try {
      if (command === "CLASSIFY") {
        await classifyIncident(incident.incident_id, incident.version, {
          severity: values.severity!,
          category: values.category!,
          responsible_queue: values.responsible_queue!,
          reason: values.reason,
        });
      } else if (command === "REQUEST_INFORMATION") {
        await requestIncidentInformation(incident.incident_id, incident.version, {
          reason: values.reason,
          information_due_at: new Date(values.information_due_at!).toISOString(),
        });
      } else if (command === "INFORMATION_RECEIVED") {
        await markIncidentInformationReceived(
          incident.incident_id,
          incident.version,
          values.reason,
        );
      } else if (command === "ESCALATE") {
        await escalateIncident(incident.incident_id, incident.version, {
          reason: values.reason,
          responsible_subject_id: values.responsible_subject_id!,
          recovery_condition: values.recovery_condition!,
        });
      } else if (command === "RESUME") {
        await resumeIncident(incident.incident_id, incident.version, {
          reason: values.reason,
          return_status: values.return_status!,
        });
      } else {
        await mergeDuplicateIncident(incident.incident_id, incident.version, {
          reason: values.reason,
          canonical_incident_id: values.canonical_incident_id!,
        });
      }
      setActive(undefined);
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  async function showHistory(incident: IncidentQueueItem) {
    setHistoryIncident(incident);
    setControls(undefined);
    setError(undefined);
    try {
      setControls((await listIncidentControls(incident.incident_id)).controls);
    } catch (caught) {
      setError(caught);
    }
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>故障受理中心</Typography.Title>
        <Alert
          type="info"
          showIcon
          message="从客户报障进入诊断与履约的人工受理关口"
          description="受理人员按设备权限查看队列，完成严重度、故障类别和责任队列分级；补资料、升级、恢复与重复归并均保留版本化操作证据，不会由模型自动改变业务状态。"
        />

        <Card title="受理筛选" extra={<Button onClick={() => void load()}>刷新队列</Button>}>
          <Space wrap>
            <Select<CustomerCaseStatus>
              allowClear
              style={{ minWidth: 160 }}
              placeholder="全部故障状态"
              options={STATUS_OPTIONS}
              value={status}
              onChange={(value) => { setStatus(value); setOffset(0); }}
            />
            <Select
              allowClear
              style={{ minWidth: 140 }}
              placeholder="全部严重度"
              options={["P1", "P2", "P3", "P4"].map((value) => ({ value, label: value }))}
              value={severity}
              onChange={(value) => { setSeverity(value); setOffset(0); }}
            />
            <Input.Search
              allowClear
              style={{ width: 280 }}
              placeholder="按责任队列精确筛选"
              value={queueInput}
              onChange={(event) => setQueueInput(event.target.value)}
              onSearch={(value) => { setQueueFilter(value.trim()); setOffset(0); }}
            />
          </Space>
        </Card>

        <Card title="当前页受理概览">
          <FactGrid facts={[
            ["符合条件故障", total],
            ["当前页待受理", summary.pending],
            ["当前页待补资料", summary.needsInformation],
            ["当前页 P1", summary.critical],
            ["当前页已升级", summary.escalated],
            ["请求标识", requestId],
          ]} />
        </Card>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!incidents && !error ? <LoadingState label="正在读取授权范围内的故障队列" /> : null}
        {incidents?.length === 0 ? <EmptyState description="当前筛选条件下没有故障" /> : null}

        <List
          grid={{ gutter: 16, xs: 1, xl: 2 }}
          dataSource={incidents}
          renderItem={(incident) => (
            <List.Item>
              <Card
                title={incident.asset_display_name ?? incident.asset_id}
                extra={<StatusTag status={incident.status} version={incident.version} />}
              >
                <div className="page-stack">
                  <Typography.Paragraph ellipsis={{ rows: 2 }}>
                    {incident.description}
                  </Typography.Paragraph>
                  <Space wrap>
                    {incident.severity ? <Tag color={severityColor(incident.severity)}>{incident.severity}</Tag> : <Tag>未分级</Tag>}
                    {incident.category ? <Tag>{incident.category}</Tag> : null}
                    {incident.responsible_queue ? <Tag color="geekblue">{incident.responsible_queue}</Tag> : null}
                  </Space>
                  <FactGrid facts={[
                    ["故障单", incident.incident_id],
                    ["设备型号", incident.model_code],
                    ["服务站点", incident.site_name ?? incident.site_id],
                    ["补资料截止", formatTimestamp(incident.information_due_at)],
                    ["升级责任人", incident.escalation_responsible_subject_id],
                    ["更新时间", formatTimestamp(incident.updated_at)],
                  ]} />
                  <Space wrap>
                    {incident.legal_actions.map((command) => (
                      <Button
                        key={command}
                        danger={command === "ESCALATE" || command === "MERGE_DUPLICATE"}
                        type={command === "CLASSIFY" ? "primary" : "default"}
                        onClick={() => openCommand(incident, command as Command)}
                      >
                        {commandLabel(command as Command)}
                      </Button>
                    ))}
                    <Button onClick={() => void showHistory(incident)}>操作流水</Button>
                    <Link href={`/incidents/${incident.incident_id}`}>进入诊断详情</Link>
                  </Space>
                </div>
              </Card>
            </List.Item>
          )}
        />
        {total > PAGE_SIZE ? (
          <Pagination
            current={Math.floor(offset / PAGE_SIZE) + 1}
            pageSize={PAGE_SIZE}
            total={total}
            showSizeChanger={false}
            onChange={(page) => setOffset((page - 1) * PAGE_SIZE)}
          />
        ) : null}
      </div>

      <Modal
        open={Boolean(active)}
        title={active ? commandLabel(active.command) : "故障操作"}
        okText="确认提交"
        cancelText="取消"
        confirmLoading={busy}
        onCancel={() => setActive(undefined)}
        onOk={() => form.submit()}
        destroyOnHidden
      >
        <Form form={form} layout="vertical" onFinish={(values) => void submitCommand(values)}>
          {active?.command === "CLASSIFY" ? <ClassificationFields /> : null}
          {active?.command === "REQUEST_INFORMATION" ? (
            <Form.Item name="information_due_at" label="资料补充截止时间" rules={[{ required: true }]}>
              <Input type="datetime-local" />
            </Form.Item>
          ) : null}
          {active?.command === "ESCALATE" ? <EscalationFields /> : null}
          {active?.command === "RESUME" ? (
            <Form.Item name="return_status" label="恢复到" rules={[{ required: true }]}>
              <Select options={[
                { value: "TRIAGED", label: "已分诊" },
                { value: "DIAGNOSING", label: "诊断中" },
              ]} />
            </Form.Item>
          ) : null}
          {active?.command === "MERGE_DUPLICATE" ? (
            <Form.Item name="canonical_incident_id" label="主故障单 ID" rules={[{ required: true }]}>
              <Input placeholder="同一设备下要保留的主故障单" />
            </Form.Item>
          ) : null}
          <Form.Item name="reason" label="人工处置原因" rules={[{ required: true, min: 2 }]}>
            <Input.TextArea rows={3} placeholder="该原因将进入不可覆盖的操作流水" />
          </Form.Item>
        </Form>
      </Modal>

      <Drawer
        open={Boolean(historyIncident)}
        width={640}
        title={`故障操作流水 · ${historyIncident?.incident_id ?? ""}`}
        onClose={() => setHistoryIncident(undefined)}
      >
        {!controls ? <LoadingState label="正在读取人工处置记录" /> : null}
        {controls?.length === 0 ? <EmptyState description="该故障尚无人工处置记录" /> : null}
        <List
          dataSource={controls}
          renderItem={(control) => (
            <List.Item>
              <Card size="small" style={{ width: "100%" }}>
                <Space wrap>
                  <Tag color="blue">{commandLabel(control.command_type as Command)}</Tag>
                  <Typography.Text>{control.previous_status} → {control.target_status}</Typography.Text>
                  <Typography.Text type="secondary">v{control.incident_version}</Typography.Text>
                </Space>
                <Typography.Paragraph style={{ marginTop: 12 }}>{control.reason}</Typography.Paragraph>
                <Typography.Text type="secondary">
                  {control.actor_subject_id} · {formatTimestamp(control.occurred_at)}
                  {control.related_incident_id ? ` · 主故障 ${control.related_incident_id}` : ""}
                </Typography.Text>
              </Card>
            </List.Item>
          )}
        />
      </Drawer>
    </AppShell>
  );
}

function ClassificationFields() {
  return (
    <>
      <Form.Item name="severity" label="严重度" rules={[{ required: true }]}>
        <Select options={["P1", "P2", "P3", "P4"].map((value) => ({ value, label: value }))} />
      </Form.Item>
      <Form.Item name="category" label="故障类别" rules={[{ required: true }]}>
        <Input placeholder="例如：机械、电气、控制、软件、网络" />
      </Form.Item>
      <Form.Item name="responsible_queue" label="责任队列" rules={[{ required: true }]}>
        <Input placeholder="例如：after-sales-l1" />
      </Form.Item>
    </>
  );
}

function EscalationFields() {
  return (
    <>
      <Form.Item name="responsible_subject_id" label="升级责任人" rules={[{ required: true }]}>
        <Input placeholder="subject_id" />
      </Form.Item>
      <Form.Item name="recovery_condition" label="恢复条件" rules={[{ required: true }]}>
        <Input.TextArea rows={2} placeholder="满足什么条件后可恢复处理" />
      </Form.Item>
    </>
  );
}

function commandLabel(command: Command): string {
  return {
    CLASSIFY: "分级受理",
    REQUEST_INFORMATION: "请求补资料",
    INFORMATION_RECEIVED: "确认资料已收",
    ESCALATE: "人工升级",
    RESUME: "恢复处理",
    MERGE_DUPLICATE: "归并重复故障",
  }[command] ?? command;
}

function severityColor(severity: string): string {
  return { P1: "red", P2: "orange", P3: "gold", P4: "blue" }[severity] ?? "default";
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) return "—";
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime())
    ? value
    : timestamp.toLocaleString("zh-CN", { hour12: false });
}
