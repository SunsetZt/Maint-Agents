"use client";

import {
  Alert,
  Button,
  Card,
  Input,
  List,
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
  type DispatchWorkOrderStatus,
  type WorkOrderCenterItem,
  type WorkOrderPriority,
  listWorkOrders,
} from "@/lib/api/client";

const PAGE_SIZE = 24;
const STATUS_OPTIONS: Array<{ label: string; value: DispatchWorkOrderStatus }> = [
  { label: "待派工", value: "READY" },
  { label: "已派工", value: "ASSIGNED" },
  { label: "已接单", value: "ACCEPTED" },
  { label: "处理中", value: "IN_PROGRESS" },
  { label: "暂停", value: "ON_HOLD" },
  { label: "已升级", value: "ESCALATED" },
  { label: "待验收", value: "COMPLETED" },
  { label: "已验收", value: "VERIFIED" },
  { label: "已关闭", value: "CLOSED" },
];
const PRIORITY_OPTIONS: Array<{ label: string; value: WorkOrderPriority }> = [
  { label: "紧急", value: "CRITICAL" },
  { label: "高", value: "HIGH" },
  { label: "普通", value: "NORMAL" },
  { label: "低", value: "LOW" },
];

export default function WorkOrdersPage() {
  const [workOrders, setWorkOrders] = useState<WorkOrderCenterItem[]>();
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState<DispatchWorkOrderStatus>();
  const [priority, setPriority] = useState<WorkOrderPriority>();
  const [assigneeInput, setAssigneeInput] = useState("");
  const [assigneeFilter, setAssigneeFilter] = useState("");
  const [offset, setOffset] = useState(0);
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const result = await listWorkOrders({
        ...(status ? { status } : {}),
        ...(priority ? { priority } : {}),
        ...(assigneeFilter ? { assigned_subject_id: assigneeFilter } : {}),
        limit: PAGE_SIZE,
        offset,
      });
      setWorkOrders(result.workOrders);
      setTotal(result.total);
      setRequestId(result.requestId);
    } catch (caught) {
      setError(caught);
    }
  }, [assigneeFilter, offset, priority, status]);

  useEffect(() => { void load(); }, [load]);

  const summary = useMemo(() => ({
    breached: workOrders?.filter((item) => item.sla_status === "BREACHED").length ?? 0,
    atRisk: workOrders?.filter((item) => item.sla_status === "AT_RISK").length ?? 0,
    onHold: workOrders?.filter((item) => item.status === "ON_HOLD").length ?? 0,
    escalated: workOrders?.filter((item) => item.status === "ESCALATED").length ?? 0,
  }), [workOrders]);

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>工单运营与 SLA 中心</Typography.Title>
        <Alert
          type="info"
          showIcon
          message="从派工到闭环的统一运营视图"
          description="SLA、服务窗口和优先级来自工单计划；暂停、升级、重新计划会追加控制记录并校验版本，不会覆盖历史事实。"
        />

        <Card title="运营筛选" extra={<Button onClick={() => void load()}>刷新</Button>}>
          <Space wrap>
            <Select<DispatchWorkOrderStatus>
              allowClear
              style={{ minWidth: 160 }}
              placeholder="全部状态"
              options={STATUS_OPTIONS}
              value={status}
              onChange={(value) => { setStatus(value); setOffset(0); }}
            />
            <Select<WorkOrderPriority>
              allowClear
              style={{ minWidth: 140 }}
              placeholder="全部优先级"
              options={PRIORITY_OPTIONS}
              value={priority}
              onChange={(value) => { setPriority(value); setOffset(0); }}
            />
            <Input.Search
              allowClear
              style={{ width: 300 }}
              placeholder="按负责人 subject_id 筛选"
              value={assigneeInput}
              onChange={(event) => setAssigneeInput(event.target.value)}
              onSearch={(value) => { setAssigneeFilter(value.trim()); setOffset(0); }}
            />
          </Space>
        </Card>

        <Card title="当前页风险概览">
          <FactGrid facts={[
            ["符合条件工单", total],
            ["SLA 已超时", summary.breached],
            ["SLA 临期", summary.atRisk],
            ["暂停中", summary.onHold],
            ["已升级", summary.escalated],
            ["请求标识", requestId],
          ]} />
        </Card>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!workOrders && !error ? <LoadingState label="正在读取工单运营事实" /> : null}
        {workOrders?.length === 0 ? <EmptyState description="当前条件下没有工单" /> : null}

        <List
          grid={{ gutter: 16, xs: 1, xl: 2 }}
          dataSource={workOrders}
          renderItem={(workOrder) => (
            <List.Item>
              <Card
                title={workOrder.asset_display_name ?? workOrder.asset_id}
                extra={<StatusTag status={workOrder.status} version={workOrder.version} />}
              >
                <div className="page-stack">
                  <Space wrap>
                    <PriorityTag priority={workOrder.priority} />
                    <SlaTag status={workOrder.sla_status} />
                    {workOrder.site_name ?? workOrder.site_id ? (
                      <Tag>{workOrder.site_name ?? workOrder.site_id}</Tag>
                    ) : null}
                  </Space>
                  <Typography.Text type="secondary" ellipsis>
                    {workOrder.incident_description}
                  </Typography.Text>
                  <FactGrid facts={[
                    ["工单", workOrder.work_order_id],
                    ["负责人", workOrder.assigned_subject_id],
                    ["SLA 截止", formatTimestamp(workOrder.sla_due_at)],
                    ["服务窗口开始", formatTimestamp(workOrder.service_window_start)],
                    ["服务窗口结束", formatTimestamp(workOrder.service_window_end)],
                    ["更新时间", formatTimestamp(workOrder.updated_at)],
                  ]} />
                  <Space wrap>
                    <Link href={`/work-orders/${workOrder.work_order_id}`}>进入管控详情</Link>
                    <Link href={`/incidents/${workOrder.incident_id}`}>查看 Incident</Link>
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
    </AppShell>
  );
}

function PriorityTag({ priority }: { priority: string }) {
  const color = priority === "CRITICAL" ? "red" : priority === "HIGH" ? "orange" : "blue";
  return <Tag color={color}>优先级 {priority}</Tag>;
}

function SlaTag({ status }: { status: string }) {
  const color = status === "BREACHED" ? "red" : status === "AT_RISK" ? "orange" : "green";
  return <Tag color={color}>SLA {status}</Tag>;
}

function formatTimestamp(value: string | null): string {
  if (!value) return "—";
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime())
    ? value
    : timestamp.toLocaleString("zh-CN", { hour12: false });
}
