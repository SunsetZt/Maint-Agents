"use client";

import { Alert, Button, Card, List, Select, Space, Tag, Typography } from "antd";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import { FieldPwaStatusPanel } from "@/components/m3/FieldPwaProvider";
import {
  type DispatchWorkOrderStatus,
  type WorkOrderCenterItem,
  listMyFieldWorkOrders,
} from "@/lib/api/client";
import { pendingFieldEntries, syncFieldEntries } from "@/lib/field-offline-queue";

const ACTIVE_STATUSES: Array<{ label: string; value: DispatchWorkOrderStatus }> = [
  { label: "已派工", value: "ASSIGNED" },
  { label: "已接单", value: "ACCEPTED" },
  { label: "处理中", value: "IN_PROGRESS" },
  { label: "暂停中", value: "ON_HOLD" },
  { label: "已升级", value: "ESCALATED" },
  { label: "待验收", value: "COMPLETED" },
];

export default function FieldHomePage() {
  const [workOrders, setWorkOrders] = useState<WorkOrderCenterItem[]>();
  const [status, setStatus] = useState<DispatchWorkOrderStatus>();
  const [online, setOnline] = useState(true);
  const [pendingCount, setPendingCount] = useState(0);
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [syncing, setSyncing] = useState(false);

  const refreshLocalState = useCallback(() => {
    setOnline(typeof navigator === "undefined" ? true : navigator.onLine);
    setPendingCount(pendingFieldEntries().length);
  }, []);

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const result = await listMyFieldWorkOrders({
        ...(status ? { status } : {}),
        limit: 50,
        offset: 0,
      });
      setWorkOrders(result.workOrders);
      setRequestId(result.requestId);
    } catch (caught) {
      setError(caught);
    }
    refreshLocalState();
  }, [refreshLocalState, status]);

  useEffect(() => {
    void load();
    const handleConnectivity = () => refreshLocalState();
    window.addEventListener("online", handleConnectivity);
    window.addEventListener("offline", handleConnectivity);
    return () => {
      window.removeEventListener("online", handleConnectivity);
      window.removeEventListener("offline", handleConnectivity);
    };
  }, [load, refreshLocalState]);

  const summary = useMemo(() => ({
    active: workOrders?.filter((item) =>
      ["ACCEPTED", "IN_PROGRESS", "ON_HOLD"].includes(item.status)).length ?? 0,
    atRisk: workOrders?.filter((item) =>
      ["AT_RISK", "BREACHED"].includes(item.sla_status)).length ?? 0,
    equipment: new Set(workOrders?.map((item) => item.asset_id)).size,
  }), [workOrders]);

  async function syncPending() {
    setSyncing(true);
    setError(undefined);
    const result = await syncFieldEntries();
    setSyncing(false);
    refreshLocalState();
    if (result.error) {
      setError(result.error);
      return;
    }
    await load();
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>现场工程师工作台</Typography.Title>
        <FieldPwaStatusPanel />
        <Alert
          type={online ? "success" : "warning"}
          showIcon
          message={online ? "网络在线" : "当前离线，现场条目将保存在本机待同步队列"}
          description={`待同步条目 ${pendingCount} 条。重复同步使用同一客户端操作 ID，不会重复写入。`}
          action={<Button loading={syncing} disabled={!online || pendingCount === 0} onClick={() => void syncPending()}>立即同步</Button>}
        />

        <Card title="我的现场待办" extra={<Button onClick={() => void load()}>刷新</Button>}>
          <Space wrap>
            <Select<DispatchWorkOrderStatus>
              allowClear
              style={{ minWidth: 160 }}
              placeholder="全部分配工单"
              options={ACTIVE_STATUSES}
              value={status}
              onChange={setStatus}
            />
          </Space>
        </Card>
        <Card title="当前工作负载">
          <FactGrid facts={[
            ["分配给我的工单", workOrders?.length ?? 0],
            ["现场执行中", summary.active],
            ["SLA 临期或超时", summary.atRisk],
            ["涉及设备", summary.equipment],
            ["请求标识", requestId],
          ]} />
        </Card>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!workOrders && !error ? <LoadingState label="正在读取我的现场工单" /> : null}
        {workOrders?.length === 0 ? <EmptyState description="当前没有分配给你的工单" /> : null}
        <List
          dataSource={workOrders}
          renderItem={(workOrder) => (
            <List.Item>
              <Card
                style={{ width: "100%" }}
                title={workOrder.asset_display_name ?? workOrder.asset_id}
                extra={<StatusTag status={workOrder.status} version={workOrder.version} />}
              >
                <div className="page-stack">
                  <Space wrap>
                    <Tag color={workOrder.priority === "CRITICAL" ? "red" : "blue"}>{workOrder.priority}</Tag>
                    <Tag color={["BREACHED", "AT_RISK"].includes(workOrder.sla_status) ? "orange" : "green"}>SLA {workOrder.sla_status}</Tag>
                    <Tag>{workOrder.site_name ?? workOrder.site_id ?? "未设置站点"}</Tag>
                  </Space>
                  <Typography.Text>{workOrder.incident_description}</Typography.Text>
                  <FactGrid facts={[
                    ["工单", workOrder.work_order_id],
                    ["型号", workOrder.model_code],
                    ["服务窗口", windowText(workOrder.service_window_start, workOrder.service_window_end)],
                    ["SLA 截止", formatTimestamp(workOrder.sla_due_at)],
                  ]} />
                  <Link href={`/field/work-orders/${workOrder.work_order_id}`}>进入现场执行</Link>
                  <Link href={`/parts?assetId=${encodeURIComponent(workOrder.asset_id)}`}>查询设备备件与库存</Link>
                </div>
              </Card>
            </List.Item>
          )}
        />
      </div>
    </AppShell>
  );
}

function formatTimestamp(value: string | null): string {
  if (!value) return "—";
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function windowText(start: string | null, end: string | null): string {
  return start && end ? `${formatTimestamp(start)} — ${formatTimestamp(end)}` : "待调度";
}
