"use client";

import { Alert, Button, Card, Input, List, Modal, Select, Space, Statistic, Typography } from "antd";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import {
  type ExecutionReconciliation,
  checkExecutionReconciliation,
  escalateExecutionReconciliation,
  listExecutionReconciliations,
} from "@/lib/api/client";

type ReconciliationStatus = "PENDING" | "RESOLVED" | "ESCALATED";

export default function ExecutionReconciliationsPage() {
  const [records, setRecords] = useState<ExecutionReconciliation[]>();
  const [status, setStatus] = useState<ReconciliationStatus>();
  const [busyId, setBusyId] = useState<string>();
  const [escalating, setEscalating] = useState<ExecutionReconciliation>();
  const [reason, setReason] = useState("外部系统持续无法确认结果，转交业务人员线下核对");
  const [error, setError] = useState<unknown>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      setRecords((await listExecutionReconciliations()).reconciliations);
    } catch (caught) {
      setError(caught);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const totals = useMemo(() => ({
    pending: records?.filter((item) => item.status === "PENDING").length ?? 0,
    resolved: records?.filter((item) => item.status === "RESOLVED").length ?? 0,
    escalated: records?.filter((item) => item.status === "ESCALATED").length ?? 0,
  }), [records]);
  const visibleRecords = useMemo(
    () => status ? records?.filter((item) => item.status === status) : records,
    [records, status],
  );

  async function check(record: ExecutionReconciliation) {
    setBusyId(record.reconciliation_id);
    setError(undefined);
    try {
      await checkExecutionReconciliation(record.reconciliation_id, record.version);
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusyId(undefined);
    }
  }

  async function escalate() {
    if (!escalating) return;
    setBusyId(escalating.reconciliation_id);
    setError(undefined);
    try {
      await escalateExecutionReconciliation(
        escalating.reconciliation_id,
        escalating.version,
        reason.trim(),
      );
      setEscalating(undefined);
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusyId(undefined);
    }
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>外部副作用对账中心</Typography.Title>
        <Alert
          type="warning"
          showIcon
          message="结果未知不等于执行失败，也不允许重新发起写操作"
          description="平台只使用原 operation_id 查询企业系统结果。确认成功后生成一次本地业务结果；持续未知时转人工升级，不会盲目重试库存、支付或设备动作。"
        />
        <Space wrap>
          <Statistic title="待核对" value={totals.pending} />
          <Statistic title="已确认" value={totals.resolved} />
          <Statistic title="人工升级" value={totals.escalated} />
          <Select
            allowClear
            placeholder="全部状态"
            value={status}
            style={{ width: 180 }}
            options={[
              { value: "PENDING", label: "待核对" },
              { value: "RESOLVED", label: "已确认" },
              { value: "ESCALATED", label: "人工升级" },
            ]}
            onChange={(value: ReconciliationStatus | undefined) => setStatus(value)}
          />
          <Button onClick={() => void load()}>刷新</Button>
        </Space>
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!records && !error ? <LoadingState label="正在读取对账任务" /> : null}
        {visibleRecords?.length === 0 ? <EmptyState description="当前筛选条件下没有对账任务" /> : null}
        <List
          dataSource={visibleRecords}
          renderItem={(record) => (
            <List.Item>
              <Card
                id={record.reconciliation_id}
                style={{ width: "100%" }}
                title={record.reconciliation_id}
                extra={<StatusTag status={record.status} version={record.version} />}
              >
                <div className="page-stack">
                  {record.tool_id === "customer.notify" ? (
                    <Alert type="info" showIcon message="Portal 已发布，供应商已受理，最终送达待确认" description="查询只读取原 operation_id 的供应商状态，不会重新发送通知。失败或退订后需重新创建提案并完成新的审批。" />
                  ) : null}
                  {record.tool_id === "purchase.request" ? (
                    <Alert
                      type="info"
                      showIcon
                      message="ERP 采购申请结果待确认"
                      description="平台只使用原 operation_id 查询 ERP 受理结果，不会重新提交采购申请。受理仅代表 ERP 接收了申请，不代表已选供应商、通过预算审批或生成采购订单。"
                    />
                  ) : null}
                  {record.tool_id === "refund.request" ? (
                    <Alert
                      type="info"
                      showIcon
                      message="企业财务退款申请结果待确认"
                      description="平台只使用原 operation_id 查询财务系统，不会重新提交退款申请或发起付款。财务接收状态与支付状态分别记录，只有受信 PAID 回执才表示已支付。"
                    />
                  ) : null}
                  {record.tool_id === "parts.issue" ? (
                    <Alert
                      type="warning"
                      showIcon
                      message="WMS 出库结果待确认"
                      description="平台只按原 operation_id 查询 WMS，不会重新提交出库；明确匹配成功前，预留不推进为 ISSUED，现场用料保持关闭。"
                    />
                  ) : null}
                  {record.tool_id === "parts.consume" || record.tool_id === "parts.return" ? (
                    <Alert
                      type="warning"
                      showIcon
                      message={record.tool_id === "parts.consume" ? "WMS 实际消耗结果待确认" : "WMS 退料结果待确认"}
                      description={record.tool_id === "parts.consume"
                        ? "平台只按原 operation_id 查询 WMS，不会重新提交实际消耗；明确匹配成功前不计入最终消耗。"
                        : "平台只按原 operation_id 查询 WMS，不会重新提交退料；明确匹配成功前不计入最终退料。"}
                    />
                  ) : null}
                  {record.tool_id === "work_order.assign" && record.status !== "ESCALATED" ? (
                    <Alert
                      type="info"
                      showIcon
                      message="企业 FSM 派工结果待确认"
                      description="平台只使用原 operation_id 查询企业 FSM，不会重新提交、自动换人或绕过审批。结果确认前 WorkOrder 保持待派工，但竞争派工与改期已被占位阻止。"
                    />
                  ) : null}
                  {record.tool_id === "work_order.assign" && record.status === "ESCALATED" ? (
                    <Alert
                      type="error"
                      showIcon
                      message="FSM 已接收但本地派工需要人工处理"
                      description="平台保留外部已接收证据，不覆盖异常 WorkOrder，也不会创建重复 Assignment。请人工核对占位、状态与版本后处理。"
                    />
                  ) : null}
                  <FactGrid facts={[
                    ["工具", record.tool_id],
                    ["结果", record.outcome],
                    ["故障单", record.incident_id],
                    ["操作号", record.operation_id],
                    ["初始尝试", record.initial_execution_attempt_id],
                    ["结果尝试", record.resolved_execution_attempt_id],
                    ["外部记录", record.external_reference_id],
                    ["最近核对", record.last_checked_at],
                    ["原因", record.reason],
                  ]} />
                  <Space wrap>
                    <Link href={`/incidents/${record.incident_id}`}>查看故障上下文</Link>
                    {record.legal_actions.includes("CHECK") ? (
                      <Button
                        type="primary"
                        loading={busyId === record.reconciliation_id}
                        onClick={() => void check(record)}
                      >
                        {reconciliationCheckLabel(record.tool_id)}
                      </Button>
                    ) : null}
                    {record.legal_actions.includes("ESCALATE") ? (
                      <Button danger onClick={() => setEscalating(record)}>
                        转人工核对
                      </Button>
                    ) : null}
                  </Space>
                </div>
              </Card>
            </List.Item>
          )}
        />
      </div>
      <Modal
        open={Boolean(escalating)}
        title="转人工核对"
        okText="确认升级"
        cancelText="取消"
        okButtonProps={{ danger: true, disabled: reason.trim().length < 10 }}
        confirmLoading={Boolean(escalating && busyId === escalating.reconciliation_id)}
        onCancel={() => setEscalating(undefined)}
        onOk={() => void escalate()}
      >
        <Alert
          type="info"
          showIcon
          message="升级不会把未知结果标记为失败，也不会创建第二次外部请求。"
          style={{ marginBottom: 16 }}
        />
        <Input.TextArea value={reason} rows={4} maxLength={255} showCount onChange={(event) => setReason(event.target.value)} />
      </Modal>
    </AppShell>
  );
}

function reconciliationCheckLabel(toolId: string): string {
  if (toolId === "customer.notify") return "查询通知结果";
  if (toolId === "purchase.request") return "查询 ERP 采购申请结果";
  if (toolId === "refund.request") return "查询财务退款申请结果";
  if (toolId === "parts.issue") return "查询 WMS 出库结果";
  if (toolId === "parts.consume") return "查询 WMS 实际消耗结果";
  if (toolId === "parts.return") return "查询 WMS 退料结果";
  if (toolId === "work_order.assign") return "查询原派工状态";
  return "使用原操作号查询结果";
}
