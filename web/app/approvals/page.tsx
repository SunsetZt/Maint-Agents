"use client";

import { Alert, Button, Card, Input, List, Modal, Space, Typography } from "antd";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import {
  ApiClientError,
  type ApprovalExecution,
  type ApprovalView,
  decideApproval,
  executeApproval,
  listApprovals,
} from "@/lib/api/client";

export default function ApprovalsPage() {
  const [approvals, setApprovals] = useState<ApprovalView[]>();
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [loadError, setLoadError] = useState<unknown>();
  const [actionError, setActionError] = useState<{
    approvalId: string;
    error: unknown;
  }>();
  const [busyId, setBusyId] = useState<string>();
  const [execution, setExecution] = useState<ApprovalExecution>();
  const [executionToolId, setExecutionToolId] = useState<string>();
  const [reason, setReason] = useState("业务事实与风险边界已复核");
  const [showExpired, setShowExpired] = useState(false);

  const load = useCallback(async () => {
    setLoadError(undefined);
    try {
      setApprovals((await listApprovals()).approvals);
      setNowMs(Date.now());
    } catch (caught) { setLoadError(caught); }
  }, []);
  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    const nextExpiry = approvals
      ?.filter((approval) => approval.status === "PENDING" || approval.status === "APPROVED")
      .map((approval) => Date.parse(approval.expires_at))
      .filter((expiresAt) => Number.isFinite(expiresAt) && expiresAt > nowMs)
      .sort((left, right) => left - right)[0];
    if (nextExpiry === undefined) return;
    const delay = Math.min(Math.max(nextExpiry - Date.now() + 25, 0), 2_147_483_647);
    const timer = window.setTimeout(() => setNowMs(Date.now()), delay);
    return () => window.clearTimeout(timer);
  }, [approvals, nowMs]);

  const approvalGroups = useMemo(() => {
    if (!approvals) return undefined;
    const current: ApprovalView[] = [];
    const expired: ApprovalView[] = [];
    for (const approval of approvals) {
      (isApprovalExpired(approval, nowMs) ? expired : current).push(approval);
    }
    current.sort(compareApprovals);
    expired.sort(compareApprovals);
    return {
      current,
      expired,
      visible: showExpired ? [...current, ...expired] : current,
    };
  }, [approvals, nowMs, showExpired]);

  async function decide(approval: ApprovalView, decision: "APPROVED" | "REJECTED") {
    setBusyId(approval.approval_id);
    setActionError(undefined);
    try {
      await decideApproval(approval.approval_id, approval.version, decision, reason);
      await load();
    } catch (caught) {
      setActionError({ approvalId: approval.approval_id, error: caught });
      if (caught instanceof ApiClientError && caught.code === "approval_conflict") {
        await load();
      }
    } finally { setBusyId(undefined); }
  }

  async function execute(approval: ApprovalView) {
    setBusyId(approval.approval_id);
    setActionError(undefined);
    try {
      setExecutionToolId(approval.tool_id);
      setExecution((await executeApproval(
        approval.approval_id,
        approval.version,
        approval.parameters,
        crypto.randomUUID(),
      )).execution);
      await load();
    } catch (caught) {
      setActionError({ approvalId: approval.approval_id, error: caught });
    } finally { setBusyId(undefined); }
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>高风险操作审批箱</Typography.Title>
        <Alert type="warning" showIcon message="职责分离与执行时复核" description="发起人不能审批自己的请求；批准后执行仍会重新校验参数绑定、资源版本和幂等状态。零件预留会复核库存，客户通知会按审批可见内容原样发布。" />
        <Input value={reason} onChange={(event) => setReason(event.target.value)} addonBefore="审批理由" />
        {loadError ? <ErrorState error={loadError} onRetry={() => void load()} /> : null}
        {!approvals && !loadError ? <LoadingState /> : null}
        {approvals?.length === 0 ? <EmptyState description="当前身份没有可见审批" /> : null}
        {approvalGroups && approvals?.length ? (
          <Alert
            type={approvalGroups.current.length > 0 ? "success" : "warning"}
            showIcon
            message={approvalGroups.current.length > 0
              ? `当前有 ${approvalGroups.current.length} 条未过期审批`
              : "当前没有未过期审批"}
            description={approvalGroups.expired.length > 0
              ? `已默认隐藏 ${approvalGroups.expired.length} 条过期记录；重新发起的有效审批会优先显示。`
              : "审批已按可操作性和有效期排序，最新有效请求优先显示。"}
            action={(
              <Space wrap>
                <Button onClick={() => void load()}>刷新审批箱</Button>
                {approvalGroups.expired.length > 0 ? (
                  <Button onClick={() => setShowExpired((current) => !current)}>
                    {showExpired
                      ? "隐藏过期记录"
                      : `查看 ${approvalGroups.expired.length} 条过期记录`}
                  </Button>
                ) : null}
              </Space>
            )}
          />
        ) : null}
        <List
          grid={{ gutter: 16, xs: 1, md: 2 }}
          dataSource={approvalGroups?.visible}
          renderItem={(approval) => {
            const expired = isApprovalExpired(approval, nowMs);
            return (
              <List.Item>
              <Card title={approval.approval_id} extra={<StatusTag status={approval.status} version={approval.version} />}>
                <div className="page-stack">
                  <FactGrid facts={[["工具", approval.tool_id], ["提案", approval.proposal_id], ["资源版本", approval.resource_version], ["过期时间", approval.expires_at]]} />
                  {expired ? (
                    <Alert
                      type="warning"
                      showIcon
                      message="审批已过期，请重新发起"
                      description={approval.status === "APPROVED"
                        ? "该批准授权已超过有效期，不能继续执行。请返回原业务页面重新提交审批。"
                        : "该请求已超过有效期，不能再批准或拒绝。请返回原业务页面重新提交审批。"}
                    />
                  ) : null}
                  {approval.tool_id === "parts.issue" ? (
                    <Alert
                      type="warning"
                      showIcon
                      message="出库是独立高风险动作"
                      description="批准内容固定绑定工单、批准预留、料号和完整数量；执行时重新复核，只有 WMS 明确成功才开放现场用料。"
                    />
                  ) : null}
                  {approval.tool_id === "parts.consume" || approval.tool_id === "parts.return" ? (
                    <Alert
                      type="warning"
                      showIcon
                      message={approval.tool_id === "parts.consume" ? "WMS 实际消耗" : "WMS 退料"}
                      description={approval.tool_id === "parts.consume"
                        ? "批准内容固定绑定不可变 PART 现场记录及其完整数量；执行前会复核工单、出库事实和余额。"
                        : "批准内容固定绑定当前可退数量；执行前会复核已记录用量、成功动作和未终结动作。"}
                    />
                  ) : null}
                  <FsmApprovalBinding approval={approval} />
                  <pre className="json-report">{JSON.stringify(approval.parameters, null, 2)}</pre>
                  {approval.status === "PENDING"
                    && !expired
                    && !approval.legal_actions.includes("APPROVE")
                    && !approval.legal_actions.includes("REJECT") ? (
                      <Alert
                        type="warning"
                        showIcon
                        message="当前账号不能决定这条审批"
                        description="该请求可能由当前账号发起，或当前账号缺少审批权限。请由不同主体且具备审批权限的人员处理。"
                      />
                    ) : null}
                  {actionError?.approvalId === approval.approval_id ? (
                    <ApprovalActionError error={actionError.error} />
                  ) : null}
                  <Space wrap>
                    <Button type="primary" loading={busyId === approval.approval_id} disabled={expired || !approval.legal_actions.includes("APPROVE") || reason.trim().length < 3} onClick={() => void decide(approval, "APPROVED")}>批准</Button>
                    <Button danger loading={busyId === approval.approval_id} disabled={expired || !approval.legal_actions.includes("REJECT") || reason.trim().length < 3} onClick={() => void decide(approval, "REJECTED")}>拒绝</Button>
                    <Button loading={busyId === approval.approval_id} disabled={expired || !approval.legal_actions.includes("EXECUTE")} onClick={() => void execute(approval)}>{approvalExecutionLabel(approval)}</Button>
                  </Space>
                </div>
              </Card>
              </List.Item>
            );
          }}
        />
        <Modal open={Boolean(execution)} footer={null} onCancel={() => { setExecution(undefined); setExecutionToolId(undefined); }} title="幂等执行结果">
          {execution ? (
            <div className="page-stack">
              {execution.assignment_delivery_id ? (
                <Alert
                  type={execution.status === "SUCCEEDED" ? "success" : execution.status === "RECONCILING" ? "warning" : "error"}
                  showIcon
                  message={fsmExecutionLabel(execution.status)}
                  description="企业结果会按原 operation_id 幂等收敛；拒绝、取消或未知结果不会自动换人、重提或降级为本地派工。"
                />
              ) : null}
              {execution.part_issue_id ? (
                <Alert
                  type={execution.status === "SUCCEEDED" ? "success" : execution.status === "RECONCILING" ? "warning" : "error"}
                  showIcon
                  message={execution.status === "SUCCEEDED" ? "WMS 已确认出库" : execution.status === "RECONCILING" ? "WMS 出库结果待确认" : "WMS 出库未完成"}
                  description="结果未知时只按原 operation_id 查询，不会重新提交出库或提前开放现场用料。"
                />
              ) : null}
              {execution.part_movement_id ? (
                <Alert
                  type={execution.status === "SUCCEEDED" ? "success" : execution.status === "RECONCILING" ? "warning" : "error"}
                  showIcon
                  message={partMovementExecutionMessage(executionToolId, execution.status)}
                  description="结果未知时只按原 operation_id 查询，不会重新提交消耗或退料；只有 WMS 明确匹配成功才计入最终核销。"
                />
              ) : null}
              <FactGrid facts={[["状态", execution.status], ["零件预留", execution.reservation_id], ["备件出库事实", execution.part_issue_id], ["物料核销动作", execution.part_movement_id], ["工单", execution.work_order_id], ["客户通知", execution.notification_id], ["采购申请", execution.purchase_request_id], ["退款申请", execution.refund_request_id], ["FSM 派工交接", execution.assignment_delivery_id], ["执行尝试", execution.execution_attempt_id], ["对账任务", execution.reconciliation_id], ["说明", execution.reason]]} />
              {execution.work_order_id ? <Link href={`/work-orders/${execution.work_order_id}`}>进入工单履约</Link> : null}
              {execution.reconciliation_id ? <Link href="/reconciliations">进入外部副作用对账中心</Link> : null}
            </div>
          ) : null}
        </Modal>
      </div>
    </AppShell>
  );
}

function ApprovalActionError({ error }: { error: unknown }) {
  if (error instanceof ApiClientError && error.code === "separation_of_duties") {
    return (
      <Alert
        type="warning"
        showIcon
        message="不能审批自己发起的请求"
        description={
          <div className="page-stack">
            <span>请退出当前账号，改由不同主体且具备审批权限的人员处理。</span>
            {error.requestId ? (
              <span className="request-id">请求标识：{error.requestId}</span>
            ) : null}
          </div>
        }
      />
    );
  }
  if (error instanceof ApiClientError && error.code === "approval_conflict") {
    return (
      <Alert
        type="warning"
        showIcon
        message="审批状态已变化，已刷新最新状态"
        description={error.requestId ? `请求标识：${error.requestId}` : undefined}
      />
    );
  }
  return <ErrorState error={error} />;
}

function isApprovalExpired(approval: ApprovalView, nowMs: number): boolean {
  if (approval.status !== "PENDING" && approval.status !== "APPROVED") return false;
  const expiresAt = Date.parse(approval.expires_at);
  return Number.isFinite(expiresAt) && expiresAt <= nowMs;
}

function compareApprovals(left: ApprovalView, right: ApprovalView): number {
  const actionDifference = approvalActionPriority(left) - approvalActionPriority(right);
  if (actionDifference !== 0) return actionDifference;
  const expiryDifference = approvalExpiryMs(right) - approvalExpiryMs(left);
  if (expiryDifference !== 0) return expiryDifference;
  return right.approval_id.localeCompare(left.approval_id);
}

function approvalActionPriority(approval: ApprovalView): number {
  if (approval.legal_actions.includes("EXECUTE")) return 0;
  if (approval.legal_actions.includes("APPROVE") || approval.legal_actions.includes("REJECT")) return 1;
  return 2;
}

function approvalExpiryMs(approval: ApprovalView): number {
  const expiresAt = Date.parse(approval.expires_at);
  return Number.isFinite(expiresAt) ? expiresAt : 0;
}

function FsmApprovalBinding({ approval }: { approval: ApprovalView }) {
  if (
    approval.tool_id !== "work_order.assign"
    || approval.parameters.assignment_target !== "ENTERPRISE_FSM"
  ) return null;
  const binding = approval.parameters.fsm_binding;
  if (!binding || typeof binding !== "object" || Array.isArray(binding)) return null;
  const safe = binding as Record<string, unknown>;
  const skills = Array.isArray(safe.matched_skill_codes)
    ? safe.matched_skill_codes.map(String).join("、")
    : "—";
  return (
    <Card size="small" title="企业 FSM 候选（审批绑定）">
      <Alert
        type="info"
        showIcon
        message="批准的是固定候选与安全资格快照"
        description="执行时将重新解析同一 Profile 并复核平台人员资格；候选变化时失败关闭，不会自动换人。"
      />
      <FactGrid facts={[
        ["候选", safe.display_name],
        ["Provider", safe.provider],
        ["候选 Profile", safe.candidate_profile_id],
        ["Profile 版本", safe.profile_version],
        ["平台工程师", safe.assignee_subject_id],
        ["匹配技能", skills],
        ["预计路程", `${String(safe.travel_minutes ?? "—")} 分钟`],
        ["剩余工时", `${String(safe.remaining_work_minutes ?? "—")} 分钟`],
        ["有效期", safe.expires_at],
      ]} />
    </Card>
  );
}

function fsmExecutionLabel(status: string): string {
  if (status === "SUCCEEDED") return "FSM 已接收";
  if (status === "REJECTED") return "FSM 已拒绝";
  if (status === "CANCELLED") return "FSM 已取消";
  if (status === "RECONCILING") return "FSM 派工结果待确认";
  return status;
}

function approvalExecutionLabel(approval: ApprovalView): string {
  if (approval.tool_id === "parts.issue") return "执行 WMS 出库";
  if (approval.tool_id === "parts.consume") return "执行 WMS 实际消耗";
  if (approval.tool_id === "parts.return") return "执行 WMS 退料";
  if (approval.tool_id === "customer.notify") return "发布客户通知";
  if (approval.tool_id === "purchase.request") return "提交采购申请";
  if (approval.tool_id === "refund.request") return "提交退款申请";
  if (approval.tool_id === "work_order.assign") {
    return approval.parameters.assignment_target === "ENTERPRISE_FSM"
      ? "执行企业 FSM 派工"
      : "执行本地派工";
  }
  if (approval.tool_id === "work_order.close") return "执行工单关闭";
  return "按批准参数执行";
}

function partMovementExecutionMessage(toolId: string | undefined, status: string): string {
  const action = toolId === "parts.return" ? "退料" : "实际消耗";
  if (status === "SUCCEEDED") return `WMS 已确认${action}`;
  if (status === "RECONCILING") return `WMS ${action}结果待确认`;
  return `WMS ${action}未完成`;
}
