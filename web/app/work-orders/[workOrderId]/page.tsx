"use client";

import { Alert, Button, Card, Input, List, Select, Space, Tag, Typography } from "antd";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import {
  type WorkOrderControl,
  type WorkOrderPriority,
  type WorkOrderRepairHistory,
  type WorkOrderView,
  acceptWorkOrder,
  assignWorkOrder,
  closeWorkOrder,
  completeWorkOrder,
  escalateWorkOrder,
  getWorkOrder,
  getWorkOrderRepairHistory,
  holdWorkOrder,
  listWorkOrderControls,
  proposeWorkOrderClosure,
  replanWorkOrder,
  rescheduleWorkOrder,
  resumeWorkOrder,
  startWorkOrder,
  verifyWorkOrder,
} from "@/lib/api/client";

export default function WorkOrderPage() {
  const { workOrderId } = useParams<{ workOrderId: string }>();
  const [work, setWork] = useState<WorkOrderView>();
  const [controls, setControls] = useState<WorkOrderControl[]>();
  const [repairHistory, setRepairHistory] = useState<WorkOrderRepairHistory>();
  const [assignee, setAssignee] = useState("");
  const [rootCause, setRootCause] = useState("");
  const [action, setAction] = useState("");
  const [evidenceId, setEvidenceId] = useState("");
  const [partReservationId, setPartReservationId] = useState("");
  const [costAmount, setCostAmount] = useState("");
  const [customerConfirmation, setCustomerConfirmation] = useState("");
  const [verifyReason, setVerifyReason] = useState("现场复核通过");
  const [closureReason, setClosureReason] = useState("完工、验收、费用和客户确认事实均已齐备");
  const [closureProposalId, setClosureProposalId] = useState<string>();
  const [controlReason, setControlReason] = useState("等待现场条件恢复");
  const [recoveryCondition, setRecoveryCondition] = useState("现场安全条件满足后恢复");
  const [responsibleSubjectId, setResponsibleSubjectId] = useState("");
  const [priority, setPriority] = useState<WorkOrderPriority>("NORMAL");
  const [replanTarget, setReplanTarget] = useState<"READY" | "IN_PROGRESS">("IN_PROGRESS");
  const [windowStart, setWindowStart] = useState(() => localDateTime(1));
  const [windowEnd, setWindowEnd] = useState(() => localDateTime(3));
  const [slaDue, setSlaDue] = useState(() => localDateTime(4));
  const [error, setError] = useState<unknown>();
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const [workResult, controlResult, historyResult] = await Promise.all([
        getWorkOrder(workOrderId),
        listWorkOrderControls(workOrderId),
        getWorkOrderRepairHistory(workOrderId),
      ]);
      setWork(workResult.workOrder);
      setControls(controlResult.controls);
      setRepairHistory(historyResult.history);
      setPriority(workResult.workOrder.priority as WorkOrderPriority);
      setResponsibleSubjectId(workResult.workOrder.assigned_subject_id ?? "");
      if (workResult.workOrder.service_window_start) {
        setWindowStart(localInputFromIso(workResult.workOrder.service_window_start));
      }
      if (workResult.workOrder.service_window_end) {
        setWindowEnd(localInputFromIso(workResult.workOrder.service_window_end));
      }
      if (workResult.workOrder.sla_due_at) {
        setSlaDue(localInputFromIso(workResult.workOrder.sla_due_at));
      }
    } catch (caught) {
      setError(caught);
    }
  }, [workOrderId]);

  useEffect(() => { void load(); }, [load]);

  async function transition(
    operation: (current: WorkOrderView) => Promise<{ workOrder: WorkOrderView }>,
  ) {
    if (!work) return;
    setBusy(true);
    setError(undefined);
    try {
      const result = await operation(work);
      setWork(result.workOrder);
      const [controlResult, historyResult] = await Promise.all([
        listWorkOrderControls(workOrderId),
        getWorkOrderRepairHistory(workOrderId),
      ]);
      setControls(controlResult.controls);
      setRepairHistory(historyResult.history);
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  const plannedWindow = () => ({
    service_window_start: new Date(windowStart).toISOString(),
    service_window_end: new Date(windowEnd).toISOString(),
    sla_due_at: new Date(slaDue).toISOString(),
  });

  async function proposeClosure() {
    if (!work) return;
    setBusy(true);
    setError(undefined);
    try {
      const result = await proposeWorkOrderClosure(workOrderId, {
        work_order_version: work.version,
        reason: closureReason.trim(),
      });
      setClosureProposalId(result.proposal.proposal_id);
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  const serviceAuthorized = work?.creation_mode === "SERVICE_AUTHORIZATION";

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>工单履约与异常管控</Typography.Title>
        <Alert
          type="info"
          showIcon
          message="显式状态机与追加式控制记录"
          description="所有命令都校验服务端版本。暂停、恢复、升级、重排和重新计划不会改写历史，冲突后请重新读取。"
        />
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!work && !error ? <LoadingState /> : null}
        {work ? (
          <>
            <Card
              title={work.work_order_id}
              extra={<StatusTag status={work.status} version={work.version} />}
            >
              <Space wrap style={{ marginBottom: 16 }}>
                <Tag color={priorityColor(work.priority)}>优先级 {work.priority}</Tag>
                <Tag color={slaColor(work.sla_status)}>SLA {work.sla_status}</Tag>
                <Tag color="geekblue">当前第 {repairHistory?.current_round ?? 1} 轮</Tag>
              </Space>
              <FactGrid facts={[
                ["Incident", work.incident_id],
                ["提案", work.proposal_id],
                ["创建来源", serviceAuthorized ? "服务授权（无初始备件）" : "备件预留"],
                ["授权类型", work.authorization_type],
                ["诊断绑定", work.diagnosis_run_id ? `${work.diagnosis_run_id} / v${work.diagnosis_version}` : null],
                ["服务报价", work.service_quotation_id],
                ["零件预留", serviceAuthorized ? "无需初始备件" : work.reservation_id],
                ["负责人", work.assigned_subject_id],
                ["服务窗口开始", formatTimestamp(work.service_window_start)],
                ["服务窗口结束", formatTimestamp(work.service_window_end)],
                ["SLA 截止", formatTimestamp(work.sla_due_at)],
                ["更新时间", formatTimestamp(work.updated_at)],
              ]} />
              <Space wrap>
                <Link href="/work-orders">返回工单运营中心</Link>
                <Link href={`/incidents/${work.incident_id}`}>查看 Incident 诊断事实</Link>
                {work.status === "CLOSED" ? (
                  <Link href={`/work-orders/${workOrderId}/report`}>查看正式关闭报告</Link>
                ) : null}
              </Space>
            </Card>

            <Card
              title="维修轮次与独立验收历史"
              extra={<Tag color="geekblue">共 {repairHistory?.rounds.length ?? 0} 轮</Tag>}
            >
              {!repairHistory ? <LoadingState label="正在读取维修历史" /> : null}
              {repairHistory?.rounds.length === 0 ? (
                <EmptyState description="尚未产生完工与验收记录" />
              ) : null}
              <List
                dataSource={repairHistory?.rounds}
                renderItem={(round) => (
                  <List.Item>
                    <List.Item.Meta
                      title={(
                        <Space wrap>
                          <strong>第 {round.round_number} 轮</strong>
                          <Tag color={repairRoundColor(round.status)}>
                            {repairRoundLabel(round.status)}
                          </Tag>
                          {round.verification_passed === false ? (
                            <Tag color="red">验收未通过</Tag>
                          ) : null}
                          {round.verification_passed === true ? (
                            <Tag color="green">独立验收通过</Tag>
                          ) : null}
                        </Space>
                      )}
                      description={(
                        <div className="page-stack" style={{ gap: 8 }}>
                          {round.rework_reason ? (
                            <Alert
                              type="warning"
                              showIcon
                              message={`返工要求：${round.rework_reason}`}
                              description={`本轮仅接受现场记录 #${(round.entry_sequence_checkpoint ?? 0) + 1} 之后的新步骤、新证据和新客户签字。`}
                            />
                          ) : null}
                          <FactGrid facts={[
                            ["根因", round.root_cause],
                            ["处理动作", round.actions.join("；") || null],
                            ["证据", round.evidence_ids.join("、") || null],
                            ["领料记录", round.part_reservation_ids.join("、") || null],
                            ["客户确认", round.customer_confirmation],
                            ["现场记录范围", sequenceRange(round.field_entry_sequence_start, round.field_entry_sequence_end)],
                            ["完工时间", formatTimestamp(round.completed_at)],
                            ["验收说明", round.verification_reason],
                            ["验收时间", formatTimestamp(round.verified_at)],
                          ]} />
                        </div>
                      )}
                    />
                  </List.Item>
                )}
              />
            </Card>

            <Card title="服务端状态允许的履约操作">
              <div className="form-grid">
                {work.legal_actions.includes("ASSIGN") ? (
                  <Space.Compact block>
                    <Input value={assignee} onChange={(event) => setAssignee(event.target.value)} placeholder="负责人 subject_id" />
                    <Button type="primary" loading={busy} disabled={!assignee.trim()} onClick={() => void transition((current) => assignWorkOrder(workOrderId, current.version, assignee.trim()))}>派单</Button>
                  </Space.Compact>
                ) : null}
                {work.legal_actions.includes("ACCEPT") ? <Button type="primary" loading={busy} onClick={() => void transition((current) => acceptWorkOrder(workOrderId, current.version))}>接单</Button> : null}
                {work.legal_actions.includes("START") ? <Button type="primary" loading={busy} onClick={() => void transition((current) => startWorkOrder(workOrderId, current.version))}>开始处理</Button> : null}
                {work.legal_actions.includes("COMPLETE") ? (
                  <>
                    <Input value={rootCause} onChange={(event) => setRootCause(event.target.value)} addonBefore="根因" />
                    <Input value={action} onChange={(event) => setAction(event.target.value)} addonBefore="处理动作" />
                    <Input value={evidenceId} onChange={(event) => setEvidenceId(event.target.value)} addonBefore="完工证据 ID" />
                    {!serviceAuthorized ? <Input value={partReservationId} onChange={(event) => setPartReservationId(event.target.value)} addonBefore="领料记录 ID" /> : null}
                    <Input value={costAmount} onChange={(event) => setCostAmount(event.target.value)} addonBefore="费用金额" />
                    <Input value={customerConfirmation} onChange={(event) => setCustomerConfirmation(event.target.value)} addonBefore="客户确认" />
                    <Button
                      type="primary"
                      loading={busy}
                      disabled={!rootCause.trim() || !action.trim() || !evidenceId.trim() || (serviceAuthorized && !costAmount.trim())}
                      onClick={() => void transition((current) => completeWorkOrder(
                        workOrderId,
                        current.version,
                        {
                          root_cause: rootCause.trim(),
                          actions: [action.trim()],
                          evidence_ids: [evidenceId.trim()],
                          part_reservation_ids: !serviceAuthorized && partReservationId.trim() ? [partReservationId.trim()] : [],
                          cost_amount: costAmount.trim() || null,
                          customer_confirmation: customerConfirmation.trim() || null,
                        },
                      ))}
                    >提交完工记录</Button>
                  </>
                ) : null}
                {work.legal_actions.includes("VERIFY") ? (
                  <>
                    <Input value={verifyReason} onChange={(event) => setVerifyReason(event.target.value)} addonBefore="独立验收说明" />
                    <Space>
                      <Button type="primary" loading={busy} onClick={() => void transition((current) => verifyWorkOrder(workOrderId, current.version, true, verifyReason))}>验收通过</Button>
                      <Button danger loading={busy} onClick={() => void transition((current) => verifyWorkOrder(workOrderId, current.version, false, verifyReason))}>退回处理</Button>
                    </Space>
                  </>
                ) : null}
                {work.legal_actions.includes("CLOSE") ? (
                  <>
                    <Input.TextArea rows={2} value={closureReason} onChange={(event) => setClosureReason(event.target.value)} placeholder="Agent 关闭提案理由" />
                    <Space wrap>
                      <Button loading={busy} onClick={() => void transition((current) => closeWorkOrder(workOrderId, current.version))}>人工直接关闭</Button>
                      <Button type="primary" loading={busy} disabled={closureReason.trim().length < 3} onClick={() => void proposeClosure()}>创建 Agent 关闭提案</Button>
                    </Space>
                    {closureProposalId ? (
                      <Alert
                        type="success"
                        showIcon
                        message={`关闭提案 ${closureProposalId} 已创建`}
                        description={<Link href="/approvals">进入审批箱完成独立审批与执行</Link>}
                      />
                    ) : null}
                  </>
                ) : null}
                {work.status === "CLOSED" ? <Alert type="success" showIcon message="工单已闭环" /> : null}
              </div>
            </Card>

            <Card title="异常控制与 SLA 计划">
              <div className="form-grid">
                <Input.TextArea rows={2} value={controlReason} onChange={(event) => setControlReason(event.target.value)} placeholder="操作原因" />
                <Input.TextArea rows={2} value={recoveryCondition} onChange={(event) => setRecoveryCondition(event.target.value)} placeholder="恢复条件" />
                <Input value={responsibleSubjectId} onChange={(event) => setResponsibleSubjectId(event.target.value)} addonBefore="责任人" />
                <Space wrap>
                  {work.legal_actions.includes("HOLD") ? <Button loading={busy} disabled={!controlReason.trim() || !recoveryCondition.trim()} onClick={() => void transition((current) => holdWorkOrder(workOrderId, current.version, { reason: controlReason.trim(), recovery_condition: recoveryCondition.trim() }))}>暂停工单</Button> : null}
                  {work.legal_actions.includes("RESUME") ? <Button type="primary" loading={busy} disabled={!controlReason.trim()} onClick={() => void transition((current) => resumeWorkOrder(workOrderId, current.version, controlReason.trim()))}>恢复执行</Button> : null}
                  {work.legal_actions.includes("ESCALATE") ? <Button danger loading={busy} disabled={!controlReason.trim() || !recoveryCondition.trim() || !responsibleSubjectId.trim()} onClick={() => void transition((current) => escalateWorkOrder(workOrderId, current.version, { reason: controlReason.trim(), recovery_condition: recoveryCondition.trim(), responsible_subject_id: responsibleSubjectId.trim() }))}>升级处置</Button> : null}
                </Space>

                <Space wrap>
                  <Select<WorkOrderPriority> value={priority} options={[
                    { label: "紧急 CRITICAL", value: "CRITICAL" },
                    { label: "高 HIGH", value: "HIGH" },
                    { label: "普通 NORMAL", value: "NORMAL" },
                    { label: "低 LOW", value: "LOW" },
                  ]} onChange={setPriority} style={{ minWidth: 180 }} />
                  <label>窗口开始 <Input type="datetime-local" value={windowStart} onChange={(event) => setWindowStart(event.target.value)} /></label>
                  <label>窗口结束 <Input type="datetime-local" value={windowEnd} onChange={(event) => setWindowEnd(event.target.value)} /></label>
                  <label>SLA 截止 <Input type="datetime-local" value={slaDue} onChange={(event) => setSlaDue(event.target.value)} /></label>
                </Space>
                {work.legal_actions.includes("RESCHEDULE") ? (
                  <Button loading={busy} disabled={!controlReason.trim() || !windowStart || !windowEnd || !slaDue} onClick={() => void transition((current) => rescheduleWorkOrder(workOrderId, current.version, { priority, reason: controlReason.trim(), ...plannedWindow() }))}>保存优先级与服务窗口</Button>
                ) : null}
                {work.legal_actions.includes("REPLAN") ? (
                  <Space wrap>
                    <Select value={replanTarget} options={[
                      { label: "回到待派工", value: "READY" },
                      { label: "恢复处理中", value: "IN_PROGRESS" },
                    ]} onChange={setReplanTarget} style={{ minWidth: 180 }} />
                    <Button type="primary" loading={busy} disabled={!controlReason.trim() || (replanTarget === "IN_PROGRESS" && !responsibleSubjectId.trim())} onClick={() => void transition((current) => replanWorkOrder(workOrderId, current.version, { target_status: replanTarget, reason: controlReason.trim(), responsible_subject_id: responsibleSubjectId.trim() || null, ...plannedWindow() }))}>提交重新计划</Button>
                  </Space>
                ) : null}
              </div>
            </Card>

            <Card title="追加式控制记录">
              {controls?.length === 0 ? <EmptyState description="尚无暂停、升级或重排记录" /> : null}
              <List
                dataSource={controls}
                renderItem={(control) => (
                  <List.Item>
                    <List.Item.Meta
                      title={<Space><Tag>{control.command_type}</Tag><span>{control.previous_status} → {control.target_status}</span></Space>}
                      description={`${formatTimestamp(control.occurred_at)} · v${control.work_order_version} · ${control.reason}${control.recovery_condition ? ` · 恢复条件：${control.recovery_condition}` : ""}`}
                    />
                  </List.Item>
                )}
              />
            </Card>
          </>
        ) : null}
      </div>
    </AppShell>
  );
}

function localDateTime(hoursFromNow: number): string {
  return localInputFromIso(new Date(Date.now() + hoursFromNow * 60 * 60 * 1000).toISOString());
}

function localInputFromIso(value: string): string {
  const date = new Date(value);
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60 * 1000);
  return local.toISOString().slice(0, 16);
}

function formatTimestamp(value: string | null): string {
  if (!value) return "—";
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime())
    ? value
    : timestamp.toLocaleString("zh-CN", { hour12: false });
}

function priorityColor(priority: string): string {
  return priority === "CRITICAL" ? "red" : priority === "HIGH" ? "orange" : "blue";
}

function slaColor(status: string): string {
  return status === "BREACHED" ? "red" : status === "AT_RISK" ? "orange" : "green";
}

function repairRoundLabel(status: string): string {
  return {
    IN_PROGRESS: "处理中",
    REWORK_IN_PROGRESS: "返工处理中",
    AWAITING_VERIFICATION: "等待独立验收",
    REWORK_REQUIRED: "验收退回",
    VERIFIED: "已验收",
  }[status] ?? status;
}

function repairRoundColor(status: string): string {
  if (status === "VERIFIED") return "green";
  if (status === "REWORK_REQUIRED") return "red";
  if (status === "REWORK_IN_PROGRESS") return "orange";
  return "blue";
}

function sequenceRange(start: number | null, end: number | null): string | null {
  return start === null || end === null ? null : `#${start} — #${end}`;
}
