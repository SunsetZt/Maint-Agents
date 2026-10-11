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
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import {
  type DispatchFact,
  type DispatchWorkOrder,
  type DispatchWorkOrderStatus,
  type WorkOrderAssignmentOption,
  assignWorkOrder,
  listDispatchWorkOrders,
  listWorkOrderAssignmentOptions,
  proposeWorkOrderAssignment,
} from "@/lib/api/client";

const PAGE_SIZE = 20;
const STATUS_OPTIONS: Array<{ label: string; value: DispatchWorkOrderStatus }> = [
  { label: "待派工", value: "READY" },
  { label: "已派工", value: "ASSIGNED" },
  { label: "已接单", value: "ACCEPTED" },
  { label: "处理中", value: "IN_PROGRESS" },
  { label: "待验收", value: "COMPLETED" },
  { label: "已验收", value: "VERIFIED" },
  { label: "已关闭", value: "CLOSED" },
];

type AssignmentProposalState = {
  approvalId: string;
  target: "ENTERPRISE_FSM" | "LOCAL_DIRECTORY";
};

export default function DispatchPage() {
  const [workOrders, setWorkOrders] = useState<DispatchWorkOrder[]>();
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState<DispatchWorkOrderStatus>();
  const [assigneeInput, setAssigneeInput] = useState("");
  const [assigneeFilter, setAssigneeFilter] = useState("");
  const [offset, setOffset] = useState(0);
  const [assignments, setAssignments] = useState<Record<string, string>>({});
  const [assignmentReasons, setAssignmentReasons] = useState<Record<string, string>>({});
  const [assignmentOptions, setAssignmentOptions] = useState<Record<string, {
    options: WorkOrderAssignmentOption[];
    fsmUnavailableReason: string | null;
  }>>({});
  const [assignmentProposals, setAssignmentProposals] = useState<
    Record<string, AssignmentProposalState>
  >({});
  const assignmentProposalLocks = useRef(new Set<string>());
  const [busyId, setBusyId] = useState<string>();
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const result = await listDispatchWorkOrders({
        ...(status ? { status } : {}),
        ...(assigneeFilter ? { assigned_subject_id: assigneeFilter } : {}),
        limit: PAGE_SIZE,
        offset,
      });
      setWorkOrders(result.workOrders);
      setTotal(result.total);
      setRequestId(result.requestId);
      const optionEntries = await Promise.all(
        result.workOrders
          .filter((workOrder) => workOrder.legal_actions.includes("ASSIGN"))
          .map(async (workOrder) => {
            try {
              const resolved = await listWorkOrderAssignmentOptions(
                workOrder.work_order_id,
              );
              return [workOrder.work_order_id, resolved] as const;
            } catch {
              return [workOrder.work_order_id, {
                options: [],
                fsmUnavailableReason: "fsm_assignment_options_unavailable",
              }] as const;
            }
          }),
      );
      setAssignmentOptions(Object.fromEntries(optionEntries));
    } catch (caught) {
      setError(caught);
    }
  }, [assigneeFilter, offset, status]);

  useEffect(() => { void load(); }, [load]);

  const summary = useMemo(() => ({
    ready: workOrders?.filter((item) => item.status === "READY").length ?? 0,
    active: workOrders?.filter((item) =>
      ["ASSIGNED", "ACCEPTED", "IN_PROGRESS"].includes(item.status)).length ?? 0,
    pendingVerification: workOrders?.filter((item) => item.status === "COMPLETED").length ?? 0,
  }), [workOrders]);

  async function assign(workOrder: DispatchWorkOrder) {
    const assignee = assignments[workOrder.work_order_id]?.trim();
    if (!assignee) return;
    setBusyId(workOrder.work_order_id);
    setError(undefined);
    try {
      await assignWorkOrder(workOrder.work_order_id, workOrder.version, assignee);
      setAssignments((current) => ({ ...current, [workOrder.work_order_id]: "" }));
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusyId(undefined);
    }
  }

  async function proposeAssignment(workOrder: DispatchWorkOrder) {
    const workOrderId = workOrder.work_order_id;
    const assignee = assignments[workOrder.work_order_id]?.trim();
    const reason = assignmentReasons[workOrder.work_order_id]?.trim();
    if (
      !assignee
      || !reason
      || reason.length < 3
      || assignmentProposals[workOrderId]
      || assignmentProposalLocks.current.has(workOrderId)
    ) return;
    assignmentProposalLocks.current.add(workOrderId);
    setBusyId(workOrderId);
    setError(undefined);
    try {
      const result = await proposeWorkOrderAssignment(workOrderId, {
        work_order_version: workOrder.version,
        assignee_subject_id: assignee,
        reason,
      });
      setAssignmentProposals((current) => ({
        ...current,
        [workOrderId]: {
          approvalId: result.proposal.approval_id,
          target: "LOCAL_DIRECTORY",
        },
      }));
    } catch (caught) {
      assignmentProposalLocks.current.delete(workOrderId);
      setError(caught);
    } finally {
      setBusyId(undefined);
    }
  }

  async function proposeEnterpriseAssignment(
    workOrder: DispatchWorkOrder,
    candidateProfileId: string,
    reason: string,
  ) {
    const workOrderId = workOrder.work_order_id;
    const normalizedReason = reason.trim();
    if (
      !candidateProfileId
      || normalizedReason.length < 3
      || assignmentProposals[workOrderId]
      || assignmentProposalLocks.current.has(workOrderId)
    ) return;
    assignmentProposalLocks.current.add(workOrderId);
    setBusyId(workOrderId);
    setError(undefined);
    try {
      const result = await proposeWorkOrderAssignment(workOrderId, {
        work_order_version: workOrder.version,
        assignment_target: "ENTERPRISE_FSM",
        candidate_profile_id: candidateProfileId,
        reason: normalizedReason,
      });
      setAssignmentProposals((current) => ({
        ...current,
        [workOrderId]: {
          approvalId: result.proposal.approval_id,
          target: "ENTERPRISE_FSM",
        },
      }));
    } catch (caught) {
      assignmentProposalLocks.current.delete(workOrderId);
      setError(caught);
    } finally {
      setBusyId(undefined);
    }
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>工单调度中心</Typography.Title>
        <Alert
          type="info"
          showIcon
          message="实时排班与库存辅助派工"
          description="队列只展示当前身份设备或站点范围内的工单。FSM 排班与 WMS 库存是带来源和数据时点的只读事实；系统不会因为推荐结果自动派工。"
        />

        <Card title="调度筛选" extra={<Button onClick={() => void load()}>刷新实时事实</Button>}>
          <Space wrap>
            <Select<DispatchWorkOrderStatus>
              allowClear
              style={{ minWidth: 160 }}
              placeholder="全部工单状态"
              options={STATUS_OPTIONS}
              value={status}
              onChange={(value) => { setStatus(value); setOffset(0); }}
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

        <Card title="当前页业务概览">
          <FactGrid facts={[
            ["符合条件工单", total],
            ["当前页待派工", summary.ready],
            ["当前页执行中", summary.active],
            ["当前页待验收", summary.pendingVerification],
            ["请求标识", requestId],
          ]} />
        </Card>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!workOrders && !error ? <LoadingState label="正在读取工单、排班与库存" /> : null}
        {workOrders?.length === 0 ? <EmptyState description="当前筛选条件下没有可调度工单" /> : null}

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
                  <Typography.Text type="secondary" ellipsis>
                    {workOrder.incident_description}
                  </Typography.Text>
                  <FactGrid facts={[
                    ["工单", workOrder.work_order_id],
                    ["设备型号", workOrder.model_code],
                    ["服务站点", workOrder.site_name ?? workOrder.site_id],
                    ["当前负责人", workOrder.assigned_subject_id],
                    ["更新时间", formatTimestamp(workOrder.updated_at)],
                  ]} />

                  <Space direction="vertical" size="middle">
                    <LiveFact
                      title="FSM 工程师排班"
                      fact={workOrder.schedule}
                      failure={failureFor(workOrder, "schedule.availability")}
                      facts={workOrder.schedule ? [
                        ["可用工程师", factValue(workOrder.schedule, "available_technician_count")],
                        ["最早开始", formatTimestamp(factValue(workOrder.schedule, "next_available_start"))],
                        ["窗口结束", formatTimestamp(factValue(workOrder.schedule, "next_available_end"))],
                      ] : []}
                    />
                    <LiveFact
                      title="WMS 备件库存"
                      fact={workOrder.inventory}
                      failure={failureFor(workOrder, "parts.availability")}
                      facts={workOrder.inventory ? [
                        ["推荐料号", factValue(workOrder.inventory, "part_number")],
                        ["可用数量", factValue(workOrder.inventory, "available_quantity")],
                      ] : []}
                    />
                  </Space>

                  {workOrder.pending_assignment_operation_id ? (
                    <Alert
                      type="warning"
                      showIcon
                      message="FSM 派工结果待确认"
                      description={(
                        <Space direction="vertical" size="small">
                          <Typography.Text>
                            原操作号 {workOrder.pending_assignment_operation_id} 正在只读对账。
                            在结果收敛前不会显示派工或改期入口。
                          </Typography.Text>
                          <Link href="/reconciliations">进入对账中心</Link>
                        </Space>
                      )}
                    />
                  ) : null}

                  {workOrder.legal_actions.includes("ASSIGN") ? (
                    <div className="page-stack">
                      <EnterpriseAssignmentPanel
                        state={assignmentOptions[workOrder.work_order_id]}
                        busy={busyId === workOrder.work_order_id}
                        submittedProposal={assignmentProposals[workOrder.work_order_id]}
                        onSubmit={(candidateProfileId, reason) => {
                          void proposeEnterpriseAssignment(
                            workOrder,
                            candidateProfileId,
                            reason,
                          );
                        }}
                      />
                      <Card size="small" title="本地目录兼容派工">
                        <div className="page-stack">
                          <Alert
                            type="info"
                            showIcon
                            message="兼容与离线路径"
                            description="该路径只复核平台本地人员目录资格，不宣称已通过企业 FSM 的技能、工时、距离或劳动规则。"
                          />
                          <Input
                            placeholder="输入已登记现场工程师 subject_id"
                            disabled={Boolean(assignmentProposals[workOrder.work_order_id])}
                            value={assignments[workOrder.work_order_id] ?? ""}
                            onChange={(event) => setAssignments((current) => ({
                              ...current,
                              [workOrder.work_order_id]: event.target.value,
                            }))}
                          />
                          <Input.TextArea
                            rows={2}
                            maxLength={1000}
                            showCount
                            placeholder="填写派工依据，例如工程师角色、设备范围和现场安排"
                            disabled={Boolean(assignmentProposals[workOrder.work_order_id])}
                            value={assignmentReasons[workOrder.work_order_id] ?? ""}
                            onChange={(event) => setAssignmentReasons((current) => ({
                              ...current,
                              [workOrder.work_order_id]: event.target.value,
                            }))}
                          />
                          <Space wrap>
                            <Button
                              loading={busyId === workOrder.work_order_id}
                              disabled={
                                Boolean(assignmentProposals[workOrder.work_order_id])
                                || !assignments[workOrder.work_order_id]?.trim()
                              }
                              onClick={() => void assign(workOrder)}
                            >
                              人工直接派工
                            </Button>
                            <Button
                              loading={busyId === workOrder.work_order_id}
                              disabled={
                                Boolean(assignmentProposals[workOrder.work_order_id])
                                || !assignments[workOrder.work_order_id]?.trim()
                                || (assignmentReasons[workOrder.work_order_id]?.trim().length ?? 0) < 3
                              }
                              onClick={() => void proposeAssignment(workOrder)}
                            >
                              {assignmentProposals[workOrder.work_order_id]
                                ? "派工提案已提交"
                                : "创建本地目录派工提案"}
                            </Button>
                          </Space>
                          {assignmentProposals[workOrder.work_order_id]?.target
                            === "LOCAL_DIRECTORY" ? (
                              <AssignmentProposalSubmittedAlert
                                proposal={assignmentProposals[workOrder.work_order_id]}
                              />
                            ) : null}
                        </div>
                      </Card>
                    </div>
                  ) : null}
                  <Link href={`/work-orders/${workOrder.work_order_id}`}>进入工单履约详情</Link>
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

function EnterpriseAssignmentPanel({
  state,
  busy,
  submittedProposal,
  onSubmit,
}: {
  state?: {
    options: WorkOrderAssignmentOption[];
    fsmUnavailableReason: string | null;
  };
  busy: boolean;
  submittedProposal?: AssignmentProposalState;
  onSubmit: (candidateProfileId: string, reason: string) => void;
}) {
  const candidates = state?.options.filter(
    (option) => option.assignment_target === "ENTERPRISE_FSM",
  ) ?? [];
  return (
    <Card size="small" title="企业 FSM 合规派工">
      <div className="page-stack">
        <Alert
          type="success"
          showIcon
          message="候选资格由服务端权威系统提供"
          description="浏览器只能选择企业 FSM 返回且平台本地资格复核通过的候选；执行前仍会重新验证 Profile、时效和人员范围。"
        />
        {!state ? <LoadingState label="正在读取企业 FSM 候选" /> : null}
        {state && candidates.length === 0 ? (
          <Alert
            type="warning"
            showIcon
            message="当前没有可用的企业 FSM 候选"
            description={state.fsmUnavailableReason ?? "候选规则未返回符合条件的工程师"}
          />
        ) : null}
        {candidates.map((candidate) => (
          <Card
            key={candidate.candidate_profile_id ?? candidate.display_name}
            size="small"
            title={candidate.display_name}
            extra={<Tag color="green">{candidate.eligibility_code}</Tag>}
          >
            <FactGrid facts={[
              ["Provider", candidate.provider],
              ["候选 Profile", candidate.candidate_profile_id],
              ["Profile 版本", candidate.profile_version],
              ["平台工程师", candidate.assignee_subject_id],
              ["匹配技能", candidate.matched_skill_codes.join("、")],
              ["预计路程", `${candidate.travel_minutes ?? "—"} 分钟`],
              ["剩余工时", `${candidate.remaining_work_minutes ?? "—"} 分钟`],
              ["服务窗口", `${formatTimestamp(candidate.service_window_start ?? "—")} 至 ${formatTimestamp(candidate.service_window_end ?? "—")}`],
              ["有效期", formatTimestamp(candidate.expires_at ?? "—")],
            ]} />
          </Card>
        ))}
        {candidates.length > 0 ? (
          <EnterpriseAssignmentForm
            candidates={candidates}
            busy={busy}
            submittedProposal={submittedProposal}
            onSubmit={onSubmit}
          />
        ) : null}
      </div>
    </Card>
  );
}

function EnterpriseAssignmentForm({
  candidates,
  busy,
  submittedProposal,
  onSubmit,
}: {
  candidates: WorkOrderAssignmentOption[];
  busy: boolean;
  submittedProposal?: AssignmentProposalState;
  onSubmit: (candidateProfileId: string, reason: string) => void;
}) {
  const [selectedProfileId, setSelectedProfileId] = useState<string>();
  const reason = useRef("");
  const selected = candidates.find(
    (candidate) => candidate.candidate_profile_id === selectedProfileId,
  );
  return (
    <form
      className="page-stack"
      onSubmit={(event) => { event.preventDefault(); if (selectedProfileId && reason.current.trim().length >= 3) onSubmit(selectedProfileId, reason.current); }}
    >
      <Select
        aria-label="企业 FSM 候选"
        placeholder="显式选择一个服务端候选"
        value={selectedProfileId}
        disabled={Boolean(submittedProposal)}
        options={candidates.map((candidate) => ({
          value: candidate.candidate_profile_id ?? "",
          label: `${candidate.display_name} · 预计路程 ${candidate.travel_minutes ?? "—"} 分钟`,
        }))}
        onChange={setSelectedProfileId}
      />
      <textarea className="ant-input" rows={2}
        minLength={3} maxLength={1000} required
        placeholder="填写企业 FSM 派工理由"
        disabled={Boolean(submittedProposal)}
        onChange={(event) => { reason.current = event.target.value; }}
      />
      {selected ? (
        <Typography.Text type="secondary">
          将绑定 {selected.provider} / {selected.candidate_profile_id} /
          {" "}{selected.profile_version}，不会提交浏览器声明的资格事实。
        </Typography.Text>
      ) : null}
      {submittedProposal?.target === "ENTERPRISE_FSM" ? (
        <AssignmentProposalSubmittedAlert proposal={submittedProposal} />
      ) : null}
      <button
        className="ant-btn ant-btn-primary" type="submit"
        disabled={busy || !selectedProfileId || Boolean(submittedProposal)}
      >
        {submittedProposal
          ? "派工提案已提交"
          : busy
            ? "正在创建企业 FSM 派工提案"
            : "创建企业 FSM 派工提案"}
      </button>
    </form>
  );
}

function AssignmentProposalSubmittedAlert({
  proposal,
}: {
  proposal: AssignmentProposalState;
}) {
  return (
    <Alert
      type="success"
      showIcon
      message="派工提案已进入审批箱"
      description={(
        <Space direction="vertical" size="small">
          <Link href="/approvals">审批请求 {proposal.approvalId}</Link>
          <Typography.Text type="secondary">
            当前工单无需重复创建派工提案，请前往审批箱完成独立审批与执行。
          </Typography.Text>
        </Space>
      )}
    />
  );
}

function LiveFact({
  title,
  fact,
  failure,
  facts,
}: {
  title: string;
  fact: DispatchFact | null;
  failure?: string;
  facts: Array<[string, unknown]>;
}) {
  if (!fact) {
    return <Alert type="warning" showIcon message={`${title}暂不可用`} description={failure} />;
  }
  return (
    <Card size="small" title={title} extra={<Tag color="geekblue">{fact.source}</Tag>}>
      <FactGrid facts={facts} />
      <Typography.Text type="secondary">
        as_of {formatTimestamp(fact.as_of)} · {fact.authority.owner} · {fact.source_record_id}
      </Typography.Text>
    </Card>
  );
}

function failureFor(workOrder: DispatchWorkOrder, toolId: string): string | undefined {
  return workOrder.fact_failures.find((item) => item.tool_id === toolId)?.reason;
}

function factValue(fact: DispatchFact, field: string): string {
  const value = fact.values[field];
  return value === null || value === undefined ? "—" : String(value);
}

function formatTimestamp(value: string): string {
  if (value === "—") return value;
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime())
    ? value
    : timestamp.toLocaleString("zh-CN", { hour12: false });
}
