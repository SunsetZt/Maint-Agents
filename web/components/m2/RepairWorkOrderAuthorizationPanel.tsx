"use client";

import { Alert, Button, Card, List, Space, Tag, Typography } from "antd";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  type RepairWorkOrderOption,
  listRepairWorkOrderOptions,
  proposeRepairWorkOrder,
} from "@/lib/api/client";

type State = {
  incidentVersion: number;
  diagnosisRunId: string | null;
  diagnosisVersion: number | null;
  entitlementDecision: string | null;
  options: RepairWorkOrderOption[];
  unavailableReason: string | null;
};

type RepairWorkOrderAuthorizationPanelProps = {
  incidentId: string;
  incidentVersion: number;
  diagnosisRunId?: string;
  diagnosisStatus?: string;
  diagnosisVersion?: number;
};

export function RepairWorkOrderAuthorizationPanel({
  incidentId,
  incidentVersion,
  diagnosisRunId,
  diagnosisStatus,
  diagnosisVersion,
}: RepairWorkOrderAuthorizationPanelProps) {
  const [state, setState] = useState<State>();
  const [error, setError] = useState<unknown>();
  const [busy, setBusy] = useState(false);
  const [proposalId, setProposalId] = useState<string>();
  const proposalAttempt = useRef<{ fingerprint: string; idempotencyKey: string } | undefined>(undefined);
  const loadSequence = useRef(0);

  const load = useCallback(async () => {
    const sequence = ++loadSequence.current;
    setError(undefined);
    try {
      const result = await listRepairWorkOrderOptions(incidentId);
      if (sequence !== loadSequence.current) return;
      setState(result);
    } catch (caught) {
      if (sequence !== loadSequence.current) return;
      setError(caught);
    }
  }, [incidentId]);

  useEffect(() => {
    void load();
    return () => { loadSequence.current += 1; };
  }, [diagnosisRunId, diagnosisStatus, diagnosisVersion, incidentVersion, load]);

  async function propose(option: RepairWorkOrderOption) {
    if (!state) return;
    const fingerprint = JSON.stringify([
      incidentId,
      state.incidentVersion,
      option.authorization_type,
      option.quotation_id,
      option.quotation_version,
      option.quotation_state_version,
    ]);
    const idempotencyKey = proposalAttempt.current?.fingerprint === fingerprint
      ? proposalAttempt.current.idempotencyKey
      : `repair-work-order-${crypto.randomUUID()}`;
    proposalAttempt.current = { fingerprint, idempotencyKey };
    setBusy(true);
    setError(undefined);
    try {
      const result = await proposeRepairWorkOrder(
        incidentId,
        state.incidentVersion,
        option.authorization_type,
        idempotencyKey,
      );
      proposalAttempt.current = undefined;
      setProposalId(result.proposal.proposal_id);
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="受控维修工单授权">
      <div className="page-stack">
        <Alert type="info" showIcon message="授权、诊断、报价与无备件模式均由服务端确定" description="页面只提交当前返回选项；提案成功仅表示进入 T2 独立审批，审批执行前没有 WorkOrder 副作用。" />
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!state && !error ? <LoadingState label="正在读取维修授权选项" /> : null}
        {proposalId ? <Alert type="success" showIcon message={`${proposalId} 已提交，当前待审批`} description={<Link href="/approvals">前往审批箱完成独立复核与执行</Link>} /> : null}
        {state ? (
          <>
            <Space wrap>
              <Tag color="blue">诊断 {state.diagnosisRunId ?? "不可用"}{state.diagnosisVersion ? ` / v${state.diagnosisVersion}` : ""}</Tag>
              <Tag color={state.entitlementDecision === "COVERED" ? "green" : "gold"}>{state.entitlementDecision ?? "FACTS_UNAVAILABLE"}</Tag>
            </Space>
            {state.options.length === 0 ? (
              <Alert type="warning" showIcon message="当前无法创建维修工单提案" description={reasonLabel(state.unavailableReason)} />
            ) : (
              <List dataSource={state.options} renderItem={(option) => (
                <List.Item actions={[<Button key="propose" type="primary" loading={busy} onClick={() => void propose(option)}>提交 T2 工单审批</Button>]}>
                  <List.Item.Meta
                    title={option.authorization_type === "COVERED_SERVICE" ? "保内现场服务 · 无初始备件" : "已接受报价现场服务 · 无初始备件"}
                    description={option.quotation_id ? `报价 ${option.quotation_id} / v${option.quotation_version} · ${option.currency} ${option.total}` : "使用当前服务权益与最新终态诊断"}
                  />
                </List.Item>
              )} />
            )}
          </>
        ) : null}
      </div>
    </Card>
  );
}

function reasonLabel(reason: string | null): string {
  const labels: Record<string, string> = {
    incident_not_diagnosed: "Incident 尚未完成诊断；诊断完成后本区域会自动刷新。",
    terminal_diagnosis_required: "最新诊断尚未达到可授权的完成状态；完成后本区域会自动刷新。",
    service_entitlement_facts_unavailable: "服务权益事实暂不可用。",
    accepted_service_quotation_required: "保外维修需要客户已接受的当前报价。",
    work_order_already_exists: "该 Incident 已有关联工单。",
    repair_work_order_already_exists: "该 Incident 已有关联工单。",
  };
  return reason ? (labels[reason] ?? `原因：${reason}`) : "当前没有合法的维修授权选项。";
}
