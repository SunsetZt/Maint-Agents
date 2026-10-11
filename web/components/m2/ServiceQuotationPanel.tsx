"use client";

import { Alert, Button, Card, List, Space, Tag, Typography } from "antd";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  ApiClientError,
  type CustomerServiceQuotationView,
  type DiagnosisRun,
  type ServiceQuotation,
  type ServiceQuotationOption,
  decideCustomerServiceQuotation,
  getCustomerServiceQuotation,
  listServiceQuotationOptions,
  listServiceQuotations,
  proposeServiceQuotation,
} from "@/lib/api/client";

const TERMINAL_QUOTATION_DIAGNOSIS_STATUSES = new Set([
  "COMPLETED",
  "NEEDS_INFORMATION",
]);

type StaffQuotationState = {
  diagnosisRunId: string;
  diagnosisVersion: number;
  entitlementDecision: string;
  options: ServiceQuotationOption[];
  catalogUnavailableReason: string | null;
  quotations: ServiceQuotation[];
};

export function ServiceQuotationPanel({ incidentId, mode, diagnosisStatus }: {
  incidentId: string;
  mode: "staff" | "customer";
  diagnosisStatus?: DiagnosisRun["status"];
}) {
  return mode === "staff"
    ? (
        <StaffServiceQuotationPanel
          incidentId={incidentId}
          diagnosisStatus={diagnosisStatus}
        />
      )
    : <CustomerServiceQuotationPanel incidentId={incidentId} />;
}

function StaffServiceQuotationPanel({ incidentId, diagnosisStatus }: {
  incidentId: string;
  diagnosisStatus?: DiagnosisRun["status"];
}) {
  const [state, setState] = useState<StaffQuotationState>();
  const [error, setError] = useState<unknown>();
  const [eligibilityBlocked, setEligibilityBlocked] = useState(false);
  const [submittingCandidate, setSubmittingCandidate] = useState<string>();
  const [proposalId, setProposalId] = useState<string>();
  const quotationEligible = diagnosisStatus !== undefined
    && TERMINAL_QUOTATION_DIAGNOSIS_STATUSES.has(diagnosisStatus);

  const load = useCallback(async () => {
    setError(undefined);
    setEligibilityBlocked(false);
    if (!quotationEligible) {
      setState(undefined);
      return;
    }
    try {
      const [optionsResult, historyResult] = await Promise.all([
        listServiceQuotationOptions(incidentId),
        listServiceQuotations(incidentId),
      ]);
      setState({
        diagnosisRunId: optionsResult.diagnosisRunId,
        diagnosisVersion: optionsResult.diagnosisVersion,
        entitlementDecision: optionsResult.entitlementDecision,
        options: optionsResult.options,
        catalogUnavailableReason: optionsResult.catalogUnavailableReason,
        quotations: historyResult.quotations,
      });
    } catch (caught) {
      if (caught instanceof ApiClientError && caught.status === 409) {
        setState(undefined);
        setEligibilityBlocked(true);
        return;
      }
      setError(caught);
    }
  }, [incidentId, quotationEligible]);

  useEffect(() => { void load(); }, [load]);

  async function propose(candidateId: string) {
    setSubmittingCandidate(candidateId);
    setError(undefined);
    setProposalId(undefined);
    try {
      const result = await proposeServiceQuotation(incidentId, candidateId);
      setProposalId(result.proposal.proposal_id);
    } catch (caught) {
      setError(caught);
    } finally {
      setSubmittingCandidate(undefined);
    }
  }

  return (
    <Card title="受控服务报价">
      <div className="page-stack">
        <Alert
          type="info"
          showIcon
          message="报价金额由企业 CPQ 服务生成"
          description="页面只展示服务端候选并提交候选标识；金额、税费、权益结论和有效期均不可在此编辑。"
        />
        {!quotationEligible || eligibilityBlocked ? (
          <Alert
            type="warning"
            showIcon
            message="完成诊断后可生成服务报价"
            description="当前 Incident 尚无终态诊断。诊断达到 COMPLETED 或 NEEDS_INFORMATION 后，系统才会校验售后权益并读取企业 CPQ 候选。"
          />
        ) : null}
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {quotationEligible && !eligibilityBlocked && !state && !error ? (
          <LoadingState label="正在读取服务报价候选" />
        ) : null}
        {proposalId ? (
          <Alert
            type="success"
            showIcon
            message={`报价提案 ${proposalId} 已提交`}
            description={<Link href="/approvals">前往审批中心完成 T2 职责分离审批</Link>}
          />
        ) : null}
        {state ? (
          <>
            <Space wrap>
              <Tag color="blue">诊断 {state.diagnosisRunId} / v{state.diagnosisVersion}</Tag>
              <Tag color={state.entitlementDecision === "COVERED" ? "green" : "gold"}>
                {state.entitlementDecision}
              </Tag>
            </Space>
            {state.options.length === 0 ? (
              <Alert
                type="warning"
                showIcon
                message="当前无法创建报价提案"
                description={unavailableReason(state.catalogUnavailableReason)}
              />
            ) : (
              <List
                dataSource={state.options}
                renderItem={(option) => (
                  <List.Item
                    actions={[
                      <Button
                        key="propose"
                        type="primary"
                        loading={submittingCandidate === option.candidate_id}
                        disabled={Boolean(submittingCandidate)}
                        onClick={() => void propose(option.candidate_id)}
                      >
                        提交 T2 报价审批
                      </Button>,
                    ]}
                  >
                    <List.Item.Meta
                      title={<Space wrap><Typography.Text strong>{option.display_name}</Typography.Text><Tag>{option.provider}</Tag></Space>}
                      description={(
                        <div className="page-stack">
                          <Typography.Title level={4}>{formatMoney(option.total, option.currency)}</Typography.Title>
                          <Typography.Text type="secondary">
                            小计 {formatMoney(option.subtotal, option.currency)} · 折扣 {formatMoney(option.discount, option.currency)} · 税费 {formatMoney(option.tax, option.currency)}
                          </Typography.Text>
                          <List
                            size="small"
                            dataSource={option.line_items}
                            renderItem={(line) => (
                              <List.Item>
                                {line.description} × {line.quantity}：{formatMoney(line.line_total, option.currency)}
                              </List.Item>
                            )}
                          />
                          <Typography.Text type="secondary">
                            来源版本 {option.source_version} · 有效至 {formatTimestamp(option.expires_at)}
                          </Typography.Text>
                        </div>
                      )}
                    />
                  </List.Item>
                )}
              />
            )}
            <QuotationHistory quotations={state.quotations} />
          </>
        ) : null}
      </div>
    </Card>
  );
}

function CustomerServiceQuotationPanel({ incidentId }: { incidentId: string }) {
  const [view, setView] = useState<CustomerServiceQuotationView>();
  const [error, setError] = useState<unknown>();
  const [busy, setBusy] = useState(false);
  const [confirmation, setConfirmation] = useState<string>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      setView((await getCustomerServiceQuotation(incidentId)).quotation);
    } catch (caught) {
      setError(caught);
    }
  }, [incidentId]);

  useEffect(() => { void load(); }, [load]);

  async function decide(decision: "ACCEPTED" | "REJECTED") {
    if (!view?.current) return;
    setBusy(true);
    setError(undefined);
    setConfirmation(undefined);
    try {
      const result = await decideCustomerServiceQuotation(
        incidentId,
        decision,
        view.current.state_version,
        `portal-quotation-${crypto.randomUUID()}`,
      );
      setConfirmation(
        `报价决定 ${result.decision.decision_id} 已记录，等待售后按现有流程处理。`,
      );
      setView({
        ...view,
        current: { ...view.current, status: result.decision.status, legal_actions: [] },
      });
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  const current = view?.current;
  return (
    <Card title="服务报价">
      <div className="page-stack">
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!view && !error ? <LoadingState label="正在读取服务报价" /> : null}
        {confirmation ? (
          <Alert
            type="success"
            showIcon
            message={confirmation}
            description="本操作只记录报价决定，不会自动创建工单、开票或付款。"
          />
        ) : null}
        {view && !current ? <EmptyState description="当前没有待确认的有效服务报价" /> : null}
        {current ? (
          <>
            <Space wrap>
              <Typography.Title level={4} style={{ margin: 0 }}>当前服务报价</Typography.Title>
              <Tag color={current.status === "PUBLISHED" ? "blue" : "green"}>{current.status}</Tag>
              <Tag>{current.entitlement_decision}</Tag>
            </Space>
            <Typography.Title level={3}>{formatMoney(current.total, current.currency)}</Typography.Title>
            <Typography.Text type="secondary">
              小计 {formatMoney(current.subtotal, current.currency)} · 折扣 {formatMoney(current.discount, current.currency)} · 税费 {formatMoney(current.tax, current.currency)}
            </Typography.Text>
            <List
              size="small"
              dataSource={current.line_items}
              renderItem={(line) => (
                <List.Item>
                  {line.description} × {line.quantity}：{formatMoney(line.line_total, current.currency)}
                </List.Item>
              )}
            />
            <Typography.Text type="secondary">
              报价 v{current.version} · 有效至 {formatTimestamp(current.expires_at)}
            </Typography.Text>
            <Space wrap>
              {current.legal_actions.includes("ACCEPT") ? (
                <Button type="primary" loading={busy} onClick={() => void decide("ACCEPTED")}>接受报价</Button>
              ) : null}
              {current.legal_actions.includes("REJECT") ? (
                <Button danger loading={busy} onClick={() => void decide("REJECTED")}>拒绝报价</Button>
              ) : null}
            </Space>
          </>
        ) : null}
        {view && view.history.length > 0 ? (
          <List
            header="历史报价"
            size="small"
            dataSource={view.history}
            renderItem={(quote) => (
              <List.Item>
                v{quote.version} · {quote.status} · {formatMoney(quote.total, quote.currency)} · {formatTimestamp(quote.published_at)}
              </List.Item>
            )}
          />
        ) : null}
      </div>
    </Card>
  );
}

function QuotationHistory({ quotations }: { quotations: ServiceQuotation[] }) {
  if (quotations.length === 0) return null;
  return (
    <List
      header="已发布报价版本"
      size="small"
      dataSource={quotations}
      renderItem={(quote) => (
        <List.Item>
          <Space wrap>
            <Typography.Text>{quote.quotation_id} / v{quote.version}</Typography.Text>
            <Tag>{quote.status}</Tag>
            <Typography.Text>{formatMoney(quote.total, quote.currency)}</Typography.Text>
            <Typography.Text type="secondary">发布于 {formatTimestamp(quote.published_at)}</Typography.Text>
          </Space>
        </List.Item>
      )}
    />
  );
}

function unavailableReason(reason: string | null): string {
  const messages: Record<string, string> = {
    service_quotation_disabled: "服务报价目录尚未启用。",
    service_entitlement_not_reliable: "当前服务权益结论需要人工复核。",
    service_quotation_catalog_unavailable: "企业 CPQ 暂时不可用，请稍后重试。",
  };
  return reason ? (messages[reason] ?? `原因：${reason}`) : "没有符合当前诊断与权益约束的服务报价候选。";
}

function formatMoney(value: string, currency: string): string {
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency,
    minimumFractionDigits: 2,
  }).format(Number(value));
}

function formatTimestamp(value: string): string {
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}
