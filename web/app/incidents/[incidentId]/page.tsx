"use client";

import {
  Alert,
  Button,
  Card,
  Checkbox,
  Form,
  Input,
  InputNumber,
  List,
  Modal,
  Select,
  Space,
  Tag,
  Typography,
} from "antd";
import { useParams } from "next/navigation";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import { AgentEventTimeline } from "@/components/m2/AgentEventTimeline";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import { CitationDrawer } from "@/components/m2/CitationDrawer";
import { CustomerNotificationProposalPanel } from "@/components/m2/CustomerNotificationProposalPanel";
import { DiagnosisFeedbackPanel } from "@/components/m2/DiagnosisFeedbackPanel";
import { EquipmentControlHandoffPanel } from "@/components/m2/EquipmentControlHandoffPanel";
import { PurchaseRequestPanel } from "@/components/m2/PurchaseRequestPanel";
import { RefundRequestPanel } from "@/components/m2/RefundRequestPanel";
import { RepairWorkOrderAuthorizationPanel } from "@/components/m2/RepairWorkOrderAuthorizationPanel";
import { ServiceQuotationPanel } from "@/components/m2/ServiceQuotationPanel";
import { RealtimeAudioPanel } from "@/components/m3/RealtimeAudioPanel";
import { ModelRuntimeStatus } from "@/components/model-runtime/ModelRuntimeStatus";
import {
  ApiClientError,
  type AgUiTransportEvent,
  type CustomerCaseUpdate,
  type DiagnosisRun,
  type Incident,
  type IncidentOperations,
  type IncidentResolution,
  type SpeechSynthesis,
  cancelAgentRun,
  getDiagnosis,
  getIncident,
  listIncidentCustomerUpdates,
  runDiagnosisAgent,
  synthesizeDiagnosisSpeech,
  submitExpertDiagnosisRevision,
  takeOverDiagnosis,
  triageIncident,
} from "@/lib/api/client";

const TTS_SAFETY_WARNING = "安全提示：在检查、拆卸或维修设备前，必须停机、断电、泄压并执行上锁挂牌；确认设备无残余能量，佩戴适用的个人防护装备。未经现场负责人确认，不得执行维修动作。";
const TERMINAL_DIAGNOSIS_STATUSES = new Set([
  "COMPLETED",
  "NEEDS_INFORMATION",
  "FAILED",
  "CANCELLED",
  "WAITING_EXPERT",
  "ESCALATED",
]);

type TakeoverValues = { reason: string };
type CancelAgentValues = { reason: string };
type ResumeAgentValues = { clarification: string };
type RemoteResolutionValues = { summary: string; evidence_ids: string };
type IncidentCloseValues = {
  confirmation_type: "EXPERT" | "CUSTOMER";
  reason: string;
  customer_update_id?: string;
};
type ExpertRevisionValues = {
  outcome: "COMPLETED" | "ESCALATED";
  conclusion?: string;
  citation_ids: string[];
  contradictions?: string;
  missing_information?: string;
  next_checks?: string;
  confidence: number;
  reason: string;
};

export default function IncidentPage() {
  const { incidentId } = useParams<{ incidentId: string }>();
  const [incident, setIncident] = useState<Incident>();
  const [diagnosis, setDiagnosis] = useState<DiagnosisRun>();
  const [customerUpdates, setCustomerUpdates] = useState<CustomerCaseUpdate[]>([]);
  const [operations, setOperations] = useState<IncidentOperations>();
  const [resolution, setResolution] = useState<IncidentResolution | null>();
  const [error, setError] = useState<unknown>();
  const [busy, setBusy] = useState(false);
  const [diagnosisActionMessage, setDiagnosisActionMessage] = useState<string>();
  const [triageBusy, setTriageBusy] = useState(false);
  const [speechBusy, setSpeechBusy] = useState(false);
  const [safetyAcknowledged, setSafetyAcknowledged] = useState(false);
  const [autoSpeakAfterReanalysis, setAutoSpeakAfterReanalysis] = useState(false);
  const [pendingAutoSpeechRunId, setPendingAutoSpeechRunId] = useState<string>();
  const [speech, setSpeech] = useState<SpeechSynthesis>();
  const [takeoverOpen, setTakeoverOpen] = useState(false);
  const [revisionOpen, setRevisionOpen] = useState(false);
  const [expertBusy, setExpertBusy] = useState(false);
  const [agentControlBusy, setAgentControlBusy] = useState(false);
  const [agentInstruction, setAgentInstruction] = useState("");
  const [openInterruptId, setOpenInterruptId] = useState<string>();
  const [liveAgentRunId, setLiveAgentRunId] = useState<string>();
  const [liveAgentEvents, setLiveAgentEvents] = useState<AgUiTransportEvent[]>([]);
  const [cancelAgentOpen, setCancelAgentOpen] = useState(false);
  const [resumeAgentOpen, setResumeAgentOpen] = useState(false);
  const [remoteResolutionOpen, setRemoteResolutionOpen] = useState(false);
  const [incidentCloseOpen, setIncidentCloseOpen] = useState(false);
  const [resolutionBusy, setResolutionBusy] = useState(false);
  const [takeoverForm] = Form.useForm<TakeoverValues>();
  const [revisionForm] = Form.useForm<ExpertRevisionValues>();
  const [cancelAgentForm] = Form.useForm<CancelAgentValues>();
  const [resumeAgentForm] = Form.useForm<ResumeAgentValues>();
  const [remoteResolutionForm] = Form.useForm<RemoteResolutionValues>();
  const [incidentCloseForm] = Form.useForm<IncidentCloseValues>();
  const autoSpeechStarted = useRef(new Set<string>());
  const agentTransportController = useRef<AbortController | undefined>(undefined);
  const diagnosisResultRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const current = (await getIncident(incidentId)).incident;
      setIncident(current);
      if (["DIAGNOSED", "WORK_ORDER_CREATED", "RESOLVED", "CLOSED"].includes(current.status)) {
        const resolutionApi = await import("@/lib/api/client");
        const [authority, persistedResolution] = await Promise.all([
          resolutionApi.getIncidentOperations(incidentId),
          resolutionApi.getIncidentResolution(incidentId),
        ]);
        setOperations(authority.incident);
        setResolution(persistedResolution.resolution);
      } else {
        setOperations(undefined);
        setResolution(undefined);
      }
      try {
        setCustomerUpdates((await listIncidentCustomerUpdates(incidentId)).updates);
      } catch (caught) {
        if (!(caught instanceof ApiClientError) || caught.status !== 403) throw caught;
        setCustomerUpdates([]);
      }
      const latest = current.diagnosis_run_ids.at(-1);
      if (latest) setDiagnosis((await getDiagnosis(latest)).diagnosis);
    } catch (caught) { setError(caught); }
  }, [incidentId]);

  useEffect(() => { void load(); }, [load]);

  useEffect(() => () => agentTransportController.current?.abort(), []);

  useEffect(() => {
    if (!diagnosis || diagnosis.diagnosis_run_id !== pendingAutoSpeechRunId) return;
    if (["NEEDS_INFORMATION", "FAILED", "CANCELLED"].includes(diagnosis.status)) {
      setPendingAutoSpeechRunId(undefined);
      return;
    }
    if (
      diagnosis.status !== "COMPLETED"
      || !diagnosis.report
      || !safetyAcknowledged
      || autoSpeechStarted.current.has(diagnosis.diagnosis_run_id)
    ) return;
    autoSpeechStarted.current.add(diagnosis.diagnosis_run_id);
    setPendingAutoSpeechRunId(undefined);
    setSpeechBusy(true);
    setError(undefined);
    void synthesizeDiagnosisSpeech(
      diagnosis.diagnosis_run_id,
      true,
      `auto-reanalysis-${diagnosis.diagnosis_run_id}`,
    ).then(({ speech: generated }) => {
      setSpeech(generated);
    }).catch((caught: unknown) => {
      setError(caught);
    }).finally(() => {
      setSpeechBusy(false);
    });
  }, [diagnosis, pendingAutoSpeechRunId, safetyAcknowledged]);

  async function diagnose() {
    if (!incident || incident.status !== "TRIAGED") return;
    let transportController: AbortController | undefined;
    setBusy(true);
    setError(undefined);
    try {
      setPendingAutoSpeechRunId(undefined);
      setAutoSpeakAfterReanalysis(false);
      setSafetyAcknowledged(false);
      setSpeech(undefined);
      const runId = crypto.randomUUID();
      const instruction = agentInstruction.trim();
      const controller = beginAgentTransport(runId);
      transportController = controller;
      await runDiagnosisAgent(
        {
          threadId: incidentId,
          runId,
          state: { incidentVersion: incident.version },
          messages: instruction
            ? [{ id: crypto.randomUUID(), role: "user", content: instruction }]
            : [],
          tools: [],
          context: [],
          forwardedProps: {},
        },
        globalThis.fetch,
        controller.signal,
        appendLiveAgentEvent,
      );
      const current = (await getIncident(incidentId)).incident;
      setIncident(current);
      const latest = current.diagnosis_run_ids.at(-1);
      if (latest) {
        const completed = (await getDiagnosis(latest)).diagnosis;
        setDiagnosis(completed);
        setDiagnosisActionMessage(
          `诊断运行 ${completed.diagnosis_run_id} 已结束，最新结果已加载。`,
        );
        focusDiagnosisResult();
      }
      setAgentInstruction("");
    } catch (caught) {
      setError(caught);
    } finally {
      finishAgentTransport(transportController);
      setBusy(false);
    }
  }

  function focusDiagnosisResult(message?: string): void {
    if (message) setDiagnosisActionMessage(message);
    globalThis.setTimeout(() => {
      diagnosisResultRef.current?.scrollIntoView?.({
        behavior: "smooth",
        block: "start",
      });
    }, 0);
  }

  async function triage() {
    if (!incident || incident.status !== "SUBMITTED") return;
    setTriageBusy(true);
    setError(undefined);
    try {
      const result = await triageIncident(incident.incident_id, incident.version);
      setIncident(result.incident);
    } catch (caught) {
      setError(caught);
    } finally {
      setTriageBusy(false);
    }
  }

  async function synthesizeSpeech() {
    if (!diagnosis) return;
    setSpeechBusy(true);
    setError(undefined);
    try {
      setSpeech((await synthesizeDiagnosisSpeech(
        diagnosis.diagnosis_run_id,
        safetyAcknowledged,
        crypto.randomUUID(),
      )).speech);
    } catch (caught) { setError(caught); } finally { setSpeechBusy(false); }
  }

  async function resolveRemote(values: RemoteResolutionValues) {
    if (!operations) return;
    setResolutionBusy(true);
    setError(undefined);
    try {
      const resolutionApi = await import("@/lib/api/client");
      const result = await resolutionApi.resolveIncidentRemote(
        incidentId,
        operations.version,
        {
          summary: values.summary.trim(),
          evidence_ids: splitLines(values.evidence_ids),
        },
      );
      setOperations(result.incident);
      setResolution((await resolutionApi.getIncidentResolution(incidentId)).resolution);
      setRemoteResolutionOpen(false);
      remoteResolutionForm.resetFields();
    } catch (caught) {
      setError(caught);
    } finally {
      setResolutionBusy(false);
    }
  }

  async function confirmIncidentClose(values: IncidentCloseValues) {
    if (!operations) return;
    setResolutionBusy(true);
    setError(undefined);
    try {
      const resolutionApi = await import("@/lib/api/client");
      const result = await resolutionApi.closeIncident(
        incidentId,
        operations.version,
        {
          confirmation_type: values.confirmation_type,
          reason: values.reason.trim(),
          customer_update_id: values.customer_update_id?.trim() || null,
        },
      );
      setOperations(result.incident);
      setIncidentCloseOpen(false);
      incidentCloseForm.resetFields();
    } catch (caught) {
      setError(caught);
    } finally {
      setResolutionBusy(false);
    }
  }

  async function takeOver(values: TakeoverValues) {
    if (!diagnosis) return;
    setExpertBusy(true);
    setError(undefined);
    try {
      const result = await takeOverDiagnosis(
        diagnosis.diagnosis_run_id,
        diagnosis.version,
        values.reason,
        crypto.randomUUID(),
      );
      setDiagnosis(result.diagnosis);
      setTakeoverOpen(false);
      takeoverForm.resetFields();
    } catch (caught) {
      setError(caught);
    } finally {
      setExpertBusy(false);
    }
  }

  async function submitRevision(values: ExpertRevisionValues) {
    if (!diagnosis || !openIntervention) return;
    setExpertBusy(true);
    setError(undefined);
    try {
      const result = await submitExpertDiagnosisRevision(
        diagnosis.diagnosis_run_id,
        openIntervention.intervention_id,
        openIntervention.version,
        {
          outcome: values.outcome,
          conclusion: values.conclusion?.trim() || null,
          citation_ids: values.citation_ids ?? [],
          contradictions: splitLines(values.contradictions),
          missing_information: splitLines(values.missing_information),
          next_checks: splitLines(values.next_checks),
          confidence: values.confidence,
          reason: values.reason,
        },
      );
      setDiagnosis(result.diagnosis);
      setRevisionOpen(false);
      revisionForm.resetFields();
    } catch (caught) {
      setError(caught);
    } finally {
      setExpertBusy(false);
    }
  }

  async function cancelCurrentAgent(values: CancelAgentValues) {
    if (!diagnosis) return;
    setAgentControlBusy(true);
    setError(undefined);
    try {
      await cancelAgentRun(
        diagnosis.agent_run_id,
        diagnosis.agent_run_version,
        values.reason,
        crypto.randomUUID(),
      );
      setDiagnosis((await getDiagnosis(diagnosis.diagnosis_run_id)).diagnosis);
      setCancelAgentOpen(false);
      cancelAgentForm.resetFields();
    } catch (caught) {
      setError(caught);
    } finally {
      setAgentControlBusy(false);
    }
  }

  async function resumeCurrentAgent(values: ResumeAgentValues) {
    if (!diagnosis || !openInterruptId) return;
    let transportController: AbortController | undefined;
    setAgentControlBusy(true);
    setError(undefined);
    try {
      const runId = crypto.randomUUID();
      const messageId = crypto.randomUUID();
      const controller = beginAgentTransport(runId);
      transportController = controller;
      await runDiagnosisAgent(
        {
          threadId: incidentId,
          runId,
          parentRunId: diagnosis.agent_run_id,
          state: { agentRunVersion: diagnosis.agent_run_version },
          messages: [{ id: messageId, role: "user", content: values.clarification }],
          tools: [],
          context: [],
          forwardedProps: {},
          resume: [{
            interruptId: openInterruptId,
            status: "resolved",
            payload: { clarification: values.clarification },
          }],
        },
        globalThis.fetch,
        controller.signal,
        appendLiveAgentEvent,
      );
      const current = (await getIncident(incidentId)).incident;
      setIncident(current);
      const latest = current.diagnosis_run_ids.at(-1);
      if (latest) setDiagnosis((await getDiagnosis(latest)).diagnosis);
      setOpenInterruptId(undefined);
      setResumeAgentOpen(false);
      resumeAgentForm.resetFields();
    } catch (caught) {
      setError(caught);
    } finally {
      finishAgentTransport(transportController);
      setAgentControlBusy(false);
    }
  }

  function beginAgentTransport(runId: string): AbortController {
    agentTransportController.current?.abort();
    const controller = new AbortController();
    agentTransportController.current = controller;
    setLiveAgentRunId(runId);
    setLiveAgentEvents([]);
    return controller;
  }

  function appendLiveAgentEvent(event: AgUiTransportEvent): void {
    setLiveAgentEvents((current) => {
      if (event.id && current.some((item) => item.id === event.id)) return current;
      return [...current, event];
    });
  }

  function finishAgentTransport(controller: AbortController | undefined): void {
    if (controller === undefined || agentTransportController.current !== controller) return;
    agentTransportController.current = undefined;
    setLiveAgentRunId(undefined);
    setLiveAgentEvents([]);
  }

  const citationIds = useMemo(() => {
    const values = diagnosis?.report?.citations;
    if (!Array.isArray(values)) return [];
    return values.flatMap((value) => {
      if (!value || typeof value !== "object" || !("citation_id" in value)) return [];
      return typeof value.citation_id === "string" ? [value.citation_id] : [];
    });
  }, [diagnosis]);
  const dependencyFailure = error instanceof ApiClientError && error.category === "dependency";
  const diagnosisIsTerminal = diagnosis
    ? TERMINAL_DIAGNOSIS_STATUSES.has(diagnosis.status)
    : false;
  const openIntervention = diagnosis?.expert_interventions?.find(
    (item) => item.status === "OPEN",
  );
  const hasExpertRevision = diagnosis?.expert_interventions?.some(
    (item) => item.revisions.length > 0,
  ) ?? false;
  const aiCitationOptions = reportCitationIds(diagnosis?.ai_report).map((value) => ({
    value,
    label: value,
  }));

  function openRevision() {
    const source = openIntervention?.ai_report_snapshot ?? diagnosis?.ai_report;
    revisionForm.setFieldsValue({
      outcome: "COMPLETED",
      conclusion: reportString(source, "conclusion"),
      citation_ids: reportCitationIds(source),
      contradictions: reportStrings(source, "contradictions").join("\n"),
      missing_information: "",
      next_checks: reportStrings(source, "next_checks").join("\n"),
      confidence: reportNumber(source, "confidence") ?? 0.8,
      reason: "专家复核原始证据后修订诊断结论与后续检查建议",
    });
    setRevisionOpen(true);
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>Incident 诊断工作台</Typography.Title>
        <ModelRuntimeStatus
          requiredComponents={["diagnosis"]}
          executions={diagnosis?.model_execution ? [diagnosis.model_execution] : []}
          actionState={busy
            ? "loading"
            : diagnosis?.status === "FAILED" || error
              ? "failure"
              : diagnosis?.model_execution
                ? "success"
                : "idle"}
          actionLabel="大模型诊断"
          actionError={error ?? diagnosis?.stop_reason}
        />
        {dependencyFailure ? <Alert type="warning" showIcon message="诊断编排暂时降级" description="Incident 与既有事实仍可读取；系统不会把未完成的 Agent 结果伪装成诊断结论。" /> : null}
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!incident && !error ? <LoadingState /> : null}
        {incident ? (
          <Card title={incident.incident_id} extra={<StatusTag status={incident.status} version={incident.version} />}>
            <div className="page-stack">
              <FactGrid facts={[["设备", incident.asset_id], ["证据包", incident.evidence_bundle_id], ["报告人", incident.reporter_subject_id], ["更新时间", incident.updated_at]]} />
              <Typography.Paragraph>{incident.description}</Typography.Paragraph>
              {incident.status === "SUBMITTED" ? (
                <Alert
                  type="info"
                  showIcon
                  message="诊断前需要完成分诊"
                  description="当前 Incident 已提交但尚未分诊。完成分诊后，系统才允许按受控状态机启动 Agent 诊断。"
                  action={(
                    <Button loading={triageBusy} onClick={() => void triage()}>
                      完成分诊
                    </Button>
                  )}
                />
              ) : null}
              <RealtimeAudioPanel
                incidentId={incident.incident_id}
                diagnosisRunId={diagnosis?.diagnosis_run_id}
                diagnosisVersion={diagnosis?.version}
                speechSynthesisId={speech?.synthesis_id}
                onDiagnosisCreated={(created) => {
                  const shouldAutoSpeak = autoSpeakAfterReanalysis && safetyAcknowledged;
                  setDiagnosis(created);
                  setSpeech(undefined);
                  setPendingAutoSpeechRunId(
                    shouldAutoSpeak ? created.diagnosis_run_id : undefined,
                  );
                  setAutoSpeakAfterReanalysis(false);
                  if (!shouldAutoSpeak) setSafetyAcknowledged(false);
                }}
              />
              <Input.TextArea
                aria-label="本次诊断补充关注点"
                value={agentInstruction}
                onChange={(event) => setAgentInstruction(event.target.value)}
                placeholder="可选：填写本次需要重点核查的人工确认事实；内容会通过 AG-UI RunAgentInput 进入受控诊断上下文。"
                rows={3}
                maxLength={4000}
                showCount
                disabled={incident.status !== "TRIAGED"}
              />
              {diagnosis ? (
                <Space.Compact block>
                  <Button
                    type="primary"
                    block
                    onClick={() => focusDiagnosisResult(
                      diagnosisIsTerminal
                        ? "已定位到最新诊断结果。"
                        : "已定位到当前诊断运行进度。",
                    )}
                  >
                    {diagnosisIsTerminal ? "查看最新诊断结果" : "查看诊断进度"}
                  </Button>
                  {diagnosisIsTerminal ? (
                    <Button
                      loading={busy}
                      disabled={agentControlBusy || incident.status !== "TRIAGED"}
                      onClick={() => void diagnose()}
                    >
                      重新运行诊断
                    </Button>
                  ) : null}
                </Space.Compact>
              ) : (
                <Button
                  type="primary"
                  loading={busy}
                  disabled={agentControlBusy || incident.status !== "TRIAGED"}
                  onClick={() => void diagnose()}
                >
                  {incident.status === "SUBMITTED"
                    ? "请先完成分诊"
                    : "按当前 Incident 版本启动诊断"}
                </Button>
              )}
            </div>
          </Card>
        ) : null}
        {operations ? (
          <Card
            title="Incident 解决闭环"
            extra={<StatusTag status={operations.status} version={operations.version} />}
          >
            <div className="page-stack">
              {operations.status === "RESOLVED" ? (
                <Alert
                  type="info"
                  showIcon
                  message="等待客户或专家确认后关闭"
                  description="设备服务任务已经解决，但 Incident 仍保留独立确认阶段；训练反馈在最终关闭前保持阻断。"
                />
              ) : null}
              {resolution ? (
                <Card size="small" title={`解决事实 · ${resolution.mode}`}>
                  <FactGrid facts={[
                    ["Resolution", resolution.resolution_id],
                    ["诊断运行", resolution.diagnosis_run_id ?? "—"],
                    ["工单", resolution.work_order_id ?? "—"],
                    ["完成事实", resolution.completion_id ?? "—"],
                    ["复核事实", resolution.verification_id ?? "—"],
                    ["解决人", resolution.resolved_by_subject_id],
                    ["解决时间", resolution.resolved_at],
                  ]} />
                  <Typography.Paragraph>{resolution.summary}</Typography.Paragraph>
                  <Typography.Text type="secondary">
                    {`Evidence: ${resolution.evidence_ids.join(", ") || "—"}`}
                  </Typography.Text>
                </Card>
              ) : (
                <Typography.Text type="secondary">尚未保存 IncidentResolution。</Typography.Text>
              )}
              <Space wrap>
                {operations.legal_actions.includes("RESOLVE_REMOTE") ? (
                  <Button
                    type="primary"
                    onClick={() => {
                      remoteResolutionForm.setFieldsValue({
                        summary: "",
                        evidence_ids: incident?.evidence_bundle_id ?? "",
                      });
                      setRemoteResolutionOpen(true);
                    }}
                  >
                    记录远程解决
                  </Button>
                ) : null}
                {operations.legal_actions.includes("CLOSE") ? (
                  <Button
                    type="primary"
                    onClick={() => {
                      incidentCloseForm.setFieldsValue({
                        confirmation_type: "EXPERT",
                        reason: "专家确认解决事实、验证结果和风险处置完整",
                      });
                      setIncidentCloseOpen(true);
                    }}
                  >
                    确认并关闭 Incident
                  </Button>
                ) : null}
                <Button onClick={() => void load()}>刷新解决状态</Button>
              </Space>
            </div>
          </Card>
        ) : null}
        {incident ? (
          <CustomerNotificationProposalPanel
            incidentId={incident.incident_id}
            incidentVersion={incident.version}
            incidentStatus={incident.status}
          />
        ) : null}
        {incident ? (
          <PurchaseRequestPanel
            incidentId={incident.incident_id}
            incidentVersion={incident.version}
            incidentStatus={incident.status}
            assetId={incident.asset_id}
          />
        ) : null}
        {incident ? (
          <ServiceQuotationPanel
            incidentId={incident.incident_id}
            mode="staff"
            diagnosisStatus={diagnosis?.status}
          />
        ) : null}
        {incident ? (
          <RepairWorkOrderAuthorizationPanel
            incidentId={incident.incident_id}
            incidentVersion={incident.version}
            diagnosisRunId={diagnosis?.diagnosis_run_id}
            diagnosisStatus={diagnosis?.status}
            diagnosisVersion={diagnosis?.version}
          />
        ) : null}
        {incident ? (
          <EquipmentControlHandoffPanel
            incidentId={incident.incident_id}
            incidentVersion={incident.version}
            incidentStatus={incident.status}
            evidenceBundleId={incident.evidence_bundle_id}
          />
        ) : null}
        {incident ? (
          <RefundRequestPanel
            incidentId={incident.incident_id}
            customerUpdates={customerUpdates}
          />
        ) : null}
        {customerUpdates.length > 0 ? (
          <Card title="客户补充资料与结果确认">
            <List
              dataSource={customerUpdates}
              renderItem={(update) => (
                <List.Item>
                  <List.Item.Meta
                    title={<Space><Tag>{update.update_type}</Tag>{update.work_order_id ? <span>{update.work_order_id}</span> : null}</Space>}
                    description={`${update.occurred_at} · ${update.message}${update.evidence_ids.length > 0 ? ` · Evidence: ${update.evidence_ids.join(", ")}` : ""}`}
                  />
                </List.Item>
              )}
            />
          </Card>
        ) : null}
        {diagnosis ? (
          <>
            <div ref={diagnosisResultRef} data-testid="diagnosis-result-section">
              <Card title="诊断结果" extra={<StatusTag status={diagnosis.status} version={diagnosis.version} />}>
                <div className="page-stack">
                  {diagnosisActionMessage ? (
                    <Alert
                      type="success"
                      showIcon
                      closable
                      message={diagnosisActionMessage}
                      onClose={() => setDiagnosisActionMessage(undefined)}
                    />
                  ) : null}
                <FactGrid facts={[["诊断运行", diagnosis.diagnosis_run_id], ["Agent 运行", diagnosis.agent_run_id], ["工作流", diagnosis.workflow_id], ["停止原因", diagnosis.stop_reason]]} />
                {diagnosis.ai_report ? (
                  <Card size="small" title="AI 原始报告（不可覆盖）">
                    <pre className="json-report">{JSON.stringify(diagnosis.ai_report, null, 2)}</pre>
                  </Card>
                ) : (
                  <Typography.Text type="secondary">诊断仍在执行或等待更多事实。</Typography.Text>
                )}
                {hasExpertRevision && diagnosis.report ? (
                  <Card size="small" title="当前生效的专家修订">
                    <pre className="json-report">{JSON.stringify(diagnosis.report, null, 2)}</pre>
                  </Card>
                ) : null}
                {diagnosis.status === "WAITING_EXPERT" ? (
                  <Alert
                    type="warning"
                    showIcon
                    message="诊断已由人工专家接管"
                    description={openIntervention?.ai_report_digest
                      ? "AI 原稿已冻结，等待接管专家提交追加式修订。"
                      : "Agent 正在收口在途步骤；晚到结果只会保存为 AI 原稿，不会覆盖专家状态。"}
                  />
                ) : null}
                <Space wrap>
                  {diagnosis.legal_actions?.includes("CANCEL_AGENT") ? (
                    <Button
                      danger
                      loading={agentControlBusy}
                      disabled={busy}
                      onClick={() => {
                        cancelAgentForm.setFieldsValue({
                          reason: "现场人员主动停止本次诊断，保留已有事件并安全收口",
                        });
                        setCancelAgentOpen(true);
                      }}
                    >
                      取消 Agent 运行
                    </Button>
                  ) : null}
                  {diagnosis.legal_actions?.includes("RESUME_AGENT") ? (
                    <Button
                      type="primary"
                      loading={agentControlBusy}
                      disabled={busy || !openInterruptId}
                      onClick={() => setResumeAgentOpen(true)}
                    >
                      补充信息并恢复诊断
                    </Button>
                  ) : null}
                  {diagnosis.legal_actions?.includes("TAKE_OVER_DIAGNOSIS") ? (
                    <Button
                      danger
                      onClick={() => {
                        takeoverForm.setFieldsValue({
                          reason: "诊断结论需要领域专家结合现场风险进行独立复核",
                        });
                        setTakeoverOpen(true);
                      }}
                    >
                      专家接管
                    </Button>
                  ) : null}
                  {diagnosis.legal_actions?.includes("SUBMIT_EXPERT_REVISION") ? (
                    <Button type="primary" onClick={openRevision}>
                      提交专家修订
                    </Button>
                  ) : null}
                  {diagnosis.report && ["COMPLETED", "NEEDS_INFORMATION"].includes(diagnosis.status) ? (
                    <Link href={`/collaboration?incident_id=${encodeURIComponent(incidentId)}&diagnosis_run_id=${encodeURIComponent(diagnosis.diagnosis_run_id)}`}>
                      <Button>升级至供应商 Agent</Button>
                    </Link>
                  ) : null}
                </Space>
                {(diagnosis.expert_interventions ?? []).map((item) => (
                  <Card
                    key={item.intervention_id}
                    size="small"
                    title={`接管记录 · ${item.status}`}
                  >
                    <Typography.Paragraph>{item.takeover_reason}</Typography.Paragraph>
                    <FactGrid facts={[
                      ["接管人", item.requested_by_subject_id],
                      ["原稿摘要", item.ai_report_digest ?? "等待 Agent 收口"],
                      ["修订数", item.revisions.length],
                      ["完成时间", item.completed_at ?? "—"],
                    ]} />
                  </Card>
                ))}
                {pendingAutoSpeechRunId === diagnosis.diagnosis_run_id ? (
                  <Alert
                    type="info"
                    showIcon
                    message="已等待新诊断完成后生成受控语音"
                    description="系统仍会重新校验报告确定性、安全警告确认、模型路由和租户配额；实时语音保持连接时才会自动播报。"
                    action={(
                      <Button size="small" onClick={() => setPendingAutoSpeechRunId(undefined)}>
                        取消自动播报
                      </Button>
                    )}
                  />
                ) : null}
                {diagnosis.status === "COMPLETED" && diagnosis.report ? (
                  <Card size="small" title="免手语音播报">
                    <div className="page-stack">
                      <Alert
                        type="warning"
                        showIcon
                        message="播报包含不可省略的维修安全警告"
                        description={TTS_SAFETY_WARNING}
                      />
                      <Checkbox
                        checked={safetyAcknowledged}
                        onChange={(event) => {
                          setSafetyAcknowledged(event.target.checked);
                          if (!event.target.checked) setAutoSpeakAfterReanalysis(false);
                        }}
                      >
                        我已在屏幕上阅读安全警告；语音仅作辅助，不代表设备动作已获批准或执行。
                      </Checkbox>
                      <Checkbox
                        checked={autoSpeakAfterReanalysis}
                        disabled={!safetyAcknowledged}
                        onChange={(event) => setAutoSpeakAfterReanalysis(event.target.checked)}
                      >
                        使用确认信息重新诊断时，完成后自动生成语音；实时会话在线时立即播报。
                      </Checkbox>
                      <Button
                        loading={speechBusy}
                        disabled={!safetyAcknowledged}
                        onClick={() => void synthesizeSpeech()}
                      >
                        生成受控语音
                      </Button>
                      {speech ? (
                        <div>
                          <audio
                            controls
                            preload="none"
                            src={`/api/backend${speech.audio_url}`}
                          >
                            当前浏览器不支持音频播放，请继续使用屏幕文字。
                          </audio>
                          <Typography.Text type="secondary">
                            {`TTS ${speech.resolved_release_id} · ${speech.size_bytes} bytes`}
                          </Typography.Text>
                        </div>
                      ) : null}
                    </div>
                  </Card>
                ) : null}
                <Button onClick={() => void load()}>刷新服务端状态</Button>
                </div>
              </Card>
            </div>
            <DiagnosisFeedbackPanel diagnosisRunId={diagnosis.diagnosis_run_id} />
            <CitationDrawer citationIds={citationIds} />
          </>
        ) : null}
        {liveAgentRunId ? (
          <AgentEventTimeline
            key={`transport-${liveAgentRunId}`}
            agentRunId={liveAgentRunId}
            transportEvents={liveAgentEvents}
            onInterrupt={setOpenInterruptId}
          />
        ) : diagnosis ? (
          <AgentEventTimeline
            key={diagnosis.agent_run_id}
            agentRunId={diagnosis.agent_run_id}
            onTerminal={load}
            onInterrupt={setOpenInterruptId}
          />
        ) : null}
        <Modal
          title="记录远程解决"
          open={remoteResolutionOpen}
          onCancel={() => setRemoteResolutionOpen(false)}
          onOk={() => remoteResolutionForm.submit()}
          confirmLoading={resolutionBusy}
          okText="确认记录解决"
        >
          <Alert
            type="info"
            showIcon
            message="远程解决不会创建工单"
            description="服务端会重新校验完成诊断、证据、当前版本和关联工单；成功后 Incident 进入 RESOLVED，仍需客户或专家确认。"
            style={{ marginBottom: 16 }}
          />
          <Form
            form={remoteResolutionForm}
            layout="vertical"
            onFinish={(values) => void resolveRemote(values)}
          >
            <Form.Item name="summary" label="解决摘要" rules={[{ required: true, min: 4 }]}>
              <Input.TextArea rows={4} maxLength={4000} showCount />
            </Form.Item>
            <Form.Item
              name="evidence_ids"
              label="证据 ID（每行一项）"
              rules={[{ required: true, min: 1 }]}
            >
              <Input.TextArea rows={4} maxLength={4000} />
            </Form.Item>
          </Form>
        </Modal>
        <Modal
          title="确认并关闭 Incident"
          open={incidentCloseOpen}
          onCancel={() => setIncidentCloseOpen(false)}
          onOk={() => incidentCloseForm.submit()}
          confirmLoading={resolutionBusy}
          okText="确认关闭"
        >
          <Alert
            type="warning"
            showIcon
            message="关闭不会覆盖解决事实"
            description="专家确认需要领域专家权限；使用客户确认时，服务端会重新绑定当前 Incident、报告人和已接受的追加式确认记录。"
            style={{ marginBottom: 16 }}
          />
          <Form
            form={incidentCloseForm}
            layout="vertical"
            onFinish={(values) => void confirmIncidentClose(values)}
          >
            <Form.Item name="confirmation_type" label="确认来源" rules={[{ required: true }]}>
              <Select options={[
                { value: "EXPERT", label: "领域专家确认" },
                { value: "CUSTOMER", label: "客户接受结果" },
              ]} />
            </Form.Item>
            <Form.Item name="customer_update_id" label="客户确认记录 ID">
              <Input maxLength={128} placeholder="选择客户确认时必填" />
            </Form.Item>
            <Form.Item name="reason" label="关闭依据" rules={[{ required: true, min: 8 }]}>
              <Input.TextArea rows={4} maxLength={2000} showCount />
            </Form.Item>
          </Form>
        </Modal>
        <Modal
          title="取消 Agent 运行"
          open={cancelAgentOpen}
          onCancel={() => setCancelAgentOpen(false)}
          onOk={() => cancelAgentForm.submit()}
          confirmLoading={agentControlBusy}
          okButtonProps={{ danger: true }}
          okText="确认安全停止"
        >
          <Alert
            type="warning"
            showIcon
            message="取消只停止本次 AgentRun"
            description="已持久化事件不会删除，Incident、证据和已经完成的只读查询仍保留；晚到的模型结果不会成为正式诊断结论。"
            style={{ marginBottom: 16 }}
          />
          <Form
            form={cancelAgentForm}
            layout="vertical"
            onFinish={(values) => void cancelCurrentAgent(values)}
          >
            <Form.Item name="reason" label="取消原因" rules={[{ required: true, min: 8 }]}>
              <Input.TextArea rows={4} maxLength={1024} showCount />
            </Form.Item>
          </Form>
        </Modal>
        <Modal
          title="补充信息并恢复诊断"
          open={resumeAgentOpen}
          onCancel={() => setResumeAgentOpen(false)}
          onOk={() => resumeAgentForm.submit()}
          confirmLoading={agentControlBusy}
          okText="创建后继 AgentRun"
        >
          <Alert
            type="info"
            showIcon
            message="原运行不会被覆盖"
            description="补充内容会作为人工确认事件保存；系统创建绑定原 Run、固定模型与知识版本的后继运行。"
            style={{ marginBottom: 16 }}
          />
          <Form
            form={resumeAgentForm}
            layout="vertical"
            onFinish={(values) => void resumeCurrentAgent(values)}
          >
            <Form.Item
              name="clarification"
              label="人工确认的补充事实"
              rules={[{ required: true, min: 2 }]}
            >
              <Input.TextArea rows={6} maxLength={4000} showCount />
            </Form.Item>
          </Form>
        </Modal>
        <Modal
          title="人工专家接管诊断"
          open={takeoverOpen}
          onCancel={() => setTakeoverOpen(false)}
          onOk={() => takeoverForm.submit()}
          confirmLoading={expertBusy}
          okText="确认接管"
        >
          <Alert
            type="warning"
            showIcon
            message="接管后 Agent 不再拥有最终结论控制权"
            description="在途结果仍会保存为 AI 原稿，专家修订将作为独立追加事实保存。"
            style={{ marginBottom: 16 }}
          />
          <Form form={takeoverForm} layout="vertical" onFinish={(values) => void takeOver(values)}>
            <Form.Item name="reason" label="接管原因" rules={[{ required: true, min: 8 }]}>
              <Input.TextArea rows={4} maxLength={1024} showCount />
            </Form.Item>
          </Form>
        </Modal>
        <Modal
          title="提交追加式专家修订"
          open={revisionOpen}
          width={820}
          onCancel={() => setRevisionOpen(false)}
          onOk={() => revisionForm.submit()}
          confirmLoading={expertBusy}
          okText="提交不可变修订"
        >
          <Form
            form={revisionForm}
            layout="vertical"
            onFinish={(values) => void submitRevision(values)}
          >
            <Form.Item name="outcome" label="处置结果" rules={[{ required: true }]}>
              <Select
                options={[
                  { value: "COMPLETED", label: "完成专家修订" },
                  { value: "ESCALATED", label: "升级到独立业务处置" },
                ]}
                onChange={(value) => {
                  if (value === "ESCALATED") {
                    revisionForm.setFieldsValue({
                      conclusion: undefined,
                      citation_ids: [],
                      missing_information: "现有证据不足以安全完成诊断，需要线下专家处置",
                      confidence: 0,
                    });
                  }
                }}
              />
            </Form.Item>
            <Form.Item name="conclusion" label="专家结论">
              <Input.TextArea rows={4} maxLength={4000} showCount />
            </Form.Item>
            <Form.Item name="citation_ids" label="沿用并重新授权的原稿引用">
              <Select mode="multiple" options={aiCitationOptions} />
            </Form.Item>
            <Form.Item name="confidence" label="专家置信度" rules={[{ required: true }]}>
              <InputNumber min={0} max={1} step={0.05} style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item name="contradictions" label="反证（每行一项）">
              <Input.TextArea rows={2} />
            </Form.Item>
            <Form.Item name="missing_information" label="缺失信息（每行一项）">
              <Input.TextArea rows={2} />
            </Form.Item>
            <Form.Item name="next_checks" label="后续检查（每行一项）">
              <Input.TextArea rows={3} />
            </Form.Item>
            <Form.Item name="reason" label="修订依据" rules={[{ required: true, min: 8 }]}>
              <Input.TextArea rows={3} maxLength={1024} showCount />
            </Form.Item>
          </Form>
        </Modal>
      </div>
    </AppShell>
  );
}

function splitLines(value?: string): string[] {
  return (value ?? "")
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

function reportCitationIds(report?: Record<string, unknown> | null): string[] {
  const citations = report?.citations;
  if (!Array.isArray(citations)) return [];
  return citations.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const citationId = Reflect.get(item, "citation_id");
    return typeof citationId === "string" ? [citationId] : [];
  });
}

function reportString(report: Record<string, unknown> | null | undefined, key: string) {
  const value = report?.[key];
  return typeof value === "string" ? value : undefined;
}

function reportStrings(
  report: Record<string, unknown> | null | undefined,
  key: string,
): string[] {
  const value = report?.[key];
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function reportNumber(report: Record<string, unknown> | null | undefined, key: string) {
  const value = report?.[key];
  return typeof value === "number" ? value : undefined;
}
