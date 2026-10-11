"use client";

import { Alert, Button, Card, Checkbox, Input, List, Space, Tag, Typography } from "antd";
import Link from "next/link";
import { useParams } from "next/navigation";
import { getSession } from "next-auth/react";
import { type ComponentProps, type ReactNode, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import { DiagnosisFeedbackPanel } from "@/components/m2/DiagnosisFeedbackPanel";
import { FieldVoiceGuidancePanel } from "@/components/m3/FieldVoiceGuidancePanel";
import { RemoteExpertCollaborationPanel } from "@/components/m3/RemoteExpertCollaborationPanel";
import { ModelRuntimeStatus } from "@/components/model-runtime/ModelRuntimeStatus";
import {
  type DiagnosisRun,
  type FieldEvidenceUpload,
  type FieldEdgeDiagnosisCandidate,
  type FieldEdgeDiagnosisPack,
  type FieldEdgeDiagnosisResult,
  type FieldEntryInput,
  type FieldOcrBlockDecision,
  type FieldQrCodeDecision,
  type FieldTranscriptDecision,
  type FieldWorkOrderEntry,
  type FieldVoiceGuidance,
  type EvidenceBundle,
  type ModelExecution,
  type PartMaterialMovementProposal,
  type PartIssueProposal,
  type WorkOrderPartAccounting,
  type WorkOrderPartAllocation,
  type WorkOrderRepairHistory,
  type WorkOrderOfflinePack,
  type WorkOrderView,
  acceptWorkOrder,
  completeFieldWorkOrder,
  confirmFieldEvidenceRecognition,
  createFieldEdgeDiagnosisPack,
  createFieldOfflinePack,
  decideFieldEdgeDiagnosisCandidate,
  getCurrentFieldOfflinePack,
  getFieldEvidenceRecognition,
  getFieldVoiceGuidance,
  getWorkOrder,
  getWorkOrderPartAccounting,
  getWorkOrderPartAllocation,
  getWorkOrderRepairHistory,
  holdWorkOrder,
  importFieldEdgeDiagnosisCandidate,
  listFieldEdgeDiagnosisCandidates,
  listFieldEvidenceUploads,
  listFieldWorkOrderEntries,
  proposeWorkOrderPartIssue,
  proposeWorkOrderPartConsumption,
  proposeWorkOrderPartReturn,
  reanalyzeDiagnosisWithFieldObservations,
  revokeFieldOfflinePack,
  resumeWorkOrder,
  startWorkOrder,
  startFieldEvidenceRecognition,
} from "@/lib/api/client";
import {
  loadFieldOfflinePack,
  removeFieldOfflinePack,
  saveFieldOfflinePack,
} from "@/lib/field-offline-pack";
import {
  type FieldEvidenceDraft,
  loadFieldEvidenceDrafts,
  removeFieldEvidenceDraft,
  saveFieldEvidenceDraft,
  syncFieldEvidenceDrafts,
} from "@/lib/field-evidence-drafts";
import {
  createFieldOperationId,
  pendingFieldEntries,
  queueFieldEntry,
  syncFieldEntries,
} from "@/lib/field-offline-queue";

declare global {
  interface HTMLAnchorElement {
    click(this: HTMLAnchorElement): void;
  }
}

const EVIDENCE_SCAN_POLL_INTERVAL_MS = 2_000;
const EVIDENCE_SCAN_POLL_TIMEOUT_MS = 60_000;
type FieldEntryAppendResult = "queued" | "synced";
type FieldEntryFeedbackScope = "evidence" | "facts";
type FieldEntryActionFeedback = {
  scope: FieldEntryFeedbackScope;
  result: FieldEntryAppendResult;
  message: string;
};
type FieldEntryActionError = {
  scope: FieldEntryFeedbackScope;
  error: unknown;
};

const DEFAULT_CUSTOMER_CONFIRMATION = "确认维修结果与现场状态";

export default function FieldWorkOrderPage() {
  const { workOrderId } = useParams<{ workOrderId: string }>();
  const online = useOnlineState();
  const [work, setWork] = useState<WorkOrderView>();
  const [entries, setEntries] = useState<FieldWorkOrderEntry[]>();
  const [fieldRediagnosisGuidance, setFieldRediagnosisGuidance] = useState<FieldVoiceGuidance>();
  const [fieldRediagnosisSelection, setFieldRediagnosisSelection] = useState<string[]>([]);
  const [fieldRediagnosisResult, setFieldRediagnosisResult] = useState<DiagnosisRun>();
  const [fieldRediagnosisError, setFieldRediagnosisError] = useState<unknown>();
  const [fieldRediagnosisBusy, setFieldRediagnosisBusy] = useState(false);
  const [fieldRediagnosisIdempotencyKey, setFieldRediagnosisIdempotencyKey] = useState<string>();
  const [repairHistory, setRepairHistory] = useState<WorkOrderRepairHistory>();
  const [partAllocation, setPartAllocation] = useState<WorkOrderPartAllocation>();
  const [partAllocationError, setPartAllocationError] = useState<unknown>();
  const [partAccounting, setPartAccounting] = useState<WorkOrderPartAccounting>();
  const [partAccountingError, setPartAccountingError] = useState<unknown>();
  const [partIssueProposal, setPartIssueProposal] = useState<PartIssueProposal>();
  const [partMovementProposal, setPartMovementProposal] = useState<PartMaterialMovementProposal>();
  const [partIssueIdempotencyKey, setPartIssueIdempotencyKey] = useState<string>();
  const [partMovementIdempotencyKeys, setPartMovementIdempotencyKeys] = useState<Record<string, string>>({});
  const [partMovementBusyId, setPartMovementBusyId] = useState<string>();
  const [offlinePack, setOfflinePack] = useState<WorkOrderOfflinePack>();
  const [offlineFallback, setOfflineFallback] = useState(false);
  const [offlineUnavailable, setOfflineUnavailable] = useState(false);
  const [edgePack, setEdgePack] = useState<FieldEdgeDiagnosisPack>();
  const [edgeCandidates, setEdgeCandidates] = useState<FieldEdgeDiagnosisCandidate[]>([]);
  const [edgeError, setEdgeError] = useState<unknown>();
  const [edgeRequestId, setEdgeRequestId] = useState<string>();
  const [edgeBusy, setEdgeBusy] = useState(false);
  const [edgeReviewReason, setEdgeReviewReason] = useState("仅接受为人工复核候选");
  const [edgeObservationDrafts, setEdgeObservationDrafts] = useState<Record<string, string>>({});
  const [evidenceUploads, setEvidenceUploads] = useState<FieldEvidenceUpload[]>([]);
  const [evidenceDrafts, setEvidenceDrafts] = useState<FieldEvidenceDraft[]>([]);
  const [fieldRecognitionEvidence, setFieldRecognitionEvidence] = useState<Record<string, EvidenceBundle>>({});
  const [fieldRecognitionBusyId, setFieldRecognitionBusyId] = useState<string>();
  const [evidenceError, setEvidenceError] = useState<unknown>();
  const [authoritativeWorkLoaded, setAuthoritativeWorkLoaded] = useState(false);
  const subjectIdRef = useRef<string | undefined>(undefined);
  const evidenceSyncInFlightRef = useRef(false);
  const entryActionInFlightRef = useRef(false);
  const [pendingCount, setPendingCount] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>();
  const [entryActionBusyKey, setEntryActionBusyKey] = useState<string>();
  const [entryActionFeedback, setEntryActionFeedback] = useState<FieldEntryActionFeedback>();
  const [entryActionError, setEntryActionError] = useState<FieldEntryActionError>();
  const [locallyAttachedEvidenceIds, setLocallyAttachedEvidenceIds] = useState<string[]>([]);
  const [stepDescription, setStepDescription] = useState("");
  const [quantity, setQuantity] = useState("1");
  const [returnQuantity, setReturnQuantity] = useState("1");
  const [note, setNote] = useState("");
  const [signedBy, setSignedBy] = useState("");
  const [confirmation, setConfirmation] = useState(DEFAULT_CUSTOMER_CONFIRMATION);
  const [rootCause, setRootCause] = useState("");
  const [costAmount, setCostAmount] = useState("0.00");
  const [holdReason, setHoldReason] = useState("等待现场安全条件恢复");
  const pendingEvidenceKey = useMemo(
    () => evidenceUploads
      .filter((upload) => upload.scan_state === "PENDING")
      .map((upload) => upload.evidence_upload_id)
      .sort()
      .join("|"),
    [evidenceUploads],
  );
  const modelExecutions = useMemo<ModelExecution[]>(() => {
    const current = Object.values(fieldRecognitionEvidence).flatMap(
      (evidence) => evidence.model_executions ?? [],
    );
    if (fieldRediagnosisResult?.model_execution) {
      current.push(fieldRediagnosisResult.model_execution);
    }
    return current;
  }, [fieldRecognitionEvidence, fieldRediagnosisResult]);
  const attachedEvidenceIds = useMemo(() => {
    const evidenceIds = new Set(locallyAttachedEvidenceIds);
    for (const entry of entries ?? []) {
      if (entry.entry_type !== "EVIDENCE") continue;
      const evidenceId = stringPayloadValue(entry.payload, "evidence_id");
      if (evidenceId) evidenceIds.add(evidenceId);
    }
    for (const pending of pendingFieldEntries(workOrderId)) {
      if (pending.input.entry_type !== "EVIDENCE") continue;
      const evidenceId = pending.input.evidence_id;
      if (typeof evidenceId === "string") evidenceIds.add(evidenceId);
    }
    return evidenceIds;
  }, [entries, locallyAttachedEvidenceIds, pendingCount, workOrderId]);

  const refreshPendingCount = useCallback(() => {
    setPendingCount(pendingFieldEntries(workOrderId).length);
  }, [workOrderId]);

  const load = useCallback(async () => {
    setError(undefined);
    setEvidenceError(undefined);
    setAuthoritativeWorkLoaded(false);
    setPartAllocationError(undefined);
    setPartAccountingError(undefined);
    setEdgeError(undefined);
    setFieldRediagnosisError(undefined);
    let subjectId = subjectIdRef.current;
    const session = await getSession().catch(() => null);
    subjectId = session?.user?.subjectId ?? subjectId;
    if (subjectId) subjectIdRef.current = subjectId;
    try {
      const workResult = await getWorkOrder(workOrderId);
      const needsParts = workResult.workOrder.initial_parts_required !== false;
      const allocationPromise = needsParts ? getWorkOrderPartAllocation(workOrderId).then(
        (result) => ({ result, error: undefined }),
        (caught: unknown) => ({ result: undefined, error: caught }),
      ) : Promise.resolve({ result: undefined, error: undefined });
      const accountingPromise = needsParts ? getWorkOrderPartAccounting(workOrderId).then(
        (result) => ({ result, error: undefined }),
        (caught: unknown) => ({ result: undefined, error: caught }),
      ) : Promise.resolve({ result: undefined, error: undefined });
      const evidencePromise = listFieldEvidenceUploads(workOrderId).then(
        (result) => ({ result, error: undefined }),
        (caught: unknown) => ({ result: undefined, error: caught }),
      );
      const edgeCandidatesPromise = listFieldEdgeDiagnosisCandidates(workOrderId).then(
        (result) => ({ result, error: undefined }),
        (caught: unknown) => ({ result: undefined, error: caught }),
      );
      const guidancePromise = getFieldVoiceGuidance(workOrderId).then(
        (result) => ({ result, error: undefined }),
        (caught: unknown) => ({ result: undefined, error: caught }),
      );
      const [
        entryResult,
        historyResult,
        allocationResult,
        accountingResult,
        evidenceResult,
        edgeCandidatesResult,
        guidanceResult,
      ] = await Promise.all([
        listFieldWorkOrderEntries(workOrderId),
        getWorkOrderRepairHistory(workOrderId),
        allocationPromise,
        accountingPromise,
        evidencePromise,
        edgeCandidatesPromise,
        guidancePromise,
      ]);
      setWork(workResult.workOrder);
      setEntries(entryResult.entries);
      setRepairHistory(historyResult.history);
      setPartAllocation(allocationResult.result?.allocation);
      setPartAllocationError(allocationResult.error);
      setPartAccounting(accountingResult.result?.accounting);
      setPartAccountingError(accountingResult.error);
      setEvidenceUploads(evidenceResult.result?.uploads ?? []);
      setEvidenceError(evidenceResult.error);
      setEdgeCandidates(edgeCandidatesResult.result?.candidates ?? []);
      setEdgeError(edgeCandidatesResult.error);
      setEdgeRequestId(edgeCandidatesResult.result?.requestId);
      setFieldRediagnosisGuidance(guidanceResult.result?.guidance);
      setFieldRediagnosisError(guidanceResult.error);
      subjectId = subjectId ?? workResult.workOrder.assigned_subject_id ?? undefined;
      if (subjectId) subjectIdRef.current = subjectId;
      setEvidenceDrafts(subjectId
        ? await loadFieldEvidenceDrafts(subjectId, workOrderId)
        : []);
      setAuthoritativeWorkLoaded(true);
      setOfflineFallback(false);
      setOfflineUnavailable(false);
      try {
        const current = await getCurrentFieldOfflinePack(workOrderId);
        if (subjectId && current.pack.subject_id === subjectId) {
          await saveFieldOfflinePack(current.pack);
          setOfflinePack(current.pack);
        } else {
          setOfflinePack(undefined);
        }
      } catch {
        setOfflinePack(undefined);
        if (subjectId) {
          await Promise.resolve(removeFieldOfflinePack(subjectId, workOrderId))
            .catch(() => undefined);
        }
      }
    } catch (caught) {
      if (!navigator.onLine) {
        const cached = subjectId
          ? await loadFieldOfflinePack(subjectId, workOrderId).catch(() => undefined)
          : undefined;
        setWork(undefined);
        setEntries(undefined);
        setRepairHistory(undefined);
        setPartAllocation(undefined);
        setPartAccounting(undefined);
        setEvidenceUploads([]);
        setEvidenceDrafts([]);
        setEdgeCandidates([]);
        setEdgePack(undefined);
        setEdgeError(undefined);
        setFieldRediagnosisGuidance(undefined);
        setFieldRediagnosisError(undefined);
        if (cached) {
          setOfflinePack(cached);
          setOfflineFallback(true);
          setOfflineUnavailable(false);
          setError(undefined);
        } else {
          setOfflinePack(undefined);
          setOfflineFallback(false);
          setOfflineUnavailable(true);
          setError(undefined);
        }
      } else {
        setOfflineFallback(false);
        setOfflineUnavailable(false);
        setError(caught);
      }
    }
    refreshPendingCount();
  }, [refreshPendingCount, workOrderId]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!online || !authoritativeWorkLoaded || !pendingEvidenceKey) return;

    let cancelled = false;
    let requestInFlight = false;
    let timerId = 0;
    const startedAt = Date.now();
    const poll = async () => {
      if (cancelled || requestInFlight) return;
      if (Date.now() - startedAt >= EVIDENCE_SCAN_POLL_TIMEOUT_MS) {
        setEvidenceError(new Error(
          "证据扫描等待超时，请稍后点击“刷新扫描状态”重试",
        ));
        window.clearInterval(timerId);
        return;
      }
      requestInFlight = true;
      try {
        const result = await listFieldEvidenceUploads(workOrderId);
        if (!cancelled) {
          setEvidenceUploads(result.uploads);
          setEvidenceError(undefined);
          if (!result.uploads.some((upload) => upload.scan_state === "PENDING")) {
            window.clearInterval(timerId);
          }
        }
      } catch (caught) {
        if (!cancelled) {
          setEvidenceError(caught);
          window.clearInterval(timerId);
        }
      } finally {
        requestInFlight = false;
      }
    };
    timerId = window.setInterval(
      () => void poll(),
      EVIDENCE_SCAN_POLL_INTERVAL_MS,
    );
    return () => {
      cancelled = true;
      window.clearInterval(timerId);
    };
  }, [authoritativeWorkLoaded, online, pendingEvidenceKey, workOrderId]);

  const activeFieldRediagnosis = fieldRediagnosisGuidance?.field_reanalysis;
  useEffect(() => {
    if (
      !online
      || !activeFieldRediagnosis
      || !isFieldRediagnosisRunning(activeFieldRediagnosis.status)
    ) return;

    let cancelled = false;
    let requestInFlight = false;
    let timerId = 0;
    const poll = async () => {
      if (cancelled || requestInFlight) return;
      requestInFlight = true;
      try {
        const current = await getFieldVoiceGuidance(workOrderId);
        if (!cancelled) {
          setFieldRediagnosisGuidance(current.guidance);
          setFieldRediagnosisError(undefined);
        }
      } catch (caught) {
        if (!cancelled) {
          setFieldRediagnosisError(caught);
          window.clearInterval(timerId);
        }
      } finally {
        requestInFlight = false;
      }
    };
    timerId = window.setInterval(() => void poll(), 2_000);
    return () => {
      cancelled = true;
      window.clearInterval(timerId);
    };
  }, [activeFieldRediagnosis?.diagnosis_run_id, activeFieldRediagnosis?.status, online, workOrderId]);

  const currentRound = repairHistory?.current_round ?? 1;
  const currentRoundView = repairHistory?.rounds.find(
    (round) => round.round_number === currentRound,
  );
  const entryCheckpoint = currentRoundView?.entry_sequence_checkpoint ?? 0;
  const completionFacts = useMemo(() => {
    const currentEntries = entries?.filter((entry) => entry.sequence > entryCheckpoint);
    return {
      steps: currentEntries?.filter((entry) => entry.entry_type === "STEP" && entry.payload.outcome === "COMPLETED").length ?? 0,
      evidence: currentEntries?.filter((entry) => entry.entry_type === "EVIDENCE").length ?? 0,
      signatures: currentEntries?.filter((entry) => entry.entry_type === "SIGNATURE" && entry.payload.signature_role === "CUSTOMER").length ?? 0,
    };
  }, [entries, entryCheckpoint]);
  const partQuantity = Number(quantity);
  const fieldRediagnosisEntries = useMemo(
    () => (entries ?? []).filter((entry) => (
      entry.entry_type === "NOTE" || entry.entry_type === "AI_OBSERVATION"
    )),
    [entries],
  );
  const fieldRediagnosisCanRequest = Boolean(
    fieldRediagnosisGuidance?.legal_actions.includes("REQUEST_FIELD_REANALYSIS"),
  );
  const fieldRediagnosisTerminalDiagnosis = activeFieldRediagnosis
    && fieldRediagnosisGuidance?.diagnosis?.diagnosis_run_id
      === activeFieldRediagnosis.diagnosis_run_id
    ? fieldRediagnosisGuidance.diagnosis
    : undefined;
  const fieldRediagnosisCanFeedback = Boolean(
    activeFieldRediagnosis
    && fieldRediagnosisTerminalDiagnosis
    && ["COMPLETED", "NEEDS_INFORMATION"].includes(activeFieldRediagnosis.status),
  );
  const partQuantityValid = Boolean(
    partAllocation
    && partAllocation.usage_enabled
    && Number.isInteger(partQuantity)
    && partQuantity >= 1
    && partQuantity <= partAllocation.approved_quantity,
  );

  async function transition(operation: (current: WorkOrderView) => Promise<unknown>) {
    if (!work) return;
    setBusy(true);
    setError(undefined);
    try {
      await operation(work);
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  async function requestFieldRediagnosis() {
    if (!work || !online || fieldRediagnosisSelection.length === 0) return;
    const idempotencyKey = fieldRediagnosisIdempotencyKey ?? crypto.randomUUID();
    setFieldRediagnosisIdempotencyKey(idempotencyKey);
    setFieldRediagnosisBusy(true);
    setFieldRediagnosisError(undefined);
    try {
      const current = await getFieldVoiceGuidance(workOrderId);
      setFieldRediagnosisGuidance(current.guidance);
      const diagnosis = current.guidance.diagnosis;
      if (
        !diagnosis
        || !current.guidance.legal_actions.includes("REQUEST_FIELD_REANALYSIS")
      ) {
        throw new Error("当前工单不再允许请求中心复诊，请刷新后重试");
      }
      const selectedEntryIds = fieldRediagnosisEntries
        .filter((entry) => fieldRediagnosisSelection.includes(entry.entry_id))
        .sort((left, right) => left.sequence - right.sequence)
        .map((entry) => entry.entry_id);
      if (selectedEntryIds.length !== fieldRediagnosisSelection.length) {
        throw new Error("所选现场观察已变化，请重新选择");
      }
      const result = await reanalyzeDiagnosisWithFieldObservations(
        diagnosis.diagnosis_run_id,
        work.work_order_id,
        current.guidance.work_order_version,
        selectedEntryIds,
        diagnosis.version,
        idempotencyKey,
      );
      setFieldRediagnosisResult(result.diagnosis);
      setFieldRediagnosisSelection([]);
      setFieldRediagnosisIdempotencyKey(undefined);
      const refreshed = await getFieldVoiceGuidance(workOrderId);
      setFieldRediagnosisGuidance(refreshed.guidance);
    } catch (caught) {
      setFieldRediagnosisError(caught);
    } finally {
      setFieldRediagnosisBusy(false);
    }
  }

  async function downloadOfflinePack() {
    if (!work) return;
    setBusy(true);
    setError(undefined);
    try {
      const result = await createFieldOfflinePack(
        workOrderId,
        work.version,
        createFieldOperationId(),
      );
      await saveFieldOfflinePack(result.pack);
      setOfflinePack(result.pack);
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  async function revokeOfflinePack() {
    if (!offlinePack || !offlinePack.legal_actions.includes("REVOKE")) return;
    setBusy(true);
    setError(undefined);
    try {
      await revokeFieldOfflinePack(
        workOrderId,
        offlinePack.pack_id,
        offlinePack.version,
        "现场工程师主动撤销浏览器离线工作包",
      );
      await removeFieldOfflinePack(offlinePack.subject_id, workOrderId);
      setOfflinePack(undefined);
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  async function removeOfflinePack() {
    if (!offlinePack) return;
    setBusy(true);
    try {
      await removeFieldOfflinePack(offlinePack.subject_id, workOrderId);
      setOfflinePack(undefined);
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  async function issueEdgeDiagnosisPack() {
    if (!work?.legal_actions.includes("CREATE_EDGE_DIAGNOSIS_PACK")) return;
    setEdgeBusy(true);
    setEdgeError(undefined);
    try {
      const result = await createFieldEdgeDiagnosisPack(
        workOrderId,
        work.version,
        createFieldOperationId(),
      );
      setEdgePack(result.pack);
      setEdgeRequestId(result.requestId);
      if (!result.pack.legal_actions.includes("IMPORT_RESULT")) {
        throw new Error("服务端未授权导入该边缘诊断包");
      }
      const file = {
        schema_version: result.pack.schema_version,
        token: result.pack.token,
        key_id: result.pack.key_id,
        pack_digest: result.pack.pack_digest,
      };
      const blob = new Blob([JSON.stringify(file, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `field-edge-diagnosis-pack-${result.pack.pack_id}.json`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (caught) {
      setEdgeError(caught);
    } finally {
      setEdgeBusy(false);
    }
  }

  async function importEdgeDiagnosisResult(file: File) {
    if (file.size < 1) {
      setEdgeError(new Error("结果文件不能为空"));
      return;
    }
    if (file.size > 64 * 1024) {
      setEdgeError(new Error("结果文件不得超过 64 KiB"));
      return;
    }
    setEdgeBusy(true);
    setEdgeError(undefined);
    try {
      const parsed = JSON.parse(await file.text()) as FieldEdgeDiagnosisResult;
      const result = await importFieldEdgeDiagnosisCandidate(
        workOrderId,
        createFieldOperationId(),
        parsed,
      );
      setEdgeCandidates((current) => [
        result.candidate,
        ...current.filter((item) => item.candidate_id !== result.candidate.candidate_id),
      ]);
      setEdgeRequestId(result.requestId);
    } catch (caught) {
      setEdgeError(caught);
    } finally {
      setEdgeBusy(false);
    }
  }

  async function decideEdgeDiagnosis(
    candidate: FieldEdgeDiagnosisCandidate,
    decision: "ACCEPTED" | "REJECTED",
  ) {
    const legalAction = decision === "ACCEPTED" ? "ACCEPT" : "REJECT";
    if (!candidate.legal_actions.includes(legalAction) || edgeReviewReason.trim().length < 3) return;
    setEdgeBusy(true);
    setEdgeError(undefined);
    try {
      const result = await decideFieldEdgeDiagnosisCandidate(
        workOrderId,
        candidate.candidate_id,
        candidate.version,
        decision,
        edgeReviewReason.trim(),
      );
      setEdgeCandidates((current) => current.map((item) => (
        item.candidate_id === result.candidate.candidate_id ? result.candidate : item
      )));
      setEdgeRequestId(result.requestId);
    } catch (caught) {
      setEdgeError(caught);
    } finally {
      setEdgeBusy(false);
    }
  }

  async function queueEdgeDiagnosisObservation(
    candidate: FieldEdgeDiagnosisCandidate,
  ) {
    const observation = edgeObservationDrafts[candidate.candidate_id]?.trim() ?? "";
    if (
      candidate.status !== "ACCEPTED"
      || !candidate.legal_actions.includes("CREATE_AI_OBSERVATION")
      || !observation
    ) return;
    await append({
      entry_type: "AI_OBSERVATION",
      observation,
      edge_diagnosis_candidate_id: candidate.candidate_id,
      edge_diagnosis_candidate_version: candidate.version,
      source_items: [
        {
          source_type: "EDGE_DIAGNOSIS_CANDIDATE",
          source_id: candidate.candidate_id,
        },
      ],
    });
    setEdgeObservationDrafts((current) => ({
      ...current,
      [candidate.candidate_id]: "",
    }));
  }

  async function append(
    input: Omit<FieldEntryInput, "client_operation_id" | "occurred_at">,
  ): Promise<FieldEntryAppendResult> {
    const entry: FieldEntryInput = {
      ...input,
      client_operation_id: createFieldOperationId(),
      occurred_at: new Date().toISOString(),
    };
    queueFieldEntry(workOrderId, entry);
    refreshPendingCount();
    if (!navigator.onLine) return "queued";
    setBusy(true);
    try {
      const result = await syncFieldEntries(workOrderId);
      refreshPendingCount();
      if (result.error) {
        setError(result.error);
        return "queued";
      }
      await load();
      return "synced";
    } catch (caught) {
      setError(caught);
      return "queued";
    } finally {
      setBusy(false);
      refreshPendingCount();
    }
  }

  async function runFieldEntryAction(
    key: string,
    scope: FieldEntryFeedbackScope,
    successMessage: string,
    input: Omit<FieldEntryInput, "client_operation_id" | "occurred_at">,
    onAccepted?: () => void,
  ) {
    if (entryActionInFlightRef.current) return;
    entryActionInFlightRef.current = true;
    setEntryActionBusyKey(key);
    setEntryActionFeedback(undefined);
    setEntryActionError(undefined);
    try {
      const result = await append(input);
      onAccepted?.();
      setEntryActionFeedback({
        scope,
        result,
        message: successMessage,
      });
    } catch (caught) {
      setEntryActionError({ scope, error: caught });
    } finally {
      entryActionInFlightRef.current = false;
      setEntryActionBusyKey(undefined);
    }
  }

  async function syncPending() {
    setBusy(true);
    const result = await syncFieldEntries(workOrderId);
    setBusy(false);
    refreshPendingCount();
    if (result.error) setError(result.error);
    else await load();
  }

  async function captureEvidenceDraft(file: File) {
    const subjectId = subjectIdRef.current;
    if (!work || !subjectId || !authoritativeWorkLoaded) return;
    setBusy(true);
    setEvidenceError(undefined);
    try {
      await saveFieldEvidenceDraft({
        subjectId,
        workOrderId,
        clientOperationId: createFieldOperationId(),
        file,
      });
      setEvidenceDrafts(await loadFieldEvidenceDrafts(subjectId, workOrderId));
    } catch (caught) {
      setEvidenceError(caught);
    } finally {
      setBusy(false);
    }
  }

  async function deleteEvidenceDraft(draft: FieldEvidenceDraft) {
    const subjectId = subjectIdRef.current;
    setBusy(true);
    setEvidenceError(undefined);
    try {
      await removeFieldEvidenceDraft(draft);
      setEvidenceDrafts(subjectId
        ? await loadFieldEvidenceDrafts(subjectId, workOrderId)
        : []);
    } catch (caught) {
      setEvidenceError(caught);
    } finally {
      setBusy(false);
    }
  }

  async function syncEvidenceDrafts() {
    const subjectId = subjectIdRef.current;
    if (!work || !subjectId || evidenceSyncInFlightRef.current) return;
    evidenceSyncInFlightRef.current = true;
    setBusy(true);
    setEvidenceError(undefined);
    try {
      const result = await syncFieldEvidenceDrafts({
        subjectId,
        workOrderId,
        workOrderVersion: work.version,
        online: navigator.onLine,
        authoritativeWorkLoaded,
      });
      if (result.error) {
        setEvidenceError(result.error);
        setEvidenceDrafts(await loadFieldEvidenceDrafts(subjectId, workOrderId));
        return;
      }
      await load();
    } catch (caught) {
      setEvidenceError(caught);
      setEvidenceDrafts(
        await loadFieldEvidenceDrafts(subjectId, workOrderId).catch(() => []),
      );
    } finally {
      evidenceSyncInFlightRef.current = false;
      setBusy(false);
    }
  }

  async function refreshEvidenceUploads() {
    const result = await listFieldEvidenceUploads(workOrderId);
    setEvidenceUploads(result.uploads);
  }

  async function startRecognition(
    upload: FieldEvidenceUpload,
    temporalAnalysis = false,
  ) {
    if (!work || !navigator.onLine || !upload.legal_actions.includes("RECOGNIZE")) return;
    setFieldRecognitionBusyId(upload.evidence_upload_id);
    setEvidenceError(undefined);
    try {
      await startFieldEvidenceRecognition(
        workOrderId,
        upload.evidence_upload_id,
        work.version,
        createFieldOperationId(),
        temporalAnalysis,
      );
      await refreshEvidenceUploads();
    } catch (caught) {
      setEvidenceError(caught);
    } finally {
      setFieldRecognitionBusyId(undefined);
    }
  }

  async function openRecognition(upload: FieldEvidenceUpload) {
    if (!navigator.onLine || !upload.recognition_bundle_id) return;
    setFieldRecognitionBusyId(upload.evidence_upload_id);
    setEvidenceError(undefined);
    try {
      const result = await getFieldEvidenceRecognition(
        workOrderId,
        upload.evidence_upload_id,
      );
      setFieldRecognitionEvidence((current) => ({
        ...current,
        [upload.evidence_upload_id]: result.evidence,
      }));
    } catch (caught) {
      setEvidenceError(caught);
    } finally {
      setFieldRecognitionBusyId(undefined);
    }
  }

  async function confirmRecognition(
    upload: FieldEvidenceUpload,
    evidence: EvidenceBundle,
    corrections: Record<string, string>,
    dispositions: Record<string, "ACCEPTED" | "REJECTED">,
    ocrBlockDecisions: Record<string, FieldOcrBlockDecision>,
    transcriptDecisions: Record<string, FieldTranscriptDecision>,
    videoEventDispositions: Record<string, "ACCEPTED" | "REJECTED">,
    qrCodeDecisions: Record<string, FieldQrCodeDecision>,
  ) {
    if (
      !evidence
      || !upload.legal_actions.includes("CONFIRM_RECOGNITION")
      || !fieldRecognitionReviewComplete(
        evidence,
        corrections,
        dispositions,
        ocrBlockDecisions,
        transcriptDecisions,
        videoEventDispositions,
        qrCodeDecisions,
      )
    ) return;
    setFieldRecognitionBusyId(upload.evidence_upload_id);
    setEvidenceError(undefined);
    try {
      const result = (evidence.qr_codes ?? []).length
        ? await confirmFieldEvidenceRecognition(
            workOrderId,
            upload.evidence_upload_id,
            evidence,
            corrections,
            dispositions,
            ocrBlockDecisions,
            transcriptDecisions,
            videoEventDispositions,
            qrCodeDecisions,
          )
        : await confirmFieldEvidenceRecognition(
            workOrderId,
            upload.evidence_upload_id,
            evidence,
            corrections,
            dispositions,
            ocrBlockDecisions,
            transcriptDecisions,
            videoEventDispositions,
          );
      setFieldRecognitionEvidence((current) => ({
        ...current,
        [upload.evidence_upload_id]: result.evidence,
      }));
      await refreshEvidenceUploads();
    } catch (caught) {
      setEvidenceError(caught);
    } finally {
      setFieldRecognitionBusyId(undefined);
    }
  }

  async function proposePartIssue() {
    if (!work || !partAllocation?.legal_actions.includes("PROPOSE_ISSUE")) return;
    const idempotencyKey = partIssueIdempotencyKey ?? crypto.randomUUID();
    setPartIssueIdempotencyKey(idempotencyKey);
    setBusy(true);
    setError(undefined);
    try {
      const result = await proposeWorkOrderPartIssue(
        workOrderId,
        work.version,
        idempotencyKey,
      );
      setPartIssueProposal(result.proposal);
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  async function proposePartConsumption(fieldEntryId: string) {
    if (
      !partAccounting
      || !partAccounting.legal_actions.includes("PROPOSE_CONSUMPTION")
      || !partAccounting.eligible_entries.some((entry) => entry.entry_id === fieldEntryId)
    ) return;
    const keyId = `CONSUME:${fieldEntryId}`;
    const idempotencyKey = partMovementIdempotencyKeys[keyId] ?? crypto.randomUUID();
    setPartMovementIdempotencyKeys((current) => ({ ...current, [keyId]: idempotencyKey }));
    setPartMovementBusyId(keyId);
    setError(undefined);
    try {
      const result = await proposeWorkOrderPartConsumption(
        workOrderId,
        partAccounting.work_order_version,
        fieldEntryId,
        idempotencyKey,
      );
      setPartMovementProposal(result.proposal);
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setPartMovementBusyId(undefined);
    }
  }

  async function proposePartReturn() {
    if (!partAccounting?.legal_actions.includes("PROPOSE_RETURN")) return;
    const parsedQuantity = Number(returnQuantity);
    if (
      !Number.isInteger(parsedQuantity)
      || parsedQuantity < 1
      || parsedQuantity > partAccounting.returnable_quantity
    ) return;
    const keyId = `RETURN:${parsedQuantity}`;
    const idempotencyKey = partMovementIdempotencyKeys[keyId] ?? crypto.randomUUID();
    setPartMovementIdempotencyKeys((current) => ({ ...current, [keyId]: idempotencyKey }));
    setPartMovementBusyId(keyId);
    setError(undefined);
    try {
      const result = await proposeWorkOrderPartReturn(
        workOrderId,
        partAccounting.work_order_version,
        parsedQuantity,
        idempotencyKey,
      );
      setPartMovementProposal(result.proposal);
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setPartMovementBusyId(undefined);
    }
  }

  if (!work && !offlineFallback && !offlineUnavailable && !error) {
    return <AppShell><LoadingState label="正在读取现场执行记录" /></AppShell>;
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>现场工单执行</Typography.Title>
        <ConnectivityAlert
          offlineFallback={offlineFallback}
          pendingCount={pendingCount}
          busy={busy}
          onSync={() => void syncPending()}
        />
        {!offlineFallback ? (
          <ModelRuntimeStatus
            requiredComponents={["ocr", "vlm", "diagnosis"]}
            executions={modelExecutions}
            actionState={fieldRecognitionBusyId || fieldRediagnosisBusy
              ? "loading"
              : evidenceError || fieldRediagnosisError
                ? "failure"
                : modelExecutions.length > 0
                  ? "success"
                  : "idle"}
            actionLabel="现场模型处理"
            actionError={evidenceError ?? fieldRediagnosisError}
          />
        ) : null}
        {error && !offlineFallback ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {offlineUnavailable ? (
          <Alert
            type="warning"
            showIcon
            message="无可用离线资料"
            description="本机没有通过主体、工单、Schema、有效期和内容摘要校验的工作包；受保护内容不会被展示。"
          />
        ) : null}
        {offlineFallback && offlinePack ? <OfflineWorkPackView pack={offlinePack} /> : null}
        {work && !offlineFallback ? (
          <>
            <Card title={work.work_order_id} extra={<StatusTag status={work.status} version={work.version} />}>
              <Space wrap style={{ marginBottom: 16 }}>
                <Tag color="geekblue">第 {currentRound} 轮执行</Tag>
                {currentRoundView ? <Tag>{repairRoundLabel(currentRoundView.status)}</Tag> : null}
              </Space>
              <FactGrid facts={[
                ["优先级", work.priority],
                ["SLA", work.sla_status],
                ["截止时间", formatTimestamp(work.sla_due_at)],
                ["服务窗口", `${formatTimestamp(work.service_window_start)} — ${formatTimestamp(work.service_window_end)}`],
              ]} />
              <Space wrap>
                <Link href="/field">返回我的待办</Link>
                <Link href={`/incidents/${work.incident_id}`}>查看诊断与证据</Link>
              </Space>
            </Card>

            <FieldVoiceGuidancePanel workOrderId={workOrderId} />

            <RemoteExpertCollaborationPanel workOrderId={workOrderId} />

            {activeFieldRediagnosis || fieldRediagnosisCanRequest || fieldRediagnosisResult ? (
              <Card title="现场观察中心复诊">
                <div className="page-stack">
                  {activeFieldRediagnosis ? (
                    <Alert
                      type={fieldRediagnosisStatusType(activeFieldRediagnosis.status)}
                      showIcon
                      message="现场复诊进度"
                      description={(
                        <div className="page-stack">
                          <Space wrap>
                            <Typography.Text code>{activeFieldRediagnosis.diagnosis_run_id}</Typography.Text>
                            <StatusTag
                              status={activeFieldRediagnosis.status}
                              version={activeFieldRediagnosis.version}
                            />
                            <Typography.Text>{activeFieldRediagnosis.status}</Typography.Text>
                            <Tag>{`现场观察 ${activeFieldRediagnosis.field_entry_count} 条`}</Tag>
                            <Link href={`/incidents/${work.incident_id}`}>查看诊断运行</Link>
                          </Space>
                          <Typography.Text type="secondary">
                            {`源诊断 ${activeFieldRediagnosis.source_diagnosis_run_id} · 工单版本 v${activeFieldRediagnosis.source_work_order_version} · 更新于 ${formatTimestamp(activeFieldRediagnosis.updated_at)}`}
                          </Typography.Text>
                          {fieldRediagnosisTerminalDiagnosis?.conclusion ? (
                            <Typography.Text>{fieldRediagnosisTerminalDiagnosis.conclusion}</Typography.Text>
                          ) : null}
                          {fieldRediagnosisTerminalDiagnosis?.next_checks.map((check) => (
                            <Typography.Text key={`${activeFieldRediagnosis.diagnosis_run_id}-${check}`}>
                              {check}
                            </Typography.Text>
                          ))}
                          {isFieldRediagnosisRunning(activeFieldRediagnosis.status) ? (
                            <Typography.Text>页面仅刷新服务端进度，不会自动创建复诊或执行工单动作。</Typography.Text>
                          ) : !["COMPLETED", "NEEDS_INFORMATION"].includes(
                            activeFieldRediagnosis.status,
                          ) ? (
                            <Typography.Text>需要人工处理</Typography.Text>
                          ) : null}
                        </div>
                      )}
                    />
                  ) : null}
                  {fieldRediagnosisError ? (
                    <Alert
                      type="error"
                      showIcon
                      message={fieldRediagnosisError instanceof Error
                        ? fieldRediagnosisError.message
                        : "中心复诊当前不可用"}
                    />
                  ) : null}
                  {fieldRediagnosisCanRequest ? (
                    <>
                      <Alert
                        type="info"
                        showIcon
                        message="复诊必须由现场工程师明确发起"
                        description="这里只允许选择已经同步的 NOTE 或 AI_OBSERVATION。提交时会重新校验当前负责人、工单版本、源诊断和观察摘要；追加或同步现场事实本身不会自动触发复诊。"
                      />
                      {fieldRediagnosisEntries.length === 0 ? (
                        <EmptyState description="尚无可用于复诊的现场说明或人工复核观察" />
                      ) : (
                        <Checkbox.Group
                          value={fieldRediagnosisSelection}
                          onChange={(values) => setFieldRediagnosisSelection(values.map(String))}
                        >
                          <Space direction="vertical">
                            {fieldRediagnosisEntries.map((entry) => (
                              <Checkbox
                                key={entry.entry_id}
                                value={entry.entry_id}
                                aria-label={`选择现场观察 #${entry.sequence} ${entry.entry_type}`}
                              >
                                <Space wrap>
                                  <Tag>{entry.entry_type}</Tag>
                                  <Typography.Text>{`#${entry.sequence} ${entrySummary(entry)}`}</Typography.Text>
                                </Space>
                              </Checkbox>
                            ))}
                          </Space>
                        </Checkbox.Group>
                      )}
                      <Button
                        type="primary"
                        loading={fieldRediagnosisBusy}
                        disabled={!online || fieldRediagnosisSelection.length === 0}
                        onClick={() => void requestFieldRediagnosis()}
                      >请求中心复诊</Button>
                    </>
                  ) : null}
                  {fieldRediagnosisResult
                    && fieldRediagnosisResult.diagnosis_run_id
                      !== activeFieldRediagnosis?.diagnosis_run_id ? (
                    <Alert
                      type="success"
                      showIcon
                      message="新的中心诊断运行已创建"
                      description={(
                        <Space wrap>
                          <Typography.Text code>{fieldRediagnosisResult.diagnosis_run_id}</Typography.Text>
                          <StatusTag
                            status={fieldRediagnosisResult.status}
                            version={fieldRediagnosisResult.version}
                          />
                          <Link href={`/incidents/${work.incident_id}`}>查看新诊断运行</Link>
                        </Space>
                      )}
                    />
                  ) : null}
                  {fieldRediagnosisCanFeedback && activeFieldRediagnosis ? (
                    <DiagnosisFeedbackPanel
                      diagnosisRunId={activeFieldRediagnosis.diagnosis_run_id}
                    />
                  ) : null}
                </div>
              </Card>
            ) : null}

            <Card
              title="受治理的离线工作包"
              extra={offlinePack ? <StatusTag status={offlinePack.status} version={offlinePack.version} /> : null}
            >
              <div className="page-stack">
                <Alert
                  type="warning"
                  showIcon
                  message="只读离线资料，不授予任何服务端写权限"
                  description="工作包仅用于断网时查看已签发的最小工单、设备、诊断和引用快照；状态迁移、备件、同步与完工仍须联网并由服务端重新授权。"
                />
                {offlinePack ? (
                  <>
                    <FactGrid facts={[
                      ["工作包 ID", offlinePack.pack_id],
                      ["工单签发版本", String(offlinePack.work_order_version)],
                      ["签发时间", formatTimestamp(offlinePack.issued_at)],
                      ["到期时间", formatTimestamp(offlinePack.expires_at)],
                      ["内容摘要", offlinePack.content_hash],
                    ]} />
                    <Space wrap>
                      <Button loading={busy} onClick={() => void downloadOfflinePack()}>刷新离线工作包</Button>
                      <Button loading={busy} onClick={() => void removeOfflinePack()}>仅删除本机副本</Button>
                      {offlinePack.legal_actions.includes("REVOKE") ? (
                        <Button danger loading={busy} onClick={() => void revokeOfflinePack()}>撤销工作包</Button>
                      ) : null}
                    </Space>
                  </>
                ) : (
                  <Button type="primary" loading={busy} onClick={() => void downloadOfflinePack()}>
                    下载离线工作包
                  </Button>
                )}
              </div>
            </Card>

            <Card title="受治理的现场边缘诊断">
              <div className="page-stack">
                <Alert
                  type="warning"
                  showIcon
                  message="无设备证明，仅供人工复核"
                  description="本地模型只生成关闭式候选；导入时服务端会重新校验权限、工单、Release、Prompt、知识版本、引用和安全策略。接受或拒绝都不会自动形成现场事实、正式诊断、审批、完工、备件或设备动作。"
                />
                {edgeError ? (
                  <Alert
                    type="warning"
                    showIcon
                    message={edgeError instanceof Error && edgeError.message === "结果文件不得超过 64 KiB"
                      ? edgeError.message
                      : "边缘诊断当前不可用"}
                    description={edgeError instanceof Error && edgeError.message === "结果文件不得超过 64 KiB"
                      ? "文件已在浏览器本地拦截，未发送到服务端。"
                      : edgeError instanceof Error
                        ? edgeError.message
                        : "请检查服务端返回的 request_id 后重试；既有现场工作不受影响。"}
                  />
                ) : null}
                <Space wrap>
                  {work.legal_actions.includes("CREATE_EDGE_DIAGNOSIS_PACK") ? (
                    <Button
                      type="primary"
                      loading={edgeBusy}
                      onClick={() => void issueEdgeDiagnosisPack()}
                    >签发并下载边缘诊断包</Button>
                  ) : null}
                  {work.legal_actions.includes("CREATE_EDGE_DIAGNOSIS_PACK") ? (
                    <label>
                      <Typography.Text>导入边缘诊断结果</Typography.Text>
                      <input
                        aria-label="导入边缘诊断结果"
                        type="file"
                        accept="application/json,.json"
                        disabled={edgeBusy}
                        onChange={(event) => {
                          const file = event.target.files?.[0];
                          event.target.value = "";
                          if (file) void importEdgeDiagnosisResult(file);
                        }}
                      />
                    </label>
                  ) : null}
                </Space>
                {edgeRequestId ? <Typography.Text type="secondary">request_id: {edgeRequestId}</Typography.Text> : null}
                {edgePack ? (
                  <>
                    <FactGrid facts={[
                      ["边缘包 ID", edgePack.pack_id],
                      ["到期时间", formatTimestamp(edgePack.expires_at)],
                      ["Release", edgePack.release.release_id],
                      ["Manifest", edgePack.release.manifest_hash],
                      ["GGUF", `${edgePack.release.model_file} · ${edgePack.release.model_content_hash}`],
                      ["Prompt Bundle", `${edgePack.prompt_bundle.prompt_bundle_id} · ${edgePack.prompt_bundle.content_hash}`],
                      ["知识发布", `${edgePack.retrieval.index_release_id} · ${edgePack.retrieval.content_checksum}`],
                      ["可信公钥 key_id", edgePack.key_id],
                    ]} />
                    <Typography.Paragraph type="secondary">
                      离线设备运行：python -m industrial_ops_agent.edge.cli run；可信公钥、GGUF 与 llama-cli 必须由企业部署流程预置。
                    </Typography.Paragraph>
                  </>
                ) : null}
                <Input
                  aria-label="边缘候选复核理由"
                  value={edgeReviewReason}
                  maxLength={1000}
                  onChange={(event) => setEdgeReviewReason(event.target.value)}
                  placeholder="接受或拒绝理由"
                />
                {edgeCandidates.length === 0 ? (
                  <EmptyState description="暂无已导入的边缘诊断候选" />
                ) : (
                  <List
                    dataSource={edgeCandidates}
                    renderItem={(item) => (
                      <List.Item
                        actions={[
                          ...(item.legal_actions.includes("ACCEPT") ? [
                            <Button
                              key={`accept-${item.candidate_id}`}
                              type="primary"
                              loading={edgeBusy}
                              aria-label={`接受边缘候选 ${item.candidate_id}`}
                              onClick={() => void decideEdgeDiagnosis(item, "ACCEPTED")}
                            >接受为人工候选</Button>,
                          ] : []),
                          ...(item.legal_actions.includes("REJECT") ? [
                            <Button
                              key={`reject-${item.candidate_id}`}
                              danger
                              loading={edgeBusy}
                              aria-label={`拒绝边缘候选 ${item.candidate_id}`}
                              onClick={() => void decideEdgeDiagnosis(item, "REJECTED")}
                            >拒绝候选</Button>,
                          ] : []),
                        ]}
                      >
                        <List.Item.Meta
                          title={<Space wrap><Typography.Text strong>{item.candidate.conclusion}</Typography.Text><Tag>{item.status}</Tag><Tag color="gold">{item.runtime_evidence.runtime_attestation}</Tag></Space>}
                          description={(
                            <div className="page-stack">
                              <Typography.Text>{`置信度 ${Math.round(item.candidate.confidence * 100)}% · ${item.runtime_evidence.engine} · ${item.runtime_evidence.model_file}`}</Typography.Text>
                              {item.pack ? (
                                <Typography.Text type="secondary">
                                  {`Release ${item.pack.release_id} · Manifest ${item.pack.manifest_hash} · GGUF ${item.pack.model_file} / ${item.pack.model_content_hash} · key_id ${item.pack.key_id} · 到期 ${formatTimestamp(item.pack.expires_at)}`}
                                </Typography.Text>
                              ) : null}
                              {(item.candidate.possible_causes ?? []).map((value) => <Typography.Text key={`cause-${item.candidate_id}-${value}`}>候选原因：{value}</Typography.Text>)}
                              {(item.candidate.next_checks ?? []).map((value) => <Typography.Text key={`check-${item.candidate_id}-${value}`}>下一步检查：{value}</Typography.Text>)}
                              {(item.candidate.missing_information ?? []).map((value) => <Typography.Text key={`missing-${item.candidate_id}-${value}`}>缺失信息：{value}</Typography.Text>)}
                              {(item.candidate.safety_warnings ?? []).map((value) => <Typography.Text type="danger" key={`warning-${item.candidate_id}-${value}`}>安全提示：{value}</Typography.Text>)}
                              <Space wrap>{(item.candidate.citation_ids ?? []).map((citationId) => <Tag key={`${item.candidate_id}-${citationId}`}>{citationId}</Tag>)}</Space>
                              {item.review_reason ? <Typography.Text>复核理由：{item.review_reason}</Typography.Text> : null}
                              {item.legal_actions.includes("CREATE_AI_OBSERVATION") ? (
                                <div className="page-stack">
                                  <Alert
                                    type="info"
                                    showIcon
                                    message="创建现场观察是独立人工动作"
                                    description="请填写工程师实际看到或核实的内容。模型候选不会自动复制到事实，也不会因此成为正式诊断或设备证明。"
                                  />
                                  <Input.TextArea
                                    aria-label={`边缘候选 ${item.candidate_id} 的工程师现场观察`}
                                    value={edgeObservationDrafts[item.candidate_id] ?? ""}
                                    maxLength={2000}
                                    rows={2}
                                    onChange={(event) => setEdgeObservationDrafts((current) => ({
                                      ...current,
                                      [item.candidate_id]: event.target.value,
                                    }))}
                                    placeholder="填写现场独立核实的观察，不要复制模型候选"
                                  />
                                  <Button
                                    type="primary"
                                    loading={busy}
                                    disabled={!(edgeObservationDrafts[item.candidate_id]?.trim())}
                                    aria-label={`加入边缘 AI 观察队列 ${item.candidate_id}`}
                                    onClick={() => void queueEdgeDiagnosisObservation(item)}
                                  >加入边缘 AI 观察队列</Button>
                                </div>
                              ) : null}
                            </div>
                          )}
                        />
                      </List.Item>
                    )}
                  />
                )}
              </div>
            </Card>

            {currentRoundView?.status === "REWORK_IN_PROGRESS" ? (
              <Alert
                type="warning"
                showIcon
                message={`第 ${currentRound} 轮返工：${currentRoundView.rework_reason ?? "独立验收未通过"}`}
                description={`上一轮事实已封存。本轮必须重新记录完成步骤、现场证据和客户签字；只有 #${entryCheckpoint + 1} 之后的新记录会进入本轮完工。`}
              />
            ) : null}

            {work.creation_mode === "SERVICE_AUTHORIZATION" ? (
              <Alert type="info" showIcon message="本工单无需初始备件" description="现场步骤、证据、签字、费用与独立验收仍需完整记录；页面不会请求或展示 WMS 出库、用料、核销和退料动作。" />
            ) : null}

            {partAllocation ? (
              <Card title="权威备件分配与 WMS 出库" extra={<Space><Tag>{partAllocation.status}</Tag><Tag color={partAllocation.issue_status === "ISSUED" ? "green" : "gold"}>{partAllocation.issue_status}</Tag></Space>}>
                <FactGrid facts={[
                  ["料号", partAllocation.part_number],
                  ["预留 ID", partAllocation.reservation_id],
                  ["批准数量", String(partAllocation.approved_quantity)],
                  ["来源", partAllocation.source],
                  ["源记录", partAllocation.source_record_id],
                  ["事实时间", formatTimestamp(partAllocation.as_of)],
                ]} />
                <PartIssueState allocation={partAllocation} />
                {partIssueProposal ? (
                  <Alert
                    type="success"
                    showIcon
                    message={`出库提案已创建，等待独立审批：${partIssueProposal.approval_id}`}
                    description={`操作号 ${partIssueProposal.operation_id}；提案创建阶段未调用 WMS。`}
                  />
                ) : null}
                {partAllocation.legal_actions.includes("PROPOSE_ISSUE") && !partIssueProposal ? (
                  <Button
                    type="primary"
                    loading={busy}
                    onClick={() => void proposePartIssue()}
                  >发起 WMS 出库审批</Button>
                ) : null}
              </Card>
            ) : partAllocationError ? (
              <Alert
                type="warning"
                showIcon
                message="备件分配不可用"
                description="当前无法取得该工单的权威备件预留，用料入口已禁用；步骤、证据、说明和签字仍可继续记录。"
              />
            ) : null}

            {partAccounting ? (
              <Card
                title="权威备件核销"
                extra={<Tag color={partAccounting.close_ready ? "green" : "gold"}>{partAccounting.mode}</Tag>}
              >
                <div className="page-stack">
                  <Space wrap>
                    <Tag>已出库 {partAccounting.issued_quantity} 件</Tag>
                    <Tag>已记录现场用量 {partAccounting.recorded_usage_quantity} 件</Tag>
                    <Tag color="green">已消耗 {partAccounting.consumed_quantity} 件</Tag>
                    <Tag color="blue">已退料 {partAccounting.returned_quantity} 件</Tag>
                    <Tag color="orange">待核销 {partAccounting.unaccounted_quantity} 件</Tag>
                  </Space>
                  <FactGrid facts={[
                    ["料号", partAccounting.part_number],
                    ["出库事实", partAccounting.part_issue_id],
                    ["待审批/对账消耗", `${partAccounting.active_consumption_quantity} 件`],
                    ["待审批/对账退料", `${partAccounting.active_return_quantity} 件`],
                    ["当前可退", `${partAccounting.returnable_quantity} 件`],
                  ]} />
                  <Alert
                    type={partAccounting.close_ready ? "success" : "warning"}
                    showIcon
                    message={partAccounting.close_ready ? "关单核销已就绪" : "关单核销未就绪"}
                    description={partAccounting.close_ready
                      ? "全部现场用料已由 WMS 确认消耗，且消耗加退料等于出库数量。"
                      : partAccounting.blocking_reasons.map(partAccountingBlockingReason).join("；")}
                  />
                  {partMovementProposal ? (
                    <Alert
                      type="success"
                      showIcon
                      message={`核销提案已创建，等待独立审批：${partMovementProposal.approval_id}`}
                      description={`操作号 ${partMovementProposal.operation_id}；提案创建阶段未调用 WMS。`}
                    />
                  ) : null}
                  {partAccounting.mode === "LEGACY_ACCOUNTING_COMPATIBLE" ? (
                    <Alert
                      type="warning"
                      showIcon
                      message="存量工单保持核销兼容模式"
                      description="平台不会把历史现场记录推断成 WMS 消耗或退料，也不会补造物料动作。"
                    />
                  ) : (
                    <>
                      <Card size="small" title="待确认实际消耗的现场用料">
                        {partAccounting.eligible_entries.length === 0 ? (
                          <EmptyState description="当前没有可提交的 PART 现场记录" />
                        ) : (
                          <List
                            dataSource={partAccounting.eligible_entries}
                            renderItem={(entry) => (
                              <List.Item
                                actions={[
                                  <Button
                                    key={entry.entry_id}
                                    type="primary"
                                    loading={partMovementBusyId === `CONSUME:${entry.entry_id}`}
                                    disabled={!partAccounting.legal_actions.includes("PROPOSE_CONSUMPTION")}
                                    onClick={() => void proposePartConsumption(entry.entry_id)}
                                  >提交 {entry.quantity} 件实际消耗审批</Button>,
                                ]}
                              >
                                <List.Item.Meta
                                  title={entry.entry_id}
                                  description={`${formatTimestamp(entry.occurred_at)} · 数量由该条不可变现场记录固定`}
                                />
                              </List.Item>
                            )}
                          />
                        )}
                      </Card>
                      {partAccounting.legal_actions.includes("PROPOSE_RETURN") ? (
                        <Space.Compact block>
                          <Input
                            aria-label="退料数量"
                            type="number"
                            min={1}
                            max={partAccounting.returnable_quantity}
                            value={returnQuantity}
                            onChange={(event) => setReturnQuantity(event.target.value)}
                          />
                          <Button
                            loading={partMovementBusyId === `RETURN:${returnQuantity}`}
                            disabled={!validReturnQuantity(returnQuantity, partAccounting.returnable_quantity)}
                            onClick={() => void proposePartReturn()}
                          >提交退料审批</Button>
                        </Space.Compact>
                      ) : null}
                    </>
                  )}
                  <Card size="small" title="消耗与退料动作">
                    {partAccounting.movements.length === 0 ? (
                      <EmptyState description="尚无核销动作" />
                    ) : (
                      <List
                        dataSource={partAccounting.movements}
                        renderItem={(movement) => (
                          <List.Item>
                            <List.Item.Meta
                              title={<Space><Tag>{partMovementKindLabel(movement.movement_kind)}</Tag><StatusTag status={movement.status} /></Space>}
                              description={`${movement.movement_id} · ${movement.quantity} 件 · ${movement.operation_id}`}
                            />
                            {movement.reconciliation_id ? <Link href="/reconciliations">进入对账中心</Link> : null}
                          </List.Item>
                        )}
                      />
                    )}
                  </Card>
                </div>
              </Card>
            ) : partAccountingError && partAllocation?.issue_status === "ISSUED" ? (
              <Alert
                type="warning"
                showIcon
                message="物料核销账本不可用"
                description="当前无法取得服务端核销余额，实际消耗、退料和关单就绪入口保持关闭。"
              />
            ) : null}

            <Card title="工单状态操作">
              <Space wrap>
                {work.legal_actions.includes("ACCEPT") ? <Button type="primary" loading={busy} onClick={() => void transition((current) => acceptWorkOrder(workOrderId, current.version))}>接单</Button> : null}
                {work.legal_actions.includes("START") ? <Button type="primary" loading={busy} onClick={() => void transition((current) => startWorkOrder(workOrderId, current.version))}>确认安全条件并开始</Button> : null}
                {work.legal_actions.includes("HOLD") ? <Button loading={busy} onClick={() => void transition((current) => holdWorkOrder(workOrderId, current.version, { reason: holdReason, recovery_condition: holdReason }))}>暂停</Button> : null}
                {work.legal_actions.includes("RESUME") ? <Button type="primary" loading={busy} onClick={() => void transition((current) => resumeWorkOrder(workOrderId, current.version, holdReason))}>恢复</Button> : null}
                {(work.legal_actions.includes("HOLD") || work.legal_actions.includes("RESUME")) ? <Input style={{ minWidth: 300 }} value={holdReason} onChange={(event) => setHoldReason(event.target.value)} placeholder="暂停原因或恢复说明" /> : null}
              </Space>
            </Card>

            <Card
              title="受治理的现场证据"
              extra={<Tag color={evidenceDrafts.length > 0 ? "gold" : "default"}>本机待上传 {evidenceDrafts.length} 项</Tag>}
            >
              <div className="page-stack">
                <Alert
                  type="info"
                  showIcon
                  message="本机草稿不等于平台证据"
                  description="照片或视频先短期保存到浏览器 IndexedDB；恢复联网并重新加载当前工单后上传隔离区。扫描中的项目每 2 秒自动刷新，只有扫描为 CLEAN 的项目才能显式关联现场事实。"
                />
                {evidenceError ? (
                  <Alert
                    type="warning"
                    showIcon
                    message="现场证据暂不可用"
                    description={evidenceErrorMessage(evidenceError)}
                  />
                ) : null}
                <Space wrap>
                  <input
                    aria-label="选择现场照片或视频"
                    type="file"
                    accept="image/png,image/jpeg,video/mp4,video/webm"
                    disabled={busy || !authoritativeWorkLoaded}
                    onChange={(event) => {
                      const file = event.currentTarget.files?.[0];
                      event.currentTarget.value = "";
                      if (file) void captureEvidenceDraft(file);
                    }}
                  />
                  <Button
                    aria-label="同步本机证据"
                    type="primary"
                    loading={busy}
                    disabled={!navigator.onLine || !authoritativeWorkLoaded || evidenceDrafts.length === 0}
                    onClick={() => void syncEvidenceDrafts()}
                  >同步本机证据</Button>
                  <Button
                    disabled={!navigator.onLine || !authoritativeWorkLoaded}
                    onClick={() => void load()}
                  >刷新扫描状态</Button>
                </Space>
                <Card size="small" title="本机未上传草稿">
                  <List
                    dataSource={evidenceDrafts}
                    locale={{ emptyText: "没有本机证据草稿" }}
                    renderItem={(draft) => (
                      <List.Item
                        actions={[
                          <Button
                            key={draft.client_operation_id}
                            danger
                            disabled={busy}
                            onClick={() => void deleteEvidenceDraft(draft)}
                          >删除本机草稿</Button>,
                        ]}
                      >
                        <List.Item.Meta
                          title={`${draft.declared_mime} · ${formatBytes(draft.size_bytes)}`}
                          description={`${formatTimestamp(draft.created_at)} · ${draft.content_hash}`}
                        />
                      </List.Item>
                    )}
                  />
                </Card>
                {entryActionFeedback?.scope === "evidence" ? (
                  <Alert
                    type="success"
                    showIcon
                    message={entryActionFeedback.result === "synced"
                      ? entryActionFeedback.message
                      : `${entryActionFeedback.message}，等待同步`}
                    description={entryActionFeedback.result === "synced"
                      ? "已同步到现场事实时间线，页面状态已刷新。"
                      : "已保存到本机待同步队列；恢复联网后可使用页面同步操作。"}
                  />
                ) : null}
                {entryActionError?.scope === "evidence" ? (
                  <Alert
                    type="error"
                    showIcon
                    message="关联现场事实未完成"
                    description={fieldEntryActionErrorMessage(entryActionError.error)}
                  />
                ) : null}
                <Card size="small" title="服务端隔离与扫描状态">
                  <List
                    dataSource={evidenceUploads}
                    locale={{ emptyText: "尚无服务端现场媒体" }}
                    renderItem={(upload) => {
                      const actions: ReactNode[] = [];
                      const attachActionKey = `evidence:${upload.evidence_upload_id}`;
                      const evidenceAttached = attachedEvidenceIds.has(upload.evidence_upload_id);
                      if (evidenceAttached) {
                        actions.push(
                          <Button
                            aria-label="已关联现场事实"
                            key={`attached-${upload.evidence_upload_id}`}
                            disabled
                          >已关联现场事实</Button>,
                        );
                      } else if (upload.ready_to_attach && upload.legal_actions.includes("ATTACH")) {
                        actions.push(
                          <Button
                            aria-label="关联现场事实"
                            key={`attach-${upload.evidence_upload_id}`}
                            type="primary"
                            loading={entryActionBusyKey === attachActionKey}
                            disabled={busy || Boolean(entryActionBusyKey) || !authoritativeWorkLoaded}
                            onClick={() => void runFieldEntryAction(
                              attachActionKey,
                              "evidence",
                              "现场证据已关联",
                              {
                                entry_type: "EVIDENCE",
                                evidence_id: upload.evidence_upload_id,
                              },
                              () => setLocallyAttachedEvidenceIds((current) => (
                                current.includes(upload.evidence_upload_id)
                                  ? current
                                  : [...current, upload.evidence_upload_id]
                              )),
                            )}
                          >关联现场事实</Button>,
                        );
                      }
                      if (upload.legal_actions.includes("RECOGNIZE")) {
                        if (isFieldVideoUpload(upload)) {
                          actions.push(
                            <OnlineGuardButton
                              aria-label="开始视频识别"
                              key={`recognize-video-${upload.evidence_upload_id}`}
                              loading={fieldRecognitionBusyId === upload.evidence_upload_id}
                              disabled={!authoritativeWorkLoaded}
                              onClick={() => void startRecognition(upload, false)}
                            >开始视频识别</OnlineGuardButton>,
                            <OnlineGuardButton
                              aria-label="开始时序分析"
                              key={`recognize-temporal-${upload.evidence_upload_id}`}
                              loading={fieldRecognitionBusyId === upload.evidence_upload_id}
                              disabled={!authoritativeWorkLoaded}
                              onClick={() => void startRecognition(upload, true)}
                            >开始时序分析</OnlineGuardButton>,
                          );
                        } else {
                          actions.push(
                            <OnlineGuardButton
                              aria-label="开始识别"
                              key={`recognize-${upload.evidence_upload_id}`}
                              loading={fieldRecognitionBusyId === upload.evidence_upload_id}
                              disabled={!authoritativeWorkLoaded}
                              onClick={() => void startRecognition(upload)}
                            >开始识别</OnlineGuardButton>,
                          );
                        }
                      }
                      if (upload.recognition_bundle_id) {
                        actions.push(
                          <Button
                            aria-label="查看识别结果"
                            key={`review-${upload.evidence_upload_id}`}
                            loading={fieldRecognitionBusyId === upload.evidence_upload_id}
                            disabled={!navigator.onLine || !authoritativeWorkLoaded}
                            onClick={() => void openRecognition(upload)}
                          >查看识别结果</Button>,
                        );
                      }
                      return (
                        <List.Item actions={actions.length > 0 ? actions : undefined}>
                          <List.Item.Meta
                            title={<Space><Tag color={evidenceScanColor(upload.scan_state)}>{evidenceScanLabel(upload.scan_state)}</Tag><span>{upload.evidence_upload_id}</span>{upload.recognition_status ? <StatusTag status={upload.recognition_status} /> : null}</Space>}
                            description={<Space direction="vertical" size={0}>
                              <span>{upload.declared_mime} · {formatBytes(upload.size_bytes)} · 扫描版本 {upload.scan_version}</span>
                              {upload.recognition_bundle_status ? <span>识别包 {upload.recognition_bundle_status} · v{upload.recognition_bundle_version}</span> : null}
                              {upload.recognition_failure_summary ? <span>失败摘要：{upload.recognition_failure_summary}</span> : null}
                            </Space>}
                          />
                        </List.Item>
                      );
                    }}
                  />
                </Card>
                {evidenceUploads.map((upload) => {
                  const recognition = fieldRecognitionEvidence[upload.evidence_upload_id];
                  if (!recognition) return null;
                  return (
                    <FieldRecognitionReview
                      key={upload.evidence_upload_id}
                      upload={upload}
                      evidence={recognition}
                      busy={busy || fieldRecognitionBusyId === upload.evidence_upload_id}
                      observationRecorded={(entries ?? []).some((entry) => (
                        entry.entry_type === "AI_OBSERVATION"
                        && stringPayloadValue(entry.payload, "recognition_bundle_id") === recognition.bundle_id
                      ))}
                      onConfirm={(
                        corrections,
                        dispositions,
                        ocrBlockDecisions,
                        transcriptDecisions,
                        videoEventDispositions,
                        qrCodeDecisions,
                      ) => void confirmRecognition(
                        upload,
                        recognition,
                        corrections,
                        dispositions,
                        ocrBlockDecisions,
                        transcriptDecisions,
                        videoEventDispositions,
                        qrCodeDecisions,
                      )}
                      onQueueObservation={(observation, sourceItems) => append({
                        entry_type: "AI_OBSERVATION",
                        observation,
                        field_evidence_upload_id: upload.evidence_upload_id,
                        recognition_bundle_id: recognition.bundle_id,
                        recognition_bundle_version: recognition.version,
                        source_items: sourceItems,
                      })}
                    />
                  );
                })}
              </div>
            </Card>

            {["ACCEPTED", "IN_PROGRESS", "ON_HOLD"].includes(work.status) ? <Card title="持续追加现场事实">
              {entryActionFeedback?.scope === "facts" ? (
                <Alert
                  type="success"
                  showIcon
                  message={entryActionFeedback.result === "synced"
                    ? entryActionFeedback.message
                    : `${entryActionFeedback.message}，等待同步`}
                  description={entryActionFeedback.result === "synced"
                    ? "已同步到现场事实时间线，提交内容已清空。"
                    : "已保存到本机待同步队列，提交内容已清空；恢复联网后可使用页面同步操作。"}
                  style={{ marginBottom: 16 }}
                />
              ) : null}
              {entryActionError?.scope === "facts" ? (
                <Alert
                  type="error"
                  showIcon
                  message="现场事实记录未完成"
                  description={fieldEntryActionErrorMessage(entryActionError.error)}
                  style={{ marginBottom: 16 }}
                />
              ) : null}
              <div className="form-grid">
                <Space.Compact block>
                  <Input aria-label="现场步骤" value={stepDescription} onChange={(event) => setStepDescription(event.target.value)} placeholder="已执行步骤，例如：更换入口滤芯并复测压差" />
                  <Button
                    loading={entryActionBusyKey === "step"}
                    disabled={busy || Boolean(entryActionBusyKey) || !stepDescription.trim()}
                    onClick={() => void runFieldEntryAction(
                      "step",
                      "facts",
                      "现场步骤已记录",
                      { entry_type: "STEP", description: stepDescription.trim(), outcome: "COMPLETED" },
                      () => setStepDescription(""),
                    )}
                  >记录步骤</Button>
                </Space.Compact>
                <Space.Compact block>
                  {partAllocation?.usage_enabled ? (
                    <>
                      <Input
                        aria-label="本次记录数量"
                        value={quantity}
                        onChange={(event) => setQuantity(event.target.value)}
                        type="number"
                        min={1}
                        max={partAllocation.approved_quantity}
                        style={{ maxWidth: 160 }}
                      />
                      <Button
                        loading={busy}
                        disabled={!partQuantityValid}
                        onClick={() => void append({
                          entry_type: "PART",
                          part_reservation_id: partAllocation.reservation_id,
                          part_number: partAllocation.part_number,
                          quantity: partQuantity,
                        })}
                      >记录已批准备件用料</Button>
                    </>
                  ) : null}
                </Space.Compact>
                <Space.Compact block>
                  <Input aria-label="现场补充说明" value={note} onChange={(event) => setNote(event.target.value)} placeholder="现场补充说明" />
                  <Button
                    loading={entryActionBusyKey === "note"}
                    disabled={busy || Boolean(entryActionBusyKey) || !note.trim()}
                    onClick={() => void runFieldEntryAction(
                      "note",
                      "facts",
                      "补充说明已记录",
                      { entry_type: "NOTE", description: note.trim() },
                      () => setNote(""),
                    )}
                  >追加说明</Button>
                </Space.Compact>
                <Input aria-label="客户签字人" value={signedBy} onChange={(event) => setSignedBy(event.target.value)} addonBefore="客户签字人" />
                <Input aria-label="客户确认内容" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} addonBefore="确认内容" />
                <Button
                  loading={entryActionBusyKey === "signature"}
                  disabled={busy || Boolean(entryActionBusyKey) || !signedBy.trim() || !confirmation.trim()}
                  onClick={() => void runFieldEntryAction(
                    "signature",
                    "facts",
                    "客户签字确认已记录",
                    {
                      entry_type: "SIGNATURE",
                      signed_by: signedBy.trim(),
                      signature_role: "CUSTOMER",
                      confirmation_text: confirmation.trim(),
                    },
                    () => {
                      setSignedBy("");
                      setConfirmation(DEFAULT_CUSTOMER_CONFIRMATION);
                    },
                  )}
                >记录客户签字确认</Button>
              </div>
            </Card> : <Alert type="info" showIcon message="接单后才能追加现场执行事实" />}

            <Card title="现场事实时间线" extra={<Tag>{entries?.length ?? 0} 条已同步</Tag>}>
              {entries?.length === 0 ? <EmptyState description="尚未记录现场步骤" /> : null}
              <List
                dataSource={entries}
                renderItem={(entry) => <FieldEntryTimelineItem
                  entry={entry}
                  entryCheckpoint={entryCheckpoint}
                />}
              />
            </Card>

            {work.legal_actions.includes("COMPLETE") ? (
              <Card title="提交现场完工">
                <Alert type="info" showIcon message={`第 ${currentRound} 轮新事实：完成步骤 ${completionFacts.steps}、现场证据 ${completionFacts.evidence}、客户签字 ${completionFacts.signatures}`} />
                <div className="form-grid" style={{ marginTop: 16 }}>
                  <Input.TextArea rows={3} value={rootCause} onChange={(event) => setRootCause(event.target.value)} placeholder="最终根因" />
                  <Input value={costAmount} onChange={(event) => setCostAmount(event.target.value)} addonBefore="费用" />
                  <Button
                    type="primary"
                    loading={busy}
                    disabled={pendingCount > 0 || !rootCause.trim() || !costAmount.trim() || completionFacts.steps === 0 || completionFacts.evidence === 0 || completionFacts.signatures === 0}
                    onClick={() => void transition((current) => completeFieldWorkOrder(workOrderId, current.version, { root_cause: rootCause.trim(), cost_amount: costAmount.trim() }))}
                  >提交完工并等待独立验收</Button>
                </div>
              </Card>
            ) : null}
          </>
        ) : null}
      </div>
    </AppShell>
  );
}

function subscribeConnectivity(onStoreChange: () => void): () => void {
  window.addEventListener("online", onStoreChange);
  window.addEventListener("offline", onStoreChange);
  return () => {
    window.removeEventListener("online", onStoreChange);
    window.removeEventListener("offline", onStoreChange);
  };
}

function useOnlineState(): boolean {
  return useSyncExternalStore(
    subscribeConnectivity,
    () => navigator.onLine,
    () => true,
  );
}

function isFieldRediagnosisRunning(status: string): boolean {
  return status === "QUEUED" || status === "RUNNING";
}

function fieldRediagnosisStatusType(
  status: string,
): ComponentProps<typeof Alert>["type"] {
  if (status === "COMPLETED") return "success";
  if (status === "FAILED") return "error";
  if (isFieldRediagnosisRunning(status)) return "info";
  return "warning";
}

function OnlineGuardButton({ disabled, ...props }: ComponentProps<typeof Button>) {
  const online = useOnlineState();
  return <Button {...props} disabled={!online || disabled} />;
}

function ConnectivityAlert({
  offlineFallback,
  pendingCount,
  busy,
  onSync,
}: {
  offlineFallback: boolean;
  pendingCount: number;
  busy: boolean;
  onSync: () => void;
}) {
  const online = useOnlineState();
  return (
    <Alert
      type={online ? "success" : "warning"}
      showIcon
      message={offlineFallback ? "离线只读快照" : online ? "在线执行" : "离线记录模式"}
      description={offlineFallback
        ? "当前只展示已验证且未到期的服务端签发快照；所有依赖服务端授权的动作均已关闭。"
        : `本机待同步 ${pendingCount} 条；断网期间新增事实保留原发生时间，联网后幂等同步。`}
      action={(
        <Button
          aria-label="同步"
          loading={busy}
          disabled={!online || pendingCount === 0}
          onClick={onSync}
        >同步</Button>
      )}
    />
  );
}

function FieldRecognitionReview({
  upload,
  evidence,
  busy,
  observationRecorded,
  onConfirm,
  onQueueObservation,
}: {
  upload: FieldEvidenceUpload;
  evidence: EvidenceBundle;
  busy: boolean;
  observationRecorded: boolean;
  onConfirm: (
    corrections: Record<string, string>,
    dispositions: Record<string, "ACCEPTED" | "REJECTED">,
    ocrBlockDecisions: Record<string, FieldOcrBlockDecision>,
    transcriptDecisions: Record<string, FieldTranscriptDecision>,
    videoEventDispositions: Record<string, "ACCEPTED" | "REJECTED">,
    qrCodeDecisions: Record<string, FieldQrCodeDecision>,
  ) => void;
  onQueueObservation: (
    observation: string,
    sourceItems: NonNullable<FieldEntryInput["source_items"]>,
  ) => Promise<FieldEntryAppendResult>;
}) {
  const correctionsRef = useRef<Record<string, string>>(recognitionCorrections(evidence));
  const [correctionsComplete, setCorrectionsComplete] = useState(() => (
    fieldRecognitionCorrectionsComplete(evidence, correctionsRef.current)
  ));
  const [dispositions, setDispositions] = useState<Record<string, "ACCEPTED" | "REJECTED">>(() => (
    recognitionDispositions(evidence)
  ));
  const [ocrBlockDecisions, setOcrBlockDecisions] = useState<
    Record<string, FieldOcrBlockDecision>
  >(() => recognitionOcrBlockDecisions(evidence));
  const [transcriptDecisions, setTranscriptDecisions] = useState<
    Record<string, FieldTranscriptDecision>
  >(() => recognitionTranscriptDecisions(evidence));
  const [videoEventDispositions, setVideoEventDispositions] = useState<
    Record<string, "ACCEPTED" | "REJECTED">
  >(() => recognitionVideoEventDispositions(evidence));
  const [qrCodeDecisions, setQrCodeDecisions] = useState<
    Record<string, FieldQrCodeDecision>
  >(() => recognitionQrCodeDecisions(evidence));
  const [observation, setObservation] = useState("");
  const [selectedObservationSources, setSelectedObservationSources] = useState<string[]>([]);
  const observationQueueInFlightRef = useRef(false);
  const [observationQueueBusy, setObservationQueueBusy] = useState(false);
  const [observationQueueResult, setObservationQueueResult] = useState<FieldEntryAppendResult>();
  const [observationQueueError, setObservationQueueError] = useState<string>();

  const reviewComplete = correctionsComplete
    && evidence.visual_findings.every((finding) => (
      unsafeEvidenceSource(evidence, "VISUAL_FINDING", finding.finding_id)
        ? dispositions[finding.finding_id] === "REJECTED"
        : Boolean(dispositions[finding.finding_id])
    ))
    && fieldRecognitionOcrReviewComplete(evidence, ocrBlockDecisions)
    && fieldRecognitionTranscriptReviewComplete(evidence, transcriptDecisions)
    && fieldRecognitionVideoEventReviewComplete(evidence, videoEventDispositions)
    && fieldRecognitionQrReviewComplete(evidence, qrCodeDecisions);
  const isVideo = evidence.source_type === "video";
  const observationSources = governedObservationSources(evidence);
  const selectedSourceItems = observationSources
    .filter((source) => selectedObservationSources.includes(source.key))
    .map(({ source_type, source_id }) => ({ source_type, source_id }));
  async function queueObservation() {
    if (
      observationQueueInFlightRef.current
      || !observation.trim()
      || selectedSourceItems.length === 0
    ) return;
    observationQueueInFlightRef.current = true;
    setObservationQueueBusy(true);
    setObservationQueueResult(undefined);
    setObservationQueueError(undefined);
    try {
      const result = await onQueueObservation(
        observation.trim(),
        selectedSourceItems,
      );
      setObservation("");
      setSelectedObservationSources([]);
      setObservationQueueResult(result);
    } catch (caught) {
      setObservationQueueError(
        caught instanceof Error ? caught.message : "加入 AI 观察队列失败",
      );
    } finally {
      observationQueueInFlightRef.current = false;
      setObservationQueueBusy(false);
    }
  }
  return (
    <Card
      size="small"
      title={`${isVideo ? "视频" : "图片"}识别复核 · ${upload.evidence_upload_id}`}
      extra={<StatusTag status={evidence.status} version={evidence.version} />}
    >
      <div className="page-stack">
        <Alert
          type="info"
          showIcon
          message="模型输出是待复核的派生候选"
          description="确认不会自动关联原始证据、创建现场事实、完成工单或执行设备控制。"
        />
        {(evidence.security_findings ?? []).length ? (
          <Alert
            type="warning"
            showIcon
            message="内容安全发现"
            description={(
              <Space wrap>
                <Tag color="red">{`策略 ${evidence.security_policy_version ?? "unknown"}`}</Tag>
                {(evidence.security_findings ?? []).map((finding) => (
                  <Tag
                    color="orange"
                    key={`${finding.source_type}:${finding.source_id}:${finding.pattern_id}`}
                  >
                    {`${finding.source_type} ${finding.source_id} · ${finding.pattern_id} · ${finding.category}`}
                  </Tag>
                ))}
              </Space>
            )}
          />
        ) : null}
        <Space wrap>
          {Object.entries(evidence.processor_versions)
            .filter(([, release]) => !evidence.visual_findings.some((finding) => finding.model_release_id === release))
            .map(([processor, release]) => (
            <Tag key={processor}>{processor} · {release}</Tag>
            ))}
        </Space>
        {(evidence.qr_codes ?? []).length ? (
          <Card size="small" title="二维码参考证据（必须逐项复核）">
            <Alert
              type="warning"
              showIcon
              message="二维码链接保持不可执行"
              description="这里只显示受权限保护的参考文本；页面不会打开、预览或请求二维码 URI。"
            />
            <List
              dataSource={evidence.qr_codes ?? []}
              renderItem={(candidate) => {
                const decision = qrCodeDecisions[candidate.candidate_id];
                const unsafe = candidate.security_findings.length > 0;
                return (
                  <List.Item
                    actions={[
                      ...(!unsafe ? [
                        <Button
                          aria-label={`接受二维码 ${candidate.candidate_id}`}
                          key={`accept-field-qr-${candidate.candidate_id}`}
                          type={decision?.disposition === "ACCEPTED" ? "primary" : "default"}
                          disabled={evidence.status === "CONFIRMED"}
                          onClick={() => setQrCodeDecisions((current) => ({
                            ...current,
                            [candidate.candidate_id]: {
                              disposition: "ACCEPTED",
                              corrected_text: null,
                            },
                          }))}
                        >接受</Button>,
                      ] : []),
                      <Button
                        aria-label={`拒绝二维码 ${candidate.candidate_id}`}
                        key={`reject-field-qr-${candidate.candidate_id}`}
                        danger
                        type={decision?.disposition === "REJECTED" ? "primary" : "default"}
                        disabled={evidence.status === "CONFIRMED"}
                        onClick={() => setQrCodeDecisions((current) => ({
                          ...current,
                          [candidate.candidate_id]: {
                            disposition: "REJECTED",
                            corrected_text: null,
                          },
                        }))}
                      >拒绝</Button>,
                      <Button
                        aria-label={`安全修正二维码 ${candidate.candidate_id}`}
                        key={`correct-field-qr-${candidate.candidate_id}`}
                        type={decision?.disposition === "CORRECTED" ? "primary" : "default"}
                        disabled={evidence.status === "CONFIRMED"}
                        onClick={() => setQrCodeDecisions((current) => ({
                          ...current,
                          [candidate.candidate_id]: {
                            disposition: "CORRECTED",
                            corrected_text: current[candidate.candidate_id]?.corrected_text ?? "",
                          },
                        }))}
                      >安全修正</Button>,
                    ]}
                  >
                    <div className="page-stack" style={{ width: "100%" }}>
                      <Space wrap>
                        <Tag color="purple">二维码</Tag>
                        <Tag>{candidate.payload_kind}</Tag>
                        <Tag>{evidence.processor_versions.qr ?? "unknown"}</Tag>
                        {candidate.source_frame_id ? <Tag>{`帧 ${candidate.source_frame_id}`}</Tag> : null}
                        {unsafe ? <Tag color="red">内容安全命中</Tag> : <Tag color="green">未命中安全规则</Tag>}
                      </Space>
                      <Typography.Text code>{candidate.text}</Typography.Text>
                      <Typography.Text type="secondary">
                        {`区域 x=${candidate.bbox.x.toFixed(2)}, y=${candidate.bbox.y.toFixed(2)}, w=${candidate.bbox.width.toFixed(2)}, h=${candidate.bbox.height.toFixed(2)}`}
                      </Typography.Text>
                      {candidate.security_findings.map((finding) => (
                        <Typography.Text key={`${candidate.candidate_id}:${finding.pattern_id}`} type="danger">
                          {`${finding.pattern_id} · ${finding.category} · ${finding.policy_version}`}
                        </Typography.Text>
                      ))}
                      {decision?.disposition === "CORRECTED" ? (
                        <Input
                          aria-label={`二维码安全修正文 ${candidate.candidate_id}`}
                          value={decision.corrected_text ?? ""}
                          placeholder="输入人工核实后的安全参考文本"
                          onChange={(event) => setQrCodeDecisions((current) => ({
                            ...current,
                            [candidate.candidate_id]: {
                              disposition: "CORRECTED",
                              corrected_text: event.target.value,
                            },
                          }))}
                        />
                      ) : null}
                    </div>
                  </List.Item>
                );
              }}
            />
          </Card>
        ) : null}
        {isVideo ? (
          <>
            <Card size="small" title="视频关键帧">
              <List
                dataSource={evidence.video_keyframes}
                locale={{ emptyText: "没有视频关键帧" }}
                renderItem={(frame) => {
                  const frameFindings = evidence.visual_findings.filter(
                    (finding) => finding.source_frame_id === frame.frame_id,
                  );
                  const frameOcrBlocks = evidence.ocr_blocks.filter(
                    (block) => block.source_frame_id === frame.frame_id,
                  );
                  const frameQrCodes = (evidence.qr_codes ?? []).filter(
                    (candidate) => candidate.source_frame_id === frame.frame_id,
                  );
                  return (
                  <List.Item>
                    <Space align="start" wrap>
                      <div style={{ position: "relative", width: 240, maxWidth: "100%" }}>
                        <img
                          alt={`关键帧 ${frame.frame_id}`}
                          src={frame.image_url}
                          style={{ display: "block", width: "100%", borderRadius: 8 }}
                        />
                        {frameFindings.map((finding) => (
                          <span
                            key={finding.finding_id}
                            data-testid={`video-finding-overlay-${finding.finding_id}`}
                            aria-label={`关键帧候选区域 ${finding.finding_id}`}
                            title={`${finding.label} · ${formatConfidence(finding.confidence)}`}
                            style={{
                              position: "absolute",
                              left: `${finding.bbox.x * 100}%`,
                              top: `${finding.bbox.y * 100}%`,
                              width: `${finding.bbox.width * 100}%`,
                              height: `${finding.bbox.height * 100}%`,
                              border: `2px solid ${dispositions[finding.finding_id] === "REJECTED" ? "#8c8c8c" : "#fa8c16"}`,
                              boxShadow: "0 0 0 1px rgba(255,255,255,0.75)",
                              pointerEvents: "none",
                            }}
                          />
                        ))}
                        {frameOcrBlocks.map((block) => (
                          <span
                            key={block.block_id}
                            data-testid={`video-ocr-overlay-${block.block_id}`}
                            aria-label={`关键帧 OCR 区域 ${block.block_id}`}
                            title={`${block.text} · ${formatConfidence(block.confidence)}`}
                            style={{
                              position: "absolute",
                              left: `${block.bbox.x * 100}%`,
                              top: `${block.bbox.y * 100}%`,
                              width: `${block.bbox.width * 100}%`,
                              height: `${block.bbox.height * 100}%`,
                              border: "1px dashed #1677ff",
                              pointerEvents: "none",
                            }}
                          />
                        ))}
                        {frameQrCodes.map((candidate) => (
                          <span
                            key={candidate.candidate_id}
                            data-testid={`video-qr-overlay-${candidate.candidate_id}`}
                            aria-label={`关键帧二维码区域 ${candidate.candidate_id}`}
                            title={`二维码 · ${candidate.payload_kind}`}
                            style={{
                              position: "absolute",
                              left: `${candidate.bbox.x * 100}%`,
                              top: `${candidate.bbox.y * 100}%`,
                              width: `${candidate.bbox.width * 100}%`,
                              height: `${candidate.bbox.height * 100}%`,
                              border: "2px dotted #722ed1",
                              pointerEvents: "none",
                            }}
                          />
                        ))}
                      </div>
                      <Space direction="vertical" size={0}>
                        <Typography.Text strong>{frame.frame_id}</Typography.Text>
                        <Typography.Text type="secondary">
                          {formatVideoTime(frame.timestamp_ms)} · {frame.sampling_reason}
                        </Typography.Text>
                        <Typography.Text type="secondary" copyable>
                          sha256:{frame.image_sha256}
                        </Typography.Text>
                      </Space>
                    </Space>
                  </List.Item>
                  );
                }}
              />
            </Card>
            <Card size="small" title="多音轨 ASR 转写">
              <List
                dataSource={evidence.asr_segments}
                locale={{ emptyText: "没有音轨转写候选" }}
                renderItem={(segment) => {
                  const decision = transcriptDecisions[segment.segment_id];
                  const requiresTextReview = segment.entity_candidates.some(
                    (candidate) => candidate.requires_confirmation,
                  );
                  return (
                    <List.Item
                      actions={[
                        <Button
                          aria-label={`接受转写 ${segment.segment_id}`}
                          key={`accept-transcript-${segment.segment_id}`}
                          type={decision?.disposition === "ACCEPTED" ? "primary" : "default"}
                          disabled={unsafeEvidenceSource(
                            evidence,
                            "ASR_SEGMENT",
                            segment.segment_id,
                          ) && decision?.disposition === "REJECTED"}
                          onClick={() => setTranscriptDecisions((current) => ({
                            ...current,
                            [segment.segment_id]: {
                              disposition: "ACCEPTED",
                              corrected_text: current[segment.segment_id]?.corrected_text ?? null,
                            },
                          }))}
                        >{unsafeEvidenceSource(evidence, "ASR_SEGMENT", segment.segment_id)
                          ? "安全修正"
                          : "接受"}</Button>,
                        <Button
                          aria-label={`拒绝转写 ${segment.segment_id}`}
                          key={`reject-transcript-${segment.segment_id}`}
                          danger
                          type={decision?.disposition === "REJECTED" ? "primary" : "default"}
                          onClick={() => setTranscriptDecisions((current) => ({
                            ...current,
                            [segment.segment_id]: {
                              disposition: "REJECTED",
                              corrected_text: null,
                            },
                          }))}
                        >拒绝</Button>,
                      ]}
                    >
                      <div style={{ width: "100%" }}>
                        <List.Item.Meta
                          title={segment.text}
                          description={[
                            segment.source_audio_track_id ?? "未标识音轨",
                            `${formatVideoTime(segment.start_ms)}–${formatVideoTime(segment.end_ms)}`,
                            segment.model_release_id,
                            `置信度 ${formatConfidence(segment.confidence)}`,
                          ].join(" · ")}
                        />
                        {segment.entity_candidates.length > 0 ? (
                          <Space wrap>
                            {segment.entity_candidates.map((candidate) => (
                              <Tag
                                color={candidate.requires_confirmation ? "gold" : "default"}
                                key={candidate.entity_id}
                              >
                                {candidate.entity_type} · {candidate.value} · {formatConfidence(candidate.confidence)}
                              </Tag>
                            ))}
                          </Space>
                        ) : null}
                        {decision?.disposition === "ACCEPTED" && (
                          requiresTextReview
                          || unsafeEvidenceSource(evidence, "ASR_SEGMENT", segment.segment_id)
                        ) ? (
                          <Input
                            aria-label={`转写复核文本 ${segment.segment_id}`}
                            value={decision.corrected_text ?? ""}
                            placeholder="核对工业实体后输入完整转写文本"
                            onChange={(event) => setTranscriptDecisions((current) => ({
                              ...current,
                              [segment.segment_id]: {
                                disposition: "ACCEPTED",
                                corrected_text: event.target.value,
                              },
                            }))}
                            style={{ marginTop: 8 }}
                          />
                        ) : null}
                      </div>
                    </List.Item>
                  );
                }}
              />
            </Card>
            <Card size="small" title="视频事件时间线">
              <List
                dataSource={evidence.video_events}
                locale={{ emptyText: "没有视频事件候选" }}
                renderItem={(videoEvent) => {
                  const reviewable = isReviewableVideoEvent(videoEvent.event_type);
                  const disposition = videoEventDispositions[videoEvent.event_id];
                  return (
                    <List.Item
                      actions={reviewable ? [
                        <Button
                          aria-label={`接受视频事件 ${videoEvent.event_id}`}
                          key={`accept-video-event-${videoEvent.event_id}`}
                          type={disposition === "ACCEPTED" ? "primary" : "default"}
                          disabled={unsafeEvidenceSource(
                            evidence,
                            "VIDEO_EVENT",
                            videoEvent.event_id,
                          )}
                          onClick={() => setVideoEventDispositions((current) => ({
                            ...current,
                            [videoEvent.event_id]: "ACCEPTED",
                          }))}
                        >接受</Button>,
                        <Button
                          aria-label={`拒绝视频事件 ${videoEvent.event_id}`}
                          key={`reject-video-event-${videoEvent.event_id}`}
                          danger
                          type={disposition === "REJECTED" ? "primary" : "default"}
                          onClick={() => setVideoEventDispositions((current) => ({
                            ...current,
                            [videoEvent.event_id]: "REJECTED",
                          }))}
                        >拒绝</Button>,
                      ] : undefined}
                    >
                      <List.Item.Meta
                        title={<Space wrap><Tag>{videoEvent.event_type}</Tag><span>{videoEvent.description}</span></Space>}
                        description={[
                          `${formatVideoTime(videoEvent.start_ms)}–${formatVideoTime(videoEvent.end_ms)}`,
                          videoEvent.model_release_id ?? "采样元数据",
                          `置信度 ${formatConfidence(videoEvent.confidence)}`,
                        ].join(" · ")}
                      />
                    </List.Item>
                  );
                }}
              />
            </Card>
          </>
        ) : null}
        <Card size="small" title="OCR 文本块">
          <List
            dataSource={evidence.ocr_blocks}
            locale={{ emptyText: "没有 OCR 文本候选" }}
            renderItem={(block) => (
              <List.Item
                actions={[
                  <Button
                    aria-label={`接受 OCR ${block.block_id}`}
                    key={`accept-ocr-${block.block_id}`}
                    type={ocrBlockDecisions[block.block_id]?.disposition === "ACCEPTED" ? "primary" : "default"}
                    disabled={unsafeEvidenceSource(evidence, "OCR_BLOCK", block.block_id)}
                    onClick={() => setOcrBlockDecisions((current) => ({
                      ...current,
                      [block.block_id]: { disposition: "ACCEPTED", corrected_text: null },
                    }))}
                  >接受</Button>,
                  <Button
                    aria-label={`拒绝 OCR ${block.block_id}`}
                    key={`reject-ocr-${block.block_id}`}
                    danger
                    type={ocrBlockDecisions[block.block_id]?.disposition === "REJECTED" ? "primary" : "default"}
                    onClick={() => setOcrBlockDecisions((current) => ({
                      ...current,
                      [block.block_id]: { disposition: "REJECTED", corrected_text: null },
                    }))}
                  >拒绝</Button>,
                  <Button
                    aria-label={`修正 OCR ${block.block_id}`}
                    key={`correct-ocr-${block.block_id}`}
                    type={ocrBlockDecisions[block.block_id]?.disposition === "CORRECTED" ? "primary" : "default"}
                    onClick={() => setOcrBlockDecisions((current) => ({
                      ...current,
                      [block.block_id]: {
                        disposition: "CORRECTED",
                        corrected_text: current[block.block_id]?.corrected_text ?? "",
                      },
                    }))}
                  >修正</Button>,
                ]}
              >
                <div style={{ width: "100%" }}>
                  <List.Item.Meta
                    title={block.text}
                    description={`置信度 ${formatConfidence(block.confidence)} · ${formatBoundingBox(block.bbox)} · 第 ${block.page_number} 页`}
                  />
                  {ocrBlockDecisions[block.block_id]?.disposition === "CORRECTED" ? (
                    <Input
                      aria-label={`OCR 修正文 ${block.block_id}`}
                      value={ocrBlockDecisions[block.block_id]?.corrected_text ?? ""}
                      placeholder={`输入 ${block.block_id} 的复核文本`}
                      onChange={(event) => setOcrBlockDecisions((current) => ({
                        ...current,
                        [block.block_id]: {
                          disposition: "CORRECTED",
                          corrected_text: event.target.value,
                        },
                      }))}
                      style={{ marginTop: 8 }}
                    />
                  ) : null}
                </div>
              </List.Item>
            )}
          />
        </Card>
        <Card size="small" title="关键实体确定性校验">
          <List
            dataSource={evidence.extracted_entities}
            locale={{ emptyText: "没有关键实体候选" }}
            renderItem={(entity) => (
              <List.Item>
                <div style={{ width: "100%" }}>
                  <Space wrap>
                    <Tag>{entity.entity_type}</Tag>
                    <Tag color={entity.validation_status === "VALID" ? "green" : "gold"}>{entity.validation_status}</Tag>
                    <span>{entity.original_value}</span>
                    <span>置信度 {formatConfidence(entity.confidence)}</span>
                  </Space>
                  {entity.validation_status !== "VALID" ? (
                    <Input
                      aria-label={`修正实体 ${entity.entity_id}`}
                      defaultValue={correctionsRef.current[entity.entity_id] ?? ""}
                      placeholder={`确认或修正 ${entity.entity_id}`}
                      onChange={(event) => {
                        correctionsRef.current[entity.entity_id] = event.target.value;
                        const complete = fieldRecognitionCorrectionsComplete(
                          evidence,
                          correctionsRef.current,
                        );
                        setCorrectionsComplete((current) => current === complete ? current : complete);
                      }}
                      style={{ marginTop: 8 }}
                    />
                  ) : null}
                  {entity.validation_reason ? <Typography.Text type="secondary">{entity.validation_reason}</Typography.Text> : null}
                </div>
              </List.Item>
            )}
          />
        </Card>
        <Card size="small" title="VLM 可视发现">
          <List
            dataSource={evidence.visual_findings}
            locale={{ emptyText: "没有 VLM 发现候选" }}
            renderItem={(finding) => (
              <List.Item
                actions={[
                  <Button
                    aria-label={`接受 ${finding.finding_id}`}
                    key={`accept-${finding.finding_id}`}
                    type={dispositions[finding.finding_id] === "ACCEPTED" ? "primary" : "default"}
                    disabled={unsafeEvidenceSource(
                      evidence,
                      "VISUAL_FINDING",
                      finding.finding_id,
                    )}
                    onClick={() => setDispositions((current) => ({
                      ...current,
                      [finding.finding_id]: "ACCEPTED",
                    }))}
                  >接受</Button>,
                  <Button
                    aria-label={`拒绝 ${finding.finding_id}`}
                    key={`reject-${finding.finding_id}`}
                    danger
                    type={dispositions[finding.finding_id] === "REJECTED" ? "primary" : "default"}
                    onClick={() => setDispositions((current) => ({
                      ...current,
                      [finding.finding_id]: "REJECTED",
                    }))}
                  >拒绝</Button>,
                ]}
              >
                <List.Item.Meta
                  title={<Space><Tag>{finding.label}</Tag><span>{finding.description}</span></Space>}
                  description={`${finding.model_release_id} · 置信度 ${formatConfidence(finding.confidence)} · ${formatBoundingBox(finding.bbox)}`}
                />
              </List.Item>
            )}
          />
        </Card>
        {evidence.status === "CONFIRMED" ? (
          <>
            <Alert type="success" showIcon message="识别结果已由当前负责人确认" />
            {upload.legal_actions.includes("CREATE_AI_OBSERVATION") ? (
              <Card size="small" title="受治理的 AI 现场观察">
                <div className="page-stack">
                  <Alert
                    type="info"
                    showIcon
                    message="确认识别与创建观察是两个独立动作"
                    description="这里只把上传、识别包版本和所选来源 ID 加入既有离线事实队列；同步时服务端会重新校验负责人、工单状态及人工复核决定，并从持久化结果生成最终文本与摘要。"
                  />
                  {observationQueueResult || observationRecorded ? (
                    <Alert
                      type="success"
                      showIcon
                      message={observationQueueResult === "queued"
                        ? "AI 现场观察已加入本机队列"
                        : "AI 现场观察已记录"}
                      description={observationQueueResult === "queued"
                        ? "当前同步尚未完成；请保持联网，稍后使用页面顶部的同步按钮。"
                        : "可在下方现场事实时间线查看；如需追加新观察，请重新选择来源并填写内容。"}
                    />
                  ) : null}
                  {observationQueueError ? (
                    <Alert
                      type="error"
                      showIcon
                      message="AI 现场观察未加入队列"
                      description={observationQueueError}
                    />
                  ) : null}
                  <List
                    dataSource={observationSources}
                    locale={{ emptyText: "当前确认包没有可引用的已接受或已修正来源" }}
                    renderItem={(source) => (
                      <List.Item>
                        <Checkbox
                          aria-label={`选择观察来源 ${source.source_type} ${source.source_id}`}
                          checked={selectedObservationSources.includes(source.key)}
                          onChange={(event) => setSelectedObservationSources((current) => (
                            event.target.checked
                              ? [...current, source.key]
                              : current.filter((key) => key !== source.key)
                          ))}
                        >
                          <Space wrap>
                            <Tag>{source.source_type}</Tag>
                            <span>{source.source_id}</span>
                            <span>{source.value}</span>
                            <Tag color={source.decision === "CORRECTED" ? "gold" : "green"}>
                              {source.decision}
                            </Tag>
                          </Space>
                        </Checkbox>
                      </List.Item>
                    )}
                  />
                  <Input.TextArea
                    aria-label="工程师现场观察"
                    rows={3}
                    maxLength={2000}
                    value={observation}
                    placeholder="填写工程师基于现场环境、设备状态与已复核候选形成的观察"
                    onChange={(event) => setObservation(event.target.value)}
                  />
                  <Button
                    type="primary"
                    loading={busy || observationQueueBusy}
                    disabled={
                      busy || observationQueueBusy
                      || !observation.trim()
                      || selectedSourceItems.length === 0
                    }
                    onClick={() => void queueObservation()}
                  >加入 AI 观察队列</Button>
                </div>
              </Card>
            ) : null}
          </>
        ) : (
          <Button
            type="primary"
            loading={busy}
            disabled={!upload.legal_actions.includes("CONFIRM_RECOGNITION") || !reviewComplete}
            onClick={() => onConfirm(
              correctionsRef.current,
              dispositions,
              ocrBlockDecisions,
              transcriptDecisions,
              videoEventDispositions,
              qrCodeDecisions,
            )}
          >确认识别结果</Button>
        )}
      </div>
    </Card>
  );
}

type GovernedObservationSource = {
  key: string;
  source_type: NonNullable<FieldEntryInput["source_items"]>[number]["source_type"];
  source_id: string;
  decision: "ACCEPTED" | "CORRECTED";
  value: string;
};

function governedObservationSources(evidence: EvidenceBundle): GovernedObservationSource[] {
  const sources: GovernedObservationSource[] = [];
  for (const block of evidence.ocr_blocks) {
    if (
      unsafeEvidenceSource(evidence, "OCR_BLOCK", block.block_id)
      && block.disposition !== "CORRECTED"
    ) continue;
    if (block.disposition !== "ACCEPTED" && block.disposition !== "CORRECTED") continue;
    sources.push({
      key: `OCR_BLOCK:${block.block_id}`,
      source_type: "OCR_BLOCK",
      source_id: block.block_id,
      decision: block.disposition,
      value: block.disposition === "CORRECTED"
        ? block.corrected_text ?? block.text
        : block.text,
    });
  }
  for (const segment of evidence.asr_segments) {
    if (
      unsafeEvidenceSource(evidence, "ASR_SEGMENT", segment.segment_id)
      && !segment.corrected_text?.trim()
    ) continue;
    if (segment.disposition !== "ACCEPTED") continue;
    sources.push({
      key: `ASR_SEGMENT:${segment.segment_id}`,
      source_type: "ASR_SEGMENT",
      source_id: segment.segment_id,
      decision: "ACCEPTED",
      value: segment.corrected_text?.trim() || segment.text,
    });
  }
  for (const finding of evidence.visual_findings) {
    if (unsafeEvidenceSource(evidence, "VISUAL_FINDING", finding.finding_id)) continue;
    if (finding.disposition !== "ACCEPTED") continue;
    sources.push({
      key: `VISUAL_FINDING:${finding.finding_id}`,
      source_type: "VISUAL_FINDING",
      source_id: finding.finding_id,
      decision: "ACCEPTED",
      value: finding.description,
    });
  }
  for (const videoEvent of evidence.video_events) {
    if (unsafeEvidenceSource(evidence, "VIDEO_EVENT", videoEvent.event_id)) continue;
    if (
      videoEvent.disposition !== "ACCEPTED"
      || !isReviewableVideoEvent(videoEvent.event_type)
    ) continue;
    sources.push({
      key: `VIDEO_EVENT:${videoEvent.event_id}`,
      source_type: "VIDEO_EVENT",
      source_id: videoEvent.event_id,
      decision: "ACCEPTED",
      value: videoEvent.description,
    });
  }
  return sources;
}

function recognitionCorrections(evidence: EvidenceBundle): Record<string, string> {
  return Object.fromEntries(
    evidence.extracted_entities
      .filter((entity) => entity.corrected_value)
      .map((entity) => [entity.entity_id, entity.corrected_value ?? ""]),
  );
}

function recognitionDispositions(
  evidence: EvidenceBundle,
): Record<string, "ACCEPTED" | "REJECTED"> {
  return Object.fromEntries(
    evidence.visual_findings
      .filter((finding) => finding.disposition)
      .map((finding) => [
        finding.finding_id,
        finding.disposition as "ACCEPTED" | "REJECTED",
      ]),
  );
}

function recognitionOcrBlockDecisions(
  evidence: EvidenceBundle,
): Record<string, FieldOcrBlockDecision> {
  return Object.fromEntries(
    evidence.ocr_blocks
      .filter((block) => block.disposition)
      .map((block) => [
        block.block_id,
        {
          disposition: block.disposition as FieldOcrBlockDecision["disposition"],
          corrected_text: block.corrected_text,
        },
      ]),
  );
}

function recognitionTranscriptDecisions(
  evidence: EvidenceBundle,
): Record<string, FieldTranscriptDecision> {
  return Object.fromEntries(
    evidence.asr_segments
      .filter((segment) => segment.disposition)
      .map((segment) => [
        segment.segment_id,
        {
          disposition: segment.disposition as FieldTranscriptDecision["disposition"],
          corrected_text: segment.corrected_text,
        },
      ]),
  );
}

function recognitionVideoEventDispositions(
  evidence: EvidenceBundle,
): Record<string, "ACCEPTED" | "REJECTED"> {
  return Object.fromEntries(
    evidence.video_events
      .filter((videoEvent) => videoEvent.disposition)
      .map((videoEvent) => [
        videoEvent.event_id,
        videoEvent.disposition as "ACCEPTED" | "REJECTED",
      ]),
  );
}

function recognitionQrCodeDecisions(
  evidence: EvidenceBundle,
): Record<string, FieldQrCodeDecision> {
  return Object.fromEntries(
    (evidence.qr_codes ?? [])
      .filter((candidate) => candidate.disposition)
      .map((candidate) => [
        candidate.candidate_id,
        {
          disposition: candidate.disposition as FieldQrCodeDecision["disposition"],
          corrected_text: candidate.corrected_text,
        },
      ]),
  );
}

function fieldRecognitionReviewComplete(
  evidence: EvidenceBundle,
  corrections: Record<string, string>,
  dispositions: Record<string, "ACCEPTED" | "REJECTED">,
  ocrBlockDecisions: Record<string, FieldOcrBlockDecision>,
  transcriptDecisions: Record<string, FieldTranscriptDecision>,
  videoEventDispositions: Record<string, "ACCEPTED" | "REJECTED">,
  qrCodeDecisions: Record<string, FieldQrCodeDecision>,
): boolean {
  return evidence.visual_findings.every((finding) => (
    unsafeEvidenceSource(evidence, "VISUAL_FINDING", finding.finding_id)
      ? dispositions[finding.finding_id] === "REJECTED"
      : Boolean(dispositions[finding.finding_id])
  ))
    && fieldRecognitionCorrectionsComplete(evidence, corrections)
    && fieldRecognitionOcrReviewComplete(evidence, ocrBlockDecisions)
    && fieldRecognitionTranscriptReviewComplete(evidence, transcriptDecisions)
    && fieldRecognitionVideoEventReviewComplete(evidence, videoEventDispositions)
    && fieldRecognitionQrReviewComplete(evidence, qrCodeDecisions);
}

function fieldRecognitionQrReviewComplete(
  evidence: EvidenceBundle,
  decisions: Record<string, FieldQrCodeDecision>,
): boolean {
  return (evidence.qr_codes ?? []).every((candidate) => {
    const decision = decisions[candidate.candidate_id];
    if (!decision) return false;
    if (candidate.security_findings.length > 0 && decision.disposition === "ACCEPTED") {
      return false;
    }
    return decision.disposition !== "CORRECTED" || Boolean(decision.corrected_text?.trim());
  });
}

function fieldRecognitionOcrReviewComplete(
  evidence: EvidenceBundle,
  decisions: Record<string, FieldOcrBlockDecision>,
): boolean {
  return evidence.ocr_blocks.every((block) => {
    const decision = decisions[block.block_id];
    if (unsafeEvidenceSource(evidence, "OCR_BLOCK", block.block_id)) {
      return decision?.disposition === "REJECTED" || (
        decision?.disposition === "CORRECTED"
        && Boolean(decision.corrected_text?.trim())
      );
    }
    return Boolean(decision)
      && (decision.disposition !== "CORRECTED" || Boolean(decision.corrected_text?.trim()));
  });
}

function fieldRecognitionCorrectionsComplete(
  evidence: EvidenceBundle,
  corrections: Record<string, string>,
): boolean {
  return evidence.extracted_entities.every((entity) => (
    entity.validation_status === "VALID"
    || Boolean(corrections[entity.entity_id]?.trim())
  ));
}

function fieldRecognitionTranscriptReviewComplete(
  evidence: EvidenceBundle,
  decisions: Record<string, FieldTranscriptDecision>,
): boolean {
  return evidence.asr_segments.every((segment) => {
    const decision = decisions[segment.segment_id];
    if (!decision) return false;
    if (unsafeEvidenceSource(evidence, "ASR_SEGMENT", segment.segment_id)) {
      return decision.disposition === "REJECTED" || Boolean(
        decision.corrected_text?.trim()
        && decision.corrected_text.trim() !== segment.text.trim(),
      );
    }
    return decision.disposition !== "ACCEPTED"
      || !segment.entity_candidates.some((candidate) => candidate.requires_confirmation)
      || Boolean(decision.corrected_text?.trim());
  });
}

function fieldRecognitionVideoEventReviewComplete(
  evidence: EvidenceBundle,
  dispositions: Record<string, "ACCEPTED" | "REJECTED">,
): boolean {
  return evidence.video_events
    .filter((videoEvent) => isReviewableVideoEvent(videoEvent.event_type))
    .every((videoEvent) => (
      unsafeEvidenceSource(evidence, "VIDEO_EVENT", videoEvent.event_id)
        ? dispositions[videoEvent.event_id] === "REJECTED"
        : Boolean(dispositions[videoEvent.event_id])
    ));
}

function unsafeEvidenceSource(
  evidence: EvidenceBundle,
  sourceType: string,
  sourceId: string,
): boolean {
  return (evidence.security_findings ?? []).some((finding) => (
    finding.source_type === sourceType && finding.source_id === sourceId
  ));
}

function isReviewableVideoEvent(eventType: string): boolean {
  return [
    "visual_candidate",
    "motion_candidate",
    "signal_pattern_candidate",
    "action_sequence_candidate",
  ].includes(eventType);
}

function isFieldVideoUpload(upload: FieldEvidenceUpload): boolean {
  return (upload.detected_mime ?? upload.declared_mime).startsWith("video/");
}

function formatBoundingBox(box: { x: number; y: number; width: number; height: number }): string {
  return `区域 x=${box.x.toFixed(2)}, y=${box.y.toFixed(2)}, w=${box.width.toFixed(2)}, h=${box.height.toFixed(2)}`;
}

function formatVideoTime(milliseconds: number): string {
  const seconds = milliseconds / 1000;
  return `${seconds.toFixed(seconds % 1 === 0 ? 0 : 1)}s`;
}

function formatConfidence(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function OfflineWorkPackView({ pack }: { pack: WorkOrderOfflinePack }) {
  const { snapshot } = pack;
  return (
    <div className="page-stack">
      <Alert
        type="warning"
        showIcon
        message="只读离线资料，不授予任何服务端写权限"
        description="该快照不能发起状态迁移、备件动作、同步、完工、审批或上传；恢复联网后必须重新接受服务端当前授权和版本校验。"
      />
      <Card
        title="受治理的离线工作包"
        extra={<StatusTag status={pack.status} version={pack.version} />}
      >
        <FactGrid facts={[
          ["工作包 ID", pack.pack_id],
          ["工单签发版本", String(pack.work_order_version)],
          ["设备签发版本", String(pack.asset_version)],
          ["签发时间", formatTimestamp(pack.issued_at)],
          ["到期时间", formatTimestamp(pack.expires_at)],
          ["内容摘要", pack.content_hash],
        ]} />
      </Card>
      <Card title={`${snapshot.work_order.work_order_id} · ${snapshot.asset.display_name ?? snapshot.asset.asset_id}`}>
        <FactGrid facts={[
          ["工单状态", snapshot.work_order.status],
          ["优先级", snapshot.work_order.priority],
          ["设备型号", snapshot.asset.model_code],
          ["设备序列号", snapshot.asset.serial_number],
          ["站点", snapshot.site?.site_name ?? "—"],
          ["SLA 截止", formatTimestamp(snapshot.work_order.sla_due_at)],
        ]} />
        <Alert
          type="info"
          showIcon
          message={snapshot.incident.description ?? "故障描述未提供"}
          description={`严重度 ${snapshot.incident.severity ?? "—"} · 类别 ${snapshot.incident.category ?? "—"}`}
        />
      </Card>
      <Card title="诊断摘要">
        {snapshot.diagnosis ? (
          <div className="page-stack">
            <Typography.Paragraph>{snapshot.diagnosis.conclusion ?? "暂无诊断结论"}</Typography.Paragraph>
            <FactGrid facts={[
              ["诊断版本", String(snapshot.diagnosis.version)],
              ["置信度", snapshot.diagnosis.confidence === null ? "—" : String(snapshot.diagnosis.confidence)],
            ]} />
            <List
              header="下一步检查"
              dataSource={snapshot.diagnosis.next_checks}
              locale={{ emptyText: "暂无下一步检查" }}
              renderItem={(item) => <List.Item>{item}</List.Item>}
            />
          </div>
        ) : <EmptyState description="签发时没有可用诊断摘要" />}
      </Card>
      <Card title="引用依据">
        <List
          dataSource={snapshot.citations}
          locale={{ emptyText: "签发时没有可用引用" }}
          renderItem={(citation) => (
            <List.Item>
              <List.Item.Meta
                title={`${citation.citation_id}${citation.page_number === null ? "" : ` · 第 ${citation.page_number} 页`}`}
                description={citation.excerpt ?? citation.excerpt_checksum}
              />
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
}

function FieldEntryTimelineItem({
  entry,
  entryCheckpoint,
}: {
  entry: FieldWorkOrderEntry;
  entryCheckpoint: number;
}) {
  const currentRoundTag = entry.sequence > entryCheckpoint
    ? <Tag color="blue">本轮</Tag>
    : <Tag>历史轮次</Tag>;
  if (entry.entry_type !== "AI_OBSERVATION") {
    return (
      <List.Item>
        <List.Item.Meta
          title={<Space><Tag>{entry.entry_type}</Tag><span>#{entry.sequence}</span>{currentRoundTag}</Space>}
          description={`${formatTimestamp(entry.occurred_at)} · ${entrySummary(entry)}`}
        />
      </List.Item>
    );
  }
  const edgeCandidateId = stringPayloadValue(
    entry.payload,
    "edge_diagnosis_candidate_id",
  );
  const sourceKind = stringPayloadValue(entry.payload, "source_kind");
  const isEdgeCandidate = sourceKind === "EDGE_DIAGNOSIS_CANDIDATE"
    || Boolean(edgeCandidateId);
  const provenanceId = isEdgeCandidate
    ? edgeCandidateId ?? "未知边缘候选"
    : stringPayloadValue(entry.payload, "recognition_bundle_id") ?? "未知识别包";
  const runtimeAttestation = stringPayloadValue(entry.payload, "runtime_attestation");
  const observation = stringPayloadValue(entry.payload, "observation") ?? "现场观察";
  const sources = Array.isArray(entry.payload.sources)
    ? entry.payload.sources.filter(isObservationTimelineSource)
    : [];
  return (
    <List.Item>
      <div style={{ width: "100%" }}>
        <Space wrap>
          <Tag color="purple">AI_OBSERVATION</Tag>
          <Typography.Text strong>{`AI 现场观察 · ${provenanceId}`}</Typography.Text>
          {isEdgeCandidate ? <Tag color="gold">{runtimeAttestation ?? "UNATTESTED"}</Tag> : null}
          <span>#{entry.sequence}</span>
          {currentRoundTag}
        </Space>
        <Typography.Paragraph style={{ marginBottom: 4, marginTop: 8 }}>
          {observation}
        </Typography.Paragraph>
        <Space wrap>
          {sources.map((source) => (
            <Tag key={`${source.source_type}:${source.source_id}`}>
              {source.source_type} · {source.source_id}
            </Tag>
          ))}
          <Typography.Text type="secondary">
            {formatTimestamp(entry.occurred_at)}
          </Typography.Text>
        </Space>
      </div>
    </List.Item>
  );
}

function isObservationTimelineSource(
  value: unknown,
): value is { source_type: string; source_id: string } {
  if (!value || typeof value !== "object") return false;
  const source = value as { source_type?: unknown; source_id?: unknown };
  return typeof source.source_type === "string" && typeof source.source_id === "string";
}

function stringPayloadValue(payload: Record<string, unknown>, key: string): string | undefined {
  const value = payload[key];
  return typeof value === "string" ? value : undefined;
}

function entrySummary(entry: FieldWorkOrderEntry): string {
  const payload = entry.payload;
  return String(
    payload.description
    ?? payload.part_number
    ?? payload.evidence_id
    ?? payload.signed_by
    ?? "现场事实",
  );
}

function formatTimestamp(value: string | null): string {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "—";
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

function PartIssueState({ allocation }: { allocation: WorkOrderPartAllocation }) {
  if (allocation.issue_status === "LEGACY_COMPATIBLE") {
    return (
      <Alert
        type="warning"
        showIcon
        message="存量兼容模式：无 WMS 出库证明"
        description="现场用料仍可引用升级前批准预留，但不等于 WMS 已出库或库存已扣减。"
      />
    );
  }
  if (allocation.issue_status === "ISSUED") {
    return (
      <div className="page-stack">
        <Alert
          type="success"
          showIcon
          message="WMS 已确认出库"
          description="仅此关闭式权威结果开放现场备件用料；现场记录本身不会反向修改库存。"
        />
        <FactGrid facts={[
          ["平台出库事实", allocation.part_issue_id],
          ["WMS 出库 ID", allocation.external_issue_id],
          ["出库操作号", allocation.operation_id],
          ["出库来源", allocation.issue_source],
          ["WMS 源记录", allocation.issue_source_record_id],
          ["出库事实时间", formatTimestamp(allocation.issue_as_of)],
        ]} />
      </div>
    );
  }
  if (allocation.issue_status === "NOT_REQUESTED") {
    return (
      <Alert
        type="warning"
        showIcon
        message="备件尚未出库"
        description="当前只有批准预留。请发起独立出库审批；在 WMS 明确确认前，用料入口保持关闭。"
      />
    );
  }
  if (allocation.issue_status === "PENDING_APPROVAL") {
    return <Alert type="info" showIcon message="等待独立审批" description={`审批 ${allocation.approval_id ?? "—"} 尚未决定，WMS 未被调用。`} />;
  }
  if (allocation.issue_status === "APPROVED") {
    return <Alert type="info" showIcon message="出库已批准，等待执行" description="执行者仍会复核负责人、工单版本和完整预留绑定后才调用 WMS。" />;
  }
  if (allocation.issue_status === "RECONCILING") {
    return (
      <Alert
        type="warning"
        showIcon
        message="WMS 出库结果待确认"
        description={<span>平台不会重新提交出库，只按原操作号查询。<Link href="/reconciliations">进入对账中心</Link></span>}
      />
    );
  }
  return (
    <Alert
      type="error"
      showIcon
      message={`出库未完成：${allocation.issue_status}`}
      description="预留仍未形成权威出库事实，现场用料入口保持关闭。"
    />
  );
}

function validReturnQuantity(value: string, maximum: number): boolean {
  const quantity = Number(value);
  return Number.isInteger(quantity) && quantity >= 1 && quantity <= maximum;
}

function partMovementKindLabel(kind: string): string {
  return kind === "CONSUME" ? "实际消耗" : kind === "RETURN" ? "退料" : kind;
}

function partAccountingBlockingReason(reason: string): string {
  return {
    unconsumed_field_entries: "仍有现场用料未形成 WMS 实际消耗",
    unaccounted_issued_quantity: "消耗加退料尚未覆盖全部出库数量",
    part_movement_in_progress: "仍有消耗或退料动作等待审批/执行/对账",
  }[reason] ?? reason;
}

function evidenceScanLabel(state: string): string {
  return {
    PENDING: "扫描中",
    CLEAN: "可关联",
    REJECTED: "不可用",
    INFECTED: "不可用",
  }[state] ?? state;
}

function evidenceScanColor(state: string): string {
  return {
    PENDING: "gold",
    CLEAN: "green",
    REJECTED: "red",
    INFECTED: "red",
  }[state] ?? "default";
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
}

function evidenceErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "无法读取或同步现场证据";
}

function fieldEntryActionErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "操作未完成，请重试";
}
