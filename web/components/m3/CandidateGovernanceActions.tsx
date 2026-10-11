"use client";

import { Button, Space } from "antd";

import type { FeedbackCandidate } from "@/lib/api/client";

type CandidateActionFacts = Pick<
  FeedbackCandidate,
  "eligibility_status" | "legal_actions"
>;

export function CandidateGovernanceActions({
  candidate,
  busy = false,
  onInspect,
  onEligibility,
  onDlp,
  onAnnotation,
  onSync,
}: {
  candidate: CandidateActionFacts;
  busy?: boolean;
  onInspect?: () => void;
  onEligibility: () => void;
  onDlp: () => void;
  onAnnotation: () => void;
  onSync: () => void;
}) {
  const actions = candidate.legal_actions;
  if (!onInspect && actions.length === 0) return null;
  return (
    <Space wrap aria-busy={busy}>
      {onInspect ? <Button disabled={busy} onClick={onInspect}>查看血缘</Button> : null}
      {actions.includes("DECIDE_ELIGIBILITY") ? (
        <Button disabled={busy} onClick={onEligibility}>
          {candidate.eligibility_status === "UNDECIDED" ? "用途授权" : "重新决策"}
        </Button>
      ) : null}
      {actions.includes("RUN_DLP") ? (
        <Button disabled={busy} onClick={onDlp}>执行 DLP</Button>
      ) : null}
      {actions.includes("CREATE_ANNOTATION_TASK") ? (
        <Button disabled={busy} onClick={onAnnotation}>送标注复核</Button>
      ) : null}
      {actions.includes("SYNC_ANNOTATION") ? (
        <Button disabled={busy} onClick={onSync}>同步复核</Button>
      ) : null}
    </Space>
  );
}
