"use client";

import { Space, Tag, Typography } from "antd";

import type { GovernanceBlockerCode } from "@/lib/api/client";

const GOVERNANCE_REASON_LABELS: Record<string, string> = {
  training_not_authorized: "训练用途未授权",
  consent_basis_missing: "缺少授权依据",
  license_not_approved: "许可证未批准",
  retention_missing: "缺少保留期限",
  retention_expired: "保留期限已过期",
  dlp_not_run: "尚未执行 DLP",
  dlp_review_required: "DLP 需要人工复核",
  annotation_not_started: "尚未创建标注任务",
  annotation_review_pending: "等待标注复核",
  annotation_review_required: "标注需要继续复核",
  annotation_conflict: "标注存在冲突，等待仲裁",
  governance_status_blocked: "候选治理状态异常",
  annotation_not_approved: "标注尚未批准",
  authoritative_source_missing_or_changed: "权威业务事实缺失或已变化",
  dlp_not_current_or_approved: "DLP 结果不是当前版本或未通过",
  annotation_not_current_or_approved: "标注结果不是当前版本或未通过",
  approved_annotation_revision_missing: "缺少已批准的标注修订",
};

export function governanceReasonLabel(reason: string): string {
  return GOVERNANCE_REASON_LABELS[reason] ?? "其他治理门禁";
}

const GOVERNANCE_BLOCKER_CODES = [
  "training_not_authorized",
  "consent_basis_missing",
  "license_not_approved",
  "retention_missing",
  "retention_expired",
  "dlp_not_run",
  "dlp_review_required",
  "annotation_not_started",
  "annotation_review_pending",
  "annotation_review_required",
  "annotation_conflict",
  "governance_status_blocked",
] as const satisfies readonly GovernanceBlockerCode[];

export const GOVERNANCE_BLOCKER_OPTIONS = GOVERNANCE_BLOCKER_CODES.map((value) => ({
  value,
  label: governanceReasonLabel(value),
}));

export function GovernanceBlockers({ blockers }: { blockers: string[] }) {
  if (!blockers.length) return <Tag color="green">全部治理门禁已通过</Tag>;
  return (
    <Space wrap size={[4, 4]}>
      <Typography.Text type="secondary">当前阻断：</Typography.Text>
      {blockers.map((blocker) => (
        <Tag key={blocker} color="orange" title={blocker}>
          {governanceReasonLabel(blocker)}
        </Tag>
      ))}
    </Space>
  );
}
