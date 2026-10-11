"use client";

import { Alert, Button, Card, Input, Radio, Space, Tag, Typography } from "antd";
import { useCallback, useEffect, useState } from "react";

import { RemoteExpertAudioPanel } from "@/components/m3/RemoteExpertAudioPanel";
import {
  type ExpertCollaboration,
  type ExpertCollaborationOverview,
  type ExpertRecommendation,
  createExpertCollaboration,
  decideExpertRecommendation,
  endExpertCollaboration,
  getFieldExpertCollaboration,
} from "@/lib/api/client";

export function RemoteExpertCollaborationPanel({ workOrderId }: { workOrderId: string }) {
  const [overview, setOverview] = useState<ExpertCollaborationOverview>();
  const [collaboration, setCollaboration] = useState<ExpertCollaboration>();
  const [selectedExpert, setSelectedExpert] = useState<string>();
  const [reason, setReason] = useState("");
  const [reviewReason, setReviewReason] = useState("");
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const result = await getFieldExpertCollaboration(workOrderId);
      setOverview(result.overview);
      setCollaboration(result.overview.collaboration ?? undefined);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法读取专家协作状态。");
    }
  }, [workOrderId]);

  useEffect(() => { void load(); }, [load]);

  async function invite() {
    if (!overview || !selectedExpert || reason.trim().length < 10) return;
    setBusy(true);
    setError(undefined);
    try {
      const result = await createExpertCollaboration(
        workOrderId,
        overview.work_order_version,
        selectedExpert,
        reason.trim(),
        operationId(),
      );
      setCollaboration(result.collaboration);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "专家邀请失败。");
    } finally {
      setBusy(false);
    }
  }

  async function decideRecommendation(decision: "ACCEPT" | "REJECT") {
    const recommendation = collaboration?.recommendation;
    if (!collaboration || !recommendation || reviewReason.trim().length < 4) return;
    setBusy(true);
    setError(undefined);
    try {
      const result = await decideExpertRecommendation(
        collaboration.collaboration_id,
        recommendation.version,
        decision,
        reviewReason.trim(),
      );
      setCollaboration(result.collaboration);
      setReviewReason("");
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "专家建议复核失败。";
      await load();
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  async function end() {
    if (!collaboration) return;
    setBusy(true);
    try {
      const result = await endExpertCollaboration(
        collaboration.collaboration_id,
        collaboration.version,
        "现场参与者显式结束远程专家协作。",
      );
      setCollaboration(result.collaboration);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "结束专家协作失败。");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="远程专家协作">
      <Space direction="vertical" style={{ width: "100%" }}>
        <Typography.Text type="secondary">
          邀请只绑定当前工单和指定专家；专家接受后，双方仍需各自显式加入音频。
        </Typography.Text>
        {error ? <Alert type="error" showIcon message={error} /> : null}
        {!collaboration && overview ? (
          <>
            {overview.candidates.length ? (
              <Radio.Group
                value={selectedExpert}
                onChange={(event) => setSelectedExpert(String(event.target.value))}
              >
                <Space direction="vertical">
                  {overview.candidates.map((candidate) => (
                    <Radio key={candidate.subject_id} value={candidate.subject_id}>
                      {candidate.display_name}
                    </Radio>
                  ))}
                </Space>
              </Radio.Group>
            ) : <Alert type="info" message="当前资产范围内没有可邀请的领域专家" />}
            <Input.TextArea
              value={reason}
              rows={3}
              placeholder="说明需要专家协助判断的现场问题"
              onChange={(event) => setReason(event.target.value)}
            />
            <Button
              type="primary"
              loading={busy}
              disabled={!selectedExpert || reason.trim().length < 10}
              onClick={() => void invite()}
            >
              发送专家邀请
            </Button>
          </>
        ) : null}
        {collaboration ? (
          <>
            <Space wrap>
              <Tag color={collaboration.status === "ACTIVE" ? "green" : "gold"}>
                {collaborationStatusLabel(collaboration.status)}
              </Tag>
              <Typography.Text code>{collaboration.invited_subject_id}</Typography.Text>
              <Typography.Text>v{collaboration.version}</Typography.Text>
            </Space>
            <Typography.Paragraph>{collaboration.reason}</Typography.Paragraph>
            {collaboration.status === "ACTIVE" ? (
              <RemoteExpertAudioPanel collaborationId={collaboration.collaboration_id} />
            ) : null}
            {collaboration.recommendation ? (
              <ExpertRecommendationDetails recommendation={collaboration.recommendation} />
            ) : null}
            {collaboration.recommendation?.legal_actions.some((action) => (
              action === "ACCEPT" || action === "REJECT"
            )) ? (
              <Space direction="vertical" style={{ width: "100%" }}>
                <Input.TextArea
                  aria-label="现场复核说明"
                  value={reviewReason}
                  rows={3}
                  placeholder="说明接受或驳回该专家建议的现场核对依据"
                  onChange={(event) => setReviewReason(event.target.value)}
                />
                <Space wrap>
                  {collaboration.recommendation.legal_actions.includes("ACCEPT") ? (
                    <Button
                      type="primary"
                      loading={busy}
                      disabled={reviewReason.trim().length < 4}
                      onClick={() => void decideRecommendation("ACCEPT")}
                    >
                      接受并写入工单事实
                    </Button>
                  ) : null}
                  {collaboration.recommendation.legal_actions.includes("REJECT") ? (
                    <Button
                      danger
                      loading={busy}
                      disabled={reviewReason.trim().length < 4}
                      onClick={() => void decideRecommendation("REJECT")}
                    >
                      驳回专家建议
                    </Button>
                  ) : null}
                </Space>
              </Space>
            ) : null}
            {collaboration.legal_actions.includes("END") ? (
              <Button danger loading={busy} onClick={() => void end()}>结束专家协作</Button>
            ) : null}
          </>
        ) : null}
      </Space>
    </Card>
  );
}

export function ExpertRecommendationDetails({
  recommendation,
}: {
  recommendation: ExpertRecommendation;
}) {
  return (
    <Card size="small" title="专家结构化建议">
      <Space direction="vertical" style={{ width: "100%" }}>
        <Space wrap>
          <Tag color={recommendation.status === "ACCEPTED" ? "green" : "blue"}>
            {recommendationStatusLabel(recommendation.status)}
          </Tag>
          <Typography.Text code>{recommendation.expert_subject_id}</Typography.Text>
          <Typography.Text>建议 v{recommendation.version}</Typography.Text>
        </Space>
        <Typography.Text strong>建议摘要</Typography.Text>
        <Typography.Paragraph>{recommendation.summary}</Typography.Paragraph>
        <Typography.Text strong>判断依据</Typography.Text>
        <Typography.Paragraph>{recommendation.basis}</Typography.Paragraph>
        <Typography.Text strong>建议检查项</Typography.Text>
        <ul style={{ margin: 0, paddingInlineStart: 22 }}>
          {recommendation.recommended_checks.map((item) => <li key={item}>{item}</li>)}
        </ul>
        {recommendation.safety_notice ? (
          <Alert type="warning" showIcon message={recommendation.safety_notice} />
        ) : null}
        {recommendation.evidence_entry_ids.length ? (
          <Space wrap>
            <Typography.Text type="secondary">引用现场条目</Typography.Text>
            {recommendation.evidence_entry_ids.map((entryId) => (
              <Typography.Text code key={entryId}>{entryId}</Typography.Text>
            ))}
          </Space>
        ) : null}
        {recommendation.review_reason ? (
          <Typography.Text>复核说明：{recommendation.review_reason}</Typography.Text>
        ) : null}
        {recommendation.field_entry_id ? (
          <Space wrap>
            <Typography.Text type="secondary">工单事实</Typography.Text>
            <Typography.Text code>{recommendation.field_entry_id}</Typography.Text>
          </Space>
        ) : null}
      </Space>
    </Card>
  );
}

export function collaborationStatusLabel(status: string): string {
  return {
    WAITING_EXPERT: "等待专家接受",
    ACTIVE: "协作进行中",
    ENDED: "协作已结束",
    EXPIRED: "协作已过期",
  }[status] ?? status;
}

export function recommendationStatusLabel(status: string): string {
  return {
    PENDING_FIELD_REVIEW: "等待现场复核",
    ACCEPTED: "已接受",
    REJECTED: "已驳回",
  }[status] ?? status;
}

function operationId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `expert-invite-${Date.now()}`;
}
