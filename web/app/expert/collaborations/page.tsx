"use client";

import { Alert, Button, Card, Empty, Input, Radio, Space, Tag, Typography } from "antd";
import { useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { LoadingState } from "@/components/RequestState";
import { RemoteExpertAudioPanel } from "@/components/m3/RemoteExpertAudioPanel";
import {
  ExpertRecommendationDetails,
  collaborationStatusLabel,
} from "@/components/m3/RemoteExpertCollaborationPanel";
import {
  type ExpertCollaboration,
  type SubmitExpertRecommendationInput,
  acceptExpertCollaboration,
  endExpertCollaboration,
  listExpertCollaborations,
  submitExpertRecommendation,
} from "@/lib/api/client";

type RecommendationType = SubmitExpertRecommendationInput["recommendation_type"];

export default function ExpertCollaborationsPage() {
  const [items, setItems] = useState<ExpertCollaboration[]>();
  const [error, setError] = useState<string>();
  const [busyId, setBusyId] = useState<string>();
  const [editingId, setEditingId] = useState<string>();
  const [recommendationType, setRecommendationType] = useState<RecommendationType>("ADVICE");
  const [summary, setSummary] = useState("");
  const [basis, setBasis] = useState("");
  const [checks, setChecks] = useState("");
  const [safetyNotice, setSafetyNotice] = useState("");
  const [evidenceEntryIds, setEvidenceEntryIds] = useState("");

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const result = await listExpertCollaborations();
      setItems(result.collaborations);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法读取专家协作待办。");
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  async function accept(item: ExpertCollaboration) {
    setBusyId(item.collaboration_id);
    try {
      const result = await acceptExpertCollaboration(item.collaboration_id, item.version);
      replace(result.collaboration);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "接受邀请失败。");
    } finally {
      setBusyId(undefined);
    }
  }

  async function end(item: ExpertCollaboration) {
    setBusyId(item.collaboration_id);
    try {
      const result = await endExpertCollaboration(
        item.collaboration_id,
        item.version,
        "受邀专家显式结束远程协作。",
      );
      replace(result.collaboration);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "结束协作失败。");
    } finally {
      setBusyId(undefined);
    }
  }

  async function submitRecommendation(item: ExpertCollaboration) {
    const recommendedChecks = splitValues(checks);
    if (summary.trim().length < 10 || basis.trim().length < 10 || !recommendedChecks.length) {
      return;
    }
    setBusyId(item.collaboration_id);
    setError(undefined);
    try {
      const result = await submitExpertRecommendation(
        item.collaboration_id,
        {
          recommendation_type: recommendationType,
          summary: summary.trim(),
          basis: basis.trim(),
          recommended_checks: recommendedChecks,
          safety_notice: safetyNotice.trim() || null,
          evidence_entry_ids: splitValues(evidenceEntryIds, true),
        },
        operationId(),
      );
      replace(result.collaboration);
      resetRecommendationForm();
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "提交专家建议失败。";
      await load();
      setError(message);
    } finally {
      setBusyId(undefined);
    }
  }

  function resetRecommendationForm() {
    setEditingId(undefined);
    setRecommendationType("ADVICE");
    setSummary("");
    setBasis("");
    setChecks("");
    setSafetyNotice("");
    setEvidenceEntryIds("");
  }

  function replace(value: ExpertCollaboration) {
    setItems((current) => current?.map((item) => (
      item.collaboration_id === value.collaboration_id ? value : item
    )));
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>远程专家协作待办</Typography.Title>
        <Typography.Paragraph type="secondary">
          这里只展示当前账号被明确邀请的工单。接受邀请不会自动打开麦克风，提交建议也不会自动结束协作。
        </Typography.Paragraph>
        {error ? <Alert type="error" showIcon message={error} action={<Button onClick={() => void load()}>重试</Button>} /> : null}
        {!items && !error ? <LoadingState label="正在读取专家协作待办" /> : null}
        {items?.length === 0 ? <Empty description="暂无专家协作待办" /> : null}
        {items?.map((item) => (
          <Card key={item.collaboration_id} title={`工单 ${item.work_order_id}`}>
            <Space direction="vertical" style={{ width: "100%" }}>
              <Space wrap>
                <Tag color={item.status === "ACTIVE" ? "green" : "gold"}>
                  {collaborationStatusLabel(item.status)}
                </Tag>
                <Typography.Text code>{item.asset_id}</Typography.Text>
                <Typography.Text>v{item.version}</Typography.Text>
              </Space>
              <Typography.Paragraph>{item.reason}</Typography.Paragraph>
              {item.legal_actions.includes("ACCEPT") ? (
                <Button type="primary" loading={busyId === item.collaboration_id} onClick={() => void accept(item)}>
                  接受邀请
                </Button>
              ) : null}
              {item.status === "ACTIVE" ? (
                <RemoteExpertAudioPanel collaborationId={item.collaboration_id} />
              ) : null}
              {item.recommendation ? (
                <ExpertRecommendationDetails recommendation={item.recommendation} />
              ) : null}
              {item.legal_actions.includes("SUBMIT_RECOMMENDATION") && editingId !== item.collaboration_id ? (
                <Button onClick={() => setEditingId(item.collaboration_id)}>
                  填写结构化建议
                </Button>
              ) : null}
              {item.legal_actions.includes("SUBMIT_RECOMMENDATION") && editingId === item.collaboration_id ? (
                <Card size="small" title="提交专家结构化建议">
                  <Space direction="vertical" style={{ width: "100%" }}>
                    <Radio.Group
                      aria-label="建议类型"
                      value={recommendationType}
                      onChange={(event) => setRecommendationType(event.target.value as RecommendationType)}
                      options={[
                        { label: "处置建议", value: "ADVICE" },
                        { label: "补充证据", value: "REQUEST_MORE_EVIDENCE" },
                        { label: "停止并升级", value: "STOP_AND_ESCALATE" },
                      ]}
                    />
                    <Input.TextArea
                      aria-label="建议摘要"
                      rows={2}
                      value={summary}
                      placeholder="概括建议结论（至少 10 个字符）"
                      onChange={(event) => setSummary(event.target.value)}
                    />
                    <Input.TextArea
                      aria-label="判断依据"
                      rows={3}
                      value={basis}
                      placeholder="说明建议所依据的现场事实和专业判断"
                      onChange={(event) => setBasis(event.target.value)}
                    />
                    <Input.TextArea
                      aria-label="建议检查项"
                      rows={4}
                      value={checks}
                      placeholder="每行填写一个建议检查项"
                      onChange={(event) => setChecks(event.target.value)}
                    />
                    <Input.TextArea
                      aria-label="安全提示"
                      rows={2}
                      value={safetyNotice}
                      placeholder="可选：填写必须遵守的安全边界"
                      onChange={(event) => setSafetyNotice(event.target.value)}
                    />
                    <Input.TextArea
                      aria-label="引用现场条目"
                      rows={2}
                      value={evidenceEntryIds}
                      placeholder="可选：每行填写一个当前工单现场条目 ID"
                      onChange={(event) => setEvidenceEntryIds(event.target.value)}
                    />
                    <Space wrap>
                      <Button
                        type="primary"
                        loading={busyId === item.collaboration_id}
                        disabled={summary.trim().length < 10 || basis.trim().length < 10 || !splitValues(checks).length}
                        onClick={() => void submitRecommendation(item)}
                      >
                        提交专家建议
                      </Button>
                      <Button disabled={busyId === item.collaboration_id} onClick={resetRecommendationForm}>
                        取消
                      </Button>
                    </Space>
                  </Space>
                </Card>
              ) : null}
              {item.legal_actions.includes("END") ? (
                <Button danger loading={busyId === item.collaboration_id} onClick={() => void end(item)}>
                  结束协作
                </Button>
              ) : null}
            </Space>
          </Card>
        ))}
      </div>
    </AppShell>
  );
}

function splitValues(value: string, commas = false): string[] {
  const separator = commas ? /[\n,]/ : /\n/;
  return [...new Set(value.split(separator).map((item) => item.trim()).filter(Boolean))];
}

function operationId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `expert-recommendation-${Date.now()}`;
}
