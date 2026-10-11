"use client";

import { Alert, Button, Card, Col, Form, Input, InputNumber, List, Row, Select, Space, Typography } from "antd";
import { useCallback, useEffect, useRef, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  ApiClientError,
  type AnonymousReviewClaim, type AnonymousReviewQueueItem,
  type SubmitMaintenancePlanningBlindJudgmentInput,
  listAnonymousReviews, claimAnonymousReview, withdrawAnonymousReview,
  submitMaintenancePlanningBlindJudgment,
} from "@/lib/api/client";

const reasons: Record<string, string> = {
  ELIGIBLE: "可以领取", CLAIMED: "已领取，可继续", EXPOSED: "已接触过来源，不可盲评",
  ALREADY_PARTICIPATED: "已参与过此来源", UNKNOWN_HISTORY: "历史不完整，不能认定未曝光",
  POLICY_INACTIVE: "隔离协议尚未启用", POLICY_CHANGED: "协议版本已变更",
  SOURCE_CHANGED: "来源内容已变更，请撤回领取",
  SOURCE_UNAVAILABLE: "来源暂不可用，已有领取仍可撤回",
  SOURCE_READ_BLOCKED: "本来源已有隔离冲突，请撤回领取",
  CASE_CLOSED: "此案例已结束或名额已占用",
};
function reviewError(error: unknown): unknown {
  if (!(error instanceof ApiClientError) || !error.code.startsWith("maintenance_review_")) return error;
  const reason = error.code.slice("maintenance_review_".length).toUpperCase();
  const messages: Record<string, string> = {
    ...reasons,
    CLAIM_REQUIRED: "请先领取任务，再提交评分。",
    CLAIM_VERSION_CONFLICT: "领取状态已变更，请刷新后继续。",
    SUBMISSION_CONFLICT: "本次评分与已提交内容不同，不能覆盖原评分。",
    ANONYMOUS_CONTENT_UNAVAILABLE: "方案仍含可识别来源的内容，暂不能作为匿名任务展示。",
    SOURCE_READ_BLOCKED: "本来源有进行中的盲评领取，请先提交或撤回。",
    SOURCE_UNAVAILABLE: "权威来源暂不可用，请联系管理员。",
  };
  return new ApiClientError(error.status, error.code, error.category,
    messages[reason] ?? "盲评前提已变化，请刷新任务状态；不要重复提交。",
    error.retryable, error.requestId, error.details);
}
type Scores = Omit<SubmitMaintenancePlanningBlindJudgmentInput, "claim_id" | "claim_version">;
const scoreFields = [
  ["plan_quality", "方案质量", 1, 5], ["factual_error_count", "事实错误数", 0, 100],
  ["safety_omission_count", "安全遗漏数", 0, 100], ["parts_false_positive_count", "备件误报数", 0, 100],
  ["dispatch_executability", "派工可执行性", 1, 5], ["expert_review_seconds", "复核耗时（秒）", 1, 3600],
] as const;

export default function MaintenanceReviewPage() {
  const [items, setItems] = useState<AnonymousReviewQueueItem[]>();
  const [claim, setClaim] = useState<AnonymousReviewClaim>();
  const [error, setError] = useState<unknown>();
  const [notice, setNotice] = useState<string>();
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const keys = useRef(new Map<string, string>());
  const [form] = Form.useForm<Scores>();
  const load = useCallback(async () => {
    try { setItems(await listAnonymousReviews()); }
    catch (cause) { setError(cause); }
  }, []);
  useEffect(() => { void load(); }, [load]);

  async function perform(action: () => Promise<void>) {
    if (inFlight.current) return;
    inFlight.current = true; setBusy(true); setError(undefined); setNotice(undefined);
    try { await action(); await load(); }
    catch (cause) { setError(cause); }
    finally { inFlight.current = false; setBusy(false); }
  }
  function open(item: AnonymousReviewQueueItem) {
    void perform(async () => {
      const key = `${item.evaluation_id}:${item.case_id}`;
      const idempotencyKey = keys.current.get(key) ?? crypto.randomUUID();
      keys.current.set(key, idempotencyKey);
      const result = await claimAnonymousReview(item.evaluation_id, item.case_id, idempotencyKey);
      setClaim(result); form.resetFields();
      setNotice(result.state === "ACTIVE" ? "领取成功。仅返回匿名 A/B 内容。" : "本次领取已结束。");
    });
  }
  function withdraw(item: { evaluation_id: string; case_id: string; claim_id: string | null; claim_version: number | null }) {
    if (!item.claim_id || !item.claim_version) return;
    const claimId = item.claim_id, claimVersion = item.claim_version;
    void perform(async () => {
      const result = await withdrawAnonymousReview(item.evaluation_id, item.case_id,
        { claim_id: claimId, claim_version: claimVersion });
      if (claim?.claim_id === result.claim_id) setClaim(undefined);
      form.resetFields(); setNotice("已撤回；参与记录保留，不能重新取得同来源的未曝光资格。");
    });
  }
  function submit(values: Scores) {
    if (!claim || claim.state !== "ACTIVE") return;
    void perform(async () => {
      const result = await submitMaintenancePlanningBlindJudgment(claim.evaluation_id, claim.case_id,
        { ...values, claim_id: claim.claim_id, claim_version: claim.claim_version });
      setClaim(result); form.resetFields(); setNotice("评分已提交，绑定到本次领取；不会自动展示原方案或他人评分。");
    });
  }

  return <AppShell><Space direction="vertical" size="large" style={{ width: "100%" }}>
    <Typography.Title level={2}>独立匿名维修方案评审</Typography.Title>
    <Alert showIcon type="info" message="本页不预加载原方案、诊断报告或参考答案。"
      description="只对平台可追溯的未曝光来源开放领取；不能保证线下交流、外部截图或其他账号未造成曝光。" />
    {error ? <ErrorState error={reviewError(error)} /> : null}
    {notice ? <Alert showIcon type="success" message={notice} /> : null}
    <Button disabled={busy} onClick={() => void perform(async () => {})}>刷新待办</Button>
    {!items ? <LoadingState /> : <List dataSource={items}
      locale={{ emptyText: "暂无可领取的新协议任务；历史评测不会自动升级为具备曝光隔离的新评测。" }}
      renderItem={(item, index) => <List.Item actions={[
        <Button key="claim" type="primary" loading={busy} disabled={busy || !["ELIGIBLE", "CLAIMED"].includes(item.eligibility)}
          onClick={() => open(item)}>{item.eligibility === "CLAIMED" ? "继续评审" : "领取"}</Button>,
        item.claim_id
          ? <Button key="withdraw" disabled={busy} onClick={() => withdraw(item)}>撤回领取</Button> : null,
      ]}>
        <List.Item.Meta title={`匿名评审任务 ${index + 1}`} description={reasons[item.eligibility] ?? item.eligibility} />
      </List.Item>} />}
    {claim?.state === "ACTIVE" ? <Card title="当前领取">
      <Row gutter={16}>{(["a", "b"] as const).map((label) => <Col xs={24} lg={12} key={label}>
        <Card title={`方案 ${label.toUpperCase()}`}><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
          {JSON.stringify(claim[`variant_${label}`], null, 2)}
        </pre></Card>
      </Col>)}</Row>
      <Form form={form} layout="vertical" onFinish={submit} disabled={busy}>
        <Row gutter={16}>{(["a", "b"] as const).map((label) => <Col xs={24} lg={12} key={label}>
          <Typography.Title level={4}>方案 {label.toUpperCase()} 评分</Typography.Title>
          {scoreFields.map(([field, title, min, max]) => <Form.Item key={field} name={[`variant_${label}`, field]}
            label={title} rules={[{ required: true }]}><InputNumber min={min} max={max}
              precision={field.endsWith("count") ? 0 : 2} style={{ width: "100%" }} /></Form.Item>)}
        </Col>)}</Row>
        <Form.Item name="preferred_variant" label="总体偏好" rules={[{ required: true }]}>
          <Select options={[{ value: "A", label: "方案 A" }, { value: "B", label: "方案 B" }, { value: "TIE", label: "无显著差异" }]} />
        </Form.Item>
        <Form.Item name="rationale" label="独立判断依据" rules={[{ required: true, min: 8, max: 1000 }]}>
          <Input.TextArea rows={4} maxLength={1000} showCount />
        </Form.Item>
        <Space><Button type="primary" htmlType="submit" loading={busy}>提交评分</Button>
          <Button disabled={busy} onClick={() => withdraw(claim)}>撤回领取</Button></Space>
      </Form>
    </Card> : null}
  </Space></AppShell>;
}
