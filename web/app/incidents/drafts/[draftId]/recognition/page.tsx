"use client";

import { Alert, Button, Card, Checkbox, Image, Input, List, Progress, Space, Tag, Typography } from "antd";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import { ModelRuntimeStatus } from "@/components/model-runtime/ModelRuntimeStatus";
import {
  type EvidenceBundle,
  type Incident,
  type IncidentDraft,
  type IncidentOcrBlockDecision,
  type IncidentQrCodeDecision,
  type RecognitionRun,
  confirmRecognition,
  getDraftEvidence,
  getIncidentDraft,
  getRecognitionRun,
  startRecognition,
  submitIncident,
} from "@/lib/api/client";

type RunNotice = {
  type: "info" | "warning" | "error" | "success";
  message: string;
  description: string;
};

export default function RecognitionPage() {
  const { draftId } = useParams<{ draftId: string }>();
  const [draft, setDraft] = useState<IncidentDraft>();
  const [mediaId, setMediaId] = useState("");
  const [temporalAnalysis, setTemporalAnalysis] = useState(false);
  const [run, setRun] = useState<RecognitionRun>();
  const [evidence, setEvidence] = useState<EvidenceBundle>();
  const [entityEdits, setEntityEdits] = useState<Record<string, string>>({});
  const [transcriptEdits, setTranscriptEdits] = useState<Record<string, string>>({});
  const [rejectedSegments, setRejectedSegments] = useState<Record<string, boolean>>({});
  const [rejectedFindings, setRejectedFindings] = useState<Record<string, boolean>>({});
  const [ocrBlockDecisions, setOcrBlockDecisions] = useState<
    Record<string, IncidentOcrBlockDecision>
  >({});
  const [rejectedVideoEvents, setRejectedVideoEvents] = useState<Record<string, boolean>>({});
  const [qrCodeDecisions, setQrCodeDecisions] = useState<
    Record<string, IncidentQrCodeDecision>
  >({});
  const [incident, setIncident] = useState<Incident>();
  const [error, setError] = useState<unknown>();
  const [mediaIdError, setMediaIdError] = useState<string>();
  const [runNotice, setRunNotice] = useState<RunNotice>();
  const [busy, setBusy] = useState(false);

  const loadDraft = useCallback(async () => {
    setError(undefined);
    try {
      setDraft((await getIncidentDraft(draftId)).draft);
    } catch (caught) {
      setError(caught);
    }
  }, [draftId]);

  useEffect(() => { void loadDraft(); }, [loadDraft]);

  useEffect(() => {
    const uploadedMediaId = new URLSearchParams(window.location.search)
      .get("media_id")
      ?.trim();
    if (uploadedMediaId) setMediaId(uploadedMediaId);
  }, []);

  async function beginRecognition() {
    const normalizedMediaId = mediaId.trim();
    if (!normalizedMediaId) {
      setMediaIdError("请先在故障草稿页上传媒体并等待安全扫描通过。");
      return;
    }
    setMediaIdError(undefined);
    setRunNotice(undefined);
    setEvidence(undefined);
    await perform(async () => {
      const key = crypto.randomUUID();
      const nextRun = (await startRecognition(
        draftId,
        normalizedMediaId,
        key,
        temporalAnalysis ? "video-temporal-v1" : "local-core-v1",
      )).recognition;
      setRun(nextRun);
      setRunNotice({
        type: "info",
        message: "识别任务已进入队列",
        description: "Worker 正在生成证据。稍候点击“刷新识别状态”即可继续。",
      });
    });
  }

  async function refreshEvidence() {
    await perform(async () => {
      if (run) {
        const latestRun = (await getRecognitionRun(
          draftId,
          run.recognition_run_id,
        )).recognition;
        setRun(latestRun);
        if (latestRun.status === "FAILED") {
          setRunNotice({
            type: "error",
            message: "识别任务执行失败",
            description: recognitionFailureDescription(latestRun.failure_reason),
          });
          return;
        }
        if (latestRun.status === "QUEUED" || latestRun.status === "RUNNING") {
          setRunNotice({
            type: "info",
            message: latestRun.status === "QUEUED" ? "识别任务正在排队" : "识别任务正在处理",
            description: "证据尚未生成，请稍候再次刷新；这不是数据丢失或权限错误。",
          });
          return;
        }
      }
      adoptEvidence((await getDraftEvidence(draftId)).evidence);
      setRunNotice({
        type: "success",
        message: "已读取最新证据",
        description: "请核对识别结果并确认当前证据版本。",
      });
    });
  }

  async function approveEvidence() {
    if (!evidence) return;
    const dispositions = Object.fromEntries(
      evidence.visual_findings.map((finding) => [
        finding.finding_id,
        rejectedFindings[finding.finding_id] ? "REJECTED" as const : "ACCEPTED" as const,
      ]),
    );
    const corrections = Object.fromEntries(
      evidence.extracted_entities
        .filter((item) => item.validation_status !== "VALID")
        .map((item) => [item.entity_id, entityEdits[item.entity_id] ?? item.original_value]),
    );
    const transcriptDecisions = Object.fromEntries(
      evidence.asr_segments.map((segment) => [
        segment.segment_id,
        rejectedSegments[segment.segment_id]
          ? { disposition: "REJECTED" as const, corrected_text: null }
          : {
              disposition: "ACCEPTED" as const,
              corrected_text: transcriptEdits[segment.segment_id] ?? segment.text,
            },
      ]),
    );
    await perform(async () => {
      const result = (evidence.qr_codes ?? []).length
        ? await confirmRecognition(
            draftId,
            evidence,
            corrections,
            dispositions,
            transcriptDecisions,
            ocrBlockDecisions,
            qrCodeDecisions,
          )
        : await confirmRecognition(
            draftId,
            evidence,
            corrections,
            dispositions,
            transcriptDecisions,
            ocrBlockDecisions,
          );
      adoptEvidence(result.evidence);
      await loadDraft();
    });
  }

  function adoptEvidence(next: EvidenceBundle) {
    setEvidence(next);
    setEntityEdits(Object.fromEntries(next.extracted_entities.map((item) => [
      item.entity_id,
      item.corrected_value ?? item.original_value,
    ])));
    setTranscriptEdits(Object.fromEntries(next.asr_segments.map((item) => [
      item.segment_id,
      item.corrected_text ?? (unsafeSource(next, "ASR_SEGMENT", item.segment_id) ? "" : item.text),
    ])));
    setRejectedSegments(Object.fromEntries(next.asr_segments.map((item) => [
      item.segment_id,
      item.disposition === "REJECTED",
    ])));
    setRejectedFindings(Object.fromEntries(next.visual_findings.map((item) => [
      item.finding_id,
      item.disposition === "REJECTED",
    ])));
    setOcrBlockDecisions(Object.fromEntries(next.ocr_blocks.flatMap((item) => {
      if (item.disposition) {
        return [[item.block_id, {
          disposition: item.disposition,
          corrected_text: item.corrected_text,
        }]];
      }
      return unsafeSource(next, "OCR_BLOCK", item.block_id)
        ? []
        : [[item.block_id, { disposition: "ACCEPTED", corrected_text: null }]];
    })) as Record<string, IncidentOcrBlockDecision>);
    setRejectedVideoEvents(Object.fromEntries(next.video_events
      .filter((event) => isReviewableVideoEvent(event.event_type))
      .map((event) => [
        event.event_id,
        event.disposition === "REJECTED",
      ])));
    setQrCodeDecisions(Object.fromEntries((next.qr_codes ?? []).flatMap((candidate) => (
      candidate.disposition
        ? [[candidate.candidate_id, {
            disposition: candidate.disposition,
            corrected_text: candidate.corrected_text,
          }]]
        : []
    ))) as Record<string, IncidentQrCodeDecision>);
  }

  async function submit() {
    if (!draft || !evidence) return;
    await perform(async () => {
      const result = await submitIncident(
        draftId,
        draft.version,
        evidence.bundle_id,
        crypto.randomUUID(),
      );
      setIncident(result.incident);
    });
  }

  async function perform(operation: () => Promise<void>) {
    setBusy(true);
    setError(undefined);
    try { await operation(); } catch (caught) { setError(caught); } finally { setBusy(false); }
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>多模态识别与证据确认</Typography.Title>
        <Alert
          type="info"
          showIcon
          message="人机协同边界"
          description="OCR、实体和值域校验、视觉发现均作为服务端事实展示；提交报障前必须由当前用户确认证据版本。"
        />
        <ModelRuntimeStatus
          requiredComponents={["ocr", "vlm"]}
          executions={evidence?.model_executions ?? []}
          actionState={busy
            ? "loading"
            : runNotice?.type === "error" || error
              ? "failure"
              : evidence
                ? "success"
                : "idle"}
          actionLabel="图片识别"
          actionError={error ?? run?.failure_reason}
        />
        {error ? <ErrorState error={error} onRetry={() => void loadDraft()} /> : null}
        {!draft && !error ? <LoadingState /> : null}
        {draft ? (
          <Card title="1. 发起异步识别" extra={<StatusTag status={draft.status} version={draft.version} />}>
            <Space.Compact block>
              <Input
                aria-label="待识别媒体 ID"
                value={mediaId}
                status={mediaIdError ? "error" : undefined}
                onChange={(event) => {
                  setMediaId(event.target.value);
                  setMediaIdError(undefined);
                }}
                placeholder="上传完成后自动回填 media_id"
              />
              <Button
                type="primary"
                loading={busy}
                disabled={!mediaId.trim()}
                onClick={() => void beginRecognition()}
              >
                开始识别
              </Button>
            </Space.Compact>
            {mediaIdError ? (
              <Typography.Text type="danger">{mediaIdError}</Typography.Text>
            ) : !mediaId.trim() ? (
              <Typography.Text type="secondary">
                尚未选择媒体。请先
                <Link href={`/incidents/drafts/${draftId}`}>返回故障草稿上传文件</Link>
                ，待扫描状态变为 CLEAN 后再开始识别。
              </Typography.Text>
            ) : (
              <Typography.Text type="secondary">
                已接收上传媒体：<Typography.Text code>{mediaId.trim()}</Typography.Text>
              </Typography.Text>
            )}
            <Checkbox
              checked={temporalAnalysis}
              onChange={(event) => setTemporalAnalysis(event.target.checked)}
            >
              分析机械卡顿、指示灯闪烁和动作顺序（仅视频，最多 4 个受限关键帧窗口）
            </Checkbox>
            {run ? <FactGrid facts={[["识别运行", run.recognition_run_id], ["工作流", run.workflow_id], ["状态", run.status]]} /> : null}
          </Card>
        ) : null}
        <Card
          title="2. 查看并确认证据"
          extra={(
            <Button
              aria-label={run && ["QUEUED", "RUNNING"].includes(run.status)
                ? "刷新识别状态"
                : "读取最新证据"}
              loading={busy}
              onClick={() => void refreshEvidence()}
            >
              {run && ["QUEUED", "RUNNING"].includes(run.status)
                ? "刷新识别状态"
                : "读取最新证据"}
            </Button>
          )}
        >
          {runNotice ? (
            <Alert
              type={runNotice.type}
              showIcon
              message={runNotice.message}
              description={runNotice.description}
            />
          ) : null}
          {evidence ? (
            <div className="page-stack">
              <FactGrid facts={[["证据包", evidence.bundle_id], ["状态", evidence.status], ["来源类型", evidence.source_type], ["来源哈希", evidence.source_sha256], ["版本", evidence.version]]} />
              {(evidence.security_findings ?? []).length ? (
                <Card size="small" title="内容安全发现">
                  <div className="page-stack">
                    <Alert
                      type="warning"
                      showIcon
                      message="命中内容不能作为指令或事实直接进入自动化"
                      description="请按来源拒绝候选，或仅对 OCR/ASR 输入已核实的安全修正文；服务端会再次执行同一策略。"
                    />
                    <Space wrap>
                      <Tag color="red">{`策略 ${evidence.security_policy_version ?? "unknown"}`}</Tag>
                      {(evidence.security_findings ?? []).map((finding) => (
                        <Tag key={`${finding.source_type}:${finding.source_id}:${finding.pattern_id}`} color="orange">
                          {`${finding.source_type} ${finding.source_id} · ${finding.pattern_id} · ${finding.category}`}
                        </Tag>
                      ))}
                    </Space>
                    {evidence.ocr_blocks
                      .filter((block) => unsafeSource(evidence, "OCR_BLOCK", block.block_id))
                      .map((block) => {
                        const decision = ocrBlockDecisions[block.block_id];
                        return (
                          <div className="page-stack" key={block.block_id}>
                            <Space wrap>
                              <Button
                                aria-label={`安全修正 OCR ${block.block_id}`}
                                type={decision?.disposition === "CORRECTED" ? "primary" : "default"}
                                onClick={() => setOcrBlockDecisions((current) => ({
                                  ...current,
                                  [block.block_id]: {
                                    disposition: "CORRECTED",
                                    corrected_text: "",
                                  },
                                }))}
                              >安全修正</Button>
                              <Button
                                aria-label={`拒绝不安全 OCR ${block.block_id}`}
                                danger
                                type={decision?.disposition === "REJECTED" ? "primary" : "default"}
                                onClick={() => setOcrBlockDecisions((current) => ({
                                  ...current,
                                  [block.block_id]: {
                                    disposition: "REJECTED",
                                    corrected_text: null,
                                  },
                                }))}
                              >拒绝来源</Button>
                            </Space>
                            {decision?.disposition === "CORRECTED" ? (
                              <Input.TextArea
                                aria-label={`安全修正文 OCR ${block.block_id}`}
                                value={decision.corrected_text ?? ""}
                                placeholder="只填写现场核实后的工业事实，不复制命中内容"
                                onChange={(event) => setOcrBlockDecisions((current) => ({
                                  ...current,
                                  [block.block_id]: {
                                    disposition: "CORRECTED",
                                    corrected_text: event.target.value,
                                  },
                                }))}
                              />
                            ) : null}
                          </div>
                        );
                      })}
                  </div>
                </Card>
              ) : (
                <Alert type="success" showIcon message="派生文字未命中内容安全策略" />
              )}
              {(evidence.qr_codes ?? []).length ? (
                <Card size="small" title="二维码参考证据（必须逐项复核）">
                  <Alert
                    type="warning"
                    showIcon
                    message="二维码内容仅供人工参考"
                    description="链接载荷不会打开、预览或请求；二维码原文及修正文不会进入实体抽取或 Agent 上下文。"
                  />
                  <Space wrap style={{ margin: "12px 0" }}>
                    <Tag color="purple">{`解码器 ${evidence.processor_versions.qr ?? "unknown"}`}</Tag>
                    <Tag>{`候选 ${(evidence.qr_codes ?? []).length}`}</Tag>
                  </Space>
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
                                key={`accept-qr-${candidate.candidate_id}`}
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
                              key={`reject-qr-${candidate.candidate_id}`}
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
                              key={`correct-qr-${candidate.candidate_id}`}
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
                              {unsafe ? <Tag color="red">内容安全命中</Tag> : <Tag color="green">未命中安全规则</Tag>}
                              {candidate.source_frame_id ? <Tag>{`帧 ${candidate.source_frame_id}`}</Tag> : null}
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
                              <Input.TextArea
                                aria-label={`二维码安全修正文 ${candidate.candidate_id}`}
                                value={decision.corrected_text ?? ""}
                                placeholder="只填写人工核实后的安全参考文本"
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
              <List
                header="结构化实体"
                dataSource={evidence.extracted_entities}
                renderItem={(item) => (
                  <List.Item extra={<Progress type="circle" size={46} percent={Math.round(item.confidence * 100)} />}>
                    <List.Item.Meta
                      title={`${item.entity_type}：${item.normalized_value}`}
                      description={(
                        <div className="page-stack">
                          <span>{`${item.validation_status} · OCR 块 ${item.source_block_id}`}</span>
                          {item.validation_status !== "VALID" ? (
                            <Input
                              aria-label={`修订实体 ${item.entity_id}`}
                              disabled={evidence.status === "CONFIRMED"}
                              value={entityEdits[item.entity_id] ?? item.original_value}
                              onChange={(event) => setEntityEdits((current) => ({
                                ...current,
                                [item.entity_id]: event.target.value,
                              }))}
                            />
                          ) : null}
                        </div>
                      )}
                    />
                  </List.Item>
                )}
              />
              {evidence.video_keyframes.length ? (
                <>
                  <List
                    header="视频事件时间线（视觉事件均为待复核候选）"
                    dataSource={evidence.video_events}
                    renderItem={(item) => (
                  <List.Item
                    extra={(
                      <Space>
                        {isReviewableVideoEvent(item.event_type) ? (
                          <Button
                            danger={!rejectedVideoEvents[item.event_id]}
                            disabled={evidence.status === "CONFIRMED" || (
                              unsafeSource(evidence, "VIDEO_EVENT", item.event_id)
                              && Boolean(rejectedVideoEvents[item.event_id])
                            )}
                            onClick={() => setRejectedVideoEvents((current) => ({
                              ...current,
                              [item.event_id]: !current[item.event_id],
                            }))}
                          >
                            {rejectedVideoEvents[item.event_id]
                              ? unsafeSource(evidence, "VIDEO_EVENT", item.event_id)
                                ? "已拒绝不安全事件"
                                : "恢复并接受"
                              : "拒绝事件"}
                          </Button>
                        ) : null}
                        <Progress type="circle" size={46} percent={Math.round(item.confidence * 100)} />
                      </Space>
                    )}
                      >
                        <List.Item.Meta
                          title={(
                            <Space wrap>
                              <Tag color={item.event_type === "visual_candidate" ? "orange" : "blue"}>
                                {videoEventType(item.event_type)}
                              </Tag>
                              <span>{`${formatTimestamp(item.start_ms)}–${formatTimestamp(item.end_ms)}`}</span>
                            </Space>
                          )}
                          description={(
                            <div className="page-stack">
                              <span>{item.description}</span>
                              {item.model_release_id ? (
                                <Typography.Text type="secondary">
                                  {`时序 VLM Release：${item.model_release_id}`}
                                </Typography.Text>
                              ) : null}
                              <Space wrap>
                                {item.keyframe_ids.map((frameId) => (
                                  <a key={frameId} href={`#video-frame-${frameId}`}>{frameId}</a>
                                ))}
                                {item.finding_ids.length ? (
                                  <Typography.Text type="secondary">
                                    {`${item.finding_ids.length} 个候选区域`}
                                  </Typography.Text>
                                ) : null}
                              </Space>
                            </div>
                          )}
                        />
                      </List.Item>
                    )}
                  />
                  <List
                    header={`视频关键帧与候选区域（${evidence.video_keyframes.length}）`}
                    dataSource={evidence.video_keyframes}
                    renderItem={(item) => {
                      const frameFindings = evidence.visual_findings.filter(
                        (finding) => finding.source_frame_id === item.frame_id,
                      );
                      const frameQrCodes = (evidence.qr_codes ?? []).filter(
                        (candidate) => candidate.source_frame_id === item.frame_id,
                      );
                      return (
                        <List.Item id={`video-frame-${item.frame_id}`}>
                          <Space align="start" wrap>
                            <div style={{ position: "relative", width: 320, maxWidth: "100%" }}>
                              <Image
                                width="100%"
                                src={backendMediaUrl(item.image_url)}
                                alt={`视频关键帧 ${formatTimestamp(item.timestamp_ms)}`}
                                preview={{ mask: "查看大图" }}
                              />
                              {frameFindings.map((finding) => (
                                <span
                                  key={finding.finding_id}
                                  aria-label={`候选区域 ${finding.label}`}
                                  title={`${finding.label} · ${Math.round(finding.confidence * 100)}%`}
                                  style={{
                                    position: "absolute",
                                    left: `${finding.bbox.x * 100}%`,
                                    top: `${finding.bbox.y * 100}%`,
                                    width: `${finding.bbox.width * 100}%`,
                                    height: `${finding.bbox.height * 100}%`,
                                    border: `2px solid ${rejectedFindings[finding.finding_id] ? "#8c8c8c" : "#fa8c16"}`,
                                    boxShadow: "0 0 0 1px rgba(255,255,255,0.7)",
                                    pointerEvents: "none",
                                  }}
                                />
                              ))}
                              {frameQrCodes.map((candidate) => (
                                <span
                                  key={candidate.candidate_id}
                                  aria-label={`二维码区域 ${candidate.candidate_id}`}
                                  title={`二维码 · ${candidate.payload_kind}`}
                                  style={{
                                    position: "absolute",
                                    left: `${candidate.bbox.x * 100}%`,
                                    top: `${candidate.bbox.y * 100}%`,
                                    width: `${candidate.bbox.width * 100}%`,
                                    height: `${candidate.bbox.height * 100}%`,
                                    border: "2px solid #722ed1",
                                    boxShadow: "0 0 0 1px rgba(255,255,255,0.7)",
                                    pointerEvents: "none",
                                  }}
                                />
                              ))}
                            </div>
                            <div className="page-stack">
                              <Typography.Text strong>
                                {`${formatTimestamp(item.timestamp_ms)} · ${samplingReason(item.sampling_reason)}`}
                              </Typography.Text>
                              <Typography.Text type="secondary">{item.frame_id}</Typography.Text>
                              {frameFindings.map((finding) => (
                                <Tag
                                  key={finding.finding_id}
                                  color={rejectedFindings[finding.finding_id] ? "default" : "orange"}
                                >
                                  {`${finding.label} · ${Math.round(finding.confidence * 100)}%`}
                                </Tag>
                              ))}
                            </div>
                          </Space>
                        </List.Item>
                      );
                    }}
                  />
                </>
              ) : null}
              {evidence.asr_segments.length ? (
                <List
                  header="视频音轨转写（必须逐段接受、修订或拒绝）"
                  dataSource={evidence.asr_segments}
                  renderItem={(item) => (
                    <List.Item
                      extra={(
                        <Button
                          danger={!rejectedSegments[item.segment_id]}
                          disabled={evidence.status === "CONFIRMED" || (
                            unsafeSource(evidence, "ASR_SEGMENT", item.segment_id)
                            && Boolean(rejectedSegments[item.segment_id])
                          )}
                          onClick={() => setRejectedSegments((current) => ({
                            ...current,
                            [item.segment_id]: !current[item.segment_id],
                          }))}
                        >
                          {rejectedSegments[item.segment_id]
                            ? unsafeSource(evidence, "ASR_SEGMENT", item.segment_id)
                              ? "已拒绝不安全转写"
                              : "恢复并接受"
                            : "拒绝此段"}
                        </Button>
                      )}
                    >
                      <List.Item.Meta
                        title={(
                          <Space wrap>
                            <span>{`${item.source_audio_track_id ?? "音轨"} · ${formatTimestamp(item.start_ms)}–${formatTimestamp(item.end_ms)} · 置信度 ${Math.round(item.confidence * 100)}%`}</span>
                            {item.hotword_profile_id ? (
                              <Tag color="purple">{`热词 ${item.hotword_profile_id}`}</Tag>
                            ) : null}
                          </Space>
                        )}
                        description={(
                          <div className="page-stack">
                            {item.entity_candidates.length ? (
                              <Alert
                                type={item.entity_candidates.some(
                                  (candidate) => candidate.requires_confirmation,
                                ) ? "warning" : "info"}
                                showIcon
                                message="工业实体候选"
                                description={(
                                  <Space wrap>
                                    {item.entity_candidates.map((candidate) => (
                                      <Tag
                                        key={candidate.entity_id}
                                        color={candidate.requires_confirmation ? "orange" : "blue"}
                                      >
                                        {`${asrEntityType(candidate.entity_type)}：${candidate.normalized_value}${candidate.requires_confirmation ? "（请核对）" : ""}`}
                                      </Tag>
                                    ))}
                                  </Space>
                                )}
                              />
                            ) : null}
                            <Input.TextArea
                              aria-label={`修订音轨转写 ${item.segment_id}`}
                              disabled={
                                evidence.status === "CONFIRMED"
                                || Boolean(rejectedSegments[item.segment_id])
                              }
                              value={transcriptEdits[item.segment_id] ?? item.text}
                              onChange={(event) => setTranscriptEdits((current) => ({
                                ...current,
                                [item.segment_id]: event.target.value,
                              }))}
                            />
                          </div>
                        )}
                      />
                    </List.Item>
                  )}
                />
              ) : null}
              <List
                header="视觉发现"
                dataSource={evidence.visual_findings}
                renderItem={(item) => (
                  <List.Item
                    extra={(
                      <Button
                        danger={!rejectedFindings[item.finding_id]}
                        disabled={evidence.status === "CONFIRMED" || (
                          unsafeSource(evidence, "VISUAL_FINDING", item.finding_id)
                          && Boolean(rejectedFindings[item.finding_id])
                        )}
                        onClick={() => setRejectedFindings((current) => ({
                          ...current,
                          [item.finding_id]: !current[item.finding_id],
                        }))}
                      >
                        {rejectedFindings[item.finding_id] ? "恢复并接受" : "拒绝发现"}
                      </Button>
                    )}
                  >
                    {item.label} · {item.evidence_level} · {item.source_frame_id ? `${item.source_frame_id} · ` : ""}{item.description}
                  </List.Item>
                )}
              />
              <Button
                type="primary"
                loading={busy}
                disabled={evidence.status === "CONFIRMED" || !securityReviewComplete(
                  evidence,
                  ocrBlockDecisions,
                  transcriptEdits,
                  rejectedSegments,
                  rejectedFindings,
                  rejectedVideoEvents,
                  qrCodeDecisions,
                )}
                onClick={() => void approveEvidence()}
              >
                确认当前证据版本
              </Button>
            </div>
          ) : <Typography.Text type="secondary">识别 Worker 完成后在此读取服务端证据。</Typography.Text>}
        </Card>
        <Card title="3. 创建正式 Incident">
          <Button type="primary" loading={busy} disabled={!draft || evidence?.status !== "CONFIRMED"} onClick={() => void submit()}>
            按确认版本提交报障
          </Button>
          {incident ? (
            <Alert type="success" showIcon message={`Incident ${incident.incident_id} 已创建`} description={<Link href={`/incidents/${incident.incident_id}`}>进入诊断工作台</Link>} />
          ) : null}
        </Card>
      </div>
    </AppShell>
  );
}

function recognitionFailureDescription(reason: string | null | undefined): string {
  if (reason === "multimodal_provider_unavailable") {
    return "多模态识别组件暂不可用。请重新点击“开始识别”；若仍失败，请检查识别 Worker 运行状态。";
  }
  if (reason === "recognition_state_changed") {
    return "识别期间草稿、媒体或设备状态发生变化，请确认页面数据后重新发起识别。";
  }
  return reason
    ? `失败原因：${reason}。请重新发起识别。`
    : "请重新发起识别；若仍失败，请检查识别 Worker 运行状态。";
}

function formatTimestamp(value: number) {
  const seconds = value / 1_000;
  return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function samplingReason(value: string) {
  if (value === "first_frame") return "首帧";
  if (value === "scene_change") return "场景变化";
  return "周期采样";
}

function videoEventType(value: string) {
  if (value === "visual_candidate") return "视觉候选";
  if (value === "motion_candidate") return "运动异常候选";
  if (value === "signal_pattern_candidate") return "信号变化候选";
  if (value === "action_sequence_candidate") return "动作顺序候选";
  if (value === "scene_change") return "场景变化";
  return "采样观察";
}

function asrEntityType(value: string) {
  if (value === "alarm_code") return "报警码";
  if (value === "serial_number") return "设备/序列号";
  if (value === "part_number") return "部件号";
  if (value === "measurement") return "测量值";
  return value;
}

function unsafeSource(
  evidence: EvidenceBundle,
  sourceType: string,
  sourceId: string,
): boolean {
  return (evidence.security_findings ?? []).some((finding) => (
    finding.source_type === sourceType && finding.source_id === sourceId
  ));
}

function securityReviewComplete(
  evidence: EvidenceBundle,
  ocrDecisions: Record<string, IncidentOcrBlockDecision>,
  transcriptEdits: Record<string, string>,
  rejectedSegments: Record<string, boolean>,
  rejectedFindings: Record<string, boolean>,
  rejectedVideoEvents: Record<string, boolean>,
  qrCodeDecisions: Record<string, IncidentQrCodeDecision>,
): boolean {
  const unsafe = new Set(
    (evidence.security_findings ?? []).map((finding) => (
      `${finding.source_type}:${finding.source_id}`
    )),
  );
  return [...unsafe].every((identity) => {
    const separator = identity.indexOf(":");
    const sourceType = identity.slice(0, separator);
    const sourceId = identity.slice(separator + 1);
    if (sourceType === "OCR_BLOCK") {
      const decision = ocrDecisions[sourceId];
      return decision?.disposition === "REJECTED" || (
        decision?.disposition === "CORRECTED"
        && Boolean(decision.corrected_text?.trim())
      );
    }
    if (sourceType === "ASR_SEGMENT") {
      const source = evidence.asr_segments.find((item) => item.segment_id === sourceId);
      const correction = transcriptEdits[sourceId]?.trim();
      return Boolean(rejectedSegments[sourceId]) || Boolean(
        correction && correction !== source?.text.trim(),
      );
    }
    if (sourceType === "VISUAL_FINDING") return Boolean(rejectedFindings[sourceId]);
    if (sourceType === "VIDEO_EVENT") return Boolean(rejectedVideoEvents[sourceId]);
    return false;
  }) && (evidence.qr_codes ?? []).every((candidate) => {
    const decision = qrCodeDecisions[candidate.candidate_id];
    if (!decision) return false;
    if (candidate.security_findings.length > 0 && decision.disposition === "ACCEPTED") {
      return false;
    }
    return decision.disposition !== "CORRECTED" || Boolean(decision.corrected_text?.trim());
  });
}

function isReviewableVideoEvent(eventType: string): boolean {
  return [
    "visual_candidate",
    "motion_candidate",
    "signal_pattern_candidate",
    "action_sequence_candidate",
  ].includes(eventType);
}

function backendMediaUrl(value: string) {
  return value.startsWith("/api/v1/") ? `/api/backend${value}` : "";
}
