"use client";

import { Alert, Button, Card, Checkbox, List, Space, Tag, Typography } from "antd";
import { useCallback, useEffect, useState } from "react";

import { RealtimeAudioPanel } from "@/components/m3/RealtimeAudioPanel";
import {
  ApiClientError,
  type FieldVoiceGuidance,
  getFieldVoiceGuidance,
  synthesizeFieldVoiceGuidance,
} from "@/lib/api/client";

export function FieldVoiceGuidancePanel({ workOrderId }: { workOrderId: string }) {
  const [guidance, setGuidance] = useState<FieldVoiceGuidance>();
  const [acknowledged, setAcknowledged] = useState(false);
  const [synthesisId, setSynthesisId] = useState<string>();
  const [speechRelease, setSpeechRelease] = useState<string>();
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>();

  const load = useCallback(async () => {
    setLoading(true);
    setError(undefined);
    try {
      if (!navigator.onLine) throw new Error("实时语音需要联网，当前已降级到离线工作包和现场事实队列。");
      const result = await getFieldVoiceGuidance(workOrderId);
      setGuidance(result.guidance);
      setAcknowledged(false);
      setSynthesisId(undefined);
      setSpeechRelease(undefined);
    } catch (caught) {
      setGuidance(undefined);
      setError(caught);
    } finally {
      setLoading(false);
    }
  }, [workOrderId]);

  useEffect(() => { void load(); }, [load]);

  async function synthesize() {
    const diagnosis = guidance?.diagnosis;
    if (!guidance || !diagnosis || !acknowledged) return;
    setBusy(true);
    setError(undefined);
    try {
      const result = await synthesizeFieldVoiceGuidance(
        workOrderId,
        guidance.work_order_version,
        diagnosis.diagnosis_run_id,
        true,
        crypto.randomUUID(),
      );
      setSynthesisId(result.speech.synthesis_id);
      setSpeechRelease(result.speech.resolved_release_id);
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  if (loading) {
    return <Card title="现场实时语音指导" loading />;
  }

  if (!guidance) {
    return (
      <Card title="现场实时语音指导">
        <Alert
          type="warning"
          showIcon
          title="实时语音当前不可用"
          description={message(error)}
          action={<Button onClick={() => void load()}>重新检查</Button>}
        />
      </Card>
    );
  }

  const diagnosis = guidance.diagnosis;
  const canStart = guidance.legal_actions.includes("START_REALTIME_VOICE") && diagnosis;
  const canSpeak = guidance.legal_actions.includes("SYNTHESIZE_VOICE_GUIDANCE") && diagnosis;
  return (
    <Card
      title="现场实时语音指导"
      extra={<Tag color="blue">工单版本 {guidance.work_order_version}</Tag>}
    >
      <div className="page-stack">
        <Alert
          type="info"
          showIcon
          title="语音仅用于受控指导"
          description="转写必须逐段人工接受或拒绝；不会自动形成现场事实、重新诊断、完工、审批或设备动作。连接失败时继续使用文字、离线工作包和既有现场事实队列。"
        />
        {!diagnosis ? (
          <Alert type="warning" showIcon title="当前没有可用于实时指导的诊断" />
        ) : (
          <Card
            size="small"
            type="inner"
            title="服务端选定的最新诊断"
            extra={<Tag>{diagnosis.status}</Tag>}
          >
            <div className="page-stack">
              <Typography.Text>{diagnosis.conclusion ?? "诊断尚未形成可展示结论"}</Typography.Text>
              {diagnosis.next_checks.length ? (
                <List
                  size="small"
                  header="建议检查项"
                  dataSource={diagnosis.next_checks}
                  renderItem={(item) => <List.Item>{item}</List.Item>}
                />
              ) : null}
            </div>
          </Card>
        )}
        {canSpeak ? (
          <Card size="small" type="inner" title="安全语音播报">
            <div className="page-stack">
              <Alert
                type="warning"
                showIcon
                title="播报前必须完整阅读"
                description={guidance.safety_warning}
              />
              <Checkbox
                checked={acknowledged}
                onChange={(event) => setAcknowledged(event.target.checked)}
              >
                我已阅读完整安全警告，并确认语音不构成自动维修授权
              </Checkbox>
              <Space wrap>
                <Button
                  type="primary"
                  loading={busy}
                  disabled={!acknowledged}
                  onClick={() => void synthesize()}
                >
                  生成受控诊断语音
                </Button>
                {speechRelease ? <Tag color="green">TTS Release：{speechRelease}</Tag> : null}
              </Space>
            </div>
          </Card>
        ) : diagnosis ? (
          <Alert
            type="warning"
            showIcon
            title="当前诊断不能语音播报"
            description="诊断仍需补充信息或存在安全不确定性；可以继续实时转写，但服务端不会生成 TTS。"
          />
        ) : null}
        {error ? <Alert type="error" showIcon title={message(error)} /> : null}
        {canStart ? (
          <RealtimeAudioPanel
            incidentId={guidance.incident_id}
            diagnosisRunId={diagnosis.diagnosis_run_id}
            diagnosisVersion={diagnosis.version}
            workOrderId={guidance.work_order_id}
            workOrderVersion={guidance.work_order_version}
            speechSynthesisId={synthesisId}
            allowReanalysis={false}
          />
        ) : null}
      </div>
    </Card>
  );
}

function message(error: unknown): string {
  if (error instanceof ApiClientError) return `${error.message}（${error.code}）`;
  return error instanceof Error ? error.message : "无法读取现场语音指导，请继续使用文字模式。";
}
