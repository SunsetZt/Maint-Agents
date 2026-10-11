"use client";

import { Alert, Button, Card, Tag, Typography } from "antd";
import Link from "next/link";
import { useParams } from "next/navigation";
import { FormEvent, useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  ApiClientError,
  type IncidentDraft,
  type MediaStatus,
  getIncidentDraft,
  getMediaStatus,
  updateIncidentDraft,
  uploadDraftMedia,
} from "@/lib/api/client";

export default function IncidentDraftPage() {
  const params = useParams<{ draftId: string }>();
  const draftId = params.draftId;
  const [draft, setDraft] = useState<IncidentDraft>();
  const [description, setDescription] = useState("");
  const [media, setMedia] = useState<MediaStatus>();
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const response = await getIncidentDraft(draftId);
      setDraft(response.draft);
      setDescription(response.draft.description);
      setRequestId(response.requestId);
    } catch (caught) {
      setError(caught);
    }
  }, [draftId]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!media || media.scan_state !== "PENDING") return;
    const timer = window.setTimeout(() => {
      void getMediaStatus(media.media_id)
        .then((response) => {
          setMedia(response.media);
          setRequestId(response.requestId);
        })
        .catch(setError);
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [media]);

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!draft || !description.trim()) return;
    setSaving(true);
    setError(undefined);
    try {
      const response = await updateIncidentDraft(
        draftId,
        description.trim(),
        draft.version,
      );
      setDraft(response.draft);
      setRequestId(response.requestId);
    } catch (caught) {
      setError(caught);
    } finally {
      setSaving(false);
    }
  }

  async function upload(file: File | undefined) {
    if (!file) return;
    setError(undefined);
    try {
      const response = await uploadDraftMedia(draftId, file);
      setMedia(response.media);
      setRequestId(response.requestId);
    } catch (caught) {
      setError(caught);
    }
  }

  const conflict = error instanceof ApiClientError && error.code === "version_conflict";
  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>故障草稿</Typography.Title>
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {conflict ? (
          <Alert
            type="warning"
            message="草稿已被其他客户端更新"
            description="已保留服务端胜出版本，请重新读取后再编辑。"
            action={<Button onClick={() => void load()}>重新读取</Button>}
          />
        ) : null}
        {!draft && !error ? <LoadingState /> : null}
        {draft ? (
          <>
            <Card
              title={draft.draft_id}
              extra={<Tag color="blue">{draft.status} · v{draft.version}</Tag>}
            >
              <form className="form-grid" onSubmit={save}>
                <label htmlFor="draft-description">
                  故障描述
                  <textarea
                    id="draft-description"
                    rows={6}
                    value={description}
                    onChange={(event) => setDescription(event.target.value)}
                  />
                </label>
                <Button type="primary" htmlType="submit" loading={saving}>
                  按当前版本保存
                </Button>
              </form>
            </Card>
            <Card title="隔离媒体上传">
              <div className="page-stack">
                <Typography.Text type="secondary">
                  PNG、JPEG、PDF、WAV、FLAC、MP4 或 WebM 会先进入 quarantine；只有扫描清洁并复核哈希后才可读取。
                </Typography.Text>
                <input
                  aria-label="选择待扫描媒体"
                  type="file"
                  accept="image/png,image/jpeg,application/pdf,audio/wav,audio/flac,video/mp4,video/webm"
                  onChange={(event) => void upload(event.target.files?.[0])}
                />
                {media ? (
                  <>
                    <Alert
                      type={media.scan_state === "CLEAN" ? "success" : "info"}
                      message={`扫描状态：${media.scan_state}`}
                      description="页面只显示服务端状态，不读取未扫描媒体正文。"
                    />
                    <Typography.Text>
                      媒体 ID：
                      <Typography.Text code copyable>{media.media_id}</Typography.Text>
                    </Typography.Text>
                    {media.scan_state === "CLEAN" ? (
                      <ContentCredentialNotice media={media} />
                    ) : null}
                  </>
                ) : null}
              </div>
            </Card>
            <Link href={`/incidents/drafts/${draftId}/timeline`}>
              查看服务端只读时间线
            </Link>
            {media?.scan_state === "CLEAN" ? (
              <Link
                href={`/incidents/drafts/${draftId}/recognition?media_id=${encodeURIComponent(media.media_id)}`}
              >
                进入多模态识别与证据确认
              </Link>
            ) : (
              <Typography.Text type="secondary">
                请先上传媒体并等待安全扫描通过，再进入多模态识别。
              </Typography.Text>
            )}
          </>
        ) : null}
        {requestId ? <span className="request-id">请求标识：{requestId}</span> : null}
      </div>
    </AppShell>
  );
}

function ContentCredentialNotice({ media }: { media: MediaStatus }) {
  const status = media.content_credential_status;
  if (status === "PRESENT_TRUSTED") {
    return (
      <Alert
        type="success"
        message="内容来源凭证已通过信任验证"
        description="C2PA 处理链与签名可信，但它仍然只是来源信号，不单独证明现场内容真实。"
      />
    );
  }
  if (status === "PRESENT_VALID" || status === "PRESENT_UNVERIFIED") {
    return (
      <Alert
        type="info"
        message="检测到内容来源凭证"
        description="凭证结构与签名可读取，但当前信任锚未确认签发者；请结合上传者、设备和现场记录判断。"
      />
    );
  }
  if (status === "PRESENT_INVALID") {
    return (
      <Alert
        type="warning"
        message="内容来源凭证验证异常"
        description="凭证异常需要人工复核，但不能据此直接判定媒体伪造。"
      />
    );
  }
  if (status === "ABSENT") {
    return (
      <Alert
        type="info"
        message="未检测到内容来源凭证"
        description="缺少 C2PA 凭证不等于媒体伪造，平台仍保留文件哈希、上传者、时间和后续处理链。"
      />
    );
  }
  return (
    <Alert
      type="info"
      message={`内容来源检查：${status}`}
      description="当前无法形成可信凭证结论，不影响病毒扫描结果，也不会自动改变真假判断。"
    />
  );
}
