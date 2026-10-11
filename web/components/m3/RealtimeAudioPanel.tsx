"use client";

import { Alert, Button, Card, Input, Space, Tag, Typography } from "antd";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiClientError,
  type DiagnosisRun,
  type RealtimeSessionGrant,
  type RealtimeTranscript,
  confirmRealtimeTranscript,
  createFieldRealtimeSession,
  createRealtimeSession,
  endRealtimeSession,
  heartbeatRealtimeSession,
  reconnectRealtimeSession,
  reanalyzeDiagnosisWithTranscript,
} from "@/lib/api/client";

type ConnectionStatus =
  | "IDLE"
  | "REQUESTING_MEDIA"
  | "SIGNALING"
  | "CONNECTED"
  | "REBINDING"
  | "RECONNECTING"
  | "FAILED";

interface SignalingMessage {
  type?: string;
  sdp?: string;
  candidate?: RTCIceCandidateInit;
  message?: string;
  reason?: string;
  synthesis_id?: string;
  transcript?: RealtimeTranscript;
  sequence?: number;
}

export function RealtimeAudioPanel({
  incidentId,
  diagnosisRunId,
  diagnosisVersion,
  speechSynthesisId,
  onDiagnosisCreated,
  workOrderId,
  workOrderVersion,
  allowReanalysis = true,
}: {
  incidentId: string;
  diagnosisRunId?: string;
  diagnosisVersion?: number;
  speechSynthesisId?: string;
  onDiagnosisCreated?: (diagnosis: DiagnosisRun) => void;
  workOrderId?: string;
  workOrderVersion?: number;
  allowReanalysis?: boolean;
}) {
  const [status, setStatus] = useState<ConnectionStatus>("IDLE");
  const [error, setError] = useState<string>();
  const [ttsPlaying, setTtsPlaying] = useState(false);
  const [ttsNotice, setTtsNotice] = useState<string>();
  const [session, setSession] = useState<RealtimeSessionGrant>();
  const [transcripts, setTranscripts] = useState<RealtimeTranscript[]>([]);
  const remoteAudioRef = useRef<HTMLAudioElement>(null);
  const sessionRef = useRef<RealtimeSessionGrant | undefined>(undefined);
  const localStreamRef = useRef<MediaStream | undefined>(undefined);
  const peerRef = useRef<RTCPeerConnection | undefined>(undefined);
  const socketRef = useRef<WebSocket | undefined>(undefined);
  const heartbeatRef = useRef<number | undefined>(undefined);
  const reconnectTimerRef = useRef<number | undefined>(undefined);
  const connectionTimeoutRef = useRef<number | undefined>(undefined);
  const generationRef = useRef(0);
  const reconnectingRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const intentionalStopRef = useRef(false);
  const rebindGenerationRef = useRef(0);
  const reconnectRef = useRef<() => void>(() => undefined);
  const playedSynthesisRef = useRef<string | undefined>(undefined);

  const createSession = useCallback(async () => {
    if (workOrderId !== undefined && workOrderVersion !== undefined) {
      if (!diagnosisRunId) throw new Error("当前工单没有可用于实时语音的诊断。");
      return (await createFieldRealtimeSession(
        workOrderId,
        workOrderVersion,
        diagnosisRunId,
      )).session;
    }
    return (await createRealtimeSession(incidentId, diagnosisRunId)).session;
  }, [diagnosisRunId, incidentId, workOrderId, workOrderVersion]);

  const clearConnection = useCallback((keepMedia: boolean) => {
    generationRef.current += 1;
    if (heartbeatRef.current !== undefined) window.clearInterval(heartbeatRef.current);
    if (reconnectTimerRef.current !== undefined) window.clearTimeout(reconnectTimerRef.current);
    if (connectionTimeoutRef.current !== undefined) window.clearTimeout(connectionTimeoutRef.current);
    heartbeatRef.current = undefined;
    reconnectTimerRef.current = undefined;
    connectionTimeoutRef.current = undefined;
    socketRef.current?.close();
    peerRef.current?.close();
    socketRef.current = undefined;
    peerRef.current = undefined;
    if (!keepMedia) {
      localStreamRef.current?.getTracks().forEach((track) => track.stop());
      localStreamRef.current = undefined;
      if (remoteAudioRef.current) remoteAudioRef.current.srcObject = null;
    }
  }, []);

  const connect = useCallback(
    (grant: RealtimeSessionGrant, stream: MediaStream) => {
      const generation = generationRef.current + 1;
      clearConnection(true);
      generationRef.current = generation;
      setStatus(reconnectAttemptsRef.current > 0 ? "RECONNECTING" : "SIGNALING");
      const pendingCandidates: RTCIceCandidateInit[] = [];
      const peer = new RTCPeerConnection({
        iceServers: grant.ice_servers.map((server) => ({
          urls: server.urls,
          username: server.username ?? undefined,
          credential: server.credential ?? undefined,
        })),
      });
      const socket = new WebSocket(grant.signaling_url);
      peerRef.current = peer;
      socketRef.current = socket;
      stream.getTracks().forEach((track) => peer.addTrack(track, stream));

      peer.ontrack = (event) => {
        if (generation !== generationRef.current || !remoteAudioRef.current) return;
        remoteAudioRef.current.srcObject = event.streams[0] ?? new MediaStream([event.track]);
        void remoteAudioRef.current.play().catch(() => undefined);
      };
      peer.onicecandidate = (event) => {
        if (event.candidate && socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "candidate", candidate: event.candidate }));
        }
      };
      peer.onconnectionstatechange = () => {
        if (generation !== generationRef.current) return;
        if (peer.connectionState === "connected") {
          if (connectionTimeoutRef.current !== undefined) {
            window.clearTimeout(connectionTimeoutRef.current);
            connectionTimeoutRef.current = undefined;
          }
          reconnectAttemptsRef.current = 0;
          reconnectingRef.current = false;
          setStatus("CONNECTED");
        } else if (["failed", "disconnected"].includes(peer.connectionState)) {
          reconnectRef.current();
        }
      };
      socket.onopen = () => {
        if (generation !== generationRef.current) return;
        socket.send(JSON.stringify({
          type: "authenticate",
          session_id: grant.session_id,
          access_token: grant.access_token,
        }));
      };
      socket.onmessage = (event) => {
        if (generation !== generationRef.current || typeof event.data !== "string") return;
        let message: SignalingMessage;
        try {
          message = JSON.parse(event.data) as SignalingMessage;
        } catch {
          setError("媒体服务返回了无法识别的信令消息。");
          return;
        }
        if (message.type === "authenticated") {
          void peer.createOffer({ offerToReceiveAudio: true }).then(async (offer) => {
            await peer.setLocalDescription(offer);
            socket.send(JSON.stringify({ type: "offer", sdp: offer.sdp }));
          }).catch(() => reconnectRef.current());
        } else if (message.type === "answer" && message.sdp) {
          void peer.setRemoteDescription({ type: "answer", sdp: message.sdp }).then(async () => {
            for (const candidate of pendingCandidates.splice(0)) {
              await peer.addIceCandidate(candidate);
            }
          }).catch(() => reconnectRef.current());
        } else if (message.type === "candidate" && message.candidate) {
          if (peer.remoteDescription) void peer.addIceCandidate(message.candidate);
          else pendingCandidates.push(message.candidate);
        } else if (message.type === "tts_started") {
          setTtsPlaying(true);
          setTtsNotice(undefined);
          void remoteAudioRef.current?.play().catch(() => undefined);
        } else if (message.type === "tts_stopped") {
          setTtsPlaying(false);
          if (message.reason === "user_speech") {
            setTtsNotice("检测到你开始说话，已自动停止语音播报并继续收集本次转写。");
          } else if (message.reason === "manual") {
            setTtsNotice("语音播报已由你手动停止。");
          }
        } else if (message.type === "transcript.final" && message.transcript) {
          setTranscripts((current) => [
            ...current.filter((item) => item.segment_id !== message.transcript?.segment_id),
            message.transcript as RealtimeTranscript,
          ].sort((left, right) => left.sequence - right.sequence));
        } else if (message.type === "transcript.failed") {
          setError(`第 ${message.sequence ?? "?"} 段语音转写失败，可复述或继续使用文字输入。`);
        } else if (message.type === "error") {
          setError(message.message ?? "实时媒体服务拒绝了当前信令请求。");
        }
      };
      socket.onclose = () => {
        if (generation === generationRef.current && !intentionalStopRef.current) {
          reconnectRef.current();
        }
      };
      socket.onerror = () => {
        if (generation === generationRef.current) reconnectRef.current();
      };

      connectionTimeoutRef.current = window.setTimeout(() => {
        if (generation === generationRef.current && peer.connectionState !== "connected") {
          reconnectRef.current();
        }
      }, 12_000);

      heartbeatRef.current = window.setInterval(() => {
        const current = sessionRef.current;
        if (!current) return;
        void heartbeatRealtimeSession(current.session_id).catch(() => reconnectRef.current());
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "heartbeat", session_id: current.session_id }));
        }
      }, grant.heartbeat_interval_seconds * 1_000);
    },
    [clearConnection],
  );

  reconnectRef.current = () => {
    if (intentionalStopRef.current || reconnectingRef.current) return;
    const current = sessionRef.current;
    const stream = localStreamRef.current;
    if (!current || !stream || reconnectAttemptsRef.current >= 3) {
      clearConnection(false);
      setStatus("FAILED");
      setError("实时语音已断开，请检查网络后重新开始；诊断业务状态未受影响。");
      return;
    }
    reconnectingRef.current = true;
    reconnectAttemptsRef.current += 1;
    setStatus("RECONNECTING");
    const delay = 500 * 2 ** (reconnectAttemptsRef.current - 1);
    reconnectTimerRef.current = window.setTimeout(() => {
      void reconnectRealtimeSession(current.session_id).then(({ session: renewed }) => {
        sessionRef.current = renewed;
        setSession(renewed);
        reconnectingRef.current = false;
        connect(renewed, stream);
      }).catch((caught: unknown) => {
        reconnectingRef.current = false;
        setError(apiMessage(caught));
        reconnectRef.current();
      });
    }, delay);
  };

  const stop = useCallback(async (notifyServer: boolean) => {
    rebindGenerationRef.current += 1;
    intentionalStopRef.current = true;
    const current = sessionRef.current;
    clearConnection(false);
    sessionRef.current = undefined;
    setSession(undefined);
    setTranscripts([]);
    setTtsPlaying(false);
    setTtsNotice(undefined);
    setStatus("IDLE");
    if (notifyServer && current) {
      await endRealtimeSession(current.session_id).catch(() => undefined);
    }
  }, [clearConnection]);

  useEffect(() => () => { void stop(true); }, [stop]);

  useEffect(() => {
    const current = sessionRef.current;
    const stream = localStreamRef.current;
    if (!current || current.diagnosis_run_id === (diagnosisRunId ?? null)) return;
    if (!stream) {
      void stop(true);
      return;
    }
    let cancelled = false;
    const rebindGeneration = rebindGenerationRef.current + 1;
    rebindGenerationRef.current = rebindGeneration;
    intentionalStopRef.current = true;
    reconnectingRef.current = false;
    reconnectAttemptsRef.current = 0;
    setStatus("REBINDING");
    setError(undefined);
    setTranscripts([]);
    setTtsPlaying(false);
    setTtsNotice(undefined);
    clearConnection(true);
    void (async () => {
      await endRealtimeSession(current.session_id).catch(() => undefined);
      if (cancelled || rebindGeneration !== rebindGenerationRef.current) return;
      try {
        const rebound = await createSession();
        if (cancelled || rebindGeneration !== rebindGenerationRef.current) {
          await endRealtimeSession(rebound.session_id).catch(() => undefined);
          return;
        }
        sessionRef.current = rebound;
        setSession(rebound);
        intentionalStopRef.current = false;
        connect(rebound, stream);
      } catch (caught) {
        if (cancelled || rebindGeneration !== rebindGenerationRef.current) return;
        clearConnection(false);
        sessionRef.current = undefined;
        setSession(undefined);
        setStatus("FAILED");
        setError(`实时语音未能绑定新诊断：${apiMessage(caught)}`);
      }
    })();
    return () => {
      cancelled = true;
      if (rebindGenerationRef.current === rebindGeneration) {
        rebindGenerationRef.current += 1;
      }
    };
  }, [clearConnection, connect, createSession, diagnosisRunId, stop]);

  useEffect(() => {
    const socket = socketRef.current;
    if (
      status !== "CONNECTED"
      || !speechSynthesisId
      || playedSynthesisRef.current === speechSynthesisId
      || socket?.readyState !== WebSocket.OPEN
    ) return;
    playedSynthesisRef.current = speechSynthesisId;
    socket.send(JSON.stringify({ type: "play_tts", synthesis_id: speechSynthesisId }));
  }, [speechSynthesisId, status]);

  async function start() {
    setError(undefined);
    setStatus("REQUESTING_MEDIA");
    intentionalStopRef.current = false;
    reconnectAttemptsRef.current = 0;
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.RTCPeerConnection) {
        throw new Error("当前浏览器不支持 WebRTC 音频。");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        video: false,
      });
      localStreamRef.current = stream;
      const created = await createSession();
      sessionRef.current = created;
      setSession(created);
      connect(created, stream);
    } catch (caught) {
      const current = sessionRef.current;
      clearConnection(false);
      sessionRef.current = undefined;
      setSession(undefined);
      if (current) void endRealtimeSession(current.session_id).catch(() => undefined);
      setStatus("FAILED");
      setError(apiMessage(caught));
    }
  }

  function interruptTts() {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "interrupt_tts", session_id: sessionRef.current?.session_id }));
    }
    remoteAudioRef.current?.pause();
    setTtsPlaying(false);
  }

  return (
    <Card size="small" title="实时语音协作" extra={<Tag>{statusLabel(status)}</Tag>}>
      <div className="page-stack">
        <Typography.Text type="secondary">
          WebRTC 仅传输实时音频；转写确认、诊断结论与设备动作仍通过正式业务接口保存和授权。
        </Typography.Text>
        {error ? <Alert type="warning" showIcon message={error} /> : null}
        {ttsNotice ? <Alert type="info" showIcon message={ttsNotice} /> : null}
        <Space wrap>
          {status === "IDLE" || status === "FAILED" ? (
            <Button type="primary" onClick={() => void start()}>授权麦克风并开始</Button>
          ) : (
            <Button danger onClick={() => void stop(true)}>结束并释放媒体</Button>
          )}
          <Button disabled={!ttsPlaying && status !== "CONNECTED"} onClick={interruptTts}>
            打断语音播报
          </Button>
        </Space>
        {session ? (
          <Typography.Text type="secondary">
            {`会话 ${session.session_id} · 重连 ${session.reconnect_count}/3 · 租约至 ${new Date(session.lease_expires_at).toLocaleTimeString()}`}
          </Typography.Text>
        ) : null}
        {transcripts.map((transcript) => (
          <TranscriptReviewRow
            key={transcript.segment_id}
            transcript={transcript}
            diagnosisRunId={diagnosisRunId}
            diagnosisVersion={diagnosisVersion}
            onDiagnosisCreated={onDiagnosisCreated}
            allowReanalysis={allowReanalysis}
            onUpdated={(updated) => setTranscripts((current) => current.map((item) => (
              item.segment_id === updated.segment_id ? updated : item
            )))}
          />
        ))}
        <audio ref={remoteAudioRef} autoPlay playsInline aria-label="实时语音远端音频" />
      </div>
    </Card>
  );
}

function TranscriptReviewRow({
  transcript,
  diagnosisRunId,
  diagnosisVersion,
  onDiagnosisCreated,
  onUpdated,
  allowReanalysis,
}: {
  transcript: RealtimeTranscript;
  diagnosisRunId?: string;
  diagnosisVersion?: number;
  onDiagnosisCreated?: (diagnosis: DiagnosisRun) => void;
  onUpdated: (value: RealtimeTranscript) => void;
  allowReanalysis: boolean;
}) {
  const unsafeTranscript = (transcript.security_findings ?? []).length > 0;
  const [correction, setCorrection] = useState(unsafeTranscript ? "" : transcript.text);
  const [busy, setBusy] = useState(false);
  const [reviewError, setReviewError] = useState<string>();
  const [reanalyzing, setReanalyzing] = useState(false);
  const needsEntityConfirmation = transcript.entities.some((item) => item.requires_confirmation);
  const reviewed = transcript.status === "ACCEPTED" || transcript.status === "REJECTED";

  async function decide(decision: "ACCEPTED" | "REJECTED") {
    setBusy(true);
    setReviewError(undefined);
    try {
      const updated = await confirmRealtimeTranscript(
        transcript.segment_id,
        decision,
        decision === "ACCEPTED" ? correction : null,
        transcript.version,
      );
      onUpdated(updated.transcript);
    } catch (caught) {
      setReviewError(apiMessage(caught));
    } finally {
      setBusy(false);
    }
  }

  async function reanalyze() {
    if (!diagnosisRunId || !diagnosisVersion) return;
    setReanalyzing(true);
    setReviewError(undefined);
    try {
      const result = await reanalyzeDiagnosisWithTranscript(
        diagnosisRunId,
        transcript.segment_id,
        diagnosisVersion,
        crypto.randomUUID(),
      );
      onDiagnosisCreated?.(result.diagnosis);
    } catch (caught) {
      setReviewError(apiMessage(caught));
    } finally {
      setReanalyzing(false);
    }
  }

  return (
    <Card size="small" type="inner" title={`转写 ${transcript.sequence}`} extra={<Tag>{transcript.status}</Tag>}>
      <div className="page-stack">
        <Typography.Text type="secondary">
          {`${transcript.start_ms}–${transcript.end_ms} ms · 置信度 ${(transcript.confidence * 100).toFixed(1)}% · ${transcript.resolved_release_id}`}
        </Typography.Text>
        {transcript.agent_event_sequence ? (
          <Alert
            type="success"
            showIcon
            message={`已作为人工确认输入写入 Agent 事件 #${transcript.agent_event_sequence}`}
          />
        ) : null}
        {needsEntityConfirmation ? (
          <Alert
            type="warning"
            showIcon
            message="设备号、部件号或报警码置信度不足，请复述并核对后再确认。"
          />
        ) : null}
        {unsafeTranscript ? (
          <Alert
            type="warning"
            showIcon
            message="实时转写命中内容安全策略"
            description={(
              <Space wrap>
                <Tag color="red">{`策略 ${transcript.security_policy_version ?? "unknown"}`}</Tag>
                {(transcript.security_findings ?? []).map((finding) => (
                  <Tag
                    color="orange"
                    key={`${finding.source_id ?? transcript.segment_id}:${finding.pattern_id}`}
                  >
                    {`${finding.pattern_id} · ${finding.category}`}
                  </Tag>
                ))}
                <Typography.Text>
                  原转写不能直接进入 Agent；请拒绝并复述，或填写核实后的安全文本。
                </Typography.Text>
              </Space>
            )}
          />
        ) : null}
        {reviewError ? <Alert type="error" showIcon message={reviewError} /> : null}
        <Input.TextArea
          aria-label={unsafeTranscript ? `实时转写安全修正文 ${transcript.segment_id}` : undefined}
          value={correction}
          disabled={reviewed}
          maxLength={4_000}
          autoSize={{ minRows: 2, maxRows: 5 }}
          onChange={(event) => setCorrection(event.target.value)}
        />
        {!reviewed ? (
          <Space>
            <Button
              type="primary"
              loading={busy}
              disabled={unsafeTranscript && (
                !correction.trim() || correction.trim() === transcript.text.trim()
              )}
              onClick={() => void decide("ACCEPTED")}
            >
              {unsafeTranscript ? "确认安全修正" : "确认转写"}
            </Button>
            <Button danger loading={busy} onClick={() => void decide("REJECTED")}>
              拒绝并复述
            </Button>
          </Space>
        ) : null}
        {allowReanalysis && transcript.status === "ACCEPTED" && transcript.agent_event_id ? (
          <Button
            type="primary"
            loading={reanalyzing}
            disabled={!diagnosisRunId || !diagnosisVersion}
            onClick={() => void reanalyze()}
          >
            使用确认信息重新诊断
          </Button>
        ) : null}
      </div>
    </Card>
  );
}

function statusLabel(status: ConnectionStatus): string {
  return {
    IDLE: "未连接",
    REQUESTING_MEDIA: "等待麦克风授权",
    SIGNALING: "信令协商中",
    CONNECTED: "已连接",
    REBINDING: "正在绑定新诊断",
    RECONNECTING: "网络切换重连中",
    FAILED: "已降级为文字模式",
  }[status];
}

function apiMessage(error: unknown): string {
  if (error instanceof ApiClientError) return `${error.message}（${error.code}）`;
  if (error instanceof DOMException && error.name === "NotAllowedError") {
    return "麦克风权限被拒绝，仍可继续使用文字和上传录音。";
  }
  return error instanceof Error ? error.message : "实时语音建立失败，请继续使用文字模式。";
}
