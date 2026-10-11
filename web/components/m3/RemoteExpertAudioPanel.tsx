"use client";

import { Alert, Button, Card, Space, Tag, Typography } from "antd";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  type RealtimeSessionGrant,
  createExpertCollaborationRealtimeSession,
  endRealtimeSession,
  heartbeatRealtimeSession,
  reconnectRealtimeSession,
} from "@/lib/api/client";

type MediaStatus = "IDLE" | "REQUESTING" | "SIGNALING" | "CONNECTED" | "RECONNECTING" | "FAILED";

interface SignalingMessage {
  type?: string;
  sdp?: string;
  candidate?: RTCIceCandidateInit;
  message?: string;
}

export function RemoteExpertAudioPanel({ collaborationId }: { collaborationId: string }) {
  const [status, setStatus] = useState<MediaStatus>("IDLE");
  const [error, setError] = useState<string>();
  const [session, setSession] = useState<RealtimeSessionGrant>();
  const remoteAudioRef = useRef<HTMLAudioElement>(null);
  const sessionRef = useRef<RealtimeSessionGrant | undefined>(undefined);
  const streamRef = useRef<MediaStream | undefined>(undefined);
  const peerRef = useRef<RTCPeerConnection | undefined>(undefined);
  const socketRef = useRef<WebSocket | undefined>(undefined);
  const heartbeatRef = useRef<number | undefined>(undefined);
  const reconnectTimerRef = useRef<number | undefined>(undefined);
  const reconnectAttemptsRef = useRef(0);
  const intentionalStopRef = useRef(false);
  const generationRef = useRef(0);
  const reconnectRef = useRef<() => void>(() => undefined);

  const clearTransport = useCallback((keepMedia: boolean) => {
    generationRef.current += 1;
    if (heartbeatRef.current !== undefined) window.clearInterval(heartbeatRef.current);
    if (reconnectTimerRef.current !== undefined) window.clearTimeout(reconnectTimerRef.current);
    heartbeatRef.current = undefined;
    reconnectTimerRef.current = undefined;
    socketRef.current?.close();
    peerRef.current?.close();
    socketRef.current = undefined;
    peerRef.current = undefined;
    if (!keepMedia) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = undefined;
      if (remoteAudioRef.current) remoteAudioRef.current.srcObject = null;
    }
  }, []);

  const connect = useCallback((grant: RealtimeSessionGrant, stream: MediaStream) => {
    clearTransport(true);
    const generation = generationRef.current;
    setStatus(reconnectAttemptsRef.current ? "RECONNECTING" : "SIGNALING");
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
        reconnectAttemptsRef.current = 0;
        setStatus("CONNECTED");
      } else if (["failed", "disconnected"].includes(peer.connectionState)) {
        reconnectRef.current();
      }
    };
    socket.onopen = () => socket.send(JSON.stringify({
      type: "authenticate",
      session_id: grant.session_id,
      access_token: grant.access_token,
    }));
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
          for (const candidate of pendingCandidates.splice(0)) await peer.addIceCandidate(candidate);
        }).catch(() => reconnectRef.current());
      } else if (message.type === "candidate" && message.candidate) {
        if (peer.remoteDescription) void peer.addIceCandidate(message.candidate);
        else pendingCandidates.push(message.candidate);
      } else if (message.type === "error") {
        setError(message.message ?? "专家音频服务拒绝了当前请求。");
      }
    };
    socket.onclose = () => {
      if (!intentionalStopRef.current && generation === generationRef.current) reconnectRef.current();
    };
    socket.onerror = () => reconnectRef.current();
    heartbeatRef.current = window.setInterval(() => {
      const current = sessionRef.current;
      if (!current) return;
      void heartbeatRealtimeSession(current.session_id).catch(() => reconnectRef.current());
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "heartbeat" }));
    }, grant.heartbeat_interval_seconds * 1_000);
  }, [clearTransport]);

  reconnectRef.current = () => {
    const current = sessionRef.current;
    const stream = streamRef.current;
    if (intentionalStopRef.current || !current || !stream) return;
    if (reconnectAttemptsRef.current >= 3) {
      clearTransport(false);
      setStatus("FAILED");
      setError("专家音频连接失败，请改用企业电话或文字协作；工单和协作状态未改变。");
      return;
    }
    reconnectAttemptsRef.current += 1;
    setStatus("RECONNECTING");
    reconnectTimerRef.current = window.setTimeout(() => {
      void reconnectRealtimeSession(current.session_id).then(({ session: renewed }) => {
        sessionRef.current = renewed;
        setSession(renewed);
        connect(renewed, stream);
      }).catch(() => reconnectRef.current());
    }, 500 * 2 ** (reconnectAttemptsRef.current - 1));
  };

  const stop = useCallback(async (notifyServer: boolean) => {
    intentionalStopRef.current = true;
    const current = sessionRef.current;
    clearTransport(false);
    sessionRef.current = undefined;
    setSession(undefined);
    setStatus("IDLE");
    if (notifyServer && current) await endRealtimeSession(current.session_id).catch(() => undefined);
  }, [clearTransport]);

  useEffect(() => () => { void stop(true); }, [stop]);

  async function start() {
    intentionalStopRef.current = false;
    reconnectAttemptsRef.current = 0;
    setError(undefined);
    setStatus("REQUESTING");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;
      const result = await createExpertCollaborationRealtimeSession(collaborationId);
      sessionRef.current = result.session;
      setSession(result.session);
      connect(result.session, stream);
    } catch (caught) {
      clearTransport(false);
      setStatus("FAILED");
      setError(caught instanceof Error ? caught.message : "无法建立专家音频连接。");
    }
  }

  return (
    <Card size="small" title="双人专家音频">
      <Space direction="vertical" style={{ width: "100%" }}>
        <Space wrap>
          <Tag color={status === "CONNECTED" ? "green" : status === "FAILED" ? "red" : "blue"}>
            {mediaStatusLabel(status)}
          </Tag>
          {session ? <Typography.Text code>{session.participant_subject_id}</Typography.Text> : null}
        </Space>
        {error ? <Alert type="warning" showIcon message="音频已降级" description={error} /> : null}
        <audio ref={remoteAudioRef} autoPlay aria-label="远程专家音频" />
        {status === "IDLE" || status === "FAILED" ? (
          <Button type="primary" onClick={() => void start()}>加入专家音频</Button>
        ) : (
          <Button danger onClick={() => void stop(true)}>离开音频</Button>
        )}
        <Typography.Text type="secondary">
          音频仅在媒体进程内实时转发，不录音、不转写，也不会自动修改诊断或工单。
        </Typography.Text>
      </Space>
    </Card>
  );
}

function mediaStatusLabel(status: MediaStatus): string {
  return {
    IDLE: "尚未加入",
    REQUESTING: "申请麦克风",
    SIGNALING: "建立连接",
    CONNECTED: "通话中",
    RECONNECTING: "安全重连",
    FAILED: "连接失败",
  }[status];
}
