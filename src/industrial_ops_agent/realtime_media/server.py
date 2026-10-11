"""Standalone aiortc media terminator and governed ASR bridge."""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import wave
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from time import monotonic
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from industrial_ops_agent.realtime_media.vad import AudioSegment, PcmVadSegmenter, VadConfig


class MediaServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IOAP_MEDIA_", extra="ignore", frozen=True)

    host: str = "0.0.0.0"  # noqa: S104 - container listener
    port: int = Field(default=7880, ge=1, le=65535)
    core_api_url: str = "http://api:8000/api/v1"
    api_timeout_seconds: float = Field(default=25.0, gt=1, le=60)
    vad_energy_threshold: int = Field(default=450, ge=1, le=32_767)
    allowed_origins: tuple[str, ...] = ("http://localhost:3000",)


class AuthenticationMessage(BaseModel):
    type: str
    session_id: str = Field(min_length=1, max_length=128)
    access_token: str = Field(min_length=32, max_length=512)


@dataclass(frozen=True, slots=True)
class MediaAdmission:
    session_id: str
    media_access_token: str
    expert_collaboration_id: str | None = None
    participant_subject_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExpertAudioSeat:
    room_id: str
    participant_subject_id: str
    generation: int


@dataclass(slots=True)
class _ExpertAudioParticipant:
    seat: ExpertAudioSeat
    downlink: Any
    disconnect: Callable[[], Awaitable[None]]


class ExpertAudioRoomFull(RuntimeError):
    pass


class ExpertAudioRoomRegistry:
    """Two-seat, generation-safe in-memory room registry for live PCM only."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._rooms: dict[str, dict[str, _ExpertAudioParticipant]] = {}
        self._generation = 0

    async def join(
        self,
        room_id: str,
        participant_subject_id: str,
        downlink: Any,
        disconnect: Callable[[], Awaitable[None]],
    ) -> ExpertAudioSeat:
        replaced: _ExpertAudioParticipant | None = None
        async with self._lock:
            room = self._rooms.setdefault(room_id, {})
            if participant_subject_id not in room and len(room) >= 2:
                raise ExpertAudioRoomFull("expert audio room already has two participants")
            replaced = room.get(participant_subject_id)
            self._generation += 1
            seat = ExpertAudioSeat(room_id, participant_subject_id, self._generation)
            room[participant_subject_id] = _ExpertAudioParticipant(seat, downlink, disconnect)
        if replaced is not None:
            await replaced.disconnect()
        return seat

    async def leave(self, seat: ExpertAudioSeat) -> None:
        async with self._lock:
            room = self._rooms.get(seat.room_id)
            if room is None:
                return
            current = room.get(seat.participant_subject_id)
            if current is not None and current.seat.generation == seat.generation:
                del room[seat.participant_subject_id]
            if not room:
                self._rooms.pop(seat.room_id, None)

    async def forward(self, seat: ExpertAudioSeat, pcm: bytes) -> bool:
        if not pcm:
            return False
        async with self._lock:
            room = self._rooms.get(seat.room_id)
            if room is None:
                return False
            current = room.get(seat.participant_subject_id)
            if current is None or current.seat.generation != seat.generation:
                return False
            peer = next(
                (
                    participant
                    for subject_id, participant in room.items()
                    if subject_id != seat.participant_subject_id
                ),
                None,
            )
            downlink = peer.downlink if peer is not None else None
        if downlink is None:
            return False
        await downlink.enqueue_pcm(pcm)
        return True

    async def room_count(self) -> int:
        async with self._lock:
            return len(self._rooms)


@dataclass(slots=True)
class PlaybackState:
    task: asyncio.Task[Any] | None = None
    synthesis_id: str | None = None


class CoreApiClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    async def consume(self, session_id: str, access_token: str) -> MediaAdmission:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/realtime-sessions/consume",
                json={"session_id": session_id, "access_token": access_token},
            )
        response.raise_for_status()
        payload = response.json()["data"]
        return MediaAdmission(
            session_id=str(payload["session"]["session_id"]),
            media_access_token=str(payload["media_access_token"]),
            expert_collaboration_id=(
                str(payload["expert_collaboration_id"])
                if payload.get("expert_collaboration_id") is not None
                else None
            ),
            participant_subject_id=(
                str(payload["participant_subject_id"])
                if payload.get("participant_subject_id") is not None
                else None
            ),
        )

    async def transcribe(
        self,
        admission: MediaAdmission,
        segment: AudioSegment,
        *,
        sequence: int,
    ) -> dict[str, Any]:
        segment_id = f"segment-{admission.session_id.removeprefix('realtime-')}-{sequence}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/realtime-sessions/{admission.session_id}/transcriptions",
                headers={
                    "Authorization": f"Bearer {admission.media_access_token}",
                    "Content-Type": "audio/wav",
                    "Segment-Id": segment_id,
                    "Segment-Sequence": str(sequence),
                    "Segment-Start-Ms": str(segment.start_ms),
                    "Segment-End-Ms": str(segment.end_ms),
                },
                content=_wav(segment.pcm_s16le),
            )
        response.raise_for_status()
        value = response.json()["data"]
        if not isinstance(value, dict):
            raise ValueError("transcription response is invalid")
        return value

    async def speech(self, admission: MediaAdmission, synthesis_id: str) -> bytes:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                f"{self._base_url}/realtime-sessions/{admission.session_id}"
                f"/speech-syntheses/{synthesis_id}/audio",
                headers={"Authorization": f"Bearer {admission.media_access_token}"},
            )
        response.raise_for_status()
        if response.headers.get("content-type", "").split(";", 1)[0] != "audio/wav":
            raise ValueError("speech response is not WAV audio")
        return response.content


def create_media_app(settings: MediaServerSettings | None = None) -> FastAPI:
    resolved = settings or MediaServerSettings()
    app = FastAPI(title="Industrial Operations Realtime Media", docs_url=None, redoc_url=None)
    core = CoreApiClient(resolved.core_api_url, resolved.api_timeout_seconds)
    expert_rooms = ExpertAudioRoomRegistry()
    app.state.expert_audio_rooms = expert_rooms

    @app.get("/health/live", include_in_schema=False)
    async def live() -> dict[str, str]:
        return {"service": "industrial-ops-realtime-media", "status": "ok"}

    @app.get("/health/ready", include_in_schema=False)
    async def ready() -> dict[str, str]:
        _rtc_modules()
        return {"service": "industrial-ops-realtime-media", "status": "ready"}

    @app.websocket("/realtime")
    async def realtime(websocket: WebSocket) -> None:
        if websocket.headers.get("origin") not in resolved.allowed_origins:
            await websocket.close(code=1008, reason="origin not allowed")
            return
        await websocket.accept()
        peer: Any | None = None
        downlink: Any | None = None
        playback = PlaybackState()
        expert_seat: ExpertAudioSeat | None = None
        tasks: set[asyncio.Task[Any]] = set()
        send_lock = asyncio.Lock()
        try:
            raw_auth = await asyncio.wait_for(websocket.receive_text(), timeout=10)
            auth = AuthenticationMessage.model_validate_json(raw_auth)
            if auth.type != "authenticate":
                raise ValueError("first signaling message must authenticate")
            admission = await core.consume(auth.session_id, auth.access_token)
            rtc = _rtc_modules()
            peer = rtc["RTCPeerConnection"]()
            downlink = _new_downlink_track(rtc)
            peer.addTrack(downlink)
            if admission.expert_collaboration_id is not None:
                if admission.participant_subject_id is None:
                    raise ValueError("expert admission participant is missing")

                async def disconnect_replaced_seat() -> None:
                    with suppress(Exception):
                        await websocket.close(code=1012, reason="participant reconnected")

                expert_seat = await expert_rooms.join(
                    admission.expert_collaboration_id,
                    admission.participant_subject_id,
                    downlink,
                    disconnect_replaced_seat,
                )
            await _send(websocket, send_lock, {"type": "authenticated"})

            @peer.on("track")  # type: ignore[untyped-decorator]
            def on_track(track: Any) -> None:
                if track.kind != "audio":
                    track.stop()
                    return
                if expert_seat is not None:
                    task = asyncio.create_task(
                        _relay_expert_audio(
                            track,
                            admission,
                            expert_seat,
                            expert_rooms,
                        )
                    )
                else:
                    task = asyncio.create_task(
                        _consume_audio(
                            track,
                            admission,
                            core,
                            websocket,
                            send_lock,
                            resolved,
                            on_speech_started=lambda: _interrupt_playback(
                                playback,
                                downlink,
                                websocket,
                                send_lock,
                                reason="user_speech",
                            ),
                        )
                    )
                tasks.add(task)
                task.add_done_callback(tasks.discard)

            while True:
                payload = json.loads(await websocket.receive_text())
                message_type = payload.get("type")
                if message_type == "offer":
                    if not isinstance(payload.get("sdp"), str) or len(payload["sdp"]) > 100_000:
                        raise ValueError("offer SDP is invalid")
                    description = rtc["RTCSessionDescription"](
                        sdp=str(payload["sdp"]), type="offer"
                    )
                    await peer.setRemoteDescription(description)
                    await peer.setLocalDescription(await peer.createAnswer())
                    await _send(
                        websocket,
                        send_lock,
                        {"type": "answer", "sdp": peer.localDescription.sdp},
                    )
                elif message_type == "candidate":
                    candidate_value = payload.get("candidate")
                    if isinstance(candidate_value, dict) and candidate_value.get("candidate"):
                        candidate_text = str(candidate_value["candidate"])
                        candidate = rtc["candidate_from_sdp"](
                            candidate_text.removeprefix("candidate:")
                        )
                        candidate.sdpMid = candidate_value.get("sdpMid")
                        candidate.sdpMLineIndex = candidate_value.get("sdpMLineIndex")
                        await peer.addIceCandidate(candidate)
                elif message_type == "heartbeat":
                    await _send(websocket, send_lock, {"type": "heartbeat_ack"})
                elif message_type == "play_tts":
                    if expert_seat is not None:
                        await _send(
                            websocket,
                            send_lock,
                            {
                                "type": "error",
                                "message": "speech playback is unavailable in expert rooms",
                            },
                        )
                        continue
                    synthesis_id = payload.get("synthesis_id")
                    if not isinstance(synthesis_id, str) or not synthesis_id.startswith("tts-"):
                        raise ValueError("synthesis id is invalid")
                    await _interrupt_playback(
                        playback,
                        downlink,
                        websocket,
                        send_lock,
                        reason="replaced",
                    )
                    tts_task = asyncio.create_task(
                        _play_tts(
                            core,
                            admission,
                            synthesis_id,
                            downlink,
                            websocket,
                            send_lock,
                            playback,
                        )
                    )
                    playback.task = tts_task
                    playback.synthesis_id = synthesis_id
                    tasks.add(tts_task)
                    tts_task.add_done_callback(tasks.discard)
                elif message_type == "interrupt_tts":
                    if expert_seat is not None:
                        await _send(
                            websocket,
                            send_lock,
                            {
                                "type": "error",
                                "message": "speech playback is unavailable in expert rooms",
                            },
                        )
                        continue
                    await _interrupt_playback(
                        playback,
                        downlink,
                        websocket,
                        send_lock,
                        reason="manual",
                    )
                else:
                    await _send(
                        websocket,
                        send_lock,
                        {"type": "error", "message": "unsupported signaling message"},
                    )
        except (TimeoutError, WebSocketDisconnect):
            pass
        except Exception:
            with suppress(Exception):
                await _send(
                    websocket,
                    send_lock,
                    {"type": "error", "message": "realtime media session failed"},
                )
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if expert_seat is not None:
                await expert_rooms.leave(expert_seat)
            if peer is not None:
                await peer.close()
            with suppress(Exception):
                await websocket.close()

    return app


async def route_admitted_audio(
    admission: MediaAdmission,
    seat: ExpertAudioSeat,
    pcm: bytes,
    *,
    registry: ExpertAudioRoomRegistry,
    automation_client: Any,
) -> bool:
    """Route expert PCM to its peer without touching the automation client."""

    del automation_client
    if (
        admission.expert_collaboration_id != seat.room_id
        or admission.participant_subject_id != seat.participant_subject_id
    ):
        return False
    return await registry.forward(seat, pcm)


async def _relay_expert_audio(
    track: Any,
    admission: MediaAdmission,
    seat: ExpertAudioSeat,
    registry: ExpertAudioRoomRegistry,
) -> None:
    av = importlib.import_module("av")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=48_000)
    try:
        while True:
            frame = await track.recv()
            for converted in resampler.resample(frame):
                pcm = bytes(converted.planes[0])[: converted.samples * 2]
                await route_admitted_audio(
                    admission,
                    seat,
                    pcm,
                    registry=registry,
                    automation_client=None,
                )
    except asyncio.CancelledError:
        raise
    except Exception:
        return


async def _consume_audio(
    track: Any,
    admission: MediaAdmission,
    core: CoreApiClient,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    settings: MediaServerSettings,
    on_speech_started: Callable[[], Awaitable[bool]],
) -> None:
    av = importlib.import_module("av")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
    vad = PcmVadSegmenter(VadConfig(energy_threshold=settings.vad_energy_threshold))
    sequence = 0
    try:
        while True:
            frame = await track.recv()
            for converted in resampler.resample(frame):
                pcm = bytes(converted.planes[0])[: converted.samples * 2]
                result = vad.push_with_events(pcm)
                if result.speech_started:
                    await on_speech_started()
                for segment in result.segments:
                    sequence += 1
                    await _transcribe_and_send(
                        core, admission, segment, sequence, websocket, send_lock
                    )
    except asyncio.CancelledError:
        raise
    except Exception:
        final = vad.flush()
        if final is not None:
            sequence += 1
            with suppress(Exception):
                await _transcribe_and_send(core, admission, final, sequence, websocket, send_lock)


async def _transcribe_and_send(
    core: CoreApiClient,
    admission: MediaAdmission,
    segment: AudioSegment,
    sequence: int,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
) -> None:
    try:
        transcript = await core.transcribe(admission, segment, sequence=sequence)
        await _send(
            websocket,
            send_lock,
            {"type": "transcript.final", "transcript": transcript},
        )
    except Exception:
        await _send(
            websocket,
            send_lock,
            {"type": "transcript.failed", "sequence": sequence},
        )


async def _play_tts(
    core: CoreApiClient,
    admission: MediaAdmission,
    synthesis_id: str,
    downlink: Any,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    playback: PlaybackState,
) -> None:
    current_task = asyncio.current_task()
    try:
        content = await core.speech(admission, synthesis_id)
        generation = await downlink.load_wav(content)
        await _send(
            websocket,
            send_lock,
            {"type": "tts_started", "synthesis_id": synthesis_id},
        )
        await downlink.wait_finished(generation)
        if playback.task is current_task:
            playback.task = None
            playback.synthesis_id = None
            await _send(
                websocket,
                send_lock,
                {
                    "type": "tts_stopped",
                    "synthesis_id": synthesis_id,
                    "reason": "completed",
                },
            )
    except asyncio.CancelledError:
        await downlink.interrupt()
        raise
    except Exception:
        if playback.task is current_task:
            playback.task = None
            playback.synthesis_id = None
        await downlink.interrupt()
        await _send(
            websocket,
            send_lock,
            {"type": "tts_stopped", "synthesis_id": synthesis_id, "reason": "failed"},
        )
        await _send(
            websocket,
            send_lock,
            {"type": "error", "message": "controlled speech playback failed"},
        )


async def _interrupt_playback(
    playback: PlaybackState,
    downlink: Any,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    *,
    reason: str,
) -> bool:
    task = playback.task
    synthesis_id = playback.synthesis_id
    was_running = task is not None and not task.done()
    playback.task = None
    playback.synthesis_id = None
    if was_running and task is not None:
        task.cancel()
    had_audio = bool(await downlink.interrupt())
    if not was_running and not had_audio:
        return False
    await _send(
        websocket,
        send_lock,
        {
            "type": "tts_stopped",
            "synthesis_id": synthesis_id,
            "reason": reason,
        },
    )
    return True


async def _send(websocket: WebSocket, lock: asyncio.Lock, payload: dict[str, Any]) -> None:
    async with lock:
        await websocket.send_json(payload)


def _rtc_modules() -> dict[str, Any]:
    try:
        aiortc = importlib.import_module("aiortc")
        sdp = importlib.import_module("aiortc.sdp")
    except ImportError as exc:
        raise RuntimeError("install the realtime optional dependencies") from exc
    return {
        "RTCPeerConnection": aiortc.RTCPeerConnection,
        "MediaStreamTrack": aiortc.MediaStreamTrack,
        "RTCSessionDescription": aiortc.RTCSessionDescription,
        "candidate_from_sdp": sdp.candidate_from_sdp,
    }


def _new_downlink_track(rtc: dict[str, Any]) -> Any:
    """Create a 48 kHz mono WebRTC track with replaceable, interruptible audio."""

    av = importlib.import_module("av")
    media_stream_track = rtc["MediaStreamTrack"]

    class QueuedAudioTrack(media_stream_track):  # type: ignore[misc, valid-type]
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self._buffer_lock = asyncio.Lock()
            self._pcm = bytearray()
            self._generation = 0
            self._timestamp = 0
            self._next_frame_at: float | None = None

        async def load_wav(self, content: bytes) -> int:
            pcm = _decode_wav_pcm(content, av)
            async with self._buffer_lock:
                self._generation += 1
                self._pcm = bytearray(pcm)
                return self._generation

        async def enqueue_pcm(self, pcm: bytes) -> None:
            """Append live peer audio while bounding the queue to two seconds."""

            max_bytes = 48_000 * 2 * 2
            async with self._buffer_lock:
                self._pcm.extend(pcm)
                overflow = len(self._pcm) - max_bytes
                if overflow > 0:
                    del self._pcm[:overflow]

        async def interrupt(self) -> bool:
            async with self._buffer_lock:
                had_audio = bool(self._pcm)
                self._generation += 1
                self._pcm.clear()
                return had_audio

        async def wait_finished(self, generation: int) -> None:
            while True:
                async with self._buffer_lock:
                    if generation != self._generation or not self._pcm:
                        return
                await asyncio.sleep(0.02)

        async def recv(self) -> Any:
            frame_samples = 960
            frame_bytes = frame_samples * 2
            now = monotonic()
            if self._next_frame_at is None:
                self._next_frame_at = now
            else:
                await asyncio.sleep(max(0.0, self._next_frame_at - now))
            self._next_frame_at += frame_samples / 48_000
            async with self._buffer_lock:
                chunk = bytes(self._pcm[:frame_bytes])
                del self._pcm[: len(chunk)]
            chunk = chunk.ljust(frame_bytes, b"\0")
            frame = av.AudioFrame(format="s16", layout="mono", samples=frame_samples)
            frame.planes[0].update(chunk)
            frame.sample_rate = 48_000
            frame.pts = self._timestamp
            frame.time_base = Fraction(1, 48_000)
            self._timestamp += frame_samples
            return frame

    return QueuedAudioTrack()


def _decode_wav_pcm(content: bytes, av: Any) -> bytes:
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("speech audio is too large")
    container = av.open(io.BytesIO(content), format="wav")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=48_000)
    chunks: list[bytes] = []
    try:
        for decoded in container.decode(audio=0):
            for converted in resampler.resample(decoded):
                chunks.append(bytes(converted.planes[0])[: converted.samples * 2])
        for converted in resampler.resample(None):
            chunks.append(bytes(converted.planes[0])[: converted.samples * 2])
    finally:
        container.close()
    pcm = b"".join(chunks)
    if not pcm:
        raise ValueError("speech audio is empty")
    return pcm


def _wav(pcm_s16le: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(16_000)
        destination.writeframes(pcm_s16le)
    return output.getvalue()


def run() -> None:
    settings = MediaServerSettings()
    uvicorn.run(create_media_app(settings), host=settings.host, port=settings.port)
