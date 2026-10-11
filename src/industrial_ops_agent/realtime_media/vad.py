"""Deterministic bounded energy VAD for 16 kHz mono PCM streams."""

from __future__ import annotations

import sys
from array import array
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioSegment:
    start_ms: int
    end_ms: int
    pcm_s16le: bytes


@dataclass(frozen=True, slots=True)
class VadPushResult:
    segments: tuple[AudioSegment, ...]
    speech_started: bool


@dataclass(frozen=True, slots=True)
class VadConfig:
    sample_rate: int = 16_000
    frame_ms: int = 20
    energy_threshold: int = 450
    speech_start_ms: int = 60
    speech_end_silence_ms: int = 500
    pre_roll_ms: int = 100
    max_segment_ms: int = 15_000

    def __post_init__(self) -> None:
        if self.sample_rate != 16_000 or self.frame_ms not in {10, 20, 30}:
            raise ValueError("VAD requires 16 kHz PCM with 10, 20 or 30 ms frames")
        if not 1 <= self.energy_threshold <= 32_767:
            raise ValueError("VAD energy threshold is invalid")
        for value in (
            self.speech_start_ms,
            self.speech_end_silence_ms,
            self.pre_roll_ms,
            self.max_segment_ms,
        ):
            if value < self.frame_ms or value % self.frame_ms:
                raise ValueError("VAD durations must be positive frame multiples")


class PcmVadSegmenter:
    def __init__(self, config: VadConfig | None = None) -> None:
        config = config or VadConfig()
        self._config = config
        self._frame_bytes = config.sample_rate * config.frame_ms // 1_000 * 2
        self._buffer = bytearray()
        self._pre_roll: deque[tuple[int, bytes]] = deque(
            maxlen=config.pre_roll_ms // config.frame_ms
        )
        self._active = bytearray()
        self._active_start_frame = 0
        self._frame_index = 0
        self._speech_frames = 0
        self._silence_frames = 0

    def push(self, pcm_s16le: bytes) -> tuple[AudioSegment, ...]:
        return self.push_with_events(pcm_s16le).segments

    def push_with_events(self, pcm_s16le: bytes) -> VadPushResult:
        self._buffer.extend(pcm_s16le)
        output: list[AudioSegment] = []
        speech_started = False
        while len(self._buffer) >= self._frame_bytes:
            frame = bytes(self._buffer[: self._frame_bytes])
            del self._buffer[: self._frame_bytes]
            was_active = bool(self._active)
            segment = self._push_frame(frame)
            if not was_active and self._active:
                speech_started = True
            if segment is not None:
                output.append(segment)
        return VadPushResult(tuple(output), speech_started)

    def flush(self) -> AudioSegment | None:
        if not self._active:
            self._buffer.clear()
            self._pre_roll.clear()
            return None
        segment = self._finish()
        self._buffer.clear()
        return segment

    def _push_frame(self, frame: bytes) -> AudioSegment | None:
        current_index = self._frame_index
        self._frame_index += 1
        speech = _rms(frame) >= self._config.energy_threshold
        if not self._active:
            self._pre_roll.append((current_index, frame))
            self._speech_frames = self._speech_frames + 1 if speech else 0
            if self._speech_frames >= self._config.speech_start_ms // self._config.frame_ms:
                self._active_start_frame = self._pre_roll[0][0]
                self._active.extend(b"".join(item[1] for item in self._pre_roll))
                self._pre_roll.clear()
                self._silence_frames = 0
            return None

        self._active.extend(frame)
        self._silence_frames = 0 if speech else self._silence_frames + 1
        active_frames = len(self._active) // self._frame_bytes
        if (
            self._silence_frames
            >= self._config.speech_end_silence_ms // self._config.frame_ms
            or active_frames >= self._config.max_segment_ms // self._config.frame_ms
        ):
            return self._finish()
        return None

    def _finish(self) -> AudioSegment:
        frame_count = len(self._active) // self._frame_bytes
        segment = AudioSegment(
            start_ms=self._active_start_frame * self._config.frame_ms,
            end_ms=(self._active_start_frame + frame_count) * self._config.frame_ms,
            pcm_s16le=bytes(self._active),
        )
        self._active.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self._pre_roll.clear()
        return segment


def _rms(frame: bytes) -> int:
    samples = array("h")
    samples.frombytes(frame)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0
    return int((sum(int(value) * int(value) for value in samples) / len(samples)) ** 0.5)
