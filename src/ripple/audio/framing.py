"""Deterministic full-sequence and streaming PCM framing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ripple.audio.io import AudioBuffer

FloatFrames = npt.NDArray[np.float32]


@dataclass(slots=True)
class StreamingFramer:
    frame_samples: int = 480
    hop_samples: int = 480
    channels: int = 1
    _buffer: FloatFrames = field(init=False, repr=False)
    _frames_emitted: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.frame_samples <= 0 or self.hop_samples <= 0:
            raise ValueError("frame and hop sizes must be positive")
        if self.hop_samples > self.frame_samples:
            raise ValueError("hop_samples cannot exceed frame_samples")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        self._buffer = np.empty((self.channels, 0), dtype=np.float32)

    @property
    def buffered_samples(self) -> int:
        return int(self._buffer.shape[1])

    @property
    def frames_emitted(self) -> int:
        return self._frames_emitted

    def push(self, samples: FloatFrames) -> FloatFrames:
        chunk = np.asarray(samples, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        if chunk.ndim != 2 or chunk.shape[0] != self.channels:
            raise ValueError("chunk must have configured channel-first shape")
        if not np.isfinite(chunk).all():
            raise ValueError("chunk must contain finite samples")
        if chunk.shape[1]:
            self._buffer = np.concatenate((self._buffer, chunk), axis=1)

        count = 0
        if self.buffered_samples >= self.frame_samples:
            count = 1 + (self.buffered_samples - self.frame_samples) // self.hop_samples
        frames = np.empty((count, self.channels, self.frame_samples), dtype=np.float32)
        for index in range(count):
            start = index * self.hop_samples
            frames[index] = self._buffer[:, start : start + self.frame_samples]
        if count:
            self._buffer = self._buffer[:, count * self.hop_samples :].copy()
            self._frames_emitted += count
        frames.setflags(write=False)
        return frames

    def flush(self, *, pad: bool = False) -> FloatFrames:
        if not pad or self.buffered_samples == 0:
            result = np.empty((0, self.channels, self.frame_samples), dtype=np.float32)
        else:
            result = np.zeros((1, self.channels, self.frame_samples), dtype=np.float32)
            length = min(self.buffered_samples, self.frame_samples)
            result[0, :, :length] = self._buffer[:, :length]
            self._frames_emitted += 1
        self._buffer = np.empty((self.channels, 0), dtype=np.float32)
        result.setflags(write=False)
        return result

    def reset(self) -> None:
        self._buffer = np.empty((self.channels, 0), dtype=np.float32)
        self._frames_emitted = 0


def frame_audio(
    audio: AudioBuffer,
    *,
    frame_samples: int = 480,
    hop_samples: int | None = None,
    pad: bool = False,
) -> FloatFrames:
    framer = StreamingFramer(
        frame_samples=frame_samples,
        hop_samples=hop_samples or frame_samples,
        channels=audio.channels,
    )
    complete = framer.push(audio.samples)
    remainder = framer.flush(pad=pad)
    result = np.concatenate((complete, remainder), axis=0)
    result.setflags(write=False)
    return result

