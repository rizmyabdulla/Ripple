"""Normalized audio I/O with TorchCodec-first, SoundFile-fallback decoding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatAudio = npt.NDArray[np.float32]


class AudioBackend(StrEnum):
    AUTO = "auto"
    TORCHCODEC = "torchcodec"
    SOUNDFILE = "soundfile"


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """Read-only channel-first normalized float PCM."""

    samples: FloatAudio
    sample_rate: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]
        if samples.ndim != 2 or samples.shape[0] < 1:
            raise ValueError("audio must be channel-first with shape [channels, samples]")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not np.isfinite(samples).all():
            raise ValueError("audio samples must be finite")
        samples = np.ascontiguousarray(samples)
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[1])


def available_audio_backends() -> tuple[AudioBackend, ...]:
    backends: list[AudioBackend] = []
    if find_spec("torchcodec") is not None and find_spec("torch") is not None:
        backends.append(AudioBackend.TORCHCODEC)
    if find_spec("soundfile") is not None:
        backends.append(AudioBackend.SOUNDFILE)
    return tuple(backends)


def _load_torchcodec(path: Path) -> AudioBuffer:
    from torchcodec.decoders import AudioDecoder

    decoded = AudioDecoder(str(path)).get_all_samples()
    samples = decoded.data.detach().cpu().float().numpy()
    return AudioBuffer(samples=samples, sample_rate=int(decoded.sample_rate))


def _load_soundfile(path: Path) -> AudioBuffer:
    import soundfile as sf

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return AudioBuffer(samples=np.asarray(samples.T), sample_rate=int(sample_rate))


def load_audio(path: str | Path, backend: AudioBackend | str = AudioBackend.AUTO) -> AudioBuffer:
    """Decode without implicit channel conversion or resampling."""
    source = Path(path)
    selected = AudioBackend(backend)
    if selected is AudioBackend.AUTO:
        available = available_audio_backends()
        if not available:
            raise RuntimeError("no audio backend available; install torchcodec or soundfile")
        selected = available[0]
    if selected is AudioBackend.TORCHCODEC:
        if AudioBackend.TORCHCODEC not in available_audio_backends():
            raise RuntimeError("TorchCodec backend requested but torch/torchcodec is unavailable")
        return _load_torchcodec(source)
    if AudioBackend.SOUNDFILE not in available_audio_backends():
        raise RuntimeError("SoundFile backend requested but soundfile is unavailable")
    return _load_soundfile(source)


def save_audio(path: str | Path, audio: AudioBuffer, *, subtype: str = "PCM_16") -> None:
    """Write lossless audio through SoundFile; no implicit normalization."""
    import soundfile as sf

    sf.write(Path(path), audio.samples.T, audio.sample_rate, subtype=subtype)

