"""Deterministic audio I/O, DSP, and framing."""

from ripple.audio.framing import StreamingFramer, frame_audio
from ripple.audio.io import (
    AudioBackend,
    AudioBuffer,
    available_audio_backends,
    load_audio,
    save_audio,
)
from ripple.audio.pitch import PitchEstimate, estimate_yin
from ripple.audio.resample import resample_audio, resampled_length

__all__ = [
    "AudioBackend",
    "AudioBuffer",
    "PitchEstimate",
    "StreamingFramer",
    "available_audio_backends",
    "estimate_yin",
    "frame_audio",
    "load_audio",
    "resample_audio",
    "resampled_length",
    "save_audio",
]

