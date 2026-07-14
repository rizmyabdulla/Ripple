"""Deterministic, length-stable reference resampling."""

from __future__ import annotations

import math

import numpy as np

from ripple.audio.io import AudioBuffer


def resampled_length(input_length: int, source_rate: int, target_rate: int) -> int:
    """Return nearest duration-preserving length using integer arithmetic."""
    if input_length < 0:
        raise ValueError("input_length cannot be negative")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    return (input_length * target_rate + source_rate // 2) // source_rate


def resample_audio(audio: AudioBuffer, target_rate: int, *, half_width: int = 16) -> AudioBuffer:
    """Resample with a deterministic windowed-sinc reference implementation.

    This implementation prioritizes reproducibility and exact output length. Production
    backends may replace it only after conformance tests establish equivalent framing.
    """
    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    if half_width < 4:
        raise ValueError("half_width must be at least four samples")
    if target_rate == audio.sample_rate:
        return AudioBuffer(audio.samples.copy(), audio.sample_rate)

    output_length = resampled_length(audio.frames, audio.sample_rate, target_rate)
    output = np.empty((audio.channels, output_length), dtype=np.float32)
    if output_length == 0:
        return AudioBuffer(output, target_rate)

    ratio = audio.sample_rate / target_rate
    cutoff = min(1.0, target_rate / audio.sample_rate)
    source = audio.samples.astype(np.float64, copy=False)

    for output_index in range(output_length):
        position = output_index * ratio
        center = math.floor(position)
        first = max(0, center - half_width + 1)
        last = min(audio.frames, center + half_width + 1)
        indexes = np.arange(first, last, dtype=np.int64)
        distance = position - indexes
        window = 0.5 * (1.0 + np.cos(np.pi * distance / half_width))
        window[np.abs(distance) >= half_width] = 0.0
        weights = cutoff * np.sinc(cutoff * distance) * window
        weight_sum = float(weights.sum())
        if abs(weight_sum) < 1e-12:
            nearest = min(max(round(position), 0), audio.frames - 1)
            output[:, output_index] = source[:, nearest]
        else:
            output[:, output_index] = (source[:, indexes] @ weights / weight_sum).astype(np.float32)

    return AudioBuffer(output, target_rate)

