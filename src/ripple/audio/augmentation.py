"""Seeded training-only waveform augmentation primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ripple.audio.io import AudioBuffer


@dataclass(frozen=True, slots=True)
class AugmentationResult:
    audio: AudioBuffer
    operation: str
    seed: int
    time_labels_invalidated: bool


def apply_gain(audio: AudioBuffer, gain_db: float) -> AugmentationResult:
    if not np.isfinite(gain_db):
        raise ValueError("gain_db must be finite")
    scale = float(10.0 ** (gain_db / 20.0))
    samples = np.clip(audio.samples.astype(np.float64) * scale, -1.0, 1.0).astype(np.float32)
    return AugmentationResult(AudioBuffer(samples, audio.sample_rate), "gain", 0, False)


def add_white_noise(audio: AudioBuffer, snr_db: float, *, seed: int) -> AugmentationResult:
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    signal_power = float(np.mean(np.square(audio.samples.astype(np.float64))))
    if signal_power == 0.0:
        return AugmentationResult(audio, "white_noise", seed, False)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal(audio.samples.shape) * np.sqrt(noise_power)
    samples = np.clip(audio.samples.astype(np.float64) + noise, -1.0, 1.0).astype(np.float32)
    return AugmentationResult(AudioBuffer(samples, audio.sample_rate), "white_noise", seed, False)

