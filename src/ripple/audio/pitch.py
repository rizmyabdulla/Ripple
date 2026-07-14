"""Deterministic YIN pitch fallback used for labels and ablations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class PitchEstimate:
    frequency_hz: float
    voiced_probability: float
    periodicity: float


def estimate_yin(
    samples: npt.ArrayLike,
    sample_rate: int,
    *,
    min_frequency: float = 60.0,
    max_frequency: float = 500.0,
    threshold: float = 0.15,
) -> PitchEstimate:
    signal = np.asarray(samples, dtype=np.float64).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 < min_frequency < max_frequency < sample_rate / 2:
        raise ValueError("frequency range must be positive and below Nyquist")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    if signal.size < 2 * int(sample_rate / min_frequency):
        raise ValueError("signal is too short for the requested minimum frequency")
    if not np.isfinite(signal).all():
        raise ValueError("samples must be finite")
    signal = signal - signal.mean()
    if float(np.sqrt(np.mean(np.square(signal)))) < 1e-5:
        return PitchEstimate(0.0, 0.0, 0.0)

    min_lag = max(2, int(sample_rate / max_frequency))
    max_lag = min(int(sample_rate / min_frequency), signal.size // 2)
    difference = np.zeros(max_lag + 1, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        delta = signal[:-lag] - signal[lag:]
        difference[lag] = np.dot(delta, delta)

    cumulative = np.ones_like(difference)
    running_sum = 0.0
    for lag in range(1, max_lag + 1):
        running_sum += difference[lag]
        cumulative[lag] = difference[lag] * lag / max(running_sum, 1e-12)

    candidates = np.flatnonzero(cumulative[min_lag : max_lag + 1] < threshold)
    if candidates.size:
        lag = int(candidates[0] + min_lag)
        while lag < max_lag and cumulative[lag + 1] < cumulative[lag]:
            lag += 1
    else:
        lag = int(np.argmin(cumulative[min_lag : max_lag + 1]) + min_lag)

    periodicity = float(np.clip(1.0 - cumulative[lag], 0.0, 1.0))
    if 1 <= lag < max_lag:
        left, center, right = cumulative[lag - 1 : lag + 2]
        denominator = 2.0 * (2.0 * center - left - right)
        offset = 0.0 if abs(denominator) < 1e-12 else float((right - left) / denominator)
        lag_value = lag + float(np.clip(offset, -0.5, 0.5))
    else:
        lag_value = float(lag)
    voiced_probability = float(np.clip((periodicity - 0.5) / 0.5, 0.0, 1.0))
    frequency = float(sample_rate / lag_value) if voiced_probability > 0.0 else 0.0
    return PitchEstimate(frequency, voiced_probability, periodicity)

