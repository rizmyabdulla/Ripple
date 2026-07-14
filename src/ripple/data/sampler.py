"""Seeded temperature-balanced manifest sampling."""

from __future__ import annotations

from collections import Counter

import numpy as np
import numpy.typing as npt

from ripple.data.manifest import AudioRecord


def sampling_probabilities(
    records: tuple[AudioRecord, ...],
    *,
    temperature: float = 0.7,
) -> npt.NDArray[np.float64]:
    if not records:
        raise ValueError("records cannot be empty")
    if not 0.0 < temperature <= 1.0:
        raise ValueError("temperature must be in (0, 1]")
    counts = Counter(record.language.lower() for record in records)
    weights = np.asarray(
        [counts[record.language.lower()] ** (temperature - 1.0) for record in records],
        dtype=np.float64,
    )
    weights /= weights.sum()
    weights.setflags(write=False)
    return weights


def sample_indices(
    records: tuple[AudioRecord, ...],
    count: int,
    *,
    seed: int,
    temperature: float = 0.7,
) -> npt.NDArray[np.int64]:
    if count < 0 or seed < 0:
        raise ValueError("count and seed must be non-negative")
    probabilities = sampling_probabilities(records, temperature=temperature)
    generator = np.random.default_rng(seed)
    indexes = generator.choice(len(records), size=count, replace=True, p=probabilities)
    result = np.asarray(indexes, dtype=np.int64)
    result.setflags(write=False)
    return result

