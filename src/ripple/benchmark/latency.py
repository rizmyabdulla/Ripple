"""Warm/cold latency, deadline and real-time-factor measurement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
import time
from typing import Any, Callable

import torch


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class LatencyConfig:
    warmup_iterations: int = 10
    measured_iterations: int = 100
    audio_seconds_per_call: float = 0.02
    deadline_seconds: float | None = 0.02
    algorithmic_lookahead_ms: float = 0.0
    frame_cadence_ms: float = 20.0


@dataclass(frozen=True)
class LatencyResult:
    calls: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    minimum_ms: float
    maximum_ms: float
    mean_rtf: float
    p95_rtf: float
    process_cpu_seconds: float
    wall_seconds: float
    missed_deadlines: int
    algorithmic_lookahead_ms: float
    frame_cadence_ms: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def benchmark_latency(
    step: Callable[[], Any],
    config: LatencyConfig | None = None,
) -> LatencyResult:
    config = config or LatencyConfig()
    if config.measured_iterations <= 0 or config.audio_seconds_per_call <= 0:
        raise ValueError("measured iterations and audio duration must be positive")
    with torch.inference_mode():
        for _ in range(config.warmup_iterations):
            step()
        _synchronize()
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        timings: list[float] = []
        for _ in range(config.measured_iterations):
            _synchronize()
            started = time.perf_counter()
            step()
            _synchronize()
            timings.append(time.perf_counter() - started)
        wall_seconds = time.perf_counter() - wall_started
        process_cpu_seconds = time.process_time() - cpu_started
    rtfs = [duration / config.audio_seconds_per_call for duration in timings]
    deadline = config.deadline_seconds
    return LatencyResult(
        calls=len(timings),
        mean_ms=statistics.fmean(timings) * 1000,
        p50_ms=percentile(timings, 0.50) * 1000,
        p95_ms=percentile(timings, 0.95) * 1000,
        p99_ms=percentile(timings, 0.99) * 1000,
        minimum_ms=min(timings) * 1000,
        maximum_ms=max(timings) * 1000,
        mean_rtf=statistics.fmean(rtfs),
        p95_rtf=percentile(rtfs, 0.95),
        process_cpu_seconds=process_cpu_seconds,
        wall_seconds=wall_seconds,
        missed_deadlines=sum(duration > deadline for duration in timings) if deadline else 0,
        algorithmic_lookahead_ms=config.algorithmic_lookahead_ms,
        frame_cadence_ms=config.frame_cadence_ms,
    )


def benchmark_cold_start(factory: Callable[[], Any], first_call: Callable[[Any], Any]) -> dict[str, float]:
    started = time.perf_counter()
    backend = factory()
    _synchronize()
    loaded = time.perf_counter()
    first_call(backend)
    _synchronize()
    completed = time.perf_counter()
    return {
        "load_ms": (loaded - started) * 1000,
        "first_call_ms": (completed - loaded) * 1000,
        "load_and_first_call_ms": (completed - started) * 1000,
    }
