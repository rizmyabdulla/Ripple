"""Bounded-memory long-session stability harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import os
import time
from typing import Any, Callable, Iterable

import torch
from torch import Tensor


def _rss_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.WorkingSetSize)
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1 if os.uname().sysname == "Darwin" else 1024))


def _tensors(value: Any) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def _slope(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    count = len(values)
    mean_x = (count - 1) / 2
    mean_y = sum(values) / count
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    return numerator / denominator


@dataclass(frozen=True)
class LongSessionConfig:
    iterations: int = 1_000
    sample_every: int = 50
    reset_every: int | None = None
    packet_loss_probability: float = 0.0
    seed: int = 0
    collect_garbage: bool = True


@dataclass(frozen=True)
class LongSessionReport:
    iterations: int
    elapsed_seconds: float
    output_samples: int
    nan_inf_count: int
    maximum_state_norm: float
    rss_start_bytes: int
    rss_peak_bytes: int
    rss_end_bytes: int
    rss_slope_bytes_per_sample: float
    resets: int
    dropped_packets: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def run_long_session(
    step: Callable[[Tensor, Any], Tensor | tuple[Tensor, Any]],
    chunks: Iterable[Tensor],
    *,
    initial_state: Any = None,
    reset_state: Callable[[], Any] | None = None,
    config: LongSessionConfig | None = None,
) -> LongSessionReport:
    """Exercise a streaming step without retaining generated audio."""
    config = config or LongSessionConfig()
    if config.iterations <= 0 or config.sample_every <= 0:
        raise ValueError("iterations and sample_every must be positive")
    source = list(chunks)
    if not source:
        raise ValueError("chunks cannot be empty")
    generator = torch.Generator().manual_seed(config.seed)
    state = initial_state
    rss_samples = [_rss_bytes()]
    maximum_state_norm = 0.0
    nan_inf_count = output_samples = resets = dropped_packets = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(config.iterations):
            chunk = source[index % len(source)]
            if config.packet_loss_probability and torch.rand((), generator=generator).item() < config.packet_loss_probability:
                chunk = torch.zeros_like(chunk)
                dropped_packets += 1
            if config.reset_every and index and index % config.reset_every == 0:
                state = reset_state() if reset_state is not None else None
                resets += 1
            result = step(chunk, state)
            output, state = result if isinstance(result, tuple) else (result, state)
            output_samples += output.numel()
            nan_inf_count += int((~torch.isfinite(output)).sum())
            for tensor in _tensors(state):
                nan_inf_count += int((~torch.isfinite(tensor)).sum())
                if tensor.numel():
                    maximum_state_norm = max(
                        maximum_state_norm, float(torch.linalg.vector_norm(tensor.float()))
                    )
            if (index + 1) % config.sample_every == 0:
                if config.collect_garbage:
                    gc.collect()
                rss_samples.append(_rss_bytes())
    if rss_samples[-1] != _rss_bytes():
        rss_samples.append(_rss_bytes())
    return LongSessionReport(
        iterations=config.iterations,
        elapsed_seconds=time.perf_counter() - started,
        output_samples=output_samples,
        nan_inf_count=nan_inf_count,
        maximum_state_norm=maximum_state_norm,
        rss_start_bytes=rss_samples[0],
        rss_peak_bytes=max(rss_samples),
        rss_end_bytes=rss_samples[-1],
        rss_slope_bytes_per_sample=_slope(rss_samples),
        resets=resets,
        dropped_packets=dropped_packets,
    )
