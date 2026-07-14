"""Lightweight waveform and spectral quality metrics."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


def signal_to_noise_ratio(reference: Tensor, estimate: Tensor, eps: float = 1e-8) -> float:
    signal = reference.float().square().mean()
    noise = (reference.float() - estimate.float()).square().mean()
    return float(10 * torch.log10((signal + eps) / (noise + eps)))


def scale_invariant_sdr(reference: Tensor, estimate: Tensor, eps: float = 1e-8) -> float:
    reference = reference.float().flatten()
    estimate = estimate.float().flatten()
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    projection = (estimate @ reference) * reference / (reference.square().sum() + eps)
    noise = estimate - projection
    return float(10 * torch.log10((projection.square().sum() + eps) / (noise.square().sum() + eps)))


def log_spectral_distance(
    reference: Tensor,
    estimate: Tensor,
    *,
    fft_size: int = 512,
    hop_size: int = 128,
    eps: float = 1e-7,
) -> float:
    length = max(reference.shape[-1], fft_size)
    reference = F.pad(reference, (0, length - reference.shape[-1]))
    estimate = F.pad(estimate, (0, length - estimate.shape[-1]))
    window = torch.hann_window(fft_size, device=reference.device, dtype=reference.dtype)
    reference_db = 20 * torch.log10(
        torch.stft(reference, fft_size, hop_size, window=window, return_complex=True).abs() + eps
    )
    estimate_db = 20 * torch.log10(
        torch.stft(estimate, fft_size, hop_size, window=window, return_complex=True).abs() + eps
    )
    return float(torch.sqrt(torch.mean((reference_db - estimate_db) ** 2)))


def boundary_discontinuity(chunks: Sequence[Tensor]) -> dict[str, float]:
    if len(chunks) < 2:
        return {"mean_jump": 0.0, "max_jump": 0.0}
    jumps = torch.stack(
        [(chunks[index][..., -1] - chunks[index + 1][..., 0]).abs().float().mean() for index in range(len(chunks) - 1)]
    )
    return {"mean_jump": float(jumps.mean()), "max_jump": float(jumps.max())}


def waveform_report(reference: Tensor, estimate: Tensor) -> dict[str, float]:
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must be aligned and have identical shape")
    return {
        "snr_db": signal_to_noise_ratio(reference, estimate),
        "si_sdr_db": scale_invariant_sdr(reference, estimate),
        "mae": float(torch.mean(torch.abs(reference - estimate))),
        "peak_error": float(torch.max(torch.abs(reference - estimate))),
        "finite_fraction": float(torch.isfinite(estimate).float().mean()),
        "rms": math.sqrt(float(estimate.float().square().mean())),
    }
