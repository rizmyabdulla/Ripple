"""Objective prosody preservation metrics."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _select(first: Tensor, second: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    if mask is not None:
        first, second = first[mask.bool()], second[mask.bool()]
    return first.float().flatten(), second.float().flatten()


def pearson_correlation(first: Tensor, second: Tensor, mask: Tensor | None = None) -> float:
    first, second = _select(first, second, mask)
    if first.numel() < 2:
        return float("nan")
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    return float((first @ second) / denominator) if denominator > 0 else float("nan")


def spearman_correlation(first: Tensor, second: Tensor, mask: Tensor | None = None) -> float:
    first, second = _select(first, second, mask)
    if first.numel() < 2:
        return float("nan")
    first_rank = torch.argsort(torch.argsort(first)).float()
    second_rank = torch.argsort(torch.argsort(second)).float()
    return pearson_correlation(first_rank, second_rank)


def root_mean_squared_error(first: Tensor, second: Tensor, mask: Tensor | None = None) -> float:
    first, second = _select(first, second, mask)
    return math.sqrt(float(torch.mean((first - second) ** 2))) if first.numel() else float("nan")


def voicing_f1(predicted: Tensor, target: Tensor) -> float:
    predicted = predicted.bool()
    target = target.bool()
    true_positive = (predicted & target).sum()
    false_positive = (predicted & ~target).sum()
    false_negative = (~predicted & target).sum()
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 1.0


def prosody_report(
    predicted_f0: Tensor,
    target_f0: Tensor,
    predicted_voicing: Tensor,
    target_voicing: Tensor,
    *,
    predicted_energy: Tensor | None = None,
    target_energy: Tensor | None = None,
) -> dict[str, float]:
    voiced = predicted_voicing.bool() & target_voicing.bool()
    report = {
        "f0_pearson": pearson_correlation(predicted_f0, target_f0, voiced),
        "f0_spearman": spearman_correlation(predicted_f0, target_f0, voiced),
        "f0_rmse": root_mean_squared_error(predicted_f0, target_f0, voiced),
        "voicing_f1": voicing_f1(predicted_voicing, target_voicing),
    }
    if predicted_energy is not None and target_energy is not None:
        report["energy_pearson"] = pearson_correlation(predicted_energy, target_energy)
    return report
