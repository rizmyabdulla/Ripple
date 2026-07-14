"""Model-agnostic speaker embedding metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cosine_similarity(embeddings: Tensor, references: Tensor) -> Tensor:
    return F.cosine_similarity(embeddings, references, dim=-1)


def target_similarity(converted: Tensor, target: Tensor) -> float:
    return float(cosine_similarity(converted, target).mean())


def source_leakage(converted: Tensor, source: Tensor) -> float:
    return float(cosine_similarity(converted, source).mean())


def verification_accuracy(
    first: Tensor,
    second: Tensor,
    labels: Tensor,
    *,
    threshold: float,
) -> float:
    predictions = cosine_similarity(first, second) >= threshold
    return float((predictions == labels.bool()).float().mean())


def equal_error_rate(genuine_scores: Tensor, impostor_scores: Tensor) -> tuple[float, float]:
    """Return approximate EER and threshold using all observed scores."""
    scores = torch.cat((genuine_scores.flatten(), impostor_scores.flatten())).sort().values
    thresholds = torch.cat((scores[:1] - 1e-6, (scores[:-1] + scores[1:]) / 2, scores[-1:] + 1e-6))
    false_reject = (genuine_scores[:, None] < thresholds[None, :]).float().mean(dim=0)
    false_accept = (impostor_scores[:, None] >= thresholds[None, :]).float().mean(dim=0)
    index = torch.argmin((false_reject - false_accept).abs())
    return float((false_reject[index] + false_accept[index]) / 2), float(thresholds[index])
