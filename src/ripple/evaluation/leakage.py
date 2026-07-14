"""Source-identity leakage probes independent of an embedding model."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def classifier_accuracy(logits: Tensor, labels: Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean())


def source_target_margin(converted: Tensor, source: Tensor, target: Tensor) -> float:
    source_score = F.cosine_similarity(converted, source, dim=-1)
    target_score = F.cosine_similarity(converted, target, dim=-1)
    return float((target_score - source_score).mean())


def leakage_report(
    converted: Tensor,
    source: Tensor,
    target: Tensor,
    *,
    probe_logits: Tensor | None = None,
    source_labels: Tensor | None = None,
) -> dict[str, float]:
    report = {
        "converted_source_cosine": float(F.cosine_similarity(converted, source, dim=-1).mean()),
        "converted_target_cosine": float(F.cosine_similarity(converted, target, dim=-1).mean()),
        "target_source_margin": source_target_margin(converted, source, target),
    }
    if probe_logits is not None and source_labels is not None:
        report["source_probe_accuracy"] = classifier_accuracy(probe_logits, source_labels)
    return report
