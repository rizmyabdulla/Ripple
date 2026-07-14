"""Batch collation helpers for Ripple training."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def collate_waveforms(batch: list[dict[str, Any]]) -> dict[str, Any]:
    waveforms = torch.stack([item["waveform"] for item in batch], dim=0).unsqueeze(1)
    result: dict[str, Any] = {
        "waveform": waveforms,
        "record_id": [item["record_id"] for item in batch],
        "speaker_id": [item["speaker_id"] for item in batch],
        "language": [item["language"] for item in batch],
        "sample_rate": batch[0]["sample_rate"],
    }
    if all("teacher" in item for item in batch):
        teachers = [item["teacher"] for item in batch]
        # Teacher may be [T, C] or [1, T, C] — normalize to [B, T, C]
        normalized: list[Tensor] = []
        for teacher in teachers:
            tensor = teacher.float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            normalized.append(tensor)
        max_frames = max(item.shape[0] for item in normalized)
        channels = normalized[0].shape[-1]
        padded = torch.zeros(len(normalized), max_frames, channels)
        for index, item in enumerate(normalized):
            padded[index, : item.shape[0]] = item
        result["teacher"] = padded
    return result
