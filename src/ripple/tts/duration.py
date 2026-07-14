"""Monotonic token-duration prediction and 50 Hz length regulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DurationOutput:
    log_durations: Tensor
    durations: Tensor


class MonotonicDurationPredictor(nn.Module):
    """Compact non-autoregressive duration predictor.

    Predicted durations are integer RIF frames and therefore cannot reorder
    tokens. Padding positions always receive zero frames.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        channels: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_frames_per_token: int = 250,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or channels <= 0:
            raise ValueError("dimensions must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")
        self.max_frames_per_token = max_frames_per_token
        self.network = nn.Sequential(
            nn.Conv1d(hidden_dim, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
        )
        self.projection = nn.Linear(channels, 1)

    def forward(
        self,
        encoded: Tensor,
        padding_mask: Tensor | None = None,
        *,
        pace: float = 1.0,
    ) -> DurationOutput:
        if encoded.ndim != 3:
            raise ValueError("encoded must have shape [batch, tokens, hidden]")
        if pace <= 0:
            raise ValueError("pace must be positive")
        hidden = encoded.transpose(1, 2)
        for layer in self.network:
            if isinstance(layer, nn.LayerNorm):
                hidden = layer(hidden.transpose(1, 2)).transpose(1, 2)
            else:
                hidden = layer(hidden)
        log_durations = self.projection(hidden.transpose(1, 2)).squeeze(-1)
        durations = torch.round(torch.expm1(log_durations).clamp_min(0) / pace)
        durations = durations.clamp(min=1, max=self.max_frames_per_token).long()
        if padding_mask is not None:
            if padding_mask.shape != durations.shape:
                raise ValueError("padding_mask must have shape [batch, tokens]")
            durations = durations.masked_fill(padding_mask.bool(), 0)
            log_durations = log_durations.masked_fill(padding_mask.bool(), 0)
        return DurationOutput(log_durations, durations)


def length_regulate(encoded: Tensor, durations: Tensor) -> tuple[Tensor, Tensor]:
    """Expand token states and return a padding mask for batched RIF frames."""

    if encoded.ndim != 3 or durations.shape != encoded.shape[:2]:
        raise ValueError("expected encoded [B, T, H] and durations [B, T]")
    if (durations < 0).any():
        raise ValueError("durations cannot be negative")
    expanded = [
        torch.repeat_interleave(sequence, count.long(), dim=0)
        for sequence, count in zip(encoded, durations, strict=True)
    ]
    lengths = torch.tensor([item.shape[0] for item in expanded], device=encoded.device)
    max_length = int(lengths.max().item()) if len(expanded) else 0
    output = encoded.new_zeros((encoded.shape[0], max_length, encoded.shape[2]))
    padding_mask = torch.ones(
        (encoded.shape[0], max_length), dtype=torch.bool, device=encoded.device
    )
    for index, item in enumerate(expanded):
        output[index, : item.shape[0]] = item
        padding_mask[index, : item.shape[0]] = False
    return output, padding_mask
