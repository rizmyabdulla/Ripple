"""Model / optimizer construction for CLI training."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ripple.baselines.streamvc import StreamVCBaseline, StreamVCConfig
from ripple.models.ripple_vc import RippleVC, RippleVCConfig
from ripple.training.trainer import TrainerConfig


def build_model(kind: str = "ripple", **overrides: Any) -> nn.Module:
    key = kind.casefold()
    if key in {"ripple", "ripple-vc", "edge"}:
        config = RippleVCConfig()
        return RippleVC(config)
    if key in {"baseline", "streamvc"}:
        return StreamVCBaseline(StreamVCConfig())
    raise ValueError(f"unknown model kind {kind!r}; choose ripple or baseline")


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def trainer_config_from_flags(
    *,
    device: str,
    precision: str = "fp32",
    gradient_accumulation: int = 1,
    ema_decay: float | None = 0.999,
) -> TrainerConfig:
    mapped = {"bf16-mixed": "bf16", "16-mixed": "fp16", "32": "fp32"}.get(
        precision, precision
    )
    return TrainerConfig(
        device=device,
        precision=mapped,
        gradient_accumulation=gradient_accumulation,
        ema_decay=ema_decay,
    )
