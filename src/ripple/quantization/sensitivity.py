"""Backend-independent per-layer fake-quantization sensitivity analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn


def fake_quantize_symmetric(tensor: Tensor, num_bits: int = 8) -> Tensor:
    if not 2 <= num_bits <= 16:
        raise ValueError("num_bits must be in [2, 16]")
    maximum = tensor.detach().abs().amax()
    if maximum == 0:
        return tensor.detach().clone()
    quantization_max = 2 ** (num_bits - 1) - 1
    scale = maximum / quantization_max
    return torch.clamp(torch.round(tensor / scale), -quantization_max, quantization_max) * scale


@dataclass(frozen=True)
class LayerSensitivity:
    name: str
    baseline_score: float
    quantized_score: float
    degradation: float
    num_bits: int
    parameter_count: int

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def analyze_sensitivity(
    model: nn.Module,
    evaluate: Callable[[nn.Module], float],
    *,
    num_bits: int = 8,
    module_types: Sequence[type[nn.Module]] = (nn.Linear, nn.Conv1d, nn.Conv2d),
    higher_is_better: bool = True,
) -> list[LayerSensitivity]:
    """Quantize one layer at a time and rank quality degradation."""
    baseline = float(evaluate(model))
    results: list[LayerSensitivity] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, tuple(module_types)):
            continue
        parameters = list(module.parameters(recurse=False))
        if not parameters:
            continue
        originals = [parameter.detach().clone() for parameter in parameters]
        try:
            with torch.no_grad():
                for parameter in parameters:
                    parameter.copy_(fake_quantize_symmetric(parameter, num_bits))
            score = float(evaluate(model))
        finally:
            with torch.no_grad():
                for parameter, original in zip(parameters, originals):
                    parameter.copy_(original)
        degradation = baseline - score if higher_is_better else score - baseline
        results.append(
            LayerSensitivity(
                name=name,
                baseline_score=baseline,
                quantized_score=score,
                degradation=degradation,
                num_bits=num_bits,
                parameter_count=sum(parameter.numel() for parameter in parameters),
            )
        )
    return sorted(results, key=lambda item: item.degradation, reverse=True)
