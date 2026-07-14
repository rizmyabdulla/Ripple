"""Streaming-aware activation calibration and coverage reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import Tensor, nn


def _tensor_values(value: Any) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_values(item)


@dataclass(frozen=True)
class CalibrationRange:
    name: str
    minimum: float
    maximum: float
    observations: int
    finite: bool


@dataclass(frozen=True)
class CalibrationReport:
    ranges: tuple[CalibrationRange, ...]
    batches: int
    eligible_modules: int
    observed_modules: int

    @property
    def coverage(self) -> float:
        return self.observed_modules / self.eligible_modules if self.eligible_modules else 1.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["coverage"] = self.coverage
        return result


def _default_forward(model: nn.Module, batch: Any) -> Any:
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, (list, tuple)):
        return model(*batch)
    return model(batch)


def collect_calibration_ranges(
    model: nn.Module,
    data: Iterable[Any],
    *,
    forward: Callable[[nn.Module, Any], Any] = _default_forward,
    module_types: Sequence[type[nn.Module]] = (nn.Linear, nn.Conv1d, nn.Conv2d),
    max_batches: int | None = None,
) -> CalibrationReport:
    """Observe selected module outputs without changing model numerics."""
    statistics: dict[str, list[float | int | bool]] = {}
    hooks: list[Any] = []
    eligible = 0

    def hook_for(name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def observe(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            values = [value for value in _tensor_values(output) if value.numel()]
            if not values:
                return
            minimum = min(float(value.detach().amin()) for value in values)
            maximum = max(float(value.detach().amax()) for value in values)
            finite = all(bool(torch.isfinite(value).all()) for value in values)
            current = statistics.setdefault(name, [minimum, maximum, 0, True])
            current[0] = min(float(current[0]), minimum)
            current[1] = max(float(current[1]), maximum)
            current[2] = int(current[2]) + 1
            current[3] = bool(current[3]) and finite

        return observe

    for name, module in model.named_modules():
        if name and isinstance(module, tuple(module_types)):
            eligible += 1
            hooks.append(module.register_forward_hook(hook_for(name)))
    batches = 0
    previous_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in data:
                forward(model, batch)
                batches += 1
                if max_batches is not None and batches >= max_batches:
                    break
    finally:
        for hook in hooks:
            hook.remove()
        model.train(previous_training)
    ranges = tuple(
        CalibrationRange(name, float(values[0]), float(values[1]), int(values[2]), bool(values[3]))
        for name, values in sorted(statistics.items())
    )
    return CalibrationReport(ranges, batches, eligible, len(ranges))
