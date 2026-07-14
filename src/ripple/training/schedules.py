"""Small, serializable schedules for staged loss weighting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearWarmup:
    start_step: int
    end_step: int
    start_value: float = 0.0
    end_value: float = 1.0

    def __call__(self, step: int) -> float:
        if self.end_step <= self.start_step:
            raise ValueError("end_step must exceed start_step")
        fraction = min(1.0, max(0.0, (step - self.start_step) / (self.end_step - self.start_step)))
        return self.start_value + fraction * (self.end_value - self.start_value)


@dataclass
class BoundedLossWeight:
    """EMA-based multiplicative adaptation constrained to a safe range."""

    value: float
    minimum: float
    maximum: float
    momentum: float = 0.95
    target: float = 1.0
    _ema: float | None = None

    def update(self, observed: float) -> float:
        if self._ema is None:
            self._ema = observed
        else:
            self._ema = self.momentum * self._ema + (1 - self.momentum) * observed
        if self._ema > 0:
            proposed = self.value * (self.target / self._ema) ** (1 - self.momentum)
            self.value = min(self.maximum, max(self.minimum, proposed))
        return self.value
