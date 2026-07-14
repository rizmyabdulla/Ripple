"""Numerically stable, confidence-gated online statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class WelfordState:
    count: Tensor
    mean: Tensor
    m2: Tensor

    @property
    def variance(self) -> Tensor:
        return self.m2 / torch.clamp(self.count - 1.0, min=1.0)

    def detach(self) -> "WelfordState":
        return WelfordState(
            self.count.detach(), self.mean.detach(), self.m2.detach()
        )


class WelfordRunningStats:
    """Per-batch, per-feature Welford accumulator.

    A finite ``max_count`` turns updates into a stable bounded-rate adaptation
    after the exact warm-up period, preventing very long calls from freezing
    the estimate.
    """

    def __init__(
        self,
        features: int,
        *,
        prior_mean: float | Tensor = 0.0,
        prior_variance: float | Tensor = 1.0,
        prior_count: float = 2.0,
        eps: float = 1e-5,
        max_count: float | None = 10_000.0,
    ) -> None:
        if features < 1 or prior_count < 0:
            raise ValueError("features must be positive and prior_count non-negative")
        self.features = features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.prior_count = prior_count
        self.eps = eps
        self.max_count = max_count

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> WelfordState:
        shape = (batch_size, self.features)
        mean = torch.as_tensor(
            self.prior_mean, device=device, dtype=dtype
        ).expand(shape).clone()
        variance = torch.as_tensor(
            self.prior_variance, device=device, dtype=dtype
        ).expand(shape)
        count = torch.full(shape, self.prior_count, device=device, dtype=dtype)
        m2 = variance * torch.clamp(count - 1.0, min=0.0)
        return WelfordState(count, mean, m2)

    def update(
        self,
        values: Tensor,
        state: WelfordState,
        mask: Tensor | None = None,
    ) -> WelfordState:
        """Update from ``[batch, features]`` or ``[batch, features, time]``."""

        if values.ndim == 2:
            values = values.unsqueeze(-1)
        if values.ndim != 3 or values.shape[1] != self.features:
            raise ValueError(
                f"values must be [batch, {self.features}, time], "
                f"got {tuple(values.shape)}"
            )
        self._validate_state(values, state)
        if mask is None:
            mask = torch.ones(
                values.shape[0], values.shape[-1], device=values.device, dtype=torch.bool
            )
        if mask.ndim == 1:
            mask = mask.unsqueeze(-1)
        if tuple(mask.shape) != (values.shape[0], values.shape[-1]):
            raise ValueError("mask must be shaped [batch, time]")

        count, mean, m2 = state.count, state.mean, state.m2
        for index in range(values.shape[-1]):
            active = mask[:, index : index + 1].to(values.dtype)
            sample = values[:, :, index]
            if self.max_count is None:
                new_count = count + active
                delta = sample - mean
                mean = mean + active * delta / torch.clamp(new_count, min=1.0)
                delta2 = sample - mean
                m2 = m2 + active * delta * delta2
                count = new_count
            else:
                exact = (count < self.max_count).to(values.dtype)
                next_count = torch.minimum(
                    count + active,
                    torch.full_like(count, self.max_count),
                )
                alpha = active / torch.clamp(next_count, min=1.0)
                delta = sample - mean
                next_mean = mean + alpha * delta
                exact_m2 = m2 + active * delta * (sample - next_mean)
                ema_variance = (1.0 - alpha) * self.variance(
                    WelfordState(count, mean, m2)
                ) + alpha * delta.square()
                ema_m2 = ema_variance * torch.clamp(next_count - 1.0, min=1.0)
                mean = next_mean
                m2 = exact * exact_m2 + (1.0 - exact) * ema_m2
                count = next_count
        return WelfordState(count, mean, m2)

    def variance(self, state: WelfordState) -> Tensor:
        return state.m2 / torch.clamp(state.count - 1.0, min=1.0)

    def normalize(self, values: Tensor, state: WelfordState) -> Tensor:
        if values.ndim == 2:
            mean = state.mean
            variance = self.variance(state)
        elif values.ndim == 3:
            mean = state.mean.unsqueeze(-1)
            variance = self.variance(state).unsqueeze(-1)
        else:
            raise ValueError("values must be rank 2 or rank 3")
        return (values - mean) * torch.rsqrt(variance + self.eps)

    def decay(self, state: WelfordState, amount: float) -> WelfordState:
        if not 0.0 <= amount <= 1.0:
            raise ValueError("decay amount must be in [0, 1]")
        prior = self.initial_state(
            state.mean.shape[0], device=state.mean.device, dtype=state.mean.dtype
        )
        return WelfordState(
            torch.lerp(state.count, prior.count, amount),
            torch.lerp(state.mean, prior.mean, amount),
            torch.lerp(state.m2, prior.m2, amount),
        )

    def _validate_state(self, values: Tensor, state: WelfordState) -> None:
        expected = (values.shape[0], self.features)
        for tensor in (state.count, state.mean, state.m2):
            if tuple(tensor.shape) != expected:
                raise ValueError(f"running-stat state must have shape {expected}")


RunningStats = WelfordRunningStats
