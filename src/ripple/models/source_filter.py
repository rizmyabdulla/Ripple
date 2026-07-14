"""Deterministic harmonic/noise excitation with explicit oscillator phase."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SourceFilterConfig:
    sample_rate: int = 24_000
    frame_rate: int = 50
    harmonics: int = 8
    noise_scale: float = 0.08


@dataclass(frozen=True)
class SourceFilterState:
    phase: Tensor
    sample_index: Tensor


class HarmonicNoiseSource(nn.Module):
    def __init__(self, config: SourceFilterConfig | None = None) -> None:
        super().__init__()
        self.config = config or SourceFilterConfig()
        amplitudes = 1.0 / torch.arange(
            1, self.config.harmonics + 1, dtype=torch.float32
        )
        self.register_buffer(
            "harmonic_amplitudes", amplitudes / amplitudes.sum(), persistent=True
        )

    @property
    def hop_samples(self) -> int:
        return self.config.sample_rate // self.config.frame_rate

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> SourceFilterState:
        return SourceFilterState(
            torch.zeros(batch_size, device=device, dtype=dtype),
            torch.zeros(batch_size, device=device, dtype=torch.int64),
        )

    def forward(
        self,
        f0_hz: Tensor,
        voiced_probability: Tensor,
        periodicity: Tensor,
    ) -> Tensor:
        state = self.initial_state(
            f0_hz.shape[0], device=f0_hz.device, dtype=f0_hz.dtype
        )
        output, _ = self.step(
            f0_hz, voiced_probability, periodicity, state
        )
        return output

    def step(
        self,
        f0_hz: Tensor,
        voiced_probability: Tensor,
        periodicity: Tensor,
        state: SourceFilterState,
    ) -> tuple[Tensor, SourceFilterState]:
        self._validate(f0_hz, voiced_probability, periodicity, state)
        f0 = f0_hz.repeat_interleave(self.hop_samples, dim=-1).squeeze(1)
        voiced = voiced_probability.repeat_interleave(
            self.hop_samples, dim=-1
        ).squeeze(1)
        confidence = periodicity.repeat_interleave(
            self.hop_samples, dim=-1
        ).squeeze(1)

        increments = f0 / float(self.config.sample_rate)
        phase = state.phase.unsqueeze(-1) + torch.cumsum(increments, dim=-1)
        phase = phase - torch.floor(phase)
        harmonic_numbers = torch.arange(
            1,
            self.config.harmonics + 1,
            device=f0.device,
            dtype=f0.dtype,
        ).view(1, -1, 1)
        harmonic = torch.sin(
            2.0 * torch.pi * phase.unsqueeze(1) * harmonic_numbers
        )
        amplitudes = self.harmonic_amplitudes.to(
            device=f0.device, dtype=f0.dtype
        ).view(1, -1, 1)
        harmonic = (harmonic * amplitudes).sum(dim=1)

        offsets = torch.arange(
            f0.shape[-1], device=f0.device, dtype=torch.int64
        ).unsqueeze(0)
        absolute_index = state.sample_index.unsqueeze(-1) + offsets
        # Stateless pseudo-noise gives deterministic grouping-independent audio.
        noise_seed = torch.sin(
            absolute_index.to(f0.dtype) * 12.9898 + 78.233
        ) * 43758.5453
        noise = (noise_seed - torch.floor(noise_seed)) * 2.0 - 1.0
        harmonic_gain = voiced * confidence
        noise_gain = (1.0 - harmonic_gain) * self.config.noise_scale
        excitation = harmonic_gain * harmonic + noise_gain * noise

        next_state = SourceFilterState(
            phase[:, -1],
            state.sample_index + f0.shape[-1],
        )
        return excitation.unsqueeze(1), next_state

    @staticmethod
    def _validate(
        f0_hz: Tensor,
        voiced_probability: Tensor,
        periodicity: Tensor,
        state: SourceFilterState,
    ) -> None:
        if f0_hz.ndim != 3 or f0_hz.shape[1] != 1:
            raise ValueError("F0 must be [batch, 1, frames]")
        if voiced_probability.shape != f0_hz.shape or periodicity.shape != f0_hz.shape:
            raise ValueError("source-filter controls must share shape")
        if state.phase.shape != (f0_hz.shape[0],):
            raise ValueError("oscillator phase state has wrong shape")


HarmonicSource = HarmonicNoiseSource
OscillatorState = SourceFilterState
