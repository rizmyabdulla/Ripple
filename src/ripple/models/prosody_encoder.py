"""Zero-lookahead causal prosody estimator with online normalization."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
from torch import Tensor, nn

from ripple.models.analysis_encoder import ChannelRMSNorm
from ripple.streaming.cached_conv import CachedCausalConv1d, Conv1dState
from ripple.streaming.running_stats import WelfordRunningStats, WelfordState


@dataclass(frozen=True)
class ProsodyEncoderConfig:
    sample_rate: int = 24_000
    frame_rate: int = 50
    channels: int = 48
    strides: tuple[int, ...] = (3, 4, 5, 8)
    kernel_size: int = 7
    confidence_threshold: float = 0.55
    prior_log_f0: float = 5.3
    prior_log_energy: float = -4.0
    prior_f0_variance: float = 0.16
    prior_energy_variance: float = 2.0

    def __post_init__(self) -> None:
        if prod(self.strides) != self.sample_rate // self.frame_rate:
            raise ValueError("prosody strides must reduce audio to frame_rate")


@dataclass(frozen=True)
class ProsodyState:
    convolutions: tuple[Conv1dState, ...]
    statistics: WelfordState
    previous_raw: Tensor


@dataclass(frozen=True)
class ProsodyOutput:
    """RIF prosody values and diagnostic physical quantities."""

    values: Tensor
    log_f0: Tensor
    f0_hz: Tensor
    voiced_probability: Tensor
    periodicity: Tensor
    log_energy: Tensor


class CausalProsodyEstimator(nn.Module):
    def __init__(self, config: ProsodyEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or ProsodyEncoderConfig()
        cfg = self.config
        convolutions: list[CachedCausalConv1d] = [
            CachedCausalConv1d(1, cfg.channels, cfg.kernel_size)
        ]
        for stride in cfg.strides:
            convolutions.append(
                CachedCausalConv1d(
                    cfg.channels,
                    cfg.channels,
                    cfg.kernel_size,
                    stride=stride,
                    groups=cfg.channels,
                )
            )
        self.convolutions = nn.ModuleList(convolutions)
        self.mixers = nn.ModuleList(
            [nn.Conv1d(cfg.channels, cfg.channels, 1) for _ in convolutions]
        )
        self.norms = nn.ModuleList(
            [ChannelRMSNorm(cfg.channels) for _ in convolutions]
        )
        self.head = nn.Conv1d(cfg.channels, 4, 1)
        prior_mean = torch.tensor([cfg.prior_log_f0, cfg.prior_log_energy])
        prior_variance = torch.tensor(
            [cfg.prior_f0_variance, cfg.prior_energy_variance]
        )
        self.running_stats = WelfordRunningStats(
            2,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            prior_count=2.0,
        )

    @property
    def chunk_samples(self) -> int:
        return self.config.sample_rate // self.config.frame_rate

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ProsodyState:
        reference = self.head.weight
        resolved_dtype = dtype if dtype is not None else reference.dtype
        resolved_device = device if device is not None else reference.device
        convolutions = tuple(
            convolution.initial_state(
                batch_size, device=resolved_device, dtype=resolved_dtype
            )
            for convolution in self.convolutions
        )
        statistics = self.running_stats.initial_state(
            batch_size, device=resolved_device, dtype=resolved_dtype
        )
        previous = torch.tensor(
            [
                self.config.prior_log_f0,
                0.0,
                0.0,
                self.config.prior_log_energy,
            ],
            device=resolved_device,
            dtype=resolved_dtype,
        ).view(1, 4).expand(batch_size, -1).clone()
        return ProsodyState(convolutions, statistics, previous)

    def _network(self, pcm: Tensor) -> Tensor:
        x = pcm
        for convolution, mixer, norm in zip(
            self.convolutions, self.mixers, self.norms
        ):
            x = torch.nn.functional.silu(mixer(norm(convolution(x))))
        return self.head(x)

    def _network_step(
        self, pcm: Tensor, states: tuple[Conv1dState, ...]
    ) -> tuple[Tensor, tuple[Conv1dState, ...]]:
        if len(states) != len(self.convolutions):
            raise ValueError("prosody convolution state count does not match model")
        x = pcm
        next_states: list[Conv1dState] = []
        for convolution, mixer, norm, state in zip(
            self.convolutions, self.mixers, self.norms, states
        ):
            x, state = convolution.step(x, state)
            x = torch.nn.functional.silu(mixer(norm(x)))
            next_states.append(state)
        return self.head(x), tuple(next_states)

    def _decode(
        self, raw: Tensor, state: ProsodyState
    ) -> tuple[ProsodyOutput, ProsodyState]:
        # Keep F0 in a physical speech range even before training.
        log_f0 = torch.log(40.0 + 460.0 * torch.sigmoid(raw[:, 0:1]))
        voiced = torch.sigmoid(raw[:, 1:2])
        periodicity = torch.sigmoid(raw[:, 2:3])
        log_energy = raw[:, 3:4]
        normalized_frames: list[Tensor] = []
        previous = state.previous_raw
        statistics = state.statistics
        for index in range(raw.shape[-1]):
            statistic_values = torch.cat(
                (log_f0[:, :, index], log_energy[:, :, index]), dim=1
            )
            current = torch.cat(
                (
                    log_f0[:, :, index],
                    voiced[:, :, index],
                    periodicity[:, :, index],
                    log_energy[:, :, index],
                ),
                dim=1,
            )
            gate = (
                (voiced[:, 0, index] >= self.config.confidence_threshold)
                & (periodicity[:, 0, index] >= self.config.confidence_threshold)
            )
            statistics = self.running_stats.update(
                statistic_values, statistics, gate
            )
            normalized = self.running_stats.normalize(
                statistic_values, statistics
            )
            delta = current - previous
            frame = torch.stack(
                (
                    normalized[:, 0],
                    voiced[:, 0, index],
                    periodicity[:, 0, index],
                    normalized[:, 1],
                    delta[:, 0],
                    delta[:, 1],
                    delta[:, 2],
                    delta[:, 3],
                ),
                dim=1,
            )
            normalized_frames.append(frame.unsqueeze(-1))
            previous = current
        values = torch.cat(normalized_frames, dim=-1)
        output = ProsodyOutput(
            values,
            log_f0,
            torch.exp(log_f0),
            voiced,
            periodicity,
            log_energy,
        )
        return output, ProsodyState(state.convolutions, statistics, previous)

    def forward(self, pcm: Tensor) -> ProsodyOutput:
        self._validate_pcm(pcm)
        raw = self._network(pcm)
        state = self.initial_state(
            pcm.shape[0], device=pcm.device, dtype=pcm.dtype
        )
        output, _ = self._decode(raw, state)
        return output

    def step(
        self, pcm: Tensor, state: ProsodyState
    ) -> tuple[ProsodyOutput, ProsodyState]:
        self._validate_pcm(pcm)
        if pcm.shape[-1] % self.chunk_samples:
            raise ValueError(
                f"streaming input must contain whole {self.chunk_samples}-sample frames"
            )
        raw, convolution_states = self._network_step(pcm, state.convolutions)
        state = ProsodyState(
            convolution_states, state.statistics, state.previous_raw
        )
        return self._decode(raw, state)

    @staticmethod
    def _validate_pcm(pcm: Tensor) -> None:
        if pcm.ndim != 3 or pcm.shape[1] != 1:
            raise ValueError("PCM must be shaped [batch, 1, samples]")


ProsodyEncoder = CausalProsodyEstimator
