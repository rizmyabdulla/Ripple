"""Causal 24 kHz source analysis encoder producing RIF semantic features."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
from torch import Tensor, nn

from ripple.streaming.cached_conv import CachedCausalConv1d, Conv1dState


class ChannelRMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.square().mean(dim=1, keepdim=True) + self.eps)
        return x * scale * self.weight.view(1, -1, 1)


@dataclass(frozen=True)
class AnalysisEncoderConfig:
    sample_rate: int = 24_000
    frame_rate: int = 50
    channels: int = 96
    latent_channels: int = 128
    semantic_classes: int = 256
    semantic_dim: int = 128
    strides: tuple[int, ...] = (3, 4, 5, 8)
    initial_kernel: int = 7
    stage_kernel: int = 7

    def __post_init__(self) -> None:
        if prod(self.strides) != self.sample_rate // self.frame_rate:
            raise ValueError("encoder strides must reduce audio to frame_rate")


@dataclass(frozen=True)
class AnalysisEncoderState:
    initial: Conv1dState
    stages: tuple[Conv1dState, ...]


@dataclass(frozen=True)
class AnalysisOutput:
    semantic_logits: Tensor
    semantic_soft: Tensor
    semantic_embed: Tensor
    latent: Tensor


class _DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel: int):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, out_channels, 1)
        self.depthwise = CachedCausalConv1d(
            out_channels,
            out_channels,
            kernel,
            stride=stride,
            groups=out_channels,
        )
        self.norm = ChannelRMSNorm(out_channels)
        self.gate = nn.Conv1d(out_channels, out_channels * 2, 1)

    def _finish(self, x: Tensor) -> Tensor:
        value, gate = self.gate(self.norm(x)).chunk(2, dim=1)
        return x + value * torch.sigmoid(gate)

    def forward(self, x: Tensor) -> Tensor:
        return self._finish(self.depthwise(self.pre(x)))

    def step(self, x: Tensor, state: Conv1dState) -> tuple[Tensor, Conv1dState]:
        x, state = self.depthwise.step(self.pre(x), state)
        return self._finish(x), state


class SourceAnalysisEncoder(nn.Module):
    """Depthwise causal encoder with exact explicit-state step execution."""

    def __init__(self, config: AnalysisEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or AnalysisEncoderConfig()
        cfg = self.config
        self.initial = CachedCausalConv1d(
            1, cfg.channels, cfg.initial_kernel, stride=1
        )
        stages: list[nn.Module] = []
        in_channels = cfg.channels
        for stride in cfg.strides:
            stages.append(
                _DownsampleStage(
                    in_channels, cfg.channels, stride, cfg.stage_kernel
                )
            )
            in_channels = cfg.channels
        self.stages = nn.ModuleList(stages)
        self.to_latent = nn.Conv1d(cfg.channels, cfg.latent_channels, 1)
        self.semantic_head = nn.Conv1d(
            cfg.latent_channels, cfg.semantic_classes, 1
        )
        self.semantic_embedding = nn.Parameter(
            torch.empty(cfg.semantic_classes, cfg.semantic_dim)
        )
        nn.init.normal_(self.semantic_embedding, std=cfg.semantic_dim**-0.5)

    @property
    def chunk_samples(self) -> int:
        return self.config.sample_rate // self.config.frame_rate

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> AnalysisEncoderState:
        initial = self.initial.initial_state(
            batch_size, device=device, dtype=dtype
        )
        stage_states = tuple(
            stage.depthwise.initial_state(batch_size, device=device, dtype=dtype)
            for stage in self.stages
        )
        return AnalysisEncoderState(initial, stage_states)

    def _output(self, x: Tensor) -> AnalysisOutput:
        latent = self.to_latent(x)
        logits = self.semantic_head(latent)
        probabilities = torch.softmax(logits, dim=1)
        embedding = torch.einsum(
            "bkt,kd->bdt", probabilities, self.semantic_embedding
        )
        return AnalysisOutput(logits, probabilities, embedding, latent)

    def forward(self, pcm: Tensor) -> AnalysisOutput:
        self._validate_pcm(pcm)
        x = torch.nn.functional.silu(self.initial(pcm))
        for stage in self.stages:
            x = stage(x)
        return self._output(x)

    def step(
        self, pcm: Tensor, state: AnalysisEncoderState
    ) -> tuple[AnalysisOutput, AnalysisEncoderState]:
        self._validate_pcm(pcm)
        if pcm.shape[-1] % self.chunk_samples:
            raise ValueError(
                f"streaming input must contain whole {self.chunk_samples}-sample frames"
            )
        if len(state.stages) != len(self.stages):
            raise ValueError("analysis state stage count does not match model")
        x, initial_state = self.initial.step(pcm, state.initial)
        x = torch.nn.functional.silu(x)
        next_stages: list[Conv1dState] = []
        for stage, stage_state in zip(self.stages, state.stages):
            x, next_state = stage.step(x, stage_state)
            next_stages.append(next_state)
        return self._output(x), AnalysisEncoderState(
            initial_state, tuple(next_stages)
        )

    @staticmethod
    def _validate_pcm(pcm: Tensor) -> None:
        if pcm.ndim != 3 or pcm.shape[1] != 1:
            raise ValueError("PCM must be shaped [batch, 1, samples]")


AnalysisEncoder = SourceAnalysisEncoder
