"""Bounded 50 Hz RippleMixer in pure-convolution and local-attention modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from ripple.models.analysis_encoder import ChannelRMSNorm
from ripple.streaming.cached_conv import CachedCausalConv1d, Conv1dState
from ripple.streaming.local_attention import (
    CausalLocalAttention,
    LocalAttentionState,
)


MixerMode = Literal["pure_conv", "local_attn"]


@dataclass(frozen=True)
class RippleMixerConfig:
    channels: int = 128
    blocks: int = 4
    kernel_size: int = 9
    expansion: int = 2
    mode: MixerMode = "local_attn"
    attention_heads: int = 4
    attention_window: int = 32
    attention_every: int = 2


@dataclass(frozen=True)
class MixerBlockState:
    convolution: Conv1dState
    attention: LocalAttentionState | None = None


@dataclass(frozen=True)
class RippleMixerState:
    blocks: tuple[MixerBlockState, ...]


class _MixerBlock(nn.Module):
    def __init__(self, config: RippleMixerConfig, use_attention: bool) -> None:
        super().__init__()
        channels = config.channels
        self.norm = ChannelRMSNorm(channels)
        self.temporal = CachedCausalConv1d(
            channels,
            channels,
            config.kernel_size,
            groups=channels,
        )
        hidden = channels * config.expansion
        self.expand = nn.Conv1d(channels, hidden * 2, 1)
        self.project = nn.Conv1d(hidden, channels, 1)
        self.attention_norm = ChannelRMSNorm(channels)
        self.attention = (
            CausalLocalAttention(
                channels,
                num_heads=config.attention_heads,
                window=config.attention_window,
            )
            if use_attention
            else None
        )

    def _conv_finish(self, residual: Tensor, temporal: Tensor) -> Tensor:
        value, gate = self.expand(temporal).chunk(2, dim=1)
        return residual + self.project(torch.nn.functional.silu(value) * torch.sigmoid(gate))

    def forward(self, x: Tensor) -> Tensor:
        x = self._conv_finish(x, self.temporal(self.norm(x)))
        if self.attention is not None:
            x = x + self.attention(self.attention_norm(x))
        return x

    def step(
        self, x: Tensor, state: MixerBlockState
    ) -> tuple[Tensor, MixerBlockState]:
        temporal, convolution = self.temporal.step(
            self.norm(x), state.convolution
        )
        x = self._conv_finish(x, temporal)
        attention_state = state.attention
        if self.attention is not None:
            if attention_state is None:
                raise ValueError("missing local-attention state")
            attended, attention_state = self.attention.step(
                self.attention_norm(x), attention_state
            )
            x = x + attended
        elif attention_state is not None:
            raise ValueError("unexpected local-attention state")
        return x, MixerBlockState(convolution, attention_state)


class RippleMixer(nn.Module):
    def __init__(self, config: RippleMixerConfig | None = None) -> None:
        super().__init__()
        self.config = config or RippleMixerConfig()
        if self.config.mode not in ("pure_conv", "local_attn"):
            raise ValueError(f"unknown mixer mode: {self.config.mode}")
        self.blocks = nn.ModuleList(
            [
                _MixerBlock(
                    self.config,
                    self.config.mode == "local_attn"
                    and (index + 1) % self.config.attention_every == 0,
                )
                for index in range(self.config.blocks)
            ]
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RippleMixerState:
        states: list[MixerBlockState] = []
        for block in self.blocks:
            convolution = block.temporal.initial_state(
                batch_size, device=device, dtype=dtype
            )
            attention = (
                block.attention.initial_state(
                    batch_size, device=device, dtype=dtype
                )
                if block.attention is not None
                else None
            )
            states.append(MixerBlockState(convolution, attention))
        return RippleMixerState(tuple(states))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def step(
        self, x: Tensor, state: RippleMixerState
    ) -> tuple[Tensor, RippleMixerState]:
        if len(state.blocks) != len(self.blocks):
            raise ValueError("mixer state block count does not match model")
        states: list[MixerBlockState] = []
        for block, block_state in zip(self.blocks, state.blocks):
            x, block_state = block.step(x, block_state)
            states.append(block_state)
        return x, RippleMixerState(tuple(states))
