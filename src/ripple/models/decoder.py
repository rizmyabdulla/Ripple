"""Causal resize-convolution waveform decoder."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
from torch import Tensor, nn

from ripple.models.analysis_encoder import ChannelRMSNorm
from ripple.models.speaker_encoder import SpeakerProfile
from ripple.streaming.cached_conv import CachedCausalConv1d, Conv1dState


@dataclass(frozen=True)
class DecoderConfig:
    sample_rate: int = 24_000
    frame_rate: int = 50
    semantic_dim: int = 128
    prosody_dim: int = 8
    speaker_dim: int = 192
    token_dim: int = 64
    hidden_channels: int = 96
    upsample_scales: tuple[int, ...] = (3, 4, 5, 8)
    stage_channels: tuple[int, ...] = (96, 64, 48, 32)
    kernel_size: int = 7
    dc_coefficient: float = 0.995
    output_limit: float = 0.98

    def __post_init__(self) -> None:
        if prod(self.upsample_scales) != self.sample_rate // self.frame_rate:
            raise ValueError("decoder upsample scales must reach sample rate")
        if len(self.upsample_scales) != len(self.stage_channels):
            raise ValueError("one channel count is required per upsample stage")


@dataclass(frozen=True)
class PostFilterState:
    previous_input: Tensor
    previous_output: Tensor


@dataclass(frozen=True)
class DecoderState:
    stages: tuple[Conv1dState, ...]
    output: Conv1dState
    post_filter: PostFilterState


class _UpsampleStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int,
        speaker_dim: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.input_projection = nn.Conv1d(in_channels, out_channels, 1)
        self.depthwise = CachedCausalConv1d(
            out_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
        )
        self.norm = ChannelRMSNorm(out_channels)
        self.gate = nn.Conv1d(out_channels, out_channels * 2, 1)
        self.film = nn.Linear(speaker_dim, out_channels * 2)

    def _prepare(self, x: Tensor) -> Tensor:
        x = self.input_projection(x)
        return x.repeat_interleave(self.scale, dim=-1)

    def _finish(self, x: Tensor, filtered: Tensor, speaker: Tensor) -> Tensor:
        scale, shift = self.film(speaker).chunk(2, dim=-1)
        conditioned = self.norm(filtered) * (1.0 + 0.1 * scale.unsqueeze(-1))
        conditioned = conditioned + 0.1 * shift.unsqueeze(-1)
        value, gate = self.gate(conditioned).chunk(2, dim=1)
        return x + torch.nn.functional.silu(value) * torch.sigmoid(gate)

    def forward(self, x: Tensor, speaker: Tensor) -> Tensor:
        x = self._prepare(x)
        return self._finish(x, self.depthwise(x), speaker)

    def step(
        self, x: Tensor, speaker: Tensor, state: Conv1dState
    ) -> tuple[Tensor, Conv1dState]:
        x = self._prepare(x)
        filtered, state = self.depthwise.step(x, state)
        return self._finish(x, filtered, speaker), state


class CausalWaveformDecoder(nn.Module):
    def __init__(self, config: DecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or DecoderConfig()
        cfg = self.config
        self.input_projection = nn.Conv1d(
            cfg.semantic_dim + cfg.prosody_dim,
            cfg.hidden_channels,
            1,
        )
        self.token_projection = nn.Linear(cfg.token_dim, cfg.speaker_dim)
        stages: list[_UpsampleStage] = []
        in_channels = cfg.hidden_channels
        for scale, out_channels in zip(
            cfg.upsample_scales, cfg.stage_channels
        ):
            stages.append(
                _UpsampleStage(
                    in_channels,
                    out_channels,
                    scale,
                    cfg.speaker_dim,
                    cfg.kernel_size,
                )
            )
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.output = CachedCausalConv1d(
            in_channels, 1, cfg.kernel_size
        )
        self.excitation_gain = nn.Parameter(torch.tensor(0.1))

    @property
    def hop_samples(self) -> int:
        return self.config.sample_rate // self.config.frame_rate

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> DecoderState:
        reference = self.output.weight
        resolved_device = device if device is not None else reference.device
        resolved_dtype = dtype if dtype is not None else reference.dtype
        stage_states = tuple(
            stage.depthwise.initial_state(
                batch_size, device=resolved_device, dtype=resolved_dtype
            )
            for stage in self.stages
        )
        output = self.output.initial_state(
            batch_size, device=resolved_device, dtype=resolved_dtype
        )
        zeros = torch.zeros(
            batch_size, 1, device=resolved_device, dtype=resolved_dtype
        )
        return DecoderState(
            stage_states,
            output,
            PostFilterState(zeros, zeros.clone()),
        )

    def _speaker_condition(self, profile: SpeakerProfile) -> Tensor:
        profile.validate()
        token_summary = profile.speaker_tokens.mean(dim=1)
        return profile.speaker_global + self.token_projection(token_summary)

    def _post_process(
        self, x: Tensor, state: PostFilterState
    ) -> tuple[Tensor, PostFilterState]:
        previous_input = state.previous_input
        previous_output = state.previous_output
        frames: list[Tensor] = []
        for index in range(x.shape[-1]):
            current = x[:, :, index]
            filtered = (
                current
                - previous_input
                + self.config.dc_coefficient * previous_output
            )
            frames.append(filtered.unsqueeze(-1))
            previous_input = current
            previous_output = filtered
        y = torch.cat(frames, dim=-1)
        limit = self.config.output_limit
        y = limit * torch.tanh(y / limit)
        return y, PostFilterState(previous_input, previous_output)

    def forward(
        self,
        semantic: Tensor,
        prosody: Tensor,
        profile: SpeakerProfile,
        excitation: Tensor,
    ) -> Tensor:
        state = self.initial_state(
            semantic.shape[0], device=semantic.device, dtype=semantic.dtype
        )
        output, _ = self.step(
            semantic, prosody, profile, excitation, state
        )
        return output

    def step(
        self,
        semantic: Tensor,
        prosody: Tensor,
        profile: SpeakerProfile,
        excitation: Tensor,
        state: DecoderState,
    ) -> tuple[Tensor, DecoderState]:
        if semantic.ndim != 3 or semantic.shape[1] != self.config.semantic_dim:
            raise ValueError("semantic features have the wrong shape")
        if prosody.shape[:1] + prosody.shape[2:] != semantic.shape[:1] + semantic.shape[2:]:
            raise ValueError("semantic and prosody frame dimensions must match")
        if prosody.shape[1] != self.config.prosody_dim:
            raise ValueError("prosody features have the wrong channel count")
        expected_excitation = semantic.shape[-1] * self.hop_samples
        if excitation.shape != (
            semantic.shape[0],
            1,
            expected_excitation,
        ):
            raise ValueError("excitation has the wrong shape")
        if len(state.stages) != len(self.stages):
            raise ValueError("decoder state stage count does not match model")

        speaker = self._speaker_condition(profile)
        x = self.input_projection(torch.cat((semantic, prosody), dim=1))
        next_stages: list[Conv1dState] = []
        for stage, stage_state in zip(self.stages, state.stages):
            x, stage_state = stage.step(x, speaker, stage_state)
            next_stages.append(stage_state)
        x, output_state = self.output.step(x, state.output)
        x = x + self.excitation_gain * excitation
        x, post_state = self._post_process(x, state.post_filter)
        return x, DecoderState(tuple(next_stages), output_state, post_state)


RippleDecoder = CausalWaveformDecoder
