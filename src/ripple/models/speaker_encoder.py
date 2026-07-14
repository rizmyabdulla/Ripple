"""Lightweight target-speaker enrollment encoder and profile outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ripple.models.analysis_encoder import ChannelRMSNorm


@dataclass(frozen=True)
class SpeakerEncoderConfig:
    sample_rate: int = 24_000
    channels: int = 96
    global_dim: int = 192
    token_count: int = 4
    token_dim: int = 64
    schema_version: int = 1


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_global: Tensor
    speaker_tokens: Tensor
    pitch_mean: Tensor
    pitch_std: Tensor
    sample_rate: int = 24_000
    schema_version: int = 1

    def validate(self) -> None:
        if self.speaker_global.ndim != 2:
            raise ValueError("speaker_global must be [batch, features]")
        if self.speaker_tokens.ndim != 3:
            raise ValueError("speaker_tokens must be [batch, tokens, features]")
        if self.speaker_global.shape[0] != self.speaker_tokens.shape[0]:
            raise ValueError("speaker profile batch dimensions do not match")
        if self.pitch_mean.shape != self.pitch_std.shape:
            raise ValueError("pitch profile shapes do not match")

    def detach(self) -> SpeakerProfile:
        return SpeakerProfile(
            self.speaker_global.detach(),
            self.speaker_tokens.detach(),
            self.pitch_mean.detach(),
            self.pitch_std.detach(),
            self.sample_rate,
            self.schema_version,
        )


SpeakerProfileOutput = SpeakerProfile


class SpeakerEncoder(nn.Module):
    """Offline enrollment graph with attentive statistics and token pooling."""

    def __init__(self, config: SpeakerEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpeakerEncoderConfig()
        cfg = self.config
        self.frontend = nn.Sequential(
            nn.Conv1d(1, cfg.channels // 2, 7, stride=3, padding=3),
            nn.SiLU(),
            nn.Conv1d(
                cfg.channels // 2,
                cfg.channels,
                7,
                stride=4,
                padding=3,
                groups=cfg.channels // 2,
            ),
            nn.Conv1d(cfg.channels, cfg.channels, 1),
            nn.SiLU(),
            nn.Conv1d(
                cfg.channels,
                cfg.channels,
                5,
                stride=5,
                padding=2,
                groups=cfg.channels,
            ),
            ChannelRMSNorm(cfg.channels),
            nn.SiLU(),
        )
        self.attention = nn.Conv1d(cfg.channels, 1, 1)
        self.global_projection = nn.Linear(cfg.channels * 2, cfg.global_dim)
        self.token_queries = nn.Parameter(
            torch.randn(cfg.token_count, cfg.channels) * cfg.channels**-0.5
        )
        self.token_projection = nn.Linear(cfg.channels, cfg.token_dim)

    def forward(self, reference_pcm: Tensor) -> SpeakerProfile:
        if reference_pcm.ndim != 3 or reference_pcm.shape[1] != 1:
            raise ValueError("reference PCM must be [batch, 1, samples]")
        if reference_pcm.shape[-1] < 60:
            raise ValueError("reference PCM is too short for enrollment")
        features = self.frontend(reference_pcm)
        weights = torch.softmax(self.attention(features), dim=-1)
        mean = (features * weights).sum(dim=-1)
        variance = (
            (features - mean.unsqueeze(-1)).square() * weights
        ).sum(dim=-1)
        statistics = torch.cat((mean, torch.sqrt(variance + 1e-5)), dim=1)
        speaker_global = torch.nn.functional.normalize(
            self.global_projection(statistics), dim=1
        )

        logits = torch.einsum(
            "kc,bct->bkt", self.token_queries, features
        ) * features.shape[1] ** -0.5
        token_weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bkt,bct->bkc", token_weights, features)
        speaker_tokens = self.token_projection(pooled)

        # Robust defaults. A trained enrollment model may replace these with
        # pitch-head estimates without changing the profile schema.
        pitch_mean = torch.full(
            (reference_pcm.shape[0], 1),
            5.3,
            device=reference_pcm.device,
            dtype=reference_pcm.dtype,
        )
        pitch_std = torch.full_like(pitch_mean, 0.4)
        profile = SpeakerProfile(
            speaker_global,
            speaker_tokens,
            pitch_mean,
            pitch_std,
            self.config.sample_rate,
            self.config.schema_version,
        )
        profile.validate()
        return profile


TargetSpeakerEncoder = SpeakerEncoder
