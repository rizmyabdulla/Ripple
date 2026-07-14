"""Corrected, explicit-state StreamVC-compatible baseline interface."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ripple.models.analysis_encoder import (
    AnalysisEncoderConfig,
    AnalysisEncoderState,
    SourceAnalysisEncoder,
)
from ripple.models.decoder import (
    CausalWaveformDecoder,
    DecoderConfig,
    DecoderState,
)
from ripple.models.prosody_encoder import (
    CausalProsodyEstimator,
    ProsodyEncoderConfig,
    ProsodyState,
)
from ripple.models.source_filter import (
    HarmonicNoiseSource,
    SourceFilterConfig,
    SourceFilterState,
)
from ripple.models.speaker_encoder import (
    SpeakerEncoder,
    SpeakerEncoderConfig,
    SpeakerProfile,
)


YIN_THRESHOLD = 0.15


@dataclass(frozen=True)
class StreamVCConfig:
    sample_rate: int = 16_000
    frame_rate: int = 50
    chunk_samples: int = 320
    content_classes: int = 100
    content_dim: int = 64
    speaker_dim: int = 64
    yin_threshold: float = YIN_THRESHOLD

    def __post_init__(self) -> None:
        if self.chunk_samples != self.sample_rate // self.frame_rate:
            raise ValueError("StreamVC chunk must be one 20 ms frame")
        if not 0.0 < self.yin_threshold < 1.0:
            raise ValueError("YIN threshold must be a probability-like value")


@dataclass(frozen=True)
class StreamVCState:
    content: AnalysisEncoderState
    prosody: ProsodyState
    source: SourceFilterState
    decoder: DecoderState


@dataclass(frozen=True)
class StreamVCOutput:
    waveform: Tensor
    content_logits: Tensor
    content_probabilities: Tensor
    prosody: Tensor


class StreamVCBaseline(nn.Module):
    """Functional baseline preserving logits and all recurrent state explicitly."""

    def __init__(self, config: StreamVCConfig | None = None) -> None:
        super().__init__()
        self.config = config or StreamVCConfig()
        cfg = self.config
        self.content_encoder = SourceAnalysisEncoder(
            AnalysisEncoderConfig(
                sample_rate=cfg.sample_rate,
                frame_rate=cfg.frame_rate,
                channels=64,
                latent_channels=64,
                semantic_classes=cfg.content_classes,
                semantic_dim=cfg.content_dim,
                strides=(2, 4, 5, 8),
            )
        )
        self.prosody_encoder = CausalProsodyEstimator(
            ProsodyEncoderConfig(
                sample_rate=cfg.sample_rate,
                frame_rate=cfg.frame_rate,
                channels=32,
                strides=(2, 4, 5, 8),
                confidence_threshold=cfg.yin_threshold,
            )
        )
        self.speaker_encoder = SpeakerEncoder(
            SpeakerEncoderConfig(
                sample_rate=cfg.sample_rate,
                channels=64,
                global_dim=cfg.speaker_dim,
                token_count=4,
                token_dim=64,
            )
        )
        self.source_filter = HarmonicNoiseSource(
            SourceFilterConfig(
                sample_rate=cfg.sample_rate,
                frame_rate=cfg.frame_rate,
                harmonics=8,
            )
        )
        self.decoder = CausalWaveformDecoder(
            DecoderConfig(
                sample_rate=cfg.sample_rate,
                frame_rate=cfg.frame_rate,
                semantic_dim=cfg.content_dim,
                prosody_dim=8,
                speaker_dim=cfg.speaker_dim,
                token_dim=64,
                hidden_channels=64,
                upsample_scales=(2, 4, 5, 8),
                stage_channels=(64, 48, 32, 24),
            )
        )

    def enroll(self, reference_pcm: Tensor) -> SpeakerProfile:
        return self.speaker_encoder(reference_pcm)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> StreamVCState:
        reference = next(self.parameters())
        resolved_device = device if device is not None else reference.device
        resolved_dtype = dtype if dtype is not None else reference.dtype
        return StreamVCState(
            self.content_encoder.initial_state(
                batch_size, device=resolved_device, dtype=resolved_dtype
            ),
            self.prosody_encoder.initial_state(
                batch_size, device=resolved_device, dtype=resolved_dtype
            ),
            self.source_filter.initial_state(
                batch_size, device=resolved_device, dtype=resolved_dtype
            ),
            self.decoder.initial_state(
                batch_size, device=resolved_device, dtype=resolved_dtype
            ),
        )

    def step(
        self,
        pcm_frame: Tensor,
        profile: SpeakerProfile,
        state: StreamVCState,
    ) -> tuple[StreamVCOutput, StreamVCState]:
        if pcm_frame.shape[1:] != (1, self.config.chunk_samples):
            raise ValueError(
                f"step expects [batch, 1, {self.config.chunk_samples}]"
            )
        content, content_state = self.content_encoder.step(
            pcm_frame, state.content
        )
        prosody, prosody_state = self.prosody_encoder.step(
            pcm_frame, state.prosody
        )
        excitation, source_state = self.source_filter.step(
            prosody.f0_hz,
            prosody.voiced_probability,
            prosody.periodicity,
            state.source,
        )
        waveform, decoder_state = self.decoder.step(
            content.semantic_embed.detach(),
            prosody.values,
            profile,
            excitation,
            state.decoder,
        )
        output = StreamVCOutput(
            waveform,
            content.semantic_logits,
            content.semantic_soft,
            prosody.values,
        )
        return output, StreamVCState(
            content_state, prosody_state, source_state, decoder_state
        )

    def forward(
        self, source_pcm: Tensor, profile: SpeakerProfile
    ) -> StreamVCOutput:
        if source_pcm.shape[-1] % self.config.chunk_samples:
            raise ValueError("source must contain whole StreamVC frames")
        state = self.initial_state(
            source_pcm.shape[0],
            device=source_pcm.device,
            dtype=source_pcm.dtype,
        )
        outputs: list[StreamVCOutput] = []
        for frame in source_pcm.split(self.config.chunk_samples, dim=-1):
            output, state = self.step(frame, profile, state)
            outputs.append(output)
        return StreamVCOutput(
            torch.cat([item.waveform for item in outputs], dim=-1),
            torch.cat([item.content_logits for item in outputs], dim=-1),
            torch.cat(
                [item.content_probabilities for item in outputs], dim=-1
            ),
            torch.cat([item.prosody for item in outputs], dim=-1),
        )

    @staticmethod
    def content_loss(content_logits: Tensor, targets: Tensor) -> Tensor:
        """Cross entropy consumes raw logits; probabilities are diagnostics only."""

        if targets.ndim == 2:
            targets = targets.reshape(-1)
        logits = content_logits.permute(0, 2, 1).reshape(
            -1, content_logits.shape[1]
        )
        return F.cross_entropy(logits, targets)


def yin_voiced(cmnd: Tensor, threshold: float = YIN_THRESHOLD) -> Tensor:
    """Return whether any cumulative-mean normalized difference crosses YIN."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("YIN threshold must lie in (0, 1)")
    return (cmnd < threshold).any(dim=-1)


StreamVC = StreamVCBaseline
