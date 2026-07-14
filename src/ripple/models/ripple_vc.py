"""Integrated zero-lookahead Ripple voice-conversion model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

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
from ripple.models.ripple_mixer import (
    RippleMixer,
    RippleMixerConfig,
    RippleMixerState,
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


@dataclass(frozen=True)
class RippleVCConfig:
    analysis: AnalysisEncoderConfig = field(
        default_factory=AnalysisEncoderConfig
    )
    mixer: RippleMixerConfig = field(default_factory=RippleMixerConfig)
    prosody: ProsodyEncoderConfig = field(
        default_factory=ProsodyEncoderConfig
    )
    speaker: SpeakerEncoderConfig = field(
        default_factory=SpeakerEncoderConfig
    )
    source: SourceFilterConfig = field(default_factory=SourceFilterConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    detach_semantic_for_decoder: bool = True

    def __post_init__(self) -> None:
        sample_rates = {
            self.analysis.sample_rate,
            self.prosody.sample_rate,
            self.speaker.sample_rate,
            self.source.sample_rate,
            self.decoder.sample_rate,
        }
        if len(sample_rates) != 1:
            raise ValueError("all Ripple components must use one sample rate")
        if self.analysis.semantic_dim != self.mixer.channels:
            raise ValueError("analysis semantic dimension must match mixer channels")
        if self.mixer.channels != self.decoder.semantic_dim:
            raise ValueError("mixer channels must match decoder semantic dimension")
        if self.decoder.prosody_dim != 8:
            raise ValueError("RIF-1 requires eight prosody channels")


@dataclass(frozen=True)
class RippleVCState:
    analysis: AnalysisEncoderState
    mixer: RippleMixerState
    prosody: ProsodyState
    source: SourceFilterState
    decoder: DecoderState


@dataclass(frozen=True)
class RippleVCOutput:
    waveform: Tensor
    semantic_soft: Tensor
    semantic_embed: Tensor
    prosody: Tensor
    f0_hz: Tensor
    voiced_probability: Tensor


class RippleVC(nn.Module):
    def __init__(self, config: RippleVCConfig | None = None) -> None:
        super().__init__()
        self.config = config or RippleVCConfig()
        self.analysis_encoder = SourceAnalysisEncoder(self.config.analysis)
        self.mixer = RippleMixer(self.config.mixer)
        self.prosody_encoder = CausalProsodyEstimator(self.config.prosody)
        self.speaker_encoder = SpeakerEncoder(self.config.speaker)
        self.source_filter = HarmonicNoiseSource(self.config.source)
        self.decoder = CausalWaveformDecoder(self.config.decoder)

    @property
    def sample_rate(self) -> int:
        return self.config.analysis.sample_rate

    @property
    def chunk_samples(self) -> int:
        return self.config.analysis.sample_rate // self.config.analysis.frame_rate

    def enroll(self, target_reference: Tensor) -> SpeakerProfile:
        return self.speaker_encoder(target_reference)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RippleVCState:
        reference = next(self.parameters())
        resolved_device = device if device is not None else reference.device
        resolved_dtype = dtype if dtype is not None else reference.dtype
        return RippleVCState(
            self.analysis_encoder.initial_state(
                batch_size, device=resolved_device, dtype=resolved_dtype
            ),
            self.mixer.initial_state(
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
        state: RippleVCState,
    ) -> tuple[RippleVCOutput, RippleVCState]:
        if pcm_frame.ndim != 3 or pcm_frame.shape[1:] != (
            1,
            self.chunk_samples,
        ):
            raise ValueError(
                f"step expects [batch, 1, {self.chunk_samples}] PCM"
            )
        analysis, analysis_state = self.analysis_encoder.step(
            pcm_frame, state.analysis
        )
        semantic, mixer_state = self.mixer.step(
            analysis.semantic_embed, state.mixer
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
        decoder_semantic = (
            semantic.detach()
            if self.config.detach_semantic_for_decoder
            else semantic
        )
        waveform, decoder_state = self.decoder.step(
            decoder_semantic,
            prosody.values,
            profile,
            excitation,
            state.decoder,
        )
        output = RippleVCOutput(
            waveform,
            analysis.semantic_soft,
            semantic,
            prosody.values,
            prosody.f0_hz,
            prosody.voiced_probability,
        )
        return output, RippleVCState(
            analysis_state,
            mixer_state,
            prosody_state,
            source_state,
            decoder_state,
        )

    def forward(
        self, source_pcm: Tensor, profile: SpeakerProfile
    ) -> RippleVCOutput:
        if source_pcm.ndim != 3 or source_pcm.shape[1] != 1:
            raise ValueError("source PCM must be [batch, 1, samples]")
        if source_pcm.shape[-1] % self.chunk_samples:
            raise ValueError("source PCM must contain whole 20 ms frames")
        state = self.initial_state(
            source_pcm.shape[0],
            device=source_pcm.device,
            dtype=source_pcm.dtype,
        )
        outputs: list[RippleVCOutput] = []
        for frame in source_pcm.split(self.chunk_samples, dim=-1):
            output, state = self.step(frame, profile, state)
            outputs.append(output)
        return self._concatenate(outputs)

    @staticmethod
    def _concatenate(outputs: list[RippleVCOutput]) -> RippleVCOutput:
        if not outputs:
            raise ValueError("at least one source frame is required")
        return RippleVCOutput(
            waveform=torch.cat([item.waveform for item in outputs], dim=-1),
            semantic_soft=torch.cat(
                [item.semantic_soft for item in outputs], dim=-1
            ),
            semantic_embed=torch.cat(
                [item.semantic_embed for item in outputs], dim=-1
            ),
            prosody=torch.cat([item.prosody for item in outputs], dim=-1),
            f0_hz=torch.cat([item.f0_hz for item in outputs], dim=-1),
            voiced_probability=torch.cat(
                [item.voiced_probability for item in outputs], dim=-1
            ),
        )


RippleModel = RippleVC
