"""Stateful fixed-cadence Python session built on RippleVC's explicit state."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from ripple.models.prosody_encoder import ProsodyState
from ripple.models.ripple_vc import RippleVC, RippleVCOutput, RippleVCState
from ripple.models.source_filter import SourceFilterState
from ripple.models.speaker_encoder import SpeakerProfile
from ripple.streaming.packet_loss import (
    PacketLossPolicy,
    PacketLossState,
    PacketStatus,
)


@dataclass(frozen=True)
class SessionConfig:
    soft_reset_decay: float = 0.5
    validate_finite: bool = True


@dataclass(frozen=True)
class SessionState:
    model: RippleVCState
    packet_loss: PacketLossState
    previous_input: Tensor
    previous_output: Tensor
    frame_index: int = 0


@dataclass(frozen=True)
class SessionDiagnostics:
    packet_status: PacketStatus
    produced: bool
    consecutive_missing: int
    total_missing: int
    total_late: int
    recovering: bool
    state_healthy: bool


@dataclass(frozen=True)
class SessionOutput:
    waveform: Tensor | None
    model_output: RippleVCOutput | None
    state: SessionState
    diagnostics: SessionDiagnostics


class RippleSession:
    """Convenience session; the contained model state remains fully explicit."""

    def __init__(
        self,
        model: RippleVC,
        profile: SpeakerProfile,
        *,
        batch_size: int | None = None,
        config: SessionConfig | None = None,
        packet_policy: PacketLossPolicy | None = None,
    ) -> None:
        self.model = model
        self.profile = profile
        self.config = config or SessionConfig()
        self.packet_policy = packet_policy or PacketLossPolicy()
        inferred_batch = profile.speaker_global.shape[0]
        if batch_size is not None and batch_size != inferred_batch:
            raise ValueError("profile and requested batch sizes do not match")
        reference = profile.speaker_global
        model_state = model.initial_state(
            inferred_batch, device=reference.device, dtype=reference.dtype
        )
        zeros = torch.zeros(
            inferred_batch,
            1,
            model.chunk_samples,
            device=reference.device,
            dtype=reference.dtype,
        )
        self.state = SessionState(
            model_state,
            PacketLossState(),
            zeros,
            zeros.clone(),
        )

    def push(
        self,
        pcm: Tensor | None,
        status: PacketStatus = PacketStatus.OK,
    ) -> SessionOutput:
        concealment = self.packet_policy.apply(
            pcm,
            status,
            self.state.packet_loss,
            self.state.previous_input,
        )
        if concealment.dropped:
            self.state = replace(
                self.state, packet_loss=concealment.state
            )
            return self._result(None, None, status, produced=False)
        assert concealment.pcm is not None
        model_output, model_state = self.model.step(
            concealment.pcm, self.profile, self.state.model
        )
        if concealment.freeze_statistics:
            model_state = replace(
                model_state,
                prosody=replace(
                    model_state.prosody,
                    statistics=self.state.model.prosody.statistics,
                ),
            )

        waveform = model_output.waveform
        was_recovering = (
            status == PacketStatus.OK and concealment.state.recovering
        )
        packet_state = concealment.state
        if was_recovering:
            amount = self.packet_policy.config.recovery_crossfade
            waveform = torch.lerp(self.state.previous_output, waveform, amount)
            model_output = replace(model_output, waveform=waveform)
            packet_state = self.packet_policy.finish_recovery(packet_state)
        healthy = not self.config.validate_finite or bool(
            torch.isfinite(waveform).all().item()
        )
        self.state = SessionState(
            model_state,
            packet_state,
            concealment.pcm,
            waveform,
            self.state.frame_index + 1,
        )
        return self._result(
            waveform, model_output, status, produced=True, healthy=healthy
        )

    def soft_reset(self) -> SessionState:
        """Clear convolution history while retaining a decayed pitch prior."""

        current = self.state.model
        initial = self.model.initial_state(
            self.state.previous_input.shape[0],
            device=self.state.previous_input.device,
            dtype=self.state.previous_input.dtype,
        )
        statistics = self.model.prosody_encoder.running_stats.decay(
            current.prosody.statistics, self.config.soft_reset_decay
        )
        previous = torch.lerp(
            current.prosody.previous_raw,
            initial.prosody.previous_raw,
            self.config.soft_reset_decay,
        )
        prosody = ProsodyState(
            initial.prosody.convolutions,
            statistics,
            previous,
        )
        source = SourceFilterState(
            current.source.phase * (1.0 - self.config.soft_reset_decay),
            current.source.sample_index,
        )
        model_state = RippleVCState(
            initial.analysis,
            initial.mixer,
            prosody,
            source,
            initial.decoder,
        )
        self.state = replace(
            self.state,
            model=model_state,
            packet_loss=PacketLossState(),
            previous_input=torch.zeros_like(self.state.previous_input),
            previous_output=torch.zeros_like(self.state.previous_output),
        )
        return self.state

    def hard_reset(self) -> SessionState:
        model_state = self.model.initial_state(
            self.state.previous_input.shape[0],
            device=self.state.previous_input.device,
            dtype=self.state.previous_input.dtype,
        )
        self.state = SessionState(
            model_state,
            PacketLossState(),
            torch.zeros_like(self.state.previous_input),
            torch.zeros_like(self.state.previous_output),
            0,
        )
        return self.state

    def set_profile(
        self, profile: SpeakerProfile, *, hard_reset: bool = True
    ) -> None:
        if profile.speaker_global.shape[0] != self.profile.speaker_global.shape[0]:
            raise ValueError("new profile batch size differs from session")
        self.profile = profile
        if hard_reset:
            self.hard_reset()

    def _result(
        self,
        waveform: Tensor | None,
        model_output: RippleVCOutput | None,
        status: PacketStatus,
        *,
        produced: bool,
        healthy: bool = True,
    ) -> SessionOutput:
        packet = self.state.packet_loss
        return SessionOutput(
            waveform,
            model_output,
            self.state,
            SessionDiagnostics(
                status,
                produced,
                packet.consecutive_missing,
                packet.total_missing,
                packet.total_late,
                packet.recovering,
                healthy,
            ),
        )
