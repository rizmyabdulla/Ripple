"""Deterministic packet-loss concealment policy for fixed audio frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor


class PacketStatus(IntEnum):
    OK = 0
    MISSING = 1
    LATE = 2
    CORRUPT = 3


@dataclass(frozen=True)
class PacketLossConfig:
    repeat_attenuation: float = 0.85
    comfort_noise_level: float = 0.003
    recovery_crossfade: float = 0.5
    single_loss_limit: int = 1


@dataclass(frozen=True)
class PacketLossState:
    consecutive_missing: int = 0
    total_missing: int = 0
    total_late: int = 0
    recovering: bool = False
    noise_index: int = 0


@dataclass(frozen=True)
class ConcealmentResult:
    pcm: Tensor | None
    state: PacketLossState
    freeze_statistics: bool
    dropped: bool


class PacketLossPolicy:
    def __init__(self, config: PacketLossConfig | None = None) -> None:
        self.config = config or PacketLossConfig()

    def apply(
        self,
        pcm: Tensor | None,
        status: PacketStatus,
        state: PacketLossState,
        previous_pcm: Tensor,
    ) -> ConcealmentResult:
        if status == PacketStatus.LATE:
            return ConcealmentResult(
                None,
                PacketLossState(
                    state.consecutive_missing,
                    state.total_missing,
                    state.total_late + 1,
                    state.recovering,
                    state.noise_index,
                ),
                True,
                True,
            )
        if status in (PacketStatus.MISSING, PacketStatus.CORRUPT):
            missing = state.consecutive_missing + 1
            if missing <= self.config.single_loss_limit:
                concealed = previous_pcm * self.config.repeat_attenuation**missing
            else:
                concealed = self._comfort_noise(previous_pcm, state.noise_index)
            next_state = PacketLossState(
                missing,
                state.total_missing + 1,
                state.total_late,
                True,
                state.noise_index + previous_pcm.shape[-1],
            )
            return ConcealmentResult(concealed, next_state, True, False)
        if pcm is None:
            raise ValueError("an OK packet must include PCM")
        return ConcealmentResult(
            pcm,
            PacketLossState(
                0,
                state.total_missing,
                state.total_late,
                state.recovering,
                state.noise_index,
            ),
            state.recovering,
            False,
        )

    def finish_recovery(self, state: PacketLossState) -> PacketLossState:
        return PacketLossState(
            state.consecutive_missing,
            state.total_missing,
            state.total_late,
            False,
            state.noise_index,
        )

    def _comfort_noise(self, like: Tensor, start: int) -> Tensor:
        index = torch.arange(
            start,
            start + like.shape[-1],
            device=like.device,
            dtype=like.dtype,
        )
        seed = torch.sin(index * 17.123 + 1.337) * 13_731.11
        noise = (seed - torch.floor(seed)) * 2.0 - 1.0
        return noise.view(1, 1, -1).expand_as(like) * self.config.comfort_noise_level
