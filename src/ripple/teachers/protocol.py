"""Training-only speech teacher interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor


@dataclass(frozen=True)
class TeacherFeatures:
    values: Tensor
    frame_rate_hz: float
    teacher: str
    layer: int | None = None


@runtime_checkable
class SpeechTeacher(Protocol):
    name: str
    sample_rate: int

    @property
    def loaded(self) -> bool: ...

    def extract(self, waveform: Tensor, sample_rate: int) -> TeacherFeatures: ...
