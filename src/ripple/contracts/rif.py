"""Ripple Intermediate Features contract (RIF-1)."""

from __future__ import annotations

import math
from typing import Self

from pydantic import Field, field_validator, model_validator

from ripple.contracts.manifest import ContractModel

RIF_VERSION = "RIF-1"
RIF_SAMPLE_RATE = 24_000
RIF_FRAME_RATE = 50
RIF_FRAME_SAMPLES = RIF_SAMPLE_RATE // RIF_FRAME_RATE
RIF_SEMANTIC_EMBED_DIM = 128
RIF_PROSODY_FIELDS = (
    "normalized_log_f0",
    "voiced_probability",
    "periodicity",
    "normalized_log_energy",
    "delta_log_f0",
    "delta_voiced_probability",
    "delta_periodicity",
    "delta_log_energy",
)


class RifSpec(ContractModel):
    version: str = Field(default=RIF_VERSION, pattern=r"^RIF-1$")
    sample_rate: int = Field(default=RIF_SAMPLE_RATE, ge=1)
    frame_rate: int = Field(default=RIF_FRAME_RATE, ge=1)
    semantic_classes: int = Field(default=256, ge=256, le=512)
    semantic_embed_dim: int = Field(default=RIF_SEMANTIC_EMBED_DIM, ge=1)
    prosody_fields: tuple[str, ...] = RIF_PROSODY_FIELDS
    lookahead_frames: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_cadence(self) -> Self:
        if self.sample_rate % self.frame_rate:
            raise ValueError("sample_rate must be divisible by frame_rate")
        if self.semantic_embed_dim != RIF_SEMANTIC_EMBED_DIM:
            raise ValueError("RIF-1 semantic embedding dimension must be 128")
        if self.prosody_fields != RIF_PROSODY_FIELDS:
            raise ValueError("RIF-1 prosody field order is fixed")
        return self

    @property
    def frame_samples(self) -> int:
        return self.sample_rate // self.frame_rate


class RifFrame(ContractModel):
    """One immutable 50 Hz RIF-1 frame."""

    frame_index: int = Field(ge=0)
    semantic_soft: tuple[float, ...]
    semantic_embed: tuple[float, ...]
    prosody: tuple[float, ...]

    @field_validator("semantic_soft", "semantic_embed", "prosody")
    @classmethod
    def validate_finite(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("RIF values must be finite")
        return value

    def validate_against(self, spec: RifSpec) -> None:
        if len(self.semantic_soft) != spec.semantic_classes:
            raise ValueError("semantic_soft dimension does not match RIF spec")
        if len(self.semantic_embed) != spec.semantic_embed_dim:
            raise ValueError("semantic_embed dimension does not match RIF spec")
        if len(self.prosody) != len(spec.prosody_fields):
            raise ValueError("prosody dimension does not match RIF spec")
        if any(value < 0.0 or value > 1.0 for value in self.semantic_soft):
            raise ValueError("semantic_soft values must be probabilities")
        if not math.isclose(sum(self.semantic_soft), 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("semantic_soft probabilities must sum to one")
        if not 0.0 <= self.prosody[1] <= 1.0:
            raise ValueError("voiced_probability must be in [0, 1]")
        if not 0.0 <= self.prosody[2] <= 1.0:
            raise ValueError("periodicity must be in [0, 1]")


class RifSequence(ContractModel):
    spec: RifSpec
    frames: tuple[RifFrame, ...]

    @model_validator(mode="after")
    def validate_frames(self) -> Self:
        for expected, frame in enumerate(self.frames):
            if frame.frame_index != expected:
                raise ValueError("RIF frame indexes must be contiguous and zero-based")
            frame.validate_against(self.spec)
        return self

