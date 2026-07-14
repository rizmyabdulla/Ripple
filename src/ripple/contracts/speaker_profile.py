"""Versioned, content-addressed speaker profile schema."""

from __future__ import annotations

import math
from typing import Self

from pydantic import Field, field_validator, model_validator

from ripple.contracts.checksums import is_sha256, sha256_json
from ripple.contracts.manifest import ContractModel, Provenance

SPEAKER_PROFILE_VERSION = "ripple-speaker-profile-1"
SPEAKER_GLOBAL_DIM = 192
SPEAKER_TOKEN_DIM = 64


class PitchProfile(ContractModel):
    median_log_f0: float
    log_f0_p05: float
    log_f0_p95: float

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        values = (self.log_f0_p05, self.median_log_f0, self.log_f0_p95)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pitch profile values must be finite")
        if not self.log_f0_p05 <= self.median_log_f0 <= self.log_f0_p95:
            raise ValueError("pitch quantiles must be ordered")
        return self


class ProfileNormalization(ContractModel):
    waveform_peak: float = Field(gt=0.0, le=1.0)
    waveform_mean: float
    waveform_std: float = Field(gt=0.0)

    @field_validator("waveform_peak", "waveform_mean", "waveform_std")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("normalization values must be finite")
        return value


class SpeakerProfile(ContractModel):
    schema_version: str = Field(
        default=SPEAKER_PROFILE_VERSION,
        pattern=r"^ripple-speaker-profile-1$",
    )
    speaker_id: str = Field(min_length=1)
    consent_reference: str = Field(min_length=1)
    sample_rate: int = Field(default=24_000, ge=1)
    speaker_global: tuple[float, ...]
    speaker_tokens: tuple[tuple[float, ...], ...]
    pitch_profile: PitchProfile
    normalization: ProfileNormalization
    provenance: Provenance
    checksum: str

    @field_validator("checksum")
    @classmethod
    def validate_checksum_shape(cls, value: str) -> str:
        if not is_sha256(value):
            raise ValueError("checksum must use lowercase sha256:<64 hex> form")
        return value

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        if len(self.speaker_global) != SPEAKER_GLOBAL_DIM:
            raise ValueError(f"speaker_global must contain {SPEAKER_GLOBAL_DIM} values")
        if not 4 <= len(self.speaker_tokens) <= 8:
            raise ValueError("speaker_tokens must contain 4 to 8 tokens")
        if any(len(token) != SPEAKER_TOKEN_DIM for token in self.speaker_tokens):
            raise ValueError(f"each speaker token must contain {SPEAKER_TOKEN_DIM} values")
        token_values = tuple(value for token in self.speaker_tokens for value in token)
        values = self.speaker_global + token_values
        if not all(math.isfinite(value) for value in values):
            raise ValueError("speaker embeddings must be finite")
        if self.checksum != self.computed_checksum():
            raise ValueError("speaker profile checksum does not match canonical content")
        return self

    def computed_checksum(self) -> str:
        return sha256_json(self.model_dump(mode="json", exclude={"checksum"}))

    @classmethod
    def create(
        cls,
        *,
        speaker_id: str,
        consent_reference: str,
        speaker_global: tuple[float, ...],
        speaker_tokens: tuple[tuple[float, ...], ...],
        pitch_profile: PitchProfile,
        normalization: ProfileNormalization,
        provenance: Provenance,
        sample_rate: int = 24_000,
    ) -> Self:
        unverified = cls.model_construct(
            speaker_id=speaker_id,
            consent_reference=consent_reference,
            speaker_global=speaker_global,
            speaker_tokens=speaker_tokens,
            pitch_profile=pitch_profile,
            normalization=normalization,
            provenance=provenance,
            sample_rate=sample_rate,
        )
        return cls(
            speaker_id=speaker_id,
            consent_reference=consent_reference,
            speaker_global=speaker_global,
            speaker_tokens=speaker_tokens,
            pitch_profile=pitch_profile,
            normalization=normalization,
            provenance=provenance,
            sample_rate=sample_rate,
            checksum=sha256_json(unverified.model_dump(mode="json", exclude={"checksum"})),
        )

