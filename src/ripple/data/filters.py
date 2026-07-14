"""Deterministic metadata and quality gates for manifest ingestion."""

from __future__ import annotations

from pydantic import Field

from ripple.contracts.manifest import ContractModel
from ripple.data.manifest import AudioRecord, ConsentStatus


class QualityMetrics(ContractModel):
    clipped_fraction: float = Field(ge=0.0, le=1.0)
    snr_db: float
    speech_fraction: float = Field(ge=0.0, le=1.0)


class QualityPolicy(ContractModel):
    max_clipped_fraction: float = Field(default=0.001, ge=0.0, le=1.0)
    min_snr_db: float = 10.0
    min_speech_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    require_commercial_use: bool = True


def rejection_reasons(
    record: AudioRecord,
    metrics: QualityMetrics,
    policy: QualityPolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if metrics.clipped_fraction > policy.max_clipped_fraction:
        reasons.append("clipping")
    if metrics.snr_db < policy.min_snr_db:
        reasons.append("snr")
    if metrics.speech_fraction < policy.min_speech_fraction:
        reasons.append("speech_fraction")
    if policy.require_commercial_use and not record.consent.commercial_use:
        reasons.append("commercial_consent")
    if record.consent.status is ConsentStatus.RESTRICTED:
        reasons.append("restricted_consent")
    return tuple(reasons)

