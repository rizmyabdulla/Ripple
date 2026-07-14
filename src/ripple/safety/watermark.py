"""Watermark embedding/detection contracts and fail-closed policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


@dataclass(frozen=True)
class WatermarkPayload:
    content_id: str
    provenance_hash: str
    attributes: Mapping[str, str]


@dataclass(frozen=True)
class WatermarkEmbedResult:
    audio: Tensor
    scheme: str
    embedded: bool


@dataclass(frozen=True)
class WatermarkDetection:
    detected: bool
    confidence: float
    payload: WatermarkPayload | None = None


@runtime_checkable
class AudioWatermarker(Protocol):
    scheme: str

    def embed(
        self, audio: Tensor, sample_rate: int, payload: WatermarkPayload
    ) -> WatermarkEmbedResult: ...

    def detect(self, audio: Tensor, sample_rate: int) -> WatermarkDetection: ...


class WatermarkRequired(RuntimeError):
    pass


class WatermarkPolicy:
    """Enforce watermark presence without prescribing an implementation."""

    def __init__(
        self,
        watermarker: AudioWatermarker | None,
        *,
        required: bool = True,
        min_detection_confidence: float = 0.8,
    ) -> None:
        self.watermarker = watermarker
        self.required = required
        self.min_detection_confidence = min_detection_confidence

    def protect(
        self, audio: Tensor, sample_rate: int, payload: WatermarkPayload
    ) -> WatermarkEmbedResult:
        if self.watermarker is None:
            if self.required:
                raise WatermarkRequired("release policy requires a watermarker")
            return WatermarkEmbedResult(audio, "none", False)
        result = self.watermarker.embed(audio, sample_rate, payload)
        if self.required and not result.embedded:
            raise WatermarkRequired("watermarker did not confirm embedding")
        return result

    def verify(self, audio: Tensor, sample_rate: int) -> WatermarkDetection:
        if self.watermarker is None:
            if self.required:
                raise WatermarkRequired("release policy requires detection")
            return WatermarkDetection(False, 0.0)
        result = self.watermarker.detect(audio, sample_rate)
        if self.required and (
            not result.detected
            or result.confidence < self.min_detection_confidence
        ):
            raise WatermarkRequired("watermark missing or below confidence threshold")
        return result
