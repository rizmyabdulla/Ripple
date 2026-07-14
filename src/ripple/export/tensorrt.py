"""TensorRT specialist backend capability entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends import BackendAvailability, TensorRTExporter


def capabilities() -> BackendAvailability:
    return TensorRTExporter.probe()


def export_tensorrt(
    model: Any, destination: str | Path, **kwargs: Any
) -> Path:
    return TensorRTExporter().export(model, destination, **kwargs)
