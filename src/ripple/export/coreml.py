"""Core ML specialist backend capability entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends import BackendAvailability, CoreMLExporter


def capabilities() -> BackendAvailability:
    return CoreMLExporter.probe()


def export_coreml(model: Any, destination: str | Path, **kwargs: Any) -> Path:
    return CoreMLExporter().export(model, destination, **kwargs)
