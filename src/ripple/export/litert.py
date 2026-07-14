"""LiteRT specialist backend capability entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends import BackendAvailability, LiteRTExporter


def capabilities() -> BackendAvailability:
    return LiteRTExporter.probe()


def export_litert(model: Any, destination: str | Path, **kwargs: Any) -> Path:
    return LiteRTExporter().export(model, destination, **kwargs)
