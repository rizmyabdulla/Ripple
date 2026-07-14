"""ExecuTorch specialist backend capability entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends import BackendAvailability, ExecuTorchExporter


def capabilities() -> BackendAvailability:
    return ExecuTorchExporter.probe()


def export_executorch(
    model: Any, destination: str | Path, **kwargs: Any
) -> Path:
    return ExecuTorchExporter().export(model, destination, **kwargs)
