"""Portable process and model memory accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import Tensor, nn


def process_rss_bytes() -> int:
    from ripple.evaluation.long_session import _rss_bytes

    return _rss_bytes()


def tensor_bytes(value: Tensor) -> int:
    return value.numel() * value.element_size()


def nested_tensor_bytes(value: Any) -> int:
    if isinstance(value, Tensor):
        return tensor_bytes(value)
    if isinstance(value, dict):
        return sum(nested_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nested_tensor_bytes(item) for item in value)
    return 0


def model_memory_report(
    model: nn.Module,
    *,
    state: Any = None,
    artifact_path: str | Path | None = None,
) -> dict[str, int]:
    parameters = sum(tensor_bytes(parameter) for parameter in model.parameters())
    buffers = sum(tensor_bytes(buffer) for buffer in model.buffers())
    artifact = Path(artifact_path).stat().st_size if artifact_path is not None else 0
    return {
        "parameter_bytes": parameters,
        "buffer_bytes": buffers,
        "state_bytes": nested_tensor_bytes(state),
        "artifact_bytes": artifact,
        "rss_bytes": process_rss_bytes(),
    }
