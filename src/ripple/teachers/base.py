"""Lazy local-only adapter base for optional teacher dependencies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .protocol import TeacherFeatures


class OptionalDependencyError(RuntimeError):
    pass


class LazyTeacherAdapter(ABC):
    """Load a teacher only on first extraction and never download weights."""

    name = "teacher"
    sample_rate = 16_000
    frame_rate_hz = 50.0

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        backend: Any | None = None,
        layer: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self._backend = backend
        self.layer = layer
        self.device = device

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def _require_local_path(self) -> Path:
        if self.model_path is None:
            raise ValueError(
                f"{self.name} requires an explicit local model_path; "
                "automatic downloads are disabled"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        return self.model_path

    @abstractmethod
    def _load_backend(self) -> Any:
        raise NotImplementedError

    def _load(self) -> Any:
        if self._backend is None:
            self._backend = self._load_backend()
            if hasattr(self._backend, "eval"):
                self._backend.eval()
            if hasattr(self._backend, "to"):
                self._backend.to(self.device)
        return self._backend

    def extract(self, waveform: Tensor, sample_rate: int) -> TeacherFeatures:
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{self.name} expects {self.sample_rate} Hz audio, got {sample_rate}; "
                "resample explicitly in the data pipeline"
            )
        if waveform.ndim not in {1, 2}:
            raise ValueError("waveform must have shape [samples] or [batch, samples]")
        batched = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform
        backend = self._load()
        with torch.inference_mode():
            values = self._forward(backend, batched.to(self.device))
        if not isinstance(values, Tensor) or values.ndim != 3:
            raise RuntimeError("teacher backend must produce [batch, frames, channels]")
        return TeacherFeatures(
            values.detach().cpu(),
            self.frame_rate_hz,
            self.name,
            self.layer,
        )

    @abstractmethod
    def _forward(self, backend: Any, waveform: Tensor) -> Tensor:
        raise NotImplementedError


def import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise OptionalDependencyError(
            "teacher adapters require the optional 'transformers' dependency"
        ) from exc
    return transformers


def hidden_state(output: Any, layer: int | None) -> Tensor:
    if layer is not None:
        states = getattr(output, "hidden_states", None)
        if states is None:
            raise RuntimeError("backend did not return hidden states")
        return states[layer]
    value = getattr(output, "last_hidden_state", None)
    if value is None and isinstance(output, (tuple, list)) and output:
        value = output[0]
    if not isinstance(value, Tensor):
        raise RuntimeError("backend output has no last_hidden_state tensor")
    return value
