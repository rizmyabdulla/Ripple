"""Lazy WavLM teacher adapter."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .base import LazyTeacherAdapter, hidden_state, import_transformers


class WavLMTeacher(LazyTeacherAdapter):
    name = "wavlm"

    def _load_backend(self) -> Any:
        path = str(self._require_local_path())
        transformers = import_transformers()
        return transformers.AutoModel.from_pretrained(path, local_files_only=True)

    def _forward(self, backend: Any, waveform: Tensor) -> Tensor:
        output = backend(
            input_values=waveform,
            output_hidden_states=self.layer is not None,
            return_dict=True,
        )
        return hidden_state(output, self.layer)
