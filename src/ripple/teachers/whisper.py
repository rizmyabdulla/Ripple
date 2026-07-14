"""Lazy Whisper encoder teacher adapter."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .base import LazyTeacherAdapter, hidden_state, import_transformers


class WhisperTeacher(LazyTeacherAdapter):
    name = "whisper"

    def _load_backend(self) -> Any:
        path = str(self._require_local_path())
        transformers = import_transformers()
        extractor = transformers.AutoFeatureExtractor.from_pretrained(
            path, local_files_only=True
        )
        model = transformers.WhisperModel.from_pretrained(path, local_files_only=True)
        model.eval().to(self.device)
        return {"extractor": extractor, "model": model}

    def _forward(self, backend: Any, waveform: Tensor) -> Tensor:
        if isinstance(backend, dict):
            audio = [row.detach().cpu().numpy() for row in waveform]
            inputs = backend["extractor"](
                audio, sampling_rate=self.sample_rate, return_tensors="pt"
            )
            features = inputs.input_features.to(self.device)
            output = backend["model"].encoder(
                features,
                output_hidden_states=self.layer is not None,
                return_dict=True,
            )
        else:
            # Injected test/custom backends follow the same waveform protocol as
            # the other teachers and can own feature extraction themselves.
            output = backend(
                waveform,
                output_hidden_states=self.layer is not None,
                return_dict=True,
            )
        return hidden_state(output, self.layer)
