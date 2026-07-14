"""Optional training-only teacher adapters.

Importing this package never imports Transformers or downloads model weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import LazyTeacherAdapter, OptionalDependencyError
from .hubert import HuBERTTeacher
from .protocol import SpeechTeacher, TeacherFeatures
from .wavlm import WavLMTeacher
from .whisper import WhisperTeacher
from .xlsr import XLSRTeacher

type TeacherAdapterType = (
    type[HuBERTTeacher]
    | type[WavLMTeacher]
    | type[XLSRTeacher]
    | type[WhisperTeacher]
)

_ADAPTERS: dict[str, TeacherAdapterType] = {
    "hubert": HuBERTTeacher,
    "wavlm": WavLMTeacher,
    "xls-r": XLSRTeacher,
    "xlsr": XLSRTeacher,
    "whisper": WhisperTeacher,
}


def create_teacher(
    name: str,
    model_path: str | Path | None = None,
    **kwargs: Any,
) -> LazyTeacherAdapter:
    try:
        adapter = _ADAPTERS[name.casefold()]
    except KeyError as exc:
        raise ValueError(
            f"unknown teacher {name!r}; choose from hubert, wavlm, xls-r, whisper"
        ) from exc
    return adapter(model_path, **kwargs)


__all__ = [
    "HuBERTTeacher",
    "LazyTeacherAdapter",
    "OptionalDependencyError",
    "SpeechTeacher",
    "TeacherFeatures",
    "WavLMTeacher",
    "WhisperTeacher",
    "XLSRTeacher",
    "create_teacher",
]
