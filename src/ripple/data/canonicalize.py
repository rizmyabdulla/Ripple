"""Canonicalize audio to Ripple Edge PCM conventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ripple.audio.io import AudioBuffer, load_audio, save_audio
from ripple.audio.resample import resample_audio
from ripple.contracts.checksums import sha256_file


def canonicalize_file(
    source: Path,
    destination: Path,
    *,
    sample_rate: int = 24_000,
) -> dict[str, object]:
    """Load, mono-mix, resample, and write PCM16 WAV."""
    audio = load_audio(source)
    if audio.channels > 1:
        mono = audio.samples.mean(axis=0, keepdims=True).astype(np.float32)
        audio = AudioBuffer(mono, audio.sample_rate)
    if audio.sample_rate != sample_rate:
        audio = resample_audio(audio, sample_rate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_audio(destination, audio, subtype="PCM_16")
    return {
        "source": source.as_posix(),
        "destination": destination.as_posix(),
        "sample_rate": sample_rate,
        "channels": 1,
        "frames": audio.frames,
        "duration_seconds": audio.frames / float(sample_rate),
        "checksum": sha256_file(destination),
    }


def canonicalize_tree(
    source_root: Path,
    output_root: Path,
    *,
    sample_rate: int = 24_000,
    pattern: str = "*.wav",
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(source_root.rglob(pattern)):
        relative = path.relative_to(source_root)
        destination = output_root / relative.with_suffix(".wav")
        records.append(
            canonicalize_file(path, destination, sample_rate=sample_rate)
        )
    return records
