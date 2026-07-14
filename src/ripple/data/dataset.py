"""PyTorch dataset over sealed Ripple manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ripple.audio.io import load_audio
from ripple.audio.resample import resample_audio
from ripple.data.manifest import AudioRecord, DatasetManifest, read_manifest


def resolve_audio_path(record: AudioRecord, audio_root: Path) -> Path:
    uri = record.uri
    if "://" in uri:
        # Treat scheme-local paths as relative to audio_root after the scheme.
        relative = uri.split("://", 1)[1]
        return audio_root / relative
    return audio_root / uri


class RippleAudioDataset(Dataset[dict[str, Any]]):
    """Load waveforms (and optional teacher features) for sealed records."""

    def __init__(
        self,
        manifest: DatasetManifest | Path | str,
        audio_root: Path | str,
        *,
        feature_dir: Path | str | None = None,
        target_sample_rate: int = 24_000,
        crop_samples: int | None = 24_000,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            self.manifest = read_manifest(manifest)
        else:
            self.manifest = manifest
        self.audio_root = Path(audio_root)
        self.feature_dir = Path(feature_dir) if feature_dir is not None else None
        self.target_sample_rate = target_sample_rate
        self.crop_samples = crop_samples
        self.records = list(self.manifest.records)

    def __len__(self) -> int:
        return len(self.records)

    def _load_waveform(self, record: AudioRecord) -> Tensor:
        path = resolve_audio_path(record, self.audio_root)
        audio = load_audio(path)
        if audio.sample_rate != self.target_sample_rate:
            audio = resample_audio(audio, self.target_sample_rate)
        mono = audio.samples.mean(axis=0)
        waveform = torch.from_numpy(mono.copy()).float()
        if self.crop_samples is not None:
            if waveform.numel() < self.crop_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.crop_samples - waveform.numel())
                )
            elif waveform.numel() > self.crop_samples:
                waveform = waveform[: self.crop_samples]
        return waveform

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform = self._load_waveform(record)
        item: dict[str, Any] = {
            "record_id": record.record_id,
            "speaker_id": record.speaker_id,
            "language": record.language,
            "waveform": waveform,
            "sample_rate": self.target_sample_rate,
        }
        if self.feature_dir is not None:
            feature_path = self.feature_dir / f"{record.record_id}.pt"
            if feature_path.is_file():
                payload = torch.load(feature_path, map_location="cpu", weights_only=False)
                values = payload["values"] if isinstance(payload, dict) else payload
                item["teacher"] = values.float()
        return item
