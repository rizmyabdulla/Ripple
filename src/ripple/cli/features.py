"""Offline teacher-feature extraction CLI."""

# ruff: noqa: B008

from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from ripple.teachers import create_teacher

from .common import read_pcm16_wav

app = typer.Typer(help="Extract local-only teacher features from a JSONL manifest.")


@app.command()
def extract(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    audio_root: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Argument(...),
    teacher: str = typer.Option(..., help="hubert, wavlm, xls-r, or whisper"),
    model_path: Path = typer.Option(..., exists=True),
    layer: int | None = typer.Option(None),
    device: str = typer.Option("cpu"),
) -> None:
    adapter = create_teacher(teacher, model_path, layer=layer, device=device)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("id", ""))
            relative_path = row.get("path")
            if not item_id or not isinstance(relative_path, str):
                raise typer.BadParameter(
                    f"manifest line {line_number} requires id and path"
                )
            waveform, sample_rate = read_pcm16_wav(audio_root / relative_path)
            features = adapter.extract(waveform, sample_rate)
            torch.save(
                {
                    "values": features.values,
                    "frame_rate_hz": features.frame_rate_hz,
                    "teacher": features.teacher,
                    "layer": features.layer,
                    "source_sha256": row.get("sha256"),
                },
                output / f"{item_id}.pt",
            )
            count += 1
    typer.echo(f"Wrote {count} feature files to {output}")


if __name__ == "__main__":
    app()
