"""Manifest builder CLI."""

# ruff: noqa: B008

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import typer

app = typer.Typer(help="Build an immutable JSONL manifest from local WAV files.")


@app.command()
def build(
    source: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Argument(...),
    license_id: str = typer.Option(..., "--license", help="Dataset license/SPDX id."),
    consent_basis: str = typer.Option(
        ..., help="Recorded consent or lawful processing basis."
    ),
    language: str = typer.Option("und"),
) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(source.rglob("*.wav")):
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
        relative = path.relative_to(source).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "id": digest[:24],
                "path": relative,
                "sha256": digest,
                "sample_rate": sample_rate,
                "channels": channels,
                "frames": frames,
                "duration_seconds": frames / sample_rate,
                "language": language,
                "license": license_id,
                "consent_basis": consent_basis,
            }
        )
    if not rows:
        raise typer.BadParameter("source contains no WAV files")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    typer.echo(f"Wrote {len(rows)} records to {output}")


if __name__ == "__main__":
    app()
