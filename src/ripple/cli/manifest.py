"""Manifest builder and sealed-contract CLI."""

# ruff: noqa: B008

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import typer

from ripple.contracts.checksums import sha256_file
from ripple.data.manifest import read_manifest
from ripple.data.seal import seal_draft_manifest, summarize_manifest

from .common import echo_json

app = typer.Typer(help="Build, seal, and validate Ripple dataset manifests.")


@app.command("build")
def build(
    source: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Argument(...),
    license_id: str = typer.Option(..., "--license", help="Dataset license/SPDX id."),
    consent_basis: str = typer.Option(
        ..., help="Recorded consent or lawful processing basis."
    ),
    language: str = typer.Option("en"),
) -> None:
    """Build an informal discovery JSONL from local WAV files."""
    rows: list[dict[str, object]] = []
    for path in sorted(source.rglob("*.wav")):
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
        relative = path.relative_to(source).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Path-scoped id so identical silence/content across speakers stays unique.
        record_id = hashlib.sha256(f"{relative}:{digest}".encode()).hexdigest()[:24]
        speaker = path.relative_to(source).parts[0] if path.relative_to(source).parts else "speaker"
        rows.append(
            {
                "id": record_id,
                "path": relative,
                "sha256": digest,
                "speaker_id": speaker,
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
    typer.echo(f"Wrote {len(rows)} discovery records to {output}")


@app.command("seal")
def seal(
    draft: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Argument(...),
    dataset_id: str = typer.Option(...),
    revision: str = typer.Option(...),
    audio_root: Path | None = typer.Option(None, exists=True, file_okay=False),
    seed: int = typer.Option(17, min=0),
    overwrite: bool = typer.Option(False),
) -> None:
    """Seal discovery JSONL into an immutable DatasetManifest."""
    manifest = seal_draft_manifest(
        draft,
        dataset_id=dataset_id,
        revision=revision,
        output=output,
        audio_root=audio_root,
        seed=seed,
        overwrite=overwrite,
    )
    typer.echo(f"{output} checksum={manifest.checksum}")


@app.command("validate")
def validate(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    audio_root: Path | None = typer.Option(None, exists=True, file_okay=False),
    verify_checksums: bool = typer.Option(False),
) -> None:
    sealed = read_manifest(manifest)
    if verify_checksums:
        if audio_root is None:
            raise typer.BadParameter("--audio-root is required with --verify-checksums")
        for record in sealed.records:
            path = audio_root / record.uri
            if not path.is_file():
                raise typer.BadParameter(f"missing audio: {path}")
            digest = sha256_file(path)
            if digest != record.checksum:
                raise typer.BadParameter(
                    f"checksum mismatch for {record.record_id}: {digest} != {record.checksum}"
                )
    echo_json(summarize_manifest(sealed))


if __name__ == "__main__":
    app()
