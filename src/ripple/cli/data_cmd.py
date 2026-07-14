"""Data canonicalize and summarize commands."""

# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

from ripple.data.canonicalize import canonicalize_tree
from ripple.data.manifest import read_manifest
from ripple.data.seal import summarize_manifest

from .common import echo_json

app = typer.Typer(help="Canonicalize audio and summarize sealed manifests.")


@app.command("canonicalize")
def canonicalize(
    source: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Argument(...),
    sample_rate: int = typer.Option(24_000, min=1),
) -> None:
    rows = canonicalize_tree(source, output, sample_rate=sample_rate)
    if not rows:
        raise typer.BadParameter("source contains no WAV files")
    typer.echo(f"Canonicalized {len(rows)} files into {output}")


@app.command("summarize")
def summarize(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    echo_json(summarize_manifest(read_manifest(manifest)))


if __name__ == "__main__":
    app()
