"""Streaming conversion demo CLI adapter."""

# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

from ripple.safety import ProfilePolicy, ProfileUse

from .common import build_backend, read_consent

app = typer.Typer(help="Run chunked conversion through an installed backend.")


@app.command()
def stream(
    input_audio: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_audio: Path = typer.Argument(...),
    artifact: Path = typer.Option(..., exists=True),
    profile: Path = typer.Option(..., exists=True),
    profile_id: str = typer.Option(...),
    consent: Path = typer.Option(..., exists=True, dir_okay=False),
    backend: str = typer.Option(..., help="Factory in module:callable form."),
    chunk_ms: int = typer.Option(20, min=1),
) -> None:
    ProfilePolicy().authorize(
        read_consent(consent), ProfileUse.CONVERSION, profile_id=profile_id
    )
    runtime = build_backend(backend, artifact=artifact, profile=profile)
    method = getattr(runtime, "stream_file", None)
    if method is None:
        raise typer.BadParameter(
            "backend must implement stream_file(input, output, chunk_ms=...)"
        )
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    method(input_audio, output_audio, chunk_ms=chunk_ms)
    typer.echo(str(output_audio))


if __name__ == "__main__":
    app()
