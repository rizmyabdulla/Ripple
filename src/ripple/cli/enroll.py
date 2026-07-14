"""Consent-gated speaker enrollment CLI adapter."""

# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

from ripple.safety import ProfilePolicy

from .common import build_backend, profile_use, read_consent

app = typer.Typer(help="Create a consent-bound Ripple speaker profile.")


@app.command()
def enroll(
    reference_audio: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_profile: Path = typer.Argument(...),
    artifact: Path = typer.Option(..., exists=True),
    profile_id: str = typer.Option(...),
    consent: Path = typer.Option(..., exists=True, dir_okay=False),
    intended_use: str = typer.Option("conversion"),
    backend: str = typer.Option(..., help="Factory in module:callable form."),
    overwrite: bool = typer.Option(False),
) -> None:
    ProfilePolicy().authorize(
        read_consent(consent), profile_use(intended_use), profile_id=profile_id
    )
    if output_profile.exists() and not overwrite:
        raise typer.BadParameter("output profile exists; pass --overwrite explicitly")
    encoder = build_backend(backend, artifact=artifact)
    method = getattr(encoder, "enroll_file", None)
    if method is None:
        raise typer.BadParameter(
            "backend must implement enroll_file(reference, output, profile_id=...)"
        )
    output_profile.parent.mkdir(parents=True, exist_ok=True)
    method(reference_audio, output_profile, profile_id=profile_id)
    typer.echo(str(output_profile))


if __name__ == "__main__":
    app()
