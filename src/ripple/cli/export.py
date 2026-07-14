"""Model export CLI adapter."""

# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

from ripple.safety import sha256_file

from .common import build_backend

app = typer.Typer(help="Export and checksum a Ripple model artifact.")


@app.command("export")
def export_model(
    checkpoint: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Argument(...),
    backend: str = typer.Option(..., help="Exporter factory in module:callable form."),
    target: str = typer.Option("onnx"),
) -> None:
    exporter = build_backend(backend, target=target)
    method = getattr(exporter, "export_file", None)
    if method is None:
        raise typer.BadParameter(
            "backend must implement export_file(checkpoint, output)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    method(checkpoint, output)
    if not output.is_file():
        raise RuntimeError("export backend did not create the output artifact")
    typer.echo(f"{output} sha256={sha256_file(output)}")


if __name__ == "__main__":
    app()
