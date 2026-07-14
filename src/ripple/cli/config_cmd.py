"""Inspect resolved Ripple YAML configuration."""

# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

from .common import PROJECT_ROOT, echo_json, load_resolved_config

app = typer.Typer(help="Show resolved Ripple configuration.")


@app.command("show")
def show(
    config_root: Path = typer.Option(
        PROJECT_ROOT / "configs", exists=True, file_okay=False
    ),
) -> None:
    config = load_resolved_config(config_root)
    echo_json(config.model_dump(mode="json"))


@app.command("checksum")
def checksum(
    config_root: Path = typer.Option(
        PROJECT_ROOT / "configs", exists=True, file_okay=False
    ),
) -> None:
    config = load_resolved_config(config_root)
    typer.echo(config.checksum)


if __name__ == "__main__":
    app()
