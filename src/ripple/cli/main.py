"""Unified Ripple command-line application."""

from __future__ import annotations

import typer

from .benchmark import app as benchmark_app
from .config_cmd import app as config_app
from .convert import app as convert_app
from .data_cmd import app as data_app
from .doctor import app as doctor_app
from .enroll import app as enroll_app
from .eval_cmd import app as eval_app
from .export import app as export_app
from .features import app as features_app
from .manifest import app as manifest_app
from .stream import app as stream_app
from .train_cmd import app as train_app

app = typer.Typer(help="Ripple research, training, and runtime tools.", no_args_is_help=True)
app.add_typer(doctor_app, name="doctor")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(manifest_app, name="manifest")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(convert_app, name="convert")
app.add_typer(stream_app, name="stream")
app.add_typer(enroll_app, name="enroll")
app.add_typer(export_app, name="export")
app.add_typer(benchmark_app, name="benchmark")


def main() -> None:
    """Console-script entrypoint for the Ripple CLI."""
    app()


if __name__ == "__main__":
    main()
