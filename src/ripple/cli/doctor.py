"""Environment and project health checks."""

# ruff: noqa: B008

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import typer

from .common import PROJECT_ROOT, echo_json, load_resolved_config

app = typer.Typer(help="Diagnose Ripple environment and project readiness.")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@app.command("run")
def run(
    config_root: Path = typer.Option(
        PROJECT_ROOT / "configs",
        exists=True,
        file_okay=False,
        help="Resolved YAML config root.",
    ),
    data_root: Path = typer.Option(PROJECT_ROOT / "data", help="Writable data root."),
    checkpoint_root: Path = typer.Option(
        PROJECT_ROOT / "checkpoints", help="Writable checkpoint root."
    ),
) -> None:
    import torch

    checks: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "extras": {
            "soundfile": _module_available("soundfile"),
            "torchcodec": _module_available("torchcodec"),
            "onnxruntime": _module_available("onnxruntime"),
            "transformers": _module_available("transformers"),
            "tensorboard": _module_available("tensorboard"),
            "pyarrow": _module_available("pyarrow"),
            "jiwer": _module_available("jiwer"),
        },
    }
    try:
        config = load_resolved_config(config_root)
        checks["config_checksum"] = config.checksum
        checks["config_ok"] = True
    except typer.BadParameter as exc:
        checks["config_ok"] = False
        checks["config_error"] = str(exc)

    for label, path in (("data_root", data_root), ("checkpoint_root", checkpoint_root)):
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".ripple_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            checks[f"{label}_writable"] = True
        except OSError as exc:
            checks[f"{label}_writable"] = False
            checks[f"{label}_error"] = str(exc)

    echo_json(checks)
    failed: list[str] = []
    if checks.get("config_ok") is not True:
        failed.append("config")
    if checks.get("data_root_writable") is not True:
        failed.append("data_root")
    if checks.get("checkpoint_root_writable") is not True:
        failed.append("checkpoint_root")
    extras = checks.get("extras")
    if isinstance(extras, dict) and not extras.get("soundfile"):
        failed.append("soundfile")
    if failed:
        typer.echo(f"doctor failed: {', '.join(failed)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
