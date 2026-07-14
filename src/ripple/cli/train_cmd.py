"""Training CLI for Ripple STS stages."""

# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

from ripple.training.loop import run_training
from ripple.training.stages import DEFAULT_STAGES, TrainingStage

from .common import PROJECT_ROOT, echo_json, load_resolved_config, resolve_device, set_seed

app = typer.Typer(help="Train Ripple-VC stages (single-process).")


@app.command("stages")
def stages() -> None:
    payload = {
        stage.value: {
            "loss_weights": dict(config.loss_weights),
            "train_modules": list(config.train_modules),
            "adversarial": config.adversarial,
            "stateful": config.stateful,
        }
        for stage, config in DEFAULT_STAGES.items()
    }
    echo_json(payload)


@app.command("run")
def run(
    manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    audio_root: Path = typer.Option(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(...),
    stage: str = typer.Option(TrainingStage.RECONSTRUCTION.value),
    model_kind: str = typer.Option("ripple"),
    feature_dir: Path | None = typer.Option(None),
    config_root: Path = typer.Option(PROJECT_ROOT / "configs", exists=True, file_okay=False),
    device: str = typer.Option("cpu"),
    precision: str = typer.Option("fp32"),
    batch_size: int = typer.Option(2, min=1),
    steps: int = typer.Option(100, min=1),
    learning_rate: float = typer.Option(2e-4, min=0.0),
    crop_samples: int = typer.Option(4800, min=480),
    seed: int = typer.Option(17, min=0),
    save_every: int = typer.Option(50, min=1),
) -> None:
    resolve_device(device)
    set_seed(seed)
    config = load_resolved_config(config_root)
    if feature_dir is not None and not feature_dir.is_dir():
        raise typer.BadParameter(f"feature_dir not found: {feature_dir}")
    final = run_training(
        manifest=manifest,
        audio_root=audio_root,
        output_dir=output_dir,
        stage=stage,
        model_kind=model_kind,
        feature_dir=feature_dir,
        device=device,
        precision=precision,
        batch_size=batch_size,
        steps=steps,
        learning_rate=learning_rate,
        crop_samples=crop_samples,
        seed=seed,
        save_every=save_every,
        resolved_config=config.model_dump(mode="json"),
    )
    typer.echo(str(final))


@app.command("resume")
def resume(
    checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    audio_root: Path = typer.Option(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(...),
    stage: str = typer.Option(TrainingStage.RECONSTRUCTION.value),
    model_kind: str = typer.Option("ripple"),
    feature_dir: Path | None = typer.Option(None),
    config_root: Path = typer.Option(PROJECT_ROOT / "configs", exists=True, file_okay=False),
    device: str = typer.Option("cpu"),
    precision: str = typer.Option("fp32"),
    batch_size: int = typer.Option(2, min=1),
    steps: int = typer.Option(100, min=1),
    learning_rate: float = typer.Option(2e-4, min=0.0),
    crop_samples: int = typer.Option(4800, min=480),
    seed: int = typer.Option(17, min=0),
    save_every: int = typer.Option(50, min=1),
) -> None:
    resolve_device(device)
    set_seed(seed)
    config = load_resolved_config(config_root)
    final = run_training(
        manifest=manifest,
        audio_root=audio_root,
        output_dir=output_dir,
        stage=stage,
        model_kind=model_kind,
        feature_dir=feature_dir,
        device=device,
        precision=precision,
        batch_size=batch_size,
        steps=steps,
        learning_rate=learning_rate,
        crop_samples=crop_samples,
        seed=seed,
        save_every=save_every,
        resume_checkpoint=checkpoint,
        resolved_config=config.model_dump(mode="json"),
    )
    typer.echo(str(final))


if __name__ == "__main__":
    app()
