"""Evaluation CLI over Ripple checkpoints."""

# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import torch
import typer

from ripple.data.dataset import RippleAudioDataset
from ripple.evaluation.quality import waveform_report
from ripple.training.checkpoint import load_checkpoint
from ripple.training.factory import build_model

from .common import echo_json, resolve_device

app = typer.Typer(help="Evaluate Ripple checkpoints on sealed manifests.")


@app.command("run")
def run(
    checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    audio_root: Path = typer.Option(..., exists=True, file_okay=False),
    model_kind: str = typer.Option("ripple"),
    device: str = typer.Option("cpu"),
    max_items: int = typer.Option(8, min=1),
    crop_samples: int = typer.Option(4800, min=480),
) -> None:
    resolve_device(device)
    model = build_model(model_kind)
    load_checkpoint(checkpoint, model=model, map_location=device)
    model.to(device)
    model.eval()
    dataset = RippleAudioDataset(
        manifest, audio_root, crop_samples=crop_samples
    )
    reports: list[dict[str, float]] = []
    with torch.inference_mode():
        for index in range(min(max_items, len(dataset))):
            item = dataset[index]
            waveform = item["waveform"].unsqueeze(0).unsqueeze(0).to(device)
            profile = model.enroll(waveform)
            output = model(waveform, profile)
            length = min(output.waveform.shape[-1], waveform.shape[-1])
            reports.append(
                waveform_report(
                    waveform[..., :length].cpu(),
                    output.waveform[..., :length].cpu(),
                )
            )
    if not reports:
        raise typer.BadParameter("no items evaluated")
    keys = reports[0].keys()
    means = {key: sum(report[key] for report in reports) / len(reports) for key in keys}
    echo_json({"items": len(reports), "metrics": means})


if __name__ == "__main__":
    app()
