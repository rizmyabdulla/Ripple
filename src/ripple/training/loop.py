"""Single-process training loop used by the Ripple CLI."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ripple.training.batching import collate_waveforms
from ripple.training.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from ripple.training.factory import build_model, build_optimizer, trainer_config_from_flags
from ripple.training.stages import TrainingStage, get_stage_config
from ripple.training.steps import build_step_function
from ripple.training.trainer import Trainer


def _cycle(loader: DataLoader[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def run_training(
    *,
    manifest: Path,
    audio_root: Path,
    output_dir: Path,
    stage: str = "decoder_reconstruction",
    model_kind: str = "ripple",
    feature_dir: Path | None = None,
    device: str = "cpu",
    precision: str = "fp32",
    batch_size: int = 2,
    steps: int = 100,
    learning_rate: float = 2e-4,
    crop_samples: int = 4800,
    seed: int = 17,
    save_every: int = 50,
    resume_checkpoint: Path | None = None,
    resolved_config: Mapping[str, Any] | None = None,
) -> Path:
    """Train for a fixed step budget and return the final checkpoint path."""
    from ripple.data.dataset import RippleAudioDataset

    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_enum = TrainingStage(stage)
    stage_config = get_stage_config(stage_enum)

    dataset = RippleAudioDataset(
        manifest,
        audio_root,
        feature_dir=feature_dir,
        crop_samples=crop_samples,
    )
    if len(dataset) == 0:
        raise ValueError("training manifest has no records")
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_waveforms,
        drop_last=False,
    )

    model = build_model(model_kind)
    stage_config.apply_trainability(model)
    optimizer = build_optimizer(model, learning_rate=learning_rate)
    trainer = Trainer(
        model,
        optimizer,
        build_step_function(stage_enum),
        trainer_config_from_flags(device=device, precision=precision),
    )

    start_step = 0
    if resume_checkpoint is not None:
        metadata, _payload = load_checkpoint(
            resume_checkpoint, model=model, optimizer=optimizer
        )
        start_step = int(metadata.step)
        trainer.global_step = start_step
        trainer.optimizer_step = start_step

    config_payload = dict(resolved_config or {})
    config_payload.update(
        {
            "stage": stage_enum.value,
            "model_kind": model_kind,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "crop_samples": crop_samples,
            "seed": seed,
        }
    )

    batches = _cycle(loader)
    final_path = output_dir / "checkpoint_last.pt"
    for step in range(start_step + 1, start_step + steps + 1):
        metrics = trainer.train_batch(next(batches))
        if step % max(1, save_every) == 0 or step == start_step + steps:
            metadata = CheckpointMetadata.capture(
                stage=stage_enum.value,
                step=step,
                resolved_config=config_payload,
                dataset_manifest_hash=str(getattr(dataset.manifest, "checksum", "")),
                random_seed=seed,
            )
            path = output_dir / f"checkpoint_step_{step:06d}.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scaler=trainer.scaler,
                ema_state=trainer.ema.state_dict() if trainer.ema is not None else None,
                metadata=metadata,
                extra={"metrics": metrics},
            )
            save_checkpoint(
                final_path,
                model=model,
                optimizer=optimizer,
                scaler=trainer.scaler,
                ema_state=trainer.ema.state_dict() if trainer.ema is not None else None,
                metadata=metadata,
                extra={"metrics": metrics},
            )
            print(
                f"step={step} loss={metrics.get('loss', 0):.6f} saved={path}",
                flush=True,
            )
        elif step % 10 == 0:
            print(f"step={step} loss={metrics.get('loss', 0):.6f}", flush=True)

    return final_path
