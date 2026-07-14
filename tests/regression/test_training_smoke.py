from __future__ import annotations

import torch
from torch import nn

from ripple.training import (
    CheckpointMetadata,
    ExponentialMovingAverage,
    Trainer,
    TrainerConfig,
    TrainingStage,
    get_stage_config,
    load_checkpoint,
)


def test_native_trainer_checkpoint_and_ema_smoke(tmp_path) -> None:
    torch.manual_seed(2)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

    def step(module: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]):
        inputs, targets = batch
        loss = torch.nn.functional.mse_loss(module(inputs), targets)
        return loss, {"mse": loss}

    trainer = Trainer(
        model,
        optimizer,
        step,
        TrainerConfig(gradient_accumulation=2, ema_decay=0.9),
    )
    batch = (torch.randn(3, 4), torch.randn(3, 2))
    first = trainer.train_batch(batch)
    second = trainer.train_batch(batch)
    third = trainer.train_batch(batch)
    assert first["loss"] == second["loss"]
    assert third["loss"] < second["loss"]
    assert trainer.optimizer_step == 1
    assert trainer.ema is not None

    metadata = CheckpointMetadata(
        stage=TrainingStage.SEMANTIC.value,
        step=trainer.global_step,
        resolved_config={"lr": 0.05},
        random_seed=2,
    )
    path = tmp_path / "smoke.pt"
    digest = trainer.checkpoint(path, metadata)
    assert len(digest) == 64

    restored = nn.Linear(4, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.05)
    restored_metadata, payload = load_checkpoint(
        path, model=restored, optimizer=restored_optimizer
    )
    assert restored_metadata.step == 3
    assert payload["ema"] is not None
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=False):
        assert torch.equal(expected, actual)


def test_stage_trainability_and_ema_copy() -> None:
    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.analysis_encoder = nn.Linear(2, 2)
            self.decoder = nn.Linear(2, 2)

    model = Toy()
    get_stage_config(TrainingStage.SEMANTIC).apply_trainability(model)
    assert all(parameter.requires_grad for parameter in model.analysis_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.decoder.parameters())

    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    ema.update(model)
    copy = Toy()
    ema.copy_to(copy)
    assert all(torch.isfinite(parameter).all() for parameter in copy.parameters())
