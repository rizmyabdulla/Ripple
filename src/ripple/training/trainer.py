"""Native PyTorch training loop primitives with accumulation, AMP and EMA."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn

from .checkpoint import CheckpointMetadata, save_checkpoint


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0 <= decay < 1:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self) -> dict[str, Tensor]:
        return self.shadow

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self.shadow = {name: value.detach().clone() for name, value in state.items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


@dataclass(frozen=True)
class TrainerConfig:
    device: str = "cpu"
    precision: str = "fp32"
    gradient_accumulation: int = 1
    max_gradient_norm: float | None = None
    ema_decay: float | None = 0.999

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be positive")


StepFunction = Callable[[nn.Module, Any], Tensor | tuple[Tensor, Mapping[str, Tensor | float]]]


class Trainer:
    """Framework-neutral loop suitable for single-process or DDP models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step_function: StepFunction,
        config: TrainerConfig | None = None,
        scheduler: Any | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.step_function = step_function
        self.scheduler = scheduler
        amp_enabled = self.config.precision == "fp16" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=amp_enabled)
        self.ema = (
            ExponentialMovingAverage(self.model, self.config.ema_decay)
            if self.config.ema_decay is not None
            else None
        )
        self.global_step = 0
        self.optimizer_step = 0
        self.optimizer.zero_grad(set_to_none=True)

    def _move(self, value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, Mapping):
            return type(value)((key, self._move(item)) for key, item in value.items())
        if isinstance(value, tuple):
            return tuple(self._move(item) for item in value)
        if isinstance(value, list):
            return [self._move(item) for item in value]
        return value

    def train_batch(self, batch: Any) -> dict[str, float]:
        self.model.train()
        batch = self._move(batch)
        accumulation = self.config.gradient_accumulation
        should_step = (self.global_step + 1) % accumulation == 0
        sync_context = (
            self.model.no_sync()
            if hasattr(self.model, "no_sync") and not should_step
            else nullcontext()
        )
        amp_dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
        amp_enabled = self.config.precision != "fp32" and (
            self.device.type == "cuda" or self.config.precision == "bf16"
        )
        with sync_context, torch.autocast(self.device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = self.step_function(self.model, batch)
            loss, metrics = output if isinstance(output, tuple) else (output, {})
            scaled_loss = loss / accumulation
        self.scaler.scale(scaled_loss).backward()

        gradient_norm = None
        if should_step:
            self.scaler.unscale_(self.optimizer)
            if self.config.max_gradient_norm is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_gradient_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)
            self.optimizer_step += 1

        self.global_step += 1
        result = {
            "loss": float(loss.detach()),
            **{
                name: float(value.detach()) if isinstance(value, Tensor) else float(value)
                for name, value in metrics.items()
            },
        }
        if gradient_norm is not None:
            result["gradient_norm"] = float(gradient_norm)
        return result

    def fit(self, loader: Any, *, max_steps: int | None = None) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for batch in loader:
            history.append(self.train_batch(batch))
            if max_steps is not None and self.global_step >= max_steps:
                break
        return history

    def checkpoint(
        self,
        path: str | Path,
        metadata: CheckpointMetadata,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            ema_state=self.ema.state_dict() if self.ema is not None else None,
            metadata=metadata,
            extra={"global_step": self.global_step, "optimizer_step": self.optimizer_step, **dict(extra or {})},
        )


class AdversarialOptimizers:
    """Explicit generator/discriminator stepping scaffold."""

    def __init__(
        self,
        generator: torch.optim.Optimizer,
        discriminator: torch.optim.Optimizer,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator

    def step_discriminator(self, loss: Tensor) -> None:
        self.discriminator.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator.step()

    def step_generator(self, loss: Tensor) -> None:
        self.generator.zero_grad(set_to_none=True)
        loss.backward()
        self.generator.step()
