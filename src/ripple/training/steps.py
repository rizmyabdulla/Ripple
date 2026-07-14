"""Stage-specific training step functions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn

from ripple.models.ripple_vc import RippleVC
from ripple.training.losses import (
    MultiResolutionSTFTLoss,
    MultiScaleMelLoss,
    semantic_ce_loss,
    semantic_feature_loss,
)
from ripple.training.stages import TrainingStage, get_stage_config


def _waveform_l1(prediction: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(prediction, target)


class ReconstructionStep:
    """Auto-encode source speech with an enrolled self-profile."""

    def __init__(self) -> None:
        self.stft = MultiResolutionSTFTLoss()
        self.mel = MultiScaleMelLoss()

    def __call__(
        self, model: nn.Module, batch: Mapping[str, Any]
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        if not isinstance(model, RippleVC):
            raise TypeError("reconstruction stage expects RippleVC")
        waveform = batch["waveform"]
        profile = model.enroll(waveform)
        output = model(waveform, profile)
        pred = output.waveform
        # Align lengths if decoder hop introduces minor mismatch.
        length = min(pred.shape[-1], waveform.shape[-1])
        pred = pred[..., :length]
        target = waveform[..., :length]
        loss_wave = _waveform_l1(pred, target)
        loss_stft = self.stft(pred.squeeze(1), target.squeeze(1))
        loss_mel = self.mel(pred.squeeze(1), target.squeeze(1))
        total = loss_wave + loss_stft + loss_mel
        return total, {
            "waveform": float(loss_wave.detach()),
            "stft": float(loss_stft.detach()),
            "mel": float(loss_mel.detach()),
            "loss": float(total.detach()),
        }


class SemanticStep:
    """Distill analysis/mixer semantic features toward teacher embeddings."""

    def __init__(self) -> None:
        self._proj: nn.Module | None = None

    def __call__(
        self, model: nn.Module, batch: Mapping[str, Any]
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        if not isinstance(model, RippleVC):
            raise TypeError("semantic stage expects RippleVC")
        if "teacher" not in batch:
            raise ValueError("semantic stage requires batch['teacher'] features")
        waveform = batch["waveform"]
        teacher = batch["teacher"]
        profile = model.enroll(waveform)
        output = model(waveform, profile)
        student = output.semantic_embed  # [B, C, T]
        student_bt = student.transpose(1, 2)  # [B, T, C]
        # Align time
        frames = min(student_bt.shape[1], teacher.shape[1])
        student_bt = student_bt[:, :frames]
        teacher = teacher[:, :frames]
        if teacher.shape[-1] != student_bt.shape[-1]:
            if self._proj is None:
                self._proj = nn.Linear(teacher.shape[-1], student_bt.shape[-1]).to(
                    student_bt.device, dtype=student_bt.dtype
                )
            teacher = self._proj(teacher)
        loss_feat = semantic_feature_loss(student_bt, teacher)
        # Soft CE/KL against teacher-aligned continuous targets via cosine-as-logits proxy
        student_logits = output.semantic_soft.transpose(1, 2)[:, :frames]
        # Use argmax of student soft as hard proxy when discrete teacher units unavailable.
        hard = student_logits.argmax(dim=-1)
        loss_ce = semantic_ce_loss(student_logits, hard)
        total = loss_feat + 0.1 * loss_ce
        return total, {
            "semantic_feature": float(loss_feat.detach()),
            "semantic_ce": float(loss_ce.detach()),
            "loss": float(total.detach()),
        }


def build_step_function(stage: TrainingStage | str):
    resolved = TrainingStage(stage) if not isinstance(stage, TrainingStage) else stage
    get_stage_config(resolved)  # validate known stage
    if resolved is TrainingStage.RECONSTRUCTION:
        return ReconstructionStep()
    if resolved is TrainingStage.SEMANTIC:
        return SemanticStep()
    raise ValueError(
        f"CLI training for stage {resolved.value!r} is not wired yet; "
        "supported: decoder_reconstruction, semantic_student"
    )
