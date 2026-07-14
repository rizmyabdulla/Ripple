"""Typed training-stage definitions and curriculum defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from torch import nn


class TrainingStage(str, Enum):
    SEMANTIC = "semantic_student"
    SPEAKER = "speaker_enrollment"
    RECONSTRUCTION = "decoder_reconstruction"
    ANY_TO_ANY = "any_to_any"
    STREAMING = "streaming_realism"
    QAT = "quantization_aware"


@dataclass(frozen=True)
class StageConfig:
    stage: TrainingStage
    loss_weights: Mapping[str, float]
    train_modules: tuple[str, ...] = ()
    gradient_stop_modules: tuple[str, ...] = ()
    adversarial: bool = False
    stateful: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def apply_trainability(self, model: nn.Module) -> None:
        """Set ``requires_grad`` using top-level module names."""
        train_all = not self.train_modules
        for name, parameter in model.named_parameters():
            top_level = name.split(".", 1)[0]
            parameter.requires_grad_(train_all or top_level in self.train_modules)


DEFAULT_STAGES: dict[TrainingStage, StageConfig] = {
    TrainingStage.SEMANTIC: StageConfig(
        TrainingStage.SEMANTIC,
        {"semantic_kl": 1.0, "semantic_ce": 1.0, "semantic_feature": 1.0, "source_adv": 0.1},
        train_modules=("analysis_encoder", "ripple_mixer"),
    ),
    TrainingStage.SPEAKER: StageConfig(
        TrainingStage.SPEAKER,
        {"speaker": 1.0, "cross_crop": 0.5, "token_diversity": 0.05},
        train_modules=("speaker_encoder",),
    ),
    TrainingStage.RECONSTRUCTION: StageConfig(
        TrainingStage.RECONSTRUCTION,
        {"waveform": 1.0, "stft": 1.0, "mel": 1.0, "prosody": 0.5},
        train_modules=("decoder",),
        gradient_stop_modules=("analysis_encoder", "ripple_mixer"),
    ),
    TrainingStage.ANY_TO_ANY: StageConfig(
        TrainingStage.ANY_TO_ANY,
        {"stft": 1.0, "feature_matching": 10.0, "adversarial": 0.1, "speaker": 1.0, "source_rejection": 0.5},
        adversarial=True,
    ),
    TrainingStage.STREAMING: StageConfig(
        TrainingStage.STREAMING,
        {"boundary": 1.0, "state": 0.1, "packet_recovery": 1.0},
        stateful=True,
    ),
    TrainingStage.QAT: StageConfig(
        TrainingStage.QAT,
        {"stft": 1.0, "mel": 1.0, "speaker": 1.0, "state": 0.1},
        stateful=True,
        metadata={"preserve_high_precision": ("norm", "oscillator", "waveform_projection", "prosody")},
    ),
}


def get_stage_config(stage: TrainingStage | str) -> StageConfig:
    return DEFAULT_STAGES[TrainingStage(stage)]
