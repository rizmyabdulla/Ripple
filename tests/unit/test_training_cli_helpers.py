from __future__ import annotations

import torch

from ripple.data.split import assign_speaker_splits
from ripple.training.batching import collate_waveforms
from ripple.training.factory import build_model
from ripple.training.stages import TrainingStage
from ripple.training.steps import ReconstructionStep, build_step_function


def test_assign_speaker_splits_are_disjoint() -> None:
    assignment = assign_speaker_splits(
        ["a", "b", "c", "d", "e", "f"],
        train_ratio=0.5,
        development_ratio=0.25,
        seed=0,
    )
    assert len(assignment) == 6
    assert len(set(assignment.values())) >= 2


def test_reconstruction_step_runs_on_tiny_batch() -> None:
    model = build_model("ripple")
    waveform = torch.zeros(1, 1, 4800)
    batch = {"waveform": waveform}
    loss, metrics = ReconstructionStep()(model, batch)
    assert torch.isfinite(loss)
    assert "loss" in metrics
    assert callable(build_step_function(TrainingStage.RECONSTRUCTION))


def test_semantic_step_requires_teacher_and_runs() -> None:
    model = build_model("ripple")
    waveform = torch.zeros(1, 1, 4800)
    step = build_step_function(TrainingStage.SEMANTIC)
    try:
        step(model, {"waveform": waveform})
        raise AssertionError("expected ValueError without teacher")
    except ValueError:
        pass
    # Fake teacher: [B, T, C] — length chosen loosely; step truncates to student frames.
    teacher = torch.zeros(1, 50, 64)
    loss, metrics = step(model, {"waveform": waveform, "teacher": teacher})
    assert torch.isfinite(loss)
    assert "semantic_feature" in metrics


def test_collate_waveforms_stacks_batch() -> None:
    batch = collate_waveforms(
        [
            {
                "record_id": "a",
                "speaker_id": "s",
                "language": "en",
                "waveform": torch.zeros(480),
                "sample_rate": 24_000,
            },
            {
                "record_id": "b",
                "speaker_id": "s",
                "language": "en",
                "waveform": torch.ones(480),
                "sample_rate": 24_000,
            },
        ]
    )
    assert batch["waveform"].shape == (2, 1, 480)
