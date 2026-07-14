from __future__ import annotations

import torch

from ripple.training.losses import (
    LossComposer,
    content_invariance_loss,
    discriminator_hinge_loss,
    generator_adversarial_loss,
    gradient_reversal,
    packet_recovery_loss,
    prosody_losses,
    semantic_ce_loss,
    semantic_feature_loss,
    semantic_kl_loss,
    speaker_embedding_loss,
    state_norm_loss,
    token_diversity_loss,
)


def test_semantic_losses_are_finite_and_differentiable() -> None:
    torch.manual_seed(0)
    student = torch.randn(2, 5, 8, requires_grad=True)
    teacher = torch.randn_like(student)
    labels = teacher.argmax(dim=-1)
    loss = (
        semantic_kl_loss(student, teacher, temperature=2)
        + semantic_ce_loss(student, labels)
        + semantic_feature_loss(student, teacher)
        + content_invariance_loss(student, teacher)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_prosody_losses_handle_an_unvoiced_batch() -> None:
    shape = (2, 10)
    values = prosody_losses(
        log_f0=torch.zeros(shape, requires_grad=True),
        target_log_f0=torch.zeros(shape),
        voicing_logits=torch.zeros(shape, requires_grad=True),
        target_voicing=torch.zeros(shape),
        periodicity=torch.zeros(shape),
        target_periodicity=torch.zeros(shape),
        energy=torch.ones(shape),
        target_energy=torch.ones(shape),
    )
    assert values["prosody_f0"].item() == 0
    assert all(torch.isfinite(value) for value in values.values())


def test_speaker_state_and_adversarial_scaffolds() -> None:
    target = torch.eye(4)
    converted = target.clone()
    tokens = target.unsqueeze(0)
    assert speaker_embedding_loss(converted, target).item() == 0
    assert token_diversity_loss(tokens).item() == 0
    assert state_norm_loss(torch.zeros(2, 4)).item() == 0
    discriminator = discriminator_hinge_loss([torch.ones(2)], [-torch.ones(2)])
    generator = generator_adversarial_loss([torch.zeros(2)])
    assert discriminator.item() == 0
    assert generator.item() == 0
    features = torch.ones(2, requires_grad=True)
    gradient_reversal(features, 0.5).sum().backward()
    assert torch.equal(features.grad, torch.full_like(features, -0.5))
    assert packet_recovery_loss(
        torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])
    ).item() == 1


def test_loss_composer_keeps_raw_and_weighted_values() -> None:
    prediction = torch.tensor([2.0], requires_grad=True)
    composer = LossComposer(
        {
            "first": lambda batch: batch["prediction"].square().mean(),
            "second": lambda batch: batch["prediction"].abs().mean(),
        },
        {"first": 0.5, "second": 2.0},
    )
    total, values = composer({"prediction": prediction})
    assert total.item() == 6.0
    assert values["first"].raw.item() == 4.0
    total.backward()
    assert prediction.grad is not None
