from __future__ import annotations

import torch

from ripple.training.losses import MultiResolutionSTFTLoss, MultiScaleMelLoss, mel_filterbank, preemphasis


def test_spectral_losses_are_zeroish_for_identical_audio() -> None:
    torch.manual_seed(1)
    audio = torch.randn(2, 640)
    stft = MultiResolutionSTFTLoss(resolutions=((128, 32, 128), (64, 16, 64)))
    mel = MultiScaleMelLoss(sample_rate=24_000, fft_sizes=(64, 128), n_mels=16)
    assert stft(audio, audio).item() < 1e-6
    assert mel(audio, audio).item() < 1e-6


def test_spectral_losses_propagate_gradients_on_short_audio() -> None:
    prediction = torch.randn(1, 31, requires_grad=True)
    target = torch.randn(1, 31)
    loss = MultiResolutionSTFTLoss(resolutions=((64, 16, 64),))(prediction, target)
    loss = loss + MultiScaleMelLoss(fft_sizes=(64,), n_mels=8)(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_filterbank_and_preemphasis_shapes() -> None:
    bank = mel_filterbank(24_000, 128, 20)
    assert bank.shape == (20, 65)
    assert torch.all(bank >= 0)
    audio = torch.randn(3, 100)
    assert preemphasis(audio).shape == audio.shape
