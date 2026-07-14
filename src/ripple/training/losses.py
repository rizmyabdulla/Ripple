"""Composable, dependency-light losses used by Ripple training stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: Tensor, scale: float) -> Tensor:
        ctx.scale = scale  # type: ignore[attr-defined]
        return value.view_as(value)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.scale * gradient, None  # type: ignore[attr-defined]


def gradient_reversal(value: Tensor, scale: float = 1.0) -> Tensor:
    """Identity in the forward pass and sign-reversed gradient in backward."""
    return _GradientReversal.apply(value, scale)


def _zero_like(value: Tensor) -> Tensor:
    return value.new_zeros(())


def semantic_kl_loss(
    student_logits: Tensor,
    teacher: Tensor,
    *,
    temperature: float = 1.0,
    teacher_is_logits: bool = True,
) -> Tensor:
    """Temperature-scaled KL divergence to a teacher distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = (
        F.softmax(teacher / temperature, dim=-1)
        if teacher_is_logits
        else teacher.clamp_min(torch.finfo(teacher.dtype).tiny)
    )
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * temperature**2


def semantic_ce_loss(student_logits: Tensor, hard_targets: Tensor, ignore_index: int = -100) -> Tensor:
    """Cross entropy accepting arbitrary leading time/batch dimensions."""
    return F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        hard_targets.reshape(-1).long(),
        ignore_index=ignore_index,
    )


def semantic_feature_loss(
    student: Tensor,
    teacher: Tensor,
    *,
    l1_weight: float = 1.0,
    cosine_weight: float = 1.0,
) -> Tensor:
    """Combined feature L1 and cosine regression."""
    if student.shape != teacher.shape:
        raise ValueError(f"feature shapes differ: {student.shape} != {teacher.shape}")
    loss = _zero_like(student)
    if l1_weight:
        loss = loss + l1_weight * F.l1_loss(student, teacher)
    if cosine_weight:
        loss = loss + cosine_weight * (1 - F.cosine_similarity(student, teacher, dim=-1)).mean()
    return loss


def content_invariance_loss(first: Tensor, second: Tensor) -> Tensor:
    """Symmetric normalized feature consistency under timbre perturbations."""
    return F.smooth_l1_loss(F.normalize(first, dim=-1), F.normalize(second, dim=-1))


def contour_correlation_loss(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    prediction = prediction.float()
    target = target.float()
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    correlation = (prediction * target).sum(dim=-1) / (
        prediction.square().sum(dim=-1).sqrt() * target.square().sum(dim=-1).sqrt()
    ).clamp_min(eps)
    return (1 - correlation).mean()


def delta_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape[-1] < 2:
        return _zero_like(prediction)
    return F.smooth_l1_loss(torch.diff(prediction, dim=-1), torch.diff(target, dim=-1))


def prosody_losses(
    *,
    log_f0: Tensor,
    target_log_f0: Tensor,
    voicing_logits: Tensor,
    target_voicing: Tensor,
    periodicity: Tensor | None = None,
    target_periodicity: Tensor | None = None,
    energy: Tensor | None = None,
    target_energy: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return individually weightable prosody objectives."""
    voiced = target_voicing.bool()
    f0 = (
        F.smooth_l1_loss(log_f0[voiced], target_log_f0[voiced])
        if voiced.any()
        else _zero_like(log_f0)
    )
    result = {
        "prosody_f0": f0,
        "prosody_voicing": F.binary_cross_entropy_with_logits(
            voicing_logits, target_voicing.to(voicing_logits.dtype)
        ),
        "prosody_contour": contour_correlation_loss(log_f0, target_log_f0),
        "prosody_delta": delta_loss(log_f0, target_log_f0),
    }
    if periodicity is not None and target_periodicity is not None:
        result["prosody_periodicity"] = F.smooth_l1_loss(periodicity, target_periodicity)
    if energy is not None and target_energy is not None:
        result["prosody_energy"] = F.smooth_l1_loss(energy, target_energy)
    return result


def waveform_l1_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(prediction, target)


def preemphasis(signal: Tensor, coefficient: float = 0.97) -> Tensor:
    return torch.cat((signal[..., :1], signal[..., 1:] - coefficient * signal[..., :-1]), dim=-1)


def _stft_magnitude(audio: Tensor, fft_size: int, hop_size: int, win_length: int) -> Tensor:
    if audio.shape[-1] < win_length:
        audio = F.pad(audio, (0, win_length - audio.shape[-1]))
    window = torch.hann_window(win_length, device=audio.device, dtype=audio.dtype)
    return torch.stft(
        audio.reshape(-1, audio.shape[-1]),
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
    ).abs()


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral-convergence and log-magnitude loss across resolutions."""

    def __init__(
        self,
        resolutions: Sequence[tuple[int, int, int]] = ((512, 120, 480), (1024, 240, 960), (256, 60, 240)),
        eps: float = 1e-7,
        complex_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.eps = eps
        self.complex_weight = complex_weight

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        total = _zero_like(prediction)
        for fft_size, hop_size, win_length in self.resolutions:
            pred_mag = _stft_magnitude(prediction, fft_size, hop_size, win_length)
            target_mag = _stft_magnitude(target, fft_size, hop_size, win_length)
            convergence = torch.linalg.vector_norm(target_mag - pred_mag) / torch.linalg.vector_norm(
                target_mag
            ).clamp_min(self.eps)
            log_magnitude = F.l1_loss(torch.log(pred_mag + self.eps), torch.log(target_mag + self.eps))
            total = total + convergence + log_magnitude
            if self.complex_weight:
                total = total + self.complex_weight * F.l1_loss(pred_mag, target_mag)
        return total / len(self.resolutions)


def _hz_to_mel(value: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + value / 700.0)


def _mel_to_hz(value: Tensor) -> Tensor:
    return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    fft_size: int,
    n_mels: int,
    *,
    f_min: float = 0.0,
    f_max: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a triangular mel filterbank without torchaudio/librosa."""
    f_max = float(sample_rate / 2 if f_max is None else f_max)
    if not 0 <= f_min < f_max <= sample_rate / 2:
        raise ValueError("mel frequency range must lie within Nyquist")
    bounds = torch.linspace(
        _hz_to_mel(torch.tensor(f_min)).item(),
        _hz_to_mel(torch.tensor(f_max)).item(),
        n_mels + 2,
        device=device,
        dtype=dtype,
    )
    bins = torch.floor((fft_size + 1) * _mel_to_hz(bounds) / sample_rate).long()
    filters = torch.zeros(n_mels, fft_size // 2 + 1, device=device, dtype=dtype)
    for index in range(n_mels):
        left, center, right = (int(item) for item in bins[index : index + 3])
        center = max(center, left + 1)
        right = max(right, center + 1)
        filters[index, left:center] = torch.linspace(0, 1, center - left, device=device, dtype=dtype)
        filters[index, center:right] = torch.linspace(1, 0, right - center, device=device, dtype=dtype)
    return filters


class MultiScaleMelLoss(nn.Module):
    """Log-mel reconstruction loss over configurable FFT scales."""

    def __init__(
        self,
        sample_rate: int = 24_000,
        fft_sizes: Sequence[int] = (256, 512, 1024),
        n_mels: int = 80,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_sizes = tuple(fft_sizes)
        self.n_mels = n_mels
        self.eps = eps

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        total = _zero_like(prediction)
        for fft_size in self.fft_sizes:
            hop = fft_size // 4
            pred = _stft_magnitude(prediction, fft_size, hop, fft_size)
            truth = _stft_magnitude(target, fft_size, hop, fft_size)
            bank = mel_filterbank(
                self.sample_rate,
                fft_size,
                self.n_mels,
                device=pred.device,
                dtype=pred.dtype,
            )
            pred_mel = torch.einsum("mf,bft->bmt", bank, pred)
            target_mel = torch.einsum("mf,bft->bmt", bank, truth)
            total = total + F.l1_loss(torch.log(pred_mel + self.eps), torch.log(target_mel + self.eps))
        return total / len(self.fft_sizes)


def speaker_embedding_loss(converted: Tensor, target: Tensor) -> Tensor:
    return (1 - F.cosine_similarity(converted, target, dim=-1)).mean()


def source_rejection_loss(converted: Tensor, source: Tensor, margin: float = 0.2) -> Tensor:
    similarity = F.cosine_similarity(converted, source, dim=-1)
    return F.relu(similarity - margin).mean()


def token_diversity_loss(tokens: Tensor) -> Tensor:
    """Penalize off-diagonal cosine similarity between style tokens."""
    normalized = F.normalize(tokens, dim=-1)
    gram = normalized @ normalized.transpose(-1, -2)
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    return ((gram - identity) ** 2).mean()


def boundary_waveform_loss(left_tail: Tensor, right_head: Tensor) -> Tensor:
    return F.l1_loss(left_tail, right_head)


def feature_continuity_loss(left: Tensor, right: Tensor) -> Tensor:
    return F.smooth_l1_loss(left[..., -1, :], right[..., 0, :])


def state_reset_equivalence_loss(reset_output: Tensor, fresh_output: Tensor) -> Tensor:
    return F.mse_loss(reset_output, fresh_output)


def state_norm_loss(state: Tensor, max_norm: float = 10.0) -> Tensor:
    norms = torch.linalg.vector_norm(state.float(), dim=-1)
    return F.relu(norms - max_norm).square().mean()


def packet_recovery_loss(
    recovered: Tensor,
    target: Tensor,
    packet_mask: Tensor | None = None,
) -> Tensor:
    """Reconstruction loss focused on dropped or recovering packet regions."""
    error = torch.abs(recovered - target)
    if packet_mask is None:
        return error.mean()
    mask = packet_mask.to(error.dtype)
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(1)
    return (error * mask).sum() / mask.expand_as(error).sum().clamp_min(1)


def phase_continuity_loss(previous_phase: Tensor, current_phase: Tensor) -> Tensor:
    """Circular phase discontinuity in radians."""
    difference = torch.atan2(torch.sin(current_phase - previous_phase), torch.cos(current_phase - previous_phase))
    return difference.abs().mean()


def discriminator_hinge_loss(real_scores: Sequence[Tensor], fake_scores: Sequence[Tensor]) -> Tensor:
    if len(real_scores) != len(fake_scores) or not real_scores:
        raise ValueError("real and fake score collections must be non-empty and aligned")
    return sum((F.relu(1 - real).mean() + F.relu(1 + fake).mean()) for real, fake in zip(real_scores, fake_scores)) / len(real_scores)


def generator_adversarial_loss(fake_scores: Sequence[Tensor]) -> Tensor:
    if not fake_scores:
        raise ValueError("fake_scores cannot be empty")
    return sum(-score.mean() for score in fake_scores) / len(fake_scores)


def feature_matching_loss(
    real_features: Sequence[Sequence[Tensor]],
    fake_features: Sequence[Sequence[Tensor]],
) -> Tensor:
    terms = [
        F.l1_loss(fake, real.detach())
        for real_group, fake_group in zip(real_features, fake_features)
        for real, fake in zip(real_group, fake_group)
    ]
    if not terms:
        raise ValueError("feature collections cannot be empty")
    return torch.stack(terms).mean()


@dataclass(frozen=True)
class LossValue:
    name: str
    raw: Tensor
    weight: float

    @property
    def weighted(self) -> Tensor:
        return self.raw * self.weight


class LossComposer:
    """Compose named loss callables while preserving raw values for logging."""

    def __init__(
        self,
        terms: Mapping[str, Callable[[Mapping[str, Tensor]], Tensor]],
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.terms = dict(terms)
        self.weights = dict(weights or {})

    def __call__(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, LossValue]]:
        values = {
            name: LossValue(name, function(batch), float(self.weights.get(name, 1.0)))
            for name, function in self.terms.items()
            if self.weights.get(name, 1.0) != 0
        }
        if not values:
            reference = next(iter(batch.values()), None)
            if reference is None:
                raise ValueError("cannot infer device for an empty loss composition")
            return _zero_like(reference), {}
        total = torch.stack([value.weighted for value in values.values()]).sum()
        return total, values
