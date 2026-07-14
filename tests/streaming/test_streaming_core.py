from __future__ import annotations

import itertools

import torch

from ripple.models.source_filter import HarmonicNoiseSource
from ripple.streaming.cached_conv import CachedCausalConv1d
from ripple.streaming.local_attention import CausalLocalAttention
from ripple.streaming.packet_loss import (
    PacketLossPolicy,
    PacketLossState,
    PacketStatus,
)
from ripple.streaming.running_stats import WelfordRunningStats


def test_cached_causal_conv_matches_arbitrary_stream_grouping() -> None:
    torch.manual_seed(0)
    convolution = CachedCausalConv1d(3, 5, 7, stride=2)
    x = torch.randn(2, 3, 48)
    expected = convolution(x)
    state = convolution.initial_state(2)
    outputs = []
    offsets = (0, 1, 8, 13, 30, 48)
    for start, end in itertools.pairwise(offsets):
        chunk = x[..., start:end]
        output, state = convolution.step(chunk, state)
        outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, dim=-1), expected)
    assert state.buffer.shape[-1] == convolution.context_size


def test_local_attention_cache_is_bounded_and_equivalent() -> None:
    torch.manual_seed(1)
    attention = CausalLocalAttention(16, num_heads=4, window=5).eval()
    x = torch.randn(2, 16, 13)
    expected = attention(x)
    state = attention.initial_state(2)
    outputs = []
    for chunk in (x[..., :3], x[..., 3:8], x[..., 8:]):
        output, state = attention.step(chunk, state)
        outputs.append(output)
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1), expected, atol=1e-6, rtol=1e-5
    )
    assert state.key.shape[2] == 4
    assert int(state.length) == 4


def test_welford_running_statistics_match_batch_values() -> None:
    stats = WelfordRunningStats(
        1, prior_count=0.0, max_count=None
    )
    state = stats.initial_state(1)
    values = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    state = stats.update(values, state)
    torch.testing.assert_close(state.mean, torch.tensor([[2.5]]))
    torch.testing.assert_close(stats.variance(state), torch.tensor([[5.0 / 3.0]]))


def test_oscillator_phase_matches_frame_grouping() -> None:
    source = HarmonicNoiseSource()
    f0 = torch.tensor([[[120.0, 130.0, 140.0]]])
    voiced = torch.full_like(f0, 0.9)
    periodicity = torch.full_like(f0, 0.8)
    expected = source(f0, voiced, periodicity)
    state = source.initial_state(1)
    outputs = []
    for index in range(f0.shape[-1]):
        output, state = source.step(
            f0[..., index : index + 1],
            voiced[..., index : index + 1],
            periodicity[..., index : index + 1],
            state,
        )
        outputs.append(output)
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1), expected, atol=2e-5, rtol=2e-5
    )
    assert int(state.sample_index.item()) == 3 * source.hop_samples


def test_packet_loss_policy_repeats_then_uses_comfort_noise() -> None:
    policy = PacketLossPolicy()
    previous = torch.ones(1, 1, 480)
    first = policy.apply(
        None, PacketStatus.MISSING, PacketLossState(), previous
    )
    torch.testing.assert_close(
        first.pcm, previous * policy.config.repeat_attenuation
    )
    second = policy.apply(
        None, PacketStatus.MISSING, first.state, previous
    )
    assert second.pcm is not None
    assert second.pcm.abs().max() <= policy.config.comfort_noise_level + 1e-7
    assert second.freeze_statistics
