from __future__ import annotations

import torch
from torch.nn import functional as F

from ripple.baselines.streamvc import (
    YIN_THRESHOLD,
    StreamVCBaseline,
    StreamVCConfig,
)
from ripple.models.ripple_mixer import RippleMixer, RippleMixerConfig
from ripple.models.ripple_vc import RippleVC


def test_mixer_modes_match_streaming_reference() -> None:
    torch.manual_seed(2)
    for mode in ("pure_conv", "local_attn"):
        mixer = RippleMixer(
            RippleMixerConfig(
                channels=16,
                blocks=3,
                kernel_size=5,
                mode=mode,
                attention_heads=4,
                attention_window=6,
            )
        ).eval()
        x = torch.randn(1, 16, 11)
        expected = mixer(x)
        state = mixer.initial_state(1)
        outputs = []
        for frame in x.split(1, dim=-1):
            output, state = mixer.step(frame, state)
            outputs.append(output)
        torch.testing.assert_close(
            torch.cat(outputs, dim=-1),
            expected,
            atol=2e-6,
            rtol=2e-5,
        )


def test_ripple_full_and_step_are_equivalent() -> None:
    torch.manual_seed(3)
    model = RippleVC().eval()
    source = torch.randn(1, 1, model.chunk_samples * 2) * 0.05
    profile = model.enroll(torch.randn(1, 1, 2_400) * 0.05)
    with torch.no_grad():
        expected = model(source, profile)
        state = model.initial_state(1)
        outputs = []
        for frame in source.split(model.chunk_samples, dim=-1):
            output, state = model.step(frame, profile, state)
            outputs.append(output.waveform)
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1),
        expected.waveform,
        atol=1e-6,
        rtol=1e-5,
    )


def test_streamvc_uses_correct_yin_threshold_and_raw_logits_for_ce() -> None:
    config = StreamVCConfig()
    assert config.yin_threshold == YIN_THRESHOLD == 0.15
    model = StreamVCBaseline(config)
    logits = torch.tensor([[[2.0], [-1.0], [0.5]]])
    target = torch.tensor([[0]])
    loss = model.content_loss(logits, target)
    expected = F.cross_entropy(logits[:, :, 0], target[:, 0])
    torch.testing.assert_close(loss, expected)
    wrong = F.cross_entropy(
        torch.softmax(logits[:, :, 0], dim=1), target[:, 0]
    )
    assert not torch.isclose(loss, wrong)


def test_streamvc_state_is_explicit() -> None:
    model = StreamVCBaseline().eval()
    reference = torch.randn(1, 1, 1_600) * 0.05
    profile = model.enroll(reference)
    state = model.initial_state(1)
    output, next_state = model.step(
        torch.zeros(1, 1, model.config.chunk_samples), profile, state
    )
    assert output.waveform.shape[-1] == model.config.chunk_samples
    assert next_state is not state
