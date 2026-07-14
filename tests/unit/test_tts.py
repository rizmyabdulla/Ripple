from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ripple.teachers import HuBERTTeacher
from ripple.tts import (
    IncrementalChunkScheduler,
    MultilingualTextNormalizer,
    MultilingualTokenizer,
    TextRIF,
    TextToRIFFrontend,
)


def test_multilingual_normalization_and_byte_fallback() -> None:
    normalizer = MultilingualTextNormalizer()
    assert normalizer.normalize("  Café\t世界  ", "FR_ca").text == "Café 世界"
    tokens = MultilingualTokenizer().encode("مرحبا 🌍", language="ar")
    assert tokens.language == "ar"
    assert tokens.ids[0] == 1
    assert tokens.ids[-1] == 2
    assert all(4 <= token_id < 260 for token_id in tokens.ids[1:-1])


def test_text_frontend_emits_rif_one_shapes() -> None:
    model = TextToRIFFrontend(
        vocabulary_size=300,
        hidden_dim=32,
        semantic_classes=256,
        layers=1,
        heads=4,
        dropout=0.0,
    ).eval()
    ids = torch.tensor([[1, 20, 21, 2]])
    durations = torch.tensor([[1, 2, 1, 1]])
    with torch.inference_mode():
        rif = model(ids, durations=durations)
    assert rif.semantic_soft.shape == (1, 5, 256)
    assert rif.semantic_embed.shape == (1, 5, 128)
    assert rif.prosody.shape == (1, 5, 8)
    assert torch.allclose(rif.semantic_soft.sum(-1), torch.ones(1, 5))


def _rif(frames: int) -> TextRIF:
    probabilities = torch.full((1, frames, 256), 1 / 256)
    return TextRIF(
        probabilities,
        torch.zeros(1, frames, 128),
        torch.zeros(1, frames, 8),
        torch.zeros(1, frames, dtype=torch.bool),
        torch.ones(1, frames, dtype=torch.long),
    )


class _FrozenDecoder(nn.Module):
    rif_schema_version = "RIF-1"

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)

    def decode_step(
        self, semantic_embed: torch.Tensor, prosody: torch.Tensor, state: int
    ) -> tuple[torch.Tensor, int]:
        del prosody
        return semantic_embed[..., 0], state + semantic_embed.shape[1]


def test_incremental_scheduler_never_revises_committed_frames() -> None:
    scheduler = IncrementalChunkScheduler(chunk_frames=3, lookahead_frames=2)
    scheduler.replace_uncommitted(_rif(8))
    first = list(scheduler.chunks())
    assert [chunk.frames for chunk in first] == [3, 3]
    scheduler.replace_uncommitted(_rif(10))
    packets, state = scheduler.decode(_FrozenDecoder(), 0, final=True)
    assert sum(packet.shape[1] for packet in packets) == 4
    assert state == 4
    assert scheduler.committed_frames == 10


def test_scheduler_rejects_trainable_decoder() -> None:
    decoder = _FrozenDecoder()
    decoder.weight.requires_grad_(True)
    scheduler = IncrementalChunkScheduler()
    scheduler.replace_uncommitted(_rif(2))
    with pytest.raises(ValueError, match="must be frozen"):
        scheduler.decode(decoder, None, final=True)


class _TeacherBackend:
    def eval(self) -> _TeacherBackend:
        return self

    def to(self, device: str) -> _TeacherBackend:
        del device
        return self

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        waveform = kwargs["input_values"]
        assert isinstance(waveform, torch.Tensor)
        return SimpleNamespace(
            last_hidden_state=torch.zeros(waveform.shape[0], 4, 12)
        )


def test_teacher_adapter_is_lazy_and_supports_injected_backend() -> None:
    lazy = HuBERTTeacher()
    assert not lazy.loaded
    with pytest.raises(ValueError, match="automatic downloads are disabled"):
        lazy.extract(torch.zeros(16_000), 16_000)
    injected = HuBERTTeacher(backend=_TeacherBackend())
    result = injected.extract(torch.zeros(16_000), 16_000)
    assert result.values.shape == (1, 4, 12)
    assert result.teacher == "hubert"
