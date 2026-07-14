"""Text-to-RIF semantic and prosody frontend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .duration import DurationOutput, MonotonicDurationPredictor, length_regulate


@dataclass(frozen=True)
class TextRIF:
    """RIF-1-compatible tensors generated from text."""

    semantic_soft: Tensor
    semantic_embed: Tensor
    prosody: Tensor
    padding_mask: Tensor
    durations: Tensor
    frame_rate_hz: int = 50
    schema_version: str = "RIF-1"

    def validate(self) -> None:
        if self.semantic_soft.ndim != 3:
            raise ValueError("semantic_soft must have shape [B, F, classes]")
        if self.semantic_embed.shape[:2] != self.semantic_soft.shape[:2]:
            raise ValueError("semantic frame counts do not match")
        if self.prosody.shape[:2] != self.semantic_soft.shape[:2]:
            raise ValueError("prosody frame count does not match semantics")
        if self.padding_mask.shape != self.semantic_soft.shape[:2]:
            raise ValueError("padding mask does not match RIF frames")
        probabilities = self.semantic_soft.sum(dim=-1)
        valid = ~self.padding_mask
        if valid.any() and not torch.allclose(
            probabilities[valid],
            torch.ones_like(probabilities[valid]),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("semantic_soft must be normalized")


class TextToRIFFrontend(nn.Module):
    """Compact multilingual encoder that predicts the frozen RIF-1 contract."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        hidden_dim: int = 192,
        semantic_classes: int = 256,
        semantic_dim: int = 128,
        prosody_dim: int = 8,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        padding_id: int = 0,
        max_positions: int = 4096,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 0 or not 256 <= semantic_classes <= 512:
            raise ValueError("RIF-1 requires 256 to 512 semantic classes")
        if semantic_dim != 128:
            raise ValueError("RIF-1 requires 128-dimensional semantic embeddings")
        if prosody_dim != 8:
            raise ValueError("RIF-1 requires eight fixed prosody fields")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.padding_id = padding_id
        self.max_positions = max_positions
        self.token_embedding = nn.Embedding(
            vocabulary_size, hidden_dim, padding_idx=padding_id
        )
        self.position_embedding = nn.Embedding(max_positions, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.duration_predictor = MonotonicDurationPredictor(
            hidden_dim, channels=hidden_dim, dropout=dropout
        )
        self.semantic_projection = nn.Linear(hidden_dim, semantic_classes)
        self.semantic_embeddings = nn.Parameter(
            torch.empty(semantic_classes, semantic_dim)
        )
        self.prosody_projection = nn.Linear(hidden_dim, prosody_dim)
        nn.init.normal_(self.semantic_embeddings, mean=0.0, std=semantic_dim**-0.5)

    def forward(
        self,
        token_ids: Tensor,
        *,
        durations: Tensor | None = None,
        pace: float = 1.0,
        semantic_temperature: float = 1.0,
    ) -> TextRIF:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        if token_ids.shape[1] > self.max_positions:
            raise ValueError("token sequence exceeds max_positions")
        if semantic_temperature <= 0:
            raise ValueError("semantic_temperature must be positive")
        token_padding = token_ids.eq(self.padding_id)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        encoded = self.token_embedding(token_ids) + self.position_embedding(positions)
        encoded = self.encoder(encoded, src_key_padding_mask=token_padding)
        if durations is None:
            duration_output: DurationOutput = self.duration_predictor(
                encoded, token_padding, pace=pace
            )
            durations = duration_output.durations
        elif durations.shape != token_ids.shape:
            raise ValueError("durations must match token_ids")
        frame_states, frame_padding = length_regulate(encoded, durations)
        logits = self.semantic_projection(frame_states) / semantic_temperature
        semantic_soft = torch.softmax(logits, dim=-1)
        semantic_embed = semantic_soft @ self.semantic_embeddings
        raw_prosody = self.prosody_projection(frame_states)
        # RIF prosody: normalized log-F0, voiced probability, confidence,
        # normalized log-energy and four first-order deltas.
        prosody_parts = [raw_prosody[..., :1]]
        if raw_prosody.shape[-1] >= 2:
            prosody_parts.append(raw_prosody[..., 1:2].sigmoid())
        if raw_prosody.shape[-1] >= 3:
            prosody_parts.append(raw_prosody[..., 2:3].sigmoid())
        if raw_prosody.shape[-1] > 3:
            prosody_parts.append(raw_prosody[..., 3:])
        prosody = torch.cat(prosody_parts, dim=-1)
        result = TextRIF(
            semantic_soft,
            semantic_embed,
            prosody,
            frame_padding,
            durations,
        )
        result.validate()
        return result
