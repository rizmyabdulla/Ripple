"""Incremental RIF chunk scheduling and frozen-decoder enforcement."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from .frontend import TextRIF
from .normalization import is_safe_boundary


@runtime_checkable
class FrozenRIFDecoder(Protocol):
    """Minimum shared decoder contract consumed by Ripple-TTS."""

    rif_schema_version: str

    def decode_step(
        self, semantic_embed: Tensor, prosody: Tensor, state: Any
    ) -> tuple[Tensor, Any]: ...


@dataclass(frozen=True)
class RIFChunk:
    semantic_soft: Tensor
    semantic_embed: Tensor
    prosody: Tensor
    start_frame: int
    final: bool
    schema_version: str = "RIF-1"

    @property
    def frames(self) -> int:
        return int(self.semantic_embed.shape[1])


def require_frozen_decoder(decoder: FrozenRIFDecoder) -> None:
    """Reject decoder drift at the first TTS integration boundary."""

    if not isinstance(decoder, FrozenRIFDecoder):
        raise TypeError("decoder does not implement the FrozenRIFDecoder contract")
    if decoder.rif_schema_version != "RIF-1":
        raise ValueError("Ripple-TTS milestone one requires decoder RIF-1")
    if isinstance(decoder, nn.Module):
        trainable = [name for name, value in decoder.named_parameters() if value.requires_grad]
        if trainable:
            raise ValueError(
                "shared decoder must be frozen; trainable parameters: "
                + ", ".join(trainable[:5])
            )


class IncrementalChunkScheduler:
    """Commit immutable RIF chunks while allowing future plan replacement."""

    def __init__(
        self,
        *,
        chunk_frames: int = 10,
        lookahead_frames: int = 5,
        max_buffered_characters: int = 512,
    ) -> None:
        if chunk_frames <= 0 or lookahead_frames < 0:
            raise ValueError("invalid frame scheduling limits")
        self.chunk_frames = chunk_frames
        self.lookahead_frames = lookahead_frames
        self.max_buffered_characters = max_buffered_characters
        self.committed_frames = 0
        self._text_buffer = ""
        self._plan: TextRIF | None = None

    def push_text(self, text: str, *, final: bool = False) -> str | None:
        """Return text ready for planning at a safe boundary."""

        self._text_buffer += text
        ready = final or is_safe_boundary(self._text_buffer)
        if len(self._text_buffer) >= self.max_buffered_characters:
            ready = True
        if not ready:
            return None
        segment, self._text_buffer = self._text_buffer, ""
        return segment

    def replace_uncommitted(self, plan: TextRIF) -> None:
        """Install or revise only frames that have not been emitted."""

        plan.validate()
        if plan.semantic_embed.shape[0] != 1:
            raise ValueError("incremental scheduling currently requires batch size one")
        if self.committed_frames > plan.semantic_embed.shape[1]:
            raise ValueError("new plan cannot remove committed frames")
        self._plan = plan

    def chunks(self, *, final: bool = False) -> Iterator[RIFChunk]:
        if self._plan is None:
            return
        total_frames = self._plan.semantic_embed.shape[1]
        commit_limit = total_frames if final else max(
            self.committed_frames, total_frames - self.lookahead_frames
        )
        while self.committed_frames < commit_limit:
            stop = min(self.committed_frames + self.chunk_frames, commit_limit)
            frame_slice = slice(self.committed_frames, stop)
            yield RIFChunk(
                self._plan.semantic_soft[:, frame_slice],
                self._plan.semantic_embed[:, frame_slice],
                self._plan.prosody[:, frame_slice],
                self.committed_frames,
                final=final and stop == total_frames,
                schema_version=self._plan.schema_version,
            )
            self.committed_frames = stop

    def decode(
        self,
        decoder: FrozenRIFDecoder,
        state: Any,
        *,
        final: bool = False,
    ) -> tuple[list[Tensor], Any]:
        require_frozen_decoder(decoder)
        packets: list[Tensor] = []
        with torch.inference_mode():
            for chunk in self.chunks(final=final):
                packet, state = decoder.decode_step(
                    chunk.semantic_embed, chunk.prosody, state
                )
                packets.append(packet)
        return packets, state

    def reset(self) -> None:
        self.committed_frames = 0
        self._text_buffer = ""
        self._plan = None
