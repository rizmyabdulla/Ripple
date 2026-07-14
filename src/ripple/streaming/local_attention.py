"""Bounded-cache causal local attention."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LocalAttentionState:
    key: Tensor
    value: Tensor
    length: Tensor

    def detach(self) -> LocalAttentionState:
        return LocalAttentionState(
            self.key.detach(), self.value.detach(), self.length.detach()
        )


class CausalLocalAttention(nn.Module):
    """Multi-head attention over the current and previous ``window`` frames."""

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 4,
        window: int = 32,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if window < 1:
            raise ValueError("window must be positive")
        self.channels = channels
        self.num_heads = num_heads
        self.window = window
        self.head_dim = channels // num_heads
        self.q_proj = nn.Conv1d(channels, channels, 1, bias=bias)
        self.k_proj = nn.Conv1d(channels, channels, 1, bias=bias)
        self.v_proj = nn.Conv1d(channels, channels, 1, bias=bias)
        self.out_proj = nn.Conv1d(channels, channels, 1, bias=bias)

    @property
    def cache_size(self) -> int:
        return self.window - 1

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> LocalAttentionState:
        reference = self.q_proj.weight
        shape = (batch_size, self.num_heads, self.cache_size, self.head_dim)
        key = torch.zeros(
            shape,
            device=device if device is not None else reference.device,
            dtype=dtype if dtype is not None else reference.dtype,
        )
        return LocalAttentionState(
            key=key,
            value=torch.zeros_like(key),
            length=torch.zeros((), device=key.device, dtype=torch.int64),
        )

    def forward(self, x: Tensor) -> Tensor:
        state = self.initial_state(x.shape[0], device=x.device, dtype=x.dtype)
        y, _ = self.step(x, state)
        return y

    def step(
        self, x: Tensor, state: LocalAttentionState
    ) -> tuple[Tensor, LocalAttentionState]:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(
                f"expected [batch, {self.channels}, time], got {tuple(x.shape)}"
            )
        self._validate_state(x, state)
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        cache_len = int(state.length.item())
        old_k = state.key[:, :, self.cache_size - cache_len :, :]
        old_v = state.value[:, :, self.cache_size - cache_len :, :]
        all_k = torch.cat((old_k, k), dim=2)
        all_v = torch.cat((old_v, v), dim=2)

        frames: list[Tensor] = []
        scale = 1.0 / sqrt(self.head_dim)
        for index in range(x.shape[-1]):
            key_end = cache_len + index + 1
            key_start = max(0, key_end - self.window)
            keys = all_k[:, :, key_start:key_end, :]
            values = all_v[:, :, key_start:key_end, :]
            query = q[:, :, index : index + 1, :]
            scores = torch.matmul(query, keys.transpose(-1, -2)) * scale
            weights = torch.softmax(scores, dim=-1)
            frames.append(torch.matmul(weights, values))
        attended = torch.cat(frames, dim=2)

        retained_k = all_k[:, :, -self.cache_size :, :] if self.cache_size else all_k[:, :, :0, :]
        retained_v = all_v[:, :, -self.cache_size :, :] if self.cache_size else all_v[:, :, :0, :]
        new_len = min(self.cache_size, cache_len + x.shape[-1])
        if retained_k.shape[2] < self.cache_size:
            pad = self.cache_size - retained_k.shape[2]
            retained_k = torch.nn.functional.pad(retained_k, (0, 0, pad, 0))
            retained_v = torch.nn.functional.pad(retained_v, (0, 0, pad, 0))
        new_state = LocalAttentionState(
            retained_k,
            retained_v,
            torch.tensor(new_len, device=x.device, dtype=torch.int64),
        )
        return self.out_proj(self._merge_heads(attended)), new_state

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, _, time = x.shape
        return (
            x.view(batch, self.num_heads, self.head_dim, time)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, time, _ = x.shape
        return (
            x.permute(0, 1, 3, 2)
            .contiguous()
            .view(batch, self.channels, time)
        )

    def _validate_state(self, x: Tensor, state: LocalAttentionState) -> None:
        expected = (x.shape[0], self.num_heads, self.cache_size, self.head_dim)
        if tuple(state.key.shape) != expected or tuple(state.value.shape) != expected:
            raise ValueError(f"attention cache must have shape {expected}")
        length = int(state.length.item())
        if not 0 <= length <= self.cache_size:
            raise ValueError("attention cache length is out of bounds")
