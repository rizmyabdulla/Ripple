"""Explicit-state causal convolution primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Conv1dState:
    """Left-context samples retained by a causal convolution."""

    buffer: Tensor
    stride_phase: Tensor

    def detach(self) -> "Conv1dState":
        return Conv1dState(self.buffer.detach(), self.stride_phase.detach())


class CachedCausalConv1d(nn.Conv1d):
    """Conv1d with left-only padding and an explicit bounded cache.

    ``stride_phase`` preserves the global output lattice, so even input calls
    that are not individually divisible by the stride match full execution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        if kernel_size < 1 or stride < 1 or dilation < 1:
            raise ValueError("kernel_size, stride and dilation must be positive")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    @property
    def context_size(self) -> int:
        return self.dilation[0] * (self.kernel_size[0] - 1)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Conv1dState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        reference = self.weight
        buffer = torch.zeros(
                batch_size,
                self.in_channels,
                self.context_size,
                device=device if device is not None else reference.device,
                dtype=dtype if dtype is not None else reference.dtype,
            )
        return Conv1dState(
            buffer,
            torch.zeros((), device=buffer.device, dtype=torch.int64),
        )

    def forward(self, x: Tensor) -> Tensor:
        self._validate_input(x)
        if self.context_size:
            x = F.pad(x, (self.context_size, 0))
        return self._convolve(x)

    def step(self, x: Tensor, state: Conv1dState) -> tuple[Tensor, Conv1dState]:
        self._validate_input(x)
        expected = (x.shape[0], self.in_channels, self.context_size)
        if tuple(state.buffer.shape) != expected:
            raise ValueError(
                f"invalid convolution state shape {tuple(state.buffer.shape)}; "
                f"expected {expected}"
            )
        phase = int(state.stride_phase.item())
        stride = self.stride[0]
        if not 0 <= phase < stride:
            raise ValueError("convolution stride phase is out of bounds")
        joined = torch.cat((state.buffer, x), dim=-1)
        offset = (-phase) % stride
        convolution_input = joined[..., offset:]
        effective_kernel = self.context_size + 1
        if convolution_input.shape[-1] < effective_kernel:
            y = x.new_empty(x.shape[0], self.out_channels, 0)
        else:
            y = self._convolve(convolution_input)
        if self.context_size:
            new_buffer = joined[..., -self.context_size :]
        else:
            new_buffer = joined[..., :0]
        next_phase = torch.tensor(
            (phase + x.shape[-1]) % stride,
            device=x.device,
            dtype=torch.int64,
        )
        return y, Conv1dState(new_buffer, next_phase)

    def _convolve(self, x: Tensor) -> Tensor:
        return F.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError("expected input shaped [batch, channels, time]")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )


def stream_conv(
    module: CachedCausalConv1d, x: Tensor, chunk_size: int
) -> Tensor:
    """Reference helper used by tests and export validation."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    state = module.initial_state(x.shape[0], device=x.device, dtype=x.dtype)
    outputs: list[Tensor] = []
    for chunk in x.split(chunk_size, dim=-1):
        y, state = module.step(chunk, state)
        outputs.append(y)
    return torch.cat(outputs, dim=-1)


CachedConv1d = CachedCausalConv1d
CausalConv1d = CachedCausalConv1d
