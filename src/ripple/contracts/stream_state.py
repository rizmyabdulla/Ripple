"""Explicit bounded streaming-state schemas."""

from __future__ import annotations

import math
from enum import StrEnum
from functools import reduce
from operator import mul
from typing import Self

from pydantic import Field, model_validator

from ripple.contracts.manifest import ContractModel

STREAM_STATE_VERSION = "ripple-stream-state-1"
MAX_STATE_BYTES = 8 * 1024 * 1024


class TensorDType(StrEnum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT64 = "int64"
    INT32 = "int32"
    INT8 = "int8"
    BOOL = "bool"

    @property
    def itemsize(self) -> int:
        return {
            self.FLOAT32: 4,
            self.FLOAT16: 2,
            self.INT64: 8,
            self.INT32: 4,
            self.INT8: 1,
            self.BOOL: 1,
        }[self]


class StateTensorSpec(ContractModel):
    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.]*$")
    shape: tuple[int, ...] = Field(min_length=1)
    dtype: TensorDType
    reset_value: float | int | bool = 0

    @model_validator(mode="after")
    def validate_tensor(self) -> Self:
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("state tensor dimensions must be positive and fixed")
        if isinstance(self.reset_value, float) and not math.isfinite(self.reset_value):
            raise ValueError("reset_value must be finite")
        return self

    @property
    def size_bytes(self) -> int:
        return reduce(mul, self.shape, 1) * self.dtype.itemsize


class StreamStateSchema(ContractModel):
    schema_version: str = Field(default=STREAM_STATE_VERSION, pattern=r"^ripple-stream-state-1$")
    sample_rate: int = Field(default=24_000, ge=1)
    frame_samples: int = Field(default=480, ge=1)
    lookahead_frames: int = Field(default=0, ge=0, le=1)
    tensors: tuple[StateTensorSpec, ...]
    max_state_bytes: int = Field(default=MAX_STATE_BYTES, gt=0, le=MAX_STATE_BYTES)

    @model_validator(mode="after")
    def validate_schema(self) -> Self:
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError("state tensor names must be unique")
        if self.total_state_bytes > self.max_state_bytes:
            raise ValueError("declared state exceeds max_state_bytes")
        if self.sample_rate % self.frame_samples:
            raise ValueError("frame_samples must divide sample_rate")
        return self

    @property
    def total_state_bytes(self) -> int:
        return sum(tensor.size_bytes for tensor in self.tensors)

    @property
    def frame_rate(self) -> int:
        return self.sample_rate // self.frame_samples

