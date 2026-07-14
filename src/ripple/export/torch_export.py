"""Fixed-shape ``torch.export`` capture for a streaming step."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import OptionalDependencyError
from .state import (
    FlattenedState,
    StateTensorSpec,
    flatten_state,
    flatten_state_values,
    unflatten_state,
    validate_flat_state,
)

StepFunction = Callable[[Any, Any, Any, Any | None], tuple[Any, Any]]


@dataclass(frozen=True)
class ExportSignature:
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    state_tensors: tuple[StateTensorSpec, ...]
    pcm_shape: tuple[int, ...]
    profile_shape: tuple[int, ...] | None


@dataclass(frozen=True)
class TorchExportResult:
    program: Any
    signature: ExportSignature
    path: Path | None = None


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise OptionalDependencyError("torch.export", ("torch",), "export") from error
    return torch


def _default_step(
    model: Any, pcm: Any, state: Any, speaker_profile: Any | None
) -> tuple[Any, Any]:
    if hasattr(model, "stream_step"):
        step = model.stream_step
    elif hasattr(model, "step"):
        step = model.step
    else:
        step = model
    result = (
        step(pcm, state)
        if speaker_profile is None
        else step(pcm, state, speaker_profile)
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("streaming step must return (pcm_output, updated_state)")
    return result


def _build_adapter(
    model: Any,
    flattened: FlattenedState,
    has_profile: bool,
    invoke: StepFunction,
) -> Any:
    torch = _require_torch()

    class StreamingExportAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, pcm: Any, *values: Any) -> tuple[Any, ...]:
            if has_profile:
                profile = values[0]
                flat_state = values[1:]
            else:
                profile = None
                flat_state = values
            state = unflatten_state(flat_state, flattened.tree)
            pcm_output, updated_state = invoke(self.model, pcm, state, profile)
            updated = flatten_state_values(updated_state, flattened.tree)
            return (pcm_output, *updated)

    return StreamingExportAdapter()


def export_torch_program(
    model: Any,
    pcm: Any,
    state: Any,
    *,
    speaker_profile: Any | None = None,
    destination: str | Path | None = None,
    invoke: StepFunction = _default_step,
    chunk_samples: int = 480,
    strict: bool = True,
) -> TorchExportResult:
    """Capture one fixed streaming step and optionally save a ``.pt2`` file.

    The model must expose ``stream_step``/``step`` (or be callable) with
    ``(pcm, state[, speaker_profile])`` and return ``(pcm, updated_state)``.
    Dynamic shapes are deliberately not supplied to ``torch.export``.
    """

    torch = _require_torch()
    if not hasattr(pcm, "shape"):
        raise TypeError("pcm must be a tensor")
    pcm_shape = tuple(int(dim) for dim in pcm.shape)
    if not pcm_shape or pcm_shape[-1] != chunk_samples:
        raise ValueError(
            f"fixed PCM input must end in {chunk_samples} samples; got {pcm_shape}"
        )

    flattened = flatten_state(state)
    adapter = _build_adapter(model, flattened, speaker_profile is not None, invoke)
    args = (
        (pcm, *flattened.tensors)
        if speaker_profile is None
        else (pcm, speaker_profile, *flattened.tensors)
    )
    eager_outputs = adapter(*args)
    validate_flat_state(eager_outputs[1:], flattened.tensor_specs)
    pcm_output_shape = tuple(int(dim) for dim in eager_outputs[0].shape)
    if not pcm_output_shape or pcm_output_shape[-1] != chunk_samples:
        raise ValueError(
            f"fixed PCM output must end in {chunk_samples} samples; "
            f"got {pcm_output_shape}"
        )
    program = torch.export.export(adapter, args, dynamic_shapes=None, strict=strict)

    state_names = tuple(spec.name for spec in flattened.tensor_specs)
    input_names = (
        ("pcm", *state_names)
        if speaker_profile is None
        else ("pcm", "speaker_profile", *state_names)
    )
    output_names = ("pcm_out", *(f"{name}_out" for name in state_names))
    profile_shape = (
        None
        if speaker_profile is None
        else tuple(int(dim) for dim in speaker_profile.shape)
    )
    signature = ExportSignature(
        input_names=input_names,
        output_names=output_names,
        state_tensors=flattened.tensor_specs,
        pcm_shape=pcm_shape,
        profile_shape=profile_shape,
    )

    path: Path | None = None
    if destination is not None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.export.save(program, path)
    return TorchExportResult(program=program, signature=signature, path=path)
