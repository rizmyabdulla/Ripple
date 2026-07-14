"""Canonical fixed-shape ONNX export."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from .backends import BackendAvailability, OnnxRuntimeExporter, OptionalDependencyError
from .state import flatten_state
from .torch_export import (
    ExportSignature,
    StepFunction,
    _build_adapter,
    _default_step,
    export_torch_program,
)


@dataclass(frozen=True)
class OnnxExportResult:
    path: Path
    signature: ExportSignature


def capabilities() -> BackendAvailability:
    return OnnxRuntimeExporter.probe()


def export_onnx(
    model: Any,
    *,
    pcm: Any,
    state: Any,
    destination: str | Path,
    speaker_profile: Any | None = None,
    invoke: StepFunction = _default_step,
    chunk_samples: int = 480,
    opset_version: int = 18,
    validate: bool = True,
) -> Path:
    """Export one fixed streaming step with all state as named graph I/O."""

    try:
        import onnx  # noqa: F401
        import torch
    except ImportError as error:
        package = getattr(error, "name", None) or "torch/onnx"
        raise OptionalDependencyError("ONNX export", (package,), "onnx") from error

    path = Path(destination)
    if path.suffix.lower() != ".onnx":
        raise ValueError("ONNX destination must end in .onnx")
    path.parent.mkdir(parents=True, exist_ok=True)

    captured = export_torch_program(
        model,
        pcm,
        state,
        speaker_profile=speaker_profile,
        invoke=invoke,
        chunk_samples=chunk_samples,
    )
    common_options = {
        "input_names": list(captured.signature.input_names),
        "output_names": list(captured.signature.output_names),
        "opset_version": opset_version,
    }
    if find_spec("onnxscript") is not None:
        torch.onnx.export(
            captured.program,
            (),
            str(path),
            **common_options,
            dynamo=True,
            dynamic_shapes=None,
            external_data=False,
        )
    else:
        # PyTorch's newer dynamo ONNX path requires onnxscript. The legacy
        # exporter remains a valid fixed-shape fallback and receives no
        # dynamic_axes declaration.
        flattened = flatten_state(state)
        adapter = _build_adapter(
            model, flattened, speaker_profile is not None, invoke
        )
        args = (
            (pcm, *flattened.tensors)
            if speaker_profile is None
            else (pcm, speaker_profile, *flattened.tensors)
        )
        torch.onnx.export(
            adapter,
            args,
            str(path),
            **common_options,
            dynamo=False,
        )
    if validate:
        from .validate import validate_onnx_model

        validate_onnx_model(
            path,
            expected_inputs=captured.signature.input_names,
            expected_outputs=captured.signature.output_names,
            require_fixed_shapes=True,
        )
    return path
