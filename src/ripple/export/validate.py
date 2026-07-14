"""Structural and numerical ONNX validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .backends import OptionalDependencyError


@dataclass(frozen=True)
class OnnxValidationReport:
    path: Path
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    operators: tuple[str, ...]
    fixed_shapes: bool


def _require_onnx() -> Any:
    try:
        import onnx
    except ImportError as error:
        raise OptionalDependencyError("ONNX validation", ("onnx",), "onnx") from error
    return onnx


def _is_fixed(value_info: Any) -> bool:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return False
    for dimension in tensor_type.shape.dim:
        if dimension.dim_param or not dimension.HasField("dim_value"):
            return False
        if dimension.dim_value < 1:
            return False
    return True


def validate_onnx_model(
    path: str | Path,
    *,
    expected_inputs: Sequence[str] | None = None,
    expected_outputs: Sequence[str] | None = None,
    require_fixed_shapes: bool = True,
    supported_operators: Iterable[str] | None = None,
) -> OnnxValidationReport:
    """Run ONNX checker and enforce Ripple's fixed explicit-I/O contract."""

    onnx = _require_onnx()
    model_path = Path(path)
    model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = tuple(
        item.name for item in model.graph.input if item.name not in initializer_names
    )
    outputs = tuple(item.name for item in model.graph.output)
    if expected_inputs is not None and inputs != tuple(expected_inputs):
        raise ValueError(f"ONNX inputs {inputs} do not match expected {tuple(expected_inputs)}")
    if expected_outputs is not None and outputs != tuple(expected_outputs):
        raise ValueError(
            f"ONNX outputs {outputs} do not match expected {tuple(expected_outputs)}"
        )

    value_infos = (*model.graph.input, *model.graph.output)
    fixed = all(_is_fixed(item) for item in value_infos)
    if require_fixed_shapes and not fixed:
        dynamic = tuple(item.name for item in value_infos if not _is_fixed(item))
        raise ValueError(f"ONNX graph contains dynamic or unknown shapes: {dynamic}")

    operators = tuple(
        sorted({f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.node})
    )
    if supported_operators is not None:
        supported = set(supported_operators)
        unsupported = tuple(op for op in operators if op not in supported)
        if unsupported:
            raise ValueError(f"unsupported ONNX operators: {unsupported}")
    return OnnxValidationReport(model_path, inputs, outputs, operators, fixed)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def run_onnx(
    path: str | Path,
    inputs: Mapping[str, Any],
    *,
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> tuple[np.ndarray, ...]:
    """Execute a graph with ONNX Runtime and no implicit input reshaping."""

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise OptionalDependencyError(
            "ONNX Runtime validation", ("onnxruntime",), "onnx"
        ) from error
    options = ort.SessionOptions()
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=list(providers)
    )
    expected = tuple(item.name for item in session.get_inputs())
    if set(inputs) != set(expected):
        raise ValueError(
            f"runtime input names {tuple(inputs)} do not match graph inputs {expected}"
        )
    values = {name: _as_numpy(inputs[name]) for name in expected}
    return tuple(session.run(None, values))


def assert_outputs_close(
    reference: Sequence[Any],
    actual: Sequence[Any],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    """Compare waveform and every state output with useful output indices."""

    if len(reference) != len(actual):
        raise AssertionError(
            f"output count differs: reference={len(reference)}, actual={len(actual)}"
        )
    for index, (expected, observed) in enumerate(zip(reference, actual, strict=True)):
        try:
            np.testing.assert_allclose(
                _as_numpy(observed), _as_numpy(expected), rtol=rtol, atol=atol
            )
        except AssertionError as error:
            raise AssertionError(f"output {index} failed conformance: {error}") from error
