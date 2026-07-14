"""Backend capability discovery without importing optional SDKs eagerly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, ClassVar


class OptionalDependencyError(RuntimeError):
    """Raised when a requested deployment SDK is not installed."""

    def __init__(self, backend: str, packages: tuple[str, ...], extra: str) -> None:
        self.backend = backend
        self.packages = packages
        joined = " or ".join(packages)
        super().__init__(
            f"{backend} support requires optional dependency {joined}; "
            f"install Ripple's {extra!r} extra and the platform SDK"
        )


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    artifact_suffixes: tuple[str, ...]
    precisions: tuple[str, ...]
    explicit_state: bool
    mutable_state: bool
    fixed_shapes: bool = True
    device_specific: bool = False
    notes: str = ""


@dataclass(frozen=True)
class BackendAvailability:
    capabilities: BackendCapabilities
    available: bool
    missing_dependencies: tuple[str, ...]


class BackendExporter(ABC):
    capabilities: ClassVar[BackendCapabilities]
    dependencies: ClassVar[tuple[str, ...]]
    install_extra: ClassVar[str]

    @classmethod
    def probe(cls) -> BackendAvailability:
        missing = tuple(name for name in cls.dependencies if find_spec(name) is None)
        return BackendAvailability(cls.capabilities, not missing, missing)

    @classmethod
    def require_available(cls) -> None:
        availability = cls.probe()
        if not availability.available:
            raise OptionalDependencyError(
                cls.capabilities.name,
                availability.missing_dependencies,
                cls.install_extra,
            )

    @abstractmethod
    def export(self, model: Any, destination: str | Path, **kwargs: Any) -> Path:
        """Export a model to this backend's fixed-shape artifact."""


class OnnxRuntimeExporter(BackendExporter):
    capabilities = BackendCapabilities(
        name="onnxruntime",
        artifact_suffixes=(".onnx",),
        precisions=("fp32", "fp16", "int8"),
        explicit_state=True,
        mutable_state=False,
        notes="Canonical portable conformance backend.",
    )
    dependencies = ("onnx", "onnxruntime")
    install_extra = "onnx"

    def export(self, model: Any, destination: str | Path, **kwargs: Any) -> Path:
        self.require_available()
        from .onnx import export_onnx

        return export_onnx(model, destination=destination, **kwargs)


class TensorRTExporter(BackendExporter):
    capabilities = BackendCapabilities(
        name="tensorrt",
        artifact_suffixes=(".engine", ".plan"),
        precisions=("fp32", "fp16", "bf16", "int8"),
        explicit_state=True,
        mutable_state=False,
        device_specific=True,
        notes="Requires a device-specific optimization profile.",
    )
    dependencies = ("tensorrt",)
    install_extra = "tensorrt"

    def export(self, model: Any, destination: str | Path, **kwargs: Any) -> Path:
        self.require_available()
        raise NotImplementedError(
            "TensorRT engine construction requires a target/device build policy"
        )


class CoreMLExporter(BackendExporter):
    capabilities = BackendCapabilities(
        name="coreml",
        artifact_suffixes=(".mlpackage",),
        precisions=("fp32", "fp16", "int8"),
        explicit_state=True,
        mutable_state=True,
        notes="MLState use depends on the deployment target.",
    )
    dependencies = ("coremltools",)
    install_extra = "coreml"

    def export(self, model: Any, destination: str | Path, **kwargs: Any) -> Path:
        self.require_available()
        raise NotImplementedError(
            "Core ML conversion requires an explicit deployment target"
        )


class LiteRTExporter(BackendExporter):
    capabilities = BackendCapabilities(
        name="litert",
        artifact_suffixes=(".tflite",),
        precisions=("fp32", "fp16", "int8"),
        explicit_state=True,
        mutable_state=False,
        notes="Delegate partitioning must be validated on the target device.",
    )
    dependencies = ("ai_edge_litert",)
    install_extra = "litert"

    def export(self, model: Any, destination: str | Path, **kwargs: Any) -> Path:
        self.require_available()
        raise NotImplementedError(
            "LiteRT conversion requires a target operator and quantization policy"
        )


class ExecuTorchExporter(BackendExporter):
    capabilities = BackendCapabilities(
        name="executorch",
        artifact_suffixes=(".pte",),
        precisions=("fp32", "fp16", "int8"),
        explicit_state=True,
        mutable_state=True,
        notes="Delegate selection is target-specific.",
    )
    dependencies = ("executorch",)
    install_extra = "executorch"

    def export(self, model: Any, destination: str | Path, **kwargs: Any) -> Path:
        self.require_available()
        raise NotImplementedError(
            "ExecuTorch lowering requires an explicit delegate configuration"
        )


_BACKENDS: dict[str, type[BackendExporter]] = {
    "onnx": OnnxRuntimeExporter,
    "onnxruntime": OnnxRuntimeExporter,
    "ort": OnnxRuntimeExporter,
    "tensorrt": TensorRTExporter,
    "coreml": CoreMLExporter,
    "litert": LiteRTExporter,
    "executorch": ExecuTorchExporter,
}


def get_backend(name: str) -> type[BackendExporter]:
    try:
        return _BACKENDS[name.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"unknown backend {name!r}; expected one of {supported}") from error


def backend_capability_report() -> dict[str, BackendAvailability]:
    """Return one availability record per canonical backend."""

    classes = (
        OnnxRuntimeExporter,
        TensorRTExporter,
        CoreMLExporter,
        LiteRTExporter,
        ExecuTorchExporter,
    )
    return {backend.capabilities.name: backend.probe() for backend in classes}
