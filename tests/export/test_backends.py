from __future__ import annotations

import pytest

from ripple.export.backends import (
    OptionalDependencyError,
    backend_capability_report,
    get_backend,
)


def test_all_documented_backends_publish_fixed_shape_capabilities() -> None:
    report = backend_capability_report()
    assert set(report) == {
        "onnxruntime",
        "tensorrt",
        "coreml",
        "litert",
        "executorch",
    }
    assert all(item.capabilities.fixed_shapes for item in report.values())
    assert all(item.capabilities.explicit_state for item in report.values())


def test_missing_sdk_has_actionable_optional_dependency_error(tmp_path) -> None:
    backend = get_backend("litert")
    availability = backend.probe()
    if availability.available:
        pytest.skip("LiteRT SDK is installed in this environment")
    with pytest.raises(OptionalDependencyError, match=r"LiteRT|litert"):
        backend().export(object(), tmp_path / "model.tflite")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("mystery")
