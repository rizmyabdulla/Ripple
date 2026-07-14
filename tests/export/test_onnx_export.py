from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from ripple.export.onnx import export_onnx
from ripple.export.torch_export import export_torch_program
from ripple.export.validate import assert_outputs_close, run_onnx, validate_onnx_model


class StreamingAdd(torch.nn.Module):
    def step(
        self, pcm: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return pcm + state["cache"], {"cache": pcm * 0.5}


def test_torch_export_has_fixed_explicit_state_signature(tmp_path) -> None:
    pcm = torch.arange(480, dtype=torch.float32).reshape(1, 480)
    state = {"cache": torch.zeros_like(pcm)}
    result = export_torch_program(
        StreamingAdd(), pcm, state, destination=tmp_path / "stream.pt2"
    )

    assert result.path is not None and result.path.exists()
    assert result.signature.input_names == ("pcm", "state_cache")
    assert result.signature.output_names == ("pcm_out", "state_cache_out")
    assert result.signature.pcm_shape == (1, 480)


def test_export_rejects_non_contract_chunk() -> None:
    pcm = torch.zeros(1, 479)
    with pytest.raises(ValueError, match="480"):
        export_torch_program(
            StreamingAdd(), pcm, {"cache": torch.zeros_like(pcm)}
        )


@pytest.mark.skipif(
    any(
        importlib.util.find_spec(package) is None
        for package in ("onnx", "onnxruntime")
    ),
    reason="ONNX export dependencies are not installed",
)
def test_onnx_export_validates_and_matches_ort(tmp_path) -> None:
    pcm = torch.linspace(-1.0, 1.0, 480).reshape(1, 480)
    cache = torch.full_like(pcm, 0.25)
    path = export_onnx(
        StreamingAdd(),
        pcm=pcm,
        state={"cache": cache},
        destination=tmp_path / "stream.onnx",
    )

    report = validate_onnx_model(
        path,
        expected_inputs=("pcm", "state_cache"),
        expected_outputs=("pcm_out", "state_cache_out"),
    )
    assert report.fixed_shapes
    actual = run_onnx(
        path, {"pcm": pcm.numpy(), "state_cache": cache.numpy()}
    )
    expected = ((pcm + cache).numpy(), (pcm * 0.5).numpy())
    assert_outputs_close(expected, actual)
    assert actual[0].shape == (1, 480)
    assert actual[1].dtype == np.float32
