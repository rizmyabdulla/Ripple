from __future__ import annotations

import torch
from torch import nn

from ripple.quantization import (
    PTQConfig,
    QATConfig,
    analyze_sensitivity,
    available_backends,
    collect_calibration_ranges,
    prepare_qat,
    quantize_ptq,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(4, 8)
        self.activation = nn.ReLU()
        self.output = nn.Linear(8, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.input(values)))


def test_calibration_coverage_and_ranges() -> None:
    torch.manual_seed(4)
    model = TinyModel()
    report = collect_calibration_ranges(
        model,
        [torch.randn(3, 4) for _ in range(3)],
    )
    assert report.batches == 3
    assert report.eligible_modules == 2
    assert report.observed_modules == 2
    assert report.coverage == 1
    assert all(item.observations == 3 and item.finite for item in report.ranges)


def test_dynamic_ptq_or_explicit_backend_fallback() -> None:
    model = TinyModel().eval()
    backend = available_backends()[0] if available_backends() else "missing"
    result = quantize_ptq(model, config=PTQConfig(mode="dynamic", backend=backend))
    if available_backends():
        assert result.quantized
        output = result.model(torch.randn(2, 4))
        assert output.shape == (2, 2)
        assert torch.isfinite(output).all()
    else:
        assert not result.quantized
        assert result.warnings

    fallback = quantize_ptq(model, config=PTQConfig(backend="definitely-not-a-backend"))
    assert not fallback.quantized
    assert "unavailable" in fallback.warnings[0]


def test_qat_preparation_respects_sensitive_name_exclusions() -> None:
    backend = available_backends()[0] if available_backends() else "missing"
    result = prepare_qat(
        TinyModel(),
        QATConfig(backend=backend, preserve_name_fragments=("output",)),
    )
    if available_backends():
        assert result.prepared
        assert result.model.output.qconfig is None
        assert result.model.input.qconfig is not None
    else:
        assert not result.prepared


def test_sensitivity_restores_weights_and_ranks_layers() -> None:
    torch.manual_seed(5)
    model = TinyModel().eval()
    inputs = torch.randn(4, 4)
    target = model(inputs).detach()
    original = {name: value.clone() for name, value in model.state_dict().items()}

    def evaluate(candidate: nn.Module) -> float:
        with torch.inference_mode():
            return -float(torch.mean((candidate(inputs) - target) ** 2))

    results = analyze_sensitivity(model, evaluate, num_bits=4)
    assert {item.name for item in results} == {"input", "output"}
    assert all(item.degradation >= 0 for item in results)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
