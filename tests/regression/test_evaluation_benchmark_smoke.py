from __future__ import annotations

import json
import time

import torch
from torch import nn

from ripple.benchmark import (
    BenchmarkReport,
    LatencyConfig,
    benchmark_latency,
    model_memory_report,
)
from ripple.evaluation import (
    LongSessionConfig,
    character_error_rate,
    leakage_report,
    prosody_report,
    run_long_session,
    waveform_report,
    word_error_rate,
)


def test_objective_metrics_on_synthetic_inputs() -> None:
    assert word_error_rate("Ripple is streaming", "ripple streaming") == 1 / 3
    assert character_error_rate("abc", "adc") == 1 / 3
    reference = torch.sin(torch.linspace(0, 10, 480))
    quality = waveform_report(reference, reference)
    assert quality["mae"] == 0
    assert quality["finite_fraction"] == 1

    f0 = torch.linspace(100, 200, 20)
    voicing = torch.ones(20, dtype=torch.bool)
    prosody = prosody_report(f0, f0, voicing, voicing)
    assert prosody["f0_pearson"] > 0.999
    assert prosody["voicing_f1"] == 1

    source = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0]])
    report = leakage_report(target, source, target)
    assert report["target_source_margin"] == 1


def test_latency_long_session_memory_and_json_report(tmp_path) -> None:
    layer = nn.Linear(8, 8).eval()
    inputs = torch.randn(1, 8)
    latency = benchmark_latency(
        lambda: layer(inputs),
        LatencyConfig(warmup_iterations=1, measured_iterations=5, audio_seconds_per_call=0.02),
    )
    assert latency.calls == 5
    assert latency.p95_ms >= latency.p50_ms
    assert latency.mean_rtf >= 0

    def stream_step(chunk: torch.Tensor, state: torch.Tensor | None):
        state = torch.zeros(1) if state is None else state
        return chunk * 0.5, state * 0.9 + chunk.mean()

    session = run_long_session(
        stream_step,
        [torch.randn(1, 32)],
        config=LongSessionConfig(
            iterations=12,
            sample_every=4,
            reset_every=5,
            packet_loss_probability=0.1,
            seed=3,
            collect_garbage=False,
        ),
    )
    assert session.iterations == 12
    assert session.nan_inf_count == 0
    assert session.resets == 2
    assert session.output_samples == 12 * 32

    memory = model_memory_report(layer, state=torch.zeros(4))
    assert memory["parameter_bytes"] > 0
    assert memory["state_bytes"] == 16
    assert memory["rss_bytes"] > 0

    report = BenchmarkReport(
        model_id="synthetic",
        runtime="pytorch",
        precision="fp32",
        sample_rate=24_000,
        chunk_samples=480,
        resolved_config={"threads": 1},
        manifest_hashes={"canary": "abc"},
    )
    report.add("latency", latency)
    report.add("long_session", session)
    destination = report.write_json(tmp_path / "report.json")
    decoded = json.loads(destination.read_text(encoding="utf-8"))
    assert decoded["schema_version"] == 1
    assert decoded["sections"]["latency"]["calls"] == 5
