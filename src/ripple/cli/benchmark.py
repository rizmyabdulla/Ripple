"""Synthetic CPU latency benchmark CLI."""

# ruff: noqa: B008

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
import typer

from .common import build_backend

app = typer.Typer(help="Benchmark one synthetic streaming step on CPU.")


@app.command()
def benchmark(
    artifact: Path = typer.Argument(..., exists=True),
    backend: str = typer.Option(..., help="Runtime factory in module:callable form."),
    iterations: int = typer.Option(100, min=2),
    warmup: int = typer.Option(10, min=0),
    chunk_samples: int = typer.Option(480, min=1),
    output: Path | None = typer.Option(None),
) -> None:
    runtime = build_backend(backend, artifact=artifact, device="cpu")
    step = getattr(runtime, "benchmark_step", None)
    if step is None:
        raise typer.BadParameter(
            "backend must implement benchmark_step(pcm_tensor)"
        )
    pcm = torch.zeros(1, chunk_samples)
    with torch.inference_mode():
        for _ in range(warmup):
            step(pcm)
        timings: list[float] = []
        for _ in range(iterations):
            start = perf_counter()
            step(pcm)
            timings.append((perf_counter() - start) * 1000.0)
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    report = {
        "device": "cpu",
        "iterations": iterations,
        "chunk_samples": chunk_samples,
        "mean_ms": mean(timings),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": p95,
        "max_ms": ordered[-1],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    typer.echo(rendered)


if __name__ == "__main__":
    app()
