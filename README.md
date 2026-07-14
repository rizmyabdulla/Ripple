# Ripple

Ripple is a causal, bounded-state streaming speech-to-speech model family with a stable
RIF-1 interface for future text-to-speech. This repository implements the Phase 0–10
software scaffold from the architecture dossier: contracts, streaming models, training and
evaluation harnesses, export/runtime, quantization hooks, CLI/tools, safety governance, and
research-gate utilities.

The code is infrastructure and synthetic-test validated. It is not evidence of production
quality, latency, or trained multilingual performance.

## Layout

| Path | Role |
|---|---|
| `src/ripple/contracts` | RIF-1, speaker, stream-state, artifact schemas |
| `src/ripple/models` + `streaming` | Ripple-VC Edge and causal streaming primitives |
| `src/ripple/baselines` | Corrected StreamVC-style Yin threshold baseline |
| `src/ripple/training` / `evaluation` / `benchmark` / `quantization` | Train/eval/latency/INT8 harnesses |
| `src/ripple/export` + `runtime/` | ONNX export path and C ABI native runtime |
| `src/ripple/tts` / `teachers` / `cli` / `safety` | TTS frontend, local teachers, Typer CLI, consent |
| `docs/` | Research dossier (read-only architecture source of truth) |
| `governance/` + `adrs/` | Release cards and decision records |

## Requirements

- CPython 3.12
- `libsndfile` for SoundFile-based audio I/O
- Optional PyTorch/TorchCodec for tensor paths; ONNX Runtime for export tests

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,torch,export,data]"
```

Environment subprojects live under `envs/{dev,train,teachers,evaluation,export}`.

## Validate

```powershell
python -m pytest tests -q
python -m ruff check src tests
python -m mypy src/ripple/contracts src/ripple/research
```

## CLI

```powershell
ripple --help
```

## Licensing

This project is licensed under [Creative Commons Attribution 4.0 International
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE)
for the full text.

Dataset corpora and trained model weights are licensed separately and are tracked in
dataset/model cards and manifests.

