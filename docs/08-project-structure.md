# Clean project architecture

## 1. Repository layout

```text
Ripple/
├── docs/
├── pyproject.toml
├── configs/
│   ├── model/
│   ├── data/
│   ├── train/
│   ├── eval/
│   ├── export/
│   └── benchmark/
├── src/ripple/
│   ├── contracts/
│   │   ├── rif.py
│   │   ├── speaker_profile.py
│   │   ├── stream_state.py
│   │   └── manifest.py
│   ├── audio/
│   │   ├── io.py
│   │   ├── resample.py
│   │   ├── framing.py
│   │   ├── pitch.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── analysis_encoder.py
│   │   ├── ripple_mixer.py
│   │   ├── prosody_encoder.py
│   │   ├── speaker_encoder.py
│   │   ├── source_filter.py
│   │   ├── decoder.py
│   │   ├── discriminators.py
│   │   └── ripple_vc.py
│   ├── streaming/
│   │   ├── cached_conv.py
│   │   ├── local_attention.py
│   │   ├── running_stats.py
│   │   ├── packet_loss.py
│   │   └── session.py
│   ├── teachers/
│   │   ├── hubert.py
│   │   ├── wavlm.py
│   │   ├── multilingual.py
│   │   └── projection.py
│   ├── data/
│   │   ├── manifest.py
│   │   ├── filters.py
│   │   ├── features.py
│   │   ├── sampler.py
│   │   └── datamodule.py
│   ├── training/
│   │   ├── stages.py
│   │   ├── losses.py
│   │   ├── schedules.py
│   │   ├── checkpoint.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── intelligibility.py
│   │   ├── speaker.py
│   │   ├── prosody.py
│   │   ├── quality.py
│   │   ├── leakage.py
│   │   └── long_session.py
│   ├── export/
│   │   ├── onnx.py
│   │   ├── executorch.py
│   │   ├── coreml.py
│   │   ├── litert.py
│   │   ├── tensorrt.py
│   │   └── validate.py
│   ├── quantization/
│   │   ├── calibrate.py
│   │   ├── ptq.py
│   │   ├── qat.py
│   │   └── sensitivity.py
│   ├── benchmark/
│   │   ├── latency.py
│   │   ├── memory.py
│   │   ├── quality.py
│   │   └── report.py
│   └── cli/
├── runtime/
│   ├── include/ripple.h
│   ├── src/
│   ├── backends/
│   │   ├── onnxruntime/
│   │   ├── coreml/
│   │   ├── litert/
│   │   └── tensorrt/
│   └── tests/
├── tools/
│   ├── prepare_dataset.py
│   ├── extract_teacher_features.py
│   ├── enroll_speaker.py
│   ├── convert.py
│   ├── stream_demo.py
│   ├── export_model.py
│   └── benchmark_model.py
└── tests/
    ├── unit/
    ├── streaming/
    ├── export/
    ├── quantization/
    ├── integration/
    └── regression/
```

## 2. Dependency boundaries

### Contracts

Contains only stable schemas and validation. It must not import training, runtime backends, or data pipelines.

### Audio

Contains deterministic DSP and framing. Training augmentation is separated from production conditioning so an augmentation cannot enter the runtime graph accidentally.

### Models

Defines stateless full-sequence reference functions and explicit streaming-step functions. Models do not open files, create datasets, or select backends.

### Streaming

Owns cache/state transitions and reset semantics. The same logic is used during training simulation, PyTorch inference, export validation, and runtime tests.

### Teachers

Training-only adapters. Production packaging must fail if a teacher dependency is reachable from the exported graph.

### Data

Manifest-first ingestion and feature-version validation. It never embeds absolute machine paths in checkpoints.

### Training

Stage orchestration, losses, optimizers, and logging. Training consumes interfaces from models/contracts, not backend-specific runtime code.

### Evaluation

Independent evaluators and immutable manifests. It must support evaluating external baselines, not only Ripple checkpoints.

### Export/quantization

Converts an already functional streaming model and verifies outputs. Export code must not patch model behavior to make tests pass.

### Native runtime

Implements the stable C ABI and backend adapters. It does not contain training code or Python in the production path.

## 3. Configuration policy

- Typed configuration objects with schema validation.
- Immutable resolved config stored with every checkpoint/report.
- No executable Python hidden in YAML.
- Separate architecture, training, data, and deployment configs.
- Every exported artifact records the exact resolved model and state schema.

## 4. Testing framework

### Unit

- causal convolution matches full-sequence reference;
- state shapes and resets;
- running pitch statistics;
- oscillator phase;
- speaker-profile serialization;
- manifest validation.

### Streaming

- concatenated step output matches reference within tolerance;
- arbitrary input-call grouping produces the same frame sequence;
- zero-lookahead invariant;
- soft/hard reset behavior;
- packet-loss recovery;
- no hidden state growth.

### Export

- PyTorch versus ONNX per chunk and over long streams;
- backend state round-trip;
- fixed-shape enforcement;
- unsupported operator detection;
- specialist backend smoke tests.

### Quantization

- layer sensitivity;
- calibration coverage;
- state drift;
- quality canary suite.

### Integration

- speaker enrollment then conversion;
- profile compatibility errors;
- CLI and C ABI;
- single-artifact packaging;
- fallback behavior.

### Regression

- frozen audio fixtures and metric tolerances;
- latency/RSS budgets on named runners;
- one-hour nightly and eight-hour release tests.

## 5. Tooling

Training tools:

- manifest builder and license validator;
- teacher-feature extractor;
- distributed training launcher;
- checkpoint averaging and conversion.

Evaluation tools:

- matched-pair generator;
- objective metric runner;
- listening-test packager;
- long-session stream generator.

Performance tools:

- PyTorch profiler;
- ONNX Runtime profiling;
- backend trace parsers;
- per-module MAC/parameter/state report;
- thermal and RSS sampler;
- regression dashboard generator.

## 6. CI/CD

Pull requests:

- formatting, typing, unit tests;
- short streaming equivalence test;
- ONNX export smoke test;
- artifact schema compatibility.

Nightly:

- quality canary;
- PTQ export;
- one-hour long-session test;
- CPU latency/RSS benchmark.

Release:

- full multilingual evaluation;
- subjective test sign-off;
- eight-hour stability;
- all backend artifacts;
- license/data/model/safety cards;
- signed manifests and checksums.

## 7. Reproducibility

Every report includes:

- source commit;
- dirty-tree flag;
- resolved config;
- dataset/feature manifest hashes;
- teacher and checkpoint hashes;
- hardware/runtime versions;
- random seeds;
- output artifact checksum.

Research comparisons that cannot record this metadata are exploratory and cannot become release claims.
