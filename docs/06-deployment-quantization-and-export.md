# Deployment, quantization, export, and optimization

## 1. Runtime tiers

### ONNX Runtime: canonical portable tier

Use fixed-chunk ONNX with explicit state I/O as the reference export. ONNX Runtime provides broad CPU/GPU support, offline graph optimization, memory arenas, and quantization tooling. It is the conformance baseline against PyTorch.

### TensorRT: NVIDIA specialist

Build FP16/BF16 and selectively quantized engines for desktop GPU and Jetson. Engines are device-specific and require optimization profiles even when Ripple uses fixed shapes. TensorRT is a deployment specialization, not the canonical model format.

### Core ML: Apple specialist

Use `.mlpackage` with fixed/enumerated shapes. Where OS support permits, map explicit state to `MLState`; retain the explicit-state conformance path for older targets. Verify that all partitions remain on ANE/GPU rather than silently falling back.

### LiteRT: Android/NPU specialist

Use fixed shapes and backend-aware quantization for Android CPU/GPU/NPU. Keep the graph within supported operator sets and measure partition boundaries. A theoretically quantized graph that moves unsupported operators to CPU can be slower.

### ExecuTorch: PyTorch edge tier

Use `.pte` where its XNNPACK, Core ML, or QNN delegates provide a better integration path. Mutable-buffer support can represent state, but the public Ripple ABI remains unchanged.

### GGUF/ggml

GGUF is not a general IR for Ripple’s causal convolution/prosody graph. It is relevant only if a later TTS/dialogue component uses a supported LLM runtime. `whisper.cpp`-style custom formats are architecture-specific, not a universal export strategy.

### Custom runtime

A custom C++/Rust runtime is justified only after profiling shows framework overhead or unsupported fused kernels dominate. It must retain ONNX numerical conformance and the same state/profile schemas.

## 2. Canonical export flow

```text
PyTorch streaming module
  -> fixed-shape torch.export test
  -> ONNX with explicit states
  -> ONNX graph optimization
  -> backend-neutral numerical/streaming conformance
  -> PTQ/QAT variants
  -> TensorRT / Core ML / LiteRT / ExecuTorch specialists
  -> packaged artifact with manifest
```

Export begins in the first implementation phase. Architectures that cannot export are rejected before expensive full training.

## 3. Precision policy

### Reference

- FP32 for numerical oracle and debugging.
- BF16 for training where hardware supports it.
- FP16/BF16 as the first production-quality tier.

### INT8

INT8 is the primary edge target:

- per-channel symmetric weight quantization for Conv/Linear;
- calibrated activation quantization where target hardware benefits;
- representative streaming calibration with real states, starts, silence, and packet loss;
- keep final waveform projection, oscillator math, normalization statistics, and sensitive pitch layers at FP16/FP32 if required.

### INT4

Use only on sufficiently large matrix layers in the latent mixer or future text/LLM modules. Small convolutions, depthwise layers, speaker heads, and prosody estimators are often accuracy- or dispatch-bound rather than weight-bandwidth-bound.

### AWQ/GPTQ

AWQ and GPTQ are designed primarily for large Transformer linear layers. They are not default methods for Ripple’s convolutional/attention edge graph. A future TTS language model may use them; the VC hot path should prefer backend-native INT8, weight-only INT8/INT4, or QAT.

## 4. PTQ and QAT pipeline

1. Export FP model and establish bit-exact or tolerance-bounded stream conformance.
2. Collect calibration streams by language, speaker, SNR, pitch range, chunk transition, and state age.
3. Quantize one module family at a time.
4. Measure quality and latency, not artifact size alone.
5. Keep a sensitivity map for each layer.
6. Introduce mixed precision for outliers.
7. If PTQ misses gates, run streaming QAT with fake quantization.
8. Repeat long-session tests because recurrent state can accumulate small errors.

Quantization gates are defined in the evaluation document. No precision tier ships solely because it is faster.

## 5. Graph optimization

- Fold inference-only normalization and affine operations where numerically safe.
- Fuse Conv/Linear + bias + activation.
- Fuse speaker modulation projections shared across a profile.
- Precompute all target-profile FiLM/AdaRMS parameters that do not vary by frame.
- Keep oscillator generation in-graph to avoid host copies.
- Replace redundant transposes and contiguous conversions.
- Use offline optimized graphs to reduce startup cost.
- Specialize all dimensions: batch 1, 480 input samples, one 50 Hz latent frame.
- Benchmark layout choices per backend; channel-first is not universally optimal.

## 6. Memory optimization

- Memory-map immutable weights where supported.
- Allocate state and scratch buffers from one arena.
- Reuse output buffers.
- Store local-attention rings in the backend-preferred contiguous layout.
- Keep speaker profile tensors resident and preprojected.
- Avoid storing teacher outputs or diagnostic logits in production.
- Split enrollment and steady-state sessions while packaging them together.
- Report peak RSS after warm-up and after hours of streaming.

## 7. CPU optimization

- Tune one physical performance core first.
- Cap intra-op threads; oversubscription harms p99.
- Use SIMD kernels through XNNPACK/oneDNN/Accelerate/KleidiAI or backend equivalents.
- Measure power and thermal throttling over 30 minutes.
- Avoid Python, dynamic dispatch, and per-frame allocation.
- Prefer depthwise kernels only where the target backend implements them efficiently; a dense small convolution may be faster on some CPUs.
- Benchmark stride and channel multiples against actual vector widths.

## 8. GPU/NPU optimization

- Keep fixed shapes to maximize compilation and partitioning.
- Use FP16/BF16 before lower precision.
- Minimize graph partitions and device/host transfers.
- Precompile engines/models during packaging when allowed.
- Check first-run compilation separately from warm latency.
- Validate output under the exact device delegate; CPU fallback can hide unsupported operators.

FlashAttention may help training or large server variants, but fixed-window batch-one edge attention is too small to justify an architecture dependency on it.

## 9. Stateful export constraints

- Hidden Python state is forbidden.
- Every ring, memory vector, running statistic, and oscillator phase must be a tensor or explicit host DSP state.
- Dynamic control flow should be a host loop over fixed graph calls.
- Dynamic sequence length is not needed for the live graph.
- Quality and edge modes should be separately specialized artifacts.
- State serialization is for migration/debugging only; live sessions keep state resident.

## 10. Artifact manifest

```json
{
  "family": "ripple-vc",
  "model_version": "0.1.0",
  "rif_version": 1,
  "speaker_profile_version": 1,
  "sample_rate": 24000,
  "chunk_samples": 480,
  "lookahead_samples": 0,
  "backend": "onnxruntime-cpu",
  "precision": "int8-mixed",
  "state_tensors": [],
  "files": [],
  "checksums": {},
  "quality_gate_report": "gate-report.json"
}
```

The final schema should include exact state names, dtypes, shapes, layouts, reset policy, license, minimum runtime version, and hardware compatibility.

## 11. Fallback policy

1. Select the best installed specialist artifact.
2. Run a startup numerical and RTF smoke test.
3. If it fails, demote to the ONNX Runtime tier.
4. If a quantized health canary fails, demote precision.
5. Never change model/precision mid-utterance without reset and cross-fade.
6. Record backend selection and failure reason without recording audio.

## 12. Optimization order

1. Remove recomputation with correct caches.
2. Fix allocations and graph partitions.
3. Tune channels/strides and operator layouts.
4. Fuse graph operations.
5. Apply FP16/BF16.
6. Apply INT8 PTQ.
7. Use QAT where needed.
8. Try selective INT4.
9. Add custom kernels only for demonstrated hotspots.

Model compression before a correct streaming cache can optimize the wrong bottleneck.

## 13. Deployment risks

- Dynamic shapes causing CPU fallback on mobile NPUs.
- Mamba/selective-scan export becoming a slow loop.
- INT8 pitch or state drift accumulating over long sessions.
- Depthwise convolution performing poorly on a target backend.
- An apparently single graph partitioning across CPU/NPU every frame.
- Different resamplers or audio normalization across applications.
- Platform-specific engine artifacts being treated as portable.
- Licenses of teacher/checkpoint dependencies contaminating shipped weights.
