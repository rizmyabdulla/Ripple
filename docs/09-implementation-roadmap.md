# Complete implementation roadmap

Each phase has an exit gate. Later work should not hide a failed earlier assumption.

## Phase 0: reproducibility and baseline

**Objectives**

- Establish trustworthy StreamVC and simple causal-autoencoder baselines.

**Research tasks**

- Reconcile paper details with both unofficial repositories.
- Reproduce HuBERT labels, latency accounting, losses, and metrics.

**Implementation tasks**

- Create repository skeleton, manifests, streaming cache primitives, and benchmark harness.
- Implement a paper-faithful StreamVC baseline without copying known repository defects.

**Training requirements**

- LibriTTS-scale pilot; 16 kHz; staged content and decoder training.

**Evaluation metrics**

- WER/CER, speaker similarity, F0 correlation, DNSMOS/listening sample, full versus streamed equivalence, CPU RTF.

**Expected outputs**

- Baseline checkpoint, artifact, report, and known-gap register.

**Risks**

- No official weights/code; exact paper replication may be impossible.

**Optimization opportunities**

- Profile causal cache correctness before architecture work.

**Exit gate**

- Stable streaming baseline with explained deviation from paper metrics.

## Phase 1: data and evaluation foundation

**Objectives**

- Build licensed, multilingual, speaker-disjoint data and immutable evaluation.

**Research tasks**

- Select languages/domains and consent policy.
- Audit teacher and dataset licenses.

**Implementation tasks**

- Manifest, quality filters, feature store, split generator, and metric adapters.

**Training requirements**

- None beyond probe models.

**Evaluation metrics**

- Coverage, hours, speakers, language balance, duplicate rate, filter acceptance, provenance completeness.

**Expected outputs**

- Dataset cards, manifests, consent/opt-out process, benchmark v1.

**Risks**

- Licensing removes high-volume datasets; metadata bias.

**Optimization opportunities**

- Cache features in sharded, memory-mappable format.

**Exit gate**

- Every training sample has provenance and every test speaker is disjoint.

## Phase 2: RIF-1 semantic and prosody interfaces

**Objectives**

- Select a multilingual, low-leakage semantic target and a zero-lookahead prosody lane.

**Research tasks**

- Equal-budget HuBERT, WavLM, mHuBERT/XLS-R, Whisper/ASR, and supervised-token comparison.
- YIN versus learned causal pitch study.

**Implementation tasks**

- Teacher adapters, projection/soft-unit target, causal analysis student, prosody estimator, running statistics.

**Training requirements**

- Several hundred to thousands of multilingual hours for student pilots.

**Evaluation metrics**

- Phoneme/WER, source-speaker probe, F0/voicing, cross-language transfer, latency, state drift.

**Expected outputs**

- Versioned RIF-1 schema and selected semantic/prosody checkpoints.

**Risks**

- Best intelligibility representation may leak speaker.

**Optimization opportunities**

- Distill several teacher layers into one compact target.

**Exit gate**

- Causal student meets semantic/leakage gate and prosody beats or matches streaming YIN without future context.

## Phase 3: speaker enrollment and disentanglement

**Objectives**

- Robust unseen-speaker identity with fine timbre detail and bounded conditioning.

**Research tasks**

- ECAPA versus CAM++-like encoder.
- Global-only versus global plus 4/8 tokens.
- Timbre-shift versus anonymization perturbation.

**Implementation tasks**

- Speaker-profile schema, enrollment graph, profile cache, identity/style losses, leakage probes.

**Training requirements**

- Broad multi-speaker data with cross-crop/channel augmentation.

**Evaluation metrics**

- Speaker verification, cross-language enrollment, token diversity, source leakage, profile duration sensitivity.

**Expected outputs**

- Speaker-profile v1 and enrollment artifact.

**Risks**

- Style tokens reintroduce source content or overfit channel.

**Optimization opportunities**

- Preproject all profile modulation values.

**Exit gate**

- Token conditioning improves human/automatic similarity at acceptable cost and leakage.

## Phase 4: causal waveform decoder

**Objectives**

- 24 kHz high-quality direct waveform synthesis with 20 ms cadence.

**Research tasks**

- Resize-conv versus subpixel upsampling.
- Source-filter versus learned-only excitation.
- Discriminator and spectral-loss ablations.

**Implementation tasks**

- Decoder, oscillator/noise source, causal upsamplers, discriminators, chunk-boundary tests.

**Training requirements**

- Reconstruction training followed by adversarial fine-tuning; diverse 24 kHz data.

**Evaluation metrics**

- MOS/MUSHRA, spectral metrics, F0, boundary artifacts, RTF, memory.

**Expected outputs**

- FP16 decoder and architecture ablation.

**Risks**

- Adversarial instability; oscillator buzz; high-band hallucination.

**Optimization opportunities**

- Depthwise/dense kernel search, precomputed modulation, block fusion.

**Exit gate**

- 24 kHz quality gain over baseline while maintaining real-time streaming.

## Phase 5: integrated Ripple-VC Edge

**Objectives**

- End-to-end any-to-any zero-lookahead conversion.

**Research tasks**

- Pure convolution versus RippleMixer/Emformer.
- Gradient-stop schedule and perturbation mix.

**Implementation tasks**

- Integrate graph and state, mismatch training, packet masks, session API.

**Training requirements**

- Full curriculum through long continuous sequences.

**Evaluation metrics**

- All core quality, speaker, prosody, leakage, latency, and long-session metrics.

**Expected outputs**

- First complete FP16 model, Python streaming demo, failure taxonomy.

**Risks**

- Component metrics fail to translate end to end; target similarity/prosody trade-off.

**Optimization opportunities**

- Prune or distill mixer, reduce token bank, tune channels by component timing.

**Exit gate**

- Matched improvement over the reproduced baseline on a pre-registered Pareto criterion.

## Phase 6: native runtime and portable export

**Objectives**

- Stable C ABI and ONNX Runtime production reference.

**Research tasks**

- Backend operator/layout profiling and state representation.

**Implementation tasks**

- Fixed-shape ONNX export, explicit state, arena allocation, C/C++ runtime, artifact manifest, fallback.

**Training requirements**

- Export-aware fine-tuning only if numeric differences require it.

**Evaluation metrics**

- PyTorch/ONNX conformance, p95/p99 latency, RSS, startup, eight-hour stability.

**Expected outputs**

- Signed ONNX bundle, native SDK, example app.

**Risks**

- Unsupported operators, hidden allocations, graph partitioning.

**Optimization opportunities**

- Offline graph optimization, state layout, fusion, memory mapping.

**Exit gate**

- Portable runtime passes conformance and misses no steady-state deadlines.

## Phase 7: quantization and specialist backends

**Objectives**

- INT8 edge tier and optimized platform artifacts.

**Research tasks**

- Layer sensitivity, PTQ versus QAT, selective INT4 benefit.

**Implementation tasks**

- Streaming calibration, fake quantization, TensorRT/Core ML/LiteRT/ExecuTorch exports.

**Training requirements**

- QAT on representative streaming data if PTQ misses gates.

**Evaluation metrics**

- Quality deltas, state drift, p95 latency, RSS, energy, thermal behavior.

**Expected outputs**

- INT8 canonical artifact and platform-specific bundles.

**Risks**

- Prosody/state error accumulation; NPU CPU fallback.

**Optimization opportunities**

- Mixed precision and platform-specific channel/layout tuning.

**Exit gate**

- At least one quantized edge artifact materially improves deployment cost within all quality gates.

## Phase 8: production hardening

**Objectives**

- Robustness, observability, safety, and integration readiness.

**Research tasks**

- Watermark robustness, abuse controls, anonymization mode, domain adaptation.

**Implementation tasks**

- Telemetry, profile encryption, packet recovery, fuzzing, SDKs, model/data/safety cards.

**Training requirements**

- Target-domain and packet-loss fine-tuning; no unreviewed identity data.

**Evaluation metrics**

- Crash/fuzz results, packet recovery, watermark detection, abuse-control tests, field-device matrix.

**Expected outputs**

- Release candidate and operational runbooks.

**Risks**

- Safety controls reduce quality; device fragmentation.

**Optimization opportunities**

- Device capability detection and artifact selection.

**Exit gate**

- Security, privacy, legal, quality, and SRE sign-off.

## Phase 9: text encoder and Ripple-TTS

**Objectives**

- Enable TTS by driving RIF-1 without changing the VC decoder.

**Research tasks**

- Phoneme/byte/grapheme input, duration model, semantic distribution prediction, streaming prosody planning.
- Supervised semantic tokens versus RIF distillation.

**Implementation tasks**

- Text normalizer/G2P, multilingual text encoder, monotonic duration predictor, prosody planner, chunk scheduler.

**Training requirements**

- Licensed paired text/speech; freeze decoder first; later limited joint tuning with VC regression tests.

**Evaluation metrics**

- TTS WER/CER, MOS, speaker similarity, prosody, first packet, RTF, and unchanged VC metrics.

**Expected outputs**

- Ripple-TTS model/artifact sharing the RIF/profile/decoder contracts.

**Risks**

- Text-predicted semantic distributions differ from audio student; duration errors; joint tuning regresses VC.

**Optimization opportunities**

- Distill a large text/speech teacher; separate text planner artifact from decoder.

**Exit gate**

- TTS meets its gates and the original Ripple-VC artifact remains bit-for-bit unaffected.

## Phase 10: model-family research

**Objectives**

- Explore improvements only after a stable production baseline.

**Research tasks**

- Fused Mamba/RWKV mixer, FSQ semantic interface, one-step flow quality tier, expressive style, simultaneous translation.

**Implementation tasks**

- Isolated branches and matched benchmark adapters.

**Training requirements**

- Depends on hypothesis; no production merge before complete ablation.

**Evaluation metrics**

- Full Pareto evaluation, not paper-specific metrics alone.

**Expected outputs**

- Accepted/rejected architecture decision records.

**Risks**

- Research complexity displaces reliability.

**Optimization opportunities**

- Promote only techniques with measured end-to-end benefit.

**Exit gate**

- A new variant beats the existing production Pareto front under matched conditions.

