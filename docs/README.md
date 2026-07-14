# Ripple research and architecture dossier

Status: research proposal, July 2026  
Scope: a new streaming speech-to-speech (S2S) voice-conversion model family, not a StreamVC reimplementation.

## Executive decision

Ripple should be built around a small, causal, stateful waveform path rather than an autoregressive speech language model or an offline diffusion/flow model. The proposed v1 combines:

- a 24 kHz causal convolutional analysis encoder;
- a low-rate, bounded-state content mixer using only exportable convolution, matrix multiplication, normalization, and fixed-window attention operators;
- a speaker-normalized causal prosody lane;
- global and token-level target-speaker conditioning computed outside the per-chunk hot path;
- a causal source-filter waveform decoder with artifact-resistant upsampling;
- offline distillation from modern multilingual speech encoders;
- explicit state tensors and fixed 20 ms chunks for deployment;
- a versioned semantic/prosodic interface that a future text encoder can drive without replacing the decoder.

This is a set of testable design hypotheses, not a claim of achieved superiority. Ripple only earns “better than StreamVC” status after passing the benchmark gates in [07-evaluation-and-benchmarks.md](07-evaluation-and-benchmarks.md).

## Proposed v1 targets

- 24 kHz mono output.
- 20 ms input/output cadence.
- Zero future-lookahead default; optional 10–20 ms quality mode.
- End-to-end target: p95 below 40 ms on the reference mobile CPU, including one 20 ms frame and compute.
- Sustained real-time factor below 0.5 on one performance CPU core.
- Bounded state with no memory growth during an eight-hour stream.
- 25–35 million parameters including the enrollment encoder; a smaller hot path after speaker enrollment.
- FP16/BF16 reference, INT8 production tier, selective INT4 only for sufficiently large matrix layers.
- One bundled artifact per backend where practical, with model, state schema, DSP constants, speaker-profile schema, and metadata.

Targets are acceptance criteria and must not be reported as measured results until implemented and reproduced.

## Document map

1. [Research method and StreamVC audit](01-research-method-and-streamvc-audit.md)
2. [Post-StreamVC state of the art](02-state-of-the-art-review.md)
3. [Ripple model architecture](03-ripple-architecture.md)
4. [Data, training, and losses](04-training-data-and-losses.md)
5. [Streaming algorithm and inference engine](05-streaming-and-inference.md)
6. [Deployment, quantization, and export](06-deployment-quantization-and-export.md)
7. [Evaluation and benchmarks](07-evaluation-and-benchmarks.md)
8. [Clean project architecture](08-project-structure.md)
9. [Implementation roadmap](09-implementation-roadmap.md)
10. [Text encoder, TTS, and future research](10-tts-extension-and-future-research.md)
11. [References](REFERENCES.md)

## Non-goals for v1

- A general conversational agent, speech LLM, or simultaneous translation model.
- Autoregressive codec-token generation in the live VC path.
- Multi-step diffusion or flow-matching inference in the live VC path.
- Reusing GPL, non-commercial, research-only, or provenance-unclear weights in a commercial artifact.
- Claiming that TTS first-packet latency is equivalent to continuous S2S latency.
- Shipping HuBERT, WavLM, Whisper, or another large teacher in the edge runtime.

## Evidence policy

The dossier distinguishes:

- peer-reviewed or archival evidence;
- primary preprints and official technical reports;
- official repositories/model cards;
- vendor-reported results that need independent reproduction.

Comparisons are directional because published systems use different data, hardware, sampling rates, metrics, and latency definitions. A paper’s RTF or MOS is not treated as a matched benchmark.

