# Streaming algorithm and inference engine

## 1. Streaming contract

The inference API accepts fixed 20 ms PCM chunks at 24 kHz: 480 mono samples. Fixed chunks avoid dynamic-shape fallbacks and make latency reproducible.

Every call returns:

- zero or one 480-sample output chunk;
- updated explicit state;
- optional diagnostics such as voicing, compute time, underflow, and state-health flags.

Edge mode uses no future audio. Quality mode is a separate artifact that buffers one frame.

## 2. State schema

State is versioned and bounded:

- analysis-encoder convolution rings;
- RippleMixer convolution rings;
- local-attention K/V ring;
- optional summary-memory vectors;
- prosody estimator history;
- voiced-frame running mean/variance/count;
- oscillator phase;
- decoder upsampling/filter rings;
- limiter/DC-block state;
- packet/jitter counters.

State tensors are explicit graph inputs/outputs for the portable ONNX tier. Backends with native mutable state may internalize them, but they must preserve the same schema and reset semantics.

No state may grow with conversation length.

## 3. Session lifecycle

1. Load artifact and validate manifest/checksum.
2. Load or compute target speaker profile.
3. Allocate all state and output buffers from an arena.
4. Warm up the graph with deterministic silence.
5. Begin 20 ms push cadence.
6. On temporary packet loss, use the configured concealment policy.
7. On endpoint, either retain conversational state or perform a soft reset.
8. On target-speaker change, reset decoder modulation and optionally cross-fade.
9. Flush the finite decoder tail.
10. Destroy the session and release the arena.

## 4. Per-chunk algorithm

```text
push(pcm_480):
  x = input_conditioner(pcm_480)
  semantic, analysis_state = analysis_encoder.step(x, analysis_state)
  semantic, mixer_state = ripple_mixer.step(semantic, mixer_state)

  raw_prosody, prosody_state = prosody_encoder.step(x, prosody_state)
  prosody, stats_state = normalize_and_gate(raw_prosody, stats_state)

  excitation, phase_state = harmonic_noise_source(prosody, phase_state)
  y, decoder_state = decoder.step(
      semantic,
      prosody,
      speaker_profile,
      excitation,
      decoder_state
  )

  y, post_state = post_process(y, post_state)
  return y, all_updated_states
```

The production implementation must avoid host round-trips between these steps where backend fusion can keep tensors resident.

## 5. Startup behavior

The first frames are a common failure point because pitch statistics and convolution history are empty.

Policy:

- initialize convolution history to silence;
- initialize F0 normalization from population priors conditioned on detected voice range only when confidence is sufficient;
- update statistics only on high-confidence voiced frames;
- blend from prior to observed statistics over a bounded warm-up;
- suppress harmonic excitation during uncertain onset;
- train explicitly on session starts.

Time to first audio is one frame plus compute in edge mode. Any additional warm-up buffering is a bug unless explicitly selected as quality mode.

## 6. Silence and endpoint policy

Hard-resetting on every VAD boundary can produce voice and phase discontinuities. Never resetting can preserve drift.

Use two reset types:

- **Soft reset:** retain target profile and robust pitch prior; decay mixer memory, oscillator, and running statistics over a short silence.
- **Hard reset:** clear all source-derived state when the application changes source speaker, target profile, sample rate, or detects corruption.

Training and benchmarks must cover both.

## 7. Packet loss and jitter

Host-side jitter buffering is limited by the latency SLO. Policies:

- one missing frame: repeat/attenuate semantic state, continue oscillator phase, decay energy;
- multiple missing frames: fade to shaped comfort noise and freeze running pitch updates;
- late frame: drop rather than rewind model state;
- recovery: cross-fade one frame and gate statistic updates.

The model should receive packet-loss masks during training so concealment is not entirely external.

## 8. Speaker enrollment path

Enrollment is not part of steady-state latency.

Options:

- compute a profile from a 3–10 second reference inside the bundled artifact;
- load a signed, precomputed profile;
- average several consented references.

Profiles must be portable only across artifacts with the same profile schema/version. A profile is biometric and should be encrypted at rest.

## 9. C ABI

```c
typedef struct RippleSession RippleSession;

typedef struct {
  uint32_t sample_rate;
  uint32_t chunk_samples;
  uint32_t flags;
  uint32_t backend_preference;
} RippleConfig;

typedef struct {
  const float *pcm;
  uint32_t samples;
  uint32_t packet_status;
  uint64_t timestamp_ns;
} RippleInput;

typedef struct {
  float *pcm;
  uint32_t capacity;
  uint32_t produced;
  uint32_t health_flags;
  float compute_ms;
} RippleOutput;

int ripple_create(
  const char *artifact_path,
  const RippleConfig *config,
  RippleSession **session
);

int ripple_load_speaker_profile(
  RippleSession *session,
  const void *profile,
  size_t profile_bytes
);

int ripple_push(
  RippleSession *session,
  const RippleInput *input,
  RippleOutput *output
);

int ripple_soft_reset(RippleSession *session);
int ripple_hard_reset(RippleSession *session);
int ripple_flush(RippleSession *session, RippleOutput *output);
void ripple_destroy(RippleSession *session);
```

The C ABI isolates application code from ONNX Runtime, Core ML, LiteRT, ExecuTorch, or custom backend details.

## 10. Runtime architecture

### Frontend

- audio format validation;
- resampling only when unavoidable;
- DC removal and bounded gain normalization;
- jitter/packet metadata;
- no heavyweight Python dependency.

### Graph executor

- preallocated input/output/state tensors;
- backend-specific session;
- fixed thread count and affinity policy;
- no allocation in `push`;
- deterministic error/fallback handling.

### Profile manager

- enrollment graph;
- profile versioning and encryption hooks;
- consent/identity metadata outside the neural tensor payload.

### Telemetry

- p50/p95/p99 compute;
- underruns and packet loss;
- state norms/NaN checks;
- memory high-water mark;
- backend and precision tier;
- no raw audio logging by default.

## 11. Efficient batching

Interactive edge use is batch one. Server batching is optional:

- batch only sessions with the same artifact, chunk size, and state shapes;
- never pad an unbounded history;
- use deadline-aware microbatches;
- fall back to immediate execution as deadlines approach;
- keep each session’s state contiguous.

Throughput batching must not redefine the latency SLO.

## 12. Long-session stability

Engineering requirements:

- state norms remain bounded;
- no running-statistic overflow;
- no cumulative phase discontinuity;
- no RSS growth;
- stable output after hours of silence/speech transitions;
- deterministic reset behavior;
- no quality dependence on chunk-call grouping.

Tests should compare:

- one call per frame;
- grouped calls that are internally split to frames;
- random packet loss;
- repeated soft resets;
- an eight-hour prerecorded stream.

## 13. Single-artifact packaging

Each backend bundle contains:

- source/prosody/speaker/decoder graph or one fused graph;
- weights;
- state schema;
- RIF/profile schema;
- sample-rate and chunk constants;
- quantization scales;
- watermark configuration where enabled;
- tokenizer/semantic metadata needed only for diagnostics or TTS;
- license and data/model card;
- checksums and compatibility version.

“Single artifact” means one deployable bundle, not necessarily one tensor graph. Splitting enrollment and steady-state graphs inside the bundle can reduce hot-path memory and preserve integration simplicity.

