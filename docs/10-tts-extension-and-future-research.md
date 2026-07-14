# Text encoder, TTS scaling path, and future research

## 1. The architectural guarantee

Ripple v1 must not merely “leave room” for TTS. It must freeze a concrete interface:

```text
audio frontend -> RIF-1 semantic + prosody -> shared decoder
text frontend  -> RIF-1 semantic + prosody -> shared decoder
```

The VC artifact does not contain a text encoder. A later Ripple-TTS bundle may package both frontends and select one at runtime, while reusing the speaker profile and decoder.

## 2. Why the shared interface is credible

- StreamVC and Soft-VC show that supervised soft semantic units can drive waveform synthesis.
- CosyVoice shows that supervised semantic tokens can bridge text and speech generation.
- X-Codec/SpeechTokenizer show the importance of semantic supervision over acoustic-only codec tokens.
- Moshi/CSM/Fish demonstrate semantic-first followed by acoustic-detail generation, though at larger scale.

Ripple uses a continuous soft distribution and expected embedding rather than committing v1 to a large AR codec-token stack.

## 3. Future text frontend

### Input layer

- Unicode normalization and language identification.
- Grapheme/byte fallback for broad coverage.
- Optional phoneme/G2P path for supported languages.
- Explicit punctuation, emphasis, and speaking-style controls.

### Text encoder

A compact multilingual Transformer/Conformer encoder produces contextual text states. It can be distilled from a larger TTS/text-speech teacher. It does not need to run at audio cadence.

### Duration/alignment

A monotonic duration model expands text states to 50 Hz RIF frames. Candidate approaches:

- supervised alignments from ASR/forced alignment;
- monotonic alignment search;
- duration prediction with uncertainty;
- chunk-aware incremental alignment for streaming text.

Duration is explicit because VC obtains timing from source audio while TTS must create it.

### Semantic predictor

Predict `semantic_soft[t]` over the same RIF classes. Training uses:

- cross-entropy/KL to audio-derived RIF;
- sequence-level ASR consistency;
- hard/soft token curriculum;
- scheduled exposure to predicted rather than teacher semantic frames.

### Prosody planner

Predict normalized F0, voicing, energy, and optional style trajectories from text, speaker profile, and style instructions.

Two tiers:

- lightweight deterministic planner for edge TTS;
- richer chunk-aware autoregressive or one-step-flow planner for server quality.

Both emit the same RIF prosody tensor.

## 4. Streaming TTS behavior

Continuous S2S and streaming TTS have different latency:

- VC can emit after one source frame because timing/content already exist.
- TTS needs enough text context to decide pronunciation, duration, and prosody.

Ripple-TTS should report:

- text-ingestion delay;
- planning delay;
- first audio packet;
- steady RTF.

It must not reuse Ripple-VC’s sub-40 ms target as a misleading TTS claim.

For incremental text:

1. buffer to a safe linguistic boundary;
2. plan a short RIF segment;
3. emit through the shared decoder;
4. revise only uncommitted future frames;
5. carry bounded prosody state between segments.

CosyVoice-style chunk-aware training is the preferred reference. F5-TTS/MaskGCT-style full-sequence generation is a quality teacher, not the streaming implementation.

## 5. Protecting VC while adding TTS

- Freeze RIF-1, speaker profile, and decoder for the first TTS milestone.
- Train only text/duration/prosody modules.
- If decoder tuning becomes necessary, create RIF-2 or a new decoder version.
- Run the full VC regression suite on every shared-decoder change.
- Keep VC and TTS artifacts independently deployable.
- Do not increase VC runtime dependencies because TTS uses an LLM or tokenizer.

## 6. Optional discrete interface

An auxiliary FSQ or hard semantic representation may help text prediction and compression.

Research plan:

- train hard/FSQ targets beside soft RIF;
- decode from soft, hard, and mixed inputs;
- use semantic dropout so the decoder handles text-like distributions;
- compare WER, naturalness, leakage, and TTS learnability;
- version any accepted change.

The acoustic waveform decoder remains continuous. Full acoustic RVQ is unnecessary unless a later speech LM requires it.

## 7. Expressive and emotional speech

Add a separate style contract rather than contaminating speaker identity:

- global style/emotion embedding;
- optional low-rate local style trajectory;
- source-preserve, target-style, and text-instructed modes;
- independent speaker and emotion losses.

Evaluation must distinguish:

- target identity;
- source local prosody;
- intended emotion;
- unintended identity leakage.

## 8. Multilingual expansion

- Add languages by extending teacher coverage and paired text data.
- Keep a universal RIF where possible; add language-conditioned normalization only when evidence requires it.
- Evaluate lexical tone explicitly.
- Use grapheme/byte fallback but retain phoneme paths where pronunciation quality justifies them.
- Treat accent as a controllable attribute, not an accidental speaker feature.

## 9. Future model variants

### Fused state-space mixer

Test Mamba/RWKV only with a production fused recurrent step on every target backend. Acceptance requires better quality or state size at equal wall-clock latency, including INT8 and long streams.

### One-step flow quality decoder

A one-step consistency/flow decoder may improve server quality while preserving a shared RIF. It is a separate tier, never silently inserted into edge mode.

### Semantic/acoustic depth decoder

For server TTS/dialogue, a Moshi/CSM/Fish-style depth decoder could generate acoustic detail from low-rate semantics. This changes the runtime class and should remain outside continuous edge VC.

### Simultaneous translation

Hibiki’s multistream/adaptive-delay method is the relevant reference. Translation changes content and timing, so it should become a separate Ripple-S2ST family using RIF/decoder components, not a mode mislabeled as voice conversion.

### Enhancement front-end

Optional denoising/echo control can improve conversion under calls. It must be benchmarked for latency and speaker/prosody damage; dual-path networks are likely server/high-quality options rather than edge defaults.

### Singing

Singing requires wider pitch range, vibrato accuracy, long voiced states, breath/noise modeling, and music leakage tests. It should be a separately trained/evaluated variant.

## 10. Research questions in priority order

1. Which multilingual teacher target gives the best intelligibility/leakage Pareto front?
2. Does learned causal prosody remove all need for lookahead?
3. How many cached speaker tokens improve similarity before cost/leakage dominates?
4. Does the source-filter decoder preserve high pitch and cross-gender conversion under INT8?
5. Does a bounded memory block improve hour-long stability over pure convolution?
6. Can text predict soft RIF accurately enough without a new acoustic model?
7. Does auxiliary FSQ improve TTS while preserving VC quality?
8. Can watermarking survive Opus, resampling, and adversarial removal without audible cost?
9. Which state tensors are most sensitive to quantization drift?
10. Can a fused SSM beat fixed-window attention on real edge hardware?

## 11. Future roadmap outcomes

Near term:

- proven zero-lookahead VC;
- multilingual RIF;
- INT8 edge runtime;
- robust speaker enrollment.

Medium term:

- Ripple-TTS text frontend;
- expressive style controls;
- server quality tier;
- stronger anonymization mode.

Long term:

- simultaneous translation family;
- duplex conversational integration;
- personalized on-device adaptation with privacy-preserving training;
- codec/token interfaces for speech-language models.

The long-term family should grow by reusing contracts and validated components, not by turning the v1 edge graph into a monolith.
