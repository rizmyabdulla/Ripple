# Research method and StreamVC audit

## 1. Review method

This is a critical narrative review covering work available through July 2026. It prioritizes primary papers, proceedings, official repositories, model cards, and runtime documentation. Search themes included streaming voice conversion, causal speech synthesis, neural codecs, semantic/acoustic tokenization, state-space and chunked sequence models, speaker disentanglement, prosody, multilingual speech encoders, quantization, and edge runtimes.

Inclusion criteria:

- Direct relevance to voice conversion, streaming speech generation, neural audio codecs, or deployment.
- Primarily 2023–2026, with older foundational work retained when still architecturally important.
- A reproducible architecture or sufficiently detailed method.
- Measured speech quality, intelligibility, latency, or deployment evidence where available.

Exclusion or down-weighting criteria:

- Product announcements without an architecture or evaluation.
- “Streaming” that only means sentence-level TTS chunking.
- Offline models presented as low-latency merely because their GPU RTF is below one.
- Comparisons that omit algorithmic lookahead, device, precision, or chunk size.

No matched reimplementation study was performed. Conclusions about expected performance are therefore hypotheses to validate, not measured Ripple results.

## 2. What StreamVC established

[StreamVC](https://arxiv.org/abs/2401.03078) is an ICASSP 2024, approximately 20M-parameter, any-to-any voice-conversion system derived from SoundStream. At 16 kHz it processes one 320-sample/20 ms frame at 50 Hz.

Its canonical data flow is:

1. A causal SoundStream-style content encoder with scale 64 produces a 64-dimensional soft content latent.
2. HuBERT-Base layer-7 activations clustered into 100 classes supervise the content encoder.
3. Decoder gradients are stopped at the content latent to reduce source-speaker leakage.
4. A smaller SoundStream-style speaker encoder and learnable attention pooling produce a global 64-dimensional target-speaker embedding.
5. Three YIN thresholds produce nine pitch/periodicity/voicing values per frame; frame energy adds one channel.
6. Pitch is whitened using utterance statistics offline and running statistics online.
7. A causal SoundStream-style decoder, conditioned through FiLM, reconstructs waveform audio.
8. Training combines content cross-entropy, adversarial, feature-matching, and multi-scale spectral reconstruction losses.

The paper reports 60 ms architectural delay from output alignment and the three-frame pitch window, plus 10.8 ms compute per chunk on a single Pixel 7 CPU core with XNNPACK, for 70.8 ms end-to-end latency. This is device- and backend-specific, not a universal constant.

## 3. Baseline strengths

- It treats streaming as a training and state-management property, not an inference wrapper.
- Its convolutional state is fixed-size and naturally exportable.
- The teacher is only needed during training.
- Explicit pitch restored prosody that content units did not preserve.
- Gradient isolation and pitch whitening directly address source-timbre leakage.
- Its reported mobile result is more relevant to edge deployment than GPU-only RTF.

The paper’s ablation is especially important: removing F0 reduced F0 correlation from 0.842 to 0.461, while removing whitening also harmed pitch and speaker metrics. Explicit, speaker-normalized prosody should therefore remain a v1 requirement unless a stronger causal representation wins a matched ablation.

## 4. Baseline limitations

- 16 kHz limits high-frequency speech quality.
- A 60 ms architecture delay is no longer an aggressive target.
- A single pooled speaker vector loses phonation, channel, and fine-grained timbre detail.
- YIN depends on a future frame and can be brittle under noise, overlap, and unvoiced transitions.
- English-centric HuBERT/k-means targets do not establish multilingual robustness.
- Same-utterance reconstruction can leave a training/inference gap for unseen source/target pairs.
- Published evaluation used clean LibriTTS/VCTK conditions and did not establish packet-loss, long-call, thermal, or quantized-runtime stability.
- Speaker similarity improved after VCTK fine-tuning while F0 consistency declined, exposing a timbre/prosody trade-off.
- The official model code and weights were not released.

## 5. Audit of the two local repositories

### `C:\Users\user\Documents\StreamVC\StreamVC`

This is the closer architectural sketch, but its README explicitly says it is unofficial, has no checkpoint, and does not fully implement streaming.

Useful matches:

- Correct encoder/decoder scales and 74-channel decoder input.
- SoundStream strides `(2, 4, 5, 8)`.
- Gradient-stopped content path.
- Three-frame pitch window.
- Two-frame output trimming in decoder training.
- Two-stage content-encoder then decoder training.

Material issues:

- `streamvc/model.py` passes a YIN threshold of `1.5` instead of the paper’s `0.15`.
- Streaming buffers cover causal Conv1d but not a complete, verified graph.
- Streaming pitch whitening does not implement the paper’s running statistics.
- SoftVC `hubert_discrete` is used instead of reproducing HuBERT layer 7 plus LibriTTS k-means.
- The discriminator stack is not a full reproduction of SoundStream.

### `C:\Users\user\Documents\StreamVC\stream-vc`

This is a WIP Lightning/Hydra experiment harness, not a faithful streaming implementation.

Useful elements:

- Explicit learned-query speaker pooling.
- FiLM applied at a finer block granularity.
- A structured configuration and testing scaffold.

Material issues:

- Local attention is enabled by default although it is not in the paper.
- Content probabilities are passed to `CrossEntropyLoss`, which expects logits.
- Energy uses a 64 ms sum-of-squares window instead of 20 ms variance.
- No complete stateful streaming path was found.
- Optional GateLoop and local-attention additions are not paper-backed StreamVC behavior.

Neither repository is an authoritative baseline. The paper, SoundStream, and Soft-VC should be the baseline references; local code is useful only as an implementation study with known defects.

## 6. Assumptions Ripple retains

- Causal, fixed-cadence source processing.
- Offline or infrequent target-speaker enrollment.
- Training-only large teachers and a compact causal student.
- A hard separation between semantic content, source prosody, and target timbre.
- Gradient isolation plus explicit leakage probes.
- Speaker-normalized pitch/energy features.
- Direct waveform synthesis with adversarial and spectral objectives.
- Explicit, versioned streaming state.

## 7. Assumptions Ripple rejects

- 16 kHz as the only quality tier.
- A single target-speaker vector as sufficient conditioning.
- A future-dependent YIN window as mandatory.
- English HuBERT clusters as a universal semantic interface.
- Transposed convolution as the default upsampler.
- Mobile latency measured on one device as proof of broad edge performance.
- Adding a large Transformer, Mamba, diffusion model, or speech LM merely because its asymptotic complexity or offline quality is attractive.
- Treating either local repository as production-ready streaming code.

## 8. Research gaps Ripple must close experimentally

- A matched multilingual teacher bake-off under the same causal student budget.
- Soft continuous units versus discrete/FSQ units for leakage, TTS predictability, and quality.
- Single-vector versus token-level speaker conditioning at fixed compute.
- Learned causal pitch/prosody versus YIN/RMVPE-derived features.
- Pure convolution versus a small bounded-memory latent mixer.
- Direct waveform decoder variants under INT8.
- Quality and state drift over multi-hour streams.
- Whether an optional 10–20 ms lookahead produces enough quality gain to justify a second mode.

