# Post-StreamVC state-of-the-art review

## 1. The field split after StreamVC

The literature is easier to reason about when separated into three tasks:

1. **Continuous voice conversion/S2S:** source audio is preserved linguistically and temporally while timbre changes.
2. **TTS and zero-shot speech generation:** text or semantic tokens generate new timing and acoustics.
3. **Speech dialogue and simultaneous translation:** a language model may alter content and timing while listening and speaking.

Only the first category is the Ripple v1 task. Techniques from the other categories are useful only when they preserve continuous cadence, bounded state, and edge cost.

## 2. Streaming voice conversion

### StreamVC

StreamVC remains the strongest peer-reviewed mobile reference because it publishes causal framing and on-device compute. Its weaknesses are the 60 ms lookahead, 16 kHz output, simple speaker conditioning, and narrow evaluation.

### Seed-VC

[Seed-VC](https://arxiv.org/abs/2411.09943) uses an external timbre shifter during training and a diffusion Transformer with full-reference in-context conditioning. It directly addresses train/inference mismatch and speaker leakage and reports stronger zero-shot quality than OpenVoice and CosyVoice in its setting. Those ideas transfer; the multi-step flow/diffusion path and roughly hundreds-of-milliseconds real-time mode do not satisfy Ripple’s edge latency target. Its repository licensing also requires careful product review.

### Conan and Zero-VC

[Conan](https://arxiv.org/abs/2507.14534) uses an Emformer content extractor, adaptive style tokens, and a strictly causal shuffle vocoder. It reports a 37 ms fast mode on an A100, not a mobile CPU, so the architectural idea is stronger evidence than the device comparison.

[Zero-VC](https://arxiv.org/abs/2606.20218) is a June 2026 preprint using speaker anonymization as a content-preserving timbre perturbation. It reports one-frame, zero-lookahead operation with 20 ms algorithmic latency. This is highly relevant but too recent to treat as independently established. Ripple should test its perturbation principle, not assume its results transfer.

### DualVC, StreamVoice, MeanVC, and related work

Dual-mode/chunk-trained systems show that training across multiple chunk sizes improves streaming robustness. StreamVoice-style semantic/acoustic language models improve contextual generation but are larger and slower than call-grade convolutional paths. Mean-flow and one-step flow models reduce diffusion cost, yet their chunk sizes and pipelines generally remain above StreamVC-class delay. They are teacher or server-tier candidates, not the v1 edge hot path.

## 3. Neural codecs and tokenizers

### SoundStream and EnCodec

SoundStream’s causal SEANet and residual-vector-quantization training remain a strong basis for waveform synthesis. [EnCodec](https://arxiv.org/abs/2210.13438) adds a widely used causal 24 kHz operating point. For VC, their continuous latents or decoders are more useful than treating acoustic codec tokens as speaker-free content.

### DAC

[DAC](https://arxiv.org/abs/2306.06546) improves RVQGAN using factorized codebooks, code normalization, Snake activations, multi-scale mel losses, and improved discriminators. It is a valuable quality teacher and training reference, but its common high-fidelity configuration is non-causal and not an edge-streaming drop-in.

### Mimi and Moshi

[Mimi](https://arxiv.org/abs/2410.00037) is a causal 24 kHz codec at 12.5 Hz and about 1.1 kbps. It distills semantics into the first RVQ level and supports Moshi’s full-duplex language model. An 80 ms codec frame is attractive for reducing autoregressive steps but too coarse for Ripple’s 20 ms control cadence. Transformer codec blocks also need profiling before edge adoption.

### SNAC

[SNAC](https://arxiv.org/abs/2410.14411) quantizes at multiple temporal resolutions. This is promising for speech-language-model sequence length, but its evidence is a workshop/preprint track and multi-rate state complicates call-grade VC. It is not selected for v1.

### SpeechTokenizer, X-Codec, X-Codec2, WavTokenizer, and BigCodec

[SpeechTokenizer](https://arxiv.org/abs/2308.16692) and [X-Codec](https://arxiv.org/abs/2408.17175) demonstrate that compression codecs have semantic weaknesses and that SSL distillation improves generation intelligibility. X-Codec is peer-reviewed at AAAI 2025; X-Codec2, WavTokenizer, and BigCodec mainly target single-stream speech-language-model tokens. They motivate Ripple’s semantic supervision but not its live codec.

### RVQ versus FSQ

RVQ offers variable bitrate and coarse-to-fine structure but adds codebook management and can trade full-rate quality against quantizer dropout. [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505) avoids codebook collapse and is export-friendly. Ripple v1 does not need an acoustic quantizer in the live path. An auxiliary FSQ semantic interface remains a research option for later TTS.

## 4. Streaming sequence backbones

### Causal convolution

Causal depthwise/separable convolution remains the best-supported waveform-rate operator:

- fixed, tiny ring buffers;
- mature INT8 kernels;
- straightforward ONNX, Core ML, LiteRT, and ExecuTorch export;
- stable batch-one CPU performance.

The largest practical gains come from lowering temporal resolution, caching, and fusing operators—not from asymptotic complexity claims.

### Chunk Conformer and FastConformer

[FastConformer](https://arxiv.org/abs/2305.05084) shows meaningful speedups from aggressive subsampling and efficient convolution. Cache-aware and dynamic-chunk Conformers provide fixed attention and convolution state with a train/inference match. At a 50 Hz latent, a 16–32-frame causal window is small enough that standard attention is practical.

### Emformer

[Emformer](https://arxiv.org/abs/2010.10759) provides explicit center/right context and a bounded memory bank. Its state model is suitable for long sessions and standard export operators. Conan provides direct VC evidence for an Emformer content extractor, although not mobile CPU evidence.

### Mamba and state-space models

[Mamba](https://arxiv.org/abs/2312.00752), streaming Mamba ASR, Speech-Mamba, ConMamba, and SMAM show useful long-context speech results. The limitation is deployment: selective scan often exports as a slow loop unless a fused custom kernel is shipped. Wall-clock benefits are inconsistent on short, batch-one chunks. Ripple therefore keeps Mamba as an experimental latent-mixer branch, not the portable default.

S4/structured-state-space hybrids support the broader lesson that convolution plus a compact long-context mechanism can beat replacing the whole stack with an SSM.

### RWKV

AudioRWKV and RWKV-TTS suggest constant-state recurrence with better scaling stability than some Mamba configurations. Evidence for low-latency VC and stock edge runtimes remains thin, and WKV still needs specialized kernels.

### Hyena

ConfHyena reports training efficiency for offline speech models, but FFT/long-convolution implementations are not naturally fixed-state, low-latency edge operators. It is rejected for the v1 live path.

### Linear attention and FlashAttention

Kernelized linear attention has not displaced fixed-window attention in production streaming speech. FlashAttention is an important GPU training/server kernel but does not reduce algorithmic latency and offers little benefit for very short mobile chunks.

### Dual-path architectures

Dual-path networks are strong for separation and enhancement. Their state and frequency-path cost are unnecessary for single-speaker conversion, and long-session work shows failure modes after extended silence unless trained on continuous streams. They belong in an optional front-end, not the core.

## 5. Content and speech representations

### HuBERT and WavLM

HuBERT soft units remain a strong content target; WavLM improves general speech representations but continuous WavLM features can retain source-speaker information. Both should remain offline teachers.

### mHuBERT and XLS-R

Multilingual HuBERT and XLS-R provide broader phonetic coverage than English HuBERT. Direct, matched streaming-VC evidence is limited, so Ripple should train equal-size students against several teachers and select by multilingual WER, leakage, and latency.

### Whisper

Whisper encoder features transfer cross-lingual and accent information and are used by systems such as Seed-VC. Whisper is too large and non-causal for the edge path but useful in an offline teacher ensemble. Accent retention must be treated as a controllable property because accent and speaker identity can be entangled.

### EAT, BEST-RQ, and newer SSL encoders

EAT and BEST-RQ are efficient or discrete representation candidates, but direct post-StreamVC evidence for causal VC and leakage is weak. They belong in teacher ablations rather than the baseline.

## 6. Speaker, style, and prosody

ECAPA-TDNN and CAM++-class speaker encoders remain practical enrollment models. A single vector is insufficient for fine phonation and channel details; Seed-VC and modern in-context TTS systems support adding a small bank of reference-derived style tokens.

Speaker embeddings often contain emotion and prosody. Ripple should separate:

- identity/timbre;
- source local prosody;
- optional target global style or emotion.

Whitened F0, voicing, periodicity, and energy are still the most direct controls. A learned causal estimator distilled from robust offline pitch models can remove StreamVC’s future frame. An NSF-style harmonic/noise source, mentioned in StreamVC’s later poster and common in RVC lineage, can improve pitch fidelity while keeping control explicit.

## 7. Modern TTS systems

### CosyVoice 2/3

[CosyVoice 2](https://arxiv.org/abs/2412.10117) combines supervised semantic tokens, an LLM, FSQ, and chunk-aware causal flow matching. Its approximately 150 ms first-packet claim is for streaming TTS, not continuous S2S. Its supervised token interface and chunk-aware training are the strongest open patterns for Ripple’s later text frontend.

### MaskGCT and F5-TTS

[MaskGCT](https://arxiv.org/abs/2409.00750) uses non-autoregressive masked semantic-to-acoustic generation. [F5-TTS](https://arxiv.org/abs/2410.06885) uses flow-matching DiT. Both are quality references but are not causal streaming architectures. Released weight licenses may also restrict commercial use.

### FireRedTTS, Fish Speech, and other dual-stage systems

FireRedTTS-1S exposes a useful choice between flow and multi-stream autoregressive decoders. Fish Speech’s slow semantic AR plus fast residual-codebook AR follows the depth-decoder pattern used by Moshi and CSM. These are suitable server/TTS designs, not the initial VC path.

## 8. Dialogue and simultaneous speech models

[Moshi](https://arxiv.org/abs/2410.00037) is a true full-duplex speech-text model with practical latency around 200 ms. [Hibiki](https://arxiv.org/abs/2502.03382) extends the multistream pattern to simultaneous speech translation and demonstrates adaptive delay. Sesame CSM uses a Llama backbone plus a smaller depth decoder over Mimi tokens for contextual TTS; it is not a complete dialogue agent by itself.

GLM-4-Voice and MiniCPM-o interleave speech and text for spoken assistants. Their transferable ideas are multistream timing, semantic-first generation, and depth decoding. They are not content-preserving VC models and should not be placed in Ripple’s v1 hot path.

## 9. Synthesis

Well-established:

- Causal convolution is still the safest edge waveform backbone.
- Training-only SSL distillation enables compact causal content encoders.
- Explicit speaker-normalized prosody is essential.
- Bounded-memory chunk training matters more than nominal model family.
- Full-reference or token-level speaker conditioning improves over one global vector.
- Acoustic codec tokens alone are a poor speaker-invariant content interface.

Promising but not settled:

- Speaker-anonymization perturbation for zero-lookahead VC.
- Tiny Emformer or fixed-window Conformer mixers at 50 Hz.
- Learned causal prosody estimators.
- FSQ/supervised semantic interfaces shared with TTS.
- INT8 stateful edge deployment across multiple NPUs.

Rejected for the v1 hot path:

- Multi-step diffusion/flow;
- full speech LMs;
- 80 ms codec frames;
- non-causal DAC-style decoding;
- Mamba/RWKV without a fused portable state operator;
- Hyena and dual-path networks;
- unrestricted attention or growing caches.

