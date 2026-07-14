# Evaluation and benchmark specification

## 1. Benchmark rule

Ripple is compared only under disclosed:

- hardware, OS, runtime, precision, and thread count;
- sample rate and chunk size;
- future lookahead;
- warm/cold state;
- source/target seen/unseen status;
- language and recording condition;
- model artifact and dataset version.

RTF without algorithmic latency is insufficient. MOS without confidence intervals and listener protocol is insufficient.

## 2. Baselines

At minimum:

- StreamVC paper numbers, clearly labeled as non-matched historical reference;
- a faithful local reproduction if one can be trained and validated;
- Seed-VC quality and real-time modes;
- OpenVoice tone-color conversion where task-compatible;
- Conan and Zero-VC when code/weights and evaluation terms permit;
- a strong offline quality ceiling such as Seed-VC or a modern flow VC;
- no-conversion source and target-reference oracles.

TTS systems such as CosyVoice, F5-TTS, and MaskGCT are not direct VC baselines. They may benchmark future Ripple-TTS.

## 3. Test sets

Create immutable suites:

- unseen source and target speakers;
- same-language and cross-language conversion;
- high/low pitch and cross-gender pairs;
- clean studio, headset, room, telephony, and noisy audio;
- read, spontaneous, emotional, whispered, and singing subsets as supported;
- short utterances and continuous 30-minute/eight-hour streams;
- packet-loss/jitter conditions;
- consented human-listening subset.

All splits must be speaker-disjoint from training and teacher adaptation.

## 4. Latency

Report:

- algorithmic lookahead in milliseconds;
- frame cadence;
- per-chunk compute p50/p95/p99;
- capture-to-first-output latency;
- steady-state capture-to-playback latency;
- queue and resampler contribution;
- cold-start model load and first-call time;
- speaker enrollment time separately.

Primary edge gate:

- zero-lookahead artifact;
- p95 compute below the 20 ms chunk duration;
- target p95 end-to-end below 40 ms on the named reference device;
- no missed deadlines in a 30-minute thermal run.

## 5. Real-time factor and throughput

`RTF = wall-clock inference time / audio duration`.

Report:

- single-stream RTF;
- concurrent-stream throughput on server;
- warm and cold;
- mean plus p95;
- CPU affinity and frequency policy;
- decoder, encoder, mixer, and prosody component breakdown.

Target edge RTF is below 0.5, with a hard gate below 0.8.

## 6. CPU, power, and memory

- process CPU time and utilization;
- physical cores/threads used;
- peak RSS and steady-state RSS;
- state/scratch/weight breakdown;
- artifact size on disk;
- energy per second of audio where available;
- device temperature, frequency, and RTF over 30 minutes;
- memory growth over eight hours.

Gates:

- no statistically significant RSS slope over long stream;
- no thermal deadline failure;
- INT8 bundle target below 40 MB, subject to quality gates.

## 7. Intelligibility

- WER and CER using at least two ASR evaluators where practical;
- language-specific normalization;
- source versus converted delta;
- named entities, numbers, and code-switch subsets;
- word boundary errors around chunks and packet recovery.

An ASR model related to the content teacher must not be the only evaluator.

## 8. Speaker similarity and leakage

Target similarity:

- cosine similarity from at least two independent speaker-verification models;
- equal-error-rate or verification accuracy;
- human speaker-similarity MOS.

Source leakage:

- source-speaker classifier accuracy from semantic features;
- converted-to-source speaker similarity;
- VoicePrivacy-style attacker EER where anonymization is relevant.

High target similarity with high source leakage is a failure, not a success.

## 9. Naturalness and quality

- blinded MUSHRA or MOS with confidence intervals;
- preference tests against matched baselines;
- DNSMOS/UTMOS or successors as secondary proxies;
- ViSQOL/PESQ only where their assumptions fit;
- spectral artifact and high-band analyses.

Objective MOS predictors must not select the final model alone; codec research shows disagreement with human ratings.

## 10. Prosody and emotion

- voiced-frame F0 Pearson/Spearman correlation;
- F0 RMSE after time alignment;
- voicing decision F1;
- energy correlation;
- duration/rhythm deviation;
- boundary pitch/phase discontinuity;
- emotion classifier UAR and human emotion-preservation rating;
- separate source-prosody-preservation and target-range-remapping modes.

Tone languages require language-specific lexical-tone evaluation.

## 11. Multilingual and cross-language

- WER/CER by language, family, and resource level;
- target-speaker similarity when reference language differs from source;
- accent preservation and accent leakage ratings;
- code-switch accuracy;
- language-ID stability;
- unseen-language transfer as exploratory, not claimed support.

Report macro averages so high-resource languages do not dominate.

## 12. Streaming stability

Metrics:

- chunk-boundary spectral discontinuity;
- output amplitude and pitch jumps;
- state norm drift;
- dropout/repetition rate;
- recovery time after silence, packet loss, or reset;
- semantic flicker under different chunk-call grouping;
- deterministic output for identical state/input;
- NaN/Inf count;
- long-session similarity and quality at regular intervals.

Scenarios:

- eight hours continuous mixed speech/silence;
- one hour of silence followed by speech;
- repeated 1–5% packet loss;
- 1,000 soft resets;
- target-profile swap;
- quantized and floating-point tiers.

## 13. Quantization gates

Relative to the FP16 backend on identical streams:

- WER/CER increase no more than 1 percentage point absolute;
- target-speaker similarity degradation below a pre-registered threshold;
- F0 correlation degradation below 0.02 absolute;
- no statistically significant MOS preference for FP16 in the final confirmatory test, or an explicitly accepted tier trade-off;
- no new state drift or NaNs;
- p95 latency and RSS materially improve.

INT4 is accepted only if it creates a measured deployment benefit beyond INT8.

## 14. Ablation matrix

Mandatory:

- 16 kHz versus 24 kHz;
- pure convolution versus RippleMixer;
- local attention window sizes;
- no summary memory versus bounded memory;
- YIN versus learned causal prosody;
- no F0, raw F0, whitened F0, and target-range F0;
- global speaker vector versus global plus token bank;
- transposed versus resize/subpixel upsampling;
- decoder without versus with harmonic/noise source;
- no timbre perturbation versus Seed-VC-style versus anonymization perturbation;
- soft semantic versus hard semantic versus optional FSQ;
- zero versus one-frame lookahead;
- FP16, INT8 PTQ, INT8 QAT, and selective INT4.

Each ablation must report quality, latency, memory, and leakage—not only one metric.

## 15. Reproducible benchmark harness

The harness should:

- feed identical PCM chunks to every backend;
- pin threads and record environment;
- separate load, warm-up, and steady-state;
- collect per-component timings;
- save output, state-health summaries, and manifest hashes;
- generate machine-readable JSON and human-readable reports;
- run locally and in CI on a small canary suite;
- schedule full device/subjective suites for release candidates.

## 16. Release gate

Ripple v1 is ready only if:

1. It meets continuous real-time deadlines on the reference edge device.
2. INT8 meets quality and stability gates.
3. It improves a pre-registered combination of quality, speaker similarity, or latency over the reproduced baseline without regressing intelligibility or leakage.
4. It survives the eight-hour stream.
5. Multilingual claims are supported language by language.
6. Data, model, safety, and license cards are complete.

“Significantly better than StreamVC” requires statistical significance on matched tests, not architectural novelty.
