# Data, training strategy, and losses

## 1. Data principles

The model should be trained for the deployment distribution, not merely clean read speech. Every source must pass:

- license and commercial-use review;
- speaker-consent/provenance review;
- duplicate and near-duplicate removal;
- speaker-disjoint train/development/test splits;
- language, accent, gender, age, style, and recording-condition documentation;
- PII and opt-out handling;
- automated quality filtering followed by sampled human audit.

Large scraped corpora are not automatically acceptable. Dataset size claims from modern systems are often vendor-reported and do not substitute for a data card.

## 2. Dataset portfolio

The exact mix depends on licensing, but training should cover:

- clean multi-speaker read speech for reconstruction and intelligibility;
- multilingual speech across target language families;
- spontaneous and conversational speech;
- expressive/emotional speech with explicit rights;
- telephony, headset, room, and mobile microphone conditions;
- low-SNR and reverberant speech;
- singing only as a separately gated extension.

Candidate public sources include LibriTTS/LibriSpeech, MLS, VCTK, Common Voice, VoxPopuli, WenetSpeech, and carefully reviewed expressive datasets. Emilia-derived checkpoints or datasets require an explicit license/provenance decision rather than silent reuse.

Suggested sampling objective:

- 25% clean read speech;
- 30% multilingual and multi-accent speech;
- 20% spontaneous/conversational speech;
- 15% noisy/channel-diverse speech;
- 10% expressive speech.

These are starting weights, not fixed truths. Sampling should be temperature-balanced so high-resource English or Mandarin does not dominate.

## 3. Dataset pipeline

1. Ingest immutable manifests with URI, checksum, license, speaker, language, and consent fields.
2. Decode to a canonical lossless format; retain original files for traceability.
3. Resample with a single tested implementation.
4. Run VAD, clipping, SNR, reverberation, DNSMOS-like, language-ID, and ASR checks.
5. Segment into short training crops and long continuous streams.
6. Derive teacher features, pitch, voicing, energy, speaker embeddings, and optional emotion labels.
7. Store versioned features in sharded files with source checksums.
8. Create speaker/language/domain-disjoint evaluation manifests.
9. Publish a dataset card and exclusion log for each training release.

Feature caches must include teacher model and layer hashes. Silent regeneration with a new teacher invalidates reproducibility.

## 4. Augmentation

### Acoustic robustness

- additive noise with SNR curriculum;
- room impulse responses and microphone coloration;
- band-limit, equalization, clipping, gain, and dynamic-range compression;
- Opus/telephony codec simulation;
- packet loss, duplicated packets, jitter, and zero-filled gaps;
- speed perturbation with all time-dependent labels recomputed.

### Disentanglement

- formant/timbre shifting that preserves content and contour;
- Seed-VC-style external tone-color conversion where licensing permits;
- speaker-anonymization perturbation as a separate, validated method;
- random reference cropping and channel mismatch;
- source and target speaker mismatch in every relevant batch.

Augmentation must not destroy label validity. Pitch shifting, speed changes, and time stretching require regenerated prosody labels.

## 5. Multi-stage training

### Stage A: teacher and target selection

Train no production model yet. Compare HuBERT, WavLM, multilingual HuBERT/XLS-R, Whisper/ASR, and supervised-token targets using frozen probes:

- phoneme/ASR accuracy;
- source-speaker predictability;
- language coverage;
- prosody leakage;
- temporal stability;
- distillability into the same causal student.

Select one RIF-1 semantic target or an ensemble-projected target.

### Stage B: causal semantic student

Train source encoder and RippleMixer with streaming masks and fixed state.

Objectives:

- KL divergence to teacher soft distributions;
- cross-entropy to hard pseudo-labels;
- feature regression at selected teacher layers;
- invariance between original and timbre-perturbed audio;
- gradient-reversal source-speaker classifier;
- optional CTC/ASR auxiliary loss for multilingual intelligibility.

Use multiple chunk sizes but preserve the v1 20 ms emission cadence.

### Stage C: speaker enrollment

Train global identity and style tokens using:

- speaker classification or metric learning;
- cross-crop consistency;
- channel invariance;
- token diversity/orthogonality;
- emotion-invariance or emotion-separation losses according to product mode.

The speaker encoder must be evaluated on unseen speakers and cross-language references.

### Stage D: decoder reconstruction

Freeze or stop gradients into the semantic student. Train the causal source-filter decoder on same-utterance reconstruction with target speaker profile from another crop of the same speaker.

Start with spectral and waveform losses. Introduce discriminators only after stable reconstruction.

### Stage E: any-to-any hardening

Use timbre-shifted source audio, mismatched target profiles, and teacher targets from the unshifted source. Add converted-output speaker consistency and source-leakage losses.

Where no paired cross-speaker ground truth exists, preserve content/prosody through teacher, ASR, F0, and cycle/consistency constraints rather than pretending a waveform target exists.

### Stage F: streaming realism

Train long, stateful sequences rather than independent clips:

- 30 seconds, then several minutes;
- silence and speech resumption;
- speaker/profile changes only at defined boundaries;
- packet jitter/loss;
- running-statistic warm-up;
- state reset and no-reset examples.

Distill quality mode into zero-lookahead edge mode if the quality teacher wins.

### Stage G: quantization-aware fine-tuning

After a stable FP model:

- insert fake quantization for selected Conv/Linear layers;
- preserve norms, oscillator, final waveform projection, and sensitive prosody layers in higher precision;
- train on the same streaming schedule;
- gate every checkpoint on quality, leakage, and long-session stability.

## 6. Loss functions

Let `x` be target waveform, `x_hat` reconstruction/conversion, `S` semantic features, `P` prosody, and `E_tgt` target speaker profile.

### Semantic losses

- `L_sem_kl`: KL to teacher posterior.
- `L_sem_ce`: hard pseudo-label cross-entropy.
- `L_sem_feat`: cosine/L1 regression to projected teacher features.
- `L_content_inv`: consistency under timbre perturbation.
- `L_src_adv`: gradient-reversal source-speaker classification.
- optional `L_ctc`: text/phoneme consistency.

### Prosody losses

- voiced-frame log-F0 L1/Huber;
- voicing binary cross-entropy;
- periodicity/confidence regression;
- energy Huber;
- contour correlation and delta losses;
- oscillator phase-continuity penalty at chunk boundaries.

### Waveform and spectral losses

- waveform L1 at aligned samples;
- multi-resolution STFT spectral convergence and log-magnitude loss;
- multi-scale mel loss;
- pre-emphasized high-band loss for 24 kHz detail;
- optional complex STFT loss for phase-sensitive transients.

### Adversarial losses

- multi-period discriminator for pitch periodicity;
- multi-resolution STFT discriminator;
- optional multi-scale waveform discriminator;
- feature matching with a high relative weight;
- adversarial warm-up and adaptive weighting to prevent semantic collapse.

### Speaker/style losses

- target-speaker embedding cosine/metric loss on converted audio;
- source-speaker rejection or margin loss;
- global/token cross-crop consistency;
- style-token diversity regularization;
- optional emotion preservation or target-style loss.

### Boundary and state losses

- overlap-region waveform consistency during training only;
- adjacent-chunk feature continuity;
- state-reset equivalence at valid utterance boundaries;
- long-stream hidden-state norm regularization;
- packet-loss recovery loss.

## 7. Initial loss schedule

An initial weighting policy:

1. Semantic student: semantic and leakage losses only.
2. Decoder warm-up: spectral + waveform + prosody.
3. Add feature matching.
4. Add low-weight adversarial objectives.
5. Add speaker consistency and any-to-any perturbations.
6. Add long-stream/boundary losses.
7. Run automatic loss balancing only within bounded ranges.

SoundStream’s large feature-matching weight is a useful starting point, but Ripple must retune at 24 kHz and with its source-filter decoder.

## 8. Training systems

- PyTorch training with distributed data parallelism and BF16 where stable.
- Deterministic manifest/version tracking.
- Gradient accumulation to maintain effective batch while long-stream crops grow.
- Exponential moving average of generator weights.
- Separate optimizers for generator and discriminators.
- Activation checkpointing only in training.
- Mixed short-crop and long-stream batches.
- Periodic export-and-run tests during training, not only at the end.

## 9. Model selection

No single score selects a checkpoint. A candidate must satisfy:

- multilingual WER/CER non-inferiority;
- speaker similarity improvement without source leakage;
- F0/energy preservation;
- subjective naturalness;
- zero-lookahead streaming stability;
- target-device latency and memory;
- quantized quality gates;
- one-hour and eight-hour state tests.

Pareto-front checkpoints should be retained. A quality gain that breaks p95 latency or memory is not an edge-model improvement.

## 10. Safety and governance

- Treat voice profiles as biometric data.
- Require explicit, revocable consent for cloneable target voices.
- Maintain speaker opt-out and deletion workflows.
- Separate consented conversion from anonymization products and metrics.
- Add provenance/watermarking that survives common resampling and Opus transforms.
- Restrict high-risk identity enrollment and abuse-prone APIs.
- Log model/data versions used to create each voice profile.
- Review synthetic-audio disclosure obligations, including EU AI Act requirements, with counsel.
