# Dataset card: ripple-multilingual v0.1.0

**Status:** Draft — sources selected; **no training manifests approved until** legal review of derivative-model use, opt-out processes, and per-source license attestation. Checksums below are placeholders until first ingest.

## Identity

- Manifest version: `ripple-dataset-1` (planned)
- Manifest checksum: TBD (pin after first sealed train/dev/test JSON)
- Created: 2026-07-14
- Owner: Rizmy Abdulla
- Intended Ripple phases:
  - **Primary:** Phase 1 multilingual-core training and evaluation
  - **Also:** Phase 0 English baseline pilot as a **speaker-disjoint subset** of the English slice (LibriTTS + MLS-en), not a separate corpus family

## Sources and rights

**Global policy for this card**

- Preferred licenses: **CC BY 4.0** or **CC0** (more permissive).
- Commercial use (your answer): **yes** — only corpora where commercial use is clearly allowed may enter train manifests.
- Derivative model weight distribution: **unclear (your answer)** → **blocked for public model release** until Legal signs section Approval; internal research training may proceed only with explicit acceptance of that risk.
- Consent basis (default): **public research corpus**.
- Opt-out / deletion: **TBD** per source before any speaker-Identifiable public release.
- Legal review: **not yet**.

### Source stack (recommended)

Hours below are *available upstream*, not the capped v0.1.0 mix. Target after balancing: **1000+ validated hours** across the ten languages, with **soft per-language floors** (see Composition) so en/es/fr/pt from MLS do not drown out lower-resource languages.

#### 1. Multilingual LibriSpeech (MLS)

- Languages covered here: **en, es, fr, pt**
- Canonical URL / citation: https://www.openslr.org/94/ — Pratap et al., “MLS: A Large-Scale Multilingual Dataset for Speech Research,” Interspeech 2020
- License and version: **CC BY 4.0**
- Commercial use permitted: **yes** (per CC BY 4.0)
- Derivative model use permitted: **unclear pending Legal** (CC BY requires attribution; redistribution of weights that embed the data needs counsel review)
- Speaker consent / provenance: LibriVox public-domain / volunteer audiobook recordings curated into MLS
- Opt-out / deletion mechanism: TBD (LibriVox / Meta dataset channels)
- Legal review reference: not yet

#### 2. LibriTTS (English Phase 0 pilot slice)

- Languages covered here: **en**
- Canonical URL / citation: https://www.openslr.org/60/ — Zen et al., “LibriTTS,” Interspeech 2019
- License and version: **CC BY 4.0**
- Commercial use permitted: **yes**
- Derivative model use permitted: **unclear pending Legal**
- Speaker consent / provenance: LibriSpeech/LibriVox lineage (read speech)
- Opt-out / deletion mechanism: TBD
- Legal review reference: not yet
- Role: preferred **Phase 0** English pilot; keep official LibriSpeech/LibriTTS **test** speakers out of train

#### 3. Mozilla Common Voice (selected locales)

- Languages covered here: **en, zh, hi, es, fr, ar, bn, pt, ru, ur** (and regional variants as available, e.g. `zh-CN`, `ar`)
- Canonical URL / citation: https://commonvoice.mozilla.org/ — Ardila et al., “Common Voice,” LREC 2020
- License and version: **CC0 1.0** (corpus; verify the version you download)
- Commercial use permitted: **yes**
- Derivative model use permitted: **generally yes under CC0**; still confirm downloaded release notes and attribution/credit practice for Ripple cards
- Speaker consent / provenance: volunteer contributors under Common Voice terms
- Opt-out / deletion mechanism: Mozilla Common Voice contributor / dataset request process (document URL at ingest) — **TBD before release**
- Legal review reference: not yet
- Role: **primary** source for **zh, hi, ar, bn, ru, ur**; **supplement** for en/es/fr/pt

#### 4. FLEURS (Google)

- Languages covered here: all ten (**en, zh, hi, es, fr, ar, bn, pt, ru, ur** via FLEURS locale codes)
- Canonical URL / citation: https://huggingface.co/datasets/google/fleurs — Conneau et al., “FLEURS,” SLT 2022; **CC BY 4.0**
- Commercial use permitted: **yes** (CC BY 4.0)
- Derivative model use permitted: **unclear pending Legal**
- Speaker consent / provenance: read speech curated for few-shot multilingual evaluation (~12 h / language class)
- Opt-out / deletion mechanism: TBD (Google / HF dataset channels)
- Legal review reference: not yet
- Role: **prefer for development/test and language-balance eval**; use train split sparingly so FLEURS remains a stable external comparison

### Explicitly deferred (do not ingest into train until Legal + license attestation)

- AISHELL / WenetSpeech and other academic-only Mandarin stacks
- Any **CC BY-NC** or unclear ToS corpora (e.g. some conversational / broadcast sets)
- Children’s speech if identifiable from metadata
- Singing / music-heavy clips

No source may enter a sealed training manifest while any required field above remains unknown for that source.

## Composition

- Total validated hours: **target ≥ 1000 h** after quality filters (not yet measured)
- Speakers: **unknown until ingest**
- Languages and hours per language (plan):
  | Code | Name | Hour strategy (v0.1.0) |
  |---|---|---|
  | en | English | Cap MLS+LibriTTS+CV so English ≤ ~25% of total |
  | zh | Mandarin Chinese | Floor ≥ 50 h if available after filters; primary CV + FLEURS |
  | hi | Hindi | Floor ≥ 40 h; CV + FLEURS |
  | es | Spanish | MLS + CV; soft cap vs English |
  | fr | French | MLS + CV; soft cap vs English |
  | ar | Arabic | Floor ≥ 40 h; CV + FLEURS |
  | bn | Bengali | Floor ≥ 30 h; CV + FLEURS (expect scarce) |
  | pt | Portuguese | MLS + CV |
  | ru | Russian | Floor ≥ 40 h; CV + FLEURS |
  | ur | Urdu | Floor ≥ 30 h; CV + FLEURS (expect scarce) |
- Domains and recording conditions: **mixed** (read audiobook + crowdsourced read/prompted; limited spontaneous)
- Gender / age / accent coverage: **some sources only** (Common Voice demographics where voluntarily provided; MLS/LibriTTS limited)
- Read / spontaneous / expressive proportions: **unknown until ingest**; expect **read-heavy** (>80% read)

## Processing

- Original-format retention policy: keep vendor archives under `/data/raw/<source>/<release>/`; never mutate raw; all transforms write to `/data/canonical/`
- Decoder and version: **TorchCodec** preferred with **SoundFile** fallback; record package versions in provenance
- Canonical sample rate / format: **24 kHz, mono, PCM16 WAV** (Ripple Edge default)
- Segmentation / VAD: **reuse official source cuts first**; optional Silero VAD only for long uncut chapters — version-pin any VAD used
- Quality filters and thresholds: Ripple `QualityPolicy` defaults (`max_clipped_fraction=0.001`, `min_snr_db=10`, `min_speech_fraction=0.25`, `require_commercial_use=true`)
- Duplicate / near-duplicate method: **SHA-256 exact audio dedupe first**; optional embedding near-dupe in a later revision
- PII handling: strip speaker emails / unrestricted names from filenames and side JSON where present; do not scrape or store Common Voice account identifiers beyond dataset fields
- Teacher-feature versions and hashes:
  - Multilingual semantic teacher: **local XLS-R** (preferred) and/or **WavLM**; **HuBERT** only for English Phase 0 ablation
  - Store `TeacherIdentity` checksums in feature manifests; **no automatic downloads**

## Splits

- Train / development / test manifest checksums: TBD after first seal
- Recommended ratios: **90% / 5% / 5%** of validated hours **per language**, then concatenate (prevents English from stealing all eval mass)
- Speaker-disjoint verification: **required** (enforced by `DatasetManifest`)
- Recording / session-disjoint verification: **required** where session IDs exist; otherwise treat each clip URI as its own session
- Teacher-adaptation overlap verification: any teacher fine-tune / probe speakers must be disjoint from Ripple test
- External benchmark overlap verification: hold out and never train on:
  - LibriSpeech / LibriTTS official **test** speakers
  - MLS official **test** lists
  - Common Voice **test** validated sets for each locale used
  - FLEURS **test** (and prefer FLEURS **dev** for model selection)

## Known limitations and exclusions

- Unsupported or underrepresented languages: **bn** and **ur** will likely under-floor without supplemental licensed data; Mandarin may need a future dual-path if CV hours are insufficient
- Domain bias: **read / audiobook / prompted** speech dominates; conversational telephony and expressive speech underrepresented
- Label uncertainty: language tags from upstream locales; dialect (e.g. Arabic regional, Chinese varieties) not fully normalized in v0.1.0
- Excluded sources and reasons: NC / academic-only corpora; singing; identifiable child speech; any source with incomplete rights row

## Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Data engineering | Rizmy Abdulla | Draft accepted for ingest planning | 2026-07-14 |
| Research | Rizmy Abdulla | Draft accepted for Phase 1 design | 2026-07-14 |
| Privacy | Rizmy Abdulla | Provisional for **internal research only** until opt-out docs exist | 2026-07-14 |
| Legal | TBD | **Required before public model or dataset release** (esp. derivative weights) | — |
| Release decision | — | **Not approved for public release** | — |

### Immediate next actions

1. Legal: confirm commercial + weight-distribution posture for MLS, LibriTTS, FLEURS, Common Voice (downloaded version).
2. Download raw archives into `/data/raw/` and fill hours / speakers after filter pass.
3. Seal train/dev/test manifests and replace every TBD checksum in this card.
4. Document Common Voice opt-out URL in this card before any external sharing.
