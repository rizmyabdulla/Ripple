# Dataset card: multilingual-core fixture v2026-07-14

## Identity

- Manifest version: ripple-dataset-1
- Manifest checksum: see `fixtures/multilingual-core/manifests/*.json`
- Created: 2026-07-14
- Owner: Ripple project (local fixture)
- Intended Ripple phases: 1–5 smoke validation only

## Sources and rights

- Canonical URL and citation: synthetic fixture generator in-repo
- License and version: CC-BY-4.0 metadata tags for tooling only
- Commercial use permitted: yes (synthetic)
- Derivative model use permitted: yes (synthetic)
- Speaker consent/provenance basis: synthetic granted consent records
- Opt-out/deletion mechanism: delete fixture files
- Legal review reference: not required for synthetic non-speech URIs

No real audio is referenced. Do not train release models from this fixture alone.

## Composition

- Total validated hours: synthetic seconds only
- Speakers: six languages × three disjoint split speakers
- Languages and hours per language: equal synthetic durations
- Domains and recording conditions: read, URI placeholders
- Gender/age/accent coverage: not labeled
- Read/spontaneous/expressive proportions: 100% read placeholders

## Processing

- Original-format retention policy: fixtures only
- Decoder and version: n/a
- Canonical sample rate/format: 24 kHz mono PCM16 target
- Segmentation/VAD: none
- Quality filters and thresholds: QualityPolicy defaults
- Duplicate/near-duplicate method: unique record IDs enforced
- PII handling: none present
- Teacher-feature versions and hashes: none yet

## Splits

- Train/development/test manifest checksums: pinned in JSON files
- Speaker-disjoint verification: enforced by `DatasetManifest`
- Recording/session-disjoint verification: synthetic per-record URIs
- Teacher-adaptation overlap verification: none
- External benchmark overlap verification: none

## Known limitations and exclusions

- Unsupported or underrepresented languages: many; this is a six-language smoke set
- Domain bias: read-only placeholders
- Label uncertainty: language tags are assigned, not measured
- Excluded sources and reasons: no real corpora included

## Approval

- Fixture for CI and local tooling only.
- Not approved for public model release training.
