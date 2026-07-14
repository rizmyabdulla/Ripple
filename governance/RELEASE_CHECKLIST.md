# Ripple release checklist

## Reproducibility

- [ ] Clean source commit and signed tag recorded.
- [ ] Root and specialist environment lock checksums recorded.
- [ ] Resolved configuration, dataset manifests, teacher features, and checkpoint hashes recorded.
- [ ] Artifact is reproducible from the release workflow.

## Quality and performance

- [ ] Multilingual benchmark completed on immutable manifests.
- [ ] Listening test completed with confidence intervals.
- [ ] Source leakage and target similarity gates passed.
- [ ] Zero-lookahead and optional quality-tier latency reported separately.
- [ ] INT8 quality delta passed.
- [ ] One-hour nightly and eight-hour release stability passed.
- [ ] Thermal/device matrix passed without missed deadlines.

## Runtime

- [ ] PyTorch/ONNX/backend conformance passed.
- [ ] State/profile/RIF schemas are compatible and versioned.
- [ ] Backend fallback and corrupted-artifact handling passed.
- [ ] No per-frame allocation or unbounded state growth.
- [ ] SDK examples pass from a clean environment.

## Governance

- [ ] Dataset, model, and safety cards approved.
- [ ] Licenses for data, teachers, dependencies, and generated weights approved.
- [ ] Consent, opt-out, and profile deletion workflows tested.
- [ ] Artifact signing, profile encryption, and provenance controls tested.
- [ ] Operational and incident-response owners assigned.

