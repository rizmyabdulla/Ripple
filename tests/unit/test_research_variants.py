from ripple.research import Gate, VariantStatus, standard_variant


def test_gate_direction() -> None:
    assert Gate("latency", 20.0).passes(19.0)
    assert not Gate("latency", 20.0).passes(21.0)
    assert Gate("similarity", 0.8, maximum=False).passes(0.9)


def test_standard_variant_requires_all_edge_evidence() -> None:
    spec = standard_variant(
        name="fused-ssm",
        hypothesis="A fused recurrent step reduces bounded state at equal quality.",
        baseline_artifact="ripple-vc-edge-1",
        requires_custom_kernel=True,
    )
    passed, failures = spec.evaluate(
        {
            "p95_compute_ms": 10.0,
            "rtf": 0.4,
            "wer_delta_abs": 0.2,
            "f0_pcc_delta_abs": 0.01,
            "rss_growth_bytes_per_hour": 0.0,
            "nan_count": 0.0,
        }
    )
    assert passed
    assert failures == ()
    assert spec.status is VariantStatus.PROPOSED


def test_missing_or_regressed_gate_rejects_promotion() -> None:
    spec = standard_variant(
        name="flow-decoder",
        hypothesis="One-step flow improves quality without missing deadlines.",
        baseline_artifact="ripple-vc-edge-1",
    )
    passed, failures = spec.evaluate({"p95_compute_ms": 25.0})
    assert not passed
    assert "p95_compute_ms=25 does not satisfy <=20" in failures
    assert "missing:rtf" in failures

