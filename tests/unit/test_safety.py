from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ripple.safety import (
    ConsentDenied,
    ConsentRecord,
    HMACProvenanceSigner,
    ProfilePolicy,
    ProfileUse,
    ProvenanceRecord,
    WatermarkPayload,
    WatermarkPolicy,
    WatermarkRequired,
    sign_record,
)


def _consent(**changes: object) -> ConsentRecord:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    values: dict[str, object] = {
        "subject_id": "speaker-1",
        "profile_id": "profile-1",
        "allowed_uses": frozenset({ProfileUse.CONVERSION}),
        "granted_at": now - timedelta(days=1),
        "expires_at": now + timedelta(days=1),
        "evidence_uri": "urn:consent:abc",
    }
    values.update(changes)
    return ConsentRecord(**values)  # type: ignore[arg-type]


def test_profile_policy_is_fail_closed() -> None:
    policy = ProfilePolicy()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    policy.authorize(
        _consent(), ProfileUse.CONVERSION, profile_id="profile-1", now=now
    )
    with pytest.raises(ConsentDenied, match="tts is not authorized"):
        policy.authorize(_consent(), ProfileUse.TTS, profile_id="profile-1", now=now)
    with pytest.raises(ConsentDenied, match="consent revoked"):
        policy.authorize(
            _consent(revoked_at=now - timedelta(seconds=1)),
            ProfileUse.CONVERSION,
            profile_id="profile-1",
            now=now,
        )


def test_provenance_is_canonical_and_signed() -> None:
    record = ProvenanceRecord(
        artifact_id="output-1",
        artifact_sha256="a" * 64,
        model_id="model-1",
        model_sha256="b" * 64,
        created_at="2026-01-01T00:00:00+00:00",
        operation="conversion",
        metadata={"z": "last", "a": "first"},
    )
    signer = HMACProvenanceSigner(b"x" * 32, key_id="test-key")
    signed = sign_record(record, signer)
    assert signer.verify(record.canonical_bytes(), bytes.fromhex(signed.signature_hex))
    assert record.record_sha256 == record.record_sha256


def test_required_watermark_has_no_silent_passthrough() -> None:
    policy = WatermarkPolicy(None, required=True)
    payload = WatermarkPayload("output-1", "a" * 64, {})
    with pytest.raises(WatermarkRequired):
        policy.protect(object(), 24_000, payload)  # type: ignore[arg-type]
