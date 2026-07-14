from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ripple.contracts import (
    ArtifactFile,
    ArtifactManifest,
    PitchProfile,
    ProfileNormalization,
    Provenance,
    RifFrame,
    RifSequence,
    RifSpec,
    SpeakerProfile,
    StateTensorSpec,
    StreamStateSchema,
    TensorDType,
    sha256_bytes,
    sha256_file,
)


def provenance(checksum: str | None = None) -> Provenance:
    return Provenance(
        source_uri="https://example.test/source.wav",
        source_checksum=checksum or sha256_bytes(b"source"),
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
        producer="ripple-test",
        producer_version="1",
        source_revision="abc123",
        license_id="CC0-1.0",
    )


def test_rif_sequence_enforces_dimensions_probabilities_and_order() -> None:
    spec = RifSpec(semantic_classes=256)
    frame = RifFrame(
        frame_index=0,
        semantic_soft=(1.0,) + (0.0,) * 255,
        semantic_embed=(0.0,) * 128,
        prosody=(0.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    sequence = RifSequence(spec=spec, frames=(frame,))
    assert sequence.spec.frame_samples == 480

    with pytest.raises(ValidationError, match="sum to one"):
        RifSequence(
            spec=spec,
            frames=(frame.model_copy(update={"semantic_soft": (0.0,) * 256}),),
        )
    with pytest.raises(ValidationError, match="contiguous"):
        RifSequence(spec=spec, frames=(frame.model_copy(update={"frame_index": 1}),))


def test_speaker_profile_round_trip_is_frozen_and_tamper_evident() -> None:
    profile = SpeakerProfile.create(
        speaker_id="speaker-7",
        consent_reference="consent://speaker-7/v1",
        sample_rate=24_000,
        speaker_global=(0.01,) * 192,
        speaker_tokens=((0.02,) * 64,) * 4,
        pitch_profile=PitchProfile(median_log_f0=5.0, log_f0_p05=4.5, log_f0_p95=5.5),
        normalization=ProfileNormalization(
            waveform_peak=0.9,
            waveform_mean=0.0,
            waveform_std=0.2,
        ),
        provenance=provenance(),
    )
    restored = SpeakerProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile
    with pytest.raises(ValidationError, match="frozen"):
        restored.speaker_id = "other"  # type: ignore[misc]

    payload = profile.model_dump(mode="json")
    payload["speaker_id"] = "attacker"
    with pytest.raises(ValidationError, match="checksum"):
        SpeakerProfile.model_validate(payload)


def test_stream_state_is_fixed_and_bounded() -> None:
    schema = StreamStateSchema(
        tensors=(
            StateTensorSpec(name="mixer.conv", shape=(1, 128, 14), dtype=TensorDType.FLOAT32),
            StateTensorSpec(name="mixer.index", shape=(1,), dtype=TensorDType.INT64),
        )
    )
    assert schema.frame_rate == 50
    assert schema.total_state_bytes == 1 * 128 * 14 * 4 + 8

    with pytest.raises(ValidationError, match="unique"):
        StreamStateSchema(tensors=(schema.tensors[0], schema.tensors[0]))
    with pytest.raises(ValidationError, match="exceeds"):
        StreamStateSchema(
            tensors=(
                StateTensorSpec(
                    name="oversized",
                    shape=(8 * 1024 * 1024 + 1,),
                    dtype=TensorDType.INT8,
                ),
            )
        )


def test_artifact_manifest_verifies_file_bytes(tmp_path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    item = ArtifactFile(
        path="model.bin",
        checksum=sha256_file(model),
        size_bytes=model.stat().st_size,
        media_type="application/octet-stream",
    )
    manifest = ArtifactManifest.create(
        artifact_id="edge-test",
        model_version="0.1",
        resolved_config_checksum=sha256_bytes(b"config"),
        provenance=provenance(),
        files=(item,),
    )
    manifest.verify_files(tmp_path)

    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match=r"size mismatch|checksum mismatch"):
        manifest.verify_files(tmp_path)

