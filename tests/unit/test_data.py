from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from ripple.contracts import Provenance, sha256_bytes
from ripple.data.features import FeatureManifest, FeatureShard, TeacherIdentity
from ripple.data.filters import QualityMetrics, QualityPolicy, rejection_reasons
from ripple.data.manifest import (
    AudioRecord,
    ConsentRecord,
    ConsentStatus,
    DatasetManifest,
    DatasetSplit,
    read_manifest,
    write_manifest,
)
from ripple.data.sampler import sample_indices, sampling_probabilities

NOW = datetime(2026, 7, 14, tzinfo=UTC)


def make_record(
    record_id: str,
    speaker_id: str,
    split: DatasetSplit,
    language: str = "en-US",
) -> AudioRecord:
    checksum = sha256_bytes(record_id.encode())
    return AudioRecord(
        record_id=record_id,
        uri=f"s3://licensed-corpus/{record_id}.flac",
        checksum=checksum,
        speaker_id=speaker_id,
        language=language,
        split=split,
        duration_seconds=2.0,
        sample_rate=24_000,
        channels=1,
        license_id="CC-BY-4.0",
        consent=ConsentRecord(
            status=ConsentStatus.GRANTED,
            reference=f"consent://{speaker_id}/v1",
            commercial_use=True,
            revocable=True,
            checked_at=NOW,
        ),
        domain="read",
        provenance=Provenance(
            source_uri=f"https://example.test/{record_id}",
            source_checksum=checksum,
            created_at=NOW,
            producer="ingest",
            producer_version="1",
            license_id="CC-BY-4.0",
        ),
    )


def test_dataset_manifest_round_trip_refuses_overwrite_and_tampering(tmp_path) -> None:
    manifest = DatasetManifest.create(
        dataset_id="pilot",
        revision="2026-07-14",
        records=(
            make_record("train-1", "speaker-a", DatasetSplit.TRAIN),
            make_record("test-1", "speaker-b", DatasetSplit.TEST),
        ),
        exclusion_log_checksum=sha256_bytes(b"exclusions"),
    )
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest
    with pytest.raises(FileExistsError):
        write_manifest(manifest, path)

    payload = manifest.model_dump(mode="json")
    payload["revision"] = "tampered"
    with pytest.raises(ValidationError, match="checksum"):
        DatasetManifest.model_validate(payload)


def test_dataset_manifest_rejects_speaker_split_leakage() -> None:
    with pytest.raises(ValidationError, match="split-disjoint"):
        DatasetManifest.create(
            dataset_id="leaky",
            revision="1",
            records=(
                make_record("train", "same-speaker", DatasetSplit.TRAIN),
                make_record("test", "same-speaker", DatasetSplit.TEST),
            ),
            exclusion_log_checksum=sha256_bytes(b"none"),
        )


def test_feature_cache_pins_teacher_and_source_hashes() -> None:
    teacher = TeacherIdentity(
        model_id="teacher",
        model_checksum=sha256_bytes(b"weights"),
        layer="12",
        projection_checksum=sha256_bytes(b"projection"),
    )
    shard = FeatureShard(
        uri="s3://features/00001.safetensors",
        checksum=sha256_bytes(b"shard"),
        source_manifest_checksum=sha256_bytes(b"manifest"),
        record_ids=("record-1",),
        provenance=make_record("source", "speaker", DatasetSplit.TRAIN).provenance,
    )
    manifest = FeatureManifest.create(feature_id="semantic-v1", teacher=teacher, shards=(shard,))
    assert manifest.checksum == manifest.computed_checksum()


def test_filters_and_temperature_sampling_are_deterministic() -> None:
    records = (
        make_record("en-1", "a", DatasetSplit.TRAIN, "en"),
        make_record("en-2", "b", DatasetSplit.TRAIN, "en"),
        make_record("fr-1", "c", DatasetSplit.TRAIN, "fr"),
    )
    probabilities = sampling_probabilities(records, temperature=0.5)
    assert probabilities[2] > probabilities[0]
    assert float(probabilities.sum()) == pytest.approx(1.0)
    np.testing.assert_array_equal(
        sample_indices(records, 50, seed=7),
        sample_indices(records, 50, seed=7),
    )

    reasons = rejection_reasons(
        records[0],
        QualityMetrics(clipped_fraction=0.01, snr_db=4.0, speech_fraction=0.1),
        QualityPolicy(),
    )
    assert reasons == ("clipping", "snr", "speech_fraction")

