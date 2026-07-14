from __future__ import annotations

import zipfile

import pytest

from ripple.export.artifact import (
    build_artifact_bundle,
    verify_artifact_bundle,
)


def manifest() -> dict[str, object]:
    return {
        "family": "ripple-vc",
        "model_version": "0.1.0",
        "rif_version": 1,
        "speaker_profile_version": 1,
        "sample_rate": 24000,
        "chunk_samples": 480,
        "lookahead_samples": 0,
        "backend": "onnxruntime-cpu",
        "precision": "fp32",
        "state_tensors": [
            {
                "name": "state_cache",
                "dtype": "float32",
                "shape": [1, 8],
                "layout": "contiguous",
                "reset_policy": "hard",
            }
        ],
    }


def test_bundle_is_deterministic_and_hmac_verified(tmp_path) -> None:
    key = b"k" * 32
    first = build_artifact_bundle(
        tmp_path / "one.ripple",
        files={"graphs/model.onnx": b"model bytes"},
        manifest=manifest(),
        signing_key=key,
        signature_algorithm="hmac-sha256",
        key_id="test-key",
    )
    second = build_artifact_bundle(
        tmp_path / "two.ripple",
        files={"graphs/model.onnx": b"model bytes"},
        manifest=manifest(),
        signing_key=key,
        signature_algorithm="hmac-sha256",
        key_id="test-key",
    )

    assert first.read_bytes() == second.read_bytes()
    report = verify_artifact_bundle(
        first, verification_key=key, require_signature=True
    )
    assert report.signature_verified
    assert report.checked_files == ("graphs/model.onnx",)


def test_checksum_tampering_is_rejected(tmp_path) -> None:
    artifact = build_artifact_bundle(
        tmp_path / "valid.ripple",
        files={"model.onnx": b"original"},
        manifest=manifest(),
    )
    tampered = tmp_path / "tampered.ripple"
    with zipfile.ZipFile(artifact) as source, zipfile.ZipFile(tampered, "w") as target:
        for item in source.infolist():
            payload = b"changed" if item.filename == "model.onnx" else source.read(item)
            target.writestr(item, payload)

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_artifact_bundle(tampered)


def test_unsafe_member_name_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        build_artifact_bundle(
            tmp_path / "invalid.ripple",
            files={"../model.onnx": b"x"},
            manifest=manifest(),
        )
