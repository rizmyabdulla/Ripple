from __future__ import annotations

import json
import wave
from pathlib import Path

from typer.testing import CliRunner

from ripple.cli.main import app
from ripple.cli.manifest import app as manifest_app
from ripple.data.manifest import read_manifest


def _write_silence(path: Path, *, frames: int = 4800, speaker: str = "spk-a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(b"\x00\x00" * frames)


def test_manifest_cli_builds_provenance_complete_record(tmp_path: Path) -> None:
    source = tmp_path / "audio" / "spk-a"
    source.mkdir(parents=True)
    _write_silence(source / "sample.wav")
    manifest = tmp_path / "manifest.jsonl"
    result = CliRunner().invoke(
        manifest_app,
        [
            "build",
            str(source.parent),
            str(manifest),
            "--license",
            "CC0-1.0",
            "--consent-basis",
            "synthetic",
            "--language",
            "en",
        ],
    )
    assert result.exit_code == 0, result.output
    row = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["sample_rate"] == 24_000
    assert row["frames"] == 4800
    assert row["license"] == "CC0-1.0"
    assert row["consent_basis"] == "synthetic"


def test_doctor_and_config_cli(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "doctor",
            "run",
            "--data-root",
            str(tmp_path / "data"),
            "--checkpoint-root",
            str(tmp_path / "ckpt"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config_ok"] is True
    checksum = CliRunner().invoke(app, ["config", "checksum"])
    assert checksum.exit_code == 0, checksum.output
    assert checksum.output.strip().startswith("sha256:")


def test_manifest_seal_validate_and_train_smoke(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    for speaker in ("spk-a", "spk-b", "spk-c"):
        _write_silence(root / speaker / "utt.wav", speaker=speaker)
    draft = tmp_path / "draft.jsonl"
    sealed = tmp_path / "train.json"
    build = CliRunner().invoke(
        app,
        [
            "manifest",
            "build",
            str(root),
            str(draft),
            "--license",
            "CC-BY-4.0",
            "--consent-basis",
            "synthetic",
            "--language",
            "en",
        ],
    )
    assert build.exit_code == 0, build.output
    seal = CliRunner().invoke(
        app,
        [
            "manifest",
            "seal",
            str(draft),
            str(sealed),
            "--dataset-id",
            "cli-smoke",
            "--revision",
            "1",
            "--audio-root",
            str(root),
            "--overwrite",
        ],
    )
    assert seal.exit_code == 0, seal.output
    manifest = read_manifest(sealed)
    assert len(manifest.records) == 3
    validate = CliRunner().invoke(
        app,
        [
            "manifest",
            "validate",
            str(sealed),
            "--audio-root",
            str(root),
            "--verify-checksums",
        ],
    )
    assert validate.exit_code == 0, validate.output
    out_dir = tmp_path / "run"
    train = CliRunner().invoke(
        app,
        [
            "train",
            "run",
            "--manifest",
            str(sealed),
            "--audio-root",
            str(root),
            "--output-dir",
            str(out_dir),
            "--stage",
            "decoder_reconstruction",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--crop-samples",
            "4800",
            "--save-every",
            "2",
        ],
    )
    assert train.exit_code == 0, train.output
    last = out_dir / "checkpoint_last.pt"
    assert last.is_file()
    evaluate = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--checkpoint",
            str(last),
            "--manifest",
            str(sealed),
            "--audio-root",
            str(root),
            "--max-items",
            "1",
            "--crop-samples",
            "4800",
        ],
    )
    assert evaluate.exit_code == 0, evaluate.output
    metrics = json.loads(evaluate.output)
    assert "snr_db" in metrics["metrics"]
