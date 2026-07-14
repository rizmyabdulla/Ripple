from __future__ import annotations

import json
import wave
from pathlib import Path

from typer.testing import CliRunner

from ripple.cli.manifest import app


def _write_silence(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(b"\x00\x00" * 480)


def test_manifest_cli_builds_provenance_complete_record(tmp_path: Path) -> None:
    source = tmp_path / "audio"
    source.mkdir()
    _write_silence(source / "sample.wav")
    manifest = tmp_path / "manifest.jsonl"
    result = CliRunner().invoke(
        app,
        [
            str(source),
            str(manifest),
            "--license",
            "CC0-1.0",
            "--consent-basis",
            "synthetic",
            "--language",
            "und",
        ],
    )
    assert result.exit_code == 0, result.output
    row = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["sample_rate"] == 24_000
    assert row["frames"] == 480
    assert row["license"] == "CC0-1.0"
    assert row["consent_basis"] == "synthetic"
