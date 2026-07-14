"""Offline teacher-feature extraction CLI."""

# ruff: noqa: B008

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
import typer

from ripple.contracts.checksums import sha256_file
from ripple.contracts.manifest import Provenance
from ripple.data.features import FeatureManifest, FeatureShard, TeacherIdentity
from ripple.data.manifest import read_manifest
from ripple.teachers import create_teacher

from .common import read_pcm16_wav

app = typer.Typer(help="Extract local-only teacher features from manifests.")


def _iter_rows(manifest: Path) -> list[dict[str, object]]:
    if manifest.suffix.lower() == ".jsonl":
        rows: list[dict[str, object]] = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise typer.BadParameter(f"manifest line {line_number} must be object")
                rows.append(row)
        return rows
    sealed = read_manifest(manifest)
    return [
        {
            "id": record.record_id,
            "path": record.uri,
            "sha256": record.checksum.removeprefix("sha256:"),
        }
        for record in sealed.records
    ]


@app.command("extract")
def extract(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    audio_root: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Argument(...),
    teacher: str = typer.Option(..., help="hubert, wavlm, xls-r, or whisper"),
    model_path: Path = typer.Option(..., exists=True),
    layer: int | None = typer.Option(None),
    device: str = typer.Option("cpu"),
    feature_id: str = typer.Option("teacher-features"),
) -> None:
    adapter = create_teacher(teacher, model_path, layer=layer, device=device)
    output.mkdir(parents=True, exist_ok=True)
    rows = _iter_rows(manifest)
    record_ids: list[str] = []
    now = datetime.now(UTC)
    source_checksum = sha256_file(manifest)
    for line_number, row in enumerate(rows, start=1):
        item_id = str(row.get("id") or row.get("record_id") or "")
        relative_path = row.get("path") or row.get("uri")
        if not item_id or not isinstance(relative_path, str):
            raise typer.BadParameter(f"manifest row {line_number} requires id and path")
        waveform, sample_rate = read_pcm16_wav(audio_root / relative_path)
        features = adapter.extract(waveform, sample_rate)
        torch.save(
            {
                "values": features.values,
                "frame_rate_hz": features.frame_rate_hz,
                "teacher": features.teacher,
                "layer": features.layer,
                "source_sha256": row.get("sha256"),
            },
            output / f"{item_id}.pt",
        )
        record_ids.append(item_id)

    shard_uri = output.as_posix()
    # Content-address a placeholder shard checksum over sorted ids.
    from ripple.contracts.checksums import sha256_bytes

    shard_checksum = sha256_bytes("\n".join(record_ids).encode())
    teacher_identity = TeacherIdentity(
        model_id=teacher,
        model_checksum=sha256_file(model_path),
        layer=str(layer) if layer is not None else "default",
        projection_checksum=sha256_bytes(b"identity-projection"),
    )
    shard = FeatureShard(
        uri=shard_uri,
        checksum=shard_checksum,
        source_manifest_checksum=source_checksum,
        record_ids=tuple(record_ids),
        provenance=Provenance(
            source_uri=manifest.as_posix(),
            source_checksum=source_checksum,
            created_at=now,
            producer="ripple-features-extract",
            producer_version="1",
            license_id="local",
        ),
    )
    feature_manifest = FeatureManifest.create(
        feature_id=feature_id,
        teacher=teacher_identity,
        shards=(shard,),
    )
    manifest_path = output / "feature_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(feature_manifest.model_dump_json(indent=2))
        handle.write("\n")
    typer.echo(f"Wrote {len(record_ids)} feature files and {manifest_path}")


if __name__ == "__main__":
    app()
