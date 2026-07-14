"""Seal informal JSONL discovery manifests into DatasetManifest contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ripple.contracts.checksums import sha256_bytes, sha256_file
from ripple.contracts.manifest import Provenance
from ripple.data.manifest import (
    AudioRecord,
    ConsentRecord,
    ConsentStatus,
    DatasetManifest,
    write_manifest,
)
from ripple.data.split import assign_speaker_splits


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows


def seal_draft_manifest(
    draft_jsonl: Path,
    *,
    dataset_id: str,
    revision: str,
    output: Path,
    audio_root: Path | None = None,
    default_domain: str = "read",
    consent_status: ConsentStatus = ConsentStatus.GRANTED,
    commercial_use: bool = True,
    seed: int = 17,
    train_ratio: float = 0.9,
    development_ratio: float = 0.05,
    overwrite: bool = False,
) -> DatasetManifest:
    """Convert discovery JSONL rows into a sealed DatasetManifest."""
    rows = _read_jsonl(draft_jsonl)
    now = datetime.now(UTC)
    speaker_ids: list[str] = []
    prepared: list[dict[str, object]] = []
    for row in rows:
        record_id = str(row.get("id") or row.get("record_id") or "")
        relative = str(row.get("path") or row.get("uri") or "")
        if not record_id or not relative:
            raise ValueError("each draft row requires id/record_id and path/uri")
        speaker_raw = row.get("speaker_id")
        if speaker_raw:
            speaker = str(speaker_raw)
        else:
            parts = Path(relative).parts
            speaker = parts[0] if parts else "speaker"
        language = str(row.get("language") or "en")
        license_id = str(row.get("license") or row.get("license_id") or "")
        if not license_id:
            raise ValueError(f"record {record_id}: license is required")
        sample_rate = int(row.get("sample_rate") or 24_000)
        channels = int(row.get("channels") or 1)
        duration = float(row.get("duration_seconds") or 0.0)
        checksum = str(row.get("checksum") or "")
        if not checksum and row.get("sha256"):
            checksum = str(row["sha256"])
        if checksum and not checksum.startswith("sha256:"):
            checksum = f"sha256:{checksum}"
        if not checksum and audio_root is not None:
            checksum = sha256_file(audio_root / relative)
        if not checksum:
            raise ValueError(f"record {record_id}: checksum missing")
        if duration <= 0.0:
            frames = int(row.get("frames") or 0)
            if frames <= 0:
                raise ValueError(f"record {record_id}: duration_seconds/frames required")
            duration = frames / float(sample_rate)
        speaker_ids.append(speaker)
        prepared.append(
            {
                "record_id": record_id,
                "uri": relative.replace("\\", "/"),
                "checksum": checksum,
                "speaker_id": speaker,
                "language": language,
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "license_id": license_id,
                "domain": str(row.get("domain") or default_domain),
                "consent_reference": str(
                    row.get("consent_basis") or row.get("consent_reference") or f"consent://{speaker}"
                ),
            }
        )

    splits = assign_speaker_splits(
        speaker_ids,
        train_ratio=train_ratio,
        development_ratio=development_ratio,
        seed=seed,
    )
    records: list[AudioRecord] = []
    for item in prepared:
        speaker = str(item["speaker_id"])
        checksum = str(item["checksum"])
        records.append(
            AudioRecord(
                record_id=str(item["record_id"]),
                uri=str(item["uri"]),
                checksum=checksum,
                speaker_id=speaker,
                language=str(item["language"]),
                split=splits[speaker],
                duration_seconds=float(item["duration_seconds"]),
                sample_rate=int(item["sample_rate"]),
                channels=int(item["channels"]),
                license_id=str(item["license_id"]),
                consent=ConsentRecord(
                    status=consent_status,
                    reference=str(item["consent_reference"]),
                    commercial_use=commercial_use,
                    revocable=True,
                    checked_at=now,
                ),
                domain=str(item["domain"]),
                provenance=Provenance(
                    source_uri=str(item["uri"]),
                    source_checksum=checksum,
                    created_at=now,
                    producer="ripple-manifest-seal",
                    producer_version="1",
                    license_id=str(item["license_id"]),
                ),
            )
        )

    exclusion = sha256_bytes(b"no-exclusions")
    # Emit one consolidated sealed manifest containing all splits.
    manifest = DatasetManifest.create(
        dataset_id=dataset_id,
        revision=revision,
        records=tuple(records),
        exclusion_log_checksum=exclusion,
    )
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        output.unlink()
    write_manifest(manifest, output)
    return manifest


def summarize_manifest(manifest: DatasetManifest) -> dict[str, object]:
    hours = sum(record.duration_seconds for record in manifest.records) / 3600.0
    speakers = {record.speaker_id for record in manifest.records}
    languages: dict[str, float] = {}
    splits: dict[str, int] = {}
    for record in manifest.records:
        languages[record.language] = languages.get(record.language, 0.0) + record.duration_seconds
        splits[record.split.value] = splits.get(record.split.value, 0) + 1
    return {
        "dataset_id": manifest.dataset_id,
        "revision": manifest.revision,
        "checksum": manifest.checksum,
        "records": len(manifest.records),
        "speakers": len(speakers),
        "hours": hours,
        "languages_seconds": languages,
        "splits": splits,
    }
