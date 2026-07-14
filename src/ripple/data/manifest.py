"""Immutable, content-addressed dataset manifests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import Field, field_validator, model_validator

from ripple.contracts.checksums import is_sha256, sha256_json
from ripple.contracts.manifest import ContractModel, Provenance


class DatasetSplit(StrEnum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    TEST = "test"


class ConsentStatus(StrEnum):
    GRANTED = "granted"
    PUBLIC_DOMAIN = "public_domain"
    RESTRICTED = "restricted"


class ConsentRecord(ContractModel):
    status: ConsentStatus
    reference: str = Field(min_length=1)
    commercial_use: bool
    revocable: bool
    checked_at: datetime

    @field_validator("checked_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checked_at must be timezone-aware")
        return value.astimezone(UTC)


class AudioRecord(ContractModel):
    record_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    checksum: str
    speaker_id: str = Field(min_length=1)
    language: str = Field(min_length=2, pattern=r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
    split: DatasetSplit
    duration_seconds: float = Field(gt=0.0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(ge=1, le=2)
    license_id: str = Field(min_length=1)
    consent: ConsentRecord
    domain: str = Field(min_length=1)
    provenance: Provenance

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if not is_sha256(value):
            raise ValueError("checksum must use lowercase sha256:<64 hex> form")
        return value


class DatasetManifest(ContractModel):
    schema_version: str = Field(default="ripple-dataset-1", pattern=r"^ripple-dataset-1$")
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    records: tuple[AudioRecord, ...] = Field(min_length=1)
    exclusion_log_checksum: str
    checksum: str

    @field_validator("exclusion_log_checksum", "checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if not is_sha256(value):
            raise ValueError("checksum must use lowercase sha256:<64 hex> form")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        ids = [record.record_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("record_id values must be unique")
        locations = [(record.uri, record.checksum) for record in self.records]
        if len(locations) != len(set(locations)):
            raise ValueError("duplicate URI/checksum pairs are not allowed")
        speaker_splits: dict[str, set[DatasetSplit]] = {}
        for record in self.records:
            speaker_splits.setdefault(record.speaker_id, set()).add(record.split)
            if record.consent.status is ConsentStatus.RESTRICTED and record.consent.commercial_use:
                raise ValueError("restricted consent cannot grant commercial use")
        overlapping = [speaker for speaker, splits in speaker_splits.items() if len(splits) > 1]
        if overlapping:
            raise ValueError(f"speakers must be split-disjoint: {sorted(overlapping)}")
        if self.checksum != self.computed_checksum():
            raise ValueError("dataset manifest checksum does not match canonical content")
        return self

    def computed_checksum(self) -> str:
        return sha256_json(self.model_dump(mode="json", exclude={"checksum"}))

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        revision: str,
        records: tuple[AudioRecord, ...],
        exclusion_log_checksum: str,
    ) -> Self:
        unverified = cls.model_construct(
            dataset_id=dataset_id,
            revision=revision,
            records=records,
            exclusion_log_checksum=exclusion_log_checksum,
        )
        return cls(
            dataset_id=dataset_id,
            revision=revision,
            records=records,
            exclusion_log_checksum=exclusion_log_checksum,
            checksum=sha256_json(
                unverified.model_dump(mode="json", exclude={"checksum"})
            ),
        )


def read_manifest(path: str | Path) -> DatasetManifest:
    with Path(path).open("r", encoding="utf-8") as stream:
        return DatasetManifest.model_validate(json.load(stream))


def write_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    """Create a manifest once; refusing overwrite preserves release immutability."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(manifest.model_dump_json(indent=2))
        stream.write("\n")

