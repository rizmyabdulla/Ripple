"""Versioned teacher-feature cache contracts."""

from __future__ import annotations

from typing import Self

from pydantic import Field, field_validator, model_validator

from ripple.contracts.checksums import is_sha256, sha256_json
from ripple.contracts.manifest import ContractModel, Provenance


class TeacherIdentity(ContractModel):
    model_id: str = Field(min_length=1)
    model_checksum: str
    layer: str = Field(min_length=1)
    projection_checksum: str

    @field_validator("model_checksum", "projection_checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if not is_sha256(value):
            raise ValueError("checksum must use lowercase sha256:<64 hex> form")
        return value


class FeatureShard(ContractModel):
    uri: str = Field(min_length=1)
    checksum: str
    source_manifest_checksum: str
    record_ids: tuple[str, ...] = Field(min_length=1)
    provenance: Provenance

    @field_validator("checksum", "source_manifest_checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if not is_sha256(value):
            raise ValueError("checksum must use lowercase sha256:<64 hex> form")
        return value

    @model_validator(mode="after")
    def validate_records(self) -> Self:
        if len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError("record IDs within a shard must be unique")
        return self


class FeatureManifest(ContractModel):
    schema_version: str = Field(default="ripple-features-1", pattern=r"^ripple-features-1$")
    feature_id: str = Field(min_length=1)
    teacher: TeacherIdentity
    shards: tuple[FeatureShard, ...] = Field(min_length=1)
    checksum: str

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if not is_sha256(value):
            raise ValueError("checksum must use lowercase sha256:<64 hex> form")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        uris = [shard.uri for shard in self.shards]
        if len(uris) != len(set(uris)):
            raise ValueError("feature shard URIs must be unique")
        records = [record_id for shard in self.shards for record_id in shard.record_ids]
        if len(records) != len(set(records)):
            raise ValueError("a record cannot occur in multiple feature shards")
        if self.checksum != self.computed_checksum():
            raise ValueError("feature manifest checksum does not match content")
        return self

    def computed_checksum(self) -> str:
        return sha256_json(self.model_dump(mode="json", exclude={"checksum"}))

    @classmethod
    def create(
        cls,
        *,
        feature_id: str,
        teacher: TeacherIdentity,
        shards: tuple[FeatureShard, ...],
    ) -> Self:
        unverified = cls.model_construct(
            feature_id=feature_id,
            teacher=teacher,
            shards=shards,
        )
        return cls(
            feature_id=feature_id,
            teacher=teacher,
            shards=shards,
            checksum=sha256_json(
                unverified.model_dump(mode="json", exclude={"checksum"})
            ),
        )

