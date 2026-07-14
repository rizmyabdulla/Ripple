"""Shared provenance and deployable artifact manifest contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ripple.contracts.checksums import is_sha256, sha256_file, sha256_json


class ContractModel(BaseModel):
    """Strict immutable base for serialized contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(ContractModel):
    """Traceability metadata for generated or ingested material."""

    source_uri: str = Field(min_length=1)
    source_checksum: str
    created_at: datetime
    producer: str = Field(min_length=1)
    producer_version: str = Field(min_length=1)
    source_revision: str | None = None
    license_id: str = Field(min_length=1)

    @field_validator("source_checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if not is_sha256(value):
            raise ValueError("checksum must use lowercase sha256:<64 hex> form")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)


class ArtifactFile(ContractModel):
    path: str = Field(min_length=1)
    checksum: str
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("artifact path must be a normalized relative POSIX path")
        return str(path)

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if not is_sha256(value):
            raise ValueError("checksum must use lowercase sha256:<64 hex> form")
        return value


class ArtifactManifest(ContractModel):
    """Content-addressed manifest for a portable Ripple artifact."""

    schema_version: str = Field(default="ripple-artifact-1", pattern=r"^ripple-artifact-1$")
    artifact_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    rif_version: str = Field(default="RIF-1", pattern=r"^RIF-1$")
    profile_schema_version: str = Field(
        default="ripple-speaker-profile-1", pattern=r"^ripple-speaker-profile-1$"
    )
    state_schema_version: str = Field(
        default="ripple-stream-state-1", pattern=r"^ripple-stream-state-1$"
    )
    resolved_config_checksum: str
    provenance: Provenance
    files: tuple[ArtifactFile, ...] = Field(min_length=1)
    manifest_checksum: str

    @field_validator("resolved_config_checksum", "manifest_checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if not is_sha256(value):
            raise ValueError("checksum must use lowercase sha256:<64 hex> form")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact file paths must be unique")
        if self.manifest_checksum != self.computed_checksum():
            raise ValueError("manifest_checksum does not match canonical manifest content")
        return self

    def computed_checksum(self) -> str:
        payload = self.model_dump(mode="json", exclude={"manifest_checksum"})
        return sha256_json(payload)

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        model_version: str,
        resolved_config_checksum: str,
        provenance: Provenance,
        files: tuple[ArtifactFile, ...],
        rif_version: str = "RIF-1",
        profile_schema_version: str = "ripple-speaker-profile-1",
        state_schema_version: str = "ripple-stream-state-1",
    ) -> Self:
        unverified = cls.model_construct(
            artifact_id=artifact_id,
            model_version=model_version,
            resolved_config_checksum=resolved_config_checksum,
            provenance=provenance,
            files=files,
            rif_version=rif_version,
            profile_schema_version=profile_schema_version,
            state_schema_version=state_schema_version,
        )
        return cls(
            artifact_id=artifact_id,
            model_version=model_version,
            resolved_config_checksum=resolved_config_checksum,
            provenance=provenance,
            files=files,
            rif_version=rif_version,
            profile_schema_version=profile_schema_version,
            state_schema_version=state_schema_version,
            manifest_checksum=sha256_json(
                unverified.model_dump(mode="json", exclude={"manifest_checksum"})
            ),
        )

    def verify_files(self, root: str | Path) -> None:
        base = Path(root).resolve()
        for item in self.files:
            path = (base / Path(item.path)).resolve()
            if base not in path.parents:
                raise ValueError(f"artifact file escapes root: {item.path}")
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != item.size_bytes:
                raise ValueError(f"artifact size mismatch: {item.path}")
            if sha256_file(path) != item.checksum:
                raise ValueError(f"artifact checksum mismatch: {item.path}")

