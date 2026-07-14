"""Canonical provenance records and pluggable signing interfaces."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProvenanceRecord:
    artifact_id: str
    artifact_sha256: str
    model_id: str
    model_sha256: str
    created_at: str
    operation: str
    profile_id: str | None = None
    consent_policy_version: str | None = None
    watermark_scheme: str | None = None
    source_ids: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = "ripple-provenance-1"

    def canonical_bytes(self) -> bytes:
        if not self.artifact_id or not self.artifact_sha256:
            raise ValueError("artifact identity and checksum are required")
        payload = asdict(self)
        payload["metadata"] = dict(sorted(self.metadata.items()))
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    @property
    def record_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@runtime_checkable
class ProvenanceSigner(Protocol):
    algorithm: str
    key_id: str

    def sign(self, payload: bytes) -> bytes: ...

    def verify(self, payload: bytes, signature: bytes) -> bool: ...


class HMACProvenanceSigner:
    """Local signer useful for deployments backed by managed secret storage."""

    algorithm = "HMAC-SHA256"

    def __init__(self, key: bytes, *, key_id: str) -> None:
        if len(key) < 32:
            raise ValueError("HMAC key must be at least 32 bytes")
        if not key_id:
            raise ValueError("key_id is required")
        self._key = key
        self.key_id = key_id

    def sign(self, payload: bytes) -> bytes:
        return hmac.new(self._key, payload, hashlib.sha256).digest()

    def verify(self, payload: bytes, signature: bytes) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)


@dataclass(frozen=True)
class SignedProvenance:
    record: ProvenanceRecord
    signature_hex: str
    algorithm: str
    key_id: str


def sign_record(
    record: ProvenanceRecord, signer: ProvenanceSigner
) -> SignedProvenance:
    return SignedProvenance(
        record,
        signer.sign(record.canonical_bytes()).hex(),
        signer.algorithm,
        signer.key_id,
    )
