"""Canonical serialization and checksum helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SHA256_PREFIX = "sha256:"
SHA256_HEX_LENGTH = 64


def canonical_json(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return f"{SHA256_PREFIX}{hashlib.sha256(data).hexdigest()}"


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return f"{SHA256_PREFIX}{digest.hexdigest()}"


def is_sha256(value: str) -> bool:
    if not value.startswith(SHA256_PREFIX):
        return False
    digest = value.removeprefix(SHA256_PREFIX)
    return len(digest) == SHA256_HEX_LENGTH and all(char in "0123456789abcdef" for char in digest)


def without_keys(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    excluded = set(keys)
    return {key: item for key, item in value.items() if key not in excluded}

