"""Deterministic Ripple artifact bundles, checksums, and signatures."""

from __future__ import annotations

import hashlib
import hmac
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .backends import OptionalDependencyError

MANIFEST_NAME = "manifest.json"
SIGNATURE_NAME = "manifest.sig"
_REQUIRED_MANIFEST_FIELDS = (
    "family",
    "model_version",
    "rif_version",
    "speaker_profile_version",
    "sample_rate",
    "chunk_samples",
    "lookahead_samples",
    "backend",
    "precision",
    "state_tensors",
)


@dataclass(frozen=True)
class ArtifactVerification:
    path: Path
    manifest: dict[str, Any]
    checked_files: tuple[str, ...]
    signature_verified: bool


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or path.parts[0].endswith(":")
    ):
        raise ValueError(f"unsafe artifact member name: {name!r}")
    clean = path.as_posix()
    if clean in (MANIFEST_NAME, SIGNATURE_NAME):
        raise ValueError(f"{clean} is reserved by the artifact format")
    return clean


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = [field for field in _REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"artifact manifest missing required fields: {missing}")
    if manifest["family"] != "ripple-vc":
        raise ValueError("manifest family must be 'ripple-vc'")
    if int(manifest["sample_rate"]) <= 0 or int(manifest["chunk_samples"]) <= 0:
        raise ValueError("sample_rate and chunk_samples must be positive")
    if not isinstance(manifest["state_tensors"], list):
        raise ValueError("state_tensors must be a list")
    for state in manifest["state_tensors"]:
        required = {"name", "dtype", "shape", "layout", "reset_policy"}
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError("each state tensor requires name/dtype/shape/layout/reset_policy")


def _read_payload(source: bytes | bytearray | memoryview | str | Path) -> bytes:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    return Path(source).read_bytes()


def _load_ed25519_private(key: bytes) -> Any:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as error:
        raise OptionalDependencyError(
            "Ed25519 artifact signing", ("cryptography",), "signing"
        ) from error
    if key.startswith(b"-----BEGIN"):
        private_key = serialization.load_pem_private_key(key, password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError("PEM key is not an Ed25519 private key")
        return private_key
    if len(key) != 32:
        raise ValueError("raw Ed25519 private keys must contain 32 bytes")
    return Ed25519PrivateKey.from_private_bytes(key)


def _sign(payload: bytes, key: bytes, algorithm: str) -> bytes:
    if algorithm == "hmac-sha256":
        if len(key) < 32:
            raise ValueError("HMAC signing keys must contain at least 32 bytes")
        return hmac.new(key, payload, hashlib.sha256).digest()
    if algorithm == "ed25519":
        return _load_ed25519_private(key).sign(payload)
    raise ValueError(f"unsupported signature algorithm: {algorithm}")


def build_artifact_bundle(
    destination: str | Path,
    *,
    files: Mapping[str, bytes | bytearray | memoryview | str | Path],
    manifest: Mapping[str, Any],
    signing_key: bytes | None = None,
    signature_algorithm: str = "ed25519",
    key_id: str | None = None,
) -> Path:
    """Write a deterministic ZIP bundle with content hashes and detached signature."""

    payloads: dict[str, bytes] = {}
    for name, source in files.items():
        safe_name = _safe_member_name(name)
        if safe_name in payloads:
            raise ValueError(f"duplicate artifact member: {safe_name}")
        payloads[safe_name] = _read_payload(source)

    completed = dict(manifest)
    completed["files"] = sorted(payloads)
    completed["checksums"] = {
        name: f"sha256:{sha256_bytes(payloads[name])}" for name in sorted(payloads)
    }
    if signing_key is not None:
        completed["signature"] = {
            "algorithm": signature_algorithm,
            "file": SIGNATURE_NAME,
            "key_id": key_id or "unspecified",
        }
    else:
        completed.pop("signature", None)
    _validate_manifest(completed)

    manifest_payload = canonical_manifest_bytes(completed)
    signature = (
        None
        if signing_key is None
        else _sign(manifest_payload, signing_key, signature_algorithm)
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        members = {**payloads, MANIFEST_NAME: manifest_payload}
        if signature is not None:
            members[SIGNATURE_NAME] = signature
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name])
    return output


def _load_ed25519_public(key: bytes) -> Any:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as error:
        raise OptionalDependencyError(
            "Ed25519 artifact verification", ("cryptography",), "signing"
        ) from error
    if key.startswith(b"-----BEGIN"):
        public_key = serialization.load_pem_public_key(key)
        if not isinstance(public_key, Ed25519PublicKey):
            raise ValueError("PEM key is not an Ed25519 public key")
        return public_key
    if len(key) != 32:
        raise ValueError("raw Ed25519 public keys must contain 32 bytes")
    return Ed25519PublicKey.from_public_bytes(key)


def _verify_signature(
    payload: bytes, signature: bytes, key: bytes, algorithm: str
) -> None:
    if algorithm == "hmac-sha256":
        expected = hmac.new(key, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("artifact HMAC signature is invalid")
        return
    if algorithm == "ed25519":
        try:
            _load_ed25519_public(key).verify(signature, payload)
        except Exception as error:
            if isinstance(error, (OptionalDependencyError, ValueError)):
                raise
            raise ValueError("artifact Ed25519 signature is invalid") from error
        return
    raise ValueError(f"unsupported signature algorithm: {algorithm}")


def verify_artifact_bundle(
    path: str | Path,
    *,
    verification_key: bytes | None = None,
    require_signature: bool = False,
) -> ArtifactVerification:
    """Validate member paths, manifest schema, hashes, and optional signature."""

    bundle = Path(path)
    with zipfile.ZipFile(bundle, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("artifact contains duplicate ZIP members")
        for name in names:
            normalized = PurePosixPath(name)
            if (
                not normalized.parts
                or normalized.is_absolute()
                or ".." in normalized.parts
                or normalized.parts[0].endswith(":")
            ):
                raise ValueError(f"unsafe artifact member name: {name!r}")
        if MANIFEST_NAME not in names:
            raise ValueError("artifact has no manifest.json")
        manifest_payload = archive.read(MANIFEST_NAME)
        manifest = json.loads(manifest_payload)
        _validate_manifest(manifest)

        expected_files = tuple(manifest.get("files", ()))
        checksums = manifest.get("checksums", {})
        if set(expected_files) != set(checksums):
            raise ValueError("manifest files and checksums differ")
        allowed_members = set(expected_files) | {MANIFEST_NAME}
        if manifest.get("signature") is not None:
            allowed_members.add(SIGNATURE_NAME)
        unexpected = set(names) - allowed_members
        if unexpected:
            raise ValueError(f"artifact contains unmanifested members: {unexpected}")
        for name in expected_files:
            if name not in names:
                raise ValueError(f"artifact member is missing: {name}")
            expected = checksums[name]
            actual = f"sha256:{sha256_bytes(archive.read(name))}"
            if not hmac.compare_digest(str(expected), actual):
                raise ValueError(f"checksum mismatch for {name}")

        signature_record = manifest.get("signature")
        if require_signature and signature_record is None:
            raise ValueError("artifact signature is required")
        signature_verified = False
        if signature_record is not None and verification_key is not None:
            signature_name = signature_record.get("file")
            if signature_name != SIGNATURE_NAME or signature_name not in names:
                raise ValueError("artifact signature file is missing or invalid")
            canonical = canonical_manifest_bytes(manifest)
            if manifest_payload != canonical:
                raise ValueError("signed manifest is not canonically encoded")
            _verify_signature(
                canonical,
                archive.read(signature_name),
                verification_key,
                str(signature_record.get("algorithm")),
            )
            signature_verified = True
        elif require_signature and verification_key is None:
            raise ValueError("verification_key is required to verify artifact signature")

    return ArtifactVerification(
        path=bundle,
        manifest=manifest,
        checked_files=expected_files,
        signature_verified=signature_verified,
    )
