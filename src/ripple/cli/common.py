"""Shared helpers for thin Typer command-line adapters."""

from __future__ import annotations

import importlib
import json
import wave
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import typer
from torch import Tensor

from ripple.safety import ConsentRecord, ProfileUse, parse_allowed_uses


def load_object(spec: str) -> Any:
    if ":" not in spec:
        raise typer.BadParameter("backend must use module:object syntax")
    module_name, object_name = spec.split(":", 1)
    try:
        return getattr(importlib.import_module(module_name), object_name)
    except (ImportError, AttributeError) as exc:
        raise typer.BadParameter(f"cannot load backend {spec!r}: {exc}") from exc


def build_backend(spec: str, **kwargs: Any) -> Any:
    factory: Callable[..., Any] = load_object(spec)
    return factory(**kwargs)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_consent(path: Path) -> ConsentRecord:
    value = read_json(path)
    try:
        return ConsentRecord(
            subject_id=value["subject_id"],
            profile_id=value["profile_id"],
            allowed_uses=parse_allowed_uses(value["allowed_uses"]),
            granted_at=datetime.fromisoformat(value["granted_at"]),
            expires_at=(
                datetime.fromisoformat(value["expires_at"])
                if value.get("expires_at")
                else None
            ),
            revoked_at=(
                datetime.fromisoformat(value["revoked_at"])
                if value.get("revoked_at")
                else None
            ),
            evidence_uri=value.get("evidence_uri"),
            policy_version=value.get("policy_version", "1"),
            attributes=value.get("attributes", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise typer.BadParameter(f"invalid consent record: {exc}") from exc


def profile_use(value: str) -> ProfileUse:
    try:
        return ProfileUse(value)
    except ValueError as exc:
        raise typer.BadParameter(f"invalid profile use: {value}") from exc


def read_pcm16_wav(path: Path) -> tuple[Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise typer.BadParameter(f"{path}: only PCM16 WAV is supported")
    values = torch.frombuffer(bytearray(frames), dtype=torch.int16).float()
    values = values.reshape(-1, channels).mean(dim=1) / 32768.0
    return values, sample_rate
