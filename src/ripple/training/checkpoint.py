"""Reproducible checkpoint payloads and metadata validation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION = 1


def _git_value(args: list[str], default: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True, timeout=2
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return default


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CheckpointMetadata:
    stage: str
    step: int
    resolved_config: Mapping[str, Any]
    dataset_manifest_hash: str = ""
    feature_manifest_hash: str = ""
    teacher_hash: str = ""
    source_commit: str = "unknown"
    dirty_tree: bool = False
    random_seed: int = 0
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    torch_version: str = field(default_factory=lambda: torch.__version__)
    python_version: str = field(default_factory=platform.python_version)
    format_version: int = CHECKPOINT_FORMAT_VERSION

    @classmethod
    def capture(
        cls,
        *,
        stage: str,
        step: int,
        resolved_config: Mapping[str, Any],
        **kwargs: Any,
    ) -> CheckpointMetadata:
        commit = _git_value(["rev-parse", "HEAD"], "unknown")
        dirty = bool(_git_value(["status", "--porcelain"], ""))
        return cls(
            stage=stage,
            step=step,
            resolved_config=dict(resolved_config),
            source_commit=commit,
            dirty_tree=dirty,
            **kwargs,
        )


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    metadata: CheckpointMetadata,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    ema_state: Mapping[str, torch.Tensor] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Atomically save a checkpoint and return its SHA-256 digest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metadata": asdict(metadata),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": dict(ema_state) if ema_state is not None else None,
        "extra": dict(extra or {}),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[CheckpointMetadata, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    metadata = CheckpointMetadata(**payload["metadata"])
    if metadata.format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {metadata.format_version}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return metadata, {"ema": payload.get("ema"), "extra": payload.get("extra", {})}
