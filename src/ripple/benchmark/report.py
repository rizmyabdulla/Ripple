"""Machine-readable benchmark report schema and atomic JSON writer."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPORT_SCHEMA_VERSION = 1


def _git(args: list[str], default: Any) -> Any:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True, timeout=2
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return default


def environment_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "pid": os.getpid(),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class BenchmarkReport:
    model_id: str
    runtime: str
    precision: str
    sample_rate: int
    chunk_samples: int
    sections: dict[str, Any] = field(default_factory=dict)
    resolved_config: Mapping[str, Any] = field(default_factory=dict)
    manifest_hashes: Mapping[str, str] = field(default_factory=dict)
    artifact_hash: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: int = REPORT_SCHEMA_VERSION
    source_commit: str = field(default_factory=lambda: _git(["rev-parse", "HEAD"], "unknown"))
    dirty_tree: bool = field(default_factory=lambda: bool(_git(["status", "--porcelain"], "")))
    environment: Mapping[str, Any] = field(default_factory=environment_metadata)

    def add(self, name: str, values: Any) -> None:
        self.sections[name] = _jsonable(values)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return destination

    def to_text(self) -> str:
        lines = [
            f"Ripple benchmark: {self.model_id}",
            f"Runtime: {self.runtime} ({self.precision})",
            f"Audio: {self.sample_rate} Hz, {self.chunk_samples} samples/chunk",
        ]
        for section, values in self.sections.items():
            lines.append(f"\n[{section}]")
            if isinstance(values, Mapping):
                lines.extend(f"{key}: {value}" for key, value in values.items())
            else:
                lines.append(str(values))
        return "\n".join(lines) + "\n"
