"""Typed immutable configuration and YAML resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import Field, model_validator

from ripple.contracts.checksums import is_sha256, sha256_json
from ripple.contracts.manifest import ContractModel


class ModelConfig(ContractModel):
    sample_rate: int = Field(default=24_000, ge=1)
    frame_samples: int = Field(default=480, ge=1)
    semantic_classes: int = Field(default=256, ge=256, le=512)
    semantic_embed_dim: int = Field(default=128)
    analysis_strides: tuple[int, ...] = (3, 4, 5, 8)
    speaker_global_dim: int = Field(default=192)
    speaker_token_count: int = Field(default=4, ge=4, le=8)
    speaker_token_dim: int = Field(default=64)
    lookahead_frames: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        product = 1
        for stride in self.analysis_strides:
            if stride <= 0:
                raise ValueError("analysis strides must be positive")
            product *= stride
        if product != self.frame_samples:
            raise ValueError("analysis stride product must equal frame_samples")
        if self.sample_rate % self.frame_samples:
            raise ValueError("frame_samples must divide sample_rate")
        dimensions = (
            self.semantic_embed_dim,
            self.speaker_global_dim,
            self.speaker_token_dim,
        )
        if dimensions != (128, 192, 64):
            raise ValueError("RIF-1/profile dimensions are fixed at 128/192/64")
        return self


class DataConfig(ContractModel):
    manifest_uri: str = "fixtures/multilingual-core/manifests/train.json"
    canonical_sample_rate: int = Field(default=24_000, ge=1)
    channels: int = Field(default=1, ge=1, le=2)
    verify_checksums: bool = True
    require_consent: bool = True
    seed: int = Field(default=17, ge=0)
    languages: tuple[str, ...] = (
        "en-US",
        "es-ES",
        "fr-FR",
        "de-DE",
        "hi-IN",
        "ja-JP",
    )
    development_manifest_uri: str = "fixtures/multilingual-core/manifests/development.json"
    test_manifest_uri: str = "fixtures/multilingual-core/manifests/test.json"


class TrainConfig(ContractModel):
    seed: int = Field(default=17, ge=0)
    deterministic_algorithms: bool = True
    precision: str = Field(default="bf16-mixed", pattern=r"^(32|bf16-mixed|16-mixed)$")
    stage: str = Field(default="semantic", min_length=1)
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)


class EvalConfig(ContractModel):
    manifest_uri: str = "fixtures/multilingual-core/manifests/test.json"
    speaker_disjoint: bool = True
    zero_lookahead_required: bool = True
    long_session_seconds: int = Field(default=3600, ge=1)


class ExportConfig(ContractModel):
    backend: str = Field(
        default="onnxruntime",
        pattern=r"^(onnxruntime|executorch|coreml|litert|tensorrt)$",
    )
    opset: int = Field(default=20, ge=18)
    fixed_batch_size: int = Field(default=1, ge=1, le=1)
    verify_streaming_equivalence: bool = True


class BenchmarkConfig(ContractModel):
    warmup_iterations: int = Field(default=50, ge=1)
    measured_iterations: int = Field(default=500, ge=1)
    deadline_ms: float = Field(default=20.0, gt=0.0)
    target_rtf: float = Field(default=0.5, gt=0.0)
    state_budget_bytes: int = Field(default=8 * 1024 * 1024, gt=0, le=8 * 1024 * 1024)


class ResolvedConfig(ContractModel):
    schema_version: str = Field(default="ripple-config-1", pattern=r"^ripple-config-1$")
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    export: ExportConfig
    benchmark: BenchmarkConfig
    checksum: str

    @model_validator(mode="after")
    def validate_resolved(self) -> Self:
        if not is_sha256(self.checksum):
            raise ValueError("checksum must use lowercase sha256:<64 hex> form")
        if self.data.canonical_sample_rate != self.model.sample_rate:
            raise ValueError("data and model sample rates must match")
        if self.checksum != self.computed_checksum():
            raise ValueError("resolved config checksum does not match content")
        return self

    def computed_checksum(self) -> str:
        return sha256_json(self.model_dump(mode="json", exclude={"checksum"}))

    @classmethod
    def create(
        cls,
        *,
        model: ModelConfig,
        data: DataConfig,
        train: TrainConfig,
        eval: EvalConfig,
        export: ExportConfig,
        benchmark: BenchmarkConfig,
    ) -> Self:
        unverified = cls.model_construct(
            model=model,
            data=data,
            train=train,
            eval=eval,
            export=export,
            benchmark=benchmark,
        )
        return cls(
            model=model,
            data=data,
            train=train,
            eval=eval,
            export=export,
            benchmark=benchmark,
            checksum=sha256_json(
                unverified.model_dump(mode="json", exclude={"checksum"})
            ),
        )


_CONFIG_FILES = {
    "model": ("model", "edge.yaml"),
    "data": ("data", "default.yaml"),
    "train": ("train", "default.yaml"),
    "eval": ("eval", "default.yaml"),
    "export": ("export", "default.yaml"),
    "benchmark": ("benchmark", "default.yaml"),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def load_config(root: str | Path) -> ResolvedConfig:
    """Load the six non-executable YAML sections and freeze the resolved result."""
    base = Path(root)
    values: dict[str, dict[str, Any]] = {
        name: _read_yaml(base.joinpath(*parts))
        for name, parts in _CONFIG_FILES.items()
    }
    return ResolvedConfig.create(
        model=ModelConfig.model_validate(values["model"]),
        data=DataConfig.model_validate(values["data"]),
        train=TrainConfig.model_validate(values["train"]),
        eval=EvalConfig.model_validate(values["eval"]),
        export=ExportConfig.model_validate(values["export"]),
        benchmark=BenchmarkConfig.model_validate(values["benchmark"]),
    )

