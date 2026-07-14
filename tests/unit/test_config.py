from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ripple.contracts.config import ModelConfig, ResolvedConfig, load_config

PROJECT_ROOT = Path(__file__).parents[2]


def test_repository_config_resolves_and_is_content_addressed() -> None:
    config = load_config(PROJECT_ROOT / "configs")
    assert config.model.sample_rate == 24_000
    assert config.model.frame_samples == 480
    assert config.model.lookahead_frames == 0
    assert config.data.canonical_sample_rate == config.model.sample_rate
    assert config.checksum == config.computed_checksum()

    restored = ResolvedConfig.model_validate_json(config.model_dump_json())
    assert restored == config


def test_invalid_stride_and_checksum_are_rejected() -> None:
    with pytest.raises(ValidationError, match="stride product"):
        ModelConfig(analysis_strides=(2, 2))

    config = load_config(PROJECT_ROOT / "configs")
    tampered = config.model_dump(mode="json")
    tampered["benchmark"]["deadline_ms"] = 99.0
    with pytest.raises(ValidationError, match="checksum"):
        ResolvedConfig.model_validate(tampered)

