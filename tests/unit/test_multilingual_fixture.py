from __future__ import annotations

from pathlib import Path

from ripple.contracts.config import load_config
from ripple.data.manifest import DatasetSplit, read_manifest

PROJECT_ROOT = Path(__file__).parents[2]


def test_multilingual_fixture_manifests_are_speaker_disjoint_and_language_tagged() -> None:
    config = load_config(PROJECT_ROOT / "configs")
    assert config.data.languages == (
        "en-US",
        "es-ES",
        "fr-FR",
        "de-DE",
        "hi-IN",
        "ja-JP",
    )
    manifests = {
        DatasetSplit.TRAIN: read_manifest(PROJECT_ROOT / config.data.manifest_uri),
        DatasetSplit.DEVELOPMENT: read_manifest(
            PROJECT_ROOT / config.data.development_manifest_uri
        ),
        DatasetSplit.TEST: read_manifest(PROJECT_ROOT / config.data.test_manifest_uri),
    }
    seen_speakers: set[str] = set()
    seen_languages: set[str] = set()
    for split, manifest in manifests.items():
        assert manifest.dataset_id == "multilingual-core-fixture"
        for record in manifest.records:
            assert record.split is split
            assert record.speaker_id not in seen_speakers
            seen_speakers.add(record.speaker_id)
            seen_languages.add(record.language)
    assert seen_languages == set(config.data.languages)
