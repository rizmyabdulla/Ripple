"""Manifest-first data ingestion and feature cache contracts."""

from ripple.data.canonicalize import canonicalize_file, canonicalize_tree
from ripple.data.dataset import RippleAudioDataset, resolve_audio_path
from ripple.data.features import FeatureManifest, FeatureShard, TeacherIdentity
from ripple.data.filters import QualityMetrics, QualityPolicy, rejection_reasons
from ripple.data.manifest import (
    AudioRecord,
    ConsentRecord,
    ConsentStatus,
    DatasetManifest,
    DatasetSplit,
    read_manifest,
    write_manifest,
)
from ripple.data.sampler import sample_indices, sampling_probabilities
from ripple.data.seal import seal_draft_manifest, summarize_manifest
from ripple.data.split import assign_speaker_splits

__all__ = [
    "AudioRecord",
    "ConsentRecord",
    "ConsentStatus",
    "DatasetManifest",
    "DatasetSplit",
    "FeatureManifest",
    "FeatureShard",
    "QualityMetrics",
    "QualityPolicy",
    "RippleAudioDataset",
    "TeacherIdentity",
    "assign_speaker_splits",
    "canonicalize_file",
    "canonicalize_tree",
    "read_manifest",
    "rejection_reasons",
    "resolve_audio_path",
    "sample_indices",
    "sampling_probabilities",
    "seal_draft_manifest",
    "summarize_manifest",
    "write_manifest",
]
