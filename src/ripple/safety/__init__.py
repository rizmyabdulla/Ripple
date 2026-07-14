"""Production safety policy, watermark, and provenance contracts."""

from .consent import (
    ConsentDenied,
    ConsentRecord,
    ProfilePolicy,
    ProfileUse,
    parse_allowed_uses,
)
from .provenance import (
    HMACProvenanceSigner,
    ProvenanceRecord,
    ProvenanceSigner,
    SignedProvenance,
    sha256_file,
    sign_record,
)
from .watermark import (
    AudioWatermarker,
    WatermarkDetection,
    WatermarkEmbedResult,
    WatermarkPayload,
    WatermarkPolicy,
    WatermarkRequired,
)

__all__ = [
    "AudioWatermarker",
    "ConsentDenied",
    "ConsentRecord",
    "HMACProvenanceSigner",
    "ProfilePolicy",
    "ProfileUse",
    "ProvenanceRecord",
    "ProvenanceSigner",
    "SignedProvenance",
    "WatermarkDetection",
    "WatermarkEmbedResult",
    "WatermarkPayload",
    "WatermarkPolicy",
    "WatermarkRequired",
    "parse_allowed_uses",
    "sha256_file",
    "sign_record",
]
