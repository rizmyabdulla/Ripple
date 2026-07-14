"""Pre-registered gates for experimental Ripple model-family variants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class VariantStatus(StrEnum):
    PROPOSED = "proposed"
    RUNNING = "running"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class Gate:
    """One direction-aware acceptance gate."""

    metric: str
    threshold: float
    maximum: bool = True
    required: bool = True

    def passes(self, value: float) -> bool:
        return value <= self.threshold if self.maximum else value >= self.threshold


@dataclass(frozen=True)
class VariantSpec:
    """Experiment contract that prevents unmatched architecture promotion."""

    name: str
    hypothesis: str
    baseline_artifact: str
    gates: tuple[Gate, ...]
    requires_custom_kernel: bool = False
    required_backends: tuple[str, ...] = ("pytorch", "onnxruntime")
    status: VariantStatus = VariantStatus.PROPOSED
    metadata: Mapping[str, str] = field(default_factory=dict)

    def evaluate(self, metrics: Mapping[str, float]) -> tuple[bool, tuple[str, ...]]:
        failures: list[str] = []
        for gate in self.gates:
            value = metrics.get(gate.metric)
            if value is None:
                if gate.required:
                    failures.append(f"missing:{gate.metric}")
                continue
            if not gate.passes(value):
                relation = "<=" if gate.maximum else ">="
                failures.append(
                    f"{gate.metric}={value:.6g} does not satisfy {relation}{gate.threshold:.6g}"
                )
        return not failures, tuple(failures)


EDGE_INVARIANT_GATES = (
    Gate("p95_compute_ms", 20.0),
    Gate("rtf", 0.8),
    Gate("wer_delta_abs", 1.0),
    Gate("f0_pcc_delta_abs", 0.02),
    Gate("rss_growth_bytes_per_hour", 0.0),
    Gate("nan_count", 0.0),
)


def standard_variant(
    *,
    name: str,
    hypothesis: str,
    baseline_artifact: str,
    requires_custom_kernel: bool = False,
    extra_gates: tuple[Gate, ...] = (),
) -> VariantSpec:
    """Create a variant with the non-negotiable edge gates attached."""

    return VariantSpec(
        name=name,
        hypothesis=hypothesis,
        baseline_artifact=baseline_artifact,
        gates=EDGE_INVARIANT_GATES + extra_gates,
        requires_custom_kernel=requires_custom_kernel,
    )

