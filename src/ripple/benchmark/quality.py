"""Quality-delta and release-gate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MetricGate:
    metric: str
    maximum_degradation: float
    higher_is_better: bool = True


def compare_metrics(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
) -> dict[str, float]:
    shared = reference.keys() & candidate.keys()
    return {name: candidate[name] - reference[name] for name in sorted(shared)}


def evaluate_gates(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
    gates: list[MetricGate],
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for gate in gates:
        if gate.metric not in reference or gate.metric not in candidate:
            result[gate.metric] = False
            continue
        delta = candidate[gate.metric] - reference[gate.metric]
        degradation = -delta if gate.higher_is_better else delta
        result[gate.metric] = degradation <= gate.maximum_degradation
    return result
