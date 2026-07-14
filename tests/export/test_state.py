from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from ripple.export.state import flatten_state, unflatten_state, validate_flat_state


@dataclass
class MixerState:
    cache: torch.Tensor
    phase: torch.Tensor


def test_state_flattening_is_named_stable_and_reversible() -> None:
    state = {
        "z": torch.zeros(1, 2),
        "mixer": MixerState(torch.ones(1, 3), torch.tensor([0.25])),
    }
    flattened = flatten_state(state)

    assert [spec.name for spec in flattened.tensor_specs] == [
        "state_mixer_cache",
        "state_mixer_phase",
        "state_z",
    ]
    rebuilt = unflatten_state(flattened.tensors, flattened.tree)
    assert isinstance(rebuilt["mixer"], MixerState)
    torch.testing.assert_close(rebuilt["mixer"].cache, state["mixer"].cache)
    validate_flat_state(flattened.tensors, flattened.tensor_specs)


def test_state_rejects_hidden_python_scalars() -> None:
    with pytest.raises(TypeError, match="contain tensors only"):
        flatten_state({"counter": 1})


def test_state_shape_validation_fails_before_backend_dispatch() -> None:
    flattened = flatten_state({"cache": torch.zeros(1, 3)})
    with pytest.raises(ValueError, match="expected shape"):
        validate_flat_state((torch.zeros(1, 4),), flattened.tensor_specs)
