"""Quantization-aware training preparation and conversion."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import warnings

import torch
from torch import nn

from .ptq import QuantizationBackendUnavailable, available_backends


@dataclass(frozen=True)
class QATConfig:
    backend: str = "x86"
    excluded_module_names: tuple[str, ...] = ()
    preserve_name_fragments: tuple[str, ...] = (
        "norm",
        "oscillator",
        "waveform_projection",
        "prosody",
    )
    inplace: bool = False
    strict_backend: bool = False


@dataclass
class QATResult:
    model: nn.Module
    backend: str
    prepared: bool
    warning: str = ""


def prepare_qat(model: nn.Module, config: QATConfig | None = None) -> QATResult:
    config = config or QATConfig()
    target = model if config.inplace else copy.deepcopy(model)
    if config.backend not in available_backends():
        message = f"QAT backend {config.backend!r} unavailable; available={available_backends()}"
        if config.strict_backend:
            raise QuantizationBackendUnavailable(message)
        return QATResult(target, config.backend, False, message)
    previous_engine = torch.backends.quantized.engine
    try:
        torch.backends.quantized.engine = config.backend
        target.train()
        target.qconfig = torch.ao.quantization.get_default_qat_qconfig(config.backend)
        for name, module in target.named_modules():
            excluded = any(
                name == item or name.startswith(item + ".") for item in config.excluded_module_names
            )
            sensitive = any(fragment.casefold() in name.casefold() for fragment in config.preserve_name_fragments)
            if excluded or sensitive:
                module.qconfig = None
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            prepared = torch.ao.quantization.prepare_qat(target, inplace=True)
        message = "; ".join(dict.fromkeys(str(item.message) for item in captured))
        return QATResult(prepared, config.backend, True, message)
    except (RuntimeError, NotImplementedError) as error:
        message = f"QAT preparation failed for backend {config.backend}: {error}"
        if config.strict_backend:
            raise QuantizationBackendUnavailable(message) from error
        return QATResult(target, config.backend, False, message)
    finally:
        torch.backends.quantized.engine = previous_engine


def convert_qat(model: nn.Module, *, inplace: bool = False) -> nn.Module:
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return torch.ao.quantization.convert(model, inplace=inplace)
