"""Post-training quantization with explicit backend capability handling."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class QuantizationBackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class PTQConfig:
    mode: str = "dynamic"
    backend: str = "x86"
    dtype: torch.dtype = torch.qint8
    excluded_module_names: tuple[str, ...] = ()
    strict_backend: bool = False


@dataclass
class PTQResult:
    model: nn.Module
    mode: str
    backend: str
    quantized: bool
    warnings: tuple[str, ...] = ()


def available_backends() -> tuple[str, ...]:
    engines = torch.backends.quantized.supported_engines
    return tuple(engine for engine in engines if engine != "none")


def _run(model: nn.Module, batch: Any) -> Any:
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    return model(batch)


def quantize_ptq(
    model: nn.Module,
    *,
    config: PTQConfig | None = None,
    calibration_data: Iterable[Any] | None = None,
    calibration_forward: Callable[[nn.Module, Any], Any] = _run,
) -> PTQResult:
    """Quantize a copy of ``model``; unsupported backends degrade explicitly."""
    config = config or PTQConfig()
    if config.mode not in {"dynamic", "static"}:
        raise ValueError("PTQ mode must be dynamic or static")
    copied = copy.deepcopy(model).cpu().eval()
    if config.backend not in available_backends():
        available = available_backends()
        message = (
            f"quantization backend {config.backend!r} unavailable; available={available}"
        )
        if config.strict_backend:
            raise QuantizationBackendUnavailable(message)
        return PTQResult(copied, config.mode, config.backend, False, (message,))

    previous_engine = torch.backends.quantized.engine
    messages: list[str] = []
    try:
        torch.backends.quantized.engine = config.backend
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if config.mode == "dynamic":
                quantized = torch.ao.quantization.quantize_dynamic(
                    copied,
                    {nn.Linear},
                    dtype=config.dtype,
                    inplace=False,
                )
            else:
                if calibration_data is None:
                    raise ValueError("static PTQ requires calibration_data")
                copied.qconfig = torch.ao.quantization.get_default_qconfig(config.backend)
                for name, module in copied.named_modules():
                    excluded = any(
                        name == item or name.startswith(item + ".")
                        for item in config.excluded_module_names
                    )
                    if excluded:
                        module.qconfig = None
                prepared = torch.ao.quantization.prepare(copied, inplace=False)
                with torch.inference_mode():
                    for batch in calibration_data:
                        calibration_forward(prepared, batch)
                quantized = torch.ao.quantization.convert(prepared, inplace=False)
            messages.extend(str(item.message) for item in captured)
        return PTQResult(quantized, config.mode, config.backend, True, tuple(messages))
    except (RuntimeError, NotImplementedError) as error:
        message = f"{config.mode} PTQ failed for backend {config.backend}: {error}"
        if config.strict_backend:
            raise QuantizationBackendUnavailable(message) from error
        return PTQResult(copied, config.mode, config.backend, False, (message,))
    finally:
        torch.backends.quantized.engine = previous_engine
