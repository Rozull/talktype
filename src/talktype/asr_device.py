"""CUDA DLL preparation and NVML device selection.

`prepare_cuda_dlls()` must run before anything imports `ctranslate2` (and therefore
`faster_whisper`). This module itself never imports either.
"""

from __future__ import annotations

import os
import site
import sys
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from talktype import win32
from talktype.asr import MIB, DeviceKind, model_spec
from talktype.logging_setup import get_logger, log_event

logger = get_logger("asr_device")

# Free VRAM kept in reserve on top of each model's requirement.
VRAM_HEADROOM = 512 * MIB

DeviceReason = Literal[
    "auto",  # automatic choice, enough free VRAM
    "forced",  # the configured device was used as asked
    "insufficient_vram",
    "no_gpu",  # NVML failed: no driver, no GPU
    "forced_unavailable",  # device = "cuda" but NVML failed
    "parakeet_cpu_only",
]

_added_dll_dirs: set[str] = set()
_dll_handles: list[Any] = []  # keep the add_dll_directory cookies alive


def _site_dirs() -> list[Path]:
    dirs = [Path(p) for p in site.getsitepackages()]
    user = site.getusersitepackages()
    if user:
        dirs.append(Path(user))
    dirs.extend(Path(p) for p in sys.path if p.endswith("site-packages"))
    return dirs


def prepare_cuda_dlls(
    site: Path | Iterable[Path] | None = None,
    *,
    add_dll_directory: Callable[[str], Any] | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> list[Path]:
    """Put every ``site-packages/nvidia/*/bin`` on the DLL search path and on ``PATH``.

    Idempotent: a directory is added once per process. Returns the directories found.
    """
    roots = _site_dirs() if site is None else [site] if isinstance(site, Path) else list(site)
    add = add_dll_directory or getattr(os, "add_dll_directory", None)
    env = os.environ if environ is None else environ
    found: list[Path] = []
    for root in roots:
        nvidia = root / "nvidia"
        if nvidia.is_dir():
            found.extend(sorted(p for p in nvidia.glob("*/bin") if p.is_dir()))
    for directory in found:
        text = str(directory)
        if text not in _added_dll_dirs:
            if add is not None:
                _dll_handles.append(add(text))
            _added_dll_dirs.add(text)
        path_entries = env.get("PATH", "").split(os.pathsep)
        if text not in path_entries:
            env["PATH"] = os.pathsep.join([text, *filter(None, path_entries)])
    return found


@dataclass(frozen=True, slots=True)
class Device:
    kind: DeviceKind
    compute_type: str
    reason: DeviceReason = "auto"
    free_mib: int | None = field(default=None, compare=False)


def cpu(reason: DeviceReason, free_mib: int | None = None) -> Device:
    return Device("cpu", "int8", reason, free_mib)


def required_bytes(model_id: str) -> int | None:
    """Free VRAM needed to put `model_id` on CUDA, or None for CPU-only models."""
    vram = model_spec(model_id).vram_bytes
    return None if vram is None else vram + VRAM_HEADROOM


def select_device(model_id: str, requested: str = "auto", nvml: Any = None) -> Device:
    """Choose CUDA or CPU for `model_id`.

    - ``cpu`` requested: CPU.
    - Parakeet: always CPU (``parakeet_cpu_only``).
    - NVML failure: CPU with ``no_gpu`` (auto) or ``forced_unavailable`` (cuda).
    - ``cuda`` requested with a working GPU: CUDA.
    - auto: CUDA when ``free >= requirement + 512 MiB``, otherwise ``insufficient_vram``.
    """
    required = required_bytes(model_id)
    if requested == "cpu":
        device = cpu("forced")
    elif required is None:
        device = cpu("parakeet_cpu_only")
    else:
        try:
            free = win32.nvml_memory(0, nvml=nvml).free
        except Exception:
            device = cpu("forced_unavailable" if requested == "cuda" else "no_gpu")
        else:
            free_mib = free // MIB
            if requested == "cuda":
                device = Device("cuda", "int8_float16", "forced", free_mib)
            elif free >= required:
                device = Device("cuda", "int8_float16", "auto", free_mib)
            else:
                device = cpu("insufficient_vram", free_mib)
    log_event(
        logger,
        "device_selected",
        model=model_id,
        device=device.kind,
        compute=device.compute_type,
        free_mib=device.free_mib if device.free_mib is not None else "",
        reason=device.reason,
    )
    return device
