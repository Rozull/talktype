"""Device selection and CUDA DLL setup (UT-094–UT-098, UT-100)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from talktype.asr_device import Device, prepare_cuda_dlls, required_bytes, select_device
from tests.fakes import GIB, FakeNvml

pytestmark = pytest.mark.unit


def test_ut094_auto_with_room_picks_cuda() -> None:
    nvml = FakeNvml(free=8 * GIB)

    device = select_device("large-v3-turbo", "auto", nvml)

    assert device == Device("cuda", "int8_float16", reason="auto")
    assert device.free_mib == 8 * 1024
    assert nvml.calls == ["init", "handle:0", "meminfo", "shutdown"]


def test_ut095_boundary_is_requirement_plus_512_mib() -> None:
    required = int(2.1 * GIB)
    assert required_bytes("large-v3-turbo") == required

    assert select_device("large-v3-turbo", "auto", FakeNvml(free=required)).kind == "cuda"
    assert select_device("large-v3-turbo", "auto", FakeNvml(free=required - 1)) == Device(
        "cpu", "int8", reason="insufficient_vram"
    )


@pytest.mark.parametrize(
    ("model", "gib"),
    [("large-v3", 3.2), ("medium", 1.8), ("small", 0.8)],
)
def test_ut095_per_model_requirements(model: str, gib: float) -> None:
    required = int(gib * GIB) + 512 * 1024**2

    assert select_device(model, "auto", FakeNvml(free=required)).kind == "cuda"
    assert select_device(model, "auto", FakeNvml(free=required - 1)).kind == "cpu"


def test_ut096_nvml_failure_means_cpu_no_gpu() -> None:
    device = select_device("large-v3-turbo", "auto", FakeNvml(fail_init=True))

    assert device == Device("cpu", "int8", reason="no_gpu")


def test_ut097_forced_cuda_unavailable_falls_back_to_cpu() -> None:
    device = select_device("large-v3-turbo", "cuda", FakeNvml(fail_init=True))

    assert device == Device("cpu", "int8", reason="forced_unavailable")


def test_ut097_forced_devices_when_available() -> None:
    assert select_device("small", "cuda", FakeNvml(free=GIB)) == Device(
        "cuda", "int8_float16", reason="forced"
    )
    nvml = FakeNvml()
    assert select_device("large-v3-turbo", "cpu", nvml) == Device("cpu", "int8", reason="forced")
    assert nvml.calls == []


def test_ut098_parakeet_is_cpu_only() -> None:
    nvml = FakeNvml(free=8 * GIB)

    device = select_device("parakeet-v3", "cuda", nvml)

    assert device == Device("cpu", "int8", reason="parakeet_cpu_only")
    assert nvml.calls == []


def test_ut100_prepare_cuda_dlls_is_idempotent(tmp_path: Path) -> None:
    cublas = tmp_path / "nvidia" / "cublas" / "bin"
    cudnn = tmp_path / "nvidia" / "cudnn" / "bin"
    cublas.mkdir(parents=True)
    cudnn.mkdir(parents=True)
    (tmp_path / "nvidia" / "empty").mkdir()
    added: list[str] = []
    environ = {"PATH": r"C:\Windows"}

    found = prepare_cuda_dlls(tmp_path, add_dll_directory=added.append, environ=environ)

    assert sorted(found) == sorted([cublas, cudnn])
    assert sorted(added) == sorted([str(cublas), str(cudnn)])
    entries = environ["PATH"].split(os.pathsep)
    assert entries[-1] == r"C:\Windows"
    assert set(entries[:2]) == {str(cublas), str(cudnn)}

    path_before = environ["PATH"]
    prepare_cuda_dlls(tmp_path, add_dll_directory=added.append, environ=environ)

    assert len(added) == 2
    assert environ["PATH"] == path_before
