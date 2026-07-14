from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_mock_artifact_lifecycle(tmp_path: Path) -> None:
    cmake = shutil.which("cmake")
    ctest = shutil.which("ctest")
    if cmake is None or ctest is None:
        pytest.skip("CMake/CTest are not installed")

    runtime = Path(__file__).resolve().parents[2] / "runtime"
    build = tmp_path / "native-build"
    configure = subprocess.run(
        [cmake, "-S", str(runtime), "-B", str(build), "-DRIPPLE_BUILD_TESTS=ON"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if configure.returncode != 0 and (
        "compiler" in configure.stdout.lower() + configure.stderr.lower()
        or "generator" in configure.stdout.lower() + configure.stderr.lower()
    ):
        pytest.skip("No usable C++20 compiler/generator is installed")
    assert configure.returncode == 0, configure.stdout + configure.stderr

    compiled = subprocess.run(
        [cmake, "--build", str(build), "--config", "Release"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    tested = subprocess.run(
        [ctest, "--test-dir", str(build), "-C", "Release", "--output-on-failure"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert tested.returncode == 0, tested.stdout + tested.stderr
