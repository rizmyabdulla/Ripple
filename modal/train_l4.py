"""Modal entrypoints: CPU LibriTTS prep + L4 STS training.

Usage (repo root):

  modal setup
  modal run modal/train_l4.py::prep_libritts          # CPU: download → canonicalize → seal
  modal run modal/train_l4.py::train                   # L4: reconstruction train
  modal run modal/train_l4.py::smoke                   # L4: synthetic CUDA smoke

Volume: ripple-sts mounted at /data
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import modal

APP_NAME = "ripple-sts"
VOLUME_NAME = "ripple-sts"
MOUNT = "/data"
REPO = "/opt/ripple"

# OpenSLR LibriTTS (CC BY 4.0) — https://www.openslr.org/60/
LIBRITTS_BASE = "https://www.openslr.org/resources/60"
LIBRITTS_SPLITS: dict[str, str] = {
    "train-clean-100": f"{LIBRITTS_BASE}/train-clean-100.tar.gz",
    "dev-clean": f"{LIBRITTS_BASE}/dev-clean.tar.gz",
    "test-clean": f"{LIBRITTS_BASE}/test-clean.tar.gz",
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_REPO_ROOT = Path(__file__).resolve().parents[1]

_ripple_deps = [
    "einops>=0.8,<1",
    "numpy>=2.1,<3",
    "pydantic>=2.10,<3",
    "PyYAML>=6.0.2,<7",
    "safetensors>=0.5,<1",
    "soundfile>=0.13,<1",
    "soxr>=0.5,<1",
    "typer>=0.15,<1",
    "pyarrow>=17,<23",
]


def _base_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("libsndfile1", "ffmpeg")
        .pip_install(*_ripple_deps)
        .env({"PYTHONPATH": f"{REPO}/src", "RIPPLE_CONFIG_ROOT": f"{REPO}/configs"})
        .add_local_dir(
            _REPO_ROOT / "src" / "ripple",
            remote_path=f"{REPO}/src/ripple",
            copy=True,
        )
        .add_local_dir(
            _REPO_ROOT / "configs",
            remote_path=f"{REPO}/configs",
            copy=True,
        )
    )


# CPU prep/doctor: CPU torch only — `ripple.cli.main` imports torch transitively.
# Keep CUDA wheels off this image so prep does not need (or bill) an L4.
cpu_image = _base_image().pip_install(
    "torch==2.6.0",
    index_url="https://download.pytorch.org/whl/cpu",
)

# L4 train / smoke: CUDA torch.
gpu_image = (
    _base_image()
    .pip_install(
        "torch==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("tensorboard>=2.18,<3")
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _ensure_layout() -> dict[str, Path]:
    roots = {
        "raw": Path(MOUNT) / "raw",
        "canonical": Path(MOUNT) / "canonical",
        "manifests": Path(MOUNT) / "manifests",
        "checkpoints": Path(MOUNT) / "checkpoints",
        "features": Path(MOUNT) / "features",
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    return roots


def _download(url: str, destination: Path, *, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"skip download (exists): {destination}", flush=True)
        return destination

    partial = destination.with_suffix(destination.suffix + ".partial")
    print(f"download {url} -> {destination}", flush=True)

    def _report(block: int, block_size: int, total: int) -> None:
        if total <= 0 or block % 200 != 0:
            return
        done = min(block * block_size, total)
        pct = 100.0 * done / total
        print(f"  {pct:5.1f}% ({done // (1024 * 1024)} / {total // (1024 * 1024)} MiB)", flush=True)

    urllib.request.urlretrieve(url, partial, reporthook=_report)
    partial.replace(destination)
    return destination


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extract_ok"
    if marker.is_file():
        print(f"skip extract (marker): {destination}", flush=True)
        return
    print(f"extract {archive} -> {destination}", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        # Avoid path-traversal; Python 3.12+ supports filter=.
        try:
            tar.extractall(destination, filter=tarfile.data_filter)
        except TypeError:
            tar.extractall(destination)
    marker.write_text(archive.name + "\n", encoding="utf-8")


def _find_split_root(extract_root: Path, split: str) -> Path:
    """LibriTTS tarballs usually unpack to <root>/LibriTTS/<split>/..."""
    direct = extract_root / "LibriTTS" / split
    if direct.is_dir():
        return direct
    matches = list(extract_root.rglob(split))
    for path in matches:
        if path.is_dir() and any(path.rglob("*.wav")):
            return path
    raise FileNotFoundError(f"could not find split directory {split!r} under {extract_root}")


def _write_synthetic_wavs(root: Path, *, speakers: int = 4, utts: int = 2) -> None:
    import wave

    frames = 24_000
    for s in range(speakers):
        speaker = root / f"spk-{s:02d}"
        speaker.mkdir(parents=True, exist_ok=True)
        for u in range(utts):
            path = speaker / f"utt-{u:02d}.wav"
            if path.is_file():
                continue
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24_000)
                sample = bytes([(s * 17 + u * 3 + i) % 256 for i in range(frames * 2)])
                handle.writeframes(sample)


def _seal_tree(
    *,
    audio_root: Path,
    draft: Path,
    sealed: Path,
    dataset_id: str,
    revision: str,
    license_id: str,
    consent_basis: str,
    language: str,
) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "ripple.cli.main",
            "manifest",
            "build",
            str(audio_root),
            str(draft),
            "--license",
            license_id,
            "--consent-basis",
            consent_basis,
            "--language",
            language,
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "ripple.cli.main",
            "manifest",
            "seal",
            str(draft),
            str(sealed),
            "--dataset-id",
            dataset_id,
            "--revision",
            revision,
            "--audio-root",
            str(audio_root),
            "--overwrite",
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "ripple.cli.main",
            "manifest",
            "validate",
            str(sealed),
            "--audio-root",
            str(audio_root),
            "--verify-checksums",
        ]
    )


# ---------------------------------------------------------------------------
# CPU: LibriTTS download → canonicalize → seal
# ---------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    cpu=4,
    memory=8192,
    timeout=60 * 60 * 6,  # large tarball + canonicalize
    volumes={MOUNT: volume},
)
def prep_libritts(
    split: str = "train-clean-100",
    also_dev: bool = True,
    also_test: bool = False,
    force_download: bool = False,
    skip_canonicalize_if_exists: bool = True,
) -> dict[str, str]:
    """Download LibriTTS from OpenSLR on CPU; canonicalize; seal manifests.

    Does **not** use a GPU. Default split is Phase 0 ``train-clean-100`` (CC BY 4.0).
    """
    if split not in LIBRITTS_SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(LIBRITTS_SPLITS)}")

    roots = _ensure_layout()
    raw_root = roots["raw"] / "libritts"
    archives = raw_root / "archives"
    extract_root = raw_root / "extracted"
    archives.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    splits = [split]
    if also_dev and "dev-clean" not in splits:
        splits.append("dev-clean")
    if also_test and "test-clean" not in splits:
        splits.append("test-clean")

    sealed_paths: dict[str, str] = {}
    for name in splits:
        url = LIBRITTS_SPLITS[name]
        archive = archives / f"{name}.tar.gz"
        _download(url, archive, force=force_download)
        split_extract = extract_root / name
        _safe_extract(archive, split_extract)
        wav_root = _find_split_root(split_extract, name)

        canonical = roots["canonical"] / "libritts" / name
        if (
            skip_canonicalize_if_exists
            and canonical.is_dir()
            and any(canonical.rglob("*.wav"))
        ):
            print(f"skip canonicalize (exists): {canonical}", flush=True)
        else:
            if canonical.exists():
                shutil.rmtree(canonical)
            print(f"canonicalize {wav_root} -> {canonical}", flush=True)
            _run(
                [
                    sys.executable,
                    "-m",
                    "ripple.cli.main",
                    "data",
                    "canonicalize",
                    str(wav_root),
                    str(canonical),
                ]
            )
            volume.commit()

        role = "train" if name.startswith("train") else name.replace("-", "_")
        draft = roots["manifests"] / f"libritts_{name}.draft.jsonl"
        sealed = roots["manifests"] / f"libritts_{name}.json"
        if name.startswith("train"):
            sealed = roots["manifests"] / "train.json"
            role = "train"
        elif name.startswith("dev"):
            sealed = roots["manifests"] / "dev.json"
        elif name.startswith("test"):
            sealed = roots["manifests"] / "test.json"

        _seal_tree(
            audio_root=canonical,
            draft=draft,
            sealed=sealed,
            dataset_id=f"libritts-{name}",
            revision="1",
            license_id="CC-BY-4.0",
            consent_basis="public research corpus (LibriTTS / OpenSLR 60)",
            language="en",
        )
        sealed_paths[role] = str(sealed)
        volume.commit()

        # Provenance fingerprint of the sealed file for logs.
        digest = hashlib.sha256(sealed.read_bytes()).hexdigest()[:16]
        print(f"sealed {sealed} sha256_prefix={digest}", flush=True)

    summary = roots["manifests"] / "libritts_prep_summary.txt"
    summary.write_text(
        "\n".join(f"{k}={v}" for k, v in sorted(sealed_paths.items())) + "\n",
        encoding="utf-8",
    )
    volume.commit()
    print("prep complete:", sealed_paths, flush=True)
    return sealed_paths


# ---------------------------------------------------------------------------
# L4: train / smoke
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    gpu="L4",
    timeout=60 * 30,
    volumes={MOUNT: volume},
)
def smoke(steps: int = 20, batch_size: int = 2, crop_samples: int = 4800) -> str:
    """L4 CUDA smoke: synthetic WAVs → seal → reconstruction train."""
    import torch

    print("cuda_available=", torch.cuda.is_available(), flush=True)
    print(
        "device=",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        flush=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Modal container")

    roots = _ensure_layout()
    synth = roots["raw"] / "synthetic-en"
    _write_synthetic_wavs(synth)

    draft = roots["manifests"] / "synthetic.draft.jsonl"
    sealed = roots["manifests"] / "synthetic.train.json"
    out_dir = roots["checkpoints"] / "smoke_l4"

    _seal_tree(
        audio_root=synth,
        draft=draft,
        sealed=sealed,
        dataset_id="modal-smoke",
        revision="1",
        license_id="CC0-1.0",
        consent_basis="synthetic",
        language="en",
    )
    _run(
        [
            sys.executable,
            "-m",
            "ripple.cli.main",
            "train",
            "run",
            "--manifest",
            str(sealed),
            "--audio-root",
            str(synth),
            "--output-dir",
            str(out_dir),
            "--stage",
            "decoder_reconstruction",
            "--device",
            "cuda",
            "--precision",
            "bf16",
            "--batch-size",
            str(batch_size),
            "--crop-samples",
            str(crop_samples),
            "--steps",
            str(steps),
            "--save-every",
            str(max(1, steps // 2)),
        ]
    )
    volume.commit()
    final = out_dir / "checkpoint_last.pt"
    print("checkpoint=", final, flush=True)
    return str(final)


@app.function(
    image=gpu_image,
    gpu="L4",
    timeout=60 * 60 * 6,
    volumes={MOUNT: volume},
)
def train(
    manifest: str = f"{MOUNT}/manifests/train.json",
    audio_root: str = f"{MOUNT}/canonical/libritts/train-clean-100",
    output_dir: str = f"{MOUNT}/checkpoints/l4_recon",
    stage: str = "decoder_reconstruction",
    feature_dir: str | None = None,
    steps: int = 10_000,
    batch_size: int = 16,
    crop_samples: int = 24_000,
    learning_rate: float = 2e-4,
    save_every: int = 1_000,
    resume_checkpoint: str | None = None,
    precision: str = "bf16",
) -> str:
    """Train / resume on L4 using sealed manifests from ``prep_libritts``."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Modal container")
    print("device=", torch.cuda.get_device_name(0), flush=True)

    _ensure_layout()
    man = Path(manifest)
    audio = Path(audio_root)
    out = Path(output_dir)
    if not man.is_file():
        raise FileNotFoundError(
            f"Missing sealed manifest at {man}. Run: "
            f"modal run modal/train_l4.py::prep_libritts"
        )
    if not audio.is_dir():
        raise FileNotFoundError(f"Missing audio root: {audio}")

    cmd = [
        sys.executable,
        "-m",
        "ripple.cli.main",
        "train",
        "resume" if resume_checkpoint else "run",
        "--manifest",
        str(man),
        "--audio-root",
        str(audio),
        "--output-dir",
        str(out),
        "--stage",
        stage,
        "--device",
        "cuda",
        "--precision",
        precision,
        "--batch-size",
        str(batch_size),
        "--crop-samples",
        str(crop_samples),
        "--steps",
        str(steps),
        "--learning-rate",
        str(learning_rate),
        "--save-every",
        str(save_every),
    ]
    if resume_checkpoint:
        cmd.extend(["--checkpoint", resume_checkpoint])
    if feature_dir:
        cmd.extend(["--feature-dir", feature_dir])

    _run(cmd)
    volume.commit()
    final = out / "checkpoint_last.pt"
    print("checkpoint=", final, flush=True)
    return str(final)


@app.function(image=cpu_image, timeout=60 * 5, volumes={MOUNT: volume})
def doctor() -> None:
    """CPU doctor against the Volume."""
    _ensure_layout()
    _run(
        [
            sys.executable,
            "-m",
            "ripple.cli.main",
            "doctor",
            "run",
            "--data-root",
            MOUNT,
            "--checkpoint-root",
            f"{MOUNT}/checkpoints",
            "--config-root",
            f"{REPO}/configs",
        ]
    )
    volume.commit()


@app.local_entrypoint()
def main(mode: str = "smoke", steps: int = 20) -> None:
    if mode == "smoke":
        print("done:", smoke.remote(steps=steps))
    elif mode == "doctor":
        doctor.remote()
    elif mode == "prep":
        print("done:", prep_libritts.remote())
    else:
        raise SystemExit("Use ::smoke | ::prep_libritts | ::train | ::doctor")
