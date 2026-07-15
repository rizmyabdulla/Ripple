# Ripple on Modal (L4)

Single-GPU STS training on an **NVIDIA L4**. Prep (download → canonicalize → seal) runs on **CPU** so GPU credits are not wasted. Ripple’s train CLI is single-process (no DDP); use extra Modal accounts only to **resume**, not to parallelize one job.

## Dataset

| Thing | What it is | Train for real? |
|---|---|---|
| `fixtures/multilingual-core/` | Synthetic manifests in git | **No** |
| `::smoke` | Tiny synthetic WAVs on the Volume | **No** — CUDA check |
| `::prep_libritts` | **LibriTTS** from OpenSLR 60 (CC BY 4.0) | **Yes** — Phase 0 |
| Phase 1 plan | MLS + Common Voice + FLEURS | Later — see dataset card |

Card: [`governance/DATASET_CARD_ripple-multilingual-v0.1.0.md`](../governance/DATASET_CARD_ripple-multilingual-v0.1.0.md).

Default prep split: **`train-clean-100`** (+ **`dev-clean`**). That is the English pilot; not the full multilingual mix.

Volume layout after prep:

```text
/data/
  raw/libritts/archives/          # .tar.gz from OpenSLR
  raw/libritts/extracted/         # unpacked vendor trees
  canonical/libritts/<split>/     # 24 kHz mono PCM16
  manifests/train.json            # sealed train
  manifests/dev.json              # sealed dev (if also_dev)
  checkpoints/                    # train outputs
```

## Setup

```powershell
pip install modal
modal setup
cd C:\Users\user\Documents\StreamVC\Ripple
```

CPU images install **CPU** PyTorch (required because `ripple.cli.main` imports torch). GPU images use the CUDA wheel for L4.

## Recommended flow

### 1) Optional CUDA smoke (L4, no LibriTTS)

```powershell
modal run modal/train_l4.py::smoke
```

### 2) CPU prep — download → canonicalize → seal

Uses **CPU only** (OpenSLR download can take a while; ~GB archive).

```powershell
modal run modal/train_l4.py::prep_libritts
```

Options:

```powershell
# train-clean-100 only (skip dev)
modal run modal/train_l4.py::prep_libritts --also-dev=false

# also seal test-clean
modal run modal/train_l4.py::prep_libritts --also-test=true

# re-download archives
modal run modal/train_l4.py::prep_libritts --force-download=true
```

Idempotent: existing archives / extract markers / canonical WAVs are skipped unless forced.

### 3) L4 train

```powershell
modal run modal/train_l4.py::train -- `
  --steps 10000 `
  --batch-size 16 `
  --crop-samples 24000 `
  --precision bf16
```

Defaults already point at:

- `--manifest /data/manifests/train.json`
- `--audio-root /data/canonical/libritts/train-clean-100`
- `--output-dir /data/checkpoints/l4_recon`

### 4) Resume (new account / credit refresh)

```powershell
modal run modal/train_l4.py::train -- `
  --resume-checkpoint /data/checkpoints/l4_recon/checkpoint_last.pt `
  --steps 10000 `
  --batch-size 16 `
  --crop-samples 24000
```

Share the same Volume name `ripple-sts` across accounts, or `modal volume get` / `put` the checkpoint.

## Cost notes

| Work | Resource | Tip |
|---|---|---|
| Prep | CPU | Do **not** put download on L4 |
| Train | L4 ~`$0.000222/s` | ~**37 h** per $30 |

`volume.commit()` runs after each stage so resumes see sealed data and checkpoints.

## Fetch checkpoints locally

```powershell
modal volume get ripple-sts /checkpoints/l4_recon ./checkpoints_from_modal
```
