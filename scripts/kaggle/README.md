# Running on Kaggle

Kaggle's free GPU (T4 / P100, 16 GB, ~30 hrs/week) is the **default** compute
for this project. Nebius H100 is used once at the end for the final
reproducible run. See the compute policy in `CLAUDE.md`.

Everything in this directory is glue. No Kaggle-specific code lives in
`src/train/` or `src/data/` — the library stays portable, and the same command
runs on CPU, a T4 and an H100 with only the config changing.

---

## Two constraints that shape everything

**No internet.** Kaggle notebooks run with networking off by default. Nothing
in the training path may download at runtime, so data arrives as a Kaggle
Dataset mounted read-only at `/kaggle/input/...`.

**Sessions get killed.** 9 hours max, and less if the tab closes or the
instance is reclaimed. So training checkpoints every epoch to
`/kaggle/working` and `--resume` continues from it. A run that cannot resume is
a run that is lost.

---

## 1. Build the dataset locally

Fetch and prepare the data on your own machine, where there *is* internet:

```bash
# DrivAerNet parametric table (1.4 MB) — the stage-1 gate data
python -m src.data.drivaernet --fetch

# DrivAerML surfaces (0.73 GB per run, surface only)
python -m src.data.drivaerml --runs 30 --download
python -m src.data.preprocess --combine
```

Then stage only what training reads. Keep it small — Kaggle Datasets cap at
~100 GB but upload time is the real limit, and nothing here needs the raw
`.vtp` files unless you are training on surface fields.

```bash
mkdir -p /tmp/aero-kaggle
cp data/drivaernet/parametric.csv /tmp/aero-kaggle/
cp -r data/drivaernet/raw         /tmp/aero-kaggle/raw
# point clouds / meshes, if training on geometry:
# cp -r data/drivaerml/combined   /tmp/aero-kaggle/combined
```

Upload `/tmp/aero-kaggle` as a Kaggle Dataset (web UI, or the CLI below), and
note the slug it gets — `<username>/aero-surrogate-data`.

```bash
pip install kaggle          # needs ~/.kaggle/kaggle.json from your account page
cd /tmp/aero-kaggle
kaggle datasets init -p .
# edit dataset-metadata.json: set "title" and "id" to <username>/aero-surrogate-data
kaggle datasets create -p . --dir-mode zip
# later updates:
kaggle datasets version -p . -m "add combined meshes" --dir-mode zip
```

---

## 2. The notebook

Create a notebook, **Settings → Accelerator → GPU T4 x2** (or P100), attach the
dataset, leave internet **off**. Then:

```python
# ---- 1. the code. No internet, so push the repo as a second Dataset, or
# ---- paste it in. A Dataset is better: it versions with the data.
import sys, subprocess, os
sys.path.insert(0, "/kaggle/input/aero-surrogate-code")

# ---- 2. check the data BEFORE training. Never skip this.
!python -m src.data.dataset --check --config configs/kaggle.yaml

# ---- 3. train. Writes last.pt every epoch to /kaggle/working/runs.
!python -m src.train.train --config configs/kaggle.yaml

# ---- 4. after a kill, re-run the notebook and add --resume.
# !python -m src.train.train --config configs/kaggle.yaml --resume
```

`configs/kaggle.yaml` already points `labels_csv`, `mesh_dir`, `cache_dir` and
`out_dir` at the right Kaggle paths. Change the dataset slug in it to yours.

### Getting the checkpoint out

`/kaggle/working` is downloadable from the notebook's Output tab, and persists
between runs of the *same* notebook version. `best.pt` is what Pranoy's service
loads; `summary.json` has the metrics. Download both.

---

## Why `cache_dir` is under `/kaggle/working`, not `/kaggle/input`

`/kaggle/input` is **read-only**. The sampled point-cloud cache has to be
writable, so it goes to `/kaggle/working` and does not survive the session.
Re-sampling 30 meshes costs seconds; crashing on the first epoch costs the run.

## Why `amp: true` is safe here

AMP in this trainer is fp16 autocast, guarded by `device.type == "cuda"`, so it
is a no-op on CPU and MPS. fp16 works on T4 (Turing) and P100 alike. There is
no bf16 path, which Turing does not support.

## Why `batch_size: 8` and `n_points: 2048`

16 GB, not 80 GB. The per-point encoder holds activations for
`batch × n_points × channels`, so both numbers bound memory directly. Raise
them only for the final Nebius run, by editing `configs/baseline.yaml` — never
by assuming the bigger card.
