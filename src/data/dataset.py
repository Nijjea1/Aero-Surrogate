"""DrivAerNet loading for stage 1.  OWNER: Avneet.

Build the --check path FIRST and run it before any training:

    python -m src.data.dataset --check --config configs/baseline.yaml

It should print the columns actually found in the CSV, how many mesh/label
pairs matched, and a sample point-cloud shape. Published DrivAerNet headers may
differ from the config defaults -- fix that here, not in the trainer.

Cache sampled point clouds as .npy. You will iterate on the model a hundred
times and must not re-sample meshes each run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_config(path: str | Path) -> dict:
    raise NotImplementedError("TODO Avneet")


def sample_mesh(mesh_path: Path, n_points: int, normalize: bool) -> np.ndarray:
    raise NotImplementedError("TODO Avneet")


class DrivAerNetDataset:
    """design_id -> (point cloud, Cd). Caches sampled clouds to disk."""

    def __init__(self, cfg: dict, split: str = "train", ids: list[str] | None = None):
        raise NotImplementedError("TODO Avneet: Oct 14")


def make_splits(cfg: dict, seed: int = 0) -> dict[str, list[str]]:
    """Random split by design id. Paramveer's harness also needs a split BY
    CATEGORY for the cross-category experiment."""
    raise NotImplementedError("TODO Avneet")


def check(config_path: str) -> int:
    """Sanity check before training. Returns a shell exit code."""
    raise NotImplementedError("TODO Avneet: Oct 13")
