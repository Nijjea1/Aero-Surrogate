"""What the harness grades.  OWNER: Paramveer.

Everything here answers predict(x) -> (cd, sigma) for a batch of designs in a
checkpoint's column order, in Cd units.

* CheckpointModel loads a checkpoint Avneet trained, the way docs/HANDOFF.md
  section 3 says to (build_model, split_prediction, norm_stats, the ood dict).
  torch and src.train are imported only here, only when a checkpoint is given.
* MeanModel and LinearModel are references fitted by the harness itself. They
  put the network's numbers in context: R2 0.87 means more next to a straight
  line's score than on its own.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from .metrics import Mahalanobis


class MeanModel:
    """Predicts the training mean for every design; sigma is the training spread."""

    name = "mean"
    description = "predicts the training-set mean Cd for every design"

    def __init__(self, x_train, y_train):
        self.mu = float(np.mean(y_train))
        self.sigma = float(np.std(y_train))

    def predict(self, x):
        n = len(x)
        return np.full(n, self.mu), np.full(n, self.sigma)


class LinearModel:
    """Ridge regression on standardised inputs; sigma is the training residual
    spread, the same for every design. A model with no idea where it is weak."""

    name = "linear"
    description = "ridge regression on the same inputs, constant sigma"

    def __init__(self, x_train, y_train, ridge: float = 1e-3):
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        self.mean = x_train.mean(axis=0)
        self.std = x_train.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        z = self._design(x_train)
        reg = ridge * len(z) * np.eye(z.shape[1])
        reg[0, 0] = 0.0  # never shrink the intercept
        self.coef = np.linalg.solve(z.T @ z + reg, z.T @ y_train)
        self.sigma = float(np.std(y_train - z @ self.coef))

    def _design(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        return np.hstack([np.ones((len(z), 1)), z])

    def predict(self, x):
        mu = self._design(x) @ self.coef
        return mu, np.full(len(mu), self.sigma)


class CheckpointModel:
    """A trained checkpoint, loaded exactly as the service will load it."""

    def __init__(self, path, device: str = "cpu"):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise SystemExit(
                "grading a checkpoint needs torch. The CPU build is enough:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
            ) from exc
        from src.train.model import build_model, split_prediction

        self.path = Path(path)
        self._torch = torch
        self._split = split_prediction
        self.device = torch.device(device)

        ckpt = torch.load(self.path, map_location=self.device, weights_only=False)
        self.spec = ckpt.get("preprocess") or {}
        if self.spec.get("version") != "params-v1":
            raise SystemExit(
                f"{self.path}: preprocess version {self.spec.get('version')!r}. The harness "
                "grades parameter checkpoints (params-v1); point-cloud checkpoints need "
                "the meshes and are graded once stage 2 lands."
            )
        self.model = build_model(ckpt["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        self.columns = list(self.spec["param_columns"])
        norm = ckpt["norm_stats"]
        self.param_mean = np.asarray(norm["param_mean"], dtype=np.float64)
        self.param_std = np.asarray(norm["param_std"], dtype=np.float64)
        self.target_mean = float(norm["target_mean"])
        self.target_std = float(norm["target_std"])

        self.ood = Mahalanobis.from_checkpoint(ckpt["ood"]) if ckpt.get("ood") else None
        self.n_train = (ckpt.get("ood") or {}).get("n_train") or ckpt.get("n_train")
        self.held_out_category = ckpt.get("held_out_category")
        self.config = ckpt["config"]
        self.info = {
            "path": str(self.path).replace("\\", "/"),
            "sha256": sha256(self.path),
            "version": ckpt.get("version"),
            "schema_version": ckpt.get("schema_version"),
            "preprocess": self.spec.get("version"),
            "param_columns": self.columns,
            "held_out_category": self.held_out_category,
            "n_params": ckpt.get("n_params"),
            "n_train": self.n_train,
            "ood_threshold": self.ood.threshold if self.ood else None,
            # What training reported about itself, kept beside our own numbers
            # so any disagreement is visible rather than silently overwritten.
            "self_reported_test_metrics": ckpt.get("test_metrics"),
        }

    name = "checkpoint"
    description = "the trained surrogate"

    def standardise(self, x):
        return (np.asarray(x, dtype=np.float64) - self.param_mean) / self.param_std

    def predict(self, x, batch_size: int = 4096):
        torch = self._torch
        z = self.standardise(x).astype(np.float32)
        mus, sigmas = [], []
        with torch.no_grad():
            for i in range(0, len(z), batch_size):
                mu, sigma = self._split(self.model(torch.from_numpy(z[i:i + batch_size]).to(self.device)))
                mus.append(mu.cpu().numpy())
                sigmas.append(sigma.cpu().numpy())
        mu = np.concatenate(mus).astype(np.float64) * self.target_std + self.target_mean
        sigma = np.concatenate(sigmas).astype(np.float64) * self.target_std
        return mu, sigma

    def ood_score(self, x):
        return self.ood.score(self.standardise(x)) if self.ood else None

    def latency(self, x, repeats: int = 200) -> dict:
        """Wall-clock for one design at a time, and for the whole batch."""
        one = x[:1]
        self.predict(one)  # warm up
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.predict(one)
            times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        self.predict(x)
        batch = time.perf_counter() - t0
        return {
            "device": str(self.device),
            "single_design_ms_median": float(np.median(times) * 1e3),
            "single_design_ms_p95": float(np.quantile(times, 0.95) * 1e3),
            "batch_size": int(len(x)),
            "batch_designs_per_s": float(len(x) / batch) if batch > 0 else float("inf"),
        }


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
