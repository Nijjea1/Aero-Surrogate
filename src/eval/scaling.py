"""The data-scaling curve: how much better the model gets with more data.  OWNER: Paramveer.

    # train the stage-1 model on 5 %, 10 %, 25 %, 50 % and 100 % of the training split
    python -m src.eval.scaling train --config configs/drivaernet.yaml

    # grade them on the fixed test set and put the curve into a metrics.json
    python -m src.data.dataset --config configs/drivaernet.yaml --dump-splits runs/splits.json
    python -m src.eval.scaling update --metrics web/results/metrics.json --splits runs/splits.json \\
        runs/scaling/*/best.pt

Training is Track A's own trainer (src.train.train.train) with one change made
from outside: the random split's training ids are cut to a fraction before the
run. Validation and test are untouched, so every point is graded on the same
625 designs, and the subsets are nested (the 10 % set contains the 5 % set), so
a step on the curve is more data and nothing else. Each run writes an ordinary
checkpoint whose OOD block records its training-set size, which is what the
harness reads.

`update` scores those checkpoints with the harness's own code
(harness.score_scaling) and replaces data_scaling.model in an existing
metrics.json, leaving every other number as it was graded.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)
SUBSET_SEED = 1000            # the nested subsets; the split itself keeps the config's seed


def nested_subset(ids: list[str], fraction: float, seed: int = SUBSET_SEED) -> list[str]:
    """The first round(fraction * n) ids of one fixed shuffle, so smaller subsets nest in larger ones."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    order = np.random.default_rng(seed).permutation(len(ids))
    k = max(1, int(round(fraction * len(ids))))
    return [ids[i] for i in order[:k]]


def train_fractions(config: str, fractions=FRACTIONS, out_dir: str = "runs/scaling",
                    device: str | None = None, epochs: int | None = None) -> list[str]:
    """One training run per fraction; returns the checkpoint paths."""
    from src.data.dataset import load_config
    from src.train import train as trainer

    base = load_config(config)
    original = trainer.make_splits
    paths = []
    for f in sorted(fractions):
        cfg = copy.deepcopy(base)
        cfg["run_name"] = f"scaling-{round(f * 100):03d}"
        cfg["train"]["out_dir"] = out_dir
        if epochs:
            cfg["train"]["epochs"] = epochs

        def cut(c, seed=0, _f=f):
            splits = original(c, seed=seed)
            splits["train"] = nested_subset(splits["train"], _f)
            return splits

        trainer.make_splits = cut
        try:
            result = trainer.train(cfg, split_mode="random", device_str=device)
        finally:
            trainer.make_splits = original
        paths.append(result["checkpoint"])
    return paths


def update_metrics(metrics_path: str, splits_path: str, checkpoints: list[str], data_path=None,
                   device: str = "cpu") -> dict:
    """Score the checkpoints on the metrics' own test set and write data_scaling.model back."""
    from . import data as D
    from .harness import portable_paths, score_scaling, write

    path = Path(metrics_path)
    metrics = json.loads(path.read_text(encoding="utf-8"))
    table = D.load_table(D.find_data(data_path))
    splits = D.load_splits(splits_path)
    rs = splits["random"]
    expected = (metrics.get("splits") or {}).get("random", {}).get("test")
    if expected is not None and expected != len(rs["test"]):
        raise ValueError(f"{splits_path} has {len(rs['test'])} test designs; {metrics_path} was graded on {expected}")
    warnings: list[str] = []
    points = score_scaling(table, rs, checkpoints, device, warnings)
    block = metrics.setdefault("data_scaling", {})
    block["model"] = points
    block["command"] = " ".join(["python -m src.eval.scaling", *sys.argv[1:]]) if sys.argv[1:] else None
    block["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    block["subsets"] = (f"nested random subsets of the training split (seed {SUBSET_SEED}), validation and test fixed; "
                        "each point is its own training run, so the full-data point can differ a little from the headline model")
    stale = "no --scaling checkpoints: the data-scaling curve has the linear reference only"
    metrics["warnings"] = [w for w in metrics.get("warnings", []) if w != stale] + warnings
    write(portable_paths(metrics), path)
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train and grade the data-scaling curve.")
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train the stage-1 model on growing fractions of the training split")
    t.add_argument("--config", default="configs/drivaernet.yaml")
    t.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    t.add_argument("--out-dir", default="runs/scaling")
    t.add_argument("--device", default=None)
    t.add_argument("--epochs", type=int, default=None, help="override the config (smoke runs only)")
    u = sub.add_parser("update", help="grade scaling checkpoints into an existing metrics.json")
    u.add_argument("checkpoints", nargs="+")
    u.add_argument("--metrics", default="web/results/metrics.json")
    u.add_argument("--splits", required=True, help="runs/splits.json from src.data.dataset --dump-splits")
    u.add_argument("--data", default=None)
    u.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    if args.command == "train":
        for p in train_fractions(args.config, args.fractions, args.out_dir, args.device, args.epochs):
            print(p)
        return 0
    metrics = update_metrics(args.metrics, args.splits, args.checkpoints, args.data, args.device)
    for point in metrics["data_scaling"]["model"]:
        print(f"n_train {point['n_train']:>5}   R2 {point['r2']:.4f}   MAE {point['mae']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
