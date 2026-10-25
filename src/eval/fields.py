"""Stage-2 grading: how well the surface-pressure model predicts Cp.  OWNER: Paramveer.

    # inside a full grade
    python -m src.eval.harness ... --field-checkpoint runs/fields/best.pt
    # or into an existing metrics.json, leaving every other number alone
    python -m src.eval.fields --checkpoint runs/fields/best.pt --metrics web/results/metrics.json

Graded on the checkpoint's own held-out cars (its splits["test"], whole runs,
never points from a car it trained on), from the extracted DrivAerML arrays in
data/drivaerml/fields/run_<n>.npz (src.data.fields --extract). Every number is
in Cp units, pooled over every stored cell of every test car:

  overall          Cp R2, MAE, RMSE, and 1- and 2-sigma coverage of the
                   model's own per-point sigma
  references       the same scores for two models with nothing learned about
                   the car: the training mean everywhere, and the Cp of the
                   nearest point (by position and normal) on any training car
  per_region       MAE and area-weighted mean Cp, true against predicted, per
                   DrivAerML region: what the designer's Regions paint shows
  integrated_cd    the pressure part of drag, integrated over the predicted
                   and the true field of each car (sum Cp * -n_x * area / A_ref,
                   scaled from the stored cells to the whole surface): whether
                   a field that scores well pointwise also adds up to the
                   right force

Track A's trainer reports similar numbers about itself; these are computed
here, independently, from the arrays and the checkpoint alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

KNN_POINTS_PER_CAR = 20_000     # training cells per car in the nearest-point reference
KNN_K = 8
DEFAULT_CHUNK_SIZE = 1024      # bound local-neighbour feature allocations on CPU
NORMAL_WEIGHT = 0.25            # how much a normal counts against a position in that search


def _r2(y, p) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / ss if ss > 0 else float("nan")


def _scores(true, pred, sigma=None) -> dict:
    err = np.abs(pred - true)
    out = {"r2": _r2(true, pred), "mae": float(err.mean()), "rmse": float(np.sqrt((err ** 2).mean()))}
    if sigma is not None:
        out["coverage_1sigma"] = float((err <= sigma).mean())
        out["coverage_2sigma"] = float((err <= 2 * sigma).mean())
        out["mean_sigma"] = float(sigma.mean())
    return out


def _pressure_drag(cp, arrays, meta) -> float:
    """Pressure part of Cd over the stored cells, scaled to the whole surface."""
    area = arrays["query_area"].astype(np.float64)
    nx = arrays["query_normal"][:, 0].astype(np.float64)
    scale = float(meta.get("total_area_m2") or area.sum()) / max(float(area.sum()), 1e-12)
    return float((cp * -nx * area).sum() / float(meta["a_ref_m2"]) * scale)


def _fields_directory(root) -> Path:
    root = Path(root)
    return root if root.name == "fields" or any(root.glob("run_*.npz")) else root / "fields"


def load_runs(root: Path, runs) -> list[tuple[int, dict, dict]]:
    from src.data.fields import load_run

    out = []
    for run in runs:
        path = _fields_directory(root) / f"run_{run}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} (run `python -m src.data.fields --extract`)")
        arrays, meta = load_run(path)
        out.append((int(run), arrays, meta))
    return out


def checkpoint_predictor(path, device: str = "cpu", chunk: int = DEFAULT_CHUNK_SIZE):
    """(predict, ckpt) for a fields-v1 checkpoint; predict(arrays) -> (cp, sigma) in Cp units."""
    import torch

    from src.train.field_predict import FieldPredictor

    if chunk < 1:
        raise ValueError("chunk size must be positive")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    fp = FieldPredictor(ckpt, device=torch.device(device))
    n_cloud = int(fp.preprocess.get("n_cloud", 2048))

    @torch.no_grad()
    def predict(arrays):
        cloud = torch.tensor(arrays["cloud_xyz"][:n_cloud], device=fp.device).unsqueeze(0)
        cps, sigmas = [], []
        for start in range(0, len(arrays["query_xyz"]), chunk):
            q = torch.tensor(arrays["query_xyz"][start:start + chunk], device=fp.device).unsqueeze(0)
            n = torch.tensor(arrays["query_normal"][start:start + chunk], device=fp.device).unsqueeze(0)
            mean, sigma = fp.model.split(fp.model(cloud, q, n))
            cps.append(mean.reshape(-1).cpu().numpy() * fp.cp_std + fp.cp_mean)
            sigmas.append(sigma.reshape(-1).cpu().numpy() * fp.cp_std)
        return np.concatenate(cps).astype(np.float64), np.concatenate(sigmas).astype(np.float64)

    return predict, ckpt


def _nearest_car_reference(train, k: int = KNN_K, per_car: int = KNN_POINTS_PER_CAR, seed: int = 0, workers: int = 1):
    """Average Cp of pooled nearest training points in position/normal space."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    feats, cps = [], []
    for _, arrays, _ in train:
        n = len(arrays["cp"])
        take = rng.choice(n, size=min(per_car, n), replace=False)
        feats.append(np.hstack([arrays["query_xyz"][take], NORMAL_WEIGHT * arrays["query_normal"][take]]))
        cps.append(arrays["cp"][take])
    tree = cKDTree(np.vstack(feats).astype(np.float64))
    cp = np.concatenate(cps).astype(np.float64)

    def predict(arrays):
        q = np.hstack([arrays["query_xyz"], NORMAL_WEIGHT * arrays["query_normal"]]).astype(np.float64)
        _, idx = tree.query(q, k=k, workers=workers)
        return cp[idx].mean(axis=1)

    return predict


def grade(test, train, predict, *, reference_workers: int = 1) -> dict:
    """The field block of metrics.json from loaded runs and a predictor."""
    regions = list(test[0][2].get("regions") or [])
    mean_cp = float(np.concatenate([a["cp"] for _, a, _ in train]).astype(np.float64).mean()) if train else 0.0
    nearest = _nearest_car_reference(train, workers=reference_workers) if train else None

    pooled = {"true": [], "pred": [], "sigma": [], "mean": [], "nearest": [], "region": [], "area": []}
    per_car, cd_rows = [], []
    for run, arrays, meta in test:
        true = arrays["cp"].astype(np.float64)
        pred, sigma = predict(arrays)
        pooled["true"].append(true)
        pooled["pred"].append(pred)
        pooled["sigma"].append(sigma)
        pooled["mean"].append(np.full_like(true, mean_cp))
        if nearest is not None:
            pooled["nearest"].append(nearest(arrays))
        pooled["region"].append(arrays["region"].astype(np.int64))
        pooled["area"].append(arrays["query_area"].astype(np.float64))
        cd_true, cd_pred = _pressure_drag(true, arrays, meta), _pressure_drag(pred, arrays, meta)
        cd_rows.append((cd_true, cd_pred))
        per_car.append({"run": run, **_scores(true, pred), "cd_pressure_true": cd_true, "cd_pressure_pred": cd_pred})

    cat = {k: np.concatenate(v) for k, v in pooled.items() if v}
    out = {
        "available": True,
        "test_runs": [r for r, _, _ in test],
        "n_train_runs": len(train),
        "n_points": int(len(cat["true"])),
        "overall": _scores(cat["true"], cat["pred"], cat["sigma"]),
        "references": {"mean": _scores(cat["true"], cat["mean"])},
        "per_car": per_car,
    }
    if "nearest" in cat:
        out["references"]["nearest_training_car"] = _scores(cat["true"], cat["nearest"])

    per_region = {}
    for i, name in enumerate(regions):
        sel = cat["region"] == i
        if not sel.any():
            continue
        w = cat["area"][sel]
        per_region[name] = {
            "mae": float(np.abs(cat["pred"][sel] - cat["true"][sel]).mean()),
            "cp_true": float(np.average(cat["true"][sel], weights=w)),
            "cp_pred": float(np.average(cat["pred"][sel], weights=w)),
            "n": int(sel.sum()),
        }
    out["per_region"] = per_region

    cd_true, cd_pred = (np.array(v) for v in zip(*cd_rows))
    out["integrated_cd"] = {
        "what": "pressure part of drag, integrated over each held-out car's field and scaled to its whole surface",
        "true_mean": float(cd_true.mean()), "pred_mean": float(cd_pred.mean()),
        "mae": float(np.abs(cd_pred - cd_true).mean()),
        "mean_relative_error": float(np.mean(np.abs(cd_pred - cd_true) / np.maximum(np.abs(cd_true), 1e-9))),
    }
    return out


def grade_checkpoint(path, root="data/drivaerml", device: str = "cpu", *,
                     chunk_size: int = DEFAULT_CHUNK_SIZE, reference_workers: int = 1) -> dict:
    """Grade a field checkpoint on its own held-out cars, or say why it cannot be."""
    try:
        predict, ckpt = checkpoint_predictor(path, device, chunk=chunk_size)
    except Exception as exc:  # torch or the checkpoint missing: report, don't crash the grade
        return {"available": False, "reason": f"field checkpoint {Path(path).name} could not be loaded: {exc}"}
    splits = ckpt.get("splits") or {}
    if not splits.get("test"):
        return {"available": False, "reason": "the field checkpoint records no test runs"}
    try:
        test = load_runs(root, splits["test"])
        train = load_runs(root, splits.get("train") or [])
    except FileNotFoundError as exc:
        return {"available": False, "reason": f"field data missing: {exc}"}
    out = grade(test, train, predict, reference_workers=reference_workers)
    out["checkpoint"] = {
        "path": str(path).replace("\\", "/"), "sha256": _sha256(path), "version": ckpt.get("version"),
        "schema_version": ckpt.get("schema_version"), "n_params": ckpt.get("n_params"),
        "self_reported_test_metrics": {k: v for k, v in (ckpt.get("test_metrics") or {}).items() if k != "per_region"},
    }
    out["evaluation"] = {
        "device": device, "chunk_size": chunk_size, "reference_workers": reference_workers,
        "splits": splits,
        "arrays": [{"run": run, "sha256": _sha256(_fields_directory(root) / f"run_{run}.npz"),
                    "n_points": len(arrays["cp"])} for run, arrays, _ in train + test],
        "reference": {"name": "pooled nearest-training-point", "legacy_key": "nearest_training_car",
                      "k": KNN_K, "points_per_training_car": KNN_POINTS_PER_CAR,
                      "normal_weight": NORMAL_WEIGHT, "seed": 0},
    }
    return out


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grade a surface-pressure checkpoint into metrics.json.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", default="data/drivaerml")
    ap.add_argument("--metrics", default="web/results/metrics.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk-size", type=_positive_int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--reference-workers", type=_positive_int, default=1)
    args = ap.parse_args(argv)

    from .harness import portable_paths, write

    path = Path(args.metrics)
    try:
        metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(metrics, dict):
            raise ValueError("expected a JSON object")
    except (OSError, ValueError) as exc:
        print(f"cannot merge field metrics into {path.name}: {exc}")
        return 2
    block = grade_checkpoint(args.checkpoint, args.root, args.device,
                             chunk_size=args.chunk_size, reference_workers=args.reference_workers)
    metrics["field"] = block
    write(portable_paths(metrics), path)
    if not block["available"]:
        print(f"field not graded: {block['reason']}")
        return 1
    o, ref = block["overall"], block["references"]
    print(f"Cp R2 {o['r2']:.3f}  MAE {o['mae']:.4f}  1-sigma coverage {o['coverage_1sigma']:.2f}  "
          f"({block['n_points']:,} points, {len(block['test_runs'])} held-out cars)")
    for name, s in ref.items():
        label = "pooled nearest-training-point" if name == "nearest_training_car" else name
        print(f"  reference {label:28} R2 {s['r2']:.3f}  MAE {s['mae']:.4f}")
    c = block["integrated_cd"]
    print(f"pressure drag: true {c['true_mean']:.4f}  predicted {c['pred_mean']:.4f}  per-car MAE {c['mae']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
