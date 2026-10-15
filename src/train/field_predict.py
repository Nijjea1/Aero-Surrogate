"""Serve a stage-2 field checkpoint.  OWNER: Avneet.

    python -m src.train.field_predict --validate-regions
    python -m src.train.field_predict --mesh data/drivaerml/index/meshes/run_1.stl

Turns a trained fields-v1 checkpoint into the thing the contract actually
wants: geometry in, Cp at every face, plus Evaluation.pressure_summary with
the eight PRESSURE_REGIONS keys, plus integrated drag and lift.

THE PROBLEM THIS HAD TO SOLVE

Region labels came from the 49 semantically named solids in a DrivAerML STL
(BodyHood, BodyRoof, BodyFasciafront1...). Combining the solids throws the
names away, and a mesh the AGENT proposes -- from Paramveer's params_to_mesh --
never had them. So pressure_summary cannot come from solid names at serve time.

assign_regions_geometric below does it from position and normal instead, in the
unit-sphere frame the model already works in. It is a heuristic, so it is
VALIDATED against the real labels on DrivAerML runs where ground truth exists
(--validate-regions), and the agreement is reported rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.contracts import PRESSURE_REGIONS
from src.train.field_model import integrate_drag
from src.train.train import pick_device


@dataclass
class FieldPrediction:
    """Everything the service needs from one geometry."""

    cp: np.ndarray                      # (F,) per-face pressure coefficient
    sigma: np.ndarray                   # (F,) per-face standard deviation
    region: np.ndarray                  # (F,) index into PRESSURE_REGIONS
    pressure_summary: dict[str, float]  # the eight contract keys
    cd: float
    cl: float
    frontal_area_m2: float
    mean_sigma: float
    faces_used: int = 0
    area_scale: float = 1.0

    def to_dict(self) -> dict:
        return {
            "cd": self.cd, "cl": self.cl,
            "pressure_summary": self.pressure_summary,
            "uncertainty": self.mean_sigma,
            "frontal_area_m2": self.frontal_area_m2,
        }


# --------------------------------------------------------------------------
# Region assignment, learned rather than hand-tuned
# --------------------------------------------------------------------------
#
# The first attempt was hand-written thresholds on position and normal. It
# scored 13.7% agreement against the real solid-derived labels, with 0% recall
# on windscreen, rear_window and wheel_wells -- and wheel_wells is 73% of all
# cells, because tyres, rims and brake discs are the most finely tessellated
# parts of the mesh. Hand-tuning geometry rules was simply the wrong tool when
# 3,000,000 labelled points are already on disk.
#
# So the mapping is LEARNED: a k-nearest-neighbour vote in the 6-D space of
# (unit-sphere position, face normal), fitted on DrivAerML cells whose region
# came from the named solids. kNN rather than a network because the decision
# boundary is a partition of a surface, not a smooth function -- neighbouring
# cells on the same panel belong to the same region almost by definition, which
# is exactly what kNN exploits.
#
# It is validated on HELD-OUT runs, never on the runs it was fitted to.


@dataclass
class RegionClassifier:
    """kNN region vote over (position, normal). Learned from real labels."""

    points: list[list[float]]      # (n, 6) position + normal
    labels: list[int]              # (n,) PRESSURE_REGIONS index
    k: int = 5
    normal_weight: float = 0.35    # normals matter, but less than position
    fitted_on_runs: list[int] = None  # type: ignore[assignment]

    @staticmethod
    def _features(points: np.ndarray, normals: np.ndarray,
                  normal_weight: float) -> np.ndarray:
        return np.concatenate(
            [np.asarray(points, np.float64),
             np.asarray(normals, np.float64) * normal_weight], axis=1)

    @classmethod
    def fit(cls, root: Path, runs: list[int], max_points: int = 60_000,
            k: int = 5, normal_weight: float = 0.35) -> "RegionClassifier":
        from src.data.fields import load_run

        feats, labels = [], []
        per_run = max(1, max_points // max(len(runs), 1))
        rng = np.random.default_rng(0)
        for run in runs:
            path = root / "fields" / f"run_{run}.npz"
            if not path.exists():
                continue
            arrays, meta = load_run(path)
            names = meta["regions"]
            mapped = np.array([PRESSURE_REGIONS.index(names[i])
                               if names[i] in PRESSURE_REGIONS else -1
                               for i in arrays["region"]])
            usable = np.flatnonzero(mapped >= 0)
            if len(usable) == 0:
                continue
            take = rng.choice(usable, size=min(per_run, len(usable)),
                              replace=False)
            feats.append(cls._features(arrays["query_xyz"][take],
                                       arrays["query_normal"][take],
                                       normal_weight))
            labels.append(mapped[take])

        if not feats:
            raise RuntimeError(f"no labelled cells found for runs {runs}")
        x = np.concatenate(feats)
        y = np.concatenate(labels)
        return cls(points=x.tolist(), labels=y.astype(int).tolist(), k=int(k),
                   normal_weight=float(normal_weight), fitted_on_runs=list(runs))

    def predict(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        from scipy.spatial import cKDTree

        ref = np.asarray(self.points, np.float64)
        lab = np.asarray(self.labels)
        query = self._features(points, normals, self.normal_weight)
        k = min(self.k, len(ref))
        _, idx = cKDTree(ref).query(query, k=k, workers=-1)
        if k == 1:
            return lab[np.ravel(idx)].astype(np.int16)
        votes = lab[idx]                                   # (n, k)
        out = np.empty(len(query), dtype=np.int16)
        for i in range(len(votes)):
            counts = np.bincount(votes[i], minlength=len(PRESSURE_REGIONS))
            out[i] = counts.argmax()
        return out

    def to_dict(self) -> dict:
        return {"points": self.points, "labels": self.labels, "k": self.k,
                "normal_weight": self.normal_weight,
                "fitted_on_runs": self.fitted_on_runs}

    @classmethod
    def from_dict(cls, payload: dict) -> "RegionClassifier":
        return cls(**payload)


def validate_regions(root: Path, limit: int = 5, n_fit: int = 10) -> dict:
    """Held-out agreement between the learned classifier and the real labels.

    Fitted on the FIRST n_fit runs and scored on the NEXT `limit`, so the
    number is generalisation to unseen cars rather than recall of the cells it
    memorised.
    """
    from src.data.fields import load_run

    all_runs = sorted(int(p.stem.split("_")[1])
                      for p in (root / "fields").glob("*.npz"))
    if not all_runs:
        raise RuntimeError(f"no extracted fields in {root / 'fields'}")
    fit_runs = all_runs[:n_fit]
    eval_runs = all_runs[n_fit:n_fit + limit] or all_runs[:limit]
    classifier = RegionClassifier.fit(root, fit_runs)

    paths = [root / "fields" / f"run_{r}.npz" for r in eval_runs]
    agree_total = n_total = 0
    per_region: dict[str, dict] = {}
    for path in paths:
        arrays, meta = load_run(path)
        truth_names = meta["regions"]
        truth = np.array([PRESSURE_REGIONS.index(truth_names[i])
                          if truth_names[i] in PRESSURE_REGIONS else -1
                          for i in arrays["region"]])
        guess = classifier.predict(arrays["query_xyz"], arrays["query_normal"])
        usable = truth >= 0
        agree_total += int((truth[usable] == guess[usable]).sum())
        n_total += int(usable.sum())
        for i, name in enumerate(PRESSURE_REGIONS):
            sel = usable & (truth == i)
            if sel.any():
                entry = per_region.setdefault(name, {"agree": 0, "n": 0})
                entry["agree"] += int((guess[sel] == i).sum())
                entry["n"] += int(sel.sum())

    for name, entry in per_region.items():
        entry["recall"] = entry["agree"] / max(entry["n"], 1)
    return {
        "fitted_on": fit_runs,
        "runs": list(eval_runs),
        "overall_agreement": agree_total / max(n_total, 1),
        "n_points": n_total,
        "per_region": per_region,
    }


# --------------------------------------------------------------------------
# Predictor
# --------------------------------------------------------------------------


class FieldPredictor:
    """A fields-v1 checkpoint, applied to a mesh."""

    def __init__(self, ckpt: dict, device: torch.device | None = None,
                 regions: "RegionClassifier | None" = None):
        from src.train.train_fields import build_model_from_cfg

        self.device = device or pick_device()
        self.config = ckpt["config"]
        self.cp_mean = float(ckpt["norm_stats"]["cp_mean"])
        self.cp_std = float(ckpt["norm_stats"]["cp_std"])
        self.preprocess = ckpt.get("preprocess", {})
        self.version = ckpt.get("version", "unknown")

        model = build_model_from_cfg(self.config)
        model.freeze_mean_path()          # checkpoints are saved post-freeze
        model.load_state_dict(ckpt["state_dict"])
        self.model = model.eval().to(self.device)
        payload = ckpt.get("region_classifier")
        self.regions = regions or (RegionClassifier.from_dict(payload)
                                   if payload else None)

    @classmethod
    def load(cls, path: Path | str, device: torch.device | None = None,
             regions: "RegionClassifier | None" = None):
        return cls(torch.load(path, map_location="cpu", weights_only=False),
                   device=device, regions=regions)

    @torch.no_grad()
    def predict(self, mesh_or_path, max_faces: int | None = 200_000,
                chunk: int = 20_000) -> FieldPrediction:
        """Predict Cp at every face centroid of a mesh.

        Normalisation matches stage 1 and src.data.fields exactly -- the
        bounding-box centre over the max vertex radius. Getting that wrong
        returns a confident wrong field with no error, which is why it is
        computed here rather than left to the caller.
        """
        import trimesh

        mesh = (trimesh.load(mesh_or_path, force="mesh", process=False)
                if not hasattr(mesh_or_path, "faces") else mesh_or_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        centre = mesh.bounds.mean(axis=0)
        radius = float(np.linalg.norm(mesh.vertices - centre, axis=1).max())

        centroids = np.asarray(mesh.triangles_center, np.float64)
        normals = np.asarray(mesh.face_normals, np.float64)
        areas = np.asarray(mesh.area_faces, np.float64)

        total_area = float(areas.sum())
        if max_faces and len(centroids) > max_faces:
            keep = np.linspace(0, len(centroids) - 1, max_faces).astype(int)
            centroids, normals, areas = centroids[keep], normals[keep], areas[keep]
        # Subsampling the faces removes that fraction of the wetted area, and
        # the integral scales with area, so without this the returned Cd is a
        # fraction of the real one -- 0.08 instead of 0.31 at 200k of 753k
        # faces. Same correction as in the trainer.
        area_scale = total_area / max(float(areas.sum()), 1e-12)

        query = ((centroids - centre) / radius).astype(np.float32)
        n_cloud = int(self.preprocess.get("n_cloud", 2048))
        rng = np.random.default_rng(0)
        cloud_idx = rng.choice(len(query), size=min(n_cloud, len(query)),
                               replace=False)
        cloud = torch.tensor(query[cloud_idx], device=self.device).unsqueeze(0)

        cps, sigmas = [], []
        for start in range(0, len(query), chunk):
            q = torch.tensor(query[start:start + chunk],
                             device=self.device).unsqueeze(0)
            n = torch.tensor(normals[start:start + chunk].astype(np.float32),
                             device=self.device).unsqueeze(0)
            mean, sigma = self.model.split(self.model(cloud, q, n))
            cps.append((mean.squeeze(0).squeeze(-1).cpu().numpy()
                        * self.cp_std + self.cp_mean))
            sigmas.append(sigma.squeeze(0).squeeze(-1).cpu().numpy() * self.cp_std)

        cp = np.concatenate(cps)
        sigma = np.concatenate(sigmas)
        region = (self.regions.predict(query, normals.astype(np.float32))
                  if self.regions is not None
                  else np.zeros(len(query), dtype=np.int16))

        a_ref = _frontal_area(mesh)
        cp_t = torch.tensor(cp, dtype=torch.float32).reshape(1, -1, 1)
        nrm_t = torch.tensor(normals, dtype=torch.float32).unsqueeze(0)
        area_t = torch.tensor(areas, dtype=torch.float32).unsqueeze(0)
        ref_t = torch.tensor([a_ref], dtype=torch.float32)
        cd = float(integrate_drag(cp_t, nrm_t, area_t, ref_t, axis=0)) * area_scale
        cl = float(integrate_drag(cp_t, nrm_t, area_t, ref_t, axis=2)) * area_scale

        summary: dict[str, float] = {}
        for i, name in enumerate(PRESSURE_REGIONS):
            sel = region == i
            summary[name] = (round(float(np.average(cp[sel], weights=areas[sel])), 5)
                             if sel.any() and areas[sel].sum() > 0 else 0.0)

        return FieldPrediction(
            cp=cp, sigma=sigma, region=region, pressure_summary=summary,
            cd=cd, cl=cl, frontal_area_m2=a_ref,
            mean_sigma=float(sigma.mean()),
            faces_used=len(cp), area_scale=area_scale,
        )


def _frontal_area(mesh, n: int = 1400) -> float:
    from src.data.fields import _frontal_area as fa

    return fa(mesh, n)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Serve a stage-2 field checkpoint")
    ap.add_argument("--checkpoint", default=str(repo / "runs" / "fields" / "best.pt"))
    ap.add_argument("--root", default=str(repo / "data" / "drivaerml"))
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--validate-regions", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    if args.validate_regions:
        report = validate_regions(Path(args.root), limit=args.limit)
        print("learned region classifier vs the real solid names")
        print(f"  fitted on runs {report['fitted_on']}")
        print(f"  scored on HELD-OUT runs {report['runs']}, "
              f"{report['n_points']:,} points\n")
        print(f"  OVERALL AGREEMENT  {report['overall_agreement']:.1%}\n")
        print(f"  {'region':14} {'recall':>8} {'points':>10}")
        for name in PRESSURE_REGIONS:
            e = report["per_region"].get(name)
            if e:
                print(f"  {name:14} {e['recall']:8.1%} {e['n']:10,}")
        print("\n  Fitted on real solid-derived labels, scored on cars it never")
        print("  saw. The agent's own meshes have no solid names, so this is")
        print("  the only route to pressure_summary at serve time.")
        return 0

    if not args.mesh:
        ap.print_help()
        return 0

    predictor = FieldPredictor.load(args.checkpoint)
    pred = predictor.predict(args.mesh)
    print(f"mesh     {args.mesh}")
    print(f"model    {predictor.version}")
    print(f"faces    {len(pred.cp):,}  (area scale {pred.area_scale:.2f}x)")
    print(f"A_ref    {pred.frontal_area_m2:.4f} m2\n")
    print(f"  Cd {pred.cd:+.4f}   Cl {pred.cl:+.4f}   mean sigma {pred.mean_sigma:.4f}\n")
    print("  pressure_summary (Evaluation contract):")
    for name, value in pred.pressure_summary.items():
        print(f"    {name:14} {value:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
