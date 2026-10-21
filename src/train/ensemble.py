"""Deep ensemble, and the uncertainty decomposition it buys.  OWNER: Avneet.

    python -m src.train.ensemble --config configs/drivaernet.yaml --members 5
    python -m src.train.ensemble --config configs/drivaernet.yaml --split category

The ensemble separates mean predictive variance from disagreement between
members. Neither term alone guarantees calibrated uncertainty or reliable OOD
detection. All members can share the same extrapolation error.

Historical training-side test measurements (not the served single model):

    single model        R2 0.8750
    best member         R2 0.9092
    ensemble of 5       R2 0.9132

An earlier description called 0.9122 a hard noise ceiling using the dataset's
Std Cd column. That interpretation was unsupported: solver coefficient spread
is not the error variance of the averaged regression label. These ordinary
held-out R2 measurements remain valid; they do not prove irreducible residuals.

OOD -- the epistemic term does not save us:

                        in distribution    body style held out
    cd R^2                   0.9132              0.3550
    aleatoric sd             0.00710             0.00788
    epistemic sd             0.00260             0.00313   (+25%)
    coverage (total sd)      0.79                0.44

Error more than doubles; disagreement rises 25%. That is the SAME failure the
single model's sigma showed (+20%), so this is now the third independent method
to fail on the same test, after sigma and eight feature-space OOD estimators.

The reason is worth understanding rather than tuning around: all M members saw
the same training set, in which cat_notchback is zero for every row. They all
learned "this input is always zero" and all extrapolate the same way, so they
agree -- confidently and wrongly. Ensemble disagreement captures variance from
random initialisation, not from a systematically absent input value.

WHAT THIS MEANS. OOD detection in this project works when the novelty is
EXPLICIT IN THE INPUT and fails when the model is asked to notice it from its
own behaviour. The parameter-space distance check flags 100% of held-out-style
designs because the body-style indicator sits far outside its training range;
sigma, ensembles and feature-space distance all fail because the model's
behaviour looks normal to the model. contracts.is_in_distribution is therefore
built the right way round -- it checks the body style explicitly, and treats
sigma and ood_score as supporting evidence rather than the gate.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import load_config, make_category_splits, make_splits
from src.train.model import build_model, split_prediction
from src.train.train import (
    Norm,
    _standardise_inputs,
    build_dataset,
    evaluate,
    pick_device,
    r2_score,
    train,
)


@dataclass
class EnsemblePrediction:
    """Per-design prediction with the two uncertainties kept separate."""

    mean: np.ndarray            # (n,)   predictive mean, original units
    aleatoric: np.ndarray       # (n,)   sd from label noise
    epistemic: np.ndarray       # (n,)   sd from member disagreement
    total: np.ndarray           # (n,)   sqrt(aleatoric^2 + epistemic^2)
    members: np.ndarray         # (M, n) each member's mean, for inspection

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "aleatoric": self.aleatoric.tolist(),
            "epistemic": self.epistemic.tolist(),
            "total": self.total.tolist(),
        }


class Ensemble:
    """M trained checkpoints, predicting as one model."""

    def __init__(self, checkpoints: list[dict], device: torch.device | None = None):
        if not checkpoints:
            raise ValueError("an ensemble needs at least one member")
        self.device = device or pick_device()
        self.checkpoints = checkpoints
        self.models = []
        for ckpt in checkpoints:
            model = build_model(ckpt["config"])
            # Members trained two-stage have a frozen mean path, and the
            # freeze has to be in place before the state dict is loaded.
            if ckpt.get("two_stage_frozen", True):
                try:
                    model.freeze_mean_path()
                except Exception:  # noqa: BLE001 - homoscedastic members
                    pass
            model.load_state_dict(ckpt["state_dict"])
            model.eval().to(self.device)
            self.models.append(model)

        # Members share a config and a split, so one set of norm stats.
        self.norm = Norm(**checkpoints[0]["norm_stats"])
        self.config = checkpoints[0]["config"]

    @classmethod
    def load(cls, paths: list[Path], device: torch.device | None = None) -> "Ensemble":
        return cls([torch.load(p, map_location="cpu", weights_only=False)
                    for p in paths], device=device)

    @torch.no_grad()
    def predict(self, loader) -> EnsemblePrediction:
        """Run every member over a loader and decompose the variance."""
        mus: list[list[np.ndarray]] = [[] for _ in self.models]
        vars_: list[list[np.ndarray]] = [[] for _ in self.models]

        for x, _ in loader:
            x = x.to(self.device, non_blocking=True).float()
            x = _standardise_inputs(x, self.norm, self.device)
            for i, model in enumerate(self.models):
                mu, sigma = split_prediction(model(x))
                mus[i].append(self.norm.decode_target(mu).cpu().numpy())
                vars_[i].append(
                    (self.norm.decode_sigma(sigma) ** 2).cpu().numpy()
                )

        member_mu = np.stack([np.concatenate(m) for m in mus])      # (M, n)
        member_var = np.stack([np.concatenate(v) for v in vars_])   # (M, n)

        mean = member_mu.mean(axis=0)
        aleatoric_var = member_var.mean(axis=0)
        # Population variance of the member means: the disagreement itself.
        epistemic_var = member_mu.var(axis=0)
        return EnsemblePrediction(
            mean=mean,
            aleatoric=np.sqrt(aleatoric_var),
            epistemic=np.sqrt(epistemic_var),
            total=np.sqrt(aleatoric_var + epistemic_var),
            members=member_mu,
        )


def _targets(loader) -> np.ndarray:
    return np.concatenate([y.numpy().reshape(-1) for _, y in loader])


def score_ensemble(pred: EnsemblePrediction, truth: np.ndarray) -> dict:
    """Metrics for the ensemble, with coverage under each uncertainty.

    Reporting coverage three ways is the point: if the total is calibrated
    where aleatoric alone is not, the epistemic term is doing real work.
    """
    err = np.abs(pred.mean - truth)
    out = {
        "r2": r2_score(truth, pred.mean),
        "mae": float(err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "n": int(len(truth)),
        "mean_aleatoric": float(pred.aleatoric.mean()),
        "mean_epistemic": float(pred.epistemic.mean()),
        "mean_total": float(pred.total.mean()),
        "coverage_aleatoric": float((err <= pred.aleatoric).mean()),
        "coverage_total": float((err <= pred.total).mean()),
        "coverage_total_2sigma": float((err <= 2 * pred.total).mean()),
    }
    out["member_r2"] = [r2_score(truth, m) for m in pred.members]
    return out


def train_ensemble(cfg: dict, n_members: int = 5, split_mode: str = "random",
                   held_out: str | None = None, device_str: str | None = None,
                   limit: int | None = None) -> dict:
    """Train M members from different seeds and evaluate them as one model."""
    base_name = cfg.get("run_name", "run")
    out_root = Path(cfg["train"]["out_dir"])
    paths: list[Path] = []

    print(f"ensemble of {n_members}, split {split_mode}\n")
    for m in range(n_members):
        member = copy.deepcopy(cfg)
        member["seed"] = int(cfg.get("seed", 0)) + 1000 * m
        member["run_name"] = f"{base_name}-m{m}"
        print(f"{'=' * 64}\nmember {m}  (seed {member['seed']})\n{'=' * 64}")
        result = train(member, split_mode=split_mode, held_out=held_out,
                       device_str=device_str, limit=limit)
        paths.append(Path(result["checkpoint"]))

    # --- evaluate as an ensemble -----------------------------------------
    device = pick_device(device_str)
    ensemble = Ensemble.load(paths, device=device)
    seed = int(cfg.get("seed", 0))
    if split_mode == "category":
        splits = make_category_splits(cfg, held_out=held_out, seed=seed)
        held = splits.pop("_held_out_category")[0]
    else:
        splits = make_splits(cfg, seed=seed)
        held = None

    from torch.utils.data import DataLoader

    report: dict = {"n_members": n_members, "split": split_mode,
                    "held_out_category": held, "members": [str(p) for p in paths]}
    print(f"\n{'=' * 64}\nENSEMBLE OF {n_members}\n{'=' * 64}")
    for name in ("val", "test"):
        ds = build_dataset(cfg, name, ids=splits[name])
        if hasattr(ds, "warm_cache"):
            ds.warm_cache(verbose=False)
        loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]))
        pred = ensemble.predict(loader)
        metrics = score_ensemble(pred, _targets(loader))
        report[name] = metrics
        print(f"\n{name}  n={metrics['n']}")
        print(f"  R2            {metrics['r2']:.4f}   "
              f"(members {min(metrics['member_r2']):.4f}..{max(metrics['member_r2']):.4f})")
        print(f"  MAE           {metrics['mae']:.5f}")
        print(f"  aleatoric sd  {metrics['mean_aleatoric']:.5f}  "
              f"coverage {metrics['coverage_aleatoric']:.2f}")
        print(f"  epistemic sd  {metrics['mean_epistemic']:.5f}")
        print(f"  total sd      {metrics['mean_total']:.5f}  "
              f"coverage {metrics['coverage_total']:.2f}  "
              f"2-sigma {metrics['coverage_total_2sigma']:.2f}")

    out_dir = out_root / f"{base_name}-ensemble-{split_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ensemble.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport  {out_dir / 'ensemble.json'}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep ensemble for stage 1")
    ap.add_argument("--config", default="configs/drivaernet.yaml")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--split", choices=["random", "category"], default="random")
    ap.add_argument("--held-out-category", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.members < 2:
        raise SystemExit("--members must be at least 2; 5 is a good default")

    cfg = load_config(args.config)
    train_ensemble(cfg, n_members=args.members, split_mode=args.split,
                   held_out=args.held_out_category, device_str=args.device,
                   limit=args.limit)


if __name__ == "__main__":
    main()
