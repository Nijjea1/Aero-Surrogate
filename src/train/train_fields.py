"""Stage-2 trainer: surface pressure fields.  OWNER: Avneet.

    python -m src.train.train_fields --config configs/fields.yaml
    python -m src.train.train_fields --config configs/fields.yaml --resume

Two phases, as in stage 1 and for the same measured reason: fit the mean on
MSE, then fit the per-point variance against a FROZEN mean. Trained jointly the
Gaussian NLL weights each point by its own confidence, so sigma collapses where
the model is already confident and the field ends up confidently wrong. In
stage 1 that cost 1-sigma coverage 0.24 against a target of 0.68; here it would
be worse, because a collapsed sigma makes the whole surface look trustworthy
rather than one scalar.

WHAT IS REPORTED, and why each one

  Cp R2 and MAE, per point       does the field match at all
  coverage                       is the per-point sigma honest
  per-region Cp MAE              where on the car it is wrong. Directly the
                                 separate experimental field endpoint's regions;
                                 current agent runs do not consume this model
  integrated Cd, pred vs true    pressure-only integral consistency; spatial
                                 errors may cancel, so report per-car integral
                                 errors and pointwise errors separately
  corr(sigma, pprime2)           a descriptive diagnostic. pPrime2MeanTrim is the
                                 LES's own pressure-fluctuation variance, so it
                                 marks where the flow is physically unsteady.
                                 Correlation does not establish calibration
                                 of errors in a time-averaged pressure target
                                 or accuracy on generated viewer geometry.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.contracts import SCHEMA_VERSION
from src.data.fields import FIELD_VERSION, SurfaceFieldDataset, field_splits
from src.train.field_model import (
    SurfaceFieldModel,
    field_nll,
    integral_consistency_loss,
)
from src.train.train import git_hash, pick_device, r2_score, seed_everything

STATE_FILENAME = "last.pt"


@dataclass
class FieldNorm:
    """Cp standardisation, fitted on the TRAIN runs only."""

    mean: float
    std: float

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean

    def decode_sigma(self, s: torch.Tensor) -> torch.Tensor:
        return s * self.std


def build_model_from_cfg(cfg: dict) -> SurfaceFieldModel:
    m = cfg.get("model", {})
    return SurfaceFieldModel(
        latent_dim=m.get("latent_dim", 256),
        local_k=tuple(m["local_k"]) if isinstance(m.get("local_k"), list)
                else m.get("local_k", (8, 32, 128)),
        local_dim=m.get("local_dim", 96),
        n_freq=m.get("n_freq", 8),
        hidden=m.get("hidden", 256),
        n_layers=m.get("n_layers", 4),
        dropout=m.get("dropout", 0.0),
        heteroscedastic=m.get("heteroscedastic", True),
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def evaluate_fields(model, loader, device, norm: FieldNorm,
                    regions: list[str]) -> dict:
    """Metrics in ORIGINAL Cp units. See the module docstring for why each."""
    model.eval()
    preds, sigmas, truths, pprimes, region_ids = [], [], [], [], []
    cd_pred, cd_true = [], []
    cl_pred, cl_true = [], []
    loss_sum = n_points = 0.0

    for batch in loader:
        b = _to_device(batch, device)
        out = model(b["cloud"], b["query_xyz"], b["query_normal"])
        target = norm.encode(b["cp"])
        loss_sum += float(field_nll(out, target, model)) * b["cp"].numel()
        n_points += b["cp"].numel()

        mean, sigma = model.split(out)
        mean_cp = norm.decode(mean)
        preds.append(mean_cp.reshape(-1).cpu().numpy())
        sigmas.append(norm.decode_sigma(sigma).reshape(-1).cpu().numpy())
        truths.append(b["cp"].reshape(-1).cpu().numpy())
        pprimes.append(b["pprime2"].reshape(-1).cpu().numpy())
        region_ids.append(b["region"].reshape(-1).cpu().numpy())

        # Integrated Cd over the SAMPLED points, predicted vs true, on the same
        # points -- a consistency check on the field, not a drag measurement.
        from src.train.field_model import integrate_drag

        # Scaled from the sampled subset to the whole surface, so the number
        # is comparable with the reported Cd rather than 0.05% of it.
        scale = b["area_scale"]
        cd_pred.append((integrate_drag(mean_cp, b["query_normal"],
                                       b["query_area"], b["a_ref"], axis=0)
                        * scale).cpu().numpy())
        cd_true.append((integrate_drag(b["cp"], b["query_normal"],
                                       b["query_area"], b["a_ref"], axis=0)
                        * scale).cpu().numpy())
        # Lift as well as drag. The integral loss constrains both (after the
        # axis fix), so both have to be reported -- otherwise "the integral is
        # consistent" is a claim about a quantity nobody measured.
        cl_pred.append((integrate_drag(mean_cp, b["query_normal"],
                                       b["query_area"], b["a_ref"], axis=2)
                        * scale).cpu().numpy())
        cl_true.append((integrate_drag(b["cp"], b["query_normal"],
                                       b["query_area"], b["a_ref"], axis=2)
                        * scale).cpu().numpy())

    pred = np.concatenate(preds)
    sigma = np.concatenate(sigmas)
    true = np.concatenate(truths)
    pprime = np.concatenate(pprimes)
    region = np.concatenate(region_ids)
    err = np.abs(pred - true)

    metrics = {
        "loss": loss_sum / max(n_points, 1),
        "cp_r2": r2_score(true, pred),
        "cp_mae": float(err.mean()),
        "cp_rmse": float(np.sqrt((err**2).mean())),
        "n_points": int(len(true)),
        "mean_sigma": float(sigma.mean()),
        "coverage_1sigma": float((err <= sigma).mean()),
        "coverage_2sigma": float((err <= 2 * sigma).mean()),
    }

    # Does the model's own uncertainty track the LES's unsteadiness?
    if pprime.std() > 0 and sigma.std() > 0:
        metrics["corr_sigma_pprime2"] = float(
            np.corrcoef(sigma, np.sqrt(np.maximum(pprime, 0.0)))[0, 1]
        )

    cdp, cdt = np.concatenate(cd_pred), np.concatenate(cd_true)
    metrics["cd_partial_pred"] = float(cdp.mean())
    metrics["cd_partial_true"] = float(cdt.mean())
    metrics["cd_partial_mae"] = float(np.abs(cdp - cdt).mean())
    if cdt.std() > 0:
        metrics["cd_partial_r2"] = r2_score(cdt, cdp)

    clp, clt = np.concatenate(cl_pred), np.concatenate(cl_true)
    metrics["cl_partial_pred"] = float(clp.mean())
    metrics["cl_partial_true"] = float(clt.mean())
    metrics["cl_partial_mae"] = float(np.abs(clp - clt).mean())
    if clt.std() > 0:
        metrics["cl_partial_r2"] = r2_score(clt, clp)

    per_region = {}
    for i, name in enumerate(regions):
        sel = region == i
        if sel.any():
            per_region[name] = {
                "mae": round(float(err[sel].mean()), 5),
                "cp_true": round(float(true[sel].mean()), 5),
                "cp_pred": round(float(pred[sel].mean()), 5),
                "n": int(sel.sum()),
            }
    metrics["per_region"] = per_region
    return metrics


def _fmt(m: dict) -> str:
    bits = [f"loss {m['loss']:7.4f}", f"Cp R2 {m['cp_r2']:.4f}",
            f"MAE {m['cp_mae']:.4f}"]
    if "coverage_1sigma" in m:
        bits.append(f"cov1 {m['coverage_1sigma']:.2f}")
    if "corr_sigma_pprime2" in m:
        bits.append(f"corr {m['corr_sigma_pprime2']:+.2f}")
    return "  ".join(bits)


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_state(out_dir: Path) -> dict | None:
    path = Path(out_dir) / STATE_FILENAME
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: {path} unreadable ({exc}); starting fresh")
        return None
    if state.get("kind") != "field_training_state":
        return None
    (path.parent / (path.name + ".tmp")).unlink(missing_ok=True)
    return state


def _run_phase(phase: str, model, loaders, device, norm, regions, *, params,
               epochs: int, patience: int, lr: float, weight_decay: float,
               select_key: str, mean_only: bool, beta: float, grad_clip: float,
               log_path: Path, history: list[dict], state_path: Path,
               resume: dict | None, done: list[str], cfg: dict,
               integral_weight: float = 0.0) -> dict:
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=max(2, patience // 4))

    best_score, best_epoch, best_val = float("inf"), 0, {}
    best_state = copy.deepcopy(model.state_dict())
    start = 1

    if resume is not None and resume.get("phase") == phase:
        model.load_state_dict(resume["model"])
        for name, obj in (("optimiser", optimiser), ("scheduler", scheduler)):
            try:
                obj.load_state_dict(resume[name])
            except Exception as exc:  # noqa: BLE001
                print(f"  note: {name} not restored ({type(exc).__name__})")
        best_score, best_epoch = resume["best_score"], resume["best_epoch"]
        best_val, best_state = resume["best_val"], resume["best_state"]
        start = int(resume["epoch"]) + 1
        print(f"\n[{phase}] RESUMED at epoch {start} "
              f"(best {best_epoch}: {select_key} {best_score:.5f})")

    print(f"\n[{phase}] {epochs} epochs max, early stop {patience}, "
          f"selecting on val {select_key}")
    for epoch in range(start, epochs + 1):
        model.train()
        loaders["train"].dataset.set_epoch(epoch)
        run_loss = seen = 0.0
        for batch in loaders["train"]:
            b = _to_device(batch, device)
            optimiser.zero_grad(set_to_none=True)
            out = model(b["cloud"], b["query_xyz"], b["query_normal"])
            target = norm.encode(b["cp"])
            loss = field_nll(out, target, model, beta=beta, mean_only=mean_only)
            if integral_weight > 0.0:
                # Integrate in ORIGINAL Cp units, since that is where the
                # coefficient lives; the normalised field would give a
                # meaningless drag.
                n = model.n_targets
                decoded = torch.cat(
                    [norm.decode(out[..., :n]), out[..., n:]], dim=-1)
                loss = loss + integral_weight * integral_consistency_loss(
                    decoded, b["cp"], b["query_normal"], b["query_area"],
                    b["a_ref"], b["area_scale"], model)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimiser.step()
            run_loss += float(loss.detach()) * b["cp"].numel()
            seen += b["cp"].numel()

        val = evaluate_fields(model, loaders["val"], device, norm, regions)
        score = val[select_key] if select_key != "neg_cp_r2" else -val["cp_r2"]
        scheduler.step(score)

        row = {"phase": phase, "epoch": epoch, "train_loss": run_loss / max(seen, 1),
               "lr": optimiser.param_groups[0]["lr"],
               **{f"val_{k}": v for k, v in val.items() if k != "per_region"}}
        history.append(row)
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        improved = score < best_score - 1e-7
        if improved:
            best_score, best_epoch, best_val = score, epoch, val
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or improved or epoch == 1:
            print(f"  {epoch:4d}  train {row['train_loss']:8.4f}   "
                  f"val {_fmt(val)}{' *' if improved else ''}")

        _save_state(state_path, {
            "kind": "field_training_state", "phase": phase, "epoch": epoch,
            "model": model.state_dict(), "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(), "best_score": best_score,
            "best_epoch": best_epoch, "best_val": best_val,
            "best_state": best_state, "history": history,
            "norm": {"mean": norm.mean, "std": norm.std},
            "done_phases": list(done), "config": cfg,
        })

        if epoch - best_epoch >= patience:
            print(f"  early stop at {epoch} (best {best_epoch})")
            break

    model.load_state_dict(best_state)
    print(f"[{phase}] best epoch {best_epoch}:  {_fmt(best_val)}")
    return best_val


def train_fields(cfg: dict, device_str: str | None = None,
                 resume: bool = False) -> dict:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = pick_device(device_str)
    d, t = cfg["data"], cfg["train"]

    root = Path(d["root"])
    runs = sorted(int(p.stem.split("_")[1]) for p in (root / "fields").glob("*.npz"))
    if not runs:
        raise RuntimeError(f"no extracted fields in {root / 'fields'} -- "
                           f"run `python -m src.data.fields --extract`")
    splits = field_splits(runs, val=int(d.get("val_runs", 5)),
                          test=int(d.get("test_runs", 5)), seed=seed)

    run_name = cfg.get("run_name", "fields")
    out_dir = Path(t["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run      {run_name}\ndevice   {device}\nout      {out_dir}")
    print(f"runs     " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"         train {splits['train']}")
    print(f"         val   {splits['val']}\n         test  {splits['test']}")

    datasets = {
        name: SurfaceFieldDataset(
            root, ids, n_query=int(d.get("n_query", 4096)),
            n_cloud=int(d.get("n_cloud", 2048)), seed=seed,
            resample=(name == "train"))
        for name, ids in splits.items()
    }
    loaders = {
        name: DataLoader(ds, batch_size=int(t["batch_size"]),
                         shuffle=(name == "train"), num_workers=0)
        for name, ds in datasets.items()
    }
    regions = datasets["train"].regions

    cp_mean, cp_std = datasets["train"].target_stats()
    norm = FieldNorm(mean=cp_mean, std=cp_std)
    print(f"norm     Cp mean {cp_mean:+.4f}  sd {cp_std:.4f}  (train runs only)")

    model = build_model_from_cfg(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model    SurfaceFieldModel  {n_params:,} params")

    state = load_state(out_dir) if resume else None
    done: list[str] = list(state.get("done_phases", [])) if state else []
    history: list[dict] = list(state.get("history", [])) if state else []
    log_path = out_dir / "metrics.jsonl"
    state_path = out_dir / STATE_FILENAME
    if state is None:
        log_path.write_text("")
    else:
        print(f"\nresume   phase {state['phase']!r} epoch {state['epoch']}")
        if state["phase"] == "variance":
            model.freeze_mean_path()
        model.load_state_dict(state["model"])

    grad_clip = float(t.get("grad_clip", 1.0))
    t_start = time.time()

    if "mean" not in done:
        _run_phase("mean", model, loaders, device, norm, regions,
                   params=list(model.parameters()), epochs=int(t["epochs"]),
                   patience=int(t.get("patience", 20)), lr=float(t["lr"]),
                   weight_decay=float(t.get("weight_decay", 1e-4)),
                   select_key="cp_mae", mean_only=True, beta=0.0,
                   grad_clip=grad_clip, log_path=log_path, history=history,
                   integral_weight=float(t.get("integral_weight", 0.0)),
                   state_path=state_path,
                   resume=state if state and state["phase"] == "mean" else None,
                   done=done, cfg=cfg)
        done.append("mean")
    else:
        print("\n[mean] already complete, skipping")

    var_params = (model.freeze_mean_path() if not model._mean_path_frozen
                  else [p for p in model.parameters() if p.requires_grad])
    print(f"\nfrozen   mean path; {sum(p.numel() for p in var_params)} "
          f"variance parameters trainable")
    best_val = _run_phase(
        "variance", model, loaders, device, norm, regions, params=var_params,
        epochs=int(t.get("variance_epochs", t["epochs"])),
        patience=int(t.get("variance_patience", 20)),
        lr=float(t.get("variance_lr", t["lr"])), weight_decay=0.0,
        select_key="loss", mean_only=False, beta=0.0, grad_clip=grad_clip,
        log_path=log_path, history=history, state_path=state_path,
        resume=state if state and state["phase"] == "variance" else None,
        done=done, cfg=cfg)

    wall = time.time() - t_start
    test = evaluate_fields(model, loaders["test"], device, norm, regions)
    version = f"{run_name}-{date.today().isoformat()}-{git_hash()}"

    # The region classifier travels with the checkpoint. Without it a served
    # mesh has no region labels -- combining the solids throws their names
    # away, and a mesh from params_to_mesh never had them -- so
    # Evaluation.pressure_summary could not be filled at all.
    #
    # Fitted on the TRAIN runs only, like everything else here. Hand-written
    # geometry thresholds were tried first and scored 13.7% agreement; this
    # scores 98.6% on held-out cars.
    region_payload = None
    try:
        from src.train.field_predict import RegionClassifier

        classifier = RegionClassifier.fit(root, splits["train"])
        region_payload = classifier.to_dict()
        print(f"\nregions  kNN classifier fitted on {len(splits['train'])} "
              f"train runs, {len(classifier.labels):,} reference cells")
    except Exception as exc:  # noqa: BLE001
        print(f"\nregions  classifier not fitted: {exc}")

    torch.save({
        "state_dict": model.state_dict(), "config": cfg,
        "norm_stats": {"cp_mean": norm.mean, "cp_std": norm.std},
        "version": version, "val_metrics": best_val, "test_metrics": test,
        "preprocess": {"version": FIELD_VERSION,
                       "input": "surface point cloud + query points",
                       "n_cloud": int(d.get("n_cloud", 2048)),
                       "normalize": "bbox centre / max vertex radius, as stage 1",
                       "target": "CpMeanTrim, standardised by norm_stats"},
        "schema_version": SCHEMA_VERSION, "splits": splits,
        "git_commit": git_hash(),
        "regions": regions, "n_params": n_params, "wall_clock_s": wall,
        "region_classifier": region_payload,
    }, out_dir / "best.pt")
    (out_dir / "summary.json").write_text(json.dumps(
        {"run": run_name, "version": version, "splits": splits,
         "n_params": n_params, "wall_clock_s": wall,
         "val_metrics": best_val, "test_metrics": test}, indent=2))
    state_path.unlink(missing_ok=True)

    print(f"\n{len(history)} epochs, {wall:.1f} s")
    print(f"val    {_fmt(best_val)}")
    print(f"test   {_fmt(test)}")
    print(f"\n{'=' * 70}")
    print(f"  Cp R2 = {test['cp_r2']:.4f}   MAE = {test['cp_mae']:.4f}   "
          f"coverage = {test.get('coverage_1sigma', float('nan')):.2f}")
    print(f"  integrated Cd (pressure only, area-scaled): pred "
          f"{test['cd_partial_pred']:.4f}  true {test['cd_partial_true']:.4f}  "
          f"per-car MAE {test['cd_partial_mae']:.4f}")
    print(f"  integrated Cl (pressure only, area-scaled): pred "
          f"{test['cl_partial_pred']:.4f}  true {test['cl_partial_true']:.4f}  "
          f"per-car MAE {test['cl_partial_mae']:.4f}")
    if "corr_sigma_pprime2" in test:
        print(f"  corr(sigma, LES unsteadiness) = {test['corr_sigma_pprime2']:+.3f}")
    print(f"{'=' * 70}")
    print("\nper-region Cp MAE (this is Evaluation.pressure_summary):")
    for name, r in sorted(test["per_region"].items(), key=lambda kv: -kv[1]["mae"]):
        print(f"  {name:14} MAE {r['mae']:.4f}   true {r['cp_true']:+.4f}  "
              f"pred {r['cp_pred']:+.4f}  ({r['n']:,} pts)")
    print(f"\ncheckpoint  {out_dir / 'best.pt'}")
    return {"val": best_val, "test": test, "checkpoint": str(out_dir / "best.pt")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 surface field trainer")
    ap.add_argument("--config", default="configs/fields.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from src.data.dataset import load_config

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    train_fields(cfg, device_str=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
