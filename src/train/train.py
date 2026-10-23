"""Stage 1 trainer.  OWNER: Avneet.  GATE: real R^2 posted by Oct 15.

    python -m src.train.train --config configs/baseline.yaml

If you cannot hit the gate by the 15th, switch to AirfRANS. Do not keep pushing.

Save checkpoints Pranoy's TorchPredictor can load:
    {"state_dict", "config", "norm_stats", "version", "val_metrics"}

Normalise targets so the loss is well conditioned; undo it at inference.

One trainer serves both stage-1 paths -- point clouds and design parameters --
because the config picks the dataset and the model. Same loop, same
normalisation, same checkpoint format, so whatever Pranoy writes to load one
loads the other.

Training runs in TWO PHASES by default (train.two_stage):

  1. the mean, on MSE, selected on validation MAE
  2. the variance, on the Gaussian NLL, against a FROZEN mean, selected on
     validation NLL

They are split because the two outputs want different stopping epochs and,
trained together, actively damage each other: the NLL weights each sample by
its own confidence, so the mean stops improving while sigma quietly collapses.
Measured on the synthetic fixture, joint training gave R^2 0.885 with 1-sigma
coverage of 0.16-0.53 depending on the seed; the quadratic ridge sanity check
on the same split reached 0.960. The split fixes both ends at once -- the mean
trains on a stable loss, and the variance fits the residuals of a mean that
cannot move, which is a well-posed one-answer problem.

Scope note: evaluate() computes exactly what this file needs -- a quantity to
early-stop on and enough signal to see that the uncertainty head is alive. It
is NOT the evaluation suite. Per-category R^2, reliability diagrams, OOD
detection rates and the speedup number are Paramveer's harness, run against
the checkpoint this writes. Do not grow this function into that.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.contracts import SCHEMA_VERSION
from src.data.dataset import load_config, make_category_splits, make_splits, preprocess_spec
from src.train.model import build_model, count_parameters, gaussian_nll, split_prediction
from src.train.ood import OODDetector

# SCHEMA_VERSION is imported from src.contracts, never redeclared here. It was
# hardcoded to "1.0.0" and left behind when the contract moved to 2.0.0, so
# every params-v1 checkpoint carried a stamp that disagreed with the contract
# it was actually trained against. Found in review by Pranoy.


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------


@dataclass
class Norm:
    """Target (and for the tabular path, input) standardisation.

    Fitted on the TRAIN split only -- fitting on everything leaks the test
    distribution into the model and inflates the gate number.
    """

    target_mean: float
    target_std: float
    param_mean: list[float] | None = None
    param_std: list[float] | None = None

    def encode_target(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.target_mean) / self.target_std

    def decode_target(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.target_std + self.target_mean

    def decode_sigma(self, s: torch.Tensor) -> torch.Tensor:
        """Sigma is a scale, so it only picks up the std -- never the mean."""
        return s * self.target_std


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except Exception:  # noqa: BLE001
        return "nogit"


def build_dataset(cfg: dict, split: str, ids: list[str] | None = None):
    """The config decides which dataset. Both expose the same interface."""
    if str(cfg["data"].get("kind", "pointcloud")).lower() == "table":
        from src.data.tabular import DesignParamsDataset

        return DesignParamsDataset(cfg, split=split, ids=ids)
    from src.data.dataset import DrivAerNetDataset

    return DrivAerNetDataset(cfg, split=split, ids=ids)


def fit_norm(cfg: dict, train_ds) -> Norm:
    targets = np.asarray(train_ds.targets, dtype=np.float64)
    std = float(targets.std())
    if std < 1e-9:
        raise ValueError("train targets have no variance -- wrong target column?")

    norm = Norm(target_mean=float(targets.mean()), target_std=std)
    if str(cfg["data"].get("kind", "pointcloud")).lower() == "table":
        x = np.asarray(train_ds.x, dtype=np.float64)
        px_std = x.std(axis=0)
        px_std[px_std < 1e-9] = 1.0  # a constant column contributes nothing
        norm.param_mean = x.mean(axis=0).tolist()
        norm.param_std = px_std.tolist()
    return norm


def _standardise_inputs(x: torch.Tensor, norm: Norm, device: torch.device) -> torch.Tensor:
    """Point clouds arrive already unit-sphere normalised; parameters do not."""
    if norm.param_mean is None:
        return x
    mean = torch.tensor(norm.param_mean, dtype=x.dtype, device=device)
    std = torch.tensor(norm.param_std, dtype=x.dtype, device=device)
    return (x - mean) / std


# --------------------------------------------------------------------------
# Metrics -- only what training itself needs. See the scope note above.
# --------------------------------------------------------------------------


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination. 1.0 is perfect, 0.0 is predicting the mean."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def evaluate(model, loader, device, norm) -> dict:
    """Run the model over a loader. Returns metrics in ORIGINAL target units.

    'loss' stays in normalised units because that is what we early-stop on;
    everything else is denormalised so the numbers mean Cd.
    """
    model.eval()
    mus: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    loss_sum = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float().reshape(-1)
        x = _standardise_inputs(x, norm, device)
        y_norm = norm.encode_target(y)

        pred = model(x)
        loss_sum += float(gaussian_nll(pred, y_norm)) * y.shape[0]
        n += y.shape[0]

        mu, sigma = split_prediction(pred)
        mus.append(norm.decode_target(mu).cpu().numpy())
        sigmas.append(norm.decode_sigma(sigma).cpu().numpy())
        trues.append(y.cpu().numpy())

    mu = np.concatenate(mus) if mus else np.zeros(0)
    sigma = np.concatenate(sigmas) if sigmas else np.zeros(0)
    true = np.concatenate(trues) if trues else np.zeros(0)
    err = np.abs(mu - true)

    metrics = {
        "loss": loss_sum / max(n, 1),
        "r2": r2_score(true, mu),
        "mae": float(err.mean()) if err.size else float("nan"),
        "rmse": float(np.sqrt((err**2).mean())) if err.size else float("nan"),
        "n": int(n),
    }
    # Training diagnostic, not a calibration result: if the head is working,
    # ~68% of errors fall inside one sigma. Wildly off means the head is dead
    # (sigma pinned at a bound) or miscalibrated, and we want to see that at
    # epoch 10, not on Oct 25.
    if sigma.size and float(sigma.mean()) > 0:
        metrics["mean_sigma"] = float(sigma.mean())
        metrics["coverage_1sigma"] = float((err <= sigma).mean())
        metrics["coverage_2sigma"] = float((err <= 2 * sigma).mean())
    return metrics


def _fmt(metrics: dict) -> str:
    bits = [f"loss {metrics['loss']:7.4f}", f"R2 {metrics['r2']:.4f}",
            f"MAE {metrics['mae']:.5f}"]
    if "coverage_1sigma" in metrics:
        bits.append(f"sigma {metrics['mean_sigma']:.5f}")
        bits.append(f"cov1 {metrics['coverage_1sigma']:.2f}")
    return "  ".join(bits)


# --------------------------------------------------------------------------
# Resumable state
# --------------------------------------------------------------------------

STATE_FILENAME = "last.pt"


def _save_state(path: Path, state: dict) -> None:
    """Write training state atomically.

    temp-then-rename, because the session can die DURING the write. A
    half-written last.pt is worse than none -- it looks resumable and is not.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_state(out_dir: Path) -> dict | None:
    """Load resumable state from out_dir, or None when there is none."""
    path = Path(out_dir) / STATE_FILENAME
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: {path} is unreadable ({exc}); starting fresh")
        return None
    if state.get("kind") != "training_state":
        return None
    # A kill landing mid-write leaves the partial file behind. It is never
    # read -- that is the point of the rename -- but sweep it so the directory
    # does not accumulate one per death.
    (path.parent / (path.name + ".tmp")).unlink(missing_ok=True)
    return state


# --------------------------------------------------------------------------
# One training phase
# --------------------------------------------------------------------------


def _run_phase(
    phase: str, model, loaders, device, norm, *,
    params, epochs: int, patience: int, lr: float, weight_decay: float,
    select_key: str, mean_only: bool, beta: float,
    grad_clip: float, use_amp: bool, log_path: Path, history: list[dict],
    state_path: Path | None = None, resume: dict | None = None,
    done_phases: list[str] | None = None, cfg: dict | None = None,
) -> tuple[dict, dict]:
    """Train one phase. Returns (best_val_metrics, best_state_dict copy).

    select_key is the validation metric to minimise -- 'mae' for the mean
    phase, 'loss' for the variance phase.

    If state_path is given, the full training state is written there after
    EVERY epoch, and `resume` (a state dict loaded from it) restarts this
    phase mid-flight. Kaggle sessions are killed without warning, so a run
    that cannot resume can lose its uncheckpointed training progress.
    """
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=max(3, patience // 4)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = float("inf")
    best_epoch = 0
    best_val: dict = {}
    best_state: dict = copy.deepcopy(model.state_dict())
    start_epoch = 1

    if resume is not None and resume.get("phase") == phase:
        model.load_state_dict(resume["model"])
        # The weights are what matter. Optimiser momentum, the LR schedule
        # position and the loss scale are conveniences, and a state file
        # written by a different torch version or truncated by the kill
        # itself can be missing them. Losing momentum costs an epoch or two;
        # refusing to resume costs the whole run.
        for name, obj in (("optimiser", optimiser), ("scheduler", scheduler),
                          ("scaler", scaler)):
            try:
                obj.load_state_dict(resume[name])
            except Exception as exc:  # noqa: BLE001
                print(f"  note: {name} state not restored ({type(exc).__name__}); "
                      f"continuing with a fresh one")
        best_score = resume["best_score"]
        best_epoch = resume["best_epoch"]
        best_val = resume["best_val"]
        best_state = resume["best_state"]
        start_epoch = int(resume["epoch"]) + 1
        print(f"\n[{phase}] RESUMED at epoch {start_epoch} "
              f"(best so far {best_epoch}: val {select_key} {best_score:.5f})")

    print(f"\n[{phase}] {epochs} epochs max, early stop after {patience}, "
          f"selecting on val {select_key}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        run_loss = 0.0
        n_seen = 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float().reshape(-1)
            x = _standardise_inputs(x, norm, device)
            y_norm = norm.encode_target(y)

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = gaussian_nll(model(x), y_norm, beta=beta, mean_only=mean_only)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimiser)
            scaler.update()

            run_loss += float(loss.detach()) * y.shape[0]
            n_seen += y.shape[0]

        train_loss = run_loss / max(n_seen, 1)
        val = evaluate(model, loaders["val"], device, norm)
        score = val[select_key]
        scheduler.step(score)

        history.append({"phase": phase, "epoch": epoch, "train_loss": train_loss,
                        "lr": optimiser.param_groups[0]["lr"],
                        **{f"val_{k}": v for k, v in val.items()}})
        with log_path.open("a") as fh:
            fh.write(json.dumps(history[-1]) + "\n")

        improved = score < best_score - 1e-7
        if improved:
            best_score, best_epoch, best_val = score, epoch, val
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or improved or epoch == 1:
            print(f"  {epoch:4d}  train {train_loss:8.4f}   val {_fmt(val)}"
                  f"{' *' if improved else ''}")

        # Every epoch, not every improvement: resuming needs the LIVE optimiser
        # and scheduler state, not just the best weights.
        if state_path is not None:
            _save_state(state_path, {
                "kind": "training_state", "phase": phase, "epoch": epoch,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_score": best_score, "best_epoch": best_epoch,
                "best_val": best_val, "best_state": best_state,
                "history": history, "norm_stats": asdict(norm),
                "done_phases": list(done_phases or []),
                "config": cfg,
            })

        if epoch - best_epoch >= patience:
            print(f"  early stop at {epoch} (best {best_epoch})")
            break

    model.load_state_dict(best_state)
    print(f"[{phase}] best epoch {best_epoch}:  {_fmt(best_val)}")
    return best_val, best_state


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------


def train(cfg: dict, split_mode: str = "random", held_out: str | None = None,
          device_str: str | None = None, limit: int | None = None,
          resume: bool = False) -> dict:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = pick_device(device_str)
    t = cfg["train"]

    run_name = f"{cfg.get('run_name', 'run')}-{split_mode}"
    out_dir = Path(t["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run      {run_name}")
    print(f"device   {device}")
    print(f"out      {out_dir}")

    # --- splits -----------------------------------------------------------
    if split_mode == "category":
        splits = make_category_splits(cfg, held_out=held_out, seed=seed)
        held = splits.pop("_held_out_category")[0]
        print(f"split    held-out category {held!r} (the OOD split)")
    else:
        splits = make_splits(cfg, seed=seed)
        held = None
        print("split    random by design id")

    if limit:
        splits = {
            k: (v[:limit] if k == "train" else v[: max(2, limit // 4)])
            for k, v in splits.items()
        }
        print(f"limit    {limit} train designs (smoke run -- not a real number)")

    datasets = {k: build_dataset(cfg, k, ids=v) for k, v in splits.items()}
    for name, ds in datasets.items():
        print(f"  {name:6} {len(ds)}")

    # Warm the cache up front so epoch timings are honest and the first epoch
    # is not competing with the disk.
    total = sum(len(d) for d in datasets.values())
    secs = sum(ds.warm_cache(verbose=False) for ds in datasets.values()
               if hasattr(ds, "warm_cache"))
    if secs > 0.5:
        print(f"caching  {total} designs in {secs:.1f} s")

    norm = fit_norm(cfg, datasets["train"])
    print(f"norm     target mean {norm.target_mean:.5f}  sd {norm.target_std:.5f}")

    loaders = {
        name: DataLoader(
            ds,
            batch_size=int(t["batch_size"]),
            shuffle=(name == "train"),
            num_workers=int(t.get("num_workers", 0)),
            drop_last=(name == "train" and len(ds) > int(t["batch_size"])),
        )
        for name, ds in datasets.items()
    }

    # --- model ------------------------------------------------------------
    model = build_model(cfg).to(device)
    n_params = count_parameters(model)
    print(f"model    {cfg['model'].get('kind', 'pointnet')}  {n_params:,} params")

    use_amp = bool(t.get("amp", False)) and device.type == "cuda"
    grad_clip = float(t.get("grad_clip", 0.0))
    lr = float(t["lr"])
    weight_decay = float(t["weight_decay"])
    patience = int(t.get("patience", 30))
    epochs = int(t["epochs"])

    heteroscedastic = bool(cfg["model"].get("heteroscedastic", True))
    two_stage = bool(t.get("two_stage", True)) and heteroscedastic

    # --- resume -----------------------------------------------------------
    state = load_state(out_dir) if resume else None
    done_phases: list[str] = list(state.get("done_phases", [])) if state else []
    history: list[dict] = list(state.get("history", [])) if state else []

    log_path = out_dir / "metrics.jsonl"
    state_path = out_dir / STATE_FILENAME
    if state is None:
        log_path.write_text("")
    else:
        print(f"\nresume   {state_path}")
        print(f"         phase {state['phase']!r} epoch {state['epoch']}, "
              f"{len(done_phases)} phase(s) already done")
        # The variance phase trains a frozen model, so the freeze has to be in
        # place BEFORE its weights are loaded or the state dict will not line up
        # with a model whose mean path is still live.
        if state["phase"] == "variance":
            model.freeze_mean_path()
        model.load_state_dict(state["model"])

    t_start = time.time()

    if two_stage:
        mean_val: dict = state.get("mean_val", {}) if state else {}
        if "mean" not in done_phases:
            # Phase 1: the mean alone, on MSE, selected on MAE.
            mean_val, _ = _run_phase(
                "mean", model, loaders, device, norm,
                params=list(model.parameters()), epochs=epochs, patience=patience,
                lr=lr, weight_decay=weight_decay, select_key="mae",
                mean_only=True, beta=0.0, grad_clip=grad_clip, use_amp=use_amp,
                log_path=log_path, history=history,
                state_path=state_path,
                resume=state if state and state["phase"] == "mean" else None,
                done_phases=done_phases, cfg=cfg,
            )
            done_phases.append("mean")
        else:
            print("\n[mean] already complete, skipping")

        # Phase 2: the variance alone, on the NLL, against the frozen mean.
        if not model._mean_path_frozen:
            var_params = model.freeze_mean_path()
        else:
            var_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\nfrozen   mean path; {sum(p.numel() for p in var_params)} "
              f"variance parameters remain trainable")
        best_val, _ = _run_phase(
            "variance", model, loaders, device, norm,
            params=var_params,
            epochs=int(t.get("variance_epochs", epochs)),
            patience=int(t.get("variance_patience", patience)),
            lr=float(t.get("variance_lr", lr)), weight_decay=0.0,
            select_key="loss", mean_only=False, beta=0.0,
            grad_clip=grad_clip, use_amp=use_amp,
            log_path=log_path, history=history,
            state_path=state_path,
            resume=state if state and state["phase"] == "variance" else None,
            done_phases=done_phases, cfg=cfg,
        )
        done_phases.append("variance")
        # The mean is frozen, so phase 2 cannot have changed R2 or MAE. If it
        # has, the freeze leaked -- fail loudly rather than ship a checkpoint
        # whose reported accuracy is not the accuracy it has.
        if abs(best_val["mae"] - mean_val["mae"]) > 1e-6:
            raise RuntimeError(
                f"the mean moved during the variance phase "
                f"(val MAE {mean_val['mae']:.6f} -> {best_val['mae']:.6f}) -- "
                f"freeze_mean_path did not hold"
            )
    else:
        beta = float(t.get("nll_beta", 0.5))
        print(f"\nsingle-stage training, beta-NLL {beta}")
        best_val, _ = _run_phase(
            "joint", model, loaders, device, norm,
            params=list(model.parameters()), epochs=epochs, patience=patience,
            lr=lr, weight_decay=weight_decay,
            select_key="loss" if heteroscedastic else "mae",
            mean_only=False, beta=beta, grad_clip=grad_clip, use_amp=use_amp,
            log_path=log_path, history=history,
            state_path=state_path,
            resume=state if state and state["phase"] == "joint" else None,
            done_phases=done_phases, cfg=cfg,
        )

    wall = time.time() - t_start

    # --- OOD detector -----------------------------------------------------
    # Sigma answers "how noisy are the labels here", not "have I seen this".
    # See src/train/ood.py for the measurement that forced this.
    #
    # Fitted on the train split, threshold calibrated on VAL -- distances to a
    # set the detector was fitted on are optimistically small, and a threshold
    # read off them rejects legitimate designs at serving time.
    detector = None

    if norm.param_mean is not None:
        # Tabular path: distance in standardised parameter space.
        space = "parameters"

        def ood_inputs(name: str) -> np.ndarray:
            x = np.asarray(datasets[name].x, np.float64)
            return (x - np.asarray(norm.param_mean)) / np.asarray(norm.param_std)
    else:
        # Point-cloud path: a mesh has no parameter vector, so measure
        # distance in the model's own pooled feature space instead. Without
        # this, geometry checkpoints shipped with ood=None and the gate fell
        # back to thresholding sigma -- which was measured and does not work.
        space = "features"

        @torch.no_grad()
        def ood_inputs(name: str) -> np.ndarray:
            model.eval()
            out = []
            for x, _ in loaders[name]:
                x = x.to(device, non_blocking=True).float()
                out.append(model.features(_standardise_inputs(x, norm, device))
                           .cpu().numpy())
            return np.concatenate(out).astype(np.float64)

    try:
        detector = OODDetector.fit(
            ood_inputs("train"),
            x_calib=ood_inputs("val"),
            quantile=float(t.get("ood_quantile", 0.99)),
            space=space,
        )
        if detector.method == "knn":
            kind = f"kNN k={detector.k}"
        else:
            kind = "Mahalanobis, " + ("diagonal" if detector.diagonal
                                      else "full covariance")
        print(f"\nood      {kind} in {space} space "
              f"({detector.train_scores['n_dims']} dims)")
        if space == "features":
            print("         NOTE: feature-space detection is a weak signal, not "
                  "a gate -- see src/train/ood.py")
        print(f"         threshold {detector.threshold:.3f} at the "
              f"{detector.quantile:.0%} quantile of val")
        print(f"         val distances  p50 {detector.train_scores['p50']:.2f}  "
              f"p95 {detector.train_scores['p95']:.2f}  "
              f"max {detector.train_scores['max']:.2f}")
        flagged = float((detector.score(ood_inputs("test")) > detector.threshold).mean())
        label = "OOD designs flagged" if held else "test designs flagged (expect ~1%)"
        print(f"         {label}: {flagged:.1%}")
    except ValueError as exc:
        print(f"\nood      skipped: {exc}")

    # --- test -------------------------------------------------------------
    test = evaluate(model, loaders["test"], device, norm)
    version = f"{run_name}-{date.today().isoformat()}-{git_hash()}"
    ckpt_path = out_dir / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
            "norm_stats": asdict(norm),
            "version": version,
            "val_metrics": best_val,
            # Not in the five keys Pranoy needs, but the service cannot build
            # model input correctly without it. See preprocess_spec in
            # src/data/dataset.py and docs/HANDOFF.md.
            "preprocess": preprocess_spec(cfg),
            # The OOD detector. contracts.is_in_distribution takes its score
            # alongside sigma; a sigma-only gate was measured and does not work.
            "ood": detector.to_dict() if detector is not None else None,
            "test_metrics": test,
            "schema_version": SCHEMA_VERSION,
            # Provenance, so a checkpoint is attributable without the repo.
            "git_commit": git_hash(),
            "trained_body_styles": sorted(
                {c for c in (datasets["train"].categories or []) if c}) or None,
            # Exact design ids per split, so a result is reproducible without
            # re-deriving the split and hoping the seed still matches.
            "splits": {k: list(v) for k, v in splits.items()},
            "held_out_category": held,
            "wall_clock_s": wall,
            "n_params": n_params,
        },
        ckpt_path,
    )

    with (out_dir / "config.resolved.yaml").open("w") as fh:
        import yaml

        yaml.safe_dump(cfg, fh, sort_keys=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "run": run_name, "version": version, "split": split_mode,
                "held_out_category": held, "n_params": n_params,
                "two_stage": two_stage, "epochs_run": len(history),
                "wall_clock_s": wall, "val_metrics": best_val, "test_metrics": test,
                "preprocess": preprocess_spec(cfg),
            },
            indent=2,
        )
    )

    print(f"\n{len(history)} epochs total, {wall:.1f} s")
    print(f"val    {_fmt(best_val)}")
    print(f"test   {_fmt(test)}")
    print(f"\n{'=' * 64}")
    print(f"  TEST R2 = {test['r2']:.4f}   (n={test['n']}, MAE={test['mae']:.5f}"
          + (f", 1-sigma coverage {test['coverage_1sigma']:.2f}"
             if "coverage_1sigma" in test else "") + ")")
    print(f"{'=' * 64}")
    # The run finished, so the mid-flight state is dead weight -- and leaving
    # it would make the next run silently resume a completed one.
    state_path.unlink(missing_ok=True)

    print(f"\ncheckpoint  {ckpt_path}")
    print(f"summary     {out_dir / 'summary.json'}")
    return {"val": best_val, "test": test, "checkpoint": str(ckpt_path)}


def main() -> None:
    """Train, early-stop on val, save best.pt, print the test R2.
    Post that number in the group chat -- it is the Oct 15 gate."""
    ap = argparse.ArgumentParser(description="Stage-1 baseline trainer")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--split", choices=["random", "category"], default="random",
                    help="'category' holds one body style out entirely (the OOD split)")
    ap.add_argument("--held-out-category", default=None)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--epochs", type=int, default=None, help="override the config")
    ap.add_argument("--limit", type=int, default=None, help="tiny smoke run")
    ap.add_argument("--single-stage", action="store_true",
                    help="train mean and variance together (the old, worse recipe)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from last.pt in out_dir (for Kaggle sessions "
                         "that get killed mid-run)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.single_stage:
        cfg["train"]["two_stage"] = False

    train(cfg, split_mode=args.split, held_out=args.held_out_category,
          device_str=args.device, limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
