"""Evaluation harness.  OWNER: Paramveer.

    python -m src.eval.harness --checkpoint runs/drivaernet-random/best.pt \\
        --ood-checkpoint runs/drivaernet-category/best.pt

Avneet trains the model. Paramveer grades it. The person who trained a model is
the worst person to measure it -- and saying so in the submission is a
credibility signal.

Emits metrics.json: the single source for every number in the README, the
charts and the video.

What it measures, and on what:

  in_distribution      the random split's test set, graded on the checkpoint
                       trained on that split: R2, MAE, per body style, the
                       reliability diagram
  out_of_distribution  the held-out body style, graded on the checkpoint
                       trained without it
  ood_detection        does the gate catch what the model has not seen: the
                       held-out style, jointly extreme in-bounds corners, and
                       sigma alone as the comparison that motivates the
                       distance score
  gate                 pass rates and the error on what passes vs what is
                       stopped, at the checkpoint's calibrated threshold and
                       at the contract's fallback
  data_scaling         R2 against training-set size
  references           the same numbers for a mean predictor and a straight
                       line, fitted here, so the network's numbers have context
  field                stage 2 surface-pressure error, once there is a field

The data is read here, not through the training code, and every split is
checked for overlap. With no checkpoint the harness still runs: the references,
the noise ceiling and the distance checks need only the data.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import data as D
from .metrics import (  # noqa: F401 - re-exported: these are the harness's interface
    Mahalanobis,
    calibration_curve,
    field_error,
    gate_report,
    noise_ceiling,
    ood_detection,
    per_category,
    r2_score,
    regression_metrics,
)
from .models import CheckpointModel, LinearModel, MeanModel, sha256

METRICS_FORMAT = "aero-surrogate-metrics/1"
LABEL_PREFIXES = ("Average ", "Std ")


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------


def default_columns(table: D.Table) -> list[str]:
    """Model inputs when no checkpoint says otherwise: the contract's parameter
    names if the data has them all, else every morph column; then body style."""
    try:
        from src import contracts
        names = list(contracts.PARAM_NAMES)
    except Exception:  # pragma: no cover - contracts always imports
        names = []
    if not names or not all(n in table.columns for n in names):
        names = [c for c in table.columns if not c.startswith(LABEL_PREFIXES) and not c.startswith("cat_")]
    styles = sorted(set(table.categories))
    return names + [f"cat_{s}" for s in ("fastback", "notchback", "estateback") if s in styles]


def _grade(y, mu, sigma, cats, label_std, noise_floor=None) -> dict:
    ceilings = {c: noise_ceiling(y[cats == c], label_std[cats == c])["r2_ceiling"]
                for c in sorted(set(cats.tolist()))}
    return {
        "overall": regression_metrics(y, mu, noise_ceiling(y, label_std)["r2_ceiling"]),
        "per_category": per_category(y, mu, cats, ceilings),
        "calibration": calibration_curve(y, mu, sigma, noise_floor=noise_floor) if sigma is not None else None,
    }


def _references(x_train, y_train, x_test, y, cats, label_std) -> dict:
    out = {}
    for cls in (MeanModel, LinearModel):
        model = cls(x_train, y_train)
        mu, sigma = model.predict(x_test)
        graded = _grade(y, mu, sigma, cats, label_std)
        graded["description"] = model.description
        out[model.name] = graded
    return out


def _gate(sigma, scores, cats, y, mu, sigma_threshold, ood_threshold) -> dict:
    """Apply contracts.is_in_distribution when the contract has it, so the
    harness grades the gate the service actually runs."""
    try:
        from src.contracts import is_in_distribution
    except ImportError:
        is_in_distribution = None

    accepted, reasons = [], {"uncertainty": 0, "out_of_distribution": 0, "body_style": 0}
    for i in range(len(y)):
        s = None if scores is None else float(scores[i])
        if is_in_distribution is not None:
            ok, why = is_in_distribution(float(sigma[i]), s, str(cats[i]),
                                         uncertainty_threshold=sigma_threshold,
                                         ood_threshold=ood_threshold)
        else:
            ok = sigma[i] <= sigma_threshold and (s is None or s <= ood_threshold)
            why = None if ok else ("uncertainty" if sigma[i] > sigma_threshold else "out of distribution")
        accepted.append(ok)
        if not ok:
            key = ("uncertainty" if why.startswith("uncertainty") else
                   "body_style" if "body style" in why else "out_of_distribution")
            reasons[key] += 1
    report = gate_report(y, mu, accepted)
    report.update({"sigma_threshold": sigma_threshold, "ood_threshold": ood_threshold,
                   "rejected_by": reasons})
    return report


def _corners(x_train, columns, n: int, rng) -> dict:
    """Synthetic designs inside the data's per-parameter range: every dial at
    an end of its range ("corners"), and uniform anywhere in the box. Each
    coordinate is legal; the combinations are not like any real design."""
    aero = [j for j, c in enumerate(columns) if not c.startswith("cat_")]
    cat = [j for j, c in enumerate(columns) if c.startswith("cat_")]
    lo, hi = x_train[:, aero].min(axis=0), x_train[:, aero].max(axis=0)
    out = {}
    for kind in ("corners", "uniform"):
        x = np.zeros((n, len(columns)))
        if kind == "corners":
            x[:, aero] = np.where(rng.random((n, len(aero))) < 0.5, lo, hi)
        else:
            x[:, aero] = lo + rng.random((n, len(aero))) * (hi - lo)
        if cat:
            x[np.arange(n), np.asarray(cat)[rng.integers(0, len(cat), n)]] = 1.0
        out[kind] = x
    return out


def _standardiser(x_train):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-9] = 1.0
    return lambda x: (x - mean) / std


def _scaling_curve(table, split, columns, fractions, rng) -> list[dict]:
    x_tr = D.design_matrix(table, split["train"], columns)
    y_tr = table.column(D.TARGET, split["train"])
    x_te = D.design_matrix(table, split["test"], columns)
    y_te = table.column(D.TARGET, split["test"])
    points = []
    for f in fractions:
        k = max(len(columns) + 2, int(round(f * len(x_tr))))
        idx = rng.choice(len(x_tr), size=min(k, len(x_tr)), replace=False)
        mu, _ = LinearModel(x_tr[idx], y_tr[idx]).predict(x_te)
        points.append({"n_train": int(len(idx)), "r2": r2_score(y_te, mu),
                       "mae": float(np.abs(mu - y_te).mean())})
    return points


def score_scaling(table, split, paths, device, warnings) -> list[dict]:
    """Checkpoints trained on growing subsets, each scored on the split's test
    set: the model's own data-scaling curve, sorted by training-set size."""
    y_test = table.column(D.TARGET, split["test"])
    points = []
    for p in paths:
        m = CheckpointModel(p, device)
        if m.held_out_category:
            warnings.append(f"scaling checkpoint {p} was trained with a style held out; skipped")
            continue
        if not m.n_train:
            warnings.append(f"scaling checkpoint {p} does not record its training-set size; skipped")
            continue
        mu, sigma = m.predict(D.design_matrix(table, split["test"], m.columns))
        err = np.abs(mu - y_test)
        points.append({"n_train": int(m.n_train), "r2": r2_score(y_test, mu), "mae": float(err.mean()),
                       "coverage_1sigma": float(np.mean(err <= sigma)) if sigma is not None else None,
                       "checkpoint": m.info["path"]})
    return sorted(points, key=lambda r: r["n_train"])


def _calibration_warnings(cal: dict | None) -> list[str]:
    if not cal or not cal.get("available"):
        return []
    out = []
    for level, key, expected in ((1, "coverage_1sigma", 0.683), (2, "coverage_2sigma", 0.954)):
        if abs(cal[key] - expected) > 0.05:
            out.append(f"{level}-sigma coverage is {cal[key]:.2f} against {expected:.2f} expected: the error "
                       "bars are " + ("too narrow" if cal[key] < expected else "too wide"))
    rho = cal.get("sigma_error_spearman")
    if rho is not None and math.isfinite(rho) and rho < 0.2:
        out.append(f"sigma barely ranks designs by risk (rank correlation with the error {rho:.2f}): "
                   "it says how wide the bars are on average, not which designs to distrust")
    return out


def _git_commit() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def _contract_thresholds() -> dict:
    try:
        from src import contracts
    except Exception:  # pragma: no cover
        return {}
    return {
        "schema_version": getattr(contracts, "SCHEMA_VERSION", None),
        "uncertainty": getattr(contracts, "UNCERTAINTY_THRESHOLD", None),
        "ood_fallback": getattr(contracts, "OOD_SCORE_THRESHOLD", None),
    }


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def evaluate(
    data_path=None,
    splits_path=None,
    checkpoint=None,
    ood_checkpoint=None,
    scaling=(),
    seed: int = 0,
    cfd_hours: float | None = None,
    device: str = "cpu",
    n_synthetic: int = 2000,
    field_checkpoint=None,
    fields_root="data/drivaerml",
) -> dict:
    rng = np.random.default_rng(seed)
    warnings: list[str] = []
    path = D.find_data(data_path)
    table = D.load_table(path)
    label_std_all = table.columns.get(D.LABEL_STD)
    if label_std_all is None:
        raise KeyError(f"{path} has no {D.LABEL_STD!r} column; the noise ceiling needs it")

    splits = D.load_splits(splits_path) if splits_path else D.build_splits(table, seed=seed)
    if not splits_path:
        warnings.append("no splits file given: splits rebuilt with the training algorithm "
                        f"(seed {seed}); pass the run's runs/splits.json to grade on exactly its split")
    D.check_disjoint(splits["random"])
    if "category" in splits:
        D.check_disjoint(splits["category"])

    model = CheckpointModel(checkpoint, device) if checkpoint else None
    ood_model = CheckpointModel(ood_checkpoint, device) if ood_checkpoint else None
    columns = model.columns if model else (ood_model.columns if ood_model else default_columns(table))
    if model and ood_model and set(model.columns) != set(ood_model.columns):
        warnings.append("the two checkpoints take different inputs, so the in- and out-of-distribution "
                        "numbers are not for the same model")
    contract = _contract_thresholds()
    sigma_threshold = contract.get("uncertainty")
    noise_floor = float(np.mean(label_std_all))

    out: dict = {
        "format": METRICS_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv) if sys.argv and sys.argv[0] else None,
        "python": platform.python_version(),
        "data": {
            "path": str(path).replace("\\", "/"),
            "sha256": sha256(path),
            "n_designs": len(table.ids),
            "categories": {c: table.categories.count(c) for c in sorted(set(table.categories))},
            "noise_ceiling": noise_ceiling(table.column(D.TARGET), label_std_all),
            "licence": "DrivAerNet, CC BY-NC 4.0 (non-commercial)",
        },
        "splits": {
            "source": splits["source"],
            "seed": splits.get("seed"),
            "random": {k: len(v) for k, v in splits["random"].items()},
            "category": ({k: (len(v) if isinstance(v, list) else v) for k, v in splits["category"].items()}
                         if "category" in splits else None),
        },
        "inputs": columns,
        "contract": contract,
        "models": {},
    }

    # --- in distribution: random split ------------------------------------
    rs = splits["random"]
    x_train = D.design_matrix(table, rs["train"], columns)
    y_train = table.column(D.TARGET, rs["train"])
    x_val = D.design_matrix(table, rs["val"], columns)
    x_test = D.design_matrix(table, rs["test"], columns)
    y_test = table.column(D.TARGET, rs["test"])
    cats_test = table.category(rs["test"])
    std_test = table.column(D.LABEL_STD, rs["test"])

    ind: dict = {"split": "random", "n_test": len(y_test)}
    ind["references"] = _references(x_train, y_train, x_test, y_test, cats_test, std_test)
    if model:
        out["models"]["in_distribution"] = model.info
        if model.held_out_category:
            warnings.append(f"--checkpoint was trained with {model.held_out_category!r} held out; "
                            "the in-distribution numbers need the random-split checkpoint")
        mu, sigma = model.predict(x_test)
        ind.update(_grade(y_test, mu, sigma, cats_test, std_test, noise_floor))
        ind["model"] = "checkpoint"
        reported = (model.info.get("self_reported_test_metrics") or {}).get("r2")
        if reported is not None and abs(reported - ind["overall"]["r2"]) > 0.01:
            warnings.append(f"in-distribution R2 graded here ({ind['overall']['r2']:.4f}) differs from "
                            f"the checkpoint's own report ({reported:.4f}): check the split and data file")
        scores = model.ood_score(x_test)
        thresholds = {"calibrated": model.ood.threshold if model.ood else None,
                      "contract_fallback": contract.get("ood_fallback")}
        if sigma_threshold is not None:
            ind["gate"] = {name: _gate(sigma, scores if t is not None else None, cats_test, y_test, mu,
                                       sigma_threshold, t if t is not None else math.inf)
                           for name, t in thresholds.items() if name == "calibrated" or t is not None}
            fb = ind["gate"].get("contract_fallback")
            if fb and thresholds["calibrated"] and fb["pass_rate"] < ind["gate"]["calibrated"]["pass_rate"] - 0.1:
                warnings.append(
                    f"at the contract's fallback OOD threshold ({thresholds['contract_fallback']}) the gate "
                    f"passes {fb['pass_rate']:.1%} of ordinary test designs vs "
                    f"{ind['gate']['calibrated']['pass_rate']:.1%} at the checkpoint's calibrated "
                    f"{thresholds['calibrated']:.2f}: the service must pass the checkpoint's threshold")
        warnings += _calibration_warnings(ind["calibration"])
        out["speed"] = model.latency(x_test)
        if cfd_hours:
            out["speed"]["cfd_hours_per_design"] = cfd_hours
            out["speed"]["speedup_vs_cfd"] = cfd_hours * 3.6e6 / out["speed"]["single_design_ms_median"]
    else:
        ind["model"] = None
        warnings.append("no --checkpoint: the in-distribution numbers are reference models only")
    out["in_distribution"] = ind

    # --- distance checks the data alone supports ---------------------------
    # Fitted here, independently of any checkpoint's detector, in standardised
    # inputs, threshold at the 99% quantile of val -- the recipe in ood.py.
    stdz = _standardiser(x_train)
    det = Mahalanobis.fit(stdz(x_train), stdz(x_val))
    synth = _corners(x_train, columns, n_synthetic, rng)
    ood: dict = {
        "harness_detector": {
            "fitted_on": "random train split, standardised; threshold = 99% quantile of val",
            "threshold": det.threshold,
            "false_alarm_on_unseen_test": float((det.score(stdz(x_test)) > det.threshold).mean()),
            "corners_flagged": float((det.score(stdz(synth["corners"])) > det.threshold).mean()),
            "uniform_in_box_flagged": float((det.score(stdz(synth["uniform"])) > det.threshold).mean()),
        },
    }
    if model and model.ood:
        ood["checkpoint_detector"] = {
            "threshold": model.ood.threshold,
            "false_alarm_on_unseen_test": float((model.ood_score(x_test) > model.ood.threshold).mean()),
            "corners_flagged": float((model.ood_score(synth["corners"]) > model.ood.threshold).mean()),
            "uniform_in_box_flagged": float((model.ood_score(synth["uniform"]) > model.ood.threshold).mean()),
        }

    # --- out of distribution: one body style held out ------------------------
    if "category" in splits:
        cs = splits["category"]
        held = cs["held_out_category"]
        cx_train = D.design_matrix(table, cs["train"], columns)
        cy_train = table.column(D.TARGET, cs["train"])
        cx_val = D.design_matrix(table, cs["val"], columns)
        cx_test = D.design_matrix(table, cs["test"], columns)
        cy_test = table.column(D.TARGET, cs["test"])
        ccats = table.category(cs["test"])
        cstd = table.column(D.LABEL_STD, cs["test"])

        oodd: dict = {"held_out_category": held, "n_test": len(cy_test)}
        oodd["references"] = _references(cx_train, cy_train, cx_test, cy_test, ccats, cstd)

        cstdz = _standardiser(cx_train)
        aero = [j for j, c in enumerate(columns) if not c.startswith("cat_")]
        full = Mahalanobis.fit(cstdz(cx_train), cstdz(cx_val))
        aero_only = Mahalanobis.fit(cstdz(cx_train)[:, aero], cstdz(cx_val)[:, aero])
        ood["held_out_style"] = {
            "held_out_category": held,
            "note": ("in-distribution side is the category split's val set, which set the threshold, "
                     "so its false-alarm rate is ~1% by construction"),
            "harness_detector_all_inputs": ood_detection(full.score(cstdz(cx_val)), full.score(cstdz(cx_test)),
                                                         full.threshold),
            "harness_detector_aero_only": ood_detection(aero_only.score(cstdz(cx_val)[:, aero]),
                                                        aero_only.score(cstdz(cx_test)[:, aero]),
                                                        aero_only.threshold),
        }

        if ood_model:
            out["models"]["out_of_distribution"] = ood_model.info
            if ood_model.held_out_category and ood_model.held_out_category != held:
                warnings.append(f"--ood-checkpoint held out {ood_model.held_out_category!r} but the split "
                                f"holds out {held!r}: the out-of-distribution numbers are not valid")
            # Each checkpoint gets its inputs in its own column order.
            mx_val = D.design_matrix(table, cs["val"], ood_model.columns)
            mx_test = D.design_matrix(table, cs["test"], ood_model.columns)
            mu, sigma = ood_model.predict(mx_test)
            oodd.update(_grade(cy_test, mu, sigma, ccats, cstd, noise_floor))
            oodd["model"] = "checkpoint"
            _, sigma_val = ood_model.predict(mx_val)
            ood["held_out_style"]["sigma_alone"] = ood_detection(
                sigma_val, sigma, sigma_threshold if sigma_threshold is not None else float(np.quantile(sigma_val, 0.99)))
            if ood_model.ood:
                ood["held_out_style"]["checkpoint_detector"] = ood_detection(
                    ood_model.ood_score(mx_val), ood_model.ood_score(mx_test), ood_model.ood.threshold)
                if sigma_threshold is not None:
                    oodd["gate"] = {"calibrated": _gate(sigma, ood_model.ood_score(mx_test), ccats, cy_test, mu,
                                                        sigma_threshold, ood_model.ood.threshold)}
        else:
            oodd["model"] = None
            warnings.append("no --ood-checkpoint: the held-out-style numbers are reference models only")
        out["out_of_distribution"] = oodd
    out["ood_detection"] = ood

    # --- data scaling -----------------------------------------------------
    scaling_points = score_scaling(table, rs, scaling, device, warnings)
    out["data_scaling"] = {
        "test_set": "random split test, fixed for every point",
        "model": scaling_points,
        "linear_reference": _scaling_curve(table, rs, columns, (0.02, 0.05, 0.1, 0.25, 0.5, 1.0), rng),
    }
    if not scaling_points:
        warnings.append("no --scaling checkpoints: the data-scaling curve has the linear reference only")

    if field_checkpoint:
        from .fields import grade_checkpoint

        out["field"] = grade_checkpoint(field_checkpoint, fields_root, device)
        if not out["field"]["available"]:
            warnings.append(f"surface pressure not graded: {out['field']['reason']}")
    else:
        out["field"] = {"available": False,
                        "reason": "no --field-checkpoint: surface-pressure error needs a stage-2 checkpoint"}
    out["headline"] = _headline(out)
    out["warnings"] = warnings
    return out


def _headline(out: dict) -> dict:
    """The handful of numbers the README and the video quote. None where the
    run did not grade a checkpoint, so nothing is quoted that was not measured."""
    ind, oodd = out["in_distribution"], out.get("out_of_distribution") or {}
    held = out["ood_detection"].get("held_out_style", {})
    get = lambda d, *ks: _dig(d, ks)  # noqa: E731
    have = ind.get("model") == "checkpoint"
    have_ood = oodd.get("model") == "checkpoint"
    return {
        "r2_in_distribution": get(ind, "overall", "r2") if have else None,
        # The ceiling of the test set itself, which share_of_ceiling divides
        # by; the whole-dataset figure (0.906 on DrivAerNet) is under data.
        "r2_ceiling_test_set": get(ind, "overall", "r2_ceiling"),
        "r2_ceiling_dataset": get(out, "data", "noise_ceiling", "r2_ceiling"),
        "share_of_ceiling": get(ind, "overall", "share_of_ceiling") if have else None,
        "mae_in_distribution": get(ind, "overall", "mae") if have else None,
        "r2_linear_reference": get(ind, "references", "linear", "overall", "r2"),
        "coverage_1sigma": get(ind, "calibration", "coverage_1sigma") if have else None,
        "held_out_category": oodd.get("held_out_category"),
        "r2_held_out_category": get(oodd, "overall", "r2") if have_ood else None,
        "ood_detection_rate": get(held, "checkpoint_detector", "detection_rate") if have_ood else None,
        "ood_false_alarm_rate": (get(out["ood_detection"], "checkpoint_detector", "false_alarm_on_unseen_test")
                                 if have else None),
        "sigma_alone_detection_auroc": get(held, "sigma_alone", "auroc") if have_ood else None,
        "latency_ms": get(out, "speed", "single_design_ms_median"),
    }


def _dig(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _clean(obj):
    """JSON has no NaN or infinity; write them as null."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# A drive path (C:\ or C:/) or a rooted POSIX path, starting a word.
_ABSOLUTE = re.compile(r"(?:(?<![\w.])[A-Za-z]:[\\/]|(?<![\w.:/])/(?=[^/\s]))[^\s\"']*")


def portable_paths(obj, root: Path | None = None):
    """Absolute paths made repo-relative, or cut to the file name when they
    are outside the repo, so a published metrics file names no one's disk."""
    root = (root or Path.cwd()).resolve()

    def fix(match):
        text = match.group(0)
        try:
            return Path(text).resolve().relative_to(root).as_posix()
        except (ValueError, OSError):
            return Path(text.replace("\\", "/")).name
    if isinstance(obj, dict):
        return {k: portable_paths(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [portable_paths(v, root) for v in obj]
    if isinstance(obj, str) and ("/" in obj or "\\" in obj):
        return _ABSOLUTE.sub(fix, obj)
    return obj


def write(metrics: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(metrics), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def summary(m: dict) -> str:
    h = m["headline"]
    f = lambda v, fmt=".4f": "n/a" if v is None else format(v, fmt)  # noqa: E731
    lines = [
        f"data       {m['data']['n_designs']} designs, R2 ceiling {f(m['data']['noise_ceiling']['r2_ceiling'])}",
        f"in dist    R2 {f(h['r2_in_distribution'])}  ({f(h['share_of_ceiling'], '.0%')} of ceiling)  "
        f"MAE {f(h['mae_in_distribution'], '.5f')}  1-sigma coverage {f(h['coverage_1sigma'], '.2f')}",
        f"reference  straight line R2 {f(h['r2_linear_reference'])}",
        f"held out   {h['held_out_category'] or 'n/a'}: R2 {f(h['r2_held_out_category'])}  "
        f"detected {f(h['ood_detection_rate'], '.0%')}  sigma-alone AUROC {f(h['sigma_alone_detection_auroc'], '.2f')}",
        f"speed      {f(h['latency_ms'], '.2f')} ms per design",
    ]
    lines += [f"WARNING    {w}" for w in m["warnings"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grade a surrogate checkpoint and write metrics.json.")
    ap.add_argument("--checkpoint", help="checkpoint trained on the random split")
    ap.add_argument("--ood-checkpoint", help="checkpoint trained with one body style held out")
    ap.add_argument("--scaling", nargs="*", default=[], help="checkpoints trained on growing subsets")
    ap.add_argument("--data", help="DrivAerNet parametric CSV (default: data/drivaernet/...)")
    ap.add_argument("--splits", help="runs/splits.json from src.data.dataset --dump-splits")
    ap.add_argument("--out", default="runs/metrics.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfd-hours", type=float, help="CFD wall-clock per design, for the speedup figure")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--label", help="shown on the results page, e.g. 'Final: H100 checkpoints, Oct 26'")
    ap.add_argument("--field-checkpoint", help="stage-2 surface-pressure checkpoint (graded on its own test cars)")
    ap.add_argument("--fields-root", default="data/drivaerml", help="DrivAerML root holding fields/run_<n>.npz")
    args = ap.parse_args(argv)

    metrics = evaluate(args.data, args.splits, args.checkpoint, args.ood_checkpoint, args.scaling,
                       seed=args.seed, cfd_hours=args.cfd_hours, device=args.device,
                       field_checkpoint=args.field_checkpoint, fields_root=args.fields_root)
    metrics["label"] = args.label
    path = write(portable_paths(metrics), args.out)
    print(summary(_clean(metrics)))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
