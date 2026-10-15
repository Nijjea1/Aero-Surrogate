"""The numbers metrics.json is made of.  OWNER: Paramveer.

Plain numpy, no torch: every function takes arrays and returns plain floats,
lists and dicts, so the same code grades a checkpoint, a reference model or a
predictions file, and the tests can check each one against a known answer.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

# Central-interval levels for the reliability diagram.
COVERAGE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.683, 0.7, 0.8, 0.9, 0.95, 0.99)


def _arrays(*xs):
    out = [np.asarray(x, dtype=np.float64).reshape(-1) for x in xs]
    n = len(out[0])
    if any(len(x) != n for x in out):
        raise ValueError(f"length mismatch: {[len(x) for x in out]}")
    return out


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------


def r2_score(y_true, y_pred) -> float:
    """1 - SS_res / SS_tot. NaN when the truth has no spread to explain."""
    y_true, y_pred = _arrays(y_true, y_pred)
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) if len(y_true) else 0.0
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - float(((y_true - y_pred) ** 2).sum()) / ss_tot


def regression_metrics(y_true, y_pred, r2_ceiling: float | None = None) -> dict:
    """R2, MAE, RMSE, max error, n; plus R2 as a share of the noise ceiling."""
    y_true, y_pred = _arrays(y_true, y_pred)
    err = np.abs(y_pred - y_true)
    out = {
        "n": int(len(y_true)),
        "r2": r2_score(y_true, y_pred),
        "mae": float(err.mean()) if len(err) else float("nan"),
        "rmse": float(np.sqrt((err**2).mean())) if len(err) else float("nan"),
        "max_error": float(err.max()) if len(err) else float("nan"),
        "mean_true": float(y_true.mean()) if len(y_true) else float("nan"),
    }
    if r2_ceiling is not None and r2_ceiling > 0:
        out["r2_ceiling"] = float(r2_ceiling)
        out["share_of_ceiling"] = out["r2"] / r2_ceiling
    return out


def per_category(y_true, y_pred, categories, r2_ceilings: dict | None = None) -> dict:
    """THE number that matters. Published results show R2 of 0.82-0.90 within a
    single car category dropping to ~0.60 across the diverse set. Closing that
    gap is our contribution, so it must be measured per category."""
    y_true, y_pred = _arrays(y_true, y_pred)
    categories = np.asarray(categories)
    return {
        str(c): regression_metrics(
            y_true[categories == c], y_pred[categories == c],
            (r2_ceilings or {}).get(str(c)),
        )
        for c in sorted(set(categories.tolist()))
    }


def noise_ceiling(y_true, label_std) -> dict:
    """The R2 a perfect model of the mean would still not beat.

    DrivAerNet's `Std Cd` is the solver's own run-to-run spread on each design,
    so even the true mean misses each label by that much:

        R2_max = 1 - mean(Std Cd^2) / var(Average Cd)
    """
    y_true, label_std = _arrays(y_true, label_std)
    var = float(y_true.var())
    noise = float((label_std**2).mean())
    return {
        "n": int(len(y_true)),
        "var_cd": var,
        "mean_label_std": float(label_std.mean()),
        "r2_ceiling": 1.0 - noise / var if var > 0 else float("nan"),
    }


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def calibration_curve(y_true, y_pred, sigma, n_bins: int = 10,
                      levels=COVERAGE_LEVELS, noise_floor: float | None = None) -> dict:
    """Reliability diagram. When the model says 90%, is it right 90%?
    This is what proves the uncertainty head means something.

    Two views, because they fail differently:

    * coverage: for each nominal level p, the share of designs whose error
      sits inside the central Gaussian interval mu +- z_p sigma. Points on the
      diagonal mean the error bars are the right width on average.
    * sigma_bins: designs sorted by predicted sigma into equal-count bins, with
      the RMSE actually observed in each bin. A rising RMSE means sigma ranks
      designs by risk, which is what the gate relies on. Coverage can be
      perfect while this is flat.
    """
    y_true, y_pred, sigma = _arrays(y_true, y_pred, sigma)
    if not len(sigma) or not np.all(np.isfinite(sigma)) or float(sigma.max()) <= 0:
        return {"available": False, "reason": "no positive sigma to calibrate"}
    err = np.abs(y_pred - y_true)
    safe = np.maximum(sigma, 1e-12)

    coverage = []
    for p in levels:
        z = NormalDist().inv_cdf(0.5 + p / 2)
        coverage.append({"expected": float(p), "observed": float((err <= z * safe).mean())})
    gaps = [abs(c["observed"] - c["expected"]) for c in coverage]

    order = np.argsort(sigma, kind="stable")
    bins = []
    for idx in np.array_split(order, min(n_bins, len(order))):
        if not len(idx):
            continue
        bins.append({
            "n": int(len(idx)),
            "mean_sigma": float(sigma[idx].mean()),
            "rmse": float(np.sqrt((err[idx] ** 2).mean())),
        })

    z_scores = (y_pred - y_true) / safe
    out = {
        "available": True,
        "n": int(len(y_true)),
        "mean_sigma": float(sigma.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "coverage_1sigma": float((err <= safe).mean()),
        "coverage_2sigma": float((err <= 2 * safe).mean()),
        # Mean |observed - expected| over the levels: 0 is perfect.
        "miscalibration": float(np.mean(gaps)),
        # Above 1: errors are bigger than the bars claim (overconfident).
        "z_std": float(z_scores.std()),
        "nll": float(np.mean(0.5 * np.log(2 * math.pi * safe**2) + 0.5 * z_scores**2)),
        "sigma_error_spearman": spearman(sigma, err),
        "coverage": coverage,
        "sigma_bins": bins,
    }
    if noise_floor is not None:
        # The labels themselves are this noisy; a smaller sigma claims more
        # certainty than the simulation that produced them.
        out["noise_floor"] = float(noise_floor)
        out["share_below_noise_floor"] = float((sigma < noise_floor).mean())
    return out


def spearman(a, b) -> float:
    """Rank correlation, ties averaged. NaN if either side is constant."""
    a, b = _arrays(a, b)
    if len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # Average the ranks of tied values.
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return (sums / counts)[inverse]


# --------------------------------------------------------------------------
# Knowing when it does not know
# --------------------------------------------------------------------------


def auroc(score_in, score_out) -> float:
    """Chance a random OOD design scores higher than a random in-distribution
    one. 0.5 is a coin flip, 1.0 a perfect separation. Ties count half."""
    score_in, score_out = (np.asarray(s, dtype=np.float64).reshape(-1) for s in (score_in, score_out))
    if not len(score_in) or not len(score_out):
        return float("nan")
    ranks = _ranks(np.concatenate([score_in, score_out])) + 1
    u = ranks[len(score_in):].sum() - len(score_out) * (len(score_out) + 1) / 2
    return float(u / (len(score_in) * len(score_out)))


def ood_detection(score_in, score_out, threshold: float) -> dict:
    """Does the model flag designs it has not seen like?
    This is the headline claim: it knows when it does not know.

    Works for any score where higher means stranger: the Mahalanobis distance,
    or sigma itself, which is how we show sigma alone does not do the job.
    """
    score_in = np.asarray(score_in, dtype=np.float64).reshape(-1)
    score_out = np.asarray(score_out, dtype=np.float64).reshape(-1)
    return {
        "threshold": float(threshold),
        "n_in": int(len(score_in)),
        "n_out": int(len(score_out)),
        "detection_rate": float((score_out > threshold).mean()) if len(score_out) else float("nan"),
        "false_alarm_rate": float((score_in > threshold).mean()) if len(score_in) else float("nan"),
        "auroc": auroc(score_in, score_out),
        "mean_in": float(score_in.mean()) if len(score_in) else float("nan"),
        "mean_out": float(score_out.mean()) if len(score_out) else float("nan"),
    }


def gate_report(y_true, y_pred, accepted) -> dict:
    """What the gate buys: error on the designs it lets through vs stops."""
    y_true, y_pred = _arrays(y_true, y_pred)
    accepted = np.asarray(accepted, dtype=bool).reshape(-1)
    err = np.abs(y_pred - y_true)
    def mae(mask):
        return float(err[mask].mean()) if mask.any() else None
    return {
        "n": int(len(err)),
        "pass_rate": float(accepted.mean()) if len(err) else float("nan"),
        "mae_accepted": mae(accepted),
        "mae_rejected": mae(~accepted),
    }


class Mahalanobis:
    """Distance from a training set, the same formula as src/train/ood.py.

    Re-derived here rather than imported, so the harness can check the
    detector a checkpoint ships with, and score subsets of columns (the
    16 aero dials without the body-style indicators) the checkpoint's
    detector was not fitted on.
    """

    def __init__(self, mean, cov_inv, threshold: float = float("inf")):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov_inv = np.asarray(cov_inv, dtype=np.float64)
        self.threshold = float(threshold)

    @classmethod
    def fit(cls, x_train, x_calib=None, quantile: float = 0.99, shrinkage: float = 1e-3):
        x_train = np.asarray(x_train, dtype=np.float64)
        if x_train.ndim != 2 or len(x_train) <= x_train.shape[1]:
            raise ValueError(f"need more rows than columns, got {x_train.shape}")
        cov = np.cov(x_train, rowvar=False).reshape(x_train.shape[1], x_train.shape[1])
        cov = cov + shrinkage * np.eye(len(cov)) * np.trace(cov) / len(cov)
        det = cls(x_train.mean(axis=0), np.linalg.inv(cov))
        calib = x_train if x_calib is None else x_calib
        det.threshold = float(np.quantile(det.score(calib), quantile))
        return det

    @classmethod
    def from_checkpoint(cls, payload: dict):
        return cls(payload["mean"], payload["cov_inv"], payload["threshold"])

    def score(self, x) -> np.ndarray:
        delta = np.atleast_2d(np.asarray(x, dtype=np.float64)) - self.mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, self.cov_inv, delta), 0.0))


# --------------------------------------------------------------------------
# Stage 2: surface fields
# --------------------------------------------------------------------------


def field_error(true_field, pred_field, regions=None, region_names=None) -> dict:
    """Error on a predicted surface field (Cp per surface point).

    relative_l2 is ||pred - true|| / ||true|| over the whole surface, the
    usual headline for field surrogates. Per region gives the same in the
    eight contract regions when a region id per point is supplied.
    """
    true_field, pred_field = _arrays(true_field, pred_field)
    diff = pred_field - true_field
    norm = float(np.linalg.norm(true_field))
    out = {
        "n_points": int(len(true_field)),
        "relative_l2": float(np.linalg.norm(diff) / norm) if norm > 0 else float("nan"),
        "mae": float(np.abs(diff).mean()) if len(diff) else float("nan"),
        "rmse": float(np.sqrt((diff**2).mean())) if len(diff) else float("nan"),
    }
    if regions is not None:
        regions = np.asarray(regions).reshape(-1)
        names = region_names or sorted(set(regions.tolist()))
        out["per_region"] = {}
        for i, name in enumerate(names):
            key = i if region_names else name
            mask = regions == key
            if mask.any():
                out["per_region"][str(name)] = {
                    "n_points": int(mask.sum()),
                    "mae": float(np.abs(diff[mask]).mean()),
                    "mean_true": float(true_field[mask].mean()),
                    "mean_pred": float(pred_field[mask].mean()),
                }
    return out
