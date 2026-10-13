"""Evaluation harness.  OWNER: Paramveer.  DUE: Oct 18-22.

    python -m src.eval.harness --checkpoint runs/baseline/best.pt

Avneet trains the model. Paramveer grades it. The person who trained a model is
the worst person to measure it -- and saying so in the submission is a
credibility signal.

Emits metrics.json: the single source for every number in the README, the
charts and the video.
"""

from __future__ import annotations

import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("TODO Paramveer")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R2, MAE, RMSE, max error, n."""
    raise NotImplementedError


def per_category(y_true, y_pred, categories: list[str]) -> dict:
    """THE number that matters. Published results show R2 of 0.82-0.90 within a
    single car category dropping to ~0.60 across the diverse set. Closing that
    gap is our contribution, so it must be measured per category."""
    raise NotImplementedError


def calibration_curve(y_true, y_pred, sigma, n_bins: int = 10) -> dict:
    """Reliability diagram. When the model says 90%, is it right 90%?
    This is what proves the uncertainty head means something."""
    raise NotImplementedError


def ood_detection(sigma_in, sigma_out, threshold: float) -> dict:
    """Does the model widen its error bars on a held-out car category?
    This is the headline claim: it knows when it does not know."""
    raise NotImplementedError
