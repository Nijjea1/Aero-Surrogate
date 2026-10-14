"""Out-of-distribution detection for the surrogate.  OWNER: Avneet.

WHY THIS EXISTS

The heteroscedastic head answers "how noisy are the labels around here". That
is not the same question as "have I seen anything like this before", and the
difference is not academic -- it was measured. Retraining with one body style
held out of the training set:

    in distribution    cd R^2 0.875    sigma 0.0067    1-sigma coverage 0.62
    held-out style     cd R^2 0.479    sigma 0.0081    1-sigma coverage 0.49

Error roughly doubles. Sigma widens about 20%. A gate thresholding sigma alone
waves the design straight through, which is exactly the failure the project
claims to prevent -- an optimiser walking into a region where the model is
wrong and confident.

The reason is structural, not a tuning problem: during training every example
is by definition familiar, so the Gaussian NLL never sees an unfamiliar one and
the variance head has no way to represent novelty. You cannot learn to say "I
have not seen this" from data in which everything has been seen.

WHAT THIS DOES INSTEAD

Measures distance from the training set directly, in standardised parameter
space, with a Mahalanobis distance against the training mean and covariance:

    d(x) = sqrt( (x - mu)^T  S^-1  (x - mu) )

Mahalanobis rather than Euclidean because the morph parameters are correlated
and on wildly different scales (millimetres beside dimensionless curvatures).
S^-1 accounts for both, so "far" means far relative to how the training data
actually spreads, not far in raw units.

It needs no retraining, no extra data and no gradient. It is fitted once from
the training split, stored in the checkpoint, and evaluated in microseconds.

The threshold is CALIBRATED, not guessed: set so that a chosen fraction
(default 99%) of held-out in-distribution designs pass. That makes the number
mean something -- "this design is further from the training set than 99% of
real designs are" -- rather than being a magic constant.

TWO SPACES, AND ONLY ONE OF THEM WORKS. Measured, and this is the most
important thing in this file:

                                          val FPR   held-out category caught
    parameter space, full covariance          0.3%            100%
    feature space, Mahalanobis diagonal       3.3%              0%
    feature space, kNN k=5                    3.3%             26%
    feature space, PCA-32 + full cov          3.3%             11%
    feature space, PCA-8 reconstruction       3.3%              6%

The TABULAR path gets a real detector. The POINT-CLOUD path does not. Eight
methods were tried on the pooled PointNet features and the best caught a
quarter of genuinely out-of-distribution designs. The reason is not the
estimator: a notchback's pooled features sit comfortably inside the range of
fastback and estateback features, because the encoder was never trained to
separate body styles and max-pooled extremal features of a car are similar
across them.

So feature-space OOD here is a WEAK SIGNAL, not a gate. It is shipped with
method="knn" because 26% beats the 0% that Mahalanobis gave -- a detector that
silently does nothing is worse than one labelled weak -- but nothing should
depend on it. If the geometry model is ever served, say in the writeup that its
OOD detection is unreliable rather than implying the tabular numbers carry over.

This is also the strongest argument for serving the PARAMETRIC model behind
/evaluate, which is what contracts 2.0.0 enables: that path has a detector that
demonstrably works.

WHAT THE PARAMETER-SPACE DETECTOR CATCHES, measured rather than asserted:

    ordinary designs, body style seen in training     0.3 - 1.7% flagged
    in-bounds but jointly EXTREME parameter corners        100% flagged
    designs 2x outside PARAM_BOUNDS                        100% flagged
    held-out body style, full 19-dim distance              100% flagged
    held-out body style, aero 16 dims only                 2.6% flagged

Read the last two rows together, because they are the honest part. Designs of
an unseen body style have entirely ORDINARY aero parameters -- mean distance
4.04 against a 4.62 threshold, statistically indistinguishable from the
training set. The 100% in the row above comes from the body-style indicator
columns sitting far from anything in training, which is the same information
contracts.validate_body_style already has, more cheaply and more directly.

So this detector's UNIQUE contribution is the second row: designs that pass a
per-dimension bounds check while sitting nowhere near the training data
jointly. clamp_params accepts those -- every single coordinate is legal -- and
nothing else in the system rejects them. Sixteen parameters each at a legal
extreme is a car unlike anything ever simulated, and an optimiser pushed toward
low drag will find exactly those corners. That is the hole this closes.

The two checks are complementary and neither subsumes the other: the style
check catches an unseen baseline body, the distance catches an unreachable
combination of legal values. contracts.is_in_distribution asks both, plus
sigma, which answers the third and different question of label noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class OODDetector:
    """Mahalanobis distance to the training set, with a calibrated threshold.

    `space` records WHAT the distance is measured in:
      "parameters" -- the standardised design vector (the tabular path)
      "features"   -- the model's pooled encoder output (the point-cloud path,
                      where there is no parameter vector to compare)

    `diagonal` records whether the full covariance was estimable. With fewer
    samples than dimensions it is not -- the sample covariance is singular and
    its inverse is noise -- so we fall back to a diagonal, which keeps the
    per-dimension scaling and drops the cross-terms. Stated rather than hidden,
    because a diagonal detector is weaker and the writeup should say so.
    """

    mean: list[float]
    cov_inv: list[list[float]]
    threshold: float
    quantile: float
    n_train: int
    train_scores: dict[str, float]
    space: str = "parameters"
    diagonal: bool = False
    method: str = "mahalanobis"      # "mahalanobis" | "knn"
    k: int = 5
    reference: list[list[float]] | None = None   # knn only: the training points

    # -- fitting ----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        x_train: np.ndarray,
        x_calib: np.ndarray | None = None,
        quantile: float = 0.99,
        shrinkage: float = 1e-3,
        space: str = "parameters",
        diagonal: bool | None = None,
        method: str | None = None,
        k: int = 5,
        max_reference: int = 4000,
    ) -> "OODDetector":
        """Fit on the training inputs; calibrate the threshold on held-out ones.

        Calibrating on a split the detector was not fitted to matters: fitted
        on the training set, the distances are optimistically small, and a
        threshold read off them rejects real designs at serving time.
        """
        x_train = np.asarray(x_train, dtype=np.float64)
        if x_train.ndim != 2:
            raise ValueError(f"expected (n, p), got {x_train.shape}")
        n, p = x_train.shape
        if n < 4:
            raise ValueError(f"{n} samples is too few to fit a detector")

        # A full covariance needs comfortably more samples than dimensions;
        # below that the sample covariance is singular or near enough that its
        # inverse is noise. The point-cloud path hits this routinely -- 18
        # training meshes against a 256-dimensional feature space -- so fall
        # back to a diagonal instead of refusing to fit at all.
        if diagonal is None:
            diagonal = n < 2 * p
        if method is None:
            # In feature space a covariance -- full or diagonal -- barely
            # separates anything; measured, it caught 0% of held-out-category
            # designs. kNN distance caught 26%. Still weak, but 0% is worse
            # than weak: it is a detector that silently does nothing.
            method = "knn" if space == "features" else "mahalanobis"

        mean = x_train.mean(axis=0)
        if diagonal:
            var = x_train.var(axis=0)
            floor = shrinkage * max(float(var.mean()), 1e-12)
            cov_inv = np.diag(1.0 / np.maximum(var, floor))
        else:
            cov = np.cov(x_train, rowvar=False)
            # Ridge the diagonal: a parameter the data never varied gives a
            # singular covariance, and the inverse then blows up.
            cov = cov + shrinkage * np.eye(p) * np.trace(cov) / p
            cov_inv = np.linalg.inv(cov)

        det = cls(
            mean=mean.tolist(),
            cov_inv=cov_inv.tolist(),
            threshold=float("inf"),
            quantile=float(quantile),
            n_train=int(n),
            train_scores={},
            space=space,
            diagonal=bool(diagonal),
            method=str(method),
            k=int(k),
            reference=(x_train[
                np.linspace(0, n - 1, min(n, max_reference)).astype(int)
            ].tolist() if method == "knn" else None),
        )

        calib = x_train if x_calib is None else np.asarray(x_calib, dtype=np.float64)
        scores = det.score(calib)
        det.threshold = float(np.quantile(scores, quantile))
        det.train_scores = {
            "n_dims": int(p),
            "calibrated_on": "train" if x_calib is None else "val",
            "n_calib": int(len(calib)),
            "mean": float(scores.mean()),
            "p50": float(np.quantile(scores, 0.50)),
            "p95": float(np.quantile(scores, 0.95)),
            "p99": float(np.quantile(scores, 0.99)),
            "max": float(scores.max()),
        }
        return det

    # -- scoring ----------------------------------------------------------

    def score(self, x: np.ndarray) -> np.ndarray:
        """Distance from the training set for each row. Lower is more familiar."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if self.method == "knn":
            from scipy.spatial import cKDTree

            ref = np.asarray(self.reference, dtype=np.float64)
            k = min(int(self.k), len(ref))
            dist = cKDTree(ref).query(x, k=k)[0]
            # distance to the k-th neighbour, not the mean of the k
            return np.atleast_2d(dist)[:, -1] if k > 1 else np.ravel(dist)
        delta = x - np.asarray(self.mean)
        m = np.einsum("ij,jk,ik->i", delta, np.asarray(self.cov_inv), delta)
        return np.sqrt(np.maximum(m, 0.0))

    def score_one(self, x: np.ndarray) -> float:
        return float(self.score(x)[0])

    def is_in_distribution(self, x: np.ndarray) -> np.ndarray:
        return self.score(x) <= self.threshold

    # -- checkpoint round trip -------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "OODDetector":
        return cls(**payload)


def detection_rate(detector: OODDetector, x_ood: np.ndarray) -> float:
    """Fraction of genuinely out-of-distribution designs that get flagged.

    The headline OOD number. Reported by Paramveer's harness, not here -- this
    is the function it calls.
    """
    return float((detector.score(x_ood) > detector.threshold).mean())
