"""Stage 1 baseline model.  OWNER: Avneet.

PointNet-style regressor: point cloud in, Cd out, with a heteroscedastic head
that also predicts log-variance. That second output is what the uncertainty
gate downstream depends on -- keep it.

Deliberately small and unclever. Its job is to prove the pipeline and produce a
number to beat. The real surrogate is PhysicsNeMo's DoMINO or Transolver in
stage 2 -- do not grow this file into that.

Reference: the DrivAerNet RegDGCNN baseline is ~3M params and reports R^2 above
0.84 within a single car category.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointNetRegressor(nn.Module):
    def __init__(self, hidden_dim=256, n_layers=4, dropout=0.1, heteroscedastic=True):
        super().__init__()
        raise NotImplementedError("TODO Avneet: Oct 14")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, N, 3) -> (B, 1), or (B, 2) when heteroscedastic."""
        raise NotImplementedError


def gaussian_nll(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """NLL for the heteroscedastic head. Teaches the model to widen its own
    error bars where it cannot predict well. Clamp log-var or it runs away."""
    raise NotImplementedError("TODO Avneet")
