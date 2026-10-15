"""Stage 2: surface field prediction.  OWNER: Avneet.

Geometry in, pressure coefficient at every surface point out. Integrate those
and you get drag and lift, which is the whole reason the plan wants fields
rather than a scalar: integration is a free consistency check, a model cannot
fake drag by cancelling errors across the surface, and the agent needs to see
WHERE the drag is to reason about shape.

ARCHITECTURE, and why this shape

Modelled on NVIDIA's DoMINO, scaled to what a 24 GB laptop can train. DoMINO
encodes the geometry as a point cloud into a global multi-scale latent space,
then extracts a LOCAL geometry representation around each point where a
solution is wanted, and decodes with DeepONet-style branch/trunk structure. Its
surface model takes cell centres, neighbouring cell centres, normals, neighbour
normals and areas, and targets the pressure coefficient.

We keep all three of those ingredients:

  BRANCH -- a global encoder over a subsampled surface cloud. Per-point shared
  MLP then pooling, with the pooled summary taken at EVERY encoder depth and
  concatenated. That is the cheap honest version of "multi-scale": early layers
  summarise local orientation statistics, late layers summarise whole-body
  shape, and Cp depends on both.

  LOCAL -- for each query point, the k nearest cloud points as RELATIVE
  offsets, encoded and max-pooled. Relative, not absolute, so the feature
  describes the neighbourhood's shape rather than where the car is. This is
  what carries curvature, which a normal alone cannot: a flat panel and a tight
  radius can share a normal and behave completely differently.

  TRUNK -- query position, query normal, local feature and global latent
  concatenated, through an MLP, to (Cp mean, Cp log-variance).

WHY PER-POINT HETEROSCEDASTICITY IS WORTH THE SECOND OUTPUT

Surface pressure is not uniformly predictable. Attached flow over the hood is
nearly deterministic; the separated wake behind the rear fascia is not. A
single sigma for the whole car would be meaningless. And we can check the
result against something external rather than taking the model's word for it:
pPrime2MeanTrim in the data is the LES's own pressure-fluctuation variance, so
a model whose predicted sigma correlates with it has learned where the physics
is genuinely unsteady. That is a calibration check no amount of self-reported
confidence can provide -- which matters, because in stage 1 sigma, deep
ensembles and feature-space distance all failed as self-reported signals.

WHAT THIS IS NOT. Not DoMINO. Not a replacement for the upstream recipe, which
we are told not to reimplement and are not: this is a deliberately small model
sized for the compute policy in CLAUDE.md, and if a real H100 run ever happens
the upstream recipe is what should run on it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LOGVAR_MIN: float = -10.0
LOGVAR_MAX: float = 4.0     # wider than stage 1: Cp varies far more than Cd


def _mlp(dims: list[int], dropout: float = 0.0, final_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        last = i == len(dims) - 2
        if not last or final_act:
            layers += [nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class FourierFeatures(nn.Module):
    """Lift coordinates into sines and cosines at octave-spaced frequencies.

    gamma(p) = [p, sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^(L-1) pi p), ...]

    An MLP fed raw 3-D coordinates has a strong low-frequency bias and
    underfits high-frequency functions (Tancik et al. 2020). Surface pressure
    is high frequency -- it changes sharply at the windscreen base, the A
    pillar, the trailing edge -- so feeding xyz directly asks the trunk to do
    something MLPs are known to be bad at. The lift is parameter-free and
    costs one concatenation.
    """

    def __init__(self, n_freq: int = 8, include_input: bool = True):
        super().__init__()
        self.n_freq = int(n_freq)
        self.include_input = bool(include_input)
        self.register_buffer(
            "_bands", torch.tensor([2.0**i for i in range(self.n_freq)]),
            persistent=False,
        )

    def out_dim(self, in_dim: int) -> int:
        return in_dim * (2 * self.n_freq + (1 if self.include_input else 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_freq == 0:
            return x
        scaled = x.unsqueeze(-1) * self._bands * torch.pi   # (..., D, F)
        feats = [scaled.sin(), scaled.cos()]
        flat = torch.cat(feats, dim=-1).flatten(start_dim=-2)
        return torch.cat([x, flat], dim=-1) if self.include_input else flat


class GlobalGeometryEncoder(nn.Module):
    """Point cloud -> one latent vector, summarised at every depth.

    Permutation invariance comes from pooling, as in PointNet. Pooling at each
    depth rather than only the last is the multi-scale part: a late-layer
    summary has abstracted away the local detail that early layers still hold,
    and surface pressure depends on both.
    """

    def __init__(self, widths: tuple[int, ...] = (64, 128, 256), latent_dim: int = 256):
        super().__init__()
        blocks: list[nn.Module] = []
        in_c = 3
        for w in widths:
            blocks.append(nn.Sequential(
                nn.Conv1d(in_c, w, kernel_size=1, bias=False),
                nn.BatchNorm1d(w),
                nn.GELU(),
            ))
            in_c = w
        self.blocks = nn.ModuleList(blocks)
        # max and mean at every depth
        pooled_dim = 2 * sum(widths)
        self.project = _mlp([pooled_dim, latent_dim, latent_dim], final_act=False)
        self.latent_dim = latent_dim

    def forward(self, cloud: torch.Tensor) -> torch.Tensor:
        """cloud: (B, G, 3) -> (B, latent_dim)"""
        x = cloud.transpose(1, 2)                      # (B, 3, G)
        summaries: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            summaries += [x.amax(dim=2), x.mean(dim=2)]
        return self.project(torch.cat(summaries, dim=1))


class LocalNeighbourhoodEncoder(nn.Module):
    """Nearest cloud points as relative offsets, pooled per query point, at
    SEVERAL neighbourhood sizes.

    One k gives one spatial scale. Curvature at a trailing edge and curvature
    over a whole panel are different quantities, and DoMINO's central idea is
    multi-scale geometry representation, so several k are pooled separately and
    concatenated. The k-nearest distances are computed ONCE and sliced, so
    extra scales cost almost nothing beyond the encoder they each need.
    """

    def __init__(self, k: int | tuple[int, ...] = (8, 32, 128),
                 hidden: int = 64, out_dim: int = 64):
        super().__init__()
        ks = (int(k),) if isinstance(k, int) else tuple(int(x) for x in k)
        self.ks = tuple(sorted(ks))
        per_scale = max(16, out_dim // len(self.ks))
        # 3 offset + 1 distance: the distance is redundant in principle and
        # helpful in practice, since the net would otherwise have to learn a
        # norm before it can use scale.
        self.encoders = nn.ModuleList(
            [_mlp([4, hidden, per_scale]) for _ in self.ks]
        )
        self.out_dim = per_scale * len(self.ks)

    def forward(self, query: torch.Tensor, cloud: torch.Tensor) -> torch.Tensor:
        """query: (B, Q, 3), cloud: (B, G, 3) -> (B, Q, out_dim)"""
        g = cloud.shape[1]
        k_max = min(max(self.ks), g)
        dist = torch.cdist(query, cloud)                        # (B, Q, G)
        idx = dist.topk(k_max, dim=2, largest=False).indices    # (B, Q, k_max)

        b, q, _ = query.shape
        gathered = torch.gather(
            cloud.unsqueeze(1).expand(b, q, g, 3), 2,
            idx.unsqueeze(-1).expand(b, q, k_max, 3),
        )                                                       # (B, Q, k_max, 3)
        offset = gathered - query.unsqueeze(2)                  # relative
        radius = offset.norm(dim=-1, keepdim=True)
        packed = torch.cat([offset, radius], dim=-1)

        out = []
        for k, encoder in zip(self.ks, self.encoders):
            kk = min(k, k_max)
            out.append(encoder(packed[:, :, :kk]).amax(dim=2))
        return torch.cat(out, dim=-1)


class SurfaceFieldModel(nn.Module):
    """Geometry + query points -> Cp (and log-variance) at each query point."""

    def __init__(self, latent_dim: int = 256,
                 local_k: int | tuple[int, ...] = (8, 32, 128),
                 local_dim: int = 96, hidden: int = 256, n_layers: int = 4,
                 dropout: float = 0.0, heteroscedastic: bool = True,
                 n_targets: int = 1, n_freq: int = 8):
        super().__init__()
        self.heteroscedastic = bool(heteroscedastic)
        self.n_targets = int(n_targets)
        self.out_dim = self.n_targets * (2 if self.heteroscedastic else 1)

        self.global_encoder = GlobalGeometryEncoder(latent_dim=latent_dim)
        self.local_encoder = LocalNeighbourhoodEncoder(k=local_k, out_dim=local_dim)
        self.fourier = FourierFeatures(n_freq=n_freq)

        # lifted query xyz + query normal (3) + local + global
        in_dim = (self.fourier.out_dim(3) + 3
                  + self.local_encoder.out_dim + latent_dim)
        dims = [in_dim] + [hidden] * int(n_layers)
        self.trunk = _mlp(dims, dropout=dropout)
        # Separate heads, not one Linear emitting both. Stage 1 learned this
        # the hard way: the mean and the variance want different stopping
        # epochs, and trained jointly the NLL weights each point by its own
        # confidence, so sigma collapses and the model goes confidently wrong
        # (coverage 0.24 against a target of 0.68). The fix was to fit the mean
        # first and then the variance against a FROZEN mean, which needs the
        # heads to be separable. Half a Linear cannot be frozen.
        self.mean_head = nn.Linear(hidden, self.n_targets)
        self.logvar_head = (nn.Linear(hidden, self.n_targets)
                            if self.heteroscedastic else None)
        self._mean_path_frozen = False

    def forward(self, cloud: torch.Tensor, query_xyz: torch.Tensor,
                query_normal: torch.Tensor) -> torch.Tensor:
        """-> (B, Q, out_dim). Columns are [mean..., logvar...] per target."""
        if cloud.dim() != 3 or query_xyz.dim() != 3:
            raise ValueError(
                f"expected (B,G,3) and (B,Q,3), got {tuple(cloud.shape)} "
                f"and {tuple(query_xyz.shape)}"
            )
        glob = self.global_encoder(cloud)                        # (B, L)
        local = self.local_encoder(query_xyz, cloud)             # (B, Q, Lc)
        glob = glob.unsqueeze(1).expand(-1, query_xyz.shape[1], -1)
        h = torch.cat([self.fourier(query_xyz), query_normal, local, glob], dim=-1)
        feat = self.trunk(h)
        mean = self.mean_head(feat)
        if self.logvar_head is None:
            return mean
        return torch.cat([mean, self.logvar_head(feat)], dim=-1)

    # -- the two-stage hook, as in src/train/model.py ----------------------

    def freeze_mean_path(self) -> list[nn.Parameter]:
        """Freeze everything but logvar_head; return what is still trainable.

        BatchNorm in the global encoder keeps updating its running statistics
        from forward passes regardless of requires_grad, so the frozen modules
        are held in eval mode too -- otherwise the "frozen" mean still drifts
        and the accuracy we report is not the accuracy we ship.
        """
        if self.logvar_head is None:
            raise RuntimeError("no variance head (heteroscedastic=False)")
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith("logvar_head")
        self._mean_path_frozen = True
        self.train(self.training)
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode: bool = True):  # noqa: D102 - nn.Module override
        super().train(mode)
        if self._mean_path_frozen:
            for name, module in self.named_children():
                if name != "logvar_head":
                    module.eval()
        return self

    def split(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, Q, out_dim) -> (mean, sigma), each (B, Q, n_targets)."""
        n = self.n_targets
        mean = out[..., :n]
        if not self.heteroscedastic:
            return mean, torch.zeros_like(mean)
        logvar = out[..., n:].clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mean, torch.exp(0.5 * logvar)


def field_nll(out: torch.Tensor, target: torch.Tensor, model: SurfaceFieldModel,
              beta: float = 0.0, mean_only: bool = False) -> torch.Tensor:
    """Per-point Gaussian NLL, averaged over points and targets.

    Same variance-collapse defences as stage 1 (src/train/model.py): beta-NLL
    reweighting and a mean-only warmup. The failure mode is identical here and
    worse in consequence, because a collapsed per-point sigma would make the
    whole field look confidently wrong rather than one number.
    """
    n = model.n_targets
    mean = out[..., :n]
    if not model.heteroscedastic:
        return nn.functional.mse_loss(mean, target)
    if mean_only:
        return nn.functional.mse_loss(mean, target)

    logvar = out[..., n:].clamp(LOGVAR_MIN, LOGVAR_MAX)
    per_point = 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))
    if beta > 0.0:
        per_point = per_point * torch.exp(logvar).detach() ** beta
    return per_point.mean()


def integral_consistency_loss(
    out: torch.Tensor, target: torch.Tensor, normal: torch.Tensor,
    area: torch.Tensor, a_ref: torch.Tensor, scale: torch.Tensor,
    model: "SurfaceFieldModel", axes: tuple[int, ...] = (0, 2),
) -> torch.Tensor:
    """Squared error between the drag (and lift) INTEGRATED from the predicted
    field and from the true field, on the same points.

    WHY THIS TERM EXISTS, measured

    A per-point loss and the integral disagree about what a good field is, and
    the disagreement is systematic rather than noise. Across an ablation:

                        Cp R^2    integrated Cd error
        baseline        0.3489          -0.0260
        multi-scale     0.3912          -0.0012
        fourier         0.5100          -0.0705
        both            0.5173          -0.0638

    Fourier features buy ~0.16 of Cp R^2 and cost ~0.05 of Cd error, in both
    pairings. The reason is structural: the integral is a SIGNED area-weighted
    sum, so errors that cancel pointwise do not cancel in it, and a loss that
    only sees per-point error has no reason to care. Pointwise accuracy and
    integral accuracy are genuinely different objectives.

    Rather than choose, put the integral in the loss. It is differentiable, it
    costs one extra reduction, and it optimises the quantity the agent actually
    uses -- drag -- instead of hoping a pointwise fit implies it.

    This is a deliberately narrow version of the plan's physics-informed loss
    (cut-list item 1): not a PDE residual, just the one conservation statement
    that connects our output to our objective. If it ever has to be cut, set
    train.integral_weight to 0 and nothing else changes.
    """
    n = model.n_targets
    pred_cp = out[..., :n]
    total = out.new_zeros(())
    for axis in axes:
        pred = integrate_drag(pred_cp, normal, area, a_ref) * scale
        true = integrate_drag(target, normal, area, a_ref) * scale
        total = total + ((pred - true) ** 2).mean()
    return total / len(axes)


def integrate_drag(cp: torch.Tensor, normal: torch.Tensor, area: torch.Tensor,
                   a_ref: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Pressure-only Cd (axis 0) or Cl (axis 2) from a predicted Cp field.

    Cd_pressure = sum(Cp * -n_axis * dA) / A_ref

    Partial by design: this is a SUBSAMPLE of the surface, so it recovers the
    coefficient only up to the sampling fraction, and it omits the viscous
    term. Its job is to be differentiable and comparable between predicted and
    true Cp on the SAME points -- which makes it a consistency check on the
    field, not a drag measurement. The real integration, over every cell and
    including shear, is in src/data/preprocess.py and src/data/fields.py.
    """
    weighted = cp.squeeze(-1) * -normal[..., axis] * area
    return weighted.sum(dim=-1) / a_ref
