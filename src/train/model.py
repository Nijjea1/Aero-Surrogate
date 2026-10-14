"""Stage 1 baseline model.  OWNER: Avneet.

PointNet-style regressor: point cloud in, Cd out, with a heteroscedastic head
that also predicts log-variance. That second output is what the uncertainty
gate downstream depends on -- keep it.

Deliberately small and unclever. Its job is to prove the pipeline and produce a
number to beat. The real surrogate is PhysicsNeMo's DoMINO or Transolver in
stage 2 -- do not grow this file into that.

Reference: the DrivAerNet RegDGCNN baseline is ~3M params and reports R^2 above
0.84 within a single car category.

ParamMLPRegressor is the second, smaller head of the same stage: design
parameters straight to Cd, no geometry. It exists as insurance on the Oct 15
gate (it trains in seconds and needs no meshes) and because it takes exactly
the input the agent proposes. Same checkpoint format, same uncertainty head.

Both models share one shape: trunk -> (mean_head, logvar_head). The two heads
are separate modules, not two columns of one layer, so the trainer can fit the
mean first and the variance afterwards against a frozen mean. See
freeze_mean_path() below and the two-stage loop in train.py -- that split is
what makes the uncertainty calibrated instead of merely present.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The Gaussian NLL is unbounded below as the variance goes to zero, so an
# unclamped log-variance head runs away in the first few hundred steps: it
# drives sigma down on the easy examples and the loss follows it to -inf.
# These bounds are wide enough not to bind in normal training (sigma from
# ~0.007 to ~2.7 in normalised target units) and tight enough to stop that.
LOGVAR_MIN: float = -10.0
LOGVAR_MAX: float = 2.0


class _HeteroscedasticRegressor(nn.Module):
    """Shared plumbing: a trunk, two heads, and the freeze that decouples them."""

    def __init__(self) -> None:
        super().__init__()
        self._mean_path_frozen = False

    def _build_heads(self, feat_dim: int, hidden_dim: int, dropout: float,
                     heteroscedastic: bool) -> None:
        self.heteroscedastic = bool(heteroscedastic)
        self.out_dim = 2 if self.heteroscedastic else 1
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1) if self.heteroscedastic else None

    def _emit(self, feats: torch.Tensor) -> torch.Tensor:
        h = self.trunk(feats)
        mean = self.mean_head(h)
        if self.logvar_head is None:
            return mean
        return torch.cat([mean, self.logvar_head(h)], dim=1)

    # -- the two-stage hook -------------------------------------------------

    def freeze_mean_path(self) -> list[nn.Parameter]:
        """Freeze everything except logvar_head. Returns what is still trainable.

        Phase two of training fits the variance against a mean that cannot
        move. That is the whole point: a fixed mean has fixed residuals, so
        maximising the likelihood over sigma alone is a well-posed problem with
        one answer, instead of a tug of war where the model can lower the loss
        by getting quietly overconfident.

        BatchNorm and Dropout in the frozen part are held in eval mode too --
        otherwise the running statistics keep drifting and the "frozen" mean
        is not actually frozen.
        """
        if self.logvar_head is None:
            raise RuntimeError("no variance head to train (heteroscedastic=False)")
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith("logvar_head")
        self._mean_path_frozen = True
        self.train(self.training)  # re-apply the eval-mode override below
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode: bool = True):  # noqa: D102 - nn.Module override
        super().train(mode)
        if self._mean_path_frozen:
            for name, module in self.named_children():
                if name != "logvar_head":
                    module.eval()
        return self


def _encoder_channels(hidden_dim: int, n_layers: int) -> list[int]:
    """Channel ramp for the per-point encoder, ending at hidden_dim."""
    return [max(64, hidden_dim // 2 ** (n_layers - 1 - i)) for i in range(n_layers)]


class PointNetRegressor(_HeteroscedasticRegressor):
    def __init__(self, hidden_dim=256, n_layers=4, dropout=0.1, heteroscedastic=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        # Shared per-point MLP, implemented as 1x1 convolutions over the point
        # axis so every point is treated identically (that is the permutation
        # invariance PointNet relies on).
        channels = _encoder_channels(self.hidden_dim, int(n_layers))
        layers: list[nn.Module] = []
        in_c = 3
        for out_c in channels:
            layers += [
                nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
            ]
            in_c = out_c
        self.encoder = nn.Sequential(*layers)

        # Max pool is the symmetric function; mean pool is nearly free and
        # carries bulk information (how much surface sits where) that max
        # throws away. Concatenating both measurably helps on drag.
        #
        # Size the head off the encoder's ACTUAL final width, not hidden_dim:
        # _encoder_channels has a floor of 64, so the two differ whenever
        # hidden_dim < 64 and the head would be built at the wrong width.
        self._build_heads(2 * channels[-1], self.hidden_dim, dropout, heteroscedastic)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, N, 3) -> (B, 1), or (B, 2) when heteroscedastic."""
        if points.dim() != 3 or points.shape[-1] != 3:
            raise ValueError(f"expected (B, N, 3), got {tuple(points.shape)}")

        x = self.encoder(points.transpose(1, 2))  # (B, C, N)
        pooled = torch.cat([x.amax(dim=2), x.mean(dim=2)], dim=1)
        return self._emit(pooled)


class ParamMLPRegressor(_HeteroscedasticRegressor):
    """Design parameters -> Cd. No geometry, no meshes, trains in seconds.

    Gate insurance, and the input signature the agent actually proposes in.

    quadratic_features expands the input with every x_i * x_j. It is OFF by
    default because it was measured and it does not work: across a 72-config
    sweep on the synthetic fixture, every one of the ten best settings had it
    off, and switching it on cost roughly 0.03 test R^2. Eight parameters
    become forty-four, which at a couple of hundred designs buys overfitting
    rather than inductive bias. The flag stays so nobody spends an afternoon
    rediscovering this.

    Do not read the fixture's quadratic-ridge score (test R^2 0.96 against this
    model's 0.87) as headroom. analytic_cd in src/data/synthetic.py is built
    out of squared terms, so a quadratic ridge is the exactly-correct model
    class for it -- an oracle, not a baseline. The comparison is only
    meaningful on real labels.
    """

    def __init__(self, in_dim, hidden_dim=128, n_layers=2, dropout=0.0,
                 heteroscedastic=True, quadratic_features=False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.quadratic_features = bool(quadratic_features)

        feat_dim = self.in_dim
        if self.quadratic_features:
            # x, then every x_i * x_j for i <= j
            self.register_buffer(
                "_pair_i",
                torch.tensor([i for i in range(self.in_dim) for _ in range(i, self.in_dim)]),
                persistent=False,
            )
            self.register_buffer(
                "_pair_j",
                torch.tensor([j for i in range(self.in_dim) for j in range(i, self.in_dim)]),
                persistent=False,
            )
            feat_dim += self.in_dim * (self.in_dim + 1) // 2

        layers: list[nn.Module] = []
        in_c = feat_dim
        for _ in range(int(n_layers)):
            layers += [
                nn.Linear(in_c, int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ]
            in_c = int(hidden_dim)
        self.encoder = nn.Sequential(*layers)

        self._build_heads(int(hidden_dim), int(hidden_dim), dropout, heteroscedastic)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """params: (B, P) -> (B, 1), or (B, 2) when heteroscedastic."""
        if params.dim() != 2 or params.shape[-1] != self.in_dim:
            raise ValueError(f"expected (B, {self.in_dim}), got {tuple(params.shape)}")

        x = params
        if self.quadratic_features:
            pairs = params[:, self._pair_i] * params[:, self._pair_j]
            x = torch.cat([params, pairs], dim=1)
        return self._emit(self.encoder(x))


def split_prediction(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 1 or 2) -> (mean, sigma), both (B,). sigma is 0 without the head.

    The single place log-variance becomes a standard deviation. Everything --
    the trainer, the metrics, the service -- goes through here so the
    conversion cannot drift between training and inference.
    """
    mean = pred[:, 0]
    if pred.shape[1] == 1:
        return mean, torch.zeros_like(mean)
    logvar = pred[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX)
    return mean, torch.exp(0.5 * logvar)


def gaussian_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.0,
    mean_only: bool = False,
) -> torch.Tensor:
    """NLL for the heteroscedastic head. Teaches the model to widen its own
    error bars where it cannot predict well. Clamp log-var or it runs away.

    Clamping stops log-variance running to -inf, but it does NOT stop the other
    failure: the plain NLL weights each sample by 1/sigma^2, so the model learns
    fastest on the points it is already confident about, shrinks sigma there,
    and ends up confidently wrong. On the synthetic fixture that showed up as
    coverage falling from 0.53 to 0.24 while R^2 held at 0.89 -- a model that
    predicts well and reports nonsense error bars, which is worse than useless
    for a gate the agent is supposed to trust.

    Two knobs, both set from the config:

    beta       beta-NLL (Seitzer et al. 2022): scale each sample's loss by a
               DETACHED sigma^(2*beta). beta=0 is the plain NLL, beta=1 makes
               the mean's gradient exactly MSE, beta=0.5 is the recommended
               middle. Only matters when mean and variance train together.
    mean_only  pure MSE on the mean. Phase one of two-stage training, and the
               reason beta matters less than it looks: with the phases split,
               the mean never sees the NLL at all.
    """
    target = target.reshape(-1)
    if pred.shape[1] == 1:  # no uncertainty head: plain MSE
        return nn.functional.mse_loss(pred[:, 0], target)

    mean = pred[:, 0]
    if mean_only:
        return nn.functional.mse_loss(mean, target)

    logvar = pred[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX)
    # 0.5 * (logvar + (y - mu)^2 / var); the log(2*pi) term is constant and
    # dropped, so this value is comparable across runs but is not an absolute
    # log-likelihood.
    per_sample = 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))
    if beta > 0.0:
        per_sample = per_sample * torch.exp(logvar).detach() ** beta
    return per_sample.mean()


def build_model(cfg: dict) -> nn.Module:
    """Construct the model the config asks for. 'pointnet' is the default."""
    m = dict(cfg.get("model", {}))
    kind = str(m.pop("kind", "pointnet")).lower()

    if kind == "pointnet":
        return PointNetRegressor(
            hidden_dim=m.get("hidden_dim", 256),
            n_layers=m.get("n_layers", 4),
            dropout=m.get("dropout", 0.1),
            heteroscedastic=m.get("heteroscedastic", True),
        )
    if kind in ("param_mlp", "parammlp", "tabular"):
        in_dim = m.get("in_dim")
        if in_dim is None:
            from src.contracts import PARAM_NAMES

            in_dim = len(PARAM_NAMES)
        return ParamMLPRegressor(
            in_dim=in_dim,
            hidden_dim=m.get("hidden_dim", 128),
            n_layers=m.get("n_layers", 2),
            dropout=m.get("dropout", 0.0),
            heteroscedastic=m.get("heteroscedastic", True),
            quadratic_features=m.get("quadratic_features", False),
        )
    raise ValueError(f"unknown model.kind {kind!r} (pointnet | param_mlp)")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
