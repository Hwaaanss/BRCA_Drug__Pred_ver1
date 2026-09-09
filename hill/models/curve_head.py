"""The Hill curve head — the core of this project.

For every (sample i, drug j) pair the model emits three constrained numbers

    E_inf = sigmoid(raw_e)          in (0, 1)  residual viability at infinite dose
    s     = softplus(raw_s) + eps   > 0        Hill slope
    m     = raw_m                   in R       log-midpoint (potency)

and the predicted dose-response curve is

    r_ij(c) = E_inf + (1 - E_inf) * sigmoid(-s * (log c - m))

Structural properties (locked down by ``tests/test_curve.py``):
    c -> 0    =>  r -> 1        (full survival at zero dose)
    c -> inf  =>  r -> E_inf    (efficacy ceiling)
    s > 0     =>  r is monotonically decreasing in c -- by construction,
                  not something the optimiser has to learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hill.constants import E_INF_EPS, SIGMOID_CLAMP, SLOPE_EPS


def hill_curve(
    log_conc: torch.Tensor,   # (..., K) natural log of concentration in uM
    e_inf: torch.Tensor,      # (..., 1) or broadcastable
    slope: torch.Tensor,      # (..., 1)
    midpoint: torch.Tensor,   # (..., 1)
    clamp: float = SIGMOID_CLAMP,
) -> torch.Tensor:
    """Evaluate the three-parameter Hill curve. Returns viability in (E_inf, 1)."""
    arg = torch.clamp(-slope * (log_conc - midpoint), -clamp, clamp)  # (..., K)
    return e_inf + (1.0 - e_inf) * torch.sigmoid(arg)


def constrain_parameters(
    raw_e: torch.Tensor,
    raw_s: torch.Tensor,
    raw_m: torch.Tensor,
    slope_eps: float = SLOPE_EPS,
    e_inf_eps: float = E_INF_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map unconstrained head outputs onto the admissible parameter space."""
    e_inf = torch.sigmoid(raw_e).clamp(e_inf_eps, 1.0 - e_inf_eps)
    slope = F.softplus(raw_s) + slope_eps
    return e_inf, slope, raw_m


@dataclass
class CurveParams:
    """Container for a batch of curve parameters, all shaped (B, 1)."""

    e_inf: torch.Tensor
    slope: torch.Tensor
    midpoint: torch.Tensor

    def viability(self, log_conc: torch.Tensor) -> torch.Tensor:
        """(B, K) predicted viability at (B, K) log concentrations."""
        return hill_curve(log_conc, self.e_inf, self.slope, self.midpoint)

    def detach(self) -> "CurveParams":
        return CurveParams(self.e_inf.detach(), self.slope.detach(), self.midpoint.detach())

    def to(self, *args, **kwargs) -> "CurveParams":
        return CurveParams(
            self.e_inf.to(*args, **kwargs),
            self.slope.to(*args, **kwargs),
            self.midpoint.to(*args, **kwargs),
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {"e_inf": self.e_inf, "slope": self.slope, "midpoint": self.midpoint}


class CurveHead(nn.Module):
    """z_ij -> (E_inf, s, m).

    The three linear maps are deliberately separate: potency (m) and efficacy
    (E_inf) must be able to move independently, which is the whole point of the
    design.  The histology gate in :class:`hill.models.hill.HILL` adds to
    ``raw_e`` and ``raw_m`` *before* the constraint transform.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        init_e_logit: float = -2.0,
        init_slope_raw: float = 0.5,
        slope_eps: float = SLOPE_EPS,
    ) -> None:
        super().__init__()
        self.slope_eps = slope_eps
        self.dropout = nn.Dropout(dropout)
        self.base_e = nn.Linear(d_model, 1)
        self.base_s = nn.Linear(d_model, 1)
        self.base_m = nn.Linear(d_model, 1)
        # Start every pair near "modest efficacy, unit slope, midpoint at the
        # centre of the tested range" so the first epochs are well conditioned.
        for layer, bias in ((self.base_e, init_e_logit), (self.base_s, init_slope_raw), (self.base_m, 0.0)):
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, bias)
        # Small random weights: zero-weight init would make every pair identical
        # and kill the gradient signal through the encoder.
        for layer in (self.base_e, self.base_s, self.base_m):
            nn.init.normal_(layer.weight, std=0.01)

    def raw(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, d) -> three (B, 1) unconstrained parameters."""
        h = self.dropout(z)
        return self.base_e(h), self.base_s(h), self.base_m(h)

    def forward(self, z: torch.Tensor) -> CurveParams:
        raw_e, raw_s, raw_m = self.raw(z)
        e_inf, slope, midpoint = constrain_parameters(raw_e, raw_s, raw_m, self.slope_eps)
        return CurveParams(e_inf=e_inf, slope=slope, midpoint=midpoint)


class ScalarHead(nn.Module):
    """Baseline head for ``ScalarHILL``: z_ij -> ln IC50 (a single number).

    Same encoder, same data, same splits — only the head and the loss change.
    The HILL-vs-ScalarHILL gap is the paper's central claim, so this must stay
    architecturally minimal.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, d) -> (B, 1) predicted ln IC50."""
        return self.out(self.dropout(z))
