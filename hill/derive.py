"""Derived pharmacology quantities: IC50, AUC, Emax from predicted curves.

Nothing here is fitted; every quantity is a deterministic function of
(E_inf, s, m).  That is the point of the design: the model predicts a function
and the scalar summaries the field is used to are read off it afterwards.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


__all__ = [
    "ln_ic50",
    "ln_ic50_scalar",
    "emax",
    "auc_logspace",
    "auc_numeric",
    "viability_at",
    "derive_all",
]


def viability_at(
    log_conc: torch.Tensor, e_inf: torch.Tensor, slope: torch.Tensor, midpoint: torch.Tensor
) -> torch.Tensor:
    """r(c) at arbitrary log concentrations (broadcasting over the batch)."""
    from hill.models.curve_head import hill_curve

    return hill_curve(log_conc, e_inf, slope, midpoint)


def ln_ic50(
    e_inf: torch.Tensor, slope: torch.Tensor, midpoint: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve r(c) = 0.5 for ln IC50.

        q = (0.5 - E_inf) / (1 - E_inf)
        E_inf >= 0.5  ->  the curve never reaches 50% killing: UNDEFINED
        otherwise     ->  ln IC50 = m - logit(q) / s

    Returns
    -------
    (values, defined)
        ``values`` holds NaN wherever the IC50 does not exist.  ``defined`` is a
        boolean mask.  We never substitute a large sentinel value: a curve that
        does not cross 50% has no IC50, and that fact is a *result*, not a
        missing number to be imputed.
    """
    defined = e_inf < 0.5
    safe_e = torch.where(defined, e_inf, torch.full_like(e_inf, 0.25))
    q = (0.5 - safe_e) / (1.0 - safe_e)
    q = q.clamp(1e-12, 1.0 - 1e-12)
    logit_q = torch.log(q) - torch.log1p(-q)
    values = midpoint - logit_q / slope
    values = torch.where(defined, values, torch.full_like(values, float("nan")))
    return values, defined


def ln_ic50_scalar(e_inf: float, slope: float, midpoint: float) -> float | None:
    """Single-pair convenience wrapper. Returns ``None`` when undefined."""
    if not (e_inf < 0.5):
        return None
    q = (0.5 - e_inf) / (1.0 - e_inf)
    return float(midpoint - math.log(q / (1.0 - q)) / slope)


def emax(e_inf: torch.Tensor) -> torch.Tensor:
    """Maximal fractional kill achievable by the drug: ``1 - E_inf``."""
    return 1.0 - e_inf


def auc_logspace(
    e_inf: torch.Tensor,
    slope: torch.Tensor,
    midpoint: torch.Tensor,
    log_c_min: torch.Tensor,
    log_c_max: torch.Tensor,
    normalized: bool = True,
) -> torch.Tensor:
    """Closed-form area under r(c) over the tested log-concentration window.

        int r dx = E_inf * (x2 - x1)
                 + (1 - E_inf) / s * [softplus(-s(x1 - m)) - softplus(-s(x2 - m))]

    With ``normalized=True`` the result is the mean viability over the window
    (comparable to the GDSC published AUC: 1 = fully resistant, 0 = fully killed).
    """
    # softplus is stable for large arguments, so unlike the sigmoid in the curve
    # itself these arguments must NOT be clamped: clamping would truncate the
    # integral whenever s * (m - x) exceeds the clamp.
    width = log_c_max - log_c_min
    a1 = -slope * (log_c_min - midpoint)
    a2 = -slope * (log_c_max - midpoint)
    integral = e_inf * width + (1.0 - e_inf) * (F.softplus(a1) - F.softplus(a2)) / slope
    if normalized:
        return integral / width.clamp_min(1e-12)
    return integral


def auc_numeric(
    e_inf: torch.Tensor,
    slope: torch.Tensor,
    midpoint: torch.Tensor,
    log_c_min: torch.Tensor,
    log_c_max: torch.Tensor,
    n_grid: int = 2048,
    normalized: bool = True,
) -> torch.Tensor:
    """Trapezoidal reference implementation used to validate the closed form."""
    t = torch.linspace(0.0, 1.0, n_grid, device=e_inf.device, dtype=e_inf.dtype)  # (G,)
    xs = log_c_min.unsqueeze(-1) + (log_c_max - log_c_min).unsqueeze(-1) * t      # (..., 1, G)
    r = viability_at(xs, e_inf.unsqueeze(-1), slope.unsqueeze(-1), midpoint.unsqueeze(-1))
    integral = torch.trapezoid(r, xs, dim=-1)
    width = (log_c_max - log_c_min).clamp_min(1e-12)
    return integral / width if normalized else integral


def derive_all(
    e_inf: torch.Tensor,
    slope: torch.Tensor,
    midpoint: torch.Tensor,
    log_c_min: torch.Tensor,
    log_c_max: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """All derived quantities for a batch of predicted curves."""
    ic50, defined = ln_ic50(e_inf, slope, midpoint)
    return {
        "ln_ic50": ic50,
        "ic50_defined": defined,
        "emax": emax(e_inf),
        "e_inf": e_inf,
        "slope": slope,
        "midpoint": midpoint,
        "auc": auc_logspace(e_inf, slope, midpoint, log_c_min, log_c_max, normalized=True),
    }


def to_numpy(d: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in d.items()}
