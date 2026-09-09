"""Batched dose-response curve fitting for the Gate-0 audit.

Two nested models are fitted to the *same* normalised viability points:

    M2  (GDSC official, ``logist3``)   r(c) = sigmoid( -(x - xmid) / scal )
                                       bottom fixed at 0: every drug is assumed
                                       to kill 100% of cells at a high enough dose
    M3  (this project)                 r(c) = E_inf + (1 - E_inf) * sigmoid(-s (x - m))
                                       bottom free

All pairs are fitted simultaneously as one batched optimisation (Adam on a
padded (P, K) tensor), which makes a full-GDSC refit a minutes-long job on the
GPU instead of an hour of per-pair ``scipy`` calls.  A random subsample is
re-fitted with ``scipy.optimize.least_squares`` to prove the batched optimiser
reached the same optimum — see :func:`crosscheck_with_scipy`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from hill.constants import SIGMOID_CLAMP, SLOPE_EPS
from hill.utils.logging import get_logger

log = get_logger("audit.curvefit")


@dataclass
class FitResult:
    """Per-pair fit of one model."""

    model: str
    params: dict[str, np.ndarray]   # each (P,)
    rss: np.ndarray                 # (P,) residual sum of squares
    n_points: np.ndarray            # (P,)
    n_params: int

    @property
    def rmse(self) -> np.ndarray:
        return np.sqrt(self.rss / np.maximum(self.n_points, 1))

    def _k(self) -> int:
        return self.n_params + 1  # + sigma

    def aic(self) -> np.ndarray:
        n = np.maximum(self.n_points, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return n * np.log(np.maximum(self.rss, 1e-12) / n) + 2 * self._k()

    def bic(self) -> np.ndarray:
        n = np.maximum(self.n_points, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return n * np.log(np.maximum(self.rss, 1e-12) / n) + self._k() * np.log(n)


def _predict(model: str, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """theta: (P, n_params); x: (P, K) -> (P, K) predicted viability."""
    if model == "M2":
        xmid = theta[:, 0:1]
        scal = F.softplus(theta[:, 1:2]) + SLOPE_EPS
        arg = torch.clamp(-(x - xmid) / scal, -SIGMOID_CLAMP, SIGMOID_CLAMP)
        return torch.sigmoid(arg)
    e_inf = torch.sigmoid(theta[:, 0:1])
    slope = F.softplus(theta[:, 1:2]) + SLOPE_EPS
    m = theta[:, 2:3]
    arg = torch.clamp(-slope * (x - m), -SIGMOID_CLAMP, SIGMOID_CLAMP)
    return e_inf + (1.0 - e_inf) * torch.sigmoid(arg)


def _init_theta(
    model: str, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, restart: int, gen: torch.Generator
) -> torch.Tensor:
    """Data-driven initialisation plus jitter, so restarts explore the basin."""
    p = x.shape[0]
    device = x.device
    counts = mask.sum(1).clamp_min(1)
    x_mid = (x * mask).sum(1) / counts
    y_min = torch.where(mask, y, torch.ones_like(y)).min(dim=1).values
    jitter = torch.randn(p, generator=gen, device=device) * (0.5 * restart)
    if model == "M2":
        theta = torch.stack([x_mid + jitter, torch.zeros(p, device=device)], dim=1)
    else:
        e0 = torch.logit(y_min.clamp(0.02, 0.9))
        theta = torch.stack([e0 + 0.3 * jitter, torch.zeros(p, device=device), x_mid + jitter], dim=1)
    return theta.clone().requires_grad_(True)


def fit_curves(
    log_conc: np.ndarray,
    viability: np.ndarray,
    mask: np.ndarray,
    model: str = "M3",
    device: str | torch.device = "cpu",
    n_restarts: int = 3,
    steps: int = 800,
    lr: float = 0.2,
    seed: int = 0,
) -> FitResult:
    """Least-squares fit of ``model`` to every padded pair. Returns the best restart."""
    if model not in {"M2", "M3"}:
        raise ValueError(f"model must be M2 or M3, got {model!r}")
    dev = torch.device(device)
    x = torch.as_tensor(log_conc, dtype=torch.float32, device=dev)
    y = torch.as_tensor(viability, dtype=torch.float32, device=dev)
    m = torch.as_tensor(mask, dtype=torch.bool, device=dev)
    n_pairs = x.shape[0]
    n_params = 2 if model == "M2" else 3

    best_rss = torch.full((n_pairs,), float("inf"), device=dev)
    best_theta = torch.zeros(n_pairs, n_params, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)

    for restart in range(n_restarts):
        theta = _init_theta(model, x, y, m, restart, gen)
        opt = torch.optim.Adam([theta], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            pred = _predict(model, theta, x)
            resid = torch.where(m, pred - y, torch.zeros_like(y))
            loss = (resid**2).sum(dim=1).mean()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            pred = _predict(model, theta, x)
            rss = (torch.where(m, pred - y, torch.zeros_like(y)) ** 2).sum(dim=1)
            better = rss < best_rss
            best_rss = torch.where(better, rss, best_rss)
            best_theta = torch.where(better.unsqueeze(1), theta.detach(), best_theta)
        log.info("%s restart %d/%d: mean RSS %.5f", model, restart + 1, n_restarts, float(rss.mean()))

    with torch.no_grad():
        if model == "M2":
            params = {
                "xmid": best_theta[:, 0].cpu().numpy(),
                "scal": (F.softplus(best_theta[:, 1]) + SLOPE_EPS).cpu().numpy(),
            }
            params["slope"] = 1.0 / params["scal"]
            params["e_inf"] = np.zeros(n_pairs, dtype=np.float32)
            params["midpoint"] = params["xmid"]
        else:
            params = {
                "e_inf": torch.sigmoid(best_theta[:, 0]).cpu().numpy(),
                "slope": (F.softplus(best_theta[:, 1]) + SLOPE_EPS).cpu().numpy(),
                "midpoint": best_theta[:, 2].cpu().numpy(),
            }
    return FitResult(
        model=model,
        params=params,
        rss=best_rss.cpu().numpy(),
        n_points=m.sum(1).cpu().numpy(),
        n_params=n_params,
    )


def crosscheck_with_scipy(
    log_conc: np.ndarray,
    viability: np.ndarray,
    mask: np.ndarray,
    fit: FitResult,
    n_sample: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Refit a random subsample with ``scipy.optimize.least_squares``.

    The batched optimiser is only trustworthy if an independent solver lands on
    the same residual; this reports the discrepancy so the audit can state it.
    """
    from scipy.optimize import least_squares

    rng = np.random.default_rng(seed)
    n = log_conc.shape[0]
    idx = rng.choice(n, size=min(n_sample, n), replace=False)

    def residual_m2(p, x, y):
        xmid, log_scal = p
        scal = np.exp(log_scal)
        return 1.0 / (1.0 + np.exp(np.clip((x - xmid) / scal, -30, 30))) - y

    def residual_m3(p, x, y):
        raw_e, log_s, m = p
        e = 1.0 / (1.0 + np.exp(-raw_e))
        s = np.exp(log_s)
        return e + (1 - e) / (1.0 + np.exp(np.clip(s * (x - m), -30, 30))) - y

    residual = residual_m2 if fit.model == "M2" else residual_m3
    rel_diffs, scipy_rss, batched_rss = [], [], []
    for i in idx:
        ok = mask[i]
        if ok.sum() < 3:
            continue
        x, y = log_conc[i][ok], viability[i][ok]
        p0 = (
            np.array([float(np.mean(x)), 0.0])
            if fit.model == "M2"
            else np.array([float(np.log(max(y.min(), 0.02) / (1 - min(y.min(), 0.9)))), 0.0, float(np.mean(x))])
        )
        best = None
        for jitter in (0.0, 1.0, -1.0):
            try:
                res = least_squares(residual, p0 + jitter * 0.5, args=(x, y), max_nfev=2000)
            except Exception:  # noqa: BLE001
                continue
            val = float(np.sum(res.fun**2))
            best = val if best is None else min(best, val)
        if best is None:
            continue
        scipy_rss.append(best)
        batched_rss.append(float(fit.rss[i]))
        denom = max(best, 1e-9)
        rel_diffs.append((float(fit.rss[i]) - best) / denom)

    if not rel_diffs:
        return {"n": 0.0}
    arr = np.array(rel_diffs)
    return {
        "n": float(arr.size),
        "median_relative_rss_excess": float(np.median(arr)),
        "p90_relative_rss_excess": float(np.percentile(arr, 90)),
        "fraction_batched_worse_by_10pct": float((arr > 0.10).mean()),
        "mean_scipy_rss": float(np.mean(scipy_rss)),
        "mean_batched_rss": float(np.mean(batched_rss)),
    }


def derived_from_fit(fit: FitResult, log_c_min: np.ndarray, log_c_max: np.ndarray) -> dict[str, np.ndarray]:
    """IC50 / AUC / Emax from a fitted result, using the same code as the model."""
    from hill.derive import auc_logspace, ln_ic50

    e = torch.as_tensor(fit.params["e_inf"], dtype=torch.float64).unsqueeze(1)
    s = torch.as_tensor(fit.params["slope"], dtype=torch.float64).unsqueeze(1)
    m = torch.as_tensor(fit.params["midpoint"], dtype=torch.float64).unsqueeze(1)
    lo = torch.as_tensor(log_c_min, dtype=torch.float64).unsqueeze(1)
    hi = torch.as_tensor(log_c_max, dtype=torch.float64).unsqueeze(1)
    ic50, defined = ln_ic50(e, s, m)
    return {
        "ln_ic50": ic50.squeeze(1).numpy(),
        "ic50_defined": defined.squeeze(1).numpy(),
        "emax": (1.0 - e).squeeze(1).numpy(),
        "auc": auc_logspace(e, s, m, lo, hi, normalized=True).squeeze(1).numpy(),
    }
