"""Statistics helpers: bootstrap CIs, paired tests, FDR, likelihood-ratio tests."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with NaN-safe pairwise deletion; NaN if < 3 usable points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    xs, ys = x[ok], y[ok]
    if np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    r = stats.spearmanr(x[ok], y[ok]).statistic
    return float(r)


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap: returns (point estimate, low, high)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(arr))
    if arr.size == 1:
        return point, point, point
    if n_boot < 1:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.array([statistic(arr[i]) for i in idx], dtype=float)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_test(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Paired comparison of two methods over matched units (drugs / folds).

    Reports both the Wilcoxon signed-rank (primary, distribution-free) and the
    paired t-test, plus the effect size.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    ok = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_arr, b_arr = a_arr[ok], b_arr[ok]
    out = {"n": float(a_arr.size), "mean_diff": float("nan"),
           "wilcoxon_p": float("nan"), "ttest_p": float("nan"), "cohens_d": float("nan")}
    if a_arr.size < 2:
        return out
    diff = a_arr - b_arr
    out["mean_diff"] = float(diff.mean())
    sd = diff.std(ddof=1)
    out["cohens_d"] = float(diff.mean() / sd) if sd > 0 else float("nan")
    if np.allclose(diff, 0):
        out["wilcoxon_p"] = 1.0
        out["ttest_p"] = 1.0
        return out
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a_arr, b_arr).pvalue)
    except ValueError:
        pass
    out["ttest_p"] = float(stats.ttest_rel(a_arr, b_arr).pvalue)
    return out


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Return BH-adjusted q-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return q
    sub = p[ok]
    order = np.argsort(sub)
    ranked = sub[order]
    n = ranked.size
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def lrt_boundary_pvalue(loglik_full: float, loglik_restricted: float, df: int = 1) -> dict[str, float]:
    """Likelihood-ratio test for a single parameter released from zero.

    Reports the boundary-corrected p-value (0.5*chi2_0 + 0.5*chi2_1, appropriate
    when the null sits on the edge of the parameter space) *and* the standard
    chi2_df p-value, because gamma is free to take either sign in this model.
    """
    stat = 2.0 * (loglik_full - loglik_restricted)
    stat = float(max(stat, 0.0))
    p_chi2 = float(stats.chi2.sf(stat, df)) if stat > 0 else 1.0
    p_boundary = 0.5 * p_chi2 if stat > 0 else 1.0
    return {"lambda": stat, "p_boundary": p_boundary, "p_chi2": p_chi2, "df": float(df)}


def variance_decomposition(matrix: np.ndarray) -> dict[str, float]:
    """Two-way (row x column) variance decomposition of a label matrix.

    ``matrix`` is samples x drugs with NaN for missing entries.  Returns the
    fraction of total variance explained by the drug (column) main effect,
    the sample (row) main effect, and the residual (interaction + noise).
    """
    m = np.asarray(matrix, dtype=float)
    ok = np.isfinite(m)
    if ok.sum() < 4:
        return {"f_drug": float("nan"), "f_sample": float("nan"), "f_residual": float("nan"),
                "var_total": float("nan"), "n_obs": float(ok.sum())}
    grand = np.nanmean(m)
    col_eff = np.nanmean(m, axis=0) - grand          # drug main effect
    row_eff = np.nanmean(m, axis=1) - grand          # sample main effect
    col_eff = np.nan_to_num(col_eff)
    row_eff = np.nan_to_num(row_eff)
    fitted = grand + row_eff[:, None] + col_eff[None, :]
    resid = m - fitted
    var_total = float(np.nanvar(m))
    if var_total == 0:
        return {"f_drug": float("nan"), "f_sample": float("nan"), "f_residual": float("nan"),
                "var_total": 0.0, "n_obs": float(ok.sum())}
    ss_total = float(np.nansum((m - grand) ** 2))
    ss_col = float(np.nansum((np.broadcast_to(col_eff[None, :], m.shape) * ok) ** 2))
    ss_row = float(np.nansum((np.broadcast_to(row_eff[:, None], m.shape) * ok) ** 2))
    ss_res = float(np.nansum(resid[ok] ** 2))
    return {
        "f_drug": ss_col / ss_total,
        "f_sample": ss_row / ss_total,
        "f_residual": ss_res / ss_total,
        "var_total": var_total,
        "n_obs": float(ok.sum()),
    }
