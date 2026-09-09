"""G-3 — the decisive experiment: two-parameter vs three-parameter refit.

No deep learning is involved.  The same normalised viability points are fitted
with the GDSC official bottom-fixed model (M2) and with a free-bottom model
(M3), and the two are compared on residuals, information criteria and the
distribution of the fitted efficacy ceiling ``E_inf``.

What the outcome means
    * E_inf concentrated near 0  -> M2 was adequate and this project's premise
      is weak.  Report it and stop.
    * E_inf spread out, M3 preferred by BIC on a large share of pairs -> the
      premise holds: efficacy variation exists and the official fit cannot
      express it.
    * Overlap between "high E_inf" and "censored" pairs quantifies the claim
      that the censoring problem and the misspecification problem are one problem.

    python -m hill.audit.refit --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from hill.audit.base import audit_dir, is_synthetic, load_tables, save_json, summarise
from hill.audit.curvefit import crosscheck_with_scipy, derived_from_fit, fit_curves
from hill.config import load_config
from hill.utils.logging import get_logger
from hill.utils.stats import pearson, spearman

log = get_logger("audit.g3")

HIGH_EINF = 0.5  # at/above this the curve never crosses 50% -> IC50 undefined


def run(cfg: Any, device: str | None = None, max_pairs: int | None = None,
        steps: int = 800, n_restarts: int = 3) -> dict[str, Any]:
    _, pairs, padded = load_tables(cfg)
    device = device or ("cuda" if cfg.device == "cuda" else "cpu")

    idx = np.arange(len(pairs))
    if max_pairs and idx.size > max_pairs:
        idx = np.random.default_rng(0).choice(idx, max_pairs, replace=False)
        idx.sort()
    x, y, m = padded["log_conc"][idx], padded["viability"][idx], padded["mask"][idx]

    fit2 = fit_curves(x, y, m, "M2", device=device, steps=steps, n_restarts=n_restarts)
    fit3 = fit_curves(x, y, m, "M3", device=device, steps=steps, n_restarts=n_restarts)

    lo = np.where(m, x, np.inf).min(axis=1)
    hi = np.where(m, x, -np.inf).max(axis=1)
    d2 = derived_from_fit(fit2, lo, hi)
    d3 = derived_from_fit(fit3, lo, hi)

    delta_rmse = fit2.rmse - fit3.rmse            # > 0 means M3 fits better
    aic2, aic3 = fit2.aic(), fit3.aic()
    bic2, bic3 = fit2.bic(), fit3.bic()
    m3_wins_aic = aic3 < aic2
    m3_wins_bic = bic3 < bic2

    sub = pairs.iloc[idx].reset_index(drop=True)
    censored = sub["censored"].fillna(False).to_numpy(dtype=bool) if "censored" in sub else np.zeros(idx.size, bool)
    high_e = fit3.params["e_inf"] >= HIGH_EINF

    # Wilcoxon over pairs: is the per-pair residual improvement systematic?
    finite = np.isfinite(delta_rmse)
    try:
        w = stats.wilcoxon(fit3.rmse[finite], fit2.rmse[finite], alternative="less")
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    # per-drug IC50 vs Emax correlation under the free-bottom fit
    per_drug_rows = []
    for drug, grp in sub.assign(
        _ic50=d3["ln_ic50"], _emax=d3["emax"], _einf=fit3.params["e_inf"],
        _drmse=delta_rmse, _bicwin=m3_wins_bic,
    ).groupby("drug_id"):
        per_drug_rows.append(
            {
                "drug_id": int(drug),
                "n_pairs": int(len(grp)),
                "mean_e_inf": float(grp["_einf"].mean()),
                "frac_e_inf_ge_0.5": float((grp["_einf"] >= HIGH_EINF).mean()),
                "mean_delta_rmse": float(grp["_drmse"].mean()),
                "frac_m3_preferred_bic": float(grp["_bicwin"].mean()),
                "pcc_ic50_vs_emax": pearson(grp["_ic50"].to_numpy(), grp["_emax"].to_numpy()),
                "scc_ic50_vs_emax": spearman(grp["_ic50"].to_numpy(), grp["_emax"].to_numpy()),
            }
        )
    per_drug = pd.DataFrame(per_drug_rows)

    overlap = {
        "n_censored": int(censored.sum()),
        "n_high_einf": int(high_e.sum()),
        "n_both": int((censored & high_e).sum()),
        "jaccard": float((censored & high_e).sum() / max((censored | high_e).sum(), 1)),
        "p_high_einf_given_censored": float(high_e[censored].mean()) if censored.any() else float("nan"),
        "p_high_einf_given_uncensored": float(high_e[~censored].mean()) if (~censored).any() else float("nan"),
    }
    if censored.any() and (~censored).any():
        table = np.array(
            [[int((censored & high_e).sum()), int((censored & ~high_e).sum())],
             [int((~censored & high_e).sum()), int((~censored & ~high_e).sum())]]
        )
        try:
            odds, p = stats.fisher_exact(table)
            overlap["fisher_odds_ratio"] = float(odds)
            overlap["fisher_p"] = float(p)
        except ValueError:
            pass

    result: dict[str, Any] = {
        "gate": "G-3",
        "synthetic": is_synthetic(cfg),
        "n_pairs_audited": int(idx.size),
        "rmse_M2": summarise(fit2.rmse),
        "rmse_M3": summarise(fit3.rmse),
        "delta_rmse_M2_minus_M3": summarise(delta_rmse),
        "fraction_M3_better_rmse": float((delta_rmse > 0).mean()),
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_M3_better": w_p,
        "fraction_M3_preferred_AIC": float(m3_wins_aic.mean()),
        "fraction_M3_preferred_BIC": float(m3_wins_bic.mean()),
        "median_delta_BIC_M2_minus_M3": float(np.median(bic2 - bic3)),
        "e_inf_distribution": summarise(fit3.params["e_inf"]),
        "fraction_e_inf_ge_0.5": float(high_e.mean()),
        "fraction_e_inf_le_0.05": float((fit3.params["e_inf"] <= 0.05).mean()),
        "censoring_overlap": overlap,
        "ic50_vs_emax_per_drug": per_drug.to_dict(orient="records"),
        "mean_per_drug_pcc_ic50_vs_emax": float(np.nanmean(per_drug["pcc_ic50_vs_emax"])) if len(per_drug) else float("nan"),
        "scipy_crosscheck_M3": crosscheck_with_scipy(x, y, m, fit3, n_sample=200),
        "premise_supported": None,
    }
    result["premise_supported"] = bool(
        result["fraction_M3_preferred_BIC"] > 0.5
        and np.isfinite(w_p) and w_p < 0.05
        and result["fraction_e_inf_le_0.05"] < 0.9
    )

    refit_df = pd.DataFrame(
        {
            "pair_id": sub["pair_id"].to_numpy(),
            "cell_id": sub["cell_id"].to_numpy(),
            "drug_id": sub["drug_id"].to_numpy(),
            "m3_e_inf": fit3.params["e_inf"],
            "m3_slope": fit3.params["slope"],
            "m3_midpoint": fit3.params["midpoint"],
            "m3_ln_ic50": d3["ln_ic50"],
            "m3_ic50_defined": d3["ic50_defined"],
            "m3_emax": d3["emax"],
            "m3_auc": d3["auc"],
            "m3_rmse": fit3.rmse,
            "m2_ln_ic50": d2["ln_ic50"],
            "m2_rmse": fit2.rmse,
            "m2_slope": fit2.params["slope"],
            "delta_bic": bic2 - bic3,
            "censored_published": censored,
        }
    )
    out_dir = audit_dir(cfg)
    refit_df.to_parquet(out_dir / "g3_refit_per_pair.parquet", index=False)
    per_drug.to_csv(out_dir / "g3_refit_per_drug.csv", index=False)
    save_json(result, out_dir / "g3_refit.json")

    log.info(
        "G-3: M3 preferred by BIC on %.1f%% of pairs; median dRMSE %.4f; E_inf>=0.5 on %.1f%%; premise_supported=%s",
        100 * result["fraction_M3_preferred_BIC"],
        result["delta_rmse_M2_minus_M3"]["median"],
        100 * result["fraction_e_inf_ge_0.5"],
        result["premise_supported"],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G-3 two- vs three-parameter refit")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    res = run(cfg, max_pairs=args.max_pairs, steps=args.steps)
    print(
        f"G-3 premise_supported={res['premise_supported']} "
        f"BIC-preferred={res['fraction_M3_preferred_BIC']:.3f} "
        f"p={res['wilcoxon_p_M3_better']:.3g}"
    )
    return 0 if res["premise_supported"] else 3


if __name__ == "__main__":
    sys.exit(main())
