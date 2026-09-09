"""G-1 — is the raw viability really accessible, and is our normalisation right?

Test: fit the *GDSC official* two-parameter model (M2) to our own normalised
viability and check that it reproduces the published ``LN_IC50`` and ``AUC``.
If it does, the ingest pipeline is correct.  If it does not, the entire project
stops here (project rule R1) — no model code runs on a pipeline we cannot
validate.

    python -m hill.audit.raw_access --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from hill.audit.base import audit_dir, is_synthetic, load_tables, save_json, summarise
from hill.audit.curvefit import crosscheck_with_scipy, derived_from_fit, fit_curves
from hill.config import load_config
from hill.utils.logging import get_logger
from hill.utils.stats import pearson, spearman

log = get_logger("audit.g1")

PASS_PEARSON = 0.90


def run(cfg: Any, device: str | None = None, max_pairs: int | None = None) -> dict[str, Any]:
    points, pairs, padded = load_tables(cfg)
    device = device or ("cuda" if cfg.device == "cuda" else "cpu")

    idx = np.arange(len(pairs))
    if max_pairs and idx.size > max_pairs:
        idx = np.random.default_rng(0).choice(idx, max_pairs, replace=False)
        idx.sort()

    fit = fit_curves(
        padded["log_conc"][idx], padded["viability"][idx], padded["mask"][idx],
        model="M2", device=device,
    )
    lo = np.where(padded["mask"][idx], padded["log_conc"][idx], np.inf).min(axis=1)
    hi = np.where(padded["mask"][idx], padded["log_conc"][idx], -np.inf).max(axis=1)
    derived = derived_from_fit(fit, lo, hi)

    sub = pairs.iloc[idx]
    published_ic50 = sub["ln_ic50_published"].to_numpy(dtype=float)
    published_auc = sub.get("auc_published")
    published_auc = published_auc.to_numpy(dtype=float) if published_auc is not None else np.full(idx.size, np.nan)

    ours_ic50 = derived["ln_ic50"]
    ok = np.isfinite(ours_ic50) & np.isfinite(published_ic50)

    result: dict[str, Any] = {
        "gate": "G-1",
        "synthetic": is_synthetic(cfg),
        "n_pairs_total": int(len(pairs)),
        "n_pairs_audited": int(idx.size),
        "n_pairs_with_published_ic50": int(np.isfinite(published_ic50).sum()),
        "n_points": int(padded["mask"].sum()),
        "points_per_pair": summarise(padded["mask"].sum(axis=1)),
        "our_m2_rmse": summarise(fit.rmse),
        "ic50_pearson_vs_published": pearson(ours_ic50[ok], published_ic50[ok]),
        "ic50_spearman_vs_published": spearman(ours_ic50[ok], published_ic50[ok]),
        "ic50_median_abs_diff": float(np.median(np.abs(ours_ic50[ok] - published_ic50[ok]))) if ok.sum() else float("nan"),
        "auc_pearson_vs_published": pearson(derived["auc"], published_auc),
        "scipy_crosscheck": crosscheck_with_scipy(
            padded["log_conc"][idx], padded["viability"][idx], padded["mask"][idx], fit, n_sample=200
        ),
        "missing_pattern": {
            "pairs_without_published_fit": int(np.isnan(published_ic50).sum()),
            "fraction_pairs_without_published_fit": float(np.isnan(published_ic50).mean()),
        },
        "pass_threshold_pearson": PASS_PEARSON,
    }
    result["passed"] = bool(
        np.isfinite(result["ic50_pearson_vs_published"])
        and result["ic50_pearson_vs_published"] >= PASS_PEARSON
    )
    save_json(result, audit_dir(cfg) / "g1_raw_access.json")
    log.info(
        "G-1 %s: our-M2 vs published LN_IC50 r=%.3f (threshold %.2f)",
        "PASSED" if result["passed"] else "FAILED",
        result["ic50_pearson_vs_published"], PASS_PEARSON,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G-1 raw viability accessibility audit")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    res = run(cfg, max_pairs=args.max_pairs)
    print(f"G-1 passed={res['passed']} r={res['ic50_pearson_vs_published']:.3f}")
    return 0 if res["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
