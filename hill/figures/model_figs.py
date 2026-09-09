"""Model-behaviour figures: fitted curves, potency/efficacy separation, clinical, gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hill.figures.panels import load_json, load_table, mean_sd, save
from hill.figures.style import PAL, W_DOUBLE, W_ONEHALF, despine, panel, sig_stars
from hill.utils.stats import pearson


def _predictions(cfg: Any, step: str = "step2") -> pd.DataFrame | None:
    d = Path(cfg.paths.checkpoint_dir) / "ablation" / step / "predictions"
    files = sorted(d.glob("*_predictions.parquet")) if d.exists() else []
    if not files:
        d = Path(cfg.paths.checkpoint_dir)
        files = sorted(d.glob("**/predictions/*_predictions.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _hill(x: np.ndarray, e: float, s: float, m: float) -> np.ndarray:
    return e + (1 - e) / (1 + np.exp(np.clip(s * (x - m), -30, 30)))


def fig07_curves(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path | None:
    """Fig 7 — predicted curves against the raw measurements they were fitted to."""
    preds = _predictions(cfg)
    points = load_table(Path(cfg.paths.processed_dir) / "gdsc_points.parquet")
    if preds is None or points is None or preds.empty:
        return None
    preds = preds[np.isfinite(preds["e_inf_pred"])]
    if preds.empty:
        return None

    # pick a spread of pairs: low / mid / high predicted efficacy ceiling
    preds = preds.drop_duplicates("pair_id")
    q = preds["e_inf_pred"].rank(pct=True)
    picks = pd.concat([
        preds[q < 0.15].head(3), preds[(q >= 0.4) & (q < 0.6)].head(3), preds[q > 0.85].head(3)
    ])
    if picks.empty:
        return None
    picks = picks.head(9)

    n = len(picks)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(W_ONEHALF, W_ONEHALF * 0.34 * nrow),
                             squeeze=False)
    grouped = points.groupby("pair_id")
    for i, (_, row) in enumerate(picks.iterrows()):
        ax = axes[i // ncol][i % ncol]
        try:
            pts = grouped.get_group(row["pair_id"])
        except KeyError:
            continue
        x = np.linspace(pts["log_conc"].min() - 1, pts["log_conc"].max() + 1, 200)
        ax.plot(x, _hill(x, row["e_inf_pred"], row["slope_pred"], row["midpoint_pred"]),
                color=PAL["hill"], lw=1.3, label="predicted curve")
        ax.scatter(pts["log_conc"], pts["viability"], s=12, color="#333333", zorder=5,
                   label="measured")
        ax.axhline(0.5, color=PAL["gray"], lw=0.6, ls=":")
        ax.axhline(row["e_inf_pred"], color=PAL["efficacy"], lw=0.7, ls="--")
        tag = "no IC50" if not bool(row["ic50_defined"]) else f"lnIC50={row['ln_ic50_pred']:.2f}"
        ax.text(0.03, 0.06, f"$E_\\infty$={row['e_inf_pred']:.2f}\n{tag}", transform=ax.transAxes,
                fontsize=5.6)
        ax.set_ylim(-0.02, 1.08)
        if i % ncol == 0:
            ax.set_ylabel("viability")
        if i // ncol == nrow - 1:
            ax.set_xlabel("log concentration")
        despine(ax)
        panel(ax, "abcdefghi"[i], size=8)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=5.4, loc="upper right")
    fig.tight_layout()
    return save(fig, out_dir, "Fig7_predicted_curves", synthetic)


def fig08_decoupling(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path | None:
    """Fig 8 — potency and efficacy come out as separate, weakly related quantities."""
    preds = _predictions(cfg)
    if preds is None or preds.empty or not np.isfinite(preds["e_inf_pred"]).any():
        return None
    preds = preds[np.isfinite(preds["e_inf_pred"])].copy()

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, W_DOUBLE * 0.27))

    ax = axes[0]
    sc = ax.scatter(preds["midpoint_pred"], preds["e_inf_pred"], s=4, alpha=0.35,
                    c=preds["slope_pred"], cmap="viridis", linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("Hill slope $s$", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    ax.axhline(0.5, color=PAL["censored"], lw=0.8, ls="--")
    r = pearson(preds["midpoint_pred"].to_numpy(), preds["e_inf_pred"].to_numpy())
    ax.set_xlabel("predicted midpoint $m$ (potency)")
    ax.set_ylabel(r"predicted $E_\infty$ (1 $-$ efficacy)")
    ax.set_title(f"r = {r:.2f}", fontsize=7.5)
    panel(ax, "a")

    ax = axes[1]
    per_drug = []
    for d, g in preds.groupby("drug_index"):
        if len(g) >= max(3, cfg.eval.min_pairs_per_drug):
            per_drug.append(pearson(g["midpoint_pred"].to_numpy(), g["e_inf_pred"].to_numpy()))
    if per_drug:
        ax.hist(np.array(per_drug), bins=20, color=PAL["potency"], alpha=0.85)
        ax.axvline(float(np.nanmean(per_drug)), color=PAL["accent"], lw=1.0)
        ax.set_xlabel("per-drug corr($m$, $E_\\infty$)"); ax.set_ylabel("drugs")
        ax.set_title("Within drug", fontsize=7.5)
    panel(ax, "b")

    ax = axes[2]
    if "emax_refit_m3" in preds and np.isfinite(preds["emax_refit_m3"]).any():
        ok = np.isfinite(preds["emax_refit_m3"]) & np.isfinite(preds["emax_pred"])
        ax.scatter(preds.loc[ok, "emax_refit_m3"], preds.loc[ok, "emax_pred"], s=4, alpha=0.35,
                   color=PAL["efficacy"], linewidths=0, rasterized=True)
        ax.plot([0, 1], [0, 1], color=PAL["gray"], lw=0.8, ls="--")
        r = pearson(preds.loc[ok, "emax_refit_m3"].to_numpy(), preds.loc[ok, "emax_pred"].to_numpy())
        ax.set_xlabel("Emax, 3-parameter refit"); ax.set_ylabel("Emax, predicted")
        ax.set_title(f"r = {r:.2f}", fontsize=7.5)
    else:
        ax.text(0.5, 0.5, "run the G-3 refit\nto get observed Emax", ha="center", va="center",
                fontsize=7, color=PAL["gray"], transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    panel(ax, "c")

    for ax in axes:
        despine(ax)
    fig.tight_layout()
    return save(fig, out_dir, "Fig8_potency_efficacy", synthetic)


def fig09_clinical_gates(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path | None:
    """Fig 9 — TCGA transfer and the two histology gates with their LRT."""
    ablation = load_table(Path(cfg.paths.results_dir) / "ablation.csv")
    lrt_files = sorted((Path(cfg.paths.results_dir) / "stage1").glob("lrt_seed*.json"))
    lrt = load_json(lrt_files[0]) if lrt_files else None
    if ablation is None and lrt is None:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, W_DOUBLE * 0.27))

    ax = axes[0]
    if ablation is not None and "clinical_roc_auc" in ablation.columns:
        sub = ablation[np.isfinite(ablation["clinical_roc_auc"])]
        if not sub.empty:
            stats = mean_sd(sub, ["step"], "clinical_roc_auc").sort_values("step")
            ax.bar(stats["step"], stats["mean"], yerr=stats["std"], color=PAL["hill"],
                   width=0.55, capsize=2.5)
            ax.axhline(0.5, color=PAL["gray"], ls=":", lw=0.8)
            ax.set_xticks(stats["step"].astype(int))
            ax.set_xticklabels([f"step {int(s)}" for s in stats["step"]], fontsize=6)
            ax.set_ylabel("clinical ROC-AUC")
            ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "no clinical results", ha="center", va="center", fontsize=7,
                color=PAL["gray"], transform=ax.transAxes)
    ax.set_title("TCGA response at Cmax", fontsize=7.5)
    panel(ax, "a")

    ax = axes[1]
    if ablation is not None and {"gamma_e", "gamma_m"} & set(ablation.columns):
        for i, (col, colour, label) in enumerate(
            (("gamma_e", PAL["efficacy"], r"$\gamma_E$"), ("gamma_m", PAL["potency"], r"$\gamma_m$"))
        ):
            if col not in ablation.columns:
                continue
            vals = ablation[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            ax.bar(i, vals.mean(), yerr=vals.std(ddof=1) if vals.size > 1 else 0,
                   color=colour, width=0.5, capsize=2.5)
            ax.scatter(np.full(vals.size, i), vals, s=8, color="#222222", zorder=5)
        ax.axhline(0, color=PAL["text"], lw=0.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels([r"$\gamma_E$", r"$\gamma_m$"], fontsize=8)
        ax.set_ylabel("fitted gate coefficient")
    ax.set_title("Histology gates", fontsize=7.5)
    panel(ax, "b")

    ax = axes[2]
    ax.axis("off")
    if lrt:
        lines = [
            "Likelihood-ratio tests (nested by construction)", "",
            f"gamma_E: lambda = {lrt['lrt_gamma_E']['lambda']:.2f}",
            f"         p (boundary) = {lrt['lrt_gamma_E']['p_boundary']:.4g} "
            f"{sig_stars(lrt['lrt_gamma_E']['p_boundary'])}",
            f"         p (chi2_1)   = {lrt['lrt_gamma_E']['p_chi2']:.4g}", "",
            f"gamma_m: lambda = {lrt['lrt_gamma_m']['lambda']:.2f}",
            f"         p (boundary) = {lrt['lrt_gamma_m']['p_boundary']:.4g} "
            f"{sig_stars(lrt['lrt_gamma_m']['p_boundary'])}",
            f"         p (chi2_1)   = {lrt['lrt_gamma_m']['p_chi2']:.4g}", "",
            f"clinical AUC  gamma=0: {lrt['restricted']['clinical'].get('roc_auc', float('nan')):.3f}",
            f"              gamma_E: {lrt['gamma_E']['clinical'].get('roc_auc', float('nan')):.3f}",
            f"              both:    {lrt['both']['clinical'].get('roc_auc', float('nan')):.3f}",
        ]
        ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                fontsize=5.8, family="monospace")
    else:
        ax.text(0.5, 0.5, "run `python -m hill.stage1 --lrt`", ha="center", va="center",
                fontsize=7, color=PAL["gray"], transform=ax.transAxes)
    panel(ax, "c")

    for ax in axes[:2]:
        despine(ax)
    fig.tight_layout()
    return save(fig, out_dir, "Fig9_clinical_and_gates", synthetic)


def figS2_dataflow(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path | None:
    """Supplementary — dataset counts and tokenisation summary."""
    report = load_json(Path(cfg.paths.processed_dir) / "prepare_report.json")
    if report is None:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(W_ONEHALF, W_ONEHALF * 0.42))

    ax = axes[0]
    keys = [("n_points", "viability\nmeasurements"), ("n_pairs", "(cell, drug)\npairs"),
            ("n_cells", "cell lines"), ("n_drugs", "drugs")]
    vals = [report.get(k, 0) for k, _ in keys]
    ax.barh(np.arange(len(keys)), vals, color=PAL["hill"], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:,}", va="center", fontsize=6.5)
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([lab for _, lab in keys], fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("count (log scale)")
    panel(ax, "a")

    ax = axes[1]
    ax.axis("off")
    lines = [
        f"omics matrix: {report.get('omics_shape')}",
        f"group tokens: {report.get('n_group_tokens')}",
        f"total omics tokens: {report.get('n_omics_tokens')}",
        f"features in a pathway: {100 * float(report.get('token_coverage_fraction', float('nan'))):.1f}%",
        f"points per pair (mean): {float(report.get('points_per_pair_mean', float('nan'))):.2f}",
        f"drug features: {report.get('drug_feature_dim')} ({report.get('drug_feature_method')})",
        f"drug structure coverage: {100 * float(report.get('drug_structure_coverage', 0)):.0f}%",
        f"censored pairs: {100 * float(report.get('censored_fraction', float('nan'))):.1f}%",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=6.2, family="monospace")
    panel(ax, "b")

    despine(axes[0])
    fig.tight_layout()
    return save(fig, out_dir, "FigS2_dataflow", synthetic)
