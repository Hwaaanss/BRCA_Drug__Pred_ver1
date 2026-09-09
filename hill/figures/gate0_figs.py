"""Figures for the data audit: the concept figure and the Gate-0 evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from hill.figures.panels import load_json, load_table, save
from hill.figures.style import PAL, W_DOUBLE, W_ONEHALF, W_SINGLE, despine, p_text, panel


def _hill(x: np.ndarray, e: float, s: float, m: float) -> np.ndarray:
    return e + (1 - e) / (1 + np.exp(s * (x - m)))


def fig01_concept(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path:
    """Fig 1 — what the official label-generating model can and cannot express."""
    fig, axes2d = plt.subplots(2, 2, figsize=(W_ONEHALF, W_ONEHALF * 0.88))
    axes = axes2d.ravel()
    x = np.linspace(-4, 6, 400)

    # (a) bottom-fixed vs free-bottom fit of the same partial responder
    ax = axes[0]
    truth = _hill(x, 0.42, 1.2, 0.5)
    obs_x = np.linspace(-3, 5, 7)
    rng = np.random.default_rng(3)
    obs_y = _hill(obs_x, 0.42, 1.2, 0.5) + rng.normal(0, 0.02, obs_x.size)
    ax.plot(x, truth, color=PAL["m3"], lw=1.4, label="3-parameter (free floor)")
    ax.plot(x, _hill(x, 0.0, 0.75, 2.9), color=PAL["m2"], lw=1.4, ls="--",
            label="2-parameter (floor pinned at 0)")
    ax.scatter(obs_x, obs_y, s=10, color="#333333", zorder=5, label="observed viability")
    ax.axhline(0.5, color=PAL["gray"], lw=0.6, ls=":")
    ax.annotate("", xy=(2.9, 0.30), xytext=(0.5, 0.30),
                arrowprops=dict(arrowstyle="->", color=PAL["accent"], lw=1.0))
    ax.text(1.7, 0.34, "midpoint pushed right", fontsize=6, color=PAL["accent"], ha="center")
    ax.set_xlabel("log concentration"); ax.set_ylabel("viability")
    ax.set_ylim(-0.02, 1.05); ax.legend(loc="lower left", fontsize=5.4)
    ax.set_title("Misspecified floor", fontsize=7.5)
    panel(ax, "a")

    # (b) censoring
    ax = axes[1]
    ax.plot(x, _hill(x, 0.55, 1.0, 1.0), color=PAL["m3"], lw=1.4)
    ax.axvspan(-3, 3.2, color=PAL["gray_light"], alpha=0.5, lw=0)
    ax.axhline(0.5, color=PAL["gray"], lw=0.6, ls=":")
    ax.text(0.1, 0.93, "tested range", fontsize=6, color="#555555", transform=ax.transAxes)
    ax.annotate("never reaches 50%", xy=(4.6, 0.56), xytext=(0.30, 0.20),
                textcoords="axes fraction", fontsize=6, color=PAL["censored"],
                arrowprops=dict(arrowstyle="->", color=PAL["censored"], lw=0.8))
    ax.set_xlabel("log concentration"); ax.set_ylabel("viability"); ax.set_ylim(-0.02, 1.05)
    ax.set_title("Censored = no IC50", fontsize=7.5)
    panel(ax, "b")

    # (c) supervision signal
    ax = axes[2]
    ax.plot(x, _hill(x, 0.2, 1.1, 1.0), color=PAL["m3"], lw=1.2, alpha=0.5)
    ax.scatter(obs_x, _hill(obs_x, 0.2, 1.1, 1.0), s=14, color=PAL["m3"], zorder=5)
    ax.scatter([1.0], [0.5], s=26, marker="X", color=PAL["m2"], zorder=6)
    ax.text(0.03, 0.22, "scalar target: 1 value", fontsize=6, color=PAL["m2"],
            transform=ax.transAxes)
    ax.text(0.03, 0.10, f"curve target: K = {obs_x.size} values", fontsize=6,
            color=PAL["m3"], transform=ax.transAxes)
    ax.set_xlabel("log concentration"); ax.set_ylabel("viability"); ax.set_ylim(-0.02, 1.05)
    ax.set_title("Supervision signal", fontsize=7.5)
    panel(ax, "c")

    # (d) potency / efficacy separation
    ax = axes[3]
    ax.plot(x, _hill(x, 0.05, 1.1, 1.0), color=PAL["potency"], lw=1.4, label="$E_\\infty$ = 0.05")
    ax.plot(x, _hill(x, 0.45, 1.1, 1.0), color=PAL["efficacy"], lw=1.4, label="$E_\\infty$ = 0.45")
    ax.axhline(0.5, color=PAL["gray"], lw=0.6, ls=":")
    ax.set_xlabel("log concentration"); ax.set_ylabel("viability"); ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title("Potency vs efficacy", fontsize=7.5)
    panel(ax, "d")

    for ax in axes:
        despine(ax)
    fig.tight_layout()
    return save(fig, out_dir, "Fig1_concept", synthetic)


def fig02_refit(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path | None:
    """Fig 2 — Gate-0 G-3: does a free floor beat the official fit?"""
    stats = load_json(Path(cfg.paths.results_dir) / "gate0" / "g3_refit.json")
    per_pair = load_table(Path(cfg.paths.results_dir) / "gate0" / "g3_refit_per_pair.parquet")
    if stats is None or per_pair is None:
        return None

    fig, axes2d = plt.subplots(2, 3, figsize=(W_DOUBLE, W_DOUBLE * 0.52))
    axes = axes2d.ravel()

    ax = axes[0]
    lim = float(np.nanpercentile(np.concatenate([per_pair["m2_rmse"], per_pair["m3_rmse"]]), 99))
    ax.scatter(per_pair["m2_rmse"], per_pair["m3_rmse"], s=2, alpha=0.25, color=PAL["hill"],
               edgecolors="none", rasterized=True)
    ax.plot([0, lim], [0, lim], color=PAL["gray"], lw=0.8, ls="--")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("per-pair RMSE, 2-parameter"); ax.set_ylabel("per-pair RMSE, 3-parameter")
    ax.set_title(f"M3 better: {100 * stats['fraction_M3_better_rmse']:.0f}%", fontsize=7.5)
    panel(ax, "a")

    ax = axes[1]
    d = per_pair["delta_bic"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    clip = np.percentile(np.abs(d), 99) if d.size else 1.0
    ax.hist(np.clip(d, -clip, clip), bins=60, color=PAL["m3"], alpha=0.85)
    ax.axvline(0, color=PAL["text"], lw=0.8)
    ax.set_xlabel(r"$\Delta$BIC (M2 $-$ M3)"); ax.set_ylabel("pairs")
    ax.set_title(f"M3 preferred: {100 * stats['fraction_M3_preferred_BIC']:.0f}%", fontsize=7.5)
    panel(ax, "b")

    ax = axes[2]
    ax.hist(per_pair["m3_e_inf"], bins=50, color=PAL["efficacy"], alpha=0.85)
    ax.axvline(0.5, color=PAL["censored"], lw=1.0, ls="--")
    ax.text(0.53, 0.78, "no IC50\nexists", fontsize=6, color=PAL["censored"], transform=ax.transAxes)
    ax.set_xlabel(r"fitted $E_\infty$"); ax.set_ylabel("pairs")
    ax.set_title("Efficacy ceiling", fontsize=7.5)
    panel(ax, "c")

    ax = axes[3]
    overlap = stats.get("censoring_overlap", {})
    vals = [overlap.get("p_high_einf_given_uncensored", np.nan),
            overlap.get("p_high_einf_given_censored", np.nan)]
    ax.bar([0, 1], vals, color=[PAL["gray"], PAL["censored"]], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["uncensored", "censored"], fontsize=6.5)
    ax.set_ylabel(r"P($E_\infty \geq 0.5$)")
    if np.isfinite(overlap.get("fisher_p", np.nan)):
        ax.set_title(p_text(overlap["fisher_p"]), fontsize=7.5)
    else:
        ax.set_title("Censored pairs", fontsize=7.5)
    panel(ax, "d")

    ax = axes[4]
    per_drug = load_table(Path(cfg.paths.results_dir) / "gate0" / "g3_refit_per_drug.csv")
    if per_drug is not None and "pcc_ic50_vs_emax" in per_drug:
        vals = per_drug["pcc_ic50_vs_emax"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=24, color=PAL["potency"], alpha=0.85)
        ax.axvline(float(np.mean(vals)) if vals.size else 0, color=PAL["accent"], lw=1.0)
        ax.set_xlabel("per-drug corr(ln IC50, Emax)"); ax.set_ylabel("drugs")
        ax.set_title("Potency vs efficacy", fontsize=7.5)
    panel(ax, "e")

    ax = axes[5]
    ax.axis("off")
    lines = [
        f"pairs audited: {stats['n_pairs_audited']:,}",
        f"median RMSE  M2 {stats['rmse_M2']['median']:.4f}",
        f"             M3 {stats['rmse_M3']['median']:.4f}",
        f"Wilcoxon p (M3 < M2): {stats['wilcoxon_p_M3_better']:.2e}",
        f"AIC prefers M3: {100 * stats['fraction_M3_preferred_AIC']:.1f}%",
        f"BIC prefers M3: {100 * stats['fraction_M3_preferred_BIC']:.1f}%",
        f"E_inf <= 0.05 (M2 adequate): {100 * stats['fraction_e_inf_le_0.05']:.1f}%",
        f"E_inf >= 0.50 (no IC50):     {100 * stats['fraction_e_inf_ge_0.5']:.1f}%",
        f"premise supported: {stats['premise_supported']}",
    ]
    ax.text(0.0, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=6.2, family="monospace", color=PAL["text"])
    panel(ax, "f")

    for ax in axes[:5]:
        despine(ax)
    fig.tight_layout()
    return save(fig, out_dir, "Fig2_gate0_refit", synthetic)


def fig03_censoring(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path | None:
    """Fig 3 — how much of the published label set is extrapolated."""
    stats = load_json(Path(cfg.paths.results_dir) / "gate0" / "g2_censoring.json")
    per_drug = load_table(Path(cfg.paths.results_dir) / "gate0" / "g2_censoring_per_drug.csv")
    if stats is None or per_drug is None:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(W_ONEHALF, W_ONEHALF * 0.40),
                             gridspec_kw={"width_ratios": [1, 2.2]})

    ax = axes[0]
    frac = stats["censored_fraction_overall"]
    ax.bar([0], [frac], color=PAL["censored"], width=0.5)
    ax.bar([0], [1 - frac], bottom=[frac], color=PAL["gray_light"], width=0.5)
    ax.text(0, frac / 2, f"{100 * frac:.1f}%", ha="center", va="center", fontsize=8,
            color="white", fontweight="bold")
    ax.set_xticks([]); ax.set_ylim(0, 1); ax.set_ylabel("fraction of (cell, drug) pairs")
    ax.set_title("Censored IC50", fontsize=7.5)
    panel(ax, "a")

    ax = axes[1]
    top = per_drug.sort_values("censored_fraction", ascending=False).head(30)
    ax.bar(np.arange(len(top)), top["censored_fraction"], color=PAL["censored"], alpha=0.9)
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(top["drug_id"].astype(str), rotation=90, fontsize=5)
    ax.set_ylim(0, 1.02); ax.set_xlabel("drug id (30 most censored)")
    ax.set_ylabel("censored fraction")
    ax.set_title("Per drug", fontsize=7.5)
    panel(ax, "b")

    for ax in axes:
        despine(ax)
    fig.tight_layout()
    return save(fig, out_dir, "Fig3_censoring", synthetic)


def figS1_variance(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path | None:
    """Supplementary — G-4 variance decomposition motivating the metric choice."""
    stats = load_json(Path(cfg.paths.results_dir) / "gate0" / "g4_variance.json")
    if stats is None:
        return None
    labels = [k for k in ("ln_ic50_published", "auc_published") if k in stats]
    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.6))
    bottoms = np.zeros(len(labels))
    for comp, colour, name in (
        ("f_drug", PAL["m2"], "drug main effect"),
        ("f_sample", PAL["m3"], "sample main effect"),
        ("f_residual", PAL["gray_light"], "residual / interaction"),
    ):
        vals = np.array([stats[l].get(comp, np.nan) for l in labels], dtype=float)
        ax.bar(np.arange(len(labels)), vals, bottom=bottoms, color=colour, label=name, width=0.55)
        bottoms += np.nan_to_num(vals)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([l.replace("_published", "") for l in labels], fontsize=7)
    ax.set_ylabel("share of label variance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=6)
    ax.set_title("Why drug-pooled correlation is not reported", fontsize=8)
    despine(ax)
    fig.tight_layout()
    return save(fig, out_dir, "FigS1_variance_decomposition", synthetic)
