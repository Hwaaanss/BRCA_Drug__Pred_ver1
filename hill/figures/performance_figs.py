"""Performance figures: main comparison, ablation ladder, per-drug view, HPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hill.figures.panels import load_json, load_table, mean_sd, save
from hill.figures.style import METHOD_COLOR, PAL, W_DOUBLE, despine, p_text, panel, sig_stars
from hill.utils.stats import paired_test

_STEP_SHORT = {
    -1: "baseline", 0: "ScalarHILL", 1: "HILL (1 sigma)", 2: "HILL (sigma_j)",
    3: "+ TCGA", 4: "+ gamma_E", 5: "+ gamma_m",
}


def _method_label(row: pd.Series) -> str:
    if row.get("step", -1) == -1:
        return str(row["method"])
    return _STEP_SHORT.get(int(row["step"]), str(row["method"]))


def fig04_main_performance(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path | None:
    """Fig 4 — every method against the naive mean-effects floor, with seed spread."""
    df = load_table(Path(cfg.paths.results_dir) / "ablation.csv")
    if df is None or df.empty or "delta_pcc_vs_naive" not in df:
        return None
    df = df[np.isfinite(df["delta_pcc_vs_naive"])].copy()
    if df.empty:
        return None
    df["label"] = df.apply(_method_label, axis=1)

    order = (
        df.groupby(["method", "label"], dropna=False)["delta_pcc_vs_naive"].mean()
        .reset_index().sort_values("delta_pcc_vs_naive")
    )
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.34),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    for i, (_, row) in enumerate(order.iterrows()):
        vals = df.loc[df["method"] == row["method"], "delta_pcc_vs_naive"].to_numpy()
        colour = METHOD_COLOR.get(row["method"], PAL["baseline"])
        ax.barh(i, vals.mean(), color=colour, alpha=0.85, height=0.62)
        if vals.size > 1:
            ax.errorbar(vals.mean(), i, xerr=vals.std(ddof=1), color=PAL["text"], lw=0.8,
                        capsize=2, fmt="none")
        jitter = np.random.default_rng(0).uniform(-0.16, 0.16, vals.size)
        ax.scatter(vals, i + jitter, s=5, color="#222222", alpha=0.8, zorder=5, linewidths=0)
    ax.axvline(0, color=PAL["text"], lw=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order["label"], fontsize=6.5)
    ax.set_xlabel(r"$\Delta$PCC vs NaiveMeanEffects (per-drug mean)")
    ax.text(0.99, 0.03, f"{int(df['seed'].nunique())} seeds", transform=ax.transAxes,
            ha="right", fontsize=6, color=PAL["gray"])
    panel(ax, "a")

    # HILL vs ScalarHILL, paired by seed — the paper's central comparison
    ax = axes[1]
    hill = df[df["method"] == "step2"].sort_values(["seed", "fold"])
    scalar = df[df["method"] == "step0"].sort_values(["seed", "fold"])
    n = min(len(hill), len(scalar))
    if n >= 1:
        h = hill["delta_pcc_vs_naive"].to_numpy()[:n]
        s = scalar["delta_pcc_vs_naive"].to_numpy()[:n]
        for i in range(n):
            ax.plot([0, 1], [s[i], h[i]], color=PAL["gray"], lw=0.7, alpha=0.8, zorder=1)
        ax.scatter(np.zeros(n), s, s=18, color=PAL["scalar"], zorder=3, label="ScalarHILL")
        ax.scatter(np.ones(n), h, s=18, color=PAL["hill"], zorder=3, label="HILL")
        test = paired_test(h, s)
        ax.set_title(f"{p_text(test['wilcoxon_p'])} {sig_stars(test['wilcoxon_p'])}", fontsize=7.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["scalar\nhead", "curve\nhead"], fontsize=6.5)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel(r"$\Delta$PCC vs naive")
        ax.axhline(0, color=PAL["text"], lw=0.7, ls=":")
    panel(ax, "b")

    for ax in axes:
        despine(ax)
    fig.tight_layout()
    return save(fig, out_dir, "Fig4_main_performance", synthetic)


def fig05_ladder(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path | None:
    """Fig 5 — the ablation ladder, one change at a time."""
    df = load_table(Path(cfg.paths.results_dir) / "ablation.csv")
    if df is None or df.empty:
        return None
    ladder = df[df["step"] >= 0].copy()
    if ladder.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.30))

    ax = axes[0]
    have_pcc = ladder[np.isfinite(ladder.get("delta_pcc_vs_naive", np.nan))]
    if not have_pcc.empty:
        stats = mean_sd(have_pcc, ["step"], "delta_pcc_vs_naive").sort_values("step")
        ax.errorbar(stats["step"], stats["mean"], yerr=stats["std"], marker="o", lw=1.2,
                    color=PAL["hill"], capsize=2.5, markersize=4)
        for _, r in stats.iterrows():
            ax.annotate(f"{r['mean']:.3f}", (r["step"], r["mean"]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=6)
        moli = df[df["method"] == "moli"]["delta_pcc_vs_naive"]
        if len(moli):
            ax.axhline(moli.mean(), color=PAL["moli"], ls="--", lw=0.9)
            ax.text(0.99, moli.mean(), " MOLI", color=PAL["moli"], fontsize=6, va="bottom",
                    ha="right", transform=ax.get_yaxis_transform())
        ax.set_xticks(sorted(stats["step"].unique()))
        ax.set_xticklabels([_STEP_SHORT.get(int(s), str(s)) for s in sorted(stats["step"].unique())],
                           rotation=30, ha="right", fontsize=6)
        ax.set_ylabel(r"$\Delta$PCC vs naive")
        ax.axhline(0, color=PAL["text"], lw=0.7, ls=":")
    panel(ax, "a")

    ax = axes[1]
    if "clinical_roc_auc" in ladder.columns and np.isfinite(ladder["clinical_roc_auc"]).any():
        sub = ladder[np.isfinite(ladder["clinical_roc_auc"])]
        stats = mean_sd(sub, ["step"], "clinical_roc_auc").sort_values("step")
        ax.errorbar(stats["step"], stats["mean"], yerr=stats["std"], marker="s", lw=1.2,
                    color=PAL["efficacy"], capsize=2.5, markersize=4)
        ax.axhline(0.5, color=PAL["gray"], ls=":", lw=0.8)
        ax.set_xticks(stats["step"].astype(int))
        ax.set_xticklabels([_STEP_SHORT.get(int(s), str(s)) for s in stats["step"]],
                           rotation=30, ha="right", fontsize=6)
        ax.set_ylabel("clinical ROC-AUC (TCGA)")
    else:
        ax.text(0.5, 0.5, "no TCGA transfer results\n(steps 3-5 not run)", ha="center", va="center",
                fontsize=7, color=PAL["gray"], transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    panel(ax, "b")

    for ax in axes:
        despine(ax)
    fig.tight_layout()
    return save(fig, out_dir, "Fig5_ablation_ladder", synthetic)


def fig06_per_drug(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path | None:
    """Fig 6 — per-drug performance: where the curve head helps and where it does not."""
    pred_dir = Path(cfg.paths.checkpoint_dir) / "ablation" / "step2" / "predictions"
    files = sorted(pred_dir.glob("*_per_drug.csv")) if pred_dir.exists() else []
    scalar_dir = Path(cfg.paths.checkpoint_dir) / "ablation" / "step0" / "predictions"
    scalar_files = sorted(scalar_dir.glob("*_per_drug.csv")) if scalar_dir.exists() else []
    if not files:
        return None

    hill = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    agg = hill.groupby("drug_index").agg(
        pcc=("pcc_excl_undefined", "mean"), naive=("pcc_naive", "mean"), n=("n_pairs", "mean")
    ).reset_index()
    agg["delta"] = agg["pcc"] - agg["naive"]
    agg = agg.sort_values("delta")

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.32),
                             gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    colours = [PAL["hill"] if d > 0 else PAL["scalar"] for d in agg["delta"]]
    ax.bar(np.arange(len(agg)), agg["delta"], color=colours, width=0.8)
    ax.axhline(0, color=PAL["text"], lw=0.8)
    ax.set_xlabel("drug (sorted)"); ax.set_ylabel(r"$\Delta$PCC vs naive")
    ax.set_xticks([])
    ax.text(0.02, 0.95, f"{int((agg['delta'] > 0).sum())} / {len(agg)} drugs above the floor",
            transform=ax.transAxes, fontsize=6.5, va="top")
    panel(ax, "a")

    ax = axes[1]
    if scalar_files:
        scalar = pd.concat([pd.read_csv(f) for f in scalar_files], ignore_index=True)
        s_agg = scalar.groupby("drug_index")["pcc_excl_undefined"].mean().rename("scalar")
        merged = agg.set_index("drug_index").join(s_agg, how="inner")
        ax.scatter(merged["scalar"], merged["pcc"], s=10, color=PAL["hill"], alpha=0.8)
        lim = [-1, 1]
        ax.plot(lim, lim, color=PAL["gray"], lw=0.8, ls="--")
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel("per-drug PCC, ScalarHILL"); ax.set_ylabel("per-drug PCC, HILL")
        won = int((merged["pcc"] > merged["scalar"]).sum())
        ax.set_title(f"HILL wins {won} / {len(merged)}", fontsize=7.5)
    else:
        ax.text(0.5, 0.5, "no ScalarHILL predictions", ha="center", va="center", fontsize=7,
                color=PAL["gray"], transform=ax.transAxes)
    panel(ax, "b")

    for ax in axes:
        despine(ax)
    fig.tight_layout()
    return save(fig, out_dir, "Fig6_per_drug", synthetic)


def fig10_hpo(cfg: Any, out_dir: Path, synthetic: bool = False) -> Path | None:
    """Fig 10 (supplementary) — what the hyper-parameter search explored and found."""
    trials = load_table(Path(cfg.hpo.storage).parent / "trials.csv")
    best = load_json(Path(cfg.hpo.storage).parent / "best_params.json")
    if trials is None or trials.empty:
        return None
    value_col = "value" if "value" in trials.columns else None
    if value_col is None:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, W_DOUBLE * 0.26))

    ax = axes[0]
    finished = trials[np.isfinite(trials[value_col])]
    ax.scatter(finished["number"], finished[value_col], s=14, color=PAL["hill"])
    running_best = finished[value_col].cummax()
    ax.plot(finished["number"], running_best, color=PAL["accent"], lw=1.0)
    ax.set_xlabel("trial"); ax.set_ylabel(cfg.hpo.metric.replace("_", " "))
    ax.set_title("Search history", fontsize=7.5)
    panel(ax, "a")

    ax = axes[1]
    param_cols = [c for c in trials.columns if c.startswith("params_")]
    if param_cols and len(finished) >= 3:
        corr = []
        for c in param_cols:
            v = pd.to_numeric(finished[c], errors="coerce")
            ok = np.isfinite(v) & np.isfinite(finished[value_col])
            corr.append((c.replace("params_", ""),
                         abs(np.corrcoef(v[ok], finished[value_col][ok])[0, 1]) if ok.sum() > 2 else np.nan))
        imp = pd.DataFrame(corr, columns=["param", "abs_corr"]).dropna().sort_values("abs_corr")
        ax.barh(np.arange(len(imp)), imp["abs_corr"], color=PAL["baseline"])
        ax.set_yticks(np.arange(len(imp)))
        ax.set_yticklabels(imp["param"], fontsize=5.5)
        ax.set_xlabel("|corr| with objective")
        ax.set_title("Sensitivity", fontsize=7.5)
    panel(ax, "b")

    ax = axes[2]
    ax.axis("off")
    if best:
        lines = [f"trials: {best['n_trials_completed']} complete, {best['n_trials_pruned']} pruned",
                 f"metric: {best['metric']}",
                 f"best value: {best['best_value']:.4f}",
                 f"fold: {best['search_fold']}, epochs: {best['search_epochs']}", "", "best params:"]
        lines += [f"  {k} = {v}" for k, v in best["best_params"].items()]
        ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                fontsize=5.8, family="monospace")
    panel(ax, "c")

    for ax in axes[:2]:
        despine(ax)
    fig.tight_layout()
    return save(fig, out_dir, "Fig10_hpo", synthetic)
