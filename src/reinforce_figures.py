#!/usr/bin/env python3
"""Reinforcement figures for manuscript v6.

Fig S17: METABRIC external validation (4 panels)
Fig 5A-revised: CV-averaged modality ablation (with 95% CIs)
Fig S18: Fair embedding comparison across clinical drugs
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
BASE = PROJECT_ROOT
RES = PROJECT_ROOT / "results" / "reinforce"
FIG = PROJECT_ROOT / "research" / "figures" / "figures_v6"
os.makedirs(FIG, exist_ok=True)


def load_json(p): return json.load(open(p))


def fig_cv_ablation():
    try:
        cv = load_json(f"{RES}/cv_ablation.json")
        agg = cv['aggregate']
    except FileNotFoundError:
        cv = load_json(f"{RES}/cv_ablation_partial.json")
        fr = cv['fold_results']
        conds = ['full','drop_genomic','drop_transcriptomic','drop_proteomic','drop_histology']
        full = np.array([f['full']['pcc_drug_mean'] for f in fr])
        agg = {}
        for c in conds:
            vals = np.array([f[c]['pcc_drug_mean'] for f in fr])
            drops = full - vals
            agg[c] = {
                'drop_mean': float(drops.mean()),
                'drop_std':  float(drops.std(ddof=1)) if len(drops) > 1 else 0.0,
                'drop_ci95': [float(np.percentile(drops, 2.5)), float(np.percentile(drops, 97.5))],
            }

    conds = ['drop_genomic', 'drop_transcriptomic', 'drop_proteomic', 'drop_histology']
    labels = ['Genomic', 'Transcriptomic', 'Proteomic', 'Histopathology']
    means = [agg[c]['drop_mean'] for c in conds]
    stds  = [agg[c]['drop_std']  for c in conds]
    cis   = [agg[c]['drop_ci95'] for c in conds]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ['#9e9e9e', '#64b5f6', '#ffb74d', '#2196F3']
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=stds, color=colors, edgecolor='black', capsize=5, alpha=0.85)
    for i, (m, ci) in enumerate(zip(means, cis)):
        ax.plot(ci, [i, i], color='k', lw=1.2)
        ax.plot([ci[0]], [i], marker='|', color='k', markersize=10)
        ax.plot([ci[1]], [i], marker='|', color='k', markersize=10)
        ax.text(max(m, ci[1]) + 0.01, i, f"{m:+.3f}\n(95% CI {ci[0]:+.3f}, {ci[1]:+.3f})",
                va='center', fontsize=8)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(r'$\Delta$PCC$_{drug}$ upon modality removal (5-fold CV mean)', fontsize=11)
    ax.set_title('Modality ablation: CV-averaged effect sizes with 95% CIs',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{FIG}/Fig5A_cv_ablation.pdf", bbox_inches='tight')
    fig.savefig(f"{FIG}/Fig5A_cv_ablation.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Fig5A_cv_ablation")


def fig_metabric():
    mb = load_json(f"{RES}/metabric_validation.json")
    pred = pd.read_csv(f"{RES}/metabric_predicted_IC50.csv", index_col=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: Biomarker concordance (box plots)
    # Load clinical
    cl = pd.read_csv(DATA_ROOT / "08_metabric" / "data_clinical_sample.txt", sep="\t", comment='#',
                     low_memory=False)
    merged = pred.merge(cl[['SAMPLE_ID', 'ER_STATUS', 'HER2_STATUS']],
                        left_index=True, right_on='SAMPLE_ID', how='inner')

    ax = axes[0]
    d = [merged.loc[merged['ER_STATUS'] == 'Positive', 'Tamoxifen_1199'].dropna().values,
         merged.loc[merged['ER_STATUS'] == 'Negative', 'Tamoxifen_1199'].dropna().values,
         merged.loc[merged['ER_STATUS'] == 'Positive', 'Fulvestrant_1816'].dropna().values,
         merged.loc[merged['ER_STATUS'] == 'Negative', 'Fulvestrant_1816'].dropna().values]
    positions = [1, 2, 4, 5]
    bp = ax.boxplot(d, positions=positions, widths=0.7, patch_artist=True,
                    medianprops={'color': 'k', 'linewidth': 2})
    for b, c in zip(bp['boxes'], ['#e91e63', '#9ca3af', '#e91e63', '#9ca3af']):
        b.set_facecolor(c)
    ax.set_xticks([1.5, 4.5]); ax.set_xticklabels(['Tamoxifen', 'Fulvestrant'])
    ax.set_ylabel('Predicted IC$_{50}$', fontsize=11)
    ax.set_title('A. ER status stratification in METABRIC', fontsize=11, fontweight='bold')
    # Annotate p-values
    tam = mb['biomarker_concordance']['Tamoxifen_1199_vs_ER_STATUS']
    ful = mb['biomarker_concordance']['Fulvestrant_1816_vs_ER_STATUS']
    ax.text(1.5, ax.get_ylim()[1]*0.98, f"p = {tam['mannwhitney_p']:.1e}",
            ha='center', fontsize=9)
    ax.text(4.5, ax.get_ylim()[1]*0.98, f"p = {ful['mannwhitney_p']:.1e}",
            ha='center', fontsize=9)
    # Legend
    import matplotlib.patches as mpatches
    ax.legend([mpatches.Patch(color='#e91e63'), mpatches.Patch(color='#9ca3af')],
              ['ER+', 'ER−'], loc='lower right', fontsize=9)

    # Panel B: Drug-drug correlation conservation (scatter)
    ax = axes[1]
    corr = mb['drug_drug_correlation_conservation']
    tcga_ic = pd.read_csv(DATA_ROOT / "07_integrated" / "predicted_IC50_all_drugs.csv", index_col=0)
    DRUGS = list(pred.columns)
    t_corr = tcga_ic[DRUGS].corr(method='spearman').values
    m_corr = pred[DRUGS].corr(method='spearman').values
    idx = np.triu_indices(len(DRUGS), k=1)
    x = t_corr[idx]; y = m_corr[idx]
    ax.scatter(x, y, s=30, alpha=0.7, color='#2196F3', edgecolor='black')
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.4)
    ax.set_xlabel('TCGA drug-drug Spearman ρ', fontsize=11)
    ax.set_ylabel('METABRIC drug-drug Spearman ρ', fontsize=11)
    ax.set_title(f"B. Correlation conservation\n(Spearman ρ = {corr['spearman_rho']:.3f}, "
                 f"p = {corr['spearman_p']:.1e})", fontsize=11, fontweight='bold')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.grid(alpha=0.3)

    # Panel C: Survival HR forest
    ax = axes[2]
    surv = mb['survival']['per_drug']
    rows = []
    for d_name, r in surv.items():
        if isinstance(r.get('cox_p'), float):
            rows.append((d_name.rsplit('_', 1)[0], r['cox_hr_per_sd'], r['cox_p'],
                         r.get('cox_q_fdr', np.nan)))
    rows.sort(key=lambda x: x[2])
    names = [r[0] for r in rows]
    hrs = [r[1] for r in rows]
    qs = [r[3] for r in rows]
    y = np.arange(len(names))
    colors_hr = ['#d32f2f' if q < 0.05 else ('#f57c00' if q < 0.1 else '#9e9e9e')
                 for q in qs]
    ax.scatter(hrs, y, color=colors_hr, s=60, zorder=3)
    for i, (h, n, q) in enumerate(zip(hrs, names, qs)):
        ax.text(h+0.01, i, f"q={q:.3f}", va='center', fontsize=8)
    ax.axvline(1.0, color='k', lw=0.5, ls='--')
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Cox HR per SD of predicted IC$_{50}$', fontsize=11)
    ax.set_title(f"C. Survival association\n(n = {mb['survival']['n']} METABRIC patients)",
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{FIG}/FigS17_metabric_validation.pdf", bbox_inches='tight')
    fig.savefig(f"{FIG}/FigS17_metabric_validation.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved FigS17_metabric_validation")


def fig_fair_embedding():
    fe = load_json(f"{RES}/fair_embedding_and_bootstrap.json")
    fig, ax = plt.subplots(figsize=(10, 5))
    drugs = list(fe['drugs'].keys())
    methods = ['PathOmicDRP_embedding_256d','PCA256_4modal_raw',
               'PCA256_omics_only','Raw_4modal','ImputedIC50_13d']
    labels = ['PathOmicDRP\n(learned 256)', 'PCA-256\n(4-modal raw)',
              'PCA-256\n(omics only)', 'Raw\n(4-modal)', 'Imputed IC$_{50}$']
    colors = ['#2196F3', '#4CAF50', '#8BC34A', '#FF9800', '#9e9e9e']
    width = 0.15
    x = np.arange(len(drugs))

    for i, (m, lab, c) in enumerate(zip(methods, labels, colors)):
        aucs = [fe['drugs'][d]['methods'][m]['auc_mean'] for d in drugs]
        cis  = [fe['drugs'][d]['methods'][m].get('bootstrap_ci95', [None,None]) for d in drugs]
        ers  = np.array([[a - (ci[0] or a), (ci[1] or a) - a] for a, ci in zip(aucs, cis)]).T
        ax.bar(x + (i-2)*width, aucs, width=width, color=c, edgecolor='black',
               label=lab, alpha=0.85)
        ax.errorbar(x + (i-2)*width, aucs, yerr=ers, fmt='none', color='black',
                    capsize=2, lw=0.8)

    ax.set_xticks(x); ax.set_xticklabels(drugs, fontsize=10)
    ax.set_ylabel('Clinical AUC (5-fold CV)', fontsize=11)
    ax.set_title('Fair embedding comparison: learned fusion vs PCA-256 baselines (bootstrap 95% CIs)',
                 fontsize=11, fontweight='bold')
    ax.axhline(0.5, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{FIG}/FigS18_fair_embedding.pdf", bbox_inches='tight')
    fig.savefig(f"{FIG}/FigS18_fair_embedding.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved FigS18_fair_embedding")


if __name__ == '__main__':
    fig_cv_ablation()
    fig_metabric()
    fig_fair_embedding()
