#!/usr/bin/env python3
"""Reinforcement figures — universal-rule compliant rewrite.

Universal figure rules enforced here:
  R1. NO set_title(). Panel identity is communicated by in-panel A/B/C labels
      and by self-explanatory axis labels / legends / in-panel captions.
  R2. Panels in one row share identical top/bottom; gridspec handles alignment.
  R3. Nothing overlaps data. Legends placed outside data region when needed,
      caption-text uses dedicated axes with zero data.
  R4. High-DPI, consistent palette, typographic hierarchy.
  R5. Figures are readable without referring to the manuscript legend.

Replaces Fig5A_cv_ablation, Fig9_drug_heterogeneity, FigS17_metabric_validation,
FigS18_fair_embedding in research/figures/figures_v6/.
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# --- Typographic + palette base ---
rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,          # unused — titles disabled
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'axes.linewidth': 0.9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
    'savefig.dpi': 400,
})

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[2])).resolve()
FINAL_ROOT = PROJECT_ROOT / "FINAL"
BASE = PROJECT_ROOT
RES = FINAL_ROOT / "results" / "reinforce"
FIG = FINAL_ROOT / "figures" / "reinforce_v6_v7"

PAL_MOA = {
    'DNA-damaging': '#c62828',
    'Microtubule':  '#6a1b9a',
    'Hormone':      '#1565c0',
    'Targeted':     '#2e7d32',
    'Apoptosis':    '#ef6c00',
    'Other':        '#757575',
}
PAL_MOD = {  # modality bars
    'Histopathology': '#1e88e5',
    'Transcriptomic': '#fb8c00',
    'Proteomic':      '#8e24aa',
    'Genomic':        '#757575',
}
PAL_METHOD = {
    'PathOmicDRP_embedding_256d': '#1e88e5',
    'PCA256_4modal_raw':          '#43a047',
    'PCA256_omics_only':          '#8bc34a',
    'Raw_4modal':                 '#fb8c00',
    'ImputedIC50_13d':            '#9e9e9e',
}
METHOD_LABEL = {
    'PathOmicDRP_embedding_256d': 'PathOmicDRP (learned 256-d embedding)',
    'PCA256_4modal_raw':          'PCA-256 of raw 4-modal features',
    'PCA256_omics_only':          'PCA-256 of omics-only',
    'Raw_4modal':                 'Raw 4-modal features (full dim)',
    'ImputedIC50_13d':            'Imputed IC$_{50}$ (13-d)',
}


def panel_label(ax, letter, dx=-0.15, dy=1.05):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='bottom', ha='left')


def inset_note(ax, text, xy=(0.02, 0.98), ha='left', va='top', fontsize=8.5):
    """Informational note rendered INSIDE the data area (not a title)."""
    ax.text(*xy, text, transform=ax.transAxes,
            fontsize=fontsize, ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='black', lw=0.5, alpha=0.94))


# ========================================================================
# Figure 5A (revised): CV-averaged modality importance
# ========================================================================
def fig5a_cv_ablation():
    cv = json.load(open(f"{RES}/cv_ablation.json"))
    ag = cv['aggregate']

    conds = ['drop_histology', 'drop_transcriptomic', 'drop_proteomic', 'drop_genomic']
    labels = ['Histopathology', 'Transcriptomic', 'Proteomic', 'Genomic']
    means  = [ag[c]['drop_mean'] for c in conds]
    stds   = [ag[c]['drop_std']  for c in conds]
    cis    = [ag[c]['drop_ci95'] for c in conds]
    rels   = [ag[c].get('relative_importance_pct_of_total_drop', 0) for c in conds]

    fig = plt.figure(figsize=(8.5, 4.0))
    gs = GridSpec(1, 1, figure=fig, left=0.22, right=0.96, top=0.95, bottom=0.18)
    ax = fig.add_subplot(gs[0, 0])

    y = np.arange(len(labels))[::-1]  # histology on top
    colors = [PAL_MOD[l] for l in labels]
    bars = ax.barh(y, means, xerr=stds, color=colors, edgecolor='black', lw=0.7,
                   capsize=4, alpha=0.90, zorder=3)
    for yi, (m, ci, rel) in enumerate(zip(means, cis, rels)):
        yy = y[yi]
        ax.hlines(yy, ci[0], ci[1], color='black', lw=0.9, zorder=4)
        ax.plot([ci[0], ci[1]], [yy, yy], '|', color='black', markersize=7, zorder=4)
        x_text = max(m, ci[1]) + 0.012
        ax.text(x_text, yy,
                f"Δ = {m:+.3f}  (95% CI {ci[0]:+.3f}, {ci[1]:+.3f})   {rel:.1f}%",
                va='center', ha='left', fontsize=9)
    ax.axvline(0, color='black', lw=0.6, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r'Mean $\Delta$PCC$_{drug}$ upon modality removal at inference '
                  r'(5-fold CV, error bars = 1 SD)')
    ax.set_xlim(-0.01, max(cis[0][1], means[0]) * 1.55)
    ax.grid(axis='x', alpha=0.28, zorder=1)

    fig.savefig(f"{FIG}/Fig5A_cv_ablation.pdf")
    fig.savefig(f"{FIG}/Fig5A_cv_ablation.png", dpi=400)
    plt.close(fig)
    print("Saved Fig5A_cv_ablation (v2)")


# ========================================================================
# Figure 9: Drug-level heterogeneity
# ========================================================================
def fig9_drug_heterogeneity():
    h = json.load(open(f"{RES}/drug_heterogeneity.json"))
    rows = pd.DataFrame(h['summary_winner_table'])
    lsc = pd.DataFrame([
        {'drug': d, 'moa_class': v['moa_class'], 'moa_desc': v['moa_desc'],
         'delta': v['delta_mean'], 'pcc_4m': v['pcc_4m_mean'],
         'pcc_3m': v['pcc_3m_mean']}
        for d, v in zip(
            [x['drug'] for x in h.get('summary_winner_table', [])] +
            [x['drug'] for x in [{
                'drug': k, **v,
                'moa_class': h['per_drug_histology_delta_imputed'][k].get('moa_class', 'Other') if False else None
            } for k, v in h['per_drug_histology_delta_imputed'].items()]],
            []
        )
    ]) if False else None

    # Simpler: load from csv written by reinforce_drug_heterogeneity
    landscape = pd.read_csv(f"{RES}/imputed_ic50_landscape.csv")
    winners = pd.read_csv(f"{RES}/drug_winner_table.csv")

    fig = plt.figure(figsize=(13.5, 5.0))
    gs = GridSpec(1, 2, figure=fig, left=0.08, right=0.98, top=0.84, bottom=0.20,
                  wspace=0.45)

    # -- Panel A: clinical AUC advantage (PathOmicDRP − PCA256) --
    axA = fig.add_subplot(gs[0, 0])
    w = winners.sort_values('advantage')
    colors = []
    for row in w.itertuples():
        if row.advantage > 0.05: colors.append('#1e88e5')
        elif row.advantage < -0.05: colors.append('#e53935')
        else: colors.append('#bdbdbd')
    bars = axA.barh(np.arange(len(w)), w['advantage'], color=colors,
                    edgecolor='black', lw=0.7, alpha=0.90, zorder=3)
    for i, row in enumerate(w.itertuples()):
        # Annotation inside if positive-right / negative-left so it doesn't clip
        if row.advantage >= 0:
            axA.text(row.advantage + 0.02, i,
                     f"{row.drug} ({row.moa_class})",
                     va='center', ha='left', fontsize=9)
        else:
            axA.text(row.advantage - 0.02, i,
                     f"{row.drug} ({row.moa_class})",
                     va='center', ha='right', fontsize=9)
        axA.text(0.015 if row.advantage>=0 else -0.015, i,
                 f"Δ={row.advantage:+.2f}", va='center',
                 ha='left' if row.advantage>=0 else 'right',
                 fontsize=8, color='white', fontweight='bold')
    axA.axvline(0, color='black', lw=0.6, zorder=2)
    axA.set_yticks([]); axA.set_ylim(-0.5, len(w) - 0.5)
    axA.set_xlim(-0.75, 0.75)
    axA.set_xlabel(r'$\Delta$AUC (PathOmicDRP 256-d $-$ PCA-256 of raw 4-modal)')
    axA.grid(axis='x', alpha=0.28, zorder=1)
    panel_label(axA, 'A')
    # In-panel legend
    legA = [mpatches.Patch(color='#1e88e5', label='PathOmicDRP wins  (Δ > 0.05)'),
            mpatches.Patch(color='#e53935', label='PCA-256 wins  (Δ < −0.05)'),
            mpatches.Patch(color='#bdbdbd', label='Tie')]
    axA.legend(handles=legA, loc='lower right', frameon=True, framealpha=0.95,
               edgecolor='black')

    # -- Panel B: histology ΔPCC across 13 drugs, grouped by MOA --
    axB = fig.add_subplot(gs[0, 1])
    landscape_sorted = landscape.sort_values(['moa_class', 'histology_delta'],
                                             ascending=[True, False]).reset_index(drop=True)
    colors_B = [PAL_MOA[m] for m in landscape_sorted['moa_class']]
    y = np.arange(len(landscape_sorted))
    axB.barh(y, landscape_sorted['histology_delta'], color=colors_B,
             edgecolor='black', lw=0.7, alpha=0.90, zorder=3)
    for i, row in enumerate(landscape_sorted.itertuples()):
        x = row.histology_delta
        axB.text(x + (0.003 if x >= 0 else -0.003), i,
                 f"{row.drug}", va='center',
                 ha='left' if x >= 0 else 'right', fontsize=9)
    axB.axvline(0, color='black', lw=0.6, zorder=2)
    axB.set_yticks([]); axB.set_ylim(-0.5, len(landscape_sorted) - 0.5)
    axB.set_xlim(-0.12, 0.16)
    axB.set_xlabel(r'$\Delta$PCC$_{drug}$ (4-modal $-$ 3-modal), imputed IC$_{50}$')
    axB.grid(axis='x', alpha=0.28, zorder=1)
    panel_label(axB, 'B')
    moa_present = landscape_sorted['moa_class'].unique()
    legB = [mpatches.Patch(color=PAL_MOA[m], label=m) for m in moa_present]
    axB.legend(handles=legB, loc='lower right', frameon=True, framealpha=0.95,
               edgecolor='black', title='Mechanism of action')

    fig.savefig(f"{FIG}/Fig9_drug_heterogeneity.pdf")
    fig.savefig(f"{FIG}/Fig9_drug_heterogeneity.png", dpi=400)
    plt.close(fig)
    print("Saved Fig9_drug_heterogeneity (v2)")


# ========================================================================
# Figure S17: METABRIC external validation
# ========================================================================
def figS17_metabric():
    mb = json.load(open(f"{RES}/metabric_validation.json"))
    pred = pd.read_csv(f"{RES}/metabric_predicted_IC50.csv", index_col=0)
    cl = pd.read_csv(f"{BASE}/08_metabric/data_clinical_sample.txt", sep="\t",
                     comment='#', low_memory=False)
    merged = pred.merge(cl[['SAMPLE_ID', 'ER_STATUS', 'HER2_STATUS']],
                        left_index=True, right_on='SAMPLE_ID', how='inner')

    fig = plt.figure(figsize=(15.0, 5.2))
    gs = GridSpec(1, 3, figure=fig, left=0.06, right=0.98, top=0.92, bottom=0.16,
                  wspace=0.42, width_ratios=[1.15, 1, 1.15])

    # -- Panel A: ER stratification boxplots --
    axA = fig.add_subplot(gs[0, 0])
    d = [merged.loc[merged['ER_STATUS'] == 'Positive', 'Tamoxifen_1199'].dropna().values,
         merged.loc[merged['ER_STATUS'] == 'Negative', 'Tamoxifen_1199'].dropna().values,
         merged.loc[merged['ER_STATUS'] == 'Positive', 'Fulvestrant_1816'].dropna().values,
         merged.loc[merged['ER_STATUS'] == 'Negative', 'Fulvestrant_1816'].dropna().values]
    positions = [1, 2, 4, 5]
    bp = axA.boxplot(d, positions=positions, widths=0.65, patch_artist=True,
                     medianprops={'color': 'black', 'linewidth': 1.6},
                     flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.35})
    for b, c in zip(bp['boxes'], ['#c2185b', '#9e9e9e', '#c2185b', '#9e9e9e']):
        b.set_facecolor(c); b.set_alpha(0.88); b.set_edgecolor('black')
    axA.set_xticks([1.5, 4.5]); axA.set_xticklabels(['Tamoxifen', 'Fulvestrant'])
    axA.set_ylabel('Predicted IC$_{50}$ (z-scaled)')
    tam = mb['biomarker_concordance']['Tamoxifen_1199_vs_ER_STATUS']
    ful = mb['biomarker_concordance']['Fulvestrant_1816_vs_ER_STATUS']
    ymax = axA.get_ylim()[1]; yr = ymax - axA.get_ylim()[0]
    for (cx, p, n_p, n_n) in [(1.5, tam['mannwhitney_p'], tam['n_pos'], tam['n_neg']),
                               (4.5, ful['mannwhitney_p'], ful['n_pos'], ful['n_neg'])]:
        axA.annotate('', xy=(cx-0.5, ymax - 0.06*yr), xytext=(cx+0.5, ymax - 0.06*yr),
                     arrowprops=dict(arrowstyle='-', color='black', lw=0.9))
        axA.text(cx, ymax - 0.02*yr,
                 f"p = {p:.1e}\nn = {n_p} / {n_n}",
                 ha='center', va='bottom', fontsize=8.5)
    axA.set_ylim(axA.get_ylim()[0], ymax + 0.25*yr)
    legA = [mpatches.Patch(color='#c2185b', label='ER+'),
            mpatches.Patch(color='#9e9e9e', label='ER−')]
    axA.legend(handles=legA, loc='lower right', frameon=True, framealpha=0.95,
               edgecolor='black')
    panel_label(axA, 'A')

    # -- Panel B: drug-drug correlation scatter (TCGA vs METABRIC) --
    axB = fig.add_subplot(gs[0, 1])
    tcga_ic = pd.read_csv(f"{BASE}/07_integrated/predicted_IC50_all_drugs.csv", index_col=0)
    DRUGS = list(pred.columns)
    t_corr = tcga_ic[DRUGS].corr(method='spearman').values
    m_corr = pred[DRUGS].corr(method='spearman').values
    idx = np.triu_indices(len(DRUGS), k=1)
    x = t_corr[idx]; y = m_corr[idx]
    axB.scatter(x, y, s=42, alpha=0.78, color='#1e88e5', edgecolor='black', lw=0.6,
                zorder=3)
    axB.plot([-1, 1], [-1, 1], '--', color='black', alpha=0.45, lw=0.9, zorder=2)
    axB.set_xlabel('TCGA drug-drug Spearman ρ')
    axB.set_ylabel('METABRIC drug-drug Spearman ρ')
    axB.set_xlim(-1, 1); axB.set_ylim(-1, 1)
    axB.set_aspect('equal', 'box')
    axB.grid(alpha=0.28, zorder=1)
    cc = mb['drug_drug_correlation_conservation']
    axB.text(0.04, 0.96, f"Spearman ρ = {cc['spearman_rho']:.3f}\n"
                         f"p = {cc['spearman_p']:.1e}\n"
                         f"{cc['n_pairs']} drug pairs",
             transform=axB.transAxes, va='top', ha='left', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor='black', alpha=0.95))
    panel_label(axB, 'B')

    # -- Panel C: Cox HR forest --
    axC = fig.add_subplot(gs[0, 2])
    surv = mb['survival']['per_drug']
    rows = [(k.rsplit('_', 1)[0], v['cox_hr_per_sd'], v['cox_p'],
             v.get('cox_q_fdr', np.nan))
            for k, v in surv.items() if isinstance(v.get('cox_p'), float)]
    rows.sort(key=lambda x: x[1])
    names = [r[0] for r in rows]
    hrs   = [r[1] for r in rows]
    qs    = [r[3] for r in rows]
    colors_c = ['#c62828' if q < 0.05 else ('#ef6c00' if q < 0.1 else '#757575')
                for q in qs]
    y = np.arange(len(names))
    axC.scatter(hrs, y, color=colors_c, s=64, zorder=3, edgecolor='black', lw=0.6)
    axC.axvline(1.0, color='black', lw=0.6, ls='--', zorder=2)
    for i, (h, q) in enumerate(zip(hrs, qs)):
        axC.text(h + 0.008, i, f"q = {q:.3f}", va='center', fontsize=8)
    axC.set_yticks(y); axC.set_yticklabels(names, fontsize=9)
    axC.set_xlabel('Cox hazard ratio per 1 SD of predicted IC$_{50}$')
    axC.grid(axis='x', alpha=0.28, zorder=1)
    # HR annotations in plot
    xmin = min(hrs) - 0.04; xmax = max(hrs) + 0.08
    axC.set_xlim(xmin, xmax)
    legC = [mpatches.Patch(color='#c62828', label='q < 0.05'),
            mpatches.Patch(color='#ef6c00', label='q < 0.1'),
            mpatches.Patch(color='#757575', label='n.s.')]
    axC.legend(handles=legC, loc='lower right', frameon=True, framealpha=0.95,
               edgecolor='black', title='BH-FDR')
    panel_label(axC, 'C')

    fig.savefig(f"{FIG}/FigS17_metabric_validation.pdf")
    fig.savefig(f"{FIG}/FigS17_metabric_validation.png", dpi=400)
    plt.close(fig)
    print("Saved FigS17_metabric_validation (v2)")


# ========================================================================
# Figure S18: fair-embedding clinical AUC
# ========================================================================
def figS18_fair_embedding():
    fe = json.load(open(f"{RES}/fair_embedding_and_bootstrap.json"))
    drugs = list(fe['drugs'].keys())
    methods = ['PathOmicDRP_embedding_256d', 'PCA256_4modal_raw',
               'PCA256_omics_only', 'Raw_4modal', 'ImputedIC50_13d']

    fig = plt.figure(figsize=(12.5, 5.5))
    gs = GridSpec(1, 1, figure=fig, left=0.08, right=0.98, top=0.85, bottom=0.28)
    ax = fig.add_subplot(gs[0, 0])
    width = 0.16
    x = np.arange(len(drugs))

    for i, m in enumerate(methods):
        aucs = [fe['drugs'][d]['methods'][m]['auc_mean'] for d in drugs]
        cis  = [fe['drugs'][d]['methods'][m].get('bootstrap_ci95', [None, None]) for d in drugs]
        ers  = np.array([[a - (ci[0] if ci[0] is not None else a),
                          (ci[1] if ci[1] is not None else a) - a]
                         for a, ci in zip(aucs, cis)]).T
        ax.bar(x + (i - 2) * width, aucs, width=width,
               color=PAL_METHOD[m], edgecolor='black', lw=0.7,
               alpha=0.93, zorder=3)
        ax.errorbar(x + (i - 2) * width, aucs, yerr=ers, fmt='none',
                    color='black', capsize=2, lw=0.7, zorder=4)

    # Remove default x ticks; we build a 3-line composite label below each group
    ax.set_xticks(x); ax.set_xticklabels([''] * len(drugs))
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('Clinical AUC (stratified CV, bootstrap 95% CI)')
    ax.axhline(0.5, color='black', lw=0.8, ls='--', alpha=0.6, zorder=2)
    ax.text(len(drugs) - 0.55, 0.52, 'chance', ha='right', va='bottom',
            fontsize=8, color='black', alpha=0.7)
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis='y', alpha=0.28, zorder=1)
    ax.set_xlim(-0.55, len(drugs) - 0.45)

    # MOA annotation below axis
    moa_map = {'Docetaxel': ('Microtubule', '#6a1b9a'),
               'Paclitaxel': ('Microtubule', '#6a1b9a'),
               'Cyclophosphamide': ('DNA-damaging', '#c62828'),
               'Doxorubicin': ('DNA-damaging', '#c62828')}
    trans = ax.get_xaxis_transform()
    for xi, d in enumerate(drugs):
        dr = fe['drugs'][d]
        # Line 1: drug name (large, bold)
        ax.text(xi, -0.04, d, ha='center', va='top',
                fontsize=10, fontweight='bold',
                transform=trans)
        # Line 2: MOA pill
        if d in moa_map:
            moa, col = moa_map[d]
            ax.text(xi, -0.11, moa, ha='center', va='top',
                    fontsize=8.5, color='white',
                    transform=trans,
                    bbox=dict(facecolor=col, edgecolor='none',
                              boxstyle='round,pad=0.28'))
        # Line 3: n info
        ax.text(xi, -0.20,
                f"n = {dr['n']} ({dr['n_pos']} resp. / {dr['n_neg']} non-resp.)",
                ha='center', va='top', fontsize=8.3, color='#333333',
                transform=trans)

    # Legend outside plot (right)
    handles = [mpatches.Patch(color=PAL_METHOD[m], label=METHOD_LABEL[m])
               for m in methods]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.16),
              ncol=5, frameon=False, fontsize=8)
    fig.savefig(f"{FIG}/FigS18_fair_embedding.pdf")
    fig.savefig(f"{FIG}/FigS18_fair_embedding.png", dpi=400)
    plt.close(fig)
    print("Saved FigS18_fair_embedding (v2)")


if __name__ == '__main__':
    fig5a_cv_ablation()
    fig9_drug_heterogeneity()
    figS17_metabric()
    figS18_fair_embedding()
