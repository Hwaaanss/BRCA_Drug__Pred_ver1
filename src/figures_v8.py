#!/usr/bin/env python3
"""Generate v8 submission figures for the four new Genome Medicine analyses.

Figures:
  Fig10_decision_curves.pdf       — DCA for Dox / Cyclo / Pacl / Doce
  Fig11_depmap_survival.pdf       — DepMap BRCA essentiality x TCGA/METABRIC Cox HR
  FigS19_sota_benchmark.pdf       — PathOmicDRP vs PathomicFusion / MOLI / SuperFELT
  FigS20_cptac_biomarkers.pdf     — CPTAC ER+/ER-, HER2+/-, TNBC predicted IC50
  FigS21_cptac_drug_correlation.pdf — CPTAC vs TCGA drug-drug correlation scatter
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
ROOT = PROJECT_ROOT
OUT = PROJECT_ROOT / "research" / "figures" / "figures_v8"
os.makedirs(OUT, exist_ok=True)


def fig_decision_curves():
    df = pd.read_csv(f"{ROOT}/results/clinical_utility/decision_curves_v2.csv")
    drugs = df['drug'].unique().tolist()
    fig, axes = plt.subplots(1, len(drugs), figsize=(3.0 * len(drugs), 3.2), sharey=True)
    if len(drugs) == 1: axes = [axes]
    for ax, drug in zip(axes, drugs):
        sub = df[df['drug'] == drug].sort_values('threshold')
        ax.plot(sub['threshold'], sub['nb_model'], '-', lw=1.8, label='PathOmicDRP', color='#c0392b')
        ax.plot(sub['threshold'], sub['nb_treat_all'], '--', lw=1.2, label='Treat all', color='#7f8c8d')
        ax.axhline(0, color='k', lw=0.8, label='Treat none')
        ax.set_title(drug); ax.set_xlabel('Threshold probability')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Net benefit')
    axes[0].legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig10_decision_curves.pdf"); plt.savefig(f"{OUT}/Fig10_decision_curves.png", dpi=200)
    plt.close()


def fig_depmap_survival():
    val = pd.read_csv(f"{ROOT}/results/biological_validation/gene_validation.csv")
    # Top 30 genes by abs sum of -log10(p) across TCGA+METABRIC
    val = val.dropna(subset=['tcga_cox_p', 'metabric_cox_p', 'depmap_brca_mean_dep']).copy()
    val['score'] = -np.log10(val['tcga_cox_p'].clip(lower=1e-20)) + \
                   -np.log10(val['metabric_cox_p'].clip(lower=1e-20))
    val = val.sort_values('score', ascending=False).head(30)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    ax = axes[0]
    y = np.arange(len(val))
    ax.barh(y, -np.log10(val['tcga_cox_p'].clip(lower=1e-20)), color='#2980b9', label='TCGA', height=0.4)
    ax.barh(y + 0.4, -np.log10(val['metabric_cox_p'].clip(lower=1e-20)), color='#e67e22',
            label='METABRIC', height=0.4)
    ax.set_yticks(y + 0.2); ax.set_yticklabels(val['gene'], fontsize=7)
    ax.axvline(-np.log10(0.05), ls='--', color='r', lw=0.8)
    ax.set_xlabel(r'$-\log_{10}(p)$ for OS Cox'); ax.legend(fontsize=8)
    ax.set_title('Top 30 attention genes — survival association (TCGA & METABRIC)')
    ax.invert_yaxis(); ax.grid(alpha=0.3, axis='x')

    ax = axes[1]
    ax.scatter(val['depmap_brca_mean_dep'], np.log2(val['tcga_cox_hr']),
               s=30, alpha=0.7, color='#c0392b', label='TCGA HR')
    ax.scatter(val['depmap_brca_mean_dep'], np.log2(val['metabric_cox_hr']),
               s=30, alpha=0.7, color='#2980b9', label='METABRIC HR')
    for _, r in val.iterrows():
        ax.annotate(r['gene'], (r['depmap_brca_mean_dep'], np.log2(r['tcga_cox_hr'])),
                    fontsize=6, alpha=0.7)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0.5, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('DepMap BRCA mean dependency')
    ax.set_ylabel(r'$\log_2$ OS Cox HR')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Essentiality vs prognostic effect')
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig11_depmap_survival.pdf"); plt.savefig(f"{OUT}/Fig11_depmap_survival.png", dpi=200)
    plt.close()


def fig_sota_benchmark():
    # Use existing reinforce cv_ablation.json for PathOmicDRP + sota benchmark if available
    ref = json.load(open(f"{ROOT}/results/reinforce/cv_ablation.json"))
    pathomicdrp = {
        'mean': ref['aggregate']['full']['pcc_drug_mean'],
        'std':  ref['aggregate']['full']['pcc_drug_std'],
    }
    sota_path = f"{ROOT}/results/benchmark/sota_comparison.json"
    sota_partial = f"{ROOT}/results/benchmark/sota_comparison_partial.json"
    data = {'PathOmicDRP (ours)': pathomicdrp}
    if os.path.exists(sota_path):
        s = json.load(open(sota_path))['aggregate']
        for k, v in s.items():
            if k != 'PathOmicDRP':
                data[k] = {'mean': v['pcc_drug_mean_mean'], 'std': v.get('pcc_drug_mean_std', 0)}
    elif os.path.exists(sota_partial):
        sp = json.load(open(sota_partial))
        for m, folds in sp['results'].items():
            vals = [f['pcc_drug_mean'] for f in folds]
            data[m] = {'mean': float(np.mean(vals)),
                       'std':  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    names = list(data.keys())
    means = [data[n]['mean'] for n in names]; stds = [data[n]['std'] for n in names]
    colors = ['#c0392b' if 'PathOmicDRP' in n else '#34495e' for n in names]
    ax.bar(range(len(names)), means, yerr=stds, color=colors, capsize=4,
           error_kw={'elinewidth': 1.2})
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel('Mean per-drug Pearson r (5-fold CV)')
    ax.set_title('SOTA benchmarking on TCGA-BRCA 4-modal 431')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{OUT}/FigS19_sota_benchmark.pdf"); plt.savefig(f"{OUT}/FigS19_sota_benchmark.png", dpi=200)
    plt.close()


def fig_cptac_biomarkers():
    from io import StringIO
    d = json.load(open(f"{ROOT}/results/cptac_validation/biomarker_concordance.json"))
    pred = pd.read_csv(f"{ROOT}/results/cptac_validation/cptac_predicted_IC50.csv", index_col=0)
    with open(DATA_ROOT / "10_cptac" / "data_clinical_patient.txt") as f:
        lines = [l for l in f if not l.startswith('#')]
    cp = pd.read_csv(StringIO(''.join(lines)), sep='\t')
    with open(DATA_ROOT / "10_cptac" / "data_clinical_sample.txt") as f:
        lines = [l for l in f if not l.startswith('#')]
    cs = pd.read_csv(StringIO(''.join(lines)), sep='\t')
    clin = cs.merge(cp, on='PATIENT_ID', how='left').set_index('SAMPLE_ID')

    plots = [
        ('ER_UPDATED_CLINICAL_STATUS', 'Tamoxifen_1199', ('positive', 'negative'), ('ER+', 'ER-')),
        ('ER_UPDATED_CLINICAL_STATUS', 'Fulvestrant_1816', ('positive', 'negative'), ('ER+', 'ER-')),
        ('ERBB2_UPDATED_CLINICAL_STATUS', 'Lapatinib_1558', ('positive', 'negative'), ('HER2+', 'HER2-')),
        ('TNBC_UPDATED_CLINICAL_STATUS', 'Cisplatin_1005', ('positive', 'negative'), ('TNBC', 'non-TNBC')),
        ('TNBC_UPDATED_CLINICAL_STATUS', 'Paclitaxel_1080', ('positive', 'negative'), ('TNBC', 'non-TNBC')),
    ]
    fig, axes = plt.subplots(1, len(plots), figsize=(2.3 * len(plots), 3.2))
    for ax, (col, drug, values, labels) in zip(axes, plots):
        s = clin[col].astype(str).str.lower()
        a = pred.loc[clin.index[s == values[0]].intersection(pred.index), drug].dropna()
        b = pred.loc[clin.index[s == values[1]].intersection(pred.index), drug].dropna()
        bp = ax.boxplot([a, b], tick_labels=[f"{labels[0]}\n(n={len(a)})", f"{labels[1]}\n(n={len(b)})"],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#c0392b'); bp['boxes'][1].set_facecolor('#2980b9')
        for box in bp['boxes']: box.set_alpha(0.7)
        key = f"{col} | {drug}"
        p = d.get(key, {}).get('mannwhitney_p')
        if p is not None:
            ax.set_title(f"{drug}\nMW p={p:.2e}", fontsize=8)
        else:
            ax.set_title(drug, fontsize=8)
        ax.set_ylabel('Predicted IC50 (scaled)')
        ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{OUT}/FigS20_cptac_biomarkers.pdf"); plt.savefig(f"{OUT}/FigS20_cptac_biomarkers.png", dpi=200)
    plt.close()


def fig_cptac_drug_corr():
    C = pd.read_csv(f"{ROOT}/results/cptac_validation/drug_corr_cptac.csv", index_col=0)
    T = pd.read_csv(f"{ROOT}/results/cptac_validation/drug_corr_tcga.csv", index_col=0)
    # upper triangle scatter
    K = len(C); iu = np.triu_indices(K, k=1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    im0 = axes[0].imshow(T.values, vmin=-1, vmax=1, cmap='RdBu_r')
    axes[0].set_title('TCGA drug-drug Spearman'); axes[0].set_xticks(range(K))
    axes[0].set_xticklabels(T.columns, rotation=90, fontsize=6); axes[0].set_yticks(range(K))
    axes[0].set_yticklabels(T.index, fontsize=6)
    plt.colorbar(im0, ax=axes[0], fraction=0.04)
    im1 = axes[1].imshow(C.values, vmin=-1, vmax=1, cmap='RdBu_r')
    axes[1].set_title('CPTAC drug-drug Spearman'); axes[1].set_xticks(range(K))
    axes[1].set_xticklabels(C.columns, rotation=90, fontsize=6); axes[1].set_yticks(range(K))
    axes[1].set_yticklabels(C.index, fontsize=6)
    plt.colorbar(im1, ax=axes[1], fraction=0.04)
    axes[2].scatter(T.values[iu], C.values[iu], s=15, alpha=0.7, color='#34495e')
    from scipy.stats import pearsonr
    r, p = pearsonr(T.values[iu], C.values[iu])
    axes[2].plot([-1, 1], [-1, 1], 'k--', lw=0.6)
    axes[2].set_xlabel('TCGA drug-drug rho'); axes[2].set_ylabel('CPTAC drug-drug rho')
    axes[2].set_title(f'Concordance: Pearson r={r:.3f}, p={p:.3g}')
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/FigS21_cptac_drug_correlation.pdf"); plt.savefig(f"{OUT}/FigS21_cptac_drug_correlation.png", dpi=200)
    plt.close()


if __name__ == '__main__':
    fig_decision_curves(); print("Fig10 done")
    fig_depmap_survival(); print("Fig11 done")
    fig_sota_benchmark();  print("FigS19 done")
    fig_cptac_biomarkers();print("FigS20 done")
    fig_cptac_drug_corr(); print("FigS21 done")
    print("All v8 figures written to:", OUT)
