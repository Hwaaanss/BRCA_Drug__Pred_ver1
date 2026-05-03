#!/usr/bin/env python3
"""Drug-level heterogeneity analysis.

Integrates three performance landscapes across 13 drugs:
  1. Histology contribution (3-modal → 4-modal ΔPCC, CV mean from analysis4_ablation)
  2. Clinical AUC (PathOmicDRP 4-modal direct-output vs baselines, from w1)
  3. Fair-embedding clinical AUC (256-dim learned vs PCA-256 raw, from fair_embedding)

Classifies drugs by Mechanism of Action (MOA) and identifies patterns:
  - For which drug classes does supervised multi-modal fusion help?
  - For which does simpler dimensionality reduction (PCA-256) suffice or win?

Output: results/reinforce/drug_heterogeneity.json + figure + per-drug table.
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[2])).resolve()
FINAL_ROOT = PROJECT_ROOT / "FINAL"
RES = FINAL_ROOT / "results"
OUT = RES / "reinforce"
FIG = FINAL_ROOT / "figures" / "reinforce_v6_v7"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# Mechanism of action classification (BRCA-relevant 13 drugs)
MOA = {
    'Cisplatin':      ('DNA-damaging',   'Platinum DNA cross-linker'),
    'Gemcitabine':    ('DNA-damaging',   'Nucleoside analogue / replication stress'),
    'Cyclophosphamide': ('DNA-damaging', 'Alkylating agent'),
    'Doxorubicin':    ('DNA-damaging',   'Anthracycline / topo-II inhibitor'),
    'Epirubicin':     ('DNA-damaging',   'Anthracycline / topo-II inhibitor'),

    'Docetaxel':      ('Microtubule',    'Taxane (stabilizer)'),
    'Paclitaxel':     ('Microtubule',    'Taxane (stabilizer)'),
    'Vinblastine':    ('Microtubule',    'Vinca alkaloid (destabilizer)'),
    'Vincristine':    ('Microtubule',    'Vinca alkaloid (destabilizer)'),
    'Vinorelbine':    ('Microtubule',    'Vinca alkaloid (destabilizer)'),
    'Mitoxantrone':   ('DNA-damaging',   'Anthracenedione / topo-II inhibitor'),

    'Tamoxifen':      ('Hormone',        'SERM (ER antagonist)'),
    'Fulvestrant':    ('Hormone',        'SERD (ER degrader)'),
    'Lapatinib':      ('Targeted',       'HER2/EGFR TKI'),
    'OSI-027':        ('Targeted',       'mTORC1/2 inhibitor'),
    'Daporinad':      ('Targeted',       'NAMPT inhibitor'),
    'Venetoclax':     ('Apoptosis',      'BCL-2 inhibitor'),
    'ABT737':         ('Apoptosis',      'BCL-2/BCL-xL inhibitor'),
    'AZD5991':        ('Apoptosis',      'MCL-1 inhibitor'),
}


def main():
    # === 1. Per-drug 4m vs 3m (histology contribution) ===
    a4 = json.load(open(f"{RES}/strengthening/analysis4_ablation.json"))
    histology_contribution = {}
    for drug, v in a4['per_drug_ablation'].items():
        histology_contribution[drug] = {
            'pcc_4m_mean': v['pcc_4modal']['mean'],
            'pcc_4m_std':  v['pcc_4modal']['std'],
            'pcc_3m_mean': v['pcc_3modal']['mean'],
            'pcc_3m_std':  v['pcc_3modal']['std'],
            'delta_mean':  v['delta']['mean'],
            'delta_std':   v['delta']['std'],
            'improved_folds': v['improved_folds'],
        }

    # === 2. Fair-embedding AUC (4 drugs with clinical data) ===
    fe = json.load(open(f"{OUT}/fair_embedding_and_bootstrap.json"))
    clinical_auc = {}
    for drug, dr in fe['drugs'].items():
        clinical_auc[drug] = {
            'n': dr['n'], 'n_pos': dr['n_pos'], 'n_neg': dr['n_neg'],
            'PathOmicDRP_256d': dr['methods']['PathOmicDRP_embedding_256d'],
            'PCA256_4modal':    dr['methods']['PCA256_4modal_raw'],
            'PCA256_omics':     dr['methods']['PCA256_omics_only'],
            'Raw_4modal':       dr['methods']['Raw_4modal'],
            'ImputedIC50_13d':  dr['methods']['ImputedIC50_13d'],
        }

    # === 3. Direct PathOmicDRP_4modal clinical AUC (from w1) ===
    w1 = json.load(open(f"{RES}/strengthening/w1_clinical_validation.json"))
    direct_auc = {}
    for drug, dr in w1['drugs'].items():
        m = dr['methods'].get('PathOmicDRP_4modal', {})
        direct_auc[drug] = {
            'auc': m.get('auc'), 'ci': m.get('bootstrap_ci_95'),
            'perm_p': m.get('permutation_p'),
            'n': dr['n'], 'n_pos': dr['n_pos'], 'n_neg': dr['n_neg'],
        }

    # === 4. Per-drug winner classification (fair embedding) ===
    summary_rows = []
    for drug, ca in clinical_auc.items():
        pom = ca['PathOmicDRP_256d']['auc_mean']
        pca = ca['PCA256_4modal']['auc_mean']
        moa_cat, moa_desc = MOA.get(drug, ('Other', 'Unknown'))
        winner = 'PathOmicDRP' if pom > pca + 0.05 else ('PCA-256' if pca > pom + 0.05 else 'Tie')
        summary_rows.append({
            'drug': drug, 'moa_class': moa_cat, 'moa_desc': moa_desc,
            'PathOmicDRP_AUC': round(pom, 3),
            'PCA256_AUC':     round(pca, 3),
            'advantage':      round(pom - pca, 3),
            'winner':         winner,
            'n': ca['n'], 'n_pos': ca['n_pos'], 'n_neg': ca['n_neg'],
        })

    # === 5. Pattern synthesis ===
    df = pd.DataFrame(summary_rows)
    df.to_csv(f"{OUT}/drug_winner_table.csv", index=False)

    # Group by MOA class
    by_moa = {}
    for moa in df['moa_class'].unique():
        sub = df[df['moa_class'] == moa]
        by_moa[moa] = {
            'drugs': sub['drug'].tolist(),
            'mean_advantage_PathOmicDRP_minus_PCA256': float(sub['advantage'].mean()),
            'winners': sub['winner'].value_counts().to_dict(),
        }

    # === 6. Imputed-IC50 landscape (13 drugs, not restricted to 4 clinical) ===
    imputed_landscape = []
    for drug, hc in histology_contribution.items():
        moa_cat, moa_desc = MOA.get(drug, ('Other', 'Unknown'))
        imputed_landscape.append({
            'drug': drug, 'moa_class': moa_cat, 'moa_desc': moa_desc,
            'pcc_4m': round(hc['pcc_4m_mean'], 3),
            'pcc_3m': round(hc['pcc_3m_mean'], 3),
            'histology_delta': round(hc['delta_mean'], 3),
            'histology_helps': hc['delta_mean'] > 0,
            'improved_folds_out_of_5': hc['improved_folds'],
        })
    dfl = pd.DataFrame(imputed_landscape)
    dfl.to_csv(f"{OUT}/imputed_ic50_landscape.csv", index=False)

    by_moa_imputed = {}
    for moa in dfl['moa_class'].unique():
        sub = dfl[dfl['moa_class'] == moa]
        by_moa_imputed[moa] = {
            'n_drugs': len(sub),
            'mean_histology_delta': float(sub['histology_delta'].mean()),
            'n_drugs_helped_by_histology': int(sub['histology_helps'].sum()),
        }

    out = {
        'methods': {
            'fair_embedding': 'PathOmicDRP 256-dim learned embedding vs PCA-256 of raw 4-modal features, LogReg, stratified 3-fold CV. Used for 4 drugs with clinical outcome labels (n=25-57, n_neg=3-4).',
            'histology_contribution': '4-modal PCC minus 3-modal PCC on imputed IC50 target (analysis4_ablation.json), 5-fold CV mean per drug.',
        },
        'per_drug_clinical_fair_embedding': clinical_auc,
        'per_drug_direct_auc': direct_auc,
        'per_drug_histology_delta_imputed': histology_contribution,
        'summary_winner_table': summary_rows,
        'summary_by_moa_clinical': by_moa,
        'summary_by_moa_imputed': by_moa_imputed,
    }
    with open(f"{OUT}/drug_heterogeneity.json", 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # === Print summary ===
    print("=== Clinical fair-embedding (4 drugs) ===")
    print(df.to_string(index=False))
    print("\n=== Imputed IC50 landscape (13 drugs) ===")
    print(dfl.sort_values('histology_delta', ascending=False).to_string(index=False))
    print("\n=== By MOA (clinical) ===")
    for m, v in by_moa.items():
        print(f"  {m:15s}: drugs={v['drugs']} advantage={v['mean_advantage_PathOmicDRP_minus_PCA256']:+.3f} winners={v['winners']}")
    print("\n=== By MOA (imputed, 13 drugs) ===")
    for m, v in by_moa_imputed.items():
        print(f"  {m:15s}: n={v['n_drugs']} mean_Δhisto={v['mean_histology_delta']:+.3f} helped={v['n_drugs_helped_by_histology']}/{v['n_drugs']}")

    # === Figure: drug-level win/loss map ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: clinical fair embedding
    ax = axes[0]
    df_sorted = df.sort_values('advantage')
    colors = {'PathOmicDRP':'#2196F3','PCA-256':'#FF9800','Tie':'#9e9e9e'}
    bars = ax.barh(df_sorted['drug'], df_sorted['advantage'],
                    color=[colors[w] for w in df_sorted['winner']],
                    edgecolor='black')
    for bar, row in zip(bars, df_sorted.itertuples()):
        ax.text(row.advantage + (0.02 if row.advantage>=0 else -0.02),
                bar.get_y() + bar.get_height()/2,
                f"{row.moa_class}", va='center', fontsize=8,
                ha='left' if row.advantage>=0 else 'right')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel(r'$\Delta$AUC (PathOmicDRP 256d − PCA-256 raw)', fontsize=11)
    ax.set_title(f'A. Clinical AUC advantage by drug\n(n={len(df)} drugs with treatment-response labels)',
                 fontsize=11, fontweight='bold')
    import matplotlib.patches as mpatches
    ax.legend([mpatches.Patch(color='#2196F3'), mpatches.Patch(color='#FF9800'),
               mpatches.Patch(color='#9e9e9e')],
              ['PathOmicDRP wins (Δ>0.05)','PCA-256 wins (Δ<−0.05)','Tie'],
              fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    # Right: imputed IC50 histology delta by MOA
    ax = axes[1]
    dfl_sorted = dfl.sort_values('histology_delta')
    moa_colors = {'DNA-damaging':'#d32f2f','Microtubule':'#7b1fa2','Hormone':'#0288d1',
                  'Targeted':'#388e3c','Apoptosis':'#f57c00','Other':'#9e9e9e'}
    ax.barh(dfl_sorted['drug'], dfl_sorted['histology_delta'],
            color=[moa_colors.get(m,'#9e9e9e') for m in dfl_sorted['moa_class']],
            edgecolor='black')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel(r'$\Delta$PCC (4-modal − 3-modal), imputed IC$_{50}$', fontsize=11)
    ax.set_title('B. Histology contribution per drug (13 drugs)', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    # MOA legend
    patches = [mpatches.Patch(color=c, label=m) for m, c in moa_colors.items() if m in dfl['moa_class'].values]
    ax.legend(handles=patches, fontsize=8, loc='lower right')

    plt.tight_layout()
    fig.savefig(f"{FIG}/Fig9_drug_heterogeneity.pdf", bbox_inches='tight')
    fig.savefig(f"{FIG}/Fig9_drug_heterogeneity.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nSaved: {FIG}/Fig9_drug_heterogeneity.pdf")
    print(f"Saved: {OUT}/drug_heterogeneity.json + drug_winner_table.csv + imputed_ic50_landscape.csv")


if __name__ == '__main__':
    main()
