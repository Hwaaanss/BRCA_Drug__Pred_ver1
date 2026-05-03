#!/usr/bin/env python3
"""Biological validation of top attention/importance genes for Genome Medicine.

Methodology
-----------
1. Per-drug gene importance via OOF Pearson correlation between gene expression
   (X_transcriptomic, log1p) and model-predicted IC50 across all 431 patients.
   This is a model-agnostic post-hoc importance score that avoids the pitfall
   of drug-agnostic attention pooling.

2. Top-K (K=50) genes per drug are aggregated into a union set.

3. DepMap dependency: for each gene, compute mean CRISPRGeneDependency score
   across 81 BRCA cell lines (DepMap 22Q4).  A gene with dependency >= 0.5 in
   most lines is considered a lineage essential / dependency.

4. TCGA-BRCA survival: univariate Cox regression (gene expression -> OS).

5. METABRIC external survival: univariate Cox on METABRIC microarray.

Outputs
  results/biological_validation/top_genes_per_drug.csv
  results/biological_validation/gene_validation.csv
  results/biological_validation/summary.json
  results/biological_validation/depmap_dependency_heatmap_data.csv
"""
import os, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from lifelines import CoxPHFitter

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

BASE = DATA_ROOT / "07_integrated"
CLIN_DIR = DATA_ROOT / "01_clinical"
OOF = PROJECT_ROOT / "results" / "oof"
METABRIC = DATA_ROOT / "08_metabric"
DEPMAP = DATA_ROOT / "09_depmap"
OUT = PROJECT_ROOT / "results" / "biological_validation"
os.makedirs(OUT, exist_ok=True)

DRUGS = [
    'Cisplatin_1005', 'Docetaxel_1007', 'Paclitaxel_1080',
    'Gemcitabine_1190', 'Tamoxifen_1199', 'Lapatinib_1558',
    'Vinblastine_1004', 'OSI-027_1594', 'Daporinad_1248',
    'Venetoclax_1909', 'ABT737_1910', 'AZD5991_1720',
    'Fulvestrant_1816',
]

TOP_K = 50


def log(m): print(f"[{time.strftime('%H:%M:%S')}] [BIO] {m}", flush=True)


def load_tcga_survival():
    c = pd.read_csv(os.path.join(CLIN_DIR, 'TCGA_BRCA_clinical.csv'))
    c = c[['submitter_id', 'vital_status', 'days_to_death', 'days_to_last_follow_up', 'age_at_index']].copy()
    c['event'] = (c['vital_status'] == 'Dead').astype(int)
    c['time'] = np.where(c['event'] == 1, c['days_to_death'], c['days_to_last_follow_up'])
    c = c.dropna(subset=['time'])
    c = c[c['time'] > 0]
    return c.set_index('submitter_id')[['time', 'event', 'age_at_index']]


def load_metabric():
    # clinical
    cp = pd.read_csv(os.path.join(METABRIC, 'data_clinical_patient.txt'), sep='\t', comment='#')
    cp = cp[['PATIENT_ID', 'OS_MONTHS', 'OS_STATUS', 'AGE_AT_DIAGNOSIS']].copy()
    cp['time'] = pd.to_numeric(cp['OS_MONTHS'], errors='coerce') * 30.44
    cp['event'] = cp['OS_STATUS'].astype(str).str.contains('1:|DECEASED', case=False, na=False).astype(int)
    cp = cp.dropna(subset=['time']); cp = cp[cp['time'] > 0]
    cp = cp.set_index('PATIENT_ID')
    # expression (microarray): rows = genes (Hugo), cols = samples; first 2 cols are Hugo/Entrez
    expr = pd.read_csv(os.path.join(METABRIC, 'data_mrna_illumina_microarray.txt'),
                       sep='\t', low_memory=False)
    expr = expr.dropna(subset=['Hugo_Symbol'])
    expr = expr.drop_duplicates('Hugo_Symbol').set_index('Hugo_Symbol')
    drop_cols = [c for c in ['Hugo_Symbol', 'Entrez_Gene_Id'] if c in expr.columns]
    expr = expr.drop(columns=drop_cols, errors='ignore')
    expr = expr.apply(pd.to_numeric, errors='coerce')
    # METABRIC samples look like MB-0001 etc - matches PATIENT_ID prefix
    return cp, expr


def main():
    log("Load OOF predictions + transcriptomic")
    oof = pd.read_csv(os.path.join(OOF, 'oof_predictions.csv')).set_index('patient_id')
    trans = pd.read_csv(os.path.join(BASE, 'X_transcriptomic.csv')).set_index('patient_id')

    common = oof.index.intersection(trans.index)
    trans_c = trans.loc[common].apply(pd.to_numeric, errors='coerce')
    trans_log = np.log1p(trans_c.clip(lower=0))
    gene_names = trans.columns.tolist()
    log(f"Common patients: {len(common)}, genes: {len(gene_names)}")

    # =========================================================================
    # Step 1: Per-drug gene importance
    # =========================================================================
    top_rows = []
    per_drug_top = {}
    for drug in DRUGS:
        y = oof.loc[common, f"pred_{drug}"].values
        X = trans_log.values  # (N, G)
        # vectorized Pearson: cor = (X - mean_X) @ (y - mean_y) / (std_X * std_y * N)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        denom = (X.std(axis=0) + 1e-12) * (y.std() + 1e-12) * len(y)
        cors = (Xc.T @ yc) / denom
        abs_cors = np.abs(cors)
        top_idx = np.argsort(abs_cors)[-TOP_K:][::-1]
        top_g = [gene_names[i] for i in top_idx]
        per_drug_top[drug] = top_g
        for rank, i in enumerate(top_idx):
            top_rows.append({'drug': drug, 'rank': rank + 1, 'gene': gene_names[i],
                             'pearson_r': float(cors[i])})
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(os.path.join(OUT, 'top_genes_per_drug.csv'), index=False)
    union_genes = sorted(set(top_df['gene']))
    log(f"Union top genes: {len(union_genes)}")

    # =========================================================================
    # Step 2: DepMap dependency in BRCA cell lines
    # =========================================================================
    log("Loading DepMap (this is the slow step, ~400MB CSV)")
    model = pd.read_csv(os.path.join(DEPMAP, 'Model.csv'))
    brca_ids = model[model['OncotreePrimaryDisease'].fillna('').str.contains('Breast', case=False)]['ModelID'].tolist()
    log(f"BRCA cell lines in DepMap: {len(brca_ids)}")

    # Read only BRCA rows + union gene columns for efficiency
    # First read header to find col names
    with open(os.path.join(DEPMAP, 'CRISPRGeneDependency.csv')) as f:
        header = f.readline().strip().split(',')
    # DepMap columns are like "A1BG (1)" -> gene symbol before space
    def parse_gene(c): return c.split(' ')[0]
    col_gene = {i: parse_gene(c) for i, c in enumerate(header)}
    gene_to_col = {}
    for i, g in col_gene.items():
        if g in union_genes and g not in gene_to_col:
            gene_to_col[g] = header[i]
    log(f"Matched genes in DepMap: {len(gene_to_col)} / {len(union_genes)}")

    usecols = [header[0]] + list(gene_to_col.values())
    dep = pd.read_csv(os.path.join(DEPMAP, 'CRISPRGeneDependency.csv'),
                      usecols=usecols, index_col=0, low_memory=False)
    dep_brca = dep.loc[dep.index.intersection(brca_ids)]
    log(f"BRCA x gene dep matrix: {dep_brca.shape}")

    # Mean dependency per gene across BRCA cell lines
    dep_summary = pd.DataFrame({
        'gene': [c.split(' ')[0] for c in dep_brca.columns],
        'brca_mean_dependency': dep_brca.mean(axis=0).values,
        'brca_median_dependency': dep_brca.median(axis=0).values,
        'brca_frac_dependent_above_0p5': (dep_brca >= 0.5).mean(axis=0).values,
    })
    dep_summary.to_csv(os.path.join(OUT, 'depmap_dependency_heatmap_data.csv'), index=False)

    # =========================================================================
    # Step 3: TCGA-BRCA survival Cox
    # =========================================================================
    log("TCGA-BRCA univariate Cox")
    tcga_surv = load_tcga_survival()
    overlap_t = tcga_surv.index.intersection(trans_c.index)
    tcga_surv = tcga_surv.loc[overlap_t]
    cox_tcga = {}
    for g in union_genes:
        if g not in trans_c.columns: continue
        x = np.log1p(trans_c.loc[overlap_t, g].clip(lower=0).values)
        if np.std(x) == 0 or np.isnan(x).any(): continue
        df = pd.DataFrame({'time': tcga_surv['time'].values, 'event': tcga_surv['event'].values,
                           'expr': x, 'age': tcga_surv['age_at_index'].values})
        df = df.dropna()
        if len(df) < 50: continue
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(df, duration_col='time', event_col='event')
            cox_tcga[g] = {'hr': float(np.exp(cph.params_['expr'])),
                           'p': float(cph.summary.loc['expr', 'p']),
                           'z': float(cph.summary.loc['expr', 'z'])}
        except Exception: pass

    # =========================================================================
    # Step 4: METABRIC survival Cox
    # =========================================================================
    log("METABRIC univariate Cox")
    mb_clin, mb_expr = load_metabric()
    overlap_mb = mb_clin.index.intersection(mb_expr.columns)
    log(f"METABRIC overlap: {len(overlap_mb)}")
    cox_mb = {}
    for g in union_genes:
        if g not in mb_expr.index: continue
        x = mb_expr.loc[g, overlap_mb].values.astype(float)
        mask = ~np.isnan(x)
        if mask.sum() < 50 or np.nanstd(x) == 0: continue
        ids_ok = [pid for i, pid in enumerate(overlap_mb) if mask[i]]
        df = pd.DataFrame({'time': mb_clin.loc[ids_ok, 'time'].values,
                           'event': mb_clin.loc[ids_ok, 'event'].values,
                           'expr': x[mask],
                           'age': mb_clin.loc[ids_ok, 'AGE_AT_DIAGNOSIS'].values})
        df = df.dropna()
        if len(df) < 50: continue
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(df, duration_col='time', event_col='event')
            cox_mb[g] = {'hr': float(np.exp(cph.params_['expr'])),
                         'p': float(cph.summary.loc['expr', 'p']),
                         'z': float(cph.summary.loc['expr', 'z'])}
        except Exception: pass

    # =========================================================================
    # Step 5: Aggregate gene-level validation table
    # =========================================================================
    dep_map_gene = dep_summary.set_index('gene').to_dict('index')
    rows = []
    for g in union_genes:
        drugs_w_gene = top_df[top_df['gene'] == g]['drug'].tolist()
        row = {
            'gene': g,
            'n_drugs_top50': len(drugs_w_gene),
            'drugs': ','.join(drugs_w_gene[:5]) + ('...' if len(drugs_w_gene) > 5 else ''),
            'depmap_brca_mean_dep': dep_map_gene.get(g, {}).get('brca_mean_dependency'),
            'depmap_brca_frac_dep>0.5': dep_map_gene.get(g, {}).get('brca_frac_dependent_above_0p5'),
            'tcga_cox_hr': cox_tcga.get(g, {}).get('hr'),
            'tcga_cox_p':  cox_tcga.get(g, {}).get('p'),
            'metabric_cox_hr': cox_mb.get(g, {}).get('hr'),
            'metabric_cox_p':  cox_mb.get(g, {}).get('p'),
        }
        rows.append(row)
    val = pd.DataFrame(rows)
    # concordance: same-direction HR in TCGA and METABRIC
    val['cox_same_direction'] = ((val['tcga_cox_hr'] > 1) == (val['metabric_cox_hr'] > 1))
    val.to_csv(os.path.join(OUT, 'gene_validation.csv'), index=False)

    summary = {
        'n_top_union_genes': len(union_genes),
        'n_matched_depmap': int(val['depmap_brca_mean_dep'].notna().sum()),
        'n_depmap_essential_gte_0p5_in_majority': int(
            (val['depmap_brca_frac_dep>0.5'] >= 0.5).sum()),
        'n_tcga_cox_tested': int(val['tcga_cox_p'].notna().sum()),
        'n_tcga_cox_sig_0p05': int((val['tcga_cox_p'] < 0.05).sum()),
        'n_metabric_cox_tested': int(val['metabric_cox_p'].notna().sum()),
        'n_metabric_cox_sig_0p05': int((val['metabric_cox_p'] < 0.05).sum()),
        'n_concordant_sig_both': int(((val['tcga_cox_p'] < 0.05) &
                                     (val['metabric_cox_p'] < 0.05) &
                                     val['cox_same_direction']).sum()),
        'n_concordant_any_direction': int(val['cox_same_direction'].sum()),
    }
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"Summary: {summary}")

    # Print top 10 genes by concordant significance
    concord = val[(val['tcga_cox_p'] < 0.05) & (val['metabric_cox_p'] < 0.05) &
                  val['cox_same_direction']].copy()
    concord['min_p'] = np.minimum(concord['tcga_cox_p'].fillna(1), concord['metabric_cox_p'].fillna(1))
    concord = concord.sort_values('min_p').head(20)
    log(f"Top concordant survival-significant genes: {concord['gene'].tolist()}")


if __name__ == '__main__':
    main()
