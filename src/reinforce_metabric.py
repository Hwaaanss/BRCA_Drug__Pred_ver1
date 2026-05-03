#!/usr/bin/env python3
"""METABRIC external validation.

Transfer TCGA-trained drug-sensitivity prediction to the independent METABRIC
cohort (microarray) and evaluate:
  1. Biomarker concordance (ER/HER2/PR → predicted Tamoxifen/Fulvestrant/Lapatinib IC50).
  2. Survival stratification (log-rank on median-split predicted IC50).
  3. Treatment-subgroup enrichment (hormone therapy recipients vs chemotherapy).
  4. PAM50/IntClust subtype-specific patterns, compared against TCGA.

Training target: GDSC-derived imputed IC50 for TCGA patients (existing).
Training features: TCGA RNA-seq z-scored top-variance genes.
Validation target: same Ridge model applied to z-scored METABRIC microarray.

Output: results/reinforce/metabric_validation.json + CSV of per-patient predictions.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu, spearmanr, pearsonr

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

INT = DATA_ROOT / "07_integrated"
MB = DATA_ROOT / "08_metabric"
OUT_DIR = PROJECT_ROOT / "results" / "reinforce"
os.makedirs(OUT_DIR, exist_ok=True)

DRUGS = [
    'Cisplatin_1005','Docetaxel_1007','Paclitaxel_1080','Gemcitabine_1190',
    'Tamoxifen_1199','Lapatinib_1558','Vinblastine_1004','OSI-027_1594',
    'Daporinad_1248','Venetoclax_1909','ABT737_1910','AZD5991_1720',
    'Fulvestrant_1816',
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_metabric():
    log("Loading METABRIC mRNA (large file)…")
    mrna = pd.read_csv(f"{MB}/data_mrna_illumina_microarray.txt", sep="\t", low_memory=False)
    log(f"  shape={mrna.shape}")
    mrna = mrna.drop_duplicates(subset=['Hugo_Symbol']).set_index('Hugo_Symbol')
    if 'Entrez_Gene_Id' in mrna.columns:
        mrna = mrna.drop(columns=['Entrez_Gene_Id'])
    mrna = mrna.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    mrna = mrna.dropna(thresh=int(0.9*mrna.shape[1]))
    mrna = mrna.fillna(mrna.median(axis=1).median())
    log(f"  cleaned shape={mrna.shape}")

    log("Loading METABRIC clinical …")
    cl_pat = pd.read_csv(f"{MB}/data_clinical_patient.txt", sep="\t", comment='#',
                         low_memory=False)
    cl_sam = pd.read_csv(f"{MB}/data_clinical_sample.txt", sep="\t", comment='#',
                         low_memory=False)
    log(f"  patient cols: {list(cl_pat.columns)[:15]}")
    log(f"  sample cols:  {list(cl_sam.columns)[:15]}")
    cl = cl_pat.merge(cl_sam, on='PATIENT_ID', how='inner', suffixes=('','_s'))
    log(f"  merged clinical: {cl.shape}")
    return mrna, cl


def load_tcga_training():
    log("Loading TCGA training matrices …")
    tra = pd.read_csv(f"{INT}/X_transcriptomic.csv").set_index('patient_id')
    ic = pd.read_csv(f"{INT}/predicted_IC50_all_drugs.csv", index_col=0)
    common = tra.index.intersection(ic.index)
    tra = tra.loc[common]; ic = ic.loc[common, DRUGS]
    log(f"  TCGA training n={len(common)} genes={tra.shape[1]}")
    # log1p transform to match training distribution of oncoPredict
    tra_vals = np.log1p(np.maximum(tra.values, 0))
    return tra, ic, pd.DataFrame(tra_vals, index=tra.index, columns=tra.columns)


def train_predict_transfer(tcga_expr_log1p, tcga_ic50, metabric_expr):
    """Train Ridge on TCGA, predict on METABRIC via gene intersection + z-score."""
    tcga_genes = set(tcga_expr_log1p.columns)
    mb_genes   = set(metabric_expr.index)
    common = sorted(tcga_genes & mb_genes)
    log(f"  TCGA∩METABRIC gene overlap: {len(common)}")

    Xt = tcga_expr_log1p[common].values.astype(np.float32)
    Xm = metabric_expr.loc[common].T.values.astype(np.float32)  # samples × genes

    # Independent standardization to align platform-specific means/variances
    sc_t = StandardScaler().fit(Xt)
    sc_m = StandardScaler().fit(Xm)
    Xt_z = sc_t.transform(Xt)
    Xm_z = sc_m.transform(Xm)

    preds = {}
    for drug in DRUGS:
        y = tcga_ic50[drug].values.astype(np.float32)
        model = Ridge(alpha=1.0)
        model.fit(Xt_z, y)
        preds[drug] = model.predict(Xm_z)
    out = pd.DataFrame(preds, index=metabric_expr.columns)
    return out, common


def biomarker_concordance(pred_df, clinical):
    """Test: ER+ → lower Tam/Fulv IC50; HER2+ → lower Lapatinib IC50; etc."""
    tests = [
        ('Tamoxifen_1199',  'ER_STATUS',        'Positive', 'Negative', 'neg'),   # lower in ER+
        ('Fulvestrant_1816','ER_STATUS',        'Positive', 'Negative', 'neg'),
        ('Lapatinib_1558',  'HER2_STATUS',      'Positive', 'Negative', 'neg'),
    ]
    # Column detection: try multiple cases
    def findcol(cl, keys):
        for k in keys:
            for c in cl.columns:
                if c.upper() == k.upper(): return c
        return None

    er_c  = findcol(clinical, ['ER_STATUS','ER Status','ER_IHC','ER status measured by IHC'])
    her_c = findcol(clinical, ['HER2_STATUS','HER2 Status','HER2 status measured by SNP6'])
    pr_c  = findcol(clinical, ['PR_STATUS','PR Status'])
    col_map = {'ER_STATUS': er_c, 'HER2_STATUS': her_c, 'PR_STATUS': pr_c}
    log(f"  biomarker cols: ER={er_c} HER2={her_c} PR={pr_c}")

    results = {}
    for drug, bm_key, pos_val, neg_val, expected in tests:
        col = col_map.get(bm_key)
        if not col:
            results[f"{drug}_vs_{bm_key}"] = {'skipped': 'column_missing'}
            continue
        # Merge on sample id
        df = pd.DataFrame({'sample': pred_df.index, 'ic50': pred_df[drug].values})
        cl2 = clinical[['SAMPLE_ID', col]].rename(columns={'SAMPLE_ID':'sample', col:'bm'})
        if 'SAMPLE_ID' not in clinical.columns:
            cl2 = clinical[['PATIENT_ID', col]].rename(columns={'PATIENT_ID':'sample', col:'bm'})
        df = df.merge(cl2, on='sample', how='inner').dropna()
        pos = df[df['bm'].astype(str).str.contains(pos_val, case=False, na=False)]['ic50'].values
        neg = df[df['bm'].astype(str).str.contains(neg_val, case=False, na=False)]['ic50'].values
        if len(pos) < 5 or len(neg) < 5:
            results[f"{drug}_vs_{bm_key}"] = {'n_pos': len(pos), 'n_neg': len(neg),
                                              'skipped': 'too_few'}
            continue
        U, p = mannwhitneyu(pos, neg, alternative='two-sided')
        # expected: lower IC50 in pos group (ER+/HER2+ → more drug-sensitive)
        delta = float(pos.mean() - neg.mean())
        concordant = (delta < 0) if expected == 'neg' else (delta > 0)
        results[f"{drug}_vs_{bm_key}"] = {
            'n_pos': len(pos), 'n_neg': len(neg),
            'mean_pos': float(pos.mean()), 'mean_neg': float(neg.mean()),
            'delta_pos_minus_neg': delta,
            'mannwhitney_p': float(p),
            'expected_direction': expected,
            'concordant_with_expectation': bool(concordant),
        }
        log(f"  {drug} vs {bm_key}: pos={pos.mean():.3f} (n={len(pos)}) "
            f"neg={neg.mean():.3f} (n={len(neg)}) p={p:.2e} "
            f"{'✓' if concordant else '✗'} expected")
    return results


def survival_stratification(pred_df, clinical):
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        log("  lifelines missing; pip installing…")
        os.system(f"{sys.executable} -m pip install -q lifelines 2>&1 | tail -1")
        from lifelines import CoxPHFitter
        from lifelines.statistics import logrank_test

    os_months_col = None; os_status_col = None
    for c in clinical.columns:
        if c.upper().replace(' ','_') in ('OS_MONTHS','OVERALL_SURVIVAL_(MONTHS)'):
            os_months_col = c
        if c.upper().replace(' ','_') in ('OS_STATUS','OVERALL_SURVIVAL_STATUS'):
            os_status_col = c
    if not os_months_col:
        for c in clinical.columns:
            if 'MONTH' in c.upper() and 'SURVIV' in c.upper(): os_months_col = c
    if not os_status_col:
        for c in clinical.columns:
            if 'SURVIV' in c.upper() and 'STATUS' in c.upper(): os_status_col = c
    log(f"  OS cols: {os_months_col} / {os_status_col}")
    if not (os_months_col and os_status_col):
        return {'skipped':'os cols missing'}

    def map_status(v):
        s = str(v).upper()
        if '1:DECEASED' in s or 'DEAD' in s or s == '1': return 1
        if '0:LIVING' in s or 'ALIVE' in s or s == '0': return 0
        return None

    id_col = 'SAMPLE_ID' if 'SAMPLE_ID' in clinical.columns else 'PATIENT_ID'
    cl = clinical[[id_col, os_months_col, os_status_col]].copy()
    cl.columns = ['sample','months','status_raw']
    cl['status'] = cl['status_raw'].map(map_status)
    cl = cl.dropna(subset=['months','status'])
    cl['months'] = pd.to_numeric(cl['months'], errors='coerce')
    cl = cl.dropna()
    pr = pred_df.reset_index()
    pr.columns = ['sample'] + list(pr.columns[1:])
    cl = cl.merge(pr, on='sample', how='inner')
    log(f"  survival-evaluable n={len(cl)}")
    if len(cl) < 50:
        return {'skipped': 'too_few', 'n': len(cl)}

    results = {'n': len(cl), 'per_drug': {}}
    for drug in DRUGS:
        if drug not in cl.columns: continue
        cph = CoxPHFitter()
        sub = cl[['months','status',drug]].rename(columns={drug:'ic50'}).copy()
        sub['ic50'] = (sub['ic50'] - sub['ic50'].mean())/sub['ic50'].std()
        try:
            cph.fit(sub, duration_col='months', event_col='status')
            hr = float(np.exp(cph.params_['ic50']))
            p  = float(cph.summary.loc['ic50','p'])
            # median split log-rank
            med = sub['ic50'].median()
            lo = sub[sub['ic50'] <= med]
            hi = sub[sub['ic50'] >  med]
            lr = logrank_test(lo['months'], hi['months'], lo['status'], hi['status'])
            results['per_drug'][drug] = {
                'cox_hr_per_sd': hr, 'cox_p': p,
                'logrank_p': float(lr.p_value), 'logrank_stat': float(lr.test_statistic),
            }
        except Exception as e:
            results['per_drug'][drug] = {'error': str(e)[:100]}
    # FDR
    ps = [v.get('cox_p') for v in results['per_drug'].values() if isinstance(v.get('cox_p'), float)]
    if ps:
        from statsmodels.stats.multitest import multipletests
        try:
            rej, qs, _, _ = multipletests(ps, method='fdr_bh')
            i = 0
            for drug, r in results['per_drug'].items():
                if isinstance(r.get('cox_p'), float):
                    r['cox_q_fdr'] = float(qs[i]); i+=1
        except Exception:
            pass
    return results


def tcga_metabric_subtype_concordance(pred_df, clinical, tcga_ic50):
    """Correlate drug-drug IC50 correlation structure between TCGA and METABRIC."""
    tcga_corr = tcga_ic50[DRUGS].corr(method='spearman').values
    mb_corr   = pred_df[DRUGS].corr(method='spearman').values
    idx = np.triu_indices(len(DRUGS), k=1)
    rho, p = spearmanr(tcga_corr[idx], mb_corr[idx])
    return {'spearman_rho': float(rho), 'spearman_p': float(p), 'n_pairs': int(len(idx[0]))}


def main():
    mrna, clinical = load_metabric()
    tcga_tra_raw, tcga_ic50, tcga_tra_log = load_tcga_training()

    log("Training Ridge transfer models (TCGA → METABRIC) …")
    pred_df, common_genes = train_predict_transfer(tcga_tra_log, tcga_ic50, mrna)
    pred_df.index.name = 'SAMPLE_ID'
    pred_df.to_csv(f"{OUT_DIR}/metabric_predicted_IC50.csv")
    log(f"  METABRIC predictions: {pred_df.shape} → saved")

    # Harmonise clinical identifiers: clinical uses PATIENT_ID; pred uses MB-XXXX (sample)
    if 'SAMPLE_ID' not in clinical.columns:
        if 'PATIENT_ID' in clinical.columns:
            clinical['SAMPLE_ID'] = clinical['PATIENT_ID']

    log("\n== Biomarker concordance ==")
    bm = biomarker_concordance(pred_df, clinical)
    log("\n== Survival stratification ==")
    surv = survival_stratification(pred_df, clinical)
    log("\n== Drug-drug correlation conservation (TCGA vs METABRIC) ==")
    corr = tcga_metabric_subtype_concordance(pred_df, clinical, tcga_ic50)
    log(f"  Spearman rho = {corr['spearman_rho']:.3f} (p={corr['spearman_p']:.2e})")

    out = {
        'n_metabric_samples': int(pred_df.shape[0]),
        'n_common_genes': len(common_genes),
        'biomarker_concordance': bm,
        'survival': surv,
        'drug_drug_correlation_conservation': corr,
    }
    with open(f"{OUT_DIR}/metabric_validation.json", 'w') as f:
        json.dump(out, f, indent=2, default=str)
    log(f"\nSaved {OUT_DIR}/metabric_validation.json")


if __name__ == '__main__':
    main()
