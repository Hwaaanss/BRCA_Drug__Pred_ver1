#!/usr/bin/env python3
"""CPTAC-BRCA 3-modal partial external validation of PathOmicDRP (Genome Medicine).

CPTAC-BRCA (Krug et al. 2020, Cell) contributes 122 BRCA tumors with matched
WES + RNA-seq + MS-based proteomics and rich clinical/proteogenomic annotation
(ER / PR / ERBB2 / TNBC status, PAM50, ESTIMATE / TIL scores).  This is the
only public BRCA cohort outside TCGA with >= 3 overlapping molecular layers.

Because CPTAC proteomics uses mass spec (not TCGA RPPA), the proteomic modality
cannot be directly harmonized.  We run the 5 saved PathOmicDRP fold models in
3-modal inference mode (proteomic zeroed, histology unavailable) and evaluate:

  (i)  Biomarker concordance
       - ER+ vs ER-  -> mean predicted IC50 of endocrine drugs (Tamoxifen,
         Fulvestrant): expect ER+ to have lower predicted IC50 (more sensitive).
       - ERBB2+ vs ERBB2-  -> Lapatinib predicted IC50: expect HER2+ lower.
       - TNBC vs non-TNBC -> Cisplatin / Paclitaxel: expect TNBC lower.
  (ii) Drug-drug correlation of the CPTAC predicted IC50 matrix vs the
       TCGA OOF predicted IC50 matrix (Spearman rho).
  (iii) Proteogenomic biomarker correlations (ESTIMATE immune / stromal /
       TumorPurity) vs predicted IC50 to spot spurious confounding.

Output
  results/cptac_validation/cptac_predicted_IC50.csv
  results/cptac_validation/biomarker_concordance.json
  results/cptac_validation/cptac_vs_tcga_drug_drug_corr.csv
  results/cptac_validation/summary.json
"""
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, mannwhitneyu, pearsonr

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import PathOmicDRP, get_default_config
from train_phase3_4modal import MultiDrugDataset4Modal, collate_4modal
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TCGA = DATA_ROOT / "07_integrated"
HISTO_DIR = DATA_ROOT / "05_morphology" / "features"
FOLDS = PROJECT_ROOT / "results" / "reinforce"
CPTAC = DATA_ROOT / "10_cptac"
OOF_DIR = PROJECT_ROOT / "results" / "oof"
OUT = PROJECT_ROOT / "results" / "cptac_validation"
os.makedirs(OUT, exist_ok=True)

DRUGS = [
    'Cisplatin_1005', 'Docetaxel_1007', 'Paclitaxel_1080',
    'Gemcitabine_1190', 'Tamoxifen_1199', 'Lapatinib_1558',
    'Vinblastine_1004', 'OSI-027_1594', 'Daporinad_1248',
    'Venetoclax_1909', 'ABT737_1910', 'AZD5991_1720',
    'Fulvestrant_1816',
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] [CPTAC] {m}", flush=True)


def read_cbio_clinical(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#')]
    from io import StringIO
    return pd.read_csv(StringIO(''.join(lines)), sep='\t')


def main():
    # =========================================================================
    # 1. Build CPTAC feature matrices aligned to TCGA training features
    # =========================================================================
    log("Load TCGA training features for feature alignment")
    gen_tcga = pd.read_csv(os.path.join(TCGA, 'X_genomic.csv'))
    tra_tcga = pd.read_csv(os.path.join(TCGA, 'X_transcriptomic.csv'))
    pro_tcga = pd.read_csv(os.path.join(TCGA, 'X_proteomic.csv'))

    gen_cols = [c for c in gen_tcga.columns if c != 'patient_id']
    tra_cols = [c for c in tra_tcga.columns if c != 'patient_id']
    pro_cols = [c for c in pro_tcga.columns if c != 'patient_id']
    log(f"TCGA dims: gen={len(gen_cols)} trans={len(tra_cols)} prot={len(pro_cols)}")

    # CPTAC clinical
    clin_p = read_cbio_clinical(os.path.join(CPTAC, 'data_clinical_patient.txt'))
    clin_s = read_cbio_clinical(os.path.join(CPTAC, 'data_clinical_sample.txt'))
    clin_p.columns = [c.strip() for c in clin_p.columns]
    clin_s.columns = [c.strip() for c in clin_s.columns]
    log(f"CPTAC clinical_patient: {clin_p.shape}, sample: {clin_s.shape}")

    # Join patient + sample (columns are uppercase with underscores in this cBio export)
    clin = clin_s.merge(clin_p, on='PATIENT_ID', how='left')
    clin = clin.rename(columns={'SAMPLE_ID': 'sample_id'})
    clin['sample_id'] = clin['sample_id'].astype(str)
    samples_all = clin['sample_id'].tolist()

    # CPTAC mRNA (Hugo x samples)
    mrna = pd.read_csv(os.path.join(CPTAC, 'data_mrna_seq_fpkm.txt'), sep='\t', low_memory=False)
    mrna = mrna.dropna(subset=['Hugo_Symbol']).drop_duplicates('Hugo_Symbol')
    mrna = mrna.set_index('Hugo_Symbol')
    drop = [c for c in mrna.columns if c in ('Entrez_Gene_Id', 'Hugo_Symbol')]
    mrna = mrna.drop(columns=drop, errors='ignore').apply(pd.to_numeric, errors='coerce')
    cptac_samples = [s for s in mrna.columns if s in samples_all]
    log(f"CPTAC mRNA samples present: {len(cptac_samples)}")

    # Mutations -> binary gene matrix matching gen_cols (TCGA mutation panel)
    # gen_cols contains 192 gene symbols + TMB (last col presumably)
    tmb_col = [c for c in gen_cols if 'tmb' in c.lower() or 'TMB' in c]
    gene_cols = [c for c in gen_cols if c not in tmb_col]
    mut = pd.read_csv(os.path.join(CPTAC, 'data_mutations.txt'), sep='\t',
                      low_memory=False, usecols=lambda c: c in
                      ['Hugo_Symbol', 'Tumor_Sample_Barcode', 'Variant_Classification'])
    # keep nonsyn
    keep_vc = {'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
               'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins',
               'Splice_Site', 'Translation_Start_Site', 'Nonstop_Mutation'}
    mut = mut[mut['Variant_Classification'].isin(keep_vc)]
    log(f"CPTAC nonsyn mutations: {len(mut)}")

    cptac_gen = pd.DataFrame(0, index=cptac_samples, columns=gen_cols, dtype=np.float32)
    for s in cptac_samples:
        sub = mut[mut['Tumor_Sample_Barcode'] == s]
        for g in sub['Hugo_Symbol'].unique():
            if g in gene_cols:
                cptac_gen.loc[s, g] = 1.0
        if tmb_col:
            cptac_gen.loc[s, tmb_col[0]] = float(len(sub))
    # Log mutation coverage
    mut_rate = (cptac_gen[gene_cols].sum(axis=0) > 0).mean()
    log(f"Fraction of training panel genes mutated at least once in CPTAC: {mut_rate:.2%}")

    # Transcriptomic: align to tra_cols
    cptac_tra = pd.DataFrame(0.0, index=cptac_samples, columns=tra_cols, dtype=np.float32)
    missing = []
    for g in tra_cols:
        if g in mrna.index:
            cptac_tra[g] = mrna.loc[g, cptac_samples].astype(float).values
        else:
            missing.append(g)
    log(f"Transcriptomic genes missing in CPTAC: {len(missing)} / {len(tra_cols)}")
    # Fill NaN from per-gene coerce before transform; training used log1p(max(x,0)).
    cptac_tra_arr = np.nan_to_num(cptac_tra.values.astype(np.float32), nan=0.0)
    cptac_tra_arr = np.log1p(np.maximum(cptac_tra_arr, 0))

    # Proteomic: zero (modality unavailable in matched form)
    cptac_pro_arr = np.zeros((len(cptac_samples), len(pro_cols)), dtype=np.float32)

    # =========================================================================
    # 2. Load scalers from fold-1 TCGA training set & transform CPTAC
    # =========================================================================
    ic50_df = pd.read_csv(os.path.join(TCGA, 'predicted_IC50_all_drugs.csv'), index_col=0)
    common = sorted(set(gen_tcga['patient_id']) & set(tra_tcga['patient_id']) &
                    set(pro_tcga['patient_id']) & set(ic50_df.index))
    histo_ids = {f.replace('.pt', '') for f in os.listdir(HISTO_DIR) if f.endswith('.pt')}
    common = sorted(set(common) & histo_ids)
    # fold 1 uses first 4/5 of common as training (in KFold order) — re-instantiate dataset to fit
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(kf.split(common))
    tr_ids = [common[i] for i in tr_idx]
    tr_ds = MultiDrugDataset4Modal(tr_ids, gen_tcga, tra_tcga, pro_tcga, ic50_df, DRUGS,
                                   histo_dir=HISTO_DIR, fit=True)

    gen_arr = np.nan_to_num(cptac_gen[gen_cols].values.astype(np.float32), nan=0.0)
    gen_sc = tr_ds.scalers['gen'].transform(gen_arr)
    tra_sc = tr_ds.scalers['tra'].transform(cptac_tra_arr)
    pro_sc = tr_ds.scalers['pro'].transform(cptac_pro_arr)
    log(f"Scaled CPTAC: gen={gen_sc.shape} trans={tra_sc.shape} prot={pro_sc.shape}")

    # =========================================================================
    # 3. Inference across all 5 fold models, average
    # =========================================================================
    config = get_default_config(genomic_dim=len(gen_cols), n_pathways=len(tra_cols),
                                proteomic_dim=len(pro_cols), n_drugs=len(DRUGS),
                                use_histology=True)
    config['task'] = 'regression'; config['hidden_dim'] = 256
    all_preds = []
    for fold in range(1, 6):
        model = PathOmicDRP(config).to(DEVICE)
        state = torch.load(os.path.join(FOLDS, f"fold{fold}_model.pt"),
                           map_location=DEVICE, weights_only=True)
        model.load_state_dict(state); model.eval()
        g = torch.tensor(gen_sc, dtype=torch.float32, device=DEVICE)
        t = torch.tensor(tra_sc, dtype=torch.float32, device=DEVICE)
        p = torch.tensor(pro_sc, dtype=torch.float32, device=DEVICE)
        # zero proteomic + skip histology -> pass histology=None
        with torch.no_grad():
            # batch through to avoid OOM
            preds = []
            for i in range(0, len(g), 8):
                o = model(g[i:i+8], t[i:i+8], p[i:i+8], histology=None, histo_mask=None)
                preds.append(o['prediction'].cpu().numpy())
            P = np.concatenate(preds, axis=0)
        P = tr_ds.scalers['ic50'].inverse_transform(P)
        all_preds.append(P)
    pred_mean = np.mean(np.stack(all_preds, axis=0), axis=0)
    pred_df = pd.DataFrame(pred_mean, columns=DRUGS, index=cptac_samples)
    pred_df.index.name = 'sample_id'
    pred_df.to_csv(os.path.join(OUT, 'cptac_predicted_IC50.csv'))
    log(f"CPTAC predictions: {pred_df.shape}")

    # =========================================================================
    # 4. Biomarker concordance
    # =========================================================================
    clin_sub = clin.set_index('sample_id').loc[cptac_samples]
    biomarkers = {}

    def mw_test(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if len(a) < 3 or len(b) < 3: return None, None
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        return float(stat), float(p)

    for col_name, drug_list, expect in [
        ('ER_UPDATED_CLINICAL_STATUS', ['Tamoxifen_1199', 'Fulvestrant_1816'], 'ER+_lower'),
        ('ERBB2_UPDATED_CLINICAL_STATUS', ['Lapatinib_1558'], 'HER2+_lower'),
        ('TNBC_UPDATED_CLINICAL_STATUS', ['Cisplatin_1005', 'Paclitaxel_1080'], 'TNBC_lower'),
    ]:
        if col_name not in clin_sub.columns:
            log(f"  missing {col_name}"); continue
        vals = clin_sub[col_name].astype(str)
        if 'ER_' in col_name and 'ERBB2' not in col_name:
            pos = vals.str.lower().eq('positive')
            neg = vals.str.lower().eq('negative')
            grp_labels = ('ER+', 'ER-')
        elif 'ERBB2' in col_name:
            pos = vals.str.lower().eq('positive')
            neg = vals.str.lower().eq('negative')
            grp_labels = ('HER2+', 'HER2-')
        elif 'TNBC' in col_name:
            pos = vals.str.lower().eq('positive')
            neg = vals.str.lower().eq('negative')
            grp_labels = ('TNBC', 'non-TNBC')
        for drug in drug_list:
            a = pred_mean[pos.values, DRUGS.index(drug)]
            b = pred_mean[neg.values, DRUGS.index(drug)]
            _, p = mw_test(a, b)
            delta = float(np.nanmean(a) - np.nanmean(b)) if len(a) and len(b) else None
            biomarkers[f"{col_name} | {drug}"] = {
                'expected': expect,
                f"mean_{grp_labels[0]}": float(np.nanmean(a)) if len(a) else None,
                f"n_{grp_labels[0]}": int(len(a)),
                f"mean_{grp_labels[1]}": float(np.nanmean(b)) if len(b) else None,
                f"n_{grp_labels[1]}": int(len(b)),
                'delta': delta,
                'mannwhitney_p': p,
                'concordant_direction': bool((delta or 0) < 0) if expect.endswith('_lower') else None,
            }
            d_str = f"{delta:+.3f}" if delta is not None else "n/a"
            p_str = f"{p:.3g}" if p is not None else "n/a"
            log(f"  {col_name:40s} {drug:18s}: n=({len(a)},{len(b)}) delta={d_str} p={p_str}")

    with open(os.path.join(OUT, 'biomarker_concordance.json'), 'w') as f:
        json.dump(biomarkers, f, indent=2)

    # =========================================================================
    # 5. Drug-drug correlation CPTAC vs TCGA
    # =========================================================================
    tcga_oof = pd.read_csv(os.path.join(OOF_DIR, 'oof_predictions.csv')).set_index('patient_id')
    tcga_mat = tcga_oof[[f"pred_{d}" for d in DRUGS]].values  # (431, 13)
    # Spearman drug-drug correlation matrix on each cohort
    def drug_corr(M):
        K = M.shape[1]
        C = np.eye(K)
        for i in range(K):
            for j in range(i+1, K):
                r, _ = spearmanr(M[:, i], M[:, j])
                C[i, j] = C[j, i] = r
        return C
    C_cptac = drug_corr(pred_mean)
    C_tcga = drug_corr(tcga_mat)
    r_corr, p_corr = pearsonr(C_cptac[np.triu_indices(len(DRUGS), k=1)],
                              C_tcga[np.triu_indices(len(DRUGS), k=1)])
    pd.DataFrame(C_cptac, index=DRUGS, columns=DRUGS).to_csv(
        os.path.join(OUT, 'drug_corr_cptac.csv'))
    pd.DataFrame(C_tcga, index=DRUGS, columns=DRUGS).to_csv(
        os.path.join(OUT, 'drug_corr_tcga.csv'))
    log(f"Drug-drug corr-of-corrs (CPTAC vs TCGA): Pearson r={r_corr:.3f} p={p_corr:.3g}")

    # =========================================================================
    # 6. Proteogenomic confounding spot-check
    # =========================================================================
    confounder_cols = ['ESTIMATE_TUMORPURITY', 'ESTIMATE_IMMUNE_SCORE',
                       'ESTIMATE_STROMAL_SCORE', 'APOBEC_SIGNATURE',
                       'STEMNESS_SCORE']
    confound = {}
    for c in confounder_cols:
        if c not in clin_sub.columns: continue
        x = pd.to_numeric(clin_sub[c], errors='coerce').values
        for drug in DRUGS:
            y = pred_df[drug].values
            m = ~np.isnan(x) & ~np.isnan(y)
            if m.sum() < 20: continue
            r, p = spearmanr(x[m], y[m])
            confound[f"{c} | {drug}"] = {'spearman_r': float(r), 'p': float(p), 'n': int(m.sum())}
    # flag strong confounders
    flagged = {k: v for k, v in confound.items() if abs(v['spearman_r']) > 0.5 and v['p'] < 0.01}

    summary = {
        'n_cptac_samples': len(cptac_samples),
        'gene_panel_coverage': float(mut_rate),
        'transcriptomic_missing': len(missing),
        'drug_corr_of_corrs': {'pearson_r': float(r_corr), 'p': float(p_corr)},
        'n_biomarkers_concordant_direction': int(sum(
            1 for v in biomarkers.values() if v.get('concordant_direction') is True)),
        'n_biomarkers_tested': len(biomarkers),
        'n_strong_confounders_flagged': len(flagged),
        'strong_confounders_examples': list(flagged.keys())[:5],
    }
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"Summary: {summary}")


if __name__ == '__main__':
    main()
