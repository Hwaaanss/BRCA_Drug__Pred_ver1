#!/usr/bin/env python3
"""Fair embedding comparison + bootstrap CIs for clinical AUC.

Addresses two reviewer concerns:
  (a) PathOmicDRP embedding (256-dim) vs baselines using raw high-dim features.
      Compare against PCA-256 reduction of raw multi-modal features to isolate
      "multi-modal fusion" effect from "low-dim representation" effect.
  (b) Surface bootstrap 95% CIs for clinical AUCs (small n_neg=3–4 cases).

Output: results/reinforce/fair_embedding_and_bootstrap.json
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[2])).resolve()
FINAL_ROOT = PROJECT_ROOT / "FINAL"
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import PathOmicDRP, get_default_config
from train_phase3_4modal import MultiDrugDataset4Modal, collate_4modal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = PROJECT_ROOT
INT = DATA_ROOT / "07_integrated"
HISTO_DIR = DATA_ROOT / "05_morphology" / "features"
OUT_DIR = FINAL_ROOT / "results" / "reinforce"
os.makedirs(OUT_DIR, exist_ok=True)

DRUGS = [
    'Cisplatin_1005','Docetaxel_1007','Paclitaxel_1080','Gemcitabine_1190',
    'Tamoxifen_1199','Lapatinib_1558','Vinblastine_1004','OSI-027_1594',
    'Daporinad_1248','Venetoclax_1909','ABT737_1910','AZD5991_1720',
    'Fulvestrant_1816',
]
CLINICAL_DRUGS = ['Docetaxel','Paclitaxel','Cyclophosphamide','Doxorubicin','Tamoxifen']

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_clinical_labels(drug_df, drug_name, valid_pids):
    ps = set(valid_pids)
    treated = drug_df[drug_df['therapeutic_agents'].str.contains(drug_name, case=False, na=False)]
    labels = {}
    for _, row in treated.iterrows():
        pid = row['submitter_id']
        if pid not in ps: continue
        o = row['treatment_outcome']
        if o in ('Complete Response','Partial Response'): labels[pid] = 1
        elif o in ('Progressive Disease','Stable Disease'): labels[pid] = 0
        elif o == 'Treatment Ongoing' and drug_name == 'Tamoxifen': labels[pid] = 1
    return labels


def cv_auc(X, y, n_splits=None, seed=42):
    n_neg = int(len(y) - np.sum(y)); n_pos = int(np.sum(y))
    if n_splits is None:
        n_splits = min(5, n_neg, n_pos)
    if n_splits < 2:
        return None, None, None, None
    aucs = []; y_all_t=[]; y_all_p=[]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr,te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.1)
        clf.fit(Xtr, y[tr])
        pr = clf.predict_proba(Xte)[:,1]
        try: a = roc_auc_score(y[te], pr)
        except: a = 0.5
        aucs.append(a); y_all_t.extend(y[te].tolist()); y_all_p.extend(pr.tolist())
    return float(np.mean(aucs)), float(np.std(aucs)), np.array(y_all_t), np.array(y_all_p)


def bootstrap_auc_ci(y_true, y_pred, n_iter=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(set(yt)) < 2: continue
        try: aucs.append(roc_auc_score(yt, yp))
        except: pass
    if not aucs: return None
    aucs = np.array(aucs)
    return {
        'bootstrap_mean': float(aucs.mean()),
        'bootstrap_ci95': [float(np.percentile(aucs,2.5)), float(np.percentile(aucs,97.5))],
        'n_iter_valid': len(aucs),
    }


def main():
    log("Loading data…")
    gen_df = pd.read_csv(f"{INT}/X_genomic.csv")
    tra_df = pd.read_csv(f"{INT}/X_transcriptomic.csv")
    pro_df = pd.read_csv(f"{INT}/X_proteomic.csv")
    ic50_df = pd.read_csv(f"{INT}/predicted_IC50_all_drugs.csv", index_col=0)
    drug_df = pd.read_csv(f"{BASE}/01_clinical/TCGA_BRCA_drug_treatments.csv")

    # 4-modal intersection
    common = sorted(set(gen_df['patient_id']) & set(tra_df['patient_id'])
                    & set(pro_df['patient_id']) & set(ic50_df.index))
    histo_ids = {f.replace('.pt','') for f in os.listdir(HISTO_DIR) if f.endswith('.pt')}
    common = sorted(set(common) & histo_ids)
    pid_to_idx = {p:i for i,p in enumerate(common)}
    log(f"4-modal patients: {len(common)}")

    # Assemble raw feature matrix: [genomic | transcriptomic log1p | proteomic | histology mean]
    gen = gen_df.set_index('patient_id').loc[common].values.astype(np.float32)
    tra_raw = tra_df.set_index('patient_id').loc[common].values.astype(np.float32)
    tra = np.log1p(np.maximum(tra_raw, 0))
    pro = pro_df.set_index('patient_id').loc[common].values.astype(np.float32)
    # Top-2000 variance genes to match model input (tra df already has 2000 cols? check)
    log(f"  raw dims: gen={gen.shape[1]} tra={tra.shape[1]} pro={pro.shape[1]}")

    log("Loading histology mean features …")
    histo_mean = np.zeros((len(common), 1024), dtype=np.float32)
    for i, pid in enumerate(common):
        pt = os.path.join(HISTO_DIR, f"{pid}.pt")
        if os.path.exists(pt):
            f = torch.load(pt, map_location='cpu', weights_only=True)
            histo_mean[i] = f.mean(dim=0).numpy()

    # Concatenate omics+histo raw
    X_raw = np.concatenate([gen, tra, pro, histo_mean], axis=1)
    log(f"  X_raw: {X_raw.shape}")

    # Z-score global, then PCA-256
    sc = StandardScaler()
    X_raw_std = sc.fit_transform(X_raw)
    pca256 = PCA(n_components=256, random_state=42)
    X_pca256 = pca256.fit_transform(X_raw_std)
    log(f"  PCA-256 variance captured: {pca256.explained_variance_ratio_.sum():.3f}")

    # Omics-only PCA-256 for ablation
    X_omics = np.concatenate([gen, tra, pro], axis=1)
    X_omics_std = StandardScaler().fit_transform(X_omics)
    pca256_o = PCA(n_components=256, random_state=42)
    X_pca256_omics = pca256_o.fit_transform(X_omics_std)

    # PathOmicDRP imputed IC50 predictions (13 drugs) — use imputed IC50 as features
    ic50_feat = ic50_df.loc[common, DRUGS].values.astype(np.float32)

    # Load PathOmicDRP 4-modal embeddings from fold models (use fold1 if cv_ablation done) else best
    emb_source = None
    for path in [f"{OUT_DIR}/fold1_model.pt",
                 f"{BASE}/results/phase3_4modal_full/best_model.pt"]:
        if os.path.exists(path):
            emb_source = path; break
    log(f"Using model weights: {emb_source}")

    gen_dim = gen.shape[1]; tra_dim = tra.shape[1]; pro_dim = pro.shape[1]
    cfg = get_default_config(genomic_dim=gen_dim, n_pathways=tra_dim,
                             proteomic_dim=pro_dim, n_drugs=len(DRUGS),
                             use_histology=True)
    cfg['task']='regression'; cfg['modality_dropout']=0.1; cfg['hidden_dim']=256

    model = PathOmicDRP(cfg).to(DEVICE)
    state = torch.load(emb_source, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state); model.eval()

    # Extract 256-dim embeddings from fusion.self_attn mean-pool
    ds = MultiDrugDataset4Modal(common, gen_df, tra_df, pro_df, ic50_df, DRUGS,
                                histo_dir=HISTO_DIR, fit=True)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_4modal)
    embs = []
    def hook_fn(m,i,o): embs.append(o.detach().cpu())
    handle = model.fusion.self_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        for batch in loader:
            g = batch['genomic'].to(DEVICE); t = batch['transcriptomic'].to(DEVICE)
            p = batch['proteomic'].to(DEVICE)
            kw = {}
            if 'histology' in batch:
                kw['histology']=batch['histology'].to(DEVICE); kw['histo_mask']=batch['histo_mask'].to(DEVICE)
            model(g, t, p, **kw)
    handle.remove()
    X_pathomic = torch.cat([e.mean(dim=1) for e in embs], dim=0).numpy()
    log(f"  PathOmicDRP embedding: {X_pathomic.shape}")

    # Evaluate across clinical drugs
    method_sets = {
        'PathOmicDRP_embedding_256d': X_pathomic,
        'PCA256_4modal_raw':          X_pca256,        # fair comparison: same dim, no learned fusion
        'PCA256_omics_only':          X_pca256_omics,  # isolates histology contribution
        'Raw_4modal':                 X_raw,           # baseline high-dim
        'ImputedIC50_13d':            ic50_feat,       # oncoPredict-based linear features
    }

    results = {'methods': list(method_sets.keys()), 'drugs': {}}
    for drug in CLINICAL_DRUGS:
        labels = get_clinical_labels(drug_df, drug, common)
        valid_pids = sorted(labels.keys())
        if len(valid_pids) < 10:
            log(f"  {drug}: only {len(valid_pids)} labeled — skip"); continue
        idx = np.array([pid_to_idx[p] for p in valid_pids])
        y = np.array([labels[p] for p in valid_pids], dtype=int)
        n_pos = int(y.sum()); n_neg = int(len(y)-y.sum())
        log(f"\n{drug}: n={len(y)} (pos={n_pos}, neg={n_neg})")
        if n_pos < 3 or n_neg < 3:
            log("  insufficient — skip"); continue

        drug_res = {'n': len(y), 'n_pos': n_pos, 'n_neg': n_neg, 'methods': {}}
        for name, X in method_sets.items():
            Xd = X[idx]
            auc_mean, auc_std, y_t, y_p = cv_auc(Xd, y, seed=42)
            entry = {'auc_mean': auc_mean, 'auc_std': auc_std}
            if y_t is not None:
                entry.update(bootstrap_auc_ci(y_t, y_p, n_iter=2000, seed=42))
            drug_res['methods'][name] = entry
            log(f"  {name:30s}: AUC={auc_mean:.3f} ± {auc_std:.3f}  "
                f"95% CI [{entry.get('bootstrap_ci95',[None,None])[0]}, "
                f"{entry.get('bootstrap_ci95',[None,None])[1]}]")
        results['drugs'][drug] = drug_res

    # Aggregate mean AUC per method
    mean_aucs = {m: [] for m in method_sets}
    for drug, dr in results['drugs'].items():
        for m, e in dr['methods'].items():
            if e.get('auc_mean') is not None:
                mean_aucs[m].append(e['auc_mean'])
    results['mean_auc_across_drugs'] = {m: float(np.mean(v)) if v else None
                                         for m,v in mean_aucs.items()}
    log("\n=== Mean AUC across clinical drugs ===")
    for m,v in results['mean_auc_across_drugs'].items():
        log(f"  {m:30s}: {v}")

    with open(f"{OUT_DIR}/fair_embedding_and_bootstrap.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Saved {OUT_DIR}/fair_embedding_and_bootstrap.json")


if __name__ == '__main__':
    main()
