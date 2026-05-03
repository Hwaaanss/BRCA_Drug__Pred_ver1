#!/usr/bin/env python3
"""
PathOmicDRP: W1/W2/W3 Weakness Resolution Analyses
====================================================
W1: Expanded clinical validation + Permutation test + Bootstrap CIs
W2: Self-Attention Only model retrained at 100 epochs + clinical AUC extraction
W3: Histopathology value via clinical AUC comparison (3-modal vs 4-modal)

All results saved to results/strengthening/
"""

import os, sys, json, time, warnings, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, mannwhitneyu
from itertools import combinations

warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import PathOmicDRP, get_default_config
from architecture_comparison import SelfAttnOnly
from train_phase3_4modal import MultiDrugDataset4Modal, collate_4modal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = PROJECT_ROOT
HISTO_DIR = DATA_ROOT / "05_morphology" / "features"
RESULTS_DIR = PROJECT_ROOT / "results" / "strengthening"
FIG_DIR = PROJECT_ROOT / "research" / "figures" / "figures_v3"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NpEncoder)
    log(f"  Saved: {path}")


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load all data needed across analyses."""
    log("Loading data...")
    gen_df = pd.read_csv(f"{BASE}/07_integrated/X_genomic.csv")
    tra_df = pd.read_csv(f"{BASE}/07_integrated/X_transcriptomic.csv")
    pro_df = pd.read_csv(f"{BASE}/07_integrated/X_proteomic.csv")
    ic50_df = pd.read_csv(f"{BASE}/07_integrated/predicted_IC50_all_drugs.csv", index_col=0)
    drug_df = pd.read_csv(f"{BASE}/01_clinical/TCGA_BRCA_drug_treatments.csv")

    hids = {f.replace('.pt','') for f in os.listdir(HISTO_DIR) if f.endswith('.pt')}

    # 431 patients with all 4 modalities
    pids_4modal = sorted(
        set(gen_df['patient_id']) & set(tra_df['patient_id']) &
        set(pro_df['patient_id']) & set(ic50_df.index) & hids
    )

    # Load config and drug cols from saved results
    with open(f"{BASE}/results/phase3_4modal_full/cv_results.json") as f:
        cv = json.load(f)
    config = cv['config']
    drug_cols = cv['drugs']
    drug_names = [d.rsplit('_',1)[0] for d in drug_cols]

    log(f"  4-modal patients: {len(pids_4modal)}")
    log(f"  Drugs: {drug_names}")

    return {
        'gen_df': gen_df, 'tra_df': tra_df, 'pro_df': pro_df,
        'ic50_df': ic50_df, 'drug_df': drug_df,
        'hids': hids, 'pids_4modal': pids_4modal,
        'config': config, 'drug_cols': drug_cols, 'drug_names': drug_names,
    }


# =============================================================================
# HELPER: Get clinical labels for a drug
# =============================================================================
def get_clinical_labels(drug_df, drug_name, valid_pids):
    """Get binary clinical outcome labels for a drug."""
    pid_set = set(valid_pids)
    treated = drug_df[drug_df['therapeutic_agents'].str.contains(drug_name, case=False, na=False)]
    labels = {}
    for _, row in treated.iterrows():
        pid = row['submitter_id']
        if pid not in pid_set:
            continue
        outcome = row['treatment_outcome']
        if outcome in ('Complete Response', 'Partial Response'):
            labels[pid] = 1
        elif outcome in ('Progressive Disease', 'Stable Disease'):
            labels[pid] = 0
        elif outcome == 'Treatment Ongoing' and drug_name == 'Tamoxifen':
            labels[pid] = 1
    return labels


# =============================================================================
# HELPER: Compute clinical AUC with cross-validation
# =============================================================================
def compute_clinical_auc_cv(X, y, n_splits=None, random_state=42):
    """Compute clinical AUC via stratified CV of LogisticRegression."""
    n_neg = int(len(y) - np.sum(y))
    n_pos = int(np.sum(y))
    if n_splits is None:
        n_splits = min(5, n_neg)
    if n_splits < 2:
        return None, None

    aucs = []
    y_true_all, y_pred_all = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1)
        clf.fit(Xtr, y[tr])
        try:
            proba = clf.predict_proba(Xte)[:,1]
            aucs.append(roc_auc_score(y[te], proba))
            y_true_all.extend(y[te].tolist())
            y_pred_all.extend(proba.tolist())
        except:
            aucs.append(0.5)

    return np.mean(aucs), (np.array(y_true_all), np.array(y_pred_all))


# =============================================================================
# HELPER: Extract embeddings from a model
# =============================================================================
def extract_embeddings(model, dataset, hook_target, use_histo=True):
    """Extract embeddings from a model using a forward hook."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_4modal)
    embs = []
    def hook_fn(m, i, o):
        embs.append(o.detach().cpu())
    handle = hook_target.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            g = batch['genomic'].to(DEVICE)
            t = batch['transcriptomic'].to(DEVICE)
            p = batch['proteomic'].to(DEVICE)
            kw = {}
            if use_histo and 'histology' in batch:
                kw['histology'] = batch['histology'].to(DEVICE)
                kw['histo_mask'] = batch['histo_mask'].to(DEVICE)
            model(g, t, p, **kw)
    handle.remove()
    return torch.cat([e.mean(dim=1) for e in embs], dim=0).numpy()


# =============================================================================
# HELPER: Train a model (SelfAttnOnly or PathOmicDRP)
# =============================================================================
def train_model(model, train_ds, val_ds, n_epochs=100, lr=3e-4, patience=15):
    """Train model and return best state dict."""
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0,
                              collate_fn=collate_4modal, drop_last=len(train_ds) > 16)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0,
                            collate_fn=collate_4modal)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.01)
    criterion = nn.HuberLoss(delta=1.0)

    best_loss = float('inf')
    best_state = None
    pat_count = 0

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            g = batch['genomic'].to(DEVICE)
            t = batch['transcriptomic'].to(DEVICE)
            p = batch['proteomic'].to(DEVICE)
            y = batch['target'].to(DEVICE)
            kw = {}
            if 'histology' in batch:
                kw['histology'] = batch['histology'].to(DEVICE)
                kw['histo_mask'] = batch['histo_mask'].to(DEVICE)
            optimizer.zero_grad()
            out = model(g, t, p, **kw)['prediction']
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                g = batch['genomic'].to(DEVICE)
                t = batch['transcriptomic'].to(DEVICE)
                p = batch['proteomic'].to(DEVICE)
                y = batch['target'].to(DEVICE)
                kw = {}
                if 'histology' in batch:
                    kw['histology'] = batch['histology'].to(DEVICE)
                    kw['histo_mask'] = batch['histo_mask'].to(DEVICE)
                out = model(g, t, p, **kw)['prediction']
                vloss += criterion(out, y).item() * len(y)
                vn += len(y)
        vl = vloss / vn
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_count = 0
        else:
            pat_count += 1
        if pat_count >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE).eval()

    # Compute validation PCC
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            g = batch['genomic'].to(DEVICE)
            t = batch['transcriptomic'].to(DEVICE)
            p = batch['proteomic'].to(DEVICE)
            kw = {}
            if 'histology' in batch:
                kw['histology'] = batch['histology'].to(DEVICE)
                kw['histo_mask'] = batch['histo_mask'].to(DEVICE)
            out = model(g, t, p, **kw)['prediction'].cpu().numpy()
            pred_o = train_ds.scalers['ic50'].inverse_transform(out)
            true_o = train_ds.scalers['ic50'].inverse_transform(batch['target'].numpy())
            all_pred.append(pred_o)
            all_true.append(true_o)

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    pcc_g, _ = pearsonr(all_pred.flatten(), all_true.flatten())
    drug_pccs = []
    for d in range(all_pred.shape[1]):
        try:
            r, _ = pearsonr(all_pred[:,d], all_true[:,d])
            drug_pccs.append(r)
        except:
            drug_pccs.append(0)

    return best_state, float(pcc_g), float(np.mean(drug_pccs))


# =============================================================================
# W1: EXPANDED CLINICAL VALIDATION + PERMUTATION TEST + BOOTSTRAP CI
# =============================================================================
def analysis_w1(data):
    log("=" * 70)
    log("W1: EXPANDED CLINICAL VALIDATION + PERMUTATION + BOOTSTRAP")
    log("=" * 70)

    pids = data['pids_4modal']
    drug_df = data['drug_df']
    config = data['config']
    drug_cols = data['drug_cols']
    drug_names = data['drug_names']
    gen_df, tra_df, pro_df, ic50_df = data['gen_df'], data['tra_df'], data['pro_df'], data['ic50_df']

    # ---- Step 1: Load PathOmicDRP 4-modal model and extract embeddings ----
    log("  Loading PathOmicDRP 4-modal model...")
    model_4m = PathOmicDRP(config).to(DEVICE)
    state = torch.load(f"{BASE}/results/phase3_4modal_full/best_model.pt",
                       map_location=DEVICE, weights_only=True)
    model_4m.load_state_dict(state)
    model_4m.eval()

    dataset = MultiDrugDataset4Modal(pids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
                                      histo_dir=HISTO_DIR, fit=True)

    log("  Extracting 4-modal embeddings...")
    emb4 = extract_embeddings(model_4m, dataset, model_4m.fusion.self_attn, use_histo=True)
    log(f"  4-modal embeddings shape: {emb4.shape}")

    log("  Extracting 3-modal embeddings (no histology)...")
    emb3 = extract_embeddings(model_4m, dataset, model_4m.fusion.self_attn, use_histo=False)
    log(f"  3-modal embeddings shape: {emb3.shape}")

    # Also load 3-modal baseline model
    log("  Loading PathOmicDRP 3-modal baseline model...")
    config_3m = {**config, 'use_histology': False}
    model_3m = PathOmicDRP(config_3m).to(DEVICE)
    state_3m = torch.load(f"{BASE}/results/phase3_3modal_baseline/best_model.pt",
                          map_location=DEVICE, weights_only=True)
    model_3m.load_state_dict(state_3m)
    model_3m.eval()

    dataset_3m = MultiDrugDataset4Modal(pids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
                                         histo_dir=None, fit=True)
    emb3_native = extract_embeddings(model_3m, dataset_3m, model_3m.fusion.self_attn, use_histo=False)
    log(f"  3-modal native embeddings shape: {emb3_native.shape}")

    pid_to_idx = {p: i for i, p in enumerate(pids)}

    # ---- Step 2: ElasticNet baseline on all 431 patients (omics -> IC50) ----
    log("  Training ElasticNet baseline (CV predictions as features)...")
    # Build omics feature matrix
    gen_set = gen_df.set_index('patient_id')
    tra_set = tra_df.set_index('patient_id')
    pro_set = pro_df.set_index('patient_id')

    gen_cols_list = [c for c in gen_set.columns if c != 'patient_id']
    tra_cols_list = [c for c in tra_set.columns if c != 'patient_id']
    pro_cols_list = [c for c in pro_set.columns if c != 'patient_id']

    X_omics = np.zeros((len(pids), len(gen_cols_list) + len(tra_cols_list) + len(pro_cols_list)), dtype=np.float32)
    for i, pid in enumerate(pids):
        g_vals = gen_set.loc[pid, gen_cols_list].values.astype(np.float32) if pid in gen_set.index else np.zeros(len(gen_cols_list))
        t_vals = np.log1p(np.maximum(tra_set.loc[pid, tra_cols_list].values.astype(np.float32), 0)) if pid in tra_set.index else np.zeros(len(tra_cols_list))
        p_vals = pro_set.loc[pid, pro_cols_list].values.astype(np.float32) if pid in pro_set.index else np.zeros(len(pro_cols_list))
        X_omics[i] = np.concatenate([g_vals, t_vals, p_vals])

    Y_ic50 = ic50_df.loc[pids, drug_cols].values.astype(np.float32)

    # Cross-validated ElasticNet IC50 predictions
    log("  Running 5-fold CV ElasticNet...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    elasticnet_preds = np.zeros_like(Y_ic50)
    for fold, (tri, vai) in enumerate(kf.split(pids)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_omics[tri])
        Xva = sc.transform(X_omics[vai])
        sc_y = StandardScaler()
        Ytr = sc_y.fit_transform(Y_ic50[tri])
        for d in range(len(drug_cols)):
            en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
            en.fit(Xtr, Ytr[:, d])
            elasticnet_preds[vai, d] = sc_y.inverse_transform(
                np.column_stack([en.predict(Xva) if dd == d else np.zeros(len(vai)) for dd in range(len(drug_cols))])
            )[:, d]

    log(f"  ElasticNet predictions shape: {elasticnet_preds.shape}")

    # ---- Step 3: Clinical AUC for each drug and method ----
    log("  Computing clinical AUCs...")
    clinical_drugs = ['Docetaxel', 'Paclitaxel', 'Cyclophosphamide', 'Tamoxifen', 'Doxorubicin']

    N_PERM = 10000
    N_BOOT = 10000

    results_w1 = {'drugs': {}, 'summary': {}}

    for drug_name in clinical_drugs:
        labels = get_clinical_labels(drug_df, drug_name, pids)
        valid = [p for p in labels if p in pid_to_idx]
        if len(valid) < 10:
            log(f"    {drug_name}: only {len(valid)} patients, skipping")
            continue

        y = np.array([labels[p] for p in valid])
        n_pos = int(y.sum())
        n_neg = int(len(y) - y.sum())
        if n_pos < 3 or n_neg < 2:
            log(f"    {drug_name}: n_pos={n_pos}, n_neg={n_neg}, skipping")
            continue

        drug_result = {'n': len(y), 'n_pos': n_pos, 'n_neg': n_neg, 'methods': {}}

        # Method 1: PathOmicDRP 4-modal embeddings
        X_4m = np.array([emb4[pid_to_idx[p]] for p in valid])
        auc_4m, (yt_4m, yp_4m) = compute_clinical_auc_cv(X_4m, y)

        # Method 2: PathOmicDRP 3-modal embeddings (from 4-modal model, no histo)
        X_3m = np.array([emb3[pid_to_idx[p]] for p in valid])
        auc_3m, (yt_3m, yp_3m) = compute_clinical_auc_cv(X_3m, y)

        # Method 3: 3-modal native model embeddings
        X_3m_native = np.array([emb3_native[pid_to_idx[p]] for p in valid])
        auc_3m_native, _ = compute_clinical_auc_cv(X_3m_native, y)

        # Method 4: ElasticNet IC50 predictions as features
        X_en = np.array([elasticnet_preds[pid_to_idx[p]] for p in valid])
        auc_en, (yt_en, yp_en) = compute_clinical_auc_cv(X_en, y)

        # Method 5: Raw omics features
        X_raw = np.array([X_omics[pid_to_idx[p]] for p in valid])
        auc_raw, _ = compute_clinical_auc_cv(X_raw, y)

        drug_result['methods'] = {
            'PathOmicDRP_4modal': {'auc': float(auc_4m) if auc_4m else None},
            'PathOmicDRP_3modal_nohisto': {'auc': float(auc_3m) if auc_3m else None},
            'PathOmicDRP_3modal_native': {'auc': float(auc_3m_native) if auc_3m_native else None},
            'ElasticNet_IC50': {'auc': float(auc_en) if auc_en else None},
            'Raw_omics': {'auc': float(auc_raw) if auc_raw else None},
        }

        # ---- Permutation test for PathOmicDRP 4-modal ----
        if auc_4m is not None and yt_4m is not None:
            log(f"    {drug_name}: Running {N_PERM} permutations for 4-modal...")
            perm_aucs = []
            for _ in range(N_PERM):
                y_perm = np.random.permutation(yt_4m)
                try:
                    perm_aucs.append(roc_auc_score(y_perm, yp_4m))
                except:
                    perm_aucs.append(0.5)
            perm_aucs = np.array(perm_aucs)
            p_value_4m = float(np.mean(perm_aucs >= auc_4m))
            drug_result['methods']['PathOmicDRP_4modal']['permutation_p'] = p_value_4m
            drug_result['methods']['PathOmicDRP_4modal']['perm_auc_null_mean'] = float(perm_aucs.mean())
            drug_result['methods']['PathOmicDRP_4modal']['perm_auc_null_std'] = float(perm_aucs.std())
            log(f"      4-modal AUC={auc_4m:.3f}, permutation p={p_value_4m:.4f}")

        # ---- Permutation test for ElasticNet ----
        if auc_en is not None and yt_en is not None:
            log(f"    {drug_name}: Running {N_PERM} permutations for ElasticNet...")
            perm_aucs_en = []
            for _ in range(N_PERM):
                y_perm = np.random.permutation(yt_en)
                try:
                    perm_aucs_en.append(roc_auc_score(y_perm, yp_en))
                except:
                    perm_aucs_en.append(0.5)
            perm_aucs_en = np.array(perm_aucs_en)
            p_value_en = float(np.mean(perm_aucs_en >= auc_en))
            drug_result['methods']['ElasticNet_IC50']['permutation_p'] = p_value_en
            log(f"      ElasticNet AUC={auc_en:.3f}, permutation p={p_value_en:.4f}")

        # ---- Bootstrap 95% CI for PathOmicDRP 4-modal ----
        if yt_4m is not None and len(yt_4m) > 0:
            log(f"    {drug_name}: Running {N_BOOT} bootstrap iterations for CI...")
            boot_aucs = []
            for _ in range(N_BOOT):
                idx = np.random.choice(len(yt_4m), size=len(yt_4m), replace=True)
                yb_true = yt_4m[idx]
                yb_pred = yp_4m[idx]
                if len(np.unique(yb_true)) < 2:
                    continue
                try:
                    boot_aucs.append(roc_auc_score(yb_true, yb_pred))
                except:
                    pass
            if boot_aucs:
                boot_aucs = np.array(boot_aucs)
                ci_low = float(np.percentile(boot_aucs, 2.5))
                ci_high = float(np.percentile(boot_aucs, 97.5))
                drug_result['methods']['PathOmicDRP_4modal']['bootstrap_ci_95'] = [ci_low, ci_high]
                drug_result['methods']['PathOmicDRP_4modal']['bootstrap_mean'] = float(boot_aucs.mean())
                log(f"      Bootstrap AUC: {boot_aucs.mean():.3f} [{ci_low:.3f}, {ci_high:.3f}]")

        # ---- Bootstrap CI for ElasticNet ----
        if yt_en is not None and len(yt_en) > 0:
            boot_aucs_en = []
            for _ in range(N_BOOT):
                idx = np.random.choice(len(yt_en), size=len(yt_en), replace=True)
                if len(np.unique(yt_en[idx])) < 2:
                    continue
                try:
                    boot_aucs_en.append(roc_auc_score(yt_en[idx], yp_en[idx]))
                except:
                    pass
            if boot_aucs_en:
                boot_aucs_en = np.array(boot_aucs_en)
                drug_result['methods']['ElasticNet_IC50']['bootstrap_ci_95'] = [
                    float(np.percentile(boot_aucs_en, 2.5)),
                    float(np.percentile(boot_aucs_en, 97.5))
                ]

        # ---- Bootstrap CI for 3-modal ----
        if yt_3m is not None and len(yt_3m) > 0:
            boot_aucs_3m = []
            for _ in range(N_BOOT):
                idx = np.random.choice(len(yt_3m), size=len(yt_3m), replace=True)
                if len(np.unique(yt_3m[idx])) < 2:
                    continue
                try:
                    boot_aucs_3m.append(roc_auc_score(yt_3m[idx], yp_3m[idx]))
                except:
                    pass
            if boot_aucs_3m:
                boot_aucs_3m = np.array(boot_aucs_3m)
                drug_result['methods']['PathOmicDRP_3modal_nohisto']['bootstrap_ci_95'] = [
                    float(np.percentile(boot_aucs_3m, 2.5)),
                    float(np.percentile(boot_aucs_3m, 97.5))
                ]

        results_w1['drugs'][drug_name] = drug_result
        log(f"    {drug_name}: n={len(y)} (pos={n_pos}, neg={n_neg})")
        for method, vals in drug_result['methods'].items():
            auc_str = f"{vals['auc']:.3f}" if vals['auc'] is not None else "N/A"
            p_str = f", p={vals.get('permutation_p','N/A')}" if 'permutation_p' in vals else ""
            ci_str = f", CI=[{vals['bootstrap_ci_95'][0]:.3f},{vals['bootstrap_ci_95'][1]:.3f}]" if 'bootstrap_ci_95' in vals else ""
            log(f"      {method:30s}: AUC={auc_str}{p_str}{ci_str}")

    # ---- Summary: Mean advantage across drugs ----
    drugs_with_both = [d for d in results_w1['drugs']
                       if results_w1['drugs'][d]['methods'].get('PathOmicDRP_4modal',{}).get('auc') is not None
                       and results_w1['drugs'][d]['methods'].get('ElasticNet_IC50',{}).get('auc') is not None]

    if drugs_with_both:
        auc_4m_list = [results_w1['drugs'][d]['methods']['PathOmicDRP_4modal']['auc'] for d in drugs_with_both]
        auc_en_list = [results_w1['drugs'][d]['methods']['ElasticNet_IC50']['auc'] for d in drugs_with_both]
        auc_3m_list = [results_w1['drugs'][d]['methods']['PathOmicDRP_3modal_nohisto']['auc'] for d in drugs_with_both]

        results_w1['summary'] = {
            'drugs_compared': drugs_with_both,
            'PathOmicDRP_4modal_mean_auc': float(np.mean(auc_4m_list)),
            'ElasticNet_mean_auc': float(np.mean(auc_en_list)),
            'PathOmicDRP_3modal_mean_auc': float(np.mean(auc_3m_list)),
            'mean_advantage_over_elasticnet': float(np.mean(np.array(auc_4m_list) - np.array(auc_en_list))),
            'mean_advantage_over_3modal': float(np.mean(np.array(auc_4m_list) - np.array(auc_3m_list))),
            'n_drugs_4modal_wins_vs_en': int(np.sum(np.array(auc_4m_list) > np.array(auc_en_list))),
            'n_drugs_4modal_wins_vs_3m': int(np.sum(np.array(auc_4m_list) > np.array(auc_3m_list))),
        }
        # Sign test p-value (binomial)
        from scipy.stats import binomtest
        n_wins = results_w1['summary']['n_drugs_4modal_wins_vs_en']
        n_total = len(drugs_with_both)
        results_w1['summary']['sign_test_p_vs_elasticnet'] = float(binomtest(n_wins, n_total, 0.5, alternative='greater').pvalue)

    save_json(results_w1, f"{RESULTS_DIR}/w1_clinical_validation.json")
    return results_w1


# =============================================================================
# W2: SELF-ATTENTION ONLY - RETRAIN AT 100 EPOCHS + CLINICAL AUC
# =============================================================================
def analysis_w2(data):
    log("=" * 70)
    log("W2: SELF-ATTENTION ONLY MODEL - 100 EPOCH TRAINING + CLINICAL AUC")
    log("=" * 70)

    pids = data['pids_4modal']
    drug_df = data['drug_df']
    config = data['config']
    drug_cols = data['drug_cols']
    drug_names = data['drug_names']
    gen_df, tra_df, pro_df, ic50_df = data['gen_df'], data['tra_df'], data['pro_df'], data['ic50_df']

    # ---- Step 1: Train SelfAttnOnly at 100 epochs, save per-fold embeddings ----
    log("  Training SelfAttnOnly (100 epochs, 5-fold CV)...")
    selfattn_config = {**config, 'use_histology': True}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_pcc_global = []
    fold_pcc_drug = []

    # We need per-patient embeddings for clinical AUC
    # Collect out-of-fold embeddings
    all_fold_embs = {}  # pid -> embedding

    for fold, (tri, vai) in enumerate(kf.split(pids)):
        log(f"    Fold {fold+1}/5...")
        tr_pids = [pids[i] for i in tri]
        va_pids = [pids[i] for i in vai]

        tr_ds = MultiDrugDataset4Modal(tr_pids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
                                        histo_dir=HISTO_DIR, fit=True)
        va_ds = MultiDrugDataset4Modal(va_pids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
                                        histo_dir=HISTO_DIR, scalers=tr_ds.scalers)

        model_sa = SelfAttnOnly(selfattn_config).to(DEVICE)
        best_state, pcc_g, pcc_d = train_model(model_sa, tr_ds, va_ds, n_epochs=100, patience=15)
        fold_pcc_global.append(pcc_g)
        fold_pcc_drug.append(pcc_d)
        log(f"      PCC_global={pcc_g:.4f}, PCC_drug={pcc_d:.4f}")

        # Extract val embeddings
        model_sa.load_state_dict(best_state)
        model_sa.to(DEVICE).eval()

        va_embs = extract_embeddings(model_sa, va_ds, model_sa.self_attn, use_histo=True)
        for j, pid in enumerate(va_pids):
            all_fold_embs[pid] = va_embs[j]

        del model_sa
        torch.cuda.empty_cache()

    selfattn_pcc_global = float(np.mean(fold_pcc_global))
    selfattn_pcc_drug = float(np.mean(fold_pcc_drug))
    log(f"  SelfAttnOnly 100ep: PCC_global={selfattn_pcc_global:.4f} +/- {np.std(fold_pcc_global):.4f}")
    log(f"  SelfAttnOnly 100ep: PCC_drug={selfattn_pcc_drug:.4f} +/- {np.std(fold_pcc_drug):.4f}")

    # ---- Step 2: Also retrain PathOmicDRP at 100 epochs for fair comparison ----
    log("  Retraining PathOmicDRP (100 epochs, 5-fold CV) for fair comparison...")
    pathomic_config = {**config, 'use_histology': True}

    pathomic_fold_pcc_global = []
    pathomic_fold_pcc_drug = []
    pathomic_fold_embs = {}

    for fold, (tri, vai) in enumerate(kf.split(pids)):
        log(f"    Fold {fold+1}/5...")
        tr_pids = [pids[i] for i in tri]
        va_pids = [pids[i] for i in vai]

        tr_ds = MultiDrugDataset4Modal(tr_pids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
                                        histo_dir=HISTO_DIR, fit=True)
        va_ds = MultiDrugDataset4Modal(va_pids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
                                        histo_dir=HISTO_DIR, scalers=tr_ds.scalers)

        model_po = PathOmicDRP(pathomic_config).to(DEVICE)
        best_state, pcc_g, pcc_d = train_model(model_po, tr_ds, va_ds, n_epochs=100, patience=15)
        pathomic_fold_pcc_global.append(pcc_g)
        pathomic_fold_pcc_drug.append(pcc_d)
        log(f"      PCC_global={pcc_g:.4f}, PCC_drug={pcc_d:.4f}")

        # Extract val embeddings
        model_po.load_state_dict(best_state)
        model_po.to(DEVICE).eval()

        va_embs = extract_embeddings(model_po, va_ds, model_po.fusion.self_attn, use_histo=True)
        for j, pid in enumerate(va_pids):
            pathomic_fold_embs[pid] = va_embs[j]

        del model_po
        torch.cuda.empty_cache()

    pathomic_pcc_global = float(np.mean(pathomic_fold_pcc_global))
    pathomic_pcc_drug = float(np.mean(pathomic_fold_pcc_drug))
    log(f"  PathOmicDRP 100ep: PCC_global={pathomic_pcc_global:.4f} +/- {np.std(pathomic_fold_pcc_global):.4f}")
    log(f"  PathOmicDRP 100ep: PCC_drug={pathomic_pcc_drug:.4f} +/- {np.std(pathomic_fold_pcc_drug):.4f}")

    # ---- Step 3: Clinical AUC comparison ----
    log("  Computing clinical AUC for SelfAttnOnly vs PathOmicDRP...")

    clinical_drugs = ['Docetaxel', 'Paclitaxel', 'Cyclophosphamide', 'Tamoxifen']

    results_w2 = {
        'selfattn_100ep': {
            'pcc_global': selfattn_pcc_global,
            'pcc_global_std': float(np.std(fold_pcc_global)),
            'pcc_drug': selfattn_pcc_drug,
            'pcc_drug_std': float(np.std(fold_pcc_drug)),
            'fold_pcc_drug': fold_pcc_drug,
        },
        'pathomic_100ep': {
            'pcc_global': pathomic_pcc_global,
            'pcc_global_std': float(np.std(pathomic_fold_pcc_global)),
            'pcc_drug': pathomic_pcc_drug,
            'pcc_drug_std': float(np.std(pathomic_fold_pcc_drug)),
            'fold_pcc_drug': pathomic_fold_pcc_drug,
        },
        'clinical_auc': {},
        'comparison_80ep_vs_100ep': {
            'selfattn_80ep_pcc_drug': 0.4538,
            'selfattn_100ep_pcc_drug': selfattn_pcc_drug,
            'pathomic_80ep_pcc_drug': 0.4204,
            'pathomic_100ep_pcc_drug': pathomic_pcc_drug,
            'note': 'Architecture comparison used 80 epochs; main experiment used 100 epochs'
        }
    }

    pid_to_idx = {p: i for i, p in enumerate(pids)}

    for drug_name in clinical_drugs:
        labels = get_clinical_labels(drug_df, drug_name, pids)
        valid_sa = [p for p in labels if p in all_fold_embs]
        valid_po = [p for p in labels if p in pathomic_fold_embs]
        valid = sorted(set(valid_sa) & set(valid_po))

        if len(valid) < 10:
            continue
        y = np.array([labels[p] for p in valid])
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        if n_pos < 3 or n_neg < 2:
            continue

        # SelfAttnOnly clinical AUC
        X_sa = np.array([all_fold_embs[p] for p in valid])
        auc_sa, _ = compute_clinical_auc_cv(X_sa, y)

        # PathOmicDRP clinical AUC
        X_po = np.array([pathomic_fold_embs[p] for p in valid])
        auc_po, _ = compute_clinical_auc_cv(X_po, y)

        results_w2['clinical_auc'][drug_name] = {
            'n': len(y), 'n_pos': n_pos, 'n_neg': n_neg,
            'SelfAttnOnly_auc': float(auc_sa) if auc_sa else None,
            'PathOmicDRP_auc': float(auc_po) if auc_po else None,
            'advantage': float(auc_po - auc_sa) if (auc_po and auc_sa) else None,
        }
        log(f"    {drug_name}: SelfAttn AUC={auc_sa:.3f}, PathOmicDRP AUC={auc_po:.3f}, "
            f"advantage={auc_po-auc_sa:+.3f}")

    # Summary
    drugs_with_both = [d for d in results_w2['clinical_auc']
                       if results_w2['clinical_auc'][d]['SelfAttnOnly_auc'] is not None
                       and results_w2['clinical_auc'][d]['PathOmicDRP_auc'] is not None]
    if drugs_with_both:
        sa_aucs = [results_w2['clinical_auc'][d]['SelfAttnOnly_auc'] for d in drugs_with_both]
        po_aucs = [results_w2['clinical_auc'][d]['PathOmicDRP_auc'] for d in drugs_with_both]
        results_w2['clinical_auc_summary'] = {
            'drugs': drugs_with_both,
            'SelfAttnOnly_mean_auc': float(np.mean(sa_aucs)),
            'PathOmicDRP_mean_auc': float(np.mean(po_aucs)),
            'mean_advantage': float(np.mean(np.array(po_aucs) - np.array(sa_aucs))),
            'n_drugs_pathomic_wins': int(np.sum(np.array(po_aucs) > np.array(sa_aucs))),
        }

    save_json(results_w2, f"{RESULTS_DIR}/w2_selfattn_comparison.json")
    return results_w2


# =============================================================================
# W3: HISTOPATHOLOGY VALUE VIA CLINICAL AUC (3-MODAL vs 4-MODAL)
# =============================================================================
def analysis_w3(data, w1_results=None):
    log("=" * 70)
    log("W3: HISTOPATHOLOGY VALUE - CLINICAL AUC COMPARISON")
    log("=" * 70)

    pids = data['pids_4modal']
    drug_df = data['drug_df']
    config = data['config']
    drug_cols = data['drug_cols']
    gen_df, tra_df, pro_df, ic50_df = data['gen_df'], data['tra_df'], data['pro_df'], data['ic50_df']

    # Use existing embeddings from W1 if available, otherwise recompute
    if w1_results is not None and 'drugs' in w1_results:
        log("  Using W1 results for 3-modal vs 4-modal comparison")

    # Load models and extract embeddings
    log("  Loading PathOmicDRP 4-modal model...")
    model_4m = PathOmicDRP(config).to(DEVICE)
    state = torch.load(f"{BASE}/results/phase3_4modal_full/best_model.pt",
                       map_location=DEVICE, weights_only=True)
    model_4m.load_state_dict(state)
    model_4m.eval()

    dataset_4m = MultiDrugDataset4Modal(pids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
                                         histo_dir=HISTO_DIR, fit=True)

    emb4 = extract_embeddings(model_4m, dataset_4m, model_4m.fusion.self_attn, use_histo=True)
    emb3 = extract_embeddings(model_4m, dataset_4m, model_4m.fusion.self_attn, use_histo=False)

    # Also load native 3-modal model
    config_3m = {**config, 'use_histology': False}
    model_3m = PathOmicDRP(config_3m).to(DEVICE)
    state_3m = torch.load(f"{BASE}/results/phase3_3modal_baseline/best_model.pt",
                          map_location=DEVICE, weights_only=True)
    model_3m.load_state_dict(state_3m)
    model_3m.eval()

    dataset_3m = MultiDrugDataset4Modal(pids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
                                         histo_dir=None, fit=True)
    emb3_native = extract_embeddings(model_3m, dataset_3m, model_3m.fusion.self_attn, use_histo=False)

    pid_to_idx = {p: i for i, p in enumerate(pids)}

    clinical_drugs = ['Docetaxel', 'Paclitaxel', 'Cyclophosphamide', 'Tamoxifen']

    N_BOOT = 10000
    results_w3 = {'drugs': {}, 'summary': {}}

    auc_diffs_4v3 = []  # For paired comparison

    for drug_name in clinical_drugs:
        labels = get_clinical_labels(drug_df, drug_name, pids)
        valid = [p for p in labels if p in pid_to_idx]
        if len(valid) < 10:
            continue

        y = np.array([labels[p] for p in valid])
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        if n_pos < 3 or n_neg < 2:
            continue

        X_4m = np.array([emb4[pid_to_idx[p]] for p in valid])
        X_3m = np.array([emb3[pid_to_idx[p]] for p in valid])
        X_3m_native = np.array([emb3_native[pid_to_idx[p]] for p in valid])

        auc_4m, (yt_4m, yp_4m) = compute_clinical_auc_cv(X_4m, y)
        auc_3m, (yt_3m, yp_3m) = compute_clinical_auc_cv(X_3m, y)
        auc_3m_native, _ = compute_clinical_auc_cv(X_3m_native, y)

        diff = float(auc_4m - auc_3m) if (auc_4m and auc_3m) else 0
        auc_diffs_4v3.append(diff)

        drug_result = {
            'n': len(y), 'n_pos': n_pos, 'n_neg': n_neg,
            'auc_4modal': float(auc_4m) if auc_4m else None,
            'auc_3modal_same_model': float(auc_3m) if auc_3m else None,
            'auc_3modal_native': float(auc_3m_native) if auc_3m_native else None,
            'improvement_4m_vs_3m': diff,
        }

        # Bootstrap CI on the AUC difference
        if yt_4m is not None and yt_3m is not None:
            boot_diffs = []
            for _ in range(N_BOOT):
                idx = np.random.choice(len(yt_4m), size=len(yt_4m), replace=True)
                yb_true = yt_4m[idx]
                if len(np.unique(yb_true)) < 2:
                    continue
                try:
                    ba4 = roc_auc_score(yb_true, yp_4m[idx])
                    ba3 = roc_auc_score(yt_3m[idx], yp_3m[idx])
                    boot_diffs.append(ba4 - ba3)
                except:
                    pass
            if boot_diffs:
                boot_diffs = np.array(boot_diffs)
                drug_result['diff_bootstrap_ci_95'] = [
                    float(np.percentile(boot_diffs, 2.5)),
                    float(np.percentile(boot_diffs, 97.5))
                ]
                drug_result['diff_bootstrap_mean'] = float(boot_diffs.mean())
                # Proportion of bootstrap samples where 4-modal > 3-modal
                drug_result['prob_4modal_better'] = float(np.mean(boot_diffs > 0))

        results_w3['drugs'][drug_name] = drug_result
        log(f"  {drug_name}: 4m={auc_4m:.3f}, 3m={auc_3m:.3f}, diff={diff:+.3f}")

    # Summary statistics
    if auc_diffs_4v3:
        mean_diff = float(np.mean(auc_diffs_4v3))
        # Cohen's d for paired differences
        if np.std(auc_diffs_4v3) > 0:
            cohens_d = mean_diff / np.std(auc_diffs_4v3)
        else:
            cohens_d = float('inf') if mean_diff > 0 else 0

        # Sign test
        n_positive = sum(1 for d in auc_diffs_4v3 if d > 0)
        n_total = len(auc_diffs_4v3)
        from scipy.stats import binomtest
        sign_test_p = float(binomtest(n_positive, n_total, 0.5, alternative='greater').pvalue)

        results_w3['summary'] = {
            'n_drugs': n_total,
            'mean_auc_improvement': mean_diff,
            'individual_improvements': auc_diffs_4v3,
            'cohens_d': float(cohens_d),
            'n_drugs_improved': n_positive,
            'sign_test_p': sign_test_p,
            'all_positive': n_positive == n_total,
        }
        log(f"\n  W3 SUMMARY:")
        log(f"  Mean AUC improvement (4m vs 3m): {mean_diff:+.3f}")
        log(f"  Cohen's d: {cohens_d:.3f}")
        log(f"  Drugs improved: {n_positive}/{n_total}")
        log(f"  Sign test p-value: {sign_test_p:.4f}")

    save_json(results_w3, f"{RESULTS_DIR}/w3_histopathology_value.json")
    return results_w3


# =============================================================================
# FIGURE: Combined visualization
# =============================================================================
def create_figures(w1_results, w2_results, w3_results):
    log("=" * 70)
    log("CREATING FIGURES")
    log("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---- Panel A: W1 - Clinical AUC comparison ----
    ax = axes[0]
    if w1_results and 'drugs' in w1_results:
        drugs = sorted(w1_results['drugs'].keys())
        methods = ['PathOmicDRP_4modal', 'PathOmicDRP_3modal_nohisto', 'ElasticNet_IC50']
        colors = ['#2196F3', '#FFC107', '#FF5722']
        labels_m = ['PathOmicDRP 4-modal', 'PathOmicDRP 3-modal', 'ElasticNet']

        x = np.arange(len(drugs))
        width = 0.25

        for mi, (method, color, label) in enumerate(zip(methods, colors, labels_m)):
            vals = []
            ci_lo = []
            ci_hi = []
            for d in drugs:
                v = w1_results['drugs'][d]['methods'].get(method, {}).get('auc', 0.5)
                vals.append(v if v else 0.5)
                ci = w1_results['drugs'][d]['methods'].get(method, {}).get('bootstrap_ci_95', None)
                if ci:
                    ci_lo.append(v - ci[0])
                    ci_hi.append(ci[1] - v)
                else:
                    ci_lo.append(0)
                    ci_hi.append(0)

            bars = ax.bar(x + mi*width, vals, width, label=label, color=color, alpha=0.85)
            if any(c > 0 for c in ci_hi):
                ax.errorbar(x + mi*width, vals, yerr=[ci_lo, ci_hi],
                           fmt='none', color='black', capsize=3, linewidth=1)

            # Add significance stars
            for i, d in enumerate(drugs):
                p = w1_results['drugs'][d]['methods'].get(method, {}).get('permutation_p', None)
                if p is not None and p < 0.05:
                    star = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                    ax.text(x[i] + mi*width, vals[i] + 0.02, star,
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x + width)
        ax.set_xticklabels(drugs, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Clinical AUC', fontsize=11)
        ax.set_title('A. Clinical Outcome Prediction', fontsize=12, fontweight='bold')
        ax.axhline(0.5, ls='--', color='gray', alpha=0.5, label='Random')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_ylim(0, 1.1)

    # ---- Panel B: W2 - Architecture comparison with clinical AUC ----
    ax = axes[1]
    if w2_results and 'clinical_auc' in w2_results:
        drugs = sorted(w2_results['clinical_auc'].keys())
        x = np.arange(len(drugs))
        width = 0.35

        sa_aucs = [w2_results['clinical_auc'][d].get('SelfAttnOnly_auc', 0.5) or 0.5 for d in drugs]
        po_aucs = [w2_results['clinical_auc'][d].get('PathOmicDRP_auc', 0.5) or 0.5 for d in drugs]

        ax.bar(x - width/2, sa_aucs, width, label='Self-Attn Only', color='#9C27B0', alpha=0.85)
        ax.bar(x + width/2, po_aucs, width, label='PathOmicDRP', color='#2196F3', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(drugs, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Clinical AUC', fontsize=11)
        ax.set_title('B. Cross-Attn vs Self-Attn Clinical AUC', fontsize=12, fontweight='bold')
        ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.1)

        # Add PCC comparison as text box
        sa_pcc = w2_results['selfattn_100ep']['pcc_drug']
        po_pcc = w2_results['pathomic_100ep']['pcc_drug']
        textstr = f'PCC_drug (100ep):\nSelf-Attn: {sa_pcc:.3f}\nPathOmicDRP: {po_pcc:.3f}'
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=8,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ---- Panel C: W3 - Histopathology improvement ----
    ax = axes[2]
    if w3_results and 'drugs' in w3_results:
        drugs = sorted(w3_results['drugs'].keys())
        diffs = [w3_results['drugs'][d].get('improvement_4m_vs_3m', 0) for d in drugs]

        colors_bar = ['#4CAF50' if d > 0 else '#F44336' for d in diffs]
        bars = ax.bar(range(len(drugs)), diffs, color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.5)

        # Add CI whiskers if available
        for i, d in enumerate(drugs):
            ci = w3_results['drugs'][d].get('diff_bootstrap_ci_95', None)
            if ci:
                ax.errorbar(i, diffs[i], yerr=[[diffs[i]-ci[0]], [ci[1]-diffs[i]]],
                           fmt='none', color='black', capsize=4, linewidth=1.5)

        ax.set_xticks(range(len(drugs)))
        ax.set_xticklabels(drugs, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('AUC Improvement (4-modal - 3-modal)', fontsize=10)
        ax.set_title('C. Histopathology Value for Clinical AUC', fontsize=12, fontweight='bold')
        ax.axhline(0, ls='-', color='black', linewidth=0.8)

        # Add summary text
        if 'summary' in w3_results:
            s = w3_results['summary']
            textstr = (f"Mean: {s['mean_auc_improvement']:+.3f}\n"
                      f"Cohen's d: {s['cohens_d']:.2f}\n"
                      f"Sign test: p={s['sign_test_p']:.3f}")
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    fig_path = f"{FIG_DIR}/w1w2w3_resolution.png"
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    log(f"  Saved figure: {fig_path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    log("Starting W1/W2/W3 Resolution Analyses")
    log(f"Device: {DEVICE}")

    data = load_data()

    ALL_RESULTS = {}

    # W1: Expanded clinical validation + permutation + bootstrap
    try:
        w1 = analysis_w1(data)
        ALL_RESULTS['W1'] = w1
    except Exception as e:
        log(f"W1 FAILED: {e}")
        traceback.print_exc()
        w1 = None

    # W2: SelfAttnOnly retrained + clinical AUC
    try:
        w2 = analysis_w2(data)
        ALL_RESULTS['W2'] = w2
    except Exception as e:
        log(f"W2 FAILED: {e}")
        traceback.print_exc()
        w2 = None

    # W3: Histopathology value via clinical AUC
    try:
        w3 = analysis_w3(data, w1_results=w1)
        ALL_RESULTS['W3'] = w3
    except Exception as e:
        log(f"W3 FAILED: {e}")
        traceback.print_exc()
        w3 = None

    # Create combined figure
    try:
        create_figures(w1, w2, w3)
    except Exception as e:
        log(f"Figure creation FAILED: {e}")
        traceback.print_exc()

    # Save combined results
    save_json(ALL_RESULTS, f"{RESULTS_DIR}/w1w2w3_resolution.json")

    # Print final summary
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)

    if w1:
        log("\nW1: Clinical Validation")
        if 'summary' in w1 and w1['summary']:
            s = w1['summary']
            log(f"  PathOmicDRP 4-modal mean clinical AUC: {s.get('PathOmicDRP_4modal_mean_auc', 'N/A'):.3f}")
            log(f"  ElasticNet mean clinical AUC: {s.get('ElasticNet_mean_auc', 'N/A'):.3f}")
            log(f"  Mean advantage: {s.get('mean_advantage_over_elasticnet', 0):+.3f}")
        for drug, d in w1.get('drugs', {}).items():
            m4 = d['methods'].get('PathOmicDRP_4modal', {})
            log(f"  {drug}: AUC={m4.get('auc','N/A')}, "
                f"p={m4.get('permutation_p','N/A')}, "
                f"CI={m4.get('bootstrap_ci_95','N/A')}")

    if w2:
        log("\nW2: Architecture Comparison (100 epochs)")
        log(f"  SelfAttnOnly PCC_drug: {w2['selfattn_100ep']['pcc_drug']:.4f}")
        log(f"  PathOmicDRP PCC_drug:  {w2['pathomic_100ep']['pcc_drug']:.4f}")
        if 'clinical_auc_summary' in w2:
            s = w2['clinical_auc_summary']
            log(f"  SelfAttnOnly mean clinical AUC: {s['SelfAttnOnly_mean_auc']:.3f}")
            log(f"  PathOmicDRP mean clinical AUC:  {s['PathOmicDRP_mean_auc']:.3f}")
            log(f"  Mean advantage: {s['mean_advantage']:+.3f}")

    if w3:
        log("\nW3: Histopathology Value")
        if 'summary' in w3:
            s = w3['summary']
            log(f"  Mean AUC improvement: {s['mean_auc_improvement']:+.3f}")
            log(f"  Cohen's d: {s['cohens_d']:.3f}")
            log(f"  Sign test p: {s['sign_test_p']:.4f}")
            log(f"  All drugs improved: {s['all_positive']}")

    log("\nAll results saved to: " + RESULTS_DIR)
    log("DONE")
