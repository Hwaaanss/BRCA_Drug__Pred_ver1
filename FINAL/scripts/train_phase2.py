"""
PathOmicDRP Phase 2: Training with oncoPredict-imputed IC50 targets.

Trains multi-drug regression model on 431 patients (3-modal intersection)
with 5-fold cross-validation. Includes ablation study across modalities.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from model import PathOmicDRP, get_default_config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[2])).resolve()
FINAL_ROOT = PROJECT_ROOT / "FINAL"
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
BASE = DATA_ROOT / "07_integrated"
RESULTS = FINAL_ROOT / "results"


# ---------------------------------------------------------------------------
# Dataset for multi-drug IC50 prediction
# ---------------------------------------------------------------------------

class MultiDrugDataset(Dataset):
    """Dataset: each sample = (patient features, drug IC50 vector)."""

    def __init__(self, patient_ids, genomic_df, trans_df, prot_df, ic50_df, drug_cols, scalers=None, fit=False):
        self.pids = list(patient_ids)
        self.drug_cols = drug_cols

        # Index by patient_id
        gen = genomic_df.set_index('patient_id') if 'patient_id' in genomic_df.columns else genomic_df
        tra = trans_df.set_index('patient_id') if 'patient_id' in trans_df.columns else trans_df
        pro = prot_df.set_index('patient_id') if 'patient_id' in prot_df.columns else prot_df

        self.gen_cols = [c for c in gen.columns if c != 'patient_id']
        self.tra_cols = [c for c in tra.columns if c != 'patient_id']
        self.pro_cols = [c for c in pro.columns if c != 'patient_id']

        # Build numpy arrays (aligned to self.pids, fill missing with 0)
        def safe_loc(df, ids, cols):
            avail = df.index.intersection(ids)
            result = np.zeros((len(ids), len(cols)), dtype=np.float32)
            if len(avail) > 0:
                idx_map = {pid: i for i, pid in enumerate(ids)}
                for pid in avail:
                    result[idx_map[pid]] = df.loc[pid, cols].values.astype(np.float32)
            return result

        self.gen_data = safe_loc(gen, self.pids, self.gen_cols)
        tra_raw = safe_loc(tra, self.pids, self.tra_cols)
        self.tra_data = np.log1p(np.maximum(tra_raw, 0))
        self.pro_data = safe_loc(pro, self.pids, self.pro_cols)
        self.ic50_data = ic50_df.loc[self.pids, drug_cols].values.astype(np.float32)

        # Scaling
        if fit:
            self.scalers = {
                'gen': StandardScaler().fit(self.gen_data),
                'tra': StandardScaler().fit(self.tra_data),
                'pro': StandardScaler().fit(self.pro_data),
                'ic50': StandardScaler().fit(self.ic50_data),
            }
        elif scalers:
            self.scalers = scalers
        else:
            self.scalers = None

        if self.scalers:
            self.gen_data = self.scalers['gen'].transform(self.gen_data)
            self.tra_data = self.scalers['tra'].transform(self.tra_data)
            self.pro_data = self.scalers['pro'].transform(self.pro_data)
            self.ic50_data = self.scalers['ic50'].transform(self.ic50_data)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        return {
            'genomic': torch.tensor(self.gen_data[idx], dtype=torch.float32),
            'transcriptomic': torch.tensor(self.tra_data[idx], dtype=torch.float32),
            'proteomic': torch.tensor(self.pro_data[idx], dtype=torch.float32),
            'target': torch.tensor(self.ic50_data[idx], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        g = batch['genomic'].to(DEVICE)
        t = batch['transcriptomic'].to(DEVICE)
        p = batch['proteomic'].to(DEVICE)
        y = batch['target'].to(DEVICE)

        optimizer.zero_grad()
        out = model(g, t, p)['prediction']
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, scalers, drug_cols):
    model.eval()
    all_pred, all_true = [], []
    total_loss, n = 0, 0

    for batch in loader:
        g = batch['genomic'].to(DEVICE)
        t = batch['transcriptomic'].to(DEVICE)
        p = batch['proteomic'].to(DEVICE)
        y = batch['target'].to(DEVICE)

        out = model(g, t, p)['prediction']
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        n += len(y)

        all_pred.append(out.cpu().numpy())
        all_true.append(y.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    # Inverse transform for interpretable metrics
    if scalers and 'ic50' in scalers:
        all_pred_orig = scalers['ic50'].inverse_transform(all_pred)
        all_true_orig = scalers['ic50'].inverse_transform(all_true)
    else:
        all_pred_orig = all_pred
        all_true_orig = all_true

    # Per-drug metrics
    drug_metrics = {}
    for i, drug in enumerate(drug_cols):
        p_vals = all_pred_orig[:, i]
        t_vals = all_true_orig[:, i]
        try:
            pcc, _ = pearsonr(t_vals, p_vals)
            scc, _ = spearmanr(t_vals, p_vals)
        except:
            pcc, scc = 0, 0
        drug_metrics[drug] = {
            'pcc': pcc, 'scc': scc,
            'rmse': np.sqrt(mean_squared_error(t_vals, p_vals)),
            'r2': r2_score(t_vals, p_vals),
        }

    # Global metrics (flatten all drugs)
    flat_pred = all_pred_orig.flatten()
    flat_true = all_true_orig.flatten()
    pcc_global, _ = pearsonr(flat_true, flat_pred)
    scc_global, _ = spearmanr(flat_true, flat_pred)

    metrics = {
        'loss': total_loss / n,
        'pcc_global': pcc_global,
        'scc_global': scc_global,
        'rmse_global': np.sqrt(mean_squared_error(flat_true, flat_pred)),
        'r2_global': r2_score(flat_true, flat_pred),
        'pcc_per_drug_mean': np.mean([m['pcc'] for m in drug_metrics.values()]),
        'pcc_per_drug_median': np.median([m['pcc'] for m in drug_metrics.values()]),
    }
    return metrics, drug_metrics, all_pred_orig, all_true_orig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(
    modalities=('genomic', 'transcriptomic', 'proteomic'),
    n_drugs=13,
    n_folds=5,
    n_epochs=150,
    batch_size=32,
    lr=3e-4,
    tag="3modal",
):
    print(f"\n{'='*70}")
    print(f"Experiment: {tag} | Modalities: {modalities} | Drugs: {n_drugs}")
    print(f"{'='*70}")

    # Load data
    gen_df = pd.read_csv(os.path.join(BASE, "X_genomic.csv"))
    tra_df = pd.read_csv(os.path.join(BASE, "X_transcriptomic.csv"))
    pro_df = pd.read_csv(os.path.join(BASE, "X_proteomic.csv"))
    ic50_df = pd.read_csv(os.path.join(BASE, "predicted_IC50_all_drugs.csv"), index_col=0)

    # Select drugs with TCGA clinical overlap (for later validation)
    clinical_drugs = [
        'Cisplatin_1005', 'Docetaxel_1007', 'Paclitaxel_1080',
        'Gemcitabine_1190', 'Tamoxifen_1199', 'Fulvestrant_1012',
        'Lapatinib_1558', 'Vinblastine_1004', 'Vincristine_2048',
        'Cyclophosphamide_1014', 'Epirubicin_2066',
    ]
    # Filter to drugs available in our predictions
    drug_cols = [d for d in clinical_drugs if d in ic50_df.columns]
    if len(drug_cols) < n_drugs:
        # Add more drugs by training PCC
        stats = pd.read_csv(os.path.join(BASE, "drug_model_stats.csv"))
        extra = [d for d in stats.sort_values('train_pcc', ascending=False)['drug']
                 if d in ic50_df.columns and d not in drug_cols]
        drug_cols = drug_cols + extra[:n_drugs - len(drug_cols)]
    drug_cols = drug_cols[:n_drugs]
    print(f"Selected {len(drug_cols)} drugs: {[d.rsplit('_',1)[0] for d in drug_cols]}")

    # Common patients (all 3 modalities)
    gen_ids = set(gen_df['patient_id'])
    tra_ids = set(tra_df['patient_id'])
    pro_ids = set(pro_df['patient_id'])
    ic50_ids = set(ic50_df.index)

    if 'proteomic' in modalities:
        common = sorted(gen_ids & tra_ids & pro_ids & ic50_ids)
    else:
        common = sorted(gen_ids & tra_ids & ic50_ids)
    print(f"Patients: {len(common)}")

    # Determine input dims (always use full dims, ablation zeroes data not architecture)
    gen_dim = len([c for c in gen_df.columns if c != 'patient_id'])
    tra_dim = len([c for c in tra_df.columns if c != 'patient_id'])
    pro_dim = len([c for c in pro_df.columns if c != 'patient_id'])

    config = get_default_config(
        genomic_dim=gen_dim,
        n_pathways=tra_dim,
        proteomic_dim=pro_dim,
        n_drugs=len(drug_cols),
        use_histology=False,
    )
    config['task'] = 'regression'
    config['modality_dropout'] = 0.1
    config['hidden_dim'] = 256

    # 5-fold CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_metrics = []
    all_drug_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(common)):
        train_ids = [common[i] for i in train_idx]
        val_ids = [common[i] for i in val_idx]

        # Zero out unused modalities
        gen_input = gen_df if 'genomic' in modalities else gen_df.copy().assign(**{c: 0 for c in gen_df.columns if c != 'patient_id'})
        tra_input = tra_df if 'transcriptomic' in modalities else tra_df.copy().assign(**{c: 0 for c in tra_df.columns if c != 'patient_id'})
        pro_input = pro_df if 'proteomic' in modalities else pro_df.copy().assign(**{c: 0 for c in pro_df.columns if c != 'patient_id'})

        train_ds = MultiDrugDataset(train_ids, gen_input, tra_input, pro_input, ic50_df, drug_cols, fit=True)
        val_ds = MultiDrugDataset(val_ids, gen_input, tra_input, pro_input, ic50_df, drug_cols, scalers=train_ds.scalers)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=len(train_ids) > batch_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = PathOmicDRP(config).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        criterion = nn.HuberLoss(delta=1.0)

        best_loss = float('inf')
        best_state = None
        patience = 20
        patience_counter = 0

        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_metrics, _, _, _ = evaluate(model, val_loader, criterion, train_ds.scalers, drug_cols)
            scheduler.step()

            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 25 == 0:
                print(f"  Fold {fold+1} Ep {epoch+1:3d} | train={train_loss:.4f} | "
                      f"val_loss={val_metrics['loss']:.4f} | PCC_global={val_metrics['pcc_global']:.4f} | "
                      f"PCC_drug_mean={val_metrics['pcc_per_drug_mean']:.4f}")

            if patience_counter >= patience:
                print(f"  Fold {fold+1} early stop at epoch {epoch+1}")
                break

        # Final evaluation with best model
        model.load_state_dict(best_state)
        model.to(DEVICE)
        final_metrics, drug_met, _, _ = evaluate(model, val_loader, criterion, train_ds.scalers, drug_cols)
        all_fold_metrics.append(final_metrics)
        all_drug_metrics.append(drug_met)

        print(f"  Fold {fold+1} FINAL | PCC_global={final_metrics['pcc_global']:.4f} | "
              f"PCC_drug_mean={final_metrics['pcc_per_drug_mean']:.4f} | "
              f"R²={final_metrics['r2_global']:.4f} | RMSE={final_metrics['rmse_global']:.4f}")

    # Aggregate
    print(f"\n{'='*70}")
    print(f"CV RESULTS: {tag}")
    print(f"{'='*70}")
    for key in ['pcc_global', 'scc_global', 'r2_global', 'rmse_global', 'pcc_per_drug_mean', 'pcc_per_drug_median']:
        vals = [m[key] for m in all_fold_metrics]
        print(f"  {key:25s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Per-drug average across folds
    print(f"\n  Per-drug PCC (mean across folds):")
    drug_names_clean = [d.rsplit('_', 1)[0] for d in drug_cols]
    for i, (drug, name) in enumerate(zip(drug_cols, drug_names_clean)):
        vals = [fold_met[drug]['pcc'] for fold_met in all_drug_metrics]
        print(f"    {name:25s}: PCC={np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Save
    out_dir = os.path.join(RESULTS, f"phase2_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cv_results.json"), 'w') as f:
        json.dump({
            'tag': tag,
            'modalities': list(modalities),
            'n_patients': len(common),
            'n_drugs': len(drug_cols),
            'drugs': drug_cols,
            'fold_metrics': all_fold_metrics,
            'avg': {k: {'mean': float(np.mean([m[k] for m in all_fold_metrics])),
                        'std': float(np.std([m[k] for m in all_fold_metrics]))}
                    for k in all_fold_metrics[0]},
        }, f, indent=2, default=str)

    return all_fold_metrics


if __name__ == '__main__':
    print(f"Device: {DEVICE}")

    # --- Experiment 1: Full 3-modal (Genomic + Transcriptomic + Proteomic) ---
    full_metrics = run_experiment(
        modalities=('genomic', 'transcriptomic', 'proteomic'),
        n_drugs=13, tag="3modal_full"
    )

    # --- Ablation: Transcriptomic only ---
    trans_metrics = run_experiment(
        modalities=('transcriptomic',),
        n_drugs=13, tag="ablation_trans_only"
    )

    # --- Ablation: Genomic + Transcriptomic (no proteomic) ---
    gen_trans_metrics = run_experiment(
        modalities=('genomic', 'transcriptomic'),
        n_drugs=13, tag="ablation_gen_trans"
    )

    # --- Summary ---
    print(f"\n{'='*70}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    for name, metrics in [
        ("Trans only", trans_metrics),
        ("Gen + Trans", gen_trans_metrics),
        ("Gen + Trans + Prot (Full)", full_metrics),
    ]:
        pcc = np.mean([m['pcc_global'] for m in metrics])
        r2 = np.mean([m['r2_global'] for m in metrics])
        pcc_drug = np.mean([m['pcc_per_drug_mean'] for m in metrics])
        print(f"  {name:30s} | PCC_global={pcc:.4f} | R²={r2:.4f} | PCC_drug_mean={pcc_drug:.4f}")
