"""
PathOmicDRP Phase 3: 4-Modal Training (Genomic + Transcriptomic + Proteomic + Histology).

Extends Phase 2 by adding UNI-extracted H&E features via ABMIL.
5-fold CV with ablation: 3-modal vs 4-modal comparison.
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
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from model import PathOmicDRP, get_default_config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

BASE = DATA_ROOT / "07_integrated"
HISTO_DIR = DATA_ROOT / "05_morphology" / "features"
RESULTS = PROJECT_ROOT / "results"


class MultiDrugDataset4Modal(Dataset):
    """Dataset with 4 modalities: genomic, transcriptomic, proteomic, histology."""

    def __init__(self, patient_ids, genomic_df, trans_df, prot_df, ic50_df, drug_cols,
                 histo_dir=None, scalers=None, fit=False):
        self.pids = list(patient_ids)
        self.drug_cols = drug_cols
        self.histo_dir = histo_dir

        gen = genomic_df.set_index('patient_id') if 'patient_id' in genomic_df.columns else genomic_df
        tra = trans_df.set_index('patient_id') if 'patient_id' in trans_df.columns else trans_df
        pro = prot_df.set_index('patient_id') if 'patient_id' in prot_df.columns else prot_df

        self.gen_cols = [c for c in gen.columns if c != 'patient_id']
        self.tra_cols = [c for c in tra.columns if c != 'patient_id']
        self.pro_cols = [c for c in pro.columns if c != 'patient_id']

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

        # Pre-load histology features
        self.histo_features = {}
        if histo_dir:
            for pid in self.pids:
                pt_path = os.path.join(histo_dir, f"{pid}.pt")
                if os.path.exists(pt_path):
                    self.histo_features[pid] = torch.load(pt_path, map_location='cpu', weights_only=True)
            print(f"  Loaded histology features for {len(self.histo_features)}/{len(self.pids)} patients")

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
        pid = self.pids[idx]
        sample = {
            'genomic': torch.tensor(self.gen_data[idx], dtype=torch.float32),
            'transcriptomic': torch.tensor(self.tra_data[idx], dtype=torch.float32),
            'proteomic': torch.tensor(self.pro_data[idx], dtype=torch.float32),
            'target': torch.tensor(self.ic50_data[idx], dtype=torch.float32),
        }
        if pid in self.histo_features:
            sample['histology'] = self.histo_features[pid]
        return sample


def collate_4modal(batch):
    """Custom collate handling variable-size histology patches."""
    result = {
        'genomic': torch.stack([s['genomic'] for s in batch]),
        'transcriptomic': torch.stack([s['transcriptomic'] for s in batch]),
        'proteomic': torch.stack([s['proteomic'] for s in batch]),
        'target': torch.stack([s['target'] for s in batch]),
    }

    has_histo = [s for s in batch if 'histology' in s]
    if has_histo:
        max_patches = max(s['histology'].shape[0] for s in has_histo)
        feat_dim = has_histo[0]['histology'].shape[1]
        histo_tensor = torch.zeros(len(batch), max_patches, feat_dim)
        histo_mask = torch.zeros(len(batch), max_patches, dtype=torch.bool)
        for i, s in enumerate(batch):
            if 'histology' in s:
                n = s['histology'].shape[0]
                histo_tensor[i, :n] = s['histology']
                histo_mask[i, :n] = True
        result['histology'] = histo_tensor
        result['histo_mask'] = histo_mask

    return result


def train_epoch(model, loader, optimizer, criterion, use_histo=False):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        g = batch['genomic'].to(DEVICE)
        t = batch['transcriptomic'].to(DEVICE)
        p = batch['proteomic'].to(DEVICE)
        y = batch['target'].to(DEVICE)

        kwargs = {}
        if use_histo and 'histology' in batch:
            kwargs['histology'] = batch['histology'].to(DEVICE)
            kwargs['histo_mask'] = batch['histo_mask'].to(DEVICE)

        optimizer.zero_grad()
        out = model(g, t, p, **kwargs)['prediction']
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, scalers, drug_cols, use_histo=False):
    model.eval()
    all_pred, all_true = [], []
    total_loss, n = 0, 0

    for batch in loader:
        g = batch['genomic'].to(DEVICE)
        t = batch['transcriptomic'].to(DEVICE)
        p = batch['proteomic'].to(DEVICE)
        y = batch['target'].to(DEVICE)

        kwargs = {}
        if use_histo and 'histology' in batch:
            kwargs['histology'] = batch['histology'].to(DEVICE)
            kwargs['histo_mask'] = batch['histo_mask'].to(DEVICE)

        out = model(g, t, p, **kwargs)['prediction']
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        n += len(y)

        all_pred.append(out.cpu().numpy())
        all_true.append(y.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    if scalers and 'ic50' in scalers:
        all_pred_orig = scalers['ic50'].inverse_transform(all_pred)
        all_true_orig = scalers['ic50'].inverse_transform(all_true)
    else:
        all_pred_orig = all_pred
        all_true_orig = all_true

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
            'rmse': float(np.sqrt(mean_squared_error(t_vals, p_vals))),
            'r2': float(r2_score(t_vals, p_vals)),
        }

    flat_pred = all_pred_orig.flatten()
    flat_true = all_true_orig.flatten()
    pcc_global, _ = pearsonr(flat_true, flat_pred)
    scc_global, _ = spearmanr(flat_true, flat_pred)

    metrics = {
        'loss': total_loss / n,
        'pcc_global': float(pcc_global),
        'scc_global': float(scc_global),
        'rmse_global': float(np.sqrt(mean_squared_error(flat_true, flat_pred))),
        'r2_global': float(r2_score(flat_true, flat_pred)),
        'pcc_per_drug_mean': float(np.mean([m['pcc'] for m in drug_metrics.values()])),
        'pcc_per_drug_median': float(np.median([m['pcc'] for m in drug_metrics.values()])),
    }
    return metrics, drug_metrics, all_pred_orig, all_true_orig


def run_experiment(
    use_histology=False,
    n_drugs=13,
    n_folds=5,
    n_epochs=150,
    batch_size=16,
    lr=3e-4,
    tag="4modal",
):
    print(f"\n{'='*70}")
    print(f"Experiment: {tag} | Histology: {use_histology} | Drugs: {n_drugs}")
    print(f"{'='*70}")

    gen_df = pd.read_csv(os.path.join(BASE, "X_genomic.csv"))
    tra_df = pd.read_csv(os.path.join(BASE, "X_transcriptomic.csv"))
    pro_df = pd.read_csv(os.path.join(BASE, "X_proteomic.csv"))
    ic50_df = pd.read_csv(os.path.join(BASE, "predicted_IC50_all_drugs.csv"), index_col=0)

    # Same drug selection as Phase 2
    clinical_drugs = [
        'Cisplatin_1005', 'Docetaxel_1007', 'Paclitaxel_1080',
        'Gemcitabine_1190', 'Tamoxifen_1199', 'Fulvestrant_1012',
        'Lapatinib_1558', 'Vinblastine_1004', 'Vincristine_2048',
        'Cyclophosphamide_1014', 'Epirubicin_2066',
    ]
    drug_cols = [d for d in clinical_drugs if d in ic50_df.columns]
    if len(drug_cols) < n_drugs:
        stats = pd.read_csv(os.path.join(BASE, "drug_model_stats.csv"))
        extra = [d for d in stats.sort_values('train_pcc', ascending=False)['drug']
                 if d in ic50_df.columns and d not in drug_cols]
        drug_cols = drug_cols + extra[:n_drugs - len(drug_cols)]
    drug_cols = drug_cols[:n_drugs]
    print(f"Selected {len(drug_cols)} drugs: {[d.rsplit('_',1)[0] for d in drug_cols]}")

    # Filter to patients with histology features if using histology
    gen_ids = set(gen_df['patient_id'])
    tra_ids = set(tra_df['patient_id'])
    pro_ids = set(pro_df['patient_id'])
    ic50_ids = set(ic50_df.index)
    common = sorted(gen_ids & tra_ids & pro_ids & ic50_ids)

    if use_histology:
        histo_ids = {f.replace('.pt', '') for f in os.listdir(HISTO_DIR) if f.endswith('.pt')}
        common_with_histo = sorted(set(common) & histo_ids)
        print(f"Patients with all 4 modalities: {len(common_with_histo)}/{len(common)}")
        common = common_with_histo

    print(f"Training patients: {len(common)}")

    gen_dim = len([c for c in gen_df.columns if c != 'patient_id'])
    tra_dim = len([c for c in tra_df.columns if c != 'patient_id'])
    pro_dim = len([c for c in pro_df.columns if c != 'patient_id'])

    config = get_default_config(
        genomic_dim=gen_dim,
        n_pathways=tra_dim,
        proteomic_dim=pro_dim,
        n_drugs=len(drug_cols),
        use_histology=use_histology,
    )
    config['task'] = 'regression'
    config['modality_dropout'] = 0.1
    config['hidden_dim'] = 256

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_metrics = []
    all_drug_metrics = []
    best_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(common)):
        train_ids = [common[i] for i in train_idx]
        val_ids = [common[i] for i in val_idx]

        histo_dir = HISTO_DIR if use_histology else None
        train_ds = MultiDrugDataset4Modal(
            train_ids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
            histo_dir=histo_dir, fit=True,
        )
        val_ds = MultiDrugDataset4Modal(
            val_ids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
            histo_dir=histo_dir, scalers=train_ds.scalers,
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=2, collate_fn=collate_4modal,
            drop_last=len(train_ids) > batch_size,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=2, collate_fn=collate_4modal,
        )

        model = PathOmicDRP(config).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold == 0:
            print(f"Model params: {n_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        criterion = nn.HuberLoss(delta=1.0)

        best_loss = float('inf')
        best_state = None
        patience = 20
        patience_counter = 0

        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, use_histo=use_histology)
            val_metrics, _, _, _ = evaluate(model, val_loader, criterion, train_ds.scalers, drug_cols, use_histo=use_histology)
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

        model.load_state_dict(best_state)
        model.to(DEVICE)
        final_metrics, drug_met, _, _ = evaluate(model, val_loader, criterion, train_ds.scalers, drug_cols, use_histo=use_histology)
        all_fold_metrics.append(final_metrics)
        all_drug_metrics.append(drug_met)
        best_models.append(best_state)

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

    print(f"\n  Per-drug PCC (mean across folds):")
    drug_names_clean = [d.rsplit('_', 1)[0] for d in drug_cols]
    for i, (drug, name) in enumerate(zip(drug_cols, drug_names_clean)):
        vals = [fold_met[drug]['pcc'] for fold_met in all_drug_metrics]
        print(f"    {name:25s}: PCC={np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Save results & best model
    out_dir = os.path.join(RESULTS, f"phase3_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "cv_results.json"), 'w') as f:
        json.dump({
            'tag': tag,
            'use_histology': use_histology,
            'n_patients': len(common),
            'n_drugs': len(drug_cols),
            'drugs': drug_cols,
            'config': config,
            'fold_metrics': all_fold_metrics,
            'avg': {k: {'mean': float(np.mean([m[k] for m in all_fold_metrics])),
                        'std': float(np.std([m[k] for m in all_fold_metrics]))}
                    for k in all_fold_metrics[0]},
            'drug_metrics_per_fold': all_drug_metrics,
        }, f, indent=2, default=str)

    # Save best fold model
    best_fold_idx = np.argmin([m['loss'] for m in all_fold_metrics])
    torch.save(best_models[best_fold_idx], os.path.join(out_dir, "best_model.pt"))
    print(f"\nSaved best model (fold {best_fold_idx+1}) to {out_dir}/best_model.pt")

    return all_fold_metrics


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Histology features dir: {HISTO_DIR}")
    n_histo = len([f for f in os.listdir(HISTO_DIR) if f.endswith('.pt')])
    print(f"Available histology features: {n_histo}")

    # --- Experiment 1: 3-modal baseline (same as Phase 2, for fair comparison) ---
    baseline_metrics = run_experiment(
        use_histology=False,
        n_drugs=13, batch_size=32, tag="3modal_baseline",
    )

    # --- Experiment 2: 4-modal (with H&E histology) ---
    full_metrics = run_experiment(
        use_histology=True,
        n_drugs=13, batch_size=16, tag="4modal_full",
    )

    # --- Summary ---
    print(f"\n{'='*70}")
    print("PHASE 3: 3-MODAL vs 4-MODAL COMPARISON")
    print(f"{'='*70}")
    for name, metrics in [
        ("3-modal (Gen+Trans+Prot)", baseline_metrics),
        ("4-modal (+Histology/UNI)", full_metrics),
    ]:
        pcc = np.mean([m['pcc_global'] for m in metrics])
        r2 = np.mean([m['r2_global'] for m in metrics])
        pcc_drug = np.mean([m['pcc_per_drug_mean'] for m in metrics])
        print(f"  {name:35s} | PCC_global={pcc:.4f} | R²={r2:.4f} | PCC_drug={pcc_drug:.4f}")

    pcc_3 = np.mean([m['pcc_per_drug_mean'] for m in baseline_metrics])
    pcc_4 = np.mean([m['pcc_per_drug_mean'] for m in full_metrics])
    improvement = (pcc_4 - pcc_3) / abs(pcc_3) * 100 if pcc_3 != 0 else 0
    print(f"\n  PCC_drug improvement: {improvement:+.1f}%")
