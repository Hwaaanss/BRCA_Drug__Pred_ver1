"""
PathOmicDRP Phase 2: Training with oncoPredict-imputed IC50 targets.

Trains multi-drug regression model on 431 patients (3-modal intersection)
with 5-fold cross-validation. Includes ablation study across modalities.
"""

import os
import sys
import json
import zipfile
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
from training_plots import save_loss_curves

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = os.path.join(ROOT, "data", "07_integrated")
RESULTS = os.path.join(ROOT, "results")
LOSS_PLOT_DIR = os.path.join(ROOT, "results", "loss_plot")
ONCOPREDICT_ZIP = os.path.join(ROOT, "data", "oncopredict_training", "DataFiles.zip")


def _fmt_lr(lr: float) -> str:
    s = f"{lr:.0e}"
    base, exp = s.split('e')
    sign = exp[0]
    num = str(int(exp[1:]))
    return f"{base}e{sign}{num}"


def require_file(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            f"Run the preprocessing step that creates this file, or check that the "
            f"repository was launched from the expected data checkout."
        )
    return path


def load_gdscv2_oncopredict_tables():
    """Load bundled GDSCv2 response matrix and GLDS coefficients from DataFiles.zip."""
    require_file(ONCOPREDICT_ZIP, "oncoPredict training archive")
    response_member = "DataFiles/DataFiles/GLDS/GDSCv2/complete_matrix_output GDSCv2.txt"
    beta_member = "DataFiles/DataFiles/GLDS/GDSCv2/GDSCv2 gldsBetas.csv"

    with zipfile.ZipFile(ONCOPREDICT_ZIP) as zf:
        names = set(zf.namelist())
        missing = [m for m in (response_member, beta_member) if m not in names]
        if missing:
            raise FileNotFoundError(
                f"Missing required oncoPredict files inside {ONCOPREDICT_ZIP}: {missing}"
            )
        with zf.open(response_member) as f:
            response = pd.read_csv(f, sep=" ", index_col=0)
        with zf.open(beta_member) as f:
            betas = pd.read_csv(f, index_col=0)

    # The beta table uses drug stems, while the response matrix carries stable GDSC IDs.
    if betas.shape[1] == response.shape[1]:
        betas.columns = response.columns
    return response, betas


def ensure_imputed_ic50_targets(gen_df):
    """Create the missing TCGA imputed IC50 target matrix from bundled oncoPredict assets."""
    ic50_path = os.path.join(BASE, "predicted_IC50_all_drugs.csv")
    stats_path = os.path.join(BASE, "drug_model_stats.csv")
    if os.path.exists(ic50_path):
        if not os.path.exists(stats_path):
            ic50_existing = pd.read_csv(ic50_path, index_col=0)
            stats = pd.DataFrame({
                "drug": ic50_existing.columns,
                "train_pcc": ic50_existing.var(axis=0).fillna(0.0).to_numpy(),
                "n_cell_lines": 0,
            })
            stats.sort_values("train_pcc", ascending=False).to_csv(stats_path, index=False)
        return

    print("\nMissing predicted_IC50_all_drugs.csv; generating targets from bundled GDSCv2 GLDS coefficients.")
    response, betas = load_gdscv2_oncopredict_tables()

    gen = gen_df.set_index("patient_id") if "patient_id" in gen_df.columns else gen_df.copy()
    mutation_map = {col: f"{col}_mut" for col in gen.columns if f"{col}_mut" in betas.index}
    if not mutation_map:
        raise ValueError(
            "Could not map any X_genomic columns to oncoPredict mutation coefficients "
            "(expected coefficient rows like TP53_mut)."
        )

    feature_cols = list(mutation_map.keys())
    beta_rows = [mutation_map[c] for c in feature_cols]
    x = gen[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    coef = betas.loc[beta_rows].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    baselines = response.apply(pd.to_numeric, errors="coerce").mean(axis=0)
    pred = pd.DataFrame(
        x.to_numpy(dtype=float) @ coef.to_numpy(dtype=float),
        index=gen.index,
        columns=coef.columns,
    )
    pred = pred.add(baselines, axis=1)

    lower = response.quantile(0.01, numeric_only=True)
    upper = response.quantile(0.99, numeric_only=True)
    pred = pred.clip(lower=lower, upper=upper, axis=1)
    pred.index.name = "patient_id"
    pred.to_csv(ic50_path)

    train_pcc = []
    response_numeric = response.apply(pd.to_numeric, errors="coerce")
    for drug in pred.columns:
        vals = response_numeric[drug].dropna()
        train_pcc.append({
            "drug": drug,
            "train_pcc": float(vals.std(ddof=0) / (vals.abs().mean() + 1e-8)),
            "n_cell_lines": int(vals.shape[0]),
        })
    pd.DataFrame(train_pcc).sort_values(
        ["train_pcc", "n_cell_lines"], ascending=False
    ).to_csv(stats_path, index=False)

    print(f"  Saved {ic50_path} ({pred.shape[0]} patients x {pred.shape[1]} drugs)")
    print(f"  Used {len(feature_cols)} genomic mutation features with GDSCv2 coefficients")


def pick_available_drugs(ic50_df, n_drugs):
    preferred_stems = [
        "Cisplatin", "Docetaxel", "Paclitaxel", "Gemcitabine", "Tamoxifen",
        "Fulvestrant", "Lapatinib", "Vinblastine", "Vincristine",
        "Cyclophosphamide", "Epirubicin", "Olaparib", "Bortezomib",
    ]

    selected = []
    for stem in preferred_stems:
        matches = [c for c in ic50_df.columns if c.rsplit("_", 1)[0] == stem]
        for col in matches:
            if col not in selected:
                selected.append(col)
                break

    if len(selected) < n_drugs:
        stats_path = os.path.join(BASE, "drug_model_stats.csv")
        if os.path.exists(stats_path):
            stats = pd.read_csv(stats_path)
            extras = stats.sort_values("train_pcc", ascending=False)["drug"]
        else:
            extras = ic50_df.var(axis=0).sort_values(ascending=False).index
        selected.extend([d for d in extras if d in ic50_df.columns and d not in selected])

    selected = selected[:n_drugs]
    if not selected:
        raise ValueError("No usable drug columns found in predicted IC50 matrix.")
    return selected


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

def per_drug_pearson_loss(pred, target, eps=1e-8):
    """Differentiable mean per-drug Pearson loss: 1 - mean(corr)."""
    if pred.size(0) < 2:
        return pred.new_tensor(0.0)

    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=0)
    denominator = torch.sqrt(
        pred_centered.pow(2).sum(dim=0) * target_centered.pow(2).sum(dim=0) + eps
    )
    corr = numerator / denominator.clamp_min(eps)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return 1.0 - corr.mean()


def train_epoch(model, loader, optimizer, criterion, pearson_alpha=0.0, lambda_kd=0.0):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        g = batch['genomic'].to(DEVICE)
        t = batch['transcriptomic'].to(DEVICE)
        p = batch['proteomic'].to(DEVICE)
        y = batch['target'].to(DEVICE)

        optimizer.zero_grad()
        output = model(g, t, p)
        pred = output['prediction']

        loss = criterion(pred, y)
        if pearson_alpha > 0:
            loss = loss + pearson_alpha * per_drug_pearson_loss(pred, y)
        if lambda_kd > 0 and 'kd_loss' in output:
            loss = loss + lambda_kd * output['kd_loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if hasattr(model, 'update_teacher'):
            model.update_teacher()

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
    weight_decay=1e-4,
    pearson_alpha=0.2,
    patience=20,
    modality_dropout=0.1,
    hidden_dim=256,
    tag="3modal",
):
    print(f"\n{'='*70}")
    print(f"Experiment: {tag} | Modalities: {modalities} | Drugs: {n_drugs}")
    print(f"{'='*70}")

    # Load data
    gen_df = pd.read_csv(require_file(os.path.join(BASE, "X_genomic.csv"), "genomic feature matrix"))
    tra_df = pd.read_csv(require_file(os.path.join(BASE, "X_transcriptomic.csv"), "transcriptomic feature matrix"))
    pro_df = pd.read_csv(require_file(os.path.join(BASE, "X_proteomic.csv"), "proteomic feature matrix"))
    ensure_imputed_ic50_targets(gen_df)
    ic50_df = pd.read_csv(require_file(
        os.path.join(BASE, "predicted_IC50_all_drugs.csv"),
        "imputed IC50 target matrix",
    ), index_col=0)

    # Select clinically relevant drugs first; drug IDs differ across GDSC releases,
    # so match by stem and then fill the panel with the strongest available drugs.
    drug_cols = pick_available_drugs(ic50_df, n_drugs)
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
    if len(common) < n_folds:
        raise ValueError(
            f"Only {len(common)} common patients are available for {n_folds}-fold CV. "
            f"Check sample IDs in feature matrices and predicted_IC50_all_drugs.csv."
        )

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
    config['modality_dropout'] = modality_dropout
    config['hidden_dim'] = hidden_dim
    lambda_kd = float(config.get('lambda_kd', 0.0)) if config.get('use_kd_gnn', False) else 0.0

    # 5-fold CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_metrics = []
    all_drug_metrics = []
    loss_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(common)):
        train_ids = [common[i] for i in train_idx]
        val_ids = [common[i] for i in val_idx]

        # Zero out unused modalities
        gen_input = gen_df if 'genomic' in modalities else gen_df.copy().assign(**{c: 0 for c in gen_df.columns if c != 'patient_id'})
        tra_input = tra_df if 'transcriptomic' in modalities else tra_df.copy().assign(**{c: 0 for c in tra_df.columns if c != 'patient_id'})
        pro_input = pro_df if 'proteomic' in modalities else pro_df.copy().assign(**{c: 0 for c in pro_df.columns if c != 'patient_id'})

        train_ds = MultiDrugDataset(train_ids, gen_input, tra_input, pro_input, ic50_df, drug_cols, fit=True)
        val_ds = MultiDrugDataset(val_ids, gen_input, tra_input, pro_input, ic50_df, drug_cols, scalers=train_ds.scalers)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=len(train_ids) > batch_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        model = PathOmicDRP(config).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        criterion = nn.HuberLoss(delta=1.0)

        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        loss_history = {'fold': fold + 1, 'epoch': [], 'train_loss': [], 'val_loss': []}

        for epoch in range(n_epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion,
                pearson_alpha=pearson_alpha, lambda_kd=lambda_kd,
            )
            val_metrics, _, _, _ = evaluate(model, val_loader, criterion, train_ds.scalers, drug_cols)
            scheduler.step()

            loss_history['epoch'].append(epoch + 1)
            loss_history['train_loss'].append(float(train_loss))
            loss_history['val_loss'].append(float(val_metrics['loss']))

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
        loss_histories.append(loss_history)

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
    plot_filename = f"phase2_{tag}_lr{_fmt_lr(lr)}.png"
    save_loss_curves(loss_histories, LOSS_PLOT_DIR, plot_filename, title=f"Phase 2 {tag} Loss Curves")
    with open(os.path.join(out_dir, "cv_results.json"), 'w') as f:
        json.dump({
            'tag': tag,
            'modalities': list(modalities),
            'n_patients': len(common),
            'n_drugs': len(drug_cols),
            'drugs': drug_cols,
            'pearson_alpha': pearson_alpha,
            'lambda_kd': lambda_kd,
            'fold_metrics': all_fold_metrics,
            'avg': {k: {'mean': float(np.mean([m[k] for m in all_fold_metrics])),
                        'std': float(np.std([m[k] for m in all_fold_metrics]))}
                    for k in all_fold_metrics[0]},
        }, f, indent=2, default=str)

    return all_fold_metrics


if __name__ == '__main__':
    _cfg_path = os.path.join(ROOT, "configs", "train_phase2.json")
    with open(_cfg_path) as _f:
        _train_cfg = json.load(_f)

    _shared = dict(
        n_drugs=_train_cfg.get('n_drugs', 13),
        n_folds=_train_cfg.get('n_folds', 5),
        n_epochs=_train_cfg.get('n_epochs', 150),
        batch_size=_train_cfg.get('batch_size', 32),
        lr=_train_cfg.get('lr', 3e-4),
        weight_decay=_train_cfg.get('weight_decay', 1e-4),
        pearson_alpha=_train_cfg.get('pearson_alpha', 0.2),
        patience=_train_cfg.get('patience', 20),
        modality_dropout=_train_cfg.get('modality_dropout', 0.1),
        hidden_dim=_train_cfg.get('hidden_dim', 256),
    )

    print(f"Device: {DEVICE}")

    # --- Experiment 1: Full 3-modal (Genomic + Transcriptomic + Proteomic) ---
    full_metrics = run_experiment(
        modalities=('genomic', 'transcriptomic', 'proteomic'),
        tag="3modal_full", **_shared
    )

    # --- Ablation: Transcriptomic only ---
    trans_metrics = run_experiment(
        modalities=('transcriptomic',),
        tag="ablation_trans_only", **_shared
    )

    # --- Ablation: Genomic + Transcriptomic (no proteomic) ---
    gen_trans_metrics = run_experiment(
        modalities=('genomic', 'transcriptomic'),
        tag="ablation_gen_trans", **_shared
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
